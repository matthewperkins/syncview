"""Video panel: a background decoder (latest request wins, with a frame cache) and a widget to show frames."""
import threading
import traceback
from collections import OrderedDict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.video import keyframe_interval


class VideoDecoder(QtCore.QObject):
    """Decodes frames by index on the GPU in its own thread.

    Strategy per request (k):
      * cached                      -> return it
      * a little ahead of the decoder position (cheaper than a seek) -> decode forward, cache on the way
      * otherwise                   -> seek to the keyframe at/before k and decode up to k, caching that run
                                       (so stepping backwards inside the GOP is instant)
    """
    frame_ready = QtCore.Signal(int, object)     # frame index, QImage already scaled to the view
    opened = QtCore.Signal(int, int, int)        # n_frames, width, height
    failed = QtCore.Signal(str)

    def __init__(self, path, cache_frames=250):
        super().__init__()
        self.path = str(path)
        self.gop = keyframe_interval(path)
        self.cache = OrderedDict()
        self.cache_frames = cache_frames
        self._cv = threading.Condition()
        self._req = None
        self._stop = False
        self._next = None              # index the decoder will return next without seeking
        self.target = None             # (w, h) of the view; frames are scaled here, off the GUI thread
        threading.Thread(target=self._loop, daemon=True, name="video-decoder").start()

    def request(self, k):
        with self._cv:
            self._req = int(k)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _put(self, k, frame):
        self.cache[k] = frame
        self.cache.move_to_end(k)
        while len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)

    def _loop(self):
        import PyNvVideoCodec as nvc     # decoder (and its CUDA context) lives in this thread
        try:
            dec = nvc.SimpleDecoder(self.path, use_device_memory=False, output_color_type=nvc.OutputColorType.RGB)
            md = dec.get_stream_metadata()
            n = len(dec)
            self.opened.emit(n, md.width, md.height)
        except Exception as e:
            self.failed.emit(f"could not open video: {e}")
            return
        while True:
            with self._cv:
                while self._req is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                k, self._req = self._req, None
            if not 0 <= k < n:
                continue
            try:
                if k not in self.cache:
                    gop = self.gop or 1
                    ahead = k - self._next if self._next is not None else -1
                    if 0 <= ahead <= max(k % gop, 4):            # forward decode beats seek+decode
                        start = self._next
                    else:
                        start = k - k % gop if self.gop else k
                        dec.seek_to_index(start)
                    frames = dec.get_batch_frames(k - start + 1)
                    for j, f in enumerate(frames):
                        self._put(start + j, np.array(np.from_dlpack(f)))   # copy: decoder reuses buffers
                    self._next = start + len(frames)
                else:
                    self.cache.move_to_end(k)
                if k in self.cache:
                    self.frame_ready.emit(k, self._scaled(self.cache[k]))
            except Exception:
                traceback.print_exc()
                self._next = None


    def _scaled(self, arr):
        h, w, _ = arr.shape
        img = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        tw, th = self.target or (w, h)
        if (tw, th) != (w, h) and tw > 0 and th > 0:
            return img.scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        return img.copy()                      # own the pixels (arr may leave the cache)


class VideoView(QtWidgets.QWidget):
    """Draws the current frame scaled to fit (aspect preserved), with a small caption."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self._img = None
        self.caption = ""
        self.message = "no video loaded"
        self.decoder = None            # gets told our size so it can pre-scale frames

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.decoder is not None:
            self.decoder.target = (self.width(), self.height())

    def set_frame(self, img, caption=""):
        self._img = img
        self.caption, self.message = caption, ""
        self.update()

    def set_message(self, text):
        self._img = None
        self.message = text
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#000000"))
        if self._img is not None:
            iw, ih = self._img.width(), self._img.height()
            s = min(self.width() / iw, self.height() / ih)
            w, h = iw * s, ih * s
            target = QtCore.QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
            if (iw, ih) == (round(w), round(h)):        # pre-scaled by the decoder: plain blit
                p.drawImage(target.topLeft(), self._img)
            else:                                       # (window just resized) scale here this once
                p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
                p.drawImage(target, self._img)
            if self.caption:
                p.setPen(QtGui.QColor("#e8e8e4"))
                p.setFont(QtGui.QFont("monospace", 10))
                p.drawText(target.adjusted(8, 0, -8, -6), QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, self.caption)
        else:
            p.setPen(QtGui.QColor("#888888"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.message)
        p.end()
