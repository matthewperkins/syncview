"""Background workers: whole-session cache builds and on-demand window processing.

Both follow "latest request wins": a newer request supersedes whatever is queued, so the UI never waits
on stale work."""
import threading
import traceback

import numpy as np
from PySide6 import QtCore

from ..core.filters import pad_samples, process_window
from ..data.cache import Cancelled, PyramidTrace


class CacheBuilder(QtCore.QObject):
    progress = QtCore.Signal(float)      # 0..1 for the build in progress
    finished = QtCore.Signal()           # a build completed (or nothing was missing)

    def __init__(self, cache):
        super().__init__()
        self.cache = cache
        self._cv = threading.Condition()
        self._want = None
        self._wanted_keys = set()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cache-builder")
        self._thread.start()

    def stop(self, timeout=10.0):
        """Cancel any build (it stops at the next chunk) and wait for the thread to exit."""
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout)

    def request(self, specs):
        with self._cv:
            self._want = [dict(s) for s in specs]
            self._wanted_keys = {self.cache.key(s) for s in specs}
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while self._want is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                specs, self._want = self._want, None
            try:
                todo = self.cache.missing(specs)
                if todo:
                    keys = {self.cache.key(s) for s in todo}
                    # abort only if something being built is no longer wanted
                    self.cache.build(todo, progress=self.progress.emit,
                                     cancel=lambda: self._stop or not keys <= self._wanted_keys)
            except Cancelled:
                continue
            except Exception:       # report and keep serving later requests
                traceback.print_exc()
            self.finished.emit()


class WindowWorker(QtCore.QObject):
    """Processes just a window of data for specs that aren't cached yet (fast feedback after an edit)."""
    ready = QtCore.Signal(str, float, float, object)     # key, t_lo, t_hi, PyramidTrace

    def __init__(self, rec, cache):
        super().__init__()
        self.rec, self.cache = rec, cache
        self._cv = threading.Condition()
        self._job = None
        threading.Thread(target=self._loop, daemon=True, name="window-worker").start()

    def request(self, specs, t_lo, t_hi):
        with self._cv:
            self._job = ([dict(s) for s in specs], t_lo, t_hi)
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                (specs, t_lo, t_hi), self._job = self._job, None
            try:
                fs = self.rec.fs
                i0, i1 = int(max(t_lo, 0) * fs), int(min(t_hi, self.rec.duration) * fs)
                pad = max(pad_samples(s, fs) for s in specs)
                raw, first = self.rec.read(i0 - pad, i1 + pad)
                for s in specs:
                    t, y = process_window(self.rec, s, i0, i1, raw, first)
                    if len(t) > 1:
                        self.ready.emit(self.cache.key(s), t_lo, t_hi, PyramidTrace.from_samples(t, y))
                    with self._cv:
                        if self._job is not None:     # superseded - drop the rest
                            break
            except Exception:
                traceback.print_exc()
