"""syncview - interactive viewer for Open Ephys data (and, later, synchronized video).

    syncview --rec "OEFiles/…/Record Node 101/experiment3/recording1" [--video BASLER_CAM_….mp4]
             [--preset my_channels.json]

Navigation (click the plots first so they have keyboard focus):
    drag            pan                         wheel           change time base (zoom)
    Ctrl+wheel      scale that row's Y range    double-click    reset that row's Y range to auto
    ← / →           one video frame             Shift+← / →     10 % of the time base
    PgUp / PgDn     one full time base          Home / End      start / end of recording
    Space           play / pause                [ / ]           slower / faster playback
    click/drag the overview strip (bottom) to jump anywhere
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.oe import OERecording, SyncError, check_sync
from ..core.render import PALETTE
from ..core.video import DECODERS, video_duration
from ..data.cache import TraceCache
from .channels import ChannelPanel
from .video import VideoDecoder, VideoView
from .workers import CacheBuilder, WindowWorker

TIME_BASES = [0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200]
RATES = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 100, 300]
WINDOW_MAX_S = 180.0        # on-demand (uncached) processing only for views up to this span

# Channel map: masseter CH13-16, digastric CH1-4, antrum CH5-8, duodenum CH9-12
DEFAULT_PRESET = dict(time_base=60.0, channels=(
    [dict(ch=f"CH{c}", label=f"masseter{i + 1}", mode="hilo") for i, c in enumerate(range(13, 17))]
    + [dict(ch=f"CH{c}", label=f"digastric{i + 1}", mode="hilo") for i, c in enumerate(range(1, 5))]
    + [dict(ch=f"CH{c}", label=f"antrum{i + 1}", mode="slow") for i, c in enumerate(range(5, 9))]
    + [dict(ch=f"CH{c}", label=f"duod{i + 1}", mode="slow") for i, c in enumerate(range(9, 13))]),
    overview=dict(ch="CH5", mode="bandpower", label="antrum1 slow-wave power (0.03-0.25 Hz RMS, 30 s)"))


def fmt_time(t, span=None):
    sign = "-" if t < 0 else ""
    t = abs(t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    dec = 3 if span is None or span < 10 else (1 if span < 300 else 0)
    sec = f"{s:0{3 + dec if dec else 2}.{dec}f}"
    return f"{sign}{int(h)}:{int(m):02d}:{sec}" if h else f"{sign}{int(m)}:{sec}"


def fmt_span(s):
    return f"{s:g} s" if s < 60 else (f"{s / 60:g} min" if s < 3600 else f"{s / 3600:g} h")


def parse_span(text):
    t = text.strip().lower().replace(" ", "")
    for suf, mul in (("min", 60), ("h", 3600), ("ms", 1e-3), ("s", 1), ("m", 60)):
        if t.endswith(suf):
            return float(t[: -len(suf)]) * mul
    return float(t)


NICE_STEPS = [1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600,
              900, 1800, 3600, 7200]


def nice_spacing(span, n_major=8):
    major = next((s for s in NICE_STEPS if span / s <= n_major), NICE_STEPS[-1])
    minor = next((s for s in reversed(NICE_STEPS) if s < major and major / s in (2, 2.5, 3, 4, 5, 6)), major / 5)
    return major, minor


class _NiceTimeTicks:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setStyle(maxTextLevel=0)          # label major ticks only

    def tickSpacing(self, minVal, maxVal, size):
        major, minor = nice_spacing(maxVal - minVal, max(3, int(size / 110)))
        return [(major, 0), (minor, 0)]


class TimeAxis(_NiceTimeTicks, pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [fmt_time(v, spacing * 10) for v in values]


class RelTimeAxis(_NiceTimeTicks, pg.AxisItem):
    """Time relative to the cursor. Static while playing, so it never needs repainting."""

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            if abs(v) < spacing * 1e-3:
                out.append("0")
            elif spacing >= 60:
                out.append(("+" if v > 0 else "") + fmt_time(v, spacing * 10))
            else:
                out.append(f"{v:+.{max(0, -int(np.floor(np.log10(spacing))))}f}")
        return out


class SparseYAxis(pg.AxisItem):
    """Few, major-only ticks, so stacked rows' labels don't collide."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setStyle(maxTickLevel=0, maxTextLevel=0)

    def tickSpacing(self, minVal, maxVal, size):
        span = abs(maxVal - minVal)
        if not np.isfinite(span) or span <= 0:
            return super().tickSpacing(minVal, maxVal, size)
        if self.geometry().height() < 50:            # very short row: just mark zero
            return [(span * 2, 0)]
        # largest round step <= 0.95 * half the range: 2-3 labelled ticks, all inside the row
        target = 0.95 * span / 2
        e = 10 ** np.floor(np.log10(target))
        step = max(m * e for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8) if m * e <= target)
        return [(step, 0)]


class RowViewBox(pg.ViewBox):
    """ViewBox whose mouse gestures drive the shared clock instead of its own range."""

    def __init__(self, win, row):
        super().__init__(enableMenu=False)
        self.win, self.row = win, row
        self.setMouseEnabled(x=False, y=False)

    def wheelEvent(self, ev, axis=None):
        steps = ev.delta() / 120
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            self.win.scale_row_y(self.row, 0.8 ** steps)
        elif ev.modifiers() & QtCore.Qt.ShiftModifier:
            self.win.set_time(self.win.t - steps * 0.1 * self.win.span)
        else:
            self.win.set_span(self.win.span * 0.8 ** steps)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        dx = self.mapToView(ev.pos()).x() - self.mapToView(ev.lastPos()).x()
        self.win.set_time(self.win.t - dx)

    def mouseDoubleClickEvent(self, ev):
        self.win.reset_row_y(self.row)
        ev.accept()


class OverviewViewBox(pg.ViewBox):
    def __init__(self, win):
        super().__init__(enableMenu=False)
        self.win = win
        self.setMouseEnabled(x=False, y=False)

    def _jump(self, ev):
        ev.accept()
        self.win.set_time(self.mapToView(ev.pos()).x())

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._jump(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton:
            self._jump(ev)
        else:
            ev.ignore()

    def wheelEvent(self, ev, axis=None):
        ev.ignore()


class Row:
    def __init__(self, spec, plot, curve, label):
        self.spec, self.plot, self.curve, self.label = spec, plot, curve, label
        self.key = None
        self.auto = None          # auto y-range, computed once per trace
        self.ylim_set = None      # y-range currently applied to the plot


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, rec, cache, preset, video=None, decoder="auto", trigger_line=1):
        super().__init__()
        self.rec, self.cache = rec, cache
        self.setWindowTitle(f"syncview — {rec.rec_dir.parent.name}/{rec.rec_dir.name}  "
                            f"({rec.duration / 3600:.2f} h, {rec.fs:g} Hz)")
        self.trigger_line = trigger_line
        self.frames = rec.rising_edges(trigger_line) / rec.fs  # frame trigger times (s); refined once video is attached
        self.t = float(self.frames[0]) if len(self.frames) else 0.0
        self.span = 10.0
        self.rate = 1.0
        self.rows = []
        self.windows = {}          # key -> (t_lo, t_hi, PyramidTrace) from the on-demand worker
        self._refresh_pending = False
        self._last_window_req = None
        self._xrange = None
        self.decoder = None
        self.decoder_backend = decoder
        self.video_path = None
        self.frame_period = 0.02
        self._want_frame = None

        pg.setConfigOptions(antialias=False, background="#111111", foreground="#c3c2b7")

        # ---- central area: data rows + overview strip
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setSpacing(2)
        self.ov_widget = pg.GraphicsLayoutWidget()
        self.ov_widget.setFixedHeight(110)
        self.ov_plot = self.ov_widget.addPlot(viewBox=OverviewViewBox(self),
                                              axisItems={"bottom": TimeAxis("bottom")})
        self.ov_plot.getAxis("left").setWidth(70)
        self.ov_plot.hideButtons()
        self.ov_curve = self.ov_plot.plot(pen=pg.mkPen("#c3c2b7", width=1))
        self.ov_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(255, 255, 255, 40),
                                             pen=pg.mkPen(None))
        self.ov_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ffffff", width=1))
        self.ov_plot.addItem(self.ov_region)
        self.ov_plot.addItem(self.ov_cursor)
        self.ov_title = pg.TextItem("", color="#c3c2b7", anchor=(0, 0))
        self.ov_plot.addItem(self.ov_title, ignoreBounds=True)

        self.video_view = VideoView()
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.video_view)
        self.splitter.addWidget(self.glw)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)
        self.video_view.hide()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.splitter, 1)
        lay.addWidget(self.ov_widget)
        self.setCentralWidget(central)

        # ---- toolbar
        tb = self.addToolBar("view")
        tb.setMovable(False)
        self.play_btn = QtWidgets.QPushButton("▶")
        self.play_btn.setFixedWidth(36)
        self.play_btn.clicked.connect(self.toggle_play)
        tb.addWidget(self.play_btn)
        self.rate_cb = QtWidgets.QComboBox()
        self.rate_cb.addItems([f"{r:g}×" for r in RATES])
        self.rate_cb.setCurrentText("1×")
        self.rate_cb.currentTextChanged.connect(lambda s: setattr(self, "rate", float(s[:-1])))
        tb.addWidget(self.rate_cb)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" time base "))
        self.span_cb = QtWidgets.QComboBox()
        self.span_cb.setEditable(True)
        self.span_cb.addItems([fmt_span(s) for s in TIME_BASES])
        self.span_cb.lineEdit().returnPressed.connect(self._span_typed)
        self.span_cb.activated.connect(self._span_typed)
        self.span_cb.setMinimumWidth(90)
        tb.addWidget(self.span_cb)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" go to "))
        self.goto = QtWidgets.QLineEdit()
        self.goto.setPlaceholderText("h:mm:ss or seconds")
        self.goto.setFixedWidth(130)
        self.goto.returnPressed.connect(self._goto)
        tb.addWidget(self.goto)
        tb.addSeparator()
        vid_btn = QtWidgets.QPushButton("Video…")
        vid_btn.clicked.connect(self._choose_video)
        tb.addWidget(vid_btn)
        tb.addSeparator()
        self.clock = QtWidgets.QLabel()
        self.clock.setFont(QtGui.QFont("monospace", 11))
        tb.addWidget(self.clock)

        # ---- channel dock
        self.panel = ChannelPanel(rec.ch_names)
        dock = QtWidgets.QDockWidget("Channels", self)
        dock.setWidget(self.panel)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self._dock = dock
        self.panel.specs_changed.connect(self.set_specs)

        # ---- status bar: cache build progress
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setFormat("filtering session… %p%")
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        # ---- workers
        self.builder = CacheBuilder(cache)
        self.builder.progress.connect(self._build_progress)
        self.builder.finished.connect(self._build_finished)
        self.window_worker = WindowWorker(rec, cache)
        self.window_worker.ready.connect(self._window_ready)

        self.play_timer = QtCore.QTimer(self, interval=4)      # ~as fast as drawing allows; t follows the wall clock
        self.play_timer.timeout.connect(self._tick)
        self._last_tick = None

        # navigation keys go to the window even when a plot widget has focus
        for w in (self.glw, self.ov_widget):
            w.installEventFilter(self)
            w.viewport().installEventFilter(self)

        self.overview_spec = preset.get("overview")
        self.apply_preset(preset)
        self.resize(1700, 1000)
        if video:
            self.attach_video(video)

    NAV_KEYS = {QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown,
                QtCore.Qt.Key_Home, QtCore.Qt.Key_End, QtCore.Qt.Key_Space, QtCore.Qt.Key_BracketLeft,
                QtCore.Qt.Key_BracketRight, QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal, QtCore.Qt.Key_Minus,
                QtCore.Qt.Key_Up, QtCore.Qt.Key_Down}

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.KeyPress and ev.key() in self.NAV_KEYS:
            self.keyPressEvent(ev)
            return True
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------ presets / rows
    def preset(self):
        return dict(time_base=self.span, channels=self.panel.specs(), overview=self.overview_spec)

    def apply_preset(self, p):
        self.span = float(p.get("time_base", self.span))
        self.span_cb.setEditText(fmt_span(self.span))
        self.overview_spec = p.get("overview")
        self.panel.set_specs(p.get("channels", []))      # -> set_specs()

    def set_specs(self, specs):
        if not self.overview_spec:      # default overview: slow-wave power of the first slow row
            slow = next((s for s in specs if s["mode"] == "slow"), None)
            if slow:
                self.overview_spec = dict(ch=slow["ch"], ref=slow.get("ref"), mode="bandpower",
                                          label=f"{slow.get('label') or slow['ch']} slow-wave power")
        self.glw.clear()
        self.rows = []
        self._xrange = None
        visible = [s for s in specs if s.get("show", True)]
        self._row_index = [i for i, s in enumerate(specs) if s.get("show", True)]   # row -> table row
        first = None
        for i, s in enumerate(visible):
            label = pg.LabelItem(s.get("label") or s["ch"], color="#e8e8e4", size="10pt", justify="right")
            label.setMinimumWidth(130)
            label.setMaximumWidth(130)
            self.glw.addItem(label, row=i, col=0)
            vb = RowViewBox(self, i)
            last = i == len(visible) - 1
            axes = {"left": SparseYAxis("left")}
            if last:
                axes["bottom"] = RelTimeAxis("bottom")
            p = self.glw.addPlot(row=i, col=1, viewBox=vb, axisItems=axes)
            p.hideButtons()
            p.getAxis("left").setWidth(60)
            if last:
                p.setLabel("bottom", "time relative to cursor   ← past | future →")
            else:
                p.hideAxis("bottom")
            if first is None:
                first = p
            else:
                p.setXLink(first)
            color = s.get("color", PALETTE[i % len(PALETTE)])
            p.vb.disableAutoRange()
            curve = pg.PlotCurveItem(pen=pg.mkPen(color, width=1), skipFiniteCheck=True)
            p.addItem(curve)
            cursor = pg.InfiniteLine(0, angle=90, pen=pg.mkPen("#ffffff", width=1))
            p.addItem(cursor, ignoreBounds=True)
            msg = pg.TextItem("", color="#888888", anchor=(0.5, 0.5))
            p.addItem(msg, ignoreBounds=True)
            row = Row(s, p, curve, label)
            row.cursor, row.msg = cursor, msg
            row.key = self.cache.key(s)
            self.rows.append(row)
        # QGraphicsGridLayout keeps stale column geometry after clear(); make the plot column take the
        # free width and force a relayout now (otherwise new plots stay squeezed until the window resizes)
        lay = self.glw.ci.layout
        lay.setColumnStretchFactor(0, 0)
        lay.setColumnStretchFactor(1, 1)
        lay.invalidate()
        self.glw.ci.setGeometry(QtCore.QRectF(self.glw.viewport().rect()))
        lay.activate()
        # only rebuild what is needed; the overview trace goes first
        want = ([self.overview_spec] if self.overview_spec else []) + visible
        self.builder.request(want)
        self.windows = {k: v for k, v in self.windows.items() if k in {r.key for r in self.rows}}
        self._update_overview()
        self.refresh()

    # ------------------------------------------------------------------ clock
    def set_time(self, t):
        self.t = float(np.clip(t, 0, self.rec.duration))
        self.refresh()

    def set_span(self, s):
        self.span = float(np.clip(s, 0.05, self.rec.duration))
        self.span_cb.setEditText(fmt_span(round(self.span, 3)))
        self.refresh()

    def _span_typed(self, *_):
        try:
            self.set_span(parse_span(self.span_cb.currentText()))
        except ValueError:
            pass

    def _goto(self):
        txt = self.goto.text().strip()
        try:
            parts = [float(p) for p in txt.split(":")]
            t = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
            self.set_time(t)
        except ValueError:
            pass

    def step_frames(self, n):
        if len(self.frames):
            k = int(np.searchsorted(self.frames, self.t + 1e-6)) - 1      # frame at or before t
            k = int(np.clip(k + n, 0, len(self.frames) - 1))
            self.set_time(self.frames[k])
        else:
            self.set_time(self.t + n * 0.02)

    def current_frame(self):
        """Last frame triggered at or before the cursor (None outside the video)."""
        if not len(self.frames) or self.t > self.frames[-1] + self.frame_period:
            return None
        k = int(np.searchsorted(self.frames, self.t + 1e-6)) - 1
        return k if 0 <= k < len(self.frames) else None

    # ------------------------------------------------------------------ video
    def _choose_video(self):
        start = str(Path(self.video_path).parent) if self.video_path else ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Video recorded during this session", start,
                                                      "Video (*.mp4 *.mkv *.mov *.avi)")
        if fn:
            self.attach_video(fn)

    def attach_video(self, path):
        if self.decoder:
            self.decoder.stop()
        self.video_path = str(path)
        self.video_view.show()
        self.video_view.set_message(f"opening {Path(path).name} …")
        self.decoder = VideoDecoder(path, backend=self.decoder_backend)
        self.video_view.decoder = self.decoder
        self.decoder.target = (self.video_view.width(), self.video_view.height())
        self.decoder.opened.connect(self._video_opened)
        self.decoder.failed.connect(self.video_view.set_message)
        self.decoder.frame_ready.connect(self._frame_ready)

    def _video_opened(self, n, w, h, decoder_name):
        name = Path(self.video_path).name
        try:
            fr, issues, info = check_sync(self.rec, n, self.trigger_line, video_duration(self.video_path))
        except SyncError as e:
            self.video_view.set_message(f"{name}: {e}")
            self._sync_warning(name, [str(e)], fatal=True)
            self.decoder.stop()
            self.decoder = None
            return
        if issues:
            self._sync_warning(name, issues)
        self.frames = fr / self.rec.fs
        self.frame_period = info["period"]
        extra = info["extra"]
        self.statusBar().showMessage(f"{name}: {n} frames {w}×{h}, {extra} extra trigger(s) "
                                     f"dropped; video spans {fmt_time(self.frames[0])} – {fmt_time(self.frames[-1])}; "
                                     f"decoding on {decoder_name}",
                                     15000)
        if self.t < self.frames[0] or self.t > self.frames[-1]:
            self.t = float(self.frames[0])
        self._want_frame = None
        self.refresh()

    def _sync_warning(self, name, issues, fatal=False):
        text = "\n\n".join(f"• {m}" for m in issues)
        for m in issues:
            print(f"syncview: sync {'error' if fatal else 'warning'} ({name}): {m}", file=sys.stderr)
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical if fatal else QtWidgets.QMessageBox.Warning,
                                    "Video sync " + ("failed" if fatal else "warning"),
                                    f"{name} on TTL line {self.trigger_line}:\n\n{text}", parent=self)
        box.setModal(False)
        box.show()
        self._sync_box = box

    def _frame_ready(self, k, arr):
        if self.decoder is None:
            return
        self.video_view.set_frame(arr, f"frame {k}   trigger {fmt_time(self.frames[k])}")

    def _update_video(self):
        if self.decoder is None or not self.video_view.isVisible():
            return
        k = self.current_frame()
        if k is None:
            self._want_frame = None
            self.video_view.set_message("no video at this time")
        elif k != self._want_frame:
            self._want_frame = k
            self.decoder.request(k)

    def closeEvent(self, ev):
        if self.decoder:
            self.decoder.stop()
        super().closeEvent(ev)

    # ------------------------------------------------------------------ playback
    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("▶")
        else:
            self._last_tick = time.perf_counter()
            self.play_timer.start()
            self.play_btn.setText("⏸")

    def _tick(self):
        now = time.perf_counter()
        dt, self._last_tick = now - self._last_tick, now
        if self.t >= self.rec.duration:
            self.toggle_play()
            return
        self.set_time(self.t + dt * self.rate)

    def change_rate(self, d):
        i = int(np.clip(self.rate_cb.currentIndex() + d, 0, len(RATES) - 1))
        self.rate_cb.setCurrentIndex(i)

    # ------------------------------------------------------------------ y ranges
    def _auto_ylim(self, row, trace):
        st = trace.stats
        mode = row.spec["mode"]
        if not st:
            return (-1, 1)
        if mode == "hilo":
            m = st["absmax99.9"] * 2.0
            return (-m, m)
        if mode in ("envelope", "bandpower"):
            return (0, st["99.9"] * 1.2)
        lo, hi = st["0.5"], st["99.5"]
        r = hi - lo
        return (lo - 0.1 * r, hi + 0.1 * r)

    def _row_ylim(self, row, trace):
        if row.spec.get("ylim"):
            return row.spec["ylim"]
        if row.auto is None and trace is not None:
            row.auto = self._auto_ylim(row, trace)
        return row.auto

    def scale_row_y(self, i, f):
        row = self.rows[i]
        tr = self._trace(row)
        lo, hi = self._row_ylim(row, tr) or (-1, 1)
        if row.spec["mode"] in ("envelope", "bandpower"):
            new = (lo, lo + (hi - lo) * f)
        else:
            c = (lo + hi) / 2
            new = (c - (hi - lo) / 2 * f, c + (hi - lo) / 2 * f)
        row.spec["ylim"] = new
        self.panel.set_ylim(self._row_index[i], new)
        self.refresh()

    def reset_row_y(self, i):
        row = self.rows[i]
        row.spec.pop("ylim", None)
        row.auto = None
        self.panel.set_ylim(self._row_index[i], None)
        self.refresh()

    # ------------------------------------------------------------------ data
    def _trace(self, row):
        tr = self.cache.get(row.spec)
        if tr is not None:
            return tr
        w = self.windows.get(row.key)
        if w and w[0] <= self.t - self.span / 2 and self.t + self.span / 2 <= w[1]:
            return w[2]
        return None

    def _window_ready(self, key, t_lo, t_hi, trace):
        self.windows[key] = (t_lo, t_hi, trace)
        self.refresh()

    def _build_progress(self, f):
        self.progress.show()
        self.progress.setValue(int(f * 100))

    def _build_finished(self):
        self.progress.hide()
        for r in self.rows:
            r.auto = None           # switch from window-based to session-based auto ranges
        self._update_overview()
        self.refresh()

    def _update_overview(self):
        spec = self.overview_spec
        tr = self.cache.get(spec) if spec else None
        self.ov_title.setText(spec.get("label", "") if spec else "")
        self.ov_plot.setXRange(0, self.rec.duration, padding=0)
        if tr is None:
            self.ov_curve.clear()
            return
        x, y = tr.view(0, self.rec.duration, max(int(self.ov_plot.vb.width()), 200))
        self.ov_curve.setData(x, y)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        self.ov_plot.setYRange(lo, hi, padding=0.05)
        self.ov_title.setPos(0, hi)

    # ------------------------------------------------------------------ drawing
    def refresh(self):
        if not self._refresh_pending:
            self._refresh_pending = True
            QtCore.QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        self._refresh_pending = False
        t, span = self.t, self.span
        t_lo, t_hi = t - span / 2, t + span / 2
        need_window = []
        if self.rows and self._xrange != span:
            self.rows[0].plot.setXRange(-span / 2, span / 2, padding=0)     # rows are x-linked
            self._xrange = span
        for row in self.rows:
            tr = self._trace(row)
            if tr is None:
                row.curve.clear()
                row.msg.setText("filtering…" if span <= WINDOW_MAX_S else "filtering session… (zoom in to preview)")
                row.msg.setPos(0, 0)
                if row.ylim_set != (-1.0, 1.0):
                    row.plot.setYRange(-1, 1, padding=0)
                    row.ylim_set = (-1.0, 1.0)
                if span <= WINDOW_MAX_S:
                    need_window.append(row.spec)
                continue
            row.msg.setText("")
            n_px = max(int(row.plot.vb.width()), 100)
            x, y = tr.view(t_lo - 0.01 * span, t_hi + 0.01 * span, n_px * 1.02)
            row.curve.setData(x - t, y)
            ylim = tuple(map(float, self._row_ylim(row, tr)))
            if row.ylim_set != ylim:
                row.plot.setYRange(*ylim, padding=0)
                row.ylim_set = ylim
        if need_window:
            req = (tuple(self.cache.key(s) for s in need_window), round(t_lo, 3), round(t_hi, 3))
            if req != self._last_window_req:
                self._last_window_req = req
                self.window_worker.request(need_window, t_lo - span, t_hi + span)   # margin for panning
        # overview: one pixel there is many seconds, so only touch it when something visibly moves
        px = self.rec.duration / max(self.ov_plot.vb.width(), 1)
        ov = (round(t_lo / px), round(t_hi / px), round(t / px))
        if ov != getattr(self, "_ov_state", None):
            self._ov_state = ov
            self.ov_region.setRegion((t_lo, t_hi))
            self.ov_cursor.setValue(t)
        self._update_video()
        k = self.current_frame()
        fr = f"   frame {k}" if k is not None else ""
        self.clock.setText(f"  t = {fmt_time(t)}  ({t:.3f} s){fr}   ")

    # ------------------------------------------------------------------ keys
    def keyPressEvent(self, ev):
        k, mod = ev.key(), ev.modifiers()
        shift = bool(mod & QtCore.Qt.ShiftModifier)
        if k in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            d = -1 if k == QtCore.Qt.Key_Left else 1
            if shift:
                self.set_time(self.t + d * 0.1 * self.span)
            else:
                self.step_frames(d)
        elif k == QtCore.Qt.Key_PageUp:
            self.set_time(self.t - self.span)
        elif k == QtCore.Qt.Key_PageDown:
            self.set_time(self.t + self.span)
        elif k == QtCore.Qt.Key_Home:
            self.set_time(0)
        elif k == QtCore.Qt.Key_End:
            self.set_time(self.rec.duration)
        elif k == QtCore.Qt.Key_Space:
            self.toggle_play()
        elif k == QtCore.Qt.Key_BracketLeft:
            self.change_rate(-1)
        elif k == QtCore.Qt.Key_BracketRight:
            self.change_rate(1)
        elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.set_span(self.span * 0.8)
        elif k == QtCore.Qt.Key_Minus:
            self.set_span(self.span / 0.8)
        else:
            super().keyPressEvent(ev)

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_dock_sized", False):
            self._dock_sized = True
            QtCore.QTimer.singleShot(0, lambda: self.resizeDocks([self._dock], [470], QtCore.Qt.Horizontal))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        QtCore.QTimer.singleShot(0, self._update_overview)
        self.refresh()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True, help="Open Ephys recording folder (…/experimentN/recordingM)")
    ap.add_argument("--video", help="video recorded during this recording (one camera)")
    ap.add_argument("--preset", help="channel preset JSON (save one from the Channels panel)")
    ap.add_argument("--cache", default=None,
                    help="cache folder for filtered traces (default: syncview_cache/ next to the syncview package)")
    ap.add_argument("--stream", default="acquisition_board",
                    help="Open Ephys continuous stream holding the data and the camera TTL (default: %(default)s)")
    ap.add_argument("--trigger-line", type=int, default=1,
                    help="TTL line carrying one pulse per video frame (default: %(default)s)")
    ap.add_argument("--decoder", choices=DECODERS, default="auto",
                    help="video decoding: gpu (NVIDIA, PyNvVideoCodec), cpu (FFmpeg via PyAV), "
                         "or auto = gpu if available (default)")
    args = ap.parse_args(argv)

    try:
        rec = OERecording(args.rec, stream=args.stream)
    except (OSError, ValueError) as e:
        ap.exit(2, f"syncview: cannot open recording: {e}\n")
    cache_root = Path(args.cache) if args.cache else Path(__file__).resolve().parents[2] / "syncview_cache"
    cache = TraceCache(rec, cache_root)
    preset = json.loads(Path(args.preset).read_text()) if args.preset else DEFAULT_PRESET

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow(rec, cache, preset, video=args.video, decoder=args.decoder,
                     trigger_line=args.trigger_line)
    win.show()
    win.glw.setFocus()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
