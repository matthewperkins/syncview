"""Render synchronized video + scrolling-data clips (matplotlib, off-screen, blitted)."""
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

from .oe import frame_sample_indices
from .filters import pad_samples, process_window
from .video import FrameSource, video_time_to_frame, _ffmpeg_writer


PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]  # validated dark-mode slots


class SyncRenderer:
    def __init__(self, rec, video_paths, channels, start_frame, n_frames,
                 time_base=4.0, video_frac=0.6, out_width=None, trigger_line=1,
                 dark=True, fontsize=11, show_clock=True, label_width=170, frame_samples=None):
        self.rec, self.fs = rec, rec.fs
        self.video = FrameSource(video_paths)
        self.fr = frame_samples if frame_samples is not None else \
            frame_sample_indices(rec, self.video.n_frames, trigger_line)
        if start_frame < 0 or start_frame + n_frames > self.video.n_frames:
            raise ValueError(f"frames {start_frame}..{start_frame + n_frames} outside video (0..{self.video.n_frames})")
        self.start, self.n = int(start_frame), int(n_frames)
        self.tb, self.show_clock = float(time_base), show_clock

        # ---- geometry: videos side by side at common height, scaled to out_width
        vh = max(h for _, h in self.video.sizes)
        widths = [int(round(w * vh / h)) for w, h in self.video.sizes]
        self.vid_native = (sum(widths), vh)
        W = int(out_width or sum(widths)) // 2 * 2
        self.vid_w, self.vid_h = W, int(round(vh * W / sum(widths))) // 2 * 2
        self.vid_tile_w = [int(round(w * W / sum(widths))) for w in widths]
        self.vid_tile_w[-1] = W - sum(self.vid_tile_w[:-1])
        H = int(round(self.vid_h / video_frac)) // 2 * 2
        self.W, self.H, self.data_h = W, H, H - self.vid_h
        if self.data_h < 40 * len(channels):
            warnings.warn("Data panel is very short for this many channels - lower video_frac.")

        # ---- figure for the data panel
        fg, bg, grid = ("#e8e8e4", "#111111", "#3a3a38") if dark else ("#1a1a19", "#ffffff", "#d8d8d4")
        self.colors = dict(fg=fg, bg=bg)
        dpi = 100
        self.fig = Figure(figsize=(W / dpi, self.data_h / dpi), dpi=dpi, facecolor=bg)  # off-screen
        FigureCanvasAgg(self.fig)
        nch = len(channels)
        left, right = label_width / W, 1 - 14 / W
        bottom, top = 42 / self.data_h, 1 - 8 / self.data_h
        gap = 14 / self.data_h
        h_ax = (top - bottom - gap * (nch - 1)) / nch
        self.axes, self.lines = [], []
        half = self.tb / 2
        for i, spec in enumerate(channels):
            y0 = top - (i + 1) * h_ax - i * gap
            ax = self.fig.add_axes([left, y0, right - left, h_ax], facecolor=bg)
            ax.set_xlim(-half, half)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(colors=fg, labelsize=fontsize - 2, length=3)
            ax.yaxis.set_major_locator(MaxNLocator(2, symmetric=spec["mode"] == "hilo"))
            ax.grid(True, axis="x", color=grid, lw=0.6)
            ax.axvline(0, color=fg, lw=1.0, alpha=0.8, zorder=5)
            if i < nch - 1:
                ax.tick_params(labelbottom=False, bottom=False)
            unit = spec.get("unit", "µV")
            ax.set_ylabel(f"{spec.get('label', spec['ch'])}\n({unit})", color=fg, fontsize=fontsize,
                          rotation=0, ha="right", va="center", labelpad=8)
            color = spec.get("color", PALETTE[i % len(PALETTE)])
            ln, = ax.plot([], [], color=color, lw=spec.get("lw", 1.0), animated=True, antialiased=True)
            self.axes.append(ax)
            self.lines.append(ln)
        self.axes[-1].set_xlabel("time relative to video frame (s)   ← past | future →", color=fg,
                                 fontsize=fontsize - 1)
        self.fig.canvas.draw()  # layout is now fixed; needed for pixel widths below
        self.ax_px = int(round(self.axes[0].get_window_extent().width))

        # ---- load + process the data for the whole clip (+ filter padding)
        self.traces = []
        s0 = self.fr[self.start] - int(half * self.fs) - 2
        s1 = self.fr[self.start + self.n - 1] + int(half * self.fs) + 2
        pad = max(pad_samples(s, self.fs) for s in channels)
        raw, first = rec.read(s0 - pad, s1 + pad)   # first = absolute row index of raw[0]
        for ax, spec in zip(self.axes, channels):
            t, y = process_window(rec, spec, s0, s1, raw, first)
            if spec["mode"] == "hilo":
                tr = self._hilo(y, int(round(t[0] * self.fs)))
            else:
                tr = (t, y)
            self.traces.append(tr)
            ax.set_ylim(*self._ylim(spec, tr[1]))
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.cursor_lines = [ax.lines[0] for ax in self.axes]  # the axvline at 0, redrawn on top

        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", max(12, self.vid_h // 30))
        except OSError:
            self.font = ImageFont.load_default()

    # global pixel-column grid in absolute time, so a bin's min/max doesn't flicker frame to frame
    def _hilo(self, y, idx0):
        bin_s = self.tb / self.ax_px * self.fs               # samples per pixel column (may be fractional)
        k0 = int(np.ceil(idx0 / bin_s))
        k1 = int(np.floor((idx0 + len(y)) / bin_s))
        edges = np.round(np.arange(k0, k1 + 1) * bin_s).astype(np.int64) - idx0
        edges = edges[(edges >= 0) & (edges < len(y))]
        lo = np.minimum.reduceat(y, edges[:-1])
        hi = np.maximum.reduceat(y, edges[:-1])
        t = (idx0 + (edges[:-1] + edges[1:]) / 2) / self.fs
        # draw as  lo,hi,lo,hi …  -> the "janky" /\/\ line that keeps every peak
        tt = np.repeat(t, 2)
        yy = np.empty(2 * len(lo)); yy[0::2], yy[1::2] = lo, hi
        return tt, yy

    @staticmethod
    def _ylim(spec, y):
        if spec.get("ylim"):
            return spec["ylim"]
        y = y[np.isfinite(y)]
        if spec["mode"] == "hilo":
            m = np.percentile(np.abs(y), 99.95) * 1.1
            return -m, m
        if spec["mode"] in ("envelope", "bandpower"):
            return 0, np.percentile(y, 99.9) * 1.15
        lo, hi = np.percentile(y, [0.2, 99.8])
        r = hi - lo
        return lo - 0.1 * r, hi + 0.1 * r

    def data_panel(self, k):
        """RGB array of the data panel for clip frame k (0-based within the clip)."""
        t0 = self.fr[self.start + k] / self.fs
        half = self.tb / 2
        canvas = self.fig.canvas
        canvas.restore_region(self.background)
        for ax, ln, (t, y), cur in zip(self.axes, self.lines, self.traces, self.cursor_lines):
            a, b = np.searchsorted(t, [t0 - half - 0.01, t0 + half + 0.01])
            ln.set_data(t[a:b] - t0, y[a:b])
            ax.draw_artist(ln)
            ax.draw_artist(cur)
        return np.asarray(canvas.buffer_rgba())[:, :, :3]

    def compose(self, k, tiles):
        vid = [Image.fromarray(np.ascontiguousarray(f)) for f in tiles]
        canvas = Image.new("RGB", (self.vid_w, self.vid_h))
        x = 0
        for im, w in zip(vid, self.vid_tile_w):
            if im.size != (w, self.vid_h):
                im = im.resize((w, self.vid_h), Image.BILINEAR)
            canvas.paste(im, (x, 0))
            x += w
        if self.show_clock:
            f = self.start + k
            txt = f"t = {self.fr[f] / self.fs:9.3f} s   frame {f}"
            ImageDraw.Draw(canvas).text((10, self.vid_h - 8), txt, anchor="ld", fill=(255, 255, 255), font=self.font,
                                        stroke_width=2, stroke_fill=(0, 0, 0))
        return np.vstack([np.asarray(canvas), self.data_panel(k)])

    def frames(self):
        for k, tiles in enumerate(self.video.iter_frames(self.start, self.n)):
            yield self.compose(k, tiles)

    def preview(self, k=0):
        tiles = next(self.video.iter_frames(self.start + k, 1))
        return self.compose(k, tiles)


def build_renderer(rec, video_paths, channels, start, duration, start_ref="frame", time_base=4.0,
                   video_frac=0.6, out_width=None, trigger_line=1, **style):
    video_paths = [Path(p) for p in np.atleast_1d(video_paths)]
    n_vid = FrameSource(video_paths).n_frames
    fr = frame_sample_indices(rec, n_vid, trigger_line, verbose=False)
    cam_fps = rec.fs / np.median(np.diff(fr))
    if start_ref == "frame":
        f0 = int(start)
    elif start_ref == "video":
        f0 = video_time_to_frame(video_paths[0], start)
    elif start_ref == "ephys":
        f0 = int(np.searchsorted(fr, start * rec.fs))
    else:
        raise ValueError("start_ref must be 'frame', 'video' or 'ephys'")
    n = max(1, int(round(duration * cam_fps)))
    r = SyncRenderer(rec, video_paths, [dict(c) for c in channels], f0, n, time_base=time_base,
                     video_frac=video_frac, out_width=out_width, trigger_line=trigger_line,
                     frame_samples=fr, **style)
    r.cam_fps = cam_fps
    return r


def make_sync_video(out_path, rec, video_paths, channels, start, duration, start_ref="frame",
                    time_base=4.0, video_frac=0.6, out_width=None, fps_out=None, trigger_line=1,
                    codec=None, quality=20, **style):
    """Render a clip: behaviour video on top, data scrolling right->left underneath, with the frame's
    exact trigger time in the centre of the chart.

    start      : clip start, interpreted by start_ref:
                   "frame" - video frame index
                   "video" - seconds in the movie file as a video player shows it
                   "ephys" - seconds since the start of the Open Ephys recording
    duration   : clip length in seconds of real (camera) time
    time_base  : seconds of data visible across the chart (half past, half future)
    video_frac : fraction of the output height used by the video (rest is data)
    out_width  : output width in px (default: native width of the video(s) side by side)
    fps_out    : playback frame rate (default = camera rate -> real time; half that = 2x slow motion)
    style      : dark=True, fontsize=11, show_clock=True, label_width=170
    """
    t_build = time.time()
    r = build_renderer(rec, video_paths, channels, start, duration, start_ref, time_base,
                       video_frac, out_width, trigger_line, **style)
    fps_out = fps_out or round(r.cam_fps, 3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    proc, codec = _ffmpeg_writer(out_path, r.W, r.H, fps_out, codec, quality)
    f0, n, fr = r.start, r.n, r.fr
    print(f"frames {f0}..{f0 + n - 1} | data {fr[f0] / rec.fs:.3f}-{fr[f0 + n - 1] / rec.fs:.3f} s | "
          f"{r.W}x{r.H} @ {fps_out} fps | {codec} | setup {time.time() - t_build:.1f} s")
    t = time.time()
    try:
        for k, img in enumerate(r.frames()):
            proc.stdin.write(img.tobytes())
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{n} frames ({(k + 1) / (time.time() - t):.0f} fps)", end="\r")
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    print(f"\nwrote {out_path}  ({n} frames in {time.time() - t:.1f} s)")
    return Path(out_path)


def preview_sync_frame(rec, video_paths, channels, start, start_ref="frame", time_base=4.0,
                       video_frac=0.6, out_width=None, trigger_line=1, **style):
    """Render just one output frame (fast) to tune channels, y-limits and layout before encoding."""
    r = build_renderer(rec, video_paths, channels, start, 0, start_ref, time_base, video_frac,
                       out_width, trigger_line, **style)
    img = r.preview(0)
    try:
        from IPython.display import display
        display(Image.fromarray(img))
    except ImportError:
        pass
    return img


def channel_overview(rec, t_start=600.0, dur=60.0):
    """Band-limited RMS (µV for CH, V for ADC) of every channel over a stretch of data - helps tell channels apart."""
    i0 = int(t_start * rec.fs)
    seg = np.asarray(rec.data[i0:i0 + int(dur * rec.fs)]).astype(np.float64) * rec.bit_volts
    bands = [(0.1, 1), (1, 10), (10, 100), (58, 62), (150, 2000)]
    print(f"{'ch':>5} " + "".join(f"{f'{a}-{b} Hz':>12}" for a, b in bands))
    for c, name in enumerate(rec.ch_names):
        f, p = signal.welch(seg[:, c], rec.fs, nperseg=int(rec.fs * 4))
        rms = [np.sqrt(np.trapezoid(p[(f >= a) & (f < b)], f[(f >= a) & (f < b)])) for a, b in bands]
        print(f"{name:>5} " + "".join(f"{v:12.1f}" for v in rms))
