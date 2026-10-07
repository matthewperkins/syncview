"""Frame-exact video decoding (PyNvVideoCodec, GPU) and ffmpeg encoding."""
import subprocess
import warnings
from pathlib import Path

import numpy as np
import PyNvVideoCodec as nvc


class FrameSource:
    """Frame-exact random/sequential access to one or more videos (GPU decode via PyNvVideoCodec)."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in np.atleast_1d(paths)]
        self.decs = [nvc.SimpleDecoder(str(p), use_device_memory=False,
                                       output_color_type=nvc.OutputColorType.RGB) for p in self.paths]
        lens = [len(d) for d in self.decs]
        if len(set(lens)) > 1:
            warnings.warn(f"Videos have different frame counts {lens}; using the shortest.")
        self.n_frames = min(lens)
        self.sizes = [(d.get_stream_metadata().width, d.get_stream_metadata().height) for d in self.decs]

    def iter_frames(self, start, n, batch=50):
        for d in self.decs:
            d.seek_to_index(start)
        done = 0
        while done < n:
            k = min(batch, n - done)
            batches = [d.get_batch_frames(k) for d in self.decs]
            k = min(len(b) for b in batches)
            if k == 0:
                return
            for j in range(k):
                yield [np.from_dlpack(b[j]) for b in batches]
            done += k


def video_time_to_frame(path, t):
    """Frame index at player time t (s) in the video file (uses the real, jittery container timestamps)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    pts = np.sort(np.array([float(s) for s in out.split() if s.strip() not in ("", "N/A")]))
    return int(np.clip(np.searchsorted(pts, t - 1e-6), 0, len(pts) - 1))


def _ffmpeg_writer(path, w, h, fps, codec, quality):
    if codec is None:
        encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        codec = "h264_nvenc" if "h264_nvenc" in encs else "libx264"
    q = ["-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"] if "nvenc" in codec \
        else ["-preset", "medium", "-crf", str(quality)]
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", codec, *q, "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE), codec


def keyframe_interval(path, n_packets=2000):
    """Frames between keyframes if the stream has a fixed GOP (as NVENC/gstreamer writes), else None."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=flags",
                          "-read_intervals", f"%+#{n_packets}", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.split()
    keys = np.flatnonzero([f.startswith("K") for f in out])
    if len(keys) < 3 or keys[0] != 0:
        return None
    d = np.unique(np.diff(keys))
    return int(d[0]) if len(d) == 1 else None
