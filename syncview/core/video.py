"""Frame-exact video decoding (GPU via PyNvVideoCodec, or CPU via PyAV) and ffmpeg encoding."""
import subprocess
import warnings
from pathlib import Path

import numpy as np

DECODERS = ("auto", "gpu", "cpu")


class NvDecoder:
    """GPU (NVDEC) decoding via PyNvVideoCodec. Needs an NVIDIA GPU + driver."""
    name = "GPU (NVDEC)"

    def __init__(self, path):
        import PyNvVideoCodec as nvc
        self._dec = nvc.SimpleDecoder(str(path), use_device_memory=False, output_color_type=nvc.OutputColorType.RGB)
        md = self._dec.get_stream_metadata()
        self.width, self.height = md.width, md.height

    def __len__(self):
        return len(self._dec)

    def seek_to_index(self, i):
        self._dec.seek_to_index(int(i))

    def get_batch_frames(self, k):
        """Next k frames as (h, w, 3) uint8 arrays. They may share the decoder's buffers: copy to keep them."""
        return [np.from_dlpack(f) for f in self._dec.get_batch_frames(int(k))]


class CpuDecoder:
    """CPU decoding via PyAV (FFmpeg); same interface and frame indexing as NvDecoder.

    Frame index = rank of the packet's presentation timestamp, read from the container once at open
    (a quick demux, no decoding), so indexing stays frame-exact even when the timestamps jitter."""
    name = "CPU (FFmpeg)"

    def __init__(self, path):
        import av
        self._c = av.open(str(path))
        self._s = self._c.streams.video[0]
        self._s.thread_type = "AUTO"
        self.pts = np.sort(np.array([pk.pts for pk in self._c.demux(self._s)
                                     if pk.size and pk.pts is not None], dtype=np.int64))
        if not len(self.pts):
            raise ValueError("no timestamped video packets found")
        self.width, self.height = self._s.codec_context.width, self._s.codec_context.height
        self._frames = None
        self.seek_to_index(0)

    def __len__(self):
        return len(self.pts)

    def seek_to_index(self, i):
        # lands on the keyframe at/before frame i; get_batch_frames decodes forward and discards up to i
        self._c.seek(int(self.pts[i]), stream=self._s, backward=True, any_frame=False)
        self._frames = self._c.decode(self._s)
        self._want = int(i)

    def get_batch_frames(self, k):
        out = []
        while len(out) < k:
            f = next(self._frames, None)
            if f is None:
                break
            i = int(np.searchsorted(self.pts, f.pts)) if f.pts is not None else self._want
            if i < self._want:
                continue
            out.append(f.to_ndarray(format="rgb24"))
            self._want = i + 1
        return out


def open_decoder(path, backend="auto"):
    """A frame decoder for `path`. backend: 'gpu', 'cpu', or 'auto' (GPU if it works here, else CPU)."""
    if backend not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, not {backend!r}")
    if backend == "cpu":
        return CpuDecoder(path)
    try:
        return NvDecoder(path)
    except Exception as e:
        if backend == "gpu":
            raise
        warnings.warn(f"GPU video decoding unavailable ({type(e).__name__}: {e}); using the CPU decoder.")
        return CpuDecoder(path)


class FrameSource:
    """Frame-exact random/sequential access to one or more videos."""

    def __init__(self, paths, decoder="auto"):
        self.paths = [Path(p) for p in np.atleast_1d(paths)]
        self.decs = [open_decoder(p, decoder) for p in self.paths]
        lens = [len(d) for d in self.decs]
        if len(set(lens)) > 1:
            warnings.warn(f"Videos have different frame counts {lens}; using the shortest.")
        self.n_frames = min(lens)
        self.sizes = [(d.width, d.height) for d in self.decs]

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
                yield [b[j] for b in batches]
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
