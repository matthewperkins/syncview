"""Open Ephys binary-format loading and video-frame <-> sample synchronisation."""
import json
import warnings
from pathlib import Path

import numpy as np


class OERecording:
    """Memory-mapped access to one Open Ephys binary-format recording (…/experimentN/recordingM)."""

    def __init__(self, rec_dir, stream="acquisition_board"):
        self.rec_dir = Path(rec_dir).resolve()
        meta = json.loads((self.rec_dir / "structure.oebin").read_text())
        streams = [c["stream_name"] for c in meta["continuous"]]
        if stream not in streams:
            raise ValueError(f"No continuous stream {stream!r} in {self.rec_dir}; available: {', '.join(streams)}")
        cont = meta["continuous"][streams.index(stream)]
        self.stream = stream
        self.fs = float(cont["sample_rate"])
        self.ch_names = [c["channel_name"] for c in cont["channels"]]
        self.bit_volts = np.array([c["bit_volts"] for c in cont["channels"]])
        self.units = [c["units"] for c in cont["channels"]]
        self.ch_types = [c.get("type") for c in cont["channels"]]   # Open Ephys: 0 ephys, 1 aux, 2 adc
        folder = self.rec_dir / "continuous" / cont["folder_name"]
        self.data = np.memmap(folder / "continuous.dat", dtype="<i2", mode="r").reshape(-1, len(self.ch_names))
        sn = np.load(folder / "sample_numbers.npy", mmap_mode="r")
        self.first_sample = int(sn[0])
        self.contiguous = int(sn[-1]) - self.first_sample + 1 == len(sn)
        if not self.contiguous:
            warnings.warn("Continuous sample numbers are not contiguous (dropped samples?) - sync may be off.")

        ttl = next((e for e in meta["events"]
                    if e.get("stream_name") == stream and e["folder_name"].rstrip("/").endswith("TTL")), None)
        if ttl is None:
            warnings.warn(f"No TTL events for stream {stream!r} - video sync is impossible for this recording.")
            self.ttl_states = np.zeros(0, np.int64)
            self.ttl_index = np.zeros(0, np.int64)
        else:
            tdir = self.rec_dir / "events" / ttl["folder_name"]
            self.ttl_states = np.load(tdir / "states.npy")
            # TTL sample numbers -> row index into continuous.dat
            self.ttl_index = np.load(tdir / "sample_numbers.npy").astype(np.int64) - self.first_sample

    @property
    def n_samples(self):
        return self.data.shape[0]

    @property
    def duration(self):
        return self.n_samples / self.fs

    def ch(self, name):
        return self.ch_names.index(name)

    def rising_edges(self, line=1):
        return self.ttl_index[self.ttl_states == line]

    def edge_counts(self):
        """{TTL line: number of rising edges} for every line with activity."""
        lines, counts = np.unique(self.ttl_states[self.ttl_states > 0], return_counts=True)
        return {int(l): int(c) for l, c in zip(lines, counts)}

    def read(self, i0, i1):
        """Rows i0:i1 of all channels (int16), clipped to the recording. Returns (rows, first_row_index)."""
        a, b = max(int(i0), 0), min(int(i1), self.n_samples)
        return np.asarray(self.data[a:b]), a

    def trace(self, raw, spec):
        """Physical-unit float64 trace of spec['ch'] (minus spec['ref'] if given) from a block of raw rows."""
        c = self.ch(spec["ch"])
        x = raw[:, c].astype(np.float64) * self.bit_volts[c]
        if spec.get("ref"):
            r = self.ch(spec["ref"])
            x -= raw[:, r].astype(np.float64) * self.bit_volts[r]
        return x


class SyncError(ValueError):
    """The video cannot be aligned to this recording's triggers."""


def check_sync(rec, n_frames, line=1, video_duration=None):
    """Align video frames to trigger edges and sanity-check the result.

    Frame k <-> k-th rising edge on `line`. Extra triggers (e.g. the one sent to flush the
    MANTA gstreamer EOS) are expected at the END and are dropped.
    Returns (row index of each frame's trigger, list of human-readable warnings, info dict).
    Raises SyncError when alignment is impossible (no / too few triggers)."""
    edges = rec.rising_edges(line)
    counts = rec.edge_counts()
    others = ", ".join(f"line {l}: {c}" for l, c in counts.items() if l != line) or "none"
    if len(edges) == 0:
        raise SyncError(f"No triggers on TTL line {line}. Rising edges on other lines: {others}. "
                        f"Pick the camera trigger line with --trigger-line.")
    extra = len(edges) - n_frames
    if extra < 0:
        hint = [l for l, c in counts.items() if l != line and 0 <= c - n_frames <= 2]
        raise SyncError(f"Fewer triggers ({len(edges)}) on line {line} than video frames ({n_frames}) - "
                        f"wrong recording/video pair or trigger line?"
                        + (f" Line(s) {hint} have a matching count." if hint else f" Other lines: {others}."))
    fr = edges[:n_frames]
    d = np.diff(fr)
    med = float(np.median(d)) if len(d) else float("nan")
    issues = []
    if extra > 1:
        issues.append(f"{extra} extra triggers after the last frame (expected 0 or 1): frames may be missing "
                      f"from the video, which shifts the alignment.")
    if extra >= 1 and edges[n_frames] - fr[-1] < 1.5 * med:
        issues.append("The first unused trigger follows the last frame by only one frame period, so it looks "
                      "like a real frame rather than an end-of-recording trigger: the video may have lost a frame.")
    n_long, n_short = int(np.sum(d > 1.5 * med)), int(np.sum(d < 0.5 * med))
    if n_long:
        issues.append(f"{n_long} trigger interval(s) > 1.5x the frame period inside the video (missed triggers or "
                      f"paused camera?) - frames around them may be misaligned.")
    if n_short:
        issues.append(f"{n_short} trigger interval(s) < 0.5x the frame period (double or spurious triggers?) - "
                      f"alignment after them is likely off.")
    span = (fr[-1] - fr[0]) / rec.fs
    if video_duration and span > 0 and abs(video_duration - span) > max(0.01 * span, 2.0):
        issues.append(f"Video duration {video_duration:.1f} s differs from the trigger span {span:.1f} s by more "
                      f"than 1% - check that this is the right video and trigger line.")
    if not rec.contiguous:
        issues.append("The recording's sample numbers have gaps (dropped samples), so trigger times may be off.")
    info = dict(n_triggers=len(edges), extra=extra, period=med / rec.fs, span=span)
    return fr, issues, info


def frame_sample_indices(rec, n_frames, line=1, verbose=True, video_duration=None):
    """Continuous-data row index of the trigger for each video frame (see check_sync); problems are warned."""
    fr, issues, info = check_sync(rec, n_frames, line, video_duration)
    if verbose:
        d = np.diff(fr)
        print(f"{info['n_triggers']} triggers on line {line}, {n_frames} frames -> "
              f"{info['extra']} extra trigger(s) dropped at end")
        print(f"frame period {info['period'] * 1e3:.2f} ms ({1 / info['period']:.2f} fps); "
              f"intervals min/max {d.min()}/{d.max()} samples")
        if info["extra"]:
            print(f"gap before the dropped trigger(s): {(rec.rising_edges(line)[n_frames] - fr[-1]) / rec.fs:.2f} s")
        print(f"video spans data {fr[0] / rec.fs:.2f} s -> {fr[-1] / rec.fs:.2f} s "
              f"(recording is {rec.n_samples / rec.fs:.1f} s)")
    for msg in issues:
        warnings.warn(msg)
    return fr
