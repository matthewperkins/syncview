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
        cont = next(c for c in meta["continuous"] if c["stream_name"] == stream)
        self.fs = float(cont["sample_rate"])
        self.ch_names = [c["channel_name"] for c in cont["channels"]]
        self.bit_volts = np.array([c["bit_volts"] for c in cont["channels"]])
        self.units = [c["units"] for c in cont["channels"]]
        folder = self.rec_dir / "continuous" / cont["folder_name"]
        self.data = np.memmap(folder / "continuous.dat", dtype="<i2", mode="r").reshape(-1, len(self.ch_names))
        sn = np.load(folder / "sample_numbers.npy", mmap_mode="r")
        self.first_sample = int(sn[0])
        if int(sn[-1]) - self.first_sample + 1 != len(sn):
            warnings.warn("Continuous sample numbers are not contiguous (dropped samples?) - sync may be off.")

        ttl = next(e for e in meta["events"]
                   if e.get("stream_name") == stream and e["folder_name"].rstrip("/").endswith("TTL"))
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


def frame_sample_indices(rec, n_frames, line=1, verbose=True):
    """Continuous-data row index of the trigger for each video frame.

    Frame k <-> k-th rising edge on `line`. Extra triggers (e.g. the one sent to flush the
    MANTA gstreamer EOS) are expected at the END and are dropped."""
    edges = rec.rising_edges(line)
    extra = len(edges) - n_frames
    if extra < 0:
        raise ValueError(f"Fewer triggers ({len(edges)}) than video frames ({n_frames}) - wrong recording/video pair?")
    fr = edges[:n_frames]
    d = np.diff(fr)
    med = np.median(d)
    if verbose:
        print(f"{len(edges)} triggers on line {line}, {n_frames} frames -> {extra} extra trigger(s) dropped at end")
        print(f"frame period {med / rec.fs * 1e3:.2f} ms ({rec.fs / med:.2f} fps); "
              f"intervals min/max {d.min()}/{d.max()} samples")
        if extra:
            print(f"gap before the dropped trigger(s): {(edges[n_frames] - fr[-1]) / rec.fs:.2f} s")
        print(f"video spans data {fr[0] / rec.fs:.2f} s -> {fr[-1] / rec.fs:.2f} s "
              f"(recording is {rec.n_samples / rec.fs:.1f} s)")
    if extra > 1:
        warnings.warn(f"{extra} extra triggers (expected 0 or 1) - check alignment!")
    if np.any(d > 1.5 * med):
        warnings.warn(f"{np.sum(d > 1.5 * med)} trigger interval(s) > 1.5x nominal inside the video - missed frames?")
    return fr
