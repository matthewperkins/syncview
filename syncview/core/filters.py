"""Channel processing modes, shared by the clip renderer and the viewer's cache.

A channel *spec* is a dict: {"ch": "CH6", "mode": "hilo", "label": ..., "ref": ..., <filter overrides>}.
All filters are zero-phase (sosfiltfilt / centred moving average), so nothing is shifted in time.

Every mode maps a raw trace to an output sampled on a *global* grid: absolute row indices that are
multiples of the mode's decimation step. That makes results from different windows/chunks line up exactly.
"""
import hashlib
import json

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

MODE_DEFAULTS = {
    # slow GI signals: band-pass, plotted at plot_fs. Low edge 0.03 Hz: the slow-wave peaks sit at
    # 0.06-0.11 Hz in these recordings and the headstage's analog high-pass is ~0.094 Hz (1st order).
    "slow":      dict(band=(0.03, 100.0), order=2, notch=None, plot_fs=1000.0, pad_s=120.0),
    # EMG, high-passed, drawn as per-pixel-column min/max ("hi-lo") so peaks are never lost
    "hilo":      dict(band=(150.0, 4000.0), order=4, notch=None, pad_s=1.0),
    # EMG envelope: band-pass -> rectify -> centred boxcar -> low-pass
    "envelope":  dict(band=(150.0, 4000.0), order=4, notch=None, smooth_ms=20.0, env_lp=40.0,
                      plot_fs=1000.0, pad_s=1.0),
    # slow-wave band power (RMS in a centred moving window) - shows rhythm coming and going over minutes
    "bandpower": dict(band=(0.03, 0.25), order=2, notch=None, win_s=30.0, plot_fs=10.0, pad_s=240.0),
}
MODES = list(MODE_DEFAULTS)


def params(spec):
    """Mode defaults updated with any overrides present in the spec."""
    p = dict(MODE_DEFAULTS[spec["mode"]])
    p.update({k: spec[k] for k in p if k in spec and spec[k] is not None})
    p["band"] = tuple(p["band"])
    return p


def spec_key(spec, rec_dir=""):
    """Stable hash of everything that affects the processed trace (not label/colour/ylim)."""
    d = dict(rec=str(rec_dir), ch=spec["ch"], ref=spec.get("ref") or None, mode=spec["mode"], **params(spec))
    d.pop("pad_s")
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def out_step(spec, fs):
    """Decimation step (in raw samples) of the mode's output."""
    p = params(spec)
    return max(1, int(round(fs / p["plot_fs"]))) if "plot_fs" in p else 1


def pad_samples(spec, fs):
    return int(params(spec)["pad_s"] * fs)


def _sos(order, band, fs):
    lo, hi = band
    lo = lo if lo else None
    hi = hi if hi and hi < 0.49 * fs else None
    if lo and hi:
        return signal.butter(order, [lo, hi], "bandpass", fs=fs, output="sos")
    if lo:
        return signal.butter(order, lo, "highpass", fs=fs, output="sos")
    if hi:
        return signal.butter(order, hi, "lowpass", fs=fs, output="sos")
    return None


def _filt(x, fs, band, order):
    sos = _sos(order, band, fs)
    return x if sos is None else signal.sosfiltfilt(sos, x)


def _decimate(y, a, step, factor):
    """y sampled at absolute indices a + k*step. Keep only samples on the global grid of step*factor.
    (Caller has already low-passed y.) Returns (y', a', step')."""
    new = step * factor
    k0 = (-a % new) // step if (-a % new) % step == 0 else None
    if k0 is None:
        raise ValueError("misaligned decimation")
    return y[k0::factor], a + k0 * step, new


def process(x, a, fs, spec):
    """Apply the spec's mode to raw trace x (physical units) whose first sample is absolute row a.

    Returns (y, a_out, step): y[k] belongs to absolute row a_out + k*step, with a_out % step == 0."""
    p = params(spec)
    mode = spec["mode"]
    x = x - np.median(x[:: max(1, len(x) // 10000)])
    if p.get("notch"):
        for f0 in np.atleast_1d(p["notch"]):
            b, aa = signal.iirnotch(float(f0), 30.0, fs)
            x = signal.filtfilt(b, aa, x)

    if mode == "hilo":
        return _filt(x, fs, p["band"], p["order"]), a, 1

    if mode == "envelope":
        y = _filt(x, fs, p["band"], p["order"])
        n = max(1, int(round(p["smooth_ms"] * 1e-3 * fs)) | 1)     # odd length -> centred boxcar
        y = uniform_filter1d(np.abs(y), n)
        y = _filt(y, fs, (None, p["env_lp"]), 4)
        return _decimate(y, a, 1, out_step(spec, fs))

    if mode == "slow":
        lo, hi = p["band"]
        y = _filt(x, fs, (None, hi), p["order"])                    # low-pass at full rate
        y, a, step = _decimate(y, a, 1, out_step(spec, fs))
        y = _filt(y, fs / step, (lo, None), p["order"])             # high-pass at the plot rate
        return y, a, step

    if mode == "bandpower":
        # two-stage decimation to plot_fs, then band-pass, square, centred moving mean, sqrt
        total = out_step(spec, fs)
        f1 = 10 if total % 10 == 0 and total > 10 else 1
        y = _filt(x, fs, (None, 0.4 * fs / f1), 4) if f1 > 1 else x
        y, a, step = _decimate(y, a, 1, f1)
        f2 = total // step
        y = _filt(y, fs / step, (None, 0.4 * fs / total), 4)
        y, a, step = _decimate(y, a, step, f2)
        y = _filt(y, fs / step, p["band"], p["order"])
        n = max(1, int(round(p["win_s"] * fs / step)) | 1)
        y = np.sqrt(uniform_filter1d(y * y, n))
        return y, a, step

    raise ValueError(f"unknown mode {mode!r} (choose from {MODES})")


def process_window(rec, spec, i0, i1, raw=None, raw_first=None):
    """Processed output covering absolute rows [i0, i1) (with filter padding read around it).

    Returns (t, y): times in seconds since recording start, and values."""
    pad = pad_samples(spec, rec.fs)
    if raw is None:
        raw, raw_first = rec.read(i0 - pad, i1 + pad)
    y, a, step = process(rec.trace(raw, spec), raw_first, rec.fs, spec)
    idx = a + np.arange(len(y)) * step
    k0, k1 = np.searchsorted(idx, [i0, i1])
    return idx[k0:k1] / rec.fs, y[k0:k1]
