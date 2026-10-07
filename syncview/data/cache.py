"""Whole-session processed traces on disk + a min/max pyramid for constant-time drawing at any zoom.

Layout:  <root>/<spec_key>/
            level0.npy          processed trace, float32, sample k <-> absolute row k*step
            min{k}.npy max{k}.npy   min/max over blocks of FACTOR**k level-0 samples (k >= 1)
            meta.json           written last; its presence marks a complete trace
"""
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..core.filters import out_step, pad_samples, process, spec_key

FACTOR = 8
MMAP_MIN_BYTES = 2 << 20  # smaller pyramid levels are read into memory: every memory map keeps a file
                          # open, and macOS allows only 256 open files per process by default
CHUNK_S = 1200.0          # seconds of data processed per chunk (plus filter padding on each side)


class Cancelled(Exception):
    pass


class PyramidTrace:
    """A processed trace (sample k at time t0 + k*dt) plus its min/max pyramid."""

    def __init__(self, y, dt, t0=0.0, mins=None, maxs=None, stats=None):
        self.y, self.dt, self.t0 = y, dt, t0
        if mins is None:
            mins, maxs = [y], [y]
            mn = mx = np.asarray(y, dtype=np.float32)
            while len(mn) // FACTOR >= 256:
                mn, mx = _reduce(mn, mx)
                mins.append(mn)
                maxs.append(mx)
        self.mins, self.maxs = mins, maxs
        self.meta = dict(stats=stats or _stats(np.asarray(y)))

    @classmethod
    def from_samples(cls, t, y):
        """In-memory trace from evenly sampled (t, y), e.g. a window processed on demand."""
        dt = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 1.0
        return cls(np.asarray(y, dtype=np.float32), dt, float(t[0]))

    @property
    def stats(self):
        return self.meta["stats"]

    def view(self, t_lo, t_hi, n_px):
        """Points to draw for [t_lo, t_hi] on n_px pixel columns.

        Few samples per pixel -> the samples themselves. Otherwise min and max per pixel column
        (interleaved lo,hi,lo,hi…), taken from the coarsest pyramid level that still has >= 1 block
        per column. Columns sit on a grid anchored at t=0, so panning doesn't make them flicker."""
        n_px = max(int(n_px), 16)
        off = self.t0
        t_lo, t_hi = t_lo - off, t_hi - off
        span = t_hi - t_lo
        per_px = span / self.dt / n_px
        if per_px < 2:
            a = max(int(np.floor(t_lo / self.dt)) - 1, 0)
            b = min(int(np.ceil(t_hi / self.dt)) + 2, len(self.y))
            return off + np.arange(a, b) * self.dt, np.asarray(self.y[a:b], dtype=np.float32)
        k = min(int(np.log(per_px) / np.log(FACTOR)), len(self.mins) - 1)
        blk = self.dt * FACTOR ** k                       # seconds per block at level k
        mn, mx = self.mins[k], self.maxs[k]
        w = span / n_px                                   # seconds per pixel column
        m0, m1 = int(np.floor(t_lo / w)) - 1, int(np.ceil(t_hi / w)) + 1
        edges = np.round(np.arange(m0, m1 + 1) * w / blk).astype(np.int64)
        edges = np.clip(edges, 0, len(mn))
        good = np.diff(edges) > 0
        if not good.any():
            return np.empty(0), np.empty(0)
        a, b = edges[0], edges[-1]
        lo = np.minimum.reduceat(np.asarray(mn[a:b]), edges[:-1][good] - a) if b > a else np.empty(0)
        hi = np.maximum.reduceat(np.asarray(mx[a:b]), edges[:-1][good] - a) if b > a else np.empty(0)
        t = off + (np.arange(m0, m1)[good] + 0.5) * w
        tt = np.repeat(t, 2)
        yy = np.empty(2 * len(lo), dtype=np.float32)
        yy[0::2], yy[1::2] = lo, hi
        return tt, yy


class Trace(PyramidTrace):
    """A finished, memory-mapped processed trace from the cache."""

    def __init__(self, folder):
        self.folder = Path(folder)
        meta = json.loads((self.folder / "meta.json").read_text())
        y = _load(self.folder / "level0.npy")
        mins, maxs = [y], [y]
        for k in range(1, meta["levels"] + 1):
            mins.append(_load(self.folder / f"min{k}.npy"))
            maxs.append(_load(self.folder / f"max{k}.npy"))
        super().__init__(y, meta["dt"], 0.0, mins, maxs)
        self.meta = meta


class TraceCache:
    """Builds and serves whole-session processed traces for one recording."""

    def __init__(self, rec, root):
        self.rec = rec
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open = {}
        self._lock = threading.Lock()

    def key(self, spec):
        return spec_key(spec, self.rec.rec_dir)

    def get(self, spec):
        """The finished Trace for spec, or None if it hasn't been built."""
        key = self.key(spec)
        with self._lock:
            if key in self._open:
                return self._open[key]
            if (self.root / key / "meta.json").exists():
                self._open[key] = Trace(self.root / key)
                return self._open[key]
        return None

    def retain(self, specs):
        """Close every open trace except those of specs (frees their files and memory maps)."""
        keep = {self.key(s) for s in specs}
        with self._lock:
            self._open = {k: v for k, v in self._open.items() if k in keep}

    def missing(self, specs):
        seen, out = set(), []
        for s in specs:
            k = self.key(s)
            if k not in seen and self.get(s) is None:
                seen.add(k)
                out.append(s)
        return out

    def build(self, specs, progress=None, cancel=None, workers=8):
        """Process every not-yet-cached spec over the whole recording.

        Raw data is read once per chunk and shared by all specs. progress(fraction) is called
        after each chunk; cancel() returning True aborts (partial results are removed)."""
        specs = self.missing(specs)
        if not specs:
            return
        rec, fs = self.rec, self.rec.fs
        steps = [out_step(s, fs) for s in specs]
        align = int(np.lcm.reduce(steps))
        chunk = max(align, int(CHUNK_S * fs) // align * align)
        pad = max(pad_samples(s, fs) for s in specs)
        n = rec.n_samples
        tmp = [self.root / f"{self.key(s)}.building" for s in specs]
        outs = []
        for s, st, d in zip(specs, steps, tmp):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            outs.append(np.lib.format.open_memmap(d / "level0.npy", "w+", np.float32, ((n + st - 1) // st,)))
        try:
            with ThreadPoolExecutor(workers) as ex:
                for c0 in range(0, n, chunk):
                    if cancel and cancel():
                        raise Cancelled
                    c1 = min(c0 + chunk, n)
                    raw, first = rec.read(c0 - pad, c1 + pad)

                    def run(i):
                        y, a, st = process(rec.trace(raw, specs[i]), first, fs, specs[i])
                        k0 = (c0 - a) // st                     # c0 is a multiple of st
                        k1 = k0 + (c1 - c0 + st - 1) // st
                        seg = y[k0:k1]
                        outs[i][c0 // st: c0 // st + len(seg)] = seg
                    list(ex.map(run, range(len(specs))))
                    if progress:
                        progress(c1 / n)
            for s, st, d, y in zip(specs, steps, tmp, outs):
                if cancel and cancel():
                    raise Cancelled
                y.flush()
                levels = _build_pyramid(np.asarray(y), d)
                pct = _stats(y)
                meta = dict(spec={k: v for k, v in s.items() if not k.startswith("_")}, step=st, dt=st / fs,
                            n=len(y), levels=levels, stats=pct, built=time.ctime())
                (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
                del y
                final = self.root / self.key(s)
                shutil.rmtree(final, ignore_errors=True)
                d.rename(final)
        except BaseException:
            for d in tmp:
                shutil.rmtree(d, ignore_errors=True)
            raise


def _load(path):
    return np.load(path, mmap_mode="r" if path.stat().st_size >= MMAP_MIN_BYTES else None)


def _reduce(mn, mx):
    m = len(mn) // FACTOR * FACTOR
    new_mn = mn[:m].reshape(-1, FACTOR).min(axis=1)
    new_mx = mx[:m].reshape(-1, FACTOR).max(axis=1)
    if len(mn) > m:   # keep the ragged end as one partial block
        new_mn = np.append(new_mn, mn[m:].min())
        new_mx = np.append(new_mx, mx[m:].max())
    return new_mn.astype(np.float32), new_mx.astype(np.float32)


def _build_pyramid(y, folder, min_len=256):
    mn = mx = y
    k = 0
    while len(mn) // FACTOR >= min_len:
        mn, mx = _reduce(mn, mx)
        k += 1
        np.save(folder / f"min{k}.npy", mn)
        np.save(folder / f"max{k}.npy", mx)
    return k


def _stats(y):
    sub = np.asarray(y[:: max(1, len(y) // 2_000_000)], dtype=np.float64)
    sub = sub[np.isfinite(sub)]
    if not len(sub):
        return {}
    qs = (0.1, 0.5, 1, 50, 99, 99.5, 99.9)
    out = {str(q): float(v) for q, v in zip(qs, np.percentile(sub, qs))}
    out["absmax99.9"] = float(np.percentile(np.abs(sub), 99.9))
    return out
