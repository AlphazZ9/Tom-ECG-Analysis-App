# -*- coding: utf-8 -*-
"""
ecg.core.delineation_pool
-------------------------
Per-row delineation of ``analyse_intervals``, in worker processes on long
recordings. Kept apart from ecg.core.analysis so a worker imports numpy and
the wave template only (not NeuroKit, pandas or SciPy's interpolation).

Each row's P search starts at the previous row's T peak, so the rows form one
chain. They are cut into contiguous chunks; a worker starts its chunk
``WARMUP`` rows early to rebuild that chain, and the parent checks every
joint: when the T peak a chunk started from differs from the true one, the
rows after the joint are redone in order until they meet the worker's again.
The result is the sequential one, row for row (tests/core/
test_delineation_pool.py).

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import atexit
import logging
import os
from typing import Callable, Optional

import numpy as np

from ecg.core.wave_template import detect_waves_on_beat

log = logging.getLogger("ecg")

LANDMARKS = ("P_onset", "P_peak", "QRS_onset", "Q_peak", "S_peak", "QRS_offset",
             "J_peak", "T_peak", "T_offset")
PARALLEL_MIN_ROWS = 1500     # below, handing the rows out costs more than it saves
WARMUP = 6
_SHARED = ("cleaned", "beats_m")    # the big arrays, passed in shared memory
_POOL = None
_POOL_WORKERS = 0


class DelinCtx:
    """What delineating a row needs. Picklable; the big arrays (_SHARED)
    travel in shared memory, the rest with each chunk."""

    def __init__(self, **kw) -> None:
        self.ens_on = False
        self.__dict__.update(kw)


def nan_median_rows(block: "np.ndarray") -> "np.ndarray":
    """Median of each column ignoring NaN: faster than np.nanmedian on the
    small blocks delineation asks for tens of thousands of times."""
    s = np.sort(block, axis=0)            # NaN sorts last
    k = (~np.isnan(block)).sum(axis=0)
    c = np.arange(block.shape[1])
    lo, hi = np.maximum((k - 1) // 2, 0), np.maximum(k // 2, 0)
    return 0.5 * (s[lo, c].astype(float) + s[hi, c].astype(float))


def ensemble_members(i: int, ens_joins: "np.ndarray", valid_rp: "np.ndarray", n_ens: int,
                     ens_min: int, ens_reach: float) -> "Optional[np.ndarray]":
    """Ensemble of row *i* (a beat that joins ensembles): the joining beats
    among the N centred on it and within reach in time; None when fewer than
    ens_min."""
    n_valid = len(valid_rp)
    j0, j1 = max(0, i - n_ens // 2), min(n_valid, i + n_ens - n_ens // 2)
    m = j0 + np.flatnonzero(
        ens_joins[j0:j1] & (np.abs(valid_rp[j0:j1] - valid_rp[i]) <= ens_reach))
    return m if len(m) >= ens_min else None


def delineate_rows(c: DelinCtx, start: int, stop: int,
                   prev_t: float = np.nan) -> "dict[str, np.ndarray]":
    """Landmarks (samples) of rows start..stop-1, delineated in order; *prev_t*
    is the T peak of row start-1 (NaN: unknown or none)."""
    n = stop - start
    out = {k: np.full(n, np.nan) for k in LANDMARKS}
    fs, valid_rp, rpeaks_all = c.fs, c.valid_rp, c.rpeaks_all
    for k in range(n):
        i = start + k
        r0 = int(valid_rp[i])
        lo, hi = int(c.pre_v[i]), int(c.post_v[i])
        bt_ms = np.arange(-lo, hi) / fs * 1000
        t_prev = out["T_peak"][k - 1] if k > 0 else prev_t
        p_from = None
        if (i > 0 and c.pos_v[i] > 0 and valid_rp[i - 1] == rpeaks_all[c.pos_v[i] - 1]
                and r0 - valid_rp[i - 1] >= c.gap_min and np.isfinite(t_prev)):
            p_from = (t_prev - r0) / fs * 1000
        next_r = ((rpeaks_all[c.i_next[i]] - r0) / fs * 1000
                  if c.i_next[i] < len(rpeaks_all) else None)
        beat, on_median = None, False
        if c.ens_on:
            if c.ens_artifact[i]:
                continue
            if c.ens_joins[i]:
                members = ensemble_members(i, c.ens_joins, valid_rp, c.n_ens, c.ens_min,
                                           c.ens_reach)
                if members is not None:
                    beat = nan_median_rows(
                        c.beats_m[members, c.fixed_hw - lo:c.fixed_hw + hi])
                    on_median = True
        if beat is None:
            beat = c.cleaned[r0 - lo:r0 + hi].astype(float)
        det = detect_waves_on_beat(beat, bt_ms,
                                   c.template, p_search_from_ms=p_from,
                                   next_r_ms=next_r, p_onset_biphasic=on_median,
                                   qrs_initial_plan=(c.qrs_plans[i // c.q_step]
                                                     if on_median else None),
                                   t_phase_noise=c.t_noise if on_median else None,
                                   t_plan=c.t_plans[i // c.t_step] if on_median else None)
        for name in LANDMARKS:
            v = det.get(name, np.nan)
            out[name][k] = valid_rp[i] + v / 1000.0 * fs if np.isfinite(v) else np.nan
    return out


# ── worker side ──────────────────────────────────────────────────────────

_ATTACHED: "dict[int, tuple[list, dict]]" = {}     # ctx_id -> (shm handles, arrays)


def _attach(c: DelinCtx) -> DelinCtx:
    """*c* with its shared arrays mapped in (once per recording per worker)."""
    from multiprocessing import shared_memory
    if c.ctx_id not in _ATTACHED:
        for old_handles, _ in _ATTACHED.values():   # an earlier recording's
            for h in old_handles:
                h.close()
        _ATTACHED.clear()
        handles, arrays = [], {}
        for name, (shm_name, shape, dtype) in c.shm_specs.items():
            h = shared_memory.SharedMemory(name=shm_name)
            handles.append(h)
            arrays[name] = np.ndarray(shape, dtype=dtype, buffer=h.buf)
        _ATTACHED[c.ctx_id] = (handles, arrays)
    for name, arr in _ATTACHED[c.ctx_id][1].items():
        setattr(c, name, arr)
    return c


def _worker_chunk(job: "tuple[DelinCtx, int, int, int]") -> "tuple[int, dict, float]":
    """Rows i0..i1-1, the chain rebuilt from row s. Returns i0, their landmarks
    and the T peak row i0 started from."""
    c, s, i0, i1 = job
    out = delineate_rows(_attach(c), s, i1)
    k = i0 - s
    start_t = out["T_peak"][k - 1] if k > 0 else np.nan
    return i0, {name: v[k:] for name, v in out.items()}, float(start_t)


# ── parent side ──────────────────────────────────────────────────────────

def workers() -> int:
    return max(1, min((os.cpu_count() or 1) - 2, 12))


def _pool(n: int):
    """One pool for the session: starting processes costs about a second."""
    global _POOL, _POOL_WORKERS
    if _POOL is None or _POOL_WORKERS != n:
        import concurrent.futures as cf
        import multiprocessing as mp
        shutdown_pool()
        _POOL = cf.ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))
        _POOL_WORKERS = n
    return _POOL


def shutdown_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


atexit.register(shutdown_pool)


def delineate_all(c: DelinCtx, progress: "Callable[[int], None]") -> "dict[str, np.ndarray]":
    """Every row's landmarks, as delineate_rows(c, 0, n) gives them."""
    n = len(c.valid_rp)
    w = workers()
    if n < PARALLEL_MIN_ROWS or w < 2 or os.environ.get("ECG_DELIN_SERIAL"):
        return _delineate_serial(c, n, progress)
    try:
        return _delineate_parallel(c, n, w, progress)
    except Exception as exc:          # no worker processes here: the same rows, in order
        log.warning("Parallel delineation unavailable (%s): delineating in order", exc)
        shutdown_pool()
        return _delineate_serial(c, n, progress)


def _delineate_serial(c: DelinCtx, n: int, progress) -> "dict[str, np.ndarray]":
    out = {k: np.full(n, np.nan) for k in LANDMARKS}
    step = max(1, n // 20)
    for a in range(0, n, step):
        b = min(n, a + step)
        part = delineate_rows(c, a, b, out["T_peak"][a - 1] if a else np.nan)
        for k in LANDMARKS:
            out[k][a:b] = part[k]
        progress(b)
    return out


def _delineate_parallel(c: DelinCtx, n: int, w: int, progress) -> "dict[str, np.ndarray]":
    from multiprocessing import shared_memory
    shms = []
    try:
        specs, light = {}, DelinCtx(**{k: v for k, v in c.__dict__.items() if k not in _SHARED})
        for name in _SHARED:
            arr = getattr(c, name, None)
            if arr is None:
                continue
            arr = np.ascontiguousarray(arr)
            h = shared_memory.SharedMemory(create=True, size=max(1, arr.nbytes))
            shms.append(h)
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=h.buf)[...] = arr
            specs[name] = (h.name, arr.shape, arr.dtype.str)
        light.shm_specs = specs
        light.ctx_id = id(c) ^ (os.getpid() << 20) ^ n
        n_chunks = max(2, min(n // 50, 4 * w))
        bounds = np.linspace(0, n, n_chunks + 1).astype(int)
        jobs = [(light, max(0, int(a) - WARMUP), int(a), int(b))
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        out = {k: np.full(n, np.nan) for k in LANDMARKS}
        started_from = {}
        done = 0
        for i0, part, start_t in _pool(w).map(_worker_chunk, jobs):
            i1 = i0 + len(part["T_peak"])
            for k in LANDMARKS:
                out[k][i0:i1] = part[k]
            started_from[i0] = start_t
            done += i1 - i0
            progress(done)
    finally:
        for h in shms:
            h.close()
            try:
                h.unlink()
            except FileNotFoundError:
                pass
    # The joints, in order: a chunk that started from another T peak than the
    # true one is redone from its first row until it meets the worker's rows.
    redone = 0
    for _c, _s, i0, _i1 in jobs:
        if i0 == 0 or _same(out["T_peak"][i0 - 1], started_from[i0]):
            continue
        i, prev = i0, out["T_peak"][i0 - 1]
        while i < n:
            row = delineate_rows(c, i, i + 1, prev)
            same = all(_same(row[k][0], out[k][i]) for k in LANDMARKS)
            for k in LANDMARKS:
                out[k][i] = row[k][0]
            redone += 1
            prev, i = row["T_peak"][0], i + 1
            if same:
                break
    if redone:
        log.debug("parallel delineation: %d row(s) redone at chunk joints", redone)
    return out


def _same(a: float, b: float) -> bool:
    return bool((np.isnan(a) and np.isnan(b)) or a == b)
