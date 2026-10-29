"""
ecg.core.analysis
─────────────────
HRV analysis: time-domain, frequency-domain, nonlinear metrics, interval
delineation.  All functions are pure NumPy/SciPy and safe to call from
background threads.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, Optional, cast

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import welch as _scipy_welch

# numpy 2.x renamed trapz to trapezoid (older numpy/some builds only have
# trapz). This used to rely on app.py's own module-level shim monkey-patching
# np.trapz before `from analysis import ...` ran -- which meant this module
# could not be imported on its own on a numpy version lacking trapz unless
# app.py had already run first. Resolved locally instead, with no dependency
# on import order, so this module stays importable/testable standalone.
trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid", None)
if trapz is None:
    from scipy.integrate import trapezoid as trapz  # last-resort fallback

from ecg.core.models import AnalysisResults, MouseECG, normalize_qtc_formula
from ecg.core.wave_template import WaveTemplate, detect_waves_on_beat

# Detected locally rather than imported from theme.py: this module is pure
# NumPy/SciPy/pandas with no Tkinter/CTk dependency, and theme.py does
# `import customtkinter` plus module-level ctk.set_appearance_mode()/
# set_default_color_theme() calls -- importing NK_AVAILABLE/nk from there
# would drag that whole chain in just to check whether neurokit2 is installed.
try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    nk = None
    NK_AVAILABLE = False


log = logging.getLogger("ecg")

POINCARE_MIN_PAIRS = 10


def compute_poincare_metrics(
    rr_ms: np.ndarray,
    valid: "Optional[np.ndarray]" = None,
) -> dict[str, float | int]:
    """Return article-style CCD and robust Poincare SD1/SD2 for RR intervals.

    Poincare points are successive interval pairs (RR[i], RR[i + 1]).
    Invalid intervals are excluded without joining across them: *valid* is
    the per-interval mask to use (e.g. ``rr_interval_series()["valid"]``,
    which also honours user bounds and removed-beat breaks); when omitted,
    intervals outside [RR_MIN_MS, RR_MAX_MS] are invalid. SD1/SD2 are
    square roots of the minor/major eigenvalues of a Ledoit-Wolf shrinkage
    covariance estimate. Metrics are withheld unless at least
    ``POINCARE_MIN_PAIRS`` valid successive pairs remain.
    """
    rr = np.asarray(rr_ms, dtype=float).flatten()
    pairs = np.column_stack((rr[:-1], rr[1:])) if len(rr) > 1 else np.empty((0, 2))
    ok = (rr_valid_mask(rr) if valid is None
          else np.asarray(valid, dtype=bool).flatten() & np.isfinite(rr))
    pairs = pairs[ok[:-1] & ok[1:]] if len(rr) > 1 else pairs
    n_pairs = int(len(pairs))
    result: dict[str, float | int] = {
        "CCD": float("nan"),
        "SD1_LW": float("nan"),
        "SD2_LW": float("nan"),
        "n_poincare_pairs": n_pairs,
    }
    if n_pairs < POINCARE_MIN_PAIRS:
        return result

    # CCD is the distance from the origin to the Poincare ellipse centre.
    center = pairs.mean(axis=0)
    result["CCD"] = float(np.hypot(center[0], center[1]))

    from sklearn.covariance import LedoitWolf

    covariance = LedoitWolf().fit(pairs).covariance_
    eigenvalues = np.linalg.eigvalsh(covariance)
    result["SD1_LW"] = float(np.sqrt(max(float(eigenvalues[0]), 0.0)))
    result["SD2_LW"] = float(np.sqrt(max(float(eigenvalues[-1]), 0.0)))
    return result


def _require_nk() -> None:
    if not NK_AVAILABLE or nk is None:
        raise RuntimeError("neurokit2 is required — pip install neurokit2")


def _validate_sampling_rate(fs: float) -> None:
    """Reject a non-positive or non-finite sampling rate before RR computation."""
    try:
        ok = bool(np.isfinite(fs)) and fs > 0
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"Sampling rate must be positive and finite, got {fs!r}.")


# ════════════════════════════════════════════════════════════
#  GAP-AWARE RR SERIES
# ════════════════════════════════════════════════════════════
#
# HRV is computed on the RR intervals that fall inside the acceptance
# bounds [rr_min_ms, rr_max_ms] -- the mouse physiological range
# [RR_MIN_MS, RR_MAX_MS] by default, user-adjustable in the UI (e.g. a
# wider Max RR for an anaesthetised mouse below 180 bpm). An interval
# outside them (a genuine sinus pause, a missed beat, a residual doublet)
# is EXCLUDED -- never interpolated and never bridged: successive
# differences (RMSSD, pNN6, Poincaré pairs, asymmetry indices) only pair
# two intervals that are both valid AND adjacent in the original series.
#
# An interval that spans a beat removed by artifact correction (a "break")
# is excluded too, whatever its length: removing a premature beat merges
# its coupling interval and compensatory pause into one ~2× interval that
# is not a normal-to-normal (NN) interval in the Task Force sense, so it
# must not enter HRV as if it were one. Only the removal of a REAL beat
# is a break: removing a spurious detection (a duplicate closer than Min
# R-R, a peak splitting one normal interval) gives back the true sinus-to-
# sinus interval, like a peak deleted by hand in edit mode (the user
# asserts it was never a beat) -- see detection._removal_is_break.
#
# The number of excluded intervals, the reason, and the bounds used are
# reported alongside every result so they are disclosed rather than
# silently absorbed. This replaces an earlier all-or-nothing check that
# refused to analyse a recording as soon as a single interval was out of
# range.

def rr_valid_mask(
    rr_ms:     np.ndarray,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
) -> np.ndarray:
    """Bool mask over *rr_ms*: True where the interval is finite and inside
    [rr_min_ms, rr_max_ms] (bounds inclusive)."""
    rr = np.asarray(rr_ms, dtype=float)
    return np.isfinite(rr) & (rr >= rr_min_ms) & (rr <= rr_max_ms)


def successive_valid_pairs(
    rr_ms: np.ndarray,
    valid: "Optional[np.ndarray]" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return (RR[i], RR[i + 1]) for every pair of ADJACENT intervals that are
    both valid -- the only pairs a successive difference may be taken from.

    *valid* defaults to ``rr_valid_mask(rr_ms)``. A pair straddling an
    excluded interval is dropped rather than joined across the gap, so
    ``rr_b - rr_a`` gives the gap-aware successive differences.
    """
    rr = np.asarray(rr_ms, dtype=float).flatten()
    if len(rr) < 2:
        empty = np.array([], dtype=float)
        return empty, empty.copy()
    ok = rr_valid_mask(rr) if valid is None else np.asarray(valid, dtype=bool).flatten()
    adjacent = ok[:-1] & ok[1:]
    return rr[:-1][adjacent], rr[1:][adjacent]


def _intervals_spanning_breaks(peaks: np.ndarray, breaks: "Optional[np.ndarray]") -> np.ndarray:
    """Bool mask over the len(peaks) - 1 intervals of the SORTED *peaks*:
    True where at least one break sample lies strictly inside the interval
    (peaks[i] < b < peaks[i + 1]). A break sitting exactly on a peak -- the
    removed beat was re-detected since -- splits nothing."""
    n_iv = max(len(peaks) - 1, 0)
    if breaks is None or n_iv == 0:
        return np.zeros(n_iv, dtype=bool)
    b = np.asarray(breaks, dtype=np.float64).flatten()
    b = np.sort(b[np.isfinite(b)])
    if len(b) == 0:
        return np.zeros(n_iv, dtype=bool)
    p  = peaks.astype(np.float64)
    lo = np.searchsorted(b, p[:-1], side="right")   # first break >  peaks[i]
    hi = np.searchsorted(b, p[1:],  side="left")    # first break >= peaks[i + 1]
    return hi > lo


def _run_sums(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum of *values* over every run of consecutive True entries of *mask*
    (non-finite values count as 0). Empty array when *mask* has no True."""
    m = np.asarray(mask, dtype=bool).flatten()
    if not m.any():
        return np.array([], dtype=np.float64)
    v  = np.where(np.isfinite(values), values, 0.0).astype(np.float64)
    d  = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    cs = np.r_[0.0, np.cumsum(v)]
    return cs[ends] - cs[starts]


def rr_interval_series(
    rpeaks:    np.ndarray,
    fs:        float,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
    breaks:    "Optional[np.ndarray]" = None,
) -> dict:
    """Split the RR intervals of *rpeaks* into valid and excluded intervals.

    Peaks are sorted first. Raises ValueError for a non-positive or
    non-finite sampling rate.

    *breaks* are the sample positions of real beats removed by artifact
    correction (``apply_artifact_decisions()``'s ``break_samples``). An
    interval with a break strictly inside it spans a removed beat and is
    excluded whatever its length (see the section comment above). Each
    excluded interval is counted under ONE reason: spanning a removed beat
    takes precedence over being out of bounds, so
    ``n_excluded == n_too_short + n_too_long + n_spanning_removed``.

    Returns a dict with:
        rr_ms              : every interval (ms), len = len(rpeaks) - 1
        t_s                : time of each interval's END (rpeaks[1:] / fs), s
        valid              : bool mask over rr_ms (finite, within bounds,
                             not spanning a removed beat)
        rri, rri_time      : rr_ms[valid], t_s[valid]
        n_total            : number of intervals
        n_excluded         : number of excluded intervals
        n_too_short        : ... of which shorter than rr_min_ms
        n_too_long         : ... of which longer than rr_max_ms
        n_spanning_removed : ... of which spanning a removed beat
        excluded_time_s    : summed duration of the excluded intervals, s
        max_excluded_run_s : longest run of CONSECUTIVE excluded intervals, s
        span_s             : analysed span (first to last peak), s
        rr_min_ms, rr_max_ms : the bounds used
    """
    _validate_sampling_rate(fs)
    if not (np.isfinite(rr_min_ms) and np.isfinite(rr_max_ms)
            and 0.0 <= rr_min_ms < rr_max_ms):
        raise ValueError(
            f"RR bounds must satisfy 0 ≤ min < max, got {rr_min_ms!r}–{rr_max_ms!r} ms.")
    peaks = np.sort(np.asarray(rpeaks, dtype=int).flatten())
    if len(peaks) >= 2:
        rr_ms = np.diff(peaks).astype(np.float64) / fs * 1000.0
        t_s   = peaks[1:].astype(np.float64) / fs
    else:
        rr_ms = np.array([], dtype=np.float64)
        t_s   = np.array([], dtype=np.float64)
    finite    = np.isfinite(rr_ms)
    spanning  = finite & _intervals_spanning_breaks(peaks, breaks)
    too_short = finite & ~spanning & (rr_ms < rr_min_ms)
    too_long  = finite & ~spanning & (rr_ms > rr_max_ms)
    valid     = finite & ~spanning & ~too_short & ~too_long
    excluded  = ~valid
    runs_s    = _run_sums(rr_ms, excluded) / 1000.0
    return {
        "rr_ms":              rr_ms,
        "t_s":                t_s,
        "valid":              valid,
        "rri":                rr_ms[valid],
        "rri_time":           t_s[valid],
        "n_total":            int(len(rr_ms)),
        "n_excluded":         int(excluded.sum()),
        "n_too_short":        int(too_short.sum()),
        "n_too_long":         int(too_long.sum()),
        "n_spanning_removed": int(spanning.sum()),
        "excluded_time_s":    float(rr_ms[excluded & finite].sum() / 1000.0),
        "max_excluded_run_s": float(runs_s.max()) if len(runs_s) else 0.0,
        "span_s":             float(rr_ms[finite].sum() / 1000.0),
        "rr_min_ms":          float(rr_min_ms),
        "rr_max_ms":          float(rr_max_ms),
    }


def nk_rri_input(series: dict) -> dict:
    """NeuroKit2 HRV input built from an ``rr_interval_series()`` result.

    nk.hrv_time / nk.hrv_frequency / nk.hrv_nonlinear accept
    ``{"RRI": ms, "RRI_Time": s}``; the timestamps (each interval's end)
    let NeuroKit see where excluded intervals were removed and only take
    successive differences between truly consecutive intervals. With no
    interval excluded the results are identical to passing the peaks.
    """
    return {"RRI": series["rri"], "RRI_Time": series["rri_time"]}


def rr_exclusion_summary(series: dict) -> dict:
    """The exclusion counts of an ``rr_interval_series()`` result and the RR
    bounds used, as stored in ``hr`` and in the ``attrs`` of the HRV
    DataFrames."""
    n_total = series["n_total"]
    return {
        "n_excluded":         series["n_excluded"],
        "n_too_short":        series["n_too_short"],
        "n_too_long":         series["n_too_long"],
        "n_spanning_removed": series.get("n_spanning_removed", 0),
        "excluded_pct":       float(100.0 * series["n_excluded"] / n_total) if n_total else 0.0,
        "excluded_time_s":    series["excluded_time_s"],
        "rr_min_ms":          series.get("rr_min_ms", MouseECG.RR_MIN_MS),
        "rr_max_ms":          series.get("rr_max_ms", MouseECG.RR_MAX_MS),
    }


def rr_bounds_label(src: "Optional[dict]" = None) -> str:
    """'67–333 ms' -- the RR bounds recorded in *src* (an ``hr`` dict, HRV
    ``attrs`` or ``rr_exclusion_summary()``), the MouseECG defaults for keys
    it lacks (e.g. results restored from an older session)."""
    src = src or {}
    lo = float(src.get("rr_min_ms", MouseECG.RR_MIN_MS))
    hi = float(src.get("rr_max_ms", MouseECG.RR_MAX_MS))
    return f"{lo:.0f}–{hi:.0f} ms"


def rr_exclusion_text(src: "Optional[dict]") -> str:
    """Short disclosure of the excluded RR intervals recorded in *src* (same
    sources as ``rr_bounds_label``), '' when none were excluded -- e.g.
    '1 RR excluded (outside 67–333 ms)' or '3 RR excluded (2 outside
    67–333 ms, 1 spanning removed beats)'."""
    src = src or {}
    n = int(src.get("n_excluded", 0) or 0)
    if n <= 0:
        return ""
    n_span   = int(src.get("n_spanning_removed", 0) or 0)
    n_bounds = n - n_span
    bounds   = rr_bounds_label(src)
    if n_span and n_bounds:
        why = f"{n_bounds} outside {bounds}, {n_span} spanning removed beats"
    elif n_span:
        why = "spanning removed beats"
    else:
        why = f"outside {bounds}"
    return f"{n} RR excluded ({why})"


def _rebuild_rr_times(rr: np.ndarray, valid: np.ndarray,
                      rr_df: "Optional[pd.DataFrame]") -> np.ndarray:
    """End time (s) of every interval in *rr* for results lacking ``rr_t_s``
    (sessions saved before it existed): the end of interval i is the first
    peak plus the cumulative RR, anchored on ``rr_df``'s times (all
    intervals, or the valid ones). Relative to the first peak when no time
    is available."""
    cum = np.cumsum(np.where(np.isfinite(rr), rr, 0.0)) / 1000.0
    if rr_df is not None and "Time_s" in getattr(rr_df, "columns", []):
        tv = np.asarray(rr_df["Time_s"], dtype=np.float64).flatten()
        if len(tv) == len(rr):
            return tv
        k = np.flatnonzero(valid)
        if len(tv) and len(tv) == len(k):
            return tv[0] - cum[k[0]] + cum
    return cum


def rr_timeseries(results: "Optional[dict]") -> "pd.DataFrame":
    """Every RR interval of an ``analyse_core`` result, excluded ones included.

    Columns: Time_s (end of the interval, s), RR_ms, HR_bpm and ``valid``
    (bool: the interval was used for HRV -- inside the RR bounds and not
    spanning a removed beat, see rr_interval_series). ``rr_df`` only holds
    the valid intervals; a tachogram or an RR export needs this instead, so
    a pause or a spanning interval is shown/flagged rather than silently
    dropped.

    Times come from ``rr_t_s`` (results from older sessions rebuild them
    from ``rr_df``); ``valid`` falls back to the RR bounds recorded in
    ``hr`` (default MouseECG bounds) when ``rr_valid`` is missing. Empty
    frame (same columns) when there is no ``rr_ms``.
    """
    cols = ["Time_s", "RR_ms", "HR_bpm", "valid"]
    r = results or {}
    rr_raw = r.get("rr_ms")
    rr = (np.asarray(rr_raw, dtype=np.float64).flatten()
          if rr_raw is not None else np.array([], dtype=np.float64))
    if len(rr) == 0:
        return pd.DataFrame({c: pd.Series(dtype=bool if c == "valid" else float)
                             for c in cols})
    valid_raw = r.get("rr_valid")
    if valid_raw is not None and len(np.asarray(valid_raw).flatten()) == len(rr):
        valid = np.asarray(valid_raw, dtype=bool).flatten() & np.isfinite(rr)
    else:
        hr = r.get("hr") or {}
        valid = rr_valid_mask(rr, float(hr.get("rr_min_ms", MouseECG.RR_MIN_MS)),
                              float(hr.get("rr_max_ms", MouseECG.RR_MAX_MS)))
    t_raw = r.get("rr_t_s")
    t = np.asarray(t_raw, dtype=np.float64).flatten() if t_raw is not None else None
    if t is None or len(t) != len(rr):
        t = _rebuild_rr_times(rr, valid, r.get("rr_df"))
    with np.errstate(divide="ignore", invalid="ignore"):
        hr_bpm = 60_000.0 / rr
    hr_bpm = np.where(np.isfinite(hr_bpm), hr_bpm, np.nan)
    return pd.DataFrame({"Time_s": t, "RR_ms": rr, "HR_bpm": hr_bpm, "valid": valid})


def rr_exclusion_report(results: "Optional[dict]") -> dict:
    """How many RR intervals the HRV of *results* used, and why the others
    were excluded -- the numbers every export and report discloses.

    Keys: n_intervals, n_valid, n_excluded, n_too_short, n_too_long,
    n_spanning_removed, excluded_pct, excluded_time_s, rr_min_ms, rr_max_ms
    (the bounds used). Taken from ``hr`` when it carries the counts;
    otherwise (results from older sessions) recomputed from ``rr_ms`` and
    the validity mask, an excluded interval inside the bounds counting as
    spanning a removed beat. Counts are None when neither is available.
    """
    r  = results or {}
    hr = r.get("hr") or {}
    lo = float(hr.get("rr_min_ms", MouseECG.RR_MIN_MS))
    hi = float(hr.get("rr_max_ms", MouseECG.RR_MAX_MS))
    out: dict = {"n_intervals": None, "n_valid": None, "n_excluded": None,
                 "n_too_short": None, "n_too_long": None,
                 "n_spanning_removed": None, "excluded_pct": None,
                 "excluded_time_s": None, "rr_min_ms": lo, "rr_max_ms": hi}
    ts = rr_timeseries(r)
    if "n_excluded" in hr:
        n_ex = int(hr["n_excluded"] or 0)
        n_ok = hr.get("n_valid")
        n_ok = int(n_ok) if n_ok is not None else (int(ts["valid"].sum()) if len(ts) else None)
        out.update({
            "n_valid":            n_ok,
            "n_intervals":        (n_ok + n_ex) if n_ok is not None else None,
            "n_excluded":         n_ex,
            "n_too_short":        int(hr.get("n_too_short", 0) or 0),
            "n_too_long":         int(hr.get("n_too_long", 0) or 0),
            "n_spanning_removed": int(hr.get("n_spanning_removed", 0) or 0),
            "excluded_pct":       float(hr.get("excluded_pct", 0.0) or 0.0),
            "excluded_time_s":    float(hr.get("excluded_time_s", 0.0) or 0.0),
        })
        return out
    if len(ts) == 0:
        return out
    rr    = ts["RR_ms"].to_numpy(dtype=np.float64)
    valid = ts["valid"].to_numpy(dtype=bool)
    ex    = ~valid
    n     = len(rr)
    out.update({
        "n_intervals":        n,
        "n_valid":            int(valid.sum()),
        "n_excluded":         int(ex.sum()),
        "n_too_short":        int((ex & (rr < lo)).sum()),
        "n_too_long":         int((ex & (rr > hi)).sum()),
        "n_spanning_removed": int((ex & (rr >= lo) & (rr <= hi)).sum()),
        "excluded_pct":       float(100.0 * ex.sum() / n),
        "excluded_time_s":    float(np.nansum(rr[ex]) / 1000.0),
    })
    return out


def rr_disclosure_rows(report: dict) -> "list[tuple[str, str]]":
    """(label, value) rows disclosing an ``rr_exclusion_report()`` -- shared
    by the Excel Cover sheet, the Prism project notes and the Summary text
    report so they all say the same thing. The bounds row is always
    present; the count rows only when the counts are known."""
    bounds = rr_bounds_label(report)
    rows: "list[tuple[str, str]]" = []
    if report.get("n_intervals") is not None:
        rows.append(("RR intervals",
                     f"{report['n_intervals']}  ({report['n_valid']} used for HRV)"))
    if report.get("n_excluded") is not None:
        n_ex = int(report["n_excluded"])
        value = f"{n_ex}  ({float(report.get('excluded_pct') or 0.0):.1f} %)"
        if n_ex:
            value += (f" — {report['n_too_short']} too short, "
                      f"{report['n_too_long']} too long, "
                      f"{report['n_spanning_removed']} spanning removed beats")
        rows.append(("RR excluded", value))
        if n_ex:
            rows.append(("RR excluded time",
                         f"{float(report.get('excluded_time_s') or 0.0):.2f} s"))
    rows.append(("RR bounds (HRV)", f"{bounds}  (others excluded, not bridged)"))
    return rows


def spectral_withheld_reason(
    rr_ms:             np.ndarray,
    valid:             np.ndarray,
    max_gap_s:         float = MouseECG.SPECTRAL_MAX_GAP_S,
    max_excluded_frac: float = MouseECG.SPECTRAL_MAX_EXCLUDED_FRAC,
) -> "Optional[str]":
    """Why spectral HRV must not be computed on this RR series, or None.

    A spectrum needs a uniformly resampled RR series, so every excluded
    interval leaves a hole the interpolator fills with invented values.
    The spectrum is withheld when the longest run of CONSECUTIVE excluded
    intervals lasts more than *max_gap_s*, or when excluded intervals
    cover more than *max_excluded_frac* of the analysed span (see
    MouseECG.SPECTRAL_MAX_GAP_S / SPECTRAL_MAX_EXCLUDED_FRAC). *rr_ms* is
    every interval (ms), *valid* the mask of the ones kept.
    """
    rr = np.asarray(rr_ms, dtype=np.float64).flatten()
    ok = np.asarray(valid, dtype=bool).flatten()
    if len(rr) == 0 or len(ok) != len(rr) or ok.all():
        return None
    excluded = ~ok
    runs_s   = _run_sums(rr, excluded) / 1000.0
    longest  = float(runs_s.max()) if len(runs_s) else 0.0
    if longest > max_gap_s:
        return (f"{longest:.1f} s run of excluded RR intervals "
                f"(limit {max_gap_s:g} s) — it would be interpolated across")
    finite = np.isfinite(rr)
    span_s = float(rr[finite].sum()) / 1000.0
    if span_s > 0:
        frac = float(rr[excluded & finite].sum()) / 1000.0 / span_s
        if frac > max_excluded_frac:
            return (f"excluded RR intervals cover {100.0 * frac:.1f} % of the "
                    f"analysed time (limit {100.0 * max_excluded_frac:g} %)")
    return None


def hrv_freq_withheld_reason(results: "Optional[dict]") -> "Optional[str]":
    """Why the frequency-domain HRV of *results* was withheld (the
    ``withheld_reason`` analyse_hrv_freq() put on its empty DataFrame), or
    None when it was computed, never computed, or the reason is unknown.
    Shared by the UI status labels, the Summary report and the exports, so
    a withheld spectrum is explained wherever its values are missing."""
    fd = (results or {}).get("hrv_freq")
    if fd is None or not getattr(fd, "empty", False):
        return None
    reason = getattr(fd, "attrs", {}).get("withheld_reason")
    return str(reason) if reason else None


# NeuroKit2 decides from RRI_Time whether intervals are missing, and only
# then restricts successive differences to truly consecutive intervals. That
# heuristic (intervals_utils._intervals_missing) silently fails in two cases
# that occur here: (1) when no two kept intervals are adjacent at all (it
# requires at least one successive pair), e.g. T-wave double-counting on
# every beat or a 2:1 block with alternate intervals excluded; (2) when the
# kept timestamps happen to be uniformly spaced, which it reads as
# "interpolated data, nothing missing" -- e.g. a spurious peak splitting a
# 120 ms interval into an excluded 20 ms and a kept 100 ms one on regular
# (paced/synthetic) data. In both cases it differences straight across the
# excluded interval. So whenever anything was excluded, the successive-
# difference / Poincaré metrics are recomputed here from the explicit
# adjacency mask (successive_valid_pairs) and overwrite NeuroKit's columns
# -- same formulas as NeuroKit, never bridged -- and the remaining pair
# metrics are withheld if NeuroKit is seen to have bridged (below).

# Heart-rate asymmetry and fragmentation indices: also successive-pair
# metrics, but not recomputed here -- withheld (NaN) when NeuroKit is shown
# to have paired intervals across a gap (its SD1/SD2 disagree with ours).
_NL_PAIR_COLUMNS       = ("HRV_GI", "HRV_SI", "HRV_AI", "HRV_PI",
                          "HRV_C1d", "HRV_C1a", "HRV_SD1d", "HRV_SD1a",
                          "HRV_C2d", "HRV_C2a", "HRV_SD2d", "HRV_SD2a",
                          "HRV_Cd", "HRV_Ca", "HRV_SDNNd", "HRV_SDNNa",
                          "HRV_PIP", "HRV_IALS", "HRV_PSS", "HRV_PAS")


def _set_col(df: "pd.DataFrame", col: str, value: float) -> None:
    if col in df.columns:
        df[col] = float(value)


def _overwrite_successive_time_domain(df: "pd.DataFrame", series: dict) -> None:
    """Gap-aware RMSSD/SDSD/CVSD/SDRMSSD/pNN50/pNN20 into nk.hrv_time's frame
    (see the comment above). Formulas mirror nk.hrv_time, including its
    pNNx denominator (number of differences + 1)."""
    rr_a, rr_b = successive_valid_pairs(series["rr_ms"], series["valid"])
    d   = rr_b - rr_a
    n_d = len(d)
    nan = float("nan")
    rmssd = float(np.sqrt(np.mean(d ** 2))) if n_d else nan
    _set_col(df, "HRV_RMSSD", rmssd)
    _set_col(df, "HRV_SDSD",  float(np.std(d, ddof=1)) if n_d > 1 else nan)
    _set_col(df, "HRV_pNN50", float(np.sum(np.abs(d) > 50) / (n_d + 1) * 100) if n_d else nan)
    _set_col(df, "HRV_pNN20", float(np.sum(np.abs(d) > 20) / (n_d + 1) * 100) if n_d else nan)
    mean_nn = float(df["HRV_MeanNN"].iloc[0]) if "HRV_MeanNN" in df.columns else nan
    sdnn    = float(df["HRV_SDNN"].iloc[0])   if "HRV_SDNN"   in df.columns else nan
    ok_rmssd = np.isfinite(rmssd) and rmssd > 0
    _set_col(df, "HRV_CVSD",    rmssd / mean_nn if ok_rmssd and np.isfinite(mean_nn) and mean_nn > 0 else nan)
    _set_col(df, "HRV_SDRMSSD", sdnn / rmssd    if ok_rmssd else nan)


def _overwrite_successive_poincare(df: "pd.DataFrame", series: dict) -> bool:
    """Gap-aware Poincaré SD1/SD2 (+ SD1/SD2, ellipse area S, CSI, CVI,
    modified CSI) into nk.hrv_nonlinear's frame (see the comment above).
    Formulas mirror nk.hrv_nonlinear. Returns True when NeuroKit's own
    SD1/SD2 already agreed -- i.e. it paired the same intervals."""
    rr_a, rr_b = successive_valid_pairs(series["rr_ms"], series["valid"])
    nan = float("nan")
    if len(rr_a) > 1:
        sd1 = float(np.std((rr_a - rr_b) / np.sqrt(2), ddof=1))
        sd2 = float(np.std((rr_a + rr_b) / np.sqrt(2), ddof=1))
    else:
        sd1 = sd2 = nan

    def _same(col: str, ours: float) -> bool:
        if col not in df.columns:
            return True
        theirs = float(df[col].iloc[0])
        if not (np.isfinite(theirs) or np.isfinite(ours)):
            return True
        return bool(np.isclose(theirs, ours, rtol=1e-9, atol=1e-12))

    agreed = _same("HRV_SD1", sd1) and _same("HRV_SD2", sd2)
    t, l   = 4.0 * sd1, 4.0 * sd2
    pos    = np.isfinite(sd1) and np.isfinite(sd2) and sd1 > 0 and sd2 > 0
    _set_col(df, "HRV_SD1", sd1)
    _set_col(df, "HRV_SD2", sd2)
    _set_col(df, "HRV_SD1SD2",       sd1 / sd2 if pos else nan)
    _set_col(df, "HRV_S",            np.pi * sd1 * sd2 if pos else nan)
    _set_col(df, "HRV_CSI",          l / t if pos else nan)
    _set_col(df, "HRV_CVI",          float(np.log10(l * t)) if pos else nan)
    _set_col(df, "HRV_CSI_Modified", l ** 2 / t if pos else nan)
    return agreed


def hrv_time_domain(series: dict, fs: float) -> "pd.DataFrame":
    """Gap-aware time-domain HRV of an ``rr_interval_series()`` result.

    nk.hrv_time on the valid intervals at their true times; when anything
    was excluded, the successive-difference metrics are recomputed from
    adjacent valid pairs only (see ``_overwrite_successive_time_domain``).
    Adds the mouse-scale HRV_pNN6 and stores the exclusion summary in
    ``attrs``. Shared by analyse_core and the UI's per-epoch HRV.
    """
    _require_nk()
    assert nk is not None  # narrowed after _require_nk()
    df = nk.hrv_time(nk_rri_input(series), sampling_rate=int(fs), show=False)
    if series["n_excluded"] > 0:
        _overwrite_successive_time_domain(df, series)

    # ── pNN6: mouse-appropriate pNNx (replaces the human-calibrated pNN50) ──
    # pNN50 is near 0 % for all healthy mice because normal vagal RR swings are
    # only 5–10 ms, far below the 50 ms threshold.  pNN6 (≈ 5 % of mean RR)
    # captures the same physiological information at the right scale.
    # Reference: Thireau et al. 2008, Am J Physiol Heart Circ Physiol 294:H977.
    # Gap-aware: only differences between two adjacent valid intervals.
    try:
        rr_a, rr_b = successive_valid_pairs(series["rr_ms"], series["valid"])
        if len(rr_a):
            pnn6_val = float(np.mean(np.abs(rr_b - rr_a) > MouseECG.PNN_THRESHOLD) * 100)
        else:
            pnn6_val = float("nan")
        df["HRV_pNN6"] = pnn6_val
    except Exception as exc:
        log.warning("pNN6 computation failed: %s", exc)
        df["HRV_pNN6"] = float("nan")
    df.attrs.update(rr_exclusion_summary(series))
    return df


def _beat_window_min_rr_samples(
    rpeaks:    np.ndarray,
    fs:        float,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
) -> "Optional[int]":
    """Shortest RR interval (samples) the beat window must fit in, or None
    when there is no positive interval at all.

    Taken over the VALID intervals only (inside [rr_min_ms, rr_max_ms]):
    one spurious peak ~45 ms after a real one (artifact correction is off
    by default) used to shrink the window for the WHOLE recording to
    ±20 ms and wipe out the T wave -- hence QT -- on every beat, while HRV
    already ignores that interval. Falls back to all positive intervals when none
    is valid.
    """
    rr_samp = np.diff(np.sort(np.asarray(rpeaks, dtype=np.int64).flatten()))
    rr_samp = rr_samp[rr_samp > 0]
    if len(rr_samp) == 0:
        return None
    ok = rr_valid_mask(rr_samp / fs * 1000.0, rr_min_ms, rr_max_ms)
    return int(rr_samp[ok].min()) if ok.any() else int(rr_samp.min())


def compute_beat_correlation(
    cleaned:   np.ndarray,
    rpeaks:    np.ndarray,
    fs:        float,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
) -> dict:
    """Per-beat correlation against a beat template (two-pass: rough mean →
    quality-filtered refined template → final beat_corr). Caller guarantees
    len(rpeaks) >= 3. Returns beat_template, beat_time, beat_matrix (shrunk
    to <=60 ghost rows), beat_sd, beat_corr, peak_amps, and valid_rp (the
    rpeaks subset beat_corr/peak_amps correspond to -- peaks too close to
    the signal edges for a full beat window are excluded).

    *rr_min_ms*/*rr_max_ms*: the RR bounds whose valid intervals size the
    beat window (see ``_beat_window_min_rr_samples``).

    Pure NumPy, no NeuroKit2 dependency -- vectorised, fast even for 10 000
    beats, safe to call outside analyse_core()'s Full Analysis pipeline
    (e.g. live from the Detection tab right after peaks are detected).
    """
    beat_template = beat_time = beat_matrix = beat_sd = beat_corr = peak_amps = None
    valid_rp = np.array([], dtype=int)
    try:
        # ── Cap half_win so the window never overlaps adjacent beats ─────────
        # BEAT_HALF_WIN_S = ±100 ms is safe at resting HR (~500 bpm, RR=120 ms)
        # but at 700 bpm (RR=86 ms) a ±100 ms window contains the neighbouring
        # beats, biasing the mean template and P-wave auto-calibration.
        # Cap at 45 % of the shortest VALID RR to guarantee no overlap.
        fixed_half_win  = int(MouseECG.BEAT_HALF_WIN_S * fs)
        rr_min_samples  = _beat_window_min_rr_samples(rpeaks, fs, rr_min_ms, rr_max_ms)
        if rr_min_samples is None:
            rr_min_samples = fixed_half_win * 2
        half_win        = max(20, min(fixed_half_win, int(rr_min_samples * 0.45)))
        if half_win < fixed_half_win:
            log.debug(
                "Beat window capped: ±%d ms (min RR=%.0f ms, was ±%d ms)",
                round(half_win / fs * 1000),
                rr_min_samples / fs * 1000,
                round(fixed_half_win / fs * 1000),
            )
        valid_rp = rpeaks[(rpeaks - half_win >= 0) & (rpeaks + half_win < len(cleaned))]
        if len(valid_rp):
            # Build full matrix in one allocation — no Python loop
            idx         = valid_rp[:, None] + np.arange(-half_win, half_win)
            beat_matrix = cleaned[idx]                      # (n_beats, 2*half_win)
            beat_time     = np.arange(-half_win, half_win) / fs * 1000
            peak_amps     = beat_matrix[:, half_win].copy() # copy before shrinking matrix

            # ── Pass 1: rough mean beat from all beats ───────────────────────
            rough_mean    = beat_matrix.mean(axis=0)

            # ── Correlation via dot-product — 50× faster than corrcoef loop ──
            # Beat quality: Pearson r with rough template. Beats with r < 0.75
            # (ectopics, artefacts, transitions) are excluded from the refined
            # template, giving a much cleaner P/Q/R/S/T reference.
            tmpl_c = rough_mean - rough_mean.mean()
            tmpl_n = np.linalg.norm(tmpl_c)
            if tmpl_n > 1e-9:
                beats_c   = beat_matrix - beat_matrix.mean(axis=1, keepdims=True)
                norms     = np.linalg.norm(beats_c, axis=1, keepdims=True)
                norms     = np.where(norms < 1e-9, 1.0, norms)
                beat_corr = (beats_c @ tmpl_c) / (norms.ravel() * tmpl_n)
            else:
                beat_corr = np.ones(len(beat_matrix))

            # ── Pass 2: refined mean beat from high-quality beats only ───────
            good_mask     = beat_corr >= 0.75
            n_good        = int(good_mask.sum())
            if n_good >= max(5, len(beat_matrix) // 10):
                # Use high-quality beats for the template
                refined_matrix = beat_matrix[good_mask]
                beat_template  = refined_matrix.mean(axis=0)
                beat_sd        = refined_matrix.std(axis=0)
                log.debug("Beat template: %d/%d high-quality beats (corr≥0.75)",
                          n_good, len(beat_matrix))
            else:
                # Fall back to all beats if too few pass quality filter
                beat_template = rough_mean
                beat_sd       = beat_matrix.std(axis=0)
                log.debug("Beat template: quality filter yielded only %d/%d — using all",
                          n_good, len(beat_matrix))

            # Re-compute beat_corr against the refined template
            tmpl_c = beat_template - beat_template.mean()
            tmpl_n = np.linalg.norm(tmpl_c)
            if tmpl_n > 1e-9:
                beats_c   = beat_matrix - beat_matrix.mean(axis=1, keepdims=True)
                norms     = np.linalg.norm(beats_c, axis=1, keepdims=True)
                norms     = np.where(norms < 1e-9, 1.0, norms)
                beat_corr = (beats_c @ tmpl_c) / (norms.ravel() * tmpl_n)

            # ── Free the full beat_matrix immediately after statistics are done.
            # On a long recording (10k beats, fs=20kHz, 0.1s window) it can reach
            # 320 MB.  Only a small ghost-trace sub-sample (~60 rows) is kept for
            # the overlay in draw_template — negligible memory, same visual result.
            n_ghost   = min(60, len(valid_rp))
            rng       = np.random.default_rng(seed=0)
            ghost_idx = np.sort(rng.choice(len(valid_rp), size=n_ghost, replace=False))
            beat_matrix = beat_matrix[ghost_idx]   # ≤60 rows kept
            log.debug("Beat template: %d beats, matrix shrunk to %d ghost rows (%.1f kB)",
                      len(valid_rp), n_ghost, beat_matrix.nbytes / 1e3)
    except Exception as exc:
        log.warning("Beat template failed: %s", exc)

    return dict(
        beat_template=beat_template, beat_time=beat_time, beat_matrix=beat_matrix,
        beat_sd=beat_sd, beat_corr=beat_corr, peak_amps=peak_amps, valid_rp=valid_rp,
    )


def analyse_core(
    cleaned:     np.ndarray,
    rpeaks:      np.ndarray,
    fs:          float,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
    rr_min_ms:   float = MouseECG.RR_MIN_MS,
    rr_max_ms:   float = MouseECG.RR_MAX_MS,
    breaks:      "Optional[np.ndarray]" = None,
) -> AnalysisResults:
    """Fast core analysis: RR intervals, HR stats, time-domain HRV, beat template.

    Completes in < 2 s for any recording length.  Always run first; the other
    analyse_* functions expect its output dict to be merged into self._results.

    RR intervals outside [rr_min_ms, rr_max_ms], and intervals spanning a
    beat removed by artifact correction (*breaks*, the artifact report's
    ``break_samples``),
    are excluded from HR stats and HRV without being bridged (see
    rr_interval_series); ``rr_ms`` keeps every interval (the tachogram
    still shows a pause), ``rr_valid`` marks the ones used and ``rr_t_s``
    holds each interval's end time (``rr_timeseries()`` puts the three
    together). ``hr`` and ``hrv_time.attrs`` carry the exclusion counts and
    the bounds used.
    Raises ValueError for fewer than 3 peaks or fewer than 3 valid
    intervals.
    """
    _require_nk()
    assert nk is not None  # narrowed after _require_nk()

    def _prog(pct, msg):
        if progress_cb:
            try: progress_cb(pct, msg)
            except Exception as exc:
                log.debug("analyse_core: progress_cb raised %s — continuing", exc)

    rpeaks = np.sort(np.array(rpeaks, dtype=int).flatten())
    if len(rpeaks) < 3:
        raise ValueError(f"Too few R-peaks ({len(rpeaks)}) — need at least 3.")

    # Gap-aware RR series: out-of-range intervals and intervals spanning a
    # removed beat are excluded, not bridged, and only a recording left with
    # too few valid intervals is refused.
    rr_series = rr_interval_series(rpeaks, fs, rr_min_ms, rr_max_ms,
                                   breaks=breaks)   # also validates fs
    rr_ms     = rr_series["rr_ms"]
    rr_ok_mask = rr_series["valid"]
    rr_clean  = rr_series["rri"]
    exclusion = rr_exclusion_summary(rr_series)
    if len(rr_clean) < 3:
        n_span = exclusion["n_spanning_removed"]
        span_note = (f", {n_span} spanning beats removed by artifact correction"
                     if n_span else "")
        if n_span > max(exclusion["n_too_short"], exclusion["n_too_long"]):
            # Mostly breaks: the RR bounds are not what removed them.
            hint = ("Most of them span beats removed by artifact correction: "
                    "review the removals (Review Artifacts) or turn auto-"
                    "correction off, then re-run.")
        else:
            hint = (f"If the heart rate is genuinely slow (e.g. an anaesthetised "
                    f"mouse below {60_000.0 / rr_max_ms:.0f} bpm), widen Max RR; "
                    f"otherwise check the detected R-peaks.")
        raise ValueError(
            f"Only {len(rr_clean)} of {rr_series['n_total']} RR intervals are usable "
            f"for HRV — need at least 3. Accepted RR range: {rr_min_ms:.1f}–"
            f"{rr_max_ms:.1f} ms ({exclusion['n_too_short']} shorter, "
            f"{exclusion['n_too_long']} longer{span_note}). {hint}")
    dur_s = float(rpeaks[-1] - rpeaks[0]) / fs   # actual recording span
    log.info("analyse_core: %d peaks, %.1f s, mean RR=%.1f ms, %d/%d RR excluded",
             len(rpeaks), dur_s, float(rr_clean.mean()),
             exclusion["n_excluded"], rr_series["n_total"])

    _prog(5, "RR intervals…")
    with np.errstate(divide="ignore"):   # a 0-ms interval (duplicate peak) is filtered out below
        hr_all = 60_000.0 / rr_ms
    rr_df = pd.DataFrame({
        "Time_s": rr_series["t_s"],
        "RR_ms":  rr_ms,
        "HR_bpm": hr_all,
    })
    rr_df = rr_df[rr_ok_mask].copy()   # valid intervals at their true times

    # HR stats from the valid intervals only
    hr = {
        "mean": float(np.nanmean(60_000 / rr_clean)),
        "min":  float(60_000 / np.percentile(rr_clean, 98)),
        "max":  float(60_000 / np.percentile(rr_clean, 2)),
        "std":  float(np.nanstd(60_000 / rr_clean)),
        "n":     int(len(rpeaks)),      # total detected (status bar)
        "n_valid": int(len(rr_clean)),   # valid after RR bounds / removed-beat filter
        # n_excluded, n_too_short, n_too_long, n_spanning_removed,
        # excluded_pct, excluded_time_s, rr_min_ms, rr_max_ms
        **exclusion,
    }

    _prog(30, "Time-domain HRV (SDNN, RMSSD…)")
    # Valid intervals with their end times; successive differences never
    # straddle an excluded interval (also adds the mouse-scale pNN6).
    hrv_time = hrv_time_domain(rr_series, fs)

    # Beat template — vectorised, fast even for 10 000 beats
    _prog(65, "Beat template…")
    _bc = compute_beat_correlation(cleaned, rpeaks, fs, rr_min_ms, rr_max_ms)
    beat_template = _bc["beat_template"]
    beat_time     = _bc["beat_time"]
    beat_matrix   = _bc["beat_matrix"]
    beat_sd       = _bc["beat_sd"]
    beat_corr     = _bc["beat_corr"]
    peak_amps     = _bc["peak_amps"]

    _prog(100, "Core analysis done")
    return cast(AnalysisResults, dict(
        hr=hr, rr_ms=rr_ms, rr_valid=rr_ok_mask, rr_t_s=rr_series["t_s"], rr_df=rr_df,
        hrv_time=hrv_time,
        hrv_freq=pd.DataFrame(), hrv_nonlin=pd.DataFrame(),
        intervals=pd.DataFrame({"RR_ms": rr_ms}),
        beat_template=beat_template, beat_time=beat_time,
        beat_matrix=beat_matrix,   # ≤60 rows — negligible memory
        beat_sd=beat_sd,
        beat_corr=beat_corr, peak_amps=peak_amps,
    ))


def _withheld_frame(reason: str, series: dict) -> "pd.DataFrame":
    """Empty HRV DataFrame carrying why it is empty (attrs["withheld_reason"])
    and the exclusion summary."""
    df = pd.DataFrame()
    df.attrs.update(rr_exclusion_summary(series))
    df.attrs["withheld_reason"] = reason
    return df


def analyse_hrv_freq(
    rpeaks:      np.ndarray,
    fs:          float,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
    rr_min_ms:   float = MouseECG.RR_MIN_MS,
    rr_max_ms:   float = MouseECG.RR_MAX_MS,
    breaks:      "Optional[np.ndarray]" = None,
) -> "pd.DataFrame":
    """Frequency-domain HRV (VLF, LF, HF, LF/HF) with mouse-specific bands.

    NeuroKit2's default bands (LF: 0.04–0.15 Hz, HF: 0.15–0.4 Hz) are tuned
    for humans.  Mouse HRV uses completely different ranges (Thireau et al.
    2008, Baudrie et al. 2007):
        VLF : 0.000 – 0.400 Hz
        LF  : 0.400 – 1.500 Hz   (baroreflex / sympathovagal oscillations)
        HF  : 1.500 – 5.000 Hz   (respiratory sinus arrhythmia, breathing at 2–4 Hz)

    The minimum recommended recording length for stable spectral estimates is:
        ≥ 5 × (1 / VLF_lo) ≈ impossible → use ≥ 5 / LF_lo ≈ 12.5 s (≥ ~100 beats)

    RR intervals outside [rr_min_ms, rr_max_ms] and intervals spanning a
    removed beat (*breaks*) are excluded (see rr_interval_series); the
    valid ones are passed with their true times, so NeuroKit resamples them
    on the real time axis rather than shifting every later sample by the
    excluded duration. The exclusion counts are stored in the returned
    DataFrame's ``attrs``.

    The spectrum is WITHHELD -- an empty DataFrame whose
    ``attrs["withheld_reason"]`` says why -- when fewer than 60 valid
    intervals remain, or when the excluded intervals are long enough that
    resampling would interpolate across them (``spectral_withheld_reason``:
    a run of excluded intervals > MouseECG.SPECTRAL_MAX_GAP_S, or excluded
    time > MouseECG.SPECTRAL_MAX_EXCLUDED_FRAC of the analysed span).

    Parameters
    ----------
    rpeaks      : R-peak sample indices (from polarity-corrected filtered signal)
    fs          : Sampling frequency in Hz
    progress_cb : Optional progress callback (pct: int, msg: str) → None
    rr_min_ms, rr_max_ms : RR acceptance bounds (ms)
    breaks      : sample positions of beats removed by artifact correction
    """
    _require_nk()
    assert nk is not None  # narrowed after _require_nk()

    rr_series = rr_interval_series(rpeaks, fs, rr_min_ms, rr_max_ms,
                                   breaks=breaks)   # also validates fs
    n_valid   = len(rr_series["rri"])

    # Minimum valid-interval count for any meaningful spectral estimate
    MIN_BEATS_SPECTRAL = 60   # ~7–15 s of mouse ECG at typical HR
    if n_valid < MIN_BEATS_SPECTRAL:
        log.warning(
            "analyse_hrv_freq: only %d valid RR intervals (%d excluded) — "
            "need ≥ %d for spectral HRV",
            n_valid, rr_series["n_excluded"], MIN_BEATS_SPECTRAL,
        )
        return _withheld_frame(
            f"only {n_valid} valid RR intervals (need ≥ {MIN_BEATS_SPECTRAL})",
            rr_series)

    reason = spectral_withheld_reason(rr_series["rr_ms"], rr_series["valid"])
    if reason:
        log.warning("analyse_hrv_freq: spectrum withheld — %s", reason)
        return _withheld_frame(reason, rr_series)

    if progress_cb:
        progress_cb(10, "Frequency-domain HRV (mouse bands: LF 0.4–1.5 Hz, HF 1.5–5.0 Hz)…")

    try:
        result = nk.hrv_frequency(
            nk_rri_input(rr_series),
            sampling_rate=int(fs),
            show=False,
            normalize=True,
            # ── Mouse-specific HRV frequency bands ─────────────────────────────
            # Reference: Thireau et al. (2008) Am J Physiol Heart Circ Physiol 294:H977
            #            Baudrie et al. (2007) Am J Physiol Regul Integr Comp Physiol 293:R306
            ulf=(0.000, 0.000),   # ULF not relevant for acute recordings — set to zero
            vlf=(0.000, 0.400),   # Very-low-frequency (vasomotor / thermoregulatory)
            lf=(0.400,  1.500),   # Low-frequency  (baroreflex / sympathovagal balance)
            hf=(1.500,  5.000),   # High-frequency (respiratory sinus arrhythmia, 2–4 Hz)
            vhf=(5.000, 20.000),  # Very-high-frequency (harmonics / noise floor)
        )
        result.attrs.update(rr_exclusion_summary(rr_series))
        if progress_cb:
            progress_cb(100, "Frequency HRV done")
        return result
    except Exception as exc:
        log.warning("analyse_hrv_freq failed: %s", exc)
        return pd.DataFrame()

def analyse_hrv_nonlinear(
    rpeaks: np.ndarray,
    fs: float,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    max_beats: int = 1000,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
    breaks: "Optional[np.ndarray]" = None,
) -> pd.DataFrame:
    """Non-linear HRV via NeuroKit2: Poincaré SD1/SD2, DFA α1/α2, ApEn/SampEn,
    Porta index, and the rest of nk.hrv_nonlinear()'s standard metric set.

    This used to be a bespoke implementation that only ever returned
    HRV_SampEn (computed as the entropy of individual BEAT WAVEFORMS,
    averaged across beats) and an ad hoc nearest-neighbour distance proxy
    (HRV_NNDistLog) that no caller ever displayed or exported. Meanwhile
    every consumer of this DataFrame -- the Poincaré plot title, the
    non-linear text report, Excel/Prism export, the radar chart -- reads
    HRV_SD1, HRV_SD2, HRV_DFA_alpha1, HRV_DFA_alpha2, HRV_ApEn, HRV_PI,
    none of which that bespoke version ever populated, so those fields
    silently showed "—" on every run.

    HRV_SampEn's definition also changes here: nk.hrv_nonlinear() computes
    it on the RR-INTERVAL series (the standard textbook Sample Entropy of
    HRV, and what PARAM_INFO's own documented reference range of 0.5-2.5
    assumes), not on beat-waveform morphology -- the old per-beat-waveform
    values (~0.02 for a typical recording) were far outside that
    documented range because they were never measuring the same thing.

    Tested down to 5 beats with no crashes; nk degrades individual columns
    to NaN (e.g. HRV_DFA_alpha1 needs enough beats for a stable fit) rather
    than raising.

    RR intervals outside [rr_min_ms, rr_max_ms] and intervals spanning a
    removed beat (*breaks*) are excluded (see rr_interval_series) after the
    max_beats truncation; Poincaré pairs never straddle an excluded
    interval -- when anything was excluded, SD1/SD2 and the indices derived
    from them are recomputed from adjacent valid pairs, and the asymmetry /
    fragmentation indices are withheld (NaN, listed in
    ``attrs["pair_metrics_withheld"]``) if NeuroKit paired intervals across
    a gap. The exclusion counts are stored in the returned DataFrame's
    ``attrs``.

    Args:
        rpeaks: R-peak sample indices (ndarray)
        fs: Sampling rate (Hz)
        progress_cb: Progress callback (pct, message) -> None
        max_beats: Cap on beats passed to nk (keeps runtime bounded on long recordings)
        rr_min_ms, rr_max_ms: RR acceptance bounds (ms)
        breaks: sample positions of beats removed by artifact correction

    Returns:
        DataFrame with nk.hrv_nonlinear()'s standard non-linear HRV metrics.
    """
    _require_nk()
    assert nk is not None  # narrowed after _require_nk()

    if progress_cb:
        progress_cb(0, "Non-linear HRV: preparing R-peaks…")

    rpeaks = np.sort(np.asarray(rpeaks, dtype=int).flatten())
    n_rpeaks_orig = len(rpeaks)
    truncated = n_rpeaks_orig > max_beats
    if truncated:
        # Deliberately the first max_beats, NOT an evenly-spaced subsample:
        # SampEn/DFA are sequence-order statistics computed on adjacent RR
        # intervals -- sampling every Nth beat across the full recording
        # would feed them a sequence whose "neighbouring" intervals are
        # actually minutes apart in real time, silently changing what's
        # being measured rather than just changing how much of it. Callers
        # are told via df.attrs (see below) so the UI can disclose the
        # truncation instead of silently reporting a whole-recording number.
        rpeaks = rpeaks[:max_beats]
        log.warning(f"Limiting to {max_beats} beats (was {n_rpeaks_orig})")

    rr_series = rr_interval_series(rpeaks, fs, rr_min_ms, rr_max_ms,
                                   breaks=breaks)   # also validates fs

    if progress_cb:
        progress_cb(20, "Computing SD1/SD2, DFA, SampEn/ApEn… (may take 30 s+)")

    nk_failed_reason: "Optional[str]" = None
    try:
        result = nk.hrv_nonlinear(nk_rri_input(rr_series), sampling_rate=int(fs), show=False)
    except Exception as exc:
        # NeuroKit 0.2.13 divides by the number of non-zero successive
        # differences (_hrv_nonlinear_poincare_hra), so an RR series with
        # exactly zero beat-to-beat variability (paced or synthetic data)
        # raises ZeroDivisionError. Keep the gap-aware Poincaré set below
        # instead of returning nothing, and say why the rest is missing.
        log.warning("analyse_hrv_nonlinear failed: %s", exc)
        nk_failed_reason = f"NeuroKit non-linear HRV failed ({type(exc).__name__}: {exc})"
        poincare_cols = ("HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_S",
                         "HRV_CSI", "HRV_CVI", "HRV_CSI_Modified")
        result = pd.DataFrame({c: [float("nan")] for c in poincare_cols})
        _overwrite_successive_poincare(result, rr_series)
        if not np.isfinite(float(result["HRV_SD1"].iloc[0])):
            return pd.DataFrame()

    # Never let NeuroKit bridge a gap (see _overwrite_successive_poincare's
    # section comment): recompute the Poincaré set from adjacent valid
    # pairs; if NeuroKit's own SD1/SD2 disagreed, it paired intervals across
    # an excluded one, so its other pair-based indices are withheld too.
    pair_withheld: "list[str]" = []
    if rr_series["n_excluded"] > 0:
        if not _overwrite_successive_poincare(result, rr_series):
            pair_withheld = [c for c in _NL_PAIR_COLUMNS if c in result.columns]
            for col in pair_withheld:
                result[col] = float("nan")
            log.warning("analyse_hrv_nonlinear: NeuroKit paired RR intervals across "
                        "an excluded one — %d asymmetry/fragmentation indices withheld",
                        len(pair_withheld))

    if progress_cb:
        progress_cb(60, "Multiscale entropy…")

    # Multiscale entropy (Costa et al. 2002): SampEn recomputed at coarser
    # and coarser time scales of the SAME (already truncated) RR series
    # nk.hrv_nonlinear() just used -- catches loss of complexity that a
    # single-scale SampEn can miss (rodent pharmacology literature reports
    # cases where combined autonomic blockade only showed up at intermediate
    # MSE scales, not at scale 1). Needs materially more data than a single
    # SampEn value to be stable per scale, so this is skipped outright below
    # a minimum beat count rather than returning a noisy curve.
    # Valid intervals only: an excluded interval is dropped, NOT interpolated
    # (nk.entropy_multiscale has no notion of a gap, so the coarse-grained
    # series simply continues with the next valid interval).
    rr_ms_mse = rr_series["rri"]
    mse_scales: "list[int]" = []
    mse_values: "list[float]" = []
    mse_point: "Optional[float]" = None
    MIN_BEATS_MSE = 60   # ~30 points/scale at the coarsest scale used below
    if len(rr_ms_mse) >= MIN_BEATS_MSE:
        try:
            scale_max = max(2, min(15, len(rr_ms_mse) // 30))
            _, mse_info = nk.entropy_multiscale(
                rr_ms_mse, scale=list(range(1, scale_max + 1)), show=False)
            mse_scales = [int(s) for s in mse_info.get("Scale", [])]
            mse_values = [float(v) for v in mse_info.get("Value", [])]
            finite = [v for v in mse_values if np.isfinite(v)]
            mse_point = float(np.sum(finite)) if finite else None
        except Exception as exc:
            log.warning("analyse_hrv_nonlinear: entropy_multiscale failed: %s", exc)
            mse_scales, mse_values, mse_point = [], [], None

    if progress_cb:
        progress_cb(100, "Non-linear HRV done")

    # DataFrame.attrs is the standard pandas mechanism for attaching
    # non-tabular metadata without changing the column/row structure any
    # existing caller reads -- see analysis_controller.py's run_nonlinear()
    # for the UI-facing consumer.
    result.attrs["n_beats_used"]  = len(rpeaks)
    result.attrs["n_beats_total"] = n_rpeaks_orig
    result.attrs["truncated"]     = truncated
    result.attrs["mse_scales"]    = mse_scales
    result.attrs["mse_values"]    = mse_values
    result.attrs["mse_point_estimate"] = mse_point
    result.attrs["pair_metrics_withheld"] = pair_withheld
    result.attrs["nk_failed_reason"] = nk_failed_reason
    result.attrs.update(rr_exclusion_summary(rr_series))
    return result


def compute_qtc(
    qt_ms:   "np.ndarray",
    rr_ms:   "np.ndarray",
    formula: str = "mitchell",
) -> np.ndarray:
    """Heart-rate-corrected QT (ms) for each (QT, RR) pair.

    formula (any value ``normalize_qtc_formula`` accepts):

    * ``"mitchell"``   — QT / √(RR_ms / 100)  — Mitchell et al. 1998 (Am J
      Physiol 274:H747): Bazett normalised to a 100 ms mouse RR, so QTc
      equals QT at ~600 bpm.
    * ``"bazett"``     — QT / √(RR_s)   — human-calibrated (RR in seconds).
    * ``"fridericia"`` — QT / ∛(RR_s)   — human-calibrated (RR in seconds).

    No absolute range is applied to the result — QT and RR are bounds-checked
    upstream, and a fixed QTc range would only fit one formula's scale. NaN
    wherever QT or RR is non-finite or RR ≤ 0.
    """
    qt = np.asarray(qt_ms, dtype=float)
    rr = np.asarray(rr_ms, dtype=float)
    ok = np.isfinite(qt) & np.isfinite(rr) & (rr > 0)
    rr_safe = np.where(ok, rr, 1.0)   # keeps sqrt/cbrt warning-free on bad pairs
    key = normalize_qtc_formula(formula)
    if key == "bazett":
        qtc = qt / np.sqrt(rr_safe / 1000.0)
    elif key == "fridericia":
        qtc = qt / np.cbrt(rr_safe / 1000.0)
    else:
        qtc = qt / np.sqrt(rr_safe / 100.0)
    return np.where(ok, qtc, np.nan)


def recompute_qtc_column(intervals: "pd.DataFrame", formula: str) -> "pd.DataFrame":
    """Return a copy of an ``analyse_intervals`` DataFrame whose ``QTc_ms``
    column is recomputed from its ``QT_ms`` / ``RR_ms`` with *formula*.

    Uses exactly the per-beat RR that ``analyse_intervals`` pairs with each
    QT (floored at ``MouseECG.RR_MIN_MS``), so the result is identical to
    re-running the delineation with that formula. Frames without both
    columns (e.g. the early-return ``{"RR_ms": ...}`` shape) come back
    unchanged.
    """
    if "QT_ms" not in intervals.columns or "RR_ms" not in intervals.columns:
        return intervals
    qt = pd.to_numeric(intervals["QT_ms"], errors="coerce").to_numpy(dtype=float)
    rr = pd.to_numeric(intervals["RR_ms"], errors="coerce").to_numpy(dtype=float)
    out = intervals.copy()
    out["QTc_ms"] = compute_qtc(qt, np.clip(rr, MouseECG.RR_MIN_MS, None), formula)
    return out


def analyse_intervals(
    cleaned:           "np.ndarray",
    rpeaks:            "np.ndarray",
    fs:                float,
    rr_ms:             "np.ndarray",
    max_beats:         int = 0,          # kept for API compat, no longer used
    progress_cb:       "Optional[Callable[[int, str], None]]" = None,
    wave_template:     "Optional[WaveTemplate]" = None,
    beat_template:     "Optional[np.ndarray]"   = None,
    beat_time_arr:     "Optional[np.ndarray]"   = None,
    permissive_bounds: bool = False,
    qtc_formula:       str  = "mitchell",
    rr_min_ms:         float = MouseECG.RR_MIN_MS,
    rr_max_ms:         float = MouseECG.RR_MAX_MS,
) -> "pd.DataFrame":
    """Template-guided per-beat ECG interval delineation.  No NeuroKit2.

    Detects P/Q/S/T landmarks directly on each beat using
    ``detect_waves_on_beat``, guided by the WaveTemplate search windows.
    All beats are processed (no sub-sampling).

    Parameters
    ----------
    cleaned   : bandpass-filtered ECG (same array used by analyse_core).
    rpeaks    : sample indices of all accepted R peaks.
    fs        : sampling frequency (Hz).
    rr_ms     : RR interval array (len = len(rpeaks)-1).
    max_beats : deprecated — ignored.  Present only for backward compat.
    progress_cb : optional progress(pct, message) callback.
    wave_template : search-window template; defaults are used when None.
    beat_template, beat_time_arr : pre-computed mean beat from analyse_core
        (optional — only used to auto-update the template after the run).
    permissive_bounds : if True, relax physiological bounds by 2×.
    qtc_formula : QTc correction for the ``QTc_ms`` column — see
        ``compute_qtc`` (default Mitchell, mouse-calibrated).
    rr_min_ms, rr_max_ms : RR acceptance bounds (ms) -- the beat window is
        sized from the shortest interval inside them, so pass the bounds
        HRV used (e.g. a widened Max RR for an anaesthetised mouse, where
        no interval is valid under the defaults).

    Returns
    -------
    pd.DataFrame with columns RR_ms, PR_ms, QRS_ms, QT_ms, QTc_ms,
    R_peak_s, plus per-wave absolute (``_s``) and relative (``_ms``) positions,
    beat_idx, and an ``accepted`` boolean column for the verifier UI.
    """
    if max_beats and max_beats > 0:
        log.debug("analyse_intervals: max_beats=%d is no longer used (all beats processed)",
                  max_beats)

    def _prog(p: int, m: str) -> None:
        if progress_cb:
            progress_cb(p, m)

    if wave_template is None:
        wave_template = WaveTemplate.load()

    rpeaks_all = np.sort(np.array(rpeaks, dtype=int).flatten())
    n_beats    = len(rpeaks_all)
    if n_beats < 3:
        return pd.DataFrame({"RR_ms": rr_ms})

    # ── Beat window ──────────────────────────────────────────────────────
    # Sized from the shortest VALID RR (see _beat_window_min_rr_samples):
    # one spurious close peak must not shrink every beat's window.
    fixed_hw  = int(MouseECG.BEAT_HALF_WIN_S * fs)
    rr_min_s  = _beat_window_min_rr_samples(rpeaks_all, fs, rr_min_ms, rr_max_ms)
    if rr_min_s is None:
        rr_min_s = fixed_hw * 2
    half_win  = max(20, min(fixed_hw, int(rr_min_s * 0.45)))
    bt_ms     = np.arange(-half_win, half_win) / fs * 1000

    mask_valid = ((rpeaks_all - half_win >= 0) &
                  (rpeaks_all + half_win < len(cleaned)))
    valid_rp   = rpeaks_all[mask_valid]
    n_valid    = len(valid_rp)

    if n_valid < 2:
        log.warning("analyse_intervals: too few valid beats (%d)", n_valid)
        return pd.DataFrame({"RR_ms": rr_ms})

    _prog(5, f"Building beat matrix ({n_valid} beats)…")

    # Vectorised beat extraction
    idx_mat  = valid_rp[:, None] + np.arange(-half_win, half_win)
    beat_mat = cleaned[idx_mat].astype(float)

    # ── Per-beat detection ───────────────────────────────────────────────
    _prog(10, "Detecting P/Q/S/T landmarks per beat…")

    rf       = valid_rp.astype(float)
    p_onset  = np.full(n_valid, np.nan)
    p_peak   = np.full(n_valid, np.nan)
    q_peak   = np.full(n_valid, np.nan)
    s_peak   = np.full(n_valid, np.nan)
    j_peak   = np.full(n_valid, np.nan)
    t_peak   = np.full(n_valid, np.nan)
    t_offset = np.full(n_valid, np.nan)

    tick = max(1, n_valid // 20)
    for i in range(n_valid):
        det = detect_waves_on_beat(beat_mat[i], bt_ms, wave_template)

        def _ms_to_samp(v: float) -> float:
            return valid_rp[i] + v / 1000.0 * fs if np.isfinite(v) else np.nan

        p_onset [i] = _ms_to_samp(det["P_onset"])
        p_peak  [i] = _ms_to_samp(det["P_peak"])
        q_peak  [i] = _ms_to_samp(det["Q_peak"])
        s_peak  [i] = _ms_to_samp(det["S_peak"])
        j_peak  [i] = _ms_to_samp(det["J_peak"])
        t_peak  [i] = _ms_to_samp(det["T_peak"])
        t_offset[i] = _ms_to_samp(det["T_offset"])

        if (i + 1) % tick == 0:
            _prog(10 + int(72 * (i + 1) / n_valid),
                  f"Detecting landmarks… {i + 1}/{n_valid}")

    _prog(82, "Computing intervals…")

    # ── Interval arithmetic ──────────────────────────────────────────────
    n = n_valid - 1
    def _t(a: "np.ndarray") -> "np.ndarray":
        return a[:n]

    r_pos_n    = valid_rp[:n].astype(float)
    p_onset_n  = _t(p_onset);  p_peak_n   = _t(p_peak)
    q_peak_n   = _t(q_peak);   s_peak_n   = _t(s_peak)
    j_peak_n   = _t(j_peak)
    t_peak_n   = _t(t_peak);   t_offset_n = _t(t_offset)

    rr_arr  = np.diff(valid_rp[:n + 1]).astype(float) / fs * 1000
    pr_end  = np.where(np.isfinite(q_peak_n), q_peak_n, r_pos_n)
    pr_raw  = (pr_end     - p_onset_n) / fs * 1000
    qrs_raw = (s_peak_n   - q_peak_n)  / fs * 1000
    qt_raw  = (t_offset_n - r_pos_n)   / fs * 1000

    # ── Physiological bounds ─────────────────────────────────────────────
    _m = 2.0 if permissive_bounds else 1.0
    def _clip(arr: "np.ndarray", lo: float, hi: float) -> "np.ndarray":
        out = arr.copy().astype(float)
        out[(out < lo) | (out > hi)] = np.nan
        return out

    pr_clean  = _clip(pr_raw,  MouseECG.PR_ABS_MIN  / _m, MouseECG.PR_ABS_MAX  * _m)
    qrs_clean = _clip(qrs_raw, MouseECG.QRS_ABS_MIN / _m, MouseECG.QRS_ABS_MAX * _m)
    qt_clean  = _clip(qt_raw,  MouseECG.QT_ABS_MIN  / _m, MouseECG.QT_ABS_MAX  * _m)
    rr_clean  = _clip(rr_arr,  MouseECG.RR_MIN_MS   / _m, MouseECG.RR_MAX_MS   * _m)

    # QTc pairs each QT with the same (floored) RR as before — mirrored by
    # recompute_qtc_column. No absolute QTc clamp: QT and RR are
    # bounds-checked just above (see compute_qtc).
    qtc_clean = compute_qtc(qt_clean, np.clip(rr_clean, MouseECG.RR_MIN_MS, None),
                            qtc_formula)

    # ── DataFrame ────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "RR_ms":      rr_clean,
        "PR_ms":      pr_clean,
        "QRS_ms":     qrs_clean,
        "QT_ms":      qt_clean,
        "QTc_ms":     qtc_clean,
        "R_peak_s":   r_pos_n    / fs,
        "P_onset_s":  p_onset_n  / fs,
        "P_peak_s":   p_peak_n   / fs,
        "Q_peak_s":   q_peak_n   / fs,
        "S_peak_s":   s_peak_n   / fs,
        "J_peak_s":   j_peak_n   / fs,
        "T_peak_s":   t_peak_n   / fs,
        "T_offset_s": t_offset_n / fs,
        # ms-from-R columns for the verifier UI (drag-to-adjust)
        "P_onset_ms":  (p_onset_n  - r_pos_n) / fs * 1000,
        "P_peak_ms":   (p_peak_n   - r_pos_n) / fs * 1000,
        "Q_peak_ms":   (q_peak_n   - r_pos_n) / fs * 1000,
        "S_peak_ms":   (s_peak_n   - r_pos_n) / fs * 1000,
        "J_peak_ms":   (j_peak_n   - r_pos_n) / fs * 1000,
        "T_peak_ms":   (t_peak_n   - r_pos_n) / fs * 1000,
        "T_offset_ms": (t_offset_n - r_pos_n) / fs * 1000,
        "beat_idx":    np.arange(n, dtype=int),
        "accepted":    np.ones(n, dtype=bool),
    })

    # NaN-out wave positions for out-of-bounds intervals
    df.loc[np.isnan(pr_raw)  | (pr_raw  < MouseECG.PR_ABS_MIN)  | (pr_raw  > MouseECG.PR_ABS_MAX),
           ["P_onset_s","P_peak_s","P_onset_ms","P_peak_ms"]] = np.nan
    df.loc[np.isnan(qrs_raw) | (qrs_raw < MouseECG.QRS_ABS_MIN) | (qrs_raw > MouseECG.QRS_ABS_MAX),
           ["Q_peak_s","S_peak_s","Q_peak_ms","S_peak_ms"]] = np.nan
    df.loc[np.isnan(qt_raw)  | (qt_raw  < MouseECG.QT_ABS_MIN)  | (qt_raw  > MouseECG.QT_ABS_MAX),
           ["T_peak_s","T_offset_s","T_peak_ms","T_offset_ms"]] = np.nan

    n_complete = int((~df[["PR_ms","QRS_ms","QT_ms"]].isna().any(axis=1)).sum())
    log.info("analyse_intervals: %d beats, %d complete (%.0f%%)",
             n, n_complete, 100.0 * n_complete / max(n, 1))

    # ── Auto-update template from this run ───────────────────────────────
    if not wave_template.confirmed and n_complete >= 10:
        _ms_cols = {
            "P_onset": "P_onset_ms", "P_peak": "P_peak_ms",
            "Q_peak":  "Q_peak_ms",  "S_peak": "S_peak_ms",
            "T_peak":  "T_peak_ms",  "T_offset": "T_offset_ms",
        }
        updated = {}
        for wk, col in _ms_cols.items():
            valid = np.asarray(df[col].dropna(), dtype=float)
            if len(valid) < 5:
                updated[wk] = wave_template.landmarks.get(wk, WaveTemplate.DEFAULTS[wk])
                continue
            med    = float(np.median(valid))
            spread = max(4.0,
                         float(np.percentile(valid, 90) -
                               np.percentile(valid, 10)) * 0.75)
            updated[wk] = (round(med, 1), round(spread, 1))
        wave_template.landmarks = updated
        wave_template.source    = "auto-updated"
        try:
            wave_template.save()
            log.info("analyse_intervals: template auto-updated (%d complete beats)",
                     n_complete)
        except Exception as save_exc:
            log.warning("template auto-save failed: %s", save_exc)

    _prog(100, f"Done — {n_complete}/{n} complete beats")
    return df


# An interval row and an R peak are the same beat when their sample indices
# differ by at most this much (R_peak_s is stored as float seconds, so the
# round trip back to samples can land one sample off).
_ALIGN_TOL_SAMPLES = 1


def align_interval_column_to_peaks(
    ivl:    "Optional[pd.DataFrame]",
    rpeaks: "np.ndarray",
    fs:     float,
    column: str = "PR_ms",
) -> np.ndarray:
    """Map one per-beat column of an ``analyse_intervals()`` result onto *rpeaks*.

    analyse_intervals() does not return one row per R peak: beats too close
    to the signal edges for a full beat window are dropped, and the last
    valid beat has no following RR interval, so the DataFrame is shorter
    than the peak array and cannot be indexed positionally. Rows are matched
    to peaks by sample index instead -- ``round(R_peak_s * fs)`` against the
    sorted peaks, within ±1 sample.

    Parameters
    ----------
    ivl    : analyse_intervals() DataFrame (needs ``R_peak_s`` and *column*).
             May be None or empty.
    rpeaks : R-peak sample indices; any order.
    fs     : sampling frequency (Hz). Raises ValueError if non-positive or
             non-finite.
    column : the per-beat column to align (default ``"PR_ms"``).

    Returns
    -------
    float array of ``len(rpeaks)``, in the same order as *rpeaks*: element
    ``i`` is the row value measured at ``rpeaks[i]``, NaN where no row
    matches that peak, where the matching row has ``accepted == False``
    (rejected in the verifier), or where several rows match the same peak.
    All-NaN when *ivl* is None/empty or lacks ``R_peak_s`` or *column*.
    """
    _validate_sampling_rate(fs)
    peaks = np.asarray(rpeaks, dtype=np.int64).flatten()
    out   = np.full(len(peaks), np.nan, dtype=float)
    if (ivl is None or ivl.empty or len(peaks) == 0
            or "R_peak_s" not in ivl.columns or column not in ivl.columns):
        return out

    r_samp = np.round(pd.to_numeric(ivl["R_peak_s"], errors="coerce")
                      .to_numpy(dtype=float) * fs)
    values = pd.to_numeric(ivl[column], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(r_samp)
    if "accepted" in ivl.columns:
        # Only an explicit False rejects a row -- a missing flag (NaN) does not.
        usable &= ~ivl["accepted"].eq(False).to_numpy(dtype=bool)
    if not usable.any():
        return out
    r_samp = r_samp[usable]
    values = values[usable]

    # Nearest sorted peak for every row: the insertion point or its left
    # neighbour, whichever is closer.
    order  = np.argsort(peaks, kind="stable")
    sorted_peaks = peaks[order].astype(float)
    right  = np.clip(np.searchsorted(sorted_peaks, r_samp), 0, len(sorted_peaks) - 1)
    left   = np.clip(right - 1, 0, len(sorted_peaks) - 1)
    nearest = np.where(np.abs(sorted_peaks[left] - r_samp)
                       <= np.abs(sorted_peaks[right] - r_samp), left, right)
    matched = np.abs(sorted_peaks[nearest] - r_samp) <= _ALIGN_TOL_SAMPLES
    if not matched.any():
        return out
    nearest = nearest[matched]
    values  = values[matched]

    # Two rows on the same peak is ambiguous -- leave that peak NaN rather
    # than pick one of them arbitrarily.
    hit, n_hits = np.unique(nearest, return_counts=True)
    single = np.isin(nearest, hit[n_hits == 1])
    out[order[nearest[single]]] = values[single]
    return out



# ════════════════════════════════════════════════════════════
#  FIGURE / AXES HELPERS
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
#  SEGMENT COMPARISON
# ════════════════════════════════════════════════════════════

def compute_segment_stats(
    signal:  np.ndarray,
    rpeaks:  np.ndarray,
    fs:      float,
    t_lo:    float,
    t_hi:    float,
    label:   str = "Segment",
    lf_band: "tuple[float, float]" = (0.40, 1.50),
    hf_band: "tuple[float, float]" = (1.50, 5.00),
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
    breaks:    "Optional[np.ndarray]" = None,
) -> dict:
    """Compute HRV and cardiac stats for a time window [t_lo, t_hi] seconds.

    Parameters
    ----------
    signal  : Full ECG signal array (float64).
    rpeaks  : Full R-peak sample indices.
    fs      : Sampling rate Hz.
    t_lo    : Segment start (seconds, inclusive).
    t_hi    : Segment end   (seconds, exclusive).
    label   : Display name for this segment.
    lf_band : (lo, hi) Hz LF band for frequency HRV.
    hf_band : (lo, hi) Hz HF band for frequency HRV.
    rr_min_ms, rr_max_ms : RR acceptance bounds (ms).
    breaks  : sample positions of beats removed by artifact correction --
              intervals spanning one are excluded (see rr_interval_series).

    Returns
    -------
    dict with keys: label, n_beats, duration_s, hr_mean, hr_sd, hr_min,
    hr_max, rr_mean, rr_sd, rr_cv, sdnn, rmssd, pnn6, pnn50,
    lf_nu, hf_nu, lf_hf, sd1, sd2, sd_ratio,
    rpeaks_seg, rr_ms, t_rr, error, freq_withheld_reason (why LF/HF are
    NaN despite enough beats, else None), plus the
    ``rr_exclusion_summary()`` keys (n_excluded, ..., rr_min_ms, rr_max_ms).
    """
    _NAN = float("nan")
    _EMPTY: dict = {
        "label":        label,
        "n_beats":      0,
        "duration_s":   float(t_hi - t_lo),
        "coverage_pct": 0.0,
        "hr_mean":  _NAN, "hr_sd":    _NAN, "hr_min":  _NAN, "hr_max":   _NAN,
        "rr_mean":  _NAN, "rr_sd":    _NAN, "rr_min":  _NAN, "rr_max":   _NAN,
        "rr_cv":    _NAN, "sdnn":     _NAN, "rmssd":   _NAN,
        "pnn6":     _NAN, "pnn50":    _NAN,
        "lf_nu":    _NAN, "hf_nu":    _NAN, "lf_hf":   _NAN,
        "sd1":      _NAN, "sd2":      _NAN, "sd_ratio": _NAN,
        "rpeaks_seg": np.array([], dtype=int),
        "rr_ms":      np.array([], dtype=float),
        "t_rr":       np.array([], dtype=float),
        "error":      None,
        "freq_withheld_reason": None,
    }

    # ── 1. Select peaks inside [t_lo, t_hi] ──────────────────────────────
    i_lo = int(t_lo * fs)
    i_hi = int(t_hi * fs)
    mask = (rpeaks >= i_lo) & (rpeaks <= i_hi)
    rp   = np.sort(rpeaks[mask])
    n    = len(rp)

    if n < 10:
        out = dict(_EMPTY); out["error"] = f"Only {n} beats in window (need ≥ 10)"; return out

    # ── 2. RR intervals & acceptance filter (gap-aware, see rr_interval_series)
    ser    = rr_interval_series(rp, fs, rr_min_ms, rr_max_ms, breaks=breaks)
    rr_all = ser["rr_ms"]                              # ms, length n-1
    rr_mask = ser["valid"]
    rr_ok   = ser["rri"]
    t_ok    = ser["rri_time"]                          # time of each interval end
    rr_a, rr_b = successive_valid_pairs(rr_all, rr_mask)
    adjacent_diffs = rr_b - rr_a
    exclusion = rr_exclusion_summary(ser)

    if len(rr_ok) < 8:
        out = dict(_EMPTY); out.update(exclusion)
        out["error"] = "Too many artefact beats after physiological filter"; return out

    hr_all = 60_000.0 / rr_ok

    # ── 3. Time-domain HRV ───────────────────────────────────────────────
    sdnn  = float(np.std(rr_ok, ddof=1))
    rmssd = float(np.sqrt(np.mean(adjacent_diffs ** 2))) if len(adjacent_diffs) else _NAN
    diff_rr = np.abs(adjacent_diffs)
    pnn6    = float(100.0 * np.mean(diff_rr > 6.0))  if len(diff_rr) else _NAN
    pnn50   = float(100.0 * np.mean(diff_rr > 50.0)) if len(diff_rr) else _NAN

    # ── 4. Frequency-domain HRV (Welch on cubic-spline interpolated RR) ──
    # Same withholding rule as analyse_hrv_freq: the spline would otherwise
    # invent RR values across a long excluded stretch.
    lf_nu = hf_nu = lf_hf_r = _NAN
    freq_reason = spectral_withheld_reason(rr_all, rr_mask)
    try:
        from scipy.interpolate import CubicSpline
        from scipy.signal import welch as _welch
        if freq_reason is None and len(t_ok) >= 16 and len(rr_ok) >= 16:
            cs     = CubicSpline(t_ok, rr_ok)
            fs_i   = 20.0           # interpolation rate (Nyquist >> 5 Hz)
            t_unif = np.arange(t_ok[0], t_ok[-1], 1.0 / fs_i)
            if len(t_unif) >= 32:
                rr_i     = cs(t_unif)
                rr_i    -= rr_i.mean()
                nperseg  = min(len(t_unif), int(fs_i * 60))
                freqs, pxx = _welch(rr_i, fs=fs_i, nperseg=nperseg,
                                     noverlap=nperseg // 2, detrend="linear")
                def _band_power(lo: float, hi: float) -> float:
                    m = (freqs >= lo) & (freqs < hi)
                    return float(trapz(pxx[m], freqs[m])) if m.any() else 0.0
                lf_p  = _band_power(*lf_band)
                hf_p  = _band_power(*hf_band)
                total = lf_p + hf_p
                if total > 1e-12:
                    lf_nu    = float(100.0 * lf_p / total)
                    hf_nu    = float(100.0 * hf_p / total)
                    lf_hf_r  = float(lf_p / hf_p) if hf_p > 1e-12 else _NAN
    except Exception as exc:
        log.debug("compute_segment_stats freq HRV: %s", exc)

    # ── 5. Poincaré SD1 / SD2 ────────────────────────────────────────────
    sd1 = sd2 = sd_ratio = _NAN
    try:
        if len(adjacent_diffs) > 1:
            sd1      = float(np.std((rr_b - rr_a) / np.sqrt(2), ddof=1))
            sd2      = float(np.std((rr_b + rr_a) / np.sqrt(2), ddof=1))
            sd_ratio = sd1 / sd2 if sd2 > 1e-9 else _NAN
    except Exception:
        pass

    return {
        **exclusion,
        "label":        label,
        "n_beats":      n,
        "duration_s":   float(t_hi - t_lo),
        "coverage_pct": float(100.0 * n / max(1, len(rpeaks))),
        "hr_mean":  float(np.mean(hr_all)),   "hr_sd":  float(np.std(hr_all, ddof=1)),
        "hr_min":   float(np.min(hr_all)),    "hr_max": float(np.max(hr_all)),
        "rr_mean":  float(np.mean(rr_ok)),    "rr_sd":  float(np.std(rr_ok, ddof=1)),
        "rr_min":   float(np.min(rr_ok)),     "rr_max": float(np.max(rr_ok)),
        "rr_cv":    float(100.0 * np.std(rr_ok) / np.mean(rr_ok)),
        "sdnn":     sdnn,   "rmssd":    rmssd,
        "pnn6":     pnn6,   "pnn50":    pnn50,
        "lf_nu":    lf_nu,  "hf_nu":    hf_nu,  "lf_hf":    lf_hf_r,
        "sd1":      sd1,    "sd2":      sd2,     "sd_ratio": sd_ratio,
        "rpeaks_seg": rp,
        "rr_ms":      rr_ok,
        "t_rr":       t_ok,
        "error":      None,
        "freq_withheld_reason": freq_reason,
    }

