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
from typing import Callable, Optional, cast

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

from ecg.core.models import (
    AnalysisResults, MouseECG, QTC_FORMULA_LABELS, normalize_qtc_formula,
)
from ecg.core.wave_template import WaveTemplate, detect_waves_on_beat, format_windows

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


# Heart-rate asymmetry conventions, as stated in the Summary report and
# docs -- the same definitions as NeuroKit's HRV_PI / HRV_GI:
#   Porta index PI (Porta et al. 2008) = number of points BELOW the identity
#     line of the Poincaré plot (ΔRR < 0, accelerations) / number of points
#     OFF the line (ΔRR ≠ 0) × 100. Points on the line (ΔRR = 0) are left
#     out of the denominator; symmetry → 50 %.
#   Guzik index GI (Guzik et al. 2006) = Σ distance to the identity line of
#     the points ABOVE it (ΔRR > 0, decelerations) / Σ distance of all
#     off-line points × 100 (distance = |ΔRR| / √2); symmetry → 50 %.
RR_ASYMMETRY_TEXT = (
    "Porta PI = points below the identity line (ΔRR < 0) / off-line points "
    "(ΔRR ≠ 0); Guzik GI = Σ distance of points above the line (ΔRR > 0) / "
    "Σ distance of off-line points; successive valid ΔRR, symmetry → 50 %")


def rr_asymmetry_indices(rr_a: np.ndarray, rr_b: np.ndarray) -> dict:
    """Porta and Guzik heart-rate asymmetry indices (%) of the Poincaré
    pairs (RR[i], RR[i + 1]) = (*rr_a*, *rr_b*), e.g. from
    ``successive_valid_pairs`` -- conventions in RR_ASYMMETRY_TEXT.

    Returns porta_pct, guzik_pct (NaN when no pair is off the identity
    line), n_dec (ΔRR > 0), n_acc (ΔRR < 0), n_zero (ΔRR = 0), n_off
    (n_dec + n_acc) and n_tot (all pairs)."""
    a = np.asarray(rr_a, dtype=float).flatten()
    b = np.asarray(rr_b, dtype=float).flatten()
    n = min(len(a), len(b))
    diff = b[:n] - a[:n]
    diff = diff[np.isfinite(diff)]
    n_dec = int(np.sum(diff > 0))
    n_acc = int(np.sum(diff < 0))
    n_off = n_dec + n_acc
    dist_all = float(np.sum(np.abs(diff)))      # the 1/√2 factor cancels
    porta = 100.0 * n_acc / n_off if n_off else float("nan")
    guzik = (100.0 * float(np.sum(diff[diff > 0])) / dist_all
             if dist_all > 0 else float("nan"))
    return {"porta_pct": porta, "guzik_pct": guzik,
            "n_dec": n_dec, "n_acc": n_acc, "n_zero": int(len(diff) - n_off),
            "n_off": n_off, "n_tot": int(len(diff))}


# ════════════════════════════════════════════════════════════
#  ONE DEFINITION PER METRIC
# ════════════════════════════════════════════════════════════
#
# Mean heart rate and Poincaré SD1/SD2 used to be computed several ways in
# different views (mean of instantaneous HR vs 60000 / mean RR; SD1 from
# NeuroKit on the first 1000 beats, from inline formulas in Rolling HRV and
# Compare Segments ...), so the same recording could show different numbers
# for the "same" metric. Every view now calls the two functions below; the
# Ledoit-Wolf / robust Poincaré estimators (compute_poincare_metrics,
# ecg.core.autonomic) remain explicitly named variants (SD1_LW, ...).

def mean_hr_bpm(rr_ms: np.ndarray) -> float:
    """Mean heart rate (bpm) of the given NN intervals: 60000 / mean NN.

    The single mean-HR definition of the app (Summary, Excel, Compare
    Segments, Rolling, Epochs, tachogram titles). It is NOT the mean of the
    instantaneous HR 60000 / RR_i, which is biased upwards by RR variability
    (Jensen's inequality). Non-finite and non-positive intervals are
    ignored; NaN when none remain."""
    rr = np.asarray(rr_ms, dtype=np.float64).flatten()
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if len(rr) == 0:
        return float("nan")
    return float(60_000.0 / rr.mean())


# Percentiles of RR giving the instantaneous-HR "min" / "max": robust to the
# odd missed or extra beat that slips through the RR bounds.
HR_RANGE_PCTL = (2.0, 98.0)
HR_RANGE_TEXT = "2nd–98th percentile of instantaneous HR"


def instantaneous_hr_stats(rr_ms: np.ndarray) -> "dict[str, float]":
    """Spread of the instantaneous HR 60000 / RR_i of the given NN intervals:
    {"sd", "min", "max"} in bpm.

    The single definition of the app (Summary, KPIs, Excel, Compare
    Segments): "sd" is the sample SD (ddof = 1, like SDNN); "min" / "max"
    are the 2nd / 98th percentiles of the instantaneous HR (HR_RANGE_PCTL,
    i.e. 60000 / the 98th / 2nd RR percentile), not the absolute extremes.
    Non-finite and non-positive intervals are ignored; NaN when too few
    remain (SD needs two)."""
    rr = np.asarray(rr_ms, dtype=np.float64).flatten()
    rr = rr[np.isfinite(rr) & (rr > 0)]
    nan = float("nan")
    if len(rr) == 0:
        return {"sd": nan, "min": nan, "max": nan}
    lo, hi = HR_RANGE_PCTL
    return {
        "sd":  float(np.std(60_000.0 / rr, ddof=1)) if len(rr) > 1 else nan,
        "min": float(60_000.0 / np.percentile(rr, hi)),
        "max": float(60_000.0 / np.percentile(rr, lo)),
    }


def poincare_sd1_sd2(
    rr_ms: np.ndarray,
    valid: "Optional[np.ndarray]" = None,
) -> "dict[str, float | int]":
    """Classic Poincaré SD1 / SD2 (ms) of an RR series -- the one definition
    used by the Non-linear panel, Rolling HRV, Compare Segments and the
    Summary/exports.

    Points are the successive pairs (RR_i, RR_i+1) of two ADJACENT valid
    intervals (``successive_valid_pairs``: a pair never straddles an
    excluded interval). SD1 = SD of (RR_i - RR_i+1)/√2 and SD2 = SD of
    (RR_i + RR_i+1)/√2 (sample SD, ddof 1) -- i.e. the spread of the pairs'
    sample covariance across / along the identity line (Brennan et al.
    2001), identical to NeuroKit2's ``hrv_nonlinear`` on the same pairs.

    Returns ``{"sd1", "sd2", "sd1_sd2", "n_pairs"}``; SD1/SD2 are NaN with
    fewer than 2 pairs, SD1/SD2 ratio NaN unless both are positive."""
    rr_a, rr_b = successive_valid_pairs(rr_ms, valid)
    nan = float("nan")
    n = int(len(rr_a))
    if n < 2:
        return {"sd1": nan, "sd2": nan, "sd1_sd2": nan, "n_pairs": n}
    sd1 = float(np.std((rr_a - rr_b) / np.sqrt(2.0), ddof=1))
    sd2 = float(np.std((rr_a + rr_b) / np.sqrt(2.0), ddof=1))
    ratio = sd1 / sd2 if (np.isfinite(sd1) and np.isfinite(sd2) and sd2 > 0) else nan
    return {"sd1": sd1, "sd2": sd2, "sd1_sd2": ratio, "n_pairs": n}


# ── Frequency-domain HRV: one spectrum for the plot and the numbers ─────
#
# The PSD panel used to draw its own Welch spectrum while VLF/LF/HF/LF-HF
# came from nk.hrv_frequency (100 Hz resampling, PSD normalised to its
# maximum -- so "VLF/LF/HF" were neither ms² nor n.u.), and Compare
# Segments ran a third Welch pipeline. ``hrv_spectrum`` is now the only
# spectral estimator: analyse_hrv_freq (Frequency panel, Rolling HRV),
# compute_segment_stats (Compare Segments) and the PSD plot all use the
# spectrum it returns, so the plotted curve is exactly the one that is
# integrated into the band powers.

def hrv_spectrum(
    rri:           np.ndarray,
    rri_time:      np.ndarray,
    interp_fs:     float = MouseECG.PSD_INTERP_FS,
    target_res_hz: float = 0.02,
) -> "Optional[dict]":
    """Welch PSD (ms²/Hz) of the valid RR intervals *rri* (ms) placed at their
    true end times *rri_time* (s).

    The intervals are resampled at *interp_fs* Hz with a cubic spline on the
    real time axis (an excluded interval leaves a hole the spline bridges --
    see ``spectral_withheld_reason`` for when that is refused), the mean is
    removed, and Welch's method is applied with a Hann window, 75 % overlap
    and nperseg = clip(interp_fs / target_res_hz, 256, N // 2) samples
    (≈ 0.02 Hz resolution on long recordings, coarser on short ones).

    Returns ``{"freqs", "psd", "nperseg", "noverlap", "n_points",
    "interp_fs", "df_hz"}`` or None when fewer than 32 resampled points (or
    4 intervals) are available.
    """
    rr = np.asarray(rri, dtype=np.float64).flatten()
    t  = np.asarray(rri_time, dtype=np.float64).flatten()
    if len(rr) < 4 or len(rr) != len(t):
        return None
    order = np.argsort(t, kind="stable")
    t, rr = t[order], rr[order]
    keep = np.r_[True, np.diff(t) > 0]          # CubicSpline needs strictly increasing x
    t, rr = t[keep], rr[keep]
    if len(rr) < 4:
        return None
    t_uni = np.arange(t[0], t[-1], 1.0 / interp_fs)
    n_pts = len(t_uni)
    if n_pts < 32:
        return None
    rr_i = CubicSpline(t, rr)(t_uni)
    rr_i = rr_i - rr_i.mean()
    nperseg  = int(np.clip(interp_fs / target_res_hz, 256, n_pts // 2))
    noverlap = int(nperseg * 0.75)
    freqs, psd = _scipy_welch(rr_i, fs=interp_fs, nperseg=nperseg,
                              noverlap=noverlap, window="hann",
                              scaling="density")
    return {
        "freqs":     np.asarray(freqs, dtype=np.float64),
        "psd":       np.asarray(psd, dtype=np.float64),
        "nperseg":   nperseg,
        "noverlap":  noverlap,
        "n_points":  n_pts,
        "interp_fs": float(interp_fs),
        "df_hz":     float(freqs[1] - freqs[0]) if len(freqs) > 1 else float("nan"),
    }


def band_mask(freqs: np.ndarray, band: "tuple[float, float]") -> np.ndarray:
    """Bins of *freqs* inside the half-open band [lo, hi) -- the bins whose
    power is integrated into that band (and shaded on the PSD plot)."""
    f = np.asarray(freqs, dtype=np.float64)
    return (f >= float(band[0])) & (f < float(band[1]))


def band_power(freqs: np.ndarray, psd: np.ndarray, band: "tuple[float, float]") -> float:
    """Absolute power (ms²) of *band*: trapezoidal integral of the PSD
    (ms²/Hz) over ``band_mask``; 0.0 when the band holds fewer than 2 bins."""
    m = band_mask(freqs, band)
    if int(m.sum()) < 2:
        return 0.0
    return float(trapz(np.asarray(psd)[m], np.asarray(freqs)[m]))


def spectral_band_metrics(
    spectrum: dict,
    vlf: "tuple[float, float]" = MouseECG.VLF,
    lf:  "tuple[float, float]" = MouseECG.LF,
    hf:  "tuple[float, float]" = MouseECG.HF,
) -> "dict[str, float]":
    """VLF/LF/HF band powers and derived ratios of an ``hrv_spectrum`` result.

    HRV_VLF, HRV_LF, HRV_HF, HRV_TP (= VLF + LF + HF) are absolute powers in
    ms²; HRV_LFn = LF / (LF + HF) and HRV_HFn = HF / (LF + HF) are the
    normalised units of the Task Force (1996) -- LF / (TP - VLF) -- as
    fractions (×100 for n.u.); HRV_LFHF = LF / HF; HRV_LnHF = ln(HF);
    HRV_LFpeak / HRV_HFpeak the frequency (Hz) of the PSD maximum in each
    band. Ratios are NaN when their denominator is 0."""
    f, p = spectrum["freqs"], spectrum["psd"]
    nan = float("nan")
    p_vlf, p_lf, p_hf = band_power(f, p, vlf), band_power(f, p, lf), band_power(f, p, hf)
    lfhf_sum = p_lf + p_hf

    def _peak(band: "tuple[float, float]") -> float:
        m = band_mask(f, band)
        if not m.any() or not np.any(p[m] > 0):
            return nan
        return float(f[m][int(np.argmax(p[m]))])

    return {
        "HRV_VLF":    p_vlf,
        "HRV_LF":     p_lf,
        "HRV_HF":     p_hf,
        "HRV_TP":     p_vlf + p_lf + p_hf,
        "HRV_LFHF":   p_lf / p_hf if p_hf > 0 else nan,
        "HRV_LFn":    p_lf / lfhf_sum if lfhf_sum > 0 else nan,
        "HRV_HFn":    p_hf / lfhf_sum if lfhf_sum > 0 else nan,
        "HRV_LnHF":   float(np.log(p_hf)) if p_hf > 0 else nan,
        "HRV_LFpeak": _peak(lf),
        "HRV_HFpeak": _peak(hf),
    }


# Minimum valid-interval count for any meaningful spectral estimate
MIN_BEATS_SPECTRAL = 60   # ~7–15 s of mouse ECG at typical HR

SPECTRAL_METHOD_TEXT = (
    "Welch PSD of the valid RR intervals resampled at 20 Hz (cubic spline "
    "on their true times; Hann, 75 % overlap, ≈0.02 Hz resolution); VLF / "
    "LF / HF = absolute band powers (ms²), LFn / HFn = LF or HF / (LF + HF) "
    "(Task Force n.u.); the PSD plot shows this same spectrum")


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


# ── Analysis provenance (what a reader needs to reproduce the numbers) ──
# Shared by the Summary text report, the Excel Cover sheet, the Prism
# project notes and the PDF report so they all disclose the same thing.

def analysis_window_span(
    window:       "Optional[tuple]",
    signal_end_s: "Optional[float]" = None,
) -> "tuple[float, Optional[float]]":
    """(start, end) in s of the analysis window *window* (the (t_start,
    t_end) the results were computed on; 0 = signal start / end, as in
    AnalysisState). end is None when the window runs to the end of the
    signal and *signal_end_s* is unknown. When *signal_end_s* is known both
    bounds are clamped to it: a window entered past the end of the recording
    only covers the signal that exists."""
    t0, t1 = (window if window is not None else (0.0, 0.0))
    t0 = max(0.0, float(t0 or 0.0))
    t1 = float(t1 or 0.0)
    known = signal_end_s is not None and np.isfinite(signal_end_s)
    if t1 > 0:
        end: "Optional[float]" = t1
    elif known:
        end = float(signal_end_s)
    else:
        end = None
    if known:
        sig_end = max(0.0, float(signal_end_s))
        t0 = min(t0, sig_end)
        if end is not None:
            end = min(end, sig_end)
    return t0, end


def analysed_beat_span_s(results: "Optional[dict]") -> "Optional[float]":
    """Time from the first to the last analysed R peak (s): the sum of every
    RR interval of *results* (used for HRV or excluded). None without RR."""
    ts = rr_timeseries(results)
    if ts.empty:
        return None
    rr = ts["RR_ms"].to_numpy(dtype=np.float64)
    rr = rr[np.isfinite(rr)]
    return float(rr.sum()) / 1000.0 if len(rr) else None


def analysed_duration_s(
    results:      "Optional[dict]",
    window:       "Optional[tuple]" = None,
    signal_end_s: "Optional[float]" = None,
) -> "Optional[float]":
    """Analysed duration (s): window end − start, or -- when the window end
    is unknown -- the span of the analysed beats. Not the absolute time of
    the last beat, which also counts everything before the window start."""
    t0, t1 = analysis_window_span(window, signal_end_s)
    if t1 is not None and t1 > t0:
        return t1 - t0
    return analysed_beat_span_s(results)


def analysis_window_rows(
    results:      "Optional[dict]",
    window:       "Optional[tuple]" = None,
    signal_end_s: "Optional[float]" = None,
) -> "list[tuple[str, str]]":
    """(label, value) rows: the analysis window the results were computed
    on and the analysed duration (window end − start, plus the span of the
    analysed beats, which starts at the first R peak inside the window)."""
    t0, t1 = analysis_window_span(window, signal_end_s)
    full = (window is None or (float(window[0] or 0.0) <= 0
                               and float(window[1] or 0.0) <= 0))
    span = f"{t0:.1f} – {t1:.1f} s" if t1 is not None else f"{t0:.1f} s – end"
    # A window entered past the end of the signal is clamped to it (see
    # analysis_window_span): say so, with the value that was entered.
    entered_end = float(window[1] or 0.0) if window is not None else 0.0
    if (not full and t1 is not None and entered_end > 0
            and entered_end > t1 + 1e-9):
        span += f"  (end {entered_end:.1f} s entered, past the signal end)"
    rows = [("Analysis window",
             f"full recording ({span})" if full else span)]
    beats = analysed_beat_span_s(results)
    parts: "list[str]" = []
    if t1 is not None and t1 > t0:
        parts.append(f"{t1 - t0:.1f} s (window end − start)")
    if beats is not None:
        parts.append(f"{beats:.1f} s first → last analysed R peak")
    rows.append(("Analysed duration", "; ".join(parts) if parts else "—"))
    return rows


def artifact_correction_text(
    report:  "Optional[dict]",
    results: "Optional[dict]" = None,
) -> str:
    """One-line disclosure of the artifact report in effect (beats removed
    from the peak set, by type) -- 'not applied' without one.

    Artifact correction runs on the whole recording BEFORE the analysis
    window is applied, so the counts are said to be whole-recording ones
    (next to a windowed 'Analysis window' / 'N beats' they would otherwise
    read as analysed beats). The 'excluded from HRV' clause is added only
    when *results* carry the RR validity mask (``rr_valid`` /
    ``n_spanning_removed``): results from older versions bridged the
    intervals around a removed beat instead."""
    if not report:
        return "not applied"
    n_in  = int(report.get("n_in", 0) or 0)
    n_out = int(report.get("n_out", 0) or 0)
    removed = max(0, n_in - n_out)
    if not removed:
        return "applied — no beat removed (whole recording)"
    text = (f"{removed} of {n_in} beats removed (whole recording, before "
            f"windowing: {int(report.get('n_nonphysio', 0) or 0)} non-physiological, "
            f"{int(report.get('n_ectopic', 0) or 0)} ectopic, "
            f"{int(report.get('n_duplicate', 0) or 0)} duplicates)")
    r  = results or {}
    hr = r.get("hr") or {}
    if r.get("rr_valid") is not None or "n_spanning_removed" in hr:
        text += "; RR intervals spanning a removed beat are excluded from HRV"
    elif results is not None:
        text += ("; older results: RR intervals around a removed beat were "
                 "bridged, not excluded — re-run Analyze")
    return text


def signal_crop_info(
    t_start:         float,
    t_end:           float,
    file_duration_s: "Optional[float]" = None,
) -> dict:
    """The load-time crop ('Start crop (s)' / 'End crop (s)', FilterParams
    t_start / t_end; 0 = file start / end) as {"t_start", "t_end",
    "file_duration_s"} in FILE time, the end clamped to the file length
    when it is known (None = to the end of an unknown-length file)."""
    t0 = max(0.0, float(t_start or 0.0))
    t1 = float(t_end or 0.0)
    dur = (float(file_duration_s) if file_duration_s is not None
           and np.isfinite(file_duration_s) else None)
    end: "Optional[float]" = t1 if t1 > 0 else dur
    if dur is not None and end is not None:
        end = min(end, dur)
    return {"t_start": t0, "t_end": end, "file_duration_s": dur}


def signal_crop_text(crop: "Optional[dict]") -> "Optional[str]":
    """The 'Signal crop' provenance value for *crop* (signal_crop_info), or
    None when it was not recorded. Every time in the reports (analysis
    window, R peaks, events) is relative to the crop start."""
    if not crop:
        return None
    t0  = float(crop.get("t_start") or 0.0)
    t1  = crop.get("t_end")
    dur = crop.get("file_duration_s")
    of  = f" of a {float(dur):.1f} s file" if dur is not None else ""
    if t0 <= 0 and (t1 is None or dur is None or float(t1) >= float(dur) - 1e-9):
        return f"none (whole file{of}; times are file times)"
    end = f"{float(t1):.1f} s" if t1 is not None else "end"
    return (f"file time {t0:.1f} – {end}{of}; all times in this report are "
            f"relative to the crop start ({t0:.1f} s in the file)")


def qtc_formula_text(formula: "Optional[str]") -> str:
    """'<formula label> with the preceding RR  (report uncorrected QT
    alongside)' -- the QTc label of every report and export, from the
    formula actually used (a QTC_FORMULA_LABELS key or label)."""
    label = QTC_FORMULA_LABELS[normalize_qtc_formula(formula or "mitchell")]
    return f"{label} with the preceding RR  (report uncorrected QT alongside)"


def analysis_provenance_rows(
    results:         "Optional[dict]",
    window:          "Optional[tuple]" = None,
    signal_end_s:    "Optional[float]" = None,
    artifact_report: "Optional[dict]" = None,
    filter_text:     "Optional[str]" = None,
    qtc_formula:     "Optional[str]" = None,
    signal_crop:     "Optional[dict]" = None,
) -> "list[tuple[str, str]]":
    """(label, value) rows of everything needed to reproduce *results*:
    the load-time signal crop (when *signal_crop*, a signal_crop_info dict,
    is given -- every time is relative to its start), analysis window and
    analysed duration, filters (when *filter_text* is given), RR bounds and
    exclusions, artifact correction, QTc formula (when given) and the
    interval definitions / wave template (``interval_disclosure_rows``)."""
    r = results or {}
    rows: "list[tuple[str, str]]" = []
    crop_text = signal_crop_text(signal_crop)
    if crop_text:
        rows.append(("Signal crop", crop_text))
    rows += analysis_window_rows(r, window, signal_end_s)
    if filter_text:
        rows.append(("Filters", filter_text))
    rows += rr_disclosure_rows(rr_exclusion_report(r))
    rows.append(("Artifact correction", artifact_correction_text(artifact_report, r)))
    if qtc_formula is not None:
        rows.append(("QTc formula", qtc_formula_text(qtc_formula)))
    rows += interval_disclosure_rows(r.get("intervals"))
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


LEGACY_SPECTRAL_TEXT = (
    "legacy NeuroKit spectrum from an older version (100 Hz resampling, "
    "max-normalised PSD): VLF / LF / HF in arbitrary units, LFn / HFn = LF "
    "or HF / total power -- not comparable with current results; re-run "
    "Frequency HRV")


def hrv_freq_is_legacy(results: "Optional[dict]") -> bool:
    """True when *results* hold a frequency-domain HRV computed by an older
    version (nk.hrv_frequency): a non-empty table without the ``method``
    attr analyse_hrv_freq() now stores. Its VLF/LF/HF are arbitrary units
    and its LFn = LF / TP, so views must not label them ms² / Task Force
    n.u. (LEGACY_SPECTRAL_TEXT says what they are instead)."""
    fd = (results or {}).get("hrv_freq")
    if fd is None or getattr(fd, "empty", True):
        return False
    attrs = getattr(fd, "attrs", {}) or {}
    return not (attrs.get("method") or attrs.get("psd_freqs") is not None)


def migrate_legacy_results(results: "Optional[dict]") -> "list[str]":
    """Bring results restored from an older session in line with the
    current metric definitions, in place; returns a note per change.

    * ``hr`` mean / std / min / max are recomputed from ``rr_df`` (the valid
      intervals they always came from): older versions stored the mean of
      the instantaneous HR rather than 60000 / mean NN (mean_hr_bpm) and a
      population SD. Idempotent for current results.
    * A legacy NeuroKit frequency table (hrv_freq_is_legacy) cannot be
      recomputed without the peaks: it is kept, and every view labels it
      with LEGACY_SPECTRAL_TEXT instead of ms² / n.u. (noted here)."""
    notes: "list[str]" = []
    if not results:
        return notes
    hr  = results.get("hr")
    rdf = results.get("rr_df")
    if (isinstance(hr, dict) and isinstance(rdf, pd.DataFrame) and not rdf.empty
            and "RR_ms" in rdf.columns):
        rr = rdf["RR_ms"].to_numpy(dtype=float)
        new = {"mean": mean_hr_bpm(rr)}
        _ihr = instantaneous_hr_stats(rr)
        new.update(std=_ihr["sd"], min=_ihr["min"], max=_ihr["max"])
        old_mean = hr.get("mean")
        for k, v in new.items():
            if np.isfinite(v):
                hr[k] = v
        try:
            if old_mean is not None and abs(float(old_mean) - hr["mean"]) > 1e-6:
                notes.append(f"mean HR recomputed as 60000 / mean NN "
                             f"({float(old_mean):.1f} → {hr['mean']:.1f} bpm)")
        except (TypeError, ValueError):
            pass
    if hrv_freq_is_legacy(results):
        notes.append("frequency HRV is a " + LEGACY_SPECTRAL_TEXT)
    return notes


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
    nan = float("nan")
    pc  = poincare_sd1_sd2(series["rr_ms"], series["valid"])
    sd1, sd2 = float(pc["sd1"]), float(pc["sd2"])

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


# Per-beat delineation window (analyse_intervals): each side may reach this
# fraction of the RR on that side -- far enough for the T end at mouse heart
# rates (QT ~ 60 ms at RR 120 ms), short of the neighbouring QRS.
BEAT_WINDOW_RR_FRAC: float = 0.70
_BEAT_WINDOW_MIN_SAMPLES: int = 20


def beat_window_sides(
    rpeaks:    np.ndarray,
    fs:        float,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
) -> "tuple[np.ndarray, np.ndarray]":
    """(pre, post) window sides in samples for every peak of the SORTED
    *rpeaks*: before R up to ``BEAT_WINDOW_RR_FRAC`` x the distance to the
    previous peak, after R up to that fraction of the distance to the next
    one, each capped at ``BEAT_HALF_WIN_S`` (the cap alone without a
    neighbour). A neighbour closer than *rr_min_ms* is not a heart beat
    (e.g. a spurious peak on the T wave): the next one beyond it sizes
    the window instead. Each beat depends only on its own RRs, so one
    premature beat shortens its own and its neighbours' windows, not the
    whole recording's.
    """
    rp    = np.asarray(rpeaks, dtype=np.int64).flatten()
    cap   = int(MouseECG.BEAT_HALF_WIN_S * fs)
    pre   = np.full(len(rp), cap, dtype=np.int64)
    post  = np.full(len(rp), cap, dtype=np.int64)
    if len(rp) < 2:
        return pre, post
    gap   = max(1, int(np.ceil(rr_min_ms / 1000.0 * fs)))
    i_prev = np.searchsorted(rp, rp - gap, side="right") - 1
    i_next = np.searchsorted(rp, rp + gap, side="left")
    has_p  = i_prev >= 0
    has_n  = i_next < len(rp)
    d_prev = np.where(has_p, rp - rp[np.clip(i_prev, 0, None)], 0)
    d_next = np.where(has_n, rp[np.clip(i_next, None, len(rp) - 1)] - rp, 0)
    def _side(d: np.ndarray) -> np.ndarray:   # +1e-6: 0.7 x 90 is not 62.99
        return np.minimum(cap, np.floor(BEAT_WINDOW_RR_FRAC * d + 1e-6).astype(np.int64))

    pre[has_p]  = _side(d_prev[has_p])
    post[has_n] = _side(d_next[has_n])
    floor = min(cap, _BEAT_WINDOW_MIN_SAMPLES)
    return np.maximum(pre, floor), np.maximum(post, floor)


def beat_window_text(info: "Optional[dict]") -> str:
    """One-line description of ``analyse_intervals``'s
    ``attrs["beat_window"]`` (exports, Summary report)."""
    if not isinstance(info, dict):
        return ""
    txt = (f"per beat, up to ±{info.get('max_ms', 100):g} ms, at most "
           f"{100 * info.get('rr_frac', BEAT_WINDOW_RR_FRAC):.0f} % of the "
           "preceding RR before R and of the following RR after R")
    n_pre, n_post = int(info.get("n_short_pre", 0)), int(info.get("n_short_post", 0))
    if n_pre or n_post:
        txt += (f"; shortened on {n_pre} / {n_post} of {int(info.get('n_beats', 0))} "
                f"beats before / after R (shortest {info.get('min_pre_ms', 0):g} / "
                f"{info.get('min_post_ms', 0):g} ms) — a P onset or T end outside "
                "a beat's window is left empty")
    return txt


def _scored_beats(
    n_samples: int,
    rpeaks:    np.ndarray,
    fs:        float,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
) -> "tuple[int, np.ndarray]":
    """(half_win, valid_rp) of compute_beat_correlation: the beat half-window
    (samples) and the peaks far enough from the signal edges to be scored.
    Deterministic for the same signal length, peaks, fs and RR bounds, so
    rebuild_beat_corr_times can recover which beats an older result scored.
    """
    rpeaks = np.asarray(rpeaks)
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
    valid_rp = rpeaks[(rpeaks - half_win >= 0) & (rpeaks + half_win < n_samples)]
    return half_win, valid_rp


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
        half_win, valid_rp = _scored_beats(len(cleaned), rpeaks, fs, rr_min_ms, rr_max_ms)
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


def rebuild_beat_corr_times(
    r: "Optional[dict]", n_samples: int, rpeaks: np.ndarray, fs: float,
) -> bool:
    """Fill r["beat_corr_t_s"] for results from older versions, which did
    not record which beats compute_beat_correlation scored, by re-running
    its edge selection (_scored_beats) on the analysed *rpeaks* and the
    length of the signal they were scored on (*n_samples*), with the RR
    bounds recorded in r["hr"] (or *rpeaks* themselves when there is one
    score per peak). Accepted only when it yields exactly len(beat_corr)
    beats; True when the times were filled. A no-op (False)
    for current results, which already record them."""
    if not r or r.get("beat_corr_t_s") is not None:
        return False
    bc = r.get("beat_corr")
    if bc is None or len(bc) == 0 or rpeaks is None or len(rpeaks) == 0 or not fs:
        return False
    hr = r.get("hr") or {}
    rp = np.sort(np.asarray(rpeaks, dtype=np.int64).flatten())
    if len(rp) == len(bc):
        # One score per peak: no edge beat was left out, pairing is exact.
        r["beat_corr_t_s"] = rp.astype(float) / float(fs)
        return True
    _, valid_rp = _scored_beats(int(n_samples), rp, float(fs),
                                float(hr.get("rr_min_ms", MouseECG.RR_MIN_MS)),
                                float(hr.get("rr_max_ms", MouseECG.RR_MAX_MS)))
    if len(valid_rp) != len(bc):
        return False
    r["beat_corr_t_s"] = valid_rp.astype(float) / float(fs)
    return True


def _scored_beat_times(
    r: dict, peak_times_s: "Optional[np.ndarray]" = None,
) -> "Optional[np.ndarray]":
    """Time (s) of each beat_corr value, or None when it cannot be known.

    r["beat_corr_t_s"] when recorded (current results, or older ones
    rebuilt by rebuild_beat_corr_times). Otherwise the pairing by position
    is exact only when no edge beat was left out, i.e. when there are as
    many scores as analysed peaks -- *peak_times_s*, or the peaks implied
    by the contiguous RR series (rr_t_s, rr_ms). Any other count is
    ambiguous (which edge beats were dropped is unknown): None, and every
    beat-quality output (curve, tile, RR CSV column) is left out alike."""
    bc = r.get("beat_corr")
    if bc is None or len(bc) == 0:
        return None
    n = len(bc)
    t = r.get("beat_corr_t_s")
    if t is not None:
        t = np.asarray(t, dtype=float).flatten()
        return t if len(t) == n else None
    if peak_times_s is not None:
        pt = np.asarray(peak_times_s, dtype=float).flatten()
        return pt if len(pt) == n else None
    rr, rr_t = r.get("rr_ms"), r.get("rr_t_s")
    if rr is None or rr_t is None or len(rr) == 0 or len(rr_t) != len(rr):
        return None
    rr = np.asarray(rr, dtype=float).flatten()
    rr_t = np.asarray(rr_t, dtype=float).flatten()
    if len(rr) + 1 != n or not np.isfinite(rr[0]):
        return None
    return np.concatenate([[rr_t[0] - rr[0] / 1000.0], rr_t])


def beat_quality_times(
    r: dict, peak_times_s: "Optional[np.ndarray]" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """(time_s, corr) of every beat scored against the template.

    compute_beat_correlation leaves out peaks too close to the signal edges
    for a full beat window, so beat_corr[i] is not the i-th peak: the times
    come from the beats actually scored (_scored_beat_times). Results from
    older versions did not record them: *peak_times_s* (the analysed peaks)
    is used only when there are exactly as many as scores; otherwise both
    arrays are empty rather than shifted by the excluded leading beats."""
    empty = np.array([], dtype=float)
    t = _scored_beat_times(r, peak_times_s)
    if t is None:
        return empty, empty.copy()
    return t, np.asarray(r["beat_corr"], dtype=float).flatten()


def beat_quality_at(r: dict, times_s: np.ndarray) -> np.ndarray:
    """Template correlation of the scored beat at each of *times_s* (s),
    e.g. the start beat of each RR interval; NaN where no scored beat lies
    within 1 µs (an unscored edge beat, or results that do not tell which
    beats were scored, see _scored_beat_times)."""
    t = np.asarray(times_s, dtype=float).flatten()
    out = np.full(len(t), np.nan)
    tb = _scored_beat_times(r)
    if tb is None or not len(t):
        return out
    order = np.argsort(tb, kind="stable")
    tb = tb[order]
    bc = np.asarray(r["beat_corr"], dtype=float).flatten()[order]
    idx = np.clip(np.searchsorted(tb, t - 1e-6), 0, len(tb) - 1)
    hit = np.abs(tb[idx] - t) < 1e-6
    out[hit] = bc[idx[hit]]
    return out


def rr_quality_at(r: dict, rr_end_s: np.ndarray, rr_ms: np.ndarray) -> np.ndarray:
    """Template correlation of the LEADING beat of each RR interval (end
    time *rr_end_s*, length *rr_ms*) -- the one convention shared by the
    "Poor-shape-match time" tile (noisy_time_pct) and the RR CSV "quality"
    column. NaN for an interval starting at an unscored edge beat."""
    end = np.asarray(rr_end_s, dtype=float).flatten()
    rr = np.asarray(rr_ms, dtype=float).flatten()
    return beat_quality_at(r, end - rr / 1000.0)


def noisy_time_pct(r: dict, threshold: float = 0.90) -> float:
    """Percent of the RR time whose LEADING beat correlates below
    *threshold* with the template (the "Poor-shape-match time" tile).
    Each RR interval is attributed to the scored beat it starts at
    (rr_quality_at); intervals starting at an unscored edge beat are
    left out of both sums. NaN when nothing can be attributed, including
    older results that do not tell which beats were scored."""
    rr = r.get("rr_ms")
    bc = r.get("beat_corr")
    rr_t = r.get("rr_t_s")
    if (rr is None or bc is None or rr_t is None or len(rr) == 0 or len(bc) == 0
            or len(rr_t) != len(rr)):
        return float("nan")
    rr = np.asarray(rr, dtype=float).flatten()
    q = rr_quality_at(r, rr_t, rr)
    hit = np.isfinite(q) & np.isfinite(rr)
    rr_slice, corr_slice = rr[hit], q[hit]
    total_ms = float(np.sum(rr_slice))
    if not total_ms > 0:
        return float("nan")
    return 100.0 * float(np.sum(rr_slice[corr_slice < threshold])) / total_ms


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

    # HR stats from the valid intervals only. "mean" is 60000 / mean NN
    # (mean_hr_bpm), "std" / "min" / "max" the sample SD and 2nd / 98th
    # percentiles of the instantaneous HR (instantaneous_hr_stats): the
    # app-wide definitions, shared with Compare Segments.
    _ihr = instantaneous_hr_stats(rr_clean)
    hr = {
        "mean": mean_hr_bpm(rr_clean),
        "min":  _ihr["min"],
        "max":  _ihr["max"],
        "std":  _ihr["sd"],
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
    # Time of each scored beat: peaks too close to the signal edges for a
    # full beat window are not scored, so beat_corr[i] is NOT rpeaks[i].
    beat_corr_t_s = (np.asarray(_bc["valid_rp"], dtype=float) / fs
                     if beat_corr is not None else None)

    _prog(100, "Core analysis done")
    return cast(AnalysisResults, dict(
        hr=hr, rr_ms=rr_ms, rr_valid=rr_ok_mask, rr_t_s=rr_series["t_s"], rr_df=rr_df,
        hrv_time=hrv_time,
        hrv_freq=pd.DataFrame(), hrv_nonlin=pd.DataFrame(),
        intervals=pd.DataFrame({"RR_ms": rr_ms}),
        beat_template=beat_template, beat_time=beat_time,
        beat_matrix=beat_matrix,   # ≤60 rows — negligible memory
        beat_sd=beat_sd,
        beat_corr=beat_corr, beat_corr_t_s=beat_corr_t_s, peak_amps=peak_amps,
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
    bands:       "Optional[tuple]" = None,
) -> "pd.DataFrame":
    """Frequency-domain HRV (VLF, LF, HF, LF/HF) with mouse-specific bands.

    NeuroKit2's default bands (LF: 0.04–0.15 Hz, HF: 0.15–0.4 Hz) are tuned
    for humans.  Mouse HRV uses completely different ranges (Thireau et al.
    2008, Baudrie et al. 2007):
        VLF : 0.000 – 0.400 Hz
        LF  : 0.400 – 1.500 Hz   (baroreflex / sympathovagal oscillations)
        HF  : 1.500 – 5.000 Hz   (respiratory sinus arrhythmia, breathing at 2–4 Hz)
    *bands* = (vlf, lf, hf) overrides them (the band preset).

    The spectrum is ``hrv_spectrum`` (Welch PSD of the valid RR intervals,
    cubic-spline resampled at 20 Hz on their true times) and the metrics are
    ``spectral_band_metrics``: absolute VLF/LF/HF/TP powers in ms², LFn/HFn
    = LF or HF / (LF + HF), LF/HF, LnHF and the LF/HF peak frequencies. The
    spectrum itself is kept in ``attrs`` ("psd_freqs", "psd_power" and its
    parameters) so the PSD plot draws exactly the curve these numbers were
    integrated from. (This replaced nk.hrv_frequency, whose 100 Hz
    resampling and max-normalised PSD gave band "powers" in arbitrary units
    and differed from the plotted spectrum.)

    The minimum recommended recording length for stable spectral estimates is:
        ≥ 5 × (1 / VLF_lo) ≈ impossible → use ≥ 5 / LF_lo ≈ 12.5 s (≥ ~100 beats)

    RR intervals outside [rr_min_ms, rr_max_ms] and intervals spanning a
    removed beat (*breaks*) are excluded (see rr_interval_series); the
    valid ones are resampled on the real time axis rather than shifting
    every later sample by the excluded duration. The exclusion counts are
    stored in the returned DataFrame's ``attrs``.

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
    bands       : optional (vlf, lf, hf) band tuples (Hz)
    """
    vlf, lf, hf = bands if bands is not None else (MouseECG.VLF, MouseECG.LF, MouseECG.HF)
    rr_series = rr_interval_series(rpeaks, fs, rr_min_ms, rr_max_ms,
                                   breaks=breaks)   # also validates fs
    n_valid   = len(rr_series["rri"])

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
        progress_cb(10, f"Frequency-domain HRV (LF {lf[0]:g}–{lf[1]:g} Hz, "
                        f"HF {hf[0]:g}–{hf[1]:g} Hz)…")

    try:
        spec = hrv_spectrum(rr_series["rri"], rr_series["rri_time"])
        if spec is None:
            return _withheld_frame("RR series too short to resample for a spectrum",
                                   rr_series)
        result = pd.DataFrame([spectral_band_metrics(spec, vlf, lf, hf)])
        result.attrs.update(rr_exclusion_summary(rr_series))
        result.attrs.update({
            "psd_freqs":     spec["freqs"],
            "psd_power":     spec["psd"],
            "psd_nperseg":   spec["nperseg"],
            "psd_noverlap":  spec["noverlap"],
            "psd_n_points":  spec["n_points"],
            "psd_interp_fs": spec["interp_fs"],
            "bands":         [list(map(float, vlf)), list(map(float, lf)),
                              list(map(float, hf))],
            "method":        SPECTRAL_METHOD_TEXT,
        })
        if progress_cb:
            progress_cb(100, "Frequency HRV done")
        return result
    except Exception as exc:
        log.warning("analyse_hrv_freq failed: %s", exc)
        return pd.DataFrame()

def entropy_subset(valid: np.ndarray, cap: int) -> "tuple[int, int]":
    """Interval indices [i0, i1) of the block of CONSECUTIVE RR intervals that
    holds *cap* valid intervals with the fewest excluded ones (earliest on
    ties) -- the RR series the capped entropy/complexity metrics of
    analyse_hrv_nonlinear are computed on. The whole series (0, len) when
    it holds no more than *cap* valid intervals or *cap* <= 0."""
    ok = np.asarray(valid, dtype=bool).flatten()
    k  = np.flatnonzero(ok)
    if cap <= 0 or len(k) <= cap:
        return 0, int(len(ok))
    spans = k[cap - 1:] - k[:len(k) - cap + 1]
    j = int(np.argmin(spans))
    return int(k[j]), int(k[j + cap - 1] + 1)


def _full_window_pair_and_dfa(series: dict, fs: float) -> "pd.DataFrame":
    """NeuroKit's Poincaré, fragmentation, asymmetry and DFA (α1, α2, MFDFA)
    blocks of nk.hrv_nonlinear on EVERY valid interval of *series* -- the
    cheap part of the non-linear set, recomputed on the whole window when
    the entropy block was capped. Same NeuroKit functions and input
    handling as nk.hrv_nonlinear (0.2.13); a block that raises (e.g. the
    asymmetry indices on zero variability) is left out."""
    assert nk is not None
    from neurokit2.hrv.hrv_nonlinear import (   # NeuroKit 0.2.13 internals
        _hrv_dfa, _hrv_nonlinear_fragmentation, _hrv_nonlinear_poincare,
        _hrv_nonlinear_poincare_hra)
    from neurokit2.hrv.hrv_utils import _hrv_format_input
    rri, rri_time, rri_missing = _hrv_format_input(nk_rri_input(series),
                                                   sampling_rate=int(fs))
    out: dict = {}
    for fn in (_hrv_nonlinear_poincare, _hrv_nonlinear_fragmentation,
               _hrv_nonlinear_poincare_hra):
        part: dict = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn(rri, rri_time=rri_time, rri_missing=rri_missing, out=part)
            out.update(part)
        except Exception as exc:
            log.debug("full-window %s failed: %s", fn.__name__, exc)
    part = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _hrv_dfa(rri, part)
        out.update(part)
    except Exception as exc:
        log.debug("full-window DFA failed: %s", exc)
    return pd.DataFrame({f"HRV_{k}": [float(v)] for k, v in out.items()})


def nonlinear_subset_text(attrs: "Optional[dict]") -> str:
    """One-line disclosure of the RR intervals a non-linear HRV result used
    (empty when every metric used the whole window). Shared by the
    Non-linear status line, the Summary report and the Excel Cover sheet."""
    a = attrs or {}
    if a.get("entropy_capped"):
        return (f"Entropy/complexity metrics (SampEn, ApEn, MSE, …) from "
                f"{int(a.get('entropy_n_rr', 0)):,} of "
                f"{int(a.get('n_rr_valid', 0)):,} valid RR intervals "
                f"({float(a.get('entropy_t_start_s', 0.0)):.1f}–"
                f"{float(a.get('entropy_t_end_s', 0.0)):.1f} s; cap "
                f"{int(a.get('entropy_max_beats', 0)):,}); Poincaré, DFA and "
                f"asymmetry indices from all of them")
    if a.get("truncated"):
        # results from an older version (first max_beats beats only)
        return (f"All non-linear metrics from the first "
                f"{int(a.get('n_beats_used', 0)):,} of "
                f"{int(a.get('n_beats_total', 0)):,} beats (older analysis — "
                f"re-run Non-linear HRV)")
    return ""


_NL_ENTROPY_PREFIXES = ("HRV_ApEn", "HRV_SampEn", "HRV_ShanEn", "HRV_FuzzyEn",
                        "HRV_MSEn", "HRV_CMSEn", "HRV_RCMSEn", "HRV_CD",
                        "HRV_HFD", "HRV_KFD", "HRV_LZC", "HRV_Symbolic")


def analyse_hrv_nonlinear(
    rpeaks: np.ndarray,
    fs: float,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    max_beats: "Optional[int]" = None,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
    breaks: "Optional[np.ndarray]" = None,
) -> pd.DataFrame:
    """Non-linear HRV via NeuroKit2: Poincaré SD1/SD2, DFA α1/α2, ApEn/SampEn,
    Porta index, and the rest of nk.hrv_nonlinear()'s standard metric set.

    HRV_SampEn is computed on the RR-INTERVAL series (the standard textbook
    Sample Entropy of HRV, whose usual reference range of 0.5-2.5 assumes
    it).

    Beats used. Poincaré SD1/SD2 (``poincare_sd1_sd2``, the app-wide
    definition) and the indices derived from them, DFA α1/α2 and the
    asymmetry/fragmentation indices use EVERY valid RR interval of
    *rpeaks* (the analysis window). Only the O(N²) entropy/complexity block
    (ApEn, SampEn, FuzzyEn, ShanEn, MSEn/CMSEn/RCMSEn, CD, HFD, KFD, LZC,
    symbolic dynamics, and the MSE curve) is capped for runtime: to
    *max_beats* valid intervals (None → MouseECG.ENTROPY_MAX_BEATS; 0 or
    less → no cap), taken as the contiguous block with the fewest excluded
    intervals (``entropy_subset``) -- never an evenly-spaced subsample,
    since SampEn/MSE are sequence-order statistics. The block's count and
    time span are in ``attrs`` and disclosed (``nonlinear_subset_text``).
    (Earlier versions silently computed EVERYTHING on the first 1000 beats.)

    Tested down to 5 beats with no crashes; nk degrades individual columns
    to NaN (e.g. HRV_DFA_alpha1 needs enough beats for a stable fit) rather
    than raising.

    RR intervals outside [rr_min_ms, rr_max_ms] and intervals spanning a
    removed beat (*breaks*) are excluded (see rr_interval_series); Poincaré
    pairs never straddle an excluded interval, and the asymmetry /
    fragmentation indices are withheld (NaN, listed in
    ``attrs["pair_metrics_withheld"]``) if NeuroKit paired intervals across
    a gap. The exclusion counts are stored in the returned DataFrame's
    ``attrs``.

    Args:
        rpeaks: R-peak sample indices (ndarray) of the analysis window
        fs: Sampling rate (Hz)
        progress_cb: Progress callback (pct, message) -> None
        max_beats: Cap on valid RR intervals for the entropy/complexity block
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
    n_rpeaks = len(rpeaks)
    rr_series = rr_interval_series(rpeaks, fs, rr_min_ms, rr_max_ms,
                                   breaks=breaks)   # also validates fs
    n_valid = len(rr_series["rri"])
    cap = MouseECG.ENTROPY_MAX_BEATS if max_beats is None else int(max_beats)
    i0, i1 = entropy_subset(rr_series["valid"], cap)
    capped = (i0, i1) != (0, len(rr_series["valid"]))
    if capped:
        ent_peaks  = rpeaks[i0:i1 + 1]
        ent_series = rr_interval_series(ent_peaks, fs, rr_min_ms, rr_max_ms,
                                        breaks=breaks)
        log.info("analyse_hrv_nonlinear: entropy block on %d of %d valid RR "
                 "(%.1f–%.1f s)", len(ent_series["rri"]), n_valid,
                 ent_peaks[0] / fs, ent_peaks[-1] / fs)
    else:
        ent_peaks, ent_series = rpeaks, rr_series

    if progress_cb:
        progress_cb(20, "Computing SD1/SD2, DFA, SampEn/ApEn… (may take 30 s+)")

    nk_failed_reason: "Optional[str]" = None
    poincare_cols = ("HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_S",
                     "HRV_CSI", "HRV_CVI", "HRV_CSI_Modified")
    try:
        result = nk.hrv_nonlinear(nk_rri_input(ent_series), sampling_rate=int(fs), show=False)
    except Exception as exc:
        # NeuroKit 0.2.13 divides by the number of non-zero successive
        # differences (_hrv_nonlinear_poincare_hra), so an RR series with
        # exactly zero beat-to-beat variability (paced or synthetic data)
        # raises ZeroDivisionError. Keep the gap-aware Poincaré set below
        # instead of returning nothing, and say why the rest is missing.
        log.warning("analyse_hrv_nonlinear failed: %s", exc)
        nk_failed_reason = f"NeuroKit non-linear HRV failed ({type(exc).__name__}: {exc})"
        result = pd.DataFrame({c: [float("nan")] for c in poincare_cols})

    # Entropy block capped: the cheap metrics (Poincaré, fragmentation,
    # asymmetry, DFA) are recomputed by NeuroKit on the whole window.
    entropy_cols: "list[str]" = []
    if capped:
        if progress_cb:
            progress_cb(45, "Poincaré / DFA on every beat of the window…")
        try:
            full = _full_window_pair_and_dfa(rr_series, fs)
        except Exception as exc:          # NeuroKit internals moved: run it all
            log.warning("analyse_hrv_nonlinear: full-window NeuroKit blocks "
                        "unavailable (%s) — running nk.hrv_nonlinear on all beats", exc)
            try:
                full = nk.hrv_nonlinear(nk_rri_input(rr_series),
                                        sampling_rate=int(fs), show=False)
                full = full[[c for c in full.columns
                             if not c.startswith(_NL_ENTROPY_PREFIXES)]]
            except Exception as exc2:
                log.warning("analyse_hrv_nonlinear: full-window fallback failed: %s", exc2)
                full = pd.DataFrame()
        for col in full.columns:
            result[col] = float(full[col].iloc[0])
        # A whole-window metric that could not be computed must not keep
        # the value of the capped block.
        for col in _NL_PAIR_COLUMNS + ("HRV_DFA_alpha1", "HRV_DFA_alpha2"):
            if col in result.columns and col not in full.columns:
                result[col] = float("nan")
        entropy_cols = [c for c in result.columns if c.startswith(_NL_ENTROPY_PREFIXES)]

    # One SD1/SD2 definition (poincare_sd1_sd2) on every valid pair of the
    # window, never bridging a gap (see _overwrite_successive_poincare's
    # section comment); if NeuroKit's own SD1/SD2 disagreed, it paired
    # intervals across an excluded one, so its other pair-based indices are
    # withheld too.
    agreed = _overwrite_successive_poincare(result, rr_series)
    if nk_failed_reason is not None and not np.isfinite(float(result["HRV_SD1"].iloc[0])):
        return pd.DataFrame()
    pair_withheld: "list[str]" = []
    if rr_series["n_excluded"] > 0 and not agreed and nk_failed_reason is None:
        pair_withheld = [c for c in _NL_PAIR_COLUMNS if c in result.columns]
        for col in pair_withheld:
            result[col] = float("nan")
        log.warning("analyse_hrv_nonlinear: NeuroKit paired RR intervals across "
                    "an excluded one — %d asymmetry/fragmentation indices withheld",
                    len(pair_withheld))

    if progress_cb:
        progress_cb(60, "Multiscale entropy…")

    # Multiscale entropy (Costa et al. 2002): SampEn recomputed at coarser
    # and coarser time scales of the SAME RR series as the entropy block
    # above -- catches loss of complexity that a single-scale SampEn can
    # miss (rodent pharmacology literature reports cases where combined
    # autonomic blockade only showed up at intermediate MSE scales, not at
    # scale 1). Needs materially more data than a single SampEn value to be
    # stable per scale, so this is skipped outright below a minimum beat
    # count rather than returning a noisy curve.
    # Valid intervals only: an excluded interval is dropped, NOT interpolated
    # (nk.entropy_multiscale has no notion of a gap, so the coarse-grained
    # series simply continues with the next valid interval).
    rr_ms_mse = ent_series["rri"]
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
    result.attrs["n_beats_used"]  = n_rpeaks
    result.attrs["n_beats_total"] = n_rpeaks
    result.attrs["truncated"]     = False       # older versions: first 1000 beats only
    result.attrs["n_rr_valid"]    = n_valid
    result.attrs["window_t_start_s"] = float(rpeaks[0] / fs) if n_rpeaks else 0.0
    result.attrs["window_t_end_s"]   = float(rpeaks[-1] / fs) if n_rpeaks else 0.0
    result.attrs["entropy_capped"]    = bool(capped)
    result.attrs["entropy_max_beats"] = int(cap)
    result.attrs["entropy_n_rr"]      = int(len(ent_series["rri"]))
    result.attrs["entropy_t_start_s"] = float(ent_peaks[0] / fs) if len(ent_peaks) else 0.0
    result.attrs["entropy_t_end_s"]   = float(ent_peaks[-1] / fs) if len(ent_peaks) else 0.0
    result.attrs["entropy_columns"]   = entropy_cols
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


# ── Interval definitions (shown in the UI, exports and docs) ──────────────
# Landmarks: QRS onset = first departure from the PR-segment baseline (Q wave
# onset, or R upstroke when there is no Q); QRS offset = J point (end of the
# S wave, back at the PR baseline -- the mouse J wave that follows is early
# repolarisation, Boukens et al. 2014 J Physiol 592:4613, so it is counted in
# QT, not in QRS); T end = line fitted through the 70-30 % part of the T-wave
# trailing edge meeting the PR baseline (the mouse T-wave end is gradual and
# ambiguous: read QT with care); P onset = the same kind of line through the
# P upstroke meeting the PR baseline. None of the four is clamped to a
# template window (see detect_waves_on_beat).
INTERVAL_DEFINITIONS: "tuple[tuple[str, str], ...]" = (
    ("PR",  "P onset → QRS onset"),
    ("QRS", "QRS onset → QRS offset (J point, end of S)"),
    ("QT",  "QRS onset → T end"),
    ("QTc", "QT corrected with the preceding RR (previous R → this R)"),
)
INTERVAL_DEFINITIONS_TEXT: str = "; ".join(f"{k} = {v}" for k, v in INTERVAL_DEFINITIONS)
LEGACY_INTERVAL_DEFINITIONS_TEXT: str = (
    "legacy (saved by an older version): PR = P onset → Q nadir, "
    "QRS = Q nadir → S nadir, QT = R peak → T end, QTc re-paired with the "
    "preceding RR — re-run Intervals for the standard definitions")
# A legacy frame without R peak times cannot be re-paired (its RR_ms is the
# FOLLOWING RR): recompute_qtc_column leaves its QTc empty.
LEGACY_INTERVAL_DEFINITIONS_NO_QTC_TEXT: str = (
    "legacy (saved by an older version): PR = P onset → Q nadir, "
    "QRS = Q nadir → S nadir, QT = R peak → T end, QTc not available (no R "
    "peak times to pair it with the preceding RR) — re-run Intervals for the "
    "standard definitions")

QT_VARIABILITY_LABEL: str = (
    "Beat-to-beat QT variability (single lead — not classical QT dispersion)")


def interval_definitions_text(intervals: "Optional[pd.DataFrame]") -> str:
    """The definitions line matching *intervals*: standard, legacy, or
    legacy without a recoverable QTc (``recompute_qtc_column``)."""
    if not intervals_use_legacy_definitions(intervals):
        return INTERVAL_DEFINITIONS_TEXT
    if "R_peak_s" in intervals.columns:   # type: ignore[union-attr]
        return LEGACY_INTERVAL_DEFINITIONS_TEXT
    return LEGACY_INTERVAL_DEFINITIONS_NO_QTC_TEXT


def intervals_use_legacy_definitions(intervals: "Optional[pd.DataFrame]") -> bool:
    """True for an ``analyse_intervals`` frame computed before the standard
    definitions (no ``QRS_onset_ms`` column, e.g. from an old session):
    its PR / QRS / QT were measured Q nadir / Q→S nadirs / from R."""
    return (intervals is not None and not intervals.empty
            and "QT_ms" in intervals.columns
            and "QRS_onset_ms" not in intervals.columns)


def qt_beat_variability(qt_ms: "np.ndarray") -> dict:
    """Beat-to-beat spread of QT within one recording (single lead).

    Not classical QT dispersion, which is the max − min QT across the leads
    of ONE beat: here every value comes from the same lead, so the range is
    dominated by delineation outliers. The robust spread (IQR) is the one to
    report; range and SD are given for completeness.

    Returns {"n", "median_ms", "iqr_ms", "sd_ms", "range_ms"}; the spreads
    are NaN with fewer than 4 finite values.
    """
    qt = np.asarray(qt_ms, dtype=float)
    qt = qt[np.isfinite(qt)]
    out = {"n": int(len(qt)), "median_ms": float("nan"), "iqr_ms": float("nan"),
           "sd_ms": float("nan"), "range_ms": float("nan")}
    if len(qt):
        out["median_ms"] = float(np.median(qt))
    if len(qt) >= 4:
        q25, q75 = np.percentile(qt, [25, 75])
        out["iqr_ms"]   = float(q75 - q25)
        out["sd_ms"]    = float(np.std(qt, ddof=1))
        out["range_ms"] = float(np.max(qt) - np.min(qt))
    return out


def qt_variability_text(v: dict) -> str:
    """One-line summary of ``qt_beat_variability`` output (exports, report)."""
    if not np.isfinite(v.get("iqr_ms", float("nan"))):
        return "—"
    return (f"IQR {v['iqr_ms']:.1f} ms, SD {v['sd_ms']:.1f} ms, "
            f"range {v['range_ms']:.1f} ms (n = {v['n']} beats)")


def interval_disclosure_rows(intervals: "Optional[pd.DataFrame]") -> "list[tuple[str, str]]":
    """(label, value) rows stating the interval definitions, the QT
    variability and the wave template used for *intervals* -- shared by the Excel Cover sheet, the
    Prism notes and the Summary report. Empty without QT data."""
    if intervals is None or intervals.empty or "QT_ms" not in intervals.columns:
        return []
    rows = [("Interval definitions", interval_definitions_text(intervals))]
    qt = pd.to_numeric(intervals["QT_ms"], errors="coerce").to_numpy(dtype=float)
    rows.append(("QT variability",
                 f"{qt_variability_text(qt_beat_variability(qt))} — "
                 f"{QT_VARIABILITY_LABEL[0].lower()}{QT_VARIABILITY_LABEL[1:]}"))
    rows.append(("Wave template", wave_template_disclosure(intervals)))
    bw = beat_window_text(intervals.attrs.get("beat_window"))
    if bw:
        rows.append(("Beat window", bw))
    return rows


def _preceding_rr_legacy(intervals: "pd.DataFrame") -> np.ndarray:
    """Preceding RR for a legacy frame whose ``RR_ms`` is the FOLLOWING RR:
    row i takes row i-1's RR when the two rows are consecutive beats
    (their R_peak_s gap matches that RR within 1.5 ms), NaN otherwise."""
    rr_next = pd.to_numeric(intervals["RR_ms"], errors="coerce").to_numpy(dtype=float)
    r_s     = pd.to_numeric(intervals["R_peak_s"], errors="coerce").to_numpy(dtype=float)
    out = np.full(len(rr_next), np.nan)
    if len(rr_next) < 2:
        return out
    gap = np.diff(r_s) * 1000.0
    consecutive = np.isfinite(gap) & (np.abs(gap - rr_next[:-1]) <= 1.5)
    out[1:] = np.where(consecutive, rr_next[:-1], np.nan)
    return out


def qtc_rr_column(intervals: "Optional[pd.DataFrame]") -> str:
    """Name of the RR column each ``QTc_ms`` was corrected with: ``RR_ms``
    (the preceding RR) for current frames, ``RR_prev_ms`` for a legacy
    frame re-paired by ``recompute_qtc_column`` (its ``RR_ms`` is the
    FOLLOWING RR) -- what a QTc-vs-RR plot or a reader recomputing QTc
    must use."""
    if intervals is not None and "RR_prev_ms" in intervals.columns:
        return "RR_prev_ms"
    return "RR_ms"


def recompute_qtc_column(intervals: "pd.DataFrame", formula: str) -> "pd.DataFrame":
    """Return a copy of an ``analyse_intervals`` DataFrame whose ``QTc_ms``
    column is recomputed from its ``QT_ms`` / ``RR_ms`` with *formula*.

    Uses exactly the per-beat (preceding) RR that ``analyse_intervals``
    pairs with each QT (floored at ``MouseECG.RR_MIN_MS``), so the result is
    identical to re-running the delineation with that formula. A legacy
    frame (``intervals_use_legacy_definitions``) stored the FOLLOWING RR in
    ``RR_ms``: when it has ``R_peak_s`` its QT is re-paired with the
    preceding RR, stored next to it as ``RR_prev_ms`` (see
    ``qtc_rr_column``; ``RR_ms`` itself is left as stored, so this stays
    idempotent); without ``R_peak_s`` the preceding RR cannot be recovered
    and its QTc (and ``RR_prev_ms``) is NaN rather than paired with the
    following RR (``interval_definitions_text`` says so). Frames without
    both columns (e.g. the early-return ``{"RR_ms": ...}`` shape) come
    back unchanged.
    """
    if "QT_ms" not in intervals.columns or "RR_ms" not in intervals.columns:
        return intervals
    qt = pd.to_numeric(intervals["QT_ms"], errors="coerce").to_numpy(dtype=float)
    legacy = intervals_use_legacy_definitions(intervals)
    if legacy and "R_peak_s" in intervals.columns:
        rr = _preceding_rr_legacy(intervals)
    elif legacy:
        rr = np.full(len(intervals), np.nan)
    else:
        rr = pd.to_numeric(intervals["RR_ms"], errors="coerce").to_numpy(dtype=float)
    out = intervals.copy()
    if legacy:
        out["RR_prev_ms"] = rr
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
    wave_template : search-window template; the built-in defaults when None.
        Read only -- never modified or saved (a snapshot is stored in the
        result's ``attrs["wave_template"]``).
    beat_template, beat_time_arr : unused, kept for API compatibility.
    permissive_bounds : if True, relax physiological bounds by 2×.
    qtc_formula : QTc correction for the ``QTc_ms`` column — see
        ``compute_qtc`` (default Mitchell, mouse-calibrated).
    rr_min_ms, rr_max_ms : RR acceptance bounds (ms), pass the bounds HRV
        used (e.g. a widened Max RR for an anaesthetised mouse, where no
        interval is valid under the defaults). ``RR_ms`` -- hence QTc -- is
        NaN outside them; a neighbour closer than *rr_min_ms* does not size
        the beat window (``beat_window_sides``).

    Returns
    -------
    pd.DataFrame with one row per beat whose full window fits in the
    signal, and columns RR_ms (the PRECEDING RR, NaN for the first input
    peak), PR_ms, QRS_ms, QT_ms, QTc_ms (QT corrected with that preceding
    RR), R_peak_s, plus per-wave absolute (``_s``) and relative (``_ms``)
    positions (incl. QRS_onset / QRS_offset), beat_idx, and an ``accepted``
    boolean column for the verifier UI. Interval definitions:
    ``INTERVAL_DEFINITIONS``. ``df.attrs["wave_template"]`` is the template
    used (``WaveTemplate.to_dict``) and ``df.attrs["beat_window"]`` the
    per-beat window sizes, both disclosed by ``interval_disclosure_rows``.
    """
    if max_beats and max_beats > 0:
        log.debug("analyse_intervals: max_beats=%d is no longer used (all beats processed)",
                  max_beats)

    def _prog(p: int, m: str) -> None:
        if progress_cb:
            progress_cb(p, m)

    # None -> the built-in defaults, never a file from the user profile:
    # the caller passes the recording's template explicitly.
    if wave_template is None:
        wave_template = WaveTemplate()

    rpeaks_all = np.sort(np.array(rpeaks, dtype=int).flatten())
    n_beats    = len(rpeaks_all)
    if n_beats < 3:
        return pd.DataFrame({"RR_ms": rr_ms})

    # ── Beat window (per beat, asymmetric) ───────────────────────────────
    # Each beat gets its own window: before R up to BEAT_WINDOW_RR_FRAC of
    # ITS preceding RR, after R up to that fraction of ITS following RR,
    # both capped at ±BEAT_HALF_WIN_S (see beat_window_sides). A window
    # sized from the recording's shortest RR made one premature beat
    # shorten every beat's window (QT lost and PR cut on all of them).
    fixed_hw = int(MouseECG.BEAT_HALF_WIN_S * fs)
    pre_all, post_all = beat_window_sides(rpeaks_all, fs, rr_min_ms)

    mask_valid = ((rpeaks_all - pre_all >= 0) &
                  (rpeaks_all + post_all < len(cleaned)))
    valid_rp   = rpeaks_all[mask_valid]
    pre_v      = pre_all[mask_valid]
    post_v     = post_all[mask_valid]
    n_valid    = len(valid_rp)

    if n_valid < 2:
        log.warning("analyse_intervals: too few valid beats (%d)", n_valid)
        return pd.DataFrame({"RR_ms": rr_ms})

    _prog(5, f"Extracting {n_valid} beats…")

    # ── Per-beat detection ───────────────────────────────────────────────
    _prog(10, "Detecting P/Q/S/T landmarks per beat…")

    p_onset  = np.full(n_valid, np.nan)
    p_peak   = np.full(n_valid, np.nan)
    qrs_on   = np.full(n_valid, np.nan)
    q_peak   = np.full(n_valid, np.nan)
    s_peak   = np.full(n_valid, np.nan)
    qrs_off  = np.full(n_valid, np.nan)
    j_peak   = np.full(n_valid, np.nan)
    t_peak   = np.full(n_valid, np.nan)
    t_offset = np.full(n_valid, np.nan)

    # The P wave of a beat comes after the previous beat's T wave: at mouse
    # heart rates the window before R reaches back onto it, so the P search
    # starts at that T peak. Only the immediately preceding peak counts, and
    # only when it is a heart beat (>= rr_min_ms away, see beat_window_sides):
    # a premature beat moves the P search of its successor alone.
    gap_min = rr_min_ms / 1000.0 * fs
    pos_v   = np.searchsorted(rpeaks_all, valid_rp)
    # And its T wave ends before the next beat's P: the next heart beat (a
    # peak closer than rr_min_ms is not one) tells the T search where that is.
    i_next  = np.searchsorted(rpeaks_all, valid_rp + int(np.ceil(gap_min)), side="left")

    tick = max(1, n_valid // 20)
    for i in range(n_valid):
        r0 = int(valid_rp[i])
        lo, hi = int(pre_v[i]), int(post_v[i])
        bt_ms = np.arange(-lo, hi) / fs * 1000
        p_from = None
        if (i > 0 and pos_v[i] > 0 and valid_rp[i - 1] == rpeaks_all[pos_v[i] - 1]
                and r0 - valid_rp[i - 1] >= gap_min and np.isfinite(t_peak[i - 1])):
            p_from = (t_peak[i - 1] - r0) / fs * 1000
        next_r = ((rpeaks_all[i_next[i]] - r0) / fs * 1000
                  if i_next[i] < len(rpeaks_all) else None)
        det = detect_waves_on_beat(cleaned[r0 - lo:r0 + hi].astype(float), bt_ms,
                                   wave_template, p_search_from_ms=p_from,
                                   next_r_ms=next_r)

        def _ms_to_samp(v: float) -> float:
            return valid_rp[i] + v / 1000.0 * fs if np.isfinite(v) else np.nan

        p_onset [i] = _ms_to_samp(det["P_onset"])
        p_peak  [i] = _ms_to_samp(det["P_peak"])
        qrs_on  [i] = _ms_to_samp(det.get("QRS_onset", np.nan))
        q_peak  [i] = _ms_to_samp(det["Q_peak"])
        s_peak  [i] = _ms_to_samp(det["S_peak"])
        qrs_off [i] = _ms_to_samp(det.get("QRS_offset", np.nan))
        j_peak  [i] = _ms_to_samp(det["J_peak"])
        t_peak  [i] = _ms_to_samp(det["T_peak"])
        t_offset[i] = _ms_to_samp(det["T_offset"])

        if (i + 1) % tick == 0:
            _prog(10 + int(72 * (i + 1) / n_valid),
                  f"Detecting landmarks… {i + 1}/{n_valid}")

    _prog(82, "Computing intervals…")

    # ── Interval arithmetic ──────────────────────────────────────────────
    # One row per beat with a full window. Standard definitions, all from
    # landmarks measured on this beat (see INTERVAL_DEFINITIONS):
    #   PR  = P onset   -> QRS onset
    #   QRS = QRS onset -> QRS offset (J point)
    #   QT  = QRS onset -> T end
    # RR_ms is the PRECEDING interval (previous accepted R -> this R), the
    # one QTc is corrected with; NaN for the first peak of the input.
    n = n_valid
    r_pos_n    = valid_rp.astype(float)
    p_onset_n  = p_onset;  p_peak_n   = p_peak
    qrs_on_n   = qrs_on;   qrs_off_n  = qrs_off
    q_peak_n   = q_peak;   s_peak_n   = s_peak
    j_peak_n   = j_peak
    t_peak_n   = t_peak;   t_offset_n = t_offset

    pos_all = np.searchsorted(rpeaks_all, valid_rp)
    prev_rp = np.where(pos_all > 0, rpeaks_all[np.maximum(pos_all - 1, 0)], -1)
    rr_arr  = np.where(pos_all > 0, (valid_rp - prev_rp).astype(float) / fs * 1000,
                       np.nan)
    pr_raw  = (qrs_on_n   - p_onset_n) / fs * 1000
    qrs_raw = (qrs_off_n  - qrs_on_n)  / fs * 1000
    qt_raw  = (t_offset_n - qrs_on_n)  / fs * 1000

    # ── Physiological bounds ─────────────────────────────────────────────
    _m = 2.0 if permissive_bounds else 1.0
    def _clip(arr: "np.ndarray", lo: float, hi: float) -> "np.ndarray":
        out = arr.copy().astype(float)
        out[(out < lo) | (out > hi)] = np.nan
        return out

    pr_clean  = _clip(pr_raw,  MouseECG.PR_ABS_MIN  / _m, MouseECG.PR_ABS_MAX  * _m)
    qrs_clean = _clip(qrs_raw, MouseECG.QRS_ABS_MIN / _m, MouseECG.QRS_ABS_MAX * _m)
    qt_clean  = _clip(qt_raw,  MouseECG.QT_ABS_MIN  / _m, MouseECG.QT_ABS_MAX  * _m)
    # RR is checked against the bounds HRV used (a widened Max RR for an
    # anaesthetised mouse must not leave every QTc empty).
    rr_clean  = _clip(rr_arr,  rr_min_ms / _m, rr_max_ms * _m)

    # QTc pairs each QT with its PRECEDING (floored) RR — mirrored by
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
        "QRS_onset_s": qrs_on_n  / fs,
        "Q_peak_s":   q_peak_n   / fs,
        "S_peak_s":   s_peak_n   / fs,
        "QRS_offset_s": qrs_off_n / fs,
        "J_peak_s":   j_peak_n   / fs,
        "T_peak_s":   t_peak_n   / fs,
        "T_offset_s": t_offset_n / fs,
        # ms-from-R columns for the verifier UI (drag-to-adjust)
        "P_onset_ms":  (p_onset_n  - r_pos_n) / fs * 1000,
        "P_peak_ms":   (p_peak_n   - r_pos_n) / fs * 1000,
        "QRS_onset_ms": (qrs_on_n  - r_pos_n) / fs * 1000,
        "Q_peak_ms":   (q_peak_n   - r_pos_n) / fs * 1000,
        "S_peak_ms":   (s_peak_n   - r_pos_n) / fs * 1000,
        "QRS_offset_ms": (qrs_off_n - r_pos_n) / fs * 1000,
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
           ["Q_peak_s","S_peak_s","Q_peak_ms","S_peak_ms",
            "QRS_offset_s","QRS_offset_ms"]] = np.nan
    df.loc[np.isnan(qt_raw)  | (qt_raw  < MouseECG.QT_ABS_MIN)  | (qt_raw  > MouseECG.QT_ABS_MAX),
           ["T_peak_s","T_offset_s","T_peak_ms","T_offset_ms"]] = np.nan
    # QRS onset starts PR's end, QRS and QT: kept while any of them is
    # plausible, NaN once none is (it is then not a usable landmark).
    df.loc[df[["PR_ms", "QRS_ms", "QT_ms"]].isna().all(axis=1),
           ["QRS_onset_s", "QRS_onset_ms"]] = np.nan

    n_complete = int((~df[["PR_ms","QRS_ms","QT_ms"]].isna().any(axis=1)).sum())
    log.info("analyse_intervals: %d beats, %d complete (%.0f%%)",
             n, n_complete, 100.0 * n_complete / max(n, 1))

    # ── Template used (disclosed in exports, kept in sessions) ───────────
    # The template is NOT updated or saved from this run: a recording's
    # intervals depend only on that recording and the template chosen for
    # it (older versions re-saved a global template narrowed to the last
    # recording after every run, so results depended on the analysis
    # order). Refining from this run is an explicit editor action
    # (refine_template_from_intervals).
    df.attrs["wave_template"] = wave_template.to_dict()
    # Beat window actually used (disclosed by interval_disclosure_rows):
    # beats whose window was shortened by a close neighbour may lose their
    # P onset (before R) or T end (after R).
    df.attrs["beat_window"] = {
        "max_ms":     round(fixed_hw / fs * 1000, 1),
        "rr_frac":    BEAT_WINDOW_RR_FRAC,
        "n_beats":    int(n),
        "n_short_pre":  int((pre_v  < fixed_hw).sum()),
        "n_short_post": int((post_v < fixed_hw).sum()),
        "min_pre_ms":   round(float(pre_v.min())  / fs * 1000, 1),
        "min_post_ms":  round(float(post_v.min()) / fs * 1000, 1),
    }

    _prog(100, f"Done — {n_complete}/{n} complete beats")
    return df


def refine_template_from_intervals(
    intervals: "pd.DataFrame",
    template:  "WaveTemplate",
    min_complete: int = 10,
) -> "Optional[WaveTemplate]":
    """A NEW template whose P / Q / S / T windows are centred on this
    recording's delineation (median ± 0.75 x the 10-90 % spread, at least
    ±4 ms) -- the editor's explicit "Refine from delineation" action.

    *template* is not modified and nothing is saved. Windows with fewer
    than 5 measured beats keep *template*'s value; QRS onset / offset are
    not refined (detection does not use their windows). None when fewer
    than *min_complete* beats have PR, QRS and QT.
    """
    if intervals is None or intervals.empty:
        return None
    cols = [c for c in ("PR_ms", "QRS_ms", "QT_ms") if c in intervals.columns]
    if len(cols) < 3:
        return None
    n_complete = int((~intervals[cols].isna().any(axis=1)).sum())
    if n_complete < min_complete:
        return None
    _ms_cols = {
        "P_onset": "P_onset_ms", "P_peak": "P_peak_ms",
        "Q_peak":  "Q_peak_ms",  "S_peak": "S_peak_ms",
        "T_peak":  "T_peak_ms",  "T_offset": "T_offset_ms",
    }
    out = template.copy()
    for wk, col in _ms_cols.items():
        if col not in intervals.columns:
            continue
        valid = pd.to_numeric(intervals[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(valid) < 5:
            continue
        med    = float(np.median(valid))
        spread = max(4.0, float(np.percentile(valid, 90) -
                                np.percentile(valid, 10)) * 0.75)
        out.landmarks[wk] = (round(med, 1), round(spread, 1))
    out.source, out.origin, out.confirmed = "refined", "", False
    return out


def wave_template_disclosure(intervals: "Optional[pd.DataFrame]") -> str:
    """Which wave template guided the delineation of *intervals* (its
    ``attrs["wave_template"]``): provenance plus the landmark-window
    centres. Frames computed before templates were recorded say so."""
    data = intervals.attrs.get("wave_template") if intervals is not None else None
    if not isinstance(data, dict):
        return ("not recorded (computed by an older version, which re-saved "
                "a global template after every delineation) — re-run "
                "Intervals to record it")
    wt = WaveTemplate.from_dict(data)
    # The windows detection actually searched: a partial template (older or
    # hand-written file) has its missing landmarks filled with the defaults
    # by detect_waves_on_beat, so those are disclosed too.
    cells = format_windows({**WaveTemplate.DEFAULTS, **wt.landmarks})
    return f"{wt.provenance()}  (search windows, ms from R: {cells})"


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
    to the signal edges for a full beat window are dropped, so the
    DataFrame can be shorter than the peak array and cannot be indexed
    positionally. Rows are matched
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
    block_s:   float = MouseECG.SEGMENT_BLOCK_S,
    vlf_band:  "tuple[float, float]" = MouseECG.VLF,
) -> dict:
    """Compute HRV and cardiac stats for a time window [t_lo, t_hi] seconds.

    Every metric uses the app-wide definition: mean HR = 60000 / mean NN
    (``mean_hr_bpm``), SD1/SD2 = ``poincare_sd1_sd2``, LF/HF from the same
    ``hrv_spectrum`` / ``spectral_band_metrics`` as the Frequency panel
    (and the same ≥ MIN_BEATS_SPECTRAL valid-interval and gap-withholding
    rules) -- a segment covering the whole analysis window gives the same
    numbers as the main panels.

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
    block_s : length (s) of the non-overlapping blocks of ``segment_blocks``.
    vlf_band: (lo, hi) Hz VLF band.

    Returns
    -------
    dict with keys: label, t_lo, t_hi, n_beats, duration_s, hr_mean, hr_sd,
    hr_min, hr_max, rr_mean, rr_sd, rr_cv, sdnn, rmssd, pnn6,
    vlf, lf, hf (ms²), lf_nu, hf_nu, lf_hf, sd1, sd2, sd_ratio,
    rpeaks_seg, rr_ms, t_rr, blocks (``segment_blocks``), error,
    freq_withheld_reason (why LF/HF are NaN despite enough beats, else
    None), plus the ``rr_exclusion_summary()`` keys (n_excluded, ...,
    rr_min_ms, rr_max_ms). hr_sd / hr_min / hr_max describe the
    instantaneous HR 60000 / RR_i (instantaneous_hr_stats: sample SD,
    2nd / 98th percentiles -- the same as the Summary).
    """
    _NAN = float("nan")
    _EMPTY: dict = {
        "label":        label,
        "t_lo":         float(t_lo),
        "t_hi":         float(t_hi),
        "n_beats":      0,
        "duration_s":   float(t_hi - t_lo),
        "coverage_pct": 0.0,
        "hr_mean":  _NAN, "hr_sd":    _NAN, "hr_min":  _NAN, "hr_max":   _NAN,
        "rr_mean":  _NAN, "rr_sd":    _NAN, "rr_min":  _NAN, "rr_max":   _NAN,
        "rr_cv":    _NAN, "sdnn":     _NAN, "rmssd":   _NAN,
        "pnn6":     _NAN,
        "vlf":      _NAN, "lf":       _NAN, "hf":      _NAN,
        "lf_nu":    _NAN, "hf_nu":    _NAN, "lf_hf":   _NAN,
        "sd1":      _NAN, "sd2":      _NAN, "sd_ratio": _NAN,
        "rpeaks_seg": np.array([], dtype=int),
        "rr_ms":      np.array([], dtype=float),
        "t_rr":       np.array([], dtype=float),
        "blocks":     segment_blocks(np.array([]), np.array([]), t_lo, t_hi, block_s),
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

    # Same instantaneous-HR SD / 2nd–98th percentile range as analyse_core.
    _ihr = instantaneous_hr_stats(rr_ok)

    # ── 3. Time-domain HRV ───────────────────────────────────────────────
    sdnn  = float(np.std(rr_ok, ddof=1))
    rmssd = float(np.sqrt(np.mean(adjacent_diffs ** 2))) if len(adjacent_diffs) else _NAN
    diff_rr = np.abs(adjacent_diffs)
    pnn6    = float(100.0 * np.mean(diff_rr > MouseECG.PNN_THRESHOLD)) if len(diff_rr) else _NAN

    # ── 4. Frequency-domain HRV: the Frequency panel's spectrum ──────────
    # Same estimator and withholding rules as analyse_hrv_freq: the spline
    # would otherwise invent RR values across a long excluded stretch.
    vlf_p = lf_p = hf_p = lf_nu = hf_nu = lf_hf_r = _NAN
    freq_reason = spectral_withheld_reason(rr_all, rr_mask)
    try:
        if freq_reason is None and len(rr_ok) >= MIN_BEATS_SPECTRAL:
            spec = hrv_spectrum(rr_ok, t_ok)
            if spec is not None:
                bm = spectral_band_metrics(spec, vlf_band, lf_band, hf_band)
                vlf_p, lf_p, hf_p = bm["HRV_VLF"], bm["HRV_LF"], bm["HRV_HF"]
                lf_nu   = 100.0 * bm["HRV_LFn"] if np.isfinite(bm["HRV_LFn"]) else _NAN
                hf_nu   = 100.0 * bm["HRV_HFn"] if np.isfinite(bm["HRV_HFn"]) else _NAN
                lf_hf_r = bm["HRV_LFHF"]
    except Exception as exc:
        log.debug("compute_segment_stats freq HRV: %s", exc)

    # ── 5. Poincaré SD1 / SD2 (app-wide definition) ──────────────────────
    pc = poincare_sd1_sd2(rr_all, rr_mask)

    return {
        **exclusion,
        "label":        label,
        "t_lo":         float(t_lo),
        "t_hi":         float(t_hi),
        "n_beats":      n,
        "duration_s":   float(t_hi - t_lo),
        "coverage_pct": float(100.0 * n / max(1, len(rpeaks))),
        "hr_mean":  mean_hr_bpm(rr_ok),       "hr_sd":  _ihr["sd"],
        "hr_min":   _ihr["min"],              "hr_max": _ihr["max"],
        "rr_mean":  float(np.mean(rr_ok)),    "rr_sd":  float(np.std(rr_ok, ddof=1)),
        "rr_min":   float(np.min(rr_ok)),     "rr_max": float(np.max(rr_ok)),
        "rr_cv":    float(100.0 * np.std(rr_ok, ddof=1) / np.mean(rr_ok)),
        "sdnn":     sdnn,   "rmssd":    rmssd,
        "pnn6":     pnn6,
        "vlf":      vlf_p,  "lf":       lf_p,   "hf":       hf_p,
        "lf_nu":    lf_nu,  "hf_nu":    hf_nu,  "lf_hf":    lf_hf_r,
        "sd1":      float(pc["sd1"]), "sd2": float(pc["sd2"]),
        "sd_ratio": float(pc["sd1_sd2"]),
        "rpeaks_seg": rp,
        "rr_ms":      rr_ok,
        "t_rr":       t_ok,
        "blocks":     segment_blocks(t_ok, rr_ok, t_lo, t_hi, block_s),
        "error":      None,
        "freq_withheld_reason": freq_reason,
    }


# ── Within-recording A/B comparison without pseudoreplication ──────────
#
# Compare Segments used to run a Mann-Whitney U test on the beat-level RR
# intervals of the two segments: hundreds of strongly autocorrelated values
# treated as independent observations, so p was "significant" for almost
# any two stretches of the same recording (pseudoreplication). Each segment
# is now summarised by the means of non-overlapping blocks (default 10 s),
# the comparison is reported as block means ± SD with the number of blocks,
# and any within-recording p-value is labelled descriptive: the experimental
# unit is the animal, so group inference belongs to the Event-locked /
# cohort tools, across animals.

SEGMENT_COMPARISON_NOTE = (
    "Within-recording comparison is descriptive: blocks of one recording are "
    "not independent replicates (the experimental unit is the animal). Test "
    "group effects across animals with the Event-locked / cohort tools.")


def segment_blocks(
    t_rr:         np.ndarray,
    rr_ms:        np.ndarray,
    t_lo:         float,
    t_hi:         float,
    block_s:      float = MouseECG.SEGMENT_BLOCK_S,
    min_coverage: float = 0.5,
) -> dict:
    """Means of non-overlapping *block_s*-second blocks of a segment.

    Blocks are [t_lo + k·block_s, t_lo + (k+1)·block_s) lying entirely
    inside [t_lo, t_hi] (a partial last block is dropped). A valid RR
    interval belongs to the block holding its end time *t_rr*. A block is
    kept when its valid intervals cover at least *min_coverage* of its
    length; otherwise it is counted in ``n_dropped``.

    Returns {"block_s", "t_start" (s, kept blocks), "rr_mean" (ms),
    "hr_mean" (bpm, ``mean_hr_bpm`` of the block), "n_rr", "n_blocks",
    "n_dropped"}."""
    t  = np.asarray(t_rr, dtype=np.float64).flatten()
    rr = np.asarray(rr_ms, dtype=np.float64).flatten()
    out: dict = {"block_s": float(block_s), "t_start": [], "rr_mean": [],
                 "hr_mean": [], "n_rr": [], "n_blocks": 0, "n_dropped": 0}
    if not (np.isfinite(block_s) and block_s > 0) or t_hi <= t_lo:
        return out
    n_full = int(np.floor((float(t_hi) - float(t_lo)) / block_s + 1e-9))
    for k in range(n_full):
        b0 = float(t_lo) + k * block_s
        m  = (t >= b0) & (t < b0 + block_s) & np.isfinite(rr)
        if float(rr[m].sum()) / 1000.0 < min_coverage * block_s:
            out["n_dropped"] += 1
            continue
        out["t_start"].append(b0)
        out["rr_mean"].append(float(rr[m].mean()))
        out["hr_mean"].append(mean_hr_bpm(rr[m]))
        out["n_rr"].append(int(m.sum()))
    out["n_blocks"] = len(out["t_start"])
    for key in ("t_start", "rr_mean", "hr_mean"):
        out[key] = np.asarray(out[key], dtype=np.float64)
    out["n_rr"] = np.asarray(out["n_rr"], dtype=int)
    return out


def compare_segment_blocks(blocks_a: dict, blocks_b: dict) -> dict:
    """Descriptive comparison of two ``segment_blocks`` results.

    For "hr_mean" (bpm) and "rr_mean" (ms): mean and SD of the block means of
    each segment, B - A difference of those means, and a DESCRIPTIVE
    Mann-Whitney U p-value on the block means (None with fewer than 3 blocks
    in either segment). The p-value is not a group inference -- see
    SEGMENT_COMPARISON_NOTE.

    Returns {"block_s", "n_blocks_a", "n_blocks_b", "n_dropped_a",
    "n_dropped_b", "metrics": {key: {"mean_a", "sd_a", "mean_b", "sd_b",
    "diff", "p_descriptive"}}, "note"}."""
    nan = float("nan")

    def _mean_sd(v: np.ndarray) -> "tuple[float, float]":
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return nan, nan
        return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else nan)

    metrics: dict = {}
    for key in ("hr_mean", "rr_mean"):
        a = np.asarray(blocks_a.get(key, []), dtype=np.float64)
        b = np.asarray(blocks_b.get(key, []), dtype=np.float64)
        ma, sa = _mean_sd(a)
        mb, sb = _mean_sd(b)
        p: "Optional[float]" = None
        if len(a) >= 3 and len(b) >= 3:
            try:
                from scipy.stats import mannwhitneyu
                p = float(getattr(mannwhitneyu(a, b, alternative="two-sided"), "pvalue"))
            except Exception as exc:
                log.debug("compare_segment_blocks: Mann-Whitney failed: %s", exc)
                p = None
        metrics[key] = {"mean_a": ma, "sd_a": sa, "mean_b": mb, "sd_b": sb,
                        "diff": (mb - ma) if np.isfinite(ma) and np.isfinite(mb) else nan,
                        "p_descriptive": p}
    return {
        "block_s":     float(blocks_a.get("block_s", blocks_b.get("block_s", nan))),
        "n_blocks_a":  int(blocks_a.get("n_blocks", 0)),
        "n_blocks_b":  int(blocks_b.get("n_blocks", 0)),
        "n_dropped_a": int(blocks_a.get("n_dropped", 0)),
        "n_dropped_b": int(blocks_b.get("n_dropped", 0)),
        "metrics":     metrics,
        "note":        SEGMENT_COMPARISON_NOTE,
    }


def segment_comparison_text(cmp: dict, label_a: str = "A", label_b: str = "B") -> str:
    """Two-line on-screen / export summary of ``compare_segment_blocks``."""
    hr = cmp["metrics"]["hr_mean"]

    def _f(v: float, nd: int = 1) -> str:
        return f"{v:.{nd}f}" if v is not None and np.isfinite(v) else "—"

    p = hr.get("p_descriptive")
    p_txt = (f"descriptive Mann-Whitney on block means p = {p:.3g}"
             if p is not None else "too few blocks for a descriptive p (need ≥ 3 each)")
    return (f"HR, {cmp['block_s']:g} s block means: {label_a} {_f(hr['mean_a'])} ± "
            f"{_f(hr['sd_a'])} bpm (n = {cmp['n_blocks_a']} blocks)  ·  {label_b} "
            f"{_f(hr['mean_b'])} ± {_f(hr['sd_b'])} bpm (n = {cmp['n_blocks_b']})  ·  "
            f"Δ {_f(hr['diff'])} bpm  ·  {p_txt}\n{cmp['note']}")


def compare_segments(sa: dict, sb: dict) -> dict:
    """``compare_segment_blocks`` of two ``compute_segment_stats`` results,
    also written back into each segment dict as "n_blocks",
    "n_blocks_dropped", "hr_block_mean" and "hr_block_sd" (the block-level
    rows of the Compare Segments table and export)."""
    cmp = compare_segment_blocks(sa.get("blocks") or {}, sb.get("blocks") or {})
    hr = cmp["metrics"]["hr_mean"]
    for seg, side in ((sa, "a"), (sb, "b")):
        seg["n_blocks"]         = cmp[f"n_blocks_{side}"]
        seg["n_blocks_dropped"] = cmp[f"n_dropped_{side}"]
        seg["hr_block_mean"]    = hr[f"mean_{side}"]
        seg["hr_block_sd"]      = hr[f"sd_{side}"]
    return cmp
