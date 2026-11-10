"""
ecg.core.autonomic
──────────────────
Time-resolved cardiac sympathetic and parasympathetic indices (CSI, CPI)
from sliding-window Poincaré plots, after Candia-Rivera et al., "Modelling
the time-resolved modulations of cardiac activity in rats", J Physiol 2025
(reference MATLAB code: github.com/diegocandiar/robust_hrv --
compute_CSI_CVI_rat.m, compute_rCSI_rCVI_type.m, robustCovHRV.m).

Pure NumPy/SciPy/pandas, safe to call from background threads; no UI import.

Method (as compute_CSI_CVI_rat.m does it)
────────────────────────────────────────
Every RR interval carries the time of its END beat (``rr_interval_series``'s
``t_s``). One evaluation per beat i -- every beat later than the first
interval + T, except the very last such beat, which the reference code
drops (``nt = length(ixs) - 1``):

* window Ω_i = the intervals with t in [t_i − T, t_i], both bounds
  inclusive (T = 3 s);
* Poincaré pairs (RR_j, RR_j+1) inside Ω_i; SD1_i, SD2_i = square roots of
  the minor / major eigenvalue of their 2×2 covariance; CCD_i ("D") =
  √(m₁² + m₂²), m₁, m₂ the means of the two pair coordinates (5 % trimmed
  means in the rat code); timestamp = MEDIAN time of Ω_i's intervals;
* SD01, SD02, D0 = the same three on every pair of the recording; they
  re-centre the beat-wise series: SD1 ← SD1 − mean(SD1) + SD01 (same for
  SD2 and D);
* SD1, SD2, D resampled at 20 Hz (cubic spline, not-a-knot -- MATLAB's
  interp1 'spline'); D̃ = D mirrored about its mean;
* CSI = D̃ + ks·SD2  (a shorter cycle and larger slow oscillations raise it)
  CPI = D + kp·SD1   (a longer cycle and larger fast oscillations raise it)
  CSI_HR = D̃, CPI_HR = D, CSI_HRV = ks·SD2, CPI_HRV = kp·SD1.

All durations are in ms. Every term is a duration, so the unit only rescales
CSI/CPI; ks and kp are dimensionless and mean the same in s or ms.

Gap-aware adaptation (this app's convention, see analysis.rr_interval_series)
──────────────────────────────────────────────────────────────────────────────
The reference code runs on a clean IBI series. Here intervals outside the RR
bounds or spanning a beat removed by artifact correction are EXCLUDED, and:

* only valid intervals are evaluated or enter a window;
* a Poincaré pair joins two intervals that are both valid AND adjacent in
  the original series -- never across an excluded interval or a break;
* a window with fewer than ``min_pairs`` such pairs is withheld: SD1, SD2
  and D all NaN (so is a ``robust`` window with an undefined covariance);
* the resampling spline runs separately over each contiguous run of finite
  window values (≥ 4 points cubic, 2-3 points linear), NaN between runs;
  an ISOLATED computed window (both neighbours withheld, or the only
  window) has nothing to interpolate from and is absent from the resampled
  series -- it still enters the re-centring mean and the beat-wise table,
  and is disclosed (``params['n_windows_isolated']``, ``withheld_note``;
  ``withheld_reason`` when no resampled sample is left at all);
* the instantaneous heart rate comes from a spline of the valid RR
  intervals, one per run of adjacent valid intervals, NaN across a gap.

With nothing excluded AND every window holding at least ``min_pairs``
pairs, the results equal the reference code's (pinned by
tests/core/test_autonomic.py against a literal transliteration). The
``min_pairs`` floor is this module's addition -- the reference code computes
every window. With nothing excluded, a window ending on beat i holds
≥ min_pairs pairs exactly when its last min_pairs RR intervals sum to at
most T, so at the defaults (10 pairs, 3 s) windows are withheld wherever RR
averages more than 300 ms over 10 beats, i.e. a heart rate below about
200 bpm (bradycardic or anaesthetised mice). A partial loss is reported in
``params['withheld_note']`` (``withheld_reason`` only explains a result
with no window at all); ``min_pairs = 2`` restores the reference behaviour
whenever every window holds 2 pairs -- always within the default RR bounds.

Disclosures
───────────
* Covariance estimator. The paper text says the "robust" approach was used:
  Schäfer & Strimmer (2005) shrinkage (robustCovHRV.m -- variances shrunk
  toward their median, correlation shrunk toward 0). The published rat
  function compute_CSI_CVI_rat.m, however, uses plain ``cov()`` with 5 %
  trimmed means (its robustCovHRV call is commented out). Both are offered:
  ``"robust"`` (default, the paper text) and ``"rat_reference"`` (the
  code). The shrinkage correlation is undefined when a window's first or
  second Poincaré coordinate is constant (zero RR variance -- frequent in
  very low-HRV recordings at a coarse sampling rate, where RR values are
  quantised): such windows are withheld under ``"robust"`` (SD1, SD2 and D
  all NaN, as for too few pairs -- the MATLAB code would error there) and
  counted in ``params['n_windows_cov_undefined']``; the sample-covariance methods
  stay defined there. With two coordinates the shrinkage keeps the trace
  (the median of two variances is their mean, so SD1² + SD2² equals the
  sample value) and can only narrow the eigenvalue spread: in EVERY window
  its SD1 is at least, and its SD2 at most, the sample-covariance value --
  the ellipse is rounded, by a few % on ~25-pair mouse windows (synthetic
  series: SD1 +4 to +7 %, SD2 −1 to −5 %, data-dependent). The app's
  whole-recording Poincaré SD1_LW/SD2_LW (analysis.compute_poincare_metrics)
  use sklearn's Ledoit-Wolf shrinkage toward a scaled identity, a different
  target (and ddof 0): on ~25-pair windows it too pulls SD1 up and SD2
  down, by a different, usually larger amount (synthetic: SD1 +3 to +11 %,
  SD2 −4 to −10 % vs the exact covariance). It is offered as
  ``"ledoit_wolf"`` for comparison only.
* ks = 3, kp = 12 are the paper's RAT values (SHR strain, rest vs
  dobutamine). They are not validated in mice. CSI and CPI are linear in
  them: between any two periods, ΔCSI = −ΔD + ks·ΔSD2 and ΔCPI = ΔD +
  kp·ΔSD1 (the re-centring constants cancel), so per-event deltas of D,
  SD1 and SD2 are enough to re-calibrate ks/kp on a mouse cohort; see
  ``with_coefficients`` to recompose a result without recomputing it.
* Which index gets the flipped D. As extracted, the paper's printed
  equations (5)-(6) put the tilde (flipped D) on CPI; the published code
  flips D for CSI and keeps it for CPI, which is what the physiology
  requires (a shorter cycle must raise CSI) -- the code is followed.
* Trimmed mean. MATLAB's trimmean(x, 5) removes k = round(n·5/200) values
  from EACH end, a half-integer k rounding DOWN (MATLAB documentation,
  flag 'round'); ``scipy.stats.trim_mean`` floors instead (see
  ``matlab_trimmean``).
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from ecg.core.analysis import rr_exclusion_summary, rr_interval_series
from ecg.core.models import MouseECG


# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════

# Covariance / ellipse-centre estimators, key -> UI label (display order).
COV_METHODS: "OrderedDict[str, str]" = OrderedDict([
    ("robust",        "Robust shrinkage (Schafer-Strimmer, paper text)"),
    ("rat_reference", "Rat reference code (cov + 5 % trimmed mean)"),
    ("exact",         "Exact covariance"),
    ("approximate",   "Approximate (closed-form SD1/SD2)"),
    ("ledoit_wolf",   "Ledoit-Wolf (sklearn)"),
])
DEFAULT_METHOD = "robust"

# One-line description of each method, stored with every result.
METHOD_NOTES: "dict[str, str]" = {
    "robust": (
        "Schafer-Strimmer shrinkage covariance (robustCovHRV.m: variances "
        "shrunk toward their median, correlation toward 0) with 5 % trimmed-"
        "mean CCD -- the method the paper text describes. The published rat "
        "function (compute_CSI_CVI_rat.m) uses plain cov() instead. A window "
        "in which either Poincaré coordinate is constant (zero RR variance, "
        "e.g. very low HRV at a coarse sampling rate) has an undefined "
        "correlation and is withheld whole -- SD1, SD2 and CCD NaN "
        "(robustCovHRV.m returns NaN there)."),
    "rat_reference": (
        "Sample covariance with 5 % trimmed-mean CCD, exactly as the "
        "published compute_CSI_CVI_rat.m."),
    "exact": (
        "Sample covariance with plain-mean CCD (compute_rCSI_rCVI_type.m "
        "'exact')."),
    "approximate": (
        "Closed form SD1 = sqrt(var(dRR)/2), SD2 = sqrt(2 var(RR) - "
        "var(dRR)/2) with plain-mean CCD (compute_rCSI_rCVI_type.m "
        "'approximate'); dRR only between adjacent valid intervals. Fixed "
        "identity-line axes, so SD1 > SD2 is possible."),
    "ledoit_wolf": (
        "sklearn Ledoit-Wolf shrinkage toward a scaled identity (as the "
        "app's whole-recording SD1_LW/SD2_LW) with plain-mean CCD. Not used "
        "by the paper; on short windows it pulls SD1 up and SD2 down."),
}

DEFAULT_WINDOW_S  = 3.0    # s  -- sliding window T (paper: 3 s for rats)
DEFAULT_FS_OUT    = 20.0   # Hz -- resampling rate (compute_CSI_CVI_rat.m)
DEFAULT_MIN_PAIRS = 10     # fewer valid Poincaré pairs in a window -> NaN
TRIM_PERCENT      = 5.0    # % -- MATLAB trimmean(x, 5) for the CCD

# Candia-Rivera et al., J Physiol 2025: ks and kp were swept over 1..20 on
# the SHR rat strain (rest vs dobutamine) and set at the inflection point of
# the paired Wilcoxon Z (ks = 3 for CSI, kp = 12 for CPI), then checked on
# WKY rats. These are RAT values -- they have NOT been validated in mice.
KS_RAT_SHR = 3.0
KP_RAT_SHR = 12.0

# Shortest finite run resampled with a cubic spline (shorter -> linear).
_SPLINE_MIN_POINTS = 4

# Element budget of one padded (beats × pairs) block in the batched path.
_CHUNK_ELEMS = 2_000_000

_SHRINKAGE   = "shrinkage"
_SAMPLE      = "sample"
_CLOSED_FORM = "closed_form"
_LEDOIT_WOLF = "ledoit_wolf"

_COV_KIND = {
    "robust":        _SHRINKAGE,
    "rat_reference": _SAMPLE,
    "exact":         _SAMPLE,
    "approximate":   _CLOSED_FORM,
    "ledoit_wolf":   _LEDOIT_WOLF,
}
_TRIMMED_D = {
    "robust":        True,
    "rat_reference": True,
    "exact":         False,
    "approximate":   False,
    "ledoit_wolf":   False,
}


def _check_method(method: str) -> None:
    if method not in COV_METHODS:
        raise ValueError(f"Unknown covariance method {method!r}; expected one "
                         f"of {', '.join(COV_METHODS)}.")


def _finite_float(value: float, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return v


def _positive_float(value: float, name: str) -> float:
    v = _finite_float(value, name)
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return v


def _min_pairs_int(value) -> int:
    """*value* as an integer ≥ 2 (an int, or a float with an integral value)
    that fits in int64; anything else -- None, NaN, ±Inf, a fraction, a
    non-numeric string, a value past int64 -- raises ValueError, never
    TypeError/OverflowError."""
    msg = f"min_pairs must be an integer ≥ 2, got {value!r}."
    try:
        n = operator.index(value)                  # int, numpy integer, bool
    except TypeError:
        try:
            v = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(msg) from None
        if not (math.isfinite(v) and v.is_integer()):
            raise ValueError(msg) from None
        n = int(v)
    if n < 2 or n > np.iinfo(np.int64).max:       # compared with int64 counts
        raise ValueError(msg)
    return n


def calibration_note(ks: float, kp: float) -> str:
    """Disclosure of the ks/kp coefficients, stored with every result."""
    tail = ("CSI/CPI are linear in ks/kp (dCSI = -dD + ks*dSD2, "
            "dCPI = dD + kp*dSD1), so they can be re-calibrated on a mouse "
            "cohort from per-event deltas of D, SD1 and SD2.")
    if float(ks) == KS_RAT_SHR and float(kp) == KP_RAT_SHR:
        return ("ks = 3, kp = 12: rat (SHR) calibration of Candia-Rivera et "
                "al., J Physiol 2025 (inflection of the rest-vs-dobutamine "
                "Wilcoxon Z over k = 1-20). NOT validated in mice. " + tail)
    return (f"ks = {float(ks):g}, kp = {float(kp):g} (user-set; the paper's "
            f"rat SHR values ks = 3, kp = 12 are not validated in mice). " + tail)


# ════════════════════════════════════════════════════════════
#  MATLAB-FAITHFUL BUILDING BLOCKS
# ════════════════════════════════════════════════════════════

def _matlab_clip01(value: float) -> float:
    """``max(min(value, 1), 0)`` with MATLAB semantics: min/max ignore a NaN
    operand, so NaN (0/0) -> 1 and +Inf (x/0) -> 1."""
    v = float(value)
    if math.isnan(v) or v > 1.0:
        return 1.0
    return 0.0 if v < 0.0 else v


def _matlab_clip01_arr(values: np.ndarray) -> np.ndarray:
    """Element-wise ``_matlab_clip01``."""
    v = np.asarray(values, dtype=np.float64)
    v = np.where(np.isnan(v) | (v > 1.0), 1.0, v)
    return np.maximum(v, 0.0)


def _matlab_trim_count(n, percent: float) -> np.ndarray:
    """Values MATLAB's trimmean(x, percent) removes from EACH end of n values
    (default flag 'round'): k = n·percent/200 rounded to the nearest integer,
    a half-integer rounding DOWN -- MATLAB evaluates round(k - eps(k)),
    emulated here exactly (round half away from zero on k - eps(k) >= 0)."""
    k = np.asarray(n, dtype=np.float64) * float(percent) / 200.0
    y = k - np.spacing(k)
    r = np.floor(y)
    return (r + (y - r >= 0.5)).astype(np.int64)


def matlab_trimmean(x: np.ndarray, percent: float = TRIM_PERCENT) -> float:
    """MATLAB ``trimmean(x, percent)`` (default 'round' flag).

    NaNs are dropped (MATLAB treats them as missing). k = n·percent/200 is
    rounded to the nearest integer, a half-integer rounding down (MATLAB
    documentation, flag 'round'), and the k smallest and k largest values
    are removed before averaging. NaN when nothing would remain.

    Differs from ``scipy.stats.trim_mean(x, percent / 200)``, which removes
    floor(n·percent/200) values from each end: with percent = 5 and the
    ~24 pairs of a 3 s mouse window, MATLAB trims one value from each end
    (k = 0.6 -> 1) while scipy trims none; at n = 20 (k = 0.5) both trim
    none, at n = 60 (k = 1.5) both trim one, at n = 30 (k = 0.75) MATLAB
    trims one and scipy none.
    """
    pct = float(percent)
    if not (0.0 <= pct <= 100.0):
        raise ValueError(f"percent must be within [0, 100], got {percent!r}.")
    v = np.asarray(x, dtype=np.float64).ravel()
    v = np.sort(v[~np.isnan(v)])
    n = len(v)
    if n == 0:
        return float("nan")
    k0 = int(_matlab_trim_count(n, pct))
    if k0 >= n / 2:
        return float("nan")
    return float(np.mean(v[k0:n - k0]))


def _varcov(x: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """robustCovHRV.m's varcov: the sample covariance S of the column-centred
    data and VS, the estimated variance of each S(i, j) --
    var_k(xc_ki·xc_kj, ddof=1) · n / (n - 1)²  (Schäfer & Strimmer 2005)."""
    n = x.shape[0]
    xc = x - x.mean(axis=0)
    s = np.atleast_2d(np.cov(xc, rowvar=False, ddof=1))
    w = xc[:, :, None] * xc[:, None, :]
    vs = w.var(axis=0, ddof=1) * n / (n - 1) ** 2
    return s, vs


def shrinkage_cov_schafer_strimmer(
    x: np.ndarray,
) -> "tuple[np.ndarray, float, float]":
    """Schäfer-Strimmer shrinkage covariance -- a faithful port of
    robustCovHRV.m (Schäfer & Strimmer 2005, Opgen-Rhein & Strimmer 2007).

    *x* is an n × p data matrix (a 1-D array is one column). Returns
    ``(cov, lambda_var, lambda_cor)``:

    * variances v (sample, ddof = 1) are shrunk toward their MEDIAN:
      λ_var = Σ Var^(v_i) / Σ (v_i − median(v))²;
    * the correlation matrix r is shrunk toward the identity (off-diagonal
      target 0): λ_cor = Σ_{i>j} Var^(r_ij) / Σ_{i>j} r_ij²;
    * Var^(·) is varcov's estimator var_k(w_kij, ddof=1) · n/(n − 1)² of
      the products w_kij of centred (resp. standardised) columns;
    * both λ are clipped to [0, 1] with MATLAB's min/max NaN semantics
      (0/0 -> 1, x/0 -> 1);
    * cov = diag(√v*) · r* · diag(√v*), v* and r* the shrunk estimates.

    As in MATLAB: p = 1 returns the sample variance (λ undefined -> NaN
    here); a constant column makes the correlation 0/0, so λ_cor = 1 and
    the whole matrix NaN (MATLAB's eig would then error). Constancy is
    tested exactly (all values equal) rather than left to a 0/0 that a
    rounding error in the column mean could turn into a finite value, so
    the batched window path (``_bcov_shrinkage``) applies the very same
    rule. Fewer than 2 rows -> all NaN.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"x must be an n × p matrix, got shape {a.shape}.")
    n, p = a.shape
    nan = float("nan")
    if n < 2:
        return np.full((p, p), nan), nan, nan
    if p == 1:
        return np.atleast_2d(np.var(a[:, 0], ddof=1)), nan, nan
    with np.errstate(divide="ignore", invalid="ignore"):
        # varshrink -- eqns 10-11 of Opgen-Rhein & Strimmer (2007)
        s, vs = _varcov(a)
        v, vv = np.diag(s).copy(), np.diag(vs)
        vtarget = float(np.median(v))
        lam_var = _matlab_clip01(np.sum(vv) / np.sum((v - vtarget) ** 2))
        sv = (1.0 - lam_var) * v + lam_var * vtarget
        # corshrink -- Schäfer & Strimmer (2005), p. 4
        xc = a - a.mean(axis=0)
        sx = xc / xc.std(axis=0, ddof=1)
        # a constant column standardises to 0/0 in exact arithmetic
        sx[:, np.ptp(a, axis=0) == 0] = np.nan
        r, vr = _varcov(sx)
        lam_cor = _matlab_clip01(np.sum(np.tril(vr, -1)) / np.sum(np.tril(r, -1) ** 2))
        rhat = (1.0 - lam_cor) * r
        rhat[np.eye(p, dtype=bool)] = 1.0
        dsv = np.diag(np.sqrt(sv))
        cov = dsv @ rhat @ dsv
    return cov, lam_var, lam_cor


def _sd_from_cov(c: np.ndarray) -> "tuple[float, float]":
    """(√λ_min, √λ_max) of a symmetric 2×2 covariance, NaN if not finite."""
    if not np.all(np.isfinite(c)):
        return float("nan"), float("nan")
    ev = np.linalg.eigvalsh(c)
    return float(np.sqrt(max(float(ev[0]), 0.0))), float(np.sqrt(max(float(ev[-1]), 0.0)))


def window_poincare(
    ibi1:   np.ndarray,
    ibi2:   np.ndarray,
    method: str = DEFAULT_METHOD,
    ibi:    "Optional[np.ndarray]" = None,
) -> "tuple[float, float, float]":
    """(SD1, SD2, D) of one set of Poincaré pairs (ibi1[j], ibi2[j]).

    SD1/SD2 are the square roots of the minor/major eigenvalue of the
    pairs' covariance, estimated per *method* (see COV_METHODS):
    ``robust`` (shrinkage_cov_schafer_strimmer), ``rat_reference`` and
    ``exact`` (sample covariance), ``ledoit_wolf`` (sklearn LedoitWolf), or
    ``approximate`` -- the closed form SD1 = √(var(Δ)/2),
    SD2 = √(2·var(ibi) − var(Δ)/2) with Δ = ibi2 − ibi1 (so only differences
    between the paired, i.e. adjacent valid, intervals) and var(ibi) over
    *ibi*, the window's valid intervals; *ibi* defaults to
    ``ibi1 + [ibi2[-1]]``, which is the window when the pairs form one
    unbroken chain. A negative closed-form SD2² is clipped to 0. The closed
    form measures the spread across / along the identity line (fixed
    axes), not the ellipse's minor / major axes: with negatively correlated
    successive intervals its SD1 exceeds its SD2, whereas the eigenvalue
    methods always return the minor then the major spread.

    D = √(m₁² + m₂²), m the means of ibi1 and ibi2 -- MATLAB 5 % trimmed
    means for ``robust`` and ``rat_reference``, plain means otherwise,
    exactly as the reference MATLAB methods.

    Pairs with a non-finite member are ignored; fewer than 2 pairs give
    NaN. A non-finite covariance (``robust`` on a constant column) withholds
    the window as a whole -- SD1, SD2 and D all NaN, exactly like a window
    with too few pairs -- so D never outlives the SD1/SD2 it is combined
    with (compute_autonomic_indices re-centres, flips and resamples D on the
    support of SD1/SD2).
    """
    _check_method(method)
    x = np.asarray(ibi1, dtype=np.float64).ravel()
    y = np.asarray(ibi2, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError(f"ibi1 and ibi2 differ in length ({len(x)} vs {len(y)}).")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    nan = float("nan")
    if len(x) < 2:
        return nan, nan, nan

    if _TRIMMED_D[method]:
        mx, my = matlab_trimmean(x, TRIM_PERCENT), matlab_trimmean(y, TRIM_PERCENT)
    else:
        mx, my = float(np.mean(x)), float(np.mean(y))
    d = float(np.sqrt(mx ** 2 + my ** 2))

    kind = _COV_KIND[method]
    if kind == _CLOSED_FORM:
        if ibi is None:
            w = np.r_[x, y[-1:]]
        else:
            w = np.asarray(ibi, dtype=np.float64).ravel()
            w = w[np.isfinite(w)]
        if len(w) < 2:
            return nan, nan, d
        var_d = float(np.var(y - x, ddof=1))
        var_w = float(np.var(w, ddof=1))
        sd1 = math.sqrt(0.5 * var_d)
        sd2 = math.sqrt(max(2.0 * var_w - 0.5 * var_d, 0.0))
        return sd1, sd2, d
    if kind == _SAMPLE:
        c = np.cov(x, y, ddof=1)
    elif kind == _SHRINKAGE:
        c = shrinkage_cov_schafer_strimmer(np.column_stack((x, y)))[0]
    else:
        from sklearn.covariance import LedoitWolf
        c = LedoitWolf().fit(np.column_stack((x, y))).covariance_
    sd1, sd2 = _sd_from_cov(c)
    if not (math.isfinite(sd1) and math.isfinite(sd2)):
        return nan, nan, nan                       # withheld as a whole
    return sd1, sd2, d


# ════════════════════════════════════════════════════════════
#  BATCHED WINDOWS (same formulas, padded beats × pairs blocks)
# ════════════════════════════════════════════════════════════
#
# A recording has one window per beat (~30 000 per hour of mouse ECG), so
# the windows are evaluated as NaN-padded (beats × max pairs) blocks rather
# than one window_poincare() call each. Every statistic follows the same
# operation sequence as window_poincare / shrinkage_cov_schafer_strimmer
# (the tests pin both paths against each other and against MATLAB
# transliterations).

def _gather(values: np.ndarray, start: np.ndarray, count: np.ndarray, width: int) -> np.ndarray:
    """Rows values[start[r] : start[r] + count[r]], NaN-padded to *width*."""
    cols = np.arange(width)
    mask = cols[None, :] < count[:, None]
    out = np.full((len(start), width), np.nan)
    if width and len(values):
        idx = np.minimum(start[:, None] + cols[None, :], len(values) - 1)
        out[mask] = values[idx[mask]]
    return out


def _rmean(a: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.nansum(a, axis=1) / n


def _rcentre(a: np.ndarray, n: np.ndarray) -> np.ndarray:
    return a - _rmean(a, n)[:, None]


def _rcov(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Row-wise np.cov(a, b)[0, 1] semantics (re-centres, ddof = 1)."""
    return np.nansum(_rcentre(a, n) * _rcentre(b, n), axis=1) / (n - 1.0)


def _rtrimmean(a: np.ndarray, n: np.ndarray, percent: float) -> np.ndarray:
    """Row-wise matlab_trimmean of NaN-padded rows holding n values each."""
    srt = np.sort(a, axis=1)                       # NaN padding sorts last
    k0 = _matlab_trim_count(n, percent)
    cs = np.concatenate((np.zeros((len(a), 1)),
                         np.cumsum(np.nan_to_num(srt, nan=0.0), axis=1)), axis=1)
    rows = np.arange(len(a))
    ni = n.astype(np.int64)
    keep = ni - 2 * k0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (cs[rows, np.maximum(ni - k0, 0)] - cs[rows, np.minimum(k0, ni)]) / keep
    return np.where(keep > 0, out, np.nan)


def _bcov_sample(x, y, n):
    return _rcov(x, x, n), _rcov(y, y, n), _rcov(x, y, n)


def _bcov_shrinkage(x, y, n):
    """Row-wise shrinkage_cov_schafer_strimmer for p = 2.

    A row with a constant coordinate has an undefined correlation and gets a
    NaN covariance, as in the single-window port. It must be flagged
    explicitly: its standardised column is 0/0 = NaN, which the NaN-padding
    sums below would silently drop (giving r = 0 and a spurious finite
    circle)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        xc, yc = _rcentre(x, n), _rcentre(y, n)
        k = n / (n - 1.0) ** 2
        # varshrink
        sxx, syy = _rcov(xc, xc, n), _rcov(yc, yc, n)
        wx, wy = xc * xc, yc * yc
        vvx, vvy = _rcov(wx, wx, n) * k, _rcov(wy, wy, n) * k
        vt = (sxx + syy) / 2.0                     # median of two variances
        lam_v = _matlab_clip01_arr((vvx + vvy) / ((sxx - vt) ** 2 + (syy - vt) ** 2))
        svx = (1.0 - lam_v) * sxx + lam_v * vt
        svy = (1.0 - lam_v) * syy + lam_v * vt
        # corshrink
        sx = xc / np.sqrt(sxx)[:, None]
        sy = yc / np.sqrt(syy)[:, None]
        sxc, syc = _rcentre(sx, n), _rcentre(sy, n)
        r = _rcov(sxc, syc, n)
        w = sxc * syc
        vr = _rcov(w, w, n) * k
        lam_c = _matlab_clip01_arr(vr / r ** 2)
        rxy = (1.0 - lam_c) * r
        qx, qy = np.sqrt(svx), np.sqrt(svy)
        cxx, cyy, cxy = qx * qx, qy * qy, qx * rxy * qy
    # every row holds >= 2 values (the caller withholds shorter windows)
    const = ((np.nanmax(x, axis=1) == np.nanmin(x, axis=1))
             | (np.nanmax(y, axis=1) == np.nanmin(y, axis=1)))
    bad = const | ~np.isfinite(rxy)                # 0·NaN poisons diag(√v)·r·diag(√v)
    cxx[bad] = cyy[bad] = cxy[bad] = np.nan
    return cxx, cyy, cxy


def _bcov_ledoit_wolf(x, y, n):
    """Row-wise sklearn LedoitWolf().fit(pairs).covariance_ for p = 2
    (sklearn.covariance.ledoit_wolf_shrinkage, non-blocked)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        xc, yc = _rcentre(x, n), _rcentre(y, n)
        sx2, sy2, sxy = (np.nansum(xc * xc, axis=1), np.nansum(yc * yc, axis=1),
                         np.nansum(xc * yc, axis=1))
        exx, eyy, exy = sx2 / n, sy2 / n, sxy / n
        mu = (exx + eyy) / 2.0
        beta_ = np.nansum((xc * xc + yc * yc) ** 2, axis=1)
        delta_ = (sx2 ** 2 + sy2 ** 2 + 2.0 * sxy ** 2) / n ** 2
        beta = 1.0 / (2.0 * n) * (beta_ / n - delta_)
        delta = (delta_ - 2.0 * mu * (exx + eyy) + 2.0 * mu ** 2) / 2.0
        beta = np.minimum(beta, delta)
        shrink = np.where(beta == 0, 0.0, beta / delta)
    return ((1.0 - shrink) * exx + shrink * mu,
            (1.0 - shrink) * eyy + shrink * mu,
            (1.0 - shrink) * exy)


_BATCH_COV = {_SAMPLE: _bcov_sample, _SHRINKAGE: _bcov_shrinkage,
              _LEDOIT_WOLF: _bcov_ledoit_wolf}


def _beigen(cxx, cyy, cxy) -> "tuple[np.ndarray, np.ndarray]":
    sd1 = np.full(len(cxx), np.nan)
    sd2 = np.full(len(cxx), np.nan)
    ok = np.isfinite(cxx) & np.isfinite(cyy) & np.isfinite(cxy)
    if ok.any():
        c = np.empty((int(ok.sum()), 2, 2))
        c[:, 0, 0], c[:, 1, 1] = cxx[ok], cyy[ok]
        c[:, 0, 1] = c[:, 1, 0] = cxy[ok]
        ev = np.linalg.eigvalsh(c)
        sd1[ok] = np.sqrt(np.maximum(ev[:, 0], 0.0))
        sd2[ok] = np.sqrt(np.maximum(ev[:, -1], 0.0))
    return sd1, sd2


def _batch_window_poincare(
    px: np.ndarray, py: np.ndarray, p_start: np.ndarray, p_count: np.ndarray,
    rv: np.ndarray, w_start: np.ndarray, w_count: np.ndarray, method: str,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """window_poincare for every window at once. Window r holds the pairs
    (px, py)[p_start[r] : p_start[r] + p_count[r]] and the valid intervals
    rv[w_start[r] : w_start[r] + w_count[r]] (``approximate`` only). The
    caller withholds windows with too few pairs; every row passed here has
    at least 2. A row whose covariance is undefined is NaN in SD1, SD2 and
    D alike, as in window_poincare."""
    m = len(p_start)
    sd1, sd2, d = np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan)
    if m == 0:
        return sd1, sd2, d
    kind = _COV_KIND[method]
    width_p = int(p_count.max())
    width_w = int(w_count.max()) if kind == _CLOSED_FORM else 0
    rows_per_chunk = max(1, _CHUNK_ELEMS // max(width_p, width_w, 1))
    for lo in range(0, m, rows_per_chunk):
        sl = slice(lo, min(lo + rows_per_chunk, m))
        n = p_count[sl].astype(np.float64)
        x = _gather(px, p_start[sl], p_count[sl], width_p)
        y = _gather(py, p_start[sl], p_count[sl], width_p)
        if _TRIMMED_D[method]:
            mx, my = _rtrimmean(x, n, TRIM_PERCENT), _rtrimmean(y, n, TRIM_PERCENT)
        else:
            mx, my = _rmean(x, n), _rmean(y, n)
        d[sl] = np.sqrt(mx ** 2 + my ** 2)
        if kind == _CLOSED_FORM:
            nw = w_count[sl].astype(np.float64)
            w = _gather(rv, w_start[sl], w_count[sl], width_w)
            dd = y - x
            var_d = _rcov(dd, dd, n)
            var_w = _rcov(w, w, nw)
            sd1[sl] = np.sqrt(0.5 * var_d)
            sd2[sl] = np.sqrt(np.maximum(2.0 * var_w - 0.5 * var_d, 0.0))
        else:
            sd1[sl], sd2[sl] = _beigen(*_BATCH_COV[kind](x, y, n))
    # an undefined covariance withholds the whole window (window_poincare)
    d[~(np.isfinite(sd1) & np.isfinite(sd2))] = np.nan
    return sd1, sd2, d


# ════════════════════════════════════════════════════════════
#  RESAMPLING
# ════════════════════════════════════════════════════════════

def _time_grid(t0: float, t1: float, fs_out: float) -> np.ndarray:
    """MATLAB ``t0 : 1/fs_out : t1`` (t0 + k·step, end point kept when it
    lies on the grid up to rounding)."""
    step = 1.0 / fs_out
    q = (t1 - t0) / step
    if not np.isfinite(q) or q < 0:
        return np.array([], dtype=np.float64)
    n = int(np.floor(q + max(1e-9, 1e-12 * q)))
    return t0 + np.arange(n + 1) * step


def _grid_runs(t_src: np.ndarray, v_src: np.ndarray, t_out: np.ndarray
               ) -> "list[tuple[int, int, int, int]]":
    """(s, e, lo, hi) of every contiguous run v_src[s:e] of finite values
    (at strictly increasing *t_src*): t_out[lo:hi] are the output samples
    inside the run's time span (none when hi <= lo)."""
    if len(t_src) == 0:
        return []
    finite = np.isfinite(v_src) & np.isfinite(t_src)
    edges = np.diff(np.r_[0, finite.astype(np.int8), 0])
    # tolerance for a grid end point a rounding error past the last source time
    tol = 1e-6 * (float(t_out[1] - t_out[0]) if len(t_out) > 1 else 1.0)
    runs = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lo = int(np.searchsorted(t_out, t_src[s] - tol, side="left"))
        hi = int(np.searchsorted(t_out, t_src[e - 1] + tol, side="right"))
        runs.append((int(s), int(e), lo, hi))
    return runs


def _unresampled(s: int, e: int, lo: int, hi: int) -> bool:
    """True for a run that _resample_runs places nowhere on the grid: an
    isolated point (nothing to interpolate from) or a run whose span holds
    no output sample."""
    return e - s < 2 or hi <= lo


def _resample_runs(t_src: np.ndarray, v_src: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """*v_src* (at strictly increasing *t_src*) on *t_out*, one interpolant
    per contiguous run of finite values: a not-a-knot cubic spline (MATLAB
    interp1 'spline') for runs of ≥ 4 points, linear for 2-3 points, nothing
    for an isolated point (MATLAB's interp1 needs 2 points too; the caller
    discloses such points). NaN outside every run -- never bridged."""
    out = np.full(len(t_out), np.nan)
    if len(t_out) == 0:
        return out
    for s, e, lo, hi in _grid_runs(t_src, v_src, t_out):
        if _unresampled(s, e, lo, hi):
            continue
        ts, vs, tq = t_src[s:e], v_src[s:e], t_out[lo:hi]
        if e - s >= _SPLINE_MIN_POINTS:
            out[lo:hi] = CubicSpline(ts, vs, bc_type="not-a-knot")(tq)
        else:
            out[lo:hi] = np.interp(tq, ts, vs)
    return out


def _nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    f = np.isfinite(a)
    return float(a[f].mean()) if f.any() else float("nan")


def _flipsig(sig: np.ndarray) -> np.ndarray:
    """MATLAB flipsig: *sig* mirrored about its (nan)mean."""
    m = _nanmean(sig)
    return -(sig - m) + m


# ════════════════════════════════════════════════════════════
#  RESULT
# ════════════════════════════════════════════════════════════

@dataclass(eq=False)
class AutonomicResult:
    """Time-resolved CSI/CPI of one recording (all durations in ms).

    Per evaluated beat (window):
        t_beat   : window timestamp -- median time of its valid intervals, s
        t_end    : window end -- the evaluated beat's interval end, s
        sd1_raw, sd2_raw, d_raw : window SD1, SD2, CCD before re-centring
                   (all three NaN for a withheld window)
        n_pairs  : valid adjacent Poincaré pairs in the window
    Recording level:
        sd01, sd02, d0 : SD1, SD2, CCD over every valid pair
    Resampled on ``t`` (``fs_out`` Hz, t_beat[0] to t_beat[-1]):
        d        : re-centred CCD = CPI_HR (alias ``cpi_hr``)
        d_flip   : d mirrored about its mean = CSI_HR (alias ``csi_hr``)
        sd1, sd2 : re-centred SD1, SD2
        cpi_hrv  : kp·sd1;   csi_hrv : ks·sd2
        cpi      : d + cpi_hrv;   csi : d_flip + csi_hrv
        hr_bpm   : heart rate from the valid-RR spline (NaN across gaps)
    params : inputs, exclusion counts and disclosures (method label and
             note, calibration_note, withheld_reason, withheld_note).
    """
    t_beat:  np.ndarray
    t_end:   np.ndarray
    sd1_raw: np.ndarray
    sd2_raw: np.ndarray
    d_raw:   np.ndarray
    n_pairs: np.ndarray
    sd01:    float
    sd02:    float
    d0:      float
    t:       np.ndarray
    d:       np.ndarray
    d_flip:  np.ndarray
    sd1:     np.ndarray
    sd2:     np.ndarray
    cpi_hrv: np.ndarray
    csi_hrv: np.ndarray
    cpi:     np.ndarray
    csi:     np.ndarray
    hr_bpm:  np.ndarray
    params:  dict = field(default_factory=dict)

    @property
    def cpi_hr(self) -> np.ndarray:
        """Heart-rate (cycle-length) part of CPI: the re-centred CCD."""
        return self.d

    @property
    def csi_hr(self) -> np.ndarray:
        """Heart-rate part of CSI: the re-centred CCD flipped about its mean."""
        return self.d_flip

    @property
    def withheld_reason(self) -> "Optional[str]":
        """Why nothing could be computed, or None."""
        return self.params.get("withheld_reason")

    @property
    def withheld_note(self) -> "Optional[str]":
        """How many windows were withheld and why, when some (not all) were;
        else None."""
        return self.params.get("withheld_note")


def _compose(d, d_flip, sd1, sd2, ks: float, kp: float) -> dict:
    cpi_hrv = kp * sd1
    csi_hrv = ks * sd2
    return {"cpi_hrv": cpi_hrv, "csi_hrv": csi_hrv,
            "cpi": d + cpi_hrv, "csi": d_flip + csi_hrv}


def with_coefficients(res: AutonomicResult, ks: float, kp: float) -> AutonomicResult:
    """*res* recomposed with other ks/kp, without recomputing any window
    (CSI/CPI are linear in them) -- e.g. for a k = 1..20 calibration sweep.
    Identical to compute_autonomic_indices(..., ks=ks, kp=kp)."""
    ks = _finite_float(ks, "ks")
    kp = _finite_float(kp, "kp")
    params = dict(res.params)
    params.update(ks=ks, kp=kp, calibration_note=calibration_note(ks, kp))
    return dataclasses.replace(res, params=params,
                               **_compose(res.d, res.d_flip, res.sd1, res.sd2, ks, kp))


# ════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════

def compute_autonomic_indices(
    rpeaks:    np.ndarray,
    fs:        float,
    window_s:  float = DEFAULT_WINDOW_S,
    method:    str   = DEFAULT_METHOD,
    ks:        float = KS_RAT_SHR,
    kp:        float = KP_RAT_SHR,
    fs_out:    float = DEFAULT_FS_OUT,
    min_pairs: int   = DEFAULT_MIN_PAIRS,
    rr_min_ms: float = MouseECG.RR_MIN_MS,
    rr_max_ms: float = MouseECG.RR_MAX_MS,
    breaks:    "Optional[np.ndarray]" = None,
) -> AutonomicResult:
    """Time-resolved CSI/CPI of the R-peaks *rpeaks* (samples at *fs* Hz).

    Follows compute_CSI_CVI_rat.m beat by beat (see the module docstring):
    windows of *window_s* s ending on each evaluated beat (bounds
    inclusive, the last beat dropped as the reference code does, median
    timestamp), SD1/SD2/D per *method* (COV_METHODS, see window_poincare),
    re-centred on the whole-recording SD01/SD02/D0 (same method),
    resampled at *fs_out* Hz, CSI = D̃ + ks·SD2 and CPI = D + kp·SD1.

    Gap-aware: validity comes from ``rr_interval_series(rpeaks, fs,
    rr_min_ms, rr_max_ms, breaks)``; only valid intervals are evaluated,
    pairs only join two valid ADJACENT intervals (never across an excluded
    interval or a removed-beat break), a window with fewer than
    *min_pairs* pairs is NaN, and the resampling and the heart-rate spline
    run per contiguous run, NaN between runs. A zero-length interval (a
    duplicated R-peak, which only rr_min_ms = 0 would let through) is
    excluded too and counted as too short (``params['n_zero_length']``).

    ``params['n_windows_withheld']`` counts every NaN window -- a withheld
    window is NaN in SD1, SD2 and D alike: those with too few pairs
    (``n_windows_few_pairs``) plus those whose covariance is undefined
    (``n_windows_cov_undefined`` -- ``robust`` on a window with a constant
    Poincaré coordinate). ``params['n_windows_isolated']`` counts the
    COMPUTED windows the resampling cannot place -- isolated ones (both
    neighbours withheld, or the only window: nothing to interpolate from;
    or, at a coarse fs_out, a run between two output samples): finite in
    the beat-wise table and in the re-centring mean, absent from every
    resampled series. ``params['withheld_reason']`` explains a result with
    no resampled CSI/CPI sample at all (no computed window, or only
    isolated ones); ``params['withheld_note']`` explains a partial loss
    (how many windows and why -- with nothing excluded, a 3 s window lacks
    10 pairs wherever HR is below about 200 bpm -- and the isolated ones),
    else None. ``params['n_breaks']`` counts the distinct finite *breaks*
    that split an analysed interval (strictly between two of *rpeaks*);
    breaks outside the peaks' span or sitting on a peak split nothing and
    are not counted, so passing a whole recording's breaks with a
    segment's peaks reports only the segment's.

    ks = 3, kp = 12 are the paper's rat (SHR) values -- not validated in
    mice (see ``params['calibration_note']``).

    Raises ValueError for an unknown method, a non-positive window_s or
    fs_out, a min_pairs that is not an integer ≥ 2 (None, NaN and Inf
    included), non-finite ks/kp, and (via rr_interval_series) an invalid
    fs or RR bounds. A recording too short for any window gives an empty
    result with ``params['withheld_reason']`` set.
    """
    _check_method(method)
    window_s = _positive_float(window_s, "window_s")
    fs_out = _positive_float(fs_out, "fs_out")
    ks = _finite_float(ks, "ks")
    kp = _finite_float(kp, "kp")
    min_pairs = _min_pairs_int(min_pairs)

    series = rr_interval_series(rpeaks, fs, rr_min_ms, rr_max_ms, breaks)
    rr, t_iv, valid = series["rr_ms"], series["t_s"], series["valid"]
    # A zero-length interval (a duplicated R-peak sample) passes the bounds
    # when rr_min_ms = 0; it would repeat an interval time (no strictly
    # increasing spline abscissa) and give an infinite heart rate. Exclude
    # it, counted as too short like any interval below the lower bound.
    zero_len = valid & ~(rr > 0.0)
    n_zero_len = int(zero_len.sum())
    if n_zero_len:
        valid = valid & ~zero_len
        series = dict(series, valid=valid, rri=rr[valid], rri_time=t_iv[valid],
                      n_excluded=int(series["n_excluded"]) + n_zero_len,
                      n_too_short=int(series["n_too_short"]) + n_zero_len)

    vi = np.flatnonzero(valid)
    tv, rv = t_iv[vi], rr[vi]
    pj = (np.flatnonzero(valid[:-1] & valid[1:]) if len(valid) > 1
          else np.array([], dtype=np.int64))
    px, py = rr[pj], rr[pj + 1]
    pt1, pt2 = t_iv[pj], t_iv[pj + 1]

    # Breaks that split an ANALYSED interval (rr_interval_series's rule:
    # strictly between two peaks). The caller may pass the whole
    # recording's breaks with a segment's peaks; a break outside the
    # segment, sitting on a peak or non-finite splits nothing here.
    n_breaks = 0
    if breaks is not None:
        b = np.asarray(breaks, dtype=np.float64).ravel()
        b = np.unique(b[np.isfinite(b)])
        pk = np.sort(np.asarray(rpeaks, dtype=int).ravel()).astype(np.float64)
        if len(pk) >= 2 and len(b):
            n_breaks = int(((b > pk[0]) & (b < pk[-1]) & ~np.isin(b, pk)).sum())
    params: dict = {
        "method":       method,
        "method_label": COV_METHODS[method],
        "method_note":  METHOD_NOTES[method],
        "window_s":     window_s,
        "ks":           ks,
        "kp":           kp,
        "fs_out":       fs_out,
        "min_pairs":    min_pairs,
        "fs":           float(fs),
        "n_rpeaks":     int(np.asarray(rpeaks).size),
        "n_breaks":     n_breaks,
        "trim_percent": TRIM_PERCENT if _TRIMMED_D[method] else 0.0,
        "n_intervals":  int(series["n_total"]),
        "n_valid":      int(len(vi)),
        "n_zero_length": n_zero_len,
        "n_pairs_total": int(len(pj)),
        "reference":    "Candia-Rivera et al., J Physiol 2025 (robust_hrv)",
        "calibration_note": calibration_note(ks, kp),
    }
    params.update(rr_exclusion_summary(series))

    # ── recording-level SD01 / SD02 / D0 ──
    if len(pj) >= min_pairs:
        sd01, sd02, d0 = window_poincare(px, py, method, ibi=rv)
    else:
        sd01 = sd02 = d0 = float("nan")

    # ── evaluated beats: t > t(1) + T, minus the last (nt = length(ixs) - 1) ──
    if len(tv):
        ev = np.flatnonzero(tv > tv[0] + window_s)[:-1]
    else:
        ev = np.array([], dtype=np.int64)
    t_end = tv[ev]
    lo_t = t_end - window_s
    w_start = np.searchsorted(tv, lo_t, side="left")
    w_stop = np.searchsorted(tv, t_end, side="right")
    w_count = w_stop - w_start
    p_start = np.searchsorted(pt1, lo_t, side="left")
    p_stop = np.searchsorted(pt2, t_end, side="right")
    n_pairs = np.maximum(p_stop - p_start, 0).astype(np.int64)

    # median time of each window's valid intervals (tv is sorted)
    if len(ev):
        t_beat = 0.5 * (tv[w_start + (w_count - 1) // 2] + tv[w_start + w_count // 2])
    else:
        t_beat = np.array([], dtype=np.float64)

    sd1_raw = np.full(len(ev), np.nan)
    sd2_raw = np.full(len(ev), np.nan)
    d_raw = np.full(len(ev), np.nan)
    ok = n_pairs >= min_pairs
    if ok.any():
        s1, s2, dd = _batch_window_poincare(px, py, p_start[ok], n_pairs[ok],
                                            rv, w_start[ok], w_count[ok], method)
        sd1_raw[ok], sd2_raw[ok], d_raw[ok] = s1, s2, dd

    # a withheld window is NaN in all three (window_poincare's rule)
    computed = np.isfinite(sd1_raw) & np.isfinite(sd2_raw) & np.isfinite(d_raw)
    n_withheld = int((~computed).sum())
    n_few = int((~ok).sum())
    n_undef = int((ok & ~computed).sum())
    n_computed = int(computed.sum())
    t_out = (_time_grid(float(t_beat[0]), float(t_beat[-1]), fs_out) if len(t_beat)
             else np.array([], dtype=np.float64))
    # Computed windows the resampling places nowhere: an isolated one (both
    # neighbours withheld, or the only window -- nothing to interpolate
    # from) or, at a coarse fs_out, a run lying between two output samples.
    # They are absent from every resampled series (still in the re-centring
    # mean and the beat-wise table), so they are disclosed.
    unplaced = [(s, e) for s, e, lo, hi
                in _grid_runs(t_beat, np.where(computed, 0.0, np.nan), t_out)
                if _unresampled(s, e, lo, hi)]
    n_isolated = int(sum(e - s for s, e in unplaced))
    isolated_why = ("no adjacent computed window to interpolate from"
                    + (f", or a run between two {fs_out:g} Hz output samples"
                       if any(e - s > 1 for s, e in unplaced) else ""))
    params["n_beats"] = int(len(ev))
    params["n_windows_withheld"] = n_withheld
    params["n_windows_few_pairs"] = n_few
    params["n_windows_cov_undefined"] = n_undef
    params["n_windows_isolated"] = n_isolated
    reason: "Optional[str]" = None
    undefined = (f"{COV_METHODS[method]} estimator: a Poincaré coordinate has "
                 f"zero RR variance (constant RR)")
    # With nothing excluded, a window ending on beat i holds >= min_pairs
    # pairs iff t_i - t_(i - min_pairs) <= T: its last min_pairs RR intervals
    # average at most T / min_pairs.
    few_cause = (f"excluded intervals or breaks, or slow beating: a "
                 f"{window_s:g} s window lacks {min_pairs} pairs wherever the "
                 f"last {min_pairs} RR intervals average more than "
                 f"{1000.0 * window_s / min_pairs:.0f} ms, i.e. HR below about "
                 f"{60.0 * min_pairs / window_s:.0f} bpm; a lower min_pairs (Min pairs "
                 f"in the Autonomic view; 2 = the reference code, which computes every "
                 f"window) or a longer window keeps them")
    parts = []
    if n_few:
        parts.append(f"{n_few} with fewer than {min_pairs} valid adjacent RR "
                     f"pairs ({few_cause})")
    if n_undef:
        parts.append(f"{n_undef} with an undefined covariance (the {undefined})")
    if len(ev) == 0:
        reason = (f"recording too short: no valid beat more than {window_s:g} s "
                  f"after the first valid interval")
    elif not ok.any():
        reason = (f"no {window_s:g} s window holds {min_pairs} valid adjacent "
                  f"RR pairs ({few_cause})")
    elif not computed.any():
        reason = f"SD1/SD2 undefined in every {window_s:g} s window with the {undefined}"
    elif not (math.isfinite(sd01) and math.isfinite(sd02)):
        reason = f"whole-recording SD01/SD02 undefined with the {undefined}"
    elif n_isolated == n_computed:
        reason = (f"no CSI/CPI sample: the {n_computed} computed window(s) of "
                  f"{len(ev)} cannot be resampled ({isolated_why})"
                  + (f"; the others are withheld -- {'; '.join(parts)}" if parts else ""))
    params["withheld_reason"] = reason
    note: "Optional[str]" = None
    if reason is None and (n_withheld or n_isolated):
        segs = []
        if n_withheld:
            segs.append(f"{n_withheld} of {len(ev)} windows "
                        f"({100.0 * n_withheld / len(ev):.0f} %) withheld -- "
                        + "; ".join(parts) + "; CSI/CPI are NaN there")
        if n_isolated:
            segs.append(f"{n_isolated} more computed window(s) cannot be resampled "
                        f"({isolated_why}) and are absent from CSI/CPI too")
        note = "; ".join(segs)
    params["withheld_note"] = note

    # ── re-centring on the whole-recording values ──
    sd1_c = sd1_raw - _nanmean(sd1_raw) + sd01
    sd2_c = sd2_raw - _nanmean(sd2_raw) + sd02
    d_c = d_raw - _nanmean(d_raw) + d0

    # ── resampling ──
    sd1 = _resample_runs(t_beat, sd1_c, t_out)
    sd2 = _resample_runs(t_beat, sd2_c, t_out)
    d = _resample_runs(t_beat, d_c, t_out)
    d_flip = _flipsig(d)
    rr_valid = np.where(valid, rr, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        hr_bpm = 60_000.0 / _resample_runs(t_iv, rr_valid, t_out)

    return AutonomicResult(
        t_beat=t_beat, t_end=t_end, sd1_raw=sd1_raw, sd2_raw=sd2_raw,
        d_raw=d_raw, n_pairs=n_pairs, sd01=float(sd01), sd02=float(sd02),
        d0=float(d0), t=t_out, d=d, d_flip=d_flip, sd1=sd1, sd2=sd2,
        hr_bpm=hr_bpm, params=params,
        **_compose(d, d_flip, sd1, sd2, ks, kp))


# ════════════════════════════════════════════════════════════
#  TABLES
# ════════════════════════════════════════════════════════════

def _attrs(res: AutonomicResult) -> dict:
    a = dict(res.params)
    a.update(sd01_ms=res.sd01, sd02_ms=res.sd02, d0_ms=res.d0)
    return a


def to_dataframe(res: AutonomicResult) -> "pd.DataFrame":
    """The resampled series, one row per output sample.

    Columns: Time_s, HR_bpm, CSI, CPI, CSI_HR (flipped D), CPI_HR (D, the
    re-centred CCD, ms), CSI_HRV (ks·SD2), CPI_HRV (kp·SD1), SD1_ms,
    SD2_ms. ``attrs`` holds ``res.params`` plus sd01_ms, sd02_ms, d0_ms.
    """
    df = pd.DataFrame({
        "Time_s":  res.t,
        "HR_bpm":  res.hr_bpm,
        "CSI":     res.csi,
        "CPI":     res.cpi,
        "CSI_HR":  res.d_flip,
        "CPI_HR":  res.d,
        "CSI_HRV": res.csi_hrv,
        "CPI_HRV": res.cpi_hrv,
        "SD1_ms":  res.sd1,
        "SD2_ms":  res.sd2,
    })
    df.attrs.update(_attrs(res))
    return df


def beat_dataframe(res: AutonomicResult) -> "pd.DataFrame":
    """The beat-wise window values before re-centring, one row per window.

    Columns: Time_s (window median time), Window_end_s, SD1_ms, SD2_ms,
    CCD_ms (D), n_pairs (valid adjacent pairs). A withheld window -- fewer
    than ``min_pairs`` pairs, or an undefined ``robust`` covariance -- is
    NaN in SD1_ms, SD2_ms and CCD_ms alike. ``attrs`` as in ``to_dataframe``.
    """
    df = pd.DataFrame({
        "Time_s":       res.t_beat,
        "Window_end_s": res.t_end,
        "SD1_ms":       res.sd1_raw,
        "SD2_ms":       res.sd2_raw,
        "CCD_ms":       res.d_raw,
        "n_pairs":      res.n_pairs,
    })
    df.attrs.update(_attrs(res))
    return df
