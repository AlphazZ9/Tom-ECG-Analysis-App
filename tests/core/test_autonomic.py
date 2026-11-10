"""
Tests for ecg.core.autonomic -- time-resolved CSI/CPI after Candia-Rivera et
al., J Physiol 2025 (reference MATLAB: github.com/diegocandiar/robust_hrv).

The parity tests compare against literal transliterations of the reference
MATLAB functions written below, independently of the module (plain loops,
np.cov, a trimmean written from MATLAB's documentation, scipy CubicSpline
with not-a-knot ends standing in for interp1 'spline').
"""
import inspect
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline
from scipy.stats import trim_mean

from ecg.core import autonomic as au
from ecg.core.analysis import rr_interval_series

from tests.helpers import rpeaks_from_rr

FS = 2000.0


# ════════════════════════════════════════════════════════════
#  Synthetic data
# ════════════════════════════════════════════════════════════

def _modulated_rr(duration_s: float, seed: int = 0) -> "list[float]":
    """~120 ms intervals with respiratory-like (2.5 Hz) and slow (0.8 Hz)
    modulation plus noise, generated beat by beat in real time; all well
    inside the mouse RR bounds."""
    rng = np.random.default_rng(seed)
    rr, t = [], 0.0
    while t < duration_s:
        v = (120.0 + 4.0 * math.sin(2 * math.pi * 2.5 * t)
             + 2.0 * math.sin(2 * math.pi * 0.8 * t) + rng.normal(0.0, 1.0))
        rr.append(v)
        t += v / 1000.0
    return rr


def _clean_peaks(duration_s: float = 60.0, seed: int = 0) -> np.ndarray:
    return rpeaks_from_rr(_modulated_rr(duration_s, seed), FS)


def _ibi_and_times(peaks: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """The IBI series (ms) and interval END times (s) the MATLAB code takes."""
    peaks = np.sort(peaks)
    return np.diff(peaks).astype(np.float64) / FS * 1000.0, peaks[1:] / FS


# ════════════════════════════════════════════════════════════
#  Literal MATLAB transliterations (reference implementations)
# ════════════════════════════════════════════════════════════

def _ref_trimmean(x, percent=5.0):
    """MATLAB trimmean(x, percent), default flag 'round', written from its
    documentation: k = n*percent/200 rounded to the nearest integer (to the
    smaller one if k is a half integer); k values removed from each end."""
    v = np.sort(np.asarray(x, dtype=float))
    n = len(v)
    k = n * percent / 200.0
    k0 = math.floor(k) + (1 if k - math.floor(k) > 0.5 else 0)
    return float(np.mean(v[k0:n - k0]))


def _ref_mmin(a, b):
    """MATLAB min(a, b) for scalars: a NaN operand is ignored."""
    return b if math.isnan(a) else min(a, b)


def _ref_mmax(a, b):
    return b if math.isnan(a) else max(a, b)


def _ref_make_mean_zero(x):
    n = x.shape[0]
    return x - np.ones((n, 1)) * x.mean(axis=0)


def _ref_make_std_one(x):
    n = x.shape[0]
    return x / (np.ones((n, 1)) * x.std(axis=0, ddof=1))


def _ref_varcov(x):
    n, p = x.shape
    xc = _ref_make_mean_zero(x)
    S = np.cov(xc.T)
    XC1 = np.tile(xc.T.reshape(p, 1, n), (1, p, 1))   # repmat(reshape(xc', [p 1 n]), [1 p 1])
    XC2 = np.tile(xc.T.reshape(1, p, n), (p, 1, 1))   # repmat(reshape(xc', [1 p n]), [p 1 1])
    VS = np.var(XC1 * XC2, axis=2, ddof=1) * n / ((n - 1) ** 2)
    return S, VS


def _ref_varshrink(x):
    v, vv = _ref_varcov(x)
    v, vv = np.diag(v), np.diag(vv)
    vtarget = np.median(v)
    numerator = np.sum(vv)
    denominator = np.sum((v - vtarget) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = float(numerator / denominator)
    lam = _ref_mmax(_ref_mmin(lam, 1.0), 0.0)
    return (1 - lam) * v + lam * vtarget, lam


def _ref_corshrink(x):
    p = x.shape[1]
    sx = _ref_make_std_one(_ref_make_mean_zero(x))
    r, vr = _ref_varcov(sx)
    offdiagsumrij2 = np.sum(np.tril(r, -1) ** 2)
    offdiagsumvrij = np.sum(np.tril(vr, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = float(offdiagsumvrij / offdiagsumrij2)
    lam = _ref_mmax(_ref_mmin(lam, 1.0), 0.0)
    rhat = (1 - lam) * r
    rhat[np.eye(p, dtype=bool)] = 1
    return rhat, lam


def _ref_robust_cov_hrv(x):
    """robustCovHRV.m: returns (s, lambda_var, lambda_cor)."""
    sv, lam_v = _ref_varshrink(x)
    dsv = np.diag(np.sqrt(sv))
    r, lam_c = _ref_corshrink(x)
    return dsv @ r @ dsv, lam_v, lam_c


def _ref_eig_sd(C):
    ev = np.linalg.eigvalsh(C)                        # MATLAB eig(symmetric): ascending
    return math.sqrt(ev[0]), math.sqrt(ev[1])


def _ref_window(ibi_win, method):
    """SD1, SD2, D of one IBI vector as the MATLAB methods compute them."""
    ibi1, ibi2 = ibi_win[:-1], ibi_win[1:]
    if method == "approximate":
        sd = np.diff(ibi_win)
        sd1 = math.sqrt(0.5 * np.std(sd, ddof=1) ** 2)
        sd2 = math.sqrt(2 * (np.std(ibi_win, ddof=1) ** 2) - (0.5 * np.std(sd, ddof=1) ** 2))
        return sd1, sd2, math.sqrt(np.mean(ibi1) ** 2 + np.mean(ibi2) ** 2)
    if method == "robust":
        C = _ref_robust_cov_hrv(np.column_stack([ibi1, ibi2]))[0]
    else:
        C = np.cov(np.column_stack([ibi1, ibi2]).T)
    sd1, sd2 = _ref_eig_sd(C)
    if method in ("rat", "robust"):
        d = math.sqrt(_ref_trimmean(ibi1, 5) ** 2 + _ref_trimmean(ibi2, 5) ** 2)
    else:
        d = math.sqrt(np.mean(ibi1) ** 2 + np.mean(ibi2) ** 2)
    return sd1, sd2, d


def _ref_colon(a, d, b):
    """MATLAB a : d : b."""
    n = int(math.floor((b - a) / d + 1e-9))
    return a + np.arange(n + 1) * d


def _ref_flipsig(sig):
    msig = np.mean(sig)
    sig1 = sig - msig
    sig1 = -sig1
    return sig1 + msig


def _ref_compute(ibi, t_ibi, wind, method, ks, kp, Fs):
    """compute_CSI_CVI_rat.m (method "rat", Fs = 20) and
    compute_rCSI_rCVI_type.m (methods exact / approximate / robust), line by
    line. Deviation: compute_rCSI_rCVI_type.m forms CVI = SD1*10 + 1 and
    CSI = SD2*1 + 1 (and hard-codes Fs = 4); here, as in the rat function,
    CVI = SD1*kp and CSI = SD2*ks with Fs passed in."""
    ibi = np.asarray(ibi, dtype=float).ravel()
    t_ibi = np.asarray(t_ibi, dtype=float).ravel()
    time = _ref_colon(t_ibi[0], 1 / Fs, t_ibi[-1])

    SD01, SD02, D0 = _ref_window(ibi, method)        # first Poincaré plot

    t1 = time[0]
    t2 = t1 + wind
    ixs = np.flatnonzero(t_ibi > t2)
    nt = len(ixs) - 1
    SD1, SD2, D, t_C = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for k in range(nt):
        i = ixs[k]
        t2 = t_ibi[i]
        t1 = t_ibi[i] - wind
        ix = np.flatnonzero((t_ibi >= t1) & (t_ibi <= t2))
        SD1[k], SD2[k], D[k] = _ref_window(ibi[ix], method)
        t_C[k] = np.median(t_ibi[ix])
    raw = {"SD1": SD1.copy(), "SD2": SD2.copy(), "D": D.copy(), "t_C": t_C.copy()}

    SD1 = SD1 - np.mean(SD1) + SD01
    SD2 = SD2 - np.mean(SD2) + SD02
    D = D - np.mean(D) + D0
    CVI = SD1 * kp
    CSI = SD2 * ks

    t_out = _ref_colon(t_C[0], 1 / Fs, t_C[-1])
    CVI_out = CubicSpline(t_C, CVI, bc_type="not-a-knot")(t_out)
    CSI_out = CubicSpline(t_C, CSI, bc_type="not-a-knot")(t_out)
    Dv_out = CubicSpline(t_C, D, bc_type="not-a-knot")(t_out)
    Ds_out = _ref_flipsig(Dv_out)
    return {"rCSI": Ds_out + CSI_out, "rCVI": Dv_out + CVI_out, "Ds": Ds_out,
            "Dv": Dv_out, "CSI": CSI_out, "CVI": CVI_out, "t_out": t_out,
            "SD01": SD01, "SD02": SD02, "D0": D0, **raw}


# ════════════════════════════════════════════════════════════
#  (1) Parity with the reference MATLAB code
# ════════════════════════════════════════════════════════════

def _assert_parity(res, ref, atol=1e-9):
    np.testing.assert_allclose(res.t_beat, ref["t_C"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.sd1_raw, ref["SD1"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.sd2_raw, ref["SD2"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.d_raw, ref["D"], rtol=0, atol=atol)
    assert res.sd01 == pytest.approx(ref["SD01"], abs=atol)
    assert res.sd02 == pytest.approx(ref["SD02"], abs=atol)
    assert res.d0 == pytest.approx(ref["D0"], abs=atol)
    assert len(res.t) == len(ref["t_out"])
    np.testing.assert_allclose(res.t, ref["t_out"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.csi, ref["rCSI"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.cpi, ref["rCVI"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.d_flip, ref["Ds"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.d, ref["Dv"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.csi_hrv, ref["CSI"], rtol=0, atol=atol)
    np.testing.assert_allclose(res.cpi_hrv, ref["CVI"], rtol=0, atol=atol)


def test_rat_reference_matches_compute_csi_cvi_rat():
    peaks = _clean_peaks(60.0, seed=1)
    ibi, t_ibi = _ibi_and_times(peaks)
    assert rr_interval_series(peaks, FS)["n_excluded"] == 0

    res = au.compute_autonomic_indices(peaks, FS, window_s=3.0, method="rat_reference",
                                       ks=3.0, kp=12.0, fs_out=20.0)
    ref = _ref_compute(ibi, t_ibi, 3.0, "rat", ks=3.0, kp=12.0, Fs=20.0)

    assert len(res.t_beat) == len(ref["t_C"]) > 400
    assert np.all(res.n_pairs >= au.DEFAULT_MIN_PAIRS)
    _assert_parity(res, ref)
    assert np.all(np.isfinite(res.csi)) and np.all(np.isfinite(res.cpi))


def test_exact_matches_compute_rcsi_rcvi_type_exact():
    # compute_rCSI_rCVI_type.m hard-codes Fs = 4 Hz -- pass the same here;
    # with ks = 1, kp = 10 the only remaining difference from that file is
    # its "+1" offsets on CSI_HRV / CVI_HRV, which the reference omits.
    peaks = _clean_peaks(60.0, seed=2)
    ibi, t_ibi = _ibi_and_times(peaks)
    res = au.compute_autonomic_indices(peaks, FS, method="exact", ks=1.0, kp=10.0,
                                       fs_out=4.0)
    ref = _ref_compute(ibi, t_ibi, 3.0, "exact", ks=1.0, kp=10.0, Fs=4.0)
    _assert_parity(res, ref)


@pytest.mark.parametrize("method, ref_method", [("robust", "robust"),
                                                ("approximate", "approximate")])
def test_other_methods_match_compute_rcsi_rcvi_type(method, ref_method):
    peaks = _clean_peaks(45.0, seed=3)
    ibi, t_ibi = _ibi_and_times(peaks)
    res = au.compute_autonomic_indices(peaks, FS, method=method)
    ref = _ref_compute(ibi, t_ibi, 3.0, ref_method, ks=au.KS_RAT_SHR,
                       kp=au.KP_RAT_SHR, Fs=20.0)
    _assert_parity(res, ref)


def _slow_peaks(duration_s: float, base_ms: float, seed: int) -> np.ndarray:
    """A slow heart (RR ~ base_ms, still inside the mouse RR bounds) with a
    0.8 Hz modulation and noise."""
    rng = np.random.default_rng(seed)
    rr, t = [], 0.0
    while t < duration_s:
        v = base_ms + 6.0 * math.sin(2 * math.pi * 0.8 * t) + rng.normal(0.0, 2.0)
        rr.append(v)
        t += v / 1000.0
    return rpeaks_from_rr(rr, FS)


def test_slow_heart_rate_thins_windows_with_a_note_and_min_pairs_2_restores_parity():
    # RR ~300 ms (~200 bpm), nothing excluded: a 3 s window holds 9-10 pairs,
    # so the default min_pairs = 10 withholds part of the windows the
    # reference code computes -- and says so.
    peaks = _slow_peaks(90.0, 300.0, seed=16)
    ibi, t_ibi = _ibi_and_times(peaks)
    assert rr_interval_series(peaks, FS)["n_excluded"] == 0
    res = au.compute_autonomic_indices(peaks, FS, method="rat_reference")
    p = res.params
    n_withheld = p["n_windows_withheld"]
    assert 0 < n_withheld < p["n_beats"]
    assert p["withheld_reason"] is None
    note = res.withheld_note
    assert note.startswith(f"{n_withheld} of {p['n_beats']} windows")
    assert "300 ms" in note and "200 bpm" in note

    # The documented rule: with nothing excluded a window lacks 10 pairs
    # exactly when its last 10 RR intervals sum to more than T = 3 s.
    i = np.searchsorted(t_ibi, res.t_end)
    last10 = np.array([ibi[k - 9:k + 1].sum() for k in i])
    clear = np.abs(last10 - 3000.0) > 1e-6                 # away from exact ties
    np.testing.assert_array_equal((res.n_pairs < 10)[clear], (last10 > 3000.0)[clear])
    np.testing.assert_array_equal(np.isnan(res.sd1_raw), res.n_pairs < 10)

    # min_pairs = 2 computes every window: identical to compute_CSI_CVI_rat.m.
    full = au.compute_autonomic_indices(peaks, FS, method="rat_reference", min_pairs=2)
    assert full.params["n_windows_withheld"] == 0 and full.withheld_note is None
    _assert_parity(full, _ref_compute(ibi, t_ibi, 3.0, "rat", ks=au.KS_RAT_SHR,
                                      kp=au.KP_RAT_SHR, Fs=20.0))
    # Where both computed, the default result's windows are the reference's.
    ok = res.n_pairs >= 10
    np.testing.assert_array_equal(res.sd1_raw[ok], full.sd1_raw[ok])
    np.testing.assert_array_equal(res.d_raw[ok], full.d_raw[ok])


def _runs(mask: np.ndarray) -> "list[tuple[int, int]]":
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def test_isolated_computed_windows_are_disclosed_not_silently_dropped():
    # Around 200 bpm the default min_pairs withholds about every other
    # window, leaving computed windows whose neighbours are both withheld:
    # nothing to interpolate from, so they are absent from the resampled
    # series -- and must be counted and named, not pass as resampled.
    peaks = _slow_peaks(300.0, 300.0, seed=3)
    res = au.compute_autonomic_indices(peaks, FS, method="rat_reference")
    p = res.params
    computed = np.isfinite(res.sd1_raw)
    lone = [s for s, e in _runs(computed) if e - s == 1]
    assert len(lone) > 0
    assert p["n_windows_isolated"] == len(lone)
    assert p["withheld_reason"] is None
    note = res.withheld_note
    assert note.startswith(f"{p['n_windows_withheld']} of {p['n_beats']} windows")
    assert (f"{len(lone)} more computed window(s) cannot be resampled "
            "(no adjacent computed window to interpolate from)") in note

    # An isolated window is in the beat-wise table but on no output sample
    # (the neighbouring stretches up to the withheld windows are NaN) ...
    half = 0.5 / au.DEFAULT_FS_OUT
    for i in lone:
        lo = res.t_beat[i - 1] if i > 0 else res.t_beat[i] - half
        hi = res.t_beat[i + 1] if i + 1 < len(res.t_beat) else res.t_beat[i] + half
        near = (res.t > lo) & (res.t < hi)
        assert near.any() and np.all(np.isnan(res.csi[near]))
    # ... while every run of >= 2 computed windows is resampled over its span.
    for s, e in _runs(computed):
        if e - s >= 2:
            span = (res.t >= res.t_beat[s]) & (res.t <= res.t_beat[e - 1])
            if span.any():
                assert np.all(np.isfinite(res.csi[span]))
    # The isolated windows still enter the re-centring mean (beat-wise
    # values of every computed window): CPI_HR = D re-centred on D0.
    d_c = res.d_raw - np.nanmean(res.d_raw) + res.d0
    fin = np.isfinite(res.d)
    assert fin.any()
    long_runs = [(s, e) for s, e in _runs(computed) if e - s >= 4]
    s, e = long_runs[0]
    ref = CubicSpline(res.t_beat[s:e], d_c[s:e], bc_type="not-a-knot")
    span = (res.t >= res.t_beat[s]) & (res.t <= res.t_beat[e - 1])
    np.testing.assert_allclose(res.d[span], ref(res.t[span]), rtol=1e-12)

    # With every window computed nothing is isolated.
    full = au.compute_autonomic_indices(peaks, FS, method="rat_reference", min_pairs=2)
    assert full.params["n_windows_isolated"] == 0 and full.withheld_note is None


def test_a_single_computed_window_is_explained_not_reported_as_ok():
    # 29 peaks 120 ms apart: one evaluated beat, its window computed with
    # min_pairs = 2, but a lone point cannot be resampled (MATLAB's interp1
    # needs two points as well): no CSI/CPI sample, and withheld_reason
    # says why instead of an all-NaN result with nothing to disclose.
    peaks = np.arange(29) * 240 + 1000
    res = au.compute_autonomic_indices(peaks, FS, method="exact", min_pairs=2)
    assert len(res.t_beat) == 1 and np.isfinite(res.sd1_raw[0])
    assert len(res.t) == 1 and np.all(np.isnan(res.csi)) and np.all(np.isnan(res.cpi))
    p = res.params
    assert p["n_windows_withheld"] == 0 and p["n_windows_isolated"] == 1
    assert p["withheld_reason"] == ("no CSI/CPI sample: the 1 computed window(s) of 1 "
                                    "cannot be resampled (no adjacent computed window "
                                    "to interpolate from)")
    assert res.withheld_note is None


def test_only_isolated_windows_computed_is_explained_with_the_withheld_cause():
    # Period-3 RR pattern (280, 310, 310 ms at 1000 Hz): the last 10 RR
    # intervals sum to 2980 ms (<= 3 s, 10 pairs) on one phase out of three
    # and to 3010 ms (9 pairs) on the other two, so every computed window
    # sits between two withheld ones.
    fs = 1000.0
    peaks = rpeaks_from_rr([280.0, 310.0, 310.0] * 40, fs)
    res = au.compute_autonomic_indices(peaks, fs, method="exact")
    computed = np.isfinite(res.sd1_raw)
    assert 0 < computed.sum() < len(computed)
    assert all(e - s == 1 for s, e in _runs(computed))
    p = res.params
    assert p["n_windows_isolated"] == int(computed.sum())
    assert np.isfinite(res.sd01) and np.all(np.isnan(res.csi))
    reason = res.withheld_reason
    assert reason.startswith(f"no CSI/CPI sample: the {int(computed.sum())} computed "
                             f"window(s) of {p['n_beats']} cannot be resampled")
    assert f"the others are withheld -- {p['n_windows_few_pairs']} with fewer than 10" in reason
    assert res.withheld_note is None


def test_window_bounds_are_inclusive_and_last_beat_is_dropped():
    # At 1024 Hz, alternating 120/136-sample intervals put every beat on an
    # exact binary fraction of a second and 24 intervals in exactly 3 s, so a
    # window ending on a beat starts exactly on another one: both ends are
    # included (25 intervals, 24 pairs; an open bound would give 23).
    fs = 1024.0
    rr = [(120.0 if k % 2 else 136.0) / fs * 1000.0 for k in range(80)]
    peaks = rpeaks_from_rr(rr, fs)
    t_ibi = peaks[1:] / fs
    res = au.compute_autonomic_indices(peaks, fs, method="exact")
    ixs = np.flatnonzero(t_ibi > t_ibi[0] + 3.0)
    assert len(res.t_beat) == len(ixs) - 1
    np.testing.assert_array_equal(res.t_end, t_ibi[ixs[:-1]])
    assert np.all(res.n_pairs == 24)
    np.testing.assert_array_equal(res.t_beat, res.t_end - 1.5)    # median of 25 times


# ════════════════════════════════════════════════════════════
#  MATLAB trimmean
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n, k0", [(20, 0), (24, 1), (30, 1), (40, 1), (60, 1),
                                   (61, 2), (100, 2), (9, 0), (10, 0)])
def test_matlab_trimmean_rounds_half_integers_down(n, k0):
    # k = n * 5 / 200: n = 20 -> 0.5 -> 0, n = 24 -> 0.6 -> 1, n = 60 -> 1.5
    # -> 1, n = 61 -> 1.525 -> 2, n = 100 -> 2.5 -> 2 (MATLAB doc, 'round').
    x = np.r_[np.arange(1.0, n), 1000.0]              # one high outlier
    expected = float(np.mean(np.sort(x)[k0:n - k0]))
    assert au.matlab_trimmean(x, 5) == pytest.approx(expected, rel=1e-15)
    assert au.matlab_trimmean(x, 5) == pytest.approx(_ref_trimmean(x, 5), rel=1e-15)


def test_matlab_trimmean_differs_from_scipy_and_drops_nan():
    x = np.r_[np.arange(1.0, 24.0), 1000.0]           # n = 24: MATLAB trims 1, scipy 0
    assert au.matlab_trimmean(x, 5) == pytest.approx(np.mean(np.arange(2.0, 24.0)))
    assert trim_mean(x, 0.025) == pytest.approx(np.mean(x))
    assert au.matlab_trimmean(np.r_[x, np.nan, np.nan], 5) == au.matlab_trimmean(x, 5)
    assert math.isnan(au.matlab_trimmean([], 5))
    assert math.isnan(au.matlab_trimmean([1.0, 2.0, 3.0, 4.0], 100))   # all trimmed
    with pytest.raises(ValueError):
        au.matlab_trimmean([1.0, 2.0], 120)


# ════════════════════════════════════════════════════════════
#  (2) Schafer-Strimmer shrinkage port
# ════════════════════════════════════════════════════════════

def test_shrinkage_hand_computed_example():
    # x (n = 5): columns (7, 14, 0, 17, 5) and (1, 2, 3, 0, 2).
    # Sample covariance: Sxx = 473/10, Syy = 13/10, Sxy = -119/20.
    # varcov: Var^(Sxx) = 269643/800, Var^(Syy) = 323/800 (var of the squared
    #   centred values, ddof 1, times n/(n-1)^2 = 5/16).
    # varshrink: target = median(Sxx, Syy) = 243/10,
    #   lambda_var = (269643 + 323)/800 / (2 * 23^2) = 134983/423200.
    # corshrink: Var^(r) / r^2 = Var^(xc*yc) / Sxy^2 = (3437/200) / (119/20)^2
    #   = 982/2023 (the standardisation cancels).
    x = np.array([[7, 1], [14, 2], [0, 3], [17, 0], [5, 2]], dtype=float)
    lam_v, lam_c = Fraction(134983, 423200), Fraction(982, 2023)
    sxx, syy, sxy = Fraction(473, 10), Fraction(13, 10), Fraction(-119, 20)
    target = (sxx + syy) / 2
    svx = (1 - lam_v) * sxx + lam_v * target          # 735337/18400
    svy = (1 - lam_v) * syy + lam_v * target          # 158903/18400
    off = float(1 - lam_c) * float(sxy) * math.sqrt(float(svx * svy) / float(sxx * syy))

    cov, lv, lc = au.shrinkage_cov_schafer_strimmer(x)
    assert lv == pytest.approx(float(lam_v), rel=1e-12)
    assert lc == pytest.approx(float(lam_c), rel=1e-12)
    assert 0.0 < lv < 1.0 and 0.0 < lc < 1.0
    expected = np.array([[float(svx), off], [off, float(svy)]])
    np.testing.assert_allclose(cov, expected, rtol=1e-12)
    np.testing.assert_allclose(cov, [[39.96396739130435, -7.253723561818102],
                                     [-7.253723561818102, 8.636032608695652]], rtol=1e-12)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n, p", [(5, 2), (12, 2), (40, 2), (8, 3), (30, 4)])
def test_shrinkage_properties_and_matlab_transliteration(seed, n, p):
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(p, p))
    x = rng.normal(size=(n, p)) @ mix * rng.uniform(0.5, 5.0, size=p) + 120.0
    cov, lv, lc = au.shrinkage_cov_schafer_strimmer(x)

    assert 0.0 <= lv <= 1.0 and 0.0 <= lc <= 1.0
    np.testing.assert_allclose(cov, cov.T, rtol=0, atol=1e-12 * np.abs(cov).max())
    assert np.linalg.eigvalsh(cov).min() >= -1e-10 * np.abs(cov).max()

    # Structure: variances shrunk toward their median by lambda_var,
    # correlations toward 0 by lambda_cor.
    s = np.cov(x, rowvar=False)
    v = np.diag(s)
    sv = (1 - lv) * v + lv * np.median(v)
    np.testing.assert_allclose(np.diag(cov), sv, rtol=1e-12)
    r = s / np.sqrt(np.outer(v, v))
    off = ~np.eye(p, dtype=bool)
    np.testing.assert_allclose(cov[off], ((1 - lc) * r * np.sqrt(np.outer(sv, sv)))[off],
                               rtol=1e-10)

    ref_cov, ref_lv, ref_lc = _ref_robust_cov_hrv(x)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-12)
    assert lv == pytest.approx(ref_lv, rel=1e-12)
    assert lc == pytest.approx(ref_lc, rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_shrinkage_keeps_the_trace_and_rounds_the_ellipse_for_two_coordinates(seed):
    # p = 2: the median of two variances is their mean, so variance shrinkage
    # keeps the trace (SD1^2 + SD2^2), and shrinking both the variance
    # difference and the correlation can only narrow the eigenvalue spread --
    # SD1 >= the sample-covariance SD1 and SD2 <= its SD2 in EVERY case.
    rng = np.random.default_rng(seed)
    for _ in range(200):
        n = int(rng.integers(3, 60))
        x = (rng.normal(size=(n, 2)) @ rng.normal(size=(2, 2))
             * rng.uniform(0.1, 10.0, size=2) + 120.0)
        cov = au.shrinkage_cov_schafer_strimmer(x)[0]
        s = np.cov(x, rowvar=False)
        ev, ev_s = np.linalg.eigvalsh(cov), np.linalg.eigvalsh(s)
        tol = 1e-12 * ev_s[1]
        assert np.trace(cov) == pytest.approx(np.trace(s), rel=1e-12)
        assert ev[0] >= ev_s[0] - tol and ev[1] <= ev_s[1] + tol

    # The same on every real window of a recording: robust vs exact.
    peaks = _clean_peaks(30.0, seed=20 + seed)
    rob = au.compute_autonomic_indices(peaks, FS, method="robust")
    ex = au.compute_autonomic_indices(peaks, FS, method="exact")
    assert np.all(np.isfinite(rob.sd1_raw)) and len(rob.sd1_raw) > 100
    assert np.all(rob.sd1_raw >= ex.sd1_raw * (1 - 1e-12))
    assert np.all(rob.sd2_raw <= ex.sd2_raw * (1 + 1e-12))
    np.testing.assert_allclose(rob.sd1_raw ** 2 + rob.sd2_raw ** 2,
                               ex.sd1_raw ** 2 + ex.sd2_raw ** 2, rtol=1e-11)
    assert np.mean(rob.sd1_raw > ex.sd1_raw * (1 + 1e-6)) > 0.5     # really rounds


def test_shrinkage_reduces_to_sample_covariance_when_lambdas_are_zero():
    # |centred values| constant in each column -> no variance of the squares
    # (lambda_var = 0); products of the standardised columns constant ->
    # lambda_cor = 0.
    x = np.array([[1.0, 10.0], [3.0, 16.0], [1.0, 10.0], [3.0, 16.0]])
    cov, lv, lc = au.shrinkage_cov_schafer_strimmer(x)
    assert lv == pytest.approx(0.0, abs=1e-12)
    assert lc == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(cov, np.cov(x, rowvar=False), rtol=1e-12)


def test_shrinkage_matlab_nan_semantics():
    # Equal variances: lambda_var = x/0 = Inf -> 1 (MATLAB min/max).
    x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    cov, lv, _ = au.shrinkage_cov_schafer_strimmer(x)
    assert lv == 1.0
    np.testing.assert_allclose(np.diag(cov), np.diag(np.cov(x, rowvar=False)))
    # Constant data: 0/0 -> NaN -> 1 for both; the correlation is undefined,
    # so the matrix is NaN (MATLAB's eig would error) -- the window is
    # withheld as a whole: D is NaN too, as for a window with too few pairs.
    const = np.full((6, 2), 120.0)
    cov, lv, lc = au.shrinkage_cov_schafer_strimmer(const)
    assert lv == 1.0 and lc == 1.0
    assert np.all(np.isnan(cov))
    sd1, sd2, d = au.window_poincare(const[:, 0], const[:, 1], "robust")
    assert math.isnan(sd1) and math.isnan(sd2) and math.isnan(d)
    sd1, sd2, d = au.window_poincare(const[:, 0], const[:, 1], "exact")
    assert sd1 == 0.0 and sd2 == 0.0 and d == pytest.approx(math.hypot(120, 120))
    # p = 1 and n < 2 edge cases
    one, lv, lc = au.shrinkage_cov_schafer_strimmer(np.array([1.0, 2.0, 4.0]))
    assert one.shape == (1, 1) and one[0, 0] == pytest.approx(np.var([1, 2, 4], ddof=1))
    assert math.isnan(lv) and math.isnan(lc)
    assert np.all(np.isnan(au.shrinkage_cov_schafer_strimmer(np.ones((1, 2)))[0]))


# ════════════════════════════════════════════════════════════
#  window_poincare
# ════════════════════════════════════════════════════════════

def test_window_poincare_methods_follow_their_definitions():
    rr = np.array(_modulated_rr(3.0, seed=4))
    x, y = rr[:-1], rr[1:]
    ev = np.linalg.eigvalsh(np.cov(x, y))
    mean_d = math.sqrt(np.mean(x) ** 2 + np.mean(y) ** 2)
    trim_d = math.sqrt(_ref_trimmean(x) ** 2 + _ref_trimmean(y) ** 2)

    sd1, sd2, d = au.window_poincare(x, y, "exact")
    assert (sd1, sd2) == pytest.approx((math.sqrt(ev[0]), math.sqrt(ev[1])), rel=1e-12)
    assert d == pytest.approx(mean_d, rel=1e-15)

    sd1, sd2, d = au.window_poincare(x, y, "rat_reference")
    assert (sd1, sd2) == pytest.approx((math.sqrt(ev[0]), math.sqrt(ev[1])), rel=1e-12)
    assert d == pytest.approx(trim_d, rel=1e-15)

    sd1, sd2, d = au.window_poincare(x, y, "approximate")
    var_d = np.var(y - x, ddof=1)
    assert sd1 == pytest.approx(math.sqrt(var_d / 2), rel=1e-12)
    assert sd2 == pytest.approx(math.sqrt(2 * np.var(rr, ddof=1) - var_d / 2), rel=1e-12)
    assert d == pytest.approx(mean_d, rel=1e-15)

    sd1, sd2, d = au.window_poincare(x, y, "robust")
    ev_r = np.linalg.eigvalsh(_ref_robust_cov_hrv(np.column_stack([x, y]))[0])
    assert (sd1, sd2) == pytest.approx((math.sqrt(ev_r[0]), math.sqrt(ev_r[1])), rel=1e-12)
    assert d == pytest.approx(trim_d, rel=1e-15)

    LedoitWolf = pytest.importorskip("sklearn.covariance").LedoitWolf
    ev_lw = np.linalg.eigvalsh(LedoitWolf().fit(np.column_stack([x, y])).covariance_)
    sd1, sd2, d = au.window_poincare(x, y, "ledoit_wolf")
    assert (sd1, sd2) == pytest.approx((math.sqrt(ev_lw[0]), math.sqrt(ev_lw[1])), rel=1e-12)
    assert d == pytest.approx(mean_d, rel=1e-15)

    assert all(math.isnan(v) for v in au.window_poincare([120.0], [121.0], "exact"))
    with pytest.raises(ValueError):
        au.window_poincare(x, y, "fmcd")
    with pytest.raises(ValueError):
        au.window_poincare(x, y[:-1], "exact")


# ════════════════════════════════════════════════════════════
#  (3) Gap handling
# ════════════════════════════════════════════════════════════

PAUSE_MS = 2500.0


def _gapped_recording(seed: int = 5):
    """~70 s with a 2.5 s sinus pause (> Max RR, excluded) at ~20 s and a
    real beat removed by artifact correction at ~50 s (its merged ~240 ms
    interval is inside the bounds but spans the break, so excluded too)."""
    rr = _modulated_rr(70.0, seed=seed)
    i_pause = int(np.searchsorted(np.cumsum(rr) / 1000.0, 20.0))
    rr = rr[:i_pause] + [PAUSE_MS] + rr[i_pause:]
    peaks = rpeaks_from_rr(rr, FS)
    i_removed = int(np.searchsorted(peaks, 50.0 * FS))
    removed = int(peaks[i_removed])
    peaks = np.delete(peaks, i_removed)
    return peaks, np.array([removed])


def _expected_windows(series, window_s, min_pairs, method):
    """Window-by-window expectation built from the definition: valid
    intervals in [t_i - T, t_i]; pairs (j, j + 1) with both intervals valid,
    in the window and ADJACENT in the original series."""
    rr, t, valid = series["rr_ms"], series["t_s"], series["valid"]
    vi = np.flatnonzero(valid)
    tv = t[vi]
    ev = vi[np.flatnonzero(tv > tv[0] + window_s)[:-1]]
    out = []
    for i in ev:
        in_win = valid & (t >= t[i] - window_s) & (t <= t[i])
        idx = np.flatnonzero(in_win)
        j = idx[:-1][np.diff(idx) == 1]               # adjacent valid pairs only
        n_pairs = len(j)
        vals = (au.window_poincare(rr[j], rr[j + 1], method, ibi=rr[idx])
                if n_pairs >= min_pairs else (np.nan, np.nan, np.nan))
        out.append({"t_end": t[i], "t_med": np.median(t[idx]), "n_pairs": n_pairs,
                    "n_valid": len(idx), "vals": vals})
    return out


@pytest.mark.parametrize("method", list(au.COV_METHODS))
def test_gaps_withhold_only_affected_windows_and_are_never_bridged(method):
    if method == "ledoit_wolf":
        pytest.importorskip("sklearn")
    peaks, breaks = _gapped_recording()
    series = rr_interval_series(peaks, FS, breaks=breaks)
    assert series["n_too_long"] == 1 and series["n_spanning_removed"] == 1
    rr, t, valid = series["rr_ms"], series["t_s"], series["valid"]
    i_pause = int(np.flatnonzero(rr > 2000.0)[0])
    i_break = int(np.flatnonzero(~valid & (rr < 2000.0))[0])

    res = au.compute_autonomic_indices(peaks, FS, method=method, breaks=breaks)
    exp = _expected_windows(series, 3.0, au.DEFAULT_MIN_PAIRS, method)

    assert len(res.t_beat) == len(exp)
    np.testing.assert_array_equal(res.t_end, [e["t_end"] for e in exp])
    np.testing.assert_allclose(res.t_beat, [e["t_med"] for e in exp], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(res.n_pairs, [e["n_pairs"] for e in exp])
    vals = np.array([e["vals"] for e in exp])
    np.testing.assert_allclose(res.sd1_raw, vals[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.sd2_raw, vals[:, 1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.d_raw, vals[:, 2], rtol=1e-12, atol=1e-12)

    # The removed-beat break splits the chain: a window holding it on both
    # sides loses 2 pairs, not 1 (no pair joins across the gap) and stays
    # computed; windows overlapping the long pause have too few pairs.
    t_end = res.t_end
    holds_break = (t_end - 3.0 <= t[i_break - 1]) & (t_end >= t[i_break + 1])
    assert holds_break.any()
    n_valid = np.array([e["n_valid"] for e in exp])
    assert np.all(res.n_pairs[holds_break] == n_valid[holds_break] - 2)
    assert np.all(np.isfinite(res.sd1_raw[holds_break]))

    nan_win = np.isnan(res.sd1_raw)
    assert nan_win.any()
    np.testing.assert_array_equal(nan_win, res.n_pairs < au.DEFAULT_MIN_PAIRS)
    pause_start, pause_end = t[i_pause - 1], t[i_pause]
    overlaps_pause = (t_end - 3.0 <= pause_end) & (t_end >= pause_start)
    assert np.all(overlaps_pause[nan_win])          # NaN only next to the pause
    assert res.params["n_windows_withheld"] == int(nan_win.sum())

    # Resampled series: NaN strictly inside the withheld stretch, finite
    # inside the computed runs.
    finite_t = res.t_beat[~nan_win]
    last_before = finite_t[finite_t < pause_end].max()
    first_after = finite_t[finite_t > pause_end].min()
    inside = (res.t > last_before + 1e-6) & (res.t < first_after - 1e-6)
    assert inside.any()
    for arr in (res.csi, res.cpi, res.d, res.d_flip, res.sd1, res.sd2):
        assert np.all(np.isnan(arr[inside]))
        assert np.all(np.isfinite(arr[res.t <= last_before]))
        assert np.all(np.isfinite(arr[res.t >= first_after]))

    # Heart rate: NaN across both gaps, finite elsewhere.
    hr_gap = (((res.t > t[i_pause - 1] + 1e-6) & (res.t < t[i_pause + 1] - 1e-6))
              | ((res.t > t[i_break - 1] + 1e-6) & (res.t < t[i_break + 1] - 1e-6)))
    assert hr_gap.any()
    assert np.all(np.isnan(res.hr_bpm[hr_gap]))
    assert np.all(np.isfinite(res.hr_bpm[~hr_gap]))
    assert np.nanmin(res.hr_bpm) > 400 and np.nanmax(res.hr_bpm) < 600

    assert res.params["n_excluded"] == 2
    assert res.params["n_spanning_removed"] == 1
    assert res.params["n_breaks"] == 1


_RESULT_ARRAYS = ("t_beat", "t_end", "n_pairs", "sd1_raw", "sd2_raw", "d_raw", "t", "d",
                  "d_flip", "sd1", "sd2", "cpi_hrv", "csi_hrv", "cpi", "csi", "hr_bpm")


@pytest.mark.parametrize("chunk_elems", [1, 7, 333])
@pytest.mark.parametrize("method", list(au.COV_METHODS))
def test_chunked_window_blocks_match_a_single_block(monkeypatch, method, chunk_elems):
    # A tiny element budget splits the batched window evaluation into many
    # (beats x pairs) blocks; chunk boundaries must not change any value.
    if method == "ledoit_wolf":
        pytest.importorskip("sklearn")
    peaks, breaks = _gapped_recording()
    whole = au.compute_autonomic_indices(peaks, FS, method=method, breaks=breaks)
    assert len(whole.t_beat) > chunk_elems                  # i.e. several chunks
    monkeypatch.setattr(au, "_CHUNK_ELEMS", chunk_elems)
    parts = au.compute_autonomic_indices(peaks, FS, method=method, breaks=breaks)
    for name in _RESULT_ARRAYS:
        np.testing.assert_array_equal(getattr(parts, name), getattr(whole, name),
                                      err_msg=name)        # NaN == NaN here
    assert (parts.sd01, parts.sd02, parts.d0) == (whole.sd01, whole.sd02, whole.d0)
    assert parts.params == whole.params


def test_robust_withholds_a_constant_rr_window_as_a_whole():
    # 59 identical 100 ms intervals at 1000 Hz: the windows inside that
    # stretch have a constant Poincaré coordinate, so the shrinkage
    # correlation is undefined. Such a window must be NaN in SD1, SD2 AND D,
    # like a window with too few pairs, so that D is re-centred, mirrored and
    # resampled on the very support of SD1/SD2 (no CSI_HR where CSI is NaN).
    fs = 1000.0
    rr = _modulated_rr(20.0, seed=14) + [100.0] * 59 + _modulated_rr(20.0, seed=15)
    peaks = rpeaks_from_rr(rr, fs)
    res = au.compute_autonomic_indices(peaks, fs, method="robust")
    p = res.params

    withheld = np.isnan(res.sd1_raw)
    undefined = withheld & (res.n_pairs >= au.DEFAULT_MIN_PAIRS)
    assert p["n_windows_cov_undefined"] == int(undefined.sum()) > 0
    np.testing.assert_array_equal(np.isnan(res.sd2_raw), withheld)
    np.testing.assert_array_equal(np.isnan(res.d_raw), withheld)
    assert p["n_windows_withheld"] == int(withheld.sum())
    assert p["withheld_reason"] is None
    assert f"{p['n_windows_cov_undefined']} with an undefined covariance" in res.withheld_note

    csi_ok = np.isfinite(res.csi)
    assert csi_ok.any() and not csi_ok.all()
    for arr in (res.cpi, res.d, res.d_flip, res.sd1, res.sd2, res.csi_hrv, res.cpi_hrv):
        np.testing.assert_array_equal(np.isfinite(arr), csi_ok)
    # The CSI_HR mirror point is the mean of D over the CSI support.
    np.testing.assert_allclose((res.d + res.d_flip)[csi_ok], 2.0 * np.mean(res.d[csi_ok]),
                               rtol=1e-13)

    df, bdf = au.to_dataframe(res), au.beat_dataframe(res)
    np.testing.assert_array_equal(df["CSI_HR"].isna(), df["CSI"].isna())
    np.testing.assert_array_equal(df["CPI_HR"].isna(), df["CPI"].isna())
    assert bdf.loc[withheld, ["SD1_ms", "SD2_ms", "CCD_ms"]].isna().all().all()

    # The sample covariance stays defined there; elsewhere the trimmed CCD
    # is the very same as rat_reference's.
    rat = au.compute_autonomic_indices(peaks, fs, method="rat_reference")
    assert rat.params["n_windows_cov_undefined"] == 0
    assert np.all(np.isfinite(rat.d_raw[undefined]))
    np.testing.assert_array_equal(res.d_raw[~withheld], rat.d_raw[~withheld])


def test_bridging_the_break_would_change_the_window_values():
    # Same windows computed as if the break interval had been dropped from
    # a clean series (MATLAB on a cleaned IBI vector pairs across it): the
    # gap-aware values differ, i.e. the module really refuses to bridge.
    peaks, breaks = _gapped_recording()
    series = rr_interval_series(peaks, FS, breaks=breaks)
    rr, t, valid = series["rr_ms"], series["t_s"], series["valid"]
    i_break = int(np.flatnonzero(~valid & (rr < 2000.0))[0])
    res = au.compute_autonomic_indices(peaks, FS, method="exact", breaks=breaks)
    k = int(np.flatnonzero((res.t_end - 3.0 <= t[i_break - 1])
                           & (res.t_end >= t[i_break + 1]))[0])
    idx = np.flatnonzero(valid & (t >= res.t_end[k] - 3.0) & (t <= res.t_end[k]))
    bridged = au.window_poincare(rr[idx[:-1]], rr[idx[1:]], "exact")
    assert res.n_pairs[k] == len(idx) - 2
    assert abs(res.sd1_raw[k] - bridged[0]) > 1e-6


def test_n_breaks_counts_only_breaks_that_split_an_analysed_interval():
    # The UI passes the WHOLE recording's removed-beat breaks with the
    # analysis window's peaks, and the exporter prints n_breaks as the
    # analysed RR's breaks: a break outside the peaks' span, on a peak or
    # non-finite splits nothing and must not be counted.
    full = rpeaks_from_rr(_modulated_rr(200.0, seed=21), FS)
    idx = [200, 500, 900, 1300]
    breaks = full[idx].astype(np.float64)
    peaks = np.delete(full, idx)
    t, tb = peaks / FS, breaks / FS
    seg_none = peaks[(t >= tb[1] + 5.0) & (t < tb[2] - 5.0)]     # holds no break
    seg_one = peaks[(t >= tb[1] - 20.0) & (t < tb[1] + 20.0)]    # holds breaks[1]
    assert not np.any((breaks > seg_none[0]) & (breaks < seg_none[-1]))

    res = au.compute_autonomic_indices(seg_none, FS, breaks=breaks)
    assert res.params["n_breaks"] == 0 and res.params["n_spanning_removed"] == 0
    plain = au.compute_autonomic_indices(seg_none, FS)
    for name in _RESULT_ARRAYS:
        np.testing.assert_array_equal(getattr(res, name), getattr(plain, name), err_msg=name)

    one = au.compute_autonomic_indices(seg_one, FS, breaks=breaks)
    assert one.params["n_breaks"] == 1 and one.params["n_spanning_removed"] == 1

    # Non-finite, out-of-range, on-a-peak and duplicated entries: only the
    # distinct break strictly inside an interval counts.
    odd = [float(seg_one[10]), np.nan, -50.0, 1e12, np.inf, breaks[1], breaks[1]]
    r = au.compute_autonomic_indices(seg_one, FS, breaks=odd)
    assert r.params["n_breaks"] == 1 and r.params["n_spanning_removed"] == 1
    on_peaks = [float(seg_one[0]), float(seg_one[10]), float(seg_one[-1])]
    r = au.compute_autonomic_indices(seg_one, FS, breaks=on_peaks)
    assert r.params["n_breaks"] == 0 and r.params["n_spanning_removed"] == 0
    np.testing.assert_array_equal(
        r.csi, au.compute_autonomic_indices(seg_one, FS).csi)
    for few in (seg_one[:1], np.array([], dtype=int)):
        assert au.compute_autonomic_indices(few, FS, breaks=breaks).params["n_breaks"] == 0


# ════════════════════════════════════════════════════════════
#  (4) Physiological behaviour
# ════════════════════════════════════════════════════════════

def _step_recording(t0: float = 45.0, total: float = 90.0, seed: int = 6) -> np.ndarray:
    """Before t0: RR 130 ms, large fast (2.5 Hz) and small slow (0.7 Hz)
    oscillations. After t0: RR 110 ms (HR up), the fast oscillation shrinks
    (vagal withdrawal) and the slow one grows (sympathetic activation)."""
    rng = np.random.default_rng(seed)
    rr, t = [], 0.0
    while t < total:
        after = t >= t0
        base, a_fast, a_slow = (110.0, 1.0, 4.0) if after else (130.0, 3.0, 1.0)
        v = (base + a_fast * math.sin(2 * math.pi * 2.5 * t)
             + a_slow * math.sin(2 * math.pi * 0.7 * t) + rng.normal(0.0, 0.3))
        rr.append(v)
        t += v / 1000.0
    return rpeaks_from_rr(rr, FS)


@pytest.mark.parametrize("method", ["robust", "rat_reference", "exact", "ledoit_wolf"])
def test_heart_rate_rise_and_slow_oscillation_raise_csi_and_lower_cpi(method):
    if method == "ledoit_wolf":
        pytest.importorskip("sklearn")
    t0 = 45.0
    peaks = _step_recording(t0)
    res = au.compute_autonomic_indices(peaks, FS, method=method)
    start = peaks[0] / FS
    before = (res.t >= start + t0 - 30.0) & (res.t < start + t0)
    after = (res.t > start + t0 + 3.0) & (res.t <= start + t0 + 30.0)
    assert before.sum() > 400 and after.sum() > 400

    def delta(a):
        return float(np.nanmean(a[after]) - np.nanmean(a[before]))

    assert delta(res.csi) > 0 and delta(res.cpi) < 0
    assert delta(res.csi_hr) > 0 and delta(res.cpi_hr) < 0      # shorter cycle
    assert delta(res.csi_hrv) > 0                               # slow HRV grows
    assert delta(res.cpi_hrv) < 0                               # fast HRV shrinks
    assert delta(res.hr_bpm) > 0


# ════════════════════════════════════════════════════════════
#  (5) Linearity in ks / kp
# ════════════════════════════════════════════════════════════

def test_indices_are_linear_in_ks_and_kp():
    peaks = _clean_peaks(40.0, seed=7)
    ra = au.compute_autonomic_indices(peaks, FS, ks=2.0, kp=7.0)
    rb = au.compute_autonomic_indices(peaks, FS, ks=5.0, kp=15.0)
    for name in ("t", "d", "d_flip", "sd1", "sd2", "hr_bpm", "sd1_raw", "sd2_raw", "d_raw"):
        np.testing.assert_array_equal(getattr(ra, name), getattr(rb, name))

    np.testing.assert_array_equal(ra.csi_hrv, 2.0 * ra.sd2)
    np.testing.assert_array_equal(ra.cpi_hrv, 7.0 * ra.sd1)
    np.testing.assert_allclose(ra.csi - rb.csi, (2.0 - 5.0) * ra.sd2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ra.cpi - rb.cpi, (7.0 - 15.0) * ra.sd1, rtol=0, atol=1e-10)

    # Per-event deltas: dCSI = -dD + ks*dSD2, dCPI = dD + kp*dSD1.
    pre, post = (ra.t >= 10) & (ra.t < 20), (ra.t >= 25) & (ra.t < 35)

    def dm(a):
        return float(np.mean(a[post]) - np.mean(a[pre]))

    assert dm(ra.csi) == pytest.approx(-dm(ra.d) + 2.0 * dm(ra.sd2), abs=1e-9)
    assert dm(ra.cpi) == pytest.approx(dm(ra.d) + 7.0 * dm(ra.sd1), abs=1e-9)

    # Recomposing without recomputation gives the same arrays.
    rc = au.with_coefficients(rb, 2.0, 7.0)
    for name in ("csi", "cpi", "csi_hrv", "cpi_hrv"):
        np.testing.assert_array_equal(getattr(rc, name), getattr(ra, name))
    assert rc.params["ks"] == 2.0 and rc.params["kp"] == 7.0
    assert rb.params["ks"] == 5.0                              # original untouched


# ════════════════════════════════════════════════════════════
#  API, defaults, disclosures, tables, edge cases
# ════════════════════════════════════════════════════════════

def test_methods_and_defaults():
    assert list(au.COV_METHODS) == ["robust", "rat_reference", "exact",
                                    "approximate", "ledoit_wolf"]
    assert au.COV_METHODS["robust"] == "Robust shrinkage (Schafer-Strimmer, paper text)"
    assert au.COV_METHODS["rat_reference"] == "Rat reference code (cov + 5 % trimmed mean)"
    assert au.COV_METHODS["exact"] == "Exact covariance"
    assert au.COV_METHODS["approximate"] == "Approximate (closed-form SD1/SD2)"
    assert au.COV_METHODS["ledoit_wolf"] == "Ledoit-Wolf (sklearn)"
    assert au.DEFAULT_METHOD == "robust"
    assert set(au.METHOD_NOTES) == set(au.COV_METHODS)
    assert (au.KS_RAT_SHR, au.KP_RAT_SHR) == (3.0, 12.0)
    sig = inspect.signature(au.compute_autonomic_indices).parameters
    assert sig["window_s"].default == au.DEFAULT_WINDOW_S == 3.0
    assert sig["fs_out"].default == au.DEFAULT_FS_OUT == 20.0
    assert sig["min_pairs"].default == au.DEFAULT_MIN_PAIRS == 10
    assert sig["method"].default == "robust"
    assert sig["ks"].default == 3.0 and sig["kp"].default == 12.0


def test_params_disclose_method_calibration_and_exclusions():
    peaks = _clean_peaks(20.0, seed=8)
    res = au.compute_autonomic_indices(peaks, FS)
    p = res.params
    assert p["method"] == "robust"
    assert p["method_label"] == au.COV_METHODS["robust"]
    assert "compute_CSI_CVI_rat.m" in p["method_note"]
    assert "not validated in mice" in p["calibration_note"].lower()
    for key in ("window_s", "ks", "kp", "fs_out", "min_pairs", "rr_min_ms", "rr_max_ms",
                "n_excluded", "n_beats", "n_pairs_total", "fs", "n_breaks"):
        assert key in p
    assert p["n_excluded"] == 0 and p["withheld_reason"] is None
    assert p["n_beats"] == len(res.t_beat)
    note = au.with_coefficients(res, 4.0, 9.0).params["calibration_note"]
    assert "user-set" in note and "not validated in mice" in note


def test_rr_bounds_and_robust_default_change_the_result():
    peaks = _clean_peaks(20.0, seed=9)
    robust = au.compute_autonomic_indices(peaks, FS)
    rat = au.compute_autonomic_indices(peaks, FS, method="rat_reference")
    np.testing.assert_allclose(robust.d_raw, rat.d_raw)          # same trimmed CCD
    assert not np.allclose(robust.sd1_raw, rat.sd1_raw)          # shrinkage differs
    narrow = au.compute_autonomic_indices(peaks, FS, rr_min_ms=119.0, rr_max_ms=121.0)
    assert narrow.params["n_excluded"] > 0
    assert narrow.params["rr_min_ms"] == 119.0 and narrow.params["rr_max_ms"] == 121.0


def test_tables():
    peaks = _clean_peaks(20.0, seed=10)
    res = au.compute_autonomic_indices(peaks, FS)
    df = au.to_dataframe(res)
    assert list(df.columns) == ["Time_s", "HR_bpm", "CSI", "CPI", "CSI_HR", "CPI_HR",
                                "CSI_HRV", "CPI_HRV", "SD1_ms", "SD2_ms"]
    assert len(df) == len(res.t)
    np.testing.assert_array_equal(df["CSI"].to_numpy(), res.csi)
    np.testing.assert_array_equal(df["CPI_HR"].to_numpy(), res.cpi_hr)
    np.testing.assert_array_equal(df["CSI_HR"].to_numpy(), res.csi_hr)
    assert df.attrs["method"] == "robust" and df.attrs["d0_ms"] == res.d0
    bdf = au.beat_dataframe(res)
    assert list(bdf.columns) == ["Time_s", "Window_end_s", "SD1_ms", "SD2_ms",
                                 "CCD_ms", "n_pairs"]
    assert len(bdf) == len(res.t_beat)
    np.testing.assert_array_equal(bdf["CCD_ms"].to_numpy(), res.d_raw)
    assert isinstance(bdf.attrs["calibration_note"], str)


def test_short_or_empty_recordings_give_an_explained_empty_result():
    short = rpeaks_from_rr([120.0] * 20, FS)               # 2.4 s < window
    for peaks in (short, np.array([1000]), np.array([], dtype=int)):
        res = au.compute_autonomic_indices(peaks, FS)
        assert len(res.t_beat) == 0 and len(res.t) == 0
        assert res.withheld_reason and "too short" in res.withheld_reason
        assert au.to_dataframe(res).empty and au.beat_dataframe(res).empty


def test_too_few_pairs_everywhere_is_explained():
    peaks = _clean_peaks(20.0, seed=11)
    res = au.compute_autonomic_indices(peaks, FS, min_pairs=60)    # > pairs per 3 s
    assert len(res.t_beat) > 0
    assert np.all(np.isnan(res.sd1_raw)) and np.all(np.isnan(res.csi))
    assert "60 valid adjacent" in res.withheld_reason


@pytest.mark.parametrize("kwargs", [{"method": "fmcd"}, {"min_pairs": 1},
                                    {"min_pairs": 2.5}, {"window_s": 0.0},
                                    {"fs_out": -1.0}, {"ks": float("nan")},
                                    {"kp": float("inf")}, {"rr_min_ms": 400.0}])
def test_invalid_arguments_are_rejected(kwargs):
    peaks = _clean_peaks(10.0, seed=12)
    with pytest.raises(ValueError):
        au.compute_autonomic_indices(peaks, FS, **kwargs)
    with pytest.raises(ValueError):
        au.compute_autonomic_indices(peaks, 0.0)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), None, "",
                                 "ten", True, 0, -3, 2.5, [10], 10 ** 400, 1e300])
def test_invalid_min_pairs_always_raises_value_error(bad):
    # An unparsed or empty UI field must give the module's ValueError, never
    # a TypeError / OverflowError from an int() conversion.
    peaks = _clean_peaks(10.0, seed=12)
    with pytest.raises(ValueError, match="min_pairs must be an integer") as info:
        au.compute_autonomic_indices(peaks, FS, min_pairs=bad)
    # no misleading chained TypeError / OverflowError in the traceback
    err = info.value
    assert err.__cause__ is None
    assert err.__context__ is None or err.__suppress_context__


def test_integral_min_pairs_of_any_numeric_type_is_accepted():
    peaks = _clean_peaks(10.0, seed=12)
    ref = au.compute_autonomic_indices(peaks, FS, min_pairs=10)
    for value in (10.0, np.int64(10), np.float64(10.0)):
        res = au.compute_autonomic_indices(peaks, FS, min_pairs=value)
        assert res.params["min_pairs"] == 10 and type(res.params["min_pairs"]) is int
        np.testing.assert_array_equal(res.csi, ref.csi)


def test_core_module_has_no_ui_dependency():
    src = inspect.getsource(au)
    assert "ecg.ui" not in src and "customtkinter" not in src and "tkinter" not in src
    assert isinstance(au.to_dataframe(au.compute_autonomic_indices(
        _clean_peaks(8.0, seed=13), FS)), pd.DataFrame)
