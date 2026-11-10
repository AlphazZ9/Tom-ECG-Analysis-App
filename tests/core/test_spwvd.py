"""
Time-resolved LF / HF by the smoothed pseudo Wigner-Ville distribution
(ecg.core.spwvd, a port of robust_hrv's compute_PWVD.m): band selectivity on
pure RR oscillations, time resolution, chunked vs single-pass agreement, an
independent literal (loop-based, MATLAB-indexed) port of the reference, the
gap handling, and parameter validation.
"""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.signal import detrend, hilbert

from ecg.core import analysis
from ecg.core.spwvd import (
    SPWVD_MAX_CHUNK_BYTES,
    SPWVD_MAX_FBINS,
    SPWVD_NAN_GAP_S,
    spwvd_lf_hf,
)

from tests.helpers import rpeaks_from_rr

FS = 2000.0          # ECG sampling rate of the synthetic R-peaks
FRR = 20.0           # SPWVD resampling rate
LF_BAND = (0.4, 1.5)
HF_BAND = (1.5, 5.0)

_trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid")


# ── Synthetic RR series ──────────────────────────────────────────────────

def _rr_list(duration_s: float, comps, seed: int = 0, noise_sd: float = 0.3,
             lf_on_s: float = 0.0) -> "list[float]":
    """RR intervals (ms) around 120 ms for *duration_s* seconds: each
    interval is 120 + sum(a * sin(2 pi f t)) at its own start time t, plus
    Gaussian noise. A component given as (a, f, "late") only starts at
    *lf_on_s*."""
    rng = np.random.default_rng(seed)
    rr, t = [], 0.0
    while t < duration_s:
        v = 120.0
        for comp in comps:
            a, f = comp[0], comp[1]
            if len(comp) > 2 and t < lf_on_s:
                continue
            v += a * np.sin(2 * np.pi * f * t)
        v += rng.normal(0.0, noise_sd)
        rr.append(v)
        t += v / 1000.0
    return rr


def _series(rr_list) -> dict:
    """rr_interval_series() of the R-peaks built from *rr_list* at FS."""
    return analysis.rr_interval_series(rpeaks_from_rr(rr_list, FS), FS)


def _spwvd(series: dict, **kw) -> dict:
    return spwvd_lf_hf(series["rr_ms"], series["t_s"], series["valid"], **kw)


def _between(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (t >= lo) & (t <= hi)


# ── (1) band selectivity on pure oscillations ────────────────────────────

@pytest.mark.parametrize("f0, lf_dominant", [(0.8, True), (2.5, False)])
def test_pure_rr_oscillation_lands_in_its_mouse_band(f0, lf_dominant):
    series = _series(_rr_list(60.0, [(5.0, f0)]))
    out = _spwvd(series)

    t = out["t"]
    mid = _between(t, t[0] + 10.0, t[-1] - 10.0)
    lf, hf = np.median(out["lf"][mid]), np.median(out["hf"][mid])
    assert np.isfinite(out["lf"]).all() and np.isfinite(out["hf"]).all()
    if lf_dominant:
        assert lf > 0 and lf > 20.0 * abs(hf)
    else:
        assert hf > 0 and hf > 20.0 * abs(lf)
    p = out["params"]
    assert p["lf_band"] == LF_BAND and p["hf_band"] == HF_BAND
    assert p["fbins"] == 4096 and p["fs_resample"] == FRR
    assert not p["chunked"] and p["n_chunks"] == 1


# ── (2) time resolution ──────────────────────────────────────────────────

def test_lf_switched_on_halfway_raises_lf_in_the_second_half_only():
    series = _series(_rr_list(120.0, [(3.0, 2.5), (5.0, 0.8, "late")], lf_on_s=60.0))
    out = _spwvd(series)

    t, lf, hf = out["t"], out["lf"], out["hf"]
    first, second = _between(t, 10.0, 50.0), _between(t, 70.0, 110.0)
    assert np.median(lf[second]) > 5.0 * np.median(lf[first])
    # the HF tone is unchanged across the switch (and LF/HF cross-terms
    # oscillate fast enough in time for the kernel to remove them)
    ratio = np.median(hf[second]) / np.median(hf[first])
    assert 0.7 < ratio < 1.4


# ── (3) chunked processing vs the single pass ────────────────────────────

def test_chunked_matches_single_pass_on_a_300_s_series():
    series = _series(_rr_list(300.0, [(3.0, 0.8), (4.0, 2.5)], seed=1, noise_sd=1.0))
    single = _spwvd(series, chunk_s=1000.0)          # 300 s <= chunk_s: one pass
    chunked = _spwvd(series, chunk_s=60.0, margin_s=30.0)   # 60 s cores + 30 s margins

    assert not single["params"]["chunked"]
    n = chunked["params"]["n_samples"]
    assert n > 300 * FRR
    assert chunked["params"]["chunked"]
    assert chunked["params"]["n_chunks"] == int(np.ceil(n / (60.0 * FRR))) >= 5
    np.testing.assert_array_equal(single["t"], chunked["t"])
    assert single["params"]["crossfade_s"] == 0.0
    assert chunked["params"]["crossfade_s"] == pytest.approx(30.0)
    t = single["t"]
    centre = _between(t, t[0] + 30.0, t[-1] - 30.0)
    for key in ("lf", "hf"):
        ref, got = single[key][centre], chunked[key][centre]
        rel = np.abs(got - ref) / np.abs(ref)
        assert np.median(rel) < 1e-3
        assert np.max(np.abs(got - ref)) / np.median(np.abs(ref)) < 1e-2


def _rr_step_list(duration_s: float, t_step: float, depth_ms: float,
                  drift_ms: float = 0.0, seed: int = 1) -> "list[float]":
    """Like _rr_list with 0.8 + 2.5 Hz tones, plus a sigmoid RR step of
    -depth_ms (2 s time constant) at *t_step* (a heart-rate change such
    as a drug onset) and a quadratic drift reaching +drift_ms at both ends
    of the recording: a non-stationary series."""
    rng = np.random.default_rng(seed)
    half = duration_s / 2.0
    rr, t = [], 0.0
    while t < duration_s:
        v = (120.0 + 3.0 * np.sin(2 * np.pi * 0.8 * t) + 4.0 * np.sin(2 * np.pi * 2.5 * t)
             - depth_ms / (1.0 + np.exp(-(t - t_step) / 2.0))
             + drift_ms * ((t - half) / half) ** 2 + rng.normal(0.0, 1.0))
        rr.append(v)
        t += v / 1000.0
    return rr


@pytest.mark.parametrize("depth_ms, drift_ms", [(30.0, 0.0), (0.0, 20.0)])
def test_chunked_follows_single_pass_on_a_trended_series_without_boundary_jumps(
        depth_ms, drift_ms):
    # Default chunking (120 s cores, 60 s margins) on 300 s: cores end at
    # ~100 s and ~200 s; the HR step sits on the first core boundary. The
    # detrend / Hilbert step must run on the whole series (as in
    # compute_PWVD.m): per-chunk detrending deviated by ~7 % here (step or
    # quadratic drift), with ~7 % jumps at the core boundaries.
    series = _series(_rr_step_list(300.0, t_step=100.0, depth_ms=depth_ms,
                                   drift_ms=drift_ms))
    single = _spwvd(series, chunk_s=1000.0)
    chunked = _spwvd(series)

    p = chunked["params"]
    assert p["chunked"] and p["n_chunks"] == 3 and p["n_excluded"] == 0
    n = p["n_samples"]
    t = single["t"] - single["t"][0]
    edges = np.round(np.linspace(0, n, p["n_chunks"] + 1)).astype(int)[1:-1]
    assert abs(t[edges[0]] - 100.0) < 1.0           # the step is on a core boundary
    centre = _between(t, 30.0, t[-1] - 30.0)
    pre, post = _between(t, 70.0, 99.99), _between(t, 100.0, 129.99)
    for key in ("lf", "hf"):
        ref, got = single[key], chunked[key]
        rel = np.abs(got - ref) / np.abs(ref)
        assert np.median(rel[centre]) < 1.5e-2      # measured ~2-4e-3 (step), ~1e-3 (drift)
        assert np.max(rel[centre]) < 6e-2           # measured ~1.4e-2 (step), ~9e-3 (drift)
        # 30 s post/pre ratio around the step (the paper's statistic)
        r_ref = ref[post].mean() / ref[pre].mean()
        r_got = got[post].mean() / got[pre].mean()
        assert abs(r_got / r_ref - 1.0) < 5e-3      # measured ~1e-3
        # no stitching jump: the largest sample-to-sample change around each
        # core boundary is no larger than the single pass's own
        step_ref = np.abs(np.diff(ref)) / np.abs(ref[1:])
        step_got = np.abs(np.diff(got)) / np.abs(got[1:])
        for e in edges:
            near = slice(e - 20 * 20, e + 20 * 20)  # ±20 s
            assert step_got[near].max() < 1.5 * step_ref[near].max()


def _rr_hrv_step_list(duration_s: float, t_step: float, seed: int = 1) -> "list[float]":
    """Mouse-like RR (100 ms; 3 Hz, 0.6 Hz and a 0.05 Hz VLF modulation)
    with, at *t_step*, a sigmoid RR shortening of 15 ms AND a halving of
    the HRV amplitude: a large HR / HRV change."""
    rng = np.random.default_rng(seed)
    rr, t = [], 0.0
    while t < duration_s:
        s = 1.0 / (1.0 + np.exp(-(t - t_step) / 2.0))
        g = 1.0 - 0.5 * s
        v = (100.0 - 15.0 * s
             + g * (3.0 * np.sin(2 * np.pi * 3.0 * t) + 3.0 * np.sin(2 * np.pi * 0.6 * t))
             + 5.0 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0.0, 0.5))
        rr.append(v)
        t += v / 1000.0
    return rr


def test_a_large_change_inside_the_chunk_margins_does_not_bias_the_core_boundaries():
    # Default chunking on 300 s: core boundaries at ~100 s and ~200 s, 60 s
    # margins. The HR + HRV change at 150 s lies in the margin of the chunks
    # on both sides, so its power leaks (circular Doppler FFT) into the
    # neighbouring cores and biases the 30 s ratios at the core BOUNDARIES,
    # not only at the change (2-9 % with 30 s margins on longer series).
    series = _series(_rr_hrv_step_list(300.0, t_step=150.0))
    single = _spwvd(series, chunk_s=1000.0)
    chunked = _spwvd(series)

    p = chunked["params"]
    assert p["chunked"] and p["n_chunks"] == 3 and p["margin_s"] == 60.0
    t = single["t"]
    edges = np.round(np.linspace(0, p["n_samples"], p["n_chunks"] + 1)).astype(int)[1:-1]
    onsets = [150.0] + [float(t[e]) for e in edges]
    assert all(abs(o - 150.0) <= 60.0 for o in onsets[1:])   # the change is in both margins
    centre = _between(t, t[0] + 30.0, t[-1] - 30.0)
    for key in ("lf", "hf"):
        ref, got = single[key], chunked[key]
        rel = np.abs(got - ref) / np.abs(ref)
        assert np.median(rel[centre]) < 1e-2        # measured ~2-4e-3
        assert np.max(rel[centre]) < 1.5e-1         # measured ~6e-2 (LF, low-power samples)
        for o in onsets:
            pre, post = (t >= o - 30.0) & (t < o), (t >= o) & (t <= o + 30.0)
            r_ref = ref[post].mean() / ref[pre].mean()
            r_got = got[post].mean() / got[pre].mean()
            assert abs(r_got / r_ref - 1.0) < 1e-2  # measured <= 0.35 %


def test_stitch_weights_sum_to_one_and_stay_inside_the_spans():
    from ecg.core.spwvd import _fade_samples, _stitch_weights
    n, core, margin = 5003, 1200, 600
    n_chunks = int(np.ceil(n / core))
    edges = np.round(np.linspace(0, n, n_chunks + 1)).astype(int)
    spans = [(max(0, c0 - margin), min(n, c1 + margin), c0, c1)
             for c0, c1 in zip(edges[:-1], edges[1:])]
    fade = _fade_samples(spans, margin)
    assert fade == margin // 2
    total = np.zeros(n)
    for k, (e0, e1, c0, c1) in enumerate(spans):
        w0, w1, w = _stitch_weights(k, len(spans), c0, c1, fade)
        assert e0 <= w0 and w1 <= e1 and len(w) == w1 - w0
        assert np.all((w > 0) & (w <= 1))
        total[w0:w1] += w
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-15)
    assert _fade_samples(spans[:1], margin) == 0
    assert _fade_samples(spans, 0) == 0             # margin_s = 0: hard stitching


# ── (4) independent literal port of compute_PWVD.m ───────────────────────

def _colon(a: float, b: float) -> np.ndarray:
    """MATLAB a:b (unit step)."""
    return a + np.arange(int(np.floor(b - a + 1e-9)) + 1)


def _compute_pwvd_literal(ibi, t_ibi, frr, fbins, lf_band, hf_band,
                          v0=0.03, tau0=0.06, lam=0.3):
    """compute_PWVD.m transcribed statement by statement, MATLAB 1-based
    indices kept (shifted by one only where Python indexes), the per-sample
    tfrwv loop, the meshgrid kernel with the general mu formula and the
    MATLAB fft/ifft/transpose sequence -- deliberately NOT vectorised."""
    t_temp = t_ibi[0] + np.arange(int(np.floor((t_ibi[-1] - t_ibi[0]) * frr + 1e-9)) + 1) / frr
    ibi_int = CubicSpline(t_ibi, ibi)(t_temp)
    ibi_int = detrend(ibi_int)

    ibi_x = hilbert(ibi_int - np.mean(ibi_int)).reshape(-1, 1)
    n_ibi = np.arange(1, ibi_x.shape[0] + 1).reshape(1, -1)
    xrow, xcol = ibi_x.shape
    _trow, tcol = n_ibi.shape

    wvx = np.zeros((fbins, tcol), dtype=complex)
    for icol in range(1, tcol + 1):
        ti = int(n_ibi[0, icol - 1])
        taumax = min(ti - 1, xrow - ti, round(fbins / 2) - 1)
        tau = np.arange(-taumax, taumax + 1)
        indices = np.remainder(fbins + tau, fbins) + 1
        wvx[indices - 1, icol - 1] = (ibi_x[ti + tau - 1, 0]
                                      * np.conj(ibi_x[ti - tau - 1, xcol - 1]))
        tau = round(fbins / 2)
        if ti <= (xrow - tau) and ti >= (tau + 1):
            wvx[tau + 1 - 1, icol - 1] = 0.5 * (
                ibi_x[ti + tau - 1, 0] * np.conj(ibi_x[ti - tau - 1, xcol - 1])
                + ibi_x[ti - tau - 1, 0] * np.conj(ibi_x[ti + tau - 1, xcol - 1]))

    wvx = np.fft.fft(wvx, axis=0)
    if xcol == 1:
        wvx = np.real(wvx)
    AF = np.fft.fft(np.fft.ifft(wvx, axis=0).T, axis=0)
    ncol = AF.shape[1]
    if ncol % 512 != 0 and ncol % 1024 != 0 and ncol % 4096 != 0 and ncol % 256 != 0:
        AF = AF.T

    Crow, Ccol = AF.shape
    dy = _colon(-Ccol / 2, Ccol / 2 - 1) / (Ccol / 2)
    if Crow / 2 - np.fix(Crow / 2) == 0:
        dx = _colon(-Crow / 2, Crow / 2 - 1) / (Crow / 2)
    else:
        dx = _colon(-Crow / 2 - 1 / 2, Crow / 2 - 3 / 2) / (Crow / 2)
    x, y = np.meshgrid(dy, dx)
    tau1 = x / tau0
    v1 = y / v0
    beta, gamma, alfa, r = 1, 1, 0, 0
    mu = (tau1 ** 2 * (v1 ** 2) ** alfa + (tau1 ** 2) ** alfa * v1 ** 2
          + 2 * r * ((tau1 * v1) ** beta) ** gamma)

    t_wind = np.exp(-np.pi * (mu ** 2) ** lam)
    t_wind = t_wind / np.max(t_wind)
    Asmooth = (AF * np.fft.fftshift(t_wind)).T
    TFR_RR = np.real(np.fft.ifft(np.fft.fft(Asmooth, axis=0).T, axis=0).T)

    f = np.linspace(0, frr / 2, fbins)
    LF_ix = np.flatnonzero((f >= lf_band[0]) & (f <= lf_band[1]))
    HF_ix = np.flatnonzero((f >= hf_band[0]) & (f <= hf_band[1]))
    LF = _trapz(TFR_RR[LF_ix, :], axis=0)
    HF = _trapz(TFR_RR[HF_ix, :], axis=0)
    return LF, HF, t_temp


@pytest.mark.parametrize("fbins, odd_length", [(256, False), (256, True), (1024, False)])
def test_vectorised_port_matches_a_literal_loop_port(fbins, odd_length):
    rr = np.array(_rr_list(30.0, [(3.0, 0.8), (4.0, 2.5)], seed=2, noise_sd=1.0))
    t = np.cumsum(rr) / 1000.0

    def n_samples(k):
        return int(np.floor((t[k - 1] - t[0]) * FRR + 1e-9)) + 1

    k = len(rr)          # drop trailing intervals until the grid parity matches
    while (n_samples(k) % 2 == 1) != odd_length:
        k -= 1
    rr, t = rr[:k], t[:k]

    out = spwvd_lf_hf(rr, t, fbins=fbins)
    lf_ref, hf_ref, t_ref = _compute_pwvd_literal(rr, t, FRR, fbins, LF_BAND, HF_BAND)

    assert (len(t_ref) % 2 == 1) == odd_length
    if fbins == 256:      # long enough for the symmetrised tau = Fbins/2 row to be set
        assert len(t_ref) >= fbins + 1
    else:                 # shorter than the lag range: taumax limited by the series
        assert len(t_ref) < fbins
    np.testing.assert_allclose(out["t"], t_ref, rtol=0, atol=1e-12)
    for got, ref in ((out["lf"], lf_ref), (out["hf"], hf_ref)):
        assert np.max(np.abs(got - ref)) <= 1e-8 * np.max(np.abs(ref))


# ── (5) parameter validation ─────────────────────────────────────────────

def _short_series(duration_s: float = 20.0) -> "tuple[np.ndarray, np.ndarray]":
    rr = np.array(_rr_list(duration_s, [(3.0, 0.8)]))
    return rr, np.cumsum(rr) / 1000.0


@pytest.mark.parametrize("fbins", [0, -256, 128, 255, 300, 768, 1000, 3000, 4097,
                                   2 ** 17, 2 ** 30, 2.5, float("nan"), "4096", None, True])
def test_invalid_fbins_raises(fbins):
    # 2 ** 30 must be refused up front: building its frequency axis alone
    # would need 8.6 GB before the memory guard runs.
    rr, t = _short_series()
    with pytest.raises(ValueError, match="fbins"):
        spwvd_lf_hf(rr, t, fbins=fbins)


@pytest.mark.parametrize("fbins, reason, not_reason", [
    (768, "only powers of two", "multiple of 256"),   # a multiple of 256: no transpose
    (3072, "only powers of two", "multiple of 256"),
    (300, "not a multiple of 256", "only powers of two"),
    (128, "not a multiple of 256", "only powers of two"),
    (0, "positive", "multiple of 256"),
    (2 ** 17, "gigabytes", "multiple of 256"),
])
def test_invalid_fbins_message_gives_the_right_reason(fbins, reason, not_reason):
    rr, t = _short_series()
    with pytest.raises(ValueError) as err:
        spwvd_lf_hf(rr, t, fbins=fbins)
    assert reason in str(err.value) and not_reason not in str(err.value)


def test_max_fbins_passes_the_check_and_meets_the_memory_guard():
    rr = np.array(_rr_list(120.0, [(3.0, 0.8)]))      # 2400 samples × 65536 bins
    t = np.cumsum(rr) / 1000.0
    with pytest.raises(ValueError, match="GB") as err:
        spwvd_lf_hf(rr, t, fbins=SPWVD_MAX_FBINS)
    assert "fbins must" not in str(err.value)


def test_integral_float_fbins_is_accepted():
    rr, t = _short_series()
    a = spwvd_lf_hf(rr, t, fbins=512)
    b = spwvd_lf_hf(rr, t, fbins=np.float64(512.0))
    np.testing.assert_array_equal(a["lf"], b["lf"])
    assert b["params"]["fbins"] == 512


@pytest.mark.parametrize("kw", [
    {"hf_band": (1.5, 12.0)},                 # above fs_resample / 2
    {"lf_band": (1.5, 0.4)},                  # reversed
    {"lf_band": (0.40, 0.401), "fbins": 256},  # fewer than two bins
    {"fs_resample": 0.0},
    {"v0": -0.03},
    {"chunk_s": 0.1},                         # 2 samples at 20 Hz
    {"margin_s": -1.0},
])
def test_invalid_parameters_raise(kw):
    rr, t = _short_series()
    with pytest.raises(ValueError):
        spwvd_lf_hf(rr, t, **kw)


def test_oversized_chunk_is_refused_before_allocating():
    rr = np.full(17_000, 120.0)                # ~2040 s at 20 Hz = 40 800 samples
    t = np.cumsum(rr) / 1000.0
    with pytest.raises(ValueError, match="GB"):
        spwvd_lf_hf(rr, t, chunk_s=3000.0)     # one 40 800 × 4096 pass: ~3.9 GB
    assert SPWVD_MAX_CHUNK_BYTES == pytest.approx(1.5e9)


def test_too_few_valid_intervals_raises():
    rr = np.array([120.0, 121.0, 500.0, 600.0, 119.0])   # 3 valid
    with pytest.raises(ValueError, match="valid"):
        spwvd_lf_hf(rr, np.cumsum(rr) / 1000.0)
    with pytest.raises(ValueError, match="lengths"):
        spwvd_lf_hf(rr, np.arange(3.0))


# ── Gaps, grid and reported parameters ───────────────────────────────────

def _gappy_series() -> "tuple[dict, float, float]":
    """~20 s of rhythm, a 400 ms pause, ~10 s, a 4 s run of ten 400 ms
    intervals (> RR_MAX_MS, excluded), ~20 s. Returns the series and the
    end times of the valid intervals bracketing the 4 s run."""
    a = _rr_list(20.0, [(3.0, 0.8), (3.0, 2.5)], seed=3)
    b = _rr_list(10.0, [(3.0, 0.8), (3.0, 2.5)], seed=4)
    c = _rr_list(20.0, [(3.0, 0.8), (3.0, 2.5)], seed=5)
    series = _series(a + [400.0] + b + [400.0] * 10 + c)
    i0 = len(a) + 1 + len(b)                   # first interval of the 4 s run
    t = series["t_s"]
    return series, float(t[i0 - 1]), float(t[i0 + 10])


def test_long_excluded_gap_is_nan_and_short_one_is_bridged():
    series, t_prev, t_next = _gappy_series()
    out = _spwvd(series)

    t, p = out["t"], out["params"]
    inside = (t > t_prev) & (t < t_next)
    assert inside.sum() > 70                   # ~4.1 s at 20 Hz
    np.testing.assert_array_equal(np.isnan(out["lf"]), inside)
    np.testing.assert_array_equal(np.isnan(out["hf"]), inside)
    assert p["n_excluded"] == 11 == series["n_excluded"]
    assert p["n_valid"] == series["n_total"] - 11
    assert p["longest_gap_s"] == pytest.approx(4.0, abs=1e-9)
    assert p["n_gaps_masked"] == 1 and p["n_nan_samples"] == int(inside.sum())
    assert p["nan_gap_s"] == SPWVD_NAN_GAP_S == 2.0


def test_each_stretch_between_long_gaps_is_its_own_series():
    # Nothing is smoothed or detrended across a long gap: the first stretch
    # gives exactly what the SPWVD of that stretch alone gives (the module
    # docstring's "Gaps" section; compute_PWVD.m itself bridges everything).
    rr = np.array(_rr_list(120.0, [(3.0, 0.8), (3.0, 2.5)], seed=2))
    t = np.cumsum(rr) / 1000.0
    valid = ~((t >= 60.0) & (t <= 63.5))                  # a 3.5 s excluded run
    full = spwvd_lf_hf(rr, t, valid)
    p = full["params"]
    assert p["n_segments"] == p["n_segments_computed"] == 2 and p["n_gaps_masked"] == 1
    first = t < 60.0
    alone = spwvd_lf_hf(rr[first], t[first])
    n1 = len(alone["t"])
    np.testing.assert_array_equal(full["t"][:n1], alone["t"])
    np.testing.assert_allclose(full["lf"][:n1], alone["lf"], rtol=1e-12)
    np.testing.assert_allclose(full["hf"][:n1], alone["hf"], rtol=1e-12)
    assert np.isnan(full["lf"][n1:n1 + 60]).all()          # the gap itself


def test_a_stretch_too_short_for_the_lf_floor_is_nan_and_counted():
    # Two long excluded runs leave a 1.2 s island: too short to hold two
    # periods of the 0.4 Hz LF floor or the ~12 s lag kernel, it would give
    # finite but meaningless values, so it is NaN (science review F7).
    rr = np.array(_rr_list(200.0, [(3.0, 0.8), (3.0, 2.5)], seed=6))
    t = np.cumsum(rr) / 1000.0
    valid = ~(((t > 97.0) & (t <= 100.0)) | ((t > 101.2) & (t <= 104.0)))
    out = spwvd_lf_hf(rr, t, valid)
    p = out["params"]
    assert p["n_segments"] == 3 and p["n_segments_computed"] == 2
    assert p["n_segments_too_short"] == 1 and p["min_segment_s"] == pytest.approx(10.0)
    island = (out["t"] > 100.0) & (out["t"] < 101.2)
    assert island.sum() > 15
    assert np.isnan(out["lf"][island]).all() and np.isnan(out["hf"][island]).all()
    assert np.isfinite(out["lf"][out["t"] < 96.0]).all()
    assert np.isfinite(out["lf"][out["t"] > 105.0]).all()
    # the rat LF floor (0.1 Hz) needs 20 s stretches
    rat = spwvd_lf_hf(rr, t, valid, lf_band=(0.1, 1.0), hf_band=(1.0, 3.5))
    assert rat["params"]["min_segment_s"] == pytest.approx(20.0)


def test_every_stretch_too_short_raises():
    rr = np.array(_rr_list(8.0, [(3.0, 2.5)], seed=1))     # 8 s < the 10 s minimum
    with pytest.raises(ValueError, match="too short"):
        spwvd_lf_hf(rr, np.cumsum(rr) / 1000.0)
    rr = np.array(_rr_list(40.0, [(3.0, 2.5)], seed=1))
    t = np.cumsum(rr) / 1000.0
    valid = ~(((t > 8.0) & (t <= 11.0)) | ((t > 19.0) & (t <= 22.0))
              | ((t > 30.0) & (t <= 33.0)))                 # stretches of ~8 s
    with pytest.raises(ValueError, match="Every stretch between RR gaps"):
        spwvd_lf_hf(rr, t, valid)


def test_bridged_short_gaps_above_the_limit_are_refused():
    # Short excluded runs (< 2 s each) are bridged by the spline; when they
    # cover more than 5 % of the computed time the SPWVD is refused.
    rr = np.array(_rr_list(60.0, [(3.0, 0.8), (3.0, 2.5)], seed=7))
    t = np.cumsum(rr) / 1000.0
    valid = np.ones(len(rr), dtype=bool)
    for start in np.arange(5.0, 55.0, 5.0):                 # 10 runs of ~0.5 s: ~8 %
        valid[(t > start) & (t <= start + 0.5)] = False
    with pytest.raises(ValueError, match="bridged by the spline"):
        spwvd_lf_hf(rr, t, valid)


def test_gaps_are_found_from_the_times_when_only_valid_intervals_are_passed():
    series, _, _ = _gappy_series()
    full = _spwvd(series)
    valid_only = spwvd_lf_hf(series["rri"], series["rri_time"])

    np.testing.assert_array_equal(full["t"], valid_only["t"])
    np.testing.assert_allclose(full["lf"], valid_only["lf"], rtol=1e-12, equal_nan=True)
    assert valid_only["params"]["n_excluded"] == 0
    assert valid_only["params"]["n_gaps_masked"] == 1
    assert valid_only["params"]["longest_gap_s"] == pytest.approx(4.0, abs=1e-9)


def test_output_grid_starts_at_the_first_valid_interval():
    rr = np.array(_rr_list(30.0, [(3.0, 0.8)]))
    t = np.cumsum(rr) / 1000.0
    valid = np.ones(len(rr), dtype=bool)
    valid[:5] = False                          # leading exclusions only trim the output
    order = np.random.default_rng(0).permutation(len(rr))   # unsorted input

    out = spwvd_lf_hf(rr[order], t[order], valid[order])
    assert out["t"][0] == t[5]
    np.testing.assert_allclose(np.diff(out["t"]), 1.0 / FRR, rtol=1e-9)
    assert out["t"][-1] <= t[-1] + 1e-9 and t[-1] - out["t"][-1] < 1.0 / FRR
    assert len(out["lf"]) == len(out["hf"]) == len(out["t"]) == out["params"]["n_samples"]
    assert np.isfinite(out["lf"]).all()
    p = out["params"]
    assert p["n_excluded"] == 5 and p["n_gaps_masked"] == 0
    assert LF_BAND[0] <= p["lf_band_bins"][0] < p["lf_band_bins"][1] <= LF_BAND[1]
    assert p["n_lf_bins"] > 2 and p["n_hf_bins"] > 2
