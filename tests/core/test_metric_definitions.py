"""
One definition per metric, and non-linear HRV on every beat of the window:

* mean HR is 60000 / mean NN everywhere (core, Compare Segments);
* Poincaré SD1/SD2 come from poincare_sd1_sd2 in the Non-linear panel and
  Compare Segments, on every valid pair of the window -- also when the
  entropy block is capped for runtime;
* the capped entropy block uses the contiguous block with the fewest
  excluded intervals, and says so (count + time span);
* the spectrum that produces VLF/LF/HF and LF/HF is the one stored for the
  PSD plot, and Compare Segments uses the same estimator;
* Compare Segments summarises each segment by non-overlapping block means
  (no beat-level test), with a descriptive-only note.
"""
import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.models import MouseECG

from tests.helpers import rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0


def _modulated_rr(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.12
    return (120.0 + 4.0 * np.sin(2 * np.pi * 2.5 * t)
            + 2.0 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0.0, 1.0, n))


def _signal_for(peaks: np.ndarray) -> np.ndarray:
    return np.zeros(int(peaks[-1]) + int(FS))


# ── mean HR ───────────────────────────────────────────────────────────────

def test_mean_hr_is_60000_over_mean_nn_not_mean_of_instantaneous_hr():
    rr = np.array([100.0, 150.0, np.nan, 0.0])
    assert analysis.mean_hr_bpm(rr) == pytest.approx(60_000.0 / 125.0)
    assert analysis.mean_hr_bpm(rr) != pytest.approx(np.mean(60_000.0 / rr[:2]))
    assert np.isnan(analysis.mean_hr_bpm(np.array([])))


@requires_nk
def test_core_and_segment_mean_hr_agree():
    peaks = rpeaks_from_rr(list(_modulated_rr(600)), FS)
    core = analysis.analyse_core(_signal_for(peaks), peaks, FS)
    seg = analysis.compute_segment_stats(_signal_for(peaks), peaks, FS,
                                         0.0, peaks[-1] / FS + 0.1)
    rr = np.diff(peaks) / FS * 1000.0
    assert core["hr"]["mean"] == pytest.approx(60_000.0 / rr.mean(), rel=1e-12)
    assert seg["hr_mean"] == pytest.approx(core["hr"]["mean"], rel=1e-12)


# ── Poincaré SD1 / SD2 ────────────────────────────────────────────────────

def test_poincare_sd1_sd2_is_gap_aware_classic_definition():
    rr = np.array([120.0, 122.0, 400.0, 118.0, 121.0, 119.0, 123.0])
    pc = analysis.poincare_sd1_sd2(rr)
    a, b = np.array([120.0, 118.0, 121.0, 119.0]), np.array([122.0, 121.0, 119.0, 123.0])
    assert pc["n_pairs"] == 4
    assert pc["sd1"] == pytest.approx(np.std((a - b) / np.sqrt(2), ddof=1))
    assert pc["sd2"] == pytest.approx(np.std((a + b) / np.sqrt(2), ddof=1))
    assert pc["sd1_sd2"] == pytest.approx(pc["sd1"] / pc["sd2"])
    assert np.isnan(analysis.poincare_sd1_sd2(np.array([120.0, 121.0]))["sd1"])


@requires_nk
def test_sd1_identical_in_nonlinear_panel_and_compare_segments():
    rr = _modulated_rr(1500)
    rr[700] = 400.0                                    # one excluded pause
    peaks = rpeaks_from_rr(list(rr), FS)
    series = analysis.rr_interval_series(peaks, FS)
    ref = analysis.poincare_sd1_sd2(series["rr_ms"], series["valid"])

    nl = analysis.analyse_hrv_nonlinear(peaks, FS, max_beats=300)   # entropy capped
    seg = analysis.compute_segment_stats(_signal_for(peaks), peaks, FS,
                                         0.0, peaks[-1] / FS + 0.1)

    assert nl.attrs["entropy_capped"] is True
    for value in (float(nl["HRV_SD1"].iloc[0]), seg["sd1"]):
        assert value == pytest.approx(ref["sd1"], rel=1e-12)
    for value in (float(nl["HRV_SD2"].iloc[0]), seg["sd2"]):
        assert value == pytest.approx(ref["sd2"], rel=1e-12)
    assert ref["n_pairs"] == len(rr) - 3   # every valid pair (2 lost to the pause), not 1000


# ── Non-linear HRV on every beat; entropy cap disclosed ──────────────────

def test_entropy_subset_picks_the_block_with_fewest_exclusions():
    valid = np.ones(30, dtype=bool)
    valid[[2, 5, 8]] = False                           # early exclusions
    assert analysis.entropy_subset(valid, 10) == (9, 19)
    assert analysis.entropy_subset(valid, 0) == (0, 30)
    assert analysis.entropy_subset(valid, 100) == (0, 30)


@requires_nk
def test_capped_nonlinear_keeps_poincare_and_dfa_on_all_beats():
    rr = _modulated_rr(1200, seed=4)
    peaks = rpeaks_from_rr(list(rr), FS)
    full = analysis.analyse_hrv_nonlinear(peaks, FS, max_beats=0)     # no cap
    capped = analysis.analyse_hrv_nonlinear(peaks, FS, max_beats=400)

    assert full.attrs["entropy_capped"] is False
    assert capped.attrs["entropy_capped"] is True
    assert capped.attrs["entropy_n_rr"] == 400
    assert capped.attrs["n_rr_valid"] == 1200
    for col in ("HRV_SD1", "HRV_SD2", "HRV_DFA_alpha1", "HRV_DFA_alpha2", "HRV_PI", "HRV_GI"):
        assert float(capped[col].iloc[0]) == pytest.approx(float(full[col].iloc[0]),
                                                           rel=1e-9), col
    # SampEn comes from the disclosed 400-interval block, not the whole window
    t0, t1 = capped.attrs["entropy_t_start_s"], capped.attrs["entropy_t_end_s"]
    sub = peaks[(peaks / FS >= t0 - 1e-9) & (peaks / FS <= t1 + 1e-9)]
    assert len(sub) == 401
    ref = analysis.nk.hrv_nonlinear(sub, sampling_rate=int(FS), show=False)
    assert float(capped["HRV_SampEn"].iloc[0]) == pytest.approx(float(ref["HRV_SampEn"].iloc[0]))
    assert "HRV_SampEn" in capped.attrs["entropy_columns"]
    assert "HRV_SD1" not in capped.attrs["entropy_columns"]
    text = analysis.nonlinear_subset_text(capped.attrs)
    assert "400 of 1,200 valid RR intervals" in text and f"{t0:.1f}–{t1:.1f} s" in text


@requires_nk
def test_default_cap_never_silently_drops_beats_for_poincare():
    n = MouseECG.ENTROPY_MAX_BEATS + 500
    peaks = rpeaks_from_rr(list(_modulated_rr(n, seed=2)), FS)
    nl = analysis.analyse_hrv_nonlinear(peaks, FS)
    assert nl.attrs["n_beats_used"] == n + 1 and nl.attrs["truncated"] is False
    assert nl.attrs["entropy_n_rr"] == MouseECG.ENTROPY_MAX_BEATS
    series = analysis.rr_interval_series(peaks, FS)
    assert float(nl["HRV_SD1"].iloc[0]) == pytest.approx(
        analysis.poincare_sd1_sd2(series["rr_ms"], series["valid"])["sd1"], rel=1e-12)


def test_old_truncated_results_are_still_disclosed():
    text = analysis.nonlinear_subset_text({"truncated": True, "n_beats_used": 1000,
                                           "n_beats_total": 4000})
    assert "first 1,000 of 4,000 beats" in text
    assert analysis.nonlinear_subset_text({"entropy_capped": False}) == ""


# ── One spectrum for the numbers and the plot ────────────────────────────

@requires_nk
def test_frequency_numbers_are_integrated_from_the_stored_spectrum():
    peaks = rpeaks_from_rr(list(_modulated_rr(1500)), FS)
    fd = analysis.analyse_hrv_freq(peaks, FS)
    freqs = np.asarray(fd.attrs["psd_freqs"])
    psd = np.asarray(fd.attrs["psd_power"])

    series = analysis.rr_interval_series(peaks, FS)
    spec = analysis.hrv_spectrum(series["rri"], series["rri_time"])
    np.testing.assert_allclose(psd, spec["psd"])
    for band, col in ((MouseECG.VLF, "HRV_VLF"), (MouseECG.LF, "HRV_LF"), (MouseECG.HF, "HRV_HF")):
        m = analysis.band_mask(freqs, band)
        assert float(fd[col].iloc[0]) == pytest.approx(float(analysis.trapz(psd[m], freqs[m])))
    lf, hf = float(fd["HRV_LF"].iloc[0]), float(fd["HRV_HF"].iloc[0])
    assert float(fd["HRV_LFHF"].iloc[0]) == pytest.approx(lf / hf)
    assert float(fd["HRV_LFn"].iloc[0]) == pytest.approx(lf / (lf + hf))
    assert float(fd["HRV_LFn"].iloc[0]) + float(fd["HRV_HFn"].iloc[0]) == pytest.approx(1.0)
    # the 2.5 Hz respiratory-like modulation is the HF peak; powers are ms²
    assert float(fd["HRV_HFpeak"].iloc[0]) == pytest.approx(2.5, abs=0.05)
    assert hf > 1.0     # 4 ms sine → ~8 ms², not a max-normalised fraction


@requires_nk
def test_compare_segments_uses_the_frequency_panel_spectrum():
    peaks = rpeaks_from_rr(list(_modulated_rr(1500)), FS)
    fd = analysis.analyse_hrv_freq(peaks, FS)
    seg = analysis.compute_segment_stats(_signal_for(peaks), peaks, FS,
                                         0.0, peaks[-1] / FS + 0.1)
    assert seg["lf_nu"] == pytest.approx(100.0 * float(fd["HRV_LFn"].iloc[0]), rel=1e-12)
    assert seg["lf_hf"] == pytest.approx(float(fd["HRV_LFHF"].iloc[0]), rel=1e-12)
    assert seg["hf"] == pytest.approx(float(fd["HRV_HF"].iloc[0]), rel=1e-12)


# ── Compare Segments: block means, descriptive only ──────────────────────

def test_segment_blocks_split_into_non_overlapping_blocks():
    t = np.arange(1, 301) * 0.1                        # 30 s of 100 ms intervals
    rr = np.full(300, 100.0)
    rr[150:] = 120.0
    blk = analysis.segment_blocks(t, rr, 0.0, 30.0, block_s=10.0)
    assert blk["n_blocks"] == 3 and blk["n_dropped"] == 0
    np.testing.assert_allclose(blk["t_start"], [0.0, 10.0, 20.0])
    assert blk["hr_mean"][0] == pytest.approx(600.0)
    # partial trailing block dropped; sparse block dropped (coverage < 50 %)
    assert analysis.segment_blocks(t, rr, 0.0, 25.0, 10.0)["n_blocks"] == 2
    sparse = analysis.segment_blocks(t[:20], rr[:20], 0.0, 20.0, 10.0)
    assert sparse["n_blocks"] == 0 and sparse["n_dropped"] == 2


def test_block_comparison_is_descriptive_and_needs_three_blocks():
    a = {"block_s": 10.0, "n_blocks": 4, "n_dropped": 0,
         "hr_mean": np.array([500.0, 502.0, 498.0, 501.0]),
         "rr_mean": np.array([120.0, 119.5, 120.5, 119.8])}
    b = {"block_s": 10.0, "n_blocks": 2, "n_dropped": 1,
         "hr_mean": np.array([540.0, 545.0]), "rr_mean": np.array([111.0, 110.0])}
    cmp = analysis.compare_segment_blocks(a, b)
    hr = cmp["metrics"]["hr_mean"]
    assert cmp["n_blocks_a"] == 4 and cmp["n_blocks_b"] == 2
    assert hr["diff"] == pytest.approx(542.5 - 500.25)
    assert hr["p_descriptive"] is None                 # < 3 blocks in B
    assert "descriptive" in cmp["note"] and "animal" in cmp["note"]
    text = analysis.segment_comparison_text(cmp, "Before", "After")
    assert "n = 4 blocks" in text and "too few blocks" in text


@requires_nk
def test_compare_segments_reports_blocks_not_beats():
    rr = np.r_[_modulated_rr(1000, seed=1), _modulated_rr(1000, seed=2) - 10.0]
    peaks = rpeaks_from_rr(list(rr), FS)
    sig = _signal_for(peaks)
    t_mid = peaks[1000] / FS
    sa = analysis.compute_segment_stats(sig, peaks, FS, 0.0, t_mid, "A")
    sb = analysis.compute_segment_stats(sig, peaks, FS, t_mid, peaks[-1] / FS, "B")
    cmp = analysis.compare_segments(sa, sb)
    n_a = int(np.floor(t_mid / MouseECG.SEGMENT_BLOCK_S))
    assert sa["n_blocks"] == cmp["n_blocks_a"] == n_a
    assert sa["n_blocks"] < sa["n_beats"] / 10        # blocks, not hundreds of beats
    assert sb["hr_block_mean"] > sa["hr_block_mean"]
    p = cmp["metrics"]["hr_mean"]["p_descriptive"]
    assert p is not None and 0.0 <= p <= 1.0
