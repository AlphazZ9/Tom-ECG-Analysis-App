import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.detection import correct_rr_artifacts
from ecg.core.models import MouseECG, QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.core.wave_template import WaveTemplate

from tests.helpers import add_qrs_pulse, rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0


def _modulated_rr(n: int, seed: int = 0) -> np.ndarray:
    """~120 ms intervals with respiratory-like (2.5 Hz) and baroreflex-like
    (0.8 Hz) modulation plus noise -- enough variability for non-trivial
    RMSSD / spectral / Poincaré values, all well inside the RR bounds."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.12
    return (120.0 + 4.0 * np.sin(2 * np.pi * 2.5 * t)
            + 2.0 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0.0, 1.0, n))


def _pause_peaks(pause_ms: float = 400.0, n_side: int = 200) -> np.ndarray:
    """R-peaks with one genuine sinus pause (> RR_MAX_MS) in the middle."""
    rr = _modulated_rr(2 * n_side)
    return rpeaks_from_rr(list(rr[:n_side]) + [pause_ms] + list(rr[n_side:]), FS)


def _signal_for(peaks: np.ndarray) -> np.ndarray:
    return np.zeros(int(peaks[-1]) + int(FS))


def test_rr_interval_series_splits_valid_and_excluded_intervals():
    peaks = np.array([460, 200, 320, 340, 980, 580])  # unsorted on purpose
    series = analysis.rr_interval_series(peaks, 1000.0)

    np.testing.assert_allclose(series["rr_ms"], [120, 20, 120, 120, 400])
    np.testing.assert_allclose(series["t_s"], [0.32, 0.34, 0.46, 0.58, 0.98])
    assert series["valid"].tolist() == [True, False, True, True, False]
    np.testing.assert_allclose(series["rri"], [120, 120, 120])
    np.testing.assert_allclose(series["rri_time"], [0.32, 0.46, 0.58])
    assert series["n_total"] == 5
    assert series["n_excluded"] == 2
    assert series["n_too_short"] == 1
    assert series["n_too_long"] == 1
    assert series["excluded_time_s"] == pytest.approx(0.42)
    assert analysis.nk_rri_input(series).keys() == {"RRI", "RRI_Time"}


def test_successive_valid_pairs_never_bridge_an_excluded_interval():
    rr = np.array([120.0, 122.0, 400.0, 118.0, 121.0, 20.0, 119.0])
    rr_a, rr_b = analysis.successive_valid_pairs(rr)

    np.testing.assert_allclose(rr_a, [120.0, 118.0])
    np.testing.assert_allclose(rr_b, [122.0, 121.0])


@pytest.mark.parametrize("method", ["core", "frequency", "nonlinear"])
@pytest.mark.parametrize("fs", [0.0, -1000.0, float("nan"), float("inf")])
@requires_nk
def test_hrv_methods_reject_invalid_sampling_rate(method, fs):
    peaks = np.array([200, 320, 440, 560, 680])

    with pytest.raises(ValueError, match="Sampling rate"):
        if method == "core":
            analysis.analyse_core(np.zeros(2000), peaks, fs)
        elif method == "frequency":
            analysis.analyse_hrv_freq(peaks, fs)
        else:
            analysis.analyse_hrv_nonlinear(peaks, fs)


@requires_nk
def test_core_excludes_a_sinus_pause_without_bridging_it():
    peaks = _pause_peaks()
    rr_all = np.diff(peaks) / FS * 1000.0
    valid = (rr_all >= MouseECG.RR_MIN_MS) & (rr_all <= MouseECG.RR_MAX_MS)
    assert int((~valid).sum()) == 1  # the 400 ms pause, nothing else

    result = analysis.analyse_core(_signal_for(peaks), peaks, FS)

    hr = result["hr"]
    assert hr["n_excluded"] == 1
    assert hr["n_too_long"] == 1
    assert hr["n_too_short"] == 0
    assert hr["excluded_time_s"] == pytest.approx(0.4)
    assert hr["excluded_pct"] == pytest.approx(100.0 / len(rr_all))
    assert hr["n_valid"] == len(rr_all) - 1
    # rr_ms keeps every interval (the tachogram still shows the pause);
    # rr_valid marks the ones used for HRV.
    np.testing.assert_allclose(result["rr_ms"], rr_all)
    assert result["rr_valid"].tolist() == valid.tolist()

    td = result["hrv_time"]
    assert td.attrs["n_excluded"] == 1
    assert float(td["HRV_MeanNN"].iloc[0]) == pytest.approx(rr_all[valid].mean())
    assert float(td["HRV_MaxNN"].iloc[0]) < MouseECG.RR_MAX_MS
    adjacent = valid[:-1] & valid[1:]
    succ = np.diff(rr_all)[adjacent]
    assert float(td["HRV_RMSSD"].iloc[0]) == pytest.approx(np.sqrt(np.mean(succ ** 2)), rel=1e-12)
    assert float(td["HRV_pNN6"].iloc[0]) == pytest.approx(
        np.mean(np.abs(succ) > MouseECG.PNN_THRESHOLD) * 100)


@requires_nk
def test_core_excludes_a_too_short_interval():
    peaks = np.array([200, 320, 340, 460, 581, 699, 820])  # one 20 ms doublet gap

    result = analysis.analyse_core(np.zeros(2000), peaks, 1000.0)

    assert result["hr"]["n_excluded"] == 1
    assert result["hr"]["n_too_short"] == 1
    assert float(result["hrv_time"]["HRV_MinNN"].iloc[0]) >= MouseECG.RR_MIN_MS


@requires_nk
def test_frequency_and_nonlinear_hrv_exclude_a_pause():
    peaks = _pause_peaks()

    freq = analysis.analyse_hrv_freq(peaks, FS)
    nonlin = analysis.analyse_hrv_nonlinear(peaks, FS)

    assert not freq.empty
    assert np.isfinite(float(freq["HRV_LFHF"].iloc[0]))
    assert freq.attrs["n_excluded"] == 1
    assert freq.attrs["n_too_long"] == 1
    assert not nonlin.empty
    assert np.isfinite(float(nonlin["HRV_SD1"].iloc[0]))
    assert nonlin.attrs["n_excluded"] == 1
    assert nonlin.attrs["n_too_long"] == 1


@requires_nk
def test_core_matches_neurokit_peaks_input_when_nothing_is_excluded():
    peaks = rpeaks_from_rr(list(_modulated_rr(300)), FS)

    result = analysis.analyse_core(_signal_for(peaks), peaks, FS)
    reference = analysis.nk.hrv_time(peaks, sampling_rate=int(FS), show=False)

    assert result["hr"]["n_excluded"] == 0
    for col in ("HRV_SDNN", "HRV_RMSSD", "HRV_MeanNN"):
        assert float(result["hrv_time"][col].iloc[0]) == float(reference[col].iloc[0])


@requires_nk
def test_core_rejects_fewer_than_three_valid_intervals():
    peaks = np.array([0, 500, 1000, 1120, 1240])  # 500, 500 (too long), 120, 120

    with pytest.raises(ValueError, match="Only 2 of 4 RR intervals"):
        analysis.analyse_core(np.zeros(2000), peaks, 1000.0)


@requires_nk
def test_pause_survives_artifact_correction_and_core_analysis():
    peaks = _pause_peaks()

    corrected, report = correct_rr_artifacts(peaks, FS)
    assert report["n_in"] - report["n_out"] == 0
    np.testing.assert_array_equal(corrected, peaks)

    result = analysis.analyse_core(_signal_for(corrected), corrected, FS)
    assert result["hr"]["n_excluded"] == 1
    assert np.isfinite(float(result["hrv_time"]["HRV_RMSSD"].iloc[0]))


def test_segment_hrv_differences_do_not_bridge_invalid_rr():
    rr_ms = np.array(
        [120, 120, 20, 180, 120, 120, 130, 120, 120, 120, 120, 120, 120],
        dtype=float,
    )
    peaks = np.r_[0, np.cumsum(rr_ms)].astype(int)
    signal = np.zeros(int(peaks[-1] + 1000))

    result = analysis.compute_segment_stats(
        signal, peaks, 1000.0, 0.0, peaks[-1] / 1000.0 + 0.5
    )

    valid_rr = (rr_ms >= MouseECG.RR_MIN_MS) & (rr_ms <= MouseECG.RR_MAX_MS)
    adjacent_valid = valid_rr[:-1] & valid_rr[1:]
    adjacent_diffs = np.abs(np.diff(rr_ms)[adjacent_valid])
    expected_rmssd = float(np.sqrt(np.mean(adjacent_diffs**2)))
    expected_pnn6 = float(np.mean(adjacent_diffs > MouseECG.PNN_THRESHOLD) * 100)

    assert result["rmssd"] == pytest.approx(expected_rmssd)
    assert result["pnn6"] == pytest.approx(expected_pnn6)


# ── Configurable RR bounds (anaesthetised mice) ──────────────────────────

def _slow_peaks(rr_mean: float = 353.0, n: int = 400, seed: int = 1) -> np.ndarray:
    """An anaesthetised-like recording below 180 bpm: every RR > RR_MAX_MS."""
    rng = np.random.default_rng(seed)
    return rpeaks_from_rr(list(rr_mean + rng.normal(0.0, 5.0, n)), FS)


@requires_nk
def test_core_failure_suggests_widening_max_rr():
    peaks = _slow_peaks()

    with pytest.raises(ValueError, match="Only 0 of 400 RR intervals") as err:
        analysis.analyse_core(_signal_for(peaks), peaks, FS)
    assert "widen Max RR" in str(err.value)
    assert "anaesthetised" in str(err.value)


@requires_nk
def test_widened_max_rr_analyses_an_anaesthetised_recording():
    peaks = _slow_peaks()
    rr_all = np.diff(peaks) / FS * 1000.0

    result = analysis.analyse_core(_signal_for(peaks), peaks, FS, rr_max_ms=500.0)

    hr = result["hr"]
    assert hr["n_excluded"] == 0
    assert hr["rr_min_ms"] == MouseECG.RR_MIN_MS
    assert hr["rr_max_ms"] == 500.0
    assert hr["mean"] == pytest.approx(60_000.0 / np.mean(rr_all))   # 60000 / mean NN
    assert result["hrv_time"].attrs["rr_max_ms"] == 500.0
    freq = analysis.analyse_hrv_freq(peaks, FS, rr_max_ms=500.0)
    nonlin = analysis.analyse_hrv_nonlinear(peaks, FS, rr_max_ms=500.0)
    assert not freq.empty and freq.attrs["rr_max_ms"] == 500.0
    assert not nonlin.empty and nonlin.attrs["n_excluded"] == 0


def test_bounds_are_recorded_and_printed_as_used():
    series = analysis.rr_interval_series(np.array([0, 120, 240, 690]), 1000.0,
                                         rr_min_ms=50.0, rr_max_ms=400.0)
    summary = analysis.rr_exclusion_summary(series)

    assert summary["rr_min_ms"] == 50.0 and summary["rr_max_ms"] == 400.0
    assert analysis.rr_bounds_label(summary) == "50–400 ms"
    assert analysis.rr_exclusion_text(summary) == "1 RR excluded (outside 50–400 ms)"
    assert analysis.rr_exclusion_text({"n_excluded": 0}) == ""
    # results restored from an older session carry no bounds: defaults
    assert analysis.rr_bounds_label({}) == (
        f"{MouseECG.RR_MIN_MS:.0f}–{MouseECG.RR_MAX_MS:.0f} ms")


def test_inconsistent_rr_bounds_are_rejected():
    with pytest.raises(ValueError, match="RR bounds"):
        analysis.rr_interval_series(np.array([0, 120, 240]), 1000.0,
                                    rr_min_ms=300.0, rr_max_ms=200.0)


def test_filter_params_round_trip_max_rr():
    from ecg.core.models import FilterParams

    d = FilterParams(max_rr_ms=500.0).to_dict()
    assert d["max_rr_ms"] == 500.0
    assert FilterParams.from_dict(d).max_rr_ms == 500.0
    # sessions saved before the field existed keep the awake-mouse default
    assert FilterParams.from_dict({}).max_rr_ms == MouseECG.RR_MAX_MS


# ── Removed beats are breaks (NN-interval rule) ──────────────────────────

def test_interval_spanning_a_removed_beat_is_excluded():
    peaks = np.array([0, 120, 240, 480, 600, 720, 1120])   # 240 spans removed 360
    series = analysis.rr_interval_series(peaks, 1000.0, breaks=[360, 1000])

    assert series["valid"].tolist() == [True, True, False, True, True, False]
    assert series["n_spanning_removed"] == 2   # the 400 ms one spans 1000 too:
    assert series["n_too_long"] == 0           # ... spanning takes precedence
    assert series["n_excluded"] == 2
    assert series["n_excluded"] == (series["n_too_short"] + series["n_too_long"]
                                    + series["n_spanning_removed"])


def test_break_on_a_current_peak_splits_nothing():
    peaks = np.array([0, 120, 240, 360])
    series = analysis.rr_interval_series(peaks, 1000.0, breaks=[240])
    assert series["n_excluded"] == 0


def _pvc_recording(n_side: int = 200, seed: int = 3) -> "tuple[np.ndarray, np.ndarray]":
    """(peaks with one premature beat, the same peaks without it)."""
    rng = np.random.default_rng(seed)
    rr = list(120 + rng.normal(0, 1, n_side)) + [80.0, 160.0] + list(120 + rng.normal(0, 1, n_side))
    peaks = rpeaks_from_rr(rr, FS)
    return peaks, np.delete(peaks, n_side + 1)


@requires_nk
def test_removed_premature_beat_is_a_break_not_a_long_normal_interval():
    peaks, _ = _pvc_recording()

    corrected, report = correct_rr_artifacts(peaks, FS)
    assert report["n_ectopic"] == 1
    assert report["removed_samples"] == [int(peaks[201])]
    # a real (premature) beat was removed: it is an HRV break
    assert report["break_samples"] == report["removed_samples"]

    result = analysis.analyse_core(_signal_for(corrected), corrected, FS,
                                   breaks=report["break_samples"])
    hr = result["hr"]
    assert hr["n_excluded"] == 1
    assert hr["n_spanning_removed"] == 1
    # the merged ~240 ms coupling + compensatory interval is NOT in HRV
    assert float(result["hrv_time"]["HRV_MaxNN"].iloc[0]) < 140.0
    rr_all = np.diff(corrected) / FS * 1000.0
    valid = result["rr_valid"]
    adjacent = valid[:-1] & valid[1:]
    expected = np.sqrt(np.mean(np.diff(rr_all)[adjacent] ** 2))
    assert float(result["hrv_time"]["HRV_RMSSD"].iloc[0]) == pytest.approx(expected, rel=1e-12)
    assert "spanning removed beats" in analysis.rr_exclusion_text(hr)

    # Without the break the merged interval inflates RMSSD several-fold.
    bridged = analysis.analyse_core(_signal_for(corrected), corrected, FS)
    assert float(bridged["hrv_time"]["HRV_RMSSD"].iloc[0]) > 3 * expected


@requires_nk
def test_removed_premature_couplet_leaves_no_non_sinus_interval():
    rng = np.random.default_rng(4)
    rr = (list(120 + rng.normal(0, 1, 100)) + [80.0, 80.0, 160.0]
          + list(120 + rng.normal(0, 1, 100)))
    peaks = rpeaks_from_rr(rr, FS)

    corrected, report = correct_rr_artifacts(peaks, FS)
    assert report["n_ectopic"] == 2
    assert report["break_samples"] == report["removed_samples"]

    result = analysis.analyse_core(_signal_for(corrected), corrected, FS,
                                   breaks=report["break_samples"])
    assert result["hr"]["n_spanning_removed"] == 1
    assert float(result["hrv_time"]["HRV_MaxNN"].iloc[0]) < 140.0


def _r_plus_t_wave_signal(n_beats: int = 200, t_lag_ms: float = 40.0):
    """(signal, R peaks, R + T peaks): every beat also detected on its T
    wave *t_lag_ms* after the R peak (the T detections are smaller)."""
    rng = np.random.default_rng(5)
    peaks = rpeaks_from_rr(list(120 + rng.normal(0, 1.5, n_beats - 1)), FS)
    sig = np.zeros(int(peaks[-1]) + int(FS))
    t_peaks = peaks + int(round(t_lag_ms / 1000.0 * FS))
    for p, t in zip(peaks, t_peaks):
        add_qrs_pulse(sig, int(p), FS, amp=1.0)
        add_qrs_pulse(sig, int(t), FS, amp=0.4)
    return sig, peaks, np.sort(np.r_[peaks, t_peaks])


@requires_nk
def test_removed_t_wave_double_detections_are_not_breaks():
    # Removing a spurious detection gives back the true sinus interval, so
    # T-wave double counting corrected on every beat must still analyse
    # (it used to exclude every interval as "spanning a removed beat").
    sig, peaks, doubled = _r_plus_t_wave_signal()
    corrected, report = correct_rr_artifacts(doubled, FS, signal=sig)

    assert np.array_equal(corrected, peaks)          # the R peaks are kept
    assert report["n_duplicate"] == len(peaks)
    assert len(report["removed_samples"]) == len(peaks)
    assert report["break_samples"] == []

    result = analysis.analyse_core(sig, corrected, FS, breaks=report["break_samples"] or None)
    assert result["hr"]["n_spanning_removed"] == 0
    assert result["hr"]["n_valid"] == len(peaks) - 1


@requires_nk
def test_failure_driven_by_breaks_points_at_the_artifact_correction():
    peaks = rpeaks_from_rr([120.0] * 6, FS)
    breaks = (peaks[:-1] + peaks[1:]) // 2          # every interval spans one
    with pytest.raises(ValueError) as err:
        analysis.analyse_core(_signal_for(peaks), peaks, FS, breaks=breaks)
    msg = str(err.value)
    assert "artifact correction" in msg and "Review Artifacts" in msg
    assert "widen Max RR" not in msg


# ── NeuroKit must never bridge a gap ─────────────────────────────────────

def _successive_manual(peaks: np.ndarray, fs: float) -> "tuple[np.ndarray, np.ndarray]":
    series = analysis.rr_interval_series(peaks, fs)
    return analysis.successive_valid_pairs(series["rr_ms"], series["valid"])


@requires_nk
def test_no_adjacent_valid_pair_gives_nan_not_bridged_successive_metrics():
    # Every other interval is a 20 ms doublet: no two kept intervals are
    # adjacent, so NeuroKit's gap detection gives up and differences across
    # the doublets. Nothing successive can be computed -- NaN, like pNN6.
    peaks = np.r_[0, np.cumsum([120, 20, 100, 20, 130, 20, 110, 20, 125])].astype(int) + 200

    td = analysis.analyse_core(np.zeros(3000), peaks, 1000.0)["hrv_time"]

    for col in ("HRV_RMSSD", "HRV_SDSD", "HRV_pNN50", "HRV_pNN20", "HRV_pNN6"):
        assert np.isnan(float(td[col].iloc[0])), col
    assert float(td["HRV_MeanNN"].iloc[0]) == pytest.approx(np.mean([120, 100, 130, 110, 125]))


@requires_nk
def test_uniform_timestamps_do_not_fool_gap_detection():
    # A spurious peak splits one 120 ms interval into 20 (excluded) + 100:
    # the kept intervals' end times stay perfectly uniform, which NeuroKit
    # reads as "interpolated, nothing missing" and differences 120 -> 100
    # straight across the doublet.
    peaks = rpeaks_from_rr([120.0] * 50 + [20.0, 100.0] + [120.0] * 50, 1000.0)
    rr_a, rr_b = _successive_manual(peaks, 1000.0)
    d = rr_b - rr_a

    td = analysis.analyse_core(np.zeros(int(peaks[-1]) + 1000), peaks, 1000.0)["hrv_time"]
    assert float(td["HRV_RMSSD"].iloc[0]) == pytest.approx(np.sqrt(np.mean(d ** 2)), rel=1e-12)
    assert float(td["HRV_SDSD"].iloc[0]) == pytest.approx(np.std(d, ddof=1), rel=1e-12)
    assert float(td["HRV_pNN20"].iloc[0]) == pytest.approx(
        np.sum(np.abs(d) > 20) / (len(d) + 1) * 100)

    nl = analysis.analyse_hrv_nonlinear(peaks, 1000.0)
    sd1 = np.std((rr_a - rr_b) / np.sqrt(2), ddof=1)
    sd2 = np.std((rr_a + rr_b) / np.sqrt(2), ddof=1)
    assert float(nl["HRV_SD1"].iloc[0]) == pytest.approx(sd1, rel=1e-12)
    assert float(nl["HRV_SD2"].iloc[0]) == pytest.approx(sd2, rel=1e-12)
    assert float(nl["HRV_SD1SD2"].iloc[0]) == pytest.approx(sd1 / sd2, rel=1e-12)
    # NeuroKit's own pairs were wrong, so its asymmetry indices are withheld
    assert "HRV_PI" in nl.attrs["pair_metrics_withheld"]
    assert np.isnan(float(nl["HRV_PI"].iloc[0]))


@requires_nk
def test_nonlinear_matches_neurokit_when_nothing_is_excluded_or_gap_is_seen():
    peaks = rpeaks_from_rr(list(_modulated_rr(300)), FS)
    reference = analysis.nk.hrv_nonlinear(peaks, sampling_rate=int(FS), show=False)

    nl = analysis.analyse_hrv_nonlinear(peaks, FS)
    for col in ("HRV_SD1", "HRV_SD2", "HRV_PI", "HRV_GI"):
        assert float(nl[col].iloc[0]) == float(reference[col].iloc[0])

    # A pause NeuroKit does detect: pair metrics are kept, not withheld.
    paused = analysis.analyse_hrv_nonlinear(_pause_peaks(), FS)
    assert paused.attrs["pair_metrics_withheld"] == []
    assert np.isfinite(float(paused["HRV_PI"].iloc[0]))


@requires_nk
def test_nonlinear_keeps_gap_aware_poincare_when_neurokit_fails_on_zero_variability():
    # Exactly constant RR around an excluded pause: every valid successive
    # difference is 0 and NeuroKit's HRA indices divide by zero.
    peaks = rpeaks_from_rr([120.0] * 200 + [400.0] + [120.0] * 200, FS)

    nl = analysis.analyse_hrv_nonlinear(peaks, FS)

    assert not nl.empty
    assert float(nl["HRV_SD1"].iloc[0]) == pytest.approx(0.0, abs=1e-9)
    assert nl.attrs["nk_failed_reason"]
    assert nl.attrs["n_excluded"] == 1


# ── Spectral HRV is withheld across long gaps ────────────────────────────

def test_spectral_withheld_reason_thresholds():
    rr = np.full(1000, 120.0)                       # 120 s analysed span
    ok = np.ones(1000, dtype=bool)
    assert analysis.spectral_withheld_reason(rr, ok) is None

    run = ok.copy(); run[500:518] = False           # 18 × 120 ms = 2.16 s run
    assert "run of excluded" in analysis.spectral_withheld_reason(rr, run)
    short_run = ok.copy(); short_run[500:516] = False   # 1.92 s run, 1.6 % time
    assert analysis.spectral_withheld_reason(rr, short_run) is None

    scattered = ok.copy(); scattered[::15] = False  # 6.7 % of the time, 0.12 s runs
    assert "cover" in analysis.spectral_withheld_reason(rr, scattered)


@requires_nk
def test_frequency_hrv_withheld_across_a_long_excluded_stretch():
    rr = _modulated_rr(2500)
    peaks = rpeaks_from_rr(list(rr[:1000]) + [360.0] * 70 + list(rr[1000:]), FS)

    freq = analysis.analyse_hrv_freq(peaks, FS)

    assert freq.empty
    assert "run of excluded RR intervals" in freq.attrs["withheld_reason"]
    assert freq.attrs["n_too_long"] == 70


@requires_nk
def test_frequency_hrv_withheld_when_excluded_time_is_too_large():
    rng = np.random.default_rng(5)
    rr: "list[float]" = []
    for _ in range(60):   # a 400 ms pause every 15 beats: 19 % of the time
        rr += list(120 + rng.normal(0, 1, 14)) + [400.0]
    freq = analysis.analyse_hrv_freq(rpeaks_from_rr(rr, FS), FS)

    assert freq.empty
    assert "cover" in freq.attrs["withheld_reason"]


@requires_nk
def test_too_few_valid_intervals_also_explains_the_empty_spectrum():
    freq = analysis.analyse_hrv_freq(rpeaks_from_rr(list(_modulated_rr(30)), FS), FS)
    assert freq.empty
    assert "only 30 valid RR intervals" in freq.attrs["withheld_reason"]


def test_segment_lf_hf_withheld_across_a_long_excluded_stretch():
    rr = _modulated_rr(600)
    rr_ms = np.r_[rr[:300], [360.0] * 20, rr[300:]]
    peaks = rpeaks_from_rr(list(rr_ms), FS)
    signal = np.zeros(int(peaks[-1]) + 2000)

    seg = analysis.compute_segment_stats(signal, peaks, FS, 0.0, peaks[-1] / FS + 0.1)

    assert seg["freq_withheld_reason"] is not None
    assert np.isnan(seg["lf_hf"]) and np.isnan(seg["lf_nu"])
    assert seg["n_too_long"] == 20
    assert np.isfinite(seg["rmssd"])


# ── Beat window sized from valid intervals ───────────────────────────────

def test_one_spurious_close_peak_does_not_shrink_the_beat_window():
    # A spurious peak (e.g. on a T wave) 60 ms after beat 150 splits one RR
    # into 60 + 60 ms, both below RR_MIN_MS and so excluded from HRV.
    peaks = rpeaks_from_rr([120.0] * 300, FS)
    spurious = np.sort(np.r_[peaks, peaks[150] + int(0.060 * FS)])
    signal = np.random.default_rng(0).normal(0, 0.01, int(peaks[-1]) + 4000)

    clean = analysis.compute_beat_correlation(signal, peaks, FS)
    noisy = analysis.compute_beat_correlation(signal, spurious, FS)

    np.testing.assert_allclose(noisy["beat_time"], clean["beat_time"])
    assert analysis._beat_window_min_rr_samples(spurious, FS) == int(0.120 * FS)


def _record_windows(monkeypatch):
    """Patch detect_waves_on_beat to record each beat's (start, end) ms."""
    windows: "list[tuple[float, float]]" = []

    def _fake_detect(beat, bt_ms, template, **kw):
        windows.append((float(bt_ms[0]), float(bt_ms[-1])))
        return {k: np.nan for k in ("P_onset", "P_peak", "Q_peak", "S_peak",
                                    "J_peak", "T_peak", "T_offset")}

    monkeypatch.setattr(analysis, "detect_waves_on_beat", _fake_detect)
    return windows


def test_analyse_intervals_beat_window_ignores_a_spurious_close_peak(monkeypatch):
    # A spurious peak 60 ms after beat 30 (below RR_MIN_MS) is not a heart
    # beat: it neither shortens beat 30's window nor any other beat's.
    peaks = rpeaks_from_rr([120.0] * 60, FS)
    spurious = np.sort(np.r_[peaks, peaks[30] + int(0.060 * FS)])
    signal = np.zeros(int(peaks[-1]) + 4000)
    windows = _record_windows(monkeypatch)
    wt = WaveTemplate()
    for rp in (peaks, spurious):
        windows.clear()
        analysis.analyse_intervals(signal, rp, FS, np.diff(rp) / FS * 1000.0,
                                   wave_template=wt)
        # 70 % of the 120 ms RR on each side (the first beat has no
        # preceding RR: full ±100 ms before R).
        assert windows[0] == pytest.approx((-100.0, 83.5))
        assert windows[30] == pytest.approx((-84.0, 83.5))


def test_one_premature_beat_only_shortens_its_neighbours_windows(monkeypatch):
    # The window used to be sized from the recording's SHORTEST RR, so one
    # premature beat (85 ms) shortened every beat's window.
    rr = [180.0] * 60
    rr[30], rr[31] = 85.0, 275.0
    peaks = rpeaks_from_rr(rr, FS)
    signal = np.zeros(int(peaks[-1]) + 4000)
    windows = _record_windows(monkeypatch)
    df = analysis.analyse_intervals(signal, peaks, FS, np.asarray(rr),
                                    wave_template=WaveTemplate())
    assert windows[10] == pytest.approx((-100.0, 99.5))   # 0.7 x 180 > cap
    assert windows[30] == pytest.approx((-100.0, 59.0))   # before the premature beat
    assert windows[31] == pytest.approx((-59.5, 99.5))    # the premature beat
    shortened = [i for i, w in enumerate(windows[1:-1], 1) if w != (-100.0, 99.5)]
    assert shortened == [30, 31]
    info = df.attrs["beat_window"]
    assert (info["n_short_pre"], info["n_short_post"]) == (1, 1)
    assert info["min_post_ms"] == pytest.approx(59.5)
    text = analysis.beat_window_text(info)
    assert "70 % of the preceding RR" in text and "shortened on 1 / 1 of 61" in text


def test_analyse_intervals_beat_window_uses_the_rr_min_it_is_given(monkeypatch):
    # A 45 ms split after beat 15 is ignored under the default Min RR
    # (67 ms) but is a neighbour when Min RR is lowered below it.
    peaks = rpeaks_from_rr([400.0] * 30, FS)
    spurious = np.sort(np.r_[peaks, peaks[15] + int(0.045 * FS)])
    signal = np.zeros(int(peaks[-1]) + 4000)
    windows = _record_windows(monkeypatch)
    wt = WaveTemplate()
    rr = np.diff(spurious) / FS * 1000.0
    cap = 1000.0 * MouseECG.BEAT_HALF_WIN_S

    analysis.analyse_intervals(signal, spurious, FS, rr, wave_template=wt)
    assert windows[15] == pytest.approx((-cap, cap - 0.5))
    windows.clear()
    analysis.analyse_intervals(signal, spurious, FS, rr, wave_template=wt,
                               rr_min_ms=40.0)
    assert windows[15] == pytest.approx((-cap, 31.0))     # 0.7 x 45 ms


def test_analyse_intervals_qtc_uses_the_rr_bounds_it_is_given():
    # Anaesthetised mouse, RR 400 ms > the default 333 ms Max RR: with the
    # widened bound HRV used, RR_ms and QTc are kept (they were NaN on
    # every beat while QT was measured).
    peaks = rpeaks_from_rr([400.0] * 30, FS)
    signal = np.zeros(int(peaks[-1]) + 4000)
    for p in peaks:
        add_qrs_pulse(signal, int(p), FS)
    rr = np.diff(peaks) / FS * 1000.0
    df = analysis.analyse_intervals(signal, peaks, FS, rr, wave_template=WaveTemplate(),
                                    rr_max_ms=600.0)
    assert df["RR_ms"].iloc[1:].notna().all()
    np.testing.assert_allclose(df["RR_ms"].iloc[1:], 400.0, atol=0.5)
    ok = df["QT_ms"].notna()
    np.testing.assert_allclose(
        df.loc[ok, "QTc_ms"],
        analysis.compute_qtc(df.loc[ok, "QT_ms"], df.loc[ok, "RR_ms"], "mitchell"),
        equal_nan=True)
    default = analysis.analyse_intervals(signal, peaks, FS, rr,
                                         wave_template=WaveTemplate())
    assert default["RR_ms"].isna().all() and default["QTc_ms"].isna().all()


def test_poincare_metrics_accept_an_explicit_validity_mask():
    rr = np.array([120, 121, 119, 123, 117, 122, 120, 118, 124, 116, 121, 119, 120], dtype=float)
    valid = np.ones(len(rr), dtype=bool)
    valid[6] = False   # e.g. an interval spanning a removed beat

    result = analysis.compute_poincare_metrics(rr, valid)

    assert result["n_poincare_pairs"] == len(rr) - 1 - 2


def test_poincare_metrics_compute_ccd_and_ledoit_wolf_axes():
    from sklearn.covariance import LedoitWolf

    rr_ms = np.array([120, 121, 119, 123, 117, 122, 120, 118, 124, 116, 121, 119])
    pairs = np.column_stack((rr_ms[:-1], rr_ms[1:]))
    expected_eigenvalues = np.linalg.eigvalsh(LedoitWolf().fit(pairs).covariance_)
    expected_ccd = float(np.hypot(*pairs.mean(axis=0)))

    result = analysis.compute_poincare_metrics(rr_ms)

    assert result["CCD"] == pytest.approx(expected_ccd)
    assert result["SD1_LW"] == pytest.approx(np.sqrt(expected_eigenvalues[0]))
    assert result["SD2_LW"] == pytest.approx(np.sqrt(expected_eigenvalues[-1]))
    assert result["n_poincare_pairs"] == len(pairs)


def test_poincare_metrics_do_not_bridge_nonphysiological_rr_intervals():
    rr_ms = np.array([120, 121, 20, 122, 119, 120, 118, 121, 122, 119, 120, 121])

    result = analysis.compute_poincare_metrics(rr_ms)

    assert result["n_poincare_pairs"] == len(rr_ms) - 3
    assert np.isnan(result["CCD"])
    assert np.isnan(result["SD1_LW"])
    assert np.isnan(result["SD2_LW"])


def test_poincare_metrics_withhold_values_below_minimum_pair_count():
    rr_ms = np.full(analysis.POINCARE_MIN_PAIRS, 120.0)

    result = analysis.compute_poincare_metrics(rr_ms)

    assert result["n_poincare_pairs"] == analysis.POINCARE_MIN_PAIRS - 1
    assert np.isnan(result["CCD"])
    assert np.isnan(result["SD1_LW"])
    assert np.isnan(result["SD2_LW"])


@pytest.mark.parametrize("formula", QTC_FORMULA_LABELS)
def test_qtc_formula_label_round_trips_to_canonical_key(formula):
    label = QTC_FORMULA_LABELS[formula]

    assert normalize_qtc_formula(formula) == formula
    assert normalize_qtc_formula(label) == formula


def test_unknown_qtc_formula_defaults_to_mitchell():
    assert normalize_qtc_formula("unknown formula") == "mitchell"


def test_qtc_formula_labels_order_and_default():
    assert list(QTC_FORMULA_LABELS) == ["mitchell", "bazett", "fridericia"]
    assert QTC_FORMULA_LABELS["mitchell"] == "Mitchell (QT/√(RR/100))"


@pytest.mark.parametrize("legacy, expected", [
    ("hodges", "mitchell"),
    ("Hodges (linear HR)", "mitchell"),
    ("Mitchell (∛RR)", "mitchell"),     # was really Fridericia in seconds
    ("Bazett (√RR)", "bazett"),
    ("  BAZETT  ", "bazett"),
])
def test_legacy_qtc_formula_values_map_to_current_keys(legacy, expected):
    assert normalize_qtc_formula(legacy) == expected


@pytest.mark.parametrize("formula, expected", [
    ("mitchell",   45.64),    # 50 / √(120/100)
    ("bazett",     144.34),   # 50 / √0.12
    ("fridericia", 101.37),   # 50 / ∛0.12
])
def test_compute_qtc_reference_values(formula, expected):
    assert float(analysis.compute_qtc(50.0, 120.0, formula)) == pytest.approx(expected, abs=0.01)


def test_compute_qtc_defaults_to_mitchell():
    assert float(analysis.compute_qtc(50.0, 120.0)) == pytest.approx(45.64, abs=0.01)


@pytest.mark.parametrize("formula", QTC_FORMULA_LABELS)
def test_compute_qtc_is_finite_across_mouse_rr_range(formula):
    # No absolute QTc clamp: Bazett at RR 85 ms / QT 110 ms is ~377 ms,
    # which the old 200 ms QTC_ABS_MAX turned into NaN.
    rr = np.linspace(85.0, 150.0, 66)
    for qt in (20.0, 50.0, MouseECG.QT_ABS_MAX):
        qtc = analysis.compute_qtc(np.full_like(rr, qt), rr, formula)
        assert qtc.shape == rr.shape
        assert np.isfinite(qtc).all()
        assert (qtc > 0).all()


@pytest.mark.parametrize("formula", QTC_FORMULA_LABELS)
def test_compute_qtc_nan_for_invalid_qt_or_rr(formula):
    qt = np.array([50.0, np.nan, 50.0, 50.0, 50.0, np.inf])
    rr = np.array([120.0, 120.0, np.nan, 0.0, -120.0, 120.0])

    qtc = analysis.compute_qtc(qt, rr, formula)

    assert np.isfinite(qtc[0])
    assert np.isnan(qtc[1:]).all()


def _synth_pqrst(rr_ms: np.ndarray, fs: float = FS, seed: int = 1) -> "tuple[np.ndarray, np.ndarray]":
    """QRS pulses plus a T wave ~25 ms after each R, at the given RRs --
    enough for the default WaveTemplate to find T offsets (finite QT).
    The whole T wave lies inside the beat window (0.45 x the shortest RR,
    38 ms here): a T end past the window is NaN, not an edge value."""
    rp = rpeaks_from_rr(list(rr_ms), fs)
    sig = np.zeros(int(rp[-1] + fs))
    t = np.arange(len(sig))
    for c in rp:
        add_qrs_pulse(sig, int(c), fs)
        sig += 0.25 * np.exp(-0.5 * ((t - (c + 0.025 * fs)) / (0.004 * fs)) ** 2)
    sig += np.random.default_rng(seed).normal(0.0, 0.005, len(sig))
    return sig, rp


@pytest.fixture
def _confirmed_template(monkeypatch):
    """A confirmed template never auto-saves; patching save() as well makes
    any accidental write to the user's home profile fail loudly."""
    def _refuse(self):
        raise AssertionError("WaveTemplate.save() must not run in tests")
    monkeypatch.setattr(WaveTemplate, "save", _refuse)
    wt = WaveTemplate()
    wt.confirmed = True
    return wt


def _intervals_for(sig, rp, wt, **kw):
    rr = np.diff(rp).astype(float) / FS * 1000
    return analysis.analyse_intervals(sig, rp, FS, rr, wave_template=wt, **kw)


def test_analyse_intervals_default_qtc_is_mitchell(_confirmed_template):
    rr = np.random.default_rng(0).permutation(np.linspace(85.0, 150.0, 80))
    sig, rp = _synth_pqrst(rr)

    df = _intervals_for(sig, rp, _confirmed_template)

    assert df["QT_ms"].notna().sum() > 0.9 * len(df)
    np.testing.assert_allclose(
        df["QTc_ms"].to_numpy(),
        analysis.compute_qtc(df["QT_ms"].to_numpy(), df["RR_ms"].to_numpy(), "mitchell"),
        equal_nan=True)
    # Every measured QT with a preceding RR gets a QTc -- nothing is clamped
    # away (the first peak has no preceding RR, hence no QTc).
    with_rr = df["QT_ms"].notna() & df["RR_ms"].notna()
    assert df["QTc_ms"].notna().sum() == with_rr.sum()
    assert np.isnan(df["RR_ms"].iloc[0])


@pytest.mark.parametrize("formula", ["bazett", "fridericia"])
def test_analyse_intervals_formula_matches_recompute(_confirmed_template, formula):
    rr = np.random.default_rng(0).permutation(np.linspace(85.0, 150.0, 80))
    sig, rp = _synth_pqrst(rr)

    df_default = _intervals_for(sig, rp, _confirmed_template)
    df_formula = _intervals_for(sig, rp, _confirmed_template, qtc_formula=formula)

    np.testing.assert_allclose(
        df_formula["QTc_ms"].to_numpy(),
        analysis.compute_qtc(df_formula["QT_ms"].to_numpy(),
                             df_formula["RR_ms"].to_numpy(), formula),
        equal_nan=True)
    assert df_formula["QTc_ms"].notna().sum() == (
        df_formula["QT_ms"].notna() & df_formula["RR_ms"].notna()).sum()
    # The selector / session-restore path reproduces a fresh run exactly.
    recomputed = analysis.recompute_qtc_column(df_default, formula)
    np.testing.assert_allclose(recomputed["QTc_ms"].to_numpy(),
                               df_formula["QTc_ms"].to_numpy(), equal_nan=True)
    pd.testing.assert_frame_equal(recomputed.drop(columns="QTc_ms"),
                                  df_default.drop(columns="QTc_ms"))


def test_recompute_qtc_column_leaves_frames_without_qt_untouched():
    early = pd.DataFrame({"RR_ms": [120.0, 121.0]})   # analyse_intervals' early-return shape

    assert analysis.recompute_qtc_column(early, "bazett") is early


def test_recompute_qtc_column_fixes_legacy_cube_root_session_values():
    # A session saved under the old "Mitchell (∛RR)" stored QT / ∛RR_s.
    # (QRS_onset_ms: measured with the standard definitions, RR_ms preceding.)
    old = pd.DataFrame({"RR_ms": [100.0, 120.0], "QT_ms": [50.0, 50.0],
                        "QRS_onset_ms": [-8.0, -8.0],
                        "QTc_ms": [50.0 / np.cbrt(0.10), 50.0 / np.cbrt(0.12)]})

    fixed = analysis.recompute_qtc_column(old, normalize_qtc_formula("Mitchell (∛RR)"))

    np.testing.assert_allclose(fixed["QTc_ms"], [50.0, 50.0 / np.sqrt(1.2)])
    assert old["QTc_ms"].iloc[0] == pytest.approx(50.0 / np.cbrt(0.10))   # input not mutated