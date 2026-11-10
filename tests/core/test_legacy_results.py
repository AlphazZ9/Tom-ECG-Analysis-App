"""
Results restored from an older session are brought in line with the
current metric definitions (mean HR = 60000 / mean NN, sample SD and
2nd-98th percentile range of the instantaneous HR), and a legacy NeuroKit
frequency table (arbitrary units, LFn = LF / TP) is labelled as such in
every export instead of as ms² / Task Force n.u. Also: one definition of
the instantaneous-HR spread across the Summary and Compare Segments.
"""
import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.io.export import ExcelExporter, PrismExporter, hrv_freq_sheet_frame
from ecg.io.loaders import _deserialise_results, _serialise_results

from tests.helpers import rpeaks_from_rr

FS = 2000.0


def _rr(n: int = 900, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.12
    return 120 + 4 * np.sin(2 * np.pi * 2.5 * t) + rng.normal(0, 3, n)


def _core(n: int = 900):
    peaks = rpeaks_from_rr(list(_rr(n)), FS)
    return analysis.analyse_core(np.zeros(int(peaks[-1]) + int(FS)), peaks, FS), peaks


def _legacy_freq() -> pd.DataFrame:
    # What nk.hrv_frequency stored: no "method" / "psd_freqs" attrs.
    return pd.DataFrame({"HRV_VLF": [0.120338], "HRV_LF": [0.300191],
                         "HRV_HF": [0.7802], "HRV_TP": [1.2259],
                         "HRV_LFHF": [0.3847], "HRV_LFn": [0.244868],
                         "HRV_HFn": [0.636431], "HRV_LnHF": [-0.2482]})


def _roundtrip(res: dict) -> dict:
    return _deserialise_results(_serialise_results(res))


def _cover(results) -> dict:
    wb = ExcelExporter.build_workbook(
        results=results, signal_flt=None, signal_raw=None,
        time=np.arange(10) / 1000.0, rpeaks_ok=None, fs=2000, filepath=None,
        subject="m1", sig_quality=None, epoch_df=None, qtc_formula=None)
    ws = wb["Cover"]
    return {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
            for r in range(1, ws.max_row + 1)}


# ── instantaneous HR: one definition ─────────────────────────────────────

def test_instantaneous_hr_stats_definition():
    rr = np.array([100.0, 110.0, 120.0, 130.0, np.nan, -1.0, 0.0])
    st = analysis.instantaneous_hr_stats(rr)
    ok = rr[np.isfinite(rr) & (rr > 0)]
    assert st["sd"] == pytest.approx(np.std(60_000 / ok, ddof=1))
    assert st["min"] == pytest.approx(60_000 / np.percentile(ok, 98))
    assert st["max"] == pytest.approx(60_000 / np.percentile(ok, 2))
    assert np.isnan(analysis.instantaneous_hr_stats(np.array([]))["min"])
    assert np.isnan(analysis.instantaneous_hr_stats(np.array([100.0]))["sd"])


def test_segment_over_the_whole_window_matches_the_summary_hr_spread():
    core, peaks = _core(1200)
    seg = analysis.compute_segment_stats(
        np.zeros(int(peaks[-1]) + int(FS)), peaks, FS, 0.0, peaks[-1] / FS + 1)
    hr = core["hr"]
    assert seg["hr_mean"] == pytest.approx(hr["mean"])
    assert seg["hr_sd"] == pytest.approx(hr["std"])
    assert seg["hr_min"] == pytest.approx(hr["min"])
    assert seg["hr_max"] == pytest.approx(hr["max"])


def test_cover_hr_range_says_it_is_a_percentile_range():
    core, _ = _core()
    assert analysis.HR_RANGE_TEXT in _cover(core)["HR range"]


# ── migration of older sessions ──────────────────────────────────────────

def test_old_mean_hr_is_recomputed_on_restore():
    core, _ = _core()
    rr = core["rr_df"]["RR_ms"].to_numpy(dtype=float)
    true_mean = analysis.mean_hr_bpm(rr)
    # Older versions: mean of instantaneous HR, population SD.
    core["hr"]["mean"] = float(np.mean(60_000 / rr))
    core["hr"]["std"] = float(np.std(60_000 / rr))
    old = _roundtrip(core)

    notes = analysis.migrate_legacy_results(old)

    assert old["hr"]["mean"] == pytest.approx(true_mean)
    assert old["hr"]["std"] == pytest.approx(np.std(60_000 / rr, ddof=1))
    assert any("mean HR recomputed" in n for n in notes)
    assert _cover(old)["HR mean"].startswith(f"{true_mean:.1f} bpm")


def test_migration_is_a_no_op_for_current_results():
    core, peaks = _core()
    core["hrv_freq"] = analysis.analyse_hrv_freq(peaks, FS)
    restored = _roundtrip(core)
    before = dict(restored["hr"])
    assert analysis.migrate_legacy_results(restored) == []
    for k in ("mean", "std", "min", "max"):
        assert restored["hr"][k] == pytest.approx(before[k])
    assert not analysis.hrv_freq_is_legacy(restored)
    assert analysis.migrate_legacy_results(None) == []


def test_legacy_spectrum_is_detected_and_noted():
    core, _ = _core()
    core["hrv_freq"] = _legacy_freq()
    old = _roundtrip(core)
    assert analysis.hrv_freq_is_legacy(old)
    notes = analysis.migrate_legacy_results(old)
    assert any("legacy NeuroKit spectrum" in n for n in notes)
    # Withheld (empty) or absent spectra are not "legacy".
    assert not analysis.hrv_freq_is_legacy({"hrv_freq": pd.DataFrame()})
    assert not analysis.hrv_freq_is_legacy({})


# ── exports label the legacy spectrum ────────────────────────────────────

def test_cover_labels_a_legacy_spectrum_instead_of_ms2():
    core, _ = _core()
    core["hrv_freq"] = _legacy_freq()
    text = _cover(_roundtrip(core))["Frequency HRV"]
    assert text == analysis.LEGACY_SPECTRAL_TEXT
    assert "ms²" not in text and "re-run" in text


def test_frequency_sheet_headers_carry_units():
    core, peaks = _core()
    core["hrv_freq"] = analysis.analyse_hrv_freq(peaks, FS)
    cols = list(hrv_freq_sheet_frame(core).columns)
    assert "HRV_LF (ms²)" in cols and "HRV_LFn (fraction of LF+HF)" in cols
    assert "HRV_LFHF" in cols                      # unitless ratio unchanged

    core["hrv_freq"] = _legacy_freq()
    cols = list(hrv_freq_sheet_frame(core).columns)
    assert "HRV_LF (a.u., legacy)" in cols and "HRV_LFn (fraction of TP, legacy)" in cols

    wb = ExcelExporter.build_workbook(
        results=core, signal_flt=None, signal_raw=None,
        time=np.arange(10) / 1000.0, rpeaks_ok=None, fs=2000, filepath=None,
        subject="m1", sig_quality=None, epoch_df=None, qtc_formula=None)
    ws = wb["HRV_Frequency"]
    headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    assert "HRV_LF (a.u., legacy)" in headers


def test_prism_labels_a_legacy_spectrum(tmp_path):
    pytest.importorskip("pzfx")
    core, _ = _core()
    core["hrv_freq"] = _legacy_freq()
    path = tmp_path / "legacy.pzfx"
    PrismExporter.build_and_write(path, core)
    xml = path.read_text(encoding="utf-8")
    assert "LF (a.u., legacy)" in xml and "LF (% of TP, legacy)" in xml
    assert "LF (ms²)" not in xml and "LF (n.u., %)" not in xml
