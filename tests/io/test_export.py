"""
Metadata in the exports: the Excel Cover sheet records which formula
produced the ECG_Intervals QTc_ms column, the QTc parameter description
stays formula-neutral, and every export discloses the RR intervals excluded
from HRV (counts, reasons, bounds) and keeps them as flagged rows instead of
dropping them.
"""
import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.models import QTC_FORMULA_LABELS
from ecg.io.export import ExcelExporter, PrismExporter

from tests.helpers import rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def _cover(qtc_formula):
    wb = ExcelExporter.build_workbook(
        results={"hr": {"n": 3, "mean": 500.0, "min": 480.0, "max": 520.0, "std": 5.0}},
        signal_flt=None, signal_raw=None, time=np.arange(10) / 1000.0,
        rpeaks_ok=None, fs=1000, filepath=None, subject="m1",
        sig_quality=None, epoch_df=None, qtc_formula=qtc_formula,
    )
    ws = wb["Cover"]
    return {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
            for r in range(1, ws.max_row + 1)}


@pytest.mark.parametrize("formula, key", [
    ("fridericia", "fridericia"),
    (QTC_FORMULA_LABELS["bazett"], "bazett"),
    ("Hodges (linear HR)", "mitchell"),   # legacy selector value
])
def test_excel_cover_records_qtc_formula(formula, key):
    cover = _cover(formula)

    assert cover["QTc formula"].startswith(QTC_FORMULA_LABELS[key])


def test_excel_cover_omits_qtc_formula_when_not_given():
    assert "QTc formula" not in _cover(None)


# ── RR exclusion disclosure ───────────────────────────────────────────────

FS = 2000.0


def _pause_results() -> dict:
    """analyse_core on ~120 ms beats with one 400 ms sinus pause (excluded)."""
    rng = np.random.default_rng(0)
    rr = list(120 + rng.normal(0, 2, 150)) + [400.0] + list(120 + rng.normal(0, 2, 150))
    peaks = rpeaks_from_rr(rr, FS)
    return analysis.analyse_core(np.zeros(int(peaks[-1]) + int(FS)), peaks, FS)


def _workbook(results, qtc_formula="mitchell"):
    return ExcelExporter.build_workbook(
        results=results, signal_flt=None, signal_raw=None,
        time=np.arange(10) / 1000.0, rpeaks_ok=None, fs=2000, filepath=None,
        subject="m1", sig_quality=None, epoch_df=None, qtc_formula=qtc_formula)


def _sheet_frame(ws) -> "pd.DataFrame":
    rows = list(ws.iter_rows(values_only=True))
    return pd.DataFrame(rows[1:], columns=rows[0])


@requires_nk
def test_excel_cover_discloses_excluded_rr_and_bounds():
    ws = _workbook(_pause_results())["Cover"]
    cover = {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
             for r in range(1, ws.max_row + 1)}

    assert cover["RR intervals"] == "301  (300 used for HRV)"
    assert cover["RR excluded"].startswith("1  (0.3 %)")
    assert "1 too long" in cover["RR excluded"]
    assert cover["RR bounds (HRV)"].startswith("67–333 ms")
    assert cover["QTc formula"].startswith(QTC_FORMULA_LABELS["mitchell"])


@requires_nk
def test_excel_hr_statistics_carry_the_exclusion_counts():
    df = _sheet_frame(_workbook(_pause_results())["HR_Statistics"])

    row = df.iloc[0]
    assert row["N_RR_intervals"] == 301
    assert row["N_RR_valid"] == 300
    assert row["N_RR_excluded"] == 1
    assert row["N_RR_too_long"] == 1
    assert row["N_RR_too_short"] == 0
    assert row["N_RR_spanning_removed"] == 0
    assert row["RR_excluded_pct"] == pytest.approx(100.0 / 301)
    assert row["RR_excluded_time_s"] == pytest.approx(0.4)
    assert (row["RR_min_ms"], row["RR_max_ms"]) == pytest.approx((66.7, 333.3))


@requires_nk
def test_excel_rr_timeseries_flags_excluded_rows_instead_of_dropping_them():
    results = _pause_results()
    df = _sheet_frame(_workbook(results)["RR_Timeseries"])

    assert list(df.columns) == ["Time_s", "RR_ms", "HR_bpm", "valid"]
    assert len(df) == len(results["rr_ms"])            # every interval
    assert df["valid"].map(type).eq(bool).all()        # real TRUE/FALSE cells
    excluded = df[~df["valid"]]
    assert excluded["RR_ms"].tolist() == [pytest.approx(400.0)]


def test_excel_without_rr_information_still_builds():
    ws = _workbook({"hr": {"n": 3, "mean": 500.0, "min": 480.0,
                           "max": 520.0, "std": 5.0}})["HR_Statistics"]
    df = _sheet_frame(ws)
    assert df.iloc[0]["N_RR_excluded"] is None
    assert df.iloc[0]["RR_min_ms"] == pytest.approx(66.7)


@requires_nk
def test_prism_rr_table_keeps_excluded_intervals_as_prism_excluded_values(tmp_path):
    pytest.importorskip("pzfx")
    from pzfx import read_pzfx
    results = _pause_results()
    epochs = pd.DataFrame({"Epoch_start_s": [0.0], "Epoch_end_s": [20.0],
                           "N_beats": [160], "N_valid": [158], "N_excluded": [1],
                           "HR_mean": [500.0], "SDNN": [2.0]})
    path = tmp_path / "out.pzfx"
    PrismExporter.build_and_write(path, results, epoch_df=epochs, qtc_formula="bazett")

    rr = read_pzfx(str(path), table="RR Intervals & HR", strike_action="star")
    assert len(rr) == len(results["rr_ms"])
    # The 0/1 flag is not a Y data set of the RR/HR table (Prism would
    # plot it next to RR and HR): it has a table of its own.
    assert list(rr.columns) == ["Time (s)", "RR (ms)", "HR (bpm)"]
    flag = read_pzfx(str(path), table="RR Intervals — HRV Validity")
    assert len(flag) == len(rr)
    flagged = rr[flag["Valid (1 = used for HRV)"].astype(float).to_numpy() == 0]
    assert len(flagged) == 1
    assert str(flagged["RR (ms)"].iloc[0]).endswith("*")      # Prism-excluded
    assert float(str(flagged["RR (ms)"].iloc[0]).rstrip("*")) == pytest.approx(400.0)

    xml = path.read_text(encoding="utf-8")
    assert '<d Excluded="1">400.0</d>' in xml
    for name in ("RR intervals", "RR excluded", "RR bounds (HRV)", "QTc formula"):
        assert f"<Name>{name}</Name>" in xml
    assert "1 too long" in xml

    ep = read_pzfx(str(path), table="Epoch HRV")
    assert "N_valid" not in ep.columns and "N_excluded" not in ep.columns
    counts = read_pzfx(str(path), table="Epoch HRV — Counts")
    assert list(counts.columns) == ["Time (s)", "N_beats", "N_valid", "N_excluded"]


def test_prism_rolling_table_keeps_bookkeeping_counts_out_of_the_metrics(tmp_path):
    pytest.importorskip("pzfx")
    from pzfx import read_pzfx
    rolling = pd.DataFrame({"t_mid": [5.0, 6.0], "HR": [500.0, 505.0],
                            "SDNN": [2.0, 2.1], "n_beats": [40, 41],
                            "n_valid": [39, 38], "n_excluded": [0, 2],
                            "n_poincare_pairs": [37, 34]})
    path = tmp_path / "roll.pzfx"
    PrismExporter.build_and_write(
        path, {"hr": {"n": 3, "mean": 500.0, "min": 480.0, "max": 520.0, "std": 5.0}},
        rolling_hrv_df=rolling)

    timeline = read_pzfx(str(path), table="Rolling HRV Timeline")
    assert list(timeline.columns) == ["Time (s)", "HR (bpm)", "SDNN (ms)"]
    counts = read_pzfx(str(path), table="Rolling HRV — Window Counts")
    assert list(counts.columns) == ["Time (s)", "n_beats", "n_valid",
                                    "n_excluded", "n_poincare_pairs"]
    assert counts["n_excluded"].astype(float).tolist() == [0.0, 2.0]


def _withheld_results() -> dict:
    results = _pause_results()
    fd = pd.DataFrame()
    fd.attrs["withheld_reason"] = "2.1 s run of excluded RR intervals (limit 2 s)"
    results["hrv_freq"] = fd
    return results


@requires_nk
def test_excel_cover_and_prism_notes_explain_a_withheld_spectrum(tmp_path):
    results = _withheld_results()
    ws = _workbook(results)["Cover"]
    cover = {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
             for r in range(1, ws.max_row + 1)}
    assert cover["Frequency HRV"] == "withheld — 2.1 s run of excluded RR intervals (limit 2 s)"

    pytest.importorskip("pzfx")
    path = tmp_path / "w.pzfx"
    PrismExporter.build_and_write(path, results)
    xml = path.read_text(encoding="utf-8")
    assert "<Name>Frequency HRV</Name>" in xml
    assert "withheld — 2.1 s run of excluded RR intervals" in xml


@requires_nk
def test_computed_spectrum_adds_no_withheld_row():
    ws = _workbook(_pause_results())["Cover"]
    labels = {ws.cell(row=r, column=1).value for r in range(1, ws.max_row + 1)}
    assert "Frequency HRV" not in labels
