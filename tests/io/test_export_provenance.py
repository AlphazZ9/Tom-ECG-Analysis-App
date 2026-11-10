"""
Provenance in the exports: the Excel Cover sheet and HR_Statistics, the
Prism project notes and the PDF footer state the analysis window, the
analysed duration (window end − start, not the absolute time of the last
beat), the filters, the artifact correction and the QTc formula actually
used -- the same rows as the Summary report (the rendered PDF page is
checked in test_export_pdf_render.py).
"""
import numpy as np
import pandas as pd
import pytest

from ecg.core.models import QTC_FORMULA_LABELS
from ecg.io.export import ExcelExporter, PrismExporter
from ecg.ui.export_controller import pdf_provenance_text

ARTIFACTS = {"n_in": 503, "n_out": 500, "n_nonphysio": 1, "n_ectopic": 2, "n_duplicate": 0}


def _results(t0_s: float = 100.0, n: int = 500) -> dict:
    rr = 120.0 + np.sin(np.arange(n))     # some variability for the tables
    t = t0_s + np.cumsum(rr) / 1000.0
    td = pd.DataFrame({"HRV_MeanNN": [float(rr.mean())], "HRV_SDNN": [float(rr.std())]})
    return {"hr": {"n": n + 1, "mean": 500.0, "min": 490.0, "max": 510.0, "std": 2.0},
            "rr_ms": rr, "rr_valid": np.ones(n, dtype=bool), "rr_t_s": t,
            "rr_df": pd.DataFrame({"Time_s": t, "RR_ms": rr, "HR_bpm": 60000.0 / rr}),
            "hrv_time": td}


def _workbook(window=(100.0, 160.0), artifact_report=ARTIFACTS, formula="bazett"):
    return ExcelExporter.build_workbook(
        results=_results(), signal_flt=None, signal_raw=None,
        time=np.linspace(0.0, 600.0, 1201), rpeaks_ok=None, fs=2000, filepath=None,
        subject="m1", sig_quality=None, epoch_df=None, qtc_formula=formula,
        analysis_window=window, artifact_report=artifact_report,
        filter_text="FILTERS-X")


def _cover(wb) -> dict:
    ws = wb["Cover"]
    return {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
            for r in range(1, ws.max_row + 1)}


def test_excel_cover_states_window_duration_filters_artifacts_and_qtc():
    cover = _cover(_workbook())

    assert cover["Recording duration"] == "600.0 s"
    assert "Duration" not in cover            # was the absolute time of the last sample
    assert cover["Analysis window"] == "100.0 – 160.0 s"
    assert cover["Analysed duration"].startswith("60.0 s (window end − start)")
    assert cover["Filters"] == "FILTERS-X"
    assert cover["Artifact correction"].startswith("3 of 503 beats removed")
    assert cover["QTc formula"].startswith(QTC_FORMULA_LABELS["bazett"])
    assert "preceding RR" in cover["QTc formula"]


def test_excel_cover_full_recording_and_no_artifact_correction():
    cover = _cover(_workbook(window=(0.0, 0.0), artifact_report=None))

    assert cover["Analysis window"] == "full recording (0.0 – 600.0 s)"
    assert cover["Analysed duration"].startswith("600.0 s")
    assert cover["Artifact correction"] == "not applied"


def test_excel_hr_statistics_separate_recording_and_analysed_duration():
    ws = _workbook()["HR_Statistics"]
    rows = list(ws.iter_rows(values_only=True))
    row = dict(zip(rows[0], rows[1]))

    assert "Duration_s" not in row
    assert row["Recording_duration_s"] == pytest.approx(600.0)
    assert row["Analysed_duration_s"] == pytest.approx(60.0)
    assert (row["Window_start_s"], row["Window_end_s"]) == pytest.approx((100.0, 160.0))


def test_prism_notes_carry_the_same_provenance(tmp_path):
    pytest.importorskip("pzfx")
    path = tmp_path / "out.pzfx"
    PrismExporter.build_and_write(
        path, _results(), qtc_formula="fridericia", hrv_window=(100.0, 160.0),
        signal_end_s=600.0, artifact_report=ARTIFACTS, filter_text="FILTERS-X")
    xml = path.read_text(encoding="utf-8")

    for name in ("Analysis window", "Analysed duration", "Filters", "RR bounds (HRV)",
                 "Artifact correction", "QTc formula"):
        assert f"<Name>{name}</Name>" in xml
    assert "100.0 – 160.0 s" in xml and "FILTERS-X" in xml
    assert "3 of 503 beats removed" in xml
    assert QTC_FORMULA_LABELS["fridericia"] in xml


def test_pdf_footer_states_the_provenance():
    text = pdf_provenance_text(_results(), (100.0, 160.0), 600.0, None, "FILTERS-X")

    assert "Analysis window: 100.0 – 160.0 s" in text
    assert "Analysed duration: 60.0 s (window end − start)" in text
    assert "Filters: FILTERS-X" in text
    assert "Artifact correction: not applied" in text


def test_window_past_the_signal_end_is_clamped_everywhere():
    # 600 s signal, window entered as 500-900 s: only 100 s exist.
    wb = _workbook(window=(500.0, 900.0))
    cover = _cover(wb)
    assert cover["Analysis window"].startswith("500.0 – 600.0 s")
    assert "900.0 s entered, past the signal end" in cover["Analysis window"]
    assert cover["Analysed duration"].startswith("100.0 s (window end − start)")
    rows = list(wb["HR_Statistics"].iter_rows(values_only=True))
    row = dict(zip(rows[0], rows[1]))
    assert row["Window_end_s"] == pytest.approx(600.0)
    assert row["Analysed_duration_s"] == pytest.approx(100.0)


def test_excel_cover_and_prism_state_the_signal_crop(tmp_path):
    from ecg.core.analysis import signal_crop_info
    crop = signal_crop_info(30.0, 630.0, 900.0)
    wb = ExcelExporter.build_workbook(
        results=_results(), signal_flt=None, signal_raw=None,
        time=np.linspace(0.0, 600.0, 1201), rpeaks_ok=None, fs=2000, filepath=None,
        subject="m1", sig_quality=None, epoch_df=None, qtc_formula="mitchell",
        analysis_window=(100.0, 160.0), artifact_report=None,
        filter_text="FILTERS-X", signal_crop=crop)
    cover = _cover(wb)
    assert cover["Signal crop"].startswith("file time 30.0 – 630.0 s of a 900.0 s file")
    assert "relative to the crop start" in cover["Signal crop"]
    # Not recorded (older session): no row rather than a guess.
    assert "Signal crop" not in _cover(_workbook())

    pytest.importorskip("pzfx")
    path = tmp_path / "crop.pzfx"
    PrismExporter.build_and_write(path, _results(), hrv_window=(100.0, 160.0),
                                  signal_end_s=600.0, signal_crop=crop)
    xml = path.read_text(encoding="utf-8")
    assert "<Name>Signal crop</Name>" in xml and "file time 30.0 – 630.0 s" in xml


def test_pdf_footer_states_the_crop_and_whole_recording_artifacts():
    from ecg.core.analysis import signal_crop_info
    text = pdf_provenance_text(_results(), (100.0, 160.0), 600.0, ARTIFACTS,
                               "FILTERS-X", signal_crop_info(0.0, 0.0, 600.0))
    assert text.startswith("Signal crop: none (whole file of a 600.0 s file")
    assert "3 of 503 beats removed (whole recording, before windowing" in text
    assert "spanning a removed beat are excluded from HRV" in text
