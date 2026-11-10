"""
Provenance rows shared by the Summary report, the Excel Cover sheet, the
Prism notes and the PDF footer: the analysis window and the analysed
duration (window end − start, not the absolute time of the last beat), the
filters recorded at detection, the artifact correction and the QTc label
taken from the formula actually used.
"""
import numpy as np
import pytest

from ecg.core.analysis import (
    analysed_beat_span_s, analysed_duration_s, analysis_provenance_rows,
    analysis_window_rows, analysis_window_span, artifact_correction_text,
    qtc_formula_text,
)
from ecg.core.filtering import filter_description, recorded_filter_text
from ecg.core.models import QTC_FORMULA_LABELS


def _results(t0_s: float = 100.0, n: int = 500, rr_ms: float = 120.0) -> dict:
    """n RR intervals of rr_ms starting at t0_s (absolute time, s)."""
    rr = np.full(n, rr_ms)
    return {"hr": {"n": n + 1, "mean": 500.0, "min": 490.0, "max": 510.0, "std": 2.0},
            "rr_ms": rr, "rr_valid": np.ones(n, dtype=bool),
            "rr_t_s": t0_s + np.cumsum(rr) / 1000.0}


def _rows(rows) -> dict:
    return dict(rows)


# ── Window span / analysed duration ─────────────────────────────────────────

@pytest.mark.parametrize("window, end, expected", [
    ((10.0, 50.0), 300.0, (10.0, 50.0)),
    ((10.0, 0.0), 300.0, (10.0, 300.0)),      # 0 = to the end of the signal
    ((0.0, 0.0), None, (0.0, None)),           # end unknown
    (None, 300.0, (0.0, 300.0)),
])
def test_analysis_window_span(window, end, expected):
    assert analysis_window_span(window, end) == expected


def test_analysed_duration_is_window_length_not_last_beat_time():
    r = _results(t0_s=100.0, n=500)            # last beat at 160 s absolute

    assert analysed_duration_s(r, (100.0, 160.0), 600.0) == pytest.approx(60.0)
    # Without a known window end: the span of the analysed beats (60 s),
    # never the 160 s absolute time of the last beat.
    assert analysed_duration_s(r, (100.0, 0.0), None) == pytest.approx(60.0)
    assert analysed_beat_span_s(r) == pytest.approx(60.0)
    assert analysed_beat_span_s({}) is None


def test_window_rows_full_recording_and_windowed():
    r = _results()
    full = _rows(analysis_window_rows(r, (0.0, 0.0), 300.0))
    assert full["Analysis window"] == "full recording (0.0 – 300.0 s)"
    assert full["Analysed duration"].startswith("300.0 s (window end − start)")
    assert "60.0 s first → last analysed R peak" in full["Analysed duration"]

    win = _rows(analysis_window_rows(r, (100.0, 160.0), 300.0))
    assert win["Analysis window"] == "100.0 – 160.0 s"
    assert win["Analysed duration"].startswith("60.0 s (window end − start)")

    open_end = _rows(analysis_window_rows(r, (100.0, 0.0), None))
    assert open_end["Analysis window"] == "100.0 s – end"
    assert open_end["Analysed duration"] == "60.0 s first → last analysed R peak"


# ── Artifact correction / QTc / filters ─────────────────────────────────────

def test_artifact_correction_text():
    assert artifact_correction_text(None) == "not applied"
    assert (artifact_correction_text({"n_in": 10, "n_out": 10})
            == "applied — no beat removed (whole recording)")
    text = artifact_correction_text({"n_in": 503, "n_out": 500, "n_nonphysio": 1,
                                     "n_ectopic": 2, "n_duplicate": 0})
    assert text.startswith("3 of 503 beats removed (whole recording, before "
                           "windowing: 1 non-physiological, 2 ectopic, 0 duplicates)")


@pytest.mark.parametrize("formula", ["mitchell", "bazett", "fridericia"])
def test_qtc_label_follows_the_formula_used(formula):
    text = qtc_formula_text(formula)
    assert text.startswith(QTC_FORMULA_LABELS[formula])
    assert "preceding RR" in text
    for other in set(QTC_FORMULA_LABELS) - {formula}:
        assert QTC_FORMULA_LABELS[other] not in text


def test_filter_description_states_what_was_applied():
    assert filter_description({"no_filter": True}).startswith("none (raw signal")
    text = filter_description({"no_filter": False, "lp": 0.5, "hp": 250.0,
                               "notch": True, "clean_method": "neurokit"})
    assert "band-pass 0.5–250 Hz" in text
    assert "50 Hz notch" in text and "ecg_clean (neurokit)" in text
    assert "notch" not in filter_description({"no_filter": False, "notch": False})


def test_recorded_filter_text_fallbacks():
    assert recorded_filter_text("X", False) == "X"
    assert recorded_filter_text(None, True).startswith("none (raw signal")
    assert "not recorded" in recorded_filter_text(None, False)


def test_provenance_rows_cover_everything_needed_to_reproduce():
    rows = _rows(analysis_provenance_rows(
        _results(), window=(100.0, 160.0), signal_end_s=600.0,
        artifact_report=None, filter_text="F", qtc_formula="bazett"))

    for label in ("Analysis window", "Analysed duration", "Filters", "RR intervals",
                  "RR bounds (HRV)", "Artifact correction", "QTc formula"):
        assert label in rows
    assert rows["Filters"] == "F"
    assert rows["Artifact correction"] == "not applied"
    assert rows["QTc formula"].startswith(QTC_FORMULA_LABELS["bazett"])
    # No filters / formula given: no row rather than a guess.
    bare = _rows(analysis_provenance_rows(_results()))
    assert "Filters" not in bare and "QTc formula" not in bare


def test_preview_bundle_records_the_filters_applied(monkeypatch):
    from ecg.ui.signal_controller import SignalController
    monkeypatch.setattr(SignalController, "_preview_bundle",
                        staticmethod(lambda sig, fs, params, cb=None: {"no_filter_mode": False}))
    params = {"no_filter": False, "lp": 1.0, "hp": 100.0, "notch": False,
              "clean_method": "neurokit"}
    bundle = SignalController.compute_preview_bundle(np.zeros(10), 1000, params)

    # Described at the signal's fs: the band bandpass really applies
    # (1 Hz is below 0.01 × Nyquist at 1 kHz, so it is clipped to 5 Hz).
    assert bundle["filter_text"] == filter_description(params, 1000)
    assert bundle["filter_text"].startswith("Butterworth band-pass 5–100 Hz")
    assert "1–100 Hz entered, clipped" in bundle["filter_text"]
    # The FilterParams snapshot the signal was prepared with.
    assert bundle["prepare_params"]["lp"] == 1.0
    assert bundle["prepare_params"]["hp"] == 100.0


# ── Window past the signal end / crop / artifacts / filters at fs ───────────

@pytest.mark.parametrize("window, end, expected", [
    ((500.0, 900.0), 600.0, (500.0, 600.0)),   # end clamped to the signal
    ((0.0, 1000.0), 600.0, (0.0, 600.0)),
    ((700.0, 900.0), 600.0, (600.0, 600.0)),   # entirely past the end
    ((500.0, 900.0), None, (500.0, 900.0)),    # signal end unknown: as entered
])
def test_analysis_window_span_is_clamped_to_the_signal_end(window, end, expected):
    assert analysis_window_span(window, end) == expected


def test_analysed_duration_never_exceeds_the_signal():
    r = _results(t0_s=100.0, n=500)
    assert analysed_duration_s(r, (0.0, 1000.0), 600.0) == pytest.approx(600.0)
    assert analysed_duration_s(r, (500.0, 900.0), 600.0) == pytest.approx(100.0)
    rows = _rows(analysis_window_rows(r, (500.0, 900.0), 600.0))
    assert rows["Analysis window"] == ("500.0 – 600.0 s  (end 900.0 s entered, "
                                       "past the signal end)")
    assert rows["Analysed duration"].startswith("100.0 s (window end − start)")
    # A window inside the signal carries no note.
    assert "entered" not in _rows(analysis_window_rows(r, (100.0, 160.0), 600.0))["Analysis window"]


def test_signal_crop_text():
    from ecg.core.analysis import signal_crop_info, signal_crop_text
    assert signal_crop_text(None) is None
    assert signal_crop_text(signal_crop_info(0.0, 0.0, 600.0)) == (
        "none (whole file of a 600.0 s file; times are file times)")
    crop = signal_crop_info(30.0, 0.0, 600.0)          # 0 = to the file end
    assert crop == {"t_start": 30.0, "t_end": 600.0, "file_duration_s": 600.0}
    text = signal_crop_text(crop)
    assert text.startswith("file time 30.0 – 600.0 s of a 600.0 s file")
    assert "relative to the crop start (30.0 s in the file)" in text
    # End entered past the file: clamped to the file length.
    assert signal_crop_info(10.0, 900.0, 600.0)["t_end"] == 600.0
    rows = _rows(analysis_provenance_rows(_results(), (100.0, 160.0), 600.0,
                                          signal_crop=crop))
    assert list(rows)[0] == "Signal crop"


def test_artifact_text_is_whole_recording_and_legacy_aware():
    arep = {"n_in": 503, "n_out": 500, "n_nonphysio": 1, "n_ectopic": 2, "n_duplicate": 0}
    assert "whole recording, before windowing" in artifact_correction_text(arep)
    # No results: no claim about how HRV treated the intervals.
    assert "HRV" not in artifact_correction_text(arep)
    current = artifact_correction_text(arep, _results())
    assert current.endswith("RR intervals spanning a removed beat are excluded from HRV")
    legacy = {"hr": {"n": 10}, "rr_ms": np.full(9, 120.0)}     # no rr_valid
    text = artifact_correction_text(arep, legacy)
    assert "excluded from HRV" not in text
    assert "bridged" in text and "re-run Analyze" in text


def test_filter_description_reports_the_band_bandpass_applies():
    from ecg.core.filtering import bandpass_band
    p = {"no_filter": False, "lp": 0.5, "hp": 400.0, "notch": False}
    text = filter_description(p, 500)
    lo, hi = bandpass_band(0.5, 400.0, 500)
    assert hi == pytest.approx(247.5) and lo == pytest.approx(2.5)
    assert text.startswith("Butterworth band-pass 2.5–247.5 Hz")
    assert "0.5–400 Hz entered, clipped" in text and "fs = 500 Hz" in text
    # Inside the clip range: no note.
    ok = filter_description({"no_filter": False, "lp": 30.0, "hp": 150.0}, 2000)
    assert ok.startswith("Butterworth band-pass 30–150 Hz") and "clipped" not in ok
    # Degenerate band: the filter is skipped, and said so.
    assert bandpass_band(300.0, 400.0, 500) is None
    assert "skipped" in filter_description(
        {"no_filter": False, "lp": 300.0, "hp": 400.0}, 500)
    # Without fs: the nominal cut-offs, as before.
    assert "0.5–400 Hz" in filter_description(p)
