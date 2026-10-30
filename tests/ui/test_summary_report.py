"""
The Summary tab's text report (summary_report_text) and duration tile:
VLF / LF / HF are absolute powers in ms² (n.u. only for LFn / HFn), the
duration is the analysed one (window end − start) rather than the absolute
time of the last beat, everything is in English with units, the QTc label
comes from the formula used, and a PROVENANCE block states the window,
filters, RR bounds / exclusions, artifact correction, QTc formula,
interval definitions and wave template.
"""
import re
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core.analysis import INTERVAL_DEFINITIONS_TEXT
from ecg.core.models import QTC_FORMULA_LABELS
from ecg.ui.plot_controller import PlotController, summary_report_text
from ecg.ui.state import AnalysisState, DetectionState, SignalState


def _results(t0_s: float = 100.0, n: int = 500) -> dict:
    rr = np.full(n, 120.0)
    t = t0_s + np.cumsum(rr) / 1000.0
    fd = pd.DataFrame({"HRV_VLF": [1.5], "HRV_LF": [2.25], "HRV_HF": [3.5],
                       "HRV_LFn": [0.4], "HRV_HFn": [0.6], "HRV_LFHF": [0.667]})
    fd.attrs["method"] = "welch"          # current (non-legacy) spectrum
    td = pd.DataFrame({"HRV_MeanNN": [120.0], "HRV_SDNN": [3.0], "HRV_RMSSD": [2.0],
                       "HRV_pNN6": [10.0], "HRV_pNN20": [0.0]})
    ivl = pd.DataFrame({"PR_ms": [35.0, 36.0, 34.0, 35.0],
                        "QRS_onset_ms": [-6.0] * 4, "QRS_ms": [10.0, 11.0, 10.0, 9.0],
                        "QT_ms": [50.0, 52.0, 51.0, 49.0],
                        "QTc_ms": [45.0, 47.0, 46.0, 44.0], "RR_ms": [120.0] * 4})
    return {"hr": {"n": n + 1, "mean": 500.0, "min": 490.0, "max": 510.0, "std": 2.0},
            "rr_ms": rr, "rr_valid": np.ones(n, dtype=bool), "rr_t_s": t,
            "rr_df": pd.DataFrame({"Time_s": t, "RR_ms": rr, "HR_bpm": 60000.0 / rr}),
            "hrv_time": td, "hrv_freq": fd, "hrv_nonlin": None, "intervals": ivl}


def _report(**kw) -> str:
    args = dict(subject="m1", filepath="C:/data/m1.mat", filter_text="FILTERS-X",
                window=(100.0, 160.0), signal_end_s=600.0, artifact_report=None,
                qtc_formula="mitchell", when=datetime(2026, 1, 2, 3, 4))
    args.update(kw)
    return summary_report_text(_results(), **args)


def _line(text: str, label: str) -> str:
    return next(ln for ln in text.splitlines() if ln.strip().startswith(label))


def test_band_powers_are_ms2_and_only_lfn_hfn_are_nu():
    text = _report()
    assert _line(text, "VLF power").endswith("1.500 ms²")
    assert _line(text, "LF power").endswith("2.250 ms²")
    assert _line(text, "HF power").endswith("3.500 ms²")
    assert "n.u." not in _line(text, "LF power") + _line(text, "HF power")
    assert "40.0 %" in _line(text, "LF (n.u.)")
    assert "60.0 %" in _line(text, "HF (n.u.)")


def test_duration_is_the_analysed_window_not_the_last_beat_time():
    text = _report()
    assert _line(text, "Analysis window").endswith("100.0 – 160.0 s")
    dur = _line(text, "Analysed duration")
    assert "60.0 s (window end − start)" in dur
    assert "160" not in dur                  # last beat at 160 s absolute


def test_report_is_english_with_units():
    text = _report(asymmetry={"porta_pct": 48.0, "guzik_pct": 51.0,
                              "n_dec": 240, "n_acc": 230, "n_tot": 499})
    assert re.search(r"[éèàùâêîôûç]", text) is None
    for french in ("symétrie", "Décélérations", "Signal brut", "sans filtres", "Fichier"):
        assert french not in text
    assert _line(text, "Mean (60000/mean NN)").endswith("500.0 bpm")
    assert _line(text, "SDNN").endswith("3.000 ms")
    ivl_rows = [ln for ln in text.split("ECG INTERVALS", 1)[1].splitlines()[1:] if "±" in ln]
    assert len(ivl_rows) == 4 and all(ln.endswith(" ms") for ln in ivl_rows)
    assert "Decelerations" in text and "Accelerations" in text


@pytest.mark.parametrize("formula", ["mitchell", "bazett", "fridericia"])
def test_qtc_label_is_the_formula_used(formula):
    text = _report(qtc_formula=formula)
    label = QTC_FORMULA_LABELS[formula]
    assert _line(text, "QTc formula").split(None, 2)[2].startswith(label)
    assert f"QTc ({label.split(' ')[0]})" in text


def test_provenance_block_states_what_is_needed_to_reproduce():
    arep = {"n_in": 503, "n_out": 500, "n_nonphysio": 1, "n_ectopic": 2, "n_duplicate": 0}
    text = _report(artifact_report=arep)
    prov = text.split("PROVENANCE", 1)[1].split("HEART RATE", 1)[0]
    for label in ("Analysis window", "Analysed duration", "Filters", "RR intervals",
                  "RR bounds (HRV)", "Artifact correction", "QTc formula",
                  "Interval definitions", "Wave template"):
        assert label in prov
    assert "FILTERS-X" in prov
    assert "3 of 503 beats removed" in prov
    # The (wrapped) interval definitions as now implemented.
    flat = " ".join(prov.split())
    assert "QT = QRS onset → T end" in flat
    assert INTERVAL_DEFINITIONS_TEXT.split(";")[0] in flat


def test_withheld_spectrum_says_why():
    r = _results()
    fd = pd.DataFrame()
    fd.attrs["withheld_reason"] = "too few beats"
    r["hrv_freq"] = fd
    text = summary_report_text(r, qtc_formula="mitchell")
    assert "too few beats" in _line(text, "Withheld")
    assert "VLF power" not in text


# ── Duration tile ────────────────────────────────────────────────────────────

class _Label:
    def __init__(self) -> None:
        self.text = ""

    def configure(self, **kw) -> None:
        self.text = kw.get("text", self.text)

    def cget(self, key):
        return self.text


def test_duration_tile_shows_analysed_duration():
    r = _results(t0_s=100.0)
    app = SimpleNamespace(
        analysis=AnalysisState(results=r, analysed_window=(100.0, 160.0)),
        detection=DetectionState(),
        signal=SignalState(time=np.linspace(0.0, 600.0, 1201)),
        _kpi=defaultdict(_Label), _topbar_vals={}, quality_gauge=None)
    PlotController(app).update_kpis()   # type: ignore[arg-type]
    assert app._kpi["dur"].text == "60"

    # Full recording: window end = end of the signal.
    app.analysis.analysed_window = (0.0, 0.0)
    PlotController(app).update_kpis()   # type: ignore[arg-type]
    assert app._kpi["dur"].text == "600"


def test_missing_values_are_a_bare_dash():
    r = _results()
    r["hrv_time"] = None
    r["hrv_freq"] = pd.DataFrame({"HRV_VLF": [np.nan]}).assign(**{"HRV_LF": [np.nan]})
    r["hrv_freq"].attrs["method"] = "welch"
    text = summary_report_text(r, qtc_formula="mitchell")
    for label in ("MeanNN", "pNN6", "pNN20", "LF (n.u.)", "HF (n.u.)", "LF/HF",
                  "SampEn", "DFA α1"):
        line = _line(text, label)
        assert line.rstrip().endswith("—"), line
        assert "— %" not in line and "—  (" not in line


def test_asymmetry_block_states_the_literature_conventions():
    text = _report(asymmetry={"porta_pct": 48.0, "guzik_pct": 51.0, "n_dec": 240,
                              "n_acc": 230, "n_zero": 29, "n_tot": 499})
    block = " ".join(text.split("RR ASYMMETRY", 1)[1].split())
    assert "Porta index (PI) 48.0 %" in block
    assert "Guzik index (GI) 51.0 %" in block
    assert "Porta 2008: points below the identity line" in block
    assert "ΔRR = 0 excluded" in block
    assert "Guzik 2006: Σ distance of points above the identity line" in block
    assert "complement" not in block
    assert "No change 29 / 499" in block


def test_provenance_starts_with_the_signal_crop_when_recorded():
    from ecg.core.analysis import signal_crop_info
    text = _report(signal_crop=signal_crop_info(30.0, 630.0, 900.0))
    prov = text.split("PROVENANCE", 1)[1].split("HEART RATE", 1)[0]
    assert prov.strip().startswith("Signal crop")
    assert "file time 30.0 – 630.0 s" in prov
    assert "Signal crop" not in _report()


def test_pnn6_tile_key_is_pnn6():
    r = _results()
    app = SimpleNamespace(
        analysis=AnalysisState(results=r, analysed_window=(100.0, 160.0)),
        detection=DetectionState(),
        signal=SignalState(time=np.linspace(0.0, 600.0, 1201)),
        _kpi=defaultdict(_Label), _topbar_vals={}, quality_gauge=None)
    PlotController(app).update_kpis()   # type: ignore[arg-type]
    assert app._kpi["pnn6"].text == "10.0"
    assert "pnn50" not in app._kpi


def test_legacy_intervals_block_carries_its_own_note():
    # A legacy session's PR / QRS / QT were said to be legacy only in
    # PROVENANCE; the ECG INTERVALS block itself now says so too.
    r = _results()
    r["intervals"] = r["intervals"].drop(columns=["QRS_onset_ms"])
    text = summary_report_text(r, qtc_formula="mitchell", when=datetime(2026, 1, 2))
    block = text.split("ECG INTERVALS", 1)[1].split("═", 1)[0]
    assert "legacy definitions" in block
    assert "legacy" not in _report().split("ECG INTERVALS", 1)[1]
