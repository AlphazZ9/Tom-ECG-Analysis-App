"""
UI side of older results: a restored session gets the current HR
definitions and notes a legacy spectrum; the Frequency table labels that
spectrum in arbitrary units; a negative entropy cap is not "no cap".
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis, wave_template
from ecg.core.models import MouseECG
from ecg.io.loaders import _serialise_results
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import session_controller as sc_module
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.plot_controller import PlotController
from ecg.ui.session_controller import SessionController

from tests.core.test_legacy_results import _core, _legacy_freq
from tests.ui.test_autonomic_controller import _session_app

FS = 2000.0


@pytest.fixture(autouse=True)
def _profile(tmp_path, monkeypatch):
    path = tmp_path / ".ecg_wave_template.json"
    monkeypatch.setattr(wave_template, "WAVE_TEMPLATE_PATH", path)
    monkeypatch.setattr(wave_template, "_WAVE_TEMPLATE_PATH_LEGACY",
                        path.with_suffix(".pkl"))
    return path


def test_restoring_an_old_session_recomputes_mean_hr_and_flags_the_spectrum(monkeypatch):
    app = _session_app()
    state = SessionController(app).collect_session_state()
    core, _ = _core()
    rr = core["rr_df"]["RR_ms"].to_numpy(dtype=float)
    core["hr"]["mean"] = float(np.mean(60_000 / rr))       # older definition
    core["hrv_freq"] = _legacy_freq()
    state["results"] = _serialise_results(core)
    state = json.loads(json.dumps(state, cls=_SessionEncoder), object_hook=_session_decode_hook)

    monkeypatch.setattr(sc_module, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(1000), "ECG", None, FS))
    app._compute_preview_bundle = lambda sig, fs, params: {}
    bundle = SessionController(app).restore_session_worker(state)

    assert bundle["results"]["hr"]["mean"] == pytest.approx(analysis.mean_hr_bpm(rr))
    notes = bundle["legacy_notes"]
    assert any("mean HR recomputed" in n for n in notes)
    assert any("legacy NeuroKit spectrum" in n for n in notes)


class _Box:
    def __init__(self) -> None:
        self.text = ""


def test_frequency_table_labels_a_legacy_spectrum_in_arbitrary_units():
    boxes = {"td": _Box(), "fd": _Box()}

    def _set(box, text, tsv=None):
        box.text = text

    app = SimpleNamespace(txt_td=boxes["td"], txt_fd=boxes["fd"], _set_textbox=_set,
                          _df_to_tsv=lambda df: "")
    pc = object.__new__(PlotController)
    pc.app = app
    pc._rr_exclusion = lambda r: (0, None)

    r = {"hrv_time": pd.DataFrame(), "hrv_freq": _legacy_freq()}
    pc.plot_hrv_tables(r)
    text = boxes["fd"].text
    assert "legacy NeuroKit spectrum" in text
    assert "LF power (a.u.)" in text and "LF (% of TP)" in text
    assert "ms²" not in text and "n.u." not in text


class _Entry:
    def __init__(self, value) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value

    def delete(self, a, b) -> None:
        self.value = ""

    def insert(self, i, text) -> None:
        self.value = text


def test_a_negative_entropy_cap_falls_back_to_the_default():
    app = SimpleNamespace(ent_entropy_cap=_Entry("-5"))
    assert AnalysisController(app).entropy_cap() == MouseECG.ENTROPY_MAX_BEATS
    assert app.ent_entropy_cap.get() == str(MouseECG.ENTROPY_MAX_BEATS)   # shown
    app.ent_entropy_cap = _Entry("0")
    assert AnalysisController(app).entropy_cap() == 0          # explicit 0 = no cap
