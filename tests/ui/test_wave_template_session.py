"""
The wave template belongs to the recording (audit point 6), through the UI
layer without a Tk window:

* Intervals uses the recording's template (created once from the saved
  default, read only), hands the worker a copy and writes nothing;
* two recordings analysed in a row on one app give the second the same
  intervals as when analysed alone;
* the session saves and restores the template with its provenance (old
  sessions: none, the saved default is used later);
* the Landmarks editor applies to the recording only; the global default is
  written by "Save as default template" alone; a hand edit is tagged.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import wave_template
from ecg.core.wave_template import WaveTemplate
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import session_controller as sc_module
from ecg.ui import wave_editor as we_module
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.session_controller import (
    SessionController, template_info_text, wave_template_from_session,
)
from ecg.ui.state import AnalysisState, DetectionState, SignalState
from ecg.ui.wave_editor import WaveTemplateMiniEditor

from tests.core.test_wave_template_provenance import _REC_A, _REC_B, _train
from tests.ui.test_autonomic_controller import _session_app

FS = 2000.0


@pytest.fixture(autouse=True)
def _profile(tmp_path, monkeypatch):
    path = tmp_path / ".ecg_wave_template.json"
    monkeypatch.setattr(wave_template, "WAVE_TEMPLATE_PATH", path)
    monkeypatch.setattr(wave_template, "_WAVE_TEMPLATE_PATH_LEGACY",
                        path.with_suffix(".pkl"))
    return path


class _Label:
    def __init__(self) -> None:
        self.text, self.text_color = "", None

    def configure(self, **kw) -> None:
        self.text = kw.get("text", self.text)
        self.text_color = kw.get("text_color", self.text_color)


def _interval_app(pts):
    sig, rp = _train(pts)
    app = SimpleNamespace(
        analysis=AnalysisState(results={"hr": {"rr_min_ms": 50.0, "rr_max_ms": 400.0}}),
        detection=DetectionState(rpeaks_ok=rp),
        signal=SignalState(filtered=sig, fs=FS),
        after=lambda ms, fn=None: None,
        _set_status=lambda text, color=None: None,
        _set_progress=lambda pct, msg: None,
        _generation=0, lbl_ivl_status=_Label(), sw_permissive=None,
        btn_run_ivl=None,
    )
    app._windowed_peaks = lambda: app.detection.rpeaks_ok
    app._start_async_result = lambda btn, label, worker, done: done(worker())
    app._run_plot_chain = lambda tasks, on_complete=None: None
    app.analysis_ctrl = AnalysisController(app)
    return app


def _load(app, pts):
    """A new recording on the same app (reset_for_new_file drops the template)."""
    sig, rp = _train(pts)
    app.signal.filtered, app.detection.rpeaks_ok = sig, rp
    app.analysis.results = {"hr": {"rr_min_ms": 50.0, "rr_max_ms": 400.0}}
    app.analysis.wave_template = None


# ── Intervals ─────────────────────────────────────────────────────────────

def test_intervals_use_the_recordings_template_and_write_nothing(_profile):
    app = _interval_app(_REC_A)
    app.analysis_ctrl.run_intervals()
    wt = app.analysis.wave_template
    assert wt is not None and wt.source == "default" and not wt.confirmed
    assert wt.to_dict() == WaveTemplate().to_dict()          # not modified
    df = app.analysis.results["intervals"]
    assert df.attrs["wave_template"] == wt.to_dict()
    assert "built-in mouse defaults" in app.lbl_ivl_status.text
    assert not _profile.exists()


def test_second_recording_matches_its_analysis_alone(_profile):
    app = _interval_app(_REC_A)
    app.analysis_ctrl.run_intervals()
    _load(app, _REC_B)
    app.analysis_ctrl.run_intervals()
    after_a = app.analysis.results["intervals"]

    alone = _interval_app(_REC_B)
    alone.analysis_ctrl.run_intervals()
    cols = ["PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]
    pd.testing.assert_frame_equal(after_a[cols],
                                  alone.analysis.results["intervals"][cols])
    assert not _profile.exists()


def test_a_saved_default_is_used_read_only(_profile):
    saved = WaveTemplate()
    saved.landmarks["P_peak"] = (-48.0, 20.0)
    saved.source = "edited"
    saved.save()
    stamp = _profile.read_text(encoding="utf-8")
    app = _interval_app(_REC_A)
    app.analysis_ctrl.run_intervals()
    wt = app.analysis.wave_template
    assert wt.source == "saved-default" and wt.landmarks["P_peak"] == (-48.0, 20.0)
    assert _profile.read_text(encoding="utf-8") == stamp


# ── session ───────────────────────────────────────────────────────────────

def _restore(app, state, monkeypatch):
    monkeypatch.setattr(sc_module, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(1000), "ECG", None, FS))
    app._compute_preview_bundle = lambda sig, fs, params: {}
    return SessionController(app).restore_session_worker(state)


def test_session_round_trip_keeps_the_template_and_provenance(monkeypatch):
    app = _session_app()
    wt = WaveTemplate()
    wt.landmarks["T_offset"] = (70.0, 15.0)
    wt.source, wt.origin, wt.confirmed = "edited", "built-in mouse defaults", True
    app.analysis.wave_template = wt
    state = SessionController(app).collect_session_state()
    state = json.loads(json.dumps(state, cls=_SessionEncoder),
                       object_hook=_session_decode_hook)
    bundle = _restore(app, state, monkeypatch)
    back = bundle["wave_template"]
    assert back.to_dict() == wt.to_dict()
    assert back.provenance() == "edited on this recording (based on: built-in mouse defaults)"


def test_old_session_without_a_template_restores_none(monkeypatch):
    app = _session_app()
    state = SessionController(app).collect_session_state()
    assert "wave_template" not in state          # none chosen yet: nothing saved
    assert _restore(app, state, monkeypatch)["wave_template"] is None
    assert wave_template_from_session({"wave_template": "garbage"}) is None


def test_template_label_shows_the_provenance():
    wt = WaveTemplate()
    assert template_info_text(wt) == "Template: built-in mouse defaults"
    wt.source = "auto-calibrated"
    assert "auto-calibrated on this recording" in template_info_text(wt)


# ── Landmarks editor (no Tk window) ───────────────────────────────────────

def _editor(template):
    ed = object.__new__(WaveTemplateMiniEditor)
    ed._template, ed._parent_app = template, SimpleNamespace(_wave_template=None)
    ed._lm = {k: list(v) for k, v in {**WaveTemplate.DEFAULTS, **template.landmarks}.items()}
    ed._src, ed._origin, ed._saved = template.source, template.origin, False
    ed._intervals = None
    ed._draw = lambda: None
    ed.destroy = lambda: None
    return ed


def test_apply_changes_the_recording_only(_profile, monkeypatch):
    monkeypatch.setattr(we_module.messagebox, "showinfo", lambda *a, **k: None)
    wt = WaveTemplate()
    ed = _editor(wt)
    ed._lm["T_peak"][0] = 60.0
    ed._mark_edited()
    ed._apply()
    assert wt.landmarks["T_peak"][0] == 60.0 and wt.confirmed and ed._saved
    assert wt.provenance() == "edited on this recording (based on: built-in mouse defaults)"
    assert ed._parent_app._wave_template is wt
    assert not _profile.exists()


def test_save_as_default_is_the_explicit_write(_profile, monkeypatch):
    monkeypatch.setattr(we_module.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(we_module.messagebox, "askyesno", lambda *a, **k: False)
    wt = WaveTemplate()
    ed = _editor(wt)
    ed._set_provenance("auto-calibrated")
    ed._save_default()                         # declined: nothing happens
    assert not _profile.exists() and wt.source == "default"
    monkeypatch.setattr(we_module.messagebox, "askyesno", lambda *a, **k: True)
    ed._save_default()
    assert _profile.exists() and wt.source == "auto-calibrated"
    assert WaveTemplate.load_default().origin == (
        "originally auto-calibrated on the mean beat of another recording")


def test_a_hand_edit_keeps_its_first_origin():
    wt = WaveTemplate()
    wt.source, wt.origin = "file", "lab.json"
    ed = _editor(wt)
    ed._mark_edited()
    ed._mark_edited()
    assert ed._pending_template().provenance() == "edited on this recording (based on: loaded from lab.json)"
    ed._reset_defaults()
    assert ed._pending_template().provenance() == "built-in mouse defaults"


# ── saved default not used: reported once ─────────────────────────────────

_NARROWED = {"landmarks": {"P_peak": [-7.5, 2.0], "Q_peak": [-5.1, 2.0],
                           "T_offset": [27.4, 5.0]},
             "confirmed": True, "source": "user-confirmed",
             "created_at": "2026-06-22T17:34:46"}


def test_an_unused_saved_default_is_reported_once(_profile, monkeypatch):
    from ecg.ui import analysis_controller as ac_module
    shown = []
    monkeypatch.setattr(ac_module.messagebox, "showwarning",
                        lambda title, msg, **k: shown.append(msg))
    _profile.write_text(json.dumps(_NARROWED), encoding="utf-8")
    app = _interval_app(_REC_A)
    app.analysis_ctrl.run_intervals()
    assert app.analysis.wave_template.source == "default"
    assert len(shown) == 1 and str(_profile) in shown[0]
    assert app.analysis.results["intervals"]["PR_ms"].notna().all()
    _load(app, _REC_B)
    app.analysis_ctrl.run_intervals()
    assert len(shown) == 1                       # once per app session


def test_save_as_default_refuses_implausible_windows(_profile, monkeypatch):
    warned = []
    monkeypatch.setattr(we_module.messagebox, "showwarning",
                        lambda *a, **k: warned.append(a))
    monkeypatch.setattr(we_module.messagebox, "askyesno", lambda *a, **k: True)
    ed = _editor(WaveTemplate())
    ed._lm["P_peak"] = [-7.5, 2.0]               # on the Q wave
    ed._save_default()
    assert warned and not _profile.exists()


def test_save_as_default_records_the_recording(_profile, monkeypatch):
    monkeypatch.setattr(we_module.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(we_module.messagebox, "askyesno", lambda *a, **k: True)
    ed = _editor(WaveTemplate())
    ed._parent_app._filepath = "C:/data/mouse_A.mat"
    ed._set_provenance("auto-calibrated")
    ed._save_default()
    assert WaveTemplate.load_default().provenance().endswith(
        "originally auto-calibrated on the mean beat of mouse_A.mat")


# ── Beat Template label across recordings ─────────────────────────────────

def _label_app():
    return SimpleNamespace(analysis=AnalysisState(), lbl_session_info=_Label(),
                           lbl_template_info=_Label())


def test_template_label_does_not_keep_the_previous_recording(_profile):
    app = _label_app()
    wt = WaveTemplate()
    wt.source, wt.origin = "edited", "built-in mouse defaults"
    app.analysis.wave_template = wt
    SessionController(app).update_session_ui(has_session=False)
    assert "edited on this recording" in app.lbl_template_info.text
    app.analysis.wave_template = None            # reset_for_new_file
    SessionController(app).update_session_ui(has_session=False)
    assert app.lbl_template_info.text == (
        "Template: built-in mouse defaults (used at the next measure)")
    assert app.analysis.wave_template is None    # shown, not assigned


# ── signal controller: new file drops the template, re-detection keeps it ─

def _signal_app():
    from unittest.mock import MagicMock
    from ecg.ui.state import SessionState, UIState
    app = MagicMock()
    app.analysis, app.detection = AnalysisState(), DetectionState()
    app.signal, app.ui, app.session = SignalState(fs=FS), UIState(), SessionState()
    app._slots, app._kpi, app._generation = {}, {}, 0
    wt = WaveTemplate()
    wt.source = "auto-calibrated"
    app.analysis.wave_template = wt
    return app, wt


def _bundle():
    sig = np.zeros(4000)
    rp = np.arange(200, 3800, 400)
    return {"fs": FS, "signal_raw": sig, "signal_raw_norm": sig,
            "signal_flt": sig, "all_cands": rp, "all_proms": np.ones(len(rp)),
            "no_filter_mode": False, "rpeaks_ok": rp, "rpeaks_rej": np.array([], int),
            "thresh_amp": 0.5, "detected_ch": "ECG", "requested_ch": "ECG",
            "fs_from_file": True, "detected_fs": FS, "dur_s": 2.0}


def test_reset_for_new_file_drops_the_template():
    from ecg.ui.signal_controller import SignalController
    app, _ = _signal_app()
    SignalController(app).reset_for_new_file()
    assert app.analysis.wave_template is None


def test_raw_load_drops_the_template():
    from ecg.ui.signal_controller import SignalController
    app, _ = _signal_app()
    SignalController(app).on_raw_load_done(_bundle())
    assert app.analysis.wave_template is None


def test_re_detection_keeps_the_recordings_template():
    from ecg.ui.signal_controller import SignalController
    app, wt = _signal_app()
    SignalController(app).on_preview_done(_bundle())
    assert app.analysis.wave_template is wt and wt.source == "auto-calibrated"
    assert app.analysis.results is None          # analysis itself invalidated


def _ch_bundle(ch):
    b = _bundle()
    b["detected_ch"] = b["requested_ch"] = ch
    return b


def test_another_channel_drops_the_template():
    """A channel can be another animal: re-detecting on a different channel
    must not delineate it with windows fitted on the previous channel."""
    from ecg.ui.signal_controller import SignalController
    app, wt = _signal_app()
    ctrl = SignalController(app)
    app.signal.channel = "ECG1"
    ctrl.on_preview_done(_ch_bundle("ECG1"))     # same channel: kept
    assert app.analysis.wave_template is wt
    ctrl.on_preview_done(_ch_bundle("ECG2"))     # other channel: dropped
    assert app.analysis.wave_template is None
    assert app.signal.channel == "ECG2"
    status = app._set_status.call_args[0][0]
    assert "wave template of channel ECG1 dropped" in status
    assert "used at the next measure" in (
        app.lbl_template_info.configure.call_args.kwargs["text"])


def test_auto_detected_channel_change_drops_the_template():
    """Requested channel not found: the one load_mat_signal falls back to
    is what counts."""
    from ecg.ui.signal_controller import SignalController
    app, _ = _signal_app()
    app.signal.channel = "ECG1"
    b = _bundle()
    b["requested_ch"], b["detected_ch"] = "ECG1", "ECG2"
    SignalController(app).on_preview_done(b)
    assert app.analysis.wave_template is None


def test_raw_load_and_new_file_record_the_channel():
    from ecg.ui.signal_controller import SignalController
    app, _ = _signal_app()
    ctrl = SignalController(app)
    ctrl.on_raw_load_done(_ch_bundle("ECG3"))
    assert app.signal.channel == "ECG3"
    ctrl.reset_for_new_file()
    assert app.signal.channel is None


def test_session_restore_records_the_channel(monkeypatch):
    app = _session_app()
    state = SessionController(app).collect_session_state()
    assert _restore(app, state, monkeypatch)["detected_ch"] == "ECG"


# ── pending-template label: no repeated warning ───────────────────────────

def test_pending_label_does_not_log_the_rejection(_profile, caplog):
    from ecg.ui.session_controller import pending_template_info_text
    _profile.write_text(json.dumps(_NARROWED), encoding="utf-8")
    with caplog.at_level("WARNING"):
        for _ in range(3):
            text = pending_template_info_text()
    assert text == ("Template: built-in mouse defaults (used at the next "
                    "measure; saved default not used)")
    assert not [r for r in caplog.records if "was not used" in r.getMessage()]
    with caplog.at_level("WARNING"):
        WaveTemplate.load_default_checked()     # the Intervals path still logs
    assert any("was not used" in r.getMessage() for r in caplog.records)
