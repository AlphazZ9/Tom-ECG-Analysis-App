"""
AnalysisController's Autonomic (CSI / CPI) logic without a Tk window -- a
stub app stands in for ECGApp and runs the background job synchronously:
the computation receives the user's RR bounds, the removed-beat breaks and
the view's parameters; the result is stored, drawn and disclosed; a result
whose peaks changed meanwhile is discarded; every peak change drops it; the
parameters round-trip through the session state, tolerant of old sessions.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from ecg.core.autonomic import COV_METHODS, compute_autonomic_indices
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import analysis_controller as ac_module
from ecg.ui import detection_controller as dc_module
from ecg.ui import session_controller as sc_module
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.autonomic_view import default_autonomic_params
from ecg.ui.detection_controller import DetectionController
from ecg.ui.session_controller import SessionController
from ecg.ui.state import AnalysisState, DetectionState, SessionState, SignalState, UIState
from ecg.ui.theme import GREEN, ORANGE, RED

from tests.helpers import rpeaks_from_rr

FS = 2000.0


class _Label:
    def __init__(self, text: str = "") -> None:
        self.text, self.text_color = text, None

    def configure(self, **kw) -> None:
        self.text = kw.get("text", self.text)
        self.text_color = kw.get("text_color", self.text_color)


class _Entry:
    def __init__(self, value) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value

    def delete(self, *_a) -> None:
        self.value = ""

    def insert(self, _i, text) -> None:
        self.value = str(text)


class _Combo:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class _Check:
    def __init__(self, on: bool = False) -> None:
        self.on = on

    def get(self) -> int:
        return int(self.on)

    def select(self) -> None:
        self.on = True

    def deselect(self) -> None:
        self.on = False


class _Slot:
    def __init__(self) -> None:
        self.draw_fn = None
        self.placeholder_shown = 0

    def update(self, fn) -> None:
        self._draw_fn = self.draw_fn = fn

    def _show_placeholder(self) -> None:
        self.placeholder_shown += 1


def _safe_float(widget, default):
    try:
        return float(widget.get()) if widget is not None else default
    except (TypeError, ValueError):
        return default


def _peaks(rr_ms, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rpeaks_from_rr(list(np.asarray(rr_ms, float) + rng.normal(0, 1.5, len(rr_ms))), FS)


def _stub_app(peaks, *, window="3", method=COV_METHODS["robust"], ks="3", kp="12",
              lf_hf=False, minrr=66.7, maxrr=333.3, artifact_report=None):
    statuses: "list[tuple[str, object]]" = []
    app = SimpleNamespace(
        analysis=AnalysisState(artifact_report=artifact_report),
        detection=DetectionState(rpeaks_ok=peaks),
        signal=SignalState(fs=FS),
        ui=UIState(), session=SessionState(),
        after=lambda ms, fn=None: None,
        _set_status=lambda text, color=None: statuses.append((text, color)),
        _set_progress=lambda pct, msg: None,
        _safe_float=_safe_float,
        _generation=0,
        ent_minrr=_Entry(minrr), ent_maxrr=_Entry(maxrr),
        ent_auto_win=_Entry(window), cb_auto_method=_Combo(method),
        ent_auto_ks=_Entry(ks), ent_auto_kp=_Entry(kp), chk_auto_lfhf=_Check(lf_hf),
        btn_auto_compute=None, lbl_auto_status=_Label(),
        txt_autonomic=object(), cb_freq_band=None,
        _slots={"autonomic": _Slot()},
        _df_to_tsv=lambda df: "TSV:" + ",".join(df.columns),
    )
    app.textboxes = {}
    app._set_textbox = lambda box, text, tsv=None: app.textboxes.__setitem__(id(box), (text, tsv))
    app._windowed_peaks = lambda: app.detection.rpeaks_ok
    app._start_async_result = lambda btn, label, worker, done: done(worker())
    app.analysis_ctrl = AnalysisController(app)
    return app, statuses


def _textbox(app) -> "tuple[str, str]":
    return app.textboxes[id(app.txt_autonomic)]


# ── compute ───────────────────────────────────────────────────────────────

def test_compute_uses_the_users_bounds_breaks_and_view_parameters(monkeypatch):
    rr = [120.0] * 200 + [80.0, 160.0] + [120.0] * 200
    full = _peaks(rr)
    pvc = int(full[201])                                   # the premature beat, removed
    peaks = full[full != pvc]
    report = {"removed_samples": [pvc], "removed_types": ["ectopic"], "break_samples": [pvc]}
    app, statuses = _stub_app(peaks, window="4", method=COV_METHODS["rat_reference"],
                              ks="2", kp="10", minrr=70, maxrr=300, artifact_report=report)
    seen = {}

    def _spy(rpeaks, fs, **kw):
        seen.update(kw, rpeaks=rpeaks, fs=fs)
        return compute_autonomic_indices(rpeaks, fs, **kw)

    monkeypatch.setattr(ac_module, "compute_autonomic_indices", _spy)
    app.analysis_ctrl.compute_autonomic()

    assert seen["rr_min_ms"] == 70.0 and seen["rr_max_ms"] == 300.0
    assert list(seen["breaks"]) == [pvc]
    assert (seen["window_s"], seen["method"], seen["ks"], seen["kp"]) == (4.0, "rat_reference", 2.0, 10.0)
    res = app.analysis.autonomic_result
    assert res is not None and res.params["n_spanning_removed"] == 1
    assert app.analysis.autonomic_spwvd is None                 # LF/HF not ticked
    assert app.analysis.autonomic_params == {"window_s": 4.0, "method": "rat_reference",
                                             "ks": 2.0, "kp": 10.0, "lf_hf": False,
                                             "min_pairs": 10}
    # drawn, tabulated (with an Excel TSV) and disclosed
    fig = Figure()
    app._slots["autonomic"].draw_fn(fig)
    assert len(fig.axes) == 4
    text, tsv = _textbox(app)
    assert "SD01" in text and tsv.startswith("TSV:Method,Window_s")
    assert "1 RR excluded" in app.lbl_auto_status.text
    assert app.lbl_auto_status.text_color == ORANGE
    assert statuses[-1][0].startswith("Autonomic indices — ") and statuses[-1][1] == ORANGE


def test_clean_recording_reports_green():
    app, statuses = _stub_app(_peaks([120.0] * 400))
    app.analysis_ctrl.compute_autonomic()
    assert app.lbl_auto_status.text.endswith("✓")
    assert app.lbl_auto_status.text_color == GREEN and statuses[-1][1] == GREEN


def test_lf_hf_follows_the_band_preset(monkeypatch):
    app, _ = _stub_app(_peaks([120.0] * 400), lf_hf=True)
    app.cb_freq_band = _Combo("Mouse LF/HF custom")
    seen = {}

    def _fake_spwvd(rr, t, valid, **kw):
        seen.update(kw, n=len(rr), n_valid=int(np.sum(valid)))
        tt = np.arange(float(t[0]), float(t[-1]), 0.05)
        return {"t": tt, "lf": np.ones(len(tt)), "hf": np.ones(len(tt)),
                "params": {"lf_band": kw["lf_band"], "hf_band": kw["hf_band"],
                           "n_gaps_masked": 0}}

    monkeypatch.setattr(ac_module, "spwvd_lf_hf", _fake_spwvd)
    app.analysis_ctrl.compute_autonomic()

    assert seen["lf_band"] == (0.4, 2.0) and seen["hf_band"] == (2.0, 5.0)
    assert seen["n"] == seen["n_valid"] == 400
    assert app.analysis.autonomic_spwvd["params"]["lf_band"] == (0.4, 2.0)
    fig = Figure()
    app._slots["autonomic"].draw_fn(fig)
    assert len(fig.axes) == 5


def test_an_spwvd_failure_is_disclosed_and_csi_cpi_kept(monkeypatch):
    app, statuses = _stub_app(_peaks([120.0] * 400), lf_hf=True)

    def _fail(*_a, **_k):
        raise ValueError("estimated chunk peak memory 2.1 GB exceeds 1.5 GB")

    monkeypatch.setattr(ac_module, "spwvd_lf_hf", _fail)
    app.analysis_ctrl.compute_autonomic()
    assert app.analysis.autonomic_result is not None
    assert app.analysis.autonomic_spwvd == {"error": "estimated chunk peak memory 2.1 GB exceeds 1.5 GB"}
    assert "LF/HF not computed" in app.lbl_auto_status.text
    assert statuses[-1][1] == ORANGE
    assert "not computed — estimated chunk" in _textbox(app)[0]


def test_real_spwvd_runs_on_a_short_recording():
    app, _ = _stub_app(_peaks([120.0] * 600), lf_hf=True)
    app.analysis_ctrl.compute_autonomic()
    sp = app.analysis.autonomic_spwvd
    assert sp is not None and "error" not in sp and len(sp["t"]) > 100
    assert sp["params"]["lf_band"] == (0.4, 1.5)                # default preset


@pytest.mark.parametrize("field, value", [("ent_auto_win", "abc"), ("ent_auto_ks", "-2")])
def test_invalid_entries_do_not_compute(monkeypatch, field, value):
    app, statuses = _stub_app(_peaks([120.0] * 400))
    getattr(app, field).value = value
    monkeypatch.setattr(ac_module, "compute_autonomic_indices",
                        lambda *a, **k: pytest.fail("must not compute"))
    app.analysis_ctrl.compute_autonomic()
    assert app.analysis.autonomic_result is None
    assert app.lbl_auto_status.text_color == RED and statuses[-1][1] == RED
    assert app.analysis.autonomic_params == {}                   # unchanged


def test_too_few_peaks_or_none():
    app, statuses = _stub_app(None)
    app.analysis_ctrl.compute_autonomic()
    assert statuses[-1] == ("No peaks yet — click Analyze first.", RED)
    app, statuses = _stub_app(_peaks([120.0] * 5))
    app.analysis_ctrl.compute_autonomic()
    assert app.analysis.autonomic_result is None and "Not enough peaks" in statuses[-1][0]


def test_result_discarded_when_the_peaks_change_during_the_computation():
    app, statuses = _stub_app(_peaks([120.0] * 400))

    def _async(btn, label, worker, done):
        out = worker()
        app.detection.rpeaks_ok = app.detection.rpeaks_ok[:-1]      # an edit meanwhile
        done(out)

    app._start_async_result = _async
    app.analysis_ctrl.compute_autonomic()
    assert app.analysis.autonomic_result is None
    assert "Peaks changed during the computation" in app.lbl_auto_status.text


def test_identical_reassigned_peaks_are_not_a_change():
    app, _ = _stub_app(_peaks([120.0] * 400))

    def _async(btn, label, worker, done):
        out = worker()
        app.detection.rpeaks_ok = app.detection.rpeaks_ok.copy()   # same peaks
        done(out)

    app._start_async_result = _async
    app.analysis_ctrl.compute_autonomic()
    assert app.analysis.autonomic_result is not None


def test_result_discarded_silently_on_a_file_change():
    app, _ = _stub_app(_peaks([120.0] * 400))

    def _async(btn, label, worker, done):
        out = worker()
        app._generation += 1
        done(out)

    app._start_async_result = _async
    before = app.lbl_auto_status.text
    app.analysis_ctrl.compute_autonomic()
    assert app.analysis.autonomic_result is None and app.lbl_auto_status.text == before


# ── invalidation ──────────────────────────────────────────────────────────

def _computed_app():
    app, statuses = _stub_app(_peaks([120.0] * 400))
    app.analysis_ctrl.compute_autonomic()
    app.analysis.autonomic_spwvd = {"error": "x"}
    app.analysis.event_locked_df = object()
    assert app.analysis.autonomic_result is not None
    return app, statuses


def test_invalidate_drops_the_result_and_what_derives_from_it():
    app, _ = _computed_app()
    app.analysis.autonomic_params = {"window_s": 5.0}
    slot = app._slots["autonomic"]
    app.analysis_ctrl.invalidate_autonomic()
    assert app.analysis.autonomic_result is None
    assert app.analysis.autonomic_spwvd is None and app.analysis.event_locked_df is None
    assert slot._draw_fn is None and slot.placeholder_shown == 1
    assert _textbox(app)[0] == ""
    assert "Peaks changed" in app.lbl_auto_status.text
    assert app.analysis.autonomic_params == {"window_s": 5.0}        # kept
    # a new file: the idle hint, not "peaks changed"
    app.analysis_ctrl.invalidate_autonomic(peaks_changed=False)
    assert app.lbl_auto_status.text == AnalysisController.AUTONOMIC_IDLE_TEXT


def _detection_app():
    app, statuses = _computed_app()
    app._reset_kpis = app._reset_result_plots = lambda: None
    app.plot_ctrl = SimpleNamespace(refresh_live_rr=lambda: None)
    app.btn_check_agreement = None
    ctrl = DetectionController(app)
    ctrl.update_signal_quality = lambda accepted: None
    return app, ctrl


def test_a_peak_edit_drops_the_result_even_without_core_results():
    app, ctrl = _detection_app()
    assert app.analysis.results is None
    ctrl._invalidate_stale_results()
    assert app.analysis.autonomic_result is None


def test_any_redetection_that_changes_the_peaks_drops_the_result(monkeypatch):
    # e.g. the Abnormal Events viewer's edit mode, which re-detects without
    # invalidating anything else
    app, ctrl = _detection_app()
    peaks = app.detection.rpeaks_ok
    app.signal.filtered = np.zeros(int(peaks[-1]) + 10)
    app.detection.all_candidates = peaks
    app.detection.all_prominences = np.ones(len(peaks))
    monkeypatch.setattr(dc_module, "apply_threshold",
                        lambda *a, **k: (peaks.copy(), np.array([], int), 0.5))
    ctrl.run_detection(0.5)
    assert app.analysis.autonomic_result is not None             # same peaks: kept
    monkeypatch.setattr(dc_module, "apply_threshold",
                        lambda *a, **k: (peaks[:-3].copy(), np.array([], int), 0.5))
    ctrl.run_detection(0.5)
    assert app.analysis.autonomic_result is None


def test_analyze_that_corrects_the_peaks_drops_the_result():
    app, _ = _computed_app()
    peaks = app.detection.rpeaks_ok
    ctrl = app.analysis_ctrl
    # on_analysis_done returns early without results; the peak check runs first
    ctrl.on_analysis_done({"results": None, "rpeaks_ok": peaks.copy(), "artifact_report": None})
    assert app.analysis.autonomic_result is not None
    ctrl.on_analysis_done({"results": None, "rpeaks_ok": peaks[1:], "artifact_report": None})
    assert app.analysis.autonomic_result is None


def test_redraw_reads_the_current_annotations():
    app, _ = _computed_app()
    app.analysis.annotations = [{"t_start": 10.0, "t_end": 15.0, "label": "Drug", "color": "#123456"}]
    app.analysis_ctrl.redraw_autonomic_plot()
    fig = Figure()
    app._slots["autonomic"].draw_fn(fig)
    assert "Drug" in [t.get_text() for t in fig.axes[0].texts]


# ── parameters: widgets, app state, session ───────────────────────────────

def test_widget_params_are_read_tolerantly_and_applied_back():
    app, _ = _stub_app(None, window="7", method=COV_METHODS["exact"], ks="oops", kp="11",
                       lf_hf=True)
    app.analysis.autonomic_params = {"ks": 4.0}
    p = app.analysis_ctrl.autonomic_widget_params()
    assert p == {"window_s": 7.0, "method": "exact", "ks": 4.0, "kp": 11.0, "lf_hf": True,
                 "min_pairs": 10}

    # an invalid key keeps the value in effect (app state: kp 12 by default)
    applied = app.analysis_ctrl.apply_autonomic_params(
        {"window_s": 2, "method": "ledoit_wolf", "kp": "bad", "lf_hf": False})
    assert applied == app.analysis.autonomic_params
    assert applied["ks"] == 4.0 and applied["kp"] == 12.0
    assert (app.ent_auto_win.get(), app.cb_auto_method.get(), app.ent_auto_kp.get(),
            app.chk_auto_lfhf.on) == ("2", COV_METHODS["ledoit_wolf"], "12", False)
    # an old session (no autonomic_params): nothing changes
    assert app.analysis_ctrl.apply_autonomic_params(None) == applied


def test_widget_params_without_widgets_fall_back_to_state():
    app = SimpleNamespace(analysis=AnalysisState())
    assert AnalysisController(app).autonomic_widget_params() == default_autonomic_params()


def _session_app():
    app, _ = _stub_app(_peaks([120.0] * 50), window="5", method=COV_METHODS["approximate"],
                       ks="2.5", kp="14", lf_hf=True)
    app.sl_thr = None
    app._qtc_formula = lambda: "mitchell"
    app.signal.filepath = "recording.mat"
    return app


def test_session_round_trip_restores_the_parameters(monkeypatch):
    app = _session_app()
    state = SessionController(app).collect_session_state()
    assert state["autonomic_params"] == {"window_s": 5.0, "method": "approximate",
                                         "ks": 2.5, "kp": 14.0, "lf_hf": True,
                                         "min_pairs": 10}
    state = json.loads(json.dumps(state, cls=_SessionEncoder), object_hook=_session_decode_hook)

    monkeypatch.setattr(sc_module, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(1000), "ECG", None, FS))
    app._compute_preview_bundle = lambda sig, fs, params: {}
    bundle = SessionController(app).restore_session_worker(state)

    fresh, _ = _stub_app(None)
    fresh.analysis_ctrl.apply_autonomic_params(bundle["autonomic_params"])
    assert fresh.analysis.autonomic_params == state["autonomic_params"]
    assert fresh.ent_auto_win.get() == "5" and fresh.chk_auto_lfhf.on is True


def test_a_session_without_autonomic_params_restores_nothing(monkeypatch):
    app = _session_app()
    state = SessionController(app).collect_session_state()
    del state["autonomic_params"]
    monkeypatch.setattr(sc_module, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(1000), "ECG", None, FS))
    app._compute_preview_bundle = lambda sig, fs, params: {}
    bundle = SessionController(app).restore_session_worker(state)
    assert bundle["autonomic_params"] is None
    current = {"window_s": 9.0, "method": "exact", "ks": 1.0, "kp": 2.0, "lf_hf": False,
               "min_pairs": 10}
    app.analysis.autonomic_params = dict(current)
    app.analysis_ctrl.apply_autonomic_params(bundle["autonomic_params"])
    assert app.analysis.autonomic_params == current


# ── review fixes: staleness kept across a window change / UI rebuild ──────

def test_the_result_records_its_window_and_a_window_change_flags_it():
    app, statuses = _stub_app(_peaks([120.0] * 400))
    app.analysis_ctrl.compute_autonomic()
    res = app.analysis.autonomic_result
    assert res.params["analysis_window"] == (0.0, 0.0)
    t0, t1 = res.params["peak_span_s"]
    assert t0 == pytest.approx(app.detection.rpeaks_ok[0] / FS)
    assert t1 == pytest.approx(app.detection.rpeaks_ok[-1] / FS)
    assert app.lbl_auto_status.text_color == GREEN
    app.analysis.t_start, app.analysis.t_end = 10.0, 30.0
    text, color = app.analysis_ctrl.refresh_autonomic_status()
    assert color == ORANGE and app.lbl_auto_status.text_color == ORANGE
    assert "computed on the analysis window full signal, now 10 – 30 s" in text
    assert "press ⟳ Compute" in text and not text.endswith("✓")
    app.analysis.t_start, app.analysis.t_end = 0.0, 0.0
    assert app.analysis_ctrl.refresh_autonomic_status()[1] == GREEN


def test_a_rebuild_keeps_the_not_applied_and_peaks_changed_notices():
    app, _ = _stub_app(_peaks([120.0] * 400))
    app.analysis_ctrl.compute_autonomic()
    # ks / kp written by the calibration dialog, not applied yet
    app.ent_auto_ks.value = "17"
    text, color = app.analysis_ctrl.show_autonomic()     # what a UI rebuild calls
    assert color == ORANGE and "entries differ from the displayed result" in text
    assert "ks" in text and app.lbl_auto_status.text == text
    app.ent_auto_ks.value = "3"
    assert app.analysis_ctrl.show_autonomic()[1] == GREEN
    # a peak edit drops the result; the notice survives the rebuild
    app.analysis_ctrl.invalidate_autonomic()
    note = app.analysis.autonomic_stale_note
    assert note and "Peaks changed" in note
    app.lbl_auto_status.text = ""
    assert app.analysis_ctrl.show_autonomic() is None
    assert app.lbl_auto_status.text == note and app.lbl_auto_status.text_color == ORANGE


def test_stop_progress_survives_a_button_destroyed_by_a_rebuild():
    # A theme toggle during a long job destroys the job's button; restoring
    # it must not raise, or _finish_success would skip on_done (the result
    # would be dropped).
    import tkinter as tk
    from ecg.ui.app import ECGApp

    class _Gone:
        def winfo_exists(self):
            return False

        def configure(self, **_kw):
            raise tk.TclError('invalid command name ".!ctkbutton"')

    class _Raises(_Gone):
        def winfo_exists(self):
            raise tk.TclError("bad window path name")

    calls = []
    stub = SimpleNamespace(
        progress=SimpleNamespace(set=lambda v: calls.append(("set", v))),
        _prog_row=SimpleNamespace(pack_forget=lambda: calls.append("forget")),
        lbl_progress=SimpleNamespace(configure=lambda **kw: calls.append(kw)),
        _async_start_time=1.0)
    for button in (_Gone(), _Raises()):
        ECGApp._stop_progress(stub, button, "⟳ Compute")
    assert stub._async_start_time is None and ("set", 1.0) in calls
