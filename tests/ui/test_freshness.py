"""
Per-module freshness through the real controllers (stub app, no Tk): editing the
peaks puts core, frequency, non-linear, intervals, epochs, rolling, Autonomic,
event-locked and events out of date; recomputing brings them back. A new file
clears it; a restored session counts as up to date unless a check disagrees.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.wave_template import WaveTemplate
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.detection_controller import DetectionController
from ecg.ui.event_locked_controller import EventLockedController
from ecg.ui.freshness import (
    FreshnessService, MODULES, NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE, combine_states,
)
from ecg.ui.state import AnalysisState, DetectionState, SignalState

from tests.helpers import make_synth_ecg

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

ALL_COMPUTED = ("core", "frequency", "nonlinear", "intervals", "epochs", "rolling",
                "autonomic", "event_locked", "events")


@pytest.fixture(autouse=True)
def _no_template_writes(monkeypatch):
    def _refuse(self):
        raise AssertionError("WaveTemplate.save() must not run in tests")
    monkeypatch.setattr(WaveTemplate, "save", _refuse)
    monkeypatch.setattr(WaveTemplate, "load_default_checked",
                        classmethod(lambda cls: (cls(), None)), raising=False)


@pytest.fixture(autouse=True)
def _no_tk(monkeypatch):
    from ecg.ui import analysis_controller as ac
    box = SimpleNamespace(showwarning=lambda *a, **k: None, showerror=lambda *a, **k: None,
                          showinfo=lambda *a, **k: None)
    monkeypatch.setattr(ac, "messagebox", box)
    monkeypatch.setattr(ac.ctk, "CTkLabel",
                        lambda *a, **kw: SimpleNamespace(grid=lambda **k: None,
                                                         destroy=lambda: None))


class _Widget:
    """A label, button, entry or checkbox: whatever is asked of it."""
    def __init__(self, value="") -> None:
        self.value = value

    def get(self):
        return self.value

    def configure(self, **kw) -> None:
        pass

    def cget(self, key):
        return self.value


class _Slot:
    _draw_fn = None

    def update(self, fn) -> None:
        pass

    def _show_placeholder(self) -> None:
        pass


class _Slots(dict):
    def __missing__(self, key):
        return _Slot()

    def get(self, key, default=None):
        return _Slot()


class _App:
    """Every method the controllers call is a no-op unless defined here."""
    _generation = 0
    _async_busy = False

    def __init__(self, sig, fs, peaks) -> None:
        self.analysis = AnalysisState()
        self.detection = DetectionState(rpeaks_ok=peaks, sig_quality=80, all_candidates=peaks)
        self.signal = SignalState(fs=fs, filtered=sig, time=np.arange(len(sig)) / fs,
                                  filepath="C:/data/m1.mat")
        self._slots = _Slots()
        self._arr_card_widgets = []
        self._arr_event_scroll = None
        self.tabs = SimpleNamespace(set=lambda name: None)
        self.ent_minrr = _Widget("66.7")
        self.ent_maxrr = _Widget("333.3")
        self.ent_epoch = _Widget("5")
        self.ent_overlap = _Widget("0")
        self.ent_roll_win = _Widget("5")
        self.ent_roll_step = _Widget("2")
        self.ent_arr_baseline = _Widget("2")
        self.ent_arr_brady_pct = _Widget("20")
        self.ent_arr_min_beats = _Widget("10")
        self.ent_auto_win = _Widget("2")
        self.cb_auto_method = _Widget("robust")
        self.ent_auto_ks = _Widget("3")
        self.ent_auto_kp = _Widget("12")
        self.chk_auto_lfhf = _Widget(0)
        self.ent_auto_minpairs = _Widget("3")
        self.cb_evt_label = _Widget("All")
        self.ent_evt_pre = _Widget("1")
        self.ent_evt_post = _Widget("1")
        self.chk_evt_z = _Widget(0)
        self.sw_permissive = _Widget(0)
        self._roll_metrics = {"HR": _Widget(1), "SDNN": _Widget(1), "RMSSD": _Widget(1)}
        self.analysis_ctrl = AnalysisController(self)
        self.event_ctrl = EventLockedController(self)
        self.detection_ctrl = DetectionController(self)
        self.freshness = FreshnessService(self)

    def __getattr__(self, name):
        if name.startswith(("lbl_", "btn_", "ent_", "cb_", "sw_", "chk_", "txt_", "sec_")):
            return _Widget()
        return lambda *a, **k: None

    def _windowed_peaks(self):
        a, rp = self.analysis, self.detection.rpeaks_ok
        if rp is None or (a.t_start <= 0 and a.t_end <= 0):
            return rp
        t = rp / self.signal.fs
        mask = t >= a.t_start
        if a.t_end > 0:
            mask &= t <= a.t_end
        return rp[mask]

    def _safe_float(self, widget, default=0.0):
        try:
            return float(widget.get())
        except Exception:
            return default

    def _qtc_formula(self):
        return self.analysis.qtc_formula

    def _start_async_result(self, btn, text, worker, done):
        done(worker())

    def _run_plot_chain(self, tasks, on_complete=None):
        if on_complete is not None:
            on_complete()

    def _df_to_tsv(self, df):
        return ""

    def _get_freq_bands(self):
        return self.analysis_ctrl.get_freq_bands()

    def _snapshot_params(self):
        return {"artifact_correction": False, "analysis_t_start": self.analysis.t_start,
                "analysis_t_end": self.analysis.t_end, "auto_epochs": False,
                "hrv_rr_bounds": self.analysis_ctrl.hrv_rr_bounds(quiet=True),
                "break_tol_samples": 1}


@pytest.fixture
def app():
    sig, fs, rp = make_synth_ecg(n_beats=120, seed=5)
    return _App(sig, fs, rp)


def _analyze(a):
    ctrl = a.analysis_ctrl
    ctrl.on_analysis_done(ctrl.analysis_worker(a._snapshot_params()))


def _compute_all(a):
    ctrl = a.analysis_ctrl
    _analyze(a)
    ctrl.run_freq()
    ctrl.run_nonlinear()
    ctrl.run_intervals()
    ctrl.compute_epochs()
    ctrl.compute_rolling_hrv()
    ctrl.compute_autonomic()
    a.analysis.annotations = [{"t_start": 8.0, "t_end": 9.0, "label": "S", "color": "#000"}]
    a.event_ctrl.compute()
    ctrl.build_arrhythmia_card = lambda *args, **kw: None
    ctrl.run_arrhythmia_analysis()


def _states(a):
    return a.freshness.states()


def _edit_peaks(a):
    """What on_detail_click does: drop the results, then commit the new peaks."""
    a.detection_ctrl._invalidate_stale_results()
    a.detection.rpeaks_ok = np.delete(a.detection.rpeaks_ok, 50)


def test_nothing_computed(app):
    assert set(_states(app).values()) == {NOT_COMPUTED}
    assert list(_states(app)) == list(MODULES)


def test_compute_edit_recompute(app):
    _compute_all(app)
    assert _states(app) == {m: UP_TO_DATE for m in ALL_COMPUTED}

    _edit_peaks(app)
    states = _states(app)
    assert states == {m: OUT_OF_DATE for m in ALL_COMPUTED}
    # Rolling and events keep a result; their reasons name what changed.
    assert app.analysis.rolling_hrv_df is not None
    assert "peaks changed" in app.freshness.status("rolling").reasons
    assert app.analysis.results is None
    assert app.freshness.status("core").reasons

    _compute_all(app)
    assert _states(app) == {m: UP_TO_DATE for m in ALL_COMPUTED}


def test_analyze_alone_leaves_the_rest_not_computed(app):
    _analyze(app)
    states = _states(app)
    assert states["core"] == UP_TO_DATE
    for m in ("frequency", "nonlinear", "intervals", "epochs", "rolling",
              "autonomic", "event_locked", "events"):
        assert states[m] == NOT_COMPUTED, m


def test_rerunning_analyze_marks_dropped_extensions_out_of_date(app):
    _analyze(app)
    app.analysis_ctrl.run_nonlinear()
    assert app.freshness.status("nonlinear").state == UP_TO_DATE
    _analyze(app)           # a new core result has no non-linear part any more
    assert app.freshness.status("core").state == UP_TO_DATE
    assert app.freshness.status("nonlinear").state == OUT_OF_DATE


def test_window_change(app):
    _compute_all(app)
    app.analysis.t_start, app.analysis.t_end = 2.0, 0.0
    fresh = app.freshness
    assert fresh.status("core").reasons == ("analysis window changed",)
    assert fresh.status("epochs").state == OUT_OF_DATE
    assert fresh.status("events").state == OUT_OF_DATE
    # The Autonomic reason is the existing check's own wording, once.
    assert len(fresh.status("autonomic").reasons) == 1
    assert "computed on the analysis window" in fresh.status("autonomic").reasons[0]
    app.analysis.t_start = 0.0
    assert fresh.status("core").state == UP_TO_DATE


def test_rr_bounds_change(app):
    _compute_all(app)
    app.ent_maxrr.value = "300"
    fresh = app.freshness
    for m in ("core", "frequency", "nonlinear", "intervals", "epochs", "rolling", "autonomic"):
        assert fresh.status(m).reasons[0] == "RR bounds changed", m
    assert fresh.status("events").state == UP_TO_DATE   # classification uses no RR bounds


def test_removed_beats_change(app):
    _compute_all(app)
    rp = app.detection.rpeaks_ok
    gone = int((rp[30] + rp[31]) // 2)
    app.analysis.artifact_report = {"break_samples": [gone], "removed_samples": [gone],
                                    "removed_types": ["ectopic"], "n_in": 1, "n_out": 1,
                                    "n_duplicate": 0, "n_nonphysio": 0, "n_ectopic": 0}
    assert "removed beats changed" in app.freshness.status("core").reasons
    assert "removed beats changed" in app.freshness.status("rolling").reasons


def test_peak_edit_during_a_job_is_caught(app):
    """The basis is the one at job start: a result landing after an edit is
    out of date, not stamped with the new peaks."""
    _analyze(app)

    def edit_then_finish(btn, text, worker, done):
        out = worker()
        app.detection.rpeaks_ok = np.delete(app.detection.rpeaks_ok, 50)
        done(out)
    app._start_async_result = edit_then_finish
    app.analysis_ctrl.compute_rolling_hrv()
    assert app.freshness.status("rolling").reasons == ("peaks changed",)


def test_core_stamp_uses_the_corrected_peaks(app):
    """Analyze's artifact correction changes the peaks: core stays up to date."""
    ctrl = app.analysis_ctrl
    bundle = ctrl.analysis_worker(app._snapshot_params())
    bundle["rpeaks_ok"] = np.delete(bundle["rpeaks_ok"], 3)
    ctrl.on_analysis_done(bundle)
    assert app.freshness.status("core").state == UP_TO_DATE


def test_new_file_clears_the_stamps(app):
    _compute_all(app)
    app.analysis = AnalysisState(basis=app.analysis.basis)
    assert OUT_OF_DATE in _states(app).values()
    app.freshness.reset()
    assert set(_states(app).values()) == {NOT_COMPUTED}


def test_wave_template_changed_after_intervals(app):
    _analyze(app)
    app.analysis_ctrl.run_intervals()
    fresh = app.freshness
    assert fresh.status("intervals").state == UP_TO_DATE
    other = app.analysis.wave_template.copy()
    other.source = "changed-for-this-test"
    assert other.to_dict() != app.analysis.wave_template.to_dict()
    app.analysis.wave_template = other
    assert "wave template changed" in fresh.status("intervals").reasons


def test_autonomic_entries_differ_is_reused(app):
    _analyze(app)
    app.analysis_ctrl.compute_autonomic()
    assert app.freshness.status("autonomic").state == UP_TO_DATE
    app.ent_auto_ks.value = "5"
    status = app.freshness.status("autonomic")
    assert status.state == OUT_OF_DATE
    assert "entries differ" in status.reasons[0]


# ── restored sessions ──────────────────────────────────────────────────────

def _restored(a, window=(0.0, 0.0)):
    """State as on_restore_session_done leaves it, from a session of *a*."""
    r = a.analysis.results
    a.analysis.basis.clear()
    a.analysis.analysed_window = window
    a.freshness.stamp_restored()
    return r


def test_restored_session_is_up_to_date_and_missing_is_not_computed(app):
    _analyze(app)
    app.analysis_ctrl.compute_epochs()
    _restored(app)
    states = _states(app)
    assert states["core"] == UP_TO_DATE and states["epochs"] == UP_TO_DATE
    for m in ("frequency", "nonlinear", "intervals", "rolling", "autonomic"):
        assert states[m] == NOT_COMPUTED, m

    # ...and a later edit flags it, as after a live computation.
    _edit_peaks(app)
    assert _states(app)["core"] == OUT_OF_DATE
    assert _states(app)["epochs"] == OUT_OF_DATE


def test_restored_results_from_another_window_are_out_of_date(app):
    _analyze(app)
    app.analysis.t_start = 5.0        # the window entries were applied after Analyze
    _restored(app)
    assert app.freshness.status("core").reasons == ("analysis window changed",)


def _save_and_restore(a):
    """Through JSON, as the session file does."""
    saved = json.loads(json.dumps(a.freshness.saved_basis()))
    a.analysis.basis.clear()
    a.analysis.analysed_window = (0.0, 0.0)
    a.freshness.stamp_restored(saved)


def _compute_series(a):
    _analyze(a)
    a.analysis_ctrl.compute_epochs()
    a.analysis_ctrl.compute_rolling_hrv()


def test_restored_epochs_and_rolling_keep_a_changed_window(app):
    _compute_series(app)
    app.analysis.t_start, app.analysis.t_end = 2.0, 10.0   # applied, not re-run
    _save_and_restore(app)
    for m in ("core", "epochs", "rolling"):
        assert app.freshness.status(m).reasons == ("analysis window changed",), m
    # Back to the window they were computed on: up to date again.
    app.analysis.t_start, app.analysis.t_end = 0.0, 0.0
    for m in ("core", "epochs", "rolling"):
        assert app.freshness.status(m).state == UP_TO_DATE, m


def test_restored_epochs_and_rolling_keep_changed_rr_bounds(app):
    _compute_series(app)
    app.ent_maxrr.value = "300"
    _save_and_restore(app)
    for m in ("core", "epochs", "rolling"):
        assert app.freshness.status(m).reasons == ("RR bounds changed",), m


def test_restored_epochs_and_rolling_recomputed_later_stay_up_to_date(app):
    """Computed on a later window than the core results: they alone are current."""
    _analyze(app)
    app.analysis.t_start = 2.0
    app.analysis_ctrl.compute_epochs()
    app.analysis_ctrl.compute_rolling_hrv()
    saved = json.loads(json.dumps(app.freshness.saved_basis()))
    app.analysis.basis.clear()
    app.analysis.analysed_window = (0.0, 0.0)
    app.freshness.stamp_restored(saved)
    assert app.freshness.status("core").state == OUT_OF_DATE
    assert app.freshness.status("epochs").state == UP_TO_DATE
    assert app.freshness.status("rolling").state == UP_TO_DATE


def test_restored_rolling_stays_stale_after_a_peak_edit_before_saving(app):
    _compute_series(app)
    _edit_peaks(app)
    assert app.analysis.rolling_hrv_df is not None
    _save_and_restore(app)
    assert app.freshness.status("rolling").reasons == ("peaks changed",)
    # A second save and restore does not launder it.
    _save_and_restore(app)
    assert app.freshness.status("rolling").reasons == ("peaks changed",)


def _compute_saved_modules(a):
    _analyze(a)
    ctrl = a.analysis_ctrl
    ctrl.run_freq()
    ctrl.run_nonlinear()
    ctrl.run_intervals()
    ctrl.compute_epochs()
    ctrl.compute_rolling_hrv()


SAVED_MODULES = ("core", "frequency", "nonlinear", "intervals", "epochs", "rolling")


def test_results_older_than_a_peak_edit_stay_out_of_date_after_a_restore(app):
    """The Abnormal Events viewer's edit keeps the results: reopening the
    session must not stamp them as computed on the edited peaks."""
    _compute_saved_modules(app)
    r = app.analysis.results
    app.detection.rpeaks_ok = np.delete(app.detection.rpeaks_ok, 50)
    for _ in range(2):      # a second save and restore does not launder it
        _save_and_restore(app)
        assert app.analysis.results is r
        for m in SAVED_MODULES:
            assert app.freshness.status(m).reasons == ("peaks changed",), m


def test_an_unedited_session_is_up_to_date_after_a_restore(app):
    _compute_saved_modules(app)
    for _ in range(2):
        _save_and_restore(app)
        for m in SAVED_MODULES:
            assert app.freshness.status(m).state == UP_TO_DATE, m


def test_results_dropped_by_an_edit_stay_out_of_date_after_a_restore(app):
    """Not 'not computed': the strip still shows when they were computed."""
    _compute_series(app)
    app.analysis_ctrl.run_freq()
    _edit_peaks(app)
    assert app.analysis.results is None
    for _ in range(2):
        _save_and_restore(app)
        states = _states(app)
        for m in ("core", "frequency", "epochs", "rolling"):
            assert states[m] == OUT_OF_DATE, m
        assert app.freshness.status("core").reasons
        # Never computed: nothing to re-run.
        assert states["nonlinear"] == states["intervals"] == NOT_COMPUTED


def test_session_without_a_saved_basis_follows_the_core_results(app):
    _compute_series(app)
    app.analysis.t_start = 2.0
    _restored(app)          # an older session: nothing recorded
    for m in ("core", "epochs", "rolling"):
        assert app.freshness.status(m).reasons == ("analysis window changed",), m


@pytest.mark.parametrize("bad", [
    "nope", {"epochs": "nope", "rolling": None}, {"epochs": {}, "rolling": {"window": [1.0]}},
    {"epochs": {"window": ["a", "b"], "rr": [1, 2]}, "rolling": {"window": [0, 0], "rr": None}},
])
def test_unreadable_saved_basis_falls_back_to_the_core_results(app, bad):
    _compute_series(app)
    app.analysis.basis.clear()
    app.analysis.analysed_window = (0.0, 0.0)
    app.freshness.stamp_restored(bad)
    assert app.freshness.status("epochs").state == UP_TO_DATE
    assert app.freshness.status("rolling").state == UP_TO_DATE


def test_legacy_session_parts_ask_for_a_rerun(app):
    _analyze(app)
    r = app.analysis.results
    r["hrv_freq"] = pd.DataFrame({"HRV_LF": [0.2]})      # no method attr: NeuroKit table
    nl = pd.DataFrame({"HRV_SD1": [1.0]})
    nl.attrs["truncated"] = True
    r["hrv_nonlin"] = nl
    r["intervals"] = pd.DataFrame({"QT_ms": [50.0]})
    _restored(app)
    fresh = app.freshness
    assert fresh.status("core").state == UP_TO_DATE
    assert "re-run Frequency HRV" in fresh.status("frequency").reasons[0]
    assert "re-run Non-linear HRV" in fresh.status("nonlinear").reasons[0]
    assert "re-run Intervals" in fresh.status("intervals").reasons[0]


def test_withheld_spectrum_counts_as_computed(app):
    _analyze(app)
    fd = pd.DataFrame()
    fd.attrs["withheld_reason"] = "too short"
    app.analysis.results["hrv_freq"] = fd
    _restored(app)
    assert app.freshness.status("frequency").state == UP_TO_DATE


# ── tab rule ───────────────────────────────────────────────────────────────

def test_combine_states():
    assert combine_states([UP_TO_DATE, NOT_COMPUTED]) == UP_TO_DATE
    assert combine_states([UP_TO_DATE, OUT_OF_DATE, NOT_COMPUTED]) == OUT_OF_DATE
    assert combine_states([NOT_COMPUTED, NOT_COMPUTED]) == NOT_COMPUTED
