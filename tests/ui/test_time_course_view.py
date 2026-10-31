"""
Time course tab, without a Tk window: where the views went, the former view
names, the out-of-date banner, the one confidence rule and figure of Epochs and
Sliding, the metrics Epochs adds only when ticked, the links that land on a mode,
and Recompute. The real tab is driven by a scripted GUI run, not here.
"""
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ecg.core import analysis
from ecg.core.models import MouseECG
from ecg.ui.analysis_controller import AnalysisController, draw_windowed_panels
from ecg.ui.freshness import (
    NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE, ModuleFreshness, stale_banner_text,
)
from ecg.ui.state import AnalysisState, DetectionState, SignalState, UIState
from ecg.ui.tab_keys import (
    MODE_AUTONOMIC, MODE_EPOCHS, MODE_EVENT_LOCKED, MODE_MODULES, MODE_SLIDING,
    TAB_HRV, TAB_MODULES, TAB_ORDER, TAB_TIME_COURSE, TIME_COURSE_MODES, time_course_mode,
)
from ecg.ui.tab_time_course import _EPOCH_DEFAULTS, _SLIDING_DEFAULTS, TabTimeCourseMixin

from tests.helpers import rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0
UI_DIR = Path(__file__).resolve().parents[2] / "ecg" / "ui"


# ── the tab and its modes ─────────────────────────────────────────────────

def test_tab_sits_right_after_rhythm_and_hrv():
    assert TAB_ORDER[TAB_ORDER.index(TAB_HRV) + 1] == TAB_TIME_COURSE == "Time course"


def test_modules_moved_from_rhythm_to_time_course():
    assert set(TAB_MODULES[TAB_TIME_COURSE]) == set(MODE_MODULES.values())
    assert not set(TAB_MODULES[TAB_HRV]) & set(MODE_MODULES.values())


def test_selector_left_rhythm_and_hrv():
    src = (UI_DIR / "tab_hrv.py").read_text(encoding="utf-8")
    for gone in ("CTkSegmentedButton", "_hrv_seg", "_build_hrv_view_rolling", "_auto_seg"):
        assert gone not in src
    for gone in ("hrv_current_view", "autonomic_subview"):
        assert not hasattr(UIState(), gone)


# ── former values ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("hrv_view, subview, mode", [
    ("Epochs", "Time series", MODE_EPOCHS),
    ("Epochs", "Event-locked", MODE_EPOCHS),        # the toggle was not the view shown
    ("Rolling", "Time series", MODE_SLIDING),
    ("Autonomic", "Time series", MODE_AUTONOMIC),
    ("Autonomic", "Event-locked", MODE_EVENT_LOCKED),
    (None, "Event-locked", MODE_EVENT_LOCKED),
    (None, "Time series", MODE_AUTONOMIC),
    ("Overview", "Time series", None),              # lives on Rhythm & HRV now
    ("RR / HR", "Event-locked", None),
    (None, None, None),
])
def test_former_values_map_to_their_mode(hrv_view, subview, mode):
    assert time_course_mode(None, hrv_view, subview) == mode


def test_a_stored_mode_wins_and_former_names_are_understood():
    for m in TIME_COURSE_MODES:
        assert time_course_mode(m, "Rolling", "Event-locked") == m
    assert time_course_mode("Rolling") == MODE_SLIDING
    assert time_course_mode("Time series") == MODE_AUTONOMIC
    assert time_course_mode("nonsense") is None


def test_show_mode_keeps_the_state_and_accepts_a_former_name():
    calls = []
    app = SimpleNamespace(ui=UIState(), _time_frames={}, _time_seg=None,
                          _refresh_time_banner=lambda: calls.append(1))
    TabTimeCourseMixin._show_time_mode(app, "Rolling")
    assert app.ui.time_course_mode == MODE_SLIDING
    TabTimeCourseMixin._show_time_mode(app, "nonsense")
    assert app.ui.time_course_mode == MODE_EPOCHS


def test_session_reads_the_former_values():
    src = inspect.getsource(__import__("ecg.ui.session_controller", fromlist=["x"]))
    assert re.search(r'time_course_mode\(state\.get\("time_course_mode"\),\s*'
                     r'state\.get\("hrv_current_view"\),\s*state\.get\("autonomic_subview"\)\)', src)


# ── out-of-date banner ────────────────────────────────────────────────────

def test_banner_says_nothing_unless_out_of_date():
    for state in (UP_TO_DATE, NOT_COMPUTED):
        assert stale_banner_text("Epochs", ModuleFreshness(state), "x", "y") == ""


def test_banner_gives_the_times_of_a_peak_edit():
    status = ModuleFreshness(OUT_OF_DATE, ("peaks changed",))
    now = pd.Timestamp("2026-10-30T15:00:00").to_pydatetime()
    assert stale_banner_text("Sliding", status, "2026-10-30T14:32:00", "2026-10-30T14:35:10",
                             now=now) == (
        "Peaks were edited at 14:35, after this view was computed (14:32). "
        "Values below are out of date.")


def test_banner_says_there_are_no_values_after_a_dropped_result():
    status = ModuleFreshness(OUT_OF_DATE, ("result dropped since it was computed: re-run",))
    text = stale_banner_text("Epochs", status, "2026-10-30T14:32:00", "2026-10-30T14:35:00",
                             has_values=False)
    assert "Peaks were edited" in text and text.endswith("Recompute to see its values.")
    assert "Values below" not in text


def test_banner_gives_the_reasons_when_the_times_are_unknown_or_not_the_cause():
    status = ModuleFreshness(OUT_OF_DATE, ("analysis window changed", "RR bounds changed"))
    assert stale_banner_text("Epochs", status, None, "2026-10-30T14:35:00") == (
        "Epochs is out of date: analysis window changed; RR bounds changed. "
        "Values below are out of date.")
    restored = ModuleFreshness(OUT_OF_DATE, ("peaks changed",))
    assert "peaks changed" in stale_banner_text("Epochs", restored, None, "2026-10-30T14:35:00")
    # An edit older than the result is not why it is out of date.
    window = ModuleFreshness(OUT_OF_DATE, ("analysis window changed",))
    assert "Peaks were" not in stale_banner_text(
        "Epochs", window, "2026-10-30T14:40:00", "2026-10-30T14:35:00")


# ── Recompute runs the mode shown ─────────────────────────────────────────

def _recompute(mode: str, autonomic_state: str = UP_TO_DATE) -> "list[str]":
    calls: "list[str]" = []
    app = SimpleNamespace(
        ui=UIState(time_course_mode=mode),
        _compute_epochs=lambda: calls.append("epochs"),
        _compute_rolling_hrv=lambda: calls.append("rolling"),
        _compute_autonomic=lambda: calls.append("autonomic"),
        _compute_event_locked=lambda: calls.append("event_locked"),
        freshness=SimpleNamespace(status=lambda m: ModuleFreshness(autonomic_state)),
        analysis_ctrl=SimpleNamespace(
            compute_autonomic=lambda then=None: (calls.append("autonomic"), then())),
    )
    TabTimeCourseMixin._recompute_time_mode(app)
    return calls


def test_recompute_runs_the_mode_shown():
    assert _recompute(MODE_EPOCHS) == ["epochs"]
    assert _recompute(MODE_SLIDING) == ["rolling"]
    assert _recompute(MODE_AUTONOMIC) == ["autonomic"]
    assert _recompute(MODE_EVENT_LOCKED) == ["event_locked"]


def test_recompute_of_event_locked_first_renews_a_stale_autonomic_result():
    for state in (OUT_OF_DATE, NOT_COMPUTED):
        assert _recompute(MODE_EVENT_LOCKED, state) == ["autonomic", "event_locked"]


# ── one figure, one confidence rule ───────────────────────────────────────

def test_hollow_points_rule_is_the_one_the_legend_states():
    ctrl = AnalysisController(SimpleNamespace())
    assert ctrl.MIN_CONFIDENT_BEATS == 10 and ctrl.MIN_VALID_INTERVALS == 3
    assert "fewer than 10 valid intervals" in ctrl.WINDOWED_LEGEND
    assert "fewer than 3 valid intervals or 5 beats" in ctrl.WINDOWED_LEGEND
    assert list(ctrl._low_confidence([9, 10, 11])) == [True, False, False]
    assert list(ctrl._low_confidence([50, 50], [9, 10])) == [True, False]


def test_legend_states_the_pair_floor_of_the_lw_metrics():
    # A window with enough intervals can still be hollow: the legend says why.
    ctrl = AnalysisController(SimpleNamespace())
    assert list(ctrl._low_confidence(np.array([12]), np.array([5]))) == [True]
    assert f"fewer than {analysis.POINCARE_MIN_PAIRS} valid pairs" in ctrl.WINDOWED_LEGEND


def test_figure_has_hollow_points_and_gaps():
    df = pd.DataFrame({"HR": [1.0, 2.0, 3.0, 4.0], "SDNN": [5.0, 6.0, 7.0, 8.0]})
    fig = Figure()
    t = np.array([0.0, 10.0, 40.0, 50.0])              # a skipped window at 20-30
    low = np.array([True, False, False, False])
    draw_windowed_panels(fig, t, df, [("HR", "HR", "red", "a"), ("SDNN", "SD", "blue", "b")], low, 15.0)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        line, hollow, solid = ax.lines
        assert np.isnan(np.asarray(line.get_ydata(), dtype=float)).any()
        assert hollow.get_markerfacecolor() == "none" and len(hollow.get_xdata()) == 1
        assert len(solid.get_xdata()) == 3


def test_figure_with_a_single_panel():
    fig = Figure()
    draw_windowed_panels(fig, np.array([0.0, 1.0]), pd.DataFrame({"HR": [1.0, 2.0]}),
                         [("HR", "HR", "red", "a")], np.array([False, False]), 2.0)
    assert len(fig.axes) == 1


def test_lw_metrics_are_ticked_in_autonomic_but_stay_with_sliding():
    auto = inspect.getsource(TabTimeCourseMixin._build_time_view_autonomic)
    sliding = inspect.getsource(TabTimeCourseMixin._build_time_view_sliding)
    assert "self._roll_metrics[metric] = cb" in auto and "SD1_LW" in auto
    assert "SD1_LW" not in sliding
    assert "compute_rolling_hrv" in inspect.getsource(AnalysisController.compute_rolling_hrv)


def test_default_ticks_match_what_each_mode_always_did():
    assert _EPOCH_DEFAULTS == AnalysisController.EPOCH_BASE_METRICS
    assert _SLIDING_DEFAULTS == ("HR", "SDNN", "RMSSD")


def test_caveat_and_calibration_stay_with_autonomic():
    auto = inspect.getsource(TabTimeCourseMixin._build_time_view_autonomic)
    assert "CALIBRATION_WARNING" in auto and "_calibrate_button" in auto
    ev = inspect.getsource(TabTimeCourseMixin._build_time_view_event_locked)
    assert "_open_compare_segments" in ev and "_calibrate_button" in ev


# ── stubs for the two jobs ────────────────────────────────────────────────

class _Entry:
    def __init__(self, value) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value


class _Tick:
    def __init__(self, on: bool) -> None:
        self.on = int(on)

    def get(self) -> int:
        return self.on


class _Label:
    def configure(self, **kw) -> None:
        self.__dict__.update(kw)


class _Slot:
    def update(self, fn) -> None:
        self.draw_fn = fn

    def _show_placeholder(self) -> None:
        self.draw_fn = None


def _safe_float(widget, default):
    try:
        return float(widget.get())
    except (TypeError, ValueError):
        return default


def _rr(n: int, seed: int = 2) -> "list[float]":
    t = np.arange(n) * 0.12
    return list(120.0 + 4.0 * np.sin(2 * np.pi * 2.5 * t)
                + np.random.default_rng(seed).normal(0.0, 1.0, n))


def _app(peaks: np.ndarray, epoch_ticks: "tuple[str, ...] | None" = None):
    tabs: "list[str]" = []
    later: "list" = []
    modes: "list[str]" = []
    texts: dict = {}
    app = SimpleNamespace(
        analysis=AnalysisState(results={"hr": {}}),
        detection=DetectionState(rpeaks_ok=peaks),
        signal=SignalState(filtered=np.zeros(int(peaks[-1]) + 10), fs=FS, time=np.zeros(1)),
        ui=UIState(), _generation=0, _safe_float=_safe_float,
        after=lambda ms, fn=None: later.append(fn),
        _set_status=lambda text, color=None: None,
        ent_minrr=_Entry(MouseECG.RR_MIN_MS), ent_maxrr=_Entry(MouseECG.RR_MAX_MS),
        ent_epoch=_Entry(30), ent_overlap=_Entry(0), ent_roll_win=_Entry(30), ent_roll_step=_Entry(10),
        _roll_metrics={m: _Tick(True) for m in ("HR", "SDNN")},
        lbl_roll_status=_Label(), lbl_epoch_info=_Label(), lbl_epoch_count=_Label(),
        btn_roll_compute=None, btn_compute_epochs=None,
        _slots={"rolling_hrv": _Slot(), "epochs": _Slot()}, txt_rolling="rolling", txt_epochs="epochs",
        _set_textbox=lambda box, text, tsv=None: texts.__setitem__(box, text),
        _df_to_tsv=lambda df: "", tabs=SimpleNamespace(set=tabs.append),
        _show_time_mode=modes.append, cb_freq_band=None,
    )
    if epoch_ticks is not None:
        all_ticks = ("HR", "SDNN", "RMSSD", "pNN6", "SD1", "SD2", "LF_nu", "HF_nu", "LF_HF")
        app._epoch_metrics = {m: _Tick(m in epoch_ticks) for m in all_ticks}
    app._windowed_peaks = lambda: peaks
    app._start_async_result = lambda btn, label, worker, done: done(worker())
    return app, tabs, later, modes, texts


BASE_COLUMNS = ["Epoch_start_s", "Epoch_end_s", "N_beats", "N_valid", "N_excluded",
                "HR_mean", "MeanNN", "SDNN", "RMSSD", "pNN6"]


# ── links land on the right mode ──────────────────────────────────────────

@requires_nk
def test_a_finished_epoch_run_opens_epochs():
    app, tabs, later, modes, _ = _app(rpeaks_from_rr(_rr(1500), FS))
    AnalysisController(app).compute_epochs()
    assert tabs == [TAB_TIME_COURSE]
    later[-1]()
    assert modes == [MODE_EPOCHS]


@requires_nk
def test_a_finished_rolling_run_opens_sliding():
    app, tabs, later, modes, _ = _app(rpeaks_from_rr(_rr(1500), FS))
    AnalysisController(app).compute_rolling_hrv()
    assert tabs == [TAB_TIME_COURSE]
    later[-1]()
    assert modes == [MODE_SLIDING]


# ── Epochs: extras only when ticked ───────────────────────────────────────

@requires_nk
def test_default_epochs_frame_is_unchanged():
    # Stub without ticks, and the default ticks: the same base columns.
    peaks = rpeaks_from_rr(_rr(1500), FS)
    a, *_ = _app(peaks)
    b, *_ = _app(peaks, _EPOCH_DEFAULTS)
    AnalysisController(a).compute_epochs()
    AnalysisController(b).compute_epochs()
    assert list(a.analysis.epoch_df.columns) == BASE_COLUMNS
    pd.testing.assert_frame_equal(a.analysis.epoch_df, b.analysis.epoch_df)


@requires_nk
def test_a_ticked_extra_is_only_added_and_equals_sliding():
    peaks = rpeaks_from_rr(_rr(1500), FS)
    base, *_ = _app(peaks)
    ticked, *_ = _app(peaks, ("HR", "SD1", "LF_HF"))
    AnalysisController(base).compute_epochs()
    AnalysisController(ticked).compute_epochs()
    ep = ticked.analysis.epoch_df
    assert list(ep.columns) == BASE_COLUMNS + ["SD1", "LF_HF"]
    pd.testing.assert_frame_equal(ep[BASE_COLUMNS], base.analysis.epoch_df)
    # On the same grid, the same value as the Sliding column.
    ro, *_ = _app(peaks)
    ro.ent_roll_win, ro.ent_roll_step = _Entry(30), _Entry(30)
    ro._roll_metrics = {m: _Tick(True) for m in ("HR", "SD1", "LF_HF")}
    AnalysisController(ro).compute_rolling_hrv()
    for col in ("SD1", "LF_HF"):
        np.testing.assert_array_equal(ep[col].to_numpy(dtype=float),
                                      ro.analysis.rolling_hrv_df[col].to_numpy(dtype=float))
    # Plotted: what is ticked, and base columns stay in the frame when unticked.
    fig = Figure()
    ticked._slots["epochs"].draw_fn(fig)
    assert len(fig.axes) == 3


@requires_nk
def test_no_tick_refuses_like_sliding():
    statuses = []
    app, *_ = _app(rpeaks_from_rr(_rr(600), FS), ())
    app._set_status = lambda text, color=None: statuses.append(text)
    AnalysisController(app).compute_epochs()
    assert app.analysis.epoch_df is None and statuses == ["Select at least one metric."]


# ── redrawing a stored frame (UI rebuild, session restore) ────────────────

@requires_nk
def test_stored_frames_draw_again_after_a_rebuild():
    app, _tabs, _later, _modes, texts = _app(rpeaks_from_rr(_rr(1500), FS))
    ctrl = AnalysisController(app)
    ctrl.compute_epochs()
    ctrl.compute_rolling_hrv()
    first = dict(texts)
    fig0 = Figure()
    app._slots["rolling_hrv"].draw_fn(fig0)
    app._slots.update(rolling_hrv=_Slot(), epochs=_Slot())
    texts.clear()
    ctrl.show_epochs()
    ctrl.show_rolling()
    assert texts == first
    fig1 = Figure()
    app._slots["rolling_hrv"].draw_fn(fig1)
    assert [a.get_title(loc="left") for a in fig0.axes] == [a.get_title(loc="left") for a in fig1.axes]
    assert "window 30s · step 10s" in fig1.axes[0].get_title(loc="left")


@requires_nk
def test_frames_of_an_older_session_draw_without_their_record():
    app, _tabs, _later, _modes, texts = _app(rpeaks_from_rr(_rr(1500), FS))
    ctrl = AnalysisController(app)
    ctrl.compute_epochs()
    ctrl.compute_rolling_hrv()
    app.analysis.windowed_view = {}
    app._slots.update(rolling_hrv=_Slot(), epochs=_Slot())
    ctrl.show_epochs()
    ctrl.show_rolling()
    fig = Figure()
    app._slots["epochs"].draw_fn(fig)
    assert len(fig.axes) == 4
    fig = Figure()
    app._slots["rolling_hrv"].draw_fn(fig)
    assert len(fig.axes) == 2 and "step 10s" in fig.axes[0].get_title(loc="left")
    assert "window" not in fig.axes[0].get_title(loc="left")


def test_nothing_to_draw_without_a_frame():
    ctrl = AnalysisController(SimpleNamespace(analysis=AnalysisState()))
    assert ctrl.show_epochs() == 0 and ctrl.show_rolling() == 0


# ── frames saved before the valid counts existed (before 72e0367) ─────────

def _old_frames(app) -> None:
    """The columns the row dicts of that version did not have."""
    a = app.analysis
    a.epoch_df = a.epoch_df.drop(columns=["N_valid", "N_excluded"])
    a.rolling_hrv_df = a.rolling_hrv_df.drop(
        columns=["n_valid", "n_excluded", "n_poincare_pairs", "CCD", "SD1_LW", "SD2_LW"])
    a.windowed_view = {}


@requires_nk
def test_frames_older_than_the_valid_counts_draw_and_say_so():
    app, _tabs, _later, _modes, texts = _app(rpeaks_from_rr(_rr(1500), FS))
    ctrl = AnalysisController(app)
    ctrl.compute_epochs()
    ctrl.compute_rolling_hrv()
    _old_frames(app)
    app.analysis.epoch_df.loc[0, "N_beats"] = 7     # marked as then: by its beats
    app._slots.update(rolling_hrv=_Slot(), epochs=_Slot())
    assert ctrl.show_epochs() == 0 and ctrl.show_rolling() == 0
    fig = Figure()
    app._slots["epochs"].draw_fn(fig)
    hollow = fig.axes[0].lines[1]
    assert hollow.get_markerfacecolor() == "none" and len(hollow.get_xdata()) == 1
    fig = Figure()
    app._slots["rolling_hrv"].draw_fn(fig)
    assert fig.axes
    assert "no N_valid column" in texts["epochs"] and "10 beats (N_beats)" in texts["epochs"]
    assert "no n_valid column" in texts["rolling"] and "(n_beats)" in texts["rolling"]


@requires_nk
def test_a_lw_tick_on_a_frame_without_pairs_draws():
    app, *_ = _app(rpeaks_from_rr(_rr(1500), FS))
    ctrl = AnalysisController(app)
    app._roll_metrics = {"HR": _Tick(True), "CCD": _Tick(True)}
    ctrl.compute_rolling_hrv()
    app.analysis.rolling_hrv_df = app.analysis.rolling_hrv_df.drop(columns=["n_poincare_pairs"])
    assert ctrl.show_rolling() == 0


# ── nothing of a former result stays on screen ────────────────────────────

def _computed_app():
    app, _tabs, _later, _modes, texts = _app(rpeaks_from_rr(_rr(1500), FS))
    # The text and the Copy for Excel store, as _set_textbox keeps them.
    store: dict = {}

    def _set_textbox(box, text, tsv=None):
        texts[box] = text
        if tsv is None:
            store.pop(box, None)
        else:
            store[box] = tsv
    app._set_textbox = _set_textbox
    app._df_to_tsv = lambda df: "table"
    app.analysis_ctrl = AnalysisController(app)
    app.analysis_ctrl.compute_epochs()
    app.analysis_ctrl.compute_rolling_hrv()
    assert store == {"epochs": "table", "rolling": "table"} and app.lbl_epoch_count.text
    return app, texts, store


def _assert_epochs_empty(app, texts, store) -> None:
    assert texts["epochs"] == "" and "epochs" not in store
    assert app._slots["epochs"].draw_fn is None
    for lbl in (app.lbl_epoch_count, app.lbl_epoch_info):
        assert lbl.text == ""


@requires_nk
def test_a_new_file_empties_epochs_and_sliding():
    from unittest.mock import MagicMock
    from ecg.ui.signal_controller import SignalController
    from ecg.ui.state import SessionState
    app, texts, store = _computed_app()
    shell = MagicMock()
    shell.analysis, shell.detection = app.analysis, app.detection
    shell.signal, shell.ui, shell.session = app.signal, app.ui, SessionState()
    shell._slots, shell._kpi, shell._generation = app._slots, {}, 0
    shell._set_textbox = app._set_textbox
    for attr in ("txt_epochs", "txt_rolling", "lbl_epoch_count", "lbl_epoch_info",
                 "lbl_roll_status"):
        setattr(shell, attr, getattr(app, attr))
    shell.analysis_ctrl = AnalysisController(shell)
    shell.analysis_ctrl.invalidate_autonomic = lambda peaks_changed=True: None
    SignalController(shell).reset_for_new_file()
    assert app.analysis.epoch_df is None and app.analysis.rolling_hrv_df is None
    assert app.analysis.windowed_view == {}
    _assert_epochs_empty(app, texts, store)
    assert texts["rolling"] == "" and "rolling" not in store
    assert app.lbl_roll_status.text == "  Click Analyze first"
    assert app._slots["rolling_hrv"].draw_fn is None


@requires_nk
def test_a_peak_edit_empties_epochs_like_the_banner_says():
    from ecg.ui.plot_controller import PlotController
    app, texts, store = _computed_app()
    app.analysis.epoch_df = None                    # _drop_results
    app._set_result_btns_enabled = lambda on, attrs: None
    PlotController(app).reset_hrv_readouts()
    _assert_epochs_empty(app, texts, store)
    # The screen a UI rebuild shows for the same state.
    assert app.analysis_ctrl.show_epochs() == 0
    _assert_epochs_empty(app, texts, store)


@requires_nk
def test_a_peak_edit_keeps_the_sliding_figure_with_its_frame():
    from ecg.ui.plot_controller import PlotController
    app, _texts, store = _computed_app()
    app.analysis.epoch_df = None
    plot = PlotController(app)
    label = app.lbl_roll_status.text
    app.lbl_freq_status = app.lbl_nonlin_status = None
    app._set_result_btns_enabled = lambda on, attrs: None
    plot.reset_result_plots()
    plot.reset_tab_status_labels()
    assert app._slots["rolling_hrv"].draw_fn is not None and "rolling" in store
    assert app.lbl_roll_status.text == label
    assert app._slots["epochs"].draw_fn is None


# ── the ticks follow the session ──────────────────────────────────────────

class _Box(_Tick):
    def select(self) -> None:
        self.on = 1

    def deselect(self) -> None:
        self.on = 0


def _tick_ctrl():
    from ecg.ui.session_controller import SessionController
    app = SimpleNamespace(
        _epoch_metrics={m: _Box(m in _EPOCH_DEFAULTS)
                        for m in ("HR", "SDNN", "RMSSD", "pNN6", "SD1", "LF_HF")},
        _roll_metrics={m: _Box(m in _SLIDING_DEFAULTS) for m in ("HR", "SDNN", "RMSSD", "CCD")})
    return app, SessionController(app)


def test_restore_ticks_what_was_saved():
    app, ctrl = _tick_ctrl()
    ctrl.restore_window_ticks({"epoch_metrics": {"SD1": True, "HR": False, "gone": True},
                               "roll_metrics": {"CCD": True}})
    assert app._epoch_metrics["SD1"].on and not app._epoch_metrics["HR"].on
    assert app._roll_metrics["CCD"].on and app._roll_metrics["HR"].on


def test_restore_without_saved_ticks_uses_what_the_frames_were_computed_with():
    app, ctrl = _tick_ctrl()
    ctrl.restore_window_ticks({"windowed_view": {"epochs": {"active": ["HR", "LF_HF"]}}})
    assert {m for m, c in app._epoch_metrics.items() if c.on} == {"HR", "LF_HF"}
    assert {m for m, c in app._roll_metrics.items() if c.on} == set(_SLIDING_DEFAULTS)


def test_session_without_any_record_leaves_the_ticks():
    app, ctrl = _tick_ctrl()
    ctrl.restore_window_ticks({})
    assert {m for m, c in app._epoch_metrics.items() if c.on} == set(_EPOCH_DEFAULTS)


def test_session_round_trip_carries_the_ticks(monkeypatch):
    from ecg.ui import session_controller as sc_module
    from ecg.ui.session_controller import SessionController
    from tests.ui.test_autonomic_controller import _session_app
    app = _session_app()
    app._epoch_metrics = {"HR": _Box(True), "SD1": _Box(True), "LF_HF": _Box(False)}
    state = SessionController(app).collect_session_state()
    assert state["epoch_metrics"] == {"HR": True, "SD1": True, "LF_HF": False}
    monkeypatch.setattr(sc_module, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(1000), "ECG", None, FS))
    app._compute_preview_bundle = lambda sig, fs, params: {}
    bundle = SessionController(app).restore_session_worker(state)
    assert bundle["epoch_metrics"] == state["epoch_metrics"]


# ── opening the tab shows its mode ────────────────────────────────────────

def test_opening_the_tab_runs_the_too_short_check_of_the_mode():
    shown = []
    tab = [TAB_HRV]
    app = SimpleNamespace(ui=UIState(), tabs=SimpleNamespace(get=lambda: tab[0]),
                          _show_time_mode=shown.append)
    TabTimeCourseMixin._on_main_tab_changed(app)
    assert shown == []
    tab[0] = TAB_TIME_COURSE
    TabTimeCourseMixin._on_main_tab_changed(app)
    assert shown == [MODE_EPOCHS]
    assert "command=self._on_main_tab_changed" in (UI_DIR / "app.py").read_text(encoding="utf-8")


def test_a_warning_goes_when_the_value_fits_again():
    app, *_ = _app(rpeaks_from_rr(_rr(1500), FS))
    borders = []
    app.ent_epoch = SimpleNamespace(get=lambda: "1000",
                                    configure=lambda **kw: borders.append(kw))
    ctrl = AnalysisController(app)
    ctrl.refresh_epoch_feasibility()
    assert app.lbl_epoch_info.text.startswith("Recording too short")
    app.ent_epoch.get = lambda: "30"
    ctrl.refresh_epoch_feasibility()
    assert app.lbl_epoch_info.text == ""
    assert len(borders) == 2
