# -*- coding: utf-8 -*-
"""
Morphology & Intervals and Events: what the screens say comes from the shared
table and the run's own record, the steps from freshness, and the tab names of
older sessions lead here.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core.metric_table import MISSING, MetricTable, interval_rows
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.freshness import (
    NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE, ModuleFreshness,
)
from ecg.ui.navigation_controller import NavigationController
from ecg.ui.plot_controller import qtc_rr_fit
from ecg.ui.session_controller import _TAB_KEY_MAP
from ecg.ui.state import AnalysisState
from ecg.ui.tab_keys import TAB_EVENTS, TAB_MODULES, TAB_MORPHOLOGY, TAB_ORDER
from ecg.ui.tab_morphology import (
    interval_lines, measured_headline, morphology_steps, step_tip,
)


def _frame(n=20, measured=8):
    """*n* beats, the first *measured* with PR, QRS and QT; QTc on those too."""
    nan = np.full(n - measured, np.nan)
    vals = lambda base: np.r_[base + np.arange(measured, dtype=float), nan]
    return pd.DataFrame({
        "PR_ms": vals(40.0), "QRS_ms": vals(10.0), "QT_ms": vals(50.0), "QTc_ms": vals(45.0),
        "QRS_onset_ms": vals(-5.0),
        "RR_prev_ms": 110.0 + np.arange(n, dtype=float), "RR_ms": 110.0 + np.arange(n, dtype=float),
    })


def test_only_the_two_former_morphology_tabs_are_gone():
    assert TAB_ORDER == ("Signal & Detection", "Rhythm & HRV", "Time course",
                         "Morphology & Intervals", "Events", "Report")


@pytest.mark.parametrize("saved, tab", [
    ("Intervals", TAB_MORPHOLOGY), ("📏 Intervals", TAB_MORPHOLOGY),
    ("Beat Template", TAB_MORPHOLOGY), ("〰 Beat Template", TAB_MORPHOLOGY),
    ("Arrhythmias", TAB_EVENTS), ("⚠ Arrhythmias", TAB_EVENTS),
    ("Abnormal Events", TAB_EVENTS), ("⚠ Abnormal Events", TAB_EVENTS),
])
def test_old_session_tabs_reopen_here(saved, tab):
    assert _TAB_KEY_MAP[saved] == tab


def test_tab_state_is_that_of_the_measure():
    """Analyze gives the mean beat, not the intervals: the tab stays not computed."""
    assert TAB_MODULES[TAB_MORPHOLOGY] == ("intervals",)


def test_steps_follow_freshness():
    up, stale = ModuleFreshness(UP_TO_DATE), ModuleFreshness(OUT_OF_DATE, ("wave template changed",))
    steps = morphology_steps(up, stale, 1748)
    assert [s.state for s in steps] == [UP_TO_DATE, UP_TO_DATE, OUT_OF_DATE]
    assert steps[0].text == "Mean beat · 1748 beats extracted"
    assert step_tip(steps[2]) == "Step 3: out of date — wave template changed"
    none = ModuleFreshness(NOT_COMPUTED)
    steps = morphology_steps(none, none, None)
    assert [s.state for s in steps] == [NOT_COMPUTED] * 3
    assert steps[0].text == "Mean beat"


def test_headline_is_the_share_on_all_three_intervals():
    assert measured_headline(_frame(1748, 16)) == ("16/1748 beats measured (1%) on PR, QRS and QT", "low")
    assert measured_headline(_frame(20, 19))[1] == "good"
    assert measured_headline(_frame(20, 10))[1] == "partial"
    # analyse_core's placeholder holds RR alone: not a delineation.
    assert measured_headline(pd.DataFrame({"RR_ms": [100.0, 110.0]})) == ("", "")
    assert measured_headline(None) == ("", "")


def test_lines_read_the_frame_without_recomputing_anything_else():
    df = _frame()
    lines = {l.name: l for l in interval_lines(df)}
    assert list(lines) == ["PR", "QRS", "QT", "QTc"]
    pr = df["PR_ms"].dropna()
    assert lines["PR"].median == f"{pr.median():.1f}"
    assert lines["PR"].iqr == f"{np.percentile(pr, 25):.1f}–{np.percentile(pr, 75):.1f}"
    assert lines["PR"].measured == "40% · 8/20"
    assert lines["PR"].definition == "P onset → QRS onset"
    assert "J point" in lines["QRS"].definition      # the code's QRS, not "S end"
    assert "preceding RR" in lines["QTc"].definition


def test_quartile_rows_are_additive_and_all_tier():
    t = MetricTable(interval_rows(_frame()))
    assert t.row("intervals.PR_ms.q1").tier == "all"
    assert t.value("intervals.PR_ms") == pytest.approx(43.5)   # the median row is unchanged


def test_lines_dash_when_nothing_is_there():
    for ivl in (None, pd.DataFrame({"RR_ms": [100.0, 110.0]})):
        for line in interval_lines(ivl):
            assert (line.median, line.iqr, line.measured) == (MISSING, MISSING, MISSING)


def test_all_nan_interval_gives_a_dash_not_nan():
    df = _frame()
    df["QT_ms"] = np.nan
    qt = {l.name: l for l in interval_lines(df)}["QT"]
    assert (qt.median, qt.iqr, qt.measured) == (MISSING, MISSING, "0% · 0/20")


def test_legacy_frame_does_not_claim_the_current_definitions():
    lines = interval_lines(_frame().drop(columns="QRS_onset_ms"))   # older session
    assert all(l.definition.startswith("legacy") for l in lines)


def test_qtc_rr_fit_needs_more_than_three_beats_and_matches_linregress():
    from scipy.stats import linregress
    assert qtc_rr_fit(_frame(20, 3)) is None
    df = _frame()
    fit = qtc_rr_fit(df)
    sub = df[["RR_prev_ms", "QTc_ms"]].dropna()
    ref = linregress(sub["RR_prev_ms"], sub["QTc_ms"])
    assert fit["slope"] == ref.slope and fit["r"] == ref.rvalue and len(fit["rr"]) == 8


# ── Events ────────────────────────────────────────────────────────────────

_BL = {"t_start": 0.4, "t_end": 30.4, "baseline_hr": 342.4, "reliable": True}


def test_readout_states_baseline_value_and_av_reason():
    skipped = "AV-delay check skipped — PR measured on < 50% of beats"
    text = AnalysisController.events_readout_text({"baseline": _BL, "av_note": skipped})
    assert text == f"Baseline 0.4–30.4 s  ·  342 bpm  ·  {skipped}"
    assert "—" in AnalysisController.events_readout_text(None)
    unrel = AnalysisController.events_readout_text(
        {"baseline": {**_BL, "reliable": False}, "av_note": "AV-delay check on"})
    assert "unreliable" in unrel and unrel.endswith("AV-delay check on")
    no_hr = AnalysisController.events_readout_text(
        {"baseline": {**_BL, "baseline_hr": float("nan")}, "av_note": "AV-delay check on"})
    assert no_hr.startswith("Baseline —")


def _ev(t0, t1):
    return SimpleNamespace(t_start=t0, t_end=t1)


def test_ribbon_click_finds_the_event_under_it_or_near_it():
    app = SimpleNamespace(analysis=AnalysisState(arrhythmia_events=[_ev(10, 20), _ev(50, 50)]))
    ctrl = AnalysisController(app)
    assert ctrl.event_at_time(15.0, 1.0) == 0
    assert ctrl.event_at_time(50.5, 1.0) == 1
    assert ctrl.event_at_time(35.0, 1.0) == -1
    app.analysis.arrhythmia_events = []
    assert ctrl.event_at_time(15.0, 1.0) == -1


@pytest.mark.parametrize("cur, direction, want", [
    (-1, 1, 0), (-1, -1, 2), (0, 1, 1), (2, 1, None), (0, -1, None), (2, -1, 1),
])
def test_previous_next_step_through_events_and_stop_at_the_ends(cur, direction, want):
    app = SimpleNamespace(analysis=AnalysisState(arrhythmia_events=[_ev(0, 1)] * 3,
                                                 arr_selected_idx=cur),
                          _arr_event_scroll=None, _arr_card_widgets=[])
    ctrl = NavigationController(app)
    picked = []
    ctrl.select_arrhythmia_event = picked.append
    ctrl.arr_step_event(direction)
    assert picked == ([] if want is None else [want])


def test_previous_next_without_events_do_nothing():
    app = SimpleNamespace(analysis=AnalysisState())
    ctrl = NavigationController(app)
    ctrl.select_arrhythmia_event = lambda i: pytest.fail("no event to select")
    ctrl.arr_step_event(1)


def test_a_new_results_reset_keeps_the_ribbon_of_events_still_held():
    from ecg.ui.plot_controller import PlotController

    class Slot:
        def __init__(self):
            self._draw_fn = "drawn"

        def _show_placeholder(self):
            self.drawn = False

    app = SimpleNamespace(analysis=AnalysisState(arrhythmia_run={"baseline": _BL}),
                          _slots={"arr_ribbon": Slot(), "qtc_rr": Slot()})
    PlotController(app).reset_result_plots()
    assert app._slots["arr_ribbon"]._draw_fn == "drawn"
    assert app._slots["qtc_rr"]._draw_fn is None
    app.analysis.arrhythmia_run = None
    PlotController(app).reset_result_plots()
    assert app._slots["arr_ribbon"]._draw_fn is None


# ── What stays on screen when results go ──────────────────────────────────

class _Label:
    def __init__(self, text=""):
        self.text, self.color = text, None

    def configure(self, text=None, text_color=None):
        self.text, self.color = text, text_color

    def cget(self, _name):
        return self.text


class _Slot:
    def __init__(self):
        self._draw_fn = "drawn"

    def _show_placeholder(self):
        pass

    def update(self, fn):
        self._draw_fn = fn


def _reset_app(**analysis):
    later = []
    app = SimpleNamespace(
        analysis=AnalysisState(**analysis),
        _slots={"arr_ribbon": _Slot(), "arr_detail": _Slot(), "qtc_rr": _Slot()},
        lbl_qtc_rr=_Label("slope -0.418 ms/ms  ·  r = -0.03  ·  1308 beats"),
        lbl_arrhythmia_status=_Label("  1 event — click to explore"),
        lbl_freq_status=None, lbl_nonlin_status=None,
        after=lambda ms, fn: later.append(fn),
        analysis_ctrl=SimpleNamespace(AUTONOMIC_IDLE_TEXT=""),
        _set_result_btns_enabled=lambda on, attrs: None,
        drawn=[],
    )
    app._draw_arr_detail = lambda: app.drawn.append(app.analysis.arr_selected_idx)
    return app, later


def test_dropped_intervals_take_the_qtc_caption_with_them():
    from ecg.ui.plot_controller import PlotController
    app, _ = _reset_app()
    PlotController(app).reset_result_plots()
    assert app.lbl_qtc_rr.text == "" and app._slots["qtc_rr"]._draw_fn is None


def test_a_new_analyze_clears_the_qtc_diagnostic_of_the_old_intervals():
    from ecg.ui.plot_controller import PlotController
    app, _ = _reset_app(results={"intervals": pd.DataFrame({"RR_ms": [100.0, 110.0]})})
    ctrl = PlotController(app)
    ctrl.run_plot_chain = lambda tasks, on_complete=None, auto_epochs=False: None
    ctrl.draw_core_results()
    assert app.lbl_qtc_rr.text == "" and app._slots["qtc_rr"]._draw_fn is None


def test_new_file_clears_the_caption_and_the_edit_button():
    from unittest.mock import MagicMock
    from ecg.ui.signal_controller import SignalController
    app = MagicMock()
    app.analysis = AnalysisState(arr_edit_mode=True)
    app._slots, app._kpi, app._arr_card_widgets = {}, {}, []
    app.ui.hover_after_id = None
    seen = []
    # The button reads the mode as it is at that moment.
    app._sync_arr_edit_btn.side_effect = lambda: seen.append(app.analysis.arr_edit_mode)
    SignalController(app).reset_for_new_file()
    assert seen == [False]
    app.lbl_qtc_rr.configure.assert_called_with(text="")


def test_kept_events_keep_their_status_and_their_selected_zoom():
    from ecg.ui.plot_controller import PlotController
    app, later = _reset_app(arrhythmia_run={"baseline": _BL}, arrhythmia_events=[_ev(1, 2)],
                            arr_selected_idx=0)
    ctrl = PlotController(app)
    ctrl.reset_result_plots()
    ctrl.reset_tab_status_labels()
    assert app._slots["arr_detail"]._draw_fn == "drawn"
    assert app.lbl_arrhythmia_status.text == "  1 event — click to explore"
    # Redrawn once the new peaks are in, not from the ones being dropped.
    assert app.drawn == [] and len(later) == 1
    later[0]()
    assert app.drawn == [0]


def test_a_redraw_after_the_selection_went_does_nothing():
    from ecg.ui.plot_controller import PlotController
    app, later = _reset_app(arrhythmia_run={"baseline": _BL}, arrhythmia_events=[_ev(1, 2)],
                            arr_selected_idx=0)
    PlotController(app).reset_result_plots()
    app.analysis.arr_selected_idx = -1
    later[0]()
    assert app.drawn == []


def test_no_selection_leaves_an_empty_zoom_and_no_events_the_prompt():
    from ecg.ui.plot_controller import PlotController
    app, later = _reset_app(arrhythmia_run={"baseline": _BL}, arrhythmia_events=[_ev(1, 2)])
    PlotController(app).reset_result_plots()
    assert app._slots["arr_detail"]._draw_fn is None and later == []
    assert app._slots["arr_ribbon"]._draw_fn == "drawn"
    app, later = _reset_app()
    ctrl = PlotController(app)
    ctrl.reset_result_plots()
    ctrl.reset_tab_status_labels()
    assert app.lbl_arrhythmia_status.text == "  Click Analyze first"
    assert app._slots["arr_ribbon"]._draw_fn is None


# ── The status line follows the results ───────────────────────────────────

def _status_app(ivl, wave_template=None):
    app = SimpleNamespace(analysis=AnalysisState(results={"intervals": ivl},
                                                 wave_template=wave_template),
                          lbl_ivl_status=_Label())
    ctrl = AnalysisController(app)
    ctrl._analysis_window_label = lambda: ("", "")
    ctrl._freq_status = lambda *a: ("", "")
    ctrl._nonlin_status = lambda *a: ("", "")
    return app, ctrl


def test_restore_and_rebuild_rederive_the_measured_status():
    from ecg.core.wave_template import WaveTemplate
    from ecg.ui.theme import GREEN, RED
    wt = WaveTemplate()
    df = _frame(1748, 16)
    df.attrs["wave_template"] = wt.to_dict()
    app, ctrl = _status_app(df)
    ctrl.refresh_result_labels()
    assert app.lbl_ivl_status.text == (
        f"  16/1748 beats measured (1%)  ·  template: {wt.provenance()}"
        "  — 1732 outside bounds: try Permissive Bounds or adjust Landmarks")
    assert app.lbl_ivl_status.color == RED
    app, ctrl = _status_app(_frame(20, 20), wave_template=wt)     # no record in attrs
    ctrl.refresh_result_labels()
    assert app.lbl_ivl_status.text == (
        f"  20/20 beats measured (100%)  ·  template: {wt.provenance()}")
    assert app.lbl_ivl_status.color == GREEN


def test_status_names_the_template_the_intervals_used_not_the_current_one():
    from ecg.core.wave_template import WaveTemplate
    used, current = WaveTemplate(), WaveTemplate()
    current.source = "edited"
    df = _frame(20, 20)
    df.attrs["wave_template"] = used.to_dict()
    app, ctrl = _status_app(df, wave_template=current)
    ctrl.refresh_result_labels()
    assert used.provenance() in app.lbl_ivl_status.text
    assert current.provenance() not in app.lbl_ivl_status.text


def test_status_without_measured_intervals_points_at_the_measure():
    app, ctrl = _status_app(pd.DataFrame({"RR_ms": [100.0, 110.0]}))
    ctrl.refresh_result_labels()
    assert "Core done" in app.lbl_ivl_status.text


# ── AV-delay check after a later measure ──────────────────────────────────

_FS = 1000.0
_PEAKS = np.arange(1, 21) * 100


def _ivl_for_peaks(pr=40.0):
    return pd.DataFrame({"R_peak_s": _PEAKS / _FS, "PR_ms": np.full(len(_PEAKS), pr)})


def _classified(ivl=None):
    pr_ms, note = AnalysisController.av_delay_input(ivl, _PEAKS, _FS)
    run = {"baseline": _BL, "av_note": note, "av_on": pr_ms is not None, "pr_ms": pr_ms,
           "peaks": _PEAKS, "fs": _FS}
    app = SimpleNamespace(analysis=AnalysisState(arrhythmia_run=run, premature_summary={"n": 1}))
    app.analysis_ctrl = AnalysisController(app)
    app.analysis_ctrl.arrhythmia_is_current = lambda: True
    return app, run


def _events_status(app):
    from ecg.ui.freshness import FreshnessService
    return FreshnessService(app)._status_events()


def test_measuring_after_classifying_puts_events_out_of_date():
    app, run = _classified()
    assert run["av_note"] == "AV-delay check skipped — measure intervals on the current window"
    assert _events_status(app).state == UP_TO_DATE
    app.analysis_ctrl.note_intervals_measured(_ivl_for_peaks())
    status = _events_status(app)
    assert status.state == OUT_OF_DATE
    assert status.reasons == ("intervals measured since the classification",)
    readout = AnalysisController.events_readout_text(run)
    assert "measure intervals on" not in readout
    assert readout.endswith("AV-delay check skipped — intervals measured since: classify again")
    assert run["pr_ms"] is None        # what the events ran on stays on record


def test_a_sparse_measure_also_puts_events_out_of_date():
    app, run = _classified()
    sparse = _ivl_for_peaks()
    sparse.loc[sparse.index[3:], "PR_ms"] = np.nan
    app.analysis_ctrl.note_intervals_measured(sparse)
    assert _events_status(app).state == OUT_OF_DATE


def test_measure_with_the_same_pr_leaves_events_up_to_date():
    ivl = _ivl_for_peaks()
    app, run = _classified(ivl)
    assert run["av_on"]
    app.analysis_ctrl.note_intervals_measured(ivl.copy())
    assert _events_status(app).state == UP_TO_DATE
    assert AnalysisController.events_readout_text(run).endswith("AV-delay check on")
    # Another PR: the check would run on other values.
    app.analysis_ctrl.note_intervals_measured(_ivl_for_peaks(pr=45.0))
    assert _events_status(app).state == OUT_OF_DATE
    assert "AV-delay check on — intervals measured since" in (
        AnalysisController.events_readout_text(run))


def test_measure_before_any_classification_changes_nothing():
    app = SimpleNamespace(analysis=AnalysisState())
    AnalysisController(app).note_intervals_measured(_ivl_for_peaks())
    assert app.analysis.arrhythmia_run is None
