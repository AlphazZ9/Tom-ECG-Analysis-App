# -*- coding: utf-8 -*-
"""
The last gaps against the approved mockup: running modules on the report lines,
annotations on Epochs and Sliding, landmark labels and their drag, the step the
user is on, the compact event table and the RR strip.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecg.core.metric_table import REPORT_INTERVALS, REPORT_RHYTHM
from ecg.ui.analysis_controller import draw_windowed_panels
from ecg.ui.event_locked_view import event_rows
from ecg.ui.freshness import COMPUTING, NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE, ModuleFreshness
from ecg.ui.plot_controller import landmark_label_levels, overview_rr
from ecg.ui.report_view import report_states
from ecg.ui.tab_morphology import morphology_steps, step_tip
from ecg.ui.wave_editor import dragged_landmark, hit_landmark


def test_a_running_module_makes_its_report_line_computing():
    states = {m: UP_TO_DATE for m in ("core", "frequency", "nonlinear", "intervals")}
    lines = report_states(states, running=("frequency",))
    assert lines[REPORT_RHYTHM] == COMPUTING
    assert lines[REPORT_INTERVALS] == UP_TO_DATE
    assert report_states(states)[REPORT_RHYTHM] == UP_TO_DATE


def test_windowed_panels_draw_annotations_on_every_panel_without_stretching():
    df = pd.DataFrame({"HR": [1.0, 2.0, 3.0], "SDNN": [3.0, 2.0, 1.0]})
    t = np.array([10.0, 20.0, 30.0])
    panels = [("HR", "HR", "red", "HR"), ("SDNN", "SDNN", "blue", "SDNN")]
    anns = [{"t_start": 15.0, "t_end": 16.0, "label": "Stimulus", "color": "#6A1B9A"},
            {"t_start": 900.0, "t_end": 901.0, "label": "Far", "color": "#6A1B9A"}]
    fig = Figure()
    draw_windowed_panels(fig, t, df, panels, np.zeros(3, bool), 15.0, anns)
    for ax in fig.axes:
        assert any(list(line.get_xdata()) == [15.0, 15.0] for line in ax.lines)
        assert ax.get_xlim()[1] < 100
    assert [x.get_text() for x in fig.axes[0].texts] == ["Stimulus"]


def test_windowed_panels_values_do_not_depend_on_annotations():
    df = pd.DataFrame({"HR": [1.0, 2.0]})
    fig = Figure()
    draw_windowed_panels(fig, np.array([1.0, 2.0]), df, [("HR", "HR", "red", "HR")],
                         np.zeros(2, bool), 2.0)
    assert list(fig.axes[0].lines[0].get_ydata()) == [1.0, 2.0]


def test_landmark_labels_close_together_take_separate_rows():
    centers = [-45.0, -8.0, -5.0, 4.0, 6.0, 9.0, 30.0]
    names = ["P peak", "QRS onset", "Q peak", "S peak", "J point", "J peak", "T peak"]
    levels = landmark_label_levels(centers, names, ms_per_px=0.3, char_px=6.0)
    half = [0.5 * (len(n) + 2) * 6.0 * 0.3 for n in names]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if levels[i] == levels[j]:
                assert abs(centers[i] - centers[j]) >= half[i] + half[j]
    assert levels[0] == 0 and max(levels) >= 1


def test_drag_rules_match_the_dialog():
    lm = {"T_peak": (30.0, 10.0), "S_peak": (5.0, 3.0)}
    assert hit_landmark(lm, ["T_peak", "S_peak"], 30.5, 5.0) == ("T_peak", "center")
    assert hit_landmark(lm, ["T_peak", "S_peak"], 40.2, 5.0) == ("T_peak", "hi")
    assert hit_landmark(lm, ["T_peak", "S_peak"], 70.0, 5.0)[0] is None
    assert dragged_landmark(30.0, 10.0, "center", 33.26) == (33.3, 10.0)
    assert dragged_landmark(30.0, 10.0, "lo", 25.0) == (30.0, 5.0)
    assert dragged_landmark(30.0, 10.0, "hi", 30.5) == (30.0, 2.0)


def _fresh(state):
    return ModuleFreshness(state=state, reasons=())


def test_the_current_step_is_the_first_with_something_to_do():
    on = [s.number for s in morphology_steps(_fresh(UP_TO_DATE), _fresh(NOT_COMPUTED), 100)
          if s.current]
    assert on == [2]
    step2 = morphology_steps(_fresh(UP_TO_DATE), _fresh(OUT_OF_DATE), 100)[1]
    assert step2.current and "check the landmarks" in step_tip(step2)
    assert not any(s.current for s in morphology_steps(_fresh(UP_TO_DATE), _fresh(UP_TO_DATE), 100))
    assert morphology_steps(_fresh(NOT_COMPUTED), _fresh(NOT_COMPUTED), None)[0].current


def test_event_rows_read_the_result_and_dash_what_is_invalid():
    table = pd.DataFrame([
        {"onset_s": 100.0, "valid": True, "isolated": True, "coverage_pre": 1.0, "coverage_post": 1.0,
         "hr_valid": True, "hr_pre": 350.0, "hr_post": 347.5, "hr_delta": -2.5,
         "csi_valid": True, "csi_pre": 1.5, "csi_post": 2.25, "csi_delta": 0.75,
         "cpi_valid": True, "cpi_pre": 3.0, "cpi_post": 2.0, "cpi_delta": -1.0},
        {"onset_s": 200.0, "valid": False, "isolated": True, "coverage_pre": 0.62, "coverage_post": 1.0,
         "hr_valid": True, "hr_pre": 350.0, "hr_post": 360.0, "hr_delta": 10.0,
         "csi_valid": False, "csi_pre": np.nan, "csi_post": np.nan, "csi_delta": np.nan,
         "cpi_valid": False, "cpi_pre": np.nan, "cpi_post": np.nan, "cpi_delta": np.nan},
        {"onset_s": 300.0, "valid": False, "isolated": False, "coverage_pre": 1.0, "coverage_post": 1.0,
         "hr_valid": False, "hr_pre": np.nan, "hr_post": np.nan, "hr_delta": np.nan,
         "csi_valid": False, "csi_pre": np.nan, "csi_post": np.nan, "csi_delta": np.nan,
         "cpi_valid": False, "cpi_pre": np.nan, "cpi_post": np.nan, "cpi_delta": np.nan},
    ])
    rows = event_rows(SimpleNamespace(table=table, keys=("csi", "cpi", "hr")))
    assert (rows[0].onset, rows[0].hr, rows[0].csi, rows[0].cpi, rows[0].valid) == (
        "100.0 s", "-2.5 bpm", "1.50 → 2.25", "3.00 → 2.00", "yes")
    assert rows[1].csi == "—" and rows[1].valid == "no · 62 % coverage" and not rows[1].is_valid
    assert rows[2].hr == "—" and rows[2].valid == "no · annotation in window"


def test_overview_rr_is_the_interval_between_peaks():
    time = np.arange(1000) / 100.0
    peaks = np.array([0, 20, 40, 65, 85])
    t, rr, lo, hi = overview_rr(time, peaks, 100.0)
    assert list(t) == [0.2, 0.4, 0.65, 0.85]
    assert list(rr) == [200.0, 200.0, 250.0, 200.0]
    assert lo < 200.0 and hi > 240.0
    assert overview_rr(time, None, 100.0) is None
    assert overview_rr(time, np.array([0, 20]), 100.0) is None
    many = np.arange(0, 1000, 2)
    assert len(overview_rr(time, many, 100.0, max_points=50)[0]) == 50
