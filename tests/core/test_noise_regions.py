"""Noisy regions marked by hand: their RR intervals are left out of HR and
HRV (a missed beat there would read as a slower heart), as an interval
spanning a removed beat is."""
from __future__ import annotations

import numpy as np
import pytest

from ecg.core.analysis import noise_breaks, rr_interval_series


def test_every_interval_touching_a_region_is_left_out():
    fs = 1000.0
    peaks = np.arange(0, 20_000, 100)                    # 600 bpm, 0.1 s RR
    # 5–6 s noisy, with a beat missed in it (a 200 ms interval).
    peaks = peaks[peaks != 5_500]
    br = noise_breaks(peaks, fs, [{"t_start": 5.0, "t_end": 6.0, "note": ""}])
    ser = rr_interval_series(peaks, fs, 50, 300, breaks=br)
    t_end = ser["t_s"]
    left_out = ~ser["valid"]
    # Intervals ending after 5.0 s and starting before 6.0 s.
    starts = t_end - ser["rr_ms"] / 1000
    expect = (t_end > 5.0) & (starts < 6.0)
    assert np.array_equal(left_out, expect)
    assert ser["rri"].max() == 100.0                     # the long one is gone
    assert ser["n_spanning_removed"] == int(expect.sum())


def test_nothing_when_no_region():
    p = np.arange(0, 5000, 100)
    assert noise_breaks(p, 1000.0, []).size == 0
    assert noise_breaks(p, 1000.0, None).size == 0
    assert noise_breaks(p, 1000.0, [{"t_start": 9.0, "t_end": 10.0}]).size == 0
    assert noise_breaks(p, 1000.0, [{"t_start": "x"}, {"t_start": 2, "t_end": 1}]).size == 0


def test_the_session_brings_pacing_periods_back_as_annotations():
    from ecg.ui.session_controller import pacing_as_annotations
    got = pacing_as_annotations([{"t_start": 10, "t_end": 70, "note": "20 Hz"},
                                 {"t_start": 100, "t_end": 110, "note": ""}, {"bad": 1}])
    assert [a["label"] for a in got] == ["20 Hz", "Pacing"]
    assert got[0]["t_start"] == 10.0 and got[0]["t_end"] == 70.0


def test_a_drag_in_mark_noise_marks_a_region_and_a_right_click_removes_it():
    from types import SimpleNamespace as NS
    from ecg.ui.detection_controller import DetectionController
    from ecg.ui.state import AnalysisState, DetectionState
    changed, status = [], []
    app = NS(detection=DetectionState(), analysis=AnalysisState(),
             signal=NS(fs=1000.0, filtered=np.zeros(20_000)), ui=NS(nav_pos=0.0),
             _EDIT_UNDO_LIMIT=50, _noise_regions_changed=lambda: changed.append(1),
             _set_status=lambda *a, **k: status.append(a[0]),
             _draw_detail=lambda *a, **k: None)
    ctrl = DetectionController(app)
    app.detection.edit_mode = True
    ctrl.toggle_noise_mode()
    assert app.detection.edit_noise_mode
    ev = lambda t, b=1: NS(xdata=t, button=b, inaxes=None)
    ctrl.on_detail_click(ev(7.5))
    ctrl.on_detail_motion(ev(6.0))
    ctrl.on_detail_release(ev(6.0))
    assert app.analysis.noise_regions == [{"t_start": 6.0, "t_end": 7.5, "note": ""}]
    ctrl.on_detail_click(ev(3.0))
    ctrl.on_detail_release(ev(3.01))                    # a click: nothing
    assert len(app.analysis.noise_regions) == 1
    ctrl.on_detail_click(ev(7.0, 3))                    # right-click inside: removed
    assert app.analysis.noise_regions == [] and len(changed) == 2


def test_the_strip_names_the_noisy_regions():
    from types import SimpleNamespace as NS
    from ecg.ui.context_strip import exclusion_field
    vals = {"rr.n_excluded": 4, "rr.n_spanning_removed": 4, "rr.excluded_pct": 1.0}
    t = NS(has=lambda k: k in vals, value=lambda k, d=None: vals.get(k, d))
    assert "spanning a removed beat" in exclusion_field(t).text
    assert "4 in a noisy region or across a removed beat" in exclusion_field(t, True).text


def test_ctrl_z_undoes_a_noisy_region_without_touching_the_peaks():
    from types import SimpleNamespace as NS
    from ecg.ui.detection_controller import DetectionController
    from ecg.ui.state import AnalysisState, DetectionState
    changed = []
    app = NS(detection=DetectionState(), analysis=AnalysisState(),
             signal=NS(fs=1000.0, filtered=np.zeros(20_000)), ui=NS(nav_pos=0.0),
             _EDIT_UNDO_LIMIT=50, _noise_regions_changed=lambda: changed.append(1),
             _set_status=lambda *a, **k: None, _draw_detail=lambda *a, **k: None)
    ctrl = DetectionController(app)
    ctrl.apply_edit_state = lambda: pytest.fail("peaks re-detected")
    app.detection.edit_mode = True
    ctrl.toggle_noise_mode()
    ev = lambda t, b=1: NS(xdata=t, button=b, inaxes=None)
    ctrl.on_detail_click(ev(2.0)); ctrl.on_detail_release(ev(4.0))
    ctrl.on_detail_click(ev(8.0)); ctrl.on_detail_release(ev(9.0))
    assert len(app.analysis.noise_regions) == 2
    ctrl.undo_edit()
    assert app.analysis.noise_regions == [{"t_start": 2.0, "t_end": 4.0, "note": ""}]
    ctrl.undo_edit()
    assert app.analysis.noise_regions == []
    ctrl.redo_edit()
    assert len(app.analysis.noise_regions) == 1
    ctrl.on_detail_click(ev(3.0, 3))                    # right-click removes: undoable too
    assert app.analysis.noise_regions == []
    ctrl.undo_edit()
    assert len(app.analysis.noise_regions) == 1


def test_the_window_is_maximised_on_linux_too():
    import tkinter as tk
    from ecg.ui.app import ECGApp
    calls = []

    class Fake:
        def state(self, s):
            calls.append(("state", s))
            raise tk.TclError('bad argument "zoomed"')

        def attributes(self, *a):
            calls.append(("attributes",) + a)
    ECGApp._maximise(Fake())
    assert calls == [("state", "zoomed"), ("attributes", "-zoomed", True)]
