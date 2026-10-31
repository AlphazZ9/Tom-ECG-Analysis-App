# -*- coding: utf-8 -*-
"""
ecg.ui.tab_keys
---------------
The main tab names, in one place. These strings are the keys of app.tabs
(.add/.tab/.get/.set) and are also what a saved session stores as its current
tab, so a rename must extend session_controller._TAB_KEY_MAP. The state glyph
on the tab bar took the place of the emoji these names used to carry.
"""
from __future__ import annotations

TAB_DETECTION     = "Signal & Detection"
TAB_HRV           = "Rhythm & HRV"
TAB_TIME_COURSE   = "Time course"
TAB_MORPHOLOGY    = "Morphology & Intervals"
TAB_EVENTS        = "Events"
TAB_REPORT        = "Report"

# Left-to-right order in the tab bar.
TAB_ORDER = (TAB_DETECTION, TAB_HRV, TAB_TIME_COURSE, TAB_MORPHOLOGY,
             TAB_EVENTS, TAB_REPORT)

# The modules (ecg.ui.freshness) behind each tab's state. Signal & Detection is
# not listed: its state is that of the peaks (ecg.ui.context_strip.tab_states).
# Morphology & Intervals is the state of its measure: Analyze gives the mean beat,
# not the intervals, and its step chips say the state of each step.
TAB_MODULES = {
    TAB_HRV:           ("core", "frequency", "nonlinear"),
    TAB_TIME_COURSE:   ("epochs", "rolling", "autonomic", "event_locked"),
    TAB_MORPHOLOGY:    ("intervals",),
    TAB_EVENTS:        ("events",),
    TAB_REPORT:        ("core",),
}

# The modes of the Time course tab, and the module (ecg.ui.freshness) behind each.
MODE_EPOCHS       = "Epochs"
MODE_SLIDING      = "Sliding"
MODE_AUTONOMIC    = "Autonomic"
MODE_EVENT_LOCKED = "Event-locked"
TIME_COURSE_MODES = (MODE_EPOCHS, MODE_SLIDING, MODE_AUTONOMIC, MODE_EVENT_LOCKED)
MODE_MODULES = {
    MODE_EPOCHS:       "epochs",
    MODE_SLIDING:      "rolling",
    MODE_AUTONOMIC:    "autonomic",
    MODE_EVENT_LOCKED: "event_locked",
}

# Values of the former ui.hrv_current_view and ui.autonomic_subview. The other
# former HRV views (Overview, RR / HR...) live on Rhythm & HRV: no mode.
_FORMER_VIEW_MODES = {"Epochs": MODE_EPOCHS, "Rolling": MODE_SLIDING, "Autonomic": MODE_AUTONOMIC}
_FORMER_SUBVIEW_MODES = {"Time series": MODE_AUTONOMIC, "Event-locked": MODE_EVENT_LOCKED}


def time_course_mode(mode=None, hrv_view=None, autonomic_subview=None):
    """The mode of a stored *mode*, or else of the former pair of values (the
    Autonomic view's own toggle counts only when that view was the one shown,
    or nothing says which was). None when none of them names a mode."""
    if mode in TIME_COURSE_MODES:
        return mode
    if mode in _FORMER_VIEW_MODES:
        return _FORMER_VIEW_MODES[mode]
    if mode in _FORMER_SUBVIEW_MODES:
        return _FORMER_SUBVIEW_MODES[mode]
    if hrv_view in ("Autonomic", None) and autonomic_subview in _FORMER_SUBVIEW_MODES:
        return _FORMER_SUBVIEW_MODES[autonomic_subview]
    return _FORMER_VIEW_MODES.get(hrv_view)
