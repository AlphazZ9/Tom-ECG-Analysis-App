# -*- coding: utf-8 -*-
"""
ecg.ui.tab_keys
---------------
The main tab names, in one place. These strings are the keys of app.tabs
(.add/.tab/.get/.set) and are also what a saved session stores as its current
tab, so a rename must extend session_controller._TAB_KEY_MAP.
"""
from __future__ import annotations

TAB_DETECTION     = "📈 Detection"
TAB_HRV           = "💓 HRV"
TAB_INTERVALS     = "📏 Intervals"
TAB_BEAT_TEMPLATE = "〰 Beat Template"
TAB_EVENTS        = "⚠ Abnormal Events"
TAB_SUMMARY       = "📋 Summary"

# Left-to-right order in the tab bar.
TAB_ORDER = (TAB_DETECTION, TAB_HRV, TAB_INTERVALS,
             TAB_BEAT_TEMPLATE, TAB_EVENTS, TAB_SUMMARY)
