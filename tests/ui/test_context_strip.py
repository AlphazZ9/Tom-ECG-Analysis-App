# -*- coding: utf-8 -*-
"""The context strip states what the numbers rest on, from the table's rows and
the state of the analysis; a tab's state combines those of its modules."""
from __future__ import annotations

import datetime as dt
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pytest

from ecg.core.metric_table import build_metric_table
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.context_strip import (
    TONE_WARN, ContextStrip, ContextStripMixin, StripField, artifact_field, context_fields, exclusion_field,
    tab_states, times_text, window_text,
)
from ecg.ui.freshness import NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE
from ecg.ui.tab_keys import TAB_DETECTION, TAB_EVENTS, TAB_HRV, TAB_MORPHOLOGY
from ecg.ui.theme import THEME


def _results(**hr):
    base = dict(mean=350.0, min=337.0, max=365.0, std=8.0, n=1749, n_valid=1748, n_excluded=0,
                n_too_short=0, n_too_long=0, n_spanning_removed=0, excluded_pct=0.0,
                excluded_time_s=0.0, rr_min_ms=66.7, rr_max_ms=333.3)
    base.update(hr)
    return {"hr": base, "rr_ms": np.full(1748, 170.0), "rr_valid": np.ones(1748, dtype=bool)}


def _fields(r, **kw):
    table = build_metric_table(r, asymmetry=None) if r is not None else None
    args = dict(artifact_table=build_metric_table(None), has_signal=True, n_peaks=1700,
                window=(0.0, 0.0), signal_end_s=300.0, current_rr=(50.0, 400.0),
                qtc_formula="mitchell")
    args.update(kw)
    return {f.key: f for f in context_fields(r, table, **args)}


def test_with_results_the_strip_says_what_they_were_computed_on():
    f = _fields(_results())
    assert f["head"].text == "Analysed"
    assert f["window"].text == "Window 0–300 s"
    assert f["beats"].text == "1749 peaks"          # the results', not the current count
    assert f["rr"].text == "RR bounds 66.7–333.3 ms"  # recorded, not the edited entries
    assert f["excluded"].text == "0 RR excluded"
    assert f["artifact"].text == "Artifact correction: not applied"
    assert f["qtc"].text.startswith("QTc: Mitchell")


def test_without_results_it_says_what_the_next_analyze_would_use():
    f = _fields(None, window=(10.0, 70.5))
    assert f["head"].text == "Not analysed"
    assert f["beats"].text == "1700 peaks"
    assert f["rr"].text == "RR bounds 50.0–400.0 ms"
    assert f["window"].text == "Window 10–70.5 s"
    assert "—" in f["excluded"].text


def test_no_recording_shows_dashes():
    f = _fields(None, has_signal=False, n_peaks=None, signal_end_s=None)
    assert f["head"].text == "No recording"
    assert f["window"].text == "Window —"
    assert f["beats"].text == "— peaks"


def test_exclusion_breakdown_lists_only_what_happened():
    t = build_metric_table(_results(n_excluded=3, n_too_long=2, n_spanning_removed=1,
                                    excluded_pct=0.17, n_valid=1745), asymmetry=None)
    field = exclusion_field(t)
    assert field.tone == TONE_WARN
    assert field.text == "3 RR excluded (2 too long · 1 spanning a removed beat · 0.17 %)"


def test_artifact_field_reads_the_table():
    rep = dict(n_in=100, n_out=97, n_duplicate=1, n_nonphysio=1, n_ectopic=1)
    assert artifact_field(build_metric_table(None, artifact_report=rep)).text == \
        "Artifact correction: 3 beats removed"
    rep = dict(n_in=100, n_out=100)
    assert artifact_field(build_metric_table(None, artifact_report=rep)).text == \
        "Artifact correction: no beat removed"


def test_window_text_zero_means_signal_end():
    assert window_text((0.0, 0.0), 299.996) == "0–300 s"
    assert window_text((5.0, 20.0), 300.0) == "5–20 s"
    assert window_text((0.0, 0.0), None) == "—"


def test_times_text():
    now = dt.datetime(2026, 10, 30, 15, 0)
    assert times_text(None, None, now) == ""
    assert times_text("2026-10-30T14:32:10", None, now) == "Computed 14:32"
    assert times_text("2026-10-30T14:32:10", "2026-10-30T14:35:00", now) == \
        "Computed 14:32 · peaks edited 14:35"
    assert times_text("2026-10-29T14:32:10", None, now) == "Computed 2026-10-29 14:32"
    assert times_text("garbage", None, now) == ""


def _states(**over):
    from ecg.core.metric_table import MODULES
    return {**{m: NOT_COMPUTED for m in MODULES}, **over}


def test_tab_state_rule():
    out = tab_states(_states(), has_peaks=False)
    assert set(out.values()) == {NOT_COMPUTED}
    out = tab_states(_states(core=UP_TO_DATE, frequency=UP_TO_DATE), has_peaks=True)
    assert out[TAB_DETECTION] == UP_TO_DATE
    assert out[TAB_HRV] == UP_TO_DATE               # parts not computed do not stale it
    assert out[TAB_MORPHOLOGY] == NOT_COMPUTED
    out = tab_states(_states(core=OUT_OF_DATE, frequency=UP_TO_DATE, events=UP_TO_DATE), has_peaks=True)
    assert out[TAB_DETECTION] == UP_TO_DATE         # the peaks themselves are current
    assert out[TAB_HRV] == OUT_OF_DATE
    assert out[TAB_EVENTS] == UP_TO_DATE


def test_strip_wraps_instead_of_clipping():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.geometry("300x100")
    strip = ContextStrip(root)
    strip.pack(fill="x")
    strip.show([StripField(str(i), "a field of some width") for i in range(6)], "Computed 14:32")
    root.update()
    assert strip.text().count("|") == 5
    assert strip.times() == "Computed 14:32"
    ys = {int(strip._labels[str(i)].place_info()["y"]) for i in range(6)}
    assert len(ys) > 1
    assert int(strip._flow.cget("height")) == max(ys) + 22
    root.destroy()


def test_strip_rows_hold_the_largest_font_scale(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    monkeypatch.setattr(THEME, "font_scale", 1.5)
    root.geometry("300x200")
    strip = ContextStrip(root)
    strip.pack(fill="x")
    strip.show([StripField(str(i), "a field of some width") for i in range(4)], "")
    root.update()
    labels = [strip._labels[str(i)] for i in range(4)]
    assert all(lbl.winfo_height() >= lbl.winfo_reqheight() for lbl in labels)
    ys = sorted({lbl.winfo_y() for lbl in labels})
    assert len(ys) > 1 and ys[1] - ys[0] >= labels[0].winfo_reqheight()
    root.destroy()


def _artifact_app(peaks):
    """Beats 1000 and 2000 removed by the artifact correction."""
    report = dict(n_in=len(peaks) + 2, n_out=len(peaks), n_duplicate=0, n_nonphysio=0,
                  n_ectopic=2, removed_samples=[1000, 2000], removed_types=["ectopic"] * 2,
                  break_samples=[1000, 2000])
    app = SimpleNamespace(
        analysis=SimpleNamespace(results=None, artifact_report=report),
        detection=SimpleNamespace(rpeaks_ok=peaks))
    ctrl = AnalysisController(app)
    ctrl.break_tol_samples = lambda: 1
    app.analysis_ctrl = ctrl
    return app


def test_artifact_field_follows_beats_that_come_back():
    """Re-detection gives a removed beat back: the strip stops counting it, and
    says what a saved session would (current_artifact_report)."""
    app = _artifact_app(np.array([500, 1500, 2500]))
    art = ContextStripMixin._context_table(app)[1]
    assert artifact_field(art).text == "Artifact correction: 2 beats removed"
    assert ContextStripMixin._context_table(app)[1] is art      # cached on the tick

    app.detection.rpeaks_ok = np.array([500, 1000, 1500, 2500])
    assert artifact_field(ContextStripMixin._context_table(app)[1]).text == \
        "Artifact correction: 1 beat removed"
    app.detection.rpeaks_ok = np.array([500, 1000, 1500, 2000, 2500])
    assert artifact_field(ContextStripMixin._context_table(app)[1]).text == \
        "Artifact correction: no beat removed"


def test_hint_opens_only_when_the_status_bar_cuts_the_text():
    shown = []
    tabs = SimpleNamespace(state_text=lambda n: "A: out of date — a reason",
                           show_hint=shown.append)
    app = SimpleNamespace(tabs=tabs, _fit_status_text=lambda text: text)
    ContextStripMixin._show_tab_hint(app, "A")
    assert shown == []
    app._fit_status_text = lambda text: text[:5] + "…"
    ContextStripMixin._show_tab_hint(app, "A")
    assert shown == ["A"]
