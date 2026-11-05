# -*- coding: utf-8 -*-
"""The analysis window entries stand where the strip states the window: they
must say the applied window after it is reset (new file, Detect Peaks), and
the strip says when typed entries are not applied. With Method = ML Detector,
the quality panel's ML section leaves the correlation plot at least the height
it has without it."""
from __future__ import annotations

import ast
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import pytest

import ecg
from ecg.ui.app_helpers import HelpersMixin
from ecg.ui.context_strip import ContextStripMixin, unapplied_window_field
from ecg.ui.tab_detection import QUALITY_PANEL_W, QUALITY_PLOT_MIN_H, TabDetectionMixin

UI_DIR = Path(ecg.__file__).parent / "ui"


class _Entry:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def get(self) -> str:
        return self.text

    def delete(self, *_a) -> None:
        self.text = ""

    def insert(self, _i, val: str) -> None:
        self.text = val + self.text


def test_entries_are_rewritten_with_the_applied_window():
    app = SimpleNamespace(ent_analysis_t0=_Entry("5"), ent_analysis_t1=_Entry("100"),
                          analysis=SimpleNamespace(t_start=0.0, t_end=0.0))
    ContextStripMixin._show_window_in_entries(app)
    assert (app.ent_analysis_t0.text, app.ent_analysis_t1.text) == ("", "")
    app.analysis.t_start, app.analysis.t_end = 12.5, 60.0
    ContextStripMixin._show_window_in_entries(app)
    assert (app.ent_analysis_t0.text, app.ent_analysis_t1.text) == ("12.5", "60")


def test_every_reset_of_the_window_also_resets_the_entries():
    # Detect Peaks, a raw load and a new file set the window back to the full
    # signal: each must also empty the entries (the reported stale strip).
    src = (UI_DIR / "signal_controller.py").read_text(encoding="utf-8")
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        resets = any(isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)
                     and n.targets[0].attr == "t_start" and isinstance(n.value, ast.Constant)
                     and n.value.value == 0.0 for n in ast.walk(fn))
        if not resets:
            continue
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert "_show_window_in_entries" in attrs or "ent_analysis_t0" in attrs, fn.name


def test_typed_window_not_applied_is_said():
    assert unapplied_window_field(("", ""), (0.0, 0.0)) is None
    assert unapplied_window_field(("5.0", " 100 "), (5.0, 100.0)) is None
    f = unapplied_window_field(("5", "100"), (0.0, 0.0))
    assert f is not None and "Apply" in f.text and f.key == "window_unapplied"
    assert unapplied_window_field(("abc", ""), (0.0, 0.0)) is not None
    assert unapplied_window_field(("", ""), None) is None


class _Panel(TabDetectionMixin, HelpersMixin):
    """Just what _build_quality_panel needs."""

    def __init__(self) -> None:
        self._slots = {}

    def after(self, *_a, **_k) -> None:
        pass

    def _bind_hover_tip(self, *_a, **_k) -> None:
        pass

    def __getattr__(self, name):       # command callbacks
        if name.startswith(("_open", "_check", "_on_", "_save", "refresh")):
            return lambda *a, **k: None
        raise AttributeError(name)


def test_ml_section_keeps_the_correlation_plot_readable():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    try:
        # The panel's height on a 1366x768 screen (Signal & Detection tab).
        host = tk.Frame(root, width=QUALITY_PANEL_W, height=440)
        host.pack()
        host.grid_propagate(False)
        host.grid_rowconfigure(0, weight=1)
        p = _Panel()
        p._build_quality_panel(host)
        root.update()
        sg_h = p._slots["quality_time"].frame.winfo_height()
        p._show_ml_section(True)
        root.update()
        ml_h = p._slots["quality_time"].frame.winfo_height()
        assert p._ml_frame.winfo_ismapped()
        assert ml_h >= sg_h and ml_h >= 60, (sg_h, ml_h)
        # Every ML control is still on screen, under the plot.
        bottom = host.winfo_rooty() + host.winfo_height()
        for w in (p.sw_verified_training, p.btn_save_for_training, p.btn_train_ml):
            assert w.winfo_ismapped() and w.winfo_rooty() + w.winfo_height() <= bottom
        p._show_ml_section(False)
        root.update()
        assert p._slots["quality_time"].frame.winfo_height() == sg_h
    finally:
        root.destroy()



@pytest.mark.parametrize("height, curve_shown, hist_shown",
                         [(300, False, False), (380, True, False), (900, True, True)])
def test_short_panel_drops_plots_that_cannot_be_read(height, curve_shown, hist_shown):
    """On a 768-px screen both plots were squeezed until their ticks
    overlapped: the histogram steps aside first, then the curve (its title
    says why), and both come back when the window is tall enough."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    try:
        host = tk.Frame(root, width=QUALITY_PANEL_W, height=height)
        host.pack()
        host.grid_propagate(False)
        host.grid_rowconfigure(0, weight=1)
        p = _Panel()
        p._build_quality_panel(host)
        for _ in range(3):
            root.update()
        head, curve = p._quality_time_rows
        hist = p._quality_hist_rows[1]
        assert bool(curve.winfo_ismapped()) is curve_shown
        assert bool(hist.winfo_ismapped()) is hist_shown
        assert ("taller window" in head.cget("text")) is not curve_shown
        for box, shown in ((curve, curve_shown), (hist, hist_shown)):
            if shown:
                assert box.winfo_height() >= QUALITY_PLOT_MIN_H
        # ML mode never shows the histogram; leaving it follows the height again.
        p._show_ml_section(True)
        root.update()
        assert not hist.winfo_ismapped()
        p._show_ml_section(False)
        for _ in range(3):
            root.update()
        assert bool(hist.winfo_ismapped()) is hist_shown
        assert bool(curve.winfo_ismapped()) is curve_shown
    finally:
        root.destroy()
