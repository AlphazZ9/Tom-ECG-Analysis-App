# -*- coding: utf-8 -*-
"""TabBar keeps the CTkTabview API the app relies on, shows one state per tab
as a glyph (never by colour alone), and old sessions reopen on the right tab."""
from __future__ import annotations

import tkinter as tk

import pytest

from ecg.core.metric_table import MODULES
from ecg.ui.freshness import NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE
from ecg.ui.session_controller import _TAB_KEY_MAP
from ecg.ui.tab_keys import (
    TAB_DETECTION, TAB_EVENTS, TAB_HRV, TAB_MODULES, TAB_MORPHOLOGY,
    TAB_ORDER, TAB_REPORT,
)
from ecg.ui.tabbar import STATE_GLYPHS, TabBar


@pytest.fixture
def bar():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.withdraw()
    b = TabBar(root)
    b.pack(fill="both", expand=True)
    yield b
    root.destroy()


def test_first_tab_added_is_selected_and_set_switches(bar):
    first, second = bar.add("A"), bar.add("B")
    assert bar.get() == "A"
    assert bar.tab("A") is first and bar.tab("B") is second
    assert first.grid_info() and not second.grid_info()
    bar.set("B")
    assert bar.get() == "B"
    assert second.grid_info() and not first.grid_info()


def test_unknown_tab_raises_like_ctktabview(bar):
    bar.add("A")
    with pytest.raises(ValueError):
        bar.set("nope")
    with pytest.raises(ValueError):
        bar.tab("nope")
    assert bar.get() == "A"


def test_command_fires_on_a_real_change_only():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.withdraw()
    calls = []
    b = TabBar(root, command=lambda: calls.append(b.get()))
    b.add("A"); b.add("B")
    b.set("B"); b.set("B")
    assert calls == ["A", "B"]
    root.destroy()


def test_state_glyphs_differ_by_shape_not_only_colour():
    assert len({STATE_GLYPHS[s] for s in (UP_TO_DATE, OUT_OF_DATE, NOT_COMPUTED)}) == 3


def test_a_tab_starts_not_computed_and_shows_its_state(bar):
    bar.add("A")
    assert bar.state("A") == NOT_COMPUTED
    bar.set_state("A", OUT_OF_DATE, ("core: peaks changed",))
    assert bar.state("A") == OUT_OF_DATE
    assert bar._heads["A"]["glyph"].cget("text") == STATE_GLYPHS[OUT_OF_DATE]
    assert bar.state_text("A") == "A: out of date — core: peaks changed"
    bar.set_state("A", UP_TO_DATE)
    assert bar.state_text("A") == "A: up to date"


def test_hint_lists_every_reason_and_goes_away(bar):
    bar.add("A")
    why = ("core: peaks changed", "frequency: peaks changed", "epochs: peaks changed")
    bar.set_state("A", OUT_OF_DATE, why)
    root = bar.master
    root.deiconify()
    root.geometry("500x300")
    root.update()
    bar.show_hint("A")
    root.update()
    assert bar._hint.cget("text").splitlines() == ["A: out of date", *why]
    assert bar._hint.winfo_ismapped()
    assert bar._hint.winfo_y() >= bar._bar.winfo_height()
    bar.hide_hint()
    root.update()
    assert not bar._hint.winfo_ismapped()


@pytest.mark.parametrize("saved, tab", [
    ("Detection", TAB_DETECTION), ("📈 Detection", TAB_DETECTION),
    ("HRV", TAB_HRV), ("💓 HRV", TAB_HRV),
    ("Intervals", TAB_MORPHOLOGY), ("📏 Intervals", TAB_MORPHOLOGY),
    ("Beat Template", TAB_MORPHOLOGY), ("〰 Beat Template", TAB_MORPHOLOGY),
    ("Arrhythmias", TAB_EVENTS), ("⚠ Arrhythmias", TAB_EVENTS), ("⚠ Abnormal Events", TAB_EVENTS),
    ("Summary", TAB_REPORT), ("📋 Summary", TAB_REPORT),
])
def test_old_session_tab_names_reopen_on_the_right_tab(saved, tab):
    assert _TAB_KEY_MAP[saved] == tab


def test_current_names_pass_through_the_session_map():
    for name in TAB_ORDER:
        assert _TAB_KEY_MAP.get(name, name) == name


def test_every_tab_but_detection_lists_known_modules():
    assert set(TAB_MODULES) == set(TAB_ORDER) - {TAB_DETECTION}
    for modules in TAB_MODULES.values():
        assert set(modules) <= set(MODULES)
