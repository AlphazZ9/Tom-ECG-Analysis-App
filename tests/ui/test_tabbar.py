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
    # Compare segments, computed on demand from the peaks, shares their state;
    # Group reads a group table, not the recording open.
    assert set(TAB_MODULES) == set(TAB_ORDER) - {TAB_DETECTION, "Compare segments", "Group"}
    for modules in TAB_MODULES.values():
        assert set(modules) <= set(MODULES)


# ── Clicks on macOS: one target per head, fixed geometry, release fallback ──
# A click had to be repeated on a Mac: dead underline/gaps, heads moving
# under the pointer, no redraw before the tab-change work, and presses Tk
# Aqua drops (after a live resize, stale pointer state).
from ecg.ui.freshness import COMPUTING  # noqa: E402
from ecg.ui.tab_keys import TAB_ORDER  # noqa: E402

@pytest.fixture
def shown():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.geometry("1400x300+0+0")
    # The pixel and pointer tests ask the screen which window is at a point:
    # keep this one above whatever else is open on the desktop.
    root.attributes("-topmost", True)
    root.lift()
    calls = []
    b = TabBar(root, command=lambda: calls.append(b.get()))
    b.pack(fill="both", expand=True)
    for n in TAB_ORDER:
        b.add(n)
    root.update()
    yield b, calls
    root.destroy()

def _geo(b):
    return [(b._heads[n]["head"].winfo_x(), b._heads[n]["head"].winfo_width()) for n in b.names()]

def test_heads_do_not_move_when_selection_or_state_changes(shown):
    b, _ = shown
    before = _geo(b)
    for n in b.names():
        b.set(n); b.update()
        assert _geo(b) == before, n
    for s in (UP_TO_DATE, OUT_OF_DATE, COMPUTING, NOT_COMPUTED):
        b.set_state(b.names()[1], s); b.update()
        assert _geo(b) == before, s

def test_every_pixel_from_first_to_last_head_selects_a_tab(shown):
    b, _ = shown
    root = b.winfo_toplevel()
    first, last = b._heads[b.names()[0]]["head"], b._heads[b.names()[-1]]["head"]
    y0 = b.bar.winfo_rooty()
    for x in range(first.winfo_rootx(), last.winfo_rootx() + last.winfo_width()):
        for y in range(y0, y0 + b.bar.winfo_height()):
            w = root.winfo_containing(x, y)
            assert any(t.startswith("TabBarHead") for t in w.bindtags()), (x, y, str(w))

def test_press_on_the_underline_selects(shown):
    b, calls = shown
    name = b.names()[2]
    b._heads[name]["line"]._canvas.event_generate("<ButtonPress-1>", x=1, y=1)
    b.update()
    assert b.get() == name and calls[-1] == name

def test_release_without_press_selects_the_head_under_the_pointer(shown):
    b, _ = shown
    name = b.names()[3]
    lbl = b._heads[name]["label"]._label
    x, y = lbl.winfo_rootx() + 3, lbl.winfo_rooty() + 3
    lbl.event_generate("<ButtonRelease-1>", x=3, y=3, rootx=x, rooty=y)
    b.update()
    assert b.get() == name

def test_press_then_release_elsewhere_switches_once(shown):
    b, calls = shown
    a, other = b.names()[1], b.names()[4]
    calls.clear()
    src = b._heads[a]["label"]._label
    dst = b._heads[other]["label"]._label
    src.event_generate("<ButtonPress-1>", x=2, y=2)
    src.event_generate("<ButtonRelease-1>", x=2, y=2,
                       rootx=dst.winfo_rootx() + 2, rooty=dst.winfo_rooty() + 2)
    b.update()
    assert b.get() == a and calls == [a]

def test_press_and_release_on_same_head_fire_the_command_once(shown):
    b, calls = shown
    name = b.names()[5]
    calls.clear()
    w = b._heads[name]["glyph"]._label
    w.event_generate("<ButtonPress-1>", x=1, y=1)
    w.event_generate("<ButtonRelease-1>", x=1, y=1, rootx=w.winfo_rootx() + 1, rooty=w.winfo_rooty() + 1)
    b.update()
    assert calls == [name]

def test_head_is_redrawn_before_the_command_runs(shown):
    b, _ = shown
    name = b.names()[2]
    seen = []
    b._command = lambda: seen.append((b._heads[name]["label"]._label.winfo_ismapped(),
                                      b._heads[name]["label"]._canvas.winfo_ismapped(),
                                      b.tab(name).winfo_ismapped()))
    b._heads[name]["label"]._label.event_generate("<ButtonPress-1>", x=1, y=1)
    assert seen == [(1, 1, 1)]

def test_root_button1_binding_still_runs(shown):
    b, _ = shown
    hits = []
    b.winfo_toplevel().bind("<Button-1>", lambda e: hits.append(1), add="+")
    b._heads[b.names()[1]]["label"]._label.event_generate("<ButtonPress-1>", x=1, y=1)
    assert hits == [1]
