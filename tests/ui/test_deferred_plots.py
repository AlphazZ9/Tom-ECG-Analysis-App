"""A plot on a hidden tab draws when its tab is shown, or when flushed."""
from __future__ import annotations

import tkinter as tk

import pytest

from ecg.ui.plots import CanvasSlot


@pytest.fixture(scope="module")
def shown_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.geometry("400x300+-4000+0")        # shown, off screen
    root.update()
    if not root.winfo_viewable():
        root.destroy()
        pytest.skip("window not shown")
    yield root
    root.destroy()


def _slot_in(root, shown: bool):
    for w in root.winfo_children():         # one window for the module's tests
        w.destroy()
    page = tk.Frame(root, width=300, height=200)
    if shown:
        page.pack(fill="both", expand=True)
    slot = CanvasSlot(page, 3, 2, toolbar=False)
    root.update()
    return page, slot


def test_a_hidden_slot_draws_when_shown(shown_root):
    page, slot = _slot_in(shown_root, shown=False)
    calls = []
    slot.update(lambda fig: calls.append(fig.add_subplot(111)))
    shown_root.update()
    assert calls == []                      # nobody can see it: not drawn yet
    page.pack(fill="both", expand=True)     # its tab is shown
    shown_root.update()
    assert len(calls) >= 1
    n = len(calls)
    slot.flush()                            # nothing left to draw
    assert len(calls) == n


def test_flush_draws_a_hidden_slot_for_an_export(shown_root):
    _page, slot = _slot_in(shown_root, shown=False)
    calls = []
    slot.update(lambda fig: calls.append(fig.add_subplot(111)))
    slot.flush()
    assert len(calls) == 1 and slot.fig.axes


def test_a_shown_slot_draws_at_once(shown_root):
    _page, slot = _slot_in(shown_root, shown=True)
    calls = []
    slot.update(lambda fig: calls.append(fig.add_subplot(111)))
    assert len(calls) == 1
