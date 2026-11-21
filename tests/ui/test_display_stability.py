"""Nothing moves while a job runs: the progress bar sits in a slot of fixed
size in the header (placed, not packed), a busy button keeps its width (its
longer busy label cut to it), a figure is sized with its own dpi (the Tk
canvas renders at the screen's device pixel ratio: sized with the nominal
dpi it overflowed its frame and was resized back and forth), and Compare
segments is not rebuilt each time it is opened without a recording."""
from __future__ import annotations

import time
import tkinter as tk
from types import SimpleNamespace

import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui.app_async import fit_progress_text, freeze_button_width, release_button_width
from ecg.ui.plots import CanvasSlot


@pytest.fixture
def root():
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.geometry("900x500+0+0")
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _pump(r, sec=0.6):
    t = time.time()
    while time.time() - t < sec:
        r.update()
        time.sleep(0.01)


def test_progress_text_keeps_the_percentage_and_time_left():
    measure = 6  # px per character without a Tk font
    short = fit_progress_text("42%  Core", "  (~3s left)", width=300)
    assert short == "42%  Core  (~3s left)"
    long = fit_progress_text("42%  " + "x" * 200, "  (~3s left)", width=120)
    assert long.startswith("42%") and long.endswith("…  (~3s left)")
    assert len(long) * measure <= 120 or len(long) < 40


def test_busy_button_keeps_its_width(root):
    row = ctk.CTkFrame(root)
    row.pack(fill="x")
    btn = ctk.CTkButton(row, text="Detect Peaks")
    btn.pack(side="left")
    nxt = ctk.CTkButton(row, text="Analyze all")
    nxt.pack(side="left")
    _pump(root)
    w, x = btn.winfo_width(), nxt.winfo_rootx()
    label = freeze_button_width(btn, "Checking for saved session…")
    btn.configure(text=label)
    _pump(root)
    assert label.endswith("…") and len(label) < len("Checking for saved session…")
    assert btn.winfo_width() == w and nxt.winfo_rootx() == x
    btn.configure(text="Detect Peaks")
    release_button_width(btn)
    _pump(root)
    assert btn.winfo_width() == w and not hasattr(btn, "_busy_own_width")
    # A stub (tests, a destroyed button): the label as given.
    assert freeze_button_width(SimpleNamespace(), "Saving…") == "Saving…"


def test_progress_slot_shows_without_moving_anything(root):
    """The header's arrangement: the slot is always packed, its content placed."""
    row = ctk.CTkFrame(root, height=60)
    row.pack(fill="x")
    row.pack_propagate(False)
    right = ctk.CTkButton(row, text="Open")
    right.pack(side="right")
    slot = ctk.CTkFrame(row, width=300, height=40, fg_color="transparent")
    slot.pack(side="right")
    slot.pack_propagate(False)
    inner = ctk.CTkFrame(slot, fg_color="transparent")
    ctk.CTkProgressBar(inner).pack(fill="x")
    ctk.CTkLabel(inner, text="42%  Core HRV").pack(fill="x")
    below = ctk.CTkLabel(root, text="tabs")
    below.pack()
    _pump(root)
    before = (right.winfo_rootx(), below.winfo_rooty(), slot.winfo_width())
    inner.place(relx=0, rely=0, relwidth=1, relheight=1)
    _pump(root)
    assert (right.winfo_rootx(), below.winfo_rooty(), slot.winfo_width()) == before
    inner.place_forget()
    _pump(root)
    assert (right.winfo_rootx(), below.winfo_rooty(), slot.winfo_width()) == before


def test_a_figure_fills_its_frame_at_the_screen_dpi(root):
    holder = tk.Frame(root, width=700, height=300)
    holder.pack(fill="both", expand=True)
    slot = CanvasSlot(holder, 12, 4, toolbar=False)
    slot.update(lambda fig: fig.add_subplot(111).plot([0, 1], [0, 1]))
    _pump(root, 1.0)
    w_in, h_in = slot.fig.get_size_inches()
    assert abs(w_in * slot.fig.dpi - slot._cv_frame.winfo_width()) <= 3
    assert abs(h_in * slot.fig.dpi - slot._cv_frame.winfo_height()) <= 3
    # Settled: another fit with the same frame draws nothing.
    calls = []
    slot._draw_fn = lambda fig: calls.append(1)
    slot._fit()
    assert calls == []


def test_compare_tab_is_not_rebuilt_without_a_recording(root):
    from ecg.ui.compare_segments import CompareSegmentsMixin
    holder = ctk.CTkFrame(root)
    built = []
    app = SimpleNamespace(_compare_holder=holder, _compare_built_for=None,
                          _compare_signature=lambda: None,
                          _build_group_card=lambda h: built.append(
                              ctk.CTkLabel(h, text="group").pack()))
    CompareSegmentsMixin._refresh_compare_tab(app)
    CompareSegmentsMixin._refresh_compare_tab(app)
    assert len(built) == 1


def test_a_waiting_draw_is_made_when_its_scrolling_page_is_shown_again(root):
    """A slot inside a CTkScrollableFrame gets no <Map> when its tab comes
    back: the size check that follows draws what was deferred."""
    tab = ctk.CTkFrame(root)
    tab.pack(fill="both", expand=True)
    page = ctk.CTkScrollableFrame(tab)
    page.pack(fill="both", expand=True)
    holder = tk.Frame(page, width=600, height=250)
    holder.pack(fill="x")
    holder.pack_propagate(False)
    slot = CanvasSlot(holder, 6, 2.5, toolbar=False)
    _pump(root)
    tab.pack_forget()
    _pump(root, 0.2)
    slot.update(lambda fig: fig.add_subplot(111).set_title("drawn"))
    assert slot._pending == "draw"
    tab.pack(fill="both", expand=True)
    _pump(root, 1.0)
    slot._fit()
    assert slot._pending is None and slot.fig.axes[0].get_title() == "drawn"
