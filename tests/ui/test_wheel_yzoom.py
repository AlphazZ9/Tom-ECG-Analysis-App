"""Tachogram: the mouse wheel zooms the Y axis of the panel under the pointer,
around the pointer; the middle button restores the drawn limits."""
from __future__ import annotations

import tkinter as tk

import pytest

from ecg.ui.plots import CanvasSlot


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    r.geometry("500x400+-4000+0")           # shown, off screen
    r.update()
    if not r.winfo_viewable():
        r.destroy()
        pytest.skip("window not shown")
    yield r
    r.destroy()


def _slot(root):
    for w in root.winfo_children():
        w.destroy()
    page = tk.Frame(root, width=480, height=380)
    page.pack(fill="both", expand=True)
    slot = CanvasSlot(page, 4.8, 3.8, toolbar=False)
    slot.enable_wheel_yzoom()

    def draw(fig):
        top, bottom = fig.subplots(2, 1)
        top.set_ylim(100, 200)
        bottom.set_ylim(300, 600)
    slot.update(draw)
    slot.flush() if hasattr(slot, "flush") else None
    root.update()
    slot.canvas.draw()
    root.update()
    return slot


def _pixel(slot, ax, y_data):
    """Tk widget coordinates of the centre of *ax* at data height *y_data*."""
    w = slot.canvas.get_tk_widget()
    x_disp, y_disp = ax.transData.transform((sum(ax.get_xlim()) / 2, y_data))
    return int(round(x_disp)), int(round(w.winfo_height() - y_disp))


def _wheel(slot, x, y, delta):
    slot.canvas.get_tk_widget().event_generate("<MouseWheel>", x=x, y=y, delta=delta)


def test_wheel_zooms_the_panel_under_the_pointer_around_it(root):
    slot = _slot(root)
    top, bottom = slot.fig.axes
    x, y = _pixel(slot, top, 125)
    _wheel(slot, x, y, 120)                         # away from you: zoom in
    lo, hi = top.get_ylim()
    assert hi - lo == pytest.approx(80, rel=0.05)   # 100 ms span × 0.8
    assert lo == pytest.approx(105, abs=2) and hi == pytest.approx(185, abs=2)  # around 125
    assert bottom.get_ylim() == (300, 600)          # the other panel untouched
    _wheel(slot, x, y, -120)                        # towards you: back out
    lo, hi = top.get_ylim()
    assert hi - lo == pytest.approx(100, rel=0.05)


def test_middle_click_restores_the_drawn_limits(root):
    slot = _slot(root)
    _top, bottom = slot.fig.axes
    x, y = _pixel(slot, bottom, 450)
    for _ in range(3):
        _wheel(slot, x, y, 120)
    assert bottom.get_ylim()[1] - bottom.get_ylim()[0] < 200
    slot.canvas.get_tk_widget().event_generate("<Button-2>", x=x, y=y)
    assert bottom.get_ylim() == pytest.approx((300, 600))


def test_zooming_never_moves_or_shrinks_the_panels(root):
    """constrained_layout used to re-flow the figure whenever a zoom changed
    the tick labels' width ("140" -> "142.5"): the panels moved and shrank."""
    slot = _slot(root)
    w = slot.canvas.get_tk_widget()
    start = [ax.get_window_extent().bounds for ax in slot.fig.axes]
    for ax, y0 in zip(slot.fig.axes, (143.7, 421.3)):
        for _ in range(30):
            x, y = _pixel(slot, ax, min(max(y0, ax.get_ylim()[0]), ax.get_ylim()[1]))
            _wheel(slot, x, y, 120)
            root.update()
            slot.canvas.draw()
            assert [a.get_window_extent().bounds for a in slot.fig.axes] == start
    for ax in slot.fig.axes:                      # zoom stops at 1 % of the span
        lo, hi = ax.get_ylim()
        h_lo, h_hi = ax._wheel_home_ylim
        assert hi - lo >= 0.01 * (h_hi - h_lo) * 0.79


def test_a_redraw_lays_the_figure_out_again(root):
    slot = _slot(root)
    top, _bottom = slot.fig.axes
    x, y = _pixel(slot, top, 150)
    _wheel(slot, x, y, 120)
    assert slot.fig.get_layout_engine().__class__.__name__ != "ConstrainedLayoutEngine"
    slot._redraw(force=True)
    assert slot.fig.get_layout_engine().__class__.__name__ == "ConstrainedLayoutEngine"


def test_square_poincare_panels_zoom_together_without_reshaping(root):
    """A Y-only zoom reshaped an equal-aspect box (370 x 370 -> 652 x 267 px)
    and left the other Poincaré panel on another scale."""
    import numpy as np
    from ecg.ui.poincare_figure import draw_poincare_set
    for w_ in root.winfo_children():
        w_.destroy()
    page = tk.Frame(root, width=480, height=380)
    page.pack(fill="both", expand=True)
    slot = CanvasSlot(page, 4.8, 3.8, toolbar=False)
    slot.enable_wheel_yzoom()
    rng = np.random.default_rng(0)
    segs = []
    for m, lab in ((125, "Before"), (120, "After")):
        x = np.round(m + 2 * rng.standard_normal(2000))
        segs.append({"pairs_x": x, "pairs_y": np.r_[x[1:], m], "sd1": 0.7, "sd2": 2.0,
                     "label": lab})
    slot.update(lambda fig: draw_poincare_set(fig, segs, ("b", "orange")))
    root.update()
    slot.canvas.draw()
    root.update()
    a, b = slot.fig.axes
    boxes = [ax.get_window_extent().bounds for ax in (a, b)]
    home = a.get_xlim()
    x, y = _pixel(slot, a, 124)
    for _ in range(5):
        _wheel(slot, x, y, 120)
        root.update()
        slot.canvas.draw()
    assert [ax.get_window_extent().bounds for ax in (a, b)] == boxes
    assert a.get_xlim() == b.get_xlim() and a.get_ylim() == b.get_ylim()
    span_x, span_y = np.ptp(a.get_xlim()), np.ptp(a.get_ylim())
    assert span_x == pytest.approx(span_y) and span_x < 0.5 * np.ptp(home)
    slot.canvas.get_tk_widget().event_generate("<Button-2>", x=x, y=y)
    assert a.get_xlim() == pytest.approx(home) and b.get_xlim() == pytest.approx(home)
