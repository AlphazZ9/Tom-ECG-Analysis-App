# -*- coding: utf-8 -*-
"""
ecg.ui.widgets
---------------
Small, reusable, standalone widget factories shared across multiple UI
modules -- unlike ECGApp's private _btn()/_switch() helpers (app.py), which
are bound methods used only within app.py itself, these are plain functions
meant to be imported from more than one call site.
"""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import customtkinter as ctk  # type: ignore[import-untyped]
import numpy as np

from ecg.ui.theme import (
    CARD, BORDER, TEXT, MUTED, LIGHT, BLUE,
    GREEN, GREEN_MID, AMBER, RED,
    FONT_KPI_LABEL, FONT_KPI_VALUE, FONT_KPI_HERO, FONT_MICRO, FONT_HINT,
    button_style,
)


def make_stat_tile(
    parent,
    label: str,
    value: str,
    *,
    unit: str = "",
    sublabel: str = "",
    value_color: "Optional[str]" = None,
    hero: bool = False,
    card: bool = False,
    accent: "Optional[str]" = None,
) -> ctk.CTkFrame:
    """Build one label/value stat tile. Unplaced -- caller grids/packs it.

    Serves both a dense multi-cell strip (card=False: chrome-less, packed
    tightly into a panel that already supplies the background) and a
    standalone stat-cards panel (card=True: self-contained CARD background
    + padding) from the same function, so neither future call site needs
    its own implementation.

    Returns the outer frame with ``.value_label`` (and ``.sublabel_label``
    if *sublabel* was given) exposed as attributes, so callers can update
    the displayed value later: ``tile.value_label.configure(text="524")``.
    """
    if card:
        outer = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=8,
                              border_width=1, border_color=BORDER)
        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=10)
    else:
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        inner = outer

    if accent:
        ctk.CTkFrame(inner, fg_color=accent, height=3, corner_radius=2).pack(
            fill="x", pady=(0, 6))

    ctk.CTkLabel(inner, text=label, font=FONT_KPI_LABEL, text_color=MUTED,
                 anchor="w").pack(fill="x")

    value_row = ctk.CTkFrame(inner, fg_color="transparent")
    value_row.pack(fill="x")
    value_label = ctk.CTkLabel(
        value_row, text=value, font=FONT_KPI_HERO if hero else FONT_KPI_VALUE,
        text_color=value_color or TEXT, anchor="w")
    value_label.pack(side="left")
    if unit:
        ctk.CTkLabel(value_row, text=f" {unit}", font=FONT_KPI_LABEL,
                     text_color=MUTED, anchor="w").pack(side="left", padx=(2, 0))

    sublabel_label = None
    if sublabel:
        sublabel_label = ctk.CTkLabel(
            inner, text=sublabel, font=FONT_MICRO, text_color=LIGHT, anchor="w")
        sublabel_label.pack(fill="x")

    outer.value_label = value_label  # type: ignore[attr-defined]
    if sublabel_label is not None:
        outer.sublabel_label = sublabel_label  # type: ignore[attr-defined]
    return outer


def make_sparkline(
    parent,
    values: "Optional[np.ndarray]",
    *,
    width: int = 90,
    height: int = 26,
    color: str = BLUE,
    bg: "Optional[str]" = None,
) -> tk.Canvas:
    """Tiny polyline preview of a short numeric series (e.g. RR_ms per beat).

    Plain tkinter.Canvas rather than a matplotlib Figure -- this is meant
    to be instantiated once per row in a list of recent recordings, where
    a full Figure/FigureCanvasTkAgg per row would be far too heavy for
    what's a handful of pixels of line.
    """
    canvas = tk.Canvas(parent, width=width, height=height,
                       bg=bg or CARD, highlightthickness=0)
    if values is None:
        return canvas
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return canvas
    if len(vals) > 200:   # ~90px wide -- no need to plot more points than that
        idx = np.linspace(0, len(vals) - 1, 200).astype(int)
        vals = vals[idx]
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    pad = 3
    xs = np.linspace(pad, width - pad, len(vals))
    ys = height - pad - (vals - lo) / span * (height - 2 * pad)
    canvas.create_line(*(c for xy in zip(xs, ys) for c in xy),
                       fill=color, width=1.4, smooth=True)
    return canvas


class PillGroup(ctk.CTkFrame):
    """The mockup's mode pills: one rounded button per value, 8 px apart, the
    selected one blue. Same .set / .get as the CTkSegmentedButton it
    replaces: .set selects without calling *command*, a click selects and
    calls command(value)."""

    def __init__(self, parent, values, command=None, **kw) -> None:
        super().__init__(parent, fg_color="transparent", **kw)
        self._command = command
        self._value: "Optional[str]" = None
        self._buttons: "dict[str, ctk.CTkButton]" = {}
        for i, v in enumerate(values):
            b = ctk.CTkButton(self, text=v, width=0, command=lambda v=v: self._click(v),
                              **button_style("pill_off"))
            b.pack(side="left", padx=(0 if i == 0 else 8, 0))
            self._buttons[v] = b

    def _click(self, value: str) -> None:
        self.set(value)
        if self._command is not None:
            self._command(value)

    def set(self, value: str) -> None:
        if value not in self._buttons:
            raise ValueError(f"no pill named {value!r}")
        self._value = value
        for v, b in self._buttons.items():
            look = button_style("pill_on" if v == value else "pill_off")
            b.configure(fg_color=look["fg_color"], hover_color=look["hover_color"],
                        text_color=look["text_color"], border_color=look["border_color"],
                        font=look["font"])

    def get(self) -> str:
        return self._value or ""
