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

from typing import Optional

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui.theme import (
    CARD, BORDER, TEXT, MUTED, LIGHT,
    GREEN, GREEN_MID, AMBER, RED,
    FONT_KPI_LABEL, FONT_KPI_VALUE, FONT_KPI_HERO, FONT_MICRO, FONT_HINT,
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


# ── Quality gauge ──────────────────────────────────────────────────────────
# Extends (does not alter) the existing 3-tier GREEN/ORANGE/RED logic already
# used for lbl_quality and the Summary tab's verdict text (>=70/>=40/else) --
# this is a separate, finer-grained 4-tier scale used only by the gauge.
_QUALITY_TIERS = (
    (90, "Excellent", GREEN),
    (70, "Good",      GREEN_MID),
    (40, "Medium",    AMBER),
    (0,  "Poor",      RED),
)


def _quality_tier(score: "Optional[int]") -> "tuple[str, str]":
    """Return (label, colour) for a 0-100 signal-quality score."""
    if score is None:
        return "—", MUTED
    for lo, label, color in _QUALITY_TIERS:
        if score >= lo:
            return label, color
    return "Poor", RED


def make_quality_gauge(
    parent,
    score: "Optional[int]" = None,
    *,
    compact: bool = True,
) -> ctk.CTkFrame:
    """Build a 4-zone segmented quality gauge. Unplaced -- caller packs it.

    Zones are Poor/Medium/Good/Excellent (RED/AMBER/GREEN_MID/GREEN), widths
    proportional to the _QUALITY_TIERS boundaries, with a thin marker placed
    over the track at the current score. Mixing .place() for the marker with
    .grid() for its zone-frame siblings is safe -- each widget picks its own
    geometry manager independently of its siblings.

    compact=True (used in the top bar this phase) is a short, small-caption
    track. compact=False is a larger variant reserved for future reuse (e.g.
    the Summary tab) -- not wired anywhere yet.

    Returns the outer frame with .track_frame/.marker/.score_label exposed
    so update_quality_gauge() can reposition/relabel it later.
    """
    outer = ctk.CTkFrame(parent, fg_color="transparent")

    track_h = 8 if compact else 14
    track = ctk.CTkFrame(outer, height=track_h,
                          width=90 if compact else 220,
                          fg_color="transparent")
    track.pack(fill=("x" if not compact else "none"))
    track.pack_propagate(False)

    # Widest-to-narrowest tier order matches _QUALITY_TIERS reversed (Poor
    # first, Excellent last) so the track reads low-to-high left-to-right.
    zone_weights = [40, 30, 20, 10]  # Poor, Medium, Good, Excellent widths
    zone_colors  = [RED, AMBER, GREEN_MID, GREEN]
    for i, (weight, color) in enumerate(zip(zone_weights, zone_colors)):
        track.columnconfigure(i, weight=weight)
        ctk.CTkFrame(track, fg_color=color, corner_radius=0).grid(
            row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 1, 0))

    marker = ctk.CTkFrame(track, fg_color=TEXT, width=2, corner_radius=0)
    marker.place(relx=0.0, rely=0, relheight=1.0)

    score_label = ctk.CTkLabel(
        outer, text="—", font=FONT_MICRO if compact else FONT_HINT,
        text_color=MUTED, anchor="w")
    score_label.pack(fill="x", pady=(2, 0))

    outer.track_frame = track    # type: ignore[attr-defined]
    outer.marker = marker        # type: ignore[attr-defined]
    outer.score_label = score_label  # type: ignore[attr-defined]
    update_quality_gauge(outer, score)
    return outer


def update_quality_gauge(gauge: ctk.CTkFrame, score: "Optional[int]") -> None:
    """Reposition the marker and update the caption for a new score."""
    label, color = _quality_tier(score)
    relx = 0.0 if score is None else max(0.0, min(1.0, score / 100.0))
    gauge.marker.place(relx=relx, rely=0, relheight=1.0)  # type: ignore[attr-defined]
    text = "—" if score is None else f"{label}  ·  {score}%"
    gauge.score_label.configure(text=text, text_color=color)  # type: ignore[attr-defined]
