# -*- coding: utf-8 -*-
"""
ecg.ui.dialogs
--------------
All CTk dialog/toplevel classes:
  ThemeDialog, ArtifactReviewDialog,
  AnnotationDialog, AnnotationManagerDialog.
"""
from __future__ import annotations

import dataclasses
import logging
import matplotlib
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from tkinter import colorchooser, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from ecg.ui.theme import (
    THEME, ThemeConfig, apply_theme_config,
    BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT, PLOT,
    RED, BLUE, GREEN, GREEN_DARK, ORANGE, PURPLE, PURPLE_DARK,
    BLUE_HOVER, RED_DARK, RED_LIGHT, ORANGE_DARK, ORANGE_DEEP,
    AMBER, GRAY, GRAY_LIGHT,
    FONT_TITLE, FONT_SECTION_HDR, FONT_LABEL, FONT_SMALL,
    FONT_BODY, FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR,
    make_font,
)
from ecg.core.models import MouseECG
from ecg.core.ml_detector import (
    training_data_summary, list_training_files, delete_training_sample,
    train_model, MLPeakModel,
)
from ecg.io.db import _DB_AVAILABLE, verified_recordings, set_verified
from ecg.io.export import ANNOTATION_COLORS

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")

class ThemeDialog(ctk.CTkToplevel):
    """Polished appearance settings panel.

    Layout
    ------
    Top   : title bar + subtitle
    Left  : 6 preset cards arranged in a 2-column grid; each shows a
            mini colour swatch strip (BG / PANEL / accent) + name + description
    Right : Font family combo, size slider, 4 accent colour override pickers,
            reset-to-preset button
    Bottom: Apply / Save as Default / Close

    Applying immediately calls app._rebuild_ui(); closing without saving
    discards the preview.
    """

    _SWATCH_KEYS = ("BG", "PANEL", "CARD", "BLUE", "RED", "GREEN")

    def __init__(self, parent: "ECGApp") -> None:
        super().__init__(parent)
        self.title("Appearance Settings")
        self.geometry("980x640")
        self.resizable(True, True)
        self.grab_set()

        self._app       = parent
        self._working   = ThemeConfig.load()   # copy to work with
        self._working.apply_preset(THEME.preset_name.rstrip("*"))
        self._working.font_family = THEME.font_family
        self._working.font_scale  = THEME.font_scale
        # Carry over any custom accent overrides
        for k in ("RED","BLUE","GREEN","ORANGE"):
            self._working._colors[k] = THEME._colors.get(k, self._working._colors[k])

        self._saved = False
        self._accent_vars: "dict[str, tk.StringVar]" = {}

        self._build()
        self._highlight_active_preset()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build(self) -> None:
        self.configure(fg_color=BG)

        # Title bar
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Appearance", font=make_font(18, bold=True),
                     text_color=TEXT, anchor="w").pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(hdr, text="Choose a preset or customise colours and fonts",
                     font=make_font(11), text_color=MUTED, anchor="w").pack(
                         side="left", padx=(0, 20), pady=12)

        # Main area
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(12, 0))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=0)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # LEFT: preset grid
        self._build_preset_panel(body)

        # Divider
        ctk.CTkFrame(body, width=1, fg_color=BORDER).grid(
            row=0, column=1, sticky="ns", padx=12, pady=4)

        # RIGHT: font + accent controls
        self._build_controls_panel(body)

        # Bottom bar
        self._build_bottom_bar()

    def _build_preset_panel(self, parent) -> None:
        left = ctk.CTkFrame(parent, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(left, text="COLOUR PRESETS",
                     font=make_font(11, bold=True), text_color=MUTED,
                     anchor="w").pack(fill="x", padx=4, pady=(0, 8))

        grid = ctk.CTkFrame(left, fg_color="transparent")
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self._preset_btns: "dict[str, ctk.CTkFrame]" = {}
        row, col = 0, 0
        for name, preset in ThemeConfig.PRESETS.items():
            card = self._make_preset_card(grid, name, preset)
            card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            grid.rowconfigure(row, weight=1)
            self._preset_btns[name] = card
            col += 1
            if col > 1:
                col = 0
                row += 1

    def _make_preset_card(self, parent, name: str, preset: dict) -> ctk.CTkFrame:
        is_dark = preset.get("is_dark", False)
        bg      = preset.get("BG",    "#F0F0F0")
        panel   = preset.get("PANEL", GRAY_LIGHT)
        card_c  = preset.get("CARD",  "#FFFFFF")
        blue    = preset.get("BLUE",  "#1A56DB")
        red     = preset.get("RED",   "#C62828")
        green   = preset.get("GREEN", "#1B5E20")
        text_c  = preset.get("TEXT",  "#0F172A")
        muted_c = preset.get("MUTED", "#64748B")
        desc    = preset.get("description", "")

        outer = ctk.CTkFrame(parent, fg_color=panel, corner_radius=10,
                             border_width=2, border_color=BORDER)
        outer.bind("<Button-1>", lambda e, n=name: self._select_preset(n))

        # Colour swatch strip
        swatch_row = tk.Frame(outer, bg=panel, height=28)
        swatch_row.pack(fill="x", padx=10, pady=(10, 6))
        swatch_row.pack_propagate(False)
        for color in (bg, panel, card_c, blue, red, green):
            f = tk.Frame(swatch_row, bg=color, width=22, height=22)
            f.pack(side="left", padx=2)
            f.bind("<Button-1>", lambda e, n=name: self._select_preset(n))

        # Name + dark/light badge
        name_row = ctk.CTkFrame(outer, fg_color="transparent")
        name_row.pack(fill="x", padx=10)
        ctk.CTkLabel(name_row, text=name, font=make_font(12, bold=True),
                     text_color=TEXT, anchor="w").pack(side="left")
        badge_col  = "#2E3440" if is_dark else "#E8ECF2"
        badge_text = "dark" if is_dark else "light"
        badge_fc   = "#ECEFF4" if is_dark else "#555F6E"
        ctk.CTkLabel(name_row,
                     text=f"  {badge_text}  ",
                     font=make_font(9), fg_color=badge_col, text_color=badge_fc,
                     corner_radius=4).pack(side="left", padx=(6, 0))

        # Description
        ctk.CTkLabel(outer, text=desc, font=make_font(10), text_color=muted_c,
                     anchor="w", wraplength=220, justify="left").pack(
                         fill="x", padx=10, pady=(4, 10))

        outer.bind("<Button-1>", lambda e, n=name: self._select_preset(n))
        return outer

    def _build_controls_panel(self, parent) -> None:
        right = ctk.CTkScrollableFrame(parent, fg_color="transparent",
                                        scrollbar_button_color=BORDER,
                                        scrollbar_button_hover_color=BORDER2,
                                        width=280)
        right.grid(row=0, column=2, sticky="nsew", pady=0)

        # ── Font family ───────────────────────────────────────────────────────
        ctk.CTkLabel(right, text="FONT FAMILY", font=make_font(11, bold=True),
                     text_color=MUTED, anchor="w").pack(fill="x", padx=8, pady=(4, 4))

        font_families = self._available_fonts()
        self._font_var = tk.StringVar(value=self._working.font_family)
        self._font_combo = ctk.CTkComboBox(
            right, values=font_families, variable=self._font_var,
            fg_color=CARD, border_color=BORDER2, text_color=TEXT,
            button_color=BORDER2, button_hover_color=BORDER,
            dropdown_fg_color=PANEL, dropdown_text_color=TEXT,
            font=make_font(11), width=260, height=30,
            command=self._on_font_family_change,
        )
        self._font_combo.pack(padx=8, pady=(0, 12))

        # ── Font size slider ──────────────────────────────────────────────────
        ctk.CTkLabel(right, text="FONT SIZE", font=make_font(11, bold=True),
                     text_color=MUTED, anchor="w").pack(fill="x", padx=8, pady=(0, 4))

        size_row = ctk.CTkFrame(right, fg_color="transparent")
        size_row.pack(fill="x", padx=8, pady=(0, 2))
        ctk.CTkLabel(size_row, text="A", font=make_font(9), text_color=MUTED).pack(side="left")

        self._size_label = ctk.CTkLabel(size_row, text=f"{self._working.font_scale:.2f}x",
                                         font=make_font(11), text_color=TEXT, width=48)
        self._size_label.pack(side="right")

        self._scale_slider = ctk.CTkSlider(
            right, from_=0.75, to=1.50,  # type: ignore[arg-type]
            number_of_steps=15,
            command=self._on_scale_change,
            fg_color=BORDER, progress_color=BLUE,
            button_color=TEXT, button_hover_color=MUTED,
        )
        self._scale_slider.set(self._working.font_scale)
        self._scale_slider.pack(fill="x", padx=8, pady=(0, 16))

        # ── Accent colour overrides ───────────────────────────────────────────
        ctk.CTkFrame(right, height=1, fg_color=BORDER).pack(fill="x", padx=8, pady=(0, 10))
        ctk.CTkLabel(right, text="ACCENT COLOURS",
                     font=make_font(11, bold=True), text_color=MUTED,
                     anchor="w").pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(right,
                     text="Override individual accent colours.\n"
                          "Enter a hex value (e.g. #FF5555).",
                     font=make_font(10), text_color=MUTED,
                     anchor="w", justify="left").pack(fill="x", padx=8, pady=(0, 8))

        for token, label in [("RED","Alert / Threshold"), ("BLUE","Signal / Primary"),
                              ("GREEN","Accepted peaks"),  ("ORANGE","Warning / Edit mode")]:
            row = ctk.CTkFrame(right, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=3)

            current = self._working._colors.get(token, "#888888")
            var = tk.StringVar(value=current)
            self._accent_vars[token] = var

            # Colour preview swatch
            swatch = tk.Frame(row, bg=current, width=22, height=22, relief="flat")
            swatch.pack(side="left", padx=(0, 8))

            ctk.CTkLabel(row, text=label, font=make_font(10), text_color=TEXT,
                         anchor="w", width=120).pack(side="left")

            ent = ctk.CTkEntry(row, textvariable=var, width=80, height=26,
                               font=make_font(10), fg_color=CARD,
                               border_color=BORDER2)
            ent.pack(side="right")

            # Update swatch on entry change
            def _update_swatch(sv=var, sw=swatch, tk_name=token):
                try:
                    hex_val = sv.get().strip()
                    if len(hex_val) == 7 and hex_val.startswith("#"):
                        sw.configure(bg=hex_val)
                        self._working.set_accent(tk_name, hex_val)
                except Exception as _sw_exc:
                    log.debug("colour swatch trace: %s", _sw_exc)
            var.trace_add("write", lambda *_a, sv=var, sw=swatch, tn=token: (
                _update_swatch(sv, sw, tn)))

        ctk.CTkFrame(right, height=1, fg_color=BORDER).pack(fill="x", padx=8, pady=(12, 8))
        ctk.CTkButton(right, text="Reset accents to preset",
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=make_font(10), height=26, corner_radius=5,
                      command=self._reset_accents).pack(padx=8, fill="x")

    def _build_bottom_bar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=56)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        ctk.CTkFrame(bar, height=1, fg_color=BORDER).pack(fill="x", side="top")

        ctk.CTkButton(bar, text="Close",
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=make_font(11), width=90, height=34,
                      command=self.destroy).pack(side="right", padx=12, pady=10)
        ctk.CTkButton(bar, text="Save as Default",
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      font=make_font(11, bold=True), width=130, height=34,
                      command=self._save_default).pack(side="right", padx=(0, 6), pady=10)
        ctk.CTkButton(bar, text="Apply",
                      fg_color=BLUE, hover_color=TEXT, text_color="white",
                      font=make_font(13, bold=True), width=100, height=34,
                      command=self._apply).pack(side="right", padx=(0, 6), pady=10)

        self._preview_lbl = ctk.CTkLabel(
            bar,
            text=f"Active: {THEME.preset_name}  |  font: {THEME.font_family}  |  scale: {THEME.font_scale:.2f}x",
            font=make_font(10), text_color=MUTED, anchor="w")
        self._preview_lbl.pack(side="left", padx=16)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _available_fonts(self) -> "list[str]":
        """Return a sorted list of available system font families."""
        import tkinter.font as tkf
        try:
            families = sorted(set(tkf.families(self)))
            # Prioritise the ThemeConfig-suggested families at the top
            preferred = [f for f in ThemeConfig.DEFAULT_FONT_FAMILIES if f in families]
            rest      = [f for f in families if f not in preferred and not f.startswith("@")]
            return preferred + rest
        except Exception:
            return ThemeConfig.DEFAULT_FONT_FAMILIES

    def _highlight_active_preset(self) -> None:
        active = self._working.preset_name.rstrip("*")
        for name, card in self._preset_btns.items():
            card.configure(border_color=BLUE if name == active else BORDER,
                           border_width=3 if name == active else 2)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _select_preset(self, name: str) -> None:
        self._working.apply_preset(name)
        self._highlight_active_preset()
        # Sync accent entry vars
        for token, var in self._accent_vars.items():
            var.set(self._working._colors.get(token, ""))
        self._update_preview_label()

    def _on_font_family_change(self, val: str) -> None:
        self._working.font_family = val
        self._update_preview_label()

    def _on_scale_change(self, val: float) -> None:
        self._working.font_scale = round(float(val), 2)
        self._size_label.configure(text=f"{self._working.font_scale:.2f}x")
        self._update_preview_label()

    def _reset_accents(self) -> None:
        name = self._working.preset_name.rstrip("*")
        preset = ThemeConfig.PRESETS.get(name, {})
        for token, var in self._accent_vars.items():
            val = preset.get(token, self._working._colors.get(token, "#888888"))
            var.set(val)
            self._working._colors[token] = val
        self._working.preset_name = name   # remove * marker

    def _update_preview_label(self) -> None:
        self._preview_lbl.configure(
            text=f"Preview: {self._working.preset_name}  |  "
                 f"font: {self._working.font_family}  |  "
                 f"scale: {self._working.font_scale:.2f}x")

    def _apply(self) -> None:
        """Apply the working config to the live app (rebuild UI).

        _rebuild_ui() calls ctk.set_appearance_mode() which invalidates
        CTk widget handles inside this dialog.  We therefore close the
        dialog first, then rebuild the main window.
        """
        global THEME
        THEME.apply_preset(self._working.preset_name.rstrip("*"))
        THEME.font_family = self._working.font_family
        THEME.font_scale  = self._working.font_scale
        for k, v in self._working._colors.items():
            THEME._colors[k] = v
        THEME.is_dark = self._working.is_dark
        # Destroy dialog before rebuilding so CTk appearance change
        # does not invalidate this window's widget handles mid-flight.
        self.destroy()
        apply_theme_config(THEME)
        self._app._rebuild_ui()

    def _save_default(self) -> None:
        """Apply + persist to disk."""
        THEME.save()
        self._saved = True
        self._apply()   # destroys self; no widget access after this line


# ════════════════════════════════════════════════════════════
#  USER WAVE TEMPLATE  (P/Q/R/S/T landmark definitions)
# ════════════════════════════════════════════════════════════

WAVE_TEMPLATE_PATH:        Path = Path.home() / ".ecg_wave_template.json"
_WAVE_TEMPLATE_PATH_LEGACY: Path = Path.home() / ".ecg_wave_template.pkl"



class ArtifactReviewDialog(ctk.CTkToplevel):
    """Modal dialog that walks through each detected artifact for review.

    For each candidate the user sees:
      • A 1.5 s context window of the filtered ECG signal
      • The suspect peak highlighted in red, neighbours in green
      • RR interval values + deviation from local median
      • Type label (Non-physiological / Ectopic / Duplicate)
      • Keep / Remove buttons — decision stored immediately

    After reviewing all candidates (or clicking "Apply All"), calling
    ``get_result()`` returns the updated candidate list for
    ``apply_artifact_decisions()``.
    """

    _TYPE_COLOR = {
        "nonphysio": RED_LIGHT,   # red
        "ectopic":   AMBER,   # orange
        "duplicate": "#AB47BC",   # purple
    }
    _TYPE_LABEL = {
        "nonphysio": "Non-physiological (outside HR bounds)",
        "ectopic":   "Ectopic beat (RR deviation from local median)",
        "duplicate": "Duplicate / sub-minimum interval",
    }

    def __init__(
        self,
        parent,
        signal: np.ndarray,
        rpeaks: np.ndarray,
        fs: float,
        candidates: list[dict],
        rr_min_ms: float = MouseECG.RR_MIN_MS,
    ):
        super().__init__(parent)
        self.title("Artifact Review")
        self.geometry("960x640")
        self.minsize(820, 520)
        self.configure(fg_color=BG)
        self.grab_set()          # modal
        self.focus_force()

        self._sig        = signal
        self._rpeaks     = np.asarray(rpeaks, dtype=int)
        self._fs         = fs
        self._candidates = [dict(c) for c in candidates]   # deep copy
        self._rr_min_ms  = rr_min_ms
        self._idx        = 0      # current candidate index
        self._result     = None   # set when dialog closes

        if not self._candidates:
            # Nothing to review — close immediately
            self.after(0, self._finish)
            return

        self._build_ui()
        self._show_candidate(0)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self) -> None:
        n = len(self._candidates)

        # ── Header bar ──────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=52)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        self.lbl_title = ctk.CTkLabel(
            hdr, text="", font=make_font(13, bold=True), text_color=TEXT)
        self.lbl_title.pack(side="left", padx=16, pady=10)

        self.lbl_counter = ctk.CTkLabel(
            hdr, text=f"0 / {n}", font=FONT_SMALL, text_color=MUTED)
        self.lbl_counter.pack(side="right", padx=16)

        # ── Progress bar ────────────────────────────────────────
        self.progress = ctk.CTkProgressBar(self, height=4, corner_radius=0,
                                            fg_color=BORDER, progress_color=ORANGE)
        self.progress.set(0)
        self.progress.pack(fill="x")

        # ── Main content (plot left, info right) ─────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=12, pady=8)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=3)   # plot
        content.grid_columnconfigure(1, weight=1)   # info panel

        # ── Signal plot ─────────────────────────────────────────
        plot_card = ctk.CTkFrame(content, fg_color=CARD, corner_radius=6)
        plot_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        plot_card.grid_rowconfigure(0, weight=1)
        plot_card.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 4), dpi=96, facecolor=PLOT["bg"],
                          tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                          padx=4, pady=4)

        # ── Info + controls panel ───────────────────────────────
        info = ctk.CTkFrame(content, fg_color=CARD, corner_radius=6)
        info.grid(row=0, column=1, sticky="nsew")
        info.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(info, text="ARTIFACT INFO", font=make_font(10, bold=True),
                     text_color=MUTED).grid(row=0, column=0, sticky="w",
                                            padx=12, pady=(12, 4))

        self.lbl_type = ctk.CTkLabel(info, text="", font=FONT_SMALL,
                                      text_color=ORANGE, wraplength=200,
                                      justify="left", anchor="w")
        self.lbl_type.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))

        # Metric rows
        self._metric_labels = {}
        for row_i, (key, label) in enumerate([
            ("sample_t",  "Time"),
            ("rr_prev",   "RR before"),
            ("rr_next",   "RR after"),
            ("rr_ref",    "Local median"),
            ("deviation", "Deviation"),
        ], start=2):
            ctk.CTkLabel(info, text=label, font=FONT_SMALL,
                         text_color=MUTED, anchor="w").grid(
                row=row_i, column=0, sticky="w", padx=12, pady=1)
            lbl = ctk.CTkLabel(info, text="—", font=make_font(11, bold=True),
                                text_color=TEXT, anchor="w")
            lbl.grid(row=row_i + 1, column=0, sticky="w", padx=20, pady=(0, 4))
            self._metric_labels[key] = lbl

        # Decision indicator
        ctk.CTkFrame(info, height=1, fg_color=BORDER).grid(
            row=13, column=0, sticky="ew", padx=12, pady=8)
        ctk.CTkLabel(info, text="DECISION", font=make_font(10, bold=True),
                     text_color=MUTED).grid(row=14, column=0, sticky="w",
                                             padx=12, pady=(0, 4))
        self.lbl_decision = ctk.CTkLabel(info, text="Remove",
                                          font=make_font(13, bold=True),
                                          text_color=RED)
        self.lbl_decision.grid(row=15, column=0, sticky="w", padx=12, pady=(0, 8))

        # Keep / Remove toggle buttons
        btn_frame = ctk.CTkFrame(info, fg_color="transparent")
        btn_frame.grid(row=16, column=0, sticky="ew", padx=10, pady=(0, 8))
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self.btn_keep = ctk.CTkButton(
            btn_frame, text="✓  Keep", height=36,
            fg_color=GREEN, hover_color=GREEN, text_color="white",
            font=make_font(12, bold=True), corner_radius=5,
            command=lambda: self._set_decision("keep"))
        self.btn_keep.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_remove = ctk.CTkButton(
            btn_frame, text="✕  Remove", height=36,
            fg_color=RED_DARK, hover_color="#D32F2F", text_color="white",
            font=make_font(12, bold=True), corner_radius=5,
            command=lambda: self._set_decision("remove"))
        self.btn_remove.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # Spacer
        info.grid_rowconfigure(17, weight=1)

        # ── Navigation bar (bottom) ──────────────────────────────
        nav = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=54)
        nav.pack(fill="x", side="bottom")
        nav.pack_propagate(False)

        ctk.CTkButton(
            nav, text="◀  Prev", width=90, height=34,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_LABEL, corner_radius=5,
            command=self._prev).pack(side="left", padx=10, pady=10)

        ctk.CTkButton(
            nav, text="Next  ▶", width=90, height=34,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_LABEL, corner_radius=5,
            command=self._next).pack(side="left", padx=(0, 10), pady=10)

        ctk.CTkButton(
            nav, text="⚡  Remove All", width=130, height=34,
            fg_color="#5D4037", hover_color="#4E342E", text_color="white",
            font=FONT_LABEL, corner_radius=5,
            command=self._remove_all).pack(side="left", padx=(0, 6), pady=10)

        ctk.CTkButton(
            nav, text="✓  Keep All", width=110, height=34,
            fg_color=GREEN, hover_color=GREEN, text_color="white",
            font=FONT_LABEL, corner_radius=5,
            command=self._keep_all).pack(side="left", padx=(0, 10), pady=10)

        self.lbl_summary = ctk.CTkLabel(
            nav, text="", font=FONT_SMALL, text_color=MUTED)
        self.lbl_summary.pack(side="left", padx=10)

        ctk.CTkButton(
            nav, text="✓  Apply Decisions", width=150, height=36,
            fg_color=RED, hover_color=RED_DARK, text_color="white",
            font=make_font(12, bold=True), corner_radius=5,
            command=self._finish).pack(side="right", padx=10, pady=9)

        # Keyboard shortcuts
        self.bind("<Left>",  lambda e: self._prev())
        self.bind("<Right>", lambda e: self._next())
        self.bind("<k>",     lambda e: self._set_decision("keep"))
        self.bind("<r>",     lambda e: self._set_decision("remove"))

    # ── Candidate navigation ──────────────────────────────────

    def _show_candidate(self, idx: int) -> None:
        """Render the signal context and update info panel for candidate idx."""
        self._idx = max(0, min(idx, len(self._candidates) - 1))
        c   = self._candidates[self._idx]
        n   = len(self._candidates)
        fs  = self._fs
        sig = self._sig

        # ── Progress / counter ───────────────────────────────────
        self.progress.set((self._idx + 1) / n)
        self.lbl_counter.configure(
            text=f"{self._idx + 1} / {n}  "
                 f"({sum(1 for x in self._candidates if x['decision'] == 'remove')} to remove)")

        type_str = self._TYPE_LABEL.get(c["type"], c["type"])
        color    = self._TYPE_COLOR.get(c["type"], ORANGE)
        self.lbl_title.configure(
            text=f"Artifact #{self._idx + 1}  —  {c['type'].capitalize()}",
            text_color=color)
        self.lbl_type.configure(text=type_str, text_color=color)

        # ── Metric labels ─────────────────────────────────────────
        t_sec = c["sample"] / fs
        self._metric_labels["sample_t"].configure(text=f"{t_sec:.3f} s")
        self._metric_labels["rr_prev"].configure(
            text="—" if np.isnan(c["rr_prev_ms"])
            else f"{c['rr_prev_ms']:.1f} ms  ({60000/max(c['rr_prev_ms'],1):.0f} bpm)")
        self._metric_labels["rr_next"].configure(
            text="—" if np.isnan(c["rr_next_ms"])
            else f"{c['rr_next_ms']:.1f} ms  ({60000/max(c['rr_next_ms'],1):.0f} bpm)")
        self._metric_labels["rr_ref"].configure(
            text="—" if np.isnan(c["rr_ref_ms"])
            else f"{c['rr_ref_ms']:.1f} ms  ({60000/max(c['rr_ref_ms'],1):.0f} bpm)")
        self._metric_labels["deviation"].configure(
            text=f"{c['deviation']*100:.1f} %",
            text_color=RED if c["deviation"] > 0.3 else ORANGE)

        self._update_decision_ui(c["decision"])

        # ── Signal plot ───────────────────────────────────────────
        ctx_s   = 1.2            # seconds of context on each side
        ctx_smp = int(ctx_s * fs)
        samp    = c["sample"]
        lo_smp  = max(0, samp - ctx_smp)
        hi_smp  = min(len(sig) - 1, samp + ctx_smp)

        time_slice = np.arange(lo_smp, hi_smp) / fs
        sig_slice  = sig[lo_smp:hi_smp]

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(PLOT["axes"])
        self.fig.patch.set_facecolor(PLOT["bg"])

        # Signal trace
        ax.plot(time_slice, sig_slice, color=PLOT["signal"], lw=0.9, zorder=2)

        # Find rpeaks in window (excluding current)
        mask_win = (self._rpeaks >= lo_smp) & (self._rpeaks < hi_smp)
        nbr_peaks = self._rpeaks[mask_win]

        # Draw neighbour peaks (green)
        for rp in nbr_peaks:
            if rp != samp:
                ax.axvline(rp / fs, color=GREEN, lw=1.0, alpha=0.7, zorder=3)
                ax.scatter([rp / fs], [sig[rp]], color=GREEN, s=50,
                           zorder=4, marker="^")

        # Draw suspect peak (red)
        ax.axvline(samp / fs, color=color, lw=1.5, alpha=0.9, zorder=5)
        ax.scatter([samp / fs], [sig[samp]], color=color, s=100,
                   zorder=6, marker="^", label="Suspect beat")

        # RR interval annotations
        rp_arr = np.sort(nbr_peaks)
        idx_in_window = np.searchsorted(rp_arr, samp)
        if idx_in_window > 0:
            prev_rp = rp_arr[idx_in_window - 1]
            mid = (prev_rp + samp) / (2 * fs)
            y_ann = ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] != 0 else 0.9
            ax.annotate(f"{(samp-prev_rp)/fs*1000:.0f} ms",
                        xy=(mid, sig_slice.max() * 0.92),
                        ha="center", color=MUTED,
                        bbox=dict(boxstyle="round,pad=0.2", fc=CARD, ec="none", alpha=0.8))
        if idx_in_window < len(rp_arr) - 1:
            next_rp = rp_arr[idx_in_window + 1]
            mid = (samp + next_rp) / (2 * fs)
            ax.annotate(f"{(next_rp-samp)/fs*1000:.0f} ms",
                        xy=(mid, sig_slice.max() * 0.92),
                        ha="center", color=MUTED,
                        bbox=dict(boxstyle="round,pad=0.2", fc=CARD, ec="none", alpha=0.8))

        # Reference RR band (median ± 20%)
        if not np.isnan(c["rr_ref_ms"]):
            ref_s = c["rr_ref_ms"] / 1000
            lo_ref = samp / fs - ref_s * (1 + 0.20)
            hi_ref = samp / fs + ref_s * (1 + 0.20)
            ax.axvspan(lo_ref, hi_ref, color=BLUE, alpha=0.05, zorder=1)

        ax.set_xlabel("Time (s)", color=PLOT["muted"])
        ax.set_ylabel("Amplitude (norm.)", color=PLOT["muted"])
        ax.set_xlim(lo_smp / fs, hi_smp / fs)
        ax.tick_params(colors=PLOT["muted"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(PLOT["border"])
        ax.grid(True, color=PLOT["grid"], lw=0.5, alpha=0.6)

        title = f"Beat at {t_sec:.3f} s  —  {c['type']}  (deviation {c['deviation']*100:.1f}%)"
        ax.set_title(title, color=color, loc="left")

        self.canvas.draw_idle()
        self._update_summary_label()

    def _update_decision_ui(self, decision: str) -> None:
        if decision == "keep":
            self.lbl_decision.configure(text="✓  Keep", text_color=GREEN)
            self.btn_keep.configure(fg_color=GREEN)
            self.btn_remove.configure(fg_color=BORDER)
        else:
            self.lbl_decision.configure(text="✕  Remove", text_color=RED)
            self.btn_keep.configure(fg_color=BORDER)
            self.btn_remove.configure(fg_color=RED)

    def _update_summary_label(self) -> None:
        n_remove = sum(1 for c in self._candidates if c["decision"] == "remove")
        n_keep   = sum(1 for c in self._candidates if c["decision"] == "keep")
        self.lbl_summary.configure(
            text=f"  {n_remove} to remove  ·  {n_keep} kept")

    # ── Actions ───────────────────────────────────────────────

    def _set_decision(self, decision: str) -> None:
        self._candidates[self._idx]["decision"] = decision
        self._update_decision_ui(decision)
        self._update_summary_label()
        self.lbl_counter.configure(
            text=f"{self._idx + 1} / {len(self._candidates)}  "
                 f"({sum(1 for x in self._candidates if x['decision'] == 'remove')} to remove)")

    def _prev(self) -> None:
        if self._idx > 0:
            self._show_candidate(self._idx - 1)

    def _next(self) -> None:
        if self._idx < len(self._candidates) - 1:
            self._show_candidate(self._idx + 1)

    def _keep_all(self) -> None:
        for c in self._candidates:
            c["decision"] = "keep"
        self._show_candidate(self._idx)

    def _remove_all(self) -> None:
        for c in self._candidates:
            c["decision"] = "remove"
        self._show_candidate(self._idx)

    def _finish(self) -> None:
        self._result = self._candidates
        plt.close(self.fig)
        self.grab_release()
        self.destroy()

    def _on_close(self) -> None:
        """Window closed by user — treat as cancel (keep original decisions)."""
        self._result = None
        plt.close(self.fig)
        self.grab_release()
        self.destroy()

    def get_result(self) -> "list[dict] | None":
        """Return the reviewed candidate list, or None if dialog was cancelled."""
        return self._result


# ════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════

# ─── Interprétation ──────────────────────────────────────────

# Knowledge base: every parameter with its meaning, units, normal range (mouse)
# and interpretation thresholds.
# Format: {key: (label, unit, ref_lo, ref_hi, description, significance)}
class AnnotationDialog(ctk.CTkToplevel):
    """Modal dialog to add or edit a single time-period annotation.

    Fields: start time (s), end time (s), label text, colour picker.
    On OK, calls *on_save(ann_dict)* with the validated annotation dict.
    """

    def __init__(self, parent, on_save: "Callable[[dict], None]",
                 existing: "Optional[dict]" = None,
                 t_max: float = 1e6) -> None:
        super().__init__(parent)
        self.title("Add annotation" if existing is None else "Edit annotation")
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save
        self._t_max   = t_max
        self._color_var = tk.StringVar(value=(existing or {}).get("color", ANNOTATION_COLORS[0][0]))

        pad = dict(padx=12, pady=6)

        # ── Form grid ─────────────────────────────────────────────────────
        row = 0
        ctk.CTkLabel(self, text="Start (s):", anchor="e").grid(
            row=row, column=0, sticky="e", **pad)
        self._ent_start = ctk.CTkEntry(self, width=110)
        self._ent_start.insert(0, str((existing or {}).get("t_start", "")))
        self._ent_start.grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ctk.CTkLabel(self, text="End (s):", anchor="e").grid(
            row=row, column=0, sticky="e", **pad)
        self._ent_end = ctk.CTkEntry(self, width=110)
        self._ent_end.insert(0, str((existing or {}).get("t_end", "")))
        self._ent_end.grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ctk.CTkLabel(self, text="Label:", anchor="e").grid(
            row=row, column=0, sticky="e", **pad)
        self._ent_label = ctk.CTkEntry(self, width=220)
        self._ent_label.insert(0, (existing or {}).get("label", ""))
        self._ent_label.grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ctk.CTkLabel(self, text="Colour:", anchor="e").grid(
            row=row, column=0, sticky="e", **pad)
        color_frame = ctk.CTkFrame(self, fg_color="transparent")
        color_frame.grid(row=row, column=1, sticky="w", padx=12, pady=4)
        for hex_col, name in ANNOTATION_COLORS:
            rb = ctk.CTkRadioButton(
                color_frame, text=name, value=hex_col,
                variable=self._color_var,
                fg_color=hex_col, hover_color=hex_col,
                font=make_font(11))
            rb.pack(anchor="w", pady=1)

        row += 1
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=row, column=0, columnspan=2, pady=(4, 10))
        ctk.CTkButton(btn_frame, text="OK", width=90,
                      command=self._ok).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="Cancel", width=90,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      command=self.destroy).pack(side="left", padx=6)

        self._ent_start.focus_set()
        self.bind("<Return>", lambda _: self._ok())
        self.bind("<Escape>", lambda _: self.destroy())

    def _ok(self) -> None:
        try:
            t0 = float(self._ent_start.get())
            t1 = float(self._ent_end.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Start and end must be numbers (seconds).",
                                 parent=self)
            return
        if t0 >= t1:
            messagebox.showerror("Invalid range",
                                 "Start must be less than end.", parent=self)
            return
        if t0 < 0 or t1 > self._t_max + 1:
            messagebox.showwarning("Out of range",
                                   f"Times are outside the recording (0 – {self._t_max:.1f} s).",
                                   parent=self)
            return
        ann = {
            "t_start": round(t0, 4),
            "t_end":   round(t1, 4),
            "label":   self._ent_label.get().strip(),
            "color":   self._color_var.get(),
        }
        self._on_save(ann)
        self.destroy()



class AnnotationManagerDialog(ctk.CTkToplevel):
    """Dialog listing all annotations with Add / Edit / Delete buttons.

    Shows the full list in a scrollable table; changes are applied
    immediately to the parent app's _annotations list and the plots
    are redrawn after each change.
    """

    ROW_H = 36

    def __init__(self, parent: "ECGApp") -> None:
        super().__init__(parent)
        self.title("Annotations")
        self.geometry("660x420")
        self.resizable(True, True)
        self.grab_set()
        self._app = parent

        # ── Toolbar ───────────────────────────────────────────────────────
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="top", fill="x", padx=10, pady=(8, 4))
        ctk.CTkButton(bar, text="＋  Add annotation", width=140,
                      command=self._add).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="✏  Edit selected", width=130,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      command=self._edit).pack(side="left", padx=2)
        ctk.CTkButton(bar, text="🗑  Delete selected", width=140,
                      fg_color=RED, hover_color=RED_DARK, text_color="white",
                      command=self._delete).pack(side="left", padx=(6, 0))

        # ── Scrollable list ───────────────────────────────────────────────
        self._list_frame = ctk.CTkScrollableFrame(self, fg_color=CARD)
        self._list_frame.pack(side="top", fill="both", expand=True,
                              padx=10, pady=(0, 8))

        self._row_vars: "list[tk.BooleanVar]" = []
        self._refresh()

    def _refresh(self) -> None:
        for w in self._list_frame.winfo_children():
            w.destroy()
        self._row_vars = []

        anns = self._app._annotations
        if not anns:
            ctk.CTkLabel(self._list_frame, text="No annotations yet.",
                         text_color=MUTED).pack(padx=12, pady=20)
            return

        # Header
        hdr = ctk.CTkFrame(self._list_frame, fg_color=BORDER, height=28, corner_radius=4)
        hdr.pack(fill="x", padx=2, pady=(0, 2))
        for txt, w in [("", 30), ("Start (s)", 90), ("End (s)", 90),
                       ("Duration", 80), ("Label", 240), ("Colour", 80)]:
            ctk.CTkLabel(hdr, text=txt, width=w, font=make_font(10, bold=True),
                         text_color=TEXT).pack(side="left", padx=3)

        for i, ann in enumerate(anns):
            row = ctk.CTkFrame(self._list_frame,
                               fg_color=CARD if i % 2 == 0 else BG,
                               height=self.ROW_H, corner_radius=0)
            row.pack(fill="x", padx=2, pady=1)
            var = tk.BooleanVar(value=False)
            self._row_vars.append(var)
            ctk.CTkCheckBox(row, text="", variable=var, width=30,
                            checkbox_width=16, checkbox_height=16
                            ).pack(side="left", padx=(4, 0))
            dur = ann["t_end"] - ann["t_start"]
            for txt, w in [
                (f"{ann['t_start']:.3f}", 90),
                (f"{ann['t_end']:.3f}",   90),
                (f"{dur:.3f} s",           80),
                (ann.get("label", ""),    240),
            ]:
                ctk.CTkLabel(row, text=txt, width=w,
                             font=make_font(11), text_color=TEXT,
                             anchor="w").pack(side="left", padx=3)
            # Colour swatch
            swatch = tk.Label(row, bg=ann.get("color", GRAY),
                              width=4, relief="flat")
            swatch.pack(side="left", padx=(4, 2), ipady=8)

    def _add(self) -> None:
        t_max = float(self._app._time[-1]) if self._app._time is not None else 1e6
        def _save(ann: dict) -> None:
            self._app._annotations.append(ann)
            self._app._session_dirty = True
            self._app._draw_detail()
            self._app._update_ann_count()
            self._refresh()
        AnnotationDialog(self, on_save=_save, t_max=t_max)

    def _selected_indices(self) -> "list[int]":
        return [i for i, v in enumerate(self._row_vars) if v.get()]

    def _edit(self) -> None:
        sel = self._selected_indices()
        if not sel:
            messagebox.showinfo("No selection", "Tick one annotation to edit.", parent=self)
            return
        if len(sel) > 1:
            messagebox.showinfo("Multiple selected",
                                "Select exactly one annotation to edit.", parent=self)
            return
        idx = sel[0]
        existing = dict(self._app._annotations[idx])
        t_max = float(self._app._time[-1]) if self._app._time is not None else 1e6
        def _save(ann: dict) -> None:
            self._app._annotations[idx] = ann
            self._app._session_dirty = True
            self._app._draw_detail()
            self._app._update_ann_count()
            self._refresh()
        AnnotationDialog(self, on_save=_save, existing=existing, t_max=t_max)

    def _delete(self) -> None:
        sel = self._selected_indices()
        if not sel:
            messagebox.showinfo("No selection", "Tick annotation(s) to delete.", parent=self)
            return
        if not messagebox.askyesno("Confirm delete",
                                   f"Delete {len(sel)} annotation(s)?", parent=self):
            return
        for i in sorted(sel, reverse=True):
            del self._app._annotations[i]
        self._app._session_dirty = True
        self._app._draw_detail()
        self._app._update_ann_count()
        self._refresh()


# ════════════════════════════════════════════════════════════════════════
#  IntervalVerifierPanel — interactive beat-by-beat landmark checker
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
#  MLTrainingDialog — train/retrain the ML R-peak detector
# ════════════════════════════════════════════════════════════════════════

class MLTrainingDialog(ctk.CTkToplevel):
    """Modal dialog: shows verified-recording stats and trains the ML detector.

    Operates only on the pooled ml_training/*.npz cache and the persisted
    model (ecg.core.ml_detector) -- independent of whichever recording (if
    any) is currently open in the main window.
    """

    ROW_H = 30

    def __init__(self, parent):
        super().__init__(parent)
        self._app = parent
        self.title("ML R-Peak Detector")
        self.geometry("560x620")
        self.minsize(480, 440)
        self.resizable(True, True)
        self.configure(fg_color=BG)
        self.grab_set()
        self.focus_force()

        ctk.CTkLabel(self, text="🤖  ML R-Peak Detector", font=FONT_TITLE,
                     text_color=TEXT, anchor="w").pack(fill="x", padx=20, pady=(20, 4))
        ctk.CTkLabel(
            self, text="Trains a classifier from recordings marked "
                       "“Verified for training” (sidebar switch, "
                       "saved with the session). Verify a few recordings "
                       "with clean, corrected R-peaks, then train.",
            font=FONT_SMALL, text_color=MUTED, anchor="w", justify="left",
            wraplength=510).pack(fill="x", padx=20, pady=(0, 12))

        stats_card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=8,
                                   border_width=1, border_color=BORDER)
        stats_card.pack(fill="x", padx=20, pady=(0, 12))
        self._lbl_verified = ctk.CTkLabel(
            stats_card, text="", font=FONT_LABEL, text_color=TEXT,
            anchor="w", justify="left", wraplength=500)
        self._lbl_verified.pack(fill="x", padx=14, pady=(12, 4))
        self._lbl_model = ctk.CTkLabel(
            stats_card, text="", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left", wraplength=500)
        self._lbl_model.pack(fill="x", padx=14, pady=(0, 12))

        # ── Verified-file list ──────────────────────────────────────────
        ctk.CTkLabel(self, text="Verified files", font=FONT_LABEL,
                     text_color=TEXT, anchor="w").pack(fill="x", padx=20, pady=(0, 4))
        self._list_frame = ctk.CTkScrollableFrame(self, fg_color=CARD)
        self._list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        self._lbl_result = ctk.CTkLabel(
            self, text="", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left", wraplength=510)
        self._lbl_result.pack(fill="x", padx=20, pady=(0, 8))

        self.btn_train = ctk.CTkButton(
            self, text="▶  Train Model", command=self._train,
            fg_color=GREEN, hover_color=GREEN_DARK, text_color="white",
            font=FONT_BTN_PRIMARY, height=36, corner_radius=8)
        self.btn_train.pack(fill="x", padx=20, pady=(0, 8))

        ctk.CTkButton(
            self, text="Close", command=self.destroy,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            font=FONT_BTN_SEC, height=30, corner_radius=8,
        ).pack(fill="x", padx=20, pady=(0, 20))

        self._refresh_stats()

    def _refresh_stats(self) -> None:
        summary = training_data_summary()
        self._lbl_verified.configure(
            text=f"{summary['n_files']} verified recording(s)  —  "
                 f"{summary['n_samples']} labeled candidates "
                 f"({summary['n_positive']} R-peaks)")
        model = MLPeakModel.load()
        if model is not None:
            meta = model.meta
            self._lbl_model.configure(
                text=f"Current model — trained {str(meta.get('trained_at', '?'))[:19]} on "
                     f"{meta.get('n_training_files', '?')} file(s), "
                     f"{meta.get('n_training_samples', '?')} samples.  "
                     f"Hold-out accuracy={meta.get('holdout_accuracy', 0):.2f}, "
                     f"F1={meta.get('holdout_f1', 0):.2f}")
        else:
            self._lbl_model.configure(text="No model trained yet.")
        self._refresh_file_list()

    def _refresh_file_list(self) -> None:
        """Repopulate the scrollable list of cached training files.

        Cross-references each cached fingerprint against the sqlite
        registry (if available) to show a filename instead of a hash --
        the .npz cache itself has no notion of the original file path.
        """
        for w in self._list_frame.winfo_children():
            w.destroy()

        files = list_training_files()
        if not files:
            ctk.CTkLabel(self._list_frame, text="No verified recordings yet.",
                         text_color=MUTED).pack(padx=12, pady=16)
            return

        by_fingerprint: "dict[str, dict]" = {}
        if _DB_AVAILABLE:
            for row in verified_recordings(limit=1000):
                by_fingerprint[row.get("fingerprint", "")] = row

        for i, entry in enumerate(files):
            fp = entry["fingerprint"]
            row_data = by_fingerprint.get(fp)
            label = Path(row_data["filepath"]).name if row_data else f"Unknown file ({fp[:12]}…)"
            filepath = row_data["filepath"] if row_data else None

            row = ctk.CTkFrame(self._list_frame,
                               fg_color=CARD if i % 2 == 0 else BG,
                               height=self.ROW_H, corner_radius=0)
            row.pack(fill="x", padx=2, pady=1)
            ctk.CTkLabel(row, text=label, font=make_font(11), text_color=TEXT,
                         anchor="w").pack(side="left", padx=(6, 4), fill="x", expand=True)
            ctk.CTkLabel(row, text=f"{entry['n_samples']} samples "
                                    f"({entry['n_positive']} peaks)",
                         font=make_font(10), text_color=MUTED, width=140,
                         anchor="e").pack(side="left", padx=4)
            ctk.CTkButton(
                row, text="🗑", width=28, height=22,
                fg_color=RED, hover_color=RED_DARK, text_color="white",
                font=make_font(10),
                command=lambda fp=fp, filepath=filepath: self._remove_file(fp, filepath),
            ).pack(side="left", padx=(4, 6))

    def _remove_file(self, fingerprint: str, filepath: "Optional[str]") -> None:
        """Un-verify one cached recording: drop its .npz cache and DB flag.

        Does not retrain automatically -- the removed file simply won't be
        included next time "Train Model" is clicked.
        """
        delete_training_sample(fingerprint)
        if filepath and _DB_AVAILABLE:
            set_verified(filepath, False)
        # If the removed file happens to be the one currently open in the
        # main window, keep its sidebar switch in sync too.
        app = self._app
        if filepath and getattr(app, "signal", None) is not None and app.signal.filepath == filepath:
            app.session.verified_for_training = False
            app.session_ctrl.sync_verified_switch()
        self._refresh_stats()

    def _train(self) -> None:
        self.btn_train.configure(state="disabled", text="Training…")  # type: ignore[union-attr]
        self.update_idletasks()
        try:
            result = train_model()
        finally:
            self.btn_train.configure(state="normal", text="▶  Train Model")  # type: ignore[union-attr]
        if result.get("ok"):
            self._lbl_result.configure(
                text=f"✓ {result['message']}  (accuracy={result['accuracy']:.2f}, "
                     f"F1={result['f1']:.2f})",
                text_color=GREEN)
        else:
            self._lbl_result.configure(text=f"✗ {result['message']}", text_color=ORANGE)
        self._refresh_stats()

