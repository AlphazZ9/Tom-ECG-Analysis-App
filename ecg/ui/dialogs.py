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
    BLUE_HOVER, RED_DARK, ORANGE_DARK, ORANGE_DEEP,
    GRAY, GRAY_LIGHT, ARTIFACT_TYPE_COLOR,
    FONT_TITLE, FONT_SECTION_HDR, FONT_LABEL, FONT_SMALL,
    FONT_BODY, FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR,
    FONT_HINT, FONT_KPI_LABEL, FONT_CARD_TITLE, FONT_SUBSECTION,
    FONT_DIALOG_TITLE,
)
from ecg.core.models import (
    MouseECG, ContextRanges, EXPERIMENTAL_CONTEXTS, save_custom_context,
)
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
    """Appearance settings panel.

    Layout
    ------
    Top   : title bar + subtitle
    Left  : the 2 preset cards (Light / Dark), each showing a mini colour
            swatch strip (BG / PANEL / accent) + name + description
    Right : font size slider
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
        self._working.apply_preset(THEME.preset_name)
        self._working.font_scale = THEME.font_scale

        self._saved = False

        self._build()
        self._highlight_active_preset()

    # ── UI construction ─────────────────────────────────────────────────────

    def _build(self) -> None:
        self.configure(fg_color=PANEL)

        # Title bar
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Appearance", font=FONT_DIALOG_TITLE,
                     text_color=TEXT, anchor="w").pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(hdr, text="Choose Light or Dark",
                     font=FONT_SMALL, text_color=MUTED, anchor="w").pack(
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
                     font=FONT_SIDEBAR_HDR, text_color=MUTED,
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
        ctk.CTkLabel(name_row, text=name, font=FONT_CARD_TITLE,
                     text_color=TEXT, anchor="w").pack(side="left")
        badge_col  = "#2E3440" if is_dark else "#E8ECF2"
        badge_text = "dark" if is_dark else "light"
        badge_fc   = "#ECEFF4" if is_dark else "#555F6E"
        ctk.CTkLabel(name_row,
                     text=f"  {badge_text}  ",
                     font=FONT_KPI_LABEL, fg_color=badge_col, text_color=badge_fc,
                     corner_radius=4).pack(side="left", padx=(6, 0))

        # Description
        ctk.CTkLabel(outer, text=desc, font=FONT_HINT, text_color=muted_c,
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

        # ── Font size slider ──────────────────────────────────────────────────
        # Font family is not user-configurable -- the app is locked to one
        # fixed system font (_detect_system_font()) for uniformity. Size
        # stays adjustable since that's an accessibility need, not a style
        # choice.
        ctk.CTkLabel(right, text="FONT SIZE", font=FONT_SIDEBAR_HDR,
                     text_color=MUTED, anchor="w").pack(fill="x", padx=8, pady=(4, 4))

        size_row = ctk.CTkFrame(right, fg_color="transparent")
        size_row.pack(fill="x", padx=8, pady=(0, 2))
        ctk.CTkLabel(size_row, text="A", font=FONT_KPI_LABEL, text_color=MUTED).pack(side="left")

        self._size_label = ctk.CTkLabel(size_row, text=f"{self._working.font_scale:.2f}x",
                                         font=FONT_SMALL, text_color=TEXT, width=48)
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

    def _build_bottom_bar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=56)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        ctk.CTkFrame(bar, height=1, fg_color=BORDER).pack(fill="x", side="top")

        ctk.CTkButton(bar, text="Close",
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FONT_SMALL, width=90, height=34,
                      command=self.destroy).pack(side="right", padx=12, pady=10)
        ctk.CTkButton(bar, text="Save as Default",
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      font=FONT_SIDEBAR_HDR, width=130, height=34,
                      command=self._save_default).pack(side="right", padx=(0, 6), pady=10)
        ctk.CTkButton(bar, text="Apply",
                      fg_color=BLUE, hover_color=TEXT, text_color="white",
                      font=FONT_BTN_PRIMARY, width=100, height=34,
                      command=self._apply).pack(side="right", padx=(0, 6), pady=10)

        self._preview_lbl = ctk.CTkLabel(
            bar,
            text=f"Active: {THEME.preset_name}  |  scale: {THEME.font_scale:.2f}x",
            font=FONT_HINT, text_color=MUTED, anchor="w")
        self._preview_lbl.pack(side="left", padx=16)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _highlight_active_preset(self) -> None:
        active = self._working.preset_name
        for name, card in self._preset_btns.items():
            card.configure(border_color=BLUE if name == active else BORDER,
                           border_width=3 if name == active else 2)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _select_preset(self, name: str) -> None:
        self._working.apply_preset(name)
        self._highlight_active_preset()
        self._update_preview_label()

    def _on_scale_change(self, val: float) -> None:
        self._working.font_scale = round(float(val), 2)
        self._size_label.configure(text=f"{self._working.font_scale:.2f}x")
        self._update_preview_label()

    def _update_preview_label(self) -> None:
        self._preview_lbl.configure(
            text=f"Preview: {self._working.preset_name}  |  "
                 f"scale: {self._working.font_scale:.2f}x")

    def _apply(self) -> None:
        """Apply the working config to the live app (rebuild UI).

        _rebuild_ui() calls ctk.set_appearance_mode() which invalidates
        CTk widget handles inside this dialog.  We therefore close the
        dialog first, then rebuild the main window.
        """
        global THEME
        THEME.apply_preset(self._working.preset_name)
        THEME.font_scale = self._working.font_scale
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
        self.configure(fg_color=PANEL)
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
            hdr, text="", font=FONT_BTN_PRIMARY, text_color=TEXT)
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
        plot_card = ctk.CTkFrame(content, fg_color=PANEL, corner_radius=6)
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

        ctk.CTkLabel(info, text="ARTIFACT INFO", font=FONT_SUBSECTION,
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
            lbl = ctk.CTkLabel(info, text="—", font=FONT_SIDEBAR_HDR,
                                text_color=TEXT, anchor="w")
            lbl.grid(row=row_i + 1, column=0, sticky="w", padx=20, pady=(0, 4))
            self._metric_labels[key] = lbl

        # Decision indicator
        ctk.CTkFrame(info, height=1, fg_color=BORDER).grid(
            row=13, column=0, sticky="ew", padx=12, pady=8)
        ctk.CTkLabel(info, text="DECISION", font=FONT_SUBSECTION,
                     text_color=MUTED).grid(row=14, column=0, sticky="w",
                                             padx=12, pady=(0, 4))
        self.lbl_decision = ctk.CTkLabel(info, text="Remove",
                                          font=FONT_BTN_PRIMARY,
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
            font=FONT_CARD_TITLE, corner_radius=5,
            command=lambda: self._set_decision("keep"))
        self.btn_keep.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_remove = ctk.CTkButton(
            btn_frame, text="✕  Remove", height=36,
            fg_color=RED_DARK, hover_color="#D32F2F", text_color="white",
            font=FONT_CARD_TITLE, corner_radius=5,
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
            font=FONT_CARD_TITLE, corner_radius=5,
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
        color    = ARTIFACT_TYPE_COLOR.get(c["type"], ORANGE)
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
                font=FONT_SMALL)
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
            ctk.CTkLabel(hdr, text=txt, width=w, font=FONT_SUBSECTION,
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
                             font=FONT_SMALL, text_color=TEXT,
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


class PacingPeriodDialog(ctk.CTkToplevel):
    """Modal dialog to add or edit a single pacing/stimulation period.

    Fields: start time (s), end time (s), note text. No colour picker --
    every pacing period shares one fixed render colour (see draw_detail()).
    On OK, calls *on_save(period_dict)* with the validated period dict.
    """

    def __init__(self, parent, on_save: "Callable[[dict], None]",
                 existing: "Optional[dict]" = None,
                 t_max: float = 1e6) -> None:
        super().__init__(parent)
        self.title("Add pacing period" if existing is None else "Edit pacing period")
        self.resizable(False, False)
        self.grab_set()
        self._on_save = on_save
        self._t_max   = t_max

        pad = dict(padx=12, pady=6)

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
        ctk.CTkLabel(self, text="Note:", anchor="e").grid(
            row=row, column=0, sticky="e", **pad)
        self._ent_note = ctk.CTkEntry(self, width=220)
        self._ent_note.insert(0, (existing or {}).get("note", ""))
        self._ent_note.grid(row=row, column=1, sticky="w", **pad)

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
        period = {
            "t_start": round(t0, 4),
            "t_end":   round(t1, 4),
            "note":    self._ent_note.get().strip(),
        }
        self._on_save(period)
        self.destroy()


class PacingPeriodManagerDialog(ctk.CTkToplevel):
    """Dialog listing all pacing/stimulation periods with Add/Edit/Delete.

    Shows the full list in a scrollable table; changes are applied
    immediately to the parent app's _pacing_periods list and the plots
    are redrawn after each change.
    """

    ROW_H = 36

    def __init__(self, parent: "ECGApp") -> None:
        super().__init__(parent)
        self.title("Pacing Periods")
        self.geometry("580x420")
        self.resizable(True, True)
        self.grab_set()
        self._app = parent

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="top", fill="x", padx=10, pady=(8, 4))
        ctk.CTkButton(bar, text="＋  Add pacing period", width=150,
                      command=self._add).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="✏  Edit selected", width=130,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      command=self._edit).pack(side="left", padx=2)
        ctk.CTkButton(bar, text="🗑  Delete selected", width=140,
                      fg_color=RED, hover_color=RED_DARK, text_color="white",
                      command=self._delete).pack(side="left", padx=(6, 0))

        self._list_frame = ctk.CTkScrollableFrame(self, fg_color=CARD)
        self._list_frame.pack(side="top", fill="both", expand=True,
                              padx=10, pady=(0, 8))

        self._row_vars: "list[tk.BooleanVar]" = []
        self._refresh()

    def _refresh(self) -> None:
        for w in self._list_frame.winfo_children():
            w.destroy()
        self._row_vars = []

        periods = self._app._pacing_periods
        if not periods:
            ctk.CTkLabel(self._list_frame, text="No pacing periods yet.",
                         text_color=MUTED).pack(padx=12, pady=20)
            return

        hdr = ctk.CTkFrame(self._list_frame, fg_color=BORDER, height=28, corner_radius=4)
        hdr.pack(fill="x", padx=2, pady=(0, 2))
        for txt, w in [("", 30), ("Start (s)", 90), ("End (s)", 90),
                       ("Duration", 80), ("Note", 240)]:
            ctk.CTkLabel(hdr, text=txt, width=w, font=FONT_SUBSECTION,
                         text_color=TEXT).pack(side="left", padx=3)

        for i, pp in enumerate(periods):
            row = ctk.CTkFrame(self._list_frame,
                               fg_color=CARD if i % 2 == 0 else BG,
                               height=self.ROW_H, corner_radius=0)
            row.pack(fill="x", padx=2, pady=1)
            var = tk.BooleanVar(value=False)
            self._row_vars.append(var)
            ctk.CTkCheckBox(row, text="", variable=var, width=30,
                            checkbox_width=16, checkbox_height=16
                            ).pack(side="left", padx=(4, 0))
            dur = pp["t_end"] - pp["t_start"]
            for txt, w in [
                (f"{pp['t_start']:.3f}", 90),
                (f"{pp['t_end']:.3f}",   90),
                (f"{dur:.3f} s",          80),
                (pp.get("note", ""),    240),
            ]:
                ctk.CTkLabel(row, text=txt, width=w,
                             font=FONT_SMALL, text_color=TEXT,
                             anchor="w").pack(side="left", padx=3)

    def _add(self) -> None:
        t_max = float(self._app._time[-1]) if self._app._time is not None else 1e6
        def _save(period: dict) -> None:
            self._app._pacing_periods.append(period)
            self._app._session_dirty = True
            self._app._draw_detail()
            self._app._update_pacing_count()
            self._refresh()
        PacingPeriodDialog(self, on_save=_save, t_max=t_max)

    def _selected_indices(self) -> "list[int]":
        return [i for i, v in enumerate(self._row_vars) if v.get()]

    def _edit(self) -> None:
        sel = self._selected_indices()
        if not sel:
            messagebox.showinfo("No selection", "Tick one pacing period to edit.", parent=self)
            return
        if len(sel) > 1:
            messagebox.showinfo("Multiple selected",
                                "Select exactly one pacing period to edit.", parent=self)
            return
        idx = sel[0]
        existing = dict(self._app._pacing_periods[idx])
        t_max = float(self._app._time[-1]) if self._app._time is not None else 1e6
        def _save(period: dict) -> None:
            self._app._pacing_periods[idx] = period
            self._app._session_dirty = True
            self._app._draw_detail()
            self._app._update_pacing_count()
            self._refresh()
        PacingPeriodDialog(self, on_save=_save, existing=existing, t_max=t_max)

    def _delete(self) -> None:
        sel = self._selected_indices()
        if not sel:
            messagebox.showinfo("No selection", "Tick pacing period(s) to delete.", parent=self)
            return
        if not messagebox.askyesno("Confirm delete",
                                   f"Delete {len(sel)} pacing period(s)?", parent=self):
            return
        for i in sorted(sel, reverse=True):
            del self._app._pacing_periods[i]
        self._app._session_dirty = True
        self._app._draw_detail()
        self._app._update_pacing_count()
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

        # ── Hold-out confusion matrix (hidden until a trained model with
        # this data exists -- older models saved before this feature won't
        # have it in their meta) ──────────────────────────────────────────
        self._cm_frame = ctk.CTkFrame(self, fg_color=CARD, corner_radius=8,
                                       border_width=1, border_color=BORDER)
        self._cm_fig = Figure(figsize=(2.6, 2.0), dpi=90,
                              facecolor=PLOT["bg"], tight_layout=True)
        self._cm_canvas = FigureCanvasTkAgg(self._cm_fig, master=self._cm_frame)
        self._cm_canvas.get_tk_widget().pack(side="left", padx=(10, 4), pady=10)
        self._lbl_cm_metrics = ctk.CTkLabel(
            self._cm_frame, text="", font=FONT_SMALL, text_color=TEXT,
            anchor="w", justify="left")
        self._lbl_cm_metrics.pack(side="left", fill="x", expand=True,
                                   padx=(4, 10), pady=10)
        # Not packed yet -- _maybe_show_confusion_matrix() packs/unpacks it.

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
            self._maybe_show_confusion_matrix(
                meta.get("holdout_confusion_matrix"),
                meta.get("holdout_precision", 0.0), meta.get("holdout_recall", 0.0),
                meta.get("holdout_f1", 0.0), meta.get("holdout_n_test", 0))
        else:
            self._lbl_model.configure(text="No model trained yet.")
            self._maybe_show_confusion_matrix(None, 0.0, 0.0, 0.0, 0)
        self._refresh_file_list()

    def _maybe_show_confusion_matrix(
        self, cm: "Optional[list]", precision: float, recall: float,
        f1: float, n_test: int,
    ) -> None:
        """Render the hold-out confusion matrix, or hide the panel if absent
        (older saved models predate this field)."""
        if not cm:
            self._cm_frame.pack_forget()
            return
        cm_arr = np.asarray(cm, dtype=int)
        ax = self._cm_fig.axes[0] if self._cm_fig.axes else self._cm_fig.add_subplot(111)
        ax.clear()
        ax.imshow(cm_arr, cmap="Blues", vmin=0)
        labels = ["Non-peak", "Peak"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7, color=PLOT.get("muted", MUTED))
        ax.set_ylabel("Actual", fontsize=7, color=PLOT.get("muted", MUTED))
        vmax = max(cm_arr.max(), 1)
        for r in range(2):
            for c in range(2):
                v = int(cm_arr[r, c])
                ax.text(c, r, str(v), ha="center", va="center", fontsize=10,
                        color="white" if v > vmax * 0.5 else "black")
        ax.tick_params(colors=PLOT.get("muted", MUTED))
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._cm_fig.suptitle("Hold-out confusion matrix", fontsize=8,
                              color=PLOT.get("text", TEXT))
        self._cm_canvas.draw()

        self._lbl_cm_metrics.configure(
            text=f"Precision  {precision:.2f}\n"
                 f"Recall      {recall:.2f}\n"
                 f"F1          {f1:.2f}\n"
                 f"n (hold-out)  {n_test}")
        # before= is required: by the time this runs (end of __init__, or a
        # later _refresh_stats() call), btn_train/Close are already packed,
        # so a bare .pack() would append this AFTER them instead of in its
        # intended spot between the result label and the Train button.
        self._cm_frame.pack(fill="x", padx=20, pady=(0, 8), before=self.btn_train)

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
            ctk.CTkLabel(row, text=label, font=FONT_SMALL, text_color=TEXT,
                         anchor="w").pack(side="left", padx=(6, 4), fill="x", expand=True)
            ctk.CTkLabel(row, text=f"{entry['n_samples']} samples "
                                    f"({entry['n_positive']} peaks)",
                         font=FONT_HINT, text_color=MUTED, width=140,
                         anchor="e").pack(side="left", padx=4)
            ctk.CTkButton(
                row, text="🗑", width=28, height=22,
                fg_color=RED, hover_color=RED_DARK, text_color="white",
                font=FONT_HINT,
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


class CustomContextDialog(ctk.CTkToplevel):
    """Modal dialog: define/edit a user-owned "custom" experimental context.

    EXPERIMENTAL_CONTEXTS (ecg.core.models) ships 4 fixed mouse contexts
    with no user-editable option -- e.g. for a specific strain/substrain
    whose normal ranges differ from those 4 presets. Saving here plugs the
    result into that same dict under the "custom" key, so every panel that
    already reads reference ranges via app._current_ref() picks it up with
    no further wiring once "Custom" is selected in the Parameters dialog.
    """

    FIELDS = [
        ("hr",   "Heart rate",        "bpm"),
        ("rr",   "RR interval",       "ms"),
        ("sdnn", "SDNN",              "ms"),
        ("rmssd","RMSSD",             "ms"),
        ("pnn6", "pNN6",              "%"),
        ("lf",   "LF power",          "%"),
        ("hf",   "HF power",          "%"),
        ("lfhf", "LF/HF ratio",       ""),
        ("sd1",  "Poincaré SD1",      "ms"),
        ("sd2",  "Poincaré SD2",      "ms"),
        ("pr",   "PR interval",       "ms"),
        ("qrs",  "QRS duration",      "ms"),
        ("qt",   "QT interval",       "ms"),
        ("qtc",  "QTc (corrected)",   "ms"),
    ]

    def __init__(self, parent, on_saved: "Optional[Callable[[], None]]" = None):
        super().__init__(parent)
        self._on_saved = on_saved
        self.title("🧪  Custom Experimental Context")
        self.geometry("480x680")
        self.minsize(420, 400)
        self.resizable(True, True)
        self.configure(fg_color=BG)
        self.grab_set()
        self.focus_force()

        # Start from whatever "custom" context already exists (a previous
        # save), else from telemetry_awake as a reasonable starting point --
        # never fabricated from nothing.
        base = EXPERIMENTAL_CONTEXTS.get("custom") or EXPERIMENTAL_CONTEXTS["telemetry_awake"]

        ctk.CTkLabel(self, text="🧪  Custom Experimental Context", font=FONT_TITLE,
                     text_color=TEXT, anchor="w").pack(fill="x", padx=20, pady=(20, 4))
        ctk.CTkLabel(
            self, text="Define your own reference ranges (e.g. for a specific "
                       "strain). Once saved, select \"Custom\" in the Parameters "
                       "dialog to use these bounds everywhere the app shows a "
                       "reference band.",
            font=FONT_SMALL, text_color=MUTED, anchor="w", justify="left",
            wraplength=430).pack(fill="x", padx=20, pady=(0, 10))

        name_row = ctk.CTkFrame(self, fg_color="transparent")
        name_row.pack(fill="x", padx=20, pady=(0, 10))
        name_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(name_row, text="Name", font=FONT_SMALL, text_color=MUTED,
                     width=110, anchor="w").grid(row=0, column=0, sticky="w")
        self._ent_name = ctk.CTkEntry(name_row, font=FONT_LABEL, height=28,
                                      fg_color=CARD, border_color=BORDER2, text_color=TEXT)
        self._ent_name.insert(0, base.label if base.label != "Telemetry — awake mouse" else "Custom")
        self._ent_name.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        scroll = ctk.CTkScrollableFrame(self, fg_color=CARD, corner_radius=8,
                                        scrollbar_button_color=BORDER,
                                        scrollbar_button_hover_color=BORDER2)
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        scroll.columnconfigure(1, weight=1)
        scroll.columnconfigure(2, weight=1)

        hdr = ctk.CTkFrame(scroll, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(2, 6))
        hdr.columnconfigure(1, weight=1); hdr.columnconfigure(2, weight=1)
        ctk.CTkLabel(hdr, text="", width=118).grid(row=0, column=0)
        ctk.CTkLabel(hdr, text="Low", font=FONT_HINT, text_color=MUTED).grid(row=0, column=1)
        ctk.CTkLabel(hdr, text="High", font=FONT_HINT, text_color=MUTED).grid(row=0, column=2)

        self._entries: "dict[str, tuple[ctk.CTkEntry, ctk.CTkEntry]]" = {}
        for i, (attr, label, unit) in enumerate(self.FIELDS, start=1):
            lbl_txt = f"{label}  ({unit})" if unit else label
            ctk.CTkLabel(scroll, text=lbl_txt, font=FONT_SMALL, text_color=TEXT,
                        anchor="w", width=140, wraplength=138, justify="left"
                        ).grid(row=i, column=0, sticky="w", padx=(4, 4), pady=3)
            e_lo = ctk.CTkEntry(scroll, font=FONT_LABEL, height=26,
                                fg_color=BG, border_color=BORDER2, text_color=TEXT)
            e_lo.insert(0, str(getattr(base, f"{attr}_lo")))
            e_lo.grid(row=i, column=1, sticky="ew", padx=3, pady=3)
            e_hi = ctk.CTkEntry(scroll, font=FONT_LABEL, height=26,
                                fg_color=BG, border_color=BORDER2, text_color=TEXT)
            e_hi.insert(0, str(getattr(base, f"{attr}_hi")))
            e_hi.grid(row=i, column=2, sticky="ew", padx=3, pady=3)
            self._entries[attr] = (e_lo, e_hi)

        self._lbl_error = ctk.CTkLabel(self, text="", font=FONT_SMALL,
                                       text_color=ORANGE, anchor="w",
                                       wraplength=430, justify="left")
        self._lbl_error.pack(fill="x", padx=20, pady=(0, 4))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkButton(btn_row, text="Cancel", command=self.destroy,
                     fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                     font=FONT_BTN_SEC, height=32, corner_radius=8
                     ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(btn_row, text="✔  Save", command=self._save,
                     fg_color=GREEN, hover_color=GREEN_DARK, text_color="white",
                     font=FONT_BTN_PRIMARY, height=32, corner_radius=8
                     ).pack(side="left", fill="x", expand=True, padx=(6, 0))

    def _save(self) -> None:
        values: "dict[str, float]" = {}
        for attr, label, _unit in self.FIELDS:
            e_lo, e_hi = self._entries[attr]
            try:
                lo = float(e_lo.get()); hi = float(e_hi.get())
            except ValueError:
                self._lbl_error.configure(text=f"⚠ {label}: enter numeric values.")
                return
            if lo >= hi:
                self._lbl_error.configure(text=f"⚠ {label}: low must be less than high.")
                return
            values[f"{attr}_lo"] = lo
            values[f"{attr}_hi"] = hi

        name = self._ent_name.get().strip() or "Custom"
        ctx = ContextRanges(label=name, description="User-defined custom context.",
                            **values)
        try:
            save_custom_context(ctx)
        except Exception as exc:
            self._lbl_error.configure(text=f"⚠ Could not save: {exc}")
            return
        EXPERIMENTAL_CONTEXTS["custom"] = ctx
        if self._on_saved is not None:
            self._on_saved()
        self.destroy()

