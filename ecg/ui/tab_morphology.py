# -*- coding: utf-8 -*-
"""
ecg.ui.tab_morphology
---------------------
ECGApp Intervals and Beat Template tabs.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from ecg.core.models import QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.ui.theme import (
    THEME, BG, PANEL, BORDER, BORDER2, TEXT, MUTED, PURPLE, PURPLE_DARK,
    SPACE_S, SPACE_M, SPACE_L, FONT_LABEL, FONT_SMALL, FONT_BTN_PRIMARY,
    FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT,
)
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_INTERVALS, TAB_BEAT_TEMPLATE

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class TabMorphologyMixin:
    """ECGApp Intervals and Beat Template tabs."""

    def _build_tab_intervals(self: ECGApp) -> None:
        """Interval delineation tab: P/Q/R/S/T measurement per beat."""
        t = self.tabs.tab(TAB_INTERVALS)

        # ── Action bar ─────────────────────────────────────────────────────
        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.pack(side="top", fill="x", padx=SPACE_M, pady=(SPACE_M, SPACE_S))

        # Built with an explicit muted fg_color (see _set_btn_enabled) rather
        # than relying on CTkButton's default disabled treatment, which only
        # dims the text and left this looking clickable while disabled --
        # active fill (_RESULT_BTN_COLORS) is BLUE, not ORANGE, since ORANGE
        # is this app's warning/Edit-Mode-ON colour elsewhere and this only
        # computes measurements, it isn't a warning state.
        self.btn_run_ivl = ctk.CTkButton(
            bar, text="⚡  Measure intervals",
            command=self._run_intervals,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            font=FONT_BTN_PRIMARY, height=max(28, int(32 * THEME.font_scale)),
            corner_radius=8, state="disabled",
        )
        self.btn_run_ivl.pack(side="left")

        self.lbl_ivl_status = ctk.CTkLabel(
            bar, text="  Click Analyze first",
            font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_ivl_status.pack(side="left", padx=SPACE_M)

        # QTc formula selector -- combobox packed (side="right") BEFORE its
        # label so the label lands to the combobox's left, not its right;
        # packing them in the opposite order (as before) puts "QTc formula:"
        # to the right of the dropdown it names, reading as if it labelled
        # whatever comes after it instead.
        # Values come from QTC_FORMULA_LABELS (single source shared with
        # normalize_qtc_formula, session save/restore and the exports);
        # wide enough for the longest label ("Fridericia (QT/∛RR, human)").
        self.cb_qtc_formula = ctk.CTkComboBox(
            bar, width=int(215 * THEME.font_scale), height=28, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, button_color=BORDER2,
            text_color=TEXT, dropdown_fg_color=BG, dropdown_text_color=TEXT,
            values=list(QTC_FORMULA_LABELS.values()),
            command=self._on_qtc_formula_change,
        )
        # The formula in effect lives in app state (analysis.qtc_formula),
        # so a rebuild (theme toggle) re-selects it instead of silently
        # resetting the selector to Mitchell while the stored QTc values
        # (and the exports' formula label) still use another formula.
        self.cb_qtc_formula.set(QTC_FORMULA_LABELS[normalize_qtc_formula(
            getattr(getattr(self, "analysis", None), "qtc_formula", "mitchell"))])
        self.cb_qtc_formula.pack(side="right", padx=(0, SPACE_M))
        ctk.CTkLabel(bar, text="QTc formula:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="right", padx=(0, SPACE_S))

        # Permissive bounds toggle
        self.sw_permissive = self._switch(bar, "Permissive bounds", dict(padx=0))
        self.sw_permissive.pack(side="right", padx=(0, SPACE_L))
        self._bind_hover_tip(
            self.sw_permissive,
            "Widen the physiological PR/QRS/QT bounds ×2 before rejecting a "
            "beat's measurement — trade some specificity for more beats measured.")

        # ── Body: PR/QRS/QT/QTc distribution violin plots, full width ────────
        body = tk.Frame(t, bg=BG, bd=0, highlightthickness=0)
        body.pack(side="top", fill="both", expand=True, padx=SPACE_S, pady=(0, SPACE_S))

        self._slots["intervals"] = CanvasSlot(
            body, 14, 6, toolbar=False,
            placeholder="Click ⚡ Measure intervals to display")


    def _build_tab_beat_template(self: ECGApp) -> None:
        """Beat template tab: mean beat ± SD and morphology distributions."""
        t = self.tabs.tab(TAB_BEAT_TEMPLATE)
        t.grid_rowconfigure(1, weight=3)
        t.grid_rowconfigure(2, weight=2)
        t.grid_columnconfigure(0, weight=1)

        # ── Header bar ──────────────────────────────────────────────────────
        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_M, SPACE_S))
        ctk.CTkLabel(bar, text="BEAT TEMPLATE  —  mean ± SD  ·  morphology",
                     font=FONT_SIDEBAR_HDR, text_color=MUTED).pack(side="left")
        btn_landmarks = ctk.CTkButton(
            bar, text="⚙  Landmarks",
            command=self._open_wave_template_editor,
            fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
            font=FONT_BTN_SEC, height=28, corner_radius=6,
        )
        btn_landmarks.pack(side="right")
        self._bind_hover_tip(
            btn_landmarks,
            "Edit where P, QRS onset, Q/R/S, the J point and T are searched "
            "for on every beat (PR = P onset → QRS onset, QRS = QRS onset → "
            "J point, QT = QRS onset → T end). The template belongs to this "
            "recording (kept in its session); only “Save as default "
            "template” changes the default for recordings opened later.")
        self.lbl_template_info = ctk.CTkLabel(
            bar, text="", font=FONT_HINT, text_color=MUTED, anchor="e")
        self.lbl_template_info.pack(side="right", padx=(0, SPACE_M))  # type: ignore[union-attr]

        # ── Mean beat plot ──────────────────────────────────────────────────
        beat_card = ctk.CTkFrame(t, fg_color=PANEL, corner_radius=0)
        beat_card.grid(row=1, column=0, sticky="nsew", padx=SPACE_S, pady=(0, SPACE_S))
        beat_card.grid_rowconfigure(0, weight=1)
        beat_card.grid_columnconfigure(0, weight=1)
        self._slots["beat"] = CanvasSlot(beat_card, 14, 5.0, toolbar=False)

        # ── Distributions: amplitude + correlation ──────────────────────────
        dist_card = ctk.CTkFrame(t, fg_color=PANEL, corner_radius=0)
        dist_card.grid(row=2, column=0, sticky="nsew", padx=SPACE_S, pady=(0, SPACE_S))
        dist_card.grid_rowconfigure(0, weight=1)
        dist_card.grid_columnconfigure(0, weight=1)
        self._slots["beat_dist"] = CanvasSlot(dist_card, 14, 4.0, toolbar=False)


    # ════════════════════════════════════════════════════════
    #  QTC FORMULA
    # ════════════════════════════════════════════════════════

    def _qtc_formula(self: ECGApp) -> str:
        """Return the QTc formula key in effect ('mitchell', 'bazett' or
        'fridericia' — see QTC_FORMULA_LABELS): analysis.qtc_formula, set by
        the Intervals tab combo, which survives a UI rebuild."""
        return self.analysis_ctrl.qtc_formula()

    def _on_qtc_formula_change(self: ECGApp, _choice: str = "") -> None:
        """Re-compute QTc with the selected formula and refresh interval plots."""
        self.analysis_ctrl.on_qtc_formula_change(_choice)

    def _update_interval_plots(self: ECGApp) -> None:
        """Redraw the interval violin plots after a QTc formula change."""
        self.analysis_ctrl.update_interval_plots()
