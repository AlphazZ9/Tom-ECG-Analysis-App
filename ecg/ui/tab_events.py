# -*- coding: utf-8 -*-
"""
ecg.ui.tab_events
-----------------
ECGApp Events tab: baseline and thresholds with what the events rest on, the ribbon
with the RR series, event cards, ECG zoom with Previous, Next and Edit peaks here.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from ecg.ui.freshness import stale_banner_text
from ecg.ui.theme import (
    THEME, BG, PANEL, BORDER, BORDER2, TEXT, MUTED, PLOT, BLUE, ORANGE,
    BLUE_HOVER, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_LABEL, FONT_SMALL,
    FONT_KPI_LABEL, FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT,
    FONT_SUBSECTION,
)
from ecg.ui.theme import (
    card_style, HAIRLINE, FONT_SECTION_HDR, PAGE_PAD_X, PAGE_PAD_Y, PAGE_GAP, CARD_PAD,
)
from ecg.ui.theme import input_style
from ecg.ui.theme import button_style
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_EVENTS

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

_RIBBON_H = 220


class TabEventsMixin:
    """ECGApp Events tab construction and handlers."""

    def _build_tab_arrhythmias(self: ECGApp) -> None:
        """Build the Events tab: bar and readout, ribbon, event cards (left) +
        ECG zoom (right)."""
        page = self.tabs.tab(TAB_EVENTS)
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)
        # The mockup's cards: settings, readout and ribbon in one, then the
        # event list and the event's ECG side by side.
        top = ctk.CTkFrame(page, **card_style())
        top.grid(row=0, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(PAGE_PAD_Y, PAGE_GAP))
        t = ctk.CTkFrame(top, fg_color="transparent")
        t.pack(fill="x", padx=CARD_PAD - SPACE_S, pady=CARD_PAD - SPACE_S)
        t.grid_columnconfigure(0, weight=1)

        # ── Row 0 : Action bar ────────────────────────────────
        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_S, pady=(0, 0))

        self.btn_run_arrhythmia = ctk.CTkButton(
            bar,
            text="⚡  Classify events",
            command=self._run_arrhythmia_analysis,
            **button_style("primary"),
        )
        self.btn_run_arrhythmia.pack(side="left")  # type: ignore[union-attr]

        # ── Baseline window ──
        # The first seconds of the window is the one baseline classify_arrhythmias
        # has; a baseline before an annotation would be a new calculation.
        ctk.CTkLabel(bar, text="Baseline: first", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(SPACE_L, SPACE_XS))
        self.ent_arr_baseline = ctk.CTkEntry(bar, width=48, **input_style())
        self.ent_arr_baseline.insert(0, "30")
        self.ent_arr_baseline.pack(side="left", padx=(0, SPACE_XS))
        _tip = ctk.CTkLabel(bar, text="s of the analysis window", font=FONT_SMALL,
                            text_color=MUTED)
        _tip.pack(side="left", padx=(0, SPACE_M))
        _baseline_tip = (
            "The first N seconds of the analysis window are used as the "
            "reference heart rate that every bradycardia/tachycardia run is "
            "compared against. The window should be at least 2 × N seconds "
            "long, otherwise the baseline is flagged unreliable.")
        self._bind_hover_tip(self.ent_arr_baseline, _baseline_tip)
        self._bind_hover_tip(_tip, _baseline_tip)

        # ── Brady/tachy threshold ──
        ctk.CTkLabel(bar, text="ΔHR threshold (%):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(0, SPACE_XS))
        self.ent_arr_brady_pct = ctk.CTkEntry(bar, width=44, **input_style())
        self.ent_arr_brady_pct.insert(0, "20")
        self.ent_arr_brady_pct.pack(side="left", padx=(0, SPACE_M))

        # ── Minimum beats ──
        ctk.CTkLabel(bar, text="Min run (beats):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(0, SPACE_XS))
        self.ent_arr_min_beats = ctk.CTkEntry(bar, width=44, **input_style())
        self.ent_arr_min_beats.insert(0, "10")
        self.ent_arr_min_beats.pack(side="left", padx=(0, SPACE_M))
        self._bind_hover_tip(
            self.ent_arr_min_beats,
            "Shortest bradycardia/tachycardia (and AV-delay) run counted as "
            "an event — premature beats (RR-based) and irregular runs use "
            "their own fixed thresholds and ignore this.")

        ctk.CTkButton(
            bar,
            text="📋 Copy for Excel",
            command=self._copy_arrhythmia_tsv,
            **button_style("secondary"),
        ).pack(side="right")

        # ── Row 1 : What the events rest on ───────────────────
        info = ctk.CTkFrame(t, fg_color="transparent")
        info.grid(row=1, column=0, sticky="ew", padx=SPACE_S, pady=(SPACE_M, SPACE_S))
        self.lbl_arr_baseline = ctk.CTkLabel(
            info, text=self.analysis_ctrl.events_readout_text(self.analysis.arrhythmia_run),
            font=FONT_LABEL, text_color=TEXT, anchor="w", justify="left",
            wraplength=int(1000 * THEME.font_scale), height=20)
        self.lbl_arr_baseline.pack(side="top", fill="x")
        self.lbl_arrhythmia_status = ctk.CTkLabel(
            info, text="  Click Analyze first", height=18,
            font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_arrhythmia_status.pack(side="top", fill="x")  # type: ignore[union-attr]
        self.lbl_arr_stale = ctk.CTkLabel(
            info, text="", font=FONT_SMALL, text_color=ORANGE, anchor="w", justify="left",
            wraplength=int(1000 * THEME.font_scale), height=18)

        # ── Row 2 : Ribbon + RR series ────────────────────────
        ribbon = tk.Frame(t, bg=PLOT["bg"], bd=0, highlightthickness=1,
                          highlightbackground=HAIRLINE,
                          height=int(_RIBBON_H * THEME.font_scale))
        ribbon.grid(row=2, column=0, sticky="ew", padx=SPACE_S, pady=0)
        ribbon.pack_propagate(False)
        self._slots["arr_ribbon"] = CanvasSlot(ribbon, 14, 2.2, toolbar=False,
                                               placeholder="Click Classify events to display")

        # ── Row 3 : Split body ────────────────────────────────
        body = tk.Frame(page, bg=BG, bd=0, highlightthickness=0)
        body.grid(row=1, column=0, sticky="nsew", padx=PAGE_PAD_X, pady=(0, PAGE_PAD_Y))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0)   # event list — fixed
        body.grid_columnconfigure(1, weight=1)   # ECG viewer — elastic

        # ── Left : scrollable event list ─────────────────────
        list_outer = ctk.CTkFrame(body, width=300, **card_style())
        list_outer.grid(row=0, column=0, sticky="ns", padx=(0, PAGE_GAP))
        list_outer.grid_propagate(False)
        list_outer.grid_rowconfigure(3, weight=1)
        list_outer.grid_columnconfigure(0, weight=1)

        self.lbl_arr_events_head = ctk.CTkLabel(
            list_outer, text="EVENTS",
            font=FONT_SECTION_HDR, text_color=MUTED, anchor="w")
        self.lbl_arr_events_head.grid(row=0, column=0, sticky="ew", padx=CARD_PAD,
                                      pady=(CARD_PAD - 4, SPACE_S))
        ctk.CTkLabel(
            list_outer,
            text=("ALERT = pause/premature run (≥3)  ·  WARNING = brady/tachy/AV-delay "
                  "run, premature couplet  ·  INFO = irregular run, isolated premature beat"),
            font=FONT_HINT, text_color=MUTED, anchor="w", justify="left",
            wraplength=300 - 2 * CARD_PAD,
        ).grid(row=1, column=0, sticky="ew", padx=CARD_PAD, pady=(0, SPACE_XS))
        # Premature-beat counts / rates of the last classification
        # (AnalysisController.refresh_premature_label) -- derived from app
        # state, so a UI rebuild shows them again.
        self.lbl_arr_premature = ctk.CTkLabel(
            list_outer,
            text=self.analysis_ctrl.premature_summary_text(self.analysis.premature_summary),
            font=FONT_HINT, text_color=ORANGE, anchor="w", justify="left",
            wraplength=300 - 2 * CARD_PAD,
        )
        self.lbl_arr_premature.grid(row=2, column=0, sticky="ew", padx=CARD_PAD, pady=(0, SPACE_XS))

        self._arr_event_scroll = ctk.CTkScrollableFrame(
            list_outer, fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER2,
        )
        self._arr_event_scroll.grid(row=3, column=0, sticky="nsew", padx=SPACE_S,
                                    pady=(0, CARD_PAD - SPACE_S))
        self._arr_event_scroll.grid_columnconfigure(0, weight=1)
        self._arr_card_widgets = []

        # ── Right : ECG detail panel ──────────────────────────
        right_card = ctk.CTkFrame(body, **card_style())
        right_card.grid(row=0, column=1, sticky="nsew")
        right = ctk.CTkFrame(right_card, fg_color="transparent")
        right.pack(fill="both", expand=True, padx=CARD_PAD, pady=(CARD_PAD - 4, CARD_PAD))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Title of the event shown
        tbar = ctk.CTkFrame(right, fg_color="transparent", corner_radius=0)
        tbar.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, SPACE_M))
        self.lbl_arr_event_title = ctk.CTkLabel(
            tbar, text="← Click on an event",
            font=FONT_SECTION_HDR, text_color=MUTED, anchor="w")
        self.lbl_arr_event_title.pack(side="left")  # type: ignore[union-attr]

        # ECG canvas
        ecg_frame = tk.Frame(right, bg=PLOT["bg"], bd=0, highlightthickness=1,
                             highlightbackground=HAIRLINE)
        ecg_frame.grid(row=1, column=0, sticky="nsew")
        self._slots["arr_detail"] = CanvasSlot(ecg_frame, 14, 5.0, toolbar=False,
                                               placeholder="Click on an event to zoom on its ECG")

        # Edit / nav controls bar
        ebar = ctk.CTkFrame(right, fg_color="transparent", corner_radius=0)
        ebar.grid(row=2, column=0, sticky="ew", padx=0, pady=(SPACE_M, 0))

        self.btn_arr_prev_event = ctk.CTkButton(
            ebar,
            text="Previous",
            width=76,
            command=lambda: self._arr_step_event(-1),
            **button_style("secondary"),
        )
        self.btn_arr_prev_event.pack(side="left", padx=(SPACE_S, SPACE_XS), pady=SPACE_XS)
        self.btn_arr_next_event = ctk.CTkButton(
            ebar,
            text="Next",
            width=76,
            command=lambda: self._arr_step_event(+1),
            **button_style("secondary"),
        )
        self.btn_arr_next_event.pack(side="left", padx=SPACE_XS)

        self.btn_arr_edit = ctk.CTkButton(
            ebar,
            text="Edit peaks here",
            width=120,
            command=self._toggle_arr_edit_mode,
            **button_style("secondary"),
        )
        self.btn_arr_edit.pack(side="left", padx=SPACE_XS)

        # Undo/Redo removed here -- the toolbar's global pair (top of the
        # window) already covers this tab; it shares the same edit_undo/
        # edit_redo stack as the Detection tab and Ctrl+Z/Ctrl+Y.
        self.lbl_arr_edit_hint = ctk.CTkLabel(
            ebar, text="L-click: exclude/restore   R-click: add",
            font=FONT_HINT, text_color=ORANGE,
        )
        # Pan: ← → buttons + window size
        # Packed from the right: last on screen first.
        self.btn_arr_next = ctk.CTkButton(
            ebar,
            text="▶",
            width=32,
            command=lambda: self._arr_navigate(+1),
            **button_style("secondary"),
        )
        self.btn_arr_next.pack(side="right", padx=SPACE_XS)
        self.btn_arr_prev = ctk.CTkButton(
            ebar,
            text="◀",
            width=32,
            command=lambda: self._arr_navigate(-1),
            **button_style("secondary"),
        )
        self.btn_arr_prev.pack(side="right", padx=SPACE_XS)
        self._bind_hover_tip(self.btn_arr_prev, "Move the ECG window back in time")
        self._bind_hover_tip(self.btn_arr_next, "Move the ECG window forward in time")
        self.ent_arr_win = ctk.CTkEntry(ebar, width=48, **input_style())
        self.ent_arr_win.insert(0, "3")
        self.ent_arr_win.pack(side="right", padx=(0, SPACE_S))
        ctk.CTkLabel(ebar, text="Window (s):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="right", padx=(0, SPACE_XS))

        # Wire up click handlers on the canvases
        def _wire_arr_click(slot=self._slots["arr_detail"], ribbon_slot=self._slots["arr_ribbon"]):
            slot.canvas.mpl_connect("button_press_event", self._on_arr_detail_click)
            slot.canvas.mpl_connect("scroll_event",       self._on_arr_scroll)
            ribbon_slot.canvas.mpl_connect("button_press_event", self.analysis_ctrl.on_arr_ribbon_click)
        self.after(200, _wire_arr_click)

    def _refresh_events_readout(self: ECGApp) -> None:
        """Say so when the events shown no longer match the peaks: on the shell's
        tick, like the tab's own state."""
        lbl = self.lbl_arr_stale
        if lbl is None:
            return
        text = stale_banner_text("Events", self.freshness.status("events"), None,
                                 self.ui.peaks_edited_at)
        if lbl.cget("text") != text:
            lbl.configure(text=text)
            if text:
                lbl.pack(side="top", fill="x")
            else:
                lbl.pack_forget()

    def _run_arrhythmia_analysis(self: ECGApp) -> None:
        self.analysis_ctrl.run_arrhythmia_analysis()

    # ── Event selection & ECG viewer ─────────────────────────

    def _select_arrhythmia_event(self: ECGApp, idx: int) -> None:
        """Highlight selected card and load the ECG window for this event."""
        self.nav_ctrl.select_arrhythmia_event(idx)

    def _arr_step_event(self: ECGApp, direction: int) -> None:
        """Previous / Next event."""
        self.nav_ctrl.arr_step_event(direction)

    def _draw_arr_detail(self: ECGApp) -> None:
        """Draw ECG strip for the selected arrhythmia event, with editable R peaks."""
        self.plot_ctrl.draw_arr_detail()
    # ── Edit mode toggle for the Events tab ──────────────────

    def _toggle_arr_edit_mode(self: ECGApp) -> None:
        self.analysis_ctrl.toggle_arr_edit_mode()

    def _sync_arr_edit_btn(self: ECGApp) -> None:
        self.analysis_ctrl.sync_arr_edit_btn()

    # ── Click handler (mirrors _on_detail_click) ─────────────

    def _on_arr_detail_click(self: ECGApp, event) -> None:
        """Left-click: toggle exclusion.  Right-click: add/remove peak."""
        self.analysis_ctrl.on_arr_detail_click(event)

    # ── Scroll zoom on arrhythmia ECG ────────────────────────

    def _on_arr_scroll(self: ECGApp, event) -> None:
        self.analysis_ctrl.on_arr_scroll(event)

    # ── ◀ ▶ navigation ───────────────────────────────────────

    def _arr_navigate(self: ECGApp, direction: int) -> None:
        self.nav_ctrl.arr_navigate(direction)

    # ── sync undo/redo buttons in arrhythmia tab ─────────────

    def _update_undo_btns(self: ECGApp) -> None:
        """Update all undo/redo button instances (Detection + Events tabs)."""
        self.detection_ctrl.update_undo_btns()

    def _copy_arrhythmia_tsv(self: ECGApp) -> None:
        self.analysis_ctrl.copy_arrhythmia_tsv()
