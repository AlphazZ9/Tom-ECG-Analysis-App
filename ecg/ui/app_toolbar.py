# -*- coding: utf-8 -*-
"""
ecg.ui.app_toolbar
------------------
ECGApp toolbar construction and button-state helpers.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
import tkinter.font as tkfont
from ecg.ui.theme import (
    THEME, NK_AVAILABLE, BG, PANEL, BORDER, BORDER2, TEXT, MUTED, LIGHT, BLUE,
    GREEN, ORANGE, BLUE_HOVER, PURPLE, PURPLE_DARK, TEAL, TEAL_DARK, GREEN_DARK,
    COLOR_PRIMARY, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_SMALL, FONT_LABEL,
    FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT, FONT_TITLE, FONT_MONO,
    FONT_CARD_TITLE, button_style, disabled_style, BTN_H, BTN_H_LG, RADIUS, CARD,
)

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class ToolbarMixin:
    """ECGApp toolbar construction and button-state helpers."""

    # ─── KPI bar ──────────────────────────────────────────────

    def _build_toolbar(self: ECGApp) -> None:
        """Header spanning the full window width: identity and file chip on
        the left, the commands on the right.

        There is no sidebar any more: the file chip opens the Recording sheet
        (channel, rate, subject, crop...), the ▾ beside Open the recent files,
        and ⋯ the rarely used commands (Parameters, Theme, shortcuts, reset,
        clear session cache) -- see app_header_menus.py.

        btn_preview/btn_run/btn_save_session keep their attribute names:
        session_controller.py/signal_controller.py/analysis_controller.py all
        reference them by name for busy-state text swaps ("Loading…",
        "Analysing…", "Saving…", "Restoring…").
        """
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        bar.pack(side="top", fill="x")
        ctk.CTkFrame(bar, height=1, fg_color=BORDER).pack(side="bottom", fill="x")

        # Progress row -- child of `bar` itself (not the inline row below) so
        # that _start_async's existing self._prog_row.pack(side="bottom",
        # fill="x", ...) call spans the full toolbar width, unchanged. Not
        # packed here -- _start_async/_stop_progress show/hide it dynamically.
        self._prog_row = ctk.CTkFrame(bar, fg_color="transparent", height=20)
        self.progress = ctk.CTkProgressBar(
            self._prog_row, height=4, mode="determinate",
            progress_color=COLOR_PRIMARY, fg_color=BORDER, corner_radius=2,
        )
        self.progress.set(0)
        self.progress.pack(side="left", fill="x", expand=True, padx=(SPACE_M, SPACE_S))
        self.lbl_progress = ctk.CTkLabel(
            self._prog_row, text="", font=FONT_HINT,
            text_color=MUTED, width=260, anchor="w",
        )
        self.lbl_progress.pack(side="left", padx=(0, SPACE_M))

        # The mockup's header: 10 px above and below 40 px commands, 24 px
        # side margins, controls 8 px apart, one blue primary (Analyze all).
        h_lg = int(round(BTN_H_LG * THEME.font_scale))
        row = ctk.CTkFrame(bar, fg_color="transparent", height=h_lg + 2 * 10)
        row.pack(fill="x", padx=24 - SPACE_S)
        row.pack_propagate(False)
        gap = (0, SPACE_M)

        # Right cluster packed first: when the window is narrow, pack takes
        # the room from the file chip (packed last), never from a command.
        right = ctk.CTkFrame(row, fg_color="transparent")
        right.pack(side="right")

        def cmd(parent, text, command, kind="secondary", width=0, **kw):
            b = ctk.CTkButton(parent, text=text, command=command, width=width,
                              **{**button_style(kind, large=True), **kw})
            b.pack(side="left", padx=gap)
            return b

        # ── Open ▾ (one control: no gap between the two halves) ──────────
        open_pair = self._toolbar_chip(right)
        open_pair.pack(side="left", padx=gap)
        ctk.CTkButton(open_pair, text="Open", width=64, command=self._open_file,
                      **button_style("secondary", large=True)).pack(side="left")
        self.btn_recent = ctk.CTkButton(
            open_pair, text="▾", width=h_lg, command=self._toggle_recent_menu,
            **{**button_style("secondary", large=True), "font": FONT_LABEL})
        self.btn_recent.pack(side="left", padx=(1, 0))
        self._bind_hover_tip(self.btn_recent, "Recent files")

        # Open is a one-time setup step and Detect Peaks a step towards the
        # analysis: both white; Analyze all is the one blue command.
        self.btn_preview = cmd(right, "Detect Peaks", self._preview)
        self.btn_run = cmd(right, "Analyze all", self._run_analysis,
                           "primary" if NK_AVAILABLE else "secondary",
                           state="normal" if NK_AVAILABLE else "disabled")
        if not NK_AVAILABLE:
            self.btn_run.configure(**disabled_style())
            self._bind_hover_tip(self.btn_run, "pip install neurokit2 to enable", ORANGE)
        else:
            self._bind_hover_tip(
                self.btn_run,
                "Core analysis, then Frequency and Non-linear HRV, then epochs when their "
                "auto-run is on. Intervals and Events have their own buttons.")

        # Starts disabled, until a session exists to save --
        # _set_save_session_enabled() restores the enabled look.
        self.btn_save_session = cmd(right, "Save Session", self._save_session,
                                    state="disabled")
        self.btn_save_session.configure(**disabled_style())

        self.btn_toolbar_export = cmd(right, "Export…", self._show_export)
        self._bind_hover_tip(self.btn_toolbar_export,
                             "Report tab: what goes in the exports, and the formats")

        # ── Undo / Redo (Ctrl+Z / Ctrl+Y) ─────────────────────────────────
        self.btn_toolbar_undo = cmd(right, "↩ Undo", self._undo_edit, state="disabled")
        self.btn_toolbar_redo = cmd(right, "↪ Redo", self._redo_edit, state="disabled")
        for b in (self.btn_toolbar_undo, self.btn_toolbar_redo):
            b.configure(**disabled_style())

        # ── ⋯: Parameters, Theme, light / dark, shortcuts, reset, cache ───
        self.btn_more_menu = cmd(right, "⋯", self._toggle_more_menu, width=h_lg,
                                 font=FONT_CARD_TITLE)
        self.btn_more_menu.pack_configure(padx=0)
        self._bind_hover_tip(
            self.btn_more_menu,
            "Parameters, theme, keyboard shortcuts, reset to mouse defaults, clear session cache")

        # ── Identity: ECG mark, app name, project, file chip ──────────────
        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True, padx=(0, SPACE_L))
        mark = tk.Canvas(left, width=22, height=22, bg=PANEL, highlightthickness=0, bd=0)
        # The mockup's mark: the path M2 12h4l2-6 4 12 3-9 2 3h5 on a 24 grid.
        pts = [(2, 12), (6, 12), (8, 6), (12, 18), (15, 9), (17, 12), (22, 12)]
        mark.create_line(*[c * 22 / 24 for p in pts for c in p], fill=BLUE, width=2,
                         capstyle="round", joinstyle="round")
        mark.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(left, text="ECG Analysis", font=FONT_TITLE, text_color=TEXT,
                     anchor="w").pack(side="left", padx=(0, 10))
        self.lbl_topbar_project = ctk.CTkLabel(
            left, text="", width=0, font=FONT_SIDEBAR_HDR, text_color=MUTED, anchor="w")
        self.lbl_topbar_project.pack(side="left", padx=(0, SPACE_M))
        self._bind_hover_tip(self.lbl_topbar_project, "Project name (Recording sheet)")

        # The file chip: file · channel · rate, opening the Recording sheet.
        # A frame of labels rather than a CTkButton, so the text set by
        # _update_topbar_filename() is the file identity alone (no "▾").
        # Sized to its text; packed last, it is the one cut short when the
        # window is narrow.
        h = int(round(BTN_H * THEME.font_scale))
        self.file_chip = ctk.CTkFrame(left, fg_color=BG, corner_radius=RADIUS, border_width=1,
                                      border_color=BORDER, height=h)
        self.file_chip.pack(side="left")
        chevron = ctk.CTkLabel(self.file_chip, text="▾", font=FONT_LABEL, text_color=MUTED,
                               width=16, height=h - 4)
        chevron.pack(side="right", padx=(SPACE_XS, SPACE_M), pady=2)
        self.lbl_topbar_file = ctk.CTkLabel(
            self.file_chip, text="No file loaded", font=FONT_MONO, text_color=TEXT,
            anchor="w", height=h - 4)
        self.lbl_topbar_file.pack(side="left", fill="x", expand=True, padx=(10, 0), pady=2)
        for w in (self.file_chip, self.lbl_topbar_file, chevron):
            w.bind("<Button-1>", lambda _e: self._toggle_recording_sheet())
            w.configure(cursor="hand2")
            self._bind_hover_tip(
                w, "Recording: channel, sampling rate, subject, project, crop, notes, session")

        # ── Status readout -- the app's single non-modal feedback channel.
        # Built by _build_status_label in the slack of the tab row: kept
        # single-line and truncated to fit (see _fit_status_text()).
        self._status_font = tkfont.Font(font=FONT_SMALL)
        self.lbl_status = None
        self._status_text, self._status_color = "Ready", MUTED
        self._status_display_raw, self._status_display_color = "Ready", MUTED

    def _build_status_label(self: ECGApp, bar) -> None:
        """The status readout, in the slack of the tab row *bar*."""
        self.lbl_status = ctk.CTkLabel(
            bar, text="Ready", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left")
        self.lbl_status.pack(side="left", fill="x", expand=True, padx=(SPACE_L, SPACE_S))
        # Bound to *bar*, not lbl_status itself -- its size only changes from an
        # actual window resize, whereas lbl_status's own size could in principle
        # shift in response to its own .configure(text=...) call, which would
        # re-trigger this handler and risk a layout feedback loop.
        bar.bind("<Configure>", lambda _e: self._refresh_status_text(), add="+")

    def _set_btn_enabled(self: ECGApp, btn, enabled: bool, kind: str = "primary") -> None:
        """Enable/disable *btn* and swap its look to match: its *kind* of
        button_style() when enabled, disabled_style() when not.

        CTkButton's default disabled treatment only dims the text, not the
        background, so a disabled button with a saturated fg_color still
        reads as clickable. Call this instead of a bare
        `.configure(state=...)` at every site that flips a button's
        availability.
        """
        if btn is None:
            return
        if enabled:
            look = button_style(kind)
            btn.configure(state="normal", fg_color=look["fg_color"],
                          hover_color=look["hover_color"], text_color=look["text_color"],
                          border_width=look["border_width"],
                          border_color=look.get("border_color", look["fg_color"]))
        else:
            btn.configure(state="disabled", **disabled_style())

    def _set_save_session_enabled(self: ECGApp, enabled: bool) -> None:
        """Enable/disable Save and swap its look to match -- see _set_btn_enabled."""
        self._set_btn_enabled(self.btn_save_session, enabled, "secondary")

    def _set_check_agreement_enabled(self: ECGApp, enabled: bool) -> None:
        """Enable/disable Check Agreement and swap its look to match."""
        self._set_btn_enabled(self.btn_check_agreement, enabled, "secondary")

    def _set_review_artifacts_enabled(self: ECGApp, enabled: bool) -> None:
        """Enable/disable Review Artifacts and swap its look to match (the
        mockup's white bordered button: opening the review is not the
        panel's main action)."""
        self._set_btn_enabled(self.btn_review_art, enabled, "secondary")

    # Button kind of the on-demand per-tab compute buttons plus Save
    # Session, keyed by attribute name -- shared by every "unlock after
    # Analyze" / "reset for new file" call site so they all get the
    # muted-vs-active treatment above instead of a bare
    # `.configure(state=...)` loop that leaves a disabled button just as
    # saturated as an enabled one.
    _RESULT_BTN_KINDS: "dict[str, str]" = {
        "btn_run_freq":       "primary",
        "btn_run_nonlin":     "primary",
        "btn_run_ivl":        "primary",
        "btn_run_arrhythmia": "primary",
        "btn_save_session":   "secondary",
    }

    def _set_result_btns_enabled(self: ECGApp, enabled: bool, attrs: "tuple[str, ...]") -> None:
        """Apply _set_btn_enabled() to each button name in *attrs*."""
        for attr in attrs:
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            self._set_btn_enabled(btn, enabled, self._RESULT_BTN_KINDS.get(attr, "secondary"))
