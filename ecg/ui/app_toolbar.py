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
import tkinter.font as tkfont
from ecg.ui.theme import (
    THEME, NK_AVAILABLE, BG, PANEL, BORDER, BORDER2, TEXT, MUTED, LIGHT, BLUE,
    GREEN, ORANGE, BLUE_HOVER, PURPLE, PURPLE_DARK, TEAL, TEAL_DARK, GREEN_DARK,
    COLOR_PRIMARY, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_SMALL, FONT_LABEL,
    FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT, FONT_TITLE, FONT_MONO,
    FONT_CARD_TITLE,
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

        row = ctk.CTkFrame(bar, fg_color="transparent", height=44)
        row.pack(fill="x", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))
        row.pack_propagate(False)

        # Right cluster packed first: when the window is narrow, pack takes
        # the room from the file chip (packed last), never from a command.
        right = ctk.CTkFrame(row, fg_color="transparent")
        right.pack(side="right")

        # ── File ops chip: Open ▾ / Save ──────────────────────────────────
        file_chip = self._toolbar_chip(right)
        file_chip.pack(side="left", padx=(0, SPACE_M))
        # Open is a one-time setup step, not part of the Detect/Analyze
        # pipeline -- styled neutral rather than a third saturated-blue
        # button competing with Detect Peaks for attention.
        ctk.CTkButton(
            file_chip, text="📂  Open", width=80, height=34,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, corner_radius=8,
            command=self._open_file,
        ).pack(side="left", padx=(SPACE_S, 0))
        self.btn_recent = ctk.CTkButton(
            file_chip, text="▾", width=30, height=34,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, corner_radius=8,
            command=self._toggle_recent_menu,
        )
        self.btn_recent.pack(side="left", padx=(SPACE_XS, SPACE_S))
        self._bind_hover_tip(self.btn_recent, "Recent files")

        # Starts visibly disabled (muted, matching Undo/Redo's idle look)
        # rather than a full-saturation green that reads as clickable
        # before any session exists to save -- _set_save_session_enabled()
        # restores the green "ready" look the moment it actually is.
        self.btn_save_session = ctk.CTkButton(
            file_chip, text="💾  Save Session", command=self._save_session,
            fg_color=BORDER, hover_color=BORDER, text_color=MUTED,
            font=FONT_BTN_SEC, height=34, width=120, corner_radius=8,
            state="disabled")
        self.btn_save_session.pack(side="left", padx=(0, SPACE_S))

        # ── Analysis actions chip: Detect Peaks / Analyze / Export ────────
        analysis_chip = self._toolbar_chip(right)
        analysis_chip.pack(side="left", padx=(0, SPACE_M))

        self.btn_preview = ctk.CTkButton(
            analysis_chip, text="Detect Peaks", width=116,
            command=self._preview,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_PRIMARY, height=34, corner_radius=8)
        self.btn_preview.pack(side="left", padx=(SPACE_S, SPACE_XS))

        run_color = GREEN if NK_AVAILABLE else BORDER2
        self.btn_run = ctk.CTkButton(
            analysis_chip, text="Analyze all", width=108,
            command=self._run_analysis,
            fg_color=run_color, hover_color=GREEN_DARK if NK_AVAILABLE else BORDER2,
            text_color="white", font=FONT_BTN_PRIMARY, height=34,
            corner_radius=8, state="normal" if NK_AVAILABLE else "disabled")
        self.btn_run.pack(side="left", padx=(0, SPACE_S))
        if not NK_AVAILABLE:
            self._bind_hover_tip(self.btn_run, "pip install neurokit2 to enable", ORANGE)
        else:
            self._bind_hover_tip(
                self.btn_run,
                "Core analysis, then Frequency and Non-linear HRV, then epochs when their "
                "auto-run is on. Intervals and Events have their own buttons.")

        self.btn_toolbar_export = ctk.CTkButton(
            analysis_chip, text="Export…", width=76, height=34, font=FONT_BTN_SEC,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT, corner_radius=8,
            command=self._show_export)
        self.btn_toolbar_export.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(self.btn_toolbar_export,
                             "Report tab: what goes in the exports, and the formats")

        # ── Edit chip: Undo / Redo (Ctrl+Z / Ctrl+Y) ──────────────────────
        edit_chip = self._toolbar_chip(right)
        edit_chip.pack(side="left", padx=(0, SPACE_M))
        self.btn_toolbar_undo = ctk.CTkButton(
            edit_chip, text="↩ Undo", width=72, height=34, font=FONT_BTN_SEC,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            corner_radius=8, state="disabled", command=self._undo_edit)
        self.btn_toolbar_undo.pack(side="left", padx=(SPACE_S, SPACE_XS))
        self.btn_toolbar_redo = ctk.CTkButton(
            edit_chip, text="↪ Redo", width=72, height=34, font=FONT_BTN_SEC,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            corner_radius=8, state="disabled", command=self._redo_edit)
        self.btn_toolbar_redo.pack(side="left", padx=(0, SPACE_S))

        # ── ⋯: Parameters, Theme, light / dark, shortcuts, reset, cache ───
        self.btn_more_menu = ctk.CTkButton(
            right, text="⋯", width=40, height=34, font=FONT_CARD_TITLE,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=self._toggle_more_menu)
        self.btn_more_menu.pack(side="left")
        self._bind_hover_tip(
            self.btn_more_menu,
            "Parameters, theme, keyboard shortcuts, reset to mouse defaults, clear session cache")

        # ── Identity: app name, project, file chip ─────────────────────────
        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True, padx=(0, SPACE_M))
        ctk.CTkLabel(left, text="ECG Analysis", font=FONT_TITLE, text_color=TEXT,
                     anchor="w").pack(side="left", padx=(SPACE_XS, SPACE_M))
        self.lbl_topbar_project = ctk.CTkLabel(
            left, text="", width=0, font=FONT_SIDEBAR_HDR, text_color=TEXT, anchor="w")
        self.lbl_topbar_project.pack(side="left", padx=(0, SPACE_M))
        self._bind_hover_tip(self.lbl_topbar_project, "Project name (Recording sheet)")

        # The file chip: file · channel · rate, opening the Recording sheet.
        # A frame of labels rather than a CTkButton, so the text set by
        # _update_topbar_filename() is the file identity alone (no "▾").
        # Sized to its text; packed last, it is the one cut short when the
        # window is narrow.
        self.file_chip = ctk.CTkFrame(left, fg_color=BG, corner_radius=8, border_width=1,
                                      border_color=BORDER2)
        self.file_chip.pack(side="left")
        chevron = ctk.CTkLabel(self.file_chip, text="▾", font=FONT_LABEL, text_color=MUTED,
                               width=16, height=28)
        chevron.pack(side="right", padx=(SPACE_XS, SPACE_M))
        self.lbl_topbar_file = ctk.CTkLabel(
            self.file_chip, text="No file loaded", font=FONT_MONO, text_color=TEXT,
            anchor="w", height=28)
        self.lbl_topbar_file.pack(side="left", fill="x", expand=True, padx=(SPACE_M, 0))
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

    def _set_btn_enabled(self: ECGApp, btn, enabled: bool, active_fg: str, active_hover: str,
                         active_text: str = "white", muted_fg: str = BORDER,
                         muted_hover: str = BORDER2, muted_text: str = MUTED) -> None:
        """Enable/disable *btn* and swap its fill/text color to match.

        Generalised from the Save-Session-only helper this used to be:
        CTkButton's default disabled treatment only dims the text, not the
        background, so a disabled button with a saturated fg_color still
        reads as clickable. Call this instead of a bare
        `.configure(state=...)` at every site that flips a button's
        availability.
        """
        if btn is None:
            return
        if enabled:
            btn.configure(state="normal", fg_color=active_fg,
                          hover_color=active_hover, text_color=active_text)
        else:
            btn.configure(state="disabled", fg_color=muted_fg,
                          hover_color=muted_hover, text_color=muted_text)

    def _set_save_session_enabled(self: ECGApp, enabled: bool) -> None:
        """Enable/disable Save and swap its look to match -- see _set_btn_enabled."""
        self._set_btn_enabled(self.btn_save_session, enabled, GREEN, GREEN_DARK)

    def _set_check_agreement_enabled(self: ECGApp, enabled: bool) -> None:
        """Enable/disable Check Agreement and swap its look to match.

        Active fill is TEAL; dark-mode TEAL is lightened for legibility,
        which then needs BG (not white) text to hold WCAG contrast.
        """
        self._set_btn_enabled(
            self.btn_check_agreement, enabled, TEAL, TEAL_DARK,
            active_text=(BG if THEME.is_dark else "white"))

    def _set_review_artifacts_enabled(self: ECGApp, enabled: bool) -> None:
        """Enable/disable Review Artifacts and swap its look to match.

        Active fill is BLUE, not ORANGE -- ORANGE is this app's warning/
        Edit-Mode-ON colour elsewhere; opening the review dialog isn't
        itself a warning state.
        """
        self._set_btn_enabled(self.btn_review_art, enabled, BLUE, BLUE_HOVER)

    # Active (fg, hover) colors for the on-demand per-tab compute buttons
    # plus Save Session, keyed by attribute name -- shared by every
    # "unlock after Analyze" / "reset for new file" call site so they all
    # get the muted-vs-active treatment above instead of a bare
    # `.configure(state=...)` loop that leaves a disabled button just as
    # saturated as an enabled one.
    _RESULT_BTN_COLORS: "dict[str, tuple[str, str]]" = {
        "btn_run_freq":       (BLUE, BLUE_HOVER),
        "btn_run_nonlin":     (PURPLE, PURPLE_DARK),
        "btn_run_ivl":        (BLUE, BLUE_HOVER),
        "btn_run_arrhythmia": (BLUE, BLUE_HOVER),
        "btn_save_session":   (GREEN, GREEN_DARK),
    }

    def _set_result_btns_enabled(self: ECGApp, enabled: bool, attrs: "tuple[str, ...]") -> None:
        """Apply _set_btn_enabled() to each button name in *attrs*."""
        for attr in attrs:
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            active_fg, active_hover = self._RESULT_BTN_COLORS.get(attr, (BORDER, BORDER2))
            self._set_btn_enabled(btn, enabled, active_fg, active_hover)
