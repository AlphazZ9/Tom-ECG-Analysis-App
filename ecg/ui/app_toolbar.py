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
    COLOR_PRIMARY, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_SMALL,
    FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT, FONT_BADGE,
)
from ecg.ui.widgets import make_quality_gauge

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
        """Primary instrument-style toolbar spanning the FULL window width.

        Packed on self (not inside main -- see _build()), before the
        sidebar/main split, so it isn't confined to main's width. Consolidates
        what used to be split across the old KPI/top bar AND the sidebar's
        numbered workflow buttons (Preview/Run) into one real toolbar, per
        the Adaptive Workbench brief: Open/Save/Export/Detect Peaks/Analyze/
        Settings, Undo/Redo, project name, progress indicator.

        btn_preview/btn_run/btn_save_session are RELOCATED here (same
        attribute names, same CTkButton objects) rather than recreated --
        session_controller.py/signal_controller.py/analysis_controller.py all
        reference them by name for busy-state text swaps ("Loading…",
        "Analysing…", "Saving…", "Restoring…") and must keep working
        unchanged.
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

        # ── Left panel toggle ──────────────────────────────────────────────
        self.btn_toggle_left_panel = ctk.CTkButton(
            row, text="⟨ Sidebar", width=88, height=34,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, corner_radius=8,
            command=self._toggle_left_panel,
        )
        self.btn_toggle_left_panel.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(self.btn_toggle_left_panel, "Hide/show the left panel")

        # ── File ops chip: Open / Save ────────────────────────────────────
        # Export is reached solely from the right panel's EXPORTS section now
        # (was previously duplicated here too via a toolbar dropdown).
        file_chip = self._toolbar_chip(row)
        file_chip.pack(side="left", padx=(0, SPACE_M))

        # Open is a one-time setup step, not part of the Detect/Analyze
        # pipeline -- styled neutral (matching Theme/panel-toggle) rather
        # than a third saturated-blue button competing with Detect Peaks
        # for attention.
        ctk.CTkButton(
            file_chip, text="📂  Open", width=90, height=34,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, corner_radius=8,
            command=self._open_file,
        ).pack(side="left", padx=(SPACE_S, SPACE_XS))

        # Starts visibly disabled (muted, matching Undo/Redo's idle look)
        # rather than a full-saturation green that reads as clickable
        # before any session exists to save -- _set_save_session_enabled()
        # restores the green "ready" look the moment it actually is.
        # font=FONT_BTN_SEC to match Open in this same chip -- it used to be
        # FONT_CARD_TITLE (bold, 1pt larger), which with Open's FONT_BTN_SEC
        # made this one 34px-tall row mix three different button fonts.
        self.btn_save_session = ctk.CTkButton(
            file_chip, text="💾  Save Session", command=self._save_session,
            fg_color=BORDER, hover_color=BORDER, text_color=MUTED,
            font=FONT_BTN_SEC, height=34, corner_radius=8,
            state="disabled")
        self.btn_save_session.pack(side="left", padx=(0, SPACE_S))

        # ── Analysis actions chip: Detect Peaks / Analyze ─────────────────
        analysis_chip = self._toolbar_chip(row)
        analysis_chip.pack(side="left", padx=(0, SPACE_M))

        self.btn_preview = ctk.CTkButton(
            analysis_chip, text="Detect Peaks",
            command=self._preview,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_PRIMARY, height=34, corner_radius=8)
        self.btn_preview.pack(side="left", padx=(SPACE_S, SPACE_XS))

        run_color = GREEN if NK_AVAILABLE else BORDER2
        self.btn_run = ctk.CTkButton(
            analysis_chip, text="Analyze",
            command=self._run_analysis,
            fg_color=run_color, hover_color=GREEN_DARK if NK_AVAILABLE else BORDER2,
            text_color="white", font=FONT_BTN_PRIMARY, height=34,
            corner_radius=8, state="normal" if NK_AVAILABLE else "disabled")
        self.btn_run.pack(side="left", padx=(0, SPACE_S))
        if not NK_AVAILABLE:
            self._bind_hover_tip(self.btn_run, "pip install neurokit2 to enable", ORANGE)

        # ── Edit chip: Undo / Redo toolbar mirrors ────────────────────────
        edit_chip = self._toolbar_chip(row)
        edit_chip.pack(side="left", padx=(0, SPACE_M))
        self.btn_toolbar_undo = ctk.CTkButton(
            edit_chip, text="↩ Undo", width=88, height=34, font=FONT_BTN_SEC,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            corner_radius=8, state="disabled", command=self._undo_edit)
        self.btn_toolbar_undo.pack(side="left", padx=(SPACE_S, SPACE_XS))
        self.btn_toolbar_redo = ctk.CTkButton(
            edit_chip, text="↪ Redo", width=88, height=34, font=FONT_BTN_SEC,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            corner_radius=8, state="disabled", command=self._redo_edit)
        self.btn_toolbar_redo.pack(side="left", padx=(0, SPACE_S))

        # Thin vertical rule -- the button chips end here and the rest of
        # the bar is mostly empty space at idle (no file loaded yet), so a
        # visible seam anchors "controls" vs. "status" instead of the
        # identity label appearing to float in a blank stretch.
        ctk.CTkFrame(row, width=1, fg_color=BORDER).pack(
            side="left", fill="y", pady=SPACE_S, padx=(0, SPACE_M))

        # ── Identity: project name + file name ────────────────────────────
        left = ctk.CTkFrame(row, fg_color="transparent")
        left.pack(side="left")
        self.lbl_topbar_project = ctk.CTkLabel(
            left, text="—", font=FONT_SIDEBAR_HDR, text_color=TEXT, anchor="w")
        self.lbl_topbar_project.pack(side="left")
        ctk.CTkLabel(left, text="   ", font=FONT_HINT, text_color=LIGHT).pack(side="left")
        self.lbl_topbar_file = ctk.CTkLabel(
            left, text="No file loaded", font=FONT_HINT, text_color=MUTED, anchor="w")
        self.lbl_topbar_file.pack(side="left")

        # ── Status readout — the app's single non-modal feedback channel ──
        # Was a child of the left sidebar's DETECTION accordion, so it went
        # dark whenever that section was collapsed or the panel hidden, and
        # every hover tip (routed through the same label -- see
        # _bind_hover_tip) read as a detection setting. Living in the
        # always-visible toolbar, it survives both. Fills the slack between
        # the file identity and the quality gauge/Settings cluster.
        # No wraplength: a long status message or hover tip used to wrap to
        # 2-3 lines and, since this row has a fixed height (pack_propagate
        # (False) above), spill downward over the quality gauge/badge and
        # the Parameters/Theme/"Stats ⟩" buttons to its right, hiding them
        # entirely. Kept single-line and truncated to fit instead -- see
        # _fit_status_text()/_refresh_status_text().
        self._status_font = tkfont.Font(font=FONT_SMALL)
        self.lbl_status = ctk.CTkLabel(
            row, text="Ready", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left")
        self.lbl_status.pack(side="left", fill="x", expand=True, padx=(SPACE_L, SPACE_M))
        self._status_text, self._status_color = "Ready", MUTED
        self._status_display_raw, self._status_display_color = "Ready", MUTED
        # Bound to `row` (the toolbar row), not lbl_status itself -- row's
        # size only changes from an actual window resize, whereas
        # lbl_status's own size could in principle shift in response to
        # its own .configure(text=...) call, which would re-trigger this
        # handler and risk a layout feedback loop.
        row.bind("<Configure>", lambda _e: self._refresh_status_text())

        # ── Right: quality gauge + badge + Settings/Theme ─────────────────
        right = ctk.CTkFrame(row, fg_color="transparent")
        right.pack(side="right")

        self.quality_gauge = make_quality_gauge(right, score=None, compact=True)
        self.quality_gauge.pack(side="left", padx=(0, SPACE_M))
        # lbl_quality now IS the gauge's own caption label -- update_signal_
        # quality()'s existing lbl_quality.configure(...) call needs no
        # changes; its output is immediately superseded by the more
        # descriptive Excellent/Good/Medium/Poor caption update_quality_
        # gauge() sets right after it (see detection_controller.py).
        self.lbl_quality = self.quality_gauge.score_label

        # Quality badge — updated by _update_quality_badge() after analysis
        self._lbl_quality_badge = ctk.CTkLabel(
            right, text="", font=FONT_BADGE,
            text_color="white", fg_color="transparent",
            corner_radius=6, width=120, height=22, anchor="center")
        self._lbl_quality_badge.pack(side="left", padx=(0, SPACE_M))

        # Parameters moved back into the always-visible toolbar -- it's the
        # app's only settings dialog and the sidebar's bottom-of-scroll
        # button was invisible without deliberate scrolling on any normal
        # window height.
        ctk.CTkButton(right, text="⚙  Parameters", width=110, height=34,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      font=FONT_BTN_SEC, command=self._open_params_dialog,
                      corner_radius=8).pack(side="left", padx=(0, SPACE_S))
        ctk.CTkButton(right, text="Theme", width=76, height=34,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      font=FONT_BTN_SEC, command=self._open_theme_dialog,
                      corner_radius=8).pack(side="left", padx=(0, SPACE_S))
        ctk.CTkButton(right, text="☀/☾", width=52, height=34,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      font=FONT_BTN_SEC, corner_radius=8,
                      command=self._toggle_dark_live,
                      ).pack(side="left", padx=(0, SPACE_S))

        # ── Right panel toggle ─────────────────────────────────────────────
        self.btn_toggle_right_panel = ctk.CTkButton(
            right, text="Stats ⟩", width=88, height=34,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, corner_radius=8,
            command=self._toggle_right_panel,
        )
        self.btn_toggle_right_panel.pack(side="left")
        self._bind_hover_tip(self.btn_toggle_right_panel, "Hide/show the right panel")

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
