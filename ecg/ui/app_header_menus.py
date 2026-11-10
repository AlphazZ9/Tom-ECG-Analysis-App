# -*- coding: utf-8 -*-
"""
ecg.ui.app_header_menus
-----------------------
What the header opens: the Recording sheet (file chip), the recent files
(▾ beside Open) and the ⋯ menu (Parameters, Theme, light / dark, shortcuts,
reset to mouse defaults, clear session cache).

The sheet's entries are the ones the left sidebar used to hold, under the same
attribute names (ent_channel, ent_fs, ent_subject, ent_project_name,
ent_t_start, ent_t_end, lbl_fs_source, lbl_file, lbl_npeaks,
lbl_session_info): they exist from startup, so sessions, FilterParams and the
controllers read and write them whether the sheet is open or not.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core.models import MouseECG
from ecg.ui.popover import Popover, menu_button
from ecg.ui.theme import (
    BG, BORDER, BORDER2, TEXT, MUTED, LIGHT, BLUE, SPACE_XS, SPACE_S, SPACE_M,
    FONT_LABEL, FONT_SMALL, FONT_KPI_LABEL, FONT_HINT, FONT_BTN_SEC, FONT_CARD_TITLE,
)
from ecg.ui.theme import input_style
from ecg.ui.theme import button_style

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

_SHEET_ENTRY_W = 200
_RECENT_SHOWN = 8


class HeaderMenusMixin:
    """ECGApp Recording sheet, recent files and ⋯ menu."""

    def _build_header_menus(self: ECGApp) -> None:
        self._build_recording_sheet()
        self.more_menu = Popover(self)
        b = self.more_menu.body
        for text, command in (
                ("⚙  Parameters…", self._open_params_dialog),
                ("🎨  Theme…", self._open_theme_dialog),
                ("☀/☾  Light / dark mode", self._toggle_dark_live)):
            menu_button(b, text, self._menu_command(self.more_menu, command))
        ctk.CTkFrame(b, height=1, fg_color=BORDER).pack(fill="x", pady=SPACE_XS)
        for text, command in (
                ("⌨  Keyboard shortcuts  (F1)", self._show_shortcuts_help),
                ("↺  Reset to mouse ECG defaults", self._reset_params),
                ("🗑  Clear session cache", self._delete_session)):
            menu_button(b, text, self._menu_command(self.more_menu, command))

        self.recent_menu = Popover(self, on_open=self._fill_recent_menu)

    @staticmethod
    def _menu_command(popover: Popover, command):
        """Close the menu first: the command may open a modal dialog."""
        def run() -> None:
            popover.close()
            command()
        return run

    # ── Recording sheet ──────────────────────────────────────────

    def _sheet_entry(self: ECGApp, parent, row: int, col: int, label: str, attr: str,
                     default: str) -> ctk.CTkEntry:
        """Label above an entry in cell (row, col); bound to self.ent_<attr>."""
        cell = ctk.CTkFrame(parent, fg_color="transparent")
        cell.grid(row=row, column=col, sticky="new", padx=(0, SPACE_M) if col == 0 else 0,
                  pady=(0, SPACE_S))
        ctk.CTkLabel(cell, text=label, font=FONT_SMALL, text_color=MUTED,
                     anchor="w", height=18).pack(fill="x")
        ent = ctk.CTkEntry(cell, width=_SHEET_ENTRY_W, **input_style())
        ent.insert(0, default)
        ent.pack(fill="x")
        setattr(self, f"ent_{attr}", ent)
        return ent

    def _build_recording_sheet(self: ECGApp) -> None:
        self.recording_sheet = Popover(self, title="Recording")
        b = self.recording_sheet.body

        self.lbl_file = ctk.CTkLabel(
            b, text="No file loaded", font=FONT_CARD_TITLE, text_color=MUTED,
            anchor="w", justify="left", wraplength=2 * _SHEET_ENTRY_W)
        self.lbl_file.pack(fill="x")
        self.lbl_npeaks = ctk.CTkLabel(
            b, text="Peaks detected: —", font=FONT_SMALL, text_color=BLUE, anchor="w")
        self.lbl_npeaks.pack(fill="x", pady=(0, SPACE_S))

        g = ctk.CTkFrame(b, fg_color="transparent")
        g.pack(fill="x")
        g.grid_columnconfigure((0, 1), weight=1, uniform="sheet")

        # Channel, with the list of the file's channels beside it.
        ch = self._sheet_entry(g, 0, 0, "Channel", "channel", "ECG")
        ch.pack_configure(side="left", fill="x", expand=True)
        ch.configure(width=_SHEET_ENTRY_W - 60)
        ctk.CTkButton(
            ch.master,
            text="List…",
            width=54,
            command=self._show_channels,
            **button_style("secondary"),
        ).pack(side="left", padx=(SPACE_XS, 0))
        # Sampling rate, with where it came from (file or entered) under it.
        self._sheet_entry(g, 0, 1, "Sampling rate (Hz)", "fs", str(MouseECG.FS_DEFAULT))
        self.lbl_fs_source = ctk.CTkLabel(
            self.ent_fs.master, text="", font=FONT_KPI_LABEL, text_color=MUTED,
            anchor="w", justify="left", wraplength=_SHEET_ENTRY_W)
        self.lbl_fs_source.pack(fill="x")

        self._sheet_entry(g, 1, 0, "Subject ID", "subject", "subject_01")
        for seq in ("<FocusOut>", "<Return>"):      # the file chip shows it
            self.ent_subject.bind(seq, lambda _e: self._update_topbar_filename())  # type: ignore[union-attr]
        # Project name: a label spanning the whole working session (many
        # recordings), not per-recording metadata like Subject ID --
        # reset_for_new_file() never touches it, so it survives opening a
        # different recording in the same session.
        self._sheet_entry(g, 1, 1, "Project", "project_name", "")
        self.ent_project_name.bind("<FocusOut>", self._on_project_name_change)  # type: ignore[union-attr]
        self.ent_project_name.bind("<Return>", self._on_project_name_change)  # type: ignore[union-attr]
        # Crop: what part of the file is read at all (0 = start / end), unlike
        # the analysis window of the context strip, which is a part of it.
        self._sheet_entry(g, 2, 0, "Crop start (s)", "t_start", "0")
        self._sheet_entry(g, 2, 1, "Crop end (s)", "t_end", "0")
        ctk.CTkLabel(b, text="Crop: the part of the file read at Detect Peaks (0 = start / end).",
                     font=FONT_KPI_LABEL, text_color=LIGHT, anchor="w", justify="left",
                     wraplength=2 * _SHEET_ENTRY_W).pack(fill="x", pady=(0, SPACE_S))

        ctk.CTkButton(
            b,
            text="📝  Notes (this recording)…",
            anchor="w",
            command=self._menu_command(
                          self.recording_sheet, self._open_notes_dialog),
            **button_style("secondary"),
        ).pack(fill="x")

        ctk.CTkFrame(b, height=1, fg_color=BORDER).pack(fill="x", pady=SPACE_S)
        self.lbl_session_info = ctk.CTkLabel(
            b, text="No session saved for this file", font=FONT_HINT,
            text_color=MUTED, anchor="w", justify="left", wraplength=2 * _SHEET_ENTRY_W)
        self.lbl_session_info.pack(fill="x")
        ctk.CTkLabel(b, text="Clear session cache: ⋯ menu", font=FONT_KPI_LABEL,
                     text_color=LIGHT, anchor="w").pack(fill="x")

    def _toggle_recording_sheet(self: ECGApp) -> None:
        self.recording_sheet.toggle(self.file_chip)

    def _open_recording_sheet(self: ECGApp) -> None:
        """Open (never close) the sheet: the fs warning points to it."""
        sheet = getattr(self, "recording_sheet", None)
        if sheet is not None and not sheet.is_open():
            sheet.open(self.file_chip)

    # ── ⋯ menu / recent files ────────────────────────────────────

    def _toggle_more_menu(self: ECGApp) -> None:
        self.more_menu.toggle(self.btn_more_menu)

    def _toggle_recent_menu(self: ECGApp) -> None:
        self.recent_menu.toggle(self.btn_recent)

    def _fill_recent_menu(self: ECGApp) -> None:
        """The last files opened (newest first), then the full registry view."""
        b = self.recent_menu.body
        for child in b.winfo_children():
            child.destroy()
        paths = [p for p in self._recent if os.path.exists(p)][:_RECENT_SHOWN]
        if not paths:
            ctk.CTkLabel(b, text="No recent files yet", font=FONT_SMALL, text_color=MUTED,
                         anchor="w").pack(fill="x", pady=(0, SPACE_XS))
        for p in paths:
            btn = menu_button(b, os.path.basename(p), self._menu_command(
                self.recent_menu, lambda p=p: self._load_path(p)))
            self._bind_hover_tip(btn, p)
        ctk.CTkFrame(b, height=1, fg_color=BORDER).pack(fill="x", pady=SPACE_XS)
        menu_button(b, "All recent recordings (stats, cohort trends)…",
                    self._menu_command(self.recent_menu, self._open_recent))
