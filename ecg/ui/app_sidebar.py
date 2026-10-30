# -*- coding: utf-8 -*-
"""
ecg.ui.app_sidebar
------------------
ECGApp left sidebar construction.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from ecg.core.models import MouseECG
from ecg.ui.theme import (
    THEME, BG, PANEL, BORDER, BORDER2, TEXT, MUTED, LIGHT, BLUE, TEAL,
    TEAL_DARK, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_TITLE, FONT_LABEL,
    FONT_SMALL, FONT_KPI_LABEL, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_MICRO,
    FONT_HINT, FONT_SUBSECTION, FONT_CARD_TITLE,
)
from ecg.ui.sidebar import CollapsibleSection

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class SidebarMixin:
    """ECGApp left sidebar construction."""

    # ─── Sidebar ──────────────────────────────────────────────

    def _build_sidebar(self: ECGApp) -> None:
        """Compact sidebar: minimal always-visible workflow + scrollable settings."""
        px = dict(padx=SPACE_L)

        # ══════════════════════════════════════════════════════
        #  FIXED TOP — title, file, workflow actions, threshold
        # ══════════════════════════════════════════════════════
        top = ctk.CTkFrame(self.sidebar, fg_color=PANEL, corner_radius=0)
        top.pack(fill="x", side="top")

        # Title + version
        title_row = ctk.CTkFrame(top, fg_color="transparent")
        title_row.pack(fill="x", padx=SPACE_L, pady=(SPACE_M, SPACE_XS))
        ctk.CTkLabel(title_row, text="ECG Analysis",
                     font=FONT_TITLE, text_color=TEXT,
                     anchor="w").pack(side="left")
        ctk.CTkLabel(title_row, text="v6",
                     font=FONT_HINT, text_color=LIGHT).pack(side="right")

        # File label
        self.lbl_file = ctk.CTkLabel(
            top, text="No file loaded", font=FONT_SMALL,
            text_color=MUTED, wraplength=260, anchor="w", justify="left")
        self.lbl_file.pack(**px, pady=(0, SPACE_XS), fill="x")

        # Channel / Recent on their own row -- the "Open .mat file" button
        # that used to share this row duplicated the toolbar's own Open
        # (see _build_toolbar), and the 3-column grid clipped its label to
        # "en .mat". Channel/Recent now get the full width and a height/
        # radius matching other secondary sidebar buttons instead of the
        # cramped height=10 they had as the row's minor columns.
        open_row = ctk.CTkFrame(top, fg_color="transparent")
        open_row.pack(fill="x", expand=True, **px, pady=(0, SPACE_XS))
        open_row.rowconfigure(0, weight=1)
        open_row.columnconfigure(0, weight=1)
        open_row.columnconfigure(1, weight=1)
        ctk.CTkButton(open_row, text="Channel", height=30,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FONT_BTN_SEC, corner_radius=8,
                      command=self._show_channels).grid(row=0, column=0, sticky="ewns", padx=(0, SPACE_XS))
        ctk.CTkButton(open_row, text="Recent ▾", height=30,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FONT_BTN_SEC, corner_radius=8,
                      command=self._open_recent).grid(row=0, column=1, sticky="ewns")

        ctk.CTkFrame(top, height=1, fg_color=BORDER).pack(fill="x", padx=SPACE_L, pady=(SPACE_XS, SPACE_XS))

        # Peak count status
        self.lbl_npeaks = ctk.CTkLabel(
            top, text="Peaks detected: —",
            font=FONT_CARD_TITLE, text_color=BLUE, anchor="w")
        self.lbl_npeaks.pack(**px, fill="x", pady=(0, SPACE_XS))

        # Method/Analysis-window/Status/Threshold relocated to the right
        # panel's DETECTION accordion section (_build_detection_section()).

        # ══════════════════════════════════════════════════════
        #  SCROLLABLE SETTINGS
        # ══════════════════════════════════════════════════════
        scroll = ctk.CTkScrollableFrame(
            self.sidebar, fg_color=PANEL,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER2)
        scroll.pack(fill="both", expand=True, side="top")
        s = scroll
        fpx = dict(padx=SPACE_L)

        # ── SIGNAL ────────────────────────────────────────────
        sec_sig = CollapsibleSection(s, "SIGNAL", initially_open=False)
        self.sec_signal = sec_sig
        f = sec_sig.frame
        # Project name: a label spanning the whole working session (many
        # recordings), not per-recording metadata like Subject ID below --
        # like every other sidebar entry, reset_for_new_file() never touches
        # it, so it naturally survives opening a different recording in the
        # same session with no special-case logic needed.
        self._sidebar_entry(f, "Project name", "project_name", "", fpx)
        self.ent_project_name.bind("<FocusOut>", self._on_project_name_change)  # type: ignore[union-attr]
        self.ent_project_name.bind("<Return>", self._on_project_name_change)  # type: ignore[union-attr]
        self._sidebar_entry_row(f, fpx, [
            ("Channel", "channel", "ECG"),
            ("Subject ID", "subject", "subject_01"),
        ])
        self._btn(f, "📝  Notes (this recording)", self._open_notes_dialog, fpx, variant="secondary", h=28)
        self._sidebar_entry(f, "Sampling rate (Hz)", "fs",
                            str(MouseECG.FS_DEFAULT), fpx)
        self.lbl_fs_source = ctk.CTkLabel(
            f, text="", font=FONT_KPI_LABEL, text_color=MUTED,
            anchor="w", wraplength=230)
        self.lbl_fs_source.pack(**fpx, fill="x", pady=(0, SPACE_XS))
        self._sidebar_entry_row(f, fpx, [
            ("Crop start (s)", "t_start", "0"),
            ("Crop end (s)",   "t_end",   "0"),
        ])

        # ── DETECTION ─────────────────────────────────────────
        self._build_detection_section(s)

        # ── FILTERS ───────────────────────────────────────────
        # sw_filtering ON = DSP filtering is applied (band-pass/notch/
        # cleaning); OFF (default) = raw signal, matching the app's
        # raw-by-default behaviour on file load. DISPLAY & PREVIEW controls
        # (show raw, invert, live preview) always stay interactive -- pure
        # display toggles, or apply regardless of sw_filtering. FILTER
        # SETTINGS (notch/band-pass/cleaning) stay visible but greyed out
        # when filtering is off, so the user can see/set them in advance --
        # see on_filtering_toggle in detection_controller.py.
        sec_flt = CollapsibleSection(s, "FILTERS", initially_open=False)
        f = sec_flt.frame
        self.sw_filtering = self._switch(f, "Filtering", fpx, default_on=False)
        self.sw_filtering.configure(command=self._on_filtering_toggle)
        ctk.CTkLabel(f, text="Off analyzes the raw signal. On applies the settings below.",
                     font=FONT_HINT, text_color=MUTED,
                     anchor="w", wraplength=230).pack(**fpx, fill="x", pady=(0, SPACE_XS))
        self.lbl_filter_summary = ctk.CTkLabel(
            f, text="Processing\n•  Raw signal", font=FONT_HINT, text_color=MUTED,
            anchor="w", justify="left", wraplength=230)
        self.lbl_filter_summary.pack(**fpx, fill="x", pady=(0, SPACE_S))

        ctk.CTkLabel(f, text="DISPLAY & PREVIEW", font=FONT_SUBSECTION,
                     text_color=MUTED, anchor="w").pack(**fpx, fill="x", pady=(0, SPACE_XS))
        self.sw_show_raw = self._switch(
            f, "Show raw signal (vs filtered)", fpx, default_on=True)
        self.sw_show_raw.configure(command=self._on_show_raw_toggle)
        self.sw_invert_signal = self._switch(
            f, "⟳  Invert signal (polarity)", fpx, default_on=False)
        # A plain parameter -- doesn't auto-run the full pipeline on toggle.
        # Picked up on the next "Preview Detection" click, or immediately in
        # the before/after overlay if "Preview filtering" is on, exactly
        # like every other control in this section.
        self.sw_invert_signal.configure(command=self._refresh_filter_preview)
        ctk.CTkLabel(f, text="Useful if R peaks appear negative",
                     font=FONT_HINT, text_color=MUTED,
                     anchor="w", wraplength=230).pack(**fpx, fill="x", pady=(0, SPACE_S))
        self.sw_filter_preview = self._switch(f, "👁  Preview filtering", fpx, default_on=False)
        self.sw_filter_preview.configure(command=self._on_filter_preview_toggle)
        ctk.CTkLabel(f, text="Overlays the filtered signal on the visible window,\n"
                             "using the settings below — no need to click Detect Peaks.",
                     font=FONT_HINT, text_color=MUTED,
                     anchor="w", justify="left", wraplength=230).pack(**fpx, fill="x", pady=(0, SPACE_S))

        ctk.CTkFrame(f, height=1, fg_color=BORDER).pack(
            fill="x", padx=SPACE_M, pady=(0, SPACE_XS))
        ctk.CTkLabel(f, text="FILTER SETTINGS", font=FONT_SUBSECTION,
                     text_color=MUTED, anchor="w").pack(**fpx, fill="x", pady=(0, SPACE_XS))
        self._filter_advanced_group = ctk.CTkFrame(f, fg_color="transparent")
        self._filter_advanced_group.pack(**fpx, fill="x")
        g = self._filter_advanced_group
        self.sw_notch = self._switch(g, "Notch 50 Hz", dict(padx=0), default_on=False)
        self.sw_notch.configure(command=self._refresh_filter_preview)
        ctk.CTkLabel(g, text="Band-pass filter", font=FONT_SMALL,
                     text_color=MUTED, anchor="w").pack(anchor="w", pady=(SPACE_S, 0))
        # Labelled by frequency position (Low/High), matching filtering.py's
        # own bandpass(lo, hi) naming -- not by filter-type jargon (HP/LP),
        # which technically read correctly (HP corner = the low edge, LP
        # corner = the high edge) but was confusing at a glance.
        self._sidebar_entry_row(g, dict(padx=0), [
            ("Low cutoff (Hz)",  "lp", str(MouseECG.BP_LO_HZ)),
            ("High cutoff (Hz)", "hp", str(int(MouseECG.BP_HI_HZ))),
        ])
        for _ent in (self.ent_lp, self.ent_hp):
            if _ent is not None:
                _ent.bind("<Return>",   lambda _e: self._refresh_filter_preview())
                _ent.bind("<FocusOut>", lambda _e: self._refresh_filter_preview())
        ctk.CTkLabel(g, text="Signal cleaning:", font=FONT_SMALL,
                     text_color=MUTED, anchor="w").pack(anchor="w", pady=(SPACE_XS, 0))
        self.cb_clean = ctk.CTkComboBox(
            g, font=FONT_LABEL, height=28, fg_color=BG,
            border_color=BORDER2, button_color=BORDER2, text_color=TEXT,
            dropdown_fg_color=BG, dropdown_text_color=TEXT,
            values=["neurokit", "pantompkins1985", "elgendi2010", "hamilton2002", "biosppy"],
            command=lambda _v: self._refresh_filter_preview())
        self.cb_clean.set("neurokit")
        self.cb_clean.pack(fill="x", pady=(SPACE_XS, SPACE_S))

        # ── ARTIFACTS ─────────────────────────────────────────
        self._build_artifacts_section(s)

        # ── ANNOTATIONS ───────────────────────────────────────
        self._build_annotations_section(s)

        # ── BOTTOM BUTTONS ────────────────────────────────────
        # Session info + cache-clearing sit here (not in a named accordion
        # section) since neither the left panel's spec (Signal/Detection/
        # Filters/Artifacts/Annotations) nor the right panel's (Statistics/
        # Exports/ML Detector Saving) has room for them -- this area is
        # already the app's catch-all for cross-cutting maintenance actions.
        # Save Session itself lives in the top toolbar's File ops chip
        # (self.btn_save_session, built in _build_toolbar()).
        ctk.CTkFrame(s, height=1, fg_color=BORDER).pack(fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        self.lbl_session_info = ctk.CTkLabel(
            s, text="No session saved for this file", font=FONT_HINT,
            text_color=MUTED, anchor="w", wraplength=230, justify="left")
        self.lbl_session_info.pack(padx=SPACE_M, pady=(0, SPACE_S), fill="x")
        self._btn(s, "🗑  Clear Session Cache", self._delete_session, dict(padx=SPACE_M), variant="secondary", h=26)
        ctk.CTkFrame(s, height=1, fg_color=BORDER).pack(fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        btn_compare_segments = ctk.CTkButton(
            s, text="⚖  Compare Segments",
            command=self._open_compare_segments,
            # Dark mode lightens TEAL for legibility against the dark
            # background (_adapt_for_mode in theme.py), which then leaves
            # white label text well under WCAG contrast on the same fill --
            # a dark label reads clearly against the lightened dark-mode
            # TEAL instead.
            fg_color=TEAL, hover_color=TEAL_DARK,
            text_color=(BG if THEME.is_dark else "white"),
            font=FONT_SIDEBAR_HDR, height=28, corner_radius=8,
        )
        btn_compare_segments.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_XS))
        # Cross-reference the app's two other comparison features, which
        # otherwise live in unrelated places (DETECTION section, Recent
        # recordings popup) with nothing tying the three together.
        self._bind_hover_tip(
            btn_compare_segments,
            "Other comparisons: Check Agreement (Detection section) · "
            "Cohort Trends (Recent recordings)")
        ctk.CTkButton(
            s, text="↺  Reset to Mouse ECG Defaults",
            command=self._reset_params,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            font=FONT_HINT, height=24, corner_radius=5,
        ).pack(fill="x", padx=SPACE_M, pady=(0, SPACE_XS))
        lbl_shortcuts_help = ctk.CTkLabel(
            s, text="F1 — keyboard shortcuts",
            font=FONT_MICRO, text_color=LIGHT, anchor="center",
            cursor="hand2")
        lbl_shortcuts_help.pack(pady=(0, SPACE_S))
        lbl_shortcuts_help.bind("<Button-1>", lambda _e: self._show_shortcuts_help())
