# -*- coding: utf-8 -*-
"""
ecg.ui.tab_summary
------------------
ECGApp Summary tab construction.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from ecg.ui.theme import (
    PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, PLOT, BLUE, BLUE_DARK, PINK,
    GREEN_DARK, ORANGE_DARK, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_LABEL,
    FONT_SMALL, FONT_MONO, FONT_KPI_VALUE, FONT_KPI_LABEL, FONT_SIDEBAR_HDR,
    FONT_MICRO, FONT_SUBSECTION,
)
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_SUMMARY

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class TabSummaryMixin:
    """ECGApp Summary tab construction."""

    def _build_tab_summary(self: ECGApp) -> None:
        """Summary tab: a curated verdict, not a wall of duplicate plots.

        Every plot mirrored here used to also exist full-size in its own tab
        (HRV > Frequency/Non-linear/Rolling, Intervals, Beat Template) --
        showing it again at ~1/3 the size with the same font/legend/tick
        density as the full-tab version just looked cramped. The global KPI
        bar above the tabs already shows the headline numbers, so this tab
        no longer duplicates them either. What's left: a Signal Quality
        panel (surfacing numbers the app already computed but never showed
        anywhere -- mean beat-to-template correlation, % of low-quality
        beats, artifact-correction counts), the two densest kept visuals
        (RR tachogram, Poincare), the RR-asymmetry breakdown (unique to this
        tab, not a duplicate), a plain metrics table with no physiological
        reference-range judgments, and the detailed text report.
        """
        t = self.tabs.tab(TAB_SUMMARY)
        t.grid_rowconfigure(0, weight=0)   # verdict banner
        t.grid_rowconfigure(1, weight=0)   # action bar
        t.grid_rowconfigure(2, weight=1)   # scrollable body
        t.grid_columnconfigure(0, weight=1)

        # ── Verdict banner ───────────────────────────────────────────────────
        banner = ctk.CTkFrame(t, fg_color=PANEL, corner_radius=0, height=44)
        banner.grid(row=0, column=0, sticky="ew")
        banner.pack_propagate(False)
        self.lbl_sum_verdict = ctk.CTkLabel(
            banner, text="Run analysis to see the signal-quality summary.",
            font=FONT_KPI_LABEL, text_color=MUTED, anchor="w")
        self.lbl_sum_verdict.pack(side="left", padx=SPACE_L, pady=SPACE_S)
        ctk.CTkFrame(t, height=1, fg_color=BORDER).grid(row=0, column=0, sticky="sew")

        # ── Action bar ────────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(t, fg_color="transparent", height=40)
        btn_row.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, 0))
        btn_row.pack_propagate(False)
        for label, cmd in [
            ("📋  Copy Report",      self._copy_summary),
            ("💾  Save .txt",        self._save_summary_txt),
        ]:
            ctk.CTkButton(btn_row, text=label, command=cmd,
                          fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                          font=FONT_SMALL, height=28, corner_radius=5).pack(
                side="left", padx=(0, SPACE_S))
        # Separate attribute from the Epochs view's own lbl_epoch_info --
        # both used to share the name, so whichever view built last silently
        # clobbered the other's label reference and the first one could
        # never be updated again.
        self.lbl_summary_epoch_info = ctk.CTkLabel(btn_row, text="", font=FONT_SMALL,
                                                     text_color=MUTED)
        self.lbl_summary_epoch_info.pack(side="right")

        # ── Scrollable body ───────────────────────────────────────────────────
        outer_scroll = ctk.CTkScrollableFrame(
            t, fg_color=PANEL,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER2,
        )
        outer_scroll.grid(row=2, column=0, sticky="nsew", padx=0, pady=(SPACE_S, 0))
        outer_scroll.grid_columnconfigure(0, weight=1)
        self._sum_scroll = outer_scroll

        # ── Layout helpers ────────────────────────────────────────────────────
        def _section(parent, title: str, color: str = MUTED,
                     subtitle: str = "") -> ctk.CTkFrame:
            """Titled section with accent stripe, returns the content frame."""
            hdr = ctk.CTkFrame(parent, fg_color="transparent")
            hdr.pack(fill="x", padx=SPACE_M, pady=(SPACE_L, SPACE_S))
            stripe = ctk.CTkFrame(hdr, height=3, fg_color=color, corner_radius=2)
            stripe.pack(fill="x", pady=(0, SPACE_S))
            title_row = ctk.CTkFrame(hdr, fg_color="transparent")
            title_row.pack(fill="x")
            ctk.CTkLabel(title_row, text=title.upper(),
                         font=FONT_SIDEBAR_HDR,
                         text_color=color, anchor="w").pack(side="left")
            if subtitle:
                ctk.CTkLabel(title_row, text=f"  {subtitle}",
                             font=FONT_KPI_LABEL, text_color=MUTED,
                             anchor="w").pack(side="left", padx=(SPACE_S, 0))
            body = ctk.CTkFrame(parent, fg_color="transparent")
            body.pack(fill="x", padx=SPACE_S, pady=(SPACE_XS, 0))
            return body

        def _row(parent, specs: "list[tuple[str,int,int]]") -> None:
            """specs = [(key, height_px, weight), ...]  — one horizontal row."""
            row_frame = ctk.CTkFrame(parent, fg_color="transparent")
            row_frame.pack(fill="x", pady=(0, SPACE_S))
            for i, (key, h, w) in enumerate(specs):
                pad_right = i < len(specs) - 1
                card = ctk.CTkFrame(row_frame, fg_color=PANEL, corner_radius=6, height=h)
                px = (0, 4) if pad_right else (0, 0)
                card.pack(side="left", fill="both", expand=True, padx=px)
                card.pack_propagate(False)
                inner = tk.Frame(card, bg=PLOT["bg"], bd=0, highlightthickness=0)
                inner.pack(fill="both", expand=True, padx=SPACE_XS, pady=SPACE_XS)
                self._slots[key] = CanvasSlot(inner, 10 * w, h / 100, toolbar=False)

        # ━━━ SECTION 1 — Signal Quality ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        sec_q = _section(outer_scroll, "✓  Signal Quality", GREEN_DARK,
                         subtitle="how much to trust this recording, not the physiology itself")
        stat_row = ctk.CTkFrame(sec_q, fg_color=CARD, corner_radius=6)
        stat_row.pack(fill="x", pady=(0, SPACE_S))
        stat_grid = ctk.CTkFrame(stat_row, fg_color="transparent")
        stat_grid.pack(fill="x", padx=SPACE_L, pady=SPACE_M)
        for col in range(5):
            stat_grid.grid_columnconfigure(col, weight=1, uniform="sq")
        _SQ_DEFS = [
            ("sq_score",   "Overall score"),
            ("sq_corr",    "Mean template correlation"),
            ("sq_badbeats","Beats below 0.90 corr."),
            ("sq_noisy_time", "Time below 0.90 corr."),
            ("sq_artifact","Auto-corrected"),
        ]
        self._sum_quality_vals: "dict[str, ctk.CTkLabel]" = {}
        for i, (key, label) in enumerate(_SQ_DEFS):
            cell = ctk.CTkFrame(stat_grid, fg_color="transparent")
            cell.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else SPACE_M, 0))
            ctk.CTkLabel(cell, text=label, font=FONT_KPI_LABEL,
                         text_color=MUTED, anchor="w").pack(anchor="w")
            val_lbl = ctk.CTkLabel(cell, text="—", font=FONT_KPI_VALUE,
                                   text_color=TEXT, anchor="w")
            val_lbl.pack(anchor="w")
            self._sum_quality_vals[key] = val_lbl
        _row(sec_q, [("sum_quality_time", 180, 1)])

        # ━━━ SECTION 2 — Rythme cardiaque ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        sec1 = _section(outer_scroll, "❤  Heart Rate & Rhythm", ORANGE_DARK,
                        subtitle="densest single views — full plots live in their own tabs")
        _row(sec1, [("sum_rr", 260, 3), ("sum_poincare", 260, 2)])
        _row(sec1, [("sum_asymmetry", 220, 1)])

        # ━━━ SECTION 3 — Metrics table ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        sec2 = _section(outer_scroll, "▤  Metrics", BLUE_DARK,
                        subtitle="values only — no reference-range judgment")
        metrics_card = ctk.CTkFrame(sec2, fg_color=CARD, corner_radius=6)
        metrics_card.pack(fill="x", pady=(0, SPACE_S))
        metrics_body = ctk.CTkFrame(metrics_card, fg_color="transparent")
        metrics_body.pack(fill="x", padx=SPACE_L, pady=SPACE_M)

        _METRIC_GROUPS = [
            ("Rate",                  ORANGE_DARK, [("hr_mean", "HR mean", "bpm"),
                                                     ("hr_range", "HR range", "bpm")]),
            ("Time-domain HRV",       BLUE_DARK,   [("sdnn", "SDNN", "ms"),
                                                     ("rmssd", "RMSSD", "ms"),
                                                     ("pnn6", "pNN6", "%")]),
            ("Frequency / non-linear", "#37474F",  [("lf_hf", "LF/HF", ""),
                                                     ("sampen", "SampEn", ""),
                                                     ("dfa1", "DFA α1", "")]),
            ("Intervals (delineation)", PINK,      [("pr", "PR", "ms"),
                                                     ("qrs", "QRS", "ms"),
                                                     # Uncorrected QT next to QTc.
                                                     ("qt", "QT", "ms"),
                                                     ("qtc", "QTc", "ms")]),
        ]
        self._sum_metric_vals: "dict[str, ctk.CTkLabel]" = {}
        for gi, (gtitle, gcolor, rows) in enumerate(_METRIC_GROUPS):
            ctk.CTkLabel(metrics_body, text=gtitle.upper(), font=FONT_SUBSECTION,
                         text_color=gcolor, anchor="w").pack(
                anchor="w", pady=(SPACE_S if gi else 0, SPACE_XS))
            for key, name, unit in rows:
                r = ctk.CTkFrame(metrics_body, fg_color="transparent")
                r.pack(fill="x")
                ctk.CTkLabel(r, text=name, font=FONT_LABEL, text_color=TEXT,
                             anchor="w", width=160).pack(side="left")
                val_lbl = ctk.CTkLabel(r, text="—", font=FONT_LABEL,
                                       text_color=TEXT, anchor="e", width=80)
                val_lbl.pack(side="left")
                if unit:
                    ctk.CTkLabel(r, text=unit, font=FONT_MICRO, text_color=MUTED,
                                 anchor="w").pack(side="left", padx=(SPACE_XS, 0))
                self._sum_metric_vals[key] = val_lbl

        linkout = ctk.CTkFrame(sec2, fg_color=CARD, corner_radius=6)
        linkout.pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkLabel(linkout, text="Full plots for every metric above live in their own tabs.",
                     font=FONT_SMALL, text_color=MUTED, anchor="w").pack(
            side="left", padx=SPACE_M, pady=SPACE_S)
        ctk.CTkLabel(linkout, text="HRV · Intervals · Beat Template · Abnormal Events  →",
                     font=FONT_SMALL, text_color=BLUE, anchor="e").pack(
            side="right", padx=SPACE_M, pady=SPACE_S)

        # ━━━ SECTION 4 — Rapport texte ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        sec3 = _section(outer_scroll, "📝  Detailed Report", MUTED)
        txt_card = ctk.CTkFrame(sec3, fg_color=PANEL, corner_radius=6)
        txt_card.pack(fill="x", pady=(0, SPACE_L))
        self.txt_sum = ctk.CTkTextbox(txt_card, font=FONT_MONO, fg_color=PANEL,
                                      text_color=TEXT, border_width=0, height=380)
        self.txt_sum.pack(fill="both", expand=True, padx=SPACE_M, pady=SPACE_M)
