# -*- coding: utf-8 -*-
"""
ecg.ui.tab_hrv
--------------
ECGApp Rhythm & HRV tab: the overview page and its handlers.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from ecg.core.analysis import RR_ASYMMETRY_TEXT
from ecg.core.metric_table import MetricTable
from ecg.core.models import MouseECG
from ecg.ui.theme import (
    THEME, BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, BLUE, BLUE_HOVER, PURPLE,
    PURPLE_DARK, SPACE_XS, SPACE_S, SPACE_M, FONT_SECTION_HDR, FONT_LABEL, FONT_SMALL,
    FONT_MONO, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT,
)
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_HRV

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class TabHrvMixin:
    """ECGApp Rhythm & HRV tab: overview page and its handlers."""

    def _build_tab_hrv_unified(self: ECGApp) -> None:
        """Rhythm & HRV tab: one scrollable page."""
        t = self.tabs.tab(TAB_HRV)
        t.grid_rowconfigure(0, weight=1)
        t.grid_columnconfigure(0, weight=1)
        page = ctk.CTkFrame(t, fg_color="transparent")
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)
        self._build_hrv_view_overview(page)

    # ── Overview: one scrollable page ─────────────────────────────────────

    def _rhythm_section(self: ECGApp, parent, row: int, title: str) -> ctk.CTkFrame:
        """A section title in grid row *row* of *parent*; returns its header
        frame, for what sits next to the title."""
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.grid(row=row, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_M, SPACE_XS))
        ctk.CTkLabel(hdr, text=title.upper(), font=FONT_SIDEBAR_HDR,
                     text_color=MUTED).pack(side="left")
        return hdr

    def _rhythm_plot_card(self: ECGApp, parent, height: int, slot: str, width_in: float,
                          placeholder: str = "Click Analyze to display") -> ctk.CTkFrame:
        """A fixed-height card holding the figure of *slot*: inside a scrolling
        page a figure has no available height to fit."""
        card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0, height=height)
        card.pack_propagate(False)
        self._slots[slot] = CanvasSlot(card, width_in, height / 100, toolbar=False,
                                       placeholder=placeholder)
        return card

    def _build_hrv_view_overview(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Rhythm & HRV: KPI cards, tachogram next to Poincaré, metric table next
        to the spectrum, non-linear details, RR distribution, and the raw
        NeuroKit tables behind Show all. Numbers come from the shared table
        (PlotController.plot_rhythm_view)."""
        page = ctk.CTkScrollableFrame(parent, fg_color=PANEL,
                                      scrollbar_button_color=BORDER,
                                      scrollbar_button_hover_color=BORDER2)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1)
        self._rhythm_page = page
        self._rhythm_labels: "dict[str, ctk.CTkLabel]" = {}

        self._build_rhythm_cards(page, 0)
        self._build_rhythm_plots(page, 1)
        self._build_rhythm_metrics(page, 3)
        self._build_rhythm_nonlinear(page, 5)
        self._build_rhythm_distribution(page, 9)
        self._build_rhythm_raw(page, 11)
        self.plot_ctrl.plot_rhythm_view(
            getattr(self.plot_ctrl, "_rhythm_table", None) or MetricTable())

    def _build_rhythm_cards(self: ECGApp, page, row: int) -> None:
        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.grid(row=row, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_M, 0))
        self._rhythm_cards: "dict[str, dict]" = {}
        for i, (key, name) in enumerate((("hr", "Heart rate"), ("sdnn", "SDNN"), ("rmssd", "RMSSD"),
                                         ("pnn6", "pNN6"), ("lf_hf", "LF / HF"),
                                         ("sd1_sd2", "SD1 / SD2"))):
            bar.grid_columnconfigure(i, weight=1, uniform="kpi")
            card = ctk.CTkFrame(bar, fg_color=CARD, corner_radius=6,
                                border_width=1, border_color=BORDER)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else SPACE_S, 0))
            ctk.CTkLabel(card, text=name, font=FONT_SMALL, text_color=MUTED,
                         anchor="w").pack(fill="x", padx=SPACE_M, pady=(SPACE_S, 0))
            line = ctk.CTkFrame(card, fg_color="transparent")
            line.pack(fill="x", padx=SPACE_M)
            value = ctk.CTkLabel(line, text="—", font=FONT_SECTION_HDR, text_color=TEXT, width=0)
            value.pack(side="left")
            unit = ctk.CTkLabel(line, text="", font=FONT_HINT, text_color=MUTED, width=0)
            unit.pack(side="left", padx=(SPACE_XS, 0), pady=(SPACE_XS, 0))
            note = ctk.CTkLabel(card, text="", font=FONT_HINT, text_color=MUTED, anchor="w",
                                justify="left", wraplength=int(150 * THEME.font_scale))
            note.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_S))
            self._rhythm_cards[key] = {"value": value, "unit": unit, "note": note}

    def _build_rhythm_plots(self: ECGApp, page, row: int) -> None:
        hdr = self._rhythm_section(page, row, "Tachogram and Poincaré")
        self._rhythm_labels["excluded"] = ctk.CTkLabel(
            hdr, text="", font=FONT_SMALL, text_color=MUTED)
        self._rhythm_labels["excluded"].pack(side="left", padx=(SPACE_M, 0))

        _spk_on = self.ui.show_spike_markers
        self.btn_toggle_spikes = ctk.CTkButton(
            hdr, text="▲▼ Spike markers", width=140, height=26, font=FONT_SMALL,
            # Outline "on" look, not a solid fill: a solid BLUE toggle reads as
            # a command button.
            fg_color=PANEL if _spk_on else BORDER,
            hover_color=BORDER2,
            text_color=BLUE if _spk_on else MUTED,
            border_width=2 if _spk_on else 0,
            border_color=BLUE,
            corner_radius=8,
            command=self._toggle_spike_markers,
        )
        self.btn_toggle_spikes.pack(side="right")  # type: ignore[union-attr]
        # "Review" jumps to the next flagged beat (▲/▼) of this tachogram, in
        # Detection, centred in the signal view: the same step_spike() as the
        # Detection tab's "Spike: ◀ ▶" stepper, reachable from here. Pressing
        # it again walks every flagged beat in turn.
        self.btn_review_spikes = ctk.CTkButton(
            hdr, text="🔍 Review", width=100, height=26, font=FONT_SMALL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8,
            command=lambda: self._step_spike(+1),
        )
        self.btn_review_spikes.pack(side="right", padx=(0, SPACE_S))  # type: ignore[union-attr]
        self._bind_hover_tip(
            self.btn_review_spikes,
            "Jump to the next flagged beat (▲/▼) — press again to step "
            "through the rest")

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=row + 1, column=0, sticky="ew", padx=SPACE_M)
        body.grid_columnconfigure(0, weight=3, uniform="plots")
        body.grid_columnconfigure(1, weight=2, uniform="plots")
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_S))
        self._rhythm_plot_card(left, 440, "rr", 8.0).pack(fill="x")
        ctk.CTkLabel(
            left, text=("Left-click: go to time  ·  right-click: next spike  ·  ▲ sudden "
                        "acceleration  ▼ sudden deceleration  ·  amber >2.5 SD  ·  red >3.5 SD"),
            font=FONT_HINT, text_color=MUTED, anchor="w",
        ).pack(fill="x", pady=(SPACE_XS, 0))
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        self._rhythm_plot_card(right, 440, "poincare", 5.0).pack(fill="x")
        ctk.CTkLabel(
            right, text="Successive valid pairs only: no pair crosses an excluded interval.",
            font=FONT_HINT, text_color=MUTED, anchor="w",
        ).pack(fill="x", pady=(SPACE_XS, 0))

    def _build_rhythm_metrics(self: ECGApp, page, row: int) -> None:
        hdr = self._rhythm_section(page, row, "Metrics and power spectrum")
        self.btn_rhythm_show_all = ctk.CTkButton(
            hdr, text="", width=200, height=26, font=FONT_SMALL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT, corner_radius=8,
            command=self._toggle_rhythm_show_all)
        self.btn_rhythm_show_all.pack(side="left", padx=(SPACE_M, 0))

        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=row + 1, column=0, sticky="ew", padx=SPACE_M)
        body.grid_columnconfigure(0, weight=1, uniform="metrics")
        body.grid_columnconfigure(1, weight=1, uniform="metrics")

        scale = THEME.font_scale
        size = -int(round(12 * scale))
        font = (THEME.font_family, size)
        box = tk.Text(body, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0, wrap="word",
                      font=font, height=20, width=10, cursor="arrow", padx=SPACE_S,
                      tabs=(int(190 * scale), "right", int(205 * scale), "left",
                            int(260 * scale), "left"),
                      spacing1=1, spacing3=1)
        # Wrapped "computed on" text stays in its column.
        box.tag_configure("row", lmargin2=int(260 * scale))
        box.tag_configure("extra", lmargin2=int(260 * scale), foreground=MUTED)
        box.tag_configure("head", font=(THEME.font_family, size, "bold"), foreground=MUTED)
        box.tag_configure("group", font=(THEME.font_family, size, "bold"), foreground=BLUE,
                          spacing1=SPACE_S)
        box.grid(row=0, column=0, sticky="new", padx=(0, SPACE_S))
        box.bind("<Configure>", lambda _e: self.plot_ctrl.fit_rhythm_table())
        self.txt_rhythm_metrics = box

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="new")
        bar = ctk.CTkFrame(right, fg_color="transparent")
        bar.pack(fill="x", pady=(0, SPACE_XS))
        self.btn_run_freq = ctk.CTkButton(
            bar, text="⟳  Recompute Frequency",
            command=self._run_freq,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_SEC, height=26, corner_radius=5,
        )
        self.btn_run_freq.pack(side="left")  # type: ignore[union-attr]
        self.lbl_freq_status = ctk.CTkLabel(
            right, text="  Click Analyze first", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left", wraplength=int(520 * scale))
        self.lbl_freq_status.pack(fill="x")  # type: ignore[union-attr]
        self._rhythm_plot_card(right, 360, "psd", 7.0).pack(fill="x")

    def _build_rhythm_nonlinear(self: ECGApp, page, row: int) -> None:
        hdr = self._rhythm_section(page, row, "Non-linear details")
        self.btn_run_nonlin = ctk.CTkButton(
            hdr, text="⟳  Recompute Non-linear",
            command=self._run_nonlinear,
            fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
            font=FONT_BTN_SEC, height=26, corner_radius=5,
        )
        self.btn_run_nonlin.pack(side="left", padx=(SPACE_M, 0))  # type: ignore[union-attr]
        # Runtime cap of the O(N^2) entropy block only (SampEn, ApEn, MSE ...):
        # Poincare and DFA always use every beat of the analysis window.
        ctk.CTkLabel(hdr, text="Entropy cap (RR):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(SPACE_M, 4))
        self.ent_entropy_cap = ctk.CTkEntry(hdr, width=64, height=26, font=FONT_LABEL,
                                            fg_color=BG, border_color=BORDER2,
                                            text_color=TEXT)
        self.ent_entropy_cap.insert(0, str(MouseECG.ENTROPY_MAX_BEATS))  # type: ignore[union-attr]
        self.ent_entropy_cap.pack(side="left")  # type: ignore[union-attr]
        self._bind_hover_tip(
            self.ent_entropy_cap,
            "Valid RR intervals used by the entropy block, for runtime (0 = no cap). "
            "SD1 / SD2 and DFA always use every valid RR of the window. "
            "Takes effect at the next Analyze or Recompute.")

        self.lbl_nonlin_status = ctk.CTkLabel(
            page, text="  Click Analyze first", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left", wraplength=int(1000 * THEME.font_scale))
        self.lbl_nonlin_status.grid(row=row + 1, column=0, sticky="ew", padx=SPACE_M)  # type: ignore[union-attr]
        self._rhythm_labels["nonlinear_basis"] = ctk.CTkLabel(
            page, text="", font=FONT_SMALL, text_color=TEXT, anchor="w", justify="left",
            wraplength=int(1000 * THEME.font_scale))
        self._rhythm_labels["nonlinear_basis"].grid(row=row + 2, column=0, sticky="ew", padx=SPACE_M)

        asym = ctk.CTkFrame(page, fg_color="transparent")
        asym.grid(row=row + 3, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_XS, 0))
        self._rhythm_labels["asymmetry"] = ctk.CTkLabel(
            asym, text="", font=FONT_LABEL, text_color=TEXT, anchor="w")
        self._rhythm_labels["asymmetry"].pack(fill="x")
        ctk.CTkLabel(asym, text=RR_ASYMMETRY_TEXT, font=FONT_HINT, text_color=MUTED, anchor="w",
                     justify="left", wraplength=int(1000 * THEME.font_scale)).pack(fill="x")
        self._rhythm_plot_card(asym, 240, "asymmetry", 10.0).pack(fill="x", pady=(SPACE_XS, 0))

    def _build_rhythm_distribution(self: ECGApp, page, row: int) -> None:
        hdr = self._rhythm_section(page, row, "RR distribution (valid intervals)")
        ctk.CTkButton(hdr, text="📋 Copy for Excel", height=22,
                      font=FONT_HINT, fg_color=BORDER, hover_color=BORDER2,
                      text_color=MUTED, corner_radius=5, width=140,
                      command=lambda: self._copy_tsv(self.txt_rr)
                      ).pack(side="left", padx=(SPACE_M, 0))
        body = ctk.CTkFrame(page, fg_color="transparent")
        body.grid(row=row + 1, column=0, sticky="ew", padx=SPACE_M)
        body.grid_columnconfigure(0, weight=1, uniform="dist")
        body.grid_columnconfigure(1, weight=2, uniform="dist")
        self.txt_rr = ctk.CTkTextbox(body, font=FONT_MONO, fg_color=PANEL,
                                     text_color=TEXT, border_width=0, height=240)
        self.txt_rr.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_S))
        self._rhythm_plot_card(body, 240, "rr_hist", 7.0).grid(row=0, column=1, sticky="nsew")

    def _build_rhythm_raw(self: ECGApp, page, row: int) -> None:
        """The NeuroKit tables as computed, one under the other at full width and
        without wrapping: a wrapped line splits a number. One Copy for Excel each.
        Their contents do not depend on being shown."""
        frame = ctk.CTkFrame(page, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_M))
        self._rhythm_raw = frame
        ctk.CTkLabel(frame, text="RAW NEUROKIT OUTPUT", font=FONT_SIDEBAR_HDR, text_color=MUTED,
                     anchor="w").grid(row=0, column=0, sticky="ew", pady=(SPACE_M, SPACE_XS))
        frame.grid_columnconfigure(0, weight=1)
        for i, (title, attr) in enumerate((("HRV time domain", "txt_td"),
                                           ("Frequency domain", "txt_fd"),
                                           ("Non-linear", "txt_nl"))):
            col = ctk.CTkFrame(frame, fg_color=PANEL, corner_radius=6)
            col.grid(row=i + 1, column=0, sticky="ew", pady=(0 if i == 0 else SPACE_S, 0))
            head = ctk.CTkFrame(col, fg_color="transparent")
            head.pack(fill="x", pady=(0, SPACE_XS))
            ctk.CTkLabel(head, text=title, font=FONT_SMALL, text_color=MUTED).pack(side="left")
            ctk.CTkButton(head, text="📋 Copy for Excel", height=20,
                          font=FONT_HINT, fg_color=BORDER, hover_color=BORDER2,
                          text_color=MUTED, corner_radius=5, width=120,
                          command=lambda a=attr: self._copy_tsv(getattr(self, a))
                          ).pack(side="right")
            box = ctk.CTkTextbox(col, font=FONT_MONO, fg_color=PANEL, text_color=TEXT,
                                 border_width=0, height=260, wrap="none")
            box.pack(fill="both", expand=True)
            setattr(self, attr, box)
        self._apply_rhythm_show_all()

    def _toggle_rhythm_show_all(self: ECGApp) -> None:
        self.ui.hrv_show_all = not self.ui.hrv_show_all
        self._apply_rhythm_show_all()
        self.plot_ctrl.refresh_rhythm_table()

    def _apply_rhythm_show_all(self: ECGApp) -> None:
        show = self.ui.hrv_show_all
        self.btn_rhythm_show_all.configure(
            text="Hide extra NeuroKit metrics" if show else "Show all NeuroKit metrics")
        if show:
            self._rhythm_raw.grid()
        else:
            self._rhythm_raw.grid_remove()
