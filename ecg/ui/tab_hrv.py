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
from ecg.ui.theme import (
    BG, CARD, card_style, FONT_SECTION_HDR, FONT_KPI_HERO, PAGE_PAD_X, PAGE_PAD_Y, PAGE_GAP, CARD_PAD,
)
from ecg.ui.theme import input_style
from ecg.ui.theme import button_style, toggle_style
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

    def _rhythm_section(self: ECGApp, parent, row: int, title: str
                        ) -> "tuple[ctk.CTkFrame, ctk.CTkFrame]":
        """A card in grid row *row* of *parent*, titled *title* (the mockup's
        card: white, 1 px border, 16 px inside). Returns its header frame, for
        what sits next to the title, and its body, a one-column grid whose
        row 0 is the header."""
        card = ctk.CTkFrame(parent, **card_style())
        card.grid(row=row, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(0, PAGE_GAP))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=CARD_PAD, pady=(CARD_PAD - 4, CARD_PAD))
        body.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(body, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, SPACE_M))
        ctk.CTkLabel(hdr, text=title.upper(), font=FONT_SECTION_HDR,
                     text_color=MUTED).pack(side="left")
        return hdr, body

    def _rhythm_plot_card(self: ECGApp, parent, height: int, slot: str, width_in: float,
                          placeholder: str = "Click Analyze to display") -> ctk.CTkFrame:
        """A fixed-height card holding the figure of *slot*: inside a scrolling
        page a figure has no available height to fit."""
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=0, height=height)
        card.pack_propagate(False)
        self._slots[slot] = CanvasSlot(card, width_in, height / 100, toolbar=False,
                                       placeholder=placeholder)
        return card

    def _build_hrv_view_overview(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Rhythm & HRV: KPI cards, tachogram, Poincaré and multiscale entropy, metric table next
        to the spectrum, non-linear details, RR distribution, and the raw
        NeuroKit tables behind Show all. Numbers come from the shared table
        (PlotController.plot_rhythm_view)."""
        page = ctk.CTkScrollableFrame(parent, fg_color=BG,
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
        bar.grid(row=row, column=0, sticky="ew", padx=PAGE_PAD_X, pady=(PAGE_PAD_Y, PAGE_GAP))
        self._rhythm_cards: "dict[str, dict]" = {}
        for i, (key, name) in enumerate((("hr", "Heart rate"), ("sdnn", "SDNN"), ("rmssd", "RMSSD"),
                                         ("pnn6", "pNN6"), ("lf_hf", "LF / HF"),
                                         ("sd1_sd2", "SD1 / SD2"))):
            bar.grid_columnconfigure(i, weight=1, uniform="kpi")
            # The mockup's KPI tile: 11 px semibold name, 24 px bold value
            # with its 12 px unit, 11 px note; 12 px apart.
            card = ctk.CTkFrame(bar, **card_style())
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 12, 0))
            ctk.CTkLabel(card, text=name, font=THEME.font(11, "semibold"), text_color=MUTED,
                         anchor="w").pack(fill="x", padx=14, pady=(10, 0))
            line = ctk.CTkFrame(card, fg_color="transparent")
            line.pack(fill="x", padx=14)
            value = ctk.CTkLabel(line, text="—", font=FONT_KPI_HERO, text_color=TEXT, width=0)
            value.pack(side="left")
            unit = ctk.CTkLabel(line, text="", font=FONT_LABEL, text_color=MUTED, width=0)
            unit.pack(side="left", padx=(SPACE_S, 0), pady=(SPACE_S, 0))
            note = ctk.CTkLabel(card, text="", font=FONT_SMALL, text_color=MUTED, anchor="w",
                                justify="left", wraplength=int(150 * THEME.font_scale))
            note.pack(fill="x", padx=14, pady=(0, 10))
            self._rhythm_cards[key] = {"value": value, "unit": unit, "note": note}

    def _build_rhythm_plots(self: ECGApp, page, row: int) -> None:
        hdr, sec = self._rhythm_section(page, row, "Tachogram")
        self._rhythm_labels["excluded"] = ctk.CTkLabel(
            hdr, text="", font=FONT_SMALL, text_color=MUTED)
        self._rhythm_labels["excluded"].pack(side="left", padx=(SPACE_M, 0))

        _spk_on = self.ui.show_spike_markers
        self.btn_toggle_spikes = ctk.CTkButton(
            hdr,
            text="▲▼ Spike markers",
            width=140,
            command=self._toggle_spike_markers,
            **{**button_style("secondary"), **toggle_style(_spk_on)},
        )
        self.btn_toggle_spikes.pack(side="right")  # type: ignore[union-attr]
        # "Review" jumps to the next flagged beat (▲/▼) of this tachogram, in
        # Detection, centred in the signal view: the same step_spike() as the
        # Detection tab's "Spike: ◀ ▶" stepper, reachable from here. Pressing
        # it again walks every flagged beat in turn.
        self.btn_review_spikes = ctk.CTkButton(
            hdr,
            text="🔍 Review",
            width=100,
            command=lambda: self._step_spike(+1),
            **button_style("secondary"),
        )
        self.btn_review_spikes.pack(side="right", padx=(0, SPACE_S))  # type: ignore[union-attr]
        self._bind_hover_tip(
            self.btn_review_spikes,
            "Jump to the next flagged beat (▲/▼) — press again to step "
            "through the rest")

        # The tachogram has the full width; the Poincaré diagram and the
        # multiscale entropy have their own card below (row + 1).
        left = ctk.CTkFrame(sec, fg_color="transparent")
        left.grid(row=1, column=0, sticky="ew")
        self._rhythm_plot_card(left, 440, "rr", 12.0).pack(fill="x")
        self._slots["rr"].enable_wheel_yzoom()
        ctk.CTkLabel(
            left, text=("Left-click: go to time  ·  right-click: next spike  ·  wheel: zoom Y  ·  "
                        "middle-click: reset Y  ·  ▲ sudden "
                        "acceleration  ▼ sudden deceleration  ·  amber >2.5 SD  ·  red >3.5 SD"),
            font=FONT_HINT, text_color=MUTED, anchor="w",
        ).pack(fill="x", pady=(SPACE_XS, 0))

        _hdr, sec2 = self._rhythm_section(page, row + 1, "Poincaré and multiscale entropy")
        right = ctk.CTkFrame(sec2, fg_color="transparent")
        right.grid(row=1, column=0, sticky="ew")
        self._rhythm_plot_card(right, 420, "poincare", 12.0).pack(fill="x")
        self._slots["poincare"].enable_wheel_yzoom()
        ctk.CTkLabel(
            right, text=("Successive valid pairs only: no pair crosses an excluded interval  ·  "
                         "wheel: zoom  ·  middle-click: reset"),
            font=FONT_HINT, text_color=MUTED, anchor="w",
        ).pack(fill="x", pady=(SPACE_XS, 0))

    def _build_rhythm_metrics(self: ECGApp, page, row: int) -> None:
        hdr, sec = self._rhythm_section(page, row, "Metrics and power spectrum")
        self.btn_rhythm_show_all = ctk.CTkButton(
            hdr,
            text="",
            width=200,
            command=self._toggle_rhythm_show_all,
            **button_style("secondary"),
        )
        self.btn_rhythm_show_all.pack(side="left", padx=(SPACE_M, 0))

        body = ctk.CTkFrame(sec, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew")
        body.grid_columnconfigure(0, weight=1, uniform="metrics")
        body.grid_columnconfigure(1, weight=1, uniform="metrics")

        scale = THEME.font_scale
        size = -int(round(12 * scale))
        font = (THEME.font_family, size)
        box = tk.Text(body, bg=CARD, fg=TEXT, bd=0, highlightthickness=0, wrap="word",
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
        box.grid(row=0, column=0, sticky="new", padx=(0, CARD_PAD))
        box.bind("<Configure>", lambda _e: self.plot_ctrl.fit_rhythm_table())
        self.txt_rhythm_metrics = box

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="new")
        bar = ctk.CTkFrame(right, fg_color="transparent")
        bar.pack(fill="x", pady=(0, SPACE_S))
        self.btn_run_freq = ctk.CTkButton(
            bar,
            text="⟳  Recompute Frequency",
            command=self._run_freq,
            **button_style("primary"),
        )
        self.btn_run_freq.pack(side="left")  # type: ignore[union-attr]
        self.lbl_freq_status = ctk.CTkLabel(
            right, text="  Click Analyze first", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left", wraplength=int(520 * scale))
        self.lbl_freq_status.pack(fill="x")  # type: ignore[union-attr]
        self._rhythm_plot_card(right, 360, "psd", 7.0).pack(fill="x")

    def _build_rhythm_nonlinear(self: ECGApp, page, row: int) -> None:
        hdr, sec = self._rhythm_section(page, row, "Non-linear details")
        self.btn_run_nonlin = ctk.CTkButton(
            hdr,
            text="⟳  Recompute Non-linear",
            command=self._run_nonlinear,
            **button_style("primary"),
        )
        self.btn_run_nonlin.pack(side="left", padx=(SPACE_M, 0))  # type: ignore[union-attr]
        # Runtime cap of the O(N^2) entropy block only (SampEn, ApEn, MSE ...):
        # Poincare and DFA always use every beat of the analysis window.
        ctk.CTkLabel(hdr, text="Entropy cap (RR):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(SPACE_M, 4))
        self.ent_entropy_cap = ctk.CTkEntry(hdr, width=64, **input_style())
        self.ent_entropy_cap.insert(0, str(MouseECG.ENTROPY_MAX_BEATS))  # type: ignore[union-attr]
        self.ent_entropy_cap.pack(side="left")  # type: ignore[union-attr]
        self._bind_hover_tip(
            self.ent_entropy_cap,
            "Valid RR intervals used by the entropy block, for runtime (0 = no cap). "
            "SD1 / SD2 and DFA always use every valid RR of the window. "
            "Takes effect at the next Analyze or Recompute.")

        self.lbl_nonlin_status = ctk.CTkLabel(
            sec, text="  Click Analyze first", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left", wraplength=int(1000 * THEME.font_scale))
        self.lbl_nonlin_status.grid(row=1, column=0, sticky="ew")  # type: ignore[union-attr]
        self._rhythm_labels["nonlinear_basis"] = ctk.CTkLabel(
            sec, text="", font=FONT_SMALL, text_color=TEXT, anchor="w", justify="left",
            wraplength=int(1000 * THEME.font_scale))
        self._rhythm_labels["nonlinear_basis"].grid(row=2, column=0, sticky="ew")

        asym = ctk.CTkFrame(sec, fg_color="transparent")
        asym.grid(row=3, column=0, sticky="ew", pady=(SPACE_S, 0))
        self._rhythm_labels["asymmetry"] = ctk.CTkLabel(
            asym, text="", font=FONT_LABEL, text_color=TEXT, anchor="w")
        self._rhythm_labels["asymmetry"].pack(fill="x")
        ctk.CTkLabel(asym, text=RR_ASYMMETRY_TEXT, font=FONT_HINT, text_color=MUTED, anchor="w",
                     justify="left", wraplength=int(1000 * THEME.font_scale)).pack(fill="x")
        self._rhythm_plot_card(asym, 240, "asymmetry", 10.0).pack(fill="x", pady=(SPACE_XS, 0))

    def _build_rhythm_distribution(self: ECGApp, page, row: int) -> None:
        hdr, sec = self._rhythm_section(page, row, "RR distribution (valid intervals)")
        ctk.CTkButton(
            hdr,
            text="📋 Copy for Excel",
            width=140,
            command=lambda: self._copy_tsv(self.txt_rr),
            **button_style("secondary"),
        ).pack(side="left", padx=(SPACE_M, 0))
        body = ctk.CTkFrame(sec, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew")
        body.grid_columnconfigure(0, weight=1, uniform="dist")
        body.grid_columnconfigure(1, weight=2, uniform="dist")
        self.txt_rr = ctk.CTkTextbox(body, font=FONT_MONO, fg_color=CARD,
                                     text_color=TEXT, border_width=0, height=240)
        self.txt_rr.grid(row=0, column=0, sticky="nsew", padx=(0, CARD_PAD))
        self._rhythm_plot_card(body, 240, "rr_hist", 7.0).grid(row=0, column=1, sticky="nsew")

    def _build_rhythm_raw(self: ECGApp, page, row: int) -> None:
        """The NeuroKit tables as computed, one under the other at full width and
        without wrapping: a wrapped line splits a number. One Copy for Excel each.
        Their contents do not depend on being shown."""
        _hdr, frame = self._rhythm_section(page, row, "Raw NeuroKit output")
        self._rhythm_raw = frame.master
        for i, (title, attr) in enumerate((("HRV time domain", "txt_td"),
                                           ("Frequency domain", "txt_fd"),
                                           ("Non-linear", "txt_nl"))):
            col = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=0)
            col.grid(row=i + 1, column=0, sticky="ew", pady=(0 if i == 0 else SPACE_M, 0))
            head = ctk.CTkFrame(col, fg_color="transparent")
            head.pack(fill="x", pady=(0, SPACE_XS))
            ctk.CTkLabel(head, text=title, font=FONT_SMALL, text_color=MUTED).pack(side="left")
            ctk.CTkButton(
                head,
                text="📋 Copy for Excel",
                width=120,
                command=lambda a=attr: self._copy_tsv(getattr(self, a)),
                **button_style("secondary"),
            ).pack(side="right")
            box = ctk.CTkTextbox(col, font=FONT_MONO, fg_color=BG, text_color=TEXT,
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
