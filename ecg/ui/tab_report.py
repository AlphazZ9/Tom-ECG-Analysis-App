# -*- coding: utf-8 -*-
"""
ecg.ui.tab_report
-----------------
ECGApp Report tab: provenance, what the report holds (module, state, key result,
Include), the exports and the text report. It reads everything from the export
controller, so the screen and the exports cannot disagree.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core.metric_table import REPORT_MODULES
from ecg.core.models import QT_REPORTED
from ecg.ui.freshness import NOT_COMPUTED
from ecg.ui.popover import Popover, menu_button
from ecg.ui.tab_keys import TAB_REPORT
from ecg.ui.tabbar import STATE_GLYPHS, state_color
from ecg.ui.theme import (
    PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, BLUE, BLUE_HOVER, ORANGE, SPACE_XS, SPACE_S,
    SPACE_M, FONT_SMALL, FONT_MONO, FONT_SIDEBAR_HDR, FONT_BTN_PRIMARY, FONT_BTN_SEC,
    FONT_CARD_TITLE,
)
from ecg.ui.theme import (
    BG, HAIRLINE, RADIUS, CARD_RADIUS, card_style, button_style, FONT_BODY, FONT_SECTION_HDR, FONT_LABEL,
    PAGE_PAD_X, PAGE_PAD_Y, PAGE_GAP, CARD_PAD,
)
from ecg.ui.theme import button_style

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")

_TEXT_OPEN = "▾  Text report"
_TEXT_CLOSED = "▸  Text report"
_PROVENANCE_LABELS = tuple(
    lb for lb in ("File", "Subject", "Analysis window", "Filters", "RR bounds", "Excluded RR",
                  "Artifact correction", "HRV bands", "QTc formula", "Autonomic")
    if QT_REPORTED or lb != "QTc formula")        # no QTc while QT is not reported
_NOTE = ("This screen, the text report, the PDF and the Metrics tables of Excel and Prism "
         "read one table: each value with its unit, its basis n and its state. The other "
         "Excel sheets and Prism tables hold the series those values come from.")


class TabReportMixin:
    """ECGApp Report tab construction and refresh."""

    def _build_tab_report(self: ECGApp) -> None:
        t = self.tabs.tab(TAB_REPORT)
        t.grid_rowconfigure(0, weight=1)
        t.grid_columnconfigure(0, weight=1)
        page = ctk.CTkScrollableFrame(t, fg_color=BG, scrollbar_button_color=BORDER,
                                      scrollbar_button_hover_color=BORDER2)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=2, uniform="report")
        page.grid_columnconfigure(1, weight=3, uniform="report")
        self._report_seen = None

        self._build_report_provenance(page)
        self._build_report_modules(page)

        # ── Text report, closed by default ───────────────────
        self.btn_report_text = ctk.CTkButton(
            page, text=_TEXT_CLOSED, command=self._toggle_report_text, anchor="w",
            fg_color="transparent", hover_color=BORDER, text_color=MUTED,
            font=FONT_SECTION_HDR, height=button_style()["height"], corner_radius=RADIUS)
        self.btn_report_text.grid(row=1, column=0, columnspan=2, sticky="w",
                                  padx=PAGE_PAD_X - SPACE_M, pady=(PAGE_GAP, 0))
        self.txt_report = ctk.CTkTextbox(page, font=FONT_MONO, fg_color=CARD, text_color=TEXT,
                                         border_width=1, border_color=BORDER,
                                         corner_radius=CARD_RADIUS, height=380)
        self.txt_report.grid(row=2, column=0, columnspan=2, sticky="ew",
                             padx=PAGE_PAD_X, pady=(SPACE_S, PAGE_PAD_Y))
        self._apply_report_text_open()
        self._refresh_report(force=True)

    def _report_card(self: ECGApp, page, column: int, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(page, **card_style())
        card.grid(row=0, column=column, sticky="nsew",
                  padx=(PAGE_PAD_X, PAGE_GAP // 2) if column == 0
                  else (PAGE_GAP // 2, PAGE_PAD_X), pady=(PAGE_PAD_Y, 0))
        ctk.CTkLabel(card, text=title.upper(), font=FONT_SECTION_HDR, text_color=MUTED,
                     anchor="w").pack(fill="x", padx=CARD_PAD, pady=(CARD_PAD - 4, SPACE_M))
        return card

    def _build_report_provenance(self: ECGApp, page) -> None:
        card = self._report_card(page, 0, "Provenance")
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=CARD_PAD, pady=(0, CARD_PAD))
        grid.grid_columnconfigure(1, weight=1)
        self._report_provenance: "dict[str, ctk.CTkLabel]" = {}
        for i, label in enumerate(_PROVENANCE_LABELS):
            ctk.CTkLabel(grid, text=label, font=FONT_LABEL, text_color=MUTED, anchor="nw",
                         height=22).grid(row=i, column=0, sticky="nw", padx=(0, SPACE_M))
            value = ctk.CTkLabel(grid, text="—", font=FONT_BODY, text_color=TEXT, anchor="nw",
                                 justify="left", height=22)
            value.grid(row=i, column=1, sticky="ew")
            self._report_provenance[label] = value
        grid.bind("<Configure>", lambda e: self._wrap_report_labels(
            self._report_provenance.values(), e.width, 130))

    def _build_report_modules(self: ECGApp, page) -> None:
        card = self._report_card(page, 1, "What is in this report")
        table = ctk.CTkFrame(card, fg_color="transparent")
        table.pack(fill="x", padx=CARD_PAD)
        table.grid_columnconfigure(2, weight=1)
        for col, head in enumerate(("Module", "State", "Key result", "Include")):
            ctk.CTkLabel(table, text=head, font=FONT_SMALL, text_color=MUTED, anchor="w",
                         height=20).grid(row=0, column=col, sticky="w", padx=(0, SPACE_M))
        self._report_state_vals: "dict[str, ctk.CTkLabel]" = {}
        self._report_result_vals: "dict[str, ctk.CTkLabel]" = {}
        self._report_include_ticks: "dict[str, ctk.CTkCheckBox]" = {}
        for i, m in enumerate(REPORT_MODULES):
            row = 2 * i + 2
            ctk.CTkFrame(table, height=1, fg_color=BORDER if i == 0 else HAIRLINE).grid(
                row=row - 1, column=0, columnspan=4, sticky="ew", pady=SPACE_XS)
            ctk.CTkLabel(table, text=m.label, font=FONT_CARD_TITLE, text_color=TEXT, anchor="nw",
                         height=20).grid(row=row, column=0, sticky="nw", padx=(0, SPACE_M))
            state = ctk.CTkLabel(table, text=NOT_COMPUTED, font=FONT_SMALL, text_color=MUTED,
                                 anchor="nw", height=20)
            state.grid(row=row, column=1, sticky="nw", padx=(0, SPACE_M))
            result = ctk.CTkLabel(table, text="—", font=FONT_SMALL, text_color=TEXT, anchor="nw",
                                  justify="left", height=20)
            result.grid(row=row, column=2, sticky="ew", padx=(0, SPACE_M))
            tick = ctk.CTkCheckBox(table, text="", width=24, height=20, checkbox_width=18,
                                   checkbox_height=18, fg_color=BLUE, hover_color=BLUE_HOVER,
                                   command=lambda k=m.key: self._on_report_include(k))
            tick.grid(row=row, column=3, sticky="n")
            self._report_state_vals[m.key] = state
            self._report_result_vals[m.key] = result
            self._report_include_ticks[m.key] = tick
        table.bind("<Configure>", lambda e: self._wrap_report_labels(
            self._report_result_vals.values(), e.width, 330))

        bar = ctk.CTkFrame(card, fg_color="transparent")
        bar.pack(fill="x", padx=CARD_PAD, pady=(CARD_PAD, SPACE_S))
        ctk.CTkButton(
            bar,
            text="Export Excel",
            command=self._export_excel,
            **button_style("primary"),
        ).pack(side="left", padx=(0, SPACE_M))
        for text, command in (("GraphPad Prism", self._export_prism),
                              ("PDF report", self._export_pdf_report),
                              ("Copy text", self._copy_report),
                              ("Save .txt", self._save_report_txt)):
            ctk.CTkButton(
                bar,
                text=text,
                command=command,
                width=0,
                **button_style("secondary"),
            ).pack(side="left", padx=(0, SPACE_M))
        # The optional exports (the right panel's EXPORTS held them): one
        # click further, the three above being the ones a report needs.
        self.btn_report_more = ctk.CTkButton(
            bar,
            text="More ▾",
            command=self._toggle_report_more,
            width=80,
            **button_style("secondary"),
        )
        self.btn_report_more.pack(side="left")
        self._bind_hover_tip(self.btn_report_more,
                             "RR intervals CSV, figures (PNG), everything (ZIP), events PDF")
        self.report_more = Popover(self, title="More exports")
        for text, command in (
                ("📄  RR intervals CSV  (Ctrl+Shift+E)", self._export_rr_csv),
                ("🖼  Figures (PNG)", self._export_figures),
                ("📦  Everything (ZIP: Excel + figures)", self._export_zip),
                ("🔬  Events PDF", self._export_arrhythmia_pdf)):
            menu_button(self.report_more.body, text, self._menu_command(self.report_more, command))
        self.lbl_report_left_out = ctk.CTkLabel(card, text="", font=FONT_SMALL, text_color=ORANGE,
                                                anchor="w", justify="left", height=0)
        self.lbl_report_left_out.pack(fill="x", padx=SPACE_M)
        note = ctk.CTkLabel(card, text=_NOTE, font=FONT_SMALL, text_color=MUTED, anchor="w",
                            justify="left")
        note.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_M))
        card.bind("<Configure>", lambda e: self._wrap_report_labels(
            (note, self.lbl_report_left_out), e.width, 40))

    @staticmethod
    def _wrap_report_labels(labels, width: int, room: int) -> None:
        """Long values wrap in the room their column has. CTk pixels are scaled
        ones: its wraplength takes unscaled ones like the event's width here."""
        for lbl in labels:
            scale = lbl._get_widget_scaling() if hasattr(lbl, "_get_widget_scaling") else 1.0
            wrap = max(120, int(width / scale) - room)
            if lbl.cget("wraplength") != wrap:
                lbl.configure(wraplength=wrap)

    # ── Handlers ─────────────────────────────────────────────

    def _toggle_report_more(self: ECGApp) -> None:
        self.report_more.toggle(self.btn_report_more)

    def _on_report_include(self: ECGApp, key: str) -> None:
        """A tick is kept with the state it was given in (see include_choices)."""
        tick = self._report_include_ticks[key]
        state = self.export_ctrl.report_line_states()[key]
        self.ui.report_include[key] = [bool(tick.get()), state]
        self._refresh_report(force=True)

    def _toggle_report_text(self: ECGApp) -> None:
        self.ui.report_text_open = not self.ui.report_text_open
        self._apply_report_text_open()

    def _apply_report_text_open(self: ECGApp) -> None:
        if self.ui.report_text_open:
            self.txt_report.grid()
        else:
            self.txt_report.grid_remove()
        self.btn_report_text.configure(text=_TEXT_OPEN if self.ui.report_text_open else _TEXT_CLOSED)

    # ── Refresh ──────────────────────────────────────────────

    def _refresh_report(self: ECGApp, force: bool = False) -> None:
        """Bring the tab up to date when it is the one on screen (*force*: in
        any case). Costs a table build only when something behind it changed:
        it runs on the shell's tick."""
        if getattr(self, "_report_result_vals", None) is None:
            return
        if not force and self.tabs.get() != TAB_REPORT:
            return
        ctrl = self.export_ctrl
        seen = ctrl.report_signature()
        if seen == self._report_seen:
            return
        self._report_seen = seen
        view = ctrl.report_view()
        for label, value in view.provenance:
            self._report_provenance[label].configure(text=value)
        for line in view.lines:
            self._report_state_vals[line.key].configure(
                text=f"{STATE_GLYPHS[line.state]} {line.state}", text_color=state_color(line.state))
            self._report_result_vals[line.key].configure(text=line.result)
            tick = self._report_include_ticks[line.key]
            tick.configure(state="normal" if line.can_include else "disabled")
            if line.include:
                tick.select()
            else:
                tick.deselect()
        self.lbl_report_left_out.configure(
            text=f"Left out of the exports: {', '.join(view.left_out)}" if view.left_out else "")
        self._set_textbox(self.txt_report, view.text)
