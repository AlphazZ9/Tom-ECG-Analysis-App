# -*- coding: utf-8 -*-
"""
ecg.ui.compare_segments
-----------------------
ECGApp Compare segments tab: 1 to 6 windows of the recording, typed or
placed by a saved template around an event, their metric table, quality
checks, RR tachograms or Poincaré plots, the export and the group table. Built in the tab when it opens, rebuilt when the recording or its
peaks have changed since. Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ecg.core.models import MouseECG
from ecg.ui.theme import (
    BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT, PLOT, RED, BLUE,
    GREEN, ORANGE, PURPLE, TEAL, TEAL_DARK, GREEN_DARK, ORANGE_DARK, SPACE_XS, SPACE_S,
    SPACE_M, SPACE_L, FONT_LABEL, FONT_SMALL, FONT_KPI_VALUE, FONT_KPI_LABEL,
    FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_MICRO, FONT_BADGE,
    FONT_SUBSECTION,
)
from ecg.ui.theme import combo_style, input_style
from ecg.ui.theme import (
    THEME, CARD_PAD, FONT_HINT, FONT_SECTION_HDR, PAGE_GAP, PAGE_PAD_X, PAGE_PAD_Y,
    card_style,
)
import tkinter as tk
from ecg.ui.theme import button_style, disabled_style, toggle_style
from ecg.ui.poincare_figure import draw_poincare_set
from ecg.io.group_table import append_group_row, group_row
from ecg.io.group_stats import (
    DEFAULT_METRICS_NOTE, check_rows, paired_stats, read_group_table, segment_tags,
    smallest_wilcoxon_p,
    write_group_report,
)
from ecg.ui.group_figure import draw_group_figure
from ecg.io.segment_export import (
    ComparisonResult, comparison_tsv, write_csv, write_pzfx, write_xlsx,
)
from ecg.core.segment_plan import (
    DEFAULT_ARTEFACT_MARGIN_S, MAX_SEGMENTS, SegmentTemplate, delete_template,
    load_templates, quality_flags, resolve, save_template, segment_quality,
    template_from_windows, unique_tags,
)
from ecg.ui.plots import CanvasSlot, style_axes
from ecg.ui.tab_keys import TAB_COMPARE

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class CompareSegmentsMixin:
    """ECGApp Compare Segments tab."""

    # ════════════════════════════════════════════════════════
    #  COMPARE SEGMENTS
    # ════════════════════════════════════════════════════════

    # Rows of the Compare Segments table and export: (key in the
    # compute_segment_stats / compare_segments dict, label, unit). Mean HR
    # is 60000 / mean NN; the block rows are the non-pseudo-replicated
    # summary the descriptive comparison is based on.
    _SEGMENT_TABLE_METRICS: "list[tuple[str, str, str]]" = [
        ("n_beats",       "Beats",                      ""),
        ("duration_s",    "Duration",                   "s"),
        ("n_blocks",      "Blocks used (n)",            ""),
        ("hr_block_mean", "HR, mean of block means",    "bpm"),
        ("hr_block_sd",   "HR, SD of block means",      "bpm"),
        ("hr_mean",       "Mean HR (60000/mean NN)",    "bpm"),
        ("hr_sd",         "Instantaneous HR SD",        "bpm"),
        ("hr_min",        "Min inst. HR (2nd %ile)",    "bpm"),
        ("hr_max",        "Max inst. HR (98th %ile)",   "bpm"),
        ("rr_mean",       "Mean RR",                    "ms"),
        ("rr_sd",         "RR SD",                      "ms"),
        ("rr_cv",         "RR CV",                      "%"),
        ("sdnn",          "SDNN",                       "ms"),
        ("rmssd",         "RMSSD",                      "ms"),
        ("pnn6",          "pNN6",                       "%"),
        ("lf",            "LF power",                   "ms²"),
        ("hf",            "HF power",                   "ms²"),
        ("lf_nu",         "LF (n.u.)",                  "%"),
        ("hf_nu",         "HF (n.u.)",                  "%"),
        ("lf_hf",         "LF/HF",                      ""),
        ("sd1",           "Poincaré SD1",               "ms"),
        ("sd2",           "Poincaré SD2",               "ms"),
        ("sd_ratio",      "SD1/SD2",                    ""),
    ]

    def _open_compare_segments(self: ECGApp) -> None:
        """Show the Compare Segments tab (Ctrl+M, Time course > Autonomic)."""
        self.tabs.set(TAB_COMPARE)
        self._refresh_compare_tab()

    def _build_tab_compare(self: ECGApp) -> None:
        page = self.tabs.tab(TAB_COMPARE)
        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)
        self._compare_holder = ctk.CTkScrollableFrame(
            page, fg_color=BG, corner_radius=0, scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER2)
        self._compare_holder.grid(row=0, column=0, sticky="nsew")
        self._compare_built_for = None
        self._refresh_compare_tab()

    def _compare_signature(self: ECGApp) -> "Optional[tuple]":
        """What the tab was built from: the recording, its peaks and the
        events it offers. None without peaks."""
        if self._signal_flt is None or self._rpeaks_ok is None or self._fs is None:
            return None
        rp = self._rpeaks_ok
        return (id(self._signal_flt), len(self._signal_flt), float(self._fs),
                len(rp), int(np.sum(rp)) if len(rp) else 0,
                tuple((a.get("label"), a.get("t_start"), a.get("t_end"))
                      for a in self.analysis.annotations),
                len(self.analysis.arrhythmia_events))

    def _current_comparison(self: ECGApp) -> "Optional[ComparisonResult]":
        """The last Compare segments result, None when there is none or the
        recording, its peaks or its events changed since (the main exports
        never write a comparison of other peaks)."""
        res = getattr(self.analysis, "last_comparison", None)
        if res is None:
            return None
        basis = getattr(self.analysis, "last_comparison_basis", None)
        return res if basis is not None and basis == self._compare_signature() else None

    def _refresh_compare_tab(self: ECGApp) -> None:
        """Build the tab's content for the current recording, unless it is
        already built for it (the windows typed in then stay)."""
        holder = getattr(self, "_compare_holder", None)
        if holder is None:
            return
        sig = self._compare_signature()
        if sig is not None and sig == self._compare_built_for:
            return
        for child in holder.winfo_children():
            child.destroy()
        self._compare_built_for = sig
        if sig is None:
            ctk.CTkLabel(holder, text="Open a file and detect the peaks to compare segments "
                                      "(the group card below works without one).",
                         font=FONT_LABEL, text_color=MUTED).pack(pady=(60, 40))
        else:
            self._build_compare_segments(holder)
        self._build_group_card(holder)

    def _build_compare_segments(self: ECGApp, win) -> None:
        """The comparison inside *win* (the tab's frame): 1 to MAX_SEGMENTS
        windows, typed or placed by a template around an event, their metric
        table (Δ % against the first), RR tachograms or Poincaré plots, the
        quality of each, the export and the group table. The stats run on a
        background thread."""
        sig    = self._signal_flt
        rp     = self._rpeaks_ok.copy()
        fs     = self._fs
        dur    = float(len(sig)) / fs
        vlf, lf, hf = self._get_freq_bands()
        sub    = self.ent_subject.get().strip() if self.ent_subject else "subject"
        tip    = self._bind_hover_tip
        built_for = self._compare_built_for     # the state rp / sig come from

        def _subject() -> str:
            """The Subject ID as it is now (it may be typed after the tab was built)."""
            return self.ent_subject.get().strip() if self.ent_subject else ""

        top = ctk.CTkFrame(win, fg_color="transparent")
        top.pack(fill="x", padx=PAGE_PAD_X, pady=(PAGE_PAD_Y, PAGE_GAP))
        top.grid_columnconfigure((0, 1), weight=1, uniform="setup")

        def _section(title: str, note: str = "", column: "Optional[int]" = None
                     ) -> "tuple[ctk.CTkFrame, ctk.CTkFrame]":
            """A white card titled *title* (the pages' card: 1 px border,
            16 px inside), in column *column* of the top row, or across the
            page when None; returns its header, for what sits by the title,
            and its body."""
            if column is None:
                card = ctk.CTkFrame(win, **card_style())
                card.pack(fill="x", padx=PAGE_PAD_X, pady=(0, PAGE_GAP))
            else:
                card = ctk.CTkFrame(top, **card_style())
                card.grid(row=0, column=column, sticky="nsew",
                          padx=(0, PAGE_GAP // 2) if column == 0 else (PAGE_GAP // 2, 0))
            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(fill="both", expand=True, padx=CARD_PAD, pady=(CARD_PAD - 4, CARD_PAD))
            hdr_ = ctk.CTkFrame(body, fg_color="transparent")
            hdr_.pack(fill="x", pady=(0, SPACE_M))
            ctk.CTkLabel(hdr_, text=title.upper(), font=FONT_SECTION_HDR,
                         text_color=MUTED).pack(side="left")
            if note:
                ctk.CTkLabel(hdr_, text=note, font=FONT_SMALL,
                             text_color=MUTED).pack(side="left", padx=(SPACE_M, 0))
            return hdr_, body

        # ── Event and template ────────────────────────────────────────────
        _h, tcard = _section("Windows around an event", "the same template for every animal",
                             column=0)
        qrow = ctk.CTkFrame(tcard, fg_color="transparent")
        qrow.pack(fill="x", pady=(0, SPACE_S))

        qrow2 = ctk.CTkFrame(tcard, fg_color="transparent")
        qrow2.pack(fill="x", pady=(0, SPACE_S))
        _event_opts: "dict[str, tuple[float, float]]" = {}
        for _ann in self.analysis.annotations:
            _lbl = _ann.get("label") or "Annotation"
            _event_opts[f"📍 {_lbl}  [{_ann['t_start']:.0f}–{_ann['t_end']:.0f}s]"] = (
                float(_ann["t_start"]), float(_ann["t_end"]))
        for _ev in self.analysis.arrhythmia_events:
            _event_opts[f"⚠ {_ev.label}  [{_ev.t_start:.0f}–{_ev.t_end:.0f}s]"] = (
                float(_ev.t_start), float(_ev.t_end))

        def _on_event_pick(choice: str) -> None:
            times = _event_opts.get(choice)
            if times is None:
                return
            for e, v in ((q_evt_start, times[0]), (q_evt_end, times[1])):
                e.delete(0, "end"); e.insert(0, f"{v:.1f}")

        if _event_opts:
            ctk.CTkLabel(qrow, text="Event:", font=FONT_SMALL, text_color=MUTED).pack(side="left")
            cb_event = ctk.CTkComboBox(qrow, width=210, values=list(_event_opts.keys()),
                                       command=_on_event_pick, **combo_style())
            cb_event.set("Pick…")
            cb_event.pack(side="left", padx=(4, SPACE_M))

        ctk.CTkLabel(qrow, text="Start (s):", font=FONT_SMALL, text_color=MUTED).pack(side="left")
        q_evt_start = ctk.CTkEntry(qrow, width=70, **input_style())
        if self.ui.nav_pos > 0:
            # Where the Detection view is: most likely near the event.
            q_evt_start.insert(0, str(int(self.ui.nav_pos)))
        q_evt_start.pack(side="left", padx=(4, SPACE_M))
        ctk.CTkLabel(qrow, text="End (s):", font=FONT_SMALL, text_color=MUTED).pack(side="left")
        q_evt_end = ctk.CTkEntry(qrow, width=70, placeholder_text="= start", **input_style())
        q_evt_end.pack(side="left", padx=(4, SPACE_M))
        ctk.CTkLabel(qrow2, text="Artefact margin (s):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        ent_margin = ctk.CTkEntry(qrow2, width=48, **input_style())
        ent_margin.insert(0, f"{DEFAULT_ARTEFACT_MARGIN_S:g}")
        ent_margin.pack(side="left", padx=(4, SPACE_M))
        tip(ent_margin, "Quality check only (it does not move the windows): a window overlapping "
                        "the event or this "
                        "many seconds after it is flagged (shock / movement artefact).")

        trow = ctk.CTkFrame(tcard, fg_color="transparent")
        trow.pack(fill="x", pady=(0, SPACE_XS))
        ctk.CTkLabel(trow, text="Template:", font=FONT_SMALL, text_color=MUTED).pack(side="left")
        _templates: "dict[str, SegmentTemplate]" = {}
        cb_tpl = ctk.CTkComboBox(trow, width=300, values=[""], state="readonly", **combo_style())
        cb_tpl.pack(side="left", padx=(4, SPACE_S))
        ctk.CTkLabel(trow, text="Window (s):", font=FONT_SMALL, text_color=MUTED).pack(side="left")
        ent_win = ctk.CTkEntry(trow, width=56, placeholder_text="tpl", **input_style())
        ent_win.pack(side="left", padx=(4, SPACE_S))
        tip(ent_win, "Length of every window, in seconds (e.g. 60 for 1 min). Empty: the "
                     "template's own. A window before the event keeps its end, one after it "
                     "keeps its start (so the gap left after the event stays).")

        def _reload_templates(select: "Optional[str]" = None) -> None:
            _templates.clear()
            for t in load_templates():
                _templates[t.name + ("" if t.builtin else "  (mine)")] = t
            names = list(_templates)
            cb_tpl.configure(values=names)
            pick = next((k for k, t in _templates.items() if t.name == select), None)
            cb_tpl.set(pick or (cb_tpl.get() if cb_tpl.get() in _templates else names[0]))

        lbl_tpl_status = ctk.CTkLabel(tcard, text="", font=FONT_MICRO, text_color=MUTED,
                                      anchor="w", justify="left")

        def _event_times() -> "Optional[tuple[float, float]]":
            """(start, end) of the event typed, None when none is."""
            if not q_evt_start.get().strip():
                return None
            t0 = float(q_evt_start.get())
            t1 = float(q_evt_end.get()) if q_evt_end.get().strip() else t0
            if t1 < t0:
                raise ValueError("the event's end is before its start")
            return t0, t1

        def _apply_template() -> None:
            t = _templates.get(cb_tpl.get())
            try:
                ev = _event_times()
            except ValueError as exc:
                lbl_tpl_status.configure(text=f"⚠  {exc}", text_color=RED); return
            if t is None or ev is None:
                lbl_tpl_status.configure(text="⚠  pick an event (or type its start) and a template",
                                         text_color=RED); return
            win_txt = ent_win.get().strip()
            try:
                win_s = float(win_txt) if win_txt else None
                if win_s is not None and not (np.isfinite(win_s) and win_s > 0):
                    raise ValueError
            except ValueError:
                lbl_tpl_status.configure(text="⚠  Window (s) must be a positive number of seconds",
                                         text_color=RED); return
            wins = resolve(t, ev[0], ev[1], dur, window_s=win_s)
            _set_rows([(w.label, w.lo, w.hi) for w in wins])
            _state["template"] = t.name
            clamped = [w.label for w in wins if w.clamped]
            lbl_tpl_status.configure(
                text=("⚠  cut at the recording's edge: " + ", ".join(clamped)) if clamped else
                     "✓  " + "  ·  ".join(f"{w.label} [{w.lo:.0f}–{w.hi:.0f} s]" for w in wins),
                text_color=ORANGE if clamped else GREEN)

        def _save_template() -> None:
            try:
                ev = _event_times()
                rows = _read_rows()
            except ValueError as exc:
                lbl_tpl_status.configure(text=f"⚠  {exc}", text_color=RED); return
            if ev is None:
                lbl_tpl_status.configure(
                    text="⚠  a template is placed from an event: type or pick its start first",
                    text_color=RED); return
            dlg = ctk.CTkInputDialog(text="Template name (e.g. Footshock 5 min):",
                                     title="Save the windows as a template")
            name = (dlg.get_input() or "").strip()
            if not name:
                return
            try:
                save_template(template_from_windows(name, rows, ev[0], ev[1]))
            except (OSError, ValueError) as exc:
                messagebox.showerror("Template", str(exc), parent=win); return
            _reload_templates(select=name)
            _state["template"] = name
            lbl_tpl_status.configure(text=f"✓  template '{name}' saved", text_color=GREEN)

        def _delete_template() -> None:
            t = _templates.get(cb_tpl.get())
            if t is None or t.builtin:
                lbl_tpl_status.configure(text="⚠  built-in templates cannot be deleted",
                                         text_color=ORANGE); return
            if not messagebox.askyesno("Template", f"Delete the template '{t.name}'?",
                                       parent=win):
                return
            try:
                delete_template(t.name)
            except OSError as exc:
                messagebox.showerror("Template", str(exc), parent=win); return
            _reload_templates()
            lbl_tpl_status.configure(text=f"✓  template '{t.name}' deleted", text_color=GREEN)

        for text, cmd, style, tiptext in (
                ("Apply", _apply_template, "primary",
                 "Place the template's windows around the event"),
                ("Save as template…", _save_template, "secondary",
                 "Save the windows below, relative to the event, under a name: "
                 "apply it to the next animals"),
                ("Delete", _delete_template, "secondary", "Delete the selected template (yours only)")):
            b = ctk.CTkButton(trow, text=text, width=0, command=cmd, **button_style(style))
            b.pack(side="left", padx=(0, SPACE_S))
            tip(b, tiptext)
        lbl_tpl_status.pack(anchor="w", fill="x")

        # ── Segments: 1 to MAX_SEGMENTS rows ──────────────────────────────
        _h, seg_card = _section("Segments", f"the first is the reference  ·  {sub}  ·  "
                                            f"{dur:.1f} s  ·  {len(rp)} peaks", column=1)
        rows_frame = ctk.CTkFrame(seg_card, fg_color="transparent")
        rows_frame.pack(fill="x")
        seg_rows: "list[dict]" = []
        _state: "dict[str, Any]" = {"template": "", "segs": None, "cmps": None, "flags": None}

        def _color(i: int) -> str:
            # Segment colours in order, the first the reference; read at call
            # time so a theme change re-synced into this module applies.
            seq = (BLUE, ORANGE_DARK, GREEN, PURPLE, TEAL, RED)
            return seq[i % len(seq)]

        def _read_rows() -> "list[tuple[str, float, float]]":
            out = []
            for i, r in enumerate(seg_rows):
                try:
                    lo, hi = float(r["lo"].get()), float(r["hi"].get())
                except ValueError:
                    raise ValueError(f"segment {i + 1}: invalid time") from None
                out.append((r["lbl"].get().strip() or f"Segment {chr(65 + i)}", lo, hi))
            return out

        def _set_rows(values: "list[tuple[str, float, float]]") -> None:
            for r in seg_rows:
                r["frame"].destroy()
            seg_rows.clear()
            for label, lo, hi in values[:MAX_SEGMENTS]:
                _add_row(label, lo, hi)
            _refresh_add_btn()

        def _remove_row(r: dict) -> None:
            if len(seg_rows) <= 1:
                return
            vals = [(x["lbl"].get(), x["lo"].get(), x["hi"].get()) for x in seg_rows if x is not r]
            _set_rows([(lbl, float(lo or 0), float(hi or 0)) for lbl, lo, hi in vals])

        def _add_row(label: str, lo: float, hi: float) -> None:
            i = len(seg_rows)
            col = _color(i)
            f = ctk.CTkFrame(rows_frame, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text="■", text_color=col, font=FONT_LABEL, width=16).pack(side="left")
            lbl_e = ctk.CTkEntry(f, width=130, placeholder_text="Label", **input_style())
            lbl_e.insert(0, label)
            lbl_e.pack(side="left", padx=(SPACE_XS, SPACE_M))
            ctk.CTkLabel(f, text="From (s):", font=FONT_SMALL, text_color=MUTED).pack(side="left")
            lo_e = ctk.CTkEntry(f, width=80, **input_style())
            lo_e.insert(0, f"{lo:g}")
            lo_e.pack(side="left", padx=(4, SPACE_M))
            ctk.CTkLabel(f, text="To (s):", font=FONT_SMALL, text_color=MUTED).pack(side="left")
            hi_e = ctk.CTkEntry(f, width=80, **input_style())
            hi_e.insert(0, f"{hi:g}")
            hi_e.pack(side="left", padx=(4, SPACE_S))
            len_l = ctk.CTkLabel(f, text="", font=FONT_SMALL, text_color=MUTED, width=70, anchor="w")
            len_l.pack(side="left")
            r: dict = {"frame": f, "lbl": lbl_e, "lo": lo_e, "hi": hi_e, "len": len_l}
            x_btn = ctk.CTkButton(f, text="✕", width=28, command=lambda: _remove_row(r),
                                  **button_style("secondary"))
            x_btn.pack(side="left", padx=(0, SPACE_M))
            qc_l = ctk.CTkLabel(f, text="", font=FONT_SMALL, text_color=MUTED, anchor="w",
                                justify="left")
            qc_l.pack(side="left", fill="x", expand=True)
            r["qc"] = qc_l

            def _upd_len(_e=None) -> None:
                try:
                    d = float(hi_e.get()) - float(lo_e.get())
                    len_l.configure(text=f"= {d:g} s" if d > 0 else "⚠ end ≤ start",
                                    text_color=MUTED if d > 0 else RED)
                except ValueError:
                    len_l.configure(text="")
            for e in (lo_e, hi_e):
                e.bind("<KeyRelease>", _upd_len)
            _upd_len()
            seg_rows.append(r)

        add_bar = ctk.CTkFrame(seg_card, fg_color="transparent")
        add_bar.pack(fill="x", pady=(SPACE_XS, 0))

        def _add_segment() -> None:
            if len(seg_rows) >= MAX_SEGMENTS:
                return
            try:
                last_hi = float(seg_rows[-1]["hi"].get()) if seg_rows else 0.0
                last_d = float(seg_rows[-1]["hi"].get()) - float(seg_rows[-1]["lo"].get())
            except ValueError:
                last_hi, last_d = 0.0, 300.0
            lo = min(max(last_hi, 0.0), dur)
            _add_row(f"Segment {chr(65 + len(seg_rows))}", lo, min(lo + max(last_d, 1.0), dur))
            _refresh_add_btn()

        add_btn = ctk.CTkButton(add_bar, text="＋  Add segment", width=0, command=_add_segment,
                                **button_style("secondary"))
        add_btn.pack(side="left")
        ctk.CTkLabel(add_bar, text=f"up to {MAX_SEGMENTS}",
                     font=FONT_MICRO, text_color=MUTED).pack(side="left", padx=(SPACE_M, 0))

        def _refresh_add_btn() -> None:
            add_btn.configure(state="normal" if len(seg_rows) < MAX_SEGMENTS else "disabled")

        _set_rows([("Segment A", 0.0, float(int(dur / 2))),
                   ("Segment B", float(int(dur / 2)), float(int(dur)))])

        # ── Run / export bar ──────────────────────────────────────────────
        ctk.CTkFrame(seg_card, height=1, fg_color=BORDER).pack(fill="x", pady=SPACE_M)
        ctrl = ctk.CTkFrame(seg_card, fg_color="transparent")
        ctrl.pack(fill="x")
        lbl_status = ctk.CTkLabel(seg_card, text="", font=FONT_SMALL, text_color=MUTED,
                                  anchor="w", justify="left")
        lbl_status.pack(fill="x", pady=(SPACE_S, 0))
        progress_bar = ctk.CTkProgressBar(ctrl, height=8, mode="indeterminate",
                                          progress_color=TEAL)
        progress_bar.pack(side="right", padx=(SPACE_M, 0))
        progress_bar.pack_forget()
        run_btn = ctk.CTkButton(ctrl, text="▶  Compare", **button_style("primary"))
        run_btn.pack(side="left")
        # Length of the non-overlapping blocks each segment is summarised by
        # (compare_segment_blocks): no beat-level test, see analysis.
        ctk.CTkLabel(ctrl, text="Block (s):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(SPACE_L, 4))
        ent_block = ctk.CTkEntry(ctrl, width=55, **input_style())
        ent_block.insert(0, f"{MouseECG.SEGMENT_BLOCK_S:g}")
        ent_block.pack(side="left")

        # ── Results: block summary, table, then the plot across the page ──
        res_hdr, res = _section("Results")
        lbl_mw = ctk.CTkLabel(res, text="Set the segments above, then click ▶ Compare.",
                              font=FONT_SMALL, text_color=MUTED, anchor="w", justify="left")
        lbl_mw.pack(fill="x", pady=(0, SPACE_S))
        res.bind("<Configure>", lambda e: lbl_mw.configure(wraplength=max(200, e.width - 8)))

        # One text widget with tab stops, like the Rhythm & HRV metric table:
        # the values line up under their segment, a row per metric.
        _scale = THEME.font_scale
        _fsize = -int(round(12 * _scale))
        tbl = tk.Text(res, bg=CARD, fg=TEXT, bd=0, highlightthickness=0, wrap="none",
                      font=(THEME.font_family, _fsize), height=2, cursor="arrow",
                      padx=SPACE_S, spacing1=2, spacing3=2)
        tbl.tag_configure("head", font=(THEME.font_family, _fsize, "bold"), foreground=MUTED)
        tbl.tag_configure("stripe", background=BG)
        tbl.tag_configure("unit", foreground=MUTED)
        tbl.tag_configure("delta", foreground=MUTED,
                          font=(THEME.font_family, -int(round(11 * _scale))))
        tbl.configure(state="disabled")
        grid = ctk.CTkFrame(res, fg_color="transparent")
        grid.pack(fill="x")
        grid.grid_columnconfigure(1, weight=1)
        tbl.lift(grid)
        plot_box = ctk.CTkFrame(grid, fg_color="transparent")

        def _place(n_segs: int) -> None:
            """Table and plot side by side, the plot under the table when the
            table needs the width (more than 3 segments)."""
            tbl.grid_forget(); plot_box.grid_forget()
            tbl.grid(in_=grid, row=0, column=0, sticky="nw")
            if n_segs <= 3:
                plot_box.grid(row=0, column=1, sticky="nsew", padx=(CARD_PAD, 0))
            else:
                plot_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(SPACE_M, 0))
        view_bar = ctk.CTkFrame(plot_box, fg_color="transparent")
        view_bar.pack(fill="x", pady=(0, SPACE_S))
        _view = {"mode": "rr"}
        _view_btns: "dict[str, ctk.CTkButton]" = {}

        def _set_view(mode: str) -> None:
            _view["mode"] = mode
            for m, b in _view_btns.items():
                b.configure(**toggle_style(m == mode))
            if _state["segs"] is not None:
                _redraw_plot()

        for mode, text in (("rr", "RR tachograms"), ("poincare", "Poincaré")):
            b = ctk.CTkButton(view_bar, text=text, width=0, command=lambda m=mode: _set_view(m),
                              **{**button_style("secondary"), **toggle_style(mode == "rr")})
            b.pack(side="left", padx=(0, SPACE_S))
            _view_btns[mode] = b
        plot_card = ctk.CTkFrame(plot_box, fg_color=CARD, corner_radius=0, height=460)
        plot_card.pack(fill="x")
        plot_card.pack_propagate(False)        # a scrolling page gives no height to fit
        plot_slot = CanvasSlot(plot_card, 14, 4.6, toolbar=False, yscale_bar=True,
                               placeholder="Click ▶ Compare to overlay tachograms")
        plot_slot.enable_wheel_yzoom()
        _place(2)

        _METRICS: "list[tuple[str, str, str]]" = list(self._SEGMENT_TABLE_METRICS)

        def _fmt(v: "Any") -> str:
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    return "—"
                if abs(fv) >= 100: return f"{fv:.1f}"
                if abs(fv) >= 10:  return f"{fv:.2f}"
                return f"{fv:.3f}"
            except Exception:
                return str(v) if v else "—"

        def _delta(va: float, vb: float) -> float:
            return (100.0 * (vb - va) / abs(va)
                    if np.isfinite(va) and np.isfinite(vb) and abs(va) > 1e-9 else float("nan"))

        _COUNT_KEYS = {"n_beats", "n_blocks", "duration_s"}

        def _fmt_cell(key: str, v: float) -> str:
            if key in _COUNT_KEYS and np.isfinite(v) and float(v).is_integer():
                return f"{v:.0f}"
            return _fmt(v)

        def _populate_table(segs: "list[dict]") -> None:
            val_w, d_w = int(110 * _scale), int(80 * _scale)
            x = int(270 * _scale)
            stops = []
            for k in range(len(segs)):
                x += val_w
                stops += [x, "right"]
                if k:
                    x += d_w
                    stops += [x, "right"]
            # Text width is in characters: from the font's own "0".
            import tkinter.font as tkfont
            char_px = max(1, tkfont.Font(font=tbl.cget("font")).measure("0"))
            tbl.configure(state="normal", tabs=stops, width=int(x / char_px) + 3)
            _place(len(segs))
            tbl.delete("1.0", "end")
            for k, s_ in enumerate(segs):
                tbl.tag_configure(f"seg{k}", font=(THEME.font_family, _fsize, "bold"),
                                  foreground=_color(k))
            tbl.insert("end", "Metric", "head")
            for k, s_ in enumerate(segs):
                tbl.insert("end", "\t")
                tbl.insert("end", s_["label"], f"seg{k}")
                if k:
                    tbl.insert("end", "\tΔ %", "head")
            ref = segs[0]
            for ri, (key, name, unit) in enumerate(_METRICS):
                line = ("stripe",) if ri % 2 == 0 else ()
                tbl.insert("end", "\n" + name, line)
                if unit:
                    tbl.insert("end", f"  {unit}", ("unit",) + line)
                va = float(ref.get(key, float("nan")))
                for k, s_ in enumerate(segs):
                    v = float(s_.get(key, float("nan")))
                    tbl.insert("end", "\t" + _fmt_cell(key, v), line)
                    if not k:
                        continue
                    d = _delta(va, v)
                    if not np.isfinite(d):
                        txt = "—"
                    elif abs(d) < 0.05:
                        txt = "0.0 %"
                    else:
                        txt = f"{'▲' if d > 0 else '▼'} {abs(d):.1f} %"
                    tbl.insert("end", "\t" + txt, ("delta",) + line)
            tbl.configure(height=len(_METRICS) + 1, state="disabled")

        def _rr_drawer(segs: "list[dict]", colors: "list[str]"):
            """The superimposed RR tachograms of *segs*, as a draw function."""
            def _draw(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                for s, c in zip(segs, colors):
                    t, r = s["t_rr"], s["rr_ms"]
                    if len(t) and len(r):
                        ax.plot(t - t[0], r, color=c, lw=1.0, alpha=0.85, label=s["label"])
                        ax.axhline(float(np.mean(r)), color=c, lw=1.0, ls="--", alpha=0.5)
                ax.set_xlabel("Time within segment (s)", fontsize=9)
                ax.set_ylabel("RR (ms)", fontsize=9)
                ax.set_title("RR intervals — segments superimposed", loc="left", fontsize=9)
                ax.legend(framealpha=0, fontsize=8)
            return _draw

        def _redraw_plot() -> None:
            segs = _state["segs"]
            colors = [_color(k) for k in range(len(segs))]
            if _view["mode"] == "poincare":
                plot_slot.update(lambda fig: draw_poincare_set(fig, segs, colors))
            else:
                plot_slot.update(_rr_drawer(segs, colors))

        def _show_flags(flags: "list[list[str]]") -> None:
            for r, fl in zip(seg_rows, flags):
                r["qc"].configure(text=("⚠  " + "  ·  ".join(fl)) if fl else "✓  quality ok",
                                  text_color=ORANGE if fl else GREEN)

        def _run_compare() -> None:
            try:
                windows = _read_rows()
                ev = _event_times()
                margin = float(ent_margin.get() or 0)
            except ValueError as exc:
                lbl_status.configure(text=f"⚠  {exc}", text_color=RED); return
            try:
                block_s = float(ent_block.get())
                if not (np.isfinite(block_s) and block_s > 0):
                    raise ValueError("block length")
            except ValueError:
                lbl_status.configure(text="⚠  Block length must be a positive number of seconds",
                                     text_color=RED)
                return
            for i, (_lbl, lo, hi) in enumerate(windows):
                if lo >= hi:
                    lbl_status.configure(text=f"⚠  Segment {i + 1}: start ≥ end", text_color=RED)
                    return
                if lo < 0 or hi > dur:
                    lbl_status.configure(
                        text=f"⚠  Segment {i + 1}: outside recording ({dur:.1f} s)",
                        text_color=RED)
                    return
            run_btn.configure(state="disabled")
            progress_bar.pack(side="right", padx=(SPACE_M, 0))
            progress_bar.start()
            lbl_status.configure(text="  Computing…", text_color=ORANGE)
            # HRV inputs read on the main thread: the user's RR bounds and
            # the beats removed by artifact correction (excluded spans).
            rr_lo, rr_hi = self.analysis_ctrl.hrv_rr_bounds()
            seg_breaks   = self.analysis_ctrl.hrv_breaks(rp)

            import threading as _th

            def _worker():
                try:
                    from ecg.core.analysis import compare_segments, compute_segment_stats
                    seg_kw = dict(lf_band=lf, hf_band=hf, vlf_band=vlf,
                                  rr_min_ms=rr_lo, rr_max_ms=rr_hi,
                                  breaks=seg_breaks, block_s=block_s)
                    segs = [compute_segment_stats(sig, rp, fs, lo, hi, lbl, **seg_kw)
                            for lbl, lo, hi in windows]
                    # Descriptive block-mean comparison of each segment with the
                    # first (no beat-level test: beats are not replicates).
                    cmps = [compare_segments(segs[0], s) for s in segs[1:]]
                    flags = quality_flags(segs, ev, margin)
                except Exception as exc:
                    log.exception("Segment comparison worker failed")
                    win.after(0, lambda e=exc: _on_worker_error(e))
                    return
                win.after(0, lambda: _on_done(segs, cmps, flags))

            def _on_worker_error(exc: Exception) -> None:
                run_btn.configure(state="normal")
                progress_bar.stop(); progress_bar.pack_forget()
                lbl_status.configure(text=f"⚠  Comparison failed — {exc}", text_color=RED)

            def _on_done(segs: "list[dict]", cmps: "list[dict]", flags) -> None:
                run_btn.configure(state="normal")
                progress_bar.stop(); progress_bar.pack_forget()
                _state.update(segs=segs, cmps=cmps, flags=flags)
                self._last_seg_a = segs[0]
                self._last_seg_b = segs[1] if len(segs) > 1 else None
                # Kept for the exports here and the main ones (Report tab),
                # with the recording state it rests on (_current_comparison).
                self.analysis.last_comparison = ComparisonResult(
                    segs=segs, metrics=list(_METRICS), cmps=cmps, flags=flags,
                    template=_state["template"], subject=_subject(),
                    filepath=self.signal.filepath or "", event=ev,
                    artefact_margin_s=margin, block_s=block_s,
                    lf_band=tuple(lf), hf_band=tuple(hf))
                # The state the tab -- hence rp / sig of this run -- was
                # built for, not the app's now (peaks may have been edited
                # on another tab while the worker ran).
                self.analysis.last_comparison_basis = built_for
                for b in _export_btns:
                    self._set_btn_enabled(b, True, "secondary")
                errs = [s["error"] for s in segs if s.get("error")]
                n_flag = sum(1 for f in flags if f)
                if errs:
                    lbl_status.configure(text=f"⚠  {errs[0]}", text_color=RED)
                else:
                    withheld = [s["label"] for s in segs if s.get("freq_withheld_reason")]
                    lbl_status.configure(
                        text="✓  " + "  ·  ".join(f"{s['label']}: {s['n_beats']} beats" for s in segs)
                             + (f"  ·  LF/HF withheld for {', '.join(withheld)}" if withheld else "")
                             + (f"  ·  ⚠ {n_flag} segment(s) to check" if n_flag else ""),
                        text_color=ORANGE if (withheld or n_flag) else GREEN)
                from ecg.core.analysis import segment_comparison_text
                lines = [segment_comparison_text(c, segs[0]["label"], s["label"]).split("\n")
                         for c, s in zip(cmps, segs[1:])]
                lbl_mw.configure(text="\n".join(ln[0] for ln in lines)
                                 + (f"\n{lines[0][1]}" if lines and len(lines[0]) > 1 else ""))
                _show_flags(flags)
                _populate_table(segs)
                _redraw_plot()

            _th.Thread(target=_worker, daemon=True).start()

        # ── Export: Excel / CSV, Prism, clipboard, figures, group table ───
        def _result() -> "Optional[ComparisonResult]":
            res_ = self.analysis.last_comparison
            if res_ is None or not _state["segs"]:
                messagebox.showwarning("No results", "Run the comparison first.", parent=win)
                return None
            res_.subject = _subject()       # a subject typed after the run counts
            return res_

        def _stamp() -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

        def _base_name() -> str:
            return f"{_subject() or 'subject'}_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def _done(text: str, ok: bool = True) -> None:
            if not lbl_exp.winfo_manager():      # shown from the first export on
                lbl_exp.pack(fill="x", pady=(0, SPACE_XS), before=lbl_mw)
            lbl_exp.configure(text=("✓  " if ok else "⚠  ") + text, text_color=GREEN if ok else RED)

        def _export_excel() -> None:
            res_ = _result()
            if res_ is None:
                return
            path = filedialog.asksaveasfilename(
                parent=win, title="Export the segment comparison",
                defaultextension=".xlsx", initialfile=_base_name() + ".xlsx",
                filetypes=[("Excel workbook", "*.xlsx"), ("CSV (metric table)", "*.csv"),
                           ("All files", "*.*")])
            if not path:
                return
            try:
                if path.lower().endswith(".csv"):
                    write_csv(res_, path)
                else:
                    write_xlsx(res_, path, exported_at=_stamp())
            except Exception as exc:
                log.exception("comparison export")
                messagebox.showerror("Export", str(exc), parent=win)
                _done(f"export failed: {exc}", ok=False)
                return
            _done(f"saved → {os.path.basename(path)}")

        def _export_prism() -> None:
            res_ = _result()
            if res_ is None:
                return
            path = filedialog.asksaveasfilename(
                parent=win, title="Export the segment comparison to GraphPad Prism",
                defaultextension=".pzfx", initialfile=_base_name() + ".pzfx",
                filetypes=[("GraphPad Prism", "*.pzfx"), ("All files", "*.*")])
            if not path:
                return
            try:
                n = write_pzfx(res_, path, exported_at=_stamp())
            except Exception as exc:
                log.exception("comparison Prism export")
                messagebox.showerror("Prism export", str(exc), parent=win)
                _done(f"Prism export failed: {exc}", ok=False)
                return
            _done(f"Prism: {n} tables → {os.path.basename(path)}")

        def _copy_table() -> None:
            res_ = _result()
            if res_ is None:
                return
            win.clipboard_clear()
            win.clipboard_append(comparison_tsv(res_))
            _done("metric table copied — paste it into Excel or Prism")

        def _export_figures() -> None:
            res_ = _result()
            if res_ is None:
                return
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the comparison figures (tachograms and Poincaré)",
                defaultextension=".png", initialfile=_base_name() + ".png",
                confirmoverwrite=False,
                filetypes=[("PNG image (300 dpi)", "*.png"), ("PDF (vector)", "*.pdf"),
                           ("SVG (vector)", "*.svg"), ("All files", "*.*")])
            if not path:
                return
            base, ext = os.path.splitext(path)
            ext = ext.lower() if ext.lower() in (".png", ".pdf", ".svg") else ".png"
            targets = [base + sfx + ext for sfx in ("_tachograms", "_poincare")]
            existing = [os.path.basename(t) for t in targets if os.path.exists(t)]
            if existing and not messagebox.askyesno(
                    "Figures", "Replace " + " and ".join(existing) + "?", parent=win):
                return
            segs_ = res_.segs
            colors = [_color(k) for k in range(len(segs_))]
            n_rows = 1 + (len(segs_) > 4)
            written = []
            for suffix, figsize, draw in (
                    ("_tachograms", (10, 4), _rr_drawer(segs_, colors)),
                    ("_poincare", (min(4 * len(segs_), 16), 4.6 * n_rows),
                     lambda fig: draw_poincare_set(fig, segs_, colors, title=_subject()))):
                out = base + suffix + ext
                try:
                    fig = Figure(figsize=figsize, dpi=300, facecolor="white", layout="constrained")
                    if draw(fig) is not False:
                        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
                        written.append(os.path.basename(out))
                    plt.close(fig)
                except Exception as exc:
                    log.warning("comparison figure %s: %s", out, exc)
                    _done(f"figure {os.path.basename(out)} failed: {exc}", ok=False)
                    return
            _done("figures → " + "  +  ".join(written))

        def _add_to_group() -> None:
            res_ = _result()
            if res_ is None:
                return
            path = filedialog.asksaveasfilename(
                parent=win, title="Group table: add this recording (created if new)",
                defaultextension=".xlsx", initialfile="group_segments.xlsx",
                confirmoverwrite=False, filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")])
            if not path:
                return
            try:
                n, replaced = append_group_row(path, group_row(
                    res_.subject, res_.filepath, res_.segs, res_.metrics,
                    template=res_.template, flags=res_.flags))
            except Exception as exc:
                messagebox.showerror("Group table", str(exc), parent=win)
                return
            _done(f"{res_.subject or 'recording'} {'updated in' if replaced else 'added to'} "
                  f"{os.path.basename(path)} ({n} recording{'s' if n != 1 else ''})")

        run_btn.configure(command=_run_compare)
        lbl_exp = ctk.CTkLabel(res, text="", font=FONT_SMALL, text_color=MUTED,
                               anchor="w", justify="left")
        _export_btns: "list[ctk.CTkButton]" = []
        # Right to left: the header packs them from its right edge.
        for text, cmd, tiptext in (
                ("➕  Add to group table…", _add_to_group,
                 "One row per recording (subject, file, template, every metric and the "
                 "quality of each segment): the table for a paired / repeated-measures "
                 "test across animals. Columns are named after the segment labels: use "
                 "the same template for every animal. A recording already in it is replaced."),
                ("Figures…", _export_figures,
                 "The tachograms and the Poincaré plots, 300 dpi PNG or vector PDF / SVG"),
                ("📋  Copy table", _copy_table,
                 "The metric table as tab-separated text: paste it into Excel or Prism"),
                ("Prism…", _export_prism,
                 "GraphPad Prism file: the metric table (one column per segment), the "
                 "HR block means, the RR series and the Poincaré pairs of each segment; "
                 "the windows, template and quality in the project notes"),
                ("📊  Excel…", _export_excel,
                 "Workbook: Comparison (Δ % against the first segment), Quality, Summary, "
                 "Blocks, then the RR series and Poincaré pairs of each segment. "
                 "Choose .csv for the metric table only.")):
            b = ctk.CTkButton(res_hdr, text=text, width=0, command=cmd, state="disabled",
                              **{**button_style("secondary"), **disabled_style()})
            b.pack(side="right", padx=(SPACE_S, 0))
            tip(b, tiptext)
            _export_btns.append(b)
        ctk.CTkLabel(res_hdr, text="Export:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="right", padx=(0, SPACE_XS))

        _reload_templates()
        ctk.CTkLabel(res,
                     text=f"Equal durations give a fair comparison  ·  LF {lf[0]:.2f}–{lf[1]:.2f} Hz  "
                          f"HF {hf[0]:.2f}–{hf[1]:.2f} Hz  ·  wheel: zoom Y  ·  middle-click: reset Y",
                     font=FONT_HINT, text_color=MUTED, anchor="w").pack(
            fill="x", pady=(SPACE_XS, 0))

    # ── Group: the group table across animals ────────────────────────────

    def _build_group_card(self: ECGApp, win) -> None:
        """GROUP card: open a group table (one row per recording), choose two
        segments, check the rows, run the paired tests across animals and
        draw the group figure; save both. Independent of the recording open
        (its choices survive a rebuild of the tab, in _group_state)."""
        st = getattr(self, "_group_state", None)
        if st is None:
            st = self._group_state = {"path": "", "a": "", "b": "", "exclude": True,
                                      "name_a": "", "name_b": ""}
        tip = self._bind_hover_tip
        card = ctk.CTkFrame(win, **card_style())
        card.pack(fill="x", padx=PAGE_PAD_X, pady=(0, PAGE_GAP))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=CARD_PAD, pady=(CARD_PAD - 4, CARD_PAD))
        hdr = ctk.CTkFrame(body, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, SPACE_M))
        ctk.CTkLabel(hdr, text="GROUP — ALL ANIMALS", font=FONT_SECTION_HDR,
                     text_color=MUTED).pack(side="left")
        ctk.CTkLabel(hdr, text="paired tests across animals on a group table",
                     font=FONT_SMALL, text_color=MUTED).pack(side="left", padx=(SPACE_M, 0))

        row1 = ctk.CTkFrame(body, fg_color="transparent")
        row1.pack(fill="x", pady=(0, SPACE_S))
        lbl_file = ctk.CTkLabel(row1, text="", font=FONT_SMALL, text_color=MUTED, anchor="w")
        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkLabel(row2, text="Compare:", font=FONT_SMALL, text_color=MUTED).pack(side="left")
        cb_a = ctk.CTkComboBox(row2, width=170, values=[""], state="readonly", **combo_style())
        cb_a.pack(side="left", padx=(4, SPACE_S))
        ctk.CTkLabel(row2, text="vs", font=FONT_SMALL, text_color=MUTED).pack(side="left")
        cb_b = ctk.CTkComboBox(row2, width=170, values=[""], state="readonly", **combo_style())
        cb_b.pack(side="left", padx=(4, SPACE_M))
        ctk.CTkLabel(row2, text="Names on the figure:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        ent_na = ctk.CTkEntry(row2, width=110, placeholder_text="A", **input_style())
        ent_na.pack(side="left", padx=(4, SPACE_XS))
        ent_nb = ctk.CTkEntry(row2, width=110, placeholder_text="B", **input_style())
        ent_nb.pack(side="left", padx=(0, SPACE_M))
        for e, k in ((ent_na, "name_a"), (ent_nb, "name_b")):
            if st[k]:
                e.insert(0, st[k])
        chk_ex = ctk.CTkCheckBox(row2, text="Leave out rows to check", font=FONT_SMALL,
                                 text_color=MUTED, fg_color=BLUE, checkmark_color="white",
                                 border_color=BORDER2, width=16)
        if st["exclude"]:
            chk_ex.select()
        chk_ex.pack(side="left", padx=(0, SPACE_M))
        tip(chk_ex, "Rows with a quality alert, a missing segment, a subject on several rows, "
                    "another template or another duration are left out of the tests and the "
                    "figure (they stay listed below).")
        chk_all = ctk.CTkCheckBox(row2, text="All metrics", font=FONT_SMALL,
                                  text_color=MUTED, fg_color=BLUE, checkmark_color="white",
                                  border_color=BORDER2, width=16)
        if st.get("all"):
            chk_all.select()
        chk_all.pack(side="left", padx=(0, SPACE_M))
        tip(chk_all, "Off: one metric per question (HR, SDNN, SD1, SD2, LF/HF), so Holm is not "
                     "made stricter by equivalent tests. " + DEFAULT_METRICS_NOTE + ".")
        run_btn = ctk.CTkButton(row2, text="▶  Run", width=0, **button_style("primary"))
        run_btn.pack(side="left")

        lbl_status = ctk.CTkLabel(body, text="", font=FONT_SMALL, text_color=MUTED,
                                  anchor="w", justify="left")
        lbl_status.pack(fill="x")
        body.bind("<Configure>", lambda e: lbl_status.configure(wraplength=max(200, e.width - 8)))
        _scale = THEME.font_scale
        _fsize = -int(round(12 * _scale))
        txt = tk.Text(body, bg=CARD, fg=TEXT, bd=0, highlightthickness=0, wrap="none",
                      font=(THEME.font_family, _fsize), height=1, cursor="arrow",
                      padx=SPACE_S, spacing1=2, spacing3=2)
        txt.tag_configure("head", font=(THEME.font_family, _fsize, "bold"), foreground=MUTED)
        txt.tag_configure("stripe", background=BG)
        txt.tag_configure("sig", foreground=GREEN, font=(THEME.font_family, _fsize, "bold"))
        txt.tag_configure("warn", foreground=ORANGE)
        txt.tag_configure("muted", foreground=MUTED)
        txt.configure(state="disabled")
        plot_card = ctk.CTkFrame(body, fg_color=CARD, corner_radius=0, height=430)
        plot_card.pack_propagate(False)
        slot = CanvasSlot(plot_card, 14, 4.3, toolbar=False,
                          placeholder="Open a group table and click Run")
        res_state: "dict[str, Any]" = {"df": None, "used": None, "stats": None, "checks": None,
                                      "left_out": []}

        def _load(path: str, quiet: bool = False) -> bool:
            def fail(msg: str) -> bool:
                if quiet:            # a remembered table: say it, no dialog
                    lbl_status.configure(text=f"⚠  {msg}", text_color=ORANGE)
                else:
                    messagebox.showerror("Group table", msg, parent=win)
                return False
            try:
                df = read_group_table(path)
            except Exception as exc:
                return fail(f"{os.path.basename(path)}: {exc}")
            tags = segment_tags(df)
            if len(tags) < 2:
                return fail(f"{os.path.basename(path)} holds fewer than two segments.")
            res_state["df"] = df
            st["path"] = path
            cb_a.configure(values=tags)
            cb_b.configure(values=tags)
            # The two most filled segments by default, or the ones chosen before.
            filled = sorted(tags, key=lambda t: -int(df[f"start (s) {t}"].notna().sum()))
            a = st["a"] if st["a"] in tags else filled[0]
            b = st["b"] if st["b"] in tags and st["b"] != a else next(t for t in filled if t != a)
            cb_a.set(a)
            cb_b.set(b)
            lbl_file.configure(text=f"{os.path.basename(path)}  ·  {len(df)} rows  ·  "
                                    f"segments: {', '.join(tags)}")
            return True

        def _open() -> None:
            path = filedialog.askopenfilename(
                parent=win, title="Open a group table (Add to group table…)",
                filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
                initialdir=os.path.dirname(st["path"]) if st["path"] else None)
            if path and _load(path):
                _run()

        open_btn = ctk.CTkButton(row1, text="📂  Open group table…", width=0, command=_open,
                                 **button_style("secondary"))
        open_btn.pack(side="left", padx=(0, SPACE_M))
        lbl_file.pack(side="left", fill="x", expand=True)

        def _show(stats, checks, a: str, b: str, left_out: "list[str]") -> None:
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            x = int(260 * _scale)
            stops = []
            for w in (50, 150, 150, 110, 80, 100, 90, 90):
                x += int(w * _scale)
                stops += [x, "right"]
            txt.configure(tabs=stops)
            heads = ["Metric", "n (≠0)", f"Median {a}", f"Median {b}", "Median Δ", "up/down",
                     "Wilcoxon p", "p (Holm)", "paired t p"]
            txt.insert("end", "\t".join(heads), "head")

            def fmt(v: float, p: bool = False) -> str:
                if v is None or not np.isfinite(v):
                    return "—"
                if p:
                    return f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
                return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
            for i, r in stats.reset_index(drop=True).iterrows():
                line = ("stripe",) if i % 2 == 0 else ()
                txt.insert("end", "\n" + str(r["Metric"]), line)
                for v in (f"{int(r['n'])} ({int(r['n tested (Δ ≠ 0)'])})",
                          fmt(r[f"Median {a}"]), fmt(r[f"Median {b}"]),
                          fmt(r["Median Δ (B − A)"]), str(r["n up / down"])):
                    txt.insert("end", "\t" + v, line)
                ph = r["Wilcoxon p (Holm)"]
                txt.insert("end", "\t" + fmt(r["Wilcoxon p"], True), line)
                txt.insert("end", "\t" + fmt(ph, True),
                           ("sig",) + line if np.isfinite(ph) and ph < 0.05 else line)
                txt.insert("end", "\t" + fmt(r["Paired t p"], True), line)
            bad = [c for c in checks if not c.ok]
            if bad:
                txt.insert("end", "\n\nRows to check", "head")
                kept = dict(zip(map(id, checks), res_state.get("keep") or [True] * len(checks)))
                for c in bad:
                    tag = "kept" if kept.get(id(c), True) else "left out"
                    txt.insert("end", f"\n[{tag}] {c.subject or '—'} · {c.file}: "
                               + "; ".join(c.problems), "warn")
            n_lines = int(txt.index("end-1c").split(".")[0])
            txt.configure(height=n_lines, state="disabled")
            if not txt.winfo_manager():
                txt.pack(fill="x", pady=(SPACE_S, SPACE_M))
                plot_card.pack(fill="x")

        def _run() -> None:
            # Read again: rows added since (Add to group table…) count.
            if st["path"] and os.path.exists(st["path"]):
                a0, b0 = cb_a.get(), cb_b.get()
                if not _load(st["path"]):
                    return
                if a0 in cb_a.cget("values") and b0 in cb_b.cget("values") and a0 != b0:
                    cb_a.set(a0)
                    cb_b.set(b0)
            df = res_state["df"]
            if df is None:
                _open()
                return
            a, b = cb_a.get(), cb_b.get()
            if not a or not b or a == b:
                lbl_status.configure(text="⚠  choose two different segments", text_color=RED)
                return
            st.update(a=a, b=b, exclude=bool(chk_ex.get()), all=bool(chk_all.get()),
                      name_a=ent_na.get().strip(), name_b=ent_nb.get().strip())
            checks = check_rows(df, a, b)
            keep = [c.ok or not st["exclude"] for c in checks]
            # A row without both segments can never be tested.
            keep = [k and not any(p.startswith("no segment") for p in c.problems)
                    for k, c in zip(keep, checks)]
            used = df.loc[keep].reset_index(drop=True)
            left_out = [c.subject or c.file for c, k in zip(checks, keep) if not k]
            stats = paired_stats(used, a, b, "all" if st["all"] else None)
            res_state.update(used=used, stats=stats, checks=checks, left_out=left_out, keep=keep)
            n = len(used)
            n_test = int(stats["n tested (Δ ≠ 0)"].max()) if len(stats) else 0
            names = (st["name_a"] or a, st["name_b"] or b)
            p_min = smallest_wilcoxon_p(n_test) if n_test else float("nan")
            weak = not (p_min < 0.05)          # also NaN: nothing testable
            msg = (f"{n} animal{'s' if n != 1 else ''} tested"
                   + (f"  ·  left out: {', '.join(left_out)}" if left_out else "")
                   + (f"  ·  with n = {n_test} non-zero differences the smallest possible "
                      f"Wilcoxon p is {p_min:.3g}" if n_test else "")
                   + ("  ·  no metric can reach p < 0.05" if weak and n_test else ""))
            lbl_status.configure(text=("⚠  " if left_out or weak else "✓  ") + msg,
                                 text_color=ORANGE if left_out or weak else GREEN)
            _show(stats, checks, a, b, left_out)
            slot.update(lambda fig: draw_group_figure(fig, used, a, b, names))
            for btn_ in (btn_xlsx, btn_fig):
                self._set_btn_enabled(btn_, True, "secondary")

        run_btn.configure(command=_run)

        def _save_stats() -> None:
            if res_state["stats"] is None:
                return
            a, b = st["a"], st["b"]
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the group statistics",
                defaultextension=".xlsx", initialfile=f"group_stats_{a}_vs_{b}.xlsx",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")])
            if not path:
                return
            try:
                write_group_report(path, res_state["used"], a, b, res_state["stats"],
                                   res_state["checks"], excluded=res_state["left_out"],
                                   note=f"group table: {os.path.basename(st['path'])}")
            except Exception as exc:
                messagebox.showerror("Group statistics", str(exc), parent=win)
                return
            lbl_status.configure(text=f"✓  statistics → {os.path.basename(path)}", text_color=GREEN)

        def _save_figure() -> None:
            used = res_state["used"]
            if used is None:
                return
            a, b = st["a"], st["b"]
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the group figure",
                defaultextension=".png", initialfile=f"group_poincare_{a}_vs_{b}.png",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("PNG image (300 dpi)", "*.png"), ("PDF (vector)", "*.pdf"),
                           ("SVG (vector)", "*.svg"), ("All files", "*.*")])
            if not path:
                return
            st.update(name_a=ent_na.get().strip(), name_b=ent_nb.get().strip())
            fig = Figure(figsize=(14, 4.6), dpi=300, facecolor="white", layout="constrained")
            try:
                draw_group_figure(fig, used, a, b, (st["name_a"] or a, st["name_b"] or b))
                fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
            except Exception as exc:
                messagebox.showerror("Group figure", str(exc), parent=win)
                return
            finally:
                plt.close(fig)
            lbl_status.configure(text=f"✓  figure → {os.path.basename(path)}", text_color=GREEN)

        btn_fig = ctk.CTkButton(hdr, text="Figure…", width=0, command=_save_figure,
                                state="disabled", **{**button_style("secondary"), **disabled_style()})
        btn_fig.pack(side="right", padx=(SPACE_S, 0))
        tip(btn_fig, "The group Poincaré (one ellipse per animal, the group mean thick) and "
                     "the before-after panels: 300 dpi PNG or vector PDF / SVG")
        btn_xlsx = ctk.CTkButton(hdr, text="📊  Statistics…", width=0, command=_save_stats,
                                 state="disabled", **{**button_style("secondary"), **disabled_style()})
        btn_xlsx.pack(side="right", padx=(SPACE_S, 0))
        tip(btn_xlsx, "Excel: the paired tests (Wilcoxon, paired t, Holm), the rows checked "
                      "and the paired values of every metric, one row per animal")
        ctk.CTkLabel(body, text=(
            "One row per animal: the test unit is the animal. Use the same template and labels for "
            "every animal. Holm corrects the Wilcoxon p over the metrics listed; pick your primary "
            "metric in advance."), font=FONT_HINT, text_color=MUTED, anchor="w",
            justify="left").pack(fill="x", side="bottom", pady=(SPACE_S, 0))
        if st["path"] and os.path.exists(st["path"]) and _load(st["path"], quiet=True):
            _run()
