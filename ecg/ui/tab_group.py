# -*- coding: utf-8 -*-
"""
ecg.ui.tab_group
----------------
ECGApp Group tab: the animals of one experiment, from a group table
(ecg.io.group_data), compared without groups, by their free group or by sex,
on their Pre/Post segments, their time course aligned on an annotation, their
whole recording or their event-locked results; the figure, the results table
and the exports (statistics workbook, figure, GraphPad Prism new file or
template). Independent of the recording open, except Add open recording,
which writes it to the table.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING, Any, Optional

import customtkinter as ctk  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.io.prism_template import PrismTemplateError, fill_template, write_metric_pzfx
from ecg.io.subject_name import normalise_sex
from ecg.ui.group_tab_figure import draw_group_tab_figure
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_GROUP
from ecg.ui.theme import (
    BG, BORDER, BORDER2, CARD, GREEN, MUTED, ORANGE, RED, TEXT, THEME, BLUE, CARD_PAD,
    FONT_HINT, FONT_SECTION_HDR, FONT_SMALL, PAGE_GAP, PAGE_PAD_X, PAGE_PAD_Y, SPACE_M,
    SPACE_S, SPACE_XS, button_style, card_style, combo_style, input_style,
)

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")

ALL_EVENTS = "All events"
# The last group table opened, remembered per user (read at call time, so a
# test can point it elsewhere).
PREFS_PATH = Path.home() / ".ecg_group_table.json"
FIG_SIZE = (12.0, 4.6)


def recording_lines(results: "Optional[dict]", metric_rows: "Any", annotations: "Any",
                    event_rows: "Any", comparison: "Any" = None, sex: str = ""
                    ) -> "dict[str, Optional[pd.DataFrame]]":
    """The group-table lines of the recording open ({sheet: frame}): its
    whole-recording metrics and time course (after Analyze), annotations,
    event-locked results (the event registry) and, with *comparison* (a
    Compare segments result), its segments. A kind without data this time
    is left out, so the recording's lines from an earlier add stay (Add to
    group table from Compare segments without Analyze must not drop its
    whole-recording and time-course lines)."""
    from ecg.core.analysis import rr_timeseries
    from ecg.io.group_table import group_row
    out: "dict[str, Optional[pd.DataFrame]]" = {}
    whole = gd.whole_frame(metric_rows or [])
    if len(whole):
        out[gd.SHEET_WHOLE] = whole
    ts = rr_timeseries(results) if results else None
    tc = (gd.time_frame(ts["Time_s"], ts["RR_ms"], ts["valid"])
          if ts is not None and len(ts) else None)
    if tc is not None and len(tc):
        out[gd.SHEET_TIME] = tc
    an = gd.annotations_frame(annotations or [])
    if len(an):
        out[gd.SHEET_ANNOT] = an
    ev = gd.events_frame(event_rows or [])
    if len(ev):
        out[gd.SHEET_EVENTS] = ev
    if comparison is not None:
        row = group_row(comparison.subject, comparison.filepath, comparison.segs,
                        comparison.metrics, template=comparison.template,
                        flags=comparison.flags, sex=sex)
        for k in ("subject", "file"):
            row.pop(k, None)
        out[gd.SHEET_SEGMENTS] = pd.DataFrame([row])
    return out


def load_last_table() -> str:
    try:
        return str(json.loads(PREFS_PATH.read_text(encoding="utf-8")).get("path", ""))
    except Exception:
        return ""


def save_last_table(path: str) -> None:
    try:
        PREFS_PATH.write_text(json.dumps({"path": path}), encoding="utf-8")
    except Exception as exc:
        log.debug("group table preference not saved: %s", exc)


def _float(entry: "Any", default: float) -> float:
    try:
        v = float(str(entry.get()).replace(",", ".").replace("−", "-").strip())
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class TabGroupMixin:
    """ECGApp Group tab construction and handlers."""

    def _build_tab_group(self: ECGApp) -> None:
        page = self.tabs.tab(TAB_GROUP)
        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)
        holder = ctk.CTkScrollableFrame(page, fg_color=BG, corner_radius=0,
                                        scrollbar_button_color=BORDER,
                                        scrollbar_button_hover_color=BORDER2)
        holder.grid(row=0, column=0, sticky="nsew")
        self._build_group_page(holder)

    # ── the recording open → the group table ─────────────────────────────

    def _group_lines_of_open_recording(self: ECGApp, comparison: "Any" = None
                                       ) -> "tuple[str, str, str, dict]":
        """(subject, sex, file, {sheet: lines}) of the recording open."""
        from ecg.io import db
        sig = getattr(self, "signal", None)
        filepath = str(getattr(sig, "filepath", "") or "")
        ent = getattr(self, "ent_subject", None)
        subject = ent.get().strip() if ent is not None else ""
        ent_sex = getattr(self, "ent_sex", None)
        sex = normalise_sex(ent_sex.get()) if ent_sex is not None else ""
        analysis = getattr(self, "analysis", None)
        results = getattr(analysis, "results", None)
        rows = []
        if results:
            try:
                rows = list(self.export_ctrl.metric_table().rows)
            except Exception as exc:          # a stale module must not block the rest
                log.warning("group table: metric table: %s", exc)
        events = db.list_event_rows(filepath=filepath, valid_only=False) if filepath else []
        if comparison is None:
            cur = getattr(self, "_current_comparison", None)
            comparison = cur() if callable(cur) else None
        lines = recording_lines(results, rows, getattr(analysis, "annotations", None),
                                events, comparison, sex)
        # The subject the comparison was run under keys the row, as the
        # Compare segments group row always did (else the same file could
        # come in twice, under two subjects).
        cmp_subject = str(getattr(comparison, "subject", "") or "").strip()
        subject = cmp_subject or subject
        return subject or os.path.splitext(os.path.basename(filepath))[0], sex, filepath, lines

    def _add_open_recording_to_group_table(self: ECGApp, path: str,
                                           comparison: "Any" = None) -> "tuple[int, bool, str]":
        """Write the recording open to the group table at *path*. Returns
        (animals in the table, replaced, what was written)."""
        subject, sex, filepath, lines = self._group_lines_of_open_recording(comparison)
        if not filepath:
            raise ValueError("Open a recording first.")
        got = [k for k, v in lines.items() if v is not None]
        if not any(k in got for k in (gd.SHEET_SEGMENTS, gd.SHEET_WHOLE, gd.SHEET_TIME,
                                      gd.SHEET_EVENTS)):
            raise ValueError("Nothing to add yet: run Analyze (whole recording and time "
                             "course), Compare segments or the event-locked analysis first.")
        n, replaced = gd.add_recording(path, subject, sex, filepath, lines)
        names = {gd.SHEET_SEGMENTS: "segments", gd.SHEET_WHOLE: "whole recording",
                 gd.SHEET_TIME: "time course", gd.SHEET_EVENTS: "events",
                 gd.SHEET_ANNOT: "annotations"}
        what = ", ".join(names[k] for k in names if k in got)
        return n, replaced, f"{subject}: {what}"

    # ── the page ─────────────────────────────────────────────────────────

    def _build_group_page(self: ECGApp, win) -> None:
        """The Group tab's cards in *win*; its choices live in
        self._grouptab_state (they survive a rebuild)."""
        st = getattr(self, "_grouptab_state", None)
        if st is None:
            st = self._grouptab_state = {"path": load_last_table()}
        for k, v in (("path", ""), ("by", gc.BY_GROUP), ("analysis", gc.AN_SEGMENTS),
                     ("a", ""), ("b", ""), ("stim", ""), ("tc_metric", "HR"), ("bin", "60"),
                     ("from", "-10"), ("to", "30"), ("w0", "0"), ("w1", "5"), ("delta", True),
                     ("lines", False), ("all", False), ("event", ALL_EVENTS), ("ev_config", ""), ("fig_metric", "")):
            st.setdefault(k, v)
        tip = getattr(self, "_bind_hover_tip", lambda *a, **k: None)
        data: "dict[str, Any]" = {"sheets": None, "animals": None, "res": None}

        def _card(title: str, first: bool = False) -> "tuple[ctk.CTkFrame, ctk.CTkFrame]":
            card = ctk.CTkFrame(win, **card_style())
            card.pack(fill="x", padx=PAGE_PAD_X, pady=(PAGE_PAD_Y if first else 0, PAGE_GAP))
            body = ctk.CTkFrame(card, fg_color="transparent")
            body.pack(fill="both", expand=True, padx=CARD_PAD, pady=(CARD_PAD - 4, CARD_PAD))
            hdr = ctk.CTkFrame(body, fg_color="transparent")
            hdr.pack(fill="x", pady=(0, SPACE_S))
            ctk.CTkLabel(hdr, text=title.upper(), font=FONT_SECTION_HDR,
                         text_color=MUTED).pack(side="left")
            return hdr, body

        # ── Group table ──
        _hdr, body = _card("Group table", first=True)
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        lbl_file = ctk.CTkLabel(row, text="No group table open", font=FONT_SMALL,
                                text_color=MUTED, anchor="w", justify="left")
        lbl_msg = ctk.CTkLabel(body, text="", font=FONT_SMALL, text_color=MUTED, anchor="w",
                               justify="left")
        lbl_msg.pack(fill="x", pady=(SPACE_XS, 0))

        def _say(text: str, color: str = MUTED) -> None:
            lbl_msg.configure(text=text, text_color=color)

        # ── Animals ──
        _hdr_an, an_body = _card("Animals")
        an_grid = ctk.CTkFrame(an_body, fg_color="transparent")
        an_grid.pack(fill="x")
        ctk.CTkLabel(an_body, text=(
            "Group: type a name (Control, CNO…) or pick one; Sex comes from the file name or the "
            "Recording sheet. Unticked animals are left out of every analysis. Saved in the table."),
            font=FONT_HINT, text_color=MUTED, anchor="w", justify="left").pack(fill="x",
                                                                            pady=(SPACE_S, 0))

        # ── Compare ──
        _hdr_c, c_body = _card("Compare")
        r_by = ctk.CTkFrame(c_body, fg_color="transparent")
        r_by.pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkLabel(r_by, text="By", font=FONT_SMALL, text_color=MUTED, width=64,
                     anchor="w").pack(side="left")
        seg_by = ctk.CTkSegmentedButton(r_by, values=list(gc.BY_CHOICES), font=FONT_SMALL,
                                        selected_color=BLUE, command=lambda _v: _changed())
        seg_by.set(st["by"])
        seg_by.pack(side="left")
        r_an = ctk.CTkFrame(c_body, fg_color="transparent")
        r_an.pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkLabel(r_an, text="Analysis", font=FONT_SMALL, text_color=MUTED, width=64,
                     anchor="w").pack(side="left")
        seg_an = ctk.CTkSegmentedButton(r_an, values=list(gc.ANALYSES), font=FONT_SMALL,
                                        selected_color=BLUE, command=lambda _v: _analysis_changed())
        seg_an.set(st["analysis"])
        seg_an.pack(side="left")
        opts = ctk.CTkFrame(c_body, fg_color="transparent")
        opts.pack(fill="x")

        def _lbl(parent, text: str) -> None:
            ctk.CTkLabel(parent, text=text, font=FONT_SMALL, text_color=MUTED).pack(
                side="left", padx=(0, SPACE_XS))

        def _entry(parent, key: str, width: int = 56) -> ctk.CTkEntry:
            e = ctk.CTkEntry(parent, width=width, **input_style())
            e.insert(0, str(st[key]))
            e.pack(side="left", padx=(0, SPACE_M))
            e.bind("<Return>", lambda _e: _changed())
            e.bind("<FocusOut>", lambda _e: _changed())
            return e

        def _check(parent, text: str, key: str) -> ctk.CTkCheckBox:
            c = ctk.CTkCheckBox(parent, text=text, font=FONT_SMALL, text_color=MUTED,
                                fg_color=BLUE, checkmark_color="white", border_color=BORDER2,
                                width=16, command=lambda: _changed())
            if st[key]:
                c.select()
            c.pack(side="left", padx=(0, SPACE_M))
            return c

        def _combo(parent, key: str, width: int = 150) -> ctk.CTkComboBox:
            c = ctk.CTkComboBox(parent, width=width, values=[""], state="readonly",
                                command=lambda _v: _changed(), **combo_style())
            c.pack(side="left", padx=(0, SPACE_M))
            return c

        # Pre/Post segments
        o_seg = ctk.CTkFrame(opts, fg_color="transparent")
        _lbl(o_seg, "Compare")
        cb_a = _combo(o_seg, "a")
        _lbl(o_seg, "vs")
        cb_b = _combo(o_seg, "b")
        chk_all_seg = _check(o_seg, "All metrics", "all")
        # Time course
        o_tc = ctk.CTkFrame(opts, fg_color="transparent")
        _lbl(o_tc, "Time 0")
        cb_stim = _combo(o_tc, "stim", 170)
        _lbl(o_tc, "Metric")
        cb_tcm = ctk.CTkComboBox(o_tc, width=90, values=list(gd.TC_METRICS), state="readonly",
                                 command=lambda _v: _changed(), **combo_style())
        cb_tcm.set(st["tc_metric"])
        cb_tcm.pack(side="left", padx=(0, SPACE_M))
        _lbl(o_tc, "bins (s)")
        e_bin = _entry(o_tc, "bin", 50)
        _lbl(o_tc, "from")
        e_from = _entry(o_tc, "from", 50)
        _lbl(o_tc, "to")
        e_to = _entry(o_tc, "to", 50)
        _lbl(o_tc, "min · test window")
        e_w0 = _entry(o_tc, "w0", 44)
        _lbl(o_tc, "–")
        e_w1 = _entry(o_tc, "w1", 44)
        _lbl(o_tc, "min vs pre")
        chk_delta = _check(o_tc, "Δ from baseline", "delta")
        chk_lines = _check(o_tc, "Each animal", "lines")
        tip(e_bin, "Bin size in seconds (the table keeps 5 s sums: any multiple of 5 s is exact)")
        tip(chk_delta, "Each animal minus its own mean before the time 0 (from 'from' to 0)")
        # Whole recording
        o_wh = ctk.CTkFrame(opts, fg_color="transparent")
        chk_all_wh = _check(o_wh, "All metrics", "all")
        # Event-locked
        o_ev = ctk.CTkFrame(opts, fg_color="transparent")
        _lbl(o_ev, "Events")
        cb_ev = _combo(o_ev, "event", 170)
        _lbl(o_ev, "Configuration")
        cb_cfg = _combo(o_ev, "ev_config", 170)
        tip(cb_cfg, "The method and window the events were analysed with: one at a time, "
                    "the same event saved with two configurations is never counted twice")
        chk_all_ev = _check(o_ev, "All metrics", "all")
        frames = {gc.AN_SEGMENTS: o_seg, gc.AN_TIME: o_tc, gc.AN_WHOLE: o_wh, gc.AN_EVENTS: o_ev}

        # ── Figure ──
        hdr_f, f_body = _card("Figure")
        cb_fm = ctk.CTkComboBox(hdr_f, width=260, values=[""], state="readonly",
                                command=lambda _v: _draw(), **combo_style())
        plot_card = ctk.CTkFrame(f_body, fg_color=CARD, corner_radius=0, height=470)
        plot_card.pack(fill="x")
        plot_card.pack_propagate(False)
        slot = CanvasSlot(plot_card, FIG_SIZE[0], FIG_SIZE[1], toolbar=False,
                          placeholder="Open a group table")
        # Listed with the app's figures, so that a draw deferred while the tab
        # was hidden is made when it is shown (_flush_shown_slots).
        slots = getattr(self, "_slots", None)
        if isinstance(slots, dict):
            slots["group_tab"] = slot

        # ── Results ──
        hdr_r, r_body = _card("Results")
        _scale = THEME.font_scale
        _fsize = -int(round(12 * _scale))
        txt = tk.Text(r_body, bg=CARD, fg=TEXT, bd=0, highlightthickness=0, wrap="none",
                      font=(THEME.font_family, _fsize), height=2, cursor="arrow",
                      padx=SPACE_S, spacing1=3, spacing3=3)
        txt.tag_configure("head", font=(THEME.font_family, _fsize, "bold"), foreground=MUTED)
        txt.tag_configure("stripe", background=BG)
        txt.tag_configure("sig", foreground=GREEN)
        txt.configure(state="disabled")
        txt.pack(fill="x")
        lbl_notes = ctk.CTkLabel(r_body, text="", font=FONT_SMALL, text_color=ORANGE,
                                 anchor="w", justify="left")
        lbl_notes.pack(fill="x", pady=(SPACE_XS, 0))
        r_btns = ctk.CTkFrame(r_body, fg_color="transparent")
        r_btns.pack(fill="x", pady=(SPACE_S, 0))
        lbl_out = ctk.CTkLabel(r_body, text="", font=FONT_SMALL, text_color=MUTED,
                               anchor="w", justify="left")
        lbl_out.pack(fill="x", pady=(SPACE_XS, 0))

        wrapped: "list[ctk.CTkLabel]" = [lbl_notes, lbl_out, lbl_msg]

        def _wrap(e: "Any") -> None:
            for lbl in wrapped:
                lbl.configure(wraplength=max(200, e.width - 16))
        r_body.bind("<Configure>", _wrap)

        # ── loading ──

        def _refresh_animals() -> None:
            for w in an_grid.winfo_children():
                w.destroy()
            an = data["animals"]
            heads = ("Use", "Subject", "Sex", "Group", "File", "Data · QC")
            for c, h in enumerate(heads):
                ctk.CTkLabel(an_grid, text=h, font=FONT_SMALL, text_color=MUTED,
                             anchor="w").grid(row=0, column=c, sticky="w", padx=(0, SPACE_M))
            an_grid.grid_columnconfigure(4, weight=1)
            if an is None or an.empty:
                ctk.CTkLabel(an_grid, text="No animal yet: Add open recording…",
                             font=FONT_SMALL, text_color=MUTED).grid(row=1, column=0,
                                                                     columnspan=6, sticky="w")
                return
            groups = sorted({str(g) for g in an["group"] if str(g)}, key=str.casefold)
            for i, (_, r) in enumerate(an.iterrows(), start=1):
                key = (r["subject"], r["file"])
                var = tk.BooleanVar(value=bool(r["use"]))
                chk = ctk.CTkCheckBox(an_grid, text="", variable=var, width=16, fg_color=BLUE,
                                      checkmark_color="white", border_color=BORDER2,
                                      command=lambda k=key, v=var: _set_animal(k, "use", v.get()))
                chk.grid(row=i, column=0, sticky="w", padx=(0, SPACE_M), pady=1)
                ctk.CTkLabel(an_grid, text=str(r["subject"]), font=FONT_SMALL,
                             text_color=TEXT, anchor="w").grid(row=i, column=1, sticky="w",
                                                               padx=(0, SPACE_M))
                cb_sex = ctk.CTkComboBox(an_grid, width=64, values=["M", "F", ""],
                                         command=lambda v, k=key: _set_animal(k, "sex", v),
                                         **combo_style())
                cb_sex.set(str(r["sex"]))
                cb_sex.grid(row=i, column=2, sticky="w", padx=(0, SPACE_M), pady=1)
                cb_g = ctk.CTkComboBox(an_grid, width=130, values=groups or [""],
                                       command=lambda v, k=key: _set_animal(k, "group", v),
                                       **combo_style())
                cb_g.set(str(r["group"]))
                cb_g.grid(row=i, column=3, sticky="w", padx=(0, SPACE_M), pady=1)
                for ev in ("<Return>", "<FocusOut>"):
                    cb_g._entry.bind(ev, lambda _e, k=key, c=cb_g: _set_animal(k, "group", c.get()))
                ctk.CTkLabel(an_grid, text=str(r["file"]), font=FONT_SMALL, text_color=TEXT,
                             anchor="w").grid(row=i, column=4, sticky="w", padx=(0, SPACE_M))
                ctk.CTkLabel(an_grid, text=gd.data_text(r), font=FONT_SMALL,
                             text_color=GREEN if r["use"] else MUTED,
                             anchor="w").grid(row=i, column=5, sticky="w")

        def _set_animal(key: "tuple[str, str]", field_: str, value: Any) -> None:
            an = data["animals"]
            if an is None:
                return
            if field_ == "sex":
                value = normalise_sex(value)
            elif field_ == "group":
                value = str(value).strip()
            m = (an["subject"] == key[0]) & (an["file"] == key[1])
            if not m.any() or an.loc[m, field_].iloc[0] == value:
                return
            an.loc[m, field_] = value
            try:
                gd.save_animals(st["path"], an)
            except Exception as exc:
                _say(f"⚠  Not saved in {os.path.basename(st['path'])}: {exc} "
                     "(is it open in Excel?)", ORANGE)
            else:
                _say(f"✓  {key[0]}: {field_} saved", GREEN)
            _summary()
            if field_ == "group":
                _refresh_animals()
            _changed()

        def _summary() -> None:
            an = data["animals"]
            sheets = data["sheets"] or {}
            if an is None:
                lbl_file.configure(text="No group table open")
                return
            nm = int((an["sex"] == "M").sum())
            nf = int((an["sex"] == "F").sum())
            groups = an.loc[an["group"].astype(str) != "", "group"].value_counts()
            parts = [os.path.basename(st["path"]), f"{len(an)} animals", f"{nf} F / {nm} M"]
            if len(groups):
                parts.append(", ".join(f"{g} {n}" for g, n in groups.items()))
            kinds = [n for n, k in (("segments", gd.SHEET_SEGMENTS), ("whole", gd.SHEET_WHOLE),
                                    ("time course", gd.SHEET_TIME), ("events", gd.SHEET_EVENTS))
                     if k in sheets and len(sheets[k])]
            parts.append(", ".join(kinds) or "no data yet")
            lbl_file.configure(text="  ·  ".join(parts))

        def _load(path: str, quiet: bool = False) -> bool:
            try:
                sheets = gd.read_sheets(path)
            except Exception as exc:
                msg = f"{os.path.basename(path)}: {exc}"
                if quiet:
                    _say(f"⚠  {msg}", ORANGE)
                else:
                    messagebox.showerror("Group table", msg, parent=win)
                return False
            st["path"] = path
            save_last_table(path)
            data["sheets"] = sheets
            data["animals"] = gd.animals_frame(sheets)
            data["res"] = None
            _summary()
            _refresh_animals()
            _fill_choices()
            _set_tab_state(True)
            _changed()
            return True

        def _set_tab_state(loaded: bool) -> None:
            tabs = getattr(self, "tabs", None)
            if tabs is None:
                return
            from ecg.ui.freshness import NOT_COMPUTED, UP_TO_DATE
            try:
                tabs.set_state(TAB_GROUP, UP_TO_DATE if loaded else NOT_COMPUTED,
                               (f"group table {os.path.basename(st['path'])}",) if loaded
                               else ("no group table open",))
            except Exception:
                pass

        def _fill_choices() -> None:
            sheets = data["sheets"] or {}
            seg = sheets.get(gd.SHEET_SEGMENTS)
            tags = gc.segment_tags(seg) if seg is not None and len(seg) else []
            cb_a.configure(values=tags or [""])
            cb_b.configure(values=tags or [""])
            if tags:
                a = st["a"] if st["a"] in tags else tags[0]
                b = st["b"] if st["b"] in tags and st["b"] != a else \
                    next((t for t in tags if t != a), a)
                cb_a.set(a)
                cb_b.set(b)
            labels = gd.annotation_labels(sheets.get(gd.SHEET_ANNOT))
            cb_stim.configure(values=labels or [""])
            cb_stim.set(st["stim"] if st["stim"] in labels else (labels[0] if labels else ""))
            evs = [ALL_EVENTS] + gc.event_labels(sheets)
            cb_ev.configure(values=evs)
            cb_ev.set(st["event"] if st["event"] in evs else ALL_EVENTS)
            cfgs = gc.event_configs(sheets)
            cb_cfg.configure(values=cfgs or [""])
            cb_cfg.set(st["ev_config"] if st["ev_config"] in cfgs else (cfgs[0] if cfgs else ""))

        def _open() -> None:
            path = filedialog.askopenfilename(
                parent=win, title="Open a group table",
                filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
                initialdir=os.path.dirname(st["path"]) if st["path"] else None)
            if path:
                _load(path)

        def _reload() -> None:
            if st["path"]:
                _load(st["path"])

        def _add_open() -> None:
            # Only the table shown: a remembered one that did not open (moved,
            # deleted) is not written to without asking.
            path = st["path"] if data["sheets"] is not None else ""
            if not path:
                path = filedialog.asksaveasfilename(
                    parent=win, title="Group table: add the recording open (created if new)",
                    defaultextension=".xlsx", initialfile="group_table.xlsx",
                    confirmoverwrite=False,
                    filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")])
                if not path:
                    return
            try:
                n, replaced, what = self._add_open_recording_to_group_table(path)
            except Exception as exc:
                messagebox.showerror("Group table", str(exc), parent=win)
                return
            if not _load(path):
                return
            _say(f"✓  {what} — {'updated in' if replaced else 'added to'} "
                 f"{os.path.basename(path)} ({n} animals)", GREEN)

        for text, cmd, tiptext in (
                ("📂  Open…", _open, "Open a group table (.xlsx): one animal per recording"),
                ("Reload", _reload, "Read the table again (rows added since count)"),
                ("➕  Add open recording…", _add_open,
                 "Write the recording open to the table: its whole-recording metrics and time "
                 "course (after Analyze), its annotations, its event-locked results (saved to "
                 "the registry) and its Compare segments result. A recording already in it is "
                 "replaced; its group stays.")):
            b = ctk.CTkButton(row, text=text, width=0, command=cmd, **button_style("secondary"))
            b.pack(side="left", padx=(0, SPACE_S))
            tip(b, tiptext)
        lbl_file.pack(side="left", fill="x", expand=True, padx=(SPACE_S, 0))

        # ── running ──

        def _read_options() -> None:
            st["by"] = seg_by.get()
            st["analysis"] = seg_an.get()
            st["a"], st["b"] = cb_a.get(), cb_b.get()
            st["stim"], st["tc_metric"] = cb_stim.get(), cb_tcm.get()
            for k, e in (("bin", e_bin), ("from", e_from), ("to", e_to), ("w0", e_w0),
                         ("w1", e_w1)):
                st[k] = e.get().strip()
            st["delta"] = bool(chk_delta.get())
            st["lines"] = bool(chk_lines.get())
            st["event"] = cb_ev.get()
            st["ev_config"] = cb_cfg.get()
            src = {gc.AN_SEGMENTS: chk_all_seg, gc.AN_WHOLE: chk_all_wh,
                   gc.AN_EVENTS: chk_all_ev}.get(st["analysis"])
            if src is not None:
                st["all"] = bool(src.get())

        def _sync_all_checks() -> None:
            for c in (chk_all_seg, chk_all_wh, chk_all_ev):
                c.select() if st["all"] else c.deselect()

        def _analysis_changed() -> None:
            st["analysis"] = seg_an.get()
            _sync_all_checks()
            _show_options()
            _changed()

        def _show_options() -> None:
            for name, fr in frames.items():
                if name == seg_an.get():
                    fr.pack(fill="x")
                else:
                    fr.pack_forget()

        def _run() -> "Optional[gc.GroupResult]":
            sheets, an = data["sheets"], data["animals"]
            if sheets is None or an is None:
                return None
            mode = st["analysis"]
            metrics = "all" if st["all"] else None
            if mode == gc.AN_SEGMENTS:
                return gc.analyse_segments(sheets, an, st["by"], st["a"], st["b"], metrics)
            if mode == gc.AN_WHOLE:
                return gc.analyse_whole(sheets, an, st["by"], metrics)
            if mode == gc.AN_EVENTS:
                ev = "" if st["event"] == ALL_EVENTS else st["event"]
                return gc.analyse_events(sheets, an, st["by"], ev, metrics,
                                         config=st["ev_config"] or None)
            if not st["stim"]:
                raise ValueError("No annotation in the table to align on: annotate the "
                                 "stimulation in each recording, then Add open recording.")
            return gc.analyse_time(
                sheets, an, st["by"], st["stim"], st["tc_metric"],
                bin_s=max(5.0, _float(e_bin, 60.0)), start_min=_float(e_from, -10.0),
                end_min=_float(e_to, 30.0),
                window_min=(_float(e_w0, 0.0), _float(e_w1, 5.0)), delta=st["delta"])

        def _changed() -> None:
            _read_options()
            if data["sheets"] is None:
                return
            try:
                res = _run()
            except Exception as exc:
                data["res"] = None
                _show_result(None, str(exc))
                return
            data["res"] = res
            _show_result(res)

        def _show_result(res: "Optional[gc.GroupResult]", error: str = "") -> None:
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            for b in (btn_stats, btn_fig, btn_prism):
                b.configure(state="normal" if res is not None else "disabled")
            if res is None:
                txt.insert("end", error, "head")
                txt.configure(height=2, state="disabled")
                lbl_notes.configure(text="")
                slot.update(lambda fig: fig.text(0.5, 0.5, error or "No result",
                                                 ha="center", va="center", color="#888"))
                return
            d = res.display
            cols = [c for c in d.columns if c != "_p"]
            import tkinter.font as tkfont
            f_body = tkfont.Font(font=txt.cget("font"))
            f_head = tkfont.Font(font=(THEME.font_family, _fsize, "bold"))
            width = {c: max([f_head.measure(str(c))] + [f_body.measure(str(v)) for v in d[c]])
                     for c in cols}
            x, stops = 0, []
            for c in cols[:-1]:
                x += width[c] + int(20 * _scale)
                stops += [x, "left"]
            txt.configure(tabs=tuple(stops))
            txt.insert("end", "\t".join(cols) + "\n", "head")
            for i, (_, r) in enumerate(d.iterrows()):
                tags = ("stripe",) if i % 2 else ()
                p = r.get("_p")
                if isinstance(p, float) and math.isfinite(p) and p < 0.05:
                    tags += ("sig",)
                txt.insert("end", "\t".join(str(r[c]) for c in cols) + "\n", tags)
            txt.configure(height=min(24, len(d) + 2), state="disabled")
            lbl_notes.configure(text="\n".join(f"⚠  {n}" for n in res.notes))
            fm = []
            if res.figure.get("kind") in ("paired", "dots"):
                fm = list(res.figure.get("values", {}))
            if fm:
                cb_fm.configure(values=fm)
                cb_fm.set(st["fig_metric"] if st["fig_metric"] in fm else fm[0])
                if not cb_fm.winfo_manager():
                    cb_fm.pack(side="right")
            elif cb_fm.winfo_manager():
                cb_fm.pack_forget()
            _draw()

        def _draw() -> None:
            res = data["res"]
            if res is None:
                return
            if cb_fm.winfo_manager():
                st["fig_metric"] = cb_fm.get()
            slot.update(lambda fig: draw_group_tab_figure(
                fig, res, metric=st["fig_metric"] or None, show_animals=st["lines"]))

        # ── exports ──

        def _base() -> str:
            return os.path.splitext(os.path.basename(st["path"] or "group"))[0]

        def _said(text: str, ok: bool = True) -> None:
            lbl_out.configure(text=("✓  " if ok else "⚠  ") + text,
                              text_color=GREEN if ok else RED)

        def _save_stats() -> None:
            res = data["res"]
            if res is None:
                return
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the group statistics", defaultextension=".xlsx",
                initialfile=f"{_base()}_{res.analysis.split()[0].lower()}_stats.xlsx",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")])
            if not path:
                return
            try:
                gc.write_results(path, res, os.path.basename(st["path"]))
            except Exception as exc:
                messagebox.showerror("Group statistics", str(exc), parent=win)
                return
            _said(f"statistics → {os.path.basename(path)}")

        def _save_figure() -> None:
            res = data["res"]
            if res is None:
                return
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the group figure", defaultextension=".png",
                initialfile=f"{_base()}_{res.analysis.split()[0].lower()}.png",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("PNG image (300 dpi)", "*.png"), ("PDF (vector)", "*.pdf"),
                           ("SVG (vector)", "*.svg"), ("All files", "*.*")])
            if not path:
                return
            fig = Figure(figsize=FIG_SIZE, dpi=300, facecolor="white", layout="constrained")
            try:
                draw_group_tab_figure(fig, res, metric=st["fig_metric"] or None,
                                      show_animals=st["lines"])
                fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
            except Exception as exc:
                messagebox.showerror("Group figure", str(exc), parent=win)
                return
            finally:
                plt.close(fig)
            _said(f"figure → {os.path.basename(path)}")

        def _notes(res: "gc.GroupResult") -> "list[tuple[str, Any]]":
            return ([("Group table", os.path.basename(st["path"]))] + list(res.settings)
                    + [("Groups", ", ".join(f"{g}" for g in res.groups)),
                       ("Animals", res.n_animals)] + [("Note", n) for n in res.notes])

        def _prism_new() -> None:
            res = data["res"]
            if res is None:
                return
            path = filedialog.asksaveasfilename(
                parent=win, title="Export to GraphPad Prism", defaultextension=".pzfx",
                initialfile=f"{_base()}_{res.analysis.split()[0].lower()}.pzfx",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("GraphPad Prism", "*.pzfx"), ("All files", "*.*")])
            if not path:
                return
            try:
                n = write_metric_pzfx(res.prism, path, notes=_notes(res))
            except Exception as exc:
                log.exception("group tab Prism export")
                messagebox.showerror("Prism export", str(exc), parent=win)
                return
            _said(f"Prism: {n} tables → {os.path.basename(path)}")

        def _prism_fill() -> None:
            from ecg.ui.compare_segments import ask_filled_path, ask_prism_template
            res = data["res"]
            if res is None:
                return
            tpl = ask_prism_template(self, win)
            if not tpl:
                return
            out = ask_filled_path(win, tpl, _base())
            if not out:
                return
            try:
                report = fill_template(tpl, res.prism, out,
                                       notes=_notes(res) + [("Prism template",
                                                             os.path.basename(tpl))])
            except Exception as exc:
                if not isinstance(exc, PrismTemplateError):
                    log.exception("group tab Prism template")
                messagebox.showerror("Prism template", str(exc), parent=win)
                return
            _said(f"Prism template {os.path.basename(tpl)}: {report.summary()} → "
                  f"{os.path.basename(out)}", ok=not report.warn)

        def _prism() -> None:
            from ecg.ui.compare_segments import prism_menu
            prism_menu(win, btn_prism, _prism_new, _prism_fill)

        btn_stats = ctk.CTkButton(r_btns, text="📊  Statistics… (Excel)", width=0,
                                  command=_save_stats, state="disabled",
                                  **button_style("secondary"))
        btn_fig = ctk.CTkButton(r_btns, text="Figure…", width=0, command=_save_figure,
                                state="disabled", **button_style("secondary"))
        btn_prism = ctk.CTkButton(r_btns, text="Prism… (new file / template)", width=0,
                                  command=_prism, state="disabled", **button_style("secondary"))
        for b in (btn_stats, btn_fig, btn_prism):
            b.pack(side="left", padx=(0, SPACE_S))
        tip(btn_stats, "Excel: the results table, every number behind it (means, SEM, n, "
                       "tests, Holm), the ANOVA, the per-animal values and the settings")
        tip(btn_prism, "GraphPad Prism: per group one table per metric (one row per animal), "
                       "the changes by group, the time course as XY tables; new file or "
                       "poured into your template")
        lbl_hint = ctk.CTkLabel(r_body, text=(
            "The unit is the animal. Within a group: Wilcoxon signed-rank (paired t beside it); "
            "between groups: Mann-Whitney / Kruskal-Wallis on the change (Welch / ANOVA in the "
            "workbook); RM ANOVA: mixed, Greenhouse-Geisser for time. Holm over the metrics "
            "listed: pick your primary metric in advance."), font=FONT_HINT, text_color=MUTED,
            anchor="w", justify="left")
        lbl_hint.pack(fill="x", pady=(SPACE_S, 0))
        wrapped.append(lbl_hint)

        _sync_all_checks()
        _show_options()
        _refresh_animals()
        self._grouptab_reload = _reload
        if st["path"] and os.path.exists(st["path"]):
            _load(st["path"], quiet=True)
