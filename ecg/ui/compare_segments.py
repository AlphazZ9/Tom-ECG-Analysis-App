# -*- coding: utf-8 -*-
"""
ecg.ui.compare_segments
-----------------------
ECGApp Compare segments tab: 1 to 6 windows of the recording, typed in,
their metric table, quality checks, RR tachograms (one panel per pair in
Pairs mode) or Poincaré plots, the export and the group table. Built in the
tab when it opens, rebuilt when the recording or its peaks have changed
since. Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from tkinter import filedialog, messagebox, simpledialog
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
from ecg.io.subject_name import normalise_sex
from ecg.io.group_stats import (
    DEFAULT_METRICS_NOTE, HOLM_ALL, all_pairs_stats, check_rows,
    paired_stats, read_group_table, segment_tags, smallest_wilcoxon_p,
    write_all_pairs_report, write_group_report,
)
from ecg.ui.group_figure import MAIN_METRIC, draw_all_pairs_figure, draw_group_figure

# Group card analysis modes.
MODE_ONE = "One pair (A vs B)"
MODE_ALL = "All pairs (B vs A, D vs C, F vs E)"


def all_pairs_figsize(n_pairs: int) -> "tuple[float, float]":
    """The saved all-pairs figure: about 3.3 in per pair panel."""
    return (max(5.0, 3.3 * max(1, n_pairs)), 4.6)
from ecg.io.segment_export import (
    ComparisonResult, comparison_tsv, pzfx_notes, write_csv, write_pzfx, write_xlsx,
)
from ecg.io import prism_template
from ecg.io.prism_template import (
    PrismTemplateError, fill_template, group_tables, left_out_notes, pair_tables, sex_notes,
    recording_tables, write_metric_pzfx,
)
from ecg.core import segment_plan
from ecg.core.segment_plan import (
    MAX_SEGMENTS, PAIRS, PAIRWISE, apply_config, comparators_caption, comparators_reference,
    comparison_pairs, config_from_rows, default_stim, delete_template, marker_choices,
    quality_flags, reference_comparators, save_template, unique_tags, user_templates,
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

_VIEWS = ("rr", "poincare")


_NO_CONFIG = "(no configuration saved)"
_MARKER_PROMPT = "Stim marker ▾"


def _templates_path() -> "Any":
    """Where the configurations are kept (segment_plan.TEMPLATES_PATH, read
    at call time so a test can point it elsewhere)."""
    return segment_plan.TEMPLATES_PATH


def ask_config_name(parent: "Any", initial: str = "") -> "Optional[str]":
    """The name of a configuration to save (None: cancelled)."""
    return simpledialog.askstring("Save configuration",
                                  "Name of this configuration:",
                                  initialvalue=initial, parent=parent)


# ── GraphPad Prism: a new file, or a template filled ─────────────────────

PRISM_NEW = "New Prism file…"
PRISM_FILL = "Fill a Prism template…"
PRISM_TIP = ("Fill a Prism template…: make your graphs and tests once in Prism on an "
             "exported file and save it as .pzfx; the app then pours new data into it "
             "(tables matched by title) and Prism redraws the graphs and recomputes the tests.")


def prism_tip(app: "Any", widget: "Any", text: str) -> None:
    """The Prism… buttons' explanation, too long for the status bar: in the
    floating tooltip, the template workflow first."""
    bind = getattr(app, "_bind_float_tip", None) or app._bind_hover_tip
    bind(widget, PRISM_TIP + "\n\n" + text)


def prism_menu(parent: "Any", button: "Any", new_cmd: "Any", fill_cmd: "Any") -> "tk.Menu":
    """The Prism… menu under *button*: a new file or a template filled."""
    menu = tk.Menu(parent, tearoff=0)
    menu.add_command(label=PRISM_NEW, command=new_cmd)
    menu.add_command(label=PRISM_FILL, command=fill_cmd)
    try:
        menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
    finally:
        menu.grab_release()
    return menu


def last_prism_template(app: "Any") -> str:
    """The template filled last: this session's, else the one remembered."""
    path = getattr(app, "_prism_template_path", "") or ""
    if path and os.path.exists(path):
        return path
    return prism_template.load_last_template(prism_template.PREFS_PATH)


def ask_prism_template(app: "Any", parent: "Any") -> "Optional[str]":
    """The template to fill (a .pzfx saved from Prism), the last one offered."""
    last = last_prism_template(app)
    path = filedialog.askopenfilename(
        parent=parent, title="Choose the Prism template (.pzfx saved from Prism)",
        initialdir=os.path.dirname(last) if last else None,
        initialfile=os.path.basename(last) if last else None,
        filetypes=[("GraphPad Prism XML", "*.pzfx"), ("All files", "*.*")])
    if not path:
        return None
    app._prism_template_path = path
    prism_template.save_last_template(path, prism_template.PREFS_PATH)
    return path


def ask_filled_path(parent: "Any", template: str, base: str) -> "Optional[str]":
    """Where to write the filled template: '<base>_<template>.pzfx' next to
    the template by default, never the template itself."""
    stem = os.path.splitext(os.path.basename(template))[0]
    while True:
        path = filedialog.asksaveasfilename(
            parent=parent, title="Save the filled Prism file (the template is not changed)",
            defaultextension=".pzfx", initialdir=os.path.dirname(template) or None,
            initialfile=f"{base}_{stem}.pzfx",
            filetypes=[("GraphPad Prism", "*.pzfx"), ("All files", "*.*")])
        if not path:
            return None
        if os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(template)):
            return path
        messagebox.showwarning("Prism template", "Choose another name: the template itself "
                               "is never overwritten.", parent=parent)


def _recording_key(app: "Any") -> "Optional[tuple]":
    """The recording a Compare segments setup belongs to (its file and
    channel, and the filtered signal array, not its peaks: rows typed stay
    with the recording when its peaks are edited). The file and channel
    are part of it because CPython reuses a freed array's id: two files of
    the same length could otherwise share a key. None without a signal."""
    sig = getattr(app, "_signal_flt", None)
    if sig is None:
        return None
    signal = getattr(app, "signal", None)
    path = getattr(signal, "filepath", None) or ""
    channel = getattr(signal, "channel", None)
    return (str(path), str(channel), id(sig), len(sig))


def _fmt_s(v: "Any") -> str:
    """A time in seconds for an entry or a message: millisecond precision,
    trailing zeros dropped ("1808.183", "2000", "0.5"). Unlike ``:g`` it
    keeps every digit before the point, so 12345.678 stays 12345.678."""
    s = f"{float(v):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _finite(v: "Any") -> "Optional[float]":
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _cell(v: "Any") -> "Any":
    """A From / To cell: its number, else the text typed ("" for none)."""
    f = _finite(v)
    return f if f is not None else ("" if v is None else str(v))


def normalize_compare_setup(raw: "Any") -> "Optional[dict]":
    """A Compare segments setup as saved in a session (or snapshot before a
    UI rebuild), checked: {"rows": [[label, from, to], ...] (1 to
    MAX_SEGMENTS, from / to a number or, typed but not valid, the text),
    "block_s" (text), "ref" (a segment index, PAIRWISE or PAIRS), "view"
    ("rr" / "poincare"), "jitter" (bool), "run": the windows and block
    length of the comparison shown ({"rows": [[label, lo, hi]...],
    "block_s"}) or None, "ran": whether there is one, "stim": the Stim at
    entry, text, "" when empty}. None when *raw* is not one (an older session: the
    tab keeps its defaults)."""
    if not isinstance(raw, dict):
        return None
    rows = []
    for r in raw.get("rows") or []:
        if isinstance(r, (list, tuple)) and len(r) == 3:
            rows.append(["" if r[0] is None else str(r[0]), _cell(r[1]), _cell(r[2])])
    rows = rows[:MAX_SEGMENTS]
    ref = raw.get("ref", 0)
    if isinstance(ref, bool) or not (
            (isinstance(ref, int) and 0 <= ref < MAX_SEGMENTS) or ref in (PAIRWISE, PAIRS)):
        ref = 0
    view = raw.get("view") if raw.get("view") in _VIEWS else "rr"
    blk = raw.get("block_s")
    if isinstance(blk, str):
        block_s = blk
    elif _finite(blk) is not None:
        block_s = _fmt_s(_finite(blk))
    else:
        block_s = f"{MouseECG.SEGMENT_BLOCK_S:g}"
    run = None
    rr = raw.get("run")
    if isinstance(rr, dict):
        r_rows: "list[list]" = []
        for r in rr.get("rows") or []:
            lo = _finite(r[1]) if isinstance(r, (list, tuple)) and len(r) == 3 else None
            hi = _finite(r[2]) if lo is not None else None
            if lo is None or hi is None or lo >= hi:
                r_rows = []
                break
            r_rows.append([str(r[0]), lo, hi])
        r_blk = _finite(rr.get("block_s"))
        if r_rows and len(r_rows) <= MAX_SEGMENTS and r_blk is not None and r_blk > 0:
            run = {"rows": r_rows, "block_s": r_blk}
    if not rows and run is None:
        return None
    if not rows:
        rows = [list(r) for r in run["rows"]]
    st_raw = raw.get("stim")
    if isinstance(st_raw, str):
        stim = st_raw.strip()
    else:
        stim = _fmt_s(_finite(st_raw)) if _finite(st_raw) is not None else ""
    return {"rows": rows, "block_s": block_s, "ref": ref, "view": view,
            "jitter": bool(raw.get("jitter", True)), "ran": run is not None, "run": run,
            "stim": stim}

# Plot card of the comparison (CTk units): the tachogram view's height, and
# the height of one row of Poincaré panels -- every panel keeps that size
# however many segments are compared (more rows instead of smaller panels).
_PLOT_CARD_H = 460
_YBAR_H = 26                       # CanvasSlot's Y-scale bar above the plot
_POINCARE_ROW_H = _PLOT_CARD_H - _YBAR_H
_POINCARE_ROW_GAP = 48              # between two rows of Poincaré panels
# Narrowest column that still gives a panel its full row-height size
# (the square plot plus its Y label and ticks).
_POINCARE_COL_W = 480
# One panel of the per-pair tachograms (Pairs mode), CTk units.
_TACHO_ROW_H = 260


def tachogram_groups(n: int, comparators: "Optional[list[Optional[int]]]" = None
                     ) -> "list[tuple[int, ...]]":
    """The segments each RR tachogram panel shows: in Pairs mode
    (comparators_reference == PAIRS) one panel per pair, (comparator,
    segment), and an odd segment left alone in its own; otherwise one panel
    with every segment superimposed."""
    if n <= 0:
        return []
    if comparators is not None and len(comparators) == n \
            and comparators_reference(comparators) == PAIRS:
        groups: "list[tuple[int, ...]]" = [(j, k) for j, k in comparison_pairs(comparators)]
        paired = {i for g in groups for i in g}
        groups += [(i,) for i in range(n) if i not in paired]
        return sorted(groups, key=lambda g: g[0])
    return [tuple(range(n))]


def draw_rr_tachograms(fig, segs: "list[dict]", colors: "list[str]",
                       comparators: "Optional[list[Optional[int]]]" = None,
                       names: "Optional[list[str]]" = None) -> list:
    """The RR tachograms of *segs* on *fig*: all superimposed on one axes,
    or, in Pairs mode, one axes per pair (the two superimposed, titled
    "B vs A") stacked vertically, sharing the RR scale so the pairs compare
    at a glance (the Y-scale bar and the wheel zoom then act on every
    panel). Time is within each segment. Returns the axes."""
    names = list(names) if names is not None else [str(s.get("label", "")) for s in segs]
    groups = tachogram_groups(len(segs), comparators)
    if not groups:
        return []
    axes = list(np.atleast_1d(fig.subplots(len(groups), 1, sharey=True, squeeze=False)).ravel())
    for ax, grp in zip(axes, groups):
        style_axes(ax)
        drawn = False
        for i in grp:
            s, c = segs[i], colors[i % len(colors)]
            t, r = s["t_rr"], s["rr_ms"]
            if len(t) and len(r):
                ax.plot(t - t[0], r, color=c, lw=1.0, alpha=0.85, label=names[i])
                ax.axhline(float(np.mean(r)), color=c, lw=1.0, ls="--", alpha=0.5)
                drawn = True
        ax.set_ylabel("RR (ms)", fontsize=9)
        if len(groups) == 1:
            ax.set_title("RR intervals — segments superimposed", loc="left", fontsize=9)
        elif len(grp) == 2:
            ax.set_title(f"{names[grp[1]]} vs {names[grp[0]]}", loc="left", fontsize=9)
        else:
            ax.set_title(names[grp[0]], loc="left", fontsize=9)
        if drawn:
            ax.legend(framealpha=0, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Time within segment (s)", fontsize=9)
    return axes


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
        """What the tab was built from: the recording and its peaks (the
        annotations / events no longer place anything in it). None without
        peaks."""
        if self._signal_flt is None or self._rpeaks_ok is None or self._fs is None:
            return None
        rp = self._rpeaks_ok
        return (id(self._signal_flt), len(self._signal_flt), float(self._fs),
                len(rp), int(np.sum(rp)) if len(rp) else 0)

    def _current_comparison(self: ECGApp) -> "Optional[ComparisonResult]":
        """The last Compare segments result, None when there is none or the
        recording or its peaks changed since (the main exports
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
        # Built already for no recording: opening the tab again must not
        # destroy and rebuild it (a visible flash, and the group card's
        # table read again each time).
        if (sig is None and getattr(self, "_compare_built_empty", False)
                and self._compare_built_for is None and holder.winfo_children()):
            return
        # The rows typed for this recording survive a rebuild for its new
        # peaks (not the comparison: it rests on the old ones), unless a
        # session / UI rebuild left a setup to apply instead.
        key = _recording_key(self)
        getter = getattr(self, "_compare_setup_getter", None)
        old_key = getattr(self, "_compare_setup_key", None)
        self._compare_setup_getter = None
        pend = getattr(self, "_compare_pending_setup", None)
        if (getter is not None and sig is not None and key is not None and old_key == key
                and (pend is None or pend[0] != key)):
            try:
                carried = getter()
                carried["run"] = None
                self._compare_pending_setup = (key, carried)
            except Exception as exc:        # widgets already destroyed (UI rebuild)
                log.debug("compare rows not carried over: %s", exc)
        for child in holder.winfo_children():
            child.destroy()
        self._compare_built_for = sig
        self._compare_built_empty = sig is None
        if sig is None:
            ctk.CTkLabel(holder, text="Open a file and detect the peaks to compare segments "
                                      "(the group card below works without one).",
                         font=FONT_LABEL, text_color=MUTED).pack(pady=(60, 40))
        else:
            self._build_compare_segments(holder)
        self._build_group_card(holder)

    def _compare_setup_snapshot(self: ECGApp) -> "Optional[dict]":
        """The Compare segments setup of the current recording, for Save
        Session and a UI rebuild: the one waiting to be applied (a restored
        session, the tab not built since), else the tab's as typed. None
        when the tab holds nothing for this recording."""
        key = _recording_key(self)
        if key is None:
            return None
        pend = getattr(self, "_compare_pending_setup", None)
        if pend is not None and pend[0] == key:
            return normalize_compare_setup(pend[1])
        getter = getattr(self, "_compare_setup_getter", None)
        if getter is not None and getattr(self, "_compare_setup_key", None) == key:
            try:
                return normalize_compare_setup(getter())
            except Exception as exc:
                log.debug("compare setup snapshot failed: %s", exc, exc_info=True)
        return None

    def _restore_compare_setup(self: ECGApp, raw: "Any") -> None:
        """Apply a saved setup (rows, block, reference, view, jitter) to the
        tab, re-running the comparison it showed: now when the tab is open,
        else when it is next built for this recording. *raw* None or invalid
        (an older session): nothing is restored."""
        setup = normalize_compare_setup(raw)
        key = _recording_key(self)
        if setup is None or key is None:
            self._compare_pending_setup = None
            return
        self._compare_pending_setup = (key, setup)
        self._compare_built_for = None          # rebuilt for it, open or next opened
        try:
            is_open = self.tabs.get() == TAB_COMPARE
        except Exception:
            is_open = False
        if is_open:
            self._refresh_compare_tab()

    def _build_compare_segments(self: ECGApp, win) -> None:
        """The comparison inside *win* (the tab's frame): 1 to MAX_SEGMENTS
        windows typed in, their metric table (Δ % against a chosen reference segment, or each against
        the previous one; the first by default), RR tachograms or Poincaré plots, the
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

        def _sex() -> str:
            """The Sex entry as it is now (M / F / '')."""
            ent = getattr(self, "ent_sex", None)
            return normalise_sex(ent.get()) if ent is not None else ""

        def _section(title: str, note: str = "", first: bool = False
                     ) -> "tuple[ctk.CTkFrame, ctk.CTkFrame]":
            """A white card titled *title* (the pages' card: 1 px border,
            16 px inside), across the page (*first*: the top one, under the
            page's top padding); returns its header, for what sits by the
            title, and its body."""
            card = ctk.CTkFrame(win, **card_style())
            card.pack(fill="x", padx=PAGE_PAD_X, pady=(PAGE_PAD_Y if first else 0, PAGE_GAP))
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

        # ── Segments: 1 to MAX_SEGMENTS rows ──────────────────────────────
        _h, seg_card = _section("Segments", f"Δ % against the first unless chosen in Results"
                                            f"  ·  {sub}  ·  "
                                            f"{dur:.1f} s  ·  {len(rp)} peaks", first=True)
        # ── Configurations relative to the stimulus (one row) ─────────────
        # Saved from the rows of one animal as offsets from its stim time,
        # applied to another at its own stim time (segment_plan).
        cfg_bar = ctk.CTkFrame(seg_card, fg_color="transparent")
        cfg_bar.pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkLabel(cfg_bar, text="Stim at (s):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        ent_stim = ctk.CTkEntry(cfg_bar, width=80, placeholder_text="e.g. 1803",
                                **input_style())
        ent_stim.pack(side="left", padx=(4, SPACE_S))
        _markers = getattr(getattr(self, "signal", None), "markers", None) or {}
        _marker_opts: "dict[str, float]" = dict(marker_choices(_markers))
        cb_marker = None
        if _marker_opts:
            cb_marker = ctk.CTkComboBox(cfg_bar, width=240, state="readonly",
                                        values=list(_marker_opts), **combo_style())
            cb_marker.set(_MARKER_PROMPT)
            cb_marker.pack(side="left", padx=(0, SPACE_M))
            tip(cb_marker, "Markers of this Spike2 file (the first pulse of each train, ×n: its "
                           "number of pulses): "
                           "choosing one puts its time in Stim at.")
        ctk.CTkLabel(cfg_bar, text="Configuration:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(SPACE_S, 4))
        cb_cfg = ctk.CTkComboBox(cfg_bar, width=220, state="readonly", values=[_NO_CONFIG],
                                 **combo_style())
        cb_cfg.pack(side="left", padx=(0, SPACE_S))
        btn_cfg_apply = ctk.CTkButton(cfg_bar, text="Apply", width=0,
                                      **button_style("secondary"))
        btn_cfg_apply.pack(side="left", padx=(0, SPACE_XS))
        btn_cfg_save = ctk.CTkButton(cfg_bar, text="Save as…", width=0,
                                     **button_style("secondary"))
        btn_cfg_save.pack(side="left", padx=(0, SPACE_XS))
        btn_cfg_del = ctk.CTkButton(cfg_bar, text="Delete", width=0,
                                    **button_style("secondary"))
        btn_cfg_del.pack(side="left")
        lbl_cfg = ctk.CTkLabel(cfg_bar, text="", font=FONT_SMALL, text_color=MUTED,
                               anchor="w", justify="left")
        lbl_cfg.pack(side="left", fill="x", expand=True, padx=(SPACE_M, 0))
        tip(ent_stim, "Time of the stimulus in this recording (s). A configuration is "
                      "saved as offsets from it and applied at the stim time typed here.")
        tip(cb_cfg, "Your saved configurations: the windows as offsets from the stimulus.")
        tip(btn_cfg_apply, "Fill the rows with the chosen configuration, placed at Stim at.")
        tip(btn_cfg_save, "Save the rows below as a configuration, relative to Stim at.")
        tip(btn_cfg_del, "Delete the chosen configuration.")
        _cfgs: "dict[str, Any]" = {}

        def _cfg_say(text: str, color: "Any" = None) -> None:
            lbl_cfg.configure(text=text, text_color=MUTED if color is None else color)

        def _cfg_refresh(select: "Optional[str]" = None) -> None:
            """The saved configurations in the selector (*select* chosen,
            else the one shown when it still exists, else the first)."""
            try:
                temps = user_templates(_templates_path())
            except Exception as exc:
                log.warning("configurations not read: %s", exc)
                temps = []
            _cfgs.clear()
            _cfgs.update((t.name, t) for t in temps)
            names = list(_cfgs) or [_NO_CONFIG]
            cb_cfg.configure(values=names)
            shown = cb_cfg.get()
            cur = select if select in _cfgs else (shown if shown in _cfgs else names[0])
            cb_cfg.set(cur)
            for b in (btn_cfg_apply, btn_cfg_del):
                b.configure(state="normal" if _cfgs else "disabled")

        def _stim_value() -> "Optional[float]":
            """Stim at as a number (None: empty); ValueError when not one."""
            text = ent_stim.get().strip().replace(",", ".")
            if not text:
                return None
            v = float(text)
            if not np.isfinite(v):
                raise ValueError(text)
            return v

        def _set_stim(v: float) -> None:
            ent_stim.delete(0, "end")
            ent_stim.insert(0, _fmt_s(v))

        def _on_marker_pick(choice: str) -> None:
            t = _marker_opts.get(choice)
            if t is None:
                return
            _set_stim(t)
            _cfg_say(f"Stim at {_fmt_s(t)} s ({choice.split('  ')[0]})")

        if cb_marker is not None:
            cb_marker.configure(command=_on_marker_pick)

        def _cfg_apply() -> None:
            t = _cfgs.get(cb_cfg.get())
            if t is None:
                _cfg_say("⚠  Choose a saved configuration", RED)
                return
            try:
                stim = _stim_value()
            except ValueError:
                _cfg_say("⚠  Stim at: not a number", RED)
                return
            if stim is None:
                _cfg_say("⚠  Type the stim time in Stim at (s) first", RED)
                return
            res_w = apply_config(t, stim, dur)
            _set_rows([(w.label, w.lo, w.hi) for w in res_w])
            out = [w.label for w in res_w if w.clamped and not w.hi > w.lo]
            cut = [w.label for w in res_w if w.clamped and w.hi > w.lo]
            if out or cut:
                parts = []
                if out:
                    parts.append("outside: " + ", ".join(out))
                if cut:
                    parts.append("cut to fit: " + ", ".join(cut))
                _cfg_say(f"⚠  '{t.name}' at {_fmt_s(stim)} s, recording {dur:.1f} s — "
                         + "; ".join(parts), RED if out else ORANGE)
            else:
                _cfg_say(f"✓  '{t.name}' applied at {_fmt_s(stim)} s", GREEN)

        def _cfg_save() -> None:
            try:
                rows = _read_rows()
            except ValueError as exc:
                _cfg_say(f"⚠  {exc}", RED)
                return
            bad = [i + 1 for i, (_l, lo, hi) in enumerate(rows) if not hi > lo]
            if bad:
                _cfg_say(f"⚠  Segment {bad[0]}: start ≥ end", RED)
                return
            try:
                stim = _stim_value()
            except ValueError:
                _cfg_say("⚠  Stim at: not a number", RED)
                return
            if stim is None:
                guess = default_stim(rows)
                if guess is None:
                    _cfg_say("⚠  Type the stim time in Stim at (s): the windows are "
                             "saved relative to it", RED)
                    return
                post = next(lb for lb, _lo, _hi in rows
                            if lb.strip().lower().startswith("post"))
                if not messagebox.askyesno(
                        "Stim time",
                        f"Stim at is empty. Use {_fmt_s(guess)} s, the start of '{post}', "
                        "as the stim time?", parent=win):
                    _cfg_say("Type the stim time in Stim at (s), then Save as… again")
                    return
                stim = guess
                _set_stim(stim)
            shown = cb_cfg.get()
            name = (ask_config_name(win, shown if shown in _cfgs else "") or "").strip()
            if not name:
                return
            if name in _cfgs and not messagebox.askyesno(
                    "Save configuration", f"Replace the configuration '{name}'?",
                    parent=win):
                return
            try:
                save_template(config_from_rows(name, rows, stim), _templates_path())
            except (ValueError, OSError) as exc:
                _cfg_say(f"⚠  Not saved: {exc}", RED)
                return
            _cfg_refresh(select=name)
            _cfg_say(f"✓  '{name}' saved: {len(rows)} window(s) relative to the stim "
                     f"({_fmt_s(stim)} s here)", GREEN)

        def _cfg_delete() -> None:
            name = cb_cfg.get()
            if name not in _cfgs:
                return
            if not messagebox.askyesno("Delete configuration",
                                       f"Delete the configuration '{name}'? "
                                       "This cannot be undone.", parent=win):
                return
            try:
                delete_template(name, _templates_path())
            except OSError as exc:
                _cfg_say(f"⚠  Not deleted: {exc}", RED)
                return
            _cfg_refresh()
            _cfg_say(f"'{name}' deleted")

        btn_cfg_apply.configure(command=_cfg_apply)
        btn_cfg_save.configure(command=_cfg_save)
        btn_cfg_del.configure(command=_cfg_delete)
        _cfg_refresh()

        rows_frame = ctk.CTkFrame(seg_card, fg_color="transparent")
        rows_frame.pack(fill="x")
        seg_rows: "list[dict]" = []
        # "ref": what the Δ % are against, kept across runs -- a segment index,
        # PAIRWISE or PAIRS (segment_plan.reference_comparators).
        # "run": the windows and block length of the comparison shown.
        _state: "dict[str, Any]" = {"segs": None, "cmps": None, "flags": None, "ref": 0,
                                    "run": None, "inflight": None}

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
            # As typed: an entry not (yet) a number stays as it is.
            _set_rows([(x["lbl"].get(), x["lo"].get(), x["hi"].get())
                       for x in seg_rows if x is not r])

        def _add_row(label: str, lo: "Any", hi: "Any") -> None:
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
            lo_e.insert(0, _fmt_s(lo) if isinstance(lo, (int, float)) else str(lo))
            lo_e.pack(side="left", padx=(4, SPACE_M))
            ctk.CTkLabel(f, text="To (s):", font=FONT_SMALL, text_color=MUTED).pack(side="left")
            hi_e = ctk.CTkEntry(f, width=80, **input_style())
            hi_e.insert(0, _fmt_s(hi) if isinstance(hi, (int, float)) else str(hi))
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
                    len_l.configure(text=f"= {_fmt_s(d)} s" if d > 0 else "⚠ end ≤ start",
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
        # A slot of fixed size always in the row: the bar shows and hides in
        # it without moving the controls.
        progress_slot = ctk.CTkFrame(ctrl, fg_color="transparent", width=200, height=8)
        progress_slot.pack(side="right", padx=(SPACE_M, 0))
        progress_slot.pack_propagate(False)
        progress_bar = ctk.CTkProgressBar(progress_slot, height=8, mode="indeterminate",
                                          progress_color=TEAL)
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

        # What the Δ % are against: shown from the first run with two
        # segments or more. A segment (all against it) or the previous one
        # (pairwise: S2 vs S1, S3 vs S2...); any pair A vs B is A as the
        # reference. Changing it re-renders without recomputing the stats.
        ref_bar = ctk.CTkFrame(res, fg_color="transparent")
        ctk.CTkLabel(ref_bar, text="Δ % against:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        _ref_opts: "dict[str, Any]" = {}
        cb_ref = ctk.CTkComboBox(ref_bar, width=240, values=[""], state="readonly",
                                 **combo_style())
        cb_ref.pack(side="left", padx=(4, SPACE_M))
        lbl_ref = ctk.CTkLabel(ref_bar, text="", font=FONT_SMALL, text_color=MUTED,
                               anchor="w")
        lbl_ref.pack(side="left", fill="x", expand=True)
        tip(cb_ref, "The segment every other one is compared with (its Δ % column and the "
                    "block-mean line above), or each with the one before it. For one pair "
                    "A vs B, choose A. The exports and the quality checks follow.")
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
        # ncols: the Poincaré columns last drawn (None: no Poincaré drawn).
        _view: "dict[str, Any]" = {"mode": "rr", "ncols": None}
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
        # Display only: SD1, SD2 and the ellipse use the measured values.
        if not hasattr(self, "_compare_jitter"):
            self._compare_jitter = True                   # kept across tab rebuilds
        chk_jitter = ctk.CTkCheckBox(view_bar, text="Jitter Poincaré points", font=FONT_SMALL,
                                     text_color=MUTED, fg_color=BLUE, checkmark_color="white",
                                     border_color=BORDER2, width=16)
        if self._compare_jitter:
            chk_jitter.select()

        def _on_jitter() -> None:
            self._compare_jitter = bool(chk_jitter.get())
            if _state["segs"] is not None and _view["mode"] == "poincare":
                _redraw_plot()
        chk_jitter.configure(command=_on_jitter)
        chk_jitter.pack(side="left", padx=(SPACE_M, 0))
        tip(chk_jitter, "On: points on the sampling grid (1 ms at 1 kHz) are spread by "
                        "± half a step so the cloud's density shows. Off: every point at its "
                        "measured value. Display and figure export only; SD1, SD2 and the "
                        "ellipse never change.")
        plot_card = ctk.CTkFrame(plot_box, fg_color=CARD, corner_radius=0,
                                 height=_PLOT_CARD_H)
        plot_card.pack(fill="x")
        plot_card.pack_propagate(False)        # a scrolling page gives no height to fit
        plot_slot = CanvasSlot(plot_card, 14, 4.6, toolbar=False, yscale_bar=True,
                               placeholder="Click ▶ Compare to overlay tachograms")
        plot_slot.enable_wheel_yzoom()

        def _poincare_grid(n: int) -> "tuple[int, int]":
            """(columns, rows) of the Poincaré panels. Every panel keeps the
            size it has on a single row (one row height, _POINCARE_ROW_H): as
            many columns as the card's width holds at that size (at most 4),
            and further segments wrap onto new rows, which grow the card,
            instead of every panel shrinking as segments are added."""
            scale = plot_card._get_widget_scaling()
            width = plot_card.winfo_width() / scale     # CTk units, like heights
            fit = int(width // _POINCARE_COL_W) if width > 1 else 4
            ncols = max(1, min(n, 4, fit))
            return ncols, -(-n // ncols)

        def _cur_comps(segs: "list[dict]") -> "list[Optional[int]]":
            """The comparators shown: those of the kept result for *segs*,
            else the selector's."""
            res_ = self.analysis.last_comparison
            if res_ is not None and res_.segs is segs:
                return res_.comps
            return reference_comparators(len(segs), _state["ref"])

        def _fit_plot_card(_event: "Any" = None) -> None:
            """The card's height for the view shown: one row for the
            tachograms (one fixed-size row per pair in Pairs mode), one
            fixed-size row per row of Poincaré panels. On a resize (an
            event) that changes how many Poincaré columns the card holds --
            also when a freshly built card reaches its real width after a
            draw (session restore, UI rebuild) -- the panels are redrawn."""
            segs = _state["segs"]
            if (_event is not None and _view["mode"] == "poincare" and segs
                    and _view["ncols"] is not None
                    and _poincare_grid(len(segs))[0] != _view["ncols"]):
                _redraw_plot()
                return
            if _view["mode"] == "poincare" and segs:
                rows = _poincare_grid(len(segs))[1]
                # + the gap the layout puts between two rows of panels.
                height = (_YBAR_H + rows * _POINCARE_ROW_H
                          + (rows - 1) * _POINCARE_ROW_GAP)
            elif segs:
                panels = len(tachogram_groups(len(segs), _cur_comps(segs)))
                height = max(_PLOT_CARD_H, _YBAR_H + panels * _TACHO_ROW_H)
            else:
                height = _PLOT_CARD_H
            if int(plot_card.cget("height")) != height:
                plot_card.configure(height=height)
        # A narrower window holds fewer columns, so more rows: refit on resize.
        plot_card.bind("<Configure>", _fit_plot_card, add="+")
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

        def _seg_names(segs: "list[dict]") -> "list[str]":
            """What the screen calls the segments: their labels, unless two
            are the same (or one is empty / reserved) -- then the export tags
            (unique_tags: A, B, C...), followed by the label, so the selector,
            the table, the caption and the exports name the same segment."""
            labels = [str(s_.get("label") or "").strip() for s_ in segs]
            tags = unique_tags(labels)
            if tags == labels:
                return labels
            return [f"{t}: {lbl}" if lbl else t for t, lbl in zip(tags, labels)]

        def _populate_table(segs: "list[dict]", comps: "list[Optional[int]]") -> None:
            """The metric table: each segment's values, then the Δ % against
            its comparator (comps[k]; none for the reference, marked ref.)."""
            import tkinter.font as tkfont
            head_font = tkfont.Font(font=(THEME.font_family, _fsize, "bold"))
            gap = int(14 * _scale)
            pairwise = not isinstance(comparators_reference(comps), int)
            n = len(segs)
            names = _seg_names(segs)

            def _seg_head(k: int) -> str:
                # Pairwise has no single reference: the first segment is only
                # the start of the chain, not marked.
                return names[k] + (" (ref.)" if comps[k] is None and n > 1
                                   and not pairwise else "")

            def _d_head(k: int) -> str:
                # One reference: the caption says which, "Δ %" is enough;
                # pairwise every column names its own comparator.
                j = comps[k]
                if not pairwise or j is None:
                    return "Δ %"
                # Whole label up to 18 characters ("Pre Stim 10min" fits), so
                # the pairs of a Pre / Post protocol read without guessing.
                lbl = names[j] if len(names[j]) <= 18 else names[j][:17] + "…"
                return f"Δ % vs {lbl}"
            x = int(270 * _scale)
            stops = []
            for k in range(n):
                # Wide enough for its header (a long label no longer runs
                # into the column on its left).
                x += max(int(110 * _scale), head_font.measure(_seg_head(k)) + gap)
                stops += [x, "right"]
                if comps[k] is not None:
                    x += max(int(80 * _scale), head_font.measure(_d_head(k)) + gap)
                    stops += [x, "right"]
            # Text width is in characters: from the font's own "0".
            char_px = max(1, tkfont.Font(font=tbl.cget("font")).measure("0"))
            tbl.configure(state="normal", tabs=stops, width=int(x / char_px) + 3)
            _place(len(segs))
            tbl.delete("1.0", "end")
            for k, s_ in enumerate(segs):
                tbl.tag_configure(f"seg{k}", font=(THEME.font_family, _fsize, "bold"),
                                  foreground=_color(k))
            tbl.insert("end", "Metric", "head")
            for k in range(n):
                tbl.insert("end", "\t")
                tbl.insert("end", _seg_head(k), f"seg{k}")
                if comps[k] is not None:
                    tbl.insert("end", "\t" + _d_head(k), "head")
            for ri, (key, name, unit) in enumerate(_METRICS):
                line = ("stripe",) if ri % 2 == 0 else ()
                tbl.insert("end", "\n" + name, line)
                if unit:
                    tbl.insert("end", f"  {unit}", ("unit",) + line)
                vals = [float(s_.get(key, float("nan"))) for s_ in segs]
                for k, v in enumerate(vals):
                    tbl.insert("end", "\t" + _fmt_cell(key, v), line)
                    j = comps[k]
                    if j is None:
                        continue
                    d = _delta(vals[j], v)
                    if not np.isfinite(d):
                        txt = "—"
                    elif abs(d) < 0.05:
                        txt = "0.0 %"
                    else:
                        txt = f"{'▲' if d > 0 else '▼'} {abs(d):.1f} %"
                    tbl.insert("end", "\t" + txt, ("delta",) + line)
            tbl.configure(height=len(_METRICS) + 1, state="disabled")

        def _rr_drawer(segs: "list[dict]", colors: "list[str]",
                       comps: "Optional[list[Optional[int]]]" = None):
            """The RR tachograms of *segs*, as a draw function: superimposed,
            or one panel per pair when *comps* are Pairs (draw_rr_tachograms)."""
            names = _seg_names(segs)
            return lambda fig: draw_rr_tachograms(fig, segs, colors, comps, names)

        def _redraw_plot() -> None:
            segs = _state["segs"]
            colors = [_color(k) for k in range(len(segs))]
            _fit_plot_card()
            if _view["mode"] == "poincare":
                ncols = _poincare_grid(len(segs))[0]
                _view["ncols"] = ncols
                plot_slot.update(lambda fig: draw_poincare_set(
                    fig, segs, colors, jitter=self._compare_jitter, ncols=ncols))
            else:
                _view["ncols"] = None
                plot_slot.update(_rr_drawer(segs, colors, _cur_comps(segs)))

        def _show_flags(flags: "list[list[str]]") -> None:
            for r, fl in zip(seg_rows, flags):
                r["qc"].configure(text=("⚠  " + "  ·  ".join(fl)) if fl else "✓  quality ok",
                                  text_color=ORANGE if fl else GREEN)

        def _pair_cmps(segs: "list[dict]", comps: "list[Optional[int]]") -> "list[dict]":
            """compare_segments of every (comparator, segment) pair, in the
            order of ComparisonResult.pairs. Cheap (block means only)."""
            from ecg.core.analysis import compare_segments
            return [compare_segments(segs[j], segs[k]) for j, k in comparison_pairs(comps)]

        def _fill_ref_choices(segs: "list[dict]") -> None:
            """The selector's choices for *segs*: each segment, pairwise from
            three on (with two it is the first as the reference), and pairs
            (2 vs 1, 4 vs 3...) from four on (with three it is one pair and a
            segment alone)."""
            _ref_opts.clear()
            # Unique names: two segments with the same label stay two choices.
            for k, name in enumerate(_seg_names(segs)):
                _ref_opts[f"{name} (reference)"] = k
            if len(segs) > 2:
                _ref_opts["Previous segment (pairwise)"] = PAIRWISE
            if len(segs) > 3:
                _ref_opts["Pairs (2 vs 1, 4 vs 3, 6 vs 5)"] = PAIRS
            if _state["ref"] not in _ref_opts.values():
                _state["ref"] = 0
            cb_ref.configure(values=list(_ref_opts))
            cb_ref.set(next((name for name, v in _ref_opts.items() if v == _state["ref"]),
                            next(iter(_ref_opts), "")))
            if len(segs) > 1:
                if not ref_bar.winfo_manager():
                    ref_bar.pack(fill="x", pady=(0, SPACE_S), before=grid)
            elif ref_bar.winfo_manager():
                ref_bar.pack_forget()

        def _render_comparison() -> None:
            """The comparison text, quality flags and table of the last run,
            and what the app keeps of it, for the reference in _state."""
            from ecg.core.analysis import segment_comparison_text
            segs, cmps, flags = _state["segs"], _state["cmps"], _state["flags"]
            res_ = self.analysis.last_comparison
            comps = res_.comps if res_ is not None else reference_comparators(len(segs))
            pairs = comparison_pairs(comps)
            names = _seg_names(segs)
            # A vs B of the old two-segment API: the first compared pair.
            self._last_seg_a = segs[pairs[0][0]] if pairs else segs[0]
            self._last_seg_b = segs[pairs[0][1]] if pairs else None
            lines = [segment_comparison_text(c, names[j], names[k]).split("\n")
                     for c, (j, k) in zip(cmps, pairs)]
            lbl_mw.configure(text="\n".join(ln[0] for ln in lines)
                             + (f"\n{lines[0][1]}" if lines and len(lines[0]) > 1 else ""))
            single_ref = isinstance(comparators_reference(comps), int)
            lbl_ref.configure(text=(comparators_caption(comps, names)
                                    + ("  ·  the reference column is marked (ref.)"
                                       if single_ref else ""))
                              if len(segs) > 1 else "")
            _show_flags(flags)
            _populate_table(segs, comps)

        def _on_ref_pick(choice: str) -> None:
            """A new reference after a run: the block comparisons, flags and
            table follow it; the segment statistics are not recomputed."""
            if choice not in _ref_opts:
                return
            _state["ref"] = _ref_opts[choice]
            segs = _state["segs"]
            res_ = self.analysis.last_comparison
            if not segs or res_ is None or res_.segs is not segs:
                return
            comps = reference_comparators(len(segs), _state["ref"])
            cmps = _pair_cmps(segs, comps)
            flags = quality_flags(segs, None, comparators=comps)
            _state.update(cmps=cmps, flags=flags)
            res_.comparators, res_.cmps, res_.flags = list(comps), cmps, flags
            _render_comparison()
            # Pairs draws one tachogram panel per pair, the other modes one.
            if _view["mode"] == "rr":
                _redraw_plot()

        cb_ref.configure(command=_on_ref_pick)

        def _run_compare(run: "Optional[dict]" = None) -> None:
            """Compare the rows typed, or the windows and block length of
            *run* (a restored setup's, the rows shown being those typed)."""
            if run is not None:
                windows = [(str(lbl), float(lo), float(hi)) for lbl, lo, hi in run["rows"]]
                block_s = float(run["block_s"])
            else:
                try:
                    windows = _read_rows()
                except ValueError as exc:
                    lbl_status.configure(text=f"⚠  {exc}", text_color=RED); return
                try:
                    block_s = float(ent_block.get())
                    if not (np.isfinite(block_s) and block_s > 0):
                        raise ValueError("block length")
                except ValueError:
                    lbl_status.configure(
                        text="⚠  Block length must be a positive number of seconds",
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
            # The windows being computed: a Save Session or a UI rebuild
            # before the result lands keeps (and re-runs) this comparison.
            _state["inflight"] = {"rows": [[str(l), float(lo), float(hi)]
                                           for l, lo, hi in windows],
                                  "block_s": float(block_s)}
            run_btn.configure(state="disabled")
            progress_bar.place(relx=0, rely=0, relwidth=1, relheight=1)
            progress_bar.start()
            lbl_status.configure(text="  Computing…", text_color=ORANGE)
            # HRV inputs read on the main thread: the user's RR bounds and
            # the beats removed by artifact correction (excluded spans).
            rr_lo, rr_hi = self.analysis_ctrl.hrv_rr_bounds()
            seg_breaks   = self.analysis_ctrl.hrv_breaks(rp)
            import threading as _th

            def _worker():
                try:
                    from ecg.core.analysis import compute_segment_stats
                    seg_kw = dict(lf_band=lf, hf_band=hf, vlf_band=vlf,
                                  rr_min_ms=rr_lo, rr_max_ms=rr_hi,
                                  breaks=seg_breaks, block_s=block_s)
                    segs = [compute_segment_stats(sig, rp, fs, lo, hi, lbl, **seg_kw)
                            for lbl, lo, hi in windows]
                except Exception as exc:
                    log.exception("Segment comparison worker failed")
                    win.after(0, lambda e=exc: _on_worker_error(e))
                    return
                win.after(0, lambda: _on_done(segs, windows, block_s))

            _th.Thread(target=_worker, daemon=True).start()

        def _on_worker_error(exc: Exception) -> None:
            if not run_btn.winfo_exists():      # the tab was rebuilt meanwhile
                return
            _state["inflight"] = None
            run_btn.configure(state="normal")
            progress_bar.stop(); progress_bar.place_forget()
            lbl_status.configure(text=f"⚠  Comparison failed — {exc}", text_color=RED)

        def _on_done(segs: "list[dict]", windows: "list[tuple[str, float, float]]",
                     block_s: float) -> None:
            if not run_btn.winfo_exists():      # the tab was rebuilt meanwhile
                return
            _state["inflight"] = None
            run_btn.configure(state="normal")
            progress_bar.stop(); progress_bar.place_forget()
            # The reference as it is NOW (the selector stays usable while
            # the worker runs), kept when it still exists for these
            # segments; then the cheap block-mean comparison of each
            # segment with its comparator (descriptive, no beat-level
            # test: beats are not replicates) and the quality flags.
            _fill_ref_choices(segs)
            comps = reference_comparators(len(segs), _state["ref"])
            try:
                cmps = _pair_cmps(segs, comps)
                flags = quality_flags(segs, None, comparators=comps)
            except Exception as exc:
                log.exception("Segment comparison failed")
                _on_worker_error(exc)
                return
            run_spec = {"rows": [[str(lbl), float(lo), float(hi)] for lbl, lo, hi in windows],
                        "block_s": float(block_s)}
            _state.update(segs=segs, cmps=cmps, flags=flags, run=run_spec)
            # Kept for the exports here and the main ones (Report tab),
            # with the recording state it rests on (_current_comparison).
            self.analysis.last_comparison = ComparisonResult(
                segs=segs, metrics=list(_METRICS), cmps=cmps, flags=flags,
                template="", subject=_subject(),
                filepath=self.signal.filepath or "", event=None,
                artefact_margin_s=None, block_s=block_s,
                lf_band=tuple(lf), hf_band=tuple(hf), comparators=list(comps))
            # The state the tab -- hence rp / sig of this run -- was
            # built for, not the app's now (peaks may have been edited
            # on another tab while the worker ran).
            self.analysis.last_comparison_basis = built_for
            self.analysis.last_comparison_run = run_spec
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
            _render_comparison()
            _redraw_plot()

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

        def _done(text: str, ok: bool = True, warn: bool = False) -> None:
            if not lbl_exp.winfo_manager():      # shown from the first export on
                lbl_exp.pack(fill="x", pady=(0, SPACE_XS), before=lbl_mw)
            lbl_exp.configure(text=("✓  " if ok and not warn else "⚠  ") + text,
                              text_color=RED if not ok else ORANGE if warn else GREEN)

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
            prism_menu(win, _prism_btn[0], _export_prism_new, _export_prism_template)

        def _export_prism_template() -> None:
            res_ = _result()
            if res_ is None:
                return
            tpl = ask_prism_template(self, win)
            if not tpl:
                return
            out = ask_filled_path(win, tpl, _base_name())
            if not out:
                return
            try:
                report = fill_template(tpl, recording_tables(res_), out,
                                       notes=pzfx_notes(res_, _stamp())
                                       + [("Prism template", os.path.basename(tpl))])
            except Exception as exc:
                if not isinstance(exc, PrismTemplateError):
                    log.exception("comparison Prism template")
                messagebox.showerror("Prism template", str(exc), parent=win)
                _done(f"Prism template: {exc}", ok=False)
                return
            _done(f"Prism template {os.path.basename(tpl)}: {report.summary()} → "
                  f"{os.path.basename(out)}", warn=report.warn)

        def _export_prism_new() -> None:
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
            n_tacho = len(tachogram_groups(len(segs_), res_.comps))
            written = []
            for suffix, figsize, draw in (
                    ("_tachograms", (10, 4 if n_tacho == 1 else 3 * n_tacho),
                     _rr_drawer(segs_, colors, res_.comps)),
                    ("_poincare", (min(4 * len(segs_), 16), 4.6 * n_rows),
                     lambda fig: draw_poincare_set(fig, segs_, colors, title=_subject(),
                                                   jitter=self._compare_jitter))):
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
            add_all = getattr(self, "_add_open_recording_to_group_table", None)
            extra = ""
            try:
                if callable(add_all) and getattr(getattr(self, "signal", None), "filepath", None):
                    # The whole experiment's table (Group tab): the segments
                    # with the whole recording, time course and events.
                    n, replaced, what = add_all(path, comparison=res_)
                    extra = f" — {what.split(': ', 1)[-1]}"
                    reload_ = getattr(self, "_grouptab_reload", None)
                    gts = getattr(self, "_grouptab_state", None) or {}
                    if callable(reload_) and gts.get("path") and os.path.normcase(
                            os.path.abspath(gts["path"])) == os.path.normcase(os.path.abspath(path)):
                        reload_()
                else:
                    n, replaced = append_group_row(path, group_row(
                        res_.subject, res_.filepath, res_.segs, res_.metrics,
                        template=res_.template, flags=res_.flags, sex=_sex()))
            except Exception as exc:
                messagebox.showerror("Group table", str(exc), parent=win)
                return
            # The Group card tests one pair A vs B: unless one was chosen
            # there, start from this comparison's reference vs the first
            # segment compared with it (the group table's tags are the same
            # unique_tags as res_.tags).
            gst = getattr(self, "_group_state", None)
            if gst is not None and not (gst.get("a") and gst.get("b")) and res_.pairs:
                j, k = res_.pairs[0]
                gst["a"], gst["b"] = res_.tags[j], res_.tags[k]
            _done(f"{res_.subject or 'recording'} {'updated in' if replaced else 'added to'} "
                  f"{os.path.basename(path)} ({n} recording{'s' if n != 1 else ''}){extra}")

        run_btn.configure(command=_run_compare)
        lbl_exp = ctk.CTkLabel(res, text="", font=FONT_SMALL, text_color=MUTED,
                               anchor="w", justify="left")
        _export_btns: "list[ctk.CTkButton]" = []
        _prism_btn: "list[ctk.CTkButton]" = []
        # Right to left: the header packs them from its right edge.
        for text, cmd, tiptext in (
                ("➕  Add to group table…", _add_to_group,
                 "One row per recording (subject, sex, file, every metric and the "
                 "quality of each segment): the table for a paired / repeated-measures "
                 "test across animals. Every segment is stored, whatever the Δ % "
                 "reference; the pair tested is chosen in the Group card below (the "
                 "reference vs the first segment compared with it, until you change it). "
                 "Columns are named after the segment labels: use "
                 "the same labels and windows for every animal. A recording already in it is replaced. "
                 "Its whole-recording metrics, time course, annotations and event-locked "
                 "results go to the same table, for the Group tab."),
                ("Figures…", _export_figures,
                 "The tachograms and the Poincaré plots, 300 dpi PNG or vector PDF / SVG"),
                ("📋  Copy table", _copy_table,
                 "The metric table as tab-separated text: paste it into Excel or Prism"),
                ("Prism…", _export_prism,
                 "GraphPad Prism: the metric table, one table per metric (a column per "
                 "segment; per pair in Pairs mode), the HR block means, the RR series and "
                 "the Poincaré pairs; the windows and quality in the project notes."),
                ("📊  Excel…", _export_excel,
                 "Workbook: Comparison (Δ % against the reference chosen above the table), "
                 "Quality, Summary, "
                 "Blocks, then the RR series and Poincaré pairs of each segment. "
                 "Choose .csv for the metric table only.")):
            b = ctk.CTkButton(res_hdr, text=text, width=0, command=cmd, state="disabled",
                              **{**button_style("secondary"), **disabled_style()})
            b.pack(side="right", padx=(SPACE_S, 0))
            if cmd is _export_prism:
                prism_tip(self, b, tiptext)
                _prism_btn.append(b)
            else:
                tip(b, tiptext)
            _export_btns.append(b)
        ctk.CTkLabel(res_hdr, text="Export:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="right", padx=(0, SPACE_XS))

        ctk.CTkLabel(res,
                     text=f"Equal durations give a fair comparison  ·  LF {lf[0]:.2f}–{lf[1]:.2f} Hz  "
                          f"HF {hf[0]:.2f}–{hf[1]:.2f} Hz  ·  wheel: zoom Y  ·  middle-click: reset Y",
                     font=FONT_HINT, text_color=MUTED, anchor="w").pack(
            fill="x", pady=(SPACE_XS, 0))

        # ── Setup kept by Save Session and a UI rebuild ───────────────────
        def _num(text: str) -> "Any":
            f = _finite(text.strip()) if isinstance(text, str) else None
            return f if f is not None else text

        def _get_setup() -> dict:
            """The rows as typed, the block length, the reference, the view
            and the jitter flag, with the windows of the comparison shown
            (or being computed: it is re-run on restore)."""
            run = _state["inflight"] or _state["run"]
            return {"rows": [[r["lbl"].get(), _num(r["lo"].get()), _num(r["hi"].get())]
                             for r in seg_rows],
                    "block_s": ent_block.get(), "ref": _state["ref"],
                    "stim": ent_stim.get().strip(),
                    "view": _view["mode"], "jitter": bool(self._compare_jitter),
                    "ran": run is not None,
                    "run": dict(run) if run is not None else None}

        def _apply_setup(setup: dict) -> None:
            """A normalised setup (normalize_compare_setup) into the tab; its
            comparison shown again: the kept result when it is of these
            peaks and windows (a UI rebuild), else re-run on them."""
            _set_rows([tuple(r) for r in setup["rows"]])
            ent_stim.delete(0, "end")
            if setup.get("stim"):
                ent_stim.insert(0, setup["stim"])
            ent_block.delete(0, "end")
            ent_block.insert(0, setup["block_s"])
            _state["ref"] = setup["ref"]
            self._compare_jitter = bool(setup["jitter"])
            chk_jitter.select() if self._compare_jitter else chk_jitter.deselect()
            _set_view(setup["view"])
            run = setup.get("run")
            if run is None:
                return
            res_ = self.analysis.last_comparison
            if (res_ is not None and getattr(res_, "segs", None)
                    and getattr(self.analysis, "last_comparison_basis", None) == built_for
                    and getattr(self.analysis, "last_comparison_run", None) == run):
                _on_done(res_.segs, [tuple(r) for r in run["rows"]], run["block_s"])
            else:
                _run_compare(run)

        self._compare_setup_getter = _get_setup
        self._compare_setup_key = _recording_key(self)
        pend = getattr(self, "_compare_pending_setup", None)
        if pend is not None and pend[0] == self._compare_setup_key:
            self._compare_pending_setup = None
            try:
                _apply_setup(pend[1])
            except Exception as exc:
                log.warning("Compare segments setup not restored: %s", exc, exc_info=True)

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
        st.setdefault("mode", MODE_ONE)
        st.setdefault("holm_all", False)
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
        row_mode = ctk.CTkFrame(body, fg_color="transparent")
        row_mode.pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkLabel(row_mode, text="Analysis:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        cb_mode = ctk.CTkComboBox(row_mode, width=290, values=[MODE_ONE, MODE_ALL],
                                  state="readonly", **combo_style())
        cb_mode.set(st["mode"] if st["mode"] in (MODE_ONE, MODE_ALL) else MODE_ONE)
        cb_mode.pack(side="left", padx=(4, SPACE_M))
        tip(cb_mode, "One pair: the two segments chosen below. All pairs: every consecutive pair "
                     "of segments of the table (B vs A, D vs C, F vs E: Pre / Post 1, 5, 10 min), "
                     "the same paired tests for each, Holm within each pair; an odd last "
                     "segment is left out. A row whose segment label is not the group's is a "
                     "row to check.")
        chk_holm_all = ctk.CTkCheckBox(row_mode, text="Holm across all pairs × metrics",
                                       font=FONT_SMALL, text_color=MUTED, fg_color=BLUE,
                                       checkmark_color="white", border_color=BORDER2, width=16)
        if st["holm_all"]:
            chk_holm_all.select()
        chk_holm_all.pack(side="left", padx=(0, SPACE_M))
        tip(chk_holm_all, "All pairs only. Off: Holm within each pair, over its metrics (as for "
                          "one pair). On: also one Holm correction over every pair × metric "
                          "tested, when the pairs answer one question together.")
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

        def _sync_mode(_v: object = None) -> None:
            """All pairs: no A / B to choose, the Holm-across option instead."""
            all_pairs = cb_mode.get() == MODE_ALL
            for w in (cb_a, cb_b, ent_na, ent_nb):
                w.configure(state="disabled" if all_pairs else
                            ("readonly" if isinstance(w, ctk.CTkComboBox) else "normal"))
            chk_holm_all.configure(state="normal" if all_pairs else "disabled")

        cb_mode.configure(command=_sync_mode)
        _sync_mode()

        lbl_status = ctk.CTkLabel(body, text="", font=FONT_SMALL, text_color=MUTED,
                                  anchor="w", justify="left")
        lbl_status.pack(fill="x")
        # The Prism export's result: a line of its own, under the analysis
        # message (whose warnings stay in view).
        lbl_prism = ctk.CTkLabel(body, text="", font=FONT_SMALL, text_color=MUTED,
                                 anchor="w", justify="left")

        def _wrap(e: "Any") -> None:
            for lbl in (lbl_status, lbl_prism):
                lbl.configure(wraplength=max(200, e.width - 8))
        body.bind("<Configure>", _wrap)
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
                                      "left_out": [], "mode": MODE_ONE, "all": None}

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
            # The last run's results belong to the table it read: none until
            # the next Run (else Statistics… / Prism… would write the old
            # rows under the new file's name).
            res_state.update(used=None, all=None, stats=None, checks=None, keep=None,
                             left_out=[], path=None)
            for btn_ in (btn_xlsx, btn_fig, btn_prism):
                self._set_btn_enabled(btn_, False, "secondary")
            lbl_prism.configure(text="")
            if lbl_prism.winfo_manager():
                lbl_prism.pack_forget()
            cb_a.configure(values=tags)
            cb_b.configure(values=tags)
            # The two most filled segments by default, or the ones chosen before.
            filled = sorted(tags, key=lambda t: -int(df[f"start (s) {t}"].notna().sum()))
            a = st["a"] if st["a"] in tags else filled[0]
            b = st["b"] if st["b"] in tags and st["b"] != a else next(t for t in filled if t != a)
            for cb in (cb_a, cb_b):          # a disabled entry ignores set()
                cb.configure(state="readonly")
            cb_a.set(a)
            cb_b.set(b)
            _sync_mode()
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

        def _fmt(v: object, p: bool = False) -> str:
            try:
                v = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "—"
            if not np.isfinite(v):
                return "—"
            if p:
                return f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
            return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"

        def _show_all(res) -> None:
            """All pairs: one line per pair × metric, then the notes and the
            rows to check of each pair."""
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            x = int(250 * _scale)
            stops = [x, "left"]
            x += int(250 * _scale)
            widths = (60, 100, 90, 90) + ((120,) if res.holm_across else ())
            for w in widths:
                x += int(w * _scale)
                stops += [x, "right"]
            txt.configure(tabs=stops)
            heads = ["Pair", "Metric", "n (≠0)", "Median Δ", "Wilcoxon p", "p (Holm)"]
            if res.holm_across:
                heads.append("Holm all pairs")
            txt.insert("end", "\t".join(heads), "head")
            line_no = 0
            for k, pr in enumerate(res.pairs, 1):
                if not len(pr.stats):
                    txt.insert("end", f"\n{k}. {pr.title}\tno animal with both segments",
                               "muted")
                    continue
                for j, r in pr.stats.reset_index(drop=True).iterrows():
                    line = ("stripe",) if line_no % 2 == 0 else ()
                    line_no += 1
                    txt.insert("end", "\n" + (f"{k}. {pr.title}" if j == 0 else ""), line)
                    txt.insert("end", "\t" + str(r["Metric"]), line)
                    for v in (f"{int(r['n'])} ({int(r['n tested (Δ ≠ 0)'])})",
                              _fmt(r["Median Δ (B − A)"]), _fmt(r["Wilcoxon p"], True)):
                        txt.insert("end", "\t" + v, line)
                    hcols = ["Wilcoxon p (Holm)"] + ([HOLM_ALL] if res.holm_across else [])
                    for hc in hcols:
                        ph = float(r.get(hc, float("nan")))
                        # Significance shown on the strictest correction asked.
                        sig = np.isfinite(ph) and ph < 0.05 and hc == hcols[-1]
                        txt.insert("end", "\t" + _fmt(ph, True), ("sig",) + line if sig else line)
            notes = list(res.notes) + [f"{pr.title}: {n}" for pr in res.pairs for n in pr.notes]
            if notes:
                txt.insert("end", "\n\nNotes", "head")
                for n in notes:
                    txt.insert("end", "\n" + n, "muted")
            for k, pr in enumerate(res.pairs, 1):
                bad = [(c, kp) for c, kp in zip(pr.checks, pr.keep) if not c.ok]
                if not bad:
                    continue
                txt.insert("end", f"\n\nRows to check — {k}. {pr.title}", "head")
                for c, kp in bad:
                    missing = any(p.startswith("no segment") for p in c.problems)
                    tag = "kept" if kp else ("not in this pair" if missing else "left out")
                    txt.insert("end", f"\n[{tag}] {c.subject or '—'} · {c.file}: "
                               + "; ".join(c.problems), "muted" if missing else "warn")
            n_lines = int(txt.index("end-1c").split(".")[0])
            txt.configure(height=n_lines, state="disabled")
            if not txt.winfo_manager():
                txt.pack(fill="x", pady=(SPACE_S, SPACE_M))
                plot_card.pack(fill="x")

        def _run_all(df) -> None:
            st.update(exclude=bool(chk_ex.get()), all=bool(chk_all.get()))
            # Pairs follow each row's own order of segments (segment_layout),
            # not the column order: a typo in one animal's label is a row to
            # check, never a pair of its own.
            res = all_pairs_stats(df, "all" if st["all"] else None, exclude=st["exclude"],
                                  holm_across=st["holm_all"])
            if not res.pairs:
                lbl_status.configure(text="⚠  the table holds fewer than two segments",
                                     text_color=RED)
                return
            res_state.update(mode=MODE_ALL, all=res, used=None, stats=None, path=st["path"])
            left = sorted({s for pr in res.pairs for s in pr.left_out})
            ns = [len(pr.used) for pr in res.pairs]
            # Animals of the table missing from a pair (no such segment).
            short = [k for k, n in enumerate(ns, 1) if n < len(df)]
            n_test = max((int(pr.stats["n tested (Δ ≠ 0)"].max()) for pr in res.pairs
                          if len(pr.stats)), default=0)
            weak = not (smallest_wilcoxon_p(n_test) < 0.05) if n_test else True
            msg = (f"{len(res.pairs)} pair{'s' if len(res.pairs) != 1 else ''} tested "
                   f"({', '.join(f'{pr.b} vs {pr.a}' for pr in res.pairs)})  ·  animals per pair: "
                   f"{', '.join(map(str, ns))}"
                   + (f"  ·  segment {res.odd} has no partner: not tested" if res.odd else "")
                   + (f"  ·  left out (rows to check): {', '.join(left)}" if left else "")
                   + (f"  ·  fewer animals than rows ({len(df)}) in pair "
                      f"{', '.join(map(str, short))}" if short else "")
                   + ("  ·  Holm across all pairs × metrics" if res.holm_across else "")
                   + ("  ·  no metric can reach p < 0.05 with these n" if weak else ""))
            warn = bool(left or short or weak or res.odd or res.notes
                        or any(pr.notes for pr in res.pairs))
            lbl_status.configure(text=("⚠  " if warn else "✓  ") + msg,
                                 text_color=ORANGE if warn else GREEN)
            _show_all(res)
            slot.update(lambda fig: draw_all_pairs_figure(fig, res))
            for btn_ in (btn_xlsx, btn_fig, btn_prism):
                self._set_btn_enabled(btn_, True, "secondary")

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
            st.update(mode=cb_mode.get(), holm_all=bool(chk_holm_all.get()))
            if st["mode"] == MODE_ALL:
                _run_all(df)
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
            res_state.update(used=used, stats=stats, checks=checks, left_out=left_out, keep=keep,
                             mode=MODE_ONE, all=None, path=st["path"])
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
            for btn_ in (btn_xlsx, btn_fig, btn_prism):
                self._set_btn_enabled(btn_, True, "secondary")

        run_btn.configure(command=_run)

        def _save_all_stats() -> None:
            res = res_state["all"]
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the group statistics (all pairs)",
                defaultextension=".xlsx", initialfile="group_stats_all_pairs.xlsx",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")])
            if not path:
                return
            try:
                write_all_pairs_report(path, res,
                                       note=f"group table: {os.path.basename(st['path'])}")
            except Exception as exc:
                messagebox.showerror("Group statistics", str(exc), parent=win)
                return
            lbl_status.configure(text=f"✓  statistics → {os.path.basename(path)}", text_color=GREEN)

        def _save_all_figure() -> None:
            res = res_state["all"]
            path = filedialog.asksaveasfilename(
                parent=win, title="Save the group figure (all pairs)",
                defaultextension=".png", initialfile="group_all_pairs_HR.png",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("PNG image (300 dpi)", "*.png"), ("PDF (vector)", "*.pdf"),
                           ("SVG (vector)", "*.svg"), ("All files", "*.*")])
            if not path:
                return
            fig = Figure(figsize=all_pairs_figsize(len(res.pairs)), dpi=300, facecolor="white",
                         layout="constrained")
            try:
                draw_all_pairs_figure(fig, res, MAIN_METRIC)
                fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
            except Exception as exc:
                messagebox.showerror("Group figure", str(exc), parent=win)
                return
            finally:
                plt.close(fig)
            lbl_status.configure(text=f"✓  figure → {os.path.basename(path)}", text_color=GREEN)

        def _save_stats() -> None:
            if res_state.get("mode") == MODE_ALL and res_state.get("all") is not None:
                _save_all_stats()
                return
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
            if res_state.get("mode") == MODE_ALL and res_state.get("all") is not None:
                _save_all_figure()
                return
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

        def _prism_data() -> "Optional[tuple[list, list[tuple[str, str]], list[str]]]":
            """The tables of the last run (the rows it left out not written),
            the notes saying which rows and why, and the warnings (pairs
            without any row)."""
            name = os.path.basename(res_state.get("path") or st["path"] or "")
            if res_state.get("mode") == MODE_ALL and res_state.get("all") is not None:
                res = res_state["all"]
                notes = [("Group table", name), ("Analysis", MODE_ALL),
                         ("Rows to check", "left out" if st["exclude"] else "kept (listed)")]
                for pr in res.pairs:
                    notes.append((f"Rows written — {pr.title}", str(len(pr.used))))
                    notes += left_out_notes(pr.checks, pr.keep, pr.title)
                notes += [("Note", n) for n in res.notes]
                used_all = [pr.used for pr in res.pairs if pr.used is not None and len(pr.used)]
                notes += sex_notes(pd.concat(used_all, ignore_index=True) if used_all else None)
                empty = [pr.title for pr in res.pairs if pr.used is not None and not len(pr.used)]
                warns = ([f"no animal in pair {', '.join(empty)}: its tables are empty"]
                         if empty else [])
                return pair_tables(res), notes, warns
            used, df = res_state.get("used"), res_state.get("df")
            if used is None or df is None:
                return None
            notes = [("Group table", name),
                     ("Analysis", f"{MODE_ONE}: {st['a']} vs {st['b']}; every segment written"),
                     ("Rows to check", "left out" if st["exclude"] else "kept (listed)"),
                     ("Rows written", str(len(used)))]
            notes += left_out_notes(res_state["checks"], res_state["keep"])
            notes += sex_notes(used)
            warns = [] if len(used) else ["no animal kept: the tables are empty"]
            return group_tables(used, tags=segment_tags(df)), notes, warns

        def _group_base() -> str:
            path = res_state.get("path") or st["path"] or ""
            return os.path.splitext(os.path.basename(path))[0] or "group"

        def _prism_said(text: str, ok: bool = True, warn: bool = False) -> None:
            """The Prism result on its own line (the analysis message stays)."""
            if not lbl_prism.winfo_manager():
                lbl_prism.pack(fill="x", after=lbl_status)
            lbl_prism.configure(text=("✓  " if ok and not warn else "⚠  ") + text,
                                text_color=RED if not ok else ORANGE if warn else GREEN)

        def _prism_new() -> None:
            got = _prism_data()
            if got is None:
                return
            tables, notes, warns = got
            path = filedialog.asksaveasfilename(
                parent=win, title="Export the group table to GraphPad Prism",
                defaultextension=".pzfx", initialfile=f"{_group_base()}.pzfx",
                initialdir=os.path.dirname(st["path"]) or None,
                filetypes=[("GraphPad Prism", "*.pzfx"), ("All files", "*.*")])
            if not path:
                return
            try:
                n = write_metric_pzfx(tables, path, notes=notes)
            except Exception as exc:
                log.exception("group Prism export")
                messagebox.showerror("Prism export", str(exc), parent=win)
                _prism_said(f"Prism export failed: {exc}", ok=False)
                return
            _prism_said(f"Prism: {n} tables → {os.path.basename(path)}"
                        + "".join(f"  ·  {w}" for w in warns), warn=bool(warns))

        def _prism_fill() -> None:
            got = _prism_data()
            if got is None:
                return
            tables, notes, warns = got
            tpl = ask_prism_template(self, win)
            if not tpl:
                return
            out = ask_filled_path(win, tpl, _group_base())
            if not out:
                return
            try:
                report = fill_template(tpl, tables, out,
                                       notes=notes + [("Prism template", os.path.basename(tpl))])
            except Exception as exc:
                if not isinstance(exc, PrismTemplateError):
                    log.exception("group Prism template")
                messagebox.showerror("Prism template", str(exc), parent=win)
                _prism_said(f"Prism template: {exc}", ok=False)
                return
            _prism_said(f"Prism template {os.path.basename(tpl)}: {report.summary()} → "
                        f"{os.path.basename(out)}" + "".join(f"  ·  {w}" for w in warns),
                        warn=report.warn or bool(warns))

        def _prism_export() -> None:
            prism_menu(win, btn_prism, _prism_new, _prism_fill)

        btn_prism = ctk.CTkButton(hdr, text="Prism…", width=0, command=_prism_export,
                                  state="disabled",
                                  **{**button_style("secondary"), **disabled_style()})
        btn_prism.pack(side="right", padx=(SPACE_S, 0))
        prism_tip(self, btn_prism,
                  "GraphPad Prism: one table per metric, one row per animal and one column "
                  "per segment (all pairs: one table per pair × metric); the rows left out "
                  "by this analysis are not written (the notes say which and why).")
        btn_fig = ctk.CTkButton(hdr, text="Figure…", width=0, command=_save_figure,
                                state="disabled", **{**button_style("secondary"), **disabled_style()})
        btn_fig.pack(side="right", padx=(SPACE_S, 0))
        tip(btn_fig, "The group Poincaré (one ellipse per animal, the group mean thick) and "
                     "the before-after panels; all pairs: one Mean HR panel per pair (one line "
                     "per animal, the pair's p). 300 dpi PNG or vector PDF / SVG")
        btn_xlsx = ctk.CTkButton(hdr, text="📊  Statistics…", width=0, command=_save_stats,
                                 state="disabled", **{**button_style("secondary"), **disabled_style()})
        btn_xlsx.pack(side="right", padx=(SPACE_S, 0))
        tip(btn_xlsx, "Excel: the paired tests (Wilcoxon, paired t, Holm), the rows checked "
                      "and the paired values of every metric, one row per animal; all pairs: "
                      "a Summary sheet (pairs × metrics) and one sheet per pair")
        ctk.CTkLabel(body, text=(
            "One row per animal: the test unit is the animal. Use the same windows and labels for "
            "every animal. Holm corrects the Wilcoxon p over the metrics listed; pick your primary "
            "metric in advance."), font=FONT_HINT, text_color=MUTED, anchor="w",
            justify="left").pack(fill="x", side="bottom", pady=(SPACE_S, 0))
        if st["path"] and os.path.exists(st["path"]) and _load(st["path"], quiet=True):
            _run()
