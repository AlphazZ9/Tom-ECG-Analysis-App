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
from ecg.core.metric_families import by_family, family_of
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
from ecg.ui.theme import button_style, disabled_style, light_plot_theme, publication_style, toggle_style
from ecg.ui.poincare_figure import draw_poincare_set
from ecg.io.group_table import append_group_row, group_row
from ecg.io.subject_name import normalise_sex

from ecg.io.segment_export import (
    ComparisonResult, comparison_tsv, pzfx_notes, write_csv, write_pzfx, write_xlsx,
)
from ecg.io import prism_template
from ecg.io.prism_template import PrismTemplateError, fill_template, recording_tables
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


def own_line_stripes(text: "tk.Text", tag: str = "stripe") -> None:
    """Give each line's end the zebra *tag* of the line itself. The rows are
    inserted as "\\n" + row, so a line's newline (the part that paints to
    the right edge) carried the next row's tags: a family header then showed
    the stripe of its first metric to the right of its title."""
    last = int(text.index("end-1c").split(".")[0])
    for ln in range(1, last):
        end = f"{ln}.end"
        text.tag_remove(tag, end)
        if tag in text.tag_names(f"{ln}.0"):
            text.tag_add(tag, end)


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
            ax.legend(framealpha=0, fontsize=8, labelcolor=PLOT.get("text", "#222"))
    for ax in axes:
        ax.set_xlabel("Time within segment (s)", fontsize=9)
    return axes


# ── Group Poincaré (the group table the Group tab uses) ──────────────────


def _refresh_group_card(app: "Any") -> None:
    """The group Poincaré card brought up to date, when the app has one."""
    refresh = getattr(app, "_compare_group_refresh", None)
    if callable(refresh):
        refresh()

# The two sheets the group Poincaré needs: the TimeCourse sheet of a big
# table took most of the 1.5 s of reading it whole.
_GROUP_SHEETS = ("Animals", "Group")
_GROUP_FIG_SIZE = (14.0, 4.3)
_GROUP_PLOT_H = 430


def segment_choices(seg: "pd.DataFrame") -> "list[tuple[str, str]]":
    """(tag, shown) of each segment of a group table's Group sheet: 'Pre ·
    Baseline' when the group's label is not the tag."""
    from ecg.io.group_stats import segment_labels, segment_tags
    out = []
    for tag in segment_tags(seg):
        label = segment_labels(seg, tag)[0]
        out.append((tag, f"{tag} · {label}" if label and label != tag else tag))
    return out


def segment_color(i: int) -> str:
    """The colour of the *i*-th segment of a comparison (the first, the
    reference, blue): the tachograms, the table and the group Poincaré. Read
    at call time, so a theme change re-synced into this module applies."""
    seq = (BLUE, ORANGE_DARK, GREEN, PURPLE, TEAL, RED)
    return seq[i % len(seq)]


def group_poincare_data(sheets: "dict", a: str, b: str, leave_out: bool = True,
                        by: "Optional[str]" = None) -> "tuple[pd.DataFrame, dict]":
    """The rows of the Group sheet the group Poincaré of A → B draws, with
    the Group tab's rules: unticked animals out; with *by* (the Group tab's
    By: group or sex; None, no groups) the animals without a group or sex
    out, as there; with *leave_out* the rows to check (quality alert,
    subject on several rows, another template or duration:
    ecg.io.group_stats.check_rows) out too; then the animals with a mean RR,
    SD1 and SD2 for both segments. And the counts left out ({"unticked": n,
    "ungrouped": n, "to_check": n, "pooled": bool}). When no ticked animal
    has a group (or a sex) at all, every ticked animal is drawn ("pooled"
    True): the Group tab then asks for groups or No groups, but the Poincaré
    of all the animals is still worth seeing here. ValueError when no animal
    is ticked."""
    from ecg.io import group_compare as gc
    from ecg.io import group_data as gd
    from ecg.io.group_stats import check_rows
    from ecg.ui.group_figure import poincare_rows
    df = sheets[gd.SHEET_SEGMENTS].reset_index(drop=True)
    animals = gd.animals_frame(sheets)
    by = by or gc.BY_NONE
    gmap, _order, _notes = gc.animal_groups(animals, by)
    pooled = False
    if not gmap and by != gc.BY_NONE:
        gmap, _order, _notes = gc.animal_groups(animals, gc.BY_NONE)
        pooled = bool(gmap)
    gc._need_animals(gmap, gc.BY_NONE, animals)
    keys = list(zip(df["subject"].astype(str), df["file"].astype(str)))
    in_table = set(keys)
    ticked = {(str(r["subject"]), str(r["file"])) for _, r in animals.iterrows()
              if bool(r["use"])}
    unticked = sum(1 for _, r in animals.iterrows() if not bool(r["use"])
                   and (str(r["subject"]), str(r["file"])) in in_table)
    ungrouped = sum(1 for k in ticked & in_table if k not in gmap)
    # A row merely without one of the two segments is not "to check": it
    # simply has nothing to draw.
    doubtful = {(c.subject, c.file) for c in check_rows(df, a, b)
                if not c.ok and not all(p.startswith("no segment") for p in c.problems)}
    used = [k in gmap for k in keys]
    keep = [u and not (leave_out and k in doubtful) for u, k in zip(used, keys)]
    # Counted only when it had something to draw: a row with a quality alert
    # on A and no B is not drawn either way.
    to_check = 0
    if leave_out:
        out_rows = df.loc[[u and k in doubtful for u, k in zip(used, keys)]]
        to_check = len(poincare_rows(out_rows, a, b))
    return poincare_rows(df.loc[keep], a, b), {"unticked": unticked, "ungrouped": ungrouped,
                                               "to_check": to_check, "pooled": pooled}


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
            _refresh_group_card(self)           # the table or the pair may have changed
            return
        # Built already for no recording: opening the tab again must not
        # destroy and rebuild it (a visible flash, and the group card's
        # table read again each time).
        if (sig is None and getattr(self, "_compare_built_empty", False)
                and self._compare_built_for is None and holder.winfo_children()):
            _refresh_group_card(self)
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
                                      "(the Group tab works without one).",
                         font=FONT_LABEL, text_color=MUTED).pack(pady=(60, 40))
        else:
            self._build_compare_segments(holder)
        self._build_group_pointer(holder)

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
        # The keys chosen to be ignored (Annotations > Suggest from
        # channels…) are left out of the list; nothing is, by default.
        from ecg.io import marker_prefs
        _sig = getattr(self, "signal", None)
        _markers = marker_prefs.keep_markers(
            getattr(_sig, "markers", None) or {}, getattr(_sig, "marker_codes", None) or {},
            marker_prefs.load_ignored_keys())
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

        _color = segment_color

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
        # A metric family's header, as in the Rhythm & HRV table.
        tbl.tag_configure("family", font=(THEME.font_family, _fsize, "bold"), foreground=BLUE,
                          spacing1=SPACE_S)
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
            # One section per metric family (Heart rate, HRV time domain...),
            # each under its header, the metrics in their order inside.
            n_lines = 1
            sections = by_family(_METRICS, lambda m: family_of(m[1]))
            for fam, items in sections:
                tbl.insert("end", "\n" + fam.upper(), "family")
                n_lines += 1 + len(items)
                for ri, (key, name, unit) in enumerate(items):
                    _metric_line(key, name, unit, ri, segs, comps)
            own_line_stripes(tbl)
            # The headers' space above (spacing1) is not counted in lines:
            # without it the last metric is cut off.
            line_px = tkfont.Font(font=tbl.cget("font")).metrics("linespace") + 4
            n_lines += -(-len(sections) * SPACE_S // max(1, line_px))
            tbl.configure(height=n_lines, state="disabled")

        def _metric_line(key: str, name: str, unit: str, ri: int, segs: "list[dict]",
                         comps: "list[Optional[int]]") -> None:
            """One metric row of the table: name, unit, each segment's value
            and its Δ % against its comparator."""
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
            # An export's or an add's message is about the previous result.
            lbl_exp.configure(text="")
            lbl_exp.pack_forget()
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
            # The exported Poincaré figure has 2 panels a row (a Pre/Post
            # pair side by side), not the 4 of the on-screen view.
            p_cols = max(1, min(len(segs_), 2))
            n_rows = -(-len(segs_) // p_cols)
            n_tacho = len(tachogram_groups(len(segs_), res_.comps))
            written = []
            for suffix, figsize, draw in (
                    ("_tachograms", (10, 4 if n_tacho == 1 else 3 * n_tacho),
                     _rr_drawer(segs_, colors, res_.comps)),
                    ("_poincare", (4 * p_cols, 4.6 * n_rows),
                     lambda fig: draw_poincare_set(fig, segs_, colors, title=_subject(),
                                                   jitter=self._compare_jitter,
                                                   ncols=p_cols))):
                out = base + suffix + ext
                try:
                    # On white with the publication style, whatever the theme.
                    with publication_style():
                        fig = Figure(figsize=figsize, dpi=300, facecolor="white",
                                     layout="constrained")
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
            # The Group tab tests one pair A vs B: unless one was chosen
            # there, start from this comparison's reference vs the first
            # segment compared with it (the group table's tags are the same
            # unique_tags as res_.tags).
            gst = getattr(self, "_grouptab_state", None)
            if gst is not None and not (gst.get("a") and gst.get("b")) and res_.pairs:
                j, k = res_.pairs[0]
                gst["a"], gst["b"] = res_.tags[j], res_.tags[k]
            # The group Poincaré below: this table when the Group tab has none.
            self._cmp_group_added = path
            self._compare_group_refresh(force=True)
            # No count here: the group card below says how many animals the
            # table holds (a count in this line went stale beside it).
            _done(f"{res_.subject or 'recording'} {'updated in' if replaced else 'added to'} "
                  f"{os.path.basename(path)}{extra}")

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

    # ── Group: the group Poincaré; the tests are in the Group tab ────────

    def _build_group_pointer(self: ECGApp, win) -> None:
        """The GROUP — ALL ANIMALS card: the group Poincaré of one pair A → B
        (one thin ellipse per animal, the group mean thick, and the paired
        panels) for the group table the Group tab uses, and the way to the
        Group tab (the animals, every analysis and its exports). The table is
        read when the tab is shown, never at start-up."""
        card = ctk.CTkFrame(win, **card_style())
        card.pack(fill="x", padx=PAGE_PAD_X, pady=(0, PAGE_GAP))
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=CARD_PAD, pady=CARD_PAD - 4)
        tip = getattr(self, "_bind_hover_tip", None) or (lambda *a, **k: None)
        hdr = ctk.CTkFrame(body, fg_color="transparent")
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="GROUP — ALL ANIMALS", font=FONT_SECTION_HDR,
                     text_color=MUTED).pack(side="left")
        ctk.CTkLabel(hdr, text="The tests across animals (one pair or all pairs, by group or "
                               "sex) are in the Group tab: ➕ Add to group table above, then "
                               "open it there.", font=FONT_SMALL, text_color=MUTED,
                     anchor="w").pack(side="left", padx=(SPACE_M, SPACE_M))
        tabs = getattr(self, "tabs", None)
        if tabs is not None:
            from ecg.ui.tab_keys import TAB_GROUP
            btn_open = ctk.CTkButton(hdr, text="Open the Group tab", width=0,
                                     command=self._compare_open_group_tab,
                                     **button_style("secondary"))
            btn_open.pack(side="right")
            tip(btn_open, "The Group tab, on the group table drawn here")
        btn_fig = ctk.CTkButton(hdr, text="Figure…", width=0, state="disabled",
                                command=self._save_compare_group_figure,
                                **{**button_style("secondary"), **disabled_style()})
        btn_fig.pack(side="right", padx=(0, SPACE_S))
        tip(btn_fig, "The group Poincaré (one ellipse per animal, the group mean thick) and the "
                     "before-after panels of the pair shown: 300 dpi PNG or vector PDF / SVG")

        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(SPACE_S, 0))
        ctk.CTkLabel(row, text="Group Poincaré", font=FONT_SMALL, text_color=TEXT).pack(
            side="left", padx=(0, SPACE_M))
        combos = []
        for sep in ("→", ""):
            cb = ctk.CTkComboBox(row, width=170, values=[""], state="readonly",
                                 command=lambda _v: self._compare_group_pair_chosen(),
                                 **combo_style())
            cb.pack(side="left", padx=(0, SPACE_XS))
            combos.append(cb)
            if sep:
                ctk.CTkLabel(row, text=sep, font=FONT_SMALL, text_color=MUTED).pack(
                    side="left", padx=(0, SPACE_XS))
        for cb in combos:
            tip(cb, "The pair drawn, A → B: the Group tab's pair by default, else this "
                    "comparison's reference and the first segment compared with it")
        lbl = ctk.CTkLabel(row, text="", font=FONT_SMALL, text_color=MUTED, anchor="w",
                           justify="left")
        lbl.pack(side="left", fill="x", expand=True, padx=(SPACE_M, 0))
        plot_card = ctk.CTkFrame(body, fg_color=CARD, corner_radius=0, height=_GROUP_PLOT_H)
        plot_card.pack_propagate(False)
        slot = CanvasSlot(plot_card, _GROUP_FIG_SIZE[0], _GROUP_FIG_SIZE[1], toolbar=False,
                          placeholder="")
        # Listed with the app's figures: a draw deferred while the tab was
        # hidden is made when it is shown (_flush_shown_slots).
        slots = getattr(self, "_slots", None)
        if isinstance(slots, dict):
            slots["compare_group"] = slot
        self._cmp_group_ui = {"a": combos[0], "b": combos[1], "lbl": lbl, "fig": btn_fig,
                              "plot": plot_card, "slot": slot, "drawn": None, "tags": {},
                              "shown": None}
        self._compare_group_refresh()

    def _compare_group_path(self: ECGApp) -> str:
        """The group table of the card: the Group tab's (an Excel file open
        there), else the one this tab last added a recording to. No table is
        remembered from an earlier session."""
        gts = getattr(self, "_grouptab_state", None)
        path = (gts or {}).get("path") or ""
        return path or getattr(self, "_cmp_group_added", "") or ""

    def _compare_open_group_tab(self: ECGApp) -> None:
        """The Group tab, on the table this card draws: when the card fell
        back on the table a recording was just added to (the Group tab had
        none open), the Group tab opens that one, so the two tabs agree."""
        from ecg.ui.tab_keys import TAB_GROUP
        path = self._compare_group_path()
        gts = getattr(self, "_grouptab_state", None) or {}
        same = bool(gts.get("path")) and bool(path) and (
            os.path.normcase(os.path.abspath(gts["path"]))
            == os.path.normcase(os.path.abspath(path)))
        if path and not same and os.path.exists(path):
            open_ = getattr(self, "_grouptab_open", None)
            if callable(open_):
                open_(path)
            else:
                # Not built yet: the Group tab reads gts["path"] when it is.
                if getattr(self, "_grouptab_state", None) is None:
                    self._grouptab_state = {}
                self._grouptab_state["path"] = path
        tabs = getattr(self, "tabs", None)
        if tabs is not None:
            tabs.set(TAB_GROUP)

    def _compare_group_sheets(self: ECGApp, path: str) -> "dict":
        """{"key", "sheets": the Animals and Group sheets} of *path*, or
        {"error": why not}; read again only when the file changed (the Group
        tab saves the animals ticked in it)."""
        try:
            st = os.stat(path)
            key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
        except OSError:
            return {"error": f"Group table not found: {os.path.basename(path)}"}
        cache = getattr(self, "_cmp_group_cache", None)
        if cache is not None and cache.get("key") == key:
            return cache
        from ecg.io import group_data as gd
        try:
            cache = {"key": key, "sheets": gd.read_sheets(path, only=_GROUP_SHEETS)}
        except Exception as exc:
            cache = {"key": key, "error": f"{os.path.basename(path)}: {exc}"}
        self._cmp_group_cache = cache
        return cache

    def _compare_group_pair(self: ECGApp, tags: "list[str]") -> "tuple[str, str]":
        """A → B of the card: the one chosen in it (while the Group tab's pair
        stays the one it was chosen against), else the Group tab's, else this
        comparison's reference vs the first segment compared with it (as Add
        to group table), else the first two segments."""
        gts = getattr(self, "_grouptab_state", None) or {}
        group_pair = (gts.get("a") or "", gts.get("b") or "")
        mine = getattr(self, "_cmp_group_choice", None)
        if (mine is not None and mine["seen"] == group_pair and mine["a"] in tags
                and mine["b"] in tags):
            return mine["a"], mine["b"]
        self._cmp_group_choice = None
        if group_pair[0] in tags and group_pair[1] in tags and group_pair[0] != group_pair[1]:
            return group_pair
        current = getattr(self, "_current_comparison", None)
        res = current() if callable(current) else None
        if res is not None and getattr(res, "pairs", None):
            j, k = res.pairs[0]
            a, b = res.tags[j], res.tags[k]
            if a in tags and b in tags and a != b:
                return a, b
        return tags[0], tags[1]

    def _compare_group_pair_chosen(self: ECGApp) -> None:
        ui = getattr(self, "_cmp_group_ui", None)
        if ui is None:
            return
        gts = getattr(self, "_grouptab_state", None) or {}
        self._cmp_group_choice = {"a": ui["tags"].get(ui["a"].get(), ""),
                                  "b": ui["tags"].get(ui["b"].get(), ""),
                                  "seen": (gts.get("a") or "", gts.get("b") or "")}
        self._compare_group_refresh()

    def _compare_group_refresh(self: ECGApp, force: bool = False) -> None:
        """Bring the group Poincaré up to date: the table (read again when it
        changed), the pair, the animals used; drawn again only when one of
        them changed. While the tab is hidden it waits to be shown (opening
        the tab calls this again). *force*: read the table again."""
        ui = getattr(self, "_cmp_group_ui", None)
        if ui is None:
            return
        try:
            if not ui["lbl"].winfo_exists():
                return
        except tk.TclError:
            return
        tabs = getattr(self, "tabs", None)
        try:
            hidden = tabs is not None and tabs.get() != TAB_COMPARE
        except Exception:
            hidden = False
        if hidden:
            return
        if force:
            self._cmp_group_cache = None
        from ecg.io import group_data as gd
        from ecg.ui.group_figure import draw_group_figure

        def _combos(choices: "list[tuple[str, str]]", a: str = "", b: str = "") -> None:
            ui["tags"] = {v: t for t, v in choices}
            shown = dict(choices)
            for cb, t in ((ui["a"], a), (ui["b"], b)):
                cb.configure(values=[v for _t, v in choices] or [""],
                             state="readonly" if choices else "disabled")
                cb.set(shown.get(t, ""))

        def _enable_fig(on: bool) -> None:
            # Its look as well (a bare state change leaves it greyed).
            setter = getattr(self, "_set_btn_enabled", None)
            if callable(setter):
                setter(ui["fig"], on, "secondary")
            else:
                ui["fig"].configure(state="normal" if on else "disabled")

        def _nothing(text: str) -> None:
            ui["lbl"].configure(text=text, text_color=MUTED)
            ui["drawn"] = ui["shown"] = None
            _enable_fig(False)
            if ui["plot"].winfo_manager():
                ui["plot"].pack_forget()

        path = self._compare_group_path()
        got = self._compare_group_sheets(path) if path else {}
        # The shell's tick refreshes the open tab: nothing to do (and the
        # figure export's message kept) while nothing this card shows changed.
        gts = getattr(self, "_grouptab_state", None) or {}
        current = getattr(self, "_current_comparison", None)
        res = current() if callable(current) else None
        inputs = (path, got.get("key"), gts.get("a"), gts.get("b"), gts.get("leave_out", True),
                  gts.get("by"), str(getattr(self, "_cmp_group_choice", None)),
                  (tuple(res.tags), tuple(res.pairs[:1])) if getattr(res, "pairs", None) else None)
        if inputs == ui.get("inputs") and not force:
            return
        ui["inputs"] = inputs
        if not path:
            _combos([])
            _nothing("No group table yet: ➕ Add to group table above, or open one in the "
                     "Group tab.")
            return
        name = os.path.basename(path)
        if "error" in got:
            _combos([])
            _nothing(f"⚠  {got['error']}")
            return
        seg = got["sheets"].get(gd.SHEET_SEGMENTS)
        if seg is None or seg.empty:
            _combos([])
            _nothing(f"{name} has no segments yet (no Group sheet): ➕ Add to group table "
                     "above.")
            return
        choices = segment_choices(seg)
        if len(choices) < 2:
            _combos([])
            _nothing(f"{name} holds fewer than two segments: no pair to draw.")
            return
        a, b = self._compare_group_pair([t for t, _v in choices])
        _combos(choices, a, b)
        if a == b:
            _nothing(f"Choose two different segments · table: {name}")
            return
        leave_out = bool(gts.get("leave_out", True))
        # The Group tab's By (By group by default, as there): an animal it
        # leaves out for having no group or sex is left out here too.
        from ecg.io import group_compare as gc
        by = gts.get("by") or gc.BY_GROUP
        try:
            rows, out = group_poincare_data(got["sheets"], a, b, leave_out, by)
        except ValueError as exc:
            _nothing(f"⚠  {exc} · table: {name}")
            return
        except Exception as exc:
            log.warning("group Poincaré: %s", exc, exc_info=True)
            _nothing(f"⚠  {name}: {exc}")
            return
        extra = []
        if out["unticked"]:
            extra.append(f"{out['unticked']} unticked")
        if out["ungrouped"]:
            extra.append(f"{out['ungrouped']} " + ("of unknown sex" if by == gc.BY_SEX
                                                    else "without a group"))
        if out["to_check"]:
            extra.append(f"{out['to_check']} to check")
        tail = " (left out: " + ", ".join(extra) + ")" if extra else ""
        if out.get("pooled"):
            tail += " · no " + ("sex" if by == gc.BY_SEX else "group") + " set: all ticked animals"
        tail += f" · table: {name}"
        n = len(rows)
        if n < 2:
            _nothing(f"Only {n} animal{'s' if n != 1 else ''} with a mean RR, SD1 and SD2 for "
                     f"both {a} and {b} (2 needed){tail}")
            return
        ui["lbl"].configure(text=f"n = {n} animals{tail}", text_color=MUTED)
        _enable_fig(True)
        if not ui["plot"].winfo_manager():
            ui["plot"].pack(fill="x", pady=(SPACE_S, 0))
        shown = dict(choices)
        names = tuple(shown[t].split(" · ", 1)[-1] for t in (a, b))
        colors = self._compare_group_colors(res, a, b)
        ui["shown"] = (rows, a, b, names, path, colors)
        sig = (got["key"], a, b, leave_out, by, colors, tuple(map(str, rows.index)))
        if sig != ui["drawn"]:
            ui["drawn"] = sig
            ui["slot"].update(lambda fig: draw_group_figure(fig, rows, a, b, names, colors))

    @staticmethod
    def _compare_group_colors(res: "Any", a: str, b: str) -> "tuple[str, str]":
        """A and B in the colours this comparison gives them above (Pre blue,
        Post orange...: a segment keeps its colour down the page); a segment
        it does not hold, in the colour of its place (A blue, B orange)."""
        tags = list(getattr(res, "tags", None) or [])
        return tuple(segment_color(tags.index(t) if t in tags else k)
                     for k, t in enumerate((a, b)))

    def _save_compare_group_figure(self: ECGApp) -> None:
        """The group Poincaré shown, 300 dpi PNG or vector PDF / SVG."""
        ui = getattr(self, "_cmp_group_ui", None)
        if ui is None or ui["shown"] is None:
            return
        from ecg.ui.group_figure import draw_group_figure
        rows, a, b, names, table, colors = ui["shown"]
        parent = ui["lbl"].winfo_toplevel()
        out = filedialog.asksaveasfilename(
            parent=parent, title="Save the group Poincaré",
            defaultextension=".png", initialfile=f"group_poincare_{a}_vs_{b}.png",
            initialdir=os.path.dirname(table) or None,
            filetypes=[("PNG image (300 dpi)", "*.png"), ("PDF (vector)", "*.pdf"),
                       ("SVG (vector)", "*.svg"), ("All files", "*.*")])
        if not out:
            return
        fig = Figure(figsize=_GROUP_FIG_SIZE, dpi=300, facecolor="white", layout="constrained")
        try:
            # On white whatever the theme on screen.
            with publication_style():
                draw_group_figure(fig, rows, a, b, names, colors)
                fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        except Exception as exc:
            messagebox.showerror("Group Poincaré", str(exc), parent=parent)
            return
        finally:
            plt.close(fig)
        ui["lbl"].configure(text=f"✓  figure → {os.path.basename(out)}", text_color=GREEN)
