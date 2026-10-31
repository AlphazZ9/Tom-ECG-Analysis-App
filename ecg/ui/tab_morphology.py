# -*- coding: utf-8 -*-
"""
ecg.ui.tab_morphology
---------------------
ECGApp Morphology & Intervals tab: one flow in three steps (mean beat, landmarks
on it, measure every beat), the numeric table next to the % measured, the QTc-RR
diagnostic, and the distribution plots behind a disclosure. Numbers come from the
shared table (ecg.core.metric_table.interval_rows); step states from freshness.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
import customtkinter as ctk  # type: ignore[import-untyped]
from ecg.core.analysis import INTERVAL_DEFINITIONS
from ecg.core.metric_table import (
    INTERVAL_COLUMNS, MISSING, MetricTable, interval_rows, intervals_measured,
)
from ecg.core.models import QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.core.wave_template import WaveTemplate
from ecg.ui.freshness import NOT_COMPUTED, UP_TO_DATE, ModuleFreshness, stale_banner_text
from ecg.ui.tabbar import STATE_GLYPHS, state_color
from ecg.ui.theme import (
    THEME, BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, ORANGE, GREEN, RED, PURPLE, BLUE,
    PURPLE_DARK, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_LABEL, FONT_SMALL,
    FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT, FONT_SECTION_HDR,
)
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_MORPHOLOGY
from ecg.ui.wave_editor import WaveTemplateMiniEditor, dragged_landmark, hit_landmark

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

_DEFINITIONS = dict(INTERVAL_DEFINITIONS)
_DIST_OPEN = "▾  Distribution plots"
_DIST_CLOSED = "▸  Distribution plots"
DRAG_HINT = "Drag a landmark to adjust its search window"


@dataclass(frozen=True)
class MorphStep:
    number: int
    text: str
    state: str
    reasons: "tuple[str, ...]" = ()
    current: bool = False     # the step the user is on: highlighted


@dataclass(frozen=True)
class IntervalLine:
    name: str
    definition: str
    median: str
    iqr: str
    measured: str


def morphology_steps(core: ModuleFreshness, intervals: ModuleFreshness,
                     n_beats: "Optional[int]") -> "list[MorphStep]":
    """The three steps. The first two rest on the core results (the mean beat and
    the template drawn on it), the third on the intervals: a template applied
    since the last measure makes step 3 out of date, through its own freshness. The
    current step is the first with something left to do: nothing records a check of the
    landmarks, so step 2 lasts until the beats are measured with them."""
    beats = f" · {n_beats} beats extracted" if n_beats else ""
    on_core = core.state != UP_TO_DATE
    on_landmarks = not on_core and intervals.state != UP_TO_DATE
    return [
        MorphStep(1, f"Mean beat{beats}", core.state, core.reasons, on_core),
        MorphStep(2, "Check landmarks on the template", core.state, core.reasons, on_landmarks),
        MorphStep(3, "Measure every beat", intervals.state, intervals.reasons),
    ]


def step_tip(step: MorphStep) -> str:
    why = "; ".join(step.reasons)
    tip = f"Step {step.number}: {step.state}" + (f" — {why}" if why else "")
    if step.number == 2 and step.current:
        tip += " — check the landmarks on the mean beat, then Measure intervals"
    return tip


def measured_headline(ivl: Any) -> "tuple[str, str]":
    """('16/1748 beats measured (1%) on PR, QRS and QT', tier of that share): what
    the medians next to it rest on, in the words and grades of the Intervals status
    line; each interval's own count is in its line. ('', '') when not delineated."""
    if not MetricTable(interval_rows(ivl)).has("intervals.beats_measured"):
        return "", ""
    n_ok, total = intervals_measured(ivl)
    pct = 100.0 * n_ok / total if total else 0.0
    tier = "good" if pct >= 90 else "partial" if pct >= 50 else "low"
    return f"{n_ok}/{total} beats measured ({pct:.0f}%) on PR, QRS and QT", tier


def interval_lines(ivl: Any) -> "list[IntervalLine]":
    """One line per interval, read from the shared table; dashes when there is
    nothing measured. A frame from an older version keeps its old definitions,
    which the ones listed here do not describe."""
    t = MetricTable(interval_rows(ivl))
    lines = []
    for col, name in INTERVAL_COLUMNS:
        row = t.row(f"intervals.{col}")
        definition = _DEFINITIONS.get(name, "")
        if row is None:
            lines.append(IntervalLine(name, definition, MISSING, MISSING, MISSING))
            continue
        if row.note:
            definition = "legacy definition (older session)"
        q1, q3 = t.text(f"intervals.{col}.q1"), t.text(f"intervals.{col}.q3")
        total = row.basis_total or 0
        pct = f"{100.0 * (row.basis_n or 0) / total:.0f}%" if total else MISSING
        lines.append(IntervalLine(
            name, definition, t.text(f"intervals.{col}"),
            f"{q1}–{q3}" if q1 != MISSING else MISSING,
            f"{pct} · {row.basis_n}/{total}"))
    return lines


class LandmarkDrag:
    """Drag a landmark on the mean beat of the tab, with the rules of the Landmarks
    dialog (a line moves the centre, a window edge resizes it). The result goes to this
    recording's template with the provenance of a hand edit; a preview line follows the
    mouse, and the beat is redrawn at the release."""

    GRAB_MS = WaveTemplateMiniEditor._GRAB_R_MS

    def __init__(self, app: "ECGApp") -> None:
        self.app = app
        self.key: "Optional[str]" = None
        self.mode = "center"
        self.x: "Optional[float]" = None
        self.line: Any = None

    def _landmarks(self) -> "dict":
        """The windows drawn: this recording's, or the default it would start from."""
        wt = self.app.analysis.wave_template
        if wt is None:
            wt, _notice = WaveTemplate.load_default_checked(quiet=True)
        return wt.landmarks

    def on_press(self, event: Any) -> None:
        a = self.app.analysis
        if (event.inaxes is None or event.xdata is None or event.button != 1 or not a.results
                or "intervals" in self.app.ui.running_modules):
            return
        self.key, self.mode = hit_landmark(self._landmarks(), WaveTemplateMiniEditor._KEYS,
                                           float(event.xdata), self.GRAB_MS)
        self.x = None
        if self.key is not None:
            color = WaveTemplateMiniEditor.WAVE_COLORS.get(self.key, BLUE)
            self.line = event.inaxes.axvline(float(event.xdata), color=color, lw=2.0, zorder=9)

    def on_motion(self, event: Any) -> None:
        if self.key is None or event.xdata is None:
            return
        self.x = float(event.xdata)
        self.line.set_xdata([self.x, self.x])
        self.line.figure.canvas.draw_idle()
        c, hw = self._landmarks()[self.key]
        c, hw = dragged_landmark(c, hw, self.mode, self.x)
        name = WaveTemplateMiniEditor.WAVE_SHORT_LABELS.get(self.key, self.key)
        self.app.lbl_morph_drag_hint.configure(text=f"{name}: {c:.1f} ms ± {hw:.1f}")

    def on_release(self, event: Any) -> None:
        key, x, line = self.key, self.x, self.line
        self.key, self.x, self.line = None, None, None
        if key is None:
            return
        canvas = line.figure.canvas
        line.remove()
        app = self.app
        app.lbl_morph_drag_hint.configure(text=DRAG_HINT)
        if x is None:
            canvas.draw_idle()
            return
        wt = app.analysis_ctrl.recording_wave_template()
        c, hw = {**WaveTemplate.DEFAULTS, **wt.landmarks}[key]
        wt.landmarks[key] = dragged_landmark(c, hw, self.mode, x)
        # As WaveTemplateMiniEditor._mark_edited and _commit do.
        if wt.source != "edited":
            wt.origin = wt.provenance()
            wt.source = "edited"
        wt.created_at = ""
        wt.confirmed = True
        app._on_template_edited()


class TabMorphologyMixin:
    """ECGApp Morphology & Intervals tab."""

    def _build_tab_morphology(self: ECGApp) -> None:
        """One scrollable page: steps, beat and table, then the disclosure."""
        t = self.tabs.tab(TAB_MORPHOLOGY)
        t.grid_rowconfigure(0, weight=1)
        t.grid_columnconfigure(0, weight=1)
        page = ctk.CTkScrollableFrame(t, fg_color=PANEL, scrollbar_button_color=BORDER,
                                      scrollbar_button_hover_color=BORDER2)
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1)

        self._build_morph_steps(page, 0)

        main = ctk.CTkFrame(page, fg_color="transparent")
        main.grid(row=1, column=0, sticky="ew", padx=SPACE_S)
        main.grid_columnconfigure(0, weight=1, uniform="morph")
        main.grid_columnconfigure(1, weight=1, uniform="morph")
        self._build_morph_beat(main)
        self._build_morph_measure(main)

        self._build_morph_distributions(page, 2)
        self._ivl_shown = self  # no frame is this: the first refresh always draws
        self._refresh_morphology()

    def _build_morph_steps(self: ECGApp, page, row: int) -> None:
        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.grid(row=row, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, SPACE_S))
        self._morph_steps: "list[dict]" = []
        for i in range(3):
            chip = ctk.CTkFrame(bar, fg_color=CARD, corner_radius=6,
                                border_width=1, border_color=BORDER)
            chip.pack(side="left", padx=(0 if i == 0 else SPACE_S, 0))
            glyph = ctk.CTkLabel(chip, text=STATE_GLYPHS[NOT_COMPUTED], font=FONT_LABEL,
                                 width=0, text_color=state_color(NOT_COMPUTED))
            glyph.pack(side="left", padx=(SPACE_M, SPACE_XS), pady=SPACE_XS)
            text = ctk.CTkLabel(chip, text="", font=FONT_SMALL, width=0, text_color=TEXT)
            text.pack(side="left", padx=(0, SPACE_M), pady=SPACE_XS)
            step = {"chip": chip, "glyph": glyph, "text": text, "tip": "", "current": None}
            for w in (chip, glyph, text):
                self._bind_hover_tip(w, lambda s=step: s["tip"])
            self._morph_steps.append(step)
        self.lbl_template_info = ctk.CTkLabel(
            bar, text="", font=FONT_HINT, text_color=MUTED, anchor="e")
        self.lbl_template_info.pack(side="right", padx=(SPACE_M, 0))  # type: ignore[union-attr]

    def _build_morph_beat(self: ECGApp, main) -> None:
        """Steps 1-2: the mean beat with its landmarks and their editor."""
        card = ctk.CTkFrame(main, fg_color=PANEL, corner_radius=0)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_S))
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.pack(side="top", fill="x", padx=SPACE_S, pady=(0, SPACE_XS))
        ctk.CTkLabel(hdr, text="MEAN BEAT ± SD  ·  LANDMARKS",
                     font=FONT_SIDEBAR_HDR, text_color=MUTED).pack(side="left")
        btn_landmarks = ctk.CTkButton(
            hdr, text="⚙  Landmarks",
            command=self._open_wave_template_editor,
            fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
            font=FONT_BTN_SEC, height=28, corner_radius=6,
        )
        btn_landmarks.pack(side="right")
        self.lbl_morph_drag_hint = ctk.CTkLabel(hdr, text=DRAG_HINT, font=FONT_HINT, text_color=MUTED)
        self.lbl_morph_drag_hint.pack(side="right", padx=(0, SPACE_M))
        self._bind_hover_tip(
            btn_landmarks,
            "Edit where P, QRS onset, Q/R/S, the J point and T are searched "
            "for on every beat (PR = P onset → QRS onset, QRS = QRS onset → "
            "J point, QT = QRS onset → T end). The template belongs to this "
            "recording (kept in its session); only “Save as default "
            "template” changes the default for recordings opened later.")
        holder = ctk.CTkFrame(card, fg_color=PANEL, corner_radius=0,
                              height=int(440 * THEME.font_scale))
        holder.pack(side="top", fill="x")
        holder.pack_propagate(False)
        self._slots["beat"] = CanvasSlot(holder, 14, 5.0, toolbar=False)
        self._morph_drag = LandmarkDrag(self)
        canvas = self._slots["beat"].canvas
        canvas.mpl_connect("button_press_event", self._morph_drag.on_press)
        canvas.mpl_connect("motion_notify_event", self._morph_drag.on_motion)
        canvas.mpl_connect("button_release_event", self._morph_drag.on_release)

    def _build_morph_measure(self: ECGApp, main) -> None:
        """Step 3: QTc formula, Measure, the table and the QTc-RR diagnostic."""
        col = ctk.CTkFrame(main, fg_color="transparent")
        col.grid(row=0, column=1, sticky="new")
        col.grid_columnconfigure(0, weight=1)

        bar = ctk.CTkFrame(col, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(bar, text="QTc formula", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(0, SPACE_S))
        # Values come from QTC_FORMULA_LABELS (single source shared with
        # normalize_qtc_formula, session save/restore and the exports);
        # wide enough for the longest label ("Fridericia (QT/∛RR, human)").
        self.cb_qtc_formula = ctk.CTkComboBox(
            bar, width=int(215 * THEME.font_scale), height=28, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, button_color=BORDER2,
            text_color=TEXT, dropdown_fg_color=BG, dropdown_text_color=TEXT,
            values=list(QTC_FORMULA_LABELS.values()),
            command=self._on_qtc_formula_change,
        )
        # The formula in effect lives in app state (analysis.qtc_formula),
        # so a rebuild (theme toggle) re-selects it instead of silently
        # resetting the selector to Mitchell while the stored QTc values
        # (and the exports' formula label) still use another formula.
        self.cb_qtc_formula.set(QTC_FORMULA_LABELS[normalize_qtc_formula(
            getattr(getattr(self, "analysis", None), "qtc_formula", "mitchell"))])
        self.cb_qtc_formula.pack(side="left")

        bar2 = ctk.CTkFrame(col, fg_color="transparent")
        bar2.grid(row=1, column=0, sticky="ew", pady=(SPACE_S, 0))
        # Built with an explicit muted fg_color (see _set_btn_enabled) rather
        # than relying on CTkButton's default disabled treatment, which only
        # dims the text and left this looking clickable while disabled --
        # active fill (_RESULT_BTN_COLORS) is BLUE, not ORANGE, since ORANGE
        # is this app's warning/Edit-Mode-ON colour elsewhere and this only
        # computes measurements, it isn't a warning state.
        self.btn_run_ivl = ctk.CTkButton(
            bar2, text="⚡  Measure intervals",
            command=self._run_intervals,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            font=FONT_BTN_PRIMARY, height=max(28, int(32 * THEME.font_scale)),
            corner_radius=8, state="disabled",
        )
        self.btn_run_ivl.pack(side="left")
        self.sw_permissive = self._switch(bar2, "Permissive bounds", dict(padx=0))
        self.sw_permissive.pack(side="left", padx=(SPACE_L, 0))
        self._bind_hover_tip(
            self.sw_permissive,
            "Widen the physiological PR/QRS/QT bounds ×2 before rejecting a "
            "beat's measurement — trade some specificity for more beats measured.")

        # What every median below rests on, as large as they are.
        self.lbl_ivl_measured = ctk.CTkLabel(
            col, text="", font=FONT_SECTION_HDR, text_color=MUTED, anchor="w")
        self.lbl_ivl_measured.grid(row=2, column=0, sticky="ew", pady=(SPACE_S, 0))
        self.lbl_ivl_status = ctk.CTkLabel(
            col, text="  Click Analyze first", font=FONT_SMALL, text_color=MUTED,
            anchor="w", justify="left", wraplength=int(420 * THEME.font_scale))
        self.lbl_ivl_status.grid(row=3, column=0, sticky="ew")
        self.lbl_ivl_stale = ctk.CTkLabel(
            col, text="", font=FONT_SMALL, text_color=ORANGE, anchor="w", justify="left",
            wraplength=int(420 * THEME.font_scale))
        self.lbl_ivl_stale.grid(row=4, column=0, sticky="ew")
        self.lbl_ivl_stale.grid_remove()

        tbl = ctk.CTkFrame(col, fg_color=CARD, corner_radius=6, border_width=1, border_color=BORDER)
        tbl.grid(row=5, column=0, sticky="ew", pady=(SPACE_S, 0))
        tbl.grid_columnconfigure(0, weight=1)
        for c, px in ((1, 70), (2, 84), (3, 118)):
            tbl.grid_columnconfigure(c, minsize=int(px * THEME.font_scale))
        for c, head in enumerate(("Interval", "Median (ms)", "IQR (ms)", "Measured")):
            ctk.CTkLabel(tbl, text=head, font=FONT_HINT, text_color=MUTED,
                         anchor="w" if c == 0 else "e").grid(
                row=0, column=c, sticky="ew",
                padx=(SPACE_M if c == 0 else SPACE_S, SPACE_M if c == 3 else 0))
        self._ivl_cells: "list[dict]" = []
        for r, (_col, name) in enumerate(INTERVAL_COLUMNS, start=1):
            cell_name = ctk.CTkFrame(tbl, fg_color="transparent")
            cell_name.grid(row=r, column=0, sticky="ew", padx=(SPACE_M, 0), pady=(0, SPACE_XS))
            ctk.CTkLabel(cell_name, text=name, font=FONT_LABEL, text_color=TEXT,
                         anchor="w", height=18).pack(fill="x")
            definition = ctk.CTkLabel(cell_name, text="", font=FONT_HINT, text_color=MUTED,
                                      anchor="w", justify="left", height=14,
                                      wraplength=int(170 * THEME.font_scale))
            definition.pack(fill="x")
            cells = {"definition": definition}
            for c, key in enumerate(("median", "iqr", "measured"), start=1):
                cells[key] = ctk.CTkLabel(tbl, text=MISSING, font=FONT_LABEL,
                                          text_color=TEXT, anchor="e")
                cells[key].grid(row=r, column=c, sticky="ew",
                                padx=(SPACE_S, SPACE_M if c == 3 else 0))
            self._ivl_cells.append(cells)

        ctk.CTkLabel(col, text="QTc vs RR  ·  a flat line means the correction removed "
                               "the rate dependence",
                     font=FONT_HINT, text_color=MUTED, anchor="w", justify="left",
                     wraplength=int(420 * THEME.font_scale),
                     ).grid(row=6, column=0, sticky="ew", pady=(SPACE_S, 0))
        holder = ctk.CTkFrame(col, fg_color=PANEL, corner_radius=0,
                              height=int(170 * THEME.font_scale))
        holder.grid(row=7, column=0, sticky="ew")
        holder.pack_propagate(False)
        self._slots["qtc_rr"] = CanvasSlot(holder, 5, 1.7, toolbar=False,
                                           placeholder="Click Measure intervals to display")
        self.lbl_qtc_rr = ctk.CTkLabel(col, text="", font=FONT_HINT, text_color=MUTED, anchor="w")
        self.lbl_qtc_rr.grid(row=8, column=0, sticky="ew")

    def _build_morph_distributions(self: ECGApp, page, row: int) -> None:
        """The violins and the amplitude / correlation histograms, closed until
        asked for; they are drawn all the same, the exports read their figures."""
        self.btn_morph_dist = ctk.CTkButton(
            page, text=_DIST_CLOSED, anchor="w", command=self._toggle_morph_distributions,
            fg_color="transparent", hover_color=BORDER, text_color=MUTED,
            font=FONT_BTN_SEC, height=28, corner_radius=6)
        self.btn_morph_dist.grid(row=row, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, 0))
        self._morph_dist = ctk.CTkFrame(page, fg_color="transparent")
        self._morph_dist.grid(row=row + 1, column=0, sticky="ew", padx=SPACE_S, pady=(0, SPACE_S))
        for slot, w, h, px, placeholder in (
                ("intervals", 14, 6.0, 380, "Click Measure intervals to display"),
                ("beat_dist", 14, 4.0, 280, "Click Analyze to display")):
            holder = ctk.CTkFrame(self._morph_dist, fg_color=PANEL, corner_radius=0,
                                  height=int(px * THEME.font_scale))
            holder.pack(side="top", fill="x", pady=(SPACE_XS, 0))
            holder.pack_propagate(False)
            self._slots[slot] = CanvasSlot(holder, w, h, toolbar=False, placeholder=placeholder)
        self._apply_morph_distributions()

    def _toggle_morph_distributions(self: ECGApp) -> None:
        self.ui.morph_dist_open = not self.ui.morph_dist_open
        self._apply_morph_distributions()

    def _apply_morph_distributions(self: ECGApp) -> None:
        is_open = self.ui.morph_dist_open
        self.btn_morph_dist.configure(text=_DIST_OPEN if is_open else _DIST_CLOSED)
        if is_open:
            self._morph_dist.grid()
        else:
            self._morph_dist.grid_remove()

    # ── derived on the shell's tick, like the tab's own state ─────────────

    def _refresh_morphology(self: ECGApp) -> None:
        """Steps, table and stale line. The table is rebuilt only when the frame
        changes (a job, a formula change, a drop): the tick must stay cheap."""
        if getattr(self, "_morph_steps", None) is None:
            return
        fresh, r = self.freshness, self.analysis.results or {}
        # beat_corr has a value per beat extracted; beat_matrix keeps ≤60 of them.
        beats = r.get("beat_corr")
        n_beats = len(beats) if beats is not None else None
        ivl_status = fresh.status("intervals")
        for widgets, step in zip(self._morph_steps,
                                 morphology_steps(fresh.status("core"), ivl_status, n_beats)):
            widgets["tip"] = step_tip(step)
            text = f"{step.number}   {step.text}"
            if widgets["text"].cget("text") != text:
                widgets["text"].configure(text=text)
            if widgets["current"] != step.current:
                widgets["current"] = step.current
                widgets["chip"].configure(border_color=BLUE if step.current else BORDER,
                                          border_width=2 if step.current else 1)
            glyph = "●" if step.current else STATE_GLYPHS[step.state]
            if widgets["glyph"].cget("text") != glyph:
                widgets["glyph"].configure(
                    text=glyph, text_color=BLUE if step.current else state_color(step.state))
        ivl = r.get("intervals")
        if ivl is not self._ivl_shown:
            self._ivl_shown = ivl
            self._show_interval_table(ivl)
        banner = stale_banner_text("Intervals", ivl_status,
                                   self.ui.module_computed_at.get("intervals"),
                                   self.ui.peaks_edited_at,
                                   has_values=ivl is not None and "QT_ms" in ivl.columns)
        if self.lbl_ivl_stale.cget("text") != banner:
            self.lbl_ivl_stale.configure(text=banner)
            if banner:
                self.lbl_ivl_stale.grid()
            else:
                self.lbl_ivl_stale.grid_remove()

    def _show_interval_table(self: ECGApp, ivl: Any) -> None:
        text, tier = measured_headline(ivl)
        self.lbl_ivl_measured.configure(
            text=text, text_color={"good": GREEN, "partial": ORANGE, "low": RED}.get(tier, MUTED))
        for cells, line in zip(self._ivl_cells, interval_lines(ivl)):
            cells["definition"].configure(text=line.definition)
            cells["median"].configure(text=line.median)
            cells["iqr"].configure(text=line.iqr)
            cells["measured"].configure(text=line.measured)

    # ════════════════════════════════════════════════════════
    #  QTC FORMULA
    # ════════════════════════════════════════════════════════

    def _qtc_formula(self: ECGApp) -> str:
        """Return the QTc formula key in effect ('mitchell', 'bazett' or
        'fridericia' — see QTC_FORMULA_LABELS): analysis.qtc_formula, set by
        the Morphology & Intervals tab combo, which survives a UI rebuild."""
        return self.analysis_ctrl.qtc_formula()

    def _on_qtc_formula_change(self: ECGApp, _choice: str = "") -> None:
        """Re-compute QTc with the selected formula and refresh interval plots."""
        self.analysis_ctrl.on_qtc_formula_change(_choice)

    def _update_interval_plots(self: ECGApp) -> None:
        """Redraw the interval violin plots after a QTc formula change."""
        self.analysis_ctrl.update_interval_plots()
