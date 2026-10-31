# -*- coding: utf-8 -*-
"""
ecg.ui.context_strip
--------------------
The strip above every tab (what the numbers on screen rest on: window, beats,
RR bounds, exclusions, artifact correction, QTc formula, computed and edited
times), and the per-tab states on the tab bar.

Everything is derived from existing state on a short tick rather than pushed
from each code path: window, bounds and formula can change through many
widgets, and a missed hook would leave a stale state on screen. A refresh only
reads; nothing here computes a metric. Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import datetime as dt
import logging
import tkinter as tk
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional

from ecg.core.analysis import QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.core.metric_table import MISSING, MetricTable, build_metric_table
from ecg.ui.freshness import COMPUTING, NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE, combine_states
from ecg.ui.tab_keys import TAB_DETECTION, TAB_MODULES, TAB_REPORT
from ecg.ui.theme import THEME, BG, BORDER, TEXT, MUTED, BLUE, ORANGE, SPACE_S, SPACE_M

# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")

REFRESH_MS = 500
# Ticks without a job before a chain that stopped no longer shows as computing:
# long enough for the plots a job draws before the next one starts.
_RUNNING_GRACE_TICKS = 10
_ROW_H = 22

# A view's own status line -> the module it reports on.
_STATUS_LINES = (("lbl_freq_status", "frequency"), ("lbl_nonlin_status", "nonlinear"),
                 ("lbl_ivl_status", "intervals"), ("lbl_roll_status", "rolling"),
                 ("lbl_epoch_count", "epochs"), ("lbl_auto_status", "autonomic"),
                 ("lbl_evt_status", "event_locked"), ("lbl_arrhythmia_status", "events"))

TONE_HEAD = "head"
TONE_NORMAL = "normal"
TONE_WARN = "warn"
TONE_MUTED = "muted"


@dataclass(frozen=True)
class StripField:
    key: str
    text: str
    tone: str = TONE_NORMAL


def _font(bold: bool = False) -> tuple:
    # Plain Tk labels here (cheap to reflow); a negative size is in pixels, the
    # unit the CTk fonts around them use.
    size = -int(round(12 * THEME.font_scale))
    return (THEME.font_family, size, "bold") if bold else (THEME.font_family, size)


def _number(v: float) -> str:
    return f"{float(v):.1f}".rstrip("0").rstrip(".")


def window_text(window: Any, signal_end_s: "Optional[float]") -> str:
    """'0–300 s': *window* as stored, 0 meaning signal start / end."""
    t0, t1 = window if window is not None else (0.0, 0.0)
    t0 = float(t0 or 0.0)
    t1 = float(t1 or 0.0) or signal_end_s
    if t1 is None:
        return MISSING
    return f"{_number(t0)}–{_number(t1)} s"


def exclusion_field(t: "Optional[MetricTable]") -> StripField:
    """How many RR intervals HRV left out and why, from the table's rr.* rows."""
    if t is None or not t.has("rr.n_excluded"):
        return StripField("excluded", f"RR excluded: {MISSING}", TONE_MUTED)
    n = int(t.value("rr.n_excluded"))
    if n <= 0:
        return StripField("excluded", "0 RR excluded")
    parts = [f"{int(t.value(key, 0))} {name}" for key, name in (
        ("rr.n_too_short", "too short"), ("rr.n_too_long", "too long"),
        ("rr.n_spanning_removed", "spanning a removed beat")) if t.value(key, 0)]
    parts.append(f"{float(t.value('rr.excluded_pct', 0.0)):.2f} %")
    return StripField("excluded", f"{n} RR excluded ({' · '.join(parts)})", TONE_WARN)


def artifact_field(t: MetricTable) -> StripField:
    if not t.has("artifact.removed"):
        return StripField("artifact", "Artifact correction: not applied")
    n = int(t.value("artifact.removed"))
    return StripField("artifact", "Artifact correction: "
                      + (f"{n} beat{'s' if n != 1 else ''} removed" if n else "no beat removed"))


def times_text(computed_at: "Optional[str]", edited_at: "Optional[str]",
               now: "Optional[dt.datetime]" = None) -> str:
    """'Computed 14:32 · peaks edited 14:35'; the date is added when not today."""
    now = now or dt.datetime.now()

    def fmt(iso: "Optional[str]") -> str:
        try:
            when = dt.datetime.fromisoformat(str(iso))
        except ValueError:
            return ""
        return when.strftime("%H:%M" if when.date() == now.date() else "%Y-%m-%d %H:%M")

    parts = []
    if computed_at and fmt(computed_at):
        parts.append(f"Computed {fmt(computed_at)}")
    if edited_at and fmt(edited_at):
        parts.append(f"peaks edited {fmt(edited_at)}")
    return " · ".join(parts)


def context_fields(results: "Optional[dict]", table: "Optional[MetricTable]",
                   artifact_table: MetricTable, has_signal: bool, n_peaks: "Optional[int]",
                   window: Any, signal_end_s: "Optional[float]",
                   current_rr: "tuple[float, float]", qtc_formula: str) -> "list[StripField]":
    """The strip's fields. With core results they say what those were computed
    on; without, what the next Analyze would use."""
    if not has_signal:
        head = "No recording"
    else:
        head = "Analysed" if results is not None else "Not analysed"
    if results is not None and table is not None and table.has("hr.n_beats"):
        n_peaks = int(table.value("hr.n_beats"))
    lo, hi = current_rr
    if table is not None and table.has("rr.rr_min_ms"):
        lo, hi = table.value("rr.rr_min_ms"), table.value("rr.rr_max_ms")
    return [
        StripField("head", head, TONE_HEAD),
        StripField("window", f"Window {window_text(window, signal_end_s) if has_signal else MISSING}"),
        StripField("beats", f"{n_peaks if n_peaks is not None else MISSING} peaks"),
        StripField("rr", f"RR bounds {float(lo):.1f}–{float(hi):.1f} ms"),
        exclusion_field(table),
        artifact_field(artifact_table),
        StripField("qtc", "QTc: " + QTC_FORMULA_LABELS[normalize_qtc_formula(qtc_formula)]),
    ]


def tab_states(states: "dict[str, str]", has_peaks: bool,
               running: "Iterable[str]" = ()) -> "dict[str, str]":
    """Tab -> state. Signal & Detection judges the peaks themselves: they are
    current as soon as they exist, whatever was computed from them. A tab with
    a module in *running* is computing, whatever its parts say; Report holds
    them all, so it is computing as soon as one of them is."""
    running = set(running)
    out = {TAB_DETECTION: UP_TO_DATE if has_peaks else NOT_COMPUTED}
    for tab, modules in TAB_MODULES.items():
        out[tab] = (COMPUTING if running.intersection(modules)
                    else combine_states(states[m] for m in modules))
    if running:
        out[TAB_REPORT] = COMPUTING
    return out


class ContextStrip(tk.Frame):
    """Fields left to right, wrapping onto further rows when they do not fit;
    the times docked right."""

    def __init__(self, parent) -> None:
        super().__init__(parent, bg=BG, bd=0, highlightthickness=0)
        tk.Frame(self, height=1, bg=BORDER).pack(side="bottom", fill="x")
        self._times = tk.Label(self, text="", bg=BG, fg=MUTED, font=_font())
        self._times.pack(side="right", anchor="n", padx=(SPACE_S, SPACE_M), pady=2)
        self._flow = tk.Frame(self, bg=BG, height=_ROW_H, bd=0, highlightthickness=0)
        self._flow.pack(side="left", fill="x", expand=True, padx=(SPACE_M, 0), pady=2)
        self._flow.bind("<Configure>", lambda _e: self._reflow())
        self._labels: "dict[str, tk.Label]" = {}
        self._fields: "list[StripField]" = []

    @staticmethod
    def _color(tone: str) -> str:
        return {TONE_HEAD: BLUE, TONE_WARN: ORANGE, TONE_MUTED: MUTED}.get(tone, TEXT)

    def show(self, fields: "list[StripField]", times: str) -> None:
        if fields != self._fields:
            self._fields = list(fields)
            for f in fields:
                lbl = self._labels.get(f.key)
                if lbl is None:
                    lbl = self._labels[f.key] = tk.Label(self._flow, bg=BG, bd=0)
                lbl.configure(text=f.text, fg=self._color(f.tone), font=_font(f.tone == TONE_HEAD))
            self._reflow()
        if self._times.cget("text") != times:
            self._times.configure(text=times)
            self.after_idle(self._reflow)

    def text(self) -> str:
        return " | ".join(f.text for f in self._fields)

    def times(self) -> str:
        return self._times.cget("text")

    def _reflow(self) -> None:
        width = self._flow.winfo_width()
        if width <= 1:
            return
        # At least one label's own height: the Theme dialog scales the font.
        row_h = max([_ROW_H] + [self._labels[f.key].winfo_reqheight() for f in self._fields])
        x = y = 0
        for f in self._fields:
            lbl = self._labels[f.key]
            w = lbl.winfo_reqwidth()
            if x and x + w > width:
                x, y = 0, y + row_h
            lbl.place(x=x, y=y, height=row_h)
            x += w + SPACE_M + SPACE_S
        if int(self._flow.cget("height")) != y + row_h:
            self._flow.configure(height=y + row_h)


class ContextStripMixin:
    """ECGApp: the context strip and the per-tab states."""

    _status_lines_saved: "dict[str, tuple]" = {}

    def _build_context_strip(self: ECGApp, parent) -> None:
        self._status_lines_saved = {}
        self.context_strip = ContextStrip(parent)
        self.context_strip.pack(side="top", fill="x")

    def _bind_tab_state_hints(self: ECGApp) -> None:
        """Hovering a tab says its state and why, in the status bar; what that
        one line cannot hold opens in full under the tab."""
        for name in self.tabs.names():
            for w in self.tabs.head_widgets(name):
                self._bind_hover_tip(w, lambda n=name: self.tabs.state_text(n))
                w.bind("<Enter>", lambda _e, n=name: self._show_tab_hint(n))
                w.bind("<Leave>", lambda _e: self.tabs.hide_hint())

    def _show_tab_hint(self: ECGApp, name: str) -> None:
        text = self.tabs.state_text(name)
        if self._fit_status_text(text) != text:
            self.tabs.show_hint(name)

    def _start_shell_tick(self: ECGApp) -> None:
        self._refresh_shell()
        self.after(REFRESH_MS, self._start_shell_tick)

    def _refresh_shell(self: ECGApp) -> None:
        """Strip, tab states and the quality panel's live fields. Never raises:
        it runs on a timer, possibly while a job is landing."""
        try:
            self._refresh_tab_states()
            self._refresh_context_strip()
            self._refresh_quality_live()
            self._refresh_time_banner()
            self._refresh_morphology()
            self._refresh_events_readout()
            self._refresh_status_lines()
            self._update_topbar_filename()
            self._refresh_rhythm_stale()
            self._refresh_report()
        except Exception as exc:
            log.debug("shell refresh skipped: %s", exc, exc_info=True)

    def _refresh_tab_states(self: ECGApp) -> None:
        fresh = self.freshness
        statuses = {m: fresh.status(m) for m in fresh.states()}
        rp = self.detection.rpeaks_ok
        running = self._running_modules()
        per_tab = tab_states({m: s.state for m, s in statuses.items()},
                             rp is not None and len(rp) > 0, running)
        for tab, state in per_tab.items():
            reasons: "tuple[str, ...]" = ()
            for m in TAB_MODULES.get(tab, ()):
                if m in running:
                    reasons += (f"{m.replace('_', ' ')}: computing",)
                reasons += tuple(f"{m.replace('_', ' ')}: {why}" for why in statuses[m].reasons)
                # One state per tab: its hint says which parts that state leaves out.
                if state == UP_TO_DATE and statuses[m].state == NOT_COMPUTED:
                    reasons += (f"{m.replace('_', ' ')}: not computed",)
            self.tabs.set_state(tab, state, reasons)

    def _refresh_status_lines(self: ECGApp) -> None:
        """A view's own status line says what its last run gave. While that module is
        out of date the state on the tab speaks alone: the line is emptied, and comes
        back when the state does (an undo), unless a job or a reset wrote another."""
        saved = self._status_lines_saved
        running = self.ui.running_modules
        for attr, module in _STATUS_LINES:
            lbl = getattr(self, attr, None)
            if lbl is None:
                continue
            stale = self.freshness.status(module).state == OUT_OF_DATE and module not in running
            text = lbl.cget("text")
            if stale and text:
                saved[attr] = (text, lbl.cget("text_color"))
                lbl.configure(text="")
            elif not stale and not text and attr in saved:
                text, color = saved.pop(attr)
                lbl.configure(text=text, text_color=color)
            elif not stale and text:
                saved.pop(attr, None)

    def _running_modules(self: ECGApp) -> "set[str]":
        """ui.running_modules, dropped once no job has run for a while: a chain
        stops without saying so when a job fails or the busy guard refuses it."""
        ui = self.ui
        if ui.running_modules:
            if getattr(self, "_async_busy", False):
                ui.running_idle_ticks = 0
            else:
                ui.running_idle_ticks += 1
                if ui.running_idle_ticks > _RUNNING_GRACE_TICKS:
                    ui.running_modules.clear()
        return ui.running_modules

    def _context_table(self: ECGApp) -> "tuple[Optional[MetricTable], MetricTable]":
        """(table of the core results, table of the removals still in effect,
        as a session saves them), rebuilt only when the results, the report or
        the peaks change: the tick must stay cheap. rpeaks_ok is reassigned,
        never mutated in place (DetectionState)."""
        a = self.analysis
        r, rep, rp = a.results, a.artifact_report, self.detection.rpeaks_ok
        cached = getattr(self, "_context_table_cache", None)
        if cached is not None and cached[0] is r and cached[1] is rep and cached[2] is rp:
            return cached[3], cached[4]
        table = build_metric_table(r, asymmetry=None) if r is not None else None
        art = build_metric_table(None, artifact_report=self.analysis_ctrl.current_artifact_report())
        self._context_table_cache = (r, rep, rp, table, art)
        return table, art

    def _refresh_context_strip(self: ECGApp) -> None:
        a, sig = self.analysis, self.signal
        table, art = self._context_table()
        peaks = self._windowed_peaks()
        ctrl = self.plot_ctrl
        fields = context_fields(
            a.results, table, art, has_signal=sig.filtered is not None or sig.raw is not None,
            n_peaks=len(peaks) if peaks is not None else None,
            window=ctrl._report_window() if a.results is not None else (a.t_start, a.t_end),
            signal_end_s=ctrl._signal_end_s(),
            current_rr=self.analysis_ctrl.hrv_rr_bounds(quiet=True),
            qtc_formula=self._qtc_formula())
        self.context_strip.show(fields, times_text(self.ui.computed_at, self.ui.peaks_edited_at))
