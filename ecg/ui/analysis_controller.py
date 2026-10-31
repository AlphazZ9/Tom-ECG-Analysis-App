# -*- coding: utf-8 -*-
"""
analysis_controller.py
------------------------
AnalysisController -- HRV time/frequency/non-linear analysis, interval
delineation, arrhythmia classification and its ECG-viewer edit mode,
rolling-window and epoch HRV, the time-resolved autonomic indices (Time course >
Autonomic: CSI / CPI, optional SPWVD LF / HF), and the QTc-formula recompute.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
import pandas as pd

from ecg.core.models import (
    MouseECG, ArrhythmiaEvent, QTC_FORMULA_LABELS, normalize_qtc_formula,
    PREMATURE_KINDS, event_kind_label,
)
from ecg.core.detection import (
    active_removed_samples, arrhythmia_baseline_info, classify_arrhythmias,
    correct_rr_artifacts, drop_removed_peaks, merge_artifact_reports,
    premature_beat_summary, premature_summary_rows,
    removed_beats_for_classification,
)
from ecg.core.analysis import (
    POINCARE_MIN_PAIRS, align_interval_column_to_peaks, analyse_core,
    analyse_hrv_freq, analyse_hrv_nonlinear, analyse_intervals,
    compute_poincare_metrics, hrv_freq_withheld_reason, hrv_time_domain,
    mean_hr_bpm, poincare_sd1_sd2,
    recompute_qtc_column, rr_bounds_label, rr_exclusion_text,
    rr_interval_series, successive_valid_pairs,
)
from ecg.core.autonomic import COV_METHODS, METHOD_NOTES, compute_autonomic_indices
from ecg.core.filtering import break_line_at_gaps
from ecg.core.metric_table import MISSING, format_metric
from ecg.core.spwvd import spwvd_lf_hf
from ecg.core.wave_template import WaveTemplate
from ecg.io.session import session_exists
from ecg.ui.autonomic_view import (
    _ann_spans, analysis_window_text, autonomic_status, draw_autonomic, format_param,
    method_key, normalise_autonomic_params, params_differ,
    parse_autonomic_inputs, same_window, summary_frame, summary_text,
)
from ecg.ui.theme import (
    PLOT, RED, RED_MID, RED_LIGHT, AMBER, BLUE, BLUE_DARK, BLUE_MID,
    GREEN, GREEN_DARK, GREEN_MID, ORANGE, ORANGE_DARK, ORANGE_DEEP,
    PURPLE, CORAL, NAVY, GRAY_LIGHT, CYAN_BRIGHT, MUTED, TEXT, LIGHT,
    CARD, BORDER, BORDER2, NK_AVAILABLE, nk,
    FONT_SIDEBAR_HDR, FONT_BADGE, FONT_HINT, FONT_SUBSECTION, FONT_KPI_LABEL,
    FONT_SMALL,
    SPACE_XS, SPACE_S, SPACE_L,
)
from ecg.ui.freshness import recorded_rr_bounds
from ecg.ui.plots import style_axes
from ecg.ui.tab_keys import MODE_EPOCHS, MODE_SLIDING, TAB_EVENTS, TAB_TIME_COURSE

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


def draw_windowed_panels(fig: Any, t: np.ndarray, df: pd.DataFrame,
                         panels: "list[tuple[str, str, str, str]]",
                         low_n: np.ndarray, gap_s: float,
                         annotations: "Iterable[Mapping]" = ()) -> None:
    """The stacked per-window figure of Epochs and Sliding, one axis per
    (column, y label, colour, title) of *panels*. No fill-to-zero: it would force
    the axis to start at 0 and flatten real HR / HRV swings. The line breaks at
    gaps longer than *gap_s*, where skipped windows left no row: a gap, not
    interpolated data. Windows of *low_n* are open, lighter markers. *annotations*
    are drawn on every panel as on the Autonomic figure, without stretching the axis."""
    axes = fig.subplots(len(panels), 1, sharex=True)
    if len(panels) == 1:
        axes = [axes]
    # subplots_adjust() is a no-op (with a UserWarning) once
    # constrained_layout is active on fig -- partial-override the
    # engine's own hspace instead so the stacked panels keep
    # their tighter author-tuned spacing (vs. CanvasSlot's wider
    # default _CL_SPACE) rather than silently losing the tuning.
    try:
        fig.set_constrained_layout_pads(hspace=0.08)
    except Exception as exc:
        log.debug("draw_windowed_panels: set_constrained_layout_pads failed: %s", exc)
    for ax, (col, ylabel, color, title) in zip(axes, panels):
        style_axes(ax)
        y = df[col].to_numpy(dtype=float)
        ax.plot(*break_line_at_gaps(t, y, gap_s), color=color, lw=1.4, zorder=3)
        if low_n.any():
            ax.plot(t[low_n], y[low_n], "o", ms=4.5, mfc="none",
                    mec=color, mew=1.0, alpha=0.6, zorder=4)
        if (~low_n).any():
            ax.plot(t[~low_n], y[~low_n], "o", ms=3.0, color=color, zorder=4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, loc="left", fontsize=8)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    x0, x1 = axes[0].get_xlim()
    anns = _ann_spans(annotations, x0, x1)
    for ts, te, _label, col in anns:
        for ax in axes:
            ax.axvspan(ts, te, color=col, alpha=0.10, linewidth=0, zorder=0)
            ax.axvline(ts, color=col, lw=1.1, alpha=0.8, zorder=5)
    for ax in axes:
        ax.set_xlim(x0, x1)
    _ylo, yhi = axes[0].get_ylim()
    for ts, te, label, col in anns:
        if label:
            axes[0].text(0.5 * (max(ts, x0) + min(te, x1)), yhi, label, ha="center", va="top",
                         fontsize=7, color=col, fontweight="bold", zorder=8, clip_on=True,
                         bbox=dict(boxstyle="round,pad=0.15", fc=PLOT["bg"], ec=col,
                                   alpha=0.8, lw=0.6))
    axes[-1].set_xlabel("Time (s)")


class AnalysisController:
    # Per-epoch/per-window count of VALID RR intervals (inside the RR bounds,
    # not spanning a removed beat -- the ones the metrics are computed from)
    # below which SDNN/RMSSD/pNN6 are still computed but flagged as
    # low-confidence -- an HRV metric from under ~10 intervals is noisy. A
    # window full of beats whose intervals were mostly excluded is just as
    # noisy, so this counts valid intervals, not peaks. Below the hard floor
    # (MIN_VALID_INTERVALS valid intervals, or MIN_WINDOW_BEATS peaks) the
    # epoch/window is skipped. Shared by compute_epochs/compute_rolling_hrv
    # so "low confidence" means the same thing in both panels.
    MIN_CONFIDENT_BEATS = 10
    MIN_VALID_INTERVALS = 3
    MIN_WINDOW_BEATS    = 5
    # The ticks of Epochs: the base columns of its frame, and those only when ticked.
    EPOCH_BASE_METRICS = ("HR", "SDNN", "RMSSD", "pNN6")
    EPOCH_OPTIONAL = ("SD1", "SD2", "LF_nu", "HF_nu", "LF_HF")
    # Under the plots of Epochs and Sliding.
    WINDOWED_LEGEND = (
        f"Hollow points: fewer than {MIN_CONFIDENT_BEATS} valid intervals in the window "
        f"(with an LW metric ticked, also fewer than {POINCARE_MIN_PAIRS} valid pairs). "
        f"Gaps in the line: windows skipped (fewer than {MIN_VALID_INTERVALS} valid "
        f"intervals or {MIN_WINDOW_BEATS} beats).")

    # Single colour axis for an abnormal-event's KIND, shared by every
    # rendering of it -- the event ribbon, the RR-panel span highlighting,
    # and each event card's stripe all used to pick their colour from a
    # different mapping (kind here, severity there), so the same event
    # could show as one colour in the ribbon and a different one on its
    # own card. Severity is still shown (as plain text, e.g. "WARNING"),
    # just no longer double-booked onto the same colour channel as kind.
    _KIND_COLORS = {
        "bradycardia":    BLUE_MID,
        "tachycardia":    RED_MID,
        "pause":          RED,
        # Premature beats (RR-based; "esv_run" in older versions): one hue
        # family, darker for longer groups.
        "premature_beat":    AMBER,
        "premature_couplet": ORANGE,
        "premature_run":     ORANGE_DEEP,
        "irregular_run":  PURPLE,
        "av_delay":       CYAN_BRIGHT,
    }

    _KIND_ICONS = {
        "bradycardia": "🔵", "tachycardia": "🔴", "pause": "⏸",
        "premature_beat": "•", "premature_couplet": "⚡", "premature_run": "⚡",
        "irregular_run": "〰", "av_delay": "⏳",
    }

    # Event cards are Tk frames: a recording with thousands of isolated
    # premature beats would take minutes to build them all. Past this many
    # the list stops with a note; the counts, ribbon, RR panel and exports
    # still cover every event.
    MAX_EVENT_CARDS = 400

    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    # ── HRV inputs shared by every HRV path ───────────────────────────
    # Every HRV computation (core, frequency, non-linear, epochs, rolling,
    # segment comparison) takes the same two inputs besides the peaks: the
    # user's RR acceptance bounds and the BREAKS -- real beats removed by
    # artifact correction, whose spanning interval must be excluded rather
    # than treated as a normal-to-normal one (see core.analysis
    # rr_interval_series). Removed spurious detections (duplicates, split
    # peaks) and peaks deleted by hand in edit mode are not breaks.

    def hrv_rr_bounds(self, quiet: bool = False) -> "tuple[float, float]":
        """(min, max) RR accepted for HRV, ms -- the sidebar Min R-R / Max RR
        entries. Main thread only. An unreadable or inconsistent pair
        (min ≥ max) falls back to the MouseECG defaults, with a status note
        (unless *quiet*, for redraws that must not touch the status bar)."""
        lo = self.app._safe_float(getattr(self.app, "ent_minrr", None), MouseECG.RR_MIN_MS)
        hi = self.app._safe_float(getattr(self.app, "ent_maxrr", None), MouseECG.RR_MAX_MS)
        if 0.0 <= lo < hi:
            return float(lo), float(hi)
        if quiet:
            return MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS
        log.warning("hrv_rr_bounds: invalid RR bounds %.1f–%.1f ms — using defaults", lo, hi)
        self.app._set_status(
            f"Invalid RR bounds ({lo:g}–{hi:g} ms, need Min < Max) — using "
            f"{MouseECG.RR_MIN_MS:.0f}–{MouseECG.RR_MAX_MS:.0f} ms", ORANGE)
        return MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS

    def results_rr_bounds(self) -> "tuple[float, float]":
        """The RR bounds the current core results were computed with --
        frequency and non-linear HRV extend those results (and are drawn
        against their rr_valid mask), so they reuse them rather than a value
        edited since; re-run Analyze to apply new bounds. Falls back to
        hrv_rr_bounds() for results without them (older sessions)."""
        hr = (self.app.analysis.results or {}).get("hr") or {}
        try:
            lo, hi = float(hr["rr_min_ms"]), float(hr["rr_max_ms"])
            if 0.0 <= lo < hi:
                return lo, hi
        except (KeyError, TypeError, ValueError):
            pass
        return self.hrv_rr_bounds()

    # A removed peak counts as "back" (no longer removed, no longer a break)
    # when a current peak lies within this distance of it -- re-detection,
    # or a peak re-added by hand that snapped a few samples off. Detected
    # peaks are at least PEAK_DISTANCE_MS (40 ms) apart, so this cannot
    # mistake a kept neighbour for a removed peak coming back.
    BREAK_REDETECT_TOL_MS = 5.0

    def break_tol_samples(self, fs: "float | None" = None) -> int:
        fs = fs if fs is not None else (self.app.signal.fs or MouseECG.FS_DEFAULT)
        return max(1, int(round(self.BREAK_REDETECT_TOL_MS / 1000.0 * float(fs))))

    def hrv_breaks(self, rpeaks: "np.ndarray | None" = None) -> "np.ndarray | None":
        """Samples of the REAL beats removed by artifact correction (the
        artifact report's ``break_samples`` -- premature beats; removed
        spurious detections such as duplicates are not breaks) that are
        still absent from *rpeaks* (default: the current accepted peaks),
        or None when there are none. Main thread only."""
        arep = self.app.analysis.artifact_report or {}
        removed = arep.get("break_samples") if isinstance(arep, dict) else None
        if not removed:
            return None
        peaks = rpeaks if rpeaks is not None else self.app.detection.rpeaks_ok
        if peaks is None:
            return None
        active = active_removed_samples(removed, peaks, self.break_tol_samples())
        return active if len(active) else None

    def current_artifact_report(self) -> "dict | None":
        """The artifact report restricted to the removals still in effect on
        the current accepted peaks (counts recomputed) -- what a session
        saves, so that re-applying its removals on restore rebuilds exactly
        these peaks, not beats that have come back since (threshold change,
        re-detection). A report from an older session, which lists no
        removed samples, is returned unchanged. Main thread only."""
        arep = self.app.analysis.artifact_report
        rp = self.app.detection.rpeaks_ok
        if not isinstance(arep, dict) or rp is None or "removed_samples" not in arep:
            return arep
        return merge_artifact_reports(None, arep, rp, self.break_tol_samples())

    def reapply_artifact_removals(self) -> int:
        """Drop from the accepted peaks every peak the (restored) artifact
        report removed. Session restore re-runs detection from the full
        candidate set, which brings back every auto-corrected or reviewed
        beat; without this the restored breaks are silently dropped (their
        beat is "back") and the next frequency / non-linear / epoch /
        rolling run would use the removed premature beat's intervals as
        normal ones, contradicting the restored core results. Peaks added
        by hand are never dropped. Returns the number of peaks dropped.
        Main thread only."""
        arep = self.app.analysis.artifact_report
        rp = self.app.detection.rpeaks_ok
        removed = arep.get("removed_samples") if isinstance(arep, dict) else None
        if not removed or rp is None or len(rp) == 0:
            return 0
        kept = drop_removed_peaks(rp, removed, self.break_tol_samples(),
                                  protected=self.app.detection.manual_added)
        n_drop = int(len(rp) - len(kept))
        if n_drop:
            self.app.detection.rpeaks_ok = kept.astype(np.asarray(rp).dtype, copy=False)
            log.info("reapply_artifact_removals: %d restored removals re-applied", n_drop)
        return n_drop

    def exclusion_status_note(self) -> str:
        """'  |  1 RR excluded (outside 67–333 ms)' for the current core
        results ('' when nothing was excluded). Every step that can end the
        Analyze chain or follow it (frequency / non-linear HRV, epochs,
        rolling HRV, session restore) appends it to its final status -- in
        ORANGE -- so the disclosure is not overwritten a moment later by a
        plain green "done"."""
        hr = (self.app.analysis.results or {}).get("hr") or {}
        text = rr_exclusion_text(hr)
        return f"  |  {text}" if text else ""

    # ── Freshness: what a job ran on, captured at its start (ecg.ui.freshness) ──
    def _capture_basis(self, **kw: Any) -> "dict | None":
        svc = getattr(self.app, "freshness", None)
        return svc.capture(**kw) if svc is not None else None

    def _stamp_basis(self, module: str, basis: "dict | None") -> None:
        svc = getattr(self.app, "freshness", None)
        if svc is not None and basis is not None:
            svc.stamp(module, basis)

    def _basis_moved(self, basis: "dict | None") -> bool:
        """The peaks, removed beats or window a job started on are no longer
        the current ones. Its RR bounds are those of the core results it
        extends, compared by the state check rather than here."""
        if basis is None:
            return False
        now = self._capture_basis(rr_bounds=basis["rr"])
        return now is not None and any(now[k] != basis[k] for k in ("peaks", "removed", "window"))

    def _job_is_stale(self, what: str, label_attr: str, gen: int, basis: "dict | None") -> bool:
        """True when a job's result must be dropped: the file changed (labels
        already reset), or the peaks, removed beats or window moved while it
        ran (it says so, or its label would read "Computing" for good)."""
        # The job's end has just re-enabled its button.
        self._disable_result_btns_without_results()
        if getattr(self.app, "_generation", 0) != gen:
            log.info("%s: stale result discarded (file changed)", what)
            return True
        if self.app.analysis.results is not None and not self._basis_moved(basis):
            return False
        log.info("%s: stale result discarded (peaks or window changed)", what)
        text = f"{what} discarded — peaks or window changed while it ran: click Analyze"
        lbl = getattr(self.app, label_attr, None)
        if lbl is not None:
            lbl.configure(text=f"  {text}", text_color=ORANGE)
        self.app._set_status(text, ORANGE)
        return True

    def _disable_result_btns_without_results(self) -> None:
        if self.app.analysis.results is None:
            self.app._set_result_btns_enabled(
                False, ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl"))

    def _refused_busy(self) -> bool:
        """A job is running, so the busy guard would refuse a new one: say so
        before any label reads Computing for a job that will not run."""
        if not getattr(self.app, "_async_busy", False):
            return False
        self.app._set_status("Another operation is already running — please wait", ORANGE)
        return True

    def _when_idle(self, step: "Any") -> None:
        """Run the next step of the Analyze chain once no job is running: one
        started between two steps would have the busy guard refuse this one, and
        the rest of the chain with it. Dropped, without a word, if the results
        it would extend are gone by then."""
        results = self.app.analysis.results
        gen = getattr(self.app, "_generation", 0)

        def _try() -> None:
            if (self.app.analysis.results is not results
                    or getattr(self.app, "_generation", 0) != gen):
                log.info("Analyze chain stopped: its results were replaced or dropped")
            elif getattr(self.app, "_async_busy", False):
                self.app.after(200, _try)
            else:
                step()

        _try()

    def _mark_running(self, *modules: str) -> None:
        """For the tab's computing state. No-op when the busy guard is about to
        refuse the job."""
        running = getattr(getattr(self.app, "ui", None), "running_modules", None)
        if isinstance(running, set) and not getattr(self.app, "_async_busy", False):
            running.update(modules)
            self.app.ui.running_idle_ticks = 0

    def _mark_landed(self, module: str) -> None:
        running = getattr(getattr(self.app, "ui", None), "running_modules", None)
        if isinstance(running, set):
            running.discard(module)

    # ── Persistent result labels ──────────────────────────────────────
    # The sidebar analysis-window label (with its orange RR-exclusion line)
    # and the Frequency / Non-linear tab status labels are the disclosures
    # that outlast the transient status bar. They are derived from the
    # results here -- not only written once by the step that produced them
    # -- so a UI rebuild (theme toggle, which recreates every label at its
    # default) and a session restore show them again.

    def _analysis_window_label(self) -> "tuple[str, str]":
        """(text, colour) of lbl_analysis_window for the current results."""
        r  = self.app.analysis.results or {}
        hr = r.get("hr") or {}
        rp_ok   = self.app.detection.rpeaks_ok
        n_total = len(rp_ok) if rp_ok is not None else int(hr.get("n", 0) or 0)
        n_used  = int(hr.get("n", n_total) or n_total)
        win = self.app.analysis.analysed_window
        t0, t1 = win if win is not None else (self.app.analysis.t_start,
                                              self.app.analysis.t_end)
        n_excl = int(hr.get("n_excluded", 0) or 0)
        # Short, on its own line: the sidebar label has little width.
        excl_lbl = (f"\n{n_excl} RR excluded from HRV "
                    f"({float(hr.get('excluded_pct', 0.0) or 0.0):.1f} %)"
                    if n_excl else "")
        color = ORANGE if n_excl else GREEN
        if t0 > 0 or t1 > 0:
            dur_str = f"{t0:.1f} s → {t1:.1f} s" if t1 > 0 else f"{t0:.1f} s → end"
            return (f"✓  Analyzed  ·  {n_used}/{n_total} peaks  ·  {dur_str}{excl_lbl}",
                    color)
        return f"✓  Analyzed  ·  {n_total} peaks  ·  full signal{excl_lbl}", color

    @staticmethod
    def _freq_status(fd: "pd.DataFrame | None", done_label: str = "Done",
                     just_computed: bool = False) -> "tuple[str, str]":
        """(text, colour) of lbl_freq_status for a frequency-HRV result:
        computed, withheld (and why), or not computed yet (or, right after
        a run, *just_computed*: failed)."""
        withheld = hrv_freq_withheld_reason({"hrv_freq": fd})
        if withheld:
            return f"  Withheld — {withheld}", ORANGE
        if fd is None or fd.empty:
            if just_computed:
                return "  Frequency HRV could not be computed — see the log", ORANGE
            return "  Core done — click to compute LF / HF", BLUE
        # HRV_LF/HRV_HF are absolute band power (ms^2) -- HRV_LFn/HRV_HFn
        # are the normalised fractions (of LF+HF power) that the table and
        # PSD legend both already show, and are what "LF %"/"HF %" means
        # to a reader. Multiplying the absolute power by 100 (an earlier
        # version) produced a tiny, meaningless number that openly
        # contradicted the table two inches below it.
        n_lf = n_hf = n_lfhf = MISSING
        try:
            # Same strings as the cards, the table and the spectrum title.
            n_lf = format_metric("hrv_freq.HRV_LFn", fd["HRV_LFn"].values[0])
            n_hf = format_metric("hrv_freq.HRV_HFn", fd["HRV_HFn"].values[0])
            n_lfhf = format_metric("hrv_freq.HRV_LFHF", fd["HRV_LFHF"].values[0])
            n_lf, n_hf = (s if s == MISSING else f"{s}%" for s in (n_lf, n_hf))
        except Exception as _exc:
            log.debug("freq status: LF/HF% formatting failed: %s", _exc, exc_info=True)
        return f"  {done_label}  LFn={n_lf}  HFn={n_hf}  LF/HF={n_lfhf}", GREEN

    @staticmethod
    def _nonlin_status(nl: "pd.DataFrame | None",
                       just_computed: bool = False) -> "tuple[str, str]":
        """(text, colour) of lbl_nonlin_status for a non-linear HRV result
        (empty: not computed yet or, right after a run, failed)."""
        if nl is None or nl.empty:
            if just_computed:
                return "  Non-linear HRV could not be computed — see the log", ORANGE
            return "  Core done — click to compute SampEn / DFA (slow!)", "#9C27B0"
        sampen = MISSING
        try:
            sampen = format_metric("hrv_nonlin.HRV_SampEn", nl["HRV_SampEn"].values[0])
        except Exception as _exc:
            log.debug("non-linear status: SampEn formatting failed: %s", _exc, exc_info=True)
        # Disclose which RR intervals were used: Poincaré/DFA use the whole
        # window, only the entropy block may be capped for runtime (its
        # count and time span are shown); results from an older version
        # used the first 1000 beats for everything.
        note = ""
        if nl.attrs.get("entropy_capped"):
            note = (f"  ·  entropy from {int(nl.attrs.get('entropy_n_rr', 0)):,} of "
                    f"{int(nl.attrs.get('n_rr_valid', 0)):,} RR "
                    f"({float(nl.attrs.get('entropy_t_start_s', 0.0)):.0f}–"
                    f"{float(nl.attrs.get('entropy_t_end_s', 0.0)):.0f} s)")
        elif nl.attrs.get("truncated"):
            note = (f"  ·  first {int(nl.attrs.get('n_beats_used', 0)):,} of "
                    f"{int(nl.attrs.get('n_beats_total', 0)):,} beats (older analysis)")
        # NeuroKit paired intervals across an excluded one: its
        # asymmetry/fragmentation indices were withheld (core analysis).
        if nl.attrs.get("pair_metrics_withheld"):
            note += "  ·  asymmetry indices withheld (RR gaps)"
        return f"  Done  SampEn={sampen}{note}", ORANGE if note else GREEN

    def refresh_result_labels(self, freq_done_label: str = "Done") -> None:
        """Re-derive lbl_analysis_window, lbl_freq_status, lbl_nonlin_status
        and lbl_ivl_status from the current results (no-op without results).
        Called after Analyze, after a UI rebuild and after a session
        restore. Main thread only."""
        r = self.app.analysis.results
        if r is None:
            return
        for attr, (text, color) in (
            ("lbl_analysis_window", self._analysis_window_label()),
            ("lbl_freq_status",     self._freq_status(r.get("hrv_freq"), freq_done_label)),
            ("lbl_nonlin_status",   self._nonlin_status(r.get("hrv_nonlin"))),
            ("lbl_ivl_status",      self._intervals_status(r.get("intervals"))),
        ):
            lbl = getattr(self.app, attr, None)
            if lbl is not None:
                lbl.configure(text=text, text_color=color)

    def get_freq_bands(self) -> "tuple[tuple, tuple, tuple]":
        """Return (vlf, lf, hf) tuples from the selected band preset.

        Reads ``cb_freq_band`` if it exists; falls back to Mouse Thireau defaults.
        """
        try:
            sel = self.app.cb_freq_band.get() if self.app.cb_freq_band is not None else ""
            for name, vlf, lf, hf in MouseECG.FREQ_BAND_PRESETS:
                if name == sel:
                    return vlf, lf, hf
        except Exception:
            pass
        return (0.0, 0.4), (0.4, 1.5), (1.5, 5.0)

    def run_arrhythmia_analysis(self) -> None:
        if self.app.detection.rpeaks_ok is None or self.app.signal.fs is None:
            self.app._set_status("Click Analyze first.", RED)
            return

        rpeaks = self.app._windowed_peaks()
        if rpeaks is None or len(rpeaks) < 5:
            self.app._set_status("Not enough peaks in the analysis window.", ORANGE)
            return
        fs      = float(self.app.signal.fs)

        try:
            baseline_s = max(5.0, float(self.app.ent_arr_baseline.get()))
        except (ValueError, AttributeError):
            baseline_s = 30.0
        try:
            brady_pct = max(5.0, min(60.0, float(self.app.ent_arr_brady_pct.get())))
        except (ValueError, AttributeError):
            brady_pct = 20.0
        try:
            min_beats = max(3, int(self.app.ent_arr_min_beats.get()))
        except (ValueError, AttributeError):
            min_beats = 10
        _gen = getattr(self.app, "_generation", 0)  # snapshot — detect file change
        # The baseline classify_arrhythmias() will use for these peaks
        # (window, value, and whether it is a meaningful reference: not when
        # it covers most of the analysed span) -- cheap, so computed here.
        bl_info     = arrhythmia_baseline_info(rpeaks, fs, baseline_s)
        bl_reliable = bool(bl_info["reliable"])

        pr_ms, av_note = self.av_delay_input(
            (self.app.analysis.results or {}).get("intervals"), rpeaks, fs)

        # Real beats automatic / reviewed artifact correction removed
        # (ectopic, and very premature ones removed as duplicate /
        # non-physiological) are put back for the premature-beat
        # classification, so they are still counted (classify_arrhythmias
        # keeps only those inside the analysed span).
        removed_ect, removal_info = self.arrhythmia_removed_beats(fs)
        basis = {"peaks": np.asarray(rpeaks).copy(), "removed": removed_ect.copy()}

        def _worker():
            def _prog(p, m):
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))
            return classify_arrhythmias(
                rpeaks, fs,
                baseline_s=baseline_s,
                brady_pct=brady_pct,
                min_brady_beats=min_beats,
                progress_cb=_prog,
                pr_ms=pr_ms,
                removed_beats=removed_ect if len(removed_ect) else None,
            )

        def _done(events: "list[ArrhythmiaEvent]"):
            if getattr(self.app, "_generation", 0) != _gen:
                log.info("run_arrhythmia_analysis: stale result discarded (file changed)")
                return
            self.app.analysis.arrhythmia_events = events
            self.app.analysis.arrhythmia_basis  = basis
            self.app.analysis.arr_selected_idx  = -1
            # Counts and per-hour rates of the premature beats over the
            # analysed span (first to last analysed peak).
            summary = premature_beat_summary(
                events, bl_info["span_s"], removal_info,
                t_start=float(rpeaks[0]) / fs, t_end=float(rpeaks[-1]) / fs)
            self.app.analysis.premature_summary = summary
            self.refresh_premature_label()

            # What show_events() draws, also after a UI rebuild.
            self.app.analysis.arrhythmia_run = {
                "baseline": bl_info, "brady_pct": brady_pct, "av_note": av_note,
                "av_on": pr_ms is not None, "pr_ms": pr_ms,
                "peaks": np.asarray(rpeaks).copy(), "fs": fs}
            self.show_events()

            # ── TSV store ─────────────────────────────────────
            self.app.analysis.arrhythmia_tsv = self.arrhythmia_tsv(events, summary)

            n = len(events)
            # The baseline window spanning most of the analysed beats means
            # every brady/tachy call (or its absence) is made against a
            # reference that includes the event itself -- say so, in orange.
            bl_note = ("" if bl_reliable else
                       "  ·  baseline covers most of the window — "
                       "brady/tachy calls unreliable")
            rm_note = (f"  ·  {int(summary['n_removed'])} premature beat(s) removed by "
                       f"artifact correction still counted" if summary["n_removed"] else "")
            if summary.get("n_removed_unknown_type"):
                rm_note += (f"  ·  {int(summary['n_removed_unknown_type'])} removed peak(s) "
                            f"of unknown type (older session) judged from their interval "
                            f"only — re-run artifact correction")
            self.app._set_status(
                f"Abnormal-event classification — {n} event(s)  ·  {av_note}{rm_note}{bl_note}",
                GREEN if bl_reliable else ORANGE)
            self.app.tabs.set(TAB_EVENTS)

        if self.app.btn_run_arrhythmia is None:
            return
        self.app._start_async_result(
            self.app.btn_run_arrhythmia, "Classifying…", _worker, _done)

    @staticmethod
    def av_delay_input(ivl: "Any", rpeaks: "np.ndarray",
                       fs: float) -> "tuple[np.ndarray | None, str]":
        """(PR per peak or None, the note saying why) for the AV-delay check.

        AV conduction-delay detection needs per-beat PR data from Interval
        delineation, which is a separate, optional analysis step. Its
        DataFrame is not one row per peak (edge beats have no row), so PR
        (P onset -> QRS onset) is matched to these windowed rpeaks by sample
        index, NaN where a peak has no row (e.g. peaks edited since). The
        check only runs when at least half the peaks got a PR -- Intervals
        run on another analysis window leaves most peaks unmatched, and a
        sparse PR series would just fragment the runs."""
        pr_ms = None
        av_note = "AV-delay check skipped — measure intervals on the current window"
        if ivl is not None and not ivl.empty:
            pr_aligned = align_interval_column_to_peaks(ivl, rpeaks, fs, "PR_ms")
            if np.isfinite(pr_aligned).sum() >= 0.5 * len(rpeaks):
                pr_ms = pr_aligned
                av_note = "AV-delay check on"
            elif (np.isfinite(align_interval_column_to_peaks(ivl, rpeaks, fs, "R_peak_s")).sum()
                  >= 0.5 * len(rpeaks)):
                # Rows do match this window -- PR itself was mostly out of
                # bounds, so re-running Intervals as-is would not help.
                av_note = "AV-delay check skipped — PR measured on < 50% of beats"
        return pr_ms, av_note

    def note_intervals_measured(self, ivl: "pd.DataFrame") -> None:
        """Intervals just measured: would the last classification's AV-delay
        check now get another input? Its events and note stay as they ran, and
        Events reads out of date until they are classified again."""
        run = getattr(self.app.analysis, "arrhythmia_run", None)
        if run is None:
            return
        pr_ms, av_note = self.av_delay_input(ivl, run["peaks"], run["fs"])
        used = run.get("pr_ms")
        same_pr = (pr_ms is None and used is None) or (
            pr_ms is not None and used is not None
            and np.array_equal(pr_ms, used, equal_nan=True))
        run["av_outdated"] = not same_pr or av_note != run["av_note"]
        self.refresh_events_head()

    def arrhythmia_removed_beats(self, fs: "float | None" = None) -> "tuple[np.ndarray, dict]":
        """removed_beats_for_classification() of the current artifact
        report and accepted peaks: (samples to put back, info)."""
        return removed_beats_for_classification(
            self.app.analysis.artifact_report, self.app.detection.rpeaks_ok,
            self.break_tol_samples(fs))

    def arrhythmia_is_current(self) -> bool:
        """True when the stored abnormal-event results were classified
        from the current windowed peaks and removed beats -- False after a
        peak edit, a re-detection, a change of artifact correction or of
        the analysis window (or with no basis recorded). Without events or
        a summary there is nothing to be stale: True."""
        a = self.app.analysis
        if not a.arrhythmia_events and not a.premature_summary:
            return True
        basis = getattr(a, "arrhythmia_basis", None)
        if not basis or self.app.detection.rpeaks_ok is None:
            return False
        try:
            peaks = self.app._windowed_peaks()
            removed, _ = self.arrhythmia_removed_beats()
        except Exception as exc:
            log.debug("arrhythmia_is_current: %s", exc)
            return False
        return (self._same_peaks(basis.get("peaks"), peaks)
                and self._same_peaks(basis.get("removed"), removed))

    @staticmethod
    def arrhythmia_tsv(events: "list[ArrhythmiaEvent]",
                       summary: "dict | None" = None) -> str:
        """The "Copy for Excel" text: one row per event, then (after a
        blank line) the premature-beat counts / rates and their definition."""
        rows = ["Type\tLabel\tStart_s\tEnd_s\tDuration_s\tN_beats\tHR_bpm\t"
                "Baseline_bpm\tDelta_pct\tRR_ms\tSeverity\tN_removed\tDescription"]
        for ev in events:
            rows.append(
                f"{ev.kind}\t{event_kind_label(ev.kind)}\t{ev.t_start:.2f}\t{ev.t_end:.2f}\t"
                f"{ev.duration_s:.2f}\t{int(ev.n_beats)}\t{ev.hr_mean:.1f}\t"
                f"{ev.baseline_hr:.1f}\t{ev.delta_pct:.1f}\t"
                f"{ev.rr_mean:.1f}\t{ev.severity}\t{int(getattr(ev, 'n_removed', 0) or 0)}\t"
                f"{ev.label}"
            )
        summary_rows = premature_summary_rows(summary)
        if summary_rows:
            rows.append("")
            rows.extend(f"{label.strip()}\t{value}" for label, value in summary_rows)
        return "\n".join(rows)

    @staticmethod
    def premature_summary_text(summary: "dict | None") -> str:
        """Short multi-line text of the premature-beat counts for the
        Events list header ('' before a classification)."""
        if not summary:
            return ""
        def _r(x: float) -> str:
            return f"{x:.1f}/h" if np.isfinite(x) else "—/h"
        text = (f"Premature beats (RR-based): {int(summary['n_premature'])}  "
                f"({_r(summary['premature_per_h'])})\n"
                f"isolated {int(summary['n_isolated'])} ({_r(summary['isolated_per_h'])})  ·  "
                f"couplets {int(summary['n_couplets'])} ({_r(summary['couplets_per_h'])})  ·  "
                f"runs ≥3 {int(summary['n_runs'])} ({_r(summary['runs_per_h'])})")
        if int(summary.get("n_removed", 0) or 0):
            text += f"\n{int(summary['n_removed'])} removed by artifact correction (counted)"
        if int(summary.get("n_removed_unknown_type", 0) or 0):
            text += (f"\n{int(summary['n_removed_unknown_type'])} removed peak(s) of unknown "
                     f"type (older session): re-run artifact correction")
        text += "\nTiming only: ventricular vs supraventricular not distinguished."
        return text

    def refresh_premature_label(self) -> None:
        """Show the current premature-beat summary in the event-list header
        (no-op without the label). Main thread only."""
        lbl = getattr(self.app, "lbl_arr_premature", None)
        if lbl is None:
            return
        text = self.premature_summary_text(
            getattr(self.app.analysis, "premature_summary", None))
        try:
            lbl.configure(text=text)
        except Exception as exc:
            log.debug("lbl_arr_premature configure failed: %s", exc)

    def show_events(self) -> None:
        """The last classification on the Events tab: ribbon with the RR series,
        cards, status and readout. From its run and after a UI rebuild, which
        also shows the selected event again. No-op before a classification."""
        a = self.app.analysis
        run = getattr(a, "arrhythmia_run", None)
        if run is None:
            return
        events = a.arrhythmia_events
        self.draw_events_overview()
        self.build_event_cards()
        summary = a.premature_summary or {}
        n = len(events)
        n_prem = int(summary.get("n_premature", 0))
        bl_reliable = bool(run["baseline"]["reliable"])
        # Short form in the action bar; the full reason and remedy go to the
        # baseline readout and the status bar.
        av_short = "AV-delay check on" if run["av_on"] else "AV-delay check skipped"
        bl_short = "" if bl_reliable else "  ·  baseline unreliable"
        self.app.lbl_arrhythmia_status.configure(  # type: ignore[union-attr]
            text=(f"  {n} event{'s' if n != 1 else ''} — click to explore  ·  "
                  f"{n_prem} premature beat{'s' if n_prem != 1 else ''} (RR-based)  ·  "
                  f"{av_short}{bl_short}"),
            text_color=RED if any(e.severity=="alert" for e in events)
                       else (ORANGE if (n or not bl_reliable) else GREEN),
        )
        self.refresh_premature_label()
        self.refresh_events_head()
        sync = getattr(self.app, "_sync_arr_edit_btn", None)
        if sync is not None:
            sync()
        idx = a.arr_selected_idx
        title = getattr(self.app, "lbl_arr_event_title", None)
        if 0 <= idx < n:
            self.app._select_arrhythmia_event(idx)
        else:
            # A new classification: nothing is selected, the zoom shows no event.
            slot = self.app._slots.get("arr_detail")
            if slot is not None:
                slot._draw_fn = None
                slot._show_placeholder()
            if title is not None:
                title.configure(text="← Click on an event", text_color=MUTED)

    def refresh_events_head(self) -> None:
        """The list title and the readout on the bar (no-op without them)."""
        a = self.app.analysis
        head = getattr(self.app, "lbl_arr_events_head", None)
        if head is not None:
            n = len(a.arrhythmia_events)
            head.configure(text=f"{n} EVENT{'S' if n != 1 else ''}"
                           if getattr(a, "arrhythmia_run", None) is not None else "EVENTS")
        lbl = getattr(self.app, "lbl_arr_baseline", None)
        if lbl is not None:
            lbl.configure(text=self.events_readout_text(getattr(a, "arrhythmia_run", None)))

    @staticmethod
    def events_readout_text(run: "dict | None") -> str:
        """What the events on screen were classified against: the baseline's
        window and value, and whether the AV-delay check ran or why it did not."""
        if not run:
            return "Baseline —  ·  AV-delay check —"
        bl = run["baseline"]
        if np.isfinite(bl["baseline_hr"]):
            text = (f"Baseline {bl['t_start']:.1f}–{bl['t_end']:.1f} s  ·  "
                    f"{bl['baseline_hr']:.0f} bpm")
        else:
            text = "Baseline —"
        if not bl["reliable"]:
            text += " (unreliable: it covers most of the window)"
        if run.get("av_outdated"):
            av = "on" if run.get("av_on") else "skipped"
            return f"{text}  ·  AV-delay check {av} — intervals measured since: classify again"
        return f"{text}  ·  {run['av_note']}"

    def event_at_time(self, x: float, tol_s: float) -> int:
        """Index of the event under time *x*, or else the nearest within *tol_s*;
        -1 for none. For a click on the ribbon."""
        best, best_d = -1, float("inf")
        for i, ev in enumerate(self.app.analysis.arrhythmia_events):
            t1 = max(ev.t_end, ev.t_start + 0.05)
            d = 0.0 if ev.t_start <= x <= t1 else min(abs(x - ev.t_start), abs(x - t1))
            if d < best_d:
                best, best_d = i, d
        return best if best_d <= tol_s else -1

    def on_arr_ribbon_click(self, event) -> None:
        """A click on the ribbon or the RR series selects the event there."""
        if event.xdata is None or event.button != 1 or event.inaxes is None:
            return
        lo, hi = event.inaxes.get_xlim()
        idx = self.event_at_time(float(event.xdata), 0.01 * abs(hi - lo))
        if idx >= 0:
            self.app.nav_ctrl.select_arrhythmia_event(idx)
            self.app.nav_ctrl._reveal_arr_card(idx)

    def draw_events_overview(self) -> None:
        """Ribbon and RR series of the last classification, above the cards. All
        from what the classification ran with (arrhythmia_run), never a fresh read
        of the entries, which may have been edited since."""
        slot = self.app._slots.get("arr_ribbon")
        run = getattr(self.app.analysis, "arrhythmia_run", None)
        if slot is None or run is None:
            return
        events = self.app.analysis.arrhythmia_events
        rpeaks, fs = run["peaks"], run["fs"]
        bl_info, brady_pct = run["baseline"], run["brady_pct"]
        bl_reliable = bool(bl_info["reliable"])
        kind_colors = self._KIND_COLORS

        t_peaks = rpeaks / fs
        rr_ms   = np.diff(rpeaks).astype(float) / fs * 1000
        t_rr    = (t_peaks[:-1] + t_peaks[1:]) / 2
        # The exact stretch classify_arrhythmias() took its baseline
        # from: it starts at the first peak of the analysis window (not
        # t=0) and widens past baseline_s when that held too few beats.
        # Uses the baseline_s the classification ran with, not a fresh
        # read of the entry, so a redraw can't drift from the events.
        bl_t0, bl_t1 = bl_info["t_start"], bl_info["t_end"]

        def draw_rr_timeline(fig):
            from matplotlib.gridspec import GridSpec
            # CanvasSlot's constrained_layout would recompute/override
            # explicit margins below (same reason draw_intervals() opts
            # out) -- a fig.legend() with no reserved top margin gets
            # clipped by the figure boundary otherwise.
            try:
                fig.set_layout_engine(None)
            except Exception as exc:
                log.debug("draw_rr_timeline: set_layout_engine(None) failed: %s", exc)
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.55,
                         left=0.08, right=0.98, top=0.80, bottom=0.24)
            ax_ribbon = fig.add_subplot(gs[0, 0])
            ax        = fig.add_subplot(gs[1, 0], sharex=ax_ribbon)
            style_axes(ax_ribbon)
            style_axes(ax)

            # ── Event ribbon: ethogram-style strip, colored by kind ──
            # A beat or a pause must still show on a strip spanning the recording.
            min_w = max(0.05, 0.003 * float(t_peaks[-1] - t_peaks[0]))
            by_kind: "dict[str, list[tuple[float, float]]]" = {}
            for ev in events:
                by_kind.setdefault(ev.kind, []).append(
                    (ev.t_start, max(ev.duration_s, min_w)))
            for kind, spans in by_kind.items():
                ax_ribbon.broken_barh(
                    spans, (0.1, 0.8),
                    facecolors=kind_colors.get(kind, MUTED), zorder=2)
            ax_ribbon.set_ylim(0, 1)
            ax_ribbon.set_yticks([])
            ax_ribbon.tick_params(labelbottom=False)
            ax_ribbon.set_title("Event ribbon", loc="left", fontsize=8,
                                color=PLOT.get("muted", MUTED))
            if by_kind:
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], color=kind_colors.get(k, MUTED),
                                  lw=5, label=event_kind_label(k))
                           for k in by_kind]
                # top=0.80 above reserves real figure-level margin for
                # this -- fig.legend() (figure-relative 0-1 coords) is
                # still used over ax_ribbon.legend() since it isn't
                # bound by the ribbon axes' own (short) height.
                fig.legend(handles=handles, loc="upper center",
                          bbox_to_anchor=(0.5, 0.99), ncol=min(len(handles), 6),
                          fontsize=6.5, frameon=False,
                          labelcolor=PLOT.get("text", GRAY_LIGHT))

            # PLOT has no "ecg" key -- this always fell back to the
            # literal default (a bright cyan barely visible against the
            # Light theme's grid). "signal" is the real trace-colour
            # token every other plot in the app already uses.
            ax.plot(t_rr, rr_ms, color=PLOT.get("signal", CYAN_BRIGHT), lw=0.8, zorder=2)
            ax.set_ylabel("RR (ms)"); ax.set_xlabel("Time (s)")

            # Baseline and threshold lines (only if events contain
            # brady/tachy, or the baseline is unreliable -- then the
            # shading shows how much of the window it covers)
            brady_events = [e for e in events if e.kind in ("bradycardia","tachycardia")
                            and e.baseline_hr > 0]
            if (brady_events or not bl_reliable) and np.isfinite(bl_info["baseline_hr"]):
                # The baseline and the brady_pct the classification used
                # (captured, same clamp) -- never a fresh read of the
                # entry, which may have been edited since.
                bl_hr  = float(bl_info["baseline_hr"])
                bl_rr  = 60_000.0 / bl_hr
                bpct   = brady_pct
                brady_rr = 60_000.0 / (bl_hr * (1 - bpct / 100))
                tachy_rr = 60_000.0 / (bl_hr * (1 + bpct / 100))
                # Shade baseline window
                ax.axvspan(bl_t0, bl_t1, alpha=0.08, color=GREEN_MID,
                           zorder=0,
                           label=("Baseline window" if bl_reliable else
                                  "Baseline window (covers most of the window)"))
                ax.axvline(bl_t1, color=GREEN_MID, lw=1.0,
                           ls=":", alpha=0.6, zorder=3)
                # Baseline RR line
                ax.axhline(bl_rr, color=GREEN_MID, lw=1.2,
                           ls="--", alpha=0.8, zorder=4,
                           label=f"Baseline {bl_hr:.0f} bpm")
                # Brady / tachy threshold lines -- neutral grey, not a
                # kind colour: AMBER/BLUE_MID here used to collide with
                # premature-beat/bradycardia's OWN kind colours below, so the
                # same hue meant a different thing depending which line
                # on the plot you were looking at.
                ax.axhline(brady_rr, color=PLOT.get("muted", MUTED), lw=0.9,
                           ls="--", alpha=0.65, zorder=4,
                           label=f"Brady threshold −{bpct:.0f}%")
                ax.axhline(tachy_rr, color=PLOT.get("muted", MUTED), lw=0.9,
                           ls=":", alpha=0.65, zorder=4,
                           label=f"Tachy threshold +{bpct:.0f}%")

            ax.set_title(
                "RR series — click an event or the ribbon to zoom",
                loc="left", fontsize=9)
            for ev in events:
                c = kind_colors.get(ev.kind, "#888")
                ax.axvspan(ev.t_start, max(ev.t_end, ev.t_start + 0.05),
                           alpha=0.20, color=c, zorder=1)
                ax.axvline(ev.t_start, color=c, lw=0.7, ls="--",
                           alpha=0.55, zorder=3)
            if (brady_events or not bl_reliable) and np.isfinite(bl_info["baseline_hr"]):
                ax.legend(loc="upper right", fontsize=7,
                          facecolor=PLOT.get("bg",NAVY),
                          labelcolor=PLOT.get("text",GRAY_LIGHT),
                          edgecolor=PLOT.get("border","#333"))

        slot.update(draw_rr_timeline)

    def build_event_cards(self) -> None:
        events = self.app.analysis.arrhythmia_events
        kind_icons = self._KIND_ICONS
        # ── Build clickable event cards ───────────────────
        for w in self.app._arr_card_widgets:
            try: w.destroy()
            except Exception: pass
        self.app._arr_card_widgets.clear()

        if not events:
            lbl = ctk.CTkLabel(
                self.app._arr_event_scroll,
                text="No abnormal events detected\nfor the active context.",
                font=FONT_SMALL, text_color=MUTED, justify="center",
            )
            lbl.grid(row=0, column=0, pady=SPACE_L)
            self.app._arr_card_widgets.append(lbl)
        else:
            for idx, ev in enumerate(events[:self.MAX_EVENT_CARDS]):
                self.build_arrhythmia_card(idx, ev, kind_icons)
            if len(events) > self.MAX_EVENT_CARDS:
                lbl = ctk.CTkLabel(
                    self.app._arr_event_scroll,
                    text=(f"… {len(events) - self.MAX_EVENT_CARDS} more events not "
                          f"listed — they are in the counts, the ribbon and the exports."),
                    font=FONT_HINT, text_color=MUTED, justify="left", wraplength=230,
                )
                lbl.grid(row=self.MAX_EVENT_CARDS, column=0, sticky="ew", pady=SPACE_S)
                self.app._arr_card_widgets.append(lbl)

    def build_arrhythmia_card(
        self, idx: int, ev: "ArrhythmiaEvent", kind_icons: dict,
    ) -> None:
        # Colour is the event's KIND (matches the ribbon and the RR-panel
        # span highlighting -- see _KIND_COLORS). Severity is shown as
        # plain text instead of also claiming the colour channel, so it no
        # longer contradicts kind on cards where the two used to disagree.
        c_kind = self._KIND_COLORS.get(ev.kind, MUTED)
        icon   = kind_icons.get(ev.kind, "·")

        card = ctk.CTkFrame(
            self.app._arr_event_scroll,
            fg_color=CARD, corner_radius=6,
            border_width=2, border_color=BORDER,
        )
        card.grid(row=idx, column=0, sticky="ew", padx=SPACE_S, pady=(0, SPACE_S))
        card.grid_columnconfigure(1, weight=1)

        # height=1: a frame's default of 200 px would set the height of the card.
        ctk.CTkFrame(card, width=4, height=1, fg_color=c_kind,
                     corner_radius=0).grid(row=0, column=0, rowspan=4, sticky="ns")

        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.grid(row=0, column=1, sticky="ew", padx=(SPACE_S, SPACE_S), pady=(SPACE_S, SPACE_XS))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text=f"{icon}  {event_kind_label(ev.kind)}",
                     font=FONT_SIDEBAR_HDR, text_color=TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(hdr, text=ev.severity.upper(),
                     font=FONT_BADGE, text_color=MUTED,
                     anchor="e").grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            card,
            text=f"  {ev.t_start:.2f} s → {ev.t_end:.2f} s  "
                 f"({ev.duration_s:.1f} s)   {ev.hr_mean:.0f} bpm",
            font=FONT_HINT, text_color=LIGHT, anchor="w",
        ).grid(row=1, column=1, sticky="ew", padx=(SPACE_XS, SPACE_S))

        # Delta vs baseline line (only for brady/tachy)
        if ev.baseline_hr > 0 and ev.kind in ("bradycardia", "tachycardia"):
            arrow  = "↓" if ev.delta_pct < 0 else "↑"
            d_col  = RED_LIGHT if ev.delta_pct < 0 else CORAL
            d_text = (f"  {arrow}{abs(ev.delta_pct):.0f}% vs baseline "
                      f"({ev.baseline_hr:.0f} → {ev.hr_mean:.0f} bpm)")
            if not getattr(ev, "baseline_reliable", True):
                d_text += "  — baseline unreliable"
            ctk.CTkLabel(
                card, text=d_text,
                font=FONT_SUBSECTION, text_color=d_col, anchor="w",
            ).grid(row=2, column=1, sticky="ew", padx=(SPACE_XS, SPACE_S))
            desc_row = 3
        else:
            desc_row = 2

        ctk.CTkLabel(
            card, text=f"  {ev.label}",
            font=FONT_KPI_LABEL, text_color=MUTED, anchor="w",
            wraplength=230, justify="left",
        ).grid(row=desc_row, column=1, sticky="ew", padx=(SPACE_XS, SPACE_S), pady=(0, SPACE_S))

        def _on_click(_event=None, _idx=idx):
            self.app._select_arrhythmia_event(_idx)

        for widget in [card] + list(card.winfo_children()):
            widget.bind("<Button-1>", _on_click)
            try:
                for sub in widget.winfo_children():
                    sub.bind("<Button-1>", _on_click)
            except Exception as e:
                log.debug("winfo_children bind failed: %s", e)

        self.app._arr_card_widgets.append(card)

    def toggle_arr_edit_mode(self) -> None:
        self.app.analysis.arr_edit_mode = not self.app.analysis.arr_edit_mode
        self.sync_arr_edit_btn()
        self.app._draw_arr_detail()

    def sync_arr_edit_btn(self) -> None:
        """The button and hint follow the mode, also on a rebuilt tab."""
        if self.app.analysis.arr_edit_mode:
            self.app.btn_arr_edit.configure(
                fg_color=ORANGE, hover_color=ORANGE_DEEP,
                text_color="white", text="Edit Mode ON",
            )
            self.app.lbl_arr_edit_hint.pack(side="left", padx=(SPACE_S, 0))
        else:
            self.app.btn_arr_edit.configure(
                fg_color=BORDER, hover_color=BORDER2,
                text_color=MUTED, text="Edit peaks here",
            )
            self.app.lbl_arr_edit_hint.pack_forget()

    def on_arr_detail_click(self, event) -> None:
        """Left-click: toggle exclusion.  Right-click: add/remove peak."""
        if not self.app.analysis.arr_edit_mode:
            return
        if event.xdata is None or self.app.signal.filtered is None:
            return
        if self.app.detection.rpeaks_ok is None:
            return

        fs         = self.app.signal.fs
        click_time = float(event.xdata)
        click_samp = int(np.clip(int(round(click_time * fs)),
                                 0, len(self.app.signal.filtered) - 1))

        tol_s    = max(MouseECG.MIN_RR_MS / 1000 / 2, self.app.analysis.arr_win * 0.03)
        tol_samp = int(tol_s * fs)
        is_left  = (event.button == 1)
        is_right = (event.button == 3)

        if is_right:
            # Remove manually added peak near click
            if (self.app.detection.rpeaks_manual_added is not None
                    and len(self.app.detection.rpeaks_manual_added)):
                dists = np.abs(self.app.detection.rpeaks_manual_added - click_samp)
                ni = int(np.argmin(dists))
                if dists[ni] <= tol_samp:
                    self.app._push_edit_undo()
                    self.app.detection.manual_added.discard(int(self.app.detection.rpeaks_manual_added[ni]))
                    self.app._run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
                    self.app._draw_arr_detail()
                    self.app._draw_detail(self.app.ui.nav_pos)
                    self.app._set_status(f"Peak removed at {click_time:.3f} s", ORANGE)
                    self.app._update_undo_btns()
                    return
            # Snap to local max and add
            sig  = self.app.signal.filtered
            lo   = max(0, click_samp - tol_samp)
            hi   = min(len(sig), click_samp + tol_samp + 1)
            new_samp = lo + int(np.argmax(sig[lo:hi]))
            if (self.app.detection.rpeaks_ok is not None and len(self.app.detection.rpeaks_ok)):
                if np.min(np.abs(self.app.detection.rpeaks_ok - new_samp)) < int(MouseECG.MIN_RR_MS / 1000 * fs * 0.5):
                    self.app._set_status(
                        f"Too close to existing peak at {click_time:.3f} s", ORANGE)
                    return
            self.app._push_edit_undo()
            self.app.detection.manual_added.add(new_samp)
            self.app.detection.manual_excluded.discard(new_samp)
            self.app._run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
            self.app._draw_arr_detail()
            self.app._draw_detail(self.app.ui.nav_pos)
            self.app._set_status(f"Peak added at {new_samp / fs:.3f} s", ORANGE)
            self.app._update_undo_btns()
            return

        if not is_left:
            return

        # Toggle exclusion of nearest peak
        added_set = self.app.detection.manual_added
        base_ok   = np.array([p for p in self.app.detection.rpeaks_ok if p not in added_set], int) \
                    if self.app.detection.rpeaks_ok is not None else np.array([], int)
        excl_arr  = self.app.detection.rpeaks_manual_excl if self.app.detection.rpeaks_manual_excl is not None \
                    else np.array([], int)
        candidates = np.concatenate([base_ok, excl_arr])
        if len(candidates) == 0:
            return
        dists = np.abs(candidates / fs - click_time)
        ni    = int(np.argmin(dists))
        if dists[ni] > tol_s:
            return
        peak_idx = int(candidates[ni])
        self.app._push_edit_undo()
        if peak_idx in self.app.detection.manual_excluded:
            self.app.detection.manual_excluded.discard(peak_idx)
            msg = f"Peak restored at {peak_idx / fs:.3f} s"
        else:
            self.app.detection.manual_excluded.add(peak_idx)
            msg = f"Peak excluded at {peak_idx / fs:.3f} s"
        self.app._run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
        self.app._draw_arr_detail()
        self.app._draw_detail(self.app.ui.nav_pos)
        self.app._set_status(msg + "  — click Analyze to refresh HRV", ORANGE)
        self.app._update_undo_btns()

    def on_arr_scroll(self, event) -> None:
        if event.xdata is None or self.app.signal.time is None:
            return
        factor   = 0.8 if event.button == "up" else 1.25
        new_win  = max(0.3, min(float(self.app.signal.time[-1]), self.app.analysis.arr_win * factor))
        cursor_x = float(event.xdata)
        frac     = (cursor_x - self.app.analysis.arr_nav_pos) / max(self.app.analysis.arr_win, 1e-6)
        t_start  = max(0.0, cursor_x - frac * new_win)
        t_start  = min(t_start, max(0.0, float(self.app.signal.time[-1]) - new_win))
        self.app.analysis.arr_win     = new_win
        self.app.analysis.arr_nav_pos = t_start
        try:
            self.app.ent_arr_win.delete(0, "end")
            self.app.ent_arr_win.insert(0, f"{new_win:.2f}")
        except Exception as e:
            log.debug("ent_arr_win update failed: %s", e)
        # Debounced (mirrors the Detection tab's on_detail_scroll ->
        # scroll_sync_after_id) so a burst of scroll ticks during one zoom
        # gesture triggers one full rebuild once scrolling settles, not one
        # fig.clear()+rebuild per tick. Unlike the Detection tab this
        # doesn't also do a cheap immediate set_xlim()+draw_idle() step --
        # draw_arr_detail() doesn't expose its axes between calls the way
        # the detail-view one does -- so visual feedback lands with the
        # debounce delay rather than instantly; still far better than one
        # full rebuild per tick.
        if self.app.ui.arr_scroll_after_id is not None:
            self.app.after_cancel(self.app.ui.arr_scroll_after_id)
        self.app.ui.arr_scroll_after_id = self.app.after(120, self._flush_arr_scroll)

    def _flush_arr_scroll(self) -> None:
        self.app.ui.arr_scroll_after_id = None
        self.app._draw_arr_detail()

    def copy_arrhythmia_tsv(self) -> None:
        tsv = getattr(self.app, "_arrhythmia_tsv", None)
        if not tsv:
            self.app._set_status("Run abnormal-event classification first.", RED)
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(tsv)
        self.app._set_status("Abnormal events copied to clipboard (Excel ready)", GREEN)

    def qtc_formula(self) -> str:
        """Return 'mitchell', 'bazett', or 'fridericia' -- the formula the
        stored QTc values use (``analysis.qtc_formula``, which the Morphology
        & Intervals selector sets). Kept in app state, not read off the combobox: a
        UI rebuild recreates the combobox, and the exports must name the
        formula the values were actually computed with."""
        return normalize_qtc_formula(getattr(self.app.analysis, "qtc_formula", "mitchell"))

    def on_qtc_formula_change(self, _choice: str = "") -> None:
        """Re-compute QTc with the selected formula and refresh interval plots."""
        try:
            choice = _choice or self.app.cb_qtc_formula.get()  # type: ignore[union-attr]
        except Exception:
            choice = self.qtc_formula()
        self.app.analysis.qtc_formula = normalize_qtc_formula(choice)
        if self.app.analysis.results is None:
            return
        ivl = self.app.analysis.results.get("intervals")
        if ivl is None or ivl.empty or "QT_ms" not in ivl.columns:
            return
        formula = self.qtc_formula()
        self.app.analysis.results["intervals"] = recompute_qtc_column(ivl, formula)
        self.update_interval_plots()
        # What reads the shared table would otherwise keep the old formula's QTc.
        try:
            self.app._plot_table_views(self.app.analysis.results)
        except Exception as exc:
            log.debug("on_qtc_formula_change: table views refresh failed: %s", exc)
        self.app._set_status(
            f"QTc recomputed — formula: {QTC_FORMULA_LABELS[formula]}  ✓", GREEN)

    def update_interval_plots(self) -> None:
        """Redraw the interval violin plots after a QTc formula change."""
        if self.app.analysis.results is None:
            return
        ivl = self.app.analysis.results.get("intervals")
        if ivl is None or ivl.empty:
            return
        try:
            self.app._plot_intervals(self.app.analysis.results)
        except Exception as exc:
            log.debug("_update_interval_plots: %s", exc)

    @staticmethod
    def _window_lf_hf(ep: np.ndarray, fs: float, rr_lo: float, rr_hi: float, breaks: Any,
                      bands: Any, t0: float) -> "tuple[float, float, float]":
        """(LF n.u. %, HF n.u. %, LF/HF) of one window -- the exact same
        analyse_hrv_freq() path as the Frequency panel (mouse-specific bands,
        Welch PSD, its own MIN_BEATS_SPECTRAL guard and gap-withholding rule).
        It returns an empty DataFrame -- NaN here -- rather than a number when a
        window is too short, or its excluded intervals too long, for a meaningful
        spectral estimate, instead of silently fabricating one."""
        try:
            fdf = analyse_hrv_freq(ep, fs, rr_min_ms=rr_lo, rr_max_ms=rr_hi,
                                   breaks=breaks, bands=bands)
            if not fdf.empty:
                return (float(fdf["HRV_LFn"].values[0]) * 100.0,
                        float(fdf["HRV_HFn"].values[0]) * 100.0,
                        float(fdf["HRV_LFHF"].values[0]))
        except Exception as exc:
            log.debug("_window_lf_hf: analyse_hrv_freq failed at t=%.1f: %s", t0, exc)
        return np.nan, np.nan, np.nan

    def _low_confidence(self, n_valid: Any, n_pairs: Any = None) -> np.ndarray:
        """The windows of Epochs and Sliding shown as hollow points: too few valid
        intervals for their metrics, or too few valid pairs for the LW ones."""
        low = np.asarray(n_valid) < self.MIN_CONFIDENT_BEATS
        if n_pairs is not None:
            low = low | (np.asarray(n_pairs) < POINCARE_MIN_PAIRS)
        return low

    def _frame_confidence(self, df: pd.DataFrame, valid: str, beats: str) -> "tuple[np.ndarray, str]":
        """(hollow points, footnote) of a stored frame. A frame saved before the
        valid counts were recorded is marked as it was then, by its beat count."""
        if valid in df.columns:
            return self._low_confidence(df[valid].values), ""
        if beats in df.columns:
            low = self._low_confidence(df[beats].values)
        else:
            low = np.zeros(len(df), dtype=bool)
        return low, (f"\n\n  * Saved by an older version: no {valid} column — hollow markers "
                     f"mean fewer than {self.MIN_CONFIDENT_BEATS} beats ({beats}), and "
                     f"excluded intervals are not counted. Compute again to get them.")

    def _blank_windowed_view(self, box: str, slot: str) -> None:
        """Empty the table (with its Excel TSV) and the figure of Epochs or Sliding."""
        widget = getattr(self.app, box, None)
        if widget is not None:
            self.app._set_textbox(widget, "")
        canvas = getattr(self.app, "_slots", {}).get(slot)
        if canvas is not None:
            canvas._draw_fn = None
            canvas._show_placeholder()

    def clear_rolling_view(self) -> None:
        """No Sliding frame: nothing of an earlier one stays on screen. Main thread only."""
        self._blank_windowed_view("txt_rolling", "rolling_hrv")
        lbl = getattr(self.app, "lbl_roll_status", None)
        if lbl is not None:
            lbl.configure(text="  Click Analyze first", text_color=MUTED)

    def compute_rolling_hrv(self) -> None:
        """Compute sliding-window HRV and render the timeline plot."""
        if self.app.detection.rpeaks_ok is None or self.app.signal.fs is None:
            self.app._set_status("Click Analyze first.", RED)
            return

        try:
            requested_win_s = float(self.app.ent_roll_win.get())
            requested_step_s = float(self.app.ent_roll_step.get())
        except ValueError:
            self.app._set_status("Invalid window / step.", RED)
            return
        if (not np.isfinite(requested_win_s)
                or not np.isfinite(requested_step_s)):
            self.app._set_status(
                "Window and step must be finite numbers.",
                RED)
            return
        win_s = max(3.0, requested_win_s)
        step_s = max(1.0, requested_step_s)
        if step_s > win_s:
            self.app._set_status(
                "Step must not exceed the window length.",
                RED)
            return

        active = [m for m, cb in self.app._roll_metrics.items() if cb.get()]
        if not active:
            self.app._set_status("Select at least one metric.", RED)
            return

        rpeaks = self.app._windowed_peaks()
        if rpeaks is None or len(rpeaks) < 5:
            self.app._set_status("Not enough peaks in the analysis window.", ORANGE)
            return
        fs     = float(self.app.signal.fs)
        _gen = getattr(self.app, "_generation", 0)  # snapshot — detect file change

        # LF_nu/HF_nu/LF_HF need a per-window Welch PSD (analyse_hrv_freq) --
        # meaningfully more expensive than the O(1) time-domain stats below,
        # so only pay for it when the user actually selected one of those
        # metrics. SD1/SD2 are cheap closed-form Poincare formulas and are
        # always computed alongside the existing time-domain set.
        need_freq = bool({"LF_nu", "HF_nu", "LF_HF"} & set(active))
        need_article_poincare = bool({"CCD", "SD1_LW", "SD2_LW"} & set(active))
        # Read on the main thread (Tk widgets / app state) before the worker.
        rr_lo, rr_hi = self.hrv_rr_bounds()
        breaks       = self.hrv_breaks()
        bands        = self.get_freq_bands()
        basis        = self._capture_basis()

        def _worker():
            t_peaks  = rpeaks / fs
            t_start  = t_peaks[0]
            t_end    = t_peaks[-1]
            starts   = np.arange(t_start, t_end - win_s + step_s * 0.5, step_s)
            rows    = []
            n_skipped = 0
            n_wins  = len(starts)
            for i, t0 in enumerate(starts):
                t1   = t0 + win_s
                mask = (t_peaks >= t0) & (t_peaks < t1)
                ep   = rpeaks[mask]
                # Same inclusion floor as compute_epochs: a window with too
                # few peaks (detection dropout, long pause) or too few valid
                # intervals is skipped, not emitted as a row of NaN.
                if len(ep) < self.MIN_WINDOW_BEATS:
                    n_skipped += 1
                    continue
                # Gap-aware, like analyse_core: out-of-range RR intervals and
                # intervals spanning a removed beat are excluded, and a
                # successive difference is only taken between two adjacent
                # valid intervals (never across a gap).
                ser    = rr_interval_series(ep, fs, rr_lo, rr_hi, breaks=breaks)
                rr_all = ser["rr_ms"]
                rr     = ser["rri"]
                if len(rr) < self.MIN_VALID_INTERVALS:
                    n_skipped += 1
                    continue
                rr_a, rr_b = successive_valid_pairs(rr_all, ser["valid"])
                sdiff  = rr_b - rr_a
                diffs  = np.abs(sdiff)
                hr   = mean_hr_bpm(rr)                 # 60000 / mean NN
                sdnn = float(rr.std(ddof=1))        if len(rr) > 1 else np.nan
                rmssd = float(np.sqrt(np.mean(sdiff**2))) if len(sdiff) > 1 else np.nan
                pnn6  = float(100.0 * np.sum(diffs > MouseECG.PNN_THRESHOLD) / len(diffs)) \
                        if len(diffs) else np.nan
                # Poincare SD1/SD2 -- the app-wide definition
                # (poincare_sd1_sd2, also used by the Non-linear panel,
                # Compare Segments and the Report), over successive valid
                # pairs only. Cheap enough to always compute. SD1_LW/SD2_LW
                # below are the explicitly named Ledoit-Wolf variant.
                _pc = poincare_sd1_sd2(rr_all, ser["valid"])
                sd1, sd2 = float(_pc["sd1"]), float(_pc["sd2"])
                article_poincare = (
                    # every interval + the window's validity mask (bounds
                    # and removed-beat breaks): pairs never straddle a gap
                    compute_poincare_metrics(rr_all, ser["valid"]) if need_article_poincare else {
                        "CCD": np.nan,
                        "SD1_LW": np.nan,
                        "SD2_LW": np.nan,
                        "n_poincare_pairs": np.nan,
                    }
                )
                lf_nu = hf_nu = lf_hf = np.nan
                if need_freq:
                    lf_nu, hf_nu, lf_hf = self._window_lf_hf(
                        ep, fs, rr_lo, rr_hi, breaks, bands, t0)
                rows.append({
                    "t_mid": round(t0 + win_s / 2, 2),
                    "HR":    round(hr,    1),
                    "SDNN":  round(sdnn,  2),
                    "RMSSD": round(rmssd, 2),
                    "pNN6":  round(pnn6,  1),
                    "SD1":   round(sd1, 3) if np.isfinite(sd1) else np.nan,
                    "SD2":   round(sd2, 3) if np.isfinite(sd2) else np.nan,
                    "CCD":   round(float(article_poincare["CCD"]), 3)
                             if np.isfinite(article_poincare["CCD"]) else np.nan,
                    "SD1_LW": round(float(article_poincare["SD1_LW"]), 3)
                              if np.isfinite(article_poincare["SD1_LW"]) else np.nan,
                    "SD2_LW": round(float(article_poincare["SD2_LW"]), 3)
                              if np.isfinite(article_poincare["SD2_LW"]) else np.nan,
                    "LF_nu": round(lf_nu, 1) if np.isfinite(lf_nu) else np.nan,
                    "HF_nu": round(hf_nu, 1) if np.isfinite(hf_nu) else np.nan,
                    "LF_HF": round(lf_hf, 3) if np.isfinite(lf_hf) else np.nan,
                    "n_beats": len(ep),
                    "n_valid": len(rr),
                    "n_excluded": ser["n_excluded"],
                    "n_poincare_pairs": article_poincare["n_poincare_pairs"],
                })
                pct = int((i + 1) / max(n_wins, 1) * 100)
                if i % max(1, n_wins // 20) == 0:
                    self.app.after(0, lambda p=pct, ii=i+1, tot=n_wins:
                               self.app._set_progress(p, f"Window {ii}/{tot}…"))
            return {"rows": rows, "n_skipped": n_skipped}

        def _done(result):
            self._mark_landed("rolling")
            if getattr(self.app, "_generation", 0) != _gen:
                log.info("compute_rolling_hrv: stale result discarded (file changed)")
                return
            rows, n_skipped = result["rows"], result["n_skipped"]
            if not rows:
                self.app.lbl_roll_status.configure(  # type: ignore[union-attr]
                    text=(f"No valid windows — need ≥ {self.MIN_VALID_INTERVALS} "
                          f"valid RR intervals per window (recording too short?)"),
                    text_color=RED)
                return

            df = pd.DataFrame(rows)
            self.app.analysis.rolling_hrv_df = df
            self._stamp_basis("rolling", basis)
            self.app.analysis.windowed_view["rolling"] = {
                "win_s": win_s, "step_s": step_s, "active": active,
                "rr": [rr_lo, rr_hi], "n_skipped": n_skipped}
            n_excl_win = self.show_rolling()
            n = len(df)
            skip_note = f" · {n_skipped} skipped" if n_skipped else ""
            excl_note = (f"  ·  {n_excl_win} window(s) with excluded RR"
                         if n_excl_win else "")
            # Keep the core results' exclusion disclosure, as run_freq does.
            core_note = self.exclusion_status_note()
            self.app._set_status(
                f"Rolling HRV — {n} windows computed{skip_note}{excl_note}{core_note}",
                ORANGE if (n_excl_win or core_note) else GREEN)
            self.app.tabs.set(TAB_TIME_COURSE)
            self.app.after(50, lambda: self.app._show_time_mode(MODE_SLIDING))

        self._mark_running("rolling")
        self.app._start_async_result(
            self.app.btn_roll_compute, "Computing…", _worker, _done)

    def show_rolling(self) -> int:
        """Draw the stored Sliding frame -- figure, table (with its Excel TSV) and
        status label -- from what it was computed with; a frame saved without
        that record shows its ticked metrics and no window length. Returns the
        number of windows with excluded RR; without a frame the view is emptied
        and it returns 0. Main thread only."""
        df = self.app.analysis.rolling_hrv_df
        if df is None or df.empty:
            self.clear_rolling_view()
            return 0
        view = self.app.analysis.windowed_view.get("rolling") or {}
        t = df["t_mid"].to_numpy(dtype=float)
        step_s = float(view.get("step_s") or (np.median(np.diff(t)) if len(t) > 1 else 1.0))
        win_s = view.get("win_s")
        where = (f"window {float(win_s):.0f}s · step {step_s:.0f}s" if win_s
                 else f"step {step_s:.0f}s")
        active = [m for m in (view.get("active") or
                              [m for m, cb in self.app._roll_metrics.items() if cb.get()])
                  if m in df.columns and df[m].notna().any()]
        active = active or ["HR"]
        rr_lo, rr_hi = view.get("rr") or self.hrv_rr_bounds(quiet=True)
        n_skipped = int(view.get("n_skipped", 0))
        need_freq = bool({"LF_nu", "HF_nu", "LF_HF"} & set(active))
        need_article_poincare = bool({"CCD", "SD1_LW", "SD2_LW"} & set(active))
        # Confidence from the intervals the metrics were computed from
        low_n, old_note = self._frame_confidence(df, "n_valid", "n_beats")
        if need_article_poincare and "n_poincare_pairs" in df.columns:
            low_n = low_n | (df["n_poincare_pairs"].values < POINCARE_MIN_PAIRS)

        colors = {"HR": ORANGE_DARK, "SDNN": BLUE_DARK,
                  "RMSSD": GREEN_DARK, "pNN6": PURPLE,
                  "SD1": CORAL, "SD2": CYAN_BRIGHT,
                  "CCD": NAVY, "SD1_LW": CORAL, "SD2_LW": CYAN_BRIGHT,
                  "LF_nu": BLUE_MID, "HF_nu": GREEN, "LF_HF": AMBER}
        ylabels = {"HR": "HR (bpm)", "SDNN": "SDNN (ms)",
                   "RMSSD": "RMSSD (ms)", "pNN6": "pNN6 (%)",
                   "SD1": "SD1 (ms)", "SD2": "SD2 (ms)",
                   # Ledoit-Wolf, not the Autonomic view's "robust"
                   # (Schafer-Strimmer) estimator -- named as such.
                   "CCD": "CCD (ms)", "SD1_LW": "SD1 LW (ms)",
                   "SD2_LW": "SD2 LW (ms)",
                   "LF_nu": "LF (n.u., %)", "HF_nu": "HF (n.u., %)",
                   "LF_HF": "LF/HF"}
        panels = [(m, ylabels[m], colors[m], f"{m}  —  {where}") for m in active]
        # Annotations are read at draw time, like the Autonomic figure's.
        self.app._slots["rolling_hrv"].update(lambda fig: draw_windowed_panels(
            fig, t, df, panels, low_n, step_s * 1.5, list(self.app.analysis.annotations)))
        table_text = df.to_string(index=False) + old_note
        if low_n.any() and not old_note:
            table_text += (
                f"\n\n  * {int(low_n.sum())} window(s) have fewer than "
                f"{self.MIN_CONFIDENT_BEATS} valid RR intervals (n_valid) — "
                f"their metrics are noisy.")
            if need_article_poincare:
                table_text += (
                    f" Poincare metrics are withheld below "
                    f"{POINCARE_MIN_PAIRS} valid successive RR pairs.")
            table_text += " Hollow markers identify low-confidence windows."
        n_excl_win = int((df["n_excluded"].values > 0).sum()) if "n_excluded" in df.columns else 0
        if n_excl_win:
            bounds = rr_bounds_label({"rr_min_ms": rr_lo, "rr_max_ms": rr_hi})
            table_text += (
                f"\n\n  * {n_excl_win} window(s) contain RR intervals outside "
                f"{bounds} or spanning removed beats (n_excluded) — "
                f"excluded from every metric, not bridged.")
            if need_freq:
                table_text += (
                    " LF/HF are withheld (blank) in windows whose excluded "
                    "intervals are too long to interpolate across.")
        if n_skipped:
            table_text += (
                f"\n\n  * {n_skipped} window(s) skipped — fewer than "
                f"{self.MIN_VALID_INTERVALS} valid RR intervals (or "
                f"{self.MIN_WINDOW_BEATS} beats); no row, not NaN, and a "
                f"gap in the plotted lines.")
        self.app._set_textbox(self.app.txt_rolling, table_text,
                              tsv=self.app._df_to_tsv(df))
        skip_note = f" · {n_skipped} skipped" if n_skipped else ""
        self.app.lbl_roll_status.configure(  # type: ignore[union-attr]
            text=f"  {len(df)} windows{skip_note} · {where.removeprefix('window ')}  ✓",
            text_color=ORANGE if (n_skipped or n_excl_win) else GREEN)
        return n_excl_win

    # ── Autonomic indices (Time course > Autonomic) ───────────────────
    # Time-resolved cardiac sympathetic / parasympathetic indices (CSI /
    # CPI) of Candia-Rivera et al., J Physiol 2025 (core.autonomic), plus
    # the optional SPWVD LF / HF (core.spwvd). Same inputs as every other
    # HRV path: the analysis window's peaks, the user's RR bounds and the
    # removed-beat breaks. The parameters are app state
    # (analysis.autonomic_params: they survive a UI rebuild and are saved
    # with the session); the result is dropped with every other
    # peak-derived result (invalidate_autonomic), and recomputed, not saved.

    # Rough SPWVD cost (s of compute per s of recording) for the progress
    # message only: ~7 s per 10 min at 20 Hz / 4096 bins (60 s margins).
    _SPWVD_S_PER_S = 0.012
    AUTONOMIC_IDLE_TEXT = "  Click ⟳ Compute — uses the detected peaks"
    AUTONOMIC_PEAKS_STALE_TEXT = "  Peaks changed — click ⟳ Compute again"

    @staticmethod
    def autonomic_min_peaks(min_pairs: int) -> int:
        """Peaks below which no window can hold *min_pairs* valid pairs:
        *min_pairs* pairs need min_pairs + 1 intervals (12 peaks at the
        default 10 pairs)."""
        return max(4, int(min_pairs) + 2)

    def autonomic_widget_params(self) -> dict:
        """The Autonomic parameters as currently entered, read tolerantly:
        a missing widget or an unreadable value keeps the stored value
        (analysis.autonomic_params). For a session save and a UI rebuild,
        which must not fail on a half-typed entry. Main thread only."""
        raw: "dict[str, Any]" = {}
        for key, attr in (("window_s", "ent_auto_win"), ("method", "cb_auto_method"),
                          ("ks", "ent_auto_ks"), ("kp", "ent_auto_kp"),
                          ("lf_hf", "chk_auto_lfhf"), ("min_pairs", "ent_auto_minpairs")):
            w = getattr(self.app, attr, None)
            if w is None:
                continue
            try:
                raw[key] = w.get()
            except Exception as exc:
                log.debug("autonomic_widget_params: %s unreadable: %s", attr, exc)
        return normalise_autonomic_params(raw, base=self.app.analysis.autonomic_params)

    def apply_autonomic_params(self, params: Any) -> dict:
        """Make *params* the Autonomic parameters in effect -- normalised,
        a missing or invalid key keeping the current value (so a session
        saved before the view existed changes nothing) -- and show them in
        the widgets. Returns the parameters applied. Main thread only."""
        p = normalise_autonomic_params(params, base=self.app.analysis.autonomic_params)
        self.app.analysis.autonomic_params = p
        for attr, key in (("ent_auto_win", "window_s"), ("ent_auto_ks", "ks"),
                          ("ent_auto_kp", "kp"), ("ent_auto_minpairs", "min_pairs")):
            w = getattr(self.app, attr, None)
            if w is None:
                continue
            try:
                w.delete(0, "end")
                w.insert(0, format_param(p[key]))
            except Exception as exc:
                log.debug("apply_autonomic_params: %s: %s", attr, exc)
        cb = getattr(self.app, "cb_auto_method", None)
        if cb is not None:
            try:
                cb.set(COV_METHODS[p["method"]])
            except Exception as exc:
                log.debug("apply_autonomic_params: method combobox: %s", exc)
        chk = getattr(self.app, "chk_auto_lfhf", None)
        if chk is not None:
            try:
                chk.select() if p["lf_hf"] else chk.deselect()
            except Exception as exc:
                log.debug("apply_autonomic_params: LF/HF checkbox: %s", exc)
        return p

    def on_autonomic_method_change(self, choice: str) -> None:
        """Method combobox: show what the chosen estimator does."""
        key = method_key(choice)
        if key is not None:
            self.app._set_status(f"{COV_METHODS[key]}: {METHOD_NOTES[key]}", MUTED)

    def _set_autonomic_label(self, text: str, color: str) -> None:
        lbl = getattr(self.app, "lbl_auto_status", None)
        if lbl is not None:
            lbl.configure(text=text, text_color=color)

    def invalidate_autonomic(self, peaks_changed: bool = True) -> None:
        """Drop the Autonomic result, its LF / HF and the event-locked table
        derived from it, and blank the view -- wherever the accepted peaks
        change (edits, re-detection, artifact review, Analyze's correction)
        or a new file is loaded; *peaks_changed* False (new file) shows the
        idle hint instead of "peaks changed". The parameters are kept.
        Main thread only."""
        a = self.app.analysis
        had = a.autonomic_result is not None
        a.autonomic_result = None
        a.autonomic_spwvd = None
        self._invalidate_event_locked("peaks" if peaks_changed else None)
        slot = (getattr(self.app, "_slots", None) or {}).get("autonomic")
        if slot is not None:
            try:
                slot._draw_fn = None
                slot._show_placeholder()
            except Exception as exc:
                log.debug("invalidate_autonomic: slot reset failed: %s", exc)
        txt = getattr(self.app, "txt_autonomic", None)
        if had and txt is not None and hasattr(self.app, "_set_textbox"):
            self.app._set_textbox(txt, "")
        if had and peaks_changed:
            # Kept in state, so a UI rebuild shows it again (show_autonomic).
            a.autonomic_stale_note = self.AUTONOMIC_PEAKS_STALE_TEXT
            self._set_autonomic_label(self.AUTONOMIC_PEAKS_STALE_TEXT, ORANGE)
        elif not peaks_changed:
            a.autonomic_stale_note = None
            self._set_autonomic_label(self.AUTONOMIC_IDLE_TEXT, MUTED)

    def _invalidate_event_locked(self, reason: "str | None" = None) -> None:
        """Drop the event-locked result derived from the Autonomic result
        (Time course > Event-locked) and blank its panel; *reason*
        ("peaks", "autonomic", "annotations", None = idle) is what the
        panel's label says. Main thread only."""
        ctrl = getattr(self.app, "event_ctrl", None)
        if ctrl is not None:
            ctrl.invalidate(reason)          # clears the state too
            return
        a = self.app.analysis
        a.event_locked = None
        a.event_locked_df = None

    @staticmethod
    def _same_peaks(a: "np.ndarray | None", b: "np.ndarray | None") -> bool:
        """True when *a* and *b* are the same peak set (identity first --
        rpeaks_ok is reassigned, never mutated, whenever the peaks change)."""
        if a is b:
            return True
        if a is None or b is None:
            return False
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))

    def redraw_autonomic_plot(self) -> None:
        """Re-render the Autonomic figure from the stored result (no-op
        without one) -- e.g. after the annotations changed."""
        res = self.app.analysis.autonomic_result
        slot = (getattr(self.app, "_slots", None) or {}).get("autonomic")
        if res is None or slot is None:
            return
        spwvd = self.app.analysis.autonomic_spwvd
        # Annotations are read at draw time, so a resize replay shows the
        # current ones too.
        slot.update(lambda fig: draw_autonomic(
            fig, res, spwvd, list(self.app.analysis.annotations)))

    def redraw_windowed_plots(self) -> None:
        """Re-render Epochs and Sliding from what they show -- after the annotations
        changed. No-op on a placeholder."""
        slots = getattr(self.app, "_slots", None) or {}
        for name in ("epochs", "rolling_hrv"):
            slot = slots.get(name)
            if slot is not None and slot._draw_fn is not None:
                slot.update(slot._draw_fn)

    def autonomic_stale_reasons(self) -> "list[str]":
        """Why the displayed Autonomic result no longer matches the view:
        the analysis window was changed since it was computed (it records
        the window in ``params['analysis_window']``), or the entries hold
        other parameters than it was computed with (e.g. ks / kp written by
        the calibration dialog, not applied yet). Empty without a result.
        Main thread only."""
        a = self.app.analysis
        res = a.autonomic_result
        if res is None:
            return []
        out = []
        now = (getattr(a, "t_start", 0.0), getattr(a, "t_end", 0.0))
        if "analysis_window" in res.params and not same_window(res.params["analysis_window"], now):
            out.append(f"computed on the analysis window "
                       f"{analysis_window_text(res.params)}, now "
                       f"{analysis_window_text({'analysis_window': now})}")
        diff = params_differ(self.autonomic_widget_params(), res.params,
                             a.autonomic_spwvd is not None)
        if diff:
            out.append("entries differ from the displayed result: " + ", ".join(diff))
        return out

    def show_autonomic(self) -> "tuple[str, str] | None":
        """Draw the stored Autonomic result -- figure, recording-level table
        (with its Excel TSV) and status label. The label also says when the
        result is stale (``autonomic_stale_reasons``: analysis window
        changed, entries not applied yet), in orange, so a UI rebuild keeps
        those notices. Returns the label's (text, colour), or None without a
        result (the label then shows the stored "Peaks changed" notice, if
        any). Main thread only."""
        a = self.app.analysis
        res = a.autonomic_result
        if res is None:
            note = getattr(a, "autonomic_stale_note", None)
            if note:
                self._set_autonomic_label(note, ORANGE)
            return None
        spwvd = a.autonomic_spwvd
        self.redraw_autonomic_plot()
        txt = getattr(self.app, "txt_autonomic", None)
        if txt is not None:
            self.app._set_textbox(txt, summary_text(res, spwvd),
                                  tsv=self.app._df_to_tsv(summary_frame(res, spwvd)))
        return self.refresh_autonomic_status()

    def refresh_autonomic_status(self) -> "tuple[str, str] | None":
        """Re-derive the Autonomic status label from the stored result and
        its staleness (e.g. after the analysis window changed) without
        redrawing anything. Returns (text, colour), None without a result.
        Main thread only."""
        a = self.app.analysis
        res = a.autonomic_result
        if res is None:
            return None
        text, level = autonomic_status(res, a.autonomic_spwvd)
        color = GREEN if level == "ok" else ORANGE
        stale = self.autonomic_stale_reasons()
        if stale:
            text = (text.removesuffix("  ✓") + "  ·  ⚠ " + "; ".join(stale)
                    + " — press ⟳ Compute")
            color = ORANGE
        self._set_autonomic_label(text, color)
        return text, color

    def compute_autonomic(self, then: "Any" = None) -> None:
        """Compute the time-resolved CSI / CPI (and, when ticked, the SPWVD
        LF / HF) of the analysis window's peaks in the background, then
        draw the Autonomic view. *then* runs once a result has landed."""
        app = self.app
        if app.detection.rpeaks_ok is None or app.signal.fs is None:
            app._set_status("No peaks yet — click Analyze first.", RED)
            return
        w_pairs = getattr(app, "ent_auto_minpairs", None)
        try:
            p = parse_autonomic_inputs(
                app.ent_auto_win.get(), app.cb_auto_method.get(),
                app.ent_auto_ks.get(), app.ent_auto_kp.get(),
                app.chk_auto_lfhf.get(),
                min_pairs=(w_pairs.get() if w_pairs is not None else
                           normalise_autonomic_params(app.analysis.autonomic_params)["min_pairs"]))
        except ValueError as exc:
            self._set_autonomic_label(f"  {exc}", RED)
            app._set_status(f"Autonomic: {exc}", RED)
            return
        app.analysis.autonomic_params = p

        rpeaks = app._windowed_peaks()
        need = self.autonomic_min_peaks(p["min_pairs"])
        if rpeaks is None or len(rpeaks) < need:
            msg = f"Not enough peaks in the analysis window (need ≥ {need})."
            self._set_autonomic_label(f"  {msg}", ORANGE)
            app._set_status(msg, ORANGE)
            return
        fs = float(app.signal.fs)
        # Read on the main thread (Tk widgets / app state) before the worker.
        rr_lo, rr_hi = self.hrv_rr_bounds()
        breaks = self.hrv_breaks()
        _vlf, lf_band, hf_band = self.get_freq_bands()
        gen = getattr(app, "_generation", 0)        # snapshot — detect file change
        peaks_ref = app.detection.rpeaks_ok         # snapshot — detect peak edits
        span_s = float(rpeaks[-1] - rpeaks[0]) / fs
        # The analysis window the peaks come from, recorded in the result so
        # a later window change flags it as stale (autonomic_stale_reasons)
        # and the exports state which span it covers.
        window = (float(getattr(app.analysis, "t_start", 0.0) or 0.0),
                  float(getattr(app.analysis, "t_end", 0.0) or 0.0))
        peak_span = (float(rpeaks[0]) / fs, float(rpeaks[-1]) / fs)
        basis = self._capture_basis()

        def _progress(pct: int, msg: str) -> None:
            app.after(0, lambda: app._set_progress(pct, msg))

        def _worker() -> dict:
            _progress(5, "Poincaré windows…")
            res = compute_autonomic_indices(
                rpeaks, fs, window_s=p["window_s"], method=p["method"],
                ks=p["ks"], kp=p["kp"], rr_min_ms=rr_lo, rr_max_ms=rr_hi,
                breaks=breaks, min_pairs=p["min_pairs"])
            res.params["analysis_window"] = window
            res.params["peak_span_s"] = peak_span
            spwvd = None
            if p["lf_hf"]:
                est = span_s * self._SPWVD_S_PER_S
                _progress(30, "LF / HF (SPWVD)" + (f" — about {est:.0f} s…" if est >= 5 else "…"))
                ser = rr_interval_series(rpeaks, fs, rr_lo, rr_hi, breaks=breaks)
                try:
                    spwvd = spwvd_lf_hf(ser["rr_ms"], ser["t_s"], ser["valid"],
                                        lf_band=tuple(lf_band), hf_band=tuple(hf_band))
                except ValueError as exc:
                    # Too short, a band outside the resampled range, the
                    # memory guard: disclosed, the CSI / CPI still shown.
                    spwvd = {"error": str(exc)}
            _progress(95, "Drawing…")
            return {"result": res, "spwvd": spwvd}

        def _done(out: dict) -> None:
            self._mark_landed("autonomic")
            if getattr(app, "_generation", 0) != gen:
                log.info("compute_autonomic: stale result discarded (file changed)")
                return
            if not self._same_peaks(app.detection.rpeaks_ok, peaks_ref):
                log.info("compute_autonomic: stale result discarded (peaks changed)")
                self._set_autonomic_label(
                    "  Peaks changed during the computation — click ⟳ Compute again",
                    ORANGE)
                app._set_status("Autonomic indices discarded — the peaks changed "
                                "while computing.", ORANGE)
                return
            res, spwvd = out["result"], out["spwvd"]
            app.analysis.autonomic_result = res
            app.analysis.autonomic_spwvd = spwvd
            app.analysis.autonomic_stale_note = None
            self._stamp_basis("autonomic", basis)
            # The event-locked table derives from the previous result.
            self._invalidate_event_locked("autonomic")
            text, color = self.show_autonomic() or ("", ORANGE)
            app._set_status(f"Autonomic indices — {text.strip()}", color)
            if then is not None:
                then()

        self._mark_running("autonomic")
        app._start_async_result(app.btn_auto_compute, "Computing…", _worker, _done)

    def run_analysis(self) -> None:
        if not NK_AVAILABLE:
            messagebox.showerror("Missing", "pip install neurokit2")
            return
        if self.app.signal.filtered is None or self.app.detection.rpeaks_ok is None:
            # Run the missing prerequisite step instead of just blocking on a
            # modal the user has to dismiss and act on by hand -- preview()
            # opens a file picker first if nothing is loaded yet, then chains
            # straight back into Analyze once detection succeeds.
            self.app._set_status("No peaks yet — running Detect Peaks first…", ORANGE)
            self.app.signal_ctrl.preview(then_analyze=True)
            return
        if len(self.app.detection.rpeaks_ok) < 5:
            messagebox.showwarning(
                "Too few peaks",
                f"Only {len(self.app.detection.rpeaks_ok)} peaks detected.\n"
                "Adjust threshold / detection settings.")
            return
        # Snapshot ALL widget values on the main thread before spawning the worker.
        # The background thread must never call .get() on any Tkinter widget --
        # doing so races with the event loop and causes freezes / crashes.
        params = self.app._snapshot_params()
        # HRV acceptance bounds (validated) and the artifact report whose
        # removed beats are still absent -- both main-thread reads.
        params["hrv_rr_bounds"] = self.hrv_rr_bounds()
        params["break_tol_samples"] = self.break_tol_samples()
        prev_report = self.app.analysis.artifact_report
        self._mark_running("core", "frequency", "nonlinear")
        self.app._start_async(
            self.app.btn_run, "Analysing\u2026", "Running HRV analysis\u2026",
            lambda: self.analysis_worker(params, prev_report),
            self.on_analysis_done,
            pass_result=True,
        )

    def analysis_worker(self, params: dict,
                        prev_report: "dict | None" = None) -> dict:
        """Background worker — MUST NOT write to self.

        Runs core analysis and optional artifact correction, then returns a
        plain bundle.  ``_on_analysis_done`` writes all results to self on
        the main thread, preventing races with Tk draw callbacks.

        *prev_report* is the artifact report in effect before this run (an
        earlier auto-correction or a reviewed one). Its removals whose peak
        is still absent stay in the report -- counts and HRV breaks alike
        (merge_artifact_reports): the peaks handed to this run are already
        corrected, so a fresh correction pass finds nothing to remove and
        would otherwise report "no artifacts" while the breaks persist.
        """
        if self.app.detection.rpeaks_ok is None:
            raise RuntimeError("No peaks available — click Detect Peaks first.")
        # Take a snapshot of the peaks at worker-start time.  After this point
        # the worker operates only on local variables — no self writes.
        rp = self.app.detection.rpeaks_ok.copy()

        def _prog(pct: int, msg: str) -> None:
            self.app.after(0, lambda p=pct, m=msg: self.app._set_progress(p, m))

        rr_lo, rr_hi = params.get("hrv_rr_bounds",
                                  (MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS))
        pass_report = None
        if params["artifact_correction"]:
            _prog(2, "Artifact correction (auto)…")
            try:
                rp_corrected, pass_report = correct_rr_artifacts(
                    rp, self.app.signal.fs,
                    rr_min_ms=params.get("min_rr_ms", MouseECG.RR_MIN_MS),
                    rr_max_ms=rr_hi,
                    window_beats=11, dev_threshold=0.20,
                    signal=self.app.signal.filtered,
                )
                rp = rp_corrected
                removed = pass_report["n_in"] - pass_report["n_out"]
                log.info("Artifact correction: −%d peaks (non-physio=%d ectopic=%d dup=%d, "
                         "%d HRV breaks)",
                         removed, pass_report["n_nonphysio"], pass_report["n_ectopic"],
                         pass_report["n_duplicate"], len(pass_report["break_samples"]))
            except Exception as exc:
                log.warning("Artifact correction failed: %s", exc)
                pass_report = None
        # Cumulative report: this pass's removals plus earlier ones whose peak
        # is still absent, counts recomputed from them. Without a new pass
        # (correction off or failed) the earlier report is kept while any of
        # its removals is still in effect, and dropped once all are back.
        artifact_report = merge_artifact_reports(
            prev_report, pass_report, rp, params.get("break_tol_samples", 1))
        breaks = (np.asarray(artifact_report["break_samples"], dtype=np.int64)
                  if artifact_report and artifact_report["break_samples"] else None)

        # ── Save full (artifact-corrected) peak set BEFORE windowing ─────────
        # This is what _on_analysis_done will write back to _rpeaks_ok so that
        # changing the analysis window on the next run always starts from the
        # complete detection result — not a previously-windowed subset.
        rp_full = rp.copy()

        # ── Apply analysis window AFTER artifact correction ───────────────────
        # Window is applied to the analysis only — rp_full is always preserved.
        ana_t0 = params.get("analysis_t_start", 0.0)
        ana_t1 = params.get("analysis_t_end",   0.0)
        if (ana_t0 > 0 or ana_t1 > 0) and self.app.signal.fs is not None:
            fs_snap = self.app.signal.fs
            mask = rp / fs_snap >= ana_t0
            if ana_t1 > 0:
                mask &= rp / fs_snap <= ana_t1
            rp_windowed = rp[mask]
            if len(rp_windowed) < 5:
                raise ValueError(
                    f"Analysis window too short: only {len(rp_windowed)} peaks "
                    f"between {ana_t0:.1f} s and "
                    f"{'the end' if ana_t1 == 0 else f'{ana_t1:.1f} s'}.\n"
                    "Widen the window or click Full to reset it.")
            rp = rp_windowed
            log.info("Analysis window applied: %.1f s → %s  (%d / %d peaks)",
                     ana_t0, f"{ana_t1:.1f} s" if ana_t1 > 0 else "end",
                     len(rp), len(rp_full))

        _prog(10, "Core analysis (RR, HR, time-domain HRV, beat template)…")
        if self.app.signal.filtered is None:
            raise RuntimeError("Signal not loaded — click Detect Peaks first.")
        results = analyse_core(self.app.signal.filtered, rp, self.app.signal.fs,
                               progress_cb=lambda p, m: _prog(10 + int(p * 0.9), m),
                               rr_min_ms=rr_lo, rr_max_ms=rr_hi, breaks=breaks)

        return {
            "results":         results,
            "rpeaks_ok":       rp_full,   # ← always full set: never overwrite with windowed
            "rpeaks_analysed": rp,         # ← windowed subset used for this analysis
            "artifact_report": artifact_report,
            "auto_epochs":     params.get("auto_epochs", False),
            "analysis_window": (float(ana_t0), float(ana_t1)),
        }

    def on_analysis_done(self, bundle: dict) -> None:
        # ── Atomic state update (main thread) ────────────────────────────────
        self.app.analysis.results         = bundle["results"]
        # Analyze's artifact correction can change the peaks: the Autonomic
        # result (computed from the previous ones) is then stale.
        if not self._same_peaks(self.app.detection.rpeaks_ok, bundle["rpeaks_ok"]):
            self.invalidate_autonomic()
        # rpeaks_ok in the bundle is always the FULL artifact-corrected set,
        # never the windowed subset — so the signal view and next analysis
        # always start from the complete detection result.
        self.app.detection.rpeaks_ok       = bundle["rpeaks_ok"]
        self.app.analysis.artifact_report = bundle["artifact_report"]
        self.app.analysis.analysed_window = bundle.get(
            "analysis_window", (self.app.analysis.t_start, self.app.analysis.t_end))

        self._mark_landed("core")
        if self.app.analysis.results is None:
            return
        # Taken from the bundle, not the entries: the correction can change
        # the peaks, and the window / RR bounds may have been edited meanwhile.
        self._stamp_basis("core", self._capture_basis(
            window=self.app.analysis.analysed_window,
            rr_bounds=recorded_rr_bounds(self.app.analysis.results)))
        ui = getattr(self.app, "ui", None)
        if ui is not None:
            ui.computed_at = dt.datetime.now().isoformat(timespec="seconds")
        # Sync the peak label: show windowed count if a window was active,
        # otherwise show total artifact-corrected count.
        _rp_ok   = self.app.detection.rpeaks_ok
        _rp_used = bundle.get("rpeaks_analysed", _rp_ok)  # windowed subset

        # Re-score signal quality now that beat_corr exists -- update_signal_
        # quality() falls back to an RR-regularity heuristic when it's None
        # (i.e. always, at the earlier Detect Peaks-time call), which reads
        # a genuine bradycardia/pause as "noisy" even on a clean recording.
        # The gauge otherwise stays pinned to that provisional score forever,
        # visibly disagreeing with the beat-shape-correlation badge below it.
        if _rp_ok is not None:
            self.app.detection_ctrl.update_signal_quality(_rp_ok)
        n_total  = len(_rp_ok)  if _rp_ok   is not None else 0
        n_used   = len(_rp_used) if _rp_used is not None else 0
        n_peaks  = n_used  # use analysed count for status messages

        windowed = (n_used < n_total)
        if self.app.lbl_npeaks is not None:
            _c = GREEN if n_total > 10 else RED
            arep = bundle.get("artifact_report")
            _suffix = "  (after correction)" if (arep and arep["n_in"] != arep["n_out"]) else ""
            if windowed:
                _suffix += f"  [{n_used}/{n_total} in window]"
            self.app.lbl_npeaks.configure(  # type: ignore[union-attr]
                text=f"Peaks detected: {n_total}{_suffix}", text_color=_c)
        arep = self.app.analysis.artifact_report
        if arep and self.app._snapshot_params().get("artifact_correction"):
            removed = arep["n_in"] - arep["n_out"]
            art_str = (f"  |  −{removed} artifacts" if removed > 0
                       else "  |  no artifacts")
        else:
            art_str = ""
        hr_res  = self.app.analysis.results["hr"]
        n_valid = hr_res.get("n_valid", n_peaks)
        win_str = f" (window: {n_used} peaks)" if windowed else ""
        # RR intervals outside the bounds used, or spanning a beat removed by
        # artifact correction, are excluded from HRV (not bridged) by
        # analyse_core -- disclose it, in orange, with the bounds actually used.
        n_excl  = int(hr_res.get("n_excluded", 0) or 0)
        excl_str = f"  |  {rr_exclusion_text(hr_res)}" if n_excl else ""
        status_color = ORANGE if n_excl else GREEN
        self.app._set_status(
            f"Core analysis done — {n_used} peaks analysed{win_str} / {n_valid} valid"
            f"{art_str}{excl_str}  |  rendering…", status_color)
        self.app._update_kpis()
        self.app._draw_detail()
        # Enable the per-tab buttons now that core results are available
        self.app._set_result_btns_enabled(
            True, ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl", "btn_run_arrhythmia"))
        # Auto epochs start once frequency HRV -- itself an async job -- has
        # finished: scheduled independently (after(200) at the end of the
        # plot chain) they always hit the busy guard while run_freq was
        # still running, and never ran.
        auto_epochs = bool(bundle.get("auto_epochs", False))

        def _core_results_done():
            # Frequency-domain HRV runs automatically as part of Analyze
            # rather than needing its own click -- a Welch PSD over a few
            # hundred RR intervals is cheap, and it used to be the single most common
            # "click Analyze, then immediately click Freq HRV too" sequence
            # in the app.
            #
            # Deliberately sequenced AFTER _draw_core_results's own chain
            # finishes, not launched concurrently with it: both this and
            # run_freq() drive the same shared progress row/status label
            # through independent async completion callbacks with no
            # coordination between them -- running them at the same time let
            # whichever finished last silently overwrite the other's final
            # status text and could hide the progress row mid-render.
            self.app._set_status(
                f"Core analysis done — {n_used} peaks / {n_valid} valid{art_str}{win_str}"
                f"{excl_str}  | Running Frequency HRV…", status_color)
            # Non-linear after it, epochs last: each starts once the one before
            # has finished, or the busy guard would refuse it.
            self._when_idle(lambda: self.run_freq(then=lambda: self._when_idle(
                lambda: self.run_nonlinear(
                    then=(lambda: self._when_idle(self.app._compute_epochs))
                    if auto_epochs else None))))

        try:
            self.app._draw_core_results(on_complete=_core_results_done)
        except Exception:
            log.exception("_draw_core_results failed")
        # Populate interpretation tab with core values (freq/nonlinear added later)
        pass  # interpretation removed
        # Analysis window label (the window actually used) and the
        # Frequency / Non-linear / Intervals status labels. The window label also
        # carries the RR-exclusion note, in orange: unlike the status bar it
        # stays until the window or the results change (and is re-derived
        # after a UI rebuild or a session restore), so the disclosure
        # persists however many status messages follow.
        self.refresh_result_labels()
        # Enable Save Session now that we have results
        if self.app.btn_save_session is not None:
            self.app._set_save_session_enabled(True)
        self.app.session.dirty = True
        # Update session/template info labels
        self.app._update_session_ui(
            has_session=bool(self.app.signal.filepath and session_exists(self.app.signal.filepath)))
        # Auto-expand EXPORTS the first time analysis produces something to
        # export -- it's collapsed by default and otherwise has no cue
        # pointing to it, so exporting was effectively undiscoverable.
        if not getattr(self.app, "_exports_auto_opened", False):
            self.app._exports_auto_opened = True
            sec_exports = getattr(self.app, "sec_exports", None)
            if sec_exports is not None:
                sec_exports.open()

    def run_freq(self, then: "Any" = None) -> None:
        """Compute frequency-domain HRV in background, then render.

        *then* (optional, no arguments) runs once the rendering chain has
        finished -- Analyze chains its auto epochs this way, since starting
        them while this async job is still running is refused by the busy
        guard."""
        if self.app.analysis.results is None or self.app.detection.rpeaks_ok is None:
            messagebox.showwarning("Not ready", "Click Analyze first.")
            return
        rp = self.app._windowed_peaks()
        if rp is None or len(rp) < 5:
            messagebox.showwarning("Not ready", "Not enough peaks in the analysis window.")
            return
        if self._refused_busy():
            return
        fs  = self.app.signal.fs
        _gen = getattr(self.app, "_generation", 0)  # snapshot — detect file change
        # Same bounds/breaks as the core results this extends (main thread).
        rr_lo, rr_hi = self.results_rr_bounds()
        breaks       = self.hrv_breaks()
        bands        = self.get_freq_bands()
        basis        = self._capture_basis(rr_bounds=(rr_lo, rr_hi))
        self._mark_running("frequency")
        if self.app.lbl_freq_status is not None:
            self.app.lbl_freq_status.configure(text="  Computing…", text_color=ORANGE)  # type: ignore[union-attr]

        def _worker():
            def _prog(p, m):
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))
            return analyse_hrv_freq(rp, fs, progress_cb=_prog,
                                    rr_min_ms=rr_lo, rr_max_ms=rr_hi, breaks=breaks,
                                    bands=bands)

        def _done(result):
            self._mark_landed("frequency")
            if self._job_is_stale("Frequency HRV", "lbl_freq_status", _gen, basis):
                return
            self.app.analysis.results["hrv_freq"] = result
            self._stamp_basis("frequency", basis)
            results: dict = self.app.analysis.results  # narrow for type checkers
            tasks = [
                ("PSD",       lambda: self.app._plot_psd(results)),
                ("HRV tables (freq)", lambda: self.app._plot_hrv_tables(results)),
                ("Metrics",   lambda: self.app._plot_table_views(results)),
            ]
            # Spectrum withheld (too few valid intervals, or excluded
            # intervals too long to interpolate across): the label says
            # why instead of a row of "—" (same text after a UI rebuild or
            # a session restore, see refresh_result_labels).
            withheld = hrv_freq_withheld_reason(results)
            if self.app.lbl_freq_status is not None:
                text, color = self._freq_status(result, just_computed=True)
                self.app.lbl_freq_status.configure(  # type: ignore[union-attr]
                    text=text, text_color=color)
            # Final status keeps the core results' RR-exclusion note (and
            # its orange): this runs right after Analyze and used to replace
            # that disclosure with a plain green "done".
            excl_note = self.exclusion_status_note()
            if withheld:
                final = (f"Frequency HRV withheld — {withheld}{excl_note}", ORANGE)
            else:
                final = (f"Frequency HRV done{excl_note}", ORANGE if excl_note else GREEN)

            def _finish() -> None:
                self.app._set_status(*final)
                if then is not None:
                    then()

            self.app._run_plot_chain(tasks, on_complete=_finish)

        self.app._start_async_result(self.app.btn_run_freq, "Computing…", _worker, _done)  # type: ignore[arg-type]

    def entropy_cap(self) -> int:
        """Cap on valid RR intervals for the entropy/complexity block of
        non-linear HRV, from the Rhythm & HRV page's entry (only an explicit 0
        = no cap); the MouseECG.ENTROPY_MAX_BEATS default when the entry is
        missing, blank, invalid or negative -- a negative entry is a typo,
        not a request for the O(N²) entropy block on every beat. Poincaré
        and DFA always use every beat of the window."""
        ent = getattr(self.app, "ent_entropy_cap", None)
        try:
            txt = ent.get().strip() if ent is not None else ""
            val = int(float(txt)) if txt else MouseECG.ENTROPY_MAX_BEATS
        except (ValueError, TypeError, AttributeError, OverflowError):
            return MouseECG.ENTROPY_MAX_BEATS
        if val < 0:
            log.warning("Entropy cap %r is negative — using the default %d RR",
                        txt, MouseECG.ENTROPY_MAX_BEATS)
            try:   # show the value actually used in the entry
                ent.delete(0, "end")  # type: ignore[union-attr]
                ent.insert(0, str(MouseECG.ENTROPY_MAX_BEATS))  # type: ignore[union-attr]
            except Exception as _exc:
                log.debug("entropy cap entry reset failed: %s", _exc)
            return MouseECG.ENTROPY_MAX_BEATS
        return val

    def run_nonlinear(self, then: "Any" = None) -> None:
        """Compute non-linear HRV in background, then render.

        *then* (optional, no arguments) runs once the rendering chain has
        finished, as for run_freq: the last steps of an Analyze."""
        if self.app.analysis.results is None or self.app.detection.rpeaks_ok is None:
            messagebox.showwarning("Not ready", "Click Analyze first.")
            return
        rp = self.app._windowed_peaks()
        if rp is None or len(rp) < 5:
            messagebox.showwarning("Not ready", "Not enough peaks in the analysis window.")
            return
        sig = self.app.signal.filtered
        if sig is None:
            messagebox.showwarning("Not ready", "Signal not loaded.")
            return
        if self._refused_busy():
            return
        fs  = self.app.signal.fs
        _gen = getattr(self.app, "_generation", 0)
        # Same bounds/breaks as the core results this extends (main thread).
        rr_lo, rr_hi = self.results_rr_bounds()
        breaks       = self.hrv_breaks()
        entropy_cap  = self.entropy_cap()
        basis        = self._capture_basis(rr_bounds=(rr_lo, rr_hi))
        self._mark_running("nonlinear")
        if self.app.lbl_nonlin_status is not None:
            self.app.lbl_nonlin_status.configure(  # type: ignore[union-attr]
                text="  Computing SampEn / DFA… (may take 30 s+)", text_color=ORANGE)

        def _worker():
            def _prog(p, m):
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))
            return analyse_hrv_nonlinear(rp, fs, progress_cb=_prog,
                                         max_beats=entropy_cap,
                                         rr_min_ms=rr_lo, rr_max_ms=rr_hi,
                                         breaks=breaks)

        def _done(result):
            self._mark_landed("nonlinear")
            if self._job_is_stale("Non-linear HRV", "lbl_nonlin_status", _gen, basis):
                return
            self.app.analysis.results["hrv_nonlin"] = result
            self._stamp_basis("nonlinear", basis)
            results: dict = self.app.analysis.results  # narrow for type checkers
            tasks = [
                ("Non-linear metrics", lambda: self.app._plot_nonlinear(results)),
                ("Metrics",            lambda: self.app._plot_table_views(results)),
            ]
            # SampEn, beat-count truncation and withheld asymmetry indices
            # (same text after a UI rebuild or a session restore, see
            # refresh_result_labels).
            if self.app.lbl_nonlin_status is not None:
                text, color = self._nonlin_status(result, just_computed=True)
                self.app.lbl_nonlin_status.configure(  # type: ignore[union-attr]
                    text=text, text_color=color)
            excl_note = self.exclusion_status_note()   # same as run_freq

            def _finish() -> None:
                self.app._set_status(f"Non-linear HRV done{excl_note}",
                                     ORANGE if excl_note else GREEN)
                if then is not None:
                    then()

            self.app._run_plot_chain(tasks, on_complete=_finish)

        self.app._start_async_result(self.app.btn_run_nonlin, "Computing…", _worker, _done)  # type: ignore[arg-type]

    def recording_wave_template(self) -> WaveTemplate:
        """The current recording's wave template, created on first use from
        the saved default template (``WaveTemplate.load_default``, read
        only). It stays with the recording -- kept in its session, changed
        only from the Landmarks editor -- so the intervals of one recording
        never depend on the recordings analysed before it.

        A saved default that is not used (older-version file, invalid or
        implausible windows -- see ``WaveTemplate.load_default_checked``)
        is reported once per app session, naming the file and its windows."""
        wt = self.app.analysis.wave_template
        if wt is None:
            wt, notice = WaveTemplate.load_default_checked()
            self.app.analysis.wave_template = wt
            if notice and not getattr(self.app, "_default_template_notice_shown", False):
                self.app._default_template_notice_shown = True
                messagebox.showwarning("Saved default template not used", notice)
        return wt

    @staticmethod
    def _measured_note(ivl: "pd.DataFrame", provenance: str) -> "tuple[str, str, int, int]":
        """The Morphology status line of measured intervals: (text, colour,
        beats measured, beats)."""
        interval_cols = [c for c in ["PR_ms", "QRS_ms", "QT_ms"] if c in ivl.columns]
        n_ok    = int((~ivl[interval_cols].isna().any(axis=1)).sum()) if interval_cols else 0  # type: ignore[arg-type]
        n_total = len(ivl)
        pct     = 100.0 * n_ok / n_total if n_total else 0.0
        # Graded by completion ratio, not "any beat succeeded" -- a
        # measurement drawn from a small, possibly-biased subset (wave
        # delineation tends to fail on exactly the abnormal beats) is a
        # very different result from a near-complete one, and both used
        # to render identically as plain green.
        if pct >= 90:
            note_color = GREEN
        elif pct >= 50:
            note_color = ORANGE
        else:
            note_color = RED
        tmpl_note = f"  ·  template: {provenance}" if provenance else ""
        note = f"  {n_ok}/{n_total} beats measured ({pct:.0f}%){tmpl_note}"
        if n_ok < n_total:
            remedy = ("check template / filters" if n_ok == 0
                      else "try Permissive Bounds or adjust Landmarks")
            note += f"  — {n_total - n_ok} outside bounds: {remedy}"
        return note, note_color, n_ok, n_total

    def _intervals_status(self, ivl: "pd.DataFrame | None") -> "tuple[str, str]":
        """(text, colour) of lbl_ivl_status for these results; the template
        named is the one the intervals used, recorded in their attrs."""
        # analyse_core's placeholder is a non-empty frame with RR_ms alone.
        if ivl is None or ivl.empty or "QT_ms" not in ivl.columns:
            return "  Core done — click to delineate P/Q/S/T waves", ORANGE
        used = ivl.attrs.get("wave_template")
        wt = (WaveTemplate.from_dict(used) if isinstance(used, dict)
              else self.app.analysis.wave_template)
        note, color, _, _ = self._measured_note(ivl, wt.provenance() if wt is not None else "")
        return note, color

    def run_intervals(self) -> None:
        """Compute interval delineation (PR/QRS/QT/QTc) for every beat."""
        if self.app.analysis.results is None or self.app.signal.filtered is None or self.app.detection.rpeaks_ok is None:
            messagebox.showwarning("Not ready", "Click Analyze first.")
            return
        rp = self.app._windowed_peaks()
        if rp is None or len(rp) < 5:
            messagebox.showwarning("Not ready", "Not enough peaks in the analysis window.")
            return
        fs  = self.app.signal.fs
        sig = self.app.signal.filtered
        # Recompute rr_ms from the windowed rp — do NOT use self.app.analysis.results["rr_ms"]
        # which came from a potentially different window in Core Analysis.
        rr  = np.diff(rp).astype(float) / fs * 1000
        _gen = getattr(self.app, "_generation", 0)

        if self.app.lbl_ivl_status is not None:
            self.app.lbl_ivl_status.configure(  # type: ignore[union-attr]
                text="  Delineating all beats…", text_color=ORANGE)

        # This recording's template (the saved default -- or the built-in
        # one -- until the user changes it in the Landmarks editor). The
        # worker gets a copy: delineation never modifies or saves it.
        wt_for_worker = self.recording_wave_template().copy()
        permissive_for_worker = (bool(self.app.sw_permissive.get())  # type: ignore[union-attr]
                                  if self.app.sw_permissive is not None else False)
        # Read on the main thread (Tk widget) before the worker starts.
        qtc_formula_for_worker = self.qtc_formula()
        # The beat window is sized from the shortest VALID interval, under
        # the bounds the core results used (a widened Max RR for an
        # anaesthetised mouse leaves no interval valid under the defaults,
        # and one spurious close peak would then shrink every beat window).
        rr_lo, rr_hi = self.results_rr_bounds()
        basis = self._capture_basis(rr_bounds=(rr_lo, rr_hi))

        def _worker() -> "pd.DataFrame":
            def _prog(p: int, m: str) -> None:
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))

            return analyse_intervals(sig, rp, fs, rr,
                                     progress_cb=_prog,
                                     wave_template=wt_for_worker,
                                     permissive_bounds=permissive_for_worker,
                                     qtc_formula=qtc_formula_for_worker,
                                     rr_min_ms=rr_lo, rr_max_ms=rr_hi)

        def _done(df: "pd.DataFrame") -> None:
            if self.app.analysis.results is None or getattr(self.app, "_generation", 0) != _gen:
                log.info("_run_intervals: stale result discarded (file changed)")
                self._disable_result_btns_without_results()
                return
            # The formula selector may have changed while the worker ran.
            if self.qtc_formula() != qtc_formula_for_worker:
                df = recompute_qtc_column(df, self.qtc_formula())
            self.app.analysis.results["intervals"] = df
            self._stamp_basis("intervals", basis)
            self.note_intervals_measured(df)

            lbl_tmpl = getattr(self.app, "lbl_template_info", None)
            if lbl_tmpl is not None:
                lbl_tmpl.configure(text=f"Template: {wt_for_worker.provenance()}",
                                   text_color=MUTED if wt_for_worker.source == "default"
                                   else GREEN)
            note, note_color, n_ok, n_total = self._measured_note(df, wt_for_worker.provenance())
            if self.app.lbl_ivl_status is not None:
                self.app.lbl_ivl_status.configure(text=note, text_color=note_color)  # type: ignore[union-attr]

            results: dict = self.app.analysis.results
            self.app._run_plot_chain(
                [("ECG intervals", lambda: self.app._plot_intervals(results)),
                 # The mean beat's overlay names the windows these intervals used.
                 ("Beat template", lambda: self.app._plot_beat_template(results)),
                 ("Metrics",       lambda: self.app._plot_table_views(results))],
                on_complete=lambda: self.app._set_status(
                    f"Interval delineation done — {n_ok}/{n_total} beats", note_color))

        self.app._start_async_result(self.app.btn_run_ivl, "Delineating…", _worker, _done)  # type: ignore[arg-type]

    def refresh_epoch_feasibility(self) -> None:
        """Proactively warn if the current Epoch (s) value can't fit the
        recording, instead of only finding out after clicking Compute epochs
        and getting the inline "too short" message. Called when the Epochs
        mode is shown (see app._show_time_mode) so the warning —
        and the orange entry border — are visible before the user presses
        anything. Mirrors compute_epochs()'s own feasibility check below.
        """
        if self.app.lbl_epoch_info is None or self.app.ent_epoch is None:
            return
        if self.app.detection.rpeaks_ok is None or len(self.app.detection.rpeaks_ok) < 10:
            return
        rp = self.app._windowed_peaks()
        if rp is None or len(rp) < 10 or self.app.signal.time is None:
            return
        fs        = self.app.signal.fs
        t_peaks   = rp / fs
        dur       = float(t_peaks[-1] - t_peaks[0])
        epoch_s   = max(MouseECG.EPOCH_MIN_S,
                        self.app._safe_float(self.app.ent_epoch, MouseECG.EPOCH_DEFAULT_S))
        overlap_s = max(0.0, self.app._safe_float(self.app.ent_overlap, 0.0))
        # Mirror compute_epochs()'s own bad-overlap guard: without this, a
        # step of max(1.0, epoch_s - overlap_s) floors the invalid negative
        # to 1.0 and reports "feasible" (clear border) for a combination
        # that compute_epochs() will immediately reject with a blocking
        # "Bad overlap" modal the moment Compute epochs is clicked.
        if overlap_s >= epoch_s:
            self.app.lbl_epoch_info.configure(
                text=f"Overlap ({overlap_s:.0f}s) must be less than epoch ({epoch_s:.0f}s).",
                text_color=ORANGE)
            self.app.ent_epoch.configure(border_color=ORANGE)
            return
        step      = max(1.0, epoch_s - overlap_s)
        n_starts  = len(np.arange(0.0, dur - epoch_s + step * 0.5, step))
        if n_starts < 2:
            suggested = max(MouseECG.EPOCH_MIN_S, dur // 3)
            self.app.lbl_epoch_info.configure(
                text=(f"Recording too short for {epoch_s:.0f}s epochs "
                      f"({dur:.0f}s available) — try ~{suggested:.0f}s."),
                text_color=ORANGE)
            self.app.ent_epoch.configure(border_color=ORANGE)
        else:
            self.app.ent_epoch.configure(border_color=BORDER2)
            # A warning of a former value goes: this now runs on every opening.
            self.app.lbl_epoch_info.configure(text=self._epoch_info_text(), text_color=MUTED)

    def _epoch_info_text(self) -> str:
        df = self.app.analysis.epoch_df
        return f"{len(df)} epoch(s) computed" if df is not None and not df.empty else ""

    # Epochs and Sliding share their controls, figure and confidence rule, but
    # NOT their numeric loop. Verdict of tests/ui/test_epoch_sliding_characterisation.py:
    # identical HR, SDNN, RMSSD, pNN6 and counts on the same grid with pauses,
    # removed beats and very short windows, except RMSSD on a window with a single
    # successive valid pair: |difference| here (hrv_time_domain), NaN in Sliding
    # (its guard wants two). Merging would change one of them, so they stay separate.
    def compute_epochs(self) -> None:
        """Compute epoch-level HRV in a background thread to keep the UI responsive.

        nk.hrv_time() is called once per epoch.  On long recordings with many
        short epochs this adds up to several seconds — enough to freeze the UI
        noticeably.  The calculation is therefore moved to a daemon thread via
        _start_async_result, exactly like _run_freq / _run_nonlinear.
        """
        if self.app.detection.rpeaks_ok is None or len(self.app.detection.rpeaks_ok) < 10:
            messagebox.showwarning("No data", "Click Detect Peaks first.")
            return
        if self.app.signal.time is None:
            return

        # What is ticked; the base columns are computed whatever is plotted, the
        # others only when ticked, so a default run gives the same frame as ever.
        ticks = getattr(self.app, "_epoch_metrics", None)
        active = ([m for m, cb in ticks.items() if cb.get()] if isinstance(ticks, dict) and ticks
                  else list(self.EPOCH_BASE_METRICS))
        if not active:
            self.app._set_status("Select at least one metric.", RED)
            return
        extra = set(active) - set(self.EPOCH_BASE_METRICS)
        need_sd = bool({"SD1", "SD2"} & extra)
        need_freq = bool({"LF_nu", "HF_nu", "LF_HF"} & extra)

        epoch_s   = max(MouseECG.EPOCH_MIN_S,
                        self.app._safe_float(self.app.ent_epoch, MouseECG.EPOCH_DEFAULT_S))
        overlap_s = max(0.0, self.app._safe_float(self.app.ent_overlap, 0.0))
        fs        = self.app.signal.fs
        rp        = self.app._windowed_peaks()   # respects analysis window
        if rp is None or len(rp) < 10:
            messagebox.showwarning("No data", "Not enough peaks in the analysis window.")
            return
        # Duration from windowed peaks rather than full signal
        t_peaks = rp / fs
        dur     = float(t_peaks[-1] - t_peaks[0])

        if overlap_s >= epoch_s:
            messagebox.showwarning("Bad overlap",
                f"Overlap ({overlap_s:.0f}s) must be less than epoch ({epoch_s:.0f}s).")
            return
        step      = max(1.0, epoch_s - overlap_s)
        t_win_start = float(t_peaks[0])    # absolute start of the windowed range
        starts    = np.arange(t_win_start, t_win_start + dur - epoch_s + step * 0.5, step)
        _gen = getattr(self.app, "_generation", 0)  # snapshot — detect file change
        # Read on the main thread (Tk widgets / app state) before the worker.
        rr_lo, rr_hi = self.hrv_rr_bounds()
        breaks       = self.hrv_breaks()
        bands        = self.get_freq_bands() if need_freq else None
        basis        = self._capture_basis()
        if len(starts) < 2:
            # Inline, non-blocking: a modal here used to re-fire on every
            # auto-run (see sw_epoch) without ever applying the suggested
            # value, forcing a dismiss-retype-retry loop.
            suggested = max(MouseECG.EPOCH_MIN_S, dur // 3)
            if self.app.lbl_epoch_info is not None:
                self.app.lbl_epoch_info.configure(
                    text=(f"Recording too short for {epoch_s:.0f}s epochs "
                          f"({dur:.0f}s available) — try ~{suggested:.0f}s."),
                    text_color=ORANGE)
            return

        def _worker():
            """Runs on background thread — no Tkinter access allowed."""
            rows = []
            n_skipped = 0
            n_ep = len(starts)
            for idx_ep, t0 in enumerate(starts):
                t1    = t0 + epoch_s
                ep_rp = rp[(rp / fs >= t0) & (rp / fs < t1)]
                if len(ep_rp) < self.MIN_WINDOW_BEATS:
                    n_skipped += 1
                    continue
                # Gap-aware, like analyse_core: out-of-range RR intervals and
                # intervals spanning a removed beat are excluded, and
                # successive differences never straddle them.
                ser = rr_interval_series(ep_rp, fs, rr_lo, rr_hi, breaks=breaks)
                rr  = ser["rri"]
                if len(rr) < self.MIN_VALID_INTERVALS:
                    n_skipped += 1
                    continue
                rr_a, rr_b = successive_valid_pairs(ser["rr_ms"], ser["valid"])
                sdiff = rr_b - rr_a
                try:
                    # Same gap-aware time-domain helper as analyse_core --
                    # nk.hrv_time alone can difference across an excluded
                    # interval in short epochs (see hrv_time_domain).
                    hrv_ep = hrv_time_domain(ser, fs)
                    sdnn   = float(hrv_ep["HRV_SDNN"].values[0])
                    rmssd  = float(hrv_ep["HRV_RMSSD"].values[0])
                except Exception as exc:
                    log.warning("Epoch hrv_time failed, manual calc: %s", exc)
                    # ddof=1 (sample std) to match nk.hrv_time's own convention
                    # and compute_rolling_hrv's manual computation below --
                    # this previously defaulted to ddof=0 (population std),
                    # silently disagreeing with both whenever this fallback
                    # actually triggered.
                    sdnn  = float(rr.std(ddof=1)) if len(rr) > 1 else float("nan")
                    rmssd = float(np.sqrt(np.mean(sdiff ** 2))) if len(sdiff) else float("nan")
                diffs = np.abs(sdiff)
                pnn6  = float(100.0 * np.sum(diffs > MouseECG.PNN_THRESHOLD) / len(diffs)) \
                        if len(diffs) else float("nan")
                row = {
                    "Epoch_start_s": round(t0, 1),
                    "Epoch_end_s":   round(t1, 1),
                    "N_beats":       len(ep_rp),
                    "N_valid":       len(rr),
                    "N_excluded":    ser["n_excluded"],
                    "HR_mean":       round(mean_hr_bpm(rr), 1),   # 60000 / mean NN
                    "MeanNN":        round(float(rr.mean()), 1),
                    "SDNN":          round(sdnn, 2),
                    "RMSSD":         round(rmssd, 2),
                    "pNN6":          round(pnn6, 1),
                }
                # The ticked extras, with the same helpers and rounding as Sliding.
                if need_sd:
                    pc = poincare_sd1_sd2(ser["rr_ms"], ser["valid"])
                    row.update(SD1=round(float(pc["sd1"]), 3) if np.isfinite(pc["sd1"]) else np.nan,
                               SD2=round(float(pc["sd2"]), 3) if np.isfinite(pc["sd2"]) else np.nan)
                if need_freq:
                    lf_nu, hf_nu, lf_hf = self._window_lf_hf(
                        ep_rp, fs, rr_lo, rr_hi, breaks, bands, t0)
                    row.update(LF_nu=round(lf_nu, 1) if np.isfinite(lf_nu) else np.nan,
                               HF_nu=round(hf_nu, 1) if np.isfinite(hf_nu) else np.nan,
                               LF_HF=round(lf_hf, 3) if np.isfinite(lf_hf) else np.nan)
                rows.append({k: v for k, v in row.items()
                             if k not in self.EPOCH_OPTIONAL or k in extra})
                # Progress via after() — safe cross-thread call, throttled to every 5%
                pct = int((idx_ep + 1) / max(n_ep, 1) * 100)
                if idx_ep % max(1, n_ep // 20) == 0:
                    self.app.after(0, lambda p=pct, e=idx_ep+1, tot=n_ep:
                               self.app._set_progress(p, f"Epoch {e}/{tot}…"))
            return {"rows": rows, "n_skipped": n_skipped}

        def _done(result):
            self._mark_landed("epochs")
            if getattr(self.app, "_generation", 0) != _gen:
                log.info("compute_epochs: stale result discarded (file changed)")
                return
            rows, n_skipped = result["rows"], result["n_skipped"]
            if not rows:
                messagebox.showwarning(
                    "No epochs",
                    f"No valid epochs found — each epoch needs at least "
                    f"{self.MIN_VALID_INTERVALS} valid RR intervals.")
                return
            df    = pd.DataFrame(rows)
            self.app.analysis.epoch_df = df
            self._stamp_basis("epochs", basis)
            self.app.analysis.windowed_view["epochs"] = {
                "epoch_s": epoch_s, "step_s": step, "active": active,
                "rr": [rr_lo, rr_hi], "n_skipped": n_skipped}
            n_excl_ep = self.show_epochs()
            n_ep = len(df)
            self.app.tabs.set(TAB_TIME_COURSE)
            self.app.after(50, lambda: self.app._show_time_mode(MODE_EPOCHS))
            # Can be the last step of an Analyze (auto epochs): keep the core
            # results' exclusion disclosure (and its orange) rather than
            # ending on plain green -- an excluded interval in the tail after
            # the last whole epoch, or straddling two epochs, is in no epoch
            # but still excluded from the core HRV.
            skip_note = f" · {n_skipped} skipped" if n_skipped else ""
            excl_note = (f"  ·  {n_excl_ep} epoch(s) with excluded RR"
                         if n_excl_ep else "")
            core_note = self.exclusion_status_note()
            self.app._set_status(
                f"Epoch analysis done — {n_ep} epochs{skip_note}{excl_note}{core_note}",
                ORANGE if (n_excl_ep or core_note) else GREEN)

        self._mark_running("epochs")
        self.app._start_async_result(self.app.btn_compute_epochs, "Computing…", _worker, _done)

    def show_epochs(self) -> int:
        """Draw the stored Epochs frame -- figure, table (with its Excel TSV) and
        labels -- from what it was computed with, or from the frame itself when
        it was saved without that record. Returns the number of epochs with
        excluded RR; without a frame the view is emptied and it returns 0. Main
        thread only."""
        df = self.app.analysis.epoch_df
        if df is None or df.empty:
            self.clear_epochs_view()
            return 0
        view = self.app.analysis.windowed_view.get("epochs") or {}
        start, end = df["Epoch_start_s"].to_numpy(dtype=float), df["Epoch_end_s"].to_numpy(dtype=float)
        t_mid = (start + end) / 2
        epoch_s = float(view.get("epoch_s") or np.median(end - start))
        step = float(view.get("step_s") or (np.min(np.diff(start)) if len(start) > 1 else epoch_s))
        rr_lo, rr_hi = view.get("rr") or self.hrv_rr_bounds(quiet=True)
        n_skipped = int(view.get("n_skipped", 0))
        # Confidence from the intervals the metrics were computed from
        low_n, old_note = self._frame_confidence(df, "N_valid", "N_beats")

        specs = {
            "HR":    ("HR_mean", "HR (bpm)",     ORANGE_DARK, "Heart Rate per Epoch"),
            "SDNN":  ("SDNN",    "SDNN (ms)",    BLUE_DARK,   "SDNN per Epoch"),
            "RMSSD": ("RMSSD",   "RMSSD (ms)",   GREEN_DARK,  "RMSSD per Epoch"),
            "pNN6":  ("pNN6",    "pNN6 (%)",     PURPLE,      "pNN6 per Epoch"),
            "SD1":   ("SD1",     "SD1 (ms)",     CORAL,       "SD1 per Epoch"),
            "SD2":   ("SD2",     "SD2 (ms)",     CYAN_BRIGHT, "SD2 per Epoch"),
            "LF_nu": ("LF_nu",   "LF (n.u., %)", BLUE_MID,    "LF (n.u.) per Epoch"),
            "HF_nu": ("HF_nu",   "HF (n.u., %)", GREEN,       "HF (n.u.) per Epoch"),
            "LF_HF": ("LF_HF",   "LF/HF",        AMBER,       "LF/HF per Epoch"),
        }
        panels = [specs[m] for m in (view.get("active") or self.EPOCH_BASE_METRICS)
                  if m in specs and specs[m][0] in df.columns]
        self.app._slots["epochs"].update(lambda fig: draw_windowed_panels(
            fig, t_mid, df, panels, low_n, step * 1.5, list(self.app.analysis.annotations)))
        table_text = df.to_string(index=False) + old_note
        if low_n.any() and not old_note:
            table_text += (
                f"\n\n  * {int(low_n.sum())} epoch(s) have fewer than "
                f"{self.MIN_CONFIDENT_BEATS} valid RR intervals (N_valid) — "
                f"SDNN/RMSSD/pNN6 for those rows are noisy (hollow markers "
                f"on the plot above).")
        n_excl_ep = int((df["N_excluded"].values > 0).sum()) if "N_excluded" in df.columns else 0
        if n_excl_ep:
            bounds = rr_bounds_label({"rr_min_ms": rr_lo, "rr_max_ms": rr_hi})
            table_text += (
                f"\n\n  * {n_excl_ep} epoch(s) contain RR intervals outside "
                f"{bounds} or spanning removed beats (N_excluded) — "
                f"excluded from every metric, not bridged.")
        if n_skipped:
            table_text += (
                f"\n\n  * {n_skipped} epoch(s) skipped — fewer than "
                f"{self.MIN_VALID_INTERVALS} valid RR intervals (or "
                f"{self.MIN_WINDOW_BEATS} beats); no row, and a gap in "
                f"the plotted lines.")
        self.app._set_textbox(self.app.txt_epochs, table_text,
                              tsv=self.app._df_to_tsv(df))
        n_ep = len(df)
        # Update the label of the Epochs mode
        self.app.lbl_epoch_count.configure(
            text=f"{n_ep} epochs × {epoch_s:.0f}s", text_color=BLUE)
        self.app.lbl_epoch_info.configure(text=self._epoch_info_text(), text_color=MUTED)
        return n_excl_ep

    def clear_epochs_view(self) -> None:
        """No Epochs frame: nothing of an earlier one stays on screen, as after a
        UI rebuild. Main thread only."""
        self._blank_windowed_view("txt_epochs", "epochs")
        for attr in ("lbl_epoch_count", "lbl_epoch_info"):
            lbl = getattr(self.app, attr, None)
            if lbl is not None:
                lbl.configure(text="", text_color=MUTED)
