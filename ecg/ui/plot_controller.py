# -*- coding: utf-8 -*-
"""
plot_controller.py
-------------------
PlotController -- rendering analysis results into the matplotlib canvases
(CanvasSlot instances) embedded in the Tk UI: RR/HR tachogram, HRV tables,
PSD, radar, Poincare/non-linear, ECG interval annotation, beat template,
summary tab, arrhythmia detail view, and the main detail/overview signal
plots. Pure rendering -- it reads SignalState/DetectionState/AnalysisState/
UIState and writes into ECGApp's CanvasSlot registry (self.app._slots),
but never mutates analysis results themselves.
"""
from __future__ import annotations

import logging
import os
import textwrap
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib
import matplotlib.ticker
import matplotlib.patches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from ecg.core.models import MouseECG, QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.core.filtering import downsample_xy_for_display, recorded_filter_text
from ecg.core.wave_template import WaveTemplate
from ecg.core.analysis import (
    INTERVAL_DEFINITIONS_TEXT,
    intervals_use_legacy_definitions, qt_beat_variability, qtc_rr_column,
    compute_beat_correlation, hrv_freq_is_legacy, hrv_freq_withheld_reason,
    rr_bounds_label, LEGACY_SPECTRAL_TEXT,
    rr_exclusion_text,
    rr_interval_series, rr_timeseries,
    spectral_withheld_reason, successive_valid_pairs, rr_asymmetry_indices,
    beat_quality_times,
    MIN_BEATS_SPECTRAL, band_mask, hrv_spectrum, mean_hr_bpm,
    nonlinear_subset_text, spectral_band_metrics,
    analysis_provenance_rows,
)
from ecg.core.metric_table import (
    MetricTable, build_metric_table, interval_rows, quality_tile_texts, results_rr_valid,
)
from ecg.ui.plots import style_axes, wrap_fig_text
from ecg.ui.wave_editor import WaveTemplateEditor
from ecg.ui.theme import (
    THEME, PLOT, RED, AMBER, ORANGE, ORANGE_DARK,
    CYAN, BLUE, BLUE_DARK, BLUE_MID, PURPLE, TEAL, TEAL_DARK,
    BORDER2, RED_MID, MUTED, GRAY, GRAY_LIGHT, NAVY,
    GREEN, GREEN_DARK, PINK, AMBER_DARK, ARTIFACT_TYPE_COLOR,
)
from ecg.ui.widgets import update_quality_gauge
from ecg.ui.tab_keys import TAB_DETECTION

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


def beat_template_overlay(current: "Optional[WaveTemplate]",
                          intervals: "Optional[pd.DataFrame]",
                          ) -> "tuple[Optional[WaveTemplate], str]":
    """(template whose windows the Beat Template plot overlays, title note).

    The displayed intervals record the template that produced them
    (``attrs["wave_template"]``). When the recording's current template
    differs (applied in Landmarks without re-running Intervals, or an old
    session), the windows actually used are drawn and the note says so --
    the overlay must match the numbers on screen."""
    used_raw = intervals.attrs.get("wave_template") if intervals is not None else None
    used = WaveTemplate.from_dict(used_raw) if isinstance(used_raw, dict) else None
    if used is not None and (current is None or current.to_dict() != used.to_dict()):
        note = "  ·  windows used by the displayed intervals"
        if current is not None:
            note += " (current template differs: re-run Intervals)"
        return used, note
    if current is None:
        return None, ""
    return current, f"  ·  template: {current.provenance()}"



def summary_interval_metrics(ivl: "Optional[pd.DataFrame]") -> "dict[str, str]":
    """Summary metric-card values (median, ms) for the delineated intervals,
    keyed like the card rows: PR, QRS, uncorrected QT and QTc (QT is always
    shown next to its rate correction). Missing / empty columns are left out."""
    out: "dict[str, str]" = {}
    t = MetricTable(interval_rows(ivl))
    for col, key in [("PR_ms", "pr"), ("QRS_ms", "qrs"),
                     ("QT_ms", "qt"), ("QTc_ms", "qtc")]:
        row = t.row(f"intervals.{col}")
        if row is not None and row.basis_n:
            out[key] = f"{row.value:.0f}"
    return out


def _df_val(df: "Optional[pd.DataFrame]", col: str, decimals: int = 3) -> str:
    """Formatted first value of *df[col]*, '—' when missing / not finite."""
    try:
        v = float(df[col].values[0])   # type: ignore[index]
        return f"{v:.{decimals}f}" if np.isfinite(v) else "—"
    except Exception:
        return "—"


def summary_report_text(
    r:               dict,
    subject:         str = "",
    filepath:        "Optional[str]" = None,
    filter_text:     "Optional[str]" = None,
    window:          "Optional[tuple]" = None,
    signal_end_s:    "Optional[float]" = None,
    artifact_report: "Optional[dict]" = None,
    qtc_formula:     "Optional[str]" = "mitchell",
    asymmetry:       "Optional[dict]" = None,
    when:            "Optional[datetime]" = None,
    signal_crop:     "Optional[dict]" = None,
) -> str:
    """The Summary tab's text report for results *r* (pure: testable
    without the app). Every value carries its unit; VLF / LF / HF are
    absolute powers (ms²), n.u. only for LFn / HFn. The PROVENANCE block
    (analysis_provenance_rows -- same rows as the Excel Cover sheet and the
    Prism notes) states what is needed to reproduce the numbers: analysis
    window and analysed duration (window end − start, not the absolute
    time of the last beat), filters, RR bounds and exclusions, artifact
    correction, QTc formula, interval definitions and wave template, plus
    the load-time *signal_crop* (analysis.signal_crop_info) when recorded.
    A missing value is a bare '—' (no unit after it).
    *asymmetry* = rr_asymmetry_indices() of the successive valid RR pairs
    ({"porta_pct", "guzik_pct", "n_dec", "n_acc", "n_tot", "n_zero", ...};
    plot_summary's RR-asymmetry block) or None."""
    t   = build_metric_table(r, asymmetry=asymmetry)
    nl  = r.get("hrv_nonlin")
    ivl = r.get("intervals")
    W   = 62

    def row(label: str, value: str) -> str:
        return f"    {label:<22} {value}"

    def nu(key: str) -> str:
        v = t.finite(f"hrv_freq.{key}")
        return f"{v * 100.0:.1f}" if v is not None and np.isfinite(v * 100.0) else "—"

    def time_val(key: str) -> str:
        return t.text(f"hrv_time.{key}", 3)

    def freq_val(key: str) -> str:
        return t.text(f"hrv_freq.{key}", 3)

    def nl_val(key: str) -> str:
        return t.text(f"hrv_nonlin.{key}", 3)

    def wrapped(label: str, value: str) -> "list[str]":
        # Long provenance values (definitions, template) wrapped under
        # their label rather than running off the text box.
        body = textwrap.wrap(str(value), 56) or [""]
        return [row(label, body[0])] + [f"{'':<27}{ln}" for ln in body[1:]]

    def u(v: str, unit: str, note: str = "") -> str:
        # A missing value is '—', not '— ms' (nor '— %  (note)').
        if v == "—":
            return v
        return f"{v} {unit}{note}" if unit else f"{v}{note}"

    when = when or datetime.now()
    lines = [
        "═" * W,
        "  ECG ANALYSIS  —  Summary Report",
        f"  Subject  :  {subject}",
        f"  Date     :  {when:%Y-%m-%d  %H:%M}",
        f"  File     :  {os.path.basename(filepath or '') or '—'}",
        "═" * W, "",
        "  PROVENANCE",
    ]
    for label, value in analysis_provenance_rows(
            r, window=window, signal_end_s=signal_end_s,
            artifact_report=artifact_report,
            filter_text=filter_text or "not recorded",
            qtc_formula=qtc_formula, signal_crop=signal_crop):
        lines += wrapped(label, value)

    lines += [
        "", "  HEART RATE",
        row("Mean (60000/mean NN)", u(t.text("hr.mean", 1), "bpm")),
        # instantaneous_hr_stats: 2nd / 98th percentile, sample SD
        row("Min (2nd %ile)", u(t.text("hr.min", 1), "bpm")),
        row("Max (98th %ile)", u(t.text("hr.max", 1), "bpm")),
        row("SD (instantaneous)", u(t.text("hr.std", 1), "bpm")),
        row("Beats (n)", str(t.value("hr.n_beats", "—"))),
        "",
        "  HRV — TIME DOMAIN",
        row("MeanNN", u(time_val('HRV_MeanNN'), "ms")),
        row("SDNN", u(time_val('HRV_SDNN'), "ms")),
        row("RMSSD", u(time_val('HRV_RMSSD'), "ms")),
        row("pNN6", u(time_val('HRV_pNN6'), "%",
                      f"  (ΔRR > {MouseECG.PNN_THRESHOLD} ms)")),
        row("pNN20", u(time_val('HRV_pNN20'), "%", "  (human threshold)")),
        "",
        "  HRV — FREQUENCY DOMAIN",
    ]
    freq_withheld = hrv_freq_withheld_reason(r)
    if hrv_freq_is_legacy(r):
        # An older session's nk.hrv_frequency table: arbitrary units,
        # LFn = LF / TP -- not ms² / Task Force n.u.
        lines += [
            row("VLF power", u(freq_val('HRV_VLF'), "a.u.")),
            row("LF power", u(freq_val('HRV_LF'), "a.u.",
                              f"  ({u(nu('HRV_LFn'), '%')} of TP)")),
            row("HF power", u(freq_val('HRV_HF'), "a.u.",
                              f"  ({u(nu('HRV_HFn'), '%')} of TP)")),
            row("LF/HF", u(freq_val('HRV_LFHF'), "", "  (ratio)")),
            *wrapped("Note", LEGACY_SPECTRAL_TEXT),
        ]
    elif not freq_withheld:
        # Absolute band powers (ms²) of the spectrum drawn on the PSD
        # panel; n.u. = LF or HF / (LF + HF) only.
        lines += [
            row("VLF power", u(freq_val('HRV_VLF'), "ms²")),
            row("LF power", u(freq_val('HRV_LF'), "ms²")),
            row("HF power", u(freq_val('HRV_HF'), "ms²")),
            row("LF (n.u.)", u(nu('HRV_LFn'), "%", "  (LF / (LF + HF))")),
            row("HF (n.u.)", u(nu('HRV_HFn'), "%", "  (HF / (LF + HF))")),
            row("LF/HF", u(freq_val('HRV_LFHF'), "", "  (ratio)")),
        ]
    else:
        # A withheld spectrum says why rather than a column of "—".
        lines += wrapped("Withheld", freq_withheld)

    nl_subset = nonlinear_subset_text(getattr(nl, "attrs", {}) if nl is not None else {})
    lines += [
        "",
        "  HRV — NON-LINEAR",
        row("SD1", u(nl_val('HRV_SD1'), "ms")),
        row("SD2", u(nl_val('HRV_SD2'), "ms")),
        row("SampEn", u(nl_val('HRV_SampEn'), "", "  (dimensionless)")),
        row("ApEn", u(nl_val('HRV_ApEn'), "", "  (dimensionless)")),
        row("DFA α1", u(nl_val('HRV_DFA_alpha1'), "", "  (dimensionless)")),
        row("DFA α2", u(nl_val('HRV_DFA_alpha2'), "", "  (dimensionless)")),
    ]
    if nl_subset:
        lines += wrapped("Note", nl_subset)

    if ivl is not None and not ivl.empty and "QT_ms" in ivl.columns:
        # Definitions, QT variability and template are in PROVENANCE.
        lines += ["", "  ECG INTERVALS  (median ± SD across beats)"]
        qtc_short = QTC_FORMULA_LABELS[normalize_qtc_formula(qtc_formula or "mitchell")]
        for col, label in [("PR_ms", "PR"), ("QRS_ms", "QRS"), ("QT_ms", "QT"),
                           ("QTc_ms", f"QTc ({qtc_short.split(' ')[0]})")]:
            med = t.row(f"intervals.{col}")
            if med is not None and med.basis_n:
                lines.append(row(label, f"{med.value:.1f} ± "
                                        f"{t.value(f'intervals.{col}.sd'):.1f} ms"))
        if intervals_use_legacy_definitions(ivl):
            # The values above are not the standard PR / QRS / QT: say so
            # here, not only in PROVENANCE.
            lines += wrapped("Note", "legacy definitions (older version: PR to "
                                     "Q nadir, QRS Q→S nadir, QT from R) — re-run "
                                     "Intervals for the standard ones")

    if t.has("asymmetry.n_tot"):
        n_tot = t.value("asymmetry.n_tot")
        lines += [
            "", "  RR ASYMMETRY  (successive valid ΔRR)",
            # Literature conventions -- rr_asymmetry_indices.
            *wrapped("Porta index (PI)", u(t.text("asymmetry.porta_pct", 1), "%",
                                           "  (Porta 2008: points below the "
                                           "identity line, ΔRR < 0, / off-line "
                                           "points; ΔRR = 0 excluded; "
                                           "symmetry → 50 %)")),
            *wrapped("Guzik index (GI)", u(t.text("asymmetry.guzik_pct", 1), "%",
                                           "  (Guzik 2006: Σ distance of points "
                                           "above the identity line, ΔRR > 0, / "
                                           "Σ distance of off-line points; "
                                           "symmetry → 50 %)")),
            row("Decelerations", f"{t.value('asymmetry.n_dec')} / {n_tot}  (ΔRR > 0)"),
            row("Accelerations", f"{t.value('asymmetry.n_acc')} / {n_tot}  (ΔRR < 0)"),
        ]
        if t.has("asymmetry.n_zero"):
            lines.append(row("No change", f"{t.value('asymmetry.n_zero')} / "
                                          f"{n_tot}  (ΔRR = 0, not in PI / GI)"))
    lines += ["", "═" * W]
    return "\n".join(lines)


class PlotController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app
        # draw_overview()'s artist-mutation cache -- see that method.
        self._ov_cache: "Optional[dict]" = None

    # ── Gap-aware RR helpers ──────────────────────────────────────────
    # analyse_core keeps every RR interval in r["rr_ms"] and marks the ones
    # used for HRV in r["rr_valid"] (inside the RR bounds, not spanning a
    # beat removed by artifact correction); HRV is computed on the valid
    # ones only, never bridging an excluded interval. Results restored from
    # a session saved before rr_valid existed (and the pre-Analyze live
    # tachogram) fall back to recomputing the mask from the bounds.
    @staticmethod
    def _rr_valid(r: dict) -> np.ndarray:
        return results_rr_valid(r)

    def _report_window(self) -> "tuple":
        """(t_start, t_end) the current results were computed on (0 = signal
        start / end) -- the window entries can be edited afterwards."""
        a = self.app.analysis
        win = getattr(a, "analysed_window", None)
        return tuple(win) if win is not None else (a.t_start, a.t_end)

    def _signal_end_s(self) -> "Optional[float]":
        """Time of the last sample of the loaded signal (s), None without one."""
        t = self.app.signal.time
        return float(t[-1]) if t is not None and len(t) else None

    @classmethod
    def _rr_exclusion(cls, r: dict) -> "tuple[int, float]":
        """(n excluded RR intervals, % of all intervals) for result dict *r*."""
        hr = r.get("hr") or {}
        if "n_excluded" in hr:
            return int(hr["n_excluded"]), float(hr.get("excluded_pct", 0.0))
        valid = cls._rr_valid(r)
        n_ex  = int((~valid).sum())
        return n_ex, (100.0 * n_ex / len(valid) if len(valid) else 0.0)

    def draw_arr_detail(self) -> None:
        """Draw ECG strip for the selected arrhythmia event, with editable R peaks."""
        if self.app.signal.filtered is None or self.app.signal.time is None:
            return

        sig_flt  = self.app.signal.filtered
        time     = self.app.signal.time
        fs       = self.app.signal.fs

        try:
            win = float(self.app.ent_arr_win.get())
        except Exception:
            win = self.app.analysis.arr_win
        win = max(0.5, win)
        self.app.analysis.arr_win = win

        t_start = self.app.analysis.arr_nav_pos
        t_end   = min(float(time[-1]), t_start + win)
        # Index-slice instead of a full-length boolean mask: `time` is always
        # uniformly spaced (np.arange(n)/fs), so the window bounds map
        # directly to a sample-index range. A full `(time >= t0) & (time <= t1)`
        # mask costs O(n_samples_total) on every redraw — this view redraws on
        # every navigation step/click, which becomes noticeably slow on long
        # or high-fs recordings. Slicing is O(window size) instead.
        i0 = max(0, int(t_start * fs))
        i1 = min(len(time), int(t_end * fs) + 1)
        mask_t  = slice(i0, i1)

        # Peak arrays
        rp_ok    = self.app.detection.rpeaks_ok    if self.app.detection.rpeaks_ok    is not None else np.array([])
        rp_excl  = self.app.detection.rpeaks_manual_excl  if self.app.detection.rpeaks_manual_excl  is not None else np.array([])
        rp_added = self.app.detection.rpeaks_manual_added if self.app.detection.rpeaks_manual_added is not None else np.array([])

        def _in_win(idx: np.ndarray) -> np.ndarray:
            return (idx / fs >= t_start) & (idx / fs <= t_end)

        mask_ok    = _in_win(rp_ok)    if len(rp_ok)    else np.array([], bool)
        mask_excl  = _in_win(rp_excl)  if len(rp_excl)  else np.array([], bool)
        mask_added = _in_win(rp_added) if len(rp_added) else np.array([], bool)

        # Selected event span
        ev = (self.app.analysis.arrhythmia_events[self.app.analysis.arr_selected_idx]
              if 0 <= self.app.analysis.arr_selected_idx < len(self.app.analysis.arrhythmia_events)
              else None)
        ev_t_start = ev.t_start if ev else None
        ev_t_end   = ev.t_end   if ev else None
        sev_color  = {"alert": RED_MID, "warning": AMBER, "info": BLUE_MID
                      }.get(ev.severity, MUTED) if ev else MUTED
        ev_label   = ev.label if ev else ""
        edit_mode  = self.app.analysis.arr_edit_mode

        n_in_win   = int(mask_ok.sum())
        t_amp      = self.app.detection.thresh_amp

        def draw(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)

            # ECG trace
            ax.plot(time[mask_t], sig_flt[mask_t],
                    color=PLOT.get("signal", CYAN), lw=0.9, zorder=2,
                    label="Filtered ECG")

            # Event span highlight
            if ev_t_start is not None and ev_t_end is not None:
                _span_lo = max(ev_t_start, t_start)
                _span_hi = min(max(ev_t_end, ev_t_start + 0.05), t_end)
                if _span_lo < t_end and _span_hi > t_start:
                    ax.axvspan(_span_lo, _span_hi,
                               color=sev_color, alpha=0.14, zorder=1, linewidth=0)
                    ax.axvline(ev_t_start, color=sev_color, lw=1.0, ls="--",
                               alpha=0.7, zorder=3)
                    if ev_t_end > t_start:
                        ax.axvline(min(ev_t_end, t_end), color=sev_color,
                                   lw=1.0, ls="--", alpha=0.7, zorder=3)
                    # Label inside the span
                    _lx = max(ev_t_start, t_start) + 0.02
                    if _lx < t_end:
                        ylo, yhi = ax.get_ylim()
                        ax.text(_lx, yhi * 0.90, ev_label,
                                ha="left", va="top", fontsize=8, color=sev_color,
                                fontweight="bold", zorder=8,
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc=PLOT.get("bg",NAVY),
                                          ec=sev_color, alpha=0.85, lw=0.8))

            # R peaks
            if mask_excl.any():
                ax.scatter(rp_excl[mask_excl] / fs, sig_flt[rp_excl[mask_excl]],
                           color=RED, s=90, zorder=6, marker="x", linewidths=2,
                           label="Excluded")
            if mask_ok.any():
                ax.scatter(rp_ok[mask_ok] / fs, sig_flt[rp_ok[mask_ok]],
                           color=PLOT.get("rpeak_ok","#00E676"), s=55, zorder=5,
                           marker="o", label="Accepted")
            if mask_added.any():
                ax.scatter(rp_added[mask_added] / fs, sig_flt[rp_added[mask_added]],
                           color=CYAN, s=140, zorder=7,
                           marker="*", linewidths=1.2, edgecolors=TEAL_DARK,
                           label="Added")

            ax.axhline(t_amp, color=PLOT.get("threshold",AMBER_DARK),
                       lw=1.2, ls="--", alpha=0.6)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (norm.)")

            edit_tag = "  ·  ✏ EDIT" if edit_mode else ""
            ax.set_title(
                f"{t_start:.2f}–{t_end:.2f} s  ·  {n_in_win} peaks{edit_tag}",
                loc="left",
                color=ORANGE if edit_mode else PLOT.get("text",GRAY_LIGHT),
                fontsize=9,
            )
            ax.legend(framealpha=0, loc="upper right", fontsize=8)

        self.app._slots["arr_detail"].update(draw)


    def _draw_overview_cursor(self, ax, t_start: float, t_end: float) -> list:
        """Draw the current-window highlight (the scrubber's draggable
        "cursor": a shaded span + two boundary lines) on *ax*.

        Returns the created artists so draw_overview()'s fast path can
        .remove() them before drawing the next position, instead of
        clearing/rebuilding the whole figure for a cursor move.
        """
        span  = ax.axvspan(t_start, t_end, color=ORANGE, alpha=0.16, zorder=6, linewidth=0)
        line1 = ax.axvline(t_start, color=ORANGE, lw=1.0, alpha=0.8, zorder=7)
        line2 = ax.axvline(t_end, color=ORANGE, lw=1.0, alpha=0.8, zorder=7)
        return [span, line1, line2]

    def draw_overview(self) -> None:
        """Full-recording scrubber strip above the detail plot.

        A simple flat line spanning the whole recording, plus a "current
        window" highlight recomputed fresh from live ui.nav_pos/ent_window on
        every call (never cached, so drag/scrub always reflects the true
        position) -- the highlight is the scrubber's draggable "cursor".
        Click/drag navigation is wired by NavigationController.on_overview_*,
        not here -- this method only renders. No-ops gracefully if no
        signal is loaded yet.

        The STATIC parts of this plot (baseline, annotation/pacing marks,
        axis/tick/spine setup) only depend on the recording and the
        annotation/pacing lists, which change far less often than nav_pos
        does -- they used to be torn down and rebuilt via a full
        fig.clear() + add_subplot() + constrained-layout resolve on EVERY
        navigation step (arrow key, Go, spike stepper, minimap drag tick).
        Cached here (self._ov_cache, keyed on what actually invalidates
        them) and only the 3 small cursor artists get removed/redrawn on an
        ordinary nav-pos update -- far cheaper. Falls back to a full
        rebuild whenever the cache key changes OR the cached axes is no
        longer part of the slot's current figure (resize/theme-rebuild/
        first draw), so a missed invalidation case degrades to the
        (correct, just slower) old behaviour rather than drawing wrong.
        """
        time = self.app.signal.time
        if time is None:
            return
        t_max = float(time[-1])

        try:
            win = float(self.app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except Exception:
            win = 10.0
        t_start = self.app.ui.nav_pos
        t_end   = min(t_max, t_start + win)
        _ann_snap  = list(self.app.analysis.annotations)
        _pace_snap = list(self.app.analysis.pacing_periods)

        slot = self.app._slots.get("overview")
        if slot is None:
            return

        # Everything that affects the STATIC parts of this plot -- NOT
        # t_start/t_end, which the fast path below handles on every call.
        key = (
            id(time), t_max,
            tuple((a.get("t_start"), a.get("color")) for a in _ann_snap),
            tuple(p.get("t_start") for p in _pace_snap),
        )

        cache = self._ov_cache
        if cache is not None and cache["key"] == key and cache["ax"] in slot.fig.axes:
            ax = cache["ax"]
            for artist in cache["cursor_artists"]:
                try:
                    artist.remove()
                except Exception:
                    pass
            cache["cursor_artists"] = self._draw_overview_cursor(ax, t_start, t_end)
            slot.canvas.draw_idle()
            return

        # Full (re)build -- first draw, recording/annotations changed, or
        # the cached axes was invalidated by something external (resize,
        # theme rebuild).
        def draw(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            ax.plot([float(time[0]), t_max], [0, 0], color=PLOT["signal"], lw=1.5, alpha=0.6, zorder=2)

            # Event marks -- annotation/pacing-period start times across the
            # WHOLE recording, not just the current window. Cheap: a handful
            # of axvline calls off lists already in memory, no per-sample
            # signal work (that's what made the old envelope-plot minimap
            # slow) -- gives the scrubber some context beyond a flat line.
            for ann in _ann_snap:
                ax.axvline(float(ann["t_start"]), color=ann.get("color", ORANGE_DARK),
                           lw=1.4, alpha=0.7, zorder=4)
            for pp in _pace_snap:
                ax.axvline(float(pp["t_start"]), color=TEAL_DARK, lw=1.4, alpha=0.7, zorder=4)

            cursor_artists = self._draw_overview_cursor(ax, t_start, t_end)

            ax.set_xlim(float(time[0]), t_max)
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            # Strip is short and wide -- a handful of adaptive ticks stay
            # readable, where matplotlib's default spacing would crowd or
            # get clipped entirely at this height.
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6))
            ax.tick_params(axis="x", labelsize=8, colors=PLOT["muted"])
            ax.margins(x=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)

            self._ov_cache = {"ax": ax, "key": key, "cursor_artists": cursor_artists}

        slot.update(draw)

    def draw_detail(self, t_start: float | None = None) -> None:
        """Draw the time-windowed detail view with peak markers.

        The active signal (raw or filtered) is drawn at full opacity; the
        other signal is drawn as a ghost at 20 % opacity so the filtering
        effect is always visible.  Peak markers are computed from the filtered
        signal regardless of mode — they are not re-detected on toggle.

        Also handles two extra states, both display-only (no detection state
        is touched):
        • Raw-only (just opened, Preview Detection not yet run): signal_flt
          is None, so only the raw trace is drawn, with no peaks/threshold.
        • Filter preview overlay (self.app.ui.filter_preview_on): an on-the-fly
          filtered version of the visible window, computed from the current
          filter widget values, overlaid on the raw trace so the user can
          judge filter settings before committing to Preview Detection.
        """
        if self.app.signal.time is None:
            return

        sig_flt = self.app.signal.filtered
        sig_raw = self.app.signal.raw_norm
        time    = self.app.signal.time
        fs      = self.app.signal.fs
        raw_only = sig_flt is None
        # Force "show raw" while no filtered signal exists yet — there is
        # nothing else to display, and the toggle would otherwise blank the plot.
        show_raw = self.app.ui.show_raw or raw_only

        try:
            win = float(self.app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except Exception:
            win = 10.0

        if t_start is None:
            t_start = self.app.ui.nav_pos
        t_end  = min(time[-1], t_start + win)
        # Index-slice instead of a full-length boolean mask: `time` is always
        # uniformly spaced (np.arange(n)/fs), so the window bounds map
        # directly to a sample-index range. A full `(time >= t0) & (time <= t1)`
        # mask costs O(n_samples_total) on every redraw; in edit mode this
        # runs up to ~33x/sec (hover throttle), so on long/high-fs recordings
        # that cost is easily noticeable. Slicing is O(window size) instead.
        _i0 = max(0, int(t_start * fs))
        _i1 = min(len(time), int(t_end * fs) + 1)
        mask_t = slice(_i0, _i1)

        # ── Filter preview overlay (before/after) ──────────────────────────
        # Computed for the visible window only — cheap, display-only, never
        # touches signal_flt or detection state. Works both pre- and
        # post-Preview Detection so the user can audition new filter values.
        filt_preview = None
        if self.app.ui.filter_preview_on:
            filt_preview = self.app._compute_filter_preview_segment(t_start, t_end)

        def _time_for(sig: np.ndarray | None) -> np.ndarray:
            if sig is None or len(sig) == len(time):
                return time
            log.warning(
                "Time axis length %d does not match signal length %d; truncating for plotting",
                len(time), len(sig)
            )
            return time[:len(sig)]

        def _window_mask(sig: np.ndarray | None) -> np.ndarray:
            if sig is None or len(sig) == len(time):
                return np.arange(len(time))[mask_t]
            t = time[:len(sig)]
            return (t >= t_start) & (t <= t_end)

        rp_ok          = self.app.detection.rpeaks_ok  if self.app.detection.rpeaks_ok  is not None else np.array([])
        rp_rej         = self.app.detection.rpeaks_rej if self.app.detection.rpeaks_rej is not None else np.array([])
        rp_excl        = self.app.detection.rpeaks_manual_excl  if self.app.detection.rpeaks_manual_excl  is not None else np.array([])
        rp_added       = self.app.detection.rpeaks_manual_added if self.app.detection.rpeaks_manual_added is not None else np.array([])
        t_amp          = self.app.detection.thresh_amp
        t_frac         = float(self.app.sl_thr.get()) if self.app.sl_thr is not None else 0.0
        edit_mode      = self.app.detection.edit_mode
        no_filter_mode  = self.app.signal.no_filter_mode
        signal_inverted = self.app.signal.inverted
        _ann_snap       = list(self.app.analysis.annotations)
        hover_samp      = self.app.detection.hover_samp
        hover_near      = self.app.detection.hover_samp_near

        def _in_view(idx: np.ndarray) -> np.ndarray:
            return (idx / fs >= t_start) & (idx / fs <= t_end)

        mask_ok    = _in_view(rp_ok)    if len(rp_ok)    else np.array([], bool)
        mask_rej   = _in_view(rp_rej)   if len(rp_rej)   else np.array([], bool)
        mask_excl  = _in_view(rp_excl)  if len(rp_excl)  else np.array([], bool)
        mask_added = _in_view(rp_added) if len(rp_added) else np.array([], bool)
        n_in_view  = int(mask_ok.sum())
        n_excl     = int(mask_excl.sum())
        n_added_view = int(mask_added.sum())

        _art_snap = list(self.app.analysis.artifact_candidates)
        _art_removed_by_type: dict[str, np.ndarray] = {}
        if sig_flt is not None and _art_snap:
            for _cat in ("nonphysio", "ectopic", "duplicate"):
                _samps = np.array(
                    [c["sample"] for c in _art_snap
                     if c.get("decision") == "remove" and c.get("type") == _cat],
                    dtype=int)
                if len(_samps):
                    _samps = _samps[_in_view(_samps)]
                    if len(_samps):
                        _art_removed_by_type[_cat] = _samps

        _pace_snap = list(self.app.analysis.pacing_periods)

        # Accepted beats that 0 other detection methods confirmed (Check
        # Agreement, detection_controller.py) -- cleared whenever detection
        # re-runs, so a stale set can't linger and mislabel a beat that
        # moved.
        _disagree = self.app.detection.agreement_disagree_samples
        _disagree_view = (_disagree[_in_view(_disagree)]
                           if _disagree is not None and len(_disagree) else np.array([], dtype=int))

        # ── Per-accepted-peak confidence → marker alpha ─────────────────────
        # all_candidates/all_prominences are the pre-threshold detector
        # output (same array position/order); "prominence" is a continuous
        # per-candidate strength score for every method (the ML detector's
        # is a real classifier probability, see detect_peaks_ml), previously
        # only ever used as a single global threshold cutoff. Reusing it
        # here to fade weak-but-accepted beats lets a user spot "worth
        # double-checking" beats directly in the trace instead of only via
        # the pass/fail threshold slider. Purely visual -- doesn't change
        # which peaks are accepted.
        # Cached (see _ensure_alpha_cache) -- was rebuilt from whole-
        # recording arrays with pure-Python dict/list work on every single
        # redraw, including every hover tick in Edit mode.
        rp_ok_alpha = self._ensure_alpha_cache()
        if rp_ok_alpha is None or len(rp_ok_alpha) != len(rp_ok):
            rp_ok_alpha = np.full(len(rp_ok), 0.95, dtype=float)

        primary_sig   = sig_raw   if show_raw else sig_flt
        primary_color = PLOT["raw"]      if show_raw else PLOT["filtered"]
        ghost_sig     = sig_flt   if show_raw else sig_raw
        ghost_color   = PLOT["filtered"] if show_raw else PLOT["raw"]
        if raw_only:
            label_mode = "Raw (not yet analysed)"
        elif no_filter_mode:
            label_mode = "Unfiltered" if not show_raw else "Raw (no filter)"
        else:
            label_mode = "Raw" if show_raw else "Filtered"

        def draw(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            # Detail-plot-only typography/contrast bump (Phase 3a) — scoped
            # here, not in style_axes(), so the other plots sharing it are
            # unaffected: this is the app's single most-stared-at view.
            ax.tick_params(axis="both", labelsize=10, colors=PLOT["text"])
            ax.grid(True, color=PLOT["grid"], lw=0.5, alpha=0.85)

            # ── Pacing / stimulation period markers ─────────────────────────
            # Rendered BEHIND the trace (zorder=0.5 < ghost trace's zorder=1),
            # unlike annotation spans (zorder=6, drawn OVER the trace) -- these
            # read as ambient background context. Uses ax.get_xaxis_transform()
            # (data-x, axes-fraction-y) rather than ax.get_ylim() for the label
            # position -- the annotation block's ax.get_ylim() read below only
            # works because it runs AFTER the trace is plotted; this block runs
            # BEFORE, so ax.get_ylim() would still be the (0,1) default here.
            for pp in _pace_snap:
                p0, p1 = float(pp["t_start"]), float(pp["t_end"])
                if p1 < t_start or p0 > t_end:
                    continue
                ax.axvspan(max(p0, t_start), min(p1, t_end),
                           color=TEAL, alpha=0.15, zorder=0.5, linewidth=0)
                for _px in (p0, p1):
                    if t_start <= _px <= t_end:
                        ax.axvline(_px, color=TEAL_DARK, lw=1.0, ls=":",
                                   alpha=0.55, zorder=3.5)
                note = pp.get("note", "")
                if note:
                    _lx = p0 if t_start <= p0 <= t_end else (p0 + p1) / 2
                    if t_start <= _lx <= t_end:
                        ax.text(_lx + 0.01, 0.04, note, transform=ax.get_xaxis_transform(),
                                ha="left", va="bottom", fontsize=8,
                                color=TEAL_DARK, fontweight="bold", zorder=3.5,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec=TEAL_DARK, alpha=0.75, lw=0.7))

            # Ghost trace — suppressed in no-filter mode (signals identical)
            # and in raw-only mode (no filtered signal exists yet).
            if ghost_sig is not None and not no_filter_mode and not raw_only:
                t_ghost = _time_for(ghost_sig)
                m_ghost = _window_mask(ghost_sig)
                ax.plot(t_ghost[m_ghost], ghost_sig[m_ghost],
                        color=ghost_color, lw=0.5, alpha=0.22, zorder=1,
                        label="Filtered" if show_raw else "Raw")

            # Primary trace
            if primary_sig is not None:
                t_primary = _time_for(primary_sig)
                m_primary = _window_mask(primary_sig)
                ax.plot(t_primary[m_primary], primary_sig[m_primary],
                        color=primary_color, lw=0.9, zorder=2, label=label_mode)

            # ── Filter preview overlay (before/after, current widget values) ───
            # Independent of raw/filtered toggle and of raw_only state — shows
            # what Preview Detection WOULD produce with the current filter
            # settings, computed live on just the visible window.
            if filt_preview is not None:
                t_fp, raw_fp, filt_fp = filt_preview
                ax.plot(t_fp, filt_fp, color=PLOT["signal"], lw=1.1,
                        zorder=3, alpha=0.9, label="Filtered (preview)")

            # Rejected candidates (light grey circles)
            if mask_rej.any() and sig_flt is not None:
                ax.scatter(rp_rej[mask_rej] / fs, sig_flt[rp_rej[mask_rej]],
                           color=PLOT["rpeak_bad"], s=30, zorder=4,
                           marker="o", label="Rejected", alpha=0.5)
            # Manually excluded peaks (red X markers)
            if mask_excl.any() and sig_flt is not None:
                ax.scatter(rp_excl[mask_excl] / fs, sig_flt[rp_excl[mask_excl]],
                           color=RED, s=90, zorder=6,
                           marker="x", linewidths=2,
                           label=f"Excluded ({n_excl})")
            # Accepted peaks (green dots, faded by detection confidence —
            # see rp_ok_alpha above)
            if mask_ok.any() and sig_flt is not None:
                _rgb = mcolors.to_rgb(PLOT["rpeak_ok"])
                _rgba = np.tile(_rgb + (1.0,), (int(mask_ok.sum()), 1))
                _rgba[:, 3] = rp_ok_alpha[mask_ok]
                ax.scatter(rp_ok[mask_ok] / fs, sig_flt[rp_ok[mask_ok]],
                           color=_rgba, s=55, zorder=5,
                           marker="o", label="Accepted (paler = lower confidence)")
            # Cross-method disagreement ring — hollow amber ring around an
            # accepted beat that 0 other detection methods confirmed
            # (Check Agreement). Drawn OVER the accepted dot (zorder 5.5)
            # so it reads as "flag on this beat", not a competing marker.
            if len(_disagree_view) and sig_flt is not None:
                ax.scatter(_disagree_view / fs, sig_flt[_disagree_view],
                           s=140, zorder=5.5, marker="o",
                           facecolors="none", edgecolors=AMBER, linewidths=1.8,
                           label=f"Unconfirmed ({len(_disagree_view)})")
            # Manually added peaks (cyan star — rendered on top of everything)
            if mask_added.any() and sig_flt is not None:
                ax.scatter(rp_added[mask_added] / fs, sig_flt[rp_added[mask_added]],
                           color=CYAN, s=140, zorder=7,
                           marker="*", linewidths=1.2, edgecolors=TEAL_DARK,
                           label=f"Added ({n_added_view})")

            # ── Artifact-review markers ─────────────────────────────────────
            # A beat removed via Artifact Review otherwise vanishes with no
            # trace. zorder=4.5 sits just above "rejected" (never-accepted
            # candidates, z4) and below current-state markers (accepted z5,
            # excluded z6, added z7) -- these are historical/audit markers, so
            # a current-state marker at the same position stays dominant.
            for _cat, _samps in _art_removed_by_type.items():
                _col = ARTIFACT_TYPE_COLOR[_cat]
                ax.scatter(_samps / fs, sig_flt[_samps],
                           color=_col, s=45, zorder=4.5,
                           marker="v", linewidths=1.0, edgecolors=_col, alpha=0.85,
                           label=f"Artifact removed ({_cat})")

            # ── Hover preview (edit mode — shows snapped R-peak position) ───
            if edit_mode and hover_samp is not None and sig_flt is not None:
                h_t = hover_samp / fs
                if t_start <= h_t <= t_end:
                    h_amp = float(sig_flt[hover_samp])
                    # Color: orange = replaces nearby peak, cyan = free placement
                    h_color = ORANGE if hover_near else CYAN
                    h_label = "→ replaces nearby peak" if hover_near else "→ add here"
                    # Dashed vertical guide line
                    ax.axvline(h_t, color=h_color, lw=1.0, ls="--",
                               alpha=0.65, zorder=8)
                    # Diamond marker at snapped amplitude
                    ax.scatter([h_t], [h_amp],
                               color=h_color, s=180, marker="D",
                               alpha=0.80, zorder=10, linewidths=1.4,
                               edgecolors="white", label=h_label)
                    # Small text annotation above the marker
                    ax.annotate(
                        f"{h_t:.3f} s",
                        xy=(h_t, h_amp),
                        xytext=(0, 14), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                        color=h_color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec=h_color, alpha=0.85, lw=0.8),
                        zorder=11,
                    )

            # ── Annotation spans ─────────────────────────────────────────
            for ann in _ann_snap:
                t0  = float(ann["t_start"])
                t1  = float(ann["t_end"])
                col = ann.get("color", ORANGE_DARK)
                lbl = ann.get("label", "")
                if t1 < t_start or t0 > t_end:
                    continue
                _ylo, _yhi = ax.get_ylim()
                ax.axvspan(max(t0, t_start), min(t1, t_end),
                           color=col, alpha=0.12, zorder=6, linewidth=0)
                for _tx in (t0, t1):
                    if t_start <= _tx <= t_end:
                        ax.axvline(_tx, color=col, lw=1.2, ls="-",
                                   alpha=0.8, zorder=7)
                # Label: prefer at left edge, else at midpoint
                _lx = t0 if t_start <= t0 <= t_end else (t0 + t1) / 2
                if lbl and t_start <= _lx <= t_end:
                    ax.text(_lx + 0.01, _yhi * 0.96, lbl,
                            ha="left", va="top", fontsize=8,
                            color=col, fontweight="bold", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec=col, alpha=0.88, lw=0.8))

            if not raw_only:
                ax.axhline(t_amp, color=PLOT["threshold"], lw=1.4, ls="--",
                           label=f"Threshold ({t_frac:.2f} → {t_amp:.2f} norm.)")
            ax.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Amplitude (norm.)", fontsize=11, fontweight="bold")
            filter_tag    = ""   # no_filter is the default — no need for a warning tag
            inverted_tag  = "  ·  ↕ auto-inverted" if signal_inverted else ""
            if edit_mode:
                title_suffix = "  ·  ✏ EDIT — L-click: exclude/restore   R-click: add/replace"
            else:
                title_suffix = ""
            title_color  = ORANGE if edit_mode else PLOT["text"]
            if raw_only:
                ax.set_title(
                    f"Detail  {t_start:.1f}–{t_end:.1f} s  ·  {label_mode}"
                    "  ·  click Detect Peaks to filter & find R-peaks",
                    loc="left", color=title_color)
            else:
                ax.set_title(
                    f"Detail  {t_start:.1f}–{t_end:.1f} s  ·  {n_in_view} peaks"
                    f"  ·  {label_mode}{filter_tag}{inverted_tag}{title_suffix}",
                    loc="left", color=title_color)
            # Lower right, not upper right -- the threshold line (roughly
            # mid-height on the typical 0-7 amplitude scale) and the tallest
            # accepted-peak markers both tend to sit under an upper-right
            # legend in zoomed views. A semi-opaque panel-matched background
            # keeps it readable over the trace instead of floating text.
            ax.legend(loc="lower right", framealpha=0.85,
                      facecolor=PLOT["axes"], edgecolor=PLOT["border"])

        self.app._slots["detail"].update(draw)

    def _ensure_beat_corr_cache(self) -> bool:
        """Lazily compute+cache per-beat template correlation for the
        Detection tab's Quality mini-plot.

        Identity-checked against rpeaks_ok, not recomputed on every redraw:
        rpeaks_ok is always reassigned (never mutated in place) at its one
        recompute point (detection_controller.py's run_detection(), which
        every Edit Peaks/Undo/Redo/Free Placement/threshold change funnels
        through), so an `is` check here is a correct, self-invalidating
        cache-validity test.

        The actual recompute is debounced (250ms), not run inline the
        moment the cache goes stale: compute_beat_correlation() is real
        per-beat linear algebra over every accepted beat in the WHOLE
        recording, and run_detection() reassigns rpeaks_ok on every single
        threshold-slider tick / peak-edit click, so without debouncing this
        ran synchronously on the main thread on every one of those -- for a
        long/high-beat-count recording, easily tens to hundreds of ms of
        stutter per tick during a drag. While a recompute is pending, the
        PREVIOUS peak set's cached correlation is returned (a brief, bounded
        staleness during active editing -- the same trade-off the minimap's
        scroll-zoom resync and drag-scrub already make elsewhere in this
        file) rather than blocking; draw_detail_rrhr() gets redrawn once the
        debounced recompute actually lands. Returns False if there's no
        data to compute from / show yet.
        """
        d = self.app.detection
        if d.rpeaks_ok is d._beat_corr_for:
            return d.beat_corr is not None
        if d.corr_debounce_id is not None:
            self.app.after_cancel(d.corr_debounce_id)
        d.corr_debounce_id = self.app.after(250, self._flush_beat_corr_recompute)
        return d.beat_corr is not None

    def _flush_beat_corr_recompute(self) -> None:
        d = self.app.detection
        d.corr_debounce_id = None
        sig_flt = self.app.signal.filtered
        fs = self.app.signal.fs
        if sig_flt is None or fs is None or d.rpeaks_ok is None or len(d.rpeaks_ok) < 3:
            d.beat_corr = d.beat_corr_peaks = None
            d._beat_corr_for = d.rpeaks_ok
        else:
            # Beat window sized from the shortest interval inside the user's
            # RR bounds -- the same ones Analyze's beat template uses.
            rr_lo, rr_hi = self.app.analysis_ctrl.hrv_rr_bounds(quiet=True)
            r = compute_beat_correlation(sig_flt, d.rpeaks_ok, fs, rr_lo, rr_hi)
            d.beat_corr = r["beat_corr"]
            d.beat_corr_peaks = r["valid_rp"]
            d._beat_corr_for = d.rpeaks_ok
        # Redraw just the RR/HR/Quality panel so the fresh (or now-cleared)
        # correlation data actually appears once it lands.
        self.draw_detail_rrhr()

    def _ensure_alpha_cache(self) -> "Optional[np.ndarray]":
        """Lazily compute+cache draw_detail()'s per-accepted-peak confidence
        alpha (rp_ok_alpha), same identity-cache pattern as
        _ensure_beat_corr_cache() above.

        Without this, the dict-build + list-comprehension over
        all_candidates/all_prominences (whole-recording arrays) ran on
        every single draw_detail() call -- including every ~30ms mouse-
        hover tick while in Edit mode -- with a cost that scaled with total
        recording peak count, not window size.
        """
        d = self.app.detection
        if d.rpeaks_ok is d._alpha_cache_for:
            return d.rp_ok_alpha
        rp_ok = d.rpeaks_ok if d.rpeaks_ok is not None else np.array([])
        rp_ok_alpha = np.full(len(rp_ok), 0.95, dtype=float)
        _cands = d.all_candidates
        _proms = d.all_prominences
        if (_cands is not None and _proms is not None
                and len(_cands) == len(_proms) and len(_cands) and len(rp_ok)):
            _prom_lookup = dict(zip(_cands.tolist(), _proms.tolist()))
            _rp_proms = np.array([_prom_lookup.get(int(p), np.nan) for p in rp_ok])
            _finite = np.isfinite(_rp_proms)
            if _finite.sum() >= 2:
                _lo, _hi = np.nanpercentile(_rp_proms[_finite], [5, 95])
                if _hi > _lo:
                    _norm = np.clip((_rp_proms - _lo) / (_hi - _lo), 0.0, 1.0)
                    # Floor at 0.35 -- even the weakest accepted beat should
                    # stay visible, just visually de-emphasised, not ghosted
                    # to the point of looking like it isn't there at all.
                    rp_ok_alpha = np.where(_finite, 0.35 + 0.60 * _norm, 0.95)
        d.rp_ok_alpha = rp_ok_alpha
        d._alpha_cache_for = d.rpeaks_ok
        return d.rp_ok_alpha

    def draw_detail_rrhr(self, t_start: float | None = None) -> None:
        """RR interval + instantaneous HR + beat-template Quality, stacked,
        windowed to the current Detection-view range.

        RR/HR are computed straight from rpeaks_ok/fs -- no Core Analysis
        dependency, mirrors draw_detail()'s own data source, so they're live
        the moment Detect Peaks/Preview has run. Quality comes from
        _ensure_beat_corr_cache() -- same live timing, but cached (see that
        method's docstring) since it's real per-beat linear algebra, not
        free arithmetic like RR/HR.
        """
        if "detail_rrhr" not in self.app._slots:
            return
        fs = self.app.signal.fs
        rp = self.app.detection.rpeaks_ok
        if fs is None or rp is None or len(rp) < 2:
            self.app._slots["detail_rrhr"].update(lambda fig: None)
            return

        try:
            win = float(self.app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except Exception:
            win = 10.0
        t0 = self.app.ui.nav_pos if t_start is None else t_start
        t1 = t0 + win

        # Full-array RR/HR (cheap -- rp is beat-count sized, not
        # sample-count), associated with the SECOND peak of each pair so a
        # beat whose interval started just before the window edge still
        # shows correctly.
        rr_ms = np.diff(rp) / fs * 1000.0
        rr_t = rp[1:] / fs
        hr_bpm = 60000.0 / rr_ms
        in_view = (rr_t >= t0) & (rr_t <= t1)

        self._ensure_beat_corr_cache()
        d = self.app.detection
        has_quality = d.beat_corr is not None and d.beat_corr_peaks is not None
        if has_quality:
            q_t = d.beat_corr_peaks / fs
            in_view_q = (q_t >= t0) & (q_t <= t1)

        def draw(fig):
            gs = fig.add_gridspec(3, 1, hspace=0.15)
            ax_rr = fig.add_subplot(gs[0])
            ax_hr = fig.add_subplot(gs[1], sharex=ax_rr)
            ax_q  = fig.add_subplot(gs[2], sharex=ax_rr)
            style_axes(ax_rr)
            style_axes(ax_hr)
            style_axes(ax_q)
            ax_rr.plot(rr_t[in_view], rr_ms[in_view], color=PLOT["signal"],
                       lw=1.2, marker=".")
            ax_hr.plot(rr_t[in_view], hr_bpm[in_view], color=BLUE,
                       lw=1.2, marker=".")
            if has_quality:
                ax_q.plot(q_t[in_view_q], d.beat_corr[in_view_q], color=GREEN,
                          lw=1.2, marker=".")
                # 0.90 matches the "Beats < 0.90 corr." threshold already
                # used by the Statistics panel and Summary tab.
                ax_q.axhline(0.90, color=PLOT["muted"], lw=0.6, ls="--")
            ax_rr.set_ylabel("RR (ms)", fontsize=7, color=PLOT["muted"])
            ax_hr.set_ylabel("HR (bpm)", fontsize=7, color=PLOT["muted"])
            ax_q.set_ylabel("Quality", fontsize=7, color=PLOT["muted"])
            ax_q.set_xlabel("Time (s)", fontsize=7, color=PLOT["muted"])
            ax_rr.tick_params(labelbottom=False, labelsize=7)
            ax_hr.tick_params(labelbottom=False, labelsize=7)
            ax_q.tick_params(labelsize=7)
            ax_rr.set_xlim(t0, t1)
            ax_hr.set_xlim(t0, t1)
            ax_q.set_xlim(t0, t1)

        self.app._slots["detail_rrhr"].update(draw)

    def run_plot_chain(
        self,
        tasks: list,
        on_complete: "Optional[Callable[[], None]]" = None,
        auto_epochs: bool = False,
    ) -> None:
        """Run a list of (label, fn) plot tasks sequentially via after() chain.

        *auto_epochs* starts the epoch computation afterwards, once no other
        async job is running: *on_complete* may itself start one (Analyze's
        run_freq), and a second _start_async is refused while it runs."""
        total = len(tasks)

        def _epochs_when_idle(tries: int = 0) -> None:
            # ~60 s at most, then give up rather than poll forever
            if getattr(self.app, "_async_busy", False) and tries < 300:
                self.app.after(200, lambda: _epochs_when_idle(tries + 1))
                return
            self.app._compute_epochs()

        def _run_next(idx: int) -> None:
            if idx >= total:
                self.app._set_progress(100, "Done")
                if on_complete:
                    on_complete()
                if auto_epochs:
                    self.app.after(200, _epochs_when_idle)
                return
            label, fn = tasks[idx]
            pct = int(100 * (idx + 1) / total)
            self.app._set_progress(pct, f"Rendering {label}…")
            try:
                fn()
            except Exception:
                log.exception("Plot task '%s' failed", label)
            # after_idle, not a fixed after(25, ...): each task's own
            # CanvasSlot.update() already schedules its paint via
            # canvas.draw_idle() (non-blocking), so the 25ms bought nothing
            # but latency -- it stacked to ~225ms on a normal Analyze click
            # (draw_core_results' 5 tasks chaining straight into run_freq's
            # own 4-task chain) and ~280-300ms on export/session-restore's
            # 8-task chain. after_idle still yields back to Tk between tasks
            # (so the progress bar/label keeps updating), just without the
            # artificial tax.
            self.app.after_idle(lambda i=idx + 1: _run_next(i))

        _run_next(0)

    def draw_core_results(
        self,
        on_complete: "Optional[Callable[[], None]]" = None,
        auto_epochs: bool = False,
    ) -> None:
        """Render only the fast core plots (RR, Beat, Summary, Poincaré).

        Called immediately after core analysis.  Freq / non-linear / intervals
        are rendered separately when their per-tab buttons are clicked.
        """
        r = self.app.analysis.results
        if r is None:
            return
        results: dict = r  # narrow for type checkers
        tasks = [
            ("RR / HR tachogram", lambda: self.plot_rr(results)),
            ("HRV time-domain",   lambda: self.plot_hrv_tables(results)),
            ("Poincaré",          lambda: self.plot_nonlinear(results)),
            ("Beat template",     lambda: self.plot_beat_template(results)),
            ("Summary",           lambda: self.plot_summary(results)),
        ]
        self.run_plot_chain(tasks, on_complete=on_complete, auto_epochs=auto_epochs)

    def draw_all_results(
        self,
        on_complete: "Optional[Callable[[], None]]" = None,
        auto_epochs: bool = False,
    ) -> None:
        """Render ALL result plots (used by export and legacy callers)."""
        r = self.app.analysis.results
        if r is None:
            log.warning("_draw_all_results called with no results")
            return
        results: dict = r  # narrow for type checkers
        tasks = [
            ("RR / HR tachogram",     lambda: self.plot_rr(results)),
            ("HRV tables",            lambda: self.plot_hrv_tables(results)),
            ("Poincaré / non-linear", lambda: self.plot_nonlinear(results)),
            ("PSD",                   lambda: self.plot_psd(results)),
            ("ECG intervals",         lambda: self.plot_intervals(results)),
            ("Beat template",         lambda: self.plot_beat_template(results)),
            ("Summary",               lambda: self.plot_summary(results)),
        ]
        self.run_plot_chain(tasks, on_complete=on_complete, auto_epochs=auto_epochs)

    def refresh_live_rr(self) -> None:
        """Redraw the HRV tab's RR/HR tachogram + histogram straight from
        the current accepted peaks, without needing a full Core Analysis
        run.

        Mirrors draw_detail_rrhr()'s "no Core Analysis dependency" RR/HR
        computation (rp/fs only, same np.diff(rp)/fs*1000 arithmetic) --
        the Detection tab's own RR/HR/Quality mini-strip already updates
        live on every peak add/remove/threshold change; this brings the
        HRV tab's full tachogram in line instead of it going blank
        ("Run analysis to display") until the next Analyze click.
        Deliberately narrow in scope: only RR/HR/the histogram refresh
        this way -- SDNN/RMSSD/frequency-domain/etc. still require
        Analyze, since those are materially more expensive to recompute
        on every single edit. Called from DetectionController.run_detection
        (the one place rpeaks_ok is ever reassigned), deferred via
        after(0, ...) there since run_detection can run on a background
        thread and this does real Tkinter/matplotlib work.
        """
        rp = self.app._windowed_peaks()
        fs = self.app.signal.fs
        if rp is None or fs is None or len(rp) < 3:
            return
        # Same validity rule Analyze will apply (sidebar RR bounds, beats
        # removed by artifact correction as breaks), so an out-of-range
        # interval is already marked as excluded before Analyze.
        rr_lo, rr_hi = self.app.analysis_ctrl.hrv_rr_bounds(quiet=True)
        try:
            ser = rr_interval_series(rp, fs, rr_lo, rr_hi,
                                     breaks=self.app.analysis_ctrl.hrv_breaks(rp))
        except ValueError as exc:
            log.debug("refresh_live_rr: %s", exc)
            return
        self.plot_rr({"rr_df": pd.DataFrame(), "rr_ms": ser["rr_ms"],
                      "rr_valid": ser["valid"], "rr_t_s": ser["t_s"],
                      "hr": {"rr_min_ms": rr_lo, "rr_max_ms": rr_hi}})

    def _plot_rr_all_excluded(self, ts: "pd.DataFrame", r: dict) -> None:
        """plot_rr() when no interval is valid for HRV: every interval as an
        excluded marker, and why, instead of a stale or blank tachogram."""
        t  = ts["Time_s"].to_numpy(dtype=float)
        rr = ts["RR_ms"].to_numpy(dtype=float)
        ok = np.isfinite(rr)
        if not ok.any():
            return
        t, rr = t[ok], rr[ok]
        note = (f"all {len(rr)} RR intervals excluded from HRV "
                f"(bounds {rr_bounds_label(r.get('hr'))}) — widen Max RR "
                f"if the heart rate is genuinely slow")

        def draw_all_excluded(fig, compact: bool = False):
            ax = fig.add_subplot(111)
            style_axes(ax)
            ax.plot(t, rr, "o", ls="none", ms=3.5, mfc="none", mec=RED, mew=1.0)
            ax.set_ylabel("RR (ms)")
            ax.set_xlabel("Time (s)")
            ax.set_title(f"RR Intervals  ·  {note}", loc="left",
                         fontsize=8 if compact else 9)

        def draw_no_histogram(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            ax.text(0.5, 0.5, "No valid RR interval", ha="center", va="center",
                    color=PLOT["muted"], transform=ax.transAxes)
            ax.set_axis_off()

        self.app._slots["rr"].update(draw_all_excluded)
        self.app._slots["rr"]._draw_fn_compact = \
            lambda fig: draw_all_excluded(fig, compact=True)
        self.app._slots["rr_hist"].update(draw_no_histogram)
        self.app.ui.rr_spike_times = np.array([], dtype=float)
        if self.app.ui.rr_click_cid is not None:   # stale spike list
            try:
                self.app._slots["rr"].canvas.mpl_disconnect(self.app.ui.rr_click_cid)
            except Exception as e:
                log.debug("mpl_disconnect (rr click) failed: %s", e)
            self.app.ui.rr_click_cid = None
        self.app._set_textbox(self.app.txt_rr, f"  {note}")

    def plot_rr(self, r: dict) -> None:
        """Plot RR tachogram, HR trace, and RR distribution histogram.

        Every RR interval is drawn (core.analysis.rr_timeseries): the valid
        ones -- used for HRV -- as the line, the excluded ones (outside the
        RR bounds, or spanning a beat removed by artifact correction) as
        hollow red markers with the line broken around them, so a pause
        stays visible after Analyze instead of being joined over. Mean/SD,
        spike detection and the histogram use the valid intervals only.

        Drastic RR changes are detected and shown as orange/red markers on
        the tachogram.  Clicking any point navigates to that beat in Detection.
        Right-clicking jumps specifically to the nearest spike or excluded
        interval.
        """
        ts = rr_timeseries(r)
        if len(ts) == 0:
            log.warning("_plot_rr: no RR interval — skipping")
            return
        if not ts["valid"].any():
            # e.g. the live (pre-Analyze) view of an anaesthetised mouse
            # slower than Max RR: still show the intervals, all as excluded.
            self._plot_rr_all_excluded(ts, r)
            return

        t_all  = ts["Time_s"].to_numpy(dtype=float)
        rr_all = ts["RR_ms"].to_numpy(dtype=float)
        hr_all = ts["HR_bpm"].to_numpy(dtype=float)
        valid  = ts["valid"].to_numpy(dtype=bool)
        idx_v  = np.flatnonzero(valid)
        t_v, rr_v = t_all[idx_v], rr_all[idx_v]
        excl   = ~valid & np.isfinite(rr_all)
        n_excl = int((~valid).sum())

        # ── Detect drastic RR changes ──────────────────────────────────────
        # A beat is a "spike" if its RR deviates more than spike_thr standard
        # deviations from the local rolling median (window = 15 beats).
        # Valid intervals only; the consecutive-jump test only compares two
        # ADJACENT valid intervals, never across an excluded one.
        spike_thr  = 2.5   # SD threshold
        spike_idx  = np.array([], dtype=int)
        spike_t    = np.array([], dtype=float)
        spike_rr   = np.array([], dtype=float)
        spike_mag  = np.array([], dtype=float)  # delta-RR in ms

        if len(rr_v) >= 10:
            # Rolling median via scipy — O(n·log(w)) au lieu de O(n·w)
            from scipy.ndimage import median_filter as _median_filter
            win = min(15, len(rr_v) // 2)
            roll_med = _median_filter(rr_v.astype(float), size=win, mode="nearest")
            delta = rr_v - roll_med
            # Also flag beats where consecutive delta-RR is extreme
            prev_adjacent = np.r_[False, valid[:-1]][idx_v]
            drr_padded = np.where(prev_adjacent,
                                  rr_all[idx_v] - rr_all[np.maximum(idx_v - 1, 0)], 0.0)
            rr_sd = max(float(rr_v.std()), 1.0)
            # Spike = large deviation from local median OR large consecutive jump
            spike_mask = (np.abs(delta) > spike_thr * rr_sd) | \
                         (np.abs(drr_padded) > spike_thr * rr_sd * 1.2)
            spike_idx = np.where(spike_mask)[0]
            if len(spike_idx):
                spike_t   = t_v[spike_idx]
                spike_rr  = rr_v[spike_idx]
                spike_mag = delta[spike_idx]

        # Downsample for display -- excluded intervals are NaN in the line
        # arrays so the line breaks there (the gap-preserving decimation
        # keeps those breaks on long recordings).
        t_ds,    rr_ds = downsample_xy_for_display(t_all, np.where(valid, rr_all, np.nan))
        t_ds_hr, hr_ds = downsample_xy_for_display(t_all, np.where(valid, hr_all, np.nan))
        rr_mean = float(rr_v.mean())
        hr_mean = mean_hr_bpm(rr_v)          # 60000 / mean NN (app-wide)
        rr_sd_v = float(rr_v.std())

        # Excluded intervals: hollow red markers at their true value, except
        # far-off ones (a detection dropout of several seconds, a 0-ms
        # duplicate) which are drawn at the edge of a range derived from the
        # valid ones, so they cannot squash the valid trace into a line.
        rr_hi_bound = float((r.get("hr") or {}).get("rr_max_ms", MouseECG.RR_MAX_MS))
        ex_hi = max(2.0 * float(rr_v.max()), 1.5 * rr_hi_bound)
        ex_lo = 0.5 * float(rr_v.min())
        t_ex      = t_all[excl]
        rr_ex_raw = rr_all[excl]
        rr_ex     = np.clip(rr_ex_raw, ex_lo, ex_hi)
        hr_ex     = 60_000.0 / rr_ex
        n_off     = int(((rr_ex_raw > ex_hi) | (rr_ex_raw < ex_lo)).sum())
        ex_label  = f"excluded ({n_excl})"
        if n_off:
            ex_label += f" · {n_off} off-scale at edge"
        # BLUE_MID/ORANGE_DARK match the right-panel STATISTICS accents for
        # "RR Intervals" and "Heart Rate" respectively (app.py) -- these were
        # a hardcoded, non-theme-adaptive green that didn't match either.
        c_rr, c_hr = BLUE_MID, ORANGE_DARK
        n_spikes   = len(spike_idx)
        # Snapshot once -- draw_tachogram is a closure re-invoked on every
        # redraw (annotations, theme rebuilds, ...), and the toggle button
        # calls back into this same plot_rr(), so no live-attribute read is
        # needed inside the closure itself.
        show_markers = self.app.ui.show_spike_markers

        # Spike colours: orange = moderate, red = severe
        def _spike_color(mag: float) -> str:
            return RED if abs(mag) > 3.5 * rr_sd_v else AMBER

        def draw_tachogram(fig, compact: bool = False):
            axes = fig.subplots(2, 1, sharex=True)
            for ax in axes:
                style_axes(ax)

            # ── RR tachogram ────────────────────────────────────
            axes[0].plot(t_ds, rr_ds, color=c_rr, lw=0.8, zorder=2)
            axes[0].axhline(rr_mean, color=c_rr, ls="--", lw=0.9, alpha=0.5, zorder=1)
            if len(t_ex):
                axes[0].plot(t_ex, rr_ex, "o", ls="none", ms=4.5, mfc="none",
                             mec=RED, mew=1.1, zorder=6, label=ex_label)
                axes[0].legend(loc="upper right", fontsize=7, framealpha=0,
                               labelcolor=PLOT.get("text", GRAY_LIGHT))

            # ±1 SD reference band
            axes[0].axhspan(rr_mean - rr_sd_v, rr_mean + rr_sd_v,
                            alpha=0.06, color=c_rr, zorder=0)

            if show_markers:
                axes[0].axhline(rr_mean + spike_thr * rr_sd_v,
                                color=AMBER, lw=0.6, ls=":", alpha=0.5, zorder=1)
                axes[0].axhline(rr_mean - spike_thr * rr_sd_v,
                                color=AMBER, lw=0.6, ls=":", alpha=0.5, zorder=1)

                # Spike markers
                if len(spike_t):
                    for st, sr, sm in zip(spike_t, spike_rr, spike_mag):
                        col = _spike_color(sm)
                        axes[0].scatter([st], [sr], s=55, color=col,
                                        marker="^" if sm < 0 else "v",
                                        zorder=5, edgecolors="white",
                                        linewidths=0.6, alpha=0.9)

            spike_note = (f"  ·  {n_spikes} spike{'s' if n_spikes != 1 else ''} detected"
                          if show_markers and n_spikes else "")
            axes[0].set_ylabel("RR (ms)")
            axes[0].set_title(
                f"RR Intervals  ·  mean {rr_mean:.1f} ms  ·  SD {rr_sd_v:.1f} ms{spike_note}",
                loc="left", fontsize=8 if compact else 9)
            # Right-hand hint collides with the left title once this figure
            # is squeezed into the Summary tab's mirror -- omit it there.
            if not compact:
                axes[0].set_title("left-click: go to time  ·  right-click: next spike",
                                  loc="right", fontsize=7,
                                  color=PLOT.get("muted", "#666"))
            axes[0].tick_params(labelbottom=False)

            # ── HR trace ────────────────────────────────────────
            axes[1].plot(t_ds_hr, hr_ds, color=c_hr, lw=0.8, zorder=2)
            axes[1].axhline(hr_mean, color=c_hr, ls="--", lw=0.9, alpha=0.5, zorder=1)
            if len(t_ex):
                axes[1].plot(t_ex, hr_ex, "o", ls="none", ms=4.0, mfc="none",
                             mec=RED, mew=1.0, zorder=6)

            # Mirror spikes on HR axis
            if show_markers and len(spike_t):
                spike_hr = 60_000.0 / np.clip(spike_rr, 1, None)
                for st, shr, sm in zip(spike_t, spike_hr, spike_mag):
                    col = _spike_color(sm)
                    axes[1].scatter([st], [shr], s=45, color=col,
                                    marker="^" if sm < 0 else "v",
                                    zorder=5, edgecolors="white",
                                    linewidths=0.6, alpha=0.9)

            axes[1].set_ylabel("HR (bpm)")
            # Compact (Summary mirror) keeps the caption short -- the long
            # marker-legend form gets clipped at ~1/3 width.
            axes[1].set_xlabel(
                "Time (s)" if compact or not show_markers else
                "Time (s)  —  ▲ sudden acceleration  ▼ sudden deceleration"
                "  ·  amber >2.5 SD  ·  red >3.5 SD")
            axes[1].set_title(f"Instantaneous HR  ·  mean {hr_mean:.0f} bpm",
                              loc="left", fontsize=8 if compact else 9)

        # Capture annotations for closure
        _anns = list(self.app.analysis.annotations)
        _draw_fn_orig = draw_tachogram

        def draw_tachogram_annotated(fig, compact: bool = False):
            _draw_fn_orig(fig, compact=compact)
            axs = fig.axes
            if not axs or not _anns:
                return
            ax = axs[0]          # RR axis
            ax2 = axs[1] if len(axs) > 1 else None
            for ann in _anns:
                col = ann.get("color", ORANGE_DARK)
                lbl = ann.get("label", "")
                ts, te = ann["t_start"], ann["t_end"]
                for _ax in ([ax] if ax2 is None else [ax, ax2]):
                    _ax.axvspan(ts, te, alpha=0.10, color=col, zorder=0)
                    _ax.axvline(ts, color=col, lw=1.2, ls="-", alpha=0.8, zorder=6)
                    _ax.axvline(te, color=col, lw=0.7, ls="--", alpha=0.5, zorder=6)
                if lbl:
                    ylo, yhi = ax.get_ylim()
                    ax.text((ts + te) / 2, yhi, lbl,
                            ha="center", va="top", fontsize=7, color=col,
                            fontweight="bold", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc=PLOT.get("bg","#1A1A2E"), ec=col,
                                      alpha=0.8, lw=0.6))

        self.app._slots["rr"].update(draw_tachogram_annotated)
        # Compact variant reused by plot_summary() to mirror this plot into
        # the much narrower Summary-tab panel -- see the note there.
        self.app._slots["rr"]._draw_fn_compact = \
            lambda fig: draw_tachogram_annotated(fig, compact=True)

        # ── Wire click-to-navigate ─────────────────────────────────────────
        if self.app.ui.rr_click_cid is not None:
            try:
                self.app._slots["rr"].canvas.mpl_disconnect(self.app.ui.rr_click_cid)
            except Exception as e:
                log.debug("mpl_disconnect (rr click) failed: %s", e)

        # Store spike times -- and the excluded intervals, equally worth a
        # look (a pause, a missed beat) -- for right-click navigation, and
        # on app.ui so the Detection nav bar's prev/next-spike stepper
        # (app.py) can walk the same list without recomputing it.
        _spike_times = np.sort(np.concatenate([spike_t, t_ex])).astype(float)
        _excl_times  = set(np.round(t_ex, 6).tolist())
        self.app.ui.rr_spike_times = _spike_times

        def _on_rr_click(event):
            if event.xdata is None or event.inaxes is None:
                return
            t_clicked = float(event.xdata)

            if event.button == 3 and len(_spike_times):
                # Right-click: jump to the nearest spike / excluded interval
                dists = np.abs(_spike_times - t_clicked)
                nearest_spike_t = float(_spike_times[int(np.argmin(dists))])
                t_nav = nearest_spike_t
                what = ("excluded RR" if round(nearest_spike_t, 6) in _excl_times
                        else "spike")
                spike_info = f"{what} at {nearest_spike_t:.3f} s"
            elif event.button == 1:
                # Left-click: navigate to clicked time
                t_nav = t_clicked
                spike_info = None
            else:
                return

            if self.app.signal.time is not None:
                sig_dur = float(self.app.signal.time[-1])
                try:
                    win = float(self.app.ent_window.get())  # type: ignore[union-attr]
                except Exception:
                    win = 2.0
                self.app.ui.nav_pos = max(0.0, min(t_nav - win / 2, sig_dur - win))
            self.app._sync_nav_pos_entry()
            try:
                self.app.tabs.set(TAB_DETECTION)
            except Exception as e:
                log.debug("tabs.set Detection failed: %s", e)
            self.app._draw_detail()
            if spike_info:
                self.app._set_status(f"Navigation → {spike_info}", AMBER)

        self.app.ui.rr_click_cid = self.app._slots["rr"].canvas.mpl_connect(
            "button_press_event", _on_rr_click)
        # Statistics of the valid intervals (the ones HRV used), plus how
        # many were excluded.
        _rr_desc = pd.Series(rr_v).describe()
        _rr_desc["excluded"] = float(n_excl)
        _rr_tsv  = "Metric\tRR_ms\n" + "\n".join(
            f"{k}\t{v:.5g}" for k, v in _rr_desc.items())
        self.app._set_textbox(self.app.txt_rr,
            "\n".join(f"  {k:<14} {v:>10.2f}" for k, v in _rr_desc.items()),
            tsv=_rr_tsv)

        # Valid intervals only (already inside the RR bounds the analysis
        # used); the clip only guards results from older sessions.
        _hr_b = r.get("hr") or {}
        rr_clipped = np.clip(
            rr_v,
            float(_hr_b.get("rr_min_ms", MouseECG.RR_MIN_MS)),
            float(_hr_b.get("rr_max_ms", MouseECG.RR_MAX_MS)))
        hist_title = ("RR Distribution" if not n_excl else
                      f"RR Distribution  ·  valid only — {n_excl} excluded")

        # HRV Triangular Index / TINN -- already computed by nk.hrv_time()
        # and shown as bare numbers in the Time Domain table, with no visual
        # QC element (both are geometric measures of THIS histogram's shape,
        # so a histogram is exactly where they belong).
        hti = tinn = None
        _rt = r.get("hrv_time")
        if _rt is not None and not _rt.empty:
            try:
                hti = float(_rt["HRV_HTI"].values[0])
            except Exception:
                hti = None
            try:
                tinn = float(_rt["HRV_TINN"].values[0])
            except Exception:
                tinn = None

        def draw_histogram(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            ax.hist(rr_clipped, bins=50, color=c_rr, alpha=0.7,
                    edgecolor="white", lw=0.3)
            ax.set_xlabel("RR (ms)")
            ax.set_ylabel("Count")
            ax.set_title(hist_title, loc="left")
            # A fixed 1 ms tick spacing only looks reasonable for a narrow RR
            # range (e.g. a quiet anesthetized recording). Any recording with
            # real variability -- an awake mouse, or one containing a
            # bradycardic/pause episode -- can span 100+ ms, which crams 100+
            # overlapping tick labels onto the axis. MaxNLocator adapts the
            # tick count to whatever range this recording actually has.
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10))

            # Triangular-index fit overlay -- uses NeuroKit2's own bin width
            # ((1/128) s = 7.8125 ms) so the mode/height match HRV_HTI/
            # HRV_TINN exactly rather than this panel's 50-bin display
            # histogram. The baseline is drawn split symmetrically around
            # the histogram mode rather than re-running NeuroKit2's
            # asymmetric least-squares N/M search a second time purely for
            # this QC overlay -- an honest illustration of TINN's width, not
            # a pixel-exact reproduction of its fit.
            if hti and np.isfinite(hti) and hti > 0 and len(rr_clipped) > 3:
                binsize = (1.0 / 128.0) * 1000.0
                edges = np.arange(0, np.nanmax(rr_clipped) + binsize, binsize)
                counts, edges = np.histogram(rr_clipped, bins=edges)
                if len(counts) and counts.max() > 0:
                    x_mode = float(edges[int(np.argmax(counts))])
                    y_peak = float(counts.max())
                    half = (tinn / 2.0) if (tinn and np.isfinite(tinn) and tinn > 0) else binsize
                    ax.plot([x_mode - half, x_mode, x_mode + half], [0, y_peak, 0],
                            color=RED, lw=1.3, ls="--", alpha=0.8, zorder=5)
                    label = f"HTI {hti:.2f}"
                    if tinn and np.isfinite(tinn):
                        label += f"  ·  TINN {tinn:.1f} ms"
                    ax.text(0.98, 0.95, label, ha="right", va="top", fontsize=8,
                            color=RED, transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc=PLOT["axes"], ec="none", alpha=0.8))

        self.app._slots["rr_hist"].update(draw_histogram)

    # Display (label, unit) for the metrics researchers actually read off
    # this table -- bare NeuroKit column names carry no unit and don't say
    # which of ms/%/ratio a given row is. pNN50/pNN20 are flagged "(human)"
    # because their fixed thresholds (50 ms / 20 ms) are calibrated to human
    # RR intervals, not the ~120 ms mouse RR this app is built for -- pNN6
    # is the mouse-equivalent metric NeuroKit doesn't compute by default,
    # which is why this app adds it as its own column alongside them.
    _TD_DISPLAY = {
        "MeanNN":  ("Mean RR", "ms"),
        "SDNN":    ("SDNN", "ms"),
        "RMSSD":   ("RMSSD", "ms"),
        "SDSD":    ("SD of successive diffs", "ms"),
        "CVNN":    ("CV of RR (SDNN/mean)", ""),
        "CVSD":    ("CV of successive diffs", ""),
        "MedianNN":("Median RR", "ms"),
        "MadNN":   ("Median abs. deviation", "ms"),
        "MCVNN":   ("Median CV of RR", ""),
        "IQRNN":   ("Interquartile range", "ms"),
        "SDRMSSD": ("SDNN / RMSSD", ""),
        "Prc20NN": ("20th percentile RR", "ms"),
        "Prc80NN": ("80th percentile RR", "ms"),
        "pNN50":   ("% RR diff. > 50 ms (human)", "%"),
        "pNN20":   ("% RR diff. > 20 ms (human)", "%"),
        "pNN6":    ("% RR diff. > 6 ms (mouse)", "%"),
        "MinNN":   ("Min RR", "ms"),
        "MaxNN":   ("Max RR", "ms"),
        "HTI":     ("Triangular index", ""),
        "TINN":    ("Triangular interpolation", "ms"),
    }

    def plot_hrv_tables(self, r: dict) -> None:
        """Populate time-domain and frequency-domain HRV text boxes."""
        td_df = r["hrv_time"]
        if td_df is None or td_df.empty:
            self.app._set_textbox(self.app.txt_td, "  (not computed)")
        else:
            td_lines: list[str] = [
                "  Mouse-specific: read pNN6, not pNN50/pNN20 (human RR thresholds)",
            ]
            n_excl, _ = self._rr_exclusion(r)
            if n_excl:
                # Bounds actually used (hr); older results lacking the counts
                # fall back to the recomputed mask's count and the defaults.
                excl_src = dict(r.get("hr") or {})
                excl_src.setdefault("n_excluded", n_excl)
                td_lines.append(f"  {rr_exclusion_text(excl_src)} — not bridged")
            td_lines.append("")
            for col in td_df.columns:
                try:
                    v = float(td_df[col].values[0])
                    if not np.isfinite(v):
                        continue
                except Exception as exc:
                    log.debug("_plot_hrv_tables td skip '%s': %s", col, exc)
                    continue
                name = col.replace("HRV_", "")
                label, unit = self._TD_DISPLAY.get(name, (name, ""))
                disp = f"{label} ({unit})" if unit else label
                fmt = f"{v:.1f}" if abs(v) >= 100 else f"{v:.3f}" if abs(v) >= 1 else f"{v:.5f}"
                dots = "·" * max(2, 34 - len(disp))
                td_lines.append(f"  {disp} {dots}  {fmt:>10}")
            self.app._set_textbox(
                self.app.txt_td, "\n".join(td_lines) or "  (no finite values)",
                tsv=self.app._df_to_tsv(td_df))

        fd_df = r["hrv_freq"]
        if fd_df is None or fd_df.empty:
            withheld = getattr(fd_df, "attrs", {}).get("withheld_reason") if fd_df is not None else None
            self.app._set_textbox(
                self.app.txt_fd,
                f"  Spectrum withheld — {withheld}" if withheld else "  (not computed)")
            return

        lines: list[str] = []
        # An older session's nk.hrv_frequency table: arbitrary-unit powers,
        # LFn = LF / TP. Said first, and labelled as such below.
        legacy = hrv_freq_is_legacy(r)
        if legacy:
            lines += ["  " + ln for ln in textwrap.wrap("⚠ " + LEGACY_SPECTRAL_TEXT, 60)] + [""]
        pw_unit = "a.u." if legacy else "ms²"
        for col in fd_df.columns:
            try:
                v    = float(fd_df[col].values[0])
                name = col.replace("HRV_", "")
                if not np.isfinite(v):
                    continue
                if name in ("LFn", "HFn") and legacy:
                    lines.append(f"  {name[:2] + ' (% of TP)':<26} {v * 100:>10.1f} %")
                elif name in ("LFn", "HFn"):
                    # Task Force normalised units: LF or HF / (LF + HF).
                    lines.append(f"  {name[:2] + ' (n.u.)':<26} {v * 100:>10.1f} %")
                elif name in ("LF", "HF", "VLF", "ULF", "VHF", "TP"):
                    # Absolute band power (ms²) of the spectrum on the PSD
                    # panel -- not a fraction of anything.
                    lines.append(f"  {name + f' power ({pw_unit})':<26} {v:>10.3f}")
                elif name == "LFHF":
                    lines.append(f"  {'LF/HF ratio':<26} {v:>10.3f}")
                elif name in ("LFpeak", "HFpeak"):
                    lines.append(f"  {name[:2] + ' peak (Hz)':<26} {v:>10.3f}")
                elif name == "LnHF":
                    lines.append(f"  {f'ln HF (ln {pw_unit})':<26} {v:>10.3f}")
                else:
                    lines.append(f"  {name:<26} {v:>10.4f}")
            except Exception as exc:
                log.debug("_plot_hrv_tables fd skip '%s': %s", col, exc)

        # Build TSV alongside the display text
        _fd_tsv_rows = ["Metric\tValue"]
        for col in fd_df.columns:
            try:
                v = float(fd_df[col].values[0])
                if np.isfinite(v):
                    _fd_tsv_rows.append(f"{col.replace('HRV_', '')}\t{v:.6g}")
            except (TypeError, ValueError) as _tsv_exc:
                log.debug("_plot_hrv_tables TSV: skip col %s: %s", col, _tsv_exc)
        _fd_tsv = "\n".join(_fd_tsv_rows)
        self.app._set_textbox(self.app.txt_fd, "\n".join(lines) if lines else "  (not computed)",
                          tsv=_fd_tsv if len(_fd_tsv_rows) > 1 else None)

    def psd_for_display(self, r: dict) -> "dict":
        """The spectrum the PSD panel draws, with the band metrics read off it.

        Normally the spectrum analyse_hrv_freq() stored in
        ``hrv_freq.attrs`` -- the very curve its VLF/LF/HF/LF-HF numbers were
        integrated from, so plot and numbers cannot disagree. Before the
        frequency HRV was run (or for a result from an older version that
        did not store it) the same ``hrv_spectrum`` is computed here from
        the valid RR intervals, and its band metrics are shown instead.

        Returns {"withheld": str|None, "freqs", "psd", "metrics", "bands",
        "n_points", "df_hz", "stored": bool, "stale_numbers": bool}."""
        fd = r.get("hrv_freq")
        attrs = getattr(fd, "attrs", {}) if fd is not None else {}
        bands = attrs.get("bands") or [list(MouseECG.VLF), list(MouseECG.LF),
                                        list(MouseECG.HF)]
        bands = [tuple(float(x) for x in b) for b in bands]
        out: dict = {"withheld": None, "freqs": None, "psd": None, "metrics": {},
                     "bands": bands, "n_points": 0, "df_hz": float("nan"),
                     "stored": False, "stale_numbers": False}
        withheld = hrv_freq_withheld_reason(r)
        if withheld:
            out["withheld"] = "Spectrum withheld\n" + withheld.replace(" — ", "\n")
            return out
        if attrs.get("psd_freqs") is not None and attrs.get("psd_power") is not None:
            freqs = np.asarray(attrs["psd_freqs"], dtype=float)
            psd   = np.asarray(attrs["psd_power"], dtype=float)
            out.update(freqs=freqs, psd=psd, stored=True,
                       n_points=int(attrs.get("psd_n_points", 0) or 0),
                       df_hz=float(freqs[1] - freqs[0]) if len(freqs) > 1 else float("nan"))
            out["metrics"] = {c: float(fd[c].values[0]) for c in fd.columns
                              if c.startswith("HRV_")}
            return out
        # Not computed yet (or an older result): same estimator, on the fly.
        rdf = r.get("rr_df")
        if rdf is not None and not rdf.empty and {"Time_s", "RR_ms"} <= set(rdf.columns):
            # analyse_core's rr_df: valid intervals with their end times
            ts    = np.asarray(rdf["Time_s"].values, dtype=float)
            rr_ms = np.asarray(rdf["RR_ms"].values, dtype=float)
        else:
            rr_all = np.asarray(r["rr_ms"], dtype=float)
            valid  = self._rr_valid(r)
            ts     = (np.cumsum(rr_all) / 1000.0)[valid]   # cumsum over ALL = true times
            rr_ms  = rr_all[valid]
        if len(rr_ms) < MIN_BEATS_SPECTRAL:
            log.warning("_plot_psd: too few valid RR intervals (%d, need ≥ %d)",
                        len(rr_ms), MIN_BEATS_SPECTRAL)
            out["withheld"] = (f"Spectral HRV requires ≥ {MIN_BEATS_SPECTRAL} valid RR "
                               f"intervals\n(recording has {len(rr_ms)})")
            return out
        if r.get("rr_ms") is not None:
            reason = spectral_withheld_reason(np.asarray(r["rr_ms"], dtype=float),
                                              self._rr_valid(r))
            if reason:
                log.warning("_plot_psd: spectrum withheld — %s", reason)
                out["withheld"] = "Spectrum withheld\n" + reason.replace(" — ", "\n")
                return out
        spec = hrv_spectrum(rr_ms, ts)
        if spec is None:
            out["withheld"] = "RR series too short for a spectrum"
            return out
        out.update(freqs=spec["freqs"], psd=spec["psd"], n_points=spec["n_points"],
                   df_hz=spec["df_hz"],
                   metrics=spectral_band_metrics(spec, *bands),
                   stale_numbers=fd is not None and not fd.empty)
        return out

    def plot_psd(self, r: dict) -> None:
        """Welch PSD with mouse-specific VLF / LF / HF band shading.

        Draws ``psd_for_display``: the spectrum analyse_hrv_freq()
        integrated into the VLF/LF/HF powers and LF/HF numbers shown in the
        tables, KPIs and exports (core ``hrv_spectrum``: valid RR intervals
        cubic-spline resampled at 20 Hz on their true times, Welch with a
        Hann window, 75 % overlap, ≈0.02 Hz resolution). The shaded areas
        are exactly the bins integrated into each band (``band_mask``) and
        the legend values are those band powers (ms²) and n.u.

        The spectrum is WITHHELD (panel shows why) under the same rule as
        analyse_hrv_freq (core.analysis.spectral_withheld_reason): a run
        of excluded intervals longer than MouseECG.SPECTRAL_MAX_GAP_S, or
        excluded time above MouseECG.SPECTRAL_MAX_EXCLUDED_FRAC of the
        span -- the spline would otherwise invent RR values across it.
        """
        disp = self.psd_for_display(r)
        if disp["withheld"]:
            def draw_warn(fig, _msg=disp["withheld"]):
                ax = fig.add_subplot(111)
                style_axes(ax)
                ax.text(0.5, 0.5, _msg,
                        ha="center", va="center", color=PLOT["muted"],
                        transform=ax.transAxes, fontsize=11)
                ax.set_axis_off()
            self.app._slots["psd"].update(draw_warn)
            return
        try:
            freqs, psd = disp["freqs"], disp["psd"]
            met = disp["metrics"]
            (vlf, lf, hf) = disp["bands"]

            def _num(key: str) -> float:
                v = met.get(key, float("nan"))
                return float(v) if v is not None else float("nan")

            # Respiration-rate estimate: the HF-band peak of this spectrum
            # (HRV_HFpeak), breaths/min = Hz * 60 -- HF is labelled
            # "respiratory" (respiratory sinus arrhythmia).
            hf_peak_hz = _num("HRV_HFpeak")
            hf_peak_hz = hf_peak_hz if np.isfinite(hf_peak_hz) else None
            hf_mask = band_mask(freqs, hf)

            def _pw(key: str) -> str:
                v = _num(key)
                return f"{v:.2f} ms²" if np.isfinite(v) else "— ms²"

            def _nu(key: str) -> str:
                v = _num(key)
                return f", {100.0 * v:.1f} n.u." if np.isfinite(v) else ""

            resp = (f", resp. ≈{hf_peak_hz * 60:.0f}/min" if hf_peak_hz is not None
                    else ", respiratory")
            bands = [
                (vlf, f"VLF  {vlf[0]:g}–{vlf[1]:g} Hz  ({_pw('HRV_VLF')})", BLUE_DARK),
                (lf,  f"LF   {lf[0]:g}–{lf[1]:g} Hz  ({_pw('HRV_LF')}{_nu('HRV_LFn')}, "
                      f"baroreflex)", PURPLE),
                (hf,  f"HF   {hf[0]:g}–{hf[1]:g} Hz  ({_pw('HRV_HF')}{_nu('HRV_HFn')}{resp})",
                 GREEN_DARK),
            ]
            lfhf = _num("HRV_LFHF")
            df = disp["df_hz"]
            n_pts = disp["n_points"]
            stale = disp["stale_numbers"]

            # Band colours (BLUE_DARK/PURPLE/GREEN_DARK) are lightened for
            # dark mode by theme.py's _adapt_for_mode(); at a fixed alpha
            # that lighter fill reads as pastel on light but noticeably more
            # saturated against the dark axes background, so tone it down to
            # match how the light-mode fill reads.
            band_alpha = 0.24 if THEME.is_dark else 0.35

            def draw_psd(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                ax.semilogy(freqs, psd, color=PLOT["muted"], lw=1.0, zorder=3)
                for band, legend_label, color in bands:
                    m = band_mask(freqs, band)
                    ax.fill_between(freqs, psd, where=m, alpha=band_alpha, color=color,
                                    label=legend_label, zorder=2)
                    # Vertical band boundary lines
                    for boundary in band:
                        if 0 < boundary < MouseECG.PSD_XLIM:
                            ax.axvline(boundary, color=color, lw=0.8, ls=":",
                                       alpha=0.6, zorder=1)
                if hf_peak_hz is not None and hf_mask.any():
                    ax.plot(hf_peak_hz, float(psd[hf_mask].max()),
                            marker="v", ms=7, color=GREEN_DARK, zorder=5)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("PSD (ms²/Hz)")
                ax.set_xlim(0, MouseECG.PSD_XLIM)
                ax.legend(framealpha=0, loc="upper right", fontsize=9)
                lfhf_txt = f"  ·  LF/HF {lfhf:.2f}" if np.isfinite(lfhf) else ""
                stale_txt = ("  ·  older result: re-run Frequency HRV for matching numbers"
                             if stale else "")
                ax.set_title(
                    f"Power spectral density  (Welch · Δf={df:.3f} Hz · "
                    f"n={n_pts} pts){lfhf_txt}{stale_txt}",
                    loc="left",
                )

            self.app._slots["psd"].update(draw_psd)
        except Exception as exc:
            log.warning("_plot_psd failed: %s", exc)

    def plot_nonlinear(self, r: dict) -> None:
        """Poincaré plot and non-linear HRV metric table."""
        self.app._set_textbox(
            self.app.txt_nl,
            self.app._df_to_text(r["hrv_nonlin"], display_map=self.app._NL_DISPLAY),
            tsv=self.app._df_to_tsv(r["hrv_nonlin"]))

        rr_ms = np.asarray(r["rr_ms"], dtype=float)
        valid = self._rr_valid(r)
        # Successive VALID pairs only -- same pairs nk.hrv_nonlinear() used
        # for SD1/SD2: a pair straddling an excluded interval (e.g. a pause)
        # is dropped rather than plotted as an outlier point.
        rr_a, rr_b = successive_valid_pairs(rr_ms, valid)
        if len(rr_a) < 1:
            return
        rr_v = rr_ms[valid]

        nl   = r["hrv_nonlin"]
        sd1  = self.app._safe_df_val(nl, "HRV_SD1", 1)
        sd2  = self.app._safe_df_val(nl, "HRV_SD2", 1)
        # SD1/SD2 use every valid pair of the window (poincare_sd1_sd2), the
        # same cloud as the scatter below. Only a result from an older
        # version (everything on the first 1000 beats) is flagged; the MSE
        # curve may come from the capped entropy block (noted on its title).
        trunc_note = ""
        attrs = getattr(nl, "attrs", {})
        if attrs.get("truncated"):
            trunc_note = (f"  ·  SD1/SD2 from first {attrs['n_beats_used']:,} beats "
                          f"(older analysis)")
        mse_note = ""
        if attrs.get("entropy_capped"):
            mse_note = (f"  ·  {int(attrs.get('entropy_n_rr', 0)):,} RR, "
                        f"{float(attrs.get('entropy_t_start_s', 0.0)):.0f}–"
                        f"{float(attrs.get('entropy_t_end_s', 0.0)):.0f} s")

        # Ellipse geometry: centred on the plotted series' own mean RR;
        # SD1 = semi-minor axis (perpendicular to the identity line --
        # short-term/beat-to-beat spread), SD2 = semi-major axis (along the
        # identity line -- longer-term spread). Standard Poincaré-ellipse
        # convention; previously SD1/SD2 were only printed as title text
        # with no visual link to the scatter they describe.
        try:
            sd1_f = float(nl["HRV_SD1"].values[0])
            sd2_f = float(nl["HRV_SD2"].values[0])
            have_ellipse = np.isfinite(sd1_f) and np.isfinite(sd2_f)
        except Exception as exc:
            log.debug("_plot_nonlinear: SD1/SD2 ellipse skipped: %s", exc)
            have_ellipse = False
        center = float(rr_v.mean())

        lim  = [float(min(rr_a.min(), rr_b.min())) - 20,
                float(max(rr_a.max(), rr_b.max())) + 20]

        mse_scales = attrs.get("mse_scales") or []
        mse_values = attrs.get("mse_values") or []
        mse_point  = attrs.get("mse_point_estimate")
        have_mse   = bool(mse_scales) and bool(mse_values)

        def draw_poincare(fig, compact: bool = False):
            from matplotlib.gridspec import GridSpec
            # Same reason as draw_intervals()/draw_rr_timeline(): CanvasSlot's
            # constrained_layout would override explicit row-height margins.
            try:
                fig.set_layout_engine(None)
            except Exception as exc:
                log.debug("draw_poincare: set_layout_engine(None) failed: %s", exc)
            # Compact (Summary mirror) drops the MSE subplot entirely --
            # at ~1/3 width its subtitle overlapped the Poincaré axis
            # label/ticks above it.
            if compact:
                gs = GridSpec(1, 1, figure=fig, left=0.16, right=0.96, top=0.90, bottom=0.14)
            else:
                gs = GridSpec(2, 1, figure=fig, height_ratios=[5, 2], hspace=0.35,
                             left=0.13, right=0.96, top=0.93, bottom=0.09)
            ax = fig.add_subplot(gs[0, 0])
            style_axes(ax)
            ax.scatter(rr_a, rr_b, s=3, alpha=0.25, color=BLUE, rasterized=True)
            ax.plot(lim, lim, color=BORDER2, lw=1, ls="--", alpha=0.7)
            if have_ellipse:
                ax.add_patch(matplotlib.patches.Ellipse(
                    (center, center), width=2 * sd2_f, height=2 * sd1_f,
                    angle=45, facecolor="none", edgecolor=RED, lw=1.5,
                    alpha=0.8, zorder=4))
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            # Don't use set_aspect("equal") — it creates dead whitespace when
            # the container isn't square. Force equal axes via xlim/ylim instead.
            ax.set_xlabel(r"$RR_n$ (ms)")
            ax.set_ylabel(r"$RR_{n+1}$ (ms)")
            ax.set_title(f"Poincaré diagram  SD1={sd1}  SD2={sd2}{trunc_note}",
                         loc="left", fontsize=8 if compact else 9)

            if compact:
                return

            # Multiscale entropy curve -- see analyse_hrv_nonlinear() for why
            # this needs materially more beats than a single SampEn value and
            # is skipped (not fabricated) below a minimum beat count.
            ax_mse = fig.add_subplot(gs[1, 0])
            style_axes(ax_mse)
            if have_mse:
                ax_mse.plot(mse_scales, mse_values, color=PURPLE, lw=1.3,
                           marker="o", ms=3, zorder=2)
                pt_txt = f"  ·  Σ={mse_point:.2f}" if mse_point is not None else ""
                ax_mse.set_title(f"Multiscale entropy{pt_txt}{mse_note}", loc="left", fontsize=8)
                ax_mse.set_xlabel("Scale factor", fontsize=8)
                ax_mse.set_ylabel("SampEn", fontsize=8)
                ax_mse.set_xticks(mse_scales)
            else:
                ax_mse.set_title("Multiscale entropy", loc="left", fontsize=8)
                ax_mse.text(0.5, 0.5, f"Needs ≥60 valid RR intervals (have {len(rr_v)})",
                           ha="center", va="center", color=PLOT["muted"],
                           fontsize=8, transform=ax_mse.transAxes)
                ax_mse.set_xticks([]); ax_mse.set_yticks([])

        self.app._slots["poincare"].update(draw_poincare)
        # Compact variant reused by plot_summary() to mirror this plot into
        # the much narrower Summary-tab panel -- see the note there.
        self.app._slots["poincare"]._draw_fn_compact = \
            lambda fig: draw_poincare(fig, compact=True)

    def plot_intervals(self, r: dict) -> None:
        """Violin + box plot for PR / QRS / QT / QTc intervals."""
        ivl   = r["intervals"]
        if ivl is None or ivl.empty:
            def draw_unavailable_early(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                ax.text(0.5, 0.5, "Interval delineation not computed yet",
                        ha="center", va="center",
                        color=PLOT["muted"], transform=ax.transAxes)
                ax.set_axis_off()
            self.app._slots["intervals"].update(draw_unavailable_early)
            return
        cols  = [c for c in ["PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]
                 if c in ivl.columns and ivl[c].notna().sum() > 3]

        # Completion stats echoed into the plot title so an exported figure
        # still carries "measured on 17% of beats" instead of looking like
        # a clean, complete result (same PR/QRS/QT columns run_intervals()
        # grades the sidebar status by).
        _core_cols = [c for c in ["PR_ms", "QRS_ms", "QT_ms"] if c in ivl.columns]
        n_ok_ivl    = int((~ivl[_core_cols].isna().any(axis=1)).sum()) if _core_cols else 0
        n_total_ivl = len(ivl)
        pct_ivl     = 100.0 * n_ok_ivl / n_total_ivl if n_total_ivl else 0.0

        if not cols:
            def draw_unavailable(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                ax.text(0.5, 0.5,
                        "Interval delineation not available\n"
                        "(requires clear P/Q/S/T waves at high SNR)",
                        ha="center", va="center",
                        color=PLOT["muted"], transform=ax.transAxes, linespacing=1.8)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            self.app._slots["intervals"].update(draw_unavailable)
            return

        legacy_ivl = intervals_use_legacy_definitions(ivl)
        palette  = [BLUE_DARK, GREEN_DARK, PINK, ORANGE_DARK]
        col_data = [(col, ivl[col].dropna().values, color)
                    for col, color in zip(cols, palette)]

        # No reference bands (the experimental-context selector they came
        # from was removed) -- every _ref.get(col, ...) below falls
        # through to (None, None), which the drawing code already treats
        # as "no band to draw".
        _ref: "dict[str, tuple[float, float]]" = {}

        # QT-RR regression diagnostic: whether the currently selected QTc
        # formula (Mitchell/Bazett/Fridericia) actually removed rate-dependence.
        # A well-corrected QTc should be roughly flat against RR; a nonzero
        # slope means the correction under/over-compensates at this
        # recording's heart-rate range -- exactly the check the rodent QTc
        # literature recommends instead of trusting one formula's number
        # blindly (Bazett/Fridericia are documented to mis-correct badly at
        # rodent RR intervals).
        have_qtrr = "QTc_ms" in ivl.columns and ivl["QTc_ms"].notna().sum() > 3
        if have_qtrr:
            # The RR each QTc was corrected with (RR_prev_ms for a re-paired
            # legacy frame, whose RR_ms is the following RR).
            _rr_col = qtc_rr_column(ivl)
            _qtrr = ivl[[_rr_col, "QTc_ms"]].dropna()
            rr_qtrr  = _qtrr[_rr_col].values.astype(float)
            qtc_qtrr = _qtrr["QTc_ms"].values.astype(float)
            try:
                from scipy.stats import linregress
                _fit = linregress(rr_qtrr, qtc_qtrr)
                slope, intercept, rval = _fit.slope, _fit.intercept, _fit.rvalue
            except Exception as exc:
                log.debug("draw_intervals: QT-RR linregress failed: %s", exc)
                have_qtrr = False
                slope = intercept = rval = float("nan")

        def draw_intervals(fig):
            from matplotlib.gridspec import GridSpec
            n_cols = len(col_data) + (1 if have_qtrr else 0)
            # This figure now arrives from CanvasSlot with constrained_layout
            # active, which would recompute/override the explicit margins
            # below -- opt this one figure out so they stay pixel-identical.
            try:
                fig.set_layout_engine(None)
            except Exception as exc:
                log.debug("draw_intervals: set_layout_engine(None) failed: %s", exc)
            # Definitions line wrapped to the figure width (it was clipped on
            # figures narrower than ~7.6 in); the axes make room for it.
            defs_txt = wrap_fig_text(
                "Legacy interval definitions (older session) — re-run Intervals"
                if legacy_ivl else INTERVAL_DEFINITIONS_TEXT,
                0.96 * fig.get_figwidth(), 6.5)
            n_def_lines = defs_txt.count("\n") + 1
            gs = GridSpec(1, n_cols, figure=fig,
                          left=0.10, right=0.97, top=0.88,
                          bottom=min(0.30, 0.08 + (n_def_lines - 1) * 0.11
                                     / max(fig.get_figheight(), 1.0)),
                          wspace=0.40)
            title_color = GREEN if pct_ivl >= 90 else ORANGE_DARK if pct_ivl >= 50 else RED
            fig.suptitle(
                f"{n_ok_ivl}/{n_total_ivl} beats measured ({pct_ivl:.0f}%)",
                color=title_color, fontsize=9, y=0.985)
            # Which landmarks the numbers come from -- old sessions keep
            # their legacy (Q-nadir / from-R) values until re-run.
            fig.text(0.5, 0.005, defs_txt,
                     ha="center", va="bottom", fontsize=6.5, linespacing=1.1,
                     multialignment="center",
                     color=ORANGE_DARK if legacy_ivl else PLOT["muted"])
            for ci, (col, data, color) in enumerate(col_data):
                ax = fig.add_subplot(gs[0, ci])
                style_axes(ax)
                finite = data[np.isfinite(data)] if len(data) >= 2 else np.array([])
                if len(finite) < 2:
                    ax.text(0.5, 0.5, f"{col.replace('_ms','')}\nn<2",
                            ha="center", va="center", color=PLOT["muted"],
                            transform=ax.transAxes, fontsize=9)
                    ax.set_axis_off()
                    continue
                # Violin
                try:
                    vp = ax.violinplot(finite, positions=[0], widths=0.7,
                                       showmedians=False, showextrema=False)
                    for body in vp["bodies"]:
                        body.set_facecolor(color)
                        body.set_alpha(0.25)
                        body.set_edgecolor(color)
                        body.set_linewidth(0.8)
                except Exception:
                    pass
                # Box on top of violin
                ax.boxplot(finite, positions=[0], widths=0.18, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.35, linewidth=0.8),
                           medianprops=dict(color="white", lw=2.0),
                           whiskerprops=dict(color=color, lw=1.0, alpha=0.7),
                           capprops=dict(color=color, lw=1.0),
                           flierprops=dict(marker=".", color=MUTED, ms=2, alpha=0.4))
                # Reference range band
                lo_ref, hi_ref = _ref.get(col, (None, None))
                if lo_ref is not None:
                    ax.axhspan(lo_ref, hi_ref, color=color, alpha=0.08, zorder=0)
                    ax.axhline(lo_ref, color=color, lw=0.7, ls=":", alpha=0.5)
                    ax.axhline(hi_ref, color=color, lw=0.7, ls=":", alpha=0.5)
                # Stats annotation (compact)
                med = float(np.median(finite))
                ax.text(0.5, 0.97,
                        f"med {med:.1f}  ±{finite.std():.1f}",
                        ha="center", va="top", color=PLOT["muted"], fontsize=8,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc=PLOT["axes"], ec="none", alpha=0.8))
                label = col.replace("_ms", "")
                ax.set_title(label, color=color, fontsize=10, pad=4)
                if col == "QT_ms" and len(finite) > 3:
                    # Beat-to-beat QT variability on this single lead. It
                    # used to be shown as "dispersion" (max - min QT), but
                    # classical QT dispersion is an inter-LEAD measure of
                    # one beat; the range here is dominated by delineation
                    # outliers, so the robust IQR is shown first.
                    qtv = qt_beat_variability(finite)
                    ax.text(0.5, 0.03,
                            f"beat-to-beat IQR {qtv['iqr_ms']:.1f}  ·  "
                            f"range {qtv['range_ms']:.1f} ms",
                            ha="center", va="bottom", color=PLOT["muted"],
                            fontsize=7, transform=ax.transAxes)
                ax.set_ylabel("ms", fontsize=8)
                ax.set_xticks([])
                # Auto-range with padding, extended toward the reference
                # band when it doesn't overlap the data -- ylim used to
                # come from the data percentiles alone, so a band outside
                # that range (e.g. QRS running fast/slow vs. the selected
                # context) was drawn off-screen and never seen. The
                # extension is capped at 2x the data's own spread so a
                # reference band far wider than the data (the opposite
                # failure this tab also showed -- QTc's band filling the
                # whole panel) doesn't swallow the distribution either;
                # past the cap, a small arrow annotation says where the
                # band actually sits instead of silently clipping it.
                p2, p98 = np.percentile(finite, [2, 98])
                spread = p98 - p2
                pad = max(spread * 0.25, 5)
                ylo, yhi = p2 - pad, p98 + pad
                if lo_ref is not None:
                    cap = max(spread * 2.0, pad)
                    if lo_ref < ylo:
                        ylo = max(lo_ref - pad, ylo - cap)
                    if hi_ref > yhi:
                        yhi = min(hi_ref + pad, yhi + cap)
                ax.set_ylim(ylo, yhi)
                if lo_ref is not None and lo_ref < ylo:
                    ax.text(0.5, 0.01, f"ref {lo_ref:.0f}–{hi_ref:.0f} ↓", ha="center",
                            va="bottom", color=color, fontsize=6.5, transform=ax.transAxes)
                elif lo_ref is not None and hi_ref > yhi:
                    ax.text(0.5, 0.99, f"ref {lo_ref:.0f}–{hi_ref:.0f} ↑", ha="center",
                            va="top", color=color, fontsize=6.5, transform=ax.transAxes)

            if have_qtrr:
                axr = fig.add_subplot(gs[0, n_cols - 1])
                style_axes(axr)
                axr.scatter(rr_qtrr, qtc_qtrr, s=6, color=ORANGE_DARK,
                            alpha=0.35, linewidths=0, zorder=2)
                xs = np.array([rr_qtrr.min(), rr_qtrr.max()])
                axr.plot(xs, slope * xs + intercept, color=RED, lw=1.3,
                         zorder=3)
                axr.set_title("QTc–RR", color=ORANGE_DARK, fontsize=10, pad=4)
                axr.text(0.5, 0.97,
                         f"slope {slope:+.3f}  ·  r={rval:.2f}",
                         ha="center", va="top", color=PLOT["muted"], fontsize=8,
                         transform=axr.transAxes,
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc=PLOT["axes"], ec="none", alpha=0.8))
                axr.set_xlabel("Preceding RR (ms)", fontsize=8)
                axr.set_ylabel("QTc (ms)", fontsize=8)

        self.app._slots["intervals"].update(draw_intervals)

    def plot_beat_template(self, r: dict) -> None:
        """Average beat template, ±1 SD band, and amplitude / morphology distributions.

        All heavy numpy work (beat matrix, SD, per-beat correlations) was pre-computed
        in analyse_core() on the background thread.  This function only renders.
        """
        beat_time   = r.get("beat_time")
        mean_beat   = r.get("beat_template")
        beat_matrix = r.get("beat_matrix")
        beat_sd     = r.get("beat_sd")
        beat_corr   = r.get("beat_corr")
        peak_amps   = r.get("peak_amps")

        if beat_time is None or mean_beat is None or beat_matrix is None:
            return

        n_beats = len(beat_matrix)
        if n_beats < 4:
            log.warning("_plot_beat_template: only %d valid beats — skipping", n_beats)
            return

        stride = max(1, n_beats // 60)  # show at most ~60 individual ghost traces

        wt, src_note = beat_template_overlay(self.app.analysis.wave_template,
                                             r.get("intervals"))

        def draw_template(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            # Ghost traces (subsampled for performance)
            for beat in beat_matrix[::stride]:
                ax.plot(beat_time, beat, color=PLOT["grid"], lw=0.3, alpha=0.3)
            if beat_sd is not None:
                ax.fill_between(beat_time, mean_beat - beat_sd, mean_beat + beat_sd,
                                color=BLUE, alpha=0.2, label="±1 SD")
            ax.plot(beat_time, mean_beat, color=BLUE, lw=2.0, label="Mean beat")
            ax.axvline(0, color=RED, lw=1.2, ls="--", alpha=0.8, label="R peak")

            # ── Overlay the wave template windows (beat_template_overlay) ──
            if wt is not None:
                for wkey, (center_ms, half_ms) in wt.landmarks.items():
                    col  = WaveTemplateEditor.WAVE_COLORS.get(wkey, GRAY)
                    name = WaveTemplateEditor.WAVE_SHORT_LABELS.get(wkey, wkey)
                    ax.axvline(center_ms, color=col, lw=1.4, ls=":",
                               alpha=0.85, zorder=6)
                    ax.axvspan(center_ms - half_ms, center_ms + half_ms,
                               alpha=0.08, color=col, zorder=0)
                    # Label at the bottom of the axis
                    ax.text(center_ms, 0.02, name,
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom",
                            color=col, fontsize=9, fontweight="bold", zorder=7)

            ax.set_xlabel("Time relative to R peak (ms)")
            ax.set_ylabel("Amplitude (norm.)")
            ax.set_title(f"Mean template  (n={n_beats} beats){src_note}", loc="left")
            ax.legend(framealpha=0, loc="upper right")

        self.app._slots["beat"].update(draw_template)

        if beat_corr is None or peak_amps is None:
            return
        mean_corr = float(np.nanmean(beat_corr))

        n_bad = int(np.sum(beat_corr < 0.90)) if beat_corr is not None else 0

        def draw_distributions(fig):
            ax_amp, ax_corr = fig.subplots(1, 2)
            for ax in (ax_amp, ax_corr):
                style_axes(ax)
            ax_amp.hist(peak_amps, bins=min(50, max(10, n_beats // 4)),
                        color=BLUE, alpha=0.75, edgecolor="none")
            ax_amp.set_xlabel("R-peak amplitude (norm.)")
            ax_amp.set_ylabel("Count")
            ax_amp.set_title("R-peak amplitude", loc="left")

            bins_corr = min(40, max(10, n_beats // 4))
            ax_corr.hist(beat_corr, bins=bins_corr,
                         color=GREEN, alpha=0.75, edgecolor="none")
            ax_corr.axvline(mean_corr, color=RED, lw=1.5, ls="--",
                            label=f"mean={mean_corr:.3f}")
            ax_corr.axvline(0.90, color=ORANGE, lw=1.0, ls=":",
                            label="0.90 threshold")
            ax_corr.set_xlabel("Correlation with template")
            ax_corr.set_ylabel("Count")
            title = f"Beat Morphology  ({n_bad} beats < 0.90)"
            ax_corr.set_title(title, loc="left",
                              color=(ORANGE if n_bad > n_beats * 0.1 else PLOT["text"]))
            ax_corr.legend(framealpha=0)

        self.app._slots["beat_dist"].update(draw_distributions)

    def plot_summary(self, r: dict) -> None:
        """Populate the Summary tab: signal-quality panel, kept plots, metrics
        table, and text report.
        """
        ivl = r["intervals"]
        rr_ms = r.get("rr_ms", np.array([]))

        # RR-asymmetry indices (shown in the table, the plot and the report).
        # Literature conventions (rr_asymmetry_indices, RR_ASYMMETRY_TEXT):
        # Porta PI = accelerations / off-line points, Guzik GI = deceleration
        # share of the distances to the identity line; symmetry → 50 %.
        # Successive VALID differences only: a ΔRR across an excluded
        # interval (e.g. into/out of a pause) is dropped, not counted.
        rr_a_v, rr_b_v = successive_valid_pairs(
            np.asarray(rr_ms, dtype=float), self._rr_valid(r))
        have_asym = len(rr_a_v) >= 8
        asym_ix: "Optional[dict]" = (rr_asymmetry_indices(rr_a_v, rr_b_v)
                                     if have_asym else None)

        t = build_metric_table(
            r, sig_quality=self.app.detection.sig_quality,
            artifact_report=self.app.analysis.artifact_report,
            window=self._report_window(), signal_end_s=self._signal_end_s(),
            asymmetry=asym_ix)

        # ── Metrics table (values only -- no reference-range judgment) ─────────
        def _metric(key: str, text: str) -> None:
            lbl = self.app._sum_metric_vals.get(key)
            if lbl is not None:
                lbl.configure(text=text)

        if t.has("hr.mean_rr"):
            # 60000 / mean NN -- the app-wide mean-HR definition (hr["mean"])
            _metric("hr_mean", f"{t.value('hr.mean_rr'):.0f}")
            _metric("hr_range",
                    f"{t.value('hr.range_lo'):.0f}–{t.value('hr.range_hi'):.0f}")

        _metric("sdnn",  t.text("hrv_time.HRV_SDNN",  1))
        _metric("rmssd", t.text("hrv_time.HRV_RMSSD", 1))
        _metric("pnn6",  t.text("hrv_time.HRV_pNN6",  1))
        _metric("lf_hf", t.text("hrv_freq.HRV_LFHF", 2))
        _metric("sampen", t.text("hrv_nonlin.HRV_SampEn", 2))
        _metric("dfa1",   t.text("hrv_nonlin.HRV_DFA_alpha1", 2))

        for key, text in summary_interval_metrics(ivl).items():
            _metric(key, text)

        # ── Signal Quality panel ────────────────────────────────────────────────
        # Surfaces numbers the app already computes but never showed anywhere:
        # mean beat-to-template correlation, % of beats below the 0.90
        # threshold, and the artifact-correction breakdown. This is a
        # judgment about DATA TRUSTWORTHINESS, not physiology -- deliberately
        # has no experimental-context-dependent reference ranges.
        beat_corr = r.get("beat_corr")

        def _sq(key: str, text: str) -> None:
            lbl = self.app._sum_quality_vals.get(key)
            if lbl is not None:
                lbl.configure(text=text)

        score = self.app.detection.sig_quality
        _sq("sq_score", f"{score}%" if score is not None else "—")
        # sq_noisy_time: duration-weighted counterpart of sq_badbeats -- see
        # update_kpis() for why a beat-count percentage can understate noise
        # burden on rodent ECG.
        for key, text in quality_tile_texts(t).items():
            _sq(key, text)

        if self.app.lbl_sum_verdict is not None:
            # Analysed duration (window end − start), not the absolute time
            # of the last beat (which also counts the time before the window).
            _dur = t.finite("core.duration_s")
            score_word = "good" if (score or 0) >= 70 else ("fair" if (score or 0) >= 40 else "poor")
            score_color = GREEN if (score or 0) >= 70 else (ORANGE if (score or 0) >= 40 else RED)
            parts = [f"Signal quality {score}% — {score_word}." if score is not None else "Signal quality not computed."]
            if t.value("hr.n_beats", None):
                parts.append(f"{t.value('hr.n_beats')} beats")
            if _dur is not None:
                parts.append(f"{_dur:.0f} s analysed")
            if t.has("artifact.removed"):
                parts.append(f"{t.value('artifact.removed')} beats auto-corrected")
            self.app.lbl_sum_verdict.configure(
                text="  ·  ".join(parts), text_color=score_color)

        # ── Mirror the two kept plots (RR tachogram, Poincare) ─────────────────
        # Every other Summary plot used to mirror a full-tab draw_fn at ~1/3
        # the size with the same font/legend density -- removed in favour of
        # linking to the full tabs instead (see the "Metrics" section below).
        # These two are still worth keeping small, so plot_rr()/plot_nonlinear()
        # stash a "_draw_fn_compact" alongside the normal one (right-hand hint
        # title, MSE subplot, etc. dropped) -- fall back to the full draw_fn
        # if a slot hasn't been (re)built with one yet.
        _MIRRORS = [
            ("rr",            "sum_rr"),
            ("poincare",      "sum_poincare"),
        ]
        for src_key, dst_key in _MIRRORS:
            src = self.app._slots.get(src_key)
            dst = self.app._slots.get(dst_key)
            if src is None or dst is None:
                continue
            fn = getattr(src, "_draw_fn_compact", None) or getattr(src, "_draw_fn", None)
            if fn is not None:
                try:
                    dst.update(fn)
                except Exception as exc:
                    log.debug("sum mirror %s→%s: %s", src_key, dst_key, exc)

        # ── sum_asymmetry: RR asymmetry (Porta / Guzik index) ─────────────────
        if asym_ix is not None:
            diff = rr_b_v - rr_a_v
            n_dec = asym_ix["n_dec"]    # decelerations (RR lengthens)
            n_acc = asym_ix["n_acc"]    # accelerations (RR shortens)
            n_tot = asym_ix["n_tot"]
            porta_pct = asym_ix["porta_pct"]
            guzik_pct = asym_ix["guzik_pct"]
            # ΔRR histogram (signed)
            drr_clip = np.clip(diff, -60, 60)

            _porta_pct  = porta_pct
            _guzik_pct  = guzik_pct
            _n_dec = n_dec; _n_acc = n_acc; _n_tot = n_tot
            _n_off = asym_ix["n_off"]; _n_zero = asym_ix["n_zero"]
            _drr_clip = drr_clip

            def draw_asymmetry(fig):
                ax_bar, ax_hist = fig.subplots(1, 2)
                style_axes(ax_bar); style_axes(ax_hist)

                # Bar chart: shares of the OFF-line pairs (ΔRR ≠ 0), the
                # Porta PI denominator -- the acceleration bar IS the PI and
                # the 50 % line is symmetry. ΔRR = 0 pairs are counted aside.
                cats  = ["Decelerations\n(RR↑)", "Accelerations\n(RR↓)"]
                vals2 = ([_n_dec / _n_off * 100, _n_acc / _n_off * 100]
                         if _n_off else [0.0, 0.0])
                colors2 = [RED, BLUE_DARK]
                bars = ax_bar.bar(cats, vals2, color=colors2, alpha=0.80, width=0.5)
                ax_bar.axhline(50, color=BORDER2, lw=1.2, ls="--", alpha=0.7)
                ax_bar.set_ylabel("% of pairs with ΔRR ≠ 0")
                ax_bar.set_ylim(0, 105)
                for bar, v in zip(bars, vals2):
                    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                                v + 1.5, f"{v:.1f}%" if _n_off else "—",
                                ha="center", va="bottom", fontsize=8, color=PLOT["text"])
                ax_bar.set_xlabel(f"No change (ΔRR = 0): {_n_zero} of {_n_tot} pairs",
                                  fontsize=8)

                def _pct(v):
                    return f"{v:.1f}%" if np.isfinite(v) else "—"
                porta_str = f"Porta PI={_pct(_porta_pct)}  Guzik GI={_pct(_guzik_pct)}"
                ax_bar.set_title(f"RR Asymmetry  ({porta_str})", loc="left", fontsize=8)
                ax_bar.tick_params(axis="x", labelsize=8)

                # ΔRR histogram
                ax_hist.hist(_drr_clip[_drr_clip < 0], bins=30,
                             color=BLUE_DARK, alpha=0.70, label="accelerations")
                ax_hist.hist(_drr_clip[_drr_clip > 0], bins=30,
                             color=RED, alpha=0.70, label="decelerations")
                ax_hist.axvline(0, color=BORDER2, lw=1.2, ls="--")
                ax_hist.set_xlabel("ΔRR (ms)")
                ax_hist.set_ylabel("Beats")
                ax_hist.set_title("ΔRR Distribution", loc="left", fontsize=8)
                ax_hist.legend(framealpha=0, fontsize=8)

            dst = self.app._slots.get("sum_asymmetry")
            if dst is not None:
                dst.update(draw_asymmetry)

        # ── sum_quality_time: Qualité morphologique dans le temps ──────────────
        if beat_corr is not None and len(beat_corr) > 4 and self.app.detection.rpeaks_ok is not None:
            # Times of the beats actually scored (edge beats are not), from
            # the results; the windowed peaks only for older results.
            _rp = None
            if r.get("beat_corr_t_s") is None:
                _wp = self.app._windowed_peaks()
                _rp = (_wp if _wp is not None else self.app.detection.rpeaks_ok
                       ).astype(float) / self.app.signal.fs
            _bc_t, _bc_v = beat_quality_times(r, _rp)
            n_pts = len(_bc_v)

            # Rolling 50-beat mean over the aligned (time, corr) pairs, so
            # the line has exactly one time per value.
            _win = min(50, max(10, n_pts // 20)) if n_pts >= 10 else 0
            if _win >= 2 and n_pts >= _win:
                _kern = np.ones(_win) / _win
                _roll = np.convolve(_bc_v, _kern, mode="valid")
                _roll_t = _bc_t[_win - 1: _win - 1 + len(_roll)]
            else:
                _roll = np.array([])
                _roll_t = np.array([])

            def draw_quality_time(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                if len(_bc_v) == 0:
                    # No (time, corr) pair -- only possible for older
                    # results without beat times when the windowed peaks
                    # are empty (e.g. mid session restore). Nothing valid
                    # to plot this frame.
                    ax.text(0.5, 0.5, "No quality data for the current window",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=9, color=PLOT["muted"])
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Correlation to template")
                    ax.set_title("Morphological quality over time", loc="left")
                    return
                # Scatter individual beats, coloured by quality
                if len(_bc_t) != len(_bc_v):
                    log.debug("quality plot length mismatch: %d vs %d",
                              len(_bc_t), len(_bc_v))
                sc = ax.scatter(_bc_t, _bc_v, s=2, c=_bc_v, cmap="viridis",
                                vmin=0.7, vmax=1.0, alpha=0.35, rasterized=True, zorder=2)
                if len(_roll_t) and len(_roll):
                    ax.plot(_roll_t, _roll, color=BLUE, lw=1.8, zorder=3,
                            label=f"rolling mean (n={_win})")
                ax.axhline(0.90, color=ORANGE, lw=1.0, ls=":", alpha=0.8,
                           label="threshold 0.90")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Correlation to template")
                ax.set_title("Morphological quality over time", loc="left")
                ax.set_ylim(max(0, float(np.nanmin(_bc_v)) - 0.05), 1.02)
                ax.legend(framealpha=0, fontsize=8, loc="lower right")
                try:
                    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02,
                                 label="correlation")
                except Exception as e:
                    log.debug("colorbar render failed: %s", e)

            dst = self.app._slots.get("sum_quality_time")
            if dst is not None:
                dst.update(draw_quality_time)

        # ── Report text (summary_report_text: pure, tested) ──────────────────
        asym = asym_ix
        self.app._set_textbox(self.app.txt_sum, summary_report_text(
            r,
            subject=self.app.ent_subject.get(),
            filepath=self.app.signal.filepath,
            filter_text=recorded_filter_text(
                getattr(self.app.signal, "filter_text", None),
                self.app.signal.no_filter_mode),
            window=self._report_window(),
            signal_end_s=self._signal_end_s(),
            artifact_report=self.app.analysis.artifact_report,
            qtc_formula=self.app._qtc_formula(),
            asymmetry=asym,
            signal_crop=getattr(self.app.signal, "crop", None)))

    def reset_result_plots(self) -> None:
        """Clear stored draw_fn on every result-plot slot.

        Prevents stale draw functions from a previous file replaying
        on window resize after a new file is loaded.
        """
        result_slots = (
            "rr", "rr_hist",
            "poincare", "psd",
            "intervals",
            "beat", "beat_dist",
            "epochs", "rolling_hrv", "autonomic", "event_locked",
            "arr_detail",
            # Summary tab: kept mirrors + its own unique panels
            "sum_rr", "sum_poincare",
            "sum_asymmetry", "sum_quality_time",
        )
        for name in result_slots:
            slot = self.app._slots.get(name)
            if slot is not None:
                slot._draw_fn = None
                slot._show_placeholder()

    def reset_tab_status_labels(self) -> None:
        """Reset per-tab status labels and disable action buttons.

        Called on new file load so labels from the previous analysis
        (e.g. "Done LFn=25.0%") don't persist after loading a new file.
        """
        _neutral = "  Click Analyze first"
        if self.app.lbl_freq_status is not None:
            self.app.lbl_freq_status.configure(text=_neutral, text_color=PLOT["muted"])  # type: ignore[union-attr]
        if self.app.lbl_nonlin_status is not None:
            self.app.lbl_nonlin_status.configure(text=_neutral, text_color=PLOT["muted"])  # type: ignore[union-attr]
        lbl_ivl = getattr(self.app, "lbl_ivl_status", None)
        if lbl_ivl is not None:
            lbl_ivl.configure(text=_neutral, text_color=PLOT["muted"])
        lbl_arr = getattr(self.app, "lbl_arrhythmia_status", None)
        if lbl_arr is not None:
            lbl_arr.configure(text="  Click Analyze first", text_color=PLOT["muted"])
        lbl_roll = getattr(self.app, "lbl_roll_status", None)
        if lbl_roll is not None:
            lbl_roll.configure(text="  Click Analyze first", text_color=PLOT["muted"])
        lbl_auto = getattr(self.app, "lbl_auto_status", None)
        if lbl_auto is not None:
            lbl_auto.configure(text=self.app.analysis_ctrl.AUTONOMIC_IDLE_TEXT,
                               text_color=PLOT["muted"])
        self.app._set_result_btns_enabled(
            False, ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl", "btn_run_arrhythmia"))

    def reset_kpis(self) -> None:
        """Reset all KPI labels to dash when results are invalidated."""
        for key in ("hr_mean", "hr_range", "rr_mean", "n_beats",
                    "sdnn", "rmssd", "pnn6", "dur",
                    "sq_score", "sq_corr", "sq_badbeats", "sq_noisy_time",
                    "sq_artifact"):
            widget = self.app._kpi.get(key)
            if widget is not None:
                widget.configure(text="--")
        for key in ("hr_mean", "n_beats", "dur"):
            widget = self.app._topbar_vals.get(key)
            if widget is not None:
                widget.configure(text="—")

    def update_kpis(self) -> None:
        if self.app.analysis.results is None:
            return
        t = build_metric_table(
            self.app.analysis.results, sig_quality=self.app.detection.sig_quality,
            artifact_report=self.app.analysis.artifact_report,
            window=self._report_window(), signal_end_s=self._signal_end_s(),
            asymmetry=None)

        # Value text is bare (no unit suffix) -- units are baked into the
        # stat-panel tiles once at construction (make_stat_tile's unit=),
        # so the value stays the visually-prominent element per the redesign
        # brief ("valeurs numériques mises en évidence, unités discrètes").
        self.app._kpi["hr_mean"].configure(text=t.raw_text("hr.mean", 0))
        self.app._kpi["hr_range"].configure(
            text=f"{t.raw_text('hr.min', 0)}–{t.raw_text('hr.max', 0)}")
        self.app._kpi["rr_mean"].configure(text=t.raw_text("rr.mean", 0))
        self.app._kpi["n_beats"].configure(text=str(t.value("hr.n_valid", "—")))
        self.app._kpi["sdnn"].configure(text=t.raw_text("hrv_time.HRV_SDNN", 1))
        self.app._kpi["rmssd"].configure(text=t.raw_text("hrv_time.HRV_RMSSD", 1))
        self.app._kpi["pnn6"].configure(text=t.raw_text("hrv_time.HRV_pNN6", 1))
        # Analysed duration (window end − start) -- not rdf's last Time_s,
        # the absolute time of the last beat.
        self.app._kpi["dur"].configure(text=t.raw_text("core.duration_s", 0))

        # Top-bar compact mirrors of the 3 values shown there (§1 of the
        # Phase 1 plan) -- same source numbers as the tiles above, just a
        # second, smaller display in the full-width top bar.
        for key, widget in self.app._topbar_vals.items():
            src = self.app._kpi.get(key)
            if src is not None:
                widget.configure(text=src.cget("text"))

        # ── Signal Quality tiles ────────────────────────────────────────────
        # Same texts as plot_summary()'s Signal Quality panel.
        score = self.app.detection.sig_quality
        self.app._kpi["sq_score"].configure(text=f"{score}" if score is not None else "—")
        # sq_noisy_time: duration-weighted counterpart to sq_badbeats: rodent ECG is prone
        # to EMG contamination from near-constant movement/grooming, so a
        # beat-count percentage alone can understate how much of the actual
        # recording TIME is unreliable (a corrupted stretch may have very
        # few beats detected at all). Each RR interval's duration goes to
        # the quality of the scored beat it starts at (noisy_time_pct).
        for key, text in quality_tile_texts(t).items():
            self.app._kpi[key].configure(text=text)

        # Repaint the gauge too -- covers the case where update_kpis() is the
        # only thing that ran (e.g. _rebuild_ui()'s trailing call, where the
        # gauge widget was just destroyed/recreated but sig_quality survived
        # as plain data). update_signal_quality() also calls this directly
        # for the real-time-computation path (detection_controller.py).
        if self.app.quality_gauge is not None:
            update_quality_gauge(self.app.quality_gauge, score)
