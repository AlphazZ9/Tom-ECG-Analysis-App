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
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import welch as _scipy_welch

# numpy 2.x renamed trapz to trapezoid. Own this locally rather than relying
# on app.py's module-level np.trapz = np.trapezoid shim having already run --
# a comment here used to (incorrectly) claim "the shim at the top of this
# module ensures np.trapz always exists", but no such shim was ever defined
# in this file; it only worked because app.py happens to always be imported
# before PlotController is ever instantiated.
_trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid", None)

from ecg.core.models import MouseECG
from ecg.core.filtering import downsample_for_display, downsample_envelope
from ecg.ui.plots import style_axes
from ecg.ui.wave_editor import WaveTemplateEditor
from ecg.ui.theme import (
    PLOT, RED, AMBER, ORANGE, ORANGE_DARK, ORANGE_DEEP,
    CYAN, BLUE, BLUE_DARK, BLUE_MID, PURPLE, TEAL, TEAL_DARK,
    BORDER2, RED_MID, MUTED, GRAY, GRAY_LIGHT, NAVY,
    GREEN, GREEN_DARK, PINK, AMBER_DARK, ARTIFACT_TYPE_COLOR,
)
from ecg.ui.widgets import update_quality_gauge

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")

_OVERVIEW_MAX_POINTS = 2_000


class PlotController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

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
                    label="ECG filtré")

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
                           marker="o", label="Acceptés")
            if mask_added.any():
                ax.scatter(rp_added[mask_added] / fs, sig_flt[rp_added[mask_added]],
                           color=CYAN, s=140, zorder=7,
                           marker="*", linewidths=1.2, edgecolors="#006064",
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


    def _ensure_overview_cache(self) -> bool:
        """Lazily compute+cache the minimap's envelope arrays.

        Filtered signal only -- the minimap's job is "where am I in the
        recording," not raw-vs-filtered comparison; that toggle stays
        detail-plot-only. Returns False if there's no filtered signal yet.
        """
        time, sig_flt = self.app.signal.time, self.app.signal.filtered
        if time is None or sig_flt is None:
            return False
        if self.app.ui.ds_time is None or self.app.ui.ds_sig is None:
            mins, maxs = downsample_envelope(sig_flt, max_points=_OVERVIEW_MAX_POINTS)
            self.app.ui.ds_sig     = mins
            self.app.ui.ds_sig_max = maxs
            self.app.ui.ds_sig_mid = (mins + maxs) / 2.0
            self.app.ui.ds_time    = np.linspace(float(time[0]), float(time[-1]), len(mins))
        return True

    def draw_overview(self) -> None:
        """Full-recording minimap strip above the detail plot.

        Filled min/max envelope band (cached, filtered signal only),
        accepted R-peaks only (no rejected/excluded/added markers -- keeps
        this compact strip uncluttered), and a "current window" highlight
        recomputed fresh from live ui.nav_pos/ent_window on every call
        (never cached, so drag/scrub always reflects the true position).
        Click/drag navigation is wired by NavigationController.on_overview_*,
        not here -- this method only renders. No-ops gracefully if no
        signal is loaded yet.
        """
        if not self._ensure_overview_cache():
            return

        time, fs, sig_flt = self.app.signal.time, self.app.signal.fs, self.app.signal.filtered
        t_ds, lo_ds, hi_ds, mid_ds = (self.app.ui.ds_time, self.app.ui.ds_sig,
                                       self.app.ui.ds_sig_max, self.app.ui.ds_sig_mid)
        rp_ok = self.app.detection.rpeaks_ok if self.app.detection.rpeaks_ok is not None else np.array([])
        t_amp = self.app.detection.thresh_amp
        t_max = float(time[-1])

        try:
            win = float(self.app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except Exception:
            win = 10.0
        t_start = self.app.ui.nav_pos
        t_end   = min(t_max, t_start + win)

        def draw(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            ax.fill_between(t_ds, lo_ds, hi_ds, color=PLOT["signal"], alpha=0.55, lw=0, zorder=2)
            ax.plot(t_ds, mid_ds, color=PLOT["signal"], lw=0.5, alpha=0.35, zorder=3)
            if len(rp_ok):
                ax.scatter(rp_ok / fs, sig_flt[rp_ok], color=PLOT["rpeak_ok"],
                           s=6, zorder=5, alpha=0.6, linewidths=0)
            ax.axhline(t_amp, color=PLOT["threshold"], lw=0.8, ls="--", alpha=0.5, zorder=1)

            # Current-window indicator -- the Audacity/Premiere "viewport" box.
            ax.axvspan(t_start, t_end, color=ORANGE, alpha=0.16, zorder=6, linewidth=0)
            ax.axvline(t_start, color=ORANGE, lw=1.0, alpha=0.8, zorder=7)
            ax.axvline(t_end, color=ORANGE, lw=1.0, alpha=0.8, zorder=7)

            ax.set_xlim(float(time[0]), t_max)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=7, colors=PLOT["muted"])
            ax.margins(x=0, y=0.05)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)

        self.app._slots["overview"].update(draw)

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

        primary_sig   = sig_raw   if show_raw else sig_flt
        primary_color = PLOT["raw"]    if show_raw else PLOT["signal"]
        ghost_sig     = sig_flt   if show_raw else sig_raw
        ghost_color   = PLOT["signal"] if show_raw else PLOT["raw"]
        if raw_only:
            label_mode = "Raw (not yet analysed)"
        elif no_filter_mode:
            label_mode = "Unfiltered" if not show_raw else "Pre-norm baseline"
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
            # Accepted peaks (green dots)
            if mask_ok.any() and sig_flt is not None:
                ax.scatter(rp_ok[mask_ok] / fs, sig_flt[rp_ok[mask_ok]],
                           color=PLOT["rpeak_ok"], s=55, zorder=5,
                           marker="o", label="Accepted")
            # Manually added peaks (cyan star — rendered on top of everything)
            if mask_added.any() and sig_flt is not None:
                ax.scatter(rp_added[mask_added] / fs, sig_flt[rp_added[mask_added]],
                           color=CYAN, s=140, zorder=7,
                           marker="*", linewidths=1.2, edgecolors="#006064",
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
                           label=f"Threshold ({t_amp:.3f})")
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
                    "  ·  click '1 ▶ Preview Detection' to filter & detect",
                    loc="left", color=title_color)
            else:
                ax.set_title(
                    f"Detail  {t_start:.1f}–{t_end:.1f} s  ·  {n_in_view} peaks"
                    f"  ·  {label_mode}{filter_tag}{inverted_tag}{title_suffix}",
                    loc="left", color=title_color)
            ax.legend(framealpha=0, loc="upper right")

        self.app._slots["detail"].update(draw)

    def run_plot_chain(
        self,
        tasks: list,
        on_complete: "Optional[Callable[[], None]]" = None,
        auto_epochs: bool = False,
    ) -> None:
        """Run a list of (label, fn) plot tasks sequentially via after() chain."""
        total = len(tasks)

        def _run_next(idx: int) -> None:
            if idx >= total:
                self.app._set_progress(100, "Done")
                if on_complete:
                    on_complete()
                if auto_epochs:
                    self.app.after(200, self.app._compute_epochs)
                return
            label, fn = tasks[idx]
            pct = int(100 * (idx + 1) / total)
            self.app._set_progress(pct, f"Rendering {label}…")
            try:
                fn()
            except Exception:
                log.exception("Plot task '%s' failed", label)
            self.app.after(25, lambda i=idx + 1: _run_next(i))

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
            ("HRV radar",             lambda: self.plot_radar(results)),
            ("ECG intervals ECG preview", lambda: self.plot_intervals_ecg(results)),
            ("ECG intervals",         lambda: self.plot_intervals(results)),
            ("Beat template",         lambda: self.plot_beat_template(results)),
            ("Summary",               lambda: self.plot_summary(results)),
        ]
        self.run_plot_chain(tasks, on_complete=on_complete, auto_epochs=auto_epochs)

    def plot_rr(self, r: dict) -> None:
        """Plot RR tachogram, HR trace, and RR distribution histogram.

        Drastic RR changes are detected and shown as orange/red markers on
        the tachogram.  Clicking any point navigates to that beat in Detection.
        Right-clicking jumps specifically to the nearest spike.
        """
        import datetime as _dt
        rdf       = r["rr_df"]
        rr_ms_raw = r.get("rr_ms", np.array([]))

        # Fall back to raw RR data if filtered dataframe is empty
        if rdf.empty and len(rr_ms_raw) > 1 and self.app.detection.rpeaks_ok is not None:
            _wp_rr = self.app._windowed_peaks()
            rdf = pd.DataFrame({
                "Time_s": (_wp_rr if _wp_rr is not None else self.app.detection.rpeaks_ok)[1:len(rr_ms_raw) + 1] / self.app.signal.fs,
                "RR_ms":  rr_ms_raw,
                "HR_bpm": 60_000.0 / np.clip(rr_ms_raw, 1, None),
            })
        if rdf.empty:
            log.warning("_plot_rr: empty rdf — skipping")
            return

        t_all  = np.asarray(rdf["Time_s"].values, dtype=float)
        rr_all = np.asarray(rdf["RR_ms"].values,  dtype=float)
        hr_all = np.asarray(rdf["HR_bpm"].values,  dtype=float)

        # ── Detect drastic RR changes ──────────────────────────────────────
        # A beat is a "spike" if its RR deviates more than spike_thr standard
        # deviations from the local rolling median (window = 15 beats).
        spike_thr  = 2.5   # SD threshold
        spike_idx  = np.array([], dtype=int)
        spike_t    = np.array([], dtype=float)
        spike_rr   = np.array([], dtype=float)
        spike_mag  = np.array([], dtype=float)  # delta-RR in ms

        if len(rr_all) >= 10:
            # Rolling median via scipy — O(n·log(w)) au lieu de O(n·w)
            from scipy.ndimage import median_filter as _median_filter
            win = min(15, len(rr_all) // 2)
            roll_med = _median_filter(rr_all.astype(float), size=win, mode="nearest")
            delta = rr_all - roll_med
            # Also flag beats where consecutive delta-RR is extreme
            drr = np.diff(rr_all)
            drr_padded = np.concatenate([[0], drr])
            rr_sd = max(float(rr_all.std()), 1.0)
            # Spike = large deviation from local median OR large consecutive jump
            spike_mask = (np.abs(delta) > spike_thr * rr_sd) | \
                         (np.abs(drr_padded) > spike_thr * rr_sd * 1.2)
            spike_idx = np.where(spike_mask)[0]
            if len(spike_idx):
                spike_t   = t_all[spike_idx]
                spike_rr  = rr_all[spike_idx]
                spike_mag = delta[spike_idx]

        # Downsample for display
        t_ds   = downsample_for_display(t_all)
        rr_ds  = downsample_for_display(rr_all)
        hr_ds  = downsample_for_display(hr_all)
        rr_mean = float(rr_all.mean())
        hr_mean = float(hr_all.mean())
        rr_sd_v = float(rr_all.std())
        c_rr, c_hr = "#388E3C", ORANGE_DARK
        n_spikes   = len(spike_idx)
        updated_at = _dt.datetime.now().strftime("%H:%M:%S")

        # Spike colours: orange = moderate, red = severe
        def _spike_color(mag: float) -> str:
            return RED if abs(mag) > 3.5 * rr_sd_v else AMBER

        def draw_tachogram(fig):
            axes = fig.subplots(2, 1, sharex=True)
            for ax in axes:
                style_axes(ax)

            # ── RR tachogram ────────────────────────────────────
            axes[0].plot(t_ds, rr_ds, color=c_rr, lw=0.8, zorder=2)
            axes[0].axhline(rr_mean, color=c_rr, ls="--", lw=0.9, alpha=0.5, zorder=1)

            # ±1 SD reference band
            axes[0].axhspan(rr_mean - rr_sd_v, rr_mean + rr_sd_v,
                            alpha=0.06, color=c_rr, zorder=0)
            axes[0].axhline(rr_mean + spike_thr * rr_sd_v,
                            color=AMBER, lw=0.6, ls=":", alpha=0.5, zorder=1)
            axes[0].axhline(rr_mean - spike_thr * rr_sd_v,
                            color=AMBER, lw=0.6, ls=":", alpha=0.5, zorder=1)

            # Spike markers
            if len(spike_t):
                for st, sr, sm in zip(spike_t, spike_rr, spike_mag):
                    col = _spike_color(sm)
                    axes[0].scatter([st], [sr], s=55, color=col,
                                    marker="v" if sm < 0 else "^",
                                    zorder=5, edgecolors="white",
                                    linewidths=0.6, alpha=0.9)

            spike_note = (f"  ·  {n_spikes} spike{'s' if n_spikes != 1 else ''} détecté{'s' if n_spikes != 1 else ''}"
                          if n_spikes else "")
            axes[0].set_ylabel("RR (ms)")
            axes[0].set_title(
                f"RR Intervals  ·  moy. {rr_mean:.1f} ms  ·  SD {rr_sd_v:.1f} ms{spike_note}",
                loc="left", fontsize=9)
            axes[0].set_title(f"click=navigate  r-click=next spike  ·  {updated_at}",
                              loc="right", fontsize=7,
                              color=PLOT.get("muted", "#666"))
            axes[0].tick_params(labelbottom=False)

            # ── HR trace ────────────────────────────────────────
            axes[1].plot(t_ds, hr_ds, color=c_hr, lw=0.8, zorder=2)
            axes[1].axhline(hr_mean, color=c_hr, ls="--", lw=0.9, alpha=0.5, zorder=1)

            # Mirror spikes on HR axis
            if len(spike_t):
                spike_hr = 60_000.0 / np.clip(spike_rr, 1, None)
                for st, shr, sm in zip(spike_t, spike_hr, spike_mag):
                    col = _spike_color(sm)
                    axes[1].scatter([st], [shr], s=45, color=col,
                                    marker="v" if sm < 0 else "^",
                                    zorder=5, edgecolors="white",
                                    linewidths=0.6, alpha=0.9)

            axes[1].set_ylabel("HR (bpm)")
            axes[1].set_xlabel(
                "Time (s)  —  ▲ accélération soudaine  ▼ décélération soudaine  (seuil ±2.5 SD)")
            axes[1].set_title(f"Instantaneous HR  ·  moy. {hr_mean:.0f} bpm",
                              loc="left", fontsize=9)

        # Capture annotations for closure
        _anns = list(self.app.analysis.annotations)
        _draw_fn_orig = draw_tachogram

        def draw_tachogram_annotated(fig):
            _draw_fn_orig(fig)
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

        # ── Wire click-to-navigate ─────────────────────────────────────────
        if self.app.ui.rr_click_cid is not None:
            try:
                self.app._slots["rr"].canvas.mpl_disconnect(self.app.ui.rr_click_cid)
            except Exception as e:
                log.debug("mpl_disconnect (rr click) failed: %s", e)

        # Store spike times for right-click navigation
        _spike_times = spike_t.copy() if len(spike_t) else np.array([], dtype=float)

        def _on_rr_click(event):
            if event.xdata is None or event.inaxes is None:
                return
            t_clicked = float(event.xdata)

            if event.button == 3 and len(_spike_times):
                # Right-click: jump to the nearest spike
                dists = np.abs(_spike_times - t_clicked)
                nearest_spike_t = float(_spike_times[int(np.argmin(dists))])
                t_nav = nearest_spike_t
                spike_info = f"spike à {nearest_spike_t:.3f} s"
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
                self.app.tabs.set("Detection")
            except Exception as e:
                log.debug("tabs.set Detection failed: %s", e)
            self.app._draw_detail()
            if spike_info:
                self.app._set_status(f"Navigation → {spike_info}", AMBER)

        self.app.ui.rr_click_cid = self.app._slots["rr"].canvas.mpl_connect(
            "button_press_event", _on_rr_click)
        _rr_desc = rdf["RR_ms"].describe()
        _rr_tsv  = "Metric\tRR_ms\n" + "\n".join(
            f"{k}\t{v:.5g}" for k, v in _rr_desc.items())
        self.app._set_textbox(self.app.txt_rr,
            "\n".join(f"  {k:<14} {v:>10.2f}" for k, v in _rr_desc.items()),
            tsv=_rr_tsv)

        rr_clipped = rdf["RR_ms"].clip(MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS).values

        def draw_histogram(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            ax.hist(rr_clipped, bins=50, color=c_rr, alpha=0.7,
                    edgecolor="white", lw=0.3)
            ax.set_xlabel("RR (ms)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution RR", loc="left")
            # A fixed 1 ms tick spacing only looks reasonable for a narrow RR
            # range (e.g. a quiet anesthetized recording). Any recording with
            # real variability -- an awake mouse, or one containing a
            # bradycardic/pause episode -- can span 100+ ms, which crams 100+
            # overlapping tick labels onto the axis. MaxNLocator adapts the
            # tick count to whatever range this recording actually has.
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10))

        self.app._slots["rr_hist"].update(draw_histogram)

    def plot_hrv_tables(self, r: dict) -> None:
        """Populate time-domain and frequency-domain HRV text boxes."""
        self.app._set_textbox(self.app.txt_td, self.app._df_to_text(r["hrv_time"]),
                          tsv=self.app._df_to_tsv(r["hrv_time"]))

        fd_df = r["hrv_freq"]
        if fd_df is None or fd_df.empty:
            self.app._set_textbox(self.app.txt_fd, "  (not computed)")
            return

        lines: list[str] = []
        for col in fd_df.columns:
            try:
                v    = float(fd_df[col].values[0])
                name = col.replace("HRV_", "")
                if not np.isfinite(v):
                    continue
                if name in ("LFn", "HFn"):
                    # The actual normalized-units fractions (LF/(LF+HF) etc.) --
                    # these are the real "% of power" figures.
                    lines.append(f"  {name:<26} {v * 100:>10.1f} %")
                elif name in ("LF", "HF", "VLF", "ULF", "VHF", "TP"):
                    # Raw band power, NOT a fraction of anything. Previously
                    # LF/HF/VLF were multiplied by 100 and labelled "%" here --
                    # that showed a meaningless number (raw power x100) right
                    # next to LFn/HFn, which are the real normalized percentages
                    # and used to be shown unscaled as a bare decimal instead.
                    lines.append(f"  {name + ' power':<26} {v:>10.4f}")
                elif name == "LFHF":
                    lines.append(f"  {'LF/HF ratio':<26} {v:>10.3f}")
                elif name in ("LF_peak", "HF_peak", "VLF_peak"):
                    lines.append(f"  {name + ' (Hz)':<26} {v:>10.4f}")
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

    def plot_psd(self, r: dict) -> None:
        """Welch PSD with mouse-specific VLF / LF / HF band shading.

        RR intervals are resampled to a uniform time grid using a cubic spline
        before computing the Welch periodogram.  Cubic (vs linear) resampling
        preserves spectral shape and avoids the artificial high-frequency power
        that linear interpolation introduces.

        Mouse-specific design choices
        ─────────────────────────────
        • Interpolation rate: 20 Hz  (Nyquist = 10 Hz >> HF ceiling of 5 Hz)
        • nperseg: aims for ≥ 0.02 Hz resolution — enough to separate
          VLF (0–0.4), LF (0.4–1.5) and HF (1.5–5.0) bands cleanly.
          Formula: nperseg = max(256, min(fs_interp / 0.02, N // 2))
          e.g. 20 / 0.02 = 1000, so for long recordings nperseg = 1000.
        • noverlap: 75 % of nperseg (Welch variance reduction)
        • Window: Hann (default scipy) — good sidelobe suppression
        """
        rr_ms = r["rr_ms"]
        MIN_BEATS = 60
        if len(rr_ms) < MIN_BEATS:
            log.warning("_plot_psd: too few RR intervals (%d, need ≥ %d)", len(rr_ms), MIN_BEATS)
            def draw_warn(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                ax.text(0.5, 0.5,
                        f"Spectral HRV requires ≥ {MIN_BEATS} beats\n"
                        f"(recording has {len(rr_ms)})",
                        ha="center", va="center", color=PLOT["muted"],
                        transform=ax.transAxes, fontsize=11)
                ax.set_axis_off()
            self.app._slots["psd"].update(draw_warn)
            return
        try:
            # Build a uniformly sampled RR series via cubic spline
            ts    = np.cumsum(rr_ms) / 1000.0          # cumulative time in seconds
            dt    = 1.0 / MouseECG.PSD_INTERP_FS
            t_uni = np.arange(ts[0], ts[-1], dt)
            if len(t_uni) < 32:
                log.warning("_plot_psd: interpolated series too short (%d pts)", len(t_uni))
                return

            cs        = CubicSpline(ts, rr_ms)
            rr_interp = cs(t_uni)
            N         = len(rr_interp)

            # nperseg chosen for ≥ 0.02 Hz resolution (resolves LF band floor at 0.4 Hz cleanly)
            # Minimum 256, maximum N//2
            target_res_hz = 0.020
            nperseg = int(np.clip(
                MouseECG.PSD_INTERP_FS / target_res_hz,
                256, N // 2,
            ))
            noverlap = int(nperseg * 0.75)

            freqs, psd = _scipy_welch(
                rr_interp - rr_interp.mean(),
                fs=MouseECG.PSD_INTERP_FS,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
                scaling="density",
            )

            # Mouse-specific band definitions (Thireau 2008)
            bands = [
                (MouseECG.VLF[0], MouseECG.VLF[1], "VLF",
                 f"VLF  {MouseECG.VLF[0]}–{MouseECG.VLF[1]} Hz", BLUE_DARK),
                (MouseECG.LF[0],  MouseECG.LF[1],  "LF",
                 f"LF   {MouseECG.LF[0]}–{MouseECG.LF[1]} Hz  (baroreflex)", PURPLE),
                (MouseECG.HF[0],  MouseECG.HF[1],  "HF",
                 f"HF   {MouseECG.HF[0]}–{MouseECG.HF[1]} Hz  (respiratory)", "#1B5E20"),
            ]

            # Compute per-band power for annotation
            def _band_power(lo: float, hi: float) -> float:
                m = (freqs >= lo) & (freqs <= hi)
                return float(_trapz(psd[m], freqs[m])) if m.any() else 0.0

            band_powers = {name: _band_power(lo, hi) for lo, hi, name, _, _ in bands}
            total_power = sum(band_powers.values()) + 1e-12

            # Percentages shown in the legend: prefer the canonical values already
            # computed by analyse_hrv_freq() (via nk.hrv_frequency, the same source
            # the KPI bar/export/radar read) over recomputing our own from this
            # plot's independent Welch pipeline. The two pipelines use different
            # interpolation/windowing and can give visibly different numbers for
            # the same recording (e.g. 71.5% vs 72.9% LF in testing) -- showing
            # two different "LF%" figures in the same UI is confusing regardless
            # of which is more "correct". Only fall back to this plot's own
            # band_powers if hrv_freq wasn't computed.
            fd_df = r.get("hrv_freq")
            band_pct: "dict[str, float]" = {}
            if fd_df is not None and not fd_df.empty and "HRV_TP" in fd_df.columns:
                try:
                    tp = float(fd_df["HRV_TP"].values[0])
                    if tp > 1e-12:
                        for key in ("VLF", "LF", "HF"):
                            col = f"HRV_{key}"
                            if col in fd_df.columns:
                                band_pct[key] = float(fd_df[col].values[0]) / tp * 100
                except Exception as exc:
                    log.debug("_plot_psd: could not read hrv_freq percentages: %s", exc)

            # Frequency resolution actually achieved
            df = freqs[1] - freqs[0] if len(freqs) > 1 else float("nan")

            def draw_psd(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                ax.semilogy(freqs, psd, color="#546E7A", lw=1.0, zorder=3)
                for lo, hi, name, legend_label, color in bands:
                    m = (freqs >= lo) & (freqs <= hi)
                    pct = band_pct.get(name, band_powers[name] / total_power * 100)
                    ax.fill_between(freqs, psd, where=m, alpha=0.35, color=color,
                                    label=f"{legend_label}  ({pct:.1f}%)", zorder=2)
                    # Vertical band boundary lines
                    for boundary in (lo, hi):
                        if 0 < boundary < MouseECG.PSD_XLIM:
                            ax.axvline(boundary, color=color, lw=0.8, ls=":",
                                       alpha=0.6, zorder=1)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("PSD (ms²/Hz)")
                ax.set_xlim(0, MouseECG.PSD_XLIM)
                ax.legend(framealpha=0, loc="upper right", fontsize=9)
                ax.set_title(
                    f"Power spectral density  (Welch · Δf={df:.3f} Hz · "
                    f"n={N} pts)",
                    loc="left",
                )

            self.app._slots["psd"].update(draw_psd)
        except Exception as exc:
            log.warning("_plot_psd failed: %s", exc)

    def plot_radar(self, r: dict) -> None:
        """Normalised HRV spider / radar chart."""
        try:
            metrics: dict[str, float] = {}
            for df, keys in [
                (r["hrv_time"],   ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN6"]),
                (r["hrv_freq"],   ["HRV_LF",   "HRV_HF",    "HRV_LFHF"]),
                (r["hrv_nonlin"], ["HRV_SD1",  "HRV_SD2",   "HRV_SampEn"]),
            ]:
                if df is None or df.empty:
                    continue
                for k in keys:
                    if k not in df.columns:
                        continue
                    try:
                        v = float(df[k].values[0])
                        if np.isfinite(v):
                            metrics[k.replace("HRV_", "")] = v
                    except Exception as exc:
                        log.debug("_plot_radar skip '%s': %s", k, exc)

            if len(metrics) < 3:
                return

            labels   = list(metrics.keys())
            values   = np.array(list(metrics.values()))
            v_range  = values.max() - values.min()
            v_norm   = (values - values.min()) / (v_range + 1e-9)
            n        = len(labels)
            angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            v_closed = v_norm.tolist() + [v_norm[0]]
            a_closed = angles + angles[:1]

            # Rich labels: name + value — placed by thetagrids (stays inside bounds)
            rich_labels = [f"{lbl}\n{val:.3g}" for lbl, val in zip(labels, values)]

            def draw_radar(fig):
                ax = fig.add_subplot(111, polar=True)
                ax.set_facecolor(PLOT["axes"])
                ax.plot(a_closed, v_closed, color=RED, lw=2)
                ax.fill(a_closed, v_closed, color=RED, alpha=0.15)
                ax.set_thetagrids(np.degrees(angles), rich_labels, color=PLOT["text"])
                ax.set_ylim(0, 1)
                ax.set_yticks([0.25, 0.5, 0.75])
                ax.set_yticklabels(["25%", "50%", "75%"], color=PLOT["muted"])
                ax.grid(color=PLOT["grid"], alpha=0.5)
                ax.spines["polar"].set_color(PLOT["border"])
                ax.set_title("HRV Profile (normalised)", pad=8, color=PLOT["text"])

            self.app._slots["radar"].update(draw_radar)
        except Exception as exc:
            log.warning("_plot_radar failed: %s", exc)

    def plot_nonlinear(self, r: dict) -> None:
        """Poincaré plot and non-linear HRV metric table."""
        self.app._set_textbox(self.app.txt_nl, self.app._df_to_text(r["hrv_nonlin"]),
                          tsv=self.app._df_to_tsv(r["hrv_nonlin"]))

        rr_ms = r["rr_ms"]
        if len(rr_ms) < 2:
            return

        nl   = r["hrv_nonlin"]
        sd1  = self.app._safe_df_val(nl, "HRV_SD1", 1)
        sd2  = self.app._safe_df_val(nl, "HRV_SD2", 1)
        rr_a = rr_ms[:-1]
        rr_b = rr_ms[1:]
        lim  = [float(rr_ms.min()) - 20, float(rr_ms.max()) + 20]

        def draw_poincare(fig):
            ax = fig.add_subplot(111)
            style_axes(ax)
            ax.scatter(rr_a, rr_b, s=3, alpha=0.25, color=BLUE, rasterized=True)
            ax.plot(lim, lim, color=BORDER2, lw=1, ls="--", alpha=0.7)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            # Don't use set_aspect("equal") — it creates dead whitespace when
            # the container isn't square. Force equal axes via xlim/ylim instead.
            ax.set_xlabel("RR_n (ms)")
            ax.set_ylabel("RR_n+1 (ms)")
            ax.set_title(f"Poincaré diagram  SD1={sd1}  SD2={sd2}", loc="left")

        self.app._slots["poincare"].update(draw_poincare)

    def plot_intervals_ecg(self, r: dict) -> None:
        """ECG beat strip annotated with P / Q / R / S / T landmarks.

        Design
        ------
        • X-axis is relative time from R peak (ms) — always centred at 0.
        • 3 beats are selected with the most complete wave annotation.
        • Plus a 4th "anatomy" reference panel on the right.
        • R_peak_s is read directly from the DataFrame (no index-mapping guesses).
        """
        ivl = r.get("intervals")
        sig = self.app.signal.filtered
        fs  = self.app.signal.fs

        slot = self.app._slots.get("intervals_ecg")
        if slot is None:
            return

        # Guard: need at least a signal and some delineated beats with R_peak_s
        has_data = (
            ivl is not None
            and not ivl.empty
            and sig is not None
            and fs is not None
            and "R_peak_s" in ivl.columns
        )

        if not has_data:
            def draw_hint(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                ax.text(0.5, 0.5, "Run Interval Delineation to see annotated beats",
                        ha="center", va="center", color=PLOT["muted"],
                        transform=ax.transAxes, fontsize=11)
                ax.set_axis_off()
            slot.update(draw_hint)
            return

        # has_data is True → narrow types for static checkers
        assert ivl is not None
        assert sig is not None
        assert fs is not None

        # ── Select up to 3 beats with the most complete wave annotation ──
        WAVE_POS_COLS = ["P_peak_s", "Q_peak_s", "S_peak_s", "T_peak_s"]
        available = [c for c in WAVE_POS_COLS if c in ivl.columns]

        if available:
            completeness = ivl[available].notna().sum(axis=1)
            # Sort by completeness desc, then by proximity to median RR
            med_rr = float(ivl["RR_ms"].median()) if "RR_ms" in ivl.columns else 100.0
            rr_dist = (ivl["RR_ms"] - med_rr).abs() if "RR_ms" in ivl.columns else pd.Series(0, index=ivl.index)
            sort_key = -completeness * 10 + rr_dist.fillna(999)
            best_order = sort_key.argsort()
            picks = list(best_order[:3])
        else:
            # No wave positions at all — just pick 3 beats near median RR
            picks = list(range(min(3, len(ivl))))

        if not picks:
            slot.update(lambda fig: None)
            return

        # ── Beat window: -PRE_MS to +POST_MS relative to R peak ────────
        PRE_MS  = 65    # ms before R  (covers P onset at ~-55ms)
        POST_MS = 110   # ms after  R  (covers T offset at ~+90ms in mice)

        row_data = ivl.iloc[picks]

        def draw_ecg_preview(fig):
            # Capture colors in local scope to ensure availability in draw function
            purple_col = PURPLE
            teal_col = TEAL
            orange_deep_col = ORANGE_DEEP
            
            # Colour / marker per wave landmark
            landmark_style = {
                "P_peak_s":   dict(color="#1A56DB", marker="^",  ms=8,  mew=1.0, label="P"),
                "Q_peak_s":   dict(color=purple_col, marker="v",  ms=8,  mew=1.0, label="Q"),
                "S_peak_s":   dict(color=purple_col, marker="v",  ms=8,  mew=1.0, label="S"),
                "J_peak_s":   dict(color=teal_col, marker="^",  ms=7,  mew=1.0, label="J"),
                "T_peak_s":   dict(color="#D84315", marker="^",  ms=8,  mew=1.0, label="T"),
            }

            # Interval spans: (start_col, end_col, colour, label)
            # None means use R=0 ms as boundary
            span_defs = [
                ("P_peak_s",  "Q_peak_s",  "#1A56DB", "P"),    # P peak → Q
                ("Q_peak_s",  "S_peak_s",  purple_col, "QRS"),  # QRS complex
                (None,        "T_peak_s",  orange_deep_col, "QT"),   # R → T peak
            ]
            # 3 beat columns + 1 anatomy column
            n_beats = len(picks)
            from matplotlib.gridspec import GridSpec
            ratios = [1] * n_beats + [0.55]
            # This figure now arrives from CanvasSlot with constrained_layout
            # active, which would recompute/override the explicit margins
            # below -- opt this one figure out so they stay pixel-identical.
            try:
                fig.set_layout_engine(None)
            except Exception as exc:
                log.debug("draw_ecg_preview: set_layout_engine(None) failed: %s", exc)
            gs = GridSpec(1, n_beats + 1, figure=fig,
                          width_ratios=ratios,
                          left=0.06, right=0.98, top=0.88, bottom=0.12,
                          wspace=0.30)
            axes = [fig.add_subplot(gs[0, i]) for i in range(n_beats + 1)]

            for ax_idx, (_, row) in enumerate(row_data.iterrows()):
                ax = axes[ax_idx]
                r_t = float(row.get("R_peak_s", float("nan")))
                if not np.isfinite(r_t):
                    ax.set_axis_off()
                    continue

                r_samp = int(round(r_t * fs))
                s0 = max(0, int((r_t - PRE_MS / 1000) * fs))
                s1 = min(len(sig), int((r_t + POST_MS / 1000) * fs))
                if s1 - s0 < 4:
                    ax.set_axis_off()
                    continue

                # Relative time axis (ms, 0 = R peak)
                t_rel = (np.arange(s0, s1) - r_samp) / fs * 1000

                style_axes(ax)
                ax.plot(t_rel, sig[s0:s1], color=PLOT["signal"], lw=1.3, zorder=3)
                ax.axvline(0, color=PLOT["rpeak_ok"], lw=1.0, ls="--", alpha=0.6)

                # R peak marker
                r_samp_clipped = min(max(r_samp, s0), s1 - 1)
                ax.plot(0, sig[r_samp_clipped],
                        marker="o", color=PLOT["rpeak_ok"], ms=8, zorder=7)

                # ── Wave landmark markers ────────────────────────────────
                for col, style in landmark_style.items():
                    if col not in row.index:
                        continue
                    wt_s = float(row[col]) if pd.notna(row[col]) else float("nan")
                    if not np.isfinite(wt_s):
                        continue
                    wt_rel = (wt_s - r_t) * 1000       # ms relative to R
                    if not (t_rel[0] - 1 <= wt_rel <= t_rel[-1] + 1):
                        continue
                    wt_smp = int(round(wt_s * fs))
                    wt_smp = min(max(wt_smp, s0), s1 - 1)
                    w_amp  = sig[wt_smp]
                    ax.plot(wt_rel, w_amp,
                            linestyle="none", color=style["color"],
                            marker=style["marker"], ms=style["ms"],
                            mew=style["mew"], zorder=6)
                    # Wave letter label close to the marker
                    va = "bottom" if style["marker"] == "^" else "top"
                    offset = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    y_lbl  = w_amp + offset if va == "bottom" else w_amp - offset
                    letter = str(style["label"]).split("\n")[0]   # 'P', 'Q', 'S', 'T'
                    ax.text(wt_rel, y_lbl, letter,
                            ha="center", va=va, fontsize=8,
                            color=style["color"], fontweight="bold", zorder=8)

                # ── Interval spans (drawn in axes-fraction y to avoid ylim issues) ─
                ylo, yhi = ax.get_ylim()
                yspan    = yhi - ylo if yhi != ylo else 1.0

                def _span(c0, c1, fc, lab):
                    # x0 / x1 in relative ms
                    try:
                        x0 = (float(row[c0]) - r_t) * 1000 if (c0 and c0 in row.index and pd.notna(row[c0])) else 0.0
                        x1 = (float(row[c1]) - r_t) * 1000 if (c1 and c1 in row.index and pd.notna(row[c1])) else 0.0
                    except Exception as e:
                        log.warning("Interval float conversion failed at beat row: %s", e)
                        return
                    if not (np.isfinite(x0) and np.isfinite(x1)):
                        return
                    # For PR span: goes P_peak → R (x1=0)
                    # For QT span: goes R (x0=0) → T_peak
                    x_lo, x_hi = min(x0, x1), max(x0, x1)
                    if x_hi - x_lo < 0.5:   # skip degenerate spans
                        return
                    ax.axvspan(x_lo, x_hi, ymin=0, ymax=1,
                               color=fc, alpha=0.10, zorder=1)
                    ax.text((x_lo + x_hi) / 2, yhi - 0.04 * yspan,
                            lab, ha="center", va="top", fontsize=8,
                            color=fc, fontweight="bold", zorder=9,
                            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

                # PR interval: P_peak → R (0)
                if "P_peak_s" in row.index and pd.notna(row["P_peak_s"]):
                    _span("P_peak_s", None, "#1B5E20", "PR")
                # QRS: Q → S
                if "Q_peak_s" in row.index and "S_peak_s" in row.index:
                    _span("Q_peak_s", "S_peak_s", purple_col, "QRS")
                # QT: R (0) → T peak
                if "T_peak_s" in row.index and pd.notna(row.get("T_peak_s", float("nan"))):
                    _span(None, "T_peak_s", orange_deep_col, "QT")

                # ── Title: measured interval values ──────────────────────
                parts = []
                for col_name, label in [("PR_ms","PR"), ("QRS_ms","QRS"),
                                         ("QT_ms","QT"), ("RR_ms","RR")]:
                    v = row.get(col_name, float("nan"))
                    if pd.notna(v) and np.isfinite(float(v)):
                        parts.append(f"{label} {float(v):.0f}")
                ax.set_title("  ".join(parts) + " ms" if parts else "Pas d'intervalle",
                             fontsize=8, color=PLOT["text"])
                ax.set_xlabel("ms / R", fontsize=7)
                ax.set_xlim(-PRE_MS, POST_MS)
                # Auto-scale y to signal content (not matplotlib default)
                seg_vis = sig[s0:s1]
                if len(seg_vis) > 0:
                    sv_lo = float(np.percentile(seg_vis, 2))
                    sv_hi = float(np.percentile(seg_vis, 98))
                    sv_pad = max((sv_hi - sv_lo) * 0.25, 0.05)
                    ax.set_ylim(sv_lo - sv_pad, sv_hi + sv_pad)
                if ax_idx == 0:
                    ax.set_ylabel("Amplitude", fontsize=7)

            # ── Anatomy reference panel ──────────────────────────────────
            ax_ref = axes[-1]
            ax_ref.set_facecolor(PLOT["axes"])
            ax_ref.set_xlim(0, 10)
            ax_ref.set_ylim(0, 10)
            ax_ref.set_axis_off()
            ax_ref.set_title("Mouse ECG\ncomplex", fontsize=9,
                             color=PLOT["text"], fontweight="bold")

            # Draw a schematic mouse ECG with J wave (early repolarisation hump)
            ecg_x = [0,  1.0, 1.5, 2.0,        # isoelectric
                     2.5, 2.8,                   # P wave
                     3.1, 3.4,                   # back to iso
                     3.7, 3.9,  4.0, 4.2, 4.4,  # Q / R / S
                     4.6, 4.9,                   # J hump (early repol.)
                     5.3, 5.8, 6.5, 7.0, 7.8, 8.2, 9.5, 10.0]  # T, return
            ecg_y = [5,  5.0, 5.2, 5.0,
                     5.5, 5.8,
                     5.5, 5.0,
                     4.7, 2.0, 9.0, 2.5, 4.7,   # Q-dip, R-peak, S-dip
                     5.1, 5.4,                   # J hump
                     5.1, 5.0, 5.7, 6.1, 5.7, 5.1, 5.0, 5.0]   # T wave

            ax_ref.plot(ecg_x, ecg_y, color=PLOT["signal"], lw=2.0, zorder=3)

            # Landmark annotations
            annotations = [
                (2.8, 6.1, "P",   "#1A56DB",  8),   # P peak
                (4.0, 9.2, "R",   "#1B5E20",  9),   # R peak
                (3.9, 1.6, "Q",   purple_col,  8),   # Q dip
                (4.4, 4.2, "S",   purple_col,  8),   # S dip
                (4.9, 5.7, "J",   teal_col,  8),   # J hump (new)
                (6.5, 6.4, "T",   "#D84315",  8),   # T peak
            ]
            for (x, y, lbl, col, fs_) in annotations:
                ax_ref.text(x, y, lbl, ha="center", va="center",
                            fontsize=fs_, color=col, fontweight="bold", zorder=6)

            # Bracket spans for PR / QRS / QT
            def _bracket(x0, x1, y, label, col):
                ax_ref.annotate("", xy=(x1, y), xytext=(x0, y),
                                arrowprops=dict(arrowstyle="<->", color=col, lw=1.2))
                ax_ref.text((x0 + x1) / 2, y - 0.5, label,
                            ha="center", va="top", fontsize=7,
                            color=col, fontweight="bold")

            _bracket(2.0, 4.0, 1.2, "PR",  "#1B5E20")
            _bracket(3.7, 4.4, 0.0, "QRS", purple_col)
            _bracket(4.0, 7.0, 2.1, "QT",  orange_deep_col)

            # Reference values text
            ref_lines = [
                ("PR",  "30–55 ms",  "#1B5E20"),
                ("QRS", " 8–25 ms",  purple_col),
                ("J",   "15–25 ms",  teal_col),
                ("QT",  "30–90 ms",  "#D84315"),
            ]
            for i, (lbl, val, col) in enumerate(ref_lines):
                ax_ref.text(0.1, 9.6 - i * 1.1, f"{lbl:4s} {val}",
                            ha="left", va="top", fontsize=7.5,
                            color=col, fontfamily="monospace")

            fig.suptitle("ECG beat annotation preview  —  best delineated beats",
                         fontsize=9, color=PLOT["muted"])

        slot.update(draw_ecg_preview)

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

        palette  = [BLUE_DARK, GREEN_DARK, PINK, ORANGE_DARK]
        col_data = [(col, ivl[col].dropna().values, color)
                    for col, color in zip(cols, palette)]

        # Reference ranges for drawing expected-value bands
        _ref = {
            "PR_ms":   MouseECG.PR_NORMAL,
            "QRS_ms":  MouseECG.QRS_NORMAL,
            "QT_ms":   MouseECG.QT_NORMAL,
            "QTc_ms":  MouseECG.QTC_NORMAL,
        }

        def draw_intervals(fig):
            from matplotlib.gridspec import GridSpec
            n_cols = len(col_data)
            # This figure now arrives from CanvasSlot with constrained_layout
            # active, which would recompute/override the explicit margins
            # below -- opt this one figure out so they stay pixel-identical.
            try:
                fig.set_layout_engine(None)
            except Exception as exc:
                log.debug("draw_intervals: set_layout_engine(None) failed: %s", exc)
            gs = GridSpec(1, n_cols, figure=fig,
                          left=0.10, right=0.97, top=0.88, bottom=0.08,
                          wspace=0.40)
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
                if ci == 0:
                    ax.set_ylabel("ms", fontsize=8)
                ax.set_xticks([])
                # Auto-range with padding
                p2, p98 = np.percentile(finite, [2, 98])
                pad = max((p98 - p2) * 0.25, 5)
                ax.set_ylim(p2 - pad, p98 + pad)

        self.app._slots["intervals"].update(draw_intervals)
        # Only describe the interval measurement columns, not wave-position columns
        ivl_stats = ivl[["RR_ms", "PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]].copy()
        ivl_stats = ivl_stats[[c for c in ivl_stats.columns if c in ivl.columns]]
        _ivl_desc = ivl_stats.describe().round(2)
        # self.app._set_textbox(self.app.txt_ivl, _ivl_desc.to_string(),  # Attribut supprimé
        #                   tsv=self.app._describe_to_tsv(_ivl_desc))

    def plot_beat_template(self, r: dict) -> None:
        """Average beat template, ±1 SD band, and amplitude / morphology distributions.

        All heavy numpy work (beat matrix, SD, per-beat correlations) was pre-computed
        in run_full_analysis() on the background thread.  This function only renders.
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

        wt = self.app.analysis.wave_template

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

            # ── Overlay wave template landmarks if confirmed ───────────────
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
                src_note = f"  ·  template: {wt.source}" if wt is not None else ""
            else:
                src_note = ""

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
            ax_amp.set_title("Amplitude des peaks R", loc="left")

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
        hr  = r["hr"]
        td  = r["hrv_time"]
        fd  = r["hrv_freq"]
        nl  = r["hrv_nonlin"]
        ivl = r["intervals"]
        val = self.app._safe_df_val   # shorthand

        # Default values for metrics computed later (referenced in report text)
        porta_pct: float = float("nan")
        guzik_pct: float = float("nan")
        n_dec: int = 0; n_acc: int = 0; n_tot: int = 0

        # ── Metrics table (values only -- no reference-range judgment) ─────────
        def _metric(key: str, text: str) -> None:
            lbl = self.app._sum_metric_vals.get(key)
            if lbl is not None:
                lbl.configure(text=text)

        rdf = r.get("rr_df")
        if rdf is not None and not rdf.empty:
            _metric("hr_mean", f"{rdf['HR_bpm'].mean():.0f}")
            _metric("hr_range",
                    f"{rdf['HR_bpm'].quantile(0.02):.0f}–{rdf['HR_bpm'].quantile(0.98):.0f}")

        _metric("sdnn",  val(td, "HRV_SDNN",  1))
        _metric("rmssd", val(td, "HRV_RMSSD", 1))
        _metric("pnn6",  val(td, "HRV_pNN6",  1))

        try:
            lfhf = float(fd["HRV_LFHF"].values[0]) if (fd is not None and "HRV_LFHF" in fd.columns) else float("nan")
        except Exception:
            lfhf = float("nan")
        _metric("lf_hf", f"{lfhf:.2f}" if np.isfinite(lfhf) else "—")
        _metric("sampen", val(nl, "HRV_SampEn", 2))
        _metric("dfa1",   val(nl, "HRV_DFA_alpha1", 2))

        if ivl is not None and not ivl.empty:
            for col, key in [("PR_ms", "pr"), ("QRS_ms", "qrs"), ("QTc_ms", "qtc")]:
                if col in ivl.columns:
                    d = ivl[col].dropna()
                    if len(d):
                        _metric(key, f"{d.median():.0f}")

        # ── Signal Quality panel ────────────────────────────────────────────────
        # Surfaces numbers the app already computes but never showed anywhere:
        # mean beat-to-template correlation, % of beats below the 0.90
        # threshold, and the artifact-correction breakdown. This is a
        # judgment about DATA TRUSTWORTHINESS, not physiology -- deliberately
        # has no experimental-context-dependent reference ranges.
        beat_corr = r.get("beat_corr")
        rr_ms     = r.get("rr_ms", np.array([]))
        n_beats   = len(beat_corr) if beat_corr is not None else 0
        mean_corr = float(np.nanmean(beat_corr)) if n_beats else float("nan")
        n_bad     = int(np.sum(beat_corr < 0.90)) if n_beats else 0

        def _sq(key: str, text: str) -> None:
            lbl = self.app._sum_quality_vals.get(key)
            if lbl is not None:
                lbl.configure(text=text)

        score = self.app.detection.sig_quality
        _sq("sq_score", f"{score}%" if score is not None else "—")
        _sq("sq_corr", f"{mean_corr:.3f}" if np.isfinite(mean_corr) else "—")
        _sq("sq_badbeats",
            f"{100.0 * n_bad / n_beats:.1f}%  ({n_bad}/{n_beats})" if n_beats else "—")

        arep = self.app.analysis.artifact_report
        if arep:
            removed = arep["n_in"] - arep["n_out"]
            _sq("sq_artifact",
                f"{removed}  ({arep['n_duplicate']} dup · "
                f"{arep['n_nonphysio']} non-physio · {arep['n_ectopic']} ectopic)")
        else:
            _sq("sq_artifact", "not applied")

        if self.app.lbl_sum_verdict is not None:
            dur_s = float(rdf["Time_s"].iloc[-1]) if rdf is not None and len(rdf) else float("nan")
            score_word = "good" if (score or 0) >= 70 else ("fair" if (score or 0) >= 40 else "poor")
            score_color = GREEN if (score or 0) >= 70 else (ORANGE if (score or 0) >= 40 else RED)
            parts = [f"Signal quality {score}% — {score_word}." if score is not None else "Signal quality not computed."]
            if hr.get("n"):
                parts.append(f"{hr['n']} beats")
            if np.isfinite(dur_s):
                parts.append(f"{dur_s:.0f} s")
            if arep:
                parts.append(f"{arep['n_in'] - arep['n_out']} beats auto-corrected")
            self.app.lbl_sum_verdict.configure(
                text="  ·  ".join(parts), text_color=score_color)

        # ── Mirror the two kept plots (RR tachogram, Poincare) ─────────────────
        # Every other Summary plot used to mirror a full-tab draw_fn at ~1/3
        # the size with the same font/legend density -- removed in favour of
        # linking to the full tabs instead (see the "Metrics" section below).
        _MIRRORS = [
            ("rr",            "sum_rr"),
            ("poincare",      "sum_poincare"),
        ]
        for src_key, dst_key in _MIRRORS:
            src = self.app._slots.get(src_key)
            dst = self.app._slots.get(dst_key)
            if src is None or dst is None:
                continue
            fn = getattr(src, "_draw_fn", None)
            if fn is not None:
                try:
                    dst.update(fn)
                except Exception as exc:
                    log.debug("sum mirror %s→%s: %s", src_key, dst_key, exc)

        # ── sum_asymmetry: Asymétrie RR (Porta / Guzik index) ─────────────────
        # Porta index P0: fraction of beats with RR_n+1 < RR_n  (sym → 50 %)
        # Guzik index GI: contribution of decelerations to total variation
        # Both are markers of autonomic nervous system balance asymmetry.
        if len(rr_ms) > 8:
            rr_a = rr_ms[:-1].astype(float)
            rr_b = rr_ms[1:].astype(float)
            diff = rr_b - rr_a
            n_dec = int(np.sum(diff > 0))    # decelerations (RR lengthens)
            n_acc = int(np.sum(diff < 0))    # accelerations (RR shortens)
            n_tot = len(diff)
            porta_raw  = n_acc / n_tot if n_tot else 0.5
            guzik_num  = float(np.sum(diff[diff > 0] ** 2))
            guzik_den  = float(np.sum(diff ** 2))
            guzik_raw  = (1.0 - guzik_num / guzik_den) if guzik_den > 0 else 0.5
            porta_pct  = porta_raw  * 100
            guzik_pct  = guzik_raw  * 100
            # ΔRR histogram (signed)
            drr_clip = np.clip(diff, -60, 60)

            _porta_pct  = porta_pct
            _guzik_pct  = guzik_pct
            _n_dec = n_dec; _n_acc = n_acc; _n_tot = n_tot
            _drr_clip = drr_clip

            def draw_asymmetry(fig):
                ax_bar, ax_hist = fig.subplots(1, 2)
                style_axes(ax_bar); style_axes(ax_hist)

                # Bar chart: proportions
                cats  = ["Decelerations\n(RR↑)", "Neutral", "Accelerations\n(RR↓)"]
                n_neu = _n_tot - _n_dec - _n_acc
                vals2 = [_n_dec / _n_tot * 100,
                         n_neu  / _n_tot * 100,
                         _n_acc / _n_tot * 100]
                colors2 = ["#C62828", "#455A64", BLUE_DARK]
                bars = ax_bar.bar(cats, vals2, color=colors2, alpha=0.80, width=0.5)
                ax_bar.axhline(50, color=BORDER2, lw=1.2, ls="--", alpha=0.7)
                ax_bar.set_ylabel("% beats")
                ax_bar.set_ylim(0, 105)
                for bar, v in zip(bars, vals2):
                    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                                v + 1.5, f"{v:.1f}%",
                                ha="center", va="bottom", fontsize=8, color=PLOT["text"])
                porta_str = f"Porta={_porta_pct:.1f}%  Guzik={_guzik_pct:.1f}%"
                ax_bar.set_title(f"RR Asymmetry  ({porta_str})", loc="left", fontsize=8)
                ax_bar.tick_params(axis="x", labelsize=8)

                # ΔRR histogram
                ax_hist.hist(_drr_clip[_drr_clip < 0], bins=30,
                             color=BLUE_DARK, alpha=0.70, label="accelerations")
                ax_hist.hist(_drr_clip[_drr_clip > 0], bins=30,
                             color="#C62828", alpha=0.70, label="decelerations")
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
            _bc   = np.asarray(beat_corr, dtype=float)
            _wp = self.app._windowed_peaks()
            _rp   = (_wp if _wp is not None else self.app.detection.rpeaks_ok).astype(float) / self.app.signal.fs   # peak times (windowed)
            # Align: beat_corr has 1 value per accepted beat, first peak has no interval
            _t_bc = _rp[:len(_bc)] if len(_rp) >= len(_bc) else _rp

            # Rolling 50-beat mean
            _win  = min(50, max(10, len(_bc) // 20))
            _kern = np.ones(_win) / _win
            _roll = np.convolve(_bc, _kern, mode="valid")
            _roll_t = _t_bc[_win - 1: _win - 1 + len(_roll)]

            n_pts = min(len(_t_bc), len(_bc))
            _bc_t = _t_bc[:n_pts]
            _bc_v = _bc[:n_pts]

            def draw_quality_time(fig):
                ax = fig.add_subplot(111)
                style_axes(ax)
                # Scatter individual beats, coloured by quality
                if len(_bc_t) != len(_bc_v):
                    log.debug("quality plot length mismatch: %d vs %d",
                              len(_bc_t), len(_bc_v))
                sc = ax.scatter(_bc_t, _bc_v, s=2, c=_bc_v, cmap="RdYlGn",
                                vmin=0.7, vmax=1.0, alpha=0.35, rasterized=True, zorder=2)
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
                                 label="corrélation")
                except Exception as e:
                    log.debug("colorbar render failed: %s", e)

            dst = self.app._slots.get("sum_quality_time")
            if dst is not None:
                dst.update(draw_quality_time)

        # QT dispersion -- value only (used in the text report below); the
        # dedicated QT-variability/QT-RR-relationship plot that used to live
        # here in a Summary-only slot was dropped as part of de-duplicating
        # this tab (not mirrored from, or duplicated in, any other tab, but
        # out of scope for the slimmed-down Summary).
        _qt = np.array([])
        qt_disp: float = float("nan")
        if ivl is not None and not ivl.empty:
            _qt = ivl["QT_ms"].dropna().values.astype(float) if "QT_ms" in ivl.columns else np.array([])
            qt_disp = float(np.nanmax(_qt) - np.nanmin(_qt)) if len(_qt) > 3 else float("nan")

        # ── Texte du rapport ─────────────────────────────────────────────────
        filter_note = "  ⚠ Signal brut (sans filtres)" if self.app.signal.no_filter_mode else "  Bandpass + notch + NK clean"
        arep = self.app.analysis.artifact_report
        if arep:
            removed  = arep["n_in"] - arep["n_out"]
            art_lines = [
                "", "  ARTIFACT CORRECTION",
                f"    Before        {arep['n_in']}  beats",
                f"    After         {arep['n_out']}  beats",
                f"    Removed       {removed}",
                f"      Non-physio  {arep['n_nonphysio']}",
                f"      Ectopic     {arep['n_ectopic']}",
                f"      Duplicates  {arep['n_duplicate']}",
            ]
        else:
            art_lines = ["", "  ARTIFACT CORRECTION", "    Not applied"]

        # ── asymmetry metrics for report ──────────────────────────────────────
        asym_lines: list[str] = []
        if len(rr_ms) > 8:
            asym_lines = [
                "", "  RR ASYMMETRY (autonomic system)",
                f"    Porta index (acc.)   {porta_pct:.1f} %  (symétrie → 50 %)",
                f"    Guzik index  (acc.)  {guzik_pct:.1f} %",
                f"    Décélérations        {n_dec} / {n_tot}",
                f"    Accélérations        {n_acc} / {n_tot}",
            ]

        lines = [
            "═" * 62,
            "  ECG ANALYSIS  —  Summary Report",
            f"  Subject  :  {self.app.ent_subject.get()}",
            f"  Date     :  {datetime.now():%Y-%m-%d  %H:%M}",
            f"  File     :  {os.path.basename(self.app.signal.filepath or '')}",
            f"  Filters  :{filter_note}",
            "═" * 62, "",
            "  HEART RATE",
            f"    Moyenne          {hr['mean']:.1f} bpm",
            f"    Min  (2e %ile)   {hr['min']:.1f} bpm",
            f"    Max  (98e %ile)  {hr['max']:.1f} bpm",
            f"    SD               {hr['std']:.2f} bpm",
            f"    N battements     {hr['n']}", "",
            "  HRV — TIME DOMAIN",
            f"    MeanNN   {val(td, 'HRV_MeanNN')} ms",
            f"    SDNN     {val(td, 'HRV_SDNN')} ms",
            f"    RMSSD    {val(td, 'HRV_RMSSD')} ms",
            f"    pNN6     {val(td, 'HRV_pNN6')} %  (>{MouseECG.PNN_THRESHOLD} ms)",
            f"    pNN20    {val(td, 'HRV_pNN20')} %", "",
            "  HRV — FREQUENCY DOMAIN",
            f"    VLF      {val(fd, 'HRV_VLF')} n.u.",
            f"    LF       {val(fd, 'HRV_LF')} n.u.",
            f"    HF       {val(fd, 'HRV_HF')} n.u.",
            f"    LF/HF    {val(fd, 'HRV_LFHF')}", "",
            "  HRV — NON-LINEAR",
            f"    SD1      {val(nl, 'HRV_SD1')} ms",
            f"    SD2      {val(nl, 'HRV_SD2')} ms",
            f"    SampEn   {val(nl, 'HRV_SampEn')}",
            f"    ApEn     {val(nl, 'HRV_ApEn')}",
            f"    DFA α1   {val(nl, 'HRV_DFA_alpha1')}",
            f"    DFA α2   {val(nl, 'HRV_DFA_alpha2')}",
        ]
        if ivl is not None and not ivl.empty and "QT_ms" in ivl.columns:
            lines += ["", "  ECG INTERVALS  (median ± SD)"]
            for col in ["PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]:
                if col in ivl.columns:
                    data = ivl[col].dropna()
                    if len(data):
                        lines.append(
                            f"    {col:<16} {data.median():.1f} ± {data.std():.1f} ms")
            if len(_qt) > 3:
                lines.append(f"    QT dispersion    {qt_disp:.1f} ms")
        lines += asym_lines
        lines += art_lines
        lines += ["", "═" * 62]
        self.app._set_textbox(self.app.txt_sum, "\n".join(lines))

    def reset_result_plots(self) -> None:
        """Clear stored draw_fn on every result-plot slot.

        Prevents stale draw functions from a previous file replaying
        on window resize after a new file is loaded.
        """
        result_slots = (
            "rr", "rr_hist",
            "poincare", "psd", "radar",
            "intervals", "intervals_ecg",
            "beat", "beat_dist",
            "epochs", "rolling_hrv",
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
        (e.g. "Done LF=42%") don't persist after loading a new file.
        """
        _neutral = "  Run Core Analysis first"
        if self.app.lbl_freq_status is not None:
            self.app.lbl_freq_status.configure(text=_neutral, text_color=PLOT["muted"])  # type: ignore[union-attr]
        if self.app.lbl_nonlin_status is not None:
            self.app.lbl_nonlin_status.configure(text=_neutral, text_color=PLOT["muted"])  # type: ignore[union-attr]
        lbl_ivl = getattr(self.app, "lbl_ivl_status", None)
        if lbl_ivl is not None:
            lbl_ivl.configure(text=_neutral, text_color=PLOT["muted"])
        lbl_arr = getattr(self.app, "lbl_arrhythmia_status", None)
        if lbl_arr is not None:
            lbl_arr.configure(text="  Run Core Analysis first", text_color=PLOT["muted"])
        lbl_roll = getattr(self.app, "lbl_roll_status", None)
        if lbl_roll is not None:
            lbl_roll.configure(text="  Run Core Analysis first", text_color=PLOT["muted"])
        for btn_attr in ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl", "btn_run_arrhythmia"):
            btn = getattr(self.app, btn_attr, None)
            if btn is not None:
                btn.configure(state="disabled")

    def reset_kpis(self) -> None:
        """Reset all KPI labels to dash when results are invalidated."""
        for key in ("hr_mean", "hr_range", "rr_mean", "n_beats",
                    "sdnn", "rmssd", "pnn50", "dur",
                    "sq_score", "sq_corr", "sq_badbeats", "sq_artifact"):
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
        r   = self.app.analysis.results
        hr  = r["hr"]
        rdf = r["rr_df"]
        td  = r["hrv_time"]

        def hrv_val(key: str) -> str:
            try:
                return f"{float(td[key].values[0]):.1f}"
            except Exception:
                return "—"

        # Value text is bare (no unit suffix) -- units are baked into the
        # stat-panel tiles once at construction (make_stat_tile's unit=),
        # so the value stays the visually-prominent element per the redesign
        # brief ("valeurs numériques mises en évidence, unités discrètes").
        self.app._kpi["hr_mean"].configure(text=f"{hr['mean']:.0f}")
        self.app._kpi["hr_range"].configure(text=f"{hr['min']:.0f}–{hr['max']:.0f}")
        try:
            rr_mean = float(np.nanmean(r["rr_ms"]))
            self.app._kpi["rr_mean"].configure(text=f"{rr_mean:.0f}")
        except Exception:
            self.app._kpi["rr_mean"].configure(text="—")
        n_valid = hr.get("n_valid", hr["n"])
        self.app._kpi["n_beats"].configure(text=str(n_valid))
        self.app._kpi["sdnn"].configure(text=hrv_val("HRV_SDNN"))
        self.app._kpi["rmssd"].configure(text=hrv_val("HRV_RMSSD"))
        self.app._kpi["pnn50"].configure(text=hrv_val("HRV_pNN6"))
        dur_text = "—"
        try:
            dur_text = f"{rdf['Time_s'].iloc[-1]:.0f}"
            self.app._kpi["dur"].configure(text=dur_text)
        except Exception:
            self.app._kpi["dur"].configure(text="—")

        # Top-bar compact mirrors of the 3 values shown there (§1 of the
        # Phase 1 plan) -- same source numbers as the tiles above, just a
        # second, smaller display in the full-width top bar.
        for key, widget in self.app._topbar_vals.items():
            src = self.app._kpi.get(key)
            if src is not None:
                widget.configure(text=src.cget("text"))

        # ── Signal Quality tiles ────────────────────────────────────────────
        # Duplicates plot_summary()'s cheap ~6-line computation
        # (plot_controller.py, Summary-tab redraw) rather than sharing a
        # helper -- that method runs on a different/deferred schedule and
        # must keep working completely unchanged; this is a deliberate small
        # duplication traded for zero risk to the existing Summary tab.
        beat_corr = r.get("beat_corr")
        n_beats_c = len(beat_corr) if beat_corr is not None else 0
        mean_corr = float(np.nanmean(beat_corr)) if n_beats_c else float("nan")
        n_bad     = int(np.sum(beat_corr < 0.90)) if n_beats_c else 0

        score = self.app.detection.sig_quality
        self.app._kpi["sq_score"].configure(text=f"{score}" if score is not None else "—")
        self.app._kpi["sq_corr"].configure(
            text=f"{mean_corr:.3f}" if np.isfinite(mean_corr) else "—")
        self.app._kpi["sq_badbeats"].configure(
            text=f"{100.0 * n_bad / n_beats_c:.1f}%  ({n_bad}/{n_beats_c})"
            if n_beats_c else "—")

        arep = self.app.analysis.artifact_report
        if arep:
            removed = arep["n_in"] - arep["n_out"]
            self.app._kpi["sq_artifact"].configure(
                text=f"{removed}  ({arep['n_duplicate']} dup · "
                     f"{arep['n_nonphysio']} non-physio · {arep['n_ectopic']} ectopic)")
        else:
            self.app._kpi["sq_artifact"].configure(text="not applied")

        # Repaint the gauge too -- covers the case where update_kpis() is the
        # only thing that ran (e.g. _rebuild_ui()'s trailing call, where the
        # gauge widget was just destroyed/recreated but sig_quality survived
        # as plain data). update_signal_quality() also calls this directly
        # for the real-time-computation path (detection_controller.py).
        if self.app.quality_gauge is not None:
            update_quality_gauge(self.app.quality_gauge, score)

    def redraw_annotations(self) -> None:
        """Re-render the RR tachogram to reflect updated annotations."""
        if self.app.analysis.results is not None:
            try:
                self.plot_rr(self.app.analysis.results)
            except Exception as exc:
                log.debug("_redraw_annotations: %s", exc)
