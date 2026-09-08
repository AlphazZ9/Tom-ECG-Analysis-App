# -*- coding: utf-8 -*-
"""
detection_controller.py
-------------------------
DetectionController -- applying the amplitude threshold to peak candidates,
manual peak edit mode (click-to-exclude/add), undo/redo, hover preview, and
signal-quality scoring.
"""
from __future__ import annotations

import logging
from tkinter import messagebox
from typing import TYPE_CHECKING

import numpy as np

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core.models import MouseECG
from ecg.core.detection import (
    apply_threshold, peak_agreement_counts,
    detect_peaks_wavelet, detect_peaks_sg_derivative, detect_peaks_envelope_max,
)
from ecg.core.ml_detector import detect_peaks_ml, MLPeakModel
from ecg.ui.theme import (
    BLUE, BORDER, BORDER2, GREEN, LIGHT, MUTED, ORANGE, ORANGE_DEEP, RED,
    SPACE_M, SPACE_S,
)
from ecg.ui.widgets import update_quality_gauge
from ecg.ui.dialogs import MethodAgreementDialog

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class DetectionController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    def update_undo_btns(self) -> None:
        """Update the toolbar's undo/redo buttons (single global pair, shared
        by the Detection and Abnormal Events tabs)."""
        n_u, n_r = len(self.app.detection.edit_undo), len(self.app.detection.edit_redo)
        for attr, n in [("btn_toolbar_undo", n_u), ("btn_toolbar_redo", n_r)]:
            try:
                btn = getattr(self.app, attr)
                key = "undo" if "undo" in attr else "redo"
                sym = "↩" if key == "undo" else "↪"
                btn.configure(
                    state="normal" if n else "disabled",
                    text=f"{sym} {key.title()} ({n})" if n else f"{sym} {key.title()}",
                )
            except Exception as e:
                log.debug("undo/redo button configure failed: %s", e)

    def run_detection(self, thresh: float | None = None) -> int:
        """Apply current threshold to pre-computed candidates.

        Parameters
        ----------
        thresh : float | None
            If supplied, use this value directly (safe from a background
            thread).  If None, read ``self.app.sl_thr`` — only safe to call
            from the main thread.

        Returns the number of accepted peaks.  Fast — no signal processing.
        All Tkinter widget *writes* are marshalled through ``after(0, …)``
        so this method is safe to call from either thread.
        """
        if self.app.signal.filtered is None or self.app.detection.all_candidates is None or self.app.detection.all_prominences is None:
            return 0

        if thresh is None:
            thresh = float(self.app.sl_thr.get())   # main-thread-only path  # type: ignore[union-attr]

        accepted, rejected, thresh_amp = apply_threshold(
            self.app.signal.filtered, self.app.detection.all_candidates, self.app.detection.all_prominences, thresh,
            fs=self.app.signal.fs)

        # ── Apply manual exclusions ──────────────────────────────
        if self.app.detection.manual_excluded:
            manual_excl_mask = np.array([p in self.app.detection.manual_excluded for p in accepted], dtype=bool)
            self.app.detection.rpeaks_manual_excl = accepted[manual_excl_mask] if manual_excl_mask.any() else np.array([], int)
            accepted = accepted[~manual_excl_mask]
        else:
            self.app.detection.rpeaks_manual_excl = np.array([], dtype=int)

        # ── Merge manually-added peaks ───────────────────────────
        # Added peaks are never in the candidate set; they bypass all thresholds.
        # They are removed from the exclusion set if present (can't be both).
        if self.app.detection.manual_added:
            added_arr = np.array(sorted(self.app.detection.manual_added), dtype=int)
            self.app.detection.rpeaks_manual_added = added_arr
            # Remove from exclusion set if mistakenly present
            self.app.detection.manual_excluded -= self.app.detection.manual_added
            # Merge and sort
            accepted = np.unique(np.concatenate([accepted, added_arr]))
        else:
            self.app.detection.rpeaks_manual_added = np.array([], dtype=int)

        self.app.detection.rpeaks_ok  = accepted
        self.app.detection.rpeaks_rej = rejected
        self.app.detection.thresh_amp = thresh_amp
        n = len(accepted)

        # Widget writes: always on main thread via after(0, ...)
        color = GREEN if n > 10 else RED
        self.app.after(0, lambda _n=n, _c=color: self.app.lbl_npeaks.configure(  # type: ignore[union-attr]
            text=f"Peaks detected: {_n}", text_color=_c))
        # Reflect the resulting normalised amplitude next to the fraction so
        # "Threshold: 0.50" doesn't look disconnected from the amber line on
        # the plot -- both now read the same "fraction -> norm. amplitude".
        if getattr(self.app, "lbl_thr_hint", None) is not None:
            self.app.after(0, lambda _t=thresh, _a=thresh_amp: self.app.lbl_thr_hint.configure(  # type: ignore[union-attr]
                text=f"Current: {_t:.2f} → ≈{_a:.2f} norm. of the median R-peak "
                     "height. Lower = more peaks."))
        # Enable the artifact review button as soon as peaks are available
        self.app.after(0, lambda: self.app._set_review_artifacts_enabled(n > 4))  # type: ignore[union-attr]
        if self.app.btn_check_agreement is not None:
            self.app.after(0, lambda: self.app._set_check_agreement_enabled(n > 4))  # type: ignore[union-attr]
        self.update_signal_quality(accepted)
        return n

    def update_signal_quality(self, accepted: np.ndarray) -> None:
        """Compute a 0–100 quality score and update the KPI label.

        Quality is based on:
        1. Beat morphology (primary): mean beat-to-template correlation from
           the last analysis run.  High correlation = clean, consistent QRS.
           Falls back to RR regularity if no template analysis has been run yet.
        2. Detection completeness (secondary): ratio of detected to expected
           beats, clipped to [0.5, 1.5] so it modulates but never dominates.

        The previous formula (1 - rr_cv) was unreliable because a healthy mouse
        at high HR during stress has a low rr_cv not from noise but from genuine
        sympathetic activation.
        """
        n = len(accepted)
        if n <= 5 or self.app.signal.time is None:
            return
        dur         = self.app.signal.time[-1]
        expected_n  = dur / 60 * MouseECG.HR_REST_BPM
        ratio       = float(np.clip(n / max(expected_n, 1), 0.5, 1.5))

        # Primary quality signal: mean beat-to-template correlation
        # beat_corr is computed in analyse_core and stored in _results.
        beat_corr = None
        if self.app.analysis.results is not None:
            beat_corr = self.app.analysis.results.get("beat_corr")

        if beat_corr is not None and len(beat_corr) > 0:
            morpho = float(np.nanmean(beat_corr))            # 0–1 (Pearson r)
            morpho = float(np.clip(morpho, 0.0, 1.0))
            quality = int(np.clip(100 * morpho * ratio, 0, 100))
        else:
            # Fallback before full analysis: RR regularity heuristic
            rr_tmp  = np.diff(accepted) / self.app.signal.fs * 1000
            rr_cv   = rr_tmp.std() / (rr_tmp.mean() + 1e-6)
            quality = int(np.clip(100 * (1 - rr_cv) * ratio, 0, 100))

        self.app.detection.sig_quality = quality
        color = GREEN if quality >= 70 else (ORANGE if quality >= 40 else RED)

        def _update_gauge(q=quality, c=color) -> None:
            self.app.lbl_quality.configure(text=f"Signal quality: {q}%", text_color=c)
            # lbl_quality IS the gauge's caption label (see app._build_toolbar) --
            # this immediately overwrites the line above with the finer
            # Excellent/Good/Medium/Poor tiering, which is the intended
            # visible result; the line above needs no changes of its own.
            if self.app.quality_gauge is not None:
                update_quality_gauge(self.app.quality_gauge, q)

        self.app.after(0, _update_gauge)

    def on_det_method_change(self, choice: str) -> None:
        """Show/hide SG options frame based on selected detection method."""
        if self.app._sg_frame is not None:
            if "SG" in choice or "Derivative" in choice:
                self.app._sg_frame.pack(fill="x")
            else:
                self.app._sg_frame.pack_forget()

        # ML Detector needs a trained model before it can run, and that
        # status lives in a right-panel section that starts collapsed on
        # the opposite side of the window -- surface it inline, right next
        # to the Method combobox, instead of only in that far-away section.
        lbl = getattr(self.app, "lbl_ml_inline_status", None)
        if lbl is not None:
            if choice == "ML Detector":
                self.app.refresh_ml_status()
                lbl.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_S))
            else:
                lbl.pack_forget()

    def on_filtering_toggle(self) -> None:
        """Grey out (not hide) FILTER SETTINGS -- notch/band-pass/cleaning
        -- when Filtering is off, and refresh the "Processing" summary.

        Visible either way, so the user can see/set values in advance;
        only interaction is blocked. Distinct from the always-interactive
        DISPLAY & PREVIEW group above it (sw_show_raw/sw_invert_signal/
        sw_filter_preview), which apply regardless of this switch.
        """
        filtering_on = bool(self.app.sw_filtering.get())
        g = self.app._filter_advanced_group
        if g is not None:
            self._set_group_availability(g, filtering_on)
        self.app._update_filter_summary()

    def _set_group_availability(self, frame, available: bool) -> None:
        """Recursively enable/disable every widget under `frame`.

        CTkEntry/CTkComboBox dim automatically via state="disabled". CTkSwitch
        does NOT -- its _draw() only fades the text label, the track/button
        colours stay identical to the enabled look -- so its colours are
        explicitly swapped too: BLUE=enabled+on, BORDER2=enabled+off (the
        _switch() construction defaults), BORDER/LIGHT=unavailable, a third,
        visually distinct "can't touch this" state regardless of the
        switch's underlying on/off value. CTkLabel/CTkFrame don't support
        state= at all -- configure() failures on those are expected no-ops.
        """
        new_state = "normal" if available else "disabled"
        for widget in frame.winfo_children():
            try:
                if isinstance(widget, ctk.CTkSwitch):
                    if available:
                        widget.configure(state=new_state, progress_color=BLUE,
                                          button_color=BORDER2, text_color=MUTED)
                    else:
                        widget.configure(state=new_state, progress_color=BORDER,
                                          button_color=BORDER, text_color=LIGHT)
                else:
                    widget.configure(state=new_state)
            except Exception as e:
                log.debug("widget.configure(state) failed: %s", e)
            self._set_group_availability(widget, available)

    def on_show_raw_toggle(self) -> None:
        """Switch the overview and detail plots between raw and filtered signals.

        The raw signal is normalised (zero-mean, unit-variance) to match
        the amplitude scale of the filtered signal so that peak markers
        remain visually coherent regardless of which view is active.
        No re-processing is needed — both arrays are pre-computed.
        """
        self.app.ui.show_raw = bool(self.app.sw_show_raw.get())
        if self.app.signal.filtered is not None:
            self.app._draw_detail(self.app.ui.nav_pos)

    def push_edit_undo(self) -> None:
        """Snapshot state before a destructive edit action."""
        snap = (frozenset(self.app.detection.manual_excluded), frozenset(self.app.detection.manual_added))
        self.app.detection.edit_undo.append(snap)
        if len(self.app.detection.edit_undo) > self.app._EDIT_UNDO_LIMIT:
            self.app.detection.edit_undo.pop(0)
        self.app.detection.edit_redo.clear()
        self.update_undo_btns()

    def _invalidate_stale_results(self) -> None:
        """Call right before actually committing a peak-set change.

        Rather than leaving SDNN/RMSSD/HRV tables/Poincaré/Summary/every
        export showing the pre-edit numbers until the user remembers to
        click Analyze again. Must only be called on a code path that is
        actually about to change the accepted-peak set -- calling it
        unconditionally on click, before knowing whether the click will hit
        a peak, wipes results for a no-op click.
        """
        if self.app.analysis.results is not None:
            self.app.analysis.results  = None
            self.app.analysis.epoch_df = None
            self.app._reset_kpis()
            self.app._reset_result_plots()

    def undo_edit(self, _event=None) -> None:
        """Ctrl+Z — restore previous peak-edit state."""
        if not self.app.detection.edit_undo:
            self.app._set_status("Nothing to undo.", MUTED)
            return
        cur = (frozenset(self.app.detection.manual_excluded), frozenset(self.app.detection.manual_added))
        self.app.detection.edit_redo.append(cur)
        excl, added = self.app.detection.edit_undo.pop()
        self.app.detection.manual_excluded = set(excl)
        self.app.detection.manual_added    = set(added)
        self.apply_edit_state()
        n = len(self.app.detection.edit_undo)
        self.app._set_status(f"Undone  ({str(n) + ' left' if n else 'none'})  — Ctrl+Y to redo", ORANGE)
        self.update_undo_btns()

    def redo_edit(self, _event=None) -> None:
        """Ctrl+Y — rétablir après undo."""
        if not self.app.detection.edit_redo:
            self.app._set_status("Nothing to redo.", MUTED)
            return
        cur = (frozenset(self.app.detection.manual_excluded), frozenset(self.app.detection.manual_added))
        self.app.detection.edit_undo.append(cur)
        excl, added = self.app.detection.edit_redo.pop()
        self.app.detection.manual_excluded = set(excl)
        self.app.detection.manual_added    = set(added)
        self.apply_edit_state()
        self.app._set_status(f"Redone  ({len(self.app.detection.edit_redo)} remaining)", ORANGE)
        self.update_undo_btns()

    def apply_edit_state(self) -> None:
        if self.app.signal.filtered is not None and self.app.detection.all_candidates is not None:
            # Undo/redo change the accepted-peak set just like a direct
            # edit does -- same stale-HRV invalidation as on_detail_click.
            if self.app.analysis.results is not None:
                self.app.analysis.results  = None
                self.app.analysis.epoch_df = None
                self.app._reset_kpis()
                self.app._reset_result_plots()
            self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
            self.app._draw_detail(self.app.ui.nav_pos)
            # Also refresh the arrhythmia ECG viewer if an event is selected
            if self.app.analysis.arr_selected_idx >= 0 and self.app.analysis.arrhythmia_events:
                self.app._draw_arr_detail()

    def toggle_edit_mode(self) -> None:
        """Toggle the click-to-exclude edit mode on/off."""
        self.app.detection.edit_mode = not self.app.detection.edit_mode
        if self.app.detection.edit_mode:
            self.app.btn_edit_mode.configure(
                fg_color=ORANGE, hover_color=ORANGE_DEEP,
                text_color="white", text="Edit Mode ON",
            )
            self.app.lbl_edit_hint.pack(side="left", padx=(4, 0))  # SPACE_S in app.py
        else:
            self.app.btn_edit_mode.configure(
                fg_color=BORDER, hover_color=BORDER2,
                text_color=MUTED, text="Edit Peaks",
            )
            self.app.lbl_edit_hint.pack_forget()
            # Also turn off free placement when leaving edit mode
            if self.app.detection.edit_free_placement:
                self.app.detection.edit_free_placement = False
                try:
                    self.app.btn_free_placement.configure(
                        fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                        text="Free Placement",
                    )
                except Exception:
                    pass
        if self.app.signal.filtered is not None:
            self.app._draw_detail(self.app.ui.nav_pos)

    def toggle_free_placement(self) -> None:
        """Toggle free-placement mode: bypass proximity constraint when adding peaks.

        When active, right-clicking adds a peak at the local max *regardless* of
        how close it is to an existing peak.  This is useful for very high-rate
        signals or for correcting closely-spaced double-peaks.

        Note: edit mode must be active for this to have any effect.
        """
        self.app.detection.edit_free_placement = not self.app.detection.edit_free_placement
        if self.app.detection.edit_free_placement:
            self.app.btn_free_placement.configure(
                fg_color=ORANGE, hover_color=ORANGE_DEEP,
                text_color="white", text="Free Placement ON",
            )
            self.app._set_status(
                "Free Placement ON — R-click places a peak at the exact click position, "
                "no snapping, no proximity guard.", ORANGE)
        else:
            self.app.btn_free_placement.configure(
                fg_color=BORDER, hover_color=BORDER2,
                text_color=MUTED, text="Free Placement",
            )
            self.app._set_status("Free Placement OFF — normal proximity guard restored.", MUTED)

    def clear_manual_exclusions(self) -> None:
        """Re-include all manually excluded peaks, remove all manually added peaks."""
        if not self.app.detection.manual_excluded and not self.app.detection.manual_added:
            return
        self.push_edit_undo()
        n_excl  = len(self.app.detection.manual_excluded)
        n_added = len(self.app.detection.manual_added)
        self.app.detection.manual_excluded.clear()
        self.app.detection.manual_added.clear()
        # Invalidate any previous analysis — peaks have changed
        self.app.analysis.results  = None
        self.app.analysis.epoch_df = None
        self.app._reset_kpis()
        self.app._reset_result_plots()
        self.app.detection.rpeaks_manual_excl  = np.array([], dtype=int)
        self.app.detection.rpeaks_manual_added = np.array([], dtype=int)
        if self.app.signal.filtered is not None and self.app.detection.all_candidates is not None:
            self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
            self.app._draw_detail(self.app.ui.nav_pos)
        self.app.lbl_npeaks.configure(  # type: ignore[union-attr]
            text=f"Peaks detected: {len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0}",
            text_color=GREEN,
        )
        self.app._set_status(
            f"Manual edits cleared — {n_excl} exclusion(s), {n_added} addition(s) removed.",
            GREEN)

    def on_detail_motion(self, event) -> None:
        """Track mouse position in edit mode and compute the preview peak position.

        In normal mode: snaps to the local maximum within ±tol_samp and shows
        an orange/red marker if it would land too close to an existing peak.
        In free placement mode: the preview follows the cursor exactly (no
        snapping) and is always shown in the "ok" colour.

        Redraws are throttled to 30 ms (≈33 fps) via after().
        """
        if not self.app.detection.edit_mode:
            if self.app.detection.hover_samp is not None:
                self.app.detection.hover_samp = None
                self.app.detection.hover_samp_near = False
                if self.app.ui.hover_after_id is not None:
                    self.app.after_cancel(self.app.ui.hover_after_id)
                    self.app.ui.hover_after_id = None
                self.app._draw_detail(self.app.ui.nav_pos)
            return

        if event.xdata is None or self.app.signal.filtered is None or self.app.signal.fs is None:
            if self.app.detection.hover_samp is not None:
                self.app.detection.hover_samp = None
                self.app.detection.hover_samp_near = False
                self.app._draw_detail(self.app.ui.nav_pos)
            return

        fs         = self.app.signal.fs
        click_time = float(event.xdata)
        click_samp = int(np.clip(round(click_time * fs), 0, len(self.app.signal.filtered) - 1))

        # Free placement: follow the cursor exactly, no snapping, always "ok"
        if self.app.detection.edit_free_placement:
            new_samp = click_samp
            near     = False
        else:
            try:
                win = float(self.app.ent_window.get())
            except Exception:
                win = 2.0
            tol_samp = int(max(MouseECG.MIN_RR_MS / 1000 / 2, win * 0.03) * fs)

            # Snap to local maximum within ±tol_samp
            sig = self.app.signal.filtered
            lo  = max(0, click_samp - tol_samp)
            hi  = min(len(sig), click_samp + tol_samp + 1)
            new_samp = lo + int(np.argmax(sig[lo:hi]))

            # Show warning colour if this would land too close to an existing peak
            near = False
            if self.app.detection.rpeaks_ok is not None and len(self.app.detection.rpeaks_ok):
                min_sep = int(MouseECG.MIN_RR_MS / 1000 * fs * 0.5)
                near    = int(np.min(np.abs(self.app.detection.rpeaks_ok - new_samp))) < min_sep

        # Only schedule a redraw if the position actually changed
        if self.app.detection.hover_samp == new_samp and self.app.detection.hover_samp_near == near:
            return

        self.app.detection.hover_samp      = new_samp
        self.app.detection.hover_samp_near = near

        # Throttle: cancel any pending redraw, schedule a new one in 30 ms
        if self.app.ui.hover_after_id is not None:
            self.app.after_cancel(self.app.ui.hover_after_id)
        self.app.ui.hover_after_id = self.app.after(30, self.flush_hover_redraw)

    def flush_hover_redraw(self) -> None:
        """Execute the throttled hover redraw on the main thread.

        full=False -- a hover-preview marker only affects draw_detail()
        itself (see plot_controller.py's hover_samp usage); the minimap and
        RR/HR/Quality strip don't read hover state and are unchanged by it,
        so redrawing them here was three full-figure rebuilds for the price
        of one, on every ~30ms hover tick while in Edit mode.
        """
        self.app.ui.hover_after_id = None
        self.app._draw_detail(self.app.ui.nav_pos, full=False)

    def on_detail_click(self, event) -> None:
        """Edit-mode click handler for the detail view.

        Left-click  (button 1) near an existing peak → toggle exclusion
        Right-click (button 3) anywhere              → add peak at local max,
                                                       or remove if clicking
                                                       a manually-added peak

        Only active when ``_edit_mode`` is True.
        """
        if not self.app.detection.edit_mode:
            return
        if event.xdata is None or self.app.signal.filtered is None:
            return
        if self.app.detection.rpeaks_ok is None:
            return

        fs         = self.app.signal.fs
        click_time = float(event.xdata)      # seconds
        click_samp = int(round(click_time * fs))
        click_samp = int(np.clip(click_samp, 0, len(self.app.signal.filtered) - 1))

        # ── Tolerance ────────────────────────────────────────────────────────
        try:
            win = float(self.app.ent_window.get())
        except Exception:
            win = 2.0
        # Half the minimum RR interval (in seconds) is a natural click radius
        tol_s    = max(MouseECG.MIN_RR_MS / 1000 / 2, win * 0.03)
        tol_samp = int(tol_s * fs)

        is_left  = (event.button == 1)
        is_right = (event.button == 3)

        # ──────────────────────────────────────────────────────────────────────
        #  RIGHT-CLICK: add a new peak, or remove an existing manually-added one
        # ──────────────────────────────────────────────────────────────────────
        if is_right:
            # Every branch below this point commits a change (add/remove/
            # replace) -- unlike the left-click branch, there's no no-op
            # path here, so invalidating unconditionally is safe.
            self._invalidate_stale_results()

            # ── FREE PLACEMENT: always add at the exact clicked sample ────────
            # All guards (proximity, remove-nearby, local-max snapping) are
            # bypassed.  The peak lands precisely where the user clicked.
            if self.app.detection.edit_free_placement:
                new_samp = click_samp
                self.push_edit_undo()
                self.app.detection.manual_added.add(new_samp)
                self.app.detection.manual_excluded.discard(new_samp)
                self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
                n_ok    = len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0
                n_added = len(self.app.detection.manual_added)
                self.app._set_status(
                    f"[Free] Added peak at {new_samp / fs:.3f} s  |  "
                    f"Total added: {n_added}  |  Accepted: {n_ok}  "
                    "— click Analyze to refresh HRV.",
                    BLUE,
                )
                self.app._draw_detail(self.app.ui.nav_pos)
                return

            # ── NORMAL MODE ───────────────────────────────────────────────────
            # First check if click is near a manually-added peak → remove it
            if self.app.detection.rpeaks_manual_added is not None and len(self.app.detection.rpeaks_manual_added):
                dists = np.abs(self.app.detection.rpeaks_manual_added - click_samp)
                nearest_i = int(np.argmin(dists))
                if dists[nearest_i] <= tol_samp:
                    peak_to_remove = int(self.app.detection.rpeaks_manual_added[nearest_i])
                    self.push_edit_undo()
                    self.app.detection.manual_added.discard(peak_to_remove)
                    self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
                    n_ok = len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0
                    self.app._set_status(
                        f"Removed manually added peak at {click_time:.3f} s  |  "
                        f"Accepted: {n_ok}  — click Analyze to refresh HRV.",
                        ORANGE,
                    )
                    self.app._draw_detail(self.app.ui.nav_pos)
                    return

            # Snap to local maximum within ±tol_samp of click
            sig      = self.app.signal.filtered
            lo       = max(0, click_samp - tol_samp)
            hi       = min(len(sig), click_samp + tol_samp + 1)
            seg      = sig[lo:hi]
            local_max_offset = int(np.argmax(seg))
            new_samp = lo + local_max_offset

            # If too close to existing accepted peak → replace it
            # (exclude the old one, add the new one) instead of refusing.
            if self.app.detection.rpeaks_ok is not None and len(self.app.detection.rpeaks_ok):
                min_sep_samp = int(MouseECG.MIN_RR_MS / 1000 * fs * 0.5)
                dists_ok     = np.abs(self.app.detection.rpeaks_ok - new_samp)
                nearest_idx  = int(np.argmin(dists_ok))
                nearest_dist = int(dists_ok[nearest_idx])
                if nearest_dist < min_sep_samp:
                    # Replace: exclude the nearby peak, add the new one
                    old_peak = int(self.app.detection.rpeaks_ok[nearest_idx])
                    self.push_edit_undo()
                    self.app.detection.manual_excluded.add(old_peak)
                    self.app.detection.manual_added.discard(old_peak)
                    self.app.detection.manual_added.add(new_samp)
                    self.app.detection.manual_excluded.discard(new_samp)
                    self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
                    n_ok = len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0
                    self.app._set_status(
                        f"Replaced peak {old_peak/fs:.3f} s → {new_samp/fs:.3f} s  "
                        f"({nearest_dist/fs*1000:.1f} ms apart)  |  Accepted: {n_ok}  "
                        "— click Analyze to refresh HRV.",
                        ORANGE,
                    )
                    self.app._draw_detail(self.app.ui.nav_pos)
                    return

            self.push_edit_undo()
            self.app.detection.manual_added.add(new_samp)
            # If this sample was previously excluded, unexclude it
            self.app.detection.manual_excluded.discard(new_samp)
            self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
            n_ok    = len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0
            n_added = len(self.app.detection.manual_added)
            self.app._set_status(
                f"Added peak at {new_samp / fs:.3f} s (snapped to local max)  |  "
                f"Total added: {n_added}  |  Accepted: {n_ok}  "
                "— click Analyze to refresh HRV.",
                ORANGE,
            )
            self.app._draw_detail(self.app.ui.nav_pos)
            return

        # ──────────────────────────────────────────────────────────────────────
        #  LEFT-CLICK: toggle exclusion of the nearest existing peak
        # ──────────────────────────────────────────────────────────────────────
        if not is_left:
            return

        # Pool: all accepted peaks + all currently excluded peaks
        # (manually added peaks are excluded from toggle — use right-click to remove)
        added_set  = self.app.detection.manual_added
        base_ok    = np.array([p for p in self.app.detection.rpeaks_ok if p not in added_set], int) \
                     if self.app.detection.rpeaks_ok is not None else np.array([], int)
        excl_arr   = self.app.detection.rpeaks_manual_excl if self.app.detection.rpeaks_manual_excl is not None \
                     else np.array([], int)
        candidates = np.concatenate([base_ok, excl_arr])
        if len(candidates) == 0:
            return

        times_s   = candidates / fs
        distances = np.abs(times_s - click_time)
        nearest_i = int(np.argmin(distances))

        if distances[nearest_i] > tol_s:
            return   # click not close enough to any peak

        # Past this point the click is guaranteed to toggle a peak's
        # exclusion -- safe to invalidate now (not unconditionally at the
        # top of the handler, which would also wipe results for the two
        # no-op returns above).
        self._invalidate_stale_results()

        peak_idx = int(candidates[nearest_i])

        self.push_edit_undo()
        if peak_idx in self.app.detection.manual_excluded:
            self.app.detection.manual_excluded.discard(peak_idx)
        else:
            self.app.detection.manual_excluded.add(peak_idx)

        self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
        n_ok   = len(self.app.detection.rpeaks_ok)  if self.app.detection.rpeaks_ok   is not None else 0
        n_excl = len(self.app.detection.manual_excluded)
        self.app._set_status(
            f"Manual exclusions: {n_excl}  |  Accepted peaks: {n_ok}  "
            "— click Analyze to refresh HRV.",
            ORANGE,
        )
        self.app._draw_detail(self.app.ui.nav_pos)

    def on_threshold_slide(self, value: float) -> None:
        """Called continuously while the slider is being dragged.

        Widget label and entry are updated immediately for visual feedback.
        Detection and redraws are debounced (80 ms) so rapid drag events do
        not flood the rendering pipeline — especially important for long
        recordings where apply_threshold() + two canvas draws take ~50 ms.
        """
        self.app.lbl_thr.configure(text=f"Threshold:  {value:.3f}")
        self.app.ent_thr.delete(0, "end")  # type: ignore[union-attr]
        self.app.ent_thr.insert(0, f"{value:.3f}")  # type: ignore[union-attr]

        if self.app.signal.filtered is None or self.app.detection.all_candidates is None:
            return

        # Cancel any previously scheduled update and reschedule
        if self.app.ui.thr_debounce_id is not None:
            self.app.after_cancel(self.app.ui.thr_debounce_id)
        self.app.ui.thr_debounce_id = self.app.after(80, lambda v=value: self.apply_threshold_ui(v))

    def apply_threshold_ui(self, value: float) -> None:
        """Run detection and refresh plots — called after debounce delay.

        Always executes on the main thread (scheduled via after()), so it is
        safe to read the slider and write widgets directly.
        """
        self.app.ui.thr_debounce_id = None
        # Peaks are about to change — invalidate stale results
        if self.app.analysis.results is not None:
            self.app.analysis.results  = None
            self.app.analysis.epoch_df = None
            self.app._reset_kpis()
            self.app._reset_result_plots()
        self.run_detection(value)
        self.app._draw_detail(self.app.ui.nav_pos)

    def on_threshold_entry(self, event=None) -> None:
        """Called when the user types a value in the exact-threshold entry.

        Applies immediately — no debounce — since this is a deliberate commit.
        """
        try:
            value = max(0.01, min(2.0, float(self.app.ent_thr.get())))  # type: ignore[union-attr]
            self.app.sl_thr.set(value)  # type: ignore[union-attr]
            self.app.lbl_thr.configure(text=f"Threshold:  {value:.3f}")
            if self.app.signal.filtered is not None and self.app.detection.all_candidates is not None:
                self.apply_threshold_ui(value)
        except ValueError:
            pass

    def update_quality_badge(self) -> None:
        """Update the persistent quality badge in the KPI bar."""
        if not hasattr(self.app, "_lbl_quality_badge"):
            return
        beat_corr = (self.app.analysis.results or {}).get("beat_corr")
        if beat_corr is None or len(beat_corr) == 0:
            self.app._lbl_quality_badge.configure(text="", fg_color="transparent")
            return
        mean_r = float(np.nanmean(beat_corr))
        n_bad  = int(np.sum(beat_corr < 0.90))
        pct    = 100.0 * n_bad / max(len(beat_corr), 1)
        if mean_r >= 0.95:
            col, label = GREEN,   f"● Excellent  {mean_r:.3f}"
        elif mean_r >= 0.90:
            col, label = GREEN,   f"● Good  {mean_r:.3f}"
        elif mean_r >= 0.80:
            col, label = ORANGE,  f"● Fair  {mean_r:.3f}  ({pct:.0f}% low)"
        else:
            col, label = RED,     f"● Poor  {mean_r:.3f}  ({pct:.0f}% low)"
        self.app._lbl_quality_badge.configure(text=label, fg_color=col,
                                           text_color="white")

    def check_method_agreement(self) -> None:
        """Cross-check the active detection result against independently
        implemented detectors run on the same (already filtered) signal.

        A beat several DIFFERENT algorithms all miss is a far stronger
        "this one is questionable" signal than any single method's own
        confidence -- this is complementary to that per-peak confidence,
        not a replacement for it.

        Deliberately excludes "Auto" and whichever method is currently
        active from the comparison set: re-running the active method
        against itself would trivially "agree" on every beat, and Auto's
        candidate generation is entangled with fix_polarity()'s own
        polarity vote in a way that doesn't cleanly separate into a
        standalone re-invocation. Wavelet, SG+Derivative, and Envelope Max
        are independent implementations already; the ML detector joins
        them when a trained model exists. Any method that's unavailable
        (no PyWavelets, no trained model, or a genuine failure) is skipped
        with a reason shown to the user rather than silently dropped.
        """
        if self.app.signal.filtered is None or self.app.detection.rpeaks_ok is None \
                or len(self.app.detection.rpeaks_ok) == 0:
            messagebox.showwarning("Not ready", "Click Detect Peaks first.")
            return

        # Snapshot on the main thread -- the worker runs in a background
        # thread and must not touch Tk widgets or shared self.app.* state
        # that a new file load could reset mid-run.
        sig       = self.app.signal.filtered
        fs        = self.app.signal.fs
        primary   = self.app.detection.rpeaks_ok.copy()
        min_rr_ms = self.app._safe_float(self.app.ent_minrr, MouseECG.MIN_RR_MS)
        active    = (self.app.cb_det_method.get()
                     if self.app.cb_det_method is not None else "").lower()
        tolerance_samples = max(1, int(round(0.010 * fs)))  # 10 ms snap tolerance

        if self.app.lbl_agreement_status is not None:
            self.app.lbl_agreement_status.configure(
                text="  Running other detectors…", text_color=ORANGE)

        def _worker() -> dict:
            other_peaks: "list[np.ndarray]" = []
            methods_run: "list[str]" = []
            methods_skipped: "list[str]" = []

            def _try(name: str, active_key: str, fn) -> None:
                if active_key in active:
                    methods_skipped.append(f"{name} (already the active method)")
                    return
                try:
                    peaks, _proms, _t = fn()
                    other_peaks.append(np.asarray(peaks))
                    methods_run.append(f"{name} ({len(peaks)} peaks)")
                except Exception as exc:
                    methods_skipped.append(f"{name} ({exc})")

            _try("Wavelet", "wavelet",
                 lambda: detect_peaks_wavelet(sig, fs, min_rr_ms=min_rr_ms))
            _try("SG+Derivative", "sg",
                 lambda: detect_peaks_sg_derivative(sig, fs, min_rr_ms=min_rr_ms))
            _try("Envelope Max", "envelope",
                 lambda: detect_peaks_envelope_max(sig, fs, min_rr_ms=min_rr_ms))
            if "ml" in active or "machine" in active:
                methods_skipped.append("ML Detector (already the active method)")
            else:
                try:
                    model = MLPeakModel.load()
                    peaks, _proms, _t = detect_peaks_ml(sig, fs, model)
                    other_peaks.append(np.asarray(peaks))
                    methods_run.append(f"ML Detector ({len(peaks)} peaks)")
                except Exception as exc:
                    methods_skipped.append(f"ML Detector ({exc})")

            if not other_peaks:
                return {"ok": False, "methods_skipped": methods_skipped}

            counts     = peak_agreement_counts(primary, other_peaks, tolerance_samples)
            n_methods  = len(other_peaks)
            n_disagree = int(np.sum(counts == 0))
            n_full     = int(np.sum(counts == n_methods))
            histogram  = [int(np.sum(counts == k)) for k in range(n_methods + 1)]
            return {
                "ok": True,
                "methods_run": methods_run,
                "methods_skipped": methods_skipped,
                "n_methods": n_methods,
                "n_beats": len(primary),
                "n_disagree": n_disagree,
                "n_full": n_full,
                "histogram": histogram,
                "disagree_samples": primary[counts == 0],
            }

        def _done(result: dict) -> None:
            if self.app.detection.rpeaks_ok is None:
                return
            if not result.get("ok"):
                reason = "; ".join(result.get("methods_skipped", [])) or "no other methods available"
                if self.app.lbl_agreement_status is not None:
                    self.app.lbl_agreement_status.configure(
                        text=f"  Agreement check: {reason}", text_color=ORANGE)
                self.app._set_status(f"Cross-method agreement: {reason}", ORANGE)
                return

            self.app.detection.agreement_disagree_samples = result["disagree_samples"]
            n_beats     = result["n_beats"]
            n_disagree  = result["n_disagree"]
            pct_confirmed = 100.0 * (n_beats - n_disagree) / max(n_beats, 1)
            summary = (
                f"{result['n_methods']} other method(s) · "
                f"{pct_confirmed:.0f}% beats confirmed by ≥1 other · "
                f"{n_disagree} unconfirmed"
            )
            self.app.detection.agreement_summary = summary
            if self.app.lbl_agreement_status is not None:
                color = (GREEN if n_disagree == 0
                         else ORANGE if pct_confirmed >= 90 else RED)
                self.app.lbl_agreement_status.configure(text=f"  {summary}", text_color=color)
            skipped = result.get("methods_skipped", [])
            skipped_note = ("  ·  skipped: " + ", ".join(skipped)) if skipped else ""
            self.app._set_status(
                f"Cross-method agreement done — {summary}{skipped_note}", GREEN)
            self.app._draw_detail()
            MethodAgreementDialog(self.app, result=result, fs=fs)

        self.app._start_async_result(
            self.app.btn_check_agreement, "Checking…", _worker, _done)
