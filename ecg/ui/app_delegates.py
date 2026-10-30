# -*- coding: utf-8 -*-
"""
ecg.ui.app_delegates
--------------------
ECGApp event handlers and thin delegates to the controllers.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from tkinter import filedialog, messagebox
import numpy as np
from ecg.core.models import MouseECG
from ecg.core.detection import (
    detect_rr_artifacts, apply_artifact_decisions, merge_artifact_reports,
)
from ecg.io.loaders import list_channels
from ecg.io.session import session_exists, load_rr_series
from ecg.io.db import _DB_AVAILABLE, get_notes, set_notes, recent_recordings
from ecg.ui.theme import (
    BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT, RED, BLUE, GREEN,
    ORANGE, GREEN_DARK, SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_LABEL,
    FONT_SMALL, FONT_KPI_LABEL, FONT_BTN_PRIMARY, FONT_BTN_SEC,
    FONT_SIDEBAR_HDR, FONT_CARD_TITLE,
)
from ecg.ui.widgets import make_sparkline
from ecg.ui.dialogs import (
    ArtifactReviewDialog, AnnotationManagerDialog, PacingPeriodManagerDialog,
)
from ecg.ui.wave_editor import WaveTemplateMiniEditor

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class DelegatesMixin:
    """ECGApp event handlers and thin delegates to the controllers."""

    # ════════════════════════════════════════════════════════
    #  FREQUENCY BAND HELPER
    # ════════════════════════════════════════════════════════

    def _get_freq_bands(self: ECGApp) -> "tuple[tuple, tuple, tuple]":
        """Return (vlf, lf, hf) tuples from the selected band preset.

        Reads ``cb_freq_band`` if it exists; falls back to Mouse Thireau defaults.
        """
        return self.analysis_ctrl.get_freq_bands()

    # ════════════════════════════════════════════════════════
    #  EXPORT FIGURES (clean PNG, no spike markers)
    # ════════════════════════════════════════════════════════

    def _export_figures(self: ECGApp) -> None:
        """Export curated publication-ready PNG figures to a chosen folder."""
        self.export_ctrl.export_figures()

    # ════════════════════════════════════════════════════════
    #  SIGNAL PIPELINE  (detection engine)
    # ════════════════════════════════════════════════════════

    def _snapshot_params(self: ECGApp) -> dict:
        """Read every widget value on the main thread and return a plain dict snapshot.

        Delegates to ``FilterParams.from_widgets`` so that the field list is
        maintained in exactly one place (the FilterParams dataclass).  Any new
        parameter added to FilterParams is automatically picked up here and in
        _collect_session_state / _restore_session_worker.
        """
        return self.session_ctrl.snapshot_params()

    def _compute_preview_bundle(
        self: ECGApp,
        sig_raw:     np.ndarray,
        fs:          int,
        params:      dict,
        progress_cb: "Optional[Callable[[int, str], None]]" = None,
    ) -> dict:
        """Pure computation — no access to *self*, no Tkinter calls.

        Filters, normalises, fixes polarity, and finds all R-peak candidates.
        Returns a plain dict that ``_on_preview_done`` writes to ``self.*``
        atomically on the main thread.

        This separation is the thread-safety contract:
        * Background workers call this method and return the bundle.
        * Only the main thread writes instance variables.
        """
        return self.signal_ctrl.compute_preview_bundle(sig_raw, fs, params, progress_cb)

    def _prepare_signal(
        self: ECGApp,
        params: dict,
        progress_cb: "Optional[Callable[[int, str], None]]" = None,
    ) -> None:
        """Filter, normalise, and fix polarity. Writes results to self.

        Thin wrapper around the pure ``_compute_preview_bundle`` for callers
        that already have signal data on self and are running on the main
        thread. Not currently called anywhere in the app — session restore
        now calls ``_compute_preview_bundle`` directly from
        ``_restore_session_worker`` (a background thread) and writes the
        result to self from ``_on_restore_session_done`` (main thread) —
        kept for any future main-thread caller that wants the write-to-self
        convenience.

        The background preview path uses ``_compute_preview_bundle`` directly
        and writes to self only from ``_on_preview_done`` (main thread).
        """
        self.signal_ctrl.prepare_signal(params, progress_cb)

    def _run_detection(self: ECGApp, thresh: float | None = None) -> int:
        """Apply current threshold to pre-computed candidates.

        Parameters
        ----------
        thresh : float | None
            If supplied, use this value directly (safe from a background
            thread).  If None, read ``self.sl_thr`` — only safe to call
            from the main thread.

        Returns the number of accepted peaks.  Fast — no signal processing.
        All Tkinter widget *writes* are marshalled through ``after(0, …)``
        so this method is safe to call from either thread.
        """
        return self.detection_ctrl.run_detection(thresh)

    def _update_signal_quality(self: ECGApp, accepted: np.ndarray) -> None:
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
        self.detection_ctrl.update_signal_quality(accepted)

    def _on_project_name_change(self: ECGApp, event=None) -> None:
        """Mirror the sidebar 'Project name' entry into the top bar.

        Deliberately independent of update_kpis() (which only repaints once
        analysis results exist) -- typing a project name should show up in
        the top bar immediately, not only after running analysis. Also
        called from _rebuild_ui() after _restore_ui_state() so a typed value
        survives a theme toggle's destroy/rebuild cycle.
        """
        if self.ent_project_name is None or self.lbl_topbar_project is None:
            return
        name = self.ent_project_name.get().strip()  # type: ignore[union-attr]
        self.lbl_topbar_project.configure(text=name if name else "—")  # type: ignore[union-attr]

    def _update_topbar_filename(self: ECGApp) -> None:
        """Mirror the currently-loaded file's name into the top bar.

        lbl_topbar_file was previously write-once at construction time
        ("No file loaded") with nothing ever updating it afterward -- it
        stayed stuck on that placeholder for the rest of the app's life,
        even with a file loaded and fully analysed. Called wherever
        self.signal.filepath changes (file load) and from
        _restore_ui_state() so it survives a theme toggle's rebuild, same
        as _on_project_name_change().
        """
        if self.lbl_topbar_file is None:
            return
        fp = self.signal.filepath
        text = Path(fp).name if fp else "No file loaded"
        self.lbl_topbar_file.configure(text=text)  # type: ignore[union-attr]

    def _on_det_method_change(self: ECGApp, choice: str) -> None:
        """Show/hide SG options frame based on selected detection method."""
        self.detection_ctrl.on_det_method_change(choice)

    # ── ML detector: verified-for-training switch ────────────
    def _on_verified_training_toggle(self: ECGApp) -> None:
        self.session_ctrl.on_verified_training_toggle()

    def _save_for_training_only(self: ECGApp) -> None:
        self.session_ctrl.save_for_training_only()

    def _open_ml_training_dialog(self: ECGApp) -> None:
        """Open the Train/Retrain dialog for the ML R-peak detector."""
        from ecg.ui.dialogs import MLTrainingDialog
        dlg = MLTrainingDialog(self)
        self.wait_window(dlg)
        self.refresh_ml_status()

    def refresh_ml_status(self: ECGApp) -> None:
        """Update the ML-detector sidebar status label.

        Shows verified-file/sample counts and, if a model is trained, its
        hold-out accuracy/F1 from the metadata sidecar. Called on startup,
        after Save Session, and after the training dialog closes.
        """
        if self.lbl_ml_status is None:
            return
        from ecg.core.ml_detector import training_data_summary, MLPeakModel
        summary = training_data_summary()
        model = MLPeakModel.load()
        if model is not None:
            meta = model.meta
            text = (f"Model trained on {meta.get('n_training_files', '?')} file(s), "
                    f"{meta.get('n_training_samples', '?')} samples — "
                    f"acc={meta.get('holdout_accuracy', 0):.2f} "
                    f"f1={meta.get('holdout_f1', 0):.2f}\n"
                    f"{summary['n_files']} file(s) currently verified.")
            color = GREEN
        else:
            text = (f"No trained model yet. {summary['n_files']} file(s) verified "
                    f"({summary['n_samples']} samples).")
            color = MUTED
        self.lbl_ml_status.configure(text=text, text_color=color)  # type: ignore[union-attr]
        # Mirrors the same text next to the Method combobox when ML Detector
        # is selected (see detection_controller.on_det_method_change) --
        # the right-panel section above starts collapsed, so this is the
        # only copy visible until the user opens it.
        inline = getattr(self, "lbl_ml_inline_status", None)
        if inline is not None:
            inline.configure(text=text, text_color=color)

    # ── Filtering master toggle ───────────────────────────────

    def _on_filtering_toggle(self: ECGApp) -> None:
        """Grey out the notch/band-pass/cleaning group when Filtering is off."""
        self.detection_ctrl.on_filtering_toggle()

    _CLEAN_METHOD_LABELS = {
        "neurokit":        "NeuroKit",
        "pantompkins1985": "Pan-Tompkins",
        "elgendi2010":     "Elgendi",
        "hamilton2002":    "Hamilton",
        "biosppy":         "BioSPPy",
    }

    def _update_filter_summary(self: ECGApp) -> None:
        """Refresh the read-only 'Processing' summary from live widget values.

        Mirrors exactly what compute_preview_bundle() will actually do to
        the signal -- so it only reflects FILTER SETTINGS (band-pass/notch/
        cleaning), not the DISPLAY & PREVIEW group (show raw/invert/preview),
        which don't change what gets analysed.
        """
        if self.lbl_filter_summary is None:
            return
        if self.sw_filtering is None or not bool(self.sw_filtering.get()):
            lines = ["Processing", "•  Raw signal"]
        else:
            lp = self._safe_float(self.ent_lp, MouseECG.BP_LO_HZ)
            hp = self._safe_float(self.ent_hp, MouseECG.BP_HI_HZ)
            lines = ["Processing", f"•  Band-pass: {lp:g}–{hp:g} Hz"]
            if self.sw_notch is not None and bool(self.sw_notch.get()):
                lines.append("•  Notch: 50 Hz")
            clean_val = self.cb_clean.get() if self.cb_clean is not None else "neurokit"
            clean_label = self._CLEAN_METHOD_LABELS.get(clean_val, clean_val.title())
            lines.append(f"•  {clean_label} cleaning")
        self.lbl_filter_summary.configure(text="\n".join(lines))

    # ── Raw / Filtered toggle ─────────────────────────────────

    def _on_show_raw_toggle(self: ECGApp) -> None:
        """Switch the overview and detail plots between raw and filtered signals.

        The raw signal is normalised (zero-mean, unit-variance) to match
        the amplitude scale of the filtered signal so that peak markers
        remain visually coherent regardless of which view is active.
        No re-processing is needed — both arrays are pre-computed.
        """
        self.detection_ctrl.on_show_raw_toggle()

    # ── Overview click-to-navigate ────────────────────────────

    def _on_overview_click(self: ECGApp, event) -> None:
        """Stub — overview removed."""
        self.nav_ctrl.on_overview_click(event)

    def _on_overview_motion(self: ECGApp, event) -> None:
        """Drag-to-scrub while the mouse button is held on the minimap strip."""
        self.nav_ctrl.on_overview_motion(event)

    def _on_overview_release(self: ECGApp, event) -> None:
        """End minimap drag-to-scrub."""
        self.nav_ctrl.on_overview_release(event)

    # ── Detail scroll-wheel zoom ──────────────────────────────

    def _on_detail_scroll(self: ECGApp, event) -> None:
        """Zoom the detail view's x-axis in/out centred on the cursor position.

        Each scroll tick zooms by a factor of 1.25 (in) or 0.8 (out).
        After zooming, ``_nav_pos`` and the window-entry widget are updated
        to reflect the new visible range.

        The y-axis is intentionally unchanged — vertical zoom is handled by
        the matplotlib toolbar's Zoom-to-rectangle tool.
        """
        self.nav_ctrl.on_detail_scroll(event)

    # ── Undo / Redo for manual peak edits ────────────────────

    def _push_edit_undo(self: ECGApp) -> None:
        """Snapshot state before a destructive edit action."""
        self.detection_ctrl.push_edit_undo()

    def _undo_edit(self: ECGApp, _event=None) -> None:
        """Ctrl+Z — restore previous peak-edit state."""
        self.detection_ctrl.undo_edit(_event)

    def _redo_edit(self: ECGApp, _event=None) -> None:
        """Ctrl+Y — rétablir après undo."""
        self.detection_ctrl.redo_edit(_event)

    # ── Manual peak exclusion ─────────────────────────────────

    def _toggle_edit_mode(self: ECGApp) -> None:
        """Toggle the click-to-exclude edit mode on/off."""
        self.detection_ctrl.toggle_edit_mode()

    def _toggle_free_placement(self: ECGApp) -> None:
        """Toggle free-placement mode: bypass proximity constraint when adding peaks.

        When active, right-clicking adds a peak at the local max *regardless* of
        how close it is to an existing peak.  This is useful for very high-rate
        signals or for correcting closely-spaced double-peaks.

        Note: edit mode must be active for this to have any effect.
        """
        self.detection_ctrl.toggle_free_placement()

    def _clear_manual_exclusions(self: ECGApp) -> None:
        """Re-include all manually excluded peaks, remove all manually added peaks."""
        self.detection_ctrl.clear_manual_exclusions()

    def _on_detail_motion(self: ECGApp, event) -> None:
        """Track mouse position in edit mode and compute the preview peak position.

        In normal mode: snaps to the local maximum within ±tol_samp and shows
        an orange/red marker if it would land too close to an existing peak.
        In free placement mode: the preview follows the cursor exactly (no
        snapping) and is always shown in the "ok" colour.

        Redraws are throttled to 30 ms (≈33 fps) via after().
        """
        self.detection_ctrl.on_detail_motion(event)

    def _on_detail_click(self: ECGApp, event) -> None:
        """Edit-mode click handler for the detail view.

        Left-click  (button 1) near an existing peak → toggle exclusion
        Right-click (button 3) anywhere              → add peak at local max,
                                                       or remove if clicking
                                                       a manually-added peak

        Only active when ``_edit_mode`` is True.
        """
        self.detection_ctrl.on_detail_click(event)

    # ── Threshold slider / entry callbacks ────────────────────
    def _on_threshold_slide(self: ECGApp, value: float) -> None:
        """Called continuously while the slider is being dragged.

        Widget label and entry are updated immediately for visual feedback.
        Detection and redraws are debounced (80 ms) so rapid drag events do
        not flood the rendering pipeline — especially important for long
        recordings where apply_threshold() + two canvas draws take ~50 ms.
        """
        self.detection_ctrl.on_threshold_slide(value)

    def _on_threshold_entry(self: ECGApp, event=None) -> None:
        """Called when the user types a value in the exact-threshold entry.

        Applies immediately — no debounce — since this is a deliberate commit.
        """
        self.detection_ctrl.on_threshold_entry(event)

    # ════════════════════════════════════════════════════════
    #  ACTIONS  (file, preview, run)
    # ════════════════════════════════════════════════════════

    def _open_file(self: ECGApp) -> None:
        path = filedialog.askopenfilename(
            title="Open Spike2 .mat",
            filetypes=[("MATLAB files", "*.mat"), ("All", "*.*")],
        )
        if path:
            self._load_path(path)

    def _show_channels(self: ECGApp) -> None:
        if not self._filepath:
            messagebox.showwarning("No file", "Open a .mat file first.")
            return
        try:
            messagebox.showinfo("Channels", list_channels(self._filepath))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _load_raw_only(self: ECGApp) -> None:
        """Load the file and display ONLY the raw signal.

        No bandpass/notch/NK-clean, no polarity correction, no detection —
        just the samples read from disk, windowed by the Analysis window
        fields if set, and z-score normalised purely for display scale.

        This is what runs automatically when a file is opened.  The user
        must click '1 ▶ Preview Detection' to run the actual DSP + detection
        pipeline (see _preview / _preview_worker).
        """
        self.signal_ctrl.load_raw_only()

    def _preview(self: ECGApp) -> None:
        """Load, filter, and detect peaks — fast, no HRV."""
        self.signal_ctrl.preview()

    def _preview_worker(self: ECGApp, params: dict) -> dict:
        """Background worker — MUST NOT write to self.

        Loads and processes the signal, then returns a plain data bundle.
        All instance-variable writes happen in ``_on_preview_done`` on the
        main thread, preventing data races with Tk resize/redraw callbacks.
        """
        return self.signal_ctrl.preview_worker(params)

    def _on_preview_done(self: ECGApp, bundle: dict) -> None:
        """Atomically write all signal state on the main thread, then draw.

        This is the ONLY place that should assign signal/peak instance variables
        after a preview.  Because it runs via after(0, …) (scheduled by
        _start_async after the background worker finishes), it is guaranteed to
        execute on the Tk main thread with no concurrent background writes.
        """
        self.signal_ctrl.on_preview_done(bundle)

    def _windowed_peaks(self: ECGApp) -> "Optional[np.ndarray]":
        """Return a copy of _rpeaks_ok filtered to the current analysis window.

        If no window is set (both bounds = 0), returns the full array.
        Returns None if _rpeaks_ok is None.

        This is the single source of truth for all analysis methods
        (_run_freq, _run_nonlinear, _run_intervals, _run_arrhythmia_analysis,
        _compute_epochs, _compute_rolling_hrv) — they all call this instead of
        doing ``self._rpeaks_ok.copy()`` directly.
        """
        return self.signal_ctrl.windowed_peaks()

    def _apply_analysis_window(self: ECGApp) -> None:
        """Read the analysis window entries and store in _analysis_t_start/_end.

        Updates the feedback label with the peak count inside the window.
        Does NOT re-run detection or analysis — the window is applied on
        the next Core Analysis run.
        """
        self.signal_ctrl.apply_analysis_window()

    def _reset_analysis_window(self: ECGApp) -> None:
        """Reset analysis window to full signal."""
        self.signal_ctrl.reset_analysis_window()

    def _run_analysis(self: ECGApp) -> None:
        self.analysis_ctrl.run_analysis()

    def _on_analysis_done(self: ECGApp, bundle: dict) -> None:
        self.analysis_ctrl.on_analysis_done(bundle)


    # ── Per-module on-demand analysis ────────────────────────

    def _check_method_agreement(self: ECGApp) -> None:
        """Cross-check accepted beats against other detection methods."""
        self.detection_ctrl.check_method_agreement()

    def _open_artifact_review(self: ECGApp) -> None:
        """Detect artifact candidates and open the interactive review dialog.

        Detection runs via _start_async_result (background thread + progress
        row + busy-button guard) like Check Agreement and every other
        background op -- a synchronous call with only a button-text swap
        froze repaints on long recordings and never showed the progress bar.
        """
        if self._signal_flt is None or self._rpeaks_ok is None or len(self._rpeaks_ok) < 4:
            messagebox.showwarning("Not ready",
                                   "Click Detect Peaks first to load peaks.")
            return

        rp  = self._rpeaks_ok.copy()
        fs  = self._fs
        sig = self._signal_flt
        rr_min = float(self._safe_float(self.ent_minrr, MouseECG.RR_MIN_MS))
        _, rr_max = self.analysis_ctrl.hrv_rr_bounds()
        # Removals of an earlier report whose peak is still missing stay in
        # effect after this review (merged into the new report below).
        prev_report = self._artifact_report

        artifact_window_beats = 11

        def _worker():
            return detect_rr_artifacts(
                rp, fs,
                rr_min_ms    = rr_min,
                rr_max_ms    = rr_max,
                window_beats = artifact_window_beats,
                dev_threshold= 0.20,
                signal       = sig,
            )

        def _done(candidates) -> None:
            n = len(candidates)
            if n == 0:
                # Inline status only -- matches how every other zero-result
                # background op (e.g. Check Agreement) reports "nothing
                # found" instead of interrupting with a modal.
                self._set_status("No artifacts detected — signal looks clean ✓", GREEN)
                return

            # Count by type for display
            counts = {}
            for c in candidates:
                counts[c["type"]] = counts.get(c["type"], 0) + 1
            detail = "  ·  ".join(f"{v} {k}" for k, v in counts.items())
            self._set_status(f"Reviewing {n} candidates ({detail})", ORANGE)

            dlg = ArtifactReviewDialog(self, sig, rp, fs, candidates, rr_min_ms=rr_min,
                                       window_beats=artifact_window_beats)
            self.wait_window(dlg)   # blocks until dialog closes

            result = dlg.get_result()
            if result is None:
                self._set_status("Artifact review cancelled", MUTED)
                return

            corrected, report = apply_artifact_decisions(rp, result, fs=fs)
            removed = report["n_in"] - report["n_out"]
            # Every removed REAL beat is an HRV break (its spanning interval
            # is excluded, not treated as normal; a removed spurious peak is
            # not) -- this review's and earlier ones still absent from the
            # corrected peaks. The stored report is cumulative (counts too);
            # the status below describes this review's own removals.
            self._rpeaks_ok       = corrected
            self._artifact_report = merge_artifact_reports(
                prev_report, report, corrected,
                self.analysis_ctrl.break_tol_samples(fs))
            self._artifact_candidates = result

            # Clear any manual exclusions that overlapped removed peaks
            if removed > 0:
                removed_samples = {c["sample"] for c in result if c["decision"] == "remove"}
                self._manual_excluded -= removed_samples

            art_str = (f"Artifact review: −{removed} beats "
                       f"(non-physio={report['n_nonphysio']}  "
                       f"ectopic={report['n_ectopic']}  "
                       f"dup={report['n_duplicate']}  "
                       f"kept={report['n_kept']})")

            if removed > 0:
                # Peaks changed → previous HRV metrics are stale.  Discard them so
                # the user cannot export results that don't match the corrected peaks.
                self._results = None
                self._epoch_df = None
                # Also blank the KPI tiles and cached result plots -- every other
                # peak-editing path (detection_controller.py's on_detail_click /
                # apply_edit_state / clear_manual_exclusions / apply_threshold_ui)
                # already does this; without it the Statistics panel and Summary/
                # Poincaré/PSD plots keep showing the pre-review numbers even
                # though analysis.results was just cleared.
                self._reset_kpis()
                self._reset_result_plots()
                self.analysis_ctrl.invalidate_autonomic()
                stale_note = "  ⚠ Click Analyze to update HRV metrics."
                self._set_status(art_str + stale_note, ORANGE)
                # Visual warning on every analysis tab
                warn_text = "⚠  Peaks changed after artifact review — click Analyze"
                for attr in ("lbl_freq_status", "lbl_nonlin_status", "lbl_ivl_status"):
                    lbl = getattr(self, attr, None)
                    if lbl is not None:
                        lbl.configure(text=warn_text, text_color=ORANGE)
                self._set_result_btns_enabled(
                    False, ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl"))
            else:
                self._set_status(art_str, GREEN)

            # Refresh the overview / detail with the cleaned peaks
            self._draw_detail()
            color = GREEN if len(corrected) > 10 else RED
            self.lbl_npeaks.configure(  # type: ignore[union-attr]
                text=f"Peaks detected: {len(corrected)}  (after review)",
                text_color=color)

        self._start_async_result(self.btn_review_art, "Detecting…", _worker, _done)

    def _run_freq(self: ECGApp) -> None:
        """Compute frequency-domain HRV in background, then render."""
        self.analysis_ctrl.run_freq()

    def _run_nonlinear(self: ECGApp) -> None:
        """Compute non-linear HRV in background, then render."""
        self.analysis_ctrl.run_nonlinear()

    def _open_annotations(self: ECGApp) -> None:
        """Open the annotation manager dialog."""
        AnnotationManagerDialog(self)

    def _update_ann_count(self: ECGApp) -> None:
        """Refresh the annotation count badge in the toolbar and the
        right panel's ANNOTATIONS section summary."""
        if self.lbl_ann_count is not None:
            n = len(self._annotations)
            self.lbl_ann_count.configure(  # type: ignore[union-attr]
                text=f"{n}" if n else "",
                text_color=ORANGE if n else MUTED)
        if self.lbl_panel_ann_count is not None:
            n = len(self._annotations)
            self.lbl_panel_ann_count.configure(text=f"{n} annotations")  # type: ignore[union-attr]
        # Called after every annotation add / edit / delete: the Autonomic
        # plot draws them on every panel (no-op without a result); the
        # Event-locked label filter lists their labels, and a result locked
        # to events that changed is dropped.
        self.analysis_ctrl.redraw_autonomic_plot()
        self.event_ctrl.on_annotations_changed()

    def _open_pacing_periods(self: ECGApp) -> None:
        """Open the pacing/stimulation period manager dialog."""
        PacingPeriodManagerDialog(self)

    def _update_pacing_count(self: ECGApp) -> None:
        """Refresh the pacing-period count in the ANNOTATIONS panel section's
        summary (the toolbar's redundant "Pacing Periods" chip was removed --
        this dialog is now reached solely from that panel section)."""
        if self.lbl_panel_pacing_count is not None:
            n = len(self._pacing_periods)
            self.lbl_panel_pacing_count.configure(text=f"{n} pacing periods")  # type: ignore[union-attr]

    def _toggle_rrhr_strip(self: ECGApp) -> None:
        """Show/hide the RR/HR/Quality sub-plot strip below the detail plot.

        Hiding it lets the detail plot reclaim row 1's space (weight=0 makes
        row 0 the only expanding row); showing it restores the 70/30 split.
        """
        self.ui.rrhr_strip_visible = not self.ui.rrhr_strip_visible
        if self.ui.rrhr_strip_visible:
            self.subplot_frame.grid()
            self.plot_area.grid_rowconfigure(1, weight=3)
            self.btn_toggle_rrhr.configure(
                fg_color=PANEL, hover_color=BORDER2, text_color=BLUE, border_width=2)
        else:
            self.subplot_frame.grid_remove()
            self.plot_area.grid_rowconfigure(1, weight=0)
            self.btn_toggle_rrhr.configure(
                fg_color=BORDER, hover_color=BORDER2, text_color=MUTED, border_width=0)

    def _toggle_spike_markers(self: ECGApp) -> None:
        """Show/hide the triangular spike markers on the RR/HR tachogram.

        Leaves the trace, mean line, and ±1 SD band untouched -- only the
        spike-detection layer (markers, dotted threshold guides, the count
        in the title, and the marker legend in the HR x-label) is toggled,
        so switching it off leaves a clean, undecorated raw trace rather
        than orphaned threshold lines with nothing to explain them.
        """
        self.ui.show_spike_markers = not self.ui.show_spike_markers
        if self.ui.show_spike_markers:
            self.btn_toggle_spikes.configure(
                fg_color=PANEL, hover_color=BORDER2, text_color=BLUE, border_width=2)
        else:
            self.btn_toggle_spikes.configure(
                fg_color=BORDER, hover_color=BORDER2, text_color=MUTED, border_width=0)
        if self.analysis.results is not None:
            self.plot_ctrl.plot_rr(self.analysis.results)

    def _toggle_left_panel(self: ECGApp) -> None:
        """Hide/show the left sidebar entirely (simple hide/show, not a rail).

        Collapsing un-packs the panel so the center workspace immediately
        reclaims its width. Restoring it MUST pass before=self.main --
        self.sidebar/self.right_panel are deliberately packed before
        self.main in _build() so main's expand=True doesn't claim the full
        cavity first; a bare .pack() after .pack_forget() re-appends the
        slave to the END of the master's pack order (i.e. after main, which
        has already claimed all the space), silently producing a permanent
        zero-width panel. before=self.main reinserts it at the correct
        position regardless of call order.
        """
        self.ui.left_panel_collapsed = not self.ui.left_panel_collapsed
        if self.ui.left_panel_collapsed:
            self.sidebar.pack_forget()
            self.btn_toggle_left_panel.configure(text="⟩ Sidebar")
        else:
            self.sidebar.pack(side="left", fill="y", before=self.main)
            self.btn_toggle_left_panel.configure(text="⟨ Sidebar")

    def _toggle_right_panel(self: ECGApp) -> None:
        """Hide/show the right panel entirely -- see _toggle_left_panel()'s
        docstring for why before=self.main is required on restore."""
        self.ui.right_panel_collapsed = not self.ui.right_panel_collapsed
        if self.ui.right_panel_collapsed:
            self.right_panel.pack_forget()
            self.btn_toggle_right_panel.configure(text="Stats ⟨")
        else:
            self.right_panel.pack(side="right", fill="y", before=self.main)
            self.btn_toggle_right_panel.configure(text="Stats ⟩")

    def _run_intervals(self: ECGApp) -> None:
        """Compute interval delineation (PR/QRS/QT/QTc) for every beat."""
        self.analysis_ctrl.run_intervals()

    # ════════════════════════════════════════════════════════
    #  DRAWING — overview & detail
    # ════════════════════════════════════════════════════════

    def _compute_filter_preview_segment(
        self: ECGApp, t_start: float, t_end: float,
    ) -> "Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]":
        """Compute a live-filtered preview of the signal over [t_start, t_end].

        Operates ONLY on the visible window (+ a short margin to absorb
        filtfilt edge transients) using the CURRENT filter widget values
        (HP/LP cutoffs, notch, clean method) — never on the full recording.
        This is intentionally cheap: it never touches ``self._signal_flt``
        or any detection state, so it's safe to recompute on every redraw
        without affecting Preview Detection / Run Full Analysis results.

        Returns (t_slice, raw_slice_norm, filt_slice_norm), or None if the
        raw signal isn't loaded yet or the window is too short to filter.
        """
        return self.signal_ctrl.compute_filter_preview_segment(t_start, t_end)

    def _on_filter_preview_toggle(self: ECGApp) -> None:
        """Toggle the before/after filter overlay and redraw."""
        self.signal_ctrl.on_filter_preview_toggle()

    def _refresh_filter_preview(self: ECGApp) -> None:
        """Recompute the filter-preview overlay with current widget values.

        Bound to HP/LP entry <Return>/<FocusOut> and notch/clean-method
        changes — the preview segment isn't auto-reactive to keystrokes,
        only re-evaluated on these discrete commit events, matching how
        the rest of the sidebar (Preview Detection button) already works.
        Also the single choke point for refreshing the "Processing" summary,
        since every FILTER SETTINGS widget is already wired to this method.
        """
        self.signal_ctrl.refresh_filter_preview()
        self._update_filter_summary()

    def _draw_detail(self: ECGApp, t_start: float | None = None, full: bool = True) -> None:
        """Draw the time-windowed detail view with peak markers.

        The active signal (raw or filtered) is drawn at full opacity; the
        other signal is drawn as a ghost at 20 % opacity so the filtering
        effect is always visible.  Peak markers are computed from the filtered
        signal regardless of mode — they are not re-detected on toggle.

        Also handles two extra states, both display-only (no detection state
        is touched):
        • Raw-only (just opened, Preview Detection not yet run): signal_flt
          is None, so only the raw trace is drawn, with no peaks/threshold.
        • Filter preview overlay (self._filter_preview_on): an on-the-fly
          filtered version of the visible window, computed from the current
          filter widget values, overlaid on the raw trace so the user can
          judge filter settings before committing to Preview Detection.

        *full*: when False, skips draw_overview()/draw_detail_rrhr() -- the
        minimap and RR/HR/Quality strip, neither of which reads hover state
        or changes from a hover-only redraw (see detection_controller's
        flush_hover_redraw, the one caller that passes full=False). Both are
        a full fig.clear()+rebuild+constrained-layout-solve each, so paying
        for them on every ~30ms hover tick while in Edit mode was pure waste.
        """
        self.plot_ctrl.draw_detail(t_start)
        if full:
            self.plot_ctrl.draw_overview()
            self.plot_ctrl.draw_detail_rrhr(t_start)

    def _kb_navigate(self: ECGApp, direction: int) -> None:
        """Keyboard left/right arrow navigation — only active on Detection tab."""
        self.nav_ctrl.kb_navigate(direction)

    def _navigate(self: ECGApp, direction: int) -> None:
        """Shift the detail view left/right by 80 % of the window width."""
        self.nav_ctrl.navigate(direction)

    def _navigate_big(self: ECGApp, direction: int) -> None:
        """Jump by 10× the current window width."""
        self.nav_ctrl.navigate_big(direction)

    def _step_spike(self: ECGApp, direction: int) -> None:
        """Jump to the next/previous suspicious RR beat flagged on the HRV tab."""
        self.nav_ctrl.step_spike(direction)

    def _nav_reset(self: ECGApp) -> None:
        self.nav_ctrl.nav_reset()

    def _nav_end(self: ECGApp) -> None:
        """Jump to the end of the signal."""
        self.nav_ctrl.nav_end()

    def _nav_goto(self: ECGApp) -> None:
        """Jump to the time entered in the position field."""
        self.nav_ctrl.nav_goto()

    def _sync_nav_pos_entry(self: ECGApp) -> None:
        """Update the position entry widget to reflect _nav_pos."""
        self.nav_ctrl.sync_nav_pos_entry()

    # ════════════════════════════════════════════════════════
    #  RESULT PLOTS
    # ════════════════════════════════════════════════════════

    def _run_plot_chain(
        self: ECGApp,
        tasks: list,
        on_complete: "Optional[Callable[[], None]]" = None,
        auto_epochs: bool = False,
    ) -> None:
        """Run a list of (label, fn) plot tasks sequentially via after() chain."""
        self.plot_ctrl.run_plot_chain(tasks, on_complete=on_complete, auto_epochs=auto_epochs)

    def _draw_core_results(
        self: ECGApp,
        on_complete: "Optional[Callable[[], None]]" = None,
        auto_epochs: bool = False,
    ) -> None:
        """Render only the fast core plots (RR, Beat, Summary, Poincaré).

        Called immediately after core analysis.  Freq / non-linear / intervals
        are rendered separately when their per-tab buttons are clicked.
        """
        self.plot_ctrl.draw_core_results(on_complete=on_complete, auto_epochs=auto_epochs)

    def _draw_all_results(
        self: ECGApp,
        on_complete: "Optional[Callable[[], None]]" = None,
        auto_epochs: bool = False,
    ) -> None:
        """Render ALL result plots (used by export and legacy callers)."""
        self.plot_ctrl.draw_all_results(on_complete=on_complete, auto_epochs=auto_epochs)

    def _plot_rr(self: ECGApp, r: dict) -> None:
        """Plot RR tachogram, HR trace, and RR distribution histogram.

        Drastic RR changes are detected and shown as orange/red markers on
        the tachogram.  Clicking any point navigates to that beat in Detection.
        Right-clicking jumps specifically to the nearest spike.
        """
        self.plot_ctrl.plot_rr(r)

    def _plot_hrv_tables(self: ECGApp, r: dict) -> None:
        """Populate time-domain and frequency-domain HRV text boxes."""
        self.plot_ctrl.plot_hrv_tables(r)

    def _plot_psd(self: ECGApp, r: dict) -> None:
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
        self.plot_ctrl.plot_psd(r)

    def _plot_nonlinear(self: ECGApp, r: dict) -> None:
        """Poincaré plot and non-linear HRV metric table."""
        self.plot_ctrl.plot_nonlinear(r)

    def _plot_intervals(self: ECGApp, r: dict) -> None:
        """Violin + box plot for PR / QRS / QT / QTc intervals."""
        self.plot_ctrl.plot_intervals(r)

    def _plot_beat_template(self: ECGApp, r: dict) -> None:
        """Average beat template, ±1 SD band, and amplitude / morphology distributions.

        All heavy numpy work (beat matrix, SD, per-beat correlations) was pre-computed
        in analyse_core() on the background thread.  This function only renders.
        """
        self.plot_ctrl.plot_beat_template(r)

    def _plot_summary(self: ECGApp, r: dict) -> None:
        """Populate the Summary tab: KPI cards, all plots, and text report."""
        self.plot_ctrl.plot_summary(r)

    # ── KPI bar update ────────────────────────────────────────

    def _reset_result_plots(self: ECGApp) -> None:
        """Clear stored draw_fn on every result-plot slot.

        Prevents stale draw functions from a previous file replaying
        on window resize after a new file is loaded.
        """
        self.plot_ctrl.reset_result_plots()

    def _reset_tab_status_labels(self: ECGApp) -> None:
        """Reset per-tab status labels and disable action buttons.

        Called on new file load so labels from the previous analysis
        (e.g. "Done LFn=25.0%") don't persist after loading a new file.
        """
        self.plot_ctrl.reset_tab_status_labels()

    def _reset_kpis(self: ECGApp) -> None:
        """Reset all KPI labels to dash when results are invalidated."""
        self.plot_ctrl.reset_kpis()

    def _update_kpis(self: ECGApp) -> None:
        self.plot_ctrl.update_kpis()

    # ════════════════════════════════════════════════════════
    #  EPOCH ANALYSIS
    # ════════════════════════════════════════════════════════

    def _compute_epochs(self: ECGApp) -> None:
        """Compute epoch-level HRV in a background thread to keep the UI responsive.

        nk.hrv_time() is called once per epoch.  On long recordings with many
        short epochs this adds up to several seconds — enough to freeze the UI
        noticeably.  The calculation is therefore moved to a daemon thread via
        _start_async_result, exactly like _run_freq / _run_nonlinear.
        """
        self.analysis_ctrl.compute_epochs()

    def _refresh_epoch_feasibility(self: ECGApp) -> None:
        """Warn ahead of time if the current Epoch (s) can't fit the
        recording -- called when the Epochs sub-view is shown, see
        _on_hrv_view_change."""
        self.analysis_ctrl.refresh_epoch_feasibility()

    # ════════════════════════════════════════════════════════
    #  EXPORT
    # ════════════════════════════════════════════════════════

    def _export_excel(self: ECGApp) -> None:
        self.export_ctrl.export_excel()

    def _export_zip(self: ECGApp) -> None:
        self.export_ctrl.export_zip()

    def _export_pdf_report(self: ECGApp) -> None:
        """Generate a one-page PDF summary: ECG strip + KPI table + interpretation."""
        self.export_ctrl.export_pdf_report()

    def _export_prism(self: ECGApp) -> None:
        """Export all analysis results to a GraphPad Prism .pzfx file."""
        self.export_ctrl.export_prism()


    def _add_recent(self: ECGApp, path: str) -> None:
        self.session_ctrl.add_recent(path)

    def _open_recent(self: ECGApp) -> None:
        """Show recent recordings from SQLite registry with summary stats."""
        db_rows = recent_recordings(limit=20) if _DB_AVAILABLE else []
        db_paths = {r["filepath"] for r in db_rows}
        extra = [p for p in self._recent if p not in db_paths and os.path.exists(p)]

        if not db_rows and not extra:
            messagebox.showinfo("Recent files", "No recent recordings yet.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Recent recordings")
        win.geometry("740x480")
        win.configure(fg_color=PANEL)
        win.grab_set(); win.lift()

        hdr = ctk.CTkFrame(win, fg_color=PANEL, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="Recent recordings", font=FONT_CARD_TITLE,
                     text_color=TEXT, anchor="w").pack(side="left", padx=SPACE_L, pady=SPACE_M)
        ctk.CTkButton(hdr, text="📈 Cohort Trends", height=28, corner_radius=6,
                      fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                      font=FONT_BTN_SEC, command=self._open_cohort_trends
                      ).pack(side="right", padx=SPACE_L, pady=SPACE_M)

        scroll = ctk.CTkScrollableFrame(win, fg_color=PANEL)
        scroll.pack(fill="both", expand=True, padx=SPACE_M, pady=SPACE_M)

        def _entry(path: str, hr: str = "—", sdnn: str = "—",
                   dur: str = "—", notes: str = "",
                   sparkline: "Optional[np.ndarray]" = None) -> None:
            if not os.path.exists(path):
                return
            card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=6)
            card.pack(fill="x", pady=(0, SPACE_S))
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
            ctk.CTkButton(top, text=os.path.basename(path), anchor="w",
                          fg_color="transparent", hover_color=BORDER,
                          text_color=BLUE, font=FONT_SMALL,
                          command=lambda p=path: (win.destroy(), self._load_path(p))
                          ).pack(side="left", fill="x", expand=True)
            lbl_stats = ctk.CTkLabel(top, text=f"HR {hr}  SDNN {sdnn}  {dur}",
                                      font=FONT_KPI_LABEL, text_color=MUTED)
            lbl_stats.pack(side="right")
            if sdnn == "—":
                # SDNN wasn't cached for older registry rows -- explain the
                # dash instead of leaving it looking like a missing analysis.
                self._bind_hover_tip(
                    lbl_stats,
                    "SDNN not cached for this recording — open and save it again to backfill",
                )
            # Falls back to the old full-session-deserialize path only for
            # rows with nothing cached yet (pre-existing registry rows
            # saved before rr_sparkline existed, or paths outside the
            # SQLite registry entirely) -- self-heals to the cheap cached
            # path the next time each recording is saved.
            spark_vals = sparkline if sparkline is not None else load_rr_series(path)
            make_sparkline(top, spark_vals, width=80, height=22,
                           color=BLUE).pack(side="right", padx=(0, 10))
            if notes:
                ctk.CTkLabel(card, text=f"📝 {notes[:90]}",
                             font=FONT_KPI_LABEL, text_color=LIGHT,
                             anchor="w").pack(padx=SPACE_M, pady=(0, SPACE_S), fill="x")

        for r in db_rows:
            _spark = None
            _spark_raw = r.get("rr_sparkline")
            if _spark_raw:
                try:
                    _spark = np.asarray(json.loads(_spark_raw), dtype=float)
                except Exception as _exc:
                    log.debug("recent-recordings: bad cached rr_sparkline for %s: %s",
                              r["filepath"], _exc)
            _entry(
                r["filepath"],
                hr=f"{r['hr_mean']:.0f} bpm" if r.get("hr_mean") else "—",
                sdnn=f"{r['sdnn']:.1f} ms"   if r.get("sdnn") else "—",
                dur=f"{r['duration_s']:.0f} s" if r.get("duration_s") else "",
                notes=r.get("notes", ""),
                sparkline=_spark,
            )
        for p in extra:
            _entry(p)

    def _open_cohort_trends(self: ECGApp) -> None:
        """Open the cross-recording HR/SDNN/RMSSD trend view.

        Unlike the Recent-recordings popup (capped at 20, most-recent-
        first), this pulls the whole registry -- a cohort view is only
        useful if it isn't quietly dropping older recordings.
        """
        from ecg.ui.dialogs import CohortTrendsDialog
        rows = recent_recordings(limit=500) if _DB_AVAILABLE else []
        CohortTrendsDialog(self, rows)


    def _load_path(self: ECGApp, path: str) -> None:
        self.signal_ctrl.load_path(path)

    def _reset_for_new_file(self: ECGApp) -> None:
        """Reset ALL analysis state and UI to the startup blank slate.

        Called every time a new file is opened so there is zero carry-over
        from the previous recording.  Sidebar *parameter* widgets (channel,
        fs, thresholds, filters) are intentionally kept — users typically
        want the same settings for consecutive recordings from the same rig.

        IMPORTANT: must NOT call _init_state() — that method also zeroes out
        all widget-reference attributes (ent_channel, ent_fs, …) which are
        already live in the UI, causing AttributeError on the next snapshot.
        Only data variables are reset here.
        """
        self.signal_ctrl.reset_for_new_file()

    def _try_restore_session(self: ECGApp, path: str) -> None:
        """If a saved session exists for *path*, offer to restore it.

        Fully async, including the session-file check itself -- see
        SessionController.try_restore_session for why. It decides on its
        own whether to restore (via _start_async) or fall through to
        _load_raw_only(); there's no synchronous return value to branch on.
        """
        self.session_ctrl.try_restore_session(path)

    def _restore_session_worker(self: ECGApp, state: dict) -> dict:
        """Background worker — MUST NOT write to self or touch any Tkinter widget.

        Reloads the raw signal from the original .mat file and re-runs the
        pure ``_compute_preview_bundle`` static method with the saved filter
        parameters. Returns a plain bundle; ``_on_restore_session_done``
        writes every field to ``self`` (and to widgets) on the main thread.

        Session restore previously ran entirely in the background thread
        (including dozens of widget .configure/.delete/.insert calls and
        _draw_detail()), which is unsafe — Tkinter/Tcl objects must only be
        touched from the main thread. This mirrors the _preview_worker /
        _on_preview_done split used elsewhere in the app.
        """
        return self.session_ctrl.restore_session_worker(state)

    def _on_restore_session_done(self: ECGApp, bundle: dict, saved_at: str) -> None:
        """Write all restored state to self and to widgets — main thread only.

        Counterpart to ``_restore_session_worker``. This is the ONLY place
        that writes session-restore state to ``self``/widgets, exactly as
        ``_on_preview_done`` is the sole writer after ``_preview_worker``.
        """
        self.session_ctrl.on_restore_session_done(bundle, saved_at)

    # ════════════════════════════════════════════════════════
    #  SESSION SAVE / RESTORE
    # ════════════════════════════════════════════════════════

    def _collect_session_state(self: ECGApp) -> dict:
        """Gather all serialisable app state into a flat dict.

        Signal arrays (signal_flt, signal_raw_norm, all_cands, all_proms) are
        intentionally NOT stored.  They are derived entirely from the .mat file
        and the filter parameters; _restore_session_worker re-runs
        _compute_preview_bundle to reconstruct them in < 2 s.  This keeps session
        files small (< 200 KB for a typical recording) regardless of recording length.

        Previously (v3) the session stored the full filtered signal as a Python
        list (~58 MB for 10 min at 2 kHz, ~346 MB for 1 h), making auto-save
        impractical and restore slow.
        """
        return self.session_ctrl.collect_session_state()

    def _save_session(self: ECGApp) -> None:
        """Serialise full analysis state to a .ecgsession cache file and update registry."""
        self.session_ctrl.save_session()

    def _delete_session(self: ECGApp) -> None:
        """Delete the session cache file for the current file."""
        self.session_ctrl.delete_session()

    def _update_session_ui(self: ECGApp, has_session: bool,
                           saved_at: str = "") -> None:
        """Update the session info label and button states."""
        self.session_ctrl.update_session_ui(has_session, saved_at=saved_at)

    # ════════════════════════════════════════════════════════
    #  WAVE TEMPLATE EDITOR
    # ════════════════════════════════════════════════════════

    def _open_wave_template_editor(self: ECGApp) -> None:
        """Open the interactive P/Q/R/S/T template editor.

        If a mean beat is available it is passed to the editor for display;
        otherwise the editor shows a synthetic reference trace.
        """
        beat_time = None
        mean_beat = None
        beat_sd   = None
        if self._results is not None:
            beat_time = self._results.get("beat_time")
            mean_beat = self._results.get("beat_template")
            beat_sd   = self._results.get("beat_sd")

        intervals = (self._results.get("intervals")
                     if self._results is not None else None)
        editor = WaveTemplateMiniEditor(
            self, self.analysis_ctrl.recording_wave_template(),
            beat_time=beat_time,
            mean_beat=mean_beat,
            beat_sd=beat_sd,
            intervals=intervals,
        )
        self.wait_window(editor)  # modal — blocks until closed

        # Refresh the label whether or not the user saved
        self._update_session_ui(
            has_session=session_exists(self._filepath)
            if self._filepath else False
        )
        if editor._saved:
            self._session_dirty = True     # the template is kept in the session
            self._set_status(
                "Wave template applied to this recording — re-run Interval "
                "Delineation to apply changes.", GREEN)

    # ════════════════════════════════════════════════════════
    #  CLIPBOARD / TXT SAVE
    # ════════════════════════════════════════════════════════

    def _copy_summary(self: ECGApp) -> None:
        self.export_ctrl.copy_summary()

    def _save_summary_txt(self: ECGApp) -> None:
        self.export_ctrl.save_summary_txt()

    # ════════════════════════════════════════════════════════
    #  1. KEYBOARD SHORTCUTS
    # ════════════════════════════════════════════════════════

    def _typing_in_entry(self: ECGApp) -> bool:
        """True if Tk focus is currently inside a text-entry/textbox widget.

        Root-window bindings (self.bind(...)) propagate to every child
        regardless of focus, so without this guard, typing an ordinary
        space in "Project name" or an annotation label -- or pressing the
        Left/Right arrow keys to move the text cursor -- fired the app's
        global Space/arrow shortcuts instead: re-running full detection
        (silently discarding manual peak edits, annotations, and pacing
        periods) or shifting the plotted time window under the field being
        edited. CTkEntry/CTkTextbox are themselves composite widgets; the
        actual keyboard-receiving child Tk hands back from focus_get() is
        a plain tk.Entry ("Entry") or tk.Text ("Text").
        """
        w = self.focus_get()
        return w is not None and w.winfo_class() in ("Entry", "Text")

    def _bind_keyboard_shortcuts(self: ECGApp) -> None:
        """Register global keyboard shortcuts."""
        def _guarded(fn):
            return lambda e: None if self._typing_in_entry() else fn()

        bindings: "list[tuple[str, Any]]" = [
            ("<space>",       _guarded(self._preview)),
            ("<Control-r>",   lambda e: self._run_analysis()),
            ("<Control-o>",   lambda e: self._open_file()),
            ("<Control-s>",   lambda e: self._save_session()),
            ("<Control-e>",   lambda e: self._export_excel()),
            # Ctrl+W collides with the near-universal OS/browser "close
            # window" shortcut -- rebound to Ctrl+Shift+E (an export-domain
            # variant of Ctrl+E, Export Excel) so an instinctive Ctrl+W
            # doesn't silently export a CSV instead of doing what the user
            # expected.
            ("<Control-Shift-E>", lambda e: self._export_rr_csv()),
            ("<Control-m>",   lambda e: self._open_compare_segments()),
            ("<F1>",          lambda e: self._show_shortcuts_help()),
        ]
        for seq, cb in bindings:
            try:
                self.bind(seq, cb)
            except Exception as exc:
                log.debug("bind %s: %s", seq, exc)

    def _show_shortcuts_help(self: ECGApp) -> None:
        shortcuts = (
            "Space          Detect Peaks\n"
            "Ctrl+R         Analyze\n"
            "Ctrl+O         Open .mat file\n"
            "Ctrl+S         Save session\n"
            "Ctrl+E         Export Excel\n"
            "Ctrl+Shift+E   Export RR intervals CSV\n"
            "Ctrl+M         Compare segments\n"
            "Ctrl+Z / Y     Undo / Redo peak edits\n"
            "←  / →         Move the ECG view (Detection tab)\n"
            "F1             This help\n"
        )
        messagebox.showinfo("Keyboard shortcuts", shortcuts)
    # ════════════════════════════════════════════════════════
    #  3. EXPORT RR INTERVALS AS CSV
    # ════════════════════════════════════════════════════════

    def _export_rr_csv(self: ECGApp) -> None:
        """Export RR intervals to a lightweight CSV (no Excel dependency)."""
        self.export_ctrl.export_rr_csv()

    # ════════════════════════════════════════════════════════
    #  4. SIGNAL QUALITY BADGE (persistent header)
    # ════════════════════════════════════════════════════════

    def _update_quality_badge(self: ECGApp) -> None:
        """Update the persistent quality badge in the KPI bar."""
        self.detection_ctrl.update_quality_badge()

    # ════════════════════════════════════════════════════════
    #  7. RECORDING NOTES (per-file, saved in SQLite)
    # ════════════════════════════════════════════════════════

    def _open_notes_dialog(self: ECGApp) -> None:
        """Open a small dialog for free-text notes on the current recording."""
        if not self._filepath:
            messagebox.showinfo("Notes", "Load a file first.")
            return
        current = get_notes(self._filepath) if _DB_AVAILABLE else self._recording_notes

        win = ctk.CTkToplevel(self)
        win.title(f"Notes — {os.path.basename(self._filepath)}")
        win.geometry("500x340")
        win.configure(fg_color=BG)
        win.grab_set(); win.lift()

        ctk.CTkLabel(win, text="📝  Experiment notes",
                     font=FONT_SIDEBAR_HDR, text_color=TEXT,
                     anchor="w").pack(padx=SPACE_L, pady=(SPACE_L, SPACE_S), fill="x")
        ctk.CTkLabel(win, text="Saved per recording (animal ID, drug, dose, time…)",
                     font=FONT_KPI_LABEL, text_color=MUTED, anchor="w").pack(
            padx=SPACE_L, pady=(0, SPACE_S), fill="x")
        txt = ctk.CTkTextbox(win, fg_color=CARD, text_color=TEXT,
                             font=FONT_LABEL, border_width=1,
                             border_color=BORDER2, height=180)
        txt.pack(fill="both", expand=True, padx=SPACE_L, pady=(0, SPACE_M))
        txt.insert("1.0", current)

        def _save():
            notes = txt.get("1.0", "end").strip()
            self._recording_notes = notes
            if _DB_AVAILABLE and self._filepath:
                set_notes(self._filepath, notes)
            self._set_status("Notes saved ✓", GREEN)
            win.destroy()

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=SPACE_L, pady=(0, SPACE_L))
        ctk.CTkButton(btn_row, text="✔  Save", command=_save,
                      fg_color=GREEN, hover_color=GREEN_DARK, text_color="white",
                      font=FONT_BTN_PRIMARY, height=30, corner_radius=6).pack(side="left")
        ctk.CTkButton(btn_row, text="✗ Cancel", command=win.destroy,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FONT_BTN_SEC, height=30, corner_radius=6).pack(
            side="left", padx=(SPACE_M, 0))

    # ════════════════════════════════════════════════════════
    #  10. ARRHYTHMIA EPISODE PDF (one strip per episode)
    # ════════════════════════════════════════════════════════

    def _export_arrhythmia_pdf(self: ECGApp) -> None:
        """Export a PDF with one annotated ECG strip per arrhythmia episode."""
        self.export_ctrl.export_arrhythmia_pdf()
