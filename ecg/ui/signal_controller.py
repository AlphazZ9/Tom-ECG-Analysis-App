# -*- coding: utf-8 -*-
"""
signal_controller.py
----------------------
SignalController -- loading raw signals, filtering/polarity/peak-candidate
computation (the pure DSP pipeline), the raw-only and full preview flows
(background worker + main-thread state writer pairs), the analysis-window
crop, and the live filter-preview overlay.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Optional
from tkinter import messagebox

import numpy as np

from ecg.core.models import MouseECG
from ecg.core.filtering import bandpass, notch, normalize
from ecg.core.detection import (
    fix_polarity, apply_threshold,
    detect_peaks_sg_derivative, detect_peaks_wavelet, detect_peaks_envelope_max,
)
from ecg.core.ml_detector import detect_peaks_ml, MLPeakModel
from ecg.io.loaders import load_mat_signal
from ecg.io.db import _DB_AVAILABLE, get_notes, get_recording
from ecg.ui.theme import BLUE, GREEN, MUTED, ORANGE, RED, nk

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class SignalController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    @staticmethod
    def compute_preview_bundle(
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
        def _prog(pct: int, msg: str) -> None:
            if progress_cb:
                progress_cb(pct, msg)

        no_filter = params.get("no_filter", False)
        sig = sig_raw.copy()

        _prog(5, "Normalising raw signal…")
        signal_raw_norm = normalize(sig)

        if no_filter:
            log.info("_compute_preview_bundle: no_filter=True — skipping bandpass/notch/ecg_clean")
        else:
            _prog(10, "Bandpass filtering…")
            try:
                sig = bandpass(sig, fs, params["lp"], params["hp"])
            except Exception as exc:
                log.warning("bandpass skipped: %s", exc)

            if params["notch"]:
                _prog(25, "Notch filtering…")
                try:
                    sig = notch(sig, fs)
                except Exception as exc:
                    log.warning("notch skipped: %s — check fs vs notch frequency", exc)

            _prog(35, "NeuroKit2 clean…")
            try:
                assert nk is not None  # NK_AVAILABLE guard checked by caller
                sig = nk.ecg_clean(sig, sampling_rate=fs,
                                   method=params["clean_method"])
            except Exception as exc:
                log.warning("nk.ecg_clean skipped: %s", exc)

        _prog(45, "Normalising filtered signal…")
        sig = normalize(np.asarray(sig, dtype=float))

        # ── Manual polarity override ─────────────────────────────────────
        # If the user has explicitly toggled "Inverser signal", flip the
        # signal before auto-polarity detection.  fix_polarity will then
        # find the (now correct) positive R peaks and not re-flip.
        if params.get("invert_signal", False):
            sig = -sig
            log.info("_compute_preview_bundle: user-requested signal inversion applied")

        def _polarity_prog(pct: int, msg: str) -> None:
            _prog(45 + int(pct * 0.50), msg)

        # ── Detection method ─────────────────────────────────────────────
        det_method = params.get("detection_method", "auto")

        if "wavelet" in det_method.lower() or "cwt" in det_method.lower():
            # ── Wavelet (CWT) pipeline ────────────────────────────────────
            # fix_polarity AVANT la détection : le détecteur wavelet (comme SG)
            # travaille sur la dérivée positive → il faut que les R-peaks soient
            # positifs dans le signal, sinon seuls les artefacts positifs sont détectés.
            #
            # NOTE: unlike the SG-derivative/Envelope-Max/ML branches below,
            # this call is NOT guarded by `if not params.get("invert_signal")`
            # -- it always runs, even if the user already inverted manually.
            # That asymmetry is intentional-or-unconfirmed (not verified
            # against real double-inverted recordings) -- don't "fix" it by
            # assuming it should match the other three methods without
            # testing what Wavelet detection actually does on a
            # user-already-inverted signal first.
            _prog(48, "Polarity correction…")
            sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"])
            _prog(50, "CWT — séparation bruit / QRS / J-wave…")
            try:
                peaks_wt, proms_wt, t_amp_wt = detect_peaks_wavelet(
                    sig,
                    fs=fs,
                    min_rr_ms=params["min_rr_ms"],
                    peak_distance_ms=params.get("peak_distance_ms", MouseECG.PEAK_DISTANCE_MS),
                )
            except ImportError:
                log.warning(
                    "PyWavelets non installé — "
                    "pip install PyWavelets  (fallback: auto)"
                )
                peaks_wt  = np.array([], dtype=int)
                proms_wt  = np.array([])
                t_amp_wt  = 0.0
            except Exception as exc:
                log.warning("Wavelet detection failed, falling back to auto: %s", exc)
                peaks_wt  = np.array([], dtype=int)
                proms_wt  = np.array([])
                t_amp_wt  = 0.0

            _prog(95, f"Wavelet candidates: {len(peaks_wt):,}")
            return {
                "signal_raw_norm": signal_raw_norm,
                "signal_flt":      sig,
                "no_filter_mode":  no_filter,
                "all_cands":       peaks_wt,
                "all_proms":       proms_wt if len(proms_wt) else np.ones(len(peaks_wt)),
                "inverted":        inverted,
            }

        if "sg" in det_method.lower() or "deriv" in det_method.lower():
            # ── SG + Derivative pipeline ──────────────────────────────────
            # fix_polarity automatique si l'utilisateur n'a pas déjà inversé
            # manuellement (évite la double inversion). Le détecteur SG
            # requiert des R positifs (upstroke = dérivée positive).
            if not params.get("invert_signal", False):
                _prog(48, "Polarity correction…")
                sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"])
            else:
                inverted = False  # utilisateur a géré la polarité manuellement
            _prog(50, "SG derivative detection…")
            try:
                sg_window_ms = float(params.get("sg_window_ms", 20.0))
            except (TypeError, ValueError):
                sg_window_ms = 20.0

            try:
                peaks_sg, proms_sg, t_amp_sg = detect_peaks_sg_derivative(
                    sig,
                    fs=fs,
                    sg_window_ms=sg_window_ms,
                    min_rr_ms=params["min_rr_ms"],
                    peak_distance_ms=params.get("peak_distance_ms", MouseECG.PEAK_DISTANCE_MS),
                    target_fs=float(params.get("sg_target_fs", 10000)),
                )
            except Exception as exc:
                log.warning("SG+derivative detection failed, falling back to auto: %s", exc)
                peaks_sg = np.array([], dtype=int)
                proms_sg = np.array([])
                t_amp_sg = 0.0

            _prog(95, f"SG+Deriv candidates: {len(peaks_sg):,}")
            return {
                "signal_raw_norm": signal_raw_norm,
                "signal_flt":      sig,
                "no_filter_mode":  no_filter,
                "all_cands":       peaks_sg,
                "all_proms":       proms_sg if len(proms_sg) else np.ones(len(peaks_sg)),
                "inverted":        False,
            }

        if "envelope" in det_method.lower() or "max" in det_method.lower():
            # ── Envelope Max pipeline ─────────────────────────────────────
            # Détection par maximum local d'amplitude — robuste aux signaux
            # saturés (ADC clipping) et aux morphologies atypiques où la
            # dérivée SG est peu discriminante.
            # fix_polarity requis : le détecteur sélectionne les maxima → les
            # R-peaks doivent être des extrema positifs dans le signal.
            if not params.get("invert_signal", False):
                _prog(48, "Polarity correction…")
                sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"])
            else:
                inverted = False
            _prog(55, "Envelope Max detection…")
            try:
                peaks_em, proms_em, t_amp_em = detect_peaks_envelope_max(
                    sig,
                    fs=fs,
                    min_rr_ms=params["min_rr_ms"],
                    peak_distance_ms=params.get("peak_distance_ms", MouseECG.PEAK_DISTANCE_MS),
                )
            except Exception as exc:
                log.warning("Envelope Max detection failed, falling back to auto: %s", exc)
                peaks_em = np.array([], dtype=int)
                proms_em = np.array([])
                t_amp_em = 0.0

            _prog(95, f"Envelope Max candidates: {len(peaks_em):,}")
            return {
                "signal_raw_norm": signal_raw_norm,
                "signal_flt":      sig,
                "no_filter_mode":  no_filter,
                "all_cands":       peaks_em,
                "all_proms":       proms_em if len(proms_em) else np.ones(len(peaks_em)),
                "inverted":        inverted,
            }

        if "ml" in det_method.lower() or "machine" in det_method.lower():
            # ── ML Detector pipeline ──────────────────────────────────────
            # Classifier-based detection, trained from the user's own
            # verified/corrected recordings (see ecg.core.ml_detector).
            # Same polarity requirement as the other detectors: features
            # (prominence, amplitude, upstroke slope...) all assume the
            # R-wave is a positive-going deflection.
            if not params.get("invert_signal", False):
                _prog(48, "Polarity correction…")
                sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"])
            else:
                inverted = False
            detector_warning: Optional[str] = None

            def _ml_prog(pct: int, msg: str) -> None:
                # Rescale detect_peaks_ml's own 0-100 stage progress into the
                # 55-95 slice of the overall preview-bundle progress bar, so
                # its checkpoints (crucially, the candidate count) actually
                # reach the UI instead of sitting on one static message for
                # however long candidate generation + feature extraction take.
                _prog(55 + int(pct * 0.40), msg)

            try:
                model = MLPeakModel.load()
                peaks_ml, proms_ml, t_amp_ml = detect_peaks_ml(sig, fs, model, progress_cb=_ml_prog)
            except RuntimeError as exc:
                # No trained model yet -- a clearly different situation from
                # "detection failed on this signal", surfaced distinctly so
                # the user isn't left guessing why 0 peaks were found.
                log.warning("ML detector: %s", exc)
                detector_warning = str(exc)
                peaks_ml = np.array([], dtype=int)
                proms_ml = np.array([])
                t_amp_ml = 0.0
            except Exception as exc:
                log.warning("ML detection failed, falling back to auto: %s", exc)
                peaks_ml = np.array([], dtype=int)
                proms_ml = np.array([])
                t_amp_ml = 0.0

            _prog(95, f"ML Detector candidates: {len(peaks_ml):,}")
            return {
                "signal_raw_norm": signal_raw_norm,
                "signal_flt":      sig,
                "no_filter_mode":  no_filter,
                "all_cands":       peaks_ml,
                "all_proms":       proms_ml if len(proms_ml) else np.ones(len(peaks_ml)),
                "inverted":        inverted,
                "detector_warning": detector_warning,
            }

        # ── Auto (NeuroKit2) pipeline — original path ─────────────────────
        sig_out, inverted, cands, proms = fix_polarity(
            sig, fs, params["min_rr_ms"], progress_cb=_polarity_prog)

        _prog(95, f"Candidates found: {len(cands):,}")

        return {
            "signal_raw_norm": signal_raw_norm,
            "signal_flt":      sig_out,
            "no_filter_mode":  no_filter,
            "all_cands":       cands,
            "all_proms":       proms,
            "inverted":        inverted,
        }

    def prepare_signal(
        self,
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
        if self.app.signal.raw is None:
            return
        bundle = self.compute_preview_bundle(
            self.app.signal.raw, self.app.signal.fs, params, progress_cb)
        self.app.signal.raw_norm  = bundle["signal_raw_norm"]
        self.app.signal.filtered       = bundle["signal_flt"]
        self.app.signal.no_filter_mode   = bundle["no_filter_mode"]
        self.app.detection.all_candidates        = bundle["all_cands"]
        self.app.detection.all_prominences        = bundle["all_proms"]
        self.app.signal.inverted  = bundle.get("inverted", False)

    def load_raw_only(self) -> None:
        """Load the file and display ONLY the raw signal.

        No bandpass/notch/NK-clean, no polarity correction, no detection —
        just the samples read from disk, windowed by the Analysis window
        fields if set, and z-score normalised purely for display scale.

        This is what runs automatically when a file is opened.  The user
        must click '1 ▶ Preview Detection' to run the actual DSP + detection
        pipeline (see _preview / _preview_worker).
        """
        if not self.app.signal.filepath:
            return
        params = self.app._snapshot_params()
        self.app._start_async(
            self.app.btn_preview, "Loading…", "Loading raw signal…",
            lambda: self.load_raw_worker(params),
            self.on_raw_load_done,
            pass_result=True,
        )

    def load_raw_worker(self, params: dict) -> dict:
        """Background worker — loads the file, returns the RAW signal only.

        Deliberately mirrors only the first few steps of _preview_worker
        (file read + time-window crop). It stops before any filtering,
        polarity correction, or detection call.
        """
        if self.app.signal.filepath is None:
            raise ValueError("No file loaded.")

        def _prog(pct: int, msg: str) -> None:
            self.app.after(0, lambda p=pct, m=msg: self.app._set_progress(p, m))

        _prog(10, "Loading signal from file…")
        sig, detected_ch, _, detected_fs = load_mat_signal(
            self.app.signal.filepath, params["channel"])

        fs = int(detected_fs) if detected_fs is not None else params["fs"]

        t0 = params["t_start"]
        t1 = params["t_end"]
        i0 = int(t0 * fs) if t0 > 0 else 0
        i1 = int(t1 * fs) if t1 > 0 else len(sig)
        sig = sig[i0:i1]

        if sig.std() < 1e-10:
            raise ValueError("Signal is flat — wrong channel.")

        n_samples = len(sig)
        dur_s     = n_samples / fs
        _prog(70, f"Signal loaded — {dur_s:.0f} s  ({n_samples:,} samples)")

        # ONLY processing step: z-score normalisation for display scale.
        # This is not a "filter" — it never changes morphology, polarity,
        # or which sample is the maximum; it's purely an axis convenience,
        # identical to what "Show raw signal" already displayed pre-preview.
        signal_raw_norm = normalize(sig)
        _prog(100, "Raw signal ready")

        return {
            "fs":           fs,
            "signal_raw":   sig,
            "signal_raw_norm": signal_raw_norm,
            "dur_s":        dur_s,
            "detected_ch":  detected_ch,
            "detected_fs":  detected_fs,
            "requested_ch": params["channel"],
            "fs_from_file": detected_fs is not None,
        }

    def on_raw_load_done(self, bundle: dict) -> None:
        """Write raw-only state on the main thread and draw the raw trace.

        Mirrors the bookkeeping parts of _on_preview_done (channel/fs
        feedback, KPI reset, status message) but explicitly leaves every
        detection/filtering field at None — signal_flt, all_cands,
        rpeaks_ok, thresh_amp, etc. — so the rest of the app's existing
        "not previewed yet" guards (which already check for None) behave
        correctly until the user clicks Preview Detection.
        """
        fs        = bundle["fs"]
        sig_raw   = bundle["signal_raw"]
        n_samples = len(sig_raw)

        self.app.signal.fs               = fs
        self.app.signal.raw       = sig_raw
        self.app.signal.raw_norm  = bundle["signal_raw_norm"]
        self.app.signal.filtered       = None
        self.app.signal.time             = np.arange(n_samples) / fs
        self.app.detection.all_candidates        = None
        self.app.detection.all_prominences        = None
        self.app.signal.no_filter_mode   = True
        self.app.signal.inverted  = False
        self.app.signal.raw_only_loaded  = True
        self.app.ui.nav_pos          = 0.0
        self.app.detection.rpeaks_ok            = None
        self.app.detection.rpeaks_rej           = None
        self.app.detection.thresh_amp           = 0.0
        self.app.detection.rpeaks_manual_excl   = np.array([], dtype=int)
        self.app.detection.rpeaks_manual_added  = np.array([], dtype=int)
        self.app.detection.manual_excluded.clear()
        self.app.detection.manual_added.clear()
        self.app.analysis.results       = None
        self.app.analysis.epoch_df      = None
        self.app.analysis.annotations   = []
        self.app.analysis.pacing_periods = []
        self.app.analysis.wave_template = None
        self.app.session.dirty = False
        self.app._generation    = getattr(self.app, "_generation", 0) + 1
        self.app._reset_kpis()
        self.app._reset_result_plots()
        self.app._reset_tab_status_labels()

        if bundle["detected_ch"] != bundle["requested_ch"]:
            if self.app.ent_channel is not None:
                try:
                    self.app.ent_channel.delete(0, "end")
                    self.app.ent_channel.insert(0, bundle["detected_ch"])
                except Exception as _exc:
                    log.debug("Could not update channel entry: %s", _exc)
            self.app.lbl_file.configure(  # type: ignore[union-attr]
                text=f"Auto: {bundle['detected_ch']}", text_color=ORANGE)
        if bundle["fs_from_file"]:
            self.apply_detected_fs(bundle["detected_fs"])
        else:
            try:
                self.app.lbl_fs_source.configure(
                    text="Tip: fs not found in file — set manually above",
                    text_color=ORANGE)
            except Exception as e:
                log.debug("lbl_fs_source configure failed: %s", e)

        if self.app.lbl_npeaks is not None:
            self.app.lbl_npeaks.configure(  # type: ignore[union-attr]
                text="Peaks detected: — (click Preview Detection)", text_color=MUTED)
        if self.app.btn_review_art is not None:
            self.app.btn_review_art.configure(state="disabled")  # type: ignore[union-attr]

        dur = bundle["dur_s"]
        self.app._set_status(
            f"Raw signal loaded — {dur:.0f} s  |  {fs} Hz  "
            "→ click '1 ▶ Preview Detection' to filter and detect peaks.", BLUE)
        self.app.tabs.set("📈 Detection")
        self.app._update_ann_count()
        self.app._update_pacing_count()
        self.app.ui.nav_pos = 0.0
        self.app._sync_nav_pos_entry()
        if self.app.lbl_sig_duration is not None:
            self.app.lbl_sig_duration.configure(  # type: ignore[union-attr]
                text=f"durée totale : {dur:.1f} s", text_color=MUTED)
        self.app._draw_detail()

        self.app.analysis.t_start = 0.0
        self.app.analysis.t_end   = 0.0
        if self.app.lbl_analysis_window is not None:
            self.app.lbl_analysis_window.configure(  # type: ignore[union-attr]
                text=f"Raw signal loaded  ·  {dur:.1f} s  ·  not yet analysed",
                text_color=MUTED)

    def preview(self) -> None:
        """Load, filter, and detect peaks — fast, no HRV."""
        if not self.app.signal.filepath:
            messagebox.showwarning("No file", "Open a .mat file first.")
            return
        # Snapshot widget values on the main thread before spawning background work.
        params = self.app._snapshot_params()
        # pass_result=True: worker returns a bundle; _on_preview_done receives it.
        # This ensures ALL shared state writes happen on the main thread only.
        self.app._start_async(
            self.app.btn_preview, "Loading…", "Loading signal…",
            lambda: self.preview_worker(params),
            self.on_preview_done,
            pass_result=True,
        )

    def preview_worker(self, params: dict) -> dict:
        """Background worker — MUST NOT write to self.

        Loads and processes the signal, then returns a plain data bundle.
        All instance-variable writes happen in ``_on_preview_done`` on the
        main thread, preventing data races with Tk resize/redraw callbacks.
        """
        if self.app.signal.filepath is None:
            raise ValueError("No file loaded.")

        def _prog(pct: int, msg: str) -> None:
            """Thread-safe progress update — schedules on the main thread."""
            self.app.after(0, lambda p=pct, m=msg: self.app._set_progress(p, m))

        _prog(2, "Loading signal from file…")
        sig, detected_ch, _, detected_fs = load_mat_signal(
            self.app.signal.filepath, params["channel"])

        # Determine effective fs
        if detected_fs is not None:
            fs = int(detected_fs)
        else:
            fs = params["fs"]

        # Estimation automatique min_rr supprimée.

        t0 = params["t_start"]
        t1 = params["t_end"]
        i0 = int(t0 * fs) if t0 > 0 else 0
        i1 = int(t1 * fs) if t1 > 0 else len(sig)
        sig = sig[i0:i1]

        if sig.std() < 1e-10:
            raise ValueError("Signal is flat — wrong channel.")

        n_samples = len(sig)
        dur_s     = n_samples / fs
        _prog(5, f"Signal loaded — {dur_s:.0f} s  ({n_samples:,} samples)")

        # _compute_preview_bundle does all DSP — pure, no self writes
        def _prep_prog(pct: int, msg: str) -> None:
            _prog(5 + int(pct * 0.85), msg)

        signal_bundle = self.compute_preview_bundle(sig, fs, params, _prep_prog)

        # Run threshold detection on the computed candidates (pure computation)
        thresh = params["thresh"]
        accepted, rejected, thresh_amp = apply_threshold(
            signal_bundle["signal_flt"],
            signal_bundle["all_cands"],
            signal_bundle["all_proms"],
            thresh,
            fs=fs,
        )

        _prog(100, "Done")

        return {
            # Signal identity
            "fs":              fs,
            "signal_raw":      sig,
            "dur_s":           dur_s,
            "detected_ch":     detected_ch,
            "detected_fs":     detected_fs,
            "requested_ch":    params["channel"],
            "fs_from_file":    detected_fs is not None,
            "thresh":          thresh,
            # Prepared signal bundle
            **signal_bundle,
            # Detection results
            "rpeaks_ok":       accepted,
            "rpeaks_rej":      rejected,
            "thresh_amp":      thresh_amp,
            "recommended_min_rr_ms": None,
        }

    def apply_detected_fs(self, fs: float) -> None:
        """Update the fs entry and source label on the main thread."""
        try:
            self.app.ent_fs.delete(0, "end")
            self.app.ent_fs.insert(0, str(int(fs)))
        except Exception as _exc:
            log.debug("ent_fs update failed: %s", _exc, exc_info=True)
        self.app.lbl_fs_source.configure(
            text=f"✓ Auto-detected from file: {int(fs)} Hz",
            text_color=GREEN)

    def on_preview_done(self, bundle: dict) -> None:
        """Atomically write all signal state on the main thread, then draw.

        This is the ONLY place that should assign signal/peak instance variables
        after a preview.  Because it runs via after(0, …) (scheduled by
        _start_async after the background worker finishes), it is guaranteed to
        execute on the Tk main thread with no concurrent background writes.
        """
        fs        = bundle["fs"]
        sig_raw   = bundle["signal_raw"]
        n_samples = len(sig_raw)

        # ── Atomic state update (main thread) ────────────────────────────────
        self.app.signal.fs              = fs
        self.app.signal.raw      = sig_raw
        self.app.signal.raw_norm = bundle["signal_raw_norm"]
        self.app.signal.filtered      = bundle["signal_flt"]
        self.app.signal.time            = np.arange(n_samples) / fs
        self.app.detection.all_candidates       = bundle["all_cands"]
        self.app.detection.all_prominences       = bundle["all_proms"]
        self.app.signal.no_filter_mode  = bundle["no_filter_mode"]
        self.app.signal.inverted = bundle.get("inverted", False)
        self.app.signal.raw_only_loaded = False
        # recommended_min_rr_ms supprimé — ent_minrr non modifié automatiquement.
        self.app.ui.nav_pos         = 0.0
        # Peak detection results (computed in worker from pure candidates)
        self.app.detection.rpeaks_ok           = bundle["rpeaks_ok"]
        self.app.detection.rpeaks_rej          = bundle["rpeaks_rej"]
        self.app.detection.thresh_amp          = bundle["thresh_amp"]
        self.app.detection.rpeaks_manual_excl  = np.array([], dtype=int)
        self.app.detection.rpeaks_manual_added = np.array([], dtype=int)
        # Reset manual peak edits — new file, clean slate
        self.app.detection.manual_excluded.clear()
        self.app.detection.manual_added.clear()
        # Invalidate all previous analysis state — new file, clean slate
        self.app.analysis.results       = None
        self.app.analysis.epoch_df      = None
        self.app.analysis.annotations   = []    # annotations belong to a specific file
        self.app.analysis.pacing_periods = []   # pacing periods belong to a specific file
        self.app.analysis.wave_template = None  # template may not suit new signal
        self.app.session.dirty = False
        # Increment generation so any in-flight bg workers discard their results
        self.app._generation    = getattr(self.app, "_generation", 0) + 1
        # Reset UI to blank state
        self.app._reset_kpis()
        self.app._reset_result_plots()
        self.app._reset_tab_status_labels()

        # ── UI feedback for auto-detected channel / fs ────────────────────────
        def _subject_from_channel(name: str) -> Optional[str]:
            digits = "".join(ch for ch in name if ch.isdigit())
            return digits if digits else None

        if bundle["detected_ch"] != bundle["requested_ch"]:
            if self.app.ent_channel is not None:
                try:
                    self.app.ent_channel.delete(0, "end")
                    self.app.ent_channel.insert(0, bundle["detected_ch"])
                except Exception as _exc:
                    log.debug("Could not update channel entry: %s", _exc)

            if self.app.ent_subject is not None:
                subject_id = _subject_from_channel(bundle["detected_ch"])
                current_subject = self.app.ent_subject.get().strip()
                if subject_id and (not current_subject or current_subject.lower().startswith("subject")):
                    try:
                        self.app.ent_subject.delete(0, "end")
                        self.app.ent_subject.insert(0, subject_id)
                    except Exception as _exc:
                        log.debug("Could not update subject entry: %s", _exc)

            self.app.lbl_file.configure(  # type: ignore[union-attr]
                text=f"Auto: {bundle['detected_ch']}", text_color=ORANGE)
        if bundle["fs_from_file"]:
            self.apply_detected_fs(bundle["detected_fs"])
        else:
            try:
                self.app.lbl_fs_source.configure(
                    text="Tip: fs not found in file — set manually above",
                    text_color=ORANGE)
            except Exception as e:
                log.debug("lbl_fs_source configure failed: %s", e)

        # Update peak count label and quality score
        n = len(self.app.detection.rpeaks_ok)  # type: ignore[union-attr]
        color = GREEN if n > 10 else RED
        if self.app.lbl_npeaks is not None:
            self.app.lbl_npeaks.configure(text=f"Peaks detected: {n}", text_color=color)  # type: ignore[union-attr]
        if self.app.btn_review_art is not None:
            self.app.btn_review_art.configure(state="normal" if n > 4 else "disabled")  # type: ignore[union-attr]
        self.app._update_signal_quality(self.app.detection.rpeaks_ok)  # type: ignore[union-attr]

        dur = bundle["dur_s"]
        detector_warning = bundle.get("detector_warning")
        if detector_warning:
            self.app._set_status(detector_warning, ORANGE)
        else:
            self.app._set_status(
                f"Signal ready — {n} peaks  |  {dur:.0f} s  |  {fs} Hz  "
                "→ adjust threshold then Run Full Analysis.", GREEN)
        self.app.tabs.set("📈 Detection")
        self.app._update_ann_count()   # reflect cleared annotations immediately
        self.app._update_pacing_count()
        # Sync nav bar
        self.app.ui.nav_pos = 0.0
        self.app._sync_nav_pos_entry()
        if self.app.lbl_sig_duration is not None:
            self.app.lbl_sig_duration.configure(  # type: ignore[union-attr]
                text=f"durée totale : {dur:.1f} s", text_color=MUTED)
        self.app._draw_detail()

        # Reset analysis window on new signal load and update feedback label
        self.app.analysis.t_start = 0.0
        self.app.analysis.t_end   = 0.0
        if self.app.lbl_analysis_window is not None:
            self.app.lbl_analysis_window.configure(  # type: ignore[union-attr]
                text=f"Full signal  ·  {n} peaks  ·  {dur:.1f} s",
                text_color=MUTED)

    def windowed_peaks(self) -> "Optional[np.ndarray]":
        """Return a copy of _rpeaks_ok filtered to the current analysis window.

        If no window is set (both bounds = 0), returns the full array.
        Returns None if _rpeaks_ok is None.

        This is the single source of truth for all analysis methods
        (_run_freq, _run_nonlinear, _run_intervals, _run_arrhythmia_analysis,
        _compute_epochs, _compute_rolling_hrv) — they all call this instead of
        doing ``self.app.detection.rpeaks_ok.copy()`` directly.
        """
        if self.app.detection.rpeaks_ok is None or self.app.signal.fs is None:
            return None
        rp = self.app.detection.rpeaks_ok.copy()
        t0 = self.app.analysis.t_start
        t1 = self.app.analysis.t_end
        if t0 <= 0 and t1 <= 0:
            return rp          # no window — full signal
        fs   = self.app.signal.fs
        mask = rp / fs >= t0
        if t1 > 0:
            mask &= rp / fs <= t1
        return rp[mask]

    def apply_analysis_window(self) -> None:
        """Read the analysis window entries and store in _analysis_t_start/_end.

        Updates the feedback label with the peak count inside the window.
        Does NOT re-run detection or analysis — the window is applied on
        the next Core Analysis run.
        """
        if self.app.ent_analysis_t0 is None or self.app.ent_analysis_t1 is None:
            return

        try:
            t0_raw = self.app.ent_analysis_t0.get().strip()  # type: ignore[union-attr]
            t1_raw = self.app.ent_analysis_t1.get().strip()  # type: ignore[union-attr]
            t0 = float(t0_raw) if t0_raw else 0.0
            t1 = float(t1_raw) if t1_raw else 0.0
        except ValueError:
            self.app._set_status("Invalid window — enter numeric values.", RED)
            return

        # Validate
        if t1 > 0 and t0 >= t1:
            self.app._set_status("La borne de début doit être inférieure à la borne de fin.", RED)
            return
        if self.app.signal.time is not None and t1 > float(self.app.signal.time[-1]) + 0.1:
            self.app._set_status(
                f"La borne de fin dépasse la durée du signal ({self.app.signal.time[-1]:.1f} s).", ORANGE)

        self.app.analysis.t_start = t0
        self.app.analysis.t_end   = t1

        # Feedback: count peaks in window
        if self.app.detection.rpeaks_ok is not None and self.app.signal.fs is not None:
            fs = self.app.signal.fs
            t_end_eff = float(self.app.signal.time[-1]) if (self.app.signal.time is not None and t1 == 0) else t1
            mask = (self.app.detection.rpeaks_ok / fs >= t0)
            if t1 > 0:
                mask &= (self.app.detection.rpeaks_ok / fs <= t1)
            n = int(mask.sum())
            dur = (t_end_eff - t0) if t1 > 0 else (float(self.app.signal.time[-1]) - t0 if self.app.signal.time is not None else 0)
            label_txt = (f"✓  {n} peaks  ·  {t0:.1f} s → {t_end_eff:.1f} s  ({dur:.1f} s)"
                         if t0 > 0 or t1 > 0
                         else f"✓  {n} peaks  ·  full signal")
            color = GREEN if n >= 5 else ORANGE
        else:
            label_txt = "✓  Window applied — run analysis"
            color = MUTED

        if self.app.lbl_analysis_window is not None:
            self.app.lbl_analysis_window.configure(  # type: ignore[union-attr]
                text=label_txt, text_color=color)

        self.app._set_status(
            "Analysis window updated — re-run Core Analysis.", BLUE)

    def reset_analysis_window(self) -> None:
        """Reset analysis window to full signal."""
        self.app.analysis.t_start = 0.0
        self.app.analysis.t_end   = 0.0
        if self.app.ent_analysis_t0 is not None:
            self.app.ent_analysis_t0.delete(0, "end")  # type: ignore[union-attr]
        if self.app.ent_analysis_t1 is not None:
            self.app.ent_analysis_t1.delete(0, "end")  # type: ignore[union-attr]
        if self.app.lbl_analysis_window is not None:
            # Recompute peak count for full signal
            if self.app.detection.rpeaks_ok is not None:
                n = len(self.app.detection.rpeaks_ok)
                dur = float(self.app.signal.time[-1]) if self.app.signal.time is not None else 0
                self.app.lbl_analysis_window.configure(  # type: ignore[union-attr]
                    text=f"✓  Full signal  ·  {n} peaks  ·  {dur:.1f} s",
                    text_color=GREEN)
            else:
                self.app.lbl_analysis_window.configure(  # type: ignore[union-attr]
                    text="", text_color=MUTED)
        self.app._set_status("Analysis window reset — full signal.", MUTED)

    def compute_filter_preview_segment(
        self, t_start: float, t_end: float,
    ) -> "Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]":
        """Compute a live-filtered preview of the signal over [t_start, t_end].

        Operates ONLY on the visible window (+ a short margin to absorb
        filtfilt edge transients) using the CURRENT filter widget values
        (HP/LP cutoffs, notch, clean method) — never on the full recording.
        This is intentionally cheap: it never touches ``self.app.signal.filtered``
        or any detection state, so it's safe to recompute on every redraw
        without affecting Preview Detection / Run Full Analysis results.

        Returns (t_slice, raw_slice_norm, filt_slice_norm), or None if the
        raw signal isn't loaded yet or the window is too short to filter.
        """
        if self.app.signal.raw is None or self.app.signal.fs is None or self.app.signal.time is None:
            return None
        fs = self.app.signal.fs

        margin = 1.0  # seconds — absorbs filtfilt edge transients
        lo_t = max(0.0, t_start - margin)
        hi_t = min(float(self.app.signal.time[-1]), t_end + margin)
        lo_i = int(lo_t * fs)
        hi_i = int(hi_t * fs)
        if hi_i - lo_i < int(0.5 * fs):
            return None  # too short to filter meaningfully

        seg_raw = np.asarray(self.app.signal.raw[lo_i:hi_i], dtype=float)

        lp_v         = self.app._safe_float(self.app.ent_lp, MouseECG.BP_LO_HZ)
        hp_v         = self.app._safe_float(self.app.ent_hp, MouseECG.BP_HI_HZ)
        notch_on     = bool(self.app.sw_notch.get()) if self.app.sw_notch is not None else False
        clean_method = self.app.cb_clean.get() if self.app.cb_clean is not None else "neurokit"

        seg = seg_raw.copy()
        try:
            seg = bandpass(seg, fs, lp_v, hp_v)
        except Exception as exc:
            log.debug("filter preview: bandpass skipped — %s", exc)
        if notch_on:
            try:
                seg = notch(seg, fs)
            except Exception as exc:
                log.debug("filter preview: notch skipped — %s", exc)
        try:
            if nk is not None:
                seg = nk.ecg_clean(seg, sampling_rate=fs, method=clean_method)
        except Exception as exc:
            log.debug("filter preview: ecg_clean skipped — %s", exc)

        seg      = normalize(np.asarray(seg, dtype=float))
        raw_norm = normalize(seg_raw)

        # Trim the margin back off — only the visible window is returned
        off0 = int(round((t_start - lo_t) * fs))
        off1 = off0 + int(round((t_end - t_start) * fs))
        off1 = min(off1, len(seg), len(raw_norm))
        off0 = min(off0, off1)
        t_slice = self.app.signal.time[lo_i:hi_i][off0:off1]
        return t_slice, raw_norm[off0:off1], seg[off0:off1]

    def on_filter_preview_toggle(self) -> None:
        """Toggle the before/after filter overlay and redraw."""
        self.app.ui.filter_preview_on = bool(self.app.sw_filter_preview.get()) if self.app.sw_filter_preview is not None else False
        self.app._draw_detail()

    def refresh_filter_preview(self) -> None:
        """Recompute the filter-preview overlay with current widget values.

        Bound to HP/LP entry <Return>/<FocusOut> and notch/clean-method
        changes — the preview segment isn't auto-reactive to keystrokes,
        only re-evaluated on these discrete commit events, matching how
        the rest of the sidebar (Preview Detection button) already works.

        Deliberately a no-op when `filter_preview_on` is False -- editing
        lp/hp/notch/clean-method has no visible effect until the user
        enables "Preview filter effect" or clicks "Preview Detection". This
        is the existing convention, not a bug: any new filter widget should
        call this same method on change (see ent_lp/ent_hp/sw_notch/cb_clean
        bindings in app.py's FILTERS section) rather than forcing the
        overlay on, to keep behaviour consistent across the panel.
        """
        if self.app.ui.filter_preview_on:
            self.app._draw_detail()

    def load_path(self, path: str) -> None:
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File not found:\n{path}")
            return
        # Full reset before loading a new file so the app looks exactly like
        # it did at startup — no stale results, plots, or arrhythmia cards
        # from the previous recording.
        self.reset_for_new_file()
        self.app.signal.filepath = path
        self.app.session.recording_notes = get_notes(path) if _DB_AVAILABLE else ""
        # Mirrors recording_notes above: the registry is the authoritative
        # source (also backs the ML training dialog's file list), the
        # session JSON is a fallback restored below when sqlite is unavailable.
        if _DB_AVAILABLE:
            _rec = get_recording(path)
            self.app.session.verified_for_training = bool(_rec.get("verified_for_training")) if _rec else False
        else:
            self.app.session.verified_for_training = False
        self.app.session_ctrl.sync_verified_switch()
        self.app.lbl_file.configure(text=os.path.basename(path), text_color=GREEN)  # type: ignore[union-attr]
        self.app._add_recent(path)
        # ── Try to restore a previously saved session ───────────────
        if self.app._try_restore_session(path):
            return   # session restored — skip raw load
        self.load_raw_only()

    def reset_for_new_file(self) -> None:
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
        # ── 1. Data-only state reset (no widget refs) ─────────────────────
        self.app.signal.filepath             = None
        self.app.signal.raw           = None
        self.app.signal.raw_norm      = None
        self.app.signal.filtered           = None
        self.app.signal.time                 = None
        # _fs intentionally kept (same rig)
        self.app.detection.rpeaks_ok            = None
        self.app.detection.rpeaks_rej           = None
        self.app.detection.all_candidates            = None
        self.app.detection.all_prominences            = None
        self.app.detection.thresh_amp           = 0.0
        self.app.analysis.results              = None
        self.app.analysis.epoch_df             = None
        self.app.analysis.rolling_hrv_df       = None
        self.app.analysis.t_start      = 0.0
        self.app.analysis.t_end        = 0.0
        if self.app.ui.hover_after_id is not None:
            try:
                self.app.after_cancel(self.app.ui.hover_after_id)
            except Exception:
                pass
        self.app.detection.hover_samp       = None
        self.app.detection.hover_samp_near  = False
        self.app.ui.hover_after_id   = None
        self.app.analysis.arrhythmia_events    = []
        self.app.analysis.arrhythmia_tsv       = ""
        self.app.analysis.arr_selected_idx     = -1
        self.app.analysis.arr_nav_pos          = 0.0
        self.app.analysis.arr_win              = 3.0
        self.app.analysis.arr_edit_mode        = False
        self.app.detection.sig_quality          = None
        self.app.analysis.artifact_report      = None
        self.app.analysis.artifact_candidates  = []
        self.app.detection.manual_excluded      = set()
        self.app.detection.rpeaks_manual_excl   = None
        self.app.detection.manual_added         = set()
        self.app.detection.rpeaks_manual_added  = None
        self.app.detection.edit_mode            = False
        self.app.detection.edit_undo            = []
        self.app.detection.edit_redo            = []
        self.app.ui.nav_pos              = 0.0
        self.app.signal.inverted      = False
        self.app.signal.raw_only_loaded      = False
        self.app.ui.thr_debounce_id      = None
        self.app.analysis.annotations          = []
        self.app.analysis.pacing_periods       = []
        self.app.ui.tsv_store            = {}
        self.app.analysis.wave_template        = None
        self.app.session.dirty        = False
        self.app.session.verified_for_training = False
        self.app._generation           = getattr(self.app, "_generation", 0) + 1  # invalidate async workers

        # ── 2. Result plots → placeholder ────────────────────────────────
        for slot in self.app._slots.values():
            try:
                slot._draw_fn = None
                slot._show_placeholder()
            except Exception as e:
                log.debug("slot placeholder reset failed: %s", e)

        # ── 3. KPI bar ────────────────────────────────────────────────────
        for lbl in self.app._kpi.values():
            try:
                lbl.configure(text="—", text_color=MUTED)
            except Exception as e:
                log.debug("KPI label reset failed: %s", e)

        # ── 4. Status labels ──────────────────────────────────────────────
        self.app._set_status("File loaded — run Detection", MUTED)
        self.app._set_progress(0, "")
        if self.app.lbl_file is not None:
            self.app.lbl_file.configure(text="Loading…", text_color=MUTED)  # type: ignore[union-attr]

        # ── 5. Sidebar detection status ───────────────────────────────────
        if self.app.lbl_npeaks is not None:
            self.app.lbl_npeaks.configure(  # type: ignore[union-attr]
                text="Run detection", text_color=MUTED)

        # ── 6. Disable per-tab on-demand buttons ─────────────────────────
        for btn_attr in ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl",
                         "btn_run_arrhythmia", "btn_save_session"):
            w = getattr(self.app, btn_attr, None)
            if w is not None:
                try:
                    w.configure(state="disabled")
                except Exception as e:
                    log.debug("widget disable failed: %s", e)

        # ── 7. Per-tab status labels ──────────────────────────────────────
        if self.app.lbl_freq_status is not None:
            self.app.lbl_freq_status.configure(  # type: ignore[union-attr]
                text="  Run Core Analysis first", text_color=MUTED)
        if self.app.lbl_nonlin_status is not None:
            self.app.lbl_nonlin_status.configure(  # type: ignore[union-attr]
                text="  Run Core Analysis first", text_color=MUTED)
        if self.app.lbl_ivl_status is not None:
            self.app.lbl_ivl_status.configure(  # type: ignore[union-attr]
                text="  Run Core Analysis first", text_color=MUTED)
        if self.app.lbl_arrhythmia_status is not None:
            self.app.lbl_arrhythmia_status.configure(  # type: ignore[union-attr]
                text="  Run Core Analysis first", text_color=MUTED)
        if self.app.lbl_roll_status is not None:
            self.app.lbl_roll_status.configure(  # type: ignore[union-attr]
                text="  Run Core Analysis first", text_color=MUTED)

        # ── 8. Textboxes ──────────────────────────────────────────────────
        for tb_attr in ("txt_rr", "txt_td", "txt_fd"):
            tb = getattr(self.app, tb_attr, None)
            if tb is not None:
                try:
                    self.app._set_textbox(tb, "")
                except Exception as e:
                    log.debug("_set_textbox clear failed: %s", e)

        # ── 9. Arrhythmia event cards ─────────────────────────────────────
        if self.app._arr_card_widgets is not None:
            for w in self.app._arr_card_widgets:
                try:
                    w.destroy()
                except Exception:
                    pass
            self.app._arr_card_widgets.clear()
        if self.app.lbl_arr_event_title is not None:
            try:
                self.app.lbl_arr_event_title.configure(  # type: ignore[union-attr]
                    text="← Click on an episode", text_color=MUTED)
            except Exception as e:
                log.debug("lbl_arr_event_title reset failed: %s", e)

        # ── 9b. Interpretation cards ──────────────────────────────────────
        # Détruire TOUS les enfants du scroll (groupes + cartes) pour éviter
        # que des frames grises orphelines restent visibles après un nouveau fichier.
        # interpretation tab removed — no-op
        self.app._interp_cards = {}
        self.app._interp_ref_labels = {}

        # ── 10. Session UI ────────────────────────────────────────────────
        self.app._update_session_ui(has_session=False)

        # ── 11. Disconnect RR click handler ──────────────────────────────
        if self.app.ui.rr_click_cid is not None:
            try:
                self.app._slots["rr"].canvas.mpl_disconnect(self.app.ui.rr_click_cid)
            except Exception as e:
                log.debug("mpl_disconnect (rr_click_cid) failed: %s", e)
            self.app.ui.rr_click_cid = None

        # ── 12. Switch to Detection tab so user lands in the right place ──
        try:
            self.app.tabs.set("📈 Detection")
        except Exception as e:
            log.debug("tabs.set Detection (reset) failed: %s", e)
