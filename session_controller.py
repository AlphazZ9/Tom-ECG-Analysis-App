# -*- coding: utf-8 -*-
"""
session_controller.py
----------------------
SessionController -- persisting and restoring the .ecgsession JSON cache
(filter params, results, annotations, UI state) and the SQLite recent-
recordings registry.
"""
from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from tkinter import messagebox

import numpy as np
import pandas as pd

from models import FilterParams, EXPERIMENTAL_CONTEXTS
from loaders import load_mat_signal, _serialise_results, _deserialise_results
from session import load_session, save_session, delete_session
from db import _DB_AVAILABLE, upsert_recording
from theme import THEME, BLUE, GREEN, MUTED

if TYPE_CHECKING:
    from app import ECGApp

log = logging.getLogger("ecg")


class SessionController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    def snapshot_params(self) -> dict:
        """Read every widget value on the main thread and return a plain dict snapshot.

        Delegates to ``FilterParams.from_widgets`` so that the field list is
        maintained in exactly one place (the FilterParams dataclass).  Any new
        parameter added to FilterParams is automatically picked up here and in
        _collect_session_state / _restore_session_worker.
        """
        params = FilterParams.from_widgets(self.app).to_dict()
        # Analysis window is independent from FilterParams (not a detection param)
        params["analysis_t_start"] = self.app.analysis.t_start
        params["analysis_t_end"]   = self.app.analysis.t_end
        return params

    def add_recent(self, path: str) -> None:
        if path in self.app.session.recent_files:
            self.app.session.recent_files.remove(path)
        self.app.session.recent_files.insert(0, path)
        self.app.session.recent_files = self.app.session.recent_files[:8]

    def try_restore_session(self, path: str) -> bool:
        """If a saved session exists for *path*, offer to restore it.

        Returns True if the session restoration was initiated (caller should
        skip _preview).  The actual restore runs in a background thread via
        _start_async so the UI stays responsive during signal reload + filtering.
        """
        state = load_session(path)
        if state is None:
            self.update_session_ui(has_session=False)
            return False

        saved_at = state.get("saved_at", "unknown time")
        n_beats  = state.get("n_beats", "?")
        answer   = messagebox.askyesno(
            "Session found",
            f"A saved analysis was found for this file:\n\n"
            f"  Saved:  {saved_at}\n"
            f"  Beats:  {n_beats}\n\n"
            f"Restore it?  (No = load raw signal — click Preview Detection to re-run)",
            parent=self,
        )
        if not answer:
            self.update_session_ui(has_session=True)
            return False

        # ── Restauration asynchrone — le reload + _compute_preview_bundle peut
        # prendre 3–10 s ──
        # _restore_state_from_session est réellement découpée en deux :
        #   _restore_session_worker() : partie lourde, pure — reload .mat +
        #       _compute_preview_bundle — AUCUNE écriture sur self ni sur les
        #       widgets Tkinter. Tourne dans le thread BG lancé par _start_async.
        #   _on_restore_session_done(): toutes les écritures sur self et les
        #       widgets (y compris _run_detection, _draw_detail, etc.).
        #       Tourne sur le thread principal via after(0, …).
        _state_snap = state
        _saved_at   = saved_at

        self.app._start_async(
            self.app.btn_preview, "Restoring…", "Restoring session…",
            lambda: self.restore_session_worker(_state_snap),
            lambda bundle: self.on_restore_session_done(bundle, _saved_at),
            pass_result=True,
        )
        return True

    def restore_session_worker(self, state: dict) -> dict:
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
        if self.app.signal.filepath is None:
            raise RuntimeError("No file loaded — cannot restore session.")

        fs = int(state["fs"])

        # ── Parse saved filter params via FilterParams.from_dict ────────
        # from_dict tolerates missing keys with safe defaults, so old session
        # files (saved before any new field was added) restore without error.
        fp = FilterParams.from_dict(state.get("filter_params", {}))

        try:
            sig_raw, detected_ch, _, detected_fs = load_mat_signal(
                self.app.signal.filepath, fp.channel)
        except Exception as exc:
            raise RuntimeError(f"Could not reload signal from {self.app.signal.filepath}: {exc}") from exc

        if detected_fs is not None:
            fs = int(detected_fs)

        # Apply the same time-crop that was active when the session was saved
        i0 = int(fp.t_start * fs) if fp.t_start > 0 else 0
        i1 = int(fp.t_end   * fs) if fp.t_end   > 0 else len(sig_raw)
        sig_raw = sig_raw[i0:i1]
        n_samples = len(sig_raw)

        # ── Re-run signal processing with saved filter params (pure/static) ──
        # fp.to_dict() produces the "notch" key (not "notch_filter") that
        # _compute_preview_bundle expects.
        prepare_params = fp.to_dict()
        signal_bundle = self.app._compute_preview_bundle(sig_raw, fs, prepare_params)

        results_raw  = state.get("results")
        epoch_raw    = state.get("epoch_df")
        rolling_raw  = state.get("rolling_hrv_df")

        return {
            "fs":               fs,
            "n_samples":        n_samples,
            "signal_raw":       sig_raw,
            "signal_bundle":    signal_bundle,
            "thresh_amp":       float(state.get("threshold", float(state.get("thresh_amp", 0.5)))),
            "manual_excluded":  set(int(x) for x in state.get("manual_excluded", [])),
            "manual_added":     set(int(x) for x in state.get("manual_added",    [])),
            "signal_inverted":  bool(state.get("signal_inverted", False)),
            "no_filter_mode":   bool(state.get("no_filter_mode",  fp.no_filter)),
            "results":          _deserialise_results(results_raw) if results_raw else None,
            "artifact_report":  state.get("artifact_report"),
            "epoch_df":         pd.DataFrame(epoch_raw)   if epoch_raw   is not None else None,
            "rolling_hrv_df":   pd.DataFrame(rolling_raw) if rolling_raw is not None else None,
            "annotations":      list(state.get("annotations", [])),
            "analysis_t_start": float(state.get("analysis_t_start", 0.0)),
            "analysis_t_end":   float(state.get("analysis_t_end",   0.0)),
            "exp_context":      state.get("exp_context", "telemetry_awake"),
            "nav_pos":          float(state.get("nav_pos", 0.0)),
            "edit_mode":        bool(state.get("edit_mode", False)),
            "current_tab":      state.get("current_tab", "Detection"),
        }

    def on_restore_session_done(self, bundle: dict, saved_at: str) -> None:
        """Write all restored state to self and to widgets — main thread only.

        Counterpart to ``_restore_session_worker``. This is the ONLY place
        that writes session-restore state to ``self``/widgets, exactly as
        ``_on_preview_done`` is the sole writer after ``_preview_worker``.
        """
        # Clear figure cache on session restore
        self.app.ui.figure_cache.clear()

        fs        = bundle["fs"]
        n_samples = bundle["n_samples"]
        self.app.signal.fs  = fs

        self.app.signal.raw = bundle["signal_raw"]
        self.app.signal.time       = np.arange(n_samples) / fs
        self.app.ui.nav_pos    = 0.0
        self.app.ui.ds_time         = None
        self.app.ui.ds_sig          = None
        self.app.ui.ds_sig_max      = None
        self.app.ui.ds_sig_mid      = None
        self.app.ui.ds_raw_sig      = None
        self.app.ui.ds_raw_sig_max  = None
        self.app.ui.ds_raw_sig_mid  = None

        sb = bundle["signal_bundle"]
        self.app.signal.raw_norm = sb["signal_raw_norm"]
        self.app.signal.filtered      = sb["signal_flt"]
        self.app.detection.all_candidates       = sb["all_cands"]
        self.app.detection.all_prominences       = sb["all_proms"]
        self.app.signal.no_filter_mode  = sb["no_filter_mode"]
        self.app.signal.inverted = sb.get("inverted", False)

        # ── Restore user-edited state (session values win over freshly
        # recomputed ones, matching the original restore order) ──────────
        self.app.detection.thresh_amp      = bundle["thresh_amp"]
        self.app.detection.manual_excluded = set(bundle["manual_excluded"])
        self.app.detection.manual_added    = set(bundle["manual_added"])
        self.app.signal.inverted = bundle["signal_inverted"]
        self.app.signal.no_filter_mode  = bundle["no_filter_mode"]

        # ── Restore analysis results (DataFrames already deserialised) ───
        self.app.analysis.results         = bundle["results"]
        self.app.analysis.artifact_report = bundle["artifact_report"]
        self.app.analysis.epoch_df        = bundle["epoch_df"]
        self.app.analysis.rolling_hrv_df  = bundle["rolling_hrv_df"]
        self.app.analysis.annotations     = list(bundle["annotations"])

        self.app.analysis.t_start = bundle["analysis_t_start"]
        self.app.analysis.t_end   = bundle["analysis_t_end"]
        # Sync analysis window entry widgets with restored values
        if self.app.ent_analysis_t0 is not None:
            self.app._batch_ui_update(self.app.ent_analysis_t0, state="normal")
            self.app.ent_analysis_t0.delete(0, "end")  # type: ignore[union-attr]
            if self.app.analysis.t_start > 0:
                self.app.ent_analysis_t0.insert(0, str(self.app.analysis.t_start))  # type: ignore[union-attr]
            self.app._batch_ui_update(self.app.ent_analysis_t0, state="normal")
        if self.app.ent_analysis_t1 is not None:
            self.app._batch_ui_update(self.app.ent_analysis_t1, state="normal")
            self.app.ent_analysis_t1.delete(0, "end")  # type: ignore[union-attr]
            if self.app.analysis.t_end > 0:
                self.app.ent_analysis_t1.insert(0, str(self.app.analysis.t_end))  # type: ignore[union-attr]
            self.app._batch_ui_update(self.app.ent_analysis_t1, state="normal")

        # ── Restore UI state ─────────────────────────────────────────────
        ctx = bundle["exp_context"]
        if ctx in EXPERIMENTAL_CONTEXTS:
            self.app.analysis.exp_context = ctx
            try:
                pass  # interpretation removed  # type: ignore[union-attr]
                pass  # interpretation removed
                pass  # interpretation removed
            except Exception as e:
                log.warning("Failed to restore context from session: %s", e)

        self.app.ui.nav_pos = bundle["nav_pos"]
        self.app._sync_nav_pos_entry()

        saved_edit = bundle["edit_mode"]
        if saved_edit != self.app.detection.edit_mode:
            self.app._toggle_edit_mode()

        # ── Restore threshold slider and re-apply detection ──────────────
        thr = self.app.detection.thresh_amp
        if self.app.sl_thr is not None:
            try:
                self.app.sl_thr.set(float(thr))  # type: ignore[union-attr]
                self.app.ent_thr.delete(0, "end")  # type: ignore[union-attr]
                self.app.ent_thr.insert(0, f"{float(thr):.3f}")  # type: ignore[union-attr]
            except Exception as _exc:
                log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 7671, _exc)

        self.app._run_detection(thr)

        # ── Enable analysis buttons ──────────────────────────────────────
        self.app.btn_save_session.configure(state="normal")  # type: ignore[union-attr]
        for btn_attr in ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl"):
            if getattr(self, btn_attr, None) is not None:
                getattr(self, btn_attr).configure(state="normal")

        n_beats = len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0
        self.app._set_status(
            f"Session restored — {n_beats} beats  |  "
            f"{n_samples / fs:.0f} s recording  |  results ready", GREEN)

        # ── Render ───────────────────────────────────────────────────────
        self.app._draw_detail(self.app.ui.nav_pos)
        self.app._update_ann_count()
        # Restore active tab
        try:
            saved_tab = bundle["current_tab"]
            self.app.tabs.set(saved_tab)
        except Exception as e:
            log.debug("Failed to restore tab from session: %s", e)
        if self.app.analysis.results is not None:
            self.app.after(100, self.app._draw_all_results)
            pass  # interpretation removed
            if self.app.lbl_freq_status is not None:
                fd = self.app.analysis.results.get("hrv_freq")
                has_freq = fd is not None and not (hasattr(fd, "empty") and fd.empty)
                self.app.lbl_freq_status.configure(  # type: ignore[union-attr]
                    text="  Loaded from session ✓" if has_freq
                    else "  Core done — click to compute LF / HF",
                    text_color=GREEN if has_freq else BLUE)
        self.app._update_kpis()
        self.update_session_ui(has_session=True, saved_at=saved_at)

    def current_filter_params_dict(self) -> dict:
        """Return a serialisable filter-params dict, reading from widgets if available.

        Safe to call at any point — uses FilterParams defaults for any widget
        not yet built (e.g. during early startup or after a rebuild).
        """
        try:
            # Happy path: all widgets are built and readable
            fp = FilterParams.from_widgets(self.app)
            # Preserve the last-used no_filter state from self in case the
            # toggle widget is temporarily inconsistent during a rebuild.
            fp = dataclasses.replace(fp, no_filter=self.app.signal.no_filter_mode)
            return fp.to_dict()
        except Exception:
            # Widget not yet built — return safe defaults carrying known state
            return FilterParams(no_filter=self.app.signal.no_filter_mode).to_dict()

    def safe_get_tab(self) -> str:
        """Return the current tab name, or 'Detection' if not yet built."""
        try:
            return self.app.tabs.get()
        except Exception:
            return "Detection"

    def collect_session_state(self) -> dict:
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
        state: dict = {
            "saved_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fs":              int(self.app.signal.fs),
            "threshold":       float(self.app.detection.thresh_amp),
            "manual_excluded": list(self.app.detection.manual_excluded),
            "manual_added":    list(self.app.detection.manual_added),
            "no_filter_mode":  self.app.signal.no_filter_mode,
            "signal_inverted": self.app.signal.inverted,
            # Filter params — use FilterParams.from_widgets so the field list
            # is maintained in a single place.  Falls back to defaults for any
            # widget that hasn't been built yet (e.g. called during startup).
            "filter_params":   self.current_filter_params_dict(),
            # Beat count stored for the restore-dialog display only
            "n_beats":         len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0,
            # UI state that must survive a session restore
            "exp_context":     self.app.analysis.exp_context,
            "nav_pos":         self.app.ui.nav_pos,
            "edit_mode":       self.app.detection.edit_mode,
            "current_tab":     self.safe_get_tab(),
            "analysis_t_start": self.app.analysis.t_start,
            "analysis_t_end":   self.app.analysis.t_end,
        }
        if self.app.analysis.results is not None:
            state["results"] = _serialise_results(self.app.analysis.results)
        if self.app.analysis.artifact_report is not None:
            state["artifact_report"] = self.app.analysis.artifact_report
        if self.app.analysis.epoch_df is not None and not self.app.analysis.epoch_df.empty:
            state["epoch_df"] = self.app.analysis.epoch_df.to_dict(orient="list")
        if self.app.analysis.rolling_hrv_df is not None and not self.app.analysis.rolling_hrv_df.empty:
            state["rolling_hrv_df"] = self.app.analysis.rolling_hrv_df.to_dict(orient="list")
        if self.app.analysis.annotations:
            state["annotations"] = self.app.analysis.annotations
        return state

    def save_session(self) -> None:
        """Serialise full analysis state to a .ecgsession cache file and update registry."""
        if self.app.signal.filepath is None:
            messagebox.showwarning("No file", "Open a file first.")
            return
        if self.app.signal.filtered is None:
            messagebox.showwarning("Not ready", "Run Preview Detection first.")
            return
        try:
            self.app._set_status("Saving session…", MUTED)
            state    = self.collect_session_state()
            out_path = save_session(self.app.signal.filepath, state)
            saved_at = state["saved_at"]
            self.app.session.dirty = False
            self.update_session_ui(has_session=True, saved_at=saved_at)
            # ── SQLite registry upsert ────────────────────────────────────
            if _DB_AVAILABLE:
                from session import _file_fingerprint
                _stats: dict = {}
                if self.app.analysis.results:
                    _rdf = self.app.analysis.results.get("rr_df")
                    _hrv = self.app.analysis.results.get("hrv_td")
                    if _rdf is not None and len(_rdf):
                        _stats["hr_mean"] = float(_rdf["HR_bpm"].mean())
                    if _hrv is not None and "HRV_SDNN" in _hrv.columns:
                        _stats["sdnn"]  = float(_hrv["HRV_SDNN"].values[0])
                    if _hrv is not None and "HRV_RMSSD" in _hrv.columns:
                        _stats["rmssd"] = float(_hrv["HRV_RMSSD"].values[0])
                    if self.app.signal.time is not None and self.app.signal.fs:
                        _stats["duration_s"] = float(len(self.app.signal.time)) / self.app.signal.fs
                    if self.app.detection.rpeaks_ok is not None:
                        _stats["n_peaks"] = int(len(self.app.detection.rpeaks_ok))
                upsert_recording(
                    filepath=self.app.signal.filepath,
                    fingerprint=_file_fingerprint(self.app.signal.filepath),
                    session_path=str(out_path),
                    stats=_stats,
                    notes=self.app.session.recording_notes,
                )
            self.app._set_status(f"Session saved — {Path(out_path).name}", GREEN)
        except Exception as exc:
            log.exception("_save_session failed")
            messagebox.showerror("Save failed", str(exc))

    def delete_session(self) -> None:
        """Delete the session cache file for the current file."""
        if self.app.signal.filepath is None:
            return
        deleted = delete_session(self.app.signal.filepath)
        if deleted:
            self.update_session_ui(has_session=False)
            self.app._set_status("Session cache deleted.", MUTED)
        else:
            self.app._set_status("No session cache to delete.", MUTED)

    def update_session_ui(self, has_session: bool,
                           saved_at: str = "") -> None:
        """Update the session info label and button states."""
        if self.app.lbl_session_info is None:
            return
        if has_session and saved_at:
            self.app.lbl_session_info.configure(  # type: ignore[union-attr]
                text=f"✓ Session saved  {saved_at}", text_color=GREEN)
        elif has_session:
            self.app.lbl_session_info.configure(  # type: ignore[union-attr]
                text="✓ Session file exists for this recording", text_color=GREEN)
        else:
            self.app.lbl_session_info.configure(  # type: ignore[union-attr]
                text="No session saved for this file", text_color=MUTED)
        # Update wave template info label
        if self.app.lbl_template_info is not None and self.app.analysis.wave_template is not None:
            wt = self.app.analysis.wave_template
            if wt.confirmed:
                self.app.lbl_template_info.configure(  # type: ignore[union-attr]
                    text=f"✓ Custom template active  (source={wt.source})",
                    text_color=GREEN)
            else:
                self.app.lbl_template_info.configure(  # type: ignore[union-attr]
                    text="Using default mouse landmarks", text_color=MUTED)

    def snapshot_ui_state(self) -> dict:
        """Capture all editable widget values before a UI rebuild."""
        s: dict = {}
        for attr in ("ent_channel", "ent_subject", "ent_fs", "ent_t_start",
                     "ent_t_end", "ent_lp", "ent_hp", "ent_minrr",
                     "ent_epoch", "ent_overlap", "ent_thr", "ent_window",
                     "ent_sg_target_fs", "ent_sg_window_ms"):
            try:
                s[attr] = getattr(self, attr).get()
            except Exception:
                s[attr] = ""
        for attr in ("sw_show_raw", "sw_no_filter", "sw_notch",
                     "sw_artifact", "sw_epoch"):
            try:
                s[attr] = bool(getattr(self, attr).get())
            except Exception:
                s[attr] = False
        try:
            s["sl_thr"] = float(self.app.sl_thr.get())  # type: ignore[union-attr]
        except Exception:
            s["sl_thr"] = 0.5
        try:
            s["cb_clean"] = self.app.cb_clean.get()  # type: ignore[union-attr]
        except Exception:
            s["cb_clean"] = "neurokit"
        try:
            s["cb_det_method"] = self.app.cb_det_method.get() if self.app.cb_det_method is not None else "SG + Derivative (10 kHz)"
        except Exception:
            s["cb_det_method"] = "SG + Derivative (10 kHz)"
        try:
            s["adv_filters_open"] = getattr(self, "_adv_filters_open", False)
        except Exception:
            s["adv_filters_open"] = False
        s["exp_context"] = self.app.analysis.exp_context
        try:
            s["current_tab"] = self.app.tabs.get()
        except Exception:
            s["current_tab"] = "Detection"
        s["edit_mode"] = self.app.detection.edit_mode
        s["nav_pos"]   = self.app.ui.nav_pos
        s["dark_mode"] = self.app.ui.dark_mode
        return s

    def restore_ui_state(self, s: dict) -> None:
        """Restore widget values captured before a UI rebuild."""
        for attr in ("ent_channel", "ent_subject", "ent_fs", "ent_t_start",
                     "ent_t_end", "ent_lp", "ent_hp", "ent_minrr",
                     "ent_epoch", "ent_overlap", "ent_thr", "ent_window",
                     "ent_sg_target_fs", "ent_sg_window_ms"):
            val = s.get(attr)
            if val is None:
                continue
            try:
                w = getattr(self, attr, None)
                if w is None:
                    continue
                w.delete(0, "end")
                w.insert(0, str(val))
            except Exception as _exc:
                log.debug("_restore_ui_state entry %s: %s", attr, _exc)

        sw_map = {
            "sw_show_raw": "sw_show_raw",
            "sw_no_filter": "sw_no_filter",
            "sw_notch": "sw_notch",
            "sw_artifact": "sw_artifact",
            "sw_epoch": "sw_epoch",
        }
        for key, attr in sw_map.items():
            try:
                w = getattr(self, attr)
                if s.get(key):
                    w.select()
                else:
                    w.deselect()
            except Exception as _exc:
                log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 7929, _exc)

        try:
            self.app.sl_thr.set(float(s.get("sl_thr", 0.5)))  # type: ignore[union-attr]
        except Exception as _exc:
            log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 7934, _exc)
        try:
            self.app.cb_clean.set(s.get("cb_clean", "neurokit"))  # type: ignore[union-attr]
        except Exception as _exc:
            log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 7938, _exc)
        try:
            if self.app.cb_det_method is not None:
                dm = s.get("cb_det_method", "SG + Derivative (10 kHz)")
                self.app.cb_det_method.set(dm)
                self.app._on_det_method_change(dm)
        except Exception as _exc:
            log.debug("cb_det_method restore: %s", _exc)
        # Restore advanced-filter panel state
        try:
            was_open = bool(s.get("adv_filters_open", False))
            if was_open != getattr(self, "_adv_filters_open", False):
                self.app._btn_adv_flt.invoke()   # toggles the sub-section
        except Exception as _exc:
            log.debug("adv_filters restore: %s", _exc)
        try:
            self.app.tabs.set(s.get("current_tab", "Detection"))
        except Exception as _exc:
            log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 7942, _exc)
        # Re-apply non-widget state
        self.app.ui.nav_pos  = float(s.get("nav_pos",  0.0))

        # Restore experimental context
        ctx = s.get("exp_context", "telemetry_awake")
        if ctx in EXPERIMENTAL_CONTEXTS:
            self.app.analysis.exp_context = ctx
            try:
                pass  # interpretation removed
                pass  # interpretation removed
                pass  # interpretation removed
            except Exception as e:
                log.warning("Failed to restore context from session: %s", e)
        self.app.ui.dark_mode = bool(s.get("dark_mode", THEME.is_dark))
        saved_edit = s.get("edit_mode", False)
        if bool(saved_edit) != self.app.detection.edit_mode:
            self.app._toggle_edit_mode()
