# -*- coding: utf-8 -*-
"""
analysis_controller.py
------------------------
AnalysisController -- HRV time/frequency/non-linear analysis, interval
delineation, arrhythmia classification and its ECG-viewer edit mode,
rolling-window and epoch HRV, and the QTc-formula recompute.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
import pandas as pd

from models import (
    MouseECG, EXPERIMENTAL_CONTEXTS, _CONTEXT_FIELD_MAP, ArrhythmiaEvent,
)
from detection import classify_arrhythmias, correct_rr_artifacts
from analysis import analyse_core, analyse_hrv_freq, analyse_hrv_nonlinear, analyse_intervals
from wave_template import WaveTemplate
from session import load_session
from sidebar import IntervalVerifierPanel
from theme import (
    PLOT, RED, RED_MID, RED_LIGHT, AMBER, BLUE, BLUE_DARK, BLUE_MID,
    GREEN, GREEN_DARK, GREEN_MID, ORANGE, ORANGE_DARK, ORANGE_DEEP,
    PURPLE, CORAL, NAVY, GRAY_LIGHT, CYAN_BRIGHT, MUTED, TEXT, LIGHT,
    CARD, BORDER, BORDER2, NK_AVAILABLE, nk,
    FONT_SIDEBAR_HDR, FONT_BADGE, FONT_HINT, FONT_SUBSECTION, FONT_KPI_LABEL,
    FONT_SMALL,
)
from plots import style_axes

if TYPE_CHECKING:
    from app import ECGApp

log = logging.getLogger("ecg")

# Spacing scale mirrored from app.py's local constants (SPACE_XS/S/M/L) --
# not exported by theme.py, so duplicated here for the few widgets this
# controller still builds directly (arrhythmia event cards).
SPACE_XS = 2
SPACE_S = 4
SPACE_L = 12


class AnalysisController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    def current_ref(self, key: str) -> "tuple[float, float]":
        """Return (lo, hi) reference range for the given metric key from current experimental context."""
        ctx = EXPERIMENTAL_CONTEXTS.get(self.app.analysis.exp_context)
        if ctx is None:
            # Fallback defaults
            if "HR" in key:
                return MouseECG.HR_MIN_BPM, MouseECG.HR_MAX_BPM
            elif "RR" in key:
                return MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS
            else:
                return 0.0, 1000.0
        lo_field, hi_field = _CONTEXT_FIELD_MAP.get(key, ("hr_lo", "hr_hi"))
        return getattr(ctx, lo_field), getattr(ctx, hi_field)

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
            self.app._set_status("Run Core Analysis first.", RED)
            return

        rpeaks = self.app._windowed_peaks()
        if rpeaks is None or len(rpeaks) < 5:
            self.app._set_status("Not enough peaks in the analysis window.", ORANGE)
            return
        fs      = float(self.app.signal.fs)
        ctx_key = self.app.analysis.exp_context

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

        def _worker():
            return classify_arrhythmias(
                rpeaks, fs, ctx_key,
                baseline_s=baseline_s,
                brady_pct=brady_pct,
                min_brady_beats=min_beats,
            )

        def _done(events: "list[ArrhythmiaEvent]"):
            self.app.analysis.arrhythmia_events = events
            self.app.analysis.arr_selected_idx  = -1

            sev_colors = {"alert": RED_MID, "warning": AMBER, "info": BLUE_MID}
            kind_icons = {
                "bradycardia": "🔵", "tachycardia": "🔴", "pause": "⏸",
                "esv_run": "⚡", "irregular_run": "〰",
            }

            # ── RR timeline (right panel, initial state) ──────
            t_peaks = rpeaks / fs
            rr_ms   = np.diff(rpeaks).astype(float) / fs * 1000
            t_rr    = (t_peaks[:-1] + t_peaks[1:]) / 2

            def draw_rr_timeline(fig):
                ax = fig.subplots(1, 1)
                style_axes(ax)
                ax.plot(t_rr, rr_ms, color=PLOT.get("ecg",CYAN_BRIGHT), lw=0.8, zorder=2)
                ax.set_ylabel("RR (ms)"); ax.set_xlabel("Time (s)")

                # Baseline and threshold lines (only if events contain brady/tachy)
                brady_events = [e for e in events if e.kind in ("bradycardia","tachycardia")
                                and e.baseline_hr > 0]
                if brady_events:
                    bl_hr  = brady_events[0].baseline_hr
                    bl_rr  = 60_000.0 / bl_hr
                    try:
                        bpct = max(5.0, float(self.app.ent_arr_brady_pct.get()))
                    except Exception:
                        bpct = 20.0
                    brady_rr = 60_000.0 / (bl_hr * (1 - bpct / 100))
                    tachy_rr = 60_000.0 / (bl_hr * (1 + bpct / 100))
                    try:
                        bl_end = min(float(self.app.ent_arr_baseline.get()),
                                     float(t_rr[-1]) if len(t_rr) else 30.0)
                    except Exception:
                        bl_end = 30.0
                    # Shade baseline window
                    ax.axvspan(0, bl_end, alpha=0.08, color=GREEN_MID,
                               zorder=0, label="Baseline window")
                    ax.axvline(bl_end, color=GREEN_MID, lw=1.0,
                               ls=":", alpha=0.6, zorder=3)
                    # Baseline RR line
                    ax.axhline(bl_rr, color=GREEN_MID, lw=1.2,
                               ls="--", alpha=0.8, zorder=4,
                               label=f"Baseline {bl_hr:.0f} bpm")
                    # Brady / tachy threshold lines
                    ax.axhline(brady_rr, color=AMBER, lw=0.9,
                               ls="--", alpha=0.65, zorder=4,
                               label=f"Seuil brady −{bpct:.0f}%")
                    ax.axhline(tachy_rr, color=BLUE_MID, lw=0.9,
                               ls="--", alpha=0.65, zorder=4,
                               label=f"Seuil tachy +{bpct:.0f}%")

                ax.set_title(
                    "RR series — click an episode to zoom",
                    loc="left", fontsize=9)
                for ev in events:
                    c = sev_colors.get(ev.severity, "#888")
                    ax.axvspan(ev.t_start, max(ev.t_end, ev.t_start + 0.05),
                               alpha=0.20, color=c, zorder=1)
                    ax.axvline(ev.t_start, color=c, lw=0.7, ls="--",
                               alpha=0.55, zorder=3)
                if brady_events:
                    ax.legend(loc="upper right", fontsize=7,
                              facecolor=PLOT.get("bg",NAVY),
                              labelcolor=PLOT.get("text",GRAY_LIGHT),
                              edgecolor=PLOT.get("border","#333"))

            self.app._slots["arr_detail"].update(draw_rr_timeline)

            # ── Build clickable event cards ───────────────────
            for w in self.app._arr_card_widgets:
                try: w.destroy()
                except Exception: pass
            self.app._arr_card_widgets.clear()

            if not events:
                lbl = ctk.CTkLabel(
                    self.app._arr_event_scroll,
                    text="No arrhythmia detected\nfor the active context.",
                    font=FONT_SMALL, text_color=MUTED, justify="center",
                )
                lbl.grid(row=0, column=0, pady=SPACE_L)
                self.app._arr_card_widgets.append(lbl)
            else:
                for idx, ev in enumerate(events):
                    self.build_arrhythmia_card(idx, ev, sev_colors, kind_icons)

            # ── TSV store ─────────────────────────────────────
            tsv_rows = ["Type\tStart_s\tEnd_s\tDuration_s\tHR_bpm\tBaseline_bpm\tDelta_pct\tRR_ms\tSeverity\tDescription"]
            for ev in events:
                tsv_rows.append(
                    f"{ev.kind}\t{ev.t_start:.2f}\t{ev.t_end:.2f}\t"
                    f"{ev.duration_s:.2f}\t{ev.hr_mean:.1f}\t"
                    f"{ev.baseline_hr:.1f}\t{ev.delta_pct:.1f}\t"
                    f"{ev.rr_mean:.1f}\t{ev.severity}\t{ev.label}"
                )
            self.app.analysis.arrhythmia_tsv = "\n".join(tsv_rows)

            n = len(events)
            self.app.lbl_arrhythmia_status.configure(  # type: ignore[union-attr]
                text=f"  {n} episode{'s' if n != 1 else ''} — click to explore",
                text_color=RED if any(e.severity=="alert" for e in events)
                           else (ORANGE if n else GREEN),
            )
            self.app._set_status(f"Arrhythmia classification — {n} episode(s)", GREEN)
            self.app.tabs.set("Arrhythmias")

        if self.app.btn_run_arrhythmia is None:
            return
        self.app._start_async_result(
            self.app.btn_run_arrhythmia, "Classifying…", _worker, _done)

    def build_arrhythmia_card(
        self, idx: int, ev: "ArrhythmiaEvent",
        sev_colors: dict, kind_icons: dict,
    ) -> None:
        c_sev = sev_colors.get(ev.severity, MUTED)
        icon  = kind_icons.get(ev.kind, "·")

        card = ctk.CTkFrame(
            self.app._arr_event_scroll,
            fg_color=CARD, corner_radius=6,
            border_width=2, border_color=BORDER,
        )
        card.grid(row=idx, column=0, sticky="ew", padx=SPACE_S, pady=(0, SPACE_S))
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkFrame(card, width=4, fg_color=c_sev,
                     corner_radius=0).grid(row=0, column=0, rowspan=3, sticky="ns")

        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.grid(row=0, column=1, sticky="ew", padx=(SPACE_S, SPACE_S), pady=(SPACE_S, SPACE_XS))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text=f"{icon}  {ev.kind.replace('_', ' ').title()}",
                     font=FONT_SIDEBAR_HDR, text_color=TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(hdr, text=ev.severity.upper(),
                     font=FONT_BADGE, text_color=c_sev,
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
        if self.app.analysis.arr_edit_mode:
            self.app.btn_arr_edit.configure(
                fg_color=ORANGE, hover_color=ORANGE_DEEP,
                text_color="white", text="Edit Mode ON",
            )
            self.app.lbl_arr_edit_hint.pack(side="left", padx=(SPACE_S, 0))
        else:
            self.app.btn_arr_edit.configure(
                fg_color=BORDER, hover_color=BORDER2,
                text_color=MUTED, text="Edit Peaks",
            )
            self.app.lbl_arr_edit_hint.pack_forget()
        self.app._draw_arr_detail()

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
                        f"Too close to existing peak à {click_time:.3f} s", ORANGE)
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
        self.app._set_status(msg + "  — rerun Core Analysis pour mettre à jour HRV", ORANGE)
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
        self.app._draw_arr_detail()

    def copy_arrhythmia_tsv(self) -> None:
        tsv = getattr(self.app, "_arrhythmia_tsv", None)
        if not tsv:
            self.app._set_status("Run arrhythmia classification first.", RED)
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(tsv)
        self.app._set_status("Arrhythmias copied to clipboard (Excel ready)", GREEN)

    def qtc_formula(self) -> str:
        """Return 'mitchell' or 'bazett' from the Intervals tab combo selector."""
        try:
            val = self.app.cb_qtc_formula.get()  # type: ignore
            return "bazett" if "Bazett" in val else "mitchell"
        except Exception:
            return "mitchell"

    def on_qtc_formula_change(self, _choice: str = "") -> None:
        """Re-compute QTc with the selected formula and refresh interval plots."""
        if self.app.analysis.results is None:
            return
        ivl = self.app.analysis.results.get("intervals")
        if ivl is None or ivl.empty or "QT_ms" not in ivl.columns:
            return
        formula = self.qtc_formula()
        qt_ms  = ivl["QT_ms"].values.astype(float)
        rr_arr = ivl["RR_ms"].values.astype(float)
        rr_s   = np.clip(rr_arr, MouseECG.RR_MIN_MS, None) / 1000.0
        if formula == "bazett":
            qtc = qt_ms / np.sqrt(rr_s)
        else:
            qtc = qt_ms / (rr_s ** (1.0 / 3.0))
        qtc = np.where((qtc < MouseECG.QTC_ABS_MIN) | (qtc > MouseECG.QTC_ABS_MAX),
                       np.nan, qtc)
        ivl = ivl.copy()
        ivl["QTc_ms"] = qtc
        self.app.analysis.results["intervals"] = ivl
        self.update_interval_plots()
        fname = "Mitchell (∛RR)" if formula == "mitchell" else "Bazett (√RR)"
        self.app._set_status(f"QTc recomputed — formula: {fname}  ✓", GREEN)

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

    def compute_rolling_hrv(self) -> None:
        """Compute sliding-window HRV and render the timeline plot."""
        if self.app.detection.rpeaks_ok is None or self.app.signal.fs is None:
            self.app._set_status("Run Core Analysis first.", RED)
            return

        try:
            win_s  = max(5.0,  float(self.app.ent_roll_win.get()))
            step_s = max(1.0,  float(self.app.ent_roll_step.get()))
        except ValueError:
            self.app._set_status("Invalid window / step.", RED)
            return

        active = [m for m, cb in self.app._roll_metrics.items() if cb.get()]
        if not active:
            self.app._set_status("Sélectionner au moins une métrique.", RED)
            return

        rpeaks = self.app._windowed_peaks()
        if rpeaks is None or len(rpeaks) < 5:
            self.app._set_status("Not enough peaks in the analysis window.", ORANGE)
            return
        fs     = float(self.app.signal.fs)

        def _worker():
            t_peaks  = rpeaks / fs
            t_start  = t_peaks[0]
            t_end    = t_peaks[-1]
            starts   = np.arange(t_start, t_end - win_s + step_s * 0.5, step_s)
            rows    = []
            n_wins  = len(starts)
            for i, t0 in enumerate(starts):
                t1   = t0 + win_s
                mask = (t_peaks >= t0) & (t_peaks < t1)
                ep   = rpeaks[mask]
                if len(ep) < 5:
                    continue
                rr = np.diff(ep).astype(float) / fs * 1000
                hr   = float(60_000.0 / rr.mean()) if len(rr) else np.nan
                sdnn = float(rr.std(ddof=1))        if len(rr) > 1 else np.nan
                rmssd = float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr) > 2 else np.nan
                diffs = np.abs(np.diff(rr))
                pnn6  = float(100.0 * np.sum(diffs > MouseECG.PNN_THRESHOLD) / len(diffs)) \
                        if len(diffs) else np.nan
                rows.append({
                    "t_mid": round(t0 + win_s / 2, 2),
                    "HR":    round(hr,    1),
                    "SDNN":  round(sdnn,  2),
                    "RMSSD": round(rmssd, 2),
                    "pNN6":  round(pnn6,  1),
                    "n_beats": len(ep),
                })
                pct = int((i + 1) / max(n_wins, 1) * 100)
                if i % max(1, n_wins // 20) == 0:
                    self.app.after(0, lambda p=pct, ii=i+1, tot=n_wins:
                               self.app._set_progress(p, f"Window {ii}/{tot}…"))
            return rows

        def _done(rows):
            if not rows:
                self.app.lbl_roll_status.configure(  # type: ignore[union-attr]
                    text="No valid windows — recording too short?",
                    text_color=RED)
                return

            df = pd.DataFrame(rows)
            self.app.analysis.rolling_hrv_df = df

            ctx    = EXPERIMENTAL_CONTEXTS.get(self.app.analysis.exp_context)
            colors = {"HR": ORANGE_DARK, "SDNN": BLUE_DARK,
                      "RMSSD": GREEN_DARK, "pNN6": PURPLE}
            ylabels = {"HR": "HR (bpm)", "SDNN": "SDNN (ms)",
                       "RMSSD": "RMSSD (ms)", "pNN6": "pNN6 (%)"}
            # reference bands per metric from active context
            ref_bands: "dict[str, tuple[float,float]]" = {}
            if ctx:
                ref_bands = {
                    "HR":    (ctx.hr_lo,    ctx.hr_hi),
                    "SDNN":  (ctx.sdnn_lo,  ctx.sdnn_hi),
                    "RMSSD": (ctx.rmssd_lo, ctx.rmssd_hi),
                    "pNN6":  (ctx.pnn6_lo,  ctx.pnn6_hi),
                }

            n_plots = len(active)

            def draw_rolling(fig):
                axes = fig.subplots(n_plots, 1, sharex=True)
                if n_plots == 1:
                    axes = [axes]
                fig.subplots_adjust(hspace=0.08)
                t = df["t_mid"].values

                for ax, metric in zip(axes, active):
                    style_axes(ax)
                    y     = df[metric].values
                    color = colors[metric]

                    # Shaded reference band from context
                    if metric in ref_bands:
                        lo, hi = ref_bands[metric]
                        ax.axhspan(lo, hi, alpha=0.10, color=color,
                                   linewidth=0, zorder=0)
                        ax.axhline(lo, color=color, lw=0.6,
                                   ls="--", alpha=0.45, zorder=1)
                        ax.axhline(hi, color=color, lw=0.6,
                                   ls="--", alpha=0.45, zorder=1)

                    ax.plot(t, y, color=color, lw=1.4, zorder=3)
                    ax.fill_between(t, y, alpha=0.08, color=color, zorder=2)
                    ax.set_ylabel(ylabels[metric], fontsize=8)
                    ax.set_title(
                        f"{metric}  —  window {win_s:.0f}s · pas {step_s:.0f}s",
                        loc="left", fontsize=8)
                    if ax is not axes[-1]:
                        ax.tick_params(labelbottom=False)

                axes[-1].set_xlabel("Time (s)")
                ctx_txt = ctx.label if ctx else ""
                if ctx_txt:
                    fig.text(0.99, 0.01, f"Contexte : {ctx_txt}",
                             ha="right", va="bottom", fontsize=7,
                             color=PLOT.get("muted", "#888"),
                             transform=fig.transFigure)

            self.app._slots["rolling_hrv"].update(draw_rolling)
            n = len(df)
            self.app.lbl_roll_status.configure(  # type: ignore[union-attr]
                text=f"  {n} windows · {win_s:.0f}s · pas {step_s:.0f}s  ✓",
                text_color=GREEN)
            self.app._set_status(f"Rolling HRV — {n} windows computed", GREEN)
            self.app.tabs.set("HRV"); self.app.after(50, lambda: self.app._on_hrv_view_change("Rolling"))

        self.app._start_async_result(
            self.app.btn_roll_compute, "Computing…", _worker, _done)

    def copy_rolling_tsv(self) -> None:
        df = self.app.analysis.rolling_hrv_df
        if df is None:
            self.app._set_status("Compute Rolling HRV first.", RED)
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(self.app._df_to_tsv(df))
        self.app._set_status("Rolling HRV copié dans le presse-papiers (Excel ready)", GREEN)

    def run_analysis(self) -> None:
        if not NK_AVAILABLE:
            messagebox.showerror("Missing", "pip install neurokit2")
            return
        if self.app.signal.filtered is None or self.app.detection.rpeaks_ok is None:
            messagebox.showwarning("Not ready", "Click '\u25b6 Preview Detection' first.")
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
        self.app._start_async(
            self.app.btn_run, "Analysing\u2026", "Running HRV analysis\u2026",
            lambda: self.analysis_worker(params),
            self.on_analysis_done,
            pass_result=True,
        )

    def analysis_worker(self, params: dict) -> dict:
        """Background worker — MUST NOT write to self.

        Runs core analysis and optional artifact correction, then returns a
        plain bundle.  ``_on_analysis_done`` writes all results to self on
        the main thread, preventing races with Tk draw callbacks.
        """
        if self.app.detection.rpeaks_ok is None:
            raise RuntimeError("No peaks available — run Preview Detection first.")
        # Take a snapshot of the peaks at worker-start time.  After this point
        # the worker operates only on local variables — no self writes.
        rp = self.app.detection.rpeaks_ok.copy()

        def _prog(pct: int, msg: str) -> None:
            self.app.after(0, lambda p=pct, m=msg: self.app._set_progress(p, m))

        artifact_report = None
        if params["artifact_correction"]:
            _prog(2, "Artifact correction (auto)…")
            try:
                rp_corrected, artifact_report = correct_rr_artifacts(
                    rp, self.app.signal.fs,
                    rr_min_ms=params.get("min_rr_ms", MouseECG.RR_MIN_MS),
                    rr_max_ms=MouseECG.RR_MAX_MS,
                    window_beats=11, dev_threshold=0.20,
                    signal=self.app.signal.filtered,
                )
                rp = rp_corrected
                removed = artifact_report["n_in"] - artifact_report["n_out"]
                log.info("Artifact correction: −%d peaks (non-physio=%d ectopic=%d dup=%d)",
                         removed, artifact_report["n_nonphysio"],
                         artifact_report["n_ectopic"], artifact_report["n_duplicate"])
            except Exception as exc:
                log.warning("Artifact correction failed: %s", exc)
                artifact_report = None

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
                    f"entre {ana_t0:.1f} s et "
                    f"{'fin' if ana_t1 == 0 else f'{ana_t1:.1f} s'}.\n"
                    "Élargir la window ou la réinitialiser (bouton 'Tout').")
            rp = rp_windowed
            log.info("Analysis window applied: %.1f s → %s  (%d / %d peaks)",
                     ana_t0, f"{ana_t1:.1f} s" if ana_t1 > 0 else "end",
                     len(rp), len(rp_full))

        _prog(10, "Core analysis (RR, HR, time-domain HRV, beat template)…")
        if self.app.signal.filtered is None:
            raise RuntimeError("Signal not loaded — run Preview Detection first.")
        results = analyse_core(self.app.signal.filtered, rp, self.app.signal.fs,
                               progress_cb=lambda p, m: _prog(10 + int(p * 0.9), m))

        return {
            "results":         results,
            "rpeaks_ok":       rp_full,   # ← always full set: never overwrite with windowed
            "rpeaks_analysed": rp,         # ← windowed subset used for this analysis
            "artifact_report": artifact_report,
            "auto_epochs":     params.get("auto_epochs", False),
        }

    def on_analysis_done(self, bundle: dict) -> None:
        # ── Atomic state update (main thread) ────────────────────────────────
        self.app.analysis.results         = bundle["results"]
        # rpeaks_ok in the bundle is always the FULL artifact-corrected set,
        # never the windowed subset — so the signal view and next analysis
        # always start from the complete detection result.
        self.app.detection.rpeaks_ok       = bundle["rpeaks_ok"]
        self.app.analysis.artifact_report = bundle["artifact_report"]

        if self.app.analysis.results is None:
            return
        # Sync the peak label: show windowed count if a window was active,
        # otherwise show total artifact-corrected count.
        _rp_ok   = self.app.detection.rpeaks_ok
        _rp_used = bundle.get("rpeaks_analysed", _rp_ok)  # windowed subset
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
        n_valid = self.app.analysis.results["hr"].get("n_valid", n_peaks)
        win_str = f" (window: {n_used} peaks)" if windowed else ""
        self.app._set_status(
            f"Core analysis done — {n_used} peaks analysed{win_str} / {n_valid} valid{art_str}  |  rendering…", GREEN)
        self.app._update_kpis()
        self.app._draw_detail()
        # Enable the per-tab buttons now that core results are available
        for btn_attr in ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl", "btn_run_arrhythmia"):
            if getattr(self.app, btn_attr, None) is not None:
                getattr(self.app, btn_attr).configure(state="normal")
        # Update per-tab status labels
        if self.app.lbl_freq_status is not None:
            self.app.lbl_freq_status.configure(  # type: ignore[union-attr]
                text="  Core done — click to compute LF / HF", text_color=BLUE)
        if self.app.lbl_nonlin_status is not None:
            self.app.lbl_nonlin_status.configure(  # type: ignore[union-attr]
                text="  Core done — click to compute SampEn / DFA (slow!)", text_color="#9C27B0")
        if self.app.lbl_ivl_status is not None:
            self.app.lbl_ivl_status.configure(  # type: ignore[union-attr]
                text="  Core done — click to delineate P/Q/S/T waves", text_color=ORANGE)
        try:
            self.app._draw_core_results(
                on_complete=lambda: self.app._set_status(
                    f"Core analysis done — {n_used} peaks / {n_valid} valid{art_str}{win_str}  "
                    "| Use per-tab buttons for Freq / Non-linear / Intervals", GREEN),
                auto_epochs=bool(bundle.get("auto_epochs", False)),
            )
        except Exception:
            log.exception("_draw_core_results failed")
        # Populate interpretation tab with core values (freq/nonlinear added later)
        pass  # interpretation removed
        # Update analysis window label to reflect the window that was actually used
        if self.app.lbl_analysis_window is not None:
            t0 = self.app.analysis.t_start
            t1 = self.app.analysis.t_end
            if t0 > 0 or t1 > 0:
                dur_str = f"{t0:.1f} s → {t1:.1f} s" if t1 > 0 else f"{t0:.1f} s → fin"
                self.app.lbl_analysis_window.configure(  # type: ignore[union-attr]
                    text=f"✓  Analysed  ·  {n_used}/{n_total} peaks  ·  {dur_str}",
                    text_color=GREEN)
            else:
                self.app.lbl_analysis_window.configure(  # type: ignore[union-attr]
                    text=f"✓  Analysed  ·  {n_total} peaks  ·  full signal",
                    text_color=GREEN)
        # Enable Save Session now that we have results
        self.app._update_quality_badge()
        if self.app.btn_save_session is not None:
            self.app.btn_save_session.configure(state="normal")  # type: ignore[union-attr]
        self.app.session.dirty = True
        # Update session/template info labels
        self.app._update_session_ui(
            has_session=bool(self.app.signal.filepath and load_session(self.app.signal.filepath) is not None))

    def run_freq(self) -> None:
        """Compute frequency-domain HRV in background, then render."""
        if self.app.analysis.results is None or self.app.detection.rpeaks_ok is None:
            messagebox.showwarning("Not ready", "Run Core Analysis first.")
            return
        rp = self.app._windowed_peaks()
        if rp is None or len(rp) < 5:
            messagebox.showwarning("Not ready", "Not enough peaks in the analysis window.")
            return
        fs  = self.app.signal.fs
        _gen = getattr(self.app, "_generation", 0)  # snapshot — detect file change
        if self.app.lbl_freq_status is not None:
            self.app.lbl_freq_status.configure(text="  Computing…", text_color=ORANGE)  # type: ignore[union-attr]

        def _worker():
            def _prog(p, m):
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))
            return analyse_hrv_freq(rp, fs, progress_cb=_prog)

        def _done(result):
            if self.app.analysis.results is None or getattr(self.app, "_generation", 0) != _gen:
                log.info("_run_freq: stale result discarded (file changed)")
                return
            self.app.analysis.results["hrv_freq"] = result
            results: dict = self.app.analysis.results  # narrow for type checkers
            tasks = [
                ("PSD",       lambda: self.app._plot_psd(results)),
                ("HRV radar", lambda: self.app._plot_radar(results)),
                ("HRV tables (freq)", lambda: self.app._plot_hrv_tables(results)),
                ("Summary",   lambda: self.app._plot_summary(results)),
            ]
            n_lf = n_hf = "—"
            try:
                n_lf = f"{float(result['HRV_LF'].values[0])*100:.1f}%"
                n_hf = f"{float(result['HRV_HF'].values[0])*100:.1f}%"
            except Exception as _exc:
                log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 5940, _exc)
            if self.app.lbl_freq_status is not None:
                self.app.lbl_freq_status.configure(  # type: ignore[union-attr]
                    text=f"  Done  LF={n_lf}  HF={n_hf}", text_color=GREEN)
            self.app._run_plot_chain(
                tasks,
                on_complete=lambda: self.app._set_status("Frequency HRV done", GREEN))

        self.app._start_async_result(self.app.btn_run_freq, "Computing…", _worker, _done)  # type: ignore[arg-type]

    def run_nonlinear(self) -> None:
        """Compute non-linear HRV in background, then render."""
        if self.app.analysis.results is None or self.app.detection.rpeaks_ok is None:
            messagebox.showwarning("Not ready", "Run Core Analysis first.")
            return
        rp = self.app._windowed_peaks()
        if rp is None or len(rp) < 5:
            messagebox.showwarning("Not ready", "Not enough peaks in the analysis window.")
            return
        sig = self.app.signal.filtered
        if sig is None:
            messagebox.showwarning("Not ready", "Signal not loaded.")
            return
        fs  = self.app.signal.fs
        _gen = getattr(self.app, "_generation", 0)
        if self.app.lbl_nonlin_status is not None:
            self.app.lbl_nonlin_status.configure(  # type: ignore[union-attr]
                text="  Computing SampEn / DFA… (may take 30 s+)", text_color=ORANGE)

        def _worker():
            def _prog(p, m):
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))
            return analyse_hrv_nonlinear(sig, rp, fs, progress_cb=_prog)

        def _done(result):
            if self.app.analysis.results is None or getattr(self.app, "_generation", 0) != _gen:
                log.info("_run_nonlinear: stale result discarded (file changed)")
                return
            self.app.analysis.results["hrv_nonlin"] = result
            results: dict = self.app.analysis.results  # narrow for type checkers
            tasks = [
                ("Non-linear metrics", lambda: self.app._plot_nonlinear(results)),
                ("Summary",            lambda: self.app._plot_summary(results)),
            ]
            sampen = "—"
            try:
                sampen = f"{float(result['HRV_SampEn'].values[0]):.3f}"
            except Exception as _exc:
                log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 5979, _exc)
            if self.app.lbl_nonlin_status is not None:
                self.app.lbl_nonlin_status.configure(  # type: ignore[union-attr]
                    text=f"  Done  SampEn={sampen}", text_color=GREEN)
            self.app._run_plot_chain(
                tasks,
                on_complete=lambda: self.app._set_status("Non-linear HRV done", GREEN))

        self.app._start_async_result(self.app.btn_run_nonlin, "Computing…", _worker, _done)  # type: ignore[arg-type]

    def run_intervals(self) -> None:
        """Compute interval delineation in background, then launch verifier."""
        if self.app.analysis.results is None or self.app.signal.filtered is None or self.app.detection.rpeaks_ok is None:
            messagebox.showwarning("Not ready", "Run Core Analysis first.")
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

        # Load or create template — always pass it so auto-update works
        _wt = self.app.analysis.wave_template
        if _wt is None:
            _wt = WaveTemplate.load()
            self.app.analysis.wave_template = _wt
        wt_for_worker = _wt   # always pass (confirmed or not)
        permissive_for_worker = (bool(self.app.sw_permissive.get())  # type: ignore[union-attr]
                                  if self.app.sw_permissive is not None else False)

        def _worker() -> "tuple[pd.DataFrame, np.ndarray, np.ndarray]":
            def _prog(p: int, m: str) -> None:
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))

            # Build beat matrix here so it can be passed to the verifier
            fixed_hw  = int(MouseECG.BEAT_HALF_WIN_S * fs)
            rr_samp   = np.diff(rp) if len(rp) > 1 else np.array([fixed_hw * 2])
            rr_min_s  = int(rr_samp.min()) if len(rr_samp) else fixed_hw * 2
            half_win  = max(20, min(fixed_hw, int(rr_min_s * 0.45)))
            bt_ms     = np.arange(-half_win, half_win) / fs * 1000
            mask_v    = (rp - half_win >= 0) & (rp + half_win < len(sig))
            valid_rp  = rp[mask_v]
            if len(valid_rp) >= 2:
                idx_mat  = valid_rp[:, None] + np.arange(-half_win, half_win)
                beat_mat = sig[idx_mat].astype(float)
            else:
                beat_mat = np.zeros((0, half_win * 2))

            df = analyse_intervals(sig, rp, fs, rr,
                                   progress_cb=_prog,
                                   wave_template=wt_for_worker,
                                   permissive_bounds=permissive_for_worker)
            return df, beat_mat, bt_ms

        def _done(result: "tuple[pd.DataFrame, np.ndarray, np.ndarray]") -> None:
            if self.app.analysis.results is None or getattr(self.app, "_generation", 0) != _gen:
                log.info("_run_intervals: stale result discarded (file changed)")
                return
            df, beat_mat, bt_ms = result
            self.app.analysis.results["intervals"] = df

            interval_cols = [c for c in ["PR_ms", "QRS_ms", "QT_ms"] if c in df.columns]
            n_ok    = int((~df[interval_cols].isna().any(axis=1)).sum()) if interval_cols else 0  # type: ignore[arg-type]
            n_total = len(df)

            wt        = self.app.analysis.wave_template
            tmpl_note = f"  template:{wt.source}" if wt else ""
            note      = f"  {n_ok}/{n_total} complete — verify in panel below{tmpl_note}"
            note_color = GREEN if n_ok > 0 else ORANGE
            if n_ok == 0 and n_total > 0:
                note += "  ⚠ check template / filters"
            if self.app.lbl_ivl_status is not None:
                self.app.lbl_ivl_status.configure(text=note, text_color=note_color)  # type: ignore[union-attr]

            # Launch interactive verifier (replaces static beat strip)
            self.launch_interval_verifier(df, beat_mat, bt_ms)

            # Plot distributions immediately with all beats
            results: dict = self.app.analysis.results
            self.app._run_plot_chain(
                [("ECG intervals", lambda: self.app._plot_intervals(results)),
                 ("Summary",       lambda: self.app._plot_summary(results))],
                on_complete=lambda: self.app._set_status(
                    f"Interval delineation done — {n_ok}/{n_total} beats  "
                    "| verify in panel below then click Finalise", note_color))

        self.app._start_async_result(self.app.btn_run_ivl, "Delineating…", _worker, _done)  # type: ignore[arg-type]

    def launch_interval_verifier(
        self,
        df:       "pd.DataFrame",
        beat_mat: "np.ndarray",
        beat_time:"np.ndarray",
    ) -> None:
        """Instantiate IntervalVerifierPanel in the intervals tab.

        Called from _run_intervals _done callback (main thread only).
        The verifier renders into the intervals_ecg CanvasSlot and places
        its navigation bar into frm_ivl_nav.
        """
        slot = self.app._slots.get("intervals_ecg")
        nav  = self.app.frm_ivl_nav
        if slot is None or nav is None:
            return

        def _on_finalise(verified_df: "pd.DataFrame") -> None:
            """Replace stored intervals with verified subset; re-plot."""
            if self.app.analysis.results is None:
                return
            self.app.analysis.results["intervals"] = verified_df
            results: dict = self.app.analysis.results
            n_ok    = int((~verified_df[["PR_ms","QRS_ms","QT_ms"]]
                           .isna().any(axis=1)).sum()) if all(  # type: ignore[arg-type]
                               c in verified_df.columns for c in
                               ["PR_ms","QRS_ms","QT_ms"]) else 0
            if self.app.lbl_ivl_status is not None:
                self.app.lbl_ivl_status.configure(  # type: ignore[union-attr]
                    text=f"  ✓ Finalised — {n_ok}/{len(verified_df)} beats accepted",
                    text_color=GREEN)
            self.app._run_plot_chain(
                [("ECG intervals", lambda: self.app._plot_intervals(results)),
                 ("Summary",       lambda: self.app._plot_summary(results))],
                on_complete=lambda: self.app._set_status(
                    f"Intervals finalised — {n_ok}/{len(verified_df)} beats", GREEN))

        n_verifier = min(len(df), len(beat_mat))
        self.app._ivl_verifier = IntervalVerifierPanel(
            df        = df.iloc[:n_verifier],
            beat_mat  = beat_mat[:n_verifier],
            beat_time = beat_time,
            fs        = self.app.signal.fs,
            slot      = slot,
            nav_frame = nav,
            on_finalise = _on_finalise,
        )

    def compute_epochs(self) -> None:
        """Compute epoch-level HRV in a background thread to keep the UI responsive.

        nk.hrv_time() is called once per epoch.  On long recordings with many
        short epochs this adds up to several seconds — enough to freeze the UI
        noticeably.  The calculation is therefore moved to a daemon thread via
        _start_async_result, exactly like _run_freq / _run_nonlinear.
        """
        if self.app.detection.rpeaks_ok is None or len(self.app.detection.rpeaks_ok) < 10:
            messagebox.showwarning("No data", "Run Preview Detection first.")
            return
        if self.app.signal.time is None:
            return

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
        if len(starts) < 2:
            messagebox.showwarning(
                "Too few epochs",
                f"Recording too short for {epoch_s:.0f}s epochs. "
                f"Try a shorter epoch (e.g. {int(dur // 3)}s).")
            return

        def _worker():
            """Runs on background thread — no Tkinter access allowed."""
            rows = []
            n_ep = len(starts)
            for idx_ep, t0 in enumerate(starts):
                t1    = t0 + epoch_s
                ep_rp = rp[(rp / fs >= t0) & (rp / fs < t1)]
                if len(ep_rp) < 5:
                    continue
                rr = np.diff(ep_rp).astype(float) / fs * 1000
                try:
                    assert nk is not None  # NK_AVAILABLE checked by caller
                    hrv_ep = nk.hrv_time(ep_rp, sampling_rate=int(fs), show=False)
                    sdnn   = float(hrv_ep["HRV_SDNN"].values[0])
                    rmssd  = float(hrv_ep["HRV_RMSSD"].values[0])
                except Exception as exc:
                    log.warning("Epoch hrv_time failed, manual calc: %s", exc)
                    sdnn  = float(rr.std())
                    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
                rows.append({
                    "Epoch_start_s": round(t0, 1),
                    "Epoch_end_s":   round(t1, 1),
                    "N_beats":       len(ep_rp),
                    "HR_mean":       round(float(60_000 / rr.mean()), 1),
                    "MeanNN":        round(float(rr.mean()), 1),
                    "SDNN":          round(sdnn, 2),
                    "RMSSD":         round(rmssd, 2),
                })
                # Progress via after() — safe cross-thread call, throttled to every 5%
                pct = int((idx_ep + 1) / max(n_ep, 1) * 100)
                if idx_ep % max(1, n_ep // 20) == 0:
                    self.app.after(0, lambda p=pct, e=idx_ep+1, tot=n_ep:
                               self.app._set_progress(p, f"Epoch {e}/{tot}…"))
            return rows

        def _done(rows):
            if not rows:
                messagebox.showwarning("No epochs", "No valid epochs found.")
                return
            df    = pd.DataFrame(rows)
            self.app.analysis.epoch_df = df
            t_mid = (df["Epoch_start_s"] + df["Epoch_end_s"]) / 2

            plot_specs = [
                ("HR_mean", "HR (bpm)",   ORANGE_DARK, "Heart Rate per Epoch"),
                ("SDNN",    "SDNN (ms)",  BLUE_DARK, "SDNN per Epoch"),
                ("RMSSD",   "RMSSD (ms)", GREEN_DARK, "RMSSD per Epoch"),
            ]

            def draw_epochs(fig):
                axes = fig.subplots(3, 1, sharex=True)
                for ax, (col, ylabel, color, title) in zip(axes, plot_specs):
                    style_axes(ax)
                    y = df[col].values
                    ax.plot(t_mid, y, color=color, lw=1.5, marker="o", ms=3.5)
                    ax.fill_between(t_mid, y, alpha=0.10, color=color)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title, loc="left")
                    if ax is not axes[-1]:
                        ax.tick_params(labelbottom=False)
                axes[-1].set_xlabel("Time (s)")

            self.app._slots["epochs"].update(draw_epochs)
            self.app._set_textbox(self.app.txt_epochs, df.to_string(index=False),
                              tsv=self.app._df_to_tsv(df))
            n_ep = len(df)
            # Update the label in the Epochs tab header
            self.app.lbl_epoch_count.configure(
                text=f"{n_ep} epochs × {epoch_s:.0f}s", text_color=BLUE)
            # Update the label in the Summary tab (shows last-computed epoch info)
            self.app.lbl_epoch_info.configure(
                text=f"Last epoch run: {n_ep} × {epoch_s:.0f}s", text_color=MUTED)
            self.app.tabs.set("HRV"); self.app.after(50, lambda: self.app._on_hrv_view_change("Epochs"))
            self.app._set_status(f"Epoch analysis done — {n_ep} epochs", GREEN)

        self.app._start_async_result(self.app.btn_compute_epochs, "Computing…", _worker, _done)

    @staticmethod
    def wilcoxon_test(a: "np.ndarray", b: "np.ndarray") -> "tuple[float, str]":
        """Mann-Whitney U test for two independent RR series.

        Returns (p_value, interpretation_string).
        Falls back gracefully if scipy is unavailable.
        """
        try:
            from scipy.stats import mannwhitneyu
            if len(a) < 5 or len(b) < 5:
                return float("nan"), "n<5"
            result: Any = mannwhitneyu(a, b, alternative="two-sided")
            p_val = getattr(result, "pvalue", None)
            p = float(p_val if p_val is not None else result[1])
            interp = ("**** p<0.0001" if p < 0.0001 else
                      "*** p<0.001"   if p < 0.001  else
                      "** p<0.01"     if p < 0.01   else
                      "* p<0.05"      if p < 0.05   else
                      "ns")
            return p, interp
        except Exception:
            return float("nan"), "—"
