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

from ecg.core.models import (
    MouseECG, EXPERIMENTAL_CONTEXTS, _CONTEXT_FIELD_MAP, ArrhythmiaEvent,
)
from ecg.core.detection import classify_arrhythmias, correct_rr_artifacts
from ecg.core.analysis import analyse_core, analyse_hrv_freq, analyse_hrv_nonlinear, analyse_intervals
from ecg.core.wave_template import WaveTemplate
from ecg.io.session import load_session
from ecg.ui.sidebar import IntervalVerifierPanel
from ecg.ui.theme import (
    PLOT, RED, RED_MID, RED_LIGHT, AMBER, BLUE, BLUE_DARK, BLUE_MID,
    GREEN, GREEN_DARK, GREEN_MID, ORANGE, ORANGE_DARK, ORANGE_DEEP,
    PURPLE, CORAL, NAVY, GRAY_LIGHT, CYAN_BRIGHT, MUTED, TEXT, LIGHT,
    CARD, BORDER, BORDER2, NK_AVAILABLE, nk,
    FONT_SIDEBAR_HDR, FONT_BADGE, FONT_HINT, FONT_SUBSECTION, FONT_KPI_LABEL,
    FONT_SMALL,
    SPACE_XS, SPACE_S, SPACE_L,
)
from ecg.ui.plots import style_axes

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class AnalysisController:
    # Per-epoch/per-window beat count below which SDNN/RMSSD/pNN6 are still
    # computed (down to the hard inclusion floor of 5 beats in
    # compute_epochs/compute_rolling_hrv below) but flagged as low-confidence
    # -- an HRV metric from under ~10 intervals is noisy. Shared by both
    # methods so "low confidence" means the same thing in both panels.
    MIN_CONFIDENT_BEATS = 10

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
        _gen = getattr(self.app, "_generation", 0)  # snapshot — detect file change

        # AV conduction-delay detection needs per-beat PR data from Interval
        # delineation, which is a separate, optional analysis step -- only
        # wire it in when it has actually been run AND its beat count still
        # matches these same windowed rpeaks (both come from
        # _windowed_peaks(), but the analysis window could have been changed
        # between the two runs). Mismatched length -> skip rather than risk
        # silently misaligning PR values to the wrong beats.
        pr_ms = None
        ivl = (self.app.analysis.results or {}).get("intervals")
        if ivl is not None and not ivl.empty and "PR_ms" in ivl.columns and len(ivl) == len(rpeaks):
            pr_ms = ivl["PR_ms"].values.astype(float)

        def _worker():
            def _prog(p, m):
                self.app.after(0, lambda pp=p, mm=m: self.app._set_progress(pp, mm))
            return classify_arrhythmias(
                rpeaks, fs, ctx_key,
                baseline_s=baseline_s,
                brady_pct=brady_pct,
                min_brady_beats=min_beats,
                progress_cb=_prog,
                pr_ms=pr_ms,
            )

        def _done(events: "list[ArrhythmiaEvent]"):
            if getattr(self.app, "_generation", 0) != _gen:
                log.info("run_arrhythmia_analysis: stale result discarded (file changed)")
                return
            self.app.analysis.arrhythmia_events = events
            self.app.analysis.arr_selected_idx  = -1

            sev_colors = {"alert": RED_MID, "warning": AMBER, "info": BLUE_MID}
            kind_icons = {
                "bradycardia": "🔵", "tachycardia": "🔴", "pause": "⏸",
                "esv_run": "⚡", "irregular_run": "〰", "av_delay": "⏳",
            }
            # Per-KIND (not per-severity) colors for the event ribbon --
            # sev_colors above only has 3 buckets (alert/warning/info), so
            # e.g. "pause" and "esv_run" (both "alert") would be visually
            # identical on a severity-colored strip; this is what actually
            # lets event-type clustering be read at a glance.
            kind_colors = {
                "bradycardia": BLUE_MID, "tachycardia": RED_MID, "pause": RED,
                "esv_run": AMBER, "irregular_run": PURPLE, "av_delay": CYAN_BRIGHT,
            }

            # ── RR timeline (right panel, initial state) ──────
            t_peaks = rpeaks / fs
            rr_ms   = np.diff(rpeaks).astype(float) / fs * 1000
            t_rr    = (t_peaks[:-1] + t_peaks[1:]) / 2

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
                gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 6], hspace=0.12,
                             left=0.08, right=0.98, top=0.80, bottom=0.10)
                ax_ribbon = fig.add_subplot(gs[0, 0])
                ax        = fig.add_subplot(gs[1, 0], sharex=ax_ribbon)
                style_axes(ax_ribbon)
                style_axes(ax)

                # ── Event ribbon: ethogram-style strip, colored by kind ──
                by_kind: "dict[str, list[tuple[float, float]]]" = {}
                for ev in events:
                    by_kind.setdefault(ev.kind, []).append(
                        (ev.t_start, max(ev.duration_s, 0.05)))
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
                                      lw=5, label=k.replace("_", " ").title())
                               for k in by_kind]
                    # top=0.80 above reserves real figure-level margin for
                    # this -- fig.legend() (figure-relative 0-1 coords) is
                    # still used over ax_ribbon.legend() since it isn't
                    # bound by the ribbon axes' own (short) height.
                    fig.legend(handles=handles, loc="upper center",
                              bbox_to_anchor=(0.5, 0.99), ncol=min(len(handles), 6),
                              fontsize=6.5, frameon=False,
                              labelcolor=PLOT.get("text", GRAY_LIGHT))

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
                               label=f"Brady threshold −{bpct:.0f}%")
                    ax.axhline(tachy_rr, color=BLUE_MID, lw=0.9,
                               ls="--", alpha=0.65, zorder=4,
                               label=f"Tachy threshold +{bpct:.0f}%")

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
                    text="No abnormal events detected\nfor the active context.",
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
            self.app._set_status(f"Abnormal-event classification — {n} episode(s)", GREEN)
            self.app.tabs.set("⚠ Abnormal Events")

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
        self.app._set_status(msg + "  — rerun Core Analysis to refresh HRV", ORANGE)
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
            self.app._set_status("Run abnormal-event classification first.", RED)
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(tsv)
        self.app._set_status("Abnormal events copied to clipboard (Excel ready)", GREEN)

    def qtc_formula(self) -> str:
        """Return 'mitchell', 'bazett', or 'hodges' from the Intervals tab combo selector."""
        try:
            val = self.app.cb_qtc_formula.get()  # type: ignore
            if "Bazett" in val:
                return "bazett"
            if "Hodges" in val:
                return "hodges"
            return "mitchell"
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
        elif formula == "hodges":
            # Hodges: QTc = QT + 1.75*(HR-60), HR in bpm. Reported in the
            # literature as the best-performing corrector among Bazett/
            # Fridericia/Hodges when compared empirically in rats and mice
            # (Bazett/Fridericia are calibrated to human ~1000ms RR and
            # documented to over/under-correct badly at rodent RR scale).
            hr_bpm = 60_000.0 / np.clip(rr_arr, MouseECG.RR_MIN_MS, None)
            qtc = qt_ms + 1.75 * (hr_bpm - 60.0)
        else:
            qtc = qt_ms / (rr_s ** (1.0 / 3.0))
        qtc = np.where((qtc < MouseECG.QTC_ABS_MIN) | (qtc > MouseECG.QTC_ABS_MAX),
                       np.nan, qtc)
        ivl = ivl.copy()
        ivl["QTc_ms"] = qtc
        self.app.analysis.results["intervals"] = ivl
        self.update_interval_plots()
        fname = {"mitchell": "Mitchell (∛RR)", "bazett": "Bazett (√RR)",
                 "hodges": "Hodges (linear HR)"}[formula]
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
                # Poincare SD1/SD2 -- same closed-form definitions as the
                # Non-linear panel (SD1 from successive-difference variance,
                # SD2 from total variance), cheap enough to always compute.
                if len(rr) > 2:
                    sd1 = float(np.sqrt(0.5 * np.var(diffs, ddof=1)))
                    sd2 = float(np.sqrt(max(2.0 * np.var(rr, ddof=1) - 0.5 * np.var(diffs, ddof=1), 0.0)))
                else:
                    sd1 = sd2 = np.nan
                # LF_nu/HF_nu/LF_HF -- reuses the exact same analyse_hrv_freq()
                # path as the Frequency panel (mouse-specific bands, Welch
                # PSD, its own MIN_BEATS_SPECTRAL guard). Returns an empty
                # DataFrame -- NaN below -- rather than a number when a
                # window is too short for a meaningful spectral estimate,
                # instead of silently fabricating one.
                lf_nu = hf_nu = lf_hf = np.nan
                if need_freq:
                    try:
                        fdf = analyse_hrv_freq(ep, fs)
                        if not fdf.empty:
                            lf_nu = float(fdf["HRV_LFn"].values[0]) * 100.0
                            hf_nu = float(fdf["HRV_HFn"].values[0]) * 100.0
                            lf_hf = float(fdf["HRV_LFHF"].values[0])
                    except Exception as exc:
                        log.debug("compute_rolling_hrv: analyse_hrv_freq failed at t=%.1f: %s", t0, exc)
                rows.append({
                    "t_mid": round(t0 + win_s / 2, 2),
                    "HR":    round(hr,    1),
                    "SDNN":  round(sdnn,  2),
                    "RMSSD": round(rmssd, 2),
                    "pNN6":  round(pnn6,  1),
                    "SD1":   round(sd1, 3) if np.isfinite(sd1) else np.nan,
                    "SD2":   round(sd2, 3) if np.isfinite(sd2) else np.nan,
                    "LF_nu": round(lf_nu, 1) if np.isfinite(lf_nu) else np.nan,
                    "HF_nu": round(hf_nu, 1) if np.isfinite(hf_nu) else np.nan,
                    "LF_HF": round(lf_hf, 3) if np.isfinite(lf_hf) else np.nan,
                    "n_beats": len(ep),
                })
                pct = int((i + 1) / max(n_wins, 1) * 100)
                if i % max(1, n_wins // 20) == 0:
                    self.app.after(0, lambda p=pct, ii=i+1, tot=n_wins:
                               self.app._set_progress(p, f"Window {ii}/{tot}…"))
            return rows

        def _done(rows):
            if getattr(self.app, "_generation", 0) != _gen:
                log.info("compute_rolling_hrv: stale result discarded (file changed)")
                return
            if not rows:
                self.app.lbl_roll_status.configure(  # type: ignore[union-attr]
                    text="No valid windows — recording too short?",
                    text_color=RED)
                return

            df = pd.DataFrame(rows)
            self.app.analysis.rolling_hrv_df = df
            low_n = df["n_beats"].values < self.MIN_CONFIDENT_BEATS

            ctx    = EXPERIMENTAL_CONTEXTS.get(self.app.analysis.exp_context)
            colors = {"HR": ORANGE_DARK, "SDNN": BLUE_DARK,
                      "RMSSD": GREEN_DARK, "pNN6": PURPLE,
                      "SD1": CORAL, "SD2": CYAN_BRIGHT,
                      "LF_nu": BLUE_MID, "HF_nu": GREEN, "LF_HF": AMBER}
            ylabels = {"HR": "HR (bpm)", "SDNN": "SDNN (ms)",
                       "RMSSD": "RMSSD (ms)", "pNN6": "pNN6 (%)",
                       "SD1": "SD1 (ms)", "SD2": "SD2 (ms)",
                       "LF_nu": "LF (n.u., %)", "HF_nu": "HF (n.u., %)",
                       "LF_HF": "LF/HF"}
            # reference bands per metric from active context
            ref_bands: "dict[str, tuple[float,float]]" = {}
            if ctx:
                ref_bands = {
                    "HR":    (ctx.hr_lo,    ctx.hr_hi),
                    "SDNN":  (ctx.sdnn_lo,  ctx.sdnn_hi),
                    "RMSSD": (ctx.rmssd_lo, ctx.rmssd_hi),
                    "pNN6":  (ctx.pnn6_lo,  ctx.pnn6_hi),
                    "SD1":   (ctx.sd1_lo,   ctx.sd1_hi),
                    "SD2":   (ctx.sd2_lo,   ctx.sd2_hi),
                    "LF_nu": (ctx.lf_lo,    ctx.lf_hi),
                    "HF_nu": (ctx.hf_lo,    ctx.hf_hi),
                    "LF_HF": (ctx.lfhf_lo,  ctx.lfhf_hi),
                }

            n_plots = len(active)

            def draw_rolling(fig):
                axes = fig.subplots(n_plots, 1, sharex=True)
                if n_plots == 1:
                    axes = [axes]
                # subplots_adjust() is a no-op (with a UserWarning) once
                # constrained_layout is active on fig -- partial-override the
                # engine's own hspace instead so the stacked panels keep
                # their tighter author-tuned spacing (vs. CanvasSlot's wider
                # default _CL_SPACE) rather than silently losing the tuning.
                try:
                    fig.set_constrained_layout_pads(hspace=0.08)
                except Exception as exc:
                    log.debug("draw_rolling: set_constrained_layout_pads failed: %s", exc)
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
                    # Low-confidence windows (< MIN_CONFIDENT_BEATS beats): open,
                    # lighter markers instead of filled ones -- same convention
                    # as compute_epochs' draw_epochs().
                    if low_n.any():
                        ax.plot(t[low_n], y[low_n], "o", ms=4.5, mfc="none",
                                mec=color, mew=1.0, alpha=0.6, zorder=4)
                    if (~low_n).any():
                        ax.plot(t[~low_n], y[~low_n], "o", ms=3.0, color=color, zorder=4)
                    ax.set_ylabel(ylabels[metric], fontsize=8)
                    ax.set_title(
                        f"{metric}  —  window {win_s:.0f}s · step {step_s:.0f}s",
                        loc="left", fontsize=8)
                    if ax is not axes[-1]:
                        ax.tick_params(labelbottom=False)

                axes[-1].set_xlabel("Time (s)")
                ctx_txt = ctx.label if ctx else ""
                if ctx_txt:
                    fig.text(0.99, 0.01, f"Context: {ctx_txt}",
                             ha="right", va="bottom", fontsize=7,
                             color=PLOT.get("muted", "#888"),
                             transform=fig.transFigure)

            self.app._slots["rolling_hrv"].update(draw_rolling)
            table_text = df.to_string(index=False)
            if low_n.any():
                table_text += (
                    f"\n\n  * {int(low_n.sum())} window(s) have fewer than "
                    f"{self.MIN_CONFIDENT_BEATS} beats — SDNN/RMSSD/pNN6 for "
                    f"those rows are noisy (hollow markers on the plot above).")
            self.app._set_textbox(self.app.txt_rolling, table_text,
                              tsv=self.app._df_to_tsv(df))
            n = len(df)
            self.app.lbl_roll_status.configure(  # type: ignore[union-attr]
                text=f"  {n} windows · {win_s:.0f}s · step {step_s:.0f}s  ✓",
                text_color=GREEN)
            self.app._set_status(f"Rolling HRV — {n} windows computed", GREEN)
            self.app.tabs.set("💓 HRV"); self.app.after(50, lambda: self.app._on_hrv_view_change("Rolling"))

        self.app._start_async_result(
            self.app.btn_roll_compute, "Computing…", _worker, _done)

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
                log.debug("run_freq: LF/HF% formatting failed: %s", _exc, exc_info=True)
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
            return analyse_hrv_nonlinear(rp, fs, progress_cb=_prog)

        def _done(result):
            if self.app.analysis.results is None or getattr(self.app, "_generation", 0) != _gen:
                log.info("_run_nonlinear: stale result discarded (file changed)")
                return
            self.app.analysis.results["hrv_nonlin"] = result
            results: dict = self.app.analysis.results  # narrow for type checkers
            tasks = [
                ("Non-linear metrics", lambda: self.app._plot_nonlinear(results)),
                # The radar's SD1/SD2/SampEn axes come from hrv_nonlin -- redraw
                # it here too (not just from run_freq()) so it doesn't stay
                # stuck showing a stale/incomplete profile when Non-linear is
                # computed after (or without) Frequency having been run first.
                ("HRV radar",          lambda: self.app._plot_radar(results)),
                ("Summary",            lambda: self.app._plot_summary(results)),
            ]
            sampen = "—"
            try:
                sampen = f"{float(result['HRV_SampEn'].values[0]):.3f}"
            except Exception as _exc:
                log.debug("run_nonlinear: SampEn formatting failed: %s", _exc, exc_info=True)
            # Disclose beat-count truncation (analyse_hrv_nonlinear caps at
            # max_beats=1000 for runtime -- on a long recording that's only
            # the opening minutes, and there was previously no on-screen
            # sign of this at all, only a log line).
            trunc_note = ""
            if result.attrs.get("truncated"):
                trunc_note = (f"  ·  first {result.attrs['n_beats_used']:,} of "
                               f"{result.attrs['n_beats_total']:,} beats")
            if self.app.lbl_nonlin_status is not None:
                self.app.lbl_nonlin_status.configure(  # type: ignore[union-attr]
                    text=f"  Done  SampEn={sampen}{trunc_note}",
                    text_color=ORANGE if trunc_note else GREEN)
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
        _gen = getattr(self.app, "_generation", 0)  # snapshot — detect file change
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
                    # ddof=1 (sample std) to match nk.hrv_time's own convention
                    # and compute_rolling_hrv's manual computation below --
                    # this previously defaulted to ddof=0 (population std),
                    # silently disagreeing with both whenever this fallback
                    # actually triggered.
                    sdnn  = float(rr.std(ddof=1)) if len(rr) > 1 else float("nan")
                    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
                diffs = np.abs(np.diff(rr))
                pnn6  = float(100.0 * np.sum(diffs > MouseECG.PNN_THRESHOLD) / len(diffs)) \
                        if len(diffs) else float("nan")
                rows.append({
                    "Epoch_start_s": round(t0, 1),
                    "Epoch_end_s":   round(t1, 1),
                    "N_beats":       len(ep_rp),
                    "HR_mean":       round(float(60_000 / rr.mean()), 1),
                    "MeanNN":        round(float(rr.mean()), 1),
                    "SDNN":          round(sdnn, 2),
                    "RMSSD":         round(rmssd, 2),
                    "pNN6":          round(pnn6, 1),
                })
                # Progress via after() — safe cross-thread call, throttled to every 5%
                pct = int((idx_ep + 1) / max(n_ep, 1) * 100)
                if idx_ep % max(1, n_ep // 20) == 0:
                    self.app.after(0, lambda p=pct, e=idx_ep+1, tot=n_ep:
                               self.app._set_progress(p, f"Epoch {e}/{tot}…"))
            return rows

        def _done(rows):
            if getattr(self.app, "_generation", 0) != _gen:
                log.info("compute_epochs: stale result discarded (file changed)")
                return
            if not rows:
                messagebox.showwarning("No epochs", "No valid epochs found.")
                return
            df    = pd.DataFrame(rows)
            self.app.analysis.epoch_df = df
            t_mid = (df["Epoch_start_s"] + df["Epoch_end_s"]) / 2
            low_n = df["N_beats"].values < self.MIN_CONFIDENT_BEATS

            # Reference bands from the active experimental context -- same
            # source and pattern as compute_rolling_hrv's draw_rolling().
            ctx = EXPERIMENTAL_CONTEXTS.get(self.app.analysis.exp_context)
            ref_bands: "dict[str, tuple[float, float]]" = {}
            if ctx:
                ref_bands = {
                    "HR_mean": (ctx.hr_lo,    ctx.hr_hi),
                    "SDNN":    (ctx.sdnn_lo,  ctx.sdnn_hi),
                    "RMSSD":   (ctx.rmssd_lo, ctx.rmssd_hi),
                    "pNN6":    (ctx.pnn6_lo,  ctx.pnn6_hi),
                }

            plot_specs = [
                ("HR_mean", "HR (bpm)",   ORANGE_DARK, "Heart Rate per Epoch"),
                ("SDNN",    "SDNN (ms)",  BLUE_DARK,   "SDNN per Epoch"),
                ("RMSSD",   "RMSSD (ms)", GREEN_DARK,  "RMSSD per Epoch"),
                ("pNN6",    "pNN6 (%)",   PURPLE,      "pNN6 per Epoch"),
            ]

            def draw_epochs(fig):
                axes = fig.subplots(4, 1, sharex=True)
                try:
                    fig.set_constrained_layout_pads(hspace=0.08)
                except Exception as exc:
                    log.debug("draw_epochs: set_constrained_layout_pads failed: %s", exc)
                for ax, (col, ylabel, color, title) in zip(axes, plot_specs):
                    style_axes(ax)
                    y = df[col].values
                    if col in ref_bands:
                        lo, hi = ref_bands[col]
                        ax.axhspan(lo, hi, alpha=0.10, color=color, linewidth=0, zorder=0)
                        ax.axhline(lo, color=color, lw=0.6, ls="--", alpha=0.45, zorder=1)
                        ax.axhline(hi, color=color, lw=0.6, ls="--", alpha=0.45, zorder=1)
                    ax.plot(t_mid, y, color=color, lw=1.5, zorder=3)
                    ax.fill_between(t_mid, y, alpha=0.10, color=color, zorder=2)
                    # Low-confidence epochs (< MIN_CONFIDENT_BEATS beats): open,
                    # lighter markers instead of filled ones -- same data,
                    # visually distinguished rather than silently identical.
                    if low_n.any():
                        ax.plot(t_mid[low_n], y[low_n], "o", ms=4.5, mfc="none",
                                mec=color, mew=1.0, alpha=0.6, zorder=4)
                    if (~low_n).any():
                        ax.plot(t_mid[~low_n], y[~low_n], "o", ms=3.5, color=color, zorder=4)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title, loc="left")
                    if ax is not axes[-1]:
                        ax.tick_params(labelbottom=False)
                axes[-1].set_xlabel("Time (s)")

            self.app._slots["epochs"].update(draw_epochs)
            table_text = df.to_string(index=False)
            if low_n.any():
                table_text += (
                    f"\n\n  * {int(low_n.sum())} epoch(s) have fewer than "
                    f"{self.MIN_CONFIDENT_BEATS} beats — SDNN/RMSSD/pNN6 for "
                    f"those rows are noisy (hollow markers on the plot above).")
            self.app._set_textbox(self.app.txt_epochs, table_text,
                              tsv=self.app._df_to_tsv(df))
            n_ep = len(df)
            # Update the label in the Epochs tab header
            self.app.lbl_epoch_count.configure(
                text=f"{n_ep} epochs × {epoch_s:.0f}s", text_color=BLUE)
            # Update the label in the Summary tab (shows last-computed epoch info)
            self.app.lbl_epoch_info.configure(
                text=f"Last epoch run: {n_ep} × {epoch_s:.0f}s", text_color=MUTED)
            self.app.tabs.set("💓 HRV"); self.app.after(50, lambda: self.app._on_hrv_view_change("Epochs"))
            self.app._set_status(f"Epoch analysis done — {n_ep} epochs", GREEN)

        self.app._start_async_result(self.app.btn_compute_epochs, "Computing…", _worker, _done)

    @staticmethod
    def mannwhitney_test(a: "np.ndarray", b: "np.ndarray") -> "tuple[float, str]":
        """Mann-Whitney U test for two independent RR series.

        Named for the test it actually runs (mannwhitneyu) -- this used to be
        called wilcoxon_test, but the Wilcoxon signed-rank test is a paired
        test and doesn't apply to two independent segments the way this is
        used. Called from _open_compare_segments in app.py to test whether
        the two segments' RR-interval distributions differ significantly.

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
