# -*- coding: utf-8 -*-
"""
export_controller.py
---------------------
ExportController -- writing analysis results out to Excel, ZIP bundles,
PDF reports, GraphPad Prism, CSV, and curated PNG figures.
"""
from __future__ import annotations

import io
import logging
import os
import zipfile
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from openpyxl import Workbook
from tkinter import filedialog, messagebox

from ecg.io.export import ExcelExporter, PrismExporter, autonomic_pdf_line
from ecg.core.analysis import (
    analysis_window_rows, analysis_window_span, artifact_correction_text,
    signal_crop_text,
    hrv_freq_is_legacy, hrv_freq_withheld_reason, interval_definitions_text,
    intervals_use_legacy_definitions, mean_hr_bpm,
    rr_exclusion_report, rr_timeseries,
    successive_valid_pairs, wave_template_disclosure, rr_quality_at,
)
from ecg.core.detection import premature_summary_rows
from ecg.core.metric_table import MetricTable, build_metric_table
from ecg.core.filtering import filter_description, recorded_filter_text
from ecg.core.models import (
    MouseECG, QTC_FORMULA_LABELS, event_kind_label, normalize_qtc_formula,
)
from ecg.ui.event_locked_view import event_table_frame, export_event_params
from ecg.ui.plots import (
    ellipsize_fig_text, fig_text_width_in, style_axes, wrap_fig_text,
)
from ecg.ui.theme import (
    PLOT, GREEN, ORANGE, RED, ORANGE_DARK,
    GREEN_MID, AMBER, RED_MID, NAVY, GRAY_LIGHT, CYAN_BRIGHT,
)

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


def pdf_interval_footer(ivl: "Optional[object]") -> str:
    """PDF report footer for the intervals on the page ('' without them):
    the landmark definitions behind PR / QRS / QT and the wave template
    whose search windows produced them (same text as the "Wave template"
    row of the Summary report, Excel Cover sheet and Prism notes)."""
    if ivl is None or getattr(ivl, "empty", True) or "QT_ms" not in ivl.columns:  # type: ignore[attr-defined]
        return ""
    return ("Intervals: " + interval_definitions_text(ivl)  # type: ignore[arg-type]
            + "   |   QT IQR = beat-to-beat, single lead (not QT dispersion)"
            + "   |   Wave template: " + wave_template_disclosure(ivl))  # type: ignore[arg-type]


def pdf_provenance_text(
    results:         "Optional[dict]",
    window:          "Optional[tuple]" = None,
    signal_end_s:    "Optional[float]" = None,
    artifact_report: "Optional[dict]" = None,
    filter_text:     "Optional[str]" = None,
    signal_crop:     "Optional[dict]" = None,
) -> str:
    """PDF report footer line stating the load-time signal crop (when
    recorded -- every time is relative to its start), the analysis window,
    the analysed duration (window end − start, not the time of the last
    beat), the filters and the artifact correction -- same values as the
    Summary report's PROVENANCE block and the Excel Cover sheet."""
    rows: "list[tuple[str, str]]" = []
    crop_text = signal_crop_text(signal_crop)
    if crop_text:
        rows.append(("Signal crop", crop_text))
    rows += analysis_window_rows(results, window, signal_end_s)
    if filter_text:
        rows.append(("Filters", filter_text))
    rows.append(("Artifact correction",
                 artifact_correction_text(artifact_report, results)))
    return "   |   ".join(f"{label}: {value}" for label, value in rows)


def pdf_metric_rows(r: dict) -> "list[tuple[str, str, str]]":
    """The PDF report's "Key metrics" rows (label, value text, status) for
    results *r*, read from the metric table."""
    t = build_metric_table(r, asymmetry=None)
    metrics: "list[tuple[str, str, str]]" = []

    def _row(label: str, v: "Optional[float]", unit: str) -> None:
        if v is None:
            return
        if unit in ("—",):
            val_str = f"{v:.2f}"
        elif unit == "%":
            val_str = f"{v:.1f} %"
        else:
            val_str = f"{v:.1f} {unit}"
        metrics.append((label, val_str, ""))

    fd = r.get("hrv_freq")
    if t.has("hr.mean_rr"):
        _row("Mean HR", t.value("hr.mean_rr"), "bpm")
        _row("Mean RR", t.value("rr.mean_rr"), "ms")
    _row("SDNN",  t.finite("hrv_time.HRV_SDNN"),  "ms")
    _row("RMSSD", t.finite("hrv_time.HRV_RMSSD"), "ms")
    _row("pNN6",  t.finite("hrv_time.HRV_pNN6"),  "%")
    if fd is not None and not fd.empty and hrv_freq_is_legacy(r):
        # An older session's nk.hrv_frequency table: its LFn = LF / TP
        # is not Task Force n.u. -- flagged rather than shown as one.
        metrics.append(("LF / HF", "legacy — re-run Frequency HRV", ""))
    elif fd is not None and not fd.empty:
        # LFn / HFn (Task Force n.u., LF or HF / (LF + HF)) -- HRV_LF /
        # HRV_HF are absolute powers in ms², not fractions.
        lf = t.finite("hrv_freq.HRV_LFn"); hf = t.finite("hrv_freq.HRV_HFn")
        if lf: _row("LF n.u.", lf*100, "%")
        if hf: _row("HF n.u.", hf*100, "%")
    elif hrv_freq_withheld_reason(r):
        # Missing LF/HF rows explained (the reason is in the Summary
        # report and the Excel Cover sheet).
        metrics.append(("LF / HF", "withheld", ""))
    _row("SD1",    t.finite("hrv_nonlin.HRV_SD1"),    "ms")
    _row("SD2",    t.finite("hrv_nonlin.HRV_SD2"),    "ms")
    _row("SampEn", t.finite("hrv_nonlin.HRV_SampEn"), "—")
    for col, lbl in [("PR_ms", "PR"), ("QRS_ms", "QRS"), ("QT_ms", "QT"), ("QTc_ms", "QTc")]:
        row = t.row(f"intervals.{col}")
        if row is not None and row.basis_n:
            _row(lbl, row.value, "ms")
    # Beat-to-beat spread on this single lead -- not classical
    # (inter-lead) QT dispersion; IQR is the robust spread.
    _row("QT IQR (beat-to-beat)", t.finite("intervals.QT_iqr"), "ms")
    return metrics


class ExportController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    def autonomic_export_data(self) -> "tuple":
        """(AutonomicResult, SPWVD dict, event table, event params) of HRV >
        Autonomic for the exporters -- all None without an Autonomic
        result; the event parts None without an event-locked result."""
        a = self.app.analysis
        res = getattr(a, "autonomic_result", None)
        if res is None:
            return None, None, None, None
        el = getattr(a, "event_locked", None)
        if el is None:
            return res, a.autonomic_spwvd, None, None
        return res, a.autonomic_spwvd, event_table_frame(el), export_event_params(el)

    ARRHYTHMIA_STALE_TEXT = ("Abnormal events not exported: the peaks, artifact "
                             "correction or analysis window changed since the "
                             "classification — re-run it on the Abnormal Events tab.")

    def arrhythmia_export_data(self) -> "tuple":
        """(events, premature summary, stale) for the exporters. When the
        stored classification no longer matches the current peaks /
        removed beats / analysis window (AnalysisController.
        arrhythmia_is_current), events and summary are None and stale is
        True: its counts would sit next to results computed on another
        peak set."""
        a = self.app.analysis
        events = list(getattr(a, "arrhythmia_events", None) or [])
        summary = getattr(a, "premature_summary", None)
        if not events and not summary:
            return None, None, False
        actrl = getattr(self.app, "analysis_ctrl", None)
        if actrl is not None and not actrl.arrhythmia_is_current():
            return None, None, True
        return (events or None), summary, False

    def hrv_window(self) -> "Optional[tuple]":
        """The analysis window the core HRV results were computed on (None
        without results) -- the exports compare it with the Autonomic
        result's own span."""
        a = self.app.analysis
        if a.results is None:
            return None
        win = getattr(a, "analysed_window", None)
        return tuple(win) if win is not None else (a.t_start, a.t_end)

    def signal_end_s(self) -> "Optional[float]":
        """Time of the last sample of the loaded signal (s) -- the end of
        an analysis window that runs to the end. None without a signal."""
        t = self.app.signal.time
        return float(t[-1]) if t is not None and len(t) else None

    def signal_crop(self) -> "Optional[dict]":
        """The load-time crop of the loaded signal (SignalState.crop; None
        when not recorded) for the exports' "Signal crop" row."""
        return getattr(self.app.signal, "crop", None)

    def filter_text(self) -> str:
        """The filters the current signal / peaks were computed with
        (recorded at detection), for the exports' provenance rows."""
        sig = self.app.signal
        return recorded_filter_text(getattr(sig, "filter_text", None),
                                    bool(getattr(sig, "no_filter_mode", False)))

    def autonomic_only(self) -> bool:
        """True when only HRV > Autonomic results exist (Analyze not run):
        Export Excel then writes them alone (they need only the peaks)."""
        a = self.app.analysis
        return a.results is None and getattr(a, "autonomic_result", None) is not None

    def _needs_analyze(self, message: str) -> str:
        """*message*, plus where the Autonomic-only results can be exported
        when they exist (Export Excel)."""
        if self.autonomic_only():
            return (message + "\n\nThe HRV > Autonomic results need only the peaks: "
                    "Export Excel saves them alone before Analyze.")
        return message

    def export_figures(self) -> None:
        """Export curated publication-ready PNG figures to a chosen folder.

        Exports 11 figures at 200 DPI.  The RR tachogram is regenerated
        without spike markers so it is clean for publication.
        """
        app = self.app
        if app.analysis.results is None:
            messagebox.showwarning("No results", self._needs_analyze("Click Analyze first."))
            return

        folder = filedialog.askdirectory(title="Choose export folder for figures")
        if not folder:
            return

        import os as _os
        sub = app.ent_subject.get().strip() if app.ent_subject else "subject"
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _os.path.join(folder, f"{sub}_ECG_figures_{ts}")
        _os.makedirs(out_dir, exist_ok=True)

        def _rr_clean(fig: "matplotlib.figure.Figure") -> None:
            """Clean RR + HR tachogram — no spike annotation markers.

            Like the HRV-tab tachogram, every interval is drawn: valid ones
            as the line, intervals excluded from HRV as hollow red markers
            with the line broken around them (mean/SD from valid ones)."""
            r = app.analysis.results
            if r is None:
                return
            rts = rr_timeseries(r)
            if rts.empty or not rts["valid"].any():
                return
            from ecg.core.filtering import downsample_xy_for_display
            t_all  = rts["Time_s"].to_numpy(dtype=float)
            rr_all = rts["RR_ms"].to_numpy(dtype=float)
            hr_all = rts["HR_bpm"].to_numpy(dtype=float)
            valid  = rts["valid"].to_numpy(dtype=bool)
            excl   = ~valid & np.isfinite(rr_all)
            t_ds, rr_ds    = downsample_xy_for_display(t_all, np.where(valid, rr_all, np.nan))
            t_ds_hr, hr_ds = downsample_xy_for_display(t_all, np.where(valid, hr_all, np.nan))
            rr_v = rr_all[valid]
            rr_mean = float(rr_v.mean()); rr_sd = float(rr_v.std())
            hr_mean = mean_hr_bpm(rr_v)          # 60000 / mean NN (app-wide)
            # Same display range as plot_rr: a far-off excluded interval is
            # drawn at the edge instead of squashing the valid trace.
            rr_hi_b = float((r.get("hr") or {}).get("rr_max_ms", MouseECG.RR_MAX_MS))
            rr_ex = np.clip(rr_all[excl], 0.5 * float(rr_v.min()),
                            max(2.0 * float(rr_v.max()), 1.5 * rr_hi_b))
            t_ex  = t_all[excl]
            from matplotlib.gridspec import GridSpec as _GS
            _gs = _GS(2, 1, figure=fig, left=0.08, right=0.98,
                      top=0.93, bottom=0.09, hspace=0.22)
            axes = [fig.add_subplot(_gs[0, 0]), fig.add_subplot(_gs[1, 0])]
            axes[1].sharex(axes[0])
            for ax in axes:
                style_axes(ax)
            axes[0].plot(t_ds, rr_ds, color="#388E3C", lw=0.9, zorder=2)
            axes[0].axhline(rr_mean, color="#388E3C", ls="--", lw=0.9, alpha=0.5)
            axes[0].axhspan(rr_mean - rr_sd, rr_mean + rr_sd,
                            alpha=0.07, color="#388E3C", zorder=0)
            if len(t_ex):
                axes[0].plot(t_ex, rr_ex, "o", ls="none", ms=4.5, mfc="none",
                             mec=RED, mew=1.1, zorder=4,
                             label=f"excluded ({int((~valid).sum())})")
                axes[0].legend(loc="upper right", fontsize=7, framealpha=0,
                               labelcolor=PLOT.get("text", GRAY_LIGHT))
            axes[0].set_ylabel("RR (ms)")
            axes[0].set_title(
                f"RR Intervals  ·  mean {rr_mean:.1f} ms  ·  SD {rr_sd:.1f} ms",
                loc="left", fontsize=9)
            axes[1].plot(t_ds_hr, hr_ds, color=ORANGE_DARK, lw=0.9, zorder=2)
            axes[1].axhline(hr_mean, color=ORANGE_DARK, ls="--", lw=0.9, alpha=0.5)
            if len(t_ex):
                axes[1].plot(t_ex, 60_000.0 / rr_ex, "o", ls="none", ms=4.0,
                             mfc="none", mec=RED, mew=1.0, zorder=4)
            axes[1].set_ylabel("HR (bpm)")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_title(f"Instantaneous HR  ·  mean {hr_mean:.0f} bpm",
                              loc="left", fontsize=9)
            fig.suptitle(sub, fontsize=10, color=PLOT.get("text", "#EEE"))

        catalogue: "list[tuple[str, str, Optional[Callable]]]" = [
            ("rr",            "01_rr_tachogram",    _rr_clean),
            ("rr_hist",       "02_rr_histogram",     None),
            ("psd",           "03_psd",              None),
            ("poincare",      "04_poincare",         None),
            ("beat",          "06_beat_template",    None),
            ("beat_dist",     "07_beat_morphology",  None),
            ("intervals",     "08_intervals",        None),
            ("rolling_hrv",   "09_rolling_hrv",      None),
            ("autonomic",     "10_autonomic_csi_cpi", None),
            ("event_locked",  "11_event_locked",     None),
        ]

        saved, skipped = 0, 0
        DPI = 200; FIG_W, FIG_H = 10.0, 5.0

        for slot_key, stem, custom_fn in catalogue:
            fpath = _os.path.join(out_dir, f"{stem}.png")
            try:
                export_fig = Figure(
                    figsize=(FIG_W, FIG_H), dpi=DPI,
                    facecolor=PLOT.get("bg", "#1A1A2E"), layout="constrained")
                if hasattr(export_fig, "set_constrained_layout_pads"):
                    getattr(export_fig, "set_constrained_layout_pads")(
                        w_pad=0.08, h_pad=0.08, hspace=0.05, wspace=0.05)
                if custom_fn is not None:
                    custom_fn(export_fig)
                else:
                    slot = app._slots.get(slot_key)
                    if slot is None or slot._draw_fn is None:
                        skipped += 1
                        plt.close(export_fig)
                        continue
                    try:
                        slot._draw_fn(export_fig)
                    except Exception as exc:
                        log.warning("Export figure '%s': %s", stem, exc)
                        skipped += 1
                        plt.close(export_fig)
                        continue
                export_fig.savefig(fpath, dpi=DPI,
                                   facecolor=export_fig.get_facecolor(),
                                   bbox_inches="tight")
                plt.close(export_fig)
                saved += 1
            except Exception as exc:
                log.warning("Could not save '%s': %s", stem, exc)
                skipped += 1

        skip_note = f" ({skipped} skipped)" if skipped else ""
        app._set_status(
            f"Figures exported — {saved} PNG files{skip_note} → "
            f"{_os.path.basename(out_dir)}  ✓", GREEN)

    def build_excel_workbook(self) -> "Workbook":
        """Build a formatted openpyxl Workbook from the current results."""
        app = self.app
        if app.analysis.results is None:
            raise RuntimeError("No results available to export.")
        return ExcelExporter.build_workbook(
            results      = app.analysis.results,  # type: ignore[union-attr]
            signal_flt   = app.signal.filtered,
            signal_raw   = app.signal.raw_norm,
            time         = app.signal.time,
            rpeaks_ok    = app.detection.rpeaks_ok,
            fs           = app.signal.fs,
            filepath     = app.signal.filepath,
            subject      = app.ent_subject.get(),
            sig_quality  = app.detection.sig_quality,
            epoch_df     = app.analysis.epoch_df,
            qtc_formula  = app._qtc_formula(),
            analysis_window = self.hrv_window(),
            artifact_report = app.analysis.artifact_report,
            filter_text  = self.filter_text(),
            signal_crop  = self.signal_crop(),
        )

    def metric_table(self) -> MetricTable:
        """The metric table of everything currently computed, with each
        module's state (ecg.ui.freshness). A stale classification is left
        out, as in every export (arrhythmia_export_data)."""
        app = self.app
        a = app.analysis
        res, _sp, _table, ev_params = self.autonomic_export_data()
        events, summary, _stale = self.arrhythmia_export_data()
        fresh = getattr(app, "freshness", None)
        return build_metric_table(
            a.results, states=fresh.states() if fresh is not None else None,
            sig_quality=getattr(getattr(app, "detection", None), "sig_quality", None),
            artifact_report=a.artifact_report,
            window=self.hrv_window(), signal_end_s=self.signal_end_s(),
            epoch_df=a.epoch_df, rolling_df=a.rolling_hrv_df, autonomic=res,
            event_params=ev_params, events=events, premature=summary)

    def write_excel(self, destination) -> None:
        """Write the formatted workbook to *destination* (path or BytesIO).
        With an HRV > Autonomic result, its Autonomic / Autonomic_Series
        (/ Autonomic_LFHF / EventLocked) sheets and Cover disclosure are
        added (the Cover then states the span the indices cover against
        the core results' analysis window). Without core results, an
        Autonomic result is written alone (``autonomic_only``)."""
        app = self.app
        res, spwvd, ev_table, ev_params = self.autonomic_export_data()
        if self.autonomic_only():
            wb = ExcelExporter.build_autonomic_workbook(
                app.ent_subject.get(), app.signal.filepath, app.signal.fs, app.signal.time)
        else:
            wb = self.build_excel_workbook()
        ExcelExporter.add_annotations_sheet(wb, app.analysis.annotations)
        arr_events, arr_summary, arr_stale = self.arrhythmia_export_data()
        ExcelExporter.add_abnormal_events_sheets(
            wb, arr_events, arr_summary,
            stale_note=self.ARRHYTHMIA_STALE_TEXT if arr_stale else None)
        if res is not None:
            ExcelExporter.add_autonomic_sheets(wb, res, spwvd, ev_table, ev_params,
                                               hrv_window=self.hrv_window())
        try:
            ExcelExporter.add_metrics_sheet(wb, self.metric_table())
        except Exception:
            # Additive: never lose the rest of the workbook over it.
            log.exception("Metrics sheet not written")
        wb.save(destination)

    def export_excel(self) -> None:
        app = self.app
        only = self.autonomic_only()
        if not app.analysis.results and not only:
            messagebox.showwarning("No results", "Run analysis first.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        kind = "ECG_autonomic" if only else "ECG"
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=f"{app.ent_subject.get()}_{kind}_{ts}.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            self.write_excel(path)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        what = "Excel (Autonomic only — Analyze not run)" if only else "Excel"
        stale = self.arrhythmia_export_data()[2]
        app._set_status(f"{what} saved — {os.path.basename(path)}  ✓"
                        + (f"  ·  {self.ARRHYTHMIA_STALE_TEXT}" if stale else ""),
                        ORANGE if (only or stale) else GREEN)

    def export_zip(self) -> None:
        app = self.app
        if not app.analysis.results:
            messagebox.showwarning("No results", self._needs_analyze("Run analysis first."))
            return
        sub  = app.ent_subject.get()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            initialfile=f"{sub}_ECG_full_{ts}.zip",
            filetypes=[("ZIP", "*.zip")],
        )
        if not path:
            return

        figure_keys = [
            ("detail",    "00_detail"),
            ("rr",        "01_rr_hr"),
            ("rr_hist",   "02_rr_hist"),
            ("psd",       "03_psd"),
            ("poincare",  "05_poincare"),
            ("intervals", "06_intervals"),
            ("beat",      "07_beat"),
        ]
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                xl_buf = io.BytesIO()
                self.write_excel(xl_buf)
                xl_buf.seek(0)
                zf.writestr(f"{sub}_ECG_{ts}.xlsx", xl_buf.read())

                # The Autonomic figures only once computed (their placeholder
                # would otherwise be saved).
                for slot_key, filename in (("autonomic", "08_autonomic_csi_cpi"),
                                           ("event_locked", "09_event_locked")):
                    slot = app._slots.get(slot_key)
                    if slot is not None and getattr(slot, "_draw_fn", None) is not None:
                        figure_keys.append((slot_key, filename))
                for slot_key, filename in figure_keys:
                    slot = app._slots.get(slot_key)
                    if slot and slot.fig:
                        try:
                            buf = io.BytesIO()
                            slot.fig.savefig(buf, format="png", dpi=300,
                                             facecolor=PLOT["bg"])
                            buf.seek(0)
                            zf.writestr(f"figures/{filename}.png", buf.read())
                        except Exception as exc:
                            log.warning("Figure '%s' not saved to ZIP: %s", filename, exc)
        except Exception as exc:
            messagebox.showerror("ZIP export failed", str(exc))
            return

        app._set_status(f"ZIP saved — {os.path.basename(path)}  ✓", GREEN)

    def export_pdf_report(self) -> None:
        """Generate a one-page PDF summary: ECG strip + KPI table + interpretation."""
        app = self.app
        if app.analysis.results is None:
            messagebox.showwarning("No results", self._needs_analyze("Click Analyze first."))
            return
        try:
            import matplotlib.backends.backend_pdf as _pdf_backend
            import matplotlib.figure as _mpl_figure
            import matplotlib.lines as _mpl_lines
            import matplotlib.gridspec as gridspec
        except ImportError:
            messagebox.showerror("PDF", "matplotlib not available.")
            return

        sub  = app.ent_subject.get().strip() or "subject"
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=f"{sub}_ECG_report_{ts}.pdf",
            filetypes=[("PDF", "*.pdf")],
        )
        if not path:
            return

        r   = app.analysis.results

        # ── Collect metrics ───────────────────────────────────────────────
        ivl = r.get("intervals")
        metrics = pdf_metric_rows(r)   # (label, value, status)

        # ── Build figure ─────────────────────────────────────────────────
        bg   = PLOT.get("bg",   NAVY)
        fg   = PLOT.get("text", GRAY_LIGHT)
        mut  = PLOT.get("muted","#6B7280")

        # HRV > Autonomic computed: its method / window / ks / kp and the
        # rat-calibration caveat go in the header (one more line).
        auto_res, _sp, _evt, auto_ev = self.autonomic_export_data()
        auto_line = (autonomic_pdf_line(dict(auto_res.params),
                                        (auto_ev or {}).get("n_events"))
                     if auto_res is not None else "")
        sep_y = 0.881 if auto_line else 0.897

        fig  = _mpl_figure.Figure(figsize=(11.7, 8.3), facecolor=bg)   # A4 landscape
        # Footer (interval definitions + wave template), wrapped to the page
        # width; the panels move up by one line height per extra line.
        # Footer: provenance (window, analysed duration, filters, artifact
        # correction), then the interval definitions + wave template.
        footer = "   |   ".join(t for t in (
            pdf_provenance_text(r, self.hrv_window(), self.signal_end_s(),
                                app.analysis.artifact_report, self.filter_text(),
                                self.signal_crop()),
            pdf_interval_footer(ivl)) if t)
        footer = (wrap_fig_text(footer, (0.96 - 0.06) * fig.get_figwidth(), 6.5)
                  if footer else "")
        n_foot = footer.count("\n") + 1 if footer else 0
        gs   = gridspec.GridSpec(
            3, 2, figure=fig,
            left=0.06, right=0.96, top=0.855 if auto_line else 0.87,
            bottom=0.05 + 0.0125 * max(0, n_foot - 2),
            hspace=0.55, wspace=0.30,
            height_ratios=[2.0, 1.4, 1.3],
        )

        # Title block
        qtc_label = QTC_FORMULA_LABELS[normalize_qtc_formula(app._qtc_formula())]
        fig.text(0.06, 0.945, f"ECG Report — {sub}",
                 fontsize=15, color=fg, fontweight="bold", va="bottom")
        # One line from 0.06 to 0.96 of the page: the file path is shortened
        # in the middle to the room the date and the QTc formula leave (a
        # long path pushed the QTc text past the page edge).
        head_rest = (f"   |   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   "
                     f"QTc: {qtc_label}, preceding RR")
        head_room = ((0.96 - 0.06) * fig.get_figwidth()
                     - fig_text_width_in("File: " + head_rest, 8))
        fig.text(0.06, 0.918,
                 "File: " + ellipsize_fig_text(str(app.signal.filepath or "—"),
                                               max(0.5, head_room), 8) + head_rest,
                 fontsize=8, color=mut, va="bottom")
        # RR intervals behind every HRV value on this page, and the ones
        # excluded (not bridged) -- same counts as the Excel Cover sheet.
        rep = rr_exclusion_report(r)
        if rep["n_intervals"] is not None:
            rr_note = (f"RR: {rep['n_intervals']} intervals, "
                       f"{rep['n_excluded']} excluded from HRV "
                       f"({float(rep['excluded_pct'] or 0.0):.1f} %")
            if rep["n_excluded"]:
                rr_note += (f": {rep['n_too_short']} too short, "
                            f"{rep['n_too_long']} too long, "
                            f"{rep['n_spanning_removed']} spanning removed beats")
            rr_note += f")   |   RR bounds: {rep['rr_min_ms']:.0f}–{rep['rr_max_ms']:.0f} ms"
            fig.text(0.06, 0.902, rr_note, fontsize=8,
                     color=ORANGE if rep["n_excluded"] else mut, va="bottom")
        if auto_line:
            fig.text(0.06, 0.886, auto_line, fontsize=8, color=ORANGE, va="bottom")
        fig.add_artist(_mpl_lines.Line2D([0.06, 0.94], [sep_y, sep_y],
                                  transform=fig.transFigure,
                                  color=mut, lw=0.5))
        # Footer: the landmarks behind PR / QRS / QT on this page and the
        # wave template whose windows found them.
        if footer:
            legacy = intervals_use_legacy_definitions(ivl)
            fig.text(0.06, 0.006, footer,
                     fontsize=6.5, color=ORANGE if legacy else mut, va="bottom",
                     linespacing=1.1)

        # Panel A — ECG strip (top-left, spans 2 columns)
        ax_ecg = fig.add_subplot(gs[0, :])
        ax_ecg.set_facecolor(bg)
        for sp in ax_ecg.spines.values():
            sp.set_color(mut); sp.set_linewidth(0.5)
        ax_ecg.tick_params(colors=mut, labelsize=7)
        ax_ecg.set_xlabel("Time (s)", fontsize=8, color=mut)
        # Filters = none: the strip is the raw (z-scored) signal, not filtered.
        raw_strip = self.filter_text() == filter_description({"no_filter": True})
        ax_ecg.set_title(f"{'Raw ECG' if raw_strip else 'Filtered ECG'} — 10 s excerpt "
                         "(middle of the analysis window)",
                         loc="left", fontsize=9, color=fg)

        if app.signal.filtered is not None and app.signal.time is not None:
            fs   = app.signal.fs
            dur  = float(app.signal.time[-1])
            # Centred in the analysis window the results come from.
            w0, w1 = analysis_window_span(self.hrv_window(), dur)
            w1 = min(dur, w1 if w1 is not None else dur)
            t0   = max(w0, (w0 + w1) / 2 - 5)
            t1   = min(w1, t0 + 10)
            i0, i1 = int(t0 * fs), int(t1 * fs)
            t_seg   = app.signal.time[i0:i1]
            sig_seg = app.signal.filtered[i0:i1]
            ax_ecg.plot(t_seg, sig_seg, color=PLOT.get("signal", CYAN_BRIGHT), lw=0.7)
            # Overlay R-peaks in the window
            if app.detection.rpeaks_ok is not None:
                rp = app.detection.rpeaks_ok
                mask = (rp >= i0) & (rp < i1)
                rp_w = rp[mask]
                if len(rp_w):
                    ax_ecg.scatter(app.signal.time[rp_w], app.signal.filtered[rp_w],
                                   c="red", s=12, zorder=5)
            ax_ecg.set_xlim(t0, t1)
        else:
            ax_ecg.text(0.5, 0.5, "Signal not available",
                        ha="center", va="center", color=mut, fontsize=10,
                        transform=ax_ecg.transAxes)

        # Panel B — Metric table (middle-left)
        ax_tbl = fig.add_subplot(gs[1, 0])
        ax_tbl.set_facecolor(bg)
        ax_tbl.axis("off")
        ax_tbl.set_title("Key metrics", loc="left", fontsize=9, color=fg, pad=4)
        status_colors = {"✓": GREEN_MID, "~": AMBER, "↑": RED_MID, "↓": RED_MID}
        # Row height fitted to the row count: a fixed 0.10 left no room for
        # the interval rows (PR, QRS, QT, QTc), which come after the HRV ones.
        y = 0.95
        dy = min(0.10, 0.95 / (len(metrics) + 1))
        fs_row = 8 if dy >= 0.07 else 7
        for lbl_h, val_h, st_h in [("Parameter", "Value", "")]:
            ax_tbl.text(0.02,  y, lbl_h, fontsize=7, color=mut,
                        transform=ax_tbl.transAxes, va="top")
            ax_tbl.text(0.57,  y, val_h, fontsize=7, color=mut,
                        transform=ax_tbl.transAxes, va="top")
            y -= dy
        for lbl, val, status in metrics:
            sc = status_colors.get(status, fg)
            ax_tbl.text(0.02, y, lbl, fontsize=fs_row, color=fg,
                        transform=ax_tbl.transAxes, va="top")
            ax_tbl.text(0.57, y, val, fontsize=fs_row, color=sc,
                        transform=ax_tbl.transAxes, va="top", fontweight="bold")
            ax_tbl.text(0.90, y, status, fontsize=fs_row, color=sc,
                        transform=ax_tbl.transAxes, va="top")
            y -= dy

        # Panel C — Poincaré (middle-right) — copy from slot if available
        ax_rr = fig.add_subplot(gs[1, 1])
        ax_rr.set_facecolor(bg)
        for sp in ax_rr.spines.values():
            sp.set_color(mut); sp.set_linewidth(0.5)
        ax_rr.tick_params(colors=mut, labelsize=7)
        ax_rr.set_title("Poincaré diagram", loc="left", fontsize=9, color=fg)
        ax_rr.set_xlabel("RR_n (ms)", fontsize=7, color=mut)
        ax_rr.set_ylabel("RR_n+1 (ms)", fontsize=7, color=mut)
        rts = rr_timeseries(r)
        if not rts.empty:
            # Successive VALID pairs only (same pairs as SD1/SD2): rr_df
            # holds only the valid intervals, so pairing its consecutive
            # rows would join the two sides of an excluded pause.
            rr_a, rr_b = successive_valid_pairs(rts["RR_ms"].to_numpy(dtype=float),
                                                rts["valid"].to_numpy(dtype=bool))
            if len(rr_a) > 1:
                ax_rr.scatter(rr_a, rr_b,
                              alpha=0.35, s=4, c=PLOT.get("signal", CYAN_BRIGHT))
                lo = min(rr_a.min(), rr_b.min()) * 0.97
                hi = max(rr_a.max(), rr_b.max()) * 1.03
                ax_rr.plot([lo, hi], [lo, hi], "--", color=mut, lw=0.6)
        else:
            ax_rr.text(0.5, 0.5, "RR data not available",
                       ha="center", va="center", color=mut, fontsize=9,
                       transform=ax_rr.transAxes)

        # Panel E (row 2, full width) -- Abnormal-events summary. Used to
        # share this row with an HRV radar chart (left) that normalised
        # each metric against its experimental-context reference range;
        # with that removed, this panel now spans the full row instead of
        # leaving the left half blank.
        ax_arr = fig.add_subplot(gs[2, :])
        ax_arr.set_facecolor(bg)
        ax_arr.axis("off")
        ax_arr.set_title("Abnormal events", loc="left", fontsize=9, color=fg, pad=4)
        events, arr_summary, arr_stale = self.arrhythmia_export_data()
        events = events or []
        if arr_stale:
            ax_arr.text(0.0, 0.85, wrap_fig_text(self.ARRHYTHMIA_STALE_TEXT, 10.0, 7.5),
                        fontsize=7.5, color=mut, transform=ax_arr.transAxes, va="top")
        elif not events:
            ax_arr.text(0.0, 0.85, "None classified in this session\n"
                                    "(run Abnormal Events tab to populate)",
                        fontsize=7.5, color=mut, transform=ax_arr.transAxes, va="top")
        else:
            by_kind: "dict[str, int]" = {}
            for ev in events:
                by_kind[ev.kind] = by_kind.get(ev.kind, 0) + 1
            y = 0.92
            ax_arr.text(0.0, y, f"{len(events)} event(s) total",
                        fontsize=8, color=fg, fontweight="bold",
                        transform=ax_arr.transAxes, va="top")
            y -= 0.16
            for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
                ax_arr.text(0.0, y, f"  {event_kind_label(kind)}: {count}",
                            fontsize=7.5, color=mut, transform=ax_arr.transAxes, va="top")
                y -= 0.13
                if y < 0:
                    break
            # Right half: premature-beat counts, per-hour rates, definition.
            y = 0.92
            for label, value in premature_summary_rows(arr_summary):
                ax_arr.text(0.45, y, wrap_fig_text(f"{label.strip()}: {value}", 6.0, 7),
                            fontsize=7, color=fg if not label.startswith(" ") else mut,
                            transform=ax_arr.transAxes, va="top")
                y -= 0.13
                if y < 0:
                    break

        # ── Save PDF ─────────────────────────────────────────────────────
        try:
            with _pdf_backend.PdfPages(path) as pp:
                pp.savefig(fig, dpi=200, facecolor=bg)
            plt.close(fig)
        except Exception as exc:
            messagebox.showerror("PDF export failed", str(exc))
            return

        app._set_status(f"PDF report saved — {os.path.basename(path)}", GREEN)

    def export_prism(self) -> None:
        """Export all analysis results to a GraphPad Prism .pzfx file.

        Tables exported
        ───────────────
        (PrismExporter.build_and_write; each only when its data exists)
        XY       RR & HR tachogram, RR HRV validity, ECG intervals timeline,
                 Rolling HRV timeline + window counts, Epoch HRV + counts,
                 Autonomic CSI / CPI, D / SD1 / SD2 and SPWVD LF / HF,
                 Event-locked pre / post (valid events)
        Column   HRV time-domain, frequency-domain and non-linear summaries
                 (the non-linear one includes NeuroKit's HRV_PI / HRV_GI
                 asymmetry indices -- there is no separate asymmetry table)
        OneWay   ECG intervals per beat
        Notes    Project info: provenance, premature-beat and autonomic rows
        """
        app = self.app
        if app.analysis.results is None:
            messagebox.showwarning(
                "No results",
                self._needs_analyze("Click Analyze before exporting to Prism."))
            return

        sub  = app.ent_subject.get().strip() or "subject"
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".pzfx",
            initialfile=f"{sub}_ECG_{ts}.pzfx",
            filetypes=[("GraphPad Prism", "*.pzfx"), ("All files", "*.*")],
            title="Export to GraphPad Prism",
        )
        if not path:
            return

        arr_events, arr_summary, _arr_stale = self.arrhythmia_export_data()
        arr_events      = arr_events or []
        seg_a           = app.analysis.last_seg_a
        seg_b           = app.analysis.last_seg_b
        notes           = app.session.recording_notes
        fs              = float(app.signal.fs) if app.signal.fs else 1000.0
        auto_res, auto_sp, auto_ev, auto_evp = self.autonomic_export_data()

        try:
            n_tables = PrismExporter.build_and_write(
                path              = path,
                results           = app.analysis.results,       # type: ignore[arg-type]
                rolling_hrv_df    = app.analysis.rolling_hrv_df,
                epoch_df          = app.analysis.epoch_df,
                subject           = sub,
                arrhythmia_events = arr_events if arr_events else None,
                segment_a         = seg_a,
                segment_b         = seg_b,
                fs                = fs,
                recording_notes   = notes,
                qtc_formula       = app._qtc_formula(),
                autonomic         = auto_res,
                autonomic_spwvd   = auto_sp,
                event_table       = auto_ev,
                event_params      = auto_evp,
                hrv_window        = self.hrv_window(),
                premature_summary = arr_summary,
                signal_end_s      = self.signal_end_s(),
                artifact_report   = app.analysis.artifact_report,
                filter_text       = self.filter_text(),
                signal_crop       = self.signal_crop(),
            )
        except Exception as exc:
            log.warning("Prism export failed: %s", exc)
            messagebox.showerror("Prism export failed", str(exc))
            return

        app._set_status(
            f"Prism: {n_tables} tables → {os.path.basename(path)}", GREEN)

    def copy_summary(self) -> None:
        app = self.app
        if not app.analysis.results:
            messagebox.showwarning("No results", "Run analysis first.")
            return
        app.clipboard_clear()
        app.clipboard_append(app.txt_sum.get("1.0", "end"))
        app._set_status("Summary copied to clipboard", GREEN)

    def save_summary_txt(self) -> None:
        app = self.app
        if not app.analysis.results:
            messagebox.showwarning("No results", "Run analysis first.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"{app.ent_subject.get()}_ECG_summary_{ts}.txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(app.txt_sum.get("1.0", "end"))
        app._set_status(f"Summary saved: {os.path.basename(path)}", GREEN)

    def export_rr_csv(self) -> None:
        """Export RR intervals to a lightweight CSV (no Excel dependency)."""
        app = self.app
        if app.analysis.results is None:
            messagebox.showwarning("No results", self._needs_analyze("Click Analyze first."))
            return
        # Every interval, with a ``valid`` column (False = excluded from HRV:
        # outside the RR bounds or spanning a removed beat) rather than
        # rr_df's valid-only rows, so a pause is flagged, not dropped.
        rdf = rr_timeseries(app.analysis.results)
        if rdf.empty:
            rdf = app.analysis.results.get("rr_df")
        if rdf is None or rdf.empty:
            messagebox.showwarning("No data", "RR DataFrame is empty.")
            return
        sub = app.ent_subject.get().strip() if app.ent_subject else "subject"
        default = f"{sub}_RR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile=default,
            title="Export RR intervals as CSV",
        )
        if not path:
            return
        try:
            # Include quality flag if beat_corr is available
            out = rdf[[c for c in ("Time_s", "RR_ms", "HR_bpm", "valid")
                       if c in rdf.columns]].copy()
            # Quality of the beat STARTING each interval (Time_s - RR_ms),
            # matched by time -- edge beats are not scored, so beat_corr is
            # not row-aligned -- the same beat as the Poor-shape-match tile.
            # Left out when the results do not tell which beats were scored.
            res = app.analysis.results or {}
            if "Time_s" in out.columns and "RR_ms" in out.columns:
                q = rr_quality_at(res, out["Time_s"].to_numpy(dtype=float),
                                  out["RR_ms"].to_numpy(dtype=float))
                if np.isfinite(q).any():
                    out["quality"] = np.where(np.isfinite(q),
                                              np.where(q >= 0.90, "ok", "low"), "")
            out.to_csv(path, index=False, float_format="%.4f")
            app._set_status(f"RR CSV saved → {os.path.basename(path)}", GREEN)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_arrhythmia_pdf(self) -> None:
        """Export a PDF with one annotated ECG strip per abnormal-event episode."""
        app = self.app
        if not app.analysis.arrhythmia_events:
            messagebox.showwarning("No abnormal events", "Run abnormal-event classification first.")
            return
        if self.arrhythmia_export_data()[2]:
            messagebox.showwarning("Abnormal events out of date", self.ARRHYTHMIA_STALE_TEXT)
            return
        if app.signal.filtered is None or app.signal.time is None or app.signal.fs is None:
            messagebox.showwarning("No signal", "No ECG signal loaded.")
            return
        sub  = app.ent_subject.get().strip() if app.ent_subject else "subject"
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            initialfile=f"{sub}_abnormal_events.pdf",
            title="Export abnormal-event strips as PDF",
        )
        if not path:
            return

        import matplotlib.pyplot as _plt
        from matplotlib.backends.backend_pdf import PdfPages
        sig  = app.signal.filtered
        time = app.signal.time
        fs   = app.signal.fs
        rp   = app.detection.rpeaks_ok
        evts = app.analysis.arrhythmia_events
        WIN  = 6.0   # seconds per strip

        app._set_status("Building abnormal-event PDF…", ORANGE)
        try:
            with PdfPages(path) as pdf:
                # First page: premature-beat counts / rates and definition.
                rows = premature_summary_rows(getattr(app.analysis, "premature_summary", None))
                if rows:
                    fig0 = _plt.figure(figsize=(10, 2.8), facecolor=PLOT.get("bg", "#1A1A2E"))
                    yy = 0.92
                    for label, value in rows:
                        fig0.text(0.03, yy, wrap_fig_text(f"{label.strip()}: {value}", 9.4, 8.5),
                                  fontsize=8.5, color=PLOT.get("text", "#EEE"), va="top")
                        yy -= 0.11
                    pdf.savefig(fig0, facecolor=fig0.get_facecolor())
                    _plt.close(fig0)
                for ev in evts:
                    t_c  = (ev.t_start + ev.t_end) / 2.0
                    t_lo = max(0.0,   t_c - WIN / 2)
                    t_hi = min(time[-1], t_lo + WIN)
                    mask = (time >= t_lo) & (time <= t_hi)
                    t_s  = time[mask]
                    s_s  = sig[mask]

                    fig, ax = _plt.subplots(figsize=(10, 2.8),
                                            facecolor=PLOT.get("bg", "#1A1A2E"))
                    style_axes(ax)
                    ax.plot(t_s, s_s, color=PLOT.get("signal", "#42A5F5"),
                            lw=0.7, zorder=2)
                    # Mark episode span
                    ax.axvspan(ev.t_start, ev.t_end,
                               alpha=0.18, color=RED, zorder=0)
                    ax.axvline(ev.t_start, color=RED, lw=1.2, ls="--", alpha=0.7)
                    ax.axvline(ev.t_end,   color=RED, lw=1.2, ls="--", alpha=0.7)
                    # Mark R-peaks in window
                    if rp is not None:
                        rp_m = rp[(rp / fs >= t_lo) & (rp / fs <= t_hi)]
                        ax.scatter(rp_m / fs, sig[rp_m], s=30, color=ORANGE,
                                   zorder=5, linewidths=0)
                    hr_str = f"{ev.hr_mean:.0f} bpm"
                    d_str = (f"Δ={ev.delta_pct:+.1f}%  "
                             if ev.kind in ("bradycardia", "tachycardia") else "")
                    ax.set_title(
                        f"{event_kind_label(ev.kind)}  t={ev.t_start:.1f}–{ev.t_end:.1f} s  "
                        f"{d_str}{hr_str}  —  {ev.label}",
                        loc="left", fontsize=9,
                        color=PLOT.get("text", "#EEE"))
                    ax.set_xlabel("Time (s)", fontsize=8)
                    ax.set_ylabel("Amplitude", fontsize=8)
                    fig.tight_layout(pad=0.4)
                    pdf.savefig(fig, facecolor=fig.get_facecolor())
                    _plt.close(fig)
            app._set_status(
                f"Abnormal-event PDF saved → {os.path.basename(path)}", GREEN)
        except Exception as exc:
            messagebox.showerror("PDF export failed", str(exc))
