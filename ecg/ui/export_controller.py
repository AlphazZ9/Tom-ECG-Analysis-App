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
    hrv_freq_withheld_reason, rr_exclusion_report, rr_timeseries,
    successive_valid_pairs,
)
from ecg.core.models import MouseECG, QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.ui.event_locked_view import event_table_frame, export_event_params
from ecg.ui.plots import style_axes
from ecg.ui.theme import (
    PLOT, GREEN, ORANGE, RED, ORANGE_DARK,
    GREEN_MID, AMBER, RED_MID, NAVY, GRAY_LIGHT, CYAN_BRIGHT,
)

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


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

    def hrv_window(self) -> "Optional[tuple]":
        """The analysis window the core HRV results were computed on (None
        without results) -- the exports compare it with the Autonomic
        result's own span."""
        a = self.app.analysis
        if a.results is None:
            return None
        win = getattr(a, "analysed_window", None)
        return tuple(win) if win is not None else (a.t_start, a.t_end)

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
            hr_mean = float(hr_all[valid].mean())
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
        )

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
        if res is not None:
            ExcelExporter.add_autonomic_sheets(wb, res, spwvd, ev_table, ev_params,
                                               hrv_window=self.hrv_window())
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
        app._set_status(f"{what} saved — {os.path.basename(path)}  ✓",
                        ORANGE if only else GREEN)

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
        def _safe(df, col):
            try:
                v = float(df[col].values[0])
                return v if np.isfinite(v) else None
            except Exception:
                return None

        rdf = r.get("rr_df")
        td  = r.get("hrv_time")
        fd  = r.get("hrv_freq")
        nl  = r.get("hrv_nonlin")
        ivl = r.get("intervals")

        metrics: "list[tuple[str,str,str]]" = []   # (label, value, status)

        def _row(label, v, unit, key):
            # `key` used to look up an experimental-context reference range
            # to grade a ✓/~/↑/↓ status; kept as a no-op parameter (every
            # call site still passes one) now that the context system --
            # and with it the "normal range" this status was graded
            # against -- has been removed.
            if v is None:
                return
            status = ""
            if unit in ("—",):
                val_str = f"{v:.2f}"
            elif unit == "%":
                val_str = f"{v:.1f} %"
            else:
                val_str = f"{v:.1f} {unit}"
            metrics.append((label, val_str, status))

        if rdf is not None and not rdf.empty:
            _row("HR moyen",   float(rdf["HR_bpm"].mean()),  "bpm", "HR_mean")
            _row("RR moyen",   float(rdf["RR_ms"].mean()),   "ms",  "RR_mean")
        if td is not None and not td.empty:
            _row("SDNN",  _safe(td, "HRV_SDNN"),  "ms",  "RR_SDNN")
            _row("RMSSD", _safe(td, "HRV_RMSSD"), "ms",  "RR_RMSSD")
            _row("pNN6",  _safe(td, "HRV_pNN6"),  "%",   "RR_pNN6")
        if fd is not None and not fd.empty:
            lf = _safe(fd, "HRV_LF"); hf = _safe(fd, "HRV_HF")
            if lf: _row("LF%",    lf*100, "%", "LF_pct")
            if hf: _row("HF%",    hf*100, "%", "HF_pct")
        elif hrv_freq_withheld_reason(r):
            # Missing LF/HF rows explained (the reason is in the Summary
            # report and the Excel Cover sheet).
            metrics.append(("LF / HF", "withheld", ""))
        if nl is not None and not nl.empty:
            _row("SD1",    _safe(nl,"HRV_SD1"),       "ms", "SD1")
            _row("SD2",    _safe(nl,"HRV_SD2"),       "ms", "SD2")
            _row("SampEn", _safe(nl,"HRV_SampEn"),    "—",  "SampEn")
        if ivl is not None and not ivl.empty:
            for col, key, lbl in [("PR_ms","PR_ms","PR"),("QRS_ms","QRS_ms","QRS"),
                                   ("QT_ms","QT_ms","QT"),("QTc_ms","QTc_ms","QTc")]:
                if col in ivl.columns:
                    d = ivl[col].dropna()
                    if len(d):
                        _row(lbl, float(d.median()), "ms", key)

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
        gs   = gridspec.GridSpec(
            3, 2, figure=fig,
            left=0.06, right=0.96, top=0.855 if auto_line else 0.87, bottom=0.05,
            hspace=0.55, wspace=0.30,
            height_ratios=[2.0, 1.4, 1.3],
        )

        # Title block
        qtc_label = QTC_FORMULA_LABELS[normalize_qtc_formula(app._qtc_formula())]
        fig.text(0.06, 0.945, f"ECG Report — {sub}",
                 fontsize=15, color=fg, fontweight="bold", va="bottom")
        fig.text(0.06, 0.918,
                 f"Fichier : {app.signal.filepath or '—'}   |   "
                 f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}   |   "
                 f"QTc : {qtc_label}",
                 fontsize=8, color=mut, va="bottom")
        # RR intervals behind every HRV value on this page, and the ones
        # excluded (not bridged) -- same counts as the Excel Cover sheet.
        rep = rr_exclusion_report(r)
        if rep["n_intervals"] is not None:
            rr_note = (f"RR : {rep['n_intervals']} intervals, "
                       f"{rep['n_excluded']} excluded from HRV "
                       f"({float(rep['excluded_pct'] or 0.0):.1f} %")
            if rep["n_excluded"]:
                rr_note += (f": {rep['n_too_short']} too short, "
                            f"{rep['n_too_long']} too long, "
                            f"{rep['n_spanning_removed']} spanning removed beats")
            rr_note += f")   |   RR bounds : {rep['rr_min_ms']:.0f}–{rep['rr_max_ms']:.0f} ms"
            fig.text(0.06, 0.902, rr_note, fontsize=8,
                     color=ORANGE if rep["n_excluded"] else mut, va="bottom")
        if auto_line:
            fig.text(0.06, 0.886, auto_line, fontsize=8, color=ORANGE, va="bottom")
        fig.add_artist(_mpl_lines.Line2D([0.06, 0.94], [sep_y, sep_y],
                                  transform=fig.transFigure,
                                  color=mut, lw=0.5))

        # Panel A — ECG strip (top-left, spans 2 columns)
        ax_ecg = fig.add_subplot(gs[0, :])
        ax_ecg.set_facecolor(bg)
        for sp in ax_ecg.spines.values():
            sp.set_color(mut); sp.set_linewidth(0.5)
        ax_ecg.tick_params(colors=mut, labelsize=7)
        ax_ecg.set_xlabel("Time (s)", fontsize=8, color=mut)
        ax_ecg.set_title("Signal ECG filtré — extrait 10 s", loc="left",
                         fontsize=9, color=fg)

        if app.signal.filtered is not None and app.signal.time is not None:
            fs   = app.signal.fs
            dur  = float(app.signal.time[-1])
            t0   = max(0.0, dur / 2 - 5)
            t1   = min(dur, t0 + 10)
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
            ax_ecg.text(0.5, 0.5, "Signal non disponible",
                        ha="center", va="center", color=mut, fontsize=10,
                        transform=ax_ecg.transAxes)

        # Panel B — Metric table (middle-left)
        ax_tbl = fig.add_subplot(gs[1, 0])
        ax_tbl.set_facecolor(bg)
        ax_tbl.axis("off")
        ax_tbl.set_title("Métriques clés", loc="left", fontsize=9, color=fg, pad=4)
        status_colors = {"✓": GREEN_MID, "~": AMBER, "↑": RED_MID, "↓": RED_MID}
        col_w = [0.55, 0.30, 0.15]
        y = 0.95
        for lbl_h, val_h, st_h in [("Paramètre","Valeur","")]:
            ax_tbl.text(0.02,  y, lbl_h, fontsize=7, color=mut,
                        transform=ax_tbl.transAxes, va="top")
            ax_tbl.text(0.57,  y, val_h, fontsize=7, color=mut,
                        transform=ax_tbl.transAxes, va="top")
            y -= 0.10
        for lbl, val, status in metrics:
            sc = status_colors.get(status, fg)
            ax_tbl.text(0.02, y, lbl, fontsize=8, color=fg,
                        transform=ax_tbl.transAxes, va="top")
            ax_tbl.text(0.57, y, val, fontsize=8, color=sc,
                        transform=ax_tbl.transAxes, va="top", fontweight="bold")
            ax_tbl.text(0.90, y, status, fontsize=8, color=sc,
                        transform=ax_tbl.transAxes, va="top")
            y -= 0.10
            if y < 0:
                break

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
            ax_rr.text(0.5, 0.5, "Données RR non disponibles",
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
        events = getattr(app.analysis, "arrhythmia_events", None) or []
        if not events:
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
                ax_arr.text(0.0, y, f"  {kind.replace('_', ' ').title()}: {count}",
                            fontsize=7.5, color=mut, transform=ax_arr.transAxes, va="top")
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
        XY       RR & HR tachogram, ECG interval timelines, Rolling HRV,
                 Epoch HRV, Poincaré scatter, Beat correlation timeline,
                 QTc variability timeline
        Column   HRV time-domain, frequency-domain, non-linear summaries,
                 HR distribution, RR distribution, Beat morphology quality,
                 RR asymmetry / Porta breakdown, Segment comparison (A vs B),
                 Metadata
        OneWay   ECG intervals per beat, Abnormal-event episode table
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

        beat_corr       = app.analysis.results.get("beat_corr")
        beat_peak_times = app.analysis.results.get("beat_peak_times")
        arr_events      = app.analysis.arrhythmia_events or []
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
                beat_corr         = beat_corr,
                beat_peak_times   = beat_peak_times,
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
            beat_corr = (app.analysis.results or {}).get("beat_corr")
            if beat_corr is not None and len(beat_corr) == len(out):
                out["quality"] = np.where(beat_corr >= 0.90, "ok", "low")
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
                    hr_str = f"{ev.hr_bpm:.0f} bpm" if hasattr(ev, "hr_bpm") else ""
                    ax.set_title(
                        f"{ev.kind.upper()}  t={ev.t_start:.1f}–{ev.t_end:.1f} s  "
                        f"Δ={ev.delta_pct:+.1f}%  {hr_str}",
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
