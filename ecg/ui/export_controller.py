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

from ecg.core.models import EXPERIMENTAL_CONTEXTS
from ecg.io.export import ExcelExporter, PrismExporter
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

    def export_figures(self) -> None:
        """Export curated publication-ready PNG figures to a chosen folder.

        Exports 11 figures at 200 DPI.  The RR tachogram is regenerated
        without spike markers so it is clean for publication.
        """
        app = self.app
        if app.analysis.results is None:
            messagebox.showwarning("No results", "Run Core Analysis first.")
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
            """Clean RR + HR tachogram — no spike annotation markers."""
            r = app.analysis.results
            if r is None:
                return
            rdf = r.get("rr_df")
            if rdf is None or rdf.empty:
                return
            from ecg.core.filtering import downsample_for_display
            t_all  = rdf["Time_s"].values.astype(float)
            rr_all = rdf["RR_ms"].values.astype(float)
            hr_all = rdf["HR_bpm"].values.astype(float)
            t_ds   = downsample_for_display(t_all)
            rr_ds  = downsample_for_display(rr_all)
            hr_ds  = downsample_for_display(hr_all)
            rr_mean = float(rr_all.mean()); rr_sd = float(rr_all.std())
            hr_mean = float(hr_all.mean())
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
            axes[0].set_ylabel("RR (ms)")
            axes[0].set_title(
                f"RR Intervals  ·  mean {rr_mean:.1f} ms  ·  SD {rr_sd:.1f} ms",
                loc="left", fontsize=9)
            axes[1].plot(t_ds, hr_ds, color=ORANGE_DARK, lw=0.9, zorder=2)
            axes[1].axhline(hr_mean, color=ORANGE_DARK, ls="--", lw=0.9, alpha=0.5)
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
            ("radar",         "05_hrv_radar",        None),
            ("beat",          "06_beat_template",    None),
            ("beat_dist",     "07_beat_morphology",  None),
            ("intervals",     "08_intervals",        None),
            ("intervals_ecg", "09_intervals_ecg",    None),
            ("rolling_hrv",   "10_rolling_hrv",      None),
        ]

        saved, skipped = 0, 0
        DPI = 200; FIG_W, FIG_H = 10.0, 5.0

        for slot_key, stem, custom_fn in catalogue:
            fpath = _os.path.join(out_dir, f"{stem}.png")
            try:
                export_fig = Figure(
                    figsize=(FIG_W, FIG_H), dpi=DPI,
                    facecolor=PLOT.get("bg", "#1A1A2E"))
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

        msg = (f"Saved {saved} figures to:\n{out_dir}"
               + (f"\n({skipped} skipped — not yet computed)" if skipped else ""))
        app._set_status(f"Figures exported — {saved} PNG files  ✓", GREEN)
        messagebox.showinfo("Figures exported", msg)

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
        )

    def write_excel(self, destination) -> None:
        """Write the formatted workbook to *destination* (path or BytesIO)."""
        app = self.app
        wb = self.build_excel_workbook()
        ExcelExporter.add_annotations_sheet(wb, app.analysis.annotations)
        wb.save(destination)

    def export_excel(self) -> None:
        app = self.app
        if not app.analysis.results:
            messagebox.showwarning("No results", "Run analysis first.")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=f"{app.ent_subject.get()}_ECG_{ts}.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if not path:
            return
        try:
            self.write_excel(path)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        app._set_status("Excel saved", GREEN)
        messagebox.showinfo("Saved", path)

    def export_zip(self) -> None:
        app = self.app
        if not app.analysis.results:
            messagebox.showwarning("No results", "Run analysis first.")
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
            ("radar",     "04_radar"),
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

        app._set_status("ZIP saved", GREEN)
        messagebox.showinfo("Saved", path)

    def export_pdf_report(self) -> None:
        """Generate a one-page PDF summary: ECG strip + KPI table + interpretation."""
        app = self.app
        if app.analysis.results is None:
            messagebox.showwarning("No results", "Run Core Analysis first.")
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
        ctx = EXPERIMENTAL_CONTEXTS.get(app.analysis.exp_context)

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
            if v is None:
                return
            lo, hi = app._current_ref(key)
            margin = (hi - lo) * 0.15
            if lo <= v <= hi:
                status = "✓"
            elif lo - margin <= v <= hi + margin:
                status = "~"
            else:
                status = "↑" if v > hi else "↓"
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

        fig  = _mpl_figure.Figure(figsize=(11.7, 8.3), facecolor=bg)   # A4 landscape
        gs   = gridspec.GridSpec(
            3, 2, figure=fig,
            left=0.06, right=0.96, top=0.88, bottom=0.07,
            hspace=0.45, wspace=0.30,
            height_ratios=[2.2, 1.6, 1.0],
        )

        # Title block
        ctx_label = ctx.label if ctx else "Contexte inconnu"
        qtc_label = "Mitchell" if app._qtc_formula() == "mitchell" else "Bazett"
        fig.text(0.06, 0.94, f"ECG Report — {sub}",
                 fontsize=15, color=fg, fontweight="bold", va="bottom")
        fig.text(0.06, 0.91,
                 f"Fichier : {app.signal.filepath or '—'}   |   "
                 f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}   |   "
                 f"Contexte : {ctx_label}   |   QTc : {qtc_label}",
                 fontsize=8, color=mut, va="bottom")
        fig.add_artist(_mpl_lines.Line2D([0.06, 0.94], [0.905, 0.905],
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
            ax_ecg.plot(t_seg, sig_seg, color=PLOT.get("ecg",CYAN_BRIGHT), lw=0.7)
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
        if rdf is not None and "RR_ms" in rdf.columns:
            rr_vals = rdf["RR_ms"].dropna().values
            if len(rr_vals) > 2:
                ax_rr.scatter(rr_vals[:-1], rr_vals[1:],
                              alpha=0.35, s=4, c=PLOT.get("ecg",CYAN_BRIGHT))
                lo = min(rr_vals.min(), rr_vals.min()) * 0.97
                hi = max(rr_vals.max(), rr_vals.max()) * 1.03
                ax_rr.plot([lo, hi], [lo, hi], "--", color=mut, lw=0.6)
        else:
            ax_rr.text(0.5, 0.5, "Données RR non disponibles",
                       ha="center", va="center", color=mut, fontsize=9,
                       transform=ax_rr.transAxes)

        # Panel D — Context reference reminder (bottom, full width)
        ax_ref = fig.add_subplot(gs[2, :])
        ax_ref.set_facecolor(bg)
        ax_ref.axis("off")
        if ctx:
            ax_ref.text(0.0, 1.0, f"Contexte : {ctx.label}",
                        fontsize=8, color=fg, fontweight="bold",
                        transform=ax_ref.transAxes, va="top")
            ax_ref.text(0.0, 0.68, ctx.description,
                        fontsize=7, color=mut,
                        transform=ax_ref.transAxes, va="top", wrap=True)
            # Mini reference bar
            ref_text = (
                f"HR {ctx.hr_lo:.0f}–{ctx.hr_hi:.0f} bpm   "
                f"SDNN {ctx.sdnn_lo:.1f}–{ctx.sdnn_hi:.0f} ms   "
                f"RMSSD {ctx.rmssd_lo:.1f}–{ctx.rmssd_hi:.0f} ms   "
                f"PR {ctx.pr_lo:.0f}–{ctx.pr_hi:.0f} ms   "
                f"QRS {ctx.qrs_lo:.0f}–{ctx.qrs_hi:.0f} ms   "
                f"QTc {ctx.qtc_lo:.0f}–{ctx.qtc_hi:.0f} ms"
            )
            ax_ref.text(0.0, 0.28, f"Plages normales → {ref_text}",
                        fontsize=7, color=mut,
                        transform=ax_ref.transAxes, va="top")

        # ── Save PDF ─────────────────────────────────────────────────────
        try:
            with _pdf_backend.PdfPages(path) as pp:
                pp.savefig(fig, dpi=200, facecolor=bg)
            plt.close(fig)
        except Exception as exc:
            messagebox.showerror("PDF export failed", str(exc))
            return

        app._set_status(f"PDF report saved — {os.path.basename(path)}", GREEN)
        messagebox.showinfo("PDF saved", path)

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
        OneWay   ECG intervals per beat, Arrhythmia episode table
        """
        app = self.app
        if app.analysis.results is None:
            messagebox.showwarning(
                "No results",
                "Run Full Analysis before exporting to Prism.")
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

        ctx             = EXPERIMENTAL_CONTEXTS.get(app.analysis.exp_context)
        beat_corr       = app.analysis.results.get("beat_corr")
        beat_peak_times = app.analysis.results.get("beat_peak_times")
        arr_events      = app.analysis.arrhythmia_events or []
        seg_a           = app.analysis.last_seg_a
        seg_b           = app.analysis.last_seg_b
        notes           = app.session.recording_notes
        fs              = float(app.signal.fs) if app.signal.fs else 1000.0

        try:
            n_tables = PrismExporter.build_and_write(
                path              = path,
                results           = app.analysis.results,       # type: ignore[arg-type]
                rolling_hrv_df    = app.analysis.rolling_hrv_df,
                epoch_df          = app.analysis.epoch_df,
                subject           = sub,
                context_label     = ctx.label if ctx else "",
                arrhythmia_events = arr_events if arr_events else None,
                beat_corr         = beat_corr,
                beat_peak_times   = beat_peak_times,
                segment_a         = seg_a,
                segment_b         = seg_b,
                fs                = fs,
                recording_notes   = notes,
            )
        except Exception as exc:
            log.warning("Prism export failed: %s", exc)
            messagebox.showerror("Prism export failed", str(exc))
            return

        app._set_status(
            f"Prism: {n_tables} tables → {os.path.basename(path)}", GREEN)
        messagebox.showinfo("Prism export complete",
                            f"{n_tables} tables saved to:\n{path}")

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
            messagebox.showwarning("No results", "Run Core Analysis first.")
            return
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
            out = rdf[["Time_s", "RR_ms", "HR_bpm"]].copy()
            beat_corr = (app.analysis.results or {}).get("beat_corr")
            if beat_corr is not None and len(beat_corr) == len(out):
                out["quality"] = np.where(beat_corr >= 0.90, "ok", "low")
            out.to_csv(path, index=False, float_format="%.4f")
            app._set_status(f"RR CSV saved → {os.path.basename(path)}", GREEN)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_arrhythmia_pdf(self) -> None:
        """Export a PDF with one annotated ECG strip per arrhythmia episode."""
        app = self.app
        if not app.analysis.arrhythmia_events:
            messagebox.showwarning("No arrhythmias", "Run arrhythmia classification first.")
            return
        if app.signal.filtered is None or app.signal.time is None or app.signal.fs is None:
            messagebox.showwarning("No signal", "No ECG signal loaded.")
            return
        sub  = app.ent_subject.get().strip() if app.ent_subject else "subject"
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")],
            initialfile=f"{sub}_arrhythmias.pdf",
            title="Export arrhythmia strips as PDF",
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

        app._set_status("Building arrhythmia PDF…", ORANGE)
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
                f"Arrhythmia PDF saved → {os.path.basename(path)}", GREEN)
        except Exception as exc:
            messagebox.showerror("PDF export failed", str(exc))
