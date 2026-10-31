# -*- coding: utf-8 -*-
"""
ecg.io.export
-------------
Excel (openpyxl) and GraphPad Prism (.pzfx) exporters.
"""
from __future__ import annotations

import io
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet as OpenpyxlWorksheet
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from ecg.core.models import (
    AnalysisResults, MouseECG, PREMATURE_DEFINITION_TEXT, event_kind_label,
)
from ecg.core.detection import premature_summary_rows
from ecg.core.analysis import (
    HR_RANGE_TEXT, LEGACY_SPECTRAL_TEXT, SPECTRAL_METHOD_TEXT,
    analysed_duration_s, analysis_provenance_rows, analysis_window_span,
    hrv_freq_is_legacy, hrv_freq_withheld_reason, interval_disclosure_rows,
    nonlinear_subset_text, qtc_formula_text, rr_exclusion_report,
    RR_ASYMMETRY_TEXT,
    rr_timeseries,
)
from ecg.core.autonomic import (
    KP_RAT_SHR, KS_RAT_SHR, beat_dataframe, to_dataframe,
)

log = logging.getLogger("ecg")


# ════════════════════════════════════════════════════════════
#  ABNORMAL EVENTS / PREMATURE BEATS -- shared by Excel and Prism
# ════════════════════════════════════════════════════════════

def abnormal_events_frame(events: "Optional[list]") -> pd.DataFrame:
    """One row per classified abnormal event (ArrhythmiaEvent)."""
    rows = []
    for ev in events or []:
        rows.append({
            "Type":        ev.kind,
            "Label":       event_kind_label(ev.kind),
            "Start_s":     round(float(ev.t_start), 3),
            "End_s":       round(float(ev.t_end), 3),
            "Duration_s":  round(float(ev.duration_s), 3),
            "N_beats":     int(ev.n_beats),
            "HR_bpm":      round(float(ev.hr_mean), 1),
            "RR_ms":       round(float(ev.rr_mean), 1),
            "Baseline_bpm": round(float(ev.baseline_hr), 1),
            "Delta_pct":   round(float(ev.delta_pct), 1),
            "Severity":    ev.severity,
            "N_removed_by_correction": int(getattr(ev, "n_removed", 0) or 0),
            "Description": ev.label,
        })
    return pd.DataFrame(rows)


def premature_beats_frame(summary: "Optional[dict]") -> pd.DataFrame:
    """One-row table of a detection.premature_beat_summary() (empty
    without one). Rates are per hour of the analysed span."""
    if not summary:
        return pd.DataFrame()
    s = summary

    def _r(x: Any) -> "Optional[float]":
        v = _finite(x)
        return round(v, 2) if v is not None else None

    return pd.DataFrame([{
        "N_premature_total":   int(s["n_premature"]),
        "N_isolated":          int(s["n_isolated"]),
        "N_couplets":          int(s["n_couplets"]),
        "N_runs_ge3":          int(s["n_runs"]),
        "N_beats_in_runs":     int(s["n_beats_in_runs"]),
        "N_removed_by_correction": int(s.get("n_removed", 0) or 0),
        "Premature_per_h":     _r(s["premature_per_h"]),
        "Isolated_per_h":      _r(s["isolated_per_h"]),
        "Couplets_per_h":      _r(s["couplets_per_h"]),
        "Runs_per_h":          _r(s["runs_per_h"]),
        "Analysed_span_s":     round(float(s.get("span_s", 0.0) or 0.0), 1),
        "Window_start_s":      _r(s.get("window_start_s")),
        "Window_end_s":        _r(s.get("window_end_s")),
        "N_removed_judged_spurious": int(s.get("n_removed_spurious", 0) or 0),
        "N_removed_unknown_type":    int(s.get("n_removed_unknown_type", 0) or 0),
    }])


# ════════════════════════════════════════════════════════════
#  AUTONOMIC (CSI / CPI) DISCLOSURE -- shared by Excel, Prism and the PDF
# ════════════════════════════════════════════════════════════

AUTONOMIC_REFERENCE = ("Candia-Rivera et al., J Physiol 2025, doi:10.1113/JP288400 "
                       "(code: github.com/diegocandiar/robust_hrv)")

# Longest resampled (20 Hz) series written to one Excel sheet / Prism table;
# longer recordings are decimated for the export, and the sheet says so.
AUTONOMIC_MAX_SERIES_ROWS = 200_000
PRISM_MAX_SERIES_ROWS = 50_000


def _finite(value: Any) -> "Optional[float]":
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def autonomic_calibration_caveat(ks: Any, kp: Any) -> str:
    """One-line ks / kp disclosure: the paper's rat (SHR) values are not
    validated in mice."""
    s, p = _finite(ks), _finite(kp)
    if s == KS_RAT_SHR and p == KP_RAT_SHR:
        return ("ks = 3, kp = 12: the rat (SHR) calibration of Candia-Rivera et al. "
                "2025 — NOT validated in mice")
    return (f"ks = {s if s is not None else float('nan'):g}, "
            f"kp = {p if p is not None else float('nan'):g}: user-set (the paper's rat "
            "SHR values ks = 3, kp = 12 are not validated in mice)")


def _band(band: Any) -> str:
    try:
        lo, hi = band
        return f"{float(lo):g}–{float(hi):g} Hz"
    except (TypeError, ValueError):
        return "?"


def _window_pair(value: Any) -> "Optional[tuple[float, float]]":
    """An analysis window (t_start, t_end) as finite floats, or None."""
    try:
        t0, t1 = value
        t0, t1 = float(t0), float(t1)
    except (TypeError, ValueError):
        return None
    return (t0, t1) if np.isfinite(t0) and np.isfinite(t1) else None


def _window_text(win: "tuple[float, float]") -> str:
    """'full signal', '100 – 200 s' or '100 s – end' (0 = open bound, as in
    the app's analysis window)."""
    t0, t1 = win
    if t0 <= 0 and t1 <= 0:
        return "full signal"
    return f"{t0:g} – {t1:g} s" if t1 > 0 else f"{t0:g} s – end"


def _same_window(a: "tuple[float, float]", b: "tuple[float, float]") -> bool:
    return abs(a[0] - b[0]) <= 1e-6 and abs(a[1] - b[1]) <= 1e-6


def autonomic_span_text(params: "dict", hrv_window: Any = None) -> str:
    """The span the Autonomic indices cover -- the first / last R peak they
    were computed from (``params['peak_span_s']``) and the analysis window
    those peaks came from (``params['analysis_window']``) -- and, when
    *hrv_window* (the window the core HRV results were computed on) is
    given, whether it is the same one. '' when the result records neither
    (an older session)."""
    p = params or {}
    parts = []
    peaks = _window_pair(p.get("peak_span_s"))
    if peaks is not None:
        parts.append(f"R peaks {peaks[0]:.1f} – {peaks[1]:.1f} s")
    win = _window_pair(p.get("analysis_window"))
    if win is not None:
        parts.append(f"analysis window {_window_text(win)}")
    if not parts:
        return ""
    text = " · ".join(parts)
    core = _window_pair(hrv_window)
    if win is not None and core is not None:
        if _same_window(win, core):
            text += " (the same as the core HRV results)"
        else:
            text += (f" — NOT the core HRV results' window ({_window_text(core)}): the "
                     "two cover different spans")
    return text


def autonomic_disclosure_rows(
    params: "dict",
    spwvd_params: "Optional[dict]" = None,
    event_params: "Optional[dict]" = None,
    hrv_window: Any = None,
) -> "list[tuple[str, str]]":
    """(label, value) rows for the Excel Cover sheet, the Prism project
    notes and the PDF header when Autonomic results are exported: method,
    window, ks / kp and the rat-calibration caveat, the span the indices
    cover (``autonomic_span_text``; compared with *hrv_window*, the core
    HRV results' analysis window, when given), the RR exclusions, and
    (when given) the SPWVD LF / HF settings and the event-locked windows.
    *params* is ``AutonomicResult.params``."""
    p = params or {}
    window = _finite(p.get("window_s"))
    ks, kp = _finite(p.get("ks")), _finite(p.get("kp"))
    gains = (f"ks = {ks:g}, kp = {kp:g}" if ks is not None and kp is not None
             else f"ks = {p.get('ks', '?')}, kp = {p.get('kp', '?')}")
    rows = [
        ("Autonomic indices", f"time-resolved CSI / CPI — {AUTONOMIC_REFERENCE}"),
        ("Autonomic method", f"{p.get('method_label') or p.get('method', '?')} "
                             f"[{p.get('method', '?')}]"),
        ("Autonomic window",
         (f"{window:g} s" if window is not None else "?")
         + f" sliding Poincaré window (≥ {p.get('min_pairs', '?')} valid adjacent RR "
           f"pairs, else withheld) · {p.get('n_beats', '?')} windows, "
           f"{p.get('n_windows_withheld', 0)} withheld"),
        ("Autonomic ks / kp", gains),
        ("ks / kp calibration", autonomic_calibration_caveat(p.get("ks"), p.get("kp"))),
        *([("Autonomic span", span)] if (span := autonomic_span_text(p, hrv_window)) else []),
        ("Autonomic RR",
         f"{p.get('n_valid', '?')} of {p.get('n_intervals', '?')} intervals valid, "
         f"{p.get('n_excluded', 0)} excluded (bounds {p.get('rr_min_ms', '?')}–"
         f"{p.get('rr_max_ms', '?')} ms, {p.get('n_breaks', 0)} removed-beat breaks) — "
         "never bridged"),
    ]
    if p.get("withheld_reason"):
        rows.append(("Autonomic withheld", f"nothing computed — {p['withheld_reason']}"))
    elif p.get("withheld_note"):
        rows.append(("Autonomic withheld", str(p["withheld_note"])))
    if spwvd_params:
        rows.append(("LF / HF (SPWVD)",
                     f"LF {_band(spwvd_params.get('lf_band'))}, HF "
                     f"{_band(spwvd_params.get('hf_band'))}, "
                     f"{_finite(spwvd_params.get('fs_resample')) or 20.0:g} Hz resampling, "
                     f"{spwvd_params.get('fbins', '?')} bins — unnormalised a.u."))
    if event_params:
        e = event_params
        rows.append(("Event-locked",
                     f"{e.get('n_events', '?')} event(s), {e.get('n_valid', '?')} valid · "
                     f"pre {_finite(e.get('pre_s')) or 0:g} s / post "
                     f"{_finite(e.get('post_s')) or 0:g} s · onset = annotation start · "
                     f"label {e.get('label_filter', 'All')}"))
    return rows


def autonomic_pdf_line(params: "dict", n_events: "Optional[int]" = None) -> str:
    """The PDF report's header line when an Autonomic result exists:
    method, window, ks / kp and the rat-calibration caveat."""
    p = params or {}
    ks, kp, w = _finite(p.get("ks")), _finite(p.get("kp")), _finite(p.get("window_s"))
    rat = ks == KS_RAT_SHR and kp == KP_RAT_SHR
    line = (f"Autonomic CSI / CPI (Candia-Rivera et al. 2025) : "
            f"{p.get('method_label') or p.get('method', '?')} · window "
            + (f"{w:g} s" if w is not None else "?")
            + (f" · ks = {ks:g}, kp = {kp:g} — " if ks is not None and kp is not None
               else " · ks / kp ? — ")
            + ("rat (SHR) calibration, NOT validated in mice" if rat
               else "user-set gains (the paper's rat values 3 / 12 are not validated in mice)"))
    win = _window_pair(p.get("analysis_window"))
    if win is not None and not (win[0] <= 0 and win[1] <= 0):
        line += f" · analysis window {_window_text(win)}"
    if n_events:
        line += f" · {int(n_events)} event(s) locked"
    return line


def _decimation_step(n: int, max_rows: int) -> int:
    return max(1, int(np.ceil(n / max(1, int(max_rows)))))


def _xl_value(value: Any) -> Any:
    """A DataFrame cell as an openpyxl value: NaN / None -> empty, numpy
    scalars unwrapped, containers as text."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    return value


# ════════════════════════════════════════════════════════════
#  EXCEL EXPORTER
# ════════════════════════════════════════════════════════════


# Units of the HRV_Frequency sheet's columns (raw values, as stored):
# current analyse_hrv_freq output vs a legacy nk.hrv_frequency table.
_FREQ_UNITS = {
    "HRV_VLF": "ms²", "HRV_LF": "ms²", "HRV_HF": "ms²", "HRV_TP": "ms²",
    "HRV_LnHF": "ln ms²", "HRV_LFn": "fraction of LF+HF",
    "HRV_HFn": "fraction of LF+HF", "HRV_LFpeak": "Hz", "HRV_HFpeak": "Hz",
}
_FREQ_UNITS_LEGACY = {
    "HRV_ULF": "a.u., legacy", "HRV_VLF": "a.u., legacy", "HRV_LF": "a.u., legacy",
    "HRV_HF": "a.u., legacy", "HRV_VHF": "a.u., legacy", "HRV_TP": "a.u., legacy",
    "HRV_LnHF": "ln a.u., legacy", "HRV_LFn": "fraction of TP, legacy",
    "HRV_HFn": "fraction of TP, legacy",
}


def hrv_freq_sheet_frame(results: dict) -> "Optional[pd.DataFrame]":
    """hrv_freq with each column header carrying its unit ("HRV_LF (ms²)",
    "HRV_LFn (fraction of LF+HF)"; a legacy NeuroKit table gets "a.u.,
    legacy" / "fraction of TP, legacy"), so the raw sheet -- whose LFn /
    HFn are fractions, shown ×100 as n.u. elsewhere -- cannot be misread."""
    fd = results.get("hrv_freq")
    if fd is None or fd.empty:
        return fd
    units = _FREQ_UNITS_LEGACY if hrv_freq_is_legacy(results) else _FREQ_UNITS
    return fd.rename(columns={c: f"{c} ({units[c]})" for c in fd.columns if c in units})


def _round_or_none(v: "Optional[float]", nd: int) -> "Optional[float]":
    return round(float(v), nd) if v is not None and np.isfinite(v) else None


class ExcelExporter:
    """Pure utility — builds formatted openpyxl workbooks with no GUI dependency.

    All methods are static/class-level.  ExportController.build_excel_workbook() delegates
    here, passing only the data it needs.  This makes the export logic testable
    without instantiating the full application.

    Sheet layout
    ─────────────────────────────────────────────────────────
    Cover          — subject info, analysis date, file metadata
    ECG_Signal     — Time, Raw, Filtered (downsampled ≤ 50 k rows),
                     Is_RPeak column (1/0)
    HR_Statistics  — mean/min/max/SD HR, N beats, RR intervals used /
                     excluded for HRV and the RR bounds used
    RR_Timeseries  — beat-by-beat Time, RR, HR and ``valid`` (every
                     interval; False = excluded from HRV, not dropped)
    HRV_Time       — NeuroKit2 time-domain metrics
    HRV_Frequency  — frequency-domain metrics
    HRV_NonLinear  — non-linear metrics (SD1/SD2, SampEn …)
    ECG_Intervals  — PR / QRS / QT / QTc per beat (definitions and QT
                     beat-to-beat variability on the Cover sheet)
    Epochs         — epoch-level HR / SDNN / RMSSD (if computed)
    """

    # ── Style helpers ─────────────────────────────────────────────────

    @staticmethod
    def alt_fill(even: bool) -> "PatternFill":
        color = "EFF3FB" if even else "FFFFFF"
        return PatternFill("solid", fgColor=color)

    @staticmethod
    def peak_fill() -> "PatternFill":
        """Highlight fill for R-peak rows in the ECG_Signal sheet."""
        return PatternFill("solid", fgColor="FFF3E0")  # light amber

    @staticmethod
    def thin_border() -> "Border":
        side = Side(style="thin", color="D0D0D0")
        return Border(left=side, right=side, top=side, bottom=side)

    @staticmethod
    def autofit_columns(ws: "OpenpyxlWorksheet",
                        min_w: int = 10, max_w: int = 42) -> None:
        """Set each column width to fit its widest cell."""
        for col_cells in ws.columns:
            width = min_w
            for cell in col_cells:
                if cell.value is not None:
                    width = max(width, min(max_w, len(str(cell.value)) + 4))
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = width  # type: ignore[arg-type]

    @classmethod
    def write_dataframe(
        cls,
        ws:           "OpenpyxlWorksheet",
        df:           "pd.DataFrame",
        header_color: str      = "1565C0",
        peak_col:     "str | None" = None,
        peak_set:     "set | None" = None,
        start_row:    int      = 1,
    ) -> None:
        """Write a DataFrame into *ws* with full formatting."""
        hdr_font  = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
        hdr_fill  = PatternFill("solid", fgColor=header_color)
        hdr_align = Alignment(horizontal="center", vertical="center", wrap_text=False)
        data_font = Font(name="Calibri", size=10)
        num_align = Alignment(horizontal="right")
        border    = cls.thin_border()

        for col_idx, col_name in enumerate(df.columns, start=1):
            cell = ws.cell(row=start_row, column=col_idx, value=str(col_name))
            cell.font      = hdr_font
            cell.fill      = hdr_fill
            cell.alignment = hdr_align
            cell.border    = border

        ws.row_dimensions[start_row].height = 18

        for row_idx, (_, series) in enumerate(df.iterrows(), start=1):
            is_peak = (peak_col is not None and peak_set is not None
                       and series.get(peak_col, None) in peak_set)
            fill = cls.peak_fill() if is_peak else cls.alt_fill(row_idx % 2 == 0)

            for col_idx, (col_name, value) in enumerate(series.items(), start=1):
                xrow = start_row + row_idx
                cell = ws.cell(row=xrow, column=col_idx)
                cell.font   = data_font
                cell.fill   = fill
                cell.border = border

                if isinstance(value, (bool, np.bool_)):
                    # Before the int branch (bool is an int subclass):
                    # written as a real TRUE/FALSE cell, not 1/0.
                    cell.value = bool(value)
                elif isinstance(value, float):
                    if pd.isna(value):
                        cell.value = None
                    else:
                        # Store the full-precision value; only the display
                        # format is rounded. Writing round(value, 4) into the
                        # cell used to permanently truncate the exported
                        # number itself -- re-reading the .xlsx (e.g. with
                        # pandas) got the rounded figure, not the original.
                        cell.value     = float(value)  # type: ignore[assignment]
                        cell.alignment = num_align
                        cell.number_format = (
                            "0.000000E+00"
                            if abs(value) >= 1e6 or (abs(value) < 0.001 and value != 0)
                            else "0.0000"
                        )
                elif isinstance(value, (int, np.integer)):
                    cell.value     = int(value)  # type: ignore[assignment]
                    cell.alignment = num_align
                else:
                    cell.value = value

        ws.freeze_panes = ws.cell(row=start_row + 1, column=1)  # type: ignore[assignment]

    @classmethod
    def build_workbook(
        cls,
        results:     "AnalysisResults",
        signal_flt:  "Optional[np.ndarray]",
        signal_raw:  "Optional[np.ndarray]",
        time:        "Optional[np.ndarray]",
        rpeaks_ok:   "Optional[np.ndarray]",
        fs:          int,
        filepath:    "Optional[str]",
        subject:     str,
        sig_quality: "Optional[int]",
        epoch_df:    "Optional[pd.DataFrame]",
        qtc_formula: "Optional[str]" = None,
        analysis_window: "Optional[tuple]" = None,
        artifact_report: "Optional[dict]" = None,
        filter_text:     "Optional[str]" = None,
        signal_crop:     "Optional[dict]" = None,
    ) -> "Workbook":
        """Build and return a formatted openpyxl Workbook.

        *qtc_formula* (a QTC_FORMULA_LABELS key or label) is the formula the
        ECG_Intervals ``QTc_ms`` column was computed with; when given it is
        recorded on the Cover sheet.

        The Cover sheet also records what is needed to reproduce the
        numbers (analysis_provenance_rows, same rows as the text report
        and the Prism notes): the recording duration, *analysis_window*
        ((t_start, t_end) the results were computed on, 0 = signal start /
        end) and the analysed duration (window end − start -- not the time
        of the last beat), *signal_crop* (the load-time crop,
        analysis.signal_crop_info; every time is relative to its start),
        *filter_text* (when given), *artifact_report*
        (the artifact correction in effect; None = not applied), the QTc
        formula, interval definitions and wave template.

        The Cover and HR_Statistics sheets disclose how many RR intervals
        HRV used and how many were excluded (too short / too long /
        spanning beats removed by artifact correction) against which RR
        bounds -- the HRV sheets are computed on the valid ones only, and
        a reader of the file must be able to tell.
        """
        if "hr" not in results:
            raise ValueError(
                "AnalysisResults is missing the 'hr' key -- run the core "
                "analysis before building the export workbook."
            )
        hr = results["hr"]  # type: ignore[typeddict-item]
        rr_report = rr_exclusion_report(cast(dict, results))
        wb = Workbook()

        # ── COVER sheet ──────────────────────────────────────────────────
        ws_cover = wb.active
        assert ws_cover is not None
        ws_cover.title = "Cover"
        ws_cover.sheet_view.showGridLines = False

        title_font = Font(name="Calibri", bold=True, size=20, color="1565C0")
        sub_font   = Font(name="Calibri", bold=True, size=12, color="333333")
        val_font   = Font(name="Calibri", size=11)
        label_fill = PatternFill("solid", fgColor="EFF3FB")

        ws_cover.column_dimensions["A"].width = 28
        ws_cover.column_dimensions["B"].width = 52
        ws_cover["A1"].value = "ECG Analysis Report"
        ws_cover["A1"].font  = title_font
        ws_cover.row_dimensions[1].height = 32

        sig_end = float(time[-1]) if time is not None and len(time) else None
        dur_str = f"{sig_end:.1f} s" if sig_end is not None else "—"
        # Window, analysed duration, filters, RR bounds / exclusions,
        # artifact correction, QTc formula, interval definitions, template.
        provenance = analysis_provenance_rows(
            cast(dict, results), window=analysis_window, signal_end_s=sig_end,
            artifact_report=artifact_report, filter_text=filter_text,
            qtc_formula=None, signal_crop=signal_crop)
        n_ivl_rows = len(interval_disclosure_rows(results.get("intervals")))
        if n_ivl_rows:
            provenance, ivl_rows = provenance[:-n_ivl_rows], provenance[-n_ivl_rows:]
        else:
            ivl_rows = []
        cover_rows = [
            ("Subject",        subject),
            ("File",           os.path.basename(filepath or "—")),
            ("Analysis date",  datetime.now().strftime("%Y-%m-%d  %H:%M:%S")),
            ("Sampling rate",  f"{fs} Hz"),
            ("Recording duration", dur_str),
            ("N beats",        str(hr["n"])),
            *provenance,
            ("HR mean",        f"{hr['mean']:.1f} bpm  (60000 / mean NN)"),
            ("HR range",       f"{hr['min']:.0f} – {hr['max']:.0f} bpm  ({HR_RANGE_TEXT})"),
            ("Signal quality", f"{sig_quality}%" if sig_quality is not None else "—"),
        ]
        # A withheld spectrum leaves no HRV_Frequency sheet: say why here.
        freq_withheld = hrv_freq_withheld_reason(cast(dict, results))
        if freq_withheld:
            cover_rows.append(("Frequency HRV", f"withheld — {freq_withheld}"))
        else:
            _fd = results.get("hrv_freq")
            if _fd is not None and not _fd.empty:
                # An older session's NeuroKit table is not in ms² / n.u.
                cover_rows.append(("Frequency HRV",
                                   LEGACY_SPECTRAL_TEXT if hrv_freq_is_legacy(cast(dict, results))
                                   else SPECTRAL_METHOD_TEXT))
        # Beats behind the non-linear metrics (entropy block may be capped).
        _nl = results.get("hrv_nonlin")
        if _nl is not None and not _nl.empty:
            cover_rows.append(("Non-linear HRV",
                               nonlinear_subset_text(_nl.attrs)
                               or "all valid RR intervals of the analysis window"))
            # HRV_PI / HRV_GI of the HRV_NonLinear sheet (NeuroKit's, same
            # conventions as the text report's RR-asymmetry block).
            cover_rows.append(("HR asymmetry (HRV_PI, HRV_GI)", RR_ASYMMETRY_TEXT))
        if qtc_formula is not None:
            cover_rows.append(("QTc formula", qtc_formula_text(qtc_formula)))
        # PR / QRS / QT landmarks and the beat-to-beat QT variability of the
        # ECG_Intervals sheet (single lead -- not classical QT dispersion).
        cover_rows.extend(ivl_rows)
        for i, (label, value) in enumerate(cover_rows, start=3):
            lbl = ws_cover.cell(row=i, column=1, value=label)
            val = ws_cover.cell(row=i, column=2, value=value)
            lbl.font = sub_font;  lbl.fill = label_fill
            val.font = val_font;  val.fill = label_fill

        # ── ECG_SIGNAL sheet ─────────────────────────────────────────────
        ws_sig = cast(OpenpyxlWorksheet, wb.create_sheet("ECG_Signal"))
        MAX_ROWS = 50_000
        n      = len(time) if time is not None else 0
        step   = max(1, n // MAX_ROWS)
        idx    = np.arange(0, n, step)

        rpeaks_sorted = (np.sort(np.asarray(rpeaks_ok))
                          if rpeaks_ok is not None and len(rpeaks_ok) else np.array([], dtype=int))
        if step > 1 and len(rpeaks_sorted):
            # Each exported row now stands in for the whole sample window
            # [idx[k], idx[k]+step) that got collapsed into it. Checking
            # "is idx[k] itself a peak sample" (the old exact-index test)
            # missed virtually every peak once step > 1, since a detected
            # R-peak's sample index is essentially never an exact multiple
            # of step. Flag the row instead if ANY peak sample falls in
            # the window it represents.
            starts = np.searchsorted(rpeaks_sorted, idx, side="left")
            ends   = np.searchsorted(rpeaks_sorted, idx + step, side="left")
            is_rpeak_col = (ends > starts).astype(int).tolist()
        else:
            rp_set = set(rpeaks_sorted.tolist())
            is_rpeak_col = [1 if i in rp_set else 0 for i in idx]

        sig_df = pd.DataFrame({
            "Time_s":   time[idx]    if time       is not None else [],
            "Raw_norm": signal_raw[idx] if signal_raw is not None else np.full(len(idx), np.nan),
            "Filtered": signal_flt[idx] if signal_flt is not None else np.full(len(idx), np.nan),
            "Is_RPeak": is_rpeak_col,
        })

        cls.write_dataframe(ws_sig, sig_df, header_color="37474F",
                            peak_col="Is_RPeak", peak_set={1})
        cls.autofit_columns(ws_sig)
        for col, w in [("A", 14), ("B", 14), ("C", 14), ("D", 11)]:
            ws_sig.column_dimensions[col].width = w

        if step > 1:
            note_row = len(sig_df) + 3
            ws_sig.cell(row=note_row, column=1,
                        value=f"Note: signal downsampled 1:{step} for export "
                              f"({n:,} → {len(idx):,} rows).")
            ws_sig.cell(row=note_row, column=1).font = Font(italic=True, color="888888", size=9)

        # ── HR_STATISTICS sheet ──────────────────────────────────────────
        ws_hr = cast(OpenpyxlWorksheet, wb.create_sheet("HR_Statistics"))
        hr_df = pd.DataFrame([{
            "HR_mean_bpm": round(hr["mean"], 2),
            "HR_min_bpm":  round(hr["min"],  2),
            "HR_max_bpm":  round(hr["max"],  2),
            "HR_std_bpm":  round(hr["std"],  2),
            "N_beats":     hr["n"],
            # RR intervals used for HRV / excluded (None = unknown, e.g.
            # results from a session saved before the counts existed)
            "N_RR_intervals":        rr_report["n_intervals"],
            "N_RR_valid":            rr_report["n_valid"],
            "N_RR_excluded":         rr_report["n_excluded"],
            "N_RR_too_short":        rr_report["n_too_short"],
            "N_RR_too_long":         rr_report["n_too_long"],
            "N_RR_spanning_removed": rr_report["n_spanning_removed"],
            "RR_excluded_pct":       rr_report["excluded_pct"],
            "RR_excluded_time_s":    rr_report["excluded_time_s"],
            "RR_min_ms":             rr_report["rr_min_ms"],
            "RR_max_ms":             rr_report["rr_max_ms"],
            # Recording length vs the span the HR / HRV were computed on.
            "Recording_duration_s": round(sig_end, 1) if sig_end is not None else None,
            "Window_start_s":       analysis_window_span(analysis_window, sig_end)[0],
            "Window_end_s":         analysis_window_span(analysis_window, sig_end)[1],
            "Analysed_duration_s":  _round_or_none(
                analysed_duration_s(cast(dict, results), analysis_window, sig_end), 3),
            "Fs_Hz":       fs,
        }])
        cls.write_dataframe(ws_hr, hr_df, header_color="2E7D32")
        cls.autofit_columns(ws_hr)

        # ── Metric DataFrames ────────────────────────────────────────────
        # RR_Timeseries holds EVERY interval with a ``valid`` column rather
        # than rr_df's valid-only rows, so an excluded pause is flagged in
        # the export instead of silently missing from it.
        rr_ts = rr_timeseries(cast(dict, results))
        sheet_specs: "list[tuple[str, Optional[pd.DataFrame], str]]" = [
            ("RR_Timeseries", rr_ts if not rr_ts.empty else results.get("rr_df"), "E65100"),
            ("HRV_Time",      results.get("hrv_time"),    "1565C0"),
            ("HRV_Frequency", hrv_freq_sheet_frame(cast(dict, results)), "6A1B9A"),
            ("HRV_NonLinear", results.get("hrv_nonlin"),  "00695C"),
            ("ECG_Intervals", results.get("intervals"),   "AD1457"),
        ]
        for name, df, color in sheet_specs:
            if df is None or df.empty:
                continue
            ws = cast(OpenpyxlWorksheet, wb.create_sheet(name))
            cls.write_dataframe(ws, df, header_color=color)
            cls.autofit_columns(ws)

        # ── EPOCHS sheet (optional) ──────────────────────────────────────
        if epoch_df is not None and not epoch_df.empty:
            ws_ep = cast(OpenpyxlWorksheet, wb.create_sheet("Epochs"))
            cls.write_dataframe(ws_ep, epoch_df, header_color="455A64")
            cls.autofit_columns(ws_ep)

        return wb

    AUTONOMIC_ONLY_NOTE = (
        "not computed (Analyze was not run) — this workbook holds the Time course > "
        "Autonomic results only; run Analyze for the core HRV / interval sheets")

    @classmethod
    def build_autonomic_workbook(
        cls,
        subject:  str,
        filepath: "Optional[str]",
        fs:       "Optional[float]",
        time:     "Optional[np.ndarray]" = None,
    ) -> "Workbook":
        """A workbook holding only a Cover sheet (subject, file, rate,
        duration, and that the core HRV results were not computed), for an
        export of the Time course > Autonomic results before Analyze -- they need
        only the detected peaks. Add the Autonomic sheets with
        ``add_autonomic_sheets`` (and ``add_annotations_sheet``)."""
        wb = Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "Cover"
        ws.sheet_view.showGridLines = False
        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 52
        ws["A1"].value = "ECG Analysis Report — Autonomic only"
        ws["A1"].font = Font(name="Calibri", bold=True, size=20, color="1565C0")
        ws.row_dimensions[1].height = 32
        sub_font = Font(name="Calibri", bold=True, size=12, color="333333")
        val_font = Font(name="Calibri", size=11)
        fill = PatternFill("solid", fgColor="EFF3FB")
        rate = _finite(fs)
        rows = [
            ("Subject",       subject),
            ("File",          os.path.basename(filepath or "—")),
            ("Analysis date", datetime.now().strftime("%Y-%m-%d  %H:%M:%S")),
            ("Sampling rate", f"{rate:g} Hz" if rate is not None else "—"),
            ("Recording duration", f"{float(time[-1]):.1f} s" if time is not None and len(time)
                                   else "—"),
            ("Core HRV",      cls.AUTONOMIC_ONLY_NOTE),
        ]
        for i, (label, value) in enumerate(rows, start=3):
            a = ws.cell(row=i, column=1, value=label)
            b = ws.cell(row=i, column=2, value=value)
            a.font, a.fill = sub_font, fill
            b.font, b.fill = val_font, fill
        return wb

    @classmethod
    def add_annotations_sheet(cls, wb, annotations: "list[dict]") -> None:
        """Append an Annotations sheet to an existing workbook."""
        if not annotations:
            return
        ann_df = pd.DataFrame([{
            "Start_s":    a["t_start"],
            "End_s":      a["t_end"],
            "Duration_s": round(a["t_end"] - a["t_start"], 4),
            "Label":      a.get("label", ""),
            "Color":      a.get("color", ""),
        } for a in annotations])
        ws = cast(OpenpyxlWorksheet, wb.create_sheet("Annotations"))
        cls.write_dataframe(ws, ann_df, header_color="4A148C")
        cls.autofit_columns(ws)

    @classmethod
    def add_abnormal_events_sheets(cls, wb, events: "Optional[list]",
                                   summary: "Optional[dict]",
                                   stale_note: "Optional[str]" = None) -> None:
        """Append the Events tab's results: a Premature_Beats
        sheet (one row of counts and per-hour rates -- isolated beats,
        couplets, runs >= 3, beats removed by artifact correction and
        still counted -- with the RR-based definition below it), an
        Abnormal_Events sheet (one row per event) and the same premature
        rows on the Cover sheet. No-op before a classification.
        *stale_note* (the classification no longer matches the exported
        peaks / window; the caller then passes no events or summary) is
        written on the Cover instead."""
        if stale_note and "Cover" in wb.sheetnames:
            cover = wb["Cover"]
            row = cover.max_row + 1
            cover.cell(row=row, column=1, value="Abnormal events")
            cover.cell(row=row, column=2, value=stale_note)
        if summary:
            df = premature_beats_frame(summary)
            ws = cast(OpenpyxlWorksheet, wb.create_sheet("Premature_Beats"))
            cls.write_dataframe(ws, df, header_color="E65100")
            cls.autofit_columns(ws)
            note = ws.cell(row=len(df) + 3, column=1,
                           value=f"Definition: {PREMATURE_DEFINITION_TEXT}")
            note.font = Font(italic=True, color="888888", size=9)
            if "Cover" in wb.sheetnames:
                cover = wb["Cover"]
                row = cover.max_row + 1
                for label, value in premature_summary_rows(summary):
                    cover.cell(row=row, column=1, value=label.strip())
                    cover.cell(row=row, column=2, value=value)
                    row += 1
        if events:
            ws_ev = cast(OpenpyxlWorksheet, wb.create_sheet("Abnormal_Events"))
            cls.write_dataframe(ws_ev, abnormal_events_frame(events), header_color="B71C1C")
            cls.autofit_columns(ws_ev)

    @classmethod
    def leave_out(cls, wb, sheets: "tuple[str, ...]", cover_rows: "tuple[str, ...]",
                  note: str) -> None:
        """Take *sheets* and the *cover_rows* (by label) out of a built workbook,
        and say on the Cover what was left out (*note*; the Report tab's Include).
        No-op with nothing left out."""
        for name in sheets:
            if name in wb.sheetnames:
                wb.remove(wb[name])
        if "Cover" not in wb.sheetnames:
            return
        cover = wb["Cover"]
        for row in range(cover.max_row, 0, -1):
            if cover.cell(row=row, column=1).value in cover_rows:
                cover.delete_rows(row)
        if note:
            row = cover.max_row + 1
            cover.cell(row=row, column=1, value="Left out of this export")
            cover.cell(row=row, column=2, value=note)

    @classmethod
    def add_metrics_sheet(cls, wb, table: Any) -> None:
        """Append a Metrics sheet: one line per row of a
        core.metric_table.MetricTable -- raw value (empty when missing), unit,
        basis n, what the basis is, module state, tier. Additive: every other
        sheet is written without it."""
        ws = cast(OpenpyxlWorksheet, wb.create_sheet("Metrics"))
        cls.write_dataframe(ws, table.to_frame(), header_color="1A237E")
        cls.autofit_columns(ws)

    # ── Autonomic (CSI / CPI) and event-locked sheets ─────────────────

    @classmethod
    def _write_long(cls, ws: "OpenpyxlWorksheet", df: "pd.DataFrame",
                    header_color: str) -> None:
        """Write *df* from A1 with write_dataframe's header style and plain
        value cells (no per-cell styling -- fast for long series); NaN is an
        empty cell, values keep full precision."""
        hdr_font = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
        hdr_fill = PatternFill("solid", fgColor=header_color)
        border = cls.thin_border()
        for col_idx, name in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=col_idx, value=str(name))
            cell.font, cell.fill, cell.border = hdr_font, hdr_fill, border
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for row in df.itertuples(index=False, name=None):
            ws.append([_xl_value(v) for v in row])
        ws.freeze_panes = ws.cell(row=2, column=1)  # type: ignore[assignment]
        for col_idx, name in enumerate(df.columns, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = max(12, len(str(name)) + 4)

    @staticmethod
    def _write_block(ws: "OpenpyxlWorksheet", rows: "list[tuple[str, Any]]",
                     col: int, start_row: int = 1, set_widths: bool = True) -> int:
        """Key / value block at column *col*; a row whose value is None is
        a bold section title. Returns the next free row."""
        bold = Font(name="Calibri", bold=True, size=10, color="1565C0")
        key_font = Font(name="Calibri", bold=True, size=10)
        val_font = Font(name="Calibri", size=10)
        r = start_row
        for key, value in rows:
            k = ws.cell(row=r, column=col, value=key)
            if value is None:
                k.font = bold
            else:
                k.font = key_font
                v = ws.cell(row=r, column=col + 1, value=_xl_value(value))
                v.font = val_font
                v.alignment = Alignment(horizontal="left", wrap_text=False)
            r += 1
        if set_widths:
            ws.column_dimensions[get_column_letter(col)].width = 26
            ws.column_dimensions[get_column_letter(col + 1)].width = 60
        return r

    @classmethod
    def add_autonomic_sheets(
        cls,
        wb,
        result: Any,
        spwvd: "Optional[dict]" = None,
        event_table: "Optional[pd.DataFrame]" = None,
        event_params: "Optional[dict]" = None,
        max_series_rows: int = AUTONOMIC_MAX_SERIES_ROWS,
        hrv_window: Any = None,
    ) -> "list[str]":
        """Append the Autonomic (CSI / CPI, core.autonomic) results to *wb*:

        Autonomic         -- the beat-wise window table (window median time,
                             end, SD1 / SD2 / CCD before re-centring, pairs)
                             and, at its right, the recording-level values
                             (SD01, SD02, D0, counts) and every parameter
        Autonomic_Series  -- the 20 Hz CSI / CPI / HR / D / SD1 / SD2 series
                             (decimated past *max_series_rows*, noted)
        Autonomic_LFHF    -- the SPWVD LF / HF series, when *spwvd* is a
                             computed result
        EventLocked       -- *event_table* (one row per event: pre / post /
                             delta of every series) with *event_params*
                             (``notes`` listed under it), when given

        and the method / window / ks / kp / rat-calibration disclosure rows
        on the Cover sheet -- with the span the indices cover, compared with
        *hrv_window* (the core HRV results' analysis window) when given.
        *result* is an AutonomicResult. Returns the names of the sheets
        added."""
        added: "list[str]" = []
        p = dict(result.params)
        sp_ok = (isinstance(spwvd, dict) and "error" not in spwvd
                 and len(np.asarray(spwvd.get("t", ()))) > 0)
        sp_params = dict(spwvd.get("params", {}) or {}) if sp_ok else None   # type: ignore[union-attr]
        has_events = event_table is not None and not event_table.empty

        # ── Cover rows ──
        if "Cover" in wb.sheetnames:
            ws_c = wb["Cover"]
            sub_font = Font(name="Calibri", bold=True, size=12, color="333333")
            val_font = Font(name="Calibri", size=11)
            fill = PatternFill("solid", fgColor="EFF3FB")
            r = ws_c.max_row + 2
            for label, value in autonomic_disclosure_rows(
                    p, sp_params, event_params if has_events else None, hrv_window):
                a = ws_c.cell(row=r, column=1, value=label)
                b = ws_c.cell(row=r, column=2, value=value)
                a.font, a.fill = sub_font, fill
                b.font, b.fill = val_font, fill
                r += 1

        # ── Autonomic: beat-wise table + recording level + parameters ──
        beat = beat_dataframe(result)
        ws = cast(OpenpyxlWorksheet, wb.create_sheet("Autonomic"))
        cls._write_long(ws, beat, "B71C1C")
        hr = np.asarray(result.hr_bpm, dtype=np.float64)
        hr_mean = float(np.nanmean(hr)) if np.isfinite(hr).any() else None
        block: "list[tuple[str, Any]]" = [
            ("RECORDING LEVEL", None),
            ("SD01_ms (whole-recording SD1)", result.sd01),
            ("SD02_ms (whole-recording SD2)", result.sd02),
            ("D0_ms (whole-recording CCD)", result.d0),
            # time-average of the resampled HR series -- not the app-wide
            # mean HR (60000 / mean NN), hence the explicit name
            ("HR_series_mean_bpm (time-average of the 20 Hz HR series)", hr_mean),
            ("N_windows (beats evaluated)", p.get("n_beats")),
            ("N_windows_withheld", p.get("n_windows_withheld")),
        ]
        if sp_ok:
            for key, name in (("lf", "LF_mean_au"), ("hf", "HF_mean_au")):
                v = np.asarray(spwvd[key], dtype=np.float64)                  # type: ignore[index]
                block.append((name, float(np.nanmean(v)) if np.isfinite(v).any() else None))
        block += [("", None), ("DISCLOSURE", None)]
        block += [(k, v) for k, v in autonomic_disclosure_rows(p, sp_params,
                                                               hrv_window=hrv_window)]
        block += [("", None), ("PARAMETERS", None)]
        block += [(str(k), "" if v is None else v) for k, v in p.items()]
        if sp_ok:
            block += [("", None), ("SPWVD PARAMETERS", None)]
            block += [(f"spwvd_{k}", "" if v is None else v) for k, v in sp_params.items()]  # type: ignore[union-attr]
        cls._write_block(ws, block, col=len(beat.columns) + 2)
        added.append("Autonomic")

        # ── Autonomic_Series: the resampled indices ──
        series = to_dataframe(result)
        step = _decimation_step(len(series), max_series_rows)
        ws_s = cast(OpenpyxlWorksheet, wb.create_sheet("Autonomic_Series"))
        cls._write_long(ws_s, series.iloc[::step], "C62828")
        note = (f"Resampled at {float(p.get('fs_out', 20.0)):g} Hz; CSI = D̃ + ks·SD2, "
                f"CPI = D + kp·SD1 (D̃ = D mirrored about its mean); NaN (empty) where "
                f"a window was withheld or across an RR gap.")
        if step > 1:
            note += (f" Decimated 1:{step} for export ({len(series):,} → "
                     f"{len(series.iloc[::step]):,} rows).")
        cls._write_block(ws_s, [("NOTE", None), ("", note)], col=len(series.columns) + 2)
        added.append("Autonomic_Series")

        # ── Autonomic_LFHF ──
        if sp_ok:
            lfhf = pd.DataFrame({
                "Time_s": np.asarray(spwvd["t"], dtype=np.float64),            # type: ignore[index]
                "LF_au":  np.asarray(spwvd["lf"], dtype=np.float64),           # type: ignore[index]
                "HF_au":  np.asarray(spwvd["hf"], dtype=np.float64),           # type: ignore[index]
            })
            step_s = _decimation_step(len(lfhf), max_series_rows)
            ws_l = cast(OpenpyxlWorksheet, wb.create_sheet("Autonomic_LFHF"))
            cls._write_long(ws_l, lfhf.iloc[::step_s], "6A1B9A")
            note_l = ("SPWVD band power, unnormalised a.u. (compare within a recording, "
                      "not with Welch powers); blank across RR gaps > 2 s (each stretch "
                      "between them is computed as its own series) and over stretches "
                      "too short to compute.")
            if step_s > 1:
                note_l += f" Decimated 1:{step_s} for export."
            cls._write_block(ws_l, [("NOTE", None), ("", note_l)], col=5)
            added.append("Autonomic_LFHF")

        # ── EventLocked ──
        if has_events:
            ws_e = cast(OpenpyxlWorksheet, wb.create_sheet("EventLocked"))
            cls.write_dataframe(ws_e, event_table.reset_index(drop=True),  # type: ignore[union-attr]
                                header_color="4A148C")
            cls.autofit_columns(ws_e, max_w=24)
            ep = dict(event_params or {})
            notes = [str(n) for n in ep.pop("notes", []) or []]
            r = len(event_table) + 3                                       # type: ignore[arg-type]
            rows_e: "list[tuple[str, Any]]" = [("EVENT-LOCKED PARAMETERS", None)]
            rows_e += [(str(k), "" if v is None else v) for k, v in ep.items()]
            if notes:
                rows_e += [("", None), ("NOTES", None)] + [("", n) for n in notes]
            cls._write_block(ws_e, rows_e, col=1, start_row=r, set_widths=False)
            added.append("EventLocked")
        return added


# ── Predefined annotation colours ─────────────────────────────────────────────
# Fixed swatch options for user-created annotations -- intentionally NOT tied
# to the live UI theme (an annotation colour a user picked shouldn't change
# meaning if they later switch the app's appearance preset). Four of these
# used to be imported from theme.py (coincidentally matching its palette at
# the time), which pulled a customtkinter import (and its module-level
# ctk.set_appearance_mode()/set_default_color_theme() calls) into every
# process that imports export.py, even ones with no Tkinter/CTk dependency
# otherwise. Hardcoded here to match the previous values exactly and keep
# that contract true.
ANNOTATION_COLORS: list[tuple[str, str]] = [
    ("#E65100", "Orange"),
    ("#1A56DB", "Blue"),
    ("#AD1457", "Pink"),
    ("#2E7D32", "Green"),
    ("#6A1B9A", "Purple"),
    ("#00838F", "Teal"),
    ("#F9A825", "Yellow"),
    ("#37474F", "Dark grey"),
]



def metric_table_prism_frame(table: Any) -> "pd.DataFrame":
    """The rows with a finite value, as a Prism Column table: value and basis n, the
    unit and the state in the row title (a Prism cell holds a number)."""
    titles, values, basis = [], [], []
    for row in table.rows:
        try:
            value = float(row.value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        unit = f" ({row.unit})" if row.unit else ""
        state = f" · {row.state}" if row.state else ""
        titles.append(f"{row.group}: {row.label}{unit}{state}")
        values.append(value)
        basis.append(float(row.basis_n) if row.basis_n is not None else float("nan"))
    return pd.DataFrame({"Value": values, "Basis n": basis}, index=pd.Index(titles))


class PrismExporter:
    """Pure utility — exports ECG analysis results to GraphPad Prism .pzfx format.

    Uses the official `pzfx` Python library (pip install pzfx) which produces
    correctly structured XML accepted by Prism 5–10 without modification.

    Tables exported
    ───────────────
    XY        — RR intervals & HR tachogram (+ its HRV-validity flag),
                ECG intervals timeline, Rolling HRV timeline, Epoch HRV
                (+ their per-window / per-epoch interval counts)
    Column    — HRV time-domain summary, frequency-domain summary,
                non-linear summary  (with named row titles)
    OneWay    — ECG intervals per beat (PR / QRS / QT / QTc distributions)
    """

    @staticmethod
    def _require_pzfx() -> None:
        try:
            import pzfx  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "The 'pzfx' package is required for GraphPad Prism export.\n"
                "Install it with: pip install pzfx"
            )

    @classmethod
    def build_and_write(
        cls,
        path:            "str | Path",
        results:         "AnalysisResults",
        rolling_hrv_df:  "Optional[pd.DataFrame]" = None,
        epoch_df:        "Optional[pd.DataFrame]"  = None,
        subject:         str = "",
        context_label:   str = "",
        arrhythmia_events: "Optional[list]" = None,
        segment_a: "Optional[dict]" = None,
        segment_b: "Optional[dict]" = None,
        fs: "Optional[float]" = None,
        recording_notes: str = "",
        qtc_formula: "Optional[str]" = None,
        autonomic: Any = None,
        autonomic_spwvd: "Optional[dict]" = None,
        event_table: "Optional[pd.DataFrame]" = None,
        event_params: "Optional[dict]" = None,
        hrv_window: Any = None,
        premature_summary: "Optional[dict]" = None,
        signal_end_s:    "Optional[float]" = None,
        artifact_report: "Optional[dict]" = None,
        filter_text:     "Optional[str]" = None,
        signal_crop:     "Optional[dict]" = None,
        left_out:        str = "",
        metrics:         Any = None,
    ) -> int:
        """Build all tables and write directly to *path*.

        *metrics* (a core.metric_table.MetricTable) adds a last table of its rows:
        value and basis n, with the unit and the state in the row title.

        *left_out* (what the Report tab's Include took out of the arguments)
        is stated in the project notes.

        The project notes record what is needed to reproduce the tables
        (analysis_provenance_rows, same rows as the Excel Cover sheet):
        the analysis window *hrv_window* and the analysed duration (window
        end − start, the window end defaulting to *signal_end_s*),
        *filter_text*, RR bounds / exclusions, *artifact_report* (None =
        not applied), the QTc formula, interval definitions and template,
        and the load-time *signal_crop* (analysis.signal_crop_info).

        *premature_summary* (detection.premature_beat_summary) adds the
        RR-based premature-beat counts / rates and their definition to the
        project notes.

        *qtc_formula* (a QTC_FORMULA_LABELS key or label), when given, is
        recorded in the project notes next to the QTc tables it produced.

        *autonomic* (an AutonomicResult) adds XY tables of the 20 Hz CSI /
        CPI and D / SD1 / SD2 series (and of the SPWVD LF / HF when
        *autonomic_spwvd* is a computed result), decimated past
        PRISM_MAX_SERIES_ROWS; *event_table* (one row per event, columns
        ``k_pre`` / ``k_post``, a ``valid`` flag) a Column table of the
        valid events' pre / post values, ready for a paired test in Prism.
        The project notes then state the method, window, ks / kp and the
        rat-calibration caveat (autonomic_disclosure_rows), and the span the
        indices cover compared with *hrv_window* (the core HRV results'
        analysis window) when given.

        Returns the number of tables written.
        Raises RuntimeError if pzfx is not installed.
        """
        cls._require_pzfx()
        from pzfx import write_pzfx

        tables:    "dict[str, pd.DataFrame]" = {}
        row_names: "list[bool]"              = []
        x_cols:    "list[str | int]"         = []

        def _add(name: str, df: "pd.DataFrame",
                 x_col: "str | int" = 0,
                 row_name: bool = False) -> None:
            tables[name]    = df
            x_cols.append(x_col)
            row_names.append(row_name)

        # ── 1. RR intervals & HR tachogram (XY) ─────────────────────────
        # Every interval. The RR/HR of an interval excluded from HRV
        # (outside the RR bounds, or spanning a removed beat) is written as
        # a Prism *excluded* value (pzfx's "*" suffix), so Prism shows it
        # but leaves it out of its analyses -- consistent with the app's
        # HRV. The explicit 0/1 validity flag goes in its own table: as a Y
        # column here Prism would plot it as a data set next to RR and HR.
        # (pzfx writes every value as str(value) whatever n_digits, which
        # only sets the displayed decimals, so the "*" strings carry the
        # same precision as the plain floats.)
        rr_ts = rr_timeseries(cast(dict, results))
        rr_df = results.get("rr_df")
        if not rr_ts.empty:
            ok = rr_ts["valid"].to_numpy(dtype=bool)
            t_rr = rr_ts["Time_s"].to_numpy(dtype=float)

            def _mark(values: "pd.Series") -> "list":
                return [float(v) if keep or not np.isfinite(v) else f"{float(v)!r}*"
                        for v, keep in zip(values.to_numpy(dtype=float), ok)]

            df = pd.DataFrame({
                "Time (s)": t_rr,
                "RR (ms)":  _mark(rr_ts["RR_ms"]),
                "HR (bpm)": _mark(rr_ts["HR_bpm"]),
            })
            _add("RR Intervals & HR", df, x_col="Time (s)")
            _add("RR Intervals — HRV Validity",
                 pd.DataFrame({"Time (s)": t_rr,
                               "Valid (1 = used for HRV)": ok.astype(int)}),
                 x_col="Time (s)")
        elif rr_df is not None and not rr_df.empty:
            df = rr_df[["Time_s", "RR_ms", "HR_bpm"]].copy()
            df.columns = pd.Index(["Time (s)", "RR (ms)", "HR (bpm)"])
            _add("RR Intervals & HR", df, x_col="Time (s)")  # type: ignore[arg-type]

        # ── 2. HRV time-domain summary (Column with row titles) ──────────
        hrv_t = results.get("hrv_time")
        if hrv_t is not None and not hrv_t.empty:
            key_map = [
                ("HRV_MeanNN",   "MeanNN (ms)"),
                ("HRV_SDNN",     "SDNN (ms)"),
                ("HRV_RMSSD",    "RMSSD (ms)"),
                ("HRV_pNN6",     "pNN6 (%)"),
                ("HRV_pNN50",    "pNN50 (%)"),
                ("HRV_CVNN",     "CVNN"),
                ("HRV_MedianNN", "MedianNN (ms)"),
                ("HRV_MadNN",    "MadNN (ms)"),
                ("HRV_IQRNN",    "IQRNN (ms)"),
            ]
            rows = {
                label: float(hrv_t[col].values[0])
                for col, label in key_map
                if col in hrv_t.columns
                and pd.notna(hrv_t[col].values[0])
                and pd.api.types.is_float_dtype(hrv_t[col])
            }
            if rows:
                df = pd.DataFrame({"Value": rows.values()},
                                  index=pd.Index(list(rows.keys())))
                _add("HRV — Time Domain", df, x_col=0, row_name=True)

        # ── 3. HRV frequency-domain summary (Column) ────────────────────
        hrv_f = results.get("hrv_freq")
        if hrv_f is not None and not hrv_f.empty:
            if hrv_freq_is_legacy(cast(dict, results)):
                # Older session (nk.hrv_frequency): arbitrary units, LFn = LF / TP.
                key_map_f = [
                    ("HRV_LF",    "LF (a.u., legacy)"),
                    ("HRV_HF",    "HF (a.u., legacy)"),
                    ("HRV_VLF",   "VLF (a.u., legacy)"),
                    ("HRV_TP",    "Total Power (a.u., legacy)"),
                    ("HRV_LFn",   "LF (% of TP, legacy)"),
                    ("HRV_HFn",   "HF (% of TP, legacy)"),
                    ("HRV_LFHF",  "LF/HF ratio"),
                ]
            else:
                key_map_f = [
                    ("HRV_LF",    "LF (ms²)"),
                    ("HRV_HF",    "HF (ms²)"),
                    ("HRV_VLF",   "VLF (ms²)"),
                    ("HRV_TP",    "Total Power (ms²)"),
                    ("HRV_LFn",   "LF (n.u., %)"),
                    ("HRV_HFn",   "HF (n.u., %)"),
                    ("HRV_LFHF",  "LF/HF ratio"),
                    ("HRV_LFpeak","LF peak (Hz)"),
                    ("HRV_HFpeak","HF peak (Hz)"),
                ]
            # LFn / HFn are fractions LF or HF / (LF + HF): shown ×100 (n.u.)
            rows_f = {
                label: float(hrv_f[col].values[0]) * (100.0 if col in ("HRV_LFn", "HRV_HFn") else 1.0)
                for col, label in key_map_f
                if col in hrv_f.columns and pd.notna(hrv_f[col].values[0])
            }
            if rows_f:
                df = pd.DataFrame({"Value": rows_f.values()},
                                  index=pd.Index(list(rows_f.keys())))
                _add("HRV — Frequency Domain", df, x_col=0, row_name=True)

        # ── 4. HRV non-linear summary (Column) ──────────────────────────
        hrv_nl = results.get("hrv_nonlin")
        if hrv_nl is not None and not hrv_nl.empty:
            key_map_nl = [
                ("HRV_SD1",        "SD1 (ms)"),
                ("HRV_SD2",        "SD2 (ms)"),
                ("HRV_SD1SD2",     "SD1/SD2"),
                ("HRV_S",          "S (ms²)"),
                ("HRV_SampEn",     "SampEn"),
                ("HRV_ApEn",       "ApEn"),
                ("HRV_DFA_alpha1", "DFA α1"),
                ("HRV_DFA_alpha2", "DFA α2"),
                ("HRV_PI",         "Porta Index PI (%)"),
                ("HRV_GI",         "Guzik Index GI (%)"),
            ]
            rows_nl = {
                label: float(hrv_nl[col].values[0])
                for col, label in key_map_nl
                if col in hrv_nl.columns and pd.notna(hrv_nl[col].values[0])
            }
            if rows_nl:
                df = pd.DataFrame({"Value": rows_nl.values()},
                                  index=pd.Index(list(rows_nl.keys())))
                _add("HRV — Non-Linear", df, x_col=0, row_name=True)

        # ── 5. ECG intervals per beat (OneWay — distributions) ───────────
        ivl = results.get("intervals")
        if ivl is not None and not ivl.empty:
            col_map = [
                ("PR_ms",  "PR (ms)"),
                ("QRS_ms", "QRS (ms)"),
                ("QT_ms",  "QT (ms)"),
                ("QTc_ms", "QTc (ms)"),
            ]
            ivl_cols = {
                label: ivl[col].dropna().values
                for col, label in col_map
                if col in ivl.columns and ivl[col].notna().sum() > 0
            }
            if ivl_cols:
                # OneWay: pad shorter columns with NaN so DataFrame is rectangular
                max_len = max(len(v) for v in ivl_cols.values())
                df = pd.DataFrame({
                    k: pd.Series(
                        np.pad(np.asarray(v, dtype=float),
                               (0, max_len - len(v)),
                               constant_values=np.nan)
                    )
                    for k, v in ivl_cols.items()
                })
                _add("ECG Intervals per Beat", df, x_col=0, row_name=False)

        # ── 6. ECG intervals timeline (XY) ──────────────────────────────
        # analyse_intervals times each beat by its R peak (R_peak_s); it has
        # no Time_s column, so keying on Time_s alone never wrote this table.
        _ivl_t = next((c for c in ("Time_s", "R_peak_s")
                       if ivl is not None and c in ivl.columns), None)
        if ivl is not None and not ivl.empty and _ivl_t is not None:
            tl_cols = {
                label: ivl[col].fillna(np.nan).values
                for col, label in [
                    ("PR_ms",  "PR (ms)"), ("QRS_ms", "QRS (ms)"),
                    ("QT_ms",  "QT (ms)"), ("QTc_ms", "QTc (ms)"),
                ]
                if col in ivl.columns
            }
            if tl_cols:
                tl_df = pd.DataFrame({"Time (s)": ivl[_ivl_t].values, **tl_cols})
                _add("ECG Intervals — Timeline", tl_df, x_col="Time (s)")

        # Per-window / per-epoch bookkeeping counts: kept out of the metric
        # XY tables (Prism would plot them as data sets next to HR/SDNN...)
        # and written to a counts table of their own.
        def _add_counts(name: str, t: "pd.Series", df_src: "pd.DataFrame",
                        cols: "list[str]") -> None:
            present = [c for c in cols if c in df_src.columns]
            if present:
                _add(name, pd.DataFrame({"Time (s)": np.asarray(t, dtype=float),
                                         **{c: df_src[c].to_numpy() for c in present}}),
                     x_col="Time (s)")

        # ── 7. Rolling HRV timeline (XY) ────────────────────────────────
        if rolling_hrv_df is not None and not rolling_hrv_df.empty:
            roll_counts = ["n_beats", "n_valid", "n_excluded", "n_poincare_pairs"]
            df_r = rolling_hrv_df.drop(
                columns=[c for c in roll_counts + ["low_n"] if c in rolling_hrv_df.columns])
            rename = {"t_mid": "Time (s)", "HR": "HR (bpm)",
                      "SDNN": "SDNN (ms)", "RMSSD": "RMSSD (ms)", "pNN6": "pNN6 (%)",
                      "CCD": "CCD (ms)", "SD1_LW": "SD1 robust (ms)",
                      "SD2_LW": "SD2 robust (ms)"}
            df_r = df_r.rename(columns={k: v for k, v in rename.items()
                                        if k in df_r.columns})
            _add("Rolling HRV Timeline", df_r, x_col="Time (s)")
            if "t_mid" in rolling_hrv_df.columns:
                _add_counts("Rolling HRV — Window Counts", rolling_hrv_df["t_mid"],
                            rolling_hrv_df, roll_counts)

        # ── 8. Epoch HRV (XY) ───────────────────────────────────────────
        if epoch_df is not None and not epoch_df.empty:
            df_e = epoch_df.copy()
            df_e.insert(0, "Time (s)",
                        ((df_e["Epoch_start_s"] + df_e["Epoch_end_s"]) / 2).round(2))
            rename_e = {"HR_mean": "HR (bpm)", "MeanNN": "MeanNN (ms)",
                        "SDNN": "SDNN (ms)", "RMSSD": "RMSSD (ms)"}
            df_e = df_e.rename(columns={k: v for k, v in rename_e.items()
                                        if k in df_e.columns})
            # Keep only numeric columns
            keep = ["Time (s)"] + [c for c in df_e.columns
                                   if c not in ("Epoch_start_s", "Epoch_end_s",
                                                "N_beats", "N_valid", "N_excluded",
                                                "Time (s)")
                                   and pd.api.types.is_numeric_dtype(df_e[c])]
            _add("Epoch HRV", df_e[keep], x_col="Time (s)")  # type: ignore[arg-type]
            _add_counts("Epoch HRV — Counts", df_e["Time (s)"], epoch_df,
                        ["N_beats", "N_valid", "N_excluded"])

        # ── 9. Autonomic CSI / CPI (XY) and event-locked pre / post (Column) ──
        sp_ok = (autonomic is not None and isinstance(autonomic_spwvd, dict)
                 and "error" not in autonomic_spwvd
                 and len(np.asarray(autonomic_spwvd.get("t", ()))) > 0)
        has_events = (autonomic is not None and event_table is not None
                      and not event_table.empty)
        if autonomic is not None and len(np.asarray(autonomic.t)):
            t_a = np.asarray(autonomic.t, dtype=float)
            step = _decimation_step(len(t_a), PRISM_MAX_SERIES_ROWS)
            _add("Autonomic — CSI / CPI", pd.DataFrame({
                "Time (s)":   t_a[::step],
                "CSI (a.u.)": np.asarray(autonomic.csi, dtype=float)[::step],
                "CPI (a.u.)": np.asarray(autonomic.cpi, dtype=float)[::step],
            }), x_col="Time (s)")
            _add("Autonomic — D / SD1 / SD2", pd.DataFrame({
                "Time (s)": t_a[::step],
                "D (ms)":   np.asarray(autonomic.d, dtype=float)[::step],
                "SD1 (ms)": np.asarray(autonomic.sd1, dtype=float)[::step],
                "SD2 (ms)": np.asarray(autonomic.sd2, dtype=float)[::step],
            }), x_col="Time (s)")
        if sp_ok:
            t_s = np.asarray(autonomic_spwvd["t"], dtype=float)            # type: ignore[index]
            step_s = _decimation_step(len(t_s), PRISM_MAX_SERIES_ROWS)
            _add("Autonomic — LF / HF (SPWVD)", pd.DataFrame({
                "Time (s)":  t_s[::step_s],
                "LF (a.u.)": np.asarray(autonomic_spwvd["lf"], dtype=float)[::step_s],  # type: ignore[index]
                "HF (a.u.)": np.asarray(autonomic_spwvd["hf"], dtype=float)[::step_s],  # type: ignore[index]
            }), x_col="Time (s)")
        if has_events:
            ev = event_table[event_table["valid"].astype(bool)] if "valid" in event_table else event_table  # type: ignore[index, operator]
            cols_ev: "dict[str, np.ndarray]" = {}
            for key, label in (("csi", "CSI"), ("cpi", "CPI"), ("hr", "HR (bpm)"),
                               ("d", "D (ms)"), ("sd1", "SD1 (ms)"), ("sd2", "SD2 (ms)"),
                               ("lf", "LF"), ("hf", "HF")):
                if f"{key}_pre" in ev.columns and f"{key}_post" in ev.columns:
                    cols_ev[f"{label} pre"] = ev[f"{key}_pre"].to_numpy(dtype=float)
                    cols_ev[f"{label} post"] = ev[f"{key}_post"].to_numpy(dtype=float)
            if cols_ev and len(ev):
                titles = [f"{(str(lab) or 'event')} @ {float(on):.1f} s"
                          for lab, on in zip(ev.get("label", [""] * len(ev)),
                                             ev.get("onset_s", np.full(len(ev), np.nan)))]
                _add("Event-locked — pre / post (valid events)",
                     pd.DataFrame(cols_ev, index=pd.Index(titles)), x_col=0, row_name=True)

        # ── 10. The shared table: the same numbers as the screen ────────
        if metrics is not None:
            df_m = metric_table_prism_frame(metrics)
            if not df_m.empty:
                _add("Metrics — value and basis n", df_m, x_col=0, row_name=True)

        if not tables:
            raise ValueError("No data available for Prism export. "
                             "Run the full analysis first.")

        # Build notes info with subject/context metadata
        note_names  = ["Notes", "Subject", "Context"]
        note_values = [f"ECG Analysis TOM v6 — {subject}", subject, context_label]
        # Same provenance rows as the Excel Cover sheet: window, analysed
        # duration, filters, RR bounds / exclusions (the HRV tables are
        # computed on the valid intervals only), artifact correction, QTc
        # formula, interval definitions / QT variability / wave template.
        for label, value in analysis_provenance_rows(
                cast(dict, results), window=hrv_window, signal_end_s=signal_end_s,
                artifact_report=artifact_report, filter_text=filter_text,
                qtc_formula=qtc_formula, signal_crop=signal_crop):
            note_names.append(label)
            note_values.append(value)
        # A withheld spectrum leaves no frequency-domain table: say why.
        freq_withheld = hrv_freq_withheld_reason(cast(dict, results))
        if freq_withheld:
            note_names.append("Frequency HRV")
            note_values.append(f"withheld — {freq_withheld}")
        # Premature beats (RR-based): counts, per-hour rates, definition.
        for label, value in premature_summary_rows(premature_summary):
            note_names.append(label.strip())
            note_values.append(value)
        # Autonomic tables exported: method, window, ks / kp, rat caveat.
        if autonomic is not None:
            sp_params = (dict(autonomic_spwvd.get("params", {}) or {})       # type: ignore[union-attr]
                         if sp_ok else None)
            for label, value in autonomic_disclosure_rows(
                    dict(autonomic.params), sp_params,
                    event_params if has_events else None, hrv_window):
                note_names.append(label)
                note_values.append(value)
        if left_out:
            note_names.append("Left out of this export")
            note_values.append(left_out)
        notes = pd.DataFrame({"Name": note_names, "Value": note_values})

        with warnings.catch_warnings():
            # pzfx notes that the RR/HR columns hold "*"-suffixed (Prism-
            # excluded) values -- intended, see table 1 above.
            warnings.filterwarnings(
                "ignore", message=r".*Values ending with '\*' will be marked as excluded",
                category=UserWarning)
            write_pzfx(
                tables,
                str(path),
                row_names=row_names,
                x_col=x_cols,
                n_digits=4,
                notes={"Project Info": notes},
            )
        return len(tables)
