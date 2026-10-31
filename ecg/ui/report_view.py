# -*- coding: utf-8 -*-
"""
ecg.ui.report_view
------------------
What the Report tab says, without a window: the module lines with their state
and Include tick, the provenance block and the text report. Everything is read
from the shared table (ecg.core.metric_table) and the results; nothing here
computes a metric.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from ecg.core.analysis import (
    LEGACY_SPECTRAL_TEXT, analysis_provenance_rows, hrv_freq_is_legacy,
    hrv_freq_withheld_reason, intervals_use_legacy_definitions, nonlinear_subset_text,
)
from ecg.core.autonomic import KP_RAT_SHR, KS_RAT_SHR
from ecg.core.metric_table import (
    MISSING, REPORT_AUTONOMIC, REPORT_EVENTS, REPORT_MODULES, REPORT_QUALITY, REPORT_RHYTHM,
    REPORT_TIME_COURSE, REPORT_INTERVALS, MetricTable, build_metric_table, report_key_results,
)
from ecg.core.models import MouseECG, QTC_FORMULA_LABELS, event_kind_label, normalize_qtc_formula
from ecg.ui.context_strip import artifact_field, exclusion_field, window_text
from ecg.ui.freshness import COMPUTING, NOT_COMPUTED, UP_TO_DATE, combine_states


@dataclass(frozen=True)
class ReportLine:
    key: str
    label: str
    state: str
    result: str
    include: bool
    # Nothing to leave out or to send before it is computed.
    can_include: bool


def report_states(states: "Mapping[str, str]",
                  running: "Iterable[str]" = ()) -> "dict[str, str]":
    """Report line -> state, from the states of the modules (ecg.ui.freshness). A
    line with a module in *running* is computing, as on the tab of that module."""
    running = set(running)
    return {m.key: (COMPUTING if running.intersection(m.modules)
                    else combine_states(states.get(x, NOT_COMPUTED) for x in m.modules))
            for m in REPORT_MODULES}


def include_choices(per_line: "Mapping[str, str]",
                    ticks: "Optional[Mapping[str, Any]]" = None) -> "dict[str, bool]":
    """Line -> included. On by default only for what is up to date or still
    computing (what has landed goes out, and the export says what has not); a tick
    (see UIState.report_include) holds as long as the line stays in the state it
    was given in. Not computed is never included."""
    ticks = ticks or {}
    out = {}
    for key, state in per_line.items():
        tick = ticks.get(key)
        if state == NOT_COMPUTED:
            out[key] = False
        elif isinstance(tick, (list, tuple)) and len(tick) == 2 and tick[1] == state:
            out[key] = bool(tick[0])
        else:
            out[key] = state in (UP_TO_DATE, COMPUTING)
    return out


def left_out_labels(per_line: "Mapping[str, str]", included: "Mapping[str, bool]") -> "list[str]":
    """What was computed and is not going out, with its state."""
    return [f"{m.label} ({per_line[m.key]})" for m in REPORT_MODULES
            if per_line.get(m.key, NOT_COMPUTED) != NOT_COMPUTED and not included.get(m.key, True)]


def report_lines(t: MetricTable, per_line: "Mapping[str, str]", included: "Mapping[str, bool]",
                 reasons: "Optional[Mapping[str, tuple]]" = None, autonomic_method: str = "",
                 events_baseline: str = "") -> "list[ReportLine]":
    """The table of the tab. A line out of date says why next to its value."""
    results = report_key_results(t, autonomic_method, events_baseline)
    lines = []
    for m in REPORT_MODULES:
        state = per_line.get(m.key, NOT_COMPUTED)
        text = results[m.key]
        why = "; ".join((reasons or {}).get(m.key, ()))
        if why:
            text = why if text == MISSING else f"{text} · {why}"
        lines.append(ReportLine(m.key, m.label, state, text, bool(included.get(m.key, False)),
                                state != NOT_COMPUTED))
    return lines


def autonomic_method_text(params: "Optional[Mapping]") -> str:
    """'Robust · 3 s · ks 3 / kp 12 (rat)' of an Autonomic result; '' without one."""
    p = params or {}
    if not p:
        return ""
    parts = [str(p.get("method_label") or p.get("method") or MISSING)]
    if p.get("window_s") is not None:
        parts.append(f"{float(p['window_s']):g} s")
    ks, kp = p.get("ks"), p.get("kp")
    if ks is not None and kp is not None:
        rat = " (rat)" if ks == KS_RAT_SHR and kp == KP_RAT_SHR else ""
        parts.append(f"ks {float(ks):g} / kp {float(kp):g}{rat}")
    return " · ".join(parts)


def events_baseline_text(run: "Optional[Mapping]") -> str:
    """'baseline 509 bpm': the value of the Events tab's readout, '' without one."""
    hr = ((run or {}).get("baseline") or {}).get("baseline_hr")
    return f"baseline {hr:.0f} bpm" if hr is not None and np.isfinite(hr) else ""


def hrv_bands_text(results: "Optional[dict]") -> str:
    """The bands the spectrum on record was computed on, from its attrs."""
    fd = (results or {}).get("hrv_freq")
    bands = getattr(fd, "attrs", {}).get("bands") if fd is not None else None
    if bands is None or len(bands) < 3:
        return MISSING
    return (f"LF {float(bands[1][0]):g}–{float(bands[1][1]):g} · "
            f"HF {float(bands[2][0]):g}–{float(bands[2][1]):g} Hz")


def provenance_rows(t: MetricTable, results: "Optional[dict]", filepath: "Optional[str]",
                    subject: str, window: Any, signal_end_s: "Optional[float]",
                    filter_text: str, qtc_formula: str, autonomic_method: str,
                    current_rr: "Optional[tuple[float, float]]" = None,
                    n_peaks: "Optional[int]" = None) -> "list[tuple[str, str]]":
    """The Provenance block, in the words of the context strip where it says
    the same thing. Results dropped by a peak edit leave what the next Analyze
    would use (*current_rr*, *n_peaks*), as the strip says; only without a
    recording is there nothing to state."""
    n = t.value("hr.n_beats", None)
    current = results is None and current_rr is not None
    span = window_text(window, signal_end_s) if results is not None or current else MISSING
    lo, hi = t.finite("rr.rr_min_ms"), t.finite("rr.rr_max_ms")
    if current:
        lo, hi = current_rr
        n = n_peaks
    excluded = exclusion_field(t if results is not None else None).text
    return [
        ("File", os.path.basename(filepath or "") or MISSING),
        ("Subject", subject or MISSING),
        ("Analysis window", span + (f" · {n} beats" if n is not None else "")),
        ("Filters", filter_text if results is not None or current else MISSING),
        ("RR bounds", f"{lo:.1f}–{hi:.1f} ms" if lo is not None and hi is not None else MISSING),
        ("Excluded RR", excluded.removeprefix("RR excluded: ")),
        ("Artifact correction", artifact_field(t).text.removeprefix("Artifact correction: ")),
        ("HRV bands", hrv_bands_text(results)),
        ("QTc formula", QTC_FORMULA_LABELS[normalize_qtc_formula(qtc_formula)]),
        ("Autonomic", autonomic_method or NOT_COMPUTED),
    ]


def report_text(
    r:               dict,
    subject:         str = "",
    filepath:        "Optional[str]" = None,
    filter_text:     "Optional[str]" = None,
    window:          "Optional[tuple]" = None,
    signal_end_s:    "Optional[float]" = None,
    artifact_report: "Optional[dict]" = None,
    qtc_formula:     "Optional[str]" = "mitchell",
    asymmetry:       "Optional[dict]" = None,
    when:            "Optional[datetime]" = None,
    signal_crop:     "Optional[dict]" = None,
    lines:           "Optional[list[ReportLine]]" = None,
    included:        "Optional[Mapping[str, bool]]" = None,
    table:           "Optional[MetricTable]" = None,
    autonomic_method: str = "",
    events_baseline: str = "",
) -> str:
    """The text report of results *r* (pure: testable without the app). Every
    value carries its unit; VLF / LF / HF are absolute powers (ms²), n.u. only
    for LFn / HFn. The PROVENANCE block (analysis_provenance_rows -- same rows
    as the Excel Cover sheet and the Prism notes) states what is needed to
    reproduce the numbers: analysis window and analysed duration (window end −
    start, not the absolute time of the last beat), filters, RR bounds and
    exclusions, artifact correction, QTc formula, interval definitions and wave
    template, plus the load-time *signal_crop* (analysis.signal_crop_info) when
    recorded. A missing value is a bare '—' (no unit after it).
    *asymmetry* = rr_asymmetry_indices() of the successive valid RR pairs
    ({"porta_pct", "guzik_pct", "n_dec", "n_acc", "n_tot", "n_zero", ...}) or None.
    *lines* (report_lines) adds which modules this report holds and their state,
    also on the title of each block;
    *included* leaves out what was unticked: the caller passes *r* filtered
    (metric_table.filter_results), and the Heart rate and HRV blocks go with it.
    *table* is the shared table of everything computed (ExportController.metric_table,
    filtered the same way): it gives the Signal quality, Time course, Autonomic and
    Events blocks, so every line listed has its values below. Values print with what
    they were computed on, as on screen."""
    t   = build_metric_table(r, asymmetry=asymmetry)
    full = table if table is not None else t
    line_state = {line.key: line.state for line in lines or ()}
    nl  = r.get("hrv_nonlin")
    ivl = r.get("intervals")
    W   = 62

    def row(label: str, value: str) -> str:
        return f"    {label:<22} {value}"

    def title(text: str, line: str) -> str:
        state = line_state.get(line)
        return f"  {text}  [{state}]" if state else f"  {text}"

    def mrow(label: str, key: str, value: str) -> str:
        # What the value was computed on, the Basis of the shared table.
        rec = t.row(key)
        basis = rec.basis_text if rec is not None and value != MISSING else ""
        return row(label, f"{value}  [{basis}]" if basis else value)

    def nu(key: str) -> str:
        v = t.finite(f"hrv_freq.{key}")
        return f"{v * 100.0:.1f}" if v is not None and np.isfinite(v * 100.0) else "—"

    def time_val(key: str) -> str:
        return t.text(f"hrv_time.{key}", 3)

    def freq_val(key: str) -> str:
        return t.text(f"hrv_freq.{key}", 3)

    def nl_val(key: str) -> str:
        return t.text(f"hrv_nonlin.{key}", 3)

    def wrapped(label: str, value: str) -> "list[str]":
        # Long provenance values (definitions, template) wrapped under
        # their label rather than running off the text box.
        body = textwrap.wrap(str(value), 56) or [""]
        return [row(label, body[0])] + [f"{'':<27}{ln}" for ln in body[1:]]

    def u(v: str, unit: str, note: str = "") -> str:
        # A missing value is '—', not '— ms' (nor '— %  (note)').
        if v == "—":
            return v
        return f"{v} {unit}{note}" if unit else f"{v}{note}"

    when = when or datetime.now()
    out = [
        "═" * W,
        "  ECG ANALYSIS  —  Report",
        f"  Subject  :  {subject}",
        f"  Date     :  {when:%Y-%m-%d  %H:%M}",
        f"  File     :  {os.path.basename(filepath or '') or '—'}",
        "═" * W, "",
        "  PROVENANCE",
    ]
    for label, value in analysis_provenance_rows(
            r, window=window, signal_end_s=signal_end_s,
            artifact_report=artifact_report,
            filter_text=filter_text or "not recorded",
            qtc_formula=qtc_formula, signal_crop=signal_crop):
        out += wrapped(label, value)
    if hrv_bands_text(r) != MISSING:
        out += wrapped("HRV bands", hrv_bands_text(r))
    if autonomic_method:
        out += wrapped("Autonomic", autonomic_method)

    if lines:
        out += ["", "  WHAT IS IN THIS REPORT"]
        for line in lines:
            out.append(row(line.label, line.state + ("" if line.include or not line.can_include
                                                     else "  (left out)")))

    if full.has("quality.score") and full.finite("quality.score") is not None:
        out += ["", title("SIGNAL QUALITY", REPORT_QUALITY),
                row("Score", f"{full.text('quality.score', 0)} / 100")]
        for key, label, unit, dec in (("quality.mean_corr", "Mean beat correlation", "", 3),
                                      ("quality.bad_pct", "Beats below 0.90", "%", 1),
                                      ("quality.noisy_time_pct", "Time below 0.90", "%", 1)):
            rec = full.row(key)
            if rec is not None and full.finite(key) is not None:
                text = u(full.text(key, dec), unit)
                out.append(row(label, f"{text}  [{rec.basis_text}]" if rec.basis_text else text))

    if (included or {}).get(REPORT_RHYTHM, True):
        out += [
            "", title("HEART RATE", REPORT_RHYTHM),
            mrow("Mean (60000/mean NN)", "hr.mean", u(t.text("hr.mean", 1), "bpm")),
            # instantaneous_hr_stats: 2nd / 98th percentile, sample SD
            row("Min (2nd %ile)", u(t.text("hr.min", 1), "bpm")),
            row("Max (98th %ile)", u(t.text("hr.max", 1), "bpm")),
            row("SD (instantaneous)", u(t.text("hr.std", 1), "bpm")),
            row("Beats (n)", str(t.value("hr.n_beats", "—"))),
            "",
            "  HRV — TIME DOMAIN",
            mrow("MeanNN", "hrv_time.HRV_MeanNN", u(time_val('HRV_MeanNN'), "ms")),
            mrow("SDNN", "hrv_time.HRV_SDNN", u(time_val('HRV_SDNN'), "ms")),
            mrow("RMSSD", "hrv_time.HRV_RMSSD", u(time_val('HRV_RMSSD'), "ms")),
            mrow("pNN6", "hrv_time.HRV_pNN6", u(time_val('HRV_pNN6'), "%",
                                                f"  (ΔRR > {MouseECG.PNN_THRESHOLD} ms)")),
            mrow("pNN20", "hrv_time.HRV_pNN20", u(time_val('HRV_pNN20'), "%", "  (human threshold)")),
            "",
            "  HRV — FREQUENCY DOMAIN",
        ]
        freq_withheld = hrv_freq_withheld_reason(r)
        if hrv_freq_is_legacy(r):
            # An older session's nk.hrv_frequency table: arbitrary units,
            # LFn = LF / TP -- not ms² / Task Force n.u.
            out += [
                row("VLF power", u(freq_val('HRV_VLF'), "a.u.")),
                row("LF power", u(freq_val('HRV_LF'), "a.u.",
                                  f"  ({u(nu('HRV_LFn'), '%')} of TP)")),
                row("HF power", u(freq_val('HRV_HF'), "a.u.",
                                  f"  ({u(nu('HRV_HFn'), '%')} of TP)")),
                row("LF/HF", u(freq_val('HRV_LFHF'), "", "  (ratio)")),
                *wrapped("Note", LEGACY_SPECTRAL_TEXT),
            ]
        elif not freq_withheld:
            # Absolute band powers (ms²) of the spectrum drawn on the PSD
            # panel; n.u. = LF or HF / (LF + HF) only.
            out += [
                mrow("VLF power", "hrv_freq.HRV_VLF", u(freq_val('HRV_VLF'), "ms²")),
                mrow("LF power", "hrv_freq.HRV_LF", u(freq_val('HRV_LF'), "ms²")),
                mrow("HF power", "hrv_freq.HRV_HF", u(freq_val('HRV_HF'), "ms²")),
                row("LF (n.u.)", u(nu('HRV_LFn'), "%", "  (LF / (LF + HF))")),
                row("HF (n.u.)", u(nu('HRV_HFn'), "%", "  (HF / (LF + HF))")),
                row("LF/HF", u(freq_val('HRV_LFHF'), "", "  (ratio)")),
            ]
        else:
            # A withheld spectrum says why rather than a column of "—".
            out += wrapped("Withheld", freq_withheld)

        nl_subset = nonlinear_subset_text(getattr(nl, "attrs", {}) if nl is not None else {})
        out += [
            "",
            "  HRV — NON-LINEAR",
            mrow("SD1", "hrv_nonlin.HRV_SD1", u(nl_val('HRV_SD1'), "ms")),
            mrow("SD2", "hrv_nonlin.HRV_SD2", u(nl_val('HRV_SD2'), "ms")),
            mrow("SampEn", "hrv_nonlin.HRV_SampEn", u(nl_val('HRV_SampEn'), "", "  (dimensionless)")),
            mrow("ApEn", "hrv_nonlin.HRV_ApEn", u(nl_val('HRV_ApEn'), "", "  (dimensionless)")),
            mrow("DFA α1", "hrv_nonlin.HRV_DFA_alpha1", u(nl_val('HRV_DFA_alpha1'), "", "  (dimensionless)")),
            mrow("DFA α2", "hrv_nonlin.HRV_DFA_alpha2", u(nl_val('HRV_DFA_alpha2'), "", "  (dimensionless)")),
        ]
        if nl_subset:
            out += wrapped("Note", nl_subset)

    if ivl is not None and not ivl.empty and "QT_ms" in ivl.columns:
        # Definitions, QT variability and template are in PROVENANCE.
        out += ["", title("ECG INTERVALS  (median ± SD across beats)", REPORT_INTERVALS)]
        qtc_short = QTC_FORMULA_LABELS[normalize_qtc_formula(qtc_formula or "mitchell")]
        for col, label in [("PR_ms", "PR"), ("QRS_ms", "QRS"), ("QT_ms", "QT"),
                           ("QTc_ms", f"QTc ({qtc_short.split(' ')[0]})")]:
            med = t.row(f"intervals.{col}")
            if med is not None and med.basis_n:
                out.append(row(label, f"{med.value:.1f} ± "
                                      f"{t.value(f'intervals.{col}.sd'):.1f} ms  "
                                      f"[{med.basis_n}/{med.basis_total} beats]"))
        if intervals_use_legacy_definitions(ivl):
            # The values above are not the standard PR / QRS / QT: say so
            # here, not only in PROVENANCE.
            out += wrapped("Note", "legacy definitions (older version: PR to "
                                   "Q nadir, QRS Q→S nadir, QT from R) — re-run "
                                   "Intervals for the standard ones")

    if t.has("asymmetry.n_tot"):
        n_tot = t.value("asymmetry.n_tot")
        out += [
            "", "  RR ASYMMETRY  (successive valid ΔRR)",
            # Literature conventions -- rr_asymmetry_indices.
            *wrapped("Porta index (PI)", u(t.text("asymmetry.porta_pct", 1), "%",
                                           "  (Porta 2008: points below the "
                                           "identity line, ΔRR < 0, / off-line "
                                           "points; ΔRR = 0 excluded; "
                                           "symmetry → 50 %)")),
            *wrapped("Guzik index (GI)", u(t.text("asymmetry.guzik_pct", 1), "%",
                                           "  (Guzik 2006: Σ distance of points "
                                           "above the identity line, ΔRR > 0, / "
                                           "Σ distance of off-line points; "
                                           "symmetry → 50 %)")),
            row("Decelerations", f"{t.value('asymmetry.n_dec')} / {n_tot}  (ΔRR > 0)"),
            row("Accelerations", f"{t.value('asymmetry.n_acc')} / {n_tot}  (ΔRR < 0)"),
        ]
        if t.has("asymmetry.n_zero"):
            out.append(row("No change", f"{t.value('asymmetry.n_zero')} / "
                                        f"{n_tot}  (ΔRR = 0, not in PI / GI)"))

    def status_text(key: str) -> str:
        rec = full.row(key)
        return rec.basis_text if rec is not None and rec.basis_n is not None else ""

    epochs, sliding = status_text("epochs.status"), status_text("rolling.status")
    if epochs or sliding:
        out += ["", title("TIME COURSE", REPORT_TIME_COURSE)]
        if epochs:
            out.append(row("Epochs", epochs))
        if sliding:
            out.append(row("Sliding", sliding))
        out.append(row("Values", "per window: Excel and Prism tables"))

    windows, locked = status_text("autonomic.status"), status_text("event_locked.status")
    if windows:
        out += ["", title("AUTONOMIC CSI / CPI", REPORT_AUTONOMIC)]
        if autonomic_method:
            out.append(row("Method", autonomic_method))
        out.append(row("Windows", windows))
        for key, label in (("autonomic.SD01_ms", "SD01"), ("autonomic.SD02_ms", "SD02"),
                           ("autonomic.D0_ms", "D0")):
            rec = full.row(key)
            if rec is not None and full.finite(key) is not None:
                out.append(row(label, f"{full.text(key)} ms  [{rec.basis_text}]"))
        if locked:
            out.append(row("Event-locked", locked))
        out.append(row("Values", "per window and per event: Excel and Prism tables"))

    classified = full.row("events.status")
    if classified is not None and classified.basis_n is not None:
        out += ["", title("EVENTS", REPORT_EVENTS), row("Classified", classified.basis_text)]
        if events_baseline:
            out.append(row("Baseline", events_baseline.removeprefix("baseline ")))
        for rec in full.module_rows("events"):
            if rec.key.startswith("events.kind."):
                out.append(row(event_kind_label(rec.label), f"{rec.value}"))
            elif rec.key != "events.status" and full.finite(rec.key) is not None:
                value = full.text(rec.key, rec.decimals)
                out.append(row(rec.label[:22], f"{value} {rec.unit}".rstrip()))
    out += ["", "═" * W]
    return "\n".join(out)
