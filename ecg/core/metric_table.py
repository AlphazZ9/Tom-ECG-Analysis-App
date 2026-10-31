# -*- coding: utf-8 -*-
"""
ecg.core.metric_table
---------------------
One table of metric rows over the existing results: value, unit, basis n and
state. The KPI tiles, the Report tab and its text report, the PDF rows and the
Excel Metrics sheet read their numbers from it.

It only READS what the analysis steps stored (results, frames, attrs) or calls
the same core functions the views called; it defines no metric. Values are raw
and full precision: screens format them, exports keep them. NaN means missing.

The state of a row (up to date / out of date / not computed) comes in as an
argument from the UI layer (ecg.ui.freshness), so this module stays UI-free.

Where two views read one quantity from two sources (mean HR from ``hr`` and from
``rr_df``; the HR range from ``hr`` and from pandas quantiles of ``rr_df``), both
are kept as separate rows: the sources are not the same float in every case and
a view must not change a digit by moving here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ecg.core.analysis import (
    HR_RANGE_PCTL, _NL_ENTROPY_PREFIXES, analysed_duration_s, hrv_freq_is_legacy,
    hrv_freq_withheld_reason, intervals_use_legacy_definitions, mean_hr_bpm,
    noisy_time_pct, nonlinear_subset_text, qt_beat_variability,
    rr_asymmetry_indices, rr_exclusion_report, rr_valid_mask,
    successive_valid_pairs, LEGACY_SPECTRAL_TEXT,
)
from ecg.core.models import MouseECG

MISSING = "—"

# Module names: the same keys the freshness service reports states for.
MODULES = ("core", "frequency", "nonlinear", "intervals", "epochs", "rolling",
           "autonomic", "event_locked", "events")
MODULE_LABELS = {
    "core":         "Core analysis",
    "frequency":    "Frequency HRV",
    "nonlinear":    "Non-linear HRV",
    "intervals":    "ECG intervals",
    "epochs":       "Epoch HRV",
    "rolling":      "Sliding-window HRV",
    "autonomic":    "Autonomic indices",
    "event_locked": "Event-locked",
    "events":       "Abnormal events",
}

TIER_CURATED = "curated"
TIER_ALL = "all"

# Sentinel: build_metric_table works out the RR asymmetry itself.
AUTO_ASYMMETRY: Any = object()

MIN_PAIRS_ASYMMETRY = 8     # below this no asymmetry block is shown
BEAT_CORR_THRESHOLD = 0.90  # Poor-shape-match threshold of the quality tiles

INTERVAL_COLUMNS = (("PR_ms", "PR"), ("QRS_ms", "QRS"), ("QT_ms", "QT"), ("QTc_ms", "QTc"))
_MEASURED_COLUMNS = ("PR_ms", "QRS_ms", "QT_ms")   # the "n/N beats measured" of Intervals

# column -> (label, unit, decimals); listed = curated tier, the rest = "all"
_TIME_CURATED = {
    "HRV_MeanNN": ("MeanNN", "ms", 1),
    "HRV_SDNN":   ("SDNN", "ms", 1),
    "HRV_RMSSD":  ("RMSSD", "ms", 1),
    "HRV_pNN6":   ("pNN6", "%", 1),
}
# Thresholds calibrated to human RR intervals: behind "Show all" in mice.
_TIME_EXTRA = {
    "HRV_pNN20": ("pNN20 (human)", "%", 1),
    "HRV_pNN50": ("pNN50 (human)", "%", 1),
}
_TIME_UNITS = {
    "HRV_SDANN1": "ms", "HRV_SDNNI1": "ms", "HRV_SDANN2": "ms", "HRV_SDNNI2": "ms",
    "HRV_SDANN5": "ms", "HRV_SDNNI5": "ms", "HRV_SDSD": "ms", "HRV_MedianNN": "ms",
    "HRV_MadNN": "ms", "HRV_IQRNN": "ms", "HRV_Prc20NN": "ms", "HRV_Prc80NN": "ms",
    "HRV_pNN50": "%", "HRV_MinNN": "ms", "HRV_MaxNN": "ms", "HRV_TINN": "ms",
}
_FREQ_CURATED = {
    "HRV_VLF":  ("VLF power", "ms²", 3),
    "HRV_LF":   ("LF power", "ms²", 3),
    "HRV_HF":   ("HF power", "ms²", 3),
    "HRV_LFn":  ("LF (n.u.)", "fraction of LF+HF", 3),
    "HRV_HFn":  ("HF (n.u.)", "fraction of LF+HF", 3),
    "HRV_LFHF": ("LF/HF", "ratio", 2),
}
_FREQ_UNITS = {"HRV_TP": "ms²", "HRV_LnHF": "ln ms²", "HRV_LFpeak": "Hz", "HRV_HFpeak": "Hz"}
# An older session's NeuroKit table is not in ms² / Task Force n.u.
_FREQ_UNITS_LEGACY = {
    "HRV_ULF": "a.u., legacy", "HRV_VLF": "a.u., legacy", "HRV_LF": "a.u., legacy",
    "HRV_HF": "a.u., legacy", "HRV_VHF": "a.u., legacy", "HRV_TP": "a.u., legacy",
    "HRV_LnHF": "ln a.u., legacy", "HRV_LFn": "fraction of TP, legacy",
    "HRV_HFn": "fraction of TP, legacy", "HRV_LFHF": "ratio",
}
_NONLIN_CURATED = {
    "HRV_SD1":        ("SD1", "ms", 1),
    "HRV_SD2":        ("SD2", "ms", 1),
    "HRV_SampEn":     ("SampEn", "", 2),
    "HRV_DFA_alpha1": ("DFA α1", "", 2),
    "HRV_DFA_alpha2": ("DFA α2", "", 2),
}
_NONLIN_EXTRA = {"HRV_ApEn": ("ApEn", "", 2)}
# Computed on successive valid pairs, not on single intervals.
_PAIR_COLUMNS = {"HRV_RMSSD", "HRV_SDSD", "HRV_pNN6", "HRV_pNN20", "HRV_pNN50",
                 "HRV_SD1", "HRV_SD2"}


@dataclass(frozen=True)
class MetricRow:
    """One number and what it rests on. *value* is raw (NaN = missing; a count
    stays an int); *basis_n* of *basis_total* are the items behind it (None =
    not recorded); *note* says why a value is missing or how to read it."""
    key: str
    module: str
    group: str
    label: str
    value: Any = float("nan")
    unit: str = ""
    basis_n: "Optional[int]" = None
    basis_total: "Optional[int]" = None
    basis_text: str = ""
    state: str = ""
    tier: str = TIER_CURATED
    decimals: int = 2
    note: str = ""


class MetricTable:
    """The rows, by key. Formatting mirrors the two conventions views use:
    ``text`` gives '—' for a missing or non-finite value, ``raw_text`` formats
    whatever is there (NaN prints as 'nan') and '—' only for an absent row."""

    def __init__(self, rows: "Optional[list[MetricRow]]" = None) -> None:
        self.rows: "list[MetricRow]" = list(rows or [])
        self._by_key = {r.key: r for r in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    def has(self, key: str) -> bool:
        return key in self._by_key

    def row(self, key: str) -> "Optional[MetricRow]":
        return self._by_key.get(key)

    def value(self, key: str, default: Any = float("nan")) -> Any:
        row = self._by_key.get(key)
        return row.value if row is not None else default

    def finite(self, key: str) -> "Optional[float]":
        """The value as a float, None when absent or not finite."""
        v = _as_float(self.value(key, None))
        return v if v is not None and np.isfinite(v) else None

    def text(self, key: str, decimals: "Optional[int]" = None, missing: str = MISSING) -> str:
        v = self.finite(key)
        if v is None:
            return missing
        d = decimals if decimals is not None else self._by_key[key].decimals
        return f"{v:.{d}f}"

    def raw_text(self, key: str, decimals: int, missing: str = MISSING) -> str:
        row = self._by_key.get(key)
        return f"{row.value:.{decimals}f}" if row is not None else missing

    def module_rows(self, module: str) -> "list[MetricRow]":
        return [r for r in self.rows if r.module == module]

    def to_frame(self) -> "pd.DataFrame":
        """One line per row, raw values (the Excel Metrics sheet). dtype=object
        so a count stays an int next to a missing one."""
        return pd.DataFrame(
            [{"Key": r.key, "Module": r.module, "Group": r.group, "Metric": r.label,
              "Value": r.value, "Unit": r.unit, "Basis_n": r.basis_n,
              "Basis_total": r.basis_total, "Basis": r.basis_text, "State": r.state,
              "Tier": r.tier, "Note": r.note} for r in self.rows],
            columns=["Key", "Module", "Group", "Metric", "Value", "Unit", "Basis_n",
                     "Basis_total", "Basis", "State", "Tier", "Note"],
            dtype=object)


# ── reading helpers ──────────────────────────────────────────────────────

def _as_float(v: Any) -> "Optional[float]":
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_value(df: "Optional[pd.DataFrame]", col: str) -> "Optional[float]":
    """float(df[col].values[0]); None when that cannot be read (NaN is kept)."""
    try:
        return float(df[col].values[0])   # type: ignore[index]
    except Exception:
        return None


def _nonempty(df: Any) -> bool:
    return df is not None and not getattr(df, "empty", True)


def results_rr_valid(r: dict) -> np.ndarray:
    """The RR intervals HRV used: ``rr_valid`` of the results, or the mask
    recomputed from the bounds for results saved before it existed."""
    rr = np.asarray(r.get("rr_ms", np.array([])), dtype=float)
    valid = r.get("rr_valid")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if len(valid) == len(rr):
            return valid
    hr = r.get("hr") or {}
    return rr_valid_mask(rr, float(hr.get("rr_min_ms", MouseECG.RR_MIN_MS)),
                         float(hr.get("rr_max_ms", MouseECG.RR_MAX_MS)))


def results_asymmetry(r: dict) -> "Optional[dict]":
    """rr_asymmetry_indices() of the successive valid RR pairs of *r*, under
    the rule of the views (None below MIN_PAIRS_ASYMMETRY pairs)."""
    a, b = successive_valid_pairs(np.asarray(r.get("rr_ms", np.array([])), dtype=float),
                                  results_rr_valid(r))
    return rr_asymmetry_indices(a, b) if len(a) >= MIN_PAIRS_ASYMMETRY else None


def intervals_measured(ivl: "Optional[pd.DataFrame]") -> "tuple[int, int]":
    """(beats with PR, QRS and QT all measured, beats delineated): the
    "n/N beats measured" of the Intervals status line."""
    if ivl is None or ivl.empty:
        return 0, 0
    cols = [c for c in _MEASURED_COLUMNS if c in ivl.columns]
    n_ok = int((~ivl[cols].isna().any(axis=1)).sum()) if cols else 0
    return n_ok, len(ivl)


# ── sections ─────────────────────────────────────────────────────────────

def _rr_basis(n: "Optional[int]", total: "Optional[int]") -> str:
    if n is None:
        return ""
    if total is not None and total != n:
        return f"{n} valid of {total} RR intervals"
    return f"{n} valid RR intervals"


def _hr_rows(r: dict, rep: dict) -> "list[MetricRow]":
    hr = r.get("hr") or {}
    n, tot = rep["n_valid"], rep["n_intervals"]
    txt = _rr_basis(n, tot)
    rows = []

    def add(key, label, value, unit, dec, tier=TIER_CURATED, bn=n, bt=tot, bx=txt):
        rows.append(MetricRow(key, "core", "Heart rate", label, value, unit, bn, bt, bx,
                              tier=tier, decimals=dec))

    for k, label, dec in (("mean", "HR mean (60000 / mean NN)", 1),
                          ("min", "HR min (2nd percentile)", 1),
                          ("max", "HR max (98th percentile)", 1),
                          ("std", "HR SD (instantaneous)", 1)):
        if k in hr:
            add(f"hr.{k}", label, hr[k], "bpm", dec)
    if "n" in hr:
        add("hr.n_beats", "Beats analysed", hr["n"], "beats", 0, bn=None, bt=None, bx="")
        add("hr.n_valid", "Valid RR intervals", hr.get("n_valid", hr["n"]), "intervals", 0)

    rdf = r.get("rr_df")
    if _nonempty(rdf) and "RR_ms" in rdf.columns:
        rr = rdf["RR_ms"].to_numpy(dtype=float)
        add("hr.mean_rr", "HR mean of rr_df", mean_hr_bpm(rr), "bpm", 0, TIER_ALL)
        add("rr.mean_rr", "RR mean of rr_df", float(rdf["RR_ms"].mean()), "ms", 1, TIER_ALL)
        if "HR_bpm" in rdf.columns:
            add("hr.range_lo", "HR range low (2 % quantile of rr_df)",
                rdf["HR_bpm"].quantile(0.02), "bpm", 0, TIER_ALL)
            add("hr.range_hi", "HR range high (98 % quantile of rr_df)",
                rdf["HR_bpm"].quantile(0.98), "bpm", 0, TIER_ALL)
    try:
        # Valid intervals only -- same set as HR mean and HRV_MeanNN
        add("rr.mean", "RR mean",
            float(np.nanmean(np.asarray(r["rr_ms"], dtype=float)[results_rr_valid(r)])),
            "ms", 0)
    except Exception:
        pass
    for k, label, unit, dec in (
            ("n_intervals", "RR intervals", "intervals", 0),
            ("n_excluded", "RR excluded from HRV", "intervals", 0),
            ("n_too_short", "RR excluded: too short", "intervals", 0),
            ("n_too_long", "RR excluded: too long", "intervals", 0),
            ("n_spanning_removed", "RR excluded: spanning removed beats", "intervals", 0),
            ("excluded_pct", "RR excluded", "%", 1),
            ("excluded_time_s", "RR excluded time", "s", 2),
            ("rr_min_ms", "RR lower bound", "ms", 1),
            ("rr_max_ms", "RR upper bound", "ms", 1)):
        if rep.get(k) is not None:
            add(f"rr.{k}", label, rep[k], unit, dec, TIER_ALL, bn=None, bt=None, bx="")
    return rows


def _frame_rows(df: "Optional[pd.DataFrame]", prefix: str, module: str, group: str,
                curated: dict, units: dict, basis: "Any",
                extra: "Optional[dict]" = None) -> "list[MetricRow]":
    """One row per readable column of a one-line result frame. *basis* maps a
    column to (basis_n, basis_total, basis_text); *extra* labels columns of the
    "all" tier the way *curated* does its own."""
    if not _nonempty(df):
        return []
    rows = []
    for col in df.columns:
        v = _first_value(df, str(col))
        if v is None:
            continue
        meta = curated.get(col) or (extra or {}).get(col)
        label, unit, dec = meta if meta else (str(col), "", 3)
        unit = units.get(col, unit)
        bn, bt, bx = basis(col)
        rows.append(MetricRow(f"{prefix}.{col}", module, group, label, v, unit, bn, bt, bx,
                              tier=TIER_CURATED if col in curated else TIER_ALL,
                              decimals=dec))
    return rows


def _freq_basis(col: str, attrs: dict, n: "Optional[int]", total: "Optional[int]",
                rr_text: str) -> tuple:
    """(n, total, text) of a spectral metric: a band power also says its band,
    from the attrs analyse_hrv_freq stored."""
    bands = attrs.get("bands")
    i = {"HRV_VLF": 0, "HRV_LF": 1, "HRV_HF": 2}.get(col)
    if i is not None and bands is not None and len(bands) > i:
        lo, hi = bands[i]
        return n, total, f"Welch, {float(lo):g}–{float(hi):g} Hz; {rr_text}"
    return n, total, rr_text


def _pairs_n(r: dict) -> "Optional[int]":
    try:
        a, _b = successive_valid_pairs(np.asarray(r["rr_ms"], dtype=float), results_rr_valid(r))
        return int(len(a))
    except Exception:
        return None


def _nonlinear_basis(col: str, attrs: dict, n_pairs: "Optional[int]") -> tuple:
    """(n, total, text) from the attrs analyse_hrv_nonlinear stored: an entropy
    metric rests on the entropy block, which the cap can shorten."""
    a = attrs or {}
    n_valid = a.get("n_rr_valid")
    if a.get("truncated"):       # older version: everything on the first beats
        n, tot = a.get("n_beats_used"), a.get("n_beats_total")
        return (int(n) if n is not None else None, int(tot) if tot is not None else None,
                nonlinear_subset_text(a))
    if str(col).startswith(_NL_ENTROPY_PREFIXES) and a.get("entropy_n_rr") is not None:
        n = int(a["entropy_n_rr"])
        tot = int(n_valid) if n_valid is not None else None
        return n, tot, (nonlinear_subset_text(a)
                        or f"{n} valid RR intervals "
                           f"({float(a.get('entropy_t_start_s', 0.0)):.1f}–"
                           f"{float(a.get('entropy_t_end_s', 0.0)):.1f} s)")
    if col in _PAIR_COLUMNS and n_pairs is not None:
        return n_pairs, None, f"{n_pairs} successive valid RR pairs"
    if n_valid is not None:
        return int(n_valid), None, f"{int(n_valid)} valid RR intervals"
    return None, None, ""


def interval_rows(ivl: "Optional[pd.DataFrame]") -> "list[MetricRow]":
    """Median and SD across beats per interval column, on the beats where THAT
    column was measured: PR, QRS and QT fail on different beats, so each row's
    basis_n is its own count. ``intervals.beats_measured`` is the app's "n/N
    beats measured" (all of PR, QRS and QT on one beat), repeated in every
    row's basis_text. A delineated recording only; analyse_core's RR-only
    placeholder frame gives no row."""
    if not _nonempty(ivl):
        return []
    rows = []
    n_ok, total = intervals_measured(ivl)
    both = f"{n_ok} of {total} beats measured on PR, QRS and QT together"
    legacy = "legacy definitions: re-run Intervals" if intervals_use_legacy_definitions(ivl) else ""
    if any(c in ivl.columns for c in _MEASURED_COLUMNS):
        rows.append(MetricRow("intervals.beats_measured", "intervals", "Intervals",
                              "Beats measured (PR, QRS and QT)", n_ok, "beats", n_ok, total,
                              both, decimals=0, note=legacy))
    for col, label in INTERVAL_COLUMNS:
        if col not in ivl.columns:
            continue
        d = pd.to_numeric(ivl[col], errors="coerce").dropna()
        n = int(len(d))
        text = f"median of {n} of {total} beats with {label}; {both}"
        rows.append(MetricRow(f"intervals.{col}", "intervals", "Intervals", label,
                              float(d.median()) if n else float("nan"), "ms", n, total,
                              text, decimals=1, note=legacy))
        rows.append(MetricRow(f"intervals.{col}.sd", "intervals", "Intervals",
                              f"{label} SD across beats",
                              float(d.std()) if n else float("nan"), "ms", n, total,
                              text, tier=TIER_ALL, decimals=1))
        for name, q in (("q1", 25), ("q3", 75)):
            rows.append(MetricRow(f"intervals.{col}.{name}", "intervals", "Intervals",
                                  f"{label} {name.upper()} across beats",
                                  float(np.percentile(d, q)) if n else float("nan"), "ms", n, total,
                                  text, tier=TIER_ALL, decimals=1))
    if "QT_ms" in ivl.columns:
        q = qt_beat_variability(ivl["QT_ms"].to_numpy(dtype=float))
        rows.append(MetricRow("intervals.QT_iqr", "intervals", "Intervals",
                              "QT IQR (beat-to-beat)", q["iqr_ms"], "ms", q["n"], total,
                              f"{q['n']} of {total} beats with QT", decimals=1))
    return rows


def _quality_rows(r: dict, sig_quality: "Optional[int]") -> "list[MetricRow]":
    rows = [MetricRow("quality.score", "core", "Signal quality", "Signal quality score",
                      sig_quality if sig_quality is not None else float("nan"), "%", decimals=0)]
    bc = r.get("beat_corr")
    n = len(bc) if bc is not None else 0
    if n:
        bc = np.asarray(bc, dtype=float)
        n_bad = int(np.sum(bc < BEAT_CORR_THRESHOLD))
        text = f"{n} scored beats"
        rows += [
            MetricRow("quality.mean_corr", "core", "Signal quality", "Mean beat correlation",
                      float(np.nanmean(bc)), "", n, None, text, decimals=3),
            MetricRow("quality.n_bad", "core", "Signal quality", "Beats below 0.90",
                      n_bad, "beats", n, None, text, tier=TIER_ALL, decimals=0),
            MetricRow("quality.bad_pct", "core", "Signal quality", "Beats below 0.90",
                      100.0 * n_bad / n, "%", n, None, text, decimals=1),
        ]
    rows.append(MetricRow("quality.noisy_time_pct", "core", "Signal quality",
                          "Poor-shape-match time", noisy_time_pct(r), "%", decimals=1))
    return rows


def _artifact_rows(arep: "Optional[dict]") -> "list[MetricRow]":
    if not arep:
        return []
    return [
        MetricRow("artifact.removed", "core", "Artifact correction", "Beats removed",
                  arep["n_in"] - arep["n_out"], "beats", arep["n_in"], None,
                  f"{arep['n_in']} peaks before correction", decimals=0),
        *(MetricRow(f"artifact.{k}", "core", "Artifact correction", label, arep[k], "beats",
                    tier=TIER_ALL, decimals=0)
          for k, label in (("n_duplicate", "Removed: duplicate"),
                           ("n_nonphysio", "Removed: non-physiological"),
                           ("n_ectopic", "Removed: ectopic")) if k in arep),
    ]


def _asymmetry_rows(ix: "Optional[dict]") -> "list[MetricRow]":
    if not ix:
        return []
    n = int(ix.get("n_tot", 0) or 0)
    text = f"{n} successive valid RR pairs"
    rows = []

    def add(key, label, value, unit, dec, tier):
        rows.append(MetricRow(f"asymmetry.{key}", "core", "RR asymmetry", label, value, unit,
                              n, None, text, tier=tier, decimals=dec))

    add("porta_pct", "Porta index (PI)", ix.get("porta_pct", float("nan")), "%", 1, TIER_CURATED)
    add("guzik_pct", "Guzik index (GI)", ix.get("guzik_pct", float("nan")), "%", 1, TIER_CURATED)
    add("n_dec", "Decelerations (ΔRR > 0)", int(ix.get("n_dec", 0) or 0), "pairs", 0, TIER_ALL)
    add("n_acc", "Accelerations (ΔRR < 0)", int(ix.get("n_acc", 0) or 0), "pairs", 0, TIER_ALL)
    add("n_tot", "Pairs", n, "pairs", 0, TIER_ALL)
    if "n_zero" in ix:
        add("n_zero", "No change (ΔRR = 0)", int(ix.get("n_zero", 0) or 0), "pairs", 0, TIER_ALL)
    return rows


def _event_rows(events: "Optional[list]", summary: "Optional[dict]") -> "list[MetricRow]":
    rows = []
    if events:
        kinds: "dict[str, int]" = {}
        for ev in events:
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        rows += [MetricRow(f"events.kind.{k}", "events", "Abnormal events", k, c, "events",
                           tier=TIER_ALL, decimals=0) for k, c in sorted(kinds.items())]
    if summary:
        span = summary.get("span_s")
        text = f"{float(span):.1f} s analysed" if span is not None else ""
        for k, label, unit, dec in (
                ("n_premature", "Premature beats", "beats", 0),
                ("n_isolated", "Isolated premature beats", "beats", 0),
                ("n_couplets", "Couplets", "couplets", 0),
                ("n_runs", "Runs ≥ 3", "runs", 0),
                ("n_beats_in_runs", "Beats in runs", "beats", 0),
                ("premature_per_h", "Premature beats per hour", "/h", 2),
                ("isolated_per_h", "Isolated premature beats per hour", "/h", 2),
                ("couplets_per_h", "Couplets per hour", "/h", 2),
                ("runs_per_h", "Runs per hour", "/h", 2)):
            if k in summary:
                rows.append(MetricRow(f"events.{k}", "events", "Premature beats", label,
                                      summary[k], unit, basis_text=text, decimals=dec))
    return rows


def _autonomic_rows(result: Any) -> "list[MetricRow]":
    if result is None:
        return []
    # Whole-recording values: every valid pair, not the sliding windows (those
    # are the module's status row).
    n = dict(getattr(result, "params", {}) or {}).get("n_pairs_total")
    text = f"{n} successive valid RR pairs" if n is not None else ""
    return [MetricRow(f"autonomic.{k}", "autonomic", "Autonomic", label,
                      getattr(result, attr, float("nan")), "ms", n, None, text, decimals=2)
            for k, attr, label in (("SD01_ms", "sd01", "SD01 (whole-recording SD1)"),
                                   ("SD02_ms", "sd02", "SD02 (whole-recording SD2)"),
                                   ("D0_ms", "d0", "D0 (whole-recording CCD)"))]


def _status_rows(r: dict, rep: dict, epoch_df: Any, rolling_df: Any, autonomic: Any,
                 event_params: "Optional[dict]", events: "Optional[list]",
                 premature: "Optional[dict]") -> "list[MetricRow]":
    """One row per module, whether computed or not: what it rests on and why
    its values may be missing. Time series stay frames; this holds their n."""
    basis: "dict[str, tuple]" = {}
    notes: "dict[str, str]" = {}
    if r.get("hr"):
        basis["core"] = (rep["n_valid"], rep["n_intervals"], _rr_basis(rep["n_valid"], rep["n_intervals"]))
    fd = r.get("hrv_freq")
    if _nonempty(fd):
        basis["frequency"] = (rep["n_valid"], rep["n_intervals"], _rr_basis(rep["n_valid"], rep["n_intervals"]))
        if hrv_freq_is_legacy(r):
            notes["frequency"] = LEGACY_SPECTRAL_TEXT
    elif hrv_freq_withheld_reason(r):
        notes["frequency"] = "withheld — " + str(hrv_freq_withheld_reason(r))
    nl = r.get("hrv_nonlin")
    if _nonempty(nl):
        a = nl.attrs or {}
        n = a.get("n_rr_valid")
        basis["nonlinear"] = (int(n) if n is not None else None, None,
                              f"{int(n)} valid RR intervals" if n is not None else "")
        notes["nonlinear"] = nonlinear_subset_text(a)
    ivl = r.get("intervals")
    if _nonempty(ivl) and any(c in ivl.columns for c in _MEASURED_COLUMNS):
        n_ok, total = intervals_measured(ivl)
        basis["intervals"] = (n_ok, total, f"{n_ok} of {total} beats measured on PR, QRS and QT together")
    if _nonempty(epoch_df):
        basis["epochs"] = (len(epoch_df), None, f"{len(epoch_df)} epochs")
    if _nonempty(rolling_df):
        basis["rolling"] = (len(rolling_df), None, f"{len(rolling_df)} windows")
    if autonomic is not None:
        p = dict(getattr(autonomic, "params", {}) or {})
        n = p.get("n_beats")
        basis["autonomic"] = (n, None, f"{n} windows, {p.get('n_windows_withheld', 0)} withheld")
        if p.get("withheld_reason") or p.get("withheld_note"):
            notes["autonomic"] = str(p.get("withheld_reason") or p.get("withheld_note"))
    if event_params:
        n_ev, n_ok_ev = event_params.get("n_events"), event_params.get("n_valid")
        basis["event_locked"] = (n_ok_ev, n_ev, f"{n_ok_ev} valid of {n_ev} events")
    if events or premature:
        n_ev = len(events or [])
        basis["events"] = (n_ev, None, f"{n_ev} events classified")
    rows = []
    for m in MODULES:
        bn, bt, bx = basis.get(m, (None, None, ""))
        rows.append(MetricRow(f"{m}.status", m, "Status", MODULE_LABELS[m], float("nan"), "",
                              bn, bt, bx, note=notes.get(m, ""), decimals=0))
    return rows


# ── the table ────────────────────────────────────────────────────────────

def build_metric_table(
    results:         "Optional[dict]" = None,
    states:          "Optional[Mapping[str, str]]" = None,
    sig_quality:     "Optional[int]" = None,
    artifact_report: "Optional[dict]" = None,
    window:          "Optional[tuple]" = None,
    signal_end_s:    "Optional[float]" = None,
    asymmetry:       Any = AUTO_ASYMMETRY,
    epoch_df:        "Optional[pd.DataFrame]" = None,
    rolling_df:      "Optional[pd.DataFrame]" = None,
    autonomic:       Any = None,
    event_params:    "Optional[dict]" = None,
    events:          "Optional[list]" = None,
    premature:       "Optional[dict]" = None,
) -> MetricTable:
    """Every metric row of the current results. *states* maps a module (MODULES)
    to its state; *asymmetry* is an rr_asymmetry_indices() dict, or None for no
    asymmetry rows (default: worked out from *results*). *events* / *premature*
    are the classification the caller may show: a caller passes None for a stale
    one, as the exports do."""
    r = results or {}
    rep = rr_exclusion_report(r)
    rows: "list[MetricRow]" = []
    rows += _status_rows(r, rep, epoch_df, rolling_df, autonomic, event_params, events, premature)
    if r:
        rows += _hr_rows(r, rep)
        dur = analysed_duration_s(r, window, signal_end_s)
        if dur is not None:
            rows.append(MetricRow("core.duration_s", "core", "Heart rate", "Analysed duration",
                                  dur, "s", decimals=0))
        rows += _quality_rows(r, sig_quality)
        n_pairs = _pairs_n(r)
        n_rr = rep["n_valid"]
        rr_text = _rr_basis(n_rr, rep["n_intervals"])

        def time_basis(col):
            if col in _PAIR_COLUMNS and n_pairs is not None:
                return n_pairs, None, f"{n_pairs} successive valid RR pairs"
            return n_rr, rep["n_intervals"], rr_text

        rows += _frame_rows(r.get("hrv_time"), "hrv_time", "core", "HRV time domain",
                            _TIME_CURATED, _TIME_UNITS, time_basis, _TIME_EXTRA)
        legacy = hrv_freq_is_legacy(r)
        fd_attrs = dict(getattr(r.get("hrv_freq"), "attrs", {}) or {})
        rows += _frame_rows(r.get("hrv_freq"), "hrv_freq", "frequency", "HRV frequency domain",
                            _FREQ_CURATED, _FREQ_UNITS_LEGACY if legacy else _FREQ_UNITS,
                            lambda col: _freq_basis(col, fd_attrs, n_rr, rep["n_intervals"], rr_text))
        nl = r.get("hrv_nonlin")
        attrs = dict(getattr(nl, "attrs", {}) or {})
        rows += _frame_rows(nl, "hrv_nonlin", "nonlinear", "HRV non-linear",
                            _NONLIN_CURATED, {},
                            lambda col: _nonlinear_basis(col, attrs, n_pairs), _NONLIN_EXTRA)
        rows += interval_rows(r.get("intervals"))
        ix = results_asymmetry(r) if asymmetry is AUTO_ASYMMETRY else asymmetry
        rows += _asymmetry_rows(ix)
    rows += _artifact_rows(artifact_report)
    rows += _event_rows(events, premature)
    rows += _autonomic_rows(autonomic)
    if states:
        rows = [_with_state(row, states.get(row.module, "")) for row in rows]
    return MetricTable(rows)


def _with_state(row: MetricRow, state: str) -> MetricRow:
    return MetricRow(**{**row.__dict__, "state": state})


# ── what the Report tab lists and an export can leave out ────────────────

@dataclass(frozen=True)
class ReportModule:
    """One line of the Report tab: the MODULES behind it, by state."""
    key: str
    label: str
    modules: "tuple[str, ...]"


REPORT_QUALITY = "quality"
REPORT_RHYTHM = "rhythm"
REPORT_TIME_COURSE = "time_course"
REPORT_AUTONOMIC = "autonomic"
REPORT_INTERVALS = "intervals"
REPORT_EVENTS = "events"
REPORT_MODULES = (
    ReportModule(REPORT_QUALITY, "Signal quality", ("core",)),
    ReportModule(REPORT_RHYTHM, "Rhythm & HRV", ("core", "frequency", "nonlinear")),
    ReportModule(REPORT_TIME_COURSE, "Time course", ("epochs", "rolling")),
    # An event-locked table is read from the indices: they go out together.
    ReportModule(REPORT_AUTONOMIC, "Autonomic CSI/CPI", ("autonomic", "event_locked")),
    ReportModule(REPORT_INTERVALS, "Intervals", ("intervals",)),
    ReportModule(REPORT_EVENTS, "Events", ("events",)),
)
_MODULE_REPORT = {"core": REPORT_RHYTHM, "frequency": REPORT_RHYTHM, "nonlinear": REPORT_RHYTHM,
                  "epochs": REPORT_TIME_COURSE, "rolling": REPORT_TIME_COURSE,
                  "autonomic": REPORT_AUTONOMIC, "event_locked": REPORT_AUTONOMIC,
                  "intervals": REPORT_INTERVALS, "events": REPORT_EVENTS}
# What the results hold for a Report line; ``hr`` stays, the Cover and the
# provenance read its counts.
_RESULT_KEYS = {
    REPORT_RHYTHM: ("rr_ms", "rr_valid", "rr_t_s", "rr_df", "hrv_time", "hrv_freq", "hrv_nonlin"),
    REPORT_INTERVALS: ("intervals",),
}


def report_module_of(row: MetricRow) -> "Optional[str]":
    """The Report line a row goes out with; None for what every export states
    (the artifact correction and the analysed duration: provenance)."""
    if row.key.startswith("artifact.") or row.key == "core.duration_s":
        return None
    if row.key.startswith("quality."):
        return REPORT_QUALITY
    return _MODULE_REPORT.get(row.module)


def filter_table(t: MetricTable, included: "Mapping[str, bool]") -> MetricTable:
    """The rows of the Report lines ticked in *included* (absent = ticked)."""
    return MetricTable([r for r in t.rows
                        if included.get(report_module_of(r) or "", True)])


def filter_results(results: "Optional[dict]", included: "Mapping[str, bool]") -> "Optional[dict]":
    """*results* without what the unticked Report lines hold, the same dict when
    nothing is left out: an export of everything is an export of the results."""
    if results is None:
        return None
    drop = [k for line, keys in _RESULT_KEYS.items() if not included.get(line, True) for k in keys]
    if not drop:
        return results
    return {k: v for k, v in results.items() if k not in drop}


def report_key_results(t: MetricTable, autonomic_method: str = "",
                       events_baseline: str = "") -> "dict[str, str]":
    """The Key result of each Report line, in the text of the tab it comes from;
    *autonomic_method* and *events_baseline* are what those tabs state of how
    the values were obtained. A dash when the line has no value."""
    quality = (f"score {t.text('quality.score', 0)} · corr {t.text('quality.mean_corr', 3)}"
               if t.has("quality.score") else MISSING)
    rhythm = (f"HR {metric_text(t, 'hr.mean')} bpm · SDNN {metric_text(t, 'hrv_time.HRV_SDNN')} ms · "
              f"LF/HF {metric_text(t, 'hrv_freq.HRV_LFHF')}" if t.has("hr.mean") else MISSING)

    def part(status_key: str, name: str) -> "list[str]":
        row = t.row(status_key)
        return [f"{row.basis_n} {name}"] if row is not None and row.basis_n is not None else []

    time_course = " · ".join(part("epochs.status", "epochs")
                             + part("rolling.status", "sliding windows")) or MISSING
    auto_row, ev_row = t.row("autonomic.status"), t.row("event_locked.status")
    autonomic = " · ".join(x for x in (
        autonomic_method if auto_row is not None and auto_row.basis_text else "",
        auto_row.basis_text if auto_row is not None else "",
        ev_row.basis_text if ev_row is not None else "") if x) or MISSING
    measured = t.row("intervals.beats_measured")
    intervals = MISSING
    if measured is not None:
        intervals = (" · ".join(f"{name} {t.text(f'intervals.{col}')}" for col, name in INTERVAL_COLUMNS)
                     + f" ms · {measured.value}/{measured.basis_total} beats measured")
    ev = t.row("events.status")
    events = MISSING
    if ev is not None and ev.basis_n is not None:
        events = " · ".join(x for x in (
            f"{ev.basis_n} event{'s' if ev.basis_n != 1 else ''}",
            (f"{t.value('events.n_premature')} premature beat"
             f"{'s' if t.value('events.n_premature') != 1 else ''}") if t.has("events.n_premature") else "",
            events_baseline) if x)
    return {REPORT_QUALITY: quality, REPORT_RHYTHM: rhythm, REPORT_TIME_COURSE: time_course,
            REPORT_AUTONOMIC: autonomic, REPORT_INTERVALS: intervals, REPORT_EVENTS: events}


# ── texts shared by several views ────────────────────────────────────────

def quality_tile_texts(t: MetricTable) -> "dict[str, str]":
    """The Signal Quality values shown identically by the KPI tiles and the
    Signal & Detection panel."""
    n_row = t.row("quality.n_bad")
    bad = MISSING
    if n_row is not None and n_row.basis_n:
        bad = (f"{t.value('quality.bad_pct'):.1f}%  "
               f"({n_row.value}/{n_row.basis_n})")
    noisy = t.finite("quality.noisy_time_pct")
    return {
        "sq_corr": t.text("quality.mean_corr", 3),
        "sq_badbeats": bad,
        "sq_noisy_time": f"{noisy:.1f}%" if noisy is not None else MISSING,
        "sq_artifact": artifact_text(t),
    }


def artifact_text(t: MetricTable) -> str:
    if not t.has("artifact.removed"):
        return "not applied"
    return (f"{t.value('artifact.removed')}  ({t.value('artifact.n_duplicate', 0)} dup · "
            f"{t.value('artifact.n_nonphysio', 0)} non-physio · "
            f"{t.value('artifact.n_ectopic', 0)} ectopic)")


# ── what the Rhythm & HRV page shows ─────────────────────────────────────
# One text per metric: the KPI cards, the table, the spectrum legend and the
# Poincaré title all go through format_metric, so they cannot disagree.

# key -> (scale, decimals, unit on screen). The raw value stays in the table.
_SCREEN_FORMATS = {
    "hr.mean":              (1.0, 0, "bpm"),
    "hr.min":               (1.0, 0, "bpm"),
    "hr.max":               (1.0, 0, "bpm"),
    "hrv_time.HRV_MeanNN":  (1.0, 1, "ms"),
    "hrv_time.HRV_SDNN":    (1.0, 1, "ms"),
    "hrv_time.HRV_RMSSD":   (1.0, 1, "ms"),
    "hrv_time.HRV_pNN6":    (1.0, 1, "%"),
    "hrv_freq.HRV_VLF":     (1.0, 2, "ms²"),
    "hrv_freq.HRV_LF":      (1.0, 2, "ms²"),
    "hrv_freq.HRV_HF":      (1.0, 2, "ms²"),
    "hrv_freq.HRV_LFn":     (100.0, 1, "n.u."),
    "hrv_freq.HRV_HFn":     (100.0, 1, "n.u."),
    "hrv_freq.HRV_LFHF":    (1.0, 2, ""),
    "hrv_nonlin.HRV_SD1":   (1.0, 1, "ms"),
    "hrv_nonlin.HRV_SD2":   (1.0, 1, "ms"),
    "hrv_nonlin.HRV_SD1SD2": (1.0, 2, ""),
    "hrv_nonlin.HRV_SampEn": (1.0, 2, ""),
    "hrv_nonlin.HRV_DFA_alpha1": (1.0, 2, ""),
    "hrv_nonlin.HRV_DFA_alpha2": (1.0, 2, ""),
}
_LEGACY_SCREEN_UNITS = {"a.u., legacy": "a.u.", "ln a.u., legacy": "ln a.u.",
                        "fraction of TP, legacy": "% of TP"}

# The curated rows of the page, in order, by group.
RHYTHM_GROUPS = (
    ("TIME DOMAIN", "core", ("hrv_time.HRV_MeanNN", "hrv_time.HRV_SDNN",
                             "hrv_time.HRV_RMSSD", "hrv_time.HRV_pNN6")),
    ("FREQUENCY", "frequency", ("hrv_freq.HRV_VLF", "hrv_freq.HRV_LF", "hrv_freq.HRV_HF",
                                "hrv_freq.HRV_LFn", "hrv_freq.HRV_HFn", "hrv_freq.HRV_LFHF")),
    ("NON-LINEAR", "nonlinear", ("hrv_nonlin.HRV_SD1", "hrv_nonlin.HRV_SD2",
                                 "hrv_nonlin.HRV_SampEn", "hrv_nonlin.HRV_DFA_alpha1",
                                 "hrv_nonlin.HRV_DFA_alpha2")),
)
_RHYTHM_LABELS = {f"{prefix}.{col}": meta[0] for prefix, curated in (
    ("hrv_time", _TIME_CURATED), ("hrv_freq", _FREQ_CURATED), ("hrv_nonlin", _NONLIN_CURATED))
    for col, meta in curated.items()}


@dataclass(frozen=True)
class RhythmCard:
    key: str
    name: str
    value: str
    unit: str
    note: str


@dataclass(frozen=True)
class RhythmRow:
    group: str
    key: str
    label: str
    value: str
    unit: str
    basis: str
    tier: str = TIER_CURATED


def format_metric(key: str, value: Any, row: "Optional[MetricRow]" = None,
                  missing: str = MISSING) -> str:
    """The on-screen text of *value* for metric *key*; *missing* when it is
    absent or not finite. Keys without a format use their row's decimals."""
    v = _as_float(value)
    if v is None or not np.isfinite(v):
        return missing
    fmt = _SCREEN_FORMATS.get(key)
    if fmt is not None:
        return f"{v * fmt[0]:.{fmt[1]}f}"
    return f"{v:.{row.decimals if row is not None else 3}f}"


def metric_text(t: MetricTable, key: str, missing: str = MISSING) -> str:
    return format_metric(key, t.value(key, None), t.row(key), missing)


def screen_unit(t: MetricTable, key: str) -> str:
    """The unit next to metric_text: an older session's spectrum keeps its own."""
    row = t.row(key)
    if row is not None and "legacy" in row.unit:
        return _LEGACY_SCREEN_UNITS.get(row.unit, row.unit)
    fmt = _SCREEN_FORMATS.get(key)
    if fmt is not None:
        return fmt[2]
    return row.unit if row is not None else ""


def _module_note(t: MetricTable, module: str) -> str:
    """Why a module's rows are missing: its status note, else not computed."""
    row = t.row(f"{module}.status")
    return (row.note if row is not None else "") or "not computed"


def rhythm_cards(t: MetricTable) -> "list[RhythmCard]":
    """The six KPI cards. Notes say what the code does: the HR range is the
    2nd–98th percentile of the instantaneous HR (HR_RANGE_PCTL), not its extremes."""
    def card(key, name, value_key, note):
        return RhythmCard(key, name, metric_text(t, value_key), screen_unit(t, value_key), note)

    lo, hi = HR_RANGE_PCTL
    n_valid = t.value("hr.n_valid", None)
    pairs = t.row("hrv_time.HRV_RMSSD")
    return [
        card("hr", "Heart rate", "hr.mean",
             f"{lo:g}–{hi:g} %: {metric_text(t, 'hr.min')}–{metric_text(t, 'hr.max')}"),
        card("sdnn", "SDNN", "hrv_time.HRV_SDNN",
             f"{n_valid} valid RR" if n_valid is not None else MISSING),
        card("rmssd", "RMSSD", "hrv_time.HRV_RMSSD",
             f"{pairs.basis_n} successive valid pairs"
             if pairs is not None and pairs.basis_n is not None else MISSING),
        card("pnn6", "pNN6", "hrv_time.HRV_pNN6", "mouse threshold 6 ms"),
        card("lf_hf", "LF / HF", "hrv_freq.HRV_LFHF",
             f"LF {metric_text(t, 'hrv_freq.HRV_LFn')} · "
             f"HF {metric_text(t, 'hrv_freq.HRV_HFn')} n.u."
             if t.has("hrv_freq.HRV_LFHF") else _module_note(t, "frequency")),
        card("sd1_sd2", "SD1 / SD2", "hrv_nonlin.HRV_SD1SD2",
             f"{metric_text(t, 'hrv_nonlin.HRV_SD1')} / {metric_text(t, 'hrv_nonlin.HRV_SD2')} ms"
             if t.has("hrv_nonlin.HRV_SD1") else _module_note(t, "nonlinear")),
    ]


def rhythm_rows(t: MetricTable, show_all: bool = False) -> "list[RhythmRow]":
    """The page's metric table. Every curated row is listed, with a dash and the
    module's reason when it is missing. *show_all* adds the "all" tier: the
    human pNN thresholds and the raw NeuroKit columns with a finite value."""
    rows = []
    for group, module, keys in RHYTHM_GROUPS:
        for key in keys:
            row = t.row(key)
            rows.append(RhythmRow(group, key, row.label if row else _RHYTHM_LABELS[key],
                                  metric_text(t, key), screen_unit(t, key),
                                  row.basis_text if row else _module_note(t, module)))
    if show_all:
        groups = {"hrv_time": "TIME DOMAIN", "hrv_freq": "FREQUENCY", "hrv_nonlin": "NON-LINEAR"}
        for row in t.rows:
            group = groups.get(row.key.split(".")[0])
            if group and row.tier == TIER_ALL and t.finite(row.key) is not None:
                rows.append(RhythmRow(group, row.key, row.label.replace("HRV_", ""),
                                      metric_text(t, row.key), screen_unit(t, row.key),
                                      row.basis_text, TIER_ALL))
    return rows


def nonlinear_basis_text(t: MetricTable) -> str:
    """Which beats each non-linear block used, from the rows' own basis."""
    parts = []
    for name, key in (("Entropy", "hrv_nonlin.HRV_SampEn"), ("DFA", "hrv_nonlin.HRV_DFA_alpha1"),
                      ("SD1 / SD2", "hrv_nonlin.HRV_SD1")):
        row = t.row(key)
        if row is not None and row.basis_text:
            # A capped block's disclosure is a whole sentence, named already.
            named = row.basis_text.startswith(name)
            parts.append(row.basis_text if named else f"{name}: {row.basis_text}")
    return "  ·  ".join(parts)


def asymmetry_text(t: MetricTable) -> str:
    if not t.has("asymmetry.n_tot"):
        return f"RR asymmetry: needs ≥ {MIN_PAIRS_ASYMMETRY} successive valid RR pairs"
    return (f"Porta PI {t.text('asymmetry.porta_pct')} %  ·  "
            f"Guzik GI {t.text('asymmetry.guzik_pct')} %  ·  "
            f"{t.row('asymmetry.n_tot').basis_text}")
