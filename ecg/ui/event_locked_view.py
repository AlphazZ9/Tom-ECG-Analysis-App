# -*- coding: utf-8 -*-
"""
event_locked_view.py
--------------------
Pure helpers of the HRV > Autonomic > Event-locked panel and of the ks / kp
calibration dialog -- the statistical protocol of Candia-Rivera et al.,
J Physiol 2025 (core.event_locked) applied to the app's Autonomic result.

Event-locked panel
    events come from the time annotations (onset = the annotation's start,
    optionally one label only); every time-resolved series of the Autonomic
    result -- the CSI family (D, SD1, SD2, CSI, CPI), HR and, when computed,
    the SPWVD LF / HF -- is condensed into its mean over the window before
    and after each onset (30 s / 30 s by default), and cut around the
    onsets for the median +- MAD trace plot (raw, or z-scored per
    recording for display). The per-event table (text + Excel frame), a
    within-recording Wilcoxon (descriptive only: the events of ONE animal
    are not a population sample) and the rows written to the cohort
    registry (ecg.io.db.save_event_rows) are built here.

Calibration dialog
    the registry rows flattened into one event table, CSI / CPI re-composed
    for any ks / kp (they are linear in the gains, so the stored per-event
    window means are enough), the per-event or per-subject-mean cohort
    table, the Wilcoxon Z(k) curves for k = 1..20 with their knee, and the
    paired tests at the current gains.

No Tk widget access: EventLockedController and CalibrationDialog read the
widgets and call these, so all of it is unit-testable on plain data and a
bare matplotlib Figure. Theme colours are read from ``ecg.ui.theme`` at call
time.
"""
from __future__ import annotations

import logging
import math
import re
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ecg.core.autonomic import KP_RAT_SHR, KS_RAT_SHR, AutonomicResult
from ecg.core.event_locked import (
    aggregate, calibrate_gain, event_locked_traces, event_window_means,
    group_summary, knee_point, paired_tests, sign_stretches,
)
from ecg.ui import theme as _theme
from ecg.ui.autonomic_view import (
    _band_text, _fmt, _to_bool, _to_float, is_rat_calibration, spwvd_ok,
)
from ecg.ui.plots import style_axes

log = logging.getLogger("ecg")


# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════

ALL_LABELS = "All"
UNLABELLED = "(unlabelled)"

# Series of the per-event table, in display order.
EVENT_KEYS = ("hr", "d", "sd1", "sd2", "csi", "cpi", "lf", "hf")
# The CSI family. D, SD1, SD2 and the indices built from them share one NaN
# mask (a withheld Poincaré window blanks all of them), so an event's
# validity -- the table's ``valid`` column, the registry's valid flag -- is
# theirs. HR (its own valid-RR spline) and LF / HF (their own time axis)
# carry their own ``k_valid`` flag and never cost an event its CSI / CPI.
POINCARE_KEYS = ("d", "sd1", "sd2", "csi", "cpi")
SPWVD_KEYS = ("lf", "hf")

KEY_LABELS = {"hr": "HR", "d": "D (CCD)", "sd1": "SD1", "sd2": "SD2",
              "csi": "CSI", "cpi": "CPI", "lf": "LF", "hf": "HF"}
KEY_UNITS = {"hr": "bpm", "d": "ms", "sd1": "ms", "sd2": "ms",
             "csi": "a.u.", "cpi": "a.u.", "lf": "a.u.", "hf": "a.u."}

DEFAULT_PRE_S = 30.0          # the paper's condition windows
DEFAULT_POST_S = 30.0
PRE_POST_MIN_S = 1.0
PRE_POST_MAX_S = 600.0
MIN_COVERAGE = 0.8            # core.event_locked default
MAX_TRACES_DRAWN = 30         # individual event traces drawn behind the median

K_VALUES = tuple(range(1, 21))           # the paper's sweep
AGGREGATIONS: "OrderedDict[str, str]" = OrderedDict([
    ("event", "Per event"),
    ("subject", "Per subject mean"),
])
DEFAULT_AGGREGATION = "subject"

REFERENCE = "Candia-Rivera et al., J Physiol 2025 (doi:10.1113/JP288400)"

WITHIN_RECORDING_NOTE = (
    "Within-recording statistics: the events of ONE animal are repeated "
    "measures of that animal, not independent samples of a population: "
    "the Wilcoxon p-values below describe this recording only and are not "
    "population inference. Save the events to the cohort registry and test "
    "across animals (Calibrate ks/kp... > Per subject mean).")

VALIDITY_NOTE = (
    "An event is valid when no other annotation starts inside its pre or "
    "post window or at its onset (the paper's minimum duration per "
    "condition: successive doses closer than the windows would contaminate "
    "each other's baseline and response; every annotation counts, also "
    "those the label filter leaves out), both of its windows lie inside "
    "the CSI / CPI series and at least 80 % of each is covered by computed "
    "(not withheld) values; HR and LF / HF are judged on their own coverage "
    "(an invalid series is blank for that event and never costs it its "
    "CSI / CPI).")

MIXED_LABELS_NOTE = (
    "Events of several labels are pooled in this test ({labels}): each label "
    "is its own condition contrast; choose one label to test it alone.")

ZSCORE_NOTE = (
    "z-scoring (by the mean and SD of each whole series of this recording) "
    "is for display only; the table, the tests and the registry use the "
    "raw window means.")

CALIBRATION_RULE_NOTE = (
    "The paper sets each gain at the inflection point of the paired "
    "Wilcoxon Z(k) curve (pre vs post, k = 1..20), from which Z plateaus "
    "(rat SHR, rest vs dobutamine: ks = 3, kp = 12 — not validated in "
    "mice). CSI and CPI are linear in the gains: dCSI = -dD + ks*dSD2 and "
    "dCPI = dD + kp*dSD1, so the stored per-event deltas of D, SD1 and SD2 "
    "give Z(k) for any k without recomputing a series. The knee marked here "
    "is an objective proxy for the inflection point (Kneedle on Z, after "
    "its last sign change) — check it against the curve. Z is a rank "
    "statistic: it moves in discrete steps (a unit flipping sign or two "
    "units swapping rank) and cannot exceed a ceiling set by the number of "
    "units (2.20 for 6, 2.52 for 8, 3.06 for 12) — a small cohort often "
    "sits on that ceiling, where Z cannot rank the gains; a knee resting on "
    "fewer than 3 rank steps is not offered.")

PSEUDO_REPLICATION_NOTE = (
    "Per subject mean averages each animal's events first, so an animal "
    "with several events counts once: testing events as independent "
    "samples is pseudo-replication. Events without a subject ID are "
    "grouped per recording. The paper computed Z over trials (events: 41 "
    "SHR trials), so Per event reproduces its procedure; Per subject mean "
    "is this app's stricter default.")

IN_SAMPLE_NOTE = (
    "Gains chosen on these events and tested on the same events give "
    "optimistic p-values: picking k among 20 values is a selection the "
    "×2 Bonferroni does not correct. The paper fitted ks / kp on SHR rats "
    "and validated them on a separate WKY cohort — calibrate on one project "
    "or label, Apply, then select another project or label to test at those "
    "gains.")

SINGLE_CONTRAST_NOTE = (
    "Events of several labels are pooled ({labels}): the per-subject mean "
    "then averages different condition changes into one pre / post pair "
    "and Z(k) describes a mixed contrast. The paper calibrated on one "
    "contrast (rest vs dobutamine) — choose one label.")

DEFAULT_SUBJECT_NOTE = (
    "{n} event(s) of {r} recordings share the default-looking Subject ID "
    "{ids}: if these are different animals, the per-subject mean pools "
    "them as one — set each animal's Subject ID in the sidebar and save "
    "its events again.")

# Fewest rank steps (W+ change) a Z(k) stretch must move by for its knee to
# be offered (core.knee_point min_rank_steps).
MIN_RANK_STEPS = 3

# The sidebar's untouched Subject ID ("subject_01") and the like.
_DEFAULT_SUBJECT_RE = re.compile(r"^subject[\s_-]*\d*$", re.IGNORECASE)

EVENT_PARAM_KEYS = ("label", "pre_s", "post_s", "zscore")


# ════════════════════════════════════════════════════════════
#  PARAMETERS
# ════════════════════════════════════════════════════════════

def default_event_params() -> dict:
    """The panel's defaults: every annotation, the paper's 30 s / 30 s
    windows, z-scored display."""
    return {"label": ALL_LABELS, "pre_s": DEFAULT_PRE_S,
            "post_s": DEFAULT_POST_S, "zscore": True}


def _window_len(value: Any) -> float:
    v = _to_float(value)
    if not (PRE_POST_MIN_S <= v <= PRE_POST_MAX_S):
        raise ValueError(f"{v:g} s outside {PRE_POST_MIN_S:g}–{PRE_POST_MAX_S:g} s")
    return v


def _label_value(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"not a label: {value!r}")
    return value.strip() or ALL_LABELS


_VALIDATORS = {"label": _label_value, "pre_s": _window_len,
               "post_s": _window_len, "zscore": _to_bool}


def normalise_event_params(raw: Any, base: "Optional[Mapping]" = None) -> dict:
    """Tolerant reading of stored panel parameters (a session file, app
    state): every valid key of *raw* is taken; a missing or invalid key --
    and *raw* itself when it is not a mapping -- keeps *base*'s value
    (default: ``default_event_params()``). Never raises."""
    out = (default_event_params() if base is None
           else normalise_event_params(base))
    if not isinstance(raw, Mapping):
        return out
    for key in EVENT_PARAM_KEYS:
        if key not in raw:
            continue
        try:
            out[key] = _VALIDATORS[key](raw[key])
        except (TypeError, ValueError, OverflowError) as exc:
            log.debug("event-locked params: ignoring %s=%r (%s)", key, raw[key], exc)
    return out


def parse_event_inputs(label: Any, pre: Any, post: Any, zscore: Any = True) -> dict:
    """Strict reading of the panel's entries for a Compute click. Raises
    ValueError with a message for the status bar; the label and the z-score
    flag are read tolerantly (display / filter choices)."""
    out = {}
    for name, value in (("Pre", pre), ("Post", post)):
        try:
            out[name] = _window_len(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"{name} must be a number of seconds from {PRE_POST_MIN_S:g} to "
                f"{PRE_POST_MAX_S:g} (got {value!r}).") from None
    try:
        lab = _label_value(label)
    except ValueError:
        lab = ALL_LABELS
    try:
        z = _to_bool(zscore)
    except (TypeError, ValueError):
        z = True
    return {"label": lab, "pre_s": out["Pre"], "post_s": out["Post"], "zscore": z}


# ════════════════════════════════════════════════════════════
#  EVENTS FROM ANNOTATIONS
# ════════════════════════════════════════════════════════════

def _annotation_events(annotations: "Iterable[Mapping]") -> "list[tuple[float, str]]":
    """(t_start, label) of every well-formed annotation (finite start)."""
    out = []
    for ann in annotations or ():
        if not isinstance(ann, Mapping):
            continue
        try:
            t0 = float(ann["t_start"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(t0):
            continue
        out.append((t0, str(ann.get("label", "") or "").strip()))
    return out


def annotation_label_options(annotations: "Iterable[Mapping]") -> "list[str]":
    """Values of the label filter: "All", the distinct non-empty annotation
    labels (sorted case-insensitively), then "(unlabelled)" when some
    annotation has no label."""
    events = _annotation_events(annotations)
    labels = sorted({lab for _, lab in events if lab}, key=lambda s: (s.casefold(), s))
    out = [ALL_LABELS] + [lab for lab in labels if lab != ALL_LABELS]
    if any(not lab for _, lab in events):
        out.append(UNLABELLED)
    return out


def select_events(annotations: "Iterable[Mapping]",
                  label: str = ALL_LABELS) -> "list[tuple[float, str]]":
    """The events of the label filter -- (onset_s, label) per annotation,
    onset = the annotation's start -- sorted by onset, duplicates (same
    onset and label) once. *label* "All" takes every annotation,
    "(unlabelled)" those without a label, anything else an exact match."""
    lab_f = (label or ALL_LABELS).strip() or ALL_LABELS
    out = set()
    for t0, lab in _annotation_events(annotations):
        if (lab_f == ALL_LABELS or (lab_f == UNLABELLED and not lab)
                or lab == lab_f):
            out.add((t0, lab))
    return sorted(out)


def other_annotation_onsets(annotations: "Iterable[Mapping]",
                            events: "Iterable[tuple[float, str]]") -> "list[float]":
    """Starts of the annotations that are not among *events* (the
    ``select_events()`` list): the condition changes a label-filtered run
    leaves out but that still bound its events' windows. An annotation
    identical to an event (same start and label -- a duplicate) is the
    event itself; one at the same start with another label is a
    coincident change and is listed."""
    chosen = set(events_signature(events))
    out = []
    for t0, lab in sorted(set(_annotation_events(annotations))):
        if (round(float(t0), 6), str(lab)) not in chosen:
            out.append(float(t0))
    return out


def events_signature(events: "Iterable[tuple[float, str]]") -> "tuple":
    """Comparable identity of an event list (onsets to the µs)."""
    return tuple((round(float(t), 6), str(lab)) for t, lab in events)


# ════════════════════════════════════════════════════════════
#  EVENT-LOCKED COMPUTATION (one recording)
# ════════════════════════════════════════════════════════════

@dataclass(eq=False)
class EventLockedResult:
    """Event-locked view of one recording's Autonomic result.

    table    : core.event_window_means() table -- label, onset_s, valid
               (the CSI family's joint validity, isolation included),
               coverage_pre / coverage_post (the CSI family's), any_valid,
               isolated, prev_gap_s, next_gap_s, then per key k_pre,
               k_post, k_delta, k_coverage_pre, k_coverage_post, k_valid
               (values NaN where k is invalid)
    t_rel    : relative time of the traces, s (-pre .. +post)
    traces   : {key: (n_events, len(t_rel))} raw units
    traces_z : the same, z-scored by the whole series of this recording
    keys     : series present, in EVENT_KEYS order
    events   : the (onset_s, label) list the table was built from
    params   : label filter, windows, coverage, the Autonomic configuration
    other_onsets : the other condition changes (annotation starts outside
               the label filter) the events' isolation was judged against
    """
    table:    pd.DataFrame
    t_rel:    np.ndarray
    traces:   "dict[str, np.ndarray]"
    traces_z: "dict[str, np.ndarray]"
    keys:     "tuple[str, ...]"
    events:   "list[tuple[float, str]]"
    params:   dict = field(default_factory=dict)
    other_onsets: "list[float]" = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return int(len(self.table))

    @property
    def n_valid(self) -> int:
        return int(self.table["valid"].sum()) if len(self.table) else 0

    @property
    def has_lfhf(self) -> bool:
        return "lf" in self.keys


_EVENT_COLS = ("label", "onset_s", "valid", "coverage_pre", "coverage_post", "any_valid",
               "isolated", "prev_gap_s", "next_gap_s")
_KEY_SUFFIXES = ("_pre", "_post", "_delta", "_coverage_pre", "_coverage_post", "_valid")


def _merge_tables(base: pd.DataFrame, extra: pd.DataFrame,
                  keys: "Sequence[str]") -> pd.DataFrame:
    """*base* plus *extra*'s per-key columns (same events, same order); an
    event is ``any_valid`` when a series of either is valid. The event-level
    ``valid`` / coverage stay *base*'s."""
    out = base.copy()
    for key in keys:
        for suffix in _KEY_SUFFIXES:
            out[f"{key}{suffix}"] = extra[f"{key}{suffix}"].to_numpy()
    out["any_valid"] = (out["any_valid"].to_numpy(dtype=bool)
                        | extra["any_valid"].to_numpy(dtype=bool))
    return out


def compute_event_locked(res: AutonomicResult,
                         events: "Sequence[tuple[float, str]]",
                         pre_s: float = DEFAULT_PRE_S,
                         post_s: float = DEFAULT_POST_S,
                         spwvd: "Optional[Mapping]" = None,
                         label_filter: str = ALL_LABELS,
                         min_coverage: float = MIN_COVERAGE,
                         other_onsets: "Iterable[float]" = ()) -> EventLockedResult:
    """Pre / post window means and event-locked traces of every series of
    *res* (and of *spwvd*'s LF / HF when it is a computed result) around
    the *events* ((onset_s, label) pairs, e.g. ``select_events()``).

    The CSI family (D, SD1, SD2, CSI, CPI -- re-centred and resampled, as
    plotted) defines each event's ``valid`` flag; HR and LF / HF are judged
    separately (``k_valid``), LF / HF on their own time axis. Every series
    of an event that is not ISOLATED -- another condition change (another
    event, or an *other_onsets* change: the annotations left out by the
    label filter, ``other_annotation_onsets()``) inside its pre or post
    window, or at the same onset -- is invalid (core.event_window_means,
    the paper's minimum 30 s per condition). Traces are on one relative
    grid at the result's ``fs_out`` (LF / HF interpolated onto it without
    bridging gaps). Raises ValueError for no event or a result without at
    least two resampled samples."""
    events = [(float(t), str(lab)) for t, lab in events]
    if not events:
        raise ValueError("No event to lock to.")
    others = np.array([float(o) for o in (other_onsets or ())], dtype=np.float64)
    t = np.asarray(res.t, dtype=np.float64)
    if len(t) < 2:
        reason = res.params.get("withheld_reason") or "no window could be computed"
        raise ValueError(f"The Autonomic result has no CSI / CPI series ({reason}).")
    pre_s, post_s = float(pre_s), float(post_s)
    onsets = np.array([o for o, _ in events], dtype=np.float64)
    labels = [lab for _, lab in events]

    iso = {"other_onsets": others}
    poincare = {"d": res.d, "sd1": res.sd1, "sd2": res.sd2, "csi": res.csi, "cpi": res.cpi}
    table = event_window_means(t, poincare, onsets, labels, pre_s, post_s, min_coverage, **iso)
    hr = {"hr": res.hr_bpm}
    table = _merge_tables(table, event_window_means(t, hr, onsets, labels, pre_s,
                                                    post_s, min_coverage, **iso), ("hr",))
    p = res.params
    fs_out = float(p.get("fs_out", 20.0) or 20.0)
    raw_series = {**hr, **poincare}
    _t_rel, traces = event_locked_traces(t, raw_series, onsets, pre_s, post_s,
                                         fs_out, zscore=False)
    t_rel, traces_z = event_locked_traces(t, raw_series, onsets, pre_s, post_s,
                                          fs_out, zscore=True)
    keys = ["hr", *POINCARE_KEYS]

    sp_params: dict = {}
    if spwvd_ok(spwvd):
        ts = np.asarray(spwvd["t"], dtype=np.float64)              # type: ignore[index]
        if len(ts) >= 2:
            lfhf = {"lf": spwvd["lf"], "hf": spwvd["hf"]}          # type: ignore[index]
            table = _merge_tables(table, event_window_means(
                ts, lfhf, onsets, labels, pre_s, post_s, min_coverage, **iso), SPWVD_KEYS)
            _, tr = event_locked_traces(ts, lfhf, onsets, pre_s, post_s, fs_out, zscore=False)
            _, tr_z = event_locked_traces(ts, lfhf, onsets, pre_s, post_s, fs_out, zscore=True)
            traces.update(tr)
            traces_z.update(tr_z)
            keys += list(SPWVD_KEYS)
            sp = spwvd.get("params", {}) or {}                       # type: ignore[union-attr]
            sp_params = {"lf_band": list(sp.get("lf_band", ()) or ()),
                         "hf_band": list(sp.get("hf_band", ()) or ()),
                         "spwvd_fs_hz": sp.get("fs_resample"),
                         "spwvd_fbins": sp.get("fbins")}

    ordered = [k for k in EVENT_KEYS if k in keys]
    cols = list(_EVENT_COLS) + [f"{k}{s}" for k in ordered for s in _KEY_SUFFIXES]
    table = table[cols].reset_index(drop=True)
    table.attrs.update(pre_s=pre_s, post_s=post_s, min_coverage=float(min_coverage))

    params = {
        "label_filter": (label_filter or ALL_LABELS),
        "pre_s": pre_s, "post_s": post_s,
        "min_coverage": float(min_coverage), "fs_out": fs_out,
        "n_events": int(len(table)), "n_valid": int(table["valid"].sum()),
        "n_not_isolated": int((~table["isolated"].to_numpy(dtype=bool)).sum()),
        "n_other_onsets": int(np.isfinite(others).sum()),
        "require_isolated": True,
        "method": p.get("method"), "method_label": p.get("method_label"),
        "window_s": p.get("window_s"), "ks": p.get("ks"), "kp": p.get("kp"),
        "min_pairs": p.get("min_pairs"),
        "rr_min_ms": p.get("rr_min_ms"), "rr_max_ms": p.get("rr_max_ms"),
        "n_breaks": p.get("n_breaks"),
        "lf_hf": bool(sp_params),
        **sp_params,
        "reference": REFERENCE,
    }
    return EventLockedResult(table=table, t_rel=t_rel, traces=traces,
                             traces_z=traces_z, keys=tuple(ordered),
                             events=list(events), params=params,
                             other_onsets=[float(o) for o in others if np.isfinite(o)])


def group_stats(el: EventLockedResult, key: str, zscore: bool = True
                ) -> "tuple[np.ndarray, np.ndarray, int, np.ndarray]":
    """(median, MAD, n, traces) across the events valid for *key*, per
    relative time sample; the traces are the (n, len(t_rel)) rows used."""
    tr = (el.traces_z if zscore else el.traces)[key]
    ok = el.table[f"{key}_valid"].to_numpy(dtype=bool)
    rows = np.asarray(tr, dtype=np.float64)[ok]
    if len(rows) == 0:
        nan = np.full(len(el.t_rel), np.nan)
        return nan, nan.copy(), 0, rows
    med, mad = group_summary({key: rows})[key]
    return med, mad, int(len(rows)), rows


# ════════════════════════════════════════════════════════════
#  TABLES / TEXT / STATUS (one recording)
# ════════════════════════════════════════════════════════════

def event_table_frame(el: EventLockedResult) -> pd.DataFrame:
    """The per-event table as exported (Copy for Excel, the EventLocked
    sheet): label, onset_s, valid, isolated (no other annotation inside the
    windows), prev_gap_s / next_gap_s (to the nearest other annotation),
    coverage_pre, coverage_post, then k_pre, k_post, k_delta for every
    series present."""
    cols = ["label", "onset_s", "valid", "isolated", "prev_gap_s", "next_gap_s",
            "coverage_pre", "coverage_post"]
    cols += [f"{k}{s}" for k in el.keys for s in ("_pre", "_post", "_delta")]
    out = el.table[cols].copy()
    out["valid"] = out["valid"].astype(bool)
    out["isolated"] = out["isolated"].astype(bool)
    return out


def within_recording_tests(el: EventLockedResult) -> pd.DataFrame:
    """Wilcoxon pre vs post per series over THIS recording's events
    (core.paired_tests, Bonferroni across the series). Descriptive only --
    see WITHIN_RECORDING_NOTE."""
    return paired_tests(el.table, keys=list(el.keys))


_DECIMALS = {"hr": 1, "d": 2, "sd1": 3, "sd2": 3, "csi": 2, "cpi": 2}
# The SPWVD band powers span orders of magnitude between recordings and
# settings (unnormalised a.u.): one fixed scientific format, so a column
# never switches between plain and exponent notation.
_SCIENTIFIC = frozenset({"lf", "hf"})


def _val(key: str, value: Any, signed: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    if key in _DECIMALS:
        return f"{v:+.{_DECIMALS[key]}f}" if signed else f"{v:.{_DECIMALS[key]}f}"
    if key in _SCIENTIFIC:
        return f"{v:+.3e}" if signed else f"{v:.3e}"
    return f"{v:+.4g}" if signed else f"{v:.4g}"


def _flag_default(value: Any, default: bool) -> bool:
    """A flag cell; a missing one (None / NaN) is *default*."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return _to_bool(value)
    except (TypeError, ValueError):
        return bool(value)


def invalid_reason(row: Mapping, pre_s: float, post_s: float) -> str:
    """Why an event row of an EventLockedResult table is invalid: another
    annotation inside its windows or at its onset (not isolated), or its
    CSI / CPI windows outside the series / under-covered."""
    if not _flag_default(row.get("isolated"), True):
        prev, nxt = _num(row.get("prev_gap_s")), _num(row.get("next_gap_s"))
        if prev == 0 or nxt == 0:
            return "another annotation starts at the same onset"
        parts = []
        if math.isfinite(prev) and prev < pre_s:
            parts.append(f"another annotation {prev:.1f} s before the onset, inside "
                         f"the {pre_s:g} s pre window")
        if math.isfinite(nxt) and nxt <= post_s:
            parts.append(f"another annotation {nxt:.1f} s after the onset, inside "
                         f"the {post_s:g} s post window")
        return "; ".join(parts) or "another annotation inside its windows"
    return "CSI / CPI windows outside the series or under-covered"


def _p_text(p: Any) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"


def _gains_text(p: Mapping) -> str:
    ks, kp = float(p.get("ks", np.nan)), float(p.get("kp", np.nan))
    tail = ("rat SHR calibration — not validated in mice"
            if math.isfinite(ks) and math.isfinite(kp) and is_rat_calibration(ks, kp)
            else "user-set")
    return f"ks {ks:g} / kp {kp:g} ({tail})"


def event_table_text(el: EventLockedResult, width: int = 100) -> str:
    """The textbox under the event-locked plot: one block per event (pre,
    post, delta of every series), the within-recording Wilcoxon and the
    disclosures."""
    p = el.params
    lab = p.get("label_filter", ALL_LABELS)
    lines = [
        f"  EVENT-LOCKED  —  {el.n_events} event(s), {el.n_valid} valid  ·  "
        f"label: {lab}  ·  pre {float(p['pre_s']):g} s / post {float(p['post_s']):g} s"
        f"  ·  onset = annotation start",
        f"  Autonomic: {p.get('method_label') or p.get('method', '')}  ·  window "
        f"{float(p.get('window_s', np.nan)):g} s  ·  {_gains_text(p)}",
    ]
    if el.has_lfhf:
        lines.append(f"  LF / HF (SPWVD): LF {_band_text(p.get('lf_band'))} Hz, "
                     f"HF {_band_text(p.get('hf_band'))} Hz — unnormalised a.u.")
    t = el.table
    pre_s, post_s = float(p["pre_s"]), float(p["post_s"])
    for i in range(len(t)):
        row = t.iloc[i]
        state = ("valid" if bool(row["valid"])
                 else f"INVALID ({invalid_reason(row, pre_s, post_s)})")
        name = str(row["label"]) or UNLABELLED
        lines += ["", f"  Event {i + 1}  “{name}”  onset {float(row['onset_s']):.2f} s  —  "
                      f"{state}  (coverage pre {100 * float(row['coverage_pre']):.0f} % / "
                      f"post {100 * float(row['coverage_post']):.0f} %)"]
        if not any(bool(row[f"{k}_valid"]) for k in el.keys):
            lines.append("      no series is computable around this event")
            continue
        lines.append(f"      {'series':<9}{'pre':>12}{'post':>12}{'Δ (post − pre)':>17}")
        for k in el.keys:
            note = "" if bool(row[f"{k}_valid"]) else "   (invalid for this series)"
            lines.append(
                f"      {KEY_LABELS[k]:<9}{_val(k, row[f'{k}_pre']):>12}"
                f"{_val(k, row[f'{k}_post']):>12}{_val(k, row[f'{k}_delta'], True):>17}"
                f"  {KEY_UNITS[k]}{note}")

    tests = within_recording_tests(el)
    lines += ["", f"  WITHIN-RECORDING WILCOXON  (descriptive only — the events of ONE animal)",
              f"      {'series':<9}{'n':>4}{'median Δ':>12}{'z':>8}{'p':>11}{'p(z)':>11}"
              f"{'p (Bonf.)':>12}"]
    for _, r in tests.iterrows():
        k = str(r["key"])
        lines.append(f"      {KEY_LABELS.get(k, k):<9}{int(r['n']):>4}"
                     f"{_val(k, r['median_delta'], True):>12}{_fmt(r['z'], 2):>8}"
                     f"{_p_text(r['p_value']):>11}{_p_text(r.get('p_normal')):>11}"
                     f"{_p_text(r['p_bonferroni']):>12}")
    notes = [WITHIN_RECORDING_NOTE, VALIDITY_NOTE, ZSCORE_NOTE]
    pm = _p_method_note(tests)
    if pm:
        notes.append(pm)
    mixed = mixed_labels(el)
    if mixed:
        notes.insert(0, MIXED_LABELS_NOTE.format(labels=mixed))
    lines += ["", "  Notes"]
    for note in notes:
        lines += textwrap.wrap(note, width=width, initial_indent="  * ",
                               subsequent_indent="    ")
    return "\n".join(lines)


def mixed_labels(el: EventLockedResult) -> str:
    """'dose (3), vehicle (2)' when the VALID events span more than one
    label (a pooled contrast), else ''."""
    t = el.table
    if not len(t):
        return ""
    valid = t[t["valid"].to_numpy(dtype=bool)]
    counts = valid["label"].map(lambda v: str(v) or UNLABELLED).value_counts()
    if len(counts) < 2:
        return ""
    return ", ".join(f"{lab} ({int(n)})" for lab, n in sorted(counts.items()))


def _invalid_summary(el: EventLockedResult) -> str:
    """'2 with another annotation inside the windows; 1 with windows outside
    the series or under-covered' for the invalid events."""
    t = el.table
    bad = ~t["valid"].to_numpy(dtype=bool)
    iso = (t["isolated"].to_numpy(dtype=bool) if "isolated" in t.columns
           else np.ones(len(t), dtype=bool))
    n_iso = int((bad & ~iso).sum())
    n_cov = int(bad.sum()) - n_iso
    parts = []
    if n_iso:
        parts.append(f"{n_iso} with another annotation inside the windows")
    if n_cov:
        parts.append(f"{n_cov} with windows outside the series or under-covered")
    return "; ".join(parts)


def event_status(el: EventLockedResult) -> "tuple[str, str]":
    """(text, level) of the panel's status label: "ok" (every event valid,
    one label), "warn" (some invalid, or several labels pooled) or
    "withheld" (none valid)."""
    p = el.params
    n, nv = el.n_events, el.n_valid
    text = (f"  {n} event(s) · {nv} valid · pre {float(p['pre_s']):g} s / "
            f"post {float(p['post_s']):g} s · label: {p.get('label_filter', ALL_LABELS)}")
    if el.has_lfhf:
        text += " · LF/HF"
    if nv == 0:
        return text + f" · no valid event ({_invalid_summary(el)})", "withheld"
    mixed = mixed_labels(el)
    if nv < n:
        text += f" · {n - nv} invalid ({_invalid_summary(el)})"
    if mixed:
        text += f" · labels pooled: {mixed}"
    if nv < n or mixed:
        return text, "warn"
    return text + "  ✓", "ok"


# ════════════════════════════════════════════════════════════
#  COHORT REGISTRY ROWS
# ════════════════════════════════════════════════════════════

def _clean(value: Any) -> Any:
    """A numpy / pandas scalar as a plain Python value (NaN kept)."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def registry_rows(el: EventLockedResult) -> "list[dict]":
    """One ``db.save_event_rows`` row per event: label, onset_s, pre_s,
    post_s, valid (the CSI family's) and ``metrics`` {key: {pre, post,
    delta, valid}} for every series present (NaN -> stored as null)."""
    rows = []
    pre_s, post_s = float(el.params["pre_s"]), float(el.params["post_s"])
    for _, r in el.table.iterrows():
        metrics = {k: {"pre": _clean(r[f"{k}_pre"]), "post": _clean(r[f"{k}_post"]),
                       "delta": _clean(r[f"{k}_delta"]), "valid": bool(r[f"{k}_valid"])}
                   for k in el.keys}
        rows.append({"label": str(r["label"]), "onset_s": float(r["onset_s"]),
                     "pre_s": pre_s, "post_s": post_s, "valid": bool(r["valid"]),
                     "metrics": metrics})
    return rows


_REGISTRY_PARAM_KEYS = (
    "method", "method_label", "window_s", "ks", "kp", "min_pairs", "fs_out",
    "pre_s", "post_s", "min_coverage", "label_filter", "rr_min_ms",
    "rr_max_ms", "n_breaks", "lf_hf", "lf_band", "hf_band", "spwvd_fs_hz",
    "spwvd_fbins", "require_isolated", "n_not_isolated", "reference",
)


def is_default_subject(subject: Any) -> bool:
    """True for the sidebar's untouched Subject ID ("subject_01") and the
    like ("Subject 2", "subject") -- a placeholder rather than an animal."""
    return bool(_DEFAULT_SUBJECT_RE.match(str(subject or "").strip()))


def subject_warning(subject: str, filepath: str, project: str,
                    rows: "Iterable[Mapping]") -> str:
    """Why saving *filepath*'s events under *subject* may pool animals,
    for the save status ('' when nothing to say): the Subject ID looks
    like the default placeholder, and / or *rows* (``db.list_event_rows``
    of the project, after the save) already hold events of OTHER
    recordings under the same subject -- one animal per Subject ID, so the
    per-subject calibration averages them as one."""
    subj = str(subject or "").strip()
    if not subj:
        return ""
    others = sorted({str(r.get("filepath") or "") for r in rows or ()
                     if isinstance(r, Mapping)
                     and str(r.get("subject") or "").strip() == subj
                     and str(r.get("filepath") or "") not in ("", filepath)})
    parts = []
    if is_default_subject(subj):
        parts.append(f"Subject ID “{subj}” looks like the app's default: if it was not "
                     "set for this animal, set it in the sidebar and save again")
    if others:
        from pathlib import Path
        names = ", ".join(Path(f).name for f in others[:3]) + (" …" if len(others) > 3 else "")
        parts.append(f"subject “{subj}” (project “{project or '—'}”) now holds events of "
                     f"{len(others) + 1} recordings (also {names}): the per-subject "
                     "calibration averages them as ONE animal — give each animal its "
                     "own Subject ID")
    return "; ".join(parts) + ("." if parts else "")


def registry_params(el: EventLockedResult) -> dict:
    """The configuration stored with every registry row (its ``method`` and
    ``window_s`` are part of the row key)."""
    p = el.params
    out = {k: _clean(p[k]) for k in _REGISTRY_PARAM_KEYS if k in p and p[k] is not None}
    out["onset"] = "annotation start"
    out["units"] = ("D, SD1, SD2 in ms; CSI, CPI in a.u. (ms-scaled); HR in bpm; "
                    "LF, HF in unnormalised a.u.")
    return out


def export_event_params(el: EventLockedResult) -> dict:
    """The event-locked configuration for the exporters (Excel EventLocked
    sheet, Prism notes, the Cover rows), with the notes they list."""
    out = {k: _clean(v) for k, v in el.params.items()}
    out["onset"] = "annotation start"
    out["notes"] = [WITHIN_RECORDING_NOTE, VALIDITY_NOTE]
    return out


def export_rows(el: EventLockedResult, filepath: str = "", subject: str = "",
                project_name: str = "") -> "list[dict]":
    """``registry_rows`` with the identity columns of a registry row
    (filepath, stem, subject, project, method, window_s, params) -- the
    rows of the panel's CSV export, in db.export_event_rows_csv's format."""
    from pathlib import Path
    params = registry_params(el)
    stem = Path(filepath).stem if filepath else ""
    out = []
    for row in registry_rows(el):
        out.append({"filepath": filepath or "", "stem": stem, "subject": subject or "",
                    "project_name": project_name or "",
                    "method": params.get("method"), "window_s": params.get("window_s"),
                    **row, "params": params})
    return out


# ════════════════════════════════════════════════════════════
#  COHORT CALIBRATION (registry rows)
# ════════════════════════════════════════════════════════════

_ID_COLUMNS = ("id", "filepath", "stem", "subject", "project_name", "label",
               "onset_s", "pre_s", "post_s", "method", "window_s", "valid")


def _num(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _flag(value: Any) -> bool:
    """A stored per-metric ``valid`` sub-field; missing means valid."""
    if value is None:
        return True
    try:
        return _to_bool(value)
    except (TypeError, ValueError):
        return bool(value)


def registry_frame(rows: "Iterable[Mapping]") -> pd.DataFrame:
    """Flatten ``db.list_event_rows()`` rows into one event table: the
    identity columns, ``ks`` / ``kp`` from each row's params,
    ``subject_key`` (the subject ID, or "[recording stem]" for an event
    without one), then k_pre, k_post, k_delta and k_valid (the stored
    per-series flag AND finite pre / post) for every metric key."""
    recs: "list[dict]" = []
    keys: "list[str]" = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        params = row.get("params") if isinstance(row.get("params"), Mapping) else {}
        rec: "dict[str, Any]" = {c: row.get(c) for c in _ID_COLUMNS}
        rec["ks"] = _num(params.get("ks"))
        rec["kp"] = _num(params.get("kp"))
        for k, entry in metrics.items():
            if not isinstance(entry, Mapping):
                continue
            k = str(k)
            if k not in keys:
                keys.append(k)
            rec[f"{k}_pre"] = _num(entry.get("pre"))
            rec[f"{k}_post"] = _num(entry.get("post"))
            rec[f"{k}_delta"] = _num(entry.get("delta"))
            rec[f"{k}_valid"] = _flag(entry.get("valid"))
        recs.append(rec)
    df = pd.DataFrame(recs, columns=[*_ID_COLUMNS, "ks", "kp"]
                      + [f"{k}{s}" for k in keys for s in ("_pre", "_post", "_delta", "_valid")])
    for c in ("onset_s", "pre_s", "post_s", "window_s", "ks", "kp"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["valid"] = [True if v is None or (isinstance(v, float) and math.isnan(v))
                   else bool(v) for v in df["valid"]]
    df["valid"] = df["valid"].astype(bool)
    for k in keys:
        for s in ("_pre", "_post", "_delta"):
            df[f"{k}{s}"] = df[f"{k}{s}"].astype(float)
        flags = [False if v is None or (isinstance(v, float) and math.isnan(v)) else bool(v)
                 for v in df[f"{k}_valid"]]
        df[f"{k}_valid"] = (np.asarray(flags, dtype=bool)
                            & np.isfinite(df[f"{k}_pre"].to_numpy())
                            & np.isfinite(df[f"{k}_post"].to_numpy()))
    subj = df["subject"].fillna("").astype(str).str.strip()
    stem = df["stem"].fillna("").astype(str)
    fallback = np.where(stem != "", stem, df["filepath"].fillna("").astype(str))
    df["subject_key"] = np.where(subj != "", subj, [f"[{s}]" for s in fallback])
    df.attrs["metric_keys"] = keys
    return df


def config_of(method: Any, window: Any) -> "tuple[Optional[str], Optional[float]]":
    """The comparable (method, window_s) configuration of a registry row --
    the part of its key besides the event: method text (None when missing),
    window rounded to 1 µs (None when missing), as db.save_event_rows
    stores them."""
    m = (None if method is None or (isinstance(method, float) and math.isnan(method))
         else (str(method).strip() or None))
    w = _num(window)
    return m, (None if math.isnan(w) else round(w, 6))


def row_config(row: Mapping) -> "tuple[Optional[str], Optional[float]]":
    """config_of() of a db.list_event_rows() row."""
    return config_of(row.get("method"), row.get("window_s"))


def config_label(config: "tuple[Optional[str], Optional[float]]") -> str:
    """'robust · 3 s' for (method, window_s); '?' for a missing part."""
    m, w = config
    return f"{m or '?'} · " + (f"{w:g} s" if w is not None else "? s")


def config_options(frame: pd.DataFrame) -> "list[tuple[tuple[Optional[str], Optional[float]], int]]":
    """Distinct (method, window_s) configurations with their event counts,
    most events first -- a cohort test must not mix them (the same event
    stored with two methods would count twice)."""
    counts: "dict[tuple, int]" = {}
    for m, w in zip(frame.get("method", pd.Series(dtype=object)),
                    frame.get("window_s", pd.Series(dtype=float))):
        cfg = config_of(m, w)
        counts[cfg] = counts.get(cfg, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], config_label(kv[0])))


def filter_config(frame: pd.DataFrame,
                  config: "Optional[tuple[Optional[str], Optional[float]]]") -> pd.DataFrame:
    """The rows of *frame* stored with *config* (None: every row)."""
    if config is None or frame.empty:
        return frame.copy()
    want = config_of(config[0], config[1])
    keep = [config_of(m, w) == want for m, w in zip(frame["method"], frame["window_s"])]
    return frame[np.asarray(keep, dtype=bool)].copy()


def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name in frame.columns:
        return frame[name].to_numpy(dtype=float)
    return np.full(len(frame), np.nan)


def recompose_gains(frame: pd.DataFrame, ks: float, kp: float) -> pd.DataFrame:
    """*frame* with CSI / CPI pre, post and delta re-composed for the gains
    *ks*, *kp* (exact -- the indices are linear in them):

        CPI = D + kp·SD1                       (window means)
        CSI = D̃ + ks·SD2,  D̃ = stored CSI − ks_stored·SD2

    D̃ (D mirrored about its whole-recording mean) is not stored, so CSI
    needs the stored CSI and the ks it was computed with (the row's
    params); without them it is NaN. ``csi_valid`` / ``cpi_valid`` follow
    the finite values."""
    out = frame.copy()
    ks, kp = float(ks), float(kp)
    d_pre, d_post = _col(out, "d_pre"), _col(out, "d_post")
    s1_pre, s1_post = _col(out, "sd1_pre"), _col(out, "sd1_post")
    s2_pre, s2_post = _col(out, "sd2_pre"), _col(out, "sd2_post")
    ks_row = _col(out, "ks")
    with np.errstate(invalid="ignore"):
        cpi_pre, cpi_post = d_pre + kp * s1_pre, d_post + kp * s1_post
        csi_pre = _col(out, "csi_pre") + (ks - ks_row) * s2_pre
        csi_post = _col(out, "csi_post") + (ks - ks_row) * s2_post
    for name, pre, post in (("cpi", cpi_pre, cpi_post), ("csi", csi_pre, csi_post)):
        old = (out[f"{name}_valid"].to_numpy(dtype=bool) if f"{name}_valid" in out.columns
               else np.ones(len(out), dtype=bool))
        out[f"{name}_pre"] = pre
        out[f"{name}_post"] = post
        out[f"{name}_delta"] = post - pre
        out[f"{name}_valid"] = old & np.isfinite(pre) & np.isfinite(post)
    return out


def cohort_table(events: pd.DataFrame, aggregation: str = DEFAULT_AGGREGATION) -> pd.DataFrame:
    """The table the cohort statistics run on: the events themselves
    ("event"), or one row per subject_key with the mean of its valid
    events ("subject", core.aggregate -- avoids pseudo-replication)."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}.")
    if aggregation == "event" or events.empty:
        return events.copy()
    metric_cols = [c for c in events.columns
                   if c.endswith(("_pre", "_post", "_delta", "_valid"))
                   and c not in ("pre_s", "post_s")]
    return aggregate(events[["subject_key", "valid", *metric_cols]], by="subject_key")


@dataclass(eq=False)
class CohortCalibration:
    """Calibration of ks / kp on registry events (see cohort_calibration)."""
    events:      pd.DataFrame           # per-event rows of the configuration, re-composed
    table:       pd.DataFrame           # what the statistics run on
    aggregation: str
    curves:      "dict[str, pd.DataFrame]"   # "csi" / "cpi" -> calibrate_gain frames
    knees:       "dict[str, float]"
    tests:       pd.DataFrame           # paired_tests of CSI / CPI at ks / kp
    ks:          float
    kp:          float
    config:      "Optional[tuple[Optional[str], Optional[float]]]"
    counts:      "dict[str, int]"
    notes:       "list[str]"
    # "csi" / "cpi" -> curve diagnostics (curve_diagnostics)
    diagnostics: "dict[str, dict]" = field(default_factory=dict)
    # valid events per label, most first (several = a pooled contrast)
    labels:      "dict[str, int]" = field(default_factory=dict)


_GAIN_NAME = {"csi": "ks", "cpi": "kp"}


def curve_diagnostics(curve: pd.DataFrame, knee: float,
                      raw_knee: float) -> dict:
    """What the dialog says about one Z(k) curve (a ``calibrate_gain``
    frame), beside its knee:

    * ``runs``: ``core.sign_stretches`` of z -- more than one means the
      index REVERSES direction as k grows (its heart-rate and HRV terms
      oppose), so the knee on the last run makes it move opposite to its
      small-k direction (``reversed_knee``);
    * ``abs_max``, ``k_max`` (first, last k of the maximal |z|),
      ``k_max_ranges`` (the runs of consecutive swept k where |z| is
      maximal, as (first, last) pairs -- a reversing curve can reach it on
      both stretches), ``at_ceiling`` (that maximum is the Wilcoxon ceiling
      ``z_max``: every unit moves the same way there, Z cannot rank those
      gains);
    * ``rank_refused``: Kneedle found a knee (*raw_knee*) but it rested on
      fewer than ``MIN_RANK_STEPS`` rank steps, so none is offered."""
    k = curve["k"].to_numpy(dtype=float)
    z = curve["z"].to_numpy(dtype=float)
    zmax = curve["z_max"].to_numpy(dtype=float) if "z_max" in curve.columns else np.full(len(k), np.nan)
    out: dict = {"runs": [], "abs_max": float("nan"), "k_max": None, "k_max_ranges": [],
                 "at_ceiling": False, "z_max": float("nan"),
                 "reversed_knee": False, "rank_refused": False}
    fin = np.isfinite(k) & np.isfinite(z)
    if not fin.any():
        return out
    try:
        out["runs"] = sign_stretches(k, z)
    except ValueError:
        out["runs"] = []
    az = np.abs(z[fin])
    kf = k[fin]
    m = float(az.max())
    at = kf[np.abs(az - m) <= 1e-9 * max(1.0, m)]
    out["abs_max"] = m
    out["k_max"] = (float(at.min()), float(at.max()))
    order = np.argsort(kf, kind="stable")
    is_max = (np.abs(az - m) <= 1e-9 * max(1.0, m))[order]
    ks_sorted = kf[order]
    ranges: "list[tuple[float, float]]" = []
    for i, hit in enumerate(is_max):
        if not hit:
            continue
        if ranges and is_max[i - 1]:
            ranges[-1] = (ranges[-1][0], float(ks_sorted[i]))
        else:
            ranges.append((float(ks_sorted[i]), float(ks_sorted[i])))
    out["k_max_ranges"] = ranges
    zm = zmax[fin][np.abs(az - m) <= 1e-9 * max(1.0, m)]
    if len(zm) and np.isfinite(zm).all():
        out["z_max"] = float(zm.max())
        out["at_ceiling"] = bool(m > 0 and np.all(np.abs(zm - m) <= 1e-9 * max(1.0, m)))
    runs = out["runs"]
    if len(runs) > 1 and math.isfinite(knee):
        out["reversed_knee"] = bool(knee >= runs[-1]["k_first"])
    out["rank_refused"] = bool(math.isfinite(raw_knee) and not math.isfinite(knee))
    return out


def label_counts(events: pd.DataFrame) -> "dict[str, int]":
    """Events per label of a ``registry_frame`` (unlabelled as
    "(unlabelled)"), most first."""
    if events.empty or "label" not in events.columns:
        return {}
    labs = events["label"].fillna("").astype(str).map(lambda s: s.strip() or UNLABELLED)
    counts = labs.value_counts()
    return {str(lab): int(n) for lab, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def default_calibration_label(counts: "Mapping[str, int]") -> "Optional[str]":
    """The calibration dialog's starting label: the one with the most events
    (``label_counts`` order) -- a single condition contrast, as the paper's
    rest vs dobutamine -- or None (every label) when no event is labelled."""
    for lab in counts:
        if lab != UNLABELLED:
            return lab
    return None


def cohort_calibration(frame: pd.DataFrame, ks: float, kp: float,
                       aggregation: str = DEFAULT_AGGREGATION,
                       config: "Optional[tuple[Optional[str], Optional[float]]]" = None,
                       k_values: "Iterable[float]" = K_VALUES) -> CohortCalibration:
    """Z(k) of CSI (from ΔD and ΔSD2) and CPI (from ΔD and ΔSD1) over
    *k_values* with their knees, and the paired Wilcoxon of CSI / CPI at
    the gains *ks*, *kp* (Bonferroni over the two indices), on the events
    of *frame* (``registry_frame``) stored with *config* (None: the most
    common configuration), per event or per subject mean.

    A knee is not offered when it rests on fewer than ``MIN_RANK_STEPS``
    rank steps of Z (core.knee_point); ``diagnostics`` says when Z(k)
    reverses sign (the knee then makes the index move opposite to its
    small-k direction) or sits on the Wilcoxon ceiling, and the notes
    disclose pooled labels, default-looking Subject IDs shared by several
    recordings and in-sample testing."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}.")
    ks, kp = float(ks), float(kp)
    if config is None:
        opts = config_options(frame)
        config = opts[0][0] if opts else None
    events = recompose_gains(filter_config(frame, config), ks, kp)
    table = cohort_table(events, aggregation)
    k_list = list(k_values)
    curves, knees, diags = {}, {}, {}
    for kind, sd in (("csi", "sd2_delta"), ("cpi", "sd1_delta")):
        curve = calibrate_gain(_col(table, "d_delta"), _col(table, sd), kind, k_list)
        curves[kind] = curve
        kk = curve["k"].to_numpy(dtype=float)
        zz = curve["z"].to_numpy(dtype=float)
        try:
            raw = knee_point(kk, zz)
            knees[kind] = knee_point(kk, zz, statistic=curve["statistic"].to_numpy(dtype=float),
                                     min_rank_steps=MIN_RANK_STEPS)
        except ValueError:
            raw = knees[kind] = float("nan")
        diags[kind] = curve_diagnostics(curve, knees[kind], raw)
    # recompose_gains always provides the csi_* / cpi_* columns.
    tests = paired_tests(table, keys=["csi", "cpi"])

    n_events = int(len(events))
    counts = {
        "n_events": n_events,
        "n_subjects": int(events["subject_key"].nunique()) if n_events else 0,
        "n_recordings": int(events["filepath"].nunique()) if n_events else 0,
        "n_units": int(len(table)),
        "n_no_subject": int((events["subject"].fillna("").astype(str).str.strip() == "").sum())
                        if n_events else 0,
        "n_total": int(len(frame)),
    }
    labels = label_counts(events)
    notes: "list[str]" = []
    if len(labels) > 1:
        notes.append(SINGLE_CONTRAST_NOTE.format(
            labels=", ".join(f"{lab} {n}" for lab, n in labels.items())))
    if n_events:
        windows = (events["pre_s"].map(lambda v: f"{v:g}") + " / "
                   + events["post_s"].map(lambda v: f"{v:g}") + " s")
        mix = windows.value_counts()
        if len(mix) > 1:
            notes.append("Events with different pre / post windows are mixed: "
                         + ", ".join(f"{w} ({n})" for w, n in mix.items()) + ".")
        if counts["n_no_subject"]:
            notes.append(f"{counts['n_no_subject']} event(s) have no subject ID: "
                         "grouped per recording (one recording = one subject).")
        subj = events["subject"].fillna("").astype(str).str.strip()
        dflt = subj.map(is_default_subject).to_numpy(dtype=bool)
        if dflt.any():
            shared = [s for s in sorted(set(subj[dflt]))
                      if events.loc[(subj == s).to_numpy(), "filepath"].nunique() > 1]
            if shared:
                sel = subj.isin(shared).to_numpy()
                notes.append(DEFAULT_SUBJECT_NOTE.format(
                    n=int(sel.sum()), r=int(events.loc[sel, "filepath"].nunique()),
                    ids=", ".join(f"“{s}”" for s in shared)))
        gains = {(a, b) for a, b in zip(events["ks"], events["kp"])
                 if math.isfinite(a) and math.isfinite(b)}
        if len(gains) > 1 or (gains and gains != {(ks, kp)}):
            notes.append("Stored with other gains than the current ones: CSI / CPI "
                         "re-composed at the current ks / kp (exact — the indices "
                         "are linear in the gains).")
        n_csi_missing = int((~np.isfinite(_col(events, "csi_pre"))).sum())
        if n_csi_missing:
            notes.append(f"{n_csi_missing} event(s) without a stored CSI or its ks: "
                         "left out of the CSI test (Z(k) needs only ΔD and ΔSD2).")
    if len(frame) and config is not None and counts["n_events"] < len(frame):
        notes.append(f"{len(frame) - counts['n_events']} event(s) stored with another "
                     "method / window are not mixed in (choose the configuration above).")
    n_units = counts["n_units"]
    if 0 < n_units < 6:
        notes.append(f"Only {n_units} paired sample(s): the exact two-sided Wilcoxon "
                     "p cannot reach 0.05 below 6 — read Z(k) for its shape.")
    return CohortCalibration(events=events, table=table, aggregation=aggregation,
                             curves=curves, knees=knees, tests=tests, ks=ks, kp=kp,
                             config=config, counts=counts, notes=notes,
                             diagnostics=diags, labels=labels)


def in_sample(cal: CohortCalibration) -> "dict[str, bool]":
    """Per index, whether the current gain IS this cohort's own knee -- the
    tests at the current gains are then in-sample (gains chosen and tested
    on the same events: optimistic p-values)."""
    out = {}
    for kind, gain in (("csi", cal.ks), ("cpi", cal.kp)):
        kn = cal.knees.get(kind, float("nan"))
        out[kind] = bool(math.isfinite(kn) and math.isclose(kn, gain, abs_tol=1e-9))
    return out


def _direction(kind: str, sign: int) -> str:
    return f"{kind.upper()} {'rises' if sign > 0 else 'falls'} after the onset"


def reversal_text(cal: CohortCalibration, kind: str) -> str:
    """'' or the disclosure of a Z(k) that changes sign: the direction of
    the contrast on each stretch, and that the knee (on the last stretch)
    makes the index move opposite to its small-gain direction."""
    d = cal.diagnostics.get(kind) or {}
    runs = d.get("runs") or []
    if len(runs) < 2:
        return ""
    gain = _GAIN_NAME[kind]
    first, last = runs[0], runs[-1]
    txt = (f"Z({gain}) reverses sign: {first['z_peak']:+.2f} at {gain} = {first['k_peak']:g} "
           f"({_direction(kind, first['sign'])} for {gain} = {first['k_first']:g}–"
           f"{first['k_last']:g}) → {last['z_peak']:+.2f} at {gain} = {last['k_peak']:g} "
           f"({_direction(kind, last['sign'])} for {gain} ≥ {last['k_first']:g}): the heart-"
           f"rate and HRV terms oppose, so the direction {kind.upper()} shows depends on the "
           "gain.")
    kn = cal.knees.get(kind, float("nan"))
    if d.get("reversed_knee") and math.isfinite(kn):
        txt += (f" The knee ({gain} = {kn:g}) lies on the reversed stretch: at that gain "
                f"{kind.upper()} moves OPPOSITE to its small-gain direction — choose {gain} by "
                "the direction the physiology predicts (e.g. a sympathomimetic should raise "
                "CSI), and type it in the Autonomic view if the knee contradicts it.")
    return txt


def ceiling_text(cal: CohortCalibration, kind: str) -> str:
    """'' or the disclosure of a Z(k) sitting on the Wilcoxon ceiling, and
    the k range of maximal |Z|."""
    d = cal.diagnostics.get(kind) or {}
    gain = _GAIN_NAME[kind]
    km = d.get("k_max")
    if km is None or not math.isfinite(d.get("abs_max", float("nan"))):
        return ""
    ranges = d.get("k_max_ranges") or [km]
    rng = f"{gain} = " + ", ".join(f"{a:g}" + (f"–{b:g}" if b > a else "") for a, b in ranges)
    if d.get("at_ceiling"):
        return (f"|Z| reaches its ceiling ({d['abs_max']:.2f} for {cal.counts['n_units']} "
                f"units: every unit changes the same way) for {rng}: Z cannot rank these "
                "gains.")
    return f"|Z| is largest ({d['abs_max']:.2f}) for {rng}."


def apply_plan(cal: CohortCalibration) -> "tuple[float, float, str, bool]":
    """(ks, kp, button text, enabled) of the dialog's Apply button: each
    knee when there is one, else the current gain (kept). A knee on a
    reversed Z(k) stretch is marked ⚠ (``apply_caveats`` asks to confirm)."""
    kn_s, kn_p = cal.knees.get("csi", float("nan")), cal.knees.get("cpi", float("nan"))
    ks = kn_s if math.isfinite(kn_s) else cal.ks
    kp = kn_p if math.isfinite(kn_p) else cal.kp

    def _txt(kind: str, name: str, value: float, kn: float) -> str:
        t = f"{name} = {value:g}"
        if not math.isfinite(kn):
            return t + " (kept)"
        if (cal.diagnostics.get(kind) or {}).get("reversed_knee"):
            t += " ⚠ reversed"
        return t

    s_txt = _txt("csi", "ks", ks, kn_s)
    p_txt = _txt("cpi", "kp", kp, kn_p)
    enabled = math.isfinite(kn_s) or math.isfinite(kn_p)
    return ks, kp, f"Apply {s_txt}, {p_txt}", enabled


def apply_caveats(cal: CohortCalibration) -> "list[str]":
    """What the user must confirm before Apply writes the knees: a knee on
    a reversed Z(k) stretch (the index would move opposite to its
    small-gain direction), several labels pooled into one contrast, and
    always, last, the in-sample caveat of testing these gains on these
    events. Empty when there is no knee to apply."""
    kn = {k: cal.knees.get(k, float("nan")) for k in ("csi", "cpi")}
    if not any(math.isfinite(v) for v in kn.values()):
        return []
    out = [reversal_text(cal, kind) for kind in ("csi", "cpi")
           if math.isfinite(kn[kind]) and (cal.diagnostics.get(kind) or {}).get("reversed_knee")]
    if len(cal.labels) > 1:
        out.append(SINGLE_CONTRAST_NOTE.format(
            labels=", ".join(f"{lab} {n}" for lab, n in cal.labels.items())))
    out.append(IN_SAMPLE_NOTE)
    return out


def tests_frame(cal: CohortCalibration) -> pd.DataFrame:
    """The current-gains tests as an Excel-ready frame (one row per index),
    with whether they are in-sample (the gain is this cohort's own knee)."""
    t = cal.tests.copy()
    t.insert(0, "index", t["key"].str.upper())
    t.insert(1, "gain", [f"ks = {cal.ks:g}" if k == "csi" else f"kp = {cal.kp:g}"
                         for k in t["key"]])
    t.insert(2, "aggregation", AGGREGATIONS[cal.aggregation])
    ins = in_sample(cal)
    t["in_sample"] = [ins.get(str(k), False) for k in t["key"]]
    return t.drop(columns=["key"])


def _med_mad(med: Any, mad: Any) -> str:
    m, d = _num(med), _num(mad)
    if math.isnan(m):
        return "—"
    return f"{m:.2f} ± {d:.2f}" if math.isfinite(d) else f"{m:.2f}"


def _p_method_note(tests: pd.DataFrame) -> str:
    methods = sorted({str(m) for m in tests.get("p_method", pd.Series(dtype=object))
                      if isinstance(m, str) and m})
    if not methods:
        return ""
    return ("p: Wilcoxon signed-rank, " + " / ".join(methods)
            + " (scipy: exact up to 50 units without ties); p(z): the normal p of the "
              "z shown (MATLAB signrank's 'approximate'), which can differ near 0.05.")


def calibration_text(cal: CohortCalibration, width: int = 96) -> str:
    """The dialog's results text: counts, the knees with their caveats
    (sign reversal, Wilcoxon ceiling, rank steps), the paired tests at the
    current gains (marked in-sample when they are this cohort's knees),
    the notes."""
    c = cal.counts
    unit = "subject means" if cal.aggregation == "subject" else "events"
    cfg = config_label(cal.config) if cal.config is not None else "—"
    lab_txt = (", ".join(f"{lab} ({n})" for lab, n in cal.labels.items())
               if cal.labels else "—")
    lines = [
        f"  COHORT  —  {c['n_events']} event(s) · {c['n_subjects']} subject(s) · "
        f"{c['n_recordings']} recording(s)  ·  configuration {cfg}",
        f"  Statistics on {c['n_units']} {unit} ({AGGREGATIONS[cal.aggregation].lower()})"
        f"  ·  labels: {lab_txt}",
        "",
    ]
    for kind, gain in (("csi", "ks"), ("cpi", "kp")):
        kn = cal.knees.get(kind, float("nan"))
        d = cal.diagnostics.get(kind) or {}
        if math.isfinite(kn):
            head = f"{gain} = {kn:g}" + ("  ⚠ on a reversed stretch" if d.get("reversed_knee") else "")
        elif d.get("rank_refused"):
            head = (f"none (Z({gain}) moves by fewer than {MIN_RANK_STEPS} rank steps — "
                    "the rank statistic's discreteness, not a plateau)")
        else:
            head = "none (no plateau within k = 1–20, or too few events)"
        lines.append(f"  {kind.upper()} knee of Z({gain}):  {head}")
        for extra in (reversal_text(cal, kind), ceiling_text(cal, kind)):
            if extra:
                lines += textwrap.wrap(extra, width=width, initial_indent="      ",
                                       subsequent_indent="      ")
    ins = in_sample(cal)
    tag = ("  —  IN-SAMPLE: these gains are this cohort's own knees (optimistic p)"
           if any(ins.values()) else "")
    lines += ["",
              f"  PAIRED WILCOXON, PRE vs POST, AT THE CURRENT GAINS  (ks = {cal.ks:g}, "
              f"kp = {cal.kp:g}){tag}",
              f"      {'index':<6}{'n':>4}{'pre median ± MAD':>22}{'post median ± MAD':>22}"
              f"{'z':>8}{'p':>11}{'p(z)':>11}{'p (Bonf. ×2)':>14}"]
    for _, r in cal.tests.iterrows():
        lines.append(f"      {str(r['key']).upper():<6}{int(r['n']):>4}"
                     f"{_med_mad(r['median_pre'], r['mad_pre']):>22}"
                     f"{_med_mad(r['median_post'], r['mad_post']):>22}"
                     f"{_fmt(r['z'], 2):>8}{_p_text(r['p_value']):>11}"
                     f"{_p_text(r.get('p_normal')):>11}"
                     f"{_p_text(r['p_bonferroni']):>14}"
                     + ("  in-sample" if ins.get(str(r["key"])) else ""))
    notes = list(cal.notes)
    pm = _p_method_note(cal.tests)
    if pm:
        notes.append(pm)
    lines += ["", "  Notes"]
    for note in notes + [IN_SAMPLE_NOTE, CALIBRATION_RULE_NOTE, PSEUDO_REPLICATION_NOTE]:
        lines += textwrap.wrap(note, width=width, initial_indent="  * ",
                               subsequent_indent="    ")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════

def _key_color(key: str) -> str:
    return {"csi": _theme.RED_MID, "cpi": _theme.GREEN_DARK, "hr": _theme.ORANGE_DARK,
            "lf": _theme.PURPLE, "hf": _theme.TEAL, "d": _theme.NAVY,
            "sd1": _theme.CORAL, "sd2": _theme.BLUE_DARK}.get(key, _theme.BLUE)


def draw_event_locked(fig: Any, el: EventLockedResult, zscore: bool = True,
                      max_traces: int = MAX_TRACES_DRAWN) -> None:
    """Populate *fig* with the event-locked panels -- CSI, CPI, HR and,
    when computed, LF / HF: the group median (bold) +- MAD (shaded) across
    the events valid for each series, the individual events faintly behind
    (up to *max_traces*), a vertical line at the onset. *zscore* selects
    the per-recording z-scored traces (display only)."""
    panels: "list[tuple[str, tuple[str, ...]]]" = [
        ("CSI", ("csi",)), ("CPI", ("cpi",)), ("HR", ("hr",))]
    if el.has_lfhf:
        panels.append(("LF / HF", SPWVD_KEYS))
    axes = list(np.atleast_1d(fig.subplots(len(panels), 1, sharex=True)))
    for ax in axes:
        style_axes(ax)
    p = el.params
    muted = _theme.PLOT["muted"]
    t_rel = el.t_rel
    head = (f"Event-locked ({p.get('label_filter', ALL_LABELS)})  ·  "
            f"{p.get('method_label') or p.get('method', '')}  ·  window "
            f"{float(p.get('window_s', np.nan)):g} s  ·  {_gains_text(p)}"
            + ("  ·  z-scored per recording" if zscore else ""))
    fig.suptitle(head, fontsize=8, color=_theme.PLOT["text"])

    for ax, (name, keys) in zip(axes, panels):
        n_max = 0
        for key in keys:
            med, mad, n, rows = group_stats(el, key, zscore)
            n_max = max(n_max, n)
            if n == 0:
                continue
            col = _key_color(key)
            if n <= max_traces:
                for row in rows:
                    ax.plot(t_rel, row, color=col, lw=0.6, alpha=0.22, zorder=2)
            ax.fill_between(t_rel, med - mad, med + mad, color=col, alpha=0.22,
                            linewidth=0, zorder=3)
            ax.plot(t_rel, med, color=col, lw=1.6, zorder=4,
                    label=f"{KEY_LABELS[key]} (n = {n})")
        ax.axvline(0.0, color=muted, lw=1.0, ls="--", zorder=5)
        if zscore:
            ax.axhline(0.0, color=muted, lw=0.6, alpha=0.6, zorder=1)
        units = {KEY_UNITS[k] for k in keys}
        ax.set_ylabel("z-score" if zscore else f"{name} ({units.pop() if len(units) == 1 else 'a.u.'})",
                      fontsize=8)
        ax.set_title(f"{name} — group median ± MAD across the events valid for it",
                     loc="left", fontsize=8)
        if n_max == 0:
            ax.text(0.5, 0.5, f"No valid event for {name}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=muted)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=len(handles),
                      frameon=True, framealpha=0.8, facecolor=_theme.PLOT["bg"],
                      edgecolor="none")
    if len(t_rel):
        for ax in axes:
            ax.set_xlim(float(t_rel[0]), float(t_rel[-1]))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time from event onset (s)  —  onset = annotation start")


def draw_calibration(fig: Any, cal: CohortCalibration) -> None:
    """Z(k) of CSI (left) and CPI (right) for k = 1..20: the curve, the
    knee (star), the current gain (dashed), the paper's rat values
    (dotted grey), |Z| = 1.96, the Wilcoxon ceiling ±z_max (dash-dot) and,
    when Z(k) changes sign, the reversed stretch shaded and flagged."""
    axes = list(np.atleast_1d(fig.subplots(1, 2)))
    muted = _theme.PLOT["muted"]
    for ax, kind, gain, current, rat in (
            (axes[0], "csi", "ks", cal.ks, KS_RAT_SHR),
            (axes[1], "cpi", "kp", cal.kp, KP_RAT_SHR)):
        style_axes(ax)
        curve = cal.curves.get(kind)
        col = _key_color(kind)
        formula = "−ΔD + ks·ΔSD2" if kind == "csi" else "ΔD + kp·ΔSD1"
        kn = cal.knees.get(kind, float("nan"))
        diag = cal.diagnostics.get(kind) or {}
        runs = diag.get("runs") or []
        title = (f"{kind.upper()}: Wilcoxon Z({gain}) of {formula}"
                 + (f"  ·  knee {gain} = {kn:g}" if math.isfinite(kn) else "  ·  no knee")
                 + ("  ·  ⚠ Z reverses sign" if len(runs) > 1 else ""))
        if len(runs) > 1:
            ax.axvspan(runs[-1]["k_first"] - 0.5, runs[-1]["k_last"] + 0.5,
                       color=_theme.ORANGE, alpha=0.08, linewidth=0, zorder=0,
                       label=f"{kind.upper()} reversed (opposite direction)")
        z_ceiling = diag.get("z_max", float("nan"))
        if isinstance(z_ceiling, float) and math.isfinite(z_ceiling) and z_ceiling > 0:
            for y in (-z_ceiling, z_ceiling):
                ax.axhline(y, color=muted, lw=0.7, ls="-.", alpha=0.7, zorder=1)
            ax.text(0.01, 0.99, f"|Z| ceiling {z_ceiling:.2f} (n = {cal.counts['n_units']})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7, color=muted)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel(gain, fontsize=8)
        ax.set_ylabel("Z (post vs pre)", fontsize=8)
        for y in (-1.96, 1.96):
            ax.axhline(y, color=muted, lw=0.7, ls=":", zorder=1)
        ax.axhline(0.0, color=muted, lw=0.6, alpha=0.6, zorder=1)
        ax.axvline(rat, color=muted, lw=0.8, ls=":", alpha=0.8, zorder=1,
                   label=f"rat SHR {gain} = {rat:g}")
        ax.axvline(current, color=_theme.BLUE, lw=1.0, ls="--", zorder=2,
                   label=f"current {gain} = {current:g}")
        if curve is None or not np.isfinite(curve["z"].to_numpy(dtype=float)).any():
            ax.text(0.5, 0.5, "No event with ΔD and ΔSD — save event-locked "
                              "results to the registry first",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8,
                    color=muted, wrap=True)
        else:
            k = curve["k"].to_numpy(dtype=float)
            z = curve["z"].to_numpy(dtype=float)
            ax.plot(k, z, "-o", color=col, ms=3, lw=1.4, zorder=3, label="Z(k)")
            if 1 < len(k) <= 25 and np.all(k == np.round(k)):
                ax.set_xticks(k)
                ax.set_xlim(float(k.min()) - 0.5, float(k.max()) + 0.5)
            if math.isfinite(kn):
                zk = float(z[np.argmin(np.abs(k - kn))])
                ax.plot([kn], [zk], marker="*", ms=14, color=_theme.ORANGE,
                        mec=_theme.PLOT["text"], mew=0.6, zorder=5,
                        label=f"knee {gain} = {kn:g}")
        ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.8,
                  facecolor=_theme.PLOT["bg"], edgecolor="none")
