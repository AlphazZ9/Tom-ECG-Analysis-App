# -*- coding: utf-8 -*-
"""
ecg.core.segment_plan
---------------------
Compare segments: windows placed relative to an event, saved as templates so
every animal of a group is cut the same way, and the quality checks of each
segment.

A template window is (label, ref, lo, hi): *ref* is "start" or "end" of the
event (an annotation, a stimulus) and lo / hi are offsets in seconds from it,
negative before. "Before / after, 5 min" is Before = start -300..0 and
After = end +30..+330: the 30 s after the stimulus, where the shock or the
movement corrupts the ECG, are left out.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

TEMPLATES_PATH = Path.home() / ".ecg_segment_templates.json"
MAX_SEGMENTS = 6
REFS = ("start", "end")

# Quality thresholds of a segment (segment_quality / quality_flags).
QC_MIN_VALID_PCT = 80.0          # time covered by valid RR intervals, % of the window
QC_MIN_PAIRS = 200               # Poincaré pairs: SD1 / SD2 are unstable below
QC_DURATION_TOL = 0.02           # durations within 2 % of the first segment
QC_JUMP_FRAC = 0.20              # |RR_n+1 - RR_n| > 20 % of the median RR: an abrupt jump
QC_MAX_JUMP_PCT = 0.5            # more than 0.5 % of the pairs: missed / extra beats likely
QC_SD1_RATIO = 1.5               # SD1 > 1.5 x its trimmed value: driven by a few outlier pairs
QC_SD1_TRIM = 0.02               # trimmed SD1 leaves out the 2 % most extreme pairs
DEFAULT_ARTEFACT_MARGIN_S = 30.0


@dataclass(frozen=True)
class WindowSpec:
    label: str
    ref: str          # "start" | "end" of the event
    lo: float         # s from ref
    hi: float


@dataclass(frozen=True)
class SegmentTemplate:
    name: str
    windows: "tuple[WindowSpec, ...]"
    builtin: bool = False


@dataclass(frozen=True)
class ResolvedWindow:
    label: str
    lo: float
    hi: float
    clamped: bool


def _w(label: str, ref: str, lo: float, hi: float) -> WindowSpec:
    return WindowSpec(label, ref, float(lo), float(hi))


BUILTIN_TEMPLATES: "tuple[SegmentTemplate, ...]" = (
    SegmentTemplate("Before / after, 5 min", (
        _w("Before", "start", -300, 0), _w("After", "end", 30, 330)), builtin=True),
    SegmentTemplate("Before / after, 10 min", (
        _w("Before", "start", -600, 0), _w("After", "end", 30, 630)), builtin=True),
    SegmentTemplate("Recovery: baseline, 0–5, 10–15, 25–30 min", (
        _w("Baseline", "start", -300, 0), _w("Post 0-5", "end", 30, 330),
        _w("Post 10-15", "end", 600, 900), _w("Post 25-30", "end", 1500, 1800)),
        builtin=True),
)


def resolve(template: SegmentTemplate, evt_start: float, evt_end: "Optional[float]",
            duration: float, window_s: "Optional[float]" = None) -> "list[ResolvedWindow]":
    """The windows of *template* in recording time, for an event at
    *evt_start* .. *evt_end* (a point event when *evt_end* is None), clamped
    to [0, *duration*].

    *window_s* gives every window that length instead of the template's,
    keeping the edge nearest the event: a window before it keeps its end
    (Before −300..0 → −60..0 for 60 s), a window after it keeps its start
    (After +30..+330 → +30..+90)."""
    evt_end = evt_start if evt_end is None else evt_end
    out = []
    for w in template.windows:
        t0 = evt_start if w.ref == "start" else evt_end
        lo, hi = t0 + w.lo, t0 + w.hi
        if window_s is not None and window_s > 0:
            if w.ref == "start" and w.hi <= 0:      # before the event: keep its end
                lo = hi - window_s
            else:                                   # during / after: keep its start
                hi = lo + window_s
        clo, chi = min(max(lo, 0.0), duration), min(max(hi, 0.0), duration)
        out.append(ResolvedWindow(w.label, clo, chi, (clo, chi) != (lo, hi)))
    return out


def template_from_windows(name: str, windows: "Iterable[tuple[str, float, float]]",
                          evt_start: float, evt_end: "Optional[float]") -> SegmentTemplate:
    """A template from windows in recording time: a window that starts at or
    after the event's end is placed from the end, any other from the start
    (so a stimulus of another length keeps the post windows after it)."""
    evt_end = evt_start if evt_end is None else evt_end
    specs = []
    for label, lo, hi in windows:
        ref = "end" if lo >= evt_end else "start"
        t0 = evt_end if ref == "end" else evt_start
        specs.append(_w(label, ref, round(lo - t0, 3), round(hi - t0, 3)))
    return SegmentTemplate(name.strip(), tuple(specs))


def _to_json(t: SegmentTemplate) -> dict:
    return {"name": t.name, "windows": [
        {"label": w.label, "ref": w.ref, "lo": w.lo, "hi": w.hi} for w in t.windows]}


def _from_json(d: dict) -> "Optional[SegmentTemplate]":
    try:
        ws = tuple(_w(str(w["label"]), str(w["ref"]), float(w["lo"]), float(w["hi"]))
                   for w in d["windows"])
        name = str(d["name"]).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if (not name or not ws or len(ws) > MAX_SEGMENTS
            or any(w.ref not in REFS or not (w.hi > w.lo) for w in ws)):
        return None
    return SegmentTemplate(name, ws)


def load_templates(path: "Path | str" = TEMPLATES_PATH) -> "list[SegmentTemplate]":
    """The built-in templates, then the user's (an unreadable file or entry
    is skipped; a user template cannot take a built-in name)."""
    out = list(BUILTIN_TEMPLATES)
    names = {t.name for t in out}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for d in data.get("templates", []) if isinstance(data, dict) else []:
        t = _from_json(d) if isinstance(d, dict) else None
        if t is not None and t.name not in names:
            out.append(t)
            names.add(t.name)
    return out


def save_template(template: SegmentTemplate,
                  path: "Path | str" = TEMPLATES_PATH) -> "list[SegmentTemplate]":
    """Add *template* to the user's templates (replacing one of the same
    name). Built-in names are refused. Returns load_templates()."""
    if template.name in {t.name for t in BUILTIN_TEMPLATES}:
        raise ValueError(f"'{template.name}' is a built-in template: choose another name")
    if _from_json(_to_json(template)) is None:
        raise ValueError("a template needs a name and 1 to "
                         f"{MAX_SEGMENTS} windows, each with end > start")
    user = [t for t in load_templates(path) if not t.builtin and t.name != template.name]
    user.append(template)
    Path(path).write_text(json.dumps({"templates": [_to_json(t) for t in user]},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return load_templates(path)


def delete_template(name: str, path: "Path | str" = TEMPLATES_PATH) -> "list[SegmentTemplate]":
    user = [t for t in load_templates(path) if not t.builtin and t.name != name]
    Path(path).write_text(json.dumps({"templates": [_to_json(t) for t in user]},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return load_templates(path)


# ── quality of each segment ───────────────────────────────────────────────

def segment_quality(seg: dict) -> dict:
    """valid_pct: time covered by valid RR intervals, % of the window;
    n_pairs: Poincaré pairs; excluded_pct: RR intervals excluded, %;
    n_jumps: pairs whose two intervals differ by more than QC_JUMP_FRAC of
    the median RR (a missed or extra beat makes them, and they inflate SD1);
    sd1_robust: SD1 of the pairs less the QC_SD1_TRIM most extreme ones."""
    dur = float(seg.get("duration_s") or 0.0)
    rr = np.asarray(seg.get("rr_ms", ()), dtype=float)
    valid = float(100.0 * np.nansum(rr) / 1000.0 / dur) if dur > 0 else 0.0
    x = np.asarray(seg.get("pairs_x", ()), dtype=float)
    y = np.asarray(seg.get("pairs_y", ()), dtype=float)
    n_jumps, sd1_robust = 0, float("nan")
    if len(x) and len(x) == len(y):
        med = float(np.median(np.r_[x, y]))
        n_jumps = int(np.sum(np.abs(y - x) > QC_JUMP_FRAC * med))
        d = y - x
        # SD1 = SD(d) / sqrt 2. Trimmed of its 2 % most extreme pairs it moves
        # little on clean data (about 10 %, for a normal cloud), a lot when a
        # few missed or extra beats make it. (A MAD-based value would sit on
        # the 1 ms sampling grid.)
        dev = np.abs(d - np.median(d))
        keep = dev <= np.quantile(dev, 1.0 - QC_SD1_TRIM)
        sd1_robust = float(np.std(d[keep], ddof=1) / np.sqrt(2)) if keep.sum() > 2 else float("nan")
    return {"valid_pct": min(valid, 100.0),
            "n_pairs": int(len(x)), "n_jumps": n_jumps, "sd1_robust": sd1_robust,
            "excluded_pct": float(seg.get("excluded_pct", 0.0) or 0.0)}


def quality_flags(segs: "Sequence[dict]", event: "Optional[tuple[float, float]]" = None,
                  margin_s: float = DEFAULT_ARTEFACT_MARGIN_S) -> "list[list[str]]":
    """Per segment, what makes it doubtful: an error, little valid RR, few
    Poincaré pairs, abrupt RR jumps (missed / extra beats), a duration other than the first segment's, or an overlap
    with the event and the *margin_s* after it (stimulus artefact)."""
    ref_dur = float(segs[0].get("duration_s") or 0.0) if segs else 0.0
    out = []
    for i, s in enumerate(segs):
        flags: "list[str]" = []
        if s.get("error"):
            flags.append(str(s["error"]))
        q = segment_quality(s)
        if q["valid_pct"] < QC_MIN_VALID_PCT:
            flags.append(f"valid RR cover {q['valid_pct']:.0f} % of the window "
                         f"(< {QC_MIN_VALID_PCT:.0f} %)")
        if q["n_pairs"] < QC_MIN_PAIRS:
            flags.append(f"{q['n_pairs']} Poincaré pairs (< {QC_MIN_PAIRS})")
        if q["n_pairs"] and q["n_jumps"] > QC_MAX_JUMP_PCT / 100.0 * q["n_pairs"]:
            flags.append(f"{q['n_jumps']} abrupt RR jumps (> {QC_JUMP_FRAC:.0%} of the median): "
                         "check the peaks, they inflate SD1")
        sd1 = float(s.get("sd1", math.nan))
        rob = q["sd1_robust"]
        if np.isfinite(sd1) and np.isfinite(rob) and rob > 0 and sd1 > QC_SD1_RATIO * rob:
            flags.append(f"SD1 {sd1:.2f} ms is {sd1 / rob:.1f}× its value without the "
                         f"{QC_SD1_TRIM:.0%} most extreme pairs ({rob:.2f} ms): a few outlier "
                         "pairs drive it (check the peaks)")
        dur = float(s.get("duration_s") or 0.0)
        if i > 0 and ref_dur > 0 and abs(dur - ref_dur) > QC_DURATION_TOL * ref_dur:
            flags.append(f"lasts {dur:.0f} s, not {ref_dur:.0f} s like the first")
        if event is not None:
            a, b = event[0], event[1] + max(0.0, margin_s)
            lo, hi = float(s.get("t_lo", math.nan)), float(s.get("t_hi", math.nan))
            if lo < b and hi > a:
                flags.append(f"overlaps the event or the {margin_s:g} s after it (artefact)")
        out.append(flags)
    return out


def unique_tags(labels: "Sequence[str]") -> "list[str]":
    """Column tags of the segments: their labels, or A, B, C... when two are
    the same, one is empty, or one is a column name of the exports ("Metric",
    "Item", "Value", "Segment", or starting "Δ%")."""
    labels = [str(lbl).strip() for lbl in labels]
    reserved = {"metric", "item", "value", "segment", "subject", "file", "template"}
    if (all(labels) and len(set(labels)) == len(labels)
            and not any(lbl.lower() in reserved or lbl.startswith("Δ%") for lbl in labels)):
        return labels
    return [chr(ord("A") + i) for i in range(len(labels))]
