# -*- coding: utf-8 -*-
"""
ecg.io.segment_export
---------------------
The exports of a Compare segments result, from one place: the metric table
(each segment, Δ % against the first), the quality of each segment, the
per-block HR means, the RR series and the Poincaré pairs, as

* an Excel workbook (``write_xlsx``), or sheets added to the main workbook
  (``add_sheets``);
* a CSV of the metric table (``write_csv``) or tab-separated text for the
  clipboard (``comparison_tsv``);
* a GraphPad Prism file (``write_pzfx``), or tables for the main Prism file
  (``prism_tables``).

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ecg.core.segment_plan import segment_quality, unique_tags

# Prism XY tables are decimated past this many rows (as the main export does).
PRISM_MAX_ROWS = 20_000


@dataclass
class ComparisonResult:
    """One run of Compare segments, with what it was computed from."""
    segs: "list[dict]"                       # compute_segment_stats results, first = reference
    metrics: "Sequence[tuple[str, str, str]]"  # (key, label, unit) rows of the table
    cmps: "list[dict]" = field(default_factory=list)    # compare_segments(ref, seg_k)
    flags: "list[list[str]]" = field(default_factory=list)  # quality_flags
    template: str = ""
    subject: str = ""
    filepath: str = ""
    event: "Optional[tuple[float, float]]" = None
    artefact_margin_s: "Optional[float]" = None
    block_s: "Optional[float]" = None
    lf_band: "Optional[tuple[float, float]]" = None
    hf_band: "Optional[tuple[float, float]]" = None

    @property
    def tags(self) -> "list[str]":
        return unique_tags([s.get("label", "") for s in self.segs])


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def delta_pct(ref: float, v: float) -> float:
    """(v − ref) / |ref| × 100; NaN when either is missing or ref is 0."""
    return (100.0 * (v - ref) / abs(ref)
            if np.isfinite(ref) and np.isfinite(v) and abs(ref) > 1e-9 else float("nan"))


def comparison_frame(res: ComparisonResult) -> "pd.DataFrame":
    """One row per metric: its label (with unit), the value of each segment
    (columns named by the segment labels) and the Δ % of each against the
    first ("Δ% Ref→Seg")."""
    tags = res.tags
    rows = []
    for key, label, unit in res.metrics:
        row: "dict[str, Any]" = {"Metric": f"{label} ({unit})" if unit else label}
        vals = [_num(s.get(key)) for s in res.segs]
        for tag, v in zip(tags, vals):
            row[tag] = v
        for tag, v in zip(tags[1:], vals[1:]):
            row[f"Δ% {tags[0]}→{tag}"] = delta_pct(vals[0], v)
        rows.append(row)
    cols = ["Metric", *tags, *(f"Δ% {tags[0]}→{t}" for t in tags[1:])]
    return pd.DataFrame(rows, columns=cols)


def quality_frame(res: ComparisonResult) -> "pd.DataFrame":
    """One row per segment: its window, beats and the quality checks."""
    flags = res.flags or [[] for _ in res.segs]
    rows = []
    for tag, s, f in zip(res.tags, res.segs, flags):
        q = segment_quality(s)
        rows.append({
            "Segment": tag, "Start (s)": _num(s.get("t_lo")), "End (s)": _num(s.get("t_hi")),
            "Duration (s)": _num(s.get("duration_s")), "Beats": s.get("n_beats"),
            "Valid RR (% of window)": round(q["valid_pct"], 1),
            "Excluded RR (%)": round(q["excluded_pct"], 1),
            "Poincaré pairs": q["n_pairs"], "Abrupt RR jumps": q["n_jumps"],
            "SD1 (ms)": _num(s.get("sd1")),
            "SD1 trimmed 2 % (ms)": round(q["sd1_robust"], 3) if np.isfinite(q["sd1_robust"]) else None,
            "QC": "; ".join(f) if f else "ok",
        })
    return pd.DataFrame(rows)


def blocks_frame(res: ComparisonResult) -> "pd.DataFrame":
    """One row per block of every segment (the block comparison's data)."""
    rows = []
    for tag, s in zip(res.tags, res.segs):
        blk = s.get("blocks") or {}
        for t0, n_rr, rr_m, hr_m in zip(blk.get("t_start", []), blk.get("n_rr", []),
                                        blk.get("rr_mean", []), blk.get("hr_mean", [])):
            rows.append({"Segment": tag, "Block start (s)": round(float(t0), 2),
                         "N RR": int(n_rr), "Mean RR (ms)": round(float(rr_m), 3),
                         "HR (bpm, 60000/mean NN)": round(float(hr_m), 2)})
    return pd.DataFrame(rows, columns=["Segment", "Block start (s)", "N RR",
                                       "Mean RR (ms)", "HR (bpm, 60000/mean NN)"])


def summary_rows(res: ComparisonResult, exported_at: str = "") -> "list[tuple[str, str]]":
    """What the comparison rests on and its block-mean summary, as (item,
    value) rows: subject, file, template, event, windows, bands, block
    length, each segment against the first, and the descriptive caveat."""
    from ecg.core.analysis import segment_comparison_text
    tags = res.tags
    rows = [("Subject", res.subject or "—"),
            ("File", os.path.basename(res.filepath or "") or "—"),
            ("Template", res.template or "— (windows typed)")]
    if res.event is not None:
        rows.append(("Event (s)", f"{res.event[0]:g}–{res.event[1]:g}"))
    if res.artefact_margin_s is not None:
        rows.append(("Artefact margin (s)", f"{res.artefact_margin_s:g}"))
    for tag, s in zip(tags, res.segs):
        rows.append((f"Segment {tag} (s)", f"{_num(s.get('t_lo')):g}–{_num(s.get('t_hi')):g}"))
    rows.append(("Reference", tags[0] if tags else "—"))
    if res.lf_band:
        rows.append(("LF band (Hz)", f"{res.lf_band[0]:.2f}–{res.lf_band[1]:.2f}"))
    if res.hf_band:
        rows.append(("HF band (Hz)", f"{res.hf_band[0]:.2f}–{res.hf_band[1]:.2f}"))
    if res.block_s:
        rows.append(("Block length (s)", f"{res.block_s:g}"))
    for tag, c in zip(tags[1:], res.cmps):
        rows.append((f"{tags[0]} vs {tag}",
                     segment_comparison_text(c, tags[0], tag).split("\n")[0]))
    if res.cmps:
        rows.append(("Note", str(res.cmps[0].get("note", ""))))
    n_flag = sum(1 for f in (res.flags or []) if f)
    rows.append(("Quality", f"{n_flag} segment(s) flagged — see the Quality sheet"
                 if n_flag else "every segment passed the checks"))
    if exported_at:
        rows.append(("Exported", exported_at))
    return rows


def comparison_tsv(res: ComparisonResult) -> str:
    """The metric table as tab-separated text (paste into Excel / Prism)."""
    df = comparison_frame(res)

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return "" if not math.isfinite(v) else f"{v:.6g}"
        return str(v)
    lines = ["\t".join(df.columns)]
    lines += ["\t".join(cell(v) for v in row) for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_csv(res: ComparisonResult, path: str) -> None:
    # utf-8-sig: Excel opens a BOM-less UTF-8 CSV in the ANSI code page and
    # garbles the Δ, → and ² of the headers.
    comparison_frame(res).to_csv(path, index=False, float_format="%.6g", encoding="utf-8-sig")


# ── Excel ────────────────────────────────────────────────────────────────

def _rr_frame(s: dict) -> "pd.DataFrame":
    return pd.DataFrame({"Time (s)": np.asarray(s.get("t_rr", ()), dtype=float),
                         "RR (ms)": np.asarray(s.get("rr_ms", ()), dtype=float)})


def _poincare_frame(s: dict) -> "pd.DataFrame":
    return pd.DataFrame({"RR_n (ms)": np.asarray(s.get("pairs_x", ()), dtype=float),
                         "RR_n+1 (ms)": np.asarray(s.get("pairs_y", ()), dtype=float)})


def _sheet_name(base: str, used: "set[str]") -> str:
    """An Excel sheet name: ≤ 31 characters, none of []:*?/\\, unique."""
    name = "".join("_" if ch in '[]:*?/\\' else ch for ch in base)[:31] or "Sheet"
    stem, k = name, 2
    while name.lower() in used:
        suffix = f" ({k})"
        name = stem[:31 - len(suffix)] + suffix
        k += 1
    used.add(name.lower())
    return name


def _sheets(res: ComparisonResult, exported_at: str,
            prefix: str = "") -> "list[tuple[str, pd.DataFrame]]":
    """(title, frame) of every sheet, the comparison first."""
    out = [(f"{prefix}Comparison", comparison_frame(res)),
           (f"{prefix}Quality", quality_frame(res)),
           (f"{prefix}Summary", pd.DataFrame(summary_rows(res, exported_at),
                                             columns=["Item", "Value"])),
           (f"{prefix}Blocks", blocks_frame(res))]
    for tag, s in zip(res.tags, res.segs):
        out.append((f"{prefix}RR {tag}", _rr_frame(s)))
    for tag, s in zip(res.tags, res.segs):
        out.append((f"{prefix}Poincaré {tag}", _poincare_frame(s)))
    return out


def _format_sheet(ws: Any) -> None:
    from openpyxl.styles import Font
    ws.freeze_panes = "A2"
    for c in ws[1]:
        c.font = Font(bold=True)
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(width + 3, 70)


def write_xlsx(res: ComparisonResult, path: str, exported_at: str = "") -> None:
    """The whole comparison as a workbook: Comparison, Quality, Summary,
    Blocks, then the RR series and the Poincaré pairs of each segment."""
    used: "set[str]" = set()
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for title, df in _sheets(res, exported_at):
            df.to_excel(xw, sheet_name=_sheet_name(title, used), index=False)
        for ws in xw.book.worksheets:
            _format_sheet(ws)


def add_sheets(wb: Any, res: ComparisonResult, exported_at: str = "") -> int:
    """Append the comparison's sheets, named "Seg …", to the openpyxl
    workbook *wb* (the main export). Returns the number of sheets added."""
    used = {ws.title.lower() for ws in wb.worksheets}
    n = 0
    for title, df in _sheets(res, exported_at, prefix="Seg "):
        ws = wb.create_sheet(_sheet_name(title, used))
        ws.append([str(c) for c in df.columns])
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and not math.isfinite(v)) else
                       (v.item() if isinstance(v, np.generic) else v) for v in row])
        _format_sheet(ws)
        n += 1
    return n


# ── GraphPad Prism ───────────────────────────────────────────────────────

def _decimate(df: "pd.DataFrame", max_rows: int = PRISM_MAX_ROWS) -> "pd.DataFrame":
    if len(df) <= max_rows:
        return df
    step = int(math.ceil(len(df) / max_rows))
    return df.iloc[::step].reset_index(drop=True)


def prism_tables(res: ComparisonResult, prefix: str = ""
                 ) -> "list[tuple[str, pd.DataFrame, Any, bool]]":
    """(name, frame, x column or None, row titles) of the comparison's Prism
    tables:

    * Column "Comparison": one row per metric (row titles), one column per
      segment -- the values of this animal;
    * Column "HR block means": one column per segment, a value per block --
      plot it as a scatter / box to see the within-recording spread;
    * XY "RR <segment>": time within the segment (s) vs RR (ms);
    * XY "Poincaré <segment>": RR_n vs RR_n+1 (ms).

    The x column is given by name for an XY table and 0 (none) for a
    Column table -- pzfx counts columns from 1."""
    tags = res.tags
    out: "list[tuple[str, pd.DataFrame, Any, bool]]" = []
    table = comparison_frame(res).set_index("Metric")[tags]
    out.append((f"{prefix}Comparison", table, 0, True))
    blocks = {}
    for tag, s in zip(tags, res.segs):
        blocks[tag] = pd.Series(np.asarray((s.get("blocks") or {}).get("hr_mean", []),
                                           dtype=float))
    if any(len(v) for v in blocks.values()):
        out.append((f"{prefix}HR block means (bpm)", pd.DataFrame(blocks), 0, False))
    for tag, s in zip(tags, res.segs):
        t = np.asarray(s.get("t_rr", ()), dtype=float)
        rr = np.asarray(s.get("rr_ms", ()), dtype=float)
        if len(t):
            df = pd.DataFrame({"Time in segment (s)": t - t[0], f"RR {tag} (ms)": rr})
            out.append((f"{prefix}RR {tag}", _decimate(df), "Time in segment (s)", False))
    for tag, s in zip(tags, res.segs):
        pf = _poincare_frame(s)
        if len(pf):
            pf.columns = ["RR_n (ms)", f"RR_n+1 {tag} (ms)"]
            out.append((f"{prefix}Poincaré {tag}", _decimate(pf), "RR_n (ms)", False))
    return out


def write_pzfx(res: ComparisonResult, path: str, exported_at: str = "") -> int:
    """The comparison as a GraphPad Prism file (``prism_tables``), its
    summary and quality in the project notes. Returns the tables written.
    Raises RuntimeError when the pzfx package is missing."""
    try:
        from pzfx import write_pzfx as _write
    except ImportError as exc:
        raise RuntimeError("The 'pzfx' package is required for GraphPad Prism export.\n"
                           "Install it with: pip install pzfx") from exc
    specs = prism_tables(res)
    tables = {name: df for name, df, _x, _r in specs}
    notes_rows = summary_rows(res, exported_at)
    for _i, q in quality_frame(res).iterrows():
        notes_rows.append((f"QC {q['Segment']}", str(q["QC"])))
    notes = pd.DataFrame(notes_rows, columns=["Name", "Value"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write(tables, str(path),
               row_names=[r for _n, _d, _x, r in specs],
               x_col=[x for _n, _d, x, _r in specs],
               n_digits=4, notes={"Project Info": notes})
    return len(tables)
