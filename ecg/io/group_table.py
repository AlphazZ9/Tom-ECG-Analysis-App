# -*- coding: utf-8 -*-
"""
ecg.io.group_table
------------------
A group table of Compare segments results: one row per recording (subject,
sex, file, template, then per segment its times, quality and every metric), so
that comparing the segments across animals is a paired / repeated-measures
test on columns (Prism: a paired column table or one-way RM ANOVA; R /
Python: one row per subject). The columns are tagged with the segment
labels ("Mean HR (bpm) Before"): use the same template for every animal so
that they line up. Adding a recording already in the table (same subject
and file) replaces its row.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ecg.core.segment_plan import segment_quality, unique_tags
from ecg.io.group_stats import subject_text
from ecg.io.subject_name import normalise_sex

# The columns every group table has (the key of a row: subject and file).
# "sex" comes right after "subject" in tables made since the Sex field; an
# older table without it still opens, and gets it at its next row.
ID_COLUMNS = ("subject", "file")


def _num(v) -> "Optional[float]":
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def group_row(subject: str, filepath: "Optional[str]", segs: "Sequence[dict]",
              metrics: "Iterable[tuple[str, str, str]]", template: str = "",
              flags: "Optional[Sequence[Sequence[str]]]" = None, sex: str = "") -> dict:
    """The row of one recording: subject, sex (M / F / '', a description of
    the animal: never a metric), file, template, then per segment
    (tagged by its label) start, end, valid RR %, Poincaré pairs, QC flags
    and the label as typed, then each (key, label, unit) of *metrics*."""
    # Inner spacing does not make another segment ("Pre  5min" is "Pre 5min").
    tags = [" ".join(t.split()) for t in unique_tags([s.get("label", "") for s in segs])]
    row = {"subject": subject or "", "sex": normalise_sex(sex),
           "file": os.path.basename(filepath or ""), "template": template or ""}
    for i, (tag, seg) in enumerate(zip(tags, segs)):
        q = segment_quality(seg)
        row[f"start (s) {tag}"] = _num(seg.get("t_lo"))
        row[f"end (s) {tag}"] = _num(seg.get("t_hi"))
        row[f"valid RR (%) {tag}"] = round(q["valid_pct"], 1)
        row[f"Poincaré pairs (n) {tag}"] = q["n_pairs"]
        row[f"abrupt RR jumps (n) {tag}"] = q["n_jumps"]
        row[f"QC {tag}"] = "; ".join(flags[i]) if flags and i < len(flags) and flags[i] else "ok"
        # The label as typed: the tag is A, B, C... when labels repeat, so
        # this is what tells whether segment A is the same window in every
        # animal (ecg.io.group_stats.label_problems).
        row[f"label {tag}"] = str(seg.get("label", "") or "").strip()
        # Its place in the recording (1, 2, 3...): all pairs mode pairs the
        # segments of each row by this, not by the column order of the table.
        row[f"index {tag}"] = i + 1
    for key, label, unit in metrics:
        name = f"{label} ({unit})" if unit else label
        for tag, seg in zip(tags, segs):
            row[f"{name} {tag}"] = _num(seg.get(key))
    return row


def _norm(tag: str) -> str:
    return " ".join(str(tag).split()).casefold()


def _row_tags(keys: "Iterable[str]") -> "list[str]":
    return [k[len("start (s) "):] for k in keys if str(k).startswith("start (s) ")]


def match_existing_tags(row: dict, columns: "Sequence[str]") -> dict:
    """*row* with each segment tag new to the table renamed to the table's
    tag that differs from it only by case or spacing ("pre stim 5min" goes
    under "Pre Stim 5min"), so that a typing slip does not make a column of
    its own. The label as typed is kept in "label <tag>"."""
    existing = _row_tags(columns)
    mine = _row_tags(row)
    rename = {}
    for t in mine:
        if t in existing:
            continue
        same = [e for e in existing if _norm(e) == _norm(t) and e not in mine
                and e not in rename.values()]
        if len(same) == 1:
            rename[t] = same[0]
    if not rename:
        return row
    out = {}
    for k, v in row.items():
        owner = max((t for t in mine if str(k).endswith(f" {t}")), key=len, default=None)
        if owner in rename:
            k = k[: -len(owner)] + rename[owner]
        out[k] = v
    return out


def append_group_row(path: str, row: dict) -> "tuple[int, bool]":
    """Add *row* to the group table at *path* (an .xlsx, created if it does not
    exist), replacing the row of the same subject and file. Returns (rows in
    the table, whether a row was replaced). Columns new to the table are
    added; the order of the existing ones is kept."""
    every = pd.read_excel(path, sheet_name=None) if os.path.exists(path) else {}
    # The other sheets of the table (ecg.io.group_data: Animals, Whole,
    # TimeCourse...) are written back as they were.
    others = {k: v for k, v in every.items() if k != "Group"}
    if every and "Group" not in every and not set(every) & {
            "Animals", "Whole", "TimeCourse", "Annotations", "Events"}:
        raise ValueError(f"{os.path.basename(path)} is not a group table (no 'Group' sheet)")
    if "Group" in every:
        df = every["Group"]
        for c in ID_COLUMNS:
            if c not in df.columns:
                raise ValueError(f"{os.path.basename(path)} is not a group table "
                                 f"(no '{c}' column)")
        # As text the way the table shows it: a subject read back as 243.0
        # (a column with an empty cell is float) is "243", not "243.0", else
        # the recording would come in a second time instead of replacing.
        same = ((df["subject"].map(subject_text) == subject_text(row["subject"]))
                & (df["file"].map(subject_text) == subject_text(row["file"])))
        replaced = bool(same.any())
        row = match_existing_tags(row, list(df.columns))
        df = df.loc[~same]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df, replaced = pd.DataFrame([row]), False
    # A table made before the Sex field gets it beside the subject.
    if "sex" in df.columns and list(df.columns).index("sex") != 1:
        cols = [c for c in df.columns if c != "sex"]
        cols.insert(cols.index("subject") + 1, "sex")
        df = df[cols]
    df = df.sort_values(["subject", "file"], kind="stable").reset_index(drop=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, other in others.items():
            if name == "Animals":
                other.to_excel(xw, sheet_name=name, index=False)
        df.to_excel(xw, sheet_name="Group", index=False)
        for name, other in others.items():
            if name != "Animals":
                other.to_excel(xw, sheet_name=name, index=False)
    return len(df), replaced
