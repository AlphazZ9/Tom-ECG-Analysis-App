# -*- coding: utf-8 -*-
"""
ecg.io.group_table
------------------
A group table of Compare segments results: one row per recording (subject,
file, template, then per segment its times, quality and every metric), so
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

ID_COLUMNS = ("subject", "file")


def _num(v) -> "Optional[float]":
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def group_row(subject: str, filepath: "Optional[str]", segs: "Sequence[dict]",
              metrics: "Iterable[tuple[str, str, str]]", template: str = "",
              flags: "Optional[Sequence[Sequence[str]]]" = None) -> dict:
    """The row of one recording: subject, file, template, then per segment
    (tagged by its label) start, end, valid RR %, Poincaré pairs and QC
    flags, then each (key, label, unit) of *metrics*."""
    tags = unique_tags([s.get("label", "") for s in segs])
    row = {"subject": subject or "", "file": os.path.basename(filepath or ""),
           "template": template or ""}
    for i, (tag, seg) in enumerate(zip(tags, segs)):
        q = segment_quality(seg)
        row[f"start (s) {tag}"] = _num(seg.get("t_lo"))
        row[f"end (s) {tag}"] = _num(seg.get("t_hi"))
        row[f"valid RR (%) {tag}"] = round(q["valid_pct"], 1)
        row[f"Poincaré pairs (n) {tag}"] = q["n_pairs"]
        row[f"abrupt RR jumps (n) {tag}"] = q["n_jumps"]
        row[f"QC {tag}"] = "; ".join(flags[i]) if flags and i < len(flags) and flags[i] else "ok"
    for key, label, unit in metrics:
        name = f"{label} ({unit})" if unit else label
        for tag, seg in zip(tags, segs):
            row[f"{name} {tag}"] = _num(seg.get(key))
    return row


def append_group_row(path: str, row: dict) -> "tuple[int, bool]":
    """Add *row* to the group table at *path* (an .xlsx, created if it does not
    exist), replacing the row of the same subject and file. Returns (rows in
    the table, whether a row was replaced). Columns new to the table are
    added; the order of the existing ones is kept."""
    if os.path.exists(path):
        df = pd.read_excel(path, sheet_name="Group")
        for c in ID_COLUMNS:
            if c not in df.columns:
                raise ValueError(f"{os.path.basename(path)} is not a group table "
                                 f"(no '{c}' column)")
        same = ((df["subject"].astype(str) == str(row["subject"]))
                & (df["file"].astype(str) == str(row["file"])))
        replaced = bool(same.any())
        df = df.loc[~same]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df, replaced = pd.DataFrame([row]), False
    df = df.sort_values(["subject", "file"], kind="stable").reset_index(drop=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Group", index=False)
    return len(df), replaced
