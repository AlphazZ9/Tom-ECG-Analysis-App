# -*- coding: utf-8 -*-
"""
ecg.io.group_table
------------------
A group table of Compare Segments results: one row per recording (subject,
file, the times and every metric of segment A and of segment B), so that a
before / after comparison across mice is a paired test on two columns
(Prism: a paired column table; R / Python: one row per subject). Adding a
recording already in the table (same subject and file) replaces its row.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ID_COLUMNS = ("subject", "file")


def _num(v) -> "Optional[float]":
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def group_row(subject: str, filepath: "Optional[str]", sa: dict, sb: dict,
              metrics: "Iterable[tuple[str, str, str]]") -> dict:
    """The row of one recording: subject, file, then per segment its label,
    start and end (s), and each (key, label, unit) of *metrics* as
    "<label> (<unit>) A" / "... B"."""
    row = {"subject": subject or "", "file": os.path.basename(filepath or "")}
    for tag, seg in (("A", sa), ("B", sb)):
        row[f"segment {tag}"] = seg.get("label", tag)
        row[f"start (s) {tag}"] = _num(seg.get("t_lo"))
        row[f"end (s) {tag}"] = _num(seg.get("t_hi"))
    for key, label, unit in metrics:
        name = f"{label} ({unit})" if unit else label
        for tag, seg in (("A", sa), ("B", sb)):
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
