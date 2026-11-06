# -*- coding: utf-8 -*-
"""
ecg.io.group_stats
------------------
Group statistics on a Compare segments group table (ecg.io.group_table: one
row per recording, columns "<metric> <segment>"): the rows to check, then
for two segments A and B a paired test per metric across animals -- the
experimental unit is the animal, never the beat or the block.

Per metric: n animals with both values, medians, median and mean of the
paired differences B − A, the two-sided Wilcoxon signed-rank test (exact
for small n, scipy) and the paired t-test, and Holm-adjusted p-values over
the metrics tested. Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Per-segment columns of the group table that are not metrics.
META_PREFIXES = ("start (s)", "end (s)", "valid RR (%)", "Poincaré pairs (n)",
                 "abrupt RR jumps (n)", "QC")
ID_COLUMNS = ("subject", "file", "template")
DURATION_TOL = 0.02          # durations within 2 % of the group's median

# Metrics tested by default: one per distinct question, so that the Holm
# family is not inflated by equivalent tests. Left out on purpose: RMSSD
# (= SD1 · √2 up to the mean difference), LF n.u. and HF n.u. (HF n.u. =
# 100 − LF n.u., LF/HF a monotone function of it), pNN6 (often 0 in
# anaesthetised mice); "All metrics" tests every column.
DEFAULT_METRICS = ("Mean HR (60000/mean NN) (bpm)", "SDNN (ms)", "Poincaré SD1 (ms)",
                   "Poincaré SD2 (ms)", "LF/HF")
DEFAULT_METRICS_NOTE = ("RMSSD ≈ SD1·√2, LF n.u. / HF n.u. ≡ LF/HF and pNN6 are not tested "
                        "by default (equivalent or often 0): tick All metrics to test them")


def read_group_table(path: str) -> "pd.DataFrame":
    """The "Group" sheet of a group table written by ecg.io.group_table."""
    df = pd.read_excel(path, sheet_name="Group")
    for c in ("subject", "file"):
        if c not in df.columns:
            raise ValueError(f"{os.path.basename(path)} is not a group table (no '{c}' column)")
    return df


def segment_tags(df: "pd.DataFrame") -> "list[str]":
    """The segments of the table that hold at least one value, in column order."""
    tags = []
    for c in df.columns:
        if c.startswith("start (s) "):
            tag = c[len("start (s) "):]
            if df[c].notna().any():
                tags.append(tag)
    return tags


def metric_names(df: "pd.DataFrame", a: str, b: str) -> "list[str]":
    """Metric names (label with unit) that the table holds for both *a* and *b*."""
    out = []
    tags = [c[len("start (s) "):] for c in df.columns if c.startswith("start (s) ")]
    for c in df.columns:
        if not c.endswith(f" {a}"):
            continue
        # A column of a longer tag ending the same way ("SDNN (ms) Post A"
        # for tag "A" when "Post A" is a tag) is that tag's, not a metric.
        if any(t != a and len(t) > len(a) and c.endswith(f" {t}") for t in tags):
            continue
        name = c[: -len(a) - 1]
        if name in ID_COLUMNS or name.startswith(META_PREFIXES):
            continue
        if f"{name} {b}" in df.columns and name not in out:
            out.append(name)
    return out


@dataclass
class RowCheck:
    subject: str
    file: str
    problems: "list[str]" = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def subject_text(v: object) -> str:
    """A subject cell as text: '' for an empty cell (NaN), '243' for 243.0."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def check_rows(df: "pd.DataFrame", a: str, b: str) -> "list[RowCheck]":
    """Per row, what makes it doubtful for the A vs B comparison: a segment
    missing, a quality alert, a subject shared with another row (the unit is
    the animal), another template, or a duration other than the group's."""
    checks = []
    both = pd.Series(True, index=df.index)
    for tag in (a, b):
        col = f"start (s) {tag}"
        both &= df[col].notna() if col in df.columns else False
    # Only rows of this comparison count: another protocol's row of the same
    # animal, or its template, says nothing about this one.
    subj = df["subject"].map(subject_text)
    dup = subj.where(both).duplicated(keep=False) & both & (subj != "")
    templates = df["template"].fillna("").astype(str) if "template" in df.columns else None
    tpl_both = templates[both & (templates != "")] if templates is not None else pd.Series([], dtype=str)
    counts = tpl_both.value_counts()
    # A clear majority only: a tie names no reference.
    common_tpl = (counts.index[0] if len(counts) and (len(counts) == 1 or
                                                      counts.iloc[0] > counts.iloc[1]) else "")

    def dur(tag: str) -> "pd.Series":
        s, e = f"start (s) {tag}", f"end (s) {tag}"
        if s in df.columns and e in df.columns:
            return pd.to_numeric(df[e], errors="coerce") - pd.to_numeric(df[s], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    durs = {tag: dur(tag) for tag in (a, b)}
    med = {tag: float(np.nanmedian(d)) if d.notna().any() else math.nan for tag, d in durs.items()}
    for i, row in df.iterrows():
        rc = RowCheck(subject_text(row.get("subject")), subject_text(row.get("file")))
        if not rc.subject:
            rc.problems.append("no subject")
        elif dup.iloc[i]:
            rc.problems.append(f"subject {rc.subject} on several rows")
        for tag in (a, b):
            if pd.isna(row.get(f"start (s) {tag}")):
                rc.problems.append(f"no segment '{tag}'")
                continue
            qc = row.get(f"QC {tag}")
            if isinstance(qc, str) and qc.strip() and qc.strip().lower() != "ok":
                rc.problems.append(f"{tag}: {qc.strip()}")
            d, m = durs[tag].iloc[i], med[tag]
            if np.isfinite(d) and np.isfinite(m) and m > 0 and abs(d - m) > DURATION_TOL * m:
                rc.problems.append(f"{tag} lasts {d:g} s, not {m:g} s like the group")
        if (common_tpl and templates is not None and both.iloc[i]
                and templates.iloc[i] != common_tpl):
            rc.problems.append(f"template '{templates.iloc[i] or 'none'}', not '{common_tpl}'")
        checks.append(rc)
    return checks


def holm(p: "Sequence[float]") -> "list[float]":
    """Holm-Bonferroni adjusted p-values (NaN kept, not counted)."""
    p = [float(x) for x in p]
    idx = [i for i, x in enumerate(p) if np.isfinite(x)]
    m = len(idx)
    adj = [math.nan] * len(p)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: p[k])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def _wilcoxon(d: "np.ndarray") -> "tuple[float, float]":
    from scipy import stats
    if len(d) < 2 or np.all(d == 0):
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.wilcoxon(d, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def _ttest(d: "np.ndarray") -> float:
    from scipy import stats
    if len(d) < 2 or np.std(d, ddof=1) == 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.ttest_1samp(d, 0.0).pvalue)


def paired_stats(df: "pd.DataFrame", a: str, b: str,
                 metrics: "Optional[Sequence[str]]" = None) -> "pd.DataFrame":
    """One row per metric: n animals with both values, median A, median B,
    median and mean of B − A, Wilcoxon W and p, paired t p, and Holm-adjusted
    Wilcoxon p over the metrics tested. *metrics* None: the DEFAULT_METRICS
    the table holds (counts and durations are not outcomes, and every extra
    metric makes Holm stricter); "all": every metric of the table."""
    available = metric_names(df, a, b)
    if metrics is None:
        names = [m for m in DEFAULT_METRICS if m in available] or available
    elif metrics == "all":
        names = available
    else:
        names = list(metrics)
    rows = []
    for name in names:
        ca, cb = f"{name} {a}", f"{name} {b}"
        if ca not in df.columns or cb not in df.columns:
            continue
        va = pd.to_numeric(df[ca], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(df[cb], errors="coerce").to_numpy(float)
        ok = np.isfinite(va) & np.isfinite(vb)
        va, vb = va[ok], vb[ok]
        d = vb - va
        w, p_w = _wilcoxon(d)
        rows.append({
            "Metric": name, "n": int(ok.sum()),
            # Wilcoxon leaves out zero differences: this is its n.
            "n tested (Δ ≠ 0)": int((d != 0).sum()),
            f"Median {a}": float(np.median(va)) if len(va) else math.nan,
            f"Median {b}": float(np.median(vb)) if len(vb) else math.nan,
            "Median Δ (B − A)": float(np.median(d)) if len(d) else math.nan,
            "Mean Δ (B − A)": float(np.mean(d)) if len(d) else math.nan,
            "n up / down": f"{int((d > 0).sum())} / {int((d < 0).sum())}",
            "Wilcoxon W": w, "Wilcoxon p": p_w, "Paired t p": _ttest(d),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["Wilcoxon p (Holm)"] = holm(out["Wilcoxon p"].tolist())
    return out


def smallest_wilcoxon_p(n: int) -> float:
    """The smallest two-sided p an exact Wilcoxon signed-rank test can give
    with *n* pairs (2 / 2^n): below this the test cannot reach significance."""
    return 2.0 / (2 ** n) if n > 0 else math.nan


def write_group_report(path: str, df: "pd.DataFrame", a: str, b: str,
                       stats: "pd.DataFrame", checks: "Sequence[RowCheck]",
                       excluded: "Sequence[str]" = (), note: str = "") -> None:
    """Excel report: Statistics, Rows checked, then the paired values (one
    row per animal) of every metric tested."""
    info = [("Segments compared", f"{a} (A) vs {b} (B)"),
            ("Test", "two-sided Wilcoxon signed-rank (exact for small n); paired t for "
                     "comparison; Holm correction over the metrics"),
            ("Unit", "the animal: one row per recording, one recording per animal"),
            ("Rows left out", ", ".join(excluded) if excluded else "none")]
    n = int(stats["n tested (Δ ≠ 0)"].max()) if len(stats) else 0
    if n:
        info.append(("Smallest possible Wilcoxon p",
                     f"{smallest_wilcoxon_p(n):.3g} (n = {n} animals with a non-zero difference; "
                     "per metric: column n tested)"))
    info.append(("Metrics tested", ", ".join(stats["Metric"]) if len(stats) else "none"))
    if note:
        info.append(("Note", note))
    paired = []
    for name in stats["Metric"] if len(stats) else []:
        for _i, r in df.iterrows():
            paired.append({"Metric": name, "subject": r.get("subject"), "file": r.get("file"),
                           a: r.get(f"{name} {a}"), b: r.get(f"{name} {b}")})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        stats.to_excel(xw, sheet_name="Statistics", index=False)
        pd.DataFrame(info, columns=["Item", "Value"]).to_excel(xw, sheet_name="About", index=False)
        pd.DataFrame([{"subject": c.subject, "file": c.file,
                       "status": "ok" if c.ok else "check",
                       "problems": "; ".join(c.problems)} for c in checks]).to_excel(
            xw, sheet_name="Rows checked", index=False)
        pd.DataFrame(paired).to_excel(xw, sheet_name="Paired values", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(width + 3, 70)
