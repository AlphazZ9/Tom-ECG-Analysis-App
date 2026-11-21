# -*- coding: utf-8 -*-
"""
ecg.io.group_compare
--------------------
The analyses of the Group tab on a group table (ecg.io.group_data): the
animals split by their group, their sex or not at all, and for each kind of
data (Pre/Post segments, Time course, Whole recording, Event-locked):

* the mean ± SEM (n) of each group;
* within each group, the paired test of the condition against the
  reference (Wilcoxon signed-rank, exact for small n, and the paired t);
* between the groups, on the change (or the value for a whole recording):
  Mann-Whitney and Welch's t for two groups, Kruskal-Wallis and one-way
  ANOVA for more;
* the two-way mixed ANOVA (group between, segment / time within, the
  Greenhouse-Geisser correction for time), on the animals with every value;
* Holm over the metrics tested, within each family of tests.

The experimental unit is the animal. The results carry the per-animal
values, the tables for GraphPad Prism and the data of the figure.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ecg.io.group_data import (
    SHEET_EVENTS, SHEET_SEGMENTS, SHEET_TIME, SHEET_WHOLE, TC_METRICS, KEY, binned_series,
    onset_of, stim_onsets, window_value, SHEET_ANNOT,
)
from ecg.io.group_stats import (
    DEFAULT_METRICS, _ttest, _wilcoxon, holm, metric_names, segment_tags,
    smallest_wilcoxon_p,
)

BY_NONE = "No groups"
BY_GROUP = "Group"
BY_SEX = "Sex"
BY_CHOICES = (BY_NONE, BY_GROUP, BY_SEX)

AN_SEGMENTS = "Pre/Post segments"
AN_TIME = "Time course"
AN_WHOLE = "Whole recording"
AN_EVENTS = "Event-locked"
ANALYSES = (AN_SEGMENTS, AN_TIME, AN_WHOLE, AN_EVENTS)

ALL_LABEL = "All animals"
CONTROL_NAMES = ("control", "ctrl", "ctl", "vehicle", "veh", "saline", "sham", "wt",
                 "wild", "témoin", "temoin", "sal")
SEX_NAMES = {"M": "Males", "F": "Females"}


# ── groups ───────────────────────────────────────────────────────────────

def animal_groups(animals: "pd.DataFrame", by: str) -> "tuple[dict[tuple[str, str], str], list[str], list[str]]":
    """({(subject, file): group name} of the animals used, the group names in
    order, notes on the animals left out). *by*: BY_NONE, BY_GROUP, BY_SEX."""
    out: "dict[tuple[str, str], str]" = {}
    notes: "list[str]" = []
    for _, r in animals.iterrows():
        if not bool(r.get("use", True)):
            continue
        key = (str(r["subject"]), str(r["file"]))
        if by == BY_SEX:
            sex = str(r.get("sex", "") or "")
            if sex not in SEX_NAMES:
                notes.append(f"{r['subject']}: sex unknown, left out")
                continue
            out[key] = SEX_NAMES[sex]
        elif by == BY_GROUP:
            g = str(r.get("group", "") or "").strip()
            if not g:
                notes.append(f"{r['subject']}: no group, left out")
                continue
            out[key] = g
        else:
            out[key] = ALL_LABEL
    if by == BY_SEX:
        order = [n for n in ("Males", "Females") if n in out.values()]
    else:
        # The control group first (the reference of the figure and tables).
        order = sorted(set(out.values()),
                       key=lambda s: (not s.casefold().startswith(CONTROL_NAMES), s.casefold()))
    return out, order, notes


# ── statistics ───────────────────────────────────────────────────────────

def _num(v: Any) -> float:
    """*v* as a float, NaN when it is not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _need_animals(gmap: "Mapping", by: str) -> None:
    """No animal left once grouped: say why (rather than "no data")."""
    if gmap:
        return
    if by == BY_GROUP:
        raise ValueError("No animal has a group: type one in Animals, or choose No groups.")
    if by == BY_SEX:
        raise ValueError("No animal has a sex (M / F): set it in Animals, or choose No groups.")
    raise ValueError("No animal is ticked in Animals.")


def describe(v: "Sequence[float]") -> "tuple[float, float, int]":
    """(mean, SEM, n) of the finite values of *v*."""
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return math.nan, math.nan, 0
    sem = float(np.std(a, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return float(np.mean(a)), sem, n


def between_test(samples: "Sequence[Sequence[float]]") -> "dict[str, Any]":
    """Rank and parametric tests between independent groups: Mann-Whitney
    and Welch's t for two, Kruskal-Wallis and one-way ANOVA for more."""
    from scipy import stats
    xs = [np.asarray(s, dtype=float) for s in samples]
    xs = [x[np.isfinite(x)] for x in xs]
    out: "dict[str, Any]" = {"rank test": "", "p": math.nan, "param test": "", "p param": math.nan}
    if len(xs) < 2 or any(len(x) < 1 for x in xs) or sum(len(x) for x in xs) < 3:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(xs) == 2:
            out["rank test"] = "Mann-Whitney"
            try:
                out["p"] = float(stats.mannwhitneyu(xs[0], xs[1], alternative="two-sided").pvalue)
            except ValueError:
                pass
            out["param test"] = "Welch t"
            if all(len(x) >= 2 for x in xs):
                out["p param"] = float(stats.ttest_ind(xs[0], xs[1], equal_var=False).pvalue)
        else:
            out["rank test"] = "Kruskal-Wallis"
            try:
                out["p"] = float(stats.kruskal(*xs).pvalue)
            except ValueError:
                pass
            out["param test"] = "one-way ANOVA"
            if all(len(x) >= 2 for x in xs):
                out["p param"] = float(stats.f_oneway(*xs).pvalue)
    for k in ("p", "p param"):
        if not math.isfinite(out[k]):
            out[k] = math.nan
    return out


def within_test(ref: "Sequence[float]", cond: "Sequence[float]") -> "dict[str, Any]":
    """Paired tests of *cond* vs *ref* (same animals, in order): n, Wilcoxon
    p and paired t p on the finite pairs."""
    a = np.asarray(ref, dtype=float)
    b = np.asarray(cond, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    _, p = _wilcoxon(d)
    return {"n": int(ok.sum()), "p": p, "p t": _ttest(d)}


def _gg_epsilon(y: np.ndarray, g: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon of the within factor (columns of *y*) from
    the covariance pooled within the groups *g*."""
    n, k = y.shape
    if k < 3:
        return 1.0
    levels = list(dict.fromkeys(g.tolist()))
    resid = np.vstack([y[g == lv] - y[g == lv].mean(axis=0) for lv in levels])
    dfe = n - len(levels)
    if dfe < 1:
        return math.nan
    s = resid.T @ resid / dfe
    # Orthonormal contrasts orthogonal to the mean (Helmert).
    c = np.zeros((k - 1, k))
    for i in range(1, k):
        c[i - 1, :i] = 1.0
        c[i - 1, i] = -i
        c[i - 1] /= np.linalg.norm(c[i - 1])
    m = c @ s @ c.T
    tr = float(np.trace(m))
    tr2 = float(np.trace(m @ m))
    if tr2 <= 0:
        return math.nan
    return max(1.0 / (k - 1), min(1.0, tr * tr / ((k - 1) * tr2)))


def mixed_anova(y: "np.ndarray", groups: "Sequence[str]") -> "pd.DataFrame":
    """Two-way mixed ANOVA on *y* (animals × levels of the within factor,
    no missing value) with *groups* the between factor (one group: the
    one-way repeated-measures ANOVA). One row per effect: Source, df1, df2,
    F, p, epsilon (Greenhouse-Geisser, within effects) and p GG."""
    from scipy import stats
    y = np.asarray(y, dtype=float)
    g = np.asarray(list(groups), dtype=object)
    n, k = y.shape
    levels = list(dict.fromkeys(g.tolist()))
    a = len(levels)
    rows: "list[dict]" = []
    if n - a < 1 or k < 2:
        return pd.DataFrame(rows, columns=["Source", "df1", "df2", "F", "p", "epsilon", "p GG"])
    grand = y.mean()
    subj = y.mean(axis=1)
    lev = y.mean(axis=0)
    ss_subj = k * float(((subj - grand) ** 2).sum())
    ss_t = n * float(((lev - grand) ** 2).sum())
    ss_total = float(((y - grand) ** 2).sum())
    ss_g = 0.0
    ss_gt = 0.0
    for lv in levels:
        yg = y[g == lv]
        ng = len(yg)
        mg = yg.mean()
        ss_g += k * ng * (mg - grand) ** 2
        ss_gt += ng * float(((yg.mean(axis=0) - mg - lev + grand) ** 2).sum())
    ss_sg = ss_subj - ss_g
    ss_err = ss_total - ss_subj - ss_t - ss_gt
    df_sg, df_t = n - a, k - 1
    df_err = df_sg * df_t
    eps = _gg_epsilon(y, g)

    def _row(src: str, ss: float, df1: float, sse: float, df2: float, within: bool) -> dict:
        f = (ss / df1) / (sse / df2) if df1 > 0 and df2 > 0 and sse > 0 else math.nan
        p = float(stats.f.sf(f, df1, df2)) if math.isfinite(f) else math.nan
        e = eps if within else math.nan
        p_gg = (float(stats.f.sf(f, df1 * e, df2 * e))
                if within and math.isfinite(f) and math.isfinite(e) else math.nan)
        return {"Source": src, "df1": df1, "df2": df2, "F": f, "p": p, "epsilon": e,
                "p GG": p_gg}

    if a > 1:
        rows.append(_row("Group", ss_g, a - 1, ss_sg, df_sg, False))
    rows.append(_row("Within", ss_t, df_t, ss_err, df_err, True))
    if a > 1:
        rows.append(_row("Group × within", ss_gt, (a - 1) * df_t, ss_err, df_err, True))
    return pd.DataFrame(rows)


def fmt_p(p: float) -> str:
    if p is None or not math.isfinite(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def fmt_mean(m: float, sem: float, n: int, decimals: int = 2) -> str:
    if not math.isfinite(m):
        return "—"
    s = f"{m:.{decimals}f}"
    if math.isfinite(sem):
        s += f" ± {sem:.{decimals}f}"
    return f"{s} ({n})"


def _decimals(values: "Sequence[float]") -> int:
    a = np.abs(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    if not len(a):
        return 2
    m = float(np.median(a))
    return 0 if m >= 100 else 1 if m >= 10 else 2 if m >= 1 else 3


# ── results ──────────────────────────────────────────────────────────────

@dataclass
class GroupResult:
    analysis: str
    by: str
    groups: "list[str]"
    # Results table of the tab: Test, one column per group, p, p Holm (text).
    display: "pd.DataFrame"
    # Every number behind it (Excel).
    stats: "pd.DataFrame"
    # One row per animal: subject, file, group, the values used.
    per_animal: "pd.DataFrame"
    anova: "pd.DataFrame" = field(default_factory=pd.DataFrame)
    notes: "list[str]" = field(default_factory=list)
    prism: "list[tuple]" = field(default_factory=list)
    figure: "dict[str, Any]" = field(default_factory=dict)
    settings: "list[tuple[str, Any]]" = field(default_factory=list)

    @property
    def n_animals(self) -> int:
        return int(len(self.per_animal))


def _row_title(subject: str, sex: str) -> str:
    return f"{subject} ({sex})" if sex in ("M", "F") else subject


def _sexes(animals: "pd.DataFrame") -> "dict[tuple[str, str], str]":
    return {(str(r["subject"]), str(r["file"])): str(r.get("sex", "") or "")
            for _, r in animals.iterrows()}


def _small_n_note(ns: "Sequence[int]") -> "Optional[str]":
    ns = [n for n in ns if n > 0]
    if not ns or min(ns) > 5:
        return None
    n = min(ns)
    return (f"n = {n}: the smallest Wilcoxon p is {smallest_wilcoxon_p(n):.3g}; "
            "see also the paired t (Statistics… workbook)")


def _column_table(title: str, cols: "Mapping[str, Sequence[float]]") -> tuple:
    n = max((len(v) for v in cols.values()), default=0)
    frame = pd.DataFrame({k: list(v) + [math.nan] * (n - len(v)) for k, v in cols.items()})
    return (title, frame)


def _paired_table(title: str, rows: "Sequence[str]", cols: "Mapping[str, Sequence[float]]") -> tuple:
    frame = pd.DataFrame({k: list(v) for k, v in cols.items()}, index=list(rows))
    return (title, frame)


def _holm_column(df: "pd.DataFrame", col: str, out: str) -> None:
    if col in df.columns and len(df):
        df[out] = holm(df[col].tolist())


def _two_condition(analysis: str, by: str, order: "list[str]", gmap: "dict",
                   sexes: "dict", values: "dict[str, dict[tuple, tuple[float, float]]]",
                   names: "tuple[str, str]", notes: "list[str]",
                   short: "Mapping[str, str]", within: str = "Segment") -> GroupResult:
    """Shared by Pre/Post segments and Event-locked: per metric and animal a
    (reference, condition) pair; within-group paired tests, between-group
    tests on Δ, mixed ANOVA group × condition."""
    na, nb = names
    stat_rows: "list[dict]" = []
    disp: "list[dict]" = []
    anova_rows: "list[pd.DataFrame]" = []
    per_animal: "dict[tuple, dict]" = {}
    prism: "list[tuple]" = []
    ns: "list[int]" = []
    for metric, vals in values.items():
        keys = [k for k in gmap if k in vals]
        row: "dict[str, Any]" = {"Metric": metric}
        dec = _decimals([v for k in keys for v in vals[k]])
        dec_d = _decimals([vals[k][1] - vals[k][0] for k in keys])
        deltas: "dict[str, list[float]]" = {}
        for g in order:
            ks = [k for k in keys if gmap[k] == g]
            a = [vals[k][0] for k in ks]
            b = [vals[k][1] for k in ks]
            d = [y - x for x, y in zip(a, b)]
            deltas[g] = d
            for lab, v in ((na, a), (nb, b), ("Δ", d)):
                m, s, n = describe(v)
                row[f"{g} {lab} mean"], row[f"{g} {lab} SEM"], row[f"{g} {lab} n"] = m, s, n
            w = within_test(a, b)
            row[f"{g} n pairs"] = w["n"]
            row[f"{g} Wilcoxon p"] = w["p"]
            row[f"{g} paired t p"] = w["p t"]
            ns.append(w["n"])
            for k in ks:
                pa = per_animal.setdefault(k, {"subject": k[0], "file": k[1], "group": g})
                pa[f"{short.get(metric, metric)} {na}"] = vals[k][0]
                pa[f"{short.get(metric, metric)} {nb}"] = vals[k][1]
            prism.append(_paired_table(
                f"{metric}{' · ' + g if len(order) > 1 else ''}",
                [_row_title(k[0], sexes.get(k, "")) for k in ks], {na: a, nb: b}))
        if len(order) > 1:
            bt = between_test([deltas[g] for g in order])
            row["between test"], row["between p"] = bt["rank test"], bt["p"]
            row["between param test"], row["between param p"] = bt["param test"], bt["p param"]
            prism.append(_column_table(f"Δ {metric} ({nb} − {na})", deltas))
        complete = [k for k in keys if all(math.isfinite(x) for x in vals[k])]
        if len(complete) >= len(order) + 1:
            y = np.array([vals[k] for k in complete], dtype=float)
            an = mixed_anova(y, [gmap[k] for k in complete])
            if len(an):
                an.insert(0, "Metric", metric)
                an["Source"] = an["Source"].replace({"Within": within,
                                                     "Group × within": f"Group × {within.lower()}"})
                anova_rows.append(an)
        stat_rows.append(row)
        row["_dec"], row["_dec_d"] = dec, dec_d
    stats = pd.DataFrame(stat_rows)
    for g in order:
        _holm_column(stats, f"{g} Wilcoxon p", f"{g} Wilcoxon p (Holm)")
    if len(order) > 1:
        _holm_column(stats, "between p", "between p (Holm)")
    for _, r in stats.iterrows():
        dec = int(r["_dec"])
        d1 = {"Test": f"Δ {r['Metric']} ({nb} − {na})"}
        if len(order) > 1:
            for g in order:
                d1[g] = fmt_mean(r[f"{g} Δ mean"], r[f"{g} Δ SEM"], int(r[f"{g} Δ n"]),
                                 int(r["_dec_d"]))
            d1["p"] = f"{fmt_p(r['between p'])} ({r['between test']})"
            d1["p Holm"] = fmt_p(r["between p (Holm)"])
            d1["_p"] = r["between p (Holm)"]
            disp.append(d1)
        d2 = {"Test": f"{r['Metric']}: {nb} vs {na}"}
        for g in order:
            d2[g] = (f"{fmt_mean(r[f'{g} {na} mean'], r[f'{g} {na} SEM'], int(r[f'{g} {na} n']), dec)}"
                     f" → {fmt_mean(r[f'{g} {nb} mean'], r[f'{g} {nb} SEM'], int(r[f'{g} {nb} n']), dec)}"
                     f"   p = {fmt_p(r[f'{g} Wilcoxon p (Holm)'])}")
        d2["p"] = "within group: Wilcoxon (Holm)"
        d2["p Holm"] = ""
        d2["_p"] = min([r[f"{g} Wilcoxon p (Holm)"] for g in order
                        if math.isfinite(r[f"{g} Wilcoxon p (Holm)"])], default=math.nan)
        disp.append(d2)
    stats = stats.drop(columns=["_dec", "_dec_d"], errors="ignore")
    note = _small_n_note(ns)
    if note:
        notes.append(note)
    anova = pd.concat(anova_rows, ignore_index=True) if anova_rows else pd.DataFrame()
    pa = pd.DataFrame(list(per_animal.values()))
    return GroupResult(analysis=analysis, by=by, groups=order, display=_display(disp, order),
                       stats=stats, per_animal=pa, anova=anova, notes=notes, prism=prism)


def _display(rows: "list[dict]", order: "list[str]") -> "pd.DataFrame":
    cols = ["Test", *order, "p", "p Holm", "_p"]
    return pd.DataFrame(rows, columns=cols)


# ── the four analyses ────────────────────────────────────────────────────

def analyse_segments(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                     a: str, b: str, metrics: "Optional[Sequence[str]]" = None,
                     names: "tuple[str, str]" = ("", "")) -> GroupResult:
    """Pre/Post segments: segment *b* vs *a* of the Group sheet, per metric
    (*metrics* None: DEFAULT_METRICS the table holds; "all": all)."""
    df = sheets.get(SHEET_SEGMENTS)
    if df is None or df.empty:
        raise ValueError("The group table has no segments (Compare segments → Add to group table).")
    tags = segment_tags(df)
    if a not in tags or b not in tags or a == b:
        raise ValueError("Choose two different segments of the table.")
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by)
    available = metric_names(df, a, b)
    if metrics is None:
        names_m = [m for m in DEFAULT_METRICS if m in available] or available
    elif metrics == "all":
        names_m = available
    else:
        names_m = [m for m in metrics if m in available]
    values: "dict[str, dict]" = {}
    for m in names_m:
        vals = {}
        for _, r in df.iterrows():
            key = (str(r["subject"]), str(r["file"]))
            if key not in gmap:
                continue
            va = _num(r.get(f"{m} {a}"))
            vb = _num(r.get(f"{m} {b}"))
            if math.isfinite(va) and math.isfinite(vb):
                vals[key] = (float(va), float(vb))
        values[m] = vals
    missing = [k[0] for k in gmap if not any(k in v for v in values.values())]
    if missing:
        notes.append(f"No {a} / {b} segments: {', '.join(missing)}")
    na, nb = names[0] or a, names[1] or b
    res = _two_condition(AN_SEGMENTS, by, order, gmap, _sexes(animals), values, (na, nb),
                         notes, {})
    res.figure = {"kind": "paired", "values": values, "gmap": gmap, "names": (na, nb),
                  "metric": names_m[0] if names_m else ""}
    res.settings = [("Analysis", AN_SEGMENTS), ("By", by), ("Segments", f"{b} vs {a}")]
    return res


def whole_metrics(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    df = sheets.get(SHEET_WHOLE)
    if df is None:
        return []
    return [c for c in df.columns if c not in ("subject", "sex", "group", "use", "file")]


def analyse_whole(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                  metrics: "Optional[Sequence[str]]" = None) -> GroupResult:
    """Whole recording: each metric compared between the groups."""
    df = sheets.get(SHEET_WHOLE)
    if df is None or df.empty:
        raise ValueError("The group table has no whole-recording metrics "
                         "(Group tab → Add open recording, after Analyze).")
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by)
    available = whole_metrics(sheets)
    if metrics is None:
        names_m = [m for m in DEFAULT_METRICS if m in available] or available[:8]
    elif metrics == "all":
        names_m = available
    else:
        names_m = [m for m in metrics if m in available]
    rows, disp, prism = [], [], []
    per_animal: "dict[tuple, dict]" = {}
    values: "dict[str, dict]" = {}
    for m in names_m:
        vals = {}
        for _, r in df.iterrows():
            key = (str(r["subject"]), str(r["file"]))
            v = _num(r.get(m))
            if key in gmap and math.isfinite(v):
                vals[key] = float(v)
                per_animal.setdefault(key, {"subject": key[0], "file": key[1],
                                            "group": gmap[key]})[m] = float(v)
        values[m] = vals
        row: "dict[str, Any]" = {"Metric": m}
        by_g = {g: [vals[k] for k in gmap if gmap[k] == g and k in vals] for g in order}
        for g in order:
            mm, s, n = describe(by_g[g])
            row[f"{g} mean"], row[f"{g} SEM"], row[f"{g} n"] = mm, s, n
        if len(order) > 1:
            bt = between_test([by_g[g] for g in order])
            row.update({"between test": bt["rank test"], "between p": bt["p"],
                        "between param test": bt["param test"], "between param p": bt["p param"]})
        rows.append(row)
        prism.append(_column_table(m, by_g))
    stats = pd.DataFrame(rows)
    if len(order) > 1:
        _holm_column(stats, "between p", "between p (Holm)")
    for _, r in stats.iterrows():
        dec = _decimals([r[f"{g} mean"] for g in order])
        d = {"Test": str(r["Metric"])}
        for g in order:
            d[g] = fmt_mean(r[f"{g} mean"], r[f"{g} SEM"], int(r[f"{g} n"]), dec)
        if len(order) > 1:
            d["p"] = f"{fmt_p(r['between p'])} ({r['between test']})"
            d["p Holm"] = fmt_p(r["between p (Holm)"])
            d["_p"] = r["between p (Holm)"]
        else:
            d["p"], d["p Holm"], d["_p"] = "descriptive", "", math.nan
        disp.append(d)
    if len(order) < 2:
        notes.append("One group: descriptive only (choose By Group or Sex to compare)")
    res = GroupResult(analysis=AN_WHOLE, by=by, groups=order, display=_display(disp, order),
                      stats=stats, per_animal=pd.DataFrame(list(per_animal.values())),
                      notes=notes, prism=prism)
    res.figure = {"kind": "dots", "values": values, "gmap": gmap,
                  "metric": names_m[0] if names_m else ""}
    res.settings = [("Analysis", AN_WHOLE), ("By", by)]
    return res


def event_labels(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    df = sheets.get(SHEET_EVENTS)
    if df is None or df.empty or "label" not in df.columns:
        return []
    return sorted({str(v) for v in df["label"] if str(v) not in ("", "nan")}, key=str.casefold)


def event_metrics(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    df = sheets.get(SHEET_EVENTS)
    if df is None:
        return []
    return [c[:-4] for c in df.columns if str(c).endswith(" pre") and f"{c[:-4]} post" in df.columns]


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "")
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return bool(v)


def _config_text(method: Any, window: Any) -> str:
    m = "" if method is None or (isinstance(method, float) and not math.isfinite(method)) \
        else str(method).strip()
    w = _num(window)
    return " · ".join(p for p in (m or "default", f"{w:g} s" if math.isfinite(w) else "") if p)


def event_configs(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    """The event-locked configurations of the table ("robust · 15 s"), the
    most used first: one recording can hold the same events analysed with
    several methods or windows, which must not be pooled."""
    df = sheets.get(SHEET_EVENTS)
    if df is None or df.empty:
        return []
    texts = [_config_text(m, w) for m, w in zip(df.get("method", [None] * len(df)),
                                                 df.get("window_s", [None] * len(df)))]
    counts = pd.Series(texts).value_counts()
    return sorted(counts.index, key=lambda t: (-int(counts[t]), t))


def analyse_events(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                   label: str = "", metrics: "Optional[Sequence[str]]" = None,
                   config: "Optional[str]" = None) -> GroupResult:
    """Event-locked: per animal the mean of its events named *label* ('' :
    all) that are valid for each metric, in one configuration (*config*, one
    of event_configs(); None: the most used), post vs pre within each group
    and Δ between groups."""
    df = sheets.get(SHEET_EVENTS)
    if df is None or df.empty:
        raise ValueError("The group table has no event-locked results (save them to the event "
                         "registry in Time course → Autonomic, then Add open recording).")
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by)
    ev = df
    configs = event_configs(sheets)
    config = config if config in configs else configs[0]
    ev = ev.loc[[_config_text(m, w) == config for m, w in
                 zip(ev.get("method", [None] * len(ev)), ev.get("window_s", [None] * len(ev)))]]
    if len(configs) > 1:
        notes.append(f"Configuration {config} only (the table also holds "
                     f"{', '.join(c for c in configs if c != config)})")
    if label:
        ev = ev.loc[ev["label"].astype(str) == label]
    available = event_metrics(sheets)
    names_m = list(metrics) if metrics not in (None, "all") else available
    names_m = [m for m in names_m if m in available]
    if metrics is None:
        pref = [m for m in ("rCSI", "rCPI", "HR", "SD1", "SD2") if m in names_m]
        names_m = pref or names_m[:5]
    values: "dict[str, dict]" = {}
    for m in names_m:
        vals = {}
        for (s, f), sub in ev.groupby(list(KEY), sort=False):
            key = (str(s), str(f))
            if key not in gmap:
                continue
            pre = pd.to_numeric(sub[f"{m} pre"], errors="coerce").to_numpy(float)
            post = pd.to_numeric(sub[f"{m} post"], errors="coerce").to_numpy(float)
            # The series' own validity (tables made before it: the event's).
            vcol = f"{m} valid" if f"{m} valid" in sub.columns else "valid"
            valid = (np.array([_truthy(v) for v in sub[vcol]], dtype=bool)
                     if vcol in sub.columns else np.ones(len(sub), bool))
            ok = np.isfinite(pre) & np.isfinite(post) & valid
            if ok.any():
                vals[key] = (float(pre[ok].mean()), float(post[ok].mean()))
        values[m] = vals
    notes.append("Per animal: the mean of its events" + (f" “{label}”" if label else ""))
    res = _two_condition(AN_EVENTS, by, order, gmap, _sexes(animals), values, ("pre", "post"),
                         notes, {}, within="Pre/post")
    res.figure = {"kind": "paired", "values": values, "gmap": gmap, "names": ("pre", "post"),
                  "metric": names_m[0] if names_m else ""}
    res.settings = [("Analysis", AN_EVENTS), ("By", by), ("Events", label or "all"),
                    ("Configuration", config)]
    return res


def analyse_time(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                 stim: str, metric: str = "HR", bin_s: float = 60.0,
                 start_min: float = -10.0, end_min: float = 30.0,
                 window_min: "tuple[float, float]" = (0.0, 5.0), delta: bool = True,
                 occurrence: int = 0, min_coverage: float = 0.5) -> GroupResult:
    """Time course aligned on the annotation *stim* of each animal: *metric*
    (HR, RR, SDNN, RMSSD) in bins of *bin_s* from *start_min* to *end_min*
    min, the baseline being every bin before 0 (Δ from it with *delta*);
    the test window *window_min* against the baseline within each group,
    its Δ between groups; the mixed ANOVA group × time on the bins."""
    tc = sheets.get(SHEET_TIME)
    if tc is None or tc.empty:
        raise ValueError("The group table has no time course (Group tab → Add open recording, "
                         "after Analyze).")
    if metric not in TC_METRICS:
        raise ValueError(f"Unknown metric {metric}")
    if not (start_min < 0 < end_min):
        raise ValueError("The time course must start before 0 and end after it.")
    w0, w1 = window_min
    if not (w1 > w0):
        raise ValueError("The test window must end after it starts.")
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by)
    onsets = stim_onsets(sheets.get(SHEET_ANNOT))
    sexes = _sexes(animals)
    unit = TC_METRICS[metric]
    series: "dict[tuple, np.ndarray]" = {}
    base_v: "dict[tuple, float]" = {}
    win_v: "dict[tuple, float]" = {}
    starts = None
    no_stim, no_tc = [], []
    for (s, f), sub in tc.groupby(list(KEY), sort=False):
        key = (str(s), str(f))
        if key not in gmap:
            continue
        t0 = onset_of(onsets.get(key, []), stim, occurrence)
        if t0 is None:
            no_stim.append(key[0])
            continue
        st, v = binned_series(sub, metric, t0, start_min * 60, end_min * 60, bin_s, min_coverage)
        rel = sub.assign(t_s=sub["t_s"].to_numpy(float) - t0)
        base = window_value(rel, metric, start_min * 60, 0.0, min_coverage)
        win = window_value(rel, metric, w0 * 60, w1 * 60, min_coverage)
        starts = st
        series[key] = v - base if delta else v
        base_v[key], win_v[key] = base, win
    for k in gmap:
        if k not in series and k[0] not in no_stim and not ((tc["subject"] == k[0])
                                                          & (tc["file"] == k[1])).any():
            no_tc.append(k[0])
    if no_stim:
        notes.append(f"No “{stim}” annotation, left out: {', '.join(no_stim)}")
    if no_tc:
        notes.append(f"No time course, left out: {', '.join(no_tc)}")
    if starts is None:
        raise ValueError(f"No animal has both a time course and a “{stim}” annotation.")
    t_min = (starts + bin_s / 2) / 60.0
    label = f"{'Δ ' if delta else ''}{metric} ({unit})"
    disp, rows, prism = [], [], []
    curves: "dict[str, dict]" = {}
    ns = []
    deltas: "dict[str, list[float]]" = {}
    row: "dict[str, Any]" = {"Metric": f"{metric} ({unit})",
                             "window (min)": f"{w0:g}–{w1:g}", "baseline (min)": f"{start_min:g}–0"}
    for g in order:
        ks = [k for k in series if gmap[k] == g]
        mat = np.array([series[k] for k in ks]) if ks else np.empty((0, len(starts)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.nanmean(mat, axis=0) if len(ks) else np.full(len(starts), np.nan)
            cnt = np.sum(np.isfinite(mat), axis=0) if len(ks) else np.zeros(len(starts))
            sd = np.nanstd(mat, axis=0, ddof=1) if len(ks) > 1 else np.full(len(starts), np.nan)
        sem = np.where(cnt > 1, sd / np.sqrt(np.maximum(cnt, 1)), np.nan)
        # One curve per recording (a subject with two files keeps both), and
        # n the animals that add at least one value to the mean.
        subjects = [k[0] for k in ks]
        names = {k: (k[0] if subjects.count(k[0]) == 1 else f"{k[0]} ({k[1]})") for k in ks}
        curves[g] = {"mean": mean, "sem": sem,
                     "n": int(sum(bool(np.isfinite(series[k]).any()) for k in ks)),
                     "animals": {names[k]: series[k] for k in ks}}
        b = [base_v[k] for k in ks]
        w = [win_v[k] for k in ks]
        d = [y - x for x, y in zip(b, w)]
        deltas[g] = d
        for lab, v in (("baseline", b), ("window", w), ("Δ", d)):
            m, s, n = describe(v)
            row[f"{g} {lab} mean"], row[f"{g} {lab} SEM"], row[f"{g} {lab} n"] = m, s, n
        wt = within_test(b, w)
        row[f"{g} Wilcoxon p"], row[f"{g} paired t p"] = wt["p"], wt["p t"]
        ns.append(wt["n"])
        frame = pd.DataFrame({_row_title(k[0], sexes.get(k, "")): series[k] for k in ks})
        frame.insert(0, "Time (min)", t_min)
        prism.append((f"{label} · time course{' · ' + g if len(order) > 1 else ''}",
                      frame, "Time (min)"))
        prism.append(_paired_table(
            f"{metric} ({unit}) baseline vs {w0:g}–{w1:g} min"
            f"{' · ' + g if len(order) > 1 else ''}",
            [_row_title(k[0], sexes.get(k, "")) for k in ks],
            {f"baseline ({start_min:g}–0 min)": b, f"{w0:g}–{w1:g} min": w}))
    if len(order) > 1:
        bt = between_test([deltas[g] for g in order])
        row.update({"between test": bt["rank test"], "between p": bt["p"],
                    "between param test": bt["param test"], "between param p": bt["p param"]})
        prism.append(_column_table(f"Δ {metric} ({unit}) {w0:g}–{w1:g} min vs baseline", deltas))
    rows.append(row)
    stats = pd.DataFrame(rows)
    dec = _decimals([row.get(f"{g} Δ mean", math.nan) for g in order])
    if len(order) > 1:
        d1 = {"Test": f"Δ {metric} {w0:g}–{w1:g} min ({unit})"}
        for g in order:
            d1[g] = fmt_mean(row[f"{g} Δ mean"], row[f"{g} Δ SEM"], int(row[f"{g} Δ n"]), dec)
        d1["p"] = f"{fmt_p(row['between p'])} ({row['between test']})"
        d1["p Holm"] = ""
        d1["_p"] = row["between p"]
        disp.append(d1)
    d2 = {"Test": f"Within group, {w0:g}–{w1:g} min vs baseline"}
    for g in order:
        d2[g] = f"p = {fmt_p(row[f'{g} Wilcoxon p'])}  (t: {fmt_p(row[f'{g} paired t p'])})"
    d2["p"], d2["p Holm"] = "Wilcoxon (paired t)", ""
    d2["_p"] = min([row[f"{g} Wilcoxon p"] for g in order
                    if math.isfinite(row[f"{g} Wilcoxon p"])], default=math.nan)
    disp.append(d2)
    # Mixed ANOVA on the animals with every bin.
    complete = [k for k in series if np.all(np.isfinite(series[k]))]
    dropped = [k[0] for k in series if k not in complete]
    anova = pd.DataFrame()
    groups_c = [gmap[k] for k in complete]
    if len(complete) >= len(set(groups_c)) + 1 and len(starts) >= 2 and complete:
        anova = mixed_anova(np.array([series[k] for k in complete]), groups_c)
        anova["Source"] = anova["Source"].replace({"Within": "Time",
                                                   "Group × within": "Group × time"})
        anova.insert(0, "Metric", label)
        for _, a in anova.iterrows():
            p = a["p GG"] if math.isfinite(a["p GG"]) else a["p"]
            eps = f", GG ε = {a['epsilon']:.2f}" if math.isfinite(a["epsilon"]) else ""
            d = {"Test": f"RM ANOVA {a['Source']}"}
            d[order[0]] = (f"F({a['df1']:g}, {a['df2']:g}) = {a['F']:.2f}{eps}"
                           if math.isfinite(a["F"]) else "—")
            for g in order[1:]:
                d[g] = ""
            d["p"], d["p Holm"], d["_p"] = fmt_p(p), "", p
            disp.append(d)
    if dropped:
        notes.append(f"Left out of the ANOVA (an empty bin): {', '.join(dropped)}")
    note = _small_n_note(ns)
    if note:
        notes.append(note)
    notes.append(f"A bin covered less than {min_coverage:.0%} by valid RR is empty for that animal")
    res = GroupResult(analysis=AN_TIME, by=by, groups=order, display=_display(disp, order),
                      stats=stats, per_animal=pd.DataFrame(
                          [{"subject": k[0], "file": k[1], "group": gmap[k],
                            "baseline": base_v[k], "window": win_v[k],
                            **{f"{t:g} min": v for t, v in zip(t_min, series[k])}}
                           for k in series]),
                      anova=anova, notes=notes, prism=prism)
    res.figure = {"kind": "time", "t_min": t_min, "curves": curves, "label": label,
                  "window": (w0, w1), "stim": stim, "bin_s": bin_s}
    res.settings = [("Analysis", AN_TIME), ("By", by), ("Time 0", stim),
                    ("Metric", label), ("Bins (s)", bin_s),
                    ("From / to (min)", f"{start_min:g} / {end_min:g}"),
                    ("Test window (min)", f"{w0:g}–{w1:g}")]
    return res


# ── statistics workbook ──────────────────────────────────────────────────

def write_results(path: str, res: GroupResult, table_name: str = "") -> None:
    """The statistics of *res* as an Excel workbook: Results (the table of
    the tab), Statistics (every number), ANOVA, Per animal, Settings."""
    settings = [("Group table", table_name)] + list(res.settings) + \
        [("Groups", ", ".join(res.groups)), ("Animals", res.n_animals)] + \
        [("Note", n) for n in res.notes]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        res.display.drop(columns=["_p"], errors="ignore").to_excel(xw, sheet_name="Results",
                                                                   index=False)
        res.stats.to_excel(xw, sheet_name="Statistics", index=False)
        if len(res.anova):
            res.anova.to_excel(xw, sheet_name="ANOVA", index=False)
        res.per_animal.to_excel(xw, sheet_name="Per animal", index=False)
        pd.DataFrame(settings, columns=["Setting", "Value"]).to_excel(
            xw, sheet_name="Settings", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))
