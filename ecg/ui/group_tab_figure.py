# -*- coding: utf-8 -*-
"""
ecg.ui.group_tab_figure
-----------------------
The figure of the Group tab, from GroupResult.figure (ecg.io.group_compare):

* time: mean ± SEM per group and bin (a band), the time 0 dashed, the test
  window shaded, optionally one thin line per animal;
* paired (Pre/Post segments, Event-locked): one panel per group, one line
  per animal from the reference to the condition, the group mean thick;
* dots (Whole recording): one column per group, a dot per animal, the mean
  ± SEM.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ecg.ui.plots import style_axes
from ecg.ui.theme import PLOT

# Blue / orange first: told apart by lightness too, not hue alone.
GROUP_COLORS = ("#2563eb", "#ea580c", "#059669", "#7c3aed", "#db2777", "#0891b2")


def group_color(i: int) -> str:
    return GROUP_COLORS[i % len(GROUP_COLORS)]


def _empty(fig, text: str) -> bool:
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            color=PLOT.get("muted", "#888"))
    ax.set_axis_off()
    return False


def draw_group_tab_figure(fig, res: Any, metric: Optional[str] = None,
                          show_animals: bool = False, title: str = "") -> bool:
    """Draw *res* (a GroupResult) on *fig*; *metric* chooses the metric of a
    paired or dots figure (default: the first one tested)."""
    data = getattr(res, "figure", None) or {}
    kind = data.get("kind")
    if kind == "time":
        return _draw_time(fig, data, res.groups, show_animals, title)
    if kind in ("paired", "dots"):
        m = metric if metric in data.get("values", {}) else data.get("metric", "")
        if not m:
            return _empty(fig, "No metric to draw")
        if kind == "paired":
            return _draw_paired(fig, data, res.groups, m, title)
        return _draw_dots(fig, data, res.groups, m, title)
    return _empty(fig, "Open a group table and choose an analysis")


def _draw_time(fig, data: dict, groups: "Sequence[str]", show_animals: bool, title: str) -> bool:
    t = np.asarray(data["t_min"], dtype=float)
    muted = PLOT.get("muted", "#888")
    ax = fig.add_subplot(111)
    style_axes(ax)
    w0, w1 = data["window"]
    ax.axvspan(w0, w1, color="#2563eb", alpha=0.07, lw=0, zorder=0)
    ax.axvline(0, color=PLOT.get("text", "#222"), ls="--", lw=1, zorder=1)
    ax.text(0, 1.0, f" {data.get('stim', 'stim')}", transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=9, color=muted)
    if str(data.get("label", "")).startswith("Δ"):
        ax.axhline(0, color=muted, lw=0.6, zorder=1)
    any_line = False
    for i, g in enumerate(groups):
        c = data["curves"].get(g)
        if not c or c["n"] == 0:
            continue
        col = group_color(i)
        if show_animals:
            for v in c["animals"].values():
                ax.plot(t, v, color=col, lw=0.7, alpha=0.35, zorder=2)
        m, s = np.asarray(c["mean"], float), np.asarray(c["sem"], float)
        ax.fill_between(t, m - s, m + s, color=col, alpha=0.18, lw=0, zorder=2)
        ls = "-" if i % 2 == 0 else (0, (6, 3))
        ax.plot(t, m, color=col, lw=2.2, ls=ls, zorder=3, label=f"{g} (n = {c['n']})")
        any_line = True
    if not any_line:
        fig.delaxes(ax)
        return _empty(fig, "No animal to draw")
    ax.set_xlabel("Time from the time 0 (min)", fontsize=9)
    ax.set_ylabel(str(data.get("label", "")), fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_title((title + "  ·  " if title else "")
                 + f"mean ± SEM per {data.get('bin_s', 0):g} s bin · shaded: test window "
                   f"{w0:g}–{w1:g} min", fontsize=9, color=muted)
    return True


def _by_group(data: dict, groups: "Sequence[str]", metric: str) -> "dict[str, list]":
    vals = data["values"].get(metric, {})
    gmap = data["gmap"]
    return {g: [(k[0], vals[k]) for k in gmap if gmap[k] == g and k in vals] for g in groups}


def _draw_paired(fig, data: dict, groups: "Sequence[str]", metric: str, title: str) -> bool:
    by = _by_group(data, groups, metric)
    if not any(by.values()):
        return _empty(fig, f"No animal with both values of {metric}")
    na, nb = data.get("names", ("A", "B"))
    muted = PLOT.get("muted", "#888")
    axes = fig.subplots(1, len(groups), sharey=True, squeeze=False)[0]
    for i, (ax, g) in enumerate(zip(axes, groups)):
        style_axes(ax)
        col = group_color(i)
        pts = by[g]
        for name, (a, b) in pts:
            ax.plot([0, 1], [a, b], "-o", color=col, ms=4, lw=1, alpha=0.45)
            ax.annotate(name, (1, b), xytext=(5, -3), textcoords="offset points",
                        fontsize=7, color=muted)
        if pts:
            ma = float(np.mean([p[1][0] for p in pts]))
            mb = float(np.mean([p[1][1] for p in pts]))
            ax.plot([0, 1], [ma, mb], "-", color=PLOT.get("text", "#222"), lw=3, zorder=5)
        ax.set_xticks([0, 1], [na, nb], fontsize=9)
        ax.set_xlim(-0.3, 1.45)
        ax.set_title(f"{g} (n = {len(pts)})", fontsize=10, color=col)
        if i == 0:
            ax.set_ylabel(metric, fontsize=9)
    fig.suptitle((title + "  ·  " if title else "")
                 + "one line per animal · thick: group mean", fontsize=8, color=muted)
    return True


def _draw_dots(fig, data: dict, groups: "Sequence[str]", metric: str, title: str) -> bool:
    by = _by_group(data, groups, metric)
    if not any(by.values()):
        return _empty(fig, f"No animal with {metric}")
    muted = PLOT.get("muted", "#888")
    ax = fig.add_subplot(111)
    style_axes(ax)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        col = group_color(i)
        v = np.array([p[1] for p in by[g]], dtype=float)
        if not len(v):
            continue
        x = i + rng.uniform(-0.12, 0.12, len(v))
        ax.plot(x, v, "o", color=col, ms=6, alpha=0.75)
        for xi, (name, vi) in zip(x, by[g]):
            ax.annotate(name, (xi, vi), xytext=(6, -3), textcoords="offset points",
                        fontsize=7, color=muted)
        m = float(v.mean())
        s = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        ax.errorbar(i + 0.32, m, yerr=s, fmt="_", color=PLOT.get("text", "#222"),
                    ms=18, mew=2.5, capsize=5, lw=1.5)
    ax.set_xticks(range(len(groups)),
                  [f"{g}\n(n = {len(by[g])})" for g in groups], fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.2)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title((title + "  ·  " if title else "") + "one dot per animal · bar: mean ± SEM",
                 fontsize=9, color=muted)
    return True
