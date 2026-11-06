# -*- coding: utf-8 -*-
"""
ecg.ui.group_figure
-------------------
The group figure of a Compare segments group table, for two segments A / B:

* two Poincaré panels on the same square axes, one ellipse per animal (centre
  at its mean RR on the identity line, half-axes SD2 along it and SD1 across
  it, the app's ellipse) and the group mean ellipse (mean of the centres, of
  SD1 and of SD2) drawn thick -- the group counterpart of one animal's
  Poincaré clouds, without pooling the beats of animals of different size;
* before-after panels (one line per animal, the median thick) of the
  metrics the paired tests are run on.

Only the group table is needed: it holds each segment's mean RR, SD1 and SD2.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from ecg.ui.plots import style_axes
from ecg.ui.theme import PLOT

PAIRED_PANELS = (("Mean HR (60000/mean NN) (bpm)", "HR (bpm)"),
                 ("Poincaré SD1 (ms)", "SD1 (ms)"), ("Poincaré SD2 (ms)", "SD2 (ms)"))
_RR, _SD1, _SD2 = "Mean RR (ms)", "Poincaré SD1 (ms)", "Poincaré SD2 (ms)"


def poincare_rows(df: "pd.DataFrame", a: str, b: str) -> "pd.DataFrame":
    """The animals with a mean RR, SD1 and SD2 for both segments."""
    need = [f"{m} {t}" for m in (_RR, _SD1, _SD2) for t in (a, b)]
    if any(c not in df.columns for c in need):
        return df.iloc[0:0]
    num = df[need].apply(pd.to_numeric, errors="coerce")
    return df.loc[num.notna().all(axis=1)]


def draw_group_figure(fig, df: "pd.DataFrame", a: str, b: str,
                      names: "Sequence[str]" = (), colors: "Sequence[str]" = ("#2563eb", "#ea580c"),
                      title: str = "") -> bool:
    """The group figure on *fig*; *names* are the panel titles of A and B
    (default the segment tags). False when no animal has both segments."""
    show = {a: names[0] if len(names) > 0 else a, b: names[1] if len(names) > 1 else b}
    g = poincare_rows(df, a, b)
    muted = PLOT.get("muted", "#888")
    if g.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"No animal with a mean RR, SD1 and SD2 for both '{a}' and '{b}'",
                ha="center", va="center", transform=ax.transAxes, color=muted)
        ax.set_axis_off()
        return False
    ids = [str(s) for s in g["subject"]] if "subject" in g.columns else [""] * len(g)
    rr = {t: pd.to_numeric(g[f"{_RR} {t}"]).to_numpy(float) for t in (a, b)}
    s1 = {t: pd.to_numeric(g[f"{_SD1} {t}"]).to_numpy(float) for t in (a, b)}
    s2 = {t: pd.to_numeric(g[f"{_SD2} {t}"]).to_numpy(float) for t in (a, b)}
    all_rr = np.r_[rr[a], rr[b]]
    pad = 1.5 * float(np.r_[s2[a], s2[b]].max()) + 0.05 * max(float(np.ptp(all_rr)), 1.0)
    lim = (float(all_rr.min()) - pad, float(all_rr.max()) + pad)

    panels = [p for p in PAIRED_PANELS if f"{p[0]} {a}" in g.columns and f"{p[0]} {b}" in g.columns]
    gs = fig.add_gridspec(1, 2 + len(panels), width_ratios=[1, 1] + [0.5] * len(panels))
    for k, (tag, col) in enumerate(zip((a, b), colors)):
        ax = fig.add_subplot(gs[0, k])
        style_axes(ax)
        ax.plot(lim, lim, "--", color=muted, lw=0.8, zorder=1)
        for x, e1, e2, name in zip(rr[tag], s1[tag], s2[tag], ids):
            ax.add_patch(Ellipse((x, x), 2 * e2, 2 * e1, angle=45, fill=False, ec=col,
                                 lw=0.9, alpha=0.5, zorder=2))
            ax.plot(x, x, "o", ms=3, color=col, alpha=0.7, zorder=3)
            ax.annotate(name, (x, x), xytext=(5, -9), textcoords="offset points",
                        fontsize=7, color=muted)
        m = float(rr[tag].mean())
        ax.add_patch(Ellipse((m, m), 2 * float(s2[tag].mean()), 2 * float(s1[tag].mean()),
                             angle=45, fill=False, ec=col, lw=2.6, zorder=4))
        ax.plot(m, m, "o", ms=7, color="white", mec=col, mew=2, zorder=5)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("RR$_n$ (ms)", fontsize=9)
        ax.set_ylabel("RR$_{n+1}$ (ms)", fontsize=9)
        ax.set_title(f"{show[tag]}\nmean SD1 {s1[tag].mean():.2f} · SD2 {s2[tag].mean():.2f} ms"
                     f" · n = {len(g)}", fontsize=9, color=col)
    for j, (col_name, label) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 + j])
        style_axes(ax)
        va = pd.to_numeric(g[f"{col_name} {a}"], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(g[f"{col_name} {b}"], errors="coerce").to_numpy(float)
        ok = np.isfinite(va) & np.isfinite(vb)
        for p0, p1 in zip(va[ok], vb[ok]):
            ax.plot([0, 1], [p0, p1], "-o", color=muted, ms=3.5, lw=0.9, alpha=0.8)
        if ok.any():
            ax.plot([0, 1], [np.median(va[ok]), np.median(vb[ok])], "-", color=PLOT.get("text", "#222"),
                    lw=2.5)
        ax.set_xticks([0, 1], [show[a], show[b]], fontsize=8)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(label, fontsize=9)
    fig.suptitle((title + "  ·  " if title else "")
                 + "thin: each animal (centre = mean RR, half-axes SD2 / SD1) · thick: group mean; "
                   "right: one line per animal, median thick", fontsize=8, color=muted)
    return True
