# -*- coding: utf-8 -*-
"""
ecg.ui.poincare_figure
----------------------
Poincaré plots of the segments side by side (before / after a stimulus), as
in the HRV literature: RR_n against RR_n+1, the same square axes on both
panels so that a shift of the cloud (rate change) and its contraction (SD1,
SD2) read at a glance, the identity line, the cloud's centre and its SD1/SD2
ellipse. Only successive valid pairs are plotted (``compute_segment_stats``
"pairs_x" / "pairs_y"): an interval excluded by the RR bounds or spanning a
removed beat makes no point.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.patches import Ellipse

from ecg.ui.plots import style_axes
from ecg.ui.theme import PLOT


def shared_limits(segments: "Sequence[dict]", pad: float = 0.06) -> "Optional[tuple[float, float]]":
    """One (lo, hi) range for both axes of every panel: all the pairs of all
    the segments, widened by *pad* of the span. None without a pair."""
    v = [np.asarray(s.get(k, ()), dtype=float) for s in segments for k in ("pairs_x", "pairs_y")]
    v = np.concatenate([x[np.isfinite(x)] for x in v]) if v else np.array([])
    if not len(v):
        return None
    lo, hi = float(v.min()), float(v.max())
    half = grid_step(v) / 2              # room for the display jitter
    span = max(hi - lo, 1.0)
    return lo - half - pad * span, hi + half + pad * span


def grid_step(v: np.ndarray) -> float:
    """The sampling step the RR values sit on (1 ms at 1 kHz), or 0 when they
    are not on a grid. Found as the smallest gap between distinct values,
    accepted only when the values are few compared with the pairs."""
    u = np.unique(np.round(v, 6))
    if len(u) < 2 or len(u) > 0.25 * len(v):
        return 0.0
    return float(np.min(np.diff(u)))


def jitter_for_display(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray, float]":
    """Display only: spread the points that the sampling grid stacks on one
    spot by a uniform ±half step (fixed seed, so a figure redraws the same).
    The ellipse, the centre and SD1 / SD2 use the measured values."""
    step = grid_step(np.r_[x, y])
    if step <= 0:
        return x, y, 0.0
    rng = np.random.default_rng(0)
    h = step / 2
    return x + rng.uniform(-h, h, len(x)), y + rng.uniform(-h, h, len(y)), step


def draw_poincare_panel(ax, seg: dict, color: str, lim: "tuple[float, float]",
                        point_size: float = 6.0, jitter: bool = True) -> None:
    """One segment's Poincaré plot on *ax*, on the shared square range *lim*.
    *jitter* False draws the points at their measured values (stacked on the
    sampling grid)."""
    style_axes(ax)
    x = np.asarray(seg.get("pairs_x", ()), dtype=float)
    y = np.asarray(seg.get("pairs_y", ()), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ax.plot(lim, lim, color=PLOT.get("muted", "#888"), lw=0.8, ls="--", zorder=1)
    step = 0.0
    if len(x):
        jx, jy, step = jitter_for_display(x, y) if jitter else (x, y, 0.0)
        ax.scatter(jx, jy, s=point_size, color=color, alpha=0.35, linewidths=0, zorder=2)
        cx, cy = float(x.mean()), float(y.mean())
        sd1, sd2 = float(seg.get("sd1", np.nan)), float(seg.get("sd2", np.nan))
        if np.isfinite(sd1) and np.isfinite(sd2):
            # SD2 along the identity line, SD1 across it (±1 SD: the usual ellipse).
            ax.add_patch(Ellipse((cx, cy), width=2 * sd2, height=2 * sd1, angle=45.0,
                                 fill=False, edgecolor=color, lw=1.6, zorder=3))
            d2, d1 = sd2 / np.sqrt(2), sd1 / np.sqrt(2)
            ax.plot([cx - d2, cx + d2], [cy - d2, cy + d2], color=color, lw=1.0, zorder=3)
            ax.plot([cx + d1, cx - d1], [cy - d1, cy + d1], color=color, lw=1.0, zorder=3)
        ax.plot([cx], [cy], "o", ms=5, color="white", mec=color, mew=1.5, zorder=4)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RR$_n$ (ms)", fontsize=9)
    ax.set_ylabel("RR$_{n+1}$ (ms)", fontsize=9)
    if step:
        ax.text(0.99, 0.01, f"points jittered ±{step / 2:.2g} ms (sampling grid)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color=PLOT.get("muted", "#888"))
    sd1_t = f"{seg['sd1']:.2f}" if np.isfinite(seg.get("sd1", np.nan)) else "—"
    sd2_t = f"{seg['sd2']:.2f}" if np.isfinite(seg.get("sd2", np.nan)) else "—"
    ax.set_title(f"{seg.get('label', '')}\nSD1 {sd1_t} ms · SD2 {sd2_t} ms · {len(x)} pairs",
                 fontsize=9, color=color, loc="center")


def draw_poincare_set(fig, segs: "Sequence[dict]", colors: "Sequence[str]",
                      title: str = "", jitter: bool = True) -> bool:
    """Every segment side by side on *fig* (same square axes; up to 4 a
    row). False when no segment has a pair (nothing drawn but a note)."""
    lim = shared_limits(segs)
    if lim is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No successive valid RR pairs in any segment",
                ha="center", va="center", transform=ax.transAxes,
                color=PLOT.get("muted", "#888"))
        ax.set_axis_off()
        return False
    n = len(segs)
    ncols = min(n, 4)
    nrows = -(-n // ncols)
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
    for k, ax in enumerate(axes):
        if k < n:
            draw_poincare_panel(ax, segs[k], colors[k % len(colors)], lim, jitter=jitter)
        else:
            ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10, color=PLOT.get("text", "#222"))
    return True


def draw_poincare_pair(fig, sa: dict, sb: dict, colors: "Sequence[str]",
                       title: str = "") -> bool:
    """Two segments side by side (draw_poincare_set)."""
    return draw_poincare_set(fig, (sa, sb), colors, title)
