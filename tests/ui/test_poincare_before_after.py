"""Compare Segments: Poincaré plots before / after side by side, and the group
table of one row per recording."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from ecg.core.analysis import compute_segment_stats
from ecg.io.group_table import append_group_row, group_row
from ecg.ui.poincare_figure import (
    draw_poincare_pair, draw_poincare_set, grid_step, shared_limits,
)

FS = 1000.0
METRICS = [("hr_mean", "Mean HR", "bpm"), ("sd1", "Poincaré SD1", "ms"),
           ("sd2", "Poincaré SD2", "ms")]


def _peaks(rr_ms):
    return np.r_[0, np.cumsum(np.round(np.asarray(rr_ms) / 1000 * FS))].astype(int) + 500


def _segments():
    rng = np.random.default_rng(1)
    before = 130 + 4 * rng.standard_normal(1500)           # slower, wider
    after = 115 + 1.5 * rng.standard_normal(1500)          # faster, narrower
    rr = np.r_[before, after]
    rp = _peaks(rr)
    sig = np.zeros(rp[-1] + 1000)
    t_mid = rp[len(before)] / FS
    sa = compute_segment_stats(sig, rp, FS, 0.0, t_mid - 1, "Before")
    sb = compute_segment_stats(sig, rp, FS, t_mid + 1, rp[-1] / FS, "After")
    return sa, sb


def test_segments_carry_their_successive_valid_pairs():
    sa, sb = _segments()
    for s in (sa, sb):
        assert len(s["pairs_x"]) == len(s["pairs_y"]) > 1000
        # Each pair is RR_n, RR_n+1 of the segment's valid intervals.
        assert np.allclose(s["pairs_x"][1:], s["pairs_y"][:-1])
    assert np.mean(sa["pairs_x"]) > np.mean(sb["pairs_x"])


def test_an_excluded_interval_makes_no_pair():
    rr = np.full(400, 120.0)
    rr[200] = 1500.0                                      # a missed beat: excluded
    rp = _peaks(rr)
    s = compute_segment_stats(np.zeros(rp[-1] + 1000), rp, FS, 0.0, rp[-1] / FS, "x")
    assert np.all(s["pairs_x"] < 1000) and np.all(s["pairs_y"] < 1000)
    assert len(s["pairs_x"]) == 400 - 1 - 2               # the two pairs touching it


def test_both_panels_share_square_axes_with_their_ellipse():
    sa, sb = _segments()
    fig = Figure(figsize=(10, 5))
    assert draw_poincare_pair(fig, sa, sb, ("blue", "orange"))
    a, b = fig.axes
    assert a.get_xlim() == b.get_xlim() == a.get_ylim() == b.get_ylim()
    lo, hi = shared_limits((sa, sb))
    all_v = np.r_[sa["pairs_x"], sa["pairs_y"], sb["pairs_x"], sb["pairs_y"]]
    assert lo < all_v.min() and hi > all_v.max()
    for ax, s in ((a, sa), (b, sb)):
        (e,) = [p for p in ax.patches if isinstance(p, Ellipse)]
        assert np.isclose(e.width, 2 * s["sd2"]) and np.isclose(e.height, 2 * s["sd1"])
        assert e.angle == 45.0


def test_no_pair_draws_a_note():
    empty = {"pairs_x": np.array([]), "pairs_y": np.array([]), "label": "x"}
    assert not draw_poincare_pair(Figure(), empty, empty, ("b", "o"))


def test_group_table_one_row_per_recording(tmp_path):
    sa, sb = _segments()
    path = str(tmp_path / "group.xlsx")
    row = lambda subj, f, b: group_row(subj, f, [sa, b], METRICS, template="T5")  # noqa: E731
    assert append_group_row(path, row("M1", "D:/x/1720M.mat", sb)) == (1, False)
    assert append_group_row(path, row("M2", "D:/x/1721M.mat", sb)) == (2, False)
    # The same recording again replaces its row.
    assert append_group_row(path, row("M1", "D:/x/1720M.mat", dict(sb, sd1=99.0))) == (2, True)
    df = pd.read_excel(path, sheet_name="Group")
    assert list(df["subject"]) == ["M1", "M2"]
    # Columns tagged with the segment labels.
    assert df.loc[df.subject == "M1", "Poincaré SD1 (ms) After"].item() == 99.0
    assert {"Poincaré SD1 (ms) Before", "Mean HR (bpm) After", "start (s) Before",
            "valid RR (%) After", "Poincaré pairs (n) Before", "QC After"} <= set(df.columns)
    assert (df["template"] == "T5").all() and (df["QC Before"] == "ok").all()


def test_group_row_takes_any_number_of_segments_and_their_flags():
    sa, sb = _segments()
    sc = dict(sb, label="Late")
    row = group_row("M1", "f.mat", [sa, sb, sc], METRICS, flags=[[], ["few pairs"], []])
    assert row["QC After"] == "few pairs" and row["QC Late"] == "ok"
    assert "Mean HR (bpm) Late" in row
    # Labels that clash fall back to letters.
    row = group_row("M1", "f.mat", [sa, dict(sb, label="Before")], METRICS)
    assert "Mean HR (bpm) A" in row and "Mean HR (bpm) B" in row


def test_points_on_the_sampling_grid_are_jittered_for_display_only():
    sa, sb = _segments()                                  # RR rounded to 1 ms
    assert grid_step(sa["pairs_x"]) == 1.0
    assert grid_step(np.linspace(100, 120, 500)) == 0.0   # not on a grid
    fig = Figure(figsize=(10, 5))
    draw_poincare_pair(fig, sa, sb, ("blue", "orange"))
    ax = fig.axes[0]
    pts = ax.collections[0].get_offsets()
    d = pts[:, 0] - sa["pairs_x"]
    assert np.all(np.abs(d) <= 0.5) and np.std(d) > 0.1
    lo, hi = ax.get_xlim()
    assert lo <= pts[:, 0].min() and hi >= pts[:, 0].max()
    assert any("jittered" in t.get_text() for t in ax.texts)


def test_poincare_set_draws_one_panel_per_segment():
    sa, sb = _segments()
    fig = Figure(figsize=(12, 4))
    segs = [sa, sb, dict(sb, label="C")]
    assert draw_poincare_set(fig, segs, ("b", "orange", "g"))
    assert len(fig.axes) == 3
    assert len({ax.get_xlim() for ax in fig.axes}) == 1
    fig = Figure()
    draw_poincare_set(fig, [sa] * 5, ("b",))
    assert sum(ax.axison for ax in fig.axes) == 5            # 2 rows of 4, 3 blank


def test_jitter_can_be_turned_off():
    sa, sb = _segments()
    fig = Figure(figsize=(10, 5))
    draw_poincare_pair(fig, sa, sb, ("blue", "orange"))
    fig_off = Figure(figsize=(10, 5))
    from ecg.ui.poincare_figure import draw_poincare_set as _set
    _set(fig_off, (sa, sb), ("blue", "orange"), jitter=False)
    pts = fig_off.axes[0].collections[0].get_offsets()
    assert np.array_equal(pts[:, 0], sa["pairs_x"]) and np.array_equal(pts[:, 1], sa["pairs_y"])
    assert not any("jittered" in t.get_text() for t in fig_off.axes[0].texts)
    # The ellipse is the same either way.
    e_on = [p for p in fig.axes[0].patches if isinstance(p, Ellipse)][0]
    e_off = [p for p in fig_off.axes[0].patches if isinstance(p, Ellipse)][0]
    assert (e_on.width, e_on.height, e_on.center) == (e_off.width, e_off.height, e_off.center)
