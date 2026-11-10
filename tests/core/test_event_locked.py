import math
import subprocess
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ecg.core import event_locked as el

FS = 20.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _grid(duration_s: float = 120.0, fs: float = FS) -> np.ndarray:
    """Evenly sampled time axis 0..duration_s inclusive (exact multiples of
    1/fs, so sample 60 s is exactly 60.0)."""
    return np.arange(int(round(duration_s * fs)) + 1) / fs


# ════════════════════════════════════════════════════════════
#  event_window_means
# ════════════════════════════════════════════════════════════

def test_window_means_are_exact_on_step_data():
    t = _grid()
    step = np.where(t < 60.0, 1.0, 3.0)
    # The two onsets are 15 s apart: not isolated (test_isolation_*), which
    # this test of the window arithmetic switches off.
    df = el.event_window_means(t, {"a": step, "r": t.copy()}, [60.0, 45.0],
                               labels=["inj", "sham"], require_isolated=False)

    per_series = ["pre", "post", "delta", "coverage_pre", "coverage_post", "valid"]
    assert list(df.columns) == (
        ["label", "onset_s", "valid", "coverage_pre", "coverage_post", "any_valid",
         "isolated", "prev_gap_s", "next_gap_s"]
        + [f"a_{s}" for s in per_series] + [f"r_{s}" for s in per_series])
    assert df["isolated"].tolist() == [False, False]
    assert df["prev_gap_s"].tolist()[0] == 15.0 and np.isnan(df["prev_gap_s"].tolist()[1])
    assert df["label"].tolist() == ["inj", "sham"]
    assert df["a_valid"].dtype == bool and df["valid"].dtype == bool
    assert df["any_valid"].dtype == bool

    on_step = df.iloc[0]
    assert bool(on_step["valid"]) and bool(on_step["any_valid"])
    assert bool(on_step["a_valid"]) and bool(on_step["r_valid"])
    assert on_step["coverage_pre"] == 1.0 and on_step["coverage_post"] == 1.0
    assert on_step["a_coverage_pre"] == 1.0 and on_step["a_coverage_post"] == 1.0
    # The sample AT the onset belongs to the post window: pre is pure 1s.
    assert on_step["a_pre"] == 1.0
    assert on_step["a_post"] == 3.0
    assert on_step["a_delta"] == 2.0
    # Ramp: half-open pre window [30, 60) holds 30.00 .. 59.95,
    # closed post window [60, 90] holds 60.00 .. 90.00.
    assert on_step["r_pre"] == pytest.approx(np.mean(np.arange(600, 1200) / FS))
    assert on_step["r_post"] == pytest.approx(75.0)

    # Onset 45 s: the post window [45, 75] straddles the step at 60 s --
    # 300 samples at 1 (45.00 .. 59.95) and 301 at 3 (60.00 .. 75.00).
    straddle = df.iloc[1]
    assert straddle["a_pre"] == 1.0
    assert straddle["a_post"] == pytest.approx((300 * 1.0 + 301 * 3.0) / 601)
    assert df.attrs["pre_s"] == 30.0 and df.attrs["post_s"] == 30.0


def test_window_means_default_labels_and_custom_window_lengths():
    t = _grid()
    df = el.event_window_means(t, {"r": t.copy()}, [60.0, 70.0], pre_s=10.0, post_s=5.0)
    assert df["label"].tolist() == ["Event 1", "Event 2"]
    assert df.loc[0, "r_pre"] == pytest.approx(np.mean(np.arange(1000, 1200) / FS))
    assert df.loc[0, "r_post"] == pytest.approx(62.5)


def test_events_too_close_to_the_recording_edges_are_invalid():
    t = _grid()   # 0 .. 120 s
    df = el.event_window_means(t, {"r": t.copy()}, [20.0, 100.0, 30.0, 90.0],
                               require_isolated=False)

    # Pre window [-10, 20) only has the 400 samples from 0 s.
    assert not df.loc[0, "valid"]
    assert df.loc[0, "coverage_pre"] == pytest.approx(400 / 600)
    assert df.loc[0, "coverage_post"] == 1.0
    # Closed post window [100, 130] spans 601 grid points, only the 401 up
    # to 120 s exist.
    assert not df.loc[1, "valid"]
    assert df.loc[1, "coverage_post"] == pytest.approx(401 / 601)
    # Windows that END exactly on the first / last sample are inside.
    assert df.loc[2, "valid"] and df.loc[3, "valid"]
    # Invalid events are masked so they drop out of the statistics...
    assert np.isnan(df.loc[0, ["r_pre", "r_post", "r_delta"]].astype(float)).all()
    assert np.isnan(df.loc[1, ["r_pre", "r_post", "r_delta"]].astype(float)).all()

    # ... unless the raw means are asked for.
    raw = el.event_window_means(t, {"r": t.copy()}, [20.0], mask_invalid=False)
    assert not raw.loc[0, "valid"]
    assert raw.loc[0, "r_pre"] == pytest.approx(np.mean(np.arange(0, 400) / FS))


def test_nan_gaps_lower_coverage_and_are_skipped_in_the_means():
    t = _grid()
    step = np.where(t < 60.0, 1.0, 3.0)

    short_gap = step.copy()
    short_gap[(t >= 35.0) & (t < 38.0)] = np.nan          # 60 of 600 pre samples
    df = el.event_window_means(t, {"a": short_gap}, [60.0])
    assert df.loc[0, "coverage_pre"] == pytest.approx(0.9)
    assert df.loc[0, "valid"]
    assert df.loc[0, "a_pre"] == 1.0                       # NaNs skipped, not zeroed

    long_gap = step.copy()
    long_gap[(t >= 70.0) & (t < 80.0)] = np.nan            # 200 of 601 post samples
    df = el.event_window_means(t, {"a": long_gap}, [60.0])
    assert df.loc[0, "coverage_post"] == pytest.approx(401 / 601)
    assert not df.loc[0, "valid"]
    df = el.event_window_means(t, {"a": long_gap}, [60.0], min_coverage=0.6)
    assert df.loc[0, "valid"]
    assert df.loc[0, "a_post"] == 3.0


def test_one_missing_sample_lowers_either_coverage():
    """Coverage counts against the exact nominal grid count of each window
    (600 for [o - 30, o), 601 for [o, o + 30] at 20 Hz), so a single NaN is
    seen on both sides -- also for an onset between grid points."""
    t = _grid()
    for onset, n_pre, n_post in ((60.0, 600, 601), (60.025, 600, 600)):
        y = np.ones(len(t))
        y[int(np.ceil((onset - 10.0) * FS))] = np.nan      # inside the pre window
        y[int(np.ceil((onset + 10.0) * FS))] = np.nan      # inside the post window
        df = el.event_window_means(t, {"y": y}, [onset])
        assert df.loc[0, "coverage_pre"] == pytest.approx((n_pre - 1) / n_pre)
        assert df.loc[0, "coverage_post"] == pytest.approx((n_post - 1) / n_post)
        full = el.event_window_means(t, {"y": np.ones(len(t))}, [onset])
        assert full.loc[0, "coverage_pre"] == 1.0 and full.loc[0, "coverage_post"] == 1.0


def _long_axes(hours: float):
    """(name, t, t0, step) for an axis of *hours* at 20 Hz built both ways the
    app builds them: ``arange / fs`` and ``t0 + arange * step`` (as
    autonomic._time_grid). t0 and step are the exact grid the samples round."""
    n = int(round(hours * 3600 * FS)) + 1
    return [("arange/fs", np.arange(n) / FS, Fraction(0), Fraction(1, 20)),
            ("t0+arange*step", 1.3 + np.arange(n) * 0.05, Fraction(1.3), Fraction(0.05))]


def _exact_nominal_counts(t0, step, a, onset, b):
    """Exact (rational) grid-point counts of [a, onset) and [onset, b] on the
    grid t0 + i * step, a point within 1e-6 step of a boundary lying on it."""
    tol = step / 10 ** 6
    first = lambda x: math.ceil((Fraction(x) - tol - t0) / step)     # noqa: E731
    last = lambda x: math.floor((Fraction(x) + tol - t0) / step)     # noqa: E731
    return first(onset) - first(a), last(b) - first(onset) + 1


@pytest.mark.parametrize("hours", [12.0, 36.0, 72.0])
def test_nominal_window_counts_are_exact_on_long_recordings(hours):
    """The step's round-off used to build up from t[0] over the whole
    recording (~1e-5 step after 12 h at 20 Hz, past the 1e-6 boundary
    tolerance), so late windows were miscounted by one sample. Anchored next
    to the onset with the axis' mean step, the counts are exact for every
    window length, on- or off-grid onset, and grid construction."""
    rng = np.random.default_rng(int(hours))
    for name, t, t0, step in _long_axes(hours):
        dt = el._grid_step(t)
        mismatches = []
        for pre_s, post_s in ((30.0, 30.0), (10.03, 7.777), (300.0, 300.0), (1800.0, 1800.0)):
            for _ in range(60):
                j = int(rng.integers(len(t) // 10, len(t) - len(t) // 10))
                onset = float(t[j]) if rng.random() < 0.7 else float(t[j] + 0.05 * rng.random())
                a, b = onset - pre_s, onset + post_s
                got = el._nominal_counts(t, dt, a, onset, b)
                if got != _exact_nominal_counts(t0, step, a, onset, b):
                    mismatches.append((name, pre_s, onset, got))
        assert not mismatches, mismatches[:5]


def test_coverage_is_exact_late_in_a_12_hour_recording():
    t = np.arange(int(12 * 3600 * FS) + 1) / FS
    onset = 32400.0                                        # 9 h, on a sample
    y = np.ones(len(t))
    y[int(round((onset + 10.0) * FS))] = np.nan            # one post sample missing
    df = el.event_window_means(t, {"y": y}, [onset])
    assert df.loc[0, "coverage_post"] == pytest.approx(600 / 601)   # as on a 2 h recording
    assert df.loc[0, "coverage_pre"] == 1.0

    # Complete data with min_coverage=1: every late onset is valid, also for
    # window lengths that are not a whole number of steps.
    onsets = t[len(t) // 2: -int(60 * FS): 2000]
    for pre_s in (10.03, 7.777, 30.0):
        full = el.event_window_means(t, {"y": np.ones(len(t))}, onsets, pre_s=pre_s,
                                     min_coverage=1.0)
        assert full["valid"].all(), (pre_s, full.loc[~full["valid"], "coverage_pre"].head())
        assert (full["coverage_pre"] == 1.0).all() and (full["coverage_post"] == 1.0).all()

    # A 30 min window, one sample missing on each side, on a 36 h axis.
    t36 = 1.3 + np.arange(int(36 * 3600 * FS) + 1) * 0.05
    j = int(30 * 3600 * FS)
    y36 = np.ones(len(t36))
    y36[j - 1000] = np.nan
    y36[j + 1000] = np.nan
    long_win = el.event_window_means(t36, {"y": y36}, [t36[j]], pre_s=1800.0, post_s=1800.0)
    assert long_win.loc[0, "coverage_pre"] == pytest.approx(35999 / 36000)
    assert long_win.loc[0, "coverage_post"] == pytest.approx(36000 / 36001)


def test_grid_step_of_an_uneven_axis_falls_back_to_the_median_step():
    t = _grid()
    assert el._grid_step(t) == pytest.approx(1.0 / FS, rel=1e-15)
    dropped = np.delete(t, 1000)                           # t = 50.00 s has no sample
    assert el._grid_step(dropped) == float(np.median(np.diff(dropped)))
    # A dropped time stamp is a missing sample of the regular grid.
    df = el.event_window_means(dropped, {"y": np.ones(len(dropped))}, [60.0])
    assert df.loc[0, "coverage_pre"] == pytest.approx(599 / 600)
    assert df.loc[0, "coverage_post"] == 1.0


def test_coverage_is_joint_across_series_but_each_mean_uses_its_own_samples():
    t = _grid()
    full = np.where(t < 60.0, 1.0, 3.0)
    gappy = full.copy()
    gappy[(t >= 40.0) & (t < 50.0)] = np.nan
    df = el.event_window_means(t, {"full": full, "gappy": gappy}, [60.0],
                               min_coverage=0.5)
    assert df.loc[0, "coverage_pre"] == pytest.approx(400 / 600)
    assert df.loc[0, "full_coverage_pre"] == 1.0
    assert df.loc[0, "gappy_coverage_pre"] == pytest.approx(400 / 600)
    assert df.loc[0, "full_pre"] == 1.0 and df.loc[0, "gappy_pre"] == 1.0


def test_a_missing_series_does_not_invalidate_the_others():
    """Validity is per series: an LF series that could not be computed
    (all NaN) or that is gappy around one event must not wipe out the
    CSI statistics of that event."""
    t = _grid()
    csi = np.where(t < 60.0, 1.0, 2.0)
    df = el.event_window_means(t, {"CSI": csi, "LF": np.full(len(t), np.nan)}, [60.0])
    ev = df.iloc[0]
    assert ev["coverage_pre"] == 0.0 and ev["coverage_post"] == 0.0   # joint: not complete
    assert not bool(ev["valid"])                           # so not jointly valid,
    assert bool(ev["any_valid"])                           # but the event still counts ...
    assert bool(ev["CSI_valid"]) and not bool(ev["LF_valid"])
    assert ev["CSI_delta"] == 1.0                          # ... for CSI
    assert np.isnan(ev[["LF_pre", "LF_post", "LF_delta"]].astype(float)).all()
    assert ev["LF_coverage_pre"] == 0.0 and ev["LF_coverage_post"] == 0.0
    alone = el.event_window_means(t, {"CSI": csi}, [60.0]).iloc[0]
    assert ev["CSI_delta"] == alone["CSI_delta"]
    assert bool(alone["valid"]) and bool(alone["any_valid"])

    # Nothing usable at all -> the event is invalid.
    none = el.event_window_means(t, {"LF": np.full(len(t), np.nan)}, [60.0])
    assert not none.loc[0, "valid"] and not none.loc[0, "any_valid"]
    # Outside the recording every series is invalid, whatever its coverage.
    edge = el.event_window_means(t, {"CSI": csi, "r": t.copy()}, [100.0], min_coverage=0.5)
    assert edge.loc[0, "CSI_coverage_post"] == pytest.approx(401 / 601)
    assert not edge.loc[0, "valid"] and not edge.loc[0, "CSI_valid"]
    assert not edge.loc[0, "any_valid"]


def test_event_validity_agrees_with_the_event_coverage_columns():
    """``valid`` is the spec's joint validity (inside and JOINT coverage >=
    min_coverage on both sides), so it never contradicts ``coverage_pre`` /
    ``coverage_post``; ``any_valid`` says whether some series is usable."""
    t = _grid()
    step = np.where(t < 60.0, 1.0, 3.0)
    a, b = step.copy(), step.copy()
    a[(t >= 30.0) & (t < 45.0)] = np.nan                   # complementary 15 s gaps:
    b[(t >= 45.0) & (t < 60.0)] = np.nan                   # 50 % each, 0 % jointly
    df = el.event_window_means(t, {"a": a, "b": b}, [60.0, 80.0, 20.0, 45.0],
                               min_coverage=0.4, require_isolated=False)
    ev = df.iloc[0]
    assert ev["a_coverage_pre"] == pytest.approx(0.5) and ev["b_coverage_pre"] == pytest.approx(0.5)
    assert ev["coverage_pre"] == 0.0
    assert bool(ev["a_valid"]) and bool(ev["b_valid"]) and bool(ev["any_valid"])
    assert not bool(ev["valid"])

    for _, row in df.iterrows():
        inside = row["onset_s"] - 30.0 >= t[0] and row["onset_s"] + 30.0 <= t[-1]
        assert row["valid"] == (inside and row["coverage_pre"] >= 0.4
                                and row["coverage_post"] >= 0.4)
        assert row["any_valid"] == (row["a_valid"] or row["b_valid"])
        if row["valid"]:                                   # joint validity implies each
            assert row["a_valid"] and row["b_valid"]
    # 80 s: pre [50, 80) jointly 400/600 finite; 20 s: pre window before
    # the recording; 45 s: a's gap fills half the pre window and b's half
    # the post window, each 0.5 >= 0.4 jointly too.
    assert df["valid"].tolist() == [False, True, False, True]
    assert df["any_valid"].tolist() == [True, True, False, True]


def test_paired_tests_and_aggregate_use_each_series_own_validity():
    t = _grid()
    tables = []
    for animal in range(8):
        a = np.where(t < 60.0, 1.0, 2.0 + 0.1 * animal)
        lf = np.where(t < 60.0, 5.0, 4.0 - 0.1 * animal)
        if animal < 3:
            lf[(t >= 65.0) & (t < 85.0)] = np.nan          # LF unusable for 3 events
        ev = el.event_window_means(t, {"a": a, "lf": lf}, [60.0], mask_invalid=False)
        ev["subject"] = f"m{animal}"
        tables.append(ev)
    df = pd.concat(tables, ignore_index=True)
    assert df["any_valid"].all() and df["a_valid"].all()
    assert df["lf_valid"].tolist() == [False] * 3 + [True] * 5
    assert df["valid"].tolist() == [False] * 3 + [True] * 5   # joint validity
    assert np.isfinite(df["lf_post"]).all()                # raw means kept (mask_invalid=False)

    out = el.paired_tests(df).set_index("key")
    assert out.loc["a", "n"] == 8                          # not cut down by the LF gaps
    assert out.loc["lf", "n"] == 5                         # invalid raw LF means left out
    assert out.loc["lf", "z"] == pytest.approx(
        el.wilcoxon_paired(df.loc[3:, "lf_pre"], df.loc[3:, "lf_post"])["z"])

    agg = el.aggregate(df, "subject")
    assert agg["n_events"].tolist() == [1] * 8
    assert np.isnan(agg.loc[:2, ["lf_pre", "lf_post", "lf_delta"]].to_numpy()).all()
    np.testing.assert_allclose(agg["a_delta"], 1.0 + 0.1 * np.arange(8))
    raw = el.aggregate(df, "subject", valid_only=False)
    assert np.isfinite(raw["lf_delta"]).all()
    # Complete-case analysis: filter on the joint ``valid`` first.
    complete = el.aggregate(df[df["valid"]], "subject")
    assert complete["subject"].tolist() == [f"m{i}" for i in range(3, 8)]


def test_clashing_series_keys_are_refused():
    t = _grid()
    with pytest.raises(ValueError):
        el.event_window_means(t, {"coverage": t.copy()}, [60.0])
    with pytest.raises(ValueError):
        el.event_window_means(t, {"a": t.copy(), "a_coverage": t.copy()}, [60.0])
    with pytest.raises(ValueError):                        # "any_valid"
        el.event_window_means(t, {"any": t.copy()}, [60.0])


def test_nan_onset_gives_an_invalid_row():
    t = _grid()
    df = el.event_window_means(t, {"r": t.copy()}, [np.nan, 60.0])
    assert not df.loc[0, "valid"]
    assert np.isnan(df.loc[0, "r_delta"])
    assert df.loc[1, "valid"]


def test_window_means_validate_their_inputs():
    t = _grid()
    with pytest.raises(ValueError):
        el.event_window_means(t, {"r": t[:-1]}, [60.0])
    with pytest.raises(ValueError):
        el.event_window_means(t[::-1], {"r": t}, [60.0])
    with pytest.raises(ValueError):
        el.event_window_means(t, {"r": t}, [60.0, 70.0], labels=["only one"])
    with pytest.raises(ValueError):
        el.event_window_means(t, {"r": t}, [60.0], pre_s=0.0)


# ════════════════════════════════════════════════════════════
#  event_locked_traces / stack_traces / group_summary
# ════════════════════════════════════════════════════════════

def test_event_locked_traces_interpolate_on_a_common_relative_grid():
    t = _grid()
    t_rel, tr = el.event_locked_traces(t, {"r": t.copy()}, [60.0, 60.025, 10.0],
                                       pre_s=30.0, post_s=30.0, fs_out=20.0,
                                       zscore=False)
    assert len(t_rel) == 1201
    assert t_rel[0] == -30.0 and t_rel[600] == 0.0 and t_rel[-1] == 30.0
    assert tr["r"].shape == (3, 1201)
    # On a ramp linear interpolation is exact, on- or off-grid.
    np.testing.assert_allclose(tr["r"][0], 60.0 + t_rel)
    np.testing.assert_allclose(tr["r"][1], 60.025 + t_rel)
    # Onset 10 s: the first 20 s of the window precede the recording.
    before = (10.0 + t_rel) < 0
    assert np.isnan(tr["r"][2][before]).all()
    np.testing.assert_allclose(tr["r"][2][~before], (10.0 + t_rel)[~before])


def test_event_locked_traces_never_bridge_a_nan_gap():
    t = _grid()
    y = t.copy()
    y[1300] = np.nan                                       # t = 65.00 s
    t_rel, tr = el.event_locked_traces(t, {"y": y}, [60.0, 60.025],
                                       pre_s=30.0, post_s=30.0, zscore=False)
    on_grid = dict(zip(np.round(60.0 + t_rel, 6), tr["y"][0]))
    assert np.isnan(on_grid[65.0])
    assert on_grid[64.95] == pytest.approx(64.95)
    off_grid = dict(zip(np.round(60.025 + t_rel, 6), tr["y"][1]))
    assert np.isnan(off_grid[64.975]) and np.isnan(off_grid[65.025])
    assert off_grid[64.925] == pytest.approx(64.925)


@pytest.mark.parametrize("t", [np.arange(4801) / 20.0, 1.537 + np.arange(4801) * 0.05],
                         ids=["arange/fs", "t0+arange*dt"])
def test_on_grid_queries_keep_finite_samples_next_to_a_nan(t):
    """``onset + t_rel`` lands on a sample only up to round-off (w ~ 1e-15
    or 1 - 1e-15); such a query is ON the sample and must not pick up the
    NaN of its neighbour. Onsets are taken from the time axis itself, as
    the autonomic grid builds them, not hand-rounded."""
    checked = 0
    for j in range(700, 4100, 37):
        y = np.arange(len(t), dtype=np.float64)            # y[i] = i
        gap = j + 50
        y[gap] = np.nan
        t_rel, tr = el.event_locked_traces(t, {"y": y}, [t[j]], zscore=False)
        row = tr["y"][0]
        samples = j + np.arange(-600, 601)                 # sample hit by each query
        expected = np.where(samples == gap, np.nan, samples.astype(np.float64))
        np.testing.assert_array_equal(row, expected)
        checked += 1
    assert checked > 50

    # The first / last sample are reachable from an onset a few ulp away.
    t_rel, tr = el.event_locked_traces(t, {"y": np.arange(len(t), dtype=float)},
                                       [t[0] + 30.0, t[-1] - 30.0], zscore=False)
    assert tr["y"][0][0] == 0.0 and tr["y"][1][-1] == len(t) - 1


def test_event_locked_traces_zscore_with_the_whole_recording_statistics():
    t = _grid()
    y = 5.0 + 2.0 * np.sin(2 * np.pi * 0.05 * t)
    y[:100] = np.nan
    _, raw = el.event_locked_traces(t, {"y": y}, [60.0], zscore=False)
    _, zs = el.event_locked_traces(t, {"y": y}, [60.0], zscore=True)
    fin = y[np.isfinite(y)]
    np.testing.assert_allclose(zs["y"], (raw["y"] - fin.mean()) / fin.std(ddof=1))

    # A constant series has no spread: it is centred only (MATLAB zscore).
    _, flat = el.event_locked_traces(t, {"c": np.full(len(t), 7.0)}, [60.0])
    np.testing.assert_allclose(flat["c"], 0.0)


def test_stack_traces_concatenates_recordings():
    t = _grid()
    _, a = el.event_locked_traces(t, {"r": t.copy()}, [40.0, 60.0], zscore=False)
    _, b = el.event_locked_traces(t, {"r": t.copy()}, [80.0], zscore=False)
    stacked = el.stack_traces([a, b])
    assert stacked["r"].shape == (3, a["r"].shape[1])
    np.testing.assert_allclose(stacked["r"][2], b["r"][0])
    with pytest.raises(ValueError):
        el.stack_traces([a, {"other": b["r"]}])
    assert el.stack_traces([]) == {}


def test_group_summary_is_nan_aware_median_and_unscaled_mad():
    nan = np.nan
    a = np.array([[1.0, 2.0, nan, 1.0],
                  [2.0, 4.0, nan, nan],
                  [3.0, 6.0, nan, 3.0],
                  [4.0, 8.0, nan, nan],
                  [100.0, 10.0, nan, nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # all-NaN column is silent
        med, mad = el.group_summary({"x": a})["x"]
    np.testing.assert_allclose(med[[0, 1, 3]], [3.0, 6.0, 2.0])
    np.testing.assert_allclose(mad[[0, 1, 3]], [1.0, 2.0, 1.0])  # mad(x, 1), not x1.4826
    assert np.isnan(med[2]) and np.isnan(mad[2])

    med1, mad1 = el.group_summary({"x": np.array([1.0, 2.0])})["x"]
    np.testing.assert_allclose(med1, [1.0, 2.0])
    np.testing.assert_allclose(mad1, [0.0, 0.0])


# ════════════════════════════════════════════════════════════
#  wilcoxon_paired
# ════════════════════════════════════════════════════════════

def test_wilcoxon_matches_a_hand_calculation_and_scipy():
    d = np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0, -7.0, 8.0])
    res = el.wilcoxon_paired(np.zeros(8), d)

    # Ranks of |d| are 1..8; positive ones: 1+2+4+5+6+8 = 26.
    # E = 8*9/4 = 18, Var = 8*9*17/24 = 51 -> z = (26 - 18) / sqrt(51).
    assert res["n"] == 8 and res["n_nonzero"] == 8
    assert res["statistic"] == 26.0
    assert res["z"] == pytest.approx(8.0 / np.sqrt(51.0))
    assert res["z"] > 0 and res["median_delta"] > 0       # post > pre -> z > 0
    ref = stats.wilcoxon(d, method="approx")
    assert abs(res["z"]) == pytest.approx(abs(ref.zstatistic))
    assert res["p_value"] == pytest.approx(stats.wilcoxon(d).pvalue)

    flipped = el.wilcoxon_paired(d, np.zeros(8))
    assert flipped["z"] == pytest.approx(-res["z"])
    assert flipped["p_value"] == pytest.approx(res["p_value"])


def test_wilcoxon_tie_and_zero_corrections():
    pre = 10.0 * np.arange(10)                             # integers: exact differences
    d = np.array([0.0, 1, 1, -2, 3, 3, 3, -4, 5, 0])
    post = pre + d
    res = el.wilcoxon_paired(pre, post)

    # Zeros dropped -> n' = 8; |d| ranks 1.5,1.5,3,5,5,5,7,8; W+ = 26.
    # Var = 8*9*17/24 - ((2^3-2) + (3^3-3)) / 48 = 51 - 0.625.
    assert res["n"] == 10 and res["n_nonzero"] == 8
    assert res["statistic"] == 26.0
    assert res["z"] == pytest.approx(8.0 / np.sqrt(51.0 - 0.625))
    ref = stats.wilcoxon(post, pre, zero_method="wilcox", method="approx")
    assert abs(res["z"]) == pytest.approx(abs(ref.zstatistic))
    assert 0.0 < res["p_value"] <= 1.0


def test_wilcoxon_drops_nan_pairs_and_round_off_zeros():
    pre = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 0.1 + 0.2, 7.0])
    post = np.array([2.0, 4.0, 3.0, np.nan, 9.0, 0.3, 12.0])
    res = el.wilcoxon_paired(pre, post)
    assert res["n"] == 5                                   # two NaN pairs dropped
    assert res["n_nonzero"] == 4                           # 0.3 - (0.1 + 0.2) is round-off
    assert res["median_pre"] == pytest.approx(2.0)
    assert res["median_post"] == pytest.approx(4.0)
    assert res["mad_pre"] == pytest.approx(np.median(np.abs(np.array([1, 2, 5, 0.3, 7]) - 2.0)))
    assert res["z"] > 0


def test_wilcoxon_z_sign_follows_the_rank_sum_not_the_median():
    """Pinned convention (z = (W+ - E) / sd, the sign scipy computes before
    reporting -|z| for a two-sided test): in a skewed sample the
    median change can be negative while the positive ranks dominate; z then
    follows W+ - E, the quantity the test (and p) is built on."""
    d = np.array([-1.0, -1.1, -1.2, 5.0, 6.0, 7.0, 8.0, -1.3, -1.4, -1.5, -1.6])
    res = el.wilcoxon_paired(np.zeros(len(d)), d)
    # |d| ranks: the seven negatives take 1..7, the positives 8..11.
    # W+ = 8 + 9 + 10 + 11 = 38, E = 11*12/4 = 33, Var = 11*12*23/24 = 126.5.
    assert res["statistic"] == 38.0
    assert res["median_delta"] == pytest.approx(-1.1)
    assert res["z"] == pytest.approx(5.0 / np.sqrt(126.5))
    assert res["z"] > 0 > res["median_delta"]
    assert abs(res["z"]) == pytest.approx(abs(stats.wilcoxon(d, method="approx").zstatistic))

    # The disagreement is not only a small-sample curiosity: 51 small
    # decreases and 50 large increases give a negative median but a
    # significant excess of positive ranks, which a median-signed z would
    # report as a significant DECREASE.
    big = np.r_[-np.arange(1, 52) / 100.0, 1.0 + np.arange(50) / 100.0]
    res = el.wilcoxon_paired(np.zeros(len(big)), big)
    # Negatives take ranks 1..51 (sum 1326), positives 52..101 (sum 3825);
    # E = 101*102/4 = 2575.5, Var = 101*102*203/24 = 87137.75.
    assert res["median_delta"] < 0
    assert res["statistic"] == 3825.0
    assert res["z"] == pytest.approx((3825.0 - 2575.5) / np.sqrt(87137.75))
    assert res["z"] > 4.0 and res["p_value"] < 1e-4


def test_wilcoxon_round_off_ties_are_tied_as_in_matlab():
    """|d| = 0.30000000000000004 (0.4 - 0.1) and 0.3 (0.3 - 0) differ by one
    ulp: MATLAB tiedrank(abs(d), 0, 0, eps(x) + eps(y)) ties them, so must
    the ranks, the tie correction and the p-value."""
    pre = np.array([0.1, 0.0, 1.0, 2.0, 3.0])
    post = np.array([0.4, 0.3, 0.0, 2.5, 4.0])
    assert (post - pre)[0] != (post - pre)[1]
    res = el.wilcoxon_paired(pre, post)
    # Ties {0.3, 0.3} -> 1.5 each, 0.5 -> 3, {1, 1} -> 4.5 each; W+ = 10.5.
    # Var = 5*6*11/24 - ((2^3 - 2) + (2^3 - 2)) / 48 = 13.5.
    assert res["n_nonzero"] == 5 and res["statistic"] == 10.5
    assert res["z"] == pytest.approx(3.0 / np.sqrt(13.5))           # = 0.816497 by hand
    merged = np.array([0.3, 0.3, -1.0, 0.5, 1.0])
    assert res["p_value"] == pytest.approx(stats.wilcoxon(merged).pvalue)

    # A run of values each within tolerance of the next chains into one group.
    a = np.array([1.0, 1.0 + 2e-16, 1.0 + 4e-16, 2.0])
    out = el._merge_round_off_ties(a, np.full(4, 2.3e-16))
    np.testing.assert_array_equal(out, [1.0, 1.0, 1.0, 2.0])
    # Distinct values are left alone.
    np.testing.assert_array_equal(el._merge_round_off_ties(np.array([3.0, 1.0, 2.0]),
                                                           np.full(3, 1e-15)), [3.0, 1.0, 2.0])


def test_wilcoxon_degenerate_inputs():
    same = el.wilcoxon_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert same["n"] == 3 and same["n_nonzero"] == 0
    assert same["z"] == 0.0 and same["p_value"] == 1.0 and same["statistic"] == 0.0

    empty = el.wilcoxon_paired([np.nan, 1.0], [2.0, np.nan])
    assert empty["n"] == 0
    assert np.isnan(empty["z"]) and np.isnan(empty["p_value"])

    with pytest.raises(ValueError):
        el.wilcoxon_paired([1.0, 2.0], [1.0])


def test_paired_tests_run_every_key_with_bonferroni():
    t = _grid()
    rng = np.random.default_rng(0)
    tables = []
    for i in range(8):
        a = np.where(t < 60.0, 1.0, 2.0 + 0.1 * i) + rng.normal(0, 0.01, len(t))
        b = rng.normal(0, 1.0, len(t))
        tables.append(el.event_window_means(t, {"a": a, "b": b}, [60.0, 10.0]))
    df = pd.concat(tables, ignore_index=True)              # 8 valid + 8 edge events
    out = el.paired_tests(df)
    assert out["key"].tolist() == ["a", "b"]
    row_a = out.set_index("key").loc["a"]
    assert row_a["n"] == 8
    assert row_a["z"] == pytest.approx(
        el.wilcoxon_paired(df.loc[df.valid, "a_pre"], df.loc[df.valid, "a_post"])["z"])
    np.testing.assert_allclose(out["p_bonferroni"], np.minimum(2 * out["p_value"], 1.0))


# ════════════════════════════════════════════════════════════
#  calibrate_gain / knee_point
# ════════════════════════════════════════════════════════════

def _flip(sig: np.ndarray) -> np.ndarray:
    """MATLAB flipsig: mirror a signal around its mean."""
    return 2.0 * sig.mean() - sig


def test_calibrate_gain_matches_explicit_rcsi_rcpi_series():
    """delta(rCSI) = -delta(D) + ks*delta(SD2) and delta(rCPI) = delta(D) +
    kp*delta(SD1): the re-centring constants of compute_CSI_CVI_rat.m and the
    flip around mean(D) cancel, so per-event deltas suffice."""
    rng = np.random.default_rng(3)
    t = _grid(200.0)
    after = t >= 100.0
    k_values = [1, 3, 12]
    d_delta, sd1_delta, sd2_delta = [], [], []
    rcsi = {k: ([], []) for k in k_values}
    rcpi = {k: ([], []) for k in k_values}
    for _ in range(9):
        D = 120.0 + rng.normal(0, 3) * after + rng.normal(0, 0.5, len(t))
        SD1 = 2.0 + rng.normal(0, 0.3) * after + rng.normal(0, 0.05, len(t))
        SD2 = 6.0 + rng.normal(0, 0.8) * after + rng.normal(0, 0.1, len(t))
        D0, SD01, SD02 = rng.uniform(100, 140), rng.uniform(1, 3), rng.uniform(4, 8)
        Dc, SD1c, SD2c = (D - D.mean() + D0, SD1 - SD1.mean() + SD01,
                          SD2 - SD2.mean() + SD02)
        series = {"D": D, "SD1": SD1, "SD2": SD2}
        for k in k_values:
            series[f"rCSI{k}"] = _flip(Dc) + k * SD2c
            series[f"rCPI{k}"] = Dc + k * SD1c
        ev = el.event_window_means(t, series, [100.0]).iloc[0]
        assert ev["valid"]
        d_delta.append(ev["D_delta"])
        sd1_delta.append(ev["SD1_delta"])
        sd2_delta.append(ev["SD2_delta"])
        for k in k_values:
            assert ev[f"rCSI{k}_delta"] == pytest.approx(-ev["D_delta"] + k * ev["SD2_delta"])
            assert ev[f"rCPI{k}_delta"] == pytest.approx(ev["D_delta"] + k * ev["SD1_delta"])
            rcsi[k][0].append(ev[f"rCSI{k}_pre"])
            rcsi[k][1].append(ev[f"rCSI{k}_post"])
            rcpi[k][0].append(ev[f"rCPI{k}_pre"])
            rcpi[k][1].append(ev[f"rCPI{k}_post"])

    cal_csi = el.calibrate_gain(d_delta, sd2_delta, "csi", k_values=k_values)
    cal_cpi = el.calibrate_gain(d_delta, sd1_delta, "cpi", k_values=k_values)
    assert cal_csi["k"].tolist() == k_values
    assert (cal_csi["n"] == 9).all()
    assert cal_csi.attrs["kind"] == "csi" and cal_cpi.attrs["kind"] == "cpi"
    for i, k in enumerate(k_values):
        ref_csi = el.wilcoxon_paired(*rcsi[k])
        ref_cpi = el.wilcoxon_paired(*rcpi[k])
        assert cal_csi.loc[i, "z"] == pytest.approx(ref_csi["z"])
        assert cal_csi.loc[i, "p_value"] == pytest.approx(ref_csi["p_value"])
        assert cal_cpi.loc[i, "z"] == pytest.approx(ref_cpi["z"])
        assert cal_cpi.loc[i, "p_value"] == pytest.approx(ref_cpi["p_value"])


def test_calibrate_gain_drops_terms_that_cancel_to_round_off():
    """0.3 + 3 * -0.1 = -5.6e-17: the two terms of the index change cancel,
    and the explicit test on the rCPI window means sees a zero change. The
    calibration must drop it too, not rank it as a real negative change."""
    dd = np.array([0.3, 0.5, 0.7, 0.2, 0.9, 0.4])
    dsd1 = np.full(6, -0.1)
    assert dd[0] + 3 * dsd1[0] != 0.0                      # the round-off at stake
    cal = el.calibrate_gain(dd, dsd1, "cpi", k_values=[3]).iloc[0]
    assert cal["n"] == 6 and cal["n_nonzero"] == 5

    # Hand calculation on d = (0.2, 0.4, -0.1, 0.6, 0.1): |d| ranks 3, 4,
    # {1.5, 1.5}, 5; W+ = 3 + 4 + 5 + 1.5 = 13.5, E = 7.5,
    # Var = 5*6*11/24 - (2^3 - 2)/48 = 13.625.
    assert cal["z"] == pytest.approx(6.0 / np.sqrt(13.625))
    ref = el.wilcoxon_paired(np.zeros(5), [0.2, 0.4, -0.1, 0.6, 0.1])
    assert cal["z"] == pytest.approx(ref["z"]) and cal["p_value"] == pytest.approx(ref["p_value"])
    assert cal["median_delta"] == pytest.approx(0.15)

    # Same as the explicit test on rCPI = D + 3 * SD1 window means.
    t = _grid()
    after = t >= 60.0
    pre, post = [], []
    for d_change in dd:
        rcpi = (120.0 + d_change * after) + 3.0 * (2.0 - 0.1 * after)
        ev = el.event_window_means(t, {"rCPI": rcpi}, [60.0]).iloc[0]
        pre.append(ev["rCPI_pre"])
        post.append(ev["rCPI_post"])
    explicit = el.wilcoxon_paired(pre, post)
    assert explicit["n_nonzero"] == 5
    assert cal["z"] == pytest.approx(explicit["z"])
    assert cal["p_value"] == pytest.approx(explicit["p_value"])

    # CSI goes through the same path: -dD + k * dSD2 with dSD2 = +0.1.
    cal_csi = el.calibrate_gain(dd, -dsd1, "csi", k_values=[3]).iloc[0]
    assert cal_csi["n_nonzero"] == 5
    assert cal_csi["z"] == pytest.approx(-cal["z"])


def test_calibrate_gain_saturates_from_the_known_gain():
    # D changes have zero median and are spread over [-0.9, 0.9]; SD1 falls
    # by 0.2 in every event. The CPI change dd + k*dSD1 is negative in EVERY
    # event once 0.2*k > 0.9, i.e. from k = 5 on: |Z| then sits at its
    # maximum for n = 10 untied negatives and the knee is k = 5.
    dd = np.linspace(-0.9, 0.9, 10)
    cal = el.calibrate_gain(dd, np.full(10, -0.2), "cpi")
    assert cal["k"].tolist() == list(range(1, 21))
    z_max = (10 * 11 / 4) / np.sqrt(10 * 11 * 21 / 24)
    z = cal.set_index("k")["z"]
    np.testing.assert_allclose(z.loc[5:], -z_max)
    assert (z.loc[:4].abs() < z_max - 1e-9).all()
    assert np.all(np.diff(z.to_numpy()) <= 1e-12)          # monotone towards the plateau
    assert el.knee_point(cal["k"], cal["z"]) == 5.0

    # CSI mirrors it: -dd + k*dSD2 with dSD2 = +0.2.
    cal_csi = el.calibrate_gain(dd, np.full(10, 0.2), "csi")
    np.testing.assert_allclose(cal_csi.set_index("k")["z"].loc[5:], z_max)
    assert el.knee_point(cal_csi["k"], cal_csi["z"]) == 5.0


def test_calibrate_gain_sign_flip_when_hr_and_hrv_terms_oppose():
    # D rises (~1) while SD2 rises by 0.25: -dD + k*dSD2 is negative in every
    # event for k <= 3 and positive in every event for k >= 5.
    dd = 0.97 + 0.02 * np.arange(12)
    cal = el.calibrate_gain(dd, np.full(12, 0.25), "csi")
    z = cal.set_index("k")["z"]
    assert (z.loc[:3] < 0).all() and (z.loc[5:] > 0).all()
    # The knee is searched after the sign change, not at the dip of |z|.
    assert el.knee_point(cal["k"], cal["z"]) == 5.0


def test_calibrate_gain_drops_nan_events_and_validates_kind():
    cal = el.calibrate_gain([1.0, np.nan, -0.5, 0.2], [0.1, 0.1, np.nan, 0.3], "CVI",
                            k_values=[1, 2])
    assert (cal["n"] == 2).all()
    assert cal.attrs["kind"] == "cpi"
    with pytest.raises(ValueError):
        el.calibrate_gain([1.0], [1.0], "lf")
    with pytest.raises(ValueError):
        el.calibrate_gain([1.0, 2.0], [1.0], "csi")


def test_knee_point_on_a_saturating_curve():
    k = np.arange(1, 21)
    z = -3.0 * (1.0 - np.exp(-k / 3.0))
    knee = el.knee_point(k, z)
    # Kneedle optimum of the continuous curve: d(yn)/d(xn) = 1, i.e.
    # exp(-k/3) = (y(20) - y(1)) / 19  ->  k* = 6.54; nearest grid point 7.
    y = 3.0 * (1.0 - np.exp(-k / 3.0))
    k_star = -3.0 * np.log((y[-1] - y[0]) / 19.0)
    assert k_star == pytest.approx(6.54, abs=0.01)
    assert knee == 7.0
    assert abs(knee - k_star) <= 1.0

    # Faster saturation -> earlier knee, inside the paper's ks range.
    assert 2.0 <= el.knee_point(k, -3.0 * (1.0 - np.exp(-k / 1.5))) <= 6.0

    # Invariant to the sign and scale of z and to the order of the points.
    assert el.knee_point(k, -10.0 * z) == knee
    order = np.random.default_rng(0).permutation(len(k))
    assert el.knee_point(k[order], z[order]) == knee


def test_knee_point_after_a_sign_change_uses_the_final_stretch():
    k = np.arange(1, 21)
    # |z| falls from 6 through 0 (k = 4) and rises to a LOWER plateau of 3,
    # reached at k = 7: the raw |z| curve ends below its start, so the dip
    # is the farthest point on the plateau side of its chord.
    z = np.r_[-6.0, -4.0, -2.0, 0.0, np.minimum(np.arange(1, 17), 3.0)]
    assert el.knee_point(k, z) == 7.0                      # plateau reached at k = 7
    assert el.knee_point(k, z, after_sign_change=False) == 4.0  # raw |z| picks the dip
    # With a plateau above the start the signed distance alone avoids the dip.
    z_up = np.r_[-2.0, -1.0, 0.0, np.minimum(np.arange(1, 18), 4.0)]
    assert el.knee_point(k, z_up) == 7.0
    assert el.knee_point(k, z_up, after_sign_change=False) == 7.0


def test_knee_point_takes_the_shoulder_of_a_late_rise():
    """Z(k) moves from the heart-rate term's Z to the HRV term's Z around
    k ~ |dD| / |dSD|, late in a 1..20 sweep for mice. The knee is where the
    rise flattens, never the foot where it starts (below the chord)."""
    k = np.arange(1, 21)
    for centre, expected in ((6, 10.0), (12, 16.0), (14, 18.0)):
        z = 1.0 + 2.0 / (1.0 + np.exp(-(k - centre) / 1.5))
        knee = el.knee_point(k, z)
        assert knee == expected
        assert knee > centre                               # on the plateau side
        y = np.abs(z)                                      # first k at 95 % of the rise
        k95 = k[np.argmax(y >= y.min() + 0.95 * (y.max() - y.min()))]
        assert abs(knee - k95) <= 1.0
        assert el.knee_point(k, -z) == knee                # sign of z irrelevant

    # |Z| decaying onto a plateau (the HR term outweighs the HRV term):
    # the knee is where the fall flattens.
    assert el.knee_point(k, 1.0 + 2.0 * np.exp(-k / 3.0)) == 7.0


def test_knee_point_has_no_knee_on_an_accelerating_curve():
    k = np.arange(1, 21)
    assert np.isnan(el.knee_point(k, (k / 20.0) ** 2))            # rising, no plateau
    assert np.isnan(el.knee_point(k, 3.0 - (k / 20.0) ** 2))      # falling ever faster
    assert np.isnan(el.knee_point(k, np.exp(k / 4.0)))


def test_knee_point_refuses_a_knee_with_no_plateau_after_it():
    # A rise that keeps accelerating, whose last two points share one rank
    # step: Kneedle alone puts the "knee" at the second-to-last k.
    k = np.arange(1, 21)
    z = (k / 20.0) ** 2
    z[-1] = z[-2]
    assert el.knee_point(k, z, min_plateau_points=1) == 19.0
    assert np.isnan(el.knee_point(k, z))
    # the same with a three-point plateau is a real (if late) knee
    z3 = (k / 20.0) ** 2
    z3[-2:] = z3[-3]
    assert el.knee_point(k, z3) == 18.0


def test_knee_point_degenerate_curves_return_nan():
    k = np.arange(1, 21)
    assert np.isnan(el.knee_point(k, np.full(20, -2.5)))             # flat
    assert np.isnan(el.knee_point(k, 0.5 * k))                       # straight line
    assert np.isnan(el.knee_point(k, np.zeros(20)))                  # all zero
    assert np.isnan(el.knee_point([1, 2], [0.5, 1.0]))               # too short
    assert np.isnan(el.knee_point(k, np.full(20, np.nan)))           # nothing finite
    with pytest.raises(ValueError):
        el.knee_point([1, 1, 2], [1.0, 2.0, 3.0])


# ════════════════════════════════════════════════════════════
#  aggregate / bonferroni
# ════════════════════════════════════════════════════════════

def test_aggregate_averages_valid_events_per_subject():
    df = pd.DataFrame({
        "subject": ["m2", "m1", "m2", "m1", "m2"],
        "label":   ["a", "b", "c", "d", "e"],
        "valid":   [True, True, False, True, True],
        "onset_s": [10.0, 20.0, 30.0, 40.0, 50.0],
        "x_delta": [10.0, 1.0, 1000.0, 3.0, np.nan],
    })
    agg = el.aggregate(df, "subject")
    assert agg["subject"].tolist() == ["m1", "m2"]
    assert agg["n_events"].tolist() == [2, 2]
    assert "valid" not in agg.columns and "label" not in agg.columns
    np.testing.assert_allclose(agg["x_delta"], [2.0, 10.0])          # NaN skipped

    everything = el.aggregate(df, "subject", valid_only=False)
    assert everything["n_events"].tolist() == [2, 3]
    np.testing.assert_allclose(everything["x_delta"], [2.0, 505.0])

    df["condition"] = ["dob", "dob", "dob", "sham", "dob"]
    two = el.aggregate(df, ["subject", "condition"])
    assert list(two.columns[:3]) == ["subject", "condition", "n_events"]
    assert len(two) == 3
    with pytest.raises(KeyError):
        el.aggregate(df, "animal")


def test_aggregate_then_test_per_animal():
    t = _grid()
    tables = []
    for animal in range(6):
        y = np.where(t < 60.0, 1.0, 1.5 + 0.1 * animal)
        ev = pd.concat([el.event_window_means(t, {"y": y}, [o])      # 2 trials each,
                        for o in (60.0, 61.0)], ignore_index=True)  # separate sessions
        ev["subject"] = f"m{animal}"
        tables.append(ev)
    per_animal = el.aggregate(pd.concat(tables, ignore_index=True), "subject")
    assert len(per_animal) == 6 and (per_animal["n_events"] == 2).all()
    res = el.wilcoxon_paired(per_animal["y_pre"], per_animal["y_post"])
    assert res["n"] == 6 and res["z"] > 0


def test_aggregate_twice_per_dose_then_per_subject():
    t = _grid()
    tables = []
    for animal in range(4):
        for dose, onset, trials in ((20, 40.0, 2), (40, 80.0, 3)):
            y = np.where(t < onset, 1.0, 1.0 + dose / 10.0 + 0.1 * animal)
            # trials of separate sessions: each table holds one event
            ev = pd.concat([el.event_window_means(t, {"y": y}, [o])
                            for o in onset + np.arange(trials) * 0.5], ignore_index=True)
            ev["subject"], ev["dose"] = f"m{animal}", dose
            tables.append(ev)
    df = pd.concat(tables, ignore_index=True)
    per_dose = el.aggregate(df, ["subject", "dose"])
    assert per_dose["n_events"].tolist() == [2, 3] * 4
    per_subject = el.aggregate(per_dose, "subject")       # used to raise ValueError
    assert per_subject["subject"].tolist() == ["m0", "m1", "m2", "m3"]
    assert per_subject["n_events"].tolist() == [5] * 4     # original events, summed
    assert list(per_subject.columns[:2]) == ["subject", "n_events"]
    # Each dose weighs equally: the mean of the two per-dose means.
    expected = per_dose.groupby("subject")["y_delta"].mean().to_numpy()
    np.testing.assert_allclose(per_subject["y_delta"], expected)


def test_bonferroni_scales_and_clips():
    np.testing.assert_allclose(el.bonferroni([0.01, 0.02, 0.5]), [0.03, 0.06, 1.0])
    adj = el.bonferroni(np.array([0.01, np.nan, 0.2]))   # NaN neither counted nor adjusted
    np.testing.assert_allclose(adj[[0, 2]], [0.02, 0.4])
    assert np.isnan(adj[1])
    s = pd.Series([0.04, 0.3], index=["CSI", "CPI"], name="p")
    out = el.bonferroni(s)
    assert isinstance(out, pd.Series) and out.index.tolist() == ["CSI", "CPI"]
    np.testing.assert_allclose(out.to_numpy(), [0.08, 0.6])
    assert len(el.bonferroni([])) == 0


# ════════════════════════════════════════════════════════════
#  Isolation of events (the paper's minimum 30 s per condition)
# ════════════════════════════════════════════════════════════

def _two_step(t: np.ndarray) -> np.ndarray:
    """1 before 100 s, a +1 step at 100 s and a +5 step at 110 s."""
    return 1.0 + (t >= 100.0) + 5.0 * (t >= 110.0)


def test_isolation_overlapping_events_are_invalid_for_every_series():
    t = np.arange(0, 4000) / FS                            # 0 .. 199.95 s
    y = _two_step(t)
    df = el.event_window_means(t, {"x": y, "c": np.ones_like(t)}, [100.0, 110.0])
    assert df["isolated"].tolist() == [False, False]
    assert df["valid"].tolist() == [False, False]
    assert df["any_valid"].tolist() == [False, False]
    assert not df["x_valid"].any() and not df["c_valid"].any()
    assert df["x_delta"].isna().all() and df["c_pre"].isna().all()
    assert df["next_gap_s"].iloc[0] == pytest.approx(10.0)
    assert df["prev_gap_s"].iloc[1] == pytest.approx(10.0)
    assert np.isnan(df["prev_gap_s"].iloc[0]) and np.isnan(df["next_gap_s"].iloc[1])
    # Without the rule each event absorbs the other change: the reported
    # step at 100 s would be 4.34 instead of 1, the baseline at 110 s 1.33.
    raw = el.event_window_means(t, {"x": y}, [100.0, 110.0], require_isolated=False)
    assert raw["valid"].tolist() == [True, True]
    assert raw["isolated"].tolist() == [False, False]      # still reported
    assert raw["x_delta"].iloc[0] == pytest.approx(4.336, abs=1e-3)
    assert raw["x_pre"].iloc[1] == pytest.approx(1.333, abs=1e-3)
    # The same pair with 10 s / 9 s windows is isolated, and exact.
    short = el.event_window_means(t, {"x": y}, [100.0, 110.0], pre_s=10.0, post_s=9.0)
    assert short["isolated"].tolist() == [True, True] and short["valid"].all()
    np.testing.assert_allclose(short["x_delta"], [1.0, 5.0])


def test_isolation_boundaries_follow_the_window_boundaries():
    t = _grid(200.0)
    y = np.ones_like(t)
    # 60 -> 90 is exactly 30 s: 90's pre window [60, 90) is the condition
    # that STARTED at 60 (isolated on that side), but the change at 90 puts
    # its first sample at the end of 60's closed post window [60, 90].
    df = el.event_window_means(t, {"y": y}, [60.0, 90.0, 150.0])
    assert df["isolated"].tolist() == [False, True, True]
    assert df["valid"].tolist() == [False, True, True]
    # A change a hair beyond the post window leaves the event isolated ...
    df2 = el.event_window_means(t, {"y": y}, [60.0], other_onsets=[90.0 + 1e-3])
    assert bool(df2["isolated"].iloc[0]) and bool(df2["valid"].iloc[0])
    # ... and one a hair inside the pre window does not.
    df3 = el.event_window_means(t, {"y": y}, [60.0], other_onsets=[30.0 + 1e-3])
    assert not bool(df3["isolated"].iloc[0]) and not bool(df3["valid"].iloc[0])


def test_isolation_counts_other_condition_changes():
    t = np.arange(0, 4000) / FS
    y = _two_step(t)
    # One event of interest, the other change given as an other onset
    # (e.g. an annotation of another label filtered out of the events).
    alone = el.event_window_means(t, {"x": y}, [100.0])
    assert bool(alone["valid"].iloc[0])                    # the change at 110 is unknown
    known = el.event_window_means(t, {"x": y}, [100.0], other_onsets=[110.0, np.nan])
    assert not bool(known["isolated"].iloc[0]) and not bool(known["valid"].iloc[0])
    assert known["next_gap_s"].iloc[0] == pytest.approx(10.0)
    far = el.event_window_means(t, {"x": y}, [100.0], other_onsets=[20.0, 180.0])
    assert bool(far["isolated"].iloc[0]) and bool(far["valid"].iloc[0])


def test_isolation_coincident_onsets_are_never_isolated():
    t = _grid(200.0)
    y = np.where(t < 100.0, 1.0, 2.0)
    dup = el.event_window_means(t, {"y": y}, [100.0, 100.0], labels=["a", "b"])
    assert dup["isolated"].tolist() == [False, False]
    assert dup["valid"].tolist() == [False, False]
    assert dup["prev_gap_s"].tolist() == [0.0, 0.0] and dup["next_gap_s"].tolist() == [0.0, 0.0]
    other = el.event_window_means(t, {"y": y}, [100.0], other_onsets=[100.0])
    assert not bool(other["isolated"].iloc[0]) and not bool(other["valid"].iloc[0])
    # A NaN onset is neither isolated nor a condition change for the others.
    nan_on = el.event_window_means(t, {"y": y}, [np.nan, 100.0])
    assert nan_on["isolated"].tolist() == [False, True]
    assert nan_on["valid"].tolist() == [False, True]


def test_isolation_flag_is_stored_in_the_attrs():
    t = _grid()
    df = el.event_window_means(t, {"y": np.ones_like(t)}, [60.0])
    assert df.attrs["require_isolated"] is True
    assert el.event_window_means(t, {"y": np.ones_like(t)}, [60.0],
                                 require_isolated=False).attrs["require_isolated"] is False


# ════════════════════════════════════════════════════════════
#  Rank-statistic caveats of the calibration
# ════════════════════════════════════════════════════════════

def test_wilcoxon_reports_the_normal_p_of_z_next_to_the_exact_p():
    post = np.r_[np.arange(1.0, 11.0), -11.0, -12.0]
    res = el.wilcoxon_paired(np.zeros(12), post)
    assert res["z"] == pytest.approx(1.255, abs=1e-3)
    assert res["p_method"] == "exact"
    assert res["p_value"] == pytest.approx(0.2334, abs=1e-4)          # scipy exact
    assert res["p_normal"] == pytest.approx(2 * stats.norm.sf(res["z"]))
    assert res["p_normal"] == pytest.approx(0.2094, abs=1e-4)
    assert res["z_max"] == pytest.approx(np.sqrt(3 * 12 * 13 / (2 * 25)))  # 3.06
    tied = el.wilcoxon_paired(np.zeros(6), [1.0, 1.0, 2.0, -3.0, 4.0, 5.0])
    assert tied["p_method"] == "exact permutation"
    signs = np.where(np.arange(60) % 3 == 1, -1.0, 1.0)
    big = el.wilcoxon_paired(np.zeros(60), np.arange(1.0, 61.0) * signs)
    assert big["p_method"] == "normal approximation"
    none = el.wilcoxon_paired([1.0, 2.0], [1.0, 2.0])
    assert none["p_normal"] == 1.0 and none["z_max"] == 0.0
    t = el.paired_tests(pd.DataFrame({"a_pre": np.zeros(12), "a_post": post,
                                      "a_delta": post}))
    assert {"p_normal", "p_method", "z_max", "p_bonferroni"} <= set(t.columns)
    for n, ceiling in ((6, 2.20), (8, 2.52)):
        res_n = el.wilcoxon_paired(np.zeros(n), np.arange(1.0, n + 1))
        assert res_n["z_max"] == pytest.approx(ceiling, abs=5e-3)
        assert res_n["z"] == pytest.approx(res_n["z_max"])


def test_calibrate_gain_reports_the_rank_sum_and_the_ceiling():
    rng = np.random.default_rng(3)
    dd, ds = rng.normal(-4, 1.5, 8), rng.normal(-0.4, 0.25, 8)
    cal = el.calibrate_gain(dd, ds, "cpi")
    assert list(cal.columns) == ["k", "z", "p_value", "n", "n_nonzero", "median_delta",
                                 "statistic", "p_normal", "z_max"]
    z_max = np.sqrt(3 * 8 * 9 / (2 * 17))
    np.testing.assert_allclose(cal["z_max"], z_max)
    assert (np.abs(cal["z"]) <= z_max + 1e-12).all()
    np.testing.assert_allclose(cal["z"], (cal["statistic"] - 18.0) / np.sqrt(51.0))


def test_knee_point_refuses_a_knee_of_one_rank_step():
    """The review's case: 8 subjects, Z = -2.52 (the ceiling) for k = 1..11
    and -2.38 (one subject's rank swapped, W+ + 1) for k = 12..20. Kneedle
    alone puts the knee on that single rank step."""
    k = np.arange(1, 21)
    w = np.r_[np.zeros(11), np.ones(9)]                    # W+ = 0, then 1
    z = (w - 18.0) / np.sqrt(51.0)
    assert z[0] == pytest.approx(-2.52, abs=5e-3) and z[-1] == pytest.approx(-2.38, abs=5e-3)
    assert el.knee_point(k, z) == 12.0
    assert np.isnan(el.knee_point(k, z, statistic=w, min_rank_steps=3))
    # A real plateau (W+ climbing over many rank steps) keeps its knee.
    w2 = np.minimum(np.arange(1, 21) * 4.0, 36.0)
    z2 = (w2 - 18.0) / np.sqrt(51.0)
    assert el.knee_point(k, z2, statistic=w2, min_rank_steps=3) == el.knee_point(k, z2)
    with pytest.raises(ValueError):
        el.knee_point(k, z, statistic=w[:-1])


def test_sign_stretches_find_a_reversal_of_the_index():
    k = np.arange(1, 14)
    z = np.array([2.8, 2.8, 2.8, 2.8, 2.8, 2.6, 1.58, 0.56, -0.15, -1.17, -1.48, -1.89, -2.09])
    runs = el.sign_stretches(k, z)
    assert [r["sign"] for r in runs] == [1, -1]
    assert runs[0]["k_first"] == 1.0 and runs[0]["k_last"] == 8.0
    assert runs[0]["z_peak"] == 2.8 and runs[0]["k_peak"] == 1.0
    assert runs[1]["k_first"] == 9.0 and runs[1]["k_last"] == 13.0
    assert runs[1]["z_peak"] == -2.09 and runs[1]["k_peak"] == 13.0
    assert len(el.sign_stretches(k, np.abs(z))) == 1
    assert el.sign_stretches([1, 2, 3], [0.0, np.nan, 0.0]) == []
    # the knee knee_point offers lies on the reversed (last) run
    assert el.knee_point(k, z) >= 9.0


# ════════════════════════════════════════════════════════════
#  Layering
# ════════════════════════════════════════════════════════════

def test_event_locked_module_does_not_import_the_ui():
    code = ("import sys, ecg.core.event_locked; "
            "bad = [m for m in sys.modules if m == 'ecg.ui' or m.startswith('ecg.ui.') "
            "or m in ('customtkinter', 'tkinter')]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
