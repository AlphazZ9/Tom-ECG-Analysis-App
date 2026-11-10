"""
ecg.core.event_locked
─────────────────────
Event-locked statistics for time-resolved autonomic indices (CSI / CPI,
LF / HF, D, SD1, SD2, heart rate ...), following the statistical protocol of
Candia-Rivera et al., "Modelling the time-resolved modulations of cardiac
activity in rats", J Physiol 2025 (reference code:
github.com/diegocandiar/robust_hrv).

This module works on GENERIC, evenly resampled series -- any dict of
``{name: values}`` sharing one time axis ``t`` (seconds), e.g. the 20 Hz
outputs of a CSI/CPI or pseudo-Wigner-Ville computation. It does not compute
those series itself, and holds no state; everything is pure NumPy / SciPy /
pandas and safe to call from a background thread.

Protocol implemented (paper, "Statistical analysis"):

* each event (a condition change such as an injection) is condensed into the
  MEAN of every series over the 30 s before and the 30 s after its onset
  (``event_window_means``). As in the paper ("a minimum duration of 30 s
  for each" of the pre- and post-condition periods), an event is only kept
  when no other condition change -- another event, or an *other_onsets*
  change such as an annotation of another label -- falls inside its
  windows, and coincident changes are never kept, so successive doses
  neither contaminate each other's baseline / response nor count one
  condition's samples in two conditions of a contrast
  (``require_isolated``, the default);
* conditions are compared with a two-sided Wilcoxon signed-rank test for
  paired samples (``wilcoxon_paired``), reported with the group median and
  median absolute deviation (MAD); several comparisons are Bonferroni
  corrected (``bonferroni``);
* for visualisation, each subject's series is z-scored, cut around every
  onset on a common relative time grid, and summarised as the group median
  +- MAD per relative time (``event_locked_traces``, ``group_summary``);
* the gains ks (CSI) and kp (CPI) are chosen from the Wilcoxon Z of the
  condition contrast as a function of k = 1..20, at the "inflection point"
  of Z(k) from which it plateaus (``calibrate_gain``, ``knee_point``). The
  paper computed Z over trials (41 SHR trials), one contrast (rest vs
  dobutamine), and validated the gains on another strain (WKY): gains
  chosen on a cohort and tested on the same cohort give optimistic
  p-values, and Z(k) is a rank statistic with a ceiling (``z_max``) that a
  small cohort sits on (``knee_point``'s *min_rank_steps*);
  ``sign_stretches`` flags a Z(k) that changes sign, where the knee makes
  the index move opposite to its small-k direction.

Gain calibration rests on the linearity of the indices in k. With D the
distance of the Poincare ellipse centre to the origin and SD1/SD2 its minor
and major semi-axes (all re-centred on whole-recording values),

    rCPI(t) = D(t) + kp * SD1(t)
    rCSI(t) = flip(D)(t) + ks * SD2(t),   flip(D) = 2 * mean(D) - D

so for any pre/post window pair the re-centring constants and mean(D) cancel:

    delta(rCPI) =  delta(D) + kp * delta(SD1)
    delta(rCSI) = -delta(D) + ks * delta(SD2)

The per-event deltas of D, SD1 and SD2 are therefore sufficient to re-run
the calibration over a whole cohort without recomputing any series. D and
SD must share one unit (s or ms); the Z(k) curve does not depend on which.

The paper's rat calibration (ks = 3, kp = 12; SHR, rest vs dobutamine) is
NOT validated in mice: ``calibrate_gain`` exists so that a cohort's own
gains can be derived and disclosed rather than borrowed.

Statistics should be run per animal rather than per trial when an animal
contributes several events (``aggregate``), which avoids pseudo-replication.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "event_window_means",
    "event_locked_traces",
    "stack_traces",
    "group_summary",
    "wilcoxon_paired",
    "paired_tests",
    "calibrate_gain",
    "sign_stretches",
    "knee_point",
    "aggregate",
    "bonferroni",
]

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


# ════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ════════════════════════════════════════════════════════════

def _time_axis(t: ArrayLike) -> np.ndarray:
    """*t* as a 1-D float array; ValueError unless it has >= 2 finite,
    strictly increasing samples."""
    tt = np.asarray(t, dtype=np.float64)
    if tt.ndim != 1 or len(tt) < 2:
        raise ValueError("t must be a 1-D time axis with at least 2 samples.")
    if not np.all(np.isfinite(tt)):
        raise ValueError("t must contain only finite values.")
    if not np.all(np.diff(tt) > 0):
        raise ValueError("t must be strictly increasing.")
    return tt


def _series_arrays(series: Mapping[str, ArrayLike], n: int) -> "dict[str, np.ndarray]":
    """Every series as a 1-D float array of length *n* (ValueError otherwise)."""
    out: dict[str, np.ndarray] = {}
    for key, values in series.items():
        y = np.asarray(values, dtype=np.float64)
        if y.ndim != 1 or len(y) != n:
            raise ValueError(
                f"Series {key!r} must be 1-D with the same length as t "
                f"({n}), got shape {y.shape}.")
        out[str(key)] = y
    return out


def _onset_array(onsets: Union[float, ArrayLike]) -> np.ndarray:
    return np.atleast_1d(np.asarray(onsets, dtype=np.float64)).flatten()


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not (np.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return v


def _fraction(name: str, value: float) -> float:
    """*value* as a float in (0, 1]; ValueError otherwise (NaN included)."""
    v = float(value)
    if not (np.isfinite(v) and 0.0 < v <= 1.0):
        raise ValueError(f"{name} must lie in (0, 1], got {value!r}.")
    return v


def _onset_gaps(
    ons: np.ndarray, others: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """(prev_gap, next_gap) of every onset: the time from the nearest OTHER
    finite condition change (an onset of *ons* or of *others*) at or before
    it, and to the nearest one at or after it. NaN when there is none, and
    for a non-finite onset; a coincident change (a duplicate onset) gives 0
    both ways."""
    changes = np.sort(np.concatenate([ons[np.isfinite(ons)], others[np.isfinite(others)]]))
    prev_gap = np.full(len(ons), np.nan)
    next_gap = np.full(len(ons), np.nan)
    for j, onset in enumerate(ons):
        if not np.isfinite(onset):
            continue
        lo = int(np.searchsorted(changes, onset, side="left"))
        hi = int(np.searchsorted(changes, onset, side="right"))
        if hi - lo > 1:                       # another change at the same time
            prev_gap[j] = next_gap[j] = 0.0
            continue
        if lo > 0:
            prev_gap[j] = onset - changes[lo - 1]
        if hi < len(changes):
            next_gap[j] = changes[hi] - onset
    return prev_gap, next_gap


def _nanmean(y: np.ndarray) -> float:
    """Mean of the finite values of *y*; NaN (no warning) when there are none."""
    y = y[np.isfinite(y)]
    return float(y.mean()) if len(y) else float("nan")


def _median_mad(x: np.ndarray) -> "tuple[float, float]":
    """(median, unscaled MAD) of the finite values of *x*; NaNs when empty."""
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)))


# Event-level isolation columns of an ``event_window_means`` table.
_ISOLATION_COLS = ("isolated", "prev_gap_s", "next_gap_s")
# Per-series columns of an ``event_window_means`` table, in order.
_SERIES_SUFFIXES =("_pre", "_post", "_delta", "_coverage_pre", "_coverage_post", "_valid")
_VALUE_SUFFIXES = ("_pre", "_post", "_delta")


def _series_keys(df: pd.DataFrame) -> "list[str]":
    """Every ``k`` of an event table having ``k_pre``, ``k_post`` and
    ``k_delta`` columns (so ``coverage_pre`` / ``k_coverage_pre`` are not
    mistaken for a series), in column order."""
    cols = set(df.columns)
    return [c[:-4] for c in df.columns
            if isinstance(c, str) and c.endswith("_pre")
            and f"{c[:-4]}_post" in cols and f"{c[:-4]}_delta" in cols]


def _flag(column: pd.Series) -> np.ndarray:
    """A validity column as a bool array; a missing (NaN / None) flag is False."""
    return np.array([False if pd.isna(v) else bool(v) for v in column], dtype=bool)


# A time axis is EVENLY sampled when every step is within this fraction of
# the median step (round-off and sub-0.1 % jitter pass, a dropped sample --
# a double step -- does not).
_EVEN_STEP_RTOL = 1e-3


def _grid_step(tt: np.ndarray) -> float:
    """Sampling step of *tt* used to count each window's nominal samples.

    On an evenly sampled axis it is the mean step ``(t[-1] - t[0]) / (n - 1)``,
    exact to about ``ulp(t[-1]) / n``. The median step is only exact to
    about ``ulp(t[-1])``, since each step is the difference of two rounded
    times: ~7e-11 of a step at 20 Hz after 12 h, which a 1e-6 boundary
    tolerance no longer absorbs once multiplied by the 36 000 steps of a
    30 min window. An unevenly sampled axis (a dropped sample, say) falls
    back to the median step."""
    steps = np.diff(tt)
    med = float(np.median(steps))
    if np.all(np.abs(steps - med) <= _EVEN_STEP_RTOL * med):
        return float((tt[-1] - tt[0]) / (len(tt) - 1))
    return med


def _nominal_counts(
    tt: np.ndarray, dt: float, a: float, onset: float, b: float,
) -> "tuple[int, int]":
    """Nominal sample counts of the windows ``[a, onset)`` and ``[onset, b]``.

    They are the numbers of points of the regular grid ``ref + i * dt``
    (i any integer, so the grid extends beyond the recording) in each
    window, a point within 1e-6 step of a boundary lying on it, as for the
    samples themselves. The grid is anchored at ``ref``, the first sample at
    or after the onset (the last sample for an onset past the end): a sample
    next to the windows rather than ``t[0]``, so the step's own round-off
    builds up over one window and not over the whole recording, where it
    would shift the boundaries of a late window by more than the 1e-6
    tolerance and miscount it by one sample."""
    ref = float(tt[min(len(tt) - 1, int(np.searchsorted(tt, onset)))])

    def _first(x: float) -> int:              # first grid point >= x (- tol)
        return int(np.ceil((x - ref) / dt - 1e-6))

    g_onset = _first(onset)
    g_b = int(np.floor((b - ref) / dt + 1e-6))  # last grid point <= b (+ tol)
    return max(1, g_onset - _first(a)), max(1, g_b - g_onset + 1)


# ════════════════════════════════════════════════════════════
#  PER-EVENT WINDOW MEANS
# ════════════════════════════════════════════════════════════

def event_window_means(
    t:            ArrayLike,
    series:       Mapping[str, ArrayLike],
    onsets:       Union[float, ArrayLike],
    labels:       Optional[Sequence[object]] = None,
    pre_s:        float = 30.0,
    post_s:       float = 30.0,
    min_coverage: float = 0.8,
    *,
    mask_invalid:     bool = True,
    require_isolated: bool = True,
    other_onsets:     Union[float, ArrayLike, None] = None,
) -> pd.DataFrame:
    """Mean of every series over the window before and after each onset.

    *t* is the common, evenly sampled time axis (s) of the *series* dict
    (``{name: values}``, each the same length as *t*; NaN marks a gap).
    For an onset ``o`` the pre window is ``[o - pre_s, o)`` and the post
    window ``[o, o + post_s]``; a sample within a micro-tolerance of a
    boundary (1e-6 of the sampling step) is treated as lying on it, so a
    sample at the onset always belongs to the post window.

    Isolation (the paper's "minimum duration of 30 s for each" condition).
    The condition changes are every finite onset of *onsets* plus every
    finite *other_onsets* -- changes that are not events of this table but
    still start a new condition, e.g. the annotations of other labels when
    the events are filtered to one label (they must not repeat an onset of
    *onsets*: a repeated onset is a coincident change). An event is
    ISOLATED when no other change falls inside its windows: the nearest
    earlier change is at least ``pre_s`` before the onset (a change exactly
    at ``o - pre_s`` starts the condition the pre window measures) and the
    nearest later one more than ``post_s`` after it (a change at
    ``o + post_s`` would put its first sample in the post window), to the
    same boundary tolerance. A coincident change (the same onset twice, or
    an *other_onsets* entry at the onset) is never isolated. With
    *require_isolated* (the default) an event that is not isolated is
    invalid for every series (``valid``, ``any_valid`` and every
    ``k_valid`` False), so successive doses neither contaminate each
    other's baseline / response nor count one condition's samples in two
    conditions of one contrast; pass False to judge coverage only (the
    ``isolated`` column is filled either way).

    Coverage of a window is a count of finite samples divided by the
    window's nominal sample count -- the number of points of the regular
    sampling grid, extended beyond the recording, that fall in the window --
    capped at 1. Samples that would lie outside the recording therefore
    count as missing. The grid has the axis' step (its mean step when it is
    evenly sampled, else its median step) and passes through the sample at
    (or first after) the onset, so the count is exact on recordings of any
    length (``_nominal_counts``).

    Event-level columns:

    * ``coverage_pre`` / ``coverage_post``: the JOINT coverage, by the
      samples at which EVERY series is finite (1.0 = the event is complete
      for all series);
    * ``valid``: the onset is finite, the event is isolated (unless
      *require_isolated* is False), both windows lie entirely inside
      ``[t[0], t[-1]]``, and both joint coverages reach *min_coverage* --
      the event is usable for every series at once (``valid`` implies every
      ``k_valid``). With no series the joint coverage is that of the
      recording's samples, so ``valid`` reduces to the windows lying inside;
    * ``any_valid``: at least one ``k_valid`` is True -- the event
      contributes to some series' statistic. It equals ``valid`` with a
      single series (or none);
    * ``isolated``: no other condition change inside the windows (above);
    * ``prev_gap_s`` / ``next_gap_s``: time from the nearest other change
      at or before the onset / to the nearest one at or after it (NaN when
      there is none, 0 for a coincident change).

    Validity is also judged PER SERIES, so a series that is missing or gappy
    around an event (e.g. an LF/HF series that could not be computed)
    makes the event ``valid=False`` but never costs the other series that
    event:

    * ``k_coverage_pre`` / ``k_coverage_post``: coverage by the finite
      samples of series ``k``;
    * ``k_valid``: the onset is finite, the event is isolated (unless
      *require_isolated* is False), both windows lie entirely inside
      ``[t[0], t[-1]]``, and both of ``k``'s coverages reach *min_coverage*;
    * ``k_pre`` / ``k_post``: mean of ``k``'s own finite samples in each
      window, ``k_delta = k_post - k_pre``. With *mask_invalid* (the
      default) they are NaN when ``k_valid`` is False, so the series drops
      out of that event's statistics on its own (pairwise deletion: the
      number of events tested can differ between series, ``paired_tests``
      reports it per key); pass ``mask_invalid=False`` to inspect the raw
      means of edge or gappy events.

    So, to test one series use the rows where ``k_valid`` is True (as
    ``paired_tests`` does); to keep the events where the series are all
    usable together (complete-case analysis) filter on ``valid``; ``aggregate``
    drops only the events with no usable series (``any_valid`` False).

    Returns one row per onset (in the order given) with columns
    ``label, onset_s, valid, coverage_pre, coverage_post, any_valid,
    isolated, prev_gap_s, next_gap_s`` followed by, for every series key
    ``k``, ``k_pre, k_post, k_delta, k_coverage_pre, k_coverage_post,
    k_valid``. *labels* defaults to ``"Event 1"``, ``"Event 2"``, ... A
    series key whose column names would collide with another column (e.g.
    a key ``"coverage"`` or ``"any"``) raises ValueError.
    """
    tt = _time_axis(t)
    ys = _series_arrays(series, len(tt))
    ons = _onset_array(onsets)
    others = (np.array([], dtype=np.float64) if other_onsets is None
              else _onset_array(other_onsets))
    pre_s = _positive("pre_s", pre_s)
    post_s = _positive("post_s", post_s)
    min_coverage = float(min_coverage)
    if labels is None:
        labs: list[object] = [f"Event {i + 1}" for i in range(len(ons))]
    else:
        labs = list(labels)
        if len(labs) != len(ons):
            raise ValueError(
                f"labels has {len(labs)} entries for {len(ons)} onsets.")

    columns = ["label", "onset_s", "valid", "coverage_pre", "coverage_post", "any_valid",
               *_ISOLATION_COLS]
    for key in ys:
        columns += [f"{key}{suffix}" for suffix in _SERIES_SUFFIXES]
    clashes = sorted({c for c in columns if columns.count(c) > 1})
    if clashes:
        raise ValueError(
            f"Series keys {list(ys)} give clashing column names {clashes}; "
            "rename the series.")

    dt = _grid_step(tt)
    tol = 1e-6 * dt
    finite = {key: np.isfinite(y) for key, y in ys.items()}
    joint = np.ones(len(tt), dtype=bool)
    for f in finite.values():
        joint &= f
    prev_gaps, next_gaps = _onset_gaps(ons, others)
    # NaN gap (no other change on that side) compares False: isolated there.
    with np.errstate(invalid="ignore"):
        isolated_all = ~((prev_gaps < pre_s - tol) | (next_gaps <= post_s + tol))

    nan = float("nan")
    rows = []
    for onset, lab, prev_gap, next_gap, isolated in zip(
            ons, labs, prev_gaps, next_gaps, isolated_all):
        row: dict[str, object] = {"label": lab, "onset_s": float(onset),
                                  "isolated": bool(isolated and np.isfinite(onset)),
                                  "prev_gap_s": float(prev_gap),
                                  "next_gap_s": float(next_gap)}
        if not np.isfinite(onset):
            row.update(valid=False, coverage_pre=0.0, coverage_post=0.0, any_valid=False)
            for key in ys:
                row.update({f"{key}_pre": nan, f"{key}_post": nan, f"{key}_delta": nan,
                            f"{key}_coverage_pre": 0.0, f"{key}_coverage_post": 0.0,
                            f"{key}_valid": False})
            rows.append(row)
            continue
        a, b = onset - pre_s, onset + post_s
        i0 = int(np.searchsorted(tt, a - tol, side="left"))
        i1 = int(np.searchsorted(tt, onset - tol, side="left"))
        i2 = int(np.searchsorted(tt, b + tol, side="right"))
        n_nom_pre, n_nom_post = _nominal_counts(tt, dt, a, onset, b)
        inside = (a >= tt[0] - tol) and (b <= tt[-1] + tol)
        if require_isolated and not isolated:
            inside = False                      # invalid for every series

        def _coverages(mask: np.ndarray) -> "tuple[float, float]":
            return (min(1.0, float(mask[i0:i1].sum()) / n_nom_pre),
                    min(1.0, float(mask[i1:i2].sum()) / n_nom_post))

        cov_pre, cov_post = _coverages(joint)
        valid = bool(inside and cov_pre >= min_coverage and cov_post >= min_coverage)
        any_valid = False
        for key, y in ys.items():
            k_pre_cov, k_post_cov = _coverages(finite[key])
            k_valid = bool(inside and k_pre_cov >= min_coverage
                           and k_post_cov >= min_coverage)
            any_valid |= k_valid
            if mask_invalid and not k_valid:
                m_pre = m_post = nan
            else:
                m_pre, m_post = _nanmean(y[i0:i1]), _nanmean(y[i1:i2])
            row.update({f"{key}_pre": m_pre, f"{key}_post": m_post,
                        f"{key}_delta": m_post - m_pre,
                        f"{key}_coverage_pre": k_pre_cov,
                        f"{key}_coverage_post": k_post_cov,
                        f"{key}_valid": k_valid})
        row.update(valid=valid, coverage_pre=cov_pre, coverage_post=cov_post,
                   any_valid=bool(any_valid) if ys else valid)
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df["valid"] = df["valid"].astype(bool)
    df["any_valid"] = df["any_valid"].astype(bool)
    df["isolated"] = df["isolated"].astype(bool)
    for c in ("prev_gap_s", "next_gap_s"):
        df[c] = df[c].astype(float)
    for key in ys:
        df[f"{key}_valid"] = df[f"{key}_valid"].astype(bool)
    df.attrs.update(pre_s=pre_s, post_s=post_s, min_coverage=min_coverage,
                    require_isolated=bool(require_isolated))
    return df


# ════════════════════════════════════════════════════════════
#  EVENT-LOCKED TRACES (visualisation)
# ════════════════════════════════════════════════════════════

# A query within this fraction of a sampling step of a sample is ON it:
# ``onset + t_rel`` rarely reproduces a sample time to the last bit.
_SNAP_FRACTION = 1e-9


def _interp_no_bridge(t: np.ndarray, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Linear interpolation of (t, y) at *q* that never bridges a NaN: a query
    between two samples needs both finite, a query on a sample (to within
    ``_SNAP_FRACTION`` of the step, so round-off in the query time does not
    matter) takes that sample alone, and a query outside [t[0], t[-1]] is
    NaN."""
    n = len(t)
    i = np.clip(np.searchsorted(t, q, side="right") - 1, 0, n - 2)
    t0, t1 = t[i], t[i + 1]
    y0, y1 = y[i], y[i + 1]
    w = (q - t0) / (t1 - t0)
    w = np.where(np.abs(w) <= _SNAP_FRACTION, 0.0,
                 np.where(np.abs(w - 1.0) <= _SNAP_FRACTION, 1.0, w))
    with np.errstate(invalid="ignore"):
        out = y0 + w * (y1 - y0)
    out = np.where(w == 0.0, y0, np.where(w == 1.0, y1, out))
    # i is clipped to [0, n-2], so w < 0 / w > 1 (after snapping) means the
    # query lies before t[0] / after t[-1] by more than round-off.
    out[~np.isfinite(q) | ~(w >= 0.0) | ~(w <= 1.0)] = np.nan
    return out


def event_locked_traces(
    t:       ArrayLike,
    series:  Mapping[str, ArrayLike],
    onsets:  Union[float, ArrayLike],
    pre_s:   float = 30.0,
    post_s:  float = 30.0,
    fs_out:  float = 20.0,
    zscore:  bool = True,
) -> "tuple[np.ndarray, dict[str, np.ndarray]]":
    """Cut every series around each onset on a common relative time grid.

    ``t_rel = arange(-round(pre_s * fs_out), round(post_s * fs_out) + 1) /
    fs_out`` (s, onset at exactly 0). Each event's trace is the series
    linearly interpolated at ``onset + t_rel``; queries outside the recording
    or between samples one of which is NaN are NaN (gaps are never bridged),
    while a query that falls on a sample up to round-off takes that sample,
    whatever its neighbours.

    With *zscore* (the default) each trace is normalised by the mean and SD
    (ddof=1, finite samples) of that recording's WHOLE series -- the paper's
    per-subject z-scoring for visualisation, so subjects with different
    baselines can be overlaid. As in MATLAB's ``zscore``, a zero SD leaves
    the series centred only.

    Call once per recording and combine recordings with ``stack_traces``.
    Returns ``(t_rel, {key: array of shape (n_events, len(t_rel))})``.
    """
    tt = _time_axis(t)
    ys = _series_arrays(series, len(tt))
    ons = _onset_array(onsets)
    fs_out = _positive("fs_out", fs_out)
    if not (np.isfinite(pre_s) and pre_s >= 0 and np.isfinite(post_s) and post_s >= 0):
        raise ValueError("pre_s and post_s must be finite and non-negative.")
    n_pre = int(round(float(pre_s) * fs_out))
    n_post = int(round(float(post_s) * fs_out))
    t_rel = np.arange(-n_pre, n_post + 1, dtype=np.float64) / fs_out

    traces: dict[str, np.ndarray] = {}
    for key, y in ys.items():
        if zscore:
            fin = y[np.isfinite(y)]
            mu = float(fin.mean()) if len(fin) else 0.0
            sd = float(fin.std(ddof=1)) if len(fin) > 1 else float("nan")
            if not (np.isfinite(sd) and sd > 0):
                sd = 1.0
            y = (y - mu) / sd
        out = np.full((len(ons), len(t_rel)), np.nan)
        for j, onset in enumerate(ons):
            if np.isfinite(onset):
                out[j] = _interp_no_bridge(tt, y, onset + t_rel)
        traces[key] = out
    return t_rel, traces


def stack_traces(
    traces_list: Iterable[Mapping[str, np.ndarray]],
) -> "dict[str, np.ndarray]":
    """Concatenate the event rows of several ``event_locked_traces`` outputs
    (e.g. one per recording / subject) key by key. Every input must hold the
    same keys and the same number of relative-time samples."""
    items = [dict(tr) for tr in traces_list]
    if not items:
        return {}
    keys = list(items[0])
    for tr in items[1:]:
        if set(tr) != set(keys):
            raise ValueError("All trace dicts must hold the same series keys.")
    out: dict[str, np.ndarray] = {}
    for key in keys:
        arrs = [np.atleast_2d(np.asarray(tr[key], dtype=np.float64)) for tr in items]
        widths = {a.shape[1] for a in arrs}
        if len(widths) != 1:
            raise ValueError(
                f"Series {key!r}: traces have different lengths {sorted(widths)}.")
        out[key] = np.vstack(arrs)
    return out


def group_summary(
    traces: Mapping[str, np.ndarray],
) -> "dict[str, tuple[np.ndarray, np.ndarray]]":
    """Group median and MAD per relative time sample, ignoring NaN.

    *traces* maps each key to an (n_events, n_samples) array (a 1-D array is
    one event). MAD is the unscaled median absolute deviation
    ``median(|x - median(x)|)``, as MATLAB's ``mad(x, 1)``. A time sample
    with no finite value gives NaN for both. Returns ``{key: (median, mad)}``.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, arr in traces.items():
        a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        if a.shape[0] == 0:
            nan = np.full(a.shape[1], np.nan)
            out[key] = (nan, nan.copy())
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            med = np.nanmedian(a, axis=0)
            mad = np.nanmedian(np.abs(a - med), axis=0)
        out[key] = (med, mad)
    return out


# ════════════════════════════════════════════════════════════
#  PAIRED WILCOXON SIGNED-RANK TEST
# ════════════════════════════════════════════════════════════

def _merge_round_off_ties(a: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """*a* (>= 0) with every round-off tie group set to one value.

    As MATLAB ``tiedrank(a, 0, 0, eps)``: sorted ascending, ``a`` value j+1
    belongs to the group of value j when ``a_j + eps_j >= a_{j+1}`` (groups
    chain). Each member takes its group's smallest value, so exact-equality
    tie handling downstream (``rankdata``, ``np.unique``, scipy) sees the
    tolerance-based groups."""
    if len(a) < 2:
        return a.copy()
    order = np.argsort(a, kind="stable")
    sa, se = a[order], eps[order]
    tied = sa[:-1] + se[:-1] >= sa[1:]
    group = np.concatenate(([0], np.cumsum(~tied)))
    first = sa[np.concatenate(([0], np.flatnonzero(~tied) + 1))]
    out = np.empty_like(a)
    out[order] = first[group]
    return out


def wilcoxon_paired(pre: ArrayLike, post: ArrayLike) -> dict:
    """Two-sided Wilcoxon signed-rank test of paired *pre* / *post* values.

    Pairs with a NaN on either side are dropped (``n`` = pairs kept). With
    ``d_i = post_i - pre_i`` and ``e_i = eps(pre_i) + eps(post_i)`` (MATLAB
    ``signrank``'s round-off tolerance), zero differences ``|d_i| <= e_i``
    are dropped as in Wilcoxon's original convention (scipy
    ``zero_method="wilcox"``), so that a float round-off is not taken for a
    change (``n_nonzero`` = differences left). Ties among the remaining
    ``|d_i|`` are found with the same tolerance, as MATLAB's
    ``tiedrank(abs(d), 0, 0, e)``: in increasing order, a value within
    ``e`` of its predecessor joins its tie group (so the differences
    0.4 - 0.1 and 0.3 - 0, one ulp apart, tie). On those differences:

        R_i   = average rank of |d_i| (tie groups share their mean rank)
        W+    = sum of R_i over d_i > 0                     (``statistic``)
        E     = n'(n' + 1) / 4
        Var   = n'(n' + 1)(2n' + 1) / 24 - sum_j (t_j^3 - t_j) / 48
        z     = (W+ - E) / sqrt(Var)

    with n' = ``n_nonzero`` and t_j the size of each tie group. No
    continuity correction is applied. This is the normal approximation of
    the signed-rank statistic used by MATLAB's ``signrank`` (method
    "approximate"; the reference paper's Z(k) values come from it), with the
    same zero and tie tolerance, and ``|z|`` equals the magnitude of scipy's
    ``wilcoxon(post - pre, method="approx").zstatistic`` whenever no two
    ``|d|`` differ by round-off only. Compare MAGNITUDES across tools: scipy
    computes ``(W+ - E) / sqrt(Var)`` internally (this sign) but reports
    ``-|z|`` for a two-sided test, and the sign MATLAB attaches to
    ``st.zval`` depends on the rank sum (positive or negative) its release
    uses -- check one known case before comparing signs with MATLAB output.

    Sign convention: z > 0 when post tends to exceed pre IN RANK SUM
    (W+ > E). Its sign matches that of ``median_delta`` except in skewed
    samples where the rank sum and the median disagree, e.g. d = (-1.0 ..
    -1.6 seven times, +5 .. +8 four times): median_delta = -1.1 but
    z = +0.44. z deliberately follows the rank sum, the quantity tested
    and the source of its magnitude: a median-signed z would contradict W+
    and p there (with many events a median just below zero can coexist
    with z up to about +0.43 sqrt(n') -- e.g. +4.2, p < 1e-4, for n' = 101
    -- when the slightly fewer positive changes are all the largest),
    and would jump discontinuously in a ``calibrate_gain`` sweep when the
    median crosses zero while |z| does not (``knee_point`` keys on the sign
    change).

    ``p_value`` comes from ``scipy.stats.wilcoxon`` (two-sided, default
    method: exact for n' <= 50 without ties, exact permutation for
    n' <= 13 with ties, normal approximation otherwise), run on the
    differences with each tie group's ``|d|`` merged to one value so scipy
    sees the same ties; ``p_method`` names the method used ("exact",
    "exact permutation", "normal approximation"). Because it is exact for
    small samples it does NOT in general equal the normal p of ``z``:
    ``p_normal = 2 * (1 - Phi(|z|))`` is that one (what MATLAB's
    ``signrank`` reports next to ``zval`` with method "approximate", no
    continuity correction), and the two can fall on opposite sides of 0.05
    near |z| = 1.96. Report ``p_value`` as the test's p, and ``p_normal``
    when z is the quantity shown. ``z_max`` is the largest |z| this sample
    can reach -- every difference with the same sign, ``E / sqrt(Var)``
    (sqrt(3 n'(n' + 1) / (2 (2n' + 1))) without ties: 2.20 for n' = 6,
    2.52 for 8, 3.06 for 12) -- a ceiling a small cohort's z sits on as
    soon as every unit changes the same way. When every difference is zero
    there is no evidence of a change: statistic = 0, z = 0, p = 1. When no
    pair is left, everything is NaN.

    Medians and unscaled MADs (``median(|x - median(x)|)``) are given for
    the kept pairs; ``median_delta`` is the median of ``post - pre``.
    """
    x = np.asarray(pre, dtype=np.float64).flatten()
    y = np.asarray(post, dtype=np.float64).flatten()
    if len(x) != len(y):
        raise ValueError(f"pre and post must have the same length ({len(x)} != {len(y)}).")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(len(x))
    med_pre, mad_pre = _median_mad(x)
    med_post, mad_post = _median_mad(y)
    d = y - x
    nan = float("nan")
    result = {
        "n":            n,
        "n_nonzero":    0,
        "statistic":    nan,
        "p_value":      nan,
        "z":            nan,
        "median_pre":   med_pre,
        "median_post":  med_post,
        "mad_pre":      mad_pre,
        "mad_post":     mad_post,
        "median_delta": float(np.median(d)) if n else nan,
        "p_normal":     nan,
        "p_method":     "",
        "z_max":        nan,
    }
    if n == 0:
        return result

    eps_d = np.spacing(np.abs(x)) + np.spacing(np.abs(y))
    nonzero = np.abs(d) > eps_d
    dn = d[nonzero]
    m = int(len(dn))
    result["n_nonzero"] = m
    if m == 0:
        result.update(statistic=0.0, z=0.0, p_value=1.0, p_normal=1.0,
                      p_method="no change", z_max=0.0)
        return result

    abs_d = _merge_round_off_ties(np.abs(dn), eps_d[nonzero])
    dn = np.sign(dn) * abs_d
    ranks = stats.rankdata(abs_d, method="average")
    w_plus = float(ranks[dn > 0].sum())
    _, tie_sizes = np.unique(abs_d, return_counts=True)
    tie_term = float(np.sum(tie_sizes.astype(np.float64) ** 3 - tie_sizes)) / 48.0
    mean_w = m * (m + 1) / 4.0
    var_w = m * (m + 1) * (2 * m + 1) / 24.0 - tie_term
    z = (w_plus - mean_w) / np.sqrt(var_w) if var_w > 0 else 0.0
    z_max = mean_w / np.sqrt(var_w) if var_w > 0 else 0.0
    p = float(stats.wilcoxon(dn, alternative="two-sided").pvalue)
    # scipy's "auto" rule (no zeros are left here): exact without ties up
    # to 50 differences, an exact permutation with ties up to 13, else the
    # normal approximation.
    ties = bool(np.any(tie_sizes > 1))
    if m > 50:
        method = "normal approximation"
    elif not ties:
        method = "exact"
    elif m <= 13:
        method = "exact permutation"
    else:
        method = "normal approximation"
    p_normal = float(2.0 * stats.norm.sf(abs(float(z))))

    result.update(statistic=w_plus, z=float(z), p_value=min(max(p, 0.0), 1.0),
                  p_normal=min(max(p_normal, 0.0), 1.0), p_method=method,
                  z_max=float(z_max))
    return result


def paired_tests(
    df:   pd.DataFrame,
    keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Run ``wilcoxon_paired`` on ``k_pre`` vs ``k_post`` for every key of an
    ``event_window_means`` (or ``aggregate``) table, with a Bonferroni
    correction across the keys tested.

    *keys* defaults to every ``k`` having ``k_pre``, ``k_post`` and
    ``k_delta`` columns (so ``coverage_pre`` / ``k_coverage_pre`` are not
    mistaken for a series). Each key is tested on the rows where THAT
    series is valid (its ``k_valid`` column; tables without one fall back
    to the event-level ``valid`` column -- joint validity, which implies
    every series' -- then to every row), so a series
    missing around some events does not remove them from the other keys'
    tests and ``n`` can differ between keys. Returns one row per key:
    ``key`` plus the ``wilcoxon_paired`` fields (``p_value`` the test's p,
    ``p_method`` its method, ``p_normal`` the normal p of ``z``, ``z_max``
    the sample's |z| ceiling) and ``p_bonferroni`` (of ``p_value``).
    """
    if keys is None:
        keys = _series_keys(df)
    event_ok = _flag(df["valid"]) if "valid" in df.columns else np.ones(len(df), dtype=bool)
    rows = []
    for key in keys:
        vcol = f"{key}_valid"
        use = _flag(df[vcol]) if vcol in df.columns else event_ok
        res = wilcoxon_paired(df.loc[use, f"{key}_pre"].to_numpy(dtype=float),
                              df.loc[use, f"{key}_post"].to_numpy(dtype=float))
        rows.append({"key": key, **res})
    cols = ["key", "n", "n_nonzero", "statistic", "z", "p_value", "median_pre",
            "mad_pre", "median_post", "mad_post", "median_delta", "p_normal",
            "p_method", "z_max"]
    out = pd.DataFrame(rows, columns=cols)
    out["p_bonferroni"] = bonferroni(out["p_value"].to_numpy(dtype=float))
    return out


# ════════════════════════════════════════════════════════════
#  GAIN CALIBRATION (ks, kp)
# ════════════════════════════════════════════════════════════

_KIND_SIGN = {"csi": -1.0, "cpi": 1.0, "cvi": 1.0}


def calibrate_gain(
    delta_d:  ArrayLike,
    delta_sd: ArrayLike,
    kind:     str,
    k_values: Iterable[float] = range(1, 21),
) -> pd.DataFrame:
    """Wilcoxon Z of the condition contrast of CSI or CPI as a function of k.

    *delta_d* and *delta_sd* are per-event (or per-animal) post - pre
    changes of the ellipse-centre distance D and of SD2 (``kind="csi"``) or
    SD1 (``kind="cpi"``; ``"cvi"`` is accepted as the reference code's name
    for it), in the same unit. By linearity of the indices in k (module
    docstring) the index change for gain k is

        csi:  delta_index = -delta_d + k * delta_sd      (delta_sd = dSD2)
        cpi:  delta_index =  delta_d + k * delta_sd      (delta_sd = dSD1)

    and z / p are those of the Wilcoxon signed-rank test of delta_index
    against 0 -- the same as the test of the index's pre vs post window
    means. The test is run as ``wilcoxon_paired(-sign * delta_d,
    k * delta_sd)``, whose differences ``post - pre`` are delta_index, so
    that its round-off tolerance (the eps of both operands) scales with the
    two TERMS of the index rather than with their sum: an event whose terms
    cancel to round-off (e.g. 0.3 + 3 * -0.1 = -5.6e-17) is a zero change
    and dropped, as the explicit test on the index's window means drops it,
    instead of entering the test as a real change of rank 1. Events with a
    NaN delta are dropped. The paper sweeps k = 1..20 and takes the point
    from which |Z| plateaus (``knee_point``); its rat values (ks = 3,
    kp = 12) are not validated in mice.

    Z(k) is a RANK statistic: it is piecewise constant in k, moves only
    when some unit's index change flips sign or two units swap rank, and
    cannot exceed ``z_max`` (every unit changing the same way: 2.52 for 8
    units, 3.06 for 12). On a small cohort it often sits on that ceiling
    over much of the sweep; ``knee_point(..., statistic=...)`` uses the
    rank sum to refuse a "knee" that is only one or two rank steps.

    Returns a DataFrame with columns ``k, z, p_value, n, n_nonzero,
    median_delta, statistic, p_normal, z_max`` -- ``n`` events kept,
    ``n_nonzero`` of them with a non-zero index change (the test's sample
    size), ``median_delta`` the median index change, ``statistic`` the rank
    sum W+, ``p_normal`` the normal p of z and ``z_max`` the |z| ceiling
    (``wilcoxon_paired``; ``attrs["kind"]`` holds the normalised kind).
    """
    kind_n = str(kind).strip().lower()
    if kind_n not in _KIND_SIGN:
        raise ValueError(f"kind must be 'csi' or 'cpi', got {kind!r}.")
    dd = np.asarray(delta_d, dtype=np.float64).flatten()
    ds = np.asarray(delta_sd, dtype=np.float64).flatten()
    if len(dd) != len(ds):
        raise ValueError(
            f"delta_d and delta_sd must have the same length ({len(dd)} != {len(ds)}).")
    keep = np.isfinite(dd) & np.isfinite(ds)
    dd, ds = dd[keep], ds[keep]
    sign = _KIND_SIGN[kind_n]
    rows = []
    for k in k_values:
        # post - pre = sign * dd + k * ds = delta_index (see docstring).
        res = wilcoxon_paired(-sign * dd, float(k) * ds)
        rows.append({"k": k, "z": res["z"], "p_value": res["p_value"],
                     "n": res["n"], "n_nonzero": res["n_nonzero"],
                     "median_delta": res["median_delta"],
                     "statistic": res["statistic"], "p_normal": res["p_normal"],
                     "z_max": res["z_max"]})
    out = pd.DataFrame(rows, columns=["k", "z", "p_value", "n", "n_nonzero", "median_delta",
                                      "statistic", "p_normal", "z_max"])
    out.attrs["kind"] = "cpi" if kind_n == "cvi" else kind_n
    return out


def sign_stretches(k: ArrayLike, z: ArrayLike) -> "list[dict]":
    """The runs of constant sign of a Z(k) curve, in increasing k.

    Each run is ``{"sign": +1 / -1, "k_first", "k_last", "z_peak",
    "k_peak"}`` -- its first and last k and the point of largest |z| on it
    (the first such k). Non-finite points and points with z exactly 0 are
    skipped (they neither end nor start a run). More than one run means the
    index reverses direction as k grows: its heart-rate and HRV terms move
    the opposite way, so which direction the index shows depends on the
    gain -- a gain on the last run makes the index move OPPOSITE to its
    small-k direction."""
    kk = np.asarray(k, dtype=np.float64).flatten()
    zz = np.asarray(z, dtype=np.float64).flatten()
    if len(kk) != len(zz):
        raise ValueError(f"k and z must have the same length ({len(kk)} != {len(zz)}).")
    keep = np.isfinite(kk) & np.isfinite(zz) & (zz != 0)
    kk, zz = kk[keep], zz[keep]
    order = np.argsort(kk, kind="stable")
    kk, zz = kk[order], zz[order]
    runs: "list[dict]" = []
    i = 0
    while i < len(zz):
        s = np.sign(zz[i])
        j = i
        while j + 1 < len(zz) and np.sign(zz[j + 1]) == s:
            j += 1
        seg = np.abs(zz[i:j + 1])
        p = i + int(np.argmax(seg))
        runs.append({"sign": int(s), "k_first": float(kk[i]), "k_last": float(kk[j]),
                     "z_peak": float(zz[p]), "k_peak": float(kk[p])})
        i = j + 1
    return runs


def knee_point(
    k:   ArrayLike,
    z:   ArrayLike,
    *,
    after_sign_change: bool = True,
    tol: float = 1e-3,
    statistic: Optional[ArrayLike] = None,
    min_rank_steps: float = 0.0,
    min_plateau_points: int = 3,
) -> float:
    """Gain k at the knee of the |Z(k)| curve (Kneedle method).

    The paper picks by eye the "inflection point" of Z(k) from which it
    plateaus. As an objective, reproducible proxy this applies the Kneedle
    method (Satopaa et al., 2011) to |z|: both axes are normalised to
    [0, 1] by their min and max, and the knee is the point farthest from
    the chord joining the first and last points ON THE PLATEAU SIDE of it --
    above the chord when the curve ends higher than it starts (Kneedle's
    "concave increasing" case: |Z| rises, then saturates), below it when the
    curve ends lower (Kneedle's "convex decreasing" case: |Z| falls onto a
    plateau). The distance is signed: a point on the other side of the
    chord is where a curve starts to bend AWAY from a plateau -- the foot of
    an S-shaped rise, or an accelerating curve -- and is never chosen. So on
    an S-shaped |Z(k)| the knee is the shoulder where the rise flattens,
    wherever the rise sits in the sweep (for mice, with drug effects of
    several ms on D against ~1 ms on SD2, the heart-rate-to-HRV transition
    can lie around k = 10-15), and a curve with no plateau has no knee.
    It depends on the k range swept.

    With *after_sign_change* (the default), when z changes sign over the
    range -- the heart-rate and HRV terms move in opposite directions, so
    the index reverses direction as k grows and |z| dips through zero, which
    would make the dip the "knee" whenever |z| ends lower than it starts --
    the curve analysed starts at
    the last point before the final same-sign stretch (the crossing) and is
    oriented by that final sign (``z * sign(z_end)``, equal to |z| on the
    stretch), so the knee is where the contrast in its final direction
    plateaus. Without a sign change this is exactly Kneedle on |z|. Such a
    knee sits where the index moves OPPOSITE to its small-k direction
    (``sign_stretches``); a caller offering it must say so.

    Z(k) is a rank statistic (``calibrate_gain``): on a small cohort it
    changes in a handful of discrete rank steps, and Kneedle then puts the
    "knee" on whichever step there is. With *statistic* (the rank sum W+
    at each k, ``calibrate_gain``'s column of that name) the knee is
    refused (NaN) when W+ varies by fewer than *min_rank_steps* over the
    stretch analysed -- the curve's only movement is one or two units
    flipping sign or swapping rank (a W+ change of 1 per swap), an artefact
    of the statistic's discreteness rather than a plateau.

    A knee also needs a plateau to sit on: it is refused when fewer than
    *min_plateau_points* points (the knee included) lie from it to the end
    of the sweep. A curve that keeps accelerating but whose last two points
    happen to share one rank step otherwise gets its "knee" at the
    second-to-last k -- an end-of-sweep artefact, not a saturation.

    Non-finite points are ignored and k is sorted (duplicate k values raise
    ValueError). Returns NaN when fewer than 3 points remain, when the curve
    is flat, when it moves by fewer than *min_rank_steps* rank steps, when
    no point lies farther than *tol* (normalised perpendicular distance)
    from the chord on its plateau side -- a straight line, or a curve that
    only accelerates (no plateau within the sweep) -- or when the knee has
    fewer than *min_plateau_points* points after it, itself included.
    """
    kk = np.asarray(k, dtype=np.float64).flatten()
    zz = np.asarray(z, dtype=np.float64).flatten()
    if len(kk) != len(zz):
        raise ValueError(f"k and z must have the same length ({len(kk)} != {len(zz)}).")
    if statistic is None:
        ww = np.zeros(len(kk))
    else:
        ww = np.asarray(statistic, dtype=np.float64).flatten()
        if len(ww) != len(kk):
            raise ValueError(
                f"statistic must have the same length as k ({len(ww)} != {len(kk)}).")
    keep = np.isfinite(kk) & np.isfinite(zz)
    if statistic is not None:
        keep &= np.isfinite(ww)
    kk, zz, ww = kk[keep], zz[keep], ww[keep]
    order = np.argsort(kk, kind="stable")
    kk, zz, ww = kk[order], zz[order], ww[order]
    if len(kk) and np.any(np.diff(kk) == 0):
        raise ValueError("k values must be unique.")

    if after_sign_change and len(zz):
        signs = np.sign(zz)
        nonzero = np.flatnonzero(signs != 0)
        if not len(nonzero):
            return float("nan")
        end = int(nonzero[-1])
        last_sign = signs[end]
        other = np.flatnonzero(signs[: end + 1] != last_sign)
        start = int(other[-1]) if len(other) else 0
        kk, y = kk[start: end + 1], last_sign * zz[start: end + 1]
        ww = ww[start: end + 1]
    else:
        y = np.abs(zz)

    if len(kk) < 3:
        return float("nan")
    if statistic is not None and float(ww.max() - ww.min()) < float(min_rank_steps):
        return float("nan")
    y_span = float(y.max() - y.min())
    x_span = float(kk[-1] - kk[0])
    if not (y_span > 0 and x_span > 0):
        return float("nan")
    xn = (kk - kk[0]) / x_span                     # 0 .. 1
    yn = (y - y.min()) / y_span
    rise = float(yn[-1] - yn[0])
    # Signed perpendicular distance to the chord (0, yn[0]) -> (1, yn[-1]),
    # positive above it; flipped for a falling curve so that positive always
    # points towards the plateau corner.
    above = (yn - (yn[0] + rise * xn)) / float(np.hypot(1.0, rise))
    dist = above if rise >= 0 else -above
    i = int(np.argmax(dist))
    if not dist[i] > tol:
        return float("nan")
    if len(kk) - i < int(min_plateau_points):
        return float("nan")
    return float(kk[i])


# ════════════════════════════════════════════════════════════
#  PER-SUBJECT AGGREGATION, MULTIPLE COMPARISONS
# ════════════════════════════════════════════════════════════

def aggregate(
    df: pd.DataFrame,
    by: Union[str, Sequence[str]],
    *,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Per-subject mean of the numeric columns of an event table.

    When an animal contributes several events (doses, repeated
    injections), testing the events as independent samples is
    pseudo-replication; averaging them per animal first gives one paired
    sample per animal. *by* names the grouping column(s) (e.g.
    ``"subject"``, or ``["subject", "condition"]``).

    With *valid_only* (the default) the events where no series is usable
    are left out before averaging -- rows whose ``any_valid`` column is
    False, or whose ``valid`` column is False in a table without
    ``any_valid`` -- and so is each series' value of an event where that
    series is invalid (``k_valid`` False: ``k_pre``, ``k_post`` and
    ``k_delta`` are treated as missing -- they already are NaN unless the
    table was built with ``mask_invalid=False``). An event of an
    ``event_window_means`` table that is ``valid=False`` only because ANOTHER
    series is gappy thus still counts for the series that are valid; filter
    on ``valid`` first for a complete-case analysis. Bool columns are not
    averaged; NaN cells are skipped, so a series' mean can rest on fewer
    events than ``n_events``.

    The result can be aggregated again, e.g. per subject x dose first and
    then per subject (each dose weighted equally): an existing ``n_events``
    column is summed rather than averaged, so it keeps counting the
    original events.

    Returns the group columns, ``n_events`` (events averaged) and the mean
    of every other numeric column, sorted by group.
    """
    by_cols = [by] if isinstance(by, str) else list(by)
    missing = [c for c in by_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Grouping column(s) not in the table: {missing}.")
    data = df
    if valid_only:
        flag_col = next((c for c in ("any_valid", "valid") if c in df.columns), None)
        if flag_col is not None:
            data = df[_flag(df[flag_col])]
        key_masks = {key: ~_flag(data[f"{key}_valid"]) for key in _series_keys(data)
                     if f"{key}_valid" in data.columns}
        if any(mask.any() for mask in key_masks.values()):
            data = data.copy()
            for key, bad in key_masks.items():
                for suffix in _VALUE_SUFFIXES:
                    col = f"{key}{suffix}"
                    if pd.api.types.is_numeric_dtype(data[col]):
                        data.loc[bad, col] = np.nan
    value_cols = [c for c in data.select_dtypes(include="number").columns
                  if c not in by_cols and c != "n_events"]
    grouped = data.groupby(by_cols, sort=True, dropna=False)
    out = grouped[value_cols].mean() if value_cols else pd.DataFrame(index=grouped.size().index)
    if "n_events" in data.columns and pd.api.types.is_numeric_dtype(data["n_events"]):
        counts = grouped["n_events"].sum()
    else:
        counts = grouped.size()
    out.insert(0, "n_events", counts.astype(int))
    return out.reset_index()


def bonferroni(pvals: ArrayLike) -> Union[np.ndarray, pd.Series]:
    """Bonferroni-adjusted p-values: ``min(1, p * m)``.

    m is the number of FINITE p-values (a NaN marks a test that could not
    be run and is neither counted nor adjusted). A pandas Series comes back
    as a Series with the same index, anything else as a float ndarray of
    the input's shape.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = int(np.isfinite(p).sum())
    adj = np.minimum(p * m, 1.0) if m else p.copy()
    if isinstance(pvals, pd.Series):
        return pd.Series(adj, index=pvals.index, name=pvals.name)
    return adj
