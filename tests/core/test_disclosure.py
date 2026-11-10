"""
Presentation helpers behind the tachogram and the exports: every RR interval
stays visible (excluded ones flagged, not dropped), the exclusion counts and
the RR bounds used are reported the same way everywhere, and display
decimation keeps the line broken at every excluded interval.
"""
import numpy as np
import pytest

from ecg.core import analysis
from ecg.core.filtering import break_line_at_gaps, downsample_xy_for_display
from ecg.core.models import MouseECG

from tests.helpers import rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0


def _pause_result(pause_ms: float = 400.0, **kw) -> "tuple[dict, np.ndarray]":
    rng = np.random.default_rng(0)
    rr = list(120 + rng.normal(0, 2, 200)) + [pause_ms] + list(120 + rng.normal(0, 2, 200))
    peaks = rpeaks_from_rr(rr, FS)
    sig = np.zeros(int(peaks[-1]) + int(FS))
    return analysis.analyse_core(sig, peaks, FS, **kw), peaks


# ── rr_timeseries: every interval, excluded ones flagged ─────────────────

@requires_nk
def test_rr_timeseries_keeps_the_excluded_pause_at_its_true_time():
    result, peaks = _pause_result()
    ts = analysis.rr_timeseries(result)

    assert list(ts.columns) == ["Time_s", "RR_ms", "HR_bpm", "valid"]
    assert len(ts) == len(peaks) - 1                  # rr_df has one row fewer
    assert len(result["rr_df"]) == len(peaks) - 2
    np.testing.assert_allclose(ts["Time_s"], peaks[1:] / FS)
    excluded = ts[~ts["valid"]]
    assert excluded["RR_ms"].tolist() == [pytest.approx(400.0)]
    assert excluded["HR_bpm"].tolist() == [pytest.approx(150.0)]
    assert ts["valid"].dtype == bool


@requires_nk
def test_rr_timeseries_rebuilds_times_and_mask_for_older_sessions():
    # Sessions saved before rr_t_s / rr_valid existed: times come from
    # rr_df's valid rows plus the cumulative RR, the mask from the bounds.
    result, peaks = _pause_result()
    legacy = {k: v for k, v in result.items() if k not in ("rr_t_s", "rr_valid")}
    ts = analysis.rr_timeseries(legacy)

    np.testing.assert_allclose(ts["Time_s"], peaks[1:] / FS, atol=1e-9)
    assert ts["valid"].tolist() == result["rr_valid"].tolist()


def test_rr_timeseries_of_results_without_rr_is_empty():
    ts = analysis.rr_timeseries({"hr": {}})
    assert ts.empty and list(ts.columns) == ["Time_s", "RR_ms", "HR_bpm", "valid"]


# ── rr_exclusion_report / rr_disclosure_rows ──────────────────────────────

@requires_nk
def test_exclusion_report_reads_the_counts_and_bounds_used():
    result, _ = _pause_result(rr_max_ms=350.0)
    rep = analysis.rr_exclusion_report(result)

    assert rep["n_intervals"] == 401
    assert rep["n_valid"] == 400
    assert rep["n_excluded"] == 1 and rep["n_too_long"] == 1
    assert rep["n_too_short"] == 0 and rep["n_spanning_removed"] == 0
    assert rep["excluded_pct"] == pytest.approx(100.0 / 401)
    assert rep["excluded_time_s"] == pytest.approx(0.4)
    assert (rep["rr_min_ms"], rep["rr_max_ms"]) == (MouseECG.RR_MIN_MS, 350.0)


@requires_nk
def test_exclusion_report_recomputes_counts_for_older_results():
    # A removed beat spans the 4th interval; hr without the counts (older
    # session) -> recomputed from rr_ms + rr_valid: an excluded interval
    # inside the bounds can only be one spanning a removed beat.
    rng = np.random.default_rng(2)
    rr = list(120 + rng.normal(0, 2, 100)) + [20.0] + list(120 + rng.normal(0, 2, 100))
    peaks = rpeaks_from_rr(rr, FS)
    sig = np.zeros(int(peaks[-1]) + int(FS))
    result = analysis.analyse_core(sig, peaks, FS, breaks=[int(peaks[3]) + 50])
    new = analysis.rr_exclusion_report(result)
    assert (new["n_too_short"], new["n_spanning_removed"], new["n_excluded"]) == (1, 1, 2)

    legacy = dict(result)
    legacy["hr"] = {k: result["hr"][k] for k in ("mean", "min", "max", "std", "n")}
    old = analysis.rr_exclusion_report(legacy)
    for key in ("n_intervals", "n_valid", "n_excluded", "n_too_short",
                "n_too_long", "n_spanning_removed"):
        assert old[key] == new[key], key
    assert old["excluded_time_s"] == pytest.approx(new["excluded_time_s"])


def test_exclusion_report_without_any_rr_information():
    rep = analysis.rr_exclusion_report({"hr": {"n": 3, "mean": 500.0}})
    assert rep["n_excluded"] is None and rep["n_intervals"] is None
    rows = analysis.rr_disclosure_rows(rep)
    # the bounds are always disclosed, counts only when known
    assert [label for label, _ in rows] == ["RR bounds (HRV)"]
    assert rows[0][1].startswith("67–333 ms")


@requires_nk
def test_disclosure_rows_name_counts_reasons_and_bounds():
    result, _ = _pause_result(rr_max_ms=350.0)
    rows = dict(analysis.rr_disclosure_rows(analysis.rr_exclusion_report(result)))

    assert rows["RR intervals"] == "401  (400 used for HRV)"
    assert rows["RR excluded"].startswith("1  (0.2 %)")
    assert "1 too long" in rows["RR excluded"]
    assert "spanning removed beats" in rows["RR excluded"]
    assert rows["RR excluded time"] == "0.40 s"
    assert rows["RR bounds (HRV)"].startswith("67–350 ms")


@requires_nk
def test_disclosure_rows_without_exclusion():
    rng = np.random.default_rng(5)
    peaks = rpeaks_from_rr(list(120 + rng.normal(0, 2, 100)), FS)
    result = analysis.analyse_core(np.zeros(int(peaks[-1]) + int(FS)), peaks, FS)
    rows = dict(analysis.rr_disclosure_rows(analysis.rr_exclusion_report(result)))
    assert rows["RR excluded"] == "0  (0.0 %)"
    assert "RR excluded time" not in rows


# ── downsample_xy_for_display ─────────────────────────────────────────────

def test_short_series_is_returned_unchanged():
    x = np.arange(100.0)
    y = np.where(x == 50, np.nan, x)
    xo, yo = downsample_xy_for_display(x, y)
    np.testing.assert_array_equal(xo, x)
    np.testing.assert_array_equal(yo, y)


def _crosses_a_gap(x_in, y_in, xo, yo) -> bool:
    """True if two consecutive finite output points enclose an input gap."""
    gaps_x = x_in[np.isnan(y_in)]
    for i in range(len(xo) - 1):
        if np.isfinite(yo[i]) and np.isfinite(yo[i + 1]):
            if np.any((gaps_x > xo[i]) & (gaps_x < xo[i + 1])):
                return True
    return False


@pytest.mark.parametrize("gaps", [[5000], [12001, 12002], [0], [19999], [7, 8000, 15003]])
def test_decimated_line_stays_broken_at_every_gap(gaps):
    x = np.arange(20_000.0)
    y = np.sin(x / 300.0) + 0.001 * x
    y[gaps] = np.nan
    xo, yo = downsample_xy_for_display(x, y, max_points=6_000)

    assert len(xo) < len(x) // 2
    assert np.all(np.diff(xo) >= 0)                    # still in x order
    assert not _crosses_a_gap(x, y, xo, yo)
    assert np.isnan(yo).sum() >= 1


def test_decimation_keeps_extremes_at_their_own_x():
    x = np.arange(30_000.0)
    y = np.zeros_like(x)
    y[12_345] = 5.0                                   # narrow spike
    y[20_000] = np.nan
    xo, yo = downsample_xy_for_display(x, y, max_points=6_000)
    i = int(np.nanargmax(yo))
    assert yo[i] == 5.0 and xo[i] == 12_345.0


# ── break_line_at_gaps (rolling / epoch timelines) ───────────────────────

def test_line_breaks_where_skipped_windows_left_no_row():
    t = np.array([5.0, 6.0, 7.0, 10.0, 11.0])          # windows 8 and 9 skipped
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    xo, yo = break_line_at_gaps(t, y, max_step=1.5)
    assert xo.tolist() == [5.0, 6.0, 7.0, 8.5, 10.0, 11.0]
    assert np.isnan(yo[3]) and np.isfinite(np.delete(yo, 3)).all()


def test_regular_timeline_is_unchanged():
    t = np.arange(0.0, 10.0, 2.0)
    y = np.sin(t)
    xo, yo = break_line_at_gaps(t, y, max_step=3.0)
    np.testing.assert_array_equal(xo, t)
    np.testing.assert_array_equal(yo, y)
