"""
Beat-quality timestamps: compute_beat_correlation leaves out peaks too
close to the signal edges, so analyse_core records the time of every
scored beat (beat_corr_t_s) and the quality-over-time curve, the
"Poor-shape-match time" tile and the RR CSV quality flag use those times
(the tile and the CSV both on the beat STARTING each interval) instead of
pairing beat_corr with the peaks / intervals by position. Older results
get the times rebuilt from their peaks, or no pairing when ambiguous.
"""
import numpy as np
import pytest

from ecg.core import analysis
from ecg.core.analysis import (
    beat_quality_at, beat_quality_times, noisy_time_pct, rebuild_beat_corr_times,
    rr_quality_at,
)

from tests.helpers import make_synth_ecg

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def _edge_cropped():
    sig, fs, rp = make_synth_ecg(n_beats=40)
    crop = rp[0] - 10                    # first peak 5 ms from the signal start
    return sig[crop:], fs, rp - crop


@requires_nk
def test_analyse_core_records_the_times_of_the_scored_beats():
    sig, fs, rp = _edge_cropped()
    r = analysis.analyse_core(sig, rp, fs)
    t = r["beat_corr_t_s"]
    assert len(t) == len(r["beat_corr"]) == len(rp) - 1       # first peak not scored
    np.testing.assert_allclose(t, rp[1:] / fs)
    times, corr = beat_quality_times(r, rp / fs)
    np.testing.assert_allclose(times, rp[1:] / fs)             # not rp[:-1]
    np.testing.assert_array_equal(corr, r["beat_corr"])


def test_older_results_pair_with_the_peaks_only_when_exact():
    r = {"beat_corr": np.array([0.9, 0.95, 0.99])}
    # One score per peak: no edge beat was dropped, the pairing is exact.
    times, corr = beat_quality_times(r, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(times, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(corr, r["beat_corr"])
    # One peak more than scores: which edge beat was dropped is unknown,
    # so nothing is paired (it used to shift every score by one beat).
    assert beat_quality_times(r, np.array([1.0, 2.0, 3.0, 4.0]))[0].size == 0
    assert beat_quality_times(r)[0].size == 0
    assert beat_quality_times({})[0].size == 0


def _results_with_bad_first_scored_beat():
    # 5 peaks at 0.1 s spacing, first one (edge) unscored; scored beat at
    # 0.2 s is poor. Intervals: [0.1-0.2], [0.2-0.3], [0.3-0.4], [0.4-0.5].
    peaks_s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rr = np.array([100.0, 100.0, 100.0, 100.0])
    return {"rr_ms": rr, "rr_t_s": peaks_s[1:],
            "beat_corr": np.array([0.5, 0.99, 0.99, 0.99]),
            "beat_corr_t_s": peaks_s[1:]}


def test_noisy_time_attributes_each_interval_to_its_leading_scored_beat():
    r = _results_with_bad_first_scored_beat()
    # Interval [0.2-0.3] starts at the poor beat; [0.1-0.2] starts at the
    # unscored edge beat and [0.5-] does not exist -> 1 of 3 intervals.
    assert noisy_time_pct(r) == pytest.approx(100.0 / 3)
    # Without the scored-beat times the edge beat cannot be placed: NaN
    # rather than the old positional pairing (poor value on [0.1-0.2]).
    legacy = dict(r); legacy.pop("beat_corr_t_s")
    assert np.isnan(noisy_time_pct(legacy))
    assert np.isnan(rr_quality_at(legacy, r["rr_t_s"], r["rr_ms"])).all()


def test_older_results_with_one_score_per_peak_are_paired_exactly():
    # 5 peaks, 5 scores (no edge beat dropped): interval i starts at beat i.
    peaks_s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    r = {"rr_ms": np.full(4, 100.0), "rr_t_s": peaks_s[1:],
         "beat_corr": np.array([0.99, 0.5, 0.99, 0.99, 0.99])}
    assert noisy_time_pct(r) == pytest.approx(25.0)          # interval [0.2-0.3]
    q = rr_quality_at(r, r["rr_t_s"], r["rr_ms"])
    np.testing.assert_array_equal(q, [0.99, 0.5, 0.99, 0.99])


def test_tile_and_rr_quality_use_the_same_leading_beat():
    r = _results_with_bad_first_scored_beat()
    q = rr_quality_at(r, r["rr_t_s"], r["rr_ms"])
    # [0.1-0.2] starts at the unscored edge beat; [0.2-0.3] at the poor one.
    assert np.isnan(q[0]) and q[1] == 0.5 and (q[2:] == 0.99).all()
    hit = np.isfinite(q)
    rr = r["rr_ms"]
    assert noisy_time_pct(r) == pytest.approx(
        100.0 * rr[hit][q[hit] < 0.90].sum() / rr[hit].sum())


@requires_nk
def test_rebuild_recovers_the_scored_beats_of_older_results():
    sig, fs, rp = _edge_cropped()
    r = analysis.analyse_core(sig, rp, fs)
    legacy = dict(r); legacy.pop("beat_corr_t_s")
    assert np.isnan(noisy_time_pct(legacy))
    assert rebuild_beat_corr_times(legacy, len(sig), rp, fs)
    np.testing.assert_allclose(legacy["beat_corr_t_s"], r["beat_corr_t_s"])
    assert noisy_time_pct(legacy) == pytest.approx(noisy_time_pct(r))
    # Current results already have them: no-op.
    assert not rebuild_beat_corr_times(r, len(sig), rp, fs)


def test_rebuild_refuses_an_ambiguous_count():
    fs = 1000.0
    rp = np.array([500, 620, 740, 860, 980])          # all far from the edges
    r = {"beat_corr": np.array([0.9, 0.9, 0.9])}       # 3 scores, 5 peaks
    assert not rebuild_beat_corr_times(r, 2000, rp, fs)
    assert r.get("beat_corr_t_s") is None
    r = {"beat_corr": np.full(5, 0.9)}
    assert rebuild_beat_corr_times(r, 2000, rp, fs)
    np.testing.assert_allclose(r["beat_corr_t_s"], rp / fs)


def test_quality_at_interval_end_beats():
    r = _results_with_bad_first_scored_beat()
    q = beat_quality_at(r, np.array([0.1, 0.2, 0.5, 0.7]))
    assert np.isnan(q[0]) and q[1] == 0.5 and q[2] == 0.99 and np.isnan(q[3])
    assert np.isnan(beat_quality_at({"beat_corr": [0.9]}, np.array([0.1]))).all()
