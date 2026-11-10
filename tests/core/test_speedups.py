"""The speed-ups give the same numbers: parallel delineation, the entropy
neighbour count, the threshold caches."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis, delineation_pool, detection, fast_entropy
from ecg.core.wave_template import WaveTemplate
from tests.helpers import make_synth_ecg

FS = 2000.0


def _recording(n_beats: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sig, fs, peaks = make_synth_ecg(n_beats=n_beats, fs=FS, rr_ms=120.0, noise_sd=0.03,
                                    seed=seed, p_amp=0.15)
    # Some RR variability and a T-like bump after each R, so every landmark
    # has something to find and the P search start (previous T) matters.
    t = np.arange(len(sig))
    for p in peaks:
        sig += 0.12 * np.exp(-0.5 * ((t - p - 0.035 * fs) / (0.006 * fs)) ** 2)
    sig += 0.01 * rng.standard_normal(len(sig))
    return sig, peaks


@pytest.mark.parametrize("n_ens", [1, 32])
def test_parallel_delineation_is_the_sequential_one(monkeypatch, n_ens):
    sig, peaks = _recording(1800)
    kw = dict(wave_template=WaveTemplate(), rr_max_ms=1000.0, ensemble_beats=n_ens)
    monkeypatch.setenv("ECG_DELIN_SERIAL", "1")
    serial = analysis.analyse_intervals(sig, peaks, FS, np.diff(peaks) / 2.0, **kw)
    monkeypatch.delenv("ECG_DELIN_SERIAL")
    monkeypatch.setattr(delineation_pool, "PARALLEL_MIN_ROWS", 100)
    parallel = analysis.analyse_intervals(sig, peaks, FS, np.diff(peaks) / 2.0, **kw)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    # No warm-up: every chunk starts from an unknown T peak, so the joints
    # are repaired in the parent -- still the sequential rows.
    monkeypatch.setattr(delineation_pool, "WARMUP", 0)
    repaired = analysis.analyse_intervals(sig, peaks, FS, np.diff(peaks) / 2.0, **kw)
    pd.testing.assert_frame_equal(serial, repaired, check_exact=True)


def test_chebyshev_counts_are_the_kdtree_ones():
    from sklearn.neighbors import KDTree
    rng = np.random.default_rng(3)
    for n, m in ((500, 2), (300, 3), (40, 1)):
        x = np.round(rng.standard_normal((n, m)), 2)        # ties on the radius
        for tol in (0.05, 0.2, 0.5):
            want = KDTree(x, metric="chebyshev").query_radius(x, tol, count_only=True)
            assert np.array_equal(fast_entropy.chebyshev_counts(x, tol), want.astype(float))


def test_non_linear_hrv_is_unchanged_by_the_fast_count():
    from neurokit2.complexity import entropy_approximate, utils_entropy
    rng = np.random.default_rng(5)
    rr = 170 + np.cumsum(rng.standard_normal(700)) * 0.3 + rng.standard_normal(700)
    peaks = np.r_[0, np.cumsum(np.round(rr / 1000 * FS))].astype(int) + 1000
    assert fast_entropy.install()          # once: later calls change nothing
    fast = utils_entropy._get_count
    original = fast.__wrapped__
    try:
        utils_entropy._get_count = entropy_approximate._get_count = original
        a = analysis.analyse_hrv_nonlinear(peaks, FS)       # NeuroKit's own count
    finally:
        utils_entropy._get_count = entropy_approximate._get_count = fast
    b = analysis.analyse_hrv_nonlinear(peaks, FS)
    assert {"HRV_SampEn", "HRV_ApEn", "HRV_MSEn", "HRV_RCMSEn"} <= set(a.columns)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_threshold_caches_give_the_peaks_of_a_fresh_computation():
    sig, peaks = _recording(500, seed=2)
    cands = np.sort(np.r_[peaks, peaks[:-1] + 90])           # R peaks and T-ish decoys
    proms = np.abs(sig[cands]) + 0.01 * np.arange(len(cands)) % 0.05
    results = {}
    for t in (0.5, 0.3, 0.8, 0.5):
        results.setdefault(t, []).append(detection.apply_threshold(sig, cands, proms, t, fs=FS))
        # A fresh computation: empty caches.
        detection._REF_CACHE.update(cands=None, proms=None, key=None, refs=None)
        detection._J_CACHE.update(signal=None, key=None, slope={}, noise={})
        fresh = detection.apply_threshold(sig, cands, proms, t, fs=FS)
        got = results[t][-1]
        assert np.array_equal(got[0], fresh[0]) and np.array_equal(got[1], fresh[1])
        assert got[2] == fresh[2]
    # New arrays (another recording, an inverted signal) are not served the
    # old cache.
    a = detection.apply_threshold(-sig, cands, proms.copy(), 0.5, fs=FS)
    detection._REF_CACHE.update(cands=None)
    detection._J_CACHE.update(signal=None)
    b = detection.apply_threshold(-sig, cands, proms.copy(), 0.5, fs=FS)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
