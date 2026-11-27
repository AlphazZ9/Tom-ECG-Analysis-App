"""The fast DFA gives NeuroKit's numbers bit for bit (DFA α1 / α2 and every
MFDFA index), only faster."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

nk = pytest.importorskip("neurokit2")
from neurokit2.hrv.hrv_nonlinear import _hrv_dfa  # noqa: E402

from ecg.core.fast_dfa import hrv_dfa  # noqa: E402


def _rri(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 120 + 4 * np.sin(t / 7) + np.cumsum(rng.normal(0, 0.3, n)) * 0.1 + rng.normal(0, 1, n)


def _both(rri):
    a, b = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _hrv_dfa(rri, a)
        hrv_dfa(rri, b)
    return a, b


@pytest.mark.parametrize("n, seed", [(12, 0), (60, 1), (130, 2), (1000, 3), (4321, 4)])
def test_same_numbers_as_neurokit(n, seed):
    a, b = _both(_rri(n, seed))
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def test_short_and_flat_series():
    a, b = _both(np.full(11, 120.0))
    assert a.keys() == b.keys() and np.isnan(b["DFA_alpha1"])
    # No variance anywhere: NeuroKit raises (the analysis leaves DFA out);
    # the fast one raises the same error.
    flat = np.full(200, 120.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as e_nk:
            _hrv_dfa(flat, {})
        with pytest.raises(ValueError) as e_fast:
            hrv_dfa(flat, {})
    assert str(e_nk.value) == str(e_fast.value)


def test_faster():
    rri = _rri(8000, 5)
    t0 = time.perf_counter(); _hrv_dfa(rri, {}); slow = time.perf_counter() - t0
    t0 = time.perf_counter(); hrv_dfa(rri, {}); fast = time.perf_counter() - t0
    assert fast < slow / 2, (slow, fast)


@pytest.mark.parametrize("n_beats", [800, 3000])      # entropy block not capped / capped
def test_dfa_on_demand_gives_the_values_of_the_former_analysis(n_beats):
    from ecg.core import analysis as A
    fs = 2000.0
    rng = np.random.default_rng(n_beats)
    rr = 0.12 + 0.004 * np.sin(np.arange(n_beats) / 9) + rng.normal(0, 0.002, n_beats)
    peaks = (np.cumsum(rr) * fs).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = A.analyse_hrv_nonlinear(peaks, fs, dfa=True)
        now = A.analyse_hrv_nonlinear(peaks, fs)
        dfa = A.window_dfa(peaks, fs)
    cols = A.dfa_columns(before.columns)
    assert "HRV_DFA_alpha1" in cols and "HRV_DFA_alpha2" in cols
    assert now.attrs["dfa_computed"] is False and before.attrs["dfa_computed"] is True
    for c in cols:
        assert np.isnan(now[c].iloc[0]), c                 # not computed by default
        np.testing.assert_array_equal(dfa[c], before[c].iloc[0], err_msg=c)
    # The rest of the table is untouched.
    for c in before.columns:
        if c not in cols:
            np.testing.assert_array_equal(now[c].values, before[c].values, err_msg=c)
