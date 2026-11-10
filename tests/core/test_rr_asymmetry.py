"""
Heart-rate asymmetry indices (rr_asymmetry_indices) follow the literature
conventions: Porta PI = points below the identity line / off-line points
(ΔRR = 0 left out of the denominator), Guzik GI = Σ distance of the points
above the line / Σ distance of all off-line points -- the same definitions
as NeuroKit's HRV_PI / HRV_GI.
"""
import numpy as np
import pytest

from ecg.core import analysis
from ecg.core.analysis import (
    RR_ASYMMETRY_TEXT, rr_asymmetry_indices, successive_valid_pairs,
)

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_porta_excludes_zero_differences_from_the_denominator():
    # ΔRR: +10, -5, 0, 0, -5  -> 1 deceleration, 2 accelerations, 2 zeros
    rr = np.array([100.0, 110.0, 105.0, 105.0, 105.0, 100.0])
    ix = rr_asymmetry_indices(rr[:-1], rr[1:])
    assert (ix["n_dec"], ix["n_acc"], ix["n_zero"], ix["n_off"], ix["n_tot"]) == (1, 2, 2, 3, 5)
    assert ix["porta_pct"] == pytest.approx(100.0 * 2 / 3)       # not 2 / 5


def test_guzik_is_the_deceleration_share_of_the_distances():
    # ΔRR: +10, -5, -5 -> distances 10, 5, 5 (×1/√2, which cancels)
    rr = np.array([100.0, 110.0, 105.0, 100.0])
    ix = rr_asymmetry_indices(rr[:-1], rr[1:])
    assert ix["guzik_pct"] == pytest.approx(50.0)
    # Distances, not squared distances (that would give 100/150 = 66.7 %).
    rr = np.array([100.0, 112.0, 106.0, 100.0])                # +12, -6, -6
    assert rr_asymmetry_indices(rr[:-1], rr[1:])["guzik_pct"] == pytest.approx(50.0)
    rr = np.array([100.0, 130.0, 120.0])                       # +30, -10
    assert rr_asymmetry_indices(rr[:-1], rr[1:])["guzik_pct"] == pytest.approx(75.0)


def test_no_off_line_point_gives_nan_not_50():
    rr = np.full(10, 120.0)
    ix = rr_asymmetry_indices(rr[:-1], rr[1:])
    assert np.isnan(ix["porta_pct"]) and np.isnan(ix["guzik_pct"])
    assert ix["n_zero"] == ix["n_tot"] == 9
    empty = rr_asymmetry_indices(np.array([]), np.array([]))
    assert empty["n_tot"] == 0 and np.isnan(empty["porta_pct"])


@requires_nk
def test_matches_neurokit_pi_and_gi():
    from neurokit2.hrv.hrv_nonlinear import _hrv_nonlinear_poincare_hra
    rng = np.random.default_rng(4)
    rr = np.round(120 + rng.normal(0, 4, 400))                  # integer ms: many ΔRR = 0
    a, b = successive_valid_pairs(rr)
    ix = rr_asymmetry_indices(a, b)
    assert ix["n_zero"] > 0
    nk_out: dict = {}
    _hrv_nonlinear_poincare_hra(rr, out=nk_out)
    assert ix["porta_pct"] == pytest.approx(nk_out["PI"])
    assert ix["guzik_pct"] == pytest.approx(nk_out["GI"])


def test_convention_text_names_both_definitions():
    assert "Porta PI" in RR_ASYMMETRY_TEXT and "Guzik GI" in RR_ASYMMETRY_TEXT
    assert "ΔRR ≠ 0" in RR_ASYMMETRY_TEXT
