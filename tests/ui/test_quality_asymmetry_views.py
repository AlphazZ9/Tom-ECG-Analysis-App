"""
Beat-quality and RR-asymmetry views agree with the numbers they show:

- the RR CSV "quality" column and the "Poor-shape-match time" tile pin an
  RR interval to the same (leading) scored beat;
- a restored session from an older version gets the scored-beat times
  rebuilt from its peaks, or a status note when that is ambiguous;
- the Summary RR-asymmetry bars are shares of the pairs with ΔRR ≠ 0, so
  the acceleration bar is the Porta PI shown in the title ("—" when no
  pair is off the identity line).
"""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ecg.core import analysis
from ecg.core.analysis import noisy_time_pct, rr_asymmetry_indices, successive_valid_pairs
from ecg.ui import export_controller as ec
from ecg.ui.export_controller import ExportController
from ecg.ui.plot_controller import PlotController
from ecg.ui.session_controller import SessionController

from tests.ui.test_ui_computation_links import (  # noqa: F401  (fixtures)
    _no_template_writes, _no_tk_widgets, _stub_app, recording,
)

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


# ── RR CSV quality column <-> Poor-shape-match tile ────────────────────────

def test_rr_csv_quality_and_tile_use_the_leading_beat(recording, tmp_path, monkeypatch):
    sig, fs, rp, r = recording
    r = dict(r)
    bc = np.asarray(r["beat_corr"], dtype=float).copy()
    bc[:] = 0.99
    bc[9] = 0.1                       # one poor scored beat (peak 10: peak 1 is unscored)
    r["beat_corr"] = bc
    poor_t = r["beat_corr_t_s"][9]

    app, _, _ = _stub_app(sig, fs, rp, r)
    path = tmp_path / "rr.csv"
    monkeypatch.setattr(ec.filedialog, "asksaveasfilename", lambda **kw: str(path))
    ExportController(app).export_rr_csv()
    df = pd.read_csv(path, keep_default_na=False)

    start = df["Time_s"] - df["RR_ms"] / 1000.0
    low = df[df["quality"] == "low"]
    assert len(low) == 1
    assert start[low.index[0]] == pytest.approx(poor_t, abs=1e-3)   # starts at the poor beat
    assert df["quality"].iloc[0] == ""         # starts at the unscored edge beat

    scored = df[df["quality"] != ""]
    expected = 100.0 * low["RR_ms"].sum() / scored["RR_ms"].sum()
    assert noisy_time_pct(r) == pytest.approx(expected, rel=1e-3)


def test_rr_csv_leaves_out_quality_when_beats_are_unknown(recording, tmp_path, monkeypatch):
    sig, fs, rp, r = recording
    legacy = dict(r)
    legacy.pop("beat_corr_t_s")                # older results, edge beat dropped
    app, _, _ = _stub_app(sig, fs, rp, legacy)
    path = tmp_path / "rr.csv"
    monkeypatch.setattr(ec.filedialog, "asksaveasfilename", lambda **kw: str(path))
    ExportController(app).export_rr_csv()
    assert "quality" not in pd.read_csv(path).columns
    assert np.isnan(noisy_time_pct(legacy))   # tile "—" alike


# ── Session restore of older results ───────────────────────────────────────

def test_restore_rebuilds_the_scored_beat_times(recording):
    sig, fs, rp, r = recording
    legacy = dict(r)
    legacy.pop("beat_corr_t_s")
    app, _, _ = _stub_app(sig, fs, rp, legacy)
    note = SessionController(app).restore_beat_quality_times({"analysed_window": (0.0, 0.0)})
    assert note == ""
    np.testing.assert_allclose(app.analysis.results["beat_corr_t_s"], r["beat_corr_t_s"])


def test_restore_notes_unknown_scored_beats(recording):
    sig, fs, rp, r = recording
    legacy = dict(r)
    legacy.pop("beat_corr_t_s")
    legacy["beat_corr"] = np.asarray(r["beat_corr"])[:-3]     # count matches no rule
    app, _, _ = _stub_app(sig, fs, rp, legacy)
    note = SessionController(app).restore_beat_quality_times({"analysed_window": (0.0, 0.0)})
    assert "beat-quality times not recorded" in note
    assert app.analysis.results.get("beat_corr_t_s") is None
    # Current results: nothing to do.
    app, _, _ = _stub_app(sig, fs, rp, dict(r))
    assert SessionController(app).restore_beat_quality_times({}) == ""


# ── Summary RR-asymmetry bars ──────────────────────────────────────────────

def _asymmetry_axes(sig, fs, rp, r):
    app, _, _ = _stub_app(sig, fs, rp, r)
    PlotController(app).plot_summary(app.analysis.results)
    fig = Figure()
    app._slots["sum_asymmetry"].draw_fn(fig)
    return fig.axes[0]


def test_asymmetry_bars_are_shares_of_off_line_pairs(recording):
    sig, fs, rp, r = recording
    rng = np.random.default_rng(1)
    rr = np.round(120 + rng.normal(0, 1.5, len(r["rr_ms"])))  # integer ms: many ΔRR = 0
    r = dict(r, rr_ms=rr, rr_valid=np.ones(len(rr), dtype=bool))
    a, b = successive_valid_pairs(rr, np.ones(len(rr), dtype=bool))
    ix = rr_asymmetry_indices(a, b)
    assert ix["n_zero"] > 0

    ax = _asymmetry_axes(sig, fs, rp, r)
    heights = [p.get_height() for p in ax.patches]
    assert len(heights) == 2                                   # no "Neutral" bar
    assert heights[1] == pytest.approx(ix["porta_pct"])        # accelerations = PI
    assert sum(heights) == pytest.approx(100.0)
    assert ax.get_ylabel() == "% of pairs with ΔRR ≠ 0"
    assert f"{ix['n_zero']} of {ix['n_tot']} pairs" in ax.get_xlabel()
    assert f"Porta PI={ix['porta_pct']:.1f}%" in ax.get_title(loc="left")


def test_asymmetry_title_shows_dash_without_off_line_pairs(recording):
    sig, fs, rp, r = recording
    rr = np.full(len(r["rr_ms"]), 120.0)                       # every ΔRR = 0
    r = dict(r, rr_ms=rr, rr_valid=np.ones(len(rr), dtype=bool))
    ax = _asymmetry_axes(sig, fs, rp, r)
    title = ax.get_title(loc="left")
    assert "nan" not in title
    assert "Porta PI=—" in title and "Guzik GI=—" in title
