"""
The metric table reads the results, it does not compute: every row equals its
source exactly, including NaN, withheld spectra and legacy sessions, and states
what the value rests on (basis n), per column for the intervals.
"""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.analysis import (
    analyse_core, analyse_hrv_freq, analyse_hrv_nonlinear, mean_hr_bpm,
    migrate_legacy_results, noisy_time_pct, qt_beat_variability,
    rr_asymmetry_indices, successive_valid_pairs,
)
from ecg.core.metric_table import (
    MISSING, MODULES, MetricTable, build_metric_table, interval_rows,
    intervals_measured, quality_tile_texts,
)

from tests.helpers import make_synth_ecg

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture(scope="module")
def results():
    sig, fs, rp = make_synth_ecg(n_beats=120, seed=3)
    r = analyse_core(sig, rp, fs)
    r["hrv_freq"] = analyse_hrv_freq(rp, fs)
    r["hrv_nonlin"] = analyse_hrv_nonlinear(rp, fs)
    return r


def _same(a, b) -> bool:
    return (isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b)) or a == b


# ── every row equals its source ─────────────────────────────────────────────

def test_hr_rows_equal_the_hr_dict(results):
    t = build_metric_table(results)
    hr = results["hr"]
    for k in ("mean", "min", "max", "std"):
        assert t.value(f"hr.{k}") == hr[k]
    assert t.value("hr.n_beats") == hr["n"]
    assert t.value("hr.n_valid") == hr["n_valid"]
    assert t.row("hr.mean").unit == "bpm"
    assert t.row("hr.mean").basis_n == hr["n_valid"]


def test_rr_df_rows_equal_their_source(results):
    t = build_metric_table(results)
    rdf = results["rr_df"]
    assert t.value("hr.mean_rr") == mean_hr_bpm(rdf["RR_ms"].to_numpy(dtype=float))
    assert t.value("rr.mean_rr") == float(rdf["RR_ms"].mean())
    assert t.value("hr.range_lo") == rdf["HR_bpm"].quantile(0.02)
    assert t.value("hr.range_hi") == rdf["HR_bpm"].quantile(0.98)


@pytest.mark.parametrize("frame,prefix", [("hrv_time", "hrv_time"), ("hrv_freq", "hrv_freq"),
                                          ("hrv_nonlin", "hrv_nonlin")])
def test_every_readable_frame_column_is_a_row_with_its_exact_value(results, frame, prefix):
    t = build_metric_table(results)
    df = results[frame]
    assert len(df.columns)
    for col in df.columns:
        src = float(df[col].values[0])
        assert _same(t.value(f"{prefix}.{col}"), src), col


def test_curated_and_all_tiers(results):
    t = build_metric_table(results)
    assert t.row("hrv_time.HRV_SDNN").tier == "curated"
    assert t.row("hrv_time.HRV_SDNN").unit == "ms"
    assert t.row("hrv_time.HRV_SDNN").decimals == 1
    extra = [c for c in results["hrv_time"].columns if c not in
             ("HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN6", "HRV_pNN20")]
    assert extra and t.row(f"hrv_time.{extra[0]}").tier == "all"


def test_asymmetry_is_the_summary_calculation(results):
    t = build_metric_table(results)
    a, b = successive_valid_pairs(results["rr_ms"], results["rr_valid"])
    ix = rr_asymmetry_indices(a, b)
    assert _same(t.value("asymmetry.porta_pct"), ix["porta_pct"])
    assert _same(t.value("asymmetry.guzik_pct"), ix["guzik_pct"])
    assert t.value("asymmetry.n_dec") == ix["n_dec"]
    assert t.row("asymmetry.porta_pct").basis_n == ix["n_tot"]
    assert not build_metric_table(results, asymmetry=None).has("asymmetry.porta_pct")


def test_quality_rows_and_shared_texts(results):
    t = build_metric_table(results, sig_quality=66)
    bc = results["beat_corr"]
    n_bad = int(np.sum(bc < 0.90))
    assert t.value("quality.mean_corr") == float(np.nanmean(bc))
    assert t.value("quality.n_bad") == n_bad
    assert t.value("quality.bad_pct") == 100.0 * n_bad / len(bc)
    assert t.value("quality.score") == 66
    assert _same(t.value("quality.noisy_time_pct"), noisy_time_pct(results))
    texts = quality_tile_texts(t)
    assert texts["sq_corr"] == f"{float(np.nanmean(bc)):.3f}"
    assert texts["sq_badbeats"] == f"{100.0 * n_bad / len(bc):.1f}%  ({n_bad}/{len(bc)})"
    assert texts["sq_artifact"] == "not applied"


def test_artifact_rows_and_text():
    arep = {"n_in": 100, "n_out": 97, "n_duplicate": 1, "n_nonphysio": 0, "n_ectopic": 2}
    t = build_metric_table({}, artifact_report=arep)
    assert t.value("artifact.removed") == 3
    assert quality_tile_texts(t)["sq_artifact"] == "3  (1 dup · 0 non-physio · 2 ectopic)"


# ── missing values ─────────────────────────────────────────────────────────

def test_nan_stays_nan_and_formats_as_a_dash_or_raw():
    r = {"hrv_time": pd.DataFrame({"HRV_SDNN": [float("nan")]})}
    t = build_metric_table(r)
    assert math.isnan(t.value("hrv_time.HRV_SDNN"))
    assert t.finite("hrv_time.HRV_SDNN") is None
    assert t.text("hrv_time.HRV_SDNN", 1) == MISSING
    assert t.raw_text("hrv_time.HRV_SDNN", 1) == "nan"
    # An absent row: a dash in both conventions, never an exception.
    assert t.text("hrv_time.HRV_RMSSD", 1) == MISSING
    assert t.raw_text("hrv_time.HRV_RMSSD", 1) == MISSING


def test_unreadable_cell_gives_no_row():
    t = build_metric_table({"hrv_time": pd.DataFrame({"HRV_SDNN": ["n/a"]})})
    assert not t.has("hrv_time.HRV_SDNN")


def test_no_results_still_lists_every_module():
    t = build_metric_table(None, states={m: "not computed" for m in MODULES})
    assert [r.module for r in t.rows] == list(MODULES)
    assert all(r.state == "not computed" and r.basis_n is None for r in t.rows)
    assert all(math.isnan(r.value) for r in t.rows)


def test_withheld_spectrum_has_no_value_rows_and_says_why():
    fd = pd.DataFrame()
    fd.attrs["withheld_reason"] = "only 12 valid RR intervals (need ≥ 60)"
    t = build_metric_table({"hrv_freq": fd})
    assert not t.has("hrv_freq.HRV_LF")
    assert t.row("frequency.status").note == ("withheld — only 12 valid RR intervals "
                                              "(need ≥ 60)")
    assert t.row("frequency.status").basis_n is None


def test_legacy_spectrum_is_labelled_in_arbitrary_units():
    # No "method" attr: an older session's NeuroKit table.
    fd = pd.DataFrame({"HRV_LF": [0.2], "HRV_LFn": [0.4], "HRV_LFHF": [2.0]})
    t = build_metric_table({"hrv_freq": fd})
    assert t.value("hrv_freq.HRV_LF") == 0.2
    assert t.row("hrv_freq.HRV_LF").unit == "a.u., legacy"
    assert t.row("hrv_freq.HRV_LFn").unit == "fraction of TP, legacy"
    assert "legacy" in t.row("frequency.status").note


def test_legacy_session_results_read_the_migrated_hr():
    rr = np.array([100.0, 110.0, 120.0, 105.0])
    r = {"hr": {"mean": 123.0, "min": 1.0, "max": 2.0, "std": 3.0, "n": 5},
         "rr_df": pd.DataFrame({"RR_ms": rr, "HR_bpm": 60000.0 / rr, "Time_s": np.arange(4.0)})}
    migrate_legacy_results(r)
    t = build_metric_table(r)
    assert t.value("hr.mean") == mean_hr_bpm(rr)
    assert t.value("hr.n_valid") == 5      # no n_valid recorded: as the KPI falls back


def test_a_state_comes_from_the_argument_only(results):
    t = build_metric_table(results)
    assert {r.state for r in t.rows} == {""}
    t = build_metric_table(results, states={"core": "out of date", "frequency": "up to date"})
    assert t.row("hr.mean").state == "out of date"
    assert t.row("hrv_freq.HRV_LF").state == "up to date"
    assert t.row("intervals.status").state == ""


# ── basis n ────────────────────────────────────────────────────────────────

def _ivl():
    # PR, QRS and QT fail on different beats, as on real recordings.
    n = 10
    pr = [40.0, 41.0, 42.0] + [np.nan] * 7
    qrs = [10.0] * 9 + [np.nan]
    qt = [50.0] * 7 + [np.nan] * 3
    qtc = [39.0] * 6 + [np.nan] * 4
    return pd.DataFrame({"RR_ms": [170.0] * n, "PR_ms": pr, "QRS_ms": qrs, "QT_ms": qt, "QTc_ms": qtc})


def test_interval_rows_use_each_columns_own_beats():
    ivl = _ivl()
    t = MetricTable(interval_rows(ivl))
    assert intervals_measured(ivl) == (3, 10)
    for col, n in (("PR_ms", 3), ("QRS_ms", 9), ("QT_ms", 7), ("QTc_ms", 6)):
        d = ivl[col].dropna()
        row = t.row(f"intervals.{col}")
        assert row.value == float(d.median())
        assert t.value(f"intervals.{col}.sd") == float(d.std())
        assert (row.basis_n, row.basis_total) == (n, 10)
        # The app's "n/N beats measured" is said on every row too.
        assert "3 of 10 beats measured on PR, QRS and QT together" in row.basis_text
    measured = t.row("intervals.beats_measured")
    assert (measured.value, measured.basis_n, measured.basis_total) == (3, 3, 10)


def test_interval_iqr_equals_its_source():
    ivl = _ivl()
    q = qt_beat_variability(ivl["QT_ms"].to_numpy(dtype=float))
    row = MetricTable(interval_rows(ivl)).row("intervals.QT_iqr")
    assert _same(row.value, q["iqr_ms"]) and row.basis_n == q["n"]


def test_column_with_no_measured_beat_has_no_value():
    ivl = _ivl()
    ivl["PR_ms"] = np.nan
    row = MetricTable(interval_rows(ivl)).row("intervals.PR_ms")
    assert math.isnan(row.value) and row.basis_n == 0


def test_analyse_core_placeholder_gives_no_interval_rows(results):
    t = build_metric_table(results)     # intervals = RR_ms only until Intervals runs
    assert [r.key for r in t.module_rows("intervals")] == ["intervals.status"]
    assert t.row("intervals.status").basis_n is None


def test_entropy_basis_comes_from_the_attrs():
    attrs = {"n_rr_valid": 3000, "entropy_capped": True, "entropy_n_rr": 2000,
             "entropy_max_beats": 2000, "entropy_t_start_s": 100.0, "entropy_t_end_s": 440.0,
             "truncated": False}
    nl = pd.DataFrame({"HRV_SampEn": [1.2], "HRV_DFA_alpha1": [0.4], "HRV_SD1": [3.0]})
    nl.attrs.update(attrs)
    t = build_metric_table({"hrv_nonlin": nl, "rr_ms": np.full(5, 170.0)})
    row = t.row("hrv_nonlin.HRV_SampEn")
    assert (row.basis_n, row.basis_total) == (2000, 3000)
    assert "2,000 of 3,000" in row.basis_text and "100.0–440.0 s" in row.basis_text
    # DFA uses every valid interval, whatever the entropy cap.
    assert t.row("hrv_nonlin.HRV_DFA_alpha1").basis_n == 3000
    assert t.row("hrv_nonlin.HRV_SD1").basis_text.endswith("successive valid RR pairs")


def test_older_nonlinear_result_states_its_truncation():
    nl = pd.DataFrame({"HRV_SampEn": [1.2]})
    nl.attrs.update({"truncated": True, "n_beats_used": 1000, "n_beats_total": 4000})
    row = build_metric_table({"hrv_nonlin": nl}).row("hrv_nonlin.HRV_SampEn")
    assert (row.basis_n, row.basis_total) == (1000, 4000)
    assert "older analysis" in row.basis_text


def test_series_modules_hold_their_state_and_n():
    res = SimpleNamespace(sd01=1.5, sd02=2.5, d0=3.5,
                          params={"n_beats": 40, "n_windows_withheld": 2})
    ev = [SimpleNamespace(kind="premature_beat"), SimpleNamespace(kind="premature_beat"),
          SimpleNamespace(kind="pause")]
    t = build_metric_table(
        epoch_df=pd.DataFrame({"HR_mean": [1.0, 2.0]}),
        rolling_df=pd.DataFrame({"HR": [1.0, 2.0, 3.0]}),
        autonomic=res, event_params={"n_events": 3, "n_valid": 2},
        events=ev, premature={"n_premature": 2, "premature_per_h": 7.5, "span_s": 960.0})
    assert t.row("epochs.status").basis_n == 2
    assert t.row("rolling.status").basis_n == 3
    assert t.row("autonomic.status").basis_n == 40
    assert t.value("autonomic.SD01_ms") == 1.5
    assert t.value("autonomic.D0_ms") == 3.5
    ev_row = t.row("event_locked.status")
    assert (ev_row.basis_n, ev_row.basis_total) == (2, 3)
    assert t.value("events.kind.premature_beat") == 2
    assert t.value("events.n_premature") == 2
    assert t.value("events.premature_per_h") == 7.5
    assert t.row("events.n_premature").basis_text == "960.0 s analysed"


def test_frame_form_keeps_raw_values_and_int_counts(results):
    df = build_metric_table(results).to_frame()
    line = df[df["Key"] == "hr.n_valid"].iloc[0]
    assert line["Value"] == results["hr"]["n_valid"]
    assert isinstance(line["Basis_n"], (int, np.integer))
    line = df[df["Key"] == "hrv_time.HRV_SDNN"].iloc[0]
    assert line["Value"] == float(results["hrv_time"]["HRV_SDNN"].values[0])
