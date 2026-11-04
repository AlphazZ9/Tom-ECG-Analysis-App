"""
Rolling-median delineation (analyse_intervals, ensemble_beats) on synthetic
signals of known truth, at the noise of the real anaesthetised-mouse recordings:
noise SD 0.03 of R and a P wave of 0.03 of R, so P is at single-beat noise level.
"""
import json

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.detection import classify_arrhythmias
from ecg.core.metric_table import MetricTable, interval_rows
from ecg.core.wave_template import WaveTemplate
from ecg.io.loaders import _deserialise_results, _serialise_results

FS = 2000.0
RR = 170.0
P_ON, QRS_ON, QRS_OFF, T_END = -50.0, -8.0, 8.0, 55.0
PR_TRUE, QRS_TRUE, QT_TRUE = QRS_ON - P_ON, QRS_OFF - QRS_ON, T_END - QRS_ON
NOISE = 0.03


def _waves(pr=PR_TRUE, p_amp=0.03, qrs_off=QRS_OFF, t_end=T_END):
    p_on = QRS_ON - pr
    return [(-120.0, 0.0), (p_on, 0.0), (p_on + 8.0, p_amp), (p_on + 16.0, 0.0),
            (QRS_ON, 0.0), (-5.0, -0.2), (0.0, 1.0), (4.0, -0.3), (qrs_off, 0.0),
            (qrs_off + 4.0, 0.1), (qrs_off + 10.0, 0.0), (25.0 + qrs_off - QRS_OFF, 0.0),
            (35.0 + qrs_off - QRS_OFF, 0.2), (t_end, 0.0), (120.0, 0.0)]


def _train(rr_ms, noise=NOISE, seed=0, beat_waves=None):
    """Beats at the given RR intervals (ms); beat_waves[i] overrides beat i's waves."""
    rp = [int(0.3 * FS)]
    for rr in rr_ms:
        rp.append(rp[-1] + int(round(rr / 1000.0 * FS)))
    rp = np.array(rp)
    n = rp[-1] + int(0.3 * FS)
    t_ms = np.arange(n) / FS * 1000.0
    sig = np.zeros(n)
    default = _waves()
    for i, r in enumerate(rp):
        lo, hi = r - int(0.085 * FS), r + int(0.085 * FS)
        xs, ys = zip(*(beat_waves or {}).get(i, default))
        sig[lo:hi] += np.interp(t_ms[lo:hi] - t_ms[r], xs, ys)
    return sig + np.random.default_rng(seed).normal(0.0, noise, n), rp


def _measure(sig, rp, n, **kw):
    wt = WaveTemplate()
    wt.confirmed = True
    return analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                      wave_template=wt, ensemble_beats=n,
                                      rr_max_ms=1000.0, **kw)


# ── N = 0 / 1: today's values ──────────────────────────────────────────────

@pytest.mark.parametrize("n", [0, 1])
def test_n_0_or_1_delineates_each_beat_on_itself(monkeypatch, n):
    sig, rp = _train([RR] * 60)
    real = analysis.detect_waves_on_beat
    seen = []

    def spy(beat, bt, *a, **k):
        seen.append(beat.copy())
        return real(beat, bt, *a, **k)

    monkeypatch.setattr(analysis, "detect_waves_on_beat", spy)
    df = _measure(sig, rp, n)
    monkeypatch.undo()
    base = analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                      wave_template=_confirmed(), rr_max_ms=1000.0)
    pd.testing.assert_frame_equal(df, base, check_exact=True)
    assert "ensemble_n" not in df.columns and "RR_ensemble_ms" not in df.columns
    assert analysis.ensemble_info(df) is None and df.attrs["ensemble"] == {"n": 1}
    pre, post = analysis.beat_window_sides(rp, FS)
    for row, beat in zip(range(len(df)), seen):
        r = int(round(df["R_peak_s"].iloc[row] * FS))
        i = int(np.searchsorted(rp, r))
        assert np.array_equal(beat, sig[r - pre[i]:r + post[i]])


def _confirmed():
    wt = WaveTemplate()
    wt.confirmed = True
    return wt


# ── Accuracy at N = 32 ─────────────────────────────────────────────────────

def test_n32_recovers_pr_and_qt_where_single_beats_fail():
    sig, rp = _train([RR] * 300)
    single, df = _measure(sig, rp, 1), _measure(sig, rp, 32)
    # Every other row delineated on a median of ~32 beats.
    assert df["PR_ms"].notna().mean() > 0.8
    assert df["QT_ms"].notna().mean() > 0.95
    assert df["PR_ms"].median() == pytest.approx(PR_TRUE, abs=2.0)
    assert df["QRS_ms"].median() == pytest.approx(QRS_TRUE, abs=1.5)
    assert df["QT_ms"].median() == pytest.approx(QT_TRUE, abs=2.5)
    # The single beats are noisy, or missing, on the same signal.
    spread = lambda d, c: float(d[c].quantile(0.75) - d[c].quantile(0.25))
    assert (single["PR_ms"].notna().mean() < 0.5
            or spread(single, "PR_ms") > 3 * spread(df, "PR_ms"))
    assert spread(single, "QT_ms") > 2 * spread(df, "QT_ms")
    info = df.attrs["ensemble"]
    assert info["n"] == 32 and info["statistic"] == "median"
    assert info["window_s"] == pytest.approx(32 * RR / 1000.0, abs=0.01)
    assert info["n_artifact"] == 0 and info["n_timing"] == 0
    assert df["ensemble_n"].iloc[len(df) // 2] == 32
    assert info["members_min"] >= 16        # the first row has only one side


# ── Artifacts ──────────────────────────────────────────────────────────────

def _with_artifacts(sig, rp, beats, seed=1):
    out = sig.copy()
    rng = np.random.default_rng(seed)
    for b in beats:
        lo, hi = rp[b] - int(0.05 * FS), rp[b] + int(0.05 * FS)
        out[lo:hi] += rng.normal(0.0, 2.0, hi - lo) + 6.0 * np.sin(np.linspace(0, 9, hi - lo))
    return out


def test_scattered_artifacts_are_excluded_counted_and_leave_neighbours_intact():
    sig, rp = _train([RR] * 300)
    # 1 beat in 3 over 60 beats: below the median's breakdown, but it would still bias it.
    bad = list(range(100, 160, 3))
    df = _measure(_with_artifacts(sig, rp, bad), rp, 32)
    assert df.attrs["ensemble"]["n_artifact"] == len(bad)
    r = np.round(df["R_peak_s"].to_numpy() * FS).astype(int)
    at_bad = np.isin(r, rp[bad])
    assert (df.loc[at_bad, "ensemble_n"] == 0).all()
    assert df.loc[at_bad, ["PR_ms", "QRS_ms", "QT_ms", "QTc_ms"]].isna().all(axis=None)
    near = df[~at_bad & (r >= rp[100]) & (r <= rp[160])]
    assert (near["ensemble_n"] < 32).any()
    assert near["PR_ms"].median() == pytest.approx(PR_TRUE, abs=2.0)
    assert near["QT_ms"].median() == pytest.approx(QT_TRUE, abs=2.5)
    assert near["QT_ms"].notna().mean() > 0.9


def test_a_burst_longer_than_half_the_window_does_not_condemn_the_clean_beats():
    sig, rp = _train([RR] * 300)
    bad = list(range(120, 160))          # 40 consecutive beats > N
    df = _measure(_with_artifacts(sig, rp, bad), rp, 32)
    assert df.attrs["ensemble"]["n_artifact"] == len(bad)
    clean = df[df["ensemble_n"] > 0]
    assert len(clean) == len(df) - len(bad)
    assert clean["QT_ms"].median() == pytest.approx(QT_TRUE, abs=2.5)
    assert clean["PR_ms"].median() == pytest.approx(PR_TRUE, abs=2.0)


def test_normal_beat_noise_excludes_nothing():
    sig, rp = _train(list(np.random.default_rng(3).normal(RR, 2.0, 400)))
    assert _measure(sig, rp, 32).attrs["ensemble"]["n_artifact"] == 0


def _rows_at(df, rp, beats):
    r = np.round(df["R_peak_s"].to_numpy() * FS).astype(int)
    return np.isin(r, rp[beats])


@pytest.mark.parametrize("first, n_beats", [(150, 6), (120, 60)])
def test_a_moderate_noise_burst_is_excluded_however_long(first, n_beats):
    # Noise of 8 x the single-beat noise (~ 14 % of R on the real recordings),
    # over 1 s or over 10 s: longer than the residual's local level, which a
    # burst must not raise for itself.
    sig, rp = _train([RR] * 300)
    lo, hi = rp[first] - int(0.08 * FS), rp[first + n_beats - 1] + int(0.08 * FS)
    sig[lo:hi] += np.random.default_rng(5).normal(0.0, 8 * NOISE, hi - lo)
    df = _measure(sig, rp, 32)
    inside = _rows_at(df, rp, range(first, first + n_beats))
    assert (df.loc[inside, "ensemble_n"] == 0).all()
    assert df.attrs["ensemble"]["n_artifact"] == n_beats
    assert df.loc[~inside, "QT_ms"].median() == pytest.approx(QT_TRUE, abs=2.5)


def test_a_narrow_spike_on_the_p_wave_is_excluded():
    # 1 ms on the rail (2.5 x R), 36 ms before R: the RMS of the whole
    # window hardly sees it, nor does the R sample.
    sig, rp = _train([RR] * 200)
    k = rp[100] - int(0.036 * FS)
    sig[k:k + 2] -= 2.5
    df = _measure(sig, rp, 32)
    at = _rows_at(df, rp, [100])
    assert df.attrs["ensemble"]["n_artifact"] == 1
    assert df.loc[at, "ensemble_n"].iloc[0] == 0


# ── T end on the median beat ───────────────────────────────────────────────

def test_t_consensus_takes_the_median_t_end_of_the_rows_around():
    r = np.arange(40, dtype=float) * 340.0
    rel_o = np.where(np.arange(40) % 4 == 1, 46.0, 33.0) * 2.0     # samples from R
    rel_p = rel_o - 6.0
    rel_o[20:26] = np.nan                                          # a few rows without
    rel_o[30:40] = np.nan                                          # most rows without
    in_ens = np.ones(40, dtype=bool)
    in_ens[5] = False                                              # its own beat
    tp, to = analysis._ensemble_t_consensus(r + rel_p, r + rel_o, r, in_ens, 4, 4 * 340.0 * 2.3)
    got = to - r
    assert got[5] == rel_o[5]                                      # untouched
    assert np.allclose(got[8:18], 66.0)                            # the flips are gone
    assert np.isfinite(got[19]) and np.isnan(got[22]) and np.isnan(got[36])
    # T peak and T end of one row, so the peak stays before its end.
    ok = np.isfinite(to)
    assert np.allclose((to - tp)[ok], 6.0)


def test_adjacent_rows_agree_on_qt_and_own_rows_keep_their_beat():
    rr = [RR] * 200
    rr[99], rr[100] = 100.0, 240.0
    sig, rp = _train(rr, noise=0.04)
    df, single = _measure(sig, rp, 32), _measure(sig, rp, 1)
    info = df.attrs["ensemble"]
    assert info["t_rows"] == 2 * (32 // analysis._ENSEMBLE_T_ROWS_FRACTION) + 1
    full = df["ensemble_n"].to_numpy() >= 32
    qt = df["QT_ms"].to_numpy()
    pair = full[1:] & full[:-1] & np.isfinite(qt[1:]) & np.isfinite(qt[:-1])
    assert np.nanmax(np.abs(np.diff(qt))[pair]) < 3.0
    own = (df["ensemble_n"] == 1).to_numpy()
    assert own.sum() == 2
    # Untouched by the T consensus. (Their P search starts at the previous
    # row's T peak, which is that row's median-beat one: PR is not compared.)
    cols = ["QRS_ms", "QT_ms", "QTc_ms", "T_peak_ms", "T_offset_ms"]
    pd.testing.assert_frame_equal(df.loc[own, cols], single.loc[own, cols])


# ── Premature beats ────────────────────────────────────────────────────────

def test_premature_and_following_beat_report_themselves_and_join_no_ensemble():
    rr = [RR] * 200
    rr[99], rr[100] = 100.0, 240.0       # beat 100 premature, beat 101 after the pause
    wide = _waves(qrs_off=22.0, t_end=70.0, p_amp=0.0)
    sig, rp = _train(rr, noise=0.005, beat_waves={100: wide})
    df = _measure(sig, rp, 32)
    r = np.round(df["R_peak_s"].to_numpy() * FS).astype(int)
    row_of = {int(p): i for i, p in zip(df.index, r)}
    prem, after = row_of[int(rp[100])], row_of[int(rp[101])]
    assert df.at[prem, "ensemble_n"] == 1 and df.at[after, "ensemble_n"] == 1
    assert df.attrs["ensemble"]["n_timing"] == 2
    # It keeps its own wide QRS and no neighbour's P, not the neighbours' morphology.
    assert df.at[prem, "QRS_ms"] == pytest.approx(22.0 - QRS_ON, abs=2.0)
    assert np.isnan(df.at[prem, "PR_ms"])
    # And its own RR, not an ensemble's.
    assert df.at[prem, "RR_ensemble_ms"] == pytest.approx(df.at[prem, "RR_ms"])
    # The neighbours keep theirs: the window spans the abnormal beat, which is not in it.
    others = df.index[(np.abs(df.index - prem) <= 8) & ~df.index.isin([prem, after])]
    assert (df.loc[others, "ensemble_n"] <= 31).all()
    assert df.loc[others, "QRS_ms"].median() == pytest.approx(QRS_TRUE, abs=1.0)


def test_frequent_premature_beats_never_enter_the_ensemble_of_normal_rows():
    # Normal, normal, premature, compensatory: half the beats are excluded.
    rr = []
    waves = {}
    for k in range(60):
        rr += [RR, RR, 100.0, 240.0]
        waves[4 * k + 3] = _waves(qrs_off=22.0, t_end=70.0)
    sig, rp = _train(rr, noise=0.005, beat_waves=waves)
    df = _measure(sig, rp, 32)
    normal = df[df["ensemble_n"] > 1]
    assert (normal["ensemble_n"] <= 17).all()
    assert normal["QRS_ms"].median() == pytest.approx(QRS_TRUE, abs=1.0)
    assert df.attrs["ensemble"]["n_timing"] == (df["ensemble_n"] == 1).sum()


# ── What the window smooths ────────────────────────────────────────────────

def _pr_episode(first, n_beats, pr):
    waves = {i: _waves(pr=pr) for i in range(first, first + n_beats)}
    return _train([RR] * 300, beat_waves=waves)


def test_an_episode_shorter_than_half_the_window_is_smoothed_away():
    sig, rp = _pr_episode(150, 8, 60.0)
    df = _measure(sig, rp, 32)
    assert df.loc[148:158, "PR_ms"].median() == pytest.approx(PR_TRUE, abs=2.5)
    # It is in the signal: delineated beat by beat with little noise, it shows.
    clean, rp2 = _train([RR] * 300, noise=0.0005,
                        beat_waves={i: _waves(pr=60.0, p_amp=0.1) for i in range(150, 158)})
    one = _measure(clean, rp2, 1)
    assert one.loc[151:156, "PR_ms"].median() == pytest.approx(60.0, abs=3.0)


def test_an_episode_longer_than_the_window_is_tracked():
    sig, rp = _pr_episode(100, 80, 60.0)
    df = _measure(sig, rp, 32)
    assert df.loc[118:165, "PR_ms"].median() == pytest.approx(60.0, abs=3.0)
    assert df.loc[20:70, "PR_ms"].median() == pytest.approx(PR_TRUE, abs=2.5)
    assert df.loc[210:280, "PR_ms"].median() == pytest.approx(PR_TRUE, abs=2.5)


# ── QTc ────────────────────────────────────────────────────────────────────

def test_qtc_pairs_the_median_qt_with_the_ensemble_mean_rr():
    # Alternating RR: the row's own RR swings, the beats of its median do not on average.
    rr = [RR + (12.0 if k % 2 else -12.0) for k in range(200)]
    sig, rp = _train(rr)
    df = _measure(sig, rp, 32)
    mid = df.iloc[40:160]
    assert mid["RR_ms"].std() > 10.0            # still the row's own preceding RR
    assert mid["RR_ensemble_ms"].std() < 1.0
    assert mid["RR_ensemble_ms"].mean() == pytest.approx(RR, abs=1.0)
    want = analysis.compute_qtc(df["QT_ms"].to_numpy(), df["RR_ensemble_ms"].to_numpy(), "mitchell")
    assert np.allclose(df["QTc_ms"], want, equal_nan=True)
    own_rr = analysis.compute_qtc(mid["QT_ms"].to_numpy(), mid["RR_ms"].to_numpy(), "mitchell")
    assert mid["QTc_ms"].std() < 0.5 * np.nanstd(own_rr)
    assert analysis.qtc_rr_column(df) == "RR_ensemble_ms"


def test_recomputing_qtc_gives_what_a_run_with_that_formula_gives():
    sig, rp = _train([RR] * 80)
    df = _measure(sig, rp, 16)
    again = _measure(sig, rp, 16, qtc_formula="bazett")
    pd.testing.assert_series_equal(analysis.recompute_qtc_column(df, "bazett")["QTc_ms"],
                                   again["QTc_ms"])


# ── What is said about it ──────────────────────────────────────────────────

def test_disclosure_states_n_window_exclusions_pairing_and_limit():
    sig, rp = _train([RR] * 120)
    df = _measure(_with_artifacts(sig, rp, [50, 60]), rp, 32)
    rows = dict(analysis.interval_disclosure_rows(df))
    text = rows["Delineation"]
    assert "rolling median of 32" in text and "5.4 s" in text
    assert "2 artifact" in text and "ensemble" in text
    assert "shock or stimulus response shorter" in text
    assert "mean preceding RR" in text and "AV-delay" in text
    assert rows["QT variability"].startswith("not reported")


def test_qtc_definition_and_formula_name_the_ensemble_rr_under_smoothing():
    sig, rp = _train([RR] * 120)
    df, single = _measure(sig, rp, 32), _measure(sig, rp, 1)
    rows = dict(analysis.interval_disclosure_rows(df))
    assert "mean preceding RR of the beats in the row's median" in rows["Interval definitions"]
    assert "previous R → this R" not in rows["Interval definitions"]
    assert "mean preceding RR" in analysis.qtc_formula_text("mitchell", df)
    prov = dict(analysis.analysis_provenance_rows({"intervals": df}, qtc_formula="mitchell"))
    assert "mean preceding RR" in prov["QTc formula"]
    # Per beat, as before.
    assert analysis.interval_definitions_text(single) == analysis.INTERVAL_DEFINITIONS_TEXT
    assert analysis.qtc_formula_text("mitchell", single) == analysis.qtc_formula_text("mitchell")
    assert "with the preceding RR" in analysis.qtc_formula_text("mitchell")


def test_an_n_too_large_for_the_beats_is_labelled_per_beat():
    sig, rp = _train([RR] * 60)
    df = _measure(sig, rp, 256)
    assert df.attrs["ensemble"]["n_ensemble"] == 0
    assert analysis.ensemble_info(df) is None
    assert analysis.ensemble_beats_used(df) == 1
    assert analysis.ensemble_short_text(df) == ""
    rows = dict(analysis.interval_disclosure_rows(df))
    assert rows["Delineation"].startswith("per beat") and "256" in rows["Delineation"]
    assert "IQR" in rows["QT variability"]
    assert rows["Interval definitions"] == analysis.INTERVAL_DEFINITIONS_TEXT
    assert "every row is its own beat" in analysis.ensemble_hint_text(256, RR, df)


def test_per_beat_and_old_frames_are_labelled_per_beat():
    sig, rp = _train([RR] * 30)
    df = _measure(sig, rp, 1)
    rows = dict(analysis.interval_disclosure_rows(df))
    assert rows["Delineation"].startswith("per beat")
    assert "IQR" in rows["QT variability"]
    old = df.copy()
    old.attrs = {k: v for k, v in df.attrs.items() if k != "ensemble"}
    assert analysis.ensemble_beats_used(old) == 1
    text = dict(analysis.interval_disclosure_rows(old))["Delineation"]
    assert text.startswith("per beat") and "before the rolling median" in text


def test_shared_table_withholds_the_beat_to_beat_spread_under_smoothing():
    sig, rp = _train([RR] * 80)
    smoothed = MetricTable(interval_rows(_measure(sig, rp, 16)))
    assert not smoothed.has("intervals.QT_iqr")
    assert "across rows" in smoothed.row("intervals.QT_ms.sd").label
    assert "rolling median of 16" in smoothed.row("intervals.QT_ms").basis_text
    single = MetricTable(interval_rows(_measure(sig, rp, 1)))
    assert single.has("intervals.QT_iqr")
    assert "across beats" in single.row("intervals.QT_ms.sd").label


def test_record_survives_a_session_round_trip():
    sig, rp = _train([RR] * 60)
    df = _measure(sig, rp, 16)
    saved = json.loads(json.dumps(_serialise_results({"intervals": df})))
    back = _deserialise_results(saved)["intervals"]
    assert analysis.ensemble_beats_used(back) == 16
    assert back.attrs["ensemble"]["n_rows"] == len(df)
    assert analysis.qtc_rr_column(back) == "RR_ensemble_ms"


@pytest.mark.parametrize("given, want", [(0, 1), (1, 1), (32, 32), ("16", 16), (-3, 1),
                                         ("x", 1), (None, 1), (1e9, analysis.ENSEMBLE_BEATS_MAX)])
def test_normalize_ensemble_beats(given, want):
    assert analysis.normalize_ensemble_beats(given) == want


def test_av_delay_run_must_outlast_one_ensemble():
    # Rows of a rolling median share their beats: one error reads as a run.
    rp = (np.arange(200) * 340 + 2000).astype(int)
    pr = np.full(200, 45.0)
    pr[50:66] = 60.0                    # 16 rows
    pr[100:140] = 60.0                  # 40 rows
    runs = lambda **k: [e.n_beats for e in classify_arrhythmias(rp, FS, pr_ms=pr, **k)
                        if e.kind == "av_delay"]
    assert runs() == [16, 40]
    assert runs(min_av_beats=32) == [40]
