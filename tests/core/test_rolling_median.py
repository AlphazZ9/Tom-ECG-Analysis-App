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
from ecg.core import wave_template
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


# ── Biphasic / notched P (the 247 Footshocks morphology) ───────────────────

P_UP = -54.5     # where the P comes back up through the baseline after its trough


def _notched_p_waves(shoulder=0.02, notch=0.013, notch_ms=2.0):
    """Trough -0.038 first, then a shoulder (0.02) and a notch before the peak
    (0.047), then a negative tail: amplitudes and noise (NOISE_247) relative to
    R as on the real 247 recording."""
    keep = [w for w in _waves() if not -60.0 <= w[0] <= -30.0]
    p = [(P_UP - 4.5, 0.0), (P_UP - 2.0, -0.038), (P_UP, 0.0), (P_UP + 1.0, shoulder),
         (P_UP + notch_ms, notch), (P_UP + 4.0, 0.047), (P_UP + 6.0, 0.0),
         (P_UP + 7.5, -0.025), (P_UP + 9.5, 0.0)]
    return sorted(keep + p)


NOISE_247 = 0.0065


def _beat(waves):
    t = np.arange(-170, 170) / FS * 1000.0
    xs, ys = zip(*waves)
    return np.interp(t, xs, ys), t


def test_notched_p_onset_is_the_up_crossing_after_its_trough():
    beat, t = _beat(_notched_p_waves())
    # The leading-edge line breaks on the shoulder: no P onset at all.
    assert np.isnan(analysis.detect_waves_on_beat(beat, t, _confirmed())["P_onset"])
    det = analysis.detect_waves_on_beat(beat, t, _confirmed(), p_onset_biphasic=True)
    assert det["P_onset"] == pytest.approx(P_UP, abs=0.5)


# A shoulder that never gets back to the baseline has its up-crossing on the
# notch upstroke itself (no shoulder and a notch below the baseline): not here.
@pytest.mark.parametrize("shoulder, notch",
                         [(0.004, 0.0), (0.002, 0.0), (0.0, 0.0), (0.004, -0.01), (0.002, -0.01)])
def test_a_notched_p_with_a_small_shoulder_still_starts_at_the_up_crossing(shoulder, notch):
    # 247 at t = 70 s: the shoulder after the up-crossing hardly rises above
    # the baseline for 3 ms before the notch upstroke, which the line fit lands on
    # ~4 ms late. The onset must not depend on the shoulder's size (rows
    # flipping by 4 ms between successive medians).
    beat, t = _beat(_notched_p_waves(shoulder=shoulder, notch=notch, notch_ms=3.0))
    plain = analysis.detect_waves_on_beat(beat, t, _confirmed())["P_onset"]
    det = analysis.detect_waves_on_beat(beat, t, _confirmed(), p_onset_biphasic=True)
    assert not plain < P_UP + 1.5                    # late on the notch, or NaN
    assert det["P_onset"] == pytest.approx(P_UP, abs=0.5)


def test_n32_measures_every_pr_of_a_notched_p_at_the_up_crossing():
    waves = _notched_p_waves()
    sig, rp = _train([RR] * 200, noise=NOISE_247, beat_waves={i: waves for i in range(201)})
    single, df = _measure(sig, rp, 1), _measure(sig, rp, 32)
    assert single["PR_ms"].notna().mean() < 0.8      # line fit, beat by beat
    assert df["PR_ms"].notna().mean() > 0.99
    assert df["P_onset_ms"].median() == pytest.approx(P_UP, abs=0.5)
    assert float(df["P_onset_ms"].quantile(0.95) - df["P_onset_ms"].quantile(0.05)) < 0.5
    assert df["PR_ms"].median() == pytest.approx(QRS_ON - P_UP, abs=1.0)


def test_rows_measured_on_their_own_beat_keep_the_leading_edge_line(monkeypatch):
    # A premature beat and the one after it are timing outliers: measured on
    # themselves with the N = 1 rules, not with the biphasic ones.
    rr = [RR] * 100 + [100.0, 240.0] + [RR] * 100
    waves = _notched_p_waves()
    sig, rp = _train(rr, noise=NOISE_247, beat_waves={i: waves for i in range(len(rr) + 1)})
    real = analysis.detect_waves_on_beat
    calls = []

    def spy(beat, bt, *a, **k):
        # Not a phase-plan or an initial-wave-plan median.
        if "t_plan_out" not in k and not k.get("qrs_initial_wave"):
            calls.append((beat.copy(), bt.copy(), a, dict(k)))
        return real(beat, bt, *a, **k)

    monkeypatch.setattr(analysis, "detect_waves_on_beat", spy)
    df = _measure(sig, rp, 32)
    monkeypatch.undo()
    single = _measure(sig, rp, 1)
    own = df.index[df["ensemble_n"] == 1]
    is_own = df.index.isin(own)
    assert len(own) == 2 and len(calls) == len(df)
    for (beat, bt, a, k), o in zip(calls, is_own):
        rules = (k.get("p_onset_biphasic", False), k.get("t_phase_noise"), k.get("t_plan"),
                 k.get("qrs_initial_plan"))
        assert ((rules == (False, None, None, None)) if o
                else (rules[0] and rules[1] is not None and rules[3] is not None))
    # The N = 1 rules on the row's own beat. The P search starts after the
    # previous row's T peak, a median's on row 101 (not the N = 1 run's).
    got = [real(b, t, *a, p_search_from_ms=k.get("p_search_from_ms"),
                next_r_ms=k.get("next_r_ms"))
           for (b, t, a, k), o in zip(calls, is_own) if o]
    np.testing.assert_allclose(df.loc[own, "P_onset_ms"].to_numpy(),
                               [d["P_onset"] for d in got], equal_nan=True)
    # And the Q / R edge for QRS onset, not the initial-wave lead.
    np.testing.assert_allclose(df.loc[own, "QRS_onset_ms"].to_numpy(),
                               [d["QRS_onset"] for d in got], equal_nan=True)
    # Row 102 follows an own-beat row in both runs: the N = 1 value.
    pd.testing.assert_series_equal(df.loc[own[1:], "P_onset_ms"], single.loc[own[1:], "P_onset_ms"])


@pytest.mark.parametrize("gap_ms", [3.0, 5.0])
def test_a_t_tail_then_a_flat_stretch_is_not_a_biphasic_p(gap_ms):
    # Monophasic P from -50 ms after a negative T tail that ends gap_ms earlier:
    # the up-crossing ending that tail is not the P onset.
    keep = [w for w in _waves() if not -60.0 <= w[0] <= -30.0]
    p_on = -50.0
    tail = [(p_on - gap_ms - 10.0, 0.0), (p_on - gap_ms - 5.0, -0.03), (p_on - gap_ms, 0.0)]
    beat, t = _beat(sorted(keep + tail + [(p_on, 0.0), (p_on + 8.0, 0.03), (p_on + 16.0, 0.0)]))
    plain = analysis.detect_waves_on_beat(beat, t, _confirmed())["P_onset"]
    biph = analysis.detect_waves_on_beat(beat, t, _confirmed(), p_onset_biphasic=True)["P_onset"]
    assert biph == plain == pytest.approx(p_on, abs=0.5)


# ── QRS onset: a small initial wave before Q ──────────────────────────────

QRS_INIT_ON, Q_ON = -9.0, -5.5


def _initial_wave_waves(amp, p_end=None):
    """A positive wave of *amp* (x R) from -9 ms that runs straight into a Q
    from -5.5 ms, as on 248 (0.18) and 247 (0.03); *p_end*: a taller P ending
    there instead, no initial wave."""
    keep = [w for w in _waves() if not -9.5 <= w[0] <= -0.5]
    qrs = [(Q_ON, 0.0), (-2.5, -0.3)]
    if amp:
        qrs = [(QRS_INIT_ON, 0.0), (-7.0, amp)] + qrs
    if p_end is not None:
        keep = [w for w in keep if not -60.0 <= w[0] <= -30.0]
        qrs = [(p_end - 16.0, 0.0), (p_end - 8.0, 0.1), (p_end, 0.0)] + qrs
    return sorted(keep + qrs)


@pytest.mark.parametrize("amp", [0.03, 0.18])
def test_a_small_initial_wave_starts_the_qrs(amp):
    beat, t = _beat(_initial_wave_waves(amp))
    plain = analysis.detect_waves_on_beat(beat, t, _confirmed())
    init = analysis.detect_waves_on_beat(beat, t, _confirmed(), qrs_initial_wave=True)
    assert plain["QRS_onset"] == pytest.approx(Q_ON, abs=0.5)        # the Q edge
    assert init["QRS_onset"] == pytest.approx(QRS_INIT_ON, abs=0.3)
    assert {k: v for k, v in init.items() if k != "QRS_onset"} ==         {k: v for k, v in plain.items() if k != "QRS_onset"}


@pytest.mark.parametrize("amp, noise", [(0.03, NOISE_247), (0.18, NOISE)])
def test_rolling_median_rows_start_the_qrs_with_the_initial_wave(amp, noise):
    waves = _initial_wave_waves(amp)
    sig, rp = _train([RR] * 200, noise=noise, beat_waves={i: waves for i in range(201)})
    df, single = _measure(sig, rp, 32), _measure(sig, rp, 1)
    on = df.loc[df["ensemble_n"] > 1, "QRS_onset_ms"]
    assert on.notna().all() and on.median() == pytest.approx(QRS_INIT_ON, abs=0.5)
    assert float(on.max() - on.min()) < 1.0                  # on every row
    assert df["QRS_ms"].median() == pytest.approx(QRS_OFF - QRS_INIT_ON, abs=1.0)
    # Per beat: the Q edge, as before.
    assert single["QRS_onset_ms"].median() == pytest.approx(Q_ON, abs=0.7)


@pytest.mark.parametrize("noise", [NOISE, 2 * NOISE])
def test_noise_is_not_an_initial_wave(monkeypatch, noise):
    sig, rp = _train([RR] * 300, noise=noise)
    df = _measure(sig, rp, 32)
    assert df.attrs["ensemble"]["qrs_plans"]["rows"]["initial_wave"] == 0
    off = _without_initial_wave(monkeypatch, sig, rp, 32)
    assert df["QRS_onset_ms"].notna().mean() > 0.95
    pd.testing.assert_series_equal(df["QRS_onset_ms"], off["QRS_onset_ms"])


def _without_initial_wave(monkeypatch, sig, rp, n):
    """The frame with every row's QRS onset at the Q / R edge."""
    real = analysis.detect_waves_on_beat
    with monkeypatch.context() as m:
        m.setattr(analysis, "detect_waves_on_beat",
                  lambda *a, **k: real(*a, **{**k, "qrs_initial_plan": None}))
        return _measure(sig, rp, n)


@pytest.mark.parametrize("n, noise", [(32, NOISE), (32, 1.5 * NOISE), (8, NOISE)])
def test_an_initial_wave_at_several_times_the_noise_counts_on_every_row(n, noise):
    # A 3 %-of-R wave at ~2-3 x the 248 noise (and at N = 8): on each row's own
    # median it counted on some rows only (QRS onset -8.5 or -5.2 ms in blocks
    # of rows), its PR window ending in the wave's foot raising the threshold.
    waves = _initial_wave_waves(0.03)
    sig, rp = _train([RR] * 300, noise=noise, beat_waves={i: waves for i in range(301)})
    df = _measure(sig, rp, n)
    on = df.loc[df["ensemble_n"] > 1, "QRS_onset_ms"]
    assert on.notna().all() and on.median() == pytest.approx(QRS_INIT_ON, abs=0.5)
    assert float(on.quantile(0.99) - on.quantile(0.01)) < 1.5
    assert float(on.max()) < Q_ON - 2.0                      # no row at the Q edge
    assert df.attrs["ensemble"]["qrs_plans"]["rows"]["edge"] == 0


def _terminal_negative_p_waves(amp=0.0, ramp=0.002):
    """A biphasic P whose negative phase (-0.016 x R, flat from -26 to -20 ms)
    is flatter than the PR segment after it (a *ramp* of 0.2 % of R): the PR
    window lands in that phase, 1.6 % of R under the PR segment. *amp*: an
    initial wave from -9 ms before the Q, as in _initial_wave_waves."""
    keep = [w for w in _waves() if not -60.0 <= w[0] <= -0.5]
    p = [(-40.0, 0.0), (-35.0, 0.04), (-30.0, 0.0), (-26.0, -0.016), (-20.0, -0.016),
         (-14.0, 0.0)]
    qrs = [(QRS_INIT_ON, ramp), (-7.0, ramp + amp), (Q_ON, 0.0), (-2.5, -0.3)]
    return sorted(keep + p + qrs)


def test_a_pr_window_in_a_biphasic_p_does_not_make_the_pr_segment_an_initial_wave():
    beat, t = _beat(_terminal_negative_p_waves())
    lo, hi = wave_template.pr_baseline_window(t, beat, None, -2.5)
    assert hi < -14.0                                # in the P's negative phase
    plain = analysis.detect_waves_on_beat(beat, t, _confirmed())
    out = {}
    init = analysis.detect_waves_on_beat(beat, t, _confirmed(), qrs_initial_wave=True,
                                         qrs_initial_out=out)
    assert init["QRS_onset"] == plain["QRS_onset"]
    assert out == {"found": False, "from_q": True, "lead": 0.0}
    # A real initial wave there is measured from the level it rises from.
    beat, t = _beat(_terminal_negative_p_waves(amp=0.03))
    out = {}
    init = analysis.detect_waves_on_beat(beat, t, _confirmed(), qrs_initial_wave=True,
                                         qrs_initial_out=out)
    assert out["found"] and init["QRS_onset"] == pytest.approx(QRS_INIT_ON, abs=0.5)


@pytest.mark.parametrize("amp", [0.0, 0.03])
def test_rolling_median_rows_after_a_biphasic_p_start_the_qrs_where_it_starts(monkeypatch, amp):
    waves = _terminal_negative_p_waves(amp)
    sig, rp = _train([RR] * 300, noise=NOISE_247, beat_waves={i: waves for i in range(301)})
    df = _measure(sig, rp, 32)
    on = df.loc[df["ensemble_n"] > 1, "QRS_onset_ms"]
    if amp:
        assert on.notna().all() and on.median() == pytest.approx(QRS_INIT_ON, abs=0.5)
        assert float(on.quantile(0.99) - on.quantile(0.01)) < 1.5
    else:
        # Without a wave no row moves (it moved by 12-18 ms on 2.5 % of them).
        pd.testing.assert_series_equal(
            df["QRS_onset_ms"], _without_initial_wave(monkeypatch, sig, rp, 32)["QRS_onset_ms"])


@pytest.mark.parametrize("p_end", [-6.5, -7.5])
def test_the_end_of_a_p_running_into_the_qrs_is_not_an_initial_wave(p_end):
    beat, t = _beat(_initial_wave_waves(0.0, p_end=p_end))
    plain = analysis.detect_waves_on_beat(beat, t, _confirmed())["QRS_onset"]
    init = analysis.detect_waves_on_beat(beat, t, _confirmed(), qrs_initial_wave=True)["QRS_onset"]
    assert init == plain


# ── Biphasic T: the T end is that of the last phase on every row ──────────
# Low biphasic T as on the real 248 recording: +0.025 then -0.015 of R, the
# second phase 60 % of the first, at about the 248 noise of a 32-beat median.
# The T end of the line definition is 45 ms (the last phase's straight
# trailing edge reaching the baseline); the first phase ends at 36 ms.
T_LAST_END = 45.0


def _biphasic_t_waves(second=-0.015, shift=0.0, first=0.025):
    keep = [w for w in _waves() if not 20.0 <= w[0] <= 60.0]
    t = [(24.0, 0.0), (30.0, first), (36.0, 0.0)]
    if second:
        t += [(40.0 + shift, second), (T_LAST_END + shift, 0.0)]
    return sorted(keep + t)


def _t_train(n_beats=200, waves_of=lambda i: _biphasic_t_waves(), noise=NOISE):
    return _train([RR] * n_beats, noise=noise,
                  beat_waves={i: waves_of(i) for i in range(n_beats + 1)})


def _plans(df):
    return df.attrs["ensemble"]["t_plans"]["rows"]


def test_a_low_biphasic_t_ends_with_its_last_phase_on_every_row():
    # Each row deciding its phases on its own median, the second phase came
    # and went with the noise: T end jumping between ~36 and ~45 ms in blocks
    # of rows. The phases are now those of the median of the 8 N beats around.
    df = _measure(*_t_train(), 32)
    info = df.attrs["ensemble"]["t_plans"]
    assert info["rows"]["biphasic"] == len(df) and info["biphasic_shape"] == "+ then −"
    assert info["end_ms_median"] == pytest.approx(T_LAST_END, abs=0.7)
    to = df["T_offset_ms"]
    assert to.notna().mean() > 0.95
    assert to.quantile(0.01) > T_LAST_END - 1.5 and to.quantile(0.99) < T_LAST_END + 1.5
    assert df["T_peak_ms"].median() == pytest.approx(30.0, abs=1.0)     # the first phase
    qt = df["QT_ms"]
    assert qt.median() == pytest.approx(T_LAST_END - QRS_ON, abs=1.0)
    assert float(qt.quantile(0.75) - qt.quantile(0.25)) < 1.0


def test_the_t_end_follows_its_last_phase_when_it_moves():
    # The last phase 3 ms later from beat 100 of 200: every phase-plan median
    # (256 beats) blends both edges, and the rows read each side against
    # it -- the step under-read (~2 of its 3 ms), never reversed. Longer
    # stretches of each are read in full (test_a_qt_step_is_read_on_both_sides).
    df = _measure(*_t_train(waves_of=lambda i: _biphasic_t_waves(shift=3.0 if i >= 100 else 0.0)), 32)
    to = df["T_offset_ms"].to_numpy()
    before, after = np.nanmedian(to[20:70]), np.nanmedian(to[130:])
    assert before == pytest.approx(T_LAST_END, abs=1.5)
    assert after == pytest.approx(T_LAST_END + 3.0, abs=1.0)
    assert 1.5 < after - before < 3.5


def test_a_second_t_phase_of_15_to_30_percent_leaves_no_t_end():
    # 20 % of the first phase: is the T end that of the first phase (36) or
    # of the second (45)? Neither, on every row.
    df = _measure(*_t_train(waves_of=lambda i: _biphasic_t_waves(second=-0.005)), 32)
    assert _plans(df)["biphasic"] == _plans(df)["monophasic"] == 0
    on_median = df["ensemble_n"].to_numpy() > 1
    assert df.loc[on_median, "T_offset_ms"].isna().all()
    assert "ambiguous" in analysis.ensemble_disclosure(df)


def test_the_phase_plan_is_read_on_each_beat():
    beat, t = _beat(_biphasic_t_waves())
    kw = dict(p_onset_biphasic=True, t_phase_noise=1e-4)
    plan: dict = {}
    own = analysis.detect_waves_on_beat(beat, t, _confirmed(), t_plan_out=plan, **kw)
    assert plan["phases"] == 2 and own["T_offset"] == pytest.approx(T_LAST_END, abs=0.7)
    same = analysis.detect_waves_on_beat(beat, t, _confirmed(), t_plan=plan, **kw)
    assert same["T_offset"] == pytest.approx(own["T_offset"], abs=1e-6)
    # The whole T 2 ms later: read in full; its last phase alone 2 ms wider
    # (peak and end later, start fixed): read at ~3/4.
    later, _ = _beat([(x + 2.0 if 20.0 <= x <= 60.0 else x, y) for x, y in _biphasic_t_waves()])
    got = analysis.detect_waves_on_beat(later, t, _confirmed(), t_plan=plan, **kw)
    assert got["T_offset"] - own["T_offset"] == pytest.approx(2.0, abs=0.3)
    wider, _ = _beat(_biphasic_t_waves(shift=2.0))
    got = analysis.detect_waves_on_beat(wider, t, _confirmed(), t_plan=plan, **kw)
    assert 1.2 <= got["T_offset"] - own["T_offset"] <= 2.0
    # The whole T 12 ms later, beyond the plan's own edge (NaN before).
    far, _ = _beat([(x + 12.0 if 20.0 <= x <= 60.0 else x, y) for x, y in _biphasic_t_waves()])
    got = analysis.detect_waves_on_beat(far, t, _confirmed(), t_plan=plan, **kw)
    assert got["T_offset"] - own["T_offset"] == pytest.approx(12.0, abs=0.3)
    # No last phase on this beat: no T end -- not the first phase's end.
    mono, _ = _beat(_biphasic_t_waves(second=0.0))
    gone = analysis.detect_waves_on_beat(mono, t, _confirmed(), t_plan=plan, **kw)
    assert np.isnan(gone["T_offset"]) and gone["T_peak"] == pytest.approx(30.0, abs=0.5)
    assert analysis.detect_waves_on_beat(mono, t, _confirmed())["T_offset"] < 38.0


# A QT change during the recording, on 600 beats (100 s), the 8 N = 256-beat
# phase-plan median sliding along: the T end is read on both sides of it
# (with one plan for the whole recording, an episode or the half before a
# step had no T end at all, and a +20 ms step turned the plan monophasic).
def _qt_change_train(shift_of, waves=_biphasic_t_waves, n_beats=600, noise=NOISE, seed=0):
    return _train([RR] * n_beats, noise=noise, seed=seed,
                  beat_waves={i: waves(shift=shift_of(i)) for i in range(n_beats + 1)})


@pytest.mark.parametrize("shift", [12.0, 20.0])
def test_a_qt_step_is_read_on_both_sides(shift):
    df = _measure(*_qt_change_train(lambda i: shift if i >= 300 else 0.0), 32)
    to = df["T_offset_ms"].to_numpy()
    before, after = to[30:240], to[360:570]
    for part, end in ((before, T_LAST_END), (after, T_LAST_END + shift)):
        assert np.isfinite(part).mean() > 0.95
        assert np.nanmedian(part) == pytest.approx(end, abs=1.0)
    assert _plans(df)["monophasic"] == 0


@pytest.mark.parametrize("first, last", [(150, 450), (240, 360)])
def test_a_qt_prolongation_episode_is_tracked_not_deleted(first, last):
    # 300 beats (50 s, longer than the plan's median) and 120 (20 s, shorter:
    # those medians blend the episode, and the rows inside it read it up to
    # ~3 ms long). Never a T end longer than the episode's outside it: the
    # median of a blend has two edges and no T end (the line through them
    # ended 10-25 ms late, normal rows next to the episode reading 60-70 ms).
    df = _measure(*_qt_change_train(lambda i: 12.0 if first <= i < last else 0.0), 32)
    to = df["T_offset_ms"].to_numpy()
    inside = to[first + 25:last - 25]
    assert np.isfinite(inside).mean() > 0.95 and np.isfinite(to).mean() > 0.9
    assert T_LAST_END + 11.0 < np.nanmedian(inside) < T_LAST_END + 15.5
    for a, b in ((30, first - 60), (last + 60, 570)):
        assert np.nanmedian(to[a:b]) == pytest.approx(T_LAST_END, abs=1.0)
    out = np.r_[to[:first - 16], to[last + 16:]]       # no episode beat in the row's
    assert np.nanpercentile(out, 99) < T_LAST_END + 6.0


def _mono_t_waves(shift=0.0):
    # Monophasic T, its straight trailing edge reaching the baseline at 36 ms.
    keep = [w for w in _waves() if not 20.0 <= w[0] <= 60.0]
    return sorted(keep + [(24.0 + shift, 0.0), (30.0 + shift, 0.04), (36.0 + shift, 0.0)])


def test_a_monophasic_t_step_gets_no_constant_offset():
    # A one-phase plan read with the plan's slope put the baseline half of a
    # QT step 7 ms late and under-read the step.
    df = _measure(*_qt_change_train(lambda i: 12.0 if i >= 300 else 0.0, waves=_mono_t_waves), 32)
    to = df["T_offset_ms"].to_numpy()
    assert np.nanmedian(to[30:240]) == pytest.approx(36.0, abs=0.7)
    assert np.nanmedian(to[360:570]) == pytest.approx(48.0, abs=0.7)


def test_a_rows_t_end_does_not_depend_on_beats_minutes_away():
    # The same first 300 beats, alone or followed by a QT 12 ms longer.
    sig, rp = _qt_change_train(lambda i: 12.0 if i >= 300 else 0.0)
    full = _measure(sig, rp, 32).set_index("R_peak_s")["T_offset_ms"]
    cut = int(rp[300]) - int(0.085 * FS)
    first = rp[rp < cut - int(0.1 * FS)]
    half = _measure(sig[:cut], first, 32).set_index("R_peak_s")["T_offset_ms"]
    early = half.index[half.index < rp[170] / FS]
    a, b = full.reindex(early), half.reindex(early)
    assert a.notna().sum() == b.notna().sum() > 0.95 * len(early)
    assert float((a - b).abs().max()) < 0.3


def test_a_t_that_turns_biphasic_keeps_the_t_end_of_its_monophasic_half():
    waves = {i: (_mono_t_waves() if i < 300 else _biphasic_t_waves()) for i in range(601)}
    df = _measure(*_train([RR] * 600, beat_waves=waves), 32)
    to = df["T_offset_ms"].to_numpy()
    # (Within 8 phase plans of the change, the phases are not decided.)
    assert np.isfinite(to[30:150]).mean() > 0.95 and np.isfinite(to[450:570]).mean() > 0.95
    assert np.nanmedian(to[30:150]) == pytest.approx(36.0, abs=0.7)
    assert np.nanmedian(to[450:570]) == pytest.approx(T_LAST_END, abs=0.7)


def test_a_low_biphasic_t_keeps_its_t_end_under_more_noise():
    # 1.7 x the noise: the row's last phase alone is hardly told from noise,
    # its T end is read at the lag of the whole T (no rows lost).
    df = _measure(*_t_train(n_beats=400, noise=1.7 * NOISE), 32)
    to = df["T_offset_ms"]
    assert to.notna().mean() > 0.95
    assert to.median() == pytest.approx(T_LAST_END, abs=0.7)
    assert float(to.quantile(0.9) - to.quantile(0.1)) < 4.0


def test_an_inverted_t_after_a_deep_st_dip_is_biphasic():
    # - then + T right after a negative ST dip with no return to the baseline
    # in between: the dip and the first phase share a sign run, and the second
    # phase was compared with the dip (ambiguous whenever the dip was deep).
    keep = [w for w in _waves() if not 12.0 <= w[0] <= 60.0]
    for dip in (-0.10, -0.18, -0.25):
        waves = sorted(keep + [(13.0, 0.0), (17.0, dip), (24.0, -0.02), (30.0, -0.09),
                               (36.0, 0.0), (40.0, 0.055), (45.0, 0.0)])
        beat, t = _beat(waves)
        plan: dict = {}
        det = analysis.detect_waves_on_beat(beat, t, _confirmed(), t_plan_out=plan,
                                            p_onset_biphasic=True, t_phase_noise=1e-4)
        assert plan.get("phases") == 2 and plan["first_sign"] == -1, dip
        assert det["T_offset"] == pytest.approx(45.0, abs=0.7), dip


def test_the_t_end_level_does_not_jump_with_the_pr_stretch_chosen():
    # A PR segment with two flat stretches 0.03 apart (247): pr_baseline_window
    # takes the latest one about as flat as the flattest, and a little more
    # ripple on the late one made it jump to the early one (T end 1.7 ms
    # away). The flatness-weighted level moves by small steps.
    t = np.arange(-170, 170) / FS * 1000.0
    ripple = np.random.default_rng(3).normal(0.0, 1.0, len(t))
    pr = (t > -32) & (t < -9)
    chosen, levels = [], []
    for a in np.linspace(0.0, 0.004, 21):
        beat = np.interp(t, *zip(*_waves()))
        beat[pr] += np.interp(t[pr], [-32, -20, -16, -9], [0.02, 0.02, -0.01, -0.01])
        beat[pr] += (0.001 + a * (t[pr] > -16)) * ripple[pr]
        lo, hi = wave_template.pr_baseline_window(t, beat, None, -5.0)
        chosen.append(float(np.median(beat[(t >= lo) & (t <= hi)])))
        levels.append(wave_template.pr_baseline_level(t, beat, -5.0))
    assert np.max(np.abs(np.diff(chosen))) > 0.025   # one step: the other stretch
    assert np.max(np.abs(np.diff(levels))) < 0.005


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
    assert "biphasic P" in text and "up through the PR baseline" in text
    assert "QRS onset is the start of the first QRS deflection" in text
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
