"""
Standard ECG interval definitions on synthetic beats whose true landmarks
are known exactly (piecewise-linear waves):

    PR  = P onset   -> QRS onset
    QRS = QRS onset -> QRS offset (J point)
    QT  = QRS onset -> T end
    QTc = QT corrected with the PRECEDING RR

plus the beat-to-beat QT variability measure that replaced "QT dispersion",
the legacy-session handling, and the export disclosure rows.
"""
import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.wave_template import (
    WaveTemplate, _mouse_demo_beat, detect_waves_on_beat, edge_baseline_crossing,
    pr_baseline_window, qrs_bounds,
)
from ecg.io.export import ExcelExporter, PrismExporter

FS = 2000.0

# True landmarks of the piecewise-linear beat (ms from R).
P_ON, QRS_ON, QRS_OFF, T_END = -50.0, -8.0, 8.0, 55.0
_P   = [(-120.0, 0.0), (P_ON, 0.0), (-42.0, 0.1), (-34.0, 0.0)]
_QRS = [(QRS_ON, 0.0), (-5.0, -0.2), (0.0, 1.0), (4.0, -0.3), (QRS_OFF, 0.0),
        (12.0, 0.1), (18.0, 0.0)]
_T   = [(25.0, 0.0), (35.0, 0.2), (T_END, 0.0), (120.0, 0.0)]


@pytest.fixture(autouse=True)
def _no_template_writes(monkeypatch):
    def _refuse(self):
        raise AssertionError("WaveTemplate.save() must not run in tests")
    monkeypatch.setattr(WaveTemplate, "save", _refuse)


def _pl(t_ms, pts):
    xs, ys = zip(*pts)
    return np.interp(t_ms, xs, ys)


def _beat(qrs=_QRS):
    t = np.arange(-100.0, 100.0, 1000.0 / FS)
    return t, _pl(t, _P + qrs + _T)


def _template():
    """Confirmed (never auto-saves) template whose T-end window fits a
    55 ms T end inside the ±81 ms beat window used below."""
    wt = WaveTemplate()
    wt.landmarks["T_offset"] = (55.0, 15.0)
    wt.confirmed = True
    return wt


def _train(rr_ms, seed=0, noise=0.0, fs=FS, pts=None):
    """Piecewise-linear beats at the given RR intervals (ms)."""
    pts = _P + _QRS + _T if pts is None else pts
    rp = [int(0.3 * fs)]
    for rr in rr_ms:
        rp.append(rp[-1] + int(round(rr / 1000.0 * fs)))
    rp = np.array(rp)
    n = rp[-1] + int(0.3 * fs)
    t_ms = np.arange(n) / fs * 1000.0
    sig = np.zeros(n)
    for r in rp:
        lo, hi = r - int(0.12 * fs), r + int(0.12 * fs)
        sig[lo:hi] += _pl(t_ms[lo:hi] - t_ms[r], pts)
    sig += np.random.default_rng(seed).normal(0.0, noise, n)
    return sig, rp


# ════════════════════════════════════════════════════════════
#  Landmarks
# ════════════════════════════════════════════════════════════

def test_qrs_bounds_find_the_true_onset_and_j_point():
    t, b = _beat()
    onset, offset = qrs_bounds(t, b, -5.0, 4.0, 12.0, -30.0, -22.0)
    assert onset == pytest.approx(QRS_ON, abs=0.6)
    assert offset == pytest.approx(QRS_OFF, abs=0.6)


def test_qrs_onset_without_q_wave_is_the_r_upstroke_onset():
    t, b = _beat([(QRS_ON, 0.0), (0.0, 1.0), (4.0, -0.3), (QRS_OFF, 0.0),
                  (12.0, 0.1), (18.0, 0.0)])
    onset, _ = qrs_bounds(t, b, -12.0, 4.0, 12.0, -30.0, -22.0)
    assert onset == pytest.approx(QRS_ON, abs=0.6)


def test_j_point_is_the_notch_when_the_j_wave_starts_above_baseline():
    # No S below baseline: the R downstroke turns into the J wave at +9 ms.
    t, b = _beat([(QRS_ON, 0.0), (-5.0, -0.2), (0.0, 1.0), (6.0, 0.3),
                  (9.0, 0.25), (15.0, 0.35), (22.0, 0.0)])
    _, offset = qrs_bounds(t, b, -5.0, 9.0, 15.0, -30.0, -22.0)
    assert offset == pytest.approx(9.0, abs=0.6)


def test_flat_beat_has_no_qrs_bounds():
    t = np.arange(-100.0, 100.0, 0.5)
    assert all(np.isnan(qrs_bounds(t, np.zeros_like(t), -5.0, 4.0, None, -30.0, -22.0)))


def test_detect_waves_on_beat_reports_qrs_onset_and_offset():
    t, b = _beat()
    det = detect_waves_on_beat(b, t, WaveTemplate())
    assert det["QRS_onset"] == pytest.approx(QRS_ON, abs=0.6)
    assert det["QRS_offset"] == pytest.approx(QRS_OFF, abs=0.6)
    # Q is searched before R only: it used to land on the S trough's
    # flank inside the default window and be clamped onto R.
    assert det["Q_peak"] == pytest.approx(-5.0, abs=0.6)
    assert det["S_peak"] == pytest.approx(4.0, abs=0.6)


def test_template_saved_before_qrs_landmarks_still_detects_them():
    wt = WaveTemplate()
    wt.landmarks = {k: v for k, v in wt.landmarks.items()
                    if k not in ("QRS_onset", "QRS_offset")}
    t, b = _beat()
    det = detect_waves_on_beat(b, t, wt)
    assert det["QRS_onset"] == pytest.approx(QRS_ON, abs=0.6)
    assert det["QRS_offset"] == pytest.approx(QRS_OFF, abs=0.6)


# Wide QRS: onset up to -30 ms, before the default QRS_onset window
# (-22 to -2 ms), with its Q nadir before the default Q window (-17 ms).
_P_EARLY = [(-120.0, 0.0), (-75.0, 0.0), (-65.0, 0.1), (-56.0, 0.0)]


def _wide_beat(onset):
    t = np.arange(-100.0, 100.0, 0.5)
    qrs = [(onset, 0.0), (onset + 8.0, -0.2), (0.0, 1.0), (6.0, -0.3),
           (14.0, 0.0), (18.0, 0.1), (24.0, 0.0)]
    return t, _pl(t, _P_EARLY + qrs + [(30.0, 0.0), (40.0, 0.2), (60.0, 0.0),
                                        (120.0, 0.0)])


@pytest.mark.parametrize("onset", [-22.0, -26.0, -30.0])
def test_wide_qrs_onset_is_not_clamped_to_the_default_window(onset):
    t, b = _wide_beat(onset)
    det = detect_waves_on_beat(b, t, WaveTemplate())
    assert det["QRS_onset"] == pytest.approx(onset, abs=0.6)
    assert det["QRS_offset"] == pytest.approx(14.0, abs=0.6)
    assert det["Q_peak"] == pytest.approx(onset + 8.0, abs=0.6)
    # The Q wave is not taken for a negative P wave.
    assert det["P_peak"] == pytest.approx(-65.0, abs=1.0)


def test_pr_baseline_is_the_flat_stretch_just_before_the_qrs():
    t, b = _wide_beat(-30.0)
    lo, hi = pr_baseline_window(t, b, -65.0, -22.0)
    assert lo >= -56.0 and hi <= -30.0          # between P end and QRS onset
    assert hi == pytest.approx(-30.0, abs=0.6)  # the latest such stretch


def test_t_end_before_the_default_window_is_not_clamped_onto_it():
    # Default T_offset window 60-104 ms: a 40 ms T end used to read 60 ms.
    t, _ = _beat()
    b = _pl(t, _P + _QRS + [(20.0, 0.0), (28.0, 0.2), (40.0, 0.0), (120.0, 0.0)])
    det = detect_waves_on_beat(b, t, WaveTemplate())
    assert det["T_offset"] == pytest.approx(40.0, abs=0.6)


def test_t_end_past_the_beat_window_is_nan_not_the_edge():
    t = np.arange(-60.0, 40.0, 0.5)            # beat window ends at +40 ms
    b = _pl(t, _P + _QRS + [(20.0, 0.0), (30.0, 0.2), (60.0, 0.0), (120.0, 0.0)])
    det = detect_waves_on_beat(b, t, WaveTemplate())
    assert np.isnan(det["T_offset"])


def test_edge_line_fit_is_exact_on_a_linear_edge_and_ignores_later_waves():
    t = np.arange(0.0, 40.0, 0.5)
    y = np.interp(t, [0.0, 20.0, 30.0, 34.0, 38.0], [1.0, 0.0, 0.0, 0.8, 0.0])
    # Trailing edge from the peak at 0 down to baseline at 20; the bump at
    # 34 ms also crosses 30-70 % but lies past the edge's return to baseline.
    assert edge_baseline_crossing(t, y, 0.0, 1.0, 0.5) == pytest.approx(20.0, abs=1e-6)


def test_two_recordings_on_one_unconfirmed_template_are_independent(monkeypatch):
    """Delineating recording A neither modifies nor saves the template it
    is given (older versions narrowed it to A and re-saved it globally):
    B gives exactly the same PR / QRS / QT whether it is analysed after A
    or on a fresh template."""
    saved = []
    monkeypatch.setattr(WaveTemplate, "save", lambda self, path=None: saved.append(1))
    p = [(-120.0, 0.0), (-60.0, 0.0), (-52.0, 0.1), (-44.0, 0.0)]
    rec_a = p + _QRS + _T                                    # QRS 16, QT 63
    rec_b = p + [(-16.0, 0.0), (-10.0, -0.2), (0.0, 1.0), (8.0, -0.3),
                 (14.0, 0.0), (18.0, 0.1), (24.0, 0.0),
                 (30.0, 0.0), (40.0, 0.2), (65.0, 0.0), (120.0, 0.0)]

    def _run(pts, wt):
        sig, rp = _train(np.full(30, 200.0), pts=pts)
        df = analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                        wave_template=wt)
        return df[["PR_ms", "QRS_ms", "QT_ms"]].median()

    shared = WaveTemplate()
    before = shared.to_dict()
    _run(rec_a, shared)
    assert not saved and shared.to_dict() == before      # A left it alone
    after_a = _run(rec_b, shared)
    fresh   = _run(rec_b, WaveTemplate())
    np.testing.assert_array_equal(after_a.to_numpy(), fresh.to_numpy())
    np.testing.assert_allclose(fresh.to_numpy(), [44.0, 30.0, 81.0], atol=1.0)


def _p_beat(p_on, p_peak, q=-0.2):
    """Beat with its P onset / peak at the given times (ms from R) and a Q
    of depth *q* (R = 1)."""
    t = np.arange(-100.0, 100.0, 1000.0 / FS)
    p = [(-120.0, 0.0), (p_on, 0.0), (p_peak, 0.1), (p_peak + 8.0, 0.0)]
    qrs = [(QRS_ON, 0.0), (-5.0, q), (0.0, 1.0), (4.0, -0.3), (QRS_OFF, 0.0),
           (12.0, 0.1), (18.0, 0.0)]
    return t, _pl(t, p + qrs + _T)


@pytest.mark.parametrize("rise", [8.0, 12.0, 15.0])
def test_p_onset_follows_a_slow_p_upstroke_with_the_default_template(rise):
    # The default template capped P onset 9 ms before the P peak
    # (|P_peak window start - P_onset window start|): a 12-15 ms upstroke
    # gave PR 3-6 ms short.
    t, b = _p_beat(-42.0 - rise, -42.0)
    det = detect_waves_on_beat(b, t, WaveTemplate())
    assert det["P_onset"] == pytest.approx(-42.0 - rise, abs=0.6)


def test_long_pr_is_measured_in_full_or_rejected_not_truncated():
    # P onset before the default P_onset window (-82 ms): a 77 ms PR read
    # 74 ms; an 84 ms PR (beyond PR_ABS_MAX) read 74 ms and passed as valid.
    t, b = _p_beat(-85.0, -73.0)
    det = detect_waves_on_beat(b, t, WaveTemplate())
    assert det["P_onset"] == pytest.approx(-85.0, abs=0.6)
    for p_on, true_pr in ((-85.0, 77.0), (-92.0, 84.0)):
        p = [(-140.0, 0.0), (p_on, 0.0), (-73.0, 0.1), (-65.0, 0.0)]
        sig, rp = _train(np.full(30, 250.0), pts=p + _QRS + _T)
        df = analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                        wave_template=_template())
        if true_pr <= analysis.MouseECG.PR_ABS_MAX:
            assert df["PR_ms"].median() == pytest.approx(true_pr, abs=1.0)
        else:
            assert df["PR_ms"].isna().all()


def test_p_onset_before_the_beat_window_is_nan_not_the_edge():
    t = np.arange(-60.0, 100.0, 0.5)              # window starts at -60 ms
    b = _pl(t, [(-120.0, 0.0), (-70.0, 0.0), (-50.0, 0.1), (-42.0, 0.0)]
            + _QRS + _T)
    det = detect_waves_on_beat(b, t, WaveTemplate())
    assert np.isnan(det["P_onset"])


@pytest.mark.parametrize("q", [-0.08, -0.04, -0.03])
def test_small_q_wave_starts_the_qrs(q):
    # A Q shallower than 5 % of R used to be skipped: QRS onset moved to the
    # R upstroke (~3 ms late), QRS / QT shorter and PR longer.
    t, b = _p_beat(P_ON, -42.0, q=q)
    onset, _ = qrs_bounds(t, b, -5.0, 4.0, 12.0, -30.0, -22.0)
    assert onset == pytest.approx(QRS_ON, abs=0.6)
    assert detect_waves_on_beat(b, t, WaveTemplate())["QRS_onset"] == pytest.approx(
        QRS_ON, abs=0.6)


def test_one_premature_beat_does_not_change_the_other_beats():
    # The beat window was sized from the recording's shortest RR: one
    # 85 ms premature beat removed QT from every beat and cut PR.
    rr = np.full(60, 180.0)
    base, _ = _intervals(rr)
    rr[30], rr[31] = 85.0, 275.0
    df, rp = _intervals(rr)
    others = df.index.difference([30, 31, 32])
    np.testing.assert_allclose(df.loc[others, ["PR_ms", "QRS_ms", "QT_ms"]],
                               base.loc[others, ["PR_ms", "QRS_ms", "QT_ms"]])
    assert df.loc[others, "QT_ms"].notna().all()
    assert df["QT_ms"].median() == pytest.approx(T_END - QRS_ON, abs=1.0)
    rows = dict(analysis.interval_disclosure_rows(df))
    assert "shortened on 1 / 1 of 61 beats" in rows["Beat window"]


def test_qt_is_measured_at_a_fast_heart_rate():
    # RR 120 ms (500 bpm), T end 55 ms after R: the ±45 % window (±54 ms)
    # ended before the T end, so every QT was NaN.
    df, _ = _intervals(np.full(30, 120.0))
    assert df["QT_ms"].iloc[1:-1].notna().all()
    assert df["QT_ms"].median() == pytest.approx(T_END - QRS_ON, abs=1.0)


# A mouse beat at 500-700 bpm: T wave over by 38 ms after R, so the
# previous T ends before this P starts even at RR 90 ms.
_T_FAST = [(18.0, 0.0), (28.0, 0.2), (38.0, 0.0), (120.0, 0.0)]
_FAST = _P + _QRS + _T_FAST
_TRUE_PR = QRS_ON - P_ON


def _default_intervals(rr_ms, pts, noise=0.005, seed=0):
    """analyse_intervals with the DEFAULT template (no landmark override)."""
    sig, rp = _train(np.full(60, float(rr_ms)), seed=seed, noise=noise, pts=pts)
    df = analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                    wave_template=WaveTemplate())
    return df, sig, rp


@pytest.mark.parametrize("rr", [90.0, 100.0, 120.0])
def test_pr_is_measured_at_mouse_heart_rates_with_the_default_template(rr):
    # The beat window reaches 70 % of the RR back, onto the previous T wave:
    # the P baseline was taken there ("TP segment" 3-35 ms before the P
    # window), the flat TP stretch read as a negative P and PR came out at
    # 60 ms (RR 120) / 27 ms (RR 100) instead of 42, or NaN on most beats.
    df, _, _ = _default_intervals(rr, _FAST)
    ok = (df["PR_ms"] - _TRUE_PR).abs() <= 3.0
    assert ok.mean() >= 0.9
    assert df["P_peak_ms"].median() == pytest.approx(-42.0, abs=1.0)
    assert df["QT_ms"].median() == pytest.approx(38.0 - QRS_ON, abs=1.5)


def test_p_after_a_long_t_wave_is_found_at_rr_120():
    # The reported case: T wave until 55 ms after R, RR 120 ms -- P peak
    # read at -56 ms (true -42) and PR 60 ms on the default template.
    df, _, _ = _default_intervals(120.0, _P + _QRS + _T)
    assert ((df["PR_ms"] - _TRUE_PR).abs() <= 3.0).mean() >= 0.9
    assert df["P_peak_ms"].median() == pytest.approx(-42.0, abs=1.0)


@pytest.mark.parametrize("rr", [100.0, 200.0])
def test_no_p_wave_is_invented_from_noise_or_the_previous_t(rr):
    # No P wave at all: the old fallbacks (corrected extremum, then the
    # window maximum) turned noise or the previous T tail into a P.
    df, _, _ = _default_intervals(rr, _QRS + _T_FAST, noise=0.01)
    assert df["P_peak_ms"].isna().all() and df["PR_ms"].isna().all()


def test_a_normal_500_bpm_recording_raises_no_av_delay_event():
    # A GUI run at 500 bpm (T end 55 ms after R) gave PR 61 +/- 10 ms and 9
    # false "Sustained PR prolongation" events; a real long PR (70 ms) must
    # still raise one.
    from ecg.core.detection import classify_arrhythmias
    for pts, expect in ((_P + _QRS + _T, False), (_FAST, False),
                        ([(-120.0, 0.0), (-78.0, 0.0), (-70.0, 0.1), (-62.0, 0.0)]
                         + _QRS + _T_FAST, True)):
        df, _, rp = _default_intervals(120.0, pts)
        pr = analysis.align_interval_column_to_peaks(df, rp, FS)
        events = classify_arrhythmias(rp, FS, baseline_s=2.0, pr_ms=pr)
        assert any(e.kind == "av_delay" for e in events) is expect
        if expect:
            assert np.nanmedian(pr) == pytest.approx(70.0, abs=2.0)


def test_calibration_adds_ordered_qrs_landmarks():
    t, b = _mouse_demo_beat()
    wt = WaveTemplate()
    wt.calibrate_from_beat(t, b)
    lm = wt.landmarks
    assert lm["P_peak"][0] < lm["QRS_onset"][0] <= lm["Q_peak"][0] < 0
    assert 0 < lm["S_peak"][0] <= lm["QRS_offset"][0] < lm["T_peak"][0]


# ════════════════════════════════════════════════════════════
#  analyse_intervals
# ════════════════════════════════════════════════════════════

def _intervals(rr, **kw):
    sig, rp = _train(rr, **kw)
    df = analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                    wave_template=_template())
    return df, rp


def test_intervals_follow_the_standard_definitions():
    rr = np.random.default_rng(1).permutation(np.linspace(180.0, 220.0, 30))
    df, _ = _intervals(rr)
    ok = df[["PR_ms", "QRS_ms", "QT_ms"]].notna().all(axis=1)
    assert ok.mean() > 0.9
    d = df[ok]
    np.testing.assert_allclose(d["QRS_ms"], QRS_OFF - QRS_ON, atol=1.0)
    np.testing.assert_allclose(d["QT_ms"],  T_END - QRS_ON,   atol=1.5)
    np.testing.assert_allclose(d["PR_ms"],  QRS_ON - P_ON,    atol=1.5)
    np.testing.assert_allclose(d["QRS_onset_ms"],  QRS_ON,  atol=0.6)
    np.testing.assert_allclose(d["QRS_offset_ms"], QRS_OFF, atol=0.6)
    # PR ends and QT starts at the same landmark.
    np.testing.assert_allclose(d["PR_ms"] + d["P_onset_ms"], d["QRS_onset_ms"], atol=1e-6)


def test_noise_keeps_intervals_close_to_the_truth():
    rr = np.full(30, 200.0)
    df, _ = _intervals(rr, noise=0.003)
    assert df["QRS_ms"].median() == pytest.approx(QRS_OFF - QRS_ON, abs=1.0)
    assert df["QT_ms"].median()  == pytest.approx(T_END - QRS_ON, abs=1.0)


@pytest.mark.parametrize("fs", [1000.0, 2000.0, 4000.0])
@pytest.mark.parametrize("noise", [0.01, 0.02])
def test_noise_does_not_bias_qt_or_qrs_at_any_sampling_rate(fs, noise):
    # The single steepest-sample tangent made QT (and, at high fs, QRS)
    # shrink with noise, more so the higher the sampling rate.
    sig, rp = _train(np.full(40, 200.0), seed=3, noise=noise, fs=fs)
    df = analysis.analyse_intervals(sig, rp, fs, np.diff(rp) / fs * 1000.0,
                                    wave_template=_template())
    assert df["QT_ms"].notna().mean() > 0.9
    assert df["QT_ms"].median()  == pytest.approx(T_END - QRS_ON, abs=1.5)
    assert df["QRS_ms"].median() == pytest.approx(QRS_OFF - QRS_ON, abs=1.0)


def test_qrs_onset_is_dropped_when_no_interval_uses_it(monkeypatch):
    # Beat 10 keeps a QRS onset but loses P onset, J point and T end, so
    # none of PR / QRS / QT exists for it: its onset must not stay in the
    # table (it would feed the template and the verifier).
    real = analysis.detect_waves_on_beat
    calls = []

    def _fake(beat, bt, wt, **kw):
        det = real(beat, bt, wt, **kw)
        if len(calls) == 10:
            det.update(P_onset=np.nan, QRS_offset=np.nan, T_offset=np.nan)
        calls.append(1)
        return det

    monkeypatch.setattr(analysis, "detect_waves_on_beat", _fake)
    df, _ = _intervals(np.full(30, 200.0))
    row = df.iloc[10]
    assert np.isnan(row["PR_ms"]) and np.isnan(row["QRS_ms"]) and np.isnan(row["QT_ms"])
    assert np.isnan(row["QRS_onset_ms"]) and np.isnan(row["QRS_onset_s"])
    assert df["QRS_onset_ms"].drop(index=10).notna().all()


def test_qtc_uses_the_preceding_rr():
    rr = np.random.default_rng(2).permutation(np.linspace(180.0, 220.0, 30))
    df, rp = _intervals(rr)
    # One row per beat; RR_ms is the interval ENDING at that beat's R.
    assert len(df) == len(rp)
    rows = np.round(df["R_peak_s"].to_numpy() * FS).astype(int)
    pos = np.searchsorted(rp, rows)
    expected_rr = np.where(pos > 0, (rows - rp[np.maximum(pos - 1, 0)]) / FS * 1000.0,
                           np.nan)
    np.testing.assert_allclose(df["RR_ms"], expected_rr, equal_nan=True)
    assert np.isnan(df["RR_ms"].iloc[0]) and np.isnan(df["QTc_ms"].iloc[0])
    np.testing.assert_allclose(
        df["QTc_ms"], analysis.compute_qtc(df["QT_ms"], expected_rr, "mitchell"),
        equal_nan=True)
    assert analysis.qtc_rr_column(df) == "RR_ms" and "RR_prev_ms" not in df
    # The uncorrected QT is still there next to QTc.
    assert df["QT_ms"].notna().sum() >= df["QTc_ms"].notna().sum()


# ════════════════════════════════════════════════════════════
#  QT variability (formerly "QT dispersion")
# ════════════════════════════════════════════════════════════

def test_qt_beat_variability_values():
    v = analysis.qt_beat_variability(np.array([50.0, 52.0, 54.0, 56.0, 90.0, np.nan]))
    assert v["n"] == 5
    assert v["median_ms"] == pytest.approx(54.0)
    assert v["iqr_ms"] == pytest.approx(4.0)       # robust to the 90 ms outlier
    assert v["range_ms"] == pytest.approx(40.0)
    assert v["sd_ms"] == pytest.approx(np.std([50, 52, 54, 56, 90], ddof=1))


def test_qt_beat_variability_needs_four_beats():
    v = analysis.qt_beat_variability(np.array([50.0, 52.0, 54.0]))
    assert np.isnan(v["iqr_ms"]) and np.isnan(v["range_ms"])
    assert analysis.qt_variability_text(v) == "—"


def test_qt_variability_label_is_honest():
    lbl = analysis.QT_VARIABILITY_LABEL.lower()
    assert "beat-to-beat" in lbl and "single lead" in lbl
    assert "not classical qt dispersion" in lbl


def _frame(**extra):
    base = {"R_peak_s": [1.0, 1.1, 1.22, 1.34, 1.46],
            "RR_ms": [100.0, 120.0, 120.0, 120.0, np.nan],
            "QT_ms": [50.0, 52.0, 54.0, 56.0, 58.0]}
    base.update(extra)
    return pd.DataFrame(base)


def test_disclosure_rows_name_definitions_and_variability():
    rows = dict(analysis.interval_disclosure_rows(
        _frame(QRS_onset_ms=[-8.0] * 5)))
    assert rows["Interval definitions"] == analysis.INTERVAL_DEFINITIONS_TEXT
    assert "QT = QRS onset → T end" in rows["Interval definitions"]
    assert "preceding RR" in rows["Interval definitions"]
    assert rows["QT variability"].startswith("IQR 4.0 ms")
    assert "not classical QT dispersion" in rows["QT variability"]
    assert "dispersion" not in rows["QT variability"].split("—")[0]


def test_disclosure_rows_empty_without_qt():
    assert analysis.interval_disclosure_rows(None) == []
    assert analysis.interval_disclosure_rows(pd.DataFrame({"RR_ms": [1.0]})) == []


# ════════════════════════════════════════════════════════════
#  Old sessions
# ════════════════════════════════════════════════════════════

def test_legacy_frame_is_flagged_and_disclosed():
    old = _frame()                                   # no QRS_onset_ms column
    assert analysis.intervals_use_legacy_definitions(old)
    assert not analysis.intervals_use_legacy_definitions(_frame(QRS_onset_ms=[-8.0] * 5))
    rows = dict(analysis.interval_disclosure_rows(old))
    assert rows["Interval definitions"] == analysis.LEGACY_INTERVAL_DEFINITIONS_TEXT


def test_legacy_qtc_is_repaired_with_the_preceding_rr_idempotently():
    # Legacy RR_ms was the FOLLOWING RR: row i's preceding RR is row i-1's.
    old = _frame()
    fixed = analysis.recompute_qtc_column(old, "mitchell")
    prev = np.array([np.nan, 100.0, 120.0, 120.0, 120.0])
    np.testing.assert_allclose(
        fixed["QTc_ms"], analysis.compute_qtc(old["QT_ms"], prev, "mitchell"),
        equal_nan=True)
    pd.testing.assert_series_equal(fixed["RR_ms"], old["RR_ms"])   # left as stored
    # The RR actually used is stored next to it, for the QTc-RR plot and
    # for anyone recomputing QTc from the exported columns.
    assert analysis.qtc_rr_column(fixed) == "RR_prev_ms"
    np.testing.assert_allclose(fixed["RR_prev_ms"], prev, equal_nan=True)
    np.testing.assert_allclose(
        fixed["QTc_ms"],
        analysis.compute_qtc(fixed["QT_ms"], fixed[analysis.qtc_rr_column(fixed)],
                             "mitchell"), equal_nan=True)
    again = analysis.recompute_qtc_column(fixed, "mitchell")
    np.testing.assert_allclose(again["QTc_ms"], fixed["QTc_ms"], equal_nan=True)


def test_legacy_frame_without_r_peak_times_gets_no_qtc():
    # Its RR_ms is the FOLLOWING RR and the preceding one cannot be
    # recovered: QTc is left empty rather than paired with the wrong RR,
    # and the definitions line says so.
    old = _frame().drop(columns="R_peak_s")
    fixed = analysis.recompute_qtc_column(old, "mitchell")
    assert fixed["QTc_ms"].isna().all()
    assert analysis.qtc_rr_column(fixed) == "RR_prev_ms"
    text = analysis.interval_definitions_text(fixed)
    assert text == analysis.LEGACY_INTERVAL_DEFINITIONS_NO_QTC_TEXT
    assert "QTc not available" in text and "re-paired" not in text
    assert dict(analysis.interval_disclosure_rows(fixed))["Interval definitions"] == text
    assert analysis.interval_definitions_text(_frame()) ==         analysis.LEGACY_INTERVAL_DEFINITIONS_TEXT


def test_legacy_rows_that_are_not_consecutive_get_no_qtc():
    old = _frame(R_peak_s=[1.0, 1.1, 1.5, 1.62, 1.74])   # gap: beat(s) missing
    fixed = analysis.recompute_qtc_column(old, "mitchell")
    assert np.isnan(fixed["QTc_ms"].iloc[2])
    assert np.isfinite(fixed["QTc_ms"].iloc[1])


# ════════════════════════════════════════════════════════════
#  Exports
# ════════════════════════════════════════════════════════════

_HR = {"hr": {"n": 3, "mean": 500.0, "min": 480.0, "max": 520.0, "std": 5.0}}


def _cover(intervals):
    wb = ExcelExporter.build_workbook(
        results={**_HR, "intervals": intervals},
        signal_flt=None, signal_raw=None, time=np.arange(10) / 1000.0,
        rpeaks_ok=None, fs=1000, filepath=None, subject="m1",
        sig_quality=None, epoch_df=None, qtc_formula="mitchell")
    ws = wb["Cover"]
    return {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
            for r in range(1, ws.max_row + 1)}


def test_excel_cover_discloses_interval_definitions_and_qt_variability():
    cover = _cover(_frame(QRS_onset_ms=[-8.0] * 5))
    assert cover["Interval definitions"] == analysis.INTERVAL_DEFINITIONS_TEXT
    assert "not classical QT dispersion" in cover["QT variability"]
    assert not any("QT dispersion" == str(k) for k in cover)


def test_excel_cover_without_intervals_has_no_interval_rows():
    cover = _cover(None)
    assert "Interval definitions" not in cover and "QT variability" not in cover


def test_prism_notes_disclose_interval_definitions(tmp_path):
    pytest.importorskip("pzfx")
    path = tmp_path / "ivl.pzfx"
    ivl = _frame(QRS_onset_ms=[-8.0] * 5, PR_ms=[40.0] * 5, QRS_ms=[16.0] * 5,
                 QTc_ms=[50.0] * 5)
    PrismExporter.build_and_write(path, {**_HR, "intervals": ivl}, qtc_formula="mitchell")
    xml = path.read_text(encoding="utf-8")
    assert "<Name>Interval definitions</Name>" in xml
    assert "<Name>QT variability</Name>" in xml
    assert "QRS onset → T end" in xml


def test_prism_writes_the_interval_timeline_from_r_peak_times(tmp_path):
    # analyse_intervals frames carry R_peak_s, not Time_s: the timeline
    # table used to be skipped for every real recording.
    pytest.importorskip("pzfx")
    path = tmp_path / "ivl_tl.pzfx"
    ivl = _frame(QRS_onset_ms=[-8.0] * 5, PR_ms=[40.0] * 5, QRS_ms=[16.0] * 5,
                 QTc_ms=[50.0] * 5)
    PrismExporter.build_and_write(path, {**_HR, "intervals": ivl}, qtc_formula="mitchell")
    xml = path.read_text(encoding="utf-8")
    assert "ECG Intervals — Timeline" in xml
    assert "1.22" in xml

