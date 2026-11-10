"""Delineation at mouse heart rates when the next beat's P wave reaches the T
search (T smaller than P, or inverted) and on a band-pass filtered signal.

The T wave was the largest sample of the T window, so a small T lost to the
next P: QT was wrong and that false T peak, bounding the next beat's P search,
removed PR altogether. A 10 Hz high-pass (what "1 Hz" becomes at 2 kHz) leaves
the PR segment below the level the P starts from and a trough before the P as
prominent as the P itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg.core.analysis import align_interval_column_to_peaks, analyse_intervals
from ecg.core.detection import classify_arrhythmias
from ecg.core.filtering import bandpass
from ecg.core.wave_template import WaveTemplate

FS = 2000.0
QRS_ON = -8.5


def _wave(t, c, sig, a):
    z = (t - c) / sig
    base = np.exp(-0.5 * 2.5 ** 2)
    v = (np.exp(-0.5 * z ** 2) - base) / (1 - base)
    return np.where(np.abs(z) < 2.5, a * v, 0.0)


def _trace(rr, pr=42.0, t_amp=0.20, noise=0.0, n_beats=60, p_amp=0.12):
    rp = np.round((0.5 + np.arange(n_beats + 1) * rr / 1000.0) * FS).astype(int)
    n = rp[-1] + int(0.5 * FS)
    tm = np.arange(n) / FS * 1000
    x = np.zeros(n)
    t_end = 52.0 * np.clip(rr / 120.0, 0.7, 1.6) ** 0.25
    for r in rp:
        lo, hi = max(0, r - 400), min(n, r + 400)
        t = tm[lo:hi] - tm[r]
        x[lo:hi] += (_wave(t, QRS_ON - pr + 8.0, 3.2, p_amp)
                     + _wave(t, -5.5, 1.2, -0.12) + _wave(t, 0.0, 1.7, 1.0)
                     + _wave(t, 5.0, 1.5, -0.30) + _wave(t, 11.5, 2.5, 0.25)
                     + _wave(t, t_end - 22.5, 8.5, t_amp))
    if noise:
        x += np.random.default_rng(0).normal(0.0, noise, n)
    return x, rp, t_end - QRS_ON


def _measure(x, rp):
    return analyse_intervals(x, rp, FS, np.diff(rp) / FS * 1000,
                             wave_template=WaveTemplate())


@pytest.mark.parametrize("rr", [90, 100, 110, 120, 130, 140, 160, 200])
@pytest.mark.parametrize("t_amp", [0.10, 0.08, 0.06, -0.10])
def test_t_smaller_than_p_keeps_pr_and_qt(rr, t_amp):
    x, rp, qt = _trace(rr, t_amp=t_amp, noise=0.002)
    df = _measure(x, rp)
    assert df["PR_ms"].notna().mean() >= 0.95
    assert df["PR_ms"].median() == pytest.approx(42.0, abs=3.0)
    assert df["QT_ms"].notna().mean() >= 0.95
    assert df["QT_ms"].median() == pytest.approx(qt, abs=8.0)   # T end is cut 5 ms early


@pytest.mark.parametrize("rr", [100, 105, 110, 115, 120, 125, 130, 140, 150, 160, 200])
@pytest.mark.parametrize("noise", [0.0, 0.005])
def test_band_pass_keeps_pr_and_qt(rr, noise):
    x, rp, qt = _trace(rr, noise=noise)
    df = _measure(bandpass(x, FS, 1.0, 150.0), rp)      # really 10-150 Hz at 2 kHz
    assert df["PR_ms"].notna().mean() >= 0.95
    assert df["PR_ms"].median() == pytest.approx(42.0, abs=4.0)
    assert df["QT_ms"].notna().mean() >= 0.95
    assert df["QT_ms"].median() == pytest.approx(qt, abs=9.0)


def test_qt_kept_when_next_p_starts_before_t_end():
    # RR 85: the next P starts ~13 ms before the T end, so the edge is cut short.
    x, rp, qt = _trace(85)
    df = _measure(x, rp)
    assert df["QT_ms"].notna().mean() >= 0.95
    assert df["QT_ms"].median() == pytest.approx(qt, abs=8.0)


@pytest.mark.parametrize("rr,pr", [(120, 75.0), (150, 75.0), (200, 75.0), (250, 75.0)])
def test_long_pr_before_the_p_window_is_measured_and_raises_the_event(rr, pr):
    # P peak before the default window (-73 ms): it read NaN, so no event.
    x, rp, _ = _trace(rr, pr=pr, noise=0.005, n_beats=200)
    df = _measure(x, rp)
    assert df["PR_ms"].notna().mean() >= 0.85
    assert df["PR_ms"].median() == pytest.approx(pr, abs=3.0)
    pa = align_interval_column_to_peaks(df, rp, FS, "PR_ms")
    events = classify_arrhythmias(rp, FS, pr_ms=pa)
    assert any(e.kind == "av_delay" for e in events)


@pytest.mark.parametrize("rr", [100, 120, 150, 250])
def test_no_p_wave_gives_no_pr_rather_than_a_false_long_one(rr):
    x, rp, _ = _trace(rr, p_amp=0.0, noise=0.005, n_beats=200)
    pa = align_interval_column_to_peaks(_measure(x, rp), rp, FS, "PR_ms")
    assert np.isfinite(pa).mean() <= 0.02
    assert not any(e.kind == "av_delay" for e in classify_arrhythmias(rp, FS, pr_ms=pa))
