"""At mouse heart rates above ~450 bpm the template's T windows reach the next
beat's P wave: the T peak is never searched past the start of that P."""
from __future__ import annotations

import numpy as np

from ecg.core.wave_template import WaveTemplate, detect_waves_on_beat

FS = 1000.0
RR_MS = 120.0               # 500 bpm


def _lobe(t, centre, width, amp):
    return amp * np.exp(-0.5 * ((t - centre) / width) ** 2)


def _fast_beat(next_p_amp: float = 0.3):
    """A beat at RR 120 ms with its window reaching the next beat's P: a small
    T at ~32 ms, and the next P (trough then peak) at ~RR - 45 ms."""
    t = np.arange(-84, 84) / FS * 1000
    y = (_lobe(t, 0, 1.6, 5.0) + _lobe(t, -4, 1.5, -1.0) + _lobe(t, 4, 2.0, -2.5)
         + _lobe(t, 32, 5.0, 0.12)                                   # this beat's T
         + _lobe(t, -45, 2.5, -0.15) + _lobe(t, -38, 2.5, 0.3)        # this beat's P
         + _lobe(t, RR_MS - 45, 2.5, -0.5 * next_p_amp)               # next beat's P
         + _lobe(t, RR_MS - 38, 2.5, next_p_amp))
    rng = np.random.default_rng(0)
    return t, y + 0.004 * rng.standard_normal(len(t))


def test_the_next_p_is_not_the_t():
    t, y = _fast_beat()
    det = detect_waves_on_beat(y, t, WaveTemplate(), next_r_ms=RR_MS)
    assert np.isfinite(det["P_peak"])
    # The next P's trough starts ~50 ms before its R: never the T peak.
    assert det["T_peak"] < RR_MS - 50.0
    assert 25.0 <= det["T_peak"] <= 40.0           # the T is this beat's own


def test_without_a_next_r_nothing_is_capped():
    # The cap needs the next R: without it the search is the template's, as before.
    t, y = _fast_beat(next_p_amp=0.0)
    a = detect_waves_on_beat(y, t, WaveTemplate())
    b = detect_waves_on_beat(y, t, WaveTemplate(), next_r_ms=10_000.0)
    for k in ("T_peak", "T_offset"):
        assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k]
