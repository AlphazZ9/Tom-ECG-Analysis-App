"""Frequency HRV with long runs of excluded intervals (a noisy region marked
by hand, a long artifact): the series is cut there and the Welch PSDs of
the stretches in between are averaged, instead of withholding the spectrum
of the whole recording."""
from __future__ import annotations

import numpy as np
import pytest

from ecg.core.analysis import (
    analyse_hrv_freq, hrv_spectrum, hrv_spectrum_gapped, noise_breaks,
    rr_interval_series, spectral_band_metrics, spectrum_segments_text,
)
from ecg.core.models import MouseECG

FS = 2000.0


def _peaks(duration_s: float = 600.0, seed: int = 0) -> np.ndarray:
    """R peaks of a mouse-like RR series (~120 ms) modulated at 0.8 Hz (LF)
    and 3 Hz (HF), with a little noise."""
    rng = np.random.default_rng(seed)
    t, out = 0.5, []
    while t < duration_s:
        out.append(t)
        t += (0.120 + 0.004 * np.sin(2 * np.pi * 0.8 * t)
              + 0.003 * np.sin(2 * np.pi * 3.0 * t) + rng.normal(0, 0.0005))
    return (np.asarray(out) * FS).astype(np.int64)


def _series(peaks, regions=None):
    br = noise_breaks(peaks, FS, regions) if regions else None
    return rr_interval_series(peaks, FS, MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS, breaks=br)


def test_without_a_long_gap_the_spectrum_is_unchanged():
    ser = _series(_peaks())
    spec, reason = hrv_spectrum_gapped(ser["rr_ms"], ser["valid"], ser["t_s"])
    ref = hrv_spectrum(ser["rri"], ser["rri_time"])
    assert reason is None and spec["n_segments"] == 1
    assert np.array_equal(spec["psd"], ref["psd"]) and spec["nperseg"] == ref["nperseg"]


def test_a_noisy_region_no_longer_withholds_the_whole_spectrum():
    pk = _peaks()
    clean = spectral_band_metrics(hrv_spectrum_gapped(*(_series(pk)[k] for k in
                                                        ("rr_ms", "valid", "t_s")))[0])
    ser = _series(pk, [{"t_start": 300.0, "t_end": 306.0}])
    assert ser["max_excluded_run_s"] > MouseECG.SPECTRAL_MAX_GAP_S
    spec, reason = hrv_spectrum_gapped(ser["rr_ms"], ser["valid"], ser["t_s"])
    assert reason is None and spec["n_segments"] == 2
    got = spectral_band_metrics(spec)
    for k in ("HRV_LF", "HRV_HF"):
        assert got[k] == pytest.approx(clean[k], rel=0.15), k
    assert got["HRV_LFpeak"] == pytest.approx(0.8, abs=0.05)
    assert got["HRV_HFpeak"] == pytest.approx(3.0, abs=0.05)


def test_analyse_hrv_freq_says_it_averaged_stretches():
    pk = _peaks()
    br = noise_breaks(pk, FS, [{"t_start": 100.0, "t_end": 108.0},
                               {"t_start": 400.0, "t_end": 403.0}])
    fd = analyse_hrv_freq(pk, FS, breaks=br)
    assert not fd.empty and fd.attrs["psd_segments"] == 3
    text = spectrum_segments_text(fd.attrs)
    assert text.startswith("averaged over 3 stretches") and "of 599 s used" in text
    # Without a noisy region: nothing to say.
    assert spectrum_segments_text(analyse_hrv_freq(pk, FS).attrs) == ""


def test_stretches_too_short_are_left_out_or_the_spectrum_withheld():
    pk = _peaks(200.0)
    # A gap every 20 s: stretches of ~17 s, all used with a shorter window
    # (256 samples, 12.8 s).
    regions = [{"t_start": float(a), "t_end": float(a) + 3.0} for a in range(20, 200, 20)]
    ser = _series(pk, regions)
    spec, reason = hrv_spectrum_gapped(ser["rr_ms"], ser["valid"], ser["t_s"])
    assert reason is None and spec["n_segments"] >= 8
    # Only tiny stretches: withheld, with why.
    regions = [{"t_start": float(a), "t_end": float(a) + 2.6} for a in np.arange(1, 200, 4.0)]
    ser = _series(pk, regions)
    spec, reason = hrv_spectrum_gapped(ser["rr_ms"], ser["valid"], ser["t_s"])
    assert spec is None and reason


def test_the_excluded_time_rule_still_applies_inside_the_stretches():
    pk = _peaks()
    ser = _series(pk, [{"t_start": 300.0, "t_end": 306.0}])
    valid = ser["valid"].copy()
    # Exclude every 10th interval: short holes, but 10 % of the time.
    valid[::10] = False
    spec, reason = hrv_spectrum_gapped(ser["rr_ms"], valid, ser["t_s"])
    assert spec is None and "cover" in reason


def test_one_stretch_left_is_said_plainly():
    pk = _peaks(240.0)
    # Noise at 30-35 s: the first 30 s are shorter than the 50 s window.
    fd = analyse_hrv_freq(pk, FS, breaks=noise_breaks(pk, FS, [{"t_start": 30.0,
                                                                "t_end": 35.0}]))
    text = spectrum_segments_text(fd.attrs)
    assert text.startswith("computed on 1 stretch") and "1 stretch too short left out" in text
