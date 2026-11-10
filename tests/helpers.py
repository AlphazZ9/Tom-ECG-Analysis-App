"""
tests.helpers
─────────────
Shared synthetic-signal builders for the test suite. No pytest imports here
on purpose -- these are plain functions usable from any test module (or an
interactive REPL) without pulling in fixtures/collection machinery.
"""
from __future__ import annotations

import numpy as np


def rpeaks_from_rr(rr_ms_list: "list[float]", fs: float, start: int = 1000) -> np.ndarray:
    """Build a strictly-increasing R-peak sample-index array from a list of
    RR intervals (ms). ``len(result) == len(rr_ms_list) + 1``.

    ``start`` keeps the first peak away from sample 0 -- several detection.py
    functions look a short window *before* a peak (upstroke slope, snap),
    so a peak sitting exactly at index 0 would silently clip that window
    instead of exercising it.
    """
    rp = [start]
    for rr in rr_ms_list:
        rp.append(rp[-1] + int(round(rr / 1000.0 * fs)))
    return np.array(rp, dtype=int)


def add_qrs_pulse(sig: np.ndarray, center: int, fs: float,
                   amp: float = 1.0, qrs_width_ms: float = 8.0,
                   s_wave_frac: float = 0.35) -> None:
    """Add one realistic-ish mouse-QRS-shaped pulse to *sig* in place, centred
    at sample *center*: a sharp asymmetric R upstroke/downstroke (fast rise,
    slower decay) plus a small negative S-wave deflection just after it --
    closer to a real triphasic QRS than a bare spike, which matters for
    detect_peaks_wavelet's composite scoring (template NCC, curvature, ...).
    """
    qrs_w = qrs_width_ms / 1000.0
    lo = max(0, center - int(0.02 * fs))
    hi = min(len(sig), center + int(0.02 * fs))
    if lo >= hi:
        return
    tt = (np.arange(lo, hi) - center) / fs
    tau_r, tau_d = qrs_w * 0.15, qrs_w * 0.6
    r_wave = np.where(tt < 0, np.exp(tt / tau_r), np.exp(-tt / tau_d))
    s_center = qrs_w * 0.5
    s_wave = -s_wave_frac * np.exp(-0.5 * ((tt - s_center) / (qrs_w * 0.25)) ** 2)
    sig[lo:hi] += amp * (r_wave + s_wave)


def make_synth_ecg(n_beats: int = 40, fs: float = 2000.0, rr_ms: float = 120.0,
                    qrs_width_ms: float = 8.0, amp: float = 1.0,
                    noise_sd: float = 0.02, seed: int = 0,
                    ) -> "tuple[np.ndarray, float, np.ndarray]":
    """A crude but sharp-upstroke synthetic mouse-ECG-like train: one
    QRS-shaped pulse per beat at a uniform RR interval, plus Gaussian noise.

    Returns (signal, fs, true_peak_samples). Recall against true_peak_samples
    is checked with a small (~5 ms) tolerance in the tests, not exact
    equality -- every detector snaps to its own notion of "the peak sample",
    which can land a sample or two off the pulse's mathematical centre.
    """
    rng = np.random.default_rng(seed)
    rr_samp = int(round(rr_ms / 1000.0 * fs))
    n = rr_samp * (n_beats + 2)
    sig = np.zeros(n)
    true_peaks = []
    for b in range(1, n_beats + 1):
        center = b * rr_samp
        true_peaks.append(center)
        add_qrs_pulse(sig, center, fs, amp=amp, qrs_width_ms=qrs_width_ms)
    sig += rng.normal(0, noise_sd, n)
    return sig, fs, np.array(true_peaks, dtype=int)


def match_recall(detected: np.ndarray, true_peaks: np.ndarray, tol_samples: int) -> int:
    """Count how many *true_peaks* have a detected peak within tol_samples."""
    if len(detected) == 0:
        return 0
    matched = 0
    for tp in true_peaks:
        if np.min(np.abs(detected - tp)) <= tol_samples:
            matched += 1
    return matched
