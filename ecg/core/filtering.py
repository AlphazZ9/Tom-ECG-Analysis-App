"""
ecg.core.filtering
──────────────────
Signal-processing helpers: bandpass, notch, normalise, display downsampling.
No UI imports — pure NumPy / SciPy.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from ecg.core.models import MouseECG

log = logging.getLogger("ecg")


# ════════════════════════════════════════════════════════════
#  FILTERS
# ════════════════════════════════════════════════════════════

def bandpass(
    signal: np.ndarray,
    fs: float,
    lo: float = MouseECG.BP_LO_HZ,
    hi: float = MouseECG.BP_HI_HZ,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter.

    Returns the signal unchanged if the passband is degenerate.
    """
    nyq = fs / 2
    lo_norm = max(0.01, min(lo / nyq, 0.99))
    hi_norm = max(0.01, min(hi / nyq, 0.99))
    if lo_norm >= hi_norm:
        log.warning("bandpass: degenerate passband [%.4f, %.4f] — skipped", lo_norm, hi_norm)
        return signal
    b, a = butter(3, [lo_norm, hi_norm], btype="band")  # type: ignore[misc]
    return filtfilt(b, a, signal)


def notch(signal: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch filter (default: 50 Hz mains rejection).

    Raises
    ------
    ValueError
        If *freq* is not strictly within (0, fs/2).
    """
    nyq = fs / 2.0
    if not (0.0 < freq < nyq):
        raise ValueError(
            f"Notch frequency {freq:.1f} Hz must be in (0, {nyq:.1f}) Hz "
            f"for fs={fs:.0f} Hz. "
            "Check Signal › Sampling rate or disable the notch filter."
        )
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, signal)


def normalize(signal: np.ndarray) -> np.ndarray:
    """Return zero-mean unit-variance signal. No-op if std ≈ 0."""
    std = signal.std()
    if std < 1e-10:
        log.warning("normalize: signal std ≈ 0 — returning unchanged")
        return signal
    return (signal - signal.mean()) / std


# ════════════════════════════════════════════════════════════
#  DISPLAY DOWNSAMPLING
# ════════════════════════════════════════════════════════════

def downsample_for_display(arr: np.ndarray, max_points: int = 6_000,
                            is_time: bool = False) -> np.ndarray:
    """Decimate *arr* to at most *max_points* preserving signal extrema.

    Each bucket produces 2 output points: the min-amplitude sample and the
    max-amplitude sample, emitted in chronological order.  This guarantees
    narrow spikes are never dropped and consecutive points are always in time
    order so matplotlib draws short vertical strokes instead of sawtooth lines.
    """
    n = len(arr)
    if n <= max_points:
        return arr
    bucket   = max(1, n // (max_points // 2))
    n_trim   = (n // bucket) * bucket
    reshaped = arr[:n_trim].reshape(-1, bucket)
    n_buckets = reshaped.shape[0]

    if is_time:
        left  = reshaped[:, 0]
        right = reshaped[:, bucket - 1]
        out   = np.empty(n_buckets * 2, dtype=arr.dtype)
        out[0::2] = left
        out[1::2] = right
    else:
        argmins = reshaped.argmin(axis=1)
        argmaxs = reshaped.argmax(axis=1)
        base    = np.arange(n_buckets) * bucket
        idx_min = base + argmins
        idx_max = base + argmaxs
        first   = np.where(idx_min <= idx_max, idx_min, idx_max)
        second  = np.where(idx_min <= idx_max, idx_max, idx_min)
        out     = np.empty(n_buckets * 2, dtype=arr.dtype)
        out[0::2] = arr[first]
        out[1::2] = arr[second]
    return out


def downsample_signal(
    signal: np.ndarray,
    fs_in: float,
    fs_out: float = 10_000.0,
) -> "tuple[np.ndarray, float]":
    """Downsample *signal* from *fs_in* to *fs_out* Hz using scipy decimate.

    Returns (resampled_signal, actual_fs_out).
    If fs_in <= fs_out, returns the signal unchanged.
    """
    from scipy.signal import decimate as _decimate
    if fs_in <= fs_out:
        return signal.copy(), fs_in
    factor = int(round(fs_in / fs_out))
    if factor < 2:
        return signal.copy(), fs_in
    # scipy.decimate applies anti-aliasing filter before decimation
    out = _decimate(signal, factor, ftype="fir", zero_phase=True)
    actual_fs = fs_in / factor
    log.info(
        "downsample_signal: %d Hz → %d Hz  (factor %d,  %d→%d samples)",
        int(fs_in), int(actual_fs), factor, len(signal), len(out),
    )
    return np.asarray(out, dtype=np.float64), actual_fs


def sg_filter(
    signal: np.ndarray,
    fs: float,
    window_ms: float = 20.0,
    polyorder: int = 3,
) -> np.ndarray:
    """Apply a zero-phase Savitzky–Golay smoothing filter.

    Parameters
    ----------
    window_ms  : Filter window length in milliseconds (default 20 ms ≈ 2 × QRS width).
    polyorder  : Polynomial order (default 3 gives good impulse preservation).
    """
    from scipy.signal import savgol_filter as _savgol
    if len(signal) < polyorder + 1:
        raise ValueError(
            f"sg_filter: signal has only {len(signal)} sample(s), "
            f"need at least {polyorder + 1} for polyorder={polyorder}"
        )
    window_samples = int(round(window_ms / 1000.0 * fs))
    if window_samples < polyorder + 1:
        window_samples = polyorder + 1
    # savgol_filter requires odd window length
    if window_samples % 2 == 0:
        window_samples += 1
    # Also requires window_length <= len(signal) -- shrink to fit rather than
    # let scipy raise an opaque ValueError deep in the call stack.
    if window_samples > len(signal):
        window_samples = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    log.debug("sg_filter: window=%d samples (%.1f ms at %d Hz)", window_samples, window_ms, int(fs))
    return np.asarray(
        _savgol(signal, window_length=window_samples, polyorder=polyorder,
               mode="interp"), dtype=np.float64)


def sg_derivative_signal(
    signal: np.ndarray,
    fs: float,
    window_ms: float = 20.0,
    polyorder: int = 3,
    deriv: int = 1,
) -> np.ndarray:
    """Compute the first derivative of *signal* via Savitzky–Golay differentiation.

    This is more noise-robust than a simple np.diff because the SG filter
    simultaneously smooths and differentiates in a single convolution.
    The output is the instantaneous slope (mV/s equivalent), which has a
    sharp positive peak aligned to the ascending R-wave upstroke.
    """
    from scipy.signal import savgol_filter as _savgol
    if len(signal) < polyorder + 1:
        raise ValueError(
            f"sg_derivative_signal: signal has only {len(signal)} sample(s), "
            f"need at least {polyorder + 1} for polyorder={polyorder}"
        )
    window_samples = int(round(window_ms / 1000.0 * fs))
    if window_samples < polyorder + 1:
        window_samples = polyorder + 1
    if window_samples % 2 == 0:
        window_samples += 1
    # Also requires window_length <= len(signal) -- shrink to fit rather than
    # let scipy raise an opaque ValueError deep in the call stack.
    if window_samples > len(signal):
        window_samples = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    return np.asarray(
        _savgol(
            signal,
            window_length=window_samples,
            polyorder=polyorder,
            deriv=deriv,
            delta=1.0 / fs,
            mode="interp",
        ), dtype=np.float64)


def downsample_pair(sig: np.ndarray, time: np.ndarray,
                    max_points: int = 6_000) -> "tuple[np.ndarray, np.ndarray]":
    """Decimate signal/time pair for the overview by uniform stride."""
    n = len(sig)
    if n <= max_points:
        return time, sig
    stride = max(1, n // max_points)
    return time[::stride], sig[::stride]


def envelope_for_display(
    sig: np.ndarray,
    time: np.ndarray,
    max_buckets: int = 3_000,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Compute per-bucket min/max envelope for overview plots.

    Returns (t_mid, mins, maxs, midline).
    """
    n = len(sig)
    if n <= max_buckets:
        return time, sig, sig, sig
    bucket  = max(1, n // max_buckets)
    n_trim  = (n // bucket) * bucket
    sig_r   = sig[:n_trim].reshape(-1, bucket)
    time_r  = time[:n_trim].reshape(-1, bucket)
    t_mid   = time_r[:, bucket // 2]
    mins    = sig_r.min(axis=1)
    maxs    = sig_r.max(axis=1)
    midline = sig_r.mean(axis=1)
    return t_mid, mins, maxs, midline
