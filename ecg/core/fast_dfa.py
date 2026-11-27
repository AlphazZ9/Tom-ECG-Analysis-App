"""DFA α1 / α2 and multifractal DFA of an RR series, as NeuroKit 0.2.13
computes them (neurokit2.hrv.hrv_nonlinear._hrv_dfa and
neurokit2.complexity.fractal_dfa), without its per-window Python loop.

NeuroKit fits each window's linear trend with one vectorised np.polyfit, then
evaluates it window by window with np.polyval: ~840 000 calls for an hour of
mouse ECG (10 s of the 13 s of the whole analysis). Here the trends of every
window of a scale are evaluated at once with the same arithmetic as polyval
(Horner: (0·x + a)·x + b = a·x + b), so the fluctuations, slopes and indices
are the same numbers, bit for bit (tests/core/test_fast_dfa.py). Only the
defaults the app uses are supported: profile integration, overlapping
windows, order 1, no max-DFA."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _windows(signal: np.ndarray, window: int) -> np.ndarray:
    """NeuroKit's overlapping segments: starts 0, w/2, w, … < n − w."""
    n = len(signal)
    starts = np.arange(0, n - window, window // 2)
    return sliding_window_view(signal, window)[starts]


def _fluctuation(segments: np.ndarray, window: int, q: np.ndarray) -> np.ndarray:
    x = np.arange(window)
    coefs = np.polyfit(x[:window], segments.T, 1).T          # as NeuroKit
    trends = coefs[:, :1] * x + coefs[:, 1:2]                  # polyval, all windows
    var = np.var(segments - trends, axis=1)
    var = var[var > 1e-08 * np.max(var)]
    if len(var) == 0:
        warnings.warn("All detrended segments have no variance. Retuning NaN.")
        return np.full(len(q), np.nan)
    is0 = np.abs(q) < 0.1
    q_non0 = q[~is0].reshape(-1, 1)
    fluct = np.float_power(np.mean(np.float_power(var, q_non0 / 2), axis=1), 1 / q_non0.T)
    if np.sum(is0) > 0:
        fluc0 = np.exp(0.5 * np.mean(np.log(var)))
        fluct = np.insert(fluct, np.where(is0)[0], [fluc0])
    return fluct


def fractal_dfa(signal: Any, scale: Any, multifractal: bool = False,
                q: Any = 2) -> "tuple[Any, dict]":
    """neurokit2.fractal_dfa(signal, scale=scale, multifractal=…, q=…) with
    its defaults (integrate, overlap, order 1, maxdfa off): the same return."""
    from neurokit2.complexity.fractal_dfa import (   # NeuroKit 0.2.13 internals
        _fractal_dfa_findscales, _sanitize_q, _singularity_spectrum, _slopes)
    signal = np.asarray(signal)
    n = len(signal)
    scale = _fractal_dfa_findscales(n, scale)
    q = _sanitize_q(q, multifractal=multifractal)
    info: dict = {"scale": scale, "q": q}
    signal = np.cumsum(signal - np.mean(signal))
    fluctuations = np.zeros((len(scale), len(q)))
    for i, window in enumerate(scale):
        fluctuations[i] = _fluctuation(_windows(signal, int(window)), int(window), q)
    if len(fluctuations) == 0:
        return np.nan, info
    slopes = _slopes(scale, fluctuations, q)
    if len(slopes) == 1:
        slopes = slopes[0]
    info["Fluctuations"] = fluctuations
    info["Alpha"] = slopes
    if multifractal:
        info.update(_singularity_spectrum(q=q, slopes=slopes))
        keys = ["Peak", "Width", "Mean", "Max", "Delta", "Asymmetry", "Fluctuation", "Increment"]
        return pd.DataFrame({k: v for k, v in info.items() if k in keys}, index=[0]), info
    return slopes, info


def hrv_dfa(rri: Any, out: dict) -> dict:
    """neurokit2.hrv.hrv_nonlinear._hrv_dfa(rri, out) with its defaults:
    DFA_alpha1 (windows 4–11 beats), DFA_alpha2 (12 to N/10) and their
    MFDFA indices (q = −5…5), in *out*."""
    from neurokit2.misc import NeuroKitWarning
    if len(rri) < 12:
        out["DFA_alpha1"] = np.nan
        out["DFA_alpha2"] = np.nan
        return out
    dfa_windows = [(4, 11), (12, None)]
    max_beats = (len(rri) + 1) / 10
    n_short = int(dfa_windows[0][1] - dfa_windows[0][0] + 1)
    n_long = int(max_beats - dfa_windows[1][0] + 1)
    short = np.linspace(dfa_windows[0][0], dfa_windows[0][1], n_short).astype(int)
    out["DFA_alpha1"], _ = fractal_dfa(rri, multifractal=False, scale=short)
    m1, _ = fractal_dfa(rri, multifractal=True, q=np.arange(-5, 6), scale=short)
    for k in m1.columns:
        out["MFDFA_alpha1_" + k] = m1[k].values[0]
    if max_beats < dfa_windows[1][0] + 1:
        warnings.warn(
            "DFA_alpha2 related indices will not be calculated. "
            "The maximum duration of the windows provided for the long-term correlation is smaller "
            "than the minimum duration of windows. Refer to the `scale` argument in `nk.fractal_dfa()` "
            "for more information.", category=NeuroKitWarning)
        return out
    long_ = np.linspace(dfa_windows[1][0], int(max_beats), n_long).astype(int)
    out["DFA_alpha2"], _ = fractal_dfa(rri, multifractal=False, scale=long_)
    m2, _ = fractal_dfa(rri, multifractal=True, q=np.arange(-5, 6), scale=long_)
    for k in m2.columns:
        out["MFDFA_alpha2_" + k] = m2[k].values[0]
    return out
