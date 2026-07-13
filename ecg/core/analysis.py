"""
ecg.core.analysis
─────────────────
HRV analysis: time-domain, frequency-domain, nonlinear metrics, interval
delineation.  All functions are pure NumPy/SciPy and safe to call from
background threads.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, Optional, cast

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import welch as _scipy_welch

# numpy 2.x renamed trapz to trapezoid (older numpy/some builds only have
# trapz). This used to rely on app.py's own module-level shim monkey-patching
# np.trapz before `from analysis import ...` ran -- which meant this module
# could not be imported on its own on a numpy version lacking trapz. That's
# exactly what batch.py's subprocess workers do (they never import app.py),
# so batch mode could not run at all on numpy >= 2.0 until this was resolved
# locally, with no dependency on import order.
trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid", None)
if trapz is None:
    from scipy.integrate import trapezoid as trapz  # last-resort fallback

from ecg.core.models import AnalysisResults, MouseECG
from ecg.core.wave_template import WaveTemplate, detect_waves_on_beat

# Detected locally rather than imported from theme.py: this module is pure
# NumPy/SciPy and is imported by batch.py's subprocess workers (documented as
# having no Tkinter/CTk dependency), and theme.py does `import customtkinter`
# plus module-level ctk.set_appearance_mode()/set_default_color_theme() calls
# -- importing NK_AVAILABLE/nk from there would drag that whole chain into
# every batch worker process just to check whether neurokit2 is installed.
try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    nk = None
    NK_AVAILABLE = False


log = logging.getLogger("ecg")

def _require_nk() -> None:
    if not NK_AVAILABLE or nk is None:
        raise RuntimeError("neurokit2 is required — pip install neurokit2")


def analyse_core(
    cleaned:     np.ndarray,
    rpeaks:      np.ndarray,
    fs:          float,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> AnalysisResults:
    """Fast core analysis: RR intervals, HR stats, time-domain HRV, beat template.

    Completes in < 2 s for any recording length.  Always run first; the other
    analyse_* functions expect its output dict to be merged into self._results.
    """
    _require_nk()
    assert nk is not None  # narrowed after _require_nk()

    def _prog(pct, msg):
        if progress_cb:
            try: progress_cb(pct, msg)
            except Exception as exc:
                log.debug("analyse_core: progress_cb raised %s — continuing", exc)

    rpeaks = np.sort(np.array(rpeaks, dtype=int).flatten())
    if len(rpeaks) < 3:
        raise ValueError(f"Too few R-peaks ({len(rpeaks)}) — need at least 3.")

    rr_ms = np.diff(rpeaks).astype(np.float64) / fs * 1000
    dur_s = float(rpeaks[-1] - rpeaks[0]) / fs   # actual recording span
    log.info("analyse_core: %d peaks, %.1f s, mean RR=%.1f ms",
             len(rpeaks), dur_s, float(rr_ms.mean()))

    _prog(5, "RR intervals…")
    rr_df = pd.DataFrame({
        "Time_s": rpeaks[1:] / fs,
        "RR_ms":  rr_ms,
        "HR_bpm": 60_000.0 / rr_ms,
    })
    rr_df = rr_df[rr_df.RR_ms.between(MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS)].copy()

    # Use rr_ms directly for HR stats — physiological filtering applied to rr_df below
    rr_clean = rr_ms[(rr_ms >= MouseECG.RR_MIN_MS) & (rr_ms <= MouseECG.RR_MAX_MS)]
    if len(rr_clean) == 0:
        rr_clean = rr_ms   # fallback: all intervals
    hr = {
        "mean": float(np.nanmean(60_000 / rr_clean)),
        "min":  float(60_000 / np.percentile(rr_clean, 98)),
        "max":  float(60_000 / np.percentile(rr_clean, 2)),
        "std":  float(np.nanstd(60_000 / rr_clean)),
        "n":     int(len(rpeaks)),      # total detected (status bar)
        "n_valid": int(len(rr_clean)),   # valid after RR bounds filter
    }

    _prog(30, "Time-domain HRV (SDNN, RMSSD…)")
    # Use only peaks whose adjacent RR intervals are within physiological bounds,
    # matching the filter already applied to rr_df.  This ensures SDNN / RMSSD /
    # pNN6 are not inflated by artifact intervals that slipped through detection.
    rr_ok_mask = (rr_ms >= MouseECG.RR_MIN_MS) & (rr_ms <= MouseECG.RR_MAX_MS)
    # rr_ms[k] = interval between rpeaks[k] and rpeaks[k+1].
    # Keep peak k if its following interval is OK, and peak k+1 if preceding is OK.
    peak_keep = np.zeros(len(rpeaks), dtype=bool)
    peak_keep[:-1] |= rr_ok_mask   # peak k keeps if rr[k] is valid
    peak_keep[1:]  |= rr_ok_mask   # peak k+1 keeps if rr[k] is valid
    rpeaks_clean = rpeaks[peak_keep]
    if len(rpeaks_clean) < 3:
        rpeaks_clean = rpeaks   # fallback: use all peaks
    hrv_time = nk.hrv_time(rpeaks_clean, sampling_rate=int(fs), show=False)

    # ── pNN6: mouse-appropriate pNNx (replaces the human-calibrated pNN50) ──
    # pNN50 is near 0 % for all healthy mice because normal vagal RR swings are
    # only 5–10 ms, far below the 50 ms threshold.  pNN6 (≈ 5 % of mean RR)
    # captures the same physiological information at the right scale.
    # Reference: Thireau et al. 2008, Am J Physiol Heart Circ Physiol 294:H977.
    try:
        rr_ms_for_pnn = np.diff(rpeaks_clean).astype(float) / fs * 1000
        rr_filt = rr_ms_for_pnn[
            (rr_ms_for_pnn >= MouseECG.RR_MIN_MS) &
            (rr_ms_for_pnn <= MouseECG.RR_MAX_MS)
        ]
        if len(rr_filt) >= 2:
            succ_diffs = np.abs(np.diff(rr_filt))
            pnn6_val = float(np.mean(succ_diffs > MouseECG.PNN_THRESHOLD) * 100)
        else:
            pnn6_val = float("nan")
        hrv_time["HRV_pNN6"] = pnn6_val
    except Exception as exc:
        log.warning("pNN6 computation failed: %s", exc)
        hrv_time["HRV_pNN6"] = float("nan")

    # Beat template — vectorised, fast even for 10 000 beats
    _prog(65, "Beat template…")
    beat_template = beat_time = beat_matrix = beat_sd = beat_corr = peak_amps = None
    try:
        # ── Cap half_win so the window never overlaps adjacent beats ─────────
        # BEAT_HALF_WIN_S = ±100 ms is safe at resting HR (~500 bpm, RR=120 ms)
        # but at 700 bpm (RR=86 ms) a ±100 ms window contains the neighbouring
        # beats, biasing the mean template and P-wave auto-calibration.
        # Cap at 45 % of the shortest observed RR to guarantee no overlap.
        fixed_half_win  = int(MouseECG.BEAT_HALF_WIN_S * fs)
        rr_samples_arr  = np.diff(rpeaks) if len(rpeaks) > 1 else np.array([fixed_half_win * 2])
        rr_min_samples  = int(np.min(rr_samples_arr[rr_samples_arr > 0])) if len(rr_samples_arr) else fixed_half_win * 2
        half_win        = max(20, min(fixed_half_win, int(rr_min_samples * 0.45)))
        if half_win < fixed_half_win:
            log.debug(
                "Beat window capped: ±%d ms (min RR=%.0f ms, was ±%d ms)",
                round(half_win / fs * 1000),
                rr_min_samples / fs * 1000,
                round(fixed_half_win / fs * 1000),
            )
        valid_rp = rpeaks[(rpeaks - half_win >= 0) & (rpeaks + half_win < len(cleaned))]
        if len(valid_rp):
            # Build full matrix in one allocation — no Python loop
            idx         = valid_rp[:, None] + np.arange(-half_win, half_win)
            beat_matrix = cleaned[idx]                      # (n_beats, 2*half_win)
            beat_time     = np.arange(-half_win, half_win) / fs * 1000
            peak_amps     = beat_matrix[:, half_win].copy() # copy before shrinking matrix

            # ── Pass 1: rough mean beat from all beats ───────────────────────
            rough_mean    = beat_matrix.mean(axis=0)

            # ── Correlation via dot-product — 50× faster than corrcoef loop ──
            # Beat quality: Pearson r with rough template. Beats with r < 0.75
            # (ectopics, artefacts, transitions) are excluded from the refined
            # template, giving a much cleaner P/Q/R/S/T reference.
            tmpl_c = rough_mean - rough_mean.mean()
            tmpl_n = np.linalg.norm(tmpl_c)
            if tmpl_n > 1e-9:
                beats_c   = beat_matrix - beat_matrix.mean(axis=1, keepdims=True)
                norms     = np.linalg.norm(beats_c, axis=1, keepdims=True)
                norms     = np.where(norms < 1e-9, 1.0, norms)
                beat_corr = (beats_c @ tmpl_c) / (norms.ravel() * tmpl_n)
            else:
                beat_corr = np.ones(len(beat_matrix))

            # ── Pass 2: refined mean beat from high-quality beats only ───────
            good_mask     = beat_corr >= 0.75
            n_good        = int(good_mask.sum())
            if n_good >= max(5, len(beat_matrix) // 10):
                # Use high-quality beats for the template
                refined_matrix = beat_matrix[good_mask]
                beat_template  = refined_matrix.mean(axis=0)
                beat_sd        = refined_matrix.std(axis=0)
                log.debug("Beat template: %d/%d high-quality beats (corr≥0.75)",
                          n_good, len(beat_matrix))
            else:
                # Fall back to all beats if too few pass quality filter
                beat_template = rough_mean
                beat_sd       = beat_matrix.std(axis=0)
                log.debug("Beat template: quality filter yielded only %d/%d — using all",
                          n_good, len(beat_matrix))

            # Re-compute beat_corr against the refined template
            tmpl_c = beat_template - beat_template.mean()
            tmpl_n = np.linalg.norm(tmpl_c)
            if tmpl_n > 1e-9:
                beats_c   = beat_matrix - beat_matrix.mean(axis=1, keepdims=True)
                norms     = np.linalg.norm(beats_c, axis=1, keepdims=True)
                norms     = np.where(norms < 1e-9, 1.0, norms)
                beat_corr = (beats_c @ tmpl_c) / (norms.ravel() * tmpl_n)

            # ── Free the full beat_matrix immediately after statistics are done.
            # On a long recording (10k beats, fs=20kHz, 0.1s window) it can reach
            # 320 MB.  Only a small ghost-trace sub-sample (~60 rows) is kept for
            # the overlay in draw_template — negligible memory, same visual result.
            n_ghost   = min(60, len(valid_rp))
            rng       = np.random.default_rng(seed=0)
            ghost_idx = np.sort(rng.choice(len(valid_rp), size=n_ghost, replace=False))
            beat_matrix = beat_matrix[ghost_idx]   # ≤60 rows kept
            log.debug("Beat template: %d beats, matrix shrunk to %d ghost rows (%.1f kB)",
                      len(valid_rp), n_ghost, beat_matrix.nbytes / 1e3)
    except Exception as exc:
        log.warning("Beat template failed: %s", exc)

    _prog(100, "Core analysis done")
    return cast(AnalysisResults, dict(
        hr=hr, rr_ms=rr_ms, rr_df=rr_df,
        hrv_time=hrv_time,
        hrv_freq=pd.DataFrame(), hrv_nonlin=pd.DataFrame(),
        intervals=pd.DataFrame({"RR_ms": rr_ms}),
        beat_template=beat_template, beat_time=beat_time,
        beat_matrix=beat_matrix,   # ≤60 rows — negligible memory
        beat_sd=beat_sd,
        beat_corr=beat_corr, peak_amps=peak_amps,
    ))


def analyse_hrv_freq(
    rpeaks:      np.ndarray,
    fs:          float,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> "pd.DataFrame":
    """Frequency-domain HRV (VLF, LF, HF, LF/HF) with mouse-specific bands.

    NeuroKit2's default bands (LF: 0.04–0.15 Hz, HF: 0.15–0.4 Hz) are tuned
    for humans.  Mouse HRV uses completely different ranges (Thireau et al.
    2008, Baudrie et al. 2007):
        VLF : 0.000 – 0.400 Hz
        LF  : 0.400 – 1.500 Hz   (baroreflex / sympathovagal oscillations)
        HF  : 1.500 – 5.000 Hz   (respiratory sinus arrhythmia, breathing at 2–4 Hz)

    The minimum recommended recording length for stable spectral estimates is:
        ≥ 5 × (1 / VLF_lo) ≈ impossible → use ≥ 5 / LF_lo ≈ 12.5 s (≥ ~100 beats)

    Parameters
    ----------
    rpeaks      : R-peak sample indices (from polarity-corrected filtered signal)
    fs          : Sampling frequency in Hz
    progress_cb : Optional progress callback (pct: int, msg: str) → None
    """
    _require_nk()
    assert nk is not None  # narrowed after _require_nk()

    # Minimum beat count for any meaningful spectral estimate
    MIN_BEATS_SPECTRAL = 60   # ~7–15 s of mouse ECG at typical HR
    if len(rpeaks) < MIN_BEATS_SPECTRAL:
        log.warning(
            "analyse_hrv_freq: only %d beats — need ≥ %d for spectral HRV",
            len(rpeaks), MIN_BEATS_SPECTRAL,
        )
        return pd.DataFrame()

    if progress_cb:
        progress_cb(10, "Frequency-domain HRV (mouse bands: LF 0.4–1.5 Hz, HF 1.5–5.0 Hz)…")

    try:
        result = nk.hrv_frequency(
            rpeaks,
            sampling_rate=int(fs),
            show=False,
            normalize=True,
            # ── Mouse-specific HRV frequency bands ─────────────────────────────
            # Reference: Thireau et al. (2008) Am J Physiol Heart Circ Physiol 294:H977
            #            Baudrie et al. (2007) Am J Physiol Regul Integr Comp Physiol 293:R306
            ulf=(0.000, 0.000),   # ULF not relevant for acute recordings — set to zero
            vlf=(0.000, 0.400),   # Very-low-frequency (vasomotor / thermoregulatory)
            lf=(0.400,  1.500),   # Low-frequency  (baroreflex / sympathovagal balance)
            hf=(1.500,  5.000),   # High-frequency (respiratory sinus arrhythmia, 2–4 Hz)
            vhf=(5.000, 20.000),  # Very-high-frequency (harmonics / noise floor)
        )
        if progress_cb:
            progress_cb(100, "Frequency HRV done")
        return result
    except Exception as exc:
        log.warning("analyse_hrv_freq failed: %s", exc)
        return pd.DataFrame()

def analyse_hrv_nonlinear(
    rpeaks: np.ndarray,
    fs: float,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    max_beats: int = 1000
) -> pd.DataFrame:
    """Non-linear HRV via NeuroKit2: Poincaré SD1/SD2, DFA α1/α2, ApEn/SampEn,
    Porta index, and the rest of nk.hrv_nonlinear()'s standard metric set.

    This used to be a bespoke implementation that only ever returned
    HRV_SampEn (computed as the entropy of individual BEAT WAVEFORMS,
    averaged across beats) and an ad hoc nearest-neighbour distance proxy
    (HRV_NNDistLog) that no caller ever displayed or exported. Meanwhile
    every consumer of this DataFrame -- the Poincaré plot title, the
    non-linear text report, Excel/Prism export, the radar chart -- reads
    HRV_SD1, HRV_SD2, HRV_DFA_alpha1, HRV_DFA_alpha2, HRV_ApEn, HRV_PI,
    none of which that bespoke version ever populated, so those fields
    silently showed "—" on every run.

    HRV_SampEn's definition also changes here: nk.hrv_nonlinear() computes
    it on the RR-INTERVAL series (the standard textbook Sample Entropy of
    HRV, and what PARAM_INFO's own documented reference range of 0.5-2.5
    assumes), not on beat-waveform morphology -- the old per-beat-waveform
    values (~0.02 for a typical recording) were far outside that
    documented range because they were never measuring the same thing.

    Tested down to 5 beats with no crashes; nk degrades individual columns
    to NaN (e.g. HRV_DFA_alpha1 needs enough beats for a stable fit) rather
    than raising.

    Args:
        rpeaks: R-peak sample indices (ndarray)
        fs: Sampling rate (Hz)
        progress_cb: Progress callback (pct, message) -> None
        max_beats: Cap on beats passed to nk (keeps runtime bounded on long recordings)

    Returns:
        DataFrame with nk.hrv_nonlinear()'s standard non-linear HRV metrics.
    """
    _require_nk()
    assert nk is not None  # narrowed after _require_nk()

    if progress_cb:
        progress_cb(0, "Non-linear HRV: preparing R-peaks…")

    n_rpeaks_orig = len(rpeaks)
    if n_rpeaks_orig > max_beats:
        rpeaks = rpeaks[:max_beats]
        log.warning(f"Limiting to {max_beats} beats (was {n_rpeaks_orig})")

    if progress_cb:
        progress_cb(20, "Computing SD1/SD2, DFA, SampEn/ApEn… (may take 30 s+)")

    try:
        result = nk.hrv_nonlinear(rpeaks, sampling_rate=int(fs), show=False)
    except Exception as exc:
        log.warning("analyse_hrv_nonlinear failed: %s", exc)
        return pd.DataFrame()

    if progress_cb:
        progress_cb(100, "Non-linear HRV done")

    return result


def analyse_intervals(
    cleaned:           "np.ndarray",
    rpeaks:            "np.ndarray",
    fs:                float,
    rr_ms:             "np.ndarray",
    max_beats:         int = 0,          # kept for API compat, no longer used
    progress_cb:       "Optional[Callable[[int, str], None]]" = None,
    wave_template:     "Optional[WaveTemplate]" = None,
    beat_template:     "Optional[np.ndarray]"   = None,
    beat_time_arr:     "Optional[np.ndarray]"   = None,
    permissive_bounds: bool = False,
) -> "pd.DataFrame":
    """Template-guided per-beat ECG interval delineation.  No NeuroKit2.

    Detects P/Q/S/T landmarks directly on each beat using
    ``detect_waves_on_beat``, guided by the WaveTemplate search windows.
    All beats are processed (no sub-sampling).

    Parameters
    ----------
    cleaned   : bandpass-filtered ECG (same array used by analyse_core).
    rpeaks    : sample indices of all accepted R peaks.
    fs        : sampling frequency (Hz).
    rr_ms     : RR interval array (len = len(rpeaks)-1).
    max_beats : deprecated — ignored.  Present only for backward compat.
    progress_cb : optional progress(pct, message) callback.
    wave_template : search-window template; defaults are used when None.
    beat_template, beat_time_arr : pre-computed mean beat from analyse_core
        (optional — only used to auto-update the template after the run).
    permissive_bounds : if True, relax physiological bounds by 2×.

    Returns
    -------
    pd.DataFrame with columns RR_ms, PR_ms, QRS_ms, QT_ms, QTc_ms,
    R_peak_s, plus per-wave absolute (``_s``) and relative (``_ms``) positions,
    beat_idx, and an ``accepted`` boolean column for the verifier UI.
    """
    if max_beats and max_beats > 0:
        log.debug("analyse_intervals: max_beats=%d is no longer used (all beats processed)",
                  max_beats)

    def _prog(p: int, m: str) -> None:
        if progress_cb:
            progress_cb(p, m)

    if wave_template is None:
        wave_template = WaveTemplate.load()

    rpeaks_all = np.sort(np.array(rpeaks, dtype=int).flatten())
    n_beats    = len(rpeaks_all)
    if n_beats < 3:
        return pd.DataFrame({"RR_ms": rr_ms})

    # ── Beat window ──────────────────────────────────────────────────────
    fixed_hw  = int(MouseECG.BEAT_HALF_WIN_S * fs)
    rr_samp   = np.diff(rpeaks_all)
    rr_min_s  = int(rr_samp.min()) if len(rr_samp) else fixed_hw * 2
    half_win  = max(20, min(fixed_hw, int(rr_min_s * 0.45)))
    bt_ms     = np.arange(-half_win, half_win) / fs * 1000

    mask_valid = ((rpeaks_all - half_win >= 0) &
                  (rpeaks_all + half_win < len(cleaned)))
    valid_rp   = rpeaks_all[mask_valid]
    n_valid    = len(valid_rp)

    if n_valid < 2:
        log.warning("analyse_intervals: too few valid beats (%d)", n_valid)
        return pd.DataFrame({"RR_ms": rr_ms})

    _prog(5, f"Building beat matrix ({n_valid} beats)…")

    # Vectorised beat extraction
    idx_mat  = valid_rp[:, None] + np.arange(-half_win, half_win)
    beat_mat = cleaned[idx_mat].astype(float)

    # ── Per-beat detection ───────────────────────────────────────────────
    _prog(10, "Detecting P/Q/S/T landmarks per beat…")

    rf       = valid_rp.astype(float)
    p_onset  = np.full(n_valid, np.nan)
    p_peak   = np.full(n_valid, np.nan)
    q_peak   = np.full(n_valid, np.nan)
    s_peak   = np.full(n_valid, np.nan)
    j_peak   = np.full(n_valid, np.nan)
    t_peak   = np.full(n_valid, np.nan)
    t_offset = np.full(n_valid, np.nan)

    tick = max(1, n_valid // 20)
    for i in range(n_valid):
        det = detect_waves_on_beat(beat_mat[i], bt_ms, wave_template)

        def _ms_to_samp(v: float) -> float:
            return valid_rp[i] + v / 1000.0 * fs if np.isfinite(v) else np.nan

        p_onset [i] = _ms_to_samp(det["P_onset"])
        p_peak  [i] = _ms_to_samp(det["P_peak"])
        q_peak  [i] = _ms_to_samp(det["Q_peak"])
        s_peak  [i] = _ms_to_samp(det["S_peak"])
        j_peak  [i] = _ms_to_samp(det["J_peak"])
        t_peak  [i] = _ms_to_samp(det["T_peak"])
        t_offset[i] = _ms_to_samp(det["T_offset"])

        if (i + 1) % tick == 0:
            _prog(10 + int(72 * (i + 1) / n_valid),
                  f"Detecting landmarks… {i + 1}/{n_valid}")

    _prog(82, "Computing intervals…")

    # ── Interval arithmetic ──────────────────────────────────────────────
    n = n_valid - 1
    def _t(a: "np.ndarray") -> "np.ndarray":
        return a[:n]

    r_pos_n    = valid_rp[:n].astype(float)
    p_onset_n  = _t(p_onset);  p_peak_n   = _t(p_peak)
    q_peak_n   = _t(q_peak);   s_peak_n   = _t(s_peak)
    j_peak_n   = _t(j_peak)
    t_peak_n   = _t(t_peak);   t_offset_n = _t(t_offset)

    rr_arr  = np.diff(valid_rp[:n + 1]).astype(float) / fs * 1000
    pr_end  = np.where(np.isfinite(q_peak_n), q_peak_n, r_pos_n)
    pr_raw  = (pr_end     - p_onset_n) / fs * 1000
    qrs_raw = (s_peak_n   - q_peak_n)  / fs * 1000
    qt_raw  = (t_offset_n - r_pos_n)   / fs * 1000

    # ── Physiological bounds ─────────────────────────────────────────────
    _m = 2.0 if permissive_bounds else 1.0
    def _clip(arr: "np.ndarray", lo: float, hi: float) -> "np.ndarray":
        out = arr.copy().astype(float)
        out[(out < lo) | (out > hi)] = np.nan
        return out

    pr_clean  = _clip(pr_raw,  MouseECG.PR_ABS_MIN  / _m, MouseECG.PR_ABS_MAX  * _m)
    qrs_clean = _clip(qrs_raw, MouseECG.QRS_ABS_MIN / _m, MouseECG.QRS_ABS_MAX * _m)
    qt_clean  = _clip(qt_raw,  MouseECG.QT_ABS_MIN  / _m, MouseECG.QT_ABS_MAX  * _m)
    rr_clean  = _clip(rr_arr,  MouseECG.RR_MIN_MS   / _m, MouseECG.RR_MAX_MS   * _m)

    rr_s      = np.clip(rr_clean, MouseECG.RR_MIN_MS, None) / 1000.0
    qtc_clean = _clip(qt_clean / (rr_s ** (1.0 / 3.0)),
                      MouseECG.QTC_ABS_MIN, MouseECG.QTC_ABS_MAX)

    # ── DataFrame ────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "RR_ms":      rr_clean,
        "PR_ms":      pr_clean,
        "QRS_ms":     qrs_clean,
        "QT_ms":      qt_clean,
        "QTc_ms":     qtc_clean,
        "R_peak_s":   r_pos_n    / fs,
        "P_onset_s":  p_onset_n  / fs,
        "P_peak_s":   p_peak_n   / fs,
        "Q_peak_s":   q_peak_n   / fs,
        "S_peak_s":   s_peak_n   / fs,
        "J_peak_s":   j_peak_n   / fs,
        "T_peak_s":   t_peak_n   / fs,
        "T_offset_s": t_offset_n / fs,
        # ms-from-R columns for the verifier UI (drag-to-adjust)
        "P_onset_ms":  (p_onset_n  - r_pos_n) / fs * 1000,
        "P_peak_ms":   (p_peak_n   - r_pos_n) / fs * 1000,
        "Q_peak_ms":   (q_peak_n   - r_pos_n) / fs * 1000,
        "S_peak_ms":   (s_peak_n   - r_pos_n) / fs * 1000,
        "J_peak_ms":   (j_peak_n   - r_pos_n) / fs * 1000,
        "T_peak_ms":   (t_peak_n   - r_pos_n) / fs * 1000,
        "T_offset_ms": (t_offset_n - r_pos_n) / fs * 1000,
        "beat_idx":    np.arange(n, dtype=int),
        "accepted":    np.ones(n, dtype=bool),
    })

    # NaN-out wave positions for out-of-bounds intervals
    df.loc[np.isnan(pr_raw)  | (pr_raw  < MouseECG.PR_ABS_MIN)  | (pr_raw  > MouseECG.PR_ABS_MAX),
           ["P_onset_s","P_peak_s","P_onset_ms","P_peak_ms"]] = np.nan
    df.loc[np.isnan(qrs_raw) | (qrs_raw < MouseECG.QRS_ABS_MIN) | (qrs_raw > MouseECG.QRS_ABS_MAX),
           ["Q_peak_s","S_peak_s","Q_peak_ms","S_peak_ms"]] = np.nan
    df.loc[np.isnan(qt_raw)  | (qt_raw  < MouseECG.QT_ABS_MIN)  | (qt_raw  > MouseECG.QT_ABS_MAX),
           ["T_peak_s","T_offset_s","T_peak_ms","T_offset_ms"]] = np.nan

    n_complete = int((~df[["PR_ms","QRS_ms","QT_ms"]].isna().any(axis=1)).sum())
    log.info("analyse_intervals: %d beats, %d complete (%.0f%%)",
             n, n_complete, 100.0 * n_complete / max(n, 1))

    # ── Auto-update template from this run ───────────────────────────────
    if not wave_template.confirmed and n_complete >= 10:
        _ms_cols = {
            "P_onset": "P_onset_ms", "P_peak": "P_peak_ms",
            "Q_peak":  "Q_peak_ms",  "S_peak": "S_peak_ms",
            "T_peak":  "T_peak_ms",  "T_offset": "T_offset_ms",
        }
        updated = {}
        for wk, col in _ms_cols.items():
            valid = np.asarray(df[col].dropna(), dtype=float)
            if len(valid) < 5:
                updated[wk] = wave_template.landmarks.get(wk, WaveTemplate.DEFAULTS[wk])
                continue
            med    = float(np.median(valid))
            spread = max(4.0,
                         float(np.percentile(valid, 90) -
                               np.percentile(valid, 10)) * 0.75)
            updated[wk] = (round(med, 1), round(spread, 1))
        wave_template.landmarks = updated
        wave_template.source    = "auto-updated"
        try:
            wave_template.save()
            log.info("analyse_intervals: template auto-updated (%d complete beats)",
                     n_complete)
        except Exception as save_exc:
            log.warning("template auto-save failed: %s", save_exc)

    _prog(100, f"Done — {n_complete}/{n} complete beats")
    return df



# ---------------------------------------------------------------------------
# run_full_analysis — kept only for backward compatibility with external scripts.
# It is NOT called anywhere inside this application.  New code should call
# analyse_core / analyse_hrv_freq / analyse_hrv_nonlinear / analyse_intervals
# directly so each step can be run on demand and progress can be reported.
# This function will be removed in a future version.
# ---------------------------------------------------------------------------
def run_full_analysis(
    cleaned:     np.ndarray,
    rpeaks:      np.ndarray,
    fs:          float,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> AnalysisResults:
    """Backward-compatible wrapper: runs core + freq + nonlinear + intervals.

    .. deprecated::
        Use the individual ``analyse_*`` functions instead.
    """
    import warnings as _w
    _w.warn(
        "run_full_analysis() is deprecated and will be removed in a future version.  "
        "Call analyse_core(), analyse_hrv_freq(), analyse_hrv_nonlinear(), and "
        "analyse_intervals() individually.",
        DeprecationWarning,
        stacklevel=2,
    )
    def _sub(lo: int, hi: int, msg: str) -> "Optional[Callable[[int, str], None]]":
        if progress_cb:
            _cb = progress_cb  # capture narrowed non-None reference
            def cb(p: int, m: str) -> None:
                _cb(lo + int((hi - lo) * p / 100), m)
            return cb
        return None

    r = analyse_core(cleaned, rpeaks, fs, _sub(0, 40, "core"))
    r["hrv_freq"]   = analyse_hrv_freq(rpeaks, fs, _sub(40, 55, "freq"))
    r["hrv_nonlin"] = analyse_hrv_nonlinear(rpeaks, fs, _sub(55, 75, "nonlin"))
    r["intervals"]  = analyse_intervals(cleaned, rpeaks, fs, r["rr_ms"],  # type: ignore[typeddict-item]
                                        progress_cb=_sub(75, 100, "intervals"))
    return r


# ════════════════════════════════════════════════════════════
#  FIGURE / AXES HELPERS
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
#  SEGMENT COMPARISON
# ════════════════════════════════════════════════════════════

def compute_segment_stats(
    signal:  np.ndarray,
    rpeaks:  np.ndarray,
    fs:      float,
    t_lo:    float,
    t_hi:    float,
    label:   str = "Segment",
    lf_band: "tuple[float, float]" = (0.40, 1.50),
    hf_band: "tuple[float, float]" = (1.50, 5.00),
) -> dict:
    """Compute HRV and cardiac stats for a time window [t_lo, t_hi] seconds.

    Parameters
    ----------
    signal  : Full ECG signal array (float64).
    rpeaks  : Full R-peak sample indices.
    fs      : Sampling rate Hz.
    t_lo    : Segment start (seconds, inclusive).
    t_hi    : Segment end   (seconds, exclusive).
    label   : Display name for this segment.
    lf_band : (lo, hi) Hz LF band for frequency HRV.
    hf_band : (lo, hi) Hz HF band for frequency HRV.

    Returns
    -------
    dict with keys: label, n_beats, duration_s, hr_mean, hr_sd, hr_min,
    hr_max, rr_mean, rr_sd, rr_cv, sdnn, rmssd, pnn6, pnn50,
    lf_nu, hf_nu, lf_hf, sd1, sd2, sd_ratio,
    rpeaks_seg, rr_ms, t_rr, error.
    """
    _NAN = float("nan")
    _EMPTY: dict = {
        "label":        label,
        "n_beats":      0,
        "duration_s":   float(t_hi - t_lo),
        "coverage_pct": 0.0,
        "hr_mean":  _NAN, "hr_sd":    _NAN, "hr_min":  _NAN, "hr_max":   _NAN,
        "rr_mean":  _NAN, "rr_sd":    _NAN, "rr_min":  _NAN, "rr_max":   _NAN,
        "rr_cv":    _NAN, "sdnn":     _NAN, "rmssd":   _NAN,
        "pnn6":     _NAN, "pnn50":    _NAN,
        "lf_nu":    _NAN, "hf_nu":    _NAN, "lf_hf":   _NAN,
        "sd1":      _NAN, "sd2":      _NAN, "sd_ratio": _NAN,
        "rpeaks_seg": np.array([], dtype=int),
        "rr_ms":      np.array([], dtype=float),
        "t_rr":       np.array([], dtype=float),
        "error":      None,
    }

    # ── 1. Select peaks inside [t_lo, t_hi] ──────────────────────────────
    i_lo = int(t_lo * fs)
    i_hi = int(t_hi * fs)
    mask = (rpeaks >= i_lo) & (rpeaks <= i_hi)
    rp   = rpeaks[mask].copy()
    n    = len(rp)

    if n < 10:
        out = dict(_EMPTY); out["error"] = f"Only {n} beats in window (need ≥ 10)"; return out

    # ── 2. RR intervals & physiological filter ───────────────────────────
    rr_all = np.diff(rp).astype(float) / fs * 1000.0  # ms, length n-1
    t_rr   = rp[1:].astype(float) / fs                 # time of each interval end

    rr_mask = (rr_all >= MouseECG.RR_MIN_MS) & (rr_all <= MouseECG.RR_MAX_MS)
    rr_ok   = rr_all[rr_mask]
    t_ok    = t_rr[rr_mask]

    if len(rr_ok) < 8:
        out = dict(_EMPTY); out["error"] = "Too many artefact beats after physiological filter"; return out

    hr_all = 60_000.0 / rr_ok

    # ── 3. Time-domain HRV ───────────────────────────────────────────────
    sdnn  = float(np.std(rr_ok, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ok) ** 2))) if len(rr_ok) > 1 else _NAN
    diff_rr = np.abs(np.diff(rr_ok))
    pnn6    = float(100.0 * np.mean(diff_rr > 6.0))  if len(diff_rr) else _NAN
    pnn50   = float(100.0 * np.mean(diff_rr > 50.0)) if len(diff_rr) else _NAN

    # ── 4. Frequency-domain HRV (Welch on cubic-spline interpolated RR) ──
    lf_nu = hf_nu = lf_hf_r = _NAN
    try:
        from scipy.interpolate import CubicSpline
        from scipy.signal import welch as _welch
        if len(t_ok) >= 16 and len(rr_ok) >= 16:
            cs     = CubicSpline(t_ok, rr_ok)
            fs_i   = 20.0           # interpolation rate (Nyquist >> 5 Hz)
            t_unif = np.arange(t_ok[0], t_ok[-1], 1.0 / fs_i)
            if len(t_unif) >= 32:
                rr_i     = cs(t_unif)
                rr_i    -= rr_i.mean()
                nperseg  = min(len(t_unif), int(fs_i * 60))
                freqs, pxx = _welch(rr_i, fs=fs_i, nperseg=nperseg,
                                     noverlap=nperseg // 2, detrend="linear")
                def _band_power(lo: float, hi: float) -> float:
                    m = (freqs >= lo) & (freqs < hi)
                    return float(trapz(pxx[m], freqs[m])) if m.any() else 0.0
                lf_p  = _band_power(*lf_band)
                hf_p  = _band_power(*hf_band)
                total = lf_p + hf_p
                if total > 1e-12:
                    lf_nu    = float(100.0 * lf_p / total)
                    hf_nu    = float(100.0 * hf_p / total)
                    lf_hf_r  = float(lf_p / hf_p) if hf_p > 1e-12 else _NAN
    except Exception as exc:
        log.debug("compute_segment_stats freq HRV: %s", exc)

    # ── 5. Poincaré SD1 / SD2 ────────────────────────────────────────────
    sd1 = sd2 = sd_ratio = _NAN
    try:
        if len(rr_ok) > 2:
            rr1 = rr_ok[:-1]; rr2 = rr_ok[1:]
            sd1      = float(np.std((rr2 - rr1) / np.sqrt(2), ddof=1))
            sd2      = float(np.std((rr2 + rr1) / np.sqrt(2), ddof=1))
            sd_ratio = sd1 / sd2 if sd2 > 1e-9 else _NAN
    except Exception:
        pass

    return {
        "label":        label,
        "n_beats":      n,
        "duration_s":   float(t_hi - t_lo),
        "coverage_pct": float(100.0 * n / max(1, len(rpeaks))),
        "hr_mean":  float(np.mean(hr_all)),   "hr_sd":  float(np.std(hr_all, ddof=1)),
        "hr_min":   float(np.min(hr_all)),    "hr_max": float(np.max(hr_all)),
        "rr_mean":  float(np.mean(rr_ok)),    "rr_sd":  float(np.std(rr_ok, ddof=1)),
        "rr_min":   float(np.min(rr_ok)),     "rr_max": float(np.max(rr_ok)),
        "rr_cv":    float(100.0 * np.std(rr_ok) / np.mean(rr_ok)),
        "sdnn":     sdnn,   "rmssd":    rmssd,
        "pnn6":     pnn6,   "pnn50":    pnn50,
        "lf_nu":    lf_nu,  "hf_nu":    hf_nu,  "lf_hf":    lf_hf_r,
        "sd1":      sd1,    "sd2":      sd2,     "sd_ratio": sd_ratio,
        "rpeaks_seg": rp,
        "rr_ms":      rr_ok,
        "t_rr":       t_ok,
        "error":      None,
    }


