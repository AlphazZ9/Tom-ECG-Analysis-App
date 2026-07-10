"""
ecg.core.wave_template
──────────────────────
Beat-template calibration and P/Q/S/T wave delineation.
No UI imports — pure NumPy / SciPy.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.signal import find_peaks

from models import MouseECG

if TYPE_CHECKING:
    from app import ECGApp

WAVE_TEMPLATE_PATH: Path = Path.home() / ".ecg_wave_template.json"
_WAVE_TEMPLATE_PATH_LEGACY: Path = Path.home() / ".ecg_wave_template.pkl"

log = logging.getLogger("ecg")

class WaveTemplate:
    """User-defined P/Q/R/S/T landmark positions relative to the R peak (ms).

    All times are stored in milliseconds relative to R = 0 ms.
    Negative values are before R (P, Q), positive after (S, T).

    These positions are used as Gaussian priors in ``analyse_intervals``
    to constrain NeuroKit2's delineation search window to a physiologically
    plausible neighbourhood, greatly reducing false detections in noisy signals.

    Default values are based on mouse physiology:
        Thireau et al. (2008) Am J Physiol Heart Circ Physiol 294:H977
        Gehrmann et al. (2000) J Interv Card Electrophysiol 4:469
    """

    # ── Default windows (mouse, ms relative to R) ─────────────────────
    DEFAULTS: "dict[str, tuple[float, float]]" = {
        # wave_key: (center_ms, half_window_ms)
        # Windows deliberately overlap R=0 so narrow Q/S deflections aren't missed
        "P_peak":   (-45.0, 28.0),   # P wave:   ~45 ms before R,  ±28 ms search
        "Q_peak":   ( -5.0, 12.0),   # Q wave:   ~5 ms before R,   ±12 ms (spans -17→+7)
        "S_peak":   (  5.0, 12.0),   # S wave:   ~5 ms after  R,   ±12 ms (spans -7→+17)
        # J wave (early repolarization notch / Osborn-like hump after S, common in mice)
        # Must be detected BEFORE T so T search can skip over it
        "J_peak":   ( 15.0,  8.0),   # J wave:   ~15 ms after R,   ±8 ms  (spans 7→23)
        "T_peak":   ( 52.0, 26.0),   # T wave:   ~52 ms after R,   ±26 ms (starts after J)
        "P_onset":  (-62.0, 20.0),   # P onset:  ~62 ms before R,  ±20 ms search
        "T_offset": ( 82.0, 22.0),   # T offset: ~82 ms after  R,  ±22 ms search
    }

    def __init__(self) -> None:
        self.landmarks: "dict[str, tuple[float, float]]" = dict(self.DEFAULTS)
        self.confirmed: bool  = False   # True after user saves via the editor
        self.source:    str   = "default"
        self.created_at: str  = ""

    # ── Persistence ────────────────────────────────────────────────────

    def save(self) -> None:
        """Write this template to ``WAVE_TEMPLATE_PATH`` as JSON.

        Tuples are serialised as two-element lists (JSON has no tuple type);
        ``load()`` converts them back.
        """
        self.created_at = datetime.now().isoformat()
        data = {
            "landmarks":  {k: list(v) for k, v in self.landmarks.items()},
            "confirmed":  self.confirmed,
            "source":     self.source,
            "created_at": self.created_at,
        }
        with open(WAVE_TEMPLATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        log.info("WaveTemplate saved → %s", WAVE_TEMPLATE_PATH)

    @classmethod
    def load(cls) -> "WaveTemplate":
        """Load template from disk, falling back to defaults on any error.

        Migrates a legacy ``.pkl`` file to JSON on first run after upgrade.
        """
        obj = cls()

        # ── Primary path: JSON ──────────────────────────────────────────
        if WAVE_TEMPLATE_PATH.exists():
            try:
                with open(WAVE_TEMPLATE_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # landmarks: {key: [center, half_window]} → tuple
                obj.landmarks  = {
                    k: (float(v[0]), float(v[1]))
                    for k, v in data.get("landmarks", {}).items()
                }
                obj.confirmed  = bool(data.get("confirmed",  False))
                obj.source     = str( data.get("source",     "file"))
                obj.created_at = str( data.get("created_at", ""))
                log.info("WaveTemplate loaded (source=%s, confirmed=%s)",
                         obj.source, obj.confirmed)
                return obj
            except Exception as exc:
                log.warning("WaveTemplate.load (JSON) failed: %s — trying legacy", exc)

        # ── Fallback: migrate legacy pickle ─────────────────────────────
        if _WAVE_TEMPLATE_PATH_LEGACY.exists():
            try:
                with open(_WAVE_TEMPLATE_PATH_LEGACY, "rb") as fh:
                    legacy = pickle.load(fh)
                obj.__dict__.update(legacy)
                obj.save()                          # write JSON
                _WAVE_TEMPLATE_PATH_LEGACY.unlink() # remove old pickle
                log.info("WaveTemplate migrated from pickle → JSON (%s)",
                         WAVE_TEMPLATE_PATH)
                return obj
            except Exception as exc:
                log.warning("WaveTemplate legacy migration failed: %s — using defaults", exc)

        return obj

    def reset_defaults(self) -> None:
        self.landmarks  = dict(self.DEFAULTS)
        self.confirmed  = False
        self.source     = "default"

    # ── Search window helpers ─────────────────────────────────────────

    def search_window(self, key: str, fs: float
                      ) -> "tuple[int, int]":
        """Return (lo_samp, hi_samp) offset from R peak for *key*.

        Offsets are in samples; lo_samp may be negative (before R).
        """
        center_ms, half_ms = self.landmarks.get(key, self.DEFAULTS[key])
        lo = int((center_ms - half_ms) / 1000 * fs)
        hi = int((center_ms + half_ms) / 1000 * fs)
        return lo, hi

    def summary(self) -> str:
        lines = ["Wave template landmarks (ms rel. to R=0):"]
        for k, (c, h) in self.landmarks.items():
            lines.append(f"  {k:<12}  center={c:+.1f} ms   window=[{c-h:+.1f}, {c+h:+.1f}]")
        lines.append(f"  confirmed={self.confirmed}  source={self.source}")
        return "\n".join(lines)

    # ── Auto-calibration from real beat data ─────────────────────────────────

    def calibrate_from_beat(self, beat_time: "np.ndarray",
                             mean_beat: "np.ndarray") -> None:
        """Detect P/Q/R/S/T positions from the *mean_beat* signal.

        beat_time : 1-D array of times in ms relative to R=0 (negative before R)
        mean_beat : normalised mean beat signal (same length as beat_time)

        Algorithm (mouse-specific):
        1. Q and S are detected as the minimum before/after R in a narrow window.
        2. P peak is the maximum in a wide pre-R window outside the QRS complex.
        3. T peak is detected as the largest absolute extremum after QRS.
        4. P onset uses a slope (derivative) threshold — more robust than a simple
           zero-crossing, which fails when the baseline doesn't return to exactly 0.
        5. T offset uses the same slope approach: the signal is considered to have
           returned to baseline when its absolute derivative drops below a noise
           floor and the amplitude is close to the TP segment level.
        6. Half-windows are set to 80% of the distance to the nearest landmark
           so delineation searches stay well inside each wave without being too tight.

        After calling this method:
        • self.landmarks is updated with calibrated positions
        • self.source = "auto-calibrated"
        • self.confirmed = False  (user must still review and Save in the editor)
        """
        bt  = np.asarray(beat_time,  dtype=float)
        mb  = np.asarray(mean_beat,  dtype=float)

        dt_ms = float(bt[1] - bt[0]) if len(bt) > 1 else 0.5  # time step in ms

        def _idx(t_ms: float) -> int:
            return int(np.argmin(np.abs(bt - t_ms)))

        def _window(lo_ms: float, hi_ms: float) -> "tuple[int, int]":
            i0 = max(0, _idx(lo_ms))
            i1 = min(len(mb), _idx(hi_ms) + 1)
            return i0, i1

        def _peak_in(lo_ms: float, hi_ms: float, polarity: int = 1) -> "Optional[float]":
            """Time (ms) of max/min of mean_beat in [lo_ms, hi_ms].  None if empty."""
            i0, i1 = _window(lo_ms, hi_ms)
            if i1 <= i0:
                return None
            seg = mb[i0:i1]
            local = int(np.argmax(seg) if polarity == 1 else np.argmin(seg))
            return float(bt[i0 + local])

        def _baseline_level(from_ms: float, to_ms: float) -> float:
            """Robust baseline estimate from an isoelectric segment."""
            i0, i1 = _window(from_ms, to_ms)
            if i1 <= i0:
                return 0.0
            return float(np.median(mb[i0:i1]))

        def _slope_onset(t_peak_ms: float, search_ms: float = 25.0) -> float:
            """P-onset via max-slope tangent intersection.

            Finds the point of maximum upslope on the leading edge of the P wave,
            draws a tangent at that point, and returns where it intersects the
            isoelectric baseline.  This is the standard clinical P-onset method
            (Pan & Tompkins-style) and is robust to baseline wander.
            """
            i_pk = _idx(t_peak_ms)
            i_lo = max(0, _idx(t_peak_ms - search_ms))
            if i_pk <= i_lo + 2:
                return t_peak_ms - search_ms * 0.4
            seg    = mb[i_lo:i_pk + 1]
            slope  = np.gradient(seg, dt_ms)
            # Find the steepest rising slope on the leading edge (before peak)
            i_max_slope = int(np.argmax(slope))
            t_ms_slope  = float(bt[i_lo + i_max_slope])
            amp_slope   = float(mb[i_lo + i_max_slope])
            s           = float(slope[i_max_slope])
            if abs(s) < 1e-9:
                return float(bt[i_lo])
            # Baseline from TP segment (50-30 ms before P onset search start)
            baseline = _baseline_level(t_peak_ms - search_ms - 40,
                                        t_peak_ms - search_ms - 10)
            # Tangent: y - amp_slope = s*(t - t_ms_slope)  → t where y = baseline
            t_onset = t_ms_slope + (baseline - amp_slope) / s
            # Clamp to the search window
            return float(np.clip(t_onset, float(bt[i_lo]), t_peak_ms - 1.0))

        def _slope_offset(t_peak_ms: float, search_ms: float = 30.0,
                           t_wave_sign: int = 1) -> float:
            """T-offset via max-slope tangent intersection on the trailing edge.

            Mirrors _slope_onset but applied to the downslope after the T peak.
            Works for both positive and negative T waves (sign parameter).
            """
            i_pk = _idx(t_peak_ms)
            i_hi = min(len(mb) - 1, _idx(t_peak_ms + search_ms))
            if i_hi <= i_pk + 2:
                return t_peak_ms + search_ms * 0.4
            seg    = mb[i_pk:i_hi + 1]
            slope  = np.gradient(seg, dt_ms)
            # Steepest slope on the trailing edge (sign depends on T polarity)
            if t_wave_sign > 0:
                # Positive T: trailing edge falls → most negative slope
                i_max_slope = int(np.argmin(slope))
            else:
                # Negative T: trailing edge rises → most positive slope
                i_max_slope = int(np.argmax(slope))
            t_ms_slope  = float(bt[i_pk + i_max_slope])
            amp_slope   = float(mb[i_pk + i_max_slope])
            s           = float(slope[i_max_slope])
            if abs(s) < 1e-9:
                return float(bt[i_hi])
            # Baseline from TP segment of the NEXT beat: last 20 ms of window
            baseline = _baseline_level(t_peak_ms + search_ms - 20,
                                        t_peak_ms + search_ms)
            t_offset = t_ms_slope + (baseline - amp_slope) / s
            return float(np.clip(t_offset, t_peak_ms + 1.0, float(bt[i_hi])))

        # ── R at t=0 ─────────────────────────────────────────────────────────
        # ── Q: minimum just before R in a narrow window (-35 → -1 ms) ────────
        q_t = _peak_in(-35.0, -1.0, polarity=-1)
        q_t = q_t if q_t is not None else -10.0

        # ── S: minimum just after R in a narrow window (+1 → +35 ms) ─────────
        s_t = _peak_in(1.0, 35.0, polarity=-1)
        s_t = s_t if s_t is not None else 12.0

        # ── P peak: positive max before Q in wide search (-100 → q_t-5 ms) ───
        p_lo = max(float(bt[0]) + 2.0, -100.0)
        p_t  = _peak_in(p_lo, q_t - 5.0, polarity=1)
        p_t  = p_t if p_t is not None else -40.0

        # ── J wave: positive hump right after S (early repolarization, common in mice) ──
        j_lo = s_t + 2.0
        j_hi = min(s_t + 22.0, float(bt[-1]) - 5.0)
        j_t: "Optional[float]" = None
        if j_hi > j_lo:
            j_t = _peak_in(j_lo, j_hi, polarity=1)

        # ── T peak: largest absolute extremum after J wave (or after S+gap) ─────
        # Starting T search after J avoids confusing the J hump for the T wave
        t_lo_j = (j_t + 5.0) if j_t is not None else (s_t + 8.0)
        t_lo = max(s_t + 3.0, t_lo_j)
        t_hi = min(float(bt[-1]) - 5.0, t_lo + 100.0)
        i0t, i1t = _window(t_lo, t_hi)
        if i1t > i0t:
            seg      = mb[i0t:i1t]
            idx_abs  = int(np.argmax(np.abs(seg)))
            t_t      = float(bt[i0t + idx_abs])
            t_sign   = 1 if seg[idx_abs] >= 0 else -1
        else:
            t_t     = s_t + 35.0
            t_sign  = 1
        j_t_final = j_t if j_t is not None else s_t + 14.0

        # ── P onset: max-slope tangent intersection ────────────────────────────
        # Search up to 25 ms before P peak
        po_t = _slope_onset(p_t, search_ms=25.0)

        # ── T offset: max-slope tangent intersection ───────────────────────────
        # Search up to 30 ms after T peak
        toff_t = _slope_offset(t_t, search_ms=30.0, t_wave_sign=t_sign)

        # ── Sanity clamp: all positions must be within beat_time ──────────────
        bt_min, bt_max = float(bt[0]), float(bt[-1])
        q_t      = float(np.clip(q_t,      bt_min + 1, -1.0))
        s_t      = float(np.clip(s_t,      1.0,  bt_max - 1))
        p_t      = float(np.clip(p_t,      bt_min + 1, q_t  - 3.0))
        j_t_final= float(np.clip(j_t_final,s_t + 1.0,  t_t  - 3.0))
        t_t      = float(np.clip(t_t,      s_t   + 3.0, bt_max - 1))
        po_t     = float(np.clip(po_t,     bt_min + 1,  p_t  - 1.0))
        toff_t   = float(np.clip(toff_t,   t_t   + 1.0, bt_max - 1))

        # ── Half-windows: 80 % of distance to nearest neighbour ──────────────
        # This gives generous windows for the delineation search while keeping
        # each wave's search region clearly separated from its neighbours.
        def _hw(center: float, lo_bound: float, hi_bound: float,
                min_hw: float = 4.0) -> float:
            """Half-window = 80% of min distance to nearest boundary."""
            return max(min_hw, min(abs(center - lo_bound),
                                   abs(hi_bound - center)) * 0.80)

        self.landmarks = {
            "P_onset":  (round(po_t,     1), _hw(po_t,     bt_min,   p_t,      min_hw=4.0)),
            "P_peak":   (round(p_t,      1), _hw(p_t,      po_t,     q_t,      min_hw=6.0)),
            "Q_peak":   (round(q_t,      1), _hw(q_t,      p_t,      0.0,      min_hw=4.0)),
            "S_peak":   (round(s_t,      1), _hw(s_t,      0.0,      j_t_final, min_hw=4.0)),
            "J_peak":   (round(j_t_final,1), _hw(j_t_final,s_t,      t_t,      min_hw=4.0)),
            "T_peak":   (round(t_t,      1), _hw(t_t,      j_t_final, toff_t,  min_hw=8.0)),
            "T_offset": (round(toff_t,   1), _hw(toff_t,   t_t,      bt_max,   min_hw=5.0)),
        }
        self.source    = "auto-calibrated"
        self.confirmed = False
        log.info(
            "WaveTemplate.calibrate_from_beat: "
            "P_onset=%+.1f  P=%+.1f  Q=%+.1f  R=0  S=%+.1f  T=%+.1f  T_offset=%+.1f ms",
            po_t, p_t, q_t, s_t, t_t, toff_t)


def _mouse_demo_beat() -> "tuple[np.ndarray, np.ndarray]":
    """Return (beat_time, demo_signal) for a realistic mouse ECG morphology.

    Morphology based on:
        Gehrmann et al. (2000) J Interv Card Electrophysiol 4:469-479
        Mitchell et al. (1998) Am J Physiol Heart Circ Physiol 274:H747

    Key differences from human ECG:
    • Very narrow QRS complex (~10 ms vs ~80 ms in human)
    • Short PR interval (~40 ms vs ~160 ms in human)
    • P wave is small and rounded
    • T wave is often small and dome-shaped; can be biphasic or negative
    • Very fast heart rate (500-600 bpm) — beats are close together
    • High-frequency components due to fast depolarisation
    """
    t = np.linspace(-100, 120, 1100)   # 1 ms resolution

    # P wave: small rounded positive deflection at -40 ms, width ~12 ms
    p_wave     = 0.08 * np.exp(-0.5 * ((t + 40) / 6)**2)

    # PR segment: flat at ~0 between -28 and -12 ms
    # Q: small negative notch just before R, very narrow
    q_wave     = -0.12 * np.exp(-0.5 * ((t + 10) / 2.5)**2)

    # R: tall positive, extremely narrow (~3 ms half-width — characteristic of mice)
    r_wave     = 1.00 * np.exp(-0.5 * ((t +  0) / 2.5)**2)

    # S: deep negative right after R, narrow
    s_wave     = -0.45 * np.exp(-0.5 * ((t -  9) / 3.5)**2)

    # ST junction / J wave: pronounced positive hump in mice
    # (early repolarization, very characteristic of C57Bl/6 and related strains)
    j_wave     = 0.10 * np.exp(-0.5 * ((t - 15) / 4)**2)

    # T wave: biphasic-ish, small dome then slight negative — typical of awake C57Bl/6
    t_wave     = (0.12 * np.exp(-0.5 * ((t - 52) / 12)**2)
                  - 0.04 * np.exp(-0.5 * ((t - 72) /  8)**2))

    # Baseline wander: minimal for a clean average beat
    baseline   = 0.015 * np.sin(2 * np.pi * t / 200)

    demo = p_wave + q_wave + r_wave + s_wave + j_wave + t_wave + baseline

    # Normalise so R peak ≈ 1.0
    r_idx  = int(np.argmax(demo))
    if demo[r_idx] > 0:
        demo = demo / demo[r_idx]

    return t, demo



def detect_waves_on_beat(
    beat:      "np.ndarray",
    beat_time: "np.ndarray",
    template:  "WaveTemplate",
) -> "dict[str, float]":
    """Detect P/Q/R/S/J/T landmarks on a single beat aligned on R=0.

    Parameters
    ----------
    beat      : 1-D float array, signal values (R aligned so beat_time≈0 at R peak).
    beat_time : 1-D float array, time in ms relative to R=0.
    template  : WaveTemplate providing (center_ms, half_window_ms) per landmark.

    Returns
    -------
    dict with keys P_onset, P_peak, Q_peak, S_peak, J_peak, T_peak, T_offset.
    Values are ms relative to R=0.  NaN where a landmark cannot be found.

    Algorithm notes
    ---------------
    • P wave : detected on baseline-corrected signal (local TP segment baseline
      subtracted) so small P waves aren't hidden by low-frequency drift.
    • J wave : small positive hump right after S (early repolarization, very
      common in mice).  Detected as local maximum in a narrow window after S.
      Its position is then used to set the T wave search START, preventing the
      J hump from being misidentified as the T wave.
    • T wave : search starts after J wave (or after S + fixed gap).  Uses the
      dominant sign in the search region (abs-max polarity) so both upright and
      inverted T waves are handled correctly.
    """
    bt = np.asarray(beat_time, dtype=float)
    mb = np.asarray(beat,      dtype=float)
    n  = len(bt)

    if n < 10:
        return {k: float("nan")
                for k in ("P_onset", "P_peak", "Q_peak", "S_peak",
                          "J_peak",  "T_peak",  "T_offset")}

    dt_ms = float(bt[1] - bt[0]) if n > 1 else 0.5

    # ── Helpers ─────────────────────────────────────────────────────────
    def _idx(t_ms: float) -> int:
        return int(np.argmin(np.abs(bt - t_ms)))

    def _window(lo_ms: float, hi_ms: float) -> "tuple[int, int]":
        return max(0, _idx(lo_ms)), min(n, _idx(hi_ms) + 1)

    def _extremum(lo_ms: float, hi_ms: float,
                  polarity: int = 1) -> "Optional[float]":
        i0, i1 = _window(lo_ms, hi_ms)
        if i1 <= i0 + 1:
            return None
        seg = mb[i0:i1]
        k   = int(np.argmax(seg) if polarity == 1 else np.argmin(seg))
        return float(bt[i0 + k])

    def _extremum_corrected(lo_ms: float, hi_ms: float,
                             baseline: float, polarity: int = 1) -> "Optional[float]":
        """Like _extremum but on the baseline-subtracted signal."""
        i0, i1 = _window(lo_ms, hi_ms)
        if i1 <= i0 + 1:
            return None
        seg = mb[i0:i1] - baseline
        k   = int(np.argmax(seg) if polarity == 1 else np.argmin(seg))
        # Only return if the corrected amplitude is meaningful (> 10% of max)
        peak_amp = seg[k]
        noise_floor = np.std(mb[max(0, i0 - 10): i0]) if i0 >= 5 else 0.01
        if abs(peak_amp) < max(noise_floor * 1.2, 0.003):
            return None
        return float(bt[i0 + k])

    def _abs_extremum(lo_ms: float, hi_ms: float) -> "tuple[float, int]":
        """(time_ms, sign) of the largest-absolute-value sample."""
        i0, i1 = _window(lo_ms, hi_ms)
        if i1 <= i0 + 1:
            return float("nan"), 1
        seg = mb[i0:i1]
        k   = int(np.argmax(np.abs(seg)))
        return float(bt[i0 + k]), (1 if seg[k] >= 0 else -1)

    def _baseline(lo_ms: float, hi_ms: float) -> float:
        i0, i1 = _window(lo_ms, hi_ms)
        if i1 <= i0:
            return 0.0
        return float(np.median(mb[i0:i1]))

    def _slope_onset(t_peak_ms: float, hw_ms: float) -> float:
        """P-onset via max-slope tangent intersection on the P leading edge."""
        i_pk = _idx(t_peak_ms)
        i_lo = max(0, _idx(t_peak_ms - hw_ms))
        if i_pk <= i_lo + 2:
            return t_peak_ms - hw_ms * 0.4
        seg   = mb[i_lo:i_pk + 1]
        slope = np.gradient(seg, dt_ms)
        k     = int(np.argmax(slope))
        t_k   = float(bt[i_lo + k])
        a_k   = float(mb[i_lo + k])
        s     = float(slope[k])
        if abs(s) < 1e-9:
            return float(bt[i_lo])
        base  = _baseline(t_peak_ms - hw_ms - 40, t_peak_ms - hw_ms - 10)
        t_on  = t_k + (base - a_k) / s
        return float(np.clip(t_on, float(bt[i_lo]), t_peak_ms - 1.0))

    def _slope_offset(t_peak_ms: float, hw_ms: float, sign: int = 1) -> float:
        """T-offset via max-slope tangent intersection on the T trailing edge."""
        i_pk = _idx(t_peak_ms)
        i_hi = min(n - 1, _idx(t_peak_ms + hw_ms))
        if i_hi <= i_pk + 2:
            return t_peak_ms + hw_ms * 0.4
        seg   = mb[i_pk:i_hi + 1]
        slope = np.gradient(seg, dt_ms)
        k     = int(np.argmin(slope) if sign > 0 else np.argmax(slope))
        t_k   = float(bt[i_pk + k])
        a_k   = float(mb[i_pk + k])
        s     = float(slope[k])
        if abs(s) < 1e-9:
            return float(bt[i_hi])
        base  = _baseline(t_peak_ms + hw_ms - 20, t_peak_ms + hw_ms)
        t_off = t_k + (base - a_k) / s
        return float(np.clip(t_off, t_peak_ms + 1.0, float(bt[i_hi])))

    def _clamp(v: "Optional[float]", lo: float, hi: float) -> float:
        if v is None or not np.isfinite(v):
            return float("nan")
        return float(np.clip(v, lo, hi))

    # ── Template search windows ──────────────────────────────────────────
    def _win(key: str) -> "tuple[float, float]":
        c, h = template.landmarks.get(key, WaveTemplate.DEFAULTS[key])
        return c - h, c + h

    q_lo, q_hi   = _win("Q_peak")
    s_lo, s_hi   = _win("S_peak")
    p_lo, p_hi   = _win("P_peak")
    j_lo, j_hi   = _win("J_peak")
    t_lo, t_hi   = _win("T_peak")
    po_lo, po_hi = _win("P_onset")
    to_lo, to_hi = _win("T_offset")

    # ── Detect Q and S (minima near R) ────────────────────────────────────
    bt0, bt1 = float(bt[0]), float(bt[-1])

    q_t = _extremum(q_lo, q_hi, polarity=-1)
    s_t = _extremum(s_lo, s_hi, polarity=-1)

    # ── Detect P wave (robust multi-step) ────────────────────────────────
    #
    # Mouse P waves are small, often only 5–20 % of R amplitude, and sit on
    # top of the decaying T wave (TP segment is short at high heart rates).
    # Strategy:
    #  1. Baseline from the true TP segment: 10–30 ms before the P window.
    #     This is more reliable than using the start of the beat window,
    #     which may still contain T-wave tail.
    #  2. SG-smooth the P search segment to suppress high-frequency noise
    #     before peak picking.
    #  3. Try both polarities (positive / negative P); keep the one with the
    #     largest baseline-corrected absolute amplitude.
    #  4. Require the detected peak to exceed a local SNR threshold
    #     (>1.5× local RMS of the baseline segment) — rejects noise spikes.
    #  5. Use scipy find_peaks with a minimum prominence to avoid selecting
    #     noise bumps when no clear P wave is present.

    from scipy.signal import savgol_filter as _sg_filt, find_peaks as _fp

    # 1. True TP segment: strictly before P window but after T-offset region
    tp_lo = max(bt0, p_lo - 35.0)
    tp_hi = p_lo - 3.0
    pre_baseline = _baseline(tp_lo, tp_hi)
    tp_rms = float(np.std(mb[max(0, _idx(tp_lo)): max(1, _idx(tp_hi))]))
    snr_thresh = max(tp_rms * 1.8, 0.008)  # minimum P amplitude above baseline

    # 2. SG-smooth the P search segment (reduce noise sensitivity)
    i_p0, i_p1 = _window(p_lo, p_hi)
    seg_raw = mb[i_p0:i_p1] - pre_baseline
    n_seg = len(seg_raw)

    if n_seg >= 9:
        win_sg = min(n_seg if n_seg % 2 == 1 else n_seg - 1, 11)
        win_sg = max(win_sg, 5)
        try:
            seg_smooth = _sg_filt(seg_raw, window_length=win_sg,
                                   polyorder=3, mode="interp")
        except Exception:
            seg_smooth = seg_raw
    else:
        seg_smooth = seg_raw

    # 3. Try both polarities; pick the dominant one
    p_t: "Optional[float]" = None

    if n_seg >= 3:
        prom_min = max(snr_thresh * 0.6, 0.003)
        min_dist_samples = max(1, int(4.0 / dt_ms))  # 4 ms min peak separation

        # Positive P wave
        peaks_pos, props_pos = _fp(seg_smooth, prominence=prom_min,
                                    distance=min_dist_samples)
        # Negative P wave
        peaks_neg, props_neg = _fp(-seg_smooth, prominence=prom_min,
                                    distance=min_dist_samples)

        best_amp = 0.0
        for pk_idx, prom in zip(peaks_pos, props_pos.get("prominences", [])):
            amp = float(abs(seg_smooth[pk_idx]))
            if amp > best_amp and amp >= snr_thresh:
                best_amp = amp
                p_t = float(bt[i_p0 + pk_idx])
        for pk_idx, prom in zip(peaks_neg, props_neg.get("prominences", [])):
            amp = float(abs(seg_smooth[pk_idx]))
            if amp > best_amp and amp >= snr_thresh:
                best_amp = amp
                p_t = float(bt[i_p0 + pk_idx])

    # 4. Fallback: simple corrected extremum if find_peaks found nothing
    if p_t is None:
        p_t = _extremum_corrected(p_lo, p_hi, pre_baseline, polarity=1)
    if p_t is None:
        p_t = _extremum_corrected(p_lo, p_hi, pre_baseline, polarity=-1)
    if p_t is None:
        # Last resort: uncorrected max (may be noise, but prevents NaN cascade)
        p_t = _extremum(p_lo, p_hi, polarity=1)

    # ── Detect J wave (positive hump right after S, before T) ────────────
    # J wave is the early-repolarization hump common in mouse ECG.
    # Search in the dedicated J_peak window (defaults: 7–23 ms after R).
    # If S was detected, also enforce that J comes after S.
    j_search_lo = max(j_lo, (s_t + 2.0) if s_t is not None else j_lo)
    j_search_hi = min(j_hi, t_lo - 2.0)   # never overlap T search
    j_t: "Optional[float]" = None
    if j_search_hi > j_search_lo + 2:
        j_t = _extremum(j_search_lo, j_search_hi, polarity=1)

    # ── Detect T wave (starts after J, or after S + gap) ─────────────────
    # Starting after J prevents the J hump from being selected as T.
    t_lo_eff = max(t_lo, (j_t + 5.0) if j_t is not None else t_lo)
    t_t, t_sign = _abs_extremum(t_lo_eff, t_hi)

    # ── P onset ───────────────────────────────────────────────────────────
    po_hw = abs(p_lo - po_lo) if p_t is not None else 18.0
    po_t  = _slope_onset(p_t, po_hw) if p_t is not None else float("nan")

    # ── T offset ──────────────────────────────────────────────────────────
    to_hw = abs(to_hi - t_lo) if np.isfinite(t_t) else 22.0
    to_t  = _slope_offset(t_t, to_hw, sign=t_sign) if np.isfinite(t_t) else float("nan")

    return {
        "P_onset":  _clamp(po_t, bt0,    -2.0),
        "P_peak":   _clamp(p_t,  bt0+1,  -3.0),
        "Q_peak":   _clamp(q_t,  -35.0,  -0.5),
        "S_peak":   _clamp(s_t,   0.5,    35.0),
        "J_peak":   _clamp(j_t,   3.0,    35.0),
        "T_peak":   _clamp(t_t,   5.0,    bt1-1),
        "T_offset": _clamp(to_t,  6.0,    bt1),
    }

