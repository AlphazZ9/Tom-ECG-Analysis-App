"""
ecg.core.models
───────────────
Species-specific physiology constants, shared dataclasses, and typed dicts.
Nothing in this module may import from ecg.ui — it is UI-free by design.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, TypedDict, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

class MouseECG:
    """Single source of truth for adult mouse ECG physiology.

    Change a value here and every filter, plot, and UI default
    that references it updates automatically.
    """
    # ── Heart rate ───────────────────────────────────────────
    HR_REST_BPM:  float = 500.0    # typical resting HR (awake, unrestrained)
    HR_MIN_BPM:   float = 180.0    # lower bound (anaesthetised / deep sleep)
    HR_MAX_BPM:   float = 900.0    # upper bound — some strains / stress exceed 750

    # ── RR interval bounds (derived) ────────────────────────
    # HR_MAX_BPM = 900 → RR = 67 ms.  RR_MIN_MS is the physiological floor
    # used for artefact detection and doublet rejection — NOT for find_peaks
    # distance (see PEAK_DISTANCE_MS below).
    RR_MIN_MS:    float = round(60_000.0 / HR_MAX_BPM, 1)  # ≈ 67 ms @ 900 bpm
    RR_MAX_MS:    float = round(60_000.0 / 180.0, 1)        # ≈ 333 ms @ 180 bpm

    # ── Signal acquisition ──────────────────────────────────
    FS_DEFAULT:   int   = 2000    # Hz  — Spike2 standard for mouse ECG

    # ── pNNx threshold (mouse-specific) ────────────────────
    # pNN50 (human default) uses 50 ms, which is ~42 % of a typical mouse RR
    # (~120 ms at 500 bpm).  Normal vagal RR swings in mice are only 5–10 ms,
    # so pNN50 is near 0 % for virtually every healthy mouse.
    # pNN6 (6 ms ≈ 5 % of mean RR) is the mouse-appropriate threshold.
    # Reference: Thireau et al. 2008, Am J Physiol 294:H977.
    PNN_THRESHOLD: int = 6       # ms  — use for nk.hrv_time and manual pNN calc

    # ── Bandpass filter ─────────────────────────────────────
    BP_LO_HZ:     float = 1.0    # removes baseline wander / breathing artefact
    BP_HI_HZ:     float = 150.0  # preserves mouse QRS morphology (~10 ms)

    # ── Peak detection — DEUX RÔLES, DEUX CONSTANTES DISTINCTES ─────────────
    #
    # PEAK_DISTANCE_MS  →  distance passée à find_peaks() sur le signal de
    #                       dérivée SG.  Doit être COURT : évite que find_peaks
    #                       fusionne des R consécutifs à HR élevée, mais ne
    #                       contrôle pas le rejet R-vs-J (rôle du score composite).
    #                       Règle : PEAK_DISTANCE_MS < RR_MIN_MS
    #                       40 ms → supporte jusqu'à 1500 bpm (au-delà de toute
    #                       physiologie murine) ; les faux pics intra-QRS sont
    #                       éliminés par _select_r_in_window.
    #
    # MIN_RR_MS         →  seuil physiologique utilisé pour :
    #                         • resolve_r_vs_j_peaks (rejet doublets R-J)
    #                         • detect_rr_artifacts  (borne nonphysio)
    #                         • recover_missed_beats (détection gaps longs)
    #                       Doit refléter le vrai RR minimum murin (≈ 67 ms
    #                       @ 900 bpm = HR_MAX_BPM).
    #
    # Origine du bug observé :
    #   Avec une seule constante à 50 ms, le seuil doublet de
    #   resolve_r_vs_j_peaks était 50 ms → toute J-wave arrivant > 50 ms
    #   après le R échappait au rejet et était gardée comme pic indépendant.
    #   À 22 ms, find_peaks n'imposait presque aucune contrainte de distance,
    #   _select_r_in_window recevait plus d'ancres et choisissait correctement
    #   par score upstroke — amélioration réelle, mais pour la mauvaise raison.
    #
    PEAK_DISTANCE_MS: float = 40.0       # ms — distance find_peaks uniquement
    MIN_RR_MS:        float = RR_MIN_MS  # ≈ 67 ms — seuil physiologique

    # ── Prominence-search window bound (performance) ────────────────────────
    # scipy.signal.find_peaks(..., prominence=X)/peak_prominences/peak_widths
    # search the ENTIRE array outward from each peak for a taller point when
    # `wlen` isn't given — cheap on a bandpassed signal (no long-range trend),
    # but catastrophic on a raw one with real baseline wander: observed
    # ~130s -> ~0.1s (identical peak set) on a 95-min raw mouse ECG recording
    # once `wlen` was added. 10x RR_MAX_MS is generous — any genuine peak has
    # a comparably-tall neighbour within a couple of beats, let alone ten —
    # while still bounding the worst case to a tiny fraction of a
    # multi-million-sample recording instead of the whole array.
    PROMINENCE_WLEN_MS: float = round(10 * RR_MAX_MS, 0)  # ≈ 3330 ms

    # ── Beat template window ────────────────────────────────
    # Full mouse P-QRS-T spans ~60–120 ms → ±100 ms window captures it safely
    BEAT_HALF_WIN_S: float = 0.10    # seconds  (±100 ms)

    # ── HRV frequency bands (rodent, Thireau 2008) ──────────
    # Mouse respiratory rate ≈ 2–4 Hz → HF band extends to 5 Hz
    VLF: tuple = (0.00, 0.40)    # Hz
    LF:  tuple = (0.40, 1.50)    # Hz
    HF:  tuple = (1.50, 5.00)    # Hz
    PSD_XLIM:        float = 5.0   # Hz  — x-axis limit for PSD plot
    PSD_INTERP_FS:   float = 20.0  # Hz  — fixed interpolation rate (Nyquist >> 5 Hz)

    # ── Spectral HRV across excluded RR intervals ──────────
    # The spectrum is estimated on a uniformly resampled RR series, so an
    # excluded stretch (pause, sub-bound anaesthetised run, removed beats)
    # is filled in by the interpolator, not measured. A short hole costs a
    # little VLF smoothing; a long one fabricates a ramp that swamps LF/HF.
    # Spectral HRV is therefore WITHHELD (not computed) when either the
    # longest run of consecutive excluded intervals exceeds
    # SPECTRAL_MAX_GAP_S -- 2 s is ~1 cycle of the 0.4 Hz LF floor and
    # ~16 mouse beats -- or excluded time exceeds this fraction of the
    # analysed span.
    SPECTRAL_MAX_GAP_S:         float = 2.0    # s
    SPECTRAL_MAX_EXCLUDED_FRAC: float = 0.05   # 5 % of the analysed span

    FREQ_BAND_PRESETS: "list[tuple[str, tuple[float, float], tuple[float, float], tuple[float, float]]]" = [
        ("Mouse Thireau (default)", (0.00, 0.40), (0.40, 1.50), (1.50, 5.00)),
        ("Mouse LF/HF custom", (0.00, 0.40), (0.40, 2.00), (2.00, 5.00)),
    ]

    # ── ECG interval reference ranges (adult mouse, ms) ─────
    # Normal ranges: typical values in healthy adult mice
    PR_NORMAL:  tuple = (30,  55)
    QRS_NORMAL: tuple = ( 8,  25)
    QT_NORMAL:  tuple = (30,  90)
    QTC_NORMAL: tuple = (30, 100)   # Mitchell QTc (QT/√(RR/100)) only

    # ── Absolute acceptance bounds for delineation filtering ─
    # Values outside these are physiologically impossible in mice;
    # they indicate delineation failures and are replaced with NaN.
    # Set at ~2× the normal range to allow for outlier beats.
    # Reference: Gehrmann 2000, Mitchell 1998, Thireau 2008.
    PR_ABS_MIN:  float = 10.0   # ms  (< 10 ms = not physically possible)
    PR_ABS_MAX:  float = 80.0   # ms  (> 80 ms = delineation failure)
    QRS_ABS_MIN: float =  3.0   # ms  (< 3 ms  = noise / artefact)
    QRS_ABS_MAX: float = 35.0   # ms  (> 35 ms = pathological or error)
    QT_ABS_MIN:  float = 15.0   # ms
    QT_ABS_MAX:  float = 110.0  # ms
    # No absolute QTc bounds: QTc is derived from a QT and an RR that are
    # each already bounds-checked above, and its scale depends entirely on
    # the formula (at RR = 120 ms, QT = 50 ms gives ~46 ms with Mitchell but
    # ~144 ms with Bazett), so one fixed range can only be right for one of
    # them. A former QTC_ABS_MAX of 200 ms silently NaN'd every Hodges value
    # (QT + 1.75·(HR − 60) ≈ 820 ms at 500 bpm).

    # ── Epoch analysis ──────────────────────────────────────
    EPOCH_DEFAULT_S:  float = 60.0
    EPOCH_MIN_S:      float =  5.0


# QTc correction formulas (computed by ecg.core.analysis.compute_qtc), in
# selector order; the first is the default. Mitchell et al. 1998 (Am J
# Physiol 274:H747) is Bazett normalised to a 100 ms mouse RR rather than
# to a 1 s human RR. Bazett/Fridericia are kept for comparison with
# human-calibrated studies. The QT–RR relation is weak in mice, so the
# uncorrected QT should always be reported alongside any QTc.
QTC_FORMULA_LABELS = {
    "mitchell":   "Mitchell (QT/√(RR/100))",
    "bazett":     "Bazett (QT/√RR, human)",
    "fridericia": "Fridericia (QT/∛RR, human)",
}

# Selector labels / keys written by older versions (sessions, combobox
# values). "Mitchell (∛RR)" was really Fridericia with RR in seconds, and
# Hodges (QT + 1.75·(HR − 60), a human slope) gave ~820 ms at mouse heart
# rates — both now map to the corrected Mitchell formula.
_LEGACY_QTC_FORMULAS = {
    "hodges":             "mitchell",
    "hodges (linear hr)": "mitchell",
    "mitchell (∛rr)":     "mitchell",
    "bazett (√rr)":       "bazett",
}


def normalize_qtc_formula(value: object) -> str:
    """Return a supported QTc formula key from a key, a current or legacy
    selector label; old/unknown values default to Mitchell."""
    formula = str(value).strip().lower()
    for key, label in QTC_FORMULA_LABELS.items():
        if formula == key or formula == label.lower():
            return key
    return _LEGACY_QTC_FORMULAS.get(formula, "mitchell")


# The awake/isoflurane/ketamine-xylazine/custom experimental-context
# selector and its per-metric reference bands (radar chart, Time Domain
# table annotations, Rolling/Epochs/Intervals reference bands, PDF report)
# were removed here -- not useful in practice. Where a context value fed a
# real detection threshold (classify_arrhythmias' bradycardia/pause RR
# cutoff), the value the default "telemetry_awake" context used to
# provide is now a fixed constant instead, so detection behaviour is
# unchanged.


@dataclasses.dataclass
class FilterParams:
    """Canonical schema for all parameters that _prepare_signal consumes.

    Using a dataclass as the single source of truth means:
    - _snapshot_params, _collect_session_state, and _restore_state_from_session
      all use the same field list — impossible to add a field to one and forget
      the others (which caused the min_rr_ms KeyError and channel/crop loss bugs).
    - Type checking catches misspelled keys at analysis time, not at runtime.
    - to_dict() / from_dict() keep session serialisation consistent.

    Note: ``notch_filter`` is named with a suffix to avoid shadowing the
    module-level ``notch()`` function in callers.
    """
    channel:      str   = "ECG"
    fs:           int   = MouseECG.FS_DEFAULT
    t_start:      float = 0.0
    t_end:        float = 0.0
    lp:           float = MouseECG.BP_LO_HZ
    hp:           float = MouseECG.BP_HI_HZ
    notch_filter: bool  = False   # OFF by default — use no-filter mode
    clean_method: str   = "neurokit"
    no_filter:    bool  = True
    min_rr_ms:        float = MouseECG.MIN_RR_MS
    # Upper RR bound for HRV: intervals longer than this are excluded (not
    # bridged). The default (≈ 333 ms, 180 bpm) assumes an awake mouse --
    # widen it for anaesthetised animals. min_rr_ms is the lower bound.
    max_rr_ms:        float = MouseECG.RR_MAX_MS
    peak_distance_ms: float = MouseECG.PEAK_DISTANCE_MS
    thresh:           float = 0.5
    artifact_correction: bool = False   # OFF by default — use manual review instead
    auto_epochs:  bool  = False
    invert_signal: bool = False   # manual polarity override (flip before auto-detection)
    detection_method: str = "auto"  # "auto" | "sg_derivative" | "wavelet" | "envelope_max" | "ml"
    sg_target_fs: int = 10_000     # Hz — downsample target before SG+derivative

    def to_dict(self) -> dict:
        """Return a plain dict compatible with _prepare_signal and session JSON."""
        d = dataclasses.asdict(self)
        # _prepare_signal expects key "notch", not "notch_filter"
        d["notch"] = d.pop("notch_filter")
        # Normalise detection_method to a short key for _compute_preview_bundle.
        # Must cover every branch signal_controller.py's dispatcher checks --
        # this used to only recognise "sg"/"deriv" and silently collapsed
        # everything else (including "Wavelet (CWT)" and "Envelope Max") to
        # "auto", so selecting either of those two detectors in the UI never
        # actually reached them: the dispatcher's own wavelet/envelope
        # branches were unreachable dead code, since this is the function
        # that produces the string they'd need to match against.
        dm = str(d.get("detection_method", "auto")).lower()
        if "wavelet" in dm or "cwt" in dm:
            d["detection_method"] = "wavelet"
        elif "sg" in dm or "deriv" in dm:
            d["detection_method"] = "sg_derivative"
        elif "envelope" in dm or "max" in dm:
            d["detection_method"] = "envelope_max"
        elif "ml" in dm or "machine" in dm:
            d["detection_method"] = "ml"
        else:
            d["detection_method"] = "auto"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FilterParams":
        """Reconstruct from a session dict (tolerates missing keys via defaults)."""
        # Accept both "notch" (session/prepare_signal key) and "notch_filter"
        notch_val = d.get("notch_filter", d.get("notch", False))
        return cls(
            channel      = str(  d.get("channel",      "ECG")),
            fs           = int(  d.get("fs",            MouseECG.FS_DEFAULT)),
            t_start      = float(d.get("t_start",       0.0)),
            t_end        = float(d.get("t_end",         0.0)),
            lp           = float(d.get("lp",            MouseECG.BP_LO_HZ)),
            hp           = float(d.get("hp",            MouseECG.BP_HI_HZ)),
            notch_filter = bool( notch_val),
            clean_method = str(  d.get("clean_method",  "neurokit")),
            no_filter    = bool( d.get("no_filter",     True)),
            min_rr_ms        = float(d.get("min_rr_ms",         MouseECG.MIN_RR_MS)),
            max_rr_ms        = float(d.get("max_rr_ms",         MouseECG.RR_MAX_MS)),
            peak_distance_ms = float(d.get("peak_distance_ms",  MouseECG.PEAK_DISTANCE_MS)),
            thresh           = float(d.get("thresh",            0.5)),
            artifact_correction = bool(d.get("artifact_correction", False)),
            auto_epochs  = bool( d.get("auto_epochs",   False)),
            invert_signal = bool(d.get("invert_signal",  False)),
            detection_method = str(d.get("detection_method", "auto")),
            sg_target_fs = int( d.get("sg_target_fs",   10_000)),
        )

    @classmethod
    def from_widgets(cls, app: "ECGApp") -> "FilterParams":
        """Read all values from main-thread widgets.  Call only from main thread."""
        # sw_filtering ON means DSP filtering IS applied -- the inverse of
        # this dataclass's no_filter field (kept as-is for session-JSON
        # backward compatibility; only the UI-facing widget was renamed).
        no_filter = not bool(app.sw_filtering.get())
        return cls(
            channel      = app.ent_channel.get().strip() or "ECG",  # type: ignore[union-attr]
            fs           = int(app._safe_float(app.ent_fs, MouseECG.FS_DEFAULT)),
            t_start      = app._safe_float(app.ent_t_start, 0.0),
            t_end        = app._safe_float(app.ent_t_end,   0.0),
            lp           = app._safe_float(app.ent_lp,  MouseECG.BP_LO_HZ),
            hp           = app._safe_float(app.ent_hp,  MouseECG.BP_HI_HZ),
            notch_filter = bool(app.sw_notch.get()),  # type: ignore[union-attr]
            clean_method = app.cb_clean.get(),  # type: ignore[union-attr]
            no_filter    = no_filter,
            min_rr_ms        = app._safe_float(app.ent_minrr, MouseECG.MIN_RR_MS),
            max_rr_ms        = app._safe_float(getattr(app, "ent_maxrr", None), MouseECG.RR_MAX_MS),
            peak_distance_ms = float(getattr(app, '_peak_distance_ms', MouseECG.PEAK_DISTANCE_MS)),
            thresh           = float(app.sl_thr.get()),  # type: ignore[union-attr]
            artifact_correction = bool(app.sw_artifact.get()),
            auto_epochs  = bool(app.sw_epoch.get()),
            invert_signal = bool(app.sw_invert_signal.get()) if app.sw_invert_signal is not None else False,  # type: ignore[union-attr]
            detection_method = (app.cb_det_method.get()
                                if getattr(app, "cb_det_method", None) is not None
                                else "auto"),
            sg_target_fs = int(app._safe_float(
                getattr(app, "ent_sg_target_fs", None), 10_000)),
        )


# ════════════════════════════════════════════════════════════
#  SESSION CACHE  (save / restore analysis state)
# ════════════════════════════════════════════════════════════

SESSION_VERSION = 6          # bump when the schema changes
                             # v4: signal arrays removed from session
                             # v5: format changed from pickle to JSON
SESSION_SUFFIX  = ".ecgsession"   # file extension (content is now JSON)
SESSION_DIR     = Path.home() / ".ecg_sessions"   # default cache folder


@dataclasses.dataclass
class ArrhythmiaEvent:
    """A single classified arrhythmia episode."""
    kind:        str          # "bradycardia" | "tachycardia" | "pause" |
                              # "esv_run" | "irregular_run" | "av_delay"
                              # NOTE on "av_delay": classify_arrhythmias() only
                              # sees R-peak timing by default, so a plain
                              # "pause" is NOT distinguishable from AV block on
                              # R-peaks alone. When per-beat PR-interval data
                              # is passed in (from analyse_intervals()), a run
                              # of sustained PR prolongation is flagged as
                              # "av_delay" -- this is a heuristic conduction-
                              # delay flag (PR vs. a fixed cutoff, see
                              # detection._AV_DELAY_PR_HI_MS), NOT a definitive
                              # Mobitz I/II/3rd-degree diagnosis: distinguishing those
                              # would require seeing P waves on NON-conducted
                              # beats (no following QRS to anchor on), which
                              # this pipeline does not attempt.
    label:       str          # human-readable label
    t_start:     float        # seconds
    t_end:       float        # seconds
    n_beats:     int          # number of beats in episode
    hr_mean:     float        # mean HR in episode (bpm)
    rr_mean:     float        # mean RR (ms)
    severity:    str          # "info" | "warning" | "alert"
    baseline_hr: float = 0.0  # individual baseline HR (bpm); 0 = not applicable
    delta_pct:   float = 0.0  # % deviation from baseline (negative = brady, positive = tachy)
    # False on brady/tachy events when the baseline window covered most of
    # the analysed beats (see detection.arrhythmia_baseline_info) -- the
    # reference then includes the event itself, so the call is unreliable.
    baseline_reliable: bool = True

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_end - self.t_start)


class AnalysisResults(TypedDict, total=False):
    """Typed schema for the dict returned by ``analyse_core``.

    Using TypedDict (total=False so all keys are optional for incremental
    updates) means:
    - IDEs and type checkers understand the shape without reading the function.
    - Callers accessing r["beat_corr"] vs r.get("beat_corr") are explicit.
    - merge-updates (r.update(freq_results)) stay type-checked.

    Keys populated by analyse_core (always present after a successful run):
        hr, rr_ms, rr_valid, rr_t_s, rr_df, hrv_time, hrv_freq, hrv_nonlin,
        intervals, beat_template, beat_time, beat_matrix, beat_sd,
        beat_corr, peak_amps

    (rr_valid / rr_t_s may be missing from results restored from a session
    saved before they existed -- analysis.rr_timeseries() rebuilds both.)

    Keys added on demand by later analyse_* calls:
        hrv_freq    (analyse_hrv_freq)
        hrv_nonlin  (analyse_hrv_nonlinear)
        intervals   (analyse_intervals — overwrites the stub from analyse_core)
    """
    # ── Core (always set after analyse_core) ────────────────────────────
    hr:            dict                          # mean/min/max/std/n/n_valid + RR exclusion counts and the RR bounds used (rr_min_ms/rr_max_ms)
    rr_ms:         "np.ndarray"                  # raw RR array (ms) -- ALL intervals
    rr_valid:      "np.ndarray"                  # bool mask over rr_ms: inside RR bounds and not spanning a removed beat (used for HRV)
    rr_t_s:        "np.ndarray"                  # end time (s) of every interval in rr_ms (rpeaks[1:] / fs)
    rr_df:         "pd.DataFrame"                # per-beat Time_s / RR_ms / HR_bpm (valid intervals only)
    hrv_time:      "pd.DataFrame"                # NeuroKit2 time-domain metrics
    # ── Populated on demand (empty DataFrame until explicitly computed) ──
    hrv_freq:      "pd.DataFrame"
    hrv_nonlin:    "pd.DataFrame"
    intervals:     "pd.DataFrame"                # PR/QRS/QT/QTc per beat
    # ── Beat template (None if computation failed) ───────────────────────
    beat_template: "Optional[np.ndarray]"        # mean beat shape
    beat_time:     "Optional[np.ndarray]"        # time axis in ms
    beat_matrix:   "Optional[np.ndarray]"        # ≤60 ghost rows
    beat_sd:       "Optional[np.ndarray]"        # std per sample
    beat_corr:     "Optional[np.ndarray]"        # per-beat Pearson r with template
    peak_amps:     "Optional[np.ndarray]"        # R-peak amplitude per beat


