"""
ecg.core.wave_template
──────────────────────
Beat-template calibration and P/QRS/T wave delineation (incl. QRS onset
and QRS offset / J point).
No UI imports — pure NumPy / SciPy.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

WAVE_TEMPLATE_PATH: Path = Path.home() / ".ecg_wave_template.json"
_WAVE_TEMPLATE_PATH_LEGACY: Path = Path.home() / ".ecg_wave_template.pkl"

log = logging.getLogger("ecg")

# Provenance of a WaveTemplate (``WaveTemplate.source``) -> label. Shown in
# the UI and in exports so a reader knows which windows guided delineation.
TEMPLATE_SOURCES: "dict[str, str]" = {
    "default":         "built-in mouse defaults",
    "saved-default":   "saved default template",
    "file":            "loaded from a template file",
    "auto-calibrated": "auto-calibrated on this recording's mean beat",
    "refined":         "refined from this recording's delineation",
    "edited":          "edited on this recording",
    # Written by older versions (still found in saved files / sessions).
    "user-confirmed":  "user-confirmed template",
    "auto-updated":    "auto-updated from an earlier recording (older version)",
}

# Landmarks the delineation searches with a template window, in the order
# they are disclosed (QRS onset / J point are walked from Q / S, not
# searched in a window).
SEARCHED_LANDMARKS: "tuple[str, ...]" = (
    "P_onset", "P_peak", "Q_peak", "S_peak", "J_peak", "T_peak", "T_offset")

# A P search window centred closer than this to the Q wave lands on the Q
# wave / QRS onset (the mouse P peak lies ~20-40 ms before Q). A saved
# default failing this is not used (see WaveTemplate.plausibility_problems).
P_TO_Q_MIN_MS: float = 8.0


def _window_or_none(v: object) -> "Optional[tuple[float, float]]":
    """(centre, half-window) from a stored entry, or None when it cannot
    be a search window (not a pair, non-finite, half-window <= 0)."""
    try:
        c, h = float(v[0]), float(v[1])          # type: ignore[index]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(c) and math.isfinite(h)) or h <= 0:
        return None
    return c, h


def invalid_landmarks(raw: object) -> "list[str]":
    """Entries of a raw ``landmarks`` mapping that cannot be search windows,
    described for the user (empty when all are usable)."""
    if not isinstance(raw, dict):
        return ["landmarks (not a mapping)"]
    bad: "list[str]" = []
    for k, v in raw.items():
        if _window_or_none(v) is not None:
            continue
        try:
            c, h = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError, KeyError):
            bad.append(f"{k} (not a [centre, half-window] pair)")
            continue
        bad.append(f"{k} (non-finite)" if not (math.isfinite(c) and math.isfinite(h))
                   else f"{k} (half-window {h:g} ms <= 0)")
    return bad


def format_windows(landmarks: "dict[str, tuple[float, float]]",
                   keys: "tuple[str, ...]" = SEARCHED_LANDMARKS) -> str:
    """'P_onset -62±20, P -45±28, ...' (ms from R) for the *keys* present."""
    return ", ".join(f"{k.replace('_peak', '')} {landmarks[k][0]:+.0f}±"
                     f"{landmarks[k][1]:.0f}" for k in keys if k in landmarks)


def _that_recording(text: str) -> str:
    """A provenance written on one recording, worded for another one."""
    return (text.replace("this recording's", "that recording's")
                .replace("this recording", "that recording"))


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
        # QRS onset: first departure from the PR-segment baseline before Q
        # (or before R when there is no Q) -- start of PR's end, QRS and QT.
        "QRS_onset": (-12.0, 10.0),  # ~12 ms before R, ±10 ms (spans -22→-2)
        "S_peak":   (  5.0, 12.0),   # S wave:   ~5 ms after  R,   ±12 ms (spans -7→+17)
        # QRS offset = J point: end of the S wave, where its upstroke returns
        # to the PR baseline (the J wave that follows is early repolarisation,
        # Boukens et al. 2014, so it belongs to QT, not to QRS).
        "QRS_offset": ( 10.0, 10.0), # ~10 ms after R,  ±10 ms (spans 0→+20)
        # J wave (early repolarization notch / Osborn-like hump after S, common in mice)
        # Must be detected BEFORE T so T search can skip over it
        "J_peak":   ( 15.0,  8.0),   # J wave:   ~15 ms after R,   ±8 ms  (spans 7→23)
        "T_peak":   ( 52.0, 26.0),   # T wave:   ~52 ms after R,   ±26 ms (starts after J)
        "P_onset":  (-62.0, 20.0),   # P onset:  ~62 ms before R,  ±20 ms search
        "T_offset": ( 82.0, 22.0),   # T offset: ~82 ms after  R,  ±22 ms search
    }

    def __init__(self) -> None:
        self.landmarks: "dict[str, tuple[float, float]]" = dict(self.DEFAULTS)
        self.confirmed: bool  = False   # True once the user applied it in the editor
        # Provenance (see TEMPLATE_SOURCES / provenance()): what the windows
        # come from, and ``origin`` the detail -- the file name for "file",
        # the template's own source for "saved-default", the template that
        # was edited for "edited".
        self.source:    str   = "default"
        self.origin:    str   = ""
        self.created_at: str  = ""

    # ── Copy / (de)serialisation ───────────────────────────────────────

    def copy(self) -> "WaveTemplate":
        """Independent copy (the delineation worker gets one, so an edit
        made while it runs cannot change the windows mid-run)."""
        return WaveTemplate.from_dict(self.to_dict())

    def to_dict(self) -> dict:
        """JSON-compatible form (saved files, sessions, interval attrs).
        Tuples become two-element lists (JSON has no tuple type)."""
        return {
            "landmarks":  {k: [float(v[0]), float(v[1])]
                           for k, v in self.landmarks.items()},
            "confirmed":  bool(self.confirmed),
            "source":     str(self.source),
            "origin":     str(self.origin),
            "created_at": str(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: "Optional[dict]") -> "WaveTemplate":
        """Inverse of ``to_dict``; missing / invalid fields keep the
        defaults (a template saved before a landmark existed lacks it --
        detection then uses that landmark's default window; an entry with
        a non-finite value or a half-window <= 0 is dropped the same way)."""
        obj = cls()
        if not isinstance(data, dict):
            return obj
        lm = data.get("landmarks")
        if isinstance(lm, dict):
            parsed: "dict[str, tuple[float, float]]" = {}
            for k, v in lm.items():
                # A non-finite centre or a half-window <= 0 is no window
                # (a NaN P window made PR fail on every beat): that landmark
                # keeps its default. Files with such entries are refused.
                win = _window_or_none(v)
                if win is not None:
                    parsed[str(k)] = win
            if parsed:
                obj.landmarks = parsed
        obj.confirmed  = bool(data.get("confirmed", False))
        obj.source     = str(data.get("source", "default") or "default")
        obj.origin     = str(data.get("origin", "") or "")
        obj.created_at = str(data.get("created_at", "") or "")
        return obj

    def provenance(self) -> str:
        """Human-readable provenance, shown in the UI and in exports."""
        src, origin = self.source, self.origin
        if src == "saved-default":
            when = self.created_at[:16].replace("T", " ")
            text = "saved default template" + (f" (saved {when})" if when else "")
            if origin in TEMPLATE_SOURCES:
                # Bare source key (saved before origin notes were stored):
                # described without claiming it was made on this recording.
                tmp = WaveTemplate()
                tmp.source = origin
                origin = tmp.origin_note()
            if origin:
                text += f", {origin}"
            return text
        if src == "file":
            return f"loaded from {origin}" if origin else TEMPLATE_SOURCES["file"]
        if src == "edited":
            return TEMPLATE_SOURCES["edited"] + (f" (based on: {origin})" if origin else "")
        return TEMPLATE_SOURCES.get(src, src)

    def origin_note(self, recording: str = "") -> str:
        """Where these windows came from, worded for use on ANOTHER
        recording (a saved default): "originally auto-calibrated on the mean
        beat of <recording>", never "... on this recording". Empty for the
        built-in defaults."""
        src, origin = self.source, self.origin
        rec = recording or "another recording"
        if src == "default":
            return ""
        if src == "saved-default":
            return origin                      # already worded this way
        if src == "file":
            return f"originally loaded from {origin or 'a template file'}"
        if src == "auto-calibrated":
            return f"originally auto-calibrated on the mean beat of {rec}"
        if src == "refined":
            return f"originally refined from the delineation of {rec}"
        if src == "edited":
            return f"originally edited on {rec}" + (
                f" (based on: {_that_recording(origin)})" if origin else "")
        return TEMPLATE_SOURCES.get(src, src)

    def plausibility_problems(self) -> "list[str]":
        """Why these windows cannot guide a mouse delineation (empty when
        they can), checked on the windows detection uses (missing landmarks
        take their defaults): finite values and half-windows > 0; centres
        in the order P onset < P < Q < R (0) < S < T < T end; the P window
        centred at least P_TO_Q_MIN_MS before Q; the T-end window starting
        after the J point (QRS offset)."""
        problems = [f"{k}: invalid window" for k, v in self.landmarks.items()
                    if _window_or_none(v) is None]
        if problems:
            return problems
        lm = {**self.DEFAULTS, **self.landmarks}
        c = {k: v[0] for k, v in lm.items()}
        chain = [("P_onset", c["P_onset"]), ("P_peak", c["P_peak"]),
                 ("Q_peak", c["Q_peak"]), ("R", 0.0), ("S_peak", c["S_peak"]),
                 ("T_peak", c["T_peak"]), ("T_offset", c["T_offset"])]
        for (k0, v0), (k1, v1) in zip(chain, chain[1:]):
            if not v0 < v1:
                problems.append(f"{k0} ({v0:+.1f} ms) is not before "
                                f"{k1} ({v1:+.1f} ms)")
        gap = c["Q_peak"] - c["P_peak"]
        if 0 < gap < P_TO_Q_MIN_MS:
            problems.append(f"P window centred {gap:.1f} ms before Q (at least "
                            f"{P_TO_Q_MIN_MS:.0f} ms needed, else it finds the "
                            f"Q wave / QRS onset, not the P wave)")
        t_lo = lm["T_offset"][0] - lm["T_offset"][1]
        if t_lo <= c["QRS_offset"]:
            problems.append(f"T-end window starts at {t_lo:+.1f} ms, not after "
                            f"the J point ({c['QRS_offset']:+.1f} ms)")
        return problems

    # ── Persistence ────────────────────────────────────────────────────
    # Nothing writes a template to disk except an explicit user action
    # (editor: "Save as default template" / "Save to file..."). The
    # delineation never does: a recording's intervals depend only on that
    # recording and the template the user chose for it.

    def save(self, path: "Optional[Path]" = None, recording: str = "") -> None:
        """Write this template as JSON to *path* (default: the saved default
        template, ``WAVE_TEMPLATE_PATH``). Explicit user action only.

        *recording* (the file name the windows were made on) goes into the
        neutral ``saved_from`` note, so a recording that later starts from
        this file is told where its windows came from."""
        target = Path(path) if path is not None else WAVE_TEMPLATE_PATH
        data = self.to_dict()
        data["created_at"] = datetime.now().isoformat()
        data["saved_from"] = self.origin_note(recording)
        if recording:
            data["recording"] = str(recording)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        log.info("WaveTemplate saved → %s", target)

    @staticmethod
    def _read_json(path: Path) -> dict:
        """Raw template data in *path*; ValueError when it is not a usable
        template (unreadable, no landmarks, or a landmark that is no search
        window: non-finite value or half-window <= 0)."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            raise ValueError(f"Not a wave template file: {path.name} ({exc})") from exc
        if not isinstance(data, dict) or not isinstance(data.get("landmarks"), dict) \
                or not data["landmarks"]:
            raise ValueError(f"Not a wave template file: {path.name} (no landmarks)")
        bad = invalid_landmarks(data["landmarks"])
        if bad:
            raise ValueError(f"Invalid search windows in {path.name}: "
                             + ", ".join(bad))
        return data

    @classmethod
    def _read(cls, path: Path) -> "Optional[WaveTemplate]":
        """Template stored in *path*, or None when missing / unreadable /
        invalid."""
        if not path.exists():
            return None
        try:
            data = cls._read_json(path)
        except ValueError as exc:
            log.warning("WaveTemplate: could not read %s: %s", path, exc)
            return None
        obj = cls.from_dict(data)
        if "source" not in data:
            obj.source = "file"
        return obj

    @classmethod
    def load(cls, path: "Optional[Path]" = None) -> "WaveTemplate":
        """Load a template file as stored (default: ``WAVE_TEMPLATE_PATH``),
        falling back to the built-in defaults on any error.

        With an explicit *path* (a template file the user picked) the result
        is tagged ``source="file"``, ``origin`` = the file name, and an
        unreadable file -- or one with a non-finite value or a half-window
        <= 0 -- raises ValueError instead.

        Legacy pickle files are left untouched and never deserialised because
        pickle can execute arbitrary code during loading.
        """
        if path is not None:
            obj = cls.from_dict(cls._read_json(Path(path)))   # ValueError if bad
            obj.source, obj.origin = "file", Path(path).name
            return obj

        obj = cls._read(WAVE_TEMPLATE_PATH)
        if obj is not None:
            log.info("WaveTemplate loaded (source=%s, confirmed=%s)",
                     obj.source, obj.confirmed)
            return obj

        # Preserve old data but never execute pickle payloads during startup.
        if _WAVE_TEMPLATE_PATH_LEGACY.exists():
            log.warning(
                "Ignoring legacy pickle wave template at %s; recalibrate and "
                "save it again to use the safe JSON format",
                _WAVE_TEMPLATE_PATH_LEGACY)

        return cls()

    @classmethod
    def load_default(cls) -> "WaveTemplate":
        """The template a recording starts with: the saved default template
        when the user saved a usable one ("Save as default template"), else
        the built-in defaults. Read-only. See ``load_default_checked``."""
        return cls.load_default_checked()[0]

    @classmethod
    def load_default_checked(cls, quiet: bool = False
                             ) -> "tuple[WaveTemplate, Optional[str]]":
        """(template a recording starts with, notice or None).

        The saved default file is NOT used -- the built-in defaults are,
        the file is left in place, and the notice (naming the file, the
        reason and its windows) says so -- when it is:

        * unreadable, or has a landmark that is no window (non-finite,
          half-window <= 0);
        * "auto-updated": written silently by older versions after every
          delineation (narrowed to whichever recording was analysed last);
        * a pre-V5 file (no provenance fields): those versions also
          re-saved the template narrowed to the last recording analysed, so
          it cannot be told apart from a contaminated one;
        * implausible for a mouse beat (``plausibility_problems``).

        Such a file can still be applied deliberately with Landmarks >
        Load… (checked on the recording's own beat).

        The notice is logged as a warning unless *quiet* (a label that only
        previews the template, refreshed on every file load, must not
        repeat it in the log each time).
        """
        path = WAVE_TEMPLATE_PATH
        if not path.exists():
            # defaults (+ legacy-pickle warning, unless quiet)
            return (cls() if quiet else cls.load()), None

        def _reject(reason: str, windows: str = "") -> "tuple[WaveTemplate, str]":
            notice = (f"The saved default wave template ({path}) was not used: "
                      f"{reason}." + (f" Its windows (ms from R): {windows}."
                                      if windows else "")
                      + " The built-in mouse defaults are used instead. To use "
                      "it anyway, open Landmarks (Beat Template tab), Load… "
                      "that file, check the windows on this recording's beat "
                      "and Apply.")
            if not quiet:
                log.warning("%s", notice)
            return cls(), notice

        try:
            data = cls._read_json(path)
        except ValueError as exc:
            return _reject(str(exc))
        obj = cls.from_dict(data)
        windows = format_windows(obj.landmarks)
        if obj.source == "auto-updated":
            return _reject("it was written automatically by an older version "
                           "(auto-updated from the last recording analysed)",
                           windows)
        if "origin" not in data:
            return _reject("it was saved by an older version, which re-saved "
                           "the template narrowed to the last recording "
                           "analysed", windows)
        problems = obj.plausibility_problems()
        if problems:
            return _reject("its windows are implausible for a mouse beat ("
                           + "; ".join(problems) + ")", windows)
        note = data.get("saved_from")
        if not isinstance(note, str):
            note = obj.origin_note(str(data.get("recording", "") or ""))
        obj.origin = note
        obj.source = "saved-default"
        return obj, None

    # ── Search window helpers ─────────────────────────────────────────

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
        5b. QRS onset and QRS offset (J point) come from ``qrs_bounds`` against
           the PR-segment baseline (see that function).
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

        # ── QRS onset / offset (J point) on the mean beat ─────────────────────
        # PR-segment baseline: the few ms between the P wave and the Q wave.
        base_hi = q_t - 6.0
        base_lo = max(p_t + 4.0, base_hi - 8.0)
        if base_hi - base_lo < 2.0:
            base_lo = base_hi - 3.0
        qon_t, qoff_t = qrs_bounds(bt, mb, q_t, s_t, j_t, base_lo, base_hi)
        if not np.isfinite(qon_t):
            qon_t = q_t - 4.0
        if not np.isfinite(qoff_t):
            qoff_t = s_t + 3.0
        qon_t  = float(np.clip(qon_t,  p_t + 1.0, min(q_t, -0.5)))
        qoff_t = float(np.clip(qoff_t, max(s_t, 0.5), t_t - 1.0))

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
            "QRS_onset": (round(qon_t,   1), _hw(qon_t,    p_t,      0.0,      min_hw=4.0)),
            "Q_peak":   (round(q_t,      1), _hw(q_t,      p_t,      0.0,      min_hw=4.0)),
            "S_peak":   (round(s_t,      1), _hw(s_t,      0.0,      j_t_final, min_hw=4.0)),
            "QRS_offset": (round(qoff_t, 1), _hw(qoff_t,   0.0,      t_t,      min_hw=4.0)),
            "J_peak":   (round(j_t_final,1), _hw(j_t_final,s_t,      t_t,      min_hw=4.0)),
            "T_peak":   (round(t_t,      1), _hw(t_t,      j_t_final, toff_t,  min_hw=8.0)),
            "T_offset": (round(toff_t,   1), _hw(toff_t,   t_t,      bt_max,   min_hw=5.0)),
        }
        self.source    = "auto-calibrated"
        self.confirmed = False
        log.info(
            "WaveTemplate.calibrate_from_beat: "
            "P_onset=%+.1f  P=%+.1f  QRS_onset=%+.1f  Q=%+.1f  R=0  S=%+.1f  "
            "QRS_offset=%+.1f  T=%+.1f  T_offset=%+.1f ms",
            po_t, p_t, qon_t, q_t, s_t, qoff_t, t_t, toff_t)


# QRS boundary detection: a deflection belongs to the QRS while it stays
# more than this fraction of the R amplitude (or QRS_NOISE_K x the PR-segment
# noise SD, whichever is larger) away from the PR-segment baseline.
QRS_THRESH_FRAC: float = 0.05
QRS_NOISE_K:     float = 3.0
# A Q (or S) wave counts as the initial (terminal) QRS deflection once it
# dips below the PR baseline by more than this fraction of R (or
# QRS_NOISE_K x the noise SD): with the 5 % threshold a small Q (3-4 % of R)
# was skipped and QRS onset moved to the R upstroke, ~3 ms late.
QRS_QS_THRESH_FRAC: float = 0.02
# P onset is searched back at most this far from the P peak, whatever the
# template's P_onset window (a mouse P upstroke lasts ~5-15 ms).
P_ONSET_SPAN_MS: float = 30.0
_QRS_MAX_WALK_MS: float = 30.0   # never search further than this from Q/R/S
# The edge line may meet the baseline at most this far beyond the first
# sample back inside the threshold band (a 5 % threshold on a slow initial
# deflection sits a few ms after the true onset).
_QRS_MAX_EXTRAP_MS: float = 5.0


def _moving_average(y: "np.ndarray", w: int) -> "np.ndarray":
    """Centred *w*-sample moving average, edge-padded (no pull towards 0
    at the ends, which would move a wave edge's 30/70 % crossings)."""
    y = np.asarray(y, dtype=float)
    if w < 2 or len(y) <= w:
        return y
    left = (w - 1) // 2
    yp = np.pad(y, (left, w - 1 - left), mode="edge")
    return np.convolve(yp, np.ones(w) / w, mode="valid")


def edge_baseline_crossing(
    t:        "np.ndarray",
    y:        "np.ndarray",
    base:     float,
    ext_dev:  float,
    dt_ms:    float,
    smooth_ms: float = 4.0,
    lo_frac:  float = 0.3,
    hi_frac:  float = 0.7,
    meet:     "Optional[float]" = None,
) -> float:
    """Time (ms) at which a wave edge, extrapolated as a straight line,
    meets the baseline level *base* (or the level *meet*, when given: the
    band is still taken relative to *base*).

    ``t``/``y`` hold the edge only (from the baseline side to the wave
    extremum, in either time order); ``ext_dev`` is the extremum's signed
    deviation from *base*. The line is a least-squares fit through the
    samples whose deviation lies between ``lo_frac`` and ``hi_frac`` of
    ``ext_dev`` (selected on a ``smooth_ms`` moving average so noise does
    not scatter the band), in the contiguous run that leads away from the
    extremum. Fitting many samples removes the systematic
    bias of the single steepest-sample tangent, whose slope is picked by
    noise and is therefore too steep (the end point then lands too close
    to the extremum, and more so the higher the sampling rate). When the
    band has fewer than 3 samples, the tangent at the steepest smoothed
    sample is used. NaN when the edge is flat.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2 or not np.isfinite(ext_dev) or ext_dev == 0:
        return float("nan")
    ys = _moving_average(y, max(1, int(round(smooth_ms / dt_ms))))
    frac = (ys - base) / ext_dev
    # Only the contiguous run walking away from the extremum: whatever lies
    # past the first return below lo_frac (a later wave, noise) is not this
    # edge.
    i_e  = int(np.argmax(frac))
    step = -1 if i_e > (len(frac) - 1) / 2 else 1
    band = np.zeros(len(frac), dtype=bool)
    k = i_e
    while 0 <= k < len(frac) and frac[k] >= lo_frac:
        band[k] = frac[k] <= hi_frac
        k += step
    level = base if meet is None else float(meet)
    if int(band.sum()) >= 3:
        slope, icpt = np.polyfit(t[band], y[band], 1)
        if abs(slope) > 1e-12:
            return float((level - icpt) / slope)
    d = np.gradient(ys, dt_ms)
    k = int(np.argmax(np.abs(d)))
    s = float(d[k])
    if abs(s) < 1e-12:
        return float("nan")
    return float(t[k] + (level - y[k]) / s)


def qrs_bounds(
    beat_time: "np.ndarray",
    beat:      "np.ndarray",
    q_t:       "Optional[float]",
    s_t:       "Optional[float]",
    j_t:       "Optional[float]",
    base_lo:   float,
    base_hi:   float,
) -> "tuple[float, float]":
    """QRS onset and QRS offset (J point), ms relative to R = 0.

    Reference level: the median of the PR segment ``[base_lo, base_hi]`` (ms),
    the conventional isoelectric reference for QRS onset; its SD is the noise
    estimate. Threshold: ``max(QRS_THRESH_FRAC * |R - baseline|,
    QRS_NOISE_K * noise SD)`` for the R wave, ``max(QRS_QS_THRESH_FRAC *
    |R - baseline|, QRS_NOISE_K * noise SD)`` for a Q or S wave.

    * **Onset** -- the initial QRS deflection is the Q wave when Q dips below
      the baseline by more than the Q/S threshold, otherwise the R upstroke. From
      that wave's extremum the trace is followed backwards until it re-enters
      the threshold band; the onset is where a least-squares line through
      the 30-70 % part of that initial deflection meets the baseline
      (``edge_baseline_crossing``: exact for a linear deflection,
      insensitive to the threshold value and to sample noise).
    * **Offset (J point)** -- the terminal deflection is the S wave when S
      dips below the baseline by more than the threshold, otherwise the R
      downstroke (same Q/S threshold). The trace is followed forwards until
      it returns to the
      threshold band, and the offset is where the line fitted through
      that return meets the baseline. When the J wave rises before the
      R downstroke reaches the baseline (no S), the J point is the notch
      (minimum) between R and the J peak. The J wave itself is early
      repolarisation in the mouse (Boukens et al. 2014, J Physiol
      592:4613), so it is not part of QRS.

    NaN for either bound that cannot be located (flat beat, empty window).
    """
    bt = np.asarray(beat_time, dtype=float)
    mb = np.asarray(beat,      dtype=float)
    n  = len(bt)
    nan = float("nan")
    if n < 10:
        return nan, nan
    dt_ms = float(bt[1] - bt[0])

    def _idx(t_ms: float) -> int:
        return int(np.argmin(np.abs(bt - t_ms)))

    i0, i1 = _idx(base_lo), _idx(base_hi) + 1
    if i1 - i0 < 2:
        return nan, nan
    base  = float(np.median(mb[i0:i1]))
    noise = float(np.std(mb[i0:i1]))
    i_r   = _idx(0.0)
    r_amp = float(mb[i_r] - base)
    thr   = max(QRS_THRESH_FRAC * abs(r_amp), QRS_NOISE_K * noise)
    if not np.isfinite(thr) or thr <= 0 or abs(r_amp) <= thr:
        return nan, nan
    qs_thr = max(QRS_QS_THRESH_FRAC * abs(r_amp), QRS_NOISE_K * noise)
    dev = mb - base
    max_walk = max(2, int(round(_QRS_MAX_WALK_MS / dt_ms)))

    def _ok(t: "Optional[float]") -> bool:
        return t is not None and bool(np.isfinite(t))

    def _tangent(lo: int, hi: int, i_ext: int) -> float:
        """Baseline intersection of the edge between samples [lo, hi] that
        runs from the baseline to the extremum at ``i_ext``: least-squares
        line through the samples at 30-70 % of that extremum's deviation
        (one noisy sample no longer sets the slope, which made QRS shrink
        with noise at high sampling rates); falls back to the tangent at
        the steepest ~1 ms-smoothed sample when that band has < 3 samples.
        Exact for a linear deflection."""
        lo, hi = max(lo, 0), min(hi, n - 1)
        if hi - lo < 1:
            return float(bt[lo])
        return edge_baseline_crossing(bt[lo:hi + 1], mb[lo:hi + 1], base,
                                      float(dev[i_ext]), dt_ms, smooth_ms=1.0)

    # ── Onset ───────────────────────────────────────────────────────────
    onset = nan
    i_q = _idx(q_t) if _ok(q_t) else None
    if i_q is not None and i_q < i_r and dev[i_q] < -qs_thr:
        start, inside = i_q, (lambda v: v < -qs_thr)   # Q wave: below baseline
    else:
        start, inside = i_r, (lambda v: v > thr)       # R upstroke
    k = start
    stop = max(0, start - max_walk)
    while k > stop and inside(dev[k - 1]):
        k -= 1
    if k > stop:                       # left the QRS: k-1 is back at baseline
        t_on = _tangent(k - 1, start, start)
        onset = float(np.clip(t_on, float(bt[k - 1]) - _QRS_MAX_EXTRAP_MS,
                              float(bt[start])))

    # ── Offset (J point) ────────────────────────────────────────────────
    offset = nan
    i_j = _idx(j_t) if _ok(j_t) else None
    i_s = _idx(s_t) if _ok(s_t) else None
    if i_s is not None and i_s > i_r and dev[i_s] < -qs_thr:
        start, inside = i_s, (lambda v: v < -qs_thr)   # S wave: below baseline
    else:
        start, inside = i_r, (lambda v: v > thr)       # R downstroke
    k = start
    stop = min(n - 1, start + max_walk)
    if i_j is not None and i_j > start:
        stop = min(stop, i_j)
    while k < stop and inside(dev[k + 1]):
        k += 1
    if k < stop:                       # returned to the baseline band at k+1
        t_off = _tangent(start, k + 1, start)
        offset = float(np.clip(t_off, float(bt[start]),
                               float(bt[k + 1]) + _QRS_MAX_EXTRAP_MS))
    elif i_j is not None and stop == i_j and i_j > start + 1:
        # J wave began before the trace got back to baseline: the J point
        # is the notch between the terminal QRS deflection and the J peak.
        offset = float(bt[start + int(np.argmin(mb[start:i_j + 1]))])

    return onset, offset


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



PR_BASELINE_WIDTH_MS: float = 6.0    # length of the PR-baseline segment
_PR_BASELINE_SPAN_MS: float = 40.0   # searched up to this far before Q / R


def pr_baseline_window(
    beat_time: "np.ndarray",
    beat:      "np.ndarray",
    p_t:       "Optional[float]",
    q_t:       "Optional[float]",
) -> "tuple[float, float]":
    """(lo_ms, hi_ms) of the PR-segment baseline used for QRS onset / J point.

    Found on the beat itself between the P peak (+3 ms) and the Q nadir (or
    R): the LATEST ``PR_BASELINE_WIDTH_MS`` segment that is as flat as the
    flattest one there (SD within 1.5x) and at the same level -- i.e. the
    isoelectric stretch that ends where the QRS starts, wherever that is.
    It is not tied to a template window, so a wide QRS (onset well before
    the default window) still gets a baseline outside the QRS. NaN, NaN
    when the span is too short to hold one.
    """
    bt = np.asarray(beat_time, dtype=float)
    mb = np.asarray(beat,      dtype=float)
    nan = float("nan")
    if len(bt) < 10:
        return nan, nan
    dt_ms = float(bt[1] - bt[0])
    hi_ms = float(q_t) if (q_t is not None and np.isfinite(q_t) and q_t < 0) else 0.0
    lo_ms = hi_ms - _PR_BASELINE_SPAN_MS
    if p_t is not None and np.isfinite(p_t) and p_t + 3.0 < hi_ms:
        lo_ms = max(lo_ms, float(p_t) + 3.0)
    lo_ms = max(lo_ms, float(bt[0]))
    i0 = int(np.searchsorted(bt, lo_ms - 1e-9))
    i1 = int(np.searchsorted(bt, hi_ms - 1e-9))        # exclusive: before Q / R
    w  = max(3, int(round(PR_BASELINE_WIDTH_MS / dt_ms)))
    if i1 - i0 < w:
        if i1 - i0 < 3:
            return nan, nan
        return float(bt[i0]), float(bt[i1 - 1])
    win  = np.lib.stride_tricks.sliding_window_view(mb[i0:i1], w)
    sd   = win.std(axis=1)
    mean = win.mean(axis=1)
    j0   = int(np.argmin(sd))
    tol  = max(3.0 * float(sd[j0]), 0.01 * float(np.ptp(mb)))
    ok   = (sd <= 1.5 * float(sd[j0]) + 1e-12) & (np.abs(mean - mean[j0]) <= tol)
    j    = int(np.flatnonzero(ok)[-1])
    return float(bt[i0 + j]), float(bt[i0 + j + w - 1])


def detect_waves_on_beat(
    beat:      "np.ndarray",
    beat_time: "np.ndarray",
    template:  "WaveTemplate",
    p_search_from_ms: "Optional[float]" = None,
    next_r_ms: "Optional[float]" = None,
) -> "dict[str, float]":
    """Detect P/Q/R/S/J/T landmarks on a single beat aligned on R=0.

    Parameters
    ----------
    beat      : 1-D float array, signal values (R aligned so beat_time≈0 at R peak).
    beat_time : 1-D float array, time in ms relative to R=0.
    template  : WaveTemplate providing (center_ms, half_window_ms) per landmark.
    p_search_from_ms : earliest time (ms from R) of the P peak: the previous
        beat's T peak (``analyse_intervals``). At mouse heart rates the beat
        window reaches back onto that T wave, which is not this beat's P.
        None: the template's P window (and the beat window) only.
    next_r_ms : time (ms from R) of the next beat's R. At mouse heart rates
        the window after R reaches the next beat's P wave, which is not this
        beat's T. None: nothing is excluded from the T search.

    Returns
    -------
    dict with keys P_onset, P_peak, QRS_onset, Q_peak, S_peak, QRS_offset,
    J_peak, T_peak, T_offset. Values are ms relative to R=0.  NaN where a
    landmark cannot be found.

    Algorithm notes
    ---------------
    • P wave : detected against the PR-segment baseline (the isoelectric
      level just before the QRS, ``pr_baseline_window``), after the previous
      beat's T peak (*p_search_from_ms*); candidates ranked by prominence.
    • J wave : small positive hump right after S (early repolarization, very
      common in mice).  Detected as local maximum in a narrow window after S.
      Its position is then used to set the T wave search START, preventing the
      J hump from being misidentified as the T wave.
    • T wave : search starts after J wave (or after S + fixed gap) and stops
      at the foot of the next beat's P wave (*next_r_ms*). Largest deflection
      from the baseline among the local extrema of either polarity, so both
      upright and inverted T waves are handled correctly.
    • P onset: leading-edge line meeting the PR-segment baseline, searched
      back up to ``P_ONSET_SPAN_MS`` from the P peak (not clamped to the
      template's P_onset window; NaN when it cannot be placed).
    • QRS onset / offset (J point): ``qrs_bounds`` against the PR-segment
      baseline found by ``pr_baseline_window``. Not clamped to the template
      windows (they are interval end points: NaN rather than a window-edge
      value when they cannot be placed).
    • T end: trailing-edge line (``edge_baseline_crossing``) meeting the
      baseline, searched up to the later of the template and default T-end
      window ends; NaN, not an edge value, when it falls outside.
    """
    bt = np.asarray(beat_time, dtype=float)
    mb = np.asarray(beat,      dtype=float)
    n  = len(bt)

    if n < 10:
        return {k: float("nan")
                for k in ("P_onset", "P_peak", "QRS_onset", "Q_peak", "S_peak",
                          "QRS_offset", "J_peak",  "T_peak",  "T_offset")}

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

    def _p_onset(t_peak_ms: float, base: float) -> float:
        """P onset: the P leading edge (least-squares line through its
        30-70 % band, ``edge_baseline_crossing``) meeting the isoelectric
        level *base*, searched back at most P_ONSET_SPAN_MS from the P peak
        -- not limited to the template's P_onset window. When the edge does
        not come back to *base* in that span (P riding on the previous T
        wave) the level of the T-P junction (the lowest point before the
        P peak, on the P's side) selects the edge band instead, and the
        line is still extended to *base*, but not before that junction: the
        part of the edge above the junction is the P alone once the T has
        ended (P onset hidden under the T end at RR ~100 ms), and the P
        cannot start before the signal stops falling when the T is still
        decaying under it. NaN -- never a window-edge value -- when the
        edge starts at the span / beat-window start or the line meets the
        baseline outside the searched span."""
        nan_ = float("nan")
        i_pk = _idx(t_peak_ms)
        i_lo = _idx(max(bt0, t_peak_ms - P_ONSET_SPAN_MS))
        if i_pk <= i_lo + 3 or not np.isfinite(base):
            return nan_
        seg = mb[i_lo:i_pk + 1]
        sm  = _moving_average(seg, max(1, int(round(2.0 / dt_ms))))
        ext = float(sm[-1] - base)
        if ext == 0:
            return nan_
        frac = (sm - base) / ext
        iso, t_jct = base, None
        j = int(np.argmin(frac[:-1]))               # lowest point before the peak
        # A band-pass leaves the PR segment below the level the P starts from
        # (undershoot before the QRS), so the lowest point is a slow ramp far
        # before the P: its real foot is where the steep edge begins.
        w = max(1, int(round(1.5 / dt_ms)))
        if (frac[j] < 0.3 or j <= 1) and len(sm) > 2 * w + 1:
            slope = np.zeros(len(sm))
            slope[w:-w] = sm[2 * w:] - sm[:-2 * w]
            slope *= np.sign(ext)
            k = int(np.argmax(slope))
            while k > w and slope[k - 1] > 0.25 * slope[int(np.argmax(slope))]:
                k -= 1
            if k > w and frac[k] >= 0.3:
                j = k
        if frac[j] >= 0.3:                          # T-P junction
            if j <= 1:                              # still rising at the span start
                return nan_
            base, i_lo, seg = float(sm[j]), i_lo + j, seg[j:]
            t_jct = float(bt[i_lo])
            ext = float(sm[-1] - base)
            if ext == 0:
                return nan_
        else:
            # The edge starts on the flat stretch that ends at that lowest
            # point: an earlier, larger wave (the previous T within the span
            # at RR ~110 ms) is not part of it.
            hi_before = np.flatnonzero(frac[:j] >= 0.3)
            if len(hi_before):
                k = int(hi_before[-1]) + 1
                i_lo, seg = i_lo + k, seg[k:]
        t_on = edge_baseline_crossing(bt[i_lo:i_pk + 1], seg, base, ext,
                                      dt_ms, smooth_ms=2.0, meet=iso)
        if t_jct is not None and np.isfinite(t_on):
            t_on = max(t_on, t_jct)
        if (not np.isfinite(t_on) or t_on < float(bt[i_lo]) - dt_ms
                or t_on >= t_peak_ms - 1.0):
            return nan_
        return float(t_on)

    def _slope_offset(t_peak_ms: float, end_ms: float,
                      base: "Optional[float]") -> float:
        """T end: the T-wave trailing edge (least-squares line through its
        30-70 % amplitude band, see ``edge_baseline_crossing``) meeting the
        isoelectric level *base* (the PR-segment baseline; the last 20 ms
        before *end_ms* when there is none). NaN when that line meets the
        baseline before the T peak or past *end_ms* (the T end is not inside
        the searched span) -- never a value clamped onto an edge."""
        i_pk = _idx(t_peak_ms)
        i_hi = min(n - 1, _idx(end_ms))
        if i_hi <= i_pk + 2:
            return float("nan")
        if base is None or not np.isfinite(base):
            base = _baseline(end_ms - 20.0, end_ms)
        seg  = mb[i_pk:i_hi + 1]
        sm   = _moving_average(seg, max(1, int(round(4.0 / dt_ms))))
        ext  = float(sm[0] - base)
        i_end = i_hi
        if ext != 0:
            # The edge stops where the trace turns back up towards the T
            # peak's side: that is the next beat's P wave, possibly larger
            # than the T, which the line must not be fitted on. The line may
            # still meet the baseline up to *end_ms* (P starting before the
            # T has ended).
            frac = (sm - base) / ext
            lows = np.minimum.accumulate(frac)
            up   = np.flatnonzero(frac - lows > max(0.1, 3.0 * noise_sd / abs(ext)))
            if len(up):
                i_end = i_pk + int(np.argmin(frac[:int(up[0]) + 1]))
                if i_end <= i_pk + 2:
                    i_end = i_hi
        t_off = edge_baseline_crossing(bt[i_pk:i_end + 1], mb[i_pk:i_end + 1],
                                       base, ext, dt_ms, smooth_ms=4.0)
        if not np.isfinite(t_off) or t_off <= t_peak_ms or t_off > float(bt[i_hi]):
            return float("nan")
        return float(t_off)

    def _descend(t0: "Optional[float]", edge_ms: float, step: int) -> "Optional[float]":
        """*t0* moved downhill (on a ~1 ms moving average) in direction
        *step* while it sits on the window edge *edge_ms*, at most
        _QRS_MAX_WALK_MS; unchanged otherwise."""
        if t0 is None or abs(t0 - edge_ms) > dt_ms:
            return t0
        sm = _moving_average(mb, max(1, int(round(1.0 / dt_ms))))
        k = _idx(t0)
        stop = k + step * int(round(_QRS_MAX_WALK_MS / dt_ms))
        while 0 < k < n - 1 and k != stop and sm[k + step] < sm[k]:
            k += step
        return float(bt[k])

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
    _,     to_hi = _win("T_offset")

    # ── Detect Q and S (minima near R) ────────────────────────────────────
    bt0, bt1 = float(bt[0]), float(bt[-1])

    # Q is searched strictly before R and S strictly after it: the default
    # windows straddle R (so a narrow Q/S is not missed) and, unrestricted,
    # the Q minimum could land on the S trough (or vice versa) and be
    # clamped onto R, which collapsed QRS and the QRS onset onto R.
    q_t = _extremum(q_lo, min(q_hi, -0.5), polarity=-1)
    s_t = _extremum(max(s_lo, 0.5), s_hi, polarity=-1)
    # A minimum on the OUTER edge of its window is a wave still falling
    # beyond it (a wide QRS, or a window narrowed to another recording by
    # "Refine from delineation"): follow it down to its real nadir, else
    # the QRS onset / J point are walked from a point on the wrong edge.
    q_t = _descend(q_t, q_lo, -1)
    s_t = _descend(s_t, s_hi, +1)

    # ── Detect J wave (positive hump right after S, before T) ────────────
    # J wave is the early-repolarization hump common in mouse ECG.
    # Search in the dedicated J_peak window (defaults: 7–23 ms after R).
    # If S was detected, also enforce that J comes after S.
    j_search_lo = max(j_lo, (s_t + 2.0) if s_t is not None else j_lo)
    j_search_hi = min(j_hi, t_lo - 2.0)   # never overlap T search
    j_t: "Optional[float]" = None
    if j_search_hi > j_search_lo + 2:
        j_t = _extremum(j_search_lo, j_search_hi, polarity=1)

    # ── QRS onset / offset (J point) ──────────────────────────────────────
    # PR-segment baseline found on the beat itself (pr_baseline_window), not
    # placed from the template's QRS_onset window: a window edge inside a
    # wide QRS put the baseline in the QRS and moved its onset. Found before
    # the P wave (no P peak needed: the latest flat stretch before Q / R is
    # the PR segment) so that the P search can stop short of the QRS -- the
    # default P window reaches -17 ms and otherwise took the Q wave of a
    # wide QRS for a negative P.
    pr_lo, pr_hi = pr_baseline_window(bt, mb, None, q_t)
    if np.isfinite(pr_lo):
        qon_t, qoff_t = qrs_bounds(bt, mb, q_t, s_t, j_t, pr_lo, pr_hi)
    else:
        qon_t = qoff_t = float("nan")
    # QRS onset / offset are interval END POINTS: they are NOT clamped to
    # their template windows (a refined window, median ± 4 ms of another
    # recording, would cut a wider QRS short). qrs_bounds already
    # keeps the onset at or before the Q nadir / R and the offset at or
    # after the S nadir / R, and returns NaN when no return to baseline is
    # found within 30 ms; beyond that only the ordering with the P and T
    # peaks is required (NaN otherwise, never an edge value).

    # The P wave ends before the QRS starts.
    if np.isfinite(qon_t):
        p_hi = min(p_hi, qon_t - 2.0)

    # ── Detect P wave (robust multi-step) ────────────────────────────────
    #
    # Mouse P waves are small, often only 5–20 % of R amplitude, and sit on
    # top of the decaying T wave (TP segment is short at high heart rates).
    # Strategy:
    #  1. Reference level: the PR-segment baseline (pr_baseline_window, found
    #     above from the QRS side), noise from that segment's residual. A
    #     "TP segment" 3-35 ms before the P window was used before: at RR
    #     <= ~120 ms it lies on the previous beat's T wave (the beat window
    #     reaches 70 % of the RR back), so the T level was subtracted, the
    #     flat TP stretch read as a negative P and its slope as noise (PR
    #     60 ms instead of 42 at RR 120). The TP segment is kept only as
    #     the fallback when the beat has no PR baseline.
    #  2. Search from the previous beat's T peak (*p_search_from_ms*) at the
    #     earliest: at RR <= ~100 ms that T peak is inside the P window.
    #  3. SG-smooth the P search segment to suppress high-frequency noise
    #     before peak picking.
    #  4. scipy find_peaks, both polarities, minimum prominence; keep the
    #     candidate of largest prominence among those whose amplitude above
    #     the reference exceeds the noise threshold. Prominence, not
    #     amplitude: a P riding on the previous T's tail stands out of it,
    #     while a noise wiggle on that tail is high but not prominent.
    #     None in the P window: a stricter search before it (very long PR).
    #  5. No candidate: NaN. The former fallbacks (the corrected extremum
    #     with a 1.2 x local-SD threshold, then the plain window maximum)
    #     only ever returned a window edge or a noise wiggle -- a real P is
    #     a find_peaks candidate -- and turned a T tail or noise into a P,
    #     hence a false PR (and AV-delay events) at high heart rates.

    from scipy.signal import savgol_filter as _sg_filt, find_peaks as _fp

    # Peaks are picked on the whole stretch from the previous T peak (or the
    # beat-window start) to p_hi, then kept inside the P window: a P peak on
    # the window's edge is still a local maximum, a wave only entering the
    # window across its edge is not.
    ctx_lo = bt0
    if p_search_from_ms is not None and np.isfinite(p_search_from_ms):
        ctx_lo = max(ctx_lo, float(p_search_from_ms))
    p_lo = max(p_lo, ctx_lo)

    # 1. Reference level and noise: PR segment, else the old TP segment.
    if np.isfinite(pr_lo):
        i_b0, i_b1 = _idx(pr_lo), _idx(pr_hi) + 1
        pre_baseline = float(np.median(mb[i_b0:i_b1]))
        tp_rms = float(np.std(mb[i_b0:i_b1]))
    else:
        tp_lo = max(bt0, p_lo - 35.0)
        tp_hi = p_lo - 3.0
        pre_baseline = _baseline(tp_lo, tp_hi)
        tp_rms = float(np.std(mb[max(0, _idx(tp_lo)): max(1, _idx(tp_hi))]))
    # Noise SD of the whole beat window from its high-frequency residual
    # (robust MAD, so the QRS does not count; / 0.89 = the residual's share
    # of white noise for an 11-sample cubic SG): the PR segment is only
    # 6 ms, chosen as the flattest, so its SD alone underestimates noise and
    # let noise wiggles pass for a P wave.
    noise_sd = 0.0
    if n >= 15:
        try:
            resid = mb - _sg_filt(mb, window_length=11, polyorder=3, mode="interp")
            noise_sd = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) / 0.89
        except Exception:
            noise_sd = 0.0
    # minimum P amplitude above baseline
    snr_thresh = max(tp_rms * 1.8, noise_sd * 3.0, 0.008)

    # 3. SG-smooth the P search segment (reduce noise sensitivity)
    i_p0, i_p1 = _window(ctx_lo, p_hi)
    i_lo_p     = _idx(p_lo)
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

    # 4. Both polarities; the most prominent candidate above the noise
    p_t: "Optional[float]" = None

    if n_seg >= 3 and p_hi > p_lo:
        prom_min = max(snr_thresh * 0.6, 0.003)
        min_dist_samples = max(1, int(4.0 / dt_ms))  # 4 ms min peak separation

        best_prom = 0.0
        for sign in (1, -1):
            peaks, props = _fp(sign * seg_smooth, prominence=prom_min,
                               distance=min_dist_samples)
            for pk_idx, prom in zip(peaks, props.get("prominences", [])):
                if i_p0 + pk_idx < i_lo_p:          # before the P window
                    continue
                # A negative candidate must clearly beat the upright one: the
                # trough between T and P (band-pass undershoot) is as
                # prominent as the P itself.
                if sign < 0 and p_t is not None and prom < 2.0 * best_prom:
                    continue
                if abs(float(seg_smooth[pk_idx])) >= snr_thresh and prom > best_prom:
                    best_prom = float(prom)
                    p_t = float(bt[i_p0 + pk_idx])

        # A very long PR (> ~70 ms: first-degree AV block) puts the P peak
        # before the template window. Only with the previous T peak known, so
        # that it cannot be taken for the P, and upright and well above the
        # noise, since a false long PR raises a false AV-delay event.
        if p_t is None and ctx_lo > bt0 - 1.0 and p_search_from_ms is not None:
            peaks, props = _fp(seg_smooth, prominence=2.0 * prom_min,
                               distance=min_dist_samples)
            for pk_idx, prom in zip(peaks, props.get("prominences", [])):
                if (i_p0 + pk_idx < i_lo_p and prom > best_prom
                        and float(seg_smooth[pk_idx]) >= snr_thresh):
                    best_prom = float(prom)
                    p_t = float(bt[i_p0 + pk_idx])
            if p_t is not None:
                p_lo = min(p_lo, p_t)
    # 5. No candidate: no P wave (NaN P peak, P onset and PR).

    # ── Detect T wave (starts after J, or after S + gap) ─────────────────
    # Starting after J prevents the J hump from being selected as T.
    t_lo_eff = max(t_lo, (j_t + 5.0) if j_t is not None else t_lo)
    # Local extrema from there to the end of the beat window, against the
    # PR-segment reference. Extrema, not the largest sample: a T smaller than
    # the next beat's P wave lost to it, and to the J tail at the window start.
    i_t0   = _idx(min(t_lo_eff, (j_t + 5.0) if j_t is not None else t_lo_eff))
    t_seg  = mb[i_t0:] - pre_baseline
    t_sm   = t_seg
    t_cand: "list[tuple[float, int, float, int]]" = []     # time, sign, prom, i
    t_stop = bt1 + 1.0
    if len(t_seg) >= 5:
        w_t = max(5, int(round(5.0 / dt_ms)) | 1)
        if len(t_seg) >= w_t:
            t_sm = _sg_filt(t_seg, window_length=w_t, polyorder=3, mode="interp")
        prom_t = max(snr_thresh * 0.6, 0.003)
        for sign in (1, -1):
            peaks, props = _fp(sign * t_sm, prominence=prom_t)
            t_cand += [(float(bt[i_t0 + k]), sign, float(pr), int(k))
                       for k, pr in zip(peaks, props["prominences"])
                       if sign * t_sm[k] > 0]
        t_cand.sort()
        # The next beat's P wave: the latest extremum of this beat's P
        # polarity inside the next beat's P window (the largest one there may
        # be this beat's T at RR <= ~100 ms; noise wiggles are much smaller).
        # Alone, it is that P only in the second half of the RR, else it is
        # the T. The T search stops at the foot of that P.
        if next_r_ms is not None and np.isfinite(next_r_ms):
            p_lo0, p_hi0 = _win("P_peak")
            if p_t is not None:                 # this beat's own long PR
                p_lo0 = min(p_lo0, p_t - 5.0)
            p_sign = -1 if p_t is not None and seg_smooth[_idx(p_t) - i_p0] < 0 else 1
            in_p = [c for c in t_cand if c[1] == p_sign
                    and next_r_ms + p_lo0 <= c[0] <= next_r_ms + p_hi0]
            if in_p:
                top = max(c[2] for c in in_p)
                nxt = [c for c in in_p if c[2] >= 0.5 * top][-1]
                if nxt[0] >= 0.5 * next_r_ms or any(c[0] < nxt[0] for c in t_cand):
                    v = p_sign * t_sm
                    k = low = nxt[3]
                    while (k > 0 and v[k - 1] < v[low] + noise_sd
                           and v[k - 1] > 0.1 * v[nxt[3]]):
                        k -= 1
                        if v[k] < v[low]:
                            low = k
                    t_stop = float(bt[i_t0 + low])
    t_cand = [c for c in t_cand if t_lo_eff <= c[0] < t_stop and c[0] <= t_hi]
    # The largest sample unless an extremum comes close to it: alone it is a
    # wave's flank on the window edge, against which a noise wiggle is no T.
    t_t, _ = _abs_extremum(t_lo_eff, min(t_hi, t_stop))
    if t_cand and np.isfinite(t_t):
        best = max(t_cand, key=lambda c: abs(t_sm[c[3]]))
        if abs(t_sm[best[3]]) >= 0.5 * abs(mb[_idx(t_t)] - pre_baseline):
            t_t = best[0]
    # Noise-floor gate: _abs_extremum always returns the largest-magnitude
    # sample in its window, even on a flat/noisy segment with no real T
    # wave -- unlike the P-wave detector above, which rejects a candidate
    # below its local SNR threshold. Reuses the same reference level / noise
    # estimate (pre_baseline, tp_rms: the PR segment) computed for the P
    # wave, since both represent this beat's general baseline noise level. A rejected
    # T-peak surfaces as NaN in QT/QTc rather than a spurious value.
    if np.isfinite(t_t):
        t_amp = float(mb[_idx(t_t)] - pre_baseline)
        t_noise_floor = max(tp_rms * 1.5, 0.006)
        if abs(t_amp) < t_noise_floor:
            t_t = float("nan")

    # Reference level for P onset and T end: the PR-segment baseline, the
    # one QRS onset and the J point are measured against.
    pr_base = (float(np.median(mb[_idx(pr_lo):_idx(pr_hi) + 1]))
               if np.isfinite(pr_lo) else None)

    # ── P onset ───────────────────────────────────────────────────────────
    # An interval end point like QRS onset and T end: not clamped to the
    # P_onset template window (which capped it ~9 ms before the P peak with
    # the defaults and cut long PRs short), NaN when it cannot be placed.
    po_t = (_p_onset(p_t, pr_base if pr_base is not None else pre_baseline)
            if p_t is not None else float("nan"))

    # ── T offset ──────────────────────────────────────────────────────────
    # Searched up to the later of the template's and the default T-end
    # window ends (within the beat): a refined template, narrowed to one
    # recording's spread, must not cut a longer QT short in another.
    to_end = min(bt1, max(to_hi, sum(WaveTemplate.DEFAULTS["T_offset"])))
    # Reference level: the PR-segment baseline (pr_base, above). The end of
    # the beat window is still inside the T wave at high mouse heart rates,
    # so a baseline taken there moved the T end.
    to_t   = (_slope_offset(t_t, to_end, pr_base)
              if np.isfinite(t_t) and to_end > t_t else float("nan"))

    # Ordering with the P and T peaks (see above).
    if np.isfinite(qon_t) and (qon_t <= ((p_t + 1.0) if p_t is not None else bt0)):
        qon_t = float("nan")
    if np.isfinite(qoff_t) and (qoff_t >= ((t_t - 1.0) if np.isfinite(t_t) else bt1)):
        qoff_t = float("nan")

    # Final sanity clamps: bound each landmark by the INTERSECTION of (a) its
    # actual per-template search window (p_lo/p_hi, q_hi, ... computed
    # above from template.landmarks, so a user's calibrated template is
    # genuinely respected instead of being silently overridden) and (b) the
    # basic physiological ordering constraint the old hardcoded numbers were
    # really protecting (Q before R, S/J/T after R, etc.) intersected with
    # the beat_time array's own extent (bt0/bt1). Previously these bounds
    # were hardcoded numbers unrelated to q_lo/q_hi etc., so a custom
    # template's window had no effect on the final clamp. Interval end
    # points (P onset, QRS onset / offset, T end) are only kept inside the
    # beat, never moved onto a window edge.
    return {
        # P onset is an interval end point: NaN outside the beat, not clamped.
        "P_onset":  (float(po_t) if np.isfinite(po_t) and bt0 <= po_t <= -2.0
                     else float("nan")),
        "P_peak":   _clamp(p_t,  max(p_lo,  bt0+1),  min(p_hi,  -3.0)),
        "QRS_onset": _clamp(qon_t, bt0+1, -0.5),
        # Q / S may lie past their window's outer edge (_descend).
        "Q_peak":   _clamp(q_t,  bt0+1,              min(q_hi,  -0.5)),
        "S_peak":   _clamp(s_t,  max(s_lo,   0.5),   bt1-1),
        "QRS_offset": _clamp(qoff_t, 0.5, bt1-1),
        "J_peak":   _clamp(j_t,  max(j_lo,   3.0),   min(j_hi,  bt1-1)),
        "T_peak":   _clamp(t_t,  max(t_lo,   5.0),   min(t_hi,  bt1-1)),
        "T_offset": _clamp(to_t, 6.0, bt1),
    }

