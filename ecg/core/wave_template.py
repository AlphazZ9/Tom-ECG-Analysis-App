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
# Initial QRS wave (qrs_bounds initial_wave): a deflection of the other sign
# just before the Q (or R) the onset is walked from, peaking at most
# QRS_INITIAL_GAP_MS before the first sample where that Q (R) is back in its
# threshold band and departing, on a QRS_INITIAL_SMOOTH_MS moving average,
# by more than QRS_INITIAL_FRAC of R and QRS_INITIAL_NOISE_K x the noise SD
# from both the PR baseline and the level it rises from (the median of the
# QRS_INITIAL_FOOT_MS ending QRS_INITIAL_RISE_MS before its peak): the QRS
# starts with it (248: +18 % of R at -6 ms before a Q at -2.5; 247: +2-3 %
# at -7.5 before a Q at -4.5, 13-20 x the noise; both start at -8 to
# -9.5 ms, the Q edge at -4 / -6), where the line through the times its
# moving average rises through QRS_INITIAL_LEVELS of it meets that level.
# Without the moving average, the Q/S threshold (2 % of R or 3 x the noise)
# took single noisy samples for a wave (7 % of the rows of a 248-like
# synthetic with none, at 2 x its noise); with it, that threshold missed the
# 247 wave on 6 % of the rows. Against the PR baseline alone, a PR window in
# the last phase of a biphasic P made the whole PR segment a "wave" (onset
# -17 to -23 ms on 3 % of the rows of a synthetic without one). Decided on
# wide medians only (analysis _ensemble_qrs_plans): on a row's own N = 32
# median, the noise made a 3 %-of-R wave count on some rows only.
QRS_INITIAL_GAP_MS:    float = 3.0
QRS_INITIAL_SMOOTH_MS: float = 1.5
QRS_INITIAL_FRAC:      float = 0.01
QRS_INITIAL_NOISE_K:   float = 5.0
QRS_INITIAL_RISE_MS:   float = 5.0
QRS_INITIAL_FOOT_MS:   float = 3.0
QRS_INITIAL_LEVELS: "tuple[float, ...]" = (0.3, 0.5, 0.7)
# P onset is searched back at most this far from the P peak, whatever the
# template's P_onset window (a mouse P upstroke lasts ~5-15 ms).
P_ONSET_SPAN_MS: float = 30.0
# A P counts as biphasic (p_onset_biphasic) when the trough before its peak
# deviates from the baseline the other way by at least this fraction of the
# peak's deviation. Its onset is the up-crossing after that trough only when
# the P peak follows within P_BIPHASIC_SPAN_MS (2-6 ms on the real biphasic P
# waves at N = 32, whatever the size of a shoulder in between); a trough
# farther from the peak is the previous T's tail followed by a flat stretch,
# and the leading-edge line applies.
P_BIPHASIC_TROUGH_FRAC: float = 0.25
P_BIPHASIC_SPAN_MS:     float = 8.0
# Biphasic T (t_phase_noise): the phases of the T are the sign runs of the
# trace against the PR baseline (4 ms moving average). The run next to the
# T peak's, of the other sign, is a second T phase when it reaches both
# T_PHASE_FRAC of the peak's deviation and T_PHASE_NOISE_X times the noise
# SD of that moving average; below T_PHASE_ABSENT_FRAC of the peak, or
# T_PHASE_ABSENT_NOISE_X times the noise when a T_PHASE_FRAC phase would
# pass the noise test, it is baseline noise (and so is any run that small
# between two phases); in between the T end is ambiguous (NaN). A run is a
# T phase only if it follows the other one within T_PHASE_GAP_MS (no
# isoelectric stretch in between) and peaks inside the T search span, or at
# most T_PHASE_EDGE_MS before its start (a first phase cut by the window
# edge).
T_PHASE_FRAC:           float = 0.3
T_PHASE_ABSENT_FRAC:    float = 0.15
T_PHASE_NOISE_X:        float = 4.0
T_PHASE_ABSENT_NOISE_X: float = 2.5
T_PHASE_GAP_MS:         float = 4.0
T_PHASE_EDGE_MS:        float = 2.0
# With a phase plan (t_plan, decided on a wider median), a row's T is found
# by least squares (_t_from_plan): the plan's whole T moved within
# +-T_PLAN_SHIFT_MS of where it is on the plan, then each phase alone within
# T_PLAN_REFINE_MS of that lag, by T_PLAN_LAG_STEP_MS; a fit whose gain is
# under T_PLAN_SNR times its noise SD is not used (on a low biphasic T at
# 1.7 x the 248 noise every row keeps a T end, bias +0.4 to +0.7 ms; at
# 3.6 x 45 % do, the others are NaN). A phase that fails the noise test is
# read at the lag of the whole T, but a beat on which it is clearly missing
# (gain under T_PLAN_ABSENT_GAIN of the plan's by 2 SD) gets no T end. The
# plan's own T end is its edge line through the times its 4 ms moving
# average falls through the T_PLAN_LEVELS fractions of the phase, after the
# last sample at or above T_PLAN_EDGE_TOP of it, and none when a crossing is
# more than T_PLAN_LINE_TOL_MS off that line (a median blending a QT
# change); the plan trace is kept from 2 * T_PLAN_TOL_MS before its first
# phase's peak to as far after its T end.
T_PLAN_SNR:             float = 3.0
T_PLAN_SHIFT_MS:        float = 15.0
T_PLAN_REFINE_MS:       float = 4.0
T_PLAN_ABSENT_GAIN:     float = 0.5
T_PLAN_LINE_TOL_MS:     float = 1.0
T_PLAN_LAG_STEP_MS:     float = 0.25
T_PLAN_TOL_MS:          float = 8.0
T_PLAN_LEVELS: "tuple[float, ...]" = (0.3, 0.4, 0.5, 0.6, 0.7)
T_PLAN_EDGE_TOP:        float = 0.85
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
    initial_wave: bool = False,
    initial_out:  "Optional[dict]" = None,
    initial_plan: "Optional[dict]" = None,
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
      With *initial_wave* (the wide medians of ``analysis``
      ``_ensemble_qrs_plans``), a deflection of the other sign that the Q
      (R) runs straight out of -- peaking at most ``QRS_INITIAL_GAP_MS``
      before that wave is back in its band, and above both the PR baseline
      and the level just before its leading edge by more than
      ``QRS_INITIAL_FRAC`` of R and ``QRS_INITIAL_NOISE_K`` x the noise SD,
      on a ``QRS_INITIAL_SMOOTH_MS`` moving average -- is the initial QRS
      deflection, and the onset is its leading-edge line meeting that level
      instead (the Q edge alone started the QRS 3-4 ms late on 247 / 248);
      *initial_out* receives ``found``, ``from_q`` (walked from a Q, not
      the R) and ``lead`` (that onset minus the Q / R edge onset, ms). With
      *initial_plan* (such a dict, from the median of the beats around a
      row), the onset is the Q / R edge onset plus the plan's ``lead`` when
      the plan found a wave before the same kind of edge, and the edge onset
      otherwise: the wave is not decided again on the row. Neither on a
      single beat: the 247 wave (2-3 % of R) is under 5 x a single beat's
      noise (~1.3 % of R), so it would count on some beats only.
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
    from_q = i_q is not None and i_q < i_r and dev[i_q] < -qs_thr
    if from_q:
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
        if initial_plan is not None:
            # The initial wave as found on the median of the beats around
            # (initial_out of that median): its lead over the Q (R) edge
            # onset, on this beat's own Q (R) edge. Not decided again here:
            # near the threshold the noise of one row's median made the wave
            # count on some rows and not on the next (onset ~3 ms apart).
            if (initial_plan.get("found")
                    and bool(initial_plan.get("from_q")) == from_q):
                onset += float(initial_plan["lead"])
        elif initial_wave:
            found, onset_q = False, onset
            sgn = -1.0 if from_q else 1.0
            ma = _moving_average(mb, max(1, int(round(QRS_INITIAL_SMOOTH_MS / dt_ms))))
            dw = -sgn * (ma - base)
            rise = int(round(QRS_INITIAL_RISE_MS / dt_ms))
            foot = max(2, int(round(QRS_INITIAL_FOOT_MS / dt_ms)))
            # Not clipped to the PR window: on a median whose PR window
            # ended inside the wave's foot, that clipped the wave's peak.
            j0 = k - 1 - int(round(QRS_INITIAL_GAP_MS / dt_ms))
            if j0 - rise - foot >= 0:
                i_w = j0 + int(np.argmax(dw[j0:k]))
                f1 = i_w - rise
                f0 = f1 - foot
                # The level the wave rises from: the trace just before its
                # leading edge. A PR window in the last phase of a biphasic P
                # put the whole PR segment more than the threshold above the
                # PR "baseline" (onset at -23 ms instead of -5).
                lvl = float(np.median(mb[f0:f1 + 1]))
                a_foot = -sgn * (float(ma[i_w]) - lvl)
                # Noise: of the PR window, or of the stretch just before the
                # foot when lower (a PR window ending in the wave's foot
                # read the wave's slope as noise).
                pre = mb[max(0, f0 - (i1 - i0)):f0]
                nz = min(noise, float(np.std(pre))) if len(pre) >= 3 else noise
                w_thr = max(QRS_INITIAL_FRAC * abs(r_amp), QRS_INITIAL_NOISE_K * nz)
                # A peak inside the span (still rising at its start, it is
                # the end of a P running into the QRS), above both the PR
                # baseline and the level it rises from.
                if i_w > j0 and min(float(dw[i_w]), a_foot) > w_thr:
                    # Its leading edge: the line through the times the moving
                    # average last rises through QRS_INITIAL_LEVELS of the
                    # wave before its peak, meeting that level. The samples'
                    # 30-70 % band holds 2-3 samples of a 2 ms edge, and the
                    # steepest-sample fallback took a noise wiggle 7 ms
                    # before the wave for the edge (onset at -14 ms).
                    yy = -sgn * (ma - lvl)
                    ts = []
                    for lv in QRS_INITIAL_LEVELS:
                        j = i_w
                        while j > f0 and yy[j - 1] >= lv * a_foot:
                            j -= 1
                        if j <= f0:
                            break
                        ts.append(float(bt[j - 1]) + dt_ms * (lv * a_foot - yy[j - 1])
                                  / (yy[j] - yy[j - 1]))
                    t_w = (float(np.polyfit(QRS_INITIAL_LEVELS, ts, 1)[1])
                           if len(ts) == len(QRS_INITIAL_LEVELS) else float("nan"))
                    if np.isfinite(t_w):
                        t_w = float(np.clip(t_w, float(bt[f0]), float(bt[i_w])))
                        if t_w < onset:
                            onset, found = t_w, True
            if initial_out is not None:
                initial_out.update(found=found, from_q=bool(from_q),
                                   lead=float(onset - onset_q) if found else 0.0)

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


def pr_baseline_level(
    beat_time: "np.ndarray",
    beat:      "np.ndarray",
    q_t:       "Optional[float]",
) -> float:
    """PR-segment level the T end of a rolling median is read against
    (``detect_waves_on_beat`` *t_phase_noise*): the mean level of every
    ``PR_BASELINE_WIDTH_MS`` segment of the ``pr_baseline_window`` span,
    weighted by (flattest SD / its SD) ** 4. ``pr_baseline_window`` picks
    ONE segment, and where the PR segment slopes, two segments of about the
    same flatness took turns on neighbouring medians (247: levels 0.03
    apart, the T end jumping by 1.7 ms in blocks of rows); the weighted
    level changes smoothly with the trace. NaN when the span is too short.
    """
    bt = np.asarray(beat_time, dtype=float)
    mb = np.asarray(beat,      dtype=float)
    if len(bt) < 10:
        return float("nan")
    dt_ms = float(bt[1] - bt[0])
    hi_ms = float(q_t) if (q_t is not None and np.isfinite(q_t) and q_t < 0) else 0.0
    lo_ms = max(hi_ms - _PR_BASELINE_SPAN_MS, float(bt[0]))
    i0 = int(np.searchsorted(bt, lo_ms - 1e-9))
    i1 = int(np.searchsorted(bt, hi_ms - 1e-9))
    w  = max(3, int(round(PR_BASELINE_WIDTH_MS / dt_ms)))
    if i1 - i0 < w:
        return float("nan")
    win  = np.lib.stride_tricks.sliding_window_view(mb[i0:i1], w)
    sd   = win.std(axis=1)
    wt   = (max(float(sd.min()), 1e-12) / np.maximum(sd, 1e-12)) ** 4
    return float(wt @ win.mean(axis=1) / wt.sum())


# How far before its peak a P wave's start is looked for (detect_waves_on_beat:
# the T peak is not searched past the next beat's P start).
NEXT_P_SPAN_MS = 25.0


def detect_waves_on_beat(
    beat:      "np.ndarray",
    beat_time: "np.ndarray",
    template:  "WaveTemplate",
    p_search_from_ms: "Optional[float]" = None,
    next_r_ms: "Optional[float]" = None,
    p_onset_biphasic: bool = False,
    t_phase_noise: "Optional[float]" = None,
    t_plan: "Optional[dict]" = None,
    t_plan_out: "Optional[dict]" = None,
    qrs_initial_wave: bool = False,
    qrs_initial_out:  "Optional[dict]" = None,
    qrs_initial_plan: "Optional[dict]" = None,
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
    p_onset_biphasic : a P whose initial deflection is opposite to its peak
        (biphasic / notched P, trough at least ``P_BIPHASIC_TROUGH_FRAC`` of
        the peak's deviation) starts where the trace comes back up through
        the baseline after that trough (see ``_p_onset``). Set by
        ``analyse_intervals`` for the rows measured on a rolling median
        (ensemble_beats > 1) only: a row measured on its own beat (timing
        outlier, too few ensemble members, N = 0 or 1) keeps the leading-edge
        line.
    qrs_initial_wave : a small deflection of the other sign that the Q (or
        R) runs straight out of starts the QRS (``qrs_bounds``
        *initial_wave*), found on this beat; what was found goes into
        *qrs_initial_out*. Set by ``analyse_intervals`` on the wide medians
        of ``_ensemble_qrs_plans`` only, never on a row.
    qrs_initial_plan : the initial wave as found on the median of the beats
        around (``qrs_bounds`` *initial_plan*): a row's QRS onset is its own
        Q / R edge onset moved by that wave's lead. Passed by
        ``analyse_intervals`` for the rows measured on a rolling median
        (ensemble_beats > 1); None: the Q / R edge.
    t_phase_noise : noise SD of the beat's 4 ms moving average. Given, a
        biphasic T (two adjacent phases of opposite sign, the smaller at
        least ``T_PHASE_FRAC`` of the larger and ``T_PHASE_NOISE_X`` times
        this noise) ends where its LAST phase returns to the baseline, and
        its T peak is the peak of its FIRST phase, whichever phase is the
        larger (see ``_t_phases``); the line meeting the baseline is that
        through the times the moving average first falls through the
        ``T_PLAN_LEVELS`` fractions of the phase (``_edge``), not through
        the samples of its 30-70 % band, and the baseline is the
        flatness-weighted PR level (``pr_baseline_level``), not the one PR
        segment QRS onset is measured from. NaN T end when the second phase
        is too small to be told from noise. Passed by ``analyse_intervals`` with
        *p_onset_biphasic* (rows measured on a rolling median); None: the T
        end is that of the largest phase, so two phases of similar size
        take turns at ending the T.
    t_plan : with *t_phase_noise*, a phase plan (filled in *t_plan_out* by
        this function on a median of more beats: ``analyse_intervals`` uses
        the median of the beats around the row, ``_ensemble_t_plans``): the
        phases are not decided again on this beat, whose noise made a
        second phase near the thresholds count on some rows and not on
        others (T end jumping by one phase in blocks of rows). The plan's
        T, then each of its phases, is fitted to this beat by least squares
        over a lag; the T end is the plan's T end moved by the lag of the
        last phase (of the whole T when that phase alone is not told from
        noise); NaN T peak and T end when the T is not told from noise or
        the plan is ambiguous (see ``_t_from_plan``).
    t_plan_out : a dict filled with the phase plan of this beat (with
        *t_phase_noise*, without *t_plan*): ``phases`` (1 or 2),
        ``first_ms`` / ``first_sign``, ``last_ms`` / ``last_sign``,
        ``end_ms`` (its T end) and ``edge_t`` / ``edge_v`` (the 4 ms moving
        average against the baseline over its T);
        ``{"ambiguous": True}`` when the second phase is too small to be
        told from noise; left empty when no T end is found.

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
      window ends; NaN, not an edge value, when it falls outside. On a
      rolling median (*t_phase_noise*), that of the LAST phase of a
      biphasic T, and with *t_plan* the phase plan of a wider median read on
      the beat (``_t_from_plan``).
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

    def _p_onset(t_peak_ms: float, base: float, min_dev: float = 0.0) -> float:
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
        decaying under it. With *p_onset_biphasic*, a biphasic P (trough the
        other way first, at least *min_dev* -- the P detection threshold --
        below *base*) starts at the up-crossing after that trough instead.
        NaN -- never a window-edge value -- when the
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
        # Biphasic / notched P (p_onset_biphasic): the initial deflection is
        # opposite to the peak (and a wave: above the P detection threshold,
        # not a noise dip), and a shoulder between it and the peak breaks
        # the edge line (a flat fit through the shoulder, or the notch
        # upstroke ~4 ms late). The onset is where the trace first comes
        # back through the baseline after that trough, provided the P peak
        # follows closely: a trough followed by a flat stretch and then the
        # P is the end of the previous T, not the P's first phase. Not a
        # test of the rise right after the crossing, which a small shoulder
        # fails on one median and passes on the next (rows flipping by 4 ms).
        if (p_onset_biphasic and frac[j] <= -P_BIPHASIC_TROUGH_FRAC
                and -frac[j] * abs(ext) >= min_dev):
            up = np.flatnonzero((frac[j:-1] < 0) & (frac[j + 1:] >= 0))
            if len(up):
                q = j + int(up[0])
                t_up = float(bt[i_lo + q]) + dt_ms * (-frac[q]) / (frac[q + 1] - frac[q])
                if t_peak_ms - P_BIPHASIC_SPAN_MS <= t_up < t_peak_ms - 1.0:
                    return t_up
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

    def _edge(tt: "np.ndarray", yy: "np.ndarray", amp: float,
              t_from: float, t_to: float) -> "Optional[np.ndarray]":
        """Times at which the trailing edge of yy (the phase made positive)
        falls through T_PLAN_LEVELS of *amp*: the first falls after its
        last sample at or above T_PLAN_EDGE_TOP of *amp* in t_from..t_to
        (so a noise wiggle in the trough or near the baseline is not the
        edge); linear between samples. None when there is no such edge."""
        k0 = int(np.searchsorted(tt, t_from, side="left"))
        k1 = int(np.searchsorted(tt, t_to, side="right"))
        y = yy[k0:k1]
        top = np.flatnonzero(y >= T_PLAN_EDGE_TOP * amp)
        if not len(top):
            return None
        out = []
        for lv in T_PLAN_LEVELS:
            below = np.flatnonzero(y[top[-1]:] < lv * amp)
            if not len(below):
                return None
            m = k0 + top[-1] + int(below[0])
            y0, y1 = yy[m - 1], yy[m]
            out.append(tt[m - 1] + (y0 - lv * amp) / (y0 - y1) * (tt[m] - tt[m - 1]))
        return np.asarray(out)

    def _slope_offset(t_peak_ms: float, end_ms: float,
                      base: "Optional[float]", fit_smoothed: bool = False,
                      stop_ms: "Optional[float]" = None,
                      line_tol: "Optional[float]" = None) -> float:
        """T end: the T-wave trailing edge (least-squares line through its
        30-70 % amplitude band, see ``edge_baseline_crossing``) meeting the
        isoelectric level *base* (the PR-segment baseline; the last 20 ms
        before *end_ms* when there is none). NaN when that line meets the
        baseline before the T peak or past *end_ms* (the T end is not inside
        the searched span) -- never a value clamped onto an edge. With
        *fit_smoothed* the line is fitted on the 4 ms moving average rather
        than on the samples: the edge of a small T phase holds only ~3
        samples in its 30-70 % band, too few for a line through noise; with
        *stop_ms* the edge ends there at the latest (the phase's first
        return through the baseline, see ``_t_phases``); with *line_tol*
        (a phase plan's median, ``t_plan_out``) NaN when a crossing time is
        more than *line_tol* ms off that line: the edge is two edges, that
        median blending beats before and after a QT change (the line from
        the early 70 % and the late 30 % crossings ended 10-25 ms late)."""
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
        if stop_ms is not None and np.isfinite(stop_ms) and i_pk + 2 < _idx(stop_ms) < i_end:
            i_end = _idx(stop_ms)
        if fit_smoothed:
            # The line through the times the 4 ms moving average first falls
            # through each T_PLAN_LEVELS fraction of the phase (see _edge):
            # a line fitted on the samples of the 30-70 % band took a small
            # plateau in it for the slope (T end 25 ms late on 248).
            # The phase amplitude is that moving average's at the peak: sm,
            # edge-padded at the peak, weighs the peak sample alone, and a
            # noise spike there put 85 % of it out of the trace's reach (no
            # T end on a tenth of the 248 medians).
            sg = np.sign(ext)
            ma = sg * (_moving_average(mb, max(1, int(round(4.0 / dt_ms))))[i_pk:i_end + 1]
                       - base)
            e = (_edge(bt[i_pk:i_end + 1], ma, float(ma[0]), float(bt[i_pk]),
                       float(bt[i_end])) if ma[0] > 0 else None)
            t_off = float("nan")
            if e is not None:
                fit = np.polyfit(np.asarray(T_PLAN_LEVELS), e, 1)
                bent = np.max(np.abs(np.polyval(fit, np.asarray(T_PLAN_LEVELS)) - e))
                if line_tol is None or bent <= line_tol:
                    t_off = float(fit[1])
        else:
            t_off = edge_baseline_crossing(bt[i_pk:i_end + 1], mb[i_pk:i_end + 1],
                                           base, ext, dt_ms, smooth_ms=4.0)
        if not np.isfinite(t_off) or t_off <= t_peak_ms or t_off > float(bt[i_hi]):
            return float("nan")
        return float(t_off)

    small_phase = [False]      # set by _t_phases, read by _t_plan_of

    def _t_phases(t_peak_ms: float, lo_ms: float, hi_ms: float,
                  base: float, noise: float) -> "tuple[float, float, float]":
        """(first phase peak, last phase peak, end of the last phase's run)
        in ms of the T wave whose largest phase peaks at *t_peak_ms*, phases
        searched between *lo_ms* and *hi_ms* (the T search span), *noise*
        the SD of the 4 ms moving average; NaN when the T end is ambiguous.
        At most two phases: the largest one, and the run of the other sign
        right before it or, failing that, right after it (see T_PHASE_FRAC).
        The one before is tried first, so a T whose two phases are of
        similar size gives the same pair whichever of them is the larger.
        The run ends at the trace's first return through the baseline: the
        last phase's edge stops there, not on a later noise trough of the
        same sign (which, deeper than a low phase, took the edge line)."""
        nan_ = float("nan")
        v = _moving_average(mb, max(1, int(round(4.0 / dt_ms)))) - base
        pos, starts, ends, amps, peaks, k = _t_runs(v, lo_ms, hi_ms, t_peak_ms)
        a_main = float(amps[k])
        if a_main <= 0:
            return t_peak_ms, t_peak_ms, nan_

        def _end(j: int) -> float:
            return float(bt[min(n - 1, int(ends[j]))])
        is_phase = max(T_PHASE_FRAC * a_main, T_PHASE_NOISE_X * noise)
        # A run too small to be told from noise is no phase only when a
        # phase big enough to count would have been told from it: under more
        # noise it may be a phase the noise hides (ambiguous), and calling it
        # absent made the T monophasic on some medians, biphasic on others.
        keep = amps >= (max(T_PHASE_ABSENT_FRAC * a_main, T_PHASE_ABSENT_NOISE_X * noise)
                        if T_PHASE_FRAC * a_main >= T_PHASE_NOISE_X * noise
                        else T_PHASE_ABSENT_FRAC * a_main)

        def _side(step: int) -> str:
            j = k + step
            while 0 <= j < len(amps) and not keep[j]:
                j += step
            if not 0 <= j < len(amps) or pos[starts[j]] == pos[starts[k]]:
                return "none"
            gap = (starts[k] - ends[j]) if step < 0 else (starts[j] - ends[k])
            if (gap * dt_ms > T_PHASE_GAP_MS
                    or not lo_ms - T_PHASE_EDGE_MS <= bt[peaks[j]] <= hi_ms):
                return "none"
            if amps[j] >= is_phase:
                return "phase"
            # Ambiguous by its size (not only hidden by the noise).
            small_phase[0] = small_phase[0] or bool(amps[j] < T_PHASE_FRAC * a_main)
            return "ambiguous"

        def _next(step: int) -> int:
            j = k + step
            while not keep[j]:
                j += step
            return j

        before = _side(-1)
        if before == "phase":
            return float(bt[peaks[_next(-1)]]), t_peak_ms, _end(k)
        if before == "ambiguous":
            return nan_, nan_, nan_
        after = _side(+1)
        if after == "phase":
            j = _next(+1)
            return t_peak_ms, float(bt[peaks[j]]), _end(j)
        if after == "ambiguous":
            return nan_, nan_, nan_
        return t_peak_ms, t_peak_ms, _end(k)

    def _t_runs(v: "np.ndarray", lo_ms: float, hi_ms: float,
                t_peak_ms: float) -> tuple:
        """Sign runs of *v* (the 4 ms moving average against the baseline):
        (positive?, start, end [exclusive], amplitude, peak index) of each.
        A run whose peak is outside the T search span *lo_ms* -
        T_PHASE_EDGE_MS .. *hi_ms* is no T phase (amplitude 0), except the
        one holding the T peak *t_peak_ms*, whose amplitude and peak are the
        T peak's: a J/ST dip of the same sign with no zero crossing before a
        - then + T shares its run, and the second phase was compared with
        the dip."""
        pos = v > 0
        cut = np.flatnonzero(pos[1:] != pos[:-1]) + 1
        starts, ends = np.r_[0, cut], np.r_[cut, n]          # [start, end)
        peaks = np.array([a + int(np.argmax(np.abs(v[a:b])))
                          for a, b in zip(starts, ends)])
        amps = np.abs(v[peaks])
        amps[(bt[peaks] < lo_ms - T_PHASE_EDGE_MS) | (bt[peaks] > hi_ms)] = 0.0
        k = int(np.searchsorted(ends, _idx(t_peak_ms), side="right"))
        peaks[k] = _idx(t_peak_ms)
        amps[k] = abs(v[peaks[k]])
        return pos, starts, ends, amps, peaks, k

    def _t_plan_of(t_first: float, t_last: float, base: float) -> dict:
        """The phase plan (``t_plan_out``) of the T phases _t_phases found:
        peak time and sign of the first and the last phase, and how many
        phases (1 or 2); ``{"ambiguous": True}`` when the T end was, with
        ``"small_phase": True`` when a second phase is there but between
        T_PHASE_ABSENT_FRAC and T_PHASE_FRAC of the first."""
        if not (np.isfinite(t_first) and np.isfinite(t_last)):
            return ({"ambiguous": True, "small_phase": True} if small_phase[0]
                    else {"ambiguous": True})
        v = _moving_average(mb, max(1, int(round(4.0 / dt_ms)))) - base
        return {"phases": 1 if t_last == t_first else 2,
                "first_ms": float(t_first), "first_sign": 1 if v[_idx(t_first)] >= 0 else -1,
                "last_ms": float(t_last), "last_sign": 1 if v[_idx(t_last)] >= 0 else -1}

    def _t_from_plan(plan: dict, lo_ms: float, hi_ms: float,
                     base: float, noise: float) -> "tuple[float, float]":
        """(T peak, T end) in ms of a beat whose T phases follow *plan* (that
        of the median of the beats around it, ``t_plan``), found on this
        beat's 4 ms moving average by least squares (``_fit_lag``): first
        the lag of the plan's whole T (its phases together, within
        +-T_PLAN_SHIFT_MS), then that of each phase alone within
        T_PLAN_REFINE_MS of it. The T end is the plan's (its last-phase edge
        line meeting the baseline) moved by the lag of the last phase, or
        of the whole T when the last phase alone is not told from noise; the
        T peak is this beat's extremum inside the first phase so moved.
        Fitted on every sample of the T, the lag does not take the slope of
        a 3-8-sample edge from the noise (a line through the 30-70 % band
        spread the T end over 15 ms on a low T; an edge read at fixed
        levels lost most rows of a low phase once the noise reached ~1.7 x
        that of 248). A T that widens faster than the plan follows is read
        at ~3/4 of its line-end change, a shift in full. NaN for both when
        the plan is ambiguous or the whole T is not told from noise; NaN T
        end when this beat clearly lacks the last phase (T_PLAN_ABSENT_GAIN)
        or past *hi_ms*."""
        nan_ = float("nan")
        if plan.get("ambiguous") or not plan.get("phases"):
            return nan_, nan_
        v = _moving_average(mb, max(1, int(round(4.0 / dt_ms)))) - base
        g_t, g_v = (np.asarray(plan[k], dtype=float) for k in ("edge_t", "edge_v"))
        runs = [_plan_run(g_t, g_v, float(plan[k + "_ms"]), int(plan[k + "_sign"]))
                for k in ("first", "last")]
        if runs[0] is None or runs[1] is None:
            return nan_, nan_
        span = np.arange(-T_PLAN_SHIFT_MS, T_PLAN_SHIFT_MS + 1e-9, T_PLAN_LAG_STEP_MS)
        whole = _fit_lag(v, g_t[runs[0][0]:runs[1][1] + 1],
                         g_v[runs[0][0]:runs[1][1] + 1], noise, span)[0]
        if whole is None:
            return nan_, nan_
        # Each phase alone within T_PLAN_REFINE_MS of the whole T's lag: the
        # first one for the T peak, the last one for the T end.
        span_r = whole + span[np.abs(span) <= T_PLAN_REFINE_MS]
        ja, jb = runs[0]
        one = _fit_lag(v, g_t[ja:jb + 1], g_v[ja:jb + 1], noise, span_r)[0]
        lag_first = whole if one is None else one
        ja, jb = runs[1]
        seg = g_t[ja:jb + 1], g_v[ja:jb + 1]
        one, gain, sd = _fit_lag(v, *seg, noise, span_r)
        absent = False
        if one is None:
            # Not told from noise alone: the whole T's lag, unless this beat
            # clearly lacks the phase (its gain there under
            # T_PLAN_ABSENT_GAIN by 2 SD): no T end then.
            _l, gain, sd = _fit_lag(v, *seg, noise, np.array([whole]))
            absent = bool(gain + 2.0 * sd < T_PLAN_ABSENT_GAIN)
        lags = [lag_first, whole if one is None else one]
        f_sign = int(plan["first_sign"])
        i0 = _idx(float(g_t[runs[0][0]]) + lags[0])
        i1 = _idx(float(g_t[runs[0][1]]) + lags[0])
        t_pk = (float(bt[i0 + int(np.argmax(f_sign * v[i0:i1 + 1]))])
                if i1 >= i0 else float(plan["first_ms"]) + lags[0])
        t_end = float(plan["end_ms"]) + lags[1]
        return t_pk, (t_end if t_pk < t_end <= hi_ms and not absent else nan_)

    def _plan_run(g_t: "np.ndarray", g_v: "np.ndarray", t_ref: float,
                  sign: int) -> "Optional[tuple[int, int]]":
        """First and last index of the run of sign *sign* of the plan trace
        *g_v* holding *t_ref* (a phase peak); None when there is none."""
        sg = sign * g_v
        j = int(np.argmin(np.abs(g_t - t_ref)))
        if sg[j] <= 0:
            return None
        ja, jb = j, j
        while ja > 0 and sg[ja - 1] > 0:
            ja -= 1
        while jb < len(sg) - 1 and sg[jb + 1] > 0:
            jb += 1
        return (ja, jb) if jb - ja >= 2 else None

    def _fit_lag(v: "np.ndarray", tt: "np.ndarray", gg: "np.ndarray", noise: float,
                 lags: "np.ndarray") -> "tuple[Optional[float], float, float]":
        """(lag, gain, gain SD): the lag among *lags* (ms) at which the plan
        segment *gg* (times *tt*) moved fits this beat's moving average *v*
        best in least squares, at the plan's own amplitude (with a free gain
        the fit went to the flat baseline after the T, gain ~0; by
        correlation alone onto the R wave), and this beat's least-squares
        gain on the segment so moved, with its SD from *noise* (the SD of a
        4 ms moving average). Lag None when it is at the edge of a span of
        lags (the fit lies outside it) or the gain is under T_PLAN_SNR
        times its SD."""
        nan_ = float("nan")
        x = np.interp(tt[None, :] + lags[:, None], bt, v, left=np.nan, right=np.nan)
        res = np.sum((x - gg[None, :]) ** 2, axis=1)
        e = float(gg @ gg)
        if not np.isfinite(res).any() or e <= 0:
            return None, nan_, nan_
        k = int(np.nanargmin(res))
        # The moving average's noise is correlated over its w samples
        # (triangular autocorrelation).
        w = max(1, int(round(4.0 / dt_ms)))
        ac = np.correlate(gg, gg, "full")
        rho = np.clip(1.0 - np.abs(np.arange(-(len(gg) - 1), len(gg))) / w, 0.0, None)
        sd = noise * np.sqrt(max(float(ac @ rho), 0.0)) / e
        gain = float(x[k] @ gg) / e
        if (len(lags) > 1 and k in (0, len(lags) - 1)) or not gain >= T_PLAN_SNR * sd:
            return None, gain, sd
        return float(lags[k]), gain, sd

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
        qon_t, qoff_t = qrs_bounds(bt, mb, q_t, s_t, j_t, pr_lo, pr_hi,
                                   initial_wave=qrs_initial_wave,
                                   initial_out=qrs_initial_out,
                                   initial_plan=qrs_initial_plan)
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

    # Reference level for P onset and T end: the PR-segment baseline, the
    # one QRS onset and the J point are measured against.
    pr_base = (float(np.median(mb[_idx(pr_lo):_idx(pr_hi) + 1]))
               if np.isfinite(pr_lo) else None)

    # ── P onset ───────────────────────────────────────────────────────────
    # An interval end point like QRS onset and T end: not clamped to the
    # P_onset template window (which capped it ~9 ms before the P peak with
    # the defaults and cut long PRs short), NaN when it cannot be placed.
    po_t = (_p_onset(p_t, pr_base if pr_base is not None else pre_baseline,
                     snr_thresh)
            if p_t is not None else float("nan"))

    # ── Where the next beat's P wave starts ───────────────────────────────
    # At mouse heart rates above ~450 bpm the template's T-peak window (to
    # ~78 ms) reaches the next beat's P wave, and that P's first deflection
    # was taken as the T (T peak ~70 ms, QT ~85 ms at RR 120 ms). The next P
    # has this beat's P shape: it starts at the next R plus this beat's P
    # start -- the first sample, within NEXT_P_SPAN_MS before the P peak,
    # where the 2 ms average departs from the reference by a quarter of the P
    # amplitude (a biphasic P's leading trough included). The T PEAK is not
    # searched past it; the T end is left to the trailing-edge rules below
    # (a T that ends after the next P has started keeps its QT).
    t_peak_cap = float("inf")
    if next_r_ms is not None and np.isfinite(next_r_ms) and p_t is not None:
        dev = _moving_average(mb, max(1, int(round(2.0 / dt_ms)))) - pre_baseline
        i_pk = _idx(p_t)
        amp = abs(float(dev[i_pk]))
        i0 = _idx(p_t - NEXT_P_SPAN_MS)
        if amp > 0 and i_pk > i0:
            over = np.flatnonzero(np.abs(dev[i0:i_pk + 1]) >= 0.25 * amp)
            p_start = float(bt[i0 + int(over[0])]) if len(over) else float(p_t)
            t_peak_cap = float(next_r_ms) + p_start

    # ── Detect T wave (starts after J, or after S + gap) ─────────────────
    # Starting after J prevents the J hump from being selected as T.
    # A T-peak window cut short by the next P slides earlier by as much: at
    # RR ~85 ms the T peaks before the template window even starts (~26 ms).
    if t_peak_cap < t_hi:
        t_lo = max(5.0, t_lo - (t_hi - t_peak_cap))
        t_hi = t_peak_cap
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

    # ── T offset ──────────────────────────────────────────────────────────
    # Searched up to the later of the template's and the default T-end
    # window ends (within the beat): a refined template, narrowed to one
    # recording's spread, must not cut a longer QT short in another.
    to_end = min(bt1, max(to_hi, sum(WaveTemplate.DEFAULTS["T_offset"])))
    # Reference level: the PR-segment baseline (pr_base, above). The end of
    # the beat window is still inside the T wave at high mouse heart rates,
    # so a baseline taken there moved the T end.
    t_last, t_stop_edge = t_t, None
    # No T phase in the next beat's P wave either (t_peak_cap): its leading
    # trough is not the T's second phase.
    phase_hi = t_peak_cap
    biphasic = t_phase_noise is not None
    t_noise_ = max(0.0, float(t_phase_noise)) if biphasic else 0.0
    t_base = pr_base
    if biphasic and pr_base is not None:
        # On a rolling median, the flatness-weighted PR level: the T end of
        # a low phase moves ~1.7 ms per 0.03 of baseline (pr_baseline_level).
        lvl = pr_baseline_level(bt, mb, q_t)
        t_base = lvl if np.isfinite(lvl) else pr_base
    if biphasic and t_plan is not None and t_base is not None:
        # The phase plan of the beats around: the same phase ends the T on
        # neighbouring rows, whatever the noise of the row (_t_from_plan).
        t_t, to_t = _t_from_plan(t_plan, t_lo_eff, min(to_end, t_stop, phase_hi),
                                 t_base, t_noise_)
    else:
        if biphasic and np.isfinite(t_t) and t_base is not None:
            # Biphasic T: its peak is that of the first phase and its end that
            # of the last one, so that two phases of similar size do not take
            # turns at being the T (T end jumping by one phase between rows).
            t_first, t_last, t_stop_edge = _t_phases(
                t_t, t_lo_eff, min(to_end, t_stop, phase_hi), t_base, t_noise_)
            if t_plan_out is not None:
                t_plan_out.update(_t_plan_of(t_first, t_last, t_base))
            t_t = t_first if np.isfinite(t_first) else t_t
        to_t   = (_slope_offset(t_last, to_end, t_base, fit_smoothed=biphasic,
                                stop_ms=t_stop_edge,
                                line_tol=T_PLAN_LINE_TOL_MS if biphasic else None)
                  if np.isfinite(t_last) and to_end > t_last else float("nan"))
        if t_plan_out is not None and t_plan_out.get("phases"):
            if np.isfinite(to_t):
                # The T phases on this beat, which each row's are fitted to
                # (_t_from_plan).
                v_e = _moving_average(mb, max(1, int(round(4.0 / dt_ms)))) - t_base
                e0 = _idx(min(t_t, t_last) - 2.0 * T_PLAN_TOL_MS)
                e1 = _idx(to_t + 2.0 * T_PLAN_TOL_MS) + 1
                t_plan_out.update(end_ms=float(to_t), edge_t=bt[e0:e1].copy(),
                                  edge_v=v_e[e0:e1].copy())
            else:
                t_plan_out.clear()

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

