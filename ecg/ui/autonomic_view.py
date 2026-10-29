# -*- coding: utf-8 -*-
"""
autonomic_view.py
-----------------
Pure helpers of the HRV > Autonomic view: time-resolved cardiac sympathetic
and parasympathetic indices (CSI / CPI, core.autonomic -- Candia-Rivera et
al., J Physiol 2025) with optional SPWVD LF / HF power (core.spwvd).

Parameter parsing and session normalisation, the withheld-window spans, the
recording-level summary (text + one-row table for Excel) and the stacked
figure. No Tk widget access: AnalysisController reads the widgets and calls
these, so all of it is unit-testable on a bare matplotlib Figure.

Theme colours are read from ``ecg.ui.theme`` at call time (not copied at
import), so a theme switch needs no re-sync of this module.
"""
from __future__ import annotations

import logging
import math
import textwrap
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ecg.core.analysis import rr_bounds_label, rr_exclusion_text
from ecg.core.autonomic import (
    COV_METHODS, DEFAULT_METHOD, DEFAULT_MIN_PAIRS, DEFAULT_WINDOW_S, KP_RAT_SHR,
    KS_RAT_SHR, AutonomicResult,
)
from ecg.ui import theme as _theme
from ecg.ui.plots import style_axes

log = logging.getLogger("ecg")


# ════════════════════════════════════════════════════════════
#  PARAMETERS
# ════════════════════════════════════════════════════════════

AUTONOMIC_PARAM_KEYS = ("window_s", "method", "ks", "kp", "lf_hf", "min_pairs")

# Accepted sliding-window lengths. A window shorter than ~1 s holds fewer
# than the 10 valid pairs a window needs by default even at 600 bpm (every
# window would be withheld); past 60 s it is no longer "time-resolved" for
# a mouse, and the per-beat (beats x pairs) blocks grow quadratically.
WINDOW_MIN_S = 1.0
WINDOW_MAX_S = 60.0

# Accepted "Min pairs" floor (valid adjacent RR pairs a window needs). The
# floor is this app's addition: compute_CSI_CVI_rat.m computes every window
# (2 pairs are the minimum a covariance needs, so 2 reproduces it).
MIN_PAIRS_MIN = 2
MIN_PAIRS_MAX = 200

MIN_PAIRS_NOTE = (
    "The minimum of {n} valid adjacent RR pairs per window is this app's "
    "addition, not in the reference code (compute_CSI_CVI_rat.m computes "
    "every window): Min pairs = 2 reproduces it, a higher floor withholds "
    "windows too short to estimate a 2 × 2 covariance reliably.")

# Persistent disclosure under the controls (the coefficients are the
# paper's rat values; core.autonomic.calibration_note has the long form).
CALIBRATION_WARNING = (
    "ks = 3, kp = 12 are the rat (SHR) calibration of Candia-Rivera et al. "
    "2025 — not validated in mice. Calibrate on your own cohort "
    "(Event-locked > Calibrate).")

# The SPWVD band powers are unnormalised (core.spwvd, "Units and scale").
SPWVD_NOTE = (
    "SPWVD LF / HF power is unnormalised (a.u., scaled by the kernel and the "
    "number of frequency bins): compare it within a recording (pre vs post an "
    "event, LF vs HF trend), not with Welch band powers or across settings. "
    "The kernel constants (v0 = 0.03, tau0 = 0.06, lambda = 0.3) are the ones "
    "hard-coded in compute_PWVD.m -- the toolbox's only set (its file "
    "defaults are Frr = 4 Hz and human bands); the paper applied them at "
    "20 Hz to rats. They are defined in normalised units, so their extent in "
    "seconds and Hz depends on the 20 Hz resampling and the 4096 bins; not "
    "validated in mice. The distribution is not positive-definite, so brief "
    "negative excursions are a property of the method. RR gaps longer than "
    "2 s split the series (each stretch computed on its own, blank between "
    "them); stretches shorter than 10 s are blank.")

# Resampled samples drawn per line before the display is decimated (an
# hour at 20 Hz is 72 000 samples; every resize replays the draw).
MAX_PLOT_POINTS = 60_000


def default_autonomic_params() -> dict:
    """The view's defaults: the paper's 3 s window, robust shrinkage, the rat
    (SHR) coefficients, LF / HF off (slow), the core's 10-pair floor."""
    return {"window_s": float(DEFAULT_WINDOW_S), "method": DEFAULT_METHOD,
            "ks": float(KS_RAT_SHR), "kp": float(KP_RAT_SHR), "lf_hf": False,
            "min_pairs": int(DEFAULT_MIN_PAIRS)}


def method_key(value: Any) -> Optional[str]:
    """The COV_METHODS key for *value* -- a key or its UI label (surrounding
    blanks ignored) -- or None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v in COV_METHODS:
        return v
    for key, label in COV_METHODS.items():
        if v == label:
            return key
    return None


def format_param(value: float) -> str:
    """A parameter as typed back into its entry ('3', '2.5', '12')."""
    return f"{float(value):g}"


def _to_float(value: Any) -> float:
    """*value* as a finite float; a decimal comma is accepted ('2,5'). bool
    is refused (a checkbox value is not a number)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"not a number: {value!r}")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"not a finite number: {value!r}")
    return v


_TRUE_TEXT = frozenset({"1", "true", "yes", "on"})
_FALSE_TEXT = frozenset({"0", "false", "no", "off", ""})


def _to_bool(value: Any) -> bool:
    """A checkbox / JSON flag: bool, 0 / 1, or 'true' / 'false' style text."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_TEXT:
            return True
        if v in _FALSE_TEXT:
            return False
    raise ValueError(f"not a boolean flag: {value!r}")


def _window_value(value: Any) -> float:
    v = _to_float(value)
    if not (WINDOW_MIN_S <= v <= WINDOW_MAX_S):
        raise ValueError(f"window {v:g} s outside {WINDOW_MIN_S:g}–{WINDOW_MAX_S:g} s")
    return v


def _gain_value(value: Any) -> float:
    v = _to_float(value)
    if v < 0:
        raise ValueError(f"negative gain {v:g}")
    return v


def _method_value(value: Any) -> str:
    key = method_key(value)
    if key is None:
        raise ValueError(f"unknown method {value!r}")
    return key


def _min_pairs_value(value: Any) -> int:
    v = _to_float(value)
    if not (v.is_integer() and MIN_PAIRS_MIN <= v <= MIN_PAIRS_MAX):
        raise ValueError(f"min pairs {v:g} not an integer in "
                         f"{MIN_PAIRS_MIN}–{MIN_PAIRS_MAX}")
    return int(v)


_VALIDATORS = {
    "window_s":  _window_value,
    "method":    _method_value,
    "ks":        _gain_value,
    "kp":        _gain_value,
    "lf_hf":     _to_bool,
    "min_pairs": _min_pairs_value,
}


def normalise_autonomic_params(raw: Any, base: "Optional[Mapping]" = None) -> dict:
    """Tolerant reading of stored parameters (a session file, app state):
    every valid key of *raw* is taken; a key that is missing or invalid --
    and *raw* itself when it is not a mapping (a session saved before the
    Autonomic view existed) -- keeps *base*'s value (itself normalised;
    default: ``default_autonomic_params()``). Never raises."""
    out = (default_autonomic_params() if base is None
           else normalise_autonomic_params(base))
    if not isinstance(raw, Mapping):
        return out
    for key in AUTONOMIC_PARAM_KEYS:
        if key not in raw:
            continue
        try:
            out[key] = _VALIDATORS[key](raw[key])
        except (TypeError, ValueError, OverflowError) as exc:
            log.debug("autonomic params: ignoring %s=%r (%s)", key, raw[key], exc)
    return out


def parse_autonomic_inputs(window: Any, method: Any, ks: Any, kp: Any,
                           lf_hf: Any = False, min_pairs: Any = DEFAULT_MIN_PAIRS) -> dict:
    """Strict reading of the view's entries for a Compute click. Raises
    ValueError with a message for the status bar; the LF / HF flag is read
    tolerantly (an unreadable checkbox value means off)."""
    try:
        n_pairs = _min_pairs_value(min_pairs)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"Min pairs must be a whole number from {MIN_PAIRS_MIN} to {MIN_PAIRS_MAX} "
            f"(got {min_pairs!r}; 2 = every window, as the reference code).") from None
    try:
        window_s = _window_value(window)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"Window must be a number of seconds from {WINDOW_MIN_S:g} to "
            f"{WINDOW_MAX_S:g} (got {window!r}).") from None
    key = method_key(method)
    if key is None:
        raise ValueError(f"Unknown covariance method {method!r}.")
    gains = {}
    for name, value in (("ks", ks), ("kp", kp)):
        try:
            gains[name] = _gain_value(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite number ≥ 0 (got {value!r}).") from None
    try:
        flag = _to_bool(lf_hf)
    except (TypeError, ValueError):
        flag = False
    return {"window_s": window_s, "method": key, "ks": gains["ks"],
            "kp": gains["kp"], "lf_hf": flag, "min_pairs": n_pairs}


def params_differ(entries: Mapping, result_params: Mapping,
                  has_lfhf: bool) -> "list[str]":
    """The view parameters of *entries* (``autonomic_widget_params()``)
    that differ from those *result_params* (``AutonomicResult.params``)
    was computed with -- e.g. after ks / kp were written by the
    calibration dialog, or typed, without ⟳ Compute. *has_lfhf*: LF / HF
    was requested for the displayed result (computed or failed). Returns
    'ks 17 (shown: 3)' style texts, empty when they agree."""
    out = []
    for key, name in (("window_s", "window"), ("ks", "ks"), ("kp", "kp"),
                      ("min_pairs", "min pairs")):
        try:
            a, b = float(entries[key]), float(result_params[key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            out.append(f"{name} {a:g} (shown: {b:g})")
    m_e, m_r = entries.get("method"), result_params.get("method")
    if m_e is not None and m_r is not None and m_e != m_r:
        out.append(f"method {m_e} (shown: {m_r})")
    if "lf_hf" in entries and bool(entries["lf_hf"]) != bool(has_lfhf):
        out.append("LF/HF ticked (shown: without)" if entries["lf_hf"]
                   else "LF/HF unticked (shown: with)")
    return out


def is_rat_calibration(ks: float, kp: float) -> bool:
    """True for the paper's rat (SHR) coefficients ks = 3, kp = 12."""
    return float(ks) == KS_RAT_SHR and float(kp) == KP_RAT_SHR


# ════════════════════════════════════════════════════════════
#  RESULT HELPERS
# ════════════════════════════════════════════════════════════

def spwvd_ok(spwvd: "Optional[Mapping]") -> bool:
    """True when *spwvd* is a computed spwvd_lf_hf() result (not None, not
    an {"error": ...} placeholder) with at least one sample."""
    return (isinstance(spwvd, Mapping) and "error" not in spwvd
            and len(np.asarray(spwvd.get("t", ()))) > 0)


def spwvd_error(spwvd: "Optional[Mapping]") -> Optional[str]:
    """Why the requested LF / HF could not be computed, or None."""
    if isinstance(spwvd, Mapping) and spwvd.get("error"):
        return str(spwvd["error"])
    return None


def withheld_spans(res: AutonomicResult) -> "list[tuple[float, float]]":
    """Time spans (s) of the withheld windows -- fewer than ``min_pairs``
    valid adjacent RR pairs, or an undefined ``robust`` covariance: SD1,
    SD2 and CCD are NaN there, so CSI / CPI are blank. Each run of
    consecutive withheld windows spans from the last computed window before
    it to the first computed one after it (the stretch the resampled series
    leaves blank); a run at either end stops at its own first / last
    window. Sorted, non-overlapping."""
    t = np.asarray(res.t_beat, dtype=np.float64)
    if len(t) == 0:
        return []
    bad = ~(np.isfinite(np.asarray(res.sd1_raw, dtype=np.float64))
            & np.isfinite(np.asarray(res.sd2_raw, dtype=np.float64))
            & np.isfinite(np.asarray(res.d_raw, dtype=np.float64)))
    if not bad.any():
        return []
    fs_out = float(res.params.get("fs_out", 20.0) or 20.0)
    half = 0.5 / fs_out
    edges = np.diff(np.r_[0, bad.astype(np.int8), 0])
    spans: "list[tuple[float, float]]" = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lo = float(t[s - 1] if s > 0 else t[s])
        hi = float(t[e] if e < len(t) else t[e - 1])
        if hi - lo < 2 * half:                     # a lone edge window
            lo, hi = lo - half, hi + half
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return spans


def _nanmean(a: Any) -> float:
    v = np.asarray(a, dtype=np.float64)
    f = np.isfinite(v)
    return float(v[f].mean()) if f.any() else float("nan")


def _fmt(value: Any, decimals: int, unit: str = "") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.{decimals}f}" + (f" {unit}" if unit else "")


def _pct_text(value: Any) -> str:
    """'1.5', '0.04' (a nonzero share never prints as 0.0), '0.0'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.2f}" if 0.0 < v < 0.1 else f"{v:.1f}"


def _band_text(band: Any) -> str:
    try:
        lo, hi = band
        return f"{float(lo):g}–{float(hi):g}"
    except (TypeError, ValueError):
        return ""


def window_bounds(value: Any) -> "Optional[tuple[float, float]]":
    """An analysis window (t_start, t_end) as floats (0 = open bound, as in
    AnalysisState), or None when *value* is not such a pair."""
    try:
        t0, t1 = value
        t0, t1 = float(t0), float(t1)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(t0) and math.isfinite(t1)):
        return None
    return t0, t1


def analysis_window_text(params: Mapping) -> str:
    """'full signal', '100 – 200 s' or '100 s – end' for the analysis window
    a result was computed on (``params['analysis_window']``); '?' when the
    result does not record it."""
    win = window_bounds(params.get("analysis_window"))
    if win is None:
        return "?"
    t0, t1 = win
    if t0 <= 0 and t1 <= 0:
        return "full signal"
    return f"{t0:g} – {t1:g} s" if t1 > 0 else f"{t0:g} s – end"


def same_window(a: Any, b: Any) -> bool:
    """True when two analysis windows are the same (to 1 µs); a missing
    one never matches."""
    wa, wb = window_bounds(a), window_bounds(b)
    if wa is None or wb is None:
        return False
    return abs(wa[0] - wb[0]) <= 1e-6 and abs(wa[1] - wb[1]) <= 1e-6


def summary_frame(res: AutonomicResult, spwvd: "Optional[Mapping]" = None) -> pd.DataFrame:
    """The recording-level values as ONE row (columns = metrics), so pasting
    several recordings into Excel stacks one row per animal."""
    p = res.params
    row: "dict[str, Any]" = {
        "Method":             p.get("method", ""),
        "Window_s":           p.get("window_s", np.nan),
        "Min_pairs":          p.get("min_pairs", np.nan),
        "ks":                 p.get("ks", np.nan),
        "kp":                 p.get("kp", np.nan),
        "SD01_ms":            res.sd01,
        "SD02_ms":            res.sd02,
        "D0_ms":              res.d0,
        "N_beats_evaluated":  p.get("n_beats", 0),
        "N_windows_withheld": p.get("n_windows_withheld", 0),
        "N_RR":               p.get("n_intervals", 0),
        "N_RR_valid":         p.get("n_valid", 0),
        "N_RR_excluded":      p.get("n_excluded", 0),
        "Excluded_pct":       p.get("excluded_pct", 0.0),
        "RR_min_ms":          p.get("rr_min_ms", np.nan),
        "RR_max_ms":          p.get("rr_max_ms", np.nan),
        "HR_mean_bpm":        _nanmean(res.hr_bpm),
        "Analysis_window":    analysis_window_text(p),
    }
    if spwvd_ok(spwvd):
        sp = spwvd.get("params", {})                     # type: ignore[union-attr]
        row.update({
            "LF_mean_au": _nanmean(spwvd["lf"]),         # type: ignore[index]
            "HF_mean_au": _nanmean(spwvd["hf"]),         # type: ignore[index]
            "LF_band_Hz": _band_text(sp.get("lf_band")),
            "HF_band_Hz": _band_text(sp.get("hf_band")),
        })
    return pd.DataFrame([row])


def summary_text(res: AutonomicResult, spwvd: "Optional[Mapping]" = None,
                 width: int = 100) -> str:
    """The textbox under the plot: recording-level values, then the
    disclosures (method, calibration, withheld windows, SPWVD)."""
    p = res.params
    ks, kp = float(p.get("ks", np.nan)), float(p.get("kp", np.nan))
    lines = ["  AUTONOMIC INDICES  —  time-resolved CSI / CPI "
             "(Candia-Rivera et al., J Physiol 2025)", ""]

    def row(name: str, value: str) -> None:
        lines.append(f"  {name:<20} {value}")

    row("Method", str(p.get("method_label", p.get("method", ""))))
    row("Window", f"{float(p.get('window_s', np.nan)):g} s  (≥ {p.get('min_pairs', '?')} "
                  f"valid adjacent RR pairs, else withheld)")
    row("ks / kp", f"{ks:g} / {kp:g}  — "
                   + ("rat (SHR) values, NOT validated in mice"
                      if is_rat_calibration(ks, kp) else "user-set"))
    row("SD01", f"{_fmt(res.sd01, 2, 'ms')}   (whole-recording SD1)")
    row("SD02", f"{_fmt(res.sd02, 2, 'ms')}   (whole-recording SD2)")
    row("D0 (CCD0)", f"{_fmt(res.d0, 1, 'ms')}   (whole-recording Poincaré centre distance)")
    n_beats = int(p.get("n_beats", 0) or 0)
    n_wh = int(p.get("n_windows_withheld", 0) or 0)
    row("Beats evaluated", f"{n_beats}" + (f"   ({n_wh} windows withheld)" if n_wh else ""))
    n_excl = int(p.get("n_excluded", 0) or 0)
    row("RR intervals",
        f"{int(p.get('n_intervals', 0) or 0)} total · {int(p.get('n_valid', 0) or 0)} valid · "
        f"{n_excl} excluded ({_pct_text(p.get('excluded_pct', 0.0) or 0.0)} %)")
    if n_excl:
        row("", f"{int(p.get('n_too_short', 0) or 0)} too short · "
                f"{int(p.get('n_too_long', 0) or 0)} too long · "
                f"{int(p.get('n_spanning_removed', 0) or 0)} spanning removed beats "
                f"(bounds {rr_bounds_label(p)})")
    row("Mean HR", _fmt(_nanmean(res.hr_bpm), 1, "bpm"))
    if len(res.t):
        row("Span", f"{float(res.t[0]):.1f} – {float(res.t[-1]):.1f} s   "
                    f"(analysis window: {analysis_window_text(p)})")
    err = spwvd_error(spwvd)
    if err:
        row("LF / HF (SPWVD)", f"not computed — {err}")
    elif spwvd_ok(spwvd):
        sp = spwvd.get("params", {})                     # type: ignore[union-attr]
        row("LF / HF (SPWVD)",
            f"mean {_nanmean(spwvd['lf']):.4g} / {_nanmean(spwvd['hf']):.4g} a.u.   "  # type: ignore[index]
            f"(LF {_band_text(sp.get('lf_band'))} Hz, HF {_band_text(sp.get('hf_band'))} Hz; "
            f"{float(sp.get('fs_resample', 20.0)):g} Hz, {sp.get('fbins', '?')} bins)")
        n_gaps = int(sp.get("n_gaps_masked", 0) or 0)
        if n_gaps:
            n_short = int(sp.get("n_segments_too_short", 0) or 0)
            row("", f"split at {n_gaps} RR gap(s) > "
                    f"{float(sp.get('nan_gap_s', 2.0)):g} s, blank between the stretches"
                    + (f" and over {n_short} stretch(es) shorter than "
                       f"{float(sp.get('min_segment_s', 10.0)):g} s" if n_short else "")
                    + f" ({int(sp.get('n_nan_samples', 0) or 0)} samples)")

    notes = [str(p.get("method_note", "")), str(p.get("calibration_note", ""))]
    if p.get("min_pairs") is not None:
        notes.append(MIN_PAIRS_NOTE.format(n=p.get("min_pairs")))
    if p.get("withheld_reason"):
        notes.append(f"Nothing computed: {p['withheld_reason']}.")
    elif p.get("withheld_note"):
        notes.append(str(p["withheld_note"]) + " (shaded grey on the plot).")
    notes.append(
        "Gap-aware: RR intervals outside the bounds or spanning a removed beat "
        "are excluded; Poincaré pairs never straddle them and the series are "
        "never bridged across them.")
    if spwvd_ok(spwvd) or err:
        notes.append(SPWVD_NOTE)
    lines += ["", "  Notes"]
    for note in notes:
        if note:
            lines += textwrap.wrap(note, width=width, initial_indent="  * ",
                                   subsequent_indent="    ")
    return "\n".join(lines)


def autonomic_status(res: AutonomicResult,
                     spwvd: "Optional[Mapping]" = None) -> "tuple[str, str]":
    """(text, level) of the view's status label; level is "ok" (nothing to
    disclose), "warn" (withheld windows, excluded RR, LF / HF issue) or
    "withheld" (no window computed)."""
    p = res.params
    if p.get("withheld_reason"):
        return f"  Withheld — {p['withheld_reason']}", "withheld"
    notes = []
    n_wh = int(p.get("n_windows_withheld", 0) or 0)
    if n_wh:
        notes.append(f"{n_wh} window(s) withheld")
    excl = rr_exclusion_text(p)
    if excl:
        notes.append(excl)
    n_gaps = 0
    err = spwvd_error(spwvd)
    if err:
        notes.append(f"LF/HF not computed: {err}")
    elif spwvd_ok(spwvd):
        n_gaps = int(spwvd.get("params", {}).get("n_gaps_masked", 0) or 0)   # type: ignore[union-attr]
        notes.append("LF/HF" + (f" blank over {n_gaps} gap(s)" if n_gaps else " ✓"))
    text = (f"  {int(p.get('n_beats', 0) or 0)} beats · {p.get('method', '')} · "
            f"{float(p.get('window_s', np.nan)):g} s · ks {float(p.get('ks', np.nan)):g}"
            f" / kp {float(p.get('kp', np.nan)):g}")
    warn = bool(n_wh or excl or err or n_gaps)
    if notes:
        text += "  ·  " + "  ·  ".join(notes)
    return text + ("" if warn else "  ✓"), "warn" if warn else "ok"


# ════════════════════════════════════════════════════════════
#  FIGURE
# ════════════════════════════════════════════════════════════

def _stride(n: int, max_points: int) -> int:
    return max(1, int(math.ceil(n / max(1, int(max_points)))))


def _ann_spans(annotations: Iterable[Mapping], x0: float, x1: float
               ) -> "list[tuple[float, float, str, str]]":
    """(t_start, t_end, label, color) of the annotations overlapping
    [x0, x1], clipped to it; malformed entries are skipped."""
    out = []
    for ann in annotations or ():
        try:
            ts = float(ann["t_start"])
            te = float(ann.get("t_end", ts))
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(ts) and math.isfinite(te)):
            continue
        ts, te = min(ts, te), max(ts, te)
        if te < x0 or ts > x1:
            continue
        out.append((ts, te, str(ann.get("label", "") or ""),
                    str(ann.get("color") or _theme.ORANGE_DARK)))
    return out


def draw_autonomic(fig: Any, res: AutonomicResult,
                   spwvd: "Optional[Mapping]" = None,
                   annotations: "Iterable[Mapping]" = (),
                   max_points: int = MAX_PLOT_POINTS) -> None:
    """Populate *fig* with the stacked, shared-x Autonomic panels: HR; CCD
    (re-centred D); SD1 and SD2 (re-centred); CSI and CPI; LF and HF (when
    *spwvd* is a computed result). Withheld windows are shaded grey,
    annotations (dicts with t_start, t_end, label, color) are drawn on
    every panel and labelled on the top one; the x range is the result's,
    so annotations outside it do not stretch the axis."""
    from matplotlib.patches import Patch

    p = res.params
    show_lfhf = spwvd_ok(spwvd)
    n_panels = 5 if show_lfhf else 4
    axes = fig.subplots(n_panels, 1, sharex=True)
    axes = list(np.atleast_1d(axes))
    try:
        engine = fig.get_layout_engine()
        if engine is not None and hasattr(engine, "set"):
            engine.set(hspace=0.06)
    except Exception as exc:                       # older matplotlib
        log.debug("draw_autonomic: layout engine pads not set: %s", exc)
    for ax in axes:
        style_axes(ax)

    ks, kp = float(p.get("ks", np.nan)), float(p.get("kp", np.nan))
    window_s = float(p.get("window_s", np.nan))
    head = (f"CSI / CPI  —  {p.get('method_label', p.get('method', ''))}  ·  "
            f"window {window_s:g} s  ·  ks = {ks:g}, kp = {kp:g}")
    if is_rat_calibration(ks, kp):
        head += "  (rat SHR calibration — not validated in mice)"

    t = np.asarray(res.t, dtype=np.float64)
    if len(t) == 0:
        reason = p.get("withheld_reason") or "no window could be computed"
        axes[0].set_title(head, loc="left", fontsize=8)
        axes[0].text(0.5, 0.5, f"No CSI / CPI — {reason}", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=8, wrap=True,
                     color=_theme.PLOT["muted"])
        axes[-1].set_xlabel("Time (s)")
        return

    k = _stride(len(t), max_points)
    ts = t[::k]
    muted = _theme.PLOT["muted"]

    def _line(ax, y, color, label=None, lw=1.1):
        ax.plot(ts, np.asarray(y, dtype=np.float64)[::k], color=color, lw=lw,
                label=label, zorder=3)

    # ── panels ──
    _line(axes[0], res.hr_bpm, _theme.ORANGE_DARK)
    axes[0].set_ylabel("HR (bpm)", fontsize=8)
    axes[0].set_title(head, loc="left", fontsize=8)

    _line(axes[1], res.d, _theme.NAVY)
    axes[1].set_ylabel("CCD (ms)", fontsize=8)
    axes[1].set_title(f"CCD — Poincaré centre distance D, re-centred on D0 = "
                      f"{_fmt(res.d0, 1, 'ms')}", loc="left", fontsize=8)

    _line(axes[2], res.sd1, _theme.CORAL, "SD1")
    _line(axes[2], res.sd2, _theme.BLUE_DARK, "SD2")
    axes[2].set_ylabel("SD1 / SD2 (ms)", fontsize=8)
    axes[2].set_title(f"SD1 / SD2 — re-centred on SD01 = {_fmt(res.sd01, 2, 'ms')}, "
                      f"SD02 = {_fmt(res.sd02, 2, 'ms')}", loc="left", fontsize=8)

    _line(axes[3], res.csi, _theme.RED_MID, "CSI")
    _line(axes[3], res.cpi, _theme.GREEN_DARK, "CPI")
    axes[3].set_ylabel("CSI / CPI (a.u.)", fontsize=8)
    axes[3].set_title(f"CSI = D̃ + {ks:g}·SD2  ·  CPI = D + {kp:g}·SD1   "
                      f"(D̃ = D mirrored about its mean)", loc="left", fontsize=8)

    x0, x1 = float(t[0]), float(t[-1])
    if show_lfhf:
        st = np.asarray(spwvd["t"], dtype=np.float64)          # type: ignore[index]
        ks2 = _stride(len(st), max_points)
        sp = spwvd.get("params", {})                           # type: ignore[union-attr]
        axes[4].plot(st[::ks2], np.asarray(spwvd["lf"], dtype=np.float64)[::ks2],  # type: ignore[index]
                     color=_theme.PURPLE, lw=1.0, label="LF", zorder=3)
        axes[4].plot(st[::ks2], np.asarray(spwvd["hf"], dtype=np.float64)[::ks2],  # type: ignore[index]
                     color=_theme.TEAL, lw=1.0, label="HF", zorder=3)
        axes[4].set_ylabel("LF / HF (a.u.)", fontsize=8)
        axes[4].set_title(f"SPWVD power — LF {_band_text(sp.get('lf_band'))} Hz, "
                          f"HF {_band_text(sp.get('hf_band'))} Hz (unnormalised)",
                          loc="left", fontsize=8)
        if len(st):
            x0, x1 = min(x0, float(st[0])), max(x1, float(st[-1]))
        k = max(k, ks2)

    # ── withheld windows ──
    spans = withheld_spans(res)
    for ax in axes:
        for lo, hi in spans:
            ax.axvspan(lo, hi, color=muted, alpha=0.14, linewidth=0, zorder=0)
    legend_extra = ([Patch(color=muted, alpha=0.3,
                           label=f"withheld (< {p.get('min_pairs', '?')} valid pairs)")]
                    if spans else [])

    # ── legends ──
    for i, ax in enumerate(axes):
        handles, _labels = ax.get_legend_handles_labels()
        if i == 0:
            handles = handles + legend_extra
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=len(handles),
                      frameon=True, framealpha=0.8, facecolor=_theme.PLOT["bg"],
                      edgecolor="none")

    # ── annotations (spans on every panel, labels on the top one) ──
    anns = _ann_spans(annotations, x0, x1)
    for ts_a, te_a, _lbl, col in anns:
        for ax in axes:
            ax.axvspan(ts_a, te_a, color=col, alpha=0.10, linewidth=0, zorder=0)
            ax.axvline(ts_a, color=col, lw=1.1, alpha=0.8, zorder=5)
            if te_a > ts_a:
                ax.axvline(te_a, color=col, lw=0.7, ls="--", alpha=0.5, zorder=5)
    for ax in axes:
        ax.set_xlim(x0, x1)
    if anns:
        _ylo, yhi = axes[0].get_ylim()
        for ts_a, te_a, lbl, col in anns:
            if not lbl:
                continue
            xm = 0.5 * (max(ts_a, x0) + min(te_a, x1))
            axes[0].text(xm, yhi, lbl, ha="center", va="top", fontsize=7, color=col,
                         fontweight="bold", zorder=8, clip_on=True,
                         bbox=dict(boxstyle="round,pad=0.15", fc=_theme.PLOT["bg"],
                                   ec=col, alpha=0.8, lw=0.6))

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Time (s)" + (f"   (display decimated ×{k})" if k > 1 else ""))
