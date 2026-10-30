# -*- coding: utf-8 -*-
"""
freshness.py
------------
Per-module freshness: up to date / out of date / not computed.

Each job records a basis (peaks, removed beats, analysis window, RR bounds)
captured at its start on the main thread (``capture``), stored with its result
when it lands (``stamp``). The state is derived on demand by comparing that
basis with the current one, on top of the checks the app already has
(``autonomic_stale_reasons``, ``arrhythmia_is_current``), which are reused.

The peak-edit paths drop results rather than leaving them stale, so a module
with a stamp and no result was computed once and needs re-running: out of date.
A new file clears the stamps (``reset``); a restored session stamps what it
brought back (``stamp_restored``). Main thread only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np

from ecg.core.analysis import hrv_freq_is_legacy, intervals_use_legacy_definitions
from ecg.core.metric_table import MODULES

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

UP_TO_DATE = "up to date"
OUT_OF_DATE = "out of date"
NOT_COMPUTED = "not computed"

_BASIS_TEXT = {
    "peaks":   "peaks changed",
    "removed": "removed beats changed",
    "window":  "analysis window changed",
    "rr":      "RR bounds changed",
}
_DROPPED = "result dropped since it was computed: re-run"


@dataclass(frozen=True)
class ModuleFreshness:
    state: str
    reasons: "tuple[str, ...]" = ()


def combine_states(states: "Iterable[str]") -> str:
    """The state of a tab from its parts: any stale computed part = out of
    date; nothing computed = not computed; otherwise up to date."""
    parts = list(states)
    if OUT_OF_DATE in parts:
        return OUT_OF_DATE
    return UP_TO_DATE if UP_TO_DATE in parts else NOT_COMPUTED


def _array_signature(a: Any) -> "Optional[str]":
    if a is None:
        return None
    arr = np.asarray(a).astype(np.int64)
    return f"{arr.size}:{hashlib.sha1(arr.tobytes()).hexdigest()}"


def _window_key(window: Any) -> "tuple[float, float]":
    t0, t1 = window
    return round(float(t0 or 0.0), 6), round(float(t1 or 0.0), 6)


def _rr_key(bounds: Any) -> "tuple[float, float]":
    return round(float(bounds[0]), 6), round(float(bounds[1]), 6)


def recorded_rr_bounds(results: "Optional[dict]") -> "Optional[tuple[float, float]]":
    """The bounds core results were computed with, None when not recorded."""
    hr = (results or {}).get("hr") or {}
    try:
        return float(hr["rr_min_ms"]), float(hr["rr_max_ms"])
    except (KeyError, TypeError, ValueError):
        return None


def _frame_computed(df: Any) -> bool:
    """A result frame exists: non-empty, or empty but explained by its attrs
    (a withheld spectrum). analyse_core's blank placeholders do not count."""
    if df is None:
        return False
    return not df.empty or bool(getattr(df, "attrs", None))


def _intervals_computed(ivl: Any) -> bool:
    # analyse_core's placeholder is a non-empty frame with RR_ms alone.
    return ivl is not None and not ivl.empty and "QT_ms" in ivl.columns


class FreshnessService:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    # ── basis ────────────────────────────────────────────────────────────

    def capture(self, window: Any = None, rr_bounds: Any = None, peaks: Any = None) -> dict:
        """The current basis; a job passes what it was actually given when
        that differs from the app's current state (the window core results
        were computed on, the RR bounds frequency HRV reuses)."""
        a, det = self.app.analysis, self.app.detection
        ctrl = self.app.analysis_ctrl
        pk = peaks if peaks is not None else det.rpeaks_ok
        breaks = ctrl.hrv_breaks(pk) if pk is not None else None
        return {
            "peaks":   _array_signature(pk),
            "removed": _array_signature(breaks) if breaks is not None else None,
            "window":  _window_key(window if window is not None else (a.t_start, a.t_end)),
            "rr":      _rr_key(rr_bounds if rr_bounds is not None
                               else ctrl.hrv_rr_bounds(quiet=True)),
        }

    def stamp(self, module: str, basis: "Optional[dict]" = None) -> None:
        self.app.analysis.basis[module] = basis if basis is not None else self.capture()

    def reset(self) -> None:
        """New file: nothing was computed on it."""
        self.app.analysis.basis.clear()

    def stamp_restored(self) -> None:
        """After a session restore: every result it brought back counts as
        computed on the restored peaks (they were saved together), unless a
        check says otherwise -- core results on their recorded window and RR
        bounds, which can differ from the restored entries."""
        a = self.app.analysis
        a.basis.clear()
        r = a.results
        if r:
            basis = self.capture(window=a.analysed_window, rr_bounds=recorded_rr_bounds(r))
            a.basis["core"] = basis
            if _frame_computed(r.get("hrv_freq")):
                a.basis["frequency"] = basis
            if _frame_computed(r.get("hrv_nonlin")):
                a.basis["nonlinear"] = basis
            if _intervals_computed(r.get("intervals")):
                a.basis["intervals"] = basis
        if a.epoch_df is not None and not a.epoch_df.empty:
            a.basis["epochs"] = self.capture()
        if a.rolling_hrv_df is not None and not a.rolling_hrv_df.empty:
            a.basis["rolling"] = self.capture()

    def _differences(self, module: str, ignore: "tuple[str, ...]" = ()) -> "list[str]":
        rec = self.app.analysis.basis.get(module)
        if rec is None:
            return []
        now = self.capture()
        return [_BASIS_TEXT[k] for k in _BASIS_TEXT
                if k not in ignore and rec.get(k) != now[k]]

    def _from_basis(self, module: str, computed: bool, extra: "Iterable[str]" = (),
                    ignore: "tuple[str, ...]" = ()) -> ModuleFreshness:
        if not computed:
            if self.app.analysis.basis.get(module) is not None:
                return ModuleFreshness(OUT_OF_DATE, (_DROPPED,))
            return ModuleFreshness(NOT_COMPUTED)
        reasons = self._differences(module, ignore) + list(extra)
        return (ModuleFreshness(OUT_OF_DATE, tuple(reasons)) if reasons
                else ModuleFreshness(UP_TO_DATE))

    # ── per module ───────────────────────────────────────────────────────

    def status(self, module: str) -> ModuleFreshness:
        return getattr(self, "_status_" + module)()

    def states(self) -> "dict[str, str]":
        return {m: self.status(m).state for m in MODULES}

    def _status_core(self) -> ModuleFreshness:
        return self._from_basis("core", self.app.analysis.results is not None)

    def _status_frequency(self) -> ModuleFreshness:
        r = self.app.analysis.results or {}
        extra = ["older version's spectrum: re-run Frequency HRV"] if hrv_freq_is_legacy(r) else []
        return self._from_basis("frequency", _frame_computed(r.get("hrv_freq")), extra)

    def _status_nonlinear(self) -> ModuleFreshness:
        nl = (self.app.analysis.results or {}).get("hrv_nonlin")
        extra = (["older version's result (first beats only): re-run Non-linear HRV"]
                 if nl is not None and getattr(nl, "attrs", {}).get("truncated") else [])
        return self._from_basis("nonlinear", _frame_computed(nl), extra)

    def _status_intervals(self) -> ModuleFreshness:
        ivl = (self.app.analysis.results or {}).get("intervals")
        computed = _intervals_computed(ivl)
        extra = []
        if computed:
            if intervals_use_legacy_definitions(ivl):
                extra.append("legacy definitions: re-run Intervals")
            # Same comparison as the Beat Template overlay's note: the windows
            # the intervals used against the recording's current template.
            used = ivl.attrs.get("wave_template")
            cur = self.app.analysis.wave_template
            if isinstance(used, dict) and cur is not None and cur.to_dict() != used:
                extra.append("wave template changed")
        return self._from_basis("intervals", computed, extra)

    def _status_epochs(self) -> ModuleFreshness:
        df = self.app.analysis.epoch_df
        return self._from_basis("epochs", df is not None and not df.empty)

    def _status_rolling(self) -> ModuleFreshness:
        df = self.app.analysis.rolling_hrv_df
        return self._from_basis("rolling", df is not None and not df.empty)

    def _status_autonomic(self) -> ModuleFreshness:
        a = self.app.analysis
        if a.autonomic_result is None:
            note = getattr(a, "autonomic_stale_note", None)
            if note:
                return ModuleFreshness(OUT_OF_DATE, (str(note).strip(),))
            return self._from_basis("autonomic", False)
        # Those existing reasons already cover the analysis window.
        return self._from_basis("autonomic", True,
                                self.app.analysis_ctrl.autonomic_stale_reasons(),
                                ignore=("window",))

    def _status_event_locked(self) -> ModuleFreshness:
        a = self.app.analysis
        if a.event_locked is None:
            note = getattr(a, "event_locked_stale", None)
            if note:
                return ModuleFreshness(OUT_OF_DATE, (str(note).strip(),))
            return self._from_basis("event_locked", False)
        extra = ([] if self._status_autonomic().state == UP_TO_DATE
                 else ["Autonomic indices out of date"])
        return self._from_basis("event_locked", True, extra)

    def _status_events(self) -> ModuleFreshness:
        a = self.app.analysis
        if not a.arrhythmia_events and not a.premature_summary:
            return ModuleFreshness(NOT_COMPUTED)
        if self.app.analysis_ctrl.arrhythmia_is_current():
            return ModuleFreshness(UP_TO_DATE)
        return ModuleFreshness(OUT_OF_DATE, ("peaks, artifact correction or analysis window changed",))
