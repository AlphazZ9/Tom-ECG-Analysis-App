# -*- coding: utf-8 -*-
"""
state.py
--------
Typed state containers for ECGApp, grouped by responsibility.

These dataclasses hold every piece of non-widget instance state that used to
live as scattered ``self._xxx`` attributes directly on ``ECGApp``. Widgets
(CTk entries, buttons, labels, canvases, ...) stay directly on ``ECGApp`` --
they are the GUI, not application state.

ECGApp keeps backward-compatible ``@property`` shims (see app.py) for every
renamed attribute so existing method bodies (and the small set of sibling
modules that reach into an ECGApp instance -- dialogs.py, wave_editor.py,
models.py) keep working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

from models import MouseECG
from wave_template import WaveTemplate


@dataclass
class SignalState:
    """The loaded recording and its filtered/derived arrays."""
    filepath: Optional[str] = None
    raw: Optional[np.ndarray] = None            # raw float64 (after time-crop)
    raw_norm: Optional[np.ndarray] = None       # zero-mean/unit-var raw (display)
    filtered: Optional[np.ndarray] = None       # filtered + normalised
    time: Optional[np.ndarray] = None
    fs: int = MouseECG.FS_DEFAULT
    peak_distance_ms: float = MouseECG.PEAK_DISTANCE_MS
    # Raw-only load state -- file just opened, no filtering/detection run yet.
    # True right after _load_raw_only(); cleared once Preview Detection runs.
    raw_only_loaded: bool = False
    no_filter_mode: bool = True                 # default: detect on raw signal
    inverted: bool = False                      # True if polarity was auto-flipped


@dataclass
class DetectionState:
    """R-peak detection results and manual edit/undo state."""
    EDIT_UNDO_LIMIT: ClassVar[int] = 50

    rpeaks_ok: Optional[np.ndarray] = None      # accepted peaks
    rpeaks_rej: Optional[np.ndarray] = None     # rejected candidates
    all_candidates: Optional[np.ndarray] = None  # all candidates (post-polarity fix)
    all_prominences: Optional[np.ndarray] = None  # corresponding prominences
    thresh_amp: float = 0.0
    sig_quality: Optional[int] = None

    # Manual peak exclusion
    manual_excluded: set[int] = field(default_factory=set)   # sample indices excluded by user
    rpeaks_manual_excl: Optional[np.ndarray] = None           # view-ready array of excl. peaks
    manual_added: set[int] = field(default_factory=set)      # sample indices added by user
    rpeaks_manual_added: Optional[np.ndarray] = None          # view-ready array of added peaks
    edit_mode: bool = False                     # click-to-exclude active
    edit_free_placement: bool = False           # bypass proximity constraint
    # Undo/Redo stacks -- each entry: (frozenset excluded, frozenset added)
    edit_undo: list[tuple[frozenset, frozenset]] = field(default_factory=list)
    edit_redo: list[tuple[frozenset, frozenset]] = field(default_factory=list)

    # Hover preview state (edit mode -- shows snapped peak position on mouse move)
    hover_samp: Optional[int] = None            # snapped sample under cursor
    hover_samp_near: bool = False                # True = replaces nearby peak


@dataclass
class AnalysisState:
    """HRV/frequency/nonlinear/interval/arrhythmia analysis results."""
    results: Optional[dict] = None
    epoch_df: Optional[pd.DataFrame] = None
    rolling_hrv_df: Optional[pd.DataFrame] = None
    arrhythmia_events: list = field(default_factory=list)
    arrhythmia_tsv: str = ""
    arr_selected_idx: int = -1                  # index into arrhythmia_events
    arr_nav_pos: float = 0.0                    # current t_start in arrhythmia ECG view
    arr_win: float = 3.0                        # seconds shown in arrhythmia ECG view
    arr_edit_mode: bool = False                 # edit mode specific to arrhythmia tab
    last_seg_a: Optional[dict] = None           # last Compare A result
    last_seg_b: Optional[dict] = None           # last Compare B result
    artifact_report: Optional[dict] = None
    # Experimental context -- key into EXPERIMENTAL_CONTEXTS
    exp_context: str = "telemetry_awake"

    # Analysis window (independent from signal crop) -- restricts HRV analysis
    # to a sub-range of the already-detected signal. 0.0 = "use full signal".
    t_start: float = 0.0
    t_end: float = 0.0

    # Time annotations: list of dicts {t_start, t_end, label, color}
    annotations: list[dict] = field(default_factory=list)
    wave_template: Optional[WaveTemplate] = None


@dataclass
class UIState:
    """Transient view/render state: navigation position, caches, mpl event ids."""
    nav_pos: float = 0.0
    dark_mode: bool = False
    show_raw: bool = True                       # raw/filtered toggle
    # Filter preview (before/after overlay) -- computed on-demand for the
    # currently visible detail window only, never touches filtered signal or
    # any detection state.
    filter_preview_on: bool = False
    thr_debounce_id: "str | None" = None        # slider debounce handle
    hover_motion_cid: Optional[int] = None       # mpl event connection id
    hover_after_id: "str | None" = None          # motion debounce handle
    beat_nav_cid: Optional[int] = None           # mpl event ID for beat navigator
    rr_click_cid: Optional[int] = None           # mpl event connection id for RR click-to-navigate
    ov_ylim: Optional[tuple] = None              # y-axis zoom cache for overview
    hrv_current_view: str = "RR / HR"
    ctx_keys: list = field(default_factory=list)
    operation_start_time: Optional[float] = None
    ui_update_batch: list[tuple] = field(default_factory=list)
    figure_cache: dict = field(default_factory=dict)
    # TSV clipboard store: keyed by widget id, value = TSV string
    tsv_store: "dict[int, str]" = field(default_factory=dict)

    # Display cache -- downsampled arrays, invalidated on file load
    ds_time: Optional[np.ndarray] = None
    ds_sig: Optional[np.ndarray] = None          # envelope mins (filtered)
    ds_sig_max: Optional[np.ndarray] = None      # envelope maxs (filtered)
    ds_sig_mid: Optional[np.ndarray] = None      # envelope midline (filtered)
    ds_raw_sig: Optional[np.ndarray] = None      # envelope mins (raw)
    ds_raw_sig_max: Optional[np.ndarray] = None  # envelope maxs (raw)
    ds_raw_sig_mid: Optional[np.ndarray] = None  # envelope midline (raw)


@dataclass
class SessionState:
    """Session persistence (.ecgsession) and recent-file bookkeeping."""
    recent_files: list[str] = field(default_factory=list)
    dirty: bool = False
    recording_notes: str = ""                    # freetext notes for this recording
