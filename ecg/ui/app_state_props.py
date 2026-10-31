# -*- coding: utf-8 -*-
"""
ecg.ui.app_state_props
----------------------
ECGApp state: _init_state and the legacy self._xxx property shims.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from ecg.core.wave_template import WaveTemplate
from ecg.ui.state import (
    SignalState, DetectionState, AnalysisState, UIState, SessionState,
)
from ecg.ui.navigation_controller import NavigationController
from ecg.ui.export_controller import ExportController
from ecg.ui.plot_controller import PlotController
from ecg.ui.session_controller import SessionController
from ecg.ui.detection_controller import DetectionController
from ecg.ui.signal_controller import SignalController
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.event_locked_controller import EventLockedController
from ecg.ui.freshness import FreshnessService
from ecg.ui.plots import CanvasSlot

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class StateProps:
    """ECGApp state: _init_state and the legacy self._xxx property shims."""

    def _init_state(self: ECGApp) -> None:
        """Initialise all instance variables that hold application state.

        Non-widget state lives in typed dataclasses from state.py (`self.signal`,
        `self.detection`, `self.analysis`, `self.ui`, `self.session`). The
        "Legacy state shims" property block right after this method keeps every
        existing `self._xxx` access (inside this class and in dialogs.py /
        wave_editor.py / models.py) working unchanged. New code should read/write
        the dataclasses directly.
        """
        self.signal = SignalState()
        self.detection = DetectionState()
        self.analysis = AnalysisState()
        self.ui = UIState()
        self.session = SessionState()

        self.nav_ctrl = NavigationController(self)
        self.export_ctrl = ExportController(self)
        self.plot_ctrl = PlotController(self)
        self.session_ctrl = SessionController(self)
        self.detection_ctrl = DetectionController(self)
        self.signal_ctrl = SignalController(self)
        self.analysis_ctrl = AnalysisController(self)
        self.event_ctrl = EventLockedController(self)
        self.freshness = FreshnessService(self)

        # Widget registries (populated in _build)
        self._slots:       dict[str, CanvasSlot]   = {}
        self._kpi:         dict[str, ctk.CTkLabel] = {}
        # Compact top-bar mirrors of a few _kpi values (hr_mean/n_beats/dur) --
        # a SEPARATE dict since these are distinct widgets living in the top
        # bar, updated alongside (not instead of) the stat-panel tiles by the
        # same update_kpis() call.
        self._topbar_vals: dict[str, ctk.CTkLabel] = {}

        # ── Forward declarations for widgets created in _build ────────────────
        # Declared here so that methods called before the UI is fully constructed
        # (e.g. _collect_session_state called during startup restore) get a
        # predictable None rather than raising AttributeError, and so that
        # hasattr(self, ...) guards can be replaced with "is not None" checks.
        #
        # Top toolbar (built in _build_toolbar)
        self.ent_project_name:  "Optional[ctk.CTkEntry]"   = None
        self.lbl_topbar_project:"Optional[ctk.CTkLabel]"   = None
        self.lbl_topbar_file:   "Optional[ctk.CTkLabel]"   = None
        self.quality_gauge:     "Optional[ctk.CTkFrame]"   = None
        # Sidebar / session controls
        self.btn_save_session:  "Optional[ctk.CTkButton]"  = None
        self.lbl_session_info:  "Optional[ctk.CTkLabel]"   = None
        self.lbl_template_info: "Optional[ctk.CTkLabel]"   = None
        # Per-module on-demand analysis buttons + status labels
        self.btn_run_freq:      "Optional[ctk.CTkButton]"  = None
        self.btn_run_nonlin:    "Optional[ctk.CTkButton]"  = None
        self.btn_run_ivl:       "Optional[ctk.CTkButton]"  = None
        self.btn_run_arrhythmia:"Optional[ctk.CTkButton]"  = None
        self.lbl_freq_status:   "Optional[ctk.CTkLabel]"   = None
        self.lbl_nonlin_status: "Optional[ctk.CTkLabel]"   = None
        self.lbl_ivl_status:    "Optional[ctk.CTkLabel]"   = None
        # Filter / signal controls (widgets used in _collect_session_state)
        self.ent_channel:       "Optional[ctk.CTkEntry]"   = None
        self.ent_t_start:       "Optional[ctk.CTkEntry]"   = None
        self.ent_t_end:         "Optional[ctk.CTkEntry]"   = None
        self.ent_analysis_t0:   "Optional[ctk.CTkEntry]"   = None
        self.ent_analysis_t1:   "Optional[ctk.CTkEntry]"   = None
        self.lbl_analysis_window: "Optional[ctk.CTkLabel]" = None
        self.ent_lp:            "Optional[ctk.CTkEntry]"   = None
        self.ent_hp:            "Optional[ctk.CTkEntry]"   = None
        self.ent_minrr:         "Optional[ctk.CTkEntry]"   = None
        self.ent_maxrr:         "Optional[ctk.CTkEntry]"   = None
        self.sw_notch:         "Optional[ctk.CTkSwitch]"  = None
        self.sw_invert_signal:  "Optional[ctk.CTkSwitch]"  = None
        self.sw_filter_preview: "Optional[ctk.CTkSwitch]"  = None
        self.cb_clean:          "Optional[ctk.CTkComboBox]" = None
        self.sl_thr:            "Optional[ctk.CTkSlider]"  = None
        self.ent_thr:           "Optional[ctk.CTkEntry]"   = None
        self.sw_permissive:     "Optional[ctk.CTkSwitch]"  = None
        self.btn_annotations:   "Optional[ctk.CTkButton]"  = None
        self.lbl_ann_count:     "Optional[ctk.CTkLabel]"   = None
        self.btn_toggle_rrhr:   "Optional[ctk.CTkButton]"  = None
        self.btn_toggle_spikes: "Optional[ctk.CTkButton]"  = None
        self.btn_review_spikes: "Optional[ctk.CTkButton]"  = None
        self.btn_toggle_left_panel:  "Optional[ctk.CTkButton]" = None
        self.btn_toggle_right_panel: "Optional[ctk.CTkButton]" = None
        self.lbl_panel_ann_count:    "Optional[ctk.CTkLabel]" = None
        self.lbl_panel_pacing_count: "Optional[ctk.CTkLabel]" = None
        self.btn_copy_rr:       "Optional[ctk.CTkButton]"  = None
        self.btn_copy_ivl:      "Optional[ctk.CTkButton]"  = None
        self.btn_copy_epochs:   "Optional[ctk.CTkButton]"  = None
        self.cb_det_method:     "Optional[ctk.CTkComboBox]"  = None
        self.ent_sg_target_fs:  "Optional[ctk.CTkEntry]"     = None
        self.ent_sg_window_ms:  "Optional[ctk.CTkEntry]"     = None
        self._sg_frame:         "Optional[ctk.CTkFrame]"     = None
        self.cb_qtc_formula:    "Optional[ctk.CTkComboBox]"  = None
        self.cb_freq_band:      "Optional[ctk.CTkComboBox]"  = None   # HRV band preset
        self.lbl_ml_status:     "Optional[ctk.CTkLabel]"     = None
        self.lbl_ml_inline_status: "Optional[ctk.CTkLabel]"  = None   # ML status mirrored next to the Method combobox
        self.sw_verified_training: "Optional[ctk.CTkSwitch]" = None
        self.btn_train_ml:      "Optional[ctk.CTkButton]"    = None
        self.btn_save_for_training: "Optional[ctk.CTkButton]" = None
        # Sidebar / detection tab widgets (forward-declared)
        self.lbl_npeaks:           "Optional[ctk.CTkLabel]"              = None
        self.btn_review_art:       "Optional[ctk.CTkButton]"             = None
        self.btn_check_agreement: "Optional[ctk.CTkButton]"              = None
        self.lbl_agreement_status:"Optional[ctk.CTkLabel]"               = None
        self.lbl_file:             "Optional[ctk.CTkLabel]"              = None
        self.lbl_arrhythmia_status:"Optional[ctk.CTkLabel]"              = None
        self.lbl_roll_status:      "Optional[ctk.CTkLabel]"              = None
        # Time course > Autonomic (CSI / CPI) -- see _build_time_view_autonomic
        self.ent_auto_win:         "Optional[ctk.CTkEntry]"              = None
        self.ent_auto_minpairs:    "Optional[ctk.CTkEntry]"              = None
        self.cb_auto_method:       "Optional[ctk.CTkComboBox]"           = None
        self.ent_auto_ks:          "Optional[ctk.CTkEntry]"              = None
        self.ent_auto_kp:          "Optional[ctk.CTkEntry]"              = None
        self.chk_auto_lfhf:        "Optional[ctk.CTkCheckBox]"           = None
        self.btn_auto_compute:     "Optional[ctk.CTkButton]"             = None
        self.lbl_auto_status:      "Optional[ctk.CTkLabel]"              = None
        self.txt_autonomic:        "Optional[ctk.CTkTextbox]"            = None
        # Time course > Event-locked -- see _build_time_view_event_locked
        self.cb_evt_label:         "Optional[ctk.CTkComboBox]"           = None
        self.ent_evt_pre:          "Optional[ctk.CTkEntry]"              = None
        self.ent_evt_post:         "Optional[ctk.CTkEntry]"              = None
        self.chk_evt_z:            "Optional[ctk.CTkCheckBox]"           = None
        self.btn_evt_compute:      "Optional[ctk.CTkButton]"             = None
        self.lbl_evt_status:       "Optional[ctk.CTkLabel]"              = None
        self.txt_event_locked:     "Optional[ctk.CTkTextbox]"            = None
        self.lbl_arr_event_title:  "Optional[ctk.CTkLabel]"              = None
        self.lbl_arr_premature:    "Optional[ctk.CTkLabel]"              = None
        self.lbl_arr_baseline:     "Optional[ctk.CTkLabel]"              = None
        self.lbl_arr_stale:        "Optional[ctk.CTkLabel]"              = None
        self.lbl_arr_events_head:  "Optional[ctk.CTkLabel]"              = None
        self._arr_card_widgets:    "list[ctk.CTkBaseClass]"              = []
        self._filter_advanced_group: "Optional[ctk.CTkFrame]"            = None
        self.lbl_filter_summary:  "Optional[ctk.CTkLabel]"              = None
        self._interp_scroll:       "Optional[ctk.CTkScrollableFrame]"    = None
        # Navigation bar widgets (forward-declared so _reset_for_new_file can update them)
        self.ent_nav_pos:       "Optional[ctk.CTkEntry]"    = None
        self.lbl_sig_duration:  "Optional[ctk.CTkLabel]"    = None
        # Morphology & Intervals tab widgets
        self.ent_entropy_cap:   "Optional[ctk.CTkEntry]"    = None  # non-linear entropy cap (RR)
        # Interpretation removed
        self._interp_cards:      "dict[str, tuple[ctk.CTkLabel, ctk.CTkLabel, str]]" = {}
        self._interp_ref_labels: "dict[str, ctk.CTkLabel]" = {}
        # Time course tab internal state
        self._time_frames:       "dict[str, ctk.CTkFrame]" = {}
        self._time_seg:          "Optional[ctk.CTkSegmentedButton]" = None
        self._time_banner:       "Optional[ctk.CTkFrame]"           = None
        self._time_banner_text:  "Optional[ctk.CTkLabel]"           = None
        self._epoch_metrics:     "dict[str, ctk.CTkCheckBox]"       = {}
        # Signal & Detection tab: quality panel values
        self._quality_vals:      "dict[str, ctk.CTkLabel]" = {}

    # ─── Legacy state shims ───────────────────────────────────────────────
    # Backward-compatible properties for the old flat self._xxx attributes,
    # now held in state.py dataclasses. Every existing method body in this
    # class, plus the sibling modules that reach into an ECGApp instance
    # (dialogs.py: _rebuild_ui/_annotations/_time/_session_dirty/_draw_detail/
    # _update_ann_count; wave_editor.py: _wave_template; models.py:
    # _peak_distance_ms/_safe_float), keep working unchanged through these.
    # New/extracted code should read and write self.signal/.detection/
    # .analysis/.ui/.session directly instead of these shims.

    # -- SignalState --
    @property
    def _filepath(self: ECGApp) -> "Optional[str]":
        return self.signal.filepath
    @_filepath.setter
    def _filepath(self: ECGApp, value: "Optional[str]") -> None:
        self.signal.filepath = value

    @property
    def _signal_raw(self: ECGApp) -> "Optional[np.ndarray]":
        return self.signal.raw
    @_signal_raw.setter
    def _signal_raw(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.signal.raw = value

    @property
    def _signal_raw_norm(self: ECGApp) -> "Optional[np.ndarray]":
        return self.signal.raw_norm
    @_signal_raw_norm.setter
    def _signal_raw_norm(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.signal.raw_norm = value

    @property
    def _signal_flt(self: ECGApp) -> "Optional[np.ndarray]":
        return self.signal.filtered
    @_signal_flt.setter
    def _signal_flt(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.signal.filtered = value

    @property
    def _time(self: ECGApp) -> "Optional[np.ndarray]":
        return self.signal.time
    @_time.setter
    def _time(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.signal.time = value

    @property
    def _fs(self: ECGApp) -> int:
        return self.signal.fs
    @_fs.setter
    def _fs(self: ECGApp, value: int) -> None:
        self.signal.fs = value

    @property
    def _peak_distance_ms(self: ECGApp) -> float:
        return self.signal.peak_distance_ms
    @_peak_distance_ms.setter
    def _peak_distance_ms(self: ECGApp, value: float) -> None:
        self.signal.peak_distance_ms = value

    @property
    def _raw_only_loaded(self: ECGApp) -> bool:
        return self.signal.raw_only_loaded
    @_raw_only_loaded.setter
    def _raw_only_loaded(self: ECGApp, value: bool) -> None:
        self.signal.raw_only_loaded = value

    @property
    def _no_filter_mode(self: ECGApp) -> bool:
        return self.signal.no_filter_mode
    @_no_filter_mode.setter
    def _no_filter_mode(self: ECGApp, value: bool) -> None:
        self.signal.no_filter_mode = value

    @property
    def _signal_inverted(self: ECGApp) -> bool:
        return self.signal.inverted
    @_signal_inverted.setter
    def _signal_inverted(self: ECGApp, value: bool) -> None:
        self.signal.inverted = value

    # -- DetectionState --
    @property
    def _rpeaks_ok(self: ECGApp) -> "Optional[np.ndarray]":
        return self.detection.rpeaks_ok
    @_rpeaks_ok.setter
    def _rpeaks_ok(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.detection.rpeaks_ok = value

    @property
    def _rpeaks_rej(self: ECGApp) -> "Optional[np.ndarray]":
        return self.detection.rpeaks_rej
    @_rpeaks_rej.setter
    def _rpeaks_rej(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.detection.rpeaks_rej = value

    @property
    def _all_cands(self: ECGApp) -> "Optional[np.ndarray]":
        return self.detection.all_candidates
    @_all_cands.setter
    def _all_cands(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.detection.all_candidates = value

    @property
    def _all_proms(self: ECGApp) -> "Optional[np.ndarray]":
        return self.detection.all_prominences
    @_all_proms.setter
    def _all_proms(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.detection.all_prominences = value

    @property
    def _thresh_amp(self: ECGApp) -> float:
        return self.detection.thresh_amp
    @_thresh_amp.setter
    def _thresh_amp(self: ECGApp, value: float) -> None:
        self.detection.thresh_amp = value

    @property
    def _sig_quality(self: ECGApp) -> "Optional[int]":
        return self.detection.sig_quality
    @_sig_quality.setter
    def _sig_quality(self: ECGApp, value: "Optional[int]") -> None:
        self.detection.sig_quality = value

    @property
    def _manual_excluded(self: ECGApp) -> "set[int]":
        return self.detection.manual_excluded
    @_manual_excluded.setter
    def _manual_excluded(self: ECGApp, value: "set[int]") -> None:
        self.detection.manual_excluded = value

    @property
    def _rpeaks_manual_excl(self: ECGApp) -> "Optional[np.ndarray]":
        return self.detection.rpeaks_manual_excl
    @_rpeaks_manual_excl.setter
    def _rpeaks_manual_excl(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.detection.rpeaks_manual_excl = value

    @property
    def _manual_added(self: ECGApp) -> "set[int]":
        return self.detection.manual_added
    @_manual_added.setter
    def _manual_added(self: ECGApp, value: "set[int]") -> None:
        self.detection.manual_added = value

    @property
    def _rpeaks_manual_added(self: ECGApp) -> "Optional[np.ndarray]":
        return self.detection.rpeaks_manual_added
    @_rpeaks_manual_added.setter
    def _rpeaks_manual_added(self: ECGApp, value: "Optional[np.ndarray]") -> None:
        self.detection.rpeaks_manual_added = value

    @property
    def _edit_mode(self: ECGApp) -> bool:
        return self.detection.edit_mode
    @_edit_mode.setter
    def _edit_mode(self: ECGApp, value: bool) -> None:
        self.detection.edit_mode = value

    @property
    def _edit_free_placement(self: ECGApp) -> bool:
        return self.detection.edit_free_placement
    @_edit_free_placement.setter
    def _edit_free_placement(self: ECGApp, value: bool) -> None:
        self.detection.edit_free_placement = value

    @property
    def _edit_undo(self: ECGApp) -> "list[tuple[frozenset, frozenset]]":
        return self.detection.edit_undo
    @_edit_undo.setter
    def _edit_undo(self: ECGApp, value: "list[tuple[frozenset, frozenset]]") -> None:
        self.detection.edit_undo = value

    @property
    def _edit_redo(self: ECGApp) -> "list[tuple[frozenset, frozenset]]":
        return self.detection.edit_redo
    @_edit_redo.setter
    def _edit_redo(self: ECGApp, value: "list[tuple[frozenset, frozenset]]") -> None:
        self.detection.edit_redo = value

    @property
    def _EDIT_UNDO_LIMIT(self: ECGApp) -> int:
        return DetectionState.EDIT_UNDO_LIMIT

    @property
    def _hover_samp(self: ECGApp) -> "Optional[int]":
        return self.detection.hover_samp
    @_hover_samp.setter
    def _hover_samp(self: ECGApp, value: "Optional[int]") -> None:
        self.detection.hover_samp = value

    @property
    def _hover_samp_near(self: ECGApp) -> bool:
        return self.detection.hover_samp_near
    @_hover_samp_near.setter
    def _hover_samp_near(self: ECGApp, value: bool) -> None:
        self.detection.hover_samp_near = value

    # -- AnalysisState --
    @property
    def _results(self: ECGApp) -> "Optional[dict]":
        return self.analysis.results
    @_results.setter
    def _results(self: ECGApp, value: "Optional[dict]") -> None:
        self.analysis.results = value

    @property
    def _epoch_df(self: ECGApp) -> "Optional[pd.DataFrame]":
        return self.analysis.epoch_df
    @_epoch_df.setter
    def _epoch_df(self: ECGApp, value: "Optional[pd.DataFrame]") -> None:
        self.analysis.epoch_df = value

    @property
    def _rolling_hrv_df(self: ECGApp) -> "Optional[pd.DataFrame]":
        return self.analysis.rolling_hrv_df
    @_rolling_hrv_df.setter
    def _rolling_hrv_df(self: ECGApp, value: "Optional[pd.DataFrame]") -> None:
        self.analysis.rolling_hrv_df = value

    @property
    def _arrhythmia_events(self: ECGApp) -> "list":
        return self.analysis.arrhythmia_events
    @_arrhythmia_events.setter
    def _arrhythmia_events(self: ECGApp, value: "list") -> None:
        self.analysis.arrhythmia_events = value

    @property
    def _arrhythmia_tsv(self: ECGApp) -> str:
        return self.analysis.arrhythmia_tsv
    @_arrhythmia_tsv.setter
    def _arrhythmia_tsv(self: ECGApp, value: str) -> None:
        self.analysis.arrhythmia_tsv = value

    @property
    def _arr_selected_idx(self: ECGApp) -> int:
        return self.analysis.arr_selected_idx
    @_arr_selected_idx.setter
    def _arr_selected_idx(self: ECGApp, value: int) -> None:
        self.analysis.arr_selected_idx = value

    @property
    def _arr_nav_pos(self: ECGApp) -> float:
        return self.analysis.arr_nav_pos
    @_arr_nav_pos.setter
    def _arr_nav_pos(self: ECGApp, value: float) -> None:
        self.analysis.arr_nav_pos = value

    @property
    def _arr_win(self: ECGApp) -> float:
        return self.analysis.arr_win
    @_arr_win.setter
    def _arr_win(self: ECGApp, value: float) -> None:
        self.analysis.arr_win = value

    @property
    def _arr_edit_mode(self: ECGApp) -> bool:
        return self.analysis.arr_edit_mode
    @_arr_edit_mode.setter
    def _arr_edit_mode(self: ECGApp, value: bool) -> None:
        self.analysis.arr_edit_mode = value

    @property
    def _last_seg_a(self: ECGApp) -> "Optional[dict]":
        return self.analysis.last_seg_a
    @_last_seg_a.setter
    def _last_seg_a(self: ECGApp, value: "Optional[dict]") -> None:
        self.analysis.last_seg_a = value

    @property
    def _last_seg_b(self: ECGApp) -> "Optional[dict]":
        return self.analysis.last_seg_b
    @_last_seg_b.setter
    def _last_seg_b(self: ECGApp, value: "Optional[dict]") -> None:
        self.analysis.last_seg_b = value

    @property
    def _artifact_report(self: ECGApp) -> "Optional[dict]":
        return self.analysis.artifact_report
    @_artifact_report.setter
    def _artifact_report(self: ECGApp, value: "Optional[dict]") -> None:
        self.analysis.artifact_report = value

    @property
    def _artifact_candidates(self: ECGApp) -> "list[dict]":
        return self.analysis.artifact_candidates
    @_artifact_candidates.setter
    def _artifact_candidates(self: ECGApp, value: "list[dict]") -> None:
        self.analysis.artifact_candidates = value

    @property
    def _analysis_t_start(self: ECGApp) -> float:
        return self.analysis.t_start
    @_analysis_t_start.setter
    def _analysis_t_start(self: ECGApp, value: float) -> None:
        self.analysis.t_start = value

    @property
    def _analysis_t_end(self: ECGApp) -> float:
        return self.analysis.t_end
    @_analysis_t_end.setter
    def _analysis_t_end(self: ECGApp, value: float) -> None:
        self.analysis.t_end = value

    @property
    def _annotations(self: ECGApp) -> "list[dict]":
        return self.analysis.annotations
    @_annotations.setter
    def _annotations(self: ECGApp, value: "list[dict]") -> None:
        self.analysis.annotations = value

    @property
    def _pacing_periods(self: ECGApp) -> "list[dict]":
        return self.analysis.pacing_periods
    @_pacing_periods.setter
    def _pacing_periods(self: ECGApp, value: "list[dict]") -> None:
        self.analysis.pacing_periods = value

    @property
    def _wave_template(self: ECGApp) -> "Optional[WaveTemplate]":
        return self.analysis.wave_template
    @_wave_template.setter
    def _wave_template(self: ECGApp, value: "Optional[WaveTemplate]") -> None:
        self.analysis.wave_template = value

    # -- UIState --
    @property
    def _nav_pos(self: ECGApp) -> float:
        return self.ui.nav_pos
    @_nav_pos.setter
    def _nav_pos(self: ECGApp, value: float) -> None:
        self.ui.nav_pos = value

    @property
    def _show_raw(self: ECGApp) -> bool:
        return self.ui.show_raw
    @_show_raw.setter
    def _show_raw(self: ECGApp, value: bool) -> None:
        self.ui.show_raw = value

    @property
    def _filter_preview_on(self: ECGApp) -> bool:
        return self.ui.filter_preview_on
    @_filter_preview_on.setter
    def _filter_preview_on(self: ECGApp, value: bool) -> None:
        self.ui.filter_preview_on = value

    @property
    def _thr_debounce_id(self: ECGApp) -> "str | None":
        return self.ui.thr_debounce_id
    @_thr_debounce_id.setter
    def _thr_debounce_id(self: ECGApp, value: "str | None") -> None:
        self.ui.thr_debounce_id = value

    @property
    def _resize_debounce_id(self: ECGApp) -> "str | None":
        return self.ui.resize_debounce_id
    @_resize_debounce_id.setter
    def _resize_debounce_id(self: ECGApp, value: "str | None") -> None:
        self.ui.resize_debounce_id = value

    @property
    def _hover_motion_cid(self: ECGApp) -> "Optional[int]":
        return self.ui.hover_motion_cid
    @_hover_motion_cid.setter
    def _hover_motion_cid(self: ECGApp, value: "Optional[int]") -> None:
        self.ui.hover_motion_cid = value

    @property
    def _hover_after_id(self: ECGApp) -> "str | None":
        return self.ui.hover_after_id
    @_hover_after_id.setter
    def _hover_after_id(self: ECGApp, value: "str | None") -> None:
        self.ui.hover_after_id = value

    @property
    def _beat_nav_cid(self: ECGApp) -> "Optional[int]":
        return self.ui.beat_nav_cid
    @_beat_nav_cid.setter
    def _beat_nav_cid(self: ECGApp, value: "Optional[int]") -> None:
        self.ui.beat_nav_cid = value

    @property
    def _rr_click_cid(self: ECGApp) -> "Optional[int]":
        return self.ui.rr_click_cid
    @_rr_click_cid.setter
    def _rr_click_cid(self: ECGApp, value: "Optional[int]") -> None:
        self.ui.rr_click_cid = value


    @property
    def _ctx_keys(self: ECGApp) -> "list":
        return self.ui.ctx_keys
    @_ctx_keys.setter
    def _ctx_keys(self: ECGApp, value: "list") -> None:
        self.ui.ctx_keys = value

    @property
    def _operation_start_time(self: ECGApp) -> "Optional[float]":
        return self.ui.operation_start_time
    @_operation_start_time.setter
    def _operation_start_time(self: ECGApp, value: "Optional[float]") -> None:
        self.ui.operation_start_time = value

    @property
    def _figure_cache(self: ECGApp) -> "dict":
        return self.ui.figure_cache
    @_figure_cache.setter
    def _figure_cache(self: ECGApp, value: "dict") -> None:
        self.ui.figure_cache = value

    @property
    def _tsv_store(self: ECGApp) -> "dict[int, str]":
        return self.ui.tsv_store
    @_tsv_store.setter
    def _tsv_store(self: ECGApp, value: "dict[int, str]") -> None:
        self.ui.tsv_store = value

    # -- SessionState --
    @property
    def _recent(self: ECGApp) -> "list[str]":
        return self.session.recent_files
    @_recent.setter
    def _recent(self: ECGApp, value: "list[str]") -> None:
        self.session.recent_files = value

    @property
    def _session_dirty(self: ECGApp) -> bool:
        return self.session.dirty
    @_session_dirty.setter
    def _session_dirty(self: ECGApp, value: bool) -> None:
        self.session.dirty = value

    @property
    def _recording_notes(self: ECGApp) -> str:
        return self.session.recording_notes
    @_recording_notes.setter
    def _recording_notes(self: ECGApp, value: str) -> None:
        self.session.recording_notes = value
