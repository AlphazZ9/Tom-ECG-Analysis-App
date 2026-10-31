# -*- coding: utf-8 -*-
"""
ecg.ui.tab_time_course
----------------------
ECGApp Time course tab: how the numbers evolve over the recording, in four modes
on one tab (Epochs, Sliding, Autonomic, Event-locked), under one out-of-date
banner. Logic in AnalysisController and EventLockedController; the Compare
Segments dialog opens from Event-locked. Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from ecg.core.autonomic import COV_METHODS
from ecg.core.models import MouseECG
from ecg.ui.autonomic_view import (
    CALIBRATION_WARNING, MIN_PAIRS_MAX, MIN_PAIRS_MIN, WINDOW_MAX_S,
    WINDOW_MIN_S, format_param, normalise_autonomic_params,
)
from ecg.ui.event_locked_view import (
    EVENT_TABLE_COLUMNS, EVENT_TABLE_NOTE, EVENT_TABLE_TITLE, EventRow,
    PRE_POST_MAX_S, PRE_POST_MIN_S, annotation_label_options,
    normalise_event_params,
)
from ecg.ui.freshness import UP_TO_DATE, stale_banner_text
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import (
    MODE_AUTONOMIC, MODE_EPOCHS, MODE_EVENT_LOCKED, MODE_MODULES, MODE_SLIDING,
    TAB_REPORT, TAB_TIME_COURSE, TIME_COURSE_MODES, time_course_mode,
)
from ecg.ui.theme import (
    THEME, BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, BLUE, GREEN, ORANGE,
    BLUE_HOVER, BLUE_DEEP, PURPLE, PURPLE_DARK, GREEN_DARK, TEAL, TEAL_DARK,
    SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_LABEL, FONT_SMALL,
    FONT_MONO, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT,
)

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

# The per-window metrics both windowed modes offer, as (key, label): the keys are
# the column names of compute_rolling_hrv, and of compute_epochs but for its
# HR_mean. Display text only. Article-style LW metrics: see the Autonomic mode.
_WINDOW_METRICS = (("HR", "HR"), ("SDNN", "SDNN"), ("RMSSD", "RMSSD"), ("pNN6", "pNN6"),
                   ("SD1", "SD1"), ("SD2", "SD2"), ("LF_nu", "LF (n.u.)"),
                   ("HF_nu", "HF (n.u.)"), ("LF_HF", "LF/HF"))
_EPOCH_DEFAULTS = ("HR", "SDNN", "RMSSD", "pNN6")
_SLIDING_DEFAULTS = ("HR", "SDNN", "RMSSD")


_EVT_DETAILS_CLOSED = "▸  Every series and the within-recording test"
_EVT_DETAILS_OPEN = "▾  Every series and the within-recording test"


class TabTimeCourseMixin:
    """ECGApp Time course tab: the four modes and the out-of-date banner."""

    def _build_tab_time_course(self: ECGApp) -> None:
        t = self.tabs.tab(TAB_TIME_COURSE)
        t.grid_rowconfigure(2, weight=1)
        t.grid_columnconfigure(0, weight=1)

        # ── Row 0: out-of-date banner, shown by _refresh_time_banner ──────
        self._time_banner = ctk.CTkFrame(t, fg_color=CARD, border_color=ORANGE,
                                         border_width=1, corner_radius=6)
        self._time_banner.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, 0))
        self._time_banner.grid_columnconfigure(1, weight=1)
        # The diamond is the out-of-date glyph of the tab bar.
        ctk.CTkLabel(self._time_banner, text="◆", font=FONT_LABEL, text_color=ORANGE,
                     ).grid(row=0, column=0, padx=(SPACE_M, SPACE_S), pady=SPACE_XS)
        self._time_banner_text = ctk.CTkLabel(
            self._time_banner, text="", font=FONT_SMALL, text_color=TEXT,
            anchor="w", justify="left", wraplength=int(1000 * THEME.font_scale))
        self._time_banner_text.grid(row=0, column=1, sticky="ew", pady=SPACE_XS)
        self.btn_time_recompute = ctk.CTkButton(
            self._time_banner, text="⟳  Recompute", command=self._recompute_time_mode,
            fg_color=BLUE, hover_color=BLUE_DEEP, text_color="white",
            font=FONT_BTN_SEC, height=28, corner_radius=5)
        self.btn_time_recompute.grid(row=0, column=2, padx=SPACE_M, pady=SPACE_XS)
        self._time_banner.grid_remove()

        # ── Row 1: mode selector ──────────────────────────────────────────
        seg_bar = ctk.CTkFrame(t, fg_color=PANEL, corner_radius=0)
        seg_bar.grid(row=1, column=0, sticky="ew")
        self._time_seg = ctk.CTkSegmentedButton(
            seg_bar, values=list(TIME_COURSE_MODES), font=FONT_SMALL,
            fg_color=PANEL, selected_color=BLUE, selected_hover_color=BLUE_HOVER,
            unselected_color=CARD, unselected_hover_color=BORDER,
            text_color=TEXT, text_color_disabled=MUTED,
            command=self._show_time_mode)
        self._time_seg.pack(padx=SPACE_M, pady=SPACE_S, anchor="w")  # type: ignore[union-attr]

        # ── Row 2: one frame per mode ─────────────────────────────────────
        content = ctk.CTkFrame(t, fg_color="transparent")
        content.grid(row=2, column=0, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        self._time_frames = {}
        for name in TIME_COURSE_MODES:
            f = ctk.CTkFrame(content, fg_color="transparent")
            f.grid(row=0, column=0, sticky="nsew")
            self._time_frames[name] = f
        self._build_time_view_epochs(self._time_frames[MODE_EPOCHS])
        self._build_time_view_sliding(self._time_frames[MODE_SLIDING])
        self._build_time_view_autonomic(self._time_frames[MODE_AUTONOMIC])
        self._build_time_view_event_locked(self._time_frames[MODE_EVENT_LOCKED])

        # The mode in use before a UI rebuild (theme toggle) stays.
        self._show_time_mode(self.ui.time_course_mode)

    def _on_main_tab_changed(self: ECGApp) -> None:
        """Opening this tab shows its mode: what showing one checks runs as well,
        the too-short warning of Epochs, the default mode, above all. Report is
        brought up to date the moment it opens."""
        if self.tabs.get() == TAB_TIME_COURSE:
            self._show_time_mode(self.ui.time_course_mode)
        elif self.tabs.get() == TAB_REPORT:
            self._refresh_report()

    def _show_time_mode(self: ECGApp, mode: str) -> None:
        """Switch the visible mode: from the selector, or when a job's result
        or a session opens it. *mode* may be a former view name."""
        mode = time_course_mode(mode) or MODE_EPOCHS
        self.ui.time_course_mode = mode
        if not self._time_frames:
            return
        for name, frame in self._time_frames.items():
            if name == mode:
                frame.grid()
            else:
                frame.grid_remove()
        if self._time_seg is not None:
            self._time_seg.set(mode)
        if mode == MODE_EPOCHS:
            # Surface a too-short-recording warning as soon as the mode is
            # shown, rather than only after a failed Compute click.
            self._refresh_epoch_feasibility()
        elif mode == MODE_EVENT_LOCKED:
            # The annotations may have been replaced without an edit (new
            # file, session restore): list their current labels.
            self.event_ctrl.refresh_label_choices()
        self._refresh_time_banner()

    def _refresh_time_banner(self: ECGApp) -> None:
        """The banner of the mode shown: derived, on the shell's tick as well,
        like the tab's state it explains."""
        if self._time_banner is None:
            return
        mode = self.ui.time_course_mode
        module = MODE_MODULES[mode]
        a = self.analysis
        result = {MODE_EPOCHS: a.epoch_df, MODE_SLIDING: a.rolling_hrv_df,
                  MODE_AUTONOMIC: a.autonomic_result, MODE_EVENT_LOCKED: a.event_locked}[mode]
        text = stale_banner_text(mode, self.freshness.status(module),
                                 self.ui.module_computed_at.get(module),
                                 self.ui.peaks_edited_at, has_values=result is not None)
        if text:
            if self._time_banner_text.cget("text") != text:
                self._time_banner_text.configure(text=text)
            if not self._time_banner.winfo_manager():
                self._time_banner.grid()
        elif self._time_banner.winfo_manager():
            self._time_banner.grid_remove()

    def _recompute_time_mode(self: ECGApp) -> None:
        """The banner's Recompute: the mode shown, with the parameters on
        screen. Event-locked derives from the Autonomic result: that one first
        when it is not current."""
        mode = self.ui.time_course_mode
        if mode == MODE_EPOCHS:
            self._compute_epochs()
        elif mode == MODE_SLIDING:
            self._compute_rolling_hrv()
        elif mode == MODE_AUTONOMIC:
            self._compute_autonomic()
        elif self.freshness.status("autonomic").state == UP_TO_DATE:
            self._compute_event_locked()
        else:
            self.analysis_ctrl.compute_autonomic(then=self._compute_event_locked)

    # ── Epochs and Sliding: one layout, one set of metrics, one legend ────

    def _windowed_entry(self: ECGApp, bar, label: str, value: str) -> ctk.CTkEntry:
        ctk.CTkLabel(bar, text=label, font=FONT_SMALL, text_color=MUTED).pack(side="left")
        ent = ctk.CTkEntry(bar, width=58, height=28, font=FONT_LABEL, fg_color=BG,
                           border_color=BORDER2, text_color=TEXT)
        ent.insert(0, value)
        ent.pack(side="left", padx=(SPACE_S, SPACE_L))
        return ent

    def _windowed_compute_button(self: ECGApp, bar, command) -> ctk.CTkButton:
        btn = ctk.CTkButton(bar, text="⟳  Compute", command=command,
                            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
                            font=FONT_BTN_SEC, height=28, corner_radius=5)
        btn.pack(side="left")
        return btn

    def _windowed_metrics(self: ECGApp, bar, defaults: "tuple[str, ...]") -> "dict":
        ctk.CTkLabel(bar, text="Metrics:", font=FONT_SMALL, text_color=MUTED).pack(side="left")
        out: "dict[str, ctk.CTkCheckBox]" = {}
        for metric, label in _WINDOW_METRICS:
            cb = ctk.CTkCheckBox(bar, text=label, font=FONT_SMALL, text_color=MUTED,
                                 fg_color=BLUE, checkmark_color="white",
                                 border_color=BORDER2, width=16)
            if metric in defaults:
                cb.select()
            cb.pack(side="left", padx=(SPACE_S, 0))
            out[metric] = cb
        return out

    def _windowed_results(self: ECGApp, parent, row: int, slot: str, height_in: float,
                          table_name: str, placeholder: str):
        """Plot, legend, copy bar and table from grid row *row* of *parent*, the
        same in both modes. Returns (copy button, table)."""
        ctrl = self.analysis_ctrl
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(row, weight=1)
        plot_frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0)
        plot_frame.grid(row=row, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_XS))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self._slots[slot] = CanvasSlot(plot_frame, 14, height_in, toolbar=False,
                                       placeholder=placeholder)
        ctk.CTkLabel(
            parent, text=ctrl.WINDOWED_LEGEND, font=FONT_HINT, text_color=MUTED, anchor="w",
            justify="left", wraplength=int(1100 * THEME.font_scale),
        ).grid(row=row + 1, column=0, sticky="ew", padx=SPACE_M)
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=row + 2, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_XS, 0))
        ctk.CTkLabel(bar, text=table_name, font=FONT_SMALL, text_color=MUTED).pack(side="left")
        # Monospace, as wrap off: the values stay under their column headers.
        box = ctk.CTkTextbox(parent, font=FONT_MONO, fg_color="transparent", text_color=TEXT,
                             border_width=0, scrollbar_button_color=BORDER,
                             scrollbar_button_hover_color=BORDER2, height=130, wrap="none")
        box.grid(row=row + 3, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        btn = ctk.CTkButton(bar, text="📋  Copy for Excel", height=24, font=FONT_SMALL,
                            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                            corner_radius=5, width=160, command=lambda: self._copy_tsv(box))
        btn.pack(side="right", padx=(0, SPACE_XS))
        return btn, box

    def _build_time_view_epochs(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Consecutive windows of the analysis window (non-overlapping at the
        default). The Excel sheet and the Prism tables hold what is ticked
        beyond the base columns (AnalysisController.compute_epochs)."""
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_XS, SPACE_S))
        self.ent_epoch = self._windowed_entry(bar, "Window (s):", str(int(MouseECG.EPOCH_DEFAULT_S)))
        # Overlap, not step: what sessions store and compute_epochs reads.
        self.ent_overlap = self._windowed_entry(bar, "Overlap (s):", "0")
        self.btn_compute_epochs = self._windowed_compute_button(bar, self._compute_epochs)
        self.sw_epoch = self._switch(bar, "Auto-run after analysis", dict(padx=0))
        self.sw_epoch.pack(side="left", padx=(SPACE_L, 0))
        self.lbl_epoch_count = ctk.CTkLabel(bar, text="", font=FONT_SMALL, text_color=BLUE)
        self.lbl_epoch_count.pack(side="right", padx=(0, SPACE_S))
        self.lbl_epoch_info = ctk.CTkLabel(bar, text="", font=FONT_HINT, text_color=MUTED)
        self.lbl_epoch_info.pack(side="right", padx=(0, SPACE_S))

        metrics_bar = ctk.CTkFrame(parent, fg_color="transparent")
        metrics_bar.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        self._epoch_metrics = self._windowed_metrics(metrics_bar, _EPOCH_DEFAULTS)
        self._bind_hover_tip(
            metrics_bar,
            "The base columns (HR, SDNN, RMSSD, pNN6) are always computed; a tick "
            "there only plots them. SD1, SD2 and LF / HF are added to the table and "
            "the exports only when ticked.")

        self.btn_copy_epochs, self.txt_epochs = self._windowed_results(
            parent, 2, "epochs", 5.5, "Epoch table:", "Click Compute to display")

    def _build_time_view_sliding(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Overlapping windows every Step seconds. CCD, SD1 (LW) and SD2 (LW) are
        ticked in the Autonomic mode and computed here (_roll_metrics)."""
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_XS, SPACE_S))
        self.ent_roll_win = self._windowed_entry(bar, "Window (s):", "30")
        self.ent_roll_step = self._windowed_entry(bar, "Step (s):", "5")
        self._bind_hover_tip(
            self.ent_roll_win,
            "Minimum 3 s. For the full article method (time-resolved CSI / CPI "
            "of Candia-Rivera et al. 2025, one window per beat) use the Autonomic mode.")
        self.btn_roll_compute = self._windowed_compute_button(bar, self._compute_rolling_hrv)
        self.lbl_roll_status = ctk.CTkLabel(
            bar, text="  Click Analyze first", font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_roll_status.pack(side="left", padx=SPACE_M)  # type: ignore[union-attr]

        metrics_bar = ctk.CTkFrame(parent, fg_color="transparent")
        metrics_bar.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        self._roll_metrics = self._windowed_metrics(metrics_bar, _SLIDING_DEFAULTS)

        self.btn_copy_rolling, self.txt_rolling = self._windowed_results(
            parent, 2, "rolling_hrv", 6.0, "Window table:", "Click Compute to display")

    # ── Autonomic ─────────────────────────────────────────────────────────

    def _build_time_view_autonomic(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Time-resolved cardiac sympathetic / parasympathetic indices (CSI /
        CPI) from sliding-window Poincaré plots -- Candia-Rivera et al.,
        J Physiol 2025 (core.autonomic) -- with optional SPWVD LF / HF
        (core.spwvd). Logic in AnalysisController.compute_autonomic, pure
        helpers in ecg.ui.autonomic_view. The parameters shown are app state
        (analysis.autonomic_params), so a rebuild keeps them."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(5, weight=1)
        p = normalise_autonomic_params(self.analysis.autonomic_params)

        # ── Rows 0-1: parameters, Compute and Calibrate ──
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))
        run = ctk.CTkFrame(parent, fg_color="transparent")
        run.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))

        def _entry(label: str, value: float, width: int, tip: str) -> ctk.CTkEntry:
            ctk.CTkLabel(bar, text=label, font=FONT_SMALL,
                         text_color=MUTED).pack(side="left")
            ent = ctk.CTkEntry(bar, width=width, height=28, font=FONT_LABEL,
                               fg_color=BG, border_color=BORDER2, text_color=TEXT)
            ent.insert(0, format_param(value))
            ent.pack(side="left", padx=(SPACE_S, SPACE_L))
            self._bind_hover_tip(ent, tip)
            return ent

        self.ent_auto_win = _entry(
            "Window (s):", p["window_s"], 52,
            f"Sliding Poincaré window T, s ({WINDOW_MIN_S:g}–{WINDOW_MAX_S:g}; the "
            "paper used 3 s in rats). One window ends on every valid beat; a "
            "window with fewer valid adjacent RR pairs than Min pairs is "
            "withheld (shaded grey). Longer = smoother, less time-resolved.")
        self.ent_auto_minpairs = _entry(
            "Min pairs:", p["min_pairs"], 40,
            f"Valid adjacent RR pairs a window needs ({MIN_PAIRS_MIN}–{MIN_PAIRS_MAX}; "
            "default 10), else it is withheld. This floor is the app's addition: "
            "the reference code (compute_CSI_CVI_rat.m) computes every window — "
            "2 reproduces it. At 10 pairs a 3 s window is withheld wherever HR "
            "is below about 200 bpm.")

        ctk.CTkLabel(bar, text="Method:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        self.cb_auto_method = ctk.CTkComboBox(
            bar, width=int(330 * THEME.font_scale), height=28, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, button_color=BORDER2,
            text_color=TEXT, dropdown_fg_color=BG, dropdown_text_color=TEXT,
            values=list(COV_METHODS.values()), state="readonly",
            command=self._on_autonomic_method_change,
        )
        self.cb_auto_method.set(COV_METHODS[p["method"]])
        self.cb_auto_method.pack(side="left", padx=(SPACE_S, SPACE_L))
        self._bind_hover_tip(
            self.cb_auto_method,
            "Covariance estimator behind SD1 / SD2: Robust = Schafer-Strimmer "
            "shrinkage (what the paper text describes; default) · Rat reference = "
            "plain cov + 5 % trimmed means (the published rat code) · Exact = plain "
            "cov · Approximate = closed-form SD1 / SD2 · Ledoit-Wolf = sklearn, "
            "comparison only.")

        self.ent_auto_ks = _entry(
            "ks:", p["ks"], 48,
            "Sympathetic gain: CSI = D̃ + ks·SD2 (D̃ = the CCD mirrored about "
            "its mean, so a shorter cycle raises CSI). 3 = the paper's rat (SHR) "
            "value — not validated in mice.")
        self.ent_auto_kp = _entry(
            "kp:", p["kp"], 48,
            "Parasympathetic gain: CPI = D + kp·SD1 (a longer cycle raises "
            "CPI). 12 = the paper's rat (SHR) value — not validated in mice.")

        self.chk_auto_lfhf = ctk.CTkCheckBox(
            run, text="LF/HF (SPWVD)", font=FONT_SMALL, text_color=MUTED,
            fg_color=BLUE, checkmark_color="white", border_color=BORDER2, width=16)
        if p["lf_hf"]:
            self.chk_auto_lfhf.select()
        self.chk_auto_lfhf.pack(side="left", padx=(0, SPACE_L))
        self._bind_hover_tip(
            self.chk_auto_lfhf,
            "Also compute time-resolved LF and HF power with the smoothed pseudo "
            "Wigner-Ville distribution (20 Hz resampling, the HRV band preset). "
            "Slow: about 30 s per hour of recording.")

        self.btn_auto_compute = ctk.CTkButton(
            run, text="⟳  Compute", command=self._compute_autonomic,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_SEC, height=28, corner_radius=5)
        self.btn_auto_compute.pack(side="left")
        self._bind_hover_tip(
            self.btn_auto_compute,
            "Compute CSI / CPI on the analysis window's peaks, with the sidebar "
            "Min / Max RR bounds: intervals outside them or spanning a removed "
            "beat are excluded, never bridged. Runs in the background.")
        self._calibrate_button(run).pack(side="left", padx=(SPACE_S, 0))

        # ── Row 2: status ──
        self.lbl_auto_status = ctk.CTkLabel(
            parent, text=self.analysis_ctrl.AUTONOMIC_IDLE_TEXT,
            font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_auto_status.grid(row=2, column=0, sticky="ew", padx=SPACE_M)

        # ── Row 3: persistent calibration warning ──
        ctk.CTkLabel(
            parent, text="⚠  " + CALIBRATION_WARNING, font=FONT_HINT,
            text_color=ORANGE, anchor="w", justify="left",
            wraplength=int(1100 * THEME.font_scale),
        ).grid(row=3, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))

        # ── Row 4: article-style LW metrics ──
        # They stay in _roll_metrics and in compute_rolling_hrv, on the Sliding
        # window and step, so their values and Prism columns do not change. "(LW)"
        # in the texts: sklearn's Ledoit-Wolf shrinkage, not the paper's estimator.
        article_bar = ctk.CTkFrame(parent, fg_color="transparent")
        article_bar.grid(row=4, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        ctk.CTkLabel(article_bar, text="Article-style Poincaré (LW), per Sliding window:",
                     font=FONT_SMALL, text_color=MUTED).pack(side="left")
        for metric, label in (("CCD", "CCD"), ("SD1_LW", "SD1 (LW)"), ("SD2_LW", "SD2 (LW)")):
            cb = ctk.CTkCheckBox(
                article_bar, text=label, font=FONT_SMALL, text_color=MUTED, fg_color=BLUE,
                checkmark_color="white", border_color=BORDER2, width=16)
            cb.pack(side="left", padx=(SPACE_S, 0))
            self._roll_metrics[metric] = cb
        btn_lw = ctk.CTkButton(
            article_bar, text="⟳  Compute in Sliding", command=self._compute_rolling_hrv,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, height=26, corner_radius=5)
        btn_lw.pack(side="left", padx=(SPACE_M, 0))
        self._bind_hover_tip(
            btn_lw,
            "Computes the Sliding mode with what is ticked there and here, on its "
            "window and step, and opens it. Not the CSI / CPI above.")
        ctk.CTkLabel(
            article_bar, text="(LW = Ledoit–Wolf; ≥10 successive valid RR pairs)",
            font=FONT_HINT, text_color=MUTED,
        ).pack(side="left", padx=(SPACE_S, 0))

        # ── Row 5: stacked plot ──
        plot_frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0)
        plot_frame.grid(row=5, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_S))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self._slots["autonomic"] = CanvasSlot(
            plot_frame, 14, 7.0, toolbar=True,
            placeholder="Click Compute to display")

        # ── Rows 6-7: recording-level values ──
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.grid(row=6, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))
        ctk.CTkLabel(hdr, text="RECORDING-LEVEL VALUES", font=FONT_SIDEBAR_HDR,
                     text_color=MUTED).pack(side="left")
        btn_copy = ctk.CTkButton(
            hdr, text="📋  Copy for Excel", height=24,
            font=FONT_SMALL, fg_color=BORDER, hover_color=BORDER2,
            text_color=MUTED, corner_radius=5, width=160,
            command=lambda: self._copy_tsv(self.txt_autonomic))
        btn_copy.pack(side="right", padx=(0, SPACE_XS))
        self._bind_hover_tip(
            btn_copy,
            "Copy the recording-level values as one tab-separated Excel row "
            "(paste one row per recording)")
        self.txt_autonomic = ctk.CTkTextbox(
            parent, font=FONT_MONO, fg_color=PANEL, text_color=TEXT,
            border_width=0, scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER2, height=150)
        self.txt_autonomic.grid(row=7, column=0, sticky="ew", padx=SPACE_M,
                                pady=(0, SPACE_S))

    def _calibrate_button(self: ECGApp, parent) -> ctk.CTkButton:
        btn = ctk.CTkButton(
            parent, text="⚖  Calibrate ks/kp…", command=self._open_calibration,
            fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
            font=FONT_BTN_SEC, height=26, corner_radius=5)
        self._bind_hover_tip(
            btn,
            "Cohort calibration of the gains from the registry: Wilcoxon Z(k) "
            "of CSI and CPI for k = 1–20 and its knee (the paper's rule), per "
            "event or per subject mean.")
        return btn

    # ── Event-locked ──────────────────────────────────────────────────────

    def _build_time_view_event_locked(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Per-event 30 s pre / post means of CSI, CPI, HR, D, SD1, SD2 (and LF /
        HF) around the annotations, the median ± MAD traces, the cohort registry
        and the ks / kp calibration, and the way to the Compare Segments dialog.
        Logic in EventLockedController, pure helpers in
        ecg.ui.event_locked_view. The parameters shown are app state
        (analysis.event_locked_params), so a rebuild keeps them."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1)
        p = normalise_event_params(self.analysis.event_locked_params)

        # ── Row 0: events, windows, Compute, z-score ──
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))
        ctk.CTkLabel(bar, text="Events:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        labels = annotation_label_options(self.analysis.annotations)
        self.cb_evt_label = ctk.CTkComboBox(
            bar, width=int(190 * THEME.font_scale), height=28, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, button_color=BORDER2,
            text_color=TEXT, dropdown_fg_color=BG, dropdown_text_color=TEXT,
            values=labels, state="readonly")
        self.cb_evt_label.set(p["label"] if p["label"] in labels else labels[0])
        self.cb_evt_label.pack(side="left", padx=(SPACE_S, SPACE_L))
        self._bind_hover_tip(
            self.cb_evt_label,
            "Annotations used as events: All, one label, or the unlabelled ones. "
            "Each event's onset is its annotation's START time (add them in "
            "Annotations).")

        def _entry(label: str, value: float, tip: str) -> ctk.CTkEntry:
            ctk.CTkLabel(bar, text=label, font=FONT_SMALL,
                         text_color=MUTED).pack(side="left")
            ent = ctk.CTkEntry(bar, width=52, height=28, font=FONT_LABEL,
                               fg_color=BG, border_color=BORDER2, text_color=TEXT)
            ent.insert(0, format_param(value))
            ent.pack(side="left", padx=(SPACE_S, SPACE_L))
            self._bind_hover_tip(ent, tip)
            return ent

        self.ent_evt_pre = _entry(
            "Pre (s):", p["pre_s"],
            f"Window before each onset, s ({PRE_POST_MIN_S:g}–{PRE_POST_MAX_S:g}; the "
            "paper compared the 30 s before vs the 30 s after each condition change).")
        self.ent_evt_post = _entry(
            "Post (s):", p["post_s"],
            "Window after each onset, s (the onset sample belongs to it).")

        self.btn_evt_compute = ctk.CTkButton(
            bar, text="⟳  Compute event-locked", command=self._compute_event_locked,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_SEC, height=28, corner_radius=5)
        self.btn_evt_compute.pack(side="left", padx=(0, SPACE_L))
        self._bind_hover_tip(
            self.btn_evt_compute,
            "Mean of every series over the window before and after each event "
            "(needs a computed Autonomic result). An event is valid when no "
            "other annotation starts inside its windows (any label), and both "
            "windows lie inside the CSI / CPI series and are ≥ 80 % covered.")

        self.chk_evt_z = ctk.CTkCheckBox(
            bar, text="z-score (display)", font=FONT_SMALL, text_color=MUTED,
            fg_color=BLUE, checkmark_color="white", border_color=BORDER2, width=16,
            command=self._on_event_zscore_toggle)
        if p["zscore"]:
            self.chk_evt_z.select()
        self.chk_evt_z.pack(side="left")
        self._bind_hover_tip(
            self.chk_evt_z,
            "Plot each series z-scored by its whole-recording mean and SD (the "
            "paper's visualisation). Display only: the table, the tests and the "
            "registry use the raw window means.")

        # ── Row 1: status ──
        self.lbl_evt_status = ctk.CTkLabel(
            parent, text=self.event_ctrl.IDLE_TEXT, font=FONT_SMALL,
            text_color=MUTED, anchor="w")
        self.lbl_evt_status.grid(row=1, column=0, sticky="ew", padx=SPACE_M)

        # ── Row 2: traces next to the per-event table ──
        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_S))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=3, uniform="evt")
        body.grid_columnconfigure(1, weight=2, uniform="evt")
        plot_frame = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=0)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_S))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self._slots["event_locked"] = CanvasSlot(
            plot_frame, 9, 6.0, toolbar=True,
            placeholder="Compute the Autonomic indices, then Compute event-locked")
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self.lbl_evt_table_head = ctk.CTkLabel(
            right, text=EVENT_TABLE_TITLE, font=FONT_SIDEBAR_HDR, text_color=MUTED, anchor="w")
        self.lbl_evt_table_head.grid(row=0, column=0, sticky="ew")
        self._evt_rows_frame = ctk.CTkScrollableFrame(
            right, fg_color=PANEL, height=150, scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER2)
        self._evt_rows_frame.grid(row=1, column=0, sticky="nsew")
        self._evt_rows_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._evt_rows_frame.grid_columnconfigure(4, weight=2)
        self._fill_event_rows([])
        note = ctk.CTkLabel(right, text=EVENT_TABLE_NOTE, font=FONT_SMALL, text_color=MUTED,
                            anchor="w", justify="left", wraplength=int(420 * THEME.font_scale))
        note.grid(row=2, column=0, sticky="ew", pady=(SPACE_XS, SPACE_XS))
        right.bind("<Configure>", lambda e: self._wrap_report_labels((note,), e.width, 8))

        # ── registry / export / calibration, beside what they save ──
        bar2 = ctk.CTkFrame(right, fg_color="transparent")
        bar2.grid(row=3, column=0, sticky="ew")
        btn_save = ctk.CTkButton(
            bar2, text="💾  Save to cohort registry", command=self._save_event_locked,
            fg_color=GREEN_DARK, hover_color=GREEN, text_color="white",
            font=FONT_BTN_SEC, height=26, corner_radius=5)
        btn_save.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(
            btn_save,
            "Store this recording's per-event pre / post / delta values in the "
            "local registry with the Subject ID, the Project name and the "
            "method / window / ks / kp — the cohort the calibration runs on. "
            "With the 'All' filter, this recording's earlier events of the same "
            "method and window are replaced.")
        btn_csv = ctk.CTkButton(
            bar2, text="Export CSV…", command=self._export_event_locked_csv,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, height=26, corner_radius=5, width=110)
        btn_csv.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(btn_csv, "Save the per-event table as CSV (one line per event)")
        bar3 = ctk.CTkFrame(right, fg_color="transparent")
        bar3.grid(row=4, column=0, sticky="ew", pady=(SPACE_XS, 0))
        self._calibrate_button(bar3).pack(side="left", padx=(0, SPACE_S))
        # Not the same calculation as the means above: two windows chosen by
        # hand, with their own block statistics and Excel export.
        btn_segments = ctk.CTkButton(
            bar3, text="⚖  Compare Segments…", command=self._open_compare_segments,
            # Dark mode lightens TEAL for legibility against the dark
            # background (_adapt_for_mode in theme.py), which then leaves
            # white label text well under WCAG contrast on the same fill --
            # a dark label reads clearly against the lightened dark-mode
            # TEAL instead.
            fg_color=TEAL, hover_color=TEAL_DARK,
            text_color=(BG if THEME.is_dark else "white"),
            font=FONT_BTN_SEC, height=26, corner_radius=5)
        btn_segments.pack(side="left")
        self._bind_hover_tip(
            btn_segments,
            "Two time windows A and B chosen by hand: block statistics and tachogram, "
            "no annotation needed (Ctrl+M). Other comparisons: Check Agreement "
            "(Detection section) · Cohort Trends (Recent recordings).")

        # ── Rows 3-4: every series and the within-recording test, behind a disclosure ──
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.grid(row=3, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))
        self.btn_evt_details = ctk.CTkButton(
            hdr, text=_EVT_DETAILS_CLOSED, anchor="w", command=self._toggle_event_details,
            fg_color="transparent", hover_color=BORDER, text_color=MUTED,
            font=FONT_BTN_SEC, height=24, corner_radius=6)
        self.btn_evt_details.pack(side="left")
        btn_copy = ctk.CTkButton(
            hdr, text="📋  Copy for Excel", height=24,
            font=FONT_SMALL, fg_color=BORDER, hover_color=BORDER2,
            text_color=MUTED, corner_radius=5, width=160,
            command=lambda: self._copy_tsv(self.txt_event_locked))
        btn_copy.pack(side="right", padx=(0, SPACE_XS))
        self._bind_hover_tip(
            btn_copy,
            "Copy the per-event table (one row per event: pre / post / delta of "
            "every series) as tab-separated values for Excel")
        self.txt_event_locked = ctk.CTkTextbox(
            parent, font=FONT_MONO, fg_color=PANEL, text_color=TEXT,
            border_width=0, scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=BORDER2, height=190)
        self.txt_event_locked.grid(row=4, column=0, sticky="ew", padx=SPACE_M,
                                   pady=(0, SPACE_S))
        self.txt_event_locked.grid_remove()

    # ── Handlers ──────────────────────────────────────────────────────────

    def _compute_autonomic(self: ECGApp) -> None:
        """Compute the time-resolved CSI / CPI and draw the Autonomic view."""
        self.analysis_ctrl.compute_autonomic()

    def _on_autonomic_method_change(self: ECGApp, choice: str) -> None:
        self.analysis_ctrl.on_autonomic_method_change(choice)

    def _toggle_event_details(self: ECGApp) -> None:
        shown = not self.txt_event_locked.winfo_manager()
        if shown:
            self.txt_event_locked.grid()
        else:
            self.txt_event_locked.grid_remove()
        self.btn_evt_details.configure(text=_EVT_DETAILS_OPEN if shown else _EVT_DETAILS_CLOSED)

    def _fill_event_rows(self: ECGApp, rows: "list[EventRow]") -> None:
        """The compact per-event table, read from the result's own rows."""
        frame = self._evt_rows_frame
        for child in frame.winfo_children():
            child.destroy()
        for c, head in enumerate(EVENT_TABLE_COLUMNS):
            ctk.CTkLabel(frame, text=head, font=FONT_HINT, text_color=MUTED, height=18,
                         anchor="w" if c in (0, 4) else "e").grid(
                row=0, column=c, sticky="ew", padx=(SPACE_M if c == 4 else 0, SPACE_S))
        if not rows:
            ctk.CTkLabel(frame, text="—", font=FONT_SMALL, text_color=MUTED, height=20,
                         anchor="w").grid(row=1, column=0, sticky="ew")
        for r, row in enumerate(rows, start=1):
            for c, text in enumerate((row.onset, row.hr, row.csi, row.cpi, row.valid)):
                color = TEXT if c < 4 or row.is_valid else ORANGE
                ctk.CTkLabel(frame, text=text, font=FONT_SMALL, text_color=color, height=20,
                             anchor="w" if c in (0, 4) else "e").grid(
                    row=r, column=c, sticky="ew", padx=(SPACE_M if c == 4 else 0, SPACE_S))

    def _compute_event_locked(self: ECGApp) -> None:
        """Lock the Autonomic result to the annotations (Event-locked)."""
        self.event_ctrl.compute()

    def _on_event_zscore_toggle(self: ECGApp) -> None:
        self.event_ctrl.on_zscore_toggle()

    def _save_event_locked(self: ECGApp) -> None:
        self.event_ctrl.save_to_registry()

    def _export_event_locked_csv(self: ECGApp) -> None:
        self.event_ctrl.export_csv()

    def _open_calibration(self: ECGApp) -> None:
        self.event_ctrl.open_calibration()

    def _compute_rolling_hrv(self: ECGApp) -> None:
        """Compute sliding-window HRV and render the timeline plot."""
        self.analysis_ctrl.compute_rolling_hrv()
