# -*- coding: utf-8 -*-
"""
ecg.ui.app_right_panel
----------------------
ECGApp right panel: detection, artifacts, ML, exports, stats, annotations.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from ecg.core.models import MouseECG
from ecg.ui.theme import (
    THEME, BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT, RED, BLUE,
    BLUE_MID, BLUE_DEEP, PURPLE, PURPLE_DARK, TEAL, GREEN_DARK, ORANGE_DARK,
    SPACE_XS, SPACE_S, SPACE_M, SPACE_L, FONT_LABEL, FONT_SMALL, FONT_KPI_LABEL,
    FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_MICRO, FONT_HINT,
    FONT_SUBSECTION, FONT_CARD_TITLE,
)
from ecg.ui.widgets import make_stat_tile
from ecg.ui.sidebar import CollapsibleSection

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class RightPanelMixin:
    """ECGApp right panel: detection, artifacts, ML, exports, stats, annotations."""

    # ─── Right accordion panel ───────────────────────────────────

    def _build_right_panel(self: ECGApp) -> None:
        """Fixed-width accordion panel on the right edge of the window.

        Sibling of self.sidebar (not nested in `main`) -- packed side="right"
        before `main` is created so main's fill="both", expand=True doesn't
        claim the space first, mirroring how the sidebar itself is packed
        before main. Wrapped in a scrollable frame from the start since later
        sub-phases add more sections here.

        Order: STATISTICS / EXPORTS / ML DETECTOR SAVING. Detection/Artifacts/
        Annotations moved to the left sidebar (see _build_sidebar()).
        """
        self.right_panel = ctk.CTkFrame(self, width=280, fg_color=PANEL, corner_radius=0)
        self.right_panel.pack(side="right", fill="y")
        self.right_panel.pack_propagate(False)
        ctk.CTkFrame(self.right_panel, width=1, fg_color=BORDER).pack(side="left", fill="y")

        scroll = ctk.CTkScrollableFrame(self.right_panel, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._build_stat_section(scroll)
        self._build_exports_section(scroll)
        self._build_ml_detector_section(scroll)

    def _build_detection_section(self: ECGApp, parent) -> None:
        """DETECTION accordion section (left sidebar).

        Consolidates content that had no independent collapse state before
        (the sidebar's always-visible Method/Analysis-window/Status/Threshold
        block, plus its own Min R-R entry and ADVANCED's SG Options
        sub-group) into one open-by-default section.
        """
        fpx = dict(padx=SPACE_L)

        # ══════════════════════════════════════════════════════
        #  DETECTION
        # ══════════════════════════════════════════════════════
        sec_det = CollapsibleSection(parent, "DETECTION", initially_open=True)
        f = sec_det.frame

        # ── Method ──────────────────────────────────────────
        det_card = ctk.CTkFrame(f, fg_color=CARD, corner_radius=8,
                                border_width=1, border_color=BORDER)
        det_card.pack(fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        det_card_row = ctk.CTkFrame(det_card, fg_color="transparent")
        det_card_row.pack(fill="x", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))
        det_card_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(det_card_row, text="Method",
                     font=FONT_SIDEBAR_HDR, text_color=TEXT,
                     anchor="w").grid(row=0, column=0, sticky="w", padx=(0, SPACE_M))
        self.cb_det_method = ctk.CTkComboBox(
            det_card_row, font=FONT_LABEL, height=26,
            fg_color=BG, border_color=BLUE, button_color=BLUE,
            text_color=TEXT, dropdown_fg_color=BG, dropdown_text_color=TEXT,
            values=["Auto (NeuroKit2)", "SG + Derivative (10 kHz)", "Wavelet (CWT)", "Envelope Max", "ML Detector"],
            command=self._on_det_method_change)
        self.cb_det_method.set("SG + Derivative (10 kHz)")
        self.cb_det_method.grid(row=0, column=1, sticky="ew")
        # Shown only when Method = ML Detector: the model must be trained
        # first (right panel > ML DETECTOR SAVING), which is easy to miss
        # since that section starts collapsed on the opposite side of the
        # window -- surface its status right here instead. Hidden/packed on
        # demand by _on_det_method_change; kept in sync by refresh_ml_status.
        self.lbl_ml_inline_status = ctk.CTkLabel(
            det_card, text="", font=FONT_KPI_LABEL, text_color=MUTED,
            anchor="w", wraplength=220, justify="left")

        # ── Analysis window (inline, compact) ──────────────────
        aw_frame = ctk.CTkFrame(f, fg_color=CARD, corner_radius=8,
                                border_width=1, border_color=BORDER)
        aw_frame.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_XS))
        aw_hdr = ctk.CTkFrame(aw_frame, fg_color="transparent")
        aw_hdr.pack(fill="x", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))
        ctk.CTkLabel(aw_hdr, text="Analysis window", font=FONT_SUBSECTION,
                     text_color=TEXT, anchor="w").pack(side="left")
        ctk.CTkLabel(aw_hdr, text="optional", font=FONT_KPI_LABEL,
                     text_color=LIGHT).pack(side="right")
        aw_entries = ctk.CTkFrame(aw_frame, fg_color="transparent")
        aw_entries.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_XS))
        aw_entries.columnconfigure(1, weight=1)
        aw_entries.columnconfigure(3, weight=1)
        ctk.CTkLabel(aw_entries, text="From", font=FONT_SMALL, text_color=MUTED,
                     width=36, anchor="w").grid(row=0, column=0)
        self.ent_analysis_t0 = ctk.CTkEntry(
            aw_entries, height=24, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, text_color=TEXT,
            corner_radius=5, placeholder_text="0 s")
        self.ent_analysis_t0.grid(row=0, column=1, sticky="ew", padx=(SPACE_XS, SPACE_M))
        ctk.CTkLabel(aw_entries, text="To", font=FONT_SMALL, text_color=MUTED,
                     width=24, anchor="w").grid(row=0, column=2)
        self.ent_analysis_t1 = ctk.CTkEntry(
            aw_entries, height=24, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, text_color=TEXT,
            corner_radius=5, placeholder_text="end")
        self.ent_analysis_t1.grid(row=0, column=3, sticky="ew", padx=(SPACE_XS, 0))
        aw_btns = ctk.CTkFrame(aw_frame, fg_color="transparent")
        aw_btns.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_XS))
        ctk.CTkButton(aw_btns, text="Apply", width=70, height=26,
                      fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
                      font=FONT_SMALL, corner_radius=5,
                      command=self._apply_analysis_window).pack(side="left", padx=(0, SPACE_S))
        ctk.CTkButton(aw_btns, text="Full", width=56, height=26,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FONT_SMALL, corner_radius=5,
                      command=self._reset_analysis_window).pack(side="left")
        self.lbl_analysis_window = ctk.CTkLabel(
            aw_btns, text="", font=FONT_KPI_LABEL, text_color=MUTED, anchor="e")
        self.lbl_analysis_window.pack(side="right", fill="x", expand=True)

        ctk.CTkFrame(f, height=1, fg_color=BORDER).pack(fill="x", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))

        # Status readout lives in the always-visible toolbar now (see
        # _build_toolbar) -- it used to be a child of this collapsible
        # section, so collapsing DETECTION or hiding the left panel made
        # every busy/error/success message (most of which have nothing to
        # do with detection) silently disappear.

        # ── Threshold — always visible, prominent ─────────────
        thr_card = ctk.CTkFrame(f, fg_color=CARD, corner_radius=8,
                                border_width=1, border_color=BORDER)
        thr_card.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_S))
        thr_top_row = ctk.CTkFrame(thr_card, fg_color="transparent")
        thr_top_row.pack(fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        self.lbl_thr = ctk.CTkLabel(
            thr_top_row, text="Threshold:  0.50",
            font=FONT_CARD_TITLE, text_color=TEXT, anchor="w")
        self.lbl_thr.pack(side="left")
        ctk.CTkLabel(thr_top_row, text="strict ↑  /  sensitive ↓",
                     font=FONT_KPI_LABEL, text_color=LIGHT, anchor="e").pack(side="right")
        self.sl_thr = ctk.CTkSlider(
            thr_card, from_=0.01, to=2.0,  # type: ignore
            progress_color=RED, button_color=RED,
            button_hover_color="#FF5252", fg_color=BORDER,
            height=20, command=self._on_threshold_slide)
        self.sl_thr.set(0.50)
        self.sl_thr.pack(fill="x", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))
        thr_bot = ctk.CTkFrame(thr_card, fg_color="transparent")
        thr_bot.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_S))
        thr_bot.columnconfigure(1, weight=1)
        ctk.CTkLabel(thr_bot, text="Exact:", font=FONT_SMALL,
                     text_color=MUTED).grid(row=0, column=0, padx=(0, SPACE_S))
        self.ent_thr = ctk.CTkEntry(thr_bot, height=24, font=FONT_LABEL,
                                     fg_color=PANEL, border_color=BORDER2, text_color=TEXT)
        self.ent_thr.insert(0, "0.50")
        self.ent_thr.grid(row=0, column=1, sticky="ew")
        self.ent_thr.bind("<Return>",   self._on_threshold_entry)
        self.ent_thr.bind("<FocusOut>", self._on_threshold_entry)
        self.lbl_thr_hint = ctk.CTkLabel(
            thr_card,
            text="Fraction of the typical (median) R-peak height — 0.50 "
                 "keeps peaks ≥ 50% as tall. Lower = more peaks.",
            font=FONT_KPI_LABEL, text_color=LIGHT, anchor="w",
            wraplength=230, justify="left")
        self.lbl_thr_hint.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_S))

        # ── Min R-R ──────────────────────────────────────────
        # Named "Min R-R interval (ms)" to match the Parameters dialog's copy
        # of the same field (was "Min R-R physio (ms)" here only).
        self._sidebar_entry(f, "Min R-R interval (ms)", "minrr",
                            f"{MouseECG.MIN_RR_MS:g}", fpx)
        ctk.CTkLabel(f, text="Peaks closer together than this are merged into one beat",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=230).pack(**fpx, fill="x", pady=(0, SPACE_XS))
        # ── Max R-R ──────────────────────────────────────────
        # With Min R-R above, the RR range accepted for HRV: intervals
        # outside it are excluded (never bridged) by every HRV computation.
        # The default (≈ 333 ms = 180 bpm) assumes an awake mouse.
        self._sidebar_entry(f, "Max RR (ms)", "maxrr",
                            f"{MouseECG.RR_MAX_MS:g}", fpx)
        ctk.CTkLabel(f, text="HRV uses only RR intervals between Min and Max — "
                             "widen Max for anaesthetised mice (< 180 bpm)",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=230).pack(**fpx, fill="x", pady=(0, SPACE_XS))
        ctk.CTkLabel(f, text="SG+Deriv: downsample → Savitzky-Golay derivative\n"
                             "Wavelet: CWT separates noise / QRS / J-wave (needs PyWavelets)\n"
                             "Envelope Max: local maxima — best for clipped/saturated signals",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=230).pack(**fpx, fill="x", pady=(0, SPACE_S))

        # ── Savitzky-Golay options ──────────────────────────
        ctk.CTkFrame(f, height=1, fg_color=BORDER).pack(
            fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        ctk.CTkLabel(f, text="Savitzky-Golay options", font=FONT_SUBSECTION,
                     text_color=MUTED, anchor="w").pack(**fpx, fill="x", pady=(0, SPACE_XS))
        self._sg_frame = ctk.CTkFrame(f, fg_color="transparent")
        # "SG target fs" (not the bare "Target fs" this used to say) to match
        # the Parameters dialog's name for the same field.
        self._sidebar_entry_row(self._sg_frame, fpx, [
            ("SG target fs (Hz)", "sg_target_fs", "10000"),
            ("SG window (ms)", "sg_window_ms",  "20"),
        ])
        ctk.CTkLabel(self._sg_frame,
                     text="Signal is resampled to this rate, then differentiated; "
                          "window should span about one QRS complex",
                     font=FONT_KPI_LABEL, text_color=LIGHT, anchor="w",
                     wraplength=230, justify="left").pack(**fpx, fill="x", pady=(0, SPACE_XS))
        # Sync initial visibility to the Method combobox's default -- DETECTION
        # defaults open (unlike the old ADVANCED home for this frame, which
        # defaulted closed), so an un-synced _sg_frame would show an empty
        # "Savitzky-Golay options" header on first launch even though Method
        # defaults to "SG + Derivative (10 kHz)".
        self._on_det_method_change(self.cb_det_method.get())

        # ── Cross-method agreement ───────────────────────────
        ctk.CTkFrame(f, height=1, fg_color=BORDER).pack(
            fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        # Built with an explicit muted fg_color (see _set_btn_enabled) rather
        # than relying on CTkButton's default disabled treatment, which only
        # dims the text and left this looking clickable while disabled.
        self.btn_check_agreement = ctk.CTkButton(
            f, text="🔍  Check Agreement",
            command=self._check_method_agreement,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            font=FONT_BTN_SEC, height=28, corner_radius=8, state="disabled")
        self.btn_check_agreement.pack(**fpx, fill="x", pady=(0, SPACE_XS))
        self._bind_hover_tip(
            self.btn_check_agreement,
            "Cross-check accepted beats against other detection methods")
        self.lbl_agreement_status = ctk.CTkLabel(
            f, text="  Not run yet", font=FONT_KPI_LABEL, text_color=LIGHT,
            anchor="w", wraplength=230, justify="left")
        self.lbl_agreement_status.pack(**fpx, fill="x", pady=(0, SPACE_S))

    def _build_artifacts_section(self: ECGApp, parent) -> None:
        """ARTIFACTS accordion section (left sidebar), relocated verbatim
        from the old combined Detection/Artifacts/ML-Detector builder."""
        fpx = dict(padx=SPACE_L)
        sec_art = CollapsibleSection(parent, "ARTIFACTS", initially_open=False)
        self.sec_artifacts = sec_art
        f = sec_art.frame
        # Built with an explicit muted fg_color (see _set_btn_enabled) rather
        # than relying on CTkButton's default disabled treatment, which only
        # dims the text and left this looking clickable while disabled.
        self.btn_review_art = ctk.CTkButton(
            f, text="🔍  Review Artifacts",
            command=self._open_artifact_review,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            font=FONT_BTN_PRIMARY, height=max(30, int(34 * THEME.font_scale)),
            corner_radius=8, state="disabled")
        self.btn_review_art.pack(**fpx, fill="x", pady=(SPACE_S, SPACE_XS))
        ctk.CTkLabel(f, text="Detect + review every artifact. Click Detect Peaks first.",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=230, justify="left").pack(**fpx, fill="x", pady=(0, SPACE_S))
        self.sw_artifact = self._switch(
            f, "Auto-correct when you click Analyze", fpx, default_on=False)
        ctk.CTkLabel(f, text="OFF by default — use Review for full control",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=230).pack(**fpx, fill="x", pady=(0, SPACE_S))

    def _build_ml_detector_section(self: ECGApp, parent) -> None:
        """ML DETECTOR SAVING accordion section (right panel), relocated
        verbatim from the old combined Detection/Artifacts/ML-Detector
        builder -- same content/bindings, header relabeled per the user's
        request."""
        fpx = dict(padx=SPACE_L)
        sec_ml = CollapsibleSection(parent, "ML DETECTOR SAVING", initially_open=False)
        f = sec_ml.frame
        self.lbl_ml_status = ctk.CTkLabel(
            f, text="No trained model yet", font=FONT_HINT,
            text_color=MUTED, anchor="w", wraplength=230, justify="left")
        self.lbl_ml_status.pack(**fpx, fill="x", pady=(SPACE_S, SPACE_XS))
        self.sw_verified_training = self._switch(
            f, "✓ Verified for training", fpx, default_on=False,
            command=self._on_verified_training_toggle)
        ctk.CTkLabel(
            f, text="Marks this recording's corrected R-peaks as clean "
                    "training data. Takes effect on Save Session.",
            font=FONT_KPI_LABEL, text_color=LIGHT,
            anchor="w", wraplength=230, justify="left").pack(**fpx, fill="x", pady=(0, SPACE_S))
        self.btn_save_for_training = ctk.CTkButton(
            f, text="📥  Save for Training  (no session save)",
            command=self._save_for_training_only,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            font=FONT_BTN_SEC, height=max(28, int(30 * THEME.font_scale)),
            corner_radius=8)
        self.btn_save_for_training.pack(**fpx, fill="x", pady=(0, SPACE_S))
        ctk.CTkLabel(
            f, text="Caches this recording's peaks as training data right "
                    "away, without writing a .ecgsession file or export stats.",
            font=FONT_KPI_LABEL, text_color=LIGHT,
            anchor="w", wraplength=230, justify="left").pack(**fpx, fill="x", pady=(0, SPACE_S))
        self.btn_train_ml = ctk.CTkButton(
            f, text="🤖  Train / Retrain Model…",
            command=self._open_ml_training_dialog,
            fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
            font=FONT_BTN_PRIMARY, height=max(30, int(34 * THEME.font_scale)),
            corner_radius=8)
        self.btn_train_ml.pack(**fpx, fill="x", pady=(0, SPACE_S))
        self.after(0, self.refresh_ml_status)

    def _build_exports_section(self: ECGApp, parent) -> None:
        """EXPORTS accordion section (right panel): the export-format
        buttons, moved verbatim from the old ADVANCED section's
        "Export & Notes" group (Notes itself now lives in the SIGNAL
        section, beside Subject ID). This is now the sole entry point for
        exports -- the toolbar's old "Export ▾" dropdown duplicated the
        same 7 commands and was removed."""
        fpx = dict(padx=SPACE_L)
        sec = CollapsibleSection(parent, "EXPORTS", initially_open=False)
        self.sec_exports = sec
        f = sec.frame
        self._btn(f, "📊  Export Excel",              self._export_excel,      fpx, variant="secondary", h=28)
        self._btn(f, "📄  Export RR CSV  (Ctrl+Shift+E)", self._export_rr_csv, fpx, variant="secondary", h=28)
        self._btn(f, "🖼  Export Figures  (PNG)",      self._export_figures,    fpx, variant="secondary", h=28)
        self._btn(f, "📦  Export ZIP  (Excel+Figs)",  self._export_zip,        fpx, variant="secondary", h=28)
        self._btn(f, "📄  PDF Report  (1 page)",      self._export_pdf_report, fpx, variant="secondary", h=28)
        self._btn(f, "🔬  Export Abnormal Events PDF", self._export_arrhythmia_pdf, fpx, variant="secondary", h=28)
        self._btn(f, "🔬  Export GraphPad Prism",     self._export_prism,      fpx, variant="secondary", h=28)

    def _build_stat_section(self: ECGApp, parent) -> None:
        """Categorized statistics: Heart Rate / RR Intervals / HRV / Signal /
        Quality, as a vertically-stacked accordion section in the right
        panel. All values are still driven by update_kpis()
        (plot_controller.py), fanned out to tiles via make_stat_tile()
        (ecg/ui/widgets.py) exactly as before. Tiles are arranged 2-per-row
        within each category box (rather than one full-width column) to keep
        this section compact now that EXPORTS and ML DETECTOR SAVING also
        need to fit below it in the same panel.
        """
        sec = CollapsibleSection(parent, "STATISTICS", initially_open=True)

        # (key, label, unit, hero) -- unit is baked into the tile once at
        # construction (static), hero=True gets the one FONT_KPI_HERO use.
        categories = [
            ("Heart Rate", ORANGE_DARK, [
                ("hr_mean",  "HR Mean",  "bpm", True),
                ("hr_range", "HR Range", "",    False),
            ]),
            ("RR Intervals", BLUE_MID, [
                ("rr_mean", "Mean RR", "ms", False),
                ("n_beats", "RR intervals (n)", "",   False),
            ]),
            ("HRV", TEAL, [
                ("sdnn",  "SDNN",  "ms", False),
                ("rmssd", "RMSSD", "ms", False),
                ("pnn6",  "pNN6",  "%",  False),
            ]),
            ("Signal", BLUE_DEEP, [
                ("dur",     "Analysed duration",     "s", False),
                ("sq_corr", "Beat-shape match (r)",  "",  False),
            ]),
            ("Quality", GREEN_DARK, [
                ("sq_score",    "Overall score",              "%", False),
                ("sq_badbeats", "Poor-shape-match beats (r<0.90)", "",  False),
                ("sq_noisy_time", "Poor-shape-match time",     "%", False),
                ("sq_artifact", "Auto-corrected",              "", False),
            ]),
        ]

        for cat_name, accent, tiles in categories:
            box = ctk.CTkFrame(sec.frame, fg_color=CARD, corner_radius=8,
                                border_width=1, border_color=BORDER)
            box.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_S))
            ctk.CTkFrame(box, fg_color=accent, height=3, corner_radius=2).pack(
                fill="x", padx=SPACE_S, pady=(SPACE_XS, 0))
            ctk.CTkLabel(box, text=cat_name.upper(), font=FONT_MICRO,
                         text_color=MUTED, anchor="w").pack(
                fill="x", padx=SPACE_S, pady=(SPACE_XS, SPACE_XS))
            # 2-per-row: chunk tiles into pairs, each pair sharing a row frame.
            # An odd tile out (HRV/Quality have 3) ends up alone on its row.
            for i in range(0, len(tiles), 2):
                row = ctk.CTkFrame(box, fg_color="transparent")
                row.pack(fill="x", padx=SPACE_S, pady=(0, SPACE_XS))
                for key, label, unit, hero in tiles[i:i + 2]:
                    tile = make_stat_tile(row, label, "—", unit=unit, hero=hero, card=False)
                    tile.pack(side="left", fill="both", expand=True, padx=(0, SPACE_XS))
                    # The right panel is much narrower than the old full-width
                    # strip -- long values (e.g. sq_artifact's "N (a dup · b
                    # non-physio · c ectopic)" breakdown) need to wrap instead
                    # of clipping against the tile's now-halved width.
                    tile.value_label.configure(wraplength=95, justify="left")
                    self._kpi[key] = tile.value_label
                    if key == "n_beats":
                        # This count is n peaks - 1 (consecutive RR
                        # intervals), minus any removed as artifacts -- it
                        # will not match "Peaks detected" in the sidebar.
                        self._bind_hover_tip(
                            tile, "RR intervals = detected peaks − 1, minus "
                                  "any intervals removed as artifacts.")

    def _build_annotations_section(self: ECGApp, parent) -> None:
        """Compact ANNOTATIONS section: live counts + buttons that open the
        existing AnnotationManagerDialog/PacingPeriodManagerDialog popups
        unchanged. Kept alongside (not instead of) the Detection tab's own
        toolbar chips -- those are Detection-tab-only, this is visible from
        every tab.
        """
        sec = CollapsibleSection(parent, "ANNOTATIONS", initially_open=False)
        f = sec.frame
        fpx = dict(padx=SPACE_L)

        # Initial text set directly from current data (not via
        # _update_ann_count()/_update_pacing_count()) -- those also touch
        # self.lbl_ann_count (the Detection tab's toolbar badge), which
        # during _rebuild_ui() isn't rebuilt yet at this point in _build()'s
        # sequence (_build_tabs() runs after _build_right_panel()) and would
        # still be the just-destroyed pre-rebuild widget. _rebuild_ui()
        # refreshes both surfaces itself once _build() has fully completed.
        self.lbl_panel_ann_count = ctk.CTkLabel(
            f, text=f"{len(self._annotations)} annotations",
            font=FONT_LABEL, text_color=TEXT, anchor="w")
        self.lbl_panel_ann_count.pack(**fpx, fill="x", pady=(SPACE_S, SPACE_XS))
        self._btn(f, "📍  Manage Annotations…", self._open_annotations, fpx,
                  variant="secondary", h=28)

        ctk.CTkFrame(f, height=1, fg_color=BORDER).pack(
            fill="x", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))

        self.lbl_panel_pacing_count = ctk.CTkLabel(
            f, text=f"{len(self._pacing_periods)} pacing periods",
            font=FONT_LABEL, text_color=TEXT, anchor="w")
        self.lbl_panel_pacing_count.pack(**fpx, fill="x", pady=(SPACE_S, SPACE_XS))
        self._btn(f, "⏱  Manage Pacing Periods…", self._open_pacing_periods, fpx,
                  variant="secondary", h=28)
