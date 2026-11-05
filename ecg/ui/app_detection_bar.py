# -*- coding: utf-8 -*-
"""
ecg.ui.app_detection_bar
------------------------
The detection bar at the top of Signal & Detection: method, threshold (slider
and exact value, always in view), Min / Max RR, then chips for the Filters
popover, the annotations and the pacing periods, and the method help.

Every widget keeps the attribute name it had in the left sidebar
(cb_det_method, sl_thr, ent_thr, lbl_thr, lbl_thr_hint, ent_minrr, ent_maxrr,
sw_filtering, sw_notch, ent_lp, ent_hp, cb_clean, sw_show_raw,
sw_invert_signal, sw_filter_preview, ent_sg_target_fs, ent_sg_window_ms...):
FilterParams.from_widgets, the session save / restore and the controllers
read and write them unchanged, the popovers being built hidden at startup.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core.models import MouseECG
from ecg.ui.popover import Popover
from ecg.ui.theme import (
    BG, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT, RED, BLUE, ORANGE,
    SPACE_XS, SPACE_S, SPACE_M, FONT_LABEL, FONT_SMALL, FONT_KPI_LABEL, FONT_HINT,
    FONT_SUBSECTION, FONT_BTN_SEC, FONT_CARD_TITLE,
)
from ecg.ui.theme import BLUE, BLUE_HOVER
from ecg.ui.theme import THEME, card_style, chip_style, CARD_PAD, PAGE_GAP
from ecg.ui.theme import combo_style, input_style
from ecg.ui.theme import button_style

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

_HINT_W = 250        # wraplength of the popovers' hints
_DEFAULT_THR_HINT = ("Fraction of the typical (median) R-peak height — 0.50 "
                     "keeps peaks ≥ 50% as tall. Lower = more peaks.")
_MIN_RR_TIP = "Min RR: peaks closer together than this are merged into one beat"
_MAX_RR_TIP = ("Max RR: HRV uses only RR intervals between Min and Max — "
               "widen Max for anaesthetised mice (< 180 bpm)")
_METHOD_HELP = ("SG + Derivative: downsample → Savitzky-Golay derivative "
                "(its options: Filters chip)\n"
                "Wavelet: CWT separates noise / QRS / J-wave (needs PyWavelets)\n"
                "Envelope Max: local maxima — best for clipped / saturated signals\n"
                "ML Detector: trained on your verified recordings "
                "(status and training: Signal quality panel)\n"
                "Auto: NeuroKit2's own detector")


class DetectionBarMixin:
    """ECGApp detection bar, Filters popover and method help."""

    def _build_detection_bar(self: ECGApp, parent) -> None:
        # The mockup's detection card: 12 px above and below, 16 px inside
        # the sides, 14 px between controls.
        bar = ctk.CTkFrame(parent, **card_style())
        bar.pack(side="top", fill="x", pady=(0, PAGE_GAP))
        g = ctk.CTkFrame(bar, fg_color="transparent")
        g.pack(fill="x", padx=CARD_PAD, pady=(10, 12))
        g.grid_columnconfigure(1, weight=1)

        def caption(col: int, text: str) -> ctk.CTkLabel:
            lbl = ctk.CTkLabel(g, text=text, font=FONT_KPI_LABEL, text_color=MUTED,
                               anchor="w", height=16)
            lbl.grid(row=0, column=col, sticky="w", padx=(0, 14), pady=(0, 4))
            return lbl

        # ── Method ───────────────────────────────────────────────
        caption(0, "Method")
        self.cb_det_method = ctk.CTkComboBox(
            g,
            width=int(210 * THEME.font_scale),
            values=["Auto (NeuroKit2)", "SG + Derivative (10 kHz)", "Wavelet (CWT)", "Envelope Max", "ML Detector"],
            command=self._on_det_method_change,
            **combo_style(),
        )
        self.cb_det_method.set("SG + Derivative (10 kHz)")
        self.cb_det_method.grid(row=1, column=0, sticky="w", padx=(0, SPACE_M))
        # ML status lives in the Signal quality panel's ML section now, shown
        # with Method = ML Detector; no second copy beside the combobox.
        self.lbl_ml_inline_status = None

        # ── Threshold: slider + exact value, always in view ──────
        thr_cap = ctk.CTkFrame(g, fg_color="transparent")
        thr_cap.grid(row=0, column=1, sticky="ew", padx=(0, SPACE_M))
        self.lbl_thr = ctk.CTkLabel(thr_cap, text="Threshold:  0.50", font=FONT_SMALL,
                                    text_color=TEXT, anchor="w", height=16)
        self.lbl_thr.pack(side="left")
        ctk.CTkLabel(thr_cap, text="of the median R height · strict ↑ / sensitive ↓",
                     font=FONT_KPI_LABEL, text_color=LIGHT, height=16).pack(side="left",
                                                                           padx=(SPACE_S, 0))
        thr_row = ctk.CTkFrame(g, fg_color="transparent")
        thr_row.grid(row=1, column=1, sticky="ew", padx=(0, SPACE_M))
        self.sl_thr = ctk.CTkSlider(
            thr_row, from_=0.01, to=2.0,  # type: ignore
            progress_color=BLUE, button_color=BLUE,
            button_hover_color=BLUE_HOVER, fg_color=BORDER,
            height=20, command=self._on_threshold_slide)
        self.sl_thr.set(0.50)
        self.sl_thr.pack(side="left", fill="x", expand=True, padx=(0, SPACE_S))
        self.ent_thr = ctk.CTkEntry(thr_row, width=64, **input_style())
        self.ent_thr.insert(0, "0.50")
        self.ent_thr.pack(side="left")
        self.ent_thr.bind("<Return>",   self._on_threshold_entry)
        self.ent_thr.bind("<FocusOut>", self._on_threshold_entry)
        # The live "Current: … → ≈… norm." readout of the last detection
        # (detection_controller) sits in the method help; the slider and the
        # entry show it on hover.
        for w in (self.sl_thr, self.ent_thr):
            self._bind_hover_tip(w, lambda: self.lbl_thr_hint.cget("text"))

        # ── Min / Max RR ─────────────────────────────────────────
        # With Min RR, the RR range accepted for HRV: intervals outside it
        # are excluded (never bridged) by every HRV computation. The default
        # Max (≈ 333 ms = 180 bpm) assumes an awake mouse.
        for col, label, attr, default, tip in (
                (2, "Min RR (ms)", "minrr", f"{MouseECG.MIN_RR_MS:g}", _MIN_RR_TIP),
                (3, "Max RR (ms)", "maxrr", f"{MouseECG.RR_MAX_MS:g}", _MAX_RR_TIP)):
            caption(col, label)
            ent = ctk.CTkEntry(g, width=72, **input_style())
            ent.insert(0, default)
            ent.grid(row=1, column=col, sticky="w", padx=(0, SPACE_M))
            setattr(self, f"ent_{attr}", ent)
            self._bind_hover_tip(ent, tip)

        # ── Chips: Filters / Annotations / Pacing, then the help ─
        chips = ctk.CTkFrame(g, fg_color="transparent")
        chips.grid(row=1, column=4, sticky="e")
        self.btn_filters_chip = self._chip(chips, "Filters: off", self._toggle_filters_popover)
        self._bind_hover_tip(self.btn_filters_chip,
                             "Band-pass, notch, cleaning, invert, raw / filtered view, "
                             "filter preview, Savitzky-Golay options")
        self.btn_annotations = self._chip(chips, f"Annotations {len(self._annotations)}",
                                          self._open_annotations)
        self.btn_pacing = self._chip(chips, f"Pacing {len(self._pacing_periods)}",
                                     self._open_pacing_periods)
        self._bind_hover_tip(self.btn_pacing, "Pacing / stimulation periods")
        h = button_style()["height"]
        self.btn_method_help = ctk.CTkButton(
            chips,
            text="?",
            width=h,
            command=self._toggle_method_help,
            **button_style("secondary", font=FONT_CARD_TITLE, corner_radius=h // 2),
        )
        self.btn_method_help.pack(side="left")
        self._bind_hover_tip(self.btn_method_help, "Detection methods and threshold")

        self._build_filters_popover()
        self._build_method_help()

    def _chip(self: ECGApp, parent, text: str, command) -> ctk.CTkButton:
        """A rounded outline button of the detection bar (the mockup's chip)."""
        btn = ctk.CTkButton(parent, text=text, width=0, command=command, **chip_style())
        btn.pack(side="left", padx=(0, SPACE_M))
        return btn

    # ── Filters popover ──────────────────────────────────────────

    def _build_filters_popover(self: ECGApp) -> None:
        """sw_filtering ON = DSP filtering is applied (band-pass/notch/
        cleaning); OFF (default) = raw signal, matching the app's
        raw-by-default behaviour on file load. DISPLAY & PREVIEW controls
        (show raw, invert, live preview) always stay interactive -- pure
        display toggles, or apply regardless of sw_filtering. FILTER
        SETTINGS (notch/band-pass/cleaning) stay visible but greyed out when
        filtering is off, so the user can see/set them in advance -- see
        on_filtering_toggle in detection_controller.py."""
        self.filters_popover = Popover(self, title="Filters")
        body = self.filters_popover.body
        fpx = dict(padx=0)
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="y", anchor="n", padx=(0, SPACE_M))
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.pack(side="left", fill="y", anchor="n")

        def head(parent, text: str) -> None:
            ctk.CTkLabel(parent, text=text, font=FONT_SUBSECTION, text_color=MUTED,
                         anchor="w").pack(fill="x", pady=(0, SPACE_XS))

        def hint(parent, text: str) -> None:
            ctk.CTkLabel(parent, text=text, font=FONT_HINT, text_color=MUTED, anchor="w",
                         justify="left", wraplength=_HINT_W).pack(fill="x", pady=(0, SPACE_S))

        # ── Left: what is analysed ───────────────────────────────
        head(left, "PROCESSING")
        self.sw_filtering = self._switch(left, "Filtering", fpx, default_on=False)
        self.sw_filtering.configure(command=self._on_filtering_toggle)
        hint(left, "Off analyzes the raw signal. On applies the settings below.")
        self.lbl_filter_summary = ctk.CTkLabel(
            left, text="Processing\n•  Raw signal", font=FONT_HINT, text_color=MUTED,
            anchor="w", justify="left", wraplength=_HINT_W)
        self.lbl_filter_summary.pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkFrame(left, height=1, fg_color=BORDER).pack(fill="x", pady=(0, SPACE_XS))
        head(left, "FILTER SETTINGS")
        self._filter_advanced_group = ctk.CTkFrame(left, fg_color="transparent")
        self._filter_advanced_group.pack(fill="x")
        grp = self._filter_advanced_group
        self.sw_notch = self._switch(grp, "Notch 50 Hz", fpx, default_on=False)
        self.sw_notch.configure(command=self._refresh_filter_preview)
        ctk.CTkLabel(grp, text="Band-pass filter", font=FONT_SMALL,
                     text_color=MUTED, anchor="w").pack(anchor="w", pady=(SPACE_XS, 0))
        # Labelled by frequency position (Low/High), matching filtering.py's
        # own bandpass(lo, hi) naming -- not by filter-type jargon (HP/LP).
        self._sidebar_entry_row(grp, fpx, [
            ("Low cutoff (Hz)",  "lp", str(MouseECG.BP_LO_HZ)),
            ("High cutoff (Hz)", "hp", str(int(MouseECG.BP_HI_HZ))),
        ])
        for _ent in (self.ent_lp, self.ent_hp):
            if _ent is not None:
                _ent.bind("<Return>",   lambda _e: self._refresh_filter_preview())
                _ent.bind("<FocusOut>", lambda _e: self._refresh_filter_preview())
        ctk.CTkLabel(grp, text="Signal cleaning:", font=FONT_SMALL,
                     text_color=MUTED, anchor="w").pack(anchor="w", pady=(SPACE_XS, 0))
        self.cb_clean = ctk.CTkComboBox(
            grp,
            width=_HINT_W,
            values=["neurokit", "pantompkins1985", "elgendi2010", "hamilton2002", "biosppy"],
            command=lambda _v: self._refresh_filter_preview(),
            **combo_style(),
        )
        self.cb_clean.set("neurokit")
        self.cb_clean.pack(fill="x", pady=(SPACE_XS, 0))

        # ── Right: what is shown, and the SG detector's options ──
        head(right, "DISPLAY & PREVIEW")
        self.sw_show_raw = self._switch(right, "Show raw signal (vs filtered)", fpx, default_on=True)
        self.sw_show_raw.configure(command=self._on_show_raw_toggle)
        self.sw_invert_signal = self._switch(right, "⟳  Invert signal (polarity)", fpx,
                                             default_on=False)
        # A plain parameter -- doesn't auto-run the full pipeline on toggle.
        # Picked up on the next Detect Peaks, or immediately in the
        # before/after overlay if "Preview filtering" is on.
        self.sw_invert_signal.configure(command=self._refresh_filter_preview)
        hint(right, "Useful if R peaks appear negative")
        self.sw_filter_preview = self._switch(right, "👁  Preview filtering", fpx, default_on=False)
        self.sw_filter_preview.configure(command=self._on_filter_preview_toggle)
        hint(right, "Overlays the filtered signal on the visible window, using the "
                    "settings on the left — no need to click Detect Peaks.")
        # Savitzky-Golay options: shown with Method = SG + Derivative only
        # (on_det_method_change); the holder keeps their place in the column.
        sg_holder = ctk.CTkFrame(right, fg_color="transparent")
        sg_holder.pack(fill="x")
        self._sg_frame = ctk.CTkFrame(sg_holder, fg_color="transparent")
        ctk.CTkFrame(self._sg_frame, height=1, fg_color=BORDER).pack(fill="x", pady=(0, SPACE_XS))
        head(self._sg_frame, "SAVITZKY-GOLAY OPTIONS")
        self._sidebar_entry_row(self._sg_frame, fpx, [
            ("SG target fs (Hz)", "sg_target_fs", "10000"),
            ("SG window (ms)", "sg_window_ms",  "20"),
        ])
        hint(self._sg_frame, "Signal is resampled to this rate, then differentiated; "
                             "window should span about one QRS complex")
        self._refresh_filters_chip()

    def _toggle_filters_popover(self: ECGApp) -> None:
        self.filters_popover.toggle(self.btn_filters_chip)

    def _refresh_filters_chip(self: ECGApp) -> None:
        """'Filters: off' or what is applied ('Filters: 1–150 Hz · notch'),
        plus an inverted polarity: both change what is analysed. Called with
        every refresh of the Processing summary."""
        chip = getattr(self, "btn_filters_chip", None)
        if chip is None or self.sw_filtering is None:
            return
        parts = []
        if bool(self.sw_filtering.get()):
            lp = self._safe_float(self.ent_lp, MouseECG.BP_LO_HZ)
            hp = self._safe_float(self.ent_hp, MouseECG.BP_HI_HZ)
            parts.append(f"{lp:g}–{hp:g} Hz")
            if self.sw_notch is not None and bool(self.sw_notch.get()):
                parts.append("notch")
        if self.sw_invert_signal is not None and bool(self.sw_invert_signal.get()):
            parts.append("inverted")
        text = "Filters: " + (" · ".join(parts) if parts else "off")
        on = bool(parts)
        if chip.cget("text") != text:
            chip.configure(text=text, **chip_style(active=on))

    # ── Method help ──────────────────────────────────────────────

    def _build_method_help(self: ECGApp) -> None:
        self.method_help = Popover(self, title="Detection")
        b = self.method_help.body
        for text in (_METHOD_HELP,):
            ctk.CTkLabel(b, text=text, font=FONT_HINT, text_color=TEXT, anchor="w",
                         justify="left", wraplength=2 * _HINT_W).pack(fill="x", pady=(0, SPACE_S))
        ctk.CTkFrame(b, height=1, fg_color=BORDER).pack(fill="x", pady=(0, SPACE_S))
        self.lbl_thr_hint = ctk.CTkLabel(
            b, text=_DEFAULT_THR_HINT, font=FONT_HINT, text_color=MUTED, anchor="w",
            justify="left", wraplength=2 * _HINT_W)
        self.lbl_thr_hint.pack(fill="x", pady=(0, SPACE_S))
        for text in (_MIN_RR_TIP, _MAX_RR_TIP):
            ctk.CTkLabel(b, text=text, font=FONT_HINT, text_color=MUTED, anchor="w",
                         justify="left", wraplength=2 * _HINT_W).pack(fill="x")

    def _toggle_method_help(self: ECGApp) -> None:
        self.method_help.toggle(self.btn_method_help)

    # ── Annotation / pacing chips ────────────────────────────────

    def _refresh_annotation_chips(self: ECGApp) -> None:
        """Counts on the Annotations and Pacing chips, orange when non-zero."""
        for btn, word, n in ((getattr(self, "btn_annotations", None), "Annotations",
                              len(self._annotations)),
                             (getattr(self, "btn_pacing", None), "Pacing",
                              len(self._pacing_periods))):
            if btn is None:
                continue
            btn.configure(text=f"{word} {n}", text_color=ORANGE if n else TEXT,
                          border_color=ORANGE if n else BORDER2)
