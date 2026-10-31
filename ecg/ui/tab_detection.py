# -*- coding: utf-8 -*-
"""
ecg.ui.tab_detection
--------------------
ECGApp Signal & Detection tab construction, with the signal-quality panel.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from ecg.ui.theme import (
    PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, PLOT, BLUE, ORANGE, BLUE_HOVER,
    GREEN, RED, PURPLE, PURPLE_DARK, SPACE_XS, SPACE_S, SPACE_M, SPACE_L,
    FONT_LABEL, FONT_SMALL, FONT_MONO, FONT_SIDEBAR_HDR, FONT_HINT, FONT_CARD_TITLE,
    FONT_KPI_HERO, FONT_KPI_LABEL,
)
from ecg.core.detection import detect_rr_artifacts
from ecg.core.models import MouseECG
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_DETECTION

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

QUALITY_PANEL_W = 290


log = logging.getLogger("ecg")


class TabDetectionMixin:
    """ECGApp Signal & Detection tab construction."""

    _artifact_count_seen: Any = None     # ((id, n peaks), n candidates)
    _artifact_count_busy = False

    def _build_tab_detection(self: ECGApp) -> None:
        t = self.tabs.tab(TAB_DETECTION)

        # Layout (top → bottom, all pack):
        #   nav bar     (fill=x, no expand) — time navigation controls
        #   thin separator
        #   detail toolbar (fill=x, no expand)
        #   detail plot + RR strip, signal-quality panel on their right

        # ── Navigation bar ────────────────────────────────────────────────
        # Instrument-style layout: 3 grouped chips inside the flat PANEL bar
        # -- Transport | Position & Zoom | Readout (docked right).
        nav = ctk.CTkFrame(t, fg_color=PANEL, corner_radius=0, height=34)
        nav.pack(side="top", fill="x")
        nav.pack_propagate(False)

        # ── Transport chip: jump/step controls ──────────────────────────
        transport_chip = self._toolbar_chip(nav)
        transport_chip.pack(side="left", padx=(SPACE_M, SPACE_S))

        btn_nav_reset = ctk.CTkButton(
            transport_chip, text="⏮", width=36, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=self._nav_reset)
        btn_nav_reset.pack(side="left", padx=(SPACE_S, SPACE_XS))
        self._bind_hover_tip(btn_nav_reset, "Jump to start")

        btn_nav_big_back = ctk.CTkButton(
            transport_chip, text="◀◀", width=44, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=lambda: self._navigate_big(-1))
        btn_nav_big_back.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(btn_nav_big_back, "Big step back")

        btn_nav_back = ctk.CTkButton(
            transport_chip, text="◀", width=36, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=lambda: self._navigate(-1))
        btn_nav_back.pack(side="left", padx=(0, SPACE_M))
        self._bind_hover_tip(btn_nav_back, "Step back")

        btn_nav_fwd = ctk.CTkButton(
            transport_chip, text="▶", width=36, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=lambda: self._navigate(+1))
        btn_nav_fwd.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(btn_nav_fwd, "Step forward")

        btn_nav_big_fwd = ctk.CTkButton(
            transport_chip, text="▶▶", width=44, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=lambda: self._navigate_big(+1))
        btn_nav_big_fwd.pack(side="left", padx=(0, SPACE_XS))
        self._bind_hover_tip(btn_nav_big_fwd, "Big step forward")

        btn_nav_end = ctk.CTkButton(
            transport_chip, text="⏭", width=36, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=self._nav_end)
        btn_nav_end.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(btn_nav_end, "Jump to end")

        # ── Position & Zoom chip: current position + window size ────────
        pos_chip = self._toolbar_chip(nav)
        pos_chip.pack(side="left", padx=(0, SPACE_S))

        lbl_nav_pos = ctk.CTkLabel(pos_chip, text="t =", font=FONT_SMALL,
                     text_color=MUTED)
        lbl_nav_pos.pack(side="left", padx=(SPACE_S, SPACE_XS))
        self.ent_nav_pos = ctk.CTkEntry(
            pos_chip, width=72, height=28, font=FONT_LABEL,
            fg_color=CARD, border_color=BORDER2, text_color=TEXT,
            corner_radius=6,
            placeholder_text="0.000")
        self.ent_nav_pos.pack(side="left", padx=(0, SPACE_XS))  # type: ignore[union-attr]
        ctk.CTkLabel(pos_chip, text="s", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        _pos_tip = "Visible window start time — type a value and press Go (or Enter) to jump there"
        self._bind_hover_tip(lbl_nav_pos, _pos_tip)
        self._bind_hover_tip(self.ent_nav_pos, _pos_tip)
        ctk.CTkButton(pos_chip, text="Go", width=48, height=28, font=FONT_SMALL,
                      fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
                      corner_radius=8,
                      command=self._nav_goto).pack(side="left", padx=(SPACE_S, SPACE_L))
        # Bind Enter key on the position field
        self.ent_nav_pos.bind("<Return>", lambda _e: self._nav_goto())  # type: ignore[union-attr]

        ctk.CTkLabel(pos_chip, text="Window:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(0, SPACE_S))
        self.ent_window = ctk.CTkEntry(pos_chip, width=48, height=28, font=FONT_LABEL,
                                       fg_color=CARD, border_color=BORDER2, text_color=TEXT,
                                       corner_radius=6)
        self.ent_window.insert(0, "2")
        self.ent_window.pack(side="left", padx=(0, SPACE_XS))
        self._bind_hover_tip(self.ent_window, "Visible window duration (s)")
        # Redraw on Enter/blur, matching ent_nav_pos's <Return> binding above --
        # otherwise a typed window size only takes effect once the user also
        # clicks Go or an arrow button.
        self.ent_window.bind("<Return>", lambda _e: self._draw_detail())
        self.ent_window.bind("<FocusOut>", lambda _e: self._draw_detail())
        ctk.CTkLabel(pos_chip, text="s", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(0, SPACE_S))

        # ── Spike stepper chip: walk the HRV tab's flagged-beat list ────
        # Previously the only way to work through several suspicious beats
        # was to right-click each one on the Rhythm & HRV tachogram; this
        # walks the same app.ui.rr_spike_times list (populated by
        # PlotController.plot_rr()) without leaving the Detection tab.
        spike_chip = self._toolbar_chip(nav)
        spike_chip.pack(side="left", padx=(0, SPACE_S))
        ctk.CTkLabel(spike_chip, text="Spike:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(SPACE_S, SPACE_XS))
        btn_spike_prev = ctk.CTkButton(
            spike_chip, text="◀", width=28, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=lambda: self._step_spike(-1))
        btn_spike_prev.pack(side="left", padx=(0, SPACE_XS))
        self._bind_hover_tip(btn_spike_prev, "Previous suspicious beat")
        btn_spike_next = ctk.CTkButton(
            spike_chip, text="▶", width=28, height=28, font=FONT_LABEL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8, command=lambda: self._step_spike(+1))
        btn_spike_next.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(btn_spike_next, "Next suspicious beat")

        # ── Readout chip: duration, docked to the bar's right edge ──────
        readout_chip = self._toolbar_chip(nav)
        readout_chip.pack(side="right", padx=(SPACE_S, SPACE_M))
        self.lbl_sig_duration = ctk.CTkLabel(
            readout_chip, text="", font=FONT_MONO, text_color=MUTED, anchor="w")
        self.lbl_sig_duration.pack(side="left", padx=SPACE_S)  # type: ignore[union-attr]

        # Separator
        tk.Frame(t, height=1, bg=BORDER).pack(side="top", fill="x", padx=SPACE_M, pady=SPACE_XS)

        # Detail toolbar (fixed height)
        hdr = ctk.CTkFrame(t, fg_color="transparent", height=32)
        hdr.pack(side="top", fill="x", padx=SPACE_M, pady=(SPACE_XS, SPACE_XS))
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr, text="ECG SIGNAL", font=FONT_SIDEBAR_HDR,
                     text_color=MUTED).pack(side="left", anchor="w")

        ctk.CTkFrame(hdr, width=1, fg_color=BORDER).pack(side="left", fill="y", padx=(SPACE_L, SPACE_S), pady=SPACE_XS)

        # ── Edit tools chip ───────────────────────────────────────────────
        edit_chip = self._toolbar_chip(hdr)
        edit_chip.pack(side="left", padx=(0, SPACE_M))
        self.btn_edit_mode = ctk.CTkButton(
            edit_chip, text="Edit Peaks", width=96, height=28, font=FONT_SMALL,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            corner_radius=8,
            command=self._toggle_edit_mode,
        )
        self.btn_edit_mode.pack(side="left", padx=(SPACE_S, SPACE_XS))
        self.lbl_edit_hint = ctk.CTkLabel(
            edit_chip,
            text="L-click: exclude/restore   R-click: add   Ctrl+Z: undo",
            font=FONT_HINT, text_color=ORANGE,
        )
        # Undo/Redo used to be duplicated here (and again on the Abnormal
        # Events tab) alongside the toolbar's own pair -- three widgets for
        # one shared edit_undo/edit_redo stack. The toolbar pair (always
        # visible, works from any tab) is now the only one.
        self.btn_clear_excl = ctk.CTkButton(
            edit_chip, text="Clear Edits", width=100, height=28, font=FONT_SMALL,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            corner_radius=8,
            command=self._clear_manual_exclusions,
        )
        self.btn_clear_excl.pack(side="left", padx=(SPACE_S, SPACE_S))

        # ── Free Placement chip (bypass proximity guard) ──────────────────
        fp_chip = self._toolbar_chip(hdr)
        fp_chip.pack(side="left", padx=(0, SPACE_M))
        self.btn_free_placement = ctk.CTkButton(
            fp_chip, text="Free Placement", width=118, height=28, font=FONT_SMALL,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
            corner_radius=8,
            command=self._toggle_free_placement,
        )
        self.btn_free_placement.pack(side="left", padx=(SPACE_S, SPACE_XS))
        _fp_tip = ctk.CTkLabel(
            fp_chip, text="?", width=28, height=28,
            font=FONT_CARD_TITLE, text_color=MUTED,
            fg_color=BORDER, corner_radius=14,
        )
        _fp_tip.pack(side="left", padx=(0, SPACE_S))
        self._bind_hover_tip(
            _fp_tip,
            "Free Placement: right-click adds a peak at the exact clicked position — "
            "no snapping, no proximity guard, works even on top of existing peaks.")

        # ── Annotations chip ───────────────────────────────────────────────
        ann_chip = self._toolbar_chip(hdr)
        ann_chip.pack(side="left", padx=(0, SPACE_M))
        self.btn_annotations = ctk.CTkButton(
            ann_chip, text="Annotations", width=110, height=28, font=FONT_SMALL,
            fg_color=PURPLE_DARK, hover_color=PURPLE, text_color="white",
            corner_radius=8,
            command=self._open_annotations,
        )
        self.btn_annotations.pack(side="left", padx=(SPACE_S, SPACE_XS))  # type: ignore[union-attr]
        self.lbl_ann_count = ctk.CTkLabel(
            ann_chip, text="", font=FONT_HINT, text_color=MUTED, anchor="w")
        self.lbl_ann_count.pack(side="left", padx=(0, SPACE_S))  # type: ignore[union-attr]

        # ── RR/HR/Quality strip toggle ───────────────────────────────────────
        # "Pacing Periods" chip removed here (redundant with the ANNOTATIONS
        # panel section's "Manage Pacing Periods…" button, the sole remaining
        # entry point) -- this slot now hosts the show/hide toggle for the
        # RR/HR/Quality sub-plot strip below the detail plot.
        rrhr_chip = self._toolbar_chip(hdr)
        rrhr_chip.pack(side="left")
        _rrhr_on = self.ui.rrhr_strip_visible
        self.btn_toggle_rrhr = ctk.CTkButton(
            rrhr_chip, text="RR/HR/Quality", width=118, height=28, font=FONT_SMALL,
            # "On" look is an outline, not a solid fill -- a solid BLUE chip
            # here is indistinguishable from a primary command button like
            # "Detect Peaks", even though this is a toggle, not an action.
            fg_color=PANEL if _rrhr_on else BORDER,
            hover_color=BORDER2,
            text_color=BLUE if _rrhr_on else MUTED,
            border_width=2 if _rrhr_on else 0,
            border_color=BLUE,
            corner_radius=8,
            command=self._toggle_rrhr_strip,
        )
        self.btn_toggle_rrhr.pack(side="left", padx=(SPACE_S, SPACE_S))  # type: ignore[union-attr]

        # Full-recording scrubber strip, directly above the detail plot
        # (Audacity/Premiere-style position indicator). Fixed height
        # (pack_propagate(False), mirroring nav/hdr) rather than fill=both.
        # The RR of the whole recording + draggable viewport cursor (see draw_overview()).
        ov_frame = tk.Frame(t, bg=PLOT["bg"], bd=0, highlightthickness=0, height=72)
        ov_frame.pack(side="top", fill="x", padx=SPACE_XS, pady=(0, SPACE_XS))
        ov_frame.pack_propagate(False)
        self._slots["overview"] = CanvasSlot(ov_frame, 14, 0.5, toolbar=False)
        self._slots["overview"].canvas.mpl_connect(
            "button_press_event", self._on_overview_click)
        self._slots["overview"].canvas.mpl_connect(
            "motion_notify_event", self._on_overview_motion)
        self._slots["overview"].canvas.mpl_connect(
            "button_release_event", self._on_overview_release)
        # The matplotlib canvas covers ov_frame entirely, so the hover-tip
        # must bind to its Tk widget directly -- Enter/Leave don't bubble
        # from a covering child to an ancestor frame in Tkinter.
        self._bind_hover_tip(
            self._slots["overview"].canvas.get_tk_widget(),
            "Wheel: zoom · drag strip: scrub")

        # Plot area — detail trace (~70%) + synced RR/HR strip (~30%),
        # grid-managed so both share the remaining space proportionally
        # (same grid_rowconfigure(weight=N) idiom already used by
        # _build_hrv_view_rr's RR-tachogram/stats split).
        self.plot_area = ctk.CTkFrame(t, fg_color="transparent")
        self.plot_area.pack(side="top", fill="both", expand=True)
        self.plot_area.grid_rowconfigure(0, weight=7)
        self.plot_area.grid_rowconfigure(1, weight=3 if self.ui.rrhr_strip_visible else 0)
        self.plot_area.grid_columnconfigure(0, weight=1)
        self._build_quality_panel(self.plot_area)

        det_frame = tk.Frame(self.plot_area, bg=PLOT["bg"], bd=0, highlightthickness=0)
        det_frame.grid(row=0, column=0, sticky="nsew", padx=SPACE_XS, pady=(0, SPACE_XS))
        self._slots["detail"] = CanvasSlot(det_frame, 12, 6.5, toolbar=True)

        self._slots["detail"].canvas.mpl_connect(
            "scroll_event", self._on_detail_scroll)
        self._slots["detail"].canvas.mpl_connect(
            "button_press_event", self._on_detail_click)
        self._hover_motion_cid = self._slots["detail"].canvas.mpl_connect(
            "motion_notify_event", self._on_detail_motion)

        # RR/HR strip — synced to the same nav window as the detail plot,
        # via ECGApp._draw_detail()'s fan-out (see draw_detail_rrhr()).
        # Show/hide is toggled by btn_toggle_rrhr in the hdr toolbar above.
        self.subplot_frame = tk.Frame(self.plot_area, bg=PLOT["bg"], bd=0, highlightthickness=0)
        self.subplot_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_XS, pady=(0, SPACE_XS))
        if not self.ui.rrhr_strip_visible:
            self.subplot_frame.grid_remove()
        self._slots["detail_rrhr"] = CanvasSlot(self.subplot_frame, 12, 2.8, toolbar=False)

    def _build_quality_panel(self: ECGApp, parent) -> None:
        """Quality belongs next to the signal it judges. Values come from
        plot_quality_panel() once core results exist; the score and the
        detector agreement from _refresh_quality_live()."""
        panel = ctk.CTkFrame(parent, width=QUALITY_PANEL_W, fg_color=CARD, corner_radius=6)
        panel.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(0, SPACE_XS), pady=(0, SPACE_XS))
        panel.pack_propagate(False)

        ctk.CTkLabel(panel, text="SIGNAL QUALITY", font=FONT_SIDEBAR_HDR, text_color=MUTED,
                     anchor="w").pack(fill="x", padx=SPACE_M, pady=(SPACE_S, 0))

        score_row = ctk.CTkFrame(panel, fg_color="transparent")
        score_row.pack(fill="x", padx=SPACE_M)
        self._quality_vals = {}
        self._quality_vals["sq_score"] = ctk.CTkLabel(
            score_row, text="—", font=FONT_KPI_HERO, text_color=TEXT, anchor="w")
        self._quality_vals["sq_score"].pack(side="left")
        self.lbl_quality_word = ctk.CTkLabel(
            score_row, text="", font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_quality_word.pack(side="left", padx=(SPACE_S, 0))

        grid = ctk.CTkFrame(panel, fg_color="transparent")
        grid.pack(fill="x", padx=SPACE_M)
        grid.grid_columnconfigure((0, 1), weight=1, uniform="q")
        for i, (key, label) in enumerate((
                ("sq_corr", "Template correlation"),
                ("sq_noisy_time", "Time below 0.90"),
                ("sq_badbeats", "Beats below 0.90"),
                ("sq_agree", "Detectors agree"),
                ("sq_artifact", "Auto-corrected"))):
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=i // 2, column=i % 2, columnspan=2 if key == "sq_artifact" else 1,
                      sticky="ew", pady=(0, SPACE_XS))
            ctk.CTkLabel(cell, text=label, font=FONT_KPI_LABEL, text_color=MUTED,
                         anchor="w", height=14).pack(anchor="w")
            val = ctk.CTkLabel(cell, text="—", font=FONT_LABEL, text_color=TEXT, anchor="w",
                               height=18, wraplength=QUALITY_PANEL_W - 2 * SPACE_M, justify="left")
            val.pack(anchor="w")
            self._quality_vals[key] = val
        self._bind_hover_tip(
            self._quality_vals["sq_agree"],
            "Beats confirmed by at least one other detector, after Check Agreement "
            "(right panel, Detection section)")

        # Bottom first: the two plots share what is left.
        self.btn_review_quality = ctk.CTkButton(
            panel, text="Review artifacts", command=self._open_artifact_review,
            fg_color=BORDER, hover_color=BORDER2, text_color=MUTED, font=FONT_SMALL,
            height=28, corner_radius=6, state="disabled")
        self.btn_review_quality.pack(side="bottom", fill="x", padx=SPACE_M, pady=SPACE_S)

        plots = ctk.CTkFrame(panel, fg_color="transparent")
        plots.pack(side="top", fill="both", expand=True, padx=SPACE_S)
        plots.grid_columnconfigure(0, weight=1)
        for row, (key, title) in enumerate((("quality_time", "Template correlation over time"),
                                            ("quality_hist", "Template correlation histogram"))):
            ctk.CTkLabel(plots, text=title, font=FONT_KPI_LABEL, text_color=MUTED, anchor="w",
                         height=14).grid(row=2 * row, column=0, sticky="ew", padx=SPACE_XS)
            plots.grid_rowconfigure(2 * row + 1, weight=1)
            box = tk.Frame(plots, bg=PLOT["bg"], bd=0, highlightthickness=0, height=60)
            box.grid(row=2 * row + 1, column=0, sticky="nsew", pady=(0, SPACE_XS))
            box.pack_propagate(False)
            self._slots[key] = CanvasSlot(box, 2.6, 1.0, toolbar=False)

    def _artifact_candidate_count(self: ECGApp, rp) -> int:
        """How many candidates Review would list on these peaks: the same call as
        _open_artifact_review, read only, on its own thread so the tick stays cheap.
        0 until it is known."""
        if rp is None or self._signal_flt is None or len(rp) < 4:
            return 0
        key = (id(rp), len(rp))
        seen = self._artifact_count_seen
        if seen is not None and seen[0] == key:
            return seen[1]
        if not self._artifact_count_busy and not getattr(self, "_async_busy", False):
            self._artifact_count_busy = True
            args = (rp.copy(), self._fs, float(self._safe_float(self.ent_minrr, MouseECG.RR_MIN_MS)),
                    self.analysis_ctrl.hrv_rr_bounds(quiet=True)[1], self._signal_flt)

            def work() -> None:
                try:
                    found = detect_rr_artifacts(args[0], args[1], rr_min_ms=args[2], rr_max_ms=args[3],
                                                window_beats=11, dev_threshold=0.20, signal=args[4])
                    self._artifact_count_seen = (key, len(found))
                except Exception as exc:  # the button only loses its count
                    log.debug("artifact count failed: %s", exc)
                    self._artifact_count_seen = (key, 0)
                finally:
                    self._artifact_count_busy = False

            threading.Thread(target=work, daemon=True).start()
        return seen[1] if seen is not None else 0

    def _refresh_quality_live(self: ECGApp) -> None:
        """What follows the peaks rather than the results: the score, the last
        Check Agreement while still on these peaks, the review button."""
        det = self.detection
        score = det.sig_quality
        vals = self._quality_vals
        if score is None:
            text, word, color = "—", "", TEXT
        else:
            word = "good" if score >= 70 else ("fair" if score >= 40 else "poor")
            color = GREEN if score >= 70 else (ORANGE if score >= 40 else RED)
            text, word = f"{score}", f"/ 100 · {word}"
        if vals["sq_score"].cget("text") != text or self.lbl_quality_word.cget("text") != word:
            vals["sq_score"].configure(text=text, text_color=color)
            self.lbl_quality_word.configure(text=word)

        rp = det.rpeaks_ok
        agree = "—"
        if (det.agreement_pct is not None and rp is not None
                and self.analysis_ctrl._same_peaks(det.agreement_peaks, rp)):
            agree = f"{det.agreement_pct:.1f}%"
        if vals["sq_agree"].cget("text") != agree:
            vals["sq_agree"].configure(text=agree)

        n_cand = self._artifact_candidate_count(rp)
        label = (f"Review {n_cand} artifact candidate{'s' if n_cand > 1 else ''}"
                 if n_cand else "Review artifacts")
        btn = self.btn_review_quality
        if btn.cget("text") != label:
            btn.configure(text=label)
        enabled = rp is not None and len(rp) > 4
        if (btn.cget("state") == "normal") != enabled:
            self._set_btn_enabled(btn, enabled, BLUE, BLUE_HOVER)
