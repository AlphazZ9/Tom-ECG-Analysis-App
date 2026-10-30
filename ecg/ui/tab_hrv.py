# -*- coding: utf-8 -*-
"""
ecg.ui.tab_hrv
--------------
ECGApp HRV tab: sub-view builders and their handlers.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from ecg.core.models import MouseECG
from ecg.core.autonomic import COV_METHODS
from ecg.ui.autonomic_view import (
    CALIBRATION_WARNING, MIN_PAIRS_MAX, MIN_PAIRS_MIN, WINDOW_MAX_S,
    WINDOW_MIN_S, format_param, normalise_autonomic_params,
)
from ecg.ui.event_locked_view import (
    PRE_POST_MAX_S, PRE_POST_MIN_S, annotation_label_options,
    normalise_event_params,
)
from ecg.ui.theme import (
    THEME, BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, BLUE, GREEN, ORANGE,
    BLUE_HOVER, BLUE_DEEP, PURPLE, PURPLE_DARK, GREEN_DARK, SPACE_XS, SPACE_S,
    SPACE_M, SPACE_L, FONT_SECTION_HDR, FONT_LABEL, FONT_SMALL, FONT_BODY,
    FONT_MONO, FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR, FONT_HINT,
)
from ecg.ui.plots import CanvasSlot
from ecg.ui.tab_keys import TAB_HRV

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class TabHrvMixin:
    """ECGApp HRV tab: sub-view builders and their handlers."""

    def _build_tab_hrv_unified(self: ECGApp) -> None:
        """Unified HRV tab with internal segmented navigation.

        Merges: RR / HR  |  Temporel  |  Fréquentiel  |  Non-linéaire  |  Epochs  |  Glissant
        |  Autonomic (time-resolved CSI / CPI)
        """
        t = self.tabs.tab(TAB_HRV)
        t.grid_rowconfigure(1, weight=1)
        t.grid_columnconfigure(0, weight=1)

        # ── Top: segmented sub-tab selector ───────────────────────────────
        _HRV_VIEWS = ["RR / HR", "Time Domain", "Frequency", "Non-linear", "Epochs", "Rolling",
                      "Autonomic"]
        self._hrv_subframes: "dict[str, ctk.CTkFrame]" = {}

        seg_bar = ctk.CTkFrame(t, fg_color=PANEL, corner_radius=0)
        seg_bar.grid(row=0, column=0, sticky="ew")
        self._hrv_seg = ctk.CTkSegmentedButton(
            seg_bar,
            values=_HRV_VIEWS,
            font=FONT_SMALL,
            # Matches the main CTkTabview's convention just above it (plain
            # background, pill only on the active item) instead of its own
            # grey-boxed-every-item look -- see self.tabs in _build_tabs.
            fg_color=PANEL,
            selected_color=BLUE,
            selected_hover_color=BLUE_HOVER,
            unselected_color=CARD,
            unselected_hover_color=BORDER,
            text_color=TEXT,
            text_color_disabled=MUTED,
            command=self._on_hrv_view_change,
        )
        self._hrv_seg.pack(padx=SPACE_M, pady=SPACE_S, anchor="w")  # type: ignore[union-attr]
        self._hrv_seg.set("RR / HR")  # type: ignore[union-attr]

        # ── Content area — one frame per view, pack/pack_forget ───────────
        content_area = ctk.CTkFrame(t, fg_color="transparent")
        content_area.grid(row=1, column=0, sticky="nsew")
        content_area.grid_rowconfigure(0, weight=1)
        content_area.grid_columnconfigure(0, weight=1)
        self._hrv_content_area = content_area

        for view in _HRV_VIEWS:
            f = ctk.CTkFrame(content_area, fg_color="transparent")
            f.grid_rowconfigure(0, weight=1)
            f.grid_columnconfigure(0, weight=1)
            self._hrv_subframes[view] = f

        # ── Build each sub-view content ───────────────────────────────────
        self._build_hrv_view_rr(self._hrv_subframes["RR / HR"])
        self._build_hrv_view_temporel(self._hrv_subframes["Time Domain"])
        self._build_hrv_view_freq(self._hrv_subframes["Frequency"])
        self._build_hrv_view_nonlin(self._hrv_subframes["Non-linear"])
        self._build_hrv_view_epochs(self._hrv_subframes["Epochs"])
        self._build_hrv_view_rolling(self._hrv_subframes["Rolling"])
        self._build_hrv_view_autonomic(self._hrv_subframes["Autonomic"])

        # Show the view in use before a UI rebuild (theme toggle), else RR / HR:
        # ui.hrv_current_view is app state, like ui.autonomic_subview.
        view = (self.ui.hrv_current_view if self.ui.hrv_current_view in self._hrv_subframes
                else "RR / HR")
        self._hrv_seg.set(view)  # type: ignore[union-attr]
        self._hrv_subframes[view].pack(fill="both", expand=True)
        self._hrv_current_view = view

    def _on_hrv_view_change(self: ECGApp, view: str) -> None:
        """Switch the visible HRV sub-view."""
        if not hasattr(self, "_hrv_subframes"):
            return
        for name, frame in self._hrv_subframes.items():
            if name == view:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        self._hrv_current_view = view
        if view == "Epochs":
            # Surface a too-short-recording warning as soon as the view is
            # shown, rather than only after a failed Compute-epochs click.
            self._refresh_epoch_feasibility()

    def _build_hrv_view_rr(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """RR tachogram + histogram — formerly the 'RR / HR' tab."""
        # Ratio 4:1 (était 3:2) — les deux tachogrammes empilés (RR + HR)
        # sont le contenu principal de l'onglet et bénéficient d'un maximum
        # de hauteur ; le tableau de stats + histogramme du bas est compact
        # et n'a pas besoin d'autant d'espace que ce que 2/5 lui donnait.
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=4)
        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        spike_chip = self._toolbar_chip(parent)
        spike_chip.grid(row=0, column=0, sticky="w", padx=SPACE_S, pady=(SPACE_S, 0))
        _spk_on = self.ui.show_spike_markers
        self.btn_toggle_spikes = ctk.CTkButton(
            spike_chip, text="▲▼ Spike markers", width=140, height=26, font=FONT_SMALL,
            # Outline "on" look, not a solid fill -- see btn_toggle_rrhr above
            # for why a solid BLUE toggle reads as a command button instead.
            fg_color=PANEL if _spk_on else BORDER,
            hover_color=BORDER2,
            text_color=BLUE if _spk_on else MUTED,
            border_width=2 if _spk_on else 0,
            border_color=BLUE,
            corner_radius=8,
            command=self._toggle_spike_markers,
        )
        self.btn_toggle_spikes.pack(side="left", padx=SPACE_XS, pady=SPACE_XS)  # type: ignore[union-attr]

        # "Review" -- jumps to the next flagged beat (▲/▼ triangle) on this
        # same tachogram, in Detection, centred in the signal view. Same
        # step_spike() the Detection tab's own "Spike: ◀ ▶" stepper and this
        # plot's right-click-nearest-spike behaviour already use -- this
        # button just makes that reachable from the HRV tab itself, and
        # pressing it repeatedly walks every flagged beat in turn (wraps
        # back to the first once past the last).
        self.btn_review_spikes = ctk.CTkButton(
            spike_chip, text="🔍 Review", width=100, height=26, font=FONT_SMALL,
            fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
            corner_radius=8,
            command=lambda: self._step_spike(+1),
        )
        self.btn_review_spikes.pack(side="left", padx=SPACE_XS, pady=SPACE_XS)  # type: ignore[union-attr]
        self._bind_hover_tip(
            self.btn_review_spikes,
            "Jump to the next flagged beat (▲/▼) — press again to step "
            "through the rest")

        rr_frame = ctk.CTkFrame(parent, fg_color="transparent")
        rr_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_S, pady=(SPACE_S, SPACE_XS))
        rr_frame.grid_rowconfigure(0, weight=1)
        rr_frame.grid_columnconfigure(0, weight=1)
        self._slots["rr"] = CanvasSlot(rr_frame, 14, 5.0, toolbar=False)

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid(row=2, column=0, sticky="nsew", padx=SPACE_M, pady=(SPACE_XS, SPACE_S))
        row.grid_rowconfigure(0, weight=1)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=2)

        # Stats textbox (narrow)
        stats_card = ctk.CTkFrame(row, fg_color=PANEL, corner_radius=0)
        stats_card.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_S))
        stats_card.grid_rowconfigure(0, weight=0)
        stats_card.grid_rowconfigure(1, weight=1)
        stats_card.grid_columnconfigure(0, weight=1)
        _rr_btn = ctk.CTkButton(stats_card, text="📋 Copy for Excel", height=22,
                                font=FONT_HINT, fg_color=BORDER, hover_color=BORDER2,
                                text_color=MUTED, corner_radius=5,
                                command=lambda: self._copy_tsv(self.txt_rr))
        _rr_btn.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, 0))
        self.txt_rr = ctk.CTkTextbox(stats_card, font=FONT_MONO, fg_color=PANEL,
                                     text_color=TEXT, border_width=0)
        self.txt_rr.grid(row=1, column=0, sticky="nsew", padx=SPACE_M, pady=(SPACE_XS, SPACE_S))

        # Histogram
        hist_card = ctk.CTkFrame(row, fg_color=PANEL, corner_radius=0)
        hist_card.grid(row=0, column=1, sticky="nsew")
        hist_card.grid_rowconfigure(0, weight=1)
        hist_card.grid_columnconfigure(0, weight=1)
        self._slots["rr_hist"] = CanvasSlot(hist_card, 7, 4.5, toolbar=False)

    def _build_hrv_view_temporel(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Time-domain HRV — text tables (computed in Core Analysis)."""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=6)
        inner.grid(row=0, column=0, sticky="nsew", padx=SPACE_M, pady=SPACE_M)
        inner.grid_rowconfigure(1, weight=1)
        inner.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(inner, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_M, SPACE_XS))
        ctk.CTkLabel(hdr, text="HRV TIME DOMAIN", font=FONT_SIDEBAR_HDR,
                     text_color=MUTED).pack(side="left")
        ctk.CTkButton(hdr, text="📋 Copy for Excel", height=22,
                      font=FONT_HINT, fg_color=BORDER, hover_color=BORDER2,
                      text_color=MUTED, corner_radius=5, width=140,
                      command=lambda: self._copy_tsv(self.txt_td)
                      ).pack(side="right")
        self.txt_td = ctk.CTkTextbox(inner, font=FONT_MONO, fg_color=PANEL,
                                     text_color=TEXT, border_width=0)
        self.txt_td.grid(row=1, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_M))

    def _build_hrv_view_freq(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Frequency-domain HRV — PSD + tables."""
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Action bar
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        self.btn_run_freq = ctk.CTkButton(
            bar, text="⚡  Freq HRV",
            command=self._run_freq,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_PRIMARY, height=max(28, int(32 * THEME.font_scale)),
            corner_radius=5,
        )
        self.btn_run_freq.pack(side="left")  # type: ignore[union-attr]
        self.lbl_freq_status = ctk.CTkLabel(
            bar, text="  Click Analyze first",
            font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_freq_status.pack(side="left", padx=SPACE_M)  # type: ignore[union-attr]

        # Content: text left, PSD right
        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=SPACE_S, pady=(SPACE_XS, SPACE_S))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)

        left = ctk.CTkFrame(content, fg_color=PANEL, corner_radius=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_S))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        _hdr = ctk.CTkFrame(left, fg_color="transparent")
        _hdr.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        ctk.CTkLabel(_hdr, text="FREQUENCY DOMAIN", font=FONT_SIDEBAR_HDR,
                     text_color=MUTED).pack(side="left")
        ctk.CTkButton(_hdr, text="📋 Copy for Excel", height=20,
                      font=FONT_HINT, fg_color=BORDER, hover_color=BORDER2,
                      text_color=MUTED, corner_radius=5, width=140,
                      command=lambda: self._copy_tsv(self.txt_fd)
                      ).pack(side="right")
        self.txt_fd = ctk.CTkTextbox(left, font=FONT_MONO, fg_color=PANEL,
                                     text_color=TEXT, border_width=0)
        self.txt_fd.grid(row=1, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_S))

        right = ctk.CTkFrame(content, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        card = ctk.CTkFrame(right, fg_color=PANEL, corner_radius=0)
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)
        self._slots["psd"] = CanvasSlot(card, 8, 4.5, toolbar=False)

    def _build_hrv_view_nonlin(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Non-linear HRV — Poincaré + SampEn/DFA table."""
        parent.grid_rowconfigure(0, weight=0)
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        self.btn_run_nonlin = ctk.CTkButton(
            bar, text="⚡  Non-linear HRV",
            command=self._run_nonlinear,
            fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
            font=FONT_BTN_PRIMARY, height=max(28, int(32 * THEME.font_scale)),
            corner_radius=5,
        )
        self.btn_run_nonlin.pack(side="left")  # type: ignore[union-attr]
        # Runtime cap of the O(N^2) entropy block only (SampEn, ApEn, MSE ...):
        # Poincare and DFA always use every beat of the analysis window.
        ctk.CTkLabel(bar, text="Entropy cap (RR):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(SPACE_M, 4))
        self.ent_entropy_cap = ctk.CTkEntry(bar, width=64, height=26, font=FONT_LABEL,
                                            fg_color=BG, border_color=BORDER2,
                                            text_color=TEXT)
        self.ent_entropy_cap.insert(0, str(MouseECG.ENTROPY_MAX_BEATS))  # type: ignore[union-attr]
        self.ent_entropy_cap.pack(side="left")  # type: ignore[union-attr]
        self.lbl_nonlin_status = ctk.CTkLabel(
            bar, text="  SampEn + DFA can take 30 s+ on long recordings",
            font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_nonlin_status.pack(side="left", padx=SPACE_M)  # type: ignore[union-attr]

        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=SPACE_S, pady=(SPACE_XS, SPACE_S))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)

        left = ctk.CTkFrame(content, fg_color=PANEL, corner_radius=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_S))
        left.grid_rowconfigure(0, weight=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        _nl_hdr = ctk.CTkFrame(left, fg_color="transparent")
        _nl_hdr.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, SPACE_XS))
        ctk.CTkLabel(_nl_hdr, text="NON-LINEAR", font=FONT_SIDEBAR_HDR,
                     text_color=MUTED).pack(side="left")
        ctk.CTkButton(_nl_hdr, text="📋 Copy for Excel", height=20,
                      font=FONT_HINT, fg_color=BORDER, hover_color=BORDER2,
                      text_color=MUTED, corner_radius=5, width=140,
                      command=lambda: self._copy_tsv(self.txt_nl)
                      ).pack(side="right")
        self.txt_nl = ctk.CTkTextbox(left, font=FONT_MONO, fg_color=PANEL,
                                     text_color=TEXT, border_width=0)
        self.txt_nl.grid(row=1, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_S))

        right = ctk.CTkFrame(content, fg_color=PANEL, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self._slots["poincare"] = CanvasSlot(
            right, 8, 8, toolbar=False,
            placeholder="Click ⚡ Non-linear HRV to display")

    def _build_hrv_view_epochs(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Epoch HRV analysis — formerly the 'Epochs' tab."""
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_M, SPACE_S))
        ctk.CTkLabel(hdr, text="HRV PER EPOCH",
                     font=FONT_SECTION_HDR, text_color=MUTED).pack(side="left")
        # Left-docked, like the compute/run buttons on every other HRV tab
        # (Freq HRV, Non-linear HRV, Measure intervals, Classify Events) --
        # this used to be the only one on the right.
        self.btn_compute_epochs = ctk.CTkButton(
            hdr, text="⟳  Compute epochs", command=self._compute_epochs,
            fg_color=BLUE, hover_color=BLUE_DEEP, text_color="white",
            font=FONT_SMALL, height=28, corner_radius=5)
        self.btn_compute_epochs.pack(side="left", padx=(SPACE_M, 0))
        self.lbl_epoch_count = ctk.CTkLabel(
            hdr, text="", font=FONT_SMALL, text_color=BLUE)
        self.lbl_epoch_count.pack(side="right", padx=(0, SPACE_M))

        # Settings row
        settings_row = ctk.CTkFrame(parent, fg_color="transparent")
        settings_row.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        for lbl_t, attr_n, default in [
            ("Epoch (s)", "epoch", str(int(MouseECG.EPOCH_DEFAULT_S))),
            ("Overlap (s)", "overlap", "0"),
        ]:
            ctk.CTkLabel(settings_row, text=lbl_t, font=FONT_SMALL,
                         text_color=MUTED).pack(side="left", padx=(0, SPACE_S))
            ent = ctk.CTkEntry(settings_row, width=58, height=26,
                               font=FONT_LABEL, fg_color=BG,
                               border_color=BORDER2, text_color=TEXT)
            ent.insert(0, default)
            ent.pack(side="left", padx=(0, SPACE_L))
            setattr(self, f"ent_{attr_n}", ent)
        self.sw_epoch = self._switch(settings_row, "Auto-run after analysis",
                                     dict(padx=0))
        self.sw_epoch.pack(side="left")
        self.lbl_epoch_info = ctk.CTkLabel(
            settings_row, text="", font=FONT_HINT, text_color=MUTED)
        self.lbl_epoch_info.pack(side="right", padx=(0, SPACE_S))

        parent.grid_rowconfigure(1, weight=0)
        parent.grid_rowconfigure(2, weight=0)
        parent.grid_rowconfigure(3, weight=0)
        parent.grid_rowconfigure(4, weight=1)

        # copy bar
        ep_bar = ctk.CTkFrame(parent, fg_color="transparent")
        ep_bar.grid(row=2, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_S, 0))
        ctk.CTkLabel(ep_bar, text="Epoch table:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        self.btn_copy_epochs = ctk.CTkButton(
            ep_bar, text="📋  Copy for Excel", height=24,
            font=FONT_SMALL, fg_color=BORDER, hover_color=BORDER2,
            text_color=MUTED, corner_radius=5, width=160,
            command=lambda: self._copy_tsv(self.txt_epochs))
        self.btn_copy_epochs.pack(side="right", padx=(0, SPACE_XS))  # type: ignore[union-attr]

        ep_sf = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0)
        ep_sf.grid(row=3, column=0, sticky="nsew", padx=SPACE_M, pady=(SPACE_S, SPACE_S))
        ep_sf.grid_rowconfigure(0, weight=1)
        ep_sf.grid_columnconfigure(0, weight=1)
        self._slots["epochs"] = CanvasSlot(
            ep_sf, 14, 5.5, toolbar=False,
            placeholder="Click ⟳ Compute epochs to display")

        epochs_tb = ctk.CTkTextbox(parent, font=FONT_BODY, fg_color="transparent",
                                   text_color=TEXT, border_width=0,
                                   scrollbar_button_color=BORDER,
                                   scrollbar_button_hover_color=BORDER2,
                                   height=160)
        epochs_tb.grid(row=4, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        self.txt_epochs = epochs_tb

    def _build_hrv_view_rolling(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Rolling / sliding-window HRV — formerly the 'Rolling HRV' tab."""
        parent.grid_columnconfigure(0, weight=1)

        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_M, SPACE_S))

        ctk.CTkLabel(bar, text="Window (s):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        self.ent_roll_win = ctk.CTkEntry(
            bar, width=58, height=28, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, text_color=TEXT)
        self.ent_roll_win.insert(0, "30")
        self.ent_roll_win.pack(side="left", padx=(SPACE_S, SPACE_L))

        ctk.CTkLabel(bar, text="Step (s):", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        self.ent_roll_step = ctk.CTkEntry(
            bar, width=48, height=28, font=FONT_LABEL,
            fg_color=BG, border_color=BORDER2, text_color=TEXT)
        self.ent_roll_step.insert(0, "5")
        self.ent_roll_step.pack(side="left", padx=(SPACE_S, SPACE_L))
        self._bind_hover_tip(
            self.ent_roll_win,
            "Minimum 3 s. For the full article method (time-resolved CSI / CPI "
            "of Candia-Rivera et al. 2025, one window per beat) use HRV > Autonomic.")

        self.btn_roll_compute = ctk.CTkButton(
            bar, text="⟳  Compute", command=self._compute_rolling_hrv,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_SEC, height=28, corner_radius=5)
        self.btn_roll_compute.pack(side="left", padx=(SPACE_L, 0))

        self.lbl_roll_status = ctk.CTkLabel(
            bar, text="  Click Analyze first",
            font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_roll_status.pack(side="left", padx=SPACE_M)  # type: ignore[union-attr]

        # Metrics checkboxes get their own row -- HR/SDNN/RMSSD/pNN6 plus
        # SD1/SD2/LF_nu/HF_nu/LF_HF (task: surface compute_segment_stats'
        # richer metric set as continuous trends, not just inside the
        # Compare Segments modal) no longer fit the single window/step/
        # compute row without wrapping badly.
        metrics_bar = ctk.CTkFrame(parent, fg_color="transparent")
        metrics_bar.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        ctk.CTkLabel(metrics_bar, text="Metrics:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        self._roll_metrics: "dict[str, ctk.CTkCheckBox]" = {}
        # Display text only -- the dict key below (matched against
        # analysis_controller.compute_rolling_hrv's column names) stays the
        # raw metric name regardless of label.
        _roll_metric_labels = {"LF_nu": "LF (n.u.)", "HF_nu": "HF (n.u.)", "LF_HF": "LF/HF"}
        for metric, default in [("HR", True), ("SDNN", True), ("RMSSD", True),
                                 ("pNN6", False), ("SD1", False), ("SD2", False),
                                 ("LF_nu", False), ("HF_nu", False), ("LF_HF", False)]:
            cb = ctk.CTkCheckBox(metrics_bar, text=_roll_metric_labels.get(metric, metric),
                                 font=FONT_SMALL,
                                 text_color=MUTED, fg_color=BLUE,
                                 checkmark_color="white",
                                 border_color=BORDER2, width=16)
            if default:
                cb.select()
            cb.pack(side="left", padx=(SPACE_S, 0))
            self._roll_metrics[metric] = cb

        article_bar = ctk.CTkFrame(parent, fg_color="transparent")
        article_bar.grid(row=2, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        ctk.CTkLabel(article_bar, text="Article-style Poincare (LW):",
                     font=FONT_SMALL, text_color=MUTED).pack(side="left")
        # "(LW)" in the texts: these use sklearn's Ledoit-Wolf shrinkage, not
        # the paper's estimator -- the full article method is HRV > Autonomic.
        article_metric_labels = {
            "CCD": "CCD",
            "SD1_LW": "SD1 (LW)",
            "SD2_LW": "SD2 (LW)",
        }
        for metric in ("CCD", "SD1_LW", "SD2_LW"):
            cb = ctk.CTkCheckBox(
                article_bar, text=article_metric_labels[metric],
                font=FONT_SMALL, text_color=MUTED, fg_color=BLUE,
                checkmark_color="white", border_color=BORDER2, width=16)
            cb.pack(side="left", padx=(SPACE_S, 0))
            self._roll_metrics[metric] = cb
        ctk.CTkLabel(
            article_bar,
            text=("(LW = Ledoit–Wolf; ≥10 successive valid RR pairs — for the full "
                  "article method, CSI / CPI per beat, see HRV > Autonomic)"),
            font=FONT_HINT, text_color=MUTED,
        ).pack(side="left", padx=(SPACE_S, 0))

        parent.grid_rowconfigure(1, weight=0)
        parent.grid_rowconfigure(2, weight=0)
        parent.grid_rowconfigure(3, weight=0)
        parent.grid_rowconfigure(4, weight=1)
        parent.grid_rowconfigure(5, weight=0)

        # copy bar
        roll_bar = ctk.CTkFrame(parent, fg_color="transparent")
        roll_bar.grid(row=3, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        ctk.CTkLabel(roll_bar, text="Window table:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left")
        self.btn_copy_rolling = ctk.CTkButton(
            roll_bar, text="📋  Copy for Excel", height=24,
            font=FONT_SMALL, fg_color=BORDER, hover_color=BORDER2,
            text_color=MUTED, corner_radius=5, width=160,
            command=lambda: self._copy_tsv(self.txt_rolling))
        self.btn_copy_rolling.pack(side="right", padx=(0, SPACE_XS))

        plot_frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0)
        plot_frame.grid(row=4, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_S))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self._slots["rolling_hrv"] = CanvasSlot(
            plot_frame, 14, 6.0, toolbar=False,
            placeholder="Click ⟳ Compute to display")

        rolling_tb = ctk.CTkTextbox(parent, font=FONT_BODY, fg_color="transparent",
                                    text_color=TEXT, border_width=0,
                                    scrollbar_button_color=BORDER,
                                    scrollbar_button_hover_color=BORDER2,
                                    height=160)
        rolling_tb.grid(row=5, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        self.txt_rolling = rolling_tb

    def _build_hrv_view_autonomic(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """Time-resolved cardiac sympathetic / parasympathetic indices (CSI /
        CPI) from sliding-window Poincaré plots -- Candia-Rivera et al.,
        J Physiol 2025 (core.autonomic) -- with optional SPWVD LF / HF
        (core.spwvd). Logic in AnalysisController.compute_autonomic, pure
        helpers in ecg.ui.autonomic_view. The parameters shown are app state
        (analysis.autonomic_params), so a rebuild keeps them."""
        parent.grid_columnconfigure(0, weight=1)
        for row, weight in ((0, 0), (1, 0), (2, 0), (3, 0), (4, 1)):
            parent.grid_rowconfigure(row, weight=weight)
        p = normalise_autonomic_params(self.analysis.autonomic_params)

        # ── Row 0: parameters + Compute ──
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(SPACE_M, SPACE_XS))

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
            bar, text="LF/HF (SPWVD)", font=FONT_SMALL, text_color=MUTED,
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
            bar, text="⟳  Compute", command=self._compute_autonomic,
            fg_color=BLUE, hover_color=BLUE_HOVER, text_color="white",
            font=FONT_BTN_SEC, height=28, corner_radius=5)
        self.btn_auto_compute.pack(side="left")
        self._bind_hover_tip(
            self.btn_auto_compute,
            "Compute CSI / CPI on the analysis window's peaks, with the sidebar "
            "Min / Max RR bounds: intervals outside them or spanning a removed "
            "beat are excluded, never bridged. Runs in the background.")

        # ── Row 1: status ──
        self.lbl_auto_status = ctk.CTkLabel(
            parent, text=self.analysis_ctrl.AUTONOMIC_IDLE_TEXT,
            font=FONT_SMALL, text_color=MUTED, anchor="w")
        self.lbl_auto_status.grid(row=1, column=0, sticky="ew", padx=SPACE_M)

        # ── Row 2: persistent calibration warning ──
        ctk.CTkLabel(
            parent, text="⚠  " + CALIBRATION_WARNING, font=FONT_HINT,
            text_color=ORANGE, anchor="w", justify="left",
            wraplength=int(1100 * THEME.font_scale),
        ).grid(row=2, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))

        # ── Row 3: sub-view toggle (time series | event-locked) ──
        seg_bar = ctk.CTkFrame(parent, fg_color="transparent")
        seg_bar.grid(row=3, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_S))
        self._auto_seg = ctk.CTkSegmentedButton(
            seg_bar, values=list(self._AUTO_SUBVIEWS), font=FONT_SMALL,
            fg_color=PANEL, selected_color=BLUE, selected_hover_color=BLUE_HOVER,
            unselected_color=CARD, unselected_hover_color=BORDER,
            text_color=TEXT, text_color_disabled=MUTED,
            command=self._on_autonomic_subview_change)
        self._auto_seg.pack(side="left")  # type: ignore[union-attr]
        ctk.CTkLabel(
            seg_bar, text=("Event-locked: pre / post means around the annotations, "
                           "cohort registry and ks / kp calibration"),
            font=FONT_HINT, text_color=MUTED,
        ).pack(side="left", padx=(SPACE_L, 0))

        # ── Row 4: one content frame per sub-view ──
        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.grid(row=4, column=0, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        self._auto_subframes = {}
        for name in self._AUTO_SUBVIEWS:
            f = ctk.CTkFrame(content, fg_color="transparent")
            f.grid(row=0, column=0, sticky="nsew")
            self._auto_subframes[name] = f
        self._build_autonomic_series_panel(self._auto_subframes["Time series"])
        self._build_autonomic_event_panel(self._auto_subframes["Event-locked"])
        view = (self.ui.autonomic_subview if self.ui.autonomic_subview in self._auto_subframes
                else self._AUTO_SUBVIEWS[0])
        self._auto_seg.set(view)  # type: ignore[union-attr]
        self._show_autonomic_subview(view)

    _AUTO_SUBVIEWS = ("Time series", "Event-locked")

    def _build_autonomic_series_panel(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """HRV > Autonomic > Time series: the stacked CSI / CPI figure and
        the recording-level values."""
        parent.grid_columnconfigure(0, weight=1)
        for row, weight in ((0, 1), (1, 0), (2, 0)):
            parent.grid_rowconfigure(row, weight=weight)

        # ── Row 0: stacked plot ──
        plot_frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_S))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self._slots["autonomic"] = CanvasSlot(
            plot_frame, 14, 7.0, toolbar=True,
            placeholder="Click ⟳ Compute to display")

        # ── Rows 1-2: recording-level values ──
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))
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
            scrollbar_button_hover_color=BORDER2, height=170)
        self.txt_autonomic.grid(row=2, column=0, sticky="ew", padx=SPACE_M,
                                pady=(0, SPACE_S))

    def _build_autonomic_event_panel(self: ECGApp, parent: ctk.CTkFrame) -> None:
        """HRV > Autonomic > Event-locked: per-event 30 s pre / post means of
        CSI, CPI, HR, D, SD1, SD2 (and LF / HF) around the annotations, the
        median ± MAD traces, the cohort registry and the ks / kp
        calibration. Logic in EventLockedController, pure helpers in
        ecg.ui.event_locked_view. The parameters shown are app state
        (analysis.event_locked_params), so a rebuild keeps them."""
        parent.grid_columnconfigure(0, weight=1)
        for row, weight in ((0, 0), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0)):
            parent.grid_rowconfigure(row, weight=weight)
        p = normalise_event_params(self.analysis.event_locked_params)

        # ── Row 0: events, windows, Compute, z-score ──
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))
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

        # ── Row 1: registry / export / calibration ──
        bar2 = ctk.CTkFrame(parent, fg_color="transparent")
        bar2.grid(row=1, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))
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
        btn_cal = ctk.CTkButton(
            bar2, text="⚖  Calibrate ks/kp…", command=self._open_calibration,
            fg_color=PURPLE, hover_color=PURPLE_DARK, text_color="white",
            font=FONT_BTN_SEC, height=26, corner_radius=5)
        btn_cal.pack(side="left")
        self._bind_hover_tip(
            btn_cal,
            "Cohort calibration of the gains from the registry: Wilcoxon Z(k) "
            "of CSI and CPI for k = 1–20 and its knee (the paper's rule), per "
            "event or per subject mean.")

        # ── Row 2: status ──
        self.lbl_evt_status = ctk.CTkLabel(
            parent, text=self.event_ctrl.IDLE_TEXT, font=FONT_SMALL,
            text_color=MUTED, anchor="w")
        self.lbl_evt_status.grid(row=2, column=0, sticky="ew", padx=SPACE_M)

        # ── Row 3: event-locked traces ──
        plot_frame = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=0)
        plot_frame.grid(row=3, column=0, sticky="nsew", padx=SPACE_M, pady=(0, SPACE_S))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self._slots["event_locked"] = CanvasSlot(
            plot_frame, 14, 6.0, toolbar=True,
            placeholder="Compute the Autonomic indices, then ⟳ Compute event-locked")

        # ── Rows 4-5: per-event table ──
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.grid(row=4, column=0, sticky="ew", padx=SPACE_M, pady=(0, SPACE_XS))
        ctk.CTkLabel(hdr, text="PER-EVENT TABLE", font=FONT_SIDEBAR_HDR,
                     text_color=MUTED).pack(side="left")
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
        self.txt_event_locked.grid(row=5, column=0, sticky="ew", padx=SPACE_M,
                                   pady=(0, SPACE_S))

    def _on_autonomic_subview_change(self: ECGApp, view: str) -> None:
        """HRV > Autonomic sub-view toggle."""
        self.ui.autonomic_subview = view
        self._show_autonomic_subview(view)
        if view == "Event-locked":
            # The annotations may have been replaced without an edit (new
            # file, session restore): list their current labels.
            self.event_ctrl.refresh_label_choices()

    def _show_autonomic_subview(self: ECGApp, view: str) -> None:
        for name, frame in self._auto_subframes.items():
            if name == view:
                frame.grid()
            else:
                frame.grid_remove()

    def _compute_autonomic(self: ECGApp) -> None:
        """Compute the time-resolved CSI / CPI and draw the Autonomic view."""
        self.analysis_ctrl.compute_autonomic()

    def _on_autonomic_method_change(self: ECGApp, choice: str) -> None:
        self.analysis_ctrl.on_autonomic_method_change(choice)

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

    # ════════════════════════════════════════════════════════
    #  ROLLING HRV
    # ════════════════════════════════════════════════════════

    def _compute_rolling_hrv(self: ECGApp) -> None:
        """Compute sliding-window HRV and render the timeline plot."""
        self.analysis_ctrl.compute_rolling_hrv()
