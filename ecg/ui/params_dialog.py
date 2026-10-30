# -*- coding: utf-8 -*-
"""
ecg.ui.params_dialog
--------------------
ECGApp advanced-parameters dialog.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
from ecg.ui.theme import (
    BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT, RED, BLUE, GREEN,
    ORANGE, BLUE_DARK, PURPLE, GREEN_DARK, ORANGE_DARK, SPACE_XS, SPACE_S,
    SPACE_M, SPACE_L, FONT_LABEL, FONT_SMALL, FONT_KPI_LABEL, FONT_BTN_PRIMARY,
    FONT_BTN_SEC, FONT_SIDEBAR_HDR,
)

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp


class ParamsDialogMixin:
    """ECGApp advanced-parameters dialog."""

    def _open_params_dialog(self: ECGApp) -> None:
        """Open a dedicated floating parameters window with all settings clearly grouped."""
        win = ctk.CTkToplevel(self)
        win.title("⚙  Parameters")
        win.geometry("540x720")
        win.configure(fg_color=PANEL)
        win.resizable(True, True)
        win.grab_set()
        win.lift()

        scroll = ctk.CTkScrollableFrame(win, fg_color=PANEL,
                                        scrollbar_button_color=BORDER,
                                        scrollbar_button_hover_color=BORDER2)
        scroll.pack(fill="both", expand=True, padx=0, pady=0)
        s = scroll
        px = dict(padx=SPACE_L)

        def _sec(title: str, color: str = BLUE) -> ctk.CTkFrame:
            hdr = ctk.CTkFrame(s, fg_color=PANEL, corner_radius=8)
            hdr.pack(fill="x", padx=SPACE_M, pady=(SPACE_M, SPACE_XS))
            ctk.CTkFrame(hdr, height=3, fg_color=color, corner_radius=2).pack(fill="x")
            ctk.CTkLabel(hdr, text=title, font=FONT_SIDEBAR_HDR,
                         text_color=color, anchor="w").pack(padx=SPACE_M, pady=(SPACE_S, SPACE_S), fill="x")
            body = ctk.CTkFrame(s, fg_color=CARD, corner_radius=6)
            body.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_S))
            return body

        def _row_entry(parent, label: str, widget_attr: str) -> ctk.CTkEntry:
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", **px, pady=(SPACE_S, 0))
            row.columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=label, font=FONT_SMALL, text_color=MUTED,
                         anchor="w", width=160).grid(row=0, column=0, sticky="w")
            # Mirror value from existing sidebar widget
            src: Optional[ctk.CTkEntry] = getattr(self, widget_attr, None)
            default = src.get() if src is not None else ""
            ent = ctk.CTkEntry(row, font=FONT_LABEL, height=28, fg_color=BG,
                               border_color=BORDER2, text_color=TEXT)
            ent.insert(0, default)
            ent.grid(row=0, column=1, sticky="ew", padx=(SPACE_S, 0))
            return ent

        def _row_switch(parent, label: str, widget_attr: str) -> ctk.CTkSwitch:
            src: Optional[ctk.CTkSwitch] = getattr(self, widget_attr, None)
            is_on = bool(src.get()) if src is not None else False
            sw = ctk.CTkSwitch(parent, text=label, font=FONT_LABEL, text_color=MUTED,
                               progress_color=BLUE, button_color=BORDER2)
            if is_on:
                sw.select()
            sw.pack(**px, anchor="w", pady=(SPACE_S, SPACE_XS))
            return sw

        # ── FILE & SUBJECT ─────────────────────────────────────────────────
        f0 = _sec("📁  File & Subject", BLUE_DARK)
        dlg_channel = _row_entry(f0, "Channel name",    "ent_channel")
        dlg_subject = _row_entry(f0, "Subject ID",      "ent_subject")
        ctk.CTkFrame(f0, height=6, fg_color="transparent").pack()

        # ── SIGNAL ────────────────────────────────────────────────────────
        f1 = _sec("📡  Signal", ORANGE_DARK)
        dlg_fs      = _row_entry(f1, "Sampling rate (Hz)", "ent_fs")
        dlg_t_start = _row_entry(f1, "Start crop (s)",     "ent_t_start")
        dlg_t_end   = _row_entry(f1, "End crop (s)",       "ent_t_end")
        dlg_show_raw = _row_switch(f1, "Show raw signal (vs filtered)", "sw_show_raw")
        ctk.CTkFrame(f1, height=6, fg_color="transparent").pack()

        # ── FILTERS ───────────────────────────────────────────────────────
        f2 = _sec("🔧  Filters", PURPLE)
        dlg_filtering = _row_switch(f2, "Filtering", "sw_filtering")
        dlg_notch     = _row_switch(f2, "Notch 50 Hz", "sw_notch")
        dlg_invert    = _row_switch(f2, "Invert signal polarity", "sw_invert_signal")
        # Labelled Low/High to match the sidebar's FILTERS section -- see
        # the comment there on why (matches filtering.py's bandpass(lo, hi)
        # naming, avoids HP/LP filter-type jargon at a glance).
        dlg_hp = _row_entry(f2, "Low cutoff (Hz)",  "ent_lp")
        dlg_lp = _row_entry(f2, "High cutoff (Hz)", "ent_hp")

        # Clean method
        row_cm = ctk.CTkFrame(f2, fg_color="transparent")
        row_cm.pack(fill="x", **px, pady=(SPACE_S, SPACE_S))
        row_cm.columnconfigure(1, weight=1)
        ctk.CTkLabel(row_cm, text="NK2 clean method:", font=FONT_SMALL, text_color=MUTED,
                     anchor="w", width=160).grid(row=0, column=0, sticky="w")
        _clean_src: Optional[ctk.CTkComboBox] = getattr(self, "cb_clean", None)
        _clean_val = _clean_src.get() if _clean_src is not None else "neurokit"
        dlg_clean = ctk.CTkComboBox(row_cm, font=FONT_LABEL, height=28,
                                    fg_color=BG, border_color=BORDER2,
                                    button_color=BORDER2, text_color=TEXT,
                                    dropdown_fg_color=BG, dropdown_text_color=TEXT,
                                    values=["neurokit", "pantompkins1985",
                                            "elgendi2010", "hamilton2002", "biosppy"])
        dlg_clean.set(_clean_val)
        dlg_clean.grid(row=0, column=1, sticky="ew", padx=(SPACE_S, 0))
        ctk.CTkFrame(f2, height=6, fg_color="transparent").pack()

        # ── DETECTION (SG options) ──────────────────────────────────────────
        # Min R-R, Method and Threshold used to be mirrored here too, each a
        # second copy of a sidebar control that already applies live -- the
        # threshold slider especially, since this dialog's copy didn't even
        # re-run detection until Apply & Close, so it could show a value the
        # detector had never actually used. They now have exactly one home:
        # the sidebar's DETECTION section. Only the SG+Derivative sub-params,
        # which have no live sidebar control of their own, stay here.
        f3 = _sec("🔍  Detection (SG options)", RED)
        ctk.CTkLabel(f3, text="Min R-R, Method and Threshold: sidebar DETECTION section.",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=480, justify="left").pack(**px, pady=(SPACE_S, 0), fill="x")
        dlg_sg_target_fs = _row_entry(f3, "SG target fs (Hz)", "ent_sg_target_fs")
        dlg_sg_window_ms = _row_entry(f3, "SG window (ms)",    "ent_sg_window_ms")
        ctk.CTkLabel(f3, text="Signal is resampled to this rate, then differentiated; "
                             "window should span about one QRS complex",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=480, justify="left").pack(**px, pady=(0, SPACE_XS), fill="x")
        ctk.CTkLabel(f3, text="SG+Deriv: downsample → Savitzky-Golay derivative → R detection\n"
                             "Wavelet: CWT separates noise / QRS / J-wave (pip install PyWavelets)\n"
                             "Envelope Max: local maxima — best for clipped/saturated signals",
                     font=FONT_KPI_LABEL, text_color=LIGHT,
                     anchor="w", wraplength=480, justify="left").pack(**px, pady=(0, SPACE_S), fill="x")
        ctk.CTkFrame(f3, height=6, fg_color="transparent").pack()

        # ── ARTIFACTS ─────────────────────────────────────────────────────
        f4 = _sec("⚠️  Artifacts", ORANGE)
        dlg_artifact = _row_switch(f4, "Auto-correct when you click Analyze (OFF by default)", "sw_artifact")
        ctk.CTkLabel(
            f4,
            text="OFF recommended — use '🔍 Review Artifacts' button for full interactive control.\n"
                 "Auto-correct removes detected ectopic/non-physiological beats without review.",
            font=FONT_KPI_LABEL, text_color=LIGHT,
            anchor="w", wraplength=480, justify="left",
        ).pack(**px, pady=(0, SPACE_M), fill="x")

        # ── Buttons ───────────────────────────────────────────────────────
        ctk.CTkFrame(s, height=1, fg_color=BORDER).pack(fill="x", padx=SPACE_M, pady=(SPACE_M, SPACE_S))
        btn_row = ctk.CTkFrame(s, fg_color="transparent")
        btn_row.pack(fill="x", padx=SPACE_M, pady=(0, SPACE_M))

        def _apply():
            """Write all dialog values back to the sidebar widgets."""
            _write = lambda attr, val: (
                getattr(self, attr).delete(0, "end") or
                getattr(self, attr).insert(0, val)
            ) if getattr(self, attr, None) is not None else None

            _write("ent_channel",  dlg_channel.get())
            _write("ent_subject",  dlg_subject.get())
            _write("ent_fs",       dlg_fs.get())
            _write("ent_t_start",  dlg_t_start.get())
            _write("ent_t_end",    dlg_t_end.get())
            _write("ent_lp",       dlg_hp.get())
            _write("ent_hp",       dlg_lp.get())

            # SG params
            _write("ent_sg_target_fs", dlg_sg_target_fs.get())
            _write("ent_sg_window_ms", dlg_sg_window_ms.get())

            # ComboBoxes
            if self.cb_clean is not None:
                self.cb_clean.set(dlg_clean.get())

            # Switches
            for sw_attr, dlg_sw in [
                ("sw_filtering",    dlg_filtering),
                ("sw_notch",        dlg_notch),
                ("sw_invert_signal",dlg_invert),
                ("sw_show_raw",     dlg_show_raw),
                ("sw_artifact",     dlg_artifact),
            ]:
                w = getattr(self, sw_attr, None)
                if w is None:
                    continue
                if dlg_sw.get():
                    w.select()
                else:
                    w.deselect()

            self._on_filtering_toggle()
            self._on_show_raw_toggle()
            self._set_status("Parameters applied ✓", GREEN)
            win.destroy()

        ctk.CTkButton(btn_row, text="✔  Apply & Close", command=_apply,
                      fg_color=GREEN, hover_color=GREEN_DARK, text_color="white",
                      font=FONT_BTN_PRIMARY, height=36, corner_radius=8).pack(
            side="left", padx=(0, SPACE_M))
        ctk.CTkButton(btn_row, text="✗  Cancel", command=win.destroy,
                      fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                      font=FONT_BTN_SEC, height=36, corner_radius=8).pack(side="left")
