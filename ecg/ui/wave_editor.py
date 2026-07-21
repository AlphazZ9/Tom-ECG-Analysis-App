# -*- coding: utf-8 -*-
"""
ecg.ui.wave_editor
------------------
WaveTemplateMiniEditor -- interactive CTk dialog for calibrating
the beat-template (P/Q/S/T landmark positions).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ecg.ui.plots import style_axes
from ecg.core.wave_template import WaveTemplate, _mouse_demo_beat, detect_waves_on_beat
from ecg.ui.theme import (
    BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT,
    RED, RED_LIGHT, BLUE, GREEN, ORANGE, PLOT,
    BLUE_DARK, CORAL, TEAL, PURPLE_DARK,
    FONT_LABEL, FONT_SMALL, FONT_BTN_PRIMARY, FONT_BTN_SEC,
    FONT_HINT,
)

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")

class WaveTemplateMiniEditor(ctk.CTkToplevel):
    """Compact P/Q/R/S/T template editor — draggable lines on the mean beat.

    Shows the quality-filtered mean beat with 6 coloured vertical lines
    (landmark centres) and 6 shaded bands (search half-windows).
    Drag a line to move a centre; drag a shaded edge to resize the window.
    Click "Save & Confirm" to persist and mark the template as user-confirmed.

    Replaces the previous 1400-line WaveTemplateEditor.  All logic lives here;
    the external WaveTemplate data model is unchanged.
    """

    _KEYS   = ["P_peak", "Q_peak", "S_peak", "J_peak", "T_peak"]
    _COLS   = {
        "P_peak":  BLUE_DARK,
        "Q_peak":  "#EF9A9A", "S_peak":  CORAL,
        "J_peak":  TEAL,   # teal — J wave / early repolarization
        "T_peak":  "#66BB6A",
    }
    _LABELS = {
        "P_peak": "P peak",
        "Q_peak": "Q peak",  "S_peak": "S peak",
        "J_peak": "J peak",
        "T_peak": "T peak",
    }
    # Aliases expected by callers that reference WaveTemplateEditor.WAVE_COLORS etc.
    WAVE_COLORS       = _COLS
    WAVE_SHORT_LABELS = _LABELS
    _GRAB_R_MS = 5.0   # grab radius (ms) for click-on-line

    def __init__(
        self,
        parent:    "ECGApp",
        template:  "WaveTemplate",
        mean_beat: "Optional[np.ndarray]" = None,
        beat_time: "Optional[np.ndarray]" = None,
        beat_sd:   "Optional[np.ndarray]" = None,
    ) -> None:
        super().__init__(parent)
        self.title("Wave Template Editor")
        self.geometry("800x520")
        self.resizable(True, True)
        self.configure(fg_color=BG)

        self._template  = template
        self._parent_app = parent
        self._saved:     bool                    = False
        self._drag_key:  "Optional[str]" = None
        self._drag_mode: str             = "center"  # "center" | "lo" | "hi"

        # Use provided beat or synthetic demo
        if mean_beat is not None and beat_time is not None:
            self._mb = np.asarray(mean_beat, dtype=float)
            self._bt = np.asarray(beat_time, dtype=float)
            self._sd = np.asarray(beat_sd,   dtype=float) if beat_sd is not None else None
        else:
            self._bt, self._mb = _mouse_demo_beat()
            self._sd = None

        # Working copy of landmarks (list so we can mutate in place)
        self._lm: "dict[str, list[float]]" = {
            k: list(v) for k, v in template.landmarks.items()
        }

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_canvas()
        self._build_controls()
        self._draw()

        self.grab_set()
        self.lift()
        self.after(100, self.lift)

    # ── Canvas ────────────────────────────────────────────────────────────

    def _build_canvas(self) -> None:
        frm = tk.Frame(self, bg=PLOT["bg"])
        frm.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 2))
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(9, 4.0), dpi=90)
        self._fig.patch.set_facecolor(PLOT["bg"])
        self._ax  = self._fig.add_subplot(111)
        style_axes(self._ax)

        self._canvas = FigureCanvasTkAgg(self._fig, master=frm)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

        ctk.CTkLabel(
            self,
            text="Drag a coloured line to move a landmark centre.  "
                 "Drag a dashed edge to resize the search window.",
            font=FONT_HINT, text_color=MUTED,
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 2))

    def _build_controls(self) -> None:
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        ctk.CTkButton(
            bar, text="↺ Reset defaults", width=130, height=30,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self._reset_defaults,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            bar, text="⚡ Auto-calibrate from mean beat", width=230, height=30,
            fg_color=PURPLE_DARK, text_color="white", font=FONT_SMALL,
            command=self._auto_calibrate,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            bar, text="✗ Cancel", width=80, height=30,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self.destroy,
        ).pack(side="right")

        ctk.CTkButton(
            bar, text="✔ Save & Confirm", width=150, height=30,
            fg_color=GREEN, text_color="white", font=FONT_SMALL,
            command=self._save,
        ).pack(side="right", padx=(0, 6))

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw(self) -> None:
        ax = self._ax
        ax.cla()
        style_axes(ax)

        # Mean beat ± SD band
        if self._sd is not None:
            ax.fill_between(self._bt,
                            self._mb - self._sd,
                            self._mb + self._sd,
                            color=PLOT["signal"], alpha=0.15)
        ax.plot(self._bt, self._mb, color=PLOT["signal"], lw=1.5, zorder=2)
        ax.axvline(0, color=RED_LIGHT, lw=0.8, alpha=0.6)

        ylo, yhi = ax.get_ylim()
        yspan = yhi - ylo if yhi != ylo else 1.0

        for key in self._KEYS:
            if key not in self._lm:
                continue
            c, hw = self._lm[key][0], self._lm[key][1]
            col   = self._COLS[key]
            lo, hi = c - hw, c + hw
            ax.axvspan(lo, hi, alpha=0.12, color=col, zorder=0)
            ax.axvline(c,  color=col, lw=1.5, alpha=0.9, zorder=3)
            ax.axvline(lo, color=col, lw=0.8, ls="--", alpha=0.55, zorder=3)
            ax.axvline(hi, color=col, lw=0.8, ls="--", alpha=0.55, zorder=3)
            ax.text(c, yhi - 0.04 * yspan, self._LABELS[key],
                    ha="center", va="top", fontsize=7, color=col, rotation=90)

        ax.set_xlim(float(self._bt[0]), float(self._bt[-1]))
        ax.set_xlabel("Time from R (ms)", fontsize=9)
        ax.set_ylabel("Amplitude (norm.)", fontsize=9)
        src_note = f"  source={self._template.source}"
        conf_note = "  ✓ confirmed" if self._template.confirmed else "  (not confirmed)"
        ax.set_title(f"Wave Template{src_note}{conf_note}",
                     loc="left", fontsize=9, color=PLOT["muted"])
        self._canvas.draw_idle()

    # ── Drag interaction ─────────────────────────────────────────────────

    def _hit_test(self, x_ms: float) -> "tuple[Optional[str], str]":
        best_key: "Optional[str]" = None
        best_d   = float("inf")
        best_mode = "center"
        for key in self._KEYS:
            if key not in self._lm:
                continue
            c, hw = self._lm[key]
            for val, mode in [(c, "center"), (c - hw, "lo"), (c + hw, "hi")]:
                d = abs(x_ms - val)
                if d < best_d and d < self._GRAB_R_MS:
                    best_d    = d
                    best_key  = key
                    best_mode = mode
        return best_key, best_mode

    def _on_press(self, event: "Any") -> None:
        if event.inaxes is None or event.xdata is None or event.button != 1:
            return
        key, mode = self._hit_test(float(event.xdata))
        self._drag_key  = key
        self._drag_mode = mode

    def _on_motion(self, event: "Any") -> None:
        if self._drag_key is None or event.xdata is None:
            return
        key  = self._drag_key
        x    = float(event.xdata)
        c, hw = self._lm[key]
        if self._drag_mode == "center":
            self._lm[key][0] = round(x, 1)
        elif self._drag_mode == "lo":
            self._lm[key][1] = round(max(2.0, c - x), 1)
        elif self._drag_mode == "hi":
            self._lm[key][1] = round(max(2.0, x - c), 1)
        self._draw()

    def _on_release(self, _event: "Any") -> None:
        self._drag_key = None

    # ── Actions ──────────────────────────────────────────────────────────

    def _reset_defaults(self) -> None:
        self._lm = {k: list(v) for k, v in WaveTemplate.DEFAULTS.items()}
        self._draw()

    def _auto_calibrate(self) -> None:
        tmp = WaveTemplate()
        tmp.calibrate_from_beat(self._bt, self._mb)
        self._lm = {k: list(v) for k, v in tmp.landmarks.items()}
        self._draw()

    def _save(self) -> None:
        self._template.landmarks = {  # type: ignore[assignment]
            k: (float(v[0]), float(v[1])) for k, v in self._lm.items()
        }
        self._template.confirmed  = True
        self._template.source     = "user-confirmed"
        try:
            self._template.save()
        except Exception as exc:
            messagebox.showwarning("Save failed", str(exc), parent=self)
            return
        if hasattr(self._parent_app, "_wave_template"):
            self._parent_app._wave_template = self._template
        self._saved = True
        messagebox.showinfo(
            "Template saved",
            "Template saved and confirmed.\n"
            "Next delineation run will use these search windows.",
            parent=self,
        )
        self.destroy()


# Keep old name as alias so any leftover references don't break
WaveTemplateEditor = WaveTemplateMiniEditor

# ════════════════════════════════════════════════════════════
#  SESSION RESULT SERIALISATION HELPERS
# ════════════════════════════════════════════════════════════

