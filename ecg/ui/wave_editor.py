# -*- coding: utf-8 -*-
"""
ecg.ui.wave_editor
------------------
WaveTemplateMiniEditor -- interactive CTk dialog for calibrating
the beat-template (P/Q/S/T landmark positions).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from tkinter import filedialog, messagebox
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

def hit_landmark(landmarks: "Any", keys: "Any", x_ms: float,
                 radius_ms: float) -> "tuple[Optional[str], str]":
    """(landmark, "center" | "lo" | "hi") whose line is nearest *x_ms*, within *radius_ms*."""
    best_key: "Optional[str]" = None
    best_d   = float("inf")
    best_mode = "center"
    for key in keys:
        if key not in landmarks:
            continue
        c, hw = landmarks[key]
        for val, mode in [(c, "center"), (c - hw, "lo"), (c + hw, "hi")]:
            d = abs(x_ms - val)
            if d < best_d and d < radius_ms:
                best_d    = d
                best_key  = key
                best_mode = mode
    return best_key, best_mode


def dragged_landmark(center: float, half: float, mode: str, x_ms: float) -> "tuple[float, float]":
    """(centre, half-window) once the line of *mode* is at *x_ms*."""
    if mode == "center":
        return round(x_ms, 1), half
    if mode == "lo":
        return center, round(max(2.0, center - x_ms), 1)
    return center, round(max(2.0, x_ms - center), 1)


class WaveTemplateMiniEditor(ctk.CTkToplevel):
    """Compact P/Q/R/S/T template editor — draggable lines on the mean beat.

    Shows the quality-filtered mean beat with one coloured vertical line
    (landmark centre) and one shaded band (search half-window) per landmark
    in ``_KEYS``.
    Drag a line to move a centre; drag a shaded edge to resize the window.

    The template edited is the CURRENT RECORDING's (``app.analysis.
    wave_template``): "Apply to this recording" changes only it (kept in the
    recording's session) and writes nothing to disk. Making it the default
    for recordings opened later is the separate, explicit "Save as default
    template" (``WaveTemplate.save``); "Save to file..." / "Load..." exchange
    named template files. The provenance (default / calibrated or edited on
    this recording / loaded from a file) follows every action and is shown
    in the Morphology & Intervals tab and in exports.

    Replaces the previous 1400-line WaveTemplateEditor.  All logic lives here;
    the external WaveTemplate data model is unchanged.
    """

    # QRS onset / QRS offset (J point) bound PR, QRS and QT (PR = P onset ->
    # QRS onset, QRS = onset -> offset, QT = onset -> T end); their windows
    # only constrain where the per-beat detector may place them.
    _KEYS   = ["P_peak", "QRS_onset", "Q_peak", "S_peak", "QRS_offset",
               "J_peak", "T_peak"]
    _COLS   = {
        "P_peak":  BLUE_DARK,
        "QRS_onset": "#FFB74D",
        "Q_peak":  "#EF9A9A", "S_peak":  CORAL,
        "QRS_offset": "#BA68C8",
        "J_peak":  TEAL,   # teal — J wave / early repolarization
        "T_peak":  "#66BB6A",
    }
    _LABELS = {
        "P_peak": "P peak",
        "QRS_onset": "QRS onset",
        "Q_peak": "Q peak",  "S_peak": "S peak",
        "QRS_offset": "J point",
        "J_peak": "J peak",
        "T_peak": "T peak",
    }
    # For the plots that draw these waves.
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
        intervals: "Optional[Any]"        = None,
    ) -> None:
        super().__init__(parent)
        self.title("Wave Template Editor")
        self.geometry("800x520")
        self.resizable(True, True)
        self.configure(fg_color=BG)

        self._template  = template
        self._parent_app = parent
        # This recording's delineation (for "Refine from delineation").
        self._intervals = intervals
        # Provenance of the windows being edited (applied with them).
        self._src:    str = template.source
        self._origin: str = template.origin
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
        # A template saved before QRS onset / offset existed lacks them: show
        # (and save) the defaults so they can be adjusted.
        self._lm: "dict[str, list[float]]" = {
            k: list(v) for k, v in {**WaveTemplate.DEFAULTS,
                                    **template.landmarks}.items()
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
            bar, text="⚡ Auto-calibrate", width=120, height=30,
            fg_color=PURPLE_DARK, text_color="white", font=FONT_SMALL,
            command=self._auto_calibrate,
        ).pack(side="left", padx=(0, 6))

        if self._intervals is not None:
            ctk.CTkButton(
                bar, text="⟲ Refine from delineation", width=170, height=30,
                fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
                command=self._refine_from_intervals,
            ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            bar, text="📂 Load…", width=80, height=30,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self._load_file,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            bar, text="💾 Save to file…", width=110, height=30,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self._save_file,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            bar, text="✗ Cancel", width=80, height=30,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self.destroy,
        ).pack(side="right")

        ctk.CTkButton(
            bar, text="✔ Apply to this recording", width=170, height=30,
            fg_color=GREEN, text_color="white", font=FONT_SMALL,
            command=self._apply,
        ).pack(side="right", padx=(0, 6))

        ctk.CTkButton(
            bar, text="Save as default template", width=170, height=30,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self._save_default,
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
        ax.set_title(f"Wave Template — {self._pending_template().provenance()}",
                     loc="left", fontsize=9, color=PLOT["muted"])
        self._canvas.draw_idle()

    # ── Drag interaction ─────────────────────────────────────────────────

    def _hit_test(self, x_ms: float) -> "tuple[Optional[str], str]":
        return hit_landmark(self._lm, self._KEYS, x_ms, self._GRAB_R_MS)

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
        self._lm[key] = list(dragged_landmark(c, hw, self._drag_mode, x))
        self._mark_edited()
        self._draw()

    def _on_release(self, _event: "Any") -> None:
        self._drag_key = None

    # ── Provenance ───────────────────────────────────────────────────────

    def _pending_template(self) -> WaveTemplate:
        """The windows on screen as a WaveTemplate (with their provenance)."""
        wt = WaveTemplate()
        wt.landmarks = {k: (float(v[0]), float(v[1])) for k, v in self._lm.items()}
        wt.source, wt.origin = self._src, self._origin
        wt.created_at = self._template.created_at if self._src == "saved-default" else ""
        return wt

    def _set_provenance(self, source: str, origin: str = "") -> None:
        self._src, self._origin = source, origin

    def _mark_edited(self) -> None:
        """A hand edit: the provenance becomes "edited on this recording
        (from <what was there before>)" -- kept once already edited."""
        if self._src != "edited":
            self._set_provenance("edited", self._pending_template().provenance())

    # ── Actions ──────────────────────────────────────────────────────────

    def _reset_defaults(self) -> None:
        self._lm = {k: list(v) for k, v in WaveTemplate.DEFAULTS.items()}
        self._set_provenance("default")
        self._draw()

    def _auto_calibrate(self) -> None:
        tmp = WaveTemplate()
        tmp.calibrate_from_beat(self._bt, self._mb)
        self._lm = {k: list(v) for k, v in tmp.landmarks.items()}
        self._set_provenance("auto-calibrated")
        self._draw()

    def _refine_from_intervals(self) -> None:
        """Centre the P / Q / S / T windows on this recording's delineation
        (the explicit replacement of the old silent auto-update)."""
        from ecg.core.analysis import refine_template_from_intervals
        new = (refine_template_from_intervals(self._intervals, self._pending_template())
               if self._intervals is not None else None)
        if new is None:
            messagebox.showinfo(
                "Not enough beats",
                "Refining needs at least 10 fully delineated beats — Measure "
                "intervals first.", parent=self)
            return
        self._lm = {k: list(v) for k, v in {**WaveTemplate.DEFAULTS,
                                            **new.landmarks}.items()}
        self._set_provenance("refined")
        self._draw()

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Load wave template",
            filetypes=[("Wave template", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            wt = WaveTemplate.load(path)
        except Exception as exc:
            messagebox.showwarning("Load failed", str(exc), parent=self)
            return
        self._lm = {k: list(v) for k, v in {**WaveTemplate.DEFAULTS,
                                            **wt.landmarks}.items()}
        self._set_provenance(wt.source, wt.origin)
        self._draw()
        problems = wt.plausibility_problems()
        if problems:
            # Loaded anyway (the user picked it and sees it on the beat),
            # but these windows would not be accepted as the saved default.
            messagebox.showwarning(
                "Check these windows",
                "These search windows look implausible for a mouse beat:\n\n"
                + "\n".join(f"• {p}" for p in problems)
                + "\n\nCheck them on the beat before applying.", parent=self)

    def _save_file(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, title="Save wave template", defaultextension=".json",
            filetypes=[("Wave template", "*.json")])
        if not path:
            return
        try:
            self._pending_template().save(path)
        except Exception as exc:
            messagebox.showwarning("Save failed", str(exc), parent=self)

    def _commit(self) -> None:
        """Apply the windows (and their provenance) to this recording's
        template, in place. Writes nothing to disk."""
        pending = self._pending_template()
        self._template.landmarks  = pending.landmarks
        self._template.source     = pending.source
        self._template.origin     = pending.origin
        self._template.created_at = pending.created_at
        self._template.confirmed  = True
        if hasattr(self._parent_app, "_wave_template"):
            self._parent_app._wave_template = self._template
        self._saved = True

    def _apply(self) -> None:
        self._commit()
        messagebox.showinfo(
            "Template applied",
            "Template applied to this recording (kept in its session).\n"
            "Measure intervals again to use these search windows.",
            parent=self,
        )
        self.destroy()

    def _recording_name(self) -> str:
        """File name of the recording being edited ('' when none)."""
        fp = getattr(self._parent_app, "_filepath", None)
        return os.path.basename(str(fp)) if fp else ""

    def _save_default(self) -> None:
        """Explicit: make these windows the default for recordings opened
        from now on (recordings already analysed keep their own). Windows
        the startup check would reject are not saved as the default."""
        problems = self._pending_template().plausibility_problems()
        if problems:
            messagebox.showwarning(
                "Not saved as default",
                "These search windows look implausible for a mouse beat, so "
                "they would not be used as the default template:\n\n"
                + "\n".join(f"• {p}" for p in problems)
                + "\n\nAdjust them (or Apply to this recording only).",
                parent=self)
            return
        if not messagebox.askyesno(
                "Save as default template",
                "Use these search windows as the default template for "
                "recordings opened from now on?\n\nRecordings already "
                "analysed and saved sessions keep their own template.",
                parent=self):
            return
        try:
            self._pending_template().save(recording=self._recording_name())
        except Exception as exc:
            messagebox.showwarning("Save failed", str(exc), parent=self)
            return
        self._commit()
        messagebox.showinfo(
            "Template saved",
            "Saved as the default template and applied to this recording.\n"
            "Measure intervals again to use these search windows.",
            parent=self,
        )
        self.destroy()


# ════════════════════════════════════════════════════════════
#  SESSION RESULT SERIALISATION HELPERS
# ════════════════════════════════════════════════════════════

