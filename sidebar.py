# -*- coding: utf-8 -*-
"""
ecg.ui.sidebar
--------------
Reusable sidebar components:
  _SidebarSection  -- collapsible labelled section
  IntervalVerifierPanel  -- beat-by-beat interval review widget
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

from plots import CanvasSlot, style_axes
from theme import (
    BG, PANEL, CARD, BORDER, BORDER2, TEXT, MUTED, LIGHT, PLOT,
    RED, BLUE, GREEN, ORANGE, BLUE_HOVER, BLUE_DARK, RED_DARK, RED_LIGHT, ORANGE_DARK,
    CORAL, TEAL, PURPLE,
    FONT_LABEL, FONT_SMALL, FONT_BODY, FONT_MONO,
    FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_SIDEBAR_HDR,
    make_font,
)

log = logging.getLogger("ecg")

# Échelle d'espacement locale — miroir des constantes SPACE_* définies dans
# app.py (non importables ici sans dépendance circulaire ; app.py importe ce
# module). Garder synchronisée si l'échelle de app.py évolue.
_SPACE_XS = 2
_SPACE_S  = 4
_SPACE_M  = 8

class IntervalVerifierPanel:
    """Interactive per-beat landmark verifier embedded in the Intervals tab.

    Holds two matplotlib axes (overlay + detail) rendered into a CanvasSlot,
    plus a CTk navigation bar packed into a separate Tk frame.

    Workflow:
        1. User runs delineation → _launch_interval_verifier() instantiates this.
        2. Overlay (top): all beats as gray traces + mean in blue; current beat
           highlighted in orange; landmark positions as coloured vertical lines.
        3. Detail (bottom): current beat waveform with draggable landmark dots.
           Drag a dot left/right to override a landmark; release recomputes
           PR/QRS/QT for that beat.
        4. Navigation bar: Prev / Next, Accept / Reject, Finalise.
        5. Finalise: replaces _results["intervals"] with the verified DataFrame
           (only accepted beats) and re-plots distributions + summary.
    """

    _LAND_CFG = {
        # key: (colour, marker, label)
        "P_peak":   (BLUE_DARK, "^",  "P"),
        "Q_peak":   (RED_LIGHT, "v",  "Q"),
        "S_peak":   (CORAL, "v",  "S"),
        "J_peak":   (TEAL, "^",  "J"),   # J wave / early repolarization
        "T_peak":   ("#66BB6A", "^",  "T"),
    }

    def __init__(
        self,
        df:        "pd.DataFrame",          # output of analyse_intervals
        beat_mat:  "np.ndarray",            # (n_beats, 2*half_win) full beat matrix
        beat_time: "np.ndarray",            # ms relative to R=0
        fs:        float,
        slot:      "CanvasSlot",            # intervals_ecg slot for figure
        nav_frame: "tk.Frame",              # CTk frame for navigation bar
        on_finalise: "Callable[[pd.DataFrame], None]",
    ) -> None:
        self._df        = df.copy().reset_index(drop=True)
        self._mat       = beat_mat
        self._bt        = beat_time
        self._fs        = fs
        self._slot      = slot
        self._on_finalise = on_finalise
        self._cur       = 0
        self._n         = min(len(df), len(beat_mat))
        self._drag_key: "Optional[str]" = None
        self._dot_artists: "dict[str, Any]" = {}
        self._detail_ax: "Any" = None   # set in _draw, used by drag handler

        # Mean beat for overlay
        self._mean_beat = beat_mat.mean(axis=0) if len(beat_mat) else np.zeros_like(beat_time)

        # Pre-compute per-wave detection rates for display
        self._det_rates: "dict[str, float]" = {}
        for wave, ms_col in [("P", "P_peak_ms"), ("QRS", "QRS_ms"),
                              ("J", "J_peak_ms"), ("T", "T_peak_ms")]:
            if ms_col in df.columns:
                self._det_rates[wave] = float(df[ms_col].notna().mean() * 100)

        self._build_nav(nav_frame)
        self._draw()

    # ── Navigation bar ────────────────────────────────────────────────────

    def _build_nav(self, frame: "tk.Frame") -> None:
        # Clear any previous widgets
        for w in frame.winfo_children():
            w.destroy()

        # ── Left: navigation controls (compact row) ─────────────────────
        self.btn_prev = ctk.CTkButton(
            frame, text="◀", width=32, height=24,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self._prev)
        self.btn_prev.pack(side="left", padx=(2, 1))

        self.lbl_pos = ctk.CTkLabel(frame, text="", font=make_font(9),
                                    text_color=TEXT, width=70)
        self.lbl_pos.pack(side="left", padx=2)

        self.btn_next = ctk.CTkButton(
            frame, text="▶", width=32, height=24,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self._next)
        self.btn_next.pack(side="left", padx=(1, 6))

        # Jump-to-beat
        ctk.CTkLabel(frame, text="→", font=make_font(9),
                     text_color=MUTED).pack(side="left", padx=(0, 2))
        self.ent_beat = ctk.CTkEntry(
            frame, width=50, height=24, font=make_font(9),
            fg_color=BG, border_color=BORDER2, text_color=TEXT,
            placeholder_text="n°")
        self.ent_beat.pack(side="left", padx=(0, 6))
        self.ent_beat.bind("<Return>", self._goto_beat)

        self.btn_accept = ctk.CTkButton(
            frame, text="✓", width=28, height=24,
            fg_color=GREEN, text_color="white", font=make_font(11),
            command=self._accept_next)
        self.btn_accept.pack(side="left", padx=(0, 2))

        self.btn_reject = ctk.CTkButton(
            frame, text="✗", width=28, height=24,
            fg_color="#C62828", text_color="white", font=make_font(11),
            command=self._reject_next)
        self.btn_reject.pack(side="left", padx=(0, 6))

        self.lbl_stats = ctk.CTkLabel(frame, text="", font=make_font(8),
                                      text_color=MUTED)
        self.lbl_stats.pack(side="left", padx=4, fill="x", expand=False)

        # ── Right: detection quality + finalise (compact) ────────────────
        self.btn_done = ctk.CTkButton(
            frame, text="✔ Finaliser", width=95, height=24,
            fg_color=ORANGE, text_color="white", font=FONT_SMALL,
            command=self._finalise)
        self.btn_done.pack(side="right", padx=(4, 2))

        ctk.CTkButton(
            frame, text="Tout accepter", width=90, height=24,
            fg_color=CARD, text_color=TEXT, font=FONT_SMALL,
            command=self._accept_all_finalise,
        ).pack(side="right", padx=(2, 0))

        # Detection rate label (P / J / T %) - more compact
        rate_parts = []
        for wave in ("P", "QRS", "J", "T"):
            if wave in self._det_rates:
                r = self._det_rates[wave]
                col = GREEN if r >= 80 else (ORANGE if r >= 50 else RED)
                rate_parts.append((f"{wave} {r:.0f}%", col))
        if rate_parts:
            # Combine into single label to save space
            labels_text = "  |  ".join(txt for txt, _ in rate_parts)
            lbl_rates = ctk.CTkLabel(frame, text=labels_text, font=make_font(8),
                                    text_color=rate_parts[0][1] if rate_parts else MUTED)
            lbl_rates.pack(side="right", padx=4)

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw(self) -> None:
        i        = min(self._cur, self._n - 1)
        row      = self._df.iloc[i]
        beat     = self._mat[i] if i < len(self._mat) else self._mean_beat
        accepted = bool(row.get("accepted", True))
        bt       = self._bt          # ms relative to R=0

        # ── Compute xlim from actual beat_time (not hardcoded ±100ms) ───
        # Mouse ECG content is mainly within ±60–80ms of R; clamp to actual data
        bt_lo = float(bt[0])
        bt_hi = float(bt[-1])
        DETAIL_LO = max(bt_lo, -65.0)
        DETAIL_HI = min(bt_hi,  95.0)

        border_col = GREEN if accepted else RED

        dot_artists: "dict[str, Any]" = {}
        _self = self   # capture for closures

        # Pre-compute global y-range from all beats for consistent overlay scaling
        try:
            flat = self._mat.ravel()
            finite_vals = flat[np.isfinite(flat)]
            if len(finite_vals) > 0:
                ov_ylo = float(np.percentile(finite_vals, 1))
                ov_yhi = float(np.percentile(finite_vals, 99))
            else:
                ov_ylo, ov_yhi = -1.0, 1.0
        except Exception:
            ov_ylo, ov_yhi = -1.0, 1.0
        ov_pad = max((ov_yhi - ov_ylo) * 0.12, 0.05)
        ov_ylo -= ov_pad;  ov_yhi += ov_pad

        def _render(fig: "Any") -> None:
            # ── Two-panel layout: overview on top, detail on bottom ──────
            gs   = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.30)
            ax_ov  = fig.add_subplot(gs[0])
            ax_det = fig.add_subplot(gs[1])
            _self._detail_ax = ax_det   # only the bottom panel handles drag

            # ── TOP: beat overlay ─────────────────────────────────────────
            style_axes(ax_ov)
            stride = max(1, _self._n // 60)
            for b in _self._mat[::stride]:
                ax_ov.plot(bt, b, color=PLOT["grid"], lw=0.3, alpha=0.15, zorder=1)
            # Mean beat
            ax_ov.plot(bt, _self._mean_beat,
                       color=BLUE, lw=1.8, alpha=0.9, zorder=3)
            # Current beat (highlighted)
            ax_ov.plot(bt, beat, color=ORANGE, lw=1.6, alpha=0.95, zorder=4)
            ax_ov.axvline(0, color=RED_LIGHT, lw=0.8, ls="--", alpha=0.5, zorder=5)

            # Vertical dashed lines for detected landmarks of current beat
            for key, (col, marker, lbl) in _self._LAND_CFG.items():
                ms_col = f"{key}_ms"
                val = row.get(ms_col, np.nan)
                try:
                    v = float(val)
                except Exception:
                    v = float("nan")
                if np.isfinite(v) and DETAIL_LO - 5 <= v <= DETAIL_HI + 5:
                    ax_ov.axvline(v, color=col, lw=1.0, ls=":", alpha=0.8, zorder=6)
                    ax_ov.text(v, ov_yhi - ov_pad * 0.3, lbl, ha="center", va="top",
                               fontsize=7, color=col, fontweight="bold", zorder=7)

            ax_ov.set_xlim(DETAIL_LO, DETAIL_HI)
            ax_ov.set_ylim(ov_ylo, ov_yhi)
            ax_ov.set_ylabel("Ampl.", fontsize=7, color=PLOT["muted"])
            ax_ov.set_xlabel("ms / R", fontsize=7, color=PLOT["muted"])
            ax_ov.tick_params(labelsize=7)
            ax_ov.set_title(
                f"Ensemble ({_self._n}b gris) | moy. bleu | beat {i+1} orange",
                loc="left", fontsize=7.5, color=PLOT["muted"], pad=3)

            # ── BOTTOM: current beat detail with draggable dots ───────────
            style_axes(ax_det)
            for sp in ax_det.spines.values():
                sp.set_edgecolor(border_col)
                sp.set_linewidth(1.8)

            ax_det.plot(bt, beat, color=PLOT["signal"], lw=1.5, zorder=3)
            ax_det.axvline(0, color=RED_LIGHT, lw=1.0, alpha=0.8, zorder=4)

            # Auto-scale y on current beat only
            beat_vals = beat[np.isfinite(beat)]
            if len(beat_vals) > 0:
                det_ylo = float(np.min(beat_vals))
                det_yhi = float(np.max(beat_vals))
            else:
                det_ylo, det_yhi = -1.0, 1.0
            det_pad = max((det_yhi - det_ylo) * 0.20, 0.05)
            ax_det.set_ylim(det_ylo - det_pad, det_yhi + det_pad)

            # Shaded interval spans
            def _shade(lo_key: "Optional[str]", hi_key: "Optional[str]", fc: str) -> None:
                x0 = 0.0 if lo_key is None else float(
                    row.get(f"{lo_key}_ms", float("nan")))  # type: ignore[arg-type]
                x1 = 0.0 if hi_key is None else float(
                    row.get(f"{hi_key}_ms", float("nan")))  # type: ignore[arg-type]
                if np.isfinite(x0) and np.isfinite(x1) and abs(x1 - x0) > 0.5:
                    ax_det.axvspan(min(x0, x1), max(x0, x1),
                                   alpha=0.13, color=fc, zorder=1)

            _shade("P_peak",  "Q_peak", BLUE_DARK)
            _shade("Q_peak",  "S_peak", PURPLE)
            _shade("S_peak",  "J_peak", TEAL)
            _shade(None,      "T_peak", "#D84315")

            # Landmark dots + labels (draggable)
            yspan = max(det_yhi - det_ylo, 1e-6)
            for key, (col, marker, lbl) in _self._LAND_CFG.items():
                ms_col = f"{key}_ms"
                val = row.get(ms_col, np.nan)
                try:
                    v = float(val)   # type: ignore[arg-type]
                except Exception:
                    v = float("nan")
                if not np.isfinite(v):
                    continue
                if not (DETAIL_LO - 5 <= v <= DETAIL_HI + 5):
                    continue
                j_idx = int(np.argmin(np.abs(bt - v)))
                amp   = float(beat[j_idx])
                dot, = ax_det.plot(v, amp, "o", color=col,
                                   markersize=10, zorder=6, picker=8,
                                   markeredgecolor="white", markeredgewidth=0.8)
                offset = yspan * 0.10 * (1 if marker == "^" else -1)
                ax_det.text(v, amp + offset, lbl,
                            ha="center", va="center",
                            fontsize=9, color=col,
                            fontweight="bold", zorder=7)
                dot_artists[key] = dot

            # Title: beat number + accept/reject + interval values
            parts = []
            for col_n, lbl_n in [("PR_ms", "PR"), ("QRS_ms", "QRS"),
                                  ("QT_ms", "QT"), ("RR_ms", "RR")]:
                raw_v = row.get(col_n, float("nan"))
                try:
                    v = float(raw_v)  # type: ignore[arg-type]
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    parts.append(f"{lbl_n} {v:.0f}")

            status_sym = "✓" if accepted else "✗"
            ivl_str    = "  ".join(parts) + " ms" if parts else "aucun intervalle"
            ax_det.set_title(
                f"Beat {i+1}/{_self._n}  {status_sym}   {ivl_str}",
                loc="left", fontsize=8, color=border_col, pad=4)

            ax_det.set_xlabel("Temps depuis R (ms)", fontsize=8)
            ax_det.set_xlim(DETAIL_LO, DETAIL_HI)

            fig.canvas.mpl_connect("button_press_event",   _self._on_press)
            fig.canvas.mpl_connect("motion_notify_event",  _self._on_motion)
            fig.canvas.mpl_connect("button_release_event", _self._on_release)

        self._dot_artists = dot_artists
        self._slot.update(_render)
        self._update_nav()

    def _update_nav(self) -> None:
        n_acc = int(self._df["accepted"].sum())  # type: ignore[arg-type]
        self.lbl_pos.configure(text=f"Beat {self._cur+1} / {self._n}")
        self.lbl_stats.configure(
            text=f"  ✓ {n_acc}   ✗ {self._n - n_acc}")
        self.btn_prev.configure(state="normal" if self._cur > 0 else "disabled")
        self.btn_next.configure(
            state="normal" if self._cur < self._n - 1 else "disabled")

    # ── Navigation ────────────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._cur > 0:
            self._cur -= 1; self._draw()

    def _next(self) -> None:
        if self._cur < self._n - 1:
            self._cur += 1; self._draw()

    def _goto_beat(self, _event: "Any" = None) -> None:
        """Jump to a specific beat number from the entry widget."""
        try:
            n = int(self.ent_beat.get().strip()) - 1   # 1-indexed → 0-indexed
            n = max(0, min(self._n - 1, n))
            self._cur = n
            self._draw()
        except ValueError:
            pass

    def _set_accepted(self, val: bool) -> None:
        self._df.at[self._cur, "accepted"] = val

    def _accept_next(self) -> None:
        self._set_accepted(True)
        if self._cur < self._n - 1:
            self._cur += 1
        self._draw()

    def _reject_next(self) -> None:
        self._set_accepted(False)
        if self._cur < self._n - 1:
            self._cur += 1
        self._draw()

    def _finalise(self) -> None:
        accepted_df = self._df[self._df["accepted"]].copy()
        self._on_finalise(accepted_df)  # type: ignore[arg-type]

    def _accept_all_finalise(self) -> None:
        self._df["accepted"] = True
        self._on_finalise(self._df.copy())

    # ── Drag interaction ──────────────────────────────────────────────────

    def _on_press(self, event: "Any") -> None:
        # Only accept clicks in the detail (bottom) axes
        if event.inaxes is None or event.button != 1:
            return
        if self._detail_ax is not None and event.inaxes is not self._detail_ax:
            return
        best_key: "Optional[str]" = None
        best_d = float("inf")
        for key, dot in self._dot_artists.items():
            xd = dot.get_xdata()[0]
            d  = abs(event.xdata - xd)
            if d < best_d and d < 6.0:
                best_d = d; best_key = key
        self._drag_key = best_key

    def _on_motion(self, event: "Any") -> None:
        if self._drag_key is None or event.xdata is None:
            return
        if self._detail_ax is not None and event.inaxes is not self._detail_ax:
            return
        key    = self._drag_key
        ms_col = f"{key}_ms"
        if ms_col in self._df.columns:
            self._df.at[self._cur, ms_col] = float(event.xdata)
            # Keep absolute-time column in sync
            s_col = ms_col.replace("_ms", "_s")
            r_s   = float(str(self._df.at[self._cur, "R_peak_s"]))
            if s_col in self._df.columns and np.isfinite(r_s):
                self._df.at[self._cur, s_col] = r_s + event.xdata / 1000.0
        self._draw()

    def _on_release(self, _event: "Any") -> None:
        if self._drag_key:
            self._recompute_row(self._cur)
            self._drag_key = None
            self._draw()

    def _recompute_row(self, i: int) -> None:
        """Recompute PR / QRS / QT from current ms-from-R positions."""
        r = self._df.iloc[i]
        def _ms(key: str) -> float:
            v = r.get(f"{key}_ms", np.nan)
            # Narrowed from a bare `except Exception: return np.nan` --
            # that swallowed genuine bugs (e.g. a bad dtype landing in this
            # column) the same silent way it handles the expected/legitimate
            # case of a landmark simply not being detected for this beat.
            # Only float-conversion failures degrade to NaN now; anything
            # else is logged so it's diagnosable instead of vanishing.
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                log.debug("_recompute_row: %s_ms=%r not numeric: %s", key, v, exc)
                return np.nan
            return fv if np.isfinite(fv) else np.nan

        p_pk = _ms("P_peak");  q_pk = _ms("Q_peak")
        s_pk = _ms("S_peak");  t_pk = _ms("T_peak")

        pr  = q_pk - p_pk if (np.isfinite(p_pk) and np.isfinite(q_pk)) else np.nan
        qrs = s_pk - q_pk if (np.isfinite(q_pk) and np.isfinite(s_pk)) else np.nan
        qt  = t_pk        if np.isfinite(t_pk) else np.nan   # ms from R

        for col, val in [("PR_ms", pr), ("QRS_ms", qrs), ("QT_ms", qt)]:
            if col in self._df.columns:
                self._df.at[i, col] = val




class _SidebarSection:
    """A collapsible sidebar section with a clickable arrow header.

    The outer ``_wrapper`` frame is always packed (preserving its position in
    the parent's pack order).  Only the inner ``frame`` is shown/hidden, so
    re-opening a section never moves it to the bottom of the list.

    Usage::
        sec = _SidebarSection(parent, "FILTERS", initially_open=True)
        ctk.CTkLabel(sec.frame, text="HP cut").pack(...)
    """

    def __init__(self, parent, title: str, initially_open: bool = True) -> None:
        self._title = title
        self._open  = initially_open

        # Outer wrapper — ALWAYS packed; never removed from layout
        self._wrapper = ctk.CTkFrame(parent, fg_color="transparent")
        self._wrapper.pack(fill="x")

        # ── Header row ──────────────────────────────────────────────
        # Compact + token-based spacing (was hardcoded padx=10, pady=(8,0),
        # height=28 — inconsistent with the SPACE_* scale used everywhere
        # else in the sidebar, and taller than needed across 5 stacked
        # sections). text_color raised from MUTED to TEXT so section titles
        # read clearly at a glance instead of blending into the background.
        hdr = ctk.CTkFrame(self._wrapper, fg_color=BORDER, corner_radius=6, height=26)
        hdr.pack(fill="x", padx=_SPACE_M, pady=(_SPACE_S, 0))
        hdr.pack_propagate(False)

        arrow = "▾" if initially_open else "▸"
        self._btn = ctk.CTkButton(
            hdr, text=f"  {arrow}  {title}",
            font=FONT_SIDEBAR_HDR, text_color=TEXT,
            fg_color="transparent",
            hover_color=BORDER2,
            anchor="w", height=26, corner_radius=6,
            command=self._toggle,
        )
        self._btn.pack(fill="x")
        # ── Content frame — inside the wrapper, shown or hidden ──────
        self.frame = ctk.CTkFrame(self._wrapper, fg_color="transparent")
        if initially_open:
            self.frame.pack(fill="x")

    def _toggle(self) -> None:
        self._open = not self._open
        arrow = "▾" if self._open else "▸"
        self._btn.configure(text=f"  {arrow}  {self._title}")
        if self._open:
            self.frame.pack(fill="x")
        else:
            self.frame.pack_forget()

    def open(self) -> None:
        if not self._open:
            self._toggle()

    def close(self) -> None:
        if self._open:
            self._toggle()


