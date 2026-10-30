# -*- coding: utf-8 -*-
"""
ecg.ui.app_helpers
------------------
ECGApp widget factories, status bar, copy menus and table-to-text helpers.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
import numpy as np
import pandas as pd
from ecg.core.models import MouseECG, FilterParams
from ecg.ui.theme import (
    THEME, BG, BORDER, BORDER2, TEXT, MUTED, BLUE, GREEN, COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER, COLOR_SUCCESS, COLOR_SUCCESS_HOVER, COLOR_WARNING,
    COLOR_WARNING_HOVER, COLOR_DANGER, COLOR_DANGER_HOVER, COLOR_SECONDARY,
    COLOR_SECONDARY_HOVER, SPACE_XS, SPACE_S, SPACE_M, FONT_LABEL, FONT_SMALL,
    FONT_BODY, FONT_BTN_PRIMARY, FONT_BTN_SEC,
)

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class HelpersMixin:
    """ECGApp widget factories, status bar, copy menus and table-to-text helpers."""

    # ════════════════════════════════════════════════════════
    #  SIDEBAR WIDGET HELPERS
    # ════════════════════════════════════════════════════════


    def _sidebar_entry(self: ECGApp, parent, label: str, attr: str, default: str, pad: dict) -> None:
        """Add a labelled entry and bind it to self.ent_<attr>."""
        ctk.CTkLabel(parent, text=label, font=FONT_SMALL,
                     text_color=MUTED, anchor="w").pack(**pad, fill="x")
        e = ctk.CTkEntry(parent, font=FONT_LABEL, height=30, fg_color=BG,
                          border_color=BORDER2, text_color=TEXT)
        e.insert(0, default)
        e.pack(**pad, fill="x", pady=(SPACE_XS, SPACE_M))
        setattr(self, f"ent_{attr}", e)

    def _sidebar_entry_row(self: ECGApp, parent, pad: dict, items: list[tuple]) -> None:
        """Render a horizontal row of (label, attr, default) entry pairs."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(**pad, fill="x", pady=(0, SPACE_S))
        for i, (label, attr, default) in enumerate(items):
            col = ctk.CTkFrame(row, fg_color="transparent")
            col.pack(side="left", fill="x", expand=True,
                     padx=(0, SPACE_S) if i < len(items) - 1 else 0)
            ctk.CTkLabel(col, text=label, font=FONT_SMALL, text_color=MUTED).pack(anchor="w")
            e = ctk.CTkEntry(col, font=FONT_LABEL, height=28, fg_color=BG,
                              border_color=BORDER2, text_color=TEXT)
            e.insert(0, default)
            e.pack(fill="x")
            setattr(self, f"ent_{attr}", e)

    def _switch(self: ECGApp, parent, label: str, pad: dict, default_on: bool = False,
                command=None) -> ctk.CTkSwitch:
        sw = ctk.CTkSwitch(parent, text=label, font=FONT_LABEL,
                            text_color=MUTED, progress_color=BLUE,
                            button_color=BORDER2, command=command)
        if default_on:
            sw.select()
        sw.pack(**pad, anchor="w", pady=(0, SPACE_S))
        return sw

    def _toolbar_chip(self: ECGApp, parent) -> ctk.CTkFrame:
        """Grouped-control 'chip' background for nav/hdr instrument-toolbar
        clusters. Uses BG (a subtle off-white, distinct from the PANEL/CARD
        pure-white now used for large surfaces) rather than CARD -- since
        PANEL and CARD both became pure white, a CARD-filled chip on a
        PANEL-filled bar was indistinguishable from its background, and
        toolbar button groups lost their visual grouping entirely. BG is a
        gentle enough tint to read as "one control cluster" without
        reintroducing a visible grey block."""
        return ctk.CTkFrame(parent, fg_color=BG, corner_radius=8,
                             border_width=1, border_color=BORDER)

    def _btn(self: ECGApp, parent, text: str, command, pad: dict,
             variant: str = "secondary", h: int = 28, bold: bool = False) -> ctk.CTkButton:
        """Build a sidebar-style secondary/action button in one of 5 named
        variants (primary/success/warning/danger/secondary), matching the
        rationalised colour system in theme.py (COLOR_* semantic tokens).

        Reads the module-level COLOR_* globals fresh on every call (not a
        precomputed lookup table) so buttons rebuilt after a theme switch
        (_rebuild_ui()) always pick up the current theme's colours instead
        of whatever was active when a cached table would have been built.
        """
        if variant == "primary":
            fg, hover, text_color = COLOR_PRIMARY, COLOR_PRIMARY_HOVER, "white"
        elif variant == "success":
            fg, hover, text_color = COLOR_SUCCESS, COLOR_SUCCESS_HOVER, "white"
        elif variant == "warning":
            fg, hover, text_color = COLOR_WARNING, COLOR_WARNING_HOVER, "white"
        elif variant == "danger":
            fg, hover, text_color = COLOR_DANGER, COLOR_DANGER_HOVER, "white"
        else:
            fg, hover, text_color = COLOR_SECONDARY, COLOR_SECONDARY_HOVER, MUTED
        _h = max(24, int(h * THEME.font_scale))
        btn = ctk.CTkButton(
            parent, text=text, command=command,
            fg_color=fg, hover_color=hover, text_color=text_color,
            font=FONT_BTN_PRIMARY if bold else FONT_BTN_SEC, height=_h, corner_radius=8,
        )
        btn.pack(**pad, fill="x", pady=(0, SPACE_S))
        return btn

    def _reset_params(self: ECGApp) -> None:
        """Reset all filter/detection parameters to MouseECG species defaults."""
        fp = FilterParams()   # constructed with all dataclass defaults

        _fields: list[tuple[str, str]] = [
            ("ent_channel", fp.channel),
            ("ent_fs",      str(fp.fs)),
            ("ent_t_start", str(fp.t_start)),
            ("ent_t_end",   str(fp.t_end)),
            ("ent_lp",      str(fp.lp)),
            ("ent_hp",      str(int(fp.hp))),
            ("ent_minrr",   f"{fp.min_rr_ms:g}"),     # physiological floor (≈ 66.7 ms)
            ("ent_maxrr",   f"{fp.max_rr_ms:g}"),     # HRV RR ceiling (awake mouse)
            # peak_distance_ms stored on self._peak_distance_ms (no dedicated widget)
            ("ent_sg_target_fs", str(fp.sg_target_fs)),
            ("ent_sg_window_ms", "20"),
        ]
        for attr, val in _fields:
            w = getattr(self, attr, None)
            if w is None:
                continue
            w.delete(0, "end")
            w.insert(0, val)

        # Threshold slider + entry
        try:
            self.sl_thr.set(fp.thresh)  # type: ignore[union-attr]
            self.lbl_thr.configure(text=f"Threshold:  {fp.thresh:.2f}")
            # Detection hasn't re-run at this value yet -- drop the stale
            # "Current: … norm." readout rather than show a mismatched one.
            self.lbl_thr_hint.configure(
                text="Fraction of the typical (median) R-peak height — 0.50 "
                     "keeps peaks ≥ 50% as tall. Lower = more peaks.")
        except Exception as e:
            log.debug("sl_thr/lbl_thr restore failed: %s", e)
        try:
            self.ent_thr.delete(0, "end")  # type: ignore[union-attr]
            self.ent_thr.insert(0, str(fp.thresh))  # type: ignore[union-attr]
        except Exception as e:
            log.debug("ent_thr restore failed: %s", e)

        # Epoch entries
        try:
            self.ent_epoch.delete(0, "end")
            self.ent_epoch.insert(0, str(int(MouseECG.EPOCH_DEFAULT_S)))
            self.ent_overlap.delete(0, "end")
            self.ent_overlap.insert(0, "0")
        except Exception as e:
            log.debug("epoch/overlap entry restore failed: %s", e)

        # Switches — match FilterParams defaults (sw_filtering is the
        # inverse of fp.no_filter -- ON means filtering is applied)
        _sw_defaults: list[tuple[str, bool]] = [
            ("sw_notch",         fp.notch_filter),
            ("sw_artifact",      fp.artifact_correction),
            ("sw_filtering",     not fp.no_filter),
            ("sw_epoch",         fp.auto_epochs),
            ("sw_invert_signal", fp.invert_signal),
        ]
        for attr, on in _sw_defaults:
            w = getattr(self, attr, None)
            if w is None:
                continue
            w.select() if on else w.deselect()

        # ComboBox
        try:
            self.cb_clean.set(fp.clean_method)  # type: ignore[union-attr]
        except Exception as e:
            log.debug("cb_clean restore failed: %s", e)
        # Detection method reset
        try:
            if self.cb_det_method is not None:
                self.cb_det_method.set("SG + Derivative (10 kHz)")  # type: ignore[union-attr]
                self._on_det_method_change("SG + Derivative (10 kHz)")
        except Exception as e:
            log.debug("cb_det_method reset failed: %s", e)

        # Resync FILTER SETTINGS greying + the "Processing" summary to the
        # just-reset sw_filtering/notch/cutoff/clean values.
        self._on_filtering_toggle()
        self._update_filter_summary()

        self._set_status("Parameters reset to mouse ECG defaults ✓", GREEN)

    # ════════════════════════════════════════════════════════
    #  GENERAL HELPERS
    # ════════════════════════════════════════════════════════

    def _textbox(self: ECGApp, parent, h: int = 180, padx: int = 0, expand: bool = False):
        kwargs = dict(font=FONT_BODY, fg_color="transparent", text_color=TEXT,
                      border_width=0, scrollbar_button_color=BORDER,
                      scrollbar_button_hover_color=BORDER2)
        if h > 0:
            kwargs["height"] = h
        tb = ctk.CTkTextbox(parent, **kwargs)  # type: ignore[arg-type]
        if h < 0 or expand:
            tb.pack(fill="both", expand=True, padx=padx, pady=(0, SPACE_S))
        else:
            tb.pack(fill="x", padx=padx, pady=(0, SPACE_S))
        return tb

    def _set_textbox(self: ECGApp, widget, text: str,
                     tsv: "str | None" = None) -> None:
        """Update a CTkTextbox with *text* and optionally store *tsv* for
        'Copy as TSV' (clipboard format that pastes into Excel cell-by-cell).
        """
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        wid = id(widget)
        if tsv is not None:
            self._tsv_store[wid] = tsv
            self._bind_copy_menu(widget)
        elif wid in self._tsv_store:
            del self._tsv_store[wid]

    def _bind_copy_menu(self: ECGApp, widget) -> None:
        """Attach a right-click context menu with copy options to *widget*."""
        def _show_menu(event):
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(
                label="📋  Copy as TSV  (paste into Excel)",
                command=lambda: self._copy_tsv(widget))
            menu.add_command(
                label="📄  Copy as plain text",
                command=lambda: self._copy_plain(widget))
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
        # Bind to the underlying tk widget inside CTkTextbox
        try:
            inner = widget._textbox  # CTkTextbox wraps a tk.Text
            inner.bind("<Button-3>", _show_menu, add=True)
        except AttributeError:
            widget.bind("<Button-3>", _show_menu, add=True)

    def _copy_tsv(self: ECGApp, widget) -> None:
        """Copy the TSV version of this widget's data to the clipboard."""
        tsv = self._tsv_store.get(id(widget))
        if not tsv:
            self._copy_plain(widget)
            return
        self.clipboard_clear()
        self.clipboard_append(tsv)
        self._set_status("Copied as TSV — paste into Excel ✓", GREEN)

    def _copy_plain(self: ECGApp, widget) -> None:
        """Copy the plain-text content of *widget* to the clipboard."""
        try:
            text = widget.get("1.0", "end")
        except Exception:
            text = ""
        self.clipboard_clear()
        self.clipboard_append(text)
        self._set_status("Copied as plain text ✓", GREEN)

    def _set_status(self: ECGApp, text: str, color: str = MUTED) -> None:
        # Remembered so hover tips (below) can restore this exact message
        # on <Leave> instead of blanking the status bar.
        self._status_text, self._status_color = text, color
        self._status_display_raw, self._status_display_color = text, color
        self._apply_status_display(text, color)

    def _bind_hover_tip(self: ECGApp, widget, text: str, color: str = MUTED) -> None:
        """Show *text* in the status bar on hover; restore the previous
        status message on leave.

        Thin wrapper around the existing hand-rolled <Enter>/<Leave> ->
        status-bar hover pattern (previously duplicated once, for the Free
        Placement '?' hint). Reuses the persistent status-bar label already
        on screen -- not a floating tooltip widget. Restoring rather than
        blanking on <Leave> matters because this label is also the app's
        only non-modal feedback channel: hovering e.g. a nav arrow used to
        permanently erase whatever error/result message was showing.
        """
        def _enter(_e):
            self._status_display_raw, self._status_display_color = text, color
            self._apply_status_display(text, color)

        def _leave(_e):
            self._status_display_raw = self._status_text
            self._status_display_color = self._status_color
            self._apply_status_display(self._status_text, self._status_color)

        widget.bind("<Enter>", _enter)
        widget.bind("<Leave>", _leave)

    def _fit_status_text(self: ECGApp, text: str) -> str:
        """Truncate *text* with an ellipsis so it fits on one line of
        lbl_status, never wrapping.  See lbl_status's construction in
        _build_toolbar() for why: a wrapped 2-3 line status message used
        to spill over and hide the Parameters/Theme/"Stats ⟩" buttons.
        """
        try:
            avail = self.lbl_status.winfo_width()
            if avail < 40:  # not laid out yet -- a sane idle-width guess
                avail = 280
            avail -= 6  # small safety margin
            if self._status_font.measure(text) <= avail:
                return text
            ellipsis = "…"
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._status_font.measure(text[:mid] + ellipsis) <= avail:
                    lo = mid
                else:
                    hi = mid - 1
            return text[:lo].rstrip() + ellipsis
        except Exception:
            return text

    def _apply_status_display(self: ECGApp, text: str, color: str) -> None:
        """Fit+push *text*/*color* onto lbl_status, skipping the .configure()
        call entirely when nothing would actually change. Guards against a
        layout feedback loop: this is also called from a <Configure>
        handler on the toolbar row (see _refresh_status_text), and an
        unconditional .configure() on every callback -- even a no-op one --
        risks re-triggering <Configure> and looping."""
        fitted = self._fit_status_text(text)
        if (self.lbl_status.cget("text") == fitted
                and self.lbl_status.cget("text_color") == color):
            return
        self.lbl_status.configure(text=fitted, text_color=color)

    def _refresh_status_text(self: ECGApp) -> None:
        """Re-fit the currently-displayed status/hover text -- bound to the
        toolbar row's <Configure> so a window resize (which changes how
        much room lbl_status has) can't leave a stale, too-long truncation
        (or an un-truncated line) in place."""
        text = getattr(self, "_status_display_raw", self._status_text)
        color = getattr(self, "_status_display_color", self._status_color)
        self._apply_status_display(text, color)

    def _widget_float(self: ECGApp, widget: "ctk.CTkEntry", default: float = 0.0) -> float:
        """Read a float from a CTk Entry widget.  Returns *default* on any error.

        Use this when the input is definitely a widget.  Misspelled attribute
        names produce an AttributeError rather than returning a silent 0.0,
        making bugs in callers visible during development.
        """
        try:
            v = float(widget.get())
            if not np.isfinite(v):
                log.debug("_widget_float: non-finite value from %r — using %s", widget, default)
                return default
            return v
        except (ValueError, TypeError, AttributeError) as exc:
            log.debug("_widget_float: could not parse widget %r — using %s: %s",
                      widget, default, exc)
            return default

    def _safe_float(self: ECGApp, widget_or_value: "Any", default: float = 0.0) -> float:
        """Safely extract a float from a CTk widget or a raw scalar value.

        Kept for backward compatibility.  Prefer _widget_float() when the
        caller knows the input is a widget.
        Returns *default* if the value is missing, empty, or non-numeric.
        """
        try:
            raw = widget_or_value.get() if hasattr(widget_or_value, "get") else widget_or_value
            v   = float(raw)
            if not np.isfinite(v):
                log.debug("_safe_float: non-finite value %r — using default %s", raw, default)
                return default
            return v
        except (ValueError, TypeError) as exc:
            log.debug("_safe_float: could not parse %r — using default %s: %s",
                      widget_or_value, default, exc)
            return default

    @staticmethod
    def _safe_df_val(df: "pd.DataFrame | None", col: str, decimals: int = 3) -> str:
        """Return a formatted scalar from a DataFrame cell, or '—' on any error."""
        try:
            if df is None or col not in df.columns:
                return "—"
            v = float(df[col].values[0])
            return f"{v:.{decimals}f}" if np.isfinite(v) else "—"
        except Exception:
            return "—"

    @staticmethod
    def _df_to_tsv(df: "pd.DataFrame | None") -> str:
        """Convert a DataFrame to tab-separated values for Excel paste.

        First row = header (column names), subsequent rows = values.
        Pastes into Excel so each metric is in its own column.
        """
        if df is None or df.empty:
            return ""
        rows = ["	".join(str(c).replace("HRV_", "") for c in df.columns)]
        for _, row in df.iterrows():
            cells = []
            for v in row:
                try:
                    fv = float(v)
                    cells.append(f"{fv:.5g}" if np.isfinite(fv) else "")
                except (TypeError, ValueError):
                    cells.append(str(v) if v is not None else "")
            rows.append("	".join(cells))
        return "\n".join(rows)

    # Display (label, unit) for the non-linear metrics researchers actually
    # cite; NeuroKit's remaining ~20 columns (PIP, IALS, PSS, PAS, GI, SI,
    # AI, PI, C1d, C1a, SD1d, ...) are specialist Poincaré-subdivision
    # indices with no settled one-line gloss -- those fall through to the
    # generic underscore-to-space cleanup below rather than a guessed label.
    _NL_DISPLAY = {
        "SD1":          ("Poincaré SD1 (short-term)", "ms"),
        "SD2":          ("Poincaré SD2 (long-term)",  "ms"),
        "SD1SD2":       ("SD1 / SD2", ""),
        "CSI":          ("Cardiac Sympathetic Index", ""),
        "CVI":          ("Cardiac Vagal Index", ""),
        "CSI_Modified": ("Modified CSI", ""),
        "SampEn":       ("Sample entropy", ""),
        "ApEn":         ("Approximate entropy", ""),
        "DFA_alpha1":   ("DFA α1 (short-term)", ""),
        "DFA_alpha2":   ("DFA α2 (long-term)", ""),
    }

    @staticmethod
    def _df_to_text(df: "pd.DataFrame | None",
                     display_map: "Optional[dict[str, tuple[str, str]]]" = None) -> str:
        """Render every finite numeric column of a NeuroKit2 HRV DataFrame as readable text.

        Format uses dot-leaders for visual alignment without requiring a fixed-width font:
            SDNN ............  12.35
            RMSSD ...........   8.27

        *display_map* optionally maps a column's HRV_-stripped name to a
        (label, unit) pair; columns not in the map fall back to the raw
        name with underscores turned to spaces (still better than the
        literal NeuroKit identifier).
        """
        if df is None or df.empty:
            return "  (not computed)"
        display_map = display_map or {}
        rows = []
        for col in df.columns:
            try:
                v = float(df[col].values[0])
                if np.isfinite(v):
                    name = col.replace("HRV_", "")
                    label, unit = display_map.get(name, (name.replace("_", " "), ""))
                    disp = f"{label} ({unit})" if unit else label
                    # Adaptive precision: integers → 0 dp, small values → 4 dp
                    if abs(v) >= 100:
                        fmt = f"{v:.1f}"
                    elif abs(v) >= 1:
                        fmt = f"{v:.3f}"
                    else:
                        fmt = f"{v:.5f}"
                    # Dot-leader padding to column 30
                    dots = "·" * max(2, 30 - len(disp))
                    rows.append(f"  {disp} {dots}  {fmt:>10}")
            except Exception as exc:
                log.debug("_df_to_text skip '%s': %s", col, exc)
        return "\n".join(rows) or "  (no finite values)"
