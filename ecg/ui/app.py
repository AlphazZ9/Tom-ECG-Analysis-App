# -*- coding: utf-8 -*-
"""
ecg.ui.app
----------
ECGApp -- the main application window (CustomTkinter CTk).
Imports all other ecg.* modules and wires them together.
"""
from __future__ import annotations

import logging
import os

import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from tkinter import messagebox

import matplotlib
import matplotlib.figure
import matplotlib.ticker
if __name__ == "__main__":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
# numpy 2.x may expose only trapezoid; ensure np.trapz alias exists.
if not hasattr(np, "trapz"):
    if hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    else:
        def _trapz(y, x=None, dx=1.0, axis=-1):
            y_arr = np.asanyarray(y)
            if x is None:
                n = y_arr.shape[axis]
                x = np.arange(n, dtype=float) * dx
            else:
                x = np.asanyarray(x)
            # Broadcasting rules apply; use numpy.diff and slicing
            slice_all = [slice(None)] * y_arr.ndim
            slice1 = slice_all.copy(); slice2 = slice_all.copy()
            slice1[axis] = slice(1, None)
            slice2[axis] = slice(0, -1)
            dx_arr = np.diff(x)
            y_avg = (y_arr[tuple(slice1)] + y_arr[tuple(slice2)]) / 2.0
            return np.sum(y_avg * dx_arr, axis=axis)
        np.trapz = _trapz  # type: ignore[attr-defined]

from PIL import Image, ImageTk
from ecg.io.session import session_exists
from ecg.ui.theme import (
    THEME, apply_theme_config, NK_AVAILABLE, APP_ICON_PATH, PANEL, BLUE, GREEN,
)
from ecg.ui.dialogs import ThemeDialog

from ecg.ui.app_state_props import StateProps
from ecg.ui.app_sidebar import SidebarMixin
from ecg.ui.app_toolbar import ToolbarMixin
from ecg.ui.app_right_panel import RightPanelMixin
from ecg.ui.app_async import AsyncMixin
from ecg.ui.app_helpers import HelpersMixin
from ecg.ui.app_delegates import DelegatesMixin
from ecg.ui.params_dialog import ParamsDialogMixin
from ecg.ui.compare_segments import CompareSegmentsMixin
from ecg.ui.tab_detection import TabDetectionMixin
from ecg.ui.tab_hrv import TabHrvMixin
from ecg.ui.tab_time_course import TabTimeCourseMixin
from ecg.ui.tab_events import TabEventsMixin
from ecg.ui.tab_morphology import TabMorphologyMixin
from ecg.ui.tab_report import TabReportMixin
from ecg.ui.tab_keys import TAB_ORDER
from ecg.ui.tabbar import TabBar
from ecg.ui.context_strip import ContextStripMixin

log = logging.getLogger("ecg")


class ECGApp(
    StateProps,
    SidebarMixin,
    ToolbarMixin,
    RightPanelMixin,
    AsyncMixin,
    HelpersMixin,
    DelegatesMixin,
    ParamsDialogMixin,
    CompareSegmentsMixin,
    TabDetectionMixin,
    TabHrvMixin,
    TabTimeCourseMixin,
    TabEventsMixin,
    TabMorphologyMixin,
    TabReportMixin,
    ContextStripMixin,
    ctk.CTk,
):

    def __init__(self) -> None:
        super().__init__()
        self.title("ECG Analysis")
        self.geometry("1920x1080")
        # Toolbar width floor: below ~1600px the right-hand button cluster
        # (quality gauge/badge, Parameters, Theme, dark/light, "Stats ⟩"
        # panel toggle -- packed in _build_toolbar()) runs out of cavity and
        # Tk's pack manager unmaps whichever of them no longer fits, silently
        # dropping them instead of shrinking/overlapping. Measured: at the
        # old 1200 floor, Parameters AND the right-panel toggle were both
        # completely unmapped (not just crowded); everything needs ~1600px
        # to render at full size. 1650 keeps a small safety margin.
        self.minsize(1650, 750)
        # Open in fullscreen/maximized state
        self.after(100, lambda: self.state("zoomed"))  # Windows fullscreen
        self.configure(fg_color=PANEL)

        # Forward declarations for widgets built during _build() — lets Pylance
        # know these attributes exist even though they're assigned later.
        # (declaration moved to _init_state)
        self.ent_subject:  ctk.CTkEntry
        self.ent_fs:       ctk.CTkEntry
        self.ent_epoch:    ctk.CTkEntry
        self.ent_overlap:  ctk.CTkEntry
        self.txt_td:       ctk.CTkTextbox
        self.txt_fd:       ctk.CTkTextbox

        self._load_icon()
        self._init_state()
        self._build()
        self._bind_keyboard_shortcuts()
        self._start_shell_tick()

    # ─── Startup helpers ──────────────────────────────────────

    def _on_window_resize(self, event) -> None:
        """Adjust sidebar width proportionally when the window is resized.

        Debounced to 80 ms so rapid drag-resize events don't flood the layout
        engine with redundant configure calls.
        """
        if event.widget is not self:
            return
        # Own debounce handle -- this used to reuse _thr_debounce_id, the
        # same field the threshold-slider drag debounce (detection_
        # controller.py) owns, so a resize landing mid-drag (or vice versa)
        # would silently cancel the OTHER feature's pending job instead of
        # both being honored.
        if self._resize_debounce_id is not None:
            try:
                self.after_cancel(self._resize_debounce_id)
            except Exception:
                pass
        new_w = max(220, min(340, int(event.width * 0.20)))
        self._resize_debounce_id = self.after(
            80, lambda w=new_w: self._apply_resize(w)
        )

    def _apply_resize(self, new_w: int) -> None:
        """Apply the debounced sidebar width update."""
        self._resize_debounce_id = None
        try:
            self.sidebar.configure(width=new_w)
        except Exception as e:
            log.debug("sidebar.configure width failed: %s", e)

    def _load_icon(self) -> None:
        if not APP_ICON_PATH:
            return
        try:
            # Pillow 9+ uses Image.Resampling.LANCZOS; older versions use Image.LANCZOS
            _resample = (getattr(getattr(Image, "Resampling", None), "LANCZOS", None)
                         or getattr(Image, "LANCZOS", Image.BICUBIC))  # type: ignore[attr-defined]
            img = Image.open(APP_ICON_PATH).resize((256, 256), _resample)
            # PhotoImage is accepted at runtime; Tkinter stubs are incomplete
            self.iconphoto(True, ImageTk.PhotoImage(img))  # type: ignore[arg-type]
        except Exception as exc:
            log.debug("App icon not loaded: %s", exc)

    # ════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ════════════════════════════════════════════════════════

    def _build(self) -> None:
        self.update_idletasks()
        # Packed on self, side="top", BEFORE the sidebar -- this is what makes
        # it span the full window width rather than being confined to main's
        # width the way the old KPI bar was (sidebar's own side="left" would
        # otherwise claim the left edge first).
        self._build_toolbar()
        w = self.winfo_width() or 1480
        _sidebar_w = max(220, min(340, int(w * 0.20)))
        self.sidebar = ctk.CTkFrame(self, width=_sidebar_w, fg_color=PANEL, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.bind("<Configure>", self._on_window_resize)
        # Keyboard navigation shortcuts (active when focus is on the main
        # window -- guarded against firing while the focus is actually in a
        # text entry, where Left/Right must move the text cursor instead).
        self.bind("<Left>",  lambda e: None if self._typing_in_entry() else self._kb_navigate(-1))
        self.bind("<Right>", lambda e: None if self._typing_in_entry() else self._kb_navigate(+1))
        self._build_sidebar()
        # Apply the default state immediately so FILTER SETTINGS widgets are
        # greyed out at startup (Filtering is OFF by default).
        self.after(50, self._on_filtering_toggle)

        # Packed side="right" BEFORE main, for the same reason the sidebar is
        # packed before main -- main's expand=True must not claim the space
        # first.
        self._build_right_panel()

        # Stored as self.main (not a local var) so _toggle_left_panel()/
        # _toggle_right_panel() can re-pack a collapsed panel with
        # before=self.main -- see those methods for why that's required.
        # fg_color=PANEL (not BG) so the workspace reads as the same grey as
        # the sidebar/right_panel it sits between, instead of a subtly
        # different shade that shows as a seam down both edges.
        self.main = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        self.main.pack(side="left", fill="both", expand=True)
        self._build_context_strip(self.main)
        self._build_tabs(self.main)
        # Global keyboard shortcuts -- guarded the same way as Left/Right
        # above: Ctrl+Z is the near-universal native-undo instinct, so
        # without this guard typing in a sidebar field (Project name,
        # Subject ID, Channel) and hitting Ctrl+Z out of habit silently
        # undoes a peak edit instead of the text.
        self.bind("<Control-z>", lambda e: None if self._typing_in_entry() else self._undo_edit(e))
        self.bind("<Control-Z>", lambda e: None if self._typing_in_entry() else self._undo_edit(e))
        self.bind("<Control-y>", lambda e: None if self._typing_in_entry() else self._redo_edit(e))
        self.bind("<Control-Y>", lambda e: None if self._typing_in_entry() else self._redo_edit(e))

        # Apply persisted collapse state (defaults False -- both expanded,
        # matching how the toggle buttons were just constructed). Route
        # through the toggle methods themselves rather than duplicating
        # their pack_forget()/button-text logic here -- each toggle flips
        # the UIState bool from its current (just-initialized "expanded")
        # value, so only fire it when the persisted value says "collapsed".
        if self.ui.left_panel_collapsed:
            self.ui.left_panel_collapsed = False
            self._toggle_left_panel()
        if self.ui.right_panel_collapsed:
            self.ui.right_panel_collapsed = False
            self._toggle_right_panel()

    # ─── Tabs ─────────────────────────────────────────────────

    def _build_tabs(self, parent) -> None:
        self.tabs = TabBar(parent, command=self._on_main_tab_changed)
        self.tabs.pack(fill="both", expand=True)
        for name in TAB_ORDER:
            self.tabs.add(name)
        self._build_status_label(self.tabs.bar)

        self._build_tab_detection()
        self._build_tab_hrv_unified()
        self._build_tab_time_course()
        self._build_tab_morphology()
        self._build_tab_arrhythmias()
        self._build_tab_report()
        self._bind_tab_state_hints()

    # ════════════════════════════════════════════════════════
    #  APPEARANCE / THEME
    # ════════════════════════════════════════════════════════

    def _open_theme_dialog(self) -> None:
        """Open the appearance settings dialog."""
        dlg = ThemeDialog(self)
        self.wait_window(dlg)

    def _snapshot_ui_state(self) -> dict:
        """Capture all editable widget values before a UI rebuild."""
        return self.session_ctrl.snapshot_ui_state()

    def _restore_ui_state(self, s: dict) -> None:
        """Restore widget values captured before a UI rebuild."""
        self.session_ctrl.restore_ui_state(s)

    def _park_focus(self) -> None:
        """Move the focus to the window. Call before changing the appearance mode
        when a rebuild follows: CTk restores the focus it finds there one tick
        later (on Windows), which fails on a widget the rebuild has destroyed."""
        self.focus_set()

    def _rebuild_ui(self) -> None:
        """Destroy and recreate the entire widget tree with the current theme.

        All application data (signal, results, peaks) is preserved.
        Widget values (threshold, filter settings, etc.) are snapshot and
        restored.  Matplotlib figures are closed before destruction to avoid
        memory leaks.
        """
        # Capture widget values
        ui_state = self._snapshot_ui_state()

        # Close all matplotlib figures before destroying their canvas widgets
        for slot in self._slots.values():
            try:
                plt.close(slot.fig)
            except Exception as _exc:
                log.debug("plt.close(slot.fig) failed: %s", _exc, exc_info=True)

        # Destroy the existing UI
        for child in self.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass

        # Reset widget registries
        self._slots = {}
        self._kpi   = {}

        # Rebuild from scratch using updated globals
        self._build()

        # Restore widget values
        self._restore_ui_state(ui_state)

        # Restore file info label
        if self._filepath:
            if self.lbl_file is not None:
                self.lbl_file.configure(  # type: ignore[union-attr]
                    text=os.path.basename(self._filepath), text_color=GREEN)
        if self._rpeaks_ok is not None:
            n = len(self._rpeaks_ok)
            if self.lbl_npeaks is not None:
                self.lbl_npeaks.configure(text=f"Peaks detected: {n}", text_color=GREEN)  # type: ignore[union-attr]

        # Restore session / template labels. A recording without a template
        # yet keeps none: it is created at the next Intervals run / Landmarks
        # opening (recording_wave_template, which reports an unused saved
        # default); the label shows the one that would be used.
        if not self._session_dirty:
            self._session_dirty = False
        has_session = bool(
            self._filepath and session_exists(self._filepath))
        self._update_session_ui(has_session=has_session)

        # Enable action buttons if data is ready
        if self._signal_flt is not None and self.btn_save_session is not None:
            self._set_save_session_enabled(True)
        if self._results is not None:
            self._set_result_btns_enabled(
                True, ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl"))
            if self.btn_save_session is not None:
                self._set_save_session_enabled(True)

        # Repaint signal and results
        apply_theme_config(THEME)
        if self._signal_flt is not None:
            self._draw_detail(self._nav_pos)
        if self._results is not None:
            self.after(80, self._draw_all_results)
            pass  # interpretation removed
        self._update_kpis()
        # The rebuilt analysis-window / Frequency / Non-linear labels start
        # at their defaults: re-derive them from the results, so the orange
        # RR-exclusion line and a withheld spectrum's reason survive a theme
        # toggle (no-op without results).
        self.analysis_ctrl.refresh_result_labels()
        # Time course: results need only peaks (not core results), and their
        # rebuilt views start blank -- redraw them (no-op without a result).
        self.analysis_ctrl.show_epochs()
        self.analysis_ctrl.show_rolling()
        self.analysis_ctrl.show_autonomic()
        self.event_ctrl.show()
        # Events: cards, ribbon and readout of the last classification; the
        # rebuilt tab starts blank, and the zoom on the selected event with it.
        self.analysis_ctrl.show_events()
        # Pre-existing gap, not introduced by the Phase 1 top-bar rework:
        # the correlation badge was never repainted after a theme toggle
        # before, so it silently went blank on every _rebuild_ui(). Cheap
        # fix while already touching this exact code path.
        # Same class of gap: the Detection-tab toolbar badges and the right
        # panel's ANNOTATIONS summary both need a refresh after a full
        # rebuild -- both widgets are guaranteed valid here since _build()
        # (called above) has already fully rebuilt every tab and panel.
        self._update_ann_count()
        self._update_pacing_count()
        self._refresh_shell()

    # ════════════════════════════════════════════════════════
    #  11. LIVE THEME TOGGLE (no restart needed)
    # ════════════════════════════════════════════════════════

    def _toggle_dark_live(self) -> None:
        """Switch Light <-> Dark and rebuild the UI without restarting.

        Goes through apply_preset() (not a raw THEME.is_dark flip) so colors
        and is_dark change together -- a bare flip left CTk's native dark
        chrome sitting over the light preset's actual colors.
        """
        from ecg.ui.theme import THEME, apply_theme_config
        THEME.apply_preset("Dark" if not THEME.is_dark else "Light")
        self._park_focus()
        apply_theme_config(THEME)
        THEME.save()
        ui_state = self._snapshot_ui_state()
        self._rebuild_ui()
        self._restore_ui_state(ui_state)
        mode = "Dark" if THEME.is_dark else "Light"
        self._set_status(f"Theme: {mode} mode applied", BLUE)


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    if not NK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing dependency",
            "Install NeuroKit2:\n  pip install neurokit2",
        )
        root.destroy()
    else:
        app = ECGApp()
        app.mainloop()
