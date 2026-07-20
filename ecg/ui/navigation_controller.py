# -*- coding: utf-8 -*-
"""
navigation_controller.py
-------------------------
NavigationController -- moving the detail/arrhythmia view position around
(prev/next window, big jumps, reset, goto, scroll-to-zoom, arrhythmia-event
navigation). Pure view-position bookkeeping; drawing itself still happens
through ECGApp._draw_detail()/_draw_arr_detail() (moving to PlotController
in a later stage).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ecg.ui.theme import BLUE_DARK, BORDER, TEXT

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class NavigationController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app
        self._overview_dragging = False   # plain instance attr, mirrors wave_editor.py's _drag_key

    def kb_navigate(self, direction: int) -> None:
        """Keyboard left/right arrow navigation — only active on Detection tab."""
        app = self.app
        try:
            if app.tabs.get() != "📈 Detection":
                return
        except Exception:
            return
        self.navigate(direction)

    def navigate(self, direction: int) -> None:
        """Shift the detail view left/right by 80 % of the window width."""
        app = self.app
        if app.signal.time is None or len(app.signal.time) == 0:
            return
        try:
            win = float(app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except (ValueError, TypeError):
            win = 10.0
        max_start  = float(app.signal.time[-1]) - win
        app.ui.nav_pos = max(0.0, min(max_start, app.ui.nav_pos + direction * win * 0.8))
        self.sync_nav_pos_entry()
        app._draw_detail()

    def navigate_big(self, direction: int) -> None:
        """Jump by 10× the current window width."""
        app = self.app
        if app.signal.time is None or len(app.signal.time) == 0:
            return
        try:
            win = float(app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except (ValueError, TypeError):
            win = 10.0
        max_start = float(app.signal.time[-1]) - win
        app.ui.nav_pos = max(0.0, min(max_start, app.ui.nav_pos + direction * win * 10.0))
        self.sync_nav_pos_entry()
        app._draw_detail()

    def nav_reset(self) -> None:
        app = self.app
        app.ui.nav_pos = 0.0
        self.sync_nav_pos_entry()
        app._draw_detail()

    def nav_end(self) -> None:
        """Jump to the end of the signal."""
        app = self.app
        if app.signal.time is None or len(app.signal.time) == 0:
            return
        try:
            win = float(app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except (ValueError, TypeError):
            win = 10.0
        app.ui.nav_pos = max(0.0, float(app.signal.time[-1]) - win)
        self.sync_nav_pos_entry()
        app._draw_detail()

    def nav_goto(self) -> None:
        """Jump to the time entered in the position field."""
        app = self.app
        if app.signal.time is None or app.ent_nav_pos is None:
            return
        try:
            t_target = float(app.ent_nav_pos.get().replace(",", "."))  # type: ignore[union-attr]
        except (ValueError, AttributeError):
            return
        try:
            win = float(app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except (ValueError, TypeError):
            win = 10.0
        max_start = float(app.signal.time[-1]) - win
        app.ui.nav_pos = max(0.0, min(max_start, t_target))
        self.sync_nav_pos_entry()
        app._draw_detail()

    def sync_nav_pos_entry(self) -> None:
        """Update the position entry widget to reflect ui.nav_pos."""
        app = self.app
        if app.ent_nav_pos is None:
            return
        try:
            app.ent_nav_pos.delete(0, "end")  # type: ignore[union-attr]
            app.ent_nav_pos.insert(0, f"{app.ui.nav_pos:.3f}")  # type: ignore[union-attr]
        except Exception as e:
            log.debug("ent_nav_pos update failed: %s", e)

    def arr_navigate(self, direction: int) -> None:
        app = self.app
        if app.signal.time is None:
            return
        step = app.analysis.arr_win * 0.7
        max_t = max(0.0, float(app.signal.time[-1]) - app.analysis.arr_win)
        app.analysis.arr_nav_pos = max(0.0, min(max_t, app.analysis.arr_nav_pos + direction * step))
        app._draw_arr_detail()

    def select_arrhythmia_event(self, idx: int) -> None:
        """Highlight selected card and load the ECG window for this event."""
        app = self.app
        events = app.analysis.arrhythmia_events
        if not events or idx < 0 or idx >= len(events):
            return

        # Highlight selected card, reset others
        for i, w in enumerate(app._arr_card_widgets):
            try:
                w.configure(border_color=BLUE_DARK if i == idx else BORDER)
            except Exception as e:
                log.debug("card border_color configure failed: %s", e)

        app.analysis.arr_selected_idx = idx
        ev = events[idx]

        # Compute window: centre on episode with ±1.5 s padding
        try:
            win = float(app.ent_arr_win.get())
        except Exception:
            win = 3.0
        app.analysis.arr_win = max(0.5, win)

        padding = max(app.analysis.arr_win * 0.25, 0.5)
        ep_dur  = max(ev.duration_s, 0.1)
        t_centre = ev.t_start + ep_dur / 2
        t_start  = max(0.0, t_centre - app.analysis.arr_win / 2)
        if app.signal.time is not None:
            t_start = min(t_start, max(0.0, float(app.signal.time[-1]) - app.analysis.arr_win))
        # Expand window if episode is longer than current win
        if ep_dur + 2 * padding > app.analysis.arr_win:
            app.analysis.arr_win = min(ep_dur + 2 * padding, 30.0)
            t_start = max(0.0, ev.t_start - padding)

        app.analysis.arr_nav_pos = t_start
        app._draw_arr_detail()

        # Update title bar
        sev_icon = {"alert": "🔴", "warning": "🟠", "info": "🔵"}.get(ev.severity, "·")
        app.lbl_arr_event_title.configure(  # type: ignore[union-attr]
            text=f"{sev_icon}  #{idx+1}  {ev.kind.replace('_',' ').title()}"
                 f"  —  {ev.t_start:.2f} s → {ev.t_end:.2f} s  ({ev.hr_mean:.0f} bpm)",
            text_color=TEXT,
        )

    def on_overview_click(self, event) -> None:
        """Click-to-navigate + arm drag-to-scrub on the minimap strip.

        Centers the window on the clicked time -- matches the existing
        convention in PlotController._on_rr_click (RR/HR tachogram
        click-to-navigate), which already does `t_nav - win/2`: a click
        always brings the point of interest into context rather than
        pinning it to the window's left edge.
        """
        app = self.app
        if event.button != 1 or event.xdata is None or event.inaxes is None:
            return
        if app.signal.time is None or len(app.signal.time) == 0:
            return
        self._overview_dragging = True
        self._set_overview_nav_pos(float(event.xdata))
        app._draw_detail()

    def on_overview_motion(self, event) -> None:
        """Drag-to-scrub, debounced ~30ms.

        draw_detail() re-slices the real signal and recomputes peak markers
        over the visible window on every call (unlike wave_editor's cheap
        fixed-template redraw, which is intentionally un-throttled) -- at
        mouse-move fire rates that gap is noticeable on long/high-fs
        recordings, so PRESS/RELEASE redraw synchronously (single discrete
        actions) while MOTION is throttled, mirroring the existing
        thr_debounce_id/hover_after_id after()-based debounce pattern
        already used elsewhere in this codebase.
        """
        app = self.app
        if not self._overview_dragging or event.xdata is None:
            return
        if app.signal.time is None or len(app.signal.time) == 0:
            return
        self._set_overview_nav_pos(float(event.xdata))
        if app.ui.ov_drag_after_id is not None:
            app.after_cancel(app.ui.ov_drag_after_id)
        app.ui.ov_drag_after_id = app.after(30, self._flush_overview_drag)

    def on_overview_release(self, event) -> None:
        """End drag; guarantee one final, un-debounced redraw."""
        app = self.app
        self._overview_dragging = False
        if app.ui.ov_drag_after_id is not None:
            app.after_cancel(app.ui.ov_drag_after_id)
            app.ui.ov_drag_after_id = None
            app._draw_detail()

    def _flush_overview_drag(self) -> None:
        self.app.ui.ov_drag_after_id = None
        self.app._draw_detail()

    def _flush_scroll_sync(self) -> None:
        self.app.ui.scroll_sync_after_id = None
        self.app._draw_detail(self.app.ui.nav_pos)

    def _set_overview_nav_pos(self, t_click: float) -> None:
        """Shared clamp logic for click and drag (mirrors nav_goto's clamp)."""
        app = self.app
        try:
            win = float(app.ent_window.get())
            if not (0 < win < 1e6):
                win = 10.0
        except (ValueError, TypeError):
            win = 10.0
        sig_dur   = float(app.signal.time[-1])
        max_start = max(0.0, sig_dur - win)
        app.ui.nav_pos = max(0.0, min(max_start, t_click - win / 2.0))
        self.sync_nav_pos_entry()

    def on_overview_scroll(self, event) -> None:
        """Stub — scroll-to-zoom on the minimap is out of scope for Phase 3b."""
        pass

    def on_detail_scroll(self, event) -> None:
        """Zoom the detail view's x-axis in/out centred on the cursor position.

        Each scroll tick zooms by a factor of 1.25 (in) or 0.8 (out).
        After zooming, ``ui.nav_pos`` and the window-entry widget are updated
        to reflect the new visible range.

        The y-axis is intentionally unchanged — vertical zoom is handled by
        the matplotlib toolbar's Zoom-to-rectangle tool.
        """
        app = self.app
        if event.xdata is None or app.signal.time is None:
            return

        ax = event.inaxes
        if ax is None:
            return

        x_min, x_max = ax.get_xlim()
        cur_win  = x_max - x_min
        cursor_x = float(event.xdata)
        factor   = 0.8 if event.button == "up" else 1.25   # up = zoom in

        new_win  = max(0.5, min(float(app.signal.time[-1]), cur_win * factor))
        # Keep the point under the cursor stationary
        frac     = (cursor_x - x_min) / max(cur_win, 1e-6)
        new_xmin = cursor_x - frac * new_win
        new_xmax = new_xmin + new_win
        # Clamp to signal bounds
        new_xmin = max(0.0, new_xmin)
        new_xmax = min(float(app.signal.time[-1]), new_xmin + new_win)
        new_xmin = new_xmax - new_win   # re-apply after xmax clamp

        # Update nav state so subsequent arrow navigation is coherent
        app.ui.nav_pos = max(0.0, new_xmin)
        self.sync_nav_pos_entry()

        # Update the window entry box (fire-and-forget; it is only read on the next draw)
        try:
            app.ent_window.delete(0, "end")
            app.ent_window.insert(0, f"{new_win:.2f}")
        except Exception as _exc:
            log.debug("ent_window update failed: %s", _exc, exc_info=True)

        ax.set_xlim(new_xmin, new_xmax)
        app._slots["detail"].canvas.draw_idle()

        # The detail plot itself is updated instantly above (cheap axes-only
        # redraw, since scroll ticks can fire in rapid bursts) -- but that
        # bypasses app._draw_detail()'s fan-out, so the minimap and RR/HR
        # strip silently desync on scroll-zoom unless resynced separately.
        # Debounced (mirrors ov_drag_after_id's minimap-drag debounce) so a
        # burst of scroll ticks only triggers one full resync once scrolling
        # settles, not one per tick.
        if app.ui.scroll_sync_after_id is not None:
            app.after_cancel(app.ui.scroll_sync_after_id)
        app.ui.scroll_sync_after_id = app.after(120, self._flush_scroll_sync)
