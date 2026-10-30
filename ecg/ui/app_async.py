# -*- coding: utf-8 -*-
"""
ecg.ui.app_async
----------------
ECGApp background-job infrastructure: progress, _start_async, error dialog.
Mixin of ECGApp (see ecg/ui/app.py).
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Callable, TYPE_CHECKING
import customtkinter as ctk  # type: ignore[import-untyped]
import tkinter as tk
from ecg.ui.theme import (
    PANEL, BORDER, BORDER2, TEXT, MUTED, RED, ORANGE, SPACE_XS, SPACE_S,
    SPACE_M, FONT_LABEL, FONT_SMALL, FONT_BTN_PRIMARY, FONT_BTN_SEC, FONT_MICRO,
)

# Mixin of ECGApp: `self: ECGApp` shows pyright the whole app surface, so a
# typo in self.xxx is reported. It is deliberately not a supertype of this
# class, which is the one check switched off here.
# pyright: reportGeneralTypeIssues=false
if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class AsyncMixin:
    """ECGApp background-job infrastructure: progress, _start_async, error dialog."""

    _async_start_time: float | None

    def _start_async_result(
        self: ECGApp,
        button: ctk.CTkButton,
        busy_label: str,
        worker: Callable[[], Any],
        on_done: Callable[[Any], None],
        ) -> None:
        """Convenience wrapper: like _start_async but forwards the worker return value.

        Calls _start_async with pass_result=True so that on_done receives the
        value returned by worker().  All threading, progress, and error handling
        are identical to _start_async.
        """
        self._start_async(
            button,
            busy_label,
            "",          # no separate status message — caller sets its own
            worker,
            on_done,
            pass_result=True,
        )

    # ── Async helper ──────────────────────────────────────────

    def _set_progress(self: ECGApp, pct: int, msg: str) -> None:
        """Update the deterministic progress bar + stage label (main thread only).

        Does NOT call update_idletasks() — that would block the event loop and
        cause the UI to freeze mid-analysis.  The progress bar is updated
        asynchronously by the normal Tk event loop.
        """
        frac = max(0.0, min(1.0, pct / 100.0))
        self.progress.set(frac)
        self._last_prog_pct = pct
        self._last_prog_msg = msg
        self._render_progress_label()

    def _render_progress_label(self: ECGApp) -> None:
        """Render lbl_progress from the last known (pct, msg) + a time-left estimate.

        Called on every real progress_cb checkpoint (via _set_progress) --
        with the fake heartbeat pulse removed, this no longer ticks between
        checkpoints, so the ETA shown is the best estimate as of the last
        real checkpoint, not a live-updating countdown.
        """
        pct = getattr(self, "_last_prog_pct", 0)
        msg = getattr(self, "_last_prog_msg", "")
        start = getattr(self, "_async_start_time", None)
        suffix = ""
        if start is not None and pct >= 1:
            elapsed = time.time() - start
            eta = elapsed * (100 - pct) / pct
            suffix = f"   (~{eta:.0f}s left)"
        self.lbl_progress.configure(text=f"{pct}%  {msg}{suffix}")

    def _start_async(
        self: ECGApp,
        button: ctk.CTkButton,
        btn_busy_label: str,
        status_msg: str,
        worker: Callable[[], Any],
        on_done: Callable[..., None],
        original_label: "str | None" = None,
        pass_result: bool = False,
    ) -> None:
        """Disable *button*, show the progress bar, and run *worker* in a thread.

        On success, *on_done* is scheduled on the main thread via ``after(0, …)``.
        On failure, an error dialog is shown and the button is re-enabled.
        The original button text is restored in both cases.

        Parameters
        ----------
        button          : Button to disable while busy.
        btn_busy_label  : Text shown on the button while running.
        status_msg      : Status bar text shown while running (pass "" to skip).
        worker          : Callable executed in the background thread.
                          Must not access Tkinter widgets directly.
        on_done         : Callable executed on the main thread on success.
                          If pass_result=True it receives worker()'s return value.
        original_label  : Button text to restore; auto-detected if None.
        pass_result     : If True, worker() return value is passed to on_done(result).
                          If False (default), on_done is called with no arguments.
        """
        if getattr(self, "_async_busy", False):
            # self.progress/self.lbl_progress/_last_prog_pct/_async_start_time
            # are shared instance state with no per-operation isolation --
            # letting two _start_async calls run concurrently would interleave
            # their writes to those attributes and produce chaotic, seemingly
            # random progress-bar/label motion. Refuse the second call instead.
            self._set_status("Another operation is already running — please wait", ORANGE)
            return
        original_text = original_label or button.cget("text")
        button.configure(state="disabled", text=btn_busy_label)
        if status_msg:
            self._set_status(status_msg, ORANGE)
        self.progress.set(0)
        self._prog_row.pack(side="bottom", fill="x", padx=SPACE_M, pady=(SPACE_XS, SPACE_S))
        self._async_start_time = time.time()
        self._last_prog_pct = 0
        self._last_prog_msg = ""
        self._async_busy = True
        self._render_progress_label()

        def _thread_target() -> None:
            try:
                result = worker()
                if pass_result:
                    self.after(0, lambda r=result: _finish_success(r))
                else:
                    self.after(0, lambda: _finish_success(None))
            except Exception as exc:
                tb = traceback.format_exc()
                self.after(0, lambda e=exc, t=tb: _finish_error(e, t))

        def _finish_success(result: Any) -> None:
            self._async_busy = False
            self._stop_progress(button, original_text)
            if pass_result:
                on_done(result)
            else:
                on_done()  # noqa: pass_result=False branch — on_done takes no args

        def _finish_error(exc: Exception, tb: str) -> None:
            self._async_busy = False
            self._stop_progress(button, original_text)
            self._set_status(f"Error: {exc}", RED)
            self._show_error_dialog(original_text.strip(), exc, tb)

        threading.Thread(target=_thread_target, daemon=True).start()

    # Substrings of an exception message → a one-line, actionable hint.
    # Matched case-insensitively against str(exc); first hit wins. Keeps
    # the handful of errors a researcher can actually act on (wrong
    # channel, no peaks yet, ...) out of the generic "something broke"
    # framing a raw traceback gives everything.
    _ERROR_HINTS: "list[tuple[str, str]]" = [
        ("signal is flat", "Check the Channel field — the selected channel may be empty or wrong."),
        ("no peaks available", "Click Detect Peaks first."),
        ("signal not loaded", "Open a .mat file first."),
        ("no file loaded", "Open a .mat file first."),
    ]

    def _show_error_dialog(self: ECGApp, operation: str, exc: Exception, tb: str) -> None:
        """Show *operation* failed with str(exc) up front, traceback tucked
        behind a Copy button instead of dumped inline.

        Replaces a bare ``messagebox.showerror("Error", f"{exc}\\n\\n{tb}")``
        that put a 15-line file-path dump ahead of (or on equal footing
        with) the one sentence a non-developer can act on.
        """
        exc_str = str(exc)
        hint = next((h for pat, h in self._ERROR_HINTS if pat in exc_str.lower()), "")

        win = ctk.CTkToplevel(self)
        win.title(f"{operation} failed" if operation else "Operation failed")
        win.geometry("480x220")
        win.minsize(360, 180)
        win.configure(fg_color=PANEL)
        win.grab_set()
        win.lift()
        win.focus_force()

        ctk.CTkLabel(win, text=f"⚠  {operation} failed" if operation else "⚠  Operation failed",
                     font=FONT_BTN_PRIMARY, text_color=RED, anchor="w"
                     ).pack(fill="x", padx=20, pady=(18, 6))
        ctk.CTkLabel(win, text=exc_str, font=FONT_LABEL, text_color=TEXT,
                     anchor="w", justify="left", wraplength=430
                     ).pack(fill="x", padx=20, pady=(0, 6))
        if hint:
            ctk.CTkLabel(win, text=hint, font=FONT_SMALL, text_color=MUTED,
                         anchor="w", justify="left", wraplength=430
                         ).pack(fill="x", padx=20, pady=(0, 6))

        def _copy_details() -> None:
            try:
                self.clipboard_clear()
                self.clipboard_append(f"{operation} failed: {exc_str}\n\n{tb}")
                lbl_copied.configure(text="Copied to clipboard.")
            except Exception as _exc:
                log.debug("copy traceback to clipboard failed: %s", _exc)

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(6, 8), side="bottom")
        ctk.CTkButton(btn_row, text="Close", command=win.destroy,
                     fg_color=BORDER, hover_color=BORDER2, text_color=TEXT,
                     font=FONT_BTN_SEC, height=30, corner_radius=8
                     ).pack(side="right")
        ctk.CTkButton(btn_row, text="Copy details", command=_copy_details,
                     fg_color=BORDER, hover_color=BORDER2, text_color=MUTED,
                     font=FONT_BTN_SEC, height=30, corner_radius=8
                     ).pack(side="right", padx=(0, SPACE_S))
        lbl_copied = ctk.CTkLabel(win, text="", font=FONT_MICRO, text_color=MUTED)
        lbl_copied.pack(side="bottom", padx=20, pady=(0, 0), anchor="w")

    def _stop_progress(self: ECGApp, button: ctk.CTkButton, original_label: str) -> None:
        self.progress.set(1.0)
        self._prog_row.pack_forget()
        self.lbl_progress.configure(text="")
        self._async_start_time = None
        # A UI rebuild (theme toggle) during the job destroyed *button*; its
        # rebuilt twin is already enabled. Never let that abort the caller
        # (_finish_success would then skip on_done and drop the result).
        try:
            if button.winfo_exists():
                button.configure(state="normal", text=original_label)
        except tk.TclError as exc:
            log.debug("_stop_progress: button gone (UI rebuilt): %s", exc)
