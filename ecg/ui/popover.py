# -*- coding: utf-8 -*-
"""
ecg.ui.popover
--------------
Popover: a panel floating over the main window under the control that opened
it (the file chip, the Filters chip, the header's ⋯, Report > More...).

Its widgets are built once with the window and only shown or hidden, so every
setting it holds exists from startup: a session saves and restores it, and the
controllers write to it, whether it is open or not. A click outside it or
Escape closes it; a second click on its anchor closes it too.
"""
from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui.theme import (
    CARD, BORDER, BORDER2, TEXT, MUTED, SPACE_XS, SPACE_S, SPACE_M, FONT_SIDEBAR_HDR,
    FONT_BTN_SEC,
)

_GAP = 4         # px between the anchor and the panel
_MARGIN = 8      # px kept free at the window's edges


def _inside(widget_path: str, container) -> bool:
    """True when the Tk path *widget_path* is *container* or one of its children."""
    try:
        own = str(container)
    except Exception:
        return False
    return widget_path == own or widget_path.startswith(own + ".")


class Popover(ctk.CTkFrame):
    """Floating panel with a title row; content goes in ``.body``.

    *root* is the toplevel it floats over (placed, not a separate window, so it
    moves and themes with it). *on_open* runs before each opening, for content
    that follows the app's state (the recent files list).
    """

    def __init__(self, root, title: str = "", on_open: "Optional[Callable[[], None]]" = None,
                 width: int = 0) -> None:
        super().__init__(root, fg_color=CARD, corner_radius=8, border_width=1,
                         border_color=BORDER2)
        self._host = root
        self._anchor = None
        self._on_open = on_open
        self._is_open = False
        if title:
            head = ctk.CTkFrame(self, fg_color="transparent")
            head.pack(fill="x", padx=SPACE_M, pady=(SPACE_S, 0))
            ctk.CTkLabel(head, text=title.upper(), font=FONT_SIDEBAR_HDR, text_color=MUTED,
                         anchor="w").pack(side="left")
            ctk.CTkButton(head, text="✕", width=26, height=22, font=FONT_BTN_SEC,
                          fg_color="transparent", hover_color=BORDER, text_color=MUTED,
                          corner_radius=6, command=self.close).pack(side="right")
        self.body = ctk.CTkFrame(self, fg_color="transparent", width=width or 200, height=10)
        self.body.pack(fill="both", expand=True, padx=SPACE_M, pady=(SPACE_XS, SPACE_M))
        _register(root, self)

    # ── open / close ─────────────────────────────────────────────

    def is_open(self) -> bool:
        return self._is_open

    def open(self, anchor=None) -> None:
        """Show under *anchor* (the last one when None), kept inside the window."""
        if anchor is not None:
            self._anchor = anchor
        if self._on_open is not None:
            self._on_open()
        for other in _popovers(self._host):
            if other is not self and other.is_open():
                other.close()
        self._is_open = True
        self._place()
        self.lift()

    def close(self) -> None:
        if self._is_open:
            self._is_open = False
            self.place_forget()

    def toggle(self, anchor=None) -> None:
        if self._is_open:
            self.close()
        else:
            self.open(anchor)

    def _place(self) -> None:
        """Under the anchor's left edge, moved left / up when it would leave
        the window. place() takes unscaled CTk units; winfo_* returns pixels."""
        root = self._host
        root.update_idletasks()
        x = y = _MARGIN
        if self._anchor is not None:
            try:
                x = self._anchor.winfo_rootx() - root.winfo_rootx()
                y = self._anchor.winfo_rooty() - root.winfo_rooty() + self._anchor.winfo_height() + _GAP
            except Exception:
                pass
        scale = self._get_widget_scaling()
        self.place(x=x / scale, y=y / scale)
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        x = max(_MARGIN, min(x, root.winfo_width() - w - _MARGIN))
        y = max(_MARGIN, min(y, root.winfo_height() - h - _MARGIN))
        self.place(x=x / scale, y=y / scale)

    def _outside_click(self, widget_path: str) -> None:
        if not self._is_open:
            return
        if _inside(widget_path, self):
            return
        # A click on the anchor: its own command toggles the popover.
        if self._anchor is not None and _inside(widget_path, self._anchor):
            return
        self.close()


# ── one binding per toplevel, shared by its popovers ────────────────────────
# Bound once (add="+") and never unbound: tkinter's unbind(seq, funcid) drops
# every binding of the sequence on some versions. Popovers destroyed by a UI
# rebuild leave the list on the next event.

def _popovers(root) -> "list[Popover]":
    found = getattr(root, "_ecg_popovers", [])
    alive = [p for p in found if _exists(p)]
    root._ecg_popovers = alive
    return alive


def _exists(p: Popover) -> bool:
    try:
        return bool(p.winfo_exists())
    except Exception:
        return False


def _register(root, popover: Popover) -> None:
    _popovers(root).append(popover)
    if getattr(root, "_ecg_popover_bound", False):
        return
    root._ecg_popover_bound = True

    def _click(event) -> None:
        path = str(event.widget)
        for p in _popovers(root):
            p._outside_click(path)

    def _escape(_event) -> None:
        for p in _popovers(root):
            p.close()

    root.bind("<Button-1>", _click, add="+")
    root.bind("<Escape>", _escape, add="+")


def close_all(root) -> None:
    for p in _popovers(root):
        p.close()


def menu_button(parent, text: str, command: Callable[[], None]) -> ctk.CTkButton:
    """A full-width flat row of a popover used as a menu."""
    btn = ctk.CTkButton(parent, text=text, anchor="w", height=30, font=FONT_BTN_SEC,
                        fg_color="transparent", hover_color=BORDER, text_color=TEXT,
                        corner_radius=6, command=command)
    btn.pack(fill="x", pady=(0, SPACE_XS))
    return btn
