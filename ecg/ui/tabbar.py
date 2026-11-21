# -*- coding: utf-8 -*-
"""
ecg.ui.tabbar
-------------
TabBar: the main tab strip. Same .add/.tab/.get/.set API as the CTkTabview it
replaces, plus one state per tab (up to date / out of date / not computed)
drawn as a glyph next to the name, so a state never rests on colour alone.
"""
from __future__ import annotations

import math
import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, Optional

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui.freshness import COMPUTING, NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE
from ecg.ui.theme import (
    THEME, PANEL, BG, BORDER, TEXT, MUTED, BLUE, GREEN, ORANGE, ACCENT_TEXT, TAB_H,
    SPACE_XS, SPACE_S, SPACE_M, FONT_BODY, FONT_CARD_TITLE, FONT_HINT,
)

STATE_GLYPHS = {UP_TO_DATE: "●", OUT_OF_DATE: "◆", NOT_COMPUTED: "○", COMPUTING: "◐"}
LEGEND_ORDER = (UP_TO_DATE, OUT_OF_DATE, NOT_COMPUTED)

# The mockup's tab head: 16 px each side, 4 px between heads, the first one
# 16 px from the window edge. The 4 px gap is drawn inside the next head, so
# every pixel from the first head's left edge to the last head's right edge
# belongs to a head and selects its tab.
HEAD_PAD = 16
HEAD_GAP = 4


def state_color(state: str) -> str:
    # Looked up at draw time: GREEN / ORANGE / MUTED follow the theme.
    return {UP_TO_DATE: GREEN, OUT_OF_DATE: ORANGE, COMPUTING: BLUE}.get(state, MUTED)


def _text_width(widget: tk.Misc, font: tuple, text: str) -> int:
    """Width of `text` in `font` as a CTkLabel at scaling 1 draws it (CTk turns
    the size into pixels: -abs(size)). CTk scales the width= it is given."""
    spec = (font[0], -abs(int(font[1]))) + tuple(font[2:])
    return tkfont.Font(root=widget, font=spec).measure(text)


def _walk(widget: tk.Misc) -> "list[tk.Misc]":
    """The widget and every Tk window under it, CTk internals included (the
    canvas behind each CTkFrame/CTkLabel and the tk.Label inside a CTkLabel),
    which CTk's own .bind never reaches as a whole."""
    out = [widget]
    # tk.Misc's, not CTkFrame's: CTkFrame.winfo_children hides its canvas.
    for child in tk.Misc.winfo_children(widget):
        out.extend(_walk(child))
    return out


class TabBar(ctk.CTkFrame):
    """Tab heads in a row with the state legend on the right, the selected
    tab's frame below. The first tab added is selected. An unknown name raises
    ValueError in .tab and .set, like CTkTabview."""

    def __init__(self, parent, command: "Optional[Callable[[], None]]" = None) -> None:
        super().__init__(parent, fg_color=PANEL, corner_radius=0)
        self._command = command
        self._frames: "dict[str, ctk.CTkFrame]" = {}
        self._heads: "dict[str, dict]" = {}
        self._current: "Optional[str]" = None
        self._hint: "Optional[tk.Label]" = None
        # The head a press landed on, until its release: a release with no
        # press before it means Tk dropped the press (macOS does, e.g. after a
        # live resize or when its pointer state is stale), and selects instead.
        self._pressed: "Optional[str]" = None
        self._glyph_w = max(_text_width(self, FONT_HINT, g) for g in STATE_GLYPHS.values()) + 2

        self._bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0,
                                 height=int(round(TAB_H * THEME.font_scale)))
        self._bar.pack(side="top", fill="x", padx=0)
        self._bar.pack_propagate(False)
        ctk.CTkFrame(self, height=1, fg_color=BORDER, corner_radius=0).pack(side="top", fill="x")
        # The page under the tabs is the mockup's grey: cards stand on it.
        self._body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self._body.pack(side="top", fill="both", expand=True)
        self._body.grid_rowconfigure(0, weight=1)
        self._body.grid_columnconfigure(0, weight=1)

        legend = ctk.CTkFrame(self._bar, fg_color="transparent")
        legend.pack(side="right", padx=(SPACE_S, 24))
        for state in LEGEND_ORDER:
            ctk.CTkLabel(legend, text=STATE_GLYPHS[state], font=FONT_HINT, width=0,
                         text_color=state_color(state)).pack(side="left", padx=(SPACE_S, SPACE_XS))
            ctk.CTkLabel(legend, text=state, font=FONT_HINT, text_color=MUTED,
                         width=0).pack(side="left")

    # ── CTkTabview API ───────────────────────────────────────────────────

    def add(self, name: str) -> ctk.CTkFrame:
        if name in self._frames:
            raise ValueError(f"tab {name!r} already exists")
        self._frames[name] = ctk.CTkFrame(self._body, fg_color=BG, corner_radius=0)

        gap = 0 if not self._heads else HEAD_GAP
        head = ctk.CTkFrame(self._bar, fg_color="transparent", corner_radius=0)
        head.pack(side="left", fill="y", padx=(HEAD_PAD if not self._heads else 0, 0))
        row = ctk.CTkFrame(head, fg_color="transparent", corner_radius=0)
        row.pack(side="top", fill="both", expand=True, padx=(gap + HEAD_PAD, HEAD_PAD))
        # Fixed widths: neither a state glyph nor the selected tab's bold name
        # may change a head's width, or every head to its right would move
        # under the pointer. (width=0 would size to the text; without width
        # a CTkLabel reserves 140 px.)
        glyph = ctk.CTkLabel(row, text=STATE_GLYPHS[NOT_COMPUTED], font=FONT_HINT,
                             width=self._glyph_w, anchor="center",
                             text_color=state_color(NOT_COMPUTED))
        glyph.pack(side="left", padx=(0, SPACE_M))
        name_w = max(_text_width(self, FONT_CARD_TITLE, name),
                     _text_width(self, FONT_BODY, name)) + 2
        label = ctk.CTkLabel(row, text=name, font=FONT_BODY, text_color=TEXT,
                             width=name_w, anchor="w")
        label.pack(side="left")
        # The width as drawn, bold and regular: a measured width falls short
        # where the bold is synthesised (no bold face of the font installed),
        # and selecting the tab then widened its head and moved the others.
        drawn = 0
        for font in (FONT_CARD_TITLE, FONT_BODY):
            try:
                label.configure(font=font)
                label._label.update_idletasks()
                drawn = max(drawn, int(label._label.winfo_reqwidth()))
            except (tk.TclError, AttributeError):
                pass
        scale = float(label._get_widget_scaling()) if hasattr(label, "_get_widget_scaling") else 1.0
        if drawn and drawn / scale + 2 > name_w:
            label.configure(width=int(math.ceil(drawn / scale)) + 2)
        # width=1: a CTkFrame otherwise holds its parent at 200 px. The
        # underline starts after the gap, as when the gap sat between heads.
        line = ctk.CTkFrame(head, width=1, height=3, fg_color=PANEL, corner_radius=0)
        line.pack(side="bottom", fill="x", padx=(gap, 0))
        self._heads[name] = {"head": head, "glyph": glyph, "label": label, "line": line,
                             "state": NOT_COMPUTED, "reasons": ()}
        self._bind_head(name)
        if self._current is None:
            self.set(name)
        return self._frames[name]

    def tab(self, name: str) -> ctk.CTkFrame:
        if name not in self._frames:
            raise ValueError(f"no tab named {name!r}")
        return self._frames[name]

    def get(self) -> str:
        return self._current or ""

    def names(self) -> "list[str]":
        return list(self._frames)

    def set(self, name: str) -> None:
        self._select(name, feedback=False)

    # ── clicks ───────────────────────────────────────────────────────────

    def _bind_head(self, name: str) -> None:
        """One click target per head: a bindtag on every Tk window under it
        (outer frames, CTk canvases, the labels' inner tk.Labels, the
        underline), so no pixel of a head is dead."""
        tag = f"TabBarHead{self._heads[name]['head']._w}"
        self._heads[name]["tag"] = tag
        for w in _walk(self._heads[name]["head"]):
            tags = w.bindtags()
            if tag not in tags:
                # After the widget's own tag, before its class / toplevel /
                # all: the root's popover-closing <Button-1> still runs.
                w.bindtags(tags[:1] + (tag,) + tags[1:])
        self.bind_class(tag, "<ButtonPress-1>", lambda _e, n=name: self._on_press(n))
        self.bind_class(tag, "<ButtonRelease-1>", lambda e, n=name: self._on_release(n, e))

    def _on_press(self, name: str) -> None:
        self._pressed = name
        self._select(name, feedback=True)

    def _on_release(self, name: str, event: tk.Event) -> None:
        pressed, self._pressed = self._pressed, None
        if pressed is not None:
            return  # the press already did it (or was a drag off another head)
        # Tk never saw the press. Without it there is no implicit grab, so
        # the release came to the window under the pointer: check anyway.
        if self._head_at(event.x_root, event.y_root) == name:
            self._select(name, feedback=True)

    def _head_at(self, x_root: int, y_root: int) -> "Optional[str]":
        try:
            w = self.winfo_containing(x_root, y_root)
        except (tk.TclError, KeyError):
            return None
        while w is not None:
            for n, head in self._heads.items():
                if w is head["head"]:
                    return n
            w = getattr(w, "master", None)
        return None

    def _select(self, name: str, feedback: bool) -> None:
        frame = self.tab(name)
        if name == self._current:
            return
        if self._current is not None:
            self._frames[self._current].grid_forget()
            self._draw_head(self._current, False)
        self._current = name
        frame.grid(row=0, column=0, sticky="nsew")
        self._draw_head(name, True)
        if feedback:
            # Show the new head and page before the app's tab-change work,
            # which can take a while (Compare / Report rebuilds): on Aqua
            # nothing is drawn until the handler returns, and a click with no
            # visible effect reads as a missed one. update idletasks also
            # flushes the drawing on macOS.
            try:
                self.update_idletasks()
            except tk.TclError:
                pass
        if self._command is not None:
            self._command()

    # ── states ───────────────────────────────────────────────────────────

    def set_state(self, name: str, state: str, reasons: "tuple[str, ...]" = ()) -> None:
        head = self._heads[name]
        head["reasons"] = tuple(reasons)
        if head["state"] != state:
            head["state"] = state
            head["glyph"].configure(text=STATE_GLYPHS[state], text_color=state_color(state))

    @property
    def bar(self) -> ctk.CTkFrame:
        """The row of the heads: what follows the last head has the slack."""
        return self._bar

    def state(self, name: str) -> str:
        return self._heads[name]["state"]

    def state_text(self, name: str) -> str:
        """One line for the status bar: the tab's state and why."""
        head = self._heads[name]
        why = f" — {'; '.join(head['reasons'])}" if head["reasons"] else ""
        return f"{name}: {head['state']}{why}"

    def head_widgets(self, name: str) -> list:
        """The widgets of a tab head, for binding a hover hint."""
        head = self._heads[name]
        return [head["glyph"], head["label"]]

    def show_hint(self, name: str) -> None:
        """The tab's state and every reason, one per line, under its head: the
        status bar holds one line, cut to the width the toolbar leaves it."""
        head = self._heads[name]
        if self._hint is None:
            self._hint = tk.Label(self, bg=PANEL, fg=TEXT, justify="left", bd=0,
                                  padx=SPACE_S, pady=SPACE_XS, highlightthickness=1,
                                  highlightbackground=BORDER)
        width = self.winfo_width()
        self._hint.configure(text="\n".join((f"{name}: {head['state']}", *head["reasons"])),
                             font=(THEME.font_family, -int(round(12 * THEME.font_scale))),
                             wraplength=max(width - 4 * SPACE_S, 100))
        x = head["head"].winfo_rootx() - self.winfo_rootx()
        x = max(SPACE_S, min(x, width - self._hint.winfo_reqwidth() - SPACE_S))
        self._hint.place(x=x, y=self._bar.winfo_height())
        self._hint.lift()

    def hide_hint(self) -> None:
        if self._hint is not None:
            self._hint.place_forget()

    def _draw_head(self, name: str, selected: bool) -> None:
        head = self._heads[name]
        head["label"].configure(font=FONT_CARD_TITLE if selected else FONT_BODY,
                                text_color=ACCENT_TEXT if selected else TEXT)
        head["line"].configure(fg_color=BLUE if selected else PANEL)
