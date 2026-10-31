# -*- coding: utf-8 -*-
"""
ecg.ui.tabbar
-------------
TabBar: the main tab strip. Same .add/.tab/.get/.set API as the CTkTabview it
replaces, plus one state per tab (up to date / out of date / not computed)
drawn as a glyph next to the name, so a state never rests on colour alone.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui.freshness import COMPUTING, NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE
from ecg.ui.theme import (
    THEME, PANEL, BORDER, TEXT, MUTED, BLUE, GREEN, ORANGE,
    SPACE_XS, SPACE_S, SPACE_M, FONT_LABEL, FONT_CARD_TITLE, FONT_HINT,
)

STATE_GLYPHS = {UP_TO_DATE: "●", OUT_OF_DATE: "◆", NOT_COMPUTED: "○", COMPUTING: "◐"}
LEGEND_ORDER = (UP_TO_DATE, OUT_OF_DATE, NOT_COMPUTED)


def state_color(state: str) -> str:
    # Looked up at draw time: GREEN / ORANGE / MUTED follow the theme.
    return {UP_TO_DATE: GREEN, OUT_OF_DATE: ORANGE, COMPUTING: BLUE}.get(state, MUTED)


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

        self._bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=38)
        self._bar.pack(side="top", fill="x")
        self._bar.pack_propagate(False)
        ctk.CTkFrame(self, height=1, fg_color=BORDER, corner_radius=0).pack(side="top", fill="x")
        self._body = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        self._body.pack(side="top", fill="both", expand=True)
        self._body.grid_rowconfigure(0, weight=1)
        self._body.grid_columnconfigure(0, weight=1)

        legend = ctk.CTkFrame(self._bar, fg_color="transparent")
        legend.pack(side="right", padx=(SPACE_S, SPACE_M))
        for state in LEGEND_ORDER:
            ctk.CTkLabel(legend, text=STATE_GLYPHS[state], font=FONT_HINT, width=0,
                         text_color=state_color(state)).pack(side="left", padx=(SPACE_S, SPACE_XS))
            ctk.CTkLabel(legend, text=state, font=FONT_HINT, text_color=MUTED,
                         width=0).pack(side="left")

    # ── CTkTabview API ───────────────────────────────────────────────────

    def add(self, name: str) -> ctk.CTkFrame:
        if name in self._frames:
            raise ValueError(f"tab {name!r} already exists")
        self._frames[name] = ctk.CTkFrame(self._body, fg_color=PANEL, corner_radius=0)

        head = ctk.CTkFrame(self._bar, fg_color="transparent", corner_radius=0)
        head.pack(side="left", fill="y", padx=(SPACE_S, 0))
        row = ctk.CTkFrame(head, fg_color="transparent", corner_radius=0)
        row.pack(side="top", fill="both", expand=True, padx=SPACE_S)
        # width=0: a CTkLabel otherwise reserves 140 px, whatever its text.
        glyph = ctk.CTkLabel(row, text=STATE_GLYPHS[NOT_COMPUTED], font=FONT_HINT, width=0,
                             text_color=state_color(NOT_COMPUTED))
        glyph.pack(side="left", padx=(0, SPACE_XS))
        label = ctk.CTkLabel(row, text=name, font=FONT_LABEL, text_color=TEXT, width=0)
        label.pack(side="left")
        # width=1: a CTkFrame otherwise holds its parent at 200 px.
        line = ctk.CTkFrame(head, width=1, height=3, fg_color=PANEL, corner_radius=0)
        line.pack(side="bottom", fill="x")
        for w in (head, row, glyph, label):
            w.bind("<Button-1>", lambda _e, n=name: self.set(n))
        self._heads[name] = {"head": head, "glyph": glyph, "label": label, "line": line,
                             "state": NOT_COMPUTED, "reasons": ()}
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
        frame = self.tab(name)
        if name == self._current:
            return
        if self._current is not None:
            self._frames[self._current].grid_forget()
            self._draw_head(self._current, False)
        self._current = name
        frame.grid(row=0, column=0, sticky="nsew")
        self._draw_head(name, True)
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
        head["label"].configure(font=FONT_CARD_TITLE if selected else FONT_LABEL,
                                text_color=BLUE if selected else TEXT)
        head["line"].configure(fg_color=BLUE if selected else PANEL)
