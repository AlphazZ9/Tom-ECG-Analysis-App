"""Annotations suggested from the other channels of the recording open
(Annotations → Suggest from channels…): the marker / keyboard channels and
the stimulus pulses of its waveform channels, read in the background; the
user ticks the ones to add and can rename them."""
from __future__ import annotations

import logging
import threading
import tkinter as tk
from typing import TYPE_CHECKING, Any, Callable, Optional

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core import annotation_suggest as sug
from ecg.io import event_channels as evc
from ecg.io import marker_prefs
from ecg.io.export import ANNOTATION_COLORS
from ecg.ui.theme import (
    BG, CARD, BORDER, FONT_HINT, FONT_SMALL, FONT_SUBSECTION, MUTED, ORANGE, PANEL, TEXT,
    button_style, input_style,
)

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


def collect(filepath: Optional[str], channel: Optional[str], crop: "Optional[dict]",
            t_max: Optional[float], annotations: "list[dict]"
            ) -> "tuple[list[dict], list[str], dict[str, int]]":
    """(suggestions not annotated yet, the channels read, {keyboard key: times
    pressed}) for the file, every key included (the ignored ones are hidden
    by the dialog). Runs off the main thread: reads the file only."""
    crop = crop or {}
    t0 = float(crop.get("t_start") or 0.0)
    t1 = crop.get("t_end")
    # A crop to the file end is "no end crop" (0).
    dur = crop.get("file_duration_s")
    t1 = 0.0 if t1 is None or (dur is not None and float(t1) >= float(dur)) else float(t1)
    events = evc.read_event_channels(filepath, t0, t1)
    waves = evc.read_waveforms(filepath, skip=channel, t_start=t0, t_end=t1)
    found = (sug.from_events({k: v for k, v in events.items() if v}, t_max=t_max)
             + sug.from_waveforms(waves, t_max=t_max))
    found.sort(key=lambda d: d["t_start"])
    read = [k for k, v in events.items() if v] + list(waves)
    return sug.drop_existing(found, annotations), read, marker_prefs.key_counts(events)


class AnnotationSuggestDialog(ctk.CTkToplevel):
    """The suggestions, each with a tick and an editable label; Add selected
    calls *on_add(list of annotations)*."""

    def __init__(self, parent: Any, app: "ECGApp",
                 on_add: "Callable[[list[dict]], None]") -> None:
        super().__init__(parent)
        self.title("Suggest annotations from channels")
        self.geometry("720x460")
        self.configure(fg_color=PANEL)
        self.transient(parent.winfo_toplevel())
        self._app, self._on_add = app, on_add
        self._rows: "list[tuple[tk.BooleanVar, ctk.CTkEntry, dict]]" = []
        self._lbl = ctk.CTkLabel(self, text="Reading the marker, keyboard and stimulus "
                                            "channels…", font=FONT_SMALL, text_color=MUTED,
                                 anchor="w", justify="left", wraplength=680)
        self._lbl.pack(fill="x", padx=12, pady=(10, 4))
        # Keyboard keys: untick one to ignore it (here and in Compare
        # segments' Stim marker list), remembered between sessions.
        self._keys_row = ctk.CTkFrame(self, fg_color="transparent")
        self._keys_row.pack(fill="x", padx=12, pady=(0, 4))
        self._ignored: "set[str]" = marker_prefs.load_ignored_keys()
        self._found: "list[dict]" = []
        self._keys: "dict[str, int]" = {}
        self._read: "list[str]" = []
        self._list = ctk.CTkScrollableFrame(self, fg_color=CARD)
        self._list.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=(0, 10))
        self._btn_add = ctk.CTkButton(bar, text="Add selected", width=0, state="disabled",
                                      command=self._add, **button_style("primary"))
        self._btn_add.pack(side="right")
        ctk.CTkButton(bar, text="Close", width=0, command=self.destroy,
                      **button_style("secondary")).pack(side="right", padx=(0, 6))
        for text, val in (("Tick all", True), ("Untick all", False)):
            ctk.CTkButton(bar, text=text, width=0, command=lambda v=val: self._tick(v),
                          **button_style("secondary")).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(bar, text="A lone marker lasts 1 s; a train spans its markers or pulses.",
                     font=FONT_HINT, text_color=MUTED).pack(side="left", padx=(6, 0))
        self._result: "Optional[tuple | str]" = None
        sig = app.signal
        t = getattr(app, "_time", None)
        t_max = float(t[-1]) if t is not None and len(t) else None
        args = (sig.filepath, sig.channel, sig.crop, t_max, list(app.analysis.annotations))

        def _work() -> None:
            try:
                self._result = collect(*args)
            except Exception as exc:          # shown, never raised in a thread
                log.warning("annotation suggestions: %s", exc)
                self._result = f"Could not read the channels: {exc}"
        threading.Thread(target=_work, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self) -> None:
        if not self.winfo_exists():
            return
        if self._result is None:
            self.after(100, self._poll)
            return
        self.show(self._result)

    def show(self, result: "tuple | str") -> None:
        if isinstance(result, str):
            self._lbl.configure(text=result, text_color=ORANGE)
            return
        found, read = result[0], result[1]
        keys = result[2] if len(result) > 2 else {}
        self._found, self._read, self._keys = list(found), list(read), dict(keys)
        self._show_keys(keys)
        self._show_rows()

    def _show_keys(self, keys: "dict[str, int]") -> None:
        for w in self._keys_row.winfo_children():
            w.destroy()
        if not keys:
            return
        ctk.CTkLabel(self._keys_row, text="Keyboard keys:", font=FONT_SMALL,
                     text_color=MUTED).pack(side="left", padx=(0, 6))
        for key, n in keys.items():
            var = tk.BooleanVar(value=key not in self._ignored)
            ctk.CTkCheckBox(self._keys_row, text=f"'{key}' ×{n}", variable=var,
                            font=FONT_SMALL, width=0, checkbox_width=16, checkbox_height=16,
                            command=lambda k=key, v=var: self._set_key(k, v.get())
                            ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(self._keys_row, text="untick a key to ignore it (also in Compare "
                     "segments' Stim marker list; remembered)", font=FONT_HINT,
                     text_color=MUTED).pack(side="left", padx=(4, 0))

    def _set_key(self, key: str, keep: bool) -> None:
        if keep:
            self._ignored.discard(key)
        else:
            self._ignored.add(key)
        marker_prefs.save_ignored_keys(self._ignored)
        self._show_keys(self._keys)
        self._show_rows()

    def _visible(self) -> "list[dict]":
        return [s for s in self._found
                if not marker_prefs.is_ignored(s.get("source", ""), s.get("code", ""),
                                               self._ignored)]

    def _show_rows(self) -> None:
        for w in self._list.winfo_children():
            w.destroy()
        self._rows = []
        found, read = self._visible(), self._read
        hidden = len(self._found) - len(found)
        if not found:
            self._btn_add.configure(state="disabled")
            self._lbl.configure(
                text=(f"Every suggestion is of an ignored key ({hidden}): tick it above to "
                      "see them." if hidden else
                      "No marker, keyboard or stimulus-pulse event that is not annotated "
                      "already." if read else
                      "This file has no marker or keyboard channel and no channel of "
                      "stimulus pulses (Spike2 .smr or its MATLAB export)."),
                text_color=MUTED)
            return
        self._lbl.configure(text=f"{len(found)} suggestion{'s' if len(found) != 1 else ''} "
                                 f"from {', '.join(read)}: untick the ones to leave out, "
                                 "edit a label if needed."
                                 + (f" ({hidden} of ignored keys hidden)" if hidden else ""),
                            text_color=TEXT)
        hdr = ctk.CTkFrame(self._list, fg_color=BORDER, corner_radius=4)
        hdr.pack(fill="x", padx=2, pady=(0, 2))
        for txt, w in (("", 30), ("Start (s)", 90), ("End (s)", 90), ("Label", 380)):
            ctk.CTkLabel(hdr, text=txt, width=w, font=FONT_SUBSECTION, text_color=TEXT,
                         anchor="w").pack(side="left", padx=3)
        for i, s in enumerate(found):
            row = ctk.CTkFrame(self._list, fg_color=CARD if i % 2 == 0 else BG, corner_radius=0)
            row.pack(fill="x", padx=2, pady=1)
            var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(row, text="", variable=var, width=30, checkbox_width=16,
                            checkbox_height=16).pack(side="left", padx=(4, 0))
            for txt in (f"{s['t_start']:.3f}", f"{s['t_end']:.3f}"):
                ctk.CTkLabel(row, text=txt, width=90, font=FONT_SMALL, text_color=TEXT,
                             anchor="w").pack(side="left", padx=3)
            ent = ctk.CTkEntry(row, width=380, **input_style())
            ent.insert(0, s["label"])
            ent.pack(side="left", padx=3, pady=2)
            self._rows.append((var, ent, s))
        self._btn_add.configure(state="normal")

    def _tick(self, on: bool) -> None:
        for var, _e, _s in self._rows:
            var.set(on)

    def _add(self) -> None:
        colors = [c for c, _n in ANNOTATION_COLORS]
        by_source: "dict[str, str]" = {}
        out = []
        for var, ent, s in self._rows:
            if not var.get():
                continue
            color = by_source.setdefault(s["source"], colors[len(by_source) % len(colors)])
            out.append({"t_start": float(s["t_start"]), "t_end": float(s["t_end"]),
                        "label": ent.get().strip() or s["label"], "color": color})
        if out:
            self._on_add(out)
        self.destroy()
