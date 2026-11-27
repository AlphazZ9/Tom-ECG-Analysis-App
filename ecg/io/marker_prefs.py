"""The keyboard keys the user chose to ignore (a key pressed by mistake, one
that marks nothing in the protocol), remembered between sessions.

Every marker is always read from the file (ecg.io.loaders,
ecg.io.event_channels); this choice is applied where markers are used: the
suggested annotations and the Stim marker list of Compare segments. None is
ignored until the user says so (Annotations > Suggest from channels…)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from ecg.io.loaders import is_keyboard_channel

log = logging.getLogger("ecg")

PREFS_PATH: Path = Path.home() / ".ecg_marker_prefs.json"


def load_ignored_keys(path: "Optional[Path]" = None) -> "set[str]":
    """The keyboard keys ignored ({} when none was chosen)."""
    p = Path(path) if path is not None else PREFS_PATH
    try:
        keys = json.loads(p.read_text(encoding="utf-8")).get("ignored_keys", [])
        return {str(k) for k in keys if str(k)}
    except FileNotFoundError:
        return set()
    except Exception as exc:
        log.debug("marker preferences not read: %s", exc)
        return set()


def save_ignored_keys(keys: "Iterable[str]", path: "Optional[Path]" = None) -> None:
    p = Path(path) if path is not None else PREFS_PATH
    try:
        p.write_text(json.dumps({"ignored_keys": sorted({str(k) for k in keys if str(k)})}),
                     encoding="utf-8")
    except Exception as exc:
        log.warning("marker preferences not saved: %s", exc)


def is_ignored(channel: str, code: str, ignored: "set[str]") -> bool:
    """Whether a marker is left out: a key of *ignored* on a keyboard channel
    (codes of other channels are never filtered)."""
    return bool(ignored) and is_keyboard_channel(channel) and str(code).strip() in ignored


def keep_markers(markers: "Mapping[str, Sequence[float]]",
                 codes: "Optional[Mapping[str, Sequence[str]]]",
                 ignored: "set[str]") -> "dict[str, list[float]]":
    """*markers* ({channel: [times]}) without the ignored keys, using *codes*
    ({channel: [code]}, same order); unchanged when nothing is ignored or a
    channel has no codes."""
    out: "dict[str, list[float]]" = {}
    for ch, times in (markers or {}).items():
        cs = (codes or {}).get(ch)
        if not ignored or not is_keyboard_channel(ch) or cs is None or len(cs) != len(times):
            out[ch] = list(times)
        else:
            out[ch] = [t for t, c in zip(times, cs) if not is_ignored(ch, c, ignored)]
    return out


def key_counts(events: "Mapping[str, Sequence[tuple[float, str]]]") -> "dict[str, int]":
    """{key: how many times it was pressed} over the keyboard channels of
    *events* ({channel: [(time, code)]}), keys in order of first use."""
    out: "dict[str, int]" = {}
    for ch, evs in (events or {}).items():
        if is_keyboard_channel(ch):
            for _t, c in evs:
                c = str(c).strip()
                if c:
                    out[c] = out.get(c, 0) + 1
    return out
