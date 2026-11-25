"""The other channels of a recording, read to suggest annotations: the
marker / keyboard channels (with their codes) and the waveform channels
that carry a stimulus (TTL pulses), from a Spike2 .smr file or its MATLAB
export (.mat v5/v6 or v7.3).

Times are in the loaded signal's time base: the load-time crop *t_start*
.. *t_end* (file s, 0 = none) subtracted, what lies outside dropped."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ecg.io import loaders

log = logging.getLogger("ecg")

# A waveform channel longer than this is read decimated (pulse onsets only
# need ~1 ms).
MAX_WAVE_FS = 1000.0
MAX_EVENTS = loaders.MAX_MARKERS_PER_CHANNEL


def code_text(code: Any) -> str:
    """A marker code as typed: Spike2 keeps a keyboard key as its ASCII
    value (the first of four bytes); 0 is no code."""
    if code is None:
        return ""
    if isinstance(code, (bytes, bytearray)):
        code = code.decode("latin-1", "replace")
    if isinstance(code, str):
        c = code.strip("\x00 ").strip()
        return c
    a = np.asarray(code).flatten()
    if not a.size:
        return ""
    try:
        v = int(a[0])
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    return chr(v) if 32 < v < 127 else str(v)


def _crop(times: np.ndarray, t0: float, t1: Optional[float]) -> np.ndarray:
    keep = times >= t0
    if t1 is not None:
        keep &= times <= t1
    return keep


def _bounds(t_start: float, t_end: float) -> "tuple[float, Optional[float]]":
    t0 = float(t_start) if t_start and t_start > 0 else 0.0
    t1 = float(t_end) if t_end and t_end > 0 else None
    return t0, t1


def _events(times: Any, codes: Any, t0: float, t1: Optional[float],
            channel: str = "") -> "list[tuple[float, str]]":
    t = np.asarray(times, dtype=np.float64).flatten()
    n = len(t)
    labels = [""] * n
    if codes is not None:
        c = np.asarray(codes)
        if c.ndim == 1 and n > 1 and len(c) == n:
            labels = [code_text(x) for x in c]
        elif c.ndim == 2 and n and c.shape[0] == n:
            labels = [code_text(row) for row in c]
        elif c.ndim == 2 and n and c.shape[1] == n:
            labels = [code_text(row) for row in c.T]
        elif n == 1:
            labels = [code_text(c.flatten()[0] if c.dtype.kind in "OUS" and c.size else c)]
    ok = np.isfinite(t) & _crop(np.nan_to_num(t, nan=-1.0), t0, t1)
    out = [(round(float(x) - t0, 6), lab) for x, lab, k in zip(t, labels, ok)
           if k and not loaders.ignored_marker(channel, lab)]
    return sorted(out)[:MAX_EVENTS]


# ── marker / keyboard channels ───────────────────────────────────────────

def _smr_events(filepath: str, t0: float, t1: Optional[float]) -> "dict[str, list]":
    reader = loaders._spike2_reader(filepath)
    out: "dict[str, list]" = {}
    for i, ev in enumerate(reader.header["event_channels"]):
        name = str(ev["name"])
        try:
            ts, _dur, lab = reader.get_event_timestamps(0, 0, i)
            t = reader.rescale_event_timestamp(ts, dtype="float64", event_channel_index=i)
        except Exception as exc:
            log.debug("Spike2 marker channel %r not read: %s", name, exc)
            continue
        key, k = name, 2
        while key in out:
            key, k = f"{name} ({k})", k + 1
        out[key] = _events(t, None if lab is None else
                           np.asarray([code_text(x) for x in lab], dtype=object), t0, t1,
                           channel=name)
    return out


def _mat_struct_events(raw: "dict[str, Any]", t0: float, t1: Optional[float]
                       ) -> "dict[str, list]":
    """Spike2 exports a marker channel as a struct with ``times`` (s) and,
    for a keyboard / marker channel, ``codes``; a waveform has ``values``."""
    out: "dict[str, list]" = {}
    for key, v in raw.items():
        if key.startswith("_") or not hasattr(v, "times") or hasattr(v, "values"):
            continue
        try:
            name = str(getattr(v, "title", "") or key).strip() or key
            if name in out:
                name = key
            out[name] = _events(getattr(v, "times"), getattr(v, "codes", None), t0, t1,
                                channel=name)
        except Exception as exc:
            log.debug("mat marker channel %r not read: %s", key, exc)
    return out


def _h5_text(obj: Any) -> str:
    try:
        a = np.asarray(obj[()]).flatten()
        if a.dtype.kind in "ui":
            return "".join(chr(int(x)) for x in a if 0 < int(x) < 0x110000).strip()
        return str(a[0].decode() if isinstance(a[0], bytes) else a[0]).strip()
    except Exception:
        return ""


def _h5_events(filepath: str, t0: float, t1: Optional[float]) -> "dict[str, list]":
    import h5py  # type: ignore[import-untyped]
    out: "dict[str, list]" = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            g = f[key]
            if not isinstance(g, h5py.Group) or "times" not in g or "values" in g:
                continue
            try:
                name = (_h5_text(g["title"]) if "title" in g else "") or key
                codes = np.asarray(g["codes"][()]) if "codes" in g else None
                out[name] = _events(np.asarray(g["times"][()]), codes, t0, t1, channel=name)
            except Exception as exc:
                log.debug("mat v7.3 marker channel %r not read: %s", key, exc)
    return out


def read_event_channels(filepath: "Optional[str]", t_start: float = 0.0,
                        t_end: float = 0.0) -> "dict[str, list[tuple[float, str]]]":
    """{channel: [(time s, code)]} of every marker / keyboard channel of the
    file (code '' for a plain marker). {} when it has none or cannot be read."""
    if not filepath:
        return {}
    t0, t1 = _bounds(t_start, t_end)
    try:
        if loaders.is_spike2_file(str(filepath)):
            return _smr_events(str(filepath), t0, t1)
        import scipy.io
        try:
            raw = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            return _h5_events(str(filepath), t0, t1)
        return _mat_struct_events(raw, t0, t1)
    except Exception as exc:
        log.info("marker channels not read from %s: %s", filepath, exc)
        return {}


# ── stimulus waveforms ───────────────────────────────────────────────────

def _decimate(sig: np.ndarray, fs: float) -> "tuple[np.ndarray, float]":
    step = int(fs // MAX_WAVE_FS)
    if step <= 1:
        return sig, fs
    n = len(sig) // step * step
    # The max of each block keeps a pulse shorter than the block.
    return sig[:n].reshape(-1, step).max(axis=1), fs / step


def read_waveforms(filepath: "Optional[str]", skip: "Optional[str]" = None,
                   t_start: float = 0.0, t_end: float = 0.0, max_channels: int = 12
                   ) -> "dict[str, tuple[np.ndarray, float]]":
    """{channel: (samples, fs)} of the waveform channels of the file other
    than *skip* (the ECG), cropped like the loaded signal and at most
    MAX_WAVE_FS. Channels whose rate is unknown are left out."""
    if not filepath:
        return {}
    t0, t1 = _bounds(t_start, t_end)
    out: "dict[str, tuple[np.ndarray, float]]" = {}

    def _keep(name: str, sig: Any, fs: "Optional[float]") -> None:
        if name == skip or fs is None or not fs > 0 or len(out) >= max_channels:
            return
        x = np.asarray(sig, dtype=np.float64).flatten()
        i0 = int(t0 * fs)
        i1 = int(t1 * fs) if t1 is not None else len(x)
        x = x[i0:i1]
        if len(x) < 10:
            return
        out[name] = _decimate(x, float(fs))

    try:
        if loaders.is_spike2_file(str(filepath)):
            reader = loaders._spike2_reader(str(filepath))
            for ch in loaders._spike2_channels(reader):
                if ch["name"] != skip and len(out) < max_channels:
                    _keep(ch["name"], loaders._spike2_read(reader, ch), ch["fs"])
            return out
        import scipy.io
        try:
            raw = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            return out          # v7.3: markers only
        flat = loaders._flatten_mat_dict(raw)
        for name, sig in flat.items():
            if name == skip or np.asarray(sig).dtype.kind not in "fiub":
                continue
            fs = loaders._detect_fs_from_mat(raw, loaders._base_key(name))
            _keep(name, sig, fs)
    except Exception as exc:
        log.info("waveform channels not read from %s: %s", filepath, exc)
    return out
