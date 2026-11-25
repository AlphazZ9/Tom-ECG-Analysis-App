"""
ecg.io.loaders
──────────────
.mat (v5 and v7.3/HDF5) signal loading.
Also hosts _serialise_results / _deserialise_results used by the session layer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.io

log = logging.getLogger("ecg")

try:
    import h5py       # type: ignore[import-untyped]
    H5_AVAILABLE = True
except ImportError:
    h5py = None       # type: ignore[assignment]
    H5_AVAILABLE = False

_DROP = object()   # _json_safe(): value not representable in JSON


def _json_safe(val: Any) -> Any:
    """*val* as plain JSON types (numpy scalars unwrapped, tuples/arrays as
    lists, dicts with str keys), or _DROP when it holds anything else."""
    if val is None or isinstance(val, (bool, str)):
        return val
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        items = [_json_safe(v) for v in val]
        return _DROP if any(v is _DROP for v in items) else items
    if isinstance(val, dict):
        out = {str(k): _json_safe(v) for k, v in val.items()}
        return _DROP if any(v is _DROP for v in out.values()) else out
    return _DROP


def _serialise_results(results: dict) -> dict:
    """Convert analysis results dict to a plain JSON-compatible form.

    DataFrames become {orient:'list'} dicts; numpy arrays become lists;
    other primitives are passed through unchanged.

    A DataFrame's ``attrs`` are kept too (JSON-representable entries
    only): they carry what the table alone cannot say -- why a spectrum
    was withheld (hrv_freq ``withheld_reason``), the non-linear beat
    truncation and withheld pair metrics, the RR exclusion counts -- and
    a restored session must still disclose them.
    """
    out: dict = {}
    for key, val in results.items():
        if val is None:
            out[key] = None
        elif isinstance(val, pd.DataFrame):
            out[key] = {"__dataframe__": True, "data": val.to_dict(orient="list")}
            attrs = {k: v for k, v in ((str(k), _json_safe(v)) for k, v in val.attrs.items())
                     if v is not _DROP}
            if attrs:
                out[key]["attrs"] = attrs
        elif isinstance(val, np.ndarray):
            out[key] = {"__ndarray__": True, "data": val.tolist(), "dtype": str(val.dtype)}
        else:
            # scalars, lists, basic types — pass through as-is
            out[key] = val
    return out


def _deserialise_results(raw: dict) -> dict:
    """Inverse of _serialise_results."""
    out: dict = {}
    for key, val in raw.items():
        if val is None:
            out[key] = None
        elif isinstance(val, dict) and val.get("__dataframe__"):
            df = pd.DataFrame(val["data"])
            if isinstance(val.get("attrs"), dict):
                df.attrs.update(val["attrs"])
            out[key] = df
        elif isinstance(val, dict) and val.get("__ndarray__"):
            out[key] = np.array(val["data"], dtype=val.get("dtype", "float64"))
        else:
            out[key] = val
    return out


# ════════════════════════════════════════════════════════════
#  MAT / HDF5 LOADER
# ════════════════════════════════════════════════════════════

def _split_channels(
    key: str, arr: np.ndarray, time_axis_on_tie: int,
) -> dict[str, np.ndarray]:
    """{name: 1-D array} for one variable, one entry per channel.

    A 2-D matrix with more than one row *and* more than one column holds
    several channels side by side: flattening it would interleave them
    sample by sample (ecg0, other0, ecg1, other1, ...) into a sawtooth
    trace of twice the real duration.  It is instead split along its
    shorter axis into ``key[0]``, ``key[1]``, ... with the longer axis
    taken as time.  A square matrix is ambiguous; *time_axis_on_tie* is
    the axis the caller's file format stores samples along (0 for v5
    column-major loadmat arrays, 1 for v7.3 HDF5 where h5py exposes the
    MATLAB matrix transposed).  1-D and (n,1)/(1,n) inputs are flattened
    unchanged under *key*.
    """
    if arr.ndim == 2 and min(arr.shape) > 1:
        if arr.shape[0] != arr.shape[1]:
            time_axis = 0 if arr.shape[0] > arr.shape[1] else 1
        else:
            time_axis = time_axis_on_tie
        cols = arr if time_axis == 1 else arr.T
        return {f"{key}[{i}]": np.ascontiguousarray(c) for i, c in enumerate(cols)}
    return {key: arr.flatten()}


def _base_key(channel: str) -> str:
    """Variable name a ``key[i]`` split channel came from (MATLAB names
    cannot contain '['), so fs metadata stored on that variable is found."""
    return channel.split("[", 1)[0]


def _flatten_mat_dict(raw: dict) -> dict[str, np.ndarray]:
    """Convert a raw scipy.io.loadmat dict into {name: 1-D array} pairs.

    Handles Spike2 structs that store data under a `.values` or `.data`
    attribute, falling back to direct array conversion. Non-array keys
    (metadata starting with '_') are skipped silently.  Multi-column
    matrices are split into one channel per column (see _split_channels).
    """
    flat: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        try:
            if hasattr(value, "values"):
                arr = np.array(value.values)
            elif hasattr(value, "data"):
                arr = np.array(value.data)
            else:
                arr = np.array(value)
            flat.update(_split_channels(key, arr, time_axis_on_tie=0))
        except Exception as exc:
            log.debug("Skipping mat key '%s': %s", key, exc)
    return flat


def _detect_fs_from_mat(raw: dict, channel: str) -> Optional[float]:
    """Try to read the sampling rate from a Spike2 .mat struct.

    Spike2 exports store 1/fs as ``interval`` inside each channel struct.
    Also checks common direct fs fields (``sample_rate``, ``fs``, ``Fs``).

    Returns
    -------
    float or None
    """
    # Try channel-struct interval field (Spike2 primary path)
    ch = raw.get(channel)
    if ch is not None:
        for attr in ("interval", "start", "sample_interval"):
            v = getattr(ch, attr, None)
            if v is not None:
                try:
                    interval = float(np.array(v).flat[0])
                    if 1e-6 < interval < 1.0:   # plausible 1–1 000 000 Hz
                        return round(1.0 / interval)
                except Exception as _exc:
                    log.debug("_detect_fs_from_mat: interval field %r unusable: %s", attr, _exc)
        # Some exports store fs directly
        for attr in ("fs", "Fs", "sample_rate", "samplerate", "SampleRate"):
            v = getattr(ch, attr, None)
            if v is not None:
                try:
                    fs_val = float(np.array(v).flat[0])
                    if 100 <= fs_val <= 500_000:
                        return round(fs_val)
                except Exception as _exc:
                    log.debug("_detect_fs_from_mat: fs field %r unusable: %s", attr, _exc)

    # Top-level keys like "fs", "Fs", "sample_rate"
    for key in ("fs", "Fs", "sample_rate", "samplerate", "SampleRate",
                "sampling_rate", "samplingrate"):
        v = raw.get(key)
        if v is not None:
            try:
                fs_val = float(np.array(v).flat[0])
                if 100 <= fs_val <= 500_000:
                    return round(fs_val)
            except Exception as _exc:
                log.debug("_detect_fs_from_mat: top-level key %r unusable: %s", key, _exc)
    return None


def _detect_fs_from_hdf5_obj(f: "Any", channel: str) -> Optional[float]:
    """Same as _detect_fs_from_hdf5, but takes an already-open h5py.File.

    Split out so load_mat_signal() can open the HDF5 file once and reuse it
    for both fs-detection and channel flattening, instead of each helper
    opening (and re-parsing the file's group tree from) its own handle.
    """
    # Try channel group interval attribute
    grp = f.get(channel)
    if grp is not None and isinstance(grp, h5py.Group):
        for attr in ("interval", "sample_interval"):
            v = grp.get(attr)
            if v is not None:
                try:
                    interval = float(np.array(v).flat[0])
                    if 1e-6 < interval < 1.0:
                        return round(1.0 / interval)
                except Exception as _exc:
                    log.debug("_detect_fs_from_hdf5_obj: interval attr %r unusable: %s", attr, _exc)
        for attr in ("fs", "Fs", "sample_rate"):
            v = grp.get(attr)
            if v is not None:
                try:
                    fs_val = float(np.array(v).flat[0])
                    if 100 <= fs_val <= 500_000:
                        return round(fs_val)
                except Exception as _exc:
                    log.debug("_detect_fs_from_hdf5_obj: fs attr %r unusable: %s", attr, _exc)
    # Top-level datasets
    for key in ("fs", "Fs", "sample_rate", "samplerate"):
        v = f.get(key)
        if v is not None:
            try:
                fs_val = float(np.array(v).flat[0])
                if 100 <= fs_val <= 500_000:
                    return round(fs_val)
            except Exception as _exc:
                log.debug("_detect_fs_from_hdf5_obj: top-level key %r unusable: %s", key, _exc)
    return None


def _detect_fs_from_hdf5(filepath: str, channel: str) -> Optional[float]:
    """Try to read the sampling rate from a Spike2 v7.3 HDF5 .mat file."""
    if not H5_AVAILABLE or h5py is None:
        return None
    try:
        with h5py.File(filepath, "r") as f:
            return _detect_fs_from_hdf5_obj(f, channel)
    except Exception as exc:
        log.debug("_detect_fs_from_hdf5: %s", exc)
    return None


def _flatten_hdf5_file_obj(f: "Any") -> dict[str, np.ndarray]:
    """Same as _flatten_hdf5_file, but takes an already-open h5py.File."""
    flat: dict[str, np.ndarray] = {}
    for key in f.keys():
        group = f[key]
        found = False
        for sub in ("values", "data"):
            if isinstance(group, h5py.Group) and sub in group:
                try:
                    for name, arr in _split_channels(
                            key, np.array(group[sub]), time_axis_on_tie=1).items():
                        if len(arr) > 100:
                            flat[name] = arr
                except Exception as exc:
                    log.debug("HDF5 '%s/%s': %s", key, sub, exc)
                found = True
                break
        if not found and isinstance(group, h5py.Dataset):
            try:
                for name, arr in _split_channels(
                        key, np.array(group), time_axis_on_tie=1).items():
                    if len(arr) > 100:
                        flat[name] = arr
            except Exception as exc:
                log.debug("HDF5 dataset '%s': %s", key, exc)
    return flat


def _flatten_hdf5_file(filepath: str) -> dict[str, np.ndarray]:
    """Extract numeric 1-D arrays from an HDF5 (.mat v7.3) file.

    Searches each top-level key for a 'values' or 'data' sub-dataset
    (Spike2 convention) before falling back to treating the key itself as
    a dataset.  Entries shorter than 100 samples are discarded.

    Raises
    ------
    ImportError  if h5py is not installed.
    """
    if not H5_AVAILABLE or h5py is None:
        raise ImportError(
            "MATLAB v7.3 (HDF5) file detected — h5py is required.\n"
            "Run:  pip install h5py"
        )
    with h5py.File(filepath, "r") as f:
        return _flatten_hdf5_file_obj(f)


def _pick_best_channel(
    flat: dict[str, np.ndarray],
    preferred: str,
) -> tuple[np.ndarray, str, list[str]]:
    """Select an ECG channel from a flat {name: array} dict.

    Prefers *preferred* if present and long enough.  Otherwise scores
    each candidate by length and variance (monotone ramps score −1).
    """
    def channel_score(arr: np.ndarray) -> int:
        if len(arr) < 200 or arr.dtype.kind not in "fi":
            return -1
        diff = np.diff(arr[:500])
        if len(diff) > 0 and np.std(diff) / (abs(np.mean(diff)) + 1e-12) < 0.001:
            return -1   # monotone ramp = time vector, not signal
        return len(arr) * (2 if arr.std() > 1e-4 else 1)

    keys = sorted(flat.keys())

    if preferred in flat and len(flat[preferred]) > 100:
        return flat[preferred].astype(np.float64), preferred, keys

    best = max(flat, key=lambda k: channel_score(flat[k]), default=None)
    if best and channel_score(flat[best]) > 0:
        scores = sorted(
            ((k, channel_score(flat[k])) for k in flat), key=lambda kv: -kv[1]
        )
        runner_up = f", runner-up {scores[1][0]!r} (score {scores[1][1]})" if len(scores) > 1 else ""
        log.warning(
            "Channel '%s' not found; auto-selected '%s' (score %d%s) out of %s",
            preferred, best, channel_score(flat[best]), runner_up, keys,
        )
        return flat[best].astype(np.float64), best, keys

    raise ValueError(f"No ECG channel found. Available keys: {keys}")


# ════════════════════════════════════════════════════════════
#  SPIKE2 (.smr) LOADER
# ════════════════════════════════════════════════════════════
# Native Spike2 files are read with neo (pure Python, no CED library).
# The app works at an integer sampling rate, while Spike2 rates are set by a
# sample interval (972 µs -> 1028.807 Hz): the channel is resampled to a
# round rate by an exact rational factor, so times and HR stay exact instead
# of drifting by the dropped fraction (2.8 s over an hour at 1028 Hz).

try:
    from neo.rawio import Spike2RawIO   # type: ignore[import-untyped]
    NEO_AVAILABLE = True
except Exception:
    Spike2RawIO = None                  # type: ignore[assignment,misc]
    NEO_AVAILABLE = False

SPIKE2_EXTS = (".smr", ".smrx")
_CED_MAGIC = b"(C) CED"


def is_spike2_file(filepath: str) -> bool:
    """A Spike2 data file: by extension, or by its "(C) CED" header."""
    if str(filepath).lower().endswith(SPIKE2_EXTS):
        return True
    try:
        with open(filepath, "rb") as f:
            return _CED_MAGIC in f.read(16)
    except OSError:
        return False


def integer_rate(fs_in: float) -> "tuple[int, int, int]":
    """(fs_out, up, down) taking *fs_in* to a round integer rate by
    resample_poly(x, up, down). An integer rate is kept (up = down = 1);
    otherwise the rate goes down to a round value (1028.8 -> 1000 Hz,
    9259.3 -> 9000 Hz) through the small exact ratio Spike2's tick-based
    intervals give (243/250 for both)."""
    from fractions import Fraction
    if abs(fs_in - round(fs_in)) < 1e-6:
        return int(round(fs_in)), 1, 1
    step = 100 if fs_in >= 200 else 10 if fs_in >= 20 else 1
    fs_out = max(1, int(fs_in // step) * step)
    ratio = Fraction(fs_out / fs_in).limit_denominator(5000)
    return fs_out, ratio.numerator, ratio.denominator


def _ecg_likeness(x: np.ndarray, fs: float) -> float:
    """How much 20 s of *x* look like a mouse ECG: many sharp, regular beats
    at a mouse rate (RR 60-350 ms). 0 when not."""
    from scipy.signal import butter, find_peaks, sosfiltfilt
    if len(x) < fs * 5 or not np.isfinite(x).all() or np.std(x) == 0:
        return 0.0
    hi = min(150.0, fs / 2 - 1)
    if hi <= 6:
        return 0.0
    y = sosfiltfilt(butter(3, [5.0, hi], btype="band", fs=fs, output="sos"), x)
    pk, _ = find_peaks(np.abs(y), distance=max(1, int(0.05 * fs)),
                       prominence=3 * np.std(y))
    if len(pk) < 10:
        return 0.0
    rr = np.diff(pk) / fs * 1000.0
    med = float(np.median(rr))
    if not 60.0 <= med <= 350.0:
        return 0.0
    cv = float(np.std(rr) / np.mean(rr))
    return len(pk) / (1.0 + 10.0 * cv)


def _spike2_reader(filepath: str):
    if str(filepath).lower().endswith(".smrx"):
        raise ValueError(
            "Spike2 64-bit files (.smrx) cannot be read here. In Spike2, use "
            "File > Save As... and choose the 32-bit .smr format, or "
            "File > Export As... > MATLAB (.mat).")
    if not NEO_AVAILABLE or Spike2RawIO is None:
        raise ImportError("Spike2 (.smr) file detected — the neo package is "
                          "required.\nRun:  pip install neo")
    import warnings
    reader = Spike2RawIO(filename=str(filepath))
    with warnings.catch_warnings():
        # A channel with a zero sample interval (markers) warns on 1/0.
        warnings.simplefilter("ignore", RuntimeWarning)
        reader.parse_header()
    return reader


def _spike2_channels(reader) -> "list[dict]":
    """Every waveform channel: name, stream, position in its stream, rate."""
    chans = reader.header["signal_channels"]
    streams = list(reader.header["signal_streams"]["id"])
    out = []
    for si, sid in enumerate(streams):
        in_stream = [i for i, c in enumerate(chans) if c["stream_id"] == sid]
        for j, ci in enumerate(in_stream):
            fs = float(chans[ci]["sampling_rate"])
            if np.isfinite(fs) and fs > 0:
                out.append({"name": str(chans[ci]["name"]), "stream": si, "index": j,
                            "fs": fs, "n": int(reader.get_signal_size(0, 0, si))})
    return out


def _spike2_read(reader, ch: dict, i0: int = 0, i1: "Optional[int]" = None) -> np.ndarray:
    raw = reader.get_analogsignal_chunk(0, 0, i0, i1, stream_index=ch["stream"],
                                        channel_indexes=[ch["index"]])
    return reader.rescale_signal_raw_to_float(
        raw, dtype="float64", stream_index=ch["stream"],
        channel_indexes=[ch["index"]])[:, 0]


def load_spike2_signal(
    filepath: str, channel: str,
) -> tuple[np.ndarray, str, list[str], Optional[float]]:
    """Load the ECG channel of a Spike2 .smr file (same return as
    load_mat_signal). *channel* is used when a channel has that name (any
    case); otherwise the channel that looks most like a mouse ECG on 20 s
    taken 10 min in (or at the start of a short recording) is chosen."""
    reader = _spike2_reader(filepath)
    chans = _spike2_channels(reader)
    keys = sorted(c["name"] for c in chans)
    if not chans:
        raise ValueError("No waveform channel in this Spike2 file.")
    pick = next((c for c in chans if c["name"] == channel), None) or next(
        (c for c in chans if c["name"].lower() == str(channel).lower()), None)
    if pick is None:
        def score(c: dict) -> float:
            n = int(20 * c["fs"])
            i0 = min(int(600 * c["fs"]), max(0, c["n"] - n))
            try:
                return _ecg_likeness(_spike2_read(reader, c, i0, i0 + n), c["fs"])
            except Exception as exc:
                log.debug("Spike2 channel %r not scored: %s", c["name"], exc)
                return 0.0
        scores = {c["name"]: score(c) for c in chans}
        pick = max(chans, key=lambda c: scores[c["name"]])
        if scores[pick["name"]] <= 0:
            raise ValueError(f"No ECG-like channel found. Available channels: {keys}")
        log.warning("Channel '%s' not found; auto-selected Spike2 channel '%s' "
                    "(ECG-likeness %s) out of %s", channel, pick["name"],
                    {k: round(v, 1) for k, v in scores.items()}, keys)
    sig = _spike2_read(reader, pick)
    fs_out, up, down = integer_rate(pick["fs"])
    if (up, down) != (1, 1):
        from scipy.signal import resample_poly
        sig = resample_poly(sig, up, down)
        log.info("Spike2 channel '%s' resampled %.4f -> %d Hz (x%d/%d)",
                 pick["name"], pick["fs"], fs_out, up, down)
    return np.ascontiguousarray(sig, dtype=np.float64), pick["name"], keys, float(fs_out)


# Marker times of the last Spike2 files read: {(path, mtime, size): markers}
# (read_spike2_markers runs on every load / re-detection of the same file).
_MARKER_CACHE: "dict[tuple, dict[str, list[float]]]" = {}
MAX_MARKERS_PER_CHANNEL = 20000


# Keyboard keys left out of the markers, the suggested annotations and the
# stimulation events: '0' marks nothing in these recordings.
IGNORED_KEYS = frozenset({"0"})


def is_keyboard_channel(name: str) -> bool:
    return "keyboard" in str(name).lower()


def ignored_marker(channel: str, code: str) -> bool:
    """Whether the marker *code* of *channel* is left out (a keyboard key
    of IGNORED_KEYS)."""
    return is_keyboard_channel(channel) and str(code).strip() in IGNORED_KEYS


def _label_text(lab: Any) -> str:
    if isinstance(lab, (bytes, bytearray)):
        lab = lab.decode("latin-1", "replace")
    return str(lab).strip("\x00 ").strip()


def _spike2_markers_raw(reader) -> "dict[str, list[float]]":
    """{channel name: [times, s from the start of the file]} of every
    event / marker channel of a parsed Spike2RawIO (channels with no event
    included, empty)."""
    out: "dict[str, list[float]]" = {}
    for i, ev in enumerate(reader.header["event_channels"]):
        name = str(ev["name"])
        try:
            ts, _dur, lab = reader.get_event_timestamps(0, 0, i)
            t = reader.rescale_event_timestamp(ts, dtype="float64", event_channel_index=i)
            t = np.asarray(t, dtype=np.float64)
            if lab is not None and len(lab) == len(t) and is_keyboard_channel(name):
                keep = np.array([not ignored_marker(name, _label_text(x)) for x in lab],
                                dtype=bool)
                t = t[keep]
            t = np.sort(t[np.isfinite(t)])[:MAX_MARKERS_PER_CHANNEL]
        except Exception as exc:
            log.debug("Spike2 marker channel %r not read: %s", name, exc)
            continue
        key, k = name, 2
        while key in out:                    # two channels of the same name
            key, k = f"{name} ({k})", k + 1
        out[key] = [float(x) for x in t]
    return out


def read_spike2_markers(filepath: "Optional[str]", t_start: float = 0.0,
                        t_end: float = 0.0) -> "dict[str, list[float]]":
    """The marker / event channels of a Spike2 .smr file: {channel: [times
    in s]}, in the time base of the loaded signal -- the load-time crop
    *t_start* .. *t_end* (file s, 0 = none) subtracted and the markers
    outside it dropped (resampling the waveform does not move them).
    {} for any other file, or when the markers cannot be read (the signal
    loads anyway: load_mat_signal is unchanged)."""
    if not filepath or not is_spike2_file(str(filepath)):
        return {}
    try:
        st = Path(filepath).stat()
        key = (str(filepath), st.st_mtime, st.st_size)
    except OSError:
        key = None
    raw = _MARKER_CACHE.get(key) if key is not None else None
    if raw is None:
        try:
            raw = _spike2_markers_raw(_spike2_reader(str(filepath)))
        except Exception as exc:
            log.info("Spike2 markers not read from %s: %s", filepath, exc)
            return {}
        if key is not None:
            _MARKER_CACHE.clear()            # one file at a time
            _MARKER_CACHE[key] = raw
    t0 = float(t_start) if t_start and t_start > 0 else 0.0
    t1 = float(t_end) if t_end and t_end > 0 else None
    out = {}
    for name, times in raw.items():
        out[name] = [round(t - t0, 6) for t in times
                     if t >= t0 and (t1 is None or t <= t1)]
    return out


def list_spike2_channels(filepath: str) -> str:
    try:
        reader = _spike2_reader(filepath)
    except Exception as exc:
        return f"Could not read Spike2 file: {exc}"
    lines = ["Format: Spike2 (.smr)\n"]
    for c in _spike2_channels(reader):
        lines.append(f"  • {c['name']:<30} {c['n']:>11,} samples   {c['fs']:.2f} Hz")
    events = [str(e["name"]) for e in reader.header["event_channels"]]
    if events:
        lines.append("\n  Markers: " + ", ".join(events))
    return "\n".join(lines)


def load_mat_signal(
    filepath: str,
    channel: str,
) -> tuple[np.ndarray, str, list[str], Optional[float]]:
    """Load an ECG signal from a MATLAB .mat file (v5/v6 or v7.3 HDF5).

    Parameters
    ----------
    filepath : str   Path to the .mat file.
    channel  : str   Preferred variable name.  Auto-selects if not found.

    Returns
    -------
    signal           : np.ndarray (float64, 1-D)
    detected_channel : str
    all_keys         : list[str]
    detected_fs      : float or None  — sampling rate read from file metadata

    Raises
    ------
    ImportError  if the file is v7.3/HDF5 and h5py is not installed.
    ValueError   if no plausible ECG channel is found (see _pick_best_channel).
    Any other exception scipy.io.loadmat / h5py raise for a corrupt or
    unreadable file (e.g. OSError) propagates uncaught — callers that load
    user-supplied files should wrap this call and report the message rather
    than assume only the two cases above are possible.
    """
    # A native Spike2 file (.smr) is not MATLAB: read with neo.
    if is_spike2_file(filepath):
        return load_spike2_signal(filepath, channel)
    # Try MATLAB v5/v6 first
    try:
        raw = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        flat = _flatten_mat_dict(raw)
        sig, ch, keys = _pick_best_channel(flat, channel)
        detected_fs = _detect_fs_from_mat(raw, _base_key(ch))
        return sig, ch, keys, detected_fs
    except NotImplementedError:
        pass  # v7.3 HDF5 — fall through

    if not H5_AVAILABLE or h5py is None:
        raise ImportError(
            "MATLAB v7.3 (HDF5) file detected — h5py is required.\n"
            "Run:  pip install h5py"
        )
    # Single open shared by both the channel-flattening pass and the
    # fs-detection pass below (each used to open the file independently,
    # doubling I/O and HDF5 tree-parsing cost per load).
    with h5py.File(filepath, "r") as f:
        flat = _flatten_hdf5_file_obj(f)
        sig, ch, keys = _pick_best_channel(flat, channel)
        detected_fs = _detect_fs_from_hdf5_obj(f, _base_key(ch))
    return sig, ch, keys, detected_fs


def list_channels(filepath: str) -> str:
    """Return a human-readable listing of all channels in a .mat file.

    For MATLAB v5/v6, all keys from the raw dict are shown.
    For MATLAB v7.3 (HDF5), the same ``_flatten_hdf5_file`` helper that
    ``load_mat_signal`` uses is called so there is a single traversal code
    path for both listing and loading. Spike2 (.smr) files list their
    waveform channels and markers.
    """
    if is_spike2_file(filepath):
        return list_spike2_channels(filepath)
    try:
        raw = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        lines = ["Format: MATLAB v5/v6\n"]
        flat = _flatten_mat_dict(raw)
        for key, arr in flat.items():
            lines.append(f"  • {key:<30} {len(arr):>9,} samples   {arr.dtype}")
        return "\n".join(lines)
    except NotImplementedError:
        pass  # v7.3 HDF5 — fall through to HDF5 path below

    if not H5_AVAILABLE:
        return "MATLAB v7.3 file — install h5py to inspect channels"

    try:
        flat = _flatten_hdf5_file(filepath)
    except Exception as exc:
        return f"Could not read HDF5 channels: {exc}"

    lines = ["Format: MATLAB v7.3 (HDF5)\n"]
    for key, arr in flat.items():
        lines.append(f"  • {key:<30} {len(arr):>9,} samples   {arr.dtype}")
    if not flat:
        lines.append("  (no waveform channels found — minimum 100 samples)")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ════════════════════════════════════════════════════════════

