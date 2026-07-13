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

def _serialise_results(results: dict) -> dict:
    """Convert analysis results dict to a plain JSON-compatible form.

    DataFrames become {orient:'list'} dicts; numpy arrays become lists;
    other primitives are passed through unchanged.
    """
    out: dict = {}
    for key, val in results.items():
        if val is None:
            out[key] = None
        elif isinstance(val, pd.DataFrame):
            out[key] = {"__dataframe__": True, "data": val.to_dict(orient="list")}
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
            out[key] = pd.DataFrame(val["data"])
        elif isinstance(val, dict) and val.get("__ndarray__"):
            out[key] = np.array(val["data"], dtype=val.get("dtype", "float64"))
        else:
            out[key] = val
    return out


# ════════════════════════════════════════════════════════════
#  MAT / HDF5 LOADER
# ════════════════════════════════════════════════════════════

def _flatten_mat_dict(raw: dict) -> dict[str, np.ndarray]:
    """Convert a raw scipy.io.loadmat dict into {name: 1-D array} pairs.

    Handles Spike2 structs that store data under a `.values` or `.data`
    attribute, falling back to direct array conversion. Non-array keys
    (metadata starting with '_') are skipped silently.
    """
    flat: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        try:
            if hasattr(value, "values"):
                flat[key] = np.array(value.values).flatten()
            elif hasattr(value, "data"):
                flat[key] = np.array(value.data).flatten()
            else:
                flat[key] = np.array(value).flatten()
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
                    log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 1715, _exc)
        # Some exports store fs directly
        for attr in ("fs", "Fs", "sample_rate", "samplerate", "SampleRate"):
            v = getattr(ch, attr, None)
            if v is not None:
                try:
                    fs_val = float(np.array(v).flat[0])
                    if 100 <= fs_val <= 500_000:
                        return round(fs_val)
                except Exception as _exc:
                    log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 1725, _exc)

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
                log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 1737, _exc)
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
                    log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 1758, _exc)
        for attr in ("fs", "Fs", "sample_rate"):
            v = grp.get(attr)
            if v is not None:
                try:
                    fs_val = float(np.array(v).flat[0])
                    if 100 <= fs_val <= 500_000:
                        return round(fs_val)
                except Exception as _exc:
                    log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 1767, _exc)
    # Top-level datasets
    for key in ("fs", "Fs", "sample_rate", "samplerate"):
        v = f.get(key)
        if v is not None:
            try:
                fs_val = float(np.array(v).flat[0])
                if 100 <= fs_val <= 500_000:
                    return round(fs_val)
            except Exception as _exc:
                log.debug("%s at %s:%d — %s", type(_exc).__name__, __name__, 1777, _exc)
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
                    arr = np.array(group[sub]).flatten()
                    if len(arr) > 100:
                        flat[key] = arr
                except Exception as exc:
                    log.debug("HDF5 '%s/%s': %s", key, sub, exc)
                found = True
                break
        if not found and isinstance(group, h5py.Dataset):
            try:
                arr = np.array(group).flatten()
                if len(arr) > 100:
                    flat[key] = arr
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
    # Try MATLAB v5/v6 first
    try:
        raw = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        flat = _flatten_mat_dict(raw)
        sig, ch, keys = _pick_best_channel(flat, channel)
        detected_fs = _detect_fs_from_mat(raw, ch)
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
        detected_fs = _detect_fs_from_hdf5_obj(f, ch)
    return sig, ch, keys, detected_fs


def list_channels(filepath: str) -> str:
    """Return a human-readable listing of all channels in a .mat file.

    For MATLAB v5/v6, all keys from the raw dict are shown.
    For MATLAB v7.3 (HDF5), the same ``_flatten_hdf5_file`` helper that
    ``load_mat_signal`` uses is called so there is a single traversal code
    path for both listing and loading.
    """
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

