"""
ecg.io.session
──────────────
Session save / load — persists filter params, peak edits, and analysis
results to a per-file JSON sidecar so work survives restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from models import SESSION_DIR, SESSION_SUFFIX, SESSION_VERSION

log = logging.getLogger("ecg")

class _SessionEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types in session state.

    - numpy scalars  → Python float/int
    - numpy arrays   → {"__ndarray__": true, "data": [...], "dtype": "..."}
    - pandas DataFrames are pre-serialised by _serialise_results before reaching
      the encoder, so they arrive as plain dicts and need no special handling.
    - Python sets    → {"__set__": true, "data": [...]}
    """
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return {"__ndarray__": True, "data": o.tolist(), "dtype": str(o.dtype)}
        if isinstance(o, set):
            return {"__set__": True, "data": sorted(o)}
        return super().default(o)


def _session_decode_hook(d: dict) -> Any:
    """object_hook for json.load — reconstructs ndarray and set sentinels."""
    if d.get("__ndarray__"):
        return np.array(d["data"], dtype=d.get("dtype", "float64"))
    if d.get("__set__"):
        return set(d["data"])
    return d


def _file_fingerprint(filepath: str) -> str:
    """Return a short SHA-256 hex digest based on path + file size + mtime.

    This is intentionally lightweight — we do NOT hash the full file content
    (which can be hundreds of MB).  The combination of path, byte-size, and
    modification time is unique enough in practice for a local session cache.
    """
    try:
        st  = os.stat(filepath)
        key = f"{filepath}|{st.st_size}|{st.st_mtime}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    except OSError:
        return hashlib.sha256(filepath.encode()).hexdigest()[:16]


def _session_path(filepath: str) -> Path:
    """Return the canonical .ecgsession path for *filepath*."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem + "_" + _file_fingerprint(filepath)
    return SESSION_DIR / (stem + SESSION_SUFFIX)


def save_session(filepath: str, state: dict) -> Path:
    """Serialise *state* to a JSON ``.ecgsession`` file in the cache dir.

    The payload embeds a version tag and a file fingerprint so load_session()
    can reject stale or mismatched caches.  All numpy arrays and DataFrames
    must be pre-converted to JSON-compatible form before calling this function
    (the caller uses ``_serialise_results`` for the ``results`` sub-key).

    Returns the path of the written file.
    """
    out_path = _session_path(filepath)
    payload  = {
        "version":     SESSION_VERSION,
        "fingerprint": _file_fingerprint(filepath),
        "saved_at":    datetime.now().isoformat(),
        "state":       state,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, cls=_SessionEncoder, indent=2, ensure_ascii=False)
    log.info("Session saved → %s  (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


def load_session(filepath: str) -> "Optional[dict]":
    """Load a previously-saved session for *filepath*, or return None.

    Returns None (silently) if:
    • no cache file exists
    • the cache is for a different file (fingerprint mismatch)
    • the schema version is outdated
    • the file is corrupt / unreadable

    Transparently migrates v4 pickle sessions to v5 JSON on first run.
    """
    sp = _session_path(filepath)
    if not sp.exists():
        return None

    # ── Try JSON (v5+) ──────────────────────────────────────────────────
    try:
        with open(sp, "r", encoding="utf-8") as fh:
            payload = json.load(fh, object_hook=_session_decode_hook)
        version = payload.get("version")
        if version not in (SESSION_VERSION, 5):
            log.info("load_session: version mismatch (%s vs %d) — ignoring cache",
                     version, SESSION_VERSION)
            sp.unlink()
            return None
        if payload.get("fingerprint") != _file_fingerprint(filepath):
            log.info("load_session: fingerprint mismatch — file changed, ignoring cache")
            sp.unlink()
            return None
        if version == 5:
            state = payload["state"]
            save_session(filepath, state)
            log.info("load_session: migrated v5 JSON → v%d", SESSION_VERSION)
            return state
        log.info("Session loaded ← %s  (saved %s)", sp, payload.get("saved_at", "?"))
        return payload["state"]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass   # not JSON — try legacy pickle migration below
    except Exception as exc:
        log.warning("load_session: failed to read %s: %s", sp, exc)
        return None

    # ── Fallback: migrate v4 pickle session (one-time) ──────────────────
    try:
        with open(sp, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") not in (3, 4):
            log.info("load_session: legacy pickle version %s — ignoring",
                     payload.get("version"))
            sp.unlink()
            return None
        if payload.get("fingerprint") != _file_fingerprint(filepath):
            log.info("load_session: legacy fingerprint mismatch — ignoring")
            sp.unlink()
            return None
        state = payload["state"]
        # Re-save as v5 JSON so the migration only happens once
        save_session(filepath, state)
        log.info("load_session: migrated v%s pickle → v%d JSON",
                 payload.get("version"), SESSION_VERSION)
        return state
    except Exception as exc:
        log.warning("load_session: legacy pickle migration failed for %s: %s", sp, exc)
        sp.unlink()
        return None


def delete_session(filepath: str) -> bool:
    """Delete any cached session for *filepath*.  Returns True if deleted."""
    sp = _session_path(filepath)
    if sp.exists():
        sp.unlink()
        log.info("Session deleted: %s", sp)
        return True
    return False





# ════════════════════════════════════════════════════════════
#  THEME DIALOG
# ════════════════════════════════════════════════════════════

