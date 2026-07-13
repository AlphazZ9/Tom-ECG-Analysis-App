"""
ecg.io.db
─────────
SQLite-backed session registry — replaces the per-file JSON sidecars as the
index layer while keeping individual session files as payload.

Schema (single table):
    recordings (
        id          INTEGER PRIMARY KEY,
        filepath    TEXT    UNIQUE,
        stem        TEXT,                 -- filename without extension
        fingerprint TEXT,                 -- SHA-256[:16] of path+size+mtime
        saved_at    TEXT,                 -- ISO-8601 last save timestamp
        duration_s  REAL,
        n_peaks     INTEGER,
        hr_mean     REAL,
        sdnn        REAL,
        rmssd       REAL,
        notes       TEXT    DEFAULT '',   -- free-text experiment notes
        session_path TEXT,               -- path to JSON sidecar payload
    )

The registry lives at SESSION_DIR / "ecg_registry.db".
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from models import SESSION_DIR

log = logging.getLogger("ecg")

try:
    import sqlite3  # noqa: F401
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

_DB_PATH = SESSION_DIR / "ecg_registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath     TEXT    NOT NULL UNIQUE,
    stem         TEXT    NOT NULL DEFAULT '',
    fingerprint  TEXT    NOT NULL DEFAULT '',
    saved_at     TEXT    NOT NULL DEFAULT '',
    duration_s   REAL,
    n_peaks      INTEGER,
    hr_mean      REAL,
    sdnn         REAL,
    rmssd        REAL,
    notes        TEXT    NOT NULL DEFAULT '',
    session_path TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fp ON recordings(filepath);
CREATE INDEX IF NOT EXISTS idx_saved ON recordings(saved_at DESC);
"""


def _conn() -> sqlite3.Connection:
    """Open (or create) the registry database, applying the schema."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH), timeout=5, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    con.commit()
    return con


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_recording(
    filepath: str,
    fingerprint: str,
    session_path: str,
    stats: "Optional[dict[str, Any]]" = None,
    notes: str = "",
) -> None:
    """Insert or update a recording row with optional stats summary."""
    stem = Path(filepath).stem
    now  = datetime.now().isoformat()
    s    = stats or {}
    try:
        with _conn() as con:
            con.execute("""
                INSERT INTO recordings
                    (filepath, stem, fingerprint, saved_at, duration_s, n_peaks,
                     hr_mean, sdnn, rmssd, notes, session_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(filepath) DO UPDATE SET
                    fingerprint  = excluded.fingerprint,
                    saved_at     = excluded.saved_at,
                    duration_s   = excluded.duration_s,
                    n_peaks      = excluded.n_peaks,
                    hr_mean      = excluded.hr_mean,
                    sdnn         = excluded.sdnn,
                    rmssd        = excluded.rmssd,
                    session_path = excluded.session_path,
                    notes = CASE WHEN excluded.notes != '' THEN excluded.notes
                                 ELSE recordings.notes END
            """, (filepath, stem, fingerprint, now,
                  s.get("duration_s"), s.get("n_peaks"),
                  s.get("hr_mean"), s.get("sdnn"), s.get("rmssd"),
                  notes, session_path))
    except Exception as exc:
        log.warning("db.upsert_recording failed: %s", exc)


def get_recording(filepath: str) -> "Optional[dict]":
    """Return the row for *filepath*, or None."""
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM recordings WHERE filepath=?", (filepath,)
            ).fetchone()
            return dict(row) if row else None
    except Exception as exc:
        log.debug("db.get_recording: %s", exc)
        return None


def get_notes(filepath: str) -> str:
    """Return the notes string for a recording, or ''."""
    row = get_recording(filepath)
    return (row or {}).get("notes", "")


def set_notes(filepath: str, notes: str) -> None:
    """Update (or create) the notes for a recording."""
    try:
        with _conn() as con:
            con.execute("""
                INSERT INTO recordings (filepath, stem, fingerprint, saved_at, notes)
                VALUES (?,?,?,?,?)
                ON CONFLICT(filepath) DO UPDATE SET notes=excluded.notes
            """, (filepath, Path(filepath).stem, "", datetime.now().isoformat(), notes))
    except Exception as exc:
        log.warning("db.set_notes: %s", exc)


def recent_recordings(limit: int = 20) -> "list[dict]":
    """Return up to *limit* recent recordings ordered by saved_at DESC."""
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT * FROM recordings ORDER BY saved_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("db.recent_recordings: %s", exc)
        return []


def delete_recording(filepath: str) -> bool:
    """Remove the registry entry (does NOT delete the session payload file)."""
    try:
        with _conn() as con:
            cur = con.execute(
                "DELETE FROM recordings WHERE filepath=?", (filepath,))
            return cur.rowcount > 0
    except Exception as exc:
        log.warning("db.delete_recording: %s", exc)
        return False


def search_recordings(query: str, limit: int = 50) -> "list[dict]":
    """Full-text search on stem + notes."""
    q = f"%{query}%"
    try:
        with _conn() as con:
            rows = con.execute("""
                SELECT * FROM recordings
                WHERE stem LIKE ? OR notes LIKE ?
                ORDER BY saved_at DESC LIMIT ?
            """, (q, q, limit)).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("db.search: %s", exc)
        return []
