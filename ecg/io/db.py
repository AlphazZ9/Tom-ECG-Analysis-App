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
        verified_for_training INTEGER DEFAULT 0,  -- 1 = usable as ML training data
    )

The registry lives at SESSION_DIR / "ecg_registry.db".
"""
from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ecg.core.models import SESSION_DIR

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
    session_path TEXT    NOT NULL DEFAULT '',
    verified_for_training INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_fp ON recordings(filepath);
CREATE INDEX IF NOT EXISTS idx_saved ON recordings(saved_at DESC);
"""


_schema_ready = False   # apply _SCHEMA at most once per process, not per connection


def _conn() -> sqlite3.Connection:
    """Open (or create) the registry database, applying the schema."""
    global _schema_ready
    if not _DB_AVAILABLE:
        raise RuntimeError("sqlite3 is unavailable in this Python build")
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH), timeout=5, check_same_thread=False)
    con.row_factory = sqlite3.Row
    if not _schema_ready:
        con.executescript(_SCHEMA)
        # CREATE TABLE IF NOT EXISTS above is a no-op on a registry that
        # already existed before this column was added -- ALTER TABLE is the
        # only way to bring an older on-disk db.py up to date. Swallow the
        # "duplicate column" error on every run after the first.
        try:
            con.execute(
                "ALTER TABLE recordings ADD COLUMN verified_for_training "
                "INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        con.commit()
        _schema_ready = True
    return con


@contextlib.contextmanager
def _open():
    """Open a registry connection for one operation.

    ``with _conn() as con:`` (the previous pattern at every call site) only
    ever gets you sqlite3.Connection's own commit-on-success/rollback-on-
    exception __exit__ -- it does not close the connection afterwards.  That
    happened to work here because CPython's refcounting collects (and closes)
    the object promptly, but it's not behaviour the `with` statement itself
    guarantees, and would leak file handles under any GC that doesn't collect
    as eagerly. This wraps that same commit/rollback semantics in a try/
    finally that always calls con.close().
    """
    con = _conn()
    try:
        with con:
            yield con
    finally:
        con.close()


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_recording(
    filepath: str,
    fingerprint: str,
    session_path: str,
    stats: "Optional[dict[str, Any]]" = None,
    notes: str = "",
    verified_for_training: bool = False,
) -> None:
    """Insert or update a recording row with optional stats summary.

    *notes* is written as given, including an empty string -- it used to be
    merged via ``CASE WHEN excluded.notes != '' THEN excluded.notes ELSE
    recordings.notes END``, meant to avoid some hypothetical caller
    accidentally blanking notes with an unset default. In practice the only
    caller (SessionController.save_session) always passes the *current*
    ``self.app.session.recording_notes`` -- so a user who cleared their notes
    and saved would see the JSON session correctly record the empty string
    while this table silently kept the stale old note forever, since '' can
    never overwrite anything under that CASE. Plain overwrite matches
    set_notes()'s semantics below and keeps both write paths consistent.

    *verified_for_training* follows the same plain-overwrite rule -- the
    caller always passes the current checkbox state, so unmarking a
    previously verified file and saving correctly clears the flag here too.
    """
    stem = Path(filepath).stem
    now  = datetime.now().isoformat()
    s    = stats or {}
    try:
        with _open() as con:
            con.execute("""
                INSERT INTO recordings
                    (filepath, stem, fingerprint, saved_at, duration_s, n_peaks,
                     hr_mean, sdnn, rmssd, notes, session_path, verified_for_training)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(filepath) DO UPDATE SET
                    fingerprint  = excluded.fingerprint,
                    saved_at     = excluded.saved_at,
                    duration_s   = excluded.duration_s,
                    n_peaks      = excluded.n_peaks,
                    hr_mean      = excluded.hr_mean,
                    sdnn         = excluded.sdnn,
                    rmssd        = excluded.rmssd,
                    session_path = excluded.session_path,
                    notes        = excluded.notes,
                    verified_for_training = excluded.verified_for_training
            """, (filepath, stem, fingerprint, now,
                  s.get("duration_s"), s.get("n_peaks"),
                  s.get("hr_mean"), s.get("sdnn"), s.get("rmssd"),
                  notes, session_path, int(verified_for_training)))
    except Exception as exc:
        log.warning("db.upsert_recording failed: %s", exc)


def get_recording(filepath: str) -> "Optional[dict]":
    """Return the row for *filepath*, or None."""
    try:
        with _open() as con:
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
        with _open() as con:
            con.execute("""
                INSERT INTO recordings (filepath, stem, fingerprint, saved_at, notes)
                VALUES (?,?,?,?,?)
                ON CONFLICT(filepath) DO UPDATE SET notes=excluded.notes
            """, (filepath, Path(filepath).stem, "", datetime.now().isoformat(), notes))
    except Exception as exc:
        log.warning("db.set_notes: %s", exc)


def set_verified(filepath: str, verified: bool) -> None:
    """Update (or create) the verified-for-training flag for a recording.

    Partial update, same shape as set_notes() -- used by the ML training
    dialog's per-file "Remove" action, which must clear this flag without
    touching notes/stats/session_path for a row it otherwise knows nothing
    about.
    """
    try:
        with _open() as con:
            con.execute("""
                INSERT INTO recordings (filepath, stem, fingerprint, saved_at, verified_for_training)
                VALUES (?,?,?,?,?)
                ON CONFLICT(filepath) DO UPDATE SET verified_for_training=excluded.verified_for_training
            """, (filepath, Path(filepath).stem, "", datetime.now().isoformat(), int(verified)))
    except Exception as exc:
        log.warning("db.set_verified: %s", exc)


def recent_recordings(limit: int = 20) -> "list[dict]":
    """Return up to *limit* recent recordings ordered by saved_at DESC."""
    try:
        with _open() as con:
            rows = con.execute(
                "SELECT * FROM recordings ORDER BY saved_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("db.recent_recordings: %s", exc)
        return []


def verified_recordings(limit: int = 500) -> "list[dict]":
    """Return recordings marked verified-for-training, most recent first.

    Used by the ML training dialog to list candidate files/sample counts
    without opening every session JSON sidecar.
    """
    try:
        with _open() as con:
            rows = con.execute(
                "SELECT * FROM recordings WHERE verified_for_training=1 "
                "ORDER BY saved_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("db.verified_recordings: %s", exc)
        return []


def delete_recording(filepath: str) -> bool:
    """Remove the registry entry (does NOT delete the session payload file)."""
    try:
        with _open() as con:
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
        with _open() as con:
            rows = con.execute("""
                SELECT * FROM recordings
                WHERE stem LIKE ? OR notes LIKE ?
                ORDER BY saved_at DESC LIMIT ?
            """, (q, q, limit)).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("db.search: %s", exc)
        return []
