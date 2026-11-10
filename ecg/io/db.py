"""
ecg.io.db
─────────
SQLite-backed session registry — replaces the per-file JSON sidecars as the
index layer while keeping individual session files as payload.

Schema (two tables):
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
        project_name TEXT    DEFAULT '', -- user-entered cohort/study label
        rr_sparkline TEXT    DEFAULT '', -- JSON array, downsampled RR_ms series
                                          -- for the Recent-recordings sparkline
    )

    event_locked (                        -- cohort table of event-locked results
        id           INTEGER PRIMARY KEY,
        filepath     TEXT    NOT NULL,     -- recording the event belongs to
        stem         TEXT,
        subject      TEXT    NOT NULL DEFAULT '',   -- animal / subject identifier
        project_name TEXT    NOT NULL DEFAULT '',   -- cohort / study label
        label        TEXT    NOT NULL DEFAULT '',   -- event label, e.g. "dobutamine"
        onset_s      REAL    NOT NULL,     -- event onset, s from recording start
        pre_s        REAL,                 -- length of the pre-onset window (s)
        post_s       REAL,                 -- length of the post-onset window (s)
        method       TEXT,                 -- covariance method, from params
        window_s     REAL,                 -- Poincaré sliding-window length (s)
        params_json  TEXT    NOT NULL DEFAULT '{}', -- every analysis parameter
                                                    -- (ks, kp, bands, RR bounds...)
        metrics_json TEXT    NOT NULL DEFAULT '{}', -- {key: {"pre", "post", "delta"}}
        valid        INTEGER NOT NULL DEFAULT 1,    -- 0 = event flagged unusable
        saved_at     TEXT,                 -- ISO-8601 last save timestamp
        UNIQUE(filepath, label, onset_s, method, window_s)
    )

One event_locked row is one event of one recording analysed with one
(method, window_s) configuration: the per-event pre/post/delta values of the
robust-Poincaré (Candia-Rivera et al., J Physiol 2025) sympathetic/vagal
indices and their D/SD1/SD2 building blocks, time-resolved LF/HF, HR, ...
Because rCSI and rCPI are linear in ks/kp, the stored deltas of D, SD1 and SD2
are enough to re-calibrate ks/kp across a cohort without recomputing series.
``subject`` and ``project_name`` describe the recording, not the event: every
save rewrites them on all stored rows of that filepath (every label and
configuration), so one recording never shows up as two subjects or projects.

How the schema is applied (a deliberate deviation from simply adding the
event_locked DDL to ``_SCHEMA``): ``_SCHEMA`` still holds only the recordings
table, and the event_locked table and its indexes live in ``_EVENT_SCHEMA``,
created by ``_apply_event_schema`` with CREATE ... IF NOT EXISTS right after
the recordings ALTER TABLE migrations (and, like them, with errors logged and
swallowed). On an up-to-date registry every statement is a no-op, but on a
registry written by an older build they are writes, which a read-only file
rejects and which wait for any write lock another connection holds (a second
app instance mid-save, DB Browser for SQLite with pending edits). Inside
``_SCHEMA`` such a failure would take the whole recordings API down where it
used to keep working. Kept apart, an upgrade attempt stops at its first write
that fails, so it waits for a lock once, for at most ``_UPGRADE_BUSY_MS``
(not once per upgrade statement, and not the usual 5 s); a failed upgrade is
re-attempted by the event functions on every call (they need the table) but
by the recordings functions only every ``_UPGRADE_RETRY_S`` seconds, and the
recordings API keeps working meanwhile.

The registry lives at SESSION_DIR / "ecg_registry.db".
"""
from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Union

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
    verified_for_training INTEGER NOT NULL DEFAULT 0,
    project_name TEXT    NOT NULL DEFAULT '',
    rr_sparkline TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fp ON recordings(filepath);
CREATE INDEX IF NOT EXISTS idx_saved ON recordings(saved_at DESC);
"""

# Applied separately from _SCHEMA (see _apply_event_schema and the module
# docstring): on a registry from an older build these are real writes, not
# no-ops, and must not be able to take the recordings API down with them when
# the file cannot be written or is locked by another connection.
_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_locked (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath     TEXT    NOT NULL,
    stem         TEXT,
    subject      TEXT    NOT NULL DEFAULT '',
    project_name TEXT    NOT NULL DEFAULT '',
    label        TEXT    NOT NULL DEFAULT '',
    onset_s      REAL    NOT NULL,
    pre_s        REAL,
    post_s       REAL,
    method       TEXT,
    window_s     REAL,
    params_json  TEXT    NOT NULL DEFAULT '{}',
    metrics_json TEXT    NOT NULL DEFAULT '{}',
    valid        INTEGER NOT NULL DEFAULT 1,
    saved_at     TEXT,
    UNIQUE(filepath, label, onset_s, method, window_s)
);
CREATE INDEX IF NOT EXISTS idx_event_label   ON event_locked(label);
CREATE INDEX IF NOT EXISTS idx_event_project ON event_locked(project_name);
"""


# Normal wait for another connection's write lock (sqlite3.connect timeout).
_BUSY_TIMEOUT_S = 5
# Wait for a lock during the one-off upgrade writes of an older registry (the
# recordings ALTER TABLE blocks and the event_locked creation). Short, so a
# registry another process keeps locked cannot stall each call on the Tk main
# thread for the full _BUSY_TIMEOUT_S. SQLite applies it per statement, so
# _upgrade_schema() stops at the first failed write: one attempt waits once.
_UPGRADE_BUSY_MS = 200
# Columns added to recordings after its first release, in ALTER TABLE order.
_RECORDINGS_UPGRADE_COLUMNS = (
    "verified_for_training INTEGER NOT NULL DEFAULT 0",
    "project_name TEXT NOT NULL DEFAULT ''",
    "rr_sparkline TEXT NOT NULL DEFAULT ''",
)
# After an incomplete upgrade, the recordings functions skip the schema step
# for this long; the event functions re-attempt it on every call.
_UPGRADE_RETRY_S = 60.0

# Apply the schema at most once per process, not per connection. It stays
# False while the upgrade is incomplete (read-only or locked registry), so a
# later connection retries once the file is writable again.
_schema_ready = False
# time.monotonic() before which recordings-only calls do not retry an upgrade
# that could not complete (0.0 = no pending back-off).
_schema_retry_at = 0.0


def _add_recordings_column(con: "sqlite3.Connection", ddl: str) -> bool:
    """``ALTER TABLE recordings ADD COLUMN <ddl>``; True when the column exists.

    "duplicate column name" (every run after the first) means it is already
    there. Any other error -- a read-only or locked file -- is swallowed like
    before but reported as False, so the upgrade is retried later instead of
    the registry being marked up to date with the column still missing.
    """
    try:
        con.execute(f"ALTER TABLE recordings ADD COLUMN {ddl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" in str(exc).lower():
            return True
        log.debug("db: recordings column not added yet (%s): %s", ddl, exc)
        return False
    return True


def _apply_event_schema(con: "sqlite3.Connection") -> bool:
    """Create the event_locked table and its indexes; True when they exist.

    CREATE TABLE IF NOT EXISTS adds the table to a registry written by an
    older build without touching the recordings rows, so it needs no ALTER
    TABLE block (a column added to event_locked later must follow the ALTER
    TABLE pattern in _upgrade_schema(), exactly as recordings' columns did).
    On such a registry these statements are writes, so a read-only or locked
    file rejects them; that is logged and swallowed like the ALTER TABLE
    blocks, leaving the recordings API usable while the event functions fail
    gracefully ("no such table") through their own error handling.
    """
    try:
        con.executescript(_EVENT_SCHEMA)
        return True
    except sqlite3.OperationalError as exc:
        log.debug("db: event_locked table not available yet: %s", exc)
        return False


def _upgrade_schema(con: "sqlite3.Connection") -> bool:
    """Bring a registry written by an older build up to date; True when complete.

    Runs after ``_SCHEMA``. Every statement here is a no-op on an up-to-date
    registry (so it also succeeds on a read-only one) and a write otherwise.
    The busy timeout is _UPGRADE_BUSY_MS meanwhile, then the normal one is
    restored for the caller's own operation.

    SQLite applies a busy timeout per statement, so the attempt stops at the
    first write that fails: on a file another connection keeps write-locked
    every later write would fail too, each after its own wait (about 1.2 s
    for the four writes of the oldest registries, on every event call). One
    attempt therefore waits for a lock at most once, about _UPGRADE_BUSY_MS;
    the writes not tried are part of the whole upgrade that is retried later
    (on a read-only file every write fails at once, so nothing changes there).
    """
    con.execute(f"PRAGMA busy_timeout = {int(_UPGRADE_BUSY_MS)}")
    try:
        # CREATE TABLE IF NOT EXISTS in _SCHEMA is a no-op on a registry that
        # already existed before these columns were added -- ALTER TABLE is
        # the only way to bring an older on-disk db.py up to date. all() over
        # a generator short-circuits at the first column that fails.
        ok = all(_add_recordings_column(con, ddl)
                 for ddl in _RECORDINGS_UPGRADE_COLUMNS)
        con.commit()
        if ok:
            ok = _apply_event_schema(con)
    finally:
        con.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_S * 1000)}")
    return ok


def _conn(*, events: bool = False) -> sqlite3.Connection:
    """Open (or create) the registry database, applying the schema.

    The directory created is the one the database file actually lives in
    (``_DB_PATH.parent``, which is SESSION_DIR by default), so pointing
    ``_DB_PATH`` somewhere else never creates an empty SESSION_DIR as a side
    effect.

    *events* is True for the event_locked functions, which need that table:
    they retry an incomplete upgrade on every call, while the recordings
    functions retry it at most every _UPGRADE_RETRY_S seconds (see the module
    docstring).
    """
    global _schema_ready, _schema_retry_at
    if not _DB_AVAILABLE:
        raise RuntimeError("sqlite3 is unavailable in this Python build")
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH), timeout=_BUSY_TIMEOUT_S,
                          check_same_thread=False)
    con.row_factory = sqlite3.Row
    if not _schema_ready and (events or time.monotonic() >= _schema_retry_at):
        try:
            # Only no-ops once the recordings table exists, so this also
            # succeeds on a read-only registry (the upgrade writes follow).
            con.executescript(_SCHEMA)
            ready = _upgrade_schema(con)
        except Exception:
            # A registry that cannot take the schema (not a database, locked
            # past the timeout) must not leak the handle: the caller's
            # _open() never gets the connection to close it.
            con.close()
            raise
        _schema_ready = ready
        _schema_retry_at = 0.0 if ready else time.monotonic() + _UPGRADE_RETRY_S
    return con


@contextlib.contextmanager
def _open(*, events: bool = False):
    """Open a registry connection for one operation.

    ``with _conn() as con:`` (the previous pattern at every call site) only
    ever gets you sqlite3.Connection's own commit-on-success/rollback-on-
    exception __exit__ -- it does not close the connection afterwards.  That
    happened to work here because CPython's refcounting collects (and closes)
    the object promptly, but it's not behaviour the `with` statement itself
    guarantees, and would leak file handles under any GC that doesn't collect
    as eagerly. This wraps that same commit/rollback semantics in a try/
    finally that always calls con.close(). *events* is passed to _conn().
    """
    con = _conn(events=events)
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
    project_name: str = "",
    rr_sparkline: "Optional[list[float]]" = None,
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

    *rr_sparkline* is a small (already-downsampled, ~200 points max) RR_ms
    series, JSON-encoded and stored alongside the row so the Recent-
    recordings popup can draw its sparkline directly from this row instead
    of re-opening and deserializing the full .ecgsession JSON sidecar per
    row just to extract one column (see ecg.io.session.load_rr_series,
    still used as a fallback for rows saved before this column existed).
    """
    stem = Path(filepath).stem
    now  = datetime.now().isoformat()
    s    = stats or {}
    spark_json = json.dumps(rr_sparkline) if rr_sparkline else ""
    try:
        with _open() as con:
            con.execute("""
                INSERT INTO recordings
                    (filepath, stem, fingerprint, saved_at, duration_s, n_peaks,
                     hr_mean, sdnn, rmssd, notes, session_path, verified_for_training,
                     project_name, rr_sparkline)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    verified_for_training = excluded.verified_for_training,
                    project_name = excluded.project_name,
                    rr_sparkline = excluded.rr_sparkline
            """, (filepath, stem, fingerprint, now,
                  s.get("duration_s"), s.get("n_peaks"),
                  s.get("hr_mean"), s.get("sdnn"), s.get("rmssd"),
                  notes, session_path, int(verified_for_training),
                  project_name, spark_json))
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


# ── Event-locked cohort table ─────────────────────────────────────────────────

# Sub-fields every stored metric carries, in CSV column order.
_METRIC_FIELDS = ("pre", "post", "delta")

# Leading CSV columns (one per event_locked column, JSON payloads excluded).
_EVENT_CSV_COLUMNS = (
    "id", "filepath", "stem", "subject", "project_name", "label",
    "onset_s", "pre_s", "post_s", "method", "window_s", "valid", "saved_at",
)

# onset_s / window_s are part of the row identity. They are stored rounded to
# 1 µs (a canonical value, e.g. 612.4 rather than 612.4000000000001), but an
# existing row is matched within ±_KEY_TOL of the caller's unrounded value, not
# by equality of rounded values: two derivations of one onset a few ulp apart
# (sample / fs vs sample * (1 / fs), or a time vector built from dt) can sit
# on either side of a half-microsecond and round to different keys -- which
# happens at e.g. fs = 3200, 6400, 12800 or 16000 Hz. A stored value is within
# 0.5 µs of the value it was rounded from, so every re-derivation of the same
# onset matches it. Distinct events are always far more than 1 µs apart (one
# sample is >= 20 µs even at 50 kHz).
_KEY_DECIMALS = 6
_KEY_TOL = 1e-6


# Type names of pandas' missing-value markers (pd.NA, pd.NaT), matched by name
# so this module never imports pandas.
_NA_TYPE_NAMES = frozenset({"NAType", "NaTType"})

# Returned by _number() for a value that is present but not a usable number.
_INVALID = object()


def _is_missing(value: Any) -> bool:
    """True for None, a NaN (Python, numpy) and pandas' NA / NaT markers.

    These are how a DataFrame or a numeric form says "no value" -- e.g. a
    label column after a concat in which some rows had none. Text, including
    blank text, is not missing here (callers decide what blank text means).
    """
    if value is None or type(value).__name__ in _NA_TYPE_NAMES:
        return True
    if isinstance(value, (str, bytes)):
        return False
    try:
        return math.isnan(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _number(value: Any) -> Any:
    """Classify a numeric field: a finite float, None (missing) or _INVALID.

    Missing is None, NaN, pandas NA/NaT and blank or "nan" text. Numbers and
    numeric text -- "15" or " 612.4 " from a CSV round trip, a Tk StringVar
    or a params dict loaded from text -- give their float. Anything else
    (infinities, non-numeric text such as "15 s", containers) is _INVALID, so
    a caller can refuse it instead of silently reading it as missing.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (str, bytes)):
        value = value.strip()
        if not value:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return _INVALID
    if math.isnan(out):
        return None
    return out if math.isfinite(out) else _INVALID


def _finite_or_none(value: Any) -> "Optional[float]":
    """``float(value)`` for a finite number or numeric text, else None."""
    out = _number(value)
    return out if isinstance(out, float) else None


def _text(value: Any) -> str:
    """Stripped text of a label-like field; '' when missing (None, NaN, NA).

    ``str(float("nan"))`` would otherwise store the literal label "nan".
    """
    return "" if _is_missing(value) else str(value).strip()


def _key_float(value: Any) -> "Optional[float]":
    """A finite float rounded to the key resolution, or None."""
    out = _finite_or_none(value)
    return None if out is None else round(out, _KEY_DECIMALS)


def _near_clause(column: str, value: "Optional[float]") -> "tuple[str, list]":
    """SQL condition (and args) matching *column* within ±_KEY_TOL of *value*.

    None matches NULL only (NULL-safe, unlike ``column = NULL``). *column* is
    always one of this module's own column names, never caller input.
    """
    if value is None:
        return f"{column} IS NULL", []
    return f"{column} BETWEEN ? AND ?", [value - _KEY_TOL, value + _KEY_TOL]


# Text spellings of a false ``valid`` flag (CSV round-trips, UI checkbox vars).
# Blank text is not one of them: like in _number() it means "missing".
_FALSE_TEXT = frozenset({"0", "false", "no", "n", "off", "f"})


def _as_bool(value: Any, default: bool = True) -> bool:
    """Truthiness of a ``valid`` flag; a missing flag gives *default*.

    Missing is what _number() calls missing: None, NaN, pandas NA / NaT and
    blank or "nan" text (an empty CSV cell or form field), so an event whose
    flag was merely left empty is not silently excluded. Plain ``bool()``
    would read the strings "0" / "false" (e.g. a value that went through a
    CSV or a Tk StringVar) as True and silently keep an event the user
    excluded, so text is matched against the usual false spellings, and
    numeric text by its value ("0.0", as pandas writes a float column, is
    False). Other text ("yes", "true", "1") is True.
    """
    if _is_missing(value):
        return default
    if isinstance(value, (str, bytes)):
        text = value.decode(errors="ignore") if isinstance(value, bytes) else value
        text = text.strip().lower()
        if text in _FALSE_TEXT:
            return False
        number = _number(text)
        if number is None:                  # blank or "nan" text
            return default
        return number != 0.0 if isinstance(number, float) else True
    try:
        return bool(value)
    except Exception:       # e.g. a multi-element numpy array
        return default


def _jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into strict-JSON-safe Python values.

    numpy scalars/arrays are unwrapped through their ``tolist()`` (duck-typed,
    so this module still never imports numpy), tuples/sets become lists, dict
    keys become strings, and non-finite floats and pandas' NA / NaT markers
    become None -- the stored JSON must stay readable by tools that reject
    the non-standard ``NaN`` token. Anything else falls back to ``str()``.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if type(obj).__name__ in _NA_TYPE_NAMES:
        return None
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [_jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except Exception:
            pass
    return str(obj)


def _dumps(obj: Any) -> str:
    """Strict JSON text (no NaN/Infinity tokens) of an already-_jsonable value."""
    return json.dumps(obj, allow_nan=False, ensure_ascii=False)


def _loads_dict(text: Any) -> dict:
    """Decode a JSON object column; anything malformed or non-dict gives {}."""
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    try:
        out = json.loads(text)
    except (TypeError, ValueError) as exc:
        log.debug("db: undecodable JSON column: %s", exc)
        return {}
    return out if isinstance(out, dict) else {}


def _normalise_metrics(metrics: Any) -> "dict[str, dict]":
    """Normalise a ``{key: {"pre", "post", "delta", ...}}`` metrics mapping.

    pre/post/delta become finite floats (numeric text is parsed) or None, the
    latter also for anything that is not a number. A metric that omits
    ``delta`` but has finite pre and post gets ``delta = post - pre`` (the
    event-locked convention); an explicit ``delta: None`` is kept as None.
    Extra sub-fields (e.g. beat counts per window) are kept, JSON-sanitised.
    A metric whose value is not a mapping cannot be split into pre/post and
    is dropped with a debug log.
    """
    out: "dict[str, dict]" = {}
    if not isinstance(metrics, dict):
        return out
    for key, val in metrics.items():
        if not isinstance(val, dict):
            log.debug("db: metric %r is not a {pre, post, delta} mapping; skipped", key)
            continue
        entry: "dict[str, Any]" = {f: _finite_or_none(val.get(f)) for f in _METRIC_FIELDS}
        if "delta" not in val and entry["pre"] is not None and entry["post"] is not None:
            entry["delta"] = entry["post"] - entry["pre"]
        for sub, sub_val in val.items():
            if sub not in _METRIC_FIELDS:
                entry[str(sub)] = _jsonable(sub_val)
        out[str(key)] = entry
    return out


def _decode_event_row(row: Any) -> dict:
    """sqlite Row -> plain dict with ``metrics``/``params`` decoded, valid as bool."""
    d = dict(row)
    d["metrics"] = _loads_dict(d.pop("metrics_json", None))
    d["params"] = _loads_dict(d.pop("params_json", None))
    d["valid"] = bool(d.get("valid", 1))
    return d


def save_event_rows(
    filepath: str,
    rows: "Iterable[dict]",
    subject: str = "",
    project_name: str = "",
    params: "Optional[dict[str, Any]]" = None,
    *,
    replace: bool = False,
) -> int:
    """Insert or update the event-locked results of one recording.

    Each *rows* item describes one event::

        {"label": "dobutamine", "onset_s": 612.4, "pre_s": 30.0,
         "post_s": 30.0, "valid": True,
         "metrics": {"rCSI": {"pre": ..., "post": ..., "delta": ...},
                     "SD1":  {"pre": ..., "post": ..., "delta": ...}, ...}}

    *params* holds the analysis configuration shared by every row (method,
    window_s, ks, kp, LF/HF bands, RR bounds, ...). It is stored as JSON on
    each row, and its ``method`` / ``window_s`` entries fill the columns of
    the same name, which together with (filepath, label, onset_s) form the
    row's identity: saving the same event again with the same configuration
    updates it in place (its id is kept), while a different method or window
    length is stored as a separate row next to it. The stored params carry
    ``method`` / ``window_s`` as normalised for those columns (stripped text,
    a float), so the two never disagree.

    Numeric fields (onset_s, pre_s, post_s, window_s, metric values) accept
    numbers and numeric text ("15", " 612.4 "); None, NaN, pandas NA and
    blank text mean "missing". A ``window_s`` that is present but not a
    finite number ("15 s", inf, a list) refuses the whole save (logged,
    returns 0, nothing written or deleted): it is part of the row key, and
    reading it as missing would merge that configuration into the
    no-window one and overwrite its rows. Labels, subject and project name
    that are missing (None, NaN, pandas NA) are stored as ''.

    *subject* and *project_name* belong to the recording, not to one event:
    every save that reaches the database sets them on all stored rows of
    *filepath* -- other labels and other (method, window_s) configurations
    included, not only the rows this call writes -- in the same transaction.
    Re-saving one label after filling in the Subject ID therefore cannot
    leave the recording's other events under the old (or no) subject, which
    would make one animal count as two subjects in the cohort statistics and
    split it across project filters. Those other rows keep their metrics,
    params and ``saved_at``; a blank subject or project given here clears it
    on all of them alike.

    The upsert is NULL-safe: a configuration without ``method`` or
    ``window_s`` (or with an empty method name) stores NULL there and still
    updates the existing row instead of duplicating it (a plain SQL UNIQUE
    constraint treats NULLs as distinct). It is also tolerant to float
    noise: onset_s and window_s are stored rounded to 1 µs, and an existing
    row matches when its stored values lie within 1 µs of the ones given, so
    an onset re-derived a few ulp away (``i / fs`` vs ``i * (1 / fs)``)
    updates the row instead of adding a duplicate event; the stored onset
    keeps the value of the first save. Several rows for the same event in one
    call collapse to the last one. A missing ``valid`` (None, NaN, pandas NA,
    blank or "nan" text) means valid; text such as "0" / "false" / "0.0"
    means invalid.

    A row is rejected (skipped, logged) when it is not a dict, has no finite
    ``onset_s``, or has a ``pre_s`` / ``post_s`` that is present but not a
    finite number. Metric values that are not finite numbers are stored as
    null.

    With ``replace=True`` every row of *filepath* stored with this same
    (method, window_s) configuration that is not among *rows* is deleted, in
    the same transaction as the upserts (re-saved events keep their ids) --
    use it to re-save a recording's complete event list so an event removed
    since the last save does not linger in the cohort table. An explicitly
    empty *rows* therefore clears that configuration. If any
    supplied row is rejected, the delete is skipped (logged) and the valid
    rows are upserted as with ``replace=False``: a malformed row can never
    erase stored events. ``rows=None`` writes nothing, even with replace.

    Returns the number of distinct event rows written (inserted or updated);
    0 when nothing could be written, including when sqlite is unavailable or
    the database cannot be opened (logged, never raised).
    """
    if not filepath:
        log.warning("db.save_event_rows: empty filepath; nothing saved")
        return 0
    if rows is None:
        log.debug("db.save_event_rows: rows is None; nothing saved")
        return 0
    try:
        items = list(rows)
    except TypeError:
        log.warning("db.save_event_rows: rows is not iterable (%s); nothing saved",
                    type(rows).__name__)
        return 0
    cfg = _jsonable(dict(params)) if isinstance(params, dict) else {}
    method_raw = cfg.get("method")
    # An empty method name means "unspecified", the same key as no method.
    method = None if method_raw is None else (str(method_raw).strip() or None)
    # Parsed from the caller's value, not from cfg: _jsonable has already
    # turned an infinite window into None there, which would read as missing.
    window_given = params.get("window_s") if isinstance(params, dict) else None
    window_raw = _number(window_given)                    # matched within ±_KEY_TOL
    if window_raw is _INVALID:
        log.warning("db.save_event_rows: params window_s %r is not a number; "
                    "nothing saved for %s", window_given, filepath)
        return 0
    window_key = _key_float(window_raw)                   # stored on insert
    if "method" in cfg:
        cfg["method"] = method
    if "window_s" in cfg:
        cfg["window_s"] = window_raw
    params_json = _dumps(cfg)
    subject = _text(subject)
    project_name = _text(project_name)

    prepared: "dict[tuple[str, float], tuple]" = {}
    rejected = 0
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            log.warning("db.save_event_rows: row %d is not a dict; skipped", i)
            rejected += 1
            continue
        onset_raw = _number(row.get("onset_s"))
        if not isinstance(onset_raw, float):
            log.warning("db.save_event_rows: row %d has no finite onset_s (%r); "
                        "skipped", i, row.get("onset_s"))
            rejected += 1
            continue
        pre_s, post_s = _number(row.get("pre_s")), _number(row.get("post_s"))
        if pre_s is _INVALID or post_s is _INVALID:
            log.warning("db.save_event_rows: row %d has a non-numeric pre_s/post_s "
                        "(%r, %r); skipped", i, row.get("pre_s"), row.get("post_s"))
            rejected += 1
            continue
        onset_key = _key_float(onset_raw)
        label = _text(row.get("label"))
        valid = _as_bool(row.get("valid"), default=True)
        metrics_json = _dumps(_normalise_metrics(row.get("metrics")))
        prepared[(label, onset_key)] = (
            label, onset_raw, onset_key, pre_s, post_s, metrics_json, int(valid))
    if rejected and replace:
        log.warning("db.save_event_rows: %d of %d rows rejected, so the stored "
                    "events of %s were kept (replace skipped)",
                    rejected, len(items), filepath)
        replace = False
    if not prepared and not replace:
        return 0

    stem = Path(filepath).stem
    now = datetime.now().isoformat()
    win_sql, win_args = _near_clause("window_s", window_raw)
    try:
        with _open(events=True) as con:
            written: "set[int]" = set()
            for (label, onset_raw, onset_key, pre_s, post_s,
                 metrics_json, valid) in prepared.values():
                on_sql, on_args = _near_clause("onset_s", onset_raw)
                hit = con.execute(
                    "SELECT id FROM event_locked WHERE filepath=? AND label=? "
                    f"AND method IS ? AND {win_sql} AND {on_sql} "
                    "ORDER BY ABS(onset_s - ?), id LIMIT 1",
                    [filepath, label, method, *win_args, *on_args, onset_raw],
                ).fetchone()
                if hit is not None:
                    con.execute("""
                        UPDATE event_locked SET
                            stem=?, subject=?, project_name=?, pre_s=?, post_s=?,
                            params_json=?, metrics_json=?, valid=?, saved_at=?
                        WHERE id=?
                    """, (stem, subject, project_name, pre_s, post_s,
                          params_json, metrics_json, valid, now, hit["id"]))
                    written.add(int(hit["id"]))
                else:
                    cur = con.execute("""
                        INSERT INTO event_locked
                            (filepath, stem, subject, project_name, label,
                             onset_s, pre_s, post_s, method, window_s,
                             params_json, metrics_json, valid, saved_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (filepath, stem, subject, project_name, label,
                          onset_key, pre_s, post_s, method, window_key,
                          params_json, metrics_json, valid, now))
                    written.add(int(cur.lastrowid))
            # The recording's identity follows its latest save on every row,
            # not only the rows upserted above (see the docstring).
            con.execute(
                "UPDATE event_locked SET subject=?, project_name=? "
                "WHERE filepath=? AND (subject != ? OR project_name != ?)",
                (subject, project_name, filepath, subject, project_name))
            if replace:
                # After the upserts (same transaction), so re-saved events
                # keep their ids and only the ones no longer listed go.
                stale = [int(r["id"]) for r in con.execute(
                    "SELECT id FROM event_locked WHERE filepath=? AND method IS ? "
                    f"AND {win_sql}", [filepath, method, *win_args]).fetchall()
                    if int(r["id"]) not in written]
                con.executemany("DELETE FROM event_locked WHERE id=?",
                                [(i,) for i in stale])
        return len(written)
    except Exception as exc:
        log.warning("db.save_event_rows failed: %s", exc)
        return 0


def list_event_rows(
    label: "Optional[str]" = None,
    project_name: "Optional[str]" = None,
    valid_only: bool = True,
    filepath: "Optional[str]" = None,
) -> "list[dict]":
    """Return event-locked rows, optionally filtered, as plain dicts.

    *label*, *project_name* and *filepath* are exact-match filters (None =
    no filter; ``label=""``, or a NaN / pandas NA label, selects unlabelled
    events). With *valid_only*
    (the default) rows flagged ``valid=0`` are left out.

    Each dict carries every event_locked column except the two JSON payloads,
    which are decoded into ``metrics`` ({key: {"pre", "post", "delta", ...}})
    and ``params`` (the analysis configuration); ``valid`` is a bool. Rows
    are ordered for a cohort table: label, project, subject, recording, onset.
    Returns [] when sqlite is unavailable or the query fails (logged).
    """
    clauses: "list[str]" = []
    args: "list[Any]" = []
    # A missing label/project (NaN, pandas NA) means '', as save_event_rows
    # stores it; only None means "no filter".
    if label is not None:
        clauses.append("label=?")
        args.append(_text(label))
    if project_name is not None:
        clauses.append("project_name=?")
        args.append(_text(project_name))
    if filepath is not None:
        clauses.append("filepath=?")
        args.append(filepath)
    if valid_only:
        clauses.append("valid=1")
    sql = "SELECT * FROM event_locked"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += (" ORDER BY label COLLATE NOCASE, project_name COLLATE NOCASE, "
            "subject COLLATE NOCASE, stem COLLATE NOCASE, filepath, onset_s, "
            "method, window_s, id")
    try:
        with _open(events=True) as con:
            return [_decode_event_row(r) for r in con.execute(sql, args).fetchall()]
    except Exception as exc:
        log.debug("db.list_event_rows: %s", exc)
        return []


def list_event_labels() -> "list[str]":
    """Return the distinct non-empty event labels, sorted case-insensitively.

    Every stored row counts, including ones flagged invalid, so a label
    stays selectable while all its events are excluded. Unlabelled events
    (label '') are not listed; ``list_event_rows(label="")`` still reaches
    them. Returns [] when sqlite is unavailable or the query fails (logged).
    """
    try:
        with _open(events=True) as con:
            rows = con.execute(
                "SELECT DISTINCT label FROM event_locked WHERE label != '' "
                "ORDER BY label COLLATE NOCASE, label").fetchall()
            return [r["label"] for r in rows]
    except Exception as exc:
        log.debug("db.list_event_labels: %s", exc)
        return []


def delete_event_rows(filepath: str, label: "Optional[str]" = None) -> int:
    """Delete the event rows of *filepath* (only those with *label* if given).

    Every (method, window_s) configuration of the matched events goes. The
    recordings row and the session payload file are left untouched. Returns
    the number of rows deleted; 0 on failure or when sqlite is unavailable
    (logged).
    """
    try:
        with _open(events=True) as con:
            if label is None:
                cur = con.execute(
                    "DELETE FROM event_locked WHERE filepath=?", (filepath,))
            else:
                cur = con.execute(
                    "DELETE FROM event_locked WHERE filepath=? AND label=?",
                    (filepath, _text(label)))
            return max(cur.rowcount, 0)
    except Exception as exc:
        log.warning("db.delete_event_rows: %s", exc)
        return 0


def _csv_cell(value: Any) -> Any:
    """One CSV cell: None/NaN -> '', bool -> 1/0, containers -> JSON text.

    The value goes through _jsonable first, so numpy scalars in rows built in
    memory (rather than read back from the database) are written as plain
    numbers -- ``repr(np.float64(0.1))`` is ``'np.float64(0.1)'`` on numpy 2.
    """
    value = _jsonable(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return _dumps(value)
    if isinstance(value, float):
        return repr(value)          # shortest round-trip text; finite here
    return value


def export_event_rows_csv(
    path: "Union[str, Path]",
    rows: "Iterable[dict]",
    *,
    delimiter: str = ",",
) -> int:
    """Write event-locked *rows* (as returned by list_event_rows) to a CSV file.

    One line per event. The leading columns are the event_locked columns
    (id, filepath, stem, subject, project_name, label, onset_s, pre_s,
    post_s, method, window_s, valid, saved_at); then every metric is
    flattened into ``<key>_pre``, ``<key>_post``, ``<key>_delta`` (plus
    ``<key>_<field>`` for any extra sub-field), metrics in first-seen order
    across *rows* so rows with different metric sets share one header; then
    each analysis parameter as ``param_<name>`` (method and window_s already
    have their own columns). Missing or non-finite values are empty cells;
    ``valid`` is 1/0. Items that are not dicts are skipped.

    Rows may carry ``metrics``/``params`` either decoded (list_event_rows) or
    as their raw ``metrics_json``/``params_json`` text. The file is UTF-8 with
    a BOM so Excel shows accented subject names correctly; pass
    ``delimiter=";"`` for a locale whose Excel expects semicolons.
    Returns the number of data lines written. Unlike the database functions
    this raises OSError when the file cannot be written, so the caller can
    report it to the user.
    """
    decoded: "list[tuple[dict, dict, dict]]" = []
    metric_fields: "dict[str, list[str]]" = {}
    param_names: "list[str]" = []
    for row in (rows if rows is not None else []):
        if not isinstance(row, dict):
            log.debug("db.export_event_rows_csv: non-dict row skipped")
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = _loads_dict(row.get("metrics_json"))
        params = row.get("params")
        if not isinstance(params, dict):
            params = _loads_dict(row.get("params_json"))
        for key, entry in metrics.items():
            fields = metric_fields.setdefault(str(key), list(_METRIC_FIELDS))
            if isinstance(entry, dict):
                for sub in entry:
                    if str(sub) not in fields:
                        fields.append(str(sub))
        for name in params:
            if name not in ("method", "window_s") and name not in param_names:
                param_names.append(name)
        decoded.append((row, metrics, params))

    header = list(_EVENT_CSV_COLUMNS)
    header += [f"{key}_{sub}" for key, subs in metric_fields.items() for sub in subs]
    header += [f"param_{name}" for name in param_names]

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(header)
        for row, metrics, params in decoded:
            line = [_csv_cell(row.get(col)) for col in _EVENT_CSV_COLUMNS]
            for key, subs in metric_fields.items():
                entry = metrics.get(key)
                entry = entry if isinstance(entry, dict) else {}
                line += [_csv_cell(entry.get(sub)) for sub in subs]
            line += [_csv_cell(params.get(name)) for name in param_names]
            writer.writerow(line)
    return len(decoded)
