"""
Event-locked cohort table of the SQLite registry (ecg.io.db): save / list /
label listing / delete / CSV export of per-event pre/post/delta metrics, the
NULL-safe, float-noise-tolerant upsert key, numeric text and pandas missing
markers in the input, graceful failure when sqlite or the file is unusable,
and the in-place upgrade of a registry written before the table existed
(including one that is read-only, or write-locked by another connection, at
first).

Every test runs against a throw-away database under tmp_path (the
``registry`` fixture repoints db._DB_PATH / db.SESSION_DIR and forces the
schema to be re-applied) -- never the user's real ~/.ecg_sessions registry.
"""
from __future__ import annotations

import contextlib
import csv
import math
import os
import sqlite3
import stat
import time

import numpy as np
import pytest

from ecg.io import db


REC_A = "C:/data/mouse_A.mat"
REC_B = "C:/data/mouse_B.mat"

PARAMS = {
    "method": "exact", "window_s": 15.0, "ks": 3.0, "kp": 12.0,
    "lf_band": [0.4, 1.5], "hf_band": [1.5, 5.0], "rr_bounds_ms": [50.0, 250.0],
}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A fresh registry file under tmp_path; returns its path."""
    session_dir = tmp_path / "sessions"
    db_path = session_dir / "ecg_registry.db"
    monkeypatch.setattr(db, "SESSION_DIR", session_dir)
    monkeypatch.setattr(db, "_DB_PATH", db_path)
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_schema_retry_at", 0.0)
    return db_path


def _event(onset, label="dobutamine", valid=True, **metrics):
    """One save_event_rows() row; metrics default to a small D/SD1/SD2 set."""
    if not metrics:
        metrics = {
            "D":    {"pre": 140.0, "post": 120.0, "delta": -20.0},
            "SD1":  {"pre": 2.0, "post": 1.5, "delta": -0.5},
            "SD2":  {"pre": 6.0, "post": 8.0, "delta": 2.0},
            "rCSI": {"pre": 1.0, "post": 2.0},          # delta derived
        }
    return {"label": label, "onset_s": onset, "pre_s": 30.0, "post_s": 30.0,
            "valid": valid, "metrics": metrics}


def _raw_rows(db_path):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM event_locked ORDER BY id")]
    finally:
        con.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def test_schema_has_table_unique_key_and_indexes(registry):
    assert db.list_event_rows() == []          # opens the db, applies the schema
    assert registry.exists()

    con = sqlite3.connect(str(registry))
    try:
        cols = {r[1]: r for r in con.execute("PRAGMA table_info(event_locked)")}
        indexes = {r[1]: r for r in con.execute("PRAGMA index_list(event_locked)")}
        unique_cols = None
        for name, row in indexes.items():
            if row[2]:   # unique
                unique_cols = [c[2] for c in con.execute(f"PRAGMA index_info('{name}')")]
    finally:
        con.close()

    assert list(cols) == [
        "id", "filepath", "stem", "subject", "project_name", "label", "onset_s",
        "pre_s", "post_s", "method", "window_s", "params_json", "metrics_json",
        "valid", "saved_at",
    ]
    # (cid, name, type, notnull, dflt_value, pk)
    assert cols["filepath"][3] == 1 and cols["onset_s"][3] == 1
    assert cols["label"][3] == 1 and cols["label"][4] == "''"
    assert cols["valid"][4] == "1"
    assert cols["params_json"][4] == "'{}'" and cols["metrics_json"][4] == "'{}'"
    assert {"idx_event_label", "idx_event_project"} <= set(indexes)
    assert unique_cols == ["filepath", "label", "onset_s", "method", "window_s"]


# ── Round trip ────────────────────────────────────────────────────────────────

def test_save_and_list_round_trip(registry):
    n = db.save_event_rows(
        REC_A, [_event(612.4), _event(1800.0, label="saline")],
        subject="M12", project_name="SHR-2026", params=PARAMS)
    assert n == 2

    rows = db.list_event_rows()
    assert [r["label"] for r in rows] == ["dobutamine", "saline"]
    r = rows[0]
    assert r["filepath"] == REC_A and r["stem"] == "mouse_A"
    assert r["subject"] == "M12" and r["project_name"] == "SHR-2026"
    assert r["onset_s"] == pytest.approx(612.4)
    assert r["pre_s"] == 30.0 and r["post_s"] == 30.0
    assert r["method"] == "exact" and r["window_s"] == 15.0
    assert r["valid"] is True
    assert r["saved_at"]
    assert "metrics_json" not in r and "params_json" not in r
    assert r["params"] == PARAMS
    assert r["metrics"]["SD2"] == {"pre": 6.0, "post": 8.0, "delta": 2.0}
    # A metric without an explicit delta gets post - pre.
    assert r["metrics"]["rCSI"] == {"pre": 1.0, "post": 2.0, "delta": 1.0}


def test_numpy_and_non_finite_values_are_stored_as_strict_json(registry):
    metrics = {
        "HR":    {"pre": np.float64(520.5), "post": np.float32(560.0),
                  "delta": np.float64(39.5), "n_beats": np.int64(240)},
        "LF_HF": {"pre": float("nan"), "post": 1.2},     # delta needs both -> None
        "SD1":   {"pre": 1.0, "post": float("inf"), "delta": None},
        "bad":   3.0,                                     # not a mapping -> dropped
    }
    params = {**PARAMS, "window_s": np.float64(15.0), "bands": np.array([0.4, 1.5])}
    assert db.save_event_rows(REC_A, [_event(np.float64(100.0), **metrics)],
                              params=params) == 1

    raw = _raw_rows(registry)[0]
    assert "NaN" not in raw["metrics_json"] and "Infinity" not in raw["metrics_json"]

    m = db.list_event_rows()[0]["metrics"]
    assert m["HR"] == {"pre": 520.5, "post": 560.0, "delta": 39.5, "n_beats": 240}
    assert m["LF_HF"] == {"pre": None, "post": 1.2, "delta": None}
    assert m["SD1"] == {"pre": 1.0, "post": None, "delta": None}
    assert "bad" not in m
    p = db.list_event_rows()[0]["params"]
    assert p["bands"] == [0.4, 1.5] and p["window_s"] == 15.0


def test_rows_without_finite_onset_or_not_dicts_are_skipped(registry):
    rows = [_event(10.0), {"label": "x"}, _event(float("nan")), "junk", _event(None)]
    assert db.save_event_rows(REC_A, rows, params=PARAMS) == 1
    assert len(db.list_event_rows()) == 1
    assert db.save_event_rows("", [_event(5.0)], params=PARAMS) == 0
    assert db.save_event_rows(REC_A, [], params=PARAMS) == 0


# ── Numeric text and missing-value markers in the input ──────────────────────

def test_window_given_as_text_is_parsed_not_collapsed_to_null(registry):
    # Two windows given as text are two configurations; reading text as
    # "no window" put both under the NULL-window key, the second overwriting
    # the first.
    ev15 = _event(10.0, SD1={"pre": 1.0, "post": 2.0})
    ev30 = _event(10.0, SD1={"pre": 5.0, "post": 9.0})
    assert db.save_event_rows(REC_A, [ev15], params={**PARAMS, "window_s": "15"}) == 1
    assert db.save_event_rows(REC_A, [ev30], params={**PARAMS, "window_s": " 30 "}) == 1
    rows = db.list_event_rows()
    assert [(r["window_s"], r["metrics"]["SD1"]["pre"]) for r in rows] == [
        (15.0, 1.0), (30.0, 5.0)]
    # The stored params agree with the key column.
    assert [r["params"]["window_s"] for r in rows] == [15.0, 30.0]

    # "15" and 15.0 are the same configuration: the row is updated in place.
    assert db.save_event_rows(REC_A, [_event(10.0)], params=PARAMS) == 1
    rows = db.list_event_rows()
    assert len(rows) == 2 and rows[0]["metrics"]["SD2"]["delta"] == 2.0


def test_onset_window_lengths_and_metrics_given_as_text_are_parsed(registry):
    row = {"label": "dobutamine", "onset_s": " 612.4 ", "pre_s": "30", "post_s": b"45",
           "metrics": {"SD1": {"pre": "1.5", "post": "2.5"}, "HR": {"pre": "n/a"}}}
    assert db.save_event_rows(REC_A, [row], params=PARAMS) == 1
    r = db.list_event_rows()[0]
    assert (r["onset_s"], r["pre_s"], r["post_s"]) == (612.4, 30.0, 45.0)
    assert r["metrics"]["SD1"] == {"pre": 1.5, "post": 2.5, "delta": 1.0}
    assert r["metrics"]["HR"] == {"pre": None, "post": None, "delta": None}
    # The text onset is the same event as the float one.
    assert db.save_event_rows(REC_A, [_event(612.4)], params=PARAMS) == 1
    assert len(db.list_event_rows()) == 1

    # Missing window lengths (blank text, NaN) are stored as NULL.
    blank = {**_event(1.0), "pre_s": " ", "post_s": float("nan")}
    assert db.save_event_rows(REC_B, [blank], params=PARAMS) == 1
    b = db.list_event_rows(filepath=REC_B)[0]
    assert b["pre_s"] is None and b["post_s"] is None


@pytest.mark.parametrize("bad", ["15 s", "abc", float("inf"), "-inf",
                                 [15.0], np.array([15.0, 30.0]), {"s": 15}])
def test_unparseable_window_refuses_the_save(registry, bad):
    db.save_event_rows(REC_A, [_event(10.0)], params={**PARAMS, "window_s": None})
    db.save_event_rows(REC_A, [_event(10.0)], params=PARAMS)
    before = _raw_rows(registry)
    assert len(before) == 2

    cfg = {**PARAMS, "window_s": bad}
    newer = _event(10.0, SD1={"pre": 7.0, "post": 8.0})
    assert db.save_event_rows(REC_A, [newer], params=cfg) == 0
    assert db.save_event_rows(REC_A, [newer], params=cfg, replace=True) == 0
    assert db.save_event_rows(REC_A, [], params=cfg, replace=True) == 0
    assert _raw_rows(registry) == before            # nothing written or deleted


def test_rows_with_non_numeric_window_lengths_are_rejected(registry):
    bad = {**_event(20.0), "pre_s": "thirty"}
    worse = {**_event(30.0), "post_s": float("inf")}
    assert db.save_event_rows(REC_A, [_event(10.0), bad, worse], params=PARAMS) == 1
    assert [r["onset_s"] for r in db.list_event_rows()] == [10.0]


def test_missing_labels_from_pandas_are_stored_as_unlabelled(registry):
    pd = pytest.importorskip("pandas")
    # A concat in which some rows had no label leaves NaN in the column.
    df = pd.concat([pd.DataFrame([{"label": "dob", "onset_s": 10.0}]),
                    pd.DataFrame([{"onset_s": 20.0}, {"onset_s": 30.0}])],
                   ignore_index=True)
    rows = df.to_dict("records")
    assert isinstance(rows[1]["label"], float) and math.isnan(rows[1]["label"])
    rows[2]["label"] = pd.NA
    rows += [{"label": np.float32("nan"), "onset_s": 40.0},
             {"label": None, "onset_s": 50.0}]
    assert db.save_event_rows(REC_A, rows, subject=float("nan"),
                              project_name=pd.NA, params=PARAMS) == 5

    assert db.list_event_labels() == ["dob"]                   # no "nan" label
    unlabelled = db.list_event_rows(label="")
    assert [r["onset_s"] for r in unlabelled] == [20.0, 30.0, 40.0, 50.0]
    assert {(r["subject"], r["project_name"]) for r in db.list_event_rows()} == {("", "")}
    # A NaN label as a filter or a delete target means "unlabelled" too.
    assert len(db.list_event_rows(label=float("nan"))) == 4
    assert len(db.list_event_rows(project_name=pd.NA)) == 5
    assert db.delete_event_rows(REC_A, label=float("nan")) == 4
    assert [r["label"] for r in db.list_event_rows()] == ["dob"]


def test_pandas_missing_markers_in_params_mean_unspecified(registry):
    pd = pytest.importorskip("pandas")
    params = {"method": pd.NA, "window_s": pd.NA, "ks": 3.0, "t0": pd.NaT}
    for _ in range(2):
        assert db.save_event_rows(REC_A, [_event(42.0)], params=params) == 1
    rows = db.list_event_rows()
    assert len(rows) == 1
    assert rows[0]["method"] is None and rows[0]["window_s"] is None
    assert rows[0]["params"] == {"method": None, "window_s": None, "ks": 3.0, "t0": None}


@pytest.mark.parametrize("flag, expected", [
    (None, True), (True, True), (1, True), ("yes", True), ("1.0", True),
    (False, False), (0, False), ("0", False), ("false", False), (" No ", False),
    (np.bool_(False), False), ("0.0", False), (b"0", False),
    # Blank text is a missing flag (an empty CSV cell), like None / NaN: valid.
    ("", True), ("  ", True), ("nan", True), (float("nan"), True),
])
def test_valid_flag_coercion(registry, flag, expected):
    row = _event(1.0)
    if flag is None:
        row.pop("valid")
    else:
        row["valid"] = flag
    db.save_event_rows(REC_A, [row], params=PARAMS)
    stored = db.list_event_rows(valid_only=False)
    assert len(stored) == 1 and stored[0]["valid"] is expected


def test_missing_valid_flag_gives_the_default():
    pd = pytest.importorskip("pandas")
    # Every spelling of "missing" agrees, whichever default is asked for.
    for missing in (None, "", "  ", " NaN ", b"", float("nan"), np.float64("nan"),
                    pd.NA, pd.NaT):
        assert db._as_bool(missing, default=True) is True, repr(missing)
        assert db._as_bool(missing, default=False) is False, repr(missing)
    assert db._as_bool("0", default=True) is False
    assert db._as_bool("yes", default=False) is True


# ── Upsert identity ───────────────────────────────────────────────────────────

def test_upsert_is_idempotent_and_keeps_the_id(registry):
    db.save_event_rows(REC_A, [_event(612.4)], subject="M1", params=PARAMS)
    first = db.list_event_rows()[0]

    # Same event again: onset recomputed one ulp away, new metrics/subject.
    onset = math.nextafter(612.4, 1e9)
    newer = _event(onset, SD1={"pre": 9.0, "post": 9.5, "delta": 0.5})
    for _ in range(3):
        assert db.save_event_rows(REC_A, [newer], subject="M1-renamed",
                                  params=PARAMS) == 1

    rows = db.list_event_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == first["id"]
    assert rows[0]["subject"] == "M1-renamed"
    assert rows[0]["metrics"] == {"SD1": {"pre": 9.0, "post": 9.5, "delta": 0.5}}


def test_subject_and_project_follow_the_recording_on_every_row(registry):
    # First save with the "All" label filter (replace), Subject ID not filled
    # in yet; a second configuration of the same recording; another recording.
    db.save_event_rows(REC_A, [_event(600.0, "dob"), _event(1800.0, "dob"),
                               _event(3000.0, "saline")],
                       project_name="P0", params=PARAMS, replace=True)
    db.save_event_rows(REC_A, [_event(600.0, "dob")], project_name="P0",
                       params={**PARAMS, "method": "approximate"})
    db.save_event_rows(REC_B, [_event(600.0, "dob")], subject="M2",
                       project_name="P0", params=PARAMS)
    before = {r["id"]: r for r in db.list_event_rows(filepath=REC_A)}

    # The user fills in the Subject ID / project and re-saves with the "dob"
    # label filter, which sends only the dob rows (no replace). The count is
    # of rows written, not of identity updates.
    assert db.save_event_rows(REC_A, [_event(600.0, "dob"), _event(1800.0, "dob")],
                              subject="M1", project_name="P1", params=PARAMS) == 2

    rows = db.list_event_rows(filepath=REC_A)
    assert sorted(r["id"] for r in rows) == sorted(before)
    # One recording, one identity: the saline row and the other configuration
    # follow too, so the cohort sees one subject and the project filter the
    # whole recording.
    assert {(r["subject"], r["project_name"]) for r in rows} == {("M1", "P1")}
    assert len(db.list_event_rows(project_name="P1")) == 4
    untouched = [r for r in rows if r["label"] == "saline" or r["method"] == "approximate"]
    assert len(untouched) == 2
    for r in untouched:                      # only the identity changed on them
        old = before[r["id"]]
        assert (r["saved_at"], r["metrics"], r["params"], r["valid"]) == (
            old["saved_at"], old["metrics"], old["params"], old["valid"])
    # Another recording keeps its own identity.
    assert [(r["subject"], r["project_name"])
            for r in db.list_event_rows(filepath=REC_B)] == [("M2", "P0")]

    # A blank Subject ID clears it on every row of the recording alike.
    db.save_event_rows(REC_A, [_event(3000.0, "saline")], project_name="P1",
                       params=PARAMS)
    assert {(r["subject"], r["project_name"])
            for r in db.list_event_rows(filepath=REC_A, valid_only=False)} == {("", "P1")}


@pytest.mark.parametrize("fs, idx", [
    (3200.0, 1_959_601), (3200.0, 35), (6400.0, 82), (16000.0, 13)])
def test_onset_rederived_across_a_rounding_boundary_is_the_same_event(registry, fs, idx):
    # sample / fs vs sample * (1 / fs): a few ulp apart, but on either side of
    # a half-microsecond, so their 1 µs-rounded keys differ.
    a, b = idx / fs, idx * (1.0 / fs)
    assert db._key_float(a) != db._key_float(b)          # the case under test
    assert db.save_event_rows(REC_A, [_event(a)], params=PARAMS) == 1
    first = db.list_event_rows()[0]

    newer = _event(b, SD1={"pre": 9.0, "post": 9.5, "delta": 0.5})
    assert db.save_event_rows(REC_A, [newer], params=PARAMS) == 1
    rows = db.list_event_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == first["id"]
    assert rows[0]["onset_s"] == first["onset_s"]        # first save's value kept
    assert rows[0]["metrics"] == {"SD1": {"pre": 9.0, "post": 9.5, "delta": 0.5}}

    # Both derivations in one call are one event too, counted once.
    assert db.save_event_rows(REC_B, [_event(a), _event(b)], params=PARAMS) == 1
    assert len(db.list_event_rows(filepath=REC_B)) == 1


def test_window_rederived_across_a_rounding_boundary_is_the_same_configuration(registry):
    w_a, w_b = 48003 / 3200.0, 48003 * (1.0 / 3200.0)    # ~15.0009 s
    assert db._key_float(w_a) != db._key_float(w_b)
    assert db.save_event_rows(REC_A, [_event(10.0)], params={**PARAMS, "window_s": w_a}) == 1
    assert db.save_event_rows(REC_A, [_event(10.0)], params={**PARAMS, "window_s": w_b}) == 1
    rows = db.list_event_rows()
    assert len(rows) == 1 and rows[0]["window_s"] == pytest.approx(15.0009375, abs=1e-6)

    # replace=True with the re-derived window clears that same configuration,
    # and only that one.
    db.save_event_rows(REC_A, [_event(10.0)], params={**PARAMS, "window_s": 30.0})
    assert db.save_event_rows(REC_A, [], params={**PARAMS, "window_s": w_b},
                              replace=True) == 0
    assert [r["window_s"] for r in db.list_event_rows()] == [30.0]


def test_events_one_sample_apart_stay_distinct(registry):
    fs = 50_000.0                                    # 20 µs per sample
    onsets = [612.4, 612.4 + 1.0 / fs, 612.4 + 2.0 / fs]
    assert db.save_event_rows(REC_A, [_event(t) for t in onsets], params=PARAMS) == 3
    ids = [r["id"] for r in db.list_event_rows()]
    assert len(ids) == 3
    # A re-save matches each event to its own row, not to a neighbour.
    resaved = [_event(t, SD1={"pre": float(k), "post": 0.0}) for k, t in enumerate(onsets)]
    assert db.save_event_rows(REC_A, resaved, params=PARAMS) == 3
    rows = db.list_event_rows()
    assert [r["id"] for r in rows] == ids
    assert [r["metrics"]["SD1"]["pre"] for r in rows] == [0.0, 1.0, 2.0]


def test_method_and_window_are_part_of_the_key(registry):
    ev = [_event(612.4)]
    db.save_event_rows(REC_A, ev, params=PARAMS)
    db.save_event_rows(REC_A, ev, params={**PARAMS, "method": "approximate"})
    db.save_event_rows(REC_A, ev, params={**PARAMS, "window_s": 30.0})
    db.save_event_rows(REC_B, ev, params=PARAMS)
    db.save_event_rows(REC_A, ev, params=PARAMS)          # repeat: no new row

    rows = db.list_event_rows()
    assert len(rows) == 4
    assert {(r["filepath"], r["method"], r["window_s"]) for r in rows} == {
        (REC_A, "exact", 15.0), (REC_A, "approximate", 15.0),
        (REC_A, "exact", 30.0), (REC_B, "exact", 15.0)}


@pytest.mark.parametrize("params", [None, {}, {"method": "", "ks": 3.0},
                                    {"method": None, "window_s": float("nan")}])
def test_upsert_is_null_safe_without_method_or_window(registry, params):
    for _ in range(2):
        assert db.save_event_rows(REC_A, [_event(42.0)], params=params) == 1
    rows = db.list_event_rows()
    assert len(rows) == 1
    assert rows[0]["method"] is None and rows[0]["window_s"] is None


def test_duplicate_events_in_one_call_collapse_to_the_last(registry):
    a = _event(5.0, SD1={"pre": 1.0, "post": 2.0})
    b = _event(5.0, SD1={"pre": 3.0, "post": 4.0})
    assert db.save_event_rows(REC_A, [a, b], params=PARAMS) == 1
    assert db.list_event_rows()[0]["metrics"]["SD1"]["pre"] == 3.0


def test_replace_drops_stale_events_of_that_configuration_only(registry):
    db.save_event_rows(REC_A, [_event(10.0), _event(20.0)], params=PARAMS)
    other_cfg = {**PARAMS, "method": "approximate"}
    db.save_event_rows(REC_A, [_event(10.0)], params=other_cfg)
    db.save_event_rows(REC_B, [_event(10.0)], params=PARAMS)

    assert db.save_event_rows(REC_A, [_event(20.0)], params=PARAMS, replace=True) == 1
    left = {(r["filepath"], r["method"], r["onset_s"]) for r in db.list_event_rows()}
    assert left == {(REC_A, "exact", 20.0), (REC_A, "approximate", 10.0),
                    (REC_B, "exact", 10.0)}

    # An empty list with replace=True clears that configuration.
    assert db.save_event_rows(REC_A, [], params=PARAMS, replace=True) == 0
    left = {(r["filepath"], r["method"]) for r in db.list_event_rows()}
    assert left == {(REC_A, "approximate"), (REC_B, "exact")}


def test_replace_never_deletes_when_supplied_rows_are_rejected(registry):
    db.save_event_rows(REC_A, [_event(10.0), _event(20.0)], params=PARAMS)
    ids = [r["id"] for r in db.list_event_rows()]

    # Every supplied row rejected: the stored events stay.
    junk = [_event("abc"), _event(None), _event(float("nan")), "junk"]
    assert db.save_event_rows(REC_A, junk, params=PARAMS, replace=True) == 0
    assert [r["id"] for r in db.list_event_rows()] == ids

    # Some rejected: the valid rows are upserted, the delete is skipped.
    newer = _event(20.0, SD1={"pre": 9.0, "post": 9.5})
    assert db.save_event_rows(REC_A, [newer, {**_event(30.0), "pre_s": "x"}],
                              params=PARAMS, replace=True) == 1
    rows = db.list_event_rows()
    assert [r["id"] for r in rows] == ids
    assert rows[1]["metrics"] == {"SD1": {"pre": 9.0, "post": 9.5, "delta": 0.5}}

    # rows=None or a non-iterable is not an explicitly empty list.
    assert db.save_event_rows(REC_A, None, params=PARAMS, replace=True) == 0
    assert db.save_event_rows(REC_A, 42, params=PARAMS, replace=True) == 0
    assert [r["id"] for r in db.list_event_rows()] == ids

    # A clean list (here with a text onset) does replace.
    assert db.save_event_rows(REC_A, [_event("20.0")], params=PARAMS, replace=True) == 1
    assert [(r["id"], r["onset_s"]) for r in db.list_event_rows()] == [(ids[1], 20.0)]


# ── Filters, labels, delete ───────────────────────────────────────────────────

def _populate():
    db.save_event_rows(REC_A, [_event(10.0, "dobutamine"), _event(20.0, "Saline"),
                               _event(30.0, "", valid=False)],
                       subject="A", project_name="P1", params=PARAMS)
    db.save_event_rows(REC_B, [_event(10.0, "dobutamine", valid=False),
                               _event(50.0, "atropine"), _event(60.0, "")],
                       subject="B", project_name="P2", params=PARAMS)


def test_list_filters(registry):
    _populate()
    assert len(db.list_event_rows()) == 4
    assert len(db.list_event_rows(valid_only=False)) == 6

    dob = db.list_event_rows(label="dobutamine")
    assert [r["filepath"] for r in dob] == [REC_A]
    dob_all = db.list_event_rows(label=" dobutamine ", valid_only=False)
    assert [r["filepath"] for r in dob_all] == [REC_A, REC_B]

    assert {r["label"] for r in db.list_event_rows(project_name="P2")} == {"atropine", ""}
    assert db.list_event_rows(label="dobutamine", project_name="P2") == []
    assert [r["onset_s"] for r in db.list_event_rows(label="")] == [60.0]
    assert len(db.list_event_rows(label="", valid_only=False)) == 2
    assert {r["label"] for r in db.list_event_rows(filepath=REC_A)} == {"dobutamine", "Saline"}
    assert db.list_event_rows(label="unknown") == []


def test_list_event_labels(registry):
    assert db.list_event_labels() == []
    _populate()
    db.save_event_rows(REC_B, [_event(70.0, "Atropine-2", valid=False)], params=PARAMS)
    # Distinct, case-insensitive order, '' excluded, all-invalid labels kept.
    assert db.list_event_labels() == ["atropine", "Atropine-2", "dobutamine", "Saline"]


def test_delete_event_rows(registry):
    _populate()
    db.upsert_recording(REC_A, "fp", "a.ecgsession", stats={"n_peaks": 10})

    assert db.delete_event_rows(REC_A, label="dobutamine") == 1
    assert db.delete_event_rows(REC_A, label="dobutamine") == 0
    assert {r["label"] for r in db.list_event_rows(filepath=REC_A, valid_only=False)} == {"Saline", ""}

    assert db.delete_event_rows(REC_A) == 2
    assert db.list_event_rows(filepath=REC_A, valid_only=False) == []
    assert len(db.list_event_rows(valid_only=False)) == 3           # REC_B untouched
    assert db.get_recording(REC_A)["n_peaks"] == 10                 # recordings untouched
    assert db.delete_event_rows("C:/never/saved.mat") == 0


# ── CSV export ────────────────────────────────────────────────────────────────

def _read_csv(path, delimiter=","):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh, delimiter=delimiter))


def test_export_csv_flattens_metrics_and_params(registry, tmp_path):
    db.save_event_rows(REC_A, [_event(10.0)], subject="Souris é", params=PARAMS)
    db.save_event_rows(REC_B, [_event(20.0, HR={"pre": 500.0, "post": None,
                                                 "n_beats": 30})],
                       params=PARAMS)
    rows = sorted(db.list_event_rows(), key=lambda r: r["filepath"])   # A, then B
    out = tmp_path / "events.csv"
    assert db.export_event_rows_csv(out, rows) == 2

    table = _read_csv(out)
    header, body = table[0], table[1:]
    assert header[:13] == list(db._EVENT_CSV_COLUMNS)
    metric_cols = [c for c in header[13:] if not c.startswith("param_")]
    assert metric_cols == [
        "D_pre", "D_post", "D_delta", "SD1_pre", "SD1_post", "SD1_delta",
        "SD2_pre", "SD2_post", "SD2_delta", "rCSI_pre", "rCSI_post", "rCSI_delta",
        "HR_pre", "HR_post", "HR_delta", "HR_n_beats"]
    param_cols = [c for c in header if c.startswith("param_")]
    assert param_cols == ["param_ks", "param_kp", "param_lf_band",
                          "param_hf_band", "param_rr_bounds_ms"]
    assert "params_json" not in header and "metrics_json" not in header

    rec = [dict(zip(header, line)) for line in body]
    a, b = rec
    assert a["subject"] == "Souris é" and a["valid"] == "1"
    assert a["method"] == "exact" and float(a["window_s"]) == 15.0
    assert float(a["SD2_delta"]) == 2.0 and float(a["rCSI_delta"]) == 1.0
    assert a["HR_pre"] == "" and a["HR_n_beats"] == ""       # metric absent in row A
    assert b["D_pre"] == ""                                   # metric absent in row B
    assert float(b["HR_pre"]) == 500.0 and b["HR_post"] == "" and b["HR_delta"] == ""
    assert b["HR_n_beats"] == "30"
    assert float(a["param_ks"]) == 3.0 and a["param_lf_band"] == "[0.4, 1.5]"


def test_export_csv_accepts_raw_json_rows_numpy_and_semicolons(tmp_path):
    rows = [
        {"id": 1, "filepath": REC_A, "label": "dob", "onset_s": np.float64(0.1),
         "valid": np.bool_(False), "window_s": float("nan"),
         "metrics_json": '{"SD1": {"pre": 1.5, "post": 2.5, "delta": 1.0}}',
         "params_json": '{"ks": 3}'},
        "not a row",
    ]
    out = tmp_path / "events_semicolon.csv"
    assert db.export_event_rows_csv(out, rows, delimiter=";") == 1
    header, line = _read_csv(out, delimiter=";")
    rec = dict(zip(header, line))
    assert rec["onset_s"] == "0.1"            # not "np.float64(0.1)"
    assert rec["valid"] == "0"
    assert rec["window_s"] == ""
    assert rec["SD1_delta"] == "1.0" and rec["param_ks"] == "3"


def test_export_csv_of_no_rows_writes_the_header_only(tmp_path):
    out = tmp_path / "empty.csv"
    assert db.export_event_rows_csv(out, []) == 0
    assert _read_csv(out) == [list(db._EVENT_CSV_COLUMNS)]


# ── Pre-migration registry upgrade ────────────────────────────────────────────

# The recordings-only schemas of earlier builds: the original table, and the
# last single-table schema (before event_locked existed).
_OLD_SCHEMAS = {
    "original": """
        CREATE TABLE recordings (
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
        CREATE INDEX idx_fp ON recordings(filepath);
        CREATE INDEX idx_saved ON recordings(saved_at DESC);
    """,
    "single_table": """
        CREATE TABLE recordings (
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
        CREATE INDEX idx_fp ON recordings(filepath);
        CREATE INDEX idx_saved ON recordings(saved_at DESC);
    """,
}


def _make_old_registry(db_path, schema):
    """Write a registry of an earlier build (no event_locked) with two recordings."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(_OLD_SCHEMAS[schema])
        con.executemany(
            "INSERT INTO recordings (filepath, stem, fingerprint, saved_at, "
            "duration_s, n_peaks, hr_mean, sdnn, rmssd, notes, session_path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(REC_A, "mouse_A", "abc123", "2026-01-02T10:00:00", 3600.0, 30000,
              520.0, 8.5, 4.2, "baseline then dobutamine", "C:/s/a.ecgsession"),
             (REC_B, "mouse_B", "def456", "2026-01-03T10:00:00", 1800.0, 15000,
              540.0, 7.0, 3.1, "", "C:/s/b.ecgsession")])
        con.commit()
    finally:
        con.close()


@contextlib.contextmanager
def _read_only(path):
    """Mark *path* read-only for the block (skips where that cannot bite, e.g. root)."""
    os.chmod(path, stat.S_IREAD)
    try:
        if os.access(path, os.W_OK):
            pytest.skip("cannot make the registry file read-only here")
        yield
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("schema", sorted(_OLD_SCHEMAS))
def test_pre_migration_registry_is_upgraded_in_place(registry, schema):
    _make_old_registry(registry, schema)

    # First use by the current build: the table appears, nothing is lost.
    assert db.list_event_labels() == []
    assert db.save_event_rows(REC_A, [_event(612.4)], subject="A",
                              project_name="P1", params=PARAMS) == 1

    a = db.get_recording(REC_A)
    assert (a["stem"], a["fingerprint"], a["n_peaks"], a["hr_mean"], a["notes"],
            a["session_path"]) == ("mouse_A", "abc123", 30000, 520.0,
                                   "baseline then dobutamine", "C:/s/a.ecgsession")
    assert a["verified_for_training"] == 0 and a["project_name"] == ""
    assert a["rr_sparkline"] == ""
    assert [r["filepath"] for r in db.recent_recordings()] == [REC_B, REC_A]
    assert db.list_event_labels() == ["dobutamine"]
    assert db.list_event_rows()[0]["metrics"]["SD2"]["delta"] == 2.0

    # A second process opening the upgraded file applies the schema again
    # without error or duplication.
    db._schema_ready = False
    assert len(db.list_event_rows()) == 1
    assert len(db.recent_recordings()) == 2
    con = sqlite3.connect(str(registry))
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('recordings', 'event_locked') ORDER BY name")]
    finally:
        con.close()
    assert tables == ["event_locked", "recordings"]


@pytest.mark.parametrize("schema", sorted(_OLD_SCHEMAS))
def test_read_only_pre_migration_registry_keeps_recordings_readable(registry, schema):
    # The event table cannot be created in a read-only file; that must not
    # take the recordings API down with it (it kept working before the table
    # existed), and the event functions must fail gracefully.
    _make_old_registry(registry, schema)
    with _read_only(registry):
        assert [r["filepath"] for r in db.recent_recordings()] == [REC_B, REC_A]
        assert db.get_notes(REC_A) == "baseline then dobutamine"
        assert db.get_recording(REC_B)["n_peaks"] == 15000
        assert [r["stem"] for r in db.search_recordings("dobutamine")] == ["mouse_A"]

        assert db.list_event_rows() == []
        assert db.list_event_labels() == []
        assert db.save_event_rows(REC_A, [_event(612.4)], params=PARAMS) == 0
        assert db.delete_event_rows(REC_A) == 0
        assert db._schema_ready is False             # the upgrade is still pending

    # Writable again, same process: the next call upgrades the file in place.
    assert db.save_event_rows(REC_A, [_event(612.4)], params=PARAMS) == 1
    assert db._schema_ready is True
    assert db.list_event_labels() == ["dobutamine"]
    assert db.get_notes(REC_A) == "baseline then dobutamine"
    assert len(db.recent_recordings()) == 2


def test_read_only_upgraded_registry_still_lists_events(registry):
    _make_old_registry(registry, "single_table")
    assert db.save_event_rows(REC_A, [_event(612.4)], subject="A", params=PARAMS) == 1

    db._schema_ready = False                         # a new process opens it read-only
    with _read_only(registry):
        rows = db.list_event_rows()
        assert [(r["filepath"], r["subject"]) for r in rows] == [(REC_A, "A")]
        assert db.list_event_labels() == ["dobutamine"]
        assert len(db.recent_recordings()) == 2
        assert db.save_event_rows(REC_A, [_event(612.4, label="x")], params=PARAMS) == 0
        assert db.delete_event_rows(REC_A) == 0
        assert [r["id"] for r in db.list_event_rows()] == [rows[0]["id"]]


@contextlib.contextmanager
def _write_locked(path):
    """Hold a RESERVED (write) lock on *path* from another connection, as a
    second app instance mid-save or DB Browser with pending edits does."""
    other = sqlite3.connect(str(path), timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
        yield
    finally:
        other.execute("ROLLBACK")
        other.close()


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


@pytest.mark.parametrize("schema", sorted(_OLD_SCHEMAS))
def test_write_locked_pre_migration_registry_does_not_stall_each_call(
        registry, schema, monkeypatch):
    # The upgrade writes used to wait the full 5 s busy timeout for the other
    # connection's lock, on every call (the upgrade stays pending), read-only
    # recordings calls included.
    _make_old_registry(registry, schema)
    attempts = []
    real_upgrade = db._upgrade_schema

    def counting_upgrade(con):
        attempts.append(1)
        return real_upgrade(con)

    monkeypatch.setattr(db, "_upgrade_schema", counting_upgrade)
    budget = db._BUSY_TIMEOUT_S / 2          # far less than one full busy wait

    with _write_locked(registry):
        out, dt = _timed(db.recent_recordings)
        assert [r["filepath"] for r in out] == [REC_B, REC_A] and dt < budget
        assert db._schema_ready is False and len(attempts) == 1

        # Recordings-only calls back off: no new attempt, so no wait at all.
        for fn, args in ((db.recent_recordings, ()), (db.get_notes, (REC_A,)),
                         (db.search_recordings, ("dobutamine",))):
            out, dt = _timed(fn, *args)
            assert out and dt < budget
        assert len(attempts) == 1

        # Event calls need the table: they retry every time, briefly, and
        # fail gracefully.
        out, dt = _timed(db.list_event_rows)
        assert out == [] and dt < budget
        out, dt = _timed(db.save_event_rows, REC_A, [_event(1.0)], params=PARAMS)
        assert out == 0 and dt < budget
        assert len(attempts) == 3

        # Once the back-off has run out, a recordings call retries again.
        db._schema_retry_at = 0.0
        assert len(db.recent_recordings()) == 2
        assert len(attempts) == 4 and db._schema_ready is False

    # Lock released: the next event call completes the upgrade in place.
    assert db.save_event_rows(REC_A, [_event(612.4)], params=PARAMS) == 1
    assert db._schema_ready is True and len(attempts) == 5
    a = db.get_recording(REC_A)
    assert a["notes"] == "baseline then dobutamine" and a["verified_for_training"] == 0
    assert len(db.recent_recordings()) == 2
    assert len(attempts) == 5                        # up to date: not re-applied


@pytest.mark.parametrize("schema", sorted(_OLD_SCHEMAS))
def test_write_locked_upgrade_waits_for_the_lock_once_per_attempt(
        registry, schema, monkeypatch):
    # SQLite's busy timeout is per statement: trying every upgrade write on a
    # locked file waited _UPGRADE_BUSY_MS for each of them (about 1.2 s per
    # event call for the original schema's three ALTER TABLEs plus the
    # event_locked creation). An attempt now stops at its first failed write.
    _make_old_registry(registry, schema)
    writes = []
    real_add, real_events = db._add_recordings_column, db._apply_event_schema

    def add_column(con, ddl):
        ok = real_add(con, ddl)
        writes.append(("alter", ok))
        return ok

    def event_schema(con):
        ok = real_events(con)
        writes.append(("event_locked", ok))
        return ok

    monkeypatch.setattr(db, "_add_recordings_column", add_column)
    monkeypatch.setattr(db, "_apply_event_schema", event_schema)

    with _write_locked(registry):
        for fn in (db.list_event_labels, db.list_event_rows, db.list_event_labels):
            writes.clear()
            assert fn() == []
            failed = [w for w in writes if not w[1]]
            assert len(failed) == 1, writes          # one busy wait per call ...
            assert writes[-1] == failed[0], writes   # ... nothing tried after it
        assert db._schema_ready is False

    # Lock released: the whole upgrade completes on the next call.
    writes.clear()
    assert db.save_event_rows(REC_A, [_event(612.4)], params=PARAMS) == 1
    assert db._schema_ready is True
    assert writes == [("alter", True)] * len(db._RECORDINGS_UPGRADE_COLUMNS) + [
        ("event_locked", True)]
    a = db.get_recording(REC_A)
    assert a["notes"] == "baseline then dobutamine" and a["rr_sparkline"] == ""


def test_upgrade_restores_the_normal_busy_timeout(registry):
    _make_old_registry(registry, "single_table")
    normal_ms = db._BUSY_TIMEOUT_S * 1000
    with _write_locked(registry):                    # upgrade fails
        con = db._conn(events=True)
        try:
            assert db._schema_ready is False
            assert con.execute("PRAGMA busy_timeout").fetchone()[0] == normal_ms
        finally:
            con.close()
    con = db._conn(events=True)                      # upgrade succeeds
    try:
        assert db._schema_ready is True
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == normal_ms
    finally:
        con.close()


# ── Graceful failure ──────────────────────────────────────────────────────────

def test_sqlite_unavailable_returns_gracefully(registry, monkeypatch, tmp_path):
    monkeypatch.setattr(db, "_DB_AVAILABLE", False)
    assert db.save_event_rows(REC_A, [_event(1.0)], params=PARAMS) == 0
    assert db.list_event_rows() == []
    assert db.list_event_labels() == []
    assert db.delete_event_rows(REC_A) == 0
    assert not registry.parent.exists()          # nothing created
    # The CSV export does not need sqlite.
    assert db.export_event_rows_csv(tmp_path / "x.csv", [{"id": 1}]) == 1


def test_unreadable_registry_file_returns_gracefully_and_is_released(registry):
    registry.parent.mkdir(parents=True)
    registry.write_bytes(b"this is not an sqlite database" * 100)

    assert db.save_event_rows(REC_A, [_event(1.0)], params=PARAMS) == 0
    assert db.list_event_rows() == []
    assert db.list_event_labels() == []
    assert db.delete_event_rows(REC_A) == 0
    assert db._schema_ready is False
    # No connection handle leaked on the failed schema step: on Windows an
    # open handle would make this unlink fail.
    registry.unlink()
    assert db.save_event_rows(REC_A, [_event(1.0)], params=PARAMS) == 1


def test_export_csv_raises_when_the_file_cannot_be_written(tmp_path):
    with pytest.raises(OSError):
        db.export_event_rows_csv(tmp_path / "missing_dir" / "x.csv", [])
