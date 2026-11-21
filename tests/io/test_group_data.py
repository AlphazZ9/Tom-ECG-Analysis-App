"""The group table as a whole experiment (ecg.io.group_data): one workbook,
one animal per recording; adding a recording again replaces its lines in the
sheets it writes and keeps its group; the time course is kept as 5 s sums
from which any coarser bin is exact."""
from __future__ import annotations

import math
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from ecg.io import group_data as gd
from ecg.io.group_table import append_group_row, group_row


def _rr(n=2000, rr=120.0, seed=0, t0=0.0):
    rng = np.random.default_rng(seed)
    r = rr + rng.normal(0, 3, n)
    t = t0 + np.cumsum(r) / 1000.0
    return t, r


def test_time_frame_sums_give_the_exact_metrics_of_any_coarser_bin():
    t, r = _rr()
    valid = np.ones(len(r), bool)
    valid[[10, 11, 500]] = False
    tc = gd.time_frame(t, r, valid)
    assert list(tc.columns) == list(gd.TC_COLUMNS)
    assert int(tc["n_rr"].sum()) == int(valid.sum())
    # One 60 s window aligned on the 5 s grid: the metrics of the beats in it.
    lo, hi = 60.0, 120.0
    m = valid & (np.floor(t / 5) * 5 >= lo) & (np.floor(t / 5) * 5 < hi)
    assert gd.window_value(tc, "RR", lo, hi) == pytest.approx(r[m].mean())
    assert gd.window_value(tc, "HR", lo, hi) == pytest.approx((60000 / r[m]).mean())
    assert gd.window_value(tc, "SDNN", lo, hi) == pytest.approx(r[m].std(ddof=1))
    pair = valid[1:] & valid[:-1]
    idx = np.floor(t[1:] / 5) * 5
    d = np.diff(r)[pair & (idx >= lo) & (idx < hi)]
    assert gd.window_value(tc, "RMSSD", lo, hi) == pytest.approx(math.sqrt(np.mean(d ** 2)))


def test_a_window_with_too_few_valid_beats_is_empty():
    t, r = _rr(n=500)
    valid = np.ones(len(r), bool)
    valid[(t > 20) & (t < 50)] = False
    tc = gd.time_frame(t, r, valid)
    assert math.isnan(gd.window_value(tc, "HR", 20, 50))
    assert math.isfinite(gd.window_value(tc, "HR", 0, 20))


def test_binned_series_is_relative_to_the_time_0():
    t, r = _rr(n=3000)
    tc = gd.time_frame(t, r, np.ones(len(r), bool))
    starts, v = gd.binned_series(tc, "HR", t0_s=120.0, start_s=-60, end_s=120, bin_s=30)
    assert list(starts) == [-60, -30, 0, 30, 60, 90]
    assert v[0] == pytest.approx(gd.window_value(tc.assign(t_s=tc["t_s"] - 120), "HR", -60, -30))


def _lines(sex="F", seed=0, label="Stim"):
    t, r = _rr(seed=seed)
    return {
        gd.SHEET_WHOLE: gd.whole_frame([NS(label="Mean HR (60000/mean NN)", unit="bpm",
                                           value=500.0 + seed),
                                        NS(label="SDNN", unit="ms", value=float("nan"))]),
        gd.SHEET_TIME: gd.time_frame(t, r, np.ones(len(r), bool)),
        gd.SHEET_ANNOT: gd.annotations_frame([{"t_start": 60.0, "t_end": 61.0, "label": label}]),
        gd.SHEET_EVENTS: gd.events_frame([{"label": label, "onset_s": 60.0, "valid": True,
                                           "metrics": {"rCSI": {"pre": 1.0, "post": 2.0,
                                                                "delta": 1.0}}}]),
    }


def test_add_recording_creates_replaces_and_keeps_the_group(tmp_path):
    p = str(tmp_path / "g.xlsx")
    n, replaced = gd.add_recording(p, "1146", "F", "/data/a 1146.smr", _lines())
    assert (n, replaced) == (1, False)
    gd.add_recording(p, "206", "m", "b 206.smr", _lines(seed=1))
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    assert list(an["subject"]) == ["1146", "206"] and list(an["sex"]) == ["F", "M"]
    assert list(an["file"]) == ["a 1146.smr", "b 206.smr"]
    assert gd.data_text(an.iloc[0]) == "whole · time · 1 event"
    assert "SDNN (ms)" not in sh[gd.SHEET_WHOLE].columns          # NaN: no column
    an.loc[0, "group"] = "Control"
    an.loc[1, "use"] = False
    gd.save_animals(p, an)
    # Added again, with no events this time: replaced, its group kept, its
    # events gone, the other animal untouched.
    lines = _lines(seed=2)
    lines[gd.SHEET_EVENTS] = None
    n, replaced = gd.add_recording(p, "1146", "F", "/other/dir/a 1146.smr", lines)
    assert (n, replaced) == (2, True)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    assert an.set_index("subject").loc["1146", "group"] == "Control"
    assert not an.set_index("subject").loc["206", "use"]
    assert list(sh[gd.SHEET_EVENTS]["subject"]) == ["206"]
    assert sh[gd.SHEET_WHOLE].set_index("subject").loc["1146", "Mean HR (60000/mean NN) (bpm)"] == 502
    assert len(sh[gd.SHEET_TIME].query("subject == '206'")) > 0


def test_compare_segments_rows_keep_the_other_sheets(tmp_path):
    """Add to group table from Compare segments (append_group_row) rewrites
    the Group sheet only: the Group tab's sheets stay."""
    p = str(tmp_path / "g.xlsx")
    gd.add_recording(p, "1146", "F", "a.smr", _lines())
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0},
            {"label": "Post", "t_lo": 60.0, "t_hi": 120.0, "hr_mean": 520.0}]
    append_group_row(p, group_row("1146", "a.smr", segs, [("hr_mean", "Mean HR", "bpm")]))
    sh = gd.read_sheets(p)
    assert set(sh) >= {gd.SHEET_ANIMALS, gd.SHEET_SEGMENTS, gd.SHEET_WHOLE, gd.SHEET_TIME,
                       gd.SHEET_EVENTS}
    assert gd.animals_frame(sh).iloc[0]["has_segments"]


def test_segments_sheet_matches_the_tables_tags(tmp_path):
    p = str(tmp_path / "g.xlsx")
    for sid, lab in (("1", "Pre Stim"), ("2", "pre  stim")):
        segs = [{"label": lab, "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0},
                {"label": "Post", "t_lo": 60.0, "t_hi": 120.0, "hr_mean": 520.0}]
        row = group_row(sid, f"{sid}.smr", segs, [("hr_mean", "Mean HR", "bpm")])
        for k in ("subject", "file"):
            row.pop(k)
        gd.add_recording(p, sid, "", f"{sid}.smr", {gd.SHEET_SEGMENTS: pd.DataFrame([row])})
    seg = gd.read_sheets(p)[gd.SHEET_SEGMENTS]
    assert seg["Mean HR (bpm) Pre Stim"].notna().all()


def test_an_older_table_with_only_segments_opens(tmp_path):
    p = str(tmp_path / "old.xlsx")
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0}]
    append_group_row(p, group_row("7", "7.smr", segs, [("hr_mean", "Mean HR", "bpm")], sex="M"))
    an = gd.animals_frame(gd.read_sheets(p))
    assert list(an["subject"]) == ["7"] and an.iloc[0]["sex"] == "M" and an.iloc[0]["use"]


def test_not_a_group_table_is_refused(tmp_path):
    p = tmp_path / "x.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(p, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError):
        gd.read_sheets(str(p))


def test_onsets_and_labels():
    annot = pd.DataFrame({"subject": ["1", "1", "2"], "file": ["a", "a", "b"],
                          "label": ["Stim", "stim ", "Other"], "t_start": [30.0, 10.0, 5.0],
                          "t_end": [np.nan] * 3})
    on = gd.stim_onsets(annot)
    assert gd.onset_of(on[("1", "a")], "STIM") == 10.0
    assert gd.onset_of(on[("1", "a")], "Stim", 1) == 30.0
    assert gd.onset_of(on[("2", "b")], "Stim") is None
    assert gd.annotation_labels(annot)[0] in ("Stim", "stim")


def test_the_users_own_sheets_are_kept(tmp_path):
    """Audit: a sheet the user added (notes...) survived Compare segments but
    not the Group tab's saves."""
    p = str(tmp_path / "g.xlsx")
    gd.add_recording(p, "1", "F", "a.smr", _lines())
    sh = gd.read_sheets(p)
    sh["My notes"] = pd.DataFrame({"note": ["dose 2 mg/kg"]})
    gd.write_sheets(p, sh)
    gd.save_animals(p, gd.animals_frame(gd.read_sheets(p)))
    gd.add_recording(p, "2", "M", "b.smr", _lines(seed=1))
    notes = pd.read_excel(p, sheet_name="My notes")
    assert notes["note"].tolist() == ["dose 2 mg/kg"]


def test_a_sex_set_in_the_tab_stays_when_none_is_given(tmp_path):
    p = str(tmp_path / "g.xlsx")
    gd.add_recording(p, "1", "", "a.smr", _lines())
    an = gd.animals_frame(gd.read_sheets(p))
    an.loc[0, "sex"] = "F"
    gd.save_animals(p, an)
    gd.add_recording(p, "1", "", "a.smr", _lines())
    assert gd.animals_frame(gd.read_sheets(p)).iloc[0]["sex"] == "F"
    gd.add_recording(p, "1", "M", "a.smr", _lines())           # a sex given wins
    assert gd.animals_frame(gd.read_sheets(p)).iloc[0]["sex"] == "M"


def test_a_sheet_past_excels_limit_is_refused_clearly(tmp_path, monkeypatch):
    p = str(tmp_path / "g.xlsx")
    gd.add_recording(p, "1", "F", "a.smr", _lines())
    before = open(p, "rb").read()
    monkeypatch.setattr(gd, "EXCEL_MAX_ROWS", 50)
    with pytest.raises(ValueError, match="TimeCourse.*Excel"):
        gd.add_recording(p, "2", "F", "b.smr", _lines(seed=1))
    assert open(p, "rb").read() == before                      # nothing written


def test_a_numeric_subject_read_back_as_float_replaces_its_row(tmp_path):
    """A subject column with an empty cell is read back as float (243.0):
    Add to group table must still replace the row of subject 243."""
    p = str(tmp_path / "g.xlsx")
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0}]
    metrics = [("hr_mean", "Mean HR", "bpm")]
    append_group_row(p, group_row("243", "a.smr", segs, metrics))
    append_group_row(p, group_row("", "b.smr", segs, metrics))       # an empty subject
    n, replaced = append_group_row(p, group_row("243", "a.smr", segs, metrics))
    assert (n, replaced) == (2, True)
