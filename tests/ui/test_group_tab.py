"""Group tab: opens a group table, lists the animals (group and use saved in
the table), runs the chosen analysis at once and enables its exports; Add
open recording writes the recording open (whole, time course, annotations,
events, segments) to the table."""
from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace as NS

import numpy as np
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.io import db
from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.ui import tab_group as tg
from ecg.ui.tab_group import TabGroupMixin, recording_lines
from tests.io.test_group_compare import _table


@pytest.fixture
def root(monkeypatch, tmp_path):
    monkeypatch.setattr(tg, "PREFS_PATH", tmp_path / "group_prefs.json")
    errors: "list" = []
    monkeypatch.setattr(tg.messagebox, "showerror", lambda *a, **k: errors.append(a))
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    r.errors = errors
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _results_text(frame) -> str:
    return next(w for w in _walk(frame) if isinstance(w, tk.Text)).get("1.0", "end")


def _button(frame, text):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
                and w.cget("text") == text)


def _page(root, state):
    app = NS(_bind_hover_tip=lambda *a, **k: None, _grouptab_state=state)
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    TabGroupMixin._build_group_page(app, frame)
    frame.update()
    return app, frame


@pytest.mark.parametrize("analysis, expect", [
    (gc.AN_TIME, "RM ANOVA Group × time"),
    (gc.AN_SEGMENTS, "Δ Mean HR (bpm) (Post − Pre)"),
    (gc.AN_WHOLE, "Mean HR (60000/mean NN) (bpm)"),
    (gc.AN_EVENTS, "Δ rCSI (post − pre)"),
])
def test_each_analysis_runs_on_open(root, tmp_path, analysis, expect):
    p = _table(tmp_path)
    _app, frame = _page(root, {"path": p, "analysis": analysis, "by": gc.BY_GROUP,
                               "stim": "Stim", "from": "-4", "to": "8", "w0": "0", "w1": "2"})
    text = _results_text(frame)
    assert expect in text and "Control" in text and "CNO" in text, text
    assert _button(frame, "Figure…").cget("state") == "normal"
    assert not root.errors


def test_group_and_use_are_saved_in_the_table(root, tmp_path):
    p = _table(tmp_path, n=4)
    app, frame = _page(root, {"path": p, "analysis": gc.AN_WHOLE, "by": gc.BY_GROUP})
    combos = [w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
              and w.cget("values") and "Control" in w.cget("values")]
    combos[0].set("Vehicle")
    combos[0]._command("Vehicle")
    checks = [w for w in _walk(frame) if isinstance(w, ctk.CTkCheckBox) and w.cget("text") == ""]
    checks[1].deselect()
    checks[1]._command()
    an = gd.animals_frame(gd.read_sheets(p))
    assert an.iloc[0]["group"] == "Vehicle" and not an.iloc[1]["use"]
    assert "Vehicle" in _results_text(frame)


def test_no_table_no_result(root, tmp_path):
    _app, frame = _page(root, {"path": str(tmp_path / "missing.xlsx")})
    assert _button(frame, "Figure…").cget("state") == "disabled"


def test_a_time_course_without_annotation_says_why(root, tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    del sh[gd.SHEET_ANNOT]
    gd.write_sheets(p, sh)
    _app, frame = _page(root, {"path": p, "analysis": gc.AN_TIME})
    assert "annotation" in _results_text(frame)


def _open_recording(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rr = 120 + rng.normal(0, 2, 3000)
    t = np.cumsum(rr) / 1000
    results = {"rr_ms": rr, "rr_t_s": t, "rr_valid": np.ones(len(rr), bool), "hr": {}}
    rows = [NS(label="Mean HR (60000/mean NN)", unit="bpm", value=500.0)]
    monkeypatch.setattr(db, "list_event_rows", lambda **k: [
        {"label": "Stim", "onset_s": 100.0, "valid": True,
         "metrics": {"rCSI": {"pre": 1.0, "post": 2.0, "delta": 1.0}}}])
    app = NS(signal=NS(filepath=str(tmp_path / "Stim BNST - 1146 F.smr")),
             ent_subject=NS(get=lambda: "1146"), ent_sex=NS(get=lambda: "f"),
             analysis=NS(results=results, annotations=[{"t_start": 100.0, "label": "Stim"}]),
             export_ctrl=NS(metric_table=lambda: NS(rows=rows)),
             _current_comparison=lambda: None)
    app._group_lines_of_open_recording = (
        lambda comparison=None: TabGroupMixin._group_lines_of_open_recording(app, comparison))
    return app


def test_add_open_recording(tmp_path, monkeypatch):
    app = _open_recording(tmp_path, monkeypatch)
    p = str(tmp_path / "g.xlsx")
    n, replaced, what = TabGroupMixin._add_open_recording_to_group_table(app, p)
    assert (n, replaced) == (1, False)
    assert what == "1146: whole recording, time course, events, annotations"
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    assert an.iloc[0]["sex"] == "F" and an.iloc[0]["file"] == "Stim BNST - 1146 F.smr"
    assert sh[gd.SHEET_WHOLE].iloc[0]["Mean HR (60000/mean NN) (bpm)"] == 500.0
    assert sh[gd.SHEET_TIME]["n_rr"].sum() == 3000
    # Nothing computed: refused, nothing written.
    app.analysis.results = None
    app.analysis.annotations = []
    monkeypatch.setattr(db, "list_event_rows", lambda **k: [])
    with pytest.raises(ValueError):
        TabGroupMixin._add_open_recording_to_group_table(app, str(tmp_path / "h.xlsx"))
    assert not (tmp_path / "h.xlsx").exists()


def test_recording_lines_with_a_comparison():
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0},
            {"label": "Post", "t_lo": 60.0, "t_hi": 120.0, "hr_mean": 520.0}]
    cmp_ = NS(subject="7", filepath="/x/7.smr", segs=segs, metrics=[("hr_mean", "Mean HR", "bpm")],
              template="", flags=[[], []])
    lines = recording_lines(None, [], [], [], cmp_, "M")
    seg = lines[gd.SHEET_SEGMENTS]
    assert "subject" not in seg.columns and seg.iloc[0]["Mean HR (bpm) Post"] == 520.0
    # Not computed this time: left out, so an earlier add's lines stay.
    assert gd.SHEET_TIME not in lines and gd.SHEET_WHOLE not in lines


def test_the_last_table_is_remembered(root, tmp_path):
    p = _table(tmp_path, n=4)
    _page(root, {"path": p})
    assert tg.load_last_table() == p
    app = NS(_bind_hover_tip=lambda *a, **k: None)          # a new session
    frame = ctk.CTkFrame(root)
    TabGroupMixin._build_group_page(app, frame)
    assert app._grouptab_state["path"] == p


def test_the_comparison_subject_keys_the_row(tmp_path, monkeypatch):
    """Audit: Add to group table keyed the row on the Subject entry, not on the
    subject the comparison ran under: the same file came in twice."""
    app = _open_recording(tmp_path, monkeypatch)
    app.ent_subject = NS(get=lambda: "")
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0}]
    cmp_ = NS(subject="243", filepath=app.signal.filepath, segs=segs,
              metrics=[("hr_mean", "Mean HR", "bpm")], template="", flags=[[]])
    p = str(tmp_path / "g.xlsx")
    TabGroupMixin._add_open_recording_to_group_table(app, p, comparison=cmp_)
    assert list(gd.animals_frame(gd.read_sheets(p))["subject"]) == ["243"]


def test_add_open_asks_where_when_the_remembered_table_did_not_open(root, tmp_path, monkeypatch):
    """Audit: a remembered table that no longer opens is not written to silently."""
    gone = tmp_path / "moved.xlsx"
    asked = []
    monkeypatch.setattr(tg.filedialog, "asksaveasfilename",
                        lambda *a, **k: asked.append(k) or "")
    app, frame = _page(root, {"path": str(gone)})
    app._add_open_recording_to_group_table = lambda *a, **k: pytest.fail("written")
    _button(frame, "➕  Add open recording…").invoke()
    assert asked and not gone.exists()
