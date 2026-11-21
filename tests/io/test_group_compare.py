"""The Group tab's analyses (ecg.io.group_compare): groups by free group, sex
or none; tests within and between groups; the mixed ANOVA, checked against
pingouin's mixed_anova / rm_anova on the same data (values pinned here)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.io.prism_template import read_tables, write_metric_pzfx


def _anova_data():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(9, 5)) * np.array([1, 2, 3, 1.5, 0.5]) + np.arange(5)
    g = np.array(["A"] * 4 + ["B"] * 5)
    y[g == "B"] += 0.8
    y[g == "B", 2] += 1.5
    return y, g


def test_mixed_anova_matches_pingouin():
    y, g = _anova_data()
    an = gc.mixed_anova(y, g).set_index("Source")
    # pingouin 0.5: mixed_anova(dv, within, between, subject, correction=True)
    assert an.loc["Group", ["df1", "df2"]].tolist() == [1, 7]
    assert an.loc["Group", "F"] == pytest.approx(3.70192, abs=1e-4)
    assert an.loc["Group", "p"] == pytest.approx(0.09576, abs=1e-4)
    assert an.loc["Within", "F"] == pytest.approx(9.73795, abs=1e-4)
    assert an.loc["Within", "epsilon"] == pytest.approx(0.52324, abs=1e-4)
    assert an.loc["Within", "p GG"] == pytest.approx(0.00186, abs=1e-4)
    assert an.loc["Group × within", "F"] == pytest.approx(1.23318, abs=1e-4)
    assert an.loc["Group × within", "p GG"] == pytest.approx(0.32165, abs=1e-4)


def test_one_group_is_the_repeated_measures_anova():
    y, g = _anova_data()
    an = gc.mixed_anova(y[g == "A"], g[g == "A"])
    assert list(an["Source"]) == ["Within"]
    r = an.iloc[0]
    assert (r["df1"], r["df2"]) == (4, 12)
    assert r["F"] == pytest.approx(5.53957, abs=1e-4)
    assert r["p"] == pytest.approx(0.0092, abs=1e-4)
    assert r["epsilon"] == pytest.approx(0.49876, abs=1e-4)
    assert r["p GG"] == pytest.approx(0.04353, abs=1e-4)


def test_two_levels_need_no_correction():
    y, g = _anova_data()
    an = gc.mixed_anova(y[:, :2], g)
    assert (an["epsilon"].dropna() == 1.0).all()


def test_between_test_two_and_more_groups():
    two = gc.between_test([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert two["rank test"] == "Mann-Whitney" and two["p"] == pytest.approx(0.0286, abs=1e-3)
    three = gc.between_test([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert three["rank test"] == "Kruskal-Wallis" and three["p"] < 0.05
    assert math.isnan(gc.between_test([[1], []])["p"])


def _animals(rows):
    return pd.DataFrame(rows, columns=["subject", "sex", "group", "use", "file"])


def test_groups_by_group_sex_none_and_the_control_first():
    an = _animals([("1", "M", "CNO", True, "a"), ("2", "F", "Control", True, "b"),
                   ("3", "", "", True, "c"), ("4", "F", "CNO", False, "d")])
    gmap, order, notes = gc.animal_groups(an, gc.BY_GROUP)
    assert order == ["Control", "CNO"] and ("4", "d") not in gmap
    assert any("3" in n for n in notes)
    gmap, order, _ = gc.animal_groups(an, gc.BY_SEX)
    assert order == ["Males", "Females"] and ("3", "c") not in gmap
    gmap, order, _ = gc.animal_groups(an, gc.BY_NONE)
    assert order == [gc.ALL_LABEL] and len(gmap) == 3


def _table(tmp_path, n=8):
    """n animals, half Control (HR + 20 bpm after the stimulation), half CNO
    (+ 5), a stimulation at 300 s + 10 s per animal."""
    p = str(tmp_path / "g.xlsx")
    rng = np.random.default_rng(2)
    rows = []
    for i in range(n):
        grp = "Control" if i % 2 == 0 else "CNO"
        sex = "M" if i < n // 2 else "F"
        stim = 300.0 + 10 * i
        gain = 20.0 if grp == "Control" else 5.0
        t, rr = 0.0, []
        ts = []
        while t < 900:
            hr = 500.0 + (gain if t > stim else 0.0) + rng.normal(0, 2)
            r = 60000.0 / hr
            t += r / 1000.0
            ts.append(t)
            rr.append(r)
        ts, rr = np.array(ts), np.array(rr)
        seg = {"Mean HR (bpm) Pre": 500 + rng.normal(0, 2), "Mean HR (bpm) Post": 500 + gain,
               "start (s) Pre": stim - 120, "start (s) Post": stim, "label Pre": "Pre",
               "label Post": "Post"}
        lines = {gd.SHEET_TIME: gd.time_frame(ts, rr, np.ones(len(rr), bool)),
                 gd.SHEET_ANNOT: gd.annotations_frame([{"t_start": stim, "label": "Stim"}]),
                 gd.SHEET_WHOLE: pd.DataFrame([{"Mean HR (60000/mean NN) (bpm)":
                                                float(np.mean(60000 / rr))}]),
                 gd.SHEET_SEGMENTS: pd.DataFrame([seg]),
                 gd.SHEET_EVENTS: gd.events_frame([
                     {"label": "Stim", "onset_s": stim, "valid": True,
                      "metrics": {"rCSI": {"pre": 1.0 + 0.01 * i, "post": 1.0 + gain / 10}}},
                     {"label": "Other", "onset_s": 10.0, "valid": True,
                      "metrics": {"rCSI": {"pre": 9.0, "post": 9.0}}}])}
        gd.add_recording(p, str(100 + i), sex, f"{100 + i}.smr", lines)
        rows.append((str(100 + i), grp))
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an["group"] = an["subject"].map(dict(rows))
    gd.save_animals(p, an)
    return p


def test_time_course_by_group(tmp_path):
    p = _table(tmp_path)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    res = gc.analyse_time(sh, an, gc.BY_GROUP, "stim", "HR", bin_s=30, start_min=-4,
                          end_min=8, window_min=(0, 2))
    assert res.groups == ["Control", "CNO"]
    s = res.stats.iloc[0]
    assert s["Control Δ mean"] == pytest.approx(20, abs=1.5)
    assert s["CNO Δ mean"] == pytest.approx(5, abs=1.5)
    assert s["between p"] == pytest.approx(0.0286, abs=1e-3)      # 4 vs 4, all apart
    assert set(res.anova["Source"]) == {"Group", "Time", "Group × time"}
    curve = res.figure["curves"]["Control"]
    t = res.figure["t_min"]
    assert len(t) == 24 and abs(np.nanmean(curve["mean"][t < 0])) < 1.0
    assert np.nanmean(curve["mean"][(t > 1) & (t < 2)]) == pytest.approx(20, abs=2)
    xy = [tb for tb in res.prism if len(tb) == 3]
    assert len(xy) == 2 and xy[0][1].columns[0] == "Time (min)"
    out = tmp_path / "t.pzfx"
    write_metric_pzfx(res.prism, str(out))
    assert len(read_tables(str(out))) == len(res.prism)


def test_time_course_leaves_out_an_animal_without_the_annotation(tmp_path):
    p = _table(tmp_path, n=6)
    sh = gd.read_sheets(p)
    sh[gd.SHEET_ANNOT] = sh[gd.SHEET_ANNOT].iloc[1:]
    res = gc.analyse_time(sh, gd.animals_frame(sh), gc.BY_NONE, "Stim", "RR", bin_s=60,
                          start_min=-4, end_min=8, window_min=(0, 2), delta=False)
    assert any("100" in n for n in res.notes) and res.n_animals == 5
    with pytest.raises(ValueError):
        gc.analyse_time(sh, gd.animals_frame(sh), gc.BY_NONE, "Stim", start_min=1)


def test_segments_whole_and_events(tmp_path):
    p = _table(tmp_path)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    seg = gc.analyse_segments(sh, an, gc.BY_GROUP, "Pre", "Post")
    r = seg.stats.iloc[0]
    assert r["Control Δ mean"] == pytest.approx(20, abs=3) and r["between p"] < 0.05
    assert set(seg.anova["Source"]) == {"Group", "Segment", "Group × segment"}
    assert {"Test", "Control", "CNO", "p", "p Holm"} <= set(seg.display.columns)
    whole = gc.analyse_whole(sh, an, gc.BY_SEX)
    assert whole.groups == ["Males", "Females"] and "between p (Holm)" in whole.stats
    one = gc.analyse_whole(sh, an, gc.BY_NONE)
    assert any("descriptive" in n for n in one.notes)
    ev = gc.analyse_events(sh, an, gc.BY_GROUP, "Stim")
    r = ev.stats.set_index("Metric").loc["rCSI"]
    assert r["Control post mean"] == pytest.approx(3.0)
    assert r["CNO post mean"] == pytest.approx(1.5)
    assert gc.event_labels(sh) == ["Other", "Stim"]
    out = tmp_path / "s.xlsx"
    gc.write_results(str(out), seg, "g.xlsx")
    assert {"Results", "Statistics", "ANOVA", "Per animal", "Settings"} <= set(
        pd.read_excel(out, sheet_name=None))


def test_an_unticked_animal_is_left_out(tmp_path):
    p = _table(tmp_path)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an.loc[0, "use"] = False
    res = gc.analyse_whole(sh, an, gc.BY_NONE)
    assert res.n_animals == len(an) - 1


def _event_sheets(rows):
    ev = pd.concat([gd.events_frame([r]).assign(subject=s, file=f"{s}.smr")
                    for s, r in rows], ignore_index=True)
    an = pd.DataFrame([{"subject": s, "sex": "", "group": "", "use": True, "file": f"{s}.smr"}
                       for s in sorted({s for s, _ in rows})])
    return {gd.SHEET_EVENTS: ev}, an


def _event(onset, method, window, hr_post, valid=True, hr_valid=True):
    return {"label": "Stim", "onset_s": onset, "method": method, "window_s": window,
            "valid": valid, "metrics": {"HR": {"pre": 500.0, "post": hr_post, "valid": hr_valid},
                                        "rCSI": {"pre": 1.0, "post": 2.0, "valid": valid}}}


def test_events_keep_one_configuration():
    """Codex review: the same event saved with two methods / windows must not
    be pooled (counted twice, weighted by configuration)."""
    sheets, an = _event_sheets([
        ("1", _event(100.0, "robust", 15.0, 520.0)),
        ("1", _event(100.0, "classic", 30.0, 600.0)),
        ("2", _event(90.0, "robust", 15.0, 530.0)),
    ])
    assert gc.event_configs(sheets) == ["robust · 15 s", "classic · 30 s"]
    res = gc.analyse_events(sheets, an, gc.BY_NONE, metrics=["HR"])
    vals = res.figure["values"]["HR"]
    assert vals[("1", "1.smr")] == (500.0, 520.0) and vals[("2", "2.smr")] == (500.0, 530.0)
    assert any("classic · 30 s" in n for n in res.notes)
    other = gc.analyse_events(sheets, an, gc.BY_NONE, metrics=["HR"], config="classic · 30 s")
    assert other.figure["values"]["HR"] == {("1", "1.smr"): (500.0, 600.0)}


def test_events_use_each_metrics_own_validity():
    """Codex review: an event unusable for the CSI family can be fine for HR."""
    sheets, an = _event_sheets([
        ("1", _event(100.0, "robust", 15.0, 520.0, valid=False, hr_valid=True)),
        ("1", _event(200.0, "robust", 15.0, 540.0, valid=True, hr_valid=False)),
    ])
    res = gc.analyse_events(sheets, an, gc.BY_NONE, metrics=["HR", "rCSI"])
    assert res.figure["values"]["HR"] == {("1", "1.smr"): (500.0, 520.0)}
    assert res.figure["values"]["rCSI"] == {("1", "1.smr"): (1.0, 2.0)}



def test_no_group_set_says_so(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an["group"] = ""
    with pytest.raises(ValueError, match="No animal has a group"):
        gc.analyse_time(sh, an, gc.BY_GROUP, "Stim", start_min=-4, end_min=8, window_min=(0, 2))


def test_two_recordings_of_one_subject_draw_two_curves(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    for name in (gd.SHEET_TIME, gd.SHEET_ANNOT):
        df = sh[name]
        sh[name] = df.assign(subject=df["subject"].replace({"101": "100"}))
    an = gd.animals_frame(sh)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "Stim", bin_s=60, start_min=-4, end_min=8,
                          window_min=(0, 2))
    curves = res.figure["curves"][gc.ALL_LABEL]
    assert curves["n"] == 4 and len(curves["animals"]) == 4
    assert {"100 (100.smr)", "100 (101.smr)"} <= set(curves["animals"])
