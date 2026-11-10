"""Group statistics on a Compare segments group table: rows to check, paired
tests across animals, Holm correction, the report and the group figure."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from ecg.io import group_stats as gs


def _table(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        rr_a = 180 + 10 * rng.standard_normal()
        rows.append({
            "subject": str(240 + i), "file": f"{240 + i}.mat", "template": "Before / after, 5 min",
            "start (s) Pre": 0.0, "end (s) Pre": 300.0, "QC Pre": "ok",
            "start (s) Post": 330.0, "end (s) Post": 630.0, "QC Post": "ok",
            "valid RR (%) Pre": 100.0, "valid RR (%) Post": 100.0,
            "Mean RR (ms) Pre": rr_a, "Mean RR (ms) Post": rr_a - 8,     # HR up in all
            "Mean HR (60000/mean NN) (bpm) Pre": 60000 / rr_a,
            "Mean HR (60000/mean NN) (bpm) Post": 60000 / (rr_a - 8),
            "Poincaré SD1 (ms) Pre": 2.5 + 0.1 * i, "Poincaré SD1 (ms) Post": 2.5 + 0.1 * i,
            "Poincaré SD2 (ms) Pre": 6.0, "Poincaré SD2 (ms) Post": 5.0 + 0.2 * i,
        })
    return pd.DataFrame(rows)


def test_segments_and_metrics_are_read_from_the_columns():
    df = _table()
    df["start (s) Empty"] = np.nan
    assert gs.segment_tags(df) == ["Pre", "Post"]
    names = gs.metric_names(df, "Pre", "Post")
    assert names == ["Mean RR (ms)", "Mean HR (60000/mean NN) (bpm)",
                     "Poincaré SD1 (ms)", "Poincaré SD2 (ms)"]


def test_paired_tests_per_metric():
    st = gs.paired_stats(_table(), "Pre", "Post").set_index("Metric")
    # The default outcomes only: no Mean RR (not in DEFAULT_METRICS).
    assert list(st.index) == ["Mean HR (60000/mean NN) (bpm)", "Poincaré SD1 (ms)",
                              "Poincaré SD2 (ms)"]
    assert "Mean RR (ms)" in set(gs.paired_stats(_table(), "Pre", "Post", "all")["Metric"])
    hr = st.loc["Mean HR (60000/mean NN) (bpm)"]
    assert hr["n"] == 6 and hr["n up / down"] == "6 / 0"
    assert hr["Wilcoxon p"] == pytest.approx(gs.smallest_wilcoxon_p(6))   # 0.03125, exact
    assert hr["Median Δ (B − A)"] > 0
    sd1 = st.loc["Poincaré SD1 (ms)"]                     # no change at all: no test
    assert math.isnan(sd1["Wilcoxon p"]) and sd1["n up / down"] == "0 / 0"
    assert (st["Wilcoxon p (Holm)"].dropna() >= st["Wilcoxon p"].dropna()).all()


def test_holm():
    assert gs.holm([0.01, 0.04, 0.03, float("nan")])[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(gs.holm([float("nan")])[0])
    assert gs.smallest_wilcoxon_p(6) == pytest.approx(0.03125)


def test_rows_to_check():
    df = _table()
    df.loc[1, "QC Post"] = "overlaps the event or the 30 s after it (artefact)"
    df.loc[2, "subject"] = df.loc[3, "subject"]                 # one animal twice
    df.loc[4, "end (s) Post"] = 900.0                             # another duration
    df.loc[5, ["start (s) Post"]] = np.nan                        # no Post segment
    df.loc[0, "template"] = ""
    checks = gs.check_rows(df, "Pre", "Post")
    probs = ["; ".join(c.problems) for c in checks]
    assert "template 'none'" in probs[0]
    assert "overlaps the event" in probs[1]
    assert "on several rows" in probs[2] and "on several rows" in probs[3]
    assert "lasts 570 s, not 300 s" in probs[4]
    assert "no segment 'Post'" in probs[5]


def test_report(tmp_path):
    df = _table()
    st = gs.paired_stats(df, "Pre", "Post")
    path = tmp_path / "stats.xlsx"
    gs.write_group_report(str(path), df, "Pre", "Post", st, gs.check_rows(df, "Pre", "Post"),
                          excluded=["249"], note="x")
    book = pd.read_excel(path, sheet_name=None)
    assert list(book) == ["Statistics", "About", "Rows checked", "Paired values"]
    about = dict(zip(book["About"]["Item"], book["About"]["Value"]))
    assert about["Rows left out"] == "249" and "0.0312" in about["Smallest possible Wilcoxon p"]
    assert len(book["Paired values"]) == len(st) * len(df)


def test_group_figure_one_ellipse_per_animal_and_the_mean():
    from ecg.ui.group_figure import draw_group_figure
    df = _table()
    fig = Figure(figsize=(14, 4.6))
    assert draw_group_figure(fig, df, "Pre", "Post", ("Before", "After"))
    poin = fig.axes[:2]
    assert poin[0].get_xlim() == poin[1].get_xlim() == poin[0].get_ylim()
    ells = [p for p in poin[1].patches if isinstance(p, Ellipse)]
    assert len(ells) == len(df) + 1
    mean = ells[-1]
    assert mean.width == pytest.approx(2 * df["Poincaré SD2 (ms) Post"].mean())
    assert mean.height == pytest.approx(2 * df["Poincaré SD1 (ms) Post"].mean())
    assert len(fig.axes) == 2 + 3                                   # HR, SD1, SD2 panels
    assert not draw_group_figure(Figure(), df.iloc[0:0], "Pre", "Post")


def test_zero_differences_are_not_counted_in_the_test_n():
    df = _table()
    df["pNN6 (%) Pre"] = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    df["pNN6 (%) Post"] = [0.0, 0.0, 0.0, 2.0, 3.0, 4.0]
    r = gs.paired_stats(df, "Pre", "Post", "all").set_index("Metric").loc["pNN6 (%)"]
    assert r["n"] == 6 and r["n tested (Δ ≠ 0)"] == 3
    assert r["Wilcoxon p"] == pytest.approx(gs.smallest_wilcoxon_p(3))      # 0.25, not 0.031


def test_subjects_read_as_numbers_or_empty():
    assert gs.subject_text(243.0) == "243" and gs.subject_text(float("nan")) == ""
    assert gs.subject_text(" M1 ") == "M1" and gs.subject_text("nan") == ""
    df = _table()
    df["subject"] = [243.0, 244.0, np.nan, 246.0, 247.0, 248.0]
    checks = gs.check_rows(df, "Pre", "Post")
    assert checks[0].subject == "243" and checks[2].problems == ["no subject"]


def test_other_protocol_rows_do_not_flag_this_comparison():
    df = _table(4)
    other = _table(6)
    other = other.drop(columns=[c for c in other.columns if c.endswith((" Pre", " Post"))])
    other["start (s) Base"], other["end (s) Base"] = 0.0, 300.0
    other["template"] = "Stim 10 s"
    other["subject"] = [str(240 + i) for i in range(6)]          # same animals
    other["file"] = [f"{240 + i}_stim.mat" for i in range(6)]
    checks = gs.check_rows(pd.concat([df, other], ignore_index=True), "Pre", "Post")
    assert all(c.ok for c in checks[:4])                          # the Pre/Post rows
    assert all(any("no segment" in p for p in c.problems) for c in checks[4:])


def test_a_longer_tag_ending_like_another_is_not_a_metric():
    df = pd.DataFrame({"subject": ["1"], "file": ["f"],
                       "start (s) A": [0.0], "start (s) B": [0.0],
                       "start (s) Post A": [0.0], "start (s) Post B": [0.0],
                       "SDNN (ms) A": [1.0], "SDNN (ms) B": [2.0],
                       "SDNN (ms) Post A": [3.0], "SDNN (ms) Post B": [4.0]})
    assert gs.metric_names(df, "A", "B") == ["SDNN (ms)"]
    assert gs.metric_names(df, "Post A", "Post B") == ["SDNN (ms)"]


def test_default_family_has_one_test_per_question():
    assert "RMSSD (ms)" not in gs.DEFAULT_METRICS and "HF (n.u.) (%)" not in gs.DEFAULT_METRICS
    assert "Poincaré SD1 (ms)" in gs.DEFAULT_METRICS and "LF/HF" in gs.DEFAULT_METRICS
