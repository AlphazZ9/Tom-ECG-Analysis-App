"""Group statistics, all pairs: every consecutive pair of segments of a group
table (B vs A, D vs C, F vs E) tested across animals, Holm within each pair
and optionally across all pairs, the label check, the workbook and the
figure."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ecg.io import group_stats as gs
from ecg.io.group_table import append_group_row, group_row

HR = "Mean HR (60000/mean NN) (bpm)"
LABELS = ["Pre Stim 1min", "Post Stim 1min", "Pre Stim 5min", "Post Stim 5min",
          "Pre Stim 10min", "Post Stim 10min", "Recovery"]


def _table(n: int = 6, tags: str = "ABCDEFG", labels: bool = True) -> pd.DataFrame:
    """n animals × 7 segments (3 pairs and an odd one): HR up by 20 bpm in
    pair 1, by 10 in pair 2, no change in pair 3."""
    rng = np.random.default_rng(3)
    shift = {0: 0.0, 1: 20.0, 2: 0.0, 3: 10.0, 4: 0.0, 5: 0.0, 6: 0.0}
    rows = []
    for i in range(n):
        base = 500 + 20 * rng.standard_normal()
        row = {"subject": str(100 + i), "file": f"{100 + i}.mat", "template": "T"}
        for k, t in enumerate(tags):
            row[f"start (s) {t}"] = 60.0 * k
            row[f"end (s) {t}"] = 60.0 * k + 60.0
            row[f"QC {t}"] = "ok"
            if labels:
                row[f"label {t}"] = LABELS[k]
            hr = base + shift[k] + (0.5 * i if k == 1 else 0.0) + (0.25 * i if k == 3 else 0.0)
            if k in (4, 5):
                hr = base + (1.0 if (k == 5 and i % 2) else 0.0) - (1.0 if (k == 5 and not i % 2) else 0.0)
            row[f"{HR} {t}"] = hr
            row[f"Mean RR (ms) {t}"] = 60000.0 / hr
            row[f"SDNN (ms) {t}"] = 3.0 + 0.1 * i + (0.5 if k == 1 else 0.0)
            row[f"Poincaré SD1 (ms) {t}"] = 2.0 + 0.05 * k
            row[f"Poincaré SD2 (ms) {t}"] = 5.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_consecutive_pairs_and_the_odd_tag():
    assert gs.consecutive_pairs("ABCDEFG") == ([("A", "B"), ("C", "D"), ("E", "F")], "G")
    assert gs.consecutive_pairs("ABCD") == ([("A", "B"), ("C", "D")], None)
    assert gs.consecutive_pairs("A") == ([], "A")


def test_all_pairs_stats_three_pairs_and_an_odd_segment():
    df = _table()
    res = gs.all_pairs_stats(df)
    assert [(p.a, p.b) for p in res.pairs] == [("A", "B"), ("C", "D"), ("E", "F")]
    assert res.odd == "G" and any("G has no partner" in n for n in res.notes)
    assert not res.holm_across
    for pr in res.pairs:
        # The same per-metric tests as one pair A vs B on the same rows.
        ref = gs.paired_stats(df, pr.a, pr.b)
        pd.testing.assert_frame_equal(pr.stats, ref)
        assert gs.HOLM_ALL not in pr.stats.columns
        assert len(pr.used) == 6 and not pr.left_out
    hr = [pr.stats.set_index("Metric").loc[HR] for pr in res.pairs]
    assert hr[0]["Median Δ (B − A)"] > 19 and hr[1]["Median Δ (B − A)"] > 9
    assert hr[0]["Wilcoxon p"] == pytest.approx(gs.smallest_wilcoxon_p(6))
    assert abs(hr[2]["Median Δ (B − A)"]) <= 1 and hr[2]["Wilcoxon p"] > 0.5
    # The group's labels name the pairs.
    assert res.pairs[0].title == "Post Stim 1min vs Pre Stim 1min (B vs A)"
    assert (res.pairs[1].name_a, res.pairs[1].name_b) == ("Pre Stim 5min", "Post Stim 5min")


def test_holm_across_all_pairs_and_metrics():
    df = _table()
    res = gs.all_pairs_stats(df, holm_across=True)
    flat_p = [p for pr in res.pairs for p in pr.stats["Wilcoxon p"]]
    flat_adj = [p for pr in res.pairs for p in pr.stats[gs.HOLM_ALL]]
    expected = gs.holm(flat_p)
    for got, exp in zip(flat_adj, expected):
        assert (math.isnan(got) and math.isnan(exp)) or got == pytest.approx(exp)
    # Across pairs is at least as strict as within a pair; within is unchanged.
    for pr in res.pairs:
        ok = pr.stats[gs.HOLM_ALL].notna()
        assert (pr.stats.loc[ok, gs.HOLM_ALL] >= pr.stats.loc[ok, "Wilcoxon p (Holm)"] - 1e-12).all()
        pd.testing.assert_series_equal(pr.stats["Wilcoxon p (Holm)"],
                                       gs.paired_stats(df, pr.a, pr.b)["Wilcoxon p (Holm)"])


def test_label_consistency_rows_are_flagged_and_left_out():
    df = _table()
    df.loc[2, "label C"] = "Pre Stim 10min"             # another window under tag C
    df.loc[4, "label A"] = "  pre stim   1MIN "         # case and spacing do not count
    res = gs.all_pairs_stats(df)
    p1, p2, _p3 = res.pairs
    assert all(c.ok for c in p1.checks)
    bad = [c for c in p2.checks if not c.ok]
    assert [c.subject for c in bad] == ["102"]
    assert "C is 'Pre Stim 10min', not 'Pre Stim 5min' like the group" in bad[0].problems
    assert p2.left_out == ["102"] and len(p2.used) == 5
    assert p2.label_a == "Pre Stim 5min"
    # Not left out without "Leave out rows to check": kept, still listed.
    kept = gs.all_pairs_stats(df, exclude=False).pairs[1]
    assert len(kept.used) == 6 and not kept.checks[2].ok


def test_labels_without_majority_and_old_tables():
    df = _table(n=2)
    df.loc[1, "label A"] = "Baseline"
    probs, notes = gs.label_problems(df, ["A", "B"])
    assert probs == [[], []] and any("no majority label" in n for n in notes)
    old = _table(labels=False)
    res = gs.all_pairs_stats(old)
    assert all(c.ok for pr in res.pairs for c in pr.checks)
    assert any("no label stored" in n for n in res.pairs[0].notes)
    assert res.pairs[0].title == "B vs A"


def test_group_row_stores_labels_and_they_are_not_metrics(tmp_path):
    metrics = [("mean_hr", "Mean HR (60000/mean NN)", "bpm")]
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "mean_hr": 500.0},
            {"label": "Pre", "t_lo": 60.0, "t_hi": 120.0, "mean_hr": 520.0}]
    row = group_row("M1", "m1.mat", segs, metrics)
    assert row["label A"] == "Pre" and row["label B"] == "Pre"     # tags A, B: labels repeat
    path = str(tmp_path / "g.xlsx")
    append_group_row(path, row)
    df = gs.read_group_table(path)
    assert gs.metric_names(df, "A", "B") == [HR]


def test_an_old_table_gets_label_columns_for_new_rows(tmp_path):
    path = str(tmp_path / "g.xlsx")
    old = _table(n=2, labels=False)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        old.to_excel(xw, sheet_name="Group", index=False)
    metrics = [("mean_hr", "Mean HR (60000/mean NN)", "bpm")]
    segs = [{"label": lbl, "t_lo": 60.0 * k, "t_hi": 60.0 * k + 60, "mean_hr": 500.0 + k}
            for k, lbl in enumerate(["Pre", "Pre"])]
    append_group_row(path, group_row("999", "999.mat", segs, metrics, template="T"))
    df = gs.read_group_table(path)
    assert "label A" in df.columns and df["label A"].isna().sum() == 2
    res = gs.all_pairs_stats(df)
    assert len(res.pairs[0].used) == 3          # the old rows are still tested
    assert all(c.ok for c in res.pairs[0].checks)   # their empty labels are not a mismatch


def test_all_pairs_workbook(tmp_path):
    df = _table()
    df.loc[2, "label C"] = "Pre Stim 10min"
    res = gs.all_pairs_stats(df, holm_across=True)
    path = tmp_path / "all.xlsx"
    names = gs.write_all_pairs_report(str(path), res, note="group table: g.xlsx")
    book = pd.read_excel(path, sheet_name=None, header=None)
    assert list(book) == ["Summary"] + names and len(names) == 3
    assert names[0] == "1 B vs A"
    summ = pd.read_excel(path, sheet_name="Summary", nrows=sum(len(p.stats) for p in res.pairs))
    assert list(summ.columns) == ["Pair", "A", "B", "Comparison", "Metric", "n",
                                  "Median Δ (B − A)", "Wilcoxon p", "Wilcoxon p (Holm)",
                                  gs.HOLM_ALL]
    assert set(summ["Pair"]) == {1, 2, 3}
    hr1 = summ[(summ["Pair"] == 1) & (summ["Metric"] == HR)].iloc[0]
    st1 = res.pairs[0].stats.set_index("Metric").loc[HR]
    assert hr1["n"] == 6 and hr1["Wilcoxon p"] == pytest.approx(st1["Wilcoxon p"])
    assert hr1["Median Δ (B − A)"] == pytest.approx(st1["Median Δ (B − A)"])
    assert summ[(summ["Pair"] == 2)]["n"].iloc[0] == 5           # row 102 left out
    # str() per cell: pandas 3's astype(str) keeps empty cells as NaN.
    flat = [str(v) for v in book["Summary"].to_numpy().ravel().tolist()]
    assert any("G has no partner" in v for v in flat)
    # A pair sheet: the four single-pair tables one below the other.
    sheet = book[names[1]]
    col0 = sheet[0].astype(str).tolist()
    for title in ("Statistics", "About", "Rows checked", "Paired values"):
        assert title in col0
    assert any("not 'Pre Stim 5min' like the group" in v
               for v in map(str, sheet.to_numpy().ravel()))
    assert any("102" == v for v in map(str, sheet.to_numpy().ravel()))   # left out, listed


def test_sheet_names_are_valid_and_unique():
    pr = gs.PairResult("Pre/1:x" * 5, "Post?[1]" * 5, "", "", pd.DataFrame(), pd.DataFrame(),
                       [], [], [])
    taken: "set[str]" = set()
    a, b = gs._sheet_name(1, pr, taken), gs._sheet_name(1, pr, taken)
    assert a != b and len(a) <= 31 and len(b) <= 31
    assert not set("[]:*?/\\") & set(a + b)


def test_all_pairs_figure_one_panel_per_pair():
    from ecg.ui.group_figure import draw_all_pairs_figure
    res = gs.all_pairs_stats(_table())
    fig = Figure(figsize=(10, 4))
    assert draw_all_pairs_figure(fig, res)
    assert len(fig.axes) == 3
    ax = fig.axes[0]
    assert len(ax.lines) == 6 + 1                         # one per animal, the median
    assert "Post Stim 1min vs Pre Stim 1min" in ax.get_title() and "p = 0.031" in ax.get_title()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Pre Stim 1min", "Post Stim 1min"]
    res_h = gs.all_pairs_stats(_table(), holm_across=True)
    fig2 = Figure()
    assert draw_all_pairs_figure(fig2, res_h) and "Holm all" in fig2.axes[0].get_title()
    assert not draw_all_pairs_figure(Figure(), res, metric="nope")


def test_all_pairs_figsize():
    from ecg.ui.compare_segments import all_pairs_figsize
    assert all_pairs_figsize(3)[0] == pytest.approx(9.9) and all_pairs_figsize(1)[0] == 5.0


def test_long_tags_give_numbered_sheet_names(tmp_path):
    df = _table(tags=["Pre Stim 1min", "Post Stim 1min", "Pre Stim 10min", "Post Stim 10min"],
                labels=False)
    res = gs.all_pairs_stats(df)
    assert res.pairs[1].title == "Post Stim 10min vs Pre Stim 10min"     # tags are the labels
    names = gs.write_all_pairs_report(str(tmp_path / "x.xlsx"), res)
    assert names == ["Pair 1", "Pair 2"]


# ── Pairs follow each row's own segments, not the table's column order ────

METRICS = [("mean_hr", "Mean HR (60000/mean NN)", "bpm")]


def _rec_table(path, labels_per_row, hr_up=(20.0, 10.0)):
    """A group table written through group_row / append_group_row: one row per
    animal, 4 segments of 60 s, HR up by hr_up[j] in pair j."""
    for i, labels in enumerate(labels_per_row):
        segs = []
        for k, lbl in enumerate(labels):
            base = 500.0 + 3 * i
            hr = base + (hr_up[k // 2] + 0.5 * i if k % 2 else 0.0)
            segs.append({"label": lbl, "t_lo": 300.0 * k, "t_hi": 300.0 * k + 60,
                         "mean_hr": hr})
        append_group_row(str(path), group_row(f"M{i + 1}", f"m{i + 1}.mat", segs, METRICS,
                                              template="T"))
    return gs.read_group_table(str(path))


TYPO = ([["Pre 1", "Post 1", "Pre 5", "Post 5"]] * 3
        + [["Pre 1", "Post 1 min", "Pre 5", "Post 5 min"]] * 3)


@pytest.mark.parametrize("with_index", [True, False])
def test_a_label_typo_never_makes_a_pair_of_its_own(tmp_path, with_index):
    df = _rec_table(tmp_path / "g.xlsx", TYPO)
    assert gs.segment_tags(df) == ["Pre 1", "Post 1", "Pre 5", "Post 5", "Post 1 min",
                                   "Post 5 min"]
    if not with_index:              # a table written before "index <tag>": start times
        df = df[[c for c in df.columns if not c.startswith("index ")]]
    res = gs.all_pairs_stats(df)
    assert [(p.a, p.b) for p in res.pairs] == [("Pre 1", "Post 1"), ("Pre 5", "Post 5")]
    assert res.odd is None
    assert any("held by as many rows" in n for n in res.notes)
    for j, pr in enumerate(res.pairs):
        bad = [c for c in pr.checks if not c.ok]
        assert [c.subject for c in bad] == ["M4", "M5", "M6"]
        assert all(f"segment {2 * j + 2} is 'Post {1 if j == 0 else 5} min'" in c.problems[0]
                   for c in bad)
        assert pr.left_out == ["M4", "M5", "M6"] and len(pr.used) == 3
        assert pr.stats.set_index("Metric").loc[HR, "n"] == 3
    # Kept on request: their own Post segment is paired with their own Pre.
    kept = gs.all_pairs_stats(df, exclude=False)
    for j, pr in enumerate(kept.pairs):
        assert len(pr.used) == 6
        d = (pd.to_numeric(pr.used[f"{HR} {pr.b}"]) - pd.to_numeric(pr.used[f"{HR} {pr.a}"]))
        assert (d > (19 if j == 0 else 9)).all()


def test_a_case_slip_goes_under_the_existing_segment(tmp_path):
    rows = [["Pre Stim 1min", "Post Stim 1min", "Pre Stim 5min", "Post Stim 5min"]] * 2
    rows.append(["Pre Stim 1min", "Post Stim 1min", "pre stim 5min", "Post  Stim 5min"])
    df = _rec_table(tmp_path / "g.xlsx", rows)
    # append_group_row put M3's segments under the group's columns.
    assert gs.segment_tags(df) == rows[0]
    assert df.loc[2, "label Pre Stim 5min"] == "pre stim 5min"
    res = gs.all_pairs_stats(df)
    assert all(c.ok for pr in res.pairs for c in pr.checks)
    assert [len(pr.used) for pr in res.pairs] == [3, 3]


def test_a_case_slip_in_a_table_is_taken_as_the_same_segment():
    df = _table(n=4, tags=["Pre 1", "Post 1", "Pre 5", "Post 5"], labels=False)
    df = df.rename(columns={c: c[: -len("Pre 5")] + "pre 5" for c in df.columns
                            if c.endswith(" Pre 5")})
    df.loc[0:2, [c for c in df.columns if c.endswith(" pre 5")]] = np.nan
    df.loc[0:2, "start (s) Pre 5"] = 120.0      # three rows typed it right
    for c in [c for c in df.columns if c.endswith(" pre 5") and not c.startswith("start")]:
        df.loc[0:2, c[: -len("pre 5")] + "Pre 5"] = df.loc[3, c]
    res = gs.all_pairs_stats(df)
    assert [(p.a, p.b) for p in res.pairs] == [("Pre 1", "Post 1"), ("Pre 5", "Post 5")]
    pr = res.pairs[1]
    assert len(pr.used) == 4 and all(c.ok for c in pr.checks)
    assert any("'pre 5' taken as 'Pre 5'" in n for n in pr.notes)


def test_a_row_lacking_a_pair_and_extra_segments_are_reported(tmp_path):
    rows = [["Pre 1", "Post 1", "Pre 5", "Post 5"]] * 3 + [["Pre 1", "Post 1"]]
    df = _rec_table(tmp_path / "g.xlsx", rows)
    res = gs.all_pairs_stats(df)
    assert [len(pr.used) for pr in res.pairs] == [4, 3]
    assert any("do not all hold 4 segments" in n for n in res.notes)


def test_group_row_stores_the_place_and_collapses_spacing():
    segs = [{"label": "Pre  1", "t_lo": 0.0, "t_hi": 60.0},
            {"label": "Post 1", "t_lo": 60.0, "t_hi": 120.0}]
    row = group_row("M1", "m1.mat", segs, METRICS)
    assert row["index Pre 1"] == 1 and row["index Post 1"] == 2
    assert "start (s) Pre 1" in row
