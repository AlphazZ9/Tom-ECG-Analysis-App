"""Compare segments exports: one source for the metric table, quality,
blocks, RR and Poincaré, written to Excel, CSV, the clipboard and Prism, and
added to the main Excel / Prism exports."""
from __future__ import annotations

import numpy as np
import openpyxl
import pandas as pd
import pytest

from ecg.core.analysis import compare_segments, compute_segment_stats
from ecg.core.segment_plan import quality_flags
from ecg.io import segment_export as se

FS = 1000.0
METRICS = [("n_beats", "Beats", ""), ("hr_mean", "Mean HR", "bpm"),
           ("sd1", "Poincaré SD1", "ms"), ("lf_hf", "LF/HF", "")]


def _result(n_segs: int = 2, labels=("Before", "After", "Late")) -> se.ComparisonResult:
    rng = np.random.default_rng(3)
    rr = np.r_[130 + 3 * rng.standard_normal(1500), 115 + 1.5 * rng.standard_normal(1500),
               120 + 2 * rng.standard_normal(1500)]
    rp = (np.r_[0, np.cumsum(np.round(rr))] + 500).astype(int)
    sig = np.zeros(rp[-1] + 1000)
    edges = [rp[0] / FS, rp[1500] / FS, rp[3000] / FS, rp[-1] / FS]
    segs = [compute_segment_stats(sig, rp, FS, edges[k] + 1, edges[k + 1] - 1, labels[k])
            for k in range(n_segs)]
    cmps = [compare_segments(segs[0], s) for s in segs[1:]]
    return se.ComparisonResult(
        segs=segs, metrics=METRICS, cmps=cmps, flags=quality_flags(segs),
        template="Before / after, 5 min", subject="M1", filepath="D:/x/1720M.mat",
        event=(200.0, 210.0), artefact_margin_s=30.0, block_s=10.0,
        lf_band=(0.4, 1.5), hf_band=(1.5, 5.0))


def test_comparison_frame_has_each_segment_and_its_delta_to_the_first():
    res = _result(3)
    df = se.comparison_frame(res)
    assert list(df.columns) == ["Metric", "Before", "After", "Late",
                                "Δ% Before→After", "Δ% Before→Late"]
    assert list(df["Metric"]) == ["Beats", "Mean HR (bpm)", "Poincaré SD1 (ms)", "LF/HF"]
    hr = df.set_index("Metric").loc["Mean HR (bpm)"]
    assert hr["Δ% Before→After"] == pytest.approx(100 * (hr["After"] - hr["Before"]) / hr["Before"])
    assert se.delta_pct(0.0, 5.0) != se.delta_pct(0.0, 5.0)        # NaN on a zero reference


def test_tsv_is_the_same_table():
    res = _result()
    lines = se.comparison_tsv(res).split("\n")
    assert lines[0].split("\t") == ["Metric", "Before", "After", "Δ% Before→After"]
    assert len(lines) == 1 + len(METRICS)
    beats = lines[1].split("\t")
    assert beats[0] == "Beats" and float(beats[1]) == res.segs[0]["n_beats"]


def test_xlsx_holds_every_sheet_and_the_summary(tmp_path):
    res = _result(3)
    path = tmp_path / "cmp.xlsx"
    se.write_xlsx(res, str(path), exported_at="2026-11-06 12:00")
    book = pd.read_excel(path, sheet_name=None)
    assert list(book)[:4] == ["Comparison", "Quality", "Summary", "Blocks"]
    assert {"RR Before", "RR Late", "Poincaré After"} <= set(book)
    summary = dict(zip(book["Summary"]["Item"], book["Summary"]["Value"]))
    assert summary["Template"] == "Before / after, 5 min" and summary["Reference"] == "Before"
    assert summary["Event (s)"] == "200–210" and "Before vs After" in summary
    assert len(book["Quality"]) == 3 and set(book["Quality"]["QC"]) <= {"ok"} | set(
        "; ".join(f) for f in res.flags)
    assert len(book["Poincaré Before"]) == len(res.segs[0]["pairs_x"])


def test_csv_is_the_metric_table(tmp_path):
    res = _result()
    path = tmp_path / "cmp.csv"
    se.write_csv(res, str(path))
    assert list(pd.read_csv(path).columns) == list(se.comparison_frame(res).columns)


def test_sheet_names_are_valid_and_unique():
    used: "set[str]" = set()
    a = se._sheet_name("RR Post 0/5 [late]: a very long segment label", used)
    b = se._sheet_name("RR Post 0/5 [late]: a very long segment label", used)
    assert len(a) <= 31 and len(b) <= 31 and a != b
    assert not set('[]:*?/\\') & set(a + b)


def test_added_to_the_main_workbook_without_clashing():
    wb = openpyxl.Workbook()
    wb.active.title = "Seg Comparison"                        # a name already taken
    n = se.add_sheets(wb, _result(), exported_at="x")
    titles = [ws.title for ws in wb.worksheets]
    assert n == 4 + 2 + 2 and len(set(t.lower() for t in titles)) == len(titles)
    ws = wb[titles[1]]                                        # the comparison, renamed
    assert [c.value for c in ws[1]][:3] == ["Metric", "Before", "After"]


def test_prism_file_reads_back(tmp_path):
    pzfx = pytest.importorskip("pzfx")
    res = _result(3)
    path = tmp_path / "cmp.pzfx"
    n = se.write_pzfx(res, str(path), exported_at="x")
    tables = pzfx.read_pzfx(str(path))
    names = list(tables) if isinstance(tables, dict) else None
    assert n == len(se.prism_tables(res))
    if names is not None:
        assert "Comparison" in names and "HR block means (bpm)" in names
        assert "RR Late" in names and "Poincaré Before" in names


def test_prism_tables_xy_and_column_kinds():
    specs = {name: (df, xc, rn) for name, df, xc, rn in se.prism_tables(_result())}
    df, xc, rn = specs["Comparison"]
    assert xc == 0 and rn and list(df.columns) == ["Before", "After"]
    assert list(df.index) == ["Beats", "Mean HR (bpm)", "Poincaré SD1 (ms)", "LF/HF"]
    df, xc, rn = specs["RR After"]
    assert xc == "Time in segment (s)" and df["Time in segment (s)"].iloc[0] == 0.0
    df, xc, _rn = specs["Poincaré Before"]
    assert xc == "RR_n (ms)"
    assert specs["HR block means (bpm)"][1] == 0


def test_main_prism_export_includes_the_comparison(tmp_path):
    pytest.importorskip("pzfx")
    from ecg.core.analysis import analyse_core
    from ecg.io.export import PrismExporter
    rng = np.random.default_rng(0)
    rr = 120 + 2 * rng.standard_normal(600)
    rp = (np.r_[0, np.cumsum(np.round(rr))] + 500).astype(int)
    sig = np.zeros(rp[-1] + 1000)
    results = analyse_core(sig, rp, FS)
    without = PrismExporter.build_and_write(tmp_path / "a.pzfx", results, subject="M1")
    with_cmp = PrismExporter.build_and_write(tmp_path / "b.pzfx", results, subject="M1",
                                             comparison=_result())
    assert with_cmp == without + len(se.prism_tables(_result()))


def test_a_label_named_like_an_export_column_falls_back_to_letters():
    res = _result(2, labels=("Metric", "After"))
    df = se.comparison_frame(res)
    assert list(df.columns)[:3] == ["Metric", "A", "B"]
    assert df["Metric"].iloc[0] == "Beats"
    se.prism_tables(res)                                      # no ValueError


def test_csv_opens_in_excel_with_its_symbols(tmp_path):
    path = tmp_path / "cmp.csv"
    se.write_csv(_result(), str(path))
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and "Δ%".encode() in raw


def test_main_exports_take_a_comparison_only_while_its_basis_is_current():
    from types import SimpleNamespace
    from ecg.ui.compare_segments import CompareSegmentsMixin
    res = _result()
    sig = {"now": ("peaks", 1)}
    app = SimpleNamespace(analysis=SimpleNamespace(last_comparison=res,
                                                   last_comparison_basis=("peaks", 1)),
                          _compare_signature=lambda: sig["now"])
    assert CompareSegmentsMixin._current_comparison(app) is res
    sig["now"] = ("peaks", 2)                                  # peaks edited since
    assert CompareSegmentsMixin._current_comparison(app) is None
    app.analysis.last_comparison_basis = None
    sig["now"] = None
    assert CompareSegmentsMixin._current_comparison(app) is None
