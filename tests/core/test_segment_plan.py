"""Compare segments: windows placed around an event by templates, and the
quality checks of each segment."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ecg.core import segment_plan as sp


def test_builtin_before_after_leaves_out_the_artefact_after_the_event():
    t = next(t for t in sp.BUILTIN_TEMPLATES if t.name == "Before / after, 5 min")
    before, after = sp.resolve(t, 1800.0, 1810.0, 3600.0)
    assert (before.label, before.lo, before.hi) == ("Before", 1500.0, 1800.0)
    assert (after.label, after.lo, after.hi) == ("After", 1840.0, 2140.0)
    assert after.hi - after.lo == before.hi - before.lo
    assert not before.clamped and not after.clamped


def test_point_event_and_clamping_to_the_recording():
    t = sp.BUILTIN_TEMPLATES[0]
    before, after = sp.resolve(t, 100.0, None, 300.0)
    assert (before.lo, before.hi, before.clamped) == (0.0, 100.0, True)
    assert (after.lo, after.hi, after.clamped) == (130.0, 300.0, True)


def test_a_saved_template_follows_the_event_s_end(tmp_path):
    path = tmp_path / "t.json"
    # Windows typed around a 10 s stimulus at 1800 s.
    t = sp.template_from_windows(
        "Footshock", [("Base", 1500, 1800), ("Early", 1840, 2140), ("Late", 3300, 3600)],
        1800.0, 1810.0)
    assert [(w.ref, w.lo, w.hi) for w in t.windows] == [
        ("start", -300, 0), ("end", 30, 330), ("end", 1490, 1790)]
    names = [x.name for x in sp.save_template(t, path)]
    assert names[-1] == "Footshock" and len(names) == len(sp.BUILTIN_TEMPLATES) + 1
    # On another animal, with a 2 s stimulus at 1700 s.
    again = next(x for x in sp.load_templates(path) if x.name == "Footshock")
    w = sp.resolve(again, 1700.0, 1702.0, 4000.0)
    assert [(x.lo, x.hi) for x in w] == [(1400, 1700), (1732, 2032), (3192, 3492)]
    # Saved again: replaced, not duplicated; then deleted.
    sp.save_template(sp.template_from_windows("Footshock", [("B", 0, 10)], 20, 20), path)
    assert [x.name for x in sp.load_templates(path)].count("Footshock") == 1
    assert "Footshock" not in [x.name for x in sp.delete_template("Footshock", path)]


def test_templates_refuse_builtin_names_and_bad_windows(tmp_path):
    path = tmp_path / "t.json"
    with pytest.raises(ValueError):
        sp.save_template(sp.SegmentTemplate(sp.BUILTIN_TEMPLATES[0].name,
                                            (sp.WindowSpec("x", "start", 0, 1),)), path)
    with pytest.raises(ValueError):
        sp.save_template(sp.SegmentTemplate("bad", (sp.WindowSpec("x", "start", 5, 1),)), path)
    # An unreadable file or entry only loses itself.
    path.write_text("{ not json", encoding="utf-8")
    assert len(sp.load_templates(path)) == len(sp.BUILTIN_TEMPLATES)
    path.write_text(json.dumps({"templates": [{"name": "ok", "windows": [
        {"label": "a", "ref": "end", "lo": 0, "hi": 5}]}, {"name": "broken"}]}), encoding="utf-8")
    assert [t.name for t in sp.load_templates(path)][-1] == "ok"


def _seg(lo, hi, rr=120.0, n_pairs=2000, label="s", error=None):
    n = int((hi - lo) * 1000 / rr)
    return {"label": label, "t_lo": lo, "t_hi": hi, "duration_s": hi - lo,
            "rr_ms": np.full(n, rr), "pairs_x": np.full(n_pairs, rr),
            "pairs_y": np.full(n_pairs, rr), "excluded_pct": 0.0, "error": error}


def test_quality_of_a_clean_segment():
    q = sp.segment_quality(_seg(0, 300))
    assert q["valid_pct"] == pytest.approx(100, abs=0.1) and q["n_pairs"] == 2000
    assert sp.quality_flags([_seg(0, 300), _seg(600, 900)]) == [[], []]


def test_quality_flags_say_what_is_doubtful():
    gappy = _seg(600, 900)
    gappy["rr_ms"] = gappy["rr_ms"][: len(gappy["rr_ms"]) // 2]      # half the window
    flags = sp.quality_flags(
        [_seg(0, 300), gappy, _seg(1000, 1200), _seg(1805, 2105, n_pairs=50)],
        event=(1800.0, 1810.0), margin_s=30.0)
    assert flags[0] == []
    assert any("valid RR cover 50 %" in f for f in flags[1])
    assert any("lasts 200 s" in f for f in flags[2])
    assert any("50 Poincaré pairs" in f for f in flags[3])
    assert any("overlaps the event" in f for f in flags[3])
    # Starting after the margin is fine.
    assert sp.quality_flags([_seg(1840, 2140)], event=(1800.0, 1810.0)) == [[]]


def test_unique_tags():
    assert sp.unique_tags(["Before", "After"]) == ["Before", "After"]
    assert sp.unique_tags(["x", "x", "y"]) == ["A", "B", "C"]
    assert sp.unique_tags(["x", ""]) == ["A", "B"]


def test_abrupt_rr_jumps_are_flagged():
    s = _seg(0, 300)
    assert sp.segment_quality(s)["n_jumps"] == 0
    y = s["pairs_y"].copy()
    y[:5] = 240.0                                   # 5 missed beats: 0.25 % of 2000
    assert sp.quality_flags([dict(s, pairs_y=y)]) == [[]]
    y[:40] = 240.0                                  # 2 %: flagged
    (flags,) = sp.quality_flags([dict(s, pairs_y=y)])
    assert sp.segment_quality(dict(s, pairs_y=y))["n_jumps"] == 40
    assert any("40 abrupt RR jumps" in f for f in flags)


def test_sd1_driven_by_a_few_outliers_is_flagged():
    rng = np.random.default_rng(0)
    x = 120 + rng.standard_normal(2500)
    y = x + rng.standard_normal(2500)                 # SD1 about 0.7 ms
    s = dict(_seg(0, 300), pairs_x=x, pairs_y=y, sd1=float(np.std(y - x) / np.sqrt(2)))
    assert sp.quality_flags([s]) == [[]]
    y2 = y.copy()
    y2[:8] += 40.0                                    # 8 bad pairs, 0.3 %
    s2 = dict(s, pairs_y=y2, sd1=float(np.std(y2 - x) / np.sqrt(2)))
    (flags,) = sp.quality_flags([s2])
    assert any("outlier pairs drive it" in f for f in flags)
    assert not any("abrupt RR jumps" in f for f in flags)
