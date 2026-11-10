"""
RR-based premature beats (classify_arrhythmias): isolated beats, couplets
and runs >= 3 are all reported; beats removed by automatic artifact
correction are put back for the classification and still counted; the
summary (counts, per-hour rates) and its export rows; legacy "esv_run"
events are read as "premature_run".
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ecg.core.detection import (
    classify_arrhythmias,
    correct_rr_artifacts,
    premature_beat_summary,
    premature_summary_rows,
    removed_beats_for_classification,
    removed_ectopic_samples,
)
from ecg.core.models import (
    PREMATURE_DEFINITION_TEXT,
    ArrhythmiaEvent,
    event_kind_label,
    normalize_event_kind,
)

from tests.helpers import rpeaks_from_rr

FS = 1000.0
BASELINE_S = 2.0


def _rr_mixture() -> "list[float]":
    # one isolated premature beat (70 ms + compensatory 170 ms), one couplet,
    # one run of 4 -- separated by long stretches of sinus rhythm
    return ([120.0] * 30 + [70.0, 170.0] + [120.0] * 30
            + [70.0, 70.0, 220.0] + [120.0] * 30
            + [70.0] * 4 + [120.0] * 30)


def _premature(events, kind=None):
    kinds = ("premature_beat", "premature_couplet", "premature_run") if kind is None else (kind,)
    return [e for e in events if e.kind in kinds]


def test_isolated_couplet_and_run_are_each_reported():
    rp = rpeaks_from_rr(_rr_mixture(), FS)
    events = classify_arrhythmias(rp, FS, baseline_s=BASELINE_S)
    iso = _premature(events, "premature_beat")
    cpl = _premature(events, "premature_couplet")
    run = _premature(events, "premature_run")
    assert [e.n_beats for e in iso] == [1]
    assert [e.n_beats for e in cpl] == [2]
    assert [e.n_beats for e in run] == [4]
    assert (iso[0].severity, cpl[0].severity, run[0].severity) == ("info", "warning", "alert")
    # the isolated beat sits at the 31st peak (end of the 70 ms interval)
    assert iso[0].t_end == pytest.approx(rp[31] / FS)
    assert all("RR-based" in e.label for e in _premature(events))
    # an irregular run caused by a premature beat is not reported on top
    assert not any(e.kind == "irregular_run" for e in events)


def test_isolated_premature_beats_close_together_are_not_merged():
    rr = [120.0] * 20 + [70.0, 170.0, 120.0, 70.0, 170.0] + [120.0] * 20
    events = classify_arrhythmias(rpeaks_from_rr(rr, FS), FS, baseline_s=BASELINE_S)
    iso = _premature(events, "premature_beat")
    assert len(iso) == 2 and all(e.n_beats == 1 for e in iso)


def test_premature_beats_removed_by_auto_correction_are_still_counted():
    rp = rpeaks_from_rr(_rr_mixture(), FS)
    ref = classify_arrhythmias(rp, FS, baseline_s=BASELINE_S)
    corrected, report = correct_rr_artifacts(rp, FS)
    assert report["n_ectopic"] > 0

    # without the removal record the corrected beats vanish
    lost = classify_arrhythmias(corrected, FS, baseline_s=BASELINE_S)
    assert premature_beat_summary(lost, 1.0)["n_premature"] < 7

    removed = removed_ectopic_samples(report, corrected)
    assert len(removed) == report["n_ectopic"]
    events = classify_arrhythmias(corrected, FS, baseline_s=BASELINE_S, removed_beats=removed)
    got = [(e.kind, e.n_beats) for e in _premature(events)]
    assert got == [(e.kind, e.n_beats) for e in _premature(ref)]
    assert sum(e.n_removed for e in events) == report["n_ectopic"]
    assert any("removed by artifact correction" in e.label for e in events)
    # the merged coupling + compensatory interval is not a pause
    assert not any(e.kind == "pause" for e in events)


def test_removed_beats_outside_the_analysed_span_are_ignored():
    rp = rpeaks_from_rr([120.0] * 40, FS)
    events = classify_arrhythmias(rp, FS, baseline_s=BASELINE_S,
                                  removed_beats=[int(rp[0]) - 50, int(rp[-1]) + 50])
    assert _premature(events) == []


def test_removed_ectopic_samples_uses_type_and_activity():
    rp = np.array([0, 100, 200, 300, 400])
    report = {"removed_samples": [150, 250, 350], "removed_types": ["ectopic", "duplicate", "ectopic"]}
    assert removed_ectopic_samples(report, rp).tolist() == [150, 350]
    # a removed beat that is back in the peaks is no longer removed
    assert removed_ectopic_samples(report, np.r_[rp, 350]).tolist() == [150]
    # older reports name no type: nothing
    assert len(removed_ectopic_samples({"removed_samples": [150]}, rp)) == 0
    assert len(removed_ectopic_samples(None, rp)) == 0


def test_summary_counts_and_hourly_rates():
    rp = rpeaks_from_rr(_rr_mixture(), FS)
    events = classify_arrhythmias(rp, FS, baseline_s=BASELINE_S)
    s = premature_beat_summary(events, 1800.0)        # half an hour
    assert (s["n_isolated"], s["n_couplets"], s["n_runs"], s["n_beats_in_runs"]) == (1, 1, 1, 4)
    assert s["n_premature"] == 1 + 2 + 4
    assert s["premature_per_h"] == pytest.approx(14.0)
    assert s["isolated_per_h"] == pytest.approx(2.0)
    assert s["runs_per_h"] == pytest.approx(2.0)
    assert math.isnan(premature_beat_summary(events, 0.0)["premature_per_h"])


def test_summary_rows_disclose_definition_and_removed_beats():
    s = premature_beat_summary([], 3600.0)
    rows = dict((k.strip(), v) for k, v in premature_summary_rows(s))
    assert rows["Premature beats (RR-based)"].startswith("0 beats")
    assert rows["Definition"] == PREMATURE_DEFINITION_TEXT
    assert "supraventricular" in PREMATURE_DEFINITION_TEXT
    assert "Removed by artifact correction" not in rows
    ev = ArrhythmiaEvent("premature_beat", "x", 1.0, 1.1, 1, 800.0, 70.0, "info", n_removed=1)
    rows = dict((k.strip(), v) for k, v in premature_summary_rows(premature_beat_summary([ev], 3600.0)))
    assert rows["Removed by artifact correction"].startswith("1")
    # Singular for one (the Excel Cover read "1 beats").
    assert rows["Premature beats (RR-based)"].startswith("1 beat  (")
    assert rows["Runs ≥3"].startswith("0 runs, 0 beats")
    assert premature_summary_rows(None) == []


def test_legacy_esv_kind_is_read_as_premature_run():
    assert normalize_event_kind("esv_run") == "premature_run"
    assert normalize_event_kind("pause") == "pause"
    ev = ArrhythmiaEvent("esv_run", "Potential ESV run — 4 beats (857 bpm)",
                         1.0, 1.3, 4, 857.0, 70.0, "alert")
    assert ev.kind == "premature_run" and ev.n_premature == 4
    old = {"kind": "esv_run", "label": "Potential ESV run — 4 beats (857 bpm)",
           "t_start": 1.0, "t_end": 1.3, "n_beats": 4, "hr_mean": 857.0,
           "rr_mean": 70.0, "severity": "alert", "obsolete_key": 1}
    ev2 = ArrhythmiaEvent.from_dict(old)
    assert ev2.kind == "premature_run" and ev2.n_removed == 0
    assert "ESV" not in ev2.label and "RR-based" in ev2.label
    assert ArrhythmiaEvent.from_dict(ev2.to_dict()) == ev2


def test_kind_labels_never_say_ventricular():
    for kind in ("premature_beat", "premature_couplet", "premature_run", "esv_run"):
        label = event_kind_label(kind)
        assert "RR-based" in label
        assert "ventric" not in label.lower() and "ESV" not in label
    assert event_kind_label("irregular_run") == "Irregular run"


# ── Long runs, rate changes, very premature and legacy removals ──────────

def _run_rr(n_short: int, short: float = 75.0) -> "list[float]":
    return [120.0] * 30 + [short] * n_short + [160.0] + [120.0] * 30


@pytest.mark.parametrize("length", [3, 5, 6, 10, 20, 30])
def test_runs_of_any_length_up_to_the_cap_are_reported(length):
    # The old centred 10-neighbour median was pulled down by the run's own
    # intervals: runs of 6+ beats were never reported.
    events = classify_arrhythmias(rpeaks_from_rr(_run_rr(length), FS), FS, baseline_s=BASELINE_S)
    s = premature_beat_summary(events, 1.0)
    assert (s["n_runs"], s["n_beats_in_runs"], s["n_premature"]) == (1, length, length)


def test_a_stretch_longer_than_the_cap_is_a_rate_change_not_a_run():
    events = classify_arrhythmias(rpeaks_from_rr(_run_rr(40), FS), FS, baseline_s=BASELINE_S)
    assert _premature(events) == []
    # ... not even its tail reported as a 30-beat run
    events = classify_arrhythmias(rpeaks_from_rr(_run_rr(31), FS), FS, baseline_s=BASELINE_S)
    assert _premature(events) == []


def test_a_sustained_abrupt_rate_step_is_not_premature():
    rr = [120.0] * 30 + [60.0] * 60
    events = classify_arrhythmias(rpeaks_from_rr(rr, FS), FS, baseline_s=BASELINE_S)
    assert _premature(events) == []


def test_premature_definition_discloses_the_reference_and_cap():
    assert "before" in PREMATURE_DEFINITION_TEXT and "30 beats" in PREMATURE_DEFINITION_TEXT


def _corrected_with_info(rr):
    rp = rpeaks_from_rr(rr, FS)
    corrected, report = correct_rr_artifacts(rp, FS)
    removed, info = removed_beats_for_classification(report, corrected)
    events = classify_arrhythmias(corrected, FS, baseline_s=BASELINE_S,
                                  removed_beats=removed if len(removed) else None)
    return report, removed, info, events


def test_very_premature_beat_removed_as_duplicate_is_still_counted():
    # coupling 60 ms < Min R-R (~67 ms): correction removes it as a
    # duplicate / non-physiological peak, not as ectopic
    report, removed, info, events = _corrected_with_info(
        [120.0] * 30 + [60.0, 180.0] + [120.0] * 30)
    assert "ectopic" not in report["removed_types"]
    assert len(removed) == 1 and info["n_spurious"] == 0
    s = premature_beat_summary(events, 1.0, info)
    assert (s["n_isolated"], s["n_removed"], s["n_removed_spurious"]) == (1, 1, 0)


def test_spurious_double_detection_is_not_put_back_and_is_disclosed():
    # a peak 10 ms after its R: removing it gives back the normal interval
    rp = rpeaks_from_rr([120.0] * 60, FS)
    extra = int(rp[30]) + 10
    corrected, report = correct_rr_artifacts(np.sort(np.r_[rp, extra]), FS)
    assert report["removed_samples"] == [extra]
    removed, info = removed_beats_for_classification(report, corrected)
    assert len(removed) == 0 and info["n_spurious"] == 1
    events = classify_arrhythmias(corrected, FS, baseline_s=BASELINE_S)
    s = premature_beat_summary(events, 3600.0, info)
    assert s["n_premature"] == 0 and s["n_removed_spurious"] == 1
    rows = dict((k.strip(), v) for k, v in premature_summary_rows(s))
    assert rows["Removed as spurious detections"].startswith("1 ")


def test_spurious_peak_next_to_a_pause_is_not_put_back():
    rr = [120.0] * 30 + [300.0] + [120.0] * 30
    rp = rpeaks_from_rr(rr, FS)
    extra = int(rp[30]) + 10            # 10 ms after the beat before the pause
    report = {"removed_samples": [extra], "removed_types": ["duplicate"]}
    removed, info = removed_beats_for_classification(report, rp)
    assert len(removed) == 0 and info["n_spurious"] == 1


def test_legacy_report_without_types_is_reexamined_and_disclosed():
    rp = rpeaks_from_rr(_rr_mixture(), FS)
    corrected, report = correct_rr_artifacts(rp, FS)
    legacy = {k: v for k, v in report.items() if k not in ("removed_types", "break_samples")}
    removed, info = removed_beats_for_classification(legacy, corrected)
    assert info["n_unknown_type"] == len(report["removed_samples"])
    assert info["n_unknown_put_back"] == len(removed) > 0
    events = classify_arrhythmias(corrected, FS, baseline_s=BASELINE_S, removed_beats=removed)
    ref = classify_arrhythmias(rp, FS, baseline_s=BASELINE_S)
    assert ([(e.kind, e.n_beats) for e in _premature(events)]
            == [(e.kind, e.n_beats) for e in _premature(ref)])
    s = premature_beat_summary(events, 3600.0, info)
    rows = dict((k.strip(), v) for k, v in premature_summary_rows(s))
    assert "re-run artifact correction" in rows["Removed peaks of unknown type"]


def test_typed_report_matches_removed_ectopic_samples_for_ectopic_removals():
    rp = rpeaks_from_rr(_rr_mixture(), FS)
    corrected, report = correct_rr_artifacts(rp, FS)
    removed, info = removed_beats_for_classification(report, corrected)
    assert removed.tolist() == removed_ectopic_samples(report, corrected).tolist()
    assert info["n_unknown_type"] == 0
    assert removed_beats_for_classification(None, corrected)[0].size == 0


def test_summary_records_the_analysed_window():
    s = premature_beat_summary([], 60.0, t_start=12.0, t_end=72.0)
    assert (s["window_start_s"], s["window_end_s"]) == (12.0, 72.0)
    rows = dict((k.strip(), v) for k, v in premature_summary_rows(s))
    assert rows["Analysed window"] == "12.0–72.0 s"
    assert math.isnan(premature_beat_summary([], 60.0)["window_start_s"])
