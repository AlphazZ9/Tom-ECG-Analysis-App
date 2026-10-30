"""
The wave template is per recording and explicit (audit point 6):

* delineation never modifies or saves the template it is given, so a
  recording's intervals do not depend on the recordings analysed before it;
* nothing is written to the user profile except an explicit save;
* the template used is recorded on the result (``attrs``) and disclosed;
* provenance (built-in default / saved default / file / calibrated, refined
  or edited on this recording) survives copies, files and JSON;
* a saved default written silently by the old auto-update is ignored.

Every test redirects WAVE_TEMPLATE_PATH to tmp_path: nothing touches the
real home profile.
"""
import json

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis, wave_template
from ecg.core.wave_template import TEMPLATE_SOURCES, WaveTemplate
from ecg.io.loaders import _deserialise_results, _serialise_results

FS = 2000.0

_P   = [(-120.0, 0.0), (-60.0, 0.0), (-52.0, 0.1), (-44.0, 0.0)]
_REC_A = _P + [(-8.0, 0.0), (-5.0, -0.2), (0.0, 1.0), (4.0, -0.3), (8.0, 0.0),
               (12.0, 0.1), (18.0, 0.0), (25.0, 0.0), (35.0, 0.2), (55.0, 0.0),
               (120.0, 0.0)]
_REC_B = _P + [(-16.0, 0.0), (-10.0, -0.2), (0.0, 1.0), (8.0, -0.3), (14.0, 0.0),
               (18.0, 0.1), (24.0, 0.0), (30.0, 0.0), (40.0, 0.2), (65.0, 0.0),
               (120.0, 0.0)]


@pytest.fixture(autouse=True)
def _profile(tmp_path, monkeypatch):
    """The 'user profile' template files live in tmp_path for every test."""
    path = tmp_path / "profile" / ".ecg_wave_template.json"
    path.parent.mkdir()
    monkeypatch.setattr(wave_template, "WAVE_TEMPLATE_PATH", path)
    monkeypatch.setattr(wave_template, "_WAVE_TEMPLATE_PATH_LEGACY",
                        path.with_suffix(".pkl"))
    return path


def _train(pts, n=30, rr=200.0):
    rp = [int(0.3 * FS)]
    for _ in range(n):
        rp.append(rp[-1] + int(round(rr / 1000.0 * FS)))
    rp = np.array(rp)
    total = rp[-1] + int(0.3 * FS)
    t_ms = np.arange(total) / FS * 1000.0
    sig = np.zeros(total)
    xs, ys = zip(*pts)
    for r in rp:
        lo, hi = r - int(0.12 * FS), r + int(0.12 * FS)
        sig[lo:hi] += np.interp(t_ms[lo:hi] - t_ms[r], xs, ys)
    sig += np.random.default_rng(3).normal(0.0, 0.002, total)
    return sig, rp


def _delineate(pts, wt):
    sig, rp = _train(pts)
    return analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                      wave_template=wt)


# ── reproducibility ───────────────────────────────────────────────────────

def test_recording_b_after_a_equals_b_alone(_profile):
    shared = WaveTemplate()                 # unconfirmed: old code auto-updated it
    before = shared.to_dict()
    _delineate(_REC_A, shared)
    after_a = _delineate(_REC_B, shared)
    alone   = _delineate(_REC_B, WaveTemplate())
    assert shared.to_dict() == before
    cols = ["PR_ms", "QRS_ms", "QT_ms", "QTc_ms", "T_offset_ms", "P_onset_ms"]
    pd.testing.assert_frame_equal(after_a[cols], alone[cols])
    assert not _profile.exists()            # nothing written to the profile


def test_delineation_never_writes_the_profile(_profile, monkeypatch):
    def _refuse(self, path=None):
        raise AssertionError("delineation must not save a template")
    monkeypatch.setattr(WaveTemplate, "save", _refuse)
    for wt in (WaveTemplate(), None):
        _delineate(_REC_A, wt)
    assert not _profile.exists()


def test_no_template_means_built_in_defaults_not_the_saved_file(_profile):
    narrow = WaveTemplate()
    narrow.landmarks["T_offset"] = (40.0, 2.0)
    narrow.source = "user-confirmed"
    narrow.save()
    df = _delineate(_REC_A, None)
    assert df.attrs["wave_template"] == WaveTemplate().to_dict()


def test_explicit_save_is_the_only_write(_profile):
    wt = WaveTemplate()
    wt.landmarks["P_peak"] = (-50.0, 20.0)
    wt.source = "edited"
    _delineate(_REC_A, wt)
    assert not _profile.exists()
    wt.save()
    assert _profile.exists()
    loaded = WaveTemplate.load_default()
    assert loaded.landmarks["P_peak"] == (-50.0, 20.0)
    assert loaded.source == "saved-default"
    assert loaded.origin == "originally edited on another recording"
    assert loaded.provenance().startswith("saved default template (saved ")


# ── provenance and disclosure ─────────────────────────────────────────────

def test_result_records_and_discloses_the_template_used():
    wt = WaveTemplate()
    wt.source = "auto-calibrated"
    df = _delineate(_REC_A, wt)
    assert df.attrs["wave_template"] == wt.to_dict()
    rows = dict(analysis.interval_disclosure_rows(df))
    assert rows["Wave template"].startswith(TEMPLATE_SOURCES["auto-calibrated"])
    assert "P -45±28" in rows["Wave template"]


def test_template_survives_the_session_json_round_trip():
    wt = WaveTemplate()
    wt.source, wt.origin = "file", "cohort_A.json"
    df = _delineate(_REC_A, wt)
    raw = json.loads(json.dumps(_serialise_results({"intervals": df})))
    back = _deserialise_results(raw)["intervals"]
    assert WaveTemplate.from_dict(back.attrs["wave_template"]).provenance() == \
        "loaded from cohort_A.json"
    # QTc recomputation (formula change, session restore) keeps it.
    again = analysis.recompute_qtc_column(back, "bazett")
    assert again.attrs["wave_template"] == back.attrs["wave_template"]


def test_frame_without_a_recorded_template_says_so():
    df = pd.DataFrame({"QT_ms": [50.0, 52.0, 51.0, 49.0], "QRS_onset_ms": [-8.0] * 4})
    assert dict(analysis.interval_disclosure_rows(df))["Wave template"].startswith(
        "not recorded")


@pytest.mark.parametrize("source, origin, expected", [
    ("default", "", "built-in mouse defaults"),
    ("auto-calibrated", "", "auto-calibrated on this recording's mean beat"),
    ("refined", "", "refined from this recording's delineation"),
    ("file", "x.json", "loaded from x.json"),
    ("edited", "built-in mouse defaults",
     "edited on this recording (based on: built-in mouse defaults)"),
    ("auto-updated", "", "auto-updated from an earlier recording (older version)"),
])
def test_provenance_labels(source, origin, expected):
    wt = WaveTemplate()
    wt.source, wt.origin = source, origin
    assert wt.provenance() == expected


def test_copy_and_dict_round_trip_are_independent():
    wt = WaveTemplate()
    wt.source, wt.origin, wt.confirmed = "edited", "built-in mouse defaults", True
    cp = wt.copy()
    assert cp.to_dict() == wt.to_dict()
    cp.landmarks["P_peak"] = (0.0, 1.0)
    assert wt.landmarks["P_peak"] == WaveTemplate.DEFAULTS["P_peak"]
    assert WaveTemplate.from_dict(None).to_dict() == WaveTemplate().to_dict()


# ── saved files ───────────────────────────────────────────────────────────

def test_load_default_without_a_saved_file_is_the_built_in_default():
    wt = WaveTemplate.load_default()
    assert wt.source == "default" and wt.landmarks == WaveTemplate.DEFAULTS


def test_load_default_ignores_a_file_written_by_the_old_auto_update(_profile):
    _profile.write_text(json.dumps({
        "landmarks": {"T_offset": [40.0, 3.0]}, "confirmed": False,
        "source": "auto-updated", "created_at": "2025-01-01T00:00:00"}),
        encoding="utf-8")
    wt = WaveTemplate.load_default()
    assert wt.source == "default" and wt.landmarks == WaveTemplate.DEFAULTS
    assert _profile.exists()                 # left in place, not deleted


def test_load_named_file_tags_its_provenance(tmp_path):
    src = WaveTemplate()
    src.landmarks["T_peak"] = (48.0, 20.0)
    path = tmp_path / "strain_B6.json"
    src.save(path)
    wt = WaveTemplate.load(path)
    assert wt.source == "file" and wt.origin == "strain_B6.json"
    assert wt.landmarks["T_peak"] == (48.0, 20.0)
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        WaveTemplate.load(bad)


# ── explicit refinement (replaces the silent auto-update) ─────────────────

def test_refine_returns_a_new_template_and_leaves_the_input_alone():
    wt = WaveTemplate()
    df = _delineate(_REC_A, wt)
    refined = analysis.refine_template_from_intervals(df, wt)
    assert refined is not None and refined is not wt
    assert refined.source == "refined"
    assert wt.landmarks == WaveTemplate.DEFAULTS and wt.source == "default"
    med = float(df["T_offset_ms"].median())
    assert refined.landmarks["T_offset"][0] == pytest.approx(med, abs=0.1)
    # Detection does not use the QRS onset / J point windows: not refined.
    assert refined.landmarks["QRS_onset"] == WaveTemplate.DEFAULTS["QRS_onset"]
    assert analysis.refine_template_from_intervals(df.iloc[:5], wt) is None


# ── exports ───────────────────────────────────────────────────────────────

def test_excel_cover_names_the_template_used():
    from ecg.io.export import ExcelExporter
    wt = WaveTemplate()
    wt.source, wt.origin = "file", "cohort_A.json"
    df = _delineate(_REC_A, wt)
    wb = ExcelExporter.build_workbook(
        results={"hr": {"n": 3, "mean": 500.0, "min": 480.0, "max": 520.0,
                        "std": 5.0}, "intervals": df},
        signal_flt=None, signal_raw=None, time=np.arange(10) / 1000.0,
        rpeaks_ok=None, fs=1000, filepath=None, subject="m1",
        sig_quality=None, epoch_df=None, qtc_formula="mitchell")
    ws = wb["Cover"]
    cover = {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
             for r in range(1, ws.max_row + 1)}
    assert cover["Wave template"].startswith("loaded from cohort_A.json")


# ── saved default: checked before use (older / implausible / invalid) ─────

# Shape of a real pre-V5 profile file: "user-confirmed", no provenance
# fields, windows narrowed by the old auto-update (P right on the QRS).
_LEGACY_NARROWED = {
    "landmarks": {"P_onset": [-9.3, 4.0], "P_peak": [-7.5, 2.0],
                  "Q_peak": [-5.1, 2.0], "S_peak": [3.9, 2.0],
                  "J_peak": [7.1, 2.1], "T_peak": [19.4, 4.3],
                  "T_offset": [27.4, 5.0]},
    "confirmed": True, "source": "user-confirmed",
    "created_at": "2026-06-22T17:34:46.071089"}


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_legacy_narrowed_default_is_not_used_and_is_reported(_profile):
    _write(_profile, _LEGACY_NARROWED)
    wt, notice = WaveTemplate.load_default_checked()
    assert wt.source == "default" and wt.landmarks == WaveTemplate.DEFAULTS
    assert str(_profile) in notice and "older version" in notice
    assert "P -8±2" in notice and "T_offset +27±5" in notice   # its windows
    assert _profile.exists()                                    # left in place
    # PR is measured as with the built-in defaults, not lost on every beat.
    df = _delineate(_REC_A, wt)
    assert df["PR_ms"].notna().sum() == len(df)
    narrowed = _delineate(_REC_A, WaveTemplate.from_dict(_LEGACY_NARROWED))
    assert narrowed["PR_ms"].notna().sum() == 0


def test_implausible_v5_default_is_not_used(_profile):
    _write(_profile, {**_LEGACY_NARROWED, "source": "edited", "origin": ""})
    wt, notice = WaveTemplate.load_default_checked()
    assert wt.source == "default"
    assert "implausible" in notice and "before Q" in notice


def test_plausible_saved_default_is_used_without_notice(_profile):
    saved = WaveTemplate()
    saved.landmarks["T_offset"] = (75.0, 18.0)
    saved.source = "edited"
    saved.save(recording="m1.mat")
    wt, notice = WaveTemplate.load_default_checked()
    assert notice is None and wt.source == "saved-default"
    assert wt.landmarks["T_offset"] == (75.0, 18.0)


def test_built_in_and_calibrated_templates_are_plausible():
    from ecg.core.wave_template import _mouse_demo_beat
    assert WaveTemplate().plausibility_problems() == []
    bt, mb = _mouse_demo_beat()
    wt = WaveTemplate()
    wt.calibrate_from_beat(bt, mb)
    assert wt.plausibility_problems() == []
    bad = WaveTemplate()
    bad.landmarks["T_offset"] = (12.0, 5.0)        # T end before T, at the J point
    probs = bad.plausibility_problems()
    assert any("T_peak" in p for p in probs) and any("J point" in p for p in probs)


@pytest.mark.parametrize("entry, word", [
    ([float("nan"), 5.0], "non-finite"),
    ([82.0, -22.0], "<= 0"),
    ([82.0, 0.0], "<= 0"),
])
def test_invalid_windows_are_refused(tmp_path, _profile, entry, word):
    data = {"landmarks": {"T_offset": entry}, "source": "edited", "origin": ""}
    path = tmp_path / "bad_windows.json"
    path.write_text(json.dumps(data), encoding="utf-8")   # NaN -> JSON NaN
    with pytest.raises(ValueError, match=word):
        WaveTemplate.load(path)
    # Never used as a window (sessions / attrs): the default is kept.
    assert WaveTemplate.from_dict(data).landmarks["T_offset"] == \
        WaveTemplate.DEFAULTS["T_offset"]
    _profile.write_text(json.dumps(data), encoding="utf-8")
    wt, notice = WaveTemplate.load_default_checked()
    assert wt.source == "default" and word in notice


# ── saved default: provenance worded for another recording ────────────────

@pytest.mark.parametrize("source, origin, recording, expected", [
    ("auto-calibrated", "", "A.mat",
     "originally auto-calibrated on the mean beat of A.mat"),
    ("refined", "", "A.mat", "originally refined from the delineation of A.mat"),
    ("edited", "auto-calibrated on this recording's mean beat", "A.mat",
     "originally edited on A.mat (based on: auto-calibrated on that "
     "recording's mean beat)"),
    ("file", "lab.json", "A.mat", "originally loaded from lab.json"),
    ("auto-calibrated", "", "",
     "originally auto-calibrated on the mean beat of another recording"),
])
def test_saved_default_provenance_names_its_origin(_profile, source, origin,
                                                    recording, expected):
    wt = WaveTemplate()
    wt.source, wt.origin = source, origin
    wt.save(recording=recording)
    loaded = WaveTemplate.load_default()
    assert loaded.origin == expected
    prov = loaded.provenance()
    assert prov.startswith("saved default template (saved ") and prov.endswith(expected)
    assert "this recording" not in prov
    # Same text in the exports' "Wave template" row.
    df = _delineate(_REC_A, loaded)
    assert expected in analysis.wave_template_disclosure(df)


def test_saved_default_with_a_bare_source_key_is_worded_neutrally():
    wt = WaveTemplate()
    wt.source, wt.origin = "saved-default", "auto-calibrated"   # earlier V5 session
    assert wt.provenance() == ("saved default template, originally "
                               "auto-calibrated on the mean beat of another recording")


def test_disclosure_lists_the_default_windows_of_a_partial_template():
    wt = WaveTemplate()
    wt.landmarks = {"P_peak": (-40.0, 20.0)}
    wt.source, wt.origin = "file", "partial.json"
    text = analysis.wave_template_disclosure(_delineate(_REC_A, wt))
    assert "P -40±20" in text
    for cell in ("P_onset -62±20", "Q -5±12", "S +5±12", "J +15±8",
                 "T +52±26", "T_offset +82±22"):
        assert cell in text
