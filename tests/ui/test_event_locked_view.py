"""
HRV > Autonomic > Event-locked helpers (ecg.ui.event_locked_view) -- no Tk
window: panel parameters, events from the annotations, the per-event
window means / traces of an Autonomic result (CSI family validity, HR and
LF / HF judged on their own), the table / text / status, the cohort
registry rows, and the ks / kp calibration on registry rows (exact
re-composition of CSI / CPI for other gains, per-subject aggregation,
configuration filter, Z(k) and its knee, the tests at the current gains).
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ecg.core.autonomic import compute_autonomic_indices, with_coefficients
from ecg.core.event_locked import calibrate_gain, knee_point, paired_tests
from ecg.io import db
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import event_locked_view as ev

from tests.helpers import rpeaks_from_rr

FS = 2000.0
N_BEATS = 2400
STEP_BEAT = 1300          # RR shortens by 8 ms from this beat on (an "injection")


def _peaks(seed: int = 3, step_ms: float = -8.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.full(N_BEATS, 0.12))
    rr = (120 + 3 * np.sin(2 * np.pi * 0.8 * t) + 1.5 * np.sin(2 * np.pi * 2.5 * t)
          + rng.normal(0, 1.0, N_BEATS))
    rr[STEP_BEAT:] += step_ms
    return rpeaks_from_rr(list(rr), FS)


@pytest.fixture(scope="module")
def recording():
    pk = _peaks()
    res = compute_autonomic_indices(pk, FS)
    t_event = float(pk[STEP_BEAT]) / FS          # the last long interval ends here
    return res, t_event


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A throw-away registry under tmp_path (never the user's profile)."""
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(db, "SESSION_DIR", session_dir)
    monkeypatch.setattr(db, "_DB_PATH", session_dir / "ecg_registry.db")
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_schema_retry_at", 0.0)
    return session_dir / "ecg_registry.db"


def _anns(t_event):
    return [
        {"t_start": t_event, "t_end": t_event + 20, "label": "dobutamine", "color": "#a0a"},
        {"t_start": 80.0, "t_end": 90.0, "label": "saline", "color": "#0a0"},
        {"t_start": 2.0, "t_end": 3.0, "label": "", "color": "#00a"},
    ]


def _fake_spwvd(res, offset=0.013, gap=None):
    t = np.arange(float(res.t[0]) - 1.0 + offset, float(res.t[-1]) + 1.0, 0.05)
    lf = 100 + 10 * np.sin(t / 7.0)
    hf = 50 + 5 * np.cos(t / 3.0)
    if gap is not None:
        lf = lf.copy()
        hf = hf.copy()
        m = (t >= gap[0]) & (t <= gap[1])
        lf[m] = np.nan
        hf[m] = np.nan
    return {"t": t, "lf": lf, "hf": hf,
            "params": {"lf_band": (0.4, 1.5), "hf_band": (1.5, 5.0), "fs_resample": 20.0,
                       "fbins": 4096, "n_gaps_masked": 0, "nan_gap_s": 2.0}}


# ── parameters ────────────────────────────────────────────────────────────

def test_defaults_are_the_papers_30_s_windows():
    assert ev.default_event_params() == {"label": "All", "pre_s": 30.0, "post_s": 30.0,
                                         "zscore": True}


def test_parse_accepts_a_decimal_comma_and_rejects_bad_windows():
    p = ev.parse_event_inputs(" dobutamine ", "20,5", "30", 0)
    assert p == {"label": "dobutamine", "pre_s": 20.5, "post_s": 30.0, "zscore": False}
    for pre, post, needle in (("abc", "30", "Pre"), ("30", "0", "Post"),
                              ("700", "30", "Pre"), ("30", "nan", "Post")):
        with pytest.raises(ValueError, match=needle):
            ev.parse_event_inputs("All", pre, post)
    # a blank label means every annotation; an unreadable z flag means on
    assert ev.parse_event_inputs("", "30", "30", "maybe")["label"] == "All"
    assert ev.parse_event_inputs("", "30", "30", "maybe")["zscore"] is True


def test_normalise_tolerates_old_and_corrupt_sessions_and_json():
    base = {"label": "saline", "pre_s": 20.0, "post_s": 25.0, "zscore": False}
    assert ev.normalise_event_params(None, base) == base
    assert ev.normalise_event_params("junk", base) == base
    got = ev.normalise_event_params({"pre_s": "x", "post_s": 5000, "zscore": "true",
                                     "label": 3}, base)
    assert got == {"label": "saline", "pre_s": 20.0, "post_s": 25.0, "zscore": True}
    text = json.dumps({"event_locked_params": base}, cls=_SessionEncoder)
    back = json.loads(text, object_hook=_session_decode_hook)["event_locked_params"]
    assert ev.normalise_event_params(back) == base


# ── events from annotations ───────────────────────────────────────────────

def test_label_options_and_event_selection():
    anns = [{"t_start": 50.0, "label": "Saline"}, {"t_start": 10.0, "label": "dobu"},
            {"t_start": 10.0, "label": "dobu"},                 # duplicate
            {"t_start": 30.0, "label": ""}, {"t_start": float("nan"), "label": "x"},
            {"label": "no start"}, "junk", {"t_start": "7", "label": " dobu "}]
    assert ev.annotation_label_options(anns) == ["All", "dobu", "Saline", "(unlabelled)"]
    assert ev.select_events(anns) == [(7.0, "dobu"), (10.0, "dobu"), (30.0, ""),
                                      (50.0, "Saline")]
    assert ev.select_events(anns, "dobu") == [(7.0, "dobu"), (10.0, "dobu")]
    assert ev.select_events(anns, "(unlabelled)") == [(30.0, "")]
    assert ev.select_events(anns, "missing") == []
    assert ev.annotation_label_options([]) == ["All"]
    assert (ev.events_signature([(10.0000000001, "a")])
            == ev.events_signature([(10.0, "a")]))


# ── event-locked computation ──────────────────────────────────────────────

def test_window_means_validity_and_the_drug_response(recording):
    res, t_event = recording
    events = ev.select_events(_anns(t_event))
    el = ev.compute_event_locked(res, events, 30, 30)
    t = el.table
    assert list(t["label"]) == ["", "saline", "dobutamine"]
    assert list(t["valid"]) == [False, True, True]              # 2 s: pre window outside
    assert el.keys == ("hr", "d", "sd1", "sd2", "csi", "cpi")
    assert el.n_events == 3 and el.n_valid == 2 and not el.has_lfhf
    drug = t.iloc[2]
    assert drug["hr_delta"] > 20                                # 120 -> 112 ms
    assert drug["d_delta"] < -5 and drug["csi_delta"] > 0 and drug["cpi_delta"] < 0
    # invalid series are blank, never a partial mean
    assert all(math.isnan(t.iloc[0][f"{k}_pre"]) for k in el.keys)
    # the pre mean is the plain mean of the resampled samples in [o - 30, o)
    o = float(drug["onset_s"])
    m = (res.t >= o - 30 - 1e-9) & (res.t < o - 1e-9)
    assert drug["csi_pre"] == pytest.approx(np.nanmean(res.csi[m]), rel=1e-12)


def test_deltas_are_linear_in_the_gains(recording):
    res, t_event = recording
    el = ev.compute_event_locked(res, ev.select_events(_anns(t_event)))
    t = el.table[el.table["valid"]]
    np.testing.assert_allclose(t["csi_delta"], -t["d_delta"] + 3.0 * t["sd2_delta"],
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(t["cpi_delta"], t["d_delta"] + 12.0 * t["sd1_delta"],
                               rtol=1e-9, atol=1e-9)


def test_lf_hf_on_their_own_axis_and_validity(recording):
    res, t_event = recording
    sp = _fake_spwvd(res, gap=(70.0, 85.0))           # LF / HF blank around the saline event
    el = ev.compute_event_locked(res, ev.select_events(_anns(t_event)), spwvd=sp)
    t = el.table
    assert el.has_lfhf and el.keys[-2:] == ("lf", "hf")
    assert bool(t.loc[1, "valid"]) and not bool(t.loc[1, "lf_valid"])   # CSI kept
    assert math.isnan(t.loc[1, "lf_pre"]) and not math.isnan(t.loc[1, "csi_pre"])
    o = float(t.loc[2, "onset_s"])
    m = (sp["t"] >= o) & (sp["t"] <= o + 30 + 1e-9)
    assert t.loc[2, "lf_post"] == pytest.approx(np.mean(sp["lf"][m]), rel=1e-12)
    assert bool(t.loc[2, "any_valid"])
    assert el.params["lf_band"] == [0.4, 1.5]
    # an {"error": ...} placeholder adds nothing
    el2 = ev.compute_event_locked(res, ev.select_events(_anns(t_event)),
                                  spwvd={"error": "too short"})
    assert not el2.has_lfhf


def test_traces_share_one_relative_grid_raw_and_zscored(recording):
    res, t_event = recording
    el = ev.compute_event_locked(res, [(t_event, "dobutamine")], 30, 20,
                                 spwvd=_fake_spwvd(res))
    assert el.t_rel[0] == -30.0 and el.t_rel[-1] == 20.0 and len(el.t_rel) == 1001
    for k in el.keys:
        assert el.traces[k].shape == el.traces_z[k].shape == (1, 1001)
    # the trace at 0 is the series at the onset; z-scoring uses the whole series
    i0 = int(np.argmin(np.abs(el.t_rel)))
    y = res.csi
    expect = (np.interp(t_event, res.t, y) - np.nanmean(y)) / np.nanstd(y, ddof=1)
    assert el.traces_z["csi"][0, i0] == pytest.approx(expect, rel=1e-6)
    med, mad, n, rows = ev.group_stats(el, "csi", zscore=False)
    assert n == 1 and np.allclose(med, el.traces["csi"][0], equal_nan=True)
    assert np.allclose(mad[np.isfinite(mad)], 0.0)


def test_no_event_or_no_series_is_an_error(recording):
    res, _ = recording
    with pytest.raises(ValueError, match="No event"):
        ev.compute_event_locked(res, [])
    empty = compute_autonomic_indices(rpeaks_from_rr([120.0] * 10, FS), FS)
    with pytest.raises(ValueError, match="no CSI / CPI"):
        ev.compute_event_locked(empty, [(1.0, "x")])


def test_table_frame_text_and_status(recording):
    res, t_event = recording
    el = ev.compute_event_locked(res, ev.select_events(_anns(t_event)))
    df = ev.event_table_frame(el)
    assert list(df.columns[:8]) == ["label", "onset_s", "valid", "isolated", "prev_gap_s",
                                    "next_gap_s", "coverage_pre", "coverage_post"]
    assert "csi_delta" in df.columns and "csi_valid" not in df.columns
    assert df["valid"].dtype == bool and df["isolated"].dtype == bool and len(df) == 3
    assert df["isolated"].all()                     # 2 s, 80 s, 156 s: far apart
    text = ev.event_table_text(el)
    for needle in ("Event 3", "dobutamine", "INVALID", "WITHIN-RECORDING WILCOXON",
                   "not population inference", "rat SHR calibration", "display only",
                   "Events of several labels"):
        assert needle in text
    assert "no other annotation starts inside" in ev.VALIDITY_NOTE
    msg, level = ev.event_status(el)
    assert level == "warn" and "1 invalid" in msg and "labels pooled" in msg
    el_ok = ev.compute_event_locked(res, [(t_event, "d")])
    assert ev.event_status(el_ok)[1] == "ok"
    el_none = ev.compute_event_locked(res, [(1.0, "d")])
    assert ev.event_status(el_none)[1] == "withheld"
    tests = ev.within_recording_tests(el)
    assert list(tests["key"]) == list(el.keys) and set(tests["n"]) == {2}


def test_figure_panels_and_display_units(recording):
    res, t_event = recording
    el = ev.compute_event_locked(res, ev.select_events(_anns(t_event)))
    fig = Figure()
    ev.draw_event_locked(fig, el, zscore=True)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_ylabel() == "z-score"
    assert "n = 2" in fig.axes[0].get_legend_handles_labels()[1][0]
    assert any(list(map(float, l.get_xdata())) == [0.0, 0.0]
               for l in fig.axes[0].lines)                       # the onset line
    fig2 = Figure()
    ev.draw_event_locked(fig2, ev.compute_event_locked(
        res, ev.select_events(_anns(t_event)), spwvd=_fake_spwvd(res)), zscore=False)
    assert len(fig2.axes) == 4
    assert fig2.axes[0].get_ylabel() == "CSI (a.u.)" and fig2.axes[2].get_ylabel() == "HR (bpm)"
    fig3 = Figure()
    ev.draw_event_locked(fig3, ev.compute_event_locked(res, [(1.0, "x")]))
    assert any("No valid event" in t.get_text() for t in fig3.axes[0].texts)


# ── registry rows ─────────────────────────────────────────────────────────

def test_registry_rows_round_trip_through_the_database(recording, registry):
    res, t_event = recording
    el = ev.compute_event_locked(res, ev.select_events(_anns(t_event)))
    params = ev.registry_params(el)
    assert params["method"] == "robust" and params["window_s"] == 3.0
    assert params["ks"] == 3.0 and params["kp"] == 12.0 and params["pre_s"] == 30.0
    n = db.save_event_rows("C:/data/m1.mat", ev.registry_rows(el), subject="M1",
                           project_name="P", params=params, replace=True)
    assert n == 3
    stored = db.list_event_rows(valid_only=False)
    assert [r["valid"] for r in stored] == [False, True, True]
    frame = ev.registry_frame(db.list_event_rows())            # valid only
    frame = frame.sort_values("onset_s").reset_index(drop=True)
    mem = ev.registry_frame(ev.export_rows(el, "C:/data/m1.mat", "M1", "P"))
    mem = mem[mem["valid"]].sort_values("onset_s").reset_index(drop=True)
    for k in el.keys:
        for s in ("_pre", "_post", "_delta"):
            np.testing.assert_allclose(frame[f"{k}{s}"], mem[f"{k}{s}"], rtol=1e-12)
    assert list(frame["subject_key"]) == ["M1", "M1"]
    assert set(frame["ks"]) == {3.0} and set(frame["kp"]) == {12.0}


def test_export_rows_write_the_registry_csv_format(recording, tmp_path):
    res, t_event = recording
    el = ev.compute_event_locked(res, ev.select_events(_anns(t_event)))
    path = tmp_path / "events.csv"
    n = db.export_event_rows_csv(path, ev.export_rows(el, "C:/data/m1.mat", "M1", "P"))
    assert n == 3
    head = path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert head[:6] == ["id", "filepath", "stem", "subject", "project_name", "label"]
    assert {"csi_pre", "csi_post", "csi_delta", "csi_valid", "param_ks"} <= set(head)


# ── calibration (registry rows) ───────────────────────────────────────────

def test_recomposed_gains_equal_a_fresh_computation(recording):
    res, t_event = recording
    events = ev.select_events(_anns(t_event))
    el = ev.compute_event_locked(res, events)
    el2 = ev.compute_event_locked(with_coefficients(res, 5.0, 7.0), events)
    frame = ev.registry_frame(ev.export_rows(el, "C:/x/a.mat", "A", "P"))
    rec = ev.recompose_gains(frame, 5.0, 7.0)
    ok = el2.table["valid"].to_numpy()
    for name in ("csi", "cpi"):
        for s in ("_pre", "_post", "_delta"):
            np.testing.assert_allclose(rec[f"{name}{s}"].to_numpy()[ok],
                                       el2.table[f"{name}{s}"].to_numpy()[ok], rtol=1e-9)
    # without the stored ks, CSI cannot be re-composed (D̃ is not stored)
    frame.loc[:, "ks"] = np.nan
    assert ev.recompose_gains(frame, 5.0, 7.0)["csi_pre"].isna().all()


def _row(subject, d, s1, s2, *, label="dobu", method="robust", window=3.0,
         ks=3.0, kp=12.0, stem=None, d_pre=150.0, s1_pre=2.0, s2_pre=6.0, onset=100.0):
    """A list_event_rows()-like row with chosen deltas (CSI/CPI consistent)."""
    flip_pre = 300.0 - d_pre                                    # any D̃; deltas mirror D
    d_post, s1_post, s2_post = d_pre + d, s1_pre + s1, s2_pre + s2
    m = {"d": {"pre": d_pre, "post": d_post, "delta": d, "valid": True},
         "sd1": {"pre": s1_pre, "post": s1_post, "delta": s1, "valid": True},
         "sd2": {"pre": s2_pre, "post": s2_post, "delta": s2, "valid": True},
         "csi": {"pre": flip_pre + ks * s2_pre, "post": flip_pre - d + ks * s2_post,
                 "delta": -d + ks * s2, "valid": True},
         "cpi": {"pre": d_pre + kp * s1_pre, "post": d_post + kp * s1_post,
                 "delta": d + kp * s1, "valid": True}}
    stem = stem or f"rec_{subject or 'x'}"
    return {"id": None, "filepath": f"C:/data/{stem}.mat", "stem": stem,
            "subject": subject, "project_name": "P", "label": label, "onset_s": onset,
            "pre_s": 30.0, "post_s": 30.0, "method": method, "window_s": window,
            "valid": True, "metrics": m, "params": {"ks": ks, "kp": kp}}


def _cohort(seed=0):
    """8 animals x 2 events: D rises a little (HR falls), SD2 rises and SD1
    falls -- both Z(k) cross zero as the HRV term takes over, then plateau."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(8):
        for e in range(2):
            rows.append(_row(f"M{a}", d=2.0 + rng.normal(0, 0.6),
                             s1=-0.3 + rng.normal(0, 0.08), s2=0.5 + rng.normal(0, 0.25),
                             onset=100.0 + 200 * e))
    return rows


def test_per_subject_aggregation_counts_and_the_z_curves():
    frame = ev.registry_frame(_cohort())
    cal = ev.cohort_calibration(frame, 3.0, 12.0)              # default: per subject
    assert cal.aggregation == "subject"
    assert cal.counts == {"n_events": 16, "n_subjects": 8, "n_recordings": 8,
                          "n_units": 8, "n_no_subject": 0, "n_total": 16}
    subj = frame.groupby("subject_key")[["d_delta", "sd2_delta", "sd1_delta"]].mean()
    expect = calibrate_gain(subj["d_delta"], subj["sd2_delta"], "csi")
    np.testing.assert_allclose(cal.curves["csi"]["z"], expect["z"])
    assert list(cal.curves["cpi"]["k"]) == list(range(1, 21))
    for kind in ("csi", "cpi"):
        c = cal.curves[kind]
        assert cal.knees[kind] == knee_point(c["k"], c["z"])
    assert math.isfinite(cal.knees["csi"]) and math.isfinite(cal.knees["cpi"])
    ks, kp, text, enabled = ev.apply_plan(cal)
    assert (ks, kp) == (cal.knees["csi"], cal.knees["cpi"]) and enabled
    assert "(kept)" not in text and text.startswith("Apply ks = ")
    # per event: 16 paired samples
    cal_e = ev.cohort_calibration(frame, 3.0, 12.0, aggregation="event")
    assert cal_e.counts["n_units"] == 16 and len(cal_e.table) == 16
    with pytest.raises(ValueError):
        ev.cohort_calibration(frame, 3.0, 12.0, aggregation="median")


def test_current_gain_tests_are_the_curve_at_k_with_bonferroni_over_two():
    frame = ev.registry_frame(_cohort(1))
    cal = ev.cohort_calibration(frame, 4.0, 9.0)
    t = cal.tests.set_index("key")
    assert list(cal.tests["key"]) == ["csi", "cpi"]
    z_csi = cal.curves["csi"].set_index("k").loc[4, "z"]
    z_cpi = cal.curves["cpi"].set_index("k").loc[9, "z"]
    assert t.loc["csi", "z"] == pytest.approx(z_csi, abs=1e-9)
    assert t.loc["cpi", "z"] == pytest.approx(z_cpi, abs=1e-9)
    assert t.loc["csi", "p_bonferroni"] == pytest.approx(min(1.0, 2 * t.loc["csi", "p_value"]))
    # the per-subject recomposed CSI is what was tested
    direct = paired_tests(cal.table, keys=["csi", "cpi"])
    np.testing.assert_allclose(direct["z"], cal.tests["z"])
    text = ev.calibration_text(cal)
    assert "AT THE CURRENT GAINS" in text and "ks = 4, kp = 9" in text
    assert "inflection point" in text and "pseudo-replication" in text
    tf = ev.tests_frame(cal)
    assert list(tf["index"]) == ["CSI", "CPI"] and list(tf["gain"]) == ["ks = 4", "kp = 9"]


def test_configurations_are_never_mixed_and_missing_subjects_group_per_recording():
    rows = _cohort()
    rows += [_row("M0", 5.0, 0.1, 0.1, method="exact"), _row("M1", 5.0, 0.1, 0.1, method="exact")]
    rows += [_row("", 1.0, 0.1, 0.4, stem="rec_a"), _row("", 1.5, 0.1, 0.4, stem="rec_a"),
             _row("  ", 1.0, 0.1, 0.4, stem="rec_b")]
    frame = ev.registry_frame(rows)
    opts = ev.config_options(frame)
    assert [cfg for cfg, _n in opts] == [("robust", 3.0), ("exact", 3.0)]
    assert opts[0][1] == 19 and ev.config_label(opts[0][0]) == "robust · 3 s"
    cal = ev.cohort_calibration(frame, 3.0, 12.0)              # most common config
    assert cal.config == ("robust", 3.0) and cal.counts["n_events"] == 19
    assert cal.counts["n_subjects"] == 10 and cal.counts["n_no_subject"] == 3
    assert {"[rec_a]", "[rec_b]"} <= set(cal.events["subject_key"])
    assert any("another method / window" in n for n in cal.notes)
    assert any("no subject ID" in n for n in cal.notes)
    cal_x = ev.cohort_calibration(frame, 3.0, 12.0, config=("exact", 3))
    assert cal_x.counts["n_events"] == 2
    assert any("p cannot reach 0.05" in n for n in cal_x.notes)
    assert ev.row_config(rows[-4]) == ("exact", 3.0)


def test_rows_stored_with_other_gains_are_recomposed_and_noted():
    rows = [_row(f"M{i}", 2.0 + 0.1 * i, 0.15, 0.5, ks=5.0, kp=8.0) for i in range(4)]
    rows += [_row(f"N{i}", 2.0 + 0.1 * i, 0.15, 0.5) for i in range(4)]
    cal = ev.cohort_calibration(ev.registry_frame(rows), 3.0, 12.0, aggregation="event")
    np.testing.assert_allclose(cal.events["csi_delta"],
                               -cal.events["d_delta"] + 3.0 * cal.events["sd2_delta"])
    np.testing.assert_allclose(cal.events["cpi_delta"],
                               cal.events["d_delta"] + 12.0 * cal.events["sd1_delta"])
    assert any("other gains" in n for n in cal.notes)


def test_empty_registry_and_no_knee():
    cal = ev.cohort_calibration(ev.registry_frame([]), 3.0, 12.0)
    assert cal.counts["n_events"] == 0 and cal.config is None
    assert all(cal.curves[k]["z"].isna().all() for k in ("csi", "cpi"))
    ks, kp, text, enabled = ev.apply_plan(cal)
    assert (ks, kp, enabled) == (3.0, 12.0, False) and "(kept)" in text
    fig = Figure()
    ev.draw_calibration(fig, cal)
    assert len(fig.axes) == 2
    assert any("No event" in t.get_text() for t in fig.axes[0].texts)


def test_calibration_figure_marks_the_knee_and_the_gains():
    cal = ev.cohort_calibration(ev.registry_frame(_cohort()), 3.0, 12.0)
    fig = Figure()
    ev.draw_calibration(fig, cal)
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert f"knee ks = {cal.knees['csi']:g}" in labels
    assert "current ks = 3" in labels and "rat SHR ks = 3" in labels
    assert list(fig.axes[1].get_xticks()) == list(range(1, 21))


# ── review fixes: reversal, rank steps / ceiling, in-sample, labels, subjects

def _reversing_cohort(seed=0, n=10):
    """A tachycardic response that also lowers SD2: D falls (CSI's -ΔD term
    rises) while ks·ΔSD2 falls, so Z(ks) changes sign as ks grows."""
    rng = np.random.default_rng(seed)
    return [_row(f"M{i}", d=rng.normal(-4, 1.5), s1=rng.normal(-0.4, 0.25),
                 s2=rng.normal(-0.6, 0.3)) for i in range(n)]


def test_a_knee_on_a_reversed_z_curve_is_flagged_and_needs_confirmation():
    cal = ev.cohort_calibration(ev.registry_frame(_reversing_cohort()), 3.0, 12.0)
    d = cal.diagnostics["csi"]
    runs = d["runs"]
    assert len(runs) == 2 and runs[0]["sign"] == 1 and runs[-1]["sign"] == -1
    kn = cal.knees["csi"]
    assert math.isfinite(kn) and kn >= runs[-1]["k_first"] and d["reversed_knee"]
    ks, kp, text, enabled = ev.apply_plan(cal)
    assert ks == kn and enabled and "⚠ reversed" in text
    caveats = ev.apply_caveats(cal)
    assert len(caveats) == 2 and "reverses sign" in caveats[0] and "OPPOSITE" in caveats[0]
    assert "CSI rises after the onset" in caveats[0] and "CSI falls after the onset" in caveats[0]
    assert caveats[-1] == ev.IN_SAMPLE_NOTE
    text = ev.calibration_text(cal)
    assert "⚠ on a reversed stretch" in text and "reverses sign" in text
    # the plot says so too
    fig = Figure()
    ev.draw_calibration(fig, cal)
    assert "reverses sign" in fig.axes[0].get_title(loc="left")
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any("reversed" in lab for lab in labels)


def test_a_z_curve_on_the_wilcoxon_ceiling_or_one_rank_step_offers_no_knee():
    # CPI: every subject's index falls at every k -> Z sits on the ceiling
    cal = ev.cohort_calibration(ev.registry_frame(_reversing_cohort()), 3.0, 12.0)
    d = cal.diagnostics["cpi"]
    assert not math.isfinite(cal.knees["cpi"])
    assert d["at_ceiling"] and d["k_max_ranges"] == [(1.0, 20.0)]
    assert d["abs_max"] == pytest.approx(d["z_max"])
    assert d["z_max"] == pytest.approx(math.sqrt(3 * 10 * 11 / (2 * 21)))
    assert "reaches its ceiling" in ev.ceiling_text(cal, "cpi")
    # the reversing CSI reaches the ceiling on both stretches: two k ranges
    assert len(cal.diagnostics["csi"]["k_max_ranges"]) == 2
    # one subject flips sign at kp = 12: Z moves by ONE rank step -- Kneedle
    # puts a "knee" there, which is refused (the statistic's discreteness)
    rows = [_row("F", d=-1.15, s1=0.1, s2=0.5)]
    rows += [_row(f"M{i}", d=-4 - 0.3 * i, s1=-0.1, s2=0.5) for i in range(7)]
    cal2 = ev.cohort_calibration(ev.registry_frame(rows), 3.0, 12.0)
    curve = cal2.curves["cpi"]
    w = curve["statistic"].to_numpy(dtype=float)
    assert w.max() - w.min() == 1.0
    assert math.isfinite(knee_point(curve["k"], curve["z"]))         # the raw Kneedle knee
    assert not math.isfinite(cal2.knees["cpi"])
    assert cal2.diagnostics["cpi"]["rank_refused"]
    assert f"fewer than {ev.MIN_RANK_STEPS} rank steps" in ev.calibration_text(cal2)
    ks, kp, text, enabled = ev.apply_plan(cal2)
    assert kp == 12.0 and "kp = 12 (kept)" in text


def test_tests_at_this_cohorts_own_knees_are_marked_in_sample():
    frame = ev.registry_frame(_cohort())
    cal = ev.cohort_calibration(frame, 3.0, 12.0)
    assert ev.in_sample(cal) == {"csi": False, "cpi": False} or cal.knees["csi"] == 3.0
    ks, kp, _text, _enabled = ev.apply_plan(cal)
    cal_in = ev.cohort_calibration(frame, ks, kp)
    assert ev.in_sample(cal_in) == {"csi": True, "cpi": True}
    assert list(ev.tests_frame(cal_in)["in_sample"]) == [True, True]
    text = ev.calibration_text(cal_in)
    assert "IN-SAMPLE" in text and "optimistic" in text
    assert "separate WKY cohort" in ev.IN_SAMPLE_NOTE


def test_calibration_defaults_to_one_label_and_discloses_pooled_labels():
    rows = [_row(f"M{i}", 2.0 + 0.2 * i, -0.3, 0.5, label="dobu") for i in range(6)]
    rows += [_row(f"M{i}", 0.1 * i, 0.02, 0.01, label="saline", onset=400.0)
             for i in range(3)]
    frame = ev.registry_frame(rows)
    counts = ev.label_counts(frame)
    assert counts == {"dobu": 6, "saline": 3}
    assert ev.default_calibration_label(counts) == "dobu"
    assert ev.default_calibration_label({ev.UNLABELLED: 4}) is None
    pooled = ev.cohort_calibration(frame, 3.0, 12.0)
    assert pooled.labels == counts
    assert any("Events of several labels are pooled" in n for n in pooled.notes)
    one = ev.cohort_calibration(frame[frame["label"] == "dobu"], 3.0, 12.0)
    assert one.labels == {"dobu": 6}
    assert not any("pooled" in n for n in one.notes)
    if any(math.isfinite(v) for v in pooled.knees.values()):
        assert any("pooled" in c for c in ev.apply_caveats(pooled))


def test_default_looking_subject_ids_shared_by_recordings_are_disclosed():
    assert ev.is_default_subject("subject_01") and ev.is_default_subject(" Subject 2 ")
    assert ev.is_default_subject("subject") and not ev.is_default_subject("M7")
    assert not ev.is_default_subject("subject_A") and not ev.is_default_subject("")
    rows = [_row("subject_01", 2.0 + 0.1 * i, -0.3, 0.5, stem=f"mouse_{c}")
            for i, c in enumerate("ABC")]
    cal = ev.cohort_calibration(ev.registry_frame(rows), 3.0, 12.0)
    assert cal.counts["n_subjects"] == 1 and cal.counts["n_recordings"] == 3
    assert any("default-looking Subject ID" in n and "“subject_01”" in n for n in cal.notes)
    # the save-time warning: the placeholder, and other recordings under it
    others = [{"subject": "subject_01", "filepath": "C:/data/mouse_A.mat"}]
    msg = ev.subject_warning("subject_01", "C:/data/mouse_B.mat", "P", others)
    assert "looks like the app's default" in msg and "mouse_A.mat" in msg
    assert ev.subject_warning("M7", "C:/data/mouse_B.mat", "P", []) == ""
    assert "mouse_A.mat" in ev.subject_warning("M7", "C:/data/mouse_B.mat", "P",
                                                [{"subject": "M7",
                                                  "filepath": "C:/data/mouse_A.mat"}])


def test_lf_hf_values_keep_one_notation_and_the_tests_show_p_of_z(recording):
    assert ev._val("lf", 8637.0) == "8.637e+03"
    assert ev._val("hf", 7.203e4) == "7.203e+04"
    assert ev._val("lf", 7328.0, signed=True) == "+7.328e+03"
    assert ev._val("hr", 545.04) == "545.0" and ev._val("lf", float("nan")) == "—"
    res, t_event = recording
    el = ev.compute_event_locked(res, ev.select_events(_anns(t_event)))
    text = ev.event_table_text(el)
    assert "p(z)" in text and "the normal p of" in text and "signrank" in text
