"""
HRV > Autonomic view helpers (ecg.ui.autonomic_view) -- no Tk window:
parameter parsing (strict, for a Compute click) and normalisation (tolerant,
for sessions and UI rebuilds), the withheld-window spans, the
recording-level summary and status, and the stacked figure drawn on a bare
matplotlib Figure.
"""
import json

import numpy as np
import pytest
from matplotlib.figure import Figure

from ecg.core.autonomic import (
    COV_METHODS, DEFAULT_METHOD, KP_RAT_SHR, KS_RAT_SHR, compute_autonomic_indices,
)
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import autonomic_view as av

from tests.helpers import rpeaks_from_rr

FS = 2000.0


def _peaks(rr_ms, seed: int = 0, jitter: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rr = np.asarray(rr_ms, dtype=float) + rng.normal(0, jitter, len(rr_ms))
    return rpeaks_from_rr(list(rr), FS)


def _result(rr_ms=None, **kw):
    rr = rr_ms if rr_ms is not None else [120.0] * 600
    return compute_autonomic_indices(_peaks(rr), FS, **kw)


def _fake_spwvd(t0: float, t1: float, gaps: int = 0) -> dict:
    t = np.arange(t0, t1, 0.05)
    return {"t": t, "lf": 100 + np.sin(t), "hf": 50 + np.cos(t),
            "params": {"lf_band": (0.4, 1.5), "hf_band": (1.5, 5.0),
                       "fs_resample": 20.0, "fbins": 4096, "n_gaps_masked": gaps,
                       "nan_gap_s": 2.0, "n_nan_samples": 0}}


# ── parameters ────────────────────────────────────────────────────────────

def test_defaults_are_the_papers_rat_values_with_lf_hf_off():
    p = av.default_autonomic_params()
    assert p == {"window_s": 3.0, "method": DEFAULT_METHOD, "ks": KS_RAT_SHR,
                 "kp": KP_RAT_SHR, "lf_hf": False, "min_pairs": 10}
    assert av.is_rat_calibration(p["ks"], p["kp"])
    assert "not validated in mice" in av.CALIBRATION_WARNING


def test_parse_accepts_a_label_or_key_and_a_decimal_comma():
    label = COV_METHODS["rat_reference"]
    p = av.parse_autonomic_inputs(" 2,5 ", label, "3", "12.5", 1)
    assert p == {"window_s": 2.5, "method": "rat_reference", "ks": 3.0,
                 "kp": 12.5, "lf_hf": True, "min_pairs": 10}
    assert av.parse_autonomic_inputs("3", "exact", "0", "1", 0)["method"] == "exact"
    # an unreadable checkbox value means off, it does not block a Compute
    assert av.parse_autonomic_inputs("3", "exact", "3", "12", "maybe")["lf_hf"] is False


@pytest.mark.parametrize("window, method, ks, kp, needle", [
    ("abc", "robust", "3", "12", "Window"),
    ("nan", "robust", "3", "12", "Window"),
    ("0.5", "robust", "3", "12", "Window"),       # below WINDOW_MIN_S
    ("61", "robust", "3", "12", "Window"),        # above WINDOW_MAX_S
    ("3", "bogus", "3", "12", "method"),
    ("3", "robust", "-1", "12", "ks"),
    ("3", "robust", "3", "inf", "kp"),
    ("3", "robust", "", "12", "ks"),
])
def test_parse_rejects_bad_entries_with_a_message(window, method, ks, kp, needle):
    with pytest.raises(ValueError, match=needle):
        av.parse_autonomic_inputs(window, method, ks, kp)


def test_normalise_tolerates_old_and_corrupt_sessions():
    base = {"window_s": 5.0, "method": "exact", "ks": 4.0, "kp": 9.0, "lf_hf": True,
            "min_pairs": 10}
    # A session saved before the Autonomic view: nothing changes.
    assert av.normalise_autonomic_params(None, base=base) == base
    assert av.normalise_autonomic_params("garbage", base=base) == base
    # Per key: valid values are taken, invalid ones keep the base value.
    p = av.normalise_autonomic_params(
        {"window_s": "abc", "method": COV_METHODS["ledoit_wolf"], "ks": 6,
         "kp": -3, "lf_hf": "false", "unknown": 1}, base=base)
    assert p == {"window_s": 5.0, "method": "ledoit_wolf", "ks": 6.0, "kp": 9.0,
                 "lf_hf": False, "min_pairs": 10}
    # No base: the defaults.
    assert av.normalise_autonomic_params({}) == av.default_autonomic_params()
    # A corrupt base is itself normalised.
    assert av.normalise_autonomic_params(None, base={"window_s": 1e9})["window_s"] == 3.0


def test_params_survive_the_session_json_round_trip():
    p = av.normalise_autonomic_params(
        {"window_s": np.float64(4.0), "method": "approximate", "ks": np.int64(2),
         "kp": 15.0, "lf_hf": np.bool_(True)})
    text = json.dumps({"autonomic_params": p}, cls=_SessionEncoder)
    back = json.loads(text, object_hook=_session_decode_hook)["autonomic_params"]
    assert av.normalise_autonomic_params(back) == p


def test_method_key_and_format_param():
    for key, label in COV_METHODS.items():
        assert av.method_key(key) == key
        assert av.method_key(f"  {label} ") == key
    assert av.method_key("nope") is None and av.method_key(None) is None
    assert av.format_param(3.0) == "3" and av.format_param(2.5) == "2.5"


# ── result helpers ────────────────────────────────────────────────────────

def test_withheld_spans_cover_every_blank_csi_sample():
    # 19 s at ~320 ms: a 3 s window holds only ~8 valid pairs (< 10) there.
    res = _result([120.0] * 300 + [320.0] * 60 + [120.0] * 300)
    spans = av.withheld_spans(res)
    assert len(spans) == 1
    lo, hi = spans[0]
    t_nan = res.t[~np.isfinite(res.csi)]
    assert len(t_nan) > 0
    tol = 1.0 / res.params["fs_out"]
    assert np.all((t_nan >= lo - tol) & (t_nan <= hi + tol))
    # the computed stretch on either side is outside the span
    assert np.isfinite(res.csi[res.t < lo - tol]).all()
    assert np.isfinite(res.csi[res.t > hi + tol]).all()


def test_no_spans_when_every_window_is_computed():
    assert av.withheld_spans(_result()) == []


def test_summary_frame_is_one_row_of_recording_values():
    res = _result([120.0] * 300 + [400.0] + [120.0] * 300)      # one pause
    df = av.summary_frame(res)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["SD01_ms"] == pytest.approx(res.sd01)
    assert row["SD02_ms"] == pytest.approx(res.sd02)
    assert row["D0_ms"] == pytest.approx(res.d0)
    assert row["N_beats_evaluated"] == res.params["n_beats"]
    assert row["N_RR_excluded"] == 1 and row["Method"] == DEFAULT_METHOD
    assert "LF_mean_au" not in df.columns
    df2 = av.summary_frame(res, _fake_spwvd(0, 60))
    assert {"LF_mean_au", "HF_mean_au", "LF_band_Hz", "HF_band_Hz"} <= set(df2.columns)
    assert df2.iloc[0]["LF_band_Hz"] == "0.4–1.5"
    assert "LF_mean_au" not in av.summary_frame(res, {"error": "too short"}).columns


def test_summary_text_discloses_method_calibration_and_exclusions():
    res = _result([120.0] * 300 + [400.0] + [120.0] * 300)
    text = av.summary_text(res)
    for needle in ("SD01", "SD02", "D0 (CCD0)", "Beats evaluated",
                   "NOT validated in mice", "1 too long", res.params["method_label"],
                   "Schafer-Strimmer"):
        assert needle in text, needle
    assert "SPWVD" not in text
    text2 = av.summary_text(res, {"error": "fewer than 4 valid intervals"})
    assert "not computed — fewer than 4 valid intervals" in text2
    assert "unnormalised" in text2
    user = av.summary_text(_result(ks=5.0, kp=10.0))
    assert "user-set" in user and "NOT validated" not in user.split("Notes")[0]


def test_a_small_nonzero_exclusion_never_reads_as_zero_percent():
    rr = [120.0] * 2500 + [400.0] + [120.0] * 10
    text = av.summary_text(_result(rr))
    assert "1 excluded (0.04 %)" in text
    assert av._pct_text(0.0) == "0.0" and av._pct_text(1.234) == "1.2"


def test_status_levels():
    text, level = av.autonomic_status(_result())
    assert level == "ok" and text.endswith("✓")
    text, level = av.autonomic_status(_result([120.0] * 300 + [400.0] + [120.0] * 300))
    assert level == "warn" and "1 RR excluded" in text
    text, level = av.autonomic_status(_result(), {"error": "boom"})
    assert level == "warn" and "LF/HF not computed: boom" in text
    text, level = av.autonomic_status(_result(), _fake_spwvd(0, 60, gaps=2))
    assert level == "warn" and "blank over 2 gap(s)" in text
    text, level = av.autonomic_status(_result([120.0] * 10))   # too short
    assert level == "withheld" and text.startswith("  Withheld — recording too short")


# ── figure ────────────────────────────────────────────────────────────────

def _texts(ax) -> "list[str]":
    return [t.get_text() for t in ax.texts]


def test_figure_has_four_shared_panels_and_a_fifth_for_lf_hf():
    res = _result()
    fig = Figure()
    av.draw_autonomic(fig, res)
    assert len(fig.axes) == 4
    title = fig.axes[0].get_title(loc="left")
    assert res.params["method_label"] in title and "window 3 s" in title
    assert "ks = 3, kp = 12" in title and "not validated in mice" in title
    assert "CSI = D̃ + 3·SD2" in fig.axes[3].get_title(loc="left")
    assert [len(ax.lines) for ax in fig.axes] == [1, 1, 2, 2]

    fig5 = Figure()
    av.draw_autonomic(fig5, res, _fake_spwvd(float(res.t[0]), float(res.t[-1])))
    assert len(fig5.axes) == 5 and len(fig5.axes[4].lines) == 2
    assert "SPWVD" in fig5.axes[4].get_title(loc="left")
    # an SPWVD error draws no LF/HF panel
    fig_err = Figure()
    av.draw_autonomic(fig_err, res, {"error": "too short"})
    assert len(fig_err.axes) == 4


def test_annotations_on_every_panel_labels_on_top_x_range_kept():
    res = _result()
    t0, t1 = float(res.t[0]), float(res.t[-1])
    anns = [{"t_start": t0 + 10, "t_end": t0 + 20, "label": "Dobutamine", "color": "#123456"},
            {"t_start": t1 + 100, "t_end": t1 + 200, "label": "Outside", "color": "#654321"},
            {"label": "malformed"}]
    fig = Figure()
    av.draw_autonomic(fig, res, annotations=anns)
    for ax in fig.axes:
        assert ax.get_xlim() == pytest.approx((t0, t1))
        spans = [p for p in ax.patches if p.get_facecolor()[:3] == pytest.approx(
            (0x12 / 255, 0x34 / 255, 0x56 / 255))]
        assert len(spans) == 1
    assert "Dobutamine" in _texts(fig.axes[0])
    assert "Outside" not in _texts(fig.axes[0])
    assert all(not _texts(ax) for ax in fig.axes[1:])


def test_withheld_windows_are_shaded_and_in_the_legend():
    res = _result([120.0] * 300 + [320.0] * 60 + [120.0] * 300)
    fig = Figure()
    av.draw_autonomic(fig, res)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert any("withheld" in t.get_text() for t in legend.get_texts())
    assert all(len(ax.patches) >= 1 for ax in fig.axes)


def test_long_series_are_decimated_for_display():
    res = _result([120.0] * 3000)                     # ~6 min -> ~7000 samples
    fig = Figure()
    av.draw_autonomic(fig, res, max_points=1000)
    n = len(fig.axes[0].lines[0].get_xdata())
    assert n <= 1000
    assert "decimated" in fig.axes[-1].get_xlabel()


def test_empty_result_shows_the_reason():
    res = _result([120.0] * 10)
    assert len(res.t) == 0
    fig = Figure()
    av.draw_autonomic(fig, res)
    assert any("recording too short" in t for t in _texts(fig.axes[0]))
