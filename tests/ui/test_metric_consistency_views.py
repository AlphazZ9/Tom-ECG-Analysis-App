"""
The same numbers across views, driven through stub apps (no Tk window):
Rolling HRV's SD1/SD2, HR and LF n.u. equal the core definitions on the same
beats; the PSD panel draws the spectrum the frequency numbers were
integrated from; the Non-linear status discloses a capped entropy block and
the cap comes from the Non-linear tab; Compare Segments no longer has a
beat-level Mann-Whitney test.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.models import MouseECG
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.plot_controller import PlotController
from ecg.ui.state import AnalysisState, DetectionState, SignalState

from tests.helpers import rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0


class _Entry:
    def __init__(self, value) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value


class _Label:
    def __init__(self) -> None:
        self.text, self.text_color = "", None

    def configure(self, **kw) -> None:
        self.text = kw.get("text", self.text)
        self.text_color = kw.get("text_color", self.text_color)


class _Slot:
    def update(self, fn) -> None:
        self.draw_fn = fn


def _safe_float(widget, default):
    try:
        return float(widget.get()) if widget is not None else default
    except (TypeError, ValueError):
        return default


def _modulated_rr(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.12
    return (120.0 + 4.0 * np.sin(2 * np.pi * 2.5 * t)
            + 2.0 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0.0, 1.0, n))


def _rolling_app(peaks: np.ndarray, win_s: float, step_s: float, metrics):
    statuses: "list" = []
    app = SimpleNamespace(
        analysis=AnalysisState(results={"hr": {}}),
        detection=DetectionState(rpeaks_ok=peaks),
        signal=SignalState(filtered=np.zeros(int(peaks[-1]) + 10), fs=FS),
        after=lambda ms, fn=None: None,
        _set_status=lambda text, color=None: statuses.append((text, color)),
        _safe_float=_safe_float,
        ent_minrr=_Entry(MouseECG.RR_MIN_MS), ent_maxrr=_Entry(MouseECG.RR_MAX_MS),
        ent_roll_win=_Entry(win_s), ent_roll_step=_Entry(step_s),
        _roll_metrics={m: _Entry(1) for m in metrics},
        lbl_roll_status=_Label(), btn_roll_compute=None,
        _slots={"rolling_hrv": _Slot()}, txt_rolling=None,
        _set_textbox=lambda box, text, tsv=None: None, _df_to_tsv=lambda df: "",
        tabs=SimpleNamespace(set=lambda name: None), _generation=0,
    )
    app._windowed_peaks = lambda: peaks
    app._start_async_result = lambda btn, label, worker, done: done(worker())
    return app


@requires_nk
def test_rolling_window_matches_core_definitions_on_the_same_beats():
    peaks = rpeaks_from_rr(list(_modulated_rr(1200)), FS)
    win_s, step_s = 60.0, 60.0
    app = _rolling_app(peaks, win_s, step_s, ["HR", "SD1", "SD2", "LF_nu"])

    AnalysisController(app).compute_rolling_hrv()

    df = app.analysis.rolling_hrv_df
    t_peaks = peaks / FS
    ep = peaks[(t_peaks >= t_peaks[0]) & (t_peaks < t_peaks[0] + win_s)]
    ser = analysis.rr_interval_series(ep, FS)
    pc = analysis.poincare_sd1_sd2(ser["rr_ms"], ser["valid"])
    fd = analysis.analyse_hrv_freq(ep, FS)
    row = df.iloc[0]
    assert row["SD1"] == pytest.approx(pc["sd1"], abs=5e-4)
    assert row["SD2"] == pytest.approx(pc["sd2"], abs=5e-4)
    assert row["HR"] == pytest.approx(analysis.mean_hr_bpm(ser["rri"]), abs=0.05)
    assert row["LF_nu"] == pytest.approx(100 * float(fd["HRV_LFn"].iloc[0]), abs=0.05)


@requires_nk
def test_psd_panel_draws_the_spectrum_behind_the_numbers():
    peaks = rpeaks_from_rr(list(_modulated_rr(1500)), FS)
    core = analysis.analyse_core(np.zeros(int(peaks[-1]) + 10), peaks, FS)
    core["hrv_freq"] = analysis.analyse_hrv_freq(peaks, FS)
    stub = SimpleNamespace(_rr_valid=lambda r: np.asarray(r["rr_valid"], dtype=bool))

    disp = PlotController.psd_for_display(stub, core)

    fd = core["hrv_freq"]
    assert disp["stored"] is True and disp["withheld"] is None
    np.testing.assert_array_equal(disp["psd"], np.asarray(fd.attrs["psd_power"]))
    m = analysis.band_mask(disp["freqs"], MouseECG.LF)
    assert float(analysis.trapz(disp["psd"][m], disp["freqs"][m])) == pytest.approx(
        float(fd["HRV_LF"].iloc[0]))
    assert disp["metrics"]["HRV_LFHF"] == pytest.approx(float(fd["HRV_LFHF"].iloc[0]))

    # Not run yet: the same estimator on the core's valid RR intervals.
    core["hrv_freq"] = pd.DataFrame()
    fresh = PlotController.psd_for_display(stub, core)
    assert fresh["stored"] is False
    np.testing.assert_allclose(fresh["psd"], np.asarray(fd.attrs["psd_power"]))
    assert fresh["metrics"]["HRV_LF"] == pytest.approx(float(fd["HRV_LF"].iloc[0]))


def test_nonlinear_status_discloses_the_capped_entropy_block():
    nl = pd.DataFrame({"HRV_SampEn": [1.2]})
    nl.attrs.update({"entropy_capped": True, "entropy_n_rr": 2000, "n_rr_valid": 9000,
                     "entropy_t_start_s": 100.0, "entropy_t_end_s": 340.0})
    text, _ = AnalysisController._nonlin_status(nl)
    assert "entropy from 2,000 of 9,000 RR (100–340 s)" in text

    nl.attrs.update({"entropy_capped": False})
    text, _ = AnalysisController._nonlin_status(nl)
    assert "entropy from" not in text


def test_entropy_cap_comes_from_the_nonlinear_tab():
    app = SimpleNamespace(ent_entropy_cap=_Entry(500))
    assert AnalysisController(app).entropy_cap() == 500
    app.ent_entropy_cap = _Entry("0")
    assert AnalysisController(app).entropy_cap() == 0          # no cap
    app.ent_entropy_cap = _Entry("abc")
    assert AnalysisController(app).entropy_cap() == MouseECG.ENTROPY_MAX_BEATS
    assert AnalysisController(SimpleNamespace()).entropy_cap() == MouseECG.ENTROPY_MAX_BEATS


def test_beat_level_mann_whitney_is_gone():
    assert not hasattr(AnalysisController, "mannwhitney_test")
    from ecg.ui.app import ECGApp
    assert not hasattr(ECGApp, "_mannwhitney_test")
    keys = [k for k, _, _ in ECGApp._SEGMENT_TABLE_METRICS]
    assert {"n_blocks", "hr_block_mean", "hr_block_sd"} <= set(keys)


@requires_nk
def test_stored_spectrum_survives_a_session_round_trip():
    import json
    from ecg.io.loaders import _deserialise_results, _serialise_results
    from ecg.io.session import _SessionEncoder, _session_decode_hook

    peaks = rpeaks_from_rr(list(_modulated_rr(1500)), FS)
    core = analysis.analyse_core(np.zeros(int(peaks[-1]) + 10), peaks, FS)
    core["hrv_freq"] = analysis.analyse_hrv_freq(peaks, FS)
    text = json.dumps(_serialise_results(core), cls=_SessionEncoder)
    out = _deserialise_results(json.loads(text, object_hook=_session_decode_hook))
    stub = SimpleNamespace(_rr_valid=lambda r: np.asarray(r["rr_valid"], dtype=bool))

    disp = PlotController.psd_for_display(stub, out)

    assert disp["stored"] is True
    np.testing.assert_allclose(disp["psd"], core["hrv_freq"].attrs["psd_power"])
    assert disp["bands"][1] == tuple(MouseECG.LF)
