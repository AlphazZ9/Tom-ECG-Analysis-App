"""
AnalysisController paths around artifact correction, session restore and
the persistent disclosures, driven through a stub app (no Tk window):

* removed spurious detections are not HRV breaks, so T-wave double
  detection corrected on every beat still analyses;
* a second Analyze keeps the cumulative artifact counts;
* a session save + restore re-applies the removals that re-detection
  undoes, so the HRV breaks survive;
* the result labels (RR-exclusion line, withheld spectrum) are re-derived
  after a UI rebuild / restore;
* the auto-epochs continuation of run_freq, and the epoch status keeping
  the core exclusion note; skipped epochs break the plotted line.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ecg.core import analysis
from ecg.core.detection import correct_rr_artifacts
from ecg.core.models import MouseECG
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import analysis_controller as ac_module
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.state import AnalysisState, DetectionState, SignalState
from ecg.ui.theme import GREEN, ORANGE

from tests.helpers import add_qrs_pulse, rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0


class _Label:
    def __init__(self, text: str = "") -> None:
        self.text, self.text_color = text, None

    def configure(self, **kw) -> None:
        self.text = kw.get("text", self.text)
        self.text_color = kw.get("text_color", self.text_color)


class _Entry:
    def __init__(self, value) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value


class _Slot:
    def __init__(self) -> None:
        self.draw_fn = None

    def update(self, fn) -> None:
        self.draw_fn = fn


def _safe_float(widget, default):
    try:
        return float(widget.get()) if widget is not None else default
    except (TypeError, ValueError):
        return default


def _stub_app(peaks=None, signal=None, results=None, artifact_report=None):
    statuses: "list[tuple[str, object]]" = []
    sig = signal if signal is not None else (
        np.zeros(int(peaks[-1]) + int(FS)) if peaks is not None else None)
    app = SimpleNamespace(
        analysis=AnalysisState(results=results, artifact_report=artifact_report),
        detection=DetectionState(rpeaks_ok=peaks),
        signal=SignalState(filtered=sig, fs=FS,
                           time=np.arange(len(sig)) / FS if sig is not None else None),
        after=lambda ms, fn=None: None,
        _set_status=lambda text, color=None: statuses.append((text, color)),
        _safe_float=_safe_float,
        ent_minrr=_Entry(MouseECG.RR_MIN_MS), ent_maxrr=_Entry(MouseECG.RR_MAX_MS),
        lbl_analysis_window=_Label(), lbl_freq_status=_Label(), lbl_nonlin_status=_Label(),
    )
    app._windowed_peaks = lambda: app.detection.rpeaks_ok
    return app, statuses


def _pvc_peaks(n_side: int = 100, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rr = list(120 + rng.normal(0, 1, n_side)) + [80.0, 160.0] + list(120 + rng.normal(0, 1, n_side))
    return rpeaks_from_rr(rr, FS)


def _worker_params() -> dict:
    return {"artifact_correction": True, "min_rr_ms": MouseECG.RR_MIN_MS,
            "hrv_rr_bounds": (MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS),
            "break_tol_samples": 10}


# ── artifact correction inside Analyze ────────────────────────────────────

@requires_nk
def test_analyze_with_corrected_t_wave_double_detections_still_works():
    rng = np.random.default_rng(5)
    peaks = rpeaks_from_rr(list(120 + rng.normal(0, 1.5, 199)), FS)
    t_peaks = peaks + int(0.040 * FS)
    sig = np.zeros(int(peaks[-1]) + int(FS))
    for p, t in zip(peaks, t_peaks):
        add_qrs_pulse(sig, int(p), FS, amp=1.0)
        add_qrs_pulse(sig, int(t), FS, amp=0.4)
    app, _ = _stub_app(np.sort(np.r_[peaks, t_peaks]), signal=sig)

    bundle = AnalysisController(app).analysis_worker(_worker_params())

    assert np.array_equal(bundle["rpeaks_ok"], peaks)
    assert bundle["artifact_report"]["n_duplicate"] == len(peaks)
    assert bundle["artifact_report"]["break_samples"] == []
    hr = bundle["results"]["hr"]
    assert hr["n_valid"] == len(peaks) - 1 and hr["n_spanning_removed"] == 0


@requires_nk
def test_second_analyze_keeps_the_cumulative_artifact_counts():
    peaks = _pvc_peaks()
    app, _ = _stub_app(peaks)
    ctrl = AnalysisController(app)

    first = ctrl.analysis_worker(_worker_params())
    app.detection.rpeaks_ok = first["rpeaks_ok"]
    second = ctrl.analysis_worker(_worker_params(), first["artifact_report"])

    rep = second["artifact_report"]
    assert rep["n_in"] - rep["n_out"] == 1                  # not "no artifacts"
    assert (rep["n_in"], rep["n_out"]) == (len(peaks), len(peaks) - 1)
    assert rep["n_ectopic"] == 1
    assert rep["break_samples"] == [int(peaks[101])]
    assert second["results"]["hr"]["n_spanning_removed"] == 1


# ── session save / restore ────────────────────────────────────────────────

def _corrected_app():
    peaks = _pvc_peaks()
    corrected, report = correct_rr_artifacts(peaks, FS)
    app, _ = _stub_app(corrected, artifact_report=report)
    return peaks, corrected, app


def _json_round_trip(obj):
    return json.loads(json.dumps(obj, cls=_SessionEncoder), object_hook=_session_decode_hook)


def test_restore_reapplies_the_removals_that_redetection_undid():
    peaks, corrected, app = _corrected_app()
    saved = _json_round_trip(AnalysisController(app).current_artifact_report())

    # Restore: re-detection rebuilds the peaks from the full candidate set,
    # so the removed premature beat is back and its break is lost ...
    restored, _ = _stub_app(peaks.copy(), artifact_report=saved)
    ctrl = AnalysisController(restored)
    assert ctrl.hrv_breaks() is None
    # ... until the persisted removals are re-applied.
    assert ctrl.reapply_artifact_removals() == 1
    np.testing.assert_array_equal(restored.detection.rpeaks_ok, corrected)
    assert ctrl.hrv_breaks().tolist() == [int(peaks[101])]


def test_restore_never_drops_a_peak_added_by_hand():
    peaks, _, app = _corrected_app()
    saved = AnalysisController(app).current_artifact_report()
    restored, _ = _stub_app(peaks.copy(), artifact_report=saved)
    restored.detection.manual_added = {int(peaks[101])}
    assert AnalysisController(restored).reapply_artifact_removals() == 0
    assert len(restored.detection.rpeaks_ok) == len(peaks)


def test_saved_report_omits_removed_beats_that_came_back():
    peaks, _, app = _corrected_app()
    app.detection.rpeaks_ok = peaks.copy()      # e.g. threshold change re-detected
    saved = AnalysisController(app).current_artifact_report()
    assert saved["removed_samples"] == [] and saved["n_in"] == saved["n_out"]


# ── persistent result labels ──────────────────────────────────────────────

def _results(hrv_freq):
    hr = {"n": 400, "n_excluded": 2, "n_too_long": 2, "excluded_pct": 0.5,
          "rr_min_ms": MouseECG.RR_MIN_MS, "rr_max_ms": MouseECG.RR_MAX_MS}
    return {"hr": hr, "hrv_freq": hrv_freq, "hrv_nonlin": pd.DataFrame()}


def test_labels_are_rederived_after_a_rebuild_with_withheld_reason():
    fd = pd.DataFrame()
    fd.attrs["withheld_reason"] = "2.1 s run of excluded RR intervals (limit 2 s)"
    app, _ = _stub_app(np.arange(0, 400 * 240, 240), results=_results(fd))
    app.analysis.analysed_window = (0.0, 0.0)

    AnalysisController(app).refresh_result_labels()   # fresh labels, as after a rebuild

    assert app.lbl_analysis_window.text.endswith("\n2 RR excluded from HRV (0.5 %)")
    assert "full signal" in app.lbl_analysis_window.text
    assert app.lbl_analysis_window.text_color == ORANGE
    assert app.lbl_freq_status.text == "  Withheld — 2.1 s run of excluded RR intervals (limit 2 s)"
    assert app.lbl_freq_status.text_color == ORANGE
    assert "SampEn" in app.lbl_nonlin_status.text


def test_labels_describe_the_analysed_window_and_a_computed_spectrum():
    fd = pd.DataFrame({"HRV_LFn": [0.25], "HRV_HFn": [0.75], "HRV_LFHF": [0.33]})
    res = _results(fd)
    res["hr"]["n_excluded"] = 0
    app, _ = _stub_app(np.arange(0, 600 * 240, 240), results=res)
    app.analysis.analysed_window = (10.0, 50.0)
    app.analysis.t_start, app.analysis.t_end = 20.0, 0.0   # edited since Analyze

    AnalysisController(app).refresh_result_labels(freq_done_label="Loaded from session ✓")

    assert "400/600 peaks" in app.lbl_analysis_window.text
    assert "10.0 s → 50.0 s" in app.lbl_analysis_window.text
    assert app.lbl_analysis_window.text_color == GREEN
    assert app.lbl_freq_status.text.startswith("  Loaded from session ✓  LFn=25.0%")


def test_labels_untouched_without_results():
    app, _ = _stub_app(np.arange(0, 100 * 240, 240))
    app.lbl_freq_status.text = "  Click Analyze first"
    AnalysisController(app).refresh_result_labels()
    assert app.lbl_freq_status.text == "  Click Analyze first"


# ── run_freq continuation (auto epochs) ───────────────────────────────────

def _sync_async(app):
    app._generation = 0
    app.btn_run_freq = app.btn_compute_epochs = None
    app._start_async_result = lambda btn, label, worker, done: done(worker())
    app._run_plot_chain = lambda tasks, on_complete=None: on_complete and on_complete()
    app._plot_psd = app._plot_hrv_tables = app._plot_summary = lambda r: None


def test_run_freq_runs_its_continuation_after_the_final_status(monkeypatch):
    fd = pd.DataFrame()
    fd.attrs["withheld_reason"] = "too long"
    monkeypatch.setattr(ac_module, "analyse_hrv_freq", lambda *a, **k: fd)
    app, statuses = _stub_app(np.arange(0, 100 * 240, 240), results=_results(pd.DataFrame()))
    _sync_async(app)
    calls: "list[int]" = []

    AnalysisController(app).run_freq(then=lambda: calls.append(len(statuses)))

    assert calls == [len(statuses)]                       # after the final status
    text, color = statuses[-1]
    assert text == "Frequency HRV withheld — too long  |  2 RR excluded (outside 67–333 ms)"
    assert color == ORANGE
    assert app.lbl_freq_status.text == "  Withheld — too long"


# ── epochs ────────────────────────────────────────────────────────────────

def _epoch_app(rr: "list[float]", hr: dict, epoch_s: float):
    jitter = np.random.default_rng(7).normal(0, 1.0, len(rr))   # not perfectly regular
    peaks = rpeaks_from_rr(list(np.asarray(rr) + jitter), FS)
    app, statuses = _stub_app(peaks, results={"hr": hr})
    _sync_async(app)
    app.ent_epoch, app.ent_overlap = _Entry(epoch_s), _Entry(0)
    app._slots = {"epochs": _Slot()}
    app.txt_epochs = None
    app._set_textbox = lambda box, text, tsv=None: setattr(app, "epoch_text", text)
    app._df_to_tsv = lambda df: ""
    app.lbl_epoch_info = app.lbl_epoch_count = app.lbl_summary_epoch_info = _Label()
    app.tabs = SimpleNamespace(set=lambda name: None)
    return app, statuses


def test_epoch_status_keeps_the_core_exclusion_note():
    # 100 s at 120 ms with one 400 ms pause in the tail after the last whole
    # 30 s epoch: no epoch contains it, but the core HRV excluded it.
    rr = [120.0] * 790 + [400.0] + [120.0] * 40
    hr = {"n_excluded": 1, "n_too_long": 1,
          "rr_min_ms": MouseECG.RR_MIN_MS, "rr_max_ms": MouseECG.RR_MAX_MS}
    app, statuses = _epoch_app(rr, hr, epoch_s=30.0)

    AnalysisController(app).compute_epochs()

    assert int(app.analysis.epoch_df["N_excluded"].sum()) == 0
    text, color = statuses[-1]
    assert text.endswith("  |  1 RR excluded (outside 67–333 ms)")
    assert color == ORANGE


def test_skipped_epochs_break_the_plotted_line():
    # A 30 s stretch at 400 ms (all outside Max RR) between two 30 s
    # stretches at 120 ms: the epochs inside it have no valid interval and
    # are skipped -- the line must show a gap, not bridge it.
    rr = [120.0] * 250 + [400.0] * 75 + [120.0] * 250
    app, _ = _epoch_app(rr, {"n_excluded": 75}, epoch_s=10.0)

    AnalysisController(app).compute_epochs()

    t_mid = ((app.analysis.epoch_df["Epoch_start_s"]
              + app.analysis.epoch_df["Epoch_end_s"]) / 2).to_numpy()
    assert np.diff(t_mid).max() > 15.0                     # epochs were skipped
    assert "gap in the plotted lines" in app.epoch_text
    fig = Figure()
    app._slots["epochs"].draw_fn(fig)
    for ax in fig.axes:
        y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert np.isnan(y).any()
