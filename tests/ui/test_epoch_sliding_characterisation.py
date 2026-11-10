"""
Do Epochs and Sliding compute the same numbers on the same grid? Both loops are
driven through stub apps with window = step, on recordings that reach the code
paths where they differ: pauses, removed beats, very short windows.

The verdict decides whether the two loops may be merged (the note above
AnalysisController.compute_epochs records it): they may not, because the case
pinned in test_verdict_* gives a different RMSSD. Everything else is equal, and
is held equal here so the two cannot drift apart unnoticed.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core.models import MouseECG
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.state import AnalysisState, DetectionState, SignalState, UIState

from tests.helpers import rpeaks_from_rr

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0
# Epochs column -> the same quantity in Sliding.
SAME_QUANTITY = {"HR_mean": "HR", "SDNN": "SDNN", "RMSSD": "RMSSD", "pNN6": "pNN6",
                 "N_beats": "n_beats", "N_valid": "n_valid", "N_excluded": "n_excluded"}


class _Entry:
    def __init__(self, value) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value


class _Label:
    def configure(self, **kw) -> None:
        pass


class _Slot:
    def update(self, fn) -> None:
        pass


def _safe_float(widget, default):
    try:
        return float(widget.get())
    except (TypeError, ValueError):
        return default


def _app(peaks: np.ndarray, win_s: float, report: "dict | None" = None) -> SimpleNamespace:
    app = SimpleNamespace(
        analysis=AnalysisState(results={"hr": {}}, artifact_report=report),
        detection=DetectionState(rpeaks_ok=peaks),
        signal=SignalState(filtered=np.zeros(int(peaks[-1]) + 10), fs=FS, time=np.zeros(1)),
        ui=UIState(), after=lambda ms, fn=None: None, _generation=0,
        _set_status=lambda text, color=None: None, _safe_float=_safe_float,
        ent_minrr=_Entry(MouseECG.RR_MIN_MS), ent_maxrr=_Entry(MouseECG.RR_MAX_MS),
        ent_epoch=_Entry(win_s), ent_overlap=_Entry(0),
        ent_roll_win=_Entry(win_s), ent_roll_step=_Entry(win_s),
        _roll_metrics={m: _Entry(1) for m in ("HR", "SDNN", "RMSSD", "pNN6")},
        lbl_roll_status=_Label(), lbl_epoch_info=_Label(), lbl_epoch_count=_Label(),
        btn_roll_compute=None, btn_compute_epochs=None,
        _slots={"rolling_hrv": _Slot(), "epochs": _Slot()}, txt_rolling=None, txt_epochs=None,
        _set_textbox=lambda box, text, tsv=None: None, _df_to_tsv=lambda df: "",
        tabs=SimpleNamespace(set=lambda name: None),
    )
    app._windowed_peaks = lambda: peaks
    app._start_async_result = lambda btn, label, worker, done: done(worker())
    return app


def _both(peaks: np.ndarray, win_s: float, report: "dict | None" = None
          ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    app = _app(peaks, win_s, report)
    ctrl = AnalysisController(app)
    ctrl.compute_epochs()
    ctrl.compute_rolling_hrv()
    return app.analysis.epoch_df, app.analysis.rolling_hrv_df


def _assert_same(ep: pd.DataFrame, ro: pd.DataFrame, skip: "tuple[str, ...]" = ()) -> None:
    assert len(ep) == len(ro) > 0
    # Same grid: the mid-times differ by rounding only (start and end at 0.1 s).
    assert np.allclose((ep["Epoch_start_s"] + ep["Epoch_end_s"]) / 2, ro["t_mid"], atol=0.11)
    for e_col, r_col in SAME_QUANTITY.items():
        if e_col in skip:
            continue
        np.testing.assert_array_equal(ep[e_col].to_numpy(dtype=float),
                                      ro[r_col].to_numpy(dtype=float), err_msg=e_col)


def _noisy(n: int, seed: int = 3) -> "list[float]":
    return list(120.0 + np.random.default_rng(seed).normal(0.0, 3.0, n))


def test_clean_recording_gives_the_same_numbers():
    ep, ro = _both(rpeaks_from_rr(_noisy(1500), FS), 30.0)
    _assert_same(ep, ro)


def test_pauses_give_the_same_numbers():
    # Long pauses are excluded, and two windows are skipped altogether.
    rr = _noisy(400) + [800.0] + _noisy(400, 4) + [400.0] * 80 + _noisy(400, 5)
    ep, ro = _both(rpeaks_from_rr(rr, FS), 10.0)
    assert ep["N_excluded"].sum() > 0
    _assert_same(ep, ro)


def test_removed_beats_give_the_same_numbers():
    # The interval spanning a removed beat is excluded through the breaks.
    peaks = rpeaks_from_rr(_noisy(1000), FS)
    removed = [int(peaks[300]), int(peaks[640])]
    kept = np.delete(peaks, [300, 640])
    report = {"removed_samples": removed, "break_samples": removed}
    ep, ro = _both(kept, 15.0, report)
    assert ep["N_excluded"].sum() == 2
    _assert_same(ep, ro)


def test_very_short_windows_give_the_same_numbers():
    # At the Epoch floor of 5 s, with slow beats near the floors of both loops.
    ep, ro = _both(rpeaks_from_rr(_noisy(300) + [320.0] * 30 + _noisy(300, 6), FS),
                   MouseECG.EPOCH_MIN_S)
    _assert_same(ep, ro)


def _one_pair_recording() -> np.ndarray:
    """A window holding 3 valid intervals with one adjacent pair: valid, valid,
    excluded, valid. Long stretches around it fall on other windows."""
    before = [120.0] * 50                                    # 6 s
    # Silence up to the grid edge at 10 s: one interval, excluded and on no window.
    gap = [10000.0 - sum(before) + 60.0]
    probe = [125.0, 133.0, 700.0, 121.0]
    fill = [3200.0 - sum(probe) + 2000.0]                    # on to the next window
    after = [120.0] * 60
    return rpeaks_from_rr(before + gap + probe + fill + after, FS)


def test_verdict_rmssd_differs_with_a_single_successive_pair():
    """Why the loops stay separate. On one pair, Epochs reports |difference|
    (hrv_time_domain: any pair) and Sliding NaN (its guard wants two)."""
    ep, ro = _both(_one_pair_recording(), 5.0)
    assert len(ep) == len(ro)
    one_pair = ep["N_valid"].to_numpy() == 3
    assert one_pair.any()
    i = int(np.flatnonzero(one_pair)[0])
    assert ep["RMSSD"].iloc[i] == pytest.approx(8.0, abs=0.01)
    assert np.isnan(ro["RMSSD"].iloc[i])
    # Only that: every other quantity of that recording agrees.
    _assert_same(ep, ro, skip=("RMSSD",))
