"""
Mapping analyse_intervals() rows back onto R peaks.

analyse_intervals() returns one row per beat whose full window fits in
the signal: beats too close to the signal edges are dropped, so its columns
can never be indexed positionally against the peak array.
These tests pin align_interval_column_to_peaks()'s sample-index matching and
the end-to-end AV-delay path it unblocks (Intervals PR -> classify_arrhythmias).
"""
import numpy as np
import pandas as pd
import pytest

from ecg.core.analysis import align_interval_column_to_peaks, analyse_intervals
from ecg.core.detection import classify_arrhythmias
from ecg.core.wave_template import WaveTemplate

from tests.helpers import make_synth_ecg, rpeaks_from_rr


@pytest.fixture(autouse=True)
def _no_template_writes(monkeypatch):
    """analyse_intervals() never saves a template (older versions re-saved
    it to the user's home profile); this makes any accidental save fail
    loudly instead of touching the real profile."""
    def _refuse(self):
        raise AssertionError("WaveTemplate.save() must not run in tests")
    monkeypatch.setattr(WaveTemplate, "save", _refuse)


def _confirmed_template() -> WaveTemplate:
    wt = WaveTemplate()
    wt.confirmed = True
    return wt


def _intervals(sig: np.ndarray, rp: np.ndarray, fs: float) -> pd.DataFrame:
    rr = np.diff(rp).astype(float) / fs * 1000
    return analyse_intervals(sig, rp, fs, rr, wave_template=_confirmed_template())


# ════════════════════════════════════════════════════════════
#  align_interval_column_to_peaks
# ════════════════════════════════════════════════════════════

@pytest.mark.parametrize("crop", [0, 150])
def test_aligned_pr_matches_analyse_intervals_rows(crop):
    # crop=150 puts the first peak inside the beat half-window from the
    # signal start, so analyse_intervals drops it as an edge beat -- the
    # rows then start at the SECOND peak (and still run to the last one).
    sig, fs, rp = make_synth_ecg(n_beats=60, p_amp=0.15)
    sig, rp = sig[crop:], rp - crop
    df = _intervals(sig, rp, fs)
    assert len(df) == len(rp) - (1 if crop else 0)
    assert df["PR_ms"].notna().any()

    aligned = align_interval_column_to_peaks(df, rp, fs)

    assert aligned.shape == (len(rp),)
    assert aligned.dtype == float
    # Every finite aligned value is the PR of the row sitting on that peak.
    row_peak = np.searchsorted(rp, np.round(df["R_peak_s"].to_numpy() * fs).astype(int))
    np.testing.assert_array_equal(aligned[row_peak], df["PR_ms"].to_numpy())
    others = np.setdiff1d(np.arange(len(rp)), row_peak)
    assert np.isnan(aligned[others]).all()
    # An edge-dropped first peak has no row.
    if crop:
        assert np.isnan(aligned[0])
    first_row = 1 if crop else 0
    np.testing.assert_array_equal(
        aligned[first_row:first_row + len(df)], df["PR_ms"].to_numpy())


def test_peaks_shifted_beyond_tolerance_align_to_nothing():
    sig, fs, rp = make_synth_ecg(n_beats=60)
    df = _intervals(sig, rp, fs)

    aligned = align_interval_column_to_peaks(df, rp + 50, fs)

    assert aligned.shape == (len(rp),)
    assert np.isnan(aligned).all()


def test_one_sample_offset_is_tolerated_two_are_not():
    fs = 1000.0
    rp = rpeaks_from_rr([120.0] * 10, fs)
    df = pd.DataFrame({"R_peak_s": rp[:-1] / fs, "PR_ms": np.arange(10, dtype=float)})

    near = align_interval_column_to_peaks(df, rp + 1, fs)
    far = align_interval_column_to_peaks(df, rp + 2, fs)

    np.testing.assert_array_equal(near[:-1], np.arange(10, dtype=float))
    assert np.isnan(far).all()


def test_rejected_rows_align_to_nan():
    fs = 1000.0
    rp = rpeaks_from_rr([120.0] * 10, fs)
    df = pd.DataFrame({
        "R_peak_s": rp[:-1] / fs,
        "PR_ms":    np.full(10, 40.0),
        "accepted": [True] * 3 + [False] * 2 + [True] * 5,
    })

    aligned = align_interval_column_to_peaks(df, rp, fs)

    assert np.isnan(aligned[3:5]).all()
    np.testing.assert_array_equal(aligned[[0, 1, 2, 5, 6, 7, 8, 9]], 40.0)
    assert np.isnan(aligned[-1])


def test_result_follows_the_callers_peak_order():
    fs = 1000.0
    rp = rpeaks_from_rr([120.0] * 5, fs)            # 6 peaks
    df = pd.DataFrame({"R_peak_s": rp[:-1] / fs, "PR_ms": [30.0, 31.0, 32.0, 33.0, 34.0]})
    shuffled = rp[[3, 0, 5, 1, 4, 2]]

    aligned = align_interval_column_to_peaks(df, shuffled, fs)

    np.testing.assert_array_equal(aligned[[0, 1, 3, 4, 5]], [33.0, 30.0, 31.0, 34.0, 32.0])
    assert np.isnan(aligned[2])                      # rp[5], the last peak


@pytest.mark.parametrize("ivl", [
    None,
    pd.DataFrame(),
    pd.DataFrame({"RR_ms": [120.0, 120.0]}),                 # analyse_intervals' early-return shape
    pd.DataFrame({"R_peak_s": [1.0, 1.12], "QT_ms": [50.0, 50.0]}),  # no PR_ms column
])
def test_missing_data_gives_all_nan(ivl):
    rp = rpeaks_from_rr([120.0] * 5, 1000.0)

    aligned = align_interval_column_to_peaks(ivl, rp, 1000.0)

    assert aligned.shape == (len(rp),)
    assert np.isnan(aligned).all()


def test_invalid_sampling_rate_is_rejected():
    rp = rpeaks_from_rr([120.0] * 5, 1000.0)
    df = pd.DataFrame({"R_peak_s": rp[:-1] / 1000.0, "PR_ms": np.full(5, 40.0)})

    with pytest.raises(ValueError, match="Sampling rate"):
        align_interval_column_to_peaks(df, rp, 0.0)


# ════════════════════════════════════════════════════════════
#  End to end: Intervals-shaped PR -> AV-delay event
# ════════════════════════════════════════════════════════════

def test_aligned_intervals_pr_triggers_av_delay():
    fs = 1000.0
    rp = rpeaks_from_rr([120.0] * 40, fs)            # 41 peaks
    # analyse_intervals keeps one row per peak whose beat window fits in the
    # signal, so its frame can be shorter than the peak list (edge beats
    # dropped). Here one row fewer than there are peaks stands for any such
    # length mismatch.
    pr = np.array([40.0] * 15 + [70.0] * 15 + [40.0] * 10)
    ivl = pd.DataFrame({"R_peak_s": rp[:-1] / fs, "PR_ms": pr,
                        "accepted": np.ones(40, dtype=bool)})
    assert len(ivl) == len(rp) - 1

    # The raw column, passed positionally, is length-mismatched and the
    # AV-delay check is skipped -- the bug that kept it dead in the app.
    raw_events = classify_arrhythmias(rp, fs, baseline_s=2.0, pr_ms=ivl["PR_ms"].to_numpy())
    assert not any(e.kind == "av_delay" for e in raw_events)

    aligned = align_interval_column_to_peaks(ivl, rp, fs)
    assert len(aligned) == len(rp)
    events = classify_arrhythmias(rp, fs, baseline_s=2.0, pr_ms=aligned)

    av = [e for e in events if e.kind == "av_delay"]
    assert len(av) == 1
    assert av[0].n_beats == 15
    assert av[0].t_start == pytest.approx(rp[15] / fs)
