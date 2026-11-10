"""
Unit tests for ecg.core.detection.

Covers the module's public API: the shared candidate/threshold utilities,
all three R-peak detectors, the RR-artifact detect/apply/correct pipeline,
missed-beat recovery, and rule-based arrhythmia classification.

Detector- and classifier-level scenarios were prototyped interactively
against the real functions before being written here (see the session
notes) -- the exact expected numbers below (peak counts, event kinds,
n_beats, deviations) are what these functions actually return for these
inputs, not hand-derived guesses. Where an interaction between passes
produces a non-obvious result (e.g. a duplicate's short interval NOT also
flagging its real neighbour as "ectopic"), the test says so.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecg.core.detection import (
    active_removed_samples,
    apply_artifact_decisions,
    apply_threshold,
    arrhythmia_baseline_info,
    arrhythmia_baseline_window,
    classify_arrhythmias,
    correct_rr_artifacts,
    detect_peaks_envelope_max,
    detect_peaks_sg_derivative,
    detect_peaks_wavelet,
    detect_rr_artifacts,
    drop_removed_peaks,
    fix_polarity,
    merge_artifact_reports,
    peak_agreement_counts,
    recover_missed_beats,
    resolve_r_vs_j_peaks,
)

from tests.helpers import add_qrs_pulse, make_synth_ecg, match_recall, rpeaks_from_rr


# ════════════════════════════════════════════════════════════
#  Shared candidate / threshold utilities
# ════════════════════════════════════════════════════════════

class TestApplyThreshold:
    def test_rejects_low_prominence_peak(self):
        cands = np.array([100, 220, 340, 460, 580])
        proms = np.array([1.0, 1.0, 0.1, 1.0, 1.0])
        accepted, rejected, thresh_amp = apply_threshold(
            np.zeros(700), cands, proms, thresh_frac=0.5, fs=0)
        assert list(rejected) == [340]
        assert list(accepted) == [100, 220, 460, 580]
        assert isinstance(thresh_amp, float)

    def test_empty_candidates_returns_empty(self):
        accepted, rejected, thresh_amp = apply_threshold(
            np.zeros(10), np.array([], dtype=int), np.array([]), thresh_frac=0.5)
        assert len(accepted) == 0
        assert len(rejected) == 0
        assert thresh_amp == 0.0

    def test_all_equal_prominence_all_accepted(self):
        cands = np.array([10, 20, 30])
        proms = np.array([1.0, 1.0, 1.0])
        accepted, rejected, _ = apply_threshold(
            np.zeros(40), cands, proms, thresh_frac=0.5)
        assert list(accepted) == [10, 20, 30]
        assert len(rejected) == 0


class TestPeakAgreementCounts:
    def test_counts_matches_within_tolerance(self):
        primary = np.array([100, 200, 300, 400])
        other1 = np.array([101, 199, 5000])   # matches primary[0], primary[1]
        other2 = np.array([300, 401])          # matches primary[2], primary[3]
        counts = peak_agreement_counts(primary, [other1, other2], tolerance_samples=5)
        assert list(counts) == [1, 1, 1, 1]

    def test_no_match_beyond_tolerance(self):
        primary = np.array([100])
        other = np.array([200])
        counts = peak_agreement_counts(primary, [other], tolerance_samples=5)
        assert list(counts) == [0]

    def test_empty_primary(self):
        counts = peak_agreement_counts(np.array([]), [np.array([1, 2])], tolerance_samples=5)
        assert len(counts) == 0

    def test_empty_other_array_contributes_zero(self):
        primary = np.array([100, 200])
        counts = peak_agreement_counts(primary, [np.array([])], tolerance_samples=5)
        assert list(counts) == [0, 0]


class TestResolveRVsJPeaks:
    FS = 2000.0

    def test_strips_close_shallow_second_peak(self):
        sig = np.zeros(2000)
        add_qrs_pulse(sig, 500, self.FS, amp=1.0)
        add_qrs_pulse(sig, 520, self.FS, amp=0.3)   # 10ms later, much shallower
        out = resolve_r_vs_j_peaks(np.array([500, 520]), sig, self.FS)
        assert list(out) == [500]

    def test_well_separated_peaks_are_untouched(self):
        peaks = np.array([100, 300, 500, 700])
        out = resolve_r_vs_j_peaks(peaks, np.zeros(1000), self.FS)
        assert np.array_equal(out, peaks)

    def test_fewer_than_two_peaks_is_passthrough(self):
        out = resolve_r_vs_j_peaks(np.array([100]), np.zeros(1000), self.FS)
        assert list(out) == [100]
        out0 = resolve_r_vs_j_peaks(np.array([], dtype=int), np.zeros(1000), self.FS)
        assert len(out0) == 0


class TestFixPolarity:
    FS = 1000.0

    def _clean_positive_signal(self, seed=1):
        sig = np.zeros(4000)
        for i in range(1, 30):
            add_qrs_pulse(sig, i * 120, self.FS, amp=1.0)
        sig += np.random.default_rng(seed).normal(0, 0.01, len(sig))
        return sig

    def test_detects_normal_polarity(self):
        sig = self._clean_positive_signal()
        _, inverted, cands, proms = fix_polarity(sig, self.FS)
        assert inverted is False
        assert len(cands) > 0
        assert len(proms) == len(cands)

    def test_detects_inverted_polarity(self):
        sig = -self._clean_positive_signal()
        _, inverted, cands, _ = fix_polarity(sig, self.FS)
        assert inverted is True
        assert len(cands) > 0

    def test_force_polarity_short_circuits_vote(self):
        sig = self._clean_positive_signal()
        out_false, inverted_false, _, _ = fix_polarity(sig, self.FS, force_polarity=False)
        assert inverted_false is False
        assert np.array_equal(out_false, sig)
        out_true, inverted_true, _, _ = fix_polarity(sig, self.FS, force_polarity=True)
        assert inverted_true is True
        assert np.array_equal(out_true, -sig)


# ════════════════════════════════════════════════════════════
#  R-peak detectors
# ════════════════════════════════════════════════════════════

class TestDetectPeaksSgDerivative:
    def test_finds_all_beats_in_clean_synthetic_signal(self):
        sig, fs, true_peaks = make_synth_ecg(n_beats=40)
        peaks, proms, thresh_amp = detect_peaks_sg_derivative(sig, fs)
        tol = int(0.005 * fs)
        assert match_recall(peaks, true_peaks, tol) == len(true_peaks)
        assert len(peaks) == len(true_peaks)   # no false positives either
        assert len(proms) == len(peaks)
        assert isinstance(thresh_amp, float)

    def test_finds_all_beats_with_added_noise(self):
        sig, fs, true_peaks = make_synth_ecg(n_beats=40, noise_sd=0.08, seed=1)
        peaks, _, _ = detect_peaks_sg_derivative(sig, fs)
        tol = int(0.005 * fs)
        assert match_recall(peaks, true_peaks, tol) == len(true_peaks)

    def test_empty_signal_returns_empty(self):
        peaks, proms, thresh_amp = detect_peaks_sg_derivative(np.array([]), 2000.0)
        assert len(peaks) == 0
        assert len(proms) == 0
        assert thresh_amp == 0.0

    def test_flat_signal_finds_nothing(self):
        peaks, _, _ = detect_peaks_sg_derivative(np.zeros(int(2000 * 5)), 2000.0)
        assert len(peaks) == 0


class TestDetectPeaksWavelet:
    def test_finds_most_beats_in_clean_synthetic_signal(self):
        sig, fs, true_peaks = make_synth_ecg(n_beats=40)
        peaks, proms, thresh_amp = detect_peaks_wavelet(sig, fs)
        tol = int(0.005 * fs)
        matched = match_recall(peaks, true_peaks, tol)
        # The composite-score detector is deliberately conservative (needs
        # >=8 "confident" beats to even build its template descriptor) and
        # is measurably pickier than SG-derivative/envelope-max on a crude
        # synthetic pulse -- 90%+ recall with zero false positives is the
        # realistic bar here, not 100%.
        assert matched >= 0.8 * len(true_peaks)
        assert len(peaks) == matched, "no false-positive peaks expected on a clean signal"
        assert len(proms) == len(peaks)
        assert isinstance(thresh_amp, float)

    def test_empty_signal_returns_empty(self):
        peaks, proms, thresh_amp = detect_peaks_wavelet(np.array([]), 2000.0)
        assert len(peaks) == 0
        assert len(proms) == 0
        assert thresh_amp == 0.0

    def test_flat_signal_finds_nothing(self):
        peaks, _, _ = detect_peaks_wavelet(np.zeros(int(2000 * 5)), 2000.0)
        assert len(peaks) == 0


class TestDetectPeaksEnvelopeMax:
    def test_finds_all_beats_in_clean_synthetic_signal(self):
        sig, fs, true_peaks = make_synth_ecg(n_beats=40)
        peaks, proms, thresh_amp = detect_peaks_envelope_max(sig, fs)
        tol = int(0.005 * fs)
        assert match_recall(peaks, true_peaks, tol) == len(true_peaks)
        assert len(peaks) == len(true_peaks)
        assert len(proms) == len(peaks)
        assert isinstance(thresh_amp, float)

    def test_empty_signal_returns_empty(self):
        peaks, proms, thresh_amp = detect_peaks_envelope_max(np.array([]), 2000.0)
        assert len(peaks) == 0
        assert len(proms) == 0
        assert thresh_amp == 0.0

    def test_flat_signal_finds_nothing(self):
        peaks, _, _ = detect_peaks_envelope_max(np.zeros(int(2000 * 5)), 2000.0)
        assert len(peaks) == 0


class TestDetectorCrossConsistency:
    """All three detectors should broadly agree on a clean synthetic signal --
    a useful regression signal in itself: if one detector's peak set stops
    overlapping the others', something in its calibration likely broke."""

    def test_detectors_agree_on_clean_signal(self):
        sig, fs, true_peaks = make_synth_ecg(n_beats=40)
        tol = int(0.005 * fs)
        p_sg, _, _ = detect_peaks_sg_derivative(sig, fs)
        p_wv, _, _ = detect_peaks_wavelet(sig, fs)
        p_em, _, _ = detect_peaks_envelope_max(sig, fs)
        for name, peaks in (("sg", p_sg), ("wavelet", p_wv), ("envelope", p_em)):
            assert match_recall(peaks, true_peaks, tol) >= 0.8 * len(true_peaks), name


# ════════════════════════════════════════════════════════════
#  RR-artifact detection / correction
# ════════════════════════════════════════════════════════════

class TestDetectRrArtifacts:
    FS = 1000.0

    def test_clean_uniform_rhythm_has_no_artifacts(self):
        rp = rpeaks_from_rr([120.0] * 20, self.FS)
        assert detect_rr_artifacts(rp, self.FS) == []

    def test_fewer_than_four_peaks_returns_empty(self):
        assert detect_rr_artifacts(np.array([1, 2, 3]), self.FS) == []

    def test_duplicate_peak_flagged_and_removed(self):
        # A spurious extra peak 20ms after a real one (< RR_MIN_MS ~67ms) is
        # a classic doublet. The preceding real beat's "next RR" (only 20ms)
        # ends at the duplicate, which Pass 0 already flagged -- so Pass 2
        # (ectopic) treats that short interval as accounted for and does
        # NOT also flag the real beat (it used to, deleting a real R-peak
        # and leaving a 240ms gap after auto-correction).
        rp_clean = rpeaks_from_rr([120.0] * 20, self.FS)
        extra = rp_clean[10] + 20
        rp = np.sort(np.concatenate([rp_clean, [extra]]))

        candidates = detect_rr_artifacts(rp, self.FS)
        kinds = {c["sample"]: c["type"] for c in candidates}
        assert kinds[int(extra)] == "duplicate"
        assert int(rp_clean[10]) not in kinds  # no cascaded ectopic flag

        corrected, report = correct_rr_artifacts(rp, self.FS)
        assert extra not in corrected
        assert np.array_equal(corrected, rp_clean)
        assert report["n_in"] == len(rp)
        assert report["n_out"] == len(corrected)
        assert report["n_duplicate"] == 1

    def test_duplicate_tiebreak_picks_lower_amplitude_peak_as_duplicate(self):
        # Real call sites (analysis_controller.py) always pass `signal=`,
        # which the amplitude-based tie-break in Pass 0 uses to type the
        # smaller (less plausible) R-peak of a close pair as "duplicate" --
        # regardless of which one happens to come first/second in the array.
        # The point of *this* test is specifically the amplitude-based
        # typing; test_duplicate_peak_flagged_and_removed above covers the
        # final surviving set.
        fs = 2000.0
        rp_clean = rpeaks_from_rr([120.0] * 10, fs)
        sig = np.zeros(rp_clean[-1] + 2000)
        for p in rp_clean:
            add_qrs_pulse(sig, p, fs, amp=1.0)
        spurious = rp_clean[5] + int(0.01 * fs)  # 10ms later, much smaller bump
        add_qrs_pulse(sig, spurious, fs, amp=0.15)
        rp = np.sort(np.concatenate([rp_clean, [spurious]]))

        candidates = detect_rr_artifacts(rp, fs, signal=sig)
        kinds = {c["sample"]: c["type"] for c in candidates}
        assert kinds[int(spurious)] == "duplicate"

    def test_isolated_long_interval_flags_nothing(self):
        # One interval +41.7% longer than the local median (170ms among
        # 120ms) with no doublet involved. A long interval alone is never
        # ectopic evidence: removing either flanking beat could only make it
        # longer. It used to flag BOTH real beats around it.
        rp = rpeaks_from_rr([120.0] * 10 + [170.0] + [120.0] * 10, self.FS)
        assert detect_rr_artifacts(rp, self.FS) == []

    def test_premature_beat_flags_only_the_premature_beat(self):
        # Short coupling interval (80ms) then a compensatory pause (160ms):
        # only the beat ENDING the short interval is premature. Neither the
        # normal beat before it (short departure only) nor the one after
        # the compensatory pause (long arrival only) is flagged.
        rr = [120.0] * 10 + [80.0, 160.0] + [120.0] * 10
        rp = rpeaks_from_rr(rr, self.FS)
        candidates = detect_rr_artifacts(rp, self.FS)
        assert len(candidates) == 1
        (cand,) = candidates
        assert cand["type"] == "ectopic"
        assert cand["index"] == 11            # peak ending the 80ms interval
        assert cand["sample"] == int(rp[11])
        assert cand["rr_prev_ms"] == pytest.approx(80.0)
        assert cand["rr_ref_ms"] == pytest.approx(120.0)
        assert cand["deviation"] == pytest.approx((120.0 - 80.0) / 120.0, abs=1e-3)

    def test_sinus_pause_is_not_an_artifact(self):
        # A genuine 400ms pause among 120ms beats (> RR_MAX_MS): no
        # candidate at all, so auto-correction removes nothing and the
        # pause stays visible (it used to delete both flanking beats,
        # turning the pause into a 640ms gap).
        fs = 2000.0
        rp = rpeaks_from_rr([120.0] * 200 + [400.0] + [120.0] * 200, fs)
        assert detect_rr_artifacts(rp, fs) == []

        corrected, report = correct_rr_artifacts(rp, fs)
        assert np.array_equal(corrected, rp)
        assert report["n_in"] == report["n_out"] == len(rp)
        assert report["n_ectopic"] == 0
        assert report["removed_samples"] == []

    @staticmethod
    def _flagged(rr: "list[float]", fs: float = 1000.0) -> "dict[int, str]":
        rp = rpeaks_from_rr(rr, fs)
        return {c["index"]: c["type"] for c in detect_rr_artifacts(rp, fs)}

    def test_premature_couplet_flags_both_premature_beats(self):
        # [80, 80, 160]: beat 11 has both intervals short; beat 12 arrives
        # 80 ms after it, and once beat 11 is gone it would arrive 160 ms
        # after beat 10 -- still not a normal interval -- so it is premature
        # too. It used to survive, leaving its 160 ms compensatory interval
        # in HRV as if it were normal.
        assert self._flagged([120.0] * 10 + [80.0, 80.0, 160.0] + [120.0] * 10) \
            == {11: "ectopic", 12: "ectopic"}

    def test_premature_triplet_flags_every_premature_beat(self):
        assert self._flagged([120.0] * 10 + [80.0, 80.0, 80.0, 200.0] + [120.0] * 10) \
            == {11: "ectopic", 12: "ectopic", 13: "ectopic"}

    def test_spurious_split_does_not_cascade_to_the_next_beat(self):
        # A spurious peak splitting one normal 150 ms interval into 75 + 75:
        # removing it leaves a NORMAL interval, so the beat after it is real.
        assert self._flagged([150.0] * 10 + [75.0, 75.0] + [150.0] * 10) == {11: "ectopic"}

    def test_short_first_interval_blames_beat_zero_without_compensatory_pause(self):
        # Beat 1 departs normally: beat 0 is the spurious leading detection
        # (e.g. a filter-edge transient), not the real beat 1.
        assert self._flagged([80.0] + [120.0] * 20) == {0: "ectopic"}

    def test_short_first_interval_blames_beat_one_before_compensatory_pause(self):
        # Beat 1 is followed by a long (compensatory) interval: it is premature.
        assert self._flagged([80.0, 160.0] + [120.0] * 20) == {1: "ectopic"}

    def test_short_last_interval_blames_the_last_beat(self):
        rr = [120.0] * 20 + [80.0]
        assert self._flagged(rr) == {len(rr): "ectopic"}

    def test_short_last_interval_after_long_arrival_blames_beat_before_last(self):
        # Mirror of the first-interval rule: beat n-2 arrived late (160 ms)
        # and the last beat is back on the 120 ms rhythm (240 ms = 2 RR).
        rr = [120.0] * 20 + [160.0, 80.0]
        assert self._flagged(rr) == {len(rr) - 1: "ectopic"}


class TestActiveRemovedSamples:
    def test_keeps_only_removed_beats_still_absent(self):
        rpeaks = np.array([100, 300, 400, 600])
        # 200 and 500 are still missing; 301 is a re-detected beat (±1 sample)
        out = active_removed_samples([500, 200, 301, 200], rpeaks)
        assert out.tolist() == [200, 500]

    def test_none_or_empty_gives_empty(self):
        assert active_removed_samples(None, np.array([1, 2])).tolist() == []
        assert active_removed_samples([], np.array([1, 2])).tolist() == []


class TestApplyArtifactDecisions:
    def test_removes_only_marked_candidates(self):
        rpeaks = np.array([100, 200, 300, 400, 500])
        candidates = [
            dict(sample=200, type="duplicate", decision="remove"),
            dict(sample=400, type="ectopic", decision="keep"),
        ]
        corrected, report = apply_artifact_decisions(rpeaks, candidates)
        assert list(corrected) == [100, 300, 400, 500]
        assert report["n_in"] == 5
        assert report["n_out"] == 4
        assert report["n_duplicate"] == 1
        assert report["n_ectopic"] == 0
        assert report["n_kept"] == 1
        assert report["removed_samples"] == [200]

    def test_removed_samples_are_sorted_python_ints(self):
        rpeaks = np.array([100, 200, 300, 400, 500])
        candidates = [
            dict(sample=np.int64(400), type="ectopic", decision="remove"),
            dict(sample=200, type="ectopic", decision="remove"),
        ]
        _, report = apply_artifact_decisions(rpeaks, candidates)
        assert report["removed_samples"] == [200, 400]
        assert all(type(s) is int for s in report["removed_samples"])  # JSON-safe

    def test_no_candidates_is_a_passthrough(self):
        rpeaks = np.array([1, 2, 3])
        corrected, report = apply_artifact_decisions(rpeaks, [])
        assert list(corrected) == [1, 2, 3]
        assert report["n_in"] == report["n_out"] == 3
        assert report["removed_samples"] == []


class TestArtifactBreaks:
    """Which removals are HRV breaks: only a removed REAL beat (premature)
    leaves a non-NN interval spanning it; removing a spurious detection
    gives back the true sinus interval."""

    @staticmethod
    def _correct(rr: "list[float]", fs: float = 1000.0):
        rp = rpeaks_from_rr(rr, fs)
        corrected, report = correct_rr_artifacts(rp, fs)
        return rp, corrected, report

    def test_ectopic_candidates_carry_their_long_interval_limit(self):
        rp = rpeaks_from_rr([120.0] * 10 + [80.0, 160.0] + [120.0] * 10, 1000.0)
        (cand,) = detect_rr_artifacts(rp, 1000.0)
        assert cand["rr_long_ms"] == pytest.approx(120.0 * 1.2)

    def test_removed_premature_beat_is_a_break(self):
        rp, _, report = self._correct([120.0] * 10 + [80.0, 160.0] + [120.0] * 10)
        assert report["removed_samples"] == [int(rp[11])]
        assert report["removed_types"] == ["ectopic"]
        assert report["break_samples"] == [int(rp[11])]

    @pytest.mark.parametrize("run", [[80.0, 80.0, 160.0], [80.0, 80.0, 80.0, 200.0]])
    def test_every_beat_of_a_premature_run_is_a_break(self, run):
        _, _, report = self._correct([120.0] * 10 + run + [120.0] * 10)
        assert len(report["removed_samples"]) == len(run) - 1
        assert report["break_samples"] == report["removed_samples"]

    def test_removed_spurious_split_is_not_a_break(self):
        # 75 + 75 ms splitting one 150 ms interval: removing the split peak
        # leaves a NORMAL 150 ms interval -- a true NN interval.
        rp, _, report = self._correct([150.0] * 10 + [75.0, 75.0] + [150.0] * 10)
        assert report["removed_samples"] == [int(rp[11])]
        assert report["n_ectopic"] == 1
        assert report["break_samples"] == []

    def test_removed_duplicate_is_not_a_break(self):
        fs = 1000.0
        rp = rpeaks_from_rr([120.0] * 20, fs)
        doubled = np.sort(np.r_[rp, rp[10] + 30])
        corrected, report = correct_rr_artifacts(doubled, fs)
        assert report["n_duplicate"] == 1
        assert len(report["removed_samples"]) == 1
        assert report["break_samples"] == []

    def test_spurious_leading_detection_is_not_a_break(self):
        # beat 0 removed: no interval spans it
        rp, _, report = self._correct([80.0] + [120.0] * 20)
        assert report["removed_samples"] == [int(rp[0])]
        assert report["break_samples"] == []

    def test_premature_first_beat_before_compensatory_pause_is_a_break(self):
        rp, _, report = self._correct([80.0, 160.0] + [120.0] * 20)
        assert report["break_samples"] == [int(rp[1])]

    def test_without_fs_every_ectopic_removal_is_a_conservative_break(self):
        rp = rpeaks_from_rr([150.0] * 10 + [75.0, 75.0] + [150.0] * 10, 1000.0)
        candidates = detect_rr_artifacts(rp, 1000.0)
        _, report = apply_artifact_decisions(rp, candidates)          # no fs
        assert report["break_samples"] == report["removed_samples"]
        _, report = apply_artifact_decisions(rp, candidates, fs=1000.0)
        assert report["break_samples"] == []


class TestMergeArtifactReports:
    FS = 1000.0

    def _pvc(self):
        rp = rpeaks_from_rr([120.0] * 10 + [80.0, 160.0] + [120.0] * 10, self.FS)
        corrected, report = correct_rr_artifacts(rp, self.FS)
        return rp, corrected, report

    def test_second_pass_on_corrected_peaks_keeps_the_cumulative_counts(self):
        rp, corrected, first = self._pvc()
        again, second = correct_rr_artifacts(corrected, self.FS)
        assert second["n_in"] == second["n_out"]                    # nothing new
        merged = merge_artifact_reports(first, second, again)
        assert (merged["n_in"], merged["n_out"]) == (len(rp), len(corrected))
        assert merged["n_ectopic"] == 1
        assert merged["removed_samples"] == first["removed_samples"]
        assert merged["break_samples"] == first["break_samples"]

    def test_removal_whose_beat_is_back_is_dropped(self):
        rp, _, first = self._pvc()
        # re-detection brought the beat back (one sample off)
        back = np.sort(np.r_[np.delete(rp, 11), rp[11] + 1])
        assert merge_artifact_reports(first, None, back, tol_samples=2) is None
        kept = merge_artifact_reports(None, first, back, tol_samples=2)
        assert kept["removed_samples"] == [] and kept["n_in"] == kept["n_out"]

    def test_reports_combine_and_new_one_wins_for_n_kept(self):
        prev = dict(removed_samples=[200], removed_types=["duplicate"],
                    break_samples=[], n_kept=3)
        new = dict(removed_samples=[500], removed_types=["ectopic"],
                   break_samples=[500], n_kept=1)
        rp = np.array([100, 300, 400, 600])
        merged = merge_artifact_reports(prev, new, rp)
        assert merged["removed_samples"] == [200, 500]
        assert merged["removed_types"] == ["duplicate", "ectopic"]
        assert merged["break_samples"] == [500]
        assert (merged["n_duplicate"], merged["n_ectopic"], merged["n_kept"]) == (1, 1, 1)
        assert (merged["n_in"], merged["n_out"]) == (6, 4)

    def test_report_without_types_or_breaks_carries_no_break(self):
        legacy = dict(n_in=5, n_out=4, removed_samples=[200], n_ectopic=1)
        merged = merge_artifact_reports(legacy, None, np.array([100, 300, 400, 500]))
        assert merged["removed_samples"] == [200]
        assert merged["break_samples"] == []

    def test_nothing_to_merge(self):
        assert merge_artifact_reports(None, None, np.array([1, 2, 3])) is None


class TestDropRemovedPeaks:
    def test_drops_the_nearest_peak_within_tolerance_only(self):
        rp = np.array([100, 300, 302, 500])
        # 302 was removed; the kept 300 is within tolerance too but farther
        assert drop_removed_peaks(rp, [302], tol_samples=5).tolist() == [100, 300, 500]

    def test_far_or_protected_peaks_survive(self):
        rp = np.array([100, 300, 500])
        assert drop_removed_peaks(rp, [400], tol_samples=5).tolist() == [100, 300, 500]
        assert drop_removed_peaks(rp, [300], tol_samples=5,
                                  protected={300}).tolist() == [100, 300, 500]

    def test_nothing_removed(self):
        rp = np.array([3, 1, 2])
        assert drop_removed_peaks(rp, None).tolist() == [1, 2, 3]
        assert drop_removed_peaks(rp, []).tolist() == [1, 2, 3]


class TestRecoverMissedBeats:
    FS = 2000.0

    def test_recovers_single_dropped_beat(self):
        rr_ms = [120.0] * 10 + [240.0] + [120.0] * 10  # one beat missing -> RR doubled
        rp = rpeaks_from_rr(rr_ms, self.FS, start=2000)
        sig = np.zeros(rp[-1] + 2000)
        for p in rp:
            add_qrs_pulse(sig, p, self.FS)
        missing_center = (rp[10] + rp[11]) // 2
        add_qrs_pulse(sig, missing_center, self.FS)
        sig += np.random.default_rng(0).normal(0, 0.01, len(sig))

        rp_out, n_recovered = recover_missed_beats(rp, sig, self.FS)
        assert n_recovered == 1
        new_peaks = [p for p in rp_out if p not in rp]
        assert len(new_peaks) == 1
        assert abs(int(new_peaks[0]) - missing_center) < int(0.01 * self.FS)  # within 10ms
        assert len(rp_out) == len(rp) + 1

    def test_clean_rhythm_recovers_nothing(self):
        rp = rpeaks_from_rr([120.0] * 20, self.FS)
        sig = np.zeros(rp[-1] + 500)
        rp_out, n_recovered = recover_missed_beats(rp, sig, self.FS)
        assert n_recovered == 0
        assert np.array_equal(rp_out, rp)

    def test_fewer_than_four_peaks_returns_unchanged(self):
        rp = np.array([1, 2, 3])
        rp_out, n_recovered = recover_missed_beats(rp, np.zeros(10), self.FS)
        assert n_recovered == 0
        assert np.array_equal(rp_out, rp)


# ════════════════════════════════════════════════════════════
#  Arrhythmia classification
# ════════════════════════════════════════════════════════════

class TestClassifyArrhythmias:
    FS = 1000.0
    BASELINE_S = 2.0  # short baseline window so synthetic signals stay small

    def _events(self, rr_ms, **kw):
        rp = rpeaks_from_rr(rr_ms, self.FS)
        return classify_arrhythmias(rp, self.FS, baseline_s=self.BASELINE_S, **kw)

    def test_fewer_than_five_peaks_returns_no_events(self):
        rp = rpeaks_from_rr([120.0] * 3, self.FS)
        assert classify_arrhythmias(rp, self.FS) == []

    def test_stable_rhythm_has_no_events(self):
        events = self._events([120.0] * 60)
        assert events == []

    def test_sustained_bradycardia(self):
        events = self._events([120.0] * 20 + [170.0] * 20 + [120.0] * 20)
        assert len(events) == 1
        ev = events[0]
        assert ev.kind == "bradycardia"
        assert ev.severity == "warning"
        assert ev.n_beats == 20
        assert ev.rr_mean == pytest.approx(170.0, abs=0.5)
        assert ev.delta_pct < -20.0   # below baseline, past the 20%-default cutoff

    def test_sustained_tachycardia(self):
        events = self._events([120.0] * 20 + [90.0] * 20 + [120.0] * 20)
        assert len(events) == 1
        ev = events[0]
        assert ev.kind == "tachycardia"
        assert ev.severity == "warning"
        assert ev.n_beats == 20
        assert ev.rr_mean == pytest.approx(90.0, abs=0.5)
        assert ev.delta_pct > 20.0

    def test_pause(self):
        # 300ms is > 2x the ~120ms local median AND > the fixed 200ms pause
        # floor (60000/450 * 1.5) -- both conditions required.
        events = self._events([120.0] * 20 + [300.0] + [120.0] * 20)
        assert len(events) == 1
        ev = events[0]
        assert ev.kind == "pause"
        assert ev.severity == "alert"
        assert ev.n_beats == 2
        assert ev.rr_mean == pytest.approx(300.0, abs=0.5)

    def test_short_pause_below_floor_is_not_flagged(self):
        # 190ms: > 2x local median(120ms)=240? No -- 190 < 240, so this
        # shouldn't even reach the 200ms floor check. Confirms the pause
        # rule needs BOTH conditions, not just the fixed 200ms floor.
        events = self._events([120.0] * 20 + [190.0] + [120.0] * 20)
        assert not any(e.kind == "pause" for e in events)

    def test_premature_run(self):
        # 70ms is < 0.70 * ~120ms local median -- 4 consecutive premature
        # beats: a run (>= 3; "esv_run" in older versions).
        events = self._events([120.0] * 20 + [70.0] * 4 + [120.0] * 20)
        assert len(events) == 1
        ev = events[0]
        assert ev.kind == "premature_run"
        assert ev.severity == "alert"
        assert ev.n_beats == 4
        assert ev.rr_mean == pytest.approx(70.0, abs=0.5)

    def test_two_premature_beats_are_a_couplet(self):
        # Only 2 premature beats -- not a run (>= 3) but reported as a
        # couplet (older versions dropped it).
        events = self._events([120.0] * 20 + [70.0] * 2 + [120.0] * 20)
        assert not any(e.kind == "premature_run" for e in events)
        cpl = [e for e in events if e.kind == "premature_couplet"]
        assert len(cpl) == 1 and cpl[0].n_beats == 2 and cpl[0].severity == "warning"

    def test_irregular_run(self):
        # A period-3 oscillation (110,110,180 repeating) whose 5-sample
        # rolling MEDIAN stays close to baseline (so it doesn't also read
        # as brady/tachy) but whose 7-sample MEAN/STD-based CV exceeds the
        # 0.15 irregularity threshold.
        rr = [125.0] * 20 + [110.0, 110.0, 180.0] * 10 + [125.0] * 20
        events = self._events(rr)
        assert len(events) == 1
        ev = events[0]
        assert ev.kind == "irregular_run"
        assert ev.severity == "info"
        assert ev.n_beats >= 7

    def test_av_delay_requires_pr_ms(self):
        rp = rpeaks_from_rr([120.0] * 40, self.FS)
        pr_ms = np.array([30.0] * 15 + [70.0] * 15 + [30.0] * 11)  # len == len(rp) == 41
        assert len(pr_ms) == len(rp)

        events = classify_arrhythmias(rp, self.FS, baseline_s=self.BASELINE_S, pr_ms=pr_ms)
        assert len(events) == 1
        ev = events[0]
        assert ev.kind == "av_delay"
        assert ev.severity == "warning"
        assert ev.n_beats == 15

    def test_av_delay_skipped_without_pr_ms(self):
        rp = rpeaks_from_rr([120.0] * 40, self.FS)
        events = classify_arrhythmias(rp, self.FS, baseline_s=self.BASELINE_S, pr_ms=None)
        assert events == []

    def test_av_delay_skipped_when_pr_ms_length_mismatches_rpeaks(self):
        # classify_arrhythmias trusts positional correspondence but still
        # guards against an obviously-wrong length rather than silently
        # misaligning PR values to the wrong beats.
        rp = rpeaks_from_rr([120.0] * 40, self.FS)
        pr_ms = np.array([70.0] * 40)  # one short of len(rp) == 41
        events = classify_arrhythmias(rp, self.FS, baseline_s=self.BASELINE_S, pr_ms=pr_ms)
        assert events == []

    def test_av_delay_pr_ms_follows_unsorted_rpeaks(self):
        # pr_ms[i] belongs to rpeaks[i] in the CALLER's order -- the peaks
        # are sorted internally, and pr_ms must be reordered with them.
        rp = rpeaks_from_rr([120.0] * 40, self.FS)
        pr_ms = np.array([30.0] * 15 + [70.0] * 15 + [30.0] * 11)
        perm = np.random.default_rng(0).permutation(len(rp))

        events = classify_arrhythmias(rp[perm], self.FS, baseline_s=self.BASELINE_S,
                                      pr_ms=pr_ms[perm])
        assert len(events) == 1
        assert events[0].kind == "av_delay"
        assert events[0].n_beats == 15

    def test_events_sorted_by_start_time(self):
        # bradycardia first, then a pause well after it -- events must come
        # back in chronological order regardless of detection order inside
        # the function.
        rr = [120.0] * 15 + [170.0] * 20 + [120.0] * 15 + [300.0] + [120.0] * 15
        events = self._events(rr)
        starts = [e.t_start for e in events]
        assert starts == sorted(starts)
        assert {e.kind for e in events} == {"bradycardia", "pause"}

    # ── Baseline window is relative to the first analysed peak ──
    # 120 s at 500 bpm (RR 120 ms), then 80 s at 375 bpm (RR 160 ms, -25%).
    SLOWDOWN_RR = [120.0] * 1000 + [160.0] * 500
    SLOWDOWN_FS = 2000.0

    def test_windowed_recording_uses_window_start_as_baseline(self):
        # The app classifies the analysis-window subset of the peaks. With a
        # window starting at 60 s, an absolute "t < 30 s" baseline matched
        # no beat and fell back to the median of everything (140 ms ->
        # ~429 bpm), so the -25% slowdown stayed inside the +/-20% band and
        # went unreported. The baseline must be the first 30 s OF THE
        # WINDOW (60-90 s, all 500 bpm).
        rp = rpeaks_from_rr(self.SLOWDOWN_RR, self.SLOWDOWN_FS, start=0)
        win = rp[rp / self.SLOWDOWN_FS >= 60.0]
        events = classify_arrhythmias(win, self.SLOWDOWN_FS, baseline_s=30.0)

        brady = [e for e in events if e.kind == "bradycardia"]
        assert len(brady) == 1
        assert not any(e.kind == "tachycardia" for e in events)
        ev = brady[0]
        assert ev.baseline_hr == pytest.approx(500.0, abs=2.0)
        assert ev.t_start == pytest.approx(120.0, abs=1.0)
        assert ev.hr_mean == pytest.approx(375.0, abs=2.0)
        assert ev.delta_pct == pytest.approx(-25.0, abs=1.0)

        bl_t0, bl_t1 = arrhythmia_baseline_window(win, self.SLOWDOWN_FS, 30.0)
        assert bl_t0 == pytest.approx(60.0, abs=0.2)
        assert bl_t1 == pytest.approx(90.0, abs=0.2)

    def test_recording_starting_at_zero_is_unchanged(self):
        # Relative == absolute when the first peak sits at t=0, so a full
        # recording keeps exactly the pre-fix result: baseline from 0-30 s.
        rp = rpeaks_from_rr(self.SLOWDOWN_RR, self.SLOWDOWN_FS, start=0)
        events = classify_arrhythmias(rp, self.SLOWDOWN_FS, baseline_s=30.0)
        assert [e.kind for e in events] == ["bradycardia"]
        ev = events[0]
        assert ev.baseline_hr == pytest.approx(500.0, abs=2.0)
        assert ev.t_start == pytest.approx(120.0, abs=1.0)
        assert ev.n_beats == 500
        assert arrhythmia_baseline_window(rp, self.SLOWDOWN_FS, 30.0) == \
            pytest.approx((0.0, 30.0))

    def test_time_shift_only_shifts_events(self):
        # Prepending dead time before the first peak (e.g. a window that
        # starts later in the file) must not change any classification --
        # every event is identical apart from its times moving by the shift.
        rr = [120.0] * 300 + [160.0] * 100 + [120.0] * 100 + [300.0] + [120.0] * 50
        fs = self.SLOWDOWN_FS
        rp0 = rpeaks_from_rr(rr, fs, start=0)
        shift_s = 75.0
        rp1 = rp0 + int(shift_s * fs)
        ev0 = classify_arrhythmias(rp0, fs, baseline_s=30.0)
        ev1 = classify_arrhythmias(rp1, fs, baseline_s=30.0)
        assert {e.kind for e in ev0} >= {"bradycardia", "pause"}
        assert len(ev1) == len(ev0)
        for a, b in zip(ev0, ev1):
            assert b.kind == a.kind
            assert b.n_beats == a.n_beats
            assert b.baseline_hr == a.baseline_hr
            assert b.delta_pct == a.delta_pct
            assert b.t_start == pytest.approx(a.t_start + shift_s)
            assert b.t_end == pytest.approx(a.t_end + shift_s)

    def test_baseline_fallback_widens_from_first_peak(self):
        # Fewer than 10 RR intervals in baseline_s -> the window widens to
        # 20% of the analysed span, still measured from the first peak.
        # 300 beats at 120 ms = 36 s span; baseline_s=1 s holds 9 intervals.
        rp = rpeaks_from_rr([120.0] * 300, self.FS, start=0) + int(50.0 * self.FS)
        bl_t0, bl_t1 = arrhythmia_baseline_window(rp, self.FS, baseline_s=1.0)
        assert bl_t0 == pytest.approx(50.0)
        assert bl_t1 == pytest.approx(50.0 + 0.20 * 36.0)

    # ── Baseline covering most of a short analysed window ──
    def test_short_window_flags_the_baseline_unreliable(self, caplog):
        # A 25 s window (10 s at 500 bpm, then 15 s at 375 bpm) with a 30 s
        # baseline: the baseline is the median of the whole window, event
        # included, so the normal part comes out as "tachycardia". The call
        # still runs, but the baseline is flagged and a warning logged.
        rp = rpeaks_from_rr(self.SLOWDOWN_RR, self.SLOWDOWN_FS, start=0)
        t = rp / self.SLOWDOWN_FS
        win = rp[(t >= 110.0) & (t <= 135.0)]

        info = arrhythmia_baseline_info(win, self.SLOWDOWN_FS, 30.0)
        assert info["reliable"] is False
        assert info["span_s"] < 2 * info["length_s"]
        assert info["baseline_hr"] == pytest.approx(375.0, abs=2.0)

        with caplog.at_level("WARNING", logger="ecg"):
            events = classify_arrhythmias(win, self.SLOWDOWN_FS, baseline_s=30.0)
        rate_events = [e for e in events if e.kind in ("bradycardia", "tachycardia")]
        assert rate_events
        assert all(e.baseline_reliable is False for e in rate_events)
        # the shaded baseline is the one the events were compared against
        assert all(e.baseline_hr == pytest.approx(info["baseline_hr"], abs=0.1)
                   for e in rate_events)
        assert any("unreliable" in r.getMessage() for r in caplog.records)

    def test_long_window_keeps_a_reliable_baseline(self):
        rp = rpeaks_from_rr(self.SLOWDOWN_RR, self.SLOWDOWN_FS, start=0)
        win = rp[rp / self.SLOWDOWN_FS >= 60.0]       # 140 s span, 30 s baseline

        info = arrhythmia_baseline_info(win, self.SLOWDOWN_FS, 30.0)
        assert info["reliable"] is True
        assert (info["t_start"], info["t_end"]) == \
            pytest.approx(arrhythmia_baseline_window(win, self.SLOWDOWN_FS, 30.0))
        events = classify_arrhythmias(win, self.SLOWDOWN_FS, baseline_s=30.0)
        assert [e.kind for e in events] == ["bradycardia"]
        assert events[0].baseline_reliable is True

    def test_span_of_exactly_two_baselines_is_reliable(self):
        # 60 s at 120 ms with a 30 s baseline: the reliability limit
        # (span >= 2 x baseline window) is inclusive.
        rp = rpeaks_from_rr([120.0] * 500, self.FS, start=0)
        assert arrhythmia_baseline_info(rp, self.FS, 30.0)["reliable"] is True
        assert arrhythmia_baseline_info(rp, self.FS, 31.0)["reliable"] is False

    def test_baseline_info_with_too_few_peaks(self):
        info = arrhythmia_baseline_info(np.array([0, 120, 240, 360]), self.FS, 30.0)
        assert info["reliable"] is False
        assert np.isnan(info["t_start"]) and np.isnan(info["baseline_hr"])
