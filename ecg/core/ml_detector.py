"""
ecg.core.ml_detector
─────────────────────
Classical-ML R-peak detector, trained from recordings the user has manually
reviewed and marked "verified for training".

Two-stage design, matching how the other three detectors in detection.py
report results:
    1. A fixed, permissive candidate generator (scipy.signal.find_peaks with
       a low prominence floor) — used identically at training-label time and
       at inference time, so the classifier always sees the same candidate
       distribution regardless of which detector produced the recording's
       original review pass.
    2. Per-candidate feature extraction, reusing detection.py's existing pure
       morphology helpers (_topographic_prominences, _extended_morphology_
       descriptors) plus a few new features local to this module (raw
       amplitude, inter-candidate timing).
    3. A trained scikit-learn classifier scores each candidate; accepted
       candidates become the returned R-peaks.

Model persistence mirrors ecg.core.wave_template.WaveTemplate's save()/load()
pattern: a small JSON metadata sidecar (source/training-set size/hold-out
accuracy) plus a binary artifact (joblib, since a JSON-only format can't hold
a fitted sklearn estimator). load() never raises -- any missing or corrupt
file is treated as "no model trained yet", not an error.

Training data itself is never a raw signal on disk: marking a file "verified"
extracts features+labels for it immediately (signal and accepted R-peaks are
already in memory in the running app) into a small per-file .npz cache under
SESSION_DIR / "ml_training". Training just pools every cached file and fits a
classifier -- it never needs to reopen or re-filter the original recordings.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.signal import find_peaks

from ecg.core.models import MouseECG, SESSION_DIR
from ecg.core.detection import (
    _topographic_prominences, _extended_morphology_descriptors, _prominence_wlen_samples,
)

log = logging.getLogger("ecg")

MODEL_PATH: Path = Path.home() / ".ecg_ml_detector.joblib"
META_PATH:  Path = Path.home() / ".ecg_ml_detector.json"
TRAINING_DATA_DIR: Path = SESSION_DIR / "ml_training"

# Candidate within this many ms of an accepted R-peak counts as a positive
# training example.
LABEL_TOLERANCE_MS = 5.0

# find_peaks' `distance` constraint already bounds candidate count to
# roughly duration_ms / PEAK_DISTANCE_MS (~180k for a 2h recording at the
# app's default 40ms spacing) -- this cap is a defensive backstop, not the
# primary control, in case that assumption is ever violated (e.g. a custom
# PEAK_DISTANCE_MS override, or fs/duration combinations not anticipated
# here). Tripping it means something is genuinely degenerate about the
# input, not just "a long recording" -- see detect_peaks_ml's docstring.
MAX_CANDIDATES = 300_000

FEATURE_NAMES: "list[str]" = [
    "prominence", "amplitude",
    "width25", "width50", "width75",
    "curvature", "width_height_ratio", "slope_up", "slope_down",
    "dt_prev_ms", "dt_next_ms", "local_density",
]


# ════════════════════════════════════════════════════════════
#  CANDIDATE GENERATION + FEATURES
# ════════════════════════════════════════════════════════════

def _generate_candidates(signal: np.ndarray, fs: float) -> np.ndarray:
    """Permissive, fixed candidate generator.

    Deliberately not tied to whichever of the three existing detectors the
    user happened to use for a given file's initial review -- that varies
    file to file. A single, consistent generator keeps the feature
    distribution the classifier sees the same at training-label time and at
    inference time.

    Raises RuntimeError if the signal contains NaN/Inf: np.ptp() propagates
    NaN into the prominence floor, and scipy's prominence search around
    non-finite samples is undefined behaviour, not just "slow" -- this was
    observed to make the whole pipeline hang for minutes on a real
    recording instead of failing fast, so it's rejected explicitly here
    rather than silently sanitised (silently dropping/replacing bad
    samples would hide whatever upstream filtering step produced them).
    RuntimeError (not ValueError) specifically because signal_controller's
    ML dispatch branch catches RuntimeError to surface a clear message to
    the user instead of silently returning 0 peaks.
    """
    if len(signal) == 0:
        return np.array([], dtype=int)
    n_bad = int(np.count_nonzero(~np.isfinite(signal)))
    if n_bad:
        raise RuntimeError(
            f"Signal contains {n_bad:,} non-finite value(s) (NaN/Inf) -- "
            "cannot generate ML candidates. This usually means a filter "
            "step produced invalid output; check the channel/filter "
            "settings for this recording.")
    distance = max(1, int(round(MouseECG.PEAK_DISTANCE_MS / 1000.0 * fs)))
    sig_range = float(np.ptp(signal))
    prominence_floor = max(1e-9, 0.03 * sig_range)
    # wlen bounds the prominence search -- without it, this call is a real
    # catastrophic-slowdown risk on raw (unfiltered) signal with baseline
    # wander: observed ~130s -> ~0.1s (identical peak set) on a 95-min
    # recording. See MouseECG.PROMINENCE_WLEN_MS.
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence_floor,
                           wlen=_prominence_wlen_samples(fs))
    peaks = peaks.astype(int)
    if len(peaks) > MAX_CANDIDATES:
        raise RuntimeError(
            f"ML candidate generation produced {len(peaks):,} candidates, "
            f"over the {MAX_CANDIDATES:,} safety cap -- this points to a "
            "degenerate signal (e.g. a very flat/low-amplitude recording "
            "collapsing the prominence floor) rather than a normal long "
            "recording. Try a different detector, or check the signal "
            "quality for this file.")
    return peaks


def extract_features(signal: np.ndarray, candidates: np.ndarray, fs: float) -> np.ndarray:
    """Return an (n_candidates, len(FEATURE_NAMES)) array, column order fixed."""
    candidates = np.asarray(candidates, dtype=int)
    n = len(candidates)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)))

    prominences = _topographic_prominences(signal, candidates, fs)
    morpho = _extended_morphology_descriptors(candidates, signal, fs, prominences)
    amplitude = signal[candidates].astype(float)

    t_ms = candidates.astype(float) / fs * 1000.0
    # Inter-candidate timing. Edge candidates (no previous/next) fall back to
    # the array's own median spacing rather than 0/inf, so they don't look
    # spuriously "crowded" or "isolated" just for sitting at the boundary.
    fallback = float(np.median(np.diff(t_ms))) if n > 1 else 0.0
    dt_prev = np.full(n, fallback)
    dt_next = np.full(n, fallback)
    if n > 1:
        dt_prev[1:] = np.diff(t_ms)
        dt_next[:-1] = np.diff(t_ms)

    # Local candidate density within +/-50ms (excluding self) -- vectorised
    # via searchsorted since `candidates` (and therefore t_ms) is already
    # sorted ascending by find_peaks.
    window_ms = 50.0
    lo_idx = np.searchsorted(t_ms, t_ms - window_ms, side="left")
    hi_idx = np.searchsorted(t_ms, t_ms + window_ms, side="right")
    local_density = (hi_idx - lo_idx - 1).astype(float)

    return np.column_stack([
        prominences, amplitude,
        morpho["width25"], morpho["width50"], morpho["width75"],
        morpho["curvature"], morpho["width_height_ratio"],
        morpho["slope_up"], morpho["slope_down"],
        dt_prev, dt_next, local_density,
    ])


def _label_candidates(candidates: np.ndarray, rpeaks_ok: np.ndarray, fs: float) -> np.ndarray:
    """Binary label per candidate: 1 if within LABEL_TOLERANCE_MS of an
    accepted R-peak, else 0.
    """
    n = len(candidates)
    if n == 0:
        return np.array([], dtype=int)
    rpeaks_sorted = np.sort(np.asarray(rpeaks_ok, dtype=int)) if len(rpeaks_ok) else np.array([], dtype=int)
    if len(rpeaks_sorted) == 0:
        return np.zeros(n, dtype=int)
    if len(rpeaks_sorted) == 1:
        min_dist = np.abs(candidates - rpeaks_sorted[0])
    else:
        idx = np.searchsorted(rpeaks_sorted, candidates)
        idx = np.clip(idx, 1, len(rpeaks_sorted) - 1)
        dist_left  = np.abs(candidates - rpeaks_sorted[idx - 1])
        dist_right = np.abs(candidates - rpeaks_sorted[idx])
        min_dist = np.minimum(dist_left, dist_right)
    tol_samples = max(1, int(round(LABEL_TOLERANCE_MS / 1000.0 * fs)))
    return (min_dist <= tol_samples).astype(int)


# ════════════════════════════════════════════════════════════
#  MODEL PERSISTENCE  (mirrors ecg.core.wave_template.WaveTemplate)
# ════════════════════════════════════════════════════════════

class MLPeakModel:
    """A trained classifier + its metadata.

    load() never raises -- any missing/corrupt model is reported as "no
    model trained yet" (returns None), the same never-raise contract
    WaveTemplate.load() follows for its own persisted file.
    """

    def __init__(self, estimator, meta: dict) -> None:
        self.estimator = estimator
        self.meta = meta

    def save(self) -> None:
        import joblib
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, MODEL_PATH)
        with open(META_PATH, "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2)
        log.info("MLPeakModel saved -> %s", MODEL_PATH)

    @classmethod
    def load(cls) -> "Optional[MLPeakModel]":
        if not MODEL_PATH.exists() or not META_PATH.exists():
            return None
        try:
            import joblib
            estimator = joblib.load(MODEL_PATH)
            with open(META_PATH, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            return cls(estimator, meta)
        except Exception as exc:
            log.warning("MLPeakModel.load failed: %s -- treating as untrained", exc)
            return None

    @staticmethod
    def exists() -> bool:
        return MODEL_PATH.exists() and META_PATH.exists()


# ════════════════════════════════════════════════════════════
#  DETECTOR — same return convention as the other 3 detectors
# ════════════════════════════════════════════════════════════

def detect_peaks_ml(
    signal: np.ndarray,
    fs: float,
    model: "Optional[MLPeakModel]" = None,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> "tuple[np.ndarray, np.ndarray, float]":
    """R-peak detection via a trained classifier.

    Returns (peaks, prominences, thresh_amp) -- identical convention to
    detect_peaks_sg_derivative/_wavelet/_envelope_max, so this plugs into
    the same dispatcher.

    Raises RuntimeError if no trained model is available (mirrors
    _require_nk()'s pattern in analysis.py) -- the UI layer is responsible
    for catching this and showing a clear "not trained yet" message rather
    than silently falling back to another detector. Also raises (via
    _generate_candidates) on non-finite input or a pathological candidate
    count -- both surface through the same UI pathway as "no model yet",
    rather than grinding silently: a real recording was observed to hang
    for 600s+ with the previous static "scoring candidates…" message and
    no candidate-count visibility, indistinguishable from a normal but
    slow run until it was far too late to tell.

    *progress_cb*, if given, reports real stage checkpoints -- notably the
    candidate count as soon as it's known, which is the single most useful
    number for telling "this is just a big recording" apart from "this
    signal is degenerate" while a run is still in progress.
    """
    def _prog(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    if model is None:
        model = MLPeakModel.load()
    if model is None:
        raise RuntimeError(
            "No trained ML model available. Mark a few recordings "
            "“Verified for training” and use Train / Retrain Model first."
        )

    _prog(5, "Generating candidates…")
    candidates = _generate_candidates(signal, fs)
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([]), 0.0

    _prog(30, f"{len(candidates):,} candidates — extracting features…")
    feats = extract_features(signal, candidates, fs)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    _prog(80, f"Classifying {len(candidates):,} candidates…")
    proba = model.estimator.predict_proba(feats)[:, 1]
    peaks = np.sort(candidates[proba >= 0.5])

    _prog(95, "Finalising…")
    prominences = _topographic_prominences(signal, peaks, fs) if len(peaks) else np.array([])
    thresh_amp = float(np.percentile(prominences, 10)) if len(prominences) else 0.0
    return peaks, prominences, thresh_amp


# ════════════════════════════════════════════════════════════
#  TRAINING DATA CAPTURE
# ════════════════════════════════════════════════════════════

def save_training_sample(
    fingerprint: str, signal: np.ndarray, fs: float,
    rpeaks_ok: np.ndarray,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> int:
    """Extract features+labels for one verified recording and cache them.

    Called once, immediately, when the user marks the currently-open file
    "verified for training" -- signal and rpeaks_ok are already in memory,
    so this never needs to reopen or re-filter anything later.

    *progress_cb*, if given, is called with coarse (pct, message) stage
    checkpoints -- feature extraction has no natural per-item loop to
    report finer-grained progress from (it's vectorised, see
    extract_features), so callers driving a UI progress bar off this
    should rely on the caller-side heartbeat pulse to fill the gaps.

    Returns the number of labeled candidates saved (0 if the signal produced
    no candidates at all).
    """
    def _prog(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    _prog(10, "Generating candidates…")
    candidates = _generate_candidates(signal, fs)
    if len(candidates) == 0:
        return 0
    _prog(40, f"Extracting features ({len(candidates):,} candidates)…")
    feats  = extract_features(signal, candidates, fs)
    _prog(80, "Labeling against corrected R-peaks…")
    labels = _label_candidates(candidates, rpeaks_ok, fs)

    _prog(95, "Writing training cache…")
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAINING_DATA_DIR / f"{fingerprint}.npz"
    np.savez(out_path, features=feats, labels=labels)
    log.info("ML training sample saved: %d candidates (%d positive) -> %s",
              len(candidates), int(labels.sum()), out_path)
    return int(len(candidates))


def delete_training_sample(fingerprint: str) -> None:
    """Remove a cached training sample (e.g. the user un-marks a file)."""
    p = TRAINING_DATA_DIR / f"{fingerprint}.npz"
    if p.exists():
        p.unlink()


def training_data_summary() -> dict:
    """Count cached files + total/positive samples, for the training dialog."""
    if not TRAINING_DATA_DIR.exists():
        return {"n_files": 0, "n_samples": 0, "n_positive": 0}
    n_samples = 0
    n_positive = 0
    files = list(TRAINING_DATA_DIR.glob("*.npz"))
    for f in files:
        try:
            with np.load(f) as data:
                n_samples += len(data["labels"])
                n_positive += int(data["labels"].sum())
        except Exception as exc:
            log.warning("training_data_summary: could not read %s: %s", f, exc)
    return {"n_files": len(files), "n_samples": n_samples, "n_positive": n_positive}


def list_training_files() -> "list[dict]":
    """Return per-file cached-sample stats, most recently cached first.

    Used by the training dialog's file list -- callers cross-reference
    "fingerprint" against the sqlite registry to show a human-readable
    filename (this module has no notion of filenames, only fingerprints).
    """
    if not TRAINING_DATA_DIR.exists():
        return []
    out = []
    files = sorted(TRAINING_DATA_DIR.glob("*.npz"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            with np.load(f) as data:
                labels = data["labels"]
                out.append({
                    "fingerprint": f.stem,
                    "n_samples": int(len(labels)),
                    "n_positive": int(labels.sum()),
                })
        except Exception as exc:
            log.warning("list_training_files: could not read %s: %s", f, exc)
    return out


# ════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════

def train_model(min_samples: int = 50) -> dict:
    """Train (or retrain) the classifier from every cached verified file.

    Never raises -- always returns a dict the UI can display directly:
        {"ok": bool, "message": str, ...}
    and on success additionally: n_files, n_samples, accuracy, f1.
    """
    files = list(TRAINING_DATA_DIR.glob("*.npz")) if TRAINING_DATA_DIR.exists() else []
    if not files:
        return {"ok": False, "message":
                "No verified recordings yet. Mark at least one file "
                "“Verified for training” first."}

    X_parts: "list[np.ndarray]" = []
    y_parts: "list[np.ndarray]" = []
    for f in files:
        try:
            with np.load(f) as data:
                X_parts.append(data["features"])
                y_parts.append(data["labels"])
        except Exception as exc:
            log.warning("train_model: skipping unreadable %s: %s", f, exc)
    if not X_parts:
        return {"ok": False, "message": "Cached training files could not be read."}

    X = np.nan_to_num(np.concatenate(X_parts, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate(y_parts, axis=0)

    if len(y) < min_samples:
        return {"ok": False, "message":
                f"Only {len(y)} labeled candidates across {len(files)} file(s) -- "
                f"need at least {min_samples}. Verify a few more recordings."}
    if len(np.unique(y)) < 2:
        return {"ok": False, "message":
                "Training data has only one class (all peaks or all "
                "non-peaks) -- cannot train a classifier yet."}

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        return {"ok": False, "message": "scikit-learn is required -- pip install scikit-learn joblib"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)

    def _make_clf() -> "RandomForestClassifier":
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=0, n_jobs=-1)

    clf = _make_clf()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    # Hold-out split above is only to report an honest accuracy/F1 estimate;
    # the deployed model is refit on all available labeled data.
    clf_final = _make_clf()
    clf_final.fit(X, y)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_training_files": len(files),
        "n_training_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "feature_names": FEATURE_NAMES,
        "model_type": "RandomForestClassifier",
        "holdout_accuracy": round(accuracy, 4),
        "holdout_f1": round(f1, 4),
    }
    MLPeakModel(clf_final, meta).save()

    return {
        "ok": True,
        "message": f"Trained on {len(y)} labeled candidates from {len(files)} file(s).",
        "n_files": len(files), "n_samples": int(len(y)),
        "accuracy": accuracy, "f1": f1,
    }
