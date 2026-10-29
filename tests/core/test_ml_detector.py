import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ecg.core import ml_detector


def test_grouped_holdout_keeps_recordings_separate_and_classes_present():
    labels = np.tile(np.array([0, 1], dtype=int), 40)
    groups = np.repeat(np.array(["recording-a", "recording-b", "recording-c", "recording-d"]), 20)

    train_idx, test_idx = ml_detector._grouped_holdout_indices(labels, groups)

    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    assert set(np.unique(labels[train_idx])) == {0, 1}
    assert set(np.unique(labels[test_idx])) == {0, 1}


def test_grouped_holdout_requires_more_than_one_recording():
    labels = np.tile(np.array([0, 1], dtype=int), 40)
    groups = np.repeat("one-recording", len(labels))

    with pytest.raises(ValueError, match="At least two"):
        ml_detector._grouped_holdout_indices(labels, groups)


def test_model_safe_format_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("skops.io")
    labels = np.tile(np.array([0, 1], dtype=int), 20)
    features = np.arange(len(labels), dtype=float).reshape(-1, 1)
    estimator = RandomForestClassifier(n_estimators=4, random_state=1).fit(features, labels)
    model_path = tmp_path / "model.skops"

    monkeypatch.setattr(ml_detector, "MODEL_PATH", model_path)
    monkeypatch.setattr(ml_detector, "MODEL_BACKUP_PATH", tmp_path / "model.skops.bak")
    monkeypatch.setattr(ml_detector, "META_PATH", tmp_path / "meta.json")
    monkeypatch.setattr(ml_detector, "META_BACKUP_PATH", tmp_path / "meta.json.bak")
    monkeypatch.setitem(ml_detector._model_cache, "mtime", None)
    monkeypatch.setitem(ml_detector._model_cache, "model", None)

    ml_detector.MLPeakModel(estimator, {"model_type": "RandomForestClassifier"}).save()
    loaded = ml_detector.MLPeakModel.load()

    assert loaded is not None
    assert model_path.exists()
    np.testing.assert_allclose(loaded.estimator.predict_proba(features), estimator.predict_proba(features))


def test_model_load_rejects_unapproved_skops_types(tmp_path, monkeypatch):
    skops_io = pytest.importorskip("skops.io")
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"placeholder")
    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"model_type": "RandomForestClassifier"}', encoding="utf-8")
    monkeypatch.setattr(ml_detector, "MODEL_PATH", model_path)
    monkeypatch.setattr(ml_detector, "META_PATH", meta_path)
    monkeypatch.setattr(skops_io, "get_untrusted_types", lambda **kwargs: ["unexpected.Type"])
    monkeypatch.setattr(skops_io, "load", lambda **kwargs: pytest.fail("unsafe loader was called"))
    monkeypatch.setitem(ml_detector._model_cache, "mtime", None)
    monkeypatch.setitem(ml_detector._model_cache, "model", None)

    assert ml_detector.MLPeakModel.load() is None


def test_model_load_ignores_legacy_joblib_file(tmp_path, monkeypatch):
    legacy_path = tmp_path / "legacy.joblib"
    legacy_bytes = b"legacy pickle payload"
    legacy_path.write_bytes(legacy_bytes)
    monkeypatch.setattr(ml_detector, "MODEL_PATH", tmp_path / "missing.skops")
    monkeypatch.setattr(ml_detector, "META_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(ml_detector, "LEGACY_MODEL_PATH", legacy_path)
    monkeypatch.setitem(ml_detector._model_cache, "mtime", None)
    monkeypatch.setitem(ml_detector._model_cache, "model", None)

    assert ml_detector.MLPeakModel.load() is None
    assert legacy_path.read_bytes() == legacy_bytes