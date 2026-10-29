from ecg.core import wave_template
from ecg.io import session
from ecg.ui import theme


def test_legacy_pickle_session_is_quarantined_without_deserializing(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)
    source = tmp_path / "recording.mat"
    session_path = session._session_path(str(source))
    session_path.write_bytes(b"\x80\x04legacy pickle payload")

    assert session.load_session(str(source)) is None
    assert not session_path.exists()
    assert session_path.with_suffix(session_path.suffix + ".stale").exists()


def test_legacy_pickle_wave_template_is_ignored_and_preserved(tmp_path, monkeypatch):
    json_path = tmp_path / "wave-template.json"
    legacy_path = tmp_path / "wave-template.pkl"
    legacy_path.write_bytes(b"\x80\x04legacy pickle payload")
    monkeypatch.setattr(wave_template, "WAVE_TEMPLATE_PATH", json_path)
    monkeypatch.setattr(wave_template, "_WAVE_TEMPLATE_PATH_LEGACY", legacy_path)

    loaded = wave_template.WaveTemplate.load()

    assert loaded.landmarks == wave_template.WaveTemplate.DEFAULTS
    assert not loaded.confirmed
    assert legacy_path.exists()
    assert not json_path.exists()


def test_legacy_pickle_theme_is_ignored_and_preserved(tmp_path, monkeypatch):
    json_path = tmp_path / "theme.json"
    legacy_path = tmp_path / "theme.pkl"
    legacy_path.write_bytes(b"\x80\x04legacy pickle payload")
    monkeypatch.setattr(theme, "THEME_PATH", json_path)
    monkeypatch.setattr(theme, "_THEME_PATH_LEGACY", legacy_path)
    monkeypatch.setattr(theme, "_detect_system_font", lambda: "Arial")

    loaded = theme.ThemeConfig.load()

    assert loaded.preset_name == "Light"
    assert legacy_path.exists()
    assert not json_path.exists()