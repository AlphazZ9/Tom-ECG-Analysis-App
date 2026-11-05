# -*- coding: utf-8 -*-
"""Without the sidebars: the settings they held live in the Recording sheet,
the detection bar, the Filters popover and the context strip, under the same
attribute names. A session saves and restores them; a session or a rebuild
snapshot from before keeps working."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import ecg
from ecg.core.models import FilterParams
from ecg.ui.context_strip import pending_window_field
from ecg.ui.session_controller import SessionController

UI_DIR = Path(ecg.__file__).parent / "ui"


class _Entry:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def get(self) -> str:
        return self.text

    def delete(self, *_a) -> None:
        self.text = ""

    def insert(self, _i, val: str) -> None:
        self.text = val + self.text

    def configure(self, **_k) -> None:
        pass


class _Switch:
    def __init__(self, on: bool = False) -> None:
        self.on = on

    def get(self) -> int:
        return int(self.on)

    def select(self) -> None:
        self.on = True

    def deselect(self) -> None:
        self.on = False


class _Combo:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def _app(**over):
    methods = []
    app = SimpleNamespace(
        ent_subject=_Entry("subject_01"), ent_sg_window_ms=_Entry("20"),
        ent_project_name=_Entry(""), sw_show_raw=_Switch(True), ui=SimpleNamespace(show_raw=True),
        ent_sg_target_fs=_Entry("10000"), ent_channel=_Entry("ECG"), ent_fs=_Entry("1000"),
        ent_t_start=_Entry("0"), ent_t_end=_Entry("0"), sw_artifact=_Switch(False),
        cb_det_method=_Combo("SG + Derivative (10 kHz)"),
        _on_det_method_change=methods.append, _on_project_name_change=lambda: None)
    for k, v in over.items():
        setattr(app, k, v)
    return app, methods


def test_recording_fields_round_trip():
    src, _ = _app()
    src.ent_subject.text, src.ent_sg_window_ms.text = "mouse_A", "25"
    src.ent_project_name.text = "ProjX"
    src.sw_show_raw.on = False
    fields = SessionController(src).recording_fields()

    dst, _ = _app()
    SessionController(dst).apply_recording_fields(fields)
    assert (dst.ent_subject.text, dst.ent_sg_window_ms.text) == ("mouse_A", "25")
    assert dst.sw_show_raw.on is False and dst.ui.show_raw is False
    assert dst.ent_project_name.text == "ProjX"


def test_the_project_being_worked_under_is_not_replaced():
    dst, _ = _app()
    dst.ent_project_name.text = "Current"
    SessionController(dst).apply_recording_fields({"project_name": "Old"})
    assert dst.ent_project_name.text == "Current"


def test_a_session_without_recording_fields_keeps_the_widgets():
    dst, _ = _app()
    for fields in (None, {}, {"subject": 3, "show_raw": "yes"}):
        SessionController(dst).apply_recording_fields(fields)
    assert dst.ent_subject.text == "subject_01" and dst.sw_show_raw.on is True


def test_detection_widgets_are_restored_from_the_saved_keys_only():
    fp = FilterParams(detection_method="wavelet", sg_target_fs=8000, channel="Ch6", fs=2000,
                      t_start=5.0, t_end=60.0, artifact_correction=True)
    app, methods = _app()
    SessionController(app).restore_detection_widgets(fp, ())       # older session
    assert app.cb_det_method.value == "SG + Derivative (10 kHz)" and methods == []
    assert app.ent_channel.text == "ECG" and app.sw_artifact.on is False

    SessionController(app).restore_detection_widgets(fp, fp.to_dict().keys())
    assert app.cb_det_method.value == "Wavelet (CWT)" and methods == ["Wavelet (CWT)"]
    assert (app.ent_sg_target_fs.text, app.ent_channel.text, app.ent_fs.text) == ("8000", "Ch6", "2000")
    assert (app.ent_t_start.text, app.ent_t_end.text) == ("5", "60")
    assert app.sw_artifact.on is True


def test_every_method_key_has_a_combobox_label():
    for key in ("auto", "sg_derivative", "wavelet", "envelope_max", "ml"):
        label = SessionController._METHOD_LABELS[key]
        assert FilterParams(detection_method=label).to_dict()["detection_method"] == key


def test_a_rebuild_snapshot_from_the_sidebar_layout_is_accepted():
    # restore_ui_state reads no panel flag and needs no toggle method.
    src = (UI_DIR / "session_controller.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "restore_ui_state")
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert not names & {"_toggle_left_panel", "_toggle_right_panel",
                        "left_panel_collapsed", "right_panel_collapsed"}


def test_the_moved_settings_survive_a_rebuild():
    for attr in ("ent_analysis_t0", "ent_analysis_t1", "ent_subject", "ent_project_name",
                 "ent_sg_window_ms", "ent_sg_target_fs", "ent_minrr", "ent_maxrr", "ent_thr"):
        assert attr in SessionController._SNAPSHOT_ENTRIES
    for attr in ("sw_show_raw", "sw_filtering", "sw_notch", "sw_invert_signal", "sw_artifact"):
        assert attr in SessionController._SNAPSHOT_SWITCHES


def test_no_ui_module_refers_to_the_removed_panels():
    gone = ("app_sidebar", "app_right_panel", "ecg.ui.sidebar", "CollapsibleSection",
            "_toggle_left_panel", "_toggle_right_panel", "btn_toggle_right_panel")
    for path in UI_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not [g for g in gone if g in text], path.name


def test_pending_window_is_said_only_when_it_differs():
    assert pending_window_field(None, (0, 0), (10, 20), 300.0) is None
    assert pending_window_field({}, (0, 0), (0, 0), 300.0) is None
    f = pending_window_field({}, (0, 0), (10, 200), 300.0)
    assert f is not None and "Analyze" in f.text and "10–200 s" in f.text
