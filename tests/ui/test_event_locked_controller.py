"""
EventLockedController (Time course > Event-locked) without a Tk window
-- a stub app stands in for ECGApp: computing needs an Autonomic result and
annotations; the result is stored, drawn and disclosed; it is dropped with
the Autonomic result (peak change, recompute, new file) and when the
annotations it was locked to change; the cohort registry receives the rows
with the subject / project / configuration (throw-away database under
tmp_path); the CSV export and the calibration's Apply write where they
should; the parameters survive the session / rebuild snapshots.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ecg.core.autonomic import COV_METHODS
from ecg.io import db
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import event_locked_controller as elc_module
from ecg.ui import session_controller as sc_module
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.event_locked_controller import EventLockedController
from ecg.ui.session_controller import SessionController
from ecg.ui.state import AnalysisState, DetectionState, SessionState, SignalState, UIState
from ecg.ui.theme import GREEN, MUTED, ORANGE, RED

from tests.helpers import rpeaks_from_rr

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

    def delete(self, *_a) -> None:
        self.value = ""

    def insert(self, _i, text) -> None:
        self.value = str(text)


class _Combo:
    def __init__(self, value: str, values=()) -> None:
        self.value, self.values = value, list(values)

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value

    def configure(self, **kw) -> None:
        self.values = list(kw.get("values", self.values))


class _Check:
    def __init__(self, on: bool = False) -> None:
        self.on = on

    def get(self) -> int:
        return int(self.on)

    def select(self) -> None:
        self.on = True

    def deselect(self) -> None:
        self.on = False


class _Slot:
    def __init__(self) -> None:
        self._draw_fn = None
        self.placeholder_shown = 0

    def update(self, fn) -> None:
        self._draw_fn = fn

    def _show_placeholder(self) -> None:
        self.placeholder_shown += 1


def _safe_float(widget, default):
    try:
        return float(widget.get()) if widget is not None else default
    except (TypeError, ValueError):
        return default


def _peaks(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rr = 120 + rng.normal(0, 1.5, 2000)
    rr[1000:] -= 8.0
    return rpeaks_from_rr(list(rr), FS)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A throw-away registry under tmp_path (never the user's profile)."""
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(db, "SESSION_DIR", session_dir)
    monkeypatch.setattr(db, "_DB_PATH", session_dir / "ecg_registry.db")
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_schema_retry_at", 0.0)
    return session_dir / "ecg_registry.db"


def _stub_app(peaks=None, *, filepath="C:/data/mouse_7.mat", subject="M7",
              project="DobuStudy", annotations=None):
    statuses: "list[tuple[str, object]]" = []
    peaks = _peaks() if peaks is None else peaks
    app = SimpleNamespace(
        analysis=AnalysisState(annotations=list(annotations or [])),
        detection=DetectionState(rpeaks_ok=peaks),
        signal=SignalState(fs=FS, filepath=filepath),
        ui=UIState(), session=SessionState(),
        after=lambda ms, fn=None: None,
        _set_status=lambda text, color=None: statuses.append((text, color)),
        _set_progress=lambda pct, msg: None,
        _safe_float=_safe_float, _generation=0,
        ent_minrr=_Entry(66.7), ent_maxrr=_Entry(333.3),
        ent_auto_win=_Entry("3"), cb_auto_method=_Combo(COV_METHODS["robust"]),
        ent_auto_ks=_Entry("3"), ent_auto_kp=_Entry("12"), chk_auto_lfhf=_Check(False),
        btn_auto_compute=None, lbl_auto_status=_Label(), txt_autonomic=object(),
        cb_evt_label=_Combo("All", ["All"]), ent_evt_pre=_Entry("30"),
        ent_evt_post=_Entry("30"), chk_evt_z=_Check(True), lbl_evt_status=_Label(),
        txt_event_locked=object(),
        ent_subject=_Entry(subject), ent_project_name=_Entry(project),
        cb_freq_band=None,
        _slots={"autonomic": _Slot(), "event_locked": _Slot()},
        _df_to_tsv=lambda df: "TSV:" + ",".join(map(str, df.columns)) + f":{len(df)}",
    )
    app.textboxes = {}
    app._set_textbox = lambda box, text, tsv=None: app.textboxes.__setitem__(id(box), (text, tsv))
    app._windowed_peaks = lambda: app.detection.rpeaks_ok
    app._start_async_result = lambda btn, label, worker, done: done(worker())
    app.analysis_ctrl = AnalysisController(app)
    app.event_ctrl = EventLockedController(app)
    return app, statuses


def _event_time(app) -> float:
    return float(app.detection.rpeaks_ok[1000]) / FS


def _computed(annotations="default", **kw):
    app, statuses = _stub_app(**kw)
    app.analysis_ctrl.compute_autonomic()
    t = _event_time(app)
    if annotations == "default":
        annotations = [{"t_start": t, "t_end": t + 5, "label": "dobutamine", "color": "#a0a"},
                       {"t_start": 60.0, "t_end": 61.0, "label": "saline", "color": "#0a0"},
                       {"t_start": 1.0, "t_end": 2.0, "label": "", "color": "#00a"}]
    app.analysis.annotations = list(annotations)
    app.event_ctrl.refresh_label_choices()
    return app, statuses


# ── compute ───────────────────────────────────────────────────────────────

def test_compute_needs_an_autonomic_result_then_annotations():
    app, statuses = _stub_app()
    app.event_ctrl.compute()
    assert app.analysis.event_locked is None
    assert "Autonomic indices first" in app.lbl_evt_status.text
    assert app.lbl_evt_status.text_color == ORANGE
    app.analysis_ctrl.compute_autonomic()
    app.event_ctrl.compute()
    assert app.analysis.event_locked is None and "No annotation" in app.lbl_evt_status.text
    app.analysis.annotations = [{"t_start": 50.0, "t_end": 51.0, "label": "x"}]
    app.cb_evt_label.set("dobutamine")
    app.event_ctrl.compute()
    assert "labelled “dobutamine”" in app.lbl_evt_status.text


def test_compute_stores_draws_and_discloses():
    app, statuses = _computed()
    assert app.cb_evt_label.values == ["All", "dobutamine", "saline", "(unlabelled)"]
    app.event_ctrl.compute()
    el = app.analysis.event_locked
    assert el is not None and el.n_events == 3 and el.n_valid == 2
    assert app.analysis.event_locked_df is el.table
    assert app._slots["event_locked"]._draw_fn is not None
    text, tsv = app.textboxes[id(app.txt_event_locked)]
    assert "WITHIN-RECORDING" in text and "not population inference" in text
    assert tsv.startswith("TSV:label,onset_s,valid") and tsv.endswith(":3")
    assert app.lbl_evt_status.text_color == ORANGE and "1 invalid" in app.lbl_evt_status.text
    assert statuses[-1][0].startswith("Event-locked —")
    assert app.analysis.event_locked_params == {"label": "All", "pre_s": 30.0,
                                                "post_s": 30.0, "zscore": True}
    # the label filter
    app.cb_evt_label.set("dobutamine")
    app.event_ctrl.compute()
    el = app.analysis.event_locked
    assert el.n_events == 1 and el.n_valid == 1 and app.lbl_evt_status.text_color == GREEN
    assert el.table.loc[0, "hr_delta"] > 20


def test_invalid_window_is_refused_and_the_zscore_toggle_only_redraws():
    app, _ = _computed()
    app.ent_evt_post.insert(0, "0")
    app.event_ctrl.compute()
    assert app.analysis.event_locked is None and app.lbl_evt_status.text_color == RED
    assert "Post" in app.lbl_evt_status.text
    app.ent_evt_post.insert(0, "30")
    app.event_ctrl.compute()
    el, fn = app.analysis.event_locked, app._slots["event_locked"]._draw_fn
    app.chk_evt_z.deselect()
    app.event_ctrl.on_zscore_toggle()
    assert app.analysis.event_locked is el
    assert app._slots["event_locked"]._draw_fn is not fn
    assert app.analysis.event_locked_params["zscore"] is False


# ── invalidation ──────────────────────────────────────────────────────────

def test_dropped_with_the_autonomic_result_even_after_the_plots_were_reset():
    app, _ = _computed()
    app.event_ctrl.compute()
    slot = app._slots["event_locked"]
    slot._draw_fn = None                     # _reset_result_plots() ran first
    app.analysis_ctrl.invalidate_autonomic()
    assert app.analysis.event_locked is None and app.analysis.event_locked_df is None
    assert "Peaks changed" in app.lbl_evt_status.text
    app.analysis_ctrl.invalidate_autonomic()             # a second call keeps the notice
    assert "Peaks changed" in app.lbl_evt_status.text
    app.analysis_ctrl.invalidate_autonomic(peaks_changed=False)       # new file
    assert app.lbl_evt_status.text == EventLockedController.IDLE_TEXT
    assert app.lbl_evt_status.text_color == MUTED


def test_a_recomputed_autonomic_result_drops_the_event_locked_one():
    app, _ = _computed()
    app.event_ctrl.compute()
    app.ent_auto_ks.insert(0, "5")
    app.analysis_ctrl.compute_autonomic()
    assert app.analysis.autonomic_result.params["ks"] == 5.0
    assert app.analysis.event_locked is None
    assert "recomputed" in app.lbl_evt_status.text
    assert app.textboxes[id(app.txt_event_locked)][0] == ""


def test_annotation_changes_drop_a_stale_result_only():
    app, _ = _computed()
    app.event_ctrl.compute()
    el = app.analysis.event_locked
    app.analysis.annotations[0]["color"] = "#123456"            # cosmetic
    app.analysis.annotations[0]["t_end"] += 3.0
    app.event_ctrl.on_annotations_changed()
    assert app.analysis.event_locked is el
    app.analysis.annotations.append({"t_start": 90.0, "t_end": 91.0, "label": "new"})
    app.event_ctrl.on_annotations_changed()
    assert app.analysis.event_locked is None
    assert "Annotations changed" in app.lbl_evt_status.text
    assert "new" in app.cb_evt_label.values
    # a label-filtered result ignores annotations of other labels far from
    # its events ...
    app.cb_evt_label.set("dobutamine")
    app.event_ctrl.compute()
    el = app.analysis.event_locked
    t_ev = el.events[0][0]
    app.analysis.annotations.append({"t_start": t_ev + 80.0, "t_end": t_ev + 81.0,
                                     "label": "saline"})
    app.event_ctrl.on_annotations_changed()
    assert app.analysis.event_locked is el
    assert t_ev + 80.0 in el.other_onsets                       # tracked for the next edit
    # ... but not one inside an event's windows: it bounds the condition
    # (isolation), so the event is no longer valid.
    app.analysis.annotations.append({"t_start": t_ev + 12.0, "t_end": t_ev + 13.0,
                                     "label": "saline"})
    app.event_ctrl.on_annotations_changed()
    assert app.analysis.event_locked is None
    app.event_ctrl.compute()
    el = app.analysis.event_locked
    assert not bool(el.table["valid"].iloc[0]) and not bool(el.table["isolated"].iloc[0])
    assert el.table["next_gap_s"].iloc[0] == pytest.approx(12.0)
    assert "another annotation" in app.lbl_evt_status.text
    app.analysis.annotations.pop()
    app.event_ctrl.on_annotations_changed()
    assert app.analysis.event_locked is None
    app.event_ctrl.compute()
    el = app.analysis.event_locked
    # a deleted label falls back to All in the filter
    app.analysis.annotations = [a for a in app.analysis.annotations if a["label"] != "dobutamine"]
    app.event_ctrl.on_annotations_changed()
    assert app.cb_evt_label.get() == "All" and app.analysis.event_locked is None


# ── registry / CSV ────────────────────────────────────────────────────────

def test_save_to_registry_with_identity_and_configuration(registry):
    app, statuses = _computed()
    assert app.event_ctrl.save_to_registry() == 0                  # nothing computed
    app.event_ctrl.compute()
    assert app.event_ctrl.save_to_registry() == 3
    rows = db.list_event_rows(valid_only=False)
    assert {r["subject"] for r in rows} == {"M7"} and {r["project_name"] for r in rows} == {"DobuStudy"}
    assert {(r["method"], r["window_s"]) for r in rows} == {("robust", 3.0)}
    assert rows[0]["params"]["ks"] == 3.0 and rows[0]["params"]["pre_s"] == 30.0
    assert set(rows[0]["metrics"]) == {"hr", "d", "sd1", "sd2", "csi", "cpi"}
    assert statuses[-1][1] == GREEN and "Saved 3 event(s)" in statuses[-1][0]
    # "All" replaces this recording's events of the configuration ...
    app.analysis.annotations.pop()                                   # drop the 1 s event
    app.event_ctrl.on_annotations_changed()
    app.event_ctrl.compute()
    assert app.event_ctrl.save_to_registry() == 2
    assert len(db.list_event_rows(valid_only=False)) == 2
    # ... one label only upserts
    app.cb_evt_label.set("dobutamine")
    app.event_ctrl.compute()
    assert app.event_ctrl.save_to_registry() == 1
    assert len(db.list_event_rows(valid_only=False)) == 2


def test_save_without_subject_warns_and_without_file_refuses(registry):
    app, statuses = _computed(subject="")
    app.event_ctrl.compute()
    assert app.event_ctrl.save_to_registry() == 3
    assert statuses[-1][1] == ORANGE and "No Subject ID" in statuses[-1][0]
    app2, _ = _computed(filepath=None)
    app2.event_ctrl.compute()
    assert app2.event_ctrl.save_to_registry() == 0
    assert app2.lbl_evt_status.text_color == RED


def test_csv_export_uses_the_file_dialog(tmp_path, monkeypatch):
    app, statuses = _computed()
    app.event_ctrl.compute()
    target = tmp_path / "out.csv"
    seen = {}

    def _ask(**kw):
        seen.update(kw)
        return str(target)
    monkeypatch.setattr(elc_module.filedialog, "asksaveasfilename", _ask)
    assert app.event_ctrl.export_csv() == 3
    assert seen["initialfile"].startswith("M7_event_locked_")
    head = target.read_text(encoding="utf-8-sig").splitlines()[0]
    assert head.startswith("id,filepath,stem,subject,project_name,label")
    assert statuses[-1][1] == GREEN
    monkeypatch.setattr(elc_module.filedialog, "asksaveasfilename", lambda **kw: "")
    assert app.event_ctrl.export_csv() == 0                      # cancelled


# ── calibration / parameters ──────────────────────────────────────────────

def test_apply_gains_writes_the_entries_without_recomputing():
    app, statuses = _computed()
    res = app.analysis.autonomic_result
    assert app.event_ctrl.current_gains() == (3.0, 12.0)
    assert app.event_ctrl.current_config() == ("robust", 3.0)
    app.event_ctrl.apply_gains(4.0, 9.0)
    assert (app.ent_auto_ks.get(), app.ent_auto_kp.get()) == ("4", "9")
    assert app.analysis.autonomic_result is res
    assert "not applied yet" in statuses[-1][0] and statuses[-1][1] == ORANGE
    assert "⟳ Compute" in app.lbl_auto_status.text
    assert app.event_ctrl.current_gains() == (4.0, 9.0)


def test_parameters_survive_the_session_and_the_rebuild_snapshot():
    app, _ = _computed()
    app.ent_evt_pre.insert(0, "20")
    app.chk_evt_z.deselect()
    app.cb_evt_label.set("saline")
    p = app.event_ctrl.widget_params()
    assert p == {"label": "saline", "pre_s": 20.0, "post_s": 30.0, "zscore": False}
    app.ent_evt_pre.insert(0, "bad")                      # half-typed: keeps the state
    assert app.event_ctrl.widget_params()["pre_s"] == 30.0
    text = json.dumps({"event_locked_params": p}, cls=_SessionEncoder)
    back = json.loads(text, object_hook=_session_decode_hook)["event_locked_params"]
    app2, _ = _computed()
    applied = app2.event_ctrl.apply_params(back)
    assert applied == p
    assert (app2.ent_evt_pre.get(), app2.chk_evt_z.on, app2.cb_evt_label.get()) == ("20", False, "saline")
    # a session saved before the panel existed changes nothing
    assert app2.event_ctrl.apply_params(None) == p


def test_session_round_trip_restores_the_panel_parameters(monkeypatch):
    app, _ = _computed()
    app.ent_evt_post.insert(0, "25")
    app.cb_evt_label.set("saline")
    app.sl_thr = None
    app._qtc_formula = lambda: "mitchell"
    state = SessionController(app).collect_session_state()
    assert state["event_locked_params"] == {"label": "saline", "pre_s": 30.0,
                                            "post_s": 25.0, "zscore": True}
    state = json.loads(json.dumps(state, cls=_SessionEncoder), object_hook=_session_decode_hook)
    monkeypatch.setattr(sc_module, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(1000), "ECG", None, FS))
    app._compute_preview_bundle = lambda sig, fs, params: {}
    bundle = SessionController(app).restore_session_worker(state)
    fresh, _ = _computed()
    fresh.event_ctrl.apply_params(bundle["event_locked_params"])
    assert fresh.analysis.event_locked_params == state["event_locked_params"]
    assert (fresh.ent_evt_post.get(), fresh.cb_evt_label.get()) == ("25", "saline")
    # a session saved before the panel existed: nothing changes
    del state["event_locked_params"]
    bundle = SessionController(app).restore_session_worker(state)
    assert bundle["event_locked_params"] is None
    assert fresh.event_ctrl.apply_params(None) == fresh.analysis.event_locked_params


# ── review fixes: default Subject ID, theme colours ──────────────────────

def test_saving_under_the_default_subject_id_warns(registry):
    app, statuses = _computed(subject="subject_01", filepath="C:/data/mouse_A.mat")
    app.event_ctrl.compute()
    assert app.event_ctrl.save_to_registry() == 3
    text, color = statuses[-1]
    assert color == ORANGE and "looks like the app's default" in text
    # a second recording saved under the same (real-looking) ID: also flagged
    app2, statuses2 = _computed(subject="M7", filepath="C:/data/mouse_B.mat")
    app2.event_ctrl.compute()
    app2.event_ctrl.save_to_registry()
    assert statuses2[-1][1] == GREEN
    app3, statuses3 = _computed(subject="M7", filepath="C:/data/mouse_C.mat")
    app3.event_ctrl.compute()
    app3.event_ctrl.save_to_registry()
    assert statuses3[-1][1] == ORANGE and "mouse_B.mat" in statuses3[-1][0]


def test_status_colours_follow_a_theme_switch(monkeypatch):
    # The controller reads the theme module at call time, so a light / dark
    # switch (theme.apply_theme_config rebinding GREEN, ORANGE ...) applies
    # without a re-sync of this module.
    from ecg.ui import theme
    app, statuses = _computed()
    monkeypatch.setattr(theme, "GREEN", "#010203")
    monkeypatch.setattr(theme, "ORANGE", "#040506")
    app.event_ctrl.compute()
    assert app.lbl_evt_status.text_color in ("#010203", "#040506")
    app.event_ctrl.invalidate("annotations")
    assert app.lbl_evt_status.text_color == "#040506"
