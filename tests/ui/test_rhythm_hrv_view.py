# -*- coding: utf-8 -*-
"""
The Rhythm & HRV page and the Analyze chain behind it:

- one text per metric across KPI cards, metric table, spectrum legend and
  Poincaré title;
- the curated table, what Show all adds, and what a missing module says;
- Analyze runs Frequency, then Non-linear, then auto epochs; a result landing
  after the file, peaks or window changed is not stored; the tab reads computing
  meanwhile; a restored session does not run anything;
- the entropy cap is part of the state a UI rebuild keeps.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from ecg.core.analysis import analyse_core, analyse_hrv_freq, analyse_hrv_nonlinear
from ecg.core.metric_table import (
    MISSING, MetricTable, build_metric_table, format_metric, metric_text, rhythm_cards,
    rhythm_rows, nonlinear_basis_text, asymmetry_text,
)
from ecg.ui import analysis_controller as ac_module
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.context_strip import tab_states
from ecg.ui.freshness import COMPUTING, NOT_COMPUTED, UP_TO_DATE, FreshnessService
from ecg.ui.plot_controller import PlotController
from ecg.ui.session_controller import SessionController, _TAB_KEY_MAP
from ecg.ui.state import AnalysisState, DetectionState, SignalState, UIState
from ecg.ui.tab_keys import TAB_DETECTION, TAB_HRV, TAB_MODULES
from ecg.ui.tabbar import STATE_GLYPHS

from tests.helpers import add_qrs_pulse, rpeaks_from_rr


class _Slot:
    def __init__(self):
        self.draw_fn = None

    def update(self, fn):
        self.draw_fn = fn


class _Label:
    def __init__(self, text=""):
        self.text, self.text_color = text, None

    def configure(self, **kw):
        self.text = kw.get("text", self.text)
        self.text_color = kw.get("text_color", self.text_color)

    def cget(self, key):
        return getattr(self, key)


class _Entry:
    def __init__(self, value):
        self.value = str(value)

    def get(self):
        return self.value

    def delete(self, *_a):
        self.value = ""

    def insert(self, _i, value):
        self.value = str(value)


@pytest.fixture(scope="module")
def recording():
    fs = 1000.0
    rng = np.random.default_rng(11)
    rp = rpeaks_from_rr(list(120 + rng.normal(0, 2.0, 700)), fs)
    sig = np.zeros(int(rp[-1]) + int(fs))
    for p in rp:
        add_qrs_pulse(sig, int(p), fs, amp=1.0)
    r = analyse_core(sig, rp, fs)
    r["hrv_freq"] = analyse_hrv_freq(rp, fs)
    r["hrv_nonlin"] = analyse_hrv_nonlinear(rp, fs)
    return sig, fs, rp, r


def _fig_texts(draw_fn) -> str:
    fig = Figure()
    draw_fn(fig)
    return " ".join(t.get_text() for t in fig.findobj(Text))


def _plot_app(r, fs):
    app = SimpleNamespace(
        analysis=AnalysisState(results=r, analysed_window=(0.0, 0.0)),
        detection=DetectionState(),
        ui=UIState(),
        _slots={"psd": _Slot(), "poincare": _Slot()},
        _df_to_text=lambda df, display_map=None: "",
        _df_to_tsv=lambda df: "",
        _set_textbox=lambda *a, **k: None,
        _NL_DISPLAY={}, txt_nl=None,
    )
    return app


# ── one text per metric ──────────────────────────────────────────────────

def test_every_view_shows_the_same_string(recording):
    _sig, fs, _rp, r = recording
    t = build_metric_table(r)
    app = _plot_app(r, fs)
    ctrl = PlotController(app)
    ctrl.plot_psd(r)
    ctrl.plot_nonlinear(r)
    legend = _fig_texts(app._slots["psd"].draw_fn)
    title = _fig_texts(app._slots["poincare"].draw_fn)
    cards = {c.key: c for c in rhythm_cards(t)}
    rows = {row.key: row for row in rhythm_rows(t)}

    sd1, sd2 = metric_text(t, "hrv_nonlin.HRV_SD1"), metric_text(t, "hrv_nonlin.HRV_SD2")
    assert f"SD1={sd1}  SD2={sd2}" in title
    assert cards["sd1_sd2"].note == f"{sd1} / {sd2} ms"
    assert rows["hrv_nonlin.HRV_SD1"].value == sd1 and rows["hrv_nonlin.HRV_SD2"].value == sd2

    lfhf = metric_text(t, "hrv_freq.HRV_LFHF")
    assert f"LF/HF {lfhf}" in legend
    assert cards["lf_hf"].value == lfhf and rows["hrv_freq.HRV_LFHF"].value == lfhf

    for band in ("VLF", "LF", "HF"):
        text = metric_text(t, f"hrv_freq.HRV_{band}")
        assert f"({text} ms²" in legend
        assert rows[f"hrv_freq.HRV_{band}"].value == text
    lfn, hfn = metric_text(t, "hrv_freq.HRV_LFn"), metric_text(t, "hrv_freq.HRV_HFn")
    assert f"{lfn} n.u." in legend and f"{hfn} n.u." in legend
    assert cards["lf_hf"].note == f"LF {lfn} · HF {hfn} n.u."
    assert rows["hrv_freq.HRV_LFn"].value == lfn

    for card, key in (("sdnn", "hrv_time.HRV_SDNN"), ("rmssd", "hrv_time.HRV_RMSSD"),
                      ("pnn6", "hrv_time.HRV_pNN6")):
        assert cards[card].value == rows[key].value == metric_text(t, key)


def test_screen_text_rests_on_the_raw_value_only(recording):
    _sig, _fs, _rp, r = recording
    t = build_metric_table(r)
    # The table keeps raw values: n.u. is a display scale.
    assert t.value("hrv_freq.HRV_LFn") == float(r["hrv_freq"]["HRV_LFn"].values[0])
    assert metric_text(t, "hrv_freq.HRV_LFn") == f"{t.value('hrv_freq.HRV_LFn') * 100:.1f}"
    assert format_metric("hrv_freq.HRV_LF", float("nan")) == MISSING
    assert format_metric("hrv_freq.HRV_LF", None) == MISSING


def test_hr_note_states_the_percentile_range_the_code_uses(recording):
    _sig, _fs, _rp, r = recording
    t = build_metric_table(r)
    note = next(c for c in rhythm_cards(t) if c.key == "hr").note
    assert note == f"2–98 %: {r['hr']['min']:.0f}–{r['hr']['max']:.0f}"


# ── curated table / Show all ─────────────────────────────────────────────

def test_human_and_raw_metrics_are_behind_show_all(recording):
    _sig, _fs, _rp, r = recording
    t = build_metric_table(r)
    curated = {row.key for row in rhythm_rows(t)}
    every = rhythm_rows(t, show_all=True)
    labels = {row.label for row in every}
    assert "hrv_time.HRV_pNN6" in curated
    assert "hrv_time.HRV_pNN20" not in curated and "hrv_time.HRV_pNN50" not in curated
    assert "hrv_nonlin.HRV_ApEn" not in curated
    assert {"pNN20 (human)", "pNN50 (human)", "ApEn"} <= labels
    assert not any(label.startswith("HRV_") for label in labels)
    assert len(every) > len(curated)
    assert all(row.basis or row.tier == "all" for row in every)


def test_every_curated_row_has_a_unit_or_a_basis(recording):
    _sig, _fs, _rp, r = recording
    rows = {row.key: row for row in rhythm_rows(build_metric_table(r))}
    assert rows["hrv_time.HRV_SDNN"].unit == "ms"
    assert rows["hrv_freq.HRV_LF"].unit == "ms²"
    assert rows["hrv_freq.HRV_LFn"].unit == "n.u."
    assert rows["hrv_time.HRV_RMSSD"].basis.endswith("successive valid RR pairs")
    assert rows["hrv_freq.HRV_LF"].basis.startswith("Welch, 0.4–1.5 Hz")
    assert "valid RR intervals" in rows["hrv_nonlin.HRV_SampEn"].basis


def test_missing_modules_show_a_dash_and_why(recording):
    _sig, _fs, _rp, r = recording
    fd = pd.DataFrame()
    fd.attrs["withheld_reason"] = "2.1 s run of excluded RR intervals"
    t = build_metric_table({**r, "hrv_freq": fd, "hrv_nonlin": pd.DataFrame()})
    rows = {row.key: row for row in rhythm_rows(t)}
    assert rows["hrv_freq.HRV_LF"].value == MISSING
    assert "withheld" in rows["hrv_freq.HRV_LF"].basis
    assert rows["hrv_nonlin.HRV_SD1"].value == MISSING
    assert rows["hrv_nonlin.HRV_SD1"].basis == "not computed"
    cards = {c.key: c for c in rhythm_cards(t)}
    assert "withheld" in cards["lf_hf"].note and cards["sd1_sd2"].note == "not computed"


def test_an_empty_table_is_all_dashes():
    t = MetricTable()
    assert {c.value for c in rhythm_cards(t)} == {MISSING}
    assert {row.value for row in rhythm_rows(t)} == {MISSING}
    assert asymmetry_text(t).startswith("RR asymmetry: needs")
    assert nonlinear_basis_text(t) == ""


def test_nonlinear_details_say_which_beats_each_block_used(recording):
    _sig, _fs, _rp, r = recording
    nl = r["hrv_nonlin"].copy()
    nl.attrs.update(entropy_capped=True, entropy_n_rr=50, entropy_t_start_s=1.0,
                    entropy_t_end_s=9.0, n_rr_valid=149)
    t = build_metric_table({**r, "hrv_nonlin": nl})
    text = nonlinear_basis_text(t)
    assert "50 of 149" in text and "cap" in text
    assert "DFA: 149 valid RR intervals" in text
    assert "Entropy: Entropy" not in text


# ── asymmetry moved to the page ──────────────────────────────────────────

def test_asymmetry_text_is_the_table_with_its_pairs(recording):
    _sig, _fs, _rp, r = recording
    t = build_metric_table(r)
    text = asymmetry_text(t)
    assert f"Porta PI {t.text('asymmetry.porta_pct')} %" in text
    assert f"Guzik GI {t.text('asymmetry.guzik_pct')} %" in text
    assert "successive valid RR pairs" in text


# ── the Analyze chain ────────────────────────────────────────────────────

def _chain_app(rp, sig, fs, r):
    app = SimpleNamespace(
        analysis=AnalysisState(results=dict(r)),
        detection=DetectionState(rpeaks_ok=rp),
        signal=SignalState(filtered=sig, fs=fs, time=np.arange(len(sig)) / fs),
        ui=UIState(),
        after=lambda ms, fn=None: None,
        _set_status=lambda text, color=None: app.statuses.append(text),
        _safe_float=lambda w, d=0.0: d,
        ent_minrr=None, ent_maxrr=None, cb_freq_band=None,
        lbl_freq_status=_Label(), lbl_nonlin_status=_Label(), lbl_analysis_window=_Label(),
        btn_run_freq=None, btn_run_nonlin=None,
        statuses=[], _generation=0, order=[],
        _plot_psd=lambda r: None, _plot_hrv_tables=lambda r: None,
        _plot_table_views=lambda r: None, _plot_nonlinear=lambda r: None,
        _run_plot_chain=lambda tasks, on_complete=None: on_complete and on_complete(),
        _start_async_result=lambda btn, label, worker, done: done(worker()),
        ent_entropy_cap=_Entry(2000),
        _set_result_btns_enabled=lambda *a: app.disabled.append(a),
        disabled=[],
    )
    app._windowed_peaks = lambda: app.detection.rpeaks_ok
    app.analysis_ctrl = AnalysisController(app)
    app.freshness = FreshnessService(app)
    return app


def _fake_jobs(monkeypatch, app):
    fd = pd.DataFrame({"HRV_LF": [1.0]})
    nl = pd.DataFrame({"HRV_SD1": [2.0]})

    def freq(*a, **k):
        app.order.append("frequency")
        return fd

    def nonlin(*a, **k):
        app.order.append("nonlinear")
        return nl

    monkeypatch.setattr(ac_module, "analyse_hrv_freq", freq)
    monkeypatch.setattr(ac_module, "analyse_hrv_nonlinear", nonlin)
    return fd, nl


def _run_chain(app):
    """What on_analysis_done starts once the core plots are drawn."""
    ctrl = app.analysis_ctrl
    ctrl.run_freq(then=lambda: ctrl.run_nonlinear(then=lambda: app.order.append("epochs")))


def test_chain_order_frequency_nonlinear_epochs(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, r)
    fd, nl = _fake_jobs(monkeypatch, app)
    _run_chain(app)
    assert app.order == ["frequency", "nonlinear", "epochs"]
    assert app.analysis.results["hrv_freq"] is fd and app.analysis.results["hrv_nonlin"] is nl
    assert app.freshness.status("nonlinear").state == UP_TO_DATE
    assert not app.ui.running_modules


def test_on_analysis_done_chains_nonlinear_after_frequency(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, r)
    ctrl = app.analysis_ctrl
    calls = []
    ctrl.run_freq = lambda then=None: (calls.append("freq"), then())
    ctrl.run_nonlinear = lambda then=None: (calls.append("nonlin"), then and then())
    for name in ("_update_kpis", "_draw_detail", "_set_result_btns_enabled",
                 "_set_save_session_enabled", "_update_session_ui"):
        setattr(app, name, lambda *a, **k: None)
    app.detection_ctrl = SimpleNamespace(update_signal_quality=lambda rp: None)
    app._snapshot_params = lambda: {}
    app._compute_epochs = lambda: calls.append("epochs")
    app._draw_core_results = lambda on_complete=None: on_complete()
    app.lbl_npeaks = app.lbl_ivl_status = app.btn_save_session = None
    app.session = SimpleNamespace(dirty=False)
    app.signal.filepath = None
    ctrl.on_analysis_done({"results": dict(r), "rpeaks_ok": rp, "artifact_report": None,
                           "auto_epochs": True})
    assert calls == ["freq", "nonlin", "epochs"]


def _edit_while_running(app, edit):
    def run(btn, label, worker, done):
        out = worker()
        edit()
        done(out)
    app._start_async_result = run


def test_nonlinear_is_dropped_when_the_peaks_change_mid_run(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, {**r, "hrv_nonlin": pd.DataFrame()})
    _fake_jobs(monkeypatch, app)
    _edit_while_running(app, lambda: setattr(app.detection, "rpeaks_ok", np.delete(rp, 3)))
    reached = []
    app.analysis_ctrl.run_nonlinear(then=lambda: reached.append(1))
    assert app.analysis.results["hrv_nonlin"].empty
    assert not reached                                       # nor do later steps run
    assert "discarded" in app.lbl_nonlin_status.text
    assert "discarded" in app.statuses[-1]


def test_nonlinear_is_dropped_when_results_are_dropped_mid_run(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, r)
    _fake_jobs(monkeypatch, app)
    _edit_while_running(app, lambda: setattr(app.analysis, "results", None))
    app.analysis_ctrl.run_nonlinear()
    assert app.analysis.results is None
    assert "discarded" in app.lbl_nonlin_status.text         # not stuck on "Computing"


def test_nonlinear_is_dropped_when_the_window_changes_mid_run(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, {**r, "hrv_nonlin": pd.DataFrame()})
    _fake_jobs(monkeypatch, app)
    _edit_while_running(app, lambda: setattr(app.analysis, "t_start", 3.0))
    app.analysis_ctrl.run_nonlinear()
    assert app.analysis.results["hrv_nonlin"].empty


def test_nonlinear_is_dropped_silently_when_the_file_changes_mid_run(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, {**r, "hrv_nonlin": pd.DataFrame()})
    _fake_jobs(monkeypatch, app)
    app.lbl_nonlin_status.text = "before"

    def change_file():
        app._generation += 1
        app.lbl_nonlin_status.text = "  Click Analyze first"
    _edit_while_running(app, change_file)
    app.analysis_ctrl.run_nonlinear()
    assert app.analysis.results["hrv_nonlin"].empty
    assert app.lbl_nonlin_status.text == "  Click Analyze first"


def test_frequency_stale_stops_the_chain(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, {**r, "hrv_freq": pd.DataFrame()})
    _fake_jobs(monkeypatch, app)
    _edit_while_running(app, lambda: setattr(app.detection, "rpeaks_ok", np.delete(rp, 3)))
    _run_chain(app)
    assert app.order == ["frequency"]
    assert app.analysis.results["hrv_freq"].empty


def test_restored_session_runs_nothing():
    """A restore draws what it brought back; only Analyze starts the chain."""
    import inspect
    src = inspect.getsource(SessionController.on_restore_session_done)
    assert ".run_nonlinear(" not in src and ".run_freq(" not in src
    assert "on_analysis_done" not in src


# ── computing state of the tab ───────────────────────────────────────────

def test_tab_reads_computing_while_one_of_its_modules_runs():
    states = {m: NOT_COMPUTED for mods in TAB_MODULES.values() for m in mods}
    states["core"] = UP_TO_DATE
    assert tab_states(states, True, ())[TAB_HRV] == UP_TO_DATE
    out = tab_states(states, True, {"nonlinear"})
    assert out[TAB_HRV] == COMPUTING
    assert out[TAB_DETECTION] == UP_TO_DATE
    assert COMPUTING in STATE_GLYPHS


def test_a_chain_that_stopped_stops_reading_as_computing():
    from ecg.ui.context_strip import ContextStripMixin, _RUNNING_GRACE_TICKS
    stub = SimpleNamespace(ui=UIState(), _async_busy=False)
    stub.ui.running_modules = {"nonlinear"}
    for _ in range(_RUNNING_GRACE_TICKS):
        assert ContextStripMixin._running_modules(stub) == {"nonlinear"}
    assert ContextStripMixin._running_modules(stub) == set()
    # A job running keeps it, however long it takes.
    stub.ui.running_modules = {"nonlinear"}
    stub.ui.running_idle_ticks = 0
    stub._async_busy = True
    for _ in range(_RUNNING_GRACE_TICKS * 3):
        ContextStripMixin._running_modules(stub)
    assert stub.ui.running_modules == {"nonlinear"}


def test_a_refused_job_marks_nothing(recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, r)
    app._async_busy = True
    app.analysis_ctrl._mark_running("frequency")
    assert not app.ui.running_modules


# ── tab name, session and rebuild ────────────────────────────────────────

def test_tab_is_renamed_and_old_sessions_follow():
    assert TAB_HRV == "Rhythm & HRV"
    for saved in ("HRV", "💓 HRV"):
        assert _TAB_KEY_MAP[saved] == TAB_HRV


def test_entropy_cap_is_kept_by_a_rebuild():
    import inspect
    assert "ent_entropy_cap" in inspect.getsource(SessionController.snapshot_ui_state)
    assert "ent_entropy_cap" in inspect.getsource(SessionController.restore_ui_state)


def test_entropy_cap_entry_still_reaches_the_job(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, r)
    seen = {}

    def nonlin(*a, **k):
        seen.update(k)
        return pd.DataFrame({"HRV_SD1": [2.0]})

    monkeypatch.setattr(ac_module, "analyse_hrv_nonlinear", nonlin)
    app.ent_entropy_cap = _Entry(500)
    app.analysis_ctrl.run_nonlinear()
    assert seen["max_beats"] == 500


# ── Results dropped or a job in the way while the Analyze chain plots ──

def _plot_chain_app(results):
    """run_plot_chain on an app whose idle callbacks wait until pump()."""
    queue = []
    app = SimpleNamespace(analysis=SimpleNamespace(results=results), _generation=0,
                          _set_progress=lambda *a: None, after_idle=queue.append,
                          after=lambda ms, fn=None: None)
    def pump():
        while queue:
            queue.pop(0)()
    return app, PlotController(app), pump


@pytest.mark.parametrize("change", [
    lambda app: setattr(app.analysis, "results", None),          # peak edit
    lambda app: setattr(app.analysis, "results", {"new": 1}),    # new Analyze
    lambda app: setattr(app, "_generation", 1),                  # new file
])
def test_a_plot_chain_stops_once_its_results_are_gone(change):
    app, plot, pump = _plot_chain_app({"old": 1})
    drawn, done = [], []
    plot.run_plot_chain([("a", lambda: (drawn.append("a"), change(app))),
                         ("b", lambda: drawn.append("b"))],
                        on_complete=lambda: done.append(1))
    pump()
    assert drawn == ["a"]        # the old dict never reaches the reset views
    assert not done              # nor does the next step of the chain start


def test_a_plot_chain_on_current_results_runs_to_the_end():
    app, plot, pump = _plot_chain_app({"old": 1})
    drawn, done = [], []
    plot.run_plot_chain([("a", lambda: drawn.append("a")), ("b", lambda: drawn.append("b"))],
                        on_complete=lambda: done.append(1))
    pump()
    assert drawn == ["a", "b"] and done == [1]


def _busy_app(rp, sig, fs, r):
    app = _chain_app(rp, sig, fs, r)
    app.retries = []
    app.after = lambda ms, fn=None: app.retries.append(fn)
    app._async_busy = True
    return app


def test_a_step_waits_for_the_job_in_its_way(recording):
    sig, fs, rp, r = recording
    app = _busy_app(rp, sig, fs, r)
    ran = []
    app.analysis_ctrl._when_idle(lambda: ran.append(1))
    assert not ran and len(app.retries) == 1
    app.retries.pop()()                       # still busy: tries again
    assert not ran and len(app.retries) == 1
    app._async_busy = False
    app.retries.pop()()
    assert ran == [1]


@pytest.mark.parametrize("change", [
    lambda app: setattr(app.analysis, "results", None),
    lambda app: setattr(app, "_generation", 1),
])
def test_a_waiting_step_is_dropped_without_a_modal_when_its_results_go(
        monkeypatch, recording, change):
    sig, fs, rp, r = recording
    app = _busy_app(rp, sig, fs, r)
    modals = []
    monkeypatch.setattr(ac_module.messagebox, "showwarning", lambda *a, **k: modals.append(a))
    ctrl = app.analysis_ctrl
    ctrl._when_idle(ctrl.run_nonlinear)
    change(app)
    app._async_busy = False
    app.retries.pop()()
    assert not modals and not app.retries
    assert not app.lbl_nonlin_status.text


@pytest.mark.parametrize("run, attr", [("run_freq", "lbl_freq_status"),
                                       ("run_nonlinear", "lbl_nonlin_status")])
def test_a_refused_run_never_reads_computing(monkeypatch, recording, run, attr):
    sig, fs, rp, r = recording
    app = _busy_app(rp, sig, fs, r)
    started = []
    app._start_async_result = lambda *a: started.append(a)
    getattr(app.analysis_ctrl, run)()
    assert not started
    assert getattr(app, attr).text == ""
    assert not app.ui.running_modules
    assert "already running" in app.statuses[-1]


def test_a_dropped_job_leaves_its_recompute_button_disabled(monkeypatch, recording):
    sig, fs, rp, r = recording
    app = _chain_app(rp, sig, fs, r)
    _fake_jobs(monkeypatch, app)
    _edit_while_running(app, lambda: setattr(app.analysis, "results", None))
    app.analysis_ctrl.run_nonlinear()
    assert app.disabled and "btn_run_nonlin" in app.disabled[-1][1]


class _Box:
    def __init__(self, text):
        self.text = text


def test_dropped_results_clear_labels_boxes_and_buttons():
    boxes = {n: _Box("SDNN 4.0") for n in ("txt_rr", "txt_td", "txt_fd", "txt_nl")}
    disabled = []
    app = SimpleNamespace(
        lbl_freq_status=_Label("  Done  LF/HF=1.69"), lbl_nonlin_status=_Label("  Done  SampEn=1.21"),
        lbl_ivl_status=_Label("  10/10 beats"), ui=UIState(), **boxes,
        analysis=AnalysisState(), analysis_ctrl=None,
        _set_textbox=lambda box, text: setattr(box, "text", text),
        _set_result_btns_enabled=lambda on, attrs: disabled.append((on, attrs)),
    )
    app.analysis_ctrl = AnalysisController(app)
    app.ui.running_modules.add("nonlinear")
    PlotController(app).reset_hrv_readouts()
    assert all(b.text == "" for b in boxes.values())
    for lbl in (app.lbl_freq_status, app.lbl_nonlin_status, app.lbl_ivl_status):
        assert lbl.text == "  Click Analyze first"
    assert disabled == [(False, ("btn_run_freq", "btn_run_nonlin", "btn_run_ivl"))]
    assert not app.ui.running_modules


def test_every_peak_change_goes_through_the_full_drop():
    import inspect
    from ecg.ui.detection_controller import DetectionController
    src = inspect.getsource(DetectionController)
    assert src.count("self.app.analysis.results  = None") == 1   # only in _drop_results
    assert "_reset_hrv_readouts" in inspect.getsource(DetectionController._drop_results)


def test_status_labels_use_the_shared_metric_strings():
    fd = pd.DataFrame({"HRV_LFn": [0.628], "HRV_HFn": [0.372], "HRV_LFHF": [1.6882]})
    text, _ = AnalysisController._freq_status(fd)
    assert f"LF/HF={format_metric('hrv_freq.HRV_LFHF', 1.6882)}" in text
    assert "LF/HF=1.69" in text and "LFn=62.8%" in text and "HFn=37.2%" in text

    text, _ = AnalysisController._nonlin_status(pd.DataFrame({"HRV_SampEn": [1.2134]}))
    assert f"SampEn={format_metric('hrv_nonlin.HRV_SampEn', 1.2134)}" in text


@pytest.mark.parametrize("value", [float("-inf"), float("inf"), float("nan")])
def test_status_labels_give_a_dash_for_a_non_finite_value(value):
    text, _ = AnalysisController._nonlin_status(pd.DataFrame({"HRV_SampEn": [value]}))
    assert f"SampEn={MISSING}" in text and "inf" not in text and "nan" not in text
    fd = pd.DataFrame({"HRV_LFn": [value], "HRV_HFn": [0.5], "HRV_LFHF": [value]})
    text, _ = AnalysisController._freq_status(fd)
    assert f"LFn={MISSING} " in text and f"LF/HF={MISSING}" in text and "HFn=50.0%" in text
