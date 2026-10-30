"""
A session save / restore keeps the provenance of the saved results: the
analysis window they were computed on (not window entries applied
afterwards without re-running Analyze), and the filters and load-time crop
the signal was prepared with (not widget values edited after detection).
The restore re-prepares the signal from those, and the crop is recorded in
file time for the "Signal crop" provenance row. Older sessions without
these keys fall back to the previous behaviour.
"""
import json

import numpy as np
import pytest

from ecg.core import wave_template
from ecg.core.models import FilterParams
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import session_controller as sc_module
from ecg.ui.session_controller import SessionController

from tests.core.test_legacy_results import _core
from tests.ui.test_autonomic_controller import _session_app

FS = 2000.0
FILE_S = 20.0


@pytest.fixture(autouse=True)
def _profile(tmp_path, monkeypatch):
    path = tmp_path / ".ecg_wave_template.json"
    monkeypatch.setattr(wave_template, "WAVE_TEMPLATE_PATH", path)
    monkeypatch.setattr(wave_template, "_WAVE_TEMPLATE_PATH_LEGACY",
                        path.with_suffix(".pkl"))
    return path


def _app_with_results():
    app = _session_app()
    app.analysis.results, _ = _core()
    # Analysed on 100-160 s; a new window applied since, not re-analysed.
    app.analysis.analysed_window = (100.0, 160.0)
    app.analysis.t_start, app.analysis.t_end = 200.0, 300.0
    # Prepared with 1-100 Hz and a 5 s start crop; the widgets now differ.
    app.signal.prepare_params = FilterParams(
        lp=1.0, hp=100.0, t_start=5.0, t_end=0.0, no_filter=False).to_dict()
    return app


def _restore(monkeypatch, app, state):
    state = json.loads(json.dumps(state, cls=_SessionEncoder),
                       object_hook=_session_decode_hook)
    monkeypatch.setattr(sc_module, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(int(FILE_S * FS)), "ECG", None, FS))
    seen: list = []
    app._compute_preview_bundle = lambda sig, fs, params: seen.append((len(sig), params)) or {}
    return SessionController(app).restore_session_worker(state), seen


def test_the_analysed_window_survives_a_round_trip(monkeypatch):
    app = _app_with_results()
    state = SessionController(app).collect_session_state()
    assert state["analysed_window"] == [100.0, 160.0]
    assert (state["analysis_t_start"], state["analysis_t_end"]) == (200.0, 300.0)

    bundle, _ = _restore(monkeypatch, app, state)
    assert bundle["analysed_window"] == (100.0, 160.0)
    assert (bundle["analysis_t_start"], bundle["analysis_t_end"]) == (200.0, 300.0)


def test_an_older_session_falls_back_to_the_window_entries(monkeypatch):
    app = _app_with_results()
    state = SessionController(app).collect_session_state()
    del state["analysed_window"]
    bundle, _ = _restore(monkeypatch, app, state)
    assert bundle["analysed_window"] == (200.0, 300.0)

    # No results: no analysed window at all.
    del state["results"]
    bundle, _ = _restore(monkeypatch, app, state)
    assert bundle["analysed_window"] is None


def test_the_signal_is_reprepared_with_the_detection_params(monkeypatch):
    app = _app_with_results()
    state = SessionController(app).collect_session_state()
    widget_fp = FilterParams.from_dict(state["filter_params"])
    assert (widget_fp.lp, widget_fp.hp) != (1.0, 100.0)      # edited since
    assert state["prepare_params"]["lp"] == 1.0

    bundle, seen = _restore(monkeypatch, app, state)
    (n_samples, params), = seen
    assert (params["lp"], params["hp"], params["no_filter"]) == (1.0, 100.0, False)
    assert n_samples == int((FILE_S - 5.0) * FS)              # detection-time crop
    assert bundle["signal_crop"] == {"t_start": 5.0, "t_end": FILE_S,
                                     "file_duration_s": FILE_S}
    # The widgets still get the values they showed at save time.
    assert bundle["filter_params"] == widget_fp


def test_an_older_session_reprepares_with_the_widget_values(monkeypatch):
    app = _app_with_results()
    state = SessionController(app).collect_session_state()
    del state["prepare_params"]
    _, seen = _restore(monkeypatch, app, state)
    (n_samples, params), = seen
    widget_fp = FilterParams.from_dict(state["filter_params"])
    assert (params["lp"], params["hp"]) == (widget_fp.lp, widget_fp.hp)
    assert n_samples == int(FILE_S * FS)


def test_nothing_prepared_saves_no_prepare_params():
    app = _session_app()
    app.signal.prepare_params = None
    assert "prepare_params" not in SessionController(app).collect_session_state()


def test_loading_records_the_crop_in_file_time(monkeypatch):
    from types import SimpleNamespace
    from ecg.ui import signal_controller as sig_module
    from ecg.ui.signal_controller import SignalController
    from ecg.ui.state import SignalState

    rng = np.random.default_rng(0)
    monkeypatch.setattr(sig_module, "load_mat_signal",
                        lambda path, ch=None: (rng.normal(size=int(FILE_S * FS)),
                                               "ECG", None, FS))
    app = SimpleNamespace(signal=SignalState(filepath="recording.mat"),
                          after=lambda ms, fn: None)
    params = FilterParams(t_start=2.0, t_end=12.0).to_dict()
    bundle = SignalController(app).load_raw_worker(params)   # type: ignore[arg-type]
    assert len(bundle["signal_raw"]) == int(10.0 * FS)
    assert bundle["signal_crop"] == {"t_start": 2.0, "t_end": 12.0,
                                     "file_duration_s": FILE_S}
