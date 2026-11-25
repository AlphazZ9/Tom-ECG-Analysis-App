"""Annotations suggested from a recording's other channels: keyboard /
marker channels (with their codes) and stimulus pulses."""
from __future__ import annotations

import numpy as np
import scipy.io

from ecg.core import annotation_suggest as sug
from ecg.io import event_channels as evc
from ecg.ui.annotation_suggest_dialog import collect


def _spike2_mat(path, fs=1000.0, dur=120.0):
    n = int(fs * dur)
    t = np.arange(n) / fs
    ecg = np.sin(2 * np.pi * 8 * t)
    stim = np.zeros(n)
    for k in range(20):                                   # 20 Hz train at 60 s, 1 s
        i = int((60 + k * 0.05) * fs)
        stim[i:i + 5] = 5.0
    codes = np.zeros((3, 4), dtype=np.uint8)
    codes[:, 0] = [ord("1"), ord("2"), ord("1")]
    scipy.io.savemat(str(path), {
        "ECG": {"title": "ECG", "values": ecg, "interval": 1 / fs},
        "Stim": {"title": "Stim", "values": stim, "interval": 1 / fs},
        "Keyboard": {"title": "Keyboard", "times": np.array([10.0, 30.0, 90.0]),
                     "codes": codes, "length": 3},
    })


def test_keyboard_codes_and_a_pulse_train_are_suggested(tmp_path):
    p = tmp_path / "rec.mat"
    _spike2_mat(p)
    ev = evc.read_event_channels(str(p))
    assert ev == {"Keyboard": [(10.0, "1"), (30.0, "2"), (90.0, "1")]}
    found, read = collect(str(p), "ECG", None, 120.0, [])
    assert set(read) == {"Keyboard", "Stim"}
    labels = [(s["t_start"], s["label"]) for s in found]
    assert (10.0, "Keyboard '1'") in labels and (30.0, "Keyboard '2'") in labels
    stim = [s for s in found if s["source"] == "Stim"]
    assert len(stim) == 1 and stim[0]["t_start"] == 60.0
    assert "20 pulses, 20 Hz" in stim[0]["label"] and abs(stim[0]["t_end"] - 61.0) < 0.01
    # The ECG is no pulse train; an annotation there already: not again.
    found, _ = collect(str(p), "ECG", None, 120.0, [{"t_start": 60.2, "t_end": 61}])
    assert not [s for s in found if s["source"] == "Stim"]


def test_the_load_crop_moves_the_times(tmp_path):
    p = tmp_path / "rec.mat"
    _spike2_mat(p)
    crop = {"t_start": 20.0, "t_end": 100.0, "file_duration_s": 120.0}
    found, _ = collect(str(p), "ECG", crop, 80.0, [])
    starts = sorted(s["t_start"] for s in found)
    assert starts == [10.0, 40.0, 70.0]                 # 30, 60 (stim), 90 s in file time


def test_markers_closer_than_the_gap_make_one_train():
    got = sug.from_events({"STIM": [(5.0, ""), (5.5, ""), (6.0, ""), (50.0, "")]})
    assert [(g["t_start"], g["t_end"]) for g in got] == [(5.0, 6.5), (50.0, 51.0)]
    assert got[0]["label"] == "STIM (3 markers, 2 Hz)" and got[1]["label"] == "STIM"


def test_an_ecg_or_noise_is_no_pulse_train():
    rng = np.random.default_rng(0)
    assert not len(sug.pulse_onsets(rng.normal(size=10000), 1000.0))
    t = np.arange(10000) / 1000
    assert not len(sug.pulse_onsets(np.sin(2 * np.pi * 8 * t), 1000.0))
    assert not len(sug.pulse_onsets(np.zeros(1000), 1000.0))


def test_codes_as_typed():
    assert evc.code_text(np.array([49, 0, 0, 0])) == "1"
    assert evc.code_text(np.array([0, 0, 0, 0])) == ""
    assert evc.code_text(b"a\x00") == "a" and evc.code_text(200) == "200"


def test_spike2_keyboard_labels(monkeypatch, tmp_path):
    from ecg.io import loaders
    from tests.io.test_spike2_loader import _FakeEventReader, _ecg

    class _Keys(_FakeEventReader):
        def get_event_timestamps(self, _b, _s, i):
            ts, _d, _l = super().get_event_timestamps(_b, _s, i)
            return ts, None, np.array(["a", "b"][:len(ts)])
    reader = _Keys([("Recording", 1000.0, _ecg(1000.0, 30))],
                   {"Keyboard": [198_000_000, 396_000_000]})
    monkeypatch.setattr(loaders, "_spike2_reader", lambda _p: reader)
    path = tmp_path / "rec.smr"
    path.write_bytes(b"\x05\x00(C) CED 87")
    assert evc.read_event_channels(str(path), 100.0) == {"Keyboard": [(98.0, "a"),
                                                                      (296.0, "b")]}


def test_the_dialog_adds_the_ticked_suggestions(monkeypatch, tmp_path):
    import tkinter as tk
    from types import SimpleNamespace as NS
    import customtkinter as ctk
    import pytest
    from ecg.ui import annotation_suggest_dialog as asd
    try:
        root = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    try:
        found = [{"t_start": 10.0, "t_end": 11.0, "label": "Keyboard '1'", "source": "Keyboard"},
                 {"t_start": 60.0, "t_end": 61.0, "label": "Stim (20 pulses)", "source": "Stim"}]
        monkeypatch.setattr(asd, "collect", lambda *a: (found, ["Keyboard", "Stim"]))
        app = NS(signal=NS(filepath="x.mat", channel="ECG", crop=None), _time=None,
                 analysis=NS(annotations=[]))
        added = []
        dlg = asd.AnnotationSuggestDialog(root, app, added.extend)
        for _ in range(50):
            root.update()
            if dlg._rows:
                break
            import time
            time.sleep(0.02)
        assert len(dlg._rows) == 2
        dlg._rows[0][0].set(False)
        dlg._rows[1][1].delete(0, "end")
        dlg._rows[1][1].insert(0, "Stim 20 Hz")
        dlg._add()
        assert [(a["t_start"], a["label"]) for a in added] == [(60.0, "Stim 20 Hz")]
        assert added[0]["color"]
    finally:
        root.destroy()


def test_keyboard_0_is_left_out_of_suggestions_and_markers(monkeypatch, tmp_path):
    from ecg.io import loaders
    from tests.io.test_spike2_loader import _FakeEventReader, _ecg
    # MATLAB export: a '0' key among others.
    p = tmp_path / "rec.mat"
    codes = np.zeros((3, 4), dtype=np.uint8)
    codes[:, 0] = [ord("0"), ord("1"), ord("0")]
    scipy.io.savemat(str(p), {"Keyboard": {"title": "Keyboard", "times": np.array([5.0, 9.0, 20.0]),
                                           "codes": codes}})
    assert evc.read_event_channels(str(p)) == {"Keyboard": [(9.0, "1")]}
    # Spike2 .smr: the markers (Compare segments' stim markers) and the suggestions.

    class _Keys(_FakeEventReader):
        def get_event_timestamps(self, _b, _s, i):
            ts, _d, _l = super().get_event_timestamps(_b, _s, i)
            labs = {"Keyboard": ["0", "a", "0"], "STIM": ["0", "0", "0"]}
            return ts, None, np.array(labs[list(self.events)[i]][:len(ts)])
    reader = _Keys([("Recording", 1000.0, _ecg(1000.0, 30))],
                   {"Keyboard": [1_000_000, 2_000_000, 3_000_000],
                    "STIM": [4_000_000, 5_000_000, 6_000_000]})
    monkeypatch.setattr(loaders, "_spike2_reader", lambda _p: reader)
    smr = tmp_path / "rec.smr"
    smr.write_bytes(b"\x05\x00(C) CED 87")
    loaders._MARKER_CACHE.clear()
    m = loaders.read_spike2_markers(str(smr))
    assert m["Keyboard"] == [2.0] and m["STIM"] == [4.0, 5.0, 6.0]   # only the keyboard
    ev = evc.read_event_channels(str(smr))
    assert ev["Keyboard"] == [(2.0, "a")] and len(ev["STIM"]) == 3
