"""
Results round-trip through the session JSON: a DataFrame's ``attrs`` (why a
spectrum was withheld, the non-linear truncation / withheld pair metrics,
the RR exclusion counts) survive a save and restore, so a restored session
still discloses them.
"""
import json

import numpy as np
import pandas as pd

from ecg.core.analysis import hrv_freq_withheld_reason
from ecg.io.loaders import _deserialise_results, _serialise_results
from ecg.io.session import _SessionEncoder, _session_decode_hook


def _round_trip(results: dict) -> dict:
    text = json.dumps(_serialise_results(results), cls=_SessionEncoder)
    return _deserialise_results(json.loads(text, object_hook=_session_decode_hook))


def test_withheld_reason_and_nonlinear_attrs_survive_a_session_round_trip():
    fd = pd.DataFrame()
    fd.attrs.update({"withheld_reason": "2.1 s run of excluded RR intervals (limit 2 s)",
                     "n_excluded": np.int64(3), "excluded_pct": np.float64(1.5)})
    nl = pd.DataFrame({"HRV_SD1": [3.0]})
    nl.attrs.update({"truncated": True, "n_beats_used": 1000, "n_beats_total": 4000,
                     "pair_metrics_withheld": ["HRV_GI", "HRV_SI"],
                     "mse_values": np.array([0.5, 0.6]),
                     "not_json": object()})                 # silently dropped

    out = _round_trip({"hrv_freq": fd, "hrv_nonlin": nl, "rr_ms": np.array([120.0])})

    assert hrv_freq_withheld_reason(out) == "2.1 s run of excluded RR intervals (limit 2 s)"
    assert out["hrv_freq"].attrs["n_excluded"] == 3
    assert out["hrv_nonlin"].attrs["truncated"] is True
    assert out["hrv_nonlin"].attrs["pair_metrics_withheld"] == ["HRV_GI", "HRV_SI"]
    assert out["hrv_nonlin"].attrs["mse_values"] == [0.5, 0.6]
    assert "not_json" not in out["hrv_nonlin"].attrs
    assert out["hrv_nonlin"]["HRV_SD1"].tolist() == [3.0]


def test_dataframe_without_attrs_round_trips_unchanged():
    out = _round_trip({"rr_df": pd.DataFrame({"RR_ms": [120.0, 121.0]})})
    assert out["rr_df"]["RR_ms"].tolist() == [120.0, 121.0]
    assert out["rr_df"].attrs == {}


def test_withheld_reason_is_none_for_a_computed_or_missing_spectrum():
    assert hrv_freq_withheld_reason(None) is None
    assert hrv_freq_withheld_reason({"hrv_freq": pd.DataFrame({"HRV_LF": [1.0]})}) is None
    assert hrv_freq_withheld_reason({"hrv_freq": pd.DataFrame()}) is None
