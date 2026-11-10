"""
AnalysisController logic that needs no Tk window (a stub app stands in for
ECGApp): the QTc formula in effect is app state, so it survives a UI rebuild
and is what the exports/session record; the RR-exclusion note carried by the
final status of the Analyze chain.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core.analysis import compute_qtc
from ecg.core.models import QTC_FORMULA_LABELS
from ecg.ui.analysis_controller import AnalysisController
from ecg.ui.state import AnalysisState


class _Combo:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def _stub_app(results=None, combo_label=QTC_FORMULA_LABELS["mitchell"]):
    statuses = []
    app = SimpleNamespace(
        analysis=AnalysisState(results=results),
        cb_qtc_formula=_Combo(combo_label),
        _plot_intervals=lambda r: None,
        _plot_table_views=lambda r: None,
        _set_status=lambda text, color=None: statuses.append((text, color)),
    )
    return app, statuses


def _intervals():
    # Standard-definition frame (QRS_onset_ms): RR_ms is the preceding RR.
    return pd.DataFrame({"QT_ms": [50.0, 52.0], "RR_ms": [120.0, 118.0],
                         "QRS_onset_ms": [-8.0, -8.0]})


def test_qtc_formula_comes_from_app_state_not_the_widget():
    app, _ = _stub_app(combo_label=QTC_FORMULA_LABELS["mitchell"])
    app.analysis.qtc_formula = "bazett"
    ctrl = AnalysisController(app)
    # A rebuilt combobox shows its default; the formula in effect is kept.
    assert ctrl.qtc_formula() == "bazett"


def test_selector_change_updates_state_and_recomputes_qtc():
    app, statuses = _stub_app(results={"intervals": _intervals()})
    ctrl = AnalysisController(app)

    label = QTC_FORMULA_LABELS["fridericia"]
    app.cb_qtc_formula.set(label)
    ctrl.on_qtc_formula_change(label)

    assert app.analysis.qtc_formula == "fridericia"
    np.testing.assert_allclose(
        app.analysis.results["intervals"]["QTc_ms"],
        compute_qtc(np.array([50.0, 52.0]), np.array([120.0, 118.0]), "fridericia"))
    assert label in statuses[-1][0]


def test_selector_change_before_any_interval_is_still_remembered():
    app, _ = _stub_app(results=None)
    ctrl = AnalysisController(app)
    ctrl.on_qtc_formula_change(QTC_FORMULA_LABELS["bazett"])
    assert ctrl.qtc_formula() == "bazett"


@pytest.mark.parametrize("hr, expected", [
    ({"n_excluded": 0}, ""),
    ({"n_excluded": 1, "n_too_long": 1, "rr_min_ms": 66.7, "rr_max_ms": 333.3},
     "  |  1 RR excluded (outside 67–333 ms)"),
    ({"n_excluded": 2, "n_spanning_removed": 2}, "  |  2 RR excluded (spanning removed beats)"),
])
def test_exclusion_status_note(hr, expected):
    app, _ = _stub_app(results={"hr": hr})
    assert AnalysisController(app).exclusion_status_note() == expected


def test_exclusion_status_note_without_results():
    app, _ = _stub_app(results=None)
    assert AnalysisController(app).exclusion_status_note() == ""
