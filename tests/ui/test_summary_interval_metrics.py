"""Summary metric card: uncorrected QT is reported next to QTc."""
import numpy as np
import pandas as pd

from ecg.ui.plot_controller import summary_interval_metrics


def test_qt_is_shown_next_to_qtc():
    ivl = pd.DataFrame({"PR_ms": [40.0, 42.0, np.nan], "QRS_ms": [14.0, 16.0, 15.0],
                        "QT_ms": [50.0, 52.0, 54.0], "QTc_ms": [60.0, 62.0, 64.0]})
    assert summary_interval_metrics(ivl) == {"pr": "41", "qrs": "15",
                                             "qt": "52", "qtc": "62"}


def test_missing_or_empty_intervals_give_no_values():
    assert summary_interval_metrics(None) == {}
    assert summary_interval_metrics(pd.DataFrame()) == {}
    assert summary_interval_metrics(pd.DataFrame({"RR_ms": [100.0]})) == {}
    assert summary_interval_metrics(pd.DataFrame({"QT_ms": [np.nan]})) == {}
