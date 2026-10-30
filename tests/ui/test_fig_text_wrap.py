"""
Figure footers that state the interval definitions are wrapped to the
figure width: the legacy PDF footer overran the A4-landscape page and the
Intervals-tab definitions line was clipped on figures narrower than ~7.6 in.
"""
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure                     # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from ecg.core.analysis import (                          # noqa: E402
    INTERVAL_DEFINITIONS_TEXT, LEGACY_INTERVAL_DEFINITIONS_TEXT,
)
from ecg.ui.plots import wrap_fig_text                   # noqa: E402


def _text_extent(fig, txt):
    FigureCanvasAgg(fig).draw()
    return txt.get_window_extent(fig.canvas.get_renderer())


def test_legacy_pdf_footer_fits_the_a4_page():
    fig = Figure(figsize=(11.7, 8.3), dpi=100)
    text = ("Intervals: " + LEGACY_INTERVAL_DEFINITIONS_TEXT
            + "   |   QT IQR = beat-to-beat, single lead (not QT dispersion)")
    # Whether it needs a second line depends on the active font (the app
    # theme's or matplotlib's default); either way it must fit the page.
    wrapped = wrap_fig_text(text, (0.96 - 0.06) * fig.get_figwidth(), 6.5)
    assert wrapped.replace("\n", " ") == text            # nothing lost
    txt = fig.text(0.06, 0.006, wrapped, fontsize=6.5, va="bottom")
    bb = _text_extent(fig, txt)
    assert bb.x1 <= fig.bbox.width and bb.y0 >= 0


@pytest.mark.parametrize("width_in", [5.0, 6.0])
def test_intervals_definitions_line_fits_narrow_figures(width_in):
    fig = Figure(figsize=(width_in, 3.0), dpi=100)
    wrapped = wrap_fig_text(INTERVAL_DEFINITIONS_TEXT, 0.96 * width_in, 6.5)
    txt = fig.text(0.5, 0.005, wrapped, ha="center", va="bottom", fontsize=6.5,
                   multialignment="center")
    bb = _text_extent(fig, txt)
    assert bb.x0 >= 0 and bb.x1 <= fig.bbox.width


def test_long_text_is_split_at_spaces_without_losing_words():
    text = "Intervals: " + LEGACY_INTERVAL_DEFINITIONS_TEXT
    wrapped = wrap_fig_text(text, 3.0, 6.5)
    assert wrapped.count("\n") >= 2
    assert wrapped.replace("\n", " ") == text


def test_short_text_is_left_on_one_line():
    assert wrap_fig_text("PR = P onset", 5.0, 6.5) == "PR = P onset"


def test_pdf_footer_states_the_definitions_of_its_frame():
    import numpy as np
    import pandas as pd
    from ecg.core import analysis
    from ecg.ui.export_controller import pdf_interval_footer
    frame = pd.DataFrame({"R_peak_s": [1.0, 1.1, 1.22, 1.34, 1.46],
                          "RR_ms": [100.0, 120.0, 120.0, 120.0, np.nan],
                          "QT_ms": [50.0, 52.0, 54.0, 56.0, 58.0]})
    assert pdf_interval_footer(None) == ""
    current = pdf_interval_footer(frame.assign(QRS_onset_ms=-8.0))
    assert current.startswith("Intervals: " + INTERVAL_DEFINITIONS_TEXT)
    assert LEGACY_INTERVAL_DEFINITIONS_TEXT in pdf_interval_footer(frame)
    # A legacy frame without R peak times has no QTc: the footer says so.
    no_qtc = pdf_interval_footer(frame.drop(columns="R_peak_s"))
    assert analysis.LEGACY_INTERVAL_DEFINITIONS_NO_QTC_TEXT in no_qtc
