"""
The PDF report rendered for real: ExportController.export_pdf_report is
driven with a stub app, the save dialog and PdfPages are monkeypatched (no
file is written), and every text drawn on the figure -- figure texts, axes
titles / labels / texts, including those built by helpers (wrap_fig_text,
pdf_provenance_text, pdf_interval_footer) -- is checked: English only, and
the footer states the signal crop, the analysis window clamped to the
signal end, the analysed duration, the filters and the artifact correction.
"""
import re
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as backend_pdf
import numpy as np
import pandas as pd
import pytest

import ecg.ui.export_controller as export_controller
from ecg.core.analysis import signal_crop_info
from ecg.ui.export_controller import ExportController
from ecg.ui.state import AnalysisState, DetectionState, SignalState

FS = 200
ARTIFACTS = {"n_in": 503, "n_out": 500, "n_nonphysio": 1, "n_ectopic": 2, "n_duplicate": 0}


def _results(t0_s: float = 100.0, n: int = 500) -> dict:
    rr = 120.0 + np.sin(np.arange(n))
    t = t0_s + np.cumsum(rr) / 1000.0
    td = pd.DataFrame({"HRV_MeanNN": [float(rr.mean())], "HRV_SDNN": [3.0],
                       "HRV_RMSSD": [2.0], "HRV_pNN6": [10.0]})
    ivl = pd.DataFrame({"PR_ms": [35.0, 36.0, 34.0, 35.0],
                        "QRS_onset_ms": [-6.0] * 4, "QRS_ms": [10.0, 11.0, 10.0, 9.0],
                        "QT_ms": [50.0, 52.0, 51.0, 49.0],
                        "QTc_ms": [45.0, 47.0, 46.0, 44.0], "RR_ms": [120.0] * 4})
    return {"hr": {"n": n + 1, "mean": 500.0, "min": 490.0, "max": 510.0, "std": 2.0},
            "rr_ms": rr, "rr_valid": np.ones(n, dtype=bool), "rr_t_s": t,
            "rr_df": pd.DataFrame({"Time_s": t, "RR_ms": rr, "HR_bpm": 60000.0 / rr}),
            "hrv_time": td, "intervals": ivl}


def _app(window, crop, filepath="C:/data/m1.mat", filter_text="FILTERS-X"):
    n = 600 * FS
    sig = SignalState(filepath=filepath, fs=FS,
                      filtered=np.sin(np.arange(n) / 10.0),
                      time=np.arange(n) / FS, no_filter_mode=False,
                      filter_text=filter_text, crop=crop)
    return SimpleNamespace(
        analysis=AnalysisState(results=_results(), analysed_window=window,
                               artifact_report=ARTIFACTS),
        signal=sig,
        detection=DetectionState(rpeaks_ok=np.arange(100 * FS, 160 * FS, 24)),
        ent_subject=SimpleNamespace(get=lambda: "m1"),
        _qtc_formula=lambda: "mitchell",
        _set_status=lambda *a, **k: None)


def _render(monkeypatch, window=(500.0, 900.0), crop=None, **app_kw) -> "list[str]":
    fig = _render_fig(monkeypatch, window, crop, **app_kw)
    texts = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        texts += [ax.get_title(loc=loc) for loc in ("left", "center", "right")]
        texts += [ax.get_xlabel(), ax.get_ylabel()]
        texts += [t.get_text() for t in ax.texts]
    return [t for t in texts if t]


def _render_fig(monkeypatch, window=(500.0, 900.0), crop=None, **app_kw):
    figs: list = []

    class _Pages:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def savefig(self, fig, **kw):
            figs.append(fig)

    monkeypatch.setattr(backend_pdf, "PdfPages", _Pages)
    monkeypatch.setattr(export_controller.filedialog, "asksaveasfilename",
                        lambda **kw: "unused.pdf")
    errors: list = []
    monkeypatch.setattr(export_controller.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    ExportController(_app(window, crop, **app_kw)).export_pdf_report()  # type: ignore[arg-type]
    assert errors == [] and len(figs) == 1
    return figs[0]


def test_pdf_page_text_is_english(monkeypatch):
    texts = _render(monkeypatch)
    french = [t for t in texts
              if re.search(r"[éèàùâêîôûç]", t.replace("Poincaré", ""))]
    assert french == []
    joined = " ".join(texts)
    for word in ("Fichier", "Métriques", "Paramètre", "Valeur", "non disponible"):
        assert word not in joined


def test_pdf_footer_states_crop_clamped_window_and_artifacts(monkeypatch):
    crop = signal_crop_info(30.0, 630.0, 900.0)
    footer = " ".join(" ".join(_render(monkeypatch, crop=crop)).split())

    # Window entered past the 600 s signal end: clamped, and said so.
    assert "Analysis window: 500.0 – 600.0 s" in footer
    assert "Analysed duration: 100.0 s (window end − start)" in footer
    assert "Signal crop: file time 30.0 – 630.0 s of a 900.0 s file" in footer
    assert "relative to the crop start" in footer
    assert "Filters: FILTERS-X" in footer
    assert "3 of 503 beats removed (whole recording, before windowing" in footer
    assert "QT = QRS onset → T end" in footer


def test_pdf_footer_without_a_recorded_crop_has_no_crop_row(monkeypatch):
    footer = " ".join(_render(monkeypatch, window=(100.0, 160.0)))
    assert "Signal crop" not in footer
    assert "Analysis window: 100.0 – 160.0 s" in footer


def test_pdf_header_fits_the_page_with_a_long_file_path(monkeypatch):
    # The header line ran past the right edge, cutting the QTc formula
    # ("QTc: Mitchell (QT/√"): the path is shortened in the middle instead.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    path = "C:/Users/someone/Documents/Experiments/2026/cohort-B/" * 3 + "m1_baseline.mat"
    fig = _render_fig(monkeypatch, filepath=path)
    head = next(t for t in fig.texts if t.get_text().startswith("File: "))
    assert head.get_text().endswith(", preceding RR") and "QTc: Mitchell" in head.get_text()
    assert "…" in head.get_text() and "\n" not in head.get_text()
    assert "m1_baseline.mat" in head.get_text()
    FigureCanvasAgg(fig).draw()
    assert head.get_window_extent(fig.canvas.get_renderer()).x1 <= fig.bbox.width


def test_pdf_strip_is_not_called_filtered_without_filters(monkeypatch):
    from ecg.core.filtering import filter_description
    raw = " ".join(_render(monkeypatch, filter_text=filter_description({"no_filter": True})))
    assert "Raw ECG — 10 s excerpt" in raw and "Filtered ECG" not in raw
    assert "Filtered ECG — 10 s excerpt" in " ".join(_render(monkeypatch))


def test_pdf_key_metrics_table_shows_the_interval_rows(monkeypatch):
    # The table stopped after 9 rows, before PR, QRS, QT and QTc (HRV first).
    res = _results()
    res["hrv_freq"] = pd.DataFrame({"HRV_LFn": [0.4], "HRV_HFn": [0.6]})
    res["hrv_nonlin"] = pd.DataFrame({"HRV_SD1": [1.0], "HRV_SD2": [2.0], "HRV_SampEn": [1.5]})
    monkeypatch.setattr(export_controller, "hrv_freq_is_legacy", lambda r: False)
    monkeypatch.setitem(globals(), "_results", lambda *a, **k: res)
    fig = _render_fig(monkeypatch)
    ax = next(a for a in fig.axes if a.get_title(loc="left") == "Key metrics")
    rows = {t.get_text(): t for t in ax.texts}
    for label in ("Mean HR", "SampEn", "PR", "QRS", "QT", "QTc"):
        assert label in rows
    assert all(0.0 <= t.get_position()[1] <= 1.0 for t in ax.texts)
