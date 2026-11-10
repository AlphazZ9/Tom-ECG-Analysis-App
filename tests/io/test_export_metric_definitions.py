"""
The Excel Cover sheet states the metric definitions behind the numbers:
mean HR = 60000 / mean NN, how the frequency HRV was computed (the plotted
Welch spectrum, ms², Task Force n.u.), and which RR intervals the
non-linear metrics used (capped entropy block: count and time span).
"""
import numpy as np
import pytest

from ecg.core import analysis
from ecg.io.export import ExcelExporter

from tests.helpers import rpeaks_from_rr

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0


def _cover(results) -> dict:
    wb = ExcelExporter.build_workbook(
        results=results, signal_flt=None, signal_raw=None,
        time=np.arange(10) / 1000.0, rpeaks_ok=None, fs=2000, filepath=None,
        subject="m1", sig_quality=None, epoch_df=None, qtc_formula=None)
    ws = wb["Cover"]
    return {ws.cell(row=r, column=1).value: ws.cell(row=r, column=2).value
            for r in range(1, ws.max_row + 1)}


def _results(n: int = 900):
    rng = np.random.default_rng(3)
    t = np.arange(n) * 0.12
    rr = 120 + 4 * np.sin(2 * np.pi * 2.5 * t) + rng.normal(0, 1, n)
    peaks = rpeaks_from_rr(list(rr), FS)
    res = analysis.analyse_core(np.zeros(int(peaks[-1]) + int(FS)), peaks, FS)
    return res, peaks


@requires_nk
def test_cover_states_mean_hr_and_spectral_definitions():
    res, peaks = _results()
    res["hrv_freq"] = analysis.analyse_hrv_freq(peaks, FS)
    cover = _cover(res)
    assert cover["HR mean"].endswith("(60000 / mean NN)")
    assert "ms²" in cover["Frequency HRV"] and "(LF + HF)" in cover["Frequency HRV"]
    assert "PSD plot shows this same spectrum" in cover["Frequency HRV"]


@requires_nk
def test_cover_discloses_the_capped_entropy_block():
    res, peaks = _results()
    res["hrv_nonlin"] = analysis.analyse_hrv_nonlinear(peaks, FS, max_beats=300)
    assert "300 of 900 valid RR intervals" in _cover(res)["Non-linear HRV"]

    res["hrv_nonlin"] = analysis.analyse_hrv_nonlinear(peaks, FS, max_beats=0)
    assert _cover(res)["Non-linear HRV"] == "all valid RR intervals of the analysis window"


@requires_nk
def test_cover_states_the_asymmetry_conventions_and_prism_labels_them(tmp_path):
    res, peaks = _results()
    res["hrv_nonlin"] = analysis.analyse_hrv_nonlinear(peaks, FS, max_beats=0)
    assert _cover(res)["HR asymmetry (HRV_PI, HRV_GI)"] == analysis.RR_ASYMMETRY_TEXT

    pytest.importorskip("pzfx")
    from ecg.io.export import PrismExporter
    path = tmp_path / "m1.pzfx"
    PrismExporter.build_and_write(path, res)
    xml = path.read_text(encoding="utf-8")
    assert "Porta Index PI (%)" in xml and "Guzik Index GI (%)" in xml
