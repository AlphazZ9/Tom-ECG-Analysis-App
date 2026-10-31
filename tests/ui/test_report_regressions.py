# -*- coding: utf-8 -*-
"""
Report stage regressions: Include does not outlive its recording, the ZIP's
figures follow Include, no old tab name in what the user reads, and a theme
change leaves no focus restore on a destroyed widget.
"""
import ast
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure

from ecg.core.metric_table import REPORT_AUTONOMIC, REPORT_INTERVALS, REPORT_RHYTHM
from ecg.ui import dialogs, export_controller as ec, theme
from ecg.ui.app import ECGApp
from ecg.ui.export_controller import ExportController
from ecg.ui.freshness import FreshnessService
from ecg.ui.signal_controller import SignalController
from ecg.ui.state import AnalysisState, UIState

ROOT = Path(__file__).resolve().parents[2]


# ── a tick stands for one recording ──────────────────────────────────────

def _app_with_ticks():
    """Enough of an app for the two load paths; everything else a mock."""
    app = MagicMock()
    app.analysis = AnalysisState(basis={"core": {"window": (0, 1)}})
    app.ui = UIState(report_include={REPORT_RHYTHM: [False, "up to date"]},
                     module_computed_at={"core": "2026-01-01T10:00:00"})
    app.ui.hover_after_id = None
    app.freshness = FreshnessService(app)
    app._slots, app._kpi, app._arr_card_widgets = {}, {}, []
    return app


def _assert_nothing_carried_over(app):
    assert app.ui.report_include == {}
    assert app.analysis.basis == {}
    assert app.ui.module_computed_at == {}


def test_opening_a_file_clears_the_ticks():
    app = _app_with_ticks()
    SignalController(app).reset_for_new_file()
    _assert_nothing_carried_over(app)


def test_the_raw_load_clears_the_ticks():
    app = _app_with_ticks()
    sig = [0.0, 1.0]
    SignalController(app).on_raw_load_done(
        {"fs": 1000, "signal_raw": sig, "signal_raw_norm": sig, "signal_flt": sig,
         "all_cands": None, "all_proms": None, "no_filter_mode": False, "rpeaks_ok": None,
         "rpeaks_rej": None, "thresh_amp": 0.0, "detected_ch": "ECG", "requested_ch": "ECG",
         "fs_from_file": True, "detected_fs": 1000, "dur_s": 2.0})
    _assert_nothing_carried_over(app)


# ── the ZIP's figures follow Include ─────────────────────────────────────

_ALL = {"00_detail", "01_rr_hr", "02_rr_hist", "03_psd", "05_poincare", "06_intervals",
        "07_beat", "08_autonomic_csi_cpi", "09_event_locked"}
_OF_LINE = {REPORT_RHYTHM: {"01_rr_hr", "02_rr_hist", "03_psd", "05_poincare"},
            REPORT_INTERVALS: {"06_intervals", "07_beat"},
            REPORT_AUTONOMIC: {"08_autonomic_csi_cpi", "09_event_locked"}}


def _zip_figures(monkeypatch, tmp_path, included, drawn=True):
    """What export_zip really writes for *included*."""
    slots = {key: SimpleNamespace(fig=Figure(figsize=(1, 1), dpi=20),
                                  _draw_fn=print if drawn else None)
             for key, _, _ in ec._ZIP_FIGURES}
    app = SimpleNamespace(analysis=SimpleNamespace(results={"hr": {}}), _slots=slots,
                          ent_subject=SimpleNamespace(get=lambda: "m1"),
                          _set_status=MagicMock())
    ctrl = ExportController(app)
    out = tmp_path / "out.zip"
    monkeypatch.setattr(ec.filedialog, "asksaveasfilename", lambda **k: str(out))
    monkeypatch.setattr(ctrl, "write_excel", lambda dest: dest.write(b"x"))
    monkeypatch.setattr(ctrl, "export_data",
                        lambda: SimpleNamespace(included=included, left_out=""))
    ctrl.export_zip()
    with zipfile.ZipFile(out) as zf:
        return {Path(n).stem for n in zf.namelist() if n.startswith("figures/")}


def test_zip_holds_every_figure_with_everything_included(monkeypatch, tmp_path):
    assert _zip_figures(monkeypatch, tmp_path, {}) == _ALL


@pytest.mark.parametrize("line", sorted(_OF_LINE))
def test_zip_leaves_out_the_figures_of_an_unticked_line(monkeypatch, tmp_path, line):
    assert _zip_figures(monkeypatch, tmp_path, {line: False}) == _ALL - _OF_LINE[line]


def test_zip_skips_the_autonomic_placeholders(monkeypatch, tmp_path):
    assert (_zip_figures(monkeypatch, tmp_path, {}, drawn=False)
            == _ALL - _OF_LINE[REPORT_AUTONOMIC])


# ── old tab names ────────────────────────────────────────────────────────

# "Signal & Detection tab" and the like are the current names.
_OLD = re.compile(r"(?<!& )\b(?:Detection|HRV|Intervals?|Interval Delineation|Beat Template|"
                  r"Summary) tab\b|\bsub-view\b")
# ecg/core/wave_template.py is being worked on separately and cannot be edited:
# its template-rejected notice still says "Beat Template tab".
_EXCEPTIONS = {"wave_template.py"}


def _literals(path: Path) -> "list[str]":
    """The strings of *path* that can reach a user: docstrings do not."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                docs.add(id(node.body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_no_old_tab_name_in_a_message():
    hits = [(p.name, s) for p in (ROOT / "ecg").rglob("*.py") if p.name not in _EXCEPTIONS
            for s in _literals(p) if _OLD.search(s)]
    assert hits == []


def test_no_old_tab_name_in_the_guide():
    guide = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8").replace("&amp;", "&")
    assert _OLD.findall(guide) == []


# ── theme change ─────────────────────────────────────────────────────────

@pytest.fixture
def theme_calls(monkeypatch):
    calls = []
    # Both callers apply the config; neither may reach the profile or the real CTk.
    monkeypatch.setattr(theme.THEME, "apply_preset", lambda name: None)
    monkeypatch.setattr(theme.THEME, "save", lambda: None)
    monkeypatch.setattr(theme.THEME, "font_scale", theme.THEME.font_scale)
    monkeypatch.setattr(theme, "apply_theme_config", lambda tc: calls.append("appearance"))
    monkeypatch.setattr(dialogs, "apply_theme_config", lambda tc: calls.append("appearance"))
    return calls


def test_live_toggle_parks_the_focus_before_the_appearance_changes(theme_calls):
    app = MagicMock()
    app._park_focus.side_effect = lambda: theme_calls.append("park")
    ECGApp._toggle_dark_live(app)
    assert theme_calls == ["park", "appearance"]


def test_theme_dialog_parks_the_focus_before_the_appearance_changes(theme_calls):
    cls = next(c for c in vars(dialogs).values()
               if isinstance(c, type) and c.__module__ == dialogs.__name__
               and "_save_default" in vars(c))
    dlg = MagicMock()
    dlg._app._park_focus.side_effect = lambda: theme_calls.append("park")
    dlg._working = SimpleNamespace(preset_name="Light", font_scale=1.0)
    cls._apply(dlg)
    assert theme_calls == ["park", "appearance"]
