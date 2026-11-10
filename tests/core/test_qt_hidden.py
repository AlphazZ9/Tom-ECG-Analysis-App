"""With the default switch (QT not reported) no QT / QTc reaches a table, a
report or an export. Run in a fresh process: the suite itself runs with QT
reported (tests/conftest.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CHILD = textwrap.dedent(r"""
    import io, json, sys
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from ecg.core import models
    assert models.QT_REPORTED is False
    from ecg.core.analysis import analyse_intervals, interval_disclosure_rows
    from ecg.core.metric_table import INTERVAL_COLUMNS, build_metric_table
    from ecg.core.wave_template import WaveTemplate
    sys.path.insert(0, sys.argv[2])
    from helpers import make_synth_ecg
    sig, fs, peaks = make_synth_ecg(n_beats=60, fs=2000.0, p_amp=0.15)
    ivl = analyse_intervals(sig, peaks, fs, np.diff(peaks) / 2.0, wave_template=WaveTemplate())
    table = build_metric_table({"intervals": ivl, "hr": {}})
    keys = [r.key for r in table.rows]
    labels = [r.label for r in table.rows] + [r.basis_text for r in table.rows]
    disclosure = " ".join(f"{a} {b}" for a, b in interval_disclosure_rows(ivl))
    shown = models.reported_intervals(ivl)
    print(json.dumps({"cols": [c for c, _ in INTERVAL_COLUMNS], "keys": keys,
                      "labels": labels, "disclosure": disclosure,
                      "frame": list(shown.columns), "computed": "QT_ms" in ivl.columns}))
""")


def test_qt_is_hidden_by_default():
    out = subprocess.run([sys.executable, "-c", _CHILD, ROOT, os.path.join(ROOT, "tests")],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    d = json.loads(out.stdout.strip().splitlines()[-1])
    assert d["computed"]                                   # still computed...
    assert d["cols"] == ["PR_ms", "QRS_ms"]                # ...never shown
    assert not any("QT" in k for k in d["keys"])
    assert not any("QT" in t for t in d["labels"])
    assert not any(c in d["frame"] for c in ("QT_ms", "QTc_ms", "T_offset_ms", "T_peak_ms"))
    assert "QT and QTc not reported" in d["disclosure"]
    # No QT definition, variability or value (the search windows of the wave
    # template, a setting, still name the T).
    for word in ("QT =", "QTc =", "QT variability", "QT IQR"):
        assert word not in d["disclosure"]
