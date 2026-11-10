"""The landmark editor exposes the QRS onset / QRS offset (J point)
landmarks that PR, QRS and QT are now measured from."""
from ecg.core.wave_template import WaveTemplate
from ecg.ui.wave_editor import WaveTemplateMiniEditor


def test_editor_shows_qrs_onset_and_offset():
    keys = WaveTemplateMiniEditor._KEYS
    for k in ("QRS_onset", "QRS_offset"):
        assert k in keys and k in WaveTemplate.DEFAULTS
        assert k in WaveTemplateMiniEditor._COLS and k in WaveTemplateMiniEditor._LABELS
    # Ordered as on the trace: onset before Q, J point after S.
    assert keys.index("QRS_onset") < keys.index("Q_peak")
    assert keys.index("S_peak") < keys.index("QRS_offset") < keys.index("J_peak")
    assert WaveTemplateMiniEditor._LABELS["QRS_offset"] == "J point"
