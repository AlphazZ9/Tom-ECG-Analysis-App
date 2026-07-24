"""
ECG Analysis — mouse cardiac electrophysiology toolkit.

Package layout
--------------
ecg/
  core/
    models.py        — physiology constants, experimental-context reference
                       ranges, dataclasses, typed dicts
    filtering.py     — bandpass, notch, normalise, downsample helpers
    detection.py     — peak detection, RR artifact handling, rhythm /
                       abnormal-event classification
    analysis.py      — HRV time/freq/nonlinear, interval delineation
    wave_template.py — beat-template calibration and wave delineation
    ml_detector.py   — classical-ML R-peak detector, trainable from
                       reviewed recordings
  io/
    loaders.py       — .mat / HDF5 signal loading
    session.py       — session save/load (JSON)
    export.py        — Excel (openpyxl) and Prism exporters
  ui/
    theme.py         — colour/font globals, ThemeConfig, apply_theme_config
    plots.py         — style_axes, CanvasSlot (matplotlib ↔ Tk bridge)
    dialogs.py       — ThemeDialog, ArtifactReviewDialog, Annotation dialogs
    wave_editor.py   — WaveTemplateMiniEditor
    sidebar.py       — CollapsibleSection, IntervalVerifierPanel
    app.py           — ECGApp (main CTk window)
"""

__version__ = "6.0.0"
