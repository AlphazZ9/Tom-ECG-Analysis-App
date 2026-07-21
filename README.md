# ECG Analysis — Mouse Cardiac Electrophysiology Toolkit

*[Version française](README.fr.md)*

A desktop application (CustomTkinter) for analyzing mouse cardiac ECG
recordings: R-peak detection, HRV (time/frequency/non-linear), P-QRS-T wave
delineation, arrhythmia classification, and export to Excel / GraphPad Prism.

Built around mouse-specific physiology (Thireau et al. 2008; Mitchell et al.
1998) rather than adapted human-ECG defaults — heart rate range, pNN
threshold, HRV frequency bands, and QTc correction formula are all tuned for
mice (typical resting HR ~500 bpm, 180–900 bpm range).

## Features

- **Signal loading** — MATLAB `.mat` files, both legacy (v5/v6) and HDF5
  (v7.3, Spike2 exports), with automatic channel and sampling-rate detection.
- **Filtering** — bandpass, notch, baseline-wander removal.
- **R-peak detection** — five interchangeable methods (auto via NeuroKit2,
  Savitzky-Golay + derivative, continuous wavelet transform, envelope-max,
  and a classical-ML detector trainable from your own reviewed recordings),
  with manual edit/undo/redo and an interactive threshold slider.
- **RR artifact handling** — duplicate/non-physiological/ectopic-beat
  detection with a reviewable candidate list before anything is removed.
- **HRV analysis** — time-domain (SDNN, RMSSD, pNN6...), frequency-domain
  (LF/HF via Lomb-Scargle), and non-linear (Poincaré SD1/SD2, sample entropy,
  DFA).
- **Wave delineation** — P/Q/R/S/J/T landmarks and PR/QRS/QT/QTc intervals,
  with a calibratable per-recording wave template.
- **Arrhythmia classification** — bradycardia/tachycardia runs, pauses,
  ectopic-beat runs, irregular-rhythm runs, against a per-recording baseline.
- **Rolling HRV & epoch analysis**, and A/B segment comparison.
- **Session persistence** — save/restore a full analysis session as JSON,
  plus a local SQLite registry of recently analyzed recordings.
- **Export** — formatted Excel workbooks, GraphPad Prism `.pzfx`, PDF
  reports, CSV.
- **Theming** — light/dark presets, propagated live across the whole UI.

## Requirements

- Python ≥ 3.10
- See [`requirements.txt`](requirements.txt) for the full dependency list.
  `h5py` is required only for MATLAB v7.3/HDF5 files; `pzfx` only for
  GraphPad Prism export.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
# — or, for an editable install that also registers the `ecg-analysis`
#   console script and makes `import ecg` work from anywhere —
pip install -e .
```

## Usage

```bash
python -m ecg          # preferred entry point
python ecg_app.py       # backward-compatible entry point (same app)
ecg-analysis            # after `pip install -e .`
```

## Project layout

```
.
├── ecg/
│   ├── __init__.py       # package version, architecture overview
│   ├── __main__.py       # `python -m ecg` entry point
│   ├── core/             # pure NumPy/SciPy/pandas — no UI, no file I/O
│   │   ├── models.py         # species physiology constants, shared dataclasses
│   │   ├── filtering.py      # bandpass, notch, normalize, downsample
│   │   ├── detection.py      # R-peak detection, RR artifacts, arrhythmias
│   │   ├── analysis.py       # HRV time/freq/non-linear, interval stats
│   │   ├── wave_template.py  # beat-template calibration & delineation
│   │   └── ml_detector.py    # classical-ML R-peak detector, trainable from reviewed recordings
│   ├── io/                # file formats and persistence
│   │   ├── loaders.py        # .mat / HDF5 signal loading
│   │   ├── session.py        # session save/load (JSON)
│   │   ├── export.py         # Excel and GraphPad Prism exporters
│   │   └── db.py             # SQLite recording registry
│   └── ui/                # CustomTkinter desktop application
│       ├── app.py             # ECGApp main window
│       ├── *_controller.py    # one controller per responsibility (SRP)
│       ├── theme.py           # colour/font presets, live theme propagation
│       ├── plots.py           # matplotlib ↔ Tk canvas bridge
│       ├── dialogs.py         # secondary windows (theme, artifact review...)
│       ├── sidebar.py         # collapsible sections, interval verifier
│       ├── widgets.py         # reusable widget factories (stat tiles, quality gauge)
│       ├── wave_editor.py     # wave-template mini editor
│       └── state.py           # UI-facing state dataclasses
├── ecg_app.py             # thin backward-compatible entry point
├── setup.py               # `pip install -e .` packaging
├── requirements.txt
└── README.md / README.fr.md
```

`ecg.core` has no dependency on `ecg.ui` or `ecg.io` — the detection and HRV
algorithms are pure functions that can be tested, scripted, or reused without
launching the GUI.

## Notes for contributors

- No automated test suite exists yet; changes to `ecg.core` (detection,
  analysis, wave delineation) should be verified against a real or synthetic
  recording, not just a compile check.
- The UI is a single large window (`ecg/ui/app.py`) delegating to per-concern
  controllers in `ecg/ui/*_controller.py` — see each controller's docstring
  for its area of responsibility.
