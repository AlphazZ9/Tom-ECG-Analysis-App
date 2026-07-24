# ECG Analysis — Mouse Cardiac Electrophysiology Toolkit

*[Version française](README.fr.md)*

A desktop application (CustomTkinter) for analyzing mouse cardiac ECG
recordings: R-peak detection, HRV (time/frequency/non-linear/multiscale
entropy), P-QRS-T wave delineation, rhythm/abnormal-event classification,
and export to Excel / GraphPad Prism / PDF.

Built around mouse-specific physiology (Thireau et al. 2008; Baudrie et al.
2007; Mitchell et al. 1998) rather than adapted human-ECG defaults — heart
rate range, pNN threshold, HRV frequency bands, and QTc correction formula
are all tuned for mice (typical resting HR ~500 bpm, 180–900 bpm range).
Reference ranges for every metric are driven by a selectable experimental
context (telemetry/awake, isoflurane, ketamine-xylazine, surface electrodes,
or your own custom-defined context), not a single generic "normal" band.

**New to the app?** See the [User Guide](docs/USER_GUIDE.md) for a
screenshot walkthrough of the actual workflow — loading a recording,
detection, HRV analysis, abnormal events, and export.

## Features

### Signal & R-peak detection

- **Signal loading** — MATLAB `.mat` files, both legacy (v5/v6) and HDF5
  (v7.3, Spike2 exports), with automatic channel and sampling-rate detection.
- **Filtering** — bandpass, notch, baseline-wander removal, with a live
  before/after filter preview.
- **R-peak detection** — five interchangeable methods (auto via NeuroKit2,
  Savitzky-Golay + derivative, continuous wavelet transform, envelope-max,
  and a classical-ML detector trainable from your own reviewed recordings),
  with manual edit/undo/redo and an interactive threshold slider.
- **ML R-peak detector** — trains a Random Forest from recordings you mark
  "Verified for training"; the training dialog reports hold-out accuracy,
  F1, precision, recall, and a confusion matrix for the binary peak /
  non-peak classifier.
- **RR artifact handling** — duplicate/non-physiological/ectopic-beat
  detection with a reviewable candidate list before anything is removed.
- **Signal quality** — a 0–100 composite quality score and gauge from
  beat-template correlation, plus both a beat-count and a duration-weighted
  "% time below correlation threshold" KPI (the latter catches noisy
  stretches with very few detected beats that a pure beat-count percentage
  would understate).

### HRV analysis (6 linked sub-views)

- **RR/HR** — tachogram with automatic spike (sudden acceleration/
  deceleration) detection, click-to-navigate; RR-interval distribution
  histogram annotated with the fitted HRV triangular-index / TINN triangle.
- **Time Domain** — the full `nk.hrv_time()` metric set (SDNN, RMSSD, pNN6,
  SDANN, MedianNN, TINN, HTI...), with ✓ / ~ / ↑ / ↓ reference-range status
  against the active experimental context for the key metrics.
- **Frequency** — Welch PSD with mouse-specific VLF/LF/HF bands (not the
  human 0.04–0.15/0.15–0.4 Hz convention), plus a radar/spider profile chart
  where each axis is normalized against *its own* physiological reference
  range (not min-maxed against the other axes).
- **Non-linear** — Poincaré scatter with an SD1/SD2 ellipse overlay, sample
  entropy, DFA α1/α2, and a **multiscale entropy (MSEn)** curve across
  coarse-graining scales — catches loss of complexity a single-scale SampEn
  can miss. Beat-count truncation on long recordings (for runtime) is
  disclosed on-screen, not silent.
- **Epochs** — fixed-window HRV trends (HR, SDNN, RMSSD, pNN6) with
  reference-band shading and low-beat-count windows visually flagged.
- **Rolling** — sliding-window HRV trends; HR/SDNN/RMSSD/pNN6 plus
  SD1/SD2/LF-normalized/HF-normalized/LF-HF as opt-in metrics, all against
  reference bands, with a results table alongside the plot.

### Interval delineation

- P/Q/R/S/J/T landmark detection per beat, with a calibratable
  per-recording wave template and an interactive verifier.
- PR/QRS/QT/QTc distributions (violin + box plots) against context-aware
  reference bands.
- Selectable QTc correction formula — **Mitchell** (default, mouse-
  calibrated cube-root correction), **Bazett**, or **Hodges** (linear HR
  correction, reported in the literature as the best-performing corrector
  for rodents).
- A QT-vs-RR regression diagnostic scatter to judge whether the selected
  correction actually removed rate-dependence, instead of trusting one
  formula's number blindly.
- QT dispersion.

### Abnormal events (rhythm classification)

- Bradycardia/tachycardia runs, pauses, ectopic-beat (ESV) runs, and
  irregular-rhythm runs, classified against each recording's own baseline
  heart rate (not a fixed human bpm cutoff).
- **AV conduction-delay flagging** from sustained PR-interval prolongation,
  once Interval Delineation has been run — closes the gap where rhythm
  classification previously only ever saw R-peak timing.
- A color-coded event-type ribbon timeline (at-a-glance event clustering)
  alongside per-episode cards; click a card or ribbon segment to zoom the
  raw ECG to that episode, with a manual peak-edit mode for correcting
  detection right there.

### Experimental context & reference ranges

- Four built-in mouse physiological contexts — **telemetry / awake**,
  **isoflurane**, **ketamine/xylazine**, **surface electrodes** — each with
  its own HR, RR, SDNN, RMSSD, pNN6, LF/HF, SD1/SD2, and PR/QRS/QT/QTc
  reference ranges, selectable from the Parameters dialog.
- A user-editable **custom context** (e.g. for a specific strain or
  protocol) with its own reference-range editor, persisted to disk and
  selectable everywhere alongside the built-ins — every reference band, ✓/~/
  ↑/↓ status, and the radar chart all read from whichever context is active.

### Comparison & export

- A/B segment comparison with superimposed RR tachograms, a full metric
  table (including LF/HF-normalized and SD1/SD2), and a **Mann-Whitney U
  test** on the two segments' RR-interval distributions.
- Session persistence — save/restore a full analysis session as JSON, plus
  a local SQLite registry of recently analyzed recordings.
- Export — formatted Excel workbooks, GraphPad Prism `.pzfx`, a one-page
  PDF report (signal strip, key-metric table, Poincaré diagram, HRV radar,
  abnormal-events summary, context reference band), a per-episode annotated
  PDF for abnormal events, and CSV.

### Interface

- Adaptive Workbench layout — toolbar, collapsible left/right panels, and a
  center workspace with a position-scrubber minimap, so the working area can
  be reclaimed on smaller screens.
- Light/dark theming, propagated live across the whole UI including every
  matplotlib plot.

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
│   │   ├── models.py         # species physiology constants, experimental-context
│   │   │                     # reference ranges (built-in + user-editable custom),
│   │   │                     # shared dataclasses
│   │   ├── filtering.py      # bandpass, notch, normalize, downsample
│   │   ├── detection.py      # R-peak detection, RR artifacts, rhythm /
│   │   │                     # abnormal-event classification (incl. AV delay)
│   │   ├── analysis.py       # HRV time/freq/non-linear (incl. multiscale
│   │   │                     # entropy), interval stats
│   │   ├── wave_template.py  # beat-template calibration & delineation
│   │   └── ml_detector.py    # classical-ML R-peak detector, trainable from
│   │                         # reviewed recordings, hold-out confusion matrix
│   ├── io/                # file formats and persistence
│   │   ├── loaders.py        # .mat / HDF5 signal loading
│   │   ├── session.py        # session save/load (JSON)
│   │   ├── export.py         # Excel and GraphPad Prism exporters
│   │   └── db.py             # SQLite recording registry
│   └── ui/                # CustomTkinter desktop application
│       ├── app.py             # ECGApp main window (Adaptive Workbench layout)
│       ├── *_controller.py    # one controller per responsibility (SRP):
│       │                      # analysis, detection, signal, navigation,
│       │                      # session, export
│       ├── theme.py           # colour/font presets, live theme propagation
│       ├── plots.py           # matplotlib ↔ Tk canvas bridge
│       ├── dialogs.py         # secondary windows (theme, artifact review,
│       │                      # ML training, custom context editor...)
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
- Reference-range numbers (in `EXPERIMENTAL_CONTEXTS`, `ecg/core/models.py`)
  are sourced from the cited literature per context; if you add or adjust a
  built-in context's bounds, cite the source in its `description` field the
  same way the existing four do — this app is used to interpret real animal
  data, so unsourced numbers don't belong here. Use the in-app "Custom
  Context" editor for anything strain- or protocol-specific instead of
  hard-coding a new preset.
