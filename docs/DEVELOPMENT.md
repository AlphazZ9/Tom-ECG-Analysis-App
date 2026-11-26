# Development

*[Back to the README](../README.md) · [Methods](METHODS.md) · [Building and publishing the apps (FR)](../packaging/README_BUILD.md)*

Running from source, the code layout, and how the one-click launchers and the
standalone apps are made.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project layout](#project-layout)
- [Notes for contributors](#notes-for-contributors)
- [One-click launchers: full details and internals](#one-click-launchers-full-details-and-internals)
- [Standalone apps and publishing a version](#standalone-apps-and-publishing-a-version)

## Requirements

- Python ≥ 3.10
- See [`requirements.txt`](../requirements.txt) for the full dependency list.
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

```text
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
│   │   ├── autonomic.py      # time-resolved CSI / CPI (Candia-Rivera 2025)
│   │   ├── spwvd.py          # time-resolved LF / HF (smoothed pseudo WVD)
│   │   ├── event_locked.py   # pre / post window means, Wilcoxon, ks / kp
│   │   │                     # calibration, per-subject aggregation
│   │   ├── wave_template.py  # beat-template calibration & delineation
│   │   ├── annotation_suggest.py # annotations suggested from marker /
│   │   │                     # keyboard channels and stimulus pulses
│   │   └── ml_detector.py    # classical-ML R-peak detector, trainable from
│   │                         # reviewed recordings, hold-out confusion matrix
│   ├── io/                # file formats and persistence
│   │   ├── loaders.py        # Spike2 .smr, .mat / HDF5 signal and marker loading
│   │   ├── event_channels.py # marker, keyboard and stimulus channels of a file
│   │   ├── group_data.py     # the group table (one animal per recording)
│   │   ├── group_compare.py  # Group tab analyses and statistics
│   │   ├── session.py        # session save/load (JSON)
│   │   ├── export.py         # Excel and GraphPad Prism exporters
│   │   └── db.py             # SQLite recording registry + event-locked
│   │                         # cohort table
│   └── ui/                # CustomTkinter desktop application
│       ├── app.py             # ECGApp main window (full-width layout, no sidebars)
│       ├── app_*.py, tab_*.py # ECGApp mixins: header, detection bar and one module per tab
│       ├── context_strip.py, tabbar.py, freshness.py # strip, tab states
│       ├── report_view.py     # Report tab and text report helpers (pure, tested)
│       ├── *_controller.py    # one controller per responsibility (SRP):
│       │                      # analysis, detection, signal, navigation,
│       │                      # session, export, event-locked
│       ├── autonomic_view.py  # Time course > Autonomic helpers (pure, tested)
│       ├── event_locked_view.py # Event-locked panel + ks / kp calibration
│       │                      # helpers (pure, tested)
│       ├── theme.py           # colour/font presets, live theme propagation
│       ├── plots.py           # matplotlib ↔ Tk canvas bridge
│       ├── dialogs.py         # secondary windows (theme, artifact review,
│       │                      # ML training, ks / kp calibration...)
│       ├── popover.py         # floating panels: Recording sheet, Filters, ⋯, Report > More
│       ├── widgets.py         # reusable widget factories (stat tiles, quality gauge)
│       ├── wave_editor.py     # wave-template mini editor
│       └── state.py           # UI-facing state dataclasses
├── ecg_app.py             # thin backward-compatible entry point
├── setup.py               # `pip install -e .` packaging
├── requirements.txt
├── docs/                # User Guide, Methods, Glossary, this file
│   └── archive/          # dated working notes (audits, hand-over), for the record
├── packaging/           # PyInstaller spec, constraints.txt, README_BUILD.md
├── prototypes/qt_view/  # PySide6 + pyqtgraph prototype of the detection view
│                        # and its benchmark (not part of the app)
├── tests/               # pytest suite (core, io, ui), run by .github/workflows/tests.yml
├── tools/               # build_icons.py (app icons from packaging/icon/)
├── Lancer ECG.bat / Lancer ECG.command  # one-click launchers
└── README.md / README.fr.md
```

`ecg.core` has no dependency on `ecg.ui` or `ecg.io` — the detection and HRV
algorithms are pure functions that can be tested, scripted, or reused without
launching the GUI.

## Notes for contributors

- Run `python -m pytest -q` for the automated suite. It covers detection,
  HRV interval validation, segment statistics, and ML persistence/splitting;
  GUI workflows, file-format variations, and wave delineation still need
  broader integration coverage.
- The UI is a single large window (`ecg/ui/app.py`) delegating to per-concern
  controllers in `ecg/ui/*_controller.py` — see each controller's docstring
  for its area of responsibility.
- Interpret measurements using reference data appropriate to the animal
  strain and experimental protocol; the application does not currently grade
  metrics against protocol-specific reference ranges.

## One-click launchers: full details and internals

For users who just want to run the app (no Python knowledge needed). Get
the repository folder (`git clone`, or on GitHub **Code > Download ZIP**,
then extract the whole ZIP), and double-click the launcher in that folder:

| System | Launcher |
| --- | --- |
| Windows | `Lancer ECG.bat` |
| macOS (Apple Silicon or Intel) | `Lancer ECG.command` |

- **First launch: an Internet connection is required.** The launcher
  installs [uv](https://docs.astral.sh/uv/) if it is missing, a private
  Python 3.12 (never your system Python, nor Apple's Python with its old
  Tk 8.5) and the libraries from `requirements.txt` — a few minutes and
  roughly 400 MB. A window shows the progress (messages in French).
- **Later launches** start the app directly, in a few seconds, and work
  offline. Libraries are reinstalled automatically only when
  `requirements.txt` changes (e.g. after updating the folder).
- **Folder used** (delete it to reinstall from scratch; your settings in
  `~/.ecg_*` are not touched):
  - Windows: `%LOCALAPPDATA%\TomECG` (uv's download cache is in
    `%LOCALAPPDATA%\uv\cache`).
  - macOS: `~/Library/Application Support/TomECG` (cache: `~/.cache/uv`).
- **Windows:** the app opens without a console window; Python errors, if
  any, are written to `dernier_lancement.log` in the folder above. Running
  `"Lancer ECG.bat" debug` from a command prompt keeps the console open to
  show errors. If SmartScreen warns about the file, click *More info >
  Run anyway*.
- **macOS, first launch (Gatekeeper):** on macOS 15 (Sequoia) or later,
  double-click once, click **Done**, then open *System Settings > Privacy &
  Security*, scroll down, click **Open Anyway** next to `Lancer ECG.command`
  and confirm with your password. On macOS 14 or earlier, right-click (or
  Ctrl-click) `Lancer ECG.command` > **Open**, then confirm. Only needed
  once. If that is not enough, run in Terminal: `xattr -d com.apple.quarantine "/path/to/Lancer
  ECG.command"` (drag the file into the Terminal window to type its path),
  and `chmod +x` on the same path if macOS says the file is not executable.
  The macOS launcher has not yet been tested on a real Mac.

**Internals.** The launchers install the libraries of `requirements.txt`
pinned to the tested versions of
[`packaging/constraints.txt`](../packaging/constraints.txt)
(`uv pip install -r requirements.txt -c packaging/constraints.txt`), not the
latest releases. They store a SHA-256 fingerprint of both files in the folder
above and reinstall only when one of them changes. They stop with an error
if `ecg_app.py`, `requirements.txt` or `packaging/constraints.txt` cannot be
found next to them, i.e. if the launcher was moved out of the extracted
repository folder.

## Standalone apps and publishing a version

The ready-to-run apps (Windows, macOS Apple Silicon, macOS Intel) are built by
GitHub Actions from `packaging/ecg.spec` (PyInstaller) and published on the
[Releases page](https://github.com/AlphazZ9/Tom-ECG-Analysis-App/releases/latest).
How to publish a new version, and the first-launch warnings of these unsigned
apps, are explained in [`packaging/README_BUILD.md`](../packaging/README_BUILD.md)
(in French).
