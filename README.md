# ECG Analysis — Mouse Cardiac Electrophysiology Toolkit

*[Version française](README.fr.md)*

A desktop application for analysing **mouse** ECG recordings: it finds the
heartbeats, computes heart-rate variability and autonomic indices, measures
the P-QRS-T intervals, flags abnormal rhythm events and exports everything to
Excel, GraphPad Prism and PDF.

Its defaults are set for mouse physiology, not borrowed from human ECG:
heart-rate range (typical resting HR ~500 bpm, 180–900 bpm range), pNN
threshold, HRV frequency bands and QTc formula (Thireau et al. 2008; Baudrie
et al. 2007; Mitchell et al. 1998). The app does not grade results against
protocol-specific reference ranges: compare them with reference data for your
strain and experimental conditions.

## Download / Get started

1. **Download the app (recommended, nothing to install).** Go to the
   [latest release](https://github.com/AlphazZ9/Tom-ECG-Analysis-App/releases/latest)
   and take, under **Assets**, the zip for your computer: Windows, macOS
   Apple Silicon (M1, M2, M3…) or macOS Intel. Extract it and open
   *Tom ECG Analysis*. The Mac app needs macOS 14 (Sonoma) or later.
2. **Or use the one-click launcher** (no Python to install yourself): get
   the repository folder (`git clone`, or on GitHub **Code > Download ZIP**,
   then extract the whole ZIP) and double-click `Lancer ECG.bat` (Windows)
   or `Lancer ECG.command` (macOS) — details [below](#one-click-launcher).
3. **Or, for developers:** `pip install -r requirements.txt`, then
   `python ecg_app.py` (details in [Development](docs/DEVELOPMENT.md)).

**First launch warnings.** The apps are not signed, so the system warns once.
Windows (SmartScreen): *More info > Run anyway*. macOS: right-click (or
Ctrl-click) the app > **Open**; on macOS 15+, *System Settings > Privacy &
Security > Open Anyway*. Step-by-step help:
[packaging/README_BUILD.md](packaging/README_BUILD.md#installer-et-lancer)
(in French).

### One-click launcher

- **First launch needs an Internet connection.** The launcher installs
  [uv](https://docs.astral.sh/uv/) if missing, a private Python 3.12 (never
  your system Python, nor Apple's with its old Tk 8.5) and the libraries of
  `requirements.txt` — a few minutes and roughly 400 MB, with a progress
  window. Later launches take a few seconds and work offline; libraries are
  reinstalled only when `requirements.txt` or
  `packaging/constraints.txt` (the tested versions) changes.
- **Folder used** (delete it to reinstall; your settings in `~/.ecg_*` are
  kept): Windows `%LOCALAPPDATA%\TomECG`, macOS
  `~/Library/Application Support/TomECG`.
- **Windows:** no console window; Python errors go to
  `dernier_lancement.log` in that folder. `"Lancer ECG.bat" debug` from a
  command prompt keeps the console open.
- **macOS (Gatekeeper), once:** on macOS 15+, double-click, click **Done**,
  then *System Settings > Privacy & Security* > **Open Anyway** next to
  `Lancer ECG.command`. On macOS 14 or earlier: right-click > **Open**. The
  macOS launcher has not yet been tested on a real Mac.
- Full details (caches, `xattr` / `chmod` fallback, internals):
  [Development](docs/DEVELOPMENT.md#one-click-launchers-full-details-and-internals).

## What it does

- **Signal & R-peak detection** — loads MATLAB `.mat` files (v5/v6 and
  v7.3 / Spike2), filters, detects R peaks with five methods (including a
  trainable ML detector), reviews artifacts and scores signal quality.
  [Details](docs/METHODS.md#signal--r-peak-detection)
- **Tabs** — Signal & Detection, Rhythm & HRV, Time course, Compare segments, Morphology &
  Intervals, Events, Group (all the animals of an experiment) and Report. [Details](docs/METHODS.md#tabs)
- **HRV** — time domain, frequency (mouse VLF / LF / HF bands, powers in
  ms²), non-linear (Poincaré, DFA, entropy, multiscale entropy), epochs and
  sliding windows. [Details](docs/METHODS.md#hrv-analysis)
- **Autonomic indices (CSI / CPI)** — time-resolved sympathetic /
  parasympathetic indices after Candia-Rivera et al. 2025, SPWVD LF / HF,
  event-locked pre / post statistics and ks / kp calibration. The paper's
  gains are rat values, not validated in mice.
  [Details](docs/METHODS.md#autonomic-indices-csi--cpi--candia-rivera-et-al-2025)
- **Interval delineation** — P, QRS onset, R, S, J point and T end on every
  beat; PR, QRS, QT and QTc (Mitchell by default), on a rolling median of
  beats. [Details](docs/METHODS.md#interval-delineation)
- **Abnormal events** — bradycardia / tachycardia runs, pauses, premature
  beats, irregular runs and AV conduction delay, judged against each
  recording's own baseline. [Details](docs/METHODS.md#abnormal-events-rhythm-classification)
- **Comparison & export** — A/B segment comparison, saved sessions, cohort registry, Excel,
  Prism `.pzfx`, PDF and CSV, with the analysis provenance in every report.
  [Details](docs/METHODS.md#comparison--export)
- **Interface** — full-width layout from 1280 px wide, light / dark theme.
  [Details](docs/METHODS.md#interface)

## Typical workflow

1. [Launch the app](docs/USER_GUIDE.md#launching-the-app) and
   [load a recording](docs/USER_GUIDE.md#loading-a-recording) (`.mat` file,
   choose the ECG channel).
2. [Detect the R peaks](docs/USER_GUIDE.md#r-peak-detection) and
   [check the detection](docs/USER_GUIDE.md#reading-the-detection-view):
   correct missed or extra beats, review artifacts.
3. Click **Analyze** and read the [HRV results](docs/USER_GUIDE.md#hrv-analysis).
4. If needed, compute the
   [autonomic indices](docs/USER_GUIDE.md#autonomic-indices-csi--cpi--candia-rivera-et-al-2025)
   and the [intervals](docs/USER_GUIDE.md#interval-delineation).
5. Review the [abnormal events](docs/USER_GUIDE.md#abnormal-events).
6. Check the [Report tab](docs/USER_GUIDE.md#report-tab), then
   [export and save your work](docs/USER_GUIDE.md#exporting--saving-your-work).

## Documentation

- [User Guide](docs/USER_GUIDE.md) — screenshot walkthrough of the whole
  workflow.
- [Methods](docs/METHODS.md) — how every value is computed, defaults,
  thresholds, caveats, changes between versions, references.
- [Glossary of variables](docs/GLOSSAIRE_VARIABLES.md) — every computed
  variable explained (in French).
- [Development](docs/DEVELOPMENT.md) — running from source, project layout,
  tests, launcher internals.
- [Building and publishing the apps](packaging/README_BUILD.md) — how a new
  version is released (in French).

## Citation / references

The autonomic indices follow Candia-Rivera et al., *J Physiol* 2025
([doi:10.1113/JP288400](https://doi.org/10.1113/JP288400)) and their
[published code](https://github.com/diegocandiar/robust_hrv). The full
reference list is in [Methods > References](docs/METHODS.md#references).
