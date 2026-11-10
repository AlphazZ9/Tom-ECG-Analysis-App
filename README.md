# ECG Analysis — Mouse Cardiac Electrophysiology Toolkit

*[Version française](README.fr.md)*

A desktop application (CustomTkinter) for analyzing mouse cardiac ECG
recordings: R-peak detection, HRV (time/frequency/non-linear/multiscale
entropy), time-resolved autonomic indices (CSI / CPI, after Candia-Rivera et
al. 2025) with event-locked statistics, P-QRS-T wave delineation,
rhythm/abnormal-event classification, and export to Excel / GraphPad Prism /
PDF.

Built around mouse-specific physiology (Thireau et al. 2008; Baudrie et al.
2007; Mitchell et al. 1998) rather than adapted human-ECG defaults — heart
rate range, pNN threshold, HRV frequency bands, and QTc correction formula
are all tuned for mice (typical resting HR ~500 bpm, 180–900 bpm range).
The app does not currently provide selectable, protocol-specific reference
ranges; interpret measurements against reference data appropriate to your
strain and experimental conditions.

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

### HRV analysis (7 linked sub-views)

- **RR/HR** — tachogram with automatic spike (sudden acceleration/
  deceleration) detection, click-to-navigate; RR-interval distribution
  histogram annotated with the fitted HRV triangular-index / TINN triangle.
- **Time Domain** — the full `nk.hrv_time()` metric set (SDNN, RMSSD, pNN6,
  SDANN, MedianNN, TINN, HTI...).
- **Frequency** — Welch PSD with mouse-specific VLF/LF/HF bands (not the
  human 0.04–0.15/0.15–0.4 Hz convention).
- **Non-linear** — Poincaré scatter with an SD1/SD2 ellipse overlay, sample
  entropy, DFA α1/α2, and a **multiscale entropy (MSEn)** curve across
  coarse-graining scales — catches loss of complexity a single-scale SampEn
  can miss. Beat-count truncation on long recordings (for runtime) is
  disclosed on-screen, not silent.
- **Epochs** — fixed-window HRV trends (HR, SDNN, RMSSD, pNN6) with
  low-beat-count windows visually flagged.
- **Rolling** — sliding-window HRV trends; HR/SDNN/RMSSD/pNN6 plus
  SD1/SD2/LF-normalized/HF-normalized/LF-HF as opt-in metrics, with a
  results table alongside the plot. Article-style Poincare options add
  rolling CCD, SD1 and SD2 using Ledoit-Wolf covariance; values are withheld
  below 10 valid successive RR pairs. For the article's full method — CSI /
  CPI from one Poincaré window per beat — use the Autonomic view.
- **Autonomic** — time-resolved cardiac sympathetic / parasympathetic
  indices (CSI / CPI) of Candia-Rivera et al. 2025 with a choice of
  covariance estimator, optional SPWVD LF / HF power, event-locked pre /
  post statistics around annotations, a cohort registry and a ks / kp
  calibration dialog. The paper's gains (ks = 3, kp = 12) are rat values,
  not validated in mice — see
  [Autonomic indices](#autonomic-indices-csi--cpi--candia-rivera-et-al-2025).

### Interval delineation

- P/Q/R/S/J/T landmark detection per beat, with a calibratable
  per-recording wave template and an interactive verifier.
- PR/QRS/QT/QTc distributions (violin + box plots).
- Selectable QTc correction formula (RR in ms):
  - **Mitchell** (default, mouse-calibrated; Mitchell et al. 1998) —
    QTc = QT / √(RR / 100), i.e. Bazett normalised to a 100 ms mouse RR
    (QTc equals QT at ~600 bpm);
  - **Bazett** (human) — QTc = QT / √(RR / 1000);
  - **Fridericia** (human) — QTc = QT / ∛(RR / 1000).

  Bazett and Fridericia are normalised to a 1 s human RR and inflate QTc
  ~2–3.5× at mouse heart rates; use them only to compare with studies that
  report them. The QT–RR relation is weak in mice, so read the
  **uncorrected QT** alongside any QTc. The formula used is recorded in
  the Excel (Cover sheet), Prism (project notes) and PDF exports.
- A QTc-vs-RR regression diagnostic scatter to judge whether the selected
  correction actually removed rate-dependence (a well-corrected QTc is flat
  against RR), instead of trusting one formula's number blindly.
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

### Comparison & export

- A/B segment comparison with superimposed RR tachograms, a full metric
  table (including LF/HF-normalized and SD1/SD2), and a **Mann-Whitney U
  test** on the two segments' RR-interval distributions.
- Session persistence — save/restore a full analysis session as JSON, plus
  a local SQLite registry of recently analyzed recordings and of the
  event-locked autonomic results saved for cohort statistics.
- Export — formatted Excel workbooks, GraphPad Prism `.pzfx`, a one-page
  PDF report (signal strip, key-metric table, Poincaré diagram,
  abnormal-events summary), a per-episode annotated
  PDF for abnormal events, and CSV.

### Interface

- Adaptive Workbench layout — toolbar, collapsible left/right panels, and a
  center workspace with a position-scrubber minimap, so the working area can
  be reclaimed on smaller screens.
- Light/dark theming, propagated live across the whole UI including every
  matplotlib plot.

## Autonomic indices (CSI / CPI) — Candia-Rivera et al. 2025

**HRV > Autonomic** computes the time-resolved cardiac sympathetic and
parasympathetic indices of Candia-Rivera et al., *Modelling the
time-resolved modulations of cardiac activity in rats*, J Physiol 2025
([doi:10.1113/JP288400](https://doi.org/10.1113/JP288400)), following the
authors' published MATLAB code
([github.com/diegocandiar/robust_hrv](https://github.com/diegocandiar/robust_hrv):
`compute_CSI_CVI_rat.m`, `robustCovHRV.m`, `compute_rCSI_rCVI_type.m`,
`compute_PWVD.m`).

### Method

Each RR interval carries the time of the beat that ends it. For every beat,
a sliding Poincaré window of *T* seconds (3 s, the paper's value for rats)
holds the intervals that ended in [t − T, t]. The window's successive pairs
(RRₙ, RRₙ₊₁) give **SD1** and **SD2** — the square roots of the minor and
major eigenvalues of their 2 × 2 covariance — and **D** (CCD), the distance
of the ellipse centre to the origin, √(m₁² + m₂²) of the two coordinate
means; each window is time-stamped at the median time of its intervals. The
whole-recording SD01, SD02 and D0 re-centre the beat-wise series, which are
resampled at 20 Hz (cubic spline), and

- **CSI = D̃ + ks · SD2**, D̃ being D mirrored about its mean (a shorter
  cycle raises CSI);
- **CPI = D + kp · SD1** (a longer cycle raises CPI).

The paper's printed equations put the mirrored D on CPI; the published code
mirrors it for CSI, which is what the physiology requires — the app follows
the code.

The app's version is gap-aware: RR intervals outside the sidebar Min / Max
RR bounds, or spanning a beat removed by artifact correction, are
excluded; Poincaré pairs never straddle them; a window with fewer than
**Min pairs** valid adjacent pairs (10 by default) is withheld (blank,
shaded grey) and no series is bridged across a gap. That floor is the
app's addition — the reference code computes every window; **Min pairs = 2**
reproduces it. A 3 s window holds 10 pairs only above ~200 bpm — lower
Min pairs or use a longer window for bradycardic or anaesthetised mice.
With nothing excluded and every window computed, the results equal the
reference code's (the test suite pins them against a literal
transliteration).

### Covariance options — and a paper / code discrepancy

| Method | What it does | Where it comes from |
|---|---|---|
| **Robust shrinkage** (default) | Schäfer & Strimmer (2005) shrinkage covariance — variances shrunk toward their median, correlation toward 0 (`robustCovHRV.m`) — with a 5 % trimmed-mean D. A window with a constant Poincaré coordinate has no defined correlation and is withheld. | what the paper **text** says was used |
| **Rat reference code** | plain `cov()` with a 5 % trimmed-mean D | what the published `compute_CSI_CVI_rat.m` actually runs (its `robustCovHRV` call is commented out) |
| **Exact** | plain covariance, plain-mean D | `compute_rCSI_rCVI_type.m`, `'exact'` |
| **Approximate** | closed form SD1 = √(var(ΔRR)/2), SD2 = √(2 var(RR) − var(ΔRR)/2), plain-mean D | `compute_rCSI_rCVI_type.m`, `'approximate'` |
| **Ledoit–Wolf** | sklearn shrinkage toward a scaled identity — for comparison only | not used by the paper |

The paper text and its published rat function disagree: the text says the
robust (shrinkage) estimator was used, the code uses the plain covariance
with trimmed means. Both are offered; the default follows the text, and the
method is recorded with every result and export. Both shrinkage
estimators round the Poincaré ellipse on short windows, by a
data-dependent amount. The robust one keeps SD1² + SD2² and in every window
raises SD1 and lowers SD2 (on ~25-pair synthetic mouse windows: SD1 +4 to
+7 %, SD2 −1 to −5 % vs the exact covariance). Ledoit–Wolf — the estimator
behind the Rolling view's article-style SD1 (LW) / SD2 (LW) — shrinks
toward a different target, usually by more (typically SD1 +3 to +12 %,
SD2 −4 to −10 % on 25–35-pair windows), and can even lower SD1 in some
windows.

### Rat calibration — not validated in mice

**ks = 3 and kp = 12 are the paper's values for spontaneously hypertensive
rats** (rest vs dobutamine: the inflection point of the paired Wilcoxon Z
over k = 1–20, then checked on WKY rats). They have **not been validated in
mice**. The view, its plots and every export (Excel Cover sheet, Prism
project notes, PDF report header) state the gains and this caveat.
Calibrate them on your own cohort (below) and report the gains with the
cohort they came from.

### Time-resolved LF / HF (SPWVD)

Tick **LF/HF (SPWVD)** to add LF and HF power from the smoothed pseudo
Wigner–Ville distribution, a port of `compute_PWVD.m` (Orini et al. 2012
ambiguity-domain kernel): the valid RR intervals are spline-resampled at
20 Hz, linearly detrended and turned into an analytic signal; the
Wigner–Ville distribution (4096 lag bins) is filtered in the ambiguity
domain by the kernel exp(−π (μ²)^λ), μ = (τ/τ₀)² + (ν/ν₀)², and summed
over the LF / HF bins of the HRV band preset (mouse default LF 0.4–1.5 Hz,
HF 1.5–5 Hz; the paper's rat bands were 0.1–1 and 1–3.5 Hz). The kernel
constants ν₀ = 0.03, τ₀ = 0.06, λ = 0.3 are the ones hard-coded in
`compute_PWVD.m` — the toolbox's only set, with no per-species variant (the
file's own defaults are Frr = 4 Hz and the human bands); the paper applied
them at 20 Hz to rats. They are defined in normalised units, so their
extent in seconds and Hz depends on the 20 Hz resampling and the 4096
bins. They are not validated in mice.

RR gaps longer than 2 s split the series: each stretch between them is
computed as its own series (its own spline, detrend and analytic signal —
nothing is smoothed across a gap, whereas the reference code bridges
everything) and the gaps are blank, as are stretches too short to compute
(under 10 s with the mouse bands: two periods of the LF floor and the
~12 s lag kernel). Shorter gaps are bridged by the spline; when they cover
more than 5 % of the time the LF / HF computation is refused. Long
recordings are processed in overlapping chunks (120 s cores, 60 s
margins) to bound memory; against a single pass, 30 s post / pre ratios
agree to ~0.1–1 % (data-dependent: largest after a large HR / HRV change
inside a chunk margin) and sample-wise values to a few % (up to ~6 % near
low-power samples). The power is unnormalised (a.u.): compare it within a
recording (pre vs post an event, LF vs HF trend), not with Welch band
powers. Allow about 40 s of computation per hour of recording.

### Event-locked statistics

The **Event-locked** toggle of the Autonomic view applies the paper's
statistical protocol to the computed indices:

1. Mark each condition change (an injection, a stimulus) with an
   **annotation**: its **start** is the event onset. Pick all annotations
   or one label in **Events**.
2. Set **Pre** / **Post** (30 s / 30 s by default) and click
   **⟳ Compute event-locked**. The per-event table gives the mean of CSI,
   CPI, HR, D, SD1, SD2 (and LF / HF) over each window and the post − pre
   delta (**📋 Copy for Excel** copies one row per event). The plot shows
   the group median ± MAD across the recording's events on a −pre…+post
   time axis with the onset at 0, z-scored per recording for display as in
   the paper's figures (**z-score** toggle; the table and tests always use
   raw values). An event is valid when **no other annotation starts inside
   its pre or post window or at its onset** — the paper's minimum duration
   per condition: successive doses closer than the windows would
   contaminate each other's baseline and response, so such events are
   INVALID (every annotation counts, also those the label filter leaves
   out) — and when both windows lie inside the CSI / CPI series and are at
   least 80 % covered by computed windows.
3. The within-recording Wilcoxon under the table is descriptive only: the
   events of one animal are repeated measures of that animal, not a
   population sample. It gives z, the test's p (scipy: exact for small
   samples) and p(z), the normal p of that z (MATLAB `signrank`'s
   approximate p) — the two can fall on either side of 0.05.
4. **💾 Save to cohort registry** stores the per-event pre / post / delta
   values in the local SQLite registry with the sidebar **Subject ID** and
   **Project name** and the configuration (method, window, ks, kp, pre /
   post) — one row per event and method / window. With the *All* filter,
   the recording's earlier events of the same method and window are
   replaced. **Set each animal's Subject ID before saving**: the save
   warns (orange) when it looks like the untouched default (`subject_01`)
   or when another recording of the project is already filed under it, as
   the per-subject calibration would pool those recordings as one animal.
   **Export CSV…** writes the same rows to a file.

### Calibrating ks / kp on your cohort

CSI and CPI are linear in the gains, so for any pre / post pair
ΔCSI = −ΔD + ks · ΔSD2 and ΔCPI = ΔD + kp · ΔSD1 (the re-centring constants
cancel): the stored per-event deltas of D, SD1 and SD2 recompute the
contrast for any k without reprocessing a recording.
**⚖ Calibrate ks/kp…** reads the registry (filtered by label and project,
one method / window configuration at a time — never mixed) and plots the
paired Wilcoxon Z (post vs pre) of CSI and CPI for k = 1–20. It opens on
the label with the most events — one condition contrast, as the paper's
rest vs dobutamine; **All labels** pools different contrasts and says so.
The paper's rule is the **inflection point of Z(k)**, from which Z
plateaus; the dialog marks a knee computed as an objective proxy for it
(Kneedle on Z after its last sign change) — check it against the curve.
Aggregate **per subject mean** (the default) so that an animal with several
events counts once: testing events as independent samples is
pseudo-replication (events without a Subject ID are grouped per
recording). The paper computed Z over trials (events: 41 SHR trials), so
**Per event** reproduces its procedure; per subject mean is this app's
stricter default.

Read the curve with three caveats, all shown in the dialog:

- **Z is a rank statistic.** It moves in discrete steps (a unit flipping
  sign, two units swapping rank) and cannot exceed a ceiling set by the
  number of units (2.20 for 6, 2.52 for 8, 3.06 for 12). A small cohort
  often sits on that ceiling, where Z cannot rank the gains: the dialog
  flags it, gives the k ranges of maximal |Z|, and offers no knee resting
  on fewer than 3 rank steps.
- **Z(k) can change sign** when the heart-rate and HRV terms move in
  opposite directions (e.g. a tachycardic drug that also lowers SD2): the
  index then shows one direction at small k and the opposite one at large
  k. The dialog gives the direction on each stretch; a knee on the
  reversed stretch is marked ⚠ and Apply asks for confirmation — choose
  the gain by the direction the physiology predicts.
- **In-sample testing.** Gains chosen on a cohort and tested on the same
  cohort give optimistic p-values (picking k among 20 values is a
  selection the ×2 Bonferroni does not correct); the paper fitted the
  gains on SHR rats and validated them on WKY rats. Tests at this cohort's
  own knees are marked *in-sample*: calibrate on one project or label and
  test on another.

The table gives, at the current gains, n, the median ± MAD before and
after, z, p, p(z) and the Bonferroni-adjusted p over the two indices.
**Apply ks = …, kp = …** writes the knee values into the Autonomic view's
ks / kp entries; nothing is recomputed until you press **⟳ Compute** (then
⟳ Compute event-locked) — until then the view's status says the entries
differ from the displayed result.

### Exports

With an Autonomic result, the Excel workbook gains an **Autonomic** sheet
(the beat-wise SD1 / SD2 / CCD windows, the recording-level SD01 / SD02 /
D0 and every parameter), **Autonomic_Series** (the 20 Hz CSI / CPI / HR / D
/ SD1 / SD2), **Autonomic_LFHF** and **EventLocked** sheets; the Prism file
gains CSI / CPI, D / SD1 / SD2, LF / HF and event pre / post tables; the
HRV > Autonomic figures are included in the figure and ZIP exports. The
Excel Cover sheet, the Prism project notes and the PDF report header state
the method, the window, ks / kp and the rat-calibration caveat, and the
span the indices cover (first / last R peak, analysis window) — flagged
when it is not the core HRV results' analysis window. The Autonomic view
needs only the peaks: before Analyze, **Export Excel** writes the
Autonomic results alone (a Cover sheet and the Autonomic sheets); the
other exports need Analyze. A result computed on another analysis window
than the current one is flagged in orange in the view.

### References

- Candia-Rivera D., Carrion-Falgarona S., de Vico Fallani F., Chavez M.
  (2025). Modelling the time-resolved modulations of cardiac activity in
  rats: a study on pharmacological autonomic stimulation. *J Physiol*.
  [doi:10.1113/JP288400](https://doi.org/10.1113/JP288400)
- Candia-Rivera D., de Vico Fallani F., Chavez M. (2025). Robust and
  time-resolved estimation of cardiac sympathetic and parasympathetic
  indices. *R Soc Open Sci* 12:240750.
  [doi:10.1098/rsos.240750](https://doi.org/10.1098/rsos.240750)
- Reference code: [github.com/diegocandiar/robust_hrv](https://github.com/diegocandiar/robust_hrv)
- Schäfer J., Strimmer K. (2005). A shrinkage approach to large-scale
  covariance matrix estimation and implications for functional genomics.
  *Stat Appl Genet Mol Biol* 4:32.
- Orini M., Bailón R., Mainardi L. T., Laguna P., Flandrin P. (2012).
  Characterization of dynamic interactions between cardiovascular signals
  by time-frequency coherence. *IEEE Trans Biomed Eng* 59(3):663–673.
- Satopää V., Albrecht J., Irwin D., Raghavan B. (2011). Finding a
  "kneedle" in a haystack: detecting knee points in system behavior.
  *ICDCS Workshops* (the knee proxy).

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
│   │   └── ml_detector.py    # classical-ML R-peak detector, trainable from
│   │                         # reviewed recordings, hold-out confusion matrix
│   ├── io/                # file formats and persistence
│   │   ├── loaders.py        # .mat / HDF5 signal loading
│   │   ├── session.py        # session save/load (JSON)
│   │   ├── export.py         # Excel and GraphPad Prism exporters
│   │   └── db.py             # SQLite recording registry + event-locked
│   │                         # cohort table
│   └── ui/                # CustomTkinter desktop application
│       ├── app.py             # ECGApp main window (Adaptive Workbench layout)
│       ├── *_controller.py    # one controller per responsibility (SRP):
│       │                      # analysis, detection, signal, navigation,
│       │                      # session, export, event-locked
│       ├── autonomic_view.py  # HRV > Autonomic helpers (pure, tested)
│       ├── event_locked_view.py # Event-locked panel + ks / kp calibration
│       │                      # helpers (pure, tested)
│       ├── theme.py           # colour/font presets, live theme propagation
│       ├── plots.py           # matplotlib ↔ Tk canvas bridge
│       ├── dialogs.py         # secondary windows (theme, artifact review,
│       │                      # ML training, ks / kp calibration...)
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
