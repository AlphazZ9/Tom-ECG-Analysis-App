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

### Tabs

Five analysis tabs and a report, organised by question: **Signal & Detection**,
**Rhythm & HRV**, **Time course** (Epochs, Sliding, Autonomic, Event-locked),
**Morphology & Intervals** (steps: mean beat, landmarks, intervals), **Events** and
**Report** (provenance, module states, Include, exports and text report). One
context strip above every tab says what the numbers rest on; each tab head carries
one state (up to date, out of date, not computed); the screens, the text report, the
PDF's key metrics and the **Metrics** tables of Excel and Prism read one table of value,
unit, basis n and state (the other sheets and tables hold the series behind them). A
session saved on a former tab reopens on its new place.

Older entries below name the tabs of their time: *Summary* is now Report, *HRV*
sub-views are on Rhythm & HRV and Time course, *Rolling* is Sliding, *Beat Template*
and *Intervals* are Morphology & Intervals, *Abnormal Events* is Events.

### HRV analysis

- **RR/HR** — tachogram with automatic spike (sudden acceleration/
  deceleration) detection, click-to-navigate; RR-interval distribution
  histogram annotated with the fitted HRV triangular-index / TINN triangle.
- **Time Domain** — the full `nk.hrv_time()` metric set (SDNN, RMSSD, pNN6,
  SDANN, MedianNN, TINN, HTI...).
- **Frequency** — Welch PSD with mouse-specific VLF/LF/HF bands (not the
  human 0.04–0.15/0.15–0.4 Hz convention). One spectrum for everything: the
  valid RR intervals are resampled at 20 Hz (cubic spline on their true
  times) and Welch's method is applied (Hann window, 75 % overlap,
  ≈0.02 Hz resolution); the PSD plot draws exactly that spectrum, and
  VLF / LF / HF are its integrals over the shaded bands, as **absolute
  powers in ms²**. **LFn / HFn** are the Task Force normalised units,
  LF or HF / (LF + HF); LF/HF = LF / HF. The same estimator is used by
  Rolling HRV and Compare Segments.
- **Non-linear** — Poincaré scatter with an SD1/SD2 ellipse overlay, sample
  entropy, DFA α1/α2, and a **multiscale entropy (MSEn)** curve across
  coarse-graining scales — catches loss of complexity a single-scale SampEn
  can miss. Poincaré SD1/SD2, DFA and the asymmetry/fragmentation indices
  use **every valid beat of the analysis window**. Only the slow (O(N²))
  entropy block — SampEn, ApEn, FuzzyEn, MSE, CD, … — is capped for
  runtime, to **Entropy cap (RR)** valid intervals (default 2000, set in the
  Non-linear tab; 0 = no cap, a negative entry falls back to 2000), taken as the contiguous block with the
  fewest excluded intervals; the count and time span of that block are
  shown in the status line, on the MSE curve, in the Summary report and on
  the Excel Cover sheet.
- **One definition per metric.** Mean HR is **60000 / mean NN** everywhere
  (Summary, KPI bar, Excel, Compare Segments, Rolling, Epochs, tachogram
  titles); the SD of HR and the min / max HR describe the instantaneous HR
  60000 / RRᵢ — sample SD (n − 1) and **2nd / 98th percentiles**, the same
  in the Summary, KPI bar, Excel and Compare Segments, and labelled as such. Poincaré **SD1 / SD2** have a single
  definition — SD of (RRᵢ − RRᵢ₊₁)/√2 and of (RRᵢ + RRᵢ₊₁)/√2 over
  successive pairs of adjacent valid intervals (never across an excluded
  one) — used by the Non-linear panel, Rolling HRV, Compare Segments and the
  Summary / exports. The Ledoit-Wolf (SD1 LW / SD2 LW in Rolling) and the
  Autonomic covariance estimators are explicitly named variants.
  **Changed numbers (this version):** earlier versions computed every
  non-linear metric, SD1/SD2 included, on the **first 1000 beats** only (a
  few minutes of a long recording); took mean HR as the mean of the
  instantaneous HR (biased upwards, typically by 0.1–1 bpm, more with high
  variability); and took VLF / LF / HF from NeuroKit's spectrum (100 Hz
  resampling, PSD normalised to its maximum: the "powers" were in arbitrary
  units, mislabelled n.u.) while the PSD plot drew a different spectrum, and
  LFn / HFn were LF or HF / total power including VLF and 5–20 Hz.
  Compare Segments' LF / HF used a third Welch pipeline, and its HR SD /
  min / max were the absolute extremes (the Summary used percentiles and a
  population SD; the HR SD now reads very slightly higher). A restored
  older session has its mean HR / HR SD / range recomputed from the saved
  valid RR intervals (said in the status bar); an older frequency result
  cannot be recomputed without re-running, so it is kept but labelled
  **"legacy NeuroKit spectrum"** (arbitrary units, LFn = LF / TP) in the
  Frequency table, the Summary, the PDF, the Excel Cover / HRV_Frequency
  sheet and Prism instead of ms² / n.u. Re-run Frequency and Non-linear HRV
  on older sessions. The raw **HRV_Frequency** sheet's headers carry their
  units (e.g. `HRV_LF (ms²)`, `HRV_LFn (fraction of LF+HF)` — fractions,
  shown ×100 as n.u. everywhere else).
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

- P/Q/R/S/J/T landmark detection per beat, plus the **QRS onset** and the
  **QRS offset (J point)**, with a calibratable per-recording wave template
  and an interactive verifier.
- Standard interval definitions, measured on every beat:
  - **PR** = P onset → QRS onset;
  - **QRS** = QRS onset → QRS offset (J point);
  - **QT** = QRS onset → T end;
  - **QTc** = QT corrected with the **preceding** RR (previous R → this R).

  QRS onset is the first departure from the PR-segment baseline (Q-wave
  onset, or the R upstroke when there is no Q); the J point is where the S
  wave returns to that baseline (or the notch before the J wave when it
  never does). Both are placed where a straight line fitted (least squares)
  through the 30–70 % part of the deflection meets the baseline
  (threshold: 5 % of the R amplitude or 3× the PR-segment noise for the R
  wave; a Q or S wave counts once it dips more than 2 % of R or 3× the
  noise below the baseline). P onset is where the same kind of line
  through the P-wave upstroke meets the PR baseline, searched up to 30 ms
  before the P peak (when a P riding on the previous T never returns to
  that baseline, the line through the part of the upstroke above the T–P
  junction is used, not earlier than that junction). The P peak itself is
  searched after the previous beat's T peak and measured against the PR
  baseline (at mouse heart rates the beat window reaches back onto the
  previous T wave: a baseline taken there read PR ~60 ms instead of 42 at
  500 bpm and raised false AV-delay events); a beat with no P wave above
  3× the noise gets no P peak and no PR, rather than one made of noise or
  of the previous T tail. The PR
  baseline is the last flat stretch found on each beat before the Q/R
  (not a fixed window), so a widened QRS is measured in full. The mouse J
  wave is early repolarisation (Boukens et al. 2014, J Physiol 592:4613),
  so it is counted in QT, not in QRS. T end is where the same kind of line
  through the 70–30 % part of the T-wave trailing edge meets the PR
  baseline; the mouse T-wave end is gradual and ambiguous, so check its
  placement on the mean beat. Each beat is delineated in its own window:
  up to ±100 ms, at most 70 % of its preceding RR before R and of its
  following RR after R (a neighbour closer than Min RR is ignored); the
  exports and the Summary report state how many beats had a shortened
  window. QTc uses the preceding RR only when it lies inside the Min / Max
  RR bounds used for HRV. P onset, QRS onset, J point and T end are **not**
  clamped to the template windows: a landmark that cannot be placed inside
  the beat is left empty (NaN) instead of being set to a window edge, and
  a template auto-updated by one recording no longer shortens the next
  recording's QRS or QT. **Changed numbers (this version):** the beat
  window used to be ±45 % of the recording's *shortest* RR, so one
  premature beat shortened every beat's window (QT lost on all beats, PR
  cut) and at ~500 bpm the T end fell outside it; P onset was capped about
  9 ms before the P peak by the default template (PR 3–6 ms short with a
  12–15 ms P upstroke, long PRs truncated to plausible values — they are
  now measured, or rejected beyond 80 ms); a Q wave of 2–5 % of R was
  skipped (QRS onset ~3 ms late: QRS and QT shorter, PR longer); QTc was
  empty for every beat with an RR above the default 333 ms even when Max
  RR was widened. The
  line fit removes a noise-dependent shortening of QT (several ms at
  2–4 kHz) and of QRS; QRS onsets earlier than −22 ms and T ends earlier
  than 60 ms after R (default template) were previously reported at those
  limits. **Changed numbers (earlier):** earlier versions
  measured PR to the Q nadir, QRS from the Q nadir to the S nadir and QT
  from the R peak, and paired QTc with the following RR — QRS and QT are
  now longer (closer to the literature, e.g. Mitchell et al. 1998) and PR
  slightly shorter. Sessions saved before this change keep their stored
  intervals (flagged "legacy" in the Intervals tab and in the exports;
  their QTc is re-paired with the preceding RR, or left empty — and stated
  as such — when the frame has no R-peak times) until Intervals is re-run.
  The AV-delay check of Abnormal events uses this PR.
- **Wave template per recording.** The landmark search windows used for a
  recording are that recording's own template: the saved default template
  (or the built-in mouse defaults) until you change it in **Landmarks**
  (Beat Template tab) — auto-calibrated on the mean beat, refined from this
  recording's delineation, edited by hand or loaded from a template file.
  **Apply to this recording** writes nothing to disk; the template is kept
  in the recording's session. Only **Save as default template** (or
  **Save to file…**) writes a template, and the default only applies to
  recordings opened afterwards. Another channel of the same file is another
  recording (often another animal): detecting peaks on a different channel
  drops the template made on the previous one, and the new channel starts
  from the saved default. The Beat Template tab, the Intervals status
  line, the Summary report, the Excel Cover sheet and the Prism notes state
  which template was used ("Wave template" row: provenance and search
  windows). **Changed behaviour (this version):** earlier versions narrowed
  the template to each recording after every delineation and re-saved it
  as the global default, so a recording's intervals depended on the
  recordings analysed before it. A saved default written that way
  ("auto-updated") is now ignored in favour of the built-in defaults — save
  a template explicitly to make it the default. Intervals computed by
  earlier versions show "Wave template: not recorded".
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
- Beat-to-beat QT variability (IQR, SD and range of QT across the beats of
  one recording, single lead). This is **not** classical QT dispersion
  (max − min QT across the leads of one beat), which earlier versions
  reported under that name; the range is dominated by delineation outliers,
  so report the IQR.

### Abnormal events (rhythm classification)

- Bradycardia/tachycardia runs, pauses, premature beats and
  irregular-rhythm runs, classified against each recording's own baseline
  heart rate (not a fixed human bpm cutoff).
- **Premature beats (RR-based)** — a beat ending an RR interval shorter
  than 70 % of the reference RR: the median of the 10 intervals before
  the group of premature beats or, if smaller, of the 10 from the
  interval the rhythm returns on. The group must end — rhythm back above
  that bound — within 30 beats: a longer stretch of short intervals is a
  rate change (left to the tachycardia rule), and a group still under way
  at the end of the analysed window is not counted. Earlier versions used
  the median of the 10 neighbouring intervals, which a run of 6 or more
  beats pulled down itself, so **runs longer than about 5 beats were never
  reported** (they showed as irregular runs or tachycardia); they are now
  counted, which raises premature counts on recordings that have them. A
  limitation: a stretch of normal conduction between two 2:1-blocked
  stretches (RR halved, then doubled again) looks the same by timing and
  is reported as a run (check the bradycardia events around it). Detection
  uses R-R timing only, so it **cannot tell ventricular from
  supraventricular** premature beats (that needs QRS morphology and P
  waves); earlier versions called these "ESV" (ventricular extrasystoles)
  and only reported runs of ≥ 3. Isolated beats, couplets and runs ≥ 3 are
  now each reported, with counts and per-hour rates (per hour of the
  analysed span, first to last analysed beat) in the Events tab,
  its status line and ribbon, the "Copy for Excel" text, the Excel
  (`Premature_Beats`, `Abnormal_Events` sheets and Cover rows), the PDF
  reports and the Prism notes. Beats that artifact correction removed as
  ectopic are put back for this classification and **still counted**
  (flagged "removed by artifact correction"); before, auto-correction
  deleted them before classification, so they vanished from the counts.
  So are very premature beats (coupling interval below Min R-R, ≈ 67 ms)
  that correction removes as *duplicate* / *non-physiological*: such a
  removal is put back when the interval it leaves is long (> 1.2 × the
  local median RR) or followed by a compensatory pause, and its closest
  gap is ≥ 0.35 × that median. A removal that leaves a normal interval is
  a spurious double detection and is **not** counted; the summary gives
  how many were judged so. Sessions saved before the removal types were
  recorded have removals of unknown type: they are judged by the same
  interval test only, and the summary says how many (re-run artifact
  correction for the full check). The Excel `Premature_Beats` sheet now
  also gives the analysed window (start / end), and the exports **omit**
  the abnormal-event results (with a note on the Cover / PDF and in the
  status bar) when the peaks, the artifact correction or the analysis
  window changed after the classification — re-run it.
  The interval spanning such a removed beat is no longer reported as a
  pause, and an irregular-RR run overlapping a premature beat is not
  reported on top of it — both change event counts against older
  versions. Older "esv_run" event records are read as premature runs.
- **AV conduction-delay flagging** from sustained PR-interval prolongation,
  once Interval Delineation has been run — closes the gap where rhythm
  classification previously only ever saw R-peak timing.
- A color-coded event-type ribbon timeline (at-a-glance event clustering)
  alongside per-episode cards; click a card or ribbon segment to zoom the
  raw ECG to that episode, with a manual peak-edit mode for correcting
  detection right there.

### Comparison & export

- A/B segment comparison with superimposed RR tachograms and a full metric
  table (LF / HF power and n.u., SD1/SD2 — the same definitions as the main
  panels). Each segment is also summarised by the **means of
  non-overlapping blocks** (default 10 s, **Block (s)** in the dialog; a
  block is used when valid RR intervals cover ≥ 50 % of it): HR block mean
  ± SD with the number of blocks, and the B − A difference. A Mann-Whitney
  p on the block means is shown only as **descriptive** (≥ 3 blocks per
  segment): two stretches of one recording are not independent samples —
  the experimental unit is the animal — so group inference belongs to the
  Event-locked statistics / cohort tools, across animals. The Excel export
  adds a **Blocks** sheet and the note. **Changed behaviour (this
  version):** earlier versions ran a Mann-Whitney U test on the hundreds of
  beat-level RR intervals of each segment (autocorrelated values treated as
  independent — pseudoreplication), so p was "significant" for almost any
  two segments.
- Session persistence — save/restore a full analysis session as JSON, plus
  a local SQLite registry of recently analyzed recordings and of the
  event-locked autonomic results saved for cohort statistics.
- Export — formatted Excel workbooks, GraphPad Prism `.pzfx`, a one-page
  PDF report (signal strip, key-metric table, Poincaré diagram,
  abnormal-events summary), a per-episode annotated
  PDF for abnormal events, and CSV.
- **Provenance in every report.** The Summary text report (PROVENANCE
  block), the Excel Cover sheet, the Prism project notes and the PDF footer
  state the same rows: the **analysis window** and the **analysed
  duration** (window end − start, plus the first → last analysed R-peak
  span), the **filters** recorded when the peaks were detected, the RR
  bounds and the intervals excluded from HRV, the **artifact correction**
  (beats removed, by type), the **QTc formula actually used** (with the
  preceding RR), the interval definitions and the wave template. Every
  value carries its unit: VLF / LF / HF are powers in **ms²**, n.u. is used
  only for LFn / HFn. **Changed numbers (this version):** the "Duration"
  shown by the Summary verdict line and the duration tile was the absolute
  time of the last beat (it also counted the time before a windowed
  analysis start); it is now the analysed duration. The Excel Cover row
  "Duration" is renamed **Recording duration** (the signal length), and the
  HR_Statistics sheet's `Duration_s` column is replaced by
  `Recording_duration_s`, `Window_start_s`, `Window_end_s` and
  `Analysed_duration_s`. The Summary report and the PDF page, partly in
  French before, are now entirely in English.
  **Further provenance fixes (this version):** an analysis window whose
  end lies past the end of the signal is clamped to the signal end
  everywhere it is reported (the analysed duration and `Window_end_s` used
  to count time that does not exist; the row now says which end was
  entered). A **Signal crop** row states the load-time crop (Start / End
  crop) in file time: every time in the reports is relative to its start.
  The **Filters** row gives the band-pass cut-offs actually applied at the
  recording's fs: the filter clips each cut-off to [0.01, 0.99] × Nyquist,
  so at 2 kHz a 1 Hz low cut-off is really 10 Hz, and the row now says so
  (the filtering itself is unchanged). The artifact-correction counts are
  stated as whole-recording counts (it runs before windowing), and the
  "excluded from HRV" clause appears only for results that really exclude
  those intervals (older results bridged them). A saved session now keeps
  the analysis window the results were computed on and the filters / crop
  the signal was prepared with, and the restore re-prepares the signal
  from those rather than from widget values edited after detection
  (older sessions: as before).
- **RR asymmetry (changed numbers).** The Summary's **Porta index** and
  **Guzik index** now follow the published conventions, the same as
  NeuroKit's HRV_PI / HRV_GI on the Excel HRV_NonLinear sheet: Porta PI
  (Porta et al. 2008) = points below the Poincaré identity line
  (accelerations, ΔRR < 0) / points off the line, so ΔRR = 0 no longer
  counts in the denominator; Guzik GI (Guzik et al. 2006) = Σ distance to
  the line of the points above it (decelerations, ΔRR > 0) / Σ distance of
  all off-line points. The Summary used to show the acceleration share of
  Σ ΔRR² (the complement of a squared-distance GI) and a PI with zeros in
  the denominator. Both are 50 % for a symmetric series and NaN when no
  ΔRR differs from 0; the report, the Excel Cover sheet ("HR asymmetry")
  and the Prism Non-linear table (now with GI) state the convention. In
  that Prism table the row title **"Porta Index" is now "Porta Index PI
  (%)"** (same NeuroKit HRV_PI values) and a "Guzik Index GI (%)" row is
  added: a Prism layout that looks the row up by its old title must be
  updated. The Summary bar chart shows decelerations and accelerations as
  shares of the pairs with ΔRR ≠ 0 (the acceleration bar is the PI, the
  50 % line is symmetry) and gives the ΔRR = 0 pairs as a separate count;
  it used to divide by all pairs, so its bars did not match the PI.
- **Quality over time** is timed by the beats actually scored against the
  template: peaks too close to the signal edges are not scored, and the
  curve used to pair the scores with the peaks from the first one, shifting
  it by one beat when the first peak was excluded. The "Poor-shape-match
  time" tile and the RR CSV "quality" column both attribute each RR
  interval to the scored beat that **starts** it (the CSV row's
  Time_s − RR_ms, Time_s being the interval end), as the CSV did when no
  edge beat was dropped.
  Results from older versions, which did not record the scored beats, get
  them rebuilt on session restore from the restored peaks (same edge rule);
  if that is ambiguous, the curve, the tile ("—") and the CSV column are
  all left out, and the status bar says so.

### Interface

- Adaptive Workbench layout — toolbar, collapsible left/right panels, and a
  center workspace with a position-scrubber minimap, so the working area can
  be reclaimed on smaller screens.
- Light/dark theming, propagated live across the whole UI including every
  matplotlib plot.

## Autonomic indices (CSI / CPI) — Candia-Rivera et al. 2025

**Time course > Autonomic** computes the time-resolved cardiac sympathetic and
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
Time course > Autonomic figures are included in the figure and ZIP exports. The
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
│       ├── app_*.py, tab_*.py # ECGApp mixins: panels and one module per tab
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
