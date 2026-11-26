# Audit des bugs — ECG Tom V5 (13/11/2026)

Version auditée : branche `feature/ml-r-peak-detector`, commit `17e12f6`. Méthode : analyse statique (ruff, pyright), exécution complète de l'app sur `Baseline_STIM_BNST_206.mat`, puis revue de tout le code `ecg/` par 14 agents ; chaque bug candidat a été contre-vérifié par 2 agents indépendants (l'un cherche à le réfuter, l'autre à le reproduire). Aucun fichier du projet n'a été modifié.

Gravité : la plus haute retenue par un vérificateur qui confirme le bug. *CONFIRMÉ* = les 2 vérificateurs confirment ; *PLAUSIBLE* = 1 sur 2.

Les descriptions détaillées sont en anglais (texte des agents).


## Gravité Élevée (1)

### 1. Opening a file while a job runs leaves the old file's data under the new name

- **Où** : `ecg/ui/signal_controller.py:1059` · **Type** : data-loss · **Statut** : CONFIRMÉ
- **Scénario** : Click Detect Peaks on a long recording A (several seconds of work), then open recording B from Recent before it finishes. B's name shows in the file label and signal.filepath is B, but B is never loaded. When A's job lands, A's signal and peaks are displayed and analysed as B. Saving a session or exporting then writes A's beats under B's path or registry entry.
- **Correction proposée** : In load_path, refuse the load (or queue it) while _async_busy is set, before resetting anything. Also have on_preview_done/on_raw_load_done compare a generation captured at job start and drop stale results.


## Gravité Moyenne (20)

### 2. Permissive bounds keep the interval but blank its landmark positions

- **Où** : `ecg/core/analysis.py:3109` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : The user turns on the permissive switch and runs Intervals on a recording with a long PR, for example 90 ms (strict max 80, relaxed max 160). The row shows and exports PR_ms = 90, but P_onset_ms, P_peak_ms, P_onset_s and P_peak_s are NaN. The exported interval table has a PR with no P landmarks. 'Refine from delineation' (refine_template_from_intervals reads the *_ms landmark columns) leaves exactly those beats out. The same happens for a QT between 110 and 220 ms (T_offset blank while QT_ms is kept) and for a QRS between 35 and 70 ms.
- **Correction proposée** : Apply the same _m factor in the blanking block, for example by testing pr_clean, qrs_clean and qt_clean with isnan instead of re-testing pr_raw, qrs_raw and qt_raw against the strict limits.

### 3. resolve_r_vs_j doublet pass compares against already-discarded peaks and drops real beats

- **Où** : `ecg/core/detection.py:494` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Reproduced with beats every 120 ms (fs=1 kHz) and a steep noise spike 60 ms after the beat at 1060 (at 1120), the next beat at 1180 having a slightly slower upstroke. resolve_r_vs_j_peaks drops both 1120 and the true beat at 1180, although 1060 to 1180 is 120 ms. This leaves a missed beat (a 240 ms RR) in every detector and in apply_threshold, which all call this function.
- **Correction proposée** : Skip a pair when either index is already in `discard`, comparing instead with the last non-discarded peak. Simplest: walk the peaks with a 'last kept' pointer instead of iterating over a precomputed `active` list.

### 4. Composite QRS score rewards WIDER peaks (+1.5*z_width), contrary to its stated rationale

- **Où** : `ecg/core/detection.py:1424` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : On a synthetic 2 kHz mouse ECG with narrow R peaks (width50 ~7 samples) and T waves 45 ms later (width50 ~38 samples), z_width is about -0.67 for R and +0.71 for T. The width term gives T candidates a +2.1 score advantage over true R peaks, making wavelet detection more likely to accept T/J waves or EMG bursts and to drop narrow R peaks on real noisy recordings.
- **Correction proposée** : Use `-1.5 * z_width` (narrower is better) in both `provisional` (line 1424) and `score` (line 1465), or use a distance-to-median penalty if both too-narrow spikes and too-wide waves should be penalised.

### 5. Doublet pre-pass skips pairs after a flagged peak; triplets leave a <Min R-R gap and delete a real beat

- **Où** : `ecg/core/detection.py:2360` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Reproduced at fs=1 kHz with beats every 120 ms and two extra detections at 1085 and 1105 after the beat at 1060. detect_rr_artifacts flags 1085 as duplicate and the real beat at 1180 as ectopic. After apply_artifact_decisions the corrected peaks still hold a 45 ms RR (1060 to 1105), and a genuine beat has been removed. HRV and the event counts are both wrong.
- **Correction proposée** : Compare each peak with the last surviving (unflagged) peak, not with rpeaks[k]. Keep a 'last_kept' index, test rr between last_kept and k+1, and pick the loser between those two.

### 6. Band-pass low cut-off floor of 0.01 x Nyquist means the default 1 Hz is never applied

- **Où** : `ecg/core/filtering.py:29` · **Type** : wrong-result · **Statut** : PLAUSIBLE
- **Scénario** : Load any 2 kHz recording with the default filters (1-150 Hz). The signal is actually high-passed at 10 Hz, and the report says '1-150 Hz entered, clipped to ... 10-150 Hz'. On a 10 kHz recording it is high-passed at 50 Hz. T waves are attenuated and differentiated, so T end and QT move, and low P waves can fall under the P detection threshold.
- **Correction proposée** : Use a much lower floor (for example 1e-4 x Nyquist, or only require lo > 0), and design the filter as SOS with sosfiltfilt (butter(3, ..., output='sos')) so that low normalised cut-offs stay numerically stable. Keep the 0.99 ceiling.

### 7. Multi-column .mat matrices are flattened into one interleaved 'signal'

- **Où** : `ecg/io/loaders.py:118` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Reproduced: savemat {'data': np.column_stack([ecg, other])} with 20000 rows gives load_mat_signal(...) a signal of length 40000 that starts [ecg0, other0, ecg1, other1, ...]. Detection runs on a garbage, sawtooth-interleaved trace at the wrong effective duration, and no error is shown.
- **Correction proposée** : If value.ndim == 2 and min(shape) > 1, split it into per-column (or per-row for HDF5) channels named key[0], key[1], ... instead of flattening, or reject it with a clear message.

### 8. on_analysis_done has no file-generation check: old file's results land on a new file

- **Où** : `ecg/ui/analysis_controller.py:2074` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Click Analyze on recording A (long recording), then open recording B from File > Recent while it runs. When A's worker finishes, B's blank state receives A's results, A's peaks (rpeaks_ok) and A's artifact report. B's tabs show A's HRV, and Save Session writes A's results into B's session file.
- **Correction proposée** : Snapshot _generation in run_analysis, pass it to on_analysis_done, and return early (after _mark_landed and _disable_result_btns_without_results) when it changed.

### 9. Analyze overwrites a peak edit made while it ran, and stamps core as current

- **Où** : `ecg/ui/analysis_controller.py:2084` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Start Analyze, then exclude a peak with a left click in edit mode before it finishes. When the job lands, the excluded peak is back in rpeaks_ok and the HRV includes it. The edit only reappears on the next detection re-run, and nothing says the results ignored the edit.
- **Correction proposée** : Keep a reference to the rpeaks_ok array the worker started from. In on_analysis_done, if detection.rpeaks_ok is no longer that array, discard the bundle with an orange 'peaks changed while it ran: click Analyze' message, as _job_is_stale does.

### 10. Theme rebuild leaves Edit Peaks shown OFF while edit mode is still ON

- **Où** : `ecg/ui/app.py:251` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Turn on 'Edit Mode ON' (orange) in Signal & Detection, then use ⋯ > Light / dark mode. After the rebuild the button reads 'Edit Peaks' (off) and the click hint is gone, but left- and right-clicks on the ECG still exclude or add peaks. Clicking the button to 'turn it on' actually turns edit mode off.
- **Correction proposée** : In _rebuild_ui (or restore_ui_state), redraw the button and hint from detection.edit_mode and edit_free_placement instead of comparing with the live flag. For example, set edit_mode to False before restoring so the toggle runs, or call a sync helper.

### 11. Copy menu handler re-bound on every _set_textbox(tsv=...), so menus stack up

- **Où** : `ecg/ui/app_helpers.py:226` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : Run Analyze 3 times (or Analyze then Recompute Frequency twice), then right-click the 'Frequency domain' raw table. The Copy menu pops up 3 times in a row: on Windows tk_popup is modal, so each dismissal opens the next one. The count keeps growing for the whole session.
- **Correction proposée** : Bind the menu once per widget, for example by keeping a set of ids already bound or binding when the textbox is built. Keep the TSV lookup in _copy_tsv as it is.

### 12. Shell refresh calls _refresh_rhythm_stale(), which does not exist: Report and Compare segments never auto-refresh

- **Où** : `ecg/ui/context_strip.py:395` · **Type** : stale-state · **Statut** : CONFIRMÉ (test réel / analyse)
- **Scénario** : Every 'shell tick' raises AttributeError (caught, debug log) before _refresh_report() and _refresh_compare_tab(): with the Report tab open, finishing an analysis does not update it; Compare segments is not rebuilt for a new file/new peaks while open. Seen in the end-to-end run on Baseline_STIM_BNST_206.mat.
- **Correction proposée** : Remove the call (or implement the method).

### 13. Any peak edit or threshold nudge undoes Artifact Review removals

- **Où** : `ecg/ui/detection_controller.py:78` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Run Artifact Review and remove 12 ectopic or duplicate beats. Then left-click one peak in Edit mode to exclude it. All 12 removed beats are accepted peaks again, while the 'removed' artifact markers are still drawn at the same positions. The next Analyze uses them unless auto-correction is on.
- **Correction proposée** : In run_detection, after merging the manual sets, call analysis_ctrl.reapply_artifact_removals() (or drop_removed_peaks with the report's removed_samples, protecting manual_added). Alternatively, record review removals in manual_excluded.

### 14. 'Save as Default' saves the previous theme, not the one selected

- **Où** : `ecg/ui/dialogs.py:316` · **Type** : data-loss · **Statut** : CONFIRMÉ
- **Scénario** : The app is on Light at 1.00x. The user picks Dark, moves the slider to 1.20x and clicks 'Save as Default'. Dark 1.20x shows now, but after a restart the app opens in Light 1.00x again (the file holds the old values). Clicking a second time saves the right values, since THEME is by then the new theme.
- **Correction proposée** : Copy the working config into THEME first (THEME.apply_preset(self._working.preset_name); THEME.font_scale = self._working.font_scale), then THEME.save(), then run the destroy and rebuild part of _apply().

### 15. Calibration dialog loses the chosen configuration on every reload

- **Où** : `ecg/ui/dialogs.py:1983` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : The user opens Calibrate ks/kp and picks configuration B in the Configuration box. Then they change Label from 'All labels' to 'dobutamine' (or edit Project and press Load). The count shown for B changes, so the box jumps back to configuration A. The Z(k) curves, tests and Apply values are silently recomputed for A.
- **Correction proposée** : Before replacing self._configs, read the previous config tuple (prev = self._configs.get(self._cb_config.get())). Then match the new names on that tuple: pick = next((n for n, c in new.items() if c == prev), None).

### 16. Params dialog drops filter settings when main Filtering is OFF

- **Où** : `ecg/ui/params_dialog.py:179` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Filtering is OFF (the startup default). The user opens Parameters, turns Filtering on, sets Low cutoff 5 / High cutoff 150, turns Notch on, picks 'elgendi2010' and clicks Apply & Close. Filtering is now ON, but the band-pass is still the old values, Notch is still off and cleaning is still neurokit. The next Detect Peaks uses settings the user believes were changed.
- **Correction proposée** : Apply the switches first (sw_filtering, then call _on_filtering_toggle()) before writing the entries, combobox and notch switch. Or set each target widget to state='normal' around the write and restore its availability afterwards.

### 17. Session check/restore has no file-change guard: old session applied to newly opened file

- **Où** : `ecg/ui/session_controller.py:145` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Open a long recording A with a large .ecgsession, immediately open B from Recent -> A's 'Restore it?' prompt appears; answering Yes shows A's results, annotations and peak edits on B's signal, and a later Save Session writes them into B's session file.
- **Correction proposée** : Capture path and self.app._generation in try_restore_session; in _check_done and on_restore_session_done return early if signal.filepath != path or the generation changed; pass path explicitly to restore_session_worker instead of reading signal.filepath.

### 18. save_session_worker reads live app state off-thread; opening a file mid-save corrupts registry/ML cache

- **Où** : `ecg/ui/session_controller.py:997` · **Type** : data-loss · **Statut** : CONFIRMÉ
- **Scénario** : Verified-for-training on, click Save Session on file A (several seconds of feature extraction), press Ctrl+O and open file B -> A's registry row gets B's notes, hr_mean/sdnn/rmssd NULL, verified flag of B, A's .npz training sample is deleted, and B's sidebar says '✓ Session saved'.
- **Correction proposée** : Snapshot results-derived stats, rpeaks_ok, fs, signal duration, notes and verified flag on the main thread in save_session() and pass them to the worker; in on_save_session_done compare a captured _generation/filepath and skip UI updates when the file changed.

### 19. UI rebuild loses sw_verified_training / sw_permissive / sw_filter_preview state

- **Où** : `ecg/ui/session_controller.py:1144` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Mark a recording verified, toggle the theme -> switch reads off; Save Session still writes it to the ML training cache. Or enable Permissive bounds, toggle theme, Measure intervals -> intervals measured with strict bounds.
- **Correction proposée** : Add sw_permissive and sw_filter_preview to _SNAPSHOT_SWITCHES and call self.sync_verified_switch() at the end of restore_ui_state (or sync sw_filter_preview from ui.filter_preview_on).

### 20. 'SG window (ms)' setting is never passed to the detector

- **Où** : `ecg/ui/signal_controller.py:199` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : With SG + Derivative selected, set SG window to 6 ms (below the detector's 10 ms cap, so it would take effect) and click Detect Peaks. The detector still runs with 20 ms, capped to 10 ms, so the peaks are identical whatever value is typed.
- **Correction proposée** : Add sg_window_ms to FilterParams (from_widgets/from_dict/to_dict), or put it into the params dict in snapshot_params from ent_sg_window_ms.

### 21. Re-detect clears manual edits but keeps the undo/redo stacks

- **Où** : `ecg/ui/signal_controller.py:717` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Exclude a few peaks, change the SG/threshold/filter settings and click Detect Peaks (the status says previous edits were cleared). Pressing Ctrl+Z then restores the old exclusion and addition sets on the new detection. Old manual additions come back as cyan peaks, possibly next to newly detected ones, and the results are dropped again.
- **Correction proposée** : Clear detection.edit_undo/edit_redo in on_preview_done (and on_raw_load_done), then call detection_ctrl.update_undo_btns().


## Gravité Faible (53)

### 22. Pass 1 'nonphysio' branch can never fire; n_nonphysio is always 0

- **Où** : `ecg/core/detection.py:2382` · **Type** : other · **Statut** : CONFIRMÉ
- **Scénario** : Run Review Artifacts or auto-correction on any recording: every sub-Min-R-R removal is reported as 'duplicate' and n_nonphysio is always 0, so the 'non-physio' figure shown and saved is meaningless.
- **Correction proposée** : Either give Pass 1 its own distinct criterion (for example the user's HRV rr_min bound when it differs from the doublet threshold), or drop the category and its counters so the report does not show a figure that is always 0.

### 23. Ectopic local reference includes intervals that end on flagged (duplicate) beats

- **Où** : `ecg/core/detection.py:2422` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : In a noisy segment with several duplicate detections (each flagged in the pre-pass), each one adds a ~10-60 ms interval to the median RR of the nearby beats. refs[i] is biased low, short_lim = ref*(1-0.2) drops, and genuine premature beats in that segment fall under the threshold and are not flagged by Review Artifacts or auto-correction.
- **Correction proposée** : Change the condition to `if j < n - 1 and not flagged_mask[j + 1]:`.

### 24. Merged brady/tachy/irregular events keep the first run's label, hr_mean and rr_mean

- **Où** : `ecg/core/detection.py:3491` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Two bradycardia runs of 12 and 15 intervals separated by a 3-beat recovery are merged into one event with n_beats=27. Its label still reads '... 12 beats' with the first run's bpm, and hr_mean/rr_mean (shown and exported) describe only the first run. The same happens with consecutive irregular_run windows (label '7 beats', n_beats 14) and merged pauses.
- **Correction proposée** : On merge, recompute hr_mean/rr_mean from rr_ms over the merged span, use a weighted delta, and rebuild the label from the merged values (or store the label template and format it after merging).

### 25. Excel Metrics sheet labels event-kind rows with the raw kind key

- **Où** : `ecg/core/metric_table.py:490` · **Type** : export · **Statut** : CONFIRMÉ
- **Scénario** : Classify abnormal events, then export to Excel. The Metrics sheet lists rows such as 'premature_couplet' and 'av_delay', while the Report tab and the Events sheet show 'Premature couplet (RR-based)' and 'AV delay'.
- **Correction proposée** : Use event_kind_label(k) from ecg.core.models as the MetricRow label, and drop the event_kind_label call in report_view (or make it idempotent).

### 26. Training cache is keyed on a path+size+mtime fingerprint; touching or moving a file duplicates and orphans samples

- **Où** : `ecg/core/ml_detector.py:402` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Mark rec.mat verified and save, move it to another folder, re-open it and save again (still verified). Two training files now exist for one recording. The hold-out set can contain an exact duplicate of training data, so the 'accuracy x -> y' shown after retraining overstates generalisation. Un-marking the file leaves the first sample in place.
- **Correction proposée** : Key training samples on a content-based or registry-stable id (e.g. a hash of the first few MB plus the size). At minimum, delete samples whose registry row maps to the same recording before writing a new one.

### 27. train_model() can raise despite its 'never raises' contract; Train button fails silently

- **Où** : `ecg/core/ml_detector.py:585` · **Type** : crash · **Statut** : CONFIRMÉ
- **Scénario** : skops is not installed (it is a separate requirement) and the user clicks Train Model. Both forest fits complete, then save() raises ModuleNotFoundError. The button resets but no success or error message appears, and no model is written. The user does not know why training 'did nothing'.
- **Correction proposée** : Wrap the fit and save section in try/except Exception and return {'ok': False, 'message': f'Training failed: {exc}'}. Check for skops next to sklearn at line 527.

### 28. Saving or deleting a template after a failed read wipes every other user template

- **Où** : `ecg/core/segment_plan.py:164` · **Type** : data-loss · **Statut** : CONFIRMÉ
- **Scénario** : The templates file is truncated or corrupted (the app is killed during a save, or a sync tool or editor leaves partial JSON). The user opens Compare Segments, sees only the built-ins and saves a new template 'Footshock 5 min'. The file now holds only that template, and every template saved earlier is lost for good.
- **Correction proposée** : In save_template/delete_template, read the file with a strict loader that raises on a file that exists but cannot be read or parsed, instead of falling back to built-ins only. Refuse to overwrite it, or back it up first, and write through a temp file plus os.replace so the write is atomic.

### 29. T peak from a phase plan or a first T phase is clamped onto the T window edge

- **Où** : `ecg/core/wave_template.py:2092` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Rolling median on (ensemble_beats > 1), during a heart-rate or QT shortening where the row's T sits about 10 ms earlier than the plan median's T, with the plan's first phase near the start of the template T window. _t_from_plan returns, for example, t_pk = 20 ms while t_lo = 26 ms, and the exported T_peak becomes exactly 26 ms on those rows while T_offset follows the true lag. The T-peak-based measures and the next beat's P search start are wrong.
- **Correction proposée** : Do not clamp T_peak onto the template window when it comes from _t_from_plan or _t_phases. Keep it if it lies inside the beat (bt0..bt1) and after the QRS offset, otherwise return NaN, as is already done for the interval end points.

### 30. Integer columns written as floats with 4 decimals in every all-numeric Excel sheet

- **Où** : `ecg/io/export.py:406` · **Type** : export · **Statut** : CONFIRMÉ
- **Scénario** : Export to Excel, then open HR_Statistics: N_beats shows '1234.0000' and Fs_Hz shows '2000.0000'. In ECG_Signal the Is_RPeak column shows 1.0000/0.0000 instead of 1/0. Re-reading the file with pandas gives float columns for the counts.
- **Correction proposée** : Iterate with df.itertuples(index=False), or read the values column by column (df[col].iat[i]), so each column keeps its own dtype. Alternatively, format floats where value.is_integer() and the source column dtype is an integer as int.

### 31. ECG_Signal downsampling uses floor, so up to ~100k rows are exported instead of at most 50k

- **Où** : `ecg/io/export.py:574` · **Type** : performance · **Statut** : CONFIRMÉ
- **Scénario** : Export a 90,000-sample recording: ECG_Signal gets 90,000 fully styled rows (not ≤ 50k) and the export is noticeably slower and larger than intended.
- **Correction proposée** : step = _decimation_step(n, MAX_ROWS), i.e. max(1, ceil(n / MAX_ROWS)).

### 32. Prism export ignores arrhythmia_events: no Abnormal events table is written

- **Où** : `ecg/io/export.py:1060` · **Type** : export · **Statut** : PLAUSIBLE
- **Scénario** : Classify abnormal events, then Export to Prism: the Excel export has an Abnormal_Events sheet, but the .pzfx has no events table, and nothing says the events were left out.
- **Correction proposée** : When arrhythmia_events is non-empty, add abnormal_events_frame(arrhythmia_events) restricted to its numeric columns (Start_s, Duration_s, N_beats, HR_bpm, Delta_pct...) as a Column table with row titles 'Label @ Start_s'. Otherwise drop the parameter and fix the comment.

### 33. Prism export silently drops the recording notes it is given

- **Où** : `ecg/io/export.py:1063` · **Type** : export · **Statut** : CONFIRMÉ
- **Scénario** : Type experiment notes for a recording (Notes dialog), then Export to Prism: the .pzfx 'Project Info' has no notes entry, so the notes are lost from the export.
- **Correction proposée** : When recording_notes.strip() is non-empty, append ('Recording notes', recording_notes) to note_names / note_values.

### 34. A recording with an empty subject or file is appended again on every 'Add to group table'

- **Où** : `ecg/io/group_table.py:72` · **Type** : data-loss · **Statut** : CONFIRMÉ
- **Scénario** : Leave Subject ID empty, run Compare segments and click 'Add to group table' twice on the same recording: the group table now holds two identical rows for that file.
- **Correction proposée** : Normalise both sides before comparing, e.g. df['subject'].map(subject_text) == subject_text(row['subject']) (and the same for file), so NaN and '' compare equal.

### 35. Sampling-rate detection reads the struct's 'start' time as the sample interval

- **Où** : `ecg/io/loaders.py:137` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Reproduced: savemat {'Ch1': {'values': ecg, 'start': 0.25}} gives detected_fs = 4 Hz. With {'values': ecg, 'start': 0.25, 'fs': 2000}, it also returns 4 instead of 2000. The loader uses detected_fs instead of the entered rate, so every time axis, RR interval and HRV value is wrong by orders of magnitude.
- **Correction proposée** : Remove 'start' from the interval attribute list (it is not a sample interval). Keep only 'interval' and 'sample_interval'.

### 36. Unsigned-integer channels are never auto-selected (dtype kind 'u' rejected)

- **Où** : `ecg/io/loaders.py:280` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Reproduced: savemat {'raw': uint16 array of 20000 samples} with the default channel 'ECG' raises ValueError('No ECG channel found. Available keys: [\'raw\']'), so the file cannot be opened without guessing the variable name.
- **Correction proposée** : Accept dtype kinds 'fiu' in channel_score (the selected array is cast to float64 afterwards anyway).

### 37. Events RR panel y-limits include the baseline axvline's [0,1] ydata

- **Où** : `ecg/ui/analysis_controller.py:966` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : Classify a recording with a bradycardia or tachycardia event, or one whose baseline is unreliable (short window). The RR series axis spans about -10 ms to about 220 ms for a ~100 ms RR, so beat-to-beat RR variation is hard to see.
- **Correction proposée** : Compute the limits from rr_ms plus the axhline values only, e.g. np.concatenate([rr_ms, [bl_rr, brady_rr, tachy_rr]]) when those lines exist. Alternatively, call centered_ylim before drawing the axvline.

### 38. Failed Sliding/Epochs recompute keeps the previous frame, shown as up to date

- **Où** : `ecg/ui/analysis_controller.py:1506` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Compute Sliding with window 30 s, then recompute with a window longer than the analysed span. The label reads 'No valid windows', but the 30 s plot and table stay, and Report/Excel export the 30 s frame as current Sliding results.
- **Correction proposée** : When rows is empty, clear rolling_hrv_df / epoch_df and the windowed_view entry and blank the view (clear_rolling_view / clear_epochs_view). Alternatively, mark the old frame as out of date.

### 39. run_intervals sets 'Delineating all beats…' before a busy-guard refusal

- **Où** : `ecg/ui/analysis_controller.py:2442` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : While frequency or non-linear HRV is running (for example during the Analyze chain), click Measure intervals. The status bar says 'Another operation is already running', but the Intervals label keeps reading 'Delineating all beats…' even though no delineation runs.
- **Correction proposée** : Add 'if self._refused_busy(): return' before configuring lbl_ivl_status, as in run_freq.

### 40. run_intervals never marks 'intervals' running: landmark-edit lock and Computing state are dead

- **Où** : `ecg/ui/analysis_controller.py:2504` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Click Measure intervals on a long recording, then drag a landmark in the Morphology editor while it runs. The drag is accepted (the guard never engages), the template changes under the running job, and the Intervals tab never shows its computing state.
- **Correction proposée** : Call self._mark_running('intervals') before _start_async_result, and self._mark_landed('intervals') at the top of _done.

### 41. Theme rebuild drops the sampling-rate source note in the Recording sheet

- **Où** : `ecg/ui/app.py:253` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Load a .mat file whose fs is read from the file: the Recording sheet shows '✓ Auto-detected from file: 2000 Hz'. Toggle Light/Dark: the note is gone, and nothing says whether the rate shown came from the file or was typed.
- **Correction proposée** : Keep the fs source (text and colour) in SignalState and reapply it in _rebuild_ui.

### 42. Undo/Redo buttons show disabled after a theme rebuild despite history

- **Où** : `ecg/ui/app.py:274` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Make 5 peak edits (Undo reads '↩ Undo (5)'), then toggle Light/Dark or apply a theme. Undo is now greyed out and reads '↩ Undo' while Ctrl+Z still undoes the edits. The button stays wrong until the next edit.
- **Correction proposée** : Call self._update_undo_btns() in _rebuild_ui after _build().

### 43. Artifact review replaces rpeaks_ok without refreshing the live RR tachogram

- **Où** : `ecg/ui/app_delegates.py:571` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : Detect peaks, Analyze, open Review artifacts and remove a few beats -> HRV tab RR/HR tachogram and histogram go blank ('Run analysis') and the quality score still reflects the pre-review peaks, unlike after a manual peak edit.
- **Correction proposée** : After assigning the corrected peaks call self._update_signal_quality(corrected) and self.plot_ctrl.refresh_live_rr().

### 44. Status text width measured in points, so messages are cut ~30% early

- **Où** : `ecg/ui/app_toolbar.py:180` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : Any status message using 75–100% of the free tab-row width, such as 'Copied as TSV — paste into Excel ✓' plus a file name, is shown truncated with an ellipsis although it fits. Hovering a tab opens the floating hint needlessly.
- **Correction proposée** : Measure with the label's actual font, for example tkfont.Font(font=(family, -round(11*font_scale*widget_scaling))) or self.lbl_status._label.cget('font').

### 45. Glyph '▶' missing from Segoe UI in a Compare segments figure

- **Où** : `ecg/ui/compare_segments.py` · **Type** : ui-glitch · **Statut** : CONFIRMÉ (test réel / analyse)
- **Scénario** : matplotlib warns and draws an empty box instead of the triangle.
- **Correction proposée** : Use a glyph present in the font or DejaVu Sans for that text.

### 46. Detection view position silently used as an event, adding false artefact flags

- **Où** : `ecg/ui/compare_segments.py:241` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Open a file, detect peaks, scroll the Detection view to t = 1200 s, open Compare segments and click Compare without touching the event fields. Segment A or B shows '⚠ overlaps the event…'. After 'Add to group table…' that animal is listed under 'Rows to check' and left out of the group Wilcoxon tests.
- **Correction proposée** : Don't pre-fill the Start entry, or show the nav position only as placeholder text. Alternatively, pass ev only when the user picked or typed an event (track this with a flag set by _on_event_pick or a key binding).

### 47. Removing a row raises ValueError when any other row has a non-numeric time

- **Où** : `ecg/ui/compare_segments.py:404` · **Type** : crash · **Statut** : CONFIRMÉ
- **Scénario** : Type '12,5' in segment A's 'From (s)' field, then click ✕ on segment B. Nothing happens (a ValueError is raised in the callback) and B is not removed.
- **Correction proposée** : Wrap the conversion in try/except ValueError and keep the raw text (pass the strings through to _add_row), or show an error in lbl_status.

### 48. With one segment the block rows (Blocks used, HR block mean/SD) are always blank

- **Où** : `ecg/ui/compare_segments.py:719` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Remove segment B (✕) so that only Segment A is left and click Compare. The 'Blocks used (n)' and 'HR, mean of block means' rows show '—', and the Excel export has empty cells for them, although the segment is long enough for many blocks.
- **Correction proposée** : When len(segs) == 1, fill the block fields from segs[0]['blocks'] (for example compare_segments(segs[0], dict(segs[0])), or a small helper that writes n_blocks and the hr_mean mean/SD of the blocks).

### 49. Saved status line from the previous file comes back after loading a new one

- **Où** : `ecg/ui/context_strip.py:436` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Compute Epochs ('12 epochs × 60s'), then edit a peak: epochs go out of date and the label is emptied and saved. Open another recording: the Epochs mode now shows '12 epochs × 60s' for a file with no epochs. The same thing can happen to any line in _STATUS_LINES that a reset leaves empty.
- **Correction proposée** : Clear self._status_lines_saved on a new file (for example from freshness.reset or reset_for_new_file). Also drop a saved entry when its module becomes NOT_COMPUTED, and restore only when the module is UP_TO_DATE again.

### 50. configure(state=...) on widgets that do not take a 'state' option

- **Où** : `ecg/ui/detection_controller.py:241` · **Type** : ui-glitch · **Statut** : CONFIRMÉ (test réel / analyse)
- **Scénario** : Toggling Filtering logs two errors at startup; harmless (caught) but noisy.
- **Correction proposée** : Only configure state on widgets that support it.

### 51. Check Agreement skips the method chosen in the combobox, not the one that found the peaks

- **Où** : `ecg/ui/detection_controller.py:744` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Detect with SG + Derivative, switch the combobox to Wavelet (CWT) without re-detecting, then click Check Agreement. SG+Derivative is compared against its own peaks and agrees trivially, so the '% beats confirmed' figure is inflated. Wavelet is reported as skipped.
- **Correction proposée** : Take the active method from self.app.signal.prepare_params.get('detection_method'), and fall back to the combobox only when it is None.

### 52. Artifact review 'median ± 20%' band does not show the ±20% tolerance

- **Où** : `ecg/ui/dialogs.py:662` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : An ectopic candidate has a neighbour at 0.7 x ref (outside the ±20% tolerance). The neighbour still sits inside the blue shading, so the user reading the plot sees it as within the reference band.
- **Correction proposée** : Shade two spans, [samp/fs - 1.2*ref_s, samp/fs - 0.8*ref_s] and [samp/fs + 0.8*ref_s, samp/fs + 1.2*ref_s]. These are the expected positions of the previous and next beats within ±20%.

### 53. Kept result keeps stale prev_gap_s / next_gap_s after a far annotation edit

- **Où** : `ecg/ui/event_locked_controller.py:139` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Label filter 'dose', a dose event at 100 s, and an annotation 'vehicle' at 500 s give next_gap_s = 400. The user adds 'vehicle' at 300 s, outside the 30 s window, so the result is kept. Copy for Excel / the Excel export still show next_gap_s = 400 instead of 200, and n_other_onsets is still 1 instead of 2.
- **Correction proposée** : When the result is kept, recompute the isolation columns (core._onset_gaps on el.table onsets and the new others) and update params['n_other_onsets'] and the table/event_locked_df. Or simply invalidate whenever the set of other onsets changes.

### 54. 'Peaks changed — compute the Autonomic mode' notice survives the recompute

- **Où** : `ecg/ui/event_locked_controller.py:171` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Compute Autonomic, then event-locked, then edit a peak: the label says 'Peaks changed — compute the Autonomic mode, then event-locked, again'. Press Autonomic Compute: the event-locked label and banner reason still say to compute the Autonomic mode instead of 'Autonomic indices recomputed — click Compute event-locked again'.
- **Correction proposée** : In invalidate(), when reason is 'autonomic' and a stale note is already stored, replace it with _STALE_TEXT['autonomic'] (and update the label) even when no result was shown.

### 55. Edge event shown as 'no · 83 % coverage' although coverage passes 80 %

- **Où** : `ecg/ui/event_locked_view.py:622` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : Reproduced: a 200 s Autonomic series and an annotation at 175 s with pre/post 30 s. The post window runs 5 s past the end, so coverage_post = 25/30 = 0.83 and valid = False. event_rows gives valid='no · 83 % coverage', which reads as if 83 % fails an 80 % rule.
- **Correction proposée** : When the event is isolated and min(coverage) >= params['min_coverage'], show 'no · window outside the series' (or check onset-pre_s < t0 / onset+post_s > t_end). Otherwise show the coverage.

### 56. Empty Subject gives '_ECG_figures_…' folder and '_…' default file names

- **Où** : `ecg/ui/export_controller.py:427` · **Type** : export · **Statut** : CONFIRMÉ
- **Scénario** : Leave Subject empty and export figures -> folder '_ECG_figures_<ts>' and figure suptitles blank; Excel default name '_ECG_<ts>.xlsx'.
- **Correction proposée** : Use one helper `(app.ent_subject.get().strip() if app.ent_subject else '') or 'subject'` for every export name.

### 57. PDF header prints the QTc formula although QT/QTc are not reported

- **Où** : `ecg/ui/export_controller.py:805` · **Type** : export · **Statut** : CONFIRMÉ
- **Scénario** : Export PDF report -> header reads 'QTc: Mitchell…, preceding RR' while the footer says QT is not reported and no QT/QTc row exists in Key metrics.
- **Correction proposée** : Build head_rest with the QTc part only when QT_REPORTED is True.

### 58. Save report .txt has no error handling: a write failure is silent

- **Où** : `ecg/ui/export_controller.py:1100` · **Type** : export · **Statut** : CONFIRMÉ
- **Scénario** : Save the text report over a file that is open in another program / into a read-only folder -> nothing happens, no error shown, status bar unchanged.
- **Correction proposée** : Wrap the open/write in try/except OSError and show messagebox.showerror('Export failed', str(exc)).

### 59. Spike stepper walks the previous recording's spike times after a new file

- **Où** : `ecg/ui/navigation_controller.py:140` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Analyse recording A (it has flagged RR spikes), open recording B (raw-only load, no detection yet) and press the Detection tab's next-spike button. Instead of 'No suspicious beats… run Analyze first', the view jumps to times of A's spikes in B's signal, with the status 'Navigation → spike at …'.
- **Correction proposée** : Set app.ui.rr_spike_times = None in reset_for_new_file and on_raw_load_done, and when refresh_live_rr bails out for too few peaks.

### 60. Overview cursor jumps back to a stale position on resize or tab re-show

- **Où** : `ecg/ui/plot_controller.py:456` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Load a file and step through the recording with the arrow keys (each step takes the fast path), say from 0 s to 300 s. Then resize the window or drag the pane splitter: the orange highlight on the overview strip jumps back to 0-10 s while the detail plot still shows 300-310 s.
- **Correction proposée** : Have the draw closure read the live position (self.app.ui.nav_pos and ent_window) when it runs, instead of the captured t_start/t_end. Or, in the fast path, also replace slot._draw_fn with a closure that has the new bounds.

### 61. Intervals panel says 'not available (needs clear P/Q/S/T)' when Intervals was never run

- **Où** : `ecg/ui/plot_controller.py:1984` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : Click Analyze, do not run Intervals, then toggle the theme (or save and restore the session). The Morphology & Intervals panel now claims that interval delineation is unavailable because of the signal quality, when it simply has not been run.
- **Correction proposée** : Treat a frame that has none of the PR/QRS/QT/QTc columns as 'not computed yet'. For example, test `not any(c in ivl.columns for c in ('PR_ms','QRS_ms','QT_ms','QTc_ms'))` together with ivl.empty before the 'not available' branch.

### 62. Beat template uses the ghost-row count (max 60) as the beat count

- **Où** : `ecg/ui/plot_controller.py:2181` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Analyze a 20-minute recording with 8000 scored beats where only 7 beats (0.09%) fall below r = 0.90. The title 'Beat Morphology (7 beats < 0.90)' turns orange as a quality warning, which should need more than 800 bad beats. Both histograms also get only 15 bins whatever the beat count.
- **Correction proposée** : Use n_extracted = len(beat_corr) for the bin counts and for the 10% threshold (n_bad > 0.1 * n_extracted). Keep len(beat_matrix) only for the ghost-trace stride.

### 63. Y-scale bar: FocusOut pins a stale, rounded auto range as a permanent manual ylim

- **Où** : `ecg/ui/plots.py:301` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : In Compare Segments, run Compare, wheel-zoom the RR plot, then click into the 'Y min' field and click elsewhere. The zoom snaps back to the old, rounded range. Running Compare again on other segments, or switching to Poincaré, keeps that fixed Y range on the first axes only. Data can fall off-scale, and the first Poincaré panel loses its equal scale with the others.
- **Correction proposée** : Commit only when the field text actually changed from the last read-back (or bind Return/KP_Enter only). Also connect the draw_event readback whether or not there is a toolbar, so the fields follow wheel zooms.

### 64. Restore status says 'results ready' for a session saved without results

- **Où** : `ecg/ui/session_controller.py:595` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : Detect peaks, Save Session, reopen the file and restore -> status bar claims 'results ready' but every tab is empty and clicking Frequency HRV gives 'Click Analyze first'.
- **Correction proposée** : Only append 'results ready' (and enable the per-module buttons) when self.app.analysis.results is not None; otherwise say 'peaks restored — click Analyze'.

### 65. Abnormal-event classification is never saved or restored with the session

- **Où** : `ecg/ui/session_controller.py:698` · **Type** : data-loss · **Statut** : PLAUSIBLE
- **Scénario** : Run the Events classification, Save Session, close and reopen the file with Restore -> the Events tab is empty and the PDF report says 'None classified in this session'.
- **Correction proposée** : Serialise arrhythmia_events (ArrhythmiaEvent.to_dict), premature_summary and the basis in collect_session_state and restore them (ArrhythmiaEvent.from_dict) in on_restore_session_done, or state explicitly that events must be re-run.

### 66. on_save_for_training_done marks whatever file is open now as verified

- **Où** : `ecg/ui/session_controller.py:934` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Click 'Save for training' on file A, open file B before it finishes -> B's 'Verified for training' switch turns on; Save Session on B adds B (unreviewed peaks) to the training set.
- **Correction proposée** : Return the snapshotted filepath in the result and only update session.verified_for_training / the switch when it equals self.app.signal.filepath.

### 67. SG+Derivative branch discards fix_polarity's inversion flag

- **Où** : `ecg/ui/signal_controller.py:225` · **Type** : wrong-result · **Statut** : CONFIRMÉ
- **Scénario** : Load a recording with negative R waves and detect with the default method (SG + Derivative (10 kHz)). fix_polarity flips the signal, but the detail plot never shows 'auto-inverted' and the saved session records signal_inverted=False. Switching to Envelope Max on the same file does show the tag.
- **Correction proposée** : Return "inverted": inverted in the SG branch, like the other branches.

### 68. Crop outside the recording passes the 'flat signal' check as an empty signal

- **Où** : `ecg/ui/signal_controller.py:411` · **Type** : crash · **Statut** : CONFIRMÉ
- **Scénario** : Set Crop start = 2000 s on a 600 s file (or Crop end lower than Crop start) and open the file. It 'loads' with 0 s and no error. Pressing Apply on the analysis window, or turning on the filter preview, raises IndexError in a Tk callback, and Detect Peaks fails with an unrelated error.
- **Correction proposée** : After cropping, raise ValueError('Crop window is outside the recording / empty') if len(sig) < some minimum (e.g. 1 s of samples), before the std check.

### 69. Re-detecting keeps the previous Check Agreement markers and label

- **Où** : `ecg/ui/signal_controller.py:757` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Run Check Agreement, then change the method, filter or threshold and click Detect Peaks again. Amber 'Unconfirmed' circles from the old run stay on the trace, at samples that may no longer be peaks. The status label still shows the old '% beats confirmed' summary.
- **Correction proposée** : In on_preview_done, reset agreement_disagree_samples/summary/pct/peaks and the lbl_agreement_status text, as on_raw_load_done does.

### 70. Window-beyond-signal warning is overwritten at once

- **Où** : `ecg/ui/signal_controller.py:933` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : On a 600 s recording, enter an analysis window of 0 → 900 and press Apply. The status bar shows only 'Analysis window updated', the label says '(900.0 s)', and the 'End bound exceeds the signal duration' warning is never visible.
- **Correction proposée** : Return after the warning, or clamp t1 to the signal end. At minimum, skip the final BLUE status when the warning was issued.

### 71. New file resets edit flags but not the Edit/Undo toolbar buttons

- **Où** : `ecg/ui/signal_controller.py:1140` · **Type** : ui-glitch · **Statut** : CONFIRMÉ
- **Scénario** : With Edit Mode ON and a few undoable edits, open another file. The button still reads 'Edit Mode ON' with the hint shown, but clicks no longer edit. The first press of the button only turns edit mode back on. Undo still shows 'Undo (3)' enabled, and clicking it says 'Nothing to undo.' Free Placement also stays active for the next edit session.
- **Correction proposée** : After the reset, restyle btn_edit_mode to 'Edit Peaks', pack_forget lbl_edit_hint, reset edit_free_placement and its button, and call detection_ctrl.update_undo_btns().

### 72. Quality curve hidden ('taller window needed') at 440 px panel height with 1.15 font scale

- **Où** : `ecg/ui/tab_detection.py:556` · **Type** : ui-glitch · **Statut** : CONFIRMÉ (test réel / analyse)
- **Scénario** : On a 1366x768 screen with Windows scaling, the template-correlation curve of the Signal tab is replaced by a notice; 2 tests fail for this reason.
- **Correction proposée** : Lower the row height threshold or account for font_scale.

### 73. 'Review N artifacts' count is cached by peaks only and goes stale after a Min/Max RR edit

- **Où** : `ecg/ui/tab_detection.py:596` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : After detection the button reads 'Review 3 artifacts'. Change Max RR from 333 to 200 ms. The button still says 3, but clicking it lists many more candidates (every interval over 200 ms).
- **Correction proposée** : Include the RR bounds (and id(self._signal_flt)) in the cache key, e.g. key = (id(rp), len(rp), rr_min, rr_max).

### 74. Report Provenance window and RR bounds stay stale when no Analyze has run

- **Où** : `ecg/ui/tab_report.py:236` · **Type** : stale-state · **Statut** : CONFIRMÉ
- **Scénario** : Load a file and Detect Peaks, then open Report: it shows 'RR bounds 66.7–333.3 ms' and the full window. Go back, set Max RR to 500 and Apply a 0–300 s window, then reopen Report. The signature is unchanged, so Provenance still shows the old bounds and window.
- **Correction proposée** : Add (a.t_start, a.t_end) and analysis_ctrl.hrv_rr_bounds(quiet=True) to report_signature, or every input of provenance_rows.
