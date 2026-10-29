# -*- coding: utf-8 -*-
"""
event_locked_controller.py
--------------------------
EventLockedController -- HRV > Autonomic > Event-locked: the per-event
pre / post statistics of the Autonomic result around the time annotations
(core.event_locked, Candia-Rivera et al., J Physiol 2025), their trace
plot and table, the cohort registry (ecg.io.db event_locked table: save,
CSV export) and the ks / kp calibration dialog.

Logic lives in ecg.ui.event_locked_view (pure, tested); this controller
only reads the widgets, stores the result in app state and writes the
widgets back. Everything runs on the main thread: a recording's events are
a few slices of the 20 Hz series.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from tkinter import filedialog

from ecg.io import db
from ecg.ui.autonomic_view import _to_float, format_param, method_key
from ecg.ui.event_locked_view import (
    ALL_LABELS, annotation_label_options, compute_event_locked,
    draw_event_locked, event_status, event_table_frame, event_table_text,
    events_signature, export_rows, normalise_event_params, other_annotation_onsets,
    parse_event_inputs, registry_params, registry_rows, select_events,
    subject_warning,
)
# Colours are read from the theme module at call time (``_theme.GREEN``),
# so a light / dark switch needs no re-sync of this module.
from ecg.ui import theme as _theme

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class EventLockedController:
    IDLE_TEXT = ("  Compute the Autonomic indices (⟳ Compute above), add annotations "
                 "(onset = their start), then ⟳ Compute event-locked")
    # invalidate() reasons -> label text (None: the idle hint)
    _STALE_TEXT = {
        "peaks":       "  Peaks changed — compute the Autonomic indices, then event-locked, again",
        "autonomic":   "  Autonomic indices recomputed — click ⟳ Compute event-locked again",
        "annotations": "  Annotations changed — click ⟳ Compute event-locked again",
    }

    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    # ── parameters (app state, survive a UI rebuild, saved with the session) ──

    def widget_params(self) -> dict:
        """The panel's parameters as currently entered, read tolerantly (a
        missing widget or an unreadable value keeps the stored one). For a
        session save and a UI rebuild. Main thread only."""
        raw: "dict[str, Any]" = {}
        for key, attr in (("label", "cb_evt_label"), ("pre_s", "ent_evt_pre"),
                          ("post_s", "ent_evt_post"), ("zscore", "chk_evt_z")):
            w = getattr(self.app, attr, None)
            if w is None:
                continue
            try:
                raw[key] = w.get()
            except Exception as exc:
                log.debug("event-locked widget_params: %s unreadable: %s", attr, exc)
        return normalise_event_params(raw, base=self.app.analysis.event_locked_params)

    def apply_params(self, params: Any) -> dict:
        """Make *params* the panel's parameters (normalised; a missing or
        invalid key keeps the current value) and show them. Main thread."""
        p = normalise_event_params(params, base=self.app.analysis.event_locked_params)
        self.app.analysis.event_locked_params = p
        for attr, key in (("ent_evt_pre", "pre_s"), ("ent_evt_post", "post_s")):
            w = getattr(self.app, attr, None)
            if w is None:
                continue
            try:
                w.delete(0, "end")
                w.insert(0, format_param(p[key]))
            except Exception as exc:
                log.debug("event-locked apply_params: %s: %s", attr, exc)
        chk = getattr(self.app, "chk_evt_z", None)
        if chk is not None:
            try:
                chk.select() if p["zscore"] else chk.deselect()
            except Exception as exc:
                log.debug("event-locked apply_params: z-score checkbox: %s", exc)
        self.refresh_label_choices(prefer=p["label"])
        return p

    def refresh_label_choices(self, prefer: "Optional[str]" = None) -> "list[str]":
        """Fill the label filter with "All" + the annotation labels; keep the
        current (or *prefer*-red) choice while it still exists, else "All"."""
        values = annotation_label_options(self.app.analysis.annotations)
        cb = getattr(self.app, "cb_evt_label", None)
        if cb is None:
            return values
        try:
            current = prefer if prefer is not None else cb.get()
            cb.configure(values=values)
            cb.set(current if current in values else ALL_LABELS)
        except Exception as exc:
            log.debug("event-locked label choices: %s", exc)
        return values

    def on_annotations_changed(self) -> None:
        """After an annotation add / edit / delete: refresh the label filter;
        a result whose events no longer match the annotations is dropped
        (a colour or end-time edit keeps it)."""
        self.refresh_label_choices()
        el = self.app.analysis.event_locked
        if el is None:
            return
        anns = self.app.analysis.annotations
        now = select_events(anns, el.params.get("label_filter", ALL_LABELS))
        if events_signature(now) != events_signature(el.events):
            self.invalidate("annotations")
            return
        # The other annotations bound the windows too (isolation): one of
        # another label added / moved / removed next to an event changes
        # that event's validity; one far from every event changes nothing.
        others = other_annotation_onsets(anns, now)
        changed = ({round(o, 6) for o in others}
                   ^ {round(float(o), 6) for o in el.other_onsets})
        pre_s = float(el.params.get("pre_s", 0.0) or 0.0)
        post_s = float(el.params.get("post_s", 0.0) or 0.0)
        if any(onset - pre_s - 1e-6 <= o <= onset + post_s + 1e-6
               for o in changed for onset, _lab in el.events):
            self.invalidate("annotations")
        else:
            el.other_onsets = list(others)

    # ── display ──────────────────────────────────────────────────────────

    def _set_label(self, text: str, color: str) -> None:
        lbl = getattr(self.app, "lbl_evt_status", None)
        if lbl is not None:
            lbl.configure(text=text, text_color=color)

    def invalidate(self, reason: "Optional[str]" = None) -> None:
        """Drop the event-locked result (and its table) and blank the panel
        -- via AnalysisController._invalidate_event_locked whenever the
        Autonomic result goes, and when the annotations it was locked to
        change. *reason* "peaks", "autonomic" or "annotations" says why a
        result went stale; None shows the idle hint. Main thread only."""
        a = self.app.analysis
        # From the state, not the slot: _reset_result_plots() may already
        # have blanked the slot.
        had = a.event_locked is not None
        a.event_locked = None
        a.event_locked_df = None
        slot = (getattr(self.app, "_slots", None) or {}).get("event_locked")
        if slot is not None:
            try:
                slot._draw_fn = None
                slot._show_placeholder()
            except Exception as exc:
                log.debug("event-locked invalidate: slot reset failed: %s", exc)
        txt = getattr(self.app, "txt_event_locked", None)
        if txt is not None and hasattr(self.app, "_set_textbox"):
            self.app._set_textbox(txt, "")
        if had and reason in self._STALE_TEXT:
            # Kept in state, so a UI rebuild shows it again (show()).
            a.event_locked_stale = self._STALE_TEXT[reason]
            self._set_label(self._STALE_TEXT[reason], _theme.ORANGE)
        elif reason is None:
            a.event_locked_stale = None
            self._set_label(self.IDLE_TEXT, _theme.MUTED)
        # else: nothing was shown -- keep the label, so a repeated
        # invalidation (one peak edit can trigger several) does not erase
        # the stale notice of the first.

    def _zscore(self) -> bool:
        chk = getattr(self.app, "chk_evt_z", None)
        try:
            return bool(chk.get()) if chk is not None else bool(
                self.app.analysis.event_locked_params.get("zscore", True))
        except Exception:
            return True

    def redraw_plot(self) -> None:
        """Re-render the trace plot from the stored result (no-op without
        one) -- e.g. after the z-score toggle (display only)."""
        el = self.app.analysis.event_locked
        slot = (getattr(self.app, "_slots", None) or {}).get("event_locked")
        if el is None or slot is None:
            return
        z = self._zscore()
        slot.update(lambda fig: draw_event_locked(fig, el, zscore=z))

    def on_zscore_toggle(self) -> None:
        self.app.analysis.event_locked_params = self.widget_params()
        self.redraw_plot()

    def show(self) -> "tuple[str, str] | None":
        """Draw the stored result -- plot, table (with its Excel TSV) and
        status label. Returns the label's (text, colour), or None without a
        result (the label then shows the stored stale notice, if any -- e.g.
        after a UI rebuild). Main thread only."""
        el = self.app.analysis.event_locked
        if el is None:
            note = getattr(self.app.analysis, "event_locked_stale", None)
            if note:
                self._set_label(note, _theme.ORANGE)
            return None
        self.redraw_plot()
        txt = getattr(self.app, "txt_event_locked", None)
        if txt is not None:
            self.app._set_textbox(txt, event_table_text(el),
                                  tsv=self.app._df_to_tsv(event_table_frame(el)))
        text, level = event_status(el)
        color = (_theme.GREEN if level == "ok"
                 else (_theme.ORANGE if level == "warn" else _theme.RED))
        self._set_label(text, color)
        return text, color

    # ── compute ──────────────────────────────────────────────────────────

    def _fail(self, msg: str, color: "Optional[str]" = None) -> None:
        color = color or _theme.RED
        self._set_label(f"  {msg}", color)
        self.app._set_status(f"Event-locked: {msg}", color)

    def compute(self) -> None:
        """Lock the stored Autonomic result to the annotations of the label
        filter (onset = annotation start) and show the per-event table and
        the median ± MAD traces."""
        app = self.app
        a = app.analysis
        res = a.autonomic_result
        if res is None:
            self._fail("Compute the Autonomic indices first (⟳ Compute above).", _theme.ORANGE)
            return
        try:
            p = parse_event_inputs(app.cb_evt_label.get(), app.ent_evt_pre.get(),
                                   app.ent_evt_post.get(), app.chk_evt_z.get())
        except ValueError as exc:
            self._fail(str(exc))
            return
        a.event_locked_params = p
        events = select_events(a.annotations, p["label"])
        if not events:
            which = "" if p["label"] == ALL_LABELS else f" labelled “{p['label']}”"
            self._fail(f"No annotation{which} — add one per event (its start is the "
                       "onset) in Annotations, then compute again.", _theme.ORANGE)
            return
        try:
            # Every annotation bounds the windows, also those the label
            # filter leaves out (core.event_window_means, isolation).
            el = compute_event_locked(res, events, p["pre_s"], p["post_s"],
                                      spwvd=a.autonomic_spwvd, label_filter=p["label"],
                                      other_onsets=other_annotation_onsets(a.annotations,
                                                                           events))
        except ValueError as exc:
            self._fail(str(exc))
            return
        a.event_locked = el
        a.event_locked_df = el.table
        a.event_locked_stale = None
        text, color = self.show() or ("", _theme.ORANGE)
        app._set_status(f"Event-locked — {text.strip()}", color)

    # ── cohort registry / CSV ────────────────────────────────────────────

    def _identity(self) -> "tuple[str, str, str]":
        """(filepath, subject, project name) of the loaded recording."""
        app = self.app
        fp = app.signal.filepath or ""
        sub = proj = ""
        for attr in ("ent_subject", "ent_project_name"):
            w = getattr(app, attr, None)
            try:
                val = w.get().strip() if w is not None else ""
            except Exception:
                val = ""
            if attr == "ent_subject":
                sub = val
            else:
                proj = val
        return fp, sub, proj

    def save_to_registry(self) -> int:
        """Write the per-event rows to the cohort registry (SQLite) with the
        subject ID, project name and configuration. With the "All" label
        filter the rows replace this recording's earlier events of the same
        method / window (an event removed since is dropped); with one label
        they are upserted only. Returns the number of rows written."""
        app = self.app
        el = app.analysis.event_locked
        if el is None:
            self._fail("Compute event-locked first.", _theme.ORANGE)
            return 0
        filepath, subject, project = self._identity()
        if not filepath:
            self._fail("No recording file loaded — nothing to attach the events to.")
            return 0
        replace = el.params.get("label_filter", ALL_LABELS) == ALL_LABELS
        n = db.save_event_rows(filepath, registry_rows(el), subject=subject,
                               project_name=project, params=registry_params(el),
                               replace=replace)
        if n <= 0:
            self._fail("Nothing written to the cohort registry (database unavailable "
                       "or the rows were rejected — see the log).")
            return 0
        p = el.params
        msg = (f"Saved {n} event(s) of {os.path.basename(filepath)} to the cohort registry — "
               f"subject “{subject or '—'}”, project “{project or '—'}”, "
               f"{p.get('method')} · {float(p.get('window_s')):g} s")
        if replace:
            msg += " (this recording's earlier events of this configuration replaced)"
        color = _theme.GREEN
        if not subject:
            msg += ". No Subject ID: the calibration treats this recording as its own subject."
            color = _theme.ORANGE
        else:
            warn = subject_warning(subject, filepath, project,
                                   db.list_event_rows(project_name=project, valid_only=False))
            if warn:
                msg += ". " + warn
                color = _theme.ORANGE
        app._set_status(msg, color)
        return n

    def export_csv(self) -> int:
        """Export this recording's per-event table as CSV (registry format,
        one line per event). Returns the number of lines written."""
        app = self.app
        el = app.analysis.event_locked
        if el is None:
            self._fail("Compute event-locked first.", _theme.ORANGE)
            return 0
        filepath, subject, project = self._identity()
        stem = subject or (os.path.splitext(os.path.basename(filepath))[0] if filepath else "events")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"{stem}_event_locked_{ts}.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            title="Export event-locked table as CSV",
        )
        if not path:
            return 0
        try:
            n = db.export_event_rows_csv(path, export_rows(el, filepath, subject, project))
        except OSError as exc:
            self._fail(f"CSV export failed: {exc}")
            return 0
        app._set_status(f"Event-locked CSV — {n} event(s) → {os.path.basename(path)}  ✓",
                        _theme.GREEN)
        return n

    # ── calibration ──────────────────────────────────────────────────────

    def current_gains(self) -> "tuple[float, float]":
        """(ks, kp) as typed in the Autonomic view (unreadable: the stored)."""
        p = self.app.analysis_ctrl.autonomic_widget_params()
        out = []
        for attr, key in (("ent_auto_ks", "ks"), ("ent_auto_kp", "kp")):
            w = getattr(self.app, attr, None)
            try:
                out.append(_to_float(w.get()) if w is not None else float(p[key]))
            except (TypeError, ValueError):
                out.append(float(p[key]))
        return out[0], out[1]

    def current_config(self) -> "tuple[Optional[str], Optional[float]]":
        """(method, window_s) of the Autonomic view -- the calibration
        dialog's default configuration."""
        p = self.app.analysis_ctrl.autonomic_widget_params()
        return method_key(p.get("method")) or None, float(p["window_s"])

    def open_calibration(self) -> None:
        """Open the cohort ks / kp calibration dialog."""
        from ecg.ui.dialogs import CalibrationDialog
        ks, kp = self.current_gains()
        _fp, _sub, project = self._identity()
        CalibrationDialog(self.app, ks=ks, kp=kp, config=self.current_config(),
                          project=project, on_apply=self.apply_gains)

    def apply_gains(self, ks: float, kp: float) -> None:
        """Write *ks* / *kp* into the Autonomic view's entries. Nothing is
        recomputed until ⟳ Compute is pressed (the status says so)."""
        app = self.app
        for attr, value in (("ent_auto_ks", ks), ("ent_auto_kp", kp)):
            w = getattr(app, attr, None)
            if w is None:
                continue
            try:
                w.delete(0, "end")
                w.insert(0, format_param(value))
            except Exception as exc:
                log.debug("apply_gains: %s: %s", attr, exc)
        msg = (f"ks = {float(ks):g}, kp = {float(kp):g} written to the Autonomic view — "
               "not applied yet: press ⟳ Compute to recompute CSI / CPI with them")
        lbl = getattr(app, "lbl_auto_status", None)
        if lbl is not None:
            lbl.configure(text=f"  {msg}", text_color=_theme.ORANGE)
        app._set_status(msg, _theme.ORANGE)
