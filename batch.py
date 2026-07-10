"""
ecg.batch
─────────
Batch-processing pipeline: analyse multiple .mat files in parallel and
export one Excel workbook per file + a combined summary sheet.

Usage (standalone):
    from batch import BatchProcessor
    bp = BatchProcessor(filepaths, params, output_dir, progress_cb)
    bp.run()            # blocking
    bp.run_async()      # returns Thread
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("ecg")


# ── Worker (runs in subprocess) ───────────────────────────────────────────────

def _process_one(
    filepath: str,
    params:   dict,
    out_dir:  str,
) -> dict:
    """Process a single .mat file and return a summary dict.

    Called in a subprocess via ProcessPoolExecutor — must not reference any
    Tkinter or CTk state.
    """
    import sys, os
    # Make sure the ecg package is importable in the subprocess
    _pkg = os.path.dirname(os.path.abspath(__file__))
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)

    from loaders import load_mat_signal
    from filtering import bandpass, notch, normalize
    from detection import fix_polarity
    from analysis import analyse_core, analyse_hrv_freq
    from export import ExcelExporter
    from models import MouseECG

    stem = Path(filepath).stem
    result: dict = {"filepath": filepath, "stem": stem, "ok": False, "error": ""}

    try:
        # ── Load ─────────────────────────────────────────────────────────
        sig_raw, ch, _, detected_fs = load_mat_signal(
            filepath, params.get("channel", "ECG"))
        fs = int(detected_fs or params.get("fs", MouseECG.FS_DEFAULT))

        # ── Filter ───────────────────────────────────────────────────────
        sig = normalize(sig_raw.copy())
        if not params.get("no_filter", True):
            try:
                sig = bandpass(sig, fs, params.get("lp", MouseECG.BP_LO_HZ),
                               params.get("hp", MouseECG.BP_HI_HZ))
            except Exception:
                pass
            if params.get("notch", False):
                try:
                    sig = notch(sig, fs)
                except Exception:
                    pass
        sig = normalize(sig)

        # ── Detect ───────────────────────────────────────────────────────
        sig_out, inverted, cands, proms = fix_polarity(
            sig, fs, params.get("min_rr_ms", MouseECG.MIN_RR_MS))
        thr = float(params.get("threshold", 0.5))
        if len(proms):
            rpeaks = cands[proms >= thr * float(np.max(proms))]
        else:
            rpeaks = cands

        if len(rpeaks) < 10:
            result["error"] = f"Only {len(rpeaks)} peaks detected"
            return result

        # ── Core analysis ─────────────────────────────────────────────────
        r = analyse_core(sig_out, rpeaks, fs)
        r["hrv_freq"] = analyse_hrv_freq(rpeaks, fs)

        # ── Export ───────────────────────────────────────────────────────
        out_path = os.path.join(out_dir, f"{stem}_analysis.xlsx")
        time = np.arange(len(sig_out), dtype=float) / fs
        wb = ExcelExporter.build_workbook(
            results=r,
            signal_flt=sig_out,
            signal_raw=normalize(sig_raw.copy()),
            time=time,
            rpeaks_ok=rpeaks,
            fs=fs,
            filepath=filepath,
            subject=params.get("subject", stem),
            sig_quality=None,
            epoch_df=None,
        )
        wb.save(out_path)

        # ── Summary row ──────────────────────────────────────────────────
        rr_df = r.get("rr_df")
        hrv   = r.get("hrv_td")
        result.update({
            "ok":          True,
            "n_peaks":     len(rpeaks),
            "duration_s":  float(len(sig_out)) / fs,
            "hr_mean":     float(rr_df["HR_bpm"].mean()) if rr_df is not None and len(rr_df) else float("nan"),
            "hr_sd":       float(rr_df["HR_bpm"].std())  if rr_df is not None and len(rr_df) else float("nan"),
            "rr_mean":     float(rr_df["RR_ms"].mean())  if rr_df is not None and len(rr_df) else float("nan"),
            "sdnn":        float(hrv["HRV_SDNN"].values[0])  if hrv is not None and "HRV_SDNN" in hrv.columns else float("nan"),
            "rmssd":       float(hrv["HRV_RMSSD"].values[0]) if hrv is not None and "HRV_RMSSD" in hrv.columns else float("nan"),
            "excel_path":  out_path,
            "channel":     ch,
            "fs":          fs,
            "inverted":    inverted,
        })
    except Exception as exc:
        result["error"] = str(exc)
        log.warning("batch _process_one %s: %s", stem, exc)

    return result


# ── BatchProcessor ────────────────────────────────────────────────────────────

class BatchProcessor:
    """Run _process_one on all filepaths in parallel and collect results."""

    def __init__(
        self,
        filepaths:   "list[str]",
        params:      dict,
        output_dir:  str,
        progress_cb: "Optional[Callable[[int, int, str], None]]" = None,
        max_workers: int = 4,
    ) -> None:
        self.filepaths   = filepaths
        self.params      = params
        self.output_dir  = output_dir
        self.progress_cb = progress_cb
        self.max_workers = max_workers
        self.results:  "list[dict]" = []
        self._stopped: bool = False

    def stop(self) -> None:
        self._stopped = True

    def run(self) -> "list[dict]":
        """Run synchronously — blocks until all files are processed."""
        os.makedirs(self.output_dir, exist_ok=True)
        n = len(self.filepaths)

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_process_one, fp, self.params, self.output_dir): fp
                for fp in self.filepaths
            }
            for done_i, future in enumerate(as_completed(futures), start=1):
                if self._stopped:
                    break
                fp = futures[future]
                try:
                    res = future.result(timeout=300)
                except Exception as exc:
                    res = {"filepath": fp, "stem": Path(fp).stem,
                           "ok": False, "error": str(exc)}
                self.results.append(res)
                if self.progress_cb:
                    self.progress_cb(done_i, n, Path(fp).stem)

        self._write_summary()
        return self.results

    def run_async(self) -> threading.Thread:
        """Run in a background thread; returns the Thread."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

    def _write_summary(self) -> None:
        """Write a combined summary.xlsx into output_dir."""
        if not self.results:
            return
        rows = []
        for r in self.results:
            rows.append({
                "File":        r.get("stem", ""),
                "OK":          "✓" if r.get("ok") else "✗",
                "Error":       r.get("error", ""),
                "N peaks":     r.get("n_peaks", ""),
                "Duration (s)":r.get("duration_s", ""),
                "Mean HR":     r.get("hr_mean", ""),
                "HR SD":       r.get("hr_sd", ""),
                "Mean RR (ms)":r.get("rr_mean", ""),
                "SDNN (ms)":   r.get("sdnn", ""),
                "RMSSD (ms)":  r.get("rmssd", ""),
            })
        df = pd.DataFrame(rows)
        out = os.path.join(self.output_dir, "_batch_summary.xlsx")
        try:
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Summary", index=False)
            log.info("Batch summary → %s", out)
        except Exception as exc:
            log.warning("batch summary write failed: %s", exc)
