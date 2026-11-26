"""Benchmark: the detection view of a long recording drawn by the Qt
prototype (pyqtgraph) and by matplotlib as the app draws it today.

    python prototypes/qt_view/bench.py --synthetic 24
    python prototypes/qt_view/bench.py rec.smr

For each window (10 s to the whole recording): the time to show it (jump
there and draw) and to move it by half its width (a pan step), median of
several. matplotlib is timed on its Agg canvas, without Tk's own copy to
the screen: the app's real times are a little longer.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import qt_view  # noqa: E402

WINDOWS = [("10 s", 10.0), ("1 min", 60.0), ("10 min", 600.0), ("1 h", 3600.0), ("All", None)]
REPEAT = 9


def _median_ms(fn, n: int = REPEAT) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def bench_qt(sig, peaks, fs, name) -> "dict[str, tuple[float, float]]":
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = qt_view.DetectionView(sig, peaks, fs, name)
    win.resize(1500, 900)
    win.show()
    app.processEvents()
    end = len(sig) / fs
    out = {}
    for label, w in WINDOWS:
        w = end if w is None else min(w, end)
        start = [end * 0.37]

        def show(span=w):
            x0 = min(start[0], end - span)
            win.det.setXRange(x0, x0 + span, padding=0)
            win.det.enableAutoRange(axis="y")
            win.grab()                                   # draws now, as on screen

        def pan(span=w):
            start[0] = (start[0] + span / 2) % max(end - span, 1e-9)
            show(span)
        first = _median_ms(show, 3)
        step = _median_ms(pan)
        out[label] = (first, step)
    win.close()
    win.deleteLater()
    app.processEvents()
    return out


def bench_mpl(sig, peaks, fs) -> "dict[str, tuple[float, float]]":
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from ecg.core.filtering import downsample_xy_for_display
    fig = Figure(figsize=(15, 6.5), dpi=100)
    canvas = FigureCanvasAgg(fig)
    end = len(sig) / fs
    out = {}
    for label, w in WINDOWS:
        w = end if w is None else min(w, end)
        start = [end * 0.37]

        def draw(span=w):
            x0 = min(start[0], end - span)
            i0, i1 = int(x0 * fs), int((x0 + span) * fs)
            y = sig[i0:i1]
            x = np.arange(i0, i1) / fs
            xd, yd = downsample_xy_for_display(x, y, 6000)
            fig.clear()
            ax, ax_rr = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            ax.plot(xd, yd, lw=0.8, color="#2563EB")
            j0, j1 = np.searchsorted(peaks, [i0, i1])
            if j1 - j0 <= qt_view.MAX_MARKERS:
                p = peaks[j0:j1]
                ax.scatter(p / fs, sig[p], s=14, color="#16A34A", zorder=3)
            p = peaks[j0:j1]
            if len(p) > 1:
                tx, rr = downsample_xy_for_display(p[1:] / fs, np.diff(p) / fs * 1000, 6000)
                ax_rr.plot(tx, rr, lw=0.8, color="#64748B")
            canvas.draw()

        def pan(span=w):
            start[0] = (start[0] + span / 2) % max(end - span, 1e-9)
            draw(span)
        first = _median_ms(draw, 3)
        step = _median_ms(pan)
        out[label] = (first, step)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--synthetic", type=float, metavar="HOURS")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.synthetic:
        sig, peaks, fs, name = qt_view.synthetic(args.synthetic, args.file)
    else:
        raw, fs, name = qt_view.load_file(args.file)
        sig, peaks = qt_view.detect(raw, fs)
    print(f"{name}: {len(sig):,} samples, {len(peaks):,} peaks "
          f"(ready in {time.perf_counter() - t0:.1f} s)\n")
    qt = bench_qt(sig, peaks, fs, name)
    mpl = bench_mpl(sig, peaks, fs)
    print(f"{'window':<8} {'matplotlib show':>16} {'Qt show':>9} {'matplotlib pan':>15} {'Qt pan':>8}")
    for label, _w in WINDOWS:
        (qf, qs), (mf, ms) = qt[label], mpl[label]
        print(f"{label:<8} {mf:>13.0f} ms {qf:>6.0f} ms {ms:>12.0f} ms {qs:>5.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
