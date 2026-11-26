"""Prototype: the Signal & Detection view in PySide6 + pyqtgraph.

Not part of the app. It answers one question: is Qt + pyqtgraph smoother
than CustomTkinter + matplotlib on long recordings (24 h and more)? The
calculations are the app's own (ecg.core, ecg.io); only the interface is new.

    python prototypes/qt_view/qt_view.py                  # opens a file dialog
    python prototypes/qt_view/qt_view.py rec.smr          # a recording (.smr / .mat)
    python prototypes/qt_view/qt_view.py --synthetic 24   # 24 h built from a short one

Navigation: mouse wheel = zoom, drag = pan, drag the blue band of the
overview (top) to move through the recording.
Editing: left click near a peak excludes / restores it; right click adds a
peak at the local maximum; "Mark noise" then drag over the trace marks a
noisy region; right click inside a noisy region removes it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyqtgraph as pg  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

from ecg.core.detection import (  # noqa: E402
    apply_threshold, detect_peaks_sg_derivative, fix_polarity,
)
from ecg.core.filtering import bandpass, normalize  # noqa: E402
from ecg.core.models import MouseECG  # noqa: E402

MAX_MARKERS = 4000          # peaks drawn only when fewer are in view
NOISE_BRUSH = (154, 160, 166, 70)
NOISE_PEN = (95, 99, 104, 160)


# ── data ────────────────────────────────────────────────────────────────

def detect(sig_raw: np.ndarray, fs: float) -> "tuple[np.ndarray, np.ndarray]":
    """(filtered signal, R peaks): the app's band-pass, polarity check, SG
    derivative detection and threshold (NeuroKit's clean is left out: it
    takes minutes on a day-long recording and only refines the trace)."""
    sig = bandpass(sig_raw, fs, MouseECG.BP_LO_HZ, MouseECG.BP_HI_HZ)
    sig = normalize(np.asarray(sig, dtype=float))
    sig, _inv, _, _ = fix_polarity(sig, fs, MouseECG.MIN_RR_MS)
    cands, proms, _amp = detect_peaks_sg_derivative(sig, fs=fs, min_rr_ms=MouseECG.MIN_RR_MS,
                                                    target_fs=float(fs))
    accepted, _rej, _t = apply_threshold(sig, cands, proms, 0.5, fs=fs)
    return sig.astype(np.float32), np.sort(np.asarray(accepted, dtype=np.int64))


def load_file(path: str) -> "tuple[np.ndarray, float, str]":
    from ecg.io.loaders import load_mat_signal
    sig, ch, _keys, fs = load_mat_signal(path, "ECG")
    if not fs:
        raise ValueError("Sampling rate not found in the file.")
    return np.asarray(sig, dtype=np.float64), float(fs), f"{os.path.basename(path)} · {ch}"


def synthetic(hours: float, base: "str | None") -> "tuple[np.ndarray, np.ndarray, float, str]":
    """A recording of *hours* made by repeating a short one (detected once,
    its peaks repeated: detecting a whole day would take minutes and says
    nothing about the interface). Default: a synthetic mouse ECG, 1 kHz."""
    if base:
        raw, fs, _name = load_file(base)
    else:
        fs = 1000.0
        rng = np.random.default_rng(0)
        n = int(120 * fs)
        beats, b = [], 0.05
        while b < n / fs - 0.05:                                # ~550 bpm, slow swing
            beats.append(b)
            b += 0.11 + 0.01 * np.sin(2 * np.pi * b / 20) + rng.normal(0, 0.002)
        raw = rng.normal(0, 0.03, n)
        for b in beats:
            i = int(b * fs)
            k = np.arange(-15, 30)
            ok = (i + k >= 0) & (i + k < len(raw))
            raw[(i + k)[ok]] += (np.exp(-0.5 * (k / 2.5) ** 2) - 0.25 * np.exp(-0.5 * ((k - 12) / 4) ** 2))[ok]
    sig, peaks = detect(raw, fs)
    n_rep = max(1, int(round(hours * 3600 * fs / len(sig))))
    tile = len(sig)
    full = np.tile(sig, n_rep)
    all_peaks = (peaks[None, :] + (np.arange(n_rep, dtype=np.int64) * tile)[:, None]).ravel()
    return full, all_peaks, fs, f"synthetic · {len(full) / fs / 3600:.1f} h · {fs:.0f} Hz"


# ── views ───────────────────────────────────────────────────────────────

class TraceBox(pg.ViewBox):
    """The detail view's ViewBox: wheel zoom on x only, and a drag that marks
    a noisy region while Mark noise is on."""

    def __init__(self, owner: "DetectionView") -> None:
        super().__init__()
        self.owner = owner
        self.setMouseMode(pg.ViewBox.PanMode)
        self._drag_region: "pg.LinearRegionItem | None" = None

    def wheelEvent(self, ev, axis=None):          # x only, as in the app
        super().wheelEvent(ev, axis=0)

    def mouseDragEvent(self, ev, axis=None):
        if not self.owner.noise_mode or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        x = self.mapSceneToView(ev.scenePos()).x()
        if ev.isStart():
            x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
            self._drag_region = pg.LinearRegionItem((x0, x), movable=False,
                                                    brush=NOISE_BRUSH, pen=NOISE_PEN)
            self.addItem(self._drag_region)
        elif self._drag_region is not None:
            start = self.mapSceneToView(ev.buttonDownScenePos()).x()
            self._drag_region.setRegion(sorted((start, x)))
            if ev.isFinish():
                self.removeItem(self._drag_region)
                self._drag_region = None
                self.owner.add_noise(*sorted((start, x)))


class DetectionView(QtWidgets.QMainWindow):

    def __init__(self, sig: np.ndarray, peaks: np.ndarray, fs: float, name: str) -> None:
        super().__init__()
        self.sig, self.fs, self.name = sig, fs, name
        self.peaks = np.asarray(peaks, dtype=np.int64)
        self.excluded = np.zeros(len(self.peaks), dtype=bool)
        self.noise: "list[tuple[float, float]]" = []
        self.noise_items: "list[pg.LinearRegionItem]" = []
        self.noise_mode = False
        self.t = None                              # x is computed per view (no 24 h time array)
        self.setWindowTitle(f"ECG Analysis — Qt prototype · {name}")
        self.resize(1500, 900)
        pg.setConfigOptions(antialias=False, background="w", foreground="#333")

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(10, 8, 10, 8)
        self.setCentralWidget(central)

        bar = QtWidgets.QHBoxLayout()
        lay.addLayout(bar)
        title = QtWidgets.QLabel(f"<b>ECG Analysis</b> <span style='color:#888'>Qt prototype</span>"
                                 f" &nbsp; {name}")
        bar.addWidget(title)
        bar.addStretch(1)
        self.btn_noise = QtWidgets.QPushButton("Mark noise")
        self.btn_noise.setCheckable(True)
        self.btn_noise.toggled.connect(self._toggle_noise)
        bar.addWidget(self.btn_noise)
        for label, secs in (("10 s", 10), ("1 min", 60), ("10 min", 600), ("1 h", 3600),
                            ("All", None)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _c=False, s=secs: self.show_window(s))
            bar.addWidget(b)

        glw = pg.GraphicsLayoutWidget()
        lay.addWidget(glw, 1)
        # Overview: the heart rate of the whole recording, the shown window as a band.
        self.ov = glw.addPlot(row=0, col=0)
        self.ov.setMaximumHeight(110)
        self.ov.setMouseEnabled(x=False, y=False)
        self.ov.hideButtons()
        self.ov.setLabel("left", "HR")
        self.ov_curve = self.ov.plot(pen=pg.mkPen("#1A56DB", width=1))
        self.ov_curve.setDownsampling(auto=True, method="peak")
        self.ov_curve.setClipToView(True)
        self.band = pg.LinearRegionItem(brush=(26, 86, 219, 40))
        self.band.sigRegionChanged.connect(self._band_moved)
        self.ov.addItem(self.band)
        # Detail: the ECG and its peaks.
        self.box = TraceBox(self)
        self.det = glw.addPlot(row=1, col=0, viewBox=self.box)
        self.det.setLabel("left", "ECG (norm.)")
        self.det.showGrid(x=True, y=True, alpha=0.15)
        self.curve = self.det.plot(pen=pg.mkPen("#2563EB", width=1))
        self.curve.setDownsampling(auto=True, method="peak")
        self.curve.setClipToView(True)
        self.marks = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(16, 163, 74, 200))
        self.marks_x = pg.ScatterPlotItem(size=10, symbol="x", pen=pg.mkPen("#DC2626", width=2))
        self.det.addItem(self.marks)
        self.det.addItem(self.marks_x)
        # RR strip, x linked to the detail.
        self.rr = glw.addPlot(row=2, col=0)
        self.rr.setMaximumHeight(160)
        self.rr.setXLink(self.det)
        self.rr.setLabel("left", "RR (ms)")
        self.rr.setLabel("bottom", "Time (s)")
        self.rr_curve = self.rr.plot(pen=pg.mkPen("#64748B", width=1), symbol="o", symbolSize=3,
                                     symbolBrush="#64748B", symbolPen=None)
        self.rr_curve.setDownsampling(auto=True, method="peak")
        self.rr_curve.setClipToView(True)
        self.det.sigXRangeChanged.connect(self._range_changed)
        self.det.scene().sigMouseClicked.connect(self._clicked)

        self.status = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status, 1)
        self._set_data()
        self.show_window(10)

    # ── data in the plots ──

    def _set_data(self) -> None:
        n = len(self.sig)
        self.curve.setData(np.arange(n, dtype=np.float64) / self.fs, self.sig)
        self._rr_update()
        self.band.setBounds((0, n / self.fs))
        self.ov.setXRange(0, n / self.fs, padding=0)

    def _kept(self) -> np.ndarray:
        return self.peaks[~self.excluded]

    def _rr_update(self) -> None:
        kept = self._kept()
        t = kept[1:] / self.fs
        rr = np.diff(kept) / self.fs * 1000.0
        if self.noise:                              # intervals in noise: left out
            lo, hi = kept[:-1] / self.fs, kept[1:] / self.fs
            bad = np.zeros(len(rr), bool)
            for a, b in self.noise:
                bad |= (hi > a) & (lo < b)
            rr = np.where(bad, np.nan, rr)
        self.rr_curve.setData(t, rr, connect="finite")
        self.ov_curve.setData(t, 60000.0 / rr, connect="finite")

    def _range_changed(self, *_a) -> None:
        t0 = time.perf_counter()
        x0, x1 = self.det.viewRange()[0]
        i0, i1 = np.searchsorted(self.peaks, [x0 * self.fs, x1 * self.fs])
        if i1 - i0 <= MAX_MARKERS:
            p, ex = self.peaks[i0:i1], self.excluded[i0:i1]
            self.marks.setData(p[~ex] / self.fs, self.sig[p[~ex]])
            self.marks_x.setData(p[ex] / self.fs, self.sig[p[ex]])
        else:
            self.marks.setData([], [])
            self.marks_x.setData([], [])
        self.band.blockSignals(True)
        self.band.setRegion((x0, x1))
        self.band.blockSignals(False)
        dt = (time.perf_counter() - t0) * 1000
        n_view = int((x1 - x0) * self.fs)
        self.status.setText(f"{n_view:,} samples in view · {i1 - i0:,} peaks"
                            f"{'' if i1 - i0 <= MAX_MARKERS else ' (zoom in to see them)'}"
                            f" · {len(self.noise)} noisy region(s) · update {dt:.1f} ms")

    def _band_moved(self) -> None:
        a, b = self.band.getRegion()
        self.det.setXRange(a, b, padding=0)

    def show_window(self, secs: "float | None") -> None:
        end = len(self.sig) / self.fs
        if secs is None:
            self.det.setXRange(0, end, padding=0)
        else:
            x0 = self.det.viewRange()[0][0]
            x0 = min(max(0.0, x0), max(0.0, end - secs))
            self.det.setXRange(x0, x0 + secs, padding=0)
        self.det.enableAutoRange(axis="y")

    # ── editing ──

    def _toggle_noise(self, on: bool) -> None:
        self.noise_mode = on
        self.btn_noise.setText("Mark noise ON" if on else "Mark noise")

    def add_noise(self, a: float, b: float) -> None:
        if b - a < 0.05:
            return
        self.noise.append((a, b))
        item = pg.LinearRegionItem((a, b), movable=False, brush=NOISE_BRUSH, pen=NOISE_PEN)
        self.det.addItem(item)
        self.noise_items.append(item)
        self._rr_update()
        self._range_changed()

    def _clicked(self, ev) -> None:
        if ev.isAccepted() or not self.det.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = self.box.mapSceneToView(ev.scenePos()).x()
        x0, x1 = self.det.viewRange()[0]
        if (x1 - x0) * self.fs > 200_000:            # too far out to edit beats
            return
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            for k, (a, b) in enumerate(self.noise):
                if a <= x <= b:
                    self.det.removeItem(self.noise_items.pop(k))
                    del self.noise[k]
                    self._rr_update()
                    self._range_changed()
                    return
            tol = int(0.02 * self.fs)
            c = int(x * self.fs)
            lo, hi = max(0, c - tol), min(len(self.sig), c + tol + 1)
            new = lo + int(np.argmax(self.sig[lo:hi]))
            i = np.searchsorted(self.peaks, new)
            if i < len(self.peaks) and self.peaks[i] == new:
                return
            self.peaks = np.insert(self.peaks, i, new)
            self.excluded = np.insert(self.excluded, i, False)
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton and not self.noise_mode:
            i = np.searchsorted(self.peaks, x * self.fs)
            near = [j for j in (i - 1, i) if 0 <= j < len(self.peaks)]
            if not near:
                return
            j = min(near, key=lambda j: abs(self.peaks[j] - x * self.fs))
            if abs(self.peaks[j] - x * self.fs) > 0.03 * self.fs:
                return
            self.excluded[j] = not self.excluded[j]
        else:
            return
        ev.accept()
        self._rr_update()
        self._range_changed()


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", nargs="?", help="recording (.smr / .mat); a dialog opens if left out")
    ap.add_argument("--synthetic", type=float, metavar="HOURS",
                    help="build a recording of HOURS by repeating FILE (or a synthetic ECG)")
    args = ap.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    t0 = time.perf_counter()
    if args.synthetic:
        sig, peaks, fs, name = synthetic(args.synthetic, args.file)
    else:
        path = args.file or QtWidgets.QFileDialog.getOpenFileName(
            None, "Open a recording", "", "Recordings (*.smr *.mat);;All files (*)")[0]
        if not path:
            return 0
        raw, fs, name = load_file(path)
        sig, peaks = detect(raw, fs)
    print(f"loaded and detected in {time.perf_counter() - t0:.1f} s: {name}, "
          f"{len(sig):,} samples, {len(peaks):,} peaks")
    win = DetectionView(sig, peaks, fs, name)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
