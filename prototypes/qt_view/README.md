# Qt prototype of the Signal & Detection view

Not part of the app: a prototype to compare PySide6 + pyqtgraph with the
app's CustomTkinter + matplotlib on long recordings. The calculations are the
app's own (`ecg/core`, `ecg/io`); only the interface is new.

## Run it

```
pip install PySide6 pyqtgraph
python prototypes/qt_view/qt_view.py                  # opens a file dialog
python prototypes/qt_view/qt_view.py rec.smr          # a recording (.smr / .mat)
python prototypes/qt_view/qt_view.py --synthetic 24   # 24 h built from a short ECG
python prototypes/qt_view/qt_view.py rec.mat --synthetic 24   # 24 h made by repeating rec.mat
```

- Mouse wheel: zoom (time axis). Drag: pan. Drag the blue band of the top
  strip to move through the recording. Buttons: 10 s, 1 min, 10 min, 1 h, All.
- Left click near a peak: exclude / restore it. Right click: add a peak at the
  local maximum.
- **Mark noise**, then drag over the trace: a noisy region (its RR intervals
  are left out of the RR strip). Right click inside it: remove it.

Detection is the app's band-pass, polarity check, SG derivative detection and
threshold 0.5; NeuroKit's clean is left out (minutes on a day-long recording).
Nothing is saved.

## Benchmark

```
python prototypes/qt_view/bench.py --synthetic 24
```

Time to show each window (10 s to the whole day) and to move it by half its
width, in the prototype and with matplotlib drawn as the app draws its
detection view (min/max decimation to 6000 points, Agg canvas, without Tk's
copy to the screen, so the app's real times are a little longer).

Synthetic 24 h at 1 kHz (86.4 M samples, 789 k peaks), median of 9:

| Window | matplotlib show | Qt show | matplotlib pan | Qt pan |
|---|---|---|---|---|
| 10 s | 70 ms | 63 ms | 67 ms | 47 ms |
| 1 min | 89 ms | 40 ms | 91 ms | 49 ms |
| 10 min | 346 ms | 59 ms | 272 ms | 66 ms |
| 1 h | 343 ms | 72 ms | 355 ms | 114 ms |
| All (24 h) | 5184 ms | 76 ms | 5415 ms | 70 ms |

The same day at 2 kHz (172.8 M samples, 710 k peaks; a 240 s test recording
repeated), median of 9:

| Window | matplotlib show | Qt show | matplotlib pan | Qt pan |
|---|---|---|---|---|
| 10 s | 70 ms | 30 ms | 56 ms | 42 ms |
| 1 min | 95 ms | 37 ms | 103 ms | 52 ms |
| 10 min | 288 ms | 79 ms | 257 ms | 75 ms |
| 1 h | 475 ms | 53 ms | 391 ms | 112 ms |
| All (24 h) | 5799 ms | 67 ms | 6637 ms | 58 ms |
