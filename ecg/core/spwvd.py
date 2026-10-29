"""
ecg.core.spwvd
──────────────
Time-resolved LF / HF power of the RR series with the smoothed pseudo
Wigner-Ville distribution (SPWVD).

This is a faithful NumPy port of ``compute_PWVD.m`` from Diego Candia-Rivera's
robust_hrv toolbox (github.com/diegocandiar/robust_hrv), the spectral
reference of Candia-Rivera et al., J Physiol 2025, "Modelling the
time-resolved modulations of cardiac activity in rats", which itself
implements the ambiguity-domain kernel of Orini et al. (2012, IEEE TBME).
Pure NumPy/SciPy, no ``ecg.ui`` import, safe to call from a background
thread.

Pipeline (MATLAB statement → what this module does)
───────────────────────────────────────────────────
1. ``interp1(t_IBI, IBI, t, 'spline')`` on ``t = t_IBI(1):1/Frr:t_IBI(end)``
   → not-a-knot cubic spline (``scipy.interpolate.CubicSpline``, the same
   spline as MATLAB's) through the VALID intervals only, evaluated on an
   even grid at *fs_resample* from the first to the last valid sample.
2. ``detrend`` (linear), then ``hilbert(x - mean(x))`` → the analytic signal
   z of the demeaned, detrended series (``scipy.signal.hilbert`` uses the
   same one-sided spectrum construction as MATLAB's ``hilbert``).
3. The lag loop of the Wigner-Ville distribution (TFTB ``tfrwv``):
   ``wvx(rem(Fbins + tau, Fbins) + 1, ti) = z(ti + tau) * conj(z(ti - tau))``
   for ``|tau| <= min(ti - 1, N - ti, Fbins/2 - 1)`` plus the symmetrised
   ``tau = Fbins/2`` row, vectorised here over time (one NumPy slice per
   lag instead of one Python iteration per sample).
4. ``wvx = real(fft(wvx))`` (lag → frequency), then
   ``AF = fft(ifft(wvx).')`` (back to lag, then time → Doppler): the
   ambiguity function. The MATLAB transposes only reorient the array; here
   the array stays (lag, time) and each transform is taken along the axis
   MATLAB's column-wise ``fft`` would use.
5. MATLAB then transposes AF back when
   ``rem(size(AF,2), 512|1024|4096|256) ~= 0`` (all four). ``size(AF,2)`` is
   Fbins, so that branch is taken exactly when Fbins is NOT a multiple of
   256 -- and then the kernel coordinates below are built on swapped axes,
   which is not a meaningful TFR. Only powers of two from 256 to
   ``SPWVD_MAX_FBINS`` (65536) are accepted: every power of two >= 256 is
   a multiple of 256, so the branch is never taken. Other multiples of 256
   (768, 3072, ...) would not take it either but are not supported (the
   reference uses ``pow2(12)``, and power-of-two FFTs stay fast); the upper
   bound keeps the fbins-length vectors built before the memory guard
   small. Any other *fbins* raises ValueError.
6. Kernel: ``dy = (-Ccol/2:Ccol/2-1)/(Ccol/2)`` (lag), ``dx`` (Doppler, with
   MATLAB's even/odd-length branch), ``meshgrid``, ``tau1 = x/tau0``,
   ``v1 = y/v0``, ``mu = tau1.^2.*(v1.^2).^alfa + (tau1.^2).^alfa.*v1.^2 +
   2*r*((tau1.*v1).^beta).^gamma`` with the hard-coded ``alfa = 0``,
   ``r = 0``, ``beta = gamma = 1`` (so ``mu = tau1.^2 + v1.^2``),
   ``t_wind = exp(-pi*(mu.^2).^lambda)`` normalised by its maximum, then
   ``fftshift`` on both axes. The fftshift of a kernel that is a function of
   separable coordinates equals the kernel evaluated on fftshift-ed
   coordinate vectors, which is how it is built here (row block by row
   block, bit-identical values, no full-size kernel array).
7. ``TFR_RR = real(ifft(fft(Asmooth).').')`` with ``Asmooth = (AF.*K).'``:
   lag → frequency, Doppler → time, real part.
8. ``f = linspace(0, Frr/2, Fbins)``; ``LF = trapz(TFR_RR(LF_ix, :))`` over
   the bins with ``lo <= f <= hi`` -- MATLAB ``trapz(Y)``, i.e. UNIT
   spacing, so the band power is a sum over bins, not an integral in Hz.
   Only the band rows are carried through the last transform (step 7's
   Doppler → time ``ifft`` acts on each frequency row independently, so
   this is exact).

Note that ``linspace(0, Frr/2, Fbins)`` is the MATLAB code's frequency axis
(it is kept for fidelity); the strict WVD bin spacing would be
``Frr / (2 * Fbins)``, a relative difference of 1/Fbins (0.02 % at 4096).

Units and scale
───────────────
LF / HF are in (input RR unit)², i.e. ms² × an arbitrary scale set by
*fbins* and the kernel: the TFR is unnormalised exactly as in the MATLAB
reference, so the numbers are only comparable between windows of the same
recording analysed with the same parameters (pre vs post an event, LF vs HF
trend) -- not with Welch / Lomb band powers, and not across *fbins*. The
SPWVD is not positive definite: short negative excursions of the band
power are a property of the method, not an error.

Parameters for mice
───────────────────
The paper used Frr = 20 Hz and rat bands LF 0.1–1 Hz / HF 1–3.5 Hz. The
defaults here are 20 Hz and the app's mouse bands (MouseECG.LF / HF, 0.4–1.5
and 1.5–5 Hz); pass ``analysis_controller.get_freq_bands()`` to follow the
UI preset. The kernel constants v0 = 0.03, tau0 = 0.06, lambda = 0.3 are
the ones hard-coded in compute_PWVD.m -- the toolbox's only set, with no
per-species variant (the file's own defaults are Frr = 4 Hz and the human
bands LF 0.04–0.15 / HF 0.15–0.4 Hz; the rat bands are commented out). The
paper applied them at Frr = 20 Hz to rats. They are defined in NORMALISED
units (per lag bin of the fbins-long lag axis, per Doppler bin in
cycles/sample), so their extent in seconds and Hz depends on
*fs_resample* and *fbins*: at 20 Hz the same constants smooth over a 5×
shorter time span than at the file's 4 Hz. They have NOT been validated
in mice.

Gaps (excluded intervals)
─────────────────────────
Excluded intervals (``valid`` False: out of bounds, spanning a removed beat,
non-finite) are skipped before the spline. The uncovered time between
consecutive valid samples j-1 and j is ``(t[j] - t[j-1]) - rr[j]/1000``:
the time not explained by the valid interval j itself, i.e. the summed
duration of the excluded intervals between them. Being time-based, it also
catches gaps when only the valid intervals (``rri`` / ``rri_time``) are
passed. Then:

* a LONG gap (uncovered time > ``SPWVD_NAN_GAP_S``,
  MouseECG.SPECTRAL_MAX_GAP_S = 2 s) SPLITS the series: each stretch
  between long gaps is processed as its own compute_PWVD series (its own
  spline, linear detrend, demean, Hilbert transform and SPWVD, chunked if
  long), and every output sample between two stretches is NaN. Nothing is
  bridged or smoothed across a long gap -- a deviation from
  compute_PWVD.m, which interpolates and detrends its whole input as one
  series (the reference code has no gap handling at all);
* a stretch shorter than ``max(SPWVD_MIN_SEGMENT_S, 2 / LF low edge)``
  (10 s with the mouse bands, 20 s with the rat ones), or with fewer than
  ``SPWVD_MIN_VALID`` intervals / ``SPWVD_MIN_SAMPLES`` samples, is NaN:
  it cannot hold two periods of the slowest LF frequency nor fill the ~12 s
  lag kernel, so its values would be finite but meaningless
  (``params['n_segments_too_short']``);
* SHORT gaps inside a stretch are bridged by the spline, like NeuroKit's
  resampling; when they cover more than ``SPWVD_MAX_BRIDGED_FRAC`` (5 %)
  of the computed stretches' time the whole computation is refused
  (ValueError), as for the other spectral HRV paths.

If no stretch can be computed, ValueError.

Chunking and memory
───────────────────
One working array holds ``fbins × samples`` complex values, 16 bytes each:
16 × 4096 × 4800 ≈ 315 MB for a 240 s span at 20 Hz. Peak working memory
is that array plus the band rows carried to the last step (about 1.5× in
total with the mouse bands). A segment whose resampled series is longer
than ``chunk_s + 2 * margin_s`` is therefore processed in overlapping
chunks. Steps 1-2 (spline, linear detrend, demean, Hilbert transform)
always run ONCE on the whole segment, exactly as compute_PWVD.m does on
its series -- that is O(n) memory, 16 bytes per sample (28 MB for 24 h at
20 Hz). Only steps 3-8 (WVD, ambiguity function, kernel, band sums) run per
chunk, on the slice of that analytic signal covering the chunk's extended
span (a core of at most *chunk_s*, plus *margin_s* on each side, clipped
at the segment ends). The cores tile the segment and are stitched with a
linear cross-fade of ``margin_s`` total width centred on each core
boundary (half a margin either side, and at most half the shortest core),
so the output has no step at the boundaries.

Why this is close to, but not exactly, the single pass: the lag kernel is
defined per lag bin and decays below 1e-3 beyond ~12 s of lag at 20 Hz
with fbins = 4096 (it scales with fbins / fs), and the Doppler kernel is
defined in cycles/sample and so is independent of the chunk length; but
the time smoothing it implies is heavy-tailed (lambda = 0.3: at lag 0,
1.4 % of its mass lies beyond ±30 s and 0.6 % beyond ±60 s) and each
chunk's Doppler FFT wraps around the chunk instead of the whole series.
So power from the far end of an extended chunk leaks into its core: a
large HR / HRV change sitting in a chunk's margin biases the NEXT core
boundary, not only the samples around the change. Zero-padding each chunk
along time (a linear instead of a circular smoothing) was tried and is
worse: it trades the leak for a truncation bias of the same order
everywhere. The default margin is therefore 60 s (``SPWVD_MARGIN_S``;
1.3× the memory and run time of 30 s margins). Measured at the defaults
(120 s cores, 60 s margins), chunked vs single-pass outputs, central
region, 30 s post/pre ratios checked at the change AND at every core
boundary:

- stationary RR (0.8 + 2.5 Hz oscillations, 300 s): ~1e-5 median, ~1e-4
  max relative difference; ratios identical to 0.01 %;
- a 30 ms sigmoid HR step on a core boundary (300 s): ~0.2-0.4 % median,
  up to ~1.4 % max sample-wise; ratios within 0.4 % (1.4 % with 30 s
  margins);
- a 20 ms quadratic drift (300 s): ~0.1 % median, below 1 % max; ratios
  within 0.1 %;
- a mouse-like 400 s series (100 ms RR, 3 / 0.6 / 0.05 Hz modulation) with
  an RR -15 ms and HRV x0.5 step 100 s before a core boundary: 0.1-0.25 %
  median, up to ~2 % max sample-wise; ratios within 1.0 % at that
  boundary (2.3 % with 30 s margins, 0.25 % with 90 s margins);
- the same change on a 300 s series, 50 s from both core boundaries (in
  the margins of the chunks on both sides): 0.2-0.4 % median, up to ~6 %
  max sample-wise (LF, near low-power samples); ratios within 0.35 % at
  the change and at both boundaries.
The deviation is data-dependent -- largest near low-power samples and
after large changes inside a margin. The largest sample-to-sample change of
the stitched series near a core boundary is within ~2 % of the single
pass's own (no stitching jump).

Detrending each chunk on its own span instead (an earlier version of this
module) deviated by ~7 % median on the same trended series, with ~7 %
jumps at core boundaries; wider margins alone did not fix that. Margins
much shorter than the lag-kernel extent are allowed but lose this
accuracy; longer ones (90 s) reduce it further. Pass a *chunk_s* longer
than the recording to get the literal single pass (memory permitting: up
to ~12 min at 20 Hz with fbins = 4096 and the mouse bands under the
guard). A chunk whose estimated
peak memory exceeds ``SPWVD_MAX_CHUNK_BYTES`` (1.5 GB) is refused with a
ValueError before any fbins × samples array is allocated. The estimate
was checked against tracemalloc: the traced peak is 0.91-0.97 of it for
180-600 s single passes.

Run time is dominated by the FFTs of the working array: about 1.2 s per
240 s chunk at the defaults on a desktop CPU, i.e. ~0.7 s per minute of
recording (some 15 min for 24 h). Call it from a worker thread, and
preferably on the stretch around the events of interest rather than on a
whole day.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.fft as _sfft
from scipy.interpolate import CubicSpline
from scipy.signal import detrend as _detrend
from scipy.signal import hilbert as _hilbert

from ecg.core.analysis import rr_valid_mask
from ecg.core.models import MouseECG

# numpy 2.x renamed trapz to trapezoid; resolved locally (same shim as
# ecg.core.analysis) so this module is importable on either.
_trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid", None)
if _trapz is None:
    from scipy.integrate import trapezoid as _trapz  # last-resort fallback


# ── Public constants ─────────────────────────────────────────────────────
SPWVD_FS_HZ:           float = 20.0           # Frr of compute_CSI_CVI_rat / the paper
SPWVD_FBINS:           int   = 4096           # pow2(12) in compute_PWVD.m
SPWVD_MIN_FBINS:       int   = 256            # see module docstring, step 5
SPWVD_MAX_FBINS:       int   = 2 ** 16        # 1 MB per sample per working array
SPWVD_MAX_CHUNK_BYTES: float = 1.5e9          # refuse a chunk above ~1.5 GB peak
SPWVD_NAN_GAP_S:       float = MouseECG.SPECTRAL_MAX_GAP_S   # 2 s
SPWVD_MAX_BRIDGED_FRAC: float = MouseECG.SPECTRAL_MAX_EXCLUDED_FRAC   # 5 %
SPWVD_MIN_VALID:       int   = 4              # valid intervals needed for the spline
SPWVD_MIN_SAMPLES:     int   = 8              # resampled samples per series / chunk core
SPWVD_MIN_SEGMENT_S:   float = 10.0           # shortest computed stretch (see _min_segment_s)
SPWVD_MARGIN_S:        float = 60.0           # default chunk margin (module docstring)

# ── Kernel constants hard-coded in compute_PWVD.m (not parameters there) ─
_BETA:  float = 1.0
_GAMMA: float = 1.0
_ALFA:  float = 0.0
_R:     float = 0.0

# All FFTs run through scipy.fft so they can work in place; the result of
# each 1-D transform does not depend on the worker count.
_FFT_WORKERS: int = -1

# Uncovered time (s) below which consecutive valid samples are contiguous
# (rounding of t_s against rr_ms / 1000), not a bridged gap.
_BRIDGE_TOL_S: float = 1e-6

# Kernel rows built per block (bounds the kernel's own memory to ~32 MB).
_KERNEL_BLOCK_BYTES: int = 32 * 1024 * 1024


# ── Validation helpers ───────────────────────────────────────────────────

def _check_fbins(fbins) -> int:
    """*fbins* as an int; ValueError unless it is a power of two from
    ``SPWVD_MIN_FBINS`` (256) to ``SPWVD_MAX_FBINS`` (65536) -- an integral
    float such as 4096.0 is accepted. The upper bound keeps the
    fbins-length vectors built before the memory guard (frequency axis,
    band masks) small; the guard then bounds the fbins × samples arrays."""
    integral = (isinstance(fbins, (int, np.integer))
                or (isinstance(fbins, (float, np.floating))
                    and math.isfinite(fbins) and float(fbins).is_integer()))
    if isinstance(fbins, (bool, np.bool_)) or not integral:
        raise ValueError(
            f"fbins must be an integer power of two from {SPWVD_MIN_FBINS} to "
            f"{SPWVD_MAX_FBINS}, got {fbins!r}.")
    fb = int(fbins)
    if fb < SPWVD_MIN_FBINS or fb > SPWVD_MAX_FBINS or fb & (fb - 1):
        if fb <= 0:
            why = "it must be positive"
        elif fb % 256:
            why = ("compute_PWVD.m transposes the ambiguity function onto the wrong "
                   "axes for an fbins that is not a multiple of 256")
        elif fb > SPWVD_MAX_FBINS:
            why = (f"above {SPWVD_MAX_FBINS} the fbins × samples working array needs "
                   f"gigabytes even for a short series")
        else:
            why = ("only powers of two are supported (compute_PWVD.m uses "
                   "Fbins = pow2(12) = 4096)")
        raise ValueError(
            f"fbins must be a power of two from {SPWVD_MIN_FBINS} to {SPWVD_MAX_FBINS} "
            f"(e.g. 256, 1024, 4096), got {fb}: {why}.")
    return fb


def _check_positive(name: str, value, allow_zero: bool = False) -> float:
    """*value* as a finite float > 0 (>= 0 with *allow_zero*), else ValueError."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from None
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {bound}, got {value!r}.")
    return v


def _band_bins(f: np.ndarray, band, name: str, fs: float) -> np.ndarray:
    """Indices of ``lo <= f <= hi`` (MATLAB ``find(f>=lo & f<=hi)``) for a
    band within [0, fs/2] that covers at least two bins."""
    try:
        lo, hi = (float(band[0]), float(band[1]))
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{name} band must be a (low, high) pair in Hz, got {band!r}.") from None
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi <= fs / 2.0):
        raise ValueError(
            f"{name} band must satisfy 0 <= low < high <= fs_resample/2 = {fs / 2.0:g} Hz, "
            f"got {lo:g}–{hi:g} Hz.")
    ix = np.flatnonzero((f >= lo) & (f <= hi))
    if len(ix) < 2:
        raise ValueError(
            f"{name} band {lo:g}–{hi:g} Hz covers {len(ix)} frequency bin(s); "
            f"widen it or increase fbins.")
    return ix


def _fade_samples(spans: list, margin: int) -> int:
    """Half-width (samples) of the cross-fade centred on each interior core
    boundary: half the margin, and at most half the shortest core so that
    the fades of neighbouring boundaries never overlap. 0 for one chunk."""
    if len(spans) < 2:
        return 0
    shortest = min(c1 - c0 for _, _, c0, c1 in spans)
    return int(min(margin // 2, shortest // 2))


def _stitch_weights(k: int, n_chunks: int, c0: int, c1: int, fade: int
                    ) -> "tuple[int, int, np.ndarray]":
    """Weights with which chunk *k* (core ``[c0, c1)``) enters the output:
    ``(w0, w1, w)`` covering samples ``[w0, w1)``. 1 inside the core, a
    linear ramp over ``[c0 - fade, c0 + fade)`` (up) and ``[c1 - fade,
    c1 + fade)`` (down) at interior boundaries; a neighbour's ramp is one
    minus this one's, so the weights sum to 1 at every sample. Both chunks
    of a boundary cover its fade: ``fade <= margin // 2``."""
    w0 = c0 - fade if k > 0 else c0
    w1 = c1 + fade if k < n_chunks - 1 else c1
    w = np.ones(w1 - w0, dtype=np.float64)
    if fade:
        ramp = (np.arange(2 * fade) + 0.5) / (2 * fade)
        if k > 0:
            w[:2 * fade] = ramp
        if k < n_chunks - 1:
            w[-2 * fade:] = 1.0 - ramp
    return w0, w1, w


def _chunk_spans(n: int, core: int, margin: int) -> "list[tuple[int, int, int, int]]":
    """Chunk plan of an *n*-sample series: ``(e0, e1, c0, c1)`` per chunk,
    the extended span ``[e0, e1)`` (core plus *margin* each side, clipped
    at the ends) and the core ``[c0, c1)``; the cores tile ``[0, n)``. One
    span covering everything when ``n <= core + 2 * margin``."""
    if n <= core + 2 * margin:
        return [(0, n, 0, n)]
    n_chunks = int(math.ceil(n / core))
    edges = np.round(np.linspace(0, n, n_chunks + 1)).astype(int)
    return [(max(0, int(c0) - margin), min(n, int(c1) + margin), int(c0), int(c1))
            for c0, c1 in zip(edges[:-1], edges[1:])]


def _min_segment_s(lf_lo: float) -> float:
    """Shortest stretch of valid RR (s) the SPWVD is computed on: two
    periods of the lowest LF frequency, and at least ``SPWVD_MIN_SEGMENT_S``
    (10 s: at 20 Hz the lag kernel spans ~12 s, so a shorter stretch cannot
    fill it). 5 s -> 10 s with the mouse LF floor 0.4 Hz, 20 s with the rat
    0.1 Hz."""
    two_periods = 2.0 / lf_lo if lf_lo > 0 else 0.0
    return max(SPWVD_MIN_SEGMENT_S, two_periods)


def _segments(tv: np.ndarray, tg: np.ndarray, long_gaps: np.ndarray,
              min_span_s: float = 0.0) -> "list[tuple[int, int, int, int, bool]]":
    """Split the valid samples *tv* at every gap in *long_gaps* (index j:
    the gap between tv[j] and tv[j + 1]). Returns ``(a, b, i0, i1, ok)``
    per segment: valid samples ``[a, b)``, output-grid samples ``[i0, i1)``
    from the first grid point at or after tv[a] to the last at or before
    tv[b - 1] (the first segment starts at 0 and the last ends at
    ``len(tg)``, the grid being built on the whole valid span), and whether
    it can be computed: enough intervals and samples, and a span
    ``tv[b - 1] - tv[a]`` of at least *min_span_s*. Without a long gap this
    is the single segment ``(0, len(tv), 0, len(tg), ok)``."""
    n, nv = len(tg), len(tv)
    cuts = np.concatenate(([0], np.asarray(long_gaps, dtype=int) + 1, [nv]))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        a, b = int(a), int(b)
        i0 = 0 if a == 0 else int(np.searchsorted(tg, tv[a], side="left"))
        i1 = n if b == nv else int(np.searchsorted(tg, tv[b - 1], side="right"))
        ok = (b - a >= SPWVD_MIN_VALID and i1 - i0 >= SPWVD_MIN_SAMPLES
              and float(tv[b - 1] - tv[a]) >= float(min_span_s) - 1e-9)
        out.append((a, b, i0, max(i0, i1), ok))
    return out


def _bridged_report(tv: np.ndarray, uncovered: np.ndarray, long_gaps: np.ndarray,
                    segments: list) -> dict:
    """The gaps the spline bridges -- uncovered time > ``_BRIDGE_TOL_S``
    between consecutive valid samples of a computed segment (the long gaps
    split segments and are never bridged) -- and the share of the computed
    segments' span they cover. ``withheld_reason`` is set when that share
    exceeds ``SPWVD_MAX_BRIDGED_FRAC``."""
    inner = np.zeros(len(uncovered), dtype=bool)
    span_s = 0.0
    for a, b, _, _, ok in segments:
        if ok:
            inner[a:b - 1] = True                    # gaps between tv[a] .. tv[b-1]
            span_s += float(tv[b - 1] - tv[a])
    inner[np.asarray(long_gaps, dtype=int)] = False
    gaps = uncovered[inner & (uncovered > _BRIDGE_TOL_S)]
    bridged_s = float(gaps.sum()) if len(gaps) else 0.0
    frac = bridged_s / span_s if span_s > 0 else 0.0
    reason = None
    if frac > SPWVD_MAX_BRIDGED_FRAC:
        reason = (f"excluded RR intervals bridged by the spline cover {100.0 * frac:.1f} % "
                  f"of the analysed time (limit {100.0 * SPWVD_MAX_BRIDGED_FRAC:g} %)")
    return {
        "n_bridged_gaps":    int(len(gaps)),
        "longest_bridged_s": float(gaps.max()) if len(gaps) else 0.0,
        "bridged_s":         bridged_s,
        "bridged_frac":      frac,
        "withheld_reason":   reason,
    }


def _peak_bytes(n_samples: int, fbins: int, n_rows: int, n_series: int = 0) -> int:
    """Estimated peak working memory (bytes) of one SPWVD chunk of
    *n_samples*: the complex (fbins × n) working array, the complex band
    rows carried to the last transform and one kernel block, plus the
    complex analytic signal of the whole *n_series*-sample series that is
    kept across chunks."""
    return int(16 * n_samples * (fbins + n_rows) + 16 * n_series + _KERNEL_BLOCK_BYTES)


# ── Core transform ───────────────────────────────────────────────────────

def _kernel_coordinates(n_lag: int, n_dopp: int, v0: float, tau0: float
                        ) -> "tuple[np.ndarray, np.ndarray]":
    """``tau1.^2`` (lag) and ``v1.^2`` (Doppler) of compute_PWVD.m, each
    already fftshift-ed, as 1-D vectors (the kernel is a function of the
    two, so this equals building the meshgrid and fftshift-ing the kernel).

    ``n_lag`` is MATLAB's Ccol (= Fbins), ``n_dopp`` its Crow (samples)."""
    dy = (np.arange(n_lag) - n_lag // 2) / (n_lag / 2)        # (-Ccol/2:Ccol/2-1)/(Ccol/2)
    if n_dopp % 2 == 0:                                       # Crow/2 - fix(Crow/2) == 0
        dx = (np.arange(n_dopp) - n_dopp // 2) / (n_dopp / 2)  # (-Crow/2:Crow/2-1)/(Crow/2)
    else:                                                     # (-Crow/2-1/2:Crow/2-3/2)/(Crow/2)
        dx = (np.arange(n_dopp) - (n_dopp + 1) // 2) / (n_dopp / 2)
    tau1 = np.fft.fftshift(dy / tau0)
    v1 = np.fft.fftshift(dx / v0)
    return tau1 * tau1, v1 * v1


def _kernel_from_mu_sq(tau1_sq: np.ndarray, v1_sq: np.ndarray, lam: float) -> np.ndarray:
    """``exp(-pi*(mu.^2).^lambda)`` on the outer grid of *tau1_sq* (rows) and
    *v1_sq* (columns), with mu reduced from MATLAB's general form using its
    hard-coded constants (alfa = 0, r = 0: mu = tau1² + v1², bit-identical
    to the literal expression since x * 1 and x + 0 are exact)."""
    assert _ALFA == 0.0 and _R == 0.0, "mu reduction assumes alfa = r = 0"
    k = np.add.outer(tau1_sq, v1_sq)
    np.square(k, out=k)
    np.power(k, lam, out=k)
    np.multiply(k, -np.pi, out=k)
    np.exp(k, out=k)
    return k


def _analytic_signal(x: np.ndarray) -> np.ndarray:
    """``hilbert(ibi_int - mean(ibi_int))`` with ``ibi_int = detrend(ibi_int)``
    (compute_PWVD.m lines 5 and 13): the analytic signal of the linearly
    detrended, demeaned evenly sampled series *x*."""
    xd = _detrend(np.asarray(x, dtype=np.float64), type="linear")
    return _hilbert(xd - xd.mean())


def _spwvd_band_rows(z: np.ndarray, fbins: int, v0: float, tau0: float,
                     lam: float, rows: np.ndarray) -> np.ndarray:
    """Rows *rows* of compute_PWVD.m's ``TFR_RR`` for the analytic signal
    *z* (``_analytic_signal`` of the WHOLE series, or a slice of it when
    chunking -- the detrend / Hilbert step is not redone per chunk).

    Returns a real (len(rows), len(z)) array. The working array keeps the
    (lag, time) orientation of MATLAB's ``wvx`` throughout; every MATLAB
    transform is applied along the axis its column-wise ``fft`` would use.
    """
    z = np.asarray(z, dtype=np.complex128)
    n = len(z)
    half = fbins // 2                                # round(Fbins/2), fbins even
    zc = np.conj(z)

    # ── Wigner-Ville lag products (tfrwv loop, vectorised over time) ──
    # Column t (0-based) holds lags |tau| <= min(t, n-1-t, half-1); for a
    # given lag those are exactly the columns tau..n-1-tau.
    w = np.zeros((fbins, n), dtype=np.complex128)
    w[0] = z * zc
    for tau in range(1, min((n - 1) // 2, half - 1) + 1):
        w[tau, tau:n - tau] = z[2 * tau:] * zc[:n - 2 * tau]            # z(t+tau) conj z(t-tau)
        w[fbins - tau, tau:n - tau] = z[:n - 2 * tau] * zc[2 * tau:]    # tau < 0 row
    if n >= 2 * half + 1:                            # half <= t <= n-1-half
        w[half, half:n - half] = 0.5 * (z[2 * half:] * zc[:n - 2 * half]
                                        + z[:n - 2 * half] * zc[2 * half:])

    # wvx = fft(wvx); real(wvx)  (xcol == 1)       -- lag -> frequency
    w = _sfft.fft(w, axis=0, overwrite_x=True, workers=_FFT_WORKERS)
    w.imag[...] = 0.0
    # AF = fft(ifft(wvx).')                         -- frequency -> lag, time -> Doppler
    w = _sfft.ifft(w, axis=0, overwrite_x=True, workers=_FFT_WORKERS)
    w = _sfft.fft(w, axis=1, overwrite_x=True, workers=_FFT_WORKERS)
    # rem(Fbins, 256) == 0: the conditional AF.' is not taken (module doc, step 5).

    # ── Kernel: AF .* fftshift(t_wind), in row blocks ──
    tau1_sq, v1_sq = _kernel_coordinates(fbins, n, v0, tau0)
    # max(max(t_wind)): the kernel decreases with mu, so its maximum is the
    # value at the smallest mu (0 here: both coordinate grids contain 0).
    k_max = float(_kernel_from_mu_sq(np.array([tau1_sq.min()]),
                                     np.array([v1_sq.min()]), lam)[0, 0])
    block = max(1, _KERNEL_BLOCK_BYTES // (8 * n))
    for j0 in range(0, fbins, block):
        j1 = min(fbins, j0 + block)
        k = _kernel_from_mu_sq(tau1_sq[j0:j1], v1_sq, lam)
        k /= k_max
        w[j0:j1] *= k
    del k

    # TFR_RR = real(ifft(fft(Asmooth).').')         -- lag -> frequency, Doppler -> time
    w = _sfft.fft(w, axis=0, overwrite_x=True, workers=_FFT_WORKERS)
    band = w[rows]                                   # copy of the band rows only
    del w
    band = _sfft.ifft(band, axis=1, overwrite_x=True, workers=_FFT_WORKERS)
    return np.ascontiguousarray(band.real)


# ── Public API ───────────────────────────────────────────────────────────

def spwvd_lf_hf(
    rr_ms: np.ndarray,
    t_s: np.ndarray,
    valid: "Optional[np.ndarray]" = None,
    fs_resample: float = SPWVD_FS_HZ,
    lf_band: "tuple[float, float]" = MouseECG.LF,
    hf_band: "tuple[float, float]" = MouseECG.HF,
    fbins: int = SPWVD_FBINS,
    v0: float = 0.03,
    tau0: float = 0.06,
    lam: float = 0.3,
    chunk_s: float = 120.0,
    margin_s: float = SPWVD_MARGIN_S,
) -> dict:
    """Time-resolved LF and HF power of an RR series by the smoothed pseudo
    Wigner-Ville distribution -- a port of robust_hrv's ``compute_PWVD.m``
    (see the module docstring for the statement-by-statement mapping, the
    units, the chunking and the gap handling).

    Parameters
    ----------
    rr_ms, t_s : every RR interval (ms) and the time of its END (s), e.g.
        ``rr_interval_series()["rr_ms"]`` / ``["t_s"]``. Sorted by *t_s*
        here if needed.
    valid : bool mask over *rr_ms* (``rr_interval_series()["valid"]``,
        which honours the user's RR bounds and removed-beat breaks). None:
        intervals outside [RR_MIN_MS, RR_MAX_MS] are invalid. Non-finite
        intervals or times are always invalid. Invalid intervals are
        skipped before the spline; an uncovered stretch longer than
        ``SPWVD_NAN_GAP_S`` splits the series into segments computed
        separately (NaN between them), shorter ones are bridged.
    fs_resample : even resampling rate, Hz (Frr; 20 Hz as in the paper).
    lf_band, hf_band : (low, high) Hz, inclusive; both within
        [0, fs_resample/2] and covering >= 2 frequency bins. Defaults are
        the mouse bands MouseECG.LF / HF.
    fbins : frequency bins / lag length, a power of two from 256 to
        ``SPWVD_MAX_FBINS`` = 65536 (ValueError otherwise). 4096 as in the
        MATLAB code. The lag kernel's extent in
        seconds scales with fbins / fs_resample, so changing it changes the
        frequency resolution, not just the grid.
    v0, tau0, lam : Doppler / lag kernel widths and the kernel exponent
        (v0 = 0.03, tau0 = 0.06, lambda = 0.3: the constants hard-coded in
        compute_PWVD.m, the toolbox's only set; defined in normalised
        units, so their extent in s / Hz depends on fs_resample and fbins;
        not validated in mice -- module docstring).
    chunk_s, margin_s : core length and per-side margin of the overlapping
        chunks used when a segment's resampled series is longer than
        ``chunk_s + 2 * margin_s`` seconds (the detrend / Hilbert step still
        runs once on the whole segment; cores are cross-faded over
        *margin_s* at their boundaries). Default margin 60 s
        (``SPWVD_MARGIN_S``; see the module docstring for the measured
        accuracy). A *chunk_s* longer than the recording gives the literal
        single pass of compute_PWVD.m.

    Returns
    -------
    dict with
        t      : output time grid (s), first valid sample + k / fs_resample
        lf, hf : band power at each t (trapz over the band bins with unit
                 spacing, as MATLAB ``trapz(Y)``), NaN inside long gaps and
                 over segments too short to compute
        params : dict -- the parameters used (fs_resample, lf_band, hf_band,
                 fbins, v0, tau0, lam, chunk_s, margin_s), the band edges
                 actually covered (lf_band_bins / hf_band_bins, Hz) and their
                 bin counts (n_lf_bins / n_hf_bins), chunked / n_chunks,
                 crossfade_s (total width of each core-boundary cross-fade,
                 0 for a single pass), n_samples, n_intervals, n_valid,
                 n_excluded, longest_gap_s (longest uncovered time between
                 valid samples), nan_gap_s, n_gaps_masked (long gaps that
                 split the series), n_segments / n_segments_computed /
                 n_segments_too_short, min_segment_s, n_nan_samples, the
                 bridged-gap report (n_bridged_gaps, longest_bridged_s,
                 bridged_s, bridged_frac, max_bridged_frac),
                 peak_bytes_estimate, units, method.

    Raises
    ------
    ValueError for invalid parameters (fbins, bands, rates, kernel widths,
    chunk lengths), mismatched input lengths, fewer than
    ``SPWVD_MIN_VALID`` valid intervals or ``SPWVD_MIN_SAMPLES`` resampled
    samples, duplicate valid times, every segment between long gaps too
    short to compute (fewer than ``SPWVD_MIN_VALID`` intervals,
    ``SPWVD_MIN_SAMPLES`` samples or ``min_segment_s`` of span), bridged
    short gaps covering more than ``SPWVD_MAX_BRIDGED_FRAC`` of the
    computed segments' time, or a chunk whose estimated peak memory
    exceeds ``SPWVD_MAX_CHUNK_BYTES`` (checked before the spline and before
    any fbins × samples array is allocated; only the inputs and
    fbins-length vectors, at most 512 kB, exist by then).
    """
    fbins = _check_fbins(fbins)
    fs = _check_positive("fs_resample", fs_resample)
    v0 = _check_positive("v0", v0)
    tau0 = _check_positive("tau0", tau0)
    lam = _check_positive("lam", lam)
    chunk_s = _check_positive("chunk_s", chunk_s)
    margin_s = _check_positive("margin_s", margin_s, allow_zero=True)

    f = np.linspace(0.0, fs / 2.0, fbins)            # f = linspace(0, Frr/2, Fbins)
    lf_ix = _band_bins(f, lf_band, "LF", fs)
    hf_ix = _band_bins(f, hf_band, "HF", fs)
    rows = np.union1d(lf_ix, hf_ix)
    lf_pos = np.searchsorted(rows, lf_ix)
    hf_pos = np.searchsorted(rows, hf_ix)

    core = int(round(chunk_s * fs))
    margin = int(round(margin_s * fs))
    if core < 2 * SPWVD_MIN_SAMPLES:
        raise ValueError(
            f"chunk_s = {chunk_s:g} s is {core} sample(s) at {fs:g} Hz; "
            f"it must cover at least {2 * SPWVD_MIN_SAMPLES} samples.")

    # ── Valid intervals ──
    rr = np.asarray(rr_ms, dtype=np.float64).flatten()
    t = np.asarray(t_s, dtype=np.float64).flatten()
    if len(rr) != len(t):
        raise ValueError(f"rr_ms and t_s lengths differ ({len(rr)} vs {len(t)}).")
    if valid is None:
        ok = rr_valid_mask(rr)
    else:
        ok = np.asarray(valid, dtype=bool).flatten()
        if len(ok) != len(rr):
            raise ValueError(f"valid has {len(ok)} entries for {len(rr)} intervals.")
        ok = ok & np.isfinite(rr)
    ok = ok & np.isfinite(t)
    n_total = int(len(rr))
    n_valid = int(ok.sum())
    if n_valid < SPWVD_MIN_VALID:
        raise ValueError(
            f"SPWVD needs at least {SPWVD_MIN_VALID} valid RR intervals, got {n_valid}.")
    order = np.argsort(t[ok], kind="stable")
    tv = t[ok][order]
    rv = rr[ok][order]
    if np.any(np.diff(tv) <= 0.0):
        raise ValueError("Valid RR intervals must have distinct end times.")

    # t_temp = t_IBI(1) : 1/Frr : t_IBI(end) -- one output grid for the
    # whole series; every segment is evaluated on its own stretch of it.
    n = int(math.floor((tv[-1] - tv[0]) * fs + 1e-9)) + 1
    if n < SPWVD_MIN_SAMPLES:
        raise ValueError(
            f"The valid RR series spans {tv[-1] - tv[0]:.3g} s: {n} sample(s) at "
            f"{fs:g} Hz, fewer than the {SPWVD_MIN_SAMPLES} required.")
    tg = tv[0] + np.arange(n) / fs

    # ── Gaps: uncovered time between consecutive valid samples ──
    uncovered = np.diff(tv) - rv[1:] / 1000.0
    longest_gap = float(max(0.0, uncovered.max())) if len(uncovered) else 0.0
    long_gaps = np.flatnonzero(uncovered > SPWVD_NAN_GAP_S)
    min_seg_s = _min_segment_s(float(lf_band[0]))
    segments = _segments(tv, tg, long_gaps, min_seg_s)
    computed = [s for s in segments if s[4]]
    if not computed:
        where = (f"Every stretch between RR gaps longer than {SPWVD_NAN_GAP_S:g} s is"
                 if len(long_gaps) else "The valid RR series is")
        raise ValueError(
            f"{where} too short for the SPWVD (it needs {SPWVD_MIN_VALID} valid "
            f"intervals, {SPWVD_MIN_SAMPLES} samples at {fs:g} Hz and {min_seg_s:g} s "
            f"of continuous valid RR -- two periods of the {float(lf_band[0]):g} Hz LF "
            f"floor and the ~12 s lag kernel).")
    bridged = _bridged_report(tv, uncovered, long_gaps, segments)
    if bridged["withheld_reason"]:
        raise ValueError(bridged["withheld_reason"])

    # ── Chunk plans and memory guard (before any fbins × samples allocation) ──
    plans = [_chunk_spans(i1 - i0, core, margin) for _, _, i0, i1, _ in computed]
    need, widest, n_seg = 0, 0, 0
    for (_, _, i0, i1, _), spans in zip(computed, plans):
        w_seg = max(e1 - e0 for e0, e1, _, _ in spans)
        b_seg = _peak_bytes(w_seg, fbins, len(rows), i1 - i0)
        if b_seg > need:
            need, widest, n_seg = b_seg, w_seg, i1 - i0
    if need > SPWVD_MAX_CHUNK_BYTES:
        raise ValueError(
            f"SPWVD chunk of {widest} samples × {fbins} bins (with the {n_seg}-sample "
            f"analytic signal of the series) needs ~{need / 1e9:.1f} GB of memory "
            f"(limit {SPWVD_MAX_CHUNK_BYTES / 1e9:.1f} GB); reduce chunk_s / margin_s "
            f"or fbins.")

    # ── Each segment as its own series (spline, detrend, Hilbert, SPWVD) ──
    lf = np.full(n, np.nan, dtype=np.float64)
    hf = np.full(n, np.nan, dtype=np.float64)
    fade_max = 0
    for (a, b, i0, i1, _), spans in zip(computed, plans):
        x = CubicSpline(tv[a:b], rv[a:b], bc_type="not-a-knot")(tg[i0:i1])  # interp1(..., 'spline')
        # detrend + hilbert(x - mean(x)) ONCE over the whole segment, as in
        # compute_PWVD.m over its series; chunks only slice it (module
        # docstring, chunking).
        z = _analytic_signal(x)
        del x
        # SPWVD per chunk, cores stitched with cross-fades
        m = i1 - i0
        fade = _fade_samples(spans, margin)
        fade_max = max(fade_max, fade)
        lf_s = np.zeros(m, dtype=np.float64)
        hf_s = np.zeros(m, dtype=np.float64)
        for k, (e0, e1, c0, c1) in enumerate(spans):
            tfr = _spwvd_band_rows(z[e0:e1], fbins, v0, tau0, lam, rows)
            w0, w1, w = _stitch_weights(k, len(spans), c0, c1, fade)
            lf_s[w0:w1] += w * _trapz(tfr[lf_pos], axis=0)[w0 - e0:w1 - e0]
            hf_s[w0:w1] += w * _trapz(tfr[hf_pos], axis=0)[w0 - e0:w1 - e0]
            del tfr
        lf[i0:i1] = lf_s
        hf[i0:i1] = hf_s
        del z
    nan_mask = np.isnan(lf)

    params = {
        "method":              "SPWVD (port of robust_hrv compute_PWVD.m, Orini 2012 kernel)",
        "fs_resample":         fs,
        "lf_band":             (float(lf_band[0]), float(lf_band[1])),
        "hf_band":             (float(hf_band[0]), float(hf_band[1])),
        "lf_band_bins":        (float(f[lf_ix[0]]), float(f[lf_ix[-1]])),
        "hf_band_bins":        (float(f[hf_ix[0]]), float(f[hf_ix[-1]])),
        "n_lf_bins":           int(len(lf_ix)),
        "n_hf_bins":           int(len(hf_ix)),
        "fbins":               fbins,
        "v0":                  v0,
        "tau0":                tau0,
        "lam":                 lam,
        "chunk_s":             chunk_s,
        "margin_s":            margin_s,
        "chunked":             any(len(spans) > 1 for spans in plans),
        "n_chunks":            int(sum(len(spans) for spans in plans)),
        "crossfade_s":         2 * fade_max / fs,
        "n_samples":           int(n),
        "n_intervals":         n_total,
        "n_valid":             n_valid,
        "n_excluded":          n_total - n_valid,
        "longest_gap_s":       longest_gap,
        "nan_gap_s":           float(SPWVD_NAN_GAP_S),
        "n_gaps_masked":       int(len(long_gaps)),
        "n_segments":          int(len(segments)),
        "n_segments_computed": int(len(computed)),
        "n_segments_too_short": int(len(segments) - len(computed)),
        "min_segment_s":       float(min_seg_s),
        "n_nan_samples":       int(nan_mask.sum()),
        "n_bridged_gaps":      bridged["n_bridged_gaps"],
        "longest_bridged_s":   bridged["longest_bridged_s"],
        "bridged_s":           bridged["bridged_s"],
        "bridged_frac":        bridged["bridged_frac"],
        "max_bridged_frac":    float(SPWVD_MAX_BRIDGED_FRAC),
        "peak_bytes_estimate": int(need),
        "units":               "ms² (unnormalised TFR summed over bins; arbitrary scale)",
    }
    return {"t": tg, "lf": lf, "hf": hf, "params": params}
