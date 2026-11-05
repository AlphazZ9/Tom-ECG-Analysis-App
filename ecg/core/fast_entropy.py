# -*- coding: utf-8 -*-
"""
ecg.core.fast_entropy
---------------------
A faster neighbour count for NeuroKit2's entropy metrics (ApEn, SampEn, MSE,
CMSE, RCMSE...), the same integers as NeuroKit's own.

NeuroKit counts, for each embedded vector, the vectors within the tolerance
under the Chebyshev distance with a scikit-learn KDTree built on every call,
after parsing scikit-learn's version on every call too. Multiscale entropy
makes tens of thousands of such calls on short series, so the per-call cost
is most of non-linear HRV's run time. Here the count is a direct comparison,
max_k |x_ik - x_jk| <= tolerance -- the distance and the test the KDTree
applies to each candidate pair -- in row blocks.

Only the plain Chebyshev count is replaced (no fuzzy membership, no other
distance, no precomputed tree); everything else goes to NeuroKit's own
function. ``install()`` checks the replacement against NeuroKit's count on a
few series first and leaves NeuroKit untouched if any count differs.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("ecg")

_BLOCK = 4_000_000          # pair comparisons per block (memory bound)
_LOCK = threading.Lock()
_STATE = {"installed": None}     # None: not tried; True / False: result


def chebyshev_counts(embedded: "np.ndarray", tolerance: float) -> "np.ndarray":
    """For each row of *embedded*, how many rows (itself included) lie within
    *tolerance* under the Chebyshev distance, as float64 like NeuroKit's."""
    e = np.ascontiguousarray(embedded, dtype=np.float64)
    n = e.shape[0]
    counts = np.empty(n, dtype=np.float64)
    if n == 0:
        return counts
    rows = max(1, _BLOCK // max(1, n))
    for a in range(0, n, rows):
        b = min(n, a + rows)
        d = np.abs(e[a:b, None, 0] - e[None, :, 0])
        for k in range(1, e.shape[1]):
            np.maximum(d, np.abs(e[a:b, None, k] - e[None, :, k]), out=d)
        counts[a:b] = np.count_nonzero(d <= tolerance, axis=1)
    return counts


def _make_fast(original):
    from neurokit2.complexity.utils_complexity_embedding import complexity_embedding

    def _get_count(signal, delay=1, dimension=2, tolerance=0.1, distance="chebyshev",
                   approximate=True, fuzzy=False, kdtree=None, n=1, **kwargs):
        if distance != "chebyshev" or fuzzy or kdtree is not None or n != 1 or kwargs:
            return original(signal, delay=delay, dimension=dimension, tolerance=tolerance,
                            distance=distance, approximate=approximate, fuzzy=fuzzy,
                            kdtree=kdtree, n=n, **kwargs)
        embedded = complexity_embedding(signal, delay=delay, dimension=dimension)
        if approximate is False:
            embedded = embedded[:-1]
        return embedded, chebyshev_counts(embedded, tolerance), None

    _get_count.__wrapped__ = original      # type: ignore[attr-defined]
    return _get_count


def _self_check(original, fast) -> bool:
    """Same counts as NeuroKit's on a few series: RR-like, with ties, short."""
    rng = np.random.default_rng(0)
    series = [170 + 3 * rng.standard_normal(600), np.round(170 + 3 * rng.standard_normal(400)),
              rng.standard_normal(50), np.repeat([1.0, 2.0, 3.0], 30)]
    for x in series:
        tol = 0.2 * float(np.std(x, ddof=1))
        for dim in (1, 2, 3):
            for approx in (True, False):
                _, c0, _ = original(x, delay=1, dimension=dim, tolerance=tol,
                                    approximate=approx)
                _, c1, _ = fast(x, delay=1, dimension=dim, tolerance=tol, approximate=approx)
                if c0.shape != c1.shape or not np.array_equal(c0, c1):
                    return False
    return True


def install() -> bool:
    """Swap NeuroKit's count for chebyshev_counts, once; False (NeuroKit
    unchanged) if NeuroKit's internals are not where expected or a count of
    the self-check differs."""
    with _LOCK:
        if _STATE["installed"] is not None:
            return bool(_STATE["installed"])
        try:
            from neurokit2.complexity import entropy_approximate, utils_entropy
            original = utils_entropy._get_count
            fast = _make_fast(original)
            ok = _self_check(original, fast)
        except Exception as exc:
            log.debug("fast entropy count not installed: %s", exc)
            ok = False
        if ok:
            utils_entropy._get_count = fast
            entropy_approximate._get_count = fast
        else:
            log.info("fast entropy count not installed: NeuroKit's own count is used")
        _STATE["installed"] = ok
        return ok
