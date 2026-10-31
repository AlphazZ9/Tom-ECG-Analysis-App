"""
ecg.core.detection
──────────────────
R-peak detection, RR artifact detection/correction, and arrhythmia
classification.  No UI imports — pure NumPy / SciPy.

Trois stratégies de détection :
  1. detect_peaks_wavelet        — CWT (chapeau mexicain)
  2. detect_peaks_sg_derivative  — Savitzky-Golay + première dérivée
  3. detect_peaks_envelope_max   — Enveloppe locale / maximum absolu
                                   Idéal pour les signaux saturés (ADC clipping)
                                   et les morphologies atypiques où la dérivée
                                   est peu discriminante.

Utilitaires partagés
────────────────────
  • fix_polarity              — vote multi-fenêtres + skewness + kurtosis fallback
                                (nouveau : paramètre force_polarity pour raw display)
  • apply_threshold            — seuil prominence adaptatif
  • _adaptive_j_upstroke_ratio — seuil J/R adaptatif selon SNR et CV des upstrokes
  • resolve_r_vs_j_peaks       — disambiguation R vs J (upstroke, multi-pass doublets)
  • detect_rr_artifacts        — doublets + bornes physio + ectopique
  • apply_artifact_decisions   — applique accept/remove
  • correct_rr_artifacts       — auto-correct
  • active_removed_samples     — battements retirés encore absents (ruptures HRV)
  • merge_artifact_reports     — rapport cumulatif (passes successives)
  • drop_removed_peaks         — ré-applique les retraits (restauration de session)
  • recover_missed_beats        — battements manqués
  • classify_arrhythmias        — règle-basée
  • arrhythmia_baseline_window  — fenêtre de référence de classify_arrhythmias (affichage)
  • arrhythmia_baseline_info    — idem + valeur de référence et fiabilité
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Iterable, Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_prominences, peak_widths

from ecg.core.filtering import downsample_signal
from ecg.core.models import (
    PREMATURE_DEFINITION_TEXT,
    PREMATURE_KINDS,
    ArrhythmiaEvent,
    MouseECG,
)

log = logging.getLogger("ecg")


# ─── helper partagé ────────────────────────────────────────────────────────────

def _prominence_wlen_samples(fs: float) -> int:
    """`wlen` (samples) bounding every prominence/width search below.

    Without it, find_peaks(..., prominence=X)/peak_prominences/peak_widths
    search the entire array outward from each peak for a taller point --
    see MouseECG.PROMINENCE_WLEN_MS for why this is a real, observed
    catastrophic-slowdown risk on raw (unfiltered) signal, not just a
    theoretical one.
    """
    return max(3, int(round(MouseECG.PROMINENCE_WLEN_MS / 1000.0 * fs)))


def _detect_baseline_wander(signal: np.ndarray, fs: float) -> bool:
    """Détecte une dérive de ligne de base significative (respiration, mouvement).

    Ratio P_resp[0.5–5 Hz] / P_QRS[10–100 Hz] via Welch PSD.
    Seuil 0.5 : large marge (signal avec resp. sévère → ratio~154 ; signal
    propre → ratio~0).

    Returns True si dérive significative détectée.
    """
    from scipy.signal import welch as _welch

    if fs <= 0 or len(signal) < int(fs * 2):
        return False

    nperseg = min(1024, len(signal) // 4)
    f, psd = _welch(signal, fs, nperseg=nperseg)

    def _band(lo: float, hi: float) -> float:
        m = (f >= lo) & (f <= hi)
        x = f[m]; y = psd[m]
        return float(np.trapezoid(y, x)) if len(x) > 1 else 0.0

    resp_band_hi = min(5.0, fs / 2.0 - 1.0)
    qrs_band_hi  = min(100.0, fs / 2.0 - 1.0)
    p_resp = _band(0.5, resp_band_hi)
    p_qrs  = _band(10.0, qrs_band_hi)
    ratio  = p_resp / (p_qrs + 1e-12)
    wander = ratio > 0.5

    log.debug(
        "_detect_baseline_wander: p_resp=%.4f  p_qrs=%.4f  ratio=%.2f → %s",
        p_resp, p_qrs, ratio, "WANDER" if wander else "clean",
    )
    return wander


def _hp_filter_for_deriv(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float = 3.0,
    order: int = 2,
) -> np.ndarray:
    """Filtre passe-haut léger utilisé UNIQUEMENT pour le calcul de la dérivée SG.

    3 Hz : supprime la respiration (1–3 Hz, > 30 dB) sans affecter le QRS
    (énergie principale > 10 Hz).  Ordre 2 Butterworth = pas de ringing.
    Le signal retourné n'est utilisé que pour la dérivée — jamais pour le
    snap, le post-filter ou les prominences (signal_orig intact).
    """
    from scipy.signal import butter as _butter, filtfilt as _filtfilt

    nyq = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return signal

    b, a = _butter(order, cutoff_hz / nyq, btype="high")
    return _filtfilt(b, a, signal)


def _upstroke_slope_at(pk: int, signal: np.ndarray, fs: float,
                        window_ms: float = 3.0,
                        deriv: Optional[np.ndarray] = None) -> float:
    """Max de la dérivée discrète sur les *window_ms* ms précédant pk.

    Critère primaire de discrimination R vs P/J : la dépolarisation
    ventriculaire murine produit un front montant très abrupt (1–3 ms)
    nettement supérieur à celui de l'onde P (≈20–40 ms) ou de la J-wave.
    """
    win = max(2, int(window_ms / 1000.0 * fs))
    # If a precomputed derivative is provided (e.g. Savitzky-Golay deriv), use
    # it directly to avoid recomputing differences repeatedly.
    if deriv is not None:
        lo = max(0, pk - win)
        hi = min(len(deriv), pk + 1)
        if lo >= hi:
            return 0.0
        return float(np.max(deriv[lo:hi]))

    seg = signal[max(0, pk - win):pk + 1].astype(np.float64)
    d   = np.diff(seg)
    return float(d.max()) if len(d) else 0.0


def _topographic_prominences(signal: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Genuine topographic prominence at each peak (height above the higher
    of its two neighbouring valleys), via ``scipy.signal.peak_prominences``.

    Used so all four detection strategies (auto/wavelet/SG-derivative/
    envelope-max) report the same physical quantity as "prominence" — the
    quantity ``apply_threshold``'s single Threshold slider actually filters
    on. Previously, three of the four detectors substituted raw signal
    amplitude at the peak (``signal[peaks]``) for prominence — a much
    weaker discriminator against T-waves and baseline drift, and one that
    silently changed what the same slider value meant depending on which
    detector was active. Cheap: O(n) in practice given a bounded `wlen` --
    see MouseECG.PROMINENCE_WLEN_MS; unbounded, this is a real catastrophic-
    slowdown risk on raw (unfiltered) signal, not just a theoretical one.
    """
    if len(peaks) == 0:
        return np.array([])
    peaks_sorted = np.sort(np.asarray(peaks, dtype=int))
    try:
        proms, _, _ = peak_prominences(signal, peaks_sorted, wlen=_prominence_wlen_samples(fs))
    except Exception as exc:
        log.debug("_topographic_prominences: peak_prominences failed (%s) "
                   "— falling back to raw amplitude", exc)
        return signal[peaks_sorted]
    # Re-order to match the caller's original (possibly unsorted) peaks array.
    order = np.argsort(np.argsort(peaks))
    return proms[order]


def _snap_to_r(
    approx: int,
    signal: np.ndarray,
    fs: float,
    pre_snap: int,
    post_snap: Optional[int] = None,
    deriv: Optional[np.ndarray] = None,
) -> int:
    """Snap an approximate sample index to the best R-peak inside a window.

    Strategy:
    - Search in [approx-half_snap, approx+half_snap] for local maxima (find_peaks).
    - If multiple candidates found, restrict to those with amplitude ≥ 50% of
      the window's max, then pick the one with the largest upstroke slope
      (uses _upstroke_slope_at). Fallback to amplitude if slopes are equal-ish.
    - If no local maxima are detected, fall back to the maximum on the left
      half of the window (prefer earlier R-like position).

    Returns an integer sample index clipped to [0, len(signal)-1].
    """
    n = len(signal)
    approx_i = int(round(approx))
    post = pre_snap if post_snap is None else int(post_snap)
    lo = max(0, approx_i - int(pre_snap))
    hi = min(n, approx_i + int(post) + 1)

    if lo >= hi:
        return int(np.clip(approx_i, 0, n - 1))

    seg = signal[lo:hi]
    local_rel, _ = find_peaks(seg, prominence=0)
    if len(local_rel) == 0:
        # No clear local max: prefer the max in the left half (avoid selecting a
        # possible J-wave that tends to be after the true R). If left half is
        # empty, use global max in window.
        mid = lo + max(0, (approx_i - lo))
        left_lo = lo
        left_hi = min(hi, approx_i + 1)
        if left_hi > left_lo:
            rel = int(np.argmax(signal[left_lo:left_hi]))
            return int(left_lo + rel)
        # fallback
        return int(np.clip(approx_i, 0, n - 1))

    local_abs = local_rel + lo
    if len(local_abs) == 1:
        return int(local_abs[0])

    # Ne départager par pente montante qu'entre candidats dont l'amplitude
    # est compétitive avec le maximum du voisinage (≥ 50 %). Sans ce garde-
    # fou, un minuscule pic de bruit localement "raide" (pente calculée sur
    # 1-2 échantillons) peut battre le vrai R, dont la pente — bien que
    # nettement plus significative en valeur absolue — est mesurée sur une
    # fenêtre physiologique plus large et peut numériquement lui être
    # inférieure. Ce risque croît avec la largeur de fenêtre de recherche
    # (donc avec fs, à snap_max_ms fixe) : plus d'échantillons scannés, plus
    # de chances de croiser un maximum local non-cardiaque. Vérifié
    # empiriquement : sans ce filtre, jusqu'à ~25 % des vrais R-peaks
    # pouvaient être ré-aiguillés vers un point quasi-basal (amplitude
    # <20 % du vrai R) à fs=10 kHz sur un signal réaliste.
    amps_at = signal[local_abs]
    max_amp = float(np.max(amps_at))
    if max_amp > 0:
        competitive = local_abs[amps_at >= 0.5 * max_amp]
    else:
        competitive = local_abs
    if len(competitive) == 0:
        competitive = local_abs

    # Choose by maximal upstroke slope (primary) then amplitude (secondary),
    # among amplitude-competitive candidates only.
    best = None
    best_sl = -1.0
    best_amp = -np.inf
    for pk in competitive:
        sl = _upstroke_slope_at(int(pk), signal, fs, deriv=deriv)
        amp = float(signal[int(pk)])
        if sl > best_sl + 1e-12:
            best, best_sl, best_amp = int(pk), sl, amp
        elif abs(sl - best_sl) <= 1e-12 and amp > best_amp:
            best, best_sl, best_amp = int(pk), sl, amp

    return int(np.clip(best if best is not None else approx_i, 0, n - 1))


# ════════════════════════════════════════════════════════════
#  R-vs-J DISAMBIGUATION  (partagé entre les deux détecteurs)
# ════════════════════════════════════════════════════════════

def _adaptive_j_upstroke_ratio(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: float,
    slope_window_ms: float = 3.0,
    base_ratio: float = 0.55,
    snr_window_ms: float = 200.0,
) -> float:
    """Calcule un seuil J_UPSTROKE_RATIO adaptatif selon la qualité du signal.

    Principe
    ────────
    Le seuil fixe 0.55 suppose que la J-wave a un upstroke < 55 % de celui du R.
    Ce critère est trop strict sur des signaux à faible SNR ou à morphologie
    atypique (anesthésie, hypothermie) : la J-wave peut avoir un upstroke
    relativement plus élevé parce que le R est lui-même aplati.

    Stratégie adaptative
    ────────────────────
    1. On estime le SNR local comme le rapport entre l'amplitude médiane des
       pics candidats et le bruit de fond (écart-type du signal dans des
       fenêtres inter-pics).
    2. On estime la dispersion des upstrokes des candidats (coefficient de
       variation = std/mean). Une forte dispersion indique une morphologie
       hétérogène (anesthésie, transition) → le seuil doit être relevé pour
       ne pas rejeter des R atypiques.

    Adaptation
    ──────────
    • SNR élevé  + CV faible   → signal propre, morphologie stable    → ratio ≈ base_ratio (0.55)
    • SNR faible ou CV élevé  → signal dégradé ou morphologie variable → ratio relevé vers 0.75
      (on accepte une J-wave plus « raide » avant de la rejeter)

    Le ratio final est clampé dans [base_ratio, 0.80] pour éviter de tout
    accepter sur un signal totalement bruité.

    Parameters
    ----------
    peaks           : Indices des pics candidats (avant disambiguation).
    signal          : Signal ECG (polarité corrigée).
    fs              : Fréquence d'échantillonnage (Hz).
    slope_window_ms : Fenêtre upstroke (ms, cohérent avec le reste du pipeline).
    base_ratio      : Seuil minimal (signal propre, morphologie stable). Défaut 0.55.
    snr_window_ms   : Largeur des fenêtres inter-pics pour estimer le bruit (ms).

    Returns
    -------
    float in [base_ratio, 0.80]
    """
    if len(peaks) < 4:
        return base_ratio

    # ── 1. Upstrokes de tous les candidats ────────────────────────────────
    slopes = np.array([
        _upstroke_slope_at(int(p), signal, fs, slope_window_ms)
        for p in peaks
    ], dtype=np.float64)
    slopes = slopes[slopes > 1e-9]
    if len(slopes) < 3:
        return base_ratio

    mean_sl = float(np.mean(slopes))
    cv_sl   = float(np.std(slopes, ddof=1)) / (mean_sl + 1e-9)  # coeff de variation

    # ── 2. SNR : amplitude médiane des pics / bruit inter-pics ────────────
    amp_med = float(np.median(signal[peaks]))
    half_w  = max(1, int(snr_window_ms / 2.0 / 1000.0 * fs))
    noise_segs: list[float] = []
    for i in range(len(peaks) - 1):
        a, b = int(peaks[i]), int(peaks[i + 1])
        mid  = (a + b) // 2
        lo   = max(0,          mid - half_w)
        hi   = min(len(signal), mid + half_w)
        seg  = signal[lo:hi]
        if len(seg) > 4:
            noise_segs.append(float(np.std(seg)))
    noise_std = float(np.median(noise_segs)) if noise_segs else (amp_med * 0.1)
    snr       = amp_med / (noise_std + 1e-9)

    # ── 3. Adaptation du ratio ─────────────────────────────────────────────
    # Pénalité SNR : SNR ≥ 10 → 0.0 ; SNR ≤ 3 → 1.0 (signal très dégradé)
    snr_penalty = float(np.clip((10.0 - snr) / 7.0, 0.0, 1.0))
    # Pénalité CV  : CV ≤ 0.30 → 0.0 ; CV ≥ 0.80 → 1.0 (morphologie très hétérogène)
    cv_penalty  = float(np.clip((cv_sl - 0.30) / 0.50, 0.0, 1.0))

    # On combine les deux pénalités (max des deux pour ne pas les diluer)
    penalty = max(snr_penalty, cv_penalty)

    # ratio_max = 0.80 : même sur signal très dégradé, on ne monte pas plus haut
    ratio = base_ratio + penalty * (0.80 - base_ratio)
    ratio = float(np.clip(ratio, base_ratio, 0.80))

    log.debug(
        "_adaptive_j_upstroke_ratio: SNR=%.1f  CV_sl=%.2f  "
        "snr_pen=%.2f  cv_pen=%.2f  → ratio=%.3f",
        snr, cv_sl, snr_penalty, cv_penalty, ratio,
    )
    return ratio


def resolve_r_vs_j_peaks(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: float,
    min_rr_ms: float = MouseECG.MIN_RR_MS,
    j_window_ms: float = 60.0,
    slope_window_ms: float = 3.0,
) -> np.ndarray:
    """Filet de sécurité global : supprime les J-waves résiduelles après détection.

    Ce module est le dernier filtre du pipeline.  Il opère sur l'ensemble des
    pics détectés et traite deux cas résiduels que les étapes locales
    (_select_r_in_window, _snap_to_r) n'ont pas pu gérer :

    Cas 1 — Doublets stricts (gap < min_rr_ms)  [MULTI-PASS]
    ──────────────────────────────────────────────────────────
    Deux pics séparés de moins de min_rr_ms sont physiologiquement impossibles
    (HR max souris ≈ 800 bpm → RR_min ≈ 75 ms).  On conserve le pic avec
    l'upstroke le plus raide (critère primaire) ou la plus grande amplitude
    (fallback uniquement si les upstrokes sont quasi-identiques : ratio < 1.3).

    MULTI-PASS : la résolution est répétée jusqu'à convergence (aucun nouveau
    doublet).  Ceci traite correctement les chaînes de ≥ 3 pics très proches
    (R + J + artefact) que le balayage linéaire unique manquait parce que
    discarding peaks[i] peut créer un nouveau doublet entre peaks[i-1] et
    peaks[i+1].

    Justification du seuil 1.3 (vs 1.5 ancien)
    • Le ratio 1.5 était trop permissif : deux vrais R ne devraient jamais
      être dans ce cas, donc dès qu'il y a un doublet c'est R vs J.
    • 1.3 signifie "upstroke 30 % plus raide → c'est le R". En dessous,
      c'est ambigu et on utilise l'amplitude comme dernier recours.
    • En cas d'égalité parfaite (bruit, morphologie atypique), on garde
      le premier pic (plus précoce = plus proche du QRS).

    Cas 2 — J-wave isolée (gap entre min_rr_ms et j_window_ms)  [RATIO ADAPTATIF]
    ────────────────────────────────────────────────────────────────────────────────
    Une J-wave qui a survécu au rejet local peut encore être détectée comme
    pic indépendant si son amplitude a dépassé le seuil de dérivée.
    On la détecte par deux critères CONJONCTIFS (les deux doivent être vrais) :

    Critère A — TEMPOREL : gap ≤ j_window_ms depuis le R précédent.
    • La J-wave survient dans les 15–60 ms après le complexe QRS chez la
      souris (Nerbonne & Kass 2005, Leoni & Rosenbaum 2014).
    • On utilise 60 ms comme borne haute conservatrice.

    Critère B — UPSTROKE : upstroke(p1) < j_upstroke_ratio × upstroke(p0).
    • La J-wave a une montée nettement moins abrupte que le R précédent.
    • Seuil adaptatif (via _adaptive_j_upstroke_ratio) : base 0.55, relevé
      jusqu'à 0.80 sur signaux à faible SNR ou morphologie hétérogène
      (anesthésie, hypothermie). Sur signal propre, comportement identique
      à l'ancien seuil fixe 0.55.
    • Ce seuil remplace l'ancien critère composé (creux S + upstroke)
      car le creux S n'est pas toujours présent (morphologie R-dominant sans
      onde S profonde). L'upstroke seul est plus universel.

    Note : l'amplitude n'intervient PAS dans les critères de rejet.
    Une J-wave plus haute que R est correctement écartée sur critère
    upstroke. C'est le point central du détecteur.

    Parameters
    ----------
    peaks           : Indices des pics détectés (triés).
    signal          : Signal ECG corrigé en polarité.
    fs              : Fréquence d'échantillonnage (Hz).
    min_rr_ms       : RR minimal physiologique (ms, défaut MouseECG.MIN_RR_MS).
    j_window_ms     : Fenêtre temporelle de look-ahead J-wave (ms, défaut 60).
    slope_window_ms : Fenêtre pour _upstroke_slope_at (ms, défaut 3).
                      Cohérent avec le reste du pipeline (même valeur que
                      upstroke_window_ms dans _select_r_in_window).
    """
    if len(peaks) < 2:
        return peaks

    peaks = np.sort(peaks)
    min_dist_samp = int(min_rr_ms  / 1000.0 * fs)
    j_win_samp    = max(min_dist_samp + 1, int(j_window_ms / 1000.0 * fs))

    # Seuil de ratio upstroke pour l'arbitrage doublet :
    # si upstroke_max / upstroke_min > 1.3 → upstroke décide ;
    # sinon ambiguïté → amplitude décide.
    DOUBLET_SLOPE_RATIO = 1.3

    # Seuil adaptatif pour le rejet J-wave isolée (Cas 2).
    # Sur signal propre ≈ 0.55 (comportement identique à l'ancien seuil fixe).
    # Relevé jusqu'à 0.80 sur signal dégradé ou morphologie hétérogène.
    J_UPSTROKE_RATIO = _adaptive_j_upstroke_ratio(
        peaks, signal, fs, slope_window_ms
    )

    discard: set[int] = set()

    # ── Passe 1 : doublets stricts — résolution MULTI-PASS ─────────────────
    # On répète le balayage jusqu'à convergence (plus aucun nouveau doublet).
    # Nécessaire pour les chaînes R + J + artefact où l'élimination d'un pic
    # peut créer un nouveau doublet entre ses voisins.
    MAX_DOUBLET_PASSES = 10  # garde-fou ; converge en général en 2–3 passes
    for _pass in range(MAX_DOUBLET_PASSES):
        new_discard_this_pass = False
        active = [i for i in range(len(peaks)) if i not in discard]

        for k in range(len(active) - 1):
            i0, i1 = active[k], active[k + 1]
            p0, p1 = peaks[i0], peaks[i1]

            if p1 - p0 >= min_dist_samp:
                continue  # gap OK, pas un doublet

            # Calcul de l'upstroke instantané max (3 ms, cohérent avec pipeline)
            sl0 = _upstroke_slope_at(p0, signal, fs, slope_window_ms)
            sl1 = _upstroke_slope_at(p1, signal, fs, slope_window_ms)

            s_max = max(sl0, sl1)
            s_min = min(sl0, sl1)
            slope_ratio = s_max / (s_min + 1e-9)

            if slope_ratio >= DOUBLET_SLOPE_RATIO:
                # Upstroke discriminant → garder le plus raide
                loser = i1 if sl0 > sl1 else i0
            else:
                # Ambiguïté → fallback amplitude (dernier recours)
                amp0, amp1 = float(signal[p0]), float(signal[p1])
                if abs(amp0 - amp1) / (max(abs(amp0), abs(amp1)) + 1e-9) > 0.05:
                    loser = i1 if amp0 >= amp1 else i0
                else:
                    # Égalité parfaite → garder le premier (plus précoce = plus QRS)
                    loser = i1

            if loser not in discard:
                discard.add(loser)
                new_discard_this_pass = True
                log.debug(
                    "resolve_r_vs_j (doublet pass %d): p0=%d p1=%d gap=%d  "
                    "sl0=%.4f sl1=%.4f ratio=%.2f → discard peaks[%d]=%d",
                    _pass, p0, p1, p1 - p0, sl0, sl1, slope_ratio,
                    loser, peaks[loser],
                )

        if not new_discard_this_pass:
            break  # convergence

    # ── Passe 2 : J-wave isolée (gap entre min_rr_ms et j_window_ms) ────────
    # Critères : (A) temporel ≤ j_window_ms ET (B) upstroke(J) < ratio adaptatif × upstroke(R).
    # L'amplitude n'intervient pas : une J-wave plus haute que R est quand
    # même rejetée si son upstroke est faible.
    active = [i for i in range(len(peaks)) if i not in discard]
    for k in range(len(active) - 1):
        i0, i1 = active[k], active[k + 1]
        p0, p1 = peaks[i0], peaks[i1]
        gap = p1 - p0

        # Critère A : gap dans la fenêtre J-wave [min_rr_ms, j_window_ms]
        if gap < min_dist_samp or gap > j_win_samp:
            continue

        # Critère B : upstroke de p1 nettement inférieur à celui de p0
        sl0 = _upstroke_slope_at(p0, signal, fs, slope_window_ms)
        sl1 = _upstroke_slope_at(p1, signal, fs, slope_window_ms)

        if sl1 < J_UPSTROKE_RATIO * (sl0 + 1e-9):
            discard.add(i1)
            log.debug(
                "resolve_r_vs_j (J-wave isolée): p1=%d écarté  "
                "gap=%d samp  sl_R=%.4f  sl_J=%.4f  ratio=%.2f < %.2f",
                p1, gap, sl0, sl1, sl1 / (sl0 + 1e-9), J_UPSTROKE_RATIO,
            )

    keep_mask   = np.array([i not in discard for i in range(len(peaks))], dtype=bool)
    clean_peaks = peaks[keep_mask]
    if len(discard):
        log.info(
            "resolve_r_vs_j_peaks: supprimé %d J-wave(s)/doublon(s) → %d pics conservés  "
            "(J_ratio=%.3f)",
            len(discard), len(clean_peaks), J_UPSTROKE_RATIO,
        )
    return clean_peaks


# ════════════════════════════════════════════════════════════
#  POLARITY & CANDIDATE DETECTION
# ════════════════════════════════════════════════════════════

def fix_polarity(
    cleaned: np.ndarray,
    fs: float,
    min_dist_ms: float = MouseECG.MIN_RR_MS,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    force_polarity: Optional[bool] = None,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """Determine signal polarity and extract all R-peak candidates.

    Vote multi-fenêtres (5 fenêtres de 6 s) + skewness.
    Fallback tie-break sur kurtosis des candidats, puis percentile 99 brut.

    Parameters
    ----------
    cleaned         : Signal ECG filtré (non inversé).
    fs              : Fréquence d'échantillonnage (Hz).
    min_dist_ms     : Distance minimale entre pics candidats (ms).
    progress_cb     : Callback optionnel (percent, message).
    force_polarity  : Si fourni, court-circuite le vote :
                        False → signal tel quel (R positifs, pas d'inversion)
                        True  → signal inversé (-cleaned)
                      Permet à l'UI d'honorer le choix « raw / no auto-flip »
                      de l'utilisateur sans modifier le reste du pipeline.
    """
    min_dist = max(1, int(min_dist_ms / 1000 * fs))
    # Bounds every find_peaks(..., prominence=...) call below -- without it,
    # prominence search on a raw/unfiltered signal (real baseline wander,
    # no bandpass to remove it) is catastrophically slow. See
    # MouseECG.PROMINENCE_WLEN_MS.
    wlen = _prominence_wlen_samples(fs)

    if progress_cb:
        progress_cb(10, "Polarity detection…")

    # ── Court-circuit : polarité imposée par l'utilisateur ─────────────────
    if force_polarity is not None:
        inverted   = bool(force_polarity)
        signal_out = -cleaned if inverted else cleaned
        log.debug(
            "fix_polarity: force_polarity=%s → %s (vote skipped)",
            force_polarity, "INVERTED" if inverted else "normal",
        )
        if progress_cb:
            progress_cb(40, "Candidate peak detection…")
        height_thresh = float(np.percentile(signal_out, 10))
        cands, props  = find_peaks(
            signal_out,
            distance=min_dist,
            height=height_thresh,
            prominence=0,
            wlen=wlen,
        )
        proms = props.get("prominences", np.array([]))
        if progress_cb:
            progress_cb(90, f"Found {len(cands):,} candidates")
        return signal_out, inverted, cands, proms

    # ── Vote multi-fenêtres (5 × 6 s) + skewness ──────────────────────────
    WIN_S   = 6
    win_len = min(len(cleaned), int(WIN_S * fs))
    total_s = len(cleaned) / fs

    offsets = [
        max(0, int(p * (len(cleaned) - win_len)))
        for p in np.linspace(0.0, 1.0, 5)
    ]

    votes_pos = 0
    votes_neg = 0
    for off in offsets:
        sub  = cleaned[off:off + win_len]
        p99  = float(np.percentile(sub, 99))
        n99  = float(np.percentile(-sub, 99))
        skew = float(np.mean((sub - sub.mean()) ** 3) / (sub.std() ** 3 + 1e-12))
        score_pos = (p99 > n99 * 1.10) + (skew > 0.2)
        score_neg = (n99 > p99 * 1.10) + (skew < -0.2)
        if score_pos > score_neg:
            votes_pos += 1
        elif score_neg > score_pos:
            votes_neg += 1
        else:
            votes_pos += int(p99 >= n99)
            votes_neg += int(n99 > p99)

    # ── Fallback kurtosis si votes égaux ──────────────────────────────────
    # Sur des enregistrements courts (< 6 s) ou très filtrés, skewness et
    # percentile peuvent être ambigus.  La kurtosis des pics candidats est
    # un discriminateur robuste : les R-peaks sur la bonne polarité créent
    # une queue droite lourde (kurtosis élevée sur le signal positif).
    if votes_pos == votes_neg:
        min_dist_kurt = max(1, int(min_dist_ms / 1000 * fs))
        cands_p, _   = find_peaks(cleaned,  distance=min_dist_kurt, prominence=0, wlen=wlen)
        cands_n, _   = find_peaks(-cleaned, distance=min_dist_kurt, prominence=0, wlen=wlen)
        # Kurtosis des amplitudes des candidats (excess kurtosis)
        kurt_p = float(np.mean((cleaned[cands_p]  - cleaned[cands_p].mean())  ** 4) /
                       (cleaned[cands_p].std()  ** 4 + 1e-12) - 3) if len(cands_p) > 3 else 0.0
        kurt_n = float(np.mean((-cleaned[cands_n] - (-cleaned[cands_n]).mean()) ** 4) /
                       ((-cleaned[cands_n]).std() ** 4 + 1e-12) - 3) if len(cands_n) > 3 else 0.0
        inverted = kurt_n > kurt_p
        log.debug(
            "fix_polarity: tie-break via kurtosis  kurt_p=%.2f  kurt_n=%.2f → %s",
            kurt_p, kurt_n, "INVERTED" if inverted else "normal",
        )
    else:
        inverted = votes_neg > votes_pos

    log.debug(
        "fix_polarity: votes_pos=%d votes_neg=%d → %s  (%.0f s signal)",
        votes_pos, votes_neg,
        "INVERTED" if inverted else "normal",
        total_s,
    )

    signal_out = -cleaned if inverted else cleaned

    if progress_cb:
        progress_cb(40, "Candidate peak detection…")

    height_thresh = float(np.percentile(signal_out, 10))
    cands, props  = find_peaks(
        signal_out,
        distance=min_dist,
        height=height_thresh,
        prominence=0,
        wlen=wlen,
    )
    proms = props.get("prominences", np.array([]))

    if progress_cb:
        progress_cb(90, f"Found {len(cands):,} candidates")

    return signal_out, inverted, cands, proms


def apply_threshold(
    cleaned: np.ndarray,
    cands: np.ndarray,
    proms: np.ndarray,
    thresh_frac: float,
    fs: float = 0.0,
    adaptive_window_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Filter pre-computed peak candidates by a prominence threshold.

    Mode adaptatif (fenêtres glissantes de *adaptive_window_s* s) si
    l'enregistrement est suffisamment long ; sinon mode global.
    """
    if len(cands) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0.0

    use_adaptive = (
        fs > 0
        and len(cands) >= 10
        and (cands[-1] - cands[0]) / fs > adaptive_window_s
    )

    if use_adaptive:
        half_win_samples = int(adaptive_window_s / 2.0 * fs)
        mask = np.zeros(len(cands), dtype=bool)
        global_ref = float(np.median(proms))
        n_c = len(cands)
        # Two-pointer sliding window instead of an O(n) boolean scan per
        # candidate (O(n^2) total). `cands` is sorted ascending (guaranteed —
        # find_peaks/np.sort upstream), so as idx increases the window
        # [center-half_win, center+half_win] only ever moves forward, never
        # backward — lo/hi need only advance, never reset. Same result,
        # linear instead of quadratic in the number of candidates.
        lo = 0
        hi = 0
        for idx in range(n_c):
            center = cands[idx]
            while lo < n_c and cands[lo] < center - half_win_samples:
                lo += 1
            while hi < n_c and cands[hi] <= center + half_win_samples:
                hi += 1
            count = hi - lo
            # Médiane locale — insensible aux outliers hauts (artefacts, R ectopiques)
            ref = float(np.median(proms[lo:hi])) if count >= 4 else global_ref
            mask[idx] = proms[idx] >= thresh_frac * ref
        log.debug("apply_threshold: adaptive mode  frac=%.3f", thresh_frac)
    else:
        # Médiane globale — P75 tirait le seuil trop haut sur signaux avec outliers
        ref_prom = float(np.median(proms))
        mask     = proms >= thresh_frac * ref_prom
        log.debug("apply_threshold: global mode  frac=%.3f  ref=%.4f", thresh_frac, ref_prom)

    accepted = cands[mask]
    rejected = cands[~mask]

    if fs > 0 and len(accepted) > 1:
        accepted_clean = resolve_r_vs_j_peaks(accepted, cleaned, fs)
        removed_by_j   = np.setdiff1d(accepted, accepted_clean)
        if len(removed_by_j):
            rejected = np.sort(np.concatenate([rejected, removed_by_j]))
        accepted = accepted_clean

    thresh_amp = (
        float(np.percentile(cleaned[accepted], 5)) if len(accepted) > 0
        else float(thresh_frac * np.median(cleaned[cands]))
    )
    log.debug("apply_threshold: %d accepted / %d rejected", len(accepted), len(rejected))
    return accepted, rejected, thresh_amp


def peak_agreement_counts(
    primary: np.ndarray,
    others: "list[np.ndarray]",
    tolerance_samples: int,
) -> np.ndarray:
    """For each peak in *primary*, count how many arrays in *others* have a
    peak within *tolerance_samples* of it.

    Pure array-matching helper for cross-method agreement checks -- takes
    already-computed peak-sample arrays from independent detectors (e.g.
    Wavelet, SG+Derivative, Envelope Max, ML) and reports, per beat in the
    currently-active detection result, how many of the *other* methods
    also found a peak there. A beat with a low count across several
    independently-implemented detectors is a much stronger "this beat is
    questionable" signal than any single method's own confidence score.

    Each element of *others* must be sorted ascending (true of every
    detect_peaks_* function's output). Uses searchsorted for O(n log m)
    matching per method instead of an O(n*m) all-pairs comparison.
    """
    counts = np.zeros(len(primary), dtype=int)
    if len(primary) == 0:
        return counts
    primary = np.asarray(primary)
    for other in others:
        other = np.asarray(other)
        if len(other) == 0:
            continue
        idx    = np.searchsorted(other, primary)
        idx_lo = np.clip(idx - 1, 0, len(other) - 1)
        idx_hi = np.clip(idx, 0, len(other) - 1)
        d_lo   = np.abs(primary - other[idx_lo])
        d_hi   = np.abs(primary - other[idx_hi])
        counts += (np.minimum(d_lo, d_hi) <= tolerance_samples).astype(int)
    return counts


# ════════════════════════════════════════════════════════════
#  HELPERS PARTAGÉS — calibration CWT physique, seuillage robuste,
#  validation morphologique locale (partagée wavelet + SG)
# ════════════════════════════════════════════════════════════

def _robust_threshold(x: np.ndarray, k: float = 5.0, positive_only: bool = True) -> float:
    """Seuil robuste = médiane + k × MAD (Median Absolute Deviation).

    Remplace percentile(x, p) : un percentile fixe se déplace avec la
    *proportion* d'événements dépassant le bruit (donc avec la fréquence
    cardiaque et le taux d'artefacts), alors que médiane+k×MAD est ancré
    sur le niveau de bruit typique, indépendamment de combien de pics le
    dépassent. La MAD a un point de rupture de 50 % (contre 0 % pour un
    écart-type classique) — un motif de stimulation électrique ou une
    salve d'artefacts de mouvement ne peut pas, à lui seul, faire dériver
    le seuil. Constante 1.4826 : facteur de cohérence rendant la MAD
    comparable à un écart-type sous hypothèse de bruit gaussien
    (Rousseeuw & Croux 1993, "Alternatives to the Median Absolute
    Deviation").
    """
    vals = x[x > 0] if positive_only else x
    if len(vals) == 0:
        return 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med + k * 1.4826 * mad


def _cwt_scales_for_qrs(
    wavelet: str,
    fs: float,
    qrs_width_ms: float,
    n_scales: int = 8,
    octaves_span: float = 2.0,
) -> "tuple[np.ndarray, float]":
    """Échelles CWT calibrées physiquement sur la largeur de QRS attendue.

    Remplace une plage d'échelles codée en dur (ex. ``(1, 4)``, valide
    uniquement pour un fs implicite donné) par une conversion largeur
    QRS (ms) → échelles, via ``pywt.frequency2scale`` — le détecteur
    devient indépendant de fs et fonctionne à 1, 2, 5 ou 10 kHz sans
    changer une seule ligne de code.

    Justification physique
    ───────────────────────
    Un complexe QRS de durée T_qrs concentre l'essentiel de son énergie
    spectrale autour de f_center ≈ 1/T_qrs, étalée sur environ ±1 octave
    de part et d'autre (c'est très exactement le raisonnement qui fonde
    la bande passante 5–35 Hz de Pan & Tompkins 1985 pour un QRS humain
    ≈ 80–100 ms ; ici la règle est généralisée en fonction de T_qrs
    plutôt qu'exprimée en Hz fixes, pour couvrir un QRS murin typique de
    6–12 ms, cf. Speerschneider & Thomas 2013, "Physiology and
    pathophysiology of cardiac electrophysiology in the healthy and
    diseased mouse heart").

    La relation scale ↔ fréquence physique d'une CWT est
    ``f = f_c × fs / scale`` où f_c = fréquence centrale normalisée de
    l'ondelette-mère (``pywt.central_frequency``) — donc
    ``scale = f_c × fs / f``. On calcule les deux bornes d'échelle à
    partir des deux bornes de fréquence physiologique, et on échantillonne
    ``n_scales`` échelles linéairement entre elles.

    Retourne (scales, scale_center) où scale_center est l'échelle
    correspondant exactement à f_center — utilisée comme centre de la
    pondération gaussienne en log-échelle (voir _weighted_multiscale_energy).
    """
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "pywt est requis pour _cwt_scales_for_qrs : pip install PyWavelets"
        ) from exc

    f_c   = pywt.central_frequency(wavelet)
    T_qrs = max(1e-4, qrs_width_ms / 1000.0)
    f_center_hz = 1.0 / T_qrs
    f_lo_hz = f_center_hz / (2.0 ** (octaves_span / 2.0))
    f_hi_hz = f_center_hz * (2.0 ** (octaves_span / 2.0))

    nyq = fs / 2.0
    f_hi_hz = min(f_hi_hz, 0.95 * nyq)
    f_lo_hz = min(f_lo_hz, 0.5 * f_hi_hz)

    freq_lo_norm = max(f_lo_hz / fs, 1e-6)
    freq_hi_norm = max(f_hi_hz / fs, freq_lo_norm * 1.01)

    if hasattr(pywt, "frequency2scale"):
        # basse fréquence → grande échelle, haute fréquence → petite échelle
        scale_hi = float(pywt.frequency2scale(wavelet, freq_lo_norm))
        scale_lo = float(pywt.frequency2scale(wavelet, freq_hi_norm))
    else:
        # pywt < 1.1 : pas de frequency2scale — inversion manuelle de
        # scale2frequency(wavelet, s) ≈ f_c / s pour mexh/gaus* (relation
        # exacte pour ces familles, cf. pywt._functions.scale2frequency).
        scale_hi = f_c / freq_lo_norm
        scale_lo = f_c / freq_hi_norm

    scale_lo = max(1.0, scale_lo)
    scale_hi = max(scale_lo + 0.5, scale_hi)
    scale_center = f_c * fs * T_qrs  # = f_c / (f_center_hz/fs), la relation exacte

    scales = np.linspace(scale_lo, scale_hi, n_scales)
    return scales, float(scale_center)


def _weighted_multiscale_energy(
    coeffs: np.ndarray,
    scales: np.ndarray,
    scale_center: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Énergie CWT multi-échelle : normalisation robuste + pondération log-gaussienne.

    Deux étapes, dans cet ordre (la normalisation doit précéder la
    pondération, sinon une échelle bruyante à forte variance domine la
    somme avant même d'être atténuée) :

    1. Normalisation robuste par échelle — chaque ligne de coefficients
       est divisée par son écart-type robuste (MAD × 1.4826) plutôt que
       par son écart-type classique. Un artefact ponctuel de grande
       amplitude (décollement d'électrode, mouvement) gonfle une variance
       classique bien plus qu'une MAD (point de rupture 50 % vs 0 %), donc
       la normalisation MAD garde un seuil stable même en présence de
       quelques échantillons aberrants sur une échelle donnée.
    2. Pondération gaussienne en log2(scale), centrée sur
       log2(scale_center) (scale_center vient de _cwt_scales_for_qrs,
       donc physiquement ancré sur la largeur de QRS attendue). Les
       échelles CWT sont naturellement multiplicatives (chaque échelle
       "voit" une octave de fréquence) — une gaussienne en base log est
       donc l'équivalent naturel d'une gaussienne en fréquence, sans
       coupure dure aux bords de la bande sélectionnée : le bruit HF
       (petites échelles) et l'onde J/T (grandes échelles) sont atténués
       en douceur plutôt qu'exclus binairement, ce qui évite l'effet de
       bord (ringing, faux positifs) d'un filtre passe-bande à coupure
       abrupte.
    """
    eps = 1e-12
    med_per_scale = np.median(coeffs, axis=1, keepdims=True)
    mad_per_scale = np.median(np.abs(coeffs - med_per_scale), axis=1, keepdims=True)
    robust_std    = 1.4826 * mad_per_scale + eps
    coeffs_norm   = coeffs / robust_std

    log_scales = np.log2(scales)
    log_center = np.log2(max(scale_center, scales.min()))
    # σ = demi-étendue log de la bande sélectionnée : les bornes scale_lo/
    # scale_hi (déjà choisies à ±1 octave physiologique) se retrouvent à
    # ±1σ — poids ≈0.61 en bord de bande, décroissance douce au-delà.
    sigma_log = max(1e-6, (log_scales.max() - log_scales.min()) / 2.0)
    weights   = np.exp(-0.5 * ((log_scales - log_center) / sigma_log) ** 2)
    weights   = weights / weights.sum()

    energy = np.sum(weights[:, None] * coeffs_norm ** 2, axis=0)
    return energy, weights


def _peak_width_and_downstroke(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: float,
    search_ms: float = 15.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """Largeur à mi-hauteur et pente descendante de chaque pic.

    Recherche bornée à ±search_ms autour de chaque pic — coût
    O(n_peaks × w) avec w = quelques dizaines d'échantillons, donc
    linéaire en nombre de pics (pas de O(n_samples) par pic : on ne
    scanne jamais plus que la fenêtre physiologique locale).
    """
    n = len(peaks)
    win = max(2, int(round(search_ms / 1000.0 * fs)))
    widths       = np.empty(n, dtype=np.float64)
    slopes_down  = np.empty(n, dtype=np.float64)
    sig_len = len(signal)

    for i, pk in enumerate(peaks):
        pk = int(pk)
        amp  = float(signal[pk])
        half = amp / 2.0
        lo = max(0, pk - win)
        hi = min(sig_len, pk + win + 1)

        seg_r = signal[pk:hi]
        below_r = np.where(seg_r <= half)[0]
        t_r = int(below_r[0]) if len(below_r) else (hi - pk - 1)

        seg_l = signal[lo:pk + 1][::-1]
        below_l = np.where(seg_l <= half)[0]
        t_l = int(below_l[0]) if len(below_l) else (pk - lo)

        widths[i] = t_l + t_r
        d = np.diff(signal[pk:hi])
        slopes_down[i] = float(np.min(d)) if len(d) else 0.0  # négative

    return widths, slopes_down


def _local_median_1d(x: np.ndarray, half_window: int) -> np.ndarray:
    """Médiane glissante vectorisée (sliding_window_view), pas de boucle O(n × w).

    Remplace un pattern `for idx: median(x[idx-w:idx+w])` — O(n × w) en
    Python pur, potentiellement des milliers d'appels `np.median` sur de
    grands enregistrements — par un seul appel vectorisé.
    """
    n = len(x)
    if n == 0:
        return x
    if n <= 2 * half_window + 1:
        return np.full(n, np.median(x))
    x_padded = np.pad(x, half_window, mode="edge")
    windows  = np.lib.stride_tricks.sliding_window_view(x_padded, 2 * half_window + 1)
    return np.median(windows, axis=1)


def _local_morphology_filter(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: float,
    *,
    min_upstroke_frac: float = 0.35,
    min_amp_frac: float = 0.35,
    min_width_frac: float = 0.0,   # 0 = critère désactivé
    max_asymmetry: float = 0.0,    # 0 = critère désactivé
    window_beats: int = 10,
    deriv: "Optional[np.ndarray]" = None,
) -> np.ndarray:
    """Validation morphologique par cohérence locale — partagée entre détecteurs.

    Pour chaque candidat, compare jusqu'à 4 descripteurs (pente montante,
    amplitude, largeur à mi-hauteur, asymétrie montée/descente) à la
    MÉDIANE LOCALE du même descripteur sur les ±window_beats battements
    voisins — pas à une statistique globale. Un vrai QRS ressemble à ses
    voisins immédiats ; une onde P/T mal filtrée, un artefact de
    stimulation, ou un bruit ponctuel s'en écarte typiquement sur au
    moins un axe.

    La fenêtre LOCALE (et non globale) est délibérée : elle évite les
    faux rejets pendant une dérive d'électrode lente, une transition
    d'anesthésie, ou toute variation physiologique légitime de
    l'amplitude sur la durée de l'enregistrement — un seuil global
    échouerait à s'y adapter.

    Chaque critère est indépendamment activable (frac/asymmetry = 0
    désactive) : le détecteur SG-dérivée n'active que pente+amplitude
    (son comportement historique, préservé à l'identique) ; le détecteur
    wavelet active les 4 critères. Une seule implémentation, deux
    profils d'usage — au lieu de deux implémentations dupliquées.
    """
    n = len(peaks)
    if n == 0:
        return peaks

    slopes_up = np.array([_upstroke_slope_at(int(pk), signal, fs, deriv=deriv) for pk in peaks])
    amps      = signal[peaks.astype(int)]

    keep = np.ones(n, dtype=bool)
    local_sl  = _local_median_1d(slopes_up, window_beats)
    local_amp = _local_median_1d(amps, window_beats)
    keep &= slopes_up >= np.maximum(1e-9, min_upstroke_frac * local_sl)
    keep &= amps      >= np.maximum(0.0,  min_amp_frac      * local_amp)

    if min_width_frac > 0 or max_asymmetry > 0:
        widths, slopes_down = _peak_width_and_downstroke(peaks, signal, fs)
        if min_width_frac > 0:
            local_w = _local_median_1d(widths, window_beats)
            keep &= widths >= np.maximum(1.0, min_width_frac * local_w)
        if max_asymmetry > 0:
            # asymétrie = |pente montée| / (|pente montée| + |pente descente|).
            # 0.5 = parfaitement symétrique. Le QRS murin est typiquement
            # asymétrique (montée plus raide que la descente) mais dans une
            # plage bornée — max_asymmetry fixe l'écart toléré autour de 0.5.
            denom = np.abs(slopes_up) + np.abs(slopes_down) + 1e-12
            asym  = np.abs(slopes_up) / denom
            keep &= np.abs(asym - 0.5) <= max_asymmetry

    return peaks[keep]


def _zscore_mad(x: np.ndarray) -> np.ndarray:
    """Z-score robuste : (x − médiane) / (1.4826 × MAD).

    Utilisé pour ramener des descripteurs hétérogènes (énergie, largeur en
    échantillons, pente en unités/s, NCC sans dimension...) sur une échelle
    commune avant de les combiner dans le score composite — sans quoi une
    somme pondérée mélangerait des unités incompatibles. Auto-calibré sur
    la distribution du lot de candidats lui-même : indépendant de
    l'amplitude absolue du signal et de fs (aucune constante fixe).
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - med) / scale


def _modulus_maxima_alignment(
    coeffs: np.ndarray,
    scales: np.ndarray,
    positions: np.ndarray,
    search_samples: int,
) -> np.ndarray:
    """Dispersion inter-échelle de la position du maximum du module CWT.

    Justification (Mallat & Hwang 1992, "Singularity Detection and
    Processing with Wavelets", IEEE Trans. Information Theory) : la
    régularité locale (exposant de Lipschitz) d'une discontinuité se lit
    dans l'évolution de sa "ligne de maxima" du module |CWT| à travers les
    échelles. Une singularité franche et isolée — précisément ce qu'est
    l'upstroke du QRS, une quasi-marche — produit une ligne de maxima qui
    converge vers une position quasi-fixe quand l'échelle varie. Un burst
    EMG/respiratoire, composé de plusieurs unités motrices actives à des
    instants légèrement décalés, n'a pas de point focal unique : la
    position du maximum |coeffs[s,:]| dérive d'une échelle à l'autre.

    C'est une information ORTHOGONALE à l'énergie ou la persistance déjà
    calculées : un burst EMG peut être aussi énergique et aussi
    "persistant" (actif sur plusieurs échelles indépendamment) qu'un QRS,
    tout en n'ayant jamais une ligne de maxima alignée — c'est précisément
    ce que ce descripteur capture et qu'aucun des critères précédents ne
    testait.

    Retourne la MAD (en échantillons) de la position du maximum local à
    travers les échelles, pour chaque candidat — PETIT = ligne de maxima
    alignée (signature d'une vraie singularité) ; GRAND = dispersée
    (signature d'un événement large-bande diffus).
    """
    n_scales  = len(scales)
    n         = len(positions)
    n_samples = coeffs.shape[1]
    abs_coeffs = np.abs(coeffs)

    argmax_pos = np.empty((n_scales, n), dtype=np.float64)
    for s in range(n_scales):
        row = abs_coeffs[s]
        for i, p in enumerate(positions):
            lo = max(0, int(p) - search_samples)
            hi = min(n_samples, int(p) + search_samples + 1)
            argmax_pos[s, i] = lo + int(np.argmax(row[lo:hi]))

    med = np.median(argmax_pos, axis=0)
    mad = np.median(np.abs(argmax_pos - med), axis=0)
    return mad


def _extract_normalized_window(
    signal: np.ndarray, peak: int, half_win: int
) -> "Optional[np.ndarray]":
    """Fenêtre centrée sur *peak*, longueur fixe 2×half_win+1, z-normalisée.

    La z-normalisation (moyenne 0, écart-type 1) rend la corrélation
    ultérieure indépendante de l'amplitude absolue — deux battements de
    gains très différents mais de même morphologie produisent la même
    fenêtre normalisée.
    """
    n = len(signal)
    lo = max(0, peak - half_win)
    hi = min(n, peak + half_win + 1)
    win = signal[lo:hi]
    target_len = 2 * half_win + 1
    if len(win) < target_len:
        pad_left  = max(0, half_win - (peak - lo))
        pad_right = max(0, target_len - len(win) - pad_left)
        win = np.pad(win, (pad_left, pad_right), mode="edge")
    if len(win) != target_len:
        return None
    std = win.std()
    if std < 1e-9:
        return None
    return (win - win.mean()) / std


def _build_qrs_template(
    signal: np.ndarray,
    trusted_peaks: np.ndarray,
    half_win: int,
) -> "Optional[np.ndarray]":
    """Construit un template QRS moyen (médian) à partir de battements de confiance.

    Utilise la MÉDIANE point-par-point plutôt que la moyenne — robuste au
    cas où quelques battements du sous-ensemble "de confiance" seraient
    encore atypiques (extrasystole, léger artefact résiduel) : une moyenne
    les laisserait déformer le template, une médiane les ignore tant
    qu'ils restent minoritaires (point de rupture 50 %).

    Anti-dérive : cette fonction ne fait PAS de mise à jour incrémentale
    battement-par-battement (qui dériverait progressivement si des
    faux positifs s'y glissent). Le template est construit UNE FOIS à
    partir d'un ensemble de candidats déjà validés par les critères
    indépendants du gabarit (persistance, largeur, pente — cf.
    _composite_qrs_score, phase 1), répartis sur tout l'enregistrement
    plutôt que les seuls premiers battements (robuste à une dérive lente
    d'amplitude/impédance d'électrode sur la durée de l'acquisition).
    """
    windows = []
    for p in trusted_peaks:
        w = _extract_normalized_window(signal, int(p), half_win)
        if w is not None:
            windows.append(w)
    if len(windows) < 5:
        return None
    template = np.median(np.array(windows), axis=0)
    std = template.std()
    if std < 1e-9:
        return None
    return (template - template.mean()) / std


def _template_ncc_scores(
    signal: np.ndarray,
    peaks: np.ndarray,
    template: "Optional[np.ndarray]",
    half_win: int,
) -> np.ndarray:
    """Corrélation croisée normalisée (Pearson) de chaque candidat avec le template.

    Amplitude-indépendante par construction (les deux fenêtres comparées
    sont chacune z-normalisées). Retourne des scores dans [-1, 1] ; -1.0
    est renvoyé pour tout candidat dont la fenêtre ne peut être extraite
    proprement (bord de signal, plateau nul) — traité comme "pire cas",
    jamais comme valeur manquante silencieuse.
    """
    n = len(peaks)
    scores = np.full(n, -1.0, dtype=np.float64)
    if template is None:
        return scores
    for i, p in enumerate(peaks):
        w = _extract_normalized_window(signal, int(p), half_win)
        if w is not None and len(w) == len(template):
            c = float(np.corrcoef(w, template)[0, 1])
            if np.isfinite(c):
                scores[i] = c
    return scores


def _extended_morphology_descriptors(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: float,
    prominences: np.ndarray,
    search_ms: float = 15.0,
) -> dict:
    """Descripteurs morphologiques étendus pour le score composite.

    Littérature sur le pouvoir discriminant relatif (QRS vs onde J/P/T vs
    bruit/EMG) :
    - **Largeur** (durée du complexe) : descripteur le plus robuste et le
      plus consistant à travers les études de délinéation ECG (Martinez
      et al. 2004, délinéateur ondelette ; revue Elgendi, Jonkman & De
      Boer 2013 sur les détecteurs QRS) — un QRS est nettement plus étroit
      qu'une onde T ou qu'un burst EMG large-bande.
    - **Pente montante** : deuxième descripteur le plus robuste — c'est le
      principe fondateur des détecteurs dérivés classiques (Pan & Tompkins
      1985 ; Hamilton & Tompkins 1986).
    - **Rapport largeur/hauteur** : lie les deux précédents en un seul
      indicateur d'"impulsivité" — un QRS concentre l'énergie sur une
      durée courte relativement à son amplitude ; un burst EMG, même
      d'amplitude comparable, étale son énergie sur une durée plus longue.
      C'est un des descripteurs les plus directement pertinents pour CE
      problème précis (distinguer un événement concentré d'un événement
      diffus), complémentaire de l'alignement multi-échelle (même concept,
      mesuré dans le domaine temporel plutôt que dans le domaine ondelette).
    - **Courbure locale au sommet** (dérivée seconde discrète) : capture la
      "pointe" du pic indépendamment de sa largeur totale à la base — un
      QRS a un sommet net, un renflement EMG un sommet arrondi, même si les
      deux ont par ailleurs une largeur à mi-hauteur comparable.
    - **Symétrie / rapport montée-descente** : descripteur plus faible et
      plus bruité individuellement (la morphologie QRS varie selon
      l'espèce, le placement d'électrode) — traité comme terme de score
      SOUPLE, jamais comme seuil dur (cf. Elgendi 2013).

    Toutes les largeurs utilisent ``scipy.signal.peak_widths`` (hauteur
    relative à la PROMINENCE de chaque pic, cohérent avec
    _topographic_prominences) plutôt qu'une recherche de mi-hauteur
    maison — même convention que scipy, testée et vectorisée.
    """
    n = len(peaks)
    out = dict(width25=np.zeros(n), width50=np.zeros(n), width75=np.zeros(n),
               curvature=np.zeros(n), width_height_ratio=np.zeros(n),
               slope_up=np.zeros(n), slope_down=np.zeros(n))
    if n == 0:
        return out

    try:
        _wlen = _prominence_wlen_samples(fs)
        w25, *_ = peak_widths(signal, peaks, rel_height=0.25, wlen=_wlen)
        w50, *_ = peak_widths(signal, peaks, rel_height=0.50, wlen=_wlen)
        w75, *_ = peak_widths(signal, peaks, rel_height=0.75, wlen=_wlen)
    except Exception as exc:
        log.debug("_extended_morphology_descriptors: peak_widths failed (%s)", exc)
        w25 = w50 = w75 = np.zeros(n)

    _, slopes_down = _peak_width_and_downstroke(peaks, signal, fs, search_ms=search_ms)
    slopes_up = np.array([_upstroke_slope_at(int(pk), signal, fs) for pk in peaks])

    win = max(1, int(round(0.001 * fs)))  # ±1 ms pour la dérivée seconde discrète
    curv = np.empty(n)
    sig_len = len(signal)
    for i, pk in enumerate(peaks):
        pk = int(pk)
        lo, hi = max(0, pk - win), min(sig_len - 1, pk + win)
        curv[i] = signal[lo] - 2 * signal[pk] + signal[hi]  # < 0 pour un maximum net

    ratio = np.where(w50 > 1e-9, np.abs(prominences) / w50, 0.0)

    out.update(width25=w25, width50=w50, width75=w75, curvature=curv,
               width_height_ratio=ratio, slope_up=slopes_up, slope_down=slopes_down)
    return out


def _composite_qrs_score(
    signal: np.ndarray,
    peaks: np.ndarray,
    coeffs: np.ndarray,
    scales: np.ndarray,
    energy: np.ndarray,
    persistence: np.ndarray,
    n_scales_total: int,
    fs: float,
    scale_center: float,
    align_search_samples: int,
    template_half_win: int,
) -> "tuple[np.ndarray, dict]":
    """Score de confiance composite — remplace la cascade de seuils durs.

    Principe (cf. Elgendi, Jonkman & De Boer 2013 : les détecteurs à
    seuils EN CASCADE — chaque critère doit passer indépendamment —
    perdent de l'information : un candidat peut franchir CHAQUE seuil
    individuellement de justesse sans jamais être comparé à l'ensemble
    des preuves disponibles. Un burst respiratoire modérément énergique
    ET modérément net PEUT ainsi passer une cascade sans jamais ressembler
    globalement à un QRS. Un score fusionné combine l'évidence partielle
    de chaque critère au lieu de la traiter en tout-ou-rien.

    Deux passes :
    1. Score PROVISOIRE (tout sauf template-NCC) → sélectionne un
       sous-ensemble de confiance réparti sur tout l'enregistrement pour
       construire le template QRS adaptatif (cf. _build_qrs_template).
    2. Score FINAL = fusion pondérée de 8 descripteurs, chacun z-normalisé
       (_zscore_mad) puis pondéré selon son pouvoir discriminant reconnu
       dans la littérature (poids forts : NCC template et largeur/pente ;
       poids faibles : symétrie/courbure, cf. docstring de
       _extended_morphology_descriptors).

    Retourne (score, details) où *details* contient chaque descripteur brut
    (diagnostic / traçabilité) et le template construit (ou None).
    """
    n = len(peaks)
    if n == 0:
        return np.array([]), {}

    prominences = _topographic_prominences(signal, peaks, fs)
    # energy/persistence sont des propriétés du domaine CWT, dont le maximum
    # local se situe près de l'inflexion de montée du QRS -- PAS à
    # l'amplitude maximale où `peaks` a été replacé par le snap. Les lire
    # directement à `peaks` sous-échantillonne systématiquement même un
    # battement parfait. On relocalise le maximum d'énergie dans une petite
    # fenêtre autour de chaque candidat (même largeur que la recherche
    # d'alignement multi-échelle, déjà dérivée de qrs_width_ms).
    idx_in_energy = np.empty(n, dtype=int)
    n_energy = len(energy)
    for i, p in enumerate(peaks):
        lo = max(0, int(p) - align_search_samples)
        hi = min(n_energy, int(p) + align_search_samples + 1)
        idx_in_energy[i] = lo + int(np.argmax(energy[lo:hi]))
    persistence_frac = persistence[idx_in_energy] / float(n_scales_total)
    z_energy = _zscore_mad(energy[idx_in_energy])
    z_prom   = _zscore_mad(prominences)

    morph = _extended_morphology_descriptors(peaks, signal, fs, prominences)
    z_width   = _zscore_mad(morph["width50"])
    z_ratio   = _zscore_mad(morph["width_height_ratio"])
    z_curv    = _zscore_mad(-morph["curvature"])   # courbure plus négative = pic plus net = meilleur
    z_slope   = _zscore_mad(morph["slope_up"])
    denom = np.abs(morph["slope_up"]) + np.abs(morph["slope_down"]) + 1e-12
    asym  = np.abs(morph["slope_up"]) / denom
    z_asym = -_zscore_mad(np.abs(asym - np.median(asym)))  # proche de la population = mieux

    align_mad = _modulus_maxima_alignment(coeffs, scales, peaks, align_search_samples)
    # Comme pour ncc_raw : align_mad est borné (≥0) et se regroupe souvent
    # exactement à 0 pour une vraie singularité isolée — la MAD de la
    # population peut donc être minuscule ou nulle, ce qui ferait exploser
    # un z-score MAD pour la moindre dispersion non nulle mais encore
    # physiologiquement négligeable. Normalisation directe par la fenêtre
    # de recherche elle-même (déjà dérivée de qrs_width_ms/fs, donc
    # fs-indépendante) plutôt que par la dispersion de la population.
    z_align = -3.0 * (align_mad / max(1.0, float(align_search_samples)))

    # ── Score provisoire (sans NCC) pour choisir les battements "de confiance" ──
    provisional = (
        1.5 * z_width + 1.5 * z_slope + 1.0 * z_ratio + 1.0 * z_align +
        1.0 * z_prom + 0.5 * persistence_frac * 4.0 + 0.5 * z_curv + 0.25 * z_asym
    )
    # Sous-ensemble de confiance réparti sur l'enregistrement : on prend le
    # meilleur candidat de chaque bloc temporel plutôt que le top-N global,
    # pour éviter de biaiser le template vers une seule portion de
    # l'enregistrement (dérive d'amplitude, changement de position électrode).
    n_blocks = min(60, max(5, n // 10))
    block_ids = np.linspace(0, n_blocks, n, endpoint=False).astype(int)
    trusted = []
    strong = provisional > np.median(provisional)  # au-dessus de la médiane du lot
    for b in range(n_blocks):
        in_block = np.where((block_ids == b) & strong)[0]
        if len(in_block):
            trusted.append(peaks[in_block[np.argmax(provisional[in_block])]])
    trusted = np.array(trusted, dtype=int)

    template = _build_qrs_template(signal, trusted, template_half_win) if len(trusted) >= 8 else None
    if template is None:
        log.warning("_composite_qrs_score: template QRS non construit "
                     "(%d battement(s) de confiance, ≥8 requis) — score "
                     "composite calculé sans critère de forme (NCC).", len(trusted))
        z_ncc = np.zeros(n)
        ncc_raw = np.zeros(n)
    else:
        ncc_raw = _template_ncc_scores(signal, peaks, template, template_half_win)
        # ncc_raw vit déjà sur une échelle bornée et interprétable [-1, 1] —
        # PAS de z-score MAD ici. Les vrais QRS se regroupent très
        # étroitement près de 1.0 (corrélation quasi parfaite avec le
        # template), ce qui rend la MAD de la population minuscule ; un
        # z-score MAD transformerait alors une corrélation excellente
        # (ex. 0.986) en un score fortement négatif simplement parce
        # qu'elle est *légèrement* sous la médiane d'un groupe très
        # compact — punissant à tort d'excellents battements. Mise à
        # l'échelle directe et fixe à la place.
        z_ncc = ncc_raw * 3.0

    # ── Poids : NCC template et largeur/pente en tête (descripteurs les plus
    # discriminants, cf. littérature citée), symétrie/courbure en soutien. ──
    score = (
        2.0 * z_ncc +
        1.5 * z_width +
        1.5 * z_slope +
        1.25 * z_align +
        1.0 * z_ratio +
        1.0 * z_prom +
        1.0 * (persistence_frac * 4.0 - 2.0) +  # recentré ~[-2,2] pour peser comme les z-scores
        0.5 * z_curv +
        0.25 * z_asym
    )

    details = dict(
        prominences=prominences, persistence_frac=persistence_frac,
        width50=morph["width50"], width_height_ratio=morph["width_height_ratio"],
        slope_up=morph["slope_up"], curvature=morph["curvature"],
        align_mad=align_mad, ncc=ncc_raw, template=template,
        n_trusted=len(trusted),
    )
    return score, details


# ════════════════════════════════════════════════════════════
#  DÉTECTEUR 1 — WAVELET TRANSFORM (CWT)
# ════════════════════════════════════════════════════════════

def detect_peaks_wavelet(
    signal: np.ndarray,
    fs: float,
    wavelet: str = "mexh",
    qrs_width_ms: float = 8.0,
    min_rr_ms: float = MouseECG.MIN_RR_MS,
    peak_distance_ms: float = MouseECG.PEAK_DISTANCE_MS,
    threshold_k: float = 5.0,
    min_scale_persistence_frac: float = 0.35,
    snap_max_ms: float = 10.0,
    n_scales: int = 8,
    scales_qrs: "Optional[tuple[float, float]]" = None,
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Detect R-peaks via Continuous Wavelet Transform (CWT).

    Principe de séparation spectrale
    ─────────────────────────────────
    Le CWT décompose le signal en trois régimes :

    • Petites échelles (≪ échelle QRS)  → bruit haute-fréquence,
      EMG diaphragmatique — atténuées en douceur.
    • Échelles centrées sur l'échelle QRS physiologique (calculée depuis
      *qrs_width_ms*, indépendamment de fs) → complexe QRS, upstroke R :
      **poids maximal**.
    • Grandes échelles (≫ échelle QRS) → J-wave, T-wave, onde lente —
      atténuées en douceur.

    Contrairement à une bande passe-bande à coupure dure (ancienne version :
    scales_qrs=(1,4) codé en dur pour un fs implicite), la séparation est ici
    une pondération gaussienne continue en log-échelle — pas d'effet de bord,
    et le détecteur s'auto-calibre pour n'importe quel fs (1–10 kHz testé)
    à partir d'une seule grandeur physiologique (largeur de QRS attendue).

    Pipeline
    ────────
    1. Échelles CWT calibrées physiquement (_cwt_scales_for_qrs) — indépendant
       de fs, cf. Speerschneider & Thomas 2013 pour la largeur de QRS murine.
    2. CWT (pywt, ondelette *mexh* = chapeau mexicain, dérivée seconde de
       gaussienne — forme optimale pour un pic R isolé, cf. Li, Zheng & Tai
       1995, "Detection of ECG characteristic points using wavelet
       transforms").
    3. Énergie multi-échelle normalisée MAD + pondérée en log-échelle
       (_weighted_multiscale_energy) — cf. docstring de cette fonction.
    4. Persistance multi-échelle : chaque échelle vote indépendamment (son
       propre seuil médiane+MAD) — utilisée comme UN des 8 descripteurs du
       score composite (étape 8), pas comme filtre dur isolé.
    5. Seuil robuste médiane + k×MAD (_robust_threshold) sur l'énergie —
       funnel large, volontairement permissif (voir étape 6).
    6. find_peaks avec seulement height + distance — un funnel LÂCHE.
       Contrairement aux versions précédentes, prominence/largeur/persistance
       ne sont PLUS des seuils durs ici : c'est le score composite (étape 8),
       calculé après snap sur chaque candidat survivant, qui fait la
       discrimination réelle. Un funnel trop strict à ce stade rejetterait
       des candidats avant même qu'ils aient une chance d'être correctement
       évalués par la fusion de descripteurs.
    7. Snap vers le maximum local du signal original (borné à *snap_max_ms*).
    8. **Score composite** (_composite_qrs_score) — fusion pondérée de 8
       descripteurs z-normalisés (énergie, persistance, prominence, largeur,
       pente, courbure, alignement multi-échelle des maxima du module CWT,
       corrélation avec un template QRS adaptatif) en un seul score de
       confiance, seuillé une seule fois (médiane − k×MAD). Remplace la
       cascade de seuils durs des versions précédentes — cf. Elgendi,
       Jonkman & De Boer 2013 sur les limites des cascades de seuils
       indépendants. Détails scientifiques complets dans la docstring de
       _composite_qrs_score et de ses sous-fonctions
       (_modulus_maxima_alignment, _build_qrs_template,
       _extended_morphology_descriptors).
    9. Disambiguation R-vs-J (resolve_r_vs_j_peaks, partagée avec le
       détecteur SG-dérivée) sur les candidats ayant passé le score.

    Validation RR (intervalles physiologiquement impossibles, doublets
    résiduels, récupération de battements manqués) : **volontairement PAS
    dupliquée ici**. Ce rôle est déjà couvert par les fonctions partagées
    ``detect_rr_artifacts`` / ``correct_rr_artifacts`` / ``recover_missed_beats``,
    appelées en aval par l'orchestrateur (après le choix du détecteur, quel
    qu'il soit) — les réimplémenter dans chaque détecteur recréerait
    exactement la duplication à trois voies qui rend ce module difficile à
    maintenir. ``detect_peaks_wavelet`` reste focalisé sur un seul rôle :
    produire de bons candidats R-peak + une prominence topographique
    cohérente avec les 3 autres détecteurs (cf. _topographic_prominences).

    Pourquoi cette version rejette mieux les événements respiratoires
    ──────────────────────────────────────────────────────────────────
    Un burst EMG diaphragmatique peut, individuellement, avoir une énergie
    ondelette comparable à un QRS, une largeur comparable, voire une pente
    montante comparable — c'est précisément pourquoi une cascade de seuils
    indépendants (versions précédentes) pouvait le laisser passer : il
    suffisait de franchir CHAQUE seuil, jamais d'être comparé comme un TOUT
    à ce à quoi ressemble un QRS. Le score composite fusionne l'évidence :
    un artefact respiratoire domine rarement sur les 8 descripteurs à la
    fois, notamment sur les deux qui ciblent spécifiquement le caractère
    "concentré vs diffus" d'un événement (alignement multi-échelle des
    maxima du module CWT, rapport largeur/hauteur) et sur la ressemblance de
    forme avec le template QRS adaptatif — un burst EMG large-bande n'a
    quasiment jamais une corrélation normalisée élevée avec un QRS moyen,
    même quand son énergie ou sa largeur individuelle sont dans la plage
    plausible.
    

    Parameters
    ----------
    signal          : Signal ECG normalisé, corrigé en polarité.
    fs              : Fréquence d'échantillonnage (Hz). Aucune borne codée
                      en dur — testé 1000/2000/5000/10000 Hz.
    wavelet         : Ondelette pywt (défaut : 'mexh').
    qrs_width_ms    : Largeur de QRS attendue (ms). Détermine entièrement
                      les échelles CWT, le critère de largeur find_peaks,
                      et le centre de la pondération log-gaussienne.
                      Défaut 8 ms (QRS murin conscient, cf. Speerschneider
                      & Thomas 2013) — à ajuster si anesthésie/pathologie
                      élargit significativement le QRS.
    min_rr_ms       : Distance minimale entre pics après snap (ms).
    peak_distance_ms: Distance find_peaks (doit rester < min_rr_ms — cf.
                      commentaire MouseECG.PEAK_DISTANCE_MS).
    threshold_k     : k dans seuil = médiane + k×MAD de l'énergie.
    min_scale_persistence_frac: fraction minimale des échelles sur
                      lesquelles un candidat doit être actif.
    snap_max_ms     : Borne physio du snap vers le max local (ms).
    n_scales        : Nombre d'échelles CWT échantillonnées.
    scales_qrs      : Rétro-compatibilité — si fourni, (min, max)
                      d'échelles CWT explicites, prioritaire sur
                      qrs_width_ms (ancien comportement, pour appelants
                      existants qui dépendent d'échelles fixées à la main).

    Returns
    -------
    peaks      : Indices des R-peaks dans le signal original.
    prominences: Prominence topographique de chaque R-peak (scipy
                 peak_prominences — même convention que les 3 autres
                 détecteurs, cf. _topographic_prominences).
    thresh_amp : P10 des prominences (diagnostic interne ; le tracé du
                 seuil dans l'UI utilise le calcul propre d'apply_threshold).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 10:
        # Same floor and same graceful empty-result contract as
        # detect_peaks_sg_derivative/detect_peaks_envelope_max -- this
        # detector was missing it, and pywt.cwt() raises ValueError on an
        # empty array instead of returning "no peaks found", which crashed
        # the whole detection pipeline instead of just finding nothing.
        return np.array([], dtype=int), np.array([]), 0.0

    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "pywt est requis pour detect_peaks_wavelet : pip install PyWavelets"
        ) from exc

    # ── 1. Échelles CWT ──────────────────────────────────────────────────
    if scales_qrs is not None:
        # Rétro-compatibilité : échelles explicites, comportement pré-existant.
        scale_min = max(1.0, scales_qrs[0])
        scale_max = max(scale_min + 0.5, scales_qrs[1])
        scales = np.linspace(scale_min, scale_max, n_scales)
        scale_center = float(np.sqrt(scale_min * scale_max))  # centre géométrique
    else:
        scales, scale_center = _cwt_scales_for_qrs(
            wavelet, fs, qrs_width_ms, n_scales=n_scales)

    coeffs, _ = pywt.cwt(signal, scales, wavelet)   # (n_scales, n_samples)

    # ── 2-3. Énergie multi-échelle normalisée + pondérée ─────────────────
    energy, weights = _weighted_multiscale_energy(coeffs, scales, scale_center)

    # ── 4. Persistance multi-échelle ─────────────────────────────────────
    # Chaque échelle vote avec SON PROPRE seuil robuste (une échelle bruitée
    # ne doit pas imposer son échelle de bruit aux autres).
    per_scale_thresh = np.array([_robust_threshold(coeffs[s] ** 2, k=3.0)
                                  for s in range(len(scales))])
    persistence = ((coeffs ** 2) > per_scale_thresh[:, None]).sum(axis=0)
    min_scales_required = max(1, int(round(min_scale_persistence_frac * len(scales))))

    # ── 5. Seuil robuste sur l'énergie pondérée ──────────────────────────
    thresh = _robust_threshold(energy, k=threshold_k)
    log.debug(
        "detect_peaks_wavelet: scale_center=%.2f  scales=%.1f–%.1f  "
        "thresh(med+%.1fxMAD)=%.6f  persist_min=%d/%d",
        scale_center, scales[0], scales[-1], threshold_k, thresh,
        min_scales_required, len(scales),
    )

    # ── 6. find_peaks — funnel volontairement LÂCHE ──────────────────────
    # L'ancienne version filtrait ici par prominence + persistance en
    # cascade (deux seuils durs indépendants). Ce find_peaks ne sert plus
    # qu'à produire un ENSEMBLE DE CANDIDATS gérable (pas littéralement
    # chaque échantillon) ; c'est le score composite ci-dessous, calculé
    # APRÈS snap, qui fait la discrimination réelle en fusionnant 8
    # descripteurs au lieu de les tester un par un.
    width_min_samples = max(1, int(round(scale_center * 0.5)))
    min_dist = max(1, int(peak_distance_ms / 1000.0 * fs))

    peaks_e, _ = find_peaks(energy, height=thresh, distance=min_dist)

    if len(peaks_e) == 0:
        log.warning("detect_peaks_wavelet: aucun pic trouvé (seuil trop élevé ?)")
        return np.array([], dtype=int), np.array([]), 0.0

    # ── 7. Snap vers le maximum local du signal original ──────────────────
    half_snap = max(1, int(snap_max_ms / 1000.0 * fs))
    peaks_out = np.array([_snap_to_r(int(p), signal, fs, half_snap) for p in peaks_e])
    peaks_out = np.unique(peaks_out)
    if len(peaks_out) > 1:
        keep = [0]
        for j in range(1, len(peaks_out)):
            if peaks_out[j] - peaks_out[keep[-1]] >= min_dist:
                keep.append(j)
        peaks_out = peaks_out[keep]

    # ── 8. Score composite (fusion) ────────────────────────────────────────
    # search_samples pour l'alignement multi-échelle et demi-fenêtre du
    # template : dérivés de scale_center / qrs_width_ms, pas de constante
    # arbitraire — cf. _composite_qrs_score.
    align_search_samples = max(2, int(round(scale_center * 0.5)))
    template_half_win    = max(3, int(round(1.5 * qrs_width_ms / 1000.0 * fs)))

    score, details = _composite_qrs_score(
        signal, peaks_out, coeffs, scales, energy, persistence, len(scales),
        fs, scale_center, align_search_samples, template_half_win,
    )
    if len(score) == 0:
        return np.array([], dtype=int), np.array([]), 0.0

    # Seuil unique et robuste sur le score fusionné (médiane − k×MAD : on
    # rejette la queue basse, symétrique à la correction du seuil de
    # prominence — cf. _robust_threshold). k plus généreux qu'un critère
    # isolé car le score agrège déjà 8 preuves indépendantes : un vrai QRS
    # domine sur la plupart d'entre elles simultanément, un artefact
    # respiratoire n'en domine généralement qu'une ou deux.
    score_floor = _robust_threshold(score, k=-1.25, positive_only=False)
    accept = score >= score_floor
    peaks_out = peaks_out[accept]

    if len(peaks_out) == 0:
        log.warning("detect_peaks_wavelet: tous les candidats rejetés par le score composite")
        return np.array([], dtype=int), np.array([]), 0.0

    # ── 9. Disambiguation R-vs-J (partagée) ────────────────────────────────
    peaks_out = resolve_r_vs_j_peaks(peaks_out, signal, fs, min_rr_ms=min_rr_ms)

    prominences = _topographic_prominences(signal, peaks_out, fs)
    thresh_amp  = float(np.percentile(prominences, 10)) if len(prominences) else 0.0

    log.info(
        "detect_peaks_wavelet: %d peaks  fs=%.0f  qrs_width=%.1fms  "
        "scale_center=%.2f  n_trusted_template=%d  score_floor=%.2f",
        len(peaks_out), fs, qrs_width_ms, scale_center,
        details.get("n_trusted", 0), score_floor,
    )
    return peaks_out, prominences, thresh_amp


# ════════════════════════════════════════════════════════════════════════════
#  DÉTECTEUR 2 — SAVITZKY-GOLAY + DÉRIVÉE
# ════════════════════════════════════════════════════════════════════════════

def _estimate_dominant_rr(
    deriv_pos: np.ndarray,
    fs: float,
    min_rr_ms: float = 40.0,
    max_rr_ms: float = 400.0,
    anchor_multiplier: float = 0.15,
    rr_scale: float = 0.65,
) -> float:
    """Estime le RR dominant (ms) depuis les ancres de dérivée FILTRÉES.

    Utilise uniquement les ancres dont la hauteur de dérivée dépasse le P75
    des ancres brutes — élimine le bruit et les J-waves avant de mesurer le RR.
    Retourne 65 % du RR médian comme distance effective pour find_peaks.
    """
    dist_min = max(1, int(min_rr_ms / 1000.0 * fs))
    rough, _ = find_peaks(deriv_pos, distance=dist_min)
    if len(rough) < 4:
        return min_rr_ms
    heights = deriv_pos[rough]
    # Seuil P75 : sépare les vraies ancres R (dérivée forte) du bruit
    ref = float(np.percentile(heights, 75))
    real_anchors = rough[heights >= anchor_multiplier * ref]
    if len(real_anchors) < 4:
        real_anchors = rough  # fallback
    rr_ms = float(np.median(np.diff(real_anchors))) / fs * 1000.0
    # Auto-clamp : à très haute FC (≥ 800 bpm, RR ≈ 75 ms), rr_scale × RR
    # peut tomber sous min_rr_ms (= peak_distance_ms passé par l'appelant,
    # soit 40 ms) et provoquer des fusions de pics.
    # On garantit que la distance effective est au moins min_rr_ms.
    scaled = max(rr_ms * rr_scale, min_rr_ms)
    return float(np.clip(scaled, min_rr_ms, max_rr_ms))


def detect_peaks_sg_derivative(
    signal: np.ndarray,
    fs: float,
    min_rr_ms: float = MouseECG.MIN_RR_MS,
    peak_distance_ms: float = MouseECG.PEAK_DISTANCE_MS,
    threshold_factor: float = 0.25,
    sg_window_ms: float = 20.0,
    sg_polyorder: int = 3,
    snap_window_ms: float = 8.0,
    anchor_multiplier: float = 0.15,
    rr_scale: float = 0.65,
    target_fs: float | None = None,
    # Strictness params to prefer true R peaks over J/P/T/noise
    min_upstroke_frac: float = 0.45,
    min_peak_amp_frac: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Détecte les R-peaks par filtre Savitzky-Golay + première dérivée.

    Pipeline en 6 étapes
    ────────────────────
    1. Dérivée SG (fenêtre 20 ms).

    2. Seuil bimodal robuste.
       Les ancres de dérivée forment deux populations bien séparées :
       – Vrais R : dérivée très forte (dépolarisation ventriculaire rapide)
       – Bruit/J  : dérivée faible (~8–10× plus petite)
       On utilise P75 des ancres brutes × threshold_factor comme seuil.
       Avec P75 la séparation est parfaite quelle que soit la valeur de
       threshold_factor (0.05 à 0.90 donne le même résultat sur ce signal).

    3. Estimation adaptive du RR (sur ancres filtrées uniquement).

    4. Détection finale des ancres avec seuil et distance adaptatifs.

    5. Snap vers le sommet R — orienté ARRIÈRE puis avant.
       Le max de dérivée SG tombe SUR ou APRÈS le début de l'upstroke.
       Le sommet R est toujours dans les 0–5 ms AVANT ou les 0–5 ms APRÈS
       le max de dérivée. La J-wave est ≥ 15 ms après le R.
       Fenêtre asymétrique : [-snap, +snap/2] → priorité à l'arrière.

    6. Nettoyage doublons + resolve_r_vs_j_peaks.

    Paramètres
    ----------
    signal           : Signal ECG normalisé, R positifs.
    fs               : Fréquence d'échantillonnage (Hz).
    min_rr_ms        : RR minimum physiologique (défaut 67 ms).
    peak_distance_ms : Distance initiale pour estimation RR (défaut 40 ms).
    threshold_factor : Fraction de P75 des ancres (défaut 0.15).
                       Robuste : 0.05–0.90 donnent le même résultat si le
                       signal a une bimodalité R/bruit claire.
    sg_window_ms     : Fenêtre SG (ms, défaut 20).
    sg_polyorder     : Ordre SG (défaut 3).
    snap_window_ms   : Fenêtre de snap (ms, défaut 8).
                       Asymétrique : [-snap, +snap/2].
    """
    signal_orig = np.asarray(signal, dtype=np.float64)
    orig_fs = fs
    signal = signal_orig
    downsample_scale = 1.0

    # Only downsample when the signal is actually ABOVE target_fs.
    # If orig_fs ≤ target_fs (e.g. 1 kHz signal with target_fs=2 kHz) we keep
    # the signal as-is AND force downsample_scale=1.0 so that anchor→snap index
    # mapping is the identity.  The old code stored scale=target_fs/orig_fs=2.0
    # which doubled every anchor index, sending snap completely out of bounds.
    if target_fs is not None and target_fs > 0 and fs > target_fs:
        signal_ds, ds_fs = downsample_signal(signal_orig, fs, target_fs)
        if ds_fs < fs and len(signal_ds) >= 10:
            signal = signal_ds
            fs = ds_fs
            downsample_scale = float(orig_fs) / float(ds_fs)

    n = len(signal)
    if n < 10:
        return np.array([], dtype=int), np.array([]), 0.0

    # ── 1. Dérivée SG ──────────────────────────────────────────────────────
    # Cap strategy — FIXED physiological constant, independent of min_rr_ms:
    #   SG_WIN_CAP_MS = 10 ms  ≈ mouse QRS duration (5–15 ms, Nerbonne 2005).
    #   At 1 kHz, the default 20 ms = 21 samples spans the entire QRS complex,
    #   attenuating the derivative ×33 and shifting its peak 5 samples early.
    #   The cap is never tied to min_rr_ms (user-adjustable) to avoid collapse
    #   when non-physiological values (e.g. 1 ms) are entered by the user.
    SG_WIN_CAP_MS: float = 10.0
    sg_win_ms_effective = min(sg_window_ms, SG_WIN_CAP_MS)
    sg_win = max(sg_polyorder + 2, int(sg_win_ms_effective / 1000.0 * fs))
    if sg_win % 2 == 0:
        sg_win += 1
    sg_win = min(sg_win, n if n % 2 == 1 else n - 1)

    # ── Baseline wander correction (for derivative only) ───────────────────
    # Respiratory baseline wander (1–5 Hz, amplitude up to 2× QRS in mice)
    # creates large slow slopes in the SG derivative that:
    #   (a) inflate P75 of the bimodal threshold, and
    #   (b) generate false anchor candidates at respiratory frequency.
    # Both effects cause genuine R-peaks coinciding with a respiratory event
    # to fall below the detection threshold.
    #
    # Fix: if significant baseline wander is detected (resp/QRS power ratio > 0.5),
    # apply a Butterworth HP at 3 Hz to a COPY of the signal used ONLY for the
    # SG derivative step.  All other steps (snap, post-filter, proms) continue
    # to use signal_orig (amplitudes unaffected).
    sig_for_deriv: np.ndarray = signal
    if _detect_baseline_wander(signal, fs):
        sig_for_deriv = _hp_filter_for_deriv(signal, fs, cutoff_hz=3.0)
        log.info(
            "detect_peaks_sg_derivative: baseline wander detected "
            "→ HP 3 Hz applied for derivative only (signal_orig unchanged)"
        )

    deriv = savgol_filter(sig_for_deriv, window_length=sg_win,
                          polyorder=sg_polyorder, deriv=1, delta=1.0 / fs)
    deriv_arr = np.asarray(deriv)
    # clamp negative derivative values to zero
    deriv_pos = np.maximum(deriv_arr, 0.0)

    # ── 2. Seuil bimodal robuste (P75 des ancres brutes) ──────────────────
    dist_init = max(1, int(peak_distance_ms / 1000.0 * fs))
    rough, _  = find_peaks(deriv_pos, distance=dist_init)

    if len(rough) >= 4:
        heights      = deriv_pos[rough]
        ref_p75      = float(np.percentile(heights, 75))
        thresh_deriv = threshold_factor * ref_p75
    else:
        pos = deriv_pos[deriv_pos > 0]
        if len(pos):
            ref_p75 = float(np.percentile(pos, 75))
            thresh_deriv = threshold_factor * ref_p75
        else:
            ref_p75 = 0.0
            thresh_deriv = 0.0

    log.debug(
        "detect_peaks_sg_derivative: P75_deriv=%.4f  thresh=%.4f  "
        "factor=%.2f  n_rough=%d  fs=%.0f",
        ref_p75, thresh_deriv, threshold_factor, len(rough), fs,
    )

    # ── 3. Estimation adaptive du RR sur ancres filtrées ──────────────────
    dist_adaptive_ms = _estimate_dominant_rr(
        deriv_pos,
        fs,
        min_rr_ms=peak_distance_ms,
        max_rr_ms=400.0,
        anchor_multiplier=anchor_multiplier,
        rr_scale=rr_scale,
    )
    dist_samp = max(1, int(dist_adaptive_ms / 1000.0 * fs))

    log.debug(
        "detect_peaks_sg_derivative: dist_adaptive=%.1f ms (%d samp)",
        dist_adaptive_ms, dist_samp,
    )

    # ── 4. Détection finale des ancres ────────────────────────────────────
    anchors, _ = find_peaks(deriv_pos, height=thresh_deriv, distance=dist_samp)

    if len(anchors) == 0:
        log.warning(
            "detect_peaks_sg_derivative: aucune ancre "
            "(thresh=%.4f, factor=%.2f). Réduire threshold_factor ?",
            thresh_deriv, threshold_factor,
        )
        return np.array([], dtype=int), np.array([]), 0.0

    log.debug("detect_peaks_sg_derivative: %d ancres", len(anchors))

    # ── 5. Snap vers le sommet R ──────────────────────────────────────────
    # Fenêtre ASYMÉTRIQUE [-snap, +snap/2] :
    # Le sommet R est typiquement AVANT ou sur le max de dérivée SG (le
    # filtre SG lissé introduit un léger retard vers l'avant).
    # La J-wave arrive ≥ 15 ms après le R → exclue par la fenêtre courte
    # vers l'avant (+snap/2 = +4 ms par défaut).
    # pre_samp/post_samp are applied to `signal_orig` (via _snap_to_r below),
    # so they must be computed in ORIGINAL-fs samples, not the (possibly
    # downsampled) `fs`. Using `fs` here made the snap window too narrow by
    # exactly `downsample_scale` whenever downsampling was active — narrow
    # enough that the search window sometimes didn't reach the true R-peak
    # at all, silently falling back to a point still on the rising edge.
    pre_samp  = max(1, int(snap_window_ms / 1000.0 * orig_fs))        # 8 ms en arrière
    post_samp = max(1, int(snap_window_ms / 2.0 / 1000.0 * orig_fs))  # 4 ms en avant

    # Si le signal a été sous-échantillonné, la dérivée SG est dans l'espace
    # downsample.  On la réinterpolent sur la longueur de signal_orig pour
    # conserver l'avantage du snap upstroke-first même après downsampling.
    if downsample_scale != 1.0 and len(deriv_arr) != len(signal_orig):
        from scipy.interpolate import interp1d as _interp1d
        x_ds  = np.linspace(0, len(signal_orig) - 1, len(deriv_arr))
        x_orig = np.arange(len(signal_orig))
        _f    = _interp1d(x_ds, deriv_arr, kind="linear",
                          bounds_error=False, fill_value=0.0)
        deriv_for_snap = _f(x_orig)
    else:
        deriv_for_snap = deriv_arr  # identité si pas de downsampling

    r_peaks = np.empty(len(anchors), dtype=int)
    for i, anc in enumerate(anchors):
        approx = int(round(anc * downsample_scale))
        r_peaks[i] = _snap_to_r(
            approx=approx,
            signal=signal_orig,
            fs=orig_fs,
            pre_snap=pre_samp,
            post_snap=post_samp,
            deriv=deriv_for_snap,  # toujours fourni (orig ou réinterpolé)
        )

    # ── 6. Nettoyage ──────────────────────────────────────────────────────
    r_peaks     = np.unique(r_peaks)
    # r_peaks live in ORIGINAL-fs sample space (from _snap_to_r above) — use
    # orig_fs here, not the possibly-downsampled `fs`, for the same reason
    # as pre_samp/post_samp above.
    min_rr_samp = max(1, int(min_rr_ms / 1000.0 * orig_fs))

    if len(r_peaks) > 1:
        keep = [0]
        for j in range(1, len(r_peaks)):
            if r_peaks[j] - r_peaks[keep[-1]] >= min_rr_samp:
                keep.append(j)
            else:
                # signal_orig/orig_fs — r_peaks indices are original-space,
                # indexing the (possibly shorter, downsampled) `signal`
                # array with them would read the wrong samples entirely.
                sl_old = _upstroke_slope_at(r_peaks[keep[-1]], signal_orig, orig_fs)
                sl_new = _upstroke_slope_at(r_peaks[j],        signal_orig, orig_fs)
                if sl_new > sl_old:
                    keep[-1] = j
        r_peaks = r_peaks[keep]

    r_peaks = resolve_r_vs_j_peaks(r_peaks, signal_orig, orig_fs, min_rr_ms=min_rr_ms)
    # Post-filter: require sufficiently steep upstroke and sufficient amplitude,
    # judged against LOCAL-WINDOW medians (±10 battements) rather than global
    # ones — avoids false rejections in low-amplitude segments caused by
    # electrode drift, motion artifact, or anesthesia transitions.
    # Only this detector (detect_peaks_sg_derivative) calls
    # _local_morphology_filter — detect_peaks_wavelet uses its own
    # composite-score pipeline instead and never reaches this helper
    # (width/asymmetry criteria left disabled here — identical behaviour to
    # before).
    if len(r_peaks):
        r_peaks = _local_morphology_filter(
            r_peaks, signal_orig, orig_fs,
            min_upstroke_frac=min_upstroke_frac,
            min_amp_frac=min_peak_amp_frac,
            window_beats=10,
            deriv=deriv_arr if downsample_scale == 1.0 else None,
        )
        prominences = _topographic_prominences(signal_orig, r_peaks, orig_fs)
    else:
        prominences = np.array([])
    thresh_amp  = float(np.percentile(prominences, 10)) if len(prominences) else 0.0

    log.info(
        "detect_peaks_sg_derivative: %d R-peaks  fs=%.0f Hz  "
        "sg_win=%d samp  dist_adaptive=%.1f ms  thresh=%.4f",
        len(r_peaks), fs, sg_win, dist_adaptive_ms, thresh_deriv,
    )
    return r_peaks, prominences, thresh_amp


# ════════════════════════════════════════════════════════════
#  DÉTECTEUR 3 — ENVELOPPE LOCALE / MAXIMUM ABSOLU
# ════════════════════════════════════════════════════════════

def detect_peaks_envelope_max(
    signal:            np.ndarray,
    fs:                float,
    min_rr_ms:         float = MouseECG.MIN_RR_MS,
    peak_distance_ms:  float = MouseECG.PEAK_DISTANCE_MS,
    prominence_frac:   float = 0.30,
    plateau_snap:      bool  = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Détecte les R-peaks par recherche du maximum local d'amplitude.

    Principe
    ────────
    Contrairement au détecteur SG (basé sur la dérivée) et au détecteur
    wavelet (basé sur l'énergie QRS), ce détecteur utilise directement
    l'**amplitude absolue** du signal : chaque R-peak est le maximum local
    dans une fenêtre d'au moins `min_rr_ms`.

    Cas d'usage principal : signaux **saturés (ADC clipping)**
    ──────────────────────────────────────────────────────────
    Quand le gain d'acquisition est trop élevé, le R-peak sature à la
    valeur max de l'ADC (ex. ±4.775 V normalisé).  La dérivée SG sur un
    plateau de saturation est nulle → le détecteur SG rate ces pics ou
    les place à l'arête du plateau.  Ce détecteur localise chaque plateau
    de saturation et place le pic au **milieu** (point de référence stable,
    indépendant de la durée de clipping et de la valeur exacte de
    saturation).

    Cas d'usage secondaire : morphologies atypiques / faible SNR
    ─────────────────────────────────────────────────────────────
    Sur des signaux très filtrés, à faible fréquence d'échantillonnage
    (< 500 Hz), ou à morphologie atypique (bloc de branche, stimulé,
    anesthésie profonde), l'upstroke QRS peut être trop lent pour la
    détection par dérivée.  L'amplitude reste un discriminant fiable.

    Pipeline
    ────────
    1. **Candidats bruts** : `scipy.find_peaks` avec distance minimale
       `peak_distance_ms`.  Produit tous les maxima locaux au-dessus du
       bruit de fond.

    2. **Seuil d'amplitude bimodal** : P90 des amplitudes aux candidats ×
       `prominence_frac` (défaut 0.30).  Sur un signal ECG normal, les
       R-peaks forment une population haute bien séparée des ondes T/P/bruit
       (typiquement 3–6× plus basses).  Le seuil rejette la population basse.

       Choix de P90 (et non P75 ou médiane) : résiste à la modulation
       respiratoire (jusqu'à 30 % d'amplitude fluctuante sur les R) sans
       laisser entrer les ondes T ou les artéfacts.

    3. **Snap au milieu du plateau** (si `plateau_snap=True`) :
       Pour chaque pic retenu, si le sample adjacent à la même amplitude
       est détecté (plateau de saturation ou quantification grossière),
       on calcule l'étendue exacte du plateau et on renvoie son milieu.
       Cela donne une position de R-peak cohérente quel que soit le côté
       sur lequel `find_peaks` a placé le marqueur.

    4. **Unicité + distance minimale** : après snap, tri, dédoublonnage,
       et ré-application de la contrainte de distance.  En cas de conflit
       (deux pics trop proches), le plus haut est conservé.

    5. **R vs J disambiguation** : `resolve_r_vs_j_peaks` finale pour
       écarter les ondes J résiduelles (identique aux autres détecteurs).

    Paramètres
    ----------
    signal            : Signal ECG normalisé, R positifs (après fix_polarity).
    fs                : Fréquence d'échantillonnage (Hz).
    min_rr_ms         : RR minimum physiologique (ms, défaut MouseECG.MIN_RR_MS).
    peak_distance_ms  : Distance initiale `find_peaks` (ms, défaut
                        MouseECG.PEAK_DISTANCE_MS ≈ 40 ms).  Peut être ≤ min_rr_ms
                        pour un premier balayage large, la contrainte min_rr_ms
                        est ensuite ré-appliquée après snap.
    prominence_frac   : Fraction de P90 utilisée comme seuil d'amplitude
                        (défaut 0.30).  Augmenter (0.40–0.50) sur signaux très
                        bruités ; diminuer (0.15–0.20) pour récupérer des R-peaks
                        très atténués (modulation respiratoire sévère).
    plateau_snap      : Si True (défaut), snappe chaque pic au milieu de son
                        plateau de saturation.  Mettre False uniquement pour
                        comparaison ou débogage.

    Retour
    ------
    tuple (r_peaks, prominences, thresh_amp) :
        r_peaks       : np.ndarray[int] — indices des R-peaks dans *signal*.
        prominences   : np.ndarray[float] — prominence topographique (scipy
                        peak_prominences) de chaque R-peak, PAS l'amplitude
                        brute. Même convention que les 3 autres détecteurs
                        et que le chemin "auto" — nécessaire pour que le
                        curseur Threshold unique de l'UI ait un sens
                        cohérent quel que soit le détecteur actif.
        thresh_amp    : float — P10 des prominences (diagnostic interne ;
                        non utilisé pour tracer le seuil dans l'UI — ce
                        tracé utilise ``apply_threshold``'s propre calcul
                        sur l'amplitude des pics acceptés).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 10:
        return np.array([], dtype=int), np.array([]), 0.0

    dist_samp = max(1, int(peak_distance_ms / 1000.0 * fs))
    min_dist  = max(1, int(min_rr_ms        / 1000.0 * fs))

    # ── 1. Candidats bruts (tous maxima locaux séparés d'au moins peak_distance_ms)
    raw_peaks, _ = find_peaks(signal, distance=dist_samp, prominence=0,
                               wlen=_prominence_wlen_samples(fs))
    if len(raw_peaks) < 3:
        log.warning(
            "detect_peaks_envelope_max: trop peu de candidats bruts (%d) "
            "— vérifier la polarité ou réduire peak_distance_ms",
            len(raw_peaks),
        )
        return (raw_peaks,
                signal[raw_peaks] if len(raw_peaks) else np.array([]),
                0.0)

    # ── 2. Seuil d'amplitude : P90 × prominence_frac
    amp_at_cands = signal[raw_peaks]
    amp_p90      = float(np.percentile(amp_at_cands, 90))
    amp_thresh   = prominence_frac * amp_p90

    good = raw_peaks[amp_at_cands >= amp_thresh]
    if len(good) < 3:
        log.warning(
            "detect_peaks_envelope_max: seuil amplitude trop strict "
            "(P90=%.3f × frac=%.2f = %.3f) — %d candidats retenus. "
            "Réduire prominence_frac ?",
            amp_p90, prominence_frac, amp_thresh, len(good),
        )
        good = raw_peaks  # fallback : garder tous les candidats bruts

    log.debug(
        "detect_peaks_envelope_max: %d candidats bruts → %d après seuil "
        "P90×%.2f = %.3f",
        len(raw_peaks), len(good), prominence_frac, amp_thresh,
    )

    # ── 3. Snap au milieu du plateau de saturation ────────────────────────
    # Tolérance : 0.01 % de la plage dynamique — couvre la quantification
    # sur des ADC 12–16 bits ainsi que les erreurs float64 d'arrondi.
    sig_range  = signal.max() - signal.min() + 1e-12
    plat_tol   = max(1e-9, 1e-4 * sig_range)
    sig_max    = float(signal.max())

    snapped = np.empty(len(good), dtype=int)
    for i, pk in enumerate(good):
        pk_val = float(signal[pk])

        if plateau_snap and abs(pk_val - sig_max) < plat_tol:
            # Plateau de saturation haute — chercher l'étendue exacte
            lo = int(pk)
            while lo > 0 and abs(float(signal[lo - 1]) - pk_val) < plat_tol:
                lo -= 1
            hi = int(pk)
            while hi < n - 1 and abs(float(signal[hi + 1]) - pk_val) < plat_tol:
                hi += 1
            snapped[i] = (lo + hi) // 2

        elif plateau_snap and pk > 0 and pk < n - 1:
            # Plateau non saturé (ex. quantification grossière à basse résolution)
            pk_left = pk
            while pk_left > 0 and abs(float(signal[pk_left - 1]) - pk_val) < plat_tol:
                pk_left -= 1
            pk_right = pk
            while pk_right < n - 1 and abs(float(signal[pk_right + 1]) - pk_val) < plat_tol:
                pk_right += 1
            snapped[i] = (pk_left + pk_right) // 2

        else:
            snapped[i] = int(pk)

    # ── 4. Unicité + contrainte distance ─────────────────────────────────
    snapped = np.sort(np.unique(snapped))

    keep = np.ones(len(snapped), dtype=bool)
    for i in range(1, len(snapped)):
        if not keep[i - 1]:
            continue
        if snapped[i] - snapped[i - 1] < min_dist:
            # Conserver le plus haut des deux
            if signal[snapped[i]] >= signal[snapped[i - 1]]:
                keep[i - 1] = False
            else:
                keep[i]     = False
    snapped = snapped[keep]

    # ── 5. R vs J disambiguation ──────────────────────────────────────────
    r_peaks = resolve_r_vs_j_peaks(snapped, signal, fs, min_rr_ms=min_rr_ms)

    prominences = _topographic_prominences(signal, r_peaks, fs) if len(r_peaks) else np.array([])
    thresh_amp  = float(np.percentile(prominences, 10)) if len(prominences) else 0.0

    log.info(
        "detect_peaks_envelope_max: %d R-peaks  fs=%.0f Hz  "
        "seuil_amp=%.3f  (P90=%.3f × frac=%.2f)  "
        "plateau_snap=%s",
        len(r_peaks), fs, amp_thresh, amp_p90, prominence_frac, plateau_snap,
    )
    return r_peaks, prominences, thresh_amp




# detect_rr_artifacts Pass 2: how many times the premature-run re-check may
# extend a run of flagged ectopic beats (see the loop at the end of Pass 2).
# One pass already catches every beat of a couplet/triplet (its inner beats
# are flagged as "both intervals short"); the bound only stops a sustained
# rate change from cascading through many beats.
_ECTOPIC_RUN_ITERATIONS = 3


def detect_rr_artifacts(
    rpeaks:        np.ndarray,
    fs:            float,
    rr_min_ms:     float = MouseECG.RR_MIN_MS,
    rr_max_ms:     float = MouseECG.RR_MAX_MS,
    window_beats:  int   = 11,
    dev_threshold: float = 0.20,
    signal:        Optional[np.ndarray] = None,
) -> list[dict]:
    """Detect artifact candidates in an R-peak array WITHOUT removing anything.

    Pré-passe 0 : doublets (gap < rr_min_ms).
                  Si *signal* est fourni : élimine le pic de moindre amplitude
                  (le vrai R-peak est conservé, quel que soit sa position dans
                  la paire). Sinon (signal=None, comportement historique) :
                  élimine systématiquement le second pic de la paire — un
                  choix arbitraire qui peut conserver un artefact au lieu du
                  vrai R-peak si celui-ci se trouve être le second. Passer
                  *signal* (le tracé filtré déjà disponible chez l'appelant)
                  pour un résultat correct ; le paramètre reste optionnel
                  uniquement pour ne pas casser d'éventuels appels existants
                  qui n'ont pas le signal sous la main.
    Pass 1       : bornes physiologiques (nonphysio) — flags only the too-short
                  side (gap < rr_min_ms, likely a spurious extra peak). A gap
                  > rr_max_ms flags nothing: it almost always means a missed
                  beat or genuine pause/AV-block, not a spurious flanking
                  peak, so there is no peak here whose removal fixes anything.
    Pass 2       : ectopic — SHORT-side evidence only, against the local
                  median RR (ref, FIX ⑥ without rr_bi bias). An interval is
                  "too short" when < ref × (1 − dev_threshold) and "long"
                  when > ref × (1 + dev_threshold); intervals touching a
                  peak already flagged by Pass 0/1 are already accounted
                  for. Each too-short interval is attributed to ONE of its
                  two beats:
                    • a beat whose arrival AND departure are both too short
                      (a spurious peak splitting a normal interval, or an
                      inner beat of a premature run) is flagged;
                    • the FIRST interval of the recording, when short, is
                      ambiguous (beat 0 has no arrival interval): beat 1 is
                      flagged only if its departure is long (a compensatory
                      pause — beat 1 was premature), otherwise beat 0 is (a
                      spurious leading detection, e.g. a filter-edge
                      transient). The LAST interval mirrors this: beat n−2
                      is flagged only if its arrival is long, otherwise the
                      last beat is;
                    • otherwise the beat ENDING the short interval — the
                      premature beat — is flagged, unless the previous beat
                      was just flagged (its removal already absorbs it);
                    • premature runs: that skip is then re-checked against
                      the updated sequence (up to _ECTOPIC_RUN_ITERATIONS
                      times). A skipped beat whose own arrival was short is
                      flagged as well when the interval it would arrive with
                      after its flagged predecessor(s) are removed is LONG —
                      the predecessor was the first beat of a couplet /
                      triplet, not a spurious split (whose removal leaves a
                      normal interval) — unless that merged arrival spans
                      about two beats or more (≥ 2·ref·(1 − dev_threshold))
                      with no compensatory (long) departure after it, i.e.
                      the beat is back on the rhythm. Every beat of
                      [80, 80, 160] or [80, 80, 80, 200] among 120 ms beats
                      is caught.
                  A beat whose only short interval is its departure is the
                  normal beat before a premature one and is never flagged
                  (except at the recording edges, above).
                  deviation = (ref − shortest attributed interval) / ref.
                  A LONG interval alone never flags anything (a pause, a
                  missed beat, the compensatory pause after a premature
                  beat): removing a beat can only lengthen it further.
                  Missed beats are handled by recover_missed_beats(), pauses
                  are surfaced by classify_arrhythmias(), and analyse_core()
                  excludes the out-of-range interval from HRV without
                  bridging it -- and, given apply_artifact_decisions()'s
                  break_samples as breaks, also the interval left
                  spanning each removed real (premature) beat.

    Each candidate dict: index, sample, type, rr_prev_ms, rr_next_ms,
    rr_ref_ms, deviation, decision ("remove") and rr_long_ms -- for an
    ectopic candidate the length above which an interval counts as long
    against its local reference (rr_ref_ms × (1 + dev_threshold)), NaN
    otherwise. apply_artifact_decisions() uses it to tell a removed real
    beat (the interval spanning it is long) from a removed spurious split
    (the merged interval is normal).
    """
    rpeaks = np.sort(np.asarray(rpeaks, dtype=int).flatten())
    n      = len(rpeaks)
    if n < 4:
        return []

    rr_ms   = np.diff(rpeaks).astype(np.float64) / fs * 1000
    flagged: dict[int, dict] = {}

    def _add(i: int, kind: str, deviation: float, ref: float,
             rr_long: float = float("nan")) -> None:
        if i in flagged:
            return
        rr_prev = float(rr_ms[i - 1]) if i > 0     else float("nan")
        rr_next = float(rr_ms[i])     if i < n - 1 else float("nan")
        flagged[i] = dict(
            index=i, sample=int(rpeaks[i]), type=kind,
            rr_prev_ms=rr_prev, rr_next_ms=rr_next,
            rr_ref_ms=ref, deviation=deviation, decision="remove",
            rr_long_ms=rr_long,
        )

    # Pré-passe 0 : doublets
    for k in range(n - 1):
        if k in flagged:
            continue
        if rr_ms[k] < rr_min_ms:
            dev = round(abs(rr_ms[k] - rr_min_ms) / max(rr_min_ms, 1), 3)
            if signal is not None:
                # Keep the higher-amplitude peak — it's the more likely true
                # R-peak. Without this, the second peak of every close pair
                # was discarded unconditionally, which could throw away the
                # real R-peak and keep a spurious artifact if the artifact
                # happened to come first.
                amp_k, amp_k1 = float(signal[rpeaks[k]]), float(signal[rpeaks[k + 1]])
                loser = k if amp_k < amp_k1 else k + 1
            else:
                loser = k + 1
            _add(loser, "duplicate", dev, float("nan"))
            log.debug(
                "detect_rr_artifacts pre-pass: doublet gap=%.1f ms → discard peak[%d]",
                rr_ms[k], loser,
            )

    # Pass 1 : bornes physio
    for k in range(n - 1):
        if k in flagged or k + 1 in flagged:
            continue
        gap = rr_ms[k]
        if rr_min_ms <= gap <= rr_max_ms:
            continue
        if gap < rr_min_ms:
            dev = abs(gap - rr_min_ms) / max(rr_min_ms, 1)
            # Too-short gap: most likely an extra/duplicate detection next
            # to the true beat -- removing one peak plausibly fixes it.
            _add(k + 1, "nonphysio", round(dev, 3), float("nan"))
        # else: gap > rr_max_ms. This almost always means a genuinely missed
        # beat, sinus pause, or AV-block-like event -- neither flanking peak
        # is itself spurious, so there is no peak here whose removal would
        # "fix" anything. Flagging both used to auto-delete two real R-peaks
        # via correct_rr_artifacts() whenever artifact correction was
        # enabled, destroying the very pause/AV-block event a user would
        # want to see. Left unflagged; classify_arrhythmias() already
        # surfaces overlong gaps as "pause" events for review.

    # Pass 2 : ectopique (médiane locale sans biais rr_bi) — short side only
    half         = window_beats // 2
    flagged_mask = np.zeros(n, dtype=bool)
    for idx in flagged:
        flagged_mask[idx] = True

    # Local reference RR per unflagged beat (median of its neighbours' RRs)
    refs = np.full(n, np.nan)
    for i in range(n):
        if flagged_mask[i]:
            continue
        lo  = max(0, i - half)
        hi  = min(n, i + half + 1)
        nbr = [j for j in range(lo, hi) if j != i and not flagged_mask[j]]
        if len(nbr) < 3:
            continue
        nbr_rr = []
        for j in nbr:
            if j > 0 and not flagged_mask[j - 1]:
                nbr_rr.append(rr_ms[j - 1])
            if j < n - 1 and not flagged_mask[j]:
                nbr_rr.append(rr_ms[j])
        if len(nbr_rr) < 3:
            continue
        ref = float(np.median(nbr_rr))
        if ref >= 1e-3:
            refs[i] = ref

    # arr_short[i] / dep_short[i]: beat i's arrival (rr_ms[i-1]) / departure
    # (rr_ms[i]) interval is too short against beat i's own reference.
    # Intervals touching a Pass 0/1 peak are excluded (already accounted
    # for); NaN refs compare False, so beats without a reference never flag.
    short_lim = refs * (1.0 - dev_threshold)
    long_lim  = refs * (1.0 + dev_threshold)
    open_iv   = ~(flagged_mask[:-1] | flagged_mask[1:])
    arr_short = np.zeros(n, dtype=bool)
    dep_short = np.zeros(n, dtype=bool)
    arr_short[1:]  = open_iv & (rr_ms < short_lim[1:])
    dep_short[:-1] = open_iv & (rr_ms < short_lim[:-1])

    def _add_ectopic(i: int, short_iv: "Optional[float]" = None,
                     ref: "Optional[float]" = None) -> None:
        # short_iv/ref: the attributed short interval and its reference when
        # beat i's own masks don't hold them (recording-edge cases below).
        if short_iv is None:
            short_iv = min(
                rr_ms[i - 1] if arr_short[i] else np.inf,
                rr_ms[i]     if dep_short[i] else np.inf,
            )
        if ref is None or not np.isfinite(ref):
            ref = refs[i]
        dev = (ref - short_iv) / ref
        _add(i, "ectopic", round(float(dev), 3), round(float(ref), 1),
             rr_long=round(float(ref) * (1.0 + dev_threshold), 1))
        flagged_mask[i] = True

    # Both intervals short: this beat alone explains both of them.
    for i in np.flatnonzero(arr_short & dep_short):
        _add_ectopic(int(i))

    # Recording edges. A short FIRST interval is ambiguous -- beat 0 has no
    # arrival interval, so blaming the beat that ends it (the rule below)
    # deleted the real beat 1 whenever beat 0 itself was spurious (e.g. a
    # filter-edge transient). Beat 1 is premature only if a compensatory
    # (long) departure follows it; otherwise beat 0 goes. The LAST interval
    # mirrors this: beat n-2 goes only if it arrived after a long interval
    # (it is then the displaced one between two on-rhythm beats), otherwise
    # the last beat does (premature or a spurious trailing detection).
    if open_iv[0] and (arr_short[1] or dep_short[0]) \
            and not (flagged_mask[0] or flagged_mask[1]):
        ref_edge = refs[1] if np.isfinite(refs[1]) else refs[0]
        beat = 1 if rr_ms[1] > long_lim[1] else 0
        _add_ectopic(beat, float(rr_ms[0]), ref_edge)
    last = n - 2   # index of the last interval (beats n-2 → n-1)
    if open_iv[last] and (dep_short[n - 2] or arr_short[n - 1]) \
            and not (flagged_mask[n - 2] or flagged_mask[n - 1]):
        ref_edge = refs[n - 2] if np.isfinite(refs[n - 2]) else refs[n - 1]
        beat = n - 2 if rr_ms[last - 1] > long_lim[n - 2] else n - 1
        _add_ectopic(beat, float(rr_ms[last]), ref_edge)

    # Premature arrival: the beat ending the short interval, unless the
    # previous beat is already flagged (its removal absorbs the interval).
    for i in np.flatnonzero(arr_short):
        if flagged_mask[i] or flagged_mask[i - 1]:
            continue
        _add_ectopic(int(i))

    # Premature runs: re-check the beats skipped above against the updated
    # sequence. Removing a spurious peak that split a normal interval leaves
    # a NORMAL merged interval, so the next beat is genuine; removing the
    # first beat of a couplet/triplet leaves a LONG one (two short coupling
    # intervals merged), so the next beat is premature too and must go as
    # well -- otherwise the surviving premature beat and its compensatory
    # pause enter HRV as if they were normal intervals. A merged arrival of
    # about two beats or more is the exception: the beat is then back on
    # the rhythm (as after a single displaced beat) unless a compensatory
    # (long) departure shows it still belongs to the run -- e.g. the last
    # beat of [80, 80, 80, 200]. Each iteration can extend a run by one
    # beat; bounded so a sustained rate change can't cascade.
    double_lim = 2.0 * refs * (1.0 - dev_threshold)
    for _ in range(_ECTOPIC_RUN_ITERATIONS):
        newly: list[int] = []
        for i in np.flatnonzero(arr_short & ~flagged_mask):
            if not flagged_mask[i - 1]:
                continue
            j = i - 1
            while j >= 0 and flagged_mask[j]:
                j -= 1
            if j < 0:
                continue   # nothing left before it (beat 0 was the spurious one)
            merged_ms = float(rpeaks[i] - rpeaks[j]) / fs * 1000
            if merged_ms <= long_lim[i]:
                continue   # predecessor was a spurious split: this beat is normal
            comp_pause = i < n - 1 and rr_ms[i] > long_lim[i]
            if merged_ms < double_lim[i] or comp_pause:
                newly.append(int(i))
        if not newly:
            break
        for i in newly:
            _add_ectopic(i)

    candidates = sorted(flagged.values(), key=lambda c: c["index"])
    log.info(
        "detect_rr_artifacts: %d candidates  (duplicate=%d  nonphysio=%d  ectopic=%d)",
        len(candidates),
        sum(1 for c in candidates if c["type"] == "duplicate"),
        sum(1 for c in candidates if c["type"] == "nonphysio"),
        sum(1 for c in candidates if c["type"] == "ectopic"),
    )
    return candidates


def _removal_is_break(cand: dict, corrected: np.ndarray,
                      fs: Optional[float]) -> bool:
    """True when removing *cand* deleted a REAL beat, so the interval now
    spanning it is not a normal-to-normal interval (an HRV break).

    Only an ectopic removal can be one. A duplicate / non-physiological
    peak (closer than Min R-R to another) is a spurious detection: removing
    it gives back the true sinus-to-sinus interval, exactly like a manual
    deletion in edit mode. An ectopic candidate is a real premature beat
    when the interval spanning it in the *corrected* peaks is LONG against
    its local reference (``rr_long_ms``: coupling + compensatory interval,
    ≈ 2 RR); when that merged interval is normal the removed peak was a
    spurious split of one sinus interval, not a beat. A removed peak with
    no kept peak on one side (recording edge) is spanned by no interval.
    Without ``rr_long_ms`` or *fs* every ectopic removal is a break -- the
    conservative choice: one interval too many excluded, never a non-NN
    interval let into HRV.
    """
    if cand.get("type") != "ectopic":
        return False
    s = int(cand["sample"])
    j = int(np.searchsorted(corrected, s))
    if j == 0 or j >= len(corrected):
        return False
    rr_long = cand.get("rr_long_ms")
    try:
        rr_long = float(rr_long) if rr_long is not None else float("nan")
    except (TypeError, ValueError):
        rr_long = float("nan")
    if fs is None or not np.isfinite(rr_long) or float(fs) <= 0:
        return True
    span_ms = float(corrected[j] - corrected[j - 1]) / float(fs) * 1000.0
    return span_ms > rr_long


def removed_ectopic_samples(
    report:      "Optional[dict]",
    rpeaks:      "Optional[np.ndarray]",
    tol_samples: int = 1,
) -> np.ndarray:
    """Samples of the peaks an artifact *report* removed as "ectopic" that
    are still absent from *rpeaks* (active_removed_samples) -- the beats
    classify_arrhythmias(removed_beats=...) puts back so premature beats
    removed by automatic correction are still counted. Reports saved
    before removed_types existed name no type: nothing is returned
    (removed_beats_for_classification re-examines those, and the very
    premature beats removed as duplicate / non-physiological)."""
    if not isinstance(report, dict) or rpeaks is None:
        return np.array([], dtype=np.int64)
    samples = [int(x) for x in (report.get("removed_samples") or [])]
    types = list(report.get("removed_types") or [])
    if len(types) != len(samples):
        return np.array([], dtype=np.int64)
    ect = [x for x, t in zip(samples, types) if str(t) == "ectopic"]
    return active_removed_samples(ect, np.asarray(rpeaks), tol_samples)


# A removed duplicate / non-physiological peak (or one of unknown type) is
# put back for the premature-beat classification when the interval left
# spanning it is LONG against the local RR -- the removal deleted a real
# beat (a very premature one, coupling < Min R-R) -- and the closest gap it
# makes is at least this fraction of that local RR: a spurious double
# detection a few ms from its R (or next to a genuine pause) is not.
_PUTBACK_LONG_FACTOR       = 1.20   # same "long" bound as rr_long_ms (1 + 0.20)
_PUTBACK_MIN_GAP_FRACTION  = 0.35


def _span_is_deleted_beat(span_lo: int, span_hi: int, inner: "list[int]",
                          diffs: np.ndarray, k: int) -> bool:
    """True when the kept-peak interval [span_lo, span_hi] (index *k* in
    *diffs*, the intervals of the kept peaks), from which the *inner*
    peaks were removed, is long against the 10 intervals around it -- or
    is followed by a long one (a compensatory pause: the kept second beat
    of a couplet whose first beat alone was removed) -- and no gap the
    inner peaks would make is below _PUTBACK_MIN_GAP_FRACTION of that
    reference -- i.e. the removed peaks were real beats."""
    nbr = np.concatenate([diffs[max(0, k - 5):k], diffs[k + 1:k + 6]])
    if len(nbr) == 0:
        return False
    ref = float(np.median(nbr))
    if ref <= 0:
        return False
    long_lim = _PUTBACK_LONG_FACTOR * ref
    comp = k + 1 < len(diffs) and float(diffs[k + 1]) > long_lim
    if (span_hi - span_lo) <= long_lim and not comp:
        return False
    pts = np.array([span_lo, *sorted(inner), span_hi], dtype=np.int64)
    return float(np.diff(pts).min()) >= _PUTBACK_MIN_GAP_FRACTION * ref


def removed_beats_for_classification(
    report:      "Optional[dict]",
    rpeaks:      "Optional[np.ndarray]",
    tol_samples: int = 1,
) -> "tuple[np.ndarray, dict]":
    """The peaks artifact correction removed that classify_arrhythmias
    (removed_beats=...) must put back so the premature beats among them
    are still counted, and how the others were judged.

    Every removal still in effect (active_removed_samples) is examined:
      * type "ectopic"           -> put back (as removed_ectopic_samples);
      * "duplicate" / "nonphysio" (an interval < Min R-R, ~67 ms) -> put
        back only when the interval left spanning it in *rpeaks* is long
        (> 1.2 x the median of the 10 intervals around it), or the next one
        is (a compensatory pause), and the closest
        gap it makes is >= 0.35 x that median: a very premature REAL beat
        (coupling below Min R-R) whose removal merged coupling and
        compensatory intervals. A spurious double detection leaves a normal
        interval (or sits a few ms from its R) and is not put back;
      * no recorded type (sessions saved before removed_types existed) ->
        the same interval test, whatever the original type was.

    Returns (samples, info). info: n_put_back, n_spurious (removals judged
    spurious detections and NOT counted), n_unknown_type (removals of
    unknown type, re-examined by the interval test) and
    n_unknown_put_back. *rpeaks* are the current accepted peaks."""
    info = dict(n_put_back=0, n_spurious=0, n_unknown_type=0, n_unknown_put_back=0)
    empty = np.array([], dtype=np.int64)
    if not isinstance(report, dict) or rpeaks is None:
        return empty, info
    samples = [int(x) for x in (report.get("removed_samples") or [])]
    if not samples:
        return empty, info
    types = [str(t) for t in (report.get("removed_types") or [])]
    if len(types) != len(samples):
        types = ["unknown"] * len(samples)
    type_of = dict(zip(samples, types))
    rp = np.sort(np.asarray(rpeaks, dtype=np.int64).flatten())
    active = active_removed_samples(samples, rp, tol_samples)
    if len(active) == 0 or len(rp) < 2:
        return empty, info
    diffs = np.diff(rp)

    # Removed peaks grouped by the kept-peak interval spanning them, so a
    # removed couplet is judged on its whole merged interval.
    spans: "dict[int, list[int]]" = {}
    put_back: "list[int]" = []
    for s in active.tolist():
        t = type_of.get(int(s), "unknown")
        if t == "unknown":
            info["n_unknown_type"] += 1
        if t == "ectopic":
            put_back.append(int(s))
            continue
        j = int(np.searchsorted(rp, s))
        if j == 0 or j >= len(rp):
            if t != "unknown":
                info["n_spurious"] += 1
            continue                       # recording edge: spans no interval
        spans.setdefault(j - 1, []).append(int(s))
    for k, inner in spans.items():
        if _span_is_deleted_beat(int(rp[k]), int(rp[k + 1]), inner, diffs, k):
            put_back.extend(inner)
            info["n_unknown_put_back"] += sum(
                1 for s in inner if type_of.get(s, "unknown") == "unknown")
        else:
            info["n_spurious"] += sum(
                1 for s in inner if type_of.get(s, "unknown") != "unknown")
    info["n_put_back"] = len(put_back)
    return np.array(sorted(put_back), dtype=np.int64), info


def _artifact_counts(types: "list[str]") -> dict:
    return dict(
        n_duplicate = sum(1 for t in types if t == "duplicate"),
        n_nonphysio = sum(1 for t in types if t == "nonphysio"),
        n_ectopic   = sum(1 for t in types if t == "ectopic"),
    )


def apply_artifact_decisions(
    rpeaks:     np.ndarray,
    candidates: list[dict],
    fs:         Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """Apply accept/remove decisions from a review pass.

    Report keys:
        n_in, n_out      : peak counts before / after
        removed_samples  : sorted list of int sample indices of every peak
                           actually removed (JSON-safe)
        removed_types    : the candidate type of each, same order
        break_samples    : the subset whose removal deleted a REAL beat --
                           the BREAKS the HRV functions need
                           (``rr_interval_series(..., breaks=...)``): the
                           interval left spanning a removed premature beat
                           is not a normal-to-normal interval and is
                           excluded from HRV rather than treated as one.
                           Removing a spurious detection (duplicate, non-
                           physiological, split) restores a true NN
                           interval and is not a break (see
                           _removal_is_break; *fs* is needed to tell a
                           split from a premature beat).
        n_duplicate, n_nonphysio, n_ectopic : removed peaks per type
        n_kept           : candidates the user chose to keep
    """
    rpeaks = np.asarray(rpeaks)
    remove_by_sample = {int(c["sample"]): c for c in candidates if c["decision"] == "remove"}
    keep_mask  = np.array([int(p) not in remove_by_sample for p in rpeaks], dtype=bool)
    corrected  = rpeaks[keep_mask]
    corr_sorted = np.sort(corrected.astype(np.int64))
    removed_samples = sorted(int(p) for p in rpeaks[~keep_mask])
    removed_types   = [str(remove_by_sample[s].get("type", "unknown")) for s in removed_samples]
    break_samples   = [s for s in removed_samples
                       if _removal_is_break(remove_by_sample[s], corr_sorted, fs)]
    report = dict(
        n_in            = len(rpeaks),
        n_out           = int(keep_mask.sum()),
        removed_samples = removed_samples,
        removed_types   = removed_types,
        break_samples   = break_samples,
        **_artifact_counts(removed_types),
        n_kept          = sum(1 for c in candidates if c["decision"] == "keep"),
    )
    log.info(
        "apply_artifact_decisions: %d → %d peaks  (−%d removed, %d HRV breaks, "
        "%d kept by user)",
        report["n_in"], report["n_out"], report["n_in"] - report["n_out"],
        len(break_samples), report["n_kept"],
    )
    return corrected, report


def _report_removals(report: "Optional[dict]") -> "dict[int, tuple[str, bool]]":
    """{sample: (type, is_break)} of an artifact report's removed peaks.
    Reports saved before removed_types / break_samples existed carry no
    type ("unknown") and no break."""
    if not report:
        return {}
    samples = [int(s) for s in (report.get("removed_samples") or [])]
    types   = list(report.get("removed_types") or [])
    if len(types) != len(samples):
        types = ["unknown"] * len(samples)
    breaks  = {int(s) for s in (report.get("break_samples") or [])}
    return {s: (str(t), s in breaks) for s, t in zip(samples, types)}


def merge_artifact_reports(
    prev:        "Optional[dict]",
    new:         "Optional[dict]",
    rpeaks:      np.ndarray,
    tol_samples: int = 1,
) -> "Optional[dict]":
    """Cumulative artifact report: the removals of *prev* and *new* (an
    earlier pass and this one) whose peak is still absent from *rpeaks* --
    the current, full corrected peak set -- with every count recomputed
    from them, so the displayed "removed" figures and the HRV breaks always
    describe the same removals.

    A second auto-correction pass on already-corrected peaks removes
    nothing new, and a reviewed removal stays in effect after a later
    Analyze; both used to reset the counts to the last pass only while the
    breaks persisted. A removed peak within *tol_samples* of a current peak
    is back (re-detected, or re-added by hand) and is dropped.

    n_out = len(rpeaks), n_in = n_out + the removals still in effect;
    n_kept comes from *new* when given. Returns None when there is neither
    report, or when *new* is None and none of *prev*'s removals remain
    (a stale report). With *new* given the report is kept even with
    nothing removed ("correction applied, no artifact").
    """
    if prev is None and new is None:
        return None
    removals = _report_removals(prev)
    removals.update(_report_removals(new))
    rp = np.asarray(rpeaks, dtype=np.int64).flatten() if rpeaks is not None else np.array([], np.int64)
    still = active_removed_samples(list(removals), rp, tol_samples).tolist()
    if new is None and not still:
        return None
    types = [removals[s][0] for s in still]
    base  = dict(new if new is not None else prev)   # type: ignore[arg-type]
    base.update(
        n_out           = int(len(rp)),
        n_in            = int(len(rp)) + len(still),
        removed_samples = [int(s) for s in still],
        removed_types   = types,
        break_samples   = [int(s) for s in still if removals[s][1]],
        **_artifact_counts(types),
    )
    base.setdefault("n_kept", 0)
    return base


def drop_removed_peaks(
    rpeaks:          np.ndarray,
    removed_samples: "Optional[Iterable[int]]",
    tol_samples:     int = 1,
    protected:       "Optional[Iterable[int]]" = None,
) -> np.ndarray:
    """Re-apply earlier artifact removals to a freshly re-detected peak set
    (session restore re-runs detection from the full candidate set, which
    brings every removed beat back). For each removed sample the NEAREST
    peak within *tol_samples* is dropped -- only that one, so a kept peak
    a few samples from a removed duplicate survives. Peaks in *protected*
    (added by hand) are never dropped. Returns the sorted remaining peaks.
    """
    rp = np.sort(np.asarray(rpeaks, dtype=np.int64).flatten())
    if removed_samples is None or len(rp) == 0:
        return rp
    rs = np.unique(np.asarray(list(removed_samples), dtype=np.int64).flatten())
    if len(rs) == 0:
        return rp
    keep = np.ones(len(rp), dtype=bool)
    prot = {int(p) for p in (protected or [])}
    right = np.clip(np.searchsorted(rp, rs), 0, len(rp) - 1)
    left  = np.clip(right - 1, 0, len(rp) - 1)
    nearest = np.where(np.abs(rp[left] - rs) <= np.abs(rp[right] - rs), left, right)
    for s, k in zip(rs, nearest):
        if abs(int(rp[k]) - int(s)) <= tol_samples and int(rp[k]) not in prot:
            keep[k] = False
    return rp[keep]


def active_removed_samples(
    removed_samples: "Optional[Iterable[int]]",
    rpeaks:          np.ndarray,
    tol_samples:     int = 1,
) -> np.ndarray:
    """The removed-peak samples (an artifact report's ``removed_samples``
    or ``break_samples``) whose peak is still ABSENT from *rpeaks* --
    sorted, unique, int64.

    A removed beat can come back: every threshold change or manual edit
    re-runs detection from the full candidate set. A removed sample within
    *tol_samples* of a current peak is therefore dropped -- that beat is
    back, so no interval spans it and it must not be used as an HRV break.
    """
    if removed_samples is None:
        return np.array([], dtype=np.int64)
    rs = np.unique(np.asarray(list(removed_samples), dtype=np.int64).flatten())
    rp = np.sort(np.asarray(rpeaks, dtype=np.int64).flatten())
    if len(rs) == 0 or len(rp) == 0:
        return rs
    right = np.clip(np.searchsorted(rp, rs), 0, len(rp) - 1)
    left  = np.clip(right - 1, 0, len(rp) - 1)
    nearest = np.minimum(np.abs(rp[right] - rs), np.abs(rp[left] - rs))
    return rs[nearest > tol_samples]


def correct_rr_artifacts(
    rpeaks:        np.ndarray,
    fs:            float,
    rr_min_ms:     float = MouseECG.RR_MIN_MS,
    rr_max_ms:     float = MouseECG.RR_MAX_MS,
    window_beats:  int   = 11,
    dev_threshold: float = 0.20,
    signal:        Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Auto-correct : detect + apply default 'remove' decisions."""
    candidates = detect_rr_artifacts(rpeaks, fs, rr_min_ms, rr_max_ms,
                                     window_beats, dev_threshold, signal=signal)
    return apply_artifact_decisions(rpeaks, candidates, fs=fs)


# ════════════════════════════════════════════════════════════
#  MISSED BEAT RECOVERY
# ════════════════════════════════════════════════════════════

def recover_missed_beats(
    rpeaks:              np.ndarray,
    signal:              np.ndarray,
    fs:                  float,
    rr_long_factor:      float = 1.75,
    window_beats:        int   = 11,
    search_margin_ms:    float = 15.0,
    min_peak_height_pct: float = 0.40,
) -> tuple[np.ndarray, int]:
    """Search for missed R-peaks in suspiciously long RR intervals.

    Pour chaque intervalle RR > rr_long_factor × médiane locale, subdivise
    l'intervalle en N sous-fenêtres et cherche un maximum local suffisamment
    haut (> min_peak_height_pct × amplitude médiane des pics existants).
    """
    rpeaks = np.sort(np.asarray(rpeaks, dtype=int).flatten())
    n      = len(rpeaks)
    if n < 4:
        return rpeaks, 0

    rr_ms         = np.diff(rpeaks).astype(np.float64) / fs * 1000
    half          = window_beats // 2
    margin_samp   = max(1, int(search_margin_ms / 1000.0 * fs))
    amp_threshold = float(np.median(signal[rpeaks])) * min_peak_height_pct
    recovered: list[int] = []

    for i in range(n - 1):
        lo_nb  = max(0, i - half)
        hi_nb  = min(n - 1, i + half + 1)
        nbr_rr = [rr_ms[j] for j in range(lo_nb, hi_nb) if j != i]
        if len(nbr_rr) < 3:
            continue
        local_med = float(np.median(nbr_rr))
        if local_med < 1.0 or rr_ms[i] < rr_long_factor * local_med:
            continue

        n_missed = round(rr_ms[i] / local_med) - 1
        if n_missed < 1 or n_missed > 4:
            continue

        step_samp = int((rpeaks[i + 1] - rpeaks[i]) / (n_missed + 1))
        for k in range(1, n_missed + 1):
            expected  = rpeaks[i] + k * step_samp
            # _snap_to_r (score composite upstroke-first) plutôt qu'argmax
            # naïf : évite de sélectionner une J-wave si elle se trouve dans
            # la fenêtre ±margin_samp autour du battement interpolé.
            candidate = _snap_to_r(
                approx=expected,
                signal=signal,
                fs=fs,
                pre_snap=margin_samp,
            )
            if signal[candidate] < amp_threshold:
                log.debug(
                    "recover_missed_beats: rejected candidate at %d "
                    "(amp=%.4f < threshold=%.4f)",
                    candidate, signal[candidate], amp_threshold,
                )
                continue
            if np.any(np.abs(rpeaks - candidate) < margin_samp):
                continue
            recovered.append(candidate)
            log.debug(
                "recover_missed_beats: recovered beat at sample %d "
                "(RR_long=%.1f ms  local_med=%.1f ms  n_missed=%d)",
                candidate, rr_ms[i], local_med, n_missed,
            )

    if not recovered:
        return rpeaks, 0

    rpeaks_out = np.sort(np.concatenate([rpeaks, recovered]))
    log.info(
        "recover_missed_beats: inserted %d beat(s)  %d → %d peaks",
        len(recovered), n, len(rpeaks_out),
    )
    return rpeaks_out, len(recovered)


# ════════════════════════════════════════════════════════════
#  ARRHYTHMIA CLASSIFICATION
# ════════════════════════════════════════════════════════════

# Fixed thresholds -- these used to come from the (now-removed)
# experimental-context selector's default "telemetry_awake" context, so
# bradycardia/pause and AV-delay detection sensitivity is unchanged.
_BRADY_RR_HI_HR_LO_BPM = 450.0   # bradycardia/pause RR-interval cutoff basis
# The AV-delay cutoff is the upper end of the mouse PR reference range
# (MouseECG.PR_NORMAL), which is defined P onset -> QRS onset -- the same
# definition analyse_intervals now measures (older versions measured PR to
# the Q nadir, a few ms longer, so slightly more beats crossed it).
_AV_DELAY_PR_HI_MS     = 55.0    # sustained-PR-prolongation cutoff
# A beat is premature when the RR interval it ENDS is shorter than this
# fraction of the reference RR: the median of the PREMATURE_REF_INTERVALS
# intervals before the group's first premature beat and, whichever is
# smaller, of the PREMATURE_REF_INTERVALS intervals from the one the rhythm
# returns on (see _premature_runs). Timing only -- see
# models.PREMATURE_DEFINITION_TEXT.
PREMATURE_RR_FRACTION   = 0.70
PREMATURE_REF_INTERVALS = 10
# Longest group of consecutive premature beats reported. A longer stretch
# of short intervals is a rate change, left to the tachycardia rule -- and
# it bounds the false run a 2:1 block followed by 1:1 conduction gives.
PREMATURE_MAX_RUN       = 30


def _premature_runs(rr_ms: np.ndarray) -> "list[tuple[int, int, float]]":
    """Groups of consecutive premature intervals in *rr_ms*: (start, end,
    reference RR) with intervals start..end-1 premature, i.e. beats
    start+1..end.

    The onset reference is the median of the PREMATURE_REF_INTERVALS
    intervals BEFORE the candidate interval (fewer at the recording start;
    with fewer than 3, a centred window of 10 neighbours). The group
    extends while intervals stay < PREMATURE_RR_FRACTION x that reference
    and must END: the rhythm returns (an interval >= the bound) within
    PREMATURE_MAX_RUN intervals, and every interval of the group is also
    < PREMATURE_RR_FRACTION x the median of the PREMATURE_REF_INTERVALS
    intervals from that return interval on. So a run is shorter than the
    rhythm on BOTH sides of it, however long it is -- the centred
    10-neighbour median used before was itself pulled down by a run of 6
    or more, which then went undetected -- while a sustained rate step (no
    return) or a 1:1 stretch after a slower one (return check fails) is
    not a run; the candidates inside a stretch that never returns are
    skipped with it. A group still going at the end of the intervals is
    not reported (it never returns)."""
    rr = np.asarray(rr_ms, dtype=float)
    n = len(rr)
    k = PREMATURE_REF_INTERVALS
    if n < 2:
        return []
    before = np.full(n, np.nan)
    after = np.full(n, np.nan)
    if n >= k:
        med = np.median(np.lib.stride_tricks.sliding_window_view(rr, k), axis=1)
        before[k:] = med[:n - k]           # median(rr[i-k:i])
        after[:n - k + 1] = med            # median(rr[j:j+k])
    for i in range(min(k, n)):
        if i >= 3:
            before[i] = float(np.median(rr[:i]))
        else:
            w = np.concatenate([rr[max(0, i - 5):i], rr[i + 1:i + 6]])
            before[i] = float(np.median(w)) if len(w) else np.nan
    for j in range(max(0, n - k + 1), n):
        after[j] = float(np.median(rr[j:]))

    frac = PREMATURE_RR_FRACTION
    runs: "list[tuple[int, int, float]]" = []
    next_free = 0
    with np.errstate(invalid="ignore"):
        cand = np.flatnonzero(rr < frac * before)
    for i in cand.tolist():
        if i < next_free:
            continue
        bound = frac * before[i]
        j, stop = i + 1, min(n, i + PREMATURE_MAX_RUN)
        while j < stop and rr[j] < bound:
            j += 1
        if j >= n or rr[j] < bound:
            # Never returns: a rate change. Its later beats are not run
            # onsets either (each would see the same slower rhythm before
            # it and report the tail of the stretch as a run).
            next_free = j
            continue
        if float(rr[i:j].max()) >= frac * after[j]:
            continue                       # not shorter than what follows
        runs.append((i, j, float(min(before[i], after[j]))))
        next_free = j
    return runs


def _baseline_rr_mask(t_peaks: np.ndarray,
                      baseline_s: float) -> "tuple[np.ndarray, float]":
    """RR intervals classify_arrhythmias() takes its baseline HR from.

    t_peaks : sorted peak times (s). Returns (mask over the
    len(t_peaks) - 1 RR intervals, effective window length in s).

    The window is measured from the FIRST peak passed in, not from t=0:
    callers pass the analysis-window subset of the recording, whose first
    beat can sit minutes in. An absolute ``t < baseline_s`` test then
    matched no beat at all and the baseline silently became the median of
    every interval -- inverting brady/tachy calls on a recording that slows
    down or speeds up inside the window.
    """
    t_rel  = t_peaks[:-1] - float(t_peaks[0])
    span_s = float(t_peaks[-1] - t_peaks[0])
    bl_len = float(baseline_s)
    mask   = t_rel < bl_len
    if mask.sum() < 10:
        # Too few beats for a stable median -- widen to 20% of the span.
        bl_len = max(span_s * 0.20, float(baseline_s))
        mask   = t_rel < bl_len
    if mask.sum() < 5:
        # Still too few: the baseline is the median of every interval.
        mask   = np.ones(len(t_rel), dtype=bool)
        bl_len = span_s
    return mask, bl_len


# The baseline is only a meaningful reference when the analysed beats span
# at least this many baseline windows: below that the baseline window covers
# most (or all) of what is being classified, so a slowdown or speed-up
# inside it drags the reference along and brady/tachy calls get muted or
# inverted (e.g. a 25 s window with a 30 s baseline compares every beat with
# the median of the whole window, event included). Classification still
# runs; the baseline is flagged unreliable and the UI says so.
_BASELINE_MIN_SPAN_FACTOR = 2.0


def arrhythmia_baseline_info(
    rpeaks:     np.ndarray,
    fs:         float,
    baseline_s: float = 30.0,
) -> dict:
    """The baseline classify_arrhythmias() uses for the same rpeaks / fs /
    baseline_s, so a caller can shade exactly that window and warn when it
    is not a meaningful reference.

    Returns a dict with:
        t_start, t_end : absolute window (s) -- starts at the first peak
                         passed in; t_end is clamped to the last peak
        length_s       : effective window length (baseline_s, or widened)
        span_s         : analysed span, first to last peak (s)
        n_intervals    : RR intervals in the window
        baseline_rr_ms, baseline_hr : the baseline itself
        reliable       : False when span_s < _BASELINE_MIN_SPAN_FACTOR ×
                         length_s, i.e. the baseline window covers most of
                         the analysed beats (always False when it fell back
                         to the median of every interval)
    For fewer than 5 peaks -- classify_arrhythmias() then returns no events
    and computes no baseline -- the times and baseline are NaN and
    reliable is False.
    """
    nan = float("nan")
    if len(rpeaks) < 5:
        return {"t_start": nan, "t_end": nan, "length_s": nan, "span_s": nan,
                "n_intervals": 0, "baseline_rr_ms": nan, "baseline_hr": nan,
                "reliable": False}
    peaks   = np.sort(np.asarray(rpeaks, dtype=int))
    t_peaks = peaks / fs
    mask, bl_len = _baseline_rr_mask(t_peaks, baseline_s)
    rr_ms   = np.diff(peaks).astype(float) / fs * 1000   # as classify_arrhythmias
    bl_rr   = float(np.median(rr_ms[mask]))
    t0      = float(t_peaks[0])
    span_s  = float(t_peaks[-1]) - t0
    return {
        "t_start":        t0,
        "t_end":          t0 + min(bl_len, span_s),
        "length_s":       float(bl_len),
        "span_s":         span_s,
        "n_intervals":    int(mask.sum()),
        "baseline_rr_ms": bl_rr,
        "baseline_hr":    60_000.0 / max(bl_rr, 1.0),
        "reliable":       bool(span_s >= _BASELINE_MIN_SPAN_FACTOR * bl_len),
    }


def arrhythmia_baseline_window(
    rpeaks:     np.ndarray,
    fs:         float,
    baseline_s: float = 30.0,
) -> "tuple[float, float]":
    """Absolute (t_start, t_end) in seconds of the stretch that
    classify_arrhythmias() takes its baseline HR from, for the same
    rpeaks / fs / baseline_s -- so a plot can shade exactly that window.

    Starts at the first peak passed in; t_end is clamped to the last peak.
    Returns (nan, nan) for fewer than 5 peaks, where classify_arrhythmias()
    returns no events and computes no baseline. See
    arrhythmia_baseline_info() for the full description (and whether the
    baseline is a meaningful reference).
    """
    info = arrhythmia_baseline_info(rpeaks, fs, baseline_s)
    return info["t_start"], info["t_end"]


def classify_arrhythmias(
    rpeaks:          np.ndarray,
    fs:              float,
    baseline_s:      float = 30.0,
    brady_pct:       float = 20.0,
    min_brady_beats: int   = 10,
    progress_cb:     Optional[Callable[[int, str], None]] = None,
    pr_ms:           "Optional[np.ndarray]" = None,
    removed_beats:   "Optional[Iterable[int]]" = None,
) -> list[ArrhythmiaEvent]:
    """Rule-based arrhythmia classification for mouse ECG.

    Premature beats (RR-based): a beat ending an RR interval shorter than
        PREMATURE_RR_FRACTION (0.70) x the reference RR -- the median of
        the 10 intervals before the group's first premature beat or, if
        smaller, of the 10 from the interval the rhythm returns on
        (_premature_runs). A group must end (rhythm back above the bound)
        within PREMATURE_MAX_RUN (30) beats; a longer stretch is a rate
        change (tachycardia rule), and a group still going at the end of
        the analysed beats is not reported. Consecutive
        premature beats are grouped: 1 -> "premature_beat" (isolated,
        severity info), 2 -> "premature_couplet" (warning), >= 3 ->
        "premature_run" (alert; "esv_run" in older versions, which
        reported neither isolated beats nor couplets). Timing alone cannot
        tell ventricular from supraventricular beats, hence the names.
        Premature events are never merged with a neighbouring one (two
        isolated beats a few beats apart are two events), and an irregular
        run overlapping one is not reported (the premature beat explains
        the irregularity).

    removed_beats : samples of the REAL beats artifact correction removed
        (removed_beats_for_classification(): every "ectopic" removal still
        in effect, plus the duplicate / non-physiological ones whose
        removal merged a long interval -- very premature beats). They are absent
        from *rpeaks*, so the short interval that made them premature is
        gone; they are put back for the premature-beat classification only
        -- the counts then match a run without correction -- and each
        premature event records in ``n_removed`` how many of its beats
        were removed. An interval of *rpeaks* that spans a removed
        beat is not reported as a pause (it is the coupling plus the
        compensatory interval). Samples outside the first..last peak of
        *rpeaks* are ignored.

    baseline_s : length (s) of the reference window whose median RR sets
        the baseline HR every bradycardia/tachycardia run is compared
        against. It is RELATIVE to the start of the analysed beats -- the
        first baseline_s seconds after the first peak passed in, not the
        first baseline_s seconds of the recording -- since callers pass the
        analysis-window subset of the peaks. When that window holds fewer
        than 10 RR intervals it widens to max(20% of the analysed span,
        baseline_s), still measured from the first peak; with fewer than 5
        even then, the median of every interval is used.
        arrhythmia_baseline_window() returns the resulting window (absolute
        times) for plotting. When the analysed span is shorter than
        _BASELINE_MIN_SPAN_FACTOR (2) × that window, the baseline covers
        most of the beats being classified: classification still runs, but
        a warning is logged and every bradycardia/tachycardia event gets
        ``baseline_reliable=False`` (arrhythmia_baseline_info() reports the
        same condition for callers, even when no such event is found).

    pr_ms : optional per-beat PR interval (ms), same length and order as
        rpeaks (i.e. pr_ms[i] is the PR interval measured at rpeaks[i]),
        NaN where no PR was measured for that beat. analyse_intervals()
        returns fewer rows than peaks (edge beats are dropped), and its PR
        is P onset -> QRS onset, so its PR_ms column must be mapped onto rpeaks with
        analysis.align_interval_column_to_peaks() first -- passing the raw
        column positionally would shift every value to the wrong beat.
        When provided, enables sustained-PR-prolongation detection
        ("av_delay" kind). This function trusts positional correspondence
        and only guards against a length mismatch (AV-delay detection is
        then skipped). When None (Interval delineation hasn't been run),
        AV-delay detection is silently skipped -- every other event kind
        is unaffected, matching this function's behaviour before pr_ms
        existed.
    """
    def _prog(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    if len(rpeaks) < 5:
        return []

    # argsort rather than np.sort so pr_ms (indexed like the caller's
    # rpeaks) can be put in the same order below.
    rpeaks_in  = np.asarray(rpeaks, dtype=int)
    peak_order = np.argsort(rpeaks_in, kind="stable")
    rpeaks  = rpeaks_in[peak_order]
    t_peaks = rpeaks / fs
    rr_ms   = np.diff(rpeaks).astype(float) / fs * 1000
    n       = len(rr_ms)
    events: list[ArrhythmiaEvent] = []

    # Individual baseline HR -- window measured from the first analysed
    # peak, not from t=0 (see _baseline_rr_mask).
    bl_mask, bl_len_s = _baseline_rr_mask(t_peaks, baseline_s)
    baseline_rr_ms = float(np.median(rr_ms[bl_mask]))
    baseline_hr    = 60_000.0 / max(baseline_rr_ms, 1.0)
    span_s         = float(t_peaks[-1] - t_peaks[0])
    # Same rule as arrhythmia_baseline_info() (see _BASELINE_MIN_SPAN_FACTOR).
    baseline_reliable = span_s >= _BASELINE_MIN_SPAN_FACTOR * bl_len_s
    log.info(
        "classify_arrhythmias: baseline=%.1f bpm  (%.1f ms, %d beats, "
        "%.0f s window from t=%.1f s)",
        baseline_hr, baseline_rr_ms, int(bl_mask.sum()), bl_len_s,
        float(t_peaks[0]),
    )
    if not baseline_reliable:
        log.warning(
            "classify_arrhythmias: baseline window (%.0f s) covers most of the "
            "%.0f s analysed — bradycardia/tachycardia calls are unreliable",
            bl_len_s, span_s)

    # HR glissant (médiane symétrique sur 5 battements)
    ROLL_HALF  = 2
    rolling_rr = np.array([
        float(np.median(rr_ms[max(0, i - ROLL_HALF):min(n, i + ROLL_HALF + 1)]))
        for i in range(n)
    ])
    rolling_hr = 60_000.0 / np.maximum(rolling_rr, 1.0)

    _prog(10, "Baseline HR computed…")

    brady_hr_thresh = baseline_hr * (1.0 - brady_pct / 100.0)
    tachy_hr_thresh = baseline_hr * (1.0 + brady_pct / 100.0)
    brady_ind = rolling_hr < brady_hr_thresh
    tachy_ind = rolling_hr > tachy_hr_thresh

    def _emit(run_start: int, run_end: int, kind: str, sev: str) -> None:
        run_len = run_end - run_start
        if run_len < min_brady_beats:
            return
        rr_seg = rr_ms[run_start:run_end]
        hr_m   = float(60_000.0 / rr_seg.mean()) if len(rr_seg) else 0.0
        delta  = (hr_m - baseline_hr) / max(baseline_hr, 1.0) * 100.0
        arrow  = "\u2193" if delta < 0 else "\u2191"
        events.append(ArrhythmiaEvent(
            kind=kind,
            label=f"{arrow}{abs(delta):.0f}% vs baseline "
                  f"({baseline_hr:.0f}\u2192{hr_m:.0f} bpm)  {run_len} beats",
            t_start=float(t_peaks[run_start]),
            t_end=float(t_peaks[min(run_end, len(t_peaks) - 1)]),
            n_beats=run_len, hr_mean=hr_m, rr_mean=float(rr_seg.mean()),
            severity=sev,
            baseline_hr=round(baseline_hr, 1),
            delta_pct=round(delta, 1),
            baseline_reliable=baseline_reliable,
        ))

    def _find_runs_ind(mask: np.ndarray, kind: str, sev: str) -> None:
        in_run = False; run_start = 0
        for i, cond in enumerate(mask):
            if cond and not in_run:   in_run = True; run_start = i
            elif not cond and in_run: _emit(run_start, i, kind, sev); in_run = False
        if in_run: _emit(run_start, n, kind, sev)

    _find_runs_ind(brady_ind, "bradycardia", "warning")
    _find_runs_ind(tachy_ind, "tachycardia", "warning")

    _prog(30, "Bradycardia/tachycardia runs found…")

    WIN = 11; half = WIN // 2

    def _lmed(i: int) -> float:
        # Exclude rr_ms[i] itself from its own reference window -- a
        # centered slice rr_ms[i-half:i+half+1] always contains the tested
        # interval, biasing the local reference toward the very value being
        # judged against it (matches the self-excluding neighbor logic
        # already used in detect_rr_artifacts's Pass 2 below).
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = np.concatenate([rr_ms[lo:i], rr_ms[i + 1:hi]])
        return float(np.median(window)) if len(window) else float(rr_ms[i])

    local_med  = np.array([_lmed(i) for i in range(n)])
    rr_hi_ctx  = 60_000.0 / _BRADY_RR_HI_HR_LO_BPM

    # Real beats removed by artifact correction, inside the analysed span.
    if removed_beats is not None:
        rb = np.unique(np.asarray(list(removed_beats), dtype=np.int64).flatten())
        rb = rb[(rb > rpeaks[0]) & (rb < rpeaks[-1])]
        rb = rb[~np.isin(rb, rpeaks)]
    else:
        rb = np.array([], dtype=np.int64)
    # Interval i of rpeaks spans a removed beat -> not a pause.
    spans_removed = np.zeros(n, dtype=bool)
    if len(rb):
        spans_removed[np.searchsorted(rpeaks, rb) - 1] = True

    for i, rr in enumerate(rr_ms):
        if spans_removed[i]:
            continue
        if rr > 2.0 * local_med[i] and rr > rr_hi_ctx * 1.5:
            events.append(ArrhythmiaEvent(
                kind="pause", label=f"Pause ({rr:.0f} ms)",
                t_start=float(t_peaks[i]), t_end=float(t_peaks[i + 1]),
                n_beats=2, hr_mean=60_000 / rr, rr_mean=rr, severity="alert",
            ))

    # ── Premature beats (RR-based) ──
    # Classified on the peaks WITH the removed real beats put back (see the
    # docstring), so auto-correction does not make them vanish.
    if len(rb):
        pb_peaks = np.sort(np.concatenate([rpeaks.astype(np.int64), rb]))
        pb_removed = np.isin(pb_peaks, rb)
        pb_rr = np.diff(pb_peaks).astype(float) / fs * 1000
    else:
        pb_peaks = rpeaks.astype(np.int64)
        pb_removed = np.zeros(len(rpeaks), dtype=bool)
        pb_rr = rr_ms
    pb_t = pb_peaks / fs

    def _emit_premature(run_start: int, run_end: int, ref_ms: float) -> None:
        rl = run_end - run_start                  # premature beats in the group
        seg = pb_rr[run_start:run_end]
        hr_m = float(60_000 / seg.mean())
        n_rm = int(pb_removed[run_start + 1:run_end + 1].sum())
        if rl == 1:
            kind, sev = "premature_beat", "info"
            label = (f"Isolated premature beat (RR-based) — RR {float(seg[0]):.0f} ms "
                     f"vs local {ref_ms:.0f} ms")
        elif rl == 2:
            kind, sev = "premature_couplet", "warning"
            label = f"Premature couplet (RR-based) — 2 beats ({hr_m:.0f} bpm)"
        else:
            kind, sev = "premature_run", "alert"
            label = f"Premature run (RR-based) — {rl} beats ({hr_m:.0f} bpm)"
        if n_rm:
            label += f"  · {n_rm} removed by artifact correction"
        events.append(ArrhythmiaEvent(
            kind=kind, label=label,
            t_start=float(pb_t[run_start]),
            t_end=float(pb_t[min(run_end, len(pb_t) - 1)]),
            n_beats=rl, hr_mean=hr_m, rr_mean=float(seg.mean()),
            severity=sev, n_removed=n_rm,
        ))

    # Interval i short -> beat i+1 (the one ending it) is premature.
    for run_start, run_end, ref_ms in _premature_runs(pb_rr):
        _emit_premature(run_start, run_end, ref_ms)

    _prog(70, "Pauses/premature beats found…")

    # AV conduction delay -- only possible when per-beat PR data was passed
    # in (see this function's docstring). pr_ms is indexed like rpeaks
    # (length n+1; reordered with peak_order, a no-op for already-sorted
    # peaks); slice to n and treat pr_ms[i] as the PR measured at the
    # LEADING beat of RR-interval i, matching every other run-detection mask
    # in this function (brady_ind, tachy_ind, the premature runs all index by interval,
    # not by beat). NaN PR (beat not delineated) never counts toward a run.
    if pr_ms is not None and len(pr_ms) == len(rpeaks):
        pr_arr = np.asarray(pr_ms, dtype=float)[peak_order][:n]
        pr_hi  = _AV_DELAY_PR_HI_MS
        av_mask = np.isfinite(pr_arr) & (pr_arr > pr_hi)

        def _emit_av(run_start: int, run_end: int) -> None:
            run_len = run_end - run_start
            if run_len < min_brady_beats:
                return
            pr_seg = pr_arr[run_start:run_end]
            pr_seg = pr_seg[np.isfinite(pr_seg)]
            if len(pr_seg) == 0:
                return
            rr_seg = rr_ms[run_start:run_end]
            events.append(ArrhythmiaEvent(
                kind="av_delay",
                label=(f"Sustained PR prolongation — {run_len} beats "
                       f"(PR {float(pr_seg.mean()):.0f} ms, ref ≤{pr_hi:.0f} ms)"),
                t_start=float(t_peaks[run_start]),
                t_end=float(t_peaks[min(run_end, len(t_peaks) - 1)]),
                n_beats=run_len,
                hr_mean=float(60_000 / rr_seg.mean()) if len(rr_seg) else 0.0,
                rr_mean=float(rr_seg.mean()) if len(rr_seg) else 0.0,
                severity="warning",
            ))

        in_run = False; run_start = 0
        for i, cond in enumerate(av_mask):
            if cond and not in_run:   in_run = True; run_start = i
            elif not cond and in_run: _emit_av(run_start, i); in_run = False
        if in_run:
            _emit_av(run_start, n)

        _prog(80, "AV conduction delay checked…")

    WIN_CV = 7; i = 0
    while i <= n - WIN_CV:
        seg = rr_ms[i:i + WIN_CV]
        cv  = seg.std(ddof=1) / (seg.mean() + 1e-6)
        if cv > 0.15:
            t0 = float(t_peaks[i])
            t1 = float(t_peaks[min(i + WIN_CV, len(t_peaks) - 1)])
            overlap = any(
                e.t_start <= t1 and e.t_end >= t0
                for e in events
                if e.kind in ("bradycardia", "tachycardia", "pause", *PREMATURE_KINDS)
            )
            if not overlap:
                events.append(ArrhythmiaEvent(
                    kind="irregular_run",
                    label=f"RR irregularity — {WIN_CV} beats (CV={cv:.0%})",
                    t_start=t0, t_end=t1, n_beats=WIN_CV,
                    hr_mean=float(60_000 / seg.mean()), rr_mean=float(seg.mean()),
                    severity="info",
                ))
            i += WIN_CV
        else:
            i += 1

    _prog(90, "Irregular-RR runs found…")

    events.sort(key=lambda e: e.t_start)

    merged: list[ArrhythmiaEvent] = []
    for ev in events:
        # Premature groups are delimited beat by beat above: two of them
        # half a second apart are two events, not one longer one.
        if (merged and merged[-1].kind == ev.kind and ev.kind not in PREMATURE_KINDS
                and ev.t_start <= merged[-1].t_end + 0.5):
            p = merged[-1]
            merged[-1] = dataclasses.replace(
                p, t_end=max(p.t_end, ev.t_end),
                n_beats=p.n_beats + ev.n_beats,
                delta_pct=round((p.delta_pct + ev.delta_pct) / 2, 1),
            )
        else:
            merged.append(ev)

    log.info(
        "classify_arrhythmias: %d events (brady=%d tachy=%d pause=%d "
        "premature beat/couplet/run=%d/%d/%d irr=%d) "
        "baseline=%.0f bpm",
        len(merged),
        sum(1 for e in merged if e.kind == "bradycardia"),
        sum(1 for e in merged if e.kind == "tachycardia"),
        sum(1 for e in merged if e.kind == "pause"),
        sum(1 for e in merged if e.kind == "premature_beat"),
        sum(1 for e in merged if e.kind == "premature_couplet"),
        sum(1 for e in merged if e.kind == "premature_run"),
        sum(1 for e in merged if e.kind == "irregular_run"),
        baseline_hr,
    )
    _prog(100, "Abnormal-event classification done")
    return merged


def premature_beat_summary(
    events:  "Iterable[ArrhythmiaEvent]",
    span_s:  float,
    removal_info: "Optional[dict]" = None,
    t_start: "Optional[float]" = None,
    t_end:   "Optional[float]" = None,
) -> dict:
    """Counts and per-hour rates of the RR-based premature beats among
    classify_arrhythmias() *events*, over *span_s* seconds of analysed
    recording (first to last analysed peak, arrhythmia_baseline_info's
    ``span_s``).

    Keys: n_isolated, n_couplets, n_runs (events), n_beats_in_runs,
    n_premature (every premature beat: isolated + 2 × couplets + beats in
    runs), n_removed (of those, beats removed by artifact correction and
    put back for classification), span_s, and the *_per_h rates
    (isolated / couplets / runs / premature), NaN when span_s <= 0. Rates
    from a short recording are extrapolations -- report span_s with them.

    *removal_info* (removed_beats_for_classification's info) adds
    n_removed_spurious (peaks artifact correction removed as duplicate /
    non-physiological and judged spurious: not counted) and
    n_removed_unknown_type (removals of unknown type -- sessions saved
    before removed_types -- re-examined by the interval test only).
    *t_start* / *t_end* (s, first / last analysed peak) are stored as
    window_start_s / window_end_s (NaN when not given).
    """
    evs = [e for e in events if e.kind in PREMATURE_KINDS]
    iso = [e for e in evs if e.kind == "premature_beat"]
    cpl = [e for e in evs if e.kind == "premature_couplet"]
    run = [e for e in evs if e.kind == "premature_run"]
    span = float(span_s) if span_s is not None and np.isfinite(span_s) else 0.0
    hours = span / 3600.0

    def _rate(k: int) -> float:
        return float(k) / hours if hours > 0 else float("nan")

    n_runb = int(sum(int(e.n_beats) for e in run))
    n_prem = len(iso) + 2 * len(cpl) + n_runb
    return {
        "n_isolated":       len(iso),
        "n_couplets":       len(cpl),
        "n_runs":           len(run),
        "n_beats_in_runs":  n_runb,
        "n_premature":      n_prem,
        "n_removed":        int(sum(int(getattr(e, "n_removed", 0) or 0) for e in evs)),
        "span_s":           span,
        "isolated_per_h":   _rate(len(iso)),
        "couplets_per_h":   _rate(len(cpl)),
        "runs_per_h":       _rate(len(run)),
        "premature_per_h":  _rate(n_prem),
        "n_removed_spurious":     int((removal_info or {}).get("n_spurious", 0) or 0),
        "n_removed_unknown_type": int((removal_info or {}).get("n_unknown_type", 0) or 0),
        "window_start_s":   float(t_start) if t_start is not None else float("nan"),
        "window_end_s":     float(t_end) if t_end is not None else float("nan"),
    }


def _fmt_rate(x: float) -> str:
    return f"{x:.1f}/h" if np.isfinite(x) else "—/h"


def premature_summary_rows(summary: "Optional[dict]") -> "list[tuple[str, str]]":
    """(label, value) rows describing a premature_beat_summary() -- the
    same rows on the Events tab, the Excel and the PDF report."""
    if not summary:
        return []
    s = summary
    span = float(s.get("span_s", 0.0) or 0.0)

    def _n(k: int, word: str) -> str:
        return f"{k} {word}" if k == 1 else f"{k} {word}s"

    rows = [
        ("Premature beats (RR-based)",
         f"{_n(int(s['n_premature']), 'beat')}  ({_fmt_rate(s['premature_per_h'])})"),
        ("  Isolated",  f"{int(s['n_isolated'])}  ({_fmt_rate(s['isolated_per_h'])})"),
        ("  Couplets",  f"{int(s['n_couplets'])}  ({_fmt_rate(s['couplets_per_h'])})"),
        ("  Runs ≥3",   f"{_n(int(s['n_runs']), 'run')}, "
                        f"{_n(int(s['n_beats_in_runs']), 'beat')}  "
                        f"({_fmt_rate(s['runs_per_h'])})"),
    ]
    if int(s.get("n_removed", 0) or 0):
        rows.append(("  Removed by artifact correction",
                     f"{int(s['n_removed'])} (still counted)"))
    if int(s.get("n_removed_spurious", 0) or 0):
        rows.append(("  Removed as spurious detections",
                     f"{int(s['n_removed_spurious'])} duplicate / non-physiological "
                     f"peak(s) not counted (removal left a normal interval)"))
    if int(s.get("n_removed_unknown_type", 0) or 0):
        rows.append(("  Removed peaks of unknown type",
                     f"{int(s['n_removed_unknown_type'])} (session saved before removal "
                     f"types were recorded): judged only from the interval they leave "
                     f"— re-run artifact correction for the full check"))
    w0, w1 = s.get("window_start_s"), s.get("window_end_s")
    if w0 is not None and w1 is not None and np.isfinite(w0) and np.isfinite(w1):
        rows.append(("  Analysed window", f"{float(w0):.1f}–{float(w1):.1f} s"))
    rows.append(("  Analysed span", f"{span:.1f} s  (rates per hour of it)"))
    rows.append(("  Definition", PREMATURE_DEFINITION_TEXT))
    return rows
