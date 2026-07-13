"""
ecg.i18n
────────
Minimal bilingual string table (English / French).

Usage
-----
    from ecg.ui.i18n import t, set_language, get_language

    t("open_file")          # → "Open .mat file"  or "Ouvrir fichier .mat"
    set_language("fr")
    t("run_analysis")       # → "Lancer l'analyse complète"

The active language is stored as a module-level variable and shared
across all imports in the same process.  Call set_language() once
(e.g. from the theme dialog) to switch all future t() calls.
"""
from __future__ import annotations

_LANGUAGE: str = "en"   # "en" | "fr"


def set_language(lang: str) -> None:
    """Set the active UI language.  Accepted values: "en", "fr"."""
    global _LANGUAGE
    if lang in ("en", "fr"):
        _LANGUAGE = lang


def get_language() -> str:
    """Return the current active language code."""
    return _LANGUAGE


def t(key: str) -> str:
    """Translate *key* to the active language; fall back to the key itself."""
    table = _STRINGS.get(_LANGUAGE, _STRINGS["en"])
    return table.get(key, _STRINGS["en"].get(key, key))


# ── String table ──────────────────────────────────────────────────────────────
_STRINGS: dict[str, dict[str, str]] = {
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "en": {
        # ── Sidebar titles & buttons ──────────────────────────────────────
        "app_title":            "ECG Analysis",
        "no_file":              "No file loaded",
        "open_file":            "Open .mat file",
        "channels":             "Channels",
        "recent":               "Recent",
        "drag_drop":            "or drag & drop a .mat file",
        "peaks_detected":       "Peaks detected:",
        "peaks_run_detection":  "Run detection first",

        # ── Workflow step cards ───────────────────────────────────────────
        "step1_title":          "Detect R peaks",
        "step1_sub":            "Load signal · detect R peaks · adjust threshold",
        "btn_preview":          "Preview Detection",

        "step2_title":          "Correct artifacts",
        "step2_sub":            "Interactive review (ARTIFACTS ↓) or auto-correct.",

        "step3_title":          "Analysis window",
        "step3_sub":            "Optional — leave blank for full signal.",
        "from_s":               "From (s)",
        "to_s":                 "To (s)",
        "apply":                "Apply",
        "full":                 "Full",

        "step4_title":          "Analyse",
        "step4_sub":            "RR · HR · HRV · Beat template  —  Freq / Intervals → tabs",
        "btn_run":              "Run Full Analysis",

        # ── Status ───────────────────────────────────────────────────────
        "ready":                "Ready",
        "loading":              "Loading…",
        "analysing":            "Analysing…",
        "done":                 "Done",

        # ── Sections ─────────────────────────────────────────────────────
        "sec_file":             "FILE & SUBJECT",
        "sec_signal":           "SIGNAL",
        "sec_filters":          "FILTERS",
        "sec_detection":        "DETECTION",
        "sec_artifacts":        "ARTIFACTS",
        "sec_session":          "SESSION & EXPORT",

        # ── File/Signal labels ────────────────────────────────────────────
        "lbl_channel":          "Channel name",
        "lbl_subject":          "Subject ID",
        "lbl_fs":               "Sampling rate (Hz)",
        "lbl_t_start":          "Start (s)",
        "lbl_t_end":            "End (s)",
        "sw_show_raw":          "Show raw signal (vs filtered)",

        # ── Filter labels ─────────────────────────────────────────────────
        "sw_no_filter":         "Raw signal (no DSP filters)",
        "no_filter_hint":       "Enable filters below for advanced processing",
        "sw_invert":            "⟳  Invert signal (polarity)",
        "invert_hint":          "Useful if R peaks appear negative",
        "advanced_filters":     "Advanced filters",
        "lbl_hp":               "HP cut (Hz)",
        "lbl_lp":               "LP cut (Hz)",
        "lbl_clean":            "Clean method:",
        "sw_notch":             "Notch 50 Hz",

        # ── Detection labels ──────────────────────────────────────────────
        "lbl_minrr":            "Min R-R distance (ms)",
        "lbl_sensitivity":      "Sensitivity:",
        "strict_sensitive":     "strict ↑  /  sensitive ↓",
        "exact":                "Exact:",
        "lbl_det_method":       "Detection method:",
        "det_auto":             "Auto (NeuroKit2)",
        "det_sg_deriv":         "SG + Derivative (10 kHz)",
        "sg_target_fs":         "Target fs (Hz):",
        "sg_window_ms":         "SG window (ms):",

        # ── Artifact labels ───────────────────────────────────────────────
        "btn_review_art":       "🔍  Review Artifacts",
        "review_art_hint":      "Detect + review every artifact. Run Preview first.",
        "sw_artifact":          "Auto-correct on Full Analysis",
        "artifact_hint":        "Skips review — applies default remove decisions",

        # ── Session & Export ──────────────────────────────────────────────
        "btn_save_session":     "💾  Save Session",
        "no_session":           "No session saved for this file",
        "btn_clear_session":    "🗑  Clear Session Cache",
        "btn_export_excel":     "📊  Export Excel",
        "btn_export_zip":       "📦  Export ZIP  (Excel + Figures)",
        "btn_export_pdf":       "📄  PDF Report  (1 page)",
        "btn_export_prism":     "🔬  Export GraphPad Prism",
        "btn_params":           "⚙  Parameters",
        "reset_params":         "↺  Reset to Mouse ECG Defaults",

        # ── Navigation ───────────────────────────────────────────────────
        "window_lbl":           "Window:",
        "go":                   "Go",

        # ── Tabs ─────────────────────────────────────────────────────────
        "tab_detection":        "Detection",
        "tab_hrv":              "HRV",
        "tab_intervals":        "Intervals",
        "tab_beat":             "Beat Template",
        "tab_arrhythmias":      "Arrhythmias",
        "tab_interp":           "Interpretation",
        "tab_summary":          "Summary",

        # ── Params dialog ─────────────────────────────────────────────────
        "params_dialog_title":  "Parameters",
        "params_sec_file":      "File & Subject",
        "params_sec_signal":    "Signal",
        "params_sec_filters":   "Filters",
        "params_sec_detection": "Detection",
        "params_sec_artifacts": "Artifacts",
        "btn_apply_close":      "Apply & Close",
        "btn_cancel":           "Cancel",
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "fr": {
        # ── Sidebar titles & buttons ──────────────────────────────────────
        "app_title":            "Analyse ECG",
        "no_file":              "Aucun fichier chargé",
        "open_file":            "Ouvrir fichier .mat",
        "channels":             "Canaux",
        "recent":               "Récents",
        "drag_drop":            "ou glisser-déposer un fichier .mat",
        "peaks_detected":       "Pics détectés :",
        "peaks_run_detection":  "Lancer la détection",

        # ── Workflow step cards ───────────────────────────────────────────
        "step1_title":          "Détecter les pics R",
        "step1_sub":            "Charger · détecter les pics R · ajuster le seuil",
        "btn_preview":          "Prévisualiser la détection",

        "step2_title":          "Corriger les artefacts",
        "step2_sub":            "Revue interactive (ARTEFACTS ↓) ou correction auto.",

        "step3_title":          "Fenêtre d'analyse",
        "step3_sub":            "Optionnel — laisser vide pour le signal complet.",
        "from_s":               "Début (s)",
        "to_s":                 "Fin (s)",
        "apply":                "Appliquer",
        "full":                 "Tout",

        "step4_title":          "Analyser",
        "step4_sub":            "RR · FC · VFC · Gabarit  —  Fréq / Intervalles → onglets",
        "btn_run":              "Lancer l'analyse complète",

        # ── Status ───────────────────────────────────────────────────────
        "ready":                "Prêt",
        "loading":              "Chargement…",
        "analysing":            "Analyse en cours…",
        "done":                 "Terminé",

        # ── Sections ─────────────────────────────────────────────────────
        "sec_file":             "FICHIER & SUJET",
        "sec_signal":           "SIGNAL",
        "sec_filters":          "FILTRES",
        "sec_detection":        "DÉTECTION",
        "sec_artifacts":        "ARTEFACTS",
        "sec_session":          "SESSION & EXPORT",

        # ── File/Signal labels ────────────────────────────────────────────
        "lbl_channel":          "Nom du canal",
        "lbl_subject":          "ID sujet",
        "lbl_fs":               "Fréquence d'échantillonnage (Hz)",
        "lbl_t_start":          "Début (s)",
        "lbl_t_end":            "Fin (s)",
        "sw_show_raw":          "Afficher signal brut (vs filtré)",

        # ── Filter labels ─────────────────────────────────────────────────
        "sw_no_filter":         "Signal brut (sans filtres DSP)",
        "no_filter_hint":       "Activer les filtres ci-dessous pour traitement avancé",
        "sw_invert":            "⟳  Inverser le signal (polarité)",
        "invert_hint":          "Utile si les pics R apparaissent négatifs",
        "advanced_filters":     "Filtres avancés",
        "lbl_hp":               "Coupe-bas (Hz)",
        "lbl_lp":               "Coupe-haut (Hz)",
        "lbl_clean":            "Méthode de nettoyage :",
        "sw_notch":             "Notch 50 Hz",

        # ── Detection labels ──────────────────────────────────────────────
        "lbl_minrr":            "Distance RR min (ms)",
        "lbl_sensitivity":      "Sensibilité :",
        "strict_sensitive":     "strict ↑  /  sensible ↓",
        "exact":                "Exact :",
        "lbl_det_method":       "Méthode de détection :",
        "det_auto":             "Auto (NeuroKit2)",
        "det_sg_deriv":         "SG + Dérivée (10 kHz)",
        "sg_target_fs":         "Fs cible (Hz) :",
        "sg_window_ms":         "Fenêtre SG (ms) :",

        # ── Artifact labels ───────────────────────────────────────────────
        "btn_review_art":       "🔍  Revoir les artefacts",
        "review_art_hint":      "Détecter + réviser chaque artefact. Lancer Prévisualisation d'abord.",
        "sw_artifact":          "Auto-corriger lors de l'analyse",
        "artifact_hint":        "Ignore la revue — applique les décisions de suppression par défaut",

        # ── Session & Export ──────────────────────────────────────────────
        "btn_save_session":     "💾  Sauvegarder session",
        "no_session":           "Aucune session sauvegardée pour ce fichier",
        "btn_clear_session":    "🗑  Effacer le cache de session",
        "btn_export_excel":     "📊  Exporter Excel",
        "btn_export_zip":       "📦  Exporter ZIP  (Excel + Figures)",
        "btn_export_pdf":       "📄  Rapport PDF  (1 page)",
        "btn_export_prism":     "🔬  Exporter GraphPad Prism",
        "btn_params":           "⚙  Paramètres",
        "reset_params":         "↺  Réinitialiser valeurs souris",

        # ── Navigation ───────────────────────────────────────────────────
        "window_lbl":           "Fenêtre :",
        "go":                   "Aller",

        # ── Tabs ─────────────────────────────────────────────────────────
        "tab_detection":        "Détection",
        "tab_hrv":              "HRV",
        "tab_intervals":        "Intervalles",
        "tab_beat":             "Beat Template",
        "tab_arrhythmias":      "Arythmies",
        "tab_interp":           "Interprétation",
        "tab_summary":          "Résumé",

        # ── Params dialog ─────────────────────────────────────────────────
        "params_dialog_title":  "Paramètres",
        "params_sec_file":      "Fichier & Sujet",
        "params_sec_signal":    "Signal",
        "params_sec_filters":   "Filtres",
        "params_sec_detection": "Détection",
        "params_sec_artifacts": "Artefacts",
        "btn_apply_close":      "Appliquer et fermer",
        "btn_cancel":           "Annuler",
    },
}
