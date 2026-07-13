# ECG Analysis — Outil d'électrophysiologie cardiaque murine

*[English version](README.md)*

Application de bureau (CustomTkinter) pour l'analyse d'enregistrements ECG
cardiaques de souris : détection des pics R, VRC (temporel/fréquentiel/non
linéaire), délinéation des ondes P-QRS-T, classification des arythmies, et
export vers Excel / GraphPad Prism.

Conçue autour de la physiologie spécifique à la souris (Thireau et al. 2008 ;
Mitchell et al. 1998) plutôt qu'à partir de valeurs par défaut humaines
adaptées — plage de fréquence cardiaque, seuil pNN, bandes fréquentielles de
VRC et formule de correction du QTc sont toutes calibrées pour la souris
(FC de repos typique ~500 bpm, plage 180–900 bpm).

## Fonctionnalités

- **Chargement du signal** — fichiers MATLAB `.mat`, format historique
  (v5/v6) et HDF5 (v7.3, exports Spike2), avec détection automatique du
  canal et de la fréquence d'échantillonnage.
- **Filtrage** — passe-bande, filtre coupe-bande, suppression de la dérive
  de ligne de base.
- **Détection des pics R** — trois algorithmes interchangeables
  (Savitzky-Golay + dérivée, transformée en ondelettes continue, enveloppe
  maximale), avec édition manuelle, undo/redo et curseur de seuil interactif.
- **Gestion des artefacts RR** — détection des doublons, intervalles
  non-physiologiques et battements ectopiques, avec une liste de candidats
  à valider avant toute suppression.
- **Analyse VRC** — domaine temporel (SDNN, RMSSD, pNN6...), domaine
  fréquentiel (LF/HF via Lomb-Scargle), et non linéaire (Poincaré SD1/SD2,
  entropie d'échantillon, DFA).
- **Délinéation des ondes** — repères P/Q/R/S/J/T et intervalles
  PR/QRS/QT/QTc, avec un template d'onde calibrable par enregistrement.
- **Classification des arythmies** — épisodes de bradycardie/tachycardie,
  pauses, salves de battements ectopiques, séries irrégulières, évaluées par
  rapport à une ligne de base propre à chaque enregistrement.
- **VRC glissante et analyse par époques**, et comparaison de segments A/B.
- **Persistance de session** — sauvegarde/restauration d'une session
  d'analyse complète en JSON, plus un registre SQLite local des
  enregistrements récemment analysés.
- **Export** — classeurs Excel formatés, GraphPad Prism `.pzfx`, rapports
  PDF, CSV.
- **Traitement par lots** — analyse de plusieurs fichiers en parallèle, un
  classeur de sortie par fichier plus une feuille de synthèse combinée.
- **Thèmes** — 7 préréglages clair/sombre, propagés en direct dans toute
  l'interface.

## Prérequis

- Python ≥ 3.10
- Voir [`requirements.txt`](requirements.txt) pour la liste complète des
  dépendances. `h5py` n'est requis que pour les fichiers MATLAB v7.3/HDF5 ;
  `pzfx` uniquement pour l'export GraphPad Prism.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
# — ou, pour une installation éditable qui enregistre aussi la commande
#   `ecg-analysis` et permet `import ecg` depuis n'importe où —
pip install -e .
```

## Utilisation

```bash
python -m ecg          # point d'entrée recommandé
python ecg_app.py       # point d'entrée rétrocompatible (même application)
ecg-analysis            # après `pip install -e .`
```

## Arborescence du projet

```
.
├── ecg/
│   ├── __init__.py       # version du package, aperçu de l'architecture
│   ├── __main__.py       # point d'entrée `python -m ecg`
│   ├── batch.py          # pipeline de traitement par lots en parallèle
│   ├── core/             # pur NumPy/SciPy/pandas — sans UI, sans I/O fichier
│   │   ├── models.py         # constantes de physiologie, dataclasses partagées
│   │   ├── filtering.py      # passe-bande, coupe-bande, normalisation
│   │   ├── detection.py      # détection des pics R, artefacts RR, arythmies
│   │   ├── analysis.py       # VRC temporel/fréquentiel/non linéaire, intervalles
│   │   └── wave_template.py  # calibration du template d'onde & délinéation
│   ├── io/                # formats de fichiers et persistance
│   │   ├── loaders.py        # chargement du signal .mat / HDF5
│   │   ├── session.py        # sauvegarde/chargement de session (JSON)
│   │   ├── export.py         # exporteurs Excel et GraphPad Prism
│   │   └── db.py             # registre SQLite des enregistrements
│   └── ui/                # application de bureau CustomTkinter
│       ├── app.py             # fenêtre principale ECGApp
│       ├── *_controller.py    # un contrôleur par responsabilité (SRP)
│       ├── theme.py           # préréglages de couleur/police, propagation live
│       ├── plots.py           # pont matplotlib ↔ canvas Tk
│       ├── dialogs.py         # fenêtres secondaires (thème, revue d'artefacts...)
│       ├── sidebar.py         # sections repliables, vérificateur d'intervalles
│       ├── wave_editor.py     # mini-éditeur de template d'onde
│       ├── state.py           # dataclasses d'état côté UI
│       └── i18n.py            # table de chaînes bilingue (EN/FR)
├── ecg_app.py             # point d'entrée rétrocompatible léger
├── setup.py               # packaging pour `pip install -e .`
├── requirements.txt
└── README.md / README.fr.md
```

`ecg.core` ne dépend ni de `ecg.ui` ni de `ecg.io` — les algorithmes de
détection et de VRC sont des fonctions pures, testables, scriptables ou
réutilisables (par exemple depuis `ecg.batch`) sans lancer l'interface
graphique.

## Notes pour les contributeurs

- Aucune suite de tests automatisée n'existe pour l'instant ; les
  modifications touchant `ecg.core` (détection, analyse, délinéation des
  ondes) doivent être vérifiées sur un enregistrement réel ou synthétique,
  pas seulement par une vérification de compilation.
- L'interface est une grande fenêtre unique (`ecg/ui/app.py`) qui délègue à
  des contrôleurs par responsabilité dans `ecg/ui/*_controller.py` — voir
  la docstring de chaque contrôleur pour son périmètre.
