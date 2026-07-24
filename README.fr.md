# ECG Analysis — Outil d'électrophysiologie cardiaque murine

*[English version](README.md)*

Application de bureau (CustomTkinter) pour l'analyse d'enregistrements ECG
cardiaques de souris : détection des pics R, VRC (temporel/fréquentiel/non
linéaire/entropie multi-échelle), délinéation des ondes P-QRS-T,
classification du rythme (événements anormaux), et export vers Excel /
GraphPad Prism / PDF.

Conçue autour de la physiologie spécifique à la souris (Thireau et al. 2008 ;
Baudrie et al. 2007 ; Mitchell et al. 1998) plutôt qu'à partir de valeurs par
défaut humaines adaptées — plage de fréquence cardiaque, seuil pNN, bandes
fréquentielles de VRC et formule de correction du QTc sont toutes calibrées
pour la souris (FC de repos typique ~500 bpm, plage 180–900 bpm). Les plages
de référence de chaque métrique dépendent d'un contexte expérimental
sélectionnable (télémétrie/éveillée, isoflurane, kétamine-xylazine,
électrodes de surface, ou un contexte personnalisé défini par vous), et non
d'une unique plage « normale » générique.

**Nouveau sur l'application ?** Voir le [Guide utilisateur](docs/USER_GUIDE.md)
(en anglais) pour une visite guidée illustrée du flux de travail réel —
chargement d'un enregistrement, détection, analyse VRC, événements anormaux
et export.

## Fonctionnalités

### Signal & détection des pics R

- **Chargement du signal** — fichiers MATLAB `.mat`, format historique
  (v5/v6) et HDF5 (v7.3, exports Spike2), avec détection automatique du
  canal et de la fréquence d'échantillonnage.
- **Filtrage** — passe-bande, filtre coupe-bande, suppression de la dérive
  de ligne de base, avec un aperçu live avant/après filtrage.
- **Détection des pics R** — cinq méthodes interchangeables (automatique via
  NeuroKit2, Savitzky-Golay + dérivée, transformée en ondelettes continue,
  enveloppe maximale, et un détecteur par ML classique entraînable sur vos
  propres enregistrements validés), avec édition manuelle, undo/redo et
  curseur de seuil interactif.
- **Détecteur de pics R par ML** — entraîne une forêt aléatoire à partir des
  enregistrements marqués « Vérifié pour entraînement » ; la fenêtre
  d'entraînement affiche l'exactitude, le F1, la précision et le rappel sur
  un jeu de test isolé, ainsi qu'une matrice de confusion pour la
  classification binaire pic / non-pic.
- **Gestion des artefacts RR** — détection des doublons, intervalles
  non-physiologiques et battements ectopiques, avec une liste de candidats
  à valider avant toute suppression.
- **Qualité du signal** — un score composite de qualité (0–100) et une jauge
  calculés à partir de la corrélation au template de battement, plus deux
  indicateurs de bruit : un pourcentage par nombre de battements et un
  pourcentage pondéré par la durée « % du temps sous le seuil de
  corrélation » (ce dernier révèle les portions bruitées où très peu de
  battements sont détectés, que le seul pourcentage par battements
  sous-estimerait).

### Analyse VRC (6 vues liées)

- **RR/FC** — tachogramme avec détection automatique des pics
  (accélération/décélération brusque), navigation au clic ; histogramme de
  distribution des intervalles RR annoté avec le triangle ajusté de l'indice
  triangulaire VRC / TINN.
- **Domaine temporel** — l'ensemble complet des métriques `nk.hrv_time()`
  (SDNN, RMSSD, pNN6, SDANN, MedianNN, TINN, HTI...), avec un statut ✓ / ~ /
  ↑ / ↓ par rapport au contexte expérimental actif pour les métriques clés.
- **Fréquentiel** — PSD de Welch avec bandes VRC VLF/LF/HF spécifiques à la
  souris (pas la convention humaine 0,04–0,15/0,15–0,4 Hz), plus un
  diagramme radar où chaque axe est normalisé par rapport à *sa propre*
  plage de référence physiologique (et non par min-max entre les axes).
- **Non linéaire** — nuage de points de Poincaré avec ellipse SD1/SD2,
  entropie d'échantillon, DFA α1/α2, et une courbe d'**entropie
  multi-échelle (MSEn)** à travers les échelles de coarse-graining — détecte
  une perte de complexité que la SampEn à échelle unique peut manquer. La
  troncature du nombre de battements sur les longs enregistrements (pour le
  temps de calcul) est affichée à l'écran, jamais silencieuse.
- **Époques** — tendances VRC par fenêtres fixes (FC, SDNN, RMSSD, pNN6) avec
  ombrage des plages de référence et signalement visuel des fenêtres à
  faible nombre de battements.
- **Glissante** — tendances VRC par fenêtre glissante ; FC/SDNN/RMSSD/pNN6
  plus SD1/SD2/LF-normalisé/HF-normalisé/LF-HF en métriques optionnelles,
  toutes comparées aux plages de référence, avec un tableau de résultats à
  côté du graphique.

### Délinéation des intervalles

- Détection des repères P/Q/R/S/J/T par battement, avec un template d'onde
  calibrable par enregistrement et un vérificateur interactif.
- Distributions PR/QRS/QT/QTc (violon + boîte) par rapport aux plages de
  référence selon le contexte.
- Formule de correction QTc sélectionnable — **Mitchell** (par défaut,
  correction en racine cubique calibrée souris), **Bazett**, ou **Hodges**
  (correction linéaire par la FC, rapportée dans la littérature comme la
  plus performante chez le rongeur).
- Un diagramme de régression QT-vs-RR pour juger si la correction choisie a
  réellement supprimé la dépendance à la fréquence cardiaque, plutôt que de
  faire confiance aveuglément à un seul chiffre de formule.
- Dispersion du QT.

### Événements anormaux (classification du rythme)

- Épisodes de bradycardie/tachycardie, pauses, salves de battements
  ectopiques (ESV) et séries irrégulières, classés par rapport à la
  fréquence cardiaque de base propre à chaque enregistrement (pas un seuil
  fixe en bpm humain).
- **Signalement des délais de conduction AV** à partir d'une prolongation
  soutenue de l'intervalle PR, une fois la délinéation des intervalles
  effectuée — comble le manque où la classification du rythme ne voyait
  auparavant que le seul timing des pics R.
- Une frise chronologique colorée par type d'événement (regroupement visuel
  immédiat) accompagnée de cartes par épisode ; cliquer sur une carte ou un
  segment de la frise zoome l'ECG brut sur cet épisode, avec un mode
  d'édition manuelle des pics pour corriger la détection sur place.

### Contexte expérimental & plages de référence

- Quatre contextes physiologiques murins intégrés — **télémétrie / éveillée**,
  **isoflurane**, **kétamine/xylazine**, **électrodes de surface** — chacun
  avec ses propres plages de référence pour FC, RR, SDNN, RMSSD, pNN6,
  LF/HF, SD1/SD2 et PR/QRS/QT/QTc, sélectionnables depuis la fenêtre
  Paramètres.
- Un **contexte personnalisé** modifiable par l'utilisateur (par exemple
  pour une souche ou un protocole spécifique), avec son propre éditeur de
  plages de référence, persistant sur disque et sélectionnable partout aux
  côtés des contextes intégrés — chaque plage de référence, chaque statut
  ✓/~/↑/↓ et le diagramme radar se basent tous sur le contexte actif.

### Comparaison & export

- Comparaison de segments A/B avec tachogrammes RR superposés, un tableau
  complet des métriques (incluant LF/HF normalisés et SD1/SD2), et un
  **test U de Mann-Whitney** sur les distributions d'intervalles RR des deux
  segments.
- Persistance de session — sauvegarde/restauration d'une session d'analyse
  complète en JSON, plus un registre SQLite local des enregistrements
  récemment analysés.
- Export — classeurs Excel formatés, GraphPad Prism `.pzfx`, un rapport PDF
  d'une page (bande de signal, tableau des métriques clés, diagramme de
  Poincaré, radar VRC, résumé des événements anormaux, plage de référence du
  contexte), un PDF annoté par épisode pour les événements anormaux, et CSV.

### Interface

- Disposition Adaptive Workbench — barre d'outils, panneaux gauche/droit
  repliables, et un espace de travail central avec une mini-carte de
  défilement de position, permettant de récupérer de l'espace sur les petits
  écrans.
- Thèmes clair/sombre, propagés en direct dans toute l'interface, y compris
  chaque graphique matplotlib.

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
│   ├── core/             # pur NumPy/SciPy/pandas — sans UI, sans I/O fichier
│   │   ├── models.py         # constantes de physiologie, plages de référence du
│   │   │                     # contexte expérimental (intégrées + personnalisé
│   │   │                     # modifiable), dataclasses partagées
│   │   ├── filtering.py      # passe-bande, coupe-bande, normalisation
│   │   ├── detection.py      # détection des pics R, artefacts RR, classification
│   │   │                     # du rythme / événements anormaux (dont délai AV)
│   │   ├── analysis.py       # VRC temporel/fréquentiel/non linéaire (dont
│   │   │                     # entropie multi-échelle), intervalles
│   │   ├── wave_template.py  # calibration du template d'onde & délinéation
│   │   └── ml_detector.py    # détecteur de pics R par ML classique, entraînable
│   │                         # sur des enregistrements validés, matrice de
│   │                         # confusion sur jeu de test
│   ├── io/                # formats de fichiers et persistance
│   │   ├── loaders.py        # chargement du signal .mat / HDF5
│   │   ├── session.py        # sauvegarde/chargement de session (JSON)
│   │   ├── export.py         # exporteurs Excel et GraphPad Prism
│   │   └── db.py             # registre SQLite des enregistrements
│   └── ui/                # application de bureau CustomTkinter
│       ├── app.py             # fenêtre principale ECGApp (disposition Adaptive
│       │                      # Workbench)
│       ├── *_controller.py    # un contrôleur par responsabilité (SRP) :
│       │                      # analyse, détection, signal, navigation,
│       │                      # session, export
│       ├── theme.py           # préréglages de couleur/police, propagation live
│       ├── plots.py           # pont matplotlib ↔ canvas Tk
│       ├── dialogs.py         # fenêtres secondaires (thème, revue d'artefacts,
│       │                      # entraînement ML, éditeur de contexte
│       │                      # personnalisé...)
│       ├── sidebar.py         # sections repliables, vérificateur d'intervalles
│       ├── widgets.py         # fabriques de widgets réutilisables (tuiles stats, jauge de qualité)
│       ├── wave_editor.py     # mini-éditeur de template d'onde
│       └── state.py           # dataclasses d'état côté UI
├── ecg_app.py             # point d'entrée rétrocompatible léger
├── setup.py               # packaging pour `pip install -e .`
├── requirements.txt
└── README.md / README.fr.md
```

`ecg.core` ne dépend ni de `ecg.ui` ni de `ecg.io` — les algorithmes de
détection et de VRC sont des fonctions pures, testables, scriptables ou
réutilisables sans lancer l'interface graphique.

## Notes pour les contributeurs

- Aucune suite de tests automatisée n'existe pour l'instant ; les
  modifications touchant `ecg.core` (détection, analyse, délinéation des
  ondes) doivent être vérifiées sur un enregistrement réel ou synthétique,
  pas seulement par une vérification de compilation.
- L'interface est une grande fenêtre unique (`ecg/ui/app.py`) qui délègue à
  des contrôleurs par responsabilité dans `ecg/ui/*_controller.py` — voir
  la docstring de chaque contrôleur pour son périmètre.
- Les valeurs des plages de référence (dans `EXPERIMENTAL_CONTEXTS`,
  `ecg/core/models.py`) proviennent de la littérature citée pour chaque
  contexte ; si vous ajoutez ou ajustez les bornes d'un contexte intégré,
  citez la source dans son champ `description` comme le font déjà les
  quatre contextes existants — cette application sert à interpréter de
  vraies données animales, des valeurs sans source n'ont pas leur place
  ici. Utilisez l'éditeur de « Contexte personnalisé » intégré pour tout ce
  qui est spécifique à une souche ou un protocole plutôt que de coder en
  dur un nouveau préréglage.
