# ECG Analysis — Outil d'électrophysiologie cardiaque murine

[![Tests](https://github.com/AlphazZ9/Tom-ECG-Analysis-App/actions/workflows/tests.yml/badge.svg?branch=sync-local-work)](https://github.com/AlphazZ9/Tom-ECG-Analysis-App/actions/workflows/tests.yml)
[![Latest release](https://img.shields.io/github/v/release/AlphazZ9/Tom-ECG-Analysis-App)](https://github.com/AlphazZ9/Tom-ECG-Analysis-App/releases/latest)

*[English version](README.md)*

Application de bureau pour analyser des enregistrements ECG de **souris** :
détection des pics R, variabilité de la fréquence cardiaque (VRC), indices
autonomiques CSI / CPI, mesure des intervalles PR / QRS / QT / QTc,
événements anormaux du rythme, et export vers Excel, GraphPad Prism et PDF.
Tout est calibré pour la physiologie de la souris (FC de repos typique
~500 bpm, plage 180–900 bpm ; Thireau et al. 2008 ; Baudrie et al. 2007 ;
Mitchell et al. 1998), et non adapté de valeurs humaines. Les plages de
référence spécifiques au protocole ne sont pas configurables : interprétez
les mesures selon la souche et les conditions de votre étude.

## Télécharger / Démarrer

1. **Application prête à l'emploi (recommandé).** Téléchargez le zip de
   votre système sur la page
   [Releases](https://github.com/AlphazZ9/Tom-ECG-Analysis-App/releases/latest)
   (Windows, macOS Apple Silicon ou macOS Intel ; macOS 14 ou plus récent
   pour l'app), décompressez-le et lancez l'application.
2. **Ou le lanceur en un clic** (sans installer Python soi-même) :
   récupérez le dossier du dépôt (`git clone`, ou sur GitHub
   **Code > Download ZIP**, puis décompressez le ZIP en entier) et
   double-cliquez sur `Lancer ECG.bat` (Windows) ou `Lancer ECG.command`
   (macOS) — détails [ci-dessous](#lanceur-en-un-clic).
3. **Ou, pour les développeurs :** `pip install -r requirements.txt` puis
   `python ecg_app.py` — voir [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
   (en anglais).

**Mettre à jour.** La version s'affiche à côté du nom de l'application,
dans l'en-tête (par ex. *ECG Analysis v6.2.2*). Pour l'application
téléchargée, prenez le zip de la
[dernière version](https://github.com/AlphazZ9/Tom-ECG-Analysis-App/releases/latest)
et remplacez l'ancienne copie (sessions et réglages sont conservés : ils ne
sont pas dans l'application). Depuis le code (VS Code, lanceur) :
`git checkout sync-local-work`, puis `git pull`.

**Premier lancement :** les applications ne sont pas signées, le système
avertit une fois. Windows (SmartScreen) : *Informations complémentaires >
Exécuter quand même*. macOS : clic droit (ou Ctrl+clic) sur l'app >
**Ouvrir** ; sous macOS 15 et plus, *Réglages Système > Confidentialité et
sécurité > Ouvrir quand même*. Pas à pas complet :
[packaging/README_BUILD.md](packaging/README_BUILD.md#installer-et-lancer).

### Lanceur en un clic

- **Premier lancement : une connexion Internet est nécessaire.** Le lanceur
  installe [uv](https://docs.astral.sh/uv/) s'il est absent, un Python 3.12
  privé (jamais le Python du système, ni celui d'Apple avec son ancien
  Tk 8.5) et les bibliothèques de `requirements.txt` — quelques minutes et
  environ 400 Mo, avec une fenêtre de progression. Ensuite l'application
  démarre en quelques secondes, y compris hors ligne ; les bibliothèques ne
  sont réinstallées que si `requirements.txt` ou
  `packaging/constraints.txt` (les versions testées) change.
- **Dossier utilisé** (supprimez-le pour tout réinstaller ; vos réglages
  dans `~/.ecg_*` ne sont pas touchés) : Windows `%LOCALAPPDATA%\TomECG`
  (cache uv : `%LOCALAPPDATA%\uv\cache`) ; macOS
  `~/Library/Application Support/TomECG` (cache : `~/.cache/uv`).
- **Windows :** pas de fenêtre console ; les erreurs Python vont dans
  `dernier_lancement.log` du dossier ci-dessus. `"Lancer ECG.bat" debug`
  depuis une invite de commandes garde la console ouverte. SmartScreen :
  *Informations complémentaires > Exécuter quand même*.
- **macOS (Gatekeeper), une seule fois :** sous macOS 15 (Sequoia) ou plus
  récent, double-cliquez, cliquez sur **OK**, puis *Réglages Système >
  Confidentialité et sécurité* > **Ouvrir quand même** à côté de
  `Lancer ECG.command`, et confirmez avec votre mot de passe. Sous macOS 14
  ou plus ancien : clic droit (ou Ctrl+clic) > **Ouvrir**, puis confirmez.
  Sinon, dans le Terminal : `xattr -d com.apple.quarantine
  "/chemin/vers/Lancer ECG.command"` (glissez le fichier dans le Terminal
  pour écrire son chemin), et `chmod +x` sur le même chemin si macOS
  indique que le fichier n'est pas exécutable. Le lanceur macOS n'a pas
  encore été testé sur un vrai Mac.

## Ce que fait l'application

- **Signal et pics R** — fichiers Spike2 `.smr` et MATLAB `.mat` (v5/v6 et v7.3/HDF5, exports Spike2), filtrage, cinq détecteurs dont un ML entraînable, correction des artefacts RR, score de qualité → [détails](docs/METHODES.md#signal--détection-des-pics-r)
- **Onglets** — Signal & Detection, Rhythm & HRV, Time course, Compare segments, Morphology & Intervals, Events, Group (tous les animaux d'une expérience), Report → [détails](docs/METHODES.md#onglets)
- **VRC** — domaines temporel, fréquentiel (bandes souris, ms²) et non linéaire (Poincaré, DFA, entropie multi-échelle), époques et fenêtre glissante → [détails](docs/METHODES.md#analyse-vrc)
- **Indices autonomiques CSI / CPI** (Candia-Rivera et al. 2025), LF / HF par SPWVD, statistiques avant / après un événement, calibration ks / kp ; les gains de l'article sont des valeurs rat, non validées chez la souris → [détails](docs/METHODES.md#indices-autonomiques-csi--cpi--candia-rivera-et-al-2025)
- **Intervalles** — repères P/Q/R/S/J/T, PR, QRS, QT, QTc (Mitchell par défaut), médiane glissante de battements → [détails](docs/METHODES.md#délinéation-des-intervalles)
- **Événements anormaux** — brady/tachycardies, pauses, battements prématurés, séries irrégulières, délais de conduction AV, classés par rapport à la fréquence de base de chaque enregistrement → [détails](docs/METHODES.md#événements-anormaux-classification-du-rythme)
- **Comparaison et export** — segments A/B, sessions, registre de cohorte, Excel, Prism `.pzfx`, PDF, CSV, provenance dans chaque rapport → [détails](docs/METHODES.md#comparaison--export)
- **Interface** — thèmes clair / sombre, utilisable dès 1280 px de large → [détails](docs/METHODES.md#interface)

## Déroulement type

1. [Ouvrir l'application](docs/USER_GUIDE.md#launching-the-app) et [charger un enregistrement](docs/USER_GUIDE.md#loading-a-recording) `.smr` ou `.mat`.
2. [Détecter les pics R](docs/USER_GUIDE.md#r-peak-detection) et [vérifier la détection](docs/USER_GUIDE.md#reading-the-detection-view) : corriger à la main si besoin, valider les artefacts et marquer les passages trop bruités (exclus de la FC et de la VRC).
3. Cliquer sur **Analyze** et lire la [VRC](docs/USER_GUIDE.md#hrv-analysis).
4. Si besoin, calculer les [indices autonomiques](docs/USER_GUIDE.md#autonomic-indices-csi--cpi--candia-rivera-et-al-2025) autour de vos annotations et mesurer les [intervalles](docs/USER_GUIDE.md#interval-delineation) PR / QRS / QT / QTc (vérifier les repères sur le battement moyen).
5. Passer en revue les [événements anormaux](docs/USER_GUIDE.md#abnormal-events).
6. Vérifier l'onglet [Report](docs/USER_GUIDE.md#report-tab), puis [exporter et enregistrer la session](docs/USER_GUIDE.md#exporting--saving-your-work).

## Documentation

- [Guide utilisateur](docs/USER_GUIDE.md) (en anglais) — visite guidée illustrée de l'application
- [Méthodes et détails](docs/METHODES.md) — ce que calcule chaque module, valeurs par défaut, réserves, historique des changements ([English](docs/METHODS.md))
- [Glossaire des variables](docs/GLOSSAIRE_VARIABLES.md) — définition de chaque variable calculée
- [Développement](docs/DEVELOPMENT.md) (en anglais) — installation depuis les sources, arborescence, tests, contribution
- [Publier une version](packaging/README_BUILD.md) — construction et publication des applications autonomes

## Citation / références

Les méthodes reposent sur la littérature citée dans
[docs/METHODES.md](docs/METHODES.md), notamment Candia-Rivera et al. 2025
pour les indices CSI / CPI — liste complète dans
[Références](docs/METHODES.md#références).
