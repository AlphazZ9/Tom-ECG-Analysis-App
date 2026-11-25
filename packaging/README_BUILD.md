# Applications autonomes (Windows / macOS)

Les collègues n'ont pas besoin d'installer Python : GitHub Actions fabrique
une application prête à lancer pour chaque système, à partir de
`packaging/ecg.spec` (PyInstaller) et du workflow
`.github/workflows/build-apps.yml`.

| Fichier téléchargé                    | Pour qui                              |
|---------------------------------------|---------------------------------------|
| `Tom-ECG-Analysis-Windows.zip`        | PC Windows 10 / 11 (64 bits)          |
| `Tom-ECG-Analysis-macOS-arm64.zip`    | Mac Apple Silicon (M1, M2, M3, M4...) |
| `Tom-ECG-Analysis-macOS-x86_64.zip`   | Mac Intel (avant fin 2020)            |

Sur Mac : menu Pomme > « À propos de ce Mac ». « Puce Apple M... » = arm64,
« Processeur Intel » = x86_64.

## Publier une nouvelle version

### Méthode 1 : tag (crée une Release avec les 3 zips)

Depuis le dossier du projet, une fois le travail committé et poussé :

```bash
git tag v6.2.1          # numéro de version au format vX.Y.Z
git push origin v6.2.1
```

Le workflow « Build apps » démarre tout seul (onglet **Actions** du dépôt
GitHub). Comptez 10 à 20 minutes. À la fin, une **Release** `v6.2.1` est
créée dans l'onglet **Releases**, avec les trois zips attachés et des notes
de version générées automatiquement (modifiables ensuite sur GitHub).

Si une des trois plateformes échoue, la Release n'est pas créée : corriger,
puis supprimer et recréer le tag (`git tag -d v6.2.1`,
`git push origin :refs/tags/v6.2.1`, puis recommencer), ou choisir un
nouveau numéro.

### Méthode 2 : à la main (sans Release, pour tester)

Sur GitHub : onglet **Actions** > « Build apps » > **Run workflow** >
choisir la branche > **Run workflow**. Le bouton **Run workflow**
n'apparaît que lorsque `.github/workflows/build-apps.yml` est présent sur la
branche par défaut (`sync-local-work`) : fusionnez d'abord la PR qui
l'ajoute. Les zips apparaissent en bas de la
page du run, rubrique **Artifacts** (il faut être connecté à GitHub pour
les télécharger ; ils sont conservés 30 jours). Aucune Release n'est créée.

## Où les utilisateurs téléchargent

<https://github.com/AlphazZ9/Tom-ECG-Analysis-App/releases/latest>

Prendre le zip correspondant à son système dans la rubrique **Assets**.

## Installer et lancer

### Windows

1. Clic droit sur le zip > **Extraire tout...** (ne pas lancer depuis
   l'intérieur du zip).
2. Ouvrir le dossier `Tom ECG Analysis` et double-cliquer sur
   `Tom ECG Analysis.exe`. Le dossier `_internal` doit rester à côté de
   l'exe : déplacer le dossier entier, pas l'exe seul. Pour un raccourci :
   clic droit sur l'exe > **Afficher d'autres options** > **Envoyer vers** >
   **Bureau (créer un raccourci)**.
3. **Premier lancement** : l'application n'est pas signée, Windows
   SmartScreen affiche « Windows a protégé votre ordinateur ». Cliquer sur
   **Informations complémentaires**, puis **Exécuter quand même**. Cela n'est
   demandé qu'une fois. Un antivirus d'entreprise peut aussi la bloquer ou
   la mettre en quarantaine : voir avec le service informatique.

### macOS

1. Double-cliquer sur le zip : `Tom ECG Analysis.app` apparaît. Le glisser
   dans **Applications**.
2. **Premier lancement** : l'application n'est ni signée par un développeur
   Apple ni notarisée, macOS refuse donc de l'ouvrir par double-clic.
   - **Clic droit** (ou Ctrl + clic) sur l'app > **Ouvrir** > **Ouvrir**.
   - Sur macOS 15 (Sequoia) et plus récent, si ce bouton n'apparaît pas :
     essayer d'ouvrir l'app une fois, puis **Réglages Système** >
     **Confidentialité et sécurité**, descendre jusqu'au message sur
     « Tom ECG Analysis » > **Ouvrir quand même**, et confirmer avec le mot
     de passe du Mac.
   - Si macOS dit que l'app « est endommagée et ne peut pas être ouverte »
     (fréquent sur les Mac Apple Silicon pour une app téléchargée), ouvrir
     **Terminal** et taper :

     ```bash
     xattr -cr "/Applications/Tom ECG Analysis.app"
     ```

     puis relancer l'app normalement.
3. Le premier démarrage prend plus de temps (10 à 30 s) : macOS vérifie
   tous les fichiers. Les suivants sont plus rapides.

## Taille

- Windows : environ **230 Mo** une fois extrait, environ **100 Mo** pour le
  zip (mesuré sur un build local Windows, novembre 2026).
- macOS : du même ordre (non mesuré ici).

La taille vient surtout de NumPy, SciPy, scikit-learn, Matplotlib, pandas et
Tcl/Tk, embarqués dans l'application.

## Où sont les données de l'utilisateur

Comme avec la version Python, l'application range ses réglages, sessions,
base de données et modèle ML dans le dossier personnel (`~/.ecg_*`, soit
`C:\Users\<nom>\.ecg_*` sous Windows). Mettre à jour l'application (nouveau
zip) ne touche pas à ces fichiers.

## État des tests

- **Windows** : build testé localement (Python 3.12, PyInstaller 6.22) :
  l'application s'ouvre, un seul processus au repos, et les processus de
  calcul parallèle (délinéation des longs enregistrements) démarrent
  correctement depuis l'exe, sans relancer l'application en boucle.
- **macOS (arm64 et Intel) : non testé.** Aucun Mac n'était disponible : le
  workflow vérifie seulement que l'app est construite et que l'exécutable
  existe ; il ne la lance pas. À vérifier sur un vrai Mac à la première
  version : ouverture, chargement d'un fichier, export Excel, et une longue
  analyse (calcul parallèle).
- Le Mac Intel est construit sur l'image GitHub `macos-15-intel` (l'image
  `macos-13` a été retirée par GitHub). L'app demande **macOS 14 (Sonoma)
  ou plus récent** (`LSMinimumSystemVersion` du spec) : sur ces images,
  numpy/scipy sont installés en version « Accelerate », qui exige macOS 14.
  Sur un Mac plus ancien, utilisez le lanceur `Lancer ECG.command`.

## Construire en local (développeurs)

```bash
python -m pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm packaging/ecg.spec
```

Résultat dans `dist/` (`dist/Tom ECG Analysis/` sous Windows,
`dist/Tom ECG Analysis.app` sous macOS). `build/` et `dist/` sont ignorés
par git.

Icône : le chat rouge « narquois ». Les sources sont
`packaging/icon/icon.svg` (dessin complet, à partir de 64 px) et
`packaging/icon/icon-small.svg` (traits épais, 16 à 48 px). Après une
modification, régénérer tous les fichiers avec `python tools/build_icons.py`
(nécessite `pip install resvg-py pillow`) : `packaging/icon.ico`,
`packaging/icon.icns` (pris automatiquement par le spec) et
`ecg/ui/assets/` (icône de la fenêtre et de la barre des tâches).
