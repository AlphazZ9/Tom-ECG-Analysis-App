# ECG Analysis — Outil d'électrophysiologie cardiaque murine

*[English version](README.md)*

Application de bureau (CustomTkinter) pour l'analyse d'enregistrements ECG
cardiaques de souris : détection des pics R, VRC (temporel/fréquentiel/non
linéaire/entropie multi-échelle), indices autonomiques résolus dans le temps
(CSI / CPI, d'après Candia-Rivera et al. 2025) avec statistiques
verrouillées sur les événements, délinéation des ondes P-QRS-T,
classification du rythme (événements anormaux), et export vers Excel /
GraphPad Prism / PDF.

Conçue autour de la physiologie spécifique à la souris (Thireau et al. 2008 ;
Baudrie et al. 2007 ; Mitchell et al. 1998) plutôt qu'à partir de valeurs par
défaut humaines adaptées — plage de fréquence cardiaque, seuil pNN, bandes
fréquentielles de VRC et formule de correction du QTc sont toutes calibrées
pour la souris (FC de repos typique ~500 bpm, plage 180–900 bpm). Les plages
de référence spécifiques au protocole ne sont pas actuellement configurables
dans l'application ; interprétez les mesures selon la souche et les conditions
expérimentales de votre étude.

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

### Analyse VRC (7 vues liées)

- **RR/FC** — tachogramme avec détection automatique des pics
  (accélération/décélération brusque), navigation au clic ; histogramme de
  distribution des intervalles RR annoté avec le triangle ajusté de l'indice
  triangulaire VRC / TINN.
- **Domaine temporel** — l'ensemble complet des métriques `nk.hrv_time()`
  (SDNN, RMSSD, pNN6, SDANN, MedianNN, TINN, HTI...).
- **Fréquentiel** — PSD de Welch avec bandes VRC VLF/LF/HF spécifiques à la
  souris (pas la convention humaine 0,04–0,15/0,15–0,4 Hz).
- **Non linéaire** — nuage de points de Poincaré avec ellipse SD1/SD2,
  entropie d'échantillon, DFA α1/α2, et une courbe d'**entropie
  multi-échelle (MSEn)** à travers les échelles de coarse-graining — détecte
  une perte de complexité que la SampEn à échelle unique peut manquer. La
  troncature du nombre de battements sur les longs enregistrements (pour le
  temps de calcul) est affichée à l'écran, jamais silencieuse.
- **Époques** — tendances VRC par fenêtres fixes (FC, SDNN, RMSSD, pNN6) avec
  signalement visuel des fenêtres à faible nombre de battements.
- **Glissante** — tendances VRC par fenêtre glissante ; FC/SDNN/RMSSD/pNN6
  plus SD1/SD2/LF-normalisé/HF-normalisé/LF-HF en métriques optionnelles,
  avec un tableau de résultats à côté du graphique. Une option Poincaré
  inspirée de l'article ajoute CCD, SD1 et SD2 glissants avec covariance
  Ledoit–Wolf; les valeurs sont masquées sous 10 paires RR valides. Pour
  la méthode complète de l'article — CSI / CPI à partir d'une fenêtre de
  Poincaré par battement — utilisez la vue Autonomic.
- **Autonomic** — indices cardiaques sympathique / parasympathique
  résolus dans le temps (CSI / CPI) de Candia-Rivera et al. 2025, avec un
  choix d'estimateur de covariance, la puissance LF / HF par SPWVD en
  option, des statistiques avant / après verrouillées sur les annotations,
  un registre de cohorte et une fenêtre de calibration de ks / kp. Les
  gains de l'article (ks = 3, kp = 12) sont des valeurs rat, non validées
  chez la souris — voir
  [Indices autonomiques](#indices-autonomiques-csi--cpi--candia-rivera-et-al-2025).

### Délinéation des intervalles

- Détection des repères P/Q/R/S/J/T par battement, avec un template d'onde
  calibrable par enregistrement et un vérificateur interactif.
- Distributions PR/QRS/QT/QTc (violon + boîte).
- Formule de correction QTc sélectionnable (RR en ms) :
  - **Mitchell** (par défaut, calibrée souris ; Mitchell et al. 1998) —
    QTc = QT / √(RR / 100), c.-à-d. Bazett normalisée à un RR murin de
    100 ms (QTc égal au QT à ~600 bpm) ;
  - **Bazett** (humain) — QTc = QT / √(RR / 1000) ;
  - **Fridericia** (humain) — QTc = QT / ∛(RR / 1000).

  Bazett et Fridericia sont normalisées à un RR humain de 1 s et gonflent
  le QTc d'un facteur ~2 à 3,5 aux fréquences cardiaques murines : à n'utiliser
  que pour comparer avec des études qui les emploient. La relation QT–RR est
  faible chez la souris : lisez le **QT non corrigé** à côté de tout QTc. La
  formule utilisée est enregistrée dans les exports Excel (feuille Cover),
  Prism (notes du projet) et PDF.
- Un diagramme de régression QTc-vs-RR pour juger si la correction choisie a
  réellement supprimé la dépendance à la fréquence cardiaque (un QTc bien
  corrigé est plat en fonction du RR), plutôt que de faire confiance
  aveuglément à un seul chiffre de formule.
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

### Comparaison & export

- Comparaison de segments A/B avec tachogrammes RR superposés, un tableau
  complet des métriques (incluant LF/HF normalisés et SD1/SD2), et un
  **test U de Mann-Whitney** sur les distributions d'intervalles RR des deux
  segments.
- Persistance de session — sauvegarde/restauration d'une session d'analyse
  complète en JSON, plus un registre SQLite local des enregistrements
  récemment analysés et des résultats autonomiques par événement enregistrés
  pour les statistiques de cohorte.
- Export — classeurs Excel formatés, GraphPad Prism `.pzfx`, un rapport PDF
  d'une page (bande de signal, tableau des métriques clés, diagramme de
  Poincaré et résumé des événements anormaux), un PDF annoté par épisode
  pour les événements anormaux, et CSV.

### Interface

- Disposition Adaptive Workbench — barre d'outils, panneaux gauche/droit
  repliables, et un espace de travail central avec une mini-carte de
  défilement de position, permettant de récupérer de l'espace sur les petits
  écrans.
- Thèmes clair/sombre, propagés en direct dans toute l'interface, y compris
  chaque graphique matplotlib.

## Indices autonomiques (CSI / CPI) — Candia-Rivera et al. 2025

**VRC > Autonomic** calcule les indices cardiaques sympathique et
parasympathique résolus dans le temps de Candia-Rivera et al., *Modelling
the time-resolved modulations of cardiac activity in rats*, J Physiol 2025
([doi:10.1113/JP288400](https://doi.org/10.1113/JP288400)), en suivant le
code MATLAB publié par les auteurs
([github.com/diegocandiar/robust_hrv](https://github.com/diegocandiar/robust_hrv) :
`compute_CSI_CVI_rat.m`, `robustCovHRV.m`, `compute_rCSI_rCVI_type.m`,
`compute_PWVD.m`).

### Méthode

Chaque intervalle RR porte le temps du battement qui le termine. Pour
chaque battement, une fenêtre de Poincaré glissante de *T* secondes (3 s,
la valeur de l'article chez le rat) contient les intervalles terminés dans
[t − T, t]. Les paires successives de la fenêtre (RRₙ, RRₙ₊₁) donnent
**SD1** et **SD2** — racines carrées de la petite et de la grande valeur
propre de leur covariance 2 × 2 — et **D** (CCD), la distance du centre de
l'ellipse à l'origine, √(m₁² + m₂²) des moyennes des deux coordonnées ;
chaque fenêtre est datée au temps médian de ses intervalles. SD01, SD02 et
D0 calculés sur tout l'enregistrement recentrent les séries battement par
battement, rééchantillonnées à 20 Hz (spline cubique), et

- **CSI = D̃ + ks · SD2**, D̃ étant D réfléchi autour de sa moyenne (un
  cycle plus court augmente le CSI) ;
- **CPI = D + kp · SD1** (un cycle plus long augmente le CPI).

Les équations imprimées de l'article placent le D réfléchi sur le CPI ; le
code publié le réfléchit pour le CSI, ce qu'exige la physiologie —
l'application suit le code.

La version de l'application gère les lacunes : les intervalles RR hors des
bornes Min / Max RR de la barre latérale, ou qui enjambent un battement
supprimé par la correction d'artefacts, sont exclus ; les paires de
Poincaré ne les enjambent jamais ; une fenêtre de moins de **Min pairs**
paires adjacentes valides (10 par défaut) est masquée (vide, grisée) et
aucune série n'est interpolée à travers une lacune. Ce plancher est un
ajout de l'application — le code de référence calcule toutes les
fenêtres ; **Min pairs = 2** le reproduit. Une fenêtre de 3 s ne contient
10 paires qu'au-delà de ~200 bpm — abaissez Min pairs ou allongez la
fenêtre pour des souris bradycardes ou anesthésiées. Sans exclusion et avec
toutes les fenêtres calculées, les résultats sont identiques à ceux du code
de référence (la suite de tests les compare à une transcription
littérale).

### Options de covariance — et une divergence article / code

| Méthode | Ce qu'elle fait | Origine |
|---|---|---|
| **Rétrécissement robuste** (par défaut) | covariance à rétrécissement de Schäfer & Strimmer (2005) — variances rétrécies vers leur médiane, corrélation vers 0 (`robustCovHRV.m`) — avec un D en moyenne tronquée à 5 %. Une fenêtre où une coordonnée de Poincaré est constante n'a pas de corrélation définie et est masquée. | ce que dit le **texte** de l'article |
| **Code de référence rat** | `cov()` simple avec un D en moyenne tronquée à 5 % | ce qu'exécute réellement `compute_CSI_CVI_rat.m` publié (son appel à `robustCovHRV` est commenté) |
| **Exacte** | covariance simple, D en moyenne simple | `compute_rCSI_rCVI_type.m`, `'exact'` |
| **Approchée** | forme close SD1 = √(var(ΔRR)/2), SD2 = √(2 var(RR) − var(ΔRR)/2), D en moyenne simple | `compute_rCSI_rCVI_type.m`, `'approximate'` |
| **Ledoit–Wolf** | rétrécissement sklearn vers une identité pondérée — pour comparaison uniquement | non utilisée par l'article |

Le texte de l'article et sa fonction publiée pour le rat divergent : le
texte indique l'estimateur robuste (à rétrécissement), le code utilise la
covariance simple avec moyennes tronquées. Les deux sont proposés ; le
défaut suit le texte, et la méthode est enregistrée avec chaque résultat et
chaque export. Les deux estimateurs à rétrécissement arrondissent
l'ellipse de Poincaré sur les fenêtres courtes, d'une quantité qui dépend
des données. Le robuste conserve SD1² + SD2² et, dans chaque fenêtre,
augmente SD1 et diminue SD2 (sur des fenêtres souris synthétiques de ~25
paires : SD1 +4 à +7 %, SD2 −1 à −5 % par rapport à la covariance exacte).
Ledoit–Wolf — l'estimateur des SD1 (LW) / SD2 (LW) façon article de la vue
Glissante — rétrécit vers une autre cible, en général davantage
(typiquement SD1 +3 à +12 %, SD2 −4 à −10 % sur des fenêtres de 25–35
paires), et peut même diminuer SD1 dans certaines fenêtres.

### Calibration rat — non validée chez la souris

**ks = 3 et kp = 12 sont les valeurs de l'article pour le rat
spontanément hypertendu** (repos vs dobutamine : point d'inflexion du Z de
Wilcoxon apparié sur k = 1–20, vérifié ensuite sur des rats WKY). Elles
**n'ont pas été validées chez la souris**. La vue, ses graphiques et chaque
export (feuille Cover d'Excel, notes du projet Prism, en-tête du rapport
PDF) indiquent les gains et cette réserve. Calibrez-les sur votre propre
cohorte (ci-dessous) et rapportez les gains avec la cohorte d'origine.

### LF / HF résolus dans le temps (SPWVD)

Cochez **LF/HF (SPWVD)** pour ajouter les puissances LF et HF de la
distribution de pseudo-Wigner–Ville lissée, portage de `compute_PWVD.m`
(noyau dans le domaine d'ambiguïté d'Orini et al. 2012) : les intervalles
RR valides sont rééchantillonnés par spline à 20 Hz, sans tendance linéaire,
puis transformés en signal analytique ; la distribution de Wigner–Ville
(4096 retards) est filtrée dans le domaine d'ambiguïté par le noyau
exp(−π (μ²)^λ), μ = (τ/τ₀)² + (ν/ν₀)², et sommée sur les bandes LF / HF du
préréglage VRC (souris par défaut LF 0,4–1,5 Hz, HF 1,5–5 Hz ; l'article
utilisait chez le rat 0,1–1 et 1–3,5 Hz). Les constantes du noyau ν₀ = 0,03,
τ₀ = 0,06, λ = 0,3 sont celles codées en dur dans `compute_PWVD.m` — le
seul jeu de la boîte à outils, sans variante par espèce (les valeurs par
défaut du fichier sont Frr = 4 Hz et les bandes humaines) ; l'article les a
appliquées à 20 Hz chez le rat. Elles sont définies en unités normalisées :
leur étendue en secondes et en Hz dépend du rééchantillonnage à 20 Hz et
des 4096 retards. Elles ne sont pas validées chez la souris.

Les lacunes RR de plus de 2 s scindent la série : chaque portion entre
elles est calculée comme une série à part (sa propre spline, sa tendance et
son signal analytique — rien n'est lissé à travers une lacune, alors que le
code de référence interpole tout) et les lacunes sont vides, de même que
les portions trop courtes pour être calculées (moins de 10 s avec les
bandes souris : deux périodes de la borne basse LF et le noyau de retard
de ~12 s). Les lacunes plus courtes sont interpolées par la spline ; si
elles couvrent plus de 5 % du temps, le calcul LF / HF est refusé. Les
longs enregistrements sont traités par blocs chevauchants (cœurs de 120 s,
marges de 60 s) pour borner la mémoire ; par rapport à un calcul en une
passe, les rapports après / avant sur 30 s concordent à ~0,1–1 % près
(selon les données : l'écart est maximal après un grand changement de FC /
VRC dans une marge de bloc) et les valeurs échantillon par échantillon à
quelques % près (jusqu'à ~6 % sur les échantillons de faible puissance).
La puissance n'est pas normalisée (u.a.) : comparez-la au sein d'un
enregistrement (avant vs après un événement, tendance LF vs HF), pas avec
les puissances de Welch. Comptez environ 40 s de calcul par heure
d'enregistrement.

### Statistiques verrouillées sur les événements

Le bouton **Event-locked** de la vue Autonomic applique le protocole
statistique de l'article aux indices calculés :

1. Marquez chaque changement de condition (injection, stimulus) par une
   **annotation** : son **début** est l'instant de l'événement. Choisissez
   toutes les annotations ou un seul libellé dans **Events**.
2. Réglez **Pre** / **Post** (30 s / 30 s par défaut) puis cliquez
   **⟳ Compute event-locked**. Le tableau par événement donne la moyenne
   de CSI, CPI, FC, D, SD1, SD2 (et LF / HF) sur chaque fenêtre et le delta
   après − avant (**📋 Copy for Excel** copie une ligne par événement). Le
   graphique montre la médiane ± MAD de groupe sur les événements de
   l'enregistrement, sur un axe −pre…+post avec l'instant de l'événement en
   0, en z-score par enregistrement pour l'affichage comme dans les figures
   de l'article (case **z-score** ; le tableau et les tests utilisent
   toujours les valeurs brutes). Un événement est valide quand **aucune
   autre annotation ne commence dans sa fenêtre avant ou après, ni à son
   instant** — la durée minimale par condition de l'article : des doses
   successives plus proches que les fenêtres contamineraient la ligne de
   base et la réponse l'une de l'autre, ces événements sont donc INVALIDES
   (toutes les annotations comptent, y compris celles que le filtre de
   libellé écarte) — et quand ses deux fenêtres sont entièrement dans la
   série CSI / CPI et couvertes à au moins 80 % par des fenêtres calculées.
3. Le test de Wilcoxon intra-enregistrement sous le tableau est purement
   descriptif : les événements d'un même animal sont des mesures répétées
   de cet animal, pas un échantillon de population. Il donne z, le p du
   test (scipy : exact pour les petits échantillons) et p(z), le p normal
   de ce z (le p approché de `signrank` sous MATLAB) — les deux peuvent
   tomber de part et d'autre de 0,05.
4. **💾 Save to cohort registry** enregistre les valeurs avant / après /
   delta par événement dans le registre SQLite local avec le **Subject ID**
   et le **Project name** de la barre latérale et la configuration
   (méthode, fenêtre, ks, kp, pre / post) — une ligne par événement et par
   méthode / fenêtre. Avec le filtre *All*, les événements précédemment
   enregistrés pour cet enregistrement avec la même méthode et la même
   fenêtre sont remplacés. **Renseignez le Subject ID de chaque animal
   avant d'enregistrer** : l'enregistrement avertit (orange) quand il
   ressemble à la valeur par défaut (`subject_01`) ou quand un autre
   enregistrement du projet est déjà classé sous ce sujet, car la
   calibration par sujet regrouperait ces enregistrements comme un seul
   animal. **Export CSV…** écrit les mêmes lignes dans un fichier.

### Calibrer ks / kp sur votre cohorte

CSI et CPI sont linéaires en leurs gains : pour toute paire avant / après,
ΔCSI = −ΔD + ks · ΔSD2 et ΔCPI = ΔD + kp · ΔSD1 (les constantes de
recentrage s'annulent). Les deltas par événement de D, SD1 et SD2
enregistrés suffisent donc à recalculer le contraste pour tout k sans
retraiter un enregistrement. **⚖ Calibrate ks/kp…** lit le registre (filtré
par libellé et projet, une configuration méthode / fenêtre à la fois —
jamais mélangées) et trace le Z de Wilcoxon apparié (après vs avant) de CSI
et CPI pour k = 1–20. Elle s'ouvre sur le libellé qui a le plus
d'événements — un seul contraste de conditions, comme le repos vs
dobutamine de l'article ; **All labels** regroupe des contrastes différents
et le signale. La règle de l'article est le **point d'inflexion de Z(k)**,
à partir duquel Z forme un plateau ; la fenêtre marque un coude calculé
comme approximation objective de ce point (Kneedle sur Z après son dernier
changement de signe) — vérifiez-le sur la courbe. Agrégez en **moyenne par
sujet** (le défaut) pour qu'un animal ayant plusieurs événements ne compte
qu'une fois : tester les événements comme des échantillons indépendants
est une pseudo-réplication (les événements sans Subject ID sont regroupés
par enregistrement). L'article a calculé Z sur les essais (événements :
41 essais SHR), donc **Per event** reproduit sa procédure ; la moyenne par
sujet est le défaut, plus strict, de l'application.

Lisez la courbe avec trois réserves, toutes affichées dans la fenêtre :

- **Z est une statistique de rangs.** Il évolue par paliers (une unité qui
  change de signe, deux unités qui échangent leur rang) et ne peut dépasser
  un plafond fixé par le nombre d'unités (2,20 pour 6, 2,52 pour 8, 3,06
  pour 12). Une petite cohorte reste souvent sur ce plafond, où Z ne peut
  pas départager les gains : la fenêtre le signale, donne les plages de k
  où |Z| est maximal, et ne propose aucun coude reposant sur moins de 3
  paliers de rang.
- **Z(k) peut changer de signe** quand les termes de fréquence cardiaque et
  de VRC évoluent en sens opposés (par ex. un médicament tachycardisant qui
  diminue aussi SD2) : l'indice montre alors un sens aux petits k et le
  sens opposé aux grands k. La fenêtre donne le sens sur chaque portion ;
  un coude sur la portion inversée est marqué ⚠ et Apply demande
  confirmation — choisissez le gain selon le sens attendu par la
  physiologie.
- **Test sur l'échantillon de calibration.** Des gains choisis sur une
  cohorte puis testés sur cette même cohorte donnent des p optimistes
  (choisir k parmi 20 valeurs est une sélection que le Bonferroni ×2 ne
  corrige pas) ; l'article a ajusté les gains sur des rats SHR et les a
  validés sur des rats WKY. Les tests aux coudes de cette cohorte sont
  marqués *in-sample* : calibrez sur un projet ou un libellé et testez sur
  un autre.

Le tableau donne, aux gains actuels, n, la médiane ± MAD avant et après,
z, p, p(z) et le p corrigé de Bonferroni sur les deux indices.
**Apply ks = …, kp = …** écrit les valeurs du coude dans les champs ks / kp
de la vue Autonomic ; rien n'est recalculé tant que vous n'appuyez pas sur
**⟳ Compute** (puis ⟳ Compute event-locked) — d'ici là, l'état de la vue
indique que les champs diffèrent du résultat affiché.

### Exports

Avec un résultat Autonomic, le classeur Excel gagne une feuille
**Autonomic** (fenêtres SD1 / SD2 / CCD battement par battement, SD01 /
SD02 / D0 de l'enregistrement et tous les paramètres), **Autonomic_Series**
(CSI / CPI / FC / D / SD1 / SD2 à 20 Hz), **Autonomic_LFHF** et
**EventLocked** ; le fichier Prism gagne des tableaux CSI / CPI, D / SD1 /
SD2, LF / HF et avant / après par événement ; les figures de VRC >
Autonomic sont incluses dans les exports de figures et ZIP. La feuille
Cover d'Excel, les notes du projet Prism et l'en-tête du rapport PDF
indiquent la méthode, la fenêtre, ks / kp et la réserve de calibration rat,
ainsi que la portion couverte par les indices (premier / dernier pic R,
fenêtre d'analyse) — signalée quand ce n'est pas la fenêtre d'analyse des
résultats VRC principaux. La vue Autonomic n'a besoin que des pics : avant
Analyze, **Export Excel** écrit les résultats Autonomic seuls (une feuille
Cover et les feuilles Autonomic) ; les autres exports nécessitent Analyze.
Un résultat calculé sur une autre fenêtre d'analyse que l'actuelle est
signalé en orange dans la vue.

### Références

- Candia-Rivera D., Carrion-Falgarona S., de Vico Fallani F., Chavez M.
  (2025). Modelling the time-resolved modulations of cardiac activity in
  rats: a study on pharmacological autonomic stimulation. *J Physiol*.
  [doi:10.1113/JP288400](https://doi.org/10.1113/JP288400)
- Candia-Rivera D., de Vico Fallani F., Chavez M. (2025). Robust and
  time-resolved estimation of cardiac sympathetic and parasympathetic
  indices. *R Soc Open Sci* 12:240750.
  [doi:10.1098/rsos.240750](https://doi.org/10.1098/rsos.240750)
- Code de référence : [github.com/diegocandiar/robust_hrv](https://github.com/diegocandiar/robust_hrv)
- Schäfer J., Strimmer K. (2005). A shrinkage approach to large-scale
  covariance matrix estimation and implications for functional genomics.
  *Stat Appl Genet Mol Biol* 4:32.
- Orini M., Bailón R., Mainardi L. T., Laguna P., Flandrin P. (2012).
  Characterization of dynamic interactions between cardiovascular signals
  by time-frequency coherence. *IEEE Trans Biomed Eng* 59(3):663–673.
- Satopää V., Albrecht J., Irwin D., Raghavan B. (2011). Finding a
  "kneedle" in a haystack: detecting knee points in system behavior.
  *ICDCS Workshops* (l'approximation du coude).

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

```text
.
├── ecg/
│   ├── __init__.py       # version du package, aperçu de l'architecture
│   ├── __main__.py       # point d'entrée `python -m ecg`
│   ├── core/             # pur NumPy/SciPy/pandas — sans UI, sans I/O fichier
│   │   ├── models.py         # constantes de physiologie et dataclasses partagées
│   │   ├── filtering.py      # passe-bande, coupe-bande, normalisation
│   │   ├── detection.py      # détection des pics R, artefacts RR, classification
│   │   │                     # du rythme / événements anormaux (dont délai AV)
│   │   ├── analysis.py       # VRC temporel/fréquentiel/non linéaire (dont
│   │   │                     # entropie multi-échelle), intervalles
│   │   ├── autonomic.py      # CSI / CPI résolus dans le temps (Candia-Rivera 2025)
│   │   ├── spwvd.py          # LF / HF résolus dans le temps (pseudo-WVD lissée)
│   │   ├── event_locked.py   # moyennes avant / après, Wilcoxon, calibration
│   │   │                     # ks / kp, agrégation par sujet
│   │   ├── wave_template.py  # calibration du template d'onde & délinéation
│   │   └── ml_detector.py    # détecteur de pics R par ML classique, entraînable
│   │                         # sur des enregistrements validés, matrice de
│   │                         # confusion sur jeu de test
│   ├── io/                # formats de fichiers et persistance
│   │   ├── loaders.py        # chargement du signal .mat / HDF5
│   │   ├── session.py        # sauvegarde/chargement de session (JSON)
│   │   ├── export.py         # exporteurs Excel et GraphPad Prism
│   │   └── db.py             # registre SQLite des enregistrements + table
│   │                         # de cohorte des événements
│   └── ui/                # application de bureau CustomTkinter
│       ├── app.py             # fenêtre principale ECGApp (disposition Adaptive
│       │                      # Workbench)
│       ├── *_controller.py    # un contrôleur par responsabilité (SRP) :
│       │                      # analyse, détection, signal, navigation,
│       │                      # session, export, événements
│       ├── autonomic_view.py  # aides de VRC > Autonomic (pures, testées)
│       ├── event_locked_view.py # panneau Event-locked + calibration ks / kp
│       │                      # (aides pures, testées)
│       ├── theme.py           # préréglages de couleur/police, propagation live
│       ├── plots.py           # pont matplotlib ↔ canvas Tk
│       ├── dialogs.py         # fenêtres secondaires (thème, revue d'artefacts,
│       │                      # entraînement ML, paramètres d'analyse
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

- Lancez `python -m pytest -q` pour la suite automatisée. Elle couvre la
  détection, la validation des intervalles VRC, les statistiques de segments
  et la persistance/séparation ML ; l'interface, les formats de fichiers et
  la délinéation des ondes nécessitent encore davantage de tests d'intégration.
- L'interface est une grande fenêtre unique (`ecg/ui/app.py`) qui délègue à
  des contrôleurs par responsabilité dans `ecg/ui/*_controller.py` — voir
  la docstring de chaque contrôleur pour son périmètre.
- Interprétez les mesures selon les données de référence adaptées à la souche
  et au protocole ; l'application ne compare actuellement pas les métriques
  à des plages spécifiques au contexte expérimental.
