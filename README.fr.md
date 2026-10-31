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

### Onglets

Cinq onglets d'analyse et un rapport, organisés par question : **Signal & Detection**,
**Rhythm & HRV**, **Time course** (Epochs, Sliding, Autonomic, Event-locked),
**Morphology & Intervals** (étapes : battement moyen, repères, intervalles), **Events**
et **Report** (provenance, état des modules, Include, exports et rapport texte). Un
bandeau de contexte au-dessus de chaque onglet indique sur quoi reposent les
valeurs ; chaque onglet porte un état (up to date, out of date, not computed) ;
les écrans, le rapport texte, les valeurs clés du PDF et les tableaux **Metrics** d'Excel
et de Prism lisent un seul tableau : valeur, unité, n de base et état (les autres feuilles
et tables contiennent les séries sous-jacentes). Une session enregistrée sur un ancien
onglet se rouvre à sa nouvelle place.

Les entrées plus anciennes ci-dessous nomment les onglets de leur époque : *Summary*
est devenu Report, les sous-vues *HRV* sont sur Rhythm & HRV et Time course, *Rolling*
est devenu Sliding, *Beat Template* et *Intervals* sont Morphology & Intervals,
*Abnormal Events* est Events.

### Analyse VRC

- **RR/FC** — tachogramme avec détection automatique des pics
  (accélération/décélération brusque), navigation au clic ; histogramme de
  distribution des intervalles RR annoté avec le triangle ajusté de l'indice
  triangulaire VRC / TINN.
- **Domaine temporel** — l'ensemble complet des métriques `nk.hrv_time()`
  (SDNN, RMSSD, pNN6, SDANN, MedianNN, TINN, HTI...).
- **Fréquentiel** — PSD de Welch avec bandes VRC VLF/LF/HF spécifiques à la
  souris (pas la convention humaine 0,04–0,15/0,15–0,4 Hz). Un seul spectre
  pour tout : les intervalles RR valides sont rééchantillonnés à 20 Hz
  (spline cubique sur leurs vrais instants) puis la méthode de Welch est
  appliquée (fenêtre de Hann, recouvrement 75 %, résolution ≈ 0,02 Hz) ; le
  graphe PSD trace exactement ce spectre, et VLF / LF / HF sont ses
  intégrales sur les bandes ombrées, en **puissances absolues (ms²)**.
  **LFn / HFn** sont les unités normalisées de la Task Force, LF ou HF /
  (LF + HF) ; LF/HF = LF / HF. Le même estimateur sert à la VRC glissante et
  à la comparaison de segments.
- **Non linéaire** — nuage de points de Poincaré avec ellipse SD1/SD2,
  entropie d'échantillon, DFA α1/α2, et une courbe d'**entropie
  multi-échelle (MSEn)** à travers les échelles de coarse-graining — détecte
  une perte de complexité que la SampEn à échelle unique peut manquer.
  SD1/SD2 de Poincaré, la DFA et les indices d'asymétrie / de fragmentation
  utilisent **tous les battements valides de la fenêtre d'analyse**. Seul le
  bloc d'entropie, lent (O(N²)) — SampEn, ApEn, FuzzyEn, MSE, CD, … — est
  plafonné pour le temps de calcul, à **Entropy cap (RR)** intervalles
  valides (2000 par défaut, réglable dans l'onglet Non-linear ; 0 = pas de
  plafond, une valeur négative revient à 2000), pris comme le bloc contigu contenant le moins d'intervalles
  exclus ; le nombre et l'intervalle de temps de ce bloc sont affichés dans
  la ligne d'état, sur la courbe MSE, dans le rapport Summary et sur la
  feuille Cover d'Excel.
- **Une seule définition par métrique.** La FC moyenne vaut partout
  **60000 / NN moyen** (Summary, barre de KPI, Excel, comparaison de
  segments, VRC glissante, époques, titres des tachogrammes) ; l'écart-type
  et les min / max de FC décrivent la FC instantanée 60000 / RRᵢ — écart-type
  d'échantillon (n − 1) et **2ᵉ / 98ᵉ percentiles**, identiques dans le
  Summary, la barre de KPI, Excel et la comparaison de segments — et sont
  libellés comme tels. **SD1 / SD2** de Poincaré n'ont qu'une définition —
  écart-type de (RRᵢ − RRᵢ₊₁)/√2 et de (RRᵢ + RRᵢ₊₁)/√2 sur les paires
  d'intervalles valides adjacents (jamais à travers un intervalle exclu) —
  utilisée par le panneau Non linéaire, la VRC glissante, la comparaison de
  segments et le Summary / les exports. Les estimateurs Ledoit-Wolf (SD1 LW
  / SD2 LW de la VRC glissante) et ceux de la vue Autonomic sont des
  variantes nommées explicitement. **Chiffres modifiés (cette version) :**
  les versions précédentes calculaient toutes les métriques non linéaires,
  SD1/SD2 compris, sur les **1000 premiers battements** seulement (quelques
  minutes d'un long enregistrement) ; prenaient la FC moyenne comme la
  moyenne de la FC instantanée (biaisée vers le haut, typiquement de 0,1 à
  1 bpm, davantage si la variabilité est forte) ; et tiraient VLF / LF / HF
  du spectre de NeuroKit (rééchantillonnage à 100 Hz, PSD normalisée par son
  maximum : des « puissances » en unités arbitraires, étiquetées à tort
  n.u.) alors que le graphe PSD traçait un autre spectre, avec LFn / HFn =
  LF ou HF / puissance totale incluant VLF et 5–20 Hz. La comparaison de
  segments utilisait un troisième calcul de Welch, et ses écart-type / min /
  max de FC étaient les extrêmes absolus (le Summary utilisait des
  percentiles et un écart-type de population ; l'écart-type de FC est
  désormais très légèrement plus élevé). Une ancienne session restaurée voit
  sa FC moyenne, son écart-type et sa plage de FC recalculés à partir des
  intervalles RR valides enregistrés (indiqué dans la barre d'état) ; un
  ancien résultat fréquentiel ne peut pas être recalculé sans relancer
  l'analyse : il est conservé mais étiqueté **« legacy NeuroKit spectrum »**
  (unités arbitraires, LFn = LF / TP) dans le tableau Frequency, le Summary,
  le PDF, les feuilles Cover / HRV_Frequency d'Excel et Prism, au lieu de
  ms² / n.u. Relancez Frequency et Non-linear HRV sur les anciennes
  sessions. Les en-têtes de la feuille brute **HRV_Frequency** portent leur
  unité (p. ex. `HRV_LF (ms²)`, `HRV_LFn (fraction of LF+HF)` — des
  fractions, affichées ×100 en n.u. partout ailleurs).
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

- Détection des repères P/Q/R/S/J/T par battement, ainsi que du **début du
  QRS** et de la **fin du QRS (point J)**, avec un template d'onde calibrable
  par enregistrement et un vérificateur interactif.
- Définitions standard des intervalles, mesurées sur chaque battement :
  - **PR** = début de P → début du QRS ;
  - **QRS** = début du QRS → fin du QRS (point J) ;
  - **QT** = début du QRS → fin de T ;
  - **QTc** = QT corrigé avec le RR **précédent** (R précédent → ce R).

  Le début du QRS est le premier écart à la ligne de base du segment PR
  (début de l'onde Q, ou montée de R en l'absence de Q) ; le point J est le
  retour de l'onde S à cette ligne de base (ou l'encoche avant l'onde J
  quand elle n'y revient pas). Les deux sont placés à l'intersection avec
  la ligne de base d'une droite ajustée (moindres carrés) sur la partie
  30–70 % de la déflexion (seuil : 5 % de l'amplitude de R ou 3× le bruit
  du segment PR pour l'onde R ; une onde Q ou S est retenue dès qu'elle
  descend de plus de 2 % de R ou 3× le bruit sous la ligne de base). Le
  début de P est l'intersection avec la ligne de base PR d'une droite du
  même type ajustée sur la montée de l'onde P, cherchée jusqu'à 30 ms avant
  le pic P (si une P posée sur le T précédent ne revient jamais à cette
  ligne de base, la droite est ajustée sur la partie de la montée au-dessus
  de la jonction T–P, sans placer le début avant cette jonction). Le pic P
  est cherché après le pic T du battement précédent et mesuré par rapport à
  la ligne de base PR (à 500 bpm, une ligne de base prise sur le T précédent
  donnait un PR ~60 ms au lieu de 42 et de faux événements de retard AV) ;
  un battement sans onde P au-dessus de 3× le bruit n'a pas de PR. La ligne de base PR est le dernier segment plat trouvé
  sur chaque battement avant Q/R (et non une fenêtre fixe) : un QRS élargi
  est mesuré en entier. L'onde J murine est une repolarisation précoce
  (Boukens et al. 2014, J Physiol 592:4613) : elle est comptée dans le QT,
  pas dans le QRS. La fin de T est l'intersection avec la ligne de base PR
  d'une droite du même type ajustée sur la partie 70–30 % de la pente
  descendante de T ; chez la souris elle est progressive et ambiguë,
  vérifiez son placement sur le battement moyen. Chaque battement est
  délimité dans sa propre fenêtre : jusqu'à ±100 ms, au plus 70 % de son
  RR précédent avant R et de son RR suivant après R (un voisin plus proche
  que le RR min est ignoré) ; les exports et le rapport Summary indiquent
  combien de battements ont eu une fenêtre raccourcie. Le QTc n'utilise le
  RR précédent que s'il est dans les bornes RR min / max de la VRC. Début
  de P, début du QRS, point J et fin de T ne sont **pas** bornés aux
  fenêtres du modèle : un repère
  impossible à placer dans le battement reste vide (NaN) au lieu d'être
  mis au bord d'une fenêtre, et un modèle mis à jour automatiquement par
  un enregistrement ne raccourcit plus le QRS ou le QT du suivant.
  **Valeurs modifiées (cette version) :** la fenêtre de battement valait
  ±45 % du RR le plus *court* de l'enregistrement, si bien qu'un seul
  battement prématuré raccourcissait la fenêtre de tous les battements (QT
  perdu partout, PR tronqué) et qu'à ~500 bpm la fin de T en sortait ; le
  début de P était plafonné ~9 ms avant le pic P par le modèle par défaut
  (PR trop court de 3–6 ms pour une montée de P de 12–15 ms, PR longs
  tronqués à des valeurs plausibles — ils sont désormais mesurés, ou
  rejetés au-delà de 80 ms) ; une onde Q de 2–5 % de R était ignorée (début
  du QRS ~3 ms trop tard : QRS et QT plus courts, PR plus long) ; le QTc
  était vide pour tout RR au-dessus des 333 ms par défaut même avec un RR
  max élargi. L'ajustement de droite supprime
  un raccourcissement du QT (plusieurs ms à 2–4 kHz) et du QRS qui
  dépendait du bruit ; les débuts de QRS avant −22 ms et les fins de T
  avant 60 ms après R (modèle par défaut) étaient auparavant rapportés à
  ces limites. **Valeurs modifiées (auparavant) :** les versions
  antérieures mesuraient le PR jusqu'au creux de Q, le QRS du creux de Q au
  creux de S et le QT depuis le pic R, et appariaient le QTc au RR suivant —
  QRS et QT sont désormais plus longs (plus proches de la littérature, p. ex.
  Mitchell et al. 1998) et le PR légèrement plus court. Les sessions
  enregistrées avant ce changement gardent leurs intervalles (signalés
  « legacy » dans l'onglet Intervals et les exports ; leur QTc est réapparié
  au RR précédent, ou laissé vide — et indiqué comme tel — quand la table
  n'a pas les temps des pics R) jusqu'à une nouvelle exécution d'Intervals. Le contrôle de
  retard AV des événements anormaux utilise ce PR.
- **Template d'onde par enregistrement.** Les fenêtres de recherche des
  repères utilisées pour un enregistrement sont son propre template : le
  template par défaut enregistré (ou les valeurs souris intégrées) tant que
  vous ne le modifiez pas dans **Landmarks** (onglet Beat Template) —
  auto-calibré sur le battement moyen, affiné à partir de la délinéation de
  cet enregistrement, modifié à la main ou chargé depuis un fichier.
  **Apply to this recording** n'écrit rien sur le disque ; le template est
  conservé dans la session de l'enregistrement. Seuls **Save as default
  template** (ou **Save to file…**) écrivent un template, et le défaut ne
  s'applique qu'aux enregistrements ouverts ensuite. Une autre voie du même
  fichier est un autre enregistrement (souvent un autre animal) : détecter
  les pics sur une autre voie abandonne le template fait sur la précédente,
  et la nouvelle voie repart du défaut enregistré. L'onglet Beat
  Template, la ligne d'état d'Intervals, le rapport Summary, la feuille
  Cover Excel et les notes Prism indiquent le template utilisé (ligne
  « Wave template » : provenance et fenêtres). **Comportement modifié
  (cette version) :** les versions antérieures resserraient le template sur
  chaque enregistrement après chaque délinéation et le réenregistraient
  comme défaut global : les intervalles d'un enregistrement dépendaient de
  ceux analysés avant lui. Un défaut écrit de cette façon (« auto-updated »)
  est désormais ignoré au profit des valeurs intégrées — enregistrez un
  template explicitement pour en faire le défaut. Les intervalles calculés
  par les versions antérieures affichent « Wave template: not recorded ».
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
- Variabilité battement-à-battement du QT (IQR, écart-type et étendue du QT
  sur les battements d'un enregistrement, une seule dérivation). Ce n'est
  **pas** la dispersion du QT classique (max − min du QT entre les
  dérivations d'un même battement), nom sous lequel les versions antérieures
  la présentaient ; l'étendue est dominée par les erreurs de délinéation,
  rapportez l'IQR.

### Événements anormaux (classification du rythme)

- Épisodes de bradycardie/tachycardie, pauses, battements prématurés et
  séries irrégulières, classés par rapport à la fréquence cardiaque de base
  propre à chaque enregistrement (pas un seuil fixe en bpm humain).
- **Battements prématurés (fondés sur les RR)** — un battement qui termine
  un intervalle RR inférieur à 70 % du RR de référence : la médiane des
  10 intervalles précédant le groupe de battements prématurés ou, si elle
  est plus petite, des 10 intervalles à partir de celui où le rythme
  revient. Le groupe doit se terminer — rythme revenu au-dessus de ce
  seuil — en 30 battements au plus : une suite plus longue d'intervalles
  courts est un changement de fréquence (laissé à la règle de
  tachycardie), et un groupe encore en cours à la fin de la fenêtre
  analysée n'est pas compté. Les versions antérieures utilisaient la
  médiane des 10 intervalles voisins, que la salve elle-même faisait
  baisser dès 6 battements : **les salves de plus de 5 battements
  environ n'étaient jamais rapportées** (elles apparaissaient comme séries
  irrégulières ou tachycardie) ; elles sont désormais comptées, ce qui
  augmente les effectifs sur les enregistrements qui en contiennent.
  Limite : une période de conduction normale entre deux périodes de bloc
  2:1 (RR divisé puis doublé de nouveau) est identique en timing et est
  rapportée comme une salve (vérifier les bradycardies autour). La détection n'utilise que le timing R-R : elle
  **ne distingue pas les battements prématurés ventriculaires des
  supraventriculaires** (il faudrait la morphologie du QRS et les ondes
  P) ; les versions antérieures les appelaient « ESV » (extrasystoles
  ventriculaires) et ne rapportaient que les salves ≥ 3. Les battements
  isolés, les doublets et les salves ≥ 3 sont désormais tous rapportés,
  avec effectifs et taux par heure (par heure de la durée analysée, du
  premier au dernier battement analysé) dans l'onglet Events, sa
  ligne d'état et sa frise, le texte « Copy for Excel », l'Excel (feuilles
  `Premature_Beats`, `Abnormal_Events` et lignes de la Cover), les
  rapports PDF et les notes Prism. Les battements que la correction
  d'artefacts a supprimés comme ectopiques sont réinsérés pour cette
  classification et **toujours comptés** (signalés « removed by artifact
  correction ») ; auparavant la correction automatique les supprimait avant
  la classification et ils disparaissaient des comptes. Il en va de même
  des battements très prématurés (couplage inférieur au Min R-R, ≈ 67 ms)
  que la correction supprime comme *duplicate* / *non-physiological* : une
  telle suppression est réinsérée quand l'intervalle qu'elle laisse est
  long (> 1,2 × la médiane locale des RR) ou suivi d'une pause
  compensatoire, et que son écart le plus court est ≥ 0,35 × cette
  médiane. Une suppression qui laisse un intervalle normal est une double
  détection parasite et n'est **pas** comptée ; le résumé indique combien
  ont été jugées ainsi. Les sessions enregistrées avant que les types de
  suppression soient conservés ont des suppressions de type inconnu :
  elles ne sont jugées que par ce même test d'intervalle, et le résumé
  indique leur nombre (relancer la correction d'artefacts pour le contrôle
  complet). La feuille Excel `Premature_Beats` donne aussi la fenêtre
  analysée (début / fin), et les exports **omettent** les résultats des
  événements anormaux (avec une note sur la Cover / le PDF et dans la
  barre d'état) quand les pics, la correction d'artefacts ou la fenêtre
  d'analyse ont changé après la classification — la relancer. L'intervalle qui
  enjambe un tel battement supprimé n'est plus signalé comme pause, et une
  série RR irrégulière qui chevauche un battement prématuré n'est plus
  rapportée en plus — deux changements des effectifs d'événements par
  rapport aux versions antérieures. Les anciens événements « esv_run » sont
  lus comme des salves prématurées.
- **Signalement des délais de conduction AV** à partir d'une prolongation
  soutenue de l'intervalle PR, une fois la délinéation des intervalles
  effectuée — comble le manque où la classification du rythme ne voyait
  auparavant que le seul timing des pics R.
- Une frise chronologique colorée par type d'événement (regroupement visuel
  immédiat) accompagnée de cartes par épisode ; cliquer sur une carte ou un
  segment de la frise zoome l'ECG brut sur cet épisode, avec un mode
  d'édition manuelle des pics pour corriger la détection sur place.

### Comparaison & export

- Comparaison de segments A/B avec tachogrammes RR superposés et un tableau
  complet des métriques (puissance et n.u. LF / HF, SD1/SD2 — mêmes
  définitions que les panneaux principaux). Chaque segment est aussi résumé
  par les **moyennes de blocs non chevauchants** (10 s par défaut, champ
  **Block (s)** ; un bloc est retenu si les RR valides en couvrent ≥ 50 %) :
  FC moyenne des blocs ± écart-type avec le nombre de blocs, et la
  différence B − A. Un p de Mann-Whitney sur les moyennes de blocs n'est
  affiché qu'à titre **descriptif** (≥ 3 blocs par segment) : deux portions
  d'un même enregistrement ne sont pas des échantillons indépendants —
  l'unité expérimentale est l'animal — l'inférence de groupe relève des
  statistiques Event-locked / outils de cohorte, entre animaux. L'export
  Excel ajoute une feuille **Blocks** et la note. **Comportement modifié
  (cette version) :** les versions précédentes appliquaient un test U de
  Mann-Whitney aux centaines d'intervalles RR battement par battement de
  chaque segment (valeurs autocorrélées traitées comme indépendantes —
  pseudoréplication), d'où un p « significatif » pour presque n'importe
  quels deux segments.
- Persistance de session — sauvegarde/restauration d'une session d'analyse
  complète en JSON, plus un registre SQLite local des enregistrements
  récemment analysés et des résultats autonomiques par événement enregistrés
  pour les statistiques de cohorte.
- Export — classeurs Excel formatés, GraphPad Prism `.pzfx`, un rapport PDF
  d'une page (bande de signal, tableau des métriques clés, diagramme de
  Poincaré et résumé des événements anormaux), un PDF annoté par épisode
  pour les événements anormaux, et CSV.
- **Provenance dans chaque rapport.** Le rapport texte Summary (bloc
  PROVENANCE), la feuille Cover d'Excel, les notes du projet Prism et le
  pied de page du PDF indiquent les mêmes lignes : la **fenêtre
  d'analyse** et la **durée analysée** (fin − début de la fenêtre, plus
  l'étendue du premier au dernier pic R analysé), les **filtres**
  enregistrés lors de la détection des pics, les bornes RR et les
  intervalles exclus de la VRC, la **correction d'artefacts** (battements
  retirés, par type), la **formule de QTc réellement utilisée** (avec le RR
  précédent), les définitions des intervalles et le template d'ondes.
  Chaque valeur porte son unité : VLF / LF / HF sont des puissances en
  **ms²**, les n.u. ne concernent que LFn / HFn. **Valeurs modifiées
  (cette version) :** la « durée » affichée par la ligne de verdict du
  Summary et la tuile de durée était l'instant absolu du dernier battement
  (elle comptait aussi le temps avant le début d'une fenêtre d'analyse) ;
  c'est désormais la durée analysée. La ligne « Duration » de la feuille
  Cover devient **Recording duration** (longueur du signal), et la colonne
  `Duration_s` de la feuille HR_Statistics est remplacée par
  `Recording_duration_s`, `Window_start_s`, `Window_end_s` et
  `Analysed_duration_s`. Le rapport Summary et la page PDF, en partie en
  français auparavant, sont désormais entièrement en anglais.
  **Autres corrections de provenance (cette version) :** une fenêtre
  d'analyse dont la fin dépasse la fin du signal est ramenée à la fin du
  signal partout où elle est rapportée (la durée analysée et
  `Window_end_s` comptaient du temps inexistant ; la ligne indique la fin
  saisie). Une ligne **Signal crop** indique le recadrage au chargement
  (Start / End crop) en temps fichier : tous les temps des rapports sont
  relatifs à son début. La ligne **Filters** donne les fréquences de coupure
  du passe-bande réellement appliquées à la fréquence d'échantillonnage de
  l'enregistrement : le filtre borne chaque coupure à [0,01 ; 0,99] ×
  Nyquist, si bien qu'à 2 kHz une coupure basse de 1 Hz vaut en réalité
  10 Hz, et la ligne le dit désormais (le filtrage lui-même est inchangé).
  Les effectifs de la correction d'artefacts sont présentés comme portant
  sur tout l'enregistrement (elle s'applique avant le fenêtrage), et la
  mention « exclus de la VRC » n'apparaît que pour des résultats qui
  excluent réellement ces intervalles (les anciens résultats les
  pontaient). Une session enregistrée conserve désormais la fenêtre sur
  laquelle les résultats ont été calculés et les filtres / le recadrage
  avec lesquels le signal a été préparé, et la restauration reprépare le
  signal à partir de ceux-ci plutôt que des valeurs des widgets modifiées
  après la détection (anciennes sessions : inchangé).
- **Asymétrie RR (valeurs modifiées).** Les **indices de Porta** et de
  **Guzik** du Summary suivent désormais les conventions publiées, les
  mêmes que HRV_PI / HRV_GI de NeuroKit (feuille Excel HRV_NonLinear) :
  PI de Porta (Porta et al. 2008) = points sous la ligne d'identité du
  diagramme de Poincaré (accélérations, ΔRR < 0) / points hors de la
  ligne, les ΔRR = 0 ne comptent donc plus au dénominateur ; GI de Guzik
  (Guzik et al. 2006) = Σ des distances à la ligne des points au-dessus
  (décélérations, ΔRR > 0) / Σ des distances de tous les points hors de la
  ligne. Le Summary affichait auparavant la part des accélérations dans
  Σ ΔRR² (complément d'un GI en distances au carré) et un PI comptant les
  zéros au dénominateur. Les deux valent 50 % pour une série symétrique et
  NaN quand aucun ΔRR n'est non nul ; le rapport, la feuille Cover de
  l'Excel (« HR asymmetry ») et la table Non-linear de Prism (avec
  désormais le GI) indiquent la convention. Dans cette table Prism, le
  titre de ligne **« Porta Index » devient « Porta Index PI (%) »** (mêmes
  valeurs HRV_PI de NeuroKit) et une ligne « Guzik Index GI (%) » est
  ajoutée : une mise en page Prism qui cherche la ligne par son ancien
  titre doit être mise à jour. Le diagramme en barres du Summary montre
  décélérations et accélérations en part des paires à ΔRR ≠ 0 (la barre
  des accélérations est le PI, la ligne à 50 % la symétrie) et donne les
  paires à ΔRR = 0 comme un compte à part ; il divisait auparavant par
  toutes les paires, d'où des barres qui ne correspondaient pas au PI.
- La courbe **Quality over time** est datée par les battements
  réellement comparés au gabarit : les pics trop proches des bords du
  signal ne sont pas évalués, et la courbe associait les scores aux pics
  depuis le premier, d'où un décalage d'un battement quand le premier pic
  était exclu. La tuile « Poor-shape-match time » et la colonne
  « quality » du CSV RR attribuent toutes deux chaque intervalle RR au
  battement évalué qui le **commence** (Time_s − RR_ms de la ligne du CSV,
  Time_s étant la fin de l'intervalle), comme le CSV le faisait quand
  aucun battement de bord n'était écarté. Pour les résultats d'anciennes
  versions, qui n'enregistraient pas les battements évalués, ceux-ci sont
  reconstruits à la restauration de session à partir des pics restaurés
  (même règle de bord) ; si c'est ambigu, la courbe, la tuile (« — ») et
  la colonne du CSV sont toutes omises, et la barre d'état le signale.

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
│       ├── app_*.py, tab_*.py # mixins d'ECGApp : panneaux et un module par onglet
│       ├── context_strip.py, tabbar.py, freshness.py # bandeau, états des onglets
│       ├── report_view.py     # onglet Report et rapport texte (pur, testé)
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
