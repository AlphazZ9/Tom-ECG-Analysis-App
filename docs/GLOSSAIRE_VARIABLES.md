# Glossaire des variables calculées

Ce document définit chaque valeur que le logiciel calcule et affiche, telle que le
code la calcule (version du 6 novembre 2026, branche `feature/ml-r-peak-detector`).
Pour chaque variable : ce qu'elle représente, comment elle est obtenue, son unité,
sur quelles données elle repose, et comment la lire chez la souris.

Les noms entre `crochets` sont ceux des colonnes des exports (Excel, Prism) ou de la
feuille *Metrics* (colonne *Key*).

---

## 0. Conventions communes à tout le logiciel

### Pic R, intervalle RR, intervalle NN
- **Pic R** : instant du sommet de l'onde R d'un battement, détecté sur le signal.
- **RR** (ms) : temps entre deux pics R successifs, `(pic[i+1] − pic[i]) / fs × 1000`.
  Chaque intervalle est daté à l'instant du **battement qui le termine**.
- **NN** (« normal-to-normal ») : un RR **valide**, c'est-à-dire retenu pour les calculs.

### RR valide / RR exclu
Un RR est **valide** s'il remplit les trois conditions :
1. il est compris entre les **bornes RR** `[RR min, RR max]`, bornes incluses. Par défaut
   ce sont 66,7 et 333,3 ms, soit 900 et 180 bpm. Ces bornes se règlent dans
   Signal & Detection (« Min RR », « Max RR ») ;
2. il ne **chevauche pas un battement retiré** par la correction d'artefacts ;
3. il est fini.

Un RR exclu n'est **jamais ponté** : il n'est ni interpolé ni fusionné avec ses voisins.
Il est compté sous une seule raison, `n_too_short` + `n_too_long` + `n_spanning_removed`
= `n_excluded`.

### Paires successives valides
Paire (RRₙ, RRₙ₊₁) de deux intervalles **adjacents et tous deux valides**. Une paire ne
traverse jamais un intervalle exclu. Elle sert au Poincaré (SD1, SD2), au RMSSD, au pNN6
et aux indices d'asymétrie.

### Fenêtre d'analyse
L'intervalle de temps analysé (« Window » dans la barre de contexte ; « Full » = tout
l'enregistrement). Toutes les valeurs de Rhythm & HRV portent sur cette fenêtre.

### États d'un résultat
- **à jour** : calculé sur les pics et réglages actuels ;
- **périmé** (◆ orange) : les pics ou les réglages ont changé depuis ; les valeurs
  affichées ne correspondent plus ;
- **non calculé**.

### QT et QTc
QT et QTc sont **masqués** dans toutes les vues et tous les exports tant que leur mesure
n'est pas validée (réglage interne `QT_REPORTED = False`). Leurs définitions sont
données plus bas pour mémoire.

---

## 1. Fréquence cardiaque (Rhythm & HRV, cartes KPI, Report)

| Variable | `Key` | Définition | Unité |
|---|---|---|---|
| **HR mean** | `hr.mean` | **60 000 / moyenne des NN.** Ce n'est *pas* la moyenne des fréquences instantanées : celle-ci est biaisée vers le haut par la variabilité (inégalité de Jensen) et donnerait environ 0,1 à 1 bpm de plus. Cette définition unique sert partout (KPI, tachogramme, Epochs, Sliding, Compare segments, exports). | bpm |
| **HR min** | `hr.min` | **2ᵉ percentile** de la fréquence instantanée 60 000 / RRᵢ, soit 60 000 / 98ᵉ percentile des NN. Ce n'est pas le minimum absolu : il résiste à un battement manqué isolé. | bpm |
| **HR max** | `hr.max` | **98ᵉ percentile** de la fréquence instantanée, soit 60 000 / 2ᵉ percentile des NN. | bpm |
| **HR SD** | `hr.std` | Écart-type (ddof = 1) de la fréquence instantanée 60 000 / RRᵢ sur les NN. | bpm |
| **Beats analysed** | `hr.n_beats` | Nombre de pics R détectés dans la fenêtre. | battements |
| **Valid RR intervals** | `hr.n_valid` | Nombre de RR valides. | intervalles |
| **RR mean** | `rr.mean` | Moyenne des NN. | ms |
| **Analysed duration** | `core.duration_s` | Durée analysée (premier au dernier pic de la fenêtre). | s |

### Exclusions RR (lignes « RR excluded… »)
| Variable | Définition |
|---|---|
| `rr.n_intervals` | Nombre total d'intervalles RR de la fenêtre. |
| `rr.n_excluded` | RR exclus des calculs de HRV. |
| `rr.n_too_short` / `rr.n_too_long` | … dont exclus parce que plus courts que RR min / plus longs que RR max. |
| `rr.n_spanning_removed` | … dont exclus parce qu'ils chevauchent un battement retiré par la correction d'artefacts. |
| `rr.excluded_pct` | Part des RR exclus (%). |
| `rr.excluded_time_s` | Durée cumulée des RR exclus (s). |
| `rr.rr_min_ms`, `rr.rr_max_ms` | Bornes RR utilisées. |

---

## 2. HRV — domaine temporel

Calculée par NeuroKit2 (`nk.hrv_time`) sur les NN datés à leur vrai temps. Quand des RR
sont exclus, toutes les métriques fondées sur des **différences successives** sont
recalculées sur les seules paires successives valides.

| Variable | `Key` | Définition | Unité | Lecture chez la souris |
|---|---|---|---|---|
| **MeanNN** | `hrv_time.HRV_MeanNN` | Moyenne des NN. | ms | ≈ 120 ms à 500 bpm |
| **SDNN** | `hrv_time.HRV_SDNN` | Écart-type des NN (ddof = 1). Variabilité **globale** : rapide + lente + tendances de la fenêtre. Dépend fortement de la durée analysée : ne comparer que des fenêtres de même durée. | ms | |
| **RMSSD** | `hrv_time.HRV_RMSSD` | Racine de la moyenne des carrés des différences successives `RRₙ₊₁ − RRₙ`, sur les paires successives valides. Variabilité **battement à battement**, surtout d'origine **parasympathique (vagale)**. | ms | |
| **pNN6** | `hrv_time.HRV_pNN6` | % des paires successives valides dont `|RRₙ₊₁ − RRₙ| > 6 ms`. Seuil adapté à la souris (Thireau et al. 2008) : 6 ms ≈ 5 % du RR moyen. Indice **vagal**. | % | |
| pNN20, pNN50 *(human)* | `HRV_pNN20`, `HRV_pNN50` | Même chose avec des seuils de 20 et 50 ms, calibrés pour l'humain. Chez la souris pNN50 ≈ 0 % : affichés seulement sous « Show all ». Dénominateur NeuroKit = nombre de différences + 1. | % | à ne pas utiliser |
| SDSD | `HRV_SDSD` | Écart-type des différences successives (paires valides). | ms | proche de RMSSD |
| CVSD | `HRV_CVSD` | RMSSD / MeanNN. | — | |
| SDRMSSD | `HRV_SDRMSSD` | SDNN / RMSSD. | — | |
| MedianNN, MadNN, IQRNN | | Médiane, déviation absolue médiane, écart interquartile des NN. | ms | |
| Prc20NN, Prc80NN, MinNN, MaxNN | | 20ᵉ et 80ᵉ percentiles, minimum, maximum des NN. | ms | |
| CVNN, MCVNN | | SDNN / MeanNN ; MadNN / MedianNN. | — | |
| SDANNx, SDNNIx | | Écart-type des moyennes / moyenne des écarts-types sur des segments de x minutes (NeuroKit). Peu pertinents sur des fenêtres courtes. | ms | |
| HTI, TINN | | Indice triangulaire et largeur du triangle de l'histogramme des NN (NeuroKit). | —, ms | |

---

## 3. HRV — domaine fréquentiel

**Méthode** :
1. les NN sont placés à leur vrai temps et rééchantillonnés à **20 Hz** par spline
   cubique ;
2. la moyenne est retirée ;
3. on calcule le **spectre de Welch** : fenêtre de Hann, recouvrement de 75 %, résolution
   ≈ 0,02 Hz ;
4. chaque puissance de bande est l'**intégrale (trapèzes)** de la densité spectrale sur
   la bande, intervalle [bas, haut).

La courbe de la PSD affichée est exactement ce spectre.

**Bandes souris (Thireau 2008), par défaut** : VLF 0–0,4 Hz · LF 0,4–1,5 Hz ·
HF 1,5–5 Hz. Un préréglage « LF/HF custom » existe (LF 0,4–2, HF 2–5 Hz).

**Calcul refusé** (valeurs absentes, raison affichée) dans ces cas :
- moins de 60 NN ;
- une suite continue de RR exclus de plus de **2 s** ;
- des RR exclus couvrant plus de **5 %** du temps analysé.

L'interpolation inventerait alors une partie du signal.

| Variable | `Key` | Définition | Unité |
|---|---|---|---|
| **VLF power** | `hrv_freq.HRV_VLF` | Puissance absolue 0–0,4 Hz. Oscillations lentes (thermorégulation, hormonal, tendances). Peu fiable sur des fenêtres courtes. | ms² |
| **LF power** | `hrv_freq.HRV_LF` | Puissance absolue 0,4–1,5 Hz. Mélange **sympathique + parasympathique** (baroréflexe) ; *pas* un indice sympathique pur. | ms² |
| **HF power** | `hrv_freq.HRV_HF` | Puissance absolue 1,5–5 Hz, bande de la respiration de la souris (≈ 2–4 Hz). Indice **parasympathique** (arythmie sinusale respiratoire). | ms² |
| **LF (n.u.)** | `hrv_freq.HRV_LFn` | LF / (LF + HF), fraction affichée en %. Unités normalisées de la Task Force (1996). | n.u. |
| **HF (n.u.)** | `hrv_freq.HRV_HFn` | HF / (LF + HF). LF n.u. + HF n.u. = 100. | n.u. |
| **LF/HF** | `hrv_freq.HRV_LFHF` | LF / HF. Souvent appelé « balance sympatho-vagale » ; interprétation discutée, à donner avec LF et HF absolus. | — |
| TP | `HRV_TP` | Puissance totale = VLF + LF + HF. | ms² |
| LnHF | `HRV_LnHF` | ln(HF). | ln ms² |
| LFpeak, HFpeak | `HRV_LFpeak`, `HRV_HFpeak` | Fréquence du maximum de la PSD dans la bande LF / HF. HFpeak ≈ fréquence respiratoire. | Hz |

---

## 4. HRV — non linéaire

### Poincaré (sur **toutes** les paires successives valides de la fenêtre)
Le diagramme de Poincaré trace chaque paire (RRₙ, RRₙ₊₁). L'ellipse a pour centre la
moyenne des points ; son demi-petit axe vaut SD1 (perpendiculaire à la diagonale), son
demi-grand axe SD2 (le long de la diagonale).

À 1 kHz, les RR tombent sur une grille de 1 ms. Les points affichés sont donc décalés
aléatoirement de ± ½ pas, **pour l'affichage seulement**, et c'est indiqué sur la figure.

| Variable | `Key` | Définition | Unité | Lecture |
|---|---|---|---|---|
| **SD1** | `hrv_nonlin.HRV_SD1` | Écart-type de `(RRₙ − RRₙ₊₁)/√2` (ddof = 1). Dispersion **perpendiculaire** à la diagonale = variabilité **battement à battement**. SD1 = RMSSD/√2 (à la moyenne des différences près). | ms | vagal, court terme |
| **SD2** | `hrv_nonlin.HRV_SD2` | Écart-type de `(RRₙ + RRₙ₊₁)/√2`. Dispersion **le long** de la diagonale = variabilité **lente / à plus long terme**. | ms | lent, global |
| **SD1/SD2** | `hrv_nonlin.HRV_SD1SD2` | Rapport SD1 / SD2. | — | |
| S | `HRV_S` | Aire de l'ellipse = π·SD1·SD2. | ms² | |
| CSI, CVI, CSI modifié (NeuroKit) | `HRV_CSI`, `HRV_CVI`, `HRV_CSI_Modified` | Avec L = 4·SD2 et T = 4·SD1 : CSI = L/T, CVI = log₁₀(L·T), CSI modifié = L²/T (Toichi et al. 1997). **À ne pas confondre** avec CSI/CPI de l'onglet Autonomic (§ 8). | — | |

**Attention** : SD1 est très sensible à quelques paires aberrantes. Un battement manqué
ou en trop crée des points loin de la diagonale. C'est ce que vérifie le contrôle qualité
de Compare segments (§ 9).

### Asymétrie de la fréquence cardiaque (sur les paires successives valides ; ≥ 8 paires)
| Variable | `Key` | Définition | Lecture |
|---|---|---|---|
| **Porta index (PI)** | `asymmetry.porta_pct` | Points **sous** la diagonale (ΔRR < 0, accélérations) / points hors diagonale (ΔRR ≠ 0) × 100. | symétrie = 50 % |
| **Guzik index (GI)** | `asymmetry.guzik_pct` | Σ distances à la diagonale des points **au-dessus** (ΔRR > 0, décélérations) / Σ distances de tous les points hors diagonale × 100. | symétrie = 50 % |
| Decelerations / Accelerations / No change | `n_dec`, `n_acc`, `n_zero` | Nombre de paires avec ΔRR > 0, ΔRR < 0, ΔRR = 0. | |

NeuroKit fournit d'autres indices d'asymétrie et de fragmentation (`HRV_PI`, `HRV_GI`,
`HRV_SI`, `HRV_AI`, `HRV_C1d`, `HRV_PIP`, `HRV_IALS`, `HRV_PSS`, `HRV_PAS`…) sous
« Show all ». Ils sont mis à NaN si NeuroKit a apparié des intervalles de part et
d'autre d'un trou.

### Fractale
| Variable | `Key` | Définition | Lecture |
|---|---|---|---|
| **DFA α1** | `hrv_nonlin.HRV_DFA_alpha1` | Exposant de l'analyse des fluctuations redressées (*detrended fluctuation analysis*) sur les **petites échelles** (NeuroKit : 4–16 battements). Corrélations à court terme de la série RR. | ≈ 1 : fluctuations « 1/f » ; < 0,75 : plus aléatoire ; > 1,25 : plus régulier/corrélé |
| **DFA α2** | `hrv_nonlin.HRV_DFA_alpha2` | Même chose sur les **grandes échelles** (16–64 battements). | |
| MFDFA… | `HRV_MFDFA_*` | Variantes multifractales (NeuroKit), sous « Show all ». | |

### Entropie et complexité (sur un **bloc** de NN)
Ces calculs coûtent O(N²). Ils portent sur le **bloc contigu de 2 000 NN** contenant le
moins d'intervalles exclus (le réglage 0 = sans limite). Le nombre de NN et la période
utilisés sont indiqués à l'écran et dans les exports.

| Variable | `Key` | Définition | Lecture |
|---|---|---|---|
| **SampEn** | `hrv_nonlin.HRV_SampEn` | Entropie d'échantillon de la série RR (NeuroKit, m = 2, tolérance r = 0,2 × écart-type). −ln de la probabilité que deux séquences semblables de m points le restent à m + 1 points. | plus élevée = moins prévisible, plus complexe ; plage usuelle 0,5–2,5 |
| ApEn | `HRV_ApEn` | Entropie approximative (même principe, biaisée par l'auto-comparaison). | |
| **Multiscale entropy (courbe)** | (figure) | SampEn recalculée sur la série RR moyennée par blocs de 1, 2, … jusqu'à 15 battements (Costa et al. 2002). Il faut au moins 60 NN, le nombre d'échelles étant limité à N/30. | une perte de complexité peut n'apparaître qu'aux échelles intermédiaires |
| **Σ MSE** | (titre de la figure) | Somme des SampEn de toutes les échelles = aire sous la courbe. | |
| ShanEn, FuzzyEn, MSEn, CMSEn, RCMSEn, CD, HFD, KFD, LZC, Symbolic… | | Autres indices NeuroKit (entropies de Shannon et floue, multi-échelle composite et raffinée, dimension de corrélation, dimensions fractales de Higuchi et de Katz, complexité de Lempel-Ziv, dynamique symbolique), sous « Show all ». | |

---

## 5. Qualité du signal (Signal & Detection, Report)

| Variable | `Key` | Définition | Unité |
|---|---|---|---|
| **Beat correlation (par battement)** | (`beat_corr`) | Corrélation de Pearson entre chaque battement (fenêtre de ± 100 ms autour du pic R, réduite à 45 % du plus court RR valide à fréquence élevée) et le **gabarit** de battement. Le gabarit est construit en deux passes : 1) moyenne de tous les battements ; 2) moyenne des seuls battements corrélés ≥ 0,75 à la première. | — (−1 à 1) |
| **Template correlation** | `quality.mean_corr` | Moyenne des corrélations par battement. | — |
| **Beats below 0.90** | `quality.n_bad`, `quality.bad_pct` | Nombre et % de battements dont la corrélation au gabarit est < 0,90 (forme anormale : artefact, extrasystole, bruit). | battements, % |
| **Time below 0.90** (« Poor-shape-match time ») | `quality.noisy_time_pct` | % du **temps** (somme des RR) dont le battement de **début** d'intervalle a une corrélation < 0,90. | % |
| **Signal quality score** | `quality.score` | `100 × corrélation moyenne × ratio`, borné à 0–100. Le ratio vaut nombre de pics détectés / nombre attendu à 500 bpm, borné à 0,5–1,5. Avant l'analyse complète, la corrélation est remplacée par `1 − CV des RR`. **Indicatif** : le ratio pénalise une fréquence cardiaque basse (anesthésie) même si le signal est propre. | /100 |
| Auto-corrected (artefacts) | `artifact.removed`, `artifact.n_duplicate`, `artifact.n_nonphysio`, `artifact.n_ectopic` | Battements retirés par la correction automatique d'artefacts, et le motif : doublon, non physiologique, ectopique. | battements |
| Template correlation over time / histogram | (figures) | Corrélation de chaque battement au cours du temps (avec moyenne glissante sur 50 battements), et sa distribution. | |

---

## 6. Tachogramme et marqueurs (Rhythm & HRV, Signal & Detection)

| Élément | Définition |
|---|---|
| **Tachogramme RR / HR instantanée** | RRᵢ et 60 000 / RRᵢ de chaque intervalle valide, à l'instant de son battement de fin. Un RR exclu interrompt la ligne. L'axe Y est **centré sur la moyenne** (ligne pointillée). |
| **Spike markers** ▲/▼ | Battement dont le RR s'écarte de la médiane glissante (15 battements) de plus de **2,5 écarts-types** des NN, ou dont la différence avec le RR précédent (adjacent et valide) dépasse 2,5 × 1,2 écarts-types. ▲ = accélération brusque, ▼ = décélération. **Ambre** > 2,5 SD, **rouge** > 3,5 SD. Ce sont des repères de relecture, pas un diagnostic. |
| **Excluded (hollow red markers)** | RR exclus (hors bornes ou chevauchant un battement retiré). |

---

## 7. Morphologie et intervalles (Morphology & Intervals)

Chaque battement est **délimité** (repérage des points P début, QRS début, QRS fin,
T fin) dans des fenêtres de recherche guidées par le gabarit. Deux options :
- délimiter chaque battement sur lui-même (N = 1) ;
- délimiter chaque ligne sur la **médiane glissante de N battements voisins** alignés sur
  R (N = 32 par défaut).

Les battements prématurés, celui qui les suit et les battements artefactés ne sont jamais
mélangés aux voisins.

| Variable | `Key` | Définition | Unité | Bornes acceptées (au-delà = échec → vide) | Normale souris adulte |
|---|---|---|---|---|---|
| **PR** | `intervals.PR_ms` | **Début de l'onde P → début du QRS.** Conduction auriculo-ventriculaire. | ms | 10–80 | 30–55 |
| **QRS** | `intervals.QRS_ms` | **Début du QRS → fin du QRS** (point J, fin de S). Dépolarisation ventriculaire. | ms | 3–35 | 8–25 |
| *QT (masqué)* | `intervals.QT_ms` | Début du QRS → fin de l'onde T. | ms | 15–110 | 30–90 |
| *QTc (masqué)* | `intervals.QTc_ms` | QT corrigé par le RR **précédent** ; formule par défaut **Mitchell** : QT / √(RR/100) (Bazett normalisé à un RR souris de 100 ms). Bazett et Fridericia sont disponibles pour comparaison. En médiane glissante, le RR utilisé est la moyenne des RR précédents des battements de la médiane. | ms | — | 30–100 (Mitchell) |
| Valeur affichée | | **Médiane** sur les battements où l'intervalle a été mesuré ; chaque intervalle a son propre nombre de battements. | | | |
| SD, Q1, Q3 | `….sd`, `….q1`, `….q3` | Écart-type et 1ᵉʳ/3ᵉ quartiles **entre battements**, ou entre lignes en médiane glissante, où la dispersion est réduite par le lissage. | ms | | |
| **Beats measured** | `intervals.beats_measured` | Battements où PR **et** QRS sont tous deux mesurés / battements délimités. | battements | | |
| *QT IQR (masqué)* | `intervals.QT_iqr` | Écart interquartile du QT battement à battement (une seule dérivation : ce n'est pas la dispersion du QT classique). | ms | | |

Les intervalles sont en ms, mesurés à la fréquence d'échantillonnage du fichier (2 kHz :
résolution 0,5 ms).

---

## 8. Évolution dans le temps (Time course)

### Epochs et Sliding
- **Epochs** découpe la fenêtre en **périodes consécutives non chevauchantes** de durée
  fixe (60 s par défaut, minimum 5 s).
- **Sliding** utilise des **fenêtres glissantes** : longueur ≥ 3 s, pas ≥ 1 s, pas ≤
  longueur.

Dans chaque période ou fenêtre, les métriques sont recalculées **avec les mêmes définitions
que ci-dessus**, en excluant les RR invalides sans les ponter : HR (60 000 / moyenne NN),
SDNN, RMSSD, pNN6, SD1, SD2, LF n.u., HF n.u., LF/HF.

| Règle | Définition |
|---|---|
| Fenêtre ignorée (trou dans la ligne) | Moins de 5 battements ou moins de 3 RR valides. |
| Point **creux** (peu fiable) | Moins de 10 RR valides dans la fenêtre (et, pour une métrique LW cochée, moins de 10 paires valides). |
| **CCD** | Distance de l'origine au centre de l'ellipse de Poincaré, √(m₁² + m₂²) (m = moyennes des deux coordonnées des paires). Ce centre est déterminé surtout par la fréquence cardiaque. | 
| **SD1 (LW), SD2 (LW)** | SD1 et SD2 calculés comme les racines des valeurs propres d'une covariance de **Ledoit-Wolf** (« shrinkage ») des paires, au lieu de la covariance d'échantillon. Variante « style article », plus stable sur peu de paires (≥ 10 paires). |

### Autonomic : CSI et CPI (Candia-Rivera et al., J Physiol 2025)
Indices **résolus dans le temps** des influences sympathique et parasympathique, calculés à
partir de Poincaré glissants.

1. Pour chaque battement *i*, on prend une fenêtre des intervalles des **T = 3 s**
   précédentes (« Window (s) »). Sur les paires valides de cette fenêtre, il en faut au
   moins **10** (« Min pairs »), sinon la fenêtre n'est pas calculée.
2. Dans chaque fenêtre on calcule SD1ᵢ et SD2ᵢ, les racines des valeurs propres de la
   covariance des paires. L'estimateur est au choix (« Method ») :
   - *robust* (Schäfer-Strimmer, texte de l'article, par défaut) ;
   - *rat reference* (covariance simple avec moyennes tronquées à 5 %, code publié) ;
   - *Ledoit-Wolf*.
3. On calcule aussi **Dᵢ (CCD)**, la distance du centre de l'ellipse à l'origine. Il est
   déterminé surtout par la durée du cycle.
4. SD01, SD02 et D0 sont les mêmes grandeurs calculées sur **toutes** les paires de
   l'enregistrement. Ils recentrent les séries : SD1 ← SD1 − moyenne(SD1) + SD01, et de
   même pour SD2 et D.
5. Les séries sont rééchantillonnées à 20 Hz. D̃ est D réfléchi autour de sa moyenne
   (D̃ = 2·moyenne(D) − D).

| Variable | Définition | Lecture |
|---|---|---|
| **CSI** (indice sympathique cardiaque) | **D̃ + ks·SD2**. Il augmente quand le cycle raccourcit (D̃ ↑) et quand les oscillations lentes augmentent (SD2 ↑). | sympathique ↑ |
| **CPI** (indice parasympathique cardiaque) | **D + kp·SD1**. Il augmente quand le cycle s'allonge et quand les oscillations rapides augmentent (SD1 ↑). | parasympathique ↑ |
| CSI_HR / CPI_HR | Composantes « fréquence » : D̃ et D. | |
| CSI_HRV / CPI_HRV | Composantes « variabilité » : ks·SD2 et kp·SD1. | |
| **SD01, SD02, D0** | `autonomic.SD01_ms`, `SD02_ms`, `D0_ms` : SD1, SD2 et CCD de tout l'enregistrement (avec l'estimateur choisi). | ms |
| **ks = 3, kp = 12** | Gains de l'article, calibrés chez le **rat** (SHR). **Non validés chez la souris** : à calibrer sur votre cohorte (« Calibrate ks/kp »). | — |
| LF / HF (SPWVD) (option) | Puissances LF et HF **instantanées** obtenues par la distribution de Wigner-Ville pseudo-lissée (portage du code de référence, noyau d'Orini 2012) sur les RR valides rééchantillonnés. | ms² |
| Fenêtres retenues / non calculées | Nombre de fenêtres calculées et de fenêtres sans assez de paires. Avec 10 paires en 3 s, les fenêtres ne sont pas calculées en dessous d'environ 200 bpm (souris anesthésiée). | |

Toutes les composantes sont des durées en ms. CSI et CPI sont donc en ms, et seule leur
**variation** a un sens physiologique.

### Event-locked (en bas de la page Autonomic)
| Variable | Définition |
|---|---|
| **Pre / Post** | **Moyenne** de chaque série (CSI, CPI, HR, D, SD1, SD2, LF/HF si calculés) sur les **30 s avant** et les **30 s après** le **début** de chaque annotation choisie (« Events »). Les durées se règlent avec « Pre (s) » et « Post (s) ». |
| **Δ (delta)** | Post − Pre, par événement. |
| **Événement valide** | Aucune autre annotation (quelle que soit son étiquette) ne commence dans ses fenêtres ; les deux fenêtres sont dans la série CSI/CPI et couvertes à ≥ 80 %. |
| z-score (affichage) | Chaque série normalisée par sa moyenne et son écart-type de tout l'enregistrement. **Affichage seulement** : le tableau, les tests et le registre utilisent les valeurs brutes. |
| Courbes médiane ± MAD | Médiane et déviation absolue médiane des séries alignées sur le début des événements. |
| Test intra-enregistrement | Wilcoxon apparié pre vs post sur les événements **d'un même animal**, à titre **descriptif**. L'unité statistique est l'animal : les tests de groupe se font sur le registre de cohorte. |
| Calibration ks/kp | Z de Wilcoxon (contraste pre/post de la cohorte) en fonction de k = 1…20 ; on choisit le k au **coude** de la courbe Z(k), là où elle atteint un plateau. Possible sans recalcul parce que ΔCSI = −ΔD + ks·ΔSD2 et ΔCPI = ΔD + kp·ΔSD1. |

---

## 9. Compare segments

Le premier segment sert de **référence**.

| Variable | Définition | Unité |
|---|---|---|
| Beats | Pics R du segment. | battements |
| Duration | Fin − début. | s |
| HR, RR, SDNN, RMSSD, pNN6, VLF/LF/HF, LF/HF n.u., LF/HF, SD1, SD2, SD1/SD2 | **Mêmes définitions** que §§ 1–4, recalculées sur le segment (spectre refusé selon les mêmes règles). | |
| Mean HR (60000/mean NN) | HR moyenne du segment. | bpm |
| Instantaneous HR SD, Min/Max inst. HR | Écart-type, 2ᵉ et 98ᵉ percentiles de la HR instantanée. | bpm |
| RR CV | Écart-type / moyenne des NN × 100. | % |
| **Blocks used (n)** | Le segment est découpé en blocs **non chevauchants** de « Block (s) » secondes (10 s par défaut). Un bloc est utilisé si les RR valides couvrent au moins la moitié de sa durée. | blocs |
| **HR, mean of block means** / **SD of block means** | Moyenne et écart-type des HR des blocs. C'est le résumé de la comparaison, sans pseudo-réplication battement par battement. | bpm |
| Δ % | (segment − référence) / |référence| × 100. | % |
| p (Mann-Whitney sur les blocs) | **Descriptif seulement** (≥ 3 blocs par segment) : les blocs d'un même enregistrement ne sont pas indépendants. | — |

### Contrôles qualité d'un segment (colonne QC, feuille *Quality*)
| Contrôle | Signalé quand |
|---|---|
| Valid RR (% of window) | Les RR valides couvrent **< 80 %** de la fenêtre. |
| Poincaré pairs | **< 200** paires successives valides. |
| Durée | Diffère de plus de **2 %** de celle du premier segment. |
| Artefact | Le segment chevauche l'événement ou la **marge d'artefact** qui suit (30 s par défaut). |
| Abrupt RR jumps | Plus de **0,5 %** des paires ont `|RRₙ₊₁ − RRₙ| > 20 %` du RR médian (battements manqués / en trop). |
| **SD1 trimmed 2 %** | SD1 est **> 1,5×** sa valeur recalculée sans les 2 % de paires les plus extrêmes. SD1 est alors dominé par quelques paires aberrantes : vérifier les pics. Sur des données propres le rapport vaut ≈ 1,05. |

### Modèles de fenêtres
Chaque fenêtre d'un modèle est placée depuis le **début** de l'événement (fenêtres avant)
ou depuis sa **fin** (fenêtres après), avec un décalage en secondes. *Before/after 5 min* :
Before = début −300 à 0 s ; After = fin +30 à +330 s.

---

## 10. Événements anormaux (Events)

La **référence** (*baseline*) est la médiane des RR des **30 premières secondes**
analysées, convertie en bpm. Si cette fenêtre couvre l'essentiel de l'enregistrement, les
bradycardies et tachycardies détectées sont marquées comme peu fiables.

| Type | Règle |
|---|---|
| **Bradycardia** | La HR glissante (médiane sur 5 RR) reste **< référence − 20 %** pendant ≥ 10 battements consécutifs. |
| **Tachycardia** | La HR glissante reste **> référence + 20 %** pendant ≥ 10 battements. |
| **Pause** | Un RR **> 2 × la médiane locale** (10 RR voisins, sans lui-même) **et** > 200 ms (1,5 × 133 ms). Un intervalle qui chevauche un battement retiré n'est pas une pause. |
| **Premature beat** (isolé) | Un battement dont le RR qui le **termine** est **< 70 %** du RR de référence. La référence est la plus petite de deux médianes : celle des 10 RR avant le groupe, et celle des 10 RR à partir du retour au rythme. Critère **de rythme seulement**, pas de morphologie. |
| **Premature couplet** / **run** | 2 / ≥ 3 battements prématurés consécutifs (jusqu'à 30 ; au-delà il s'agit d'un changement de rythme, traité par la règle de tachycardie). |
| **Sustained PR prolongation** (« AV delay ») | PR **> 55 ms** (limite haute de la normale souris) pendant ≥ 10 battements consécutifs. Nécessite le calcul des intervalles. |
| **Irregular run** | 7 RR consécutifs dont le **coefficient de variation > 15 %**, hors des zones déjà classées (bradycardie, tachycardie, pause, prématurés). |

Deux événements du même type à moins de 0,5 s d'intervalle sont **fusionnés** (sauf les
prématurés).

| Variable de synthèse | `Key` | Définition |
|---|---|---|
| Premature beats | `events.n_premature` | Isolés + 2 × couplets + battements dans les runs. |
| Isolated / Couplets / Runs ≥ 3 / Beats in runs | `events.n_isolated`, `n_couplets`, `n_runs`, `n_beats_in_runs` | Comptes. |
| … per hour | `events.*_per_h` | Compte / durée analysée en heures. Sur un enregistrement court, c'est une **extrapolation**. |

Pour mémoire, les battements retirés par la correction d'artefacts sont **remis** pour
classer les prématurés, afin qu'une correction automatique ne les fasse pas disparaître.

---

## 11. Tableau de groupe (Compare segments → « Add to group table »)

Une ligne par enregistrement : `subject`, `file`, `template`, puis pour chaque segment
(colonnes suffixées par l'étiquette du segment) :
- `start (s)`, `end (s)` ;
- `valid RR (%)`, `Poincaré pairs (n)`, `abrupt RR jumps (n)` ;
- `QC` (« ok » ou la liste des alertes) ;
- toutes les métriques du § 9.

Utilisez le **même modèle** pour tous les animaux afin que les colonnes s'alignent. Pour
deux segments, faites un test apparié (Wilcoxon ou t apparié) ; pour plus de deux, un
test à mesures répétées (Friedman ou ANOVA à mesures répétées). Dans tous les cas, l'unité
statistique est l'animal.
