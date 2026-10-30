# Reprise du travail (point de sauvegarde du 30/10/2026)

Pour reprendre depuis un autre poste : `git switch feature/ml-r-peak-detector`, installer les dépendances,
copier quelques `.mat` hors du dépôt (ils ne vont pas sur GitHub).

## État de ce commit
- Audit P2/P3 : corrigé et vérifié sur signaux de synthèse.
- Refonte des onglets selon la maquette : EN COURS. Étapes 1 (découpage de `app.py`) et 2 (table commune des
  chiffres, `ecg/core/metric_table.py`, `ecg/ui/freshness.py`) faites ; la relecture de l'étape 2 n'était pas
  forcément close à ce commit. Les onglets visibles ne sont pas encore réorganisés.
- Maquette : https://claude.ai/artifact/AqCuLQZVQAnpzGvWJmEyNJ (privée, même compte).

## Plan de la refonte
1. Découpage de `app.py` et constantes d'onglets
2. Table commune des chiffres et état par module
3. Coque de l'application et Signal & Detection
4. Rhythm & HRV
5. Time course (Epochs, Sliding, Autonomic, Event-locked)
6. Morphology & Intervals, puis Events
7. Report, suppression des anciennes vues, documentation

Règle de la refonte : aucune valeur calculée ne change, seul son emplacement change.
La délinéation (`ecg/core/wave_template.py`, `analyse_intervals`) n'y est pas modifiée.

## À faire ensuite : mesure de PR et QT sur vrais signaux
Sur vrais enregistrements, PR et QT battement par battement ne sont pas fiables : l'onde P (0,2-0,25) est au
niveau du bruit (SD 0,22 pour R = 7,3). Ne pas utiliser ces chiffres pour l'instant.

Décision de l'utilisateur : moyenne glissante. Essai hors application (médiane de N voisins), % mesurés et médiane :

| Enregistrement | N | PR | QT |
|---|---|---|---|
| 248 Baseline | 1 | 2 %, 44,8 ms | 75 %, 50,3 ms (IQR 38-76) |
| 248 Baseline | 32 | 100 %, 42,9 ms | 75 %, 39,2 ms (IQR 38-50) |
| 247 Footshocks | 1 | 70 %, 47,7 ms | 84 %, 39,7 ms (IQR 38-50) |
| 247 Footshocks | 32 | 69 %, 46,2 ms | 98 %, 39,2 ms (IQR 38,5-41) |

Proposition (réglages à confirmer) : médiane glissante, N = 32 par défaut, réglable et indiqué dans les rapports.
Restent à faire :
- comprendre le PR bloqué à 70 % sur le 247 (P minuscule, peut-être crantée : hypothèse non vérifiée) ;
- corriger le début du QRS, détecté trop tard sur le 248 (-4 ms contre environ -8 ms sur le battement médian) ;
- exclure les artefacts des footshocks de la moyenne ; le lissage peut aussi atténuer une réponse brève à un choc.

Enregistrements : `D:\Expériences\Expérience N°13 - Stimulation Optogénétique du CeA en anesthésié\` (248 et 247).

## Questions ouvertes
- Relever `PR_ABS_MAX` (80 ms) ? Au-delà, le PR est vide et aucune alerte n'est levée.
- Corriger la coupure basse du filtre : « 1 Hz » devient 10 Hz à 2 kHz (filtre désactivé par défaut).
- Le détecteur de pics R a manqué 5 battements prématurés sur 8 dans un signal de test (couplage de 72 ms).
- Le PDF a annoncé 5 événements pour 4 listés : cause non trouvée.
- `~/.ecg_wave_template.json` a été modifié le 30/10 à 10 h 08 : origine non identifiée, à contrôler.
- Méthode CSI/CPI : ks/kp sont des valeurs de rat, à calibrer pour la souris.

## Règles du projet
- `ecg/core` n'importe pas `ecg.ui` ; les tests n'écrivent pas dans le profil utilisateur.
- Textes de l'interface en anglais ; échanges avec l'utilisateur en français.
