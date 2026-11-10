# Reprise du travail (point de sauvegarde du 05/11/2026, style de la maquette)

Pour reprendre depuis un autre poste : `git switch feature/ml-r-peak-detector`, installer les dépendances,
copier quelques `.mat` hors du dépôt (ils ne vont pas sur GitHub).

## État de ces commits
- Médiane glissante des battements pour PR/QT : N = 32 par défaut, réglable (Morphology & Intervals).
- Délinéation corrigée, à N > 1 seulement (N = 1 garde les anciennes règles, valeurs identiques à avant) :
  - début de P : remontée de la ligne de base après le creux d'une P biphasique ;
  - début du QRS : petite onde initiale avant le Q, décidée sur des médianes de 256 battements ;
  - fin de T : fin de la dernière phase d'une T biphasique, décidée sur des médianes locales de 8N battements.
- Disposition sans barres latérales : barre de détection en haut de Signal & Detection, fiche Recording
  (clic sur la pastille du fichier), fenêtre d'analyse dans le bandeau, menu ⋯, Report > More.
  Fonctionne dès 1280×720. Chiffres identiques à la disposition précédente.
- Style aligné sur la maquette (couleurs, boutons blancs bordés / bleu pour l'action principale,
  cartes blanches sur fond gris, typographie Segoe UI) : valeurs dans `ecg/ui/theme.py`
  (`button_style`, `card_style`, `input_style`…). Chiffres identiques à avant.
- Tests : 1 362 passent. Captures de la doc refaites (sauf les deux de Parameters).
- Maquette : https://claude.ai/artifact/AqCuLQZVQAnpzGvWJmEyNJ (privée, même compte).

Résultats à N = 32 (médianes) :

| Enregistrement | PR | QRS | QT | QTc |
|---|---|---|---|---|
| 248 Baseline | 38,8 ms | 15,3 ms | 54,0 ms | 41,3 ms |
| 247 Footshocks | 45,8 ms | 15,3 ms | 53,0 ms | 39,5 ms |

Attention : ne pas comparer des résultats obtenus avec des N différents (QT 39,7 ms à N = 1 contre
53,0 ms à N = 32 sur le 247). Le début de P retenu est 3 à 5 ms après le tout premier départ de l'onde.

Enregistrements : `D:\Expériences\Expérience N°13 - Stimulation Optogénétique du CeA en anesthésié\` (248 et 247).

## Petits défauts connus
- À 1366×720, les mini-graphes RR / HR / Quality sous le tracé ont des étiquettes qui se chevauchent.
- Deux textes d'aide disent encore « Project name » (le champ s'appelle « Project »).
- Quelques commentaires du code parlent encore de « sidebar ».
- Le paragraphe « Delineation » des exports (page de garde Excel, pied du PDF) fait ~3 000 caractères.
- Après un changement de thème, la légende du seuil peut afficher une valeur périmée (déjà le cas avant).

## Écarts à la maquette, à arbitrer
- Events : pas de ligne de base « 30 s avant une annotation » (nouveau calcul ; proposé en option).
- Epochs et Sliding gardent deux boucles de calcul : les fusionner changerait le RMSSD dans un cas testé.
- La restauration de session ne ramène pas Events, Autonomic ni Event-locked.
- Notes s'ouvre dans une fenêtre et non dans un champ de la fiche Recording.

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
