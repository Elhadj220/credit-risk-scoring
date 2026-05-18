# Credit Risk Scoring 🏦

Modèle de machine learning pour prédire le risque de défaut de paiement
d'un client bancaire (classification binaire).

## Problème métier

Prédire si un client va faire défaut sur son paiement le mois suivant,
à partir de son historique de crédit et de paiements.

## Dataset

- Source : UCI Machine Learning Repository
- 30 000 clients, 23 features
- Target : `default payment next month` (0 = pas de défaut, 1 = défaut)

## Stack technique

- Python 3.13
- pandas, scikit-learn, matplotlib, seaborn
- pytest pour les tests

## Structure du projet

credit-risk-scoring/
├── configs/ # Configuration centralisée (YAML)
├── data/ # Données brutes et traitées (non versionnées)
├── notebooks/ # Exploration et EDA
├── src/
│ ├── data/ # Chargement et validation des données
│ ├── features/ # Feature engineering
│ └── models/ # Entraînement et évaluation
└── tests/ # Tests unitaires

## Ce que tu peux dire à un recruteur

"J'ai un projet Credit Risk Scoring déployé en production — une API Django REST avec authentification JWT, un modèle Random Forest ROC-AUC 0.77, et un frontend React. Tout est accessible en ligne."

Deux URLs à mettre sur ton CV et LinkedIn :

API : https://credit-risk-scoring-d8h6.onrender.com/api/docs/
App : https://credit-risk-frontend-i78l.onrender.com
