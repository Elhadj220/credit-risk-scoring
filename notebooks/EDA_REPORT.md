# EDA Report — Credit Risk Scoring

## Dataset

- Source : UCI Machine Learning Repository
- Shape : 30 000 lignes, 24 colonnes (après nettoyage)
- Target : `target` (0 = pas de défaut, 1 = défaut)

## 1. Distribution de la target

- Pas de défaut (0) : 23 364 clients (77.9%)
- Défaut (1) : 6 636 clients (22.1%)
- Ratio : 1 défaut pour 3.5 non-défauts
- **Conclusion** : déséquilibre modéré → métrique ROC-AUC, class_weight="balanced"

## 2. Anomalies détectées et corrigées

### EDUCATION

| Valeur | Description   | Clients | Action                     |
| ------ | ------------- | ------- | -------------------------- |
| 0      | Non documenté | 14      | → regroupé dans 4 (Others) |
| 5      | Non documenté | 280     | → regroupé dans 4 (Others) |
| 6      | Non documenté | 51      | → regroupé dans 4 (Others) |

### MARRIAGE

| Valeur | Description   | Clients | Action                     |
| ------ | ------------- | ------- | -------------------------- |
| 0      | Non documenté | 54      | → regroupé dans 3 (Others) |

## 3. Features les plus importantes

### Corrélation positive avec le défaut (risque ↑)

| Feature | Corrélation | Interprétation                                |
| ------- | ----------- | --------------------------------------------- |
| PAY_0   | +0.325      | Retard paiement récent = signal d'alerte fort |
| PAY_2   | +0.264      | Historique retards confirmé                   |
| PAY_3   | +0.235      | Pattern de retards persistant                 |

### Corrélation négative avec le défaut (risque ↓)

| Feature   | Corrélation | Interprétation                                     |
| --------- | ----------- | -------------------------------------------------- |
| LIMIT_BAL | -0.154      | Limite élevée = client fiable validé par la banque |
| PAY_AMT1  | -0.073      | Montant payé élevé = capacité de remboursement     |

## 4. Décisions pour le modèle

- **Métrique principale** : ROC-AUC
- **Gestion déséquilibre** : class_weight="balanced"
- **Features clés** : PAY_0 à PAY_6 sont les prédicteurs les plus forts
- **Multicolinéarité** : à vérifier (BILL_AMT corrélées entre elles)

## 5. Multicolinéarité détectée

### BILL_AMT (corrélations 0.80 - 0.95)

- 6 features quasi-identiques → condensées en features agrégées
- BILL_AMT_MEAN, BILL_AMT_TREND à créer en feature engineering

### PAY_2 à PAY_6 (corrélations 0.71 - 0.82)

- Historique de retards redondant
- PAY_AMT_MEAN à créer en feature engineering

**Action** : feature engineering semaine 3 → réduction multicolinéarité
