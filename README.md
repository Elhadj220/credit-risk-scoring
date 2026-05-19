# 🏦 Credit Risk Scoring

> Système de scoring de risque crédit basé sur le Machine Learning — déployé en production

[![Live Demo](https://img.shields.io/badge/Demo-Live-green)](https://credit-risk-frontend-i78l.onrender.com)
[![API Docs](https://img.shields.io/badge/API-Docs-blue)](https://credit-risk-scoring-d8h6.onrender.com/api/docs/)
[![GitHub](https://img.shields.io/badge/GitHub-Elhadj220-black)](https://github.com/Elhadj220)

---

## 🎯 Problème métier

Prédire si un client bancaire va faire **défaut sur son paiement** le mois suivant,
à partir de son historique de crédit — pour aider les banques à mieux gérer le risque.

---

## 🚀 Demo en production

| Service               | URL                                                     |
| --------------------- | ------------------------------------------------------- |
| **Application Web**   | https://credit-risk-frontend-i78l.onrender.com          |
| **API Documentation** | https://credit-risk-scoring-d8h6.onrender.com/api/docs/ |

---

## 📊 Résultats ML

| Métrique         | Baseline | Random Forest |
| ---------------- | -------- | ------------- |
| ROC-AUC          | 0.495    | **0.775**     |
| Recall défaut    | 0.21     | **0.55**      |
| Precision défaut | 0.21     | **0.51**      |

**Optimisation du seuil de décision** — F2-score → Recall passe de 0.55 à **0.73**
→ 378 mauvais payeurs supplémentaires détectés

---

## 🛠️ Stack technique

**Machine Learning**

- Python, pandas, scikit-learn
- Random Forest avec class_weight="balanced"
- Feature Engineering : agrégation BILL_AMT, PAY_RATIO

**Backend**

- Django 6 + Django REST Framework
- JWT Authentication (SimpleJWT)
- Rate limiting, Swagger/OpenAPI docs

**Frontend**

- React 19 + Vite
- Tailwind CSS
- Axios + React Router

**DevOps**

- Docker + docker-compose
- Déploiement cloud : Render
- CI/CD automatique via GitHub

---

## 📁 Structure du projet

credit-risk-scoring/
├── src/
│ ├── data/ # Chargement et validation
│ ├── features/ # Feature engineering
│ └── models/ # Entraînement et évaluation
├── api/ # Configuration Django
├── predictor/ # App Django REST
├── frontend/ # React app
├── notebooks/ # EDA et analyse
├── tests/ # Tests unitaires
└── configs/ # Configuration YAML

---

## ⚙️ Installation locale

```bash
# Backend
git clone https://github.com/Elhadj220/credit-risk-scoring
cd credit-risk-scoring
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver

# Frontend
cd frontend
npm install
npm run dev
```

---

## 📈 Dataset

- **Source** : UCI Machine Learning Repository
- **Taille** : 30 000 clients, 24 features
- **Target** : défaut de paiement (binaire)
- **Déséquilibre** : 78% / 22% → géré avec class_weight

---

## 👤 Auteur

**Aladji Yero Gano**

- LinkedIn : [linkedin.com/in/Leuz](https://linkedin.com/in/)
- GitHub : [github.com/Elhadj220](https://github.com/Elhadj220)
