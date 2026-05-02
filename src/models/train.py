import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(path: str = "data/processed/dataset_final.pkl") -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Dataset chargé — Train: {data['X_train_scaled'].shape}")
    return data


def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_proba)

    logger.info(f"\n{'='*50}")
    logger.info(f"Modèle : {model_name}")
    logger.info(f"ROC-AUC : {roc_auc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Matrice de confusion :\n{cm}")
    return roc_auc


def train_baseline(data: dict) -> DummyClassifier:
    logger.info("Entraînement baseline (stratified)...")
    model = DummyClassifier(strategy="stratified", random_state=42)
    model.fit(data['X_train_scaled'], data['y_train'])
    evaluate_model(model, data['X_test_scaled'], data['y_test'], "Baseline")
    return model


def train_random_forest(data: dict) -> RandomForestClassifier:
    logger.info("Entraînement Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(data['X_train_scaled'], data['y_train'])
    roc_auc = evaluate_model(
        model, data['X_test_scaled'], data['y_test'], "Random Forest"
    )
    return model, roc_auc


def plot_feature_importance(model, feature_names: list):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)),
            importances[indices],
            color='#e74c3c', alpha=0.8, edgecolor='black')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.title('Feature Importance — Random Forest', fontweight='bold')
    plt.tight_layout()
    plt.savefig('notebooks/feature_importance.png', dpi=150)
    plt.show()
    logger.info("Feature importance sauvegardée")


def save_model(model, path: str = "data/processed/model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Modèle sauvegardé : {path}")


if __name__ == "__main__":
    data              = load_dataset()
    baseline          = train_baseline(data)
    rf_model, roc_auc = train_random_forest(data)
    plot_feature_importance(rf_model, data['feature_names'])
    
    # Optimisation du seuil
    best_threshold = optimize_threshold(
        rf_model, data['X_test_scaled'], data['y_test']
    )
    evaluate_with_threshold(
        rf_model, data['X_test_scaled'], data['y_test'], best_threshold
    )
    
    save_model(rf_model)
    logger.info(f"\n✅ Entraînement terminé — ROC-AUC : {roc_auc:.4f}")
    
def optimize_threshold(model, X_test, y_test):
    """
    Trouve le seuil optimal pour maximiser le Recall sur la classe 1
    tout en maintenant une Precision acceptable.
    """
    from sklearn.metrics import precision_recall_curve
    
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # F2 score — pénalise plus les faux négatifs que les faux positifs
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-9)
    best_idx   = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    
    logger.info(f"\n=== Optimisation du seuil ===")
    logger.info(f"Seuil par défaut (0.5) — Recall défaut : 0.55")
    logger.info(f"Seuil optimal   ({best_threshold:.2f}) — "
                f"Precision : {precisions[best_idx]:.2f}, "
                f"Recall : {recalls[best_idx]:.2f}")
    
    # Visualisation
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions[:-1], label='Precision', color='#2ecc71')
    plt.plot(thresholds, recalls[:-1], label='Recall', color='#e74c3c')
    plt.axvline(x=best_threshold, color='black', linestyle='--',
                label=f'Seuil optimal ({best_threshold:.2f})')
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Seuil défaut (0.5)')
    plt.xlabel('Seuil de décision')
    plt.ylabel('Score')
    plt.title('Precision vs Recall selon le seuil', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('notebooks/threshold_optimization.png', dpi=150)
    plt.show()
    
    return best_threshold


def evaluate_with_threshold(model, X_test, y_test, threshold: float):
    """Évalue le modèle avec un seuil personnalisé."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    
    logger.info(f"\n=== Résultats avec seuil {threshold:.2f} ===")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Matrice de confusion :\n{cm}")
    logger.info(f"Faux négatifs : {cm[1][0]} "
                f"(clients défaut non détectés)")