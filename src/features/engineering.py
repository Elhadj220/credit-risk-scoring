import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les valeurs anormales des variables catégorielles.
    
    EDUCATION : valeurs 0, 5, 6 non documentées → regroupées dans 4 (Others)
    MARRIAGE  : valeur 0 non documentée → regroupée dans 3 (Others)
    
    Décision basée sur l'EDA :
    - EDUCATION 0 : 14 clients, 0.0% défaut → erreur saisie
    - EDUCATION 5,6 : comportement similaire à Others (5.7% vs 6.4%, 15.7%)
    - MARRIAGE 0 : 54 clients, non documenté
    """
    df = df.copy()  # on ne modifie jamais le DataFrame original

    # EDUCATION
    before = df['EDUCATION'].value_counts().to_dict()
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    logger.info(f"EDUCATION nettoyé — valeurs uniques : {sorted(df['EDUCATION'].unique())}")

    # MARRIAGE
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    logger.info(f"MARRIAGE nettoyé — valeurs uniques : {sorted(df['MARRIAGE'].unique())}")

    return df


def rename_target(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme la target pour simplifier l'usage dans le code."""
    df = df.copy()
    df = df.rename(columns={'default payment next month': 'target'})
    logger.info("Target renommée : 'default payment next month' → 'target'")
    return df


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime la colonne ID — inutile pour le modèle."""
    df = df.copy()
    df = df.drop(columns=['ID'])
    logger.info("Colonne ID supprimée")
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de nettoyage complet.
    Appelle les fonctions dans l'ordre logique.
    """
    logger.info("Démarrage du pipeline de nettoyage...")
    df = drop_id(df)
    df = clean_categorical_features(df)
    df = rename_target(df)
    logger.info(f"Pipeline terminé — shape final : {df.shape}")
    return df