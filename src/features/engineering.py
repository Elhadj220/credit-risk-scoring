import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime la colonne ID — inutile pour le modèle."""
    df = df.copy()
    df = df.drop(columns=['ID'])
    logger.info("Colonne ID supprimée")
    return df


def clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les valeurs anormales des variables catégorielles.

    EDUCATION : valeurs 0, 5, 6 non documentées → regroupées dans 4 (Others)
    MARRIAGE  : valeur 0 non documentée → regroupée dans 3 (Others)
    """
    df = df.copy()

    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    logger.info(f"EDUCATION nettoyé — valeurs uniques : {sorted(df['EDUCATION'].unique())}")

    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    logger.info(f"MARRIAGE nettoyé — valeurs uniques : {sorted(df['MARRIAGE'].unique())}")

    return df


def rename_target(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme la target pour simplifier l'usage dans le code."""
    df = df.copy()
    df = df.rename(columns={'default payment next month': 'target'})
    logger.info("Target renommée : 'default payment next month' → 'target'")
    return df


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features agrégées pour réduire la multicolinéarité.

    BILL_AMT1-6 corrélées à 0.80-0.95 → condensées en 3 features
    PAY_AMT1-6 corrélées → condensées en 2 features
    """
    df = df.copy()

    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_cols  = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    df['BILL_AMT_MEAN']  = df[bill_cols].mean(axis=1)
    df['BILL_AMT_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    df['BILL_AMT_MAX']   = df[bill_cols].max(axis=1)
    df['PAY_AMT_MEAN']   = df[pay_cols].mean(axis=1)

    df['PAY_RATIO'] = np.where(
        df['BILL_AMT_MEAN'] > 0,
        df['PAY_AMT_MEAN'] / df['BILL_AMT_MEAN'],
        0
    )

    df = df.drop(columns=bill_cols + pay_cols)

    logger.info(f"Features agrégées créées — shape : {df.shape}")
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de nettoyage et feature engineering complet."""
    logger.info("Démarrage du pipeline de nettoyage...")
    df = drop_id(df)
    df = clean_categorical_features(df)
    df = rename_target(df)
    df = create_aggregated_features(df)
    logger.info(f"Pipeline terminé — shape final : {df.shape}")
    return df