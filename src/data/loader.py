import logging
import yaml
import pandas as pd
from pathlib import Path

# Configuration du logger — standard en entreprise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Charge la configuration depuis le fichier YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration chargée depuis {config_path}")
    return config


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Charge les données brutes depuis le chemin défini dans la config.
    
    Returns:
        pd.DataFrame: données brutes non modifiées
    """
    raw_path = Path(config["data"]["raw_path"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {raw_path}")

    logger.info(f"Chargement des données depuis {raw_path}")
    df = pd.read_excel(raw_path, header=1)

    logger.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def validate_data(df: pd.DataFrame, config: dict) -> bool:
    """
    Validation basique du dataset — vérifie que les colonnes attendues existent.
    
    Returns:
        bool: True si valide, lève une exception sinon
    """
    target = config["data"]["target_column"]

    if target not in df.columns:
        raise ValueError(f"Colonne target '{target}' absente du dataset")

    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 50]

    if not high_missing.empty:
        logger.warning(f"Colonnes avec >50% de valeurs manquantes : {high_missing.to_dict()}")

    logger.info(f"Validation OK — target '{target}' présente, "
                f"{df.isnull().sum().sum()} valeurs manquantes au total")
    return True