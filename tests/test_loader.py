import pytest
import pandas as pd
from src.data.loader import load_config, load_raw_data, validate_data


def test_load_config():
    """La config doit charger et contenir les clés attendues."""
    config = load_config()
    assert "data" in config
    assert "model" in config
    assert "target_column" in config["data"]


def test_load_raw_data():
    """Le dataset doit avoir 30000 lignes et 25 colonnes."""
    config = load_config()
    df = load_raw_data(config)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (30000, 25)


def test_validate_data():
    """La validation doit retourner True sur un dataset valide."""
    config = load_config()
    df = load_raw_data(config)
    assert validate_data(df, config) is True


def test_target_column_exists():
    """La colonne target doit être présente et binaire."""
    config = load_config()
    df = load_raw_data(config)
    target = config["data"]["target_column"]
    assert target in df.columns
    assert df[target].nunique() == 2