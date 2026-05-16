"""
preprocessing.py -- Limpieza, casteo y split temporal del dataset CU Venta.

Recibe la configuracion inyectada (no usa globals). Descarta columnas con
exceso de NaN, imputa numericas y separa df_val (p_codmes ==
split_cfg.validation_codmes) del resto; sobre el resto aplica train/test
split aleatorio.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PreprocessingConfig, SplitConfig


def _drop_high_nan(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    cols_drop = [c for c in df.columns if df[c].isna().mean() * 100 > threshold]
    return df.drop(columns=cols_drop), cols_drop


def _numeric_feature_cols(df: pd.DataFrame, target_col: str, post_cols: list[str]) -> list[str]:
    excluded = {target_col, *post_cols}
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def process_vars(
    df: pd.DataFrame, target_col: str, prep_cfg: PreprocessingConfig,
) -> pd.DataFrame:
    """Normaliza nulls y castea numericas a float32."""
    df = df.replace(prep_cfg.na_tokens, np.nan)
    numeric_cols = _numeric_feature_cols(df, target_col, prep_cfg.post_cols)
    df[numeric_cols] = (
        df[numeric_cols].fillna(prep_cfg.na_fill_numeric).astype("float32")
    )
    return _coerce_dtypes(df, prep_cfg.float_decimals)


def _coerce_dtypes(df: pd.DataFrame, float_decimals: int) -> pd.DataFrame:
    """Tipos listos para ML: bool/int -> int; floats -> redondeados."""
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "bool" or pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].astype(int)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(float).round(float_decimals)
    return df


def run_preprocessing(
    data_path: str,
    target_col: str,
    prep_cfg: PreprocessingConfig,
    split_cfg: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Pipeline completo de preprocesamiento.
    Returns: df_train, df_test, df_val, metadata
    """
    df = pd.read_csv(data_path)
    df, cols_drop = _drop_high_nan(df, prep_cfg.nan_threshold)
    df = process_vars(df, target_col, prep_cfg)

    df_val = df[df["p_codmes"] == split_cfg.validation_codmes].copy()
    df_main = df[df["p_codmes"] != split_cfg.validation_codmes].copy()
    df_train, df_test = train_test_split(
        df_main,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
    )
    return df_train, df_test, df_val, {"dropped": cols_drop}


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    run_preprocessing(
        data_path=cfg.input_path,
        target_col=cfg.target_col,
        prep_cfg=cfg.preprocessing,
        split_cfg=cfg.split,
    )
