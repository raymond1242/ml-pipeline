"""
preprocessing.py -- Limpieza, casteo y split temporal del dataset CU Venta.

Lee data/raw/Data_CU_venta.csv, descarta columnas con exceso de NaN,
imputa numericas y separa df_val (p_codmes == VALIDATION_CODMES) del resto;
sobre el resto aplica train/test split aleatorio.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/raw/Data_CU_venta.csv"
TARGET_COL = "target"
NAN_THRESHOLD = 80
VALIDATION_CODMES = 201912.0
TEST_SIZE = 0.30
RANDOM_STATE = 123
NA_FILL_NUMERIC = -9999999
NA_TOKENS = ["", "null", "None"]
FLOAT_DECIMALS = 4

POST_COLS = [
    "p_codmes",
    "key_value",
    "grp_campecs06m",
    "prob_value",
    "monto",
]


def _drop_high_nan(
    df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, list[str]]:
    cols_drop = [c for c in df.columns if df[c].isna().mean() * 100 > threshold]
    return df.drop(columns=cols_drop), cols_drop


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COL, *POST_COLS}
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def process_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nulls y castea numericas a float32."""
    df = df.replace(NA_TOKENS, np.nan)
    numeric_cols = _numeric_feature_cols(df)
    df[numeric_cols] = df[numeric_cols].fillna(NA_FILL_NUMERIC).astype("float32")
    return _coerce_dtypes(df)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos listos para ML: bool/int -> int; floats -> redondeados."""
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "bool" or pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].astype(int)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(float).round(FLOAT_DECIMALS)
    return df


def run_preprocessing(
    data_path: str = INPUT_FILE,
    nan_threshold: float = NAN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Pipeline completo de preprocesamiento.
    Returns: df_train, df_test, df_val, metadata
    """
    df = pd.read_csv(data_path)
    df, cols_drop = _drop_high_nan(df, nan_threshold)
    df = process_vars(df)

    df_val = df[df["p_codmes"] == VALIDATION_CODMES].copy()
    df_main = df[df["p_codmes"] != VALIDATION_CODMES].copy()
    df_train, df_test = train_test_split(
        df_main,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return df_train, df_test, df_val, {"dropped": cols_drop}


if __name__ == "__main__":
    run_preprocessing()
