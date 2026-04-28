"""
preprocessing.py -- Limpieza y transformacion del dataset CU Venta.
Produce: df_train.csv, df_test.csv, df_val.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

NAN_THRESHOLD = 80  # columnas con > 80 % NaN se eliminan
VALIDATION_CODMES = 201912.0  # mes reservado para validacion
TEST_SIZE = 0.30
RANDOM_STATE = 123


def run_preprocessing(data_path, nan_threshold=NAN_THRESHOLD):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    Returns:
    df_train, df_test, df_val, metadata
    """
    df = pd.read_csv(data_path)
    # Eliminar columnas con exceso de NaN
    cols_drop = [c for c in df.columns if df[c].isna().mean() * 100 > nan_threshold]

    df = df.drop(columns=cols_drop)
    # Imputaciones y encodings ...
    # (implementar segun logica de clase)
    # Split temporal
    df_val = df[df["p_codmes"] == VALIDATION_CODMES].copy()
    df_main = df[df["p_codmes"] != VALIDATION_CODMES].copy()
    df_train, df_test = train_test_split(
        df_main, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return df_train, df_test, df_val, {"dropped": cols_drop}
