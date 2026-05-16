"""
main.py -- Orquestador del pipeline ML E2E.

Pasos:
1. Preprocesamiento  : lee data/raw/Data_CU_venta.csv, devuelve df_train/df_test/df_val.
2. Entrenamiento     : train/test -> modelo campeon en models/<timestamp>/.
3. Scoring + monitoreo + postprocesamiento sobre la validacion temporal (df_val).
"""

import logging

import pandas as pd

from monitoring import run_monitoring
from postprocessing import run_postprocessing, save_replica
from preprocessing import INPUT_FILE, POST_COLS, TARGET_COL, VALIDATION_CODMES, run_preprocessing
from training import auto_train

INPUT_PATH = INPUT_FILE
TLV_OUTPUT = "preprocess_data/output_tlv.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    excluded = {TARGET_COL, *POST_COLS}
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def main():
    # 1. Preprocesamiento
    df_train, df_test, df_val, meta = run_preprocessing(INPUT_PATH)
    logger.info("Columnas descartadas por NaN: %d", len(meta["dropped"]))

    features = _feature_cols(df_train)
    train_df = df_train[[TARGET_COL, *features]]
    test_df = df_test[[TARGET_COL, *features]]

    # 2. Entrenamiento
    result = auto_train(train_df=train_df, test_df=test_df)
    if result is None:
        logger.warning("Sin modelo campeon - abortando pipeline.")
        return
    _, champion = result
    model = champion["model"]

    # 3. Scoring sobre train y validacion temporal
    train_scores = model.predict_proba(train_df.drop(columns=[TARGET_COL]))[:, 1]
    val_scores = model.predict_proba(df_val[features])[:, 1]

    # 4. Monitoreo (incluye recall por decil)
    run_monitoring(train_scores, val_scores, y_val=df_val[TARGET_COL])

    # 5. Postprocesamiento + replica
    df_post = df_val[POST_COLS].copy()
    df_resultado = run_postprocessing(val_scores, df_post, TLV_OUTPUT)
    save_replica(df_resultado, table="EC_OMNICANAL", partition=str(int(VALIDATION_CODMES)))


if __name__ == "__main__":
    main()
