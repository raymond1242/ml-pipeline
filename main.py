"""
main.py -- Orquestador del pipeline ML E2E.

Pasos:
1. Preprocesamiento  : lee train_date/, escribe preprocess_data/.
2. Entrenamiento     : train/test -> modelo campeon en models/<timestamp>/.
3. Scoring + monitoreo + postprocesamiento sobre el set de test.
"""

import pandas as pd

from monitoring import compute_recall_by_decile, run_monitoring
from postprocessing import run_postprocessing, save_replica
from preprocessing import MODEL_NAME, OUTPUT_DIR as PREPROCESS_DIR, run_preprocessing
from training import TARGET_COL, auto_train

TRAIN_VARS = f"{PREPROCESS_DIR}/preprocessed/train_vars_{MODEL_NAME}.csv"
TEST_VARS = f"{PREPROCESS_DIR}/preprocessed/test_vars_{MODEL_NAME}.csv"
TEST_POST = f"{PREPROCESS_DIR}/postprocessed/test_post_{MODEL_NAME}.csv"
TLV_OUTPUT = f"{PREPROCESS_DIR}/output_tlv.csv"


def main():
    # 1. Preprocesamiento
    run_preprocessing()

    # 2. Entrenamiento
    result = auto_train(train_path=TRAIN_VARS, test_path=TEST_VARS)
    if result is None:
        print("Sin modelo campeon - abortando pipeline.")
        return
    _, champion = result
    model = champion["model"]

    # 3. Scoring sobre train y test (test = proxy de validacion)
    df_train = pd.read_csv(TRAIN_VARS)
    df_test = pd.read_csv(TEST_VARS)
    train_scores = model.predict_proba(df_train.drop(columns=[TARGET_COL]))[:, 1]
    test_scores = model.predict_proba(df_test.drop(columns=[TARGET_COL]))[:, 1]

    # 4. Monitoreo
    run_monitoring(train_scores, test_scores, y_val=df_test[TARGET_COL])
    compute_recall_by_decile(df_test[TARGET_COL], test_scores)

    # 5. Postprocesamiento + replica
    df_post = pd.read_csv(TEST_POST)
    df_resultado = run_postprocessing(test_scores, df_post, TLV_OUTPUT)
    save_replica(df_resultado, table="EC_OMNICANAL", partition="202412")


if __name__ == "__main__":
    main()
