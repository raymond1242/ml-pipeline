"""
main.py -- Orquestador del pipeline ML E2E.

Pasos:
1. Preprocesamiento  : lee train_date/, escribe preprocess_data/.
2. Entrenamiento     : train/test -> modelo campeon en models/<timestamp>/.
3. Scoring + monitoreo + postprocesamiento sobre el set de test.
"""

import pandas as pd

from monitoring import run_monitoring
from postprocessing import run_postprocessing, save_replica
from preprocessing import MODEL_NAME, OUTPUT_DIR as PREPROCESS_DIR, run_preprocessing
from training import TARGET_COL, auto_train

TRAIN_FEATURES = f"{PREPROCESS_DIR}/features/train_{MODEL_NAME}.csv"
TEST_FEATURES = f"{PREPROCESS_DIR}/features/test_{MODEL_NAME}.csv"
TEST_BUSINESS = f"{PREPROCESS_DIR}/business/test_{MODEL_NAME}.csv"
TLV_OUTPUT = f"{PREPROCESS_DIR}/output_tlv.csv"


def main():
    # 1. Preprocesamiento
    run_preprocessing()

    # 2. Entrenamiento
    result = auto_train(train_path=TRAIN_FEATURES, test_path=TEST_FEATURES)
    if result is None:
        print("Sin modelo campeon - abortando pipeline.")
        return
    _, champion = result
    model = champion["model"]

    # 3. Scoring sobre train y test (test = proxy de validacion)
    df_train = pd.read_csv(TRAIN_FEATURES)
    df_test = pd.read_csv(TEST_FEATURES)
    train_scores = model.predict_proba(df_train.drop(columns=[TARGET_COL]))[:, 1]
    test_scores = model.predict_proba(df_test.drop(columns=[TARGET_COL]))[:, 1]

    # 4. Monitoreo (incluye recall por decil)
    run_monitoring(train_scores, test_scores, y_val=df_test[TARGET_COL])

    # 5. Postprocesamiento + replica
    df_post = pd.read_csv(TEST_BUSINESS)
    df_resultado = run_postprocessing(test_scores, df_post, TLV_OUTPUT)
    save_replica(df_resultado, table="EC_OMNICANAL", partition="202412")


if __name__ == "__main__":
    main()
