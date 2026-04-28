"""
main.py -- Orquestador del pipeline ML E2E.
Ejecutar:
python main.py --input "DIRECCION DRIVE/Data_CU_venta.csv"
"""

from preprocessing import run_preprocessing
from training import train_and_log
from monitoring import run_monitoring, compute_recall_by_decile
from postprocessing import run_postprocessing, save_replica

INPUT_PATH = "DIRECCION DRIVE/Data_CU_venta.csv"
OUTPUT_DIR = "data/processed"
POST_PATH = "data/postprocessed/output_tlv.csv"


def main():
    # 1. Preprocesamiento
    df_train, df_test, df_val, meta = run_preprocessing(INPUT_PATH)
    # 2. Entrenamiento con busqueda de hiperparametros
    run_id, model = train_and_log(
        train_path=OUTPUT_DIR + "/df_train.csv",
        test_path=OUTPUT_DIR + "/df_test.csv",
        val_path=OUTPUT_DIR + "/df_val.csv",
    )
    # 3. Monitoreo
    val_scores = model.predict_proba(
        df_val.drop(columns=["p_codmes", "key_value", "target"])
    )[:, 1]
    run_monitoring(df_train, df_val, val_scores=val_scores)
    compute_recall_by_decile(df_val["target"], val_scores)
    # 4. Postprocesamiento y replica
    df_resultado = run_postprocessing(val_scores, df_val, POST_PATH)
    save_replica(df_resultado, table="EC_OMNICANAL", partition="202412")


if __name__ == "__main__":
    main()
