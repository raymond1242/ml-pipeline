"""
main.py -- Orquestador del pipeline ML E2E.

Carga la configuracion desde config.yaml e inyecta a cada paso.

Pasos:
1. Preprocesamiento  : lee CSV, devuelve df_train/df_test/df_val.
2. Entrenamiento     : tuning Optuna + refit -> modelo campeon.
3. Scoring + monitoreo + postprocesamiento sobre la validacion temporal.
"""

import logging

import pandas as pd

from config import load_config
from monitoring import run_monitoring
from postprocessing import run_postprocessing, save_replica
from preprocessing import run_preprocessing
from training import auto_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _feature_cols(df: pd.DataFrame, target_col: str, post_cols: list[str]) -> list[str]:
    excluded = {target_col, *post_cols}
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def main():
    cfg = load_config()

    # 1. Preprocesamiento
    df_train, df_test, df_val, meta = run_preprocessing(
        data_path=cfg.input_path,
        target_col=cfg.target_col,
        prep_cfg=cfg.preprocessing,
        split_cfg=cfg.split,
    )
    logger.info("Columnas descartadas por NaN: %d", len(meta["dropped"]))

    features = _feature_cols(df_train, cfg.target_col, cfg.preprocessing.post_cols)
    train_df = df_train[[cfg.target_col, *features]]
    test_df = df_test[[cfg.target_col, *features]]

    # 2. Entrenamiento
    result = auto_train(
        train_df=train_df,
        test_df=test_df,
        target_col=cfg.target_col,
        train_cfg=cfg.training,
        split_cfg=cfg.split,
    )
    if result is None:
        logger.warning("Sin modelo campeon - abortando pipeline.")
        return
    _, champion = result
    model = champion["model"]

    # 3. Scoring sobre train y validacion temporal
    train_scores = model.predict_proba(train_df.drop(columns=[cfg.target_col]))[:, 1]
    val_scores = model.predict_proba(df_val[features])[:, 1]

    # 4. Monitoreo (incluye recall por decil)
    run_monitoring(
        train_scores, val_scores,
        y_val=df_val[cfg.target_col],
        output_dir=cfg.monitoring.output_dir,
    )

    # 5. Postprocesamiento + replica
    df_post = df_val[cfg.preprocessing.post_cols].copy()
    df_resultado = run_postprocessing(
        val_scores, df_post, cfg.postprocessing.tlv_output,
    )
    save_replica(
        df_resultado,
        table=cfg.postprocessing.replica.table,
        partition=str(int(cfg.split.validation_codmes)),
        dir_s3=cfg.postprocessing.replica.dir_s3,
        dir_athena=cfg.postprocessing.replica.dir_athena,
        dir_onpremise=cfg.postprocessing.replica.dir_onpremise,
    )


if __name__ == "__main__":
    main()
