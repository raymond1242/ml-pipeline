"""
main.py -- Orquestador del pipeline ML E2E.

Carga la configuracion desde config.yaml e inyecta a cada paso.

Pasos:
1. Preprocesamiento  : lee CSV, devuelve df_train/df_test/df_val.
2. Entrenamiento     : tuning Optuna + refit -> modelo campeon.
3. Scoring + monitoreo + postprocesamiento sobre la validacion temporal.
"""

import logging
from contextlib import nullcontext

import mlflow
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
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def main():
    cfg = load_config()

    if cfg.mlflow.enabled:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_ctx = (
        mlflow.start_run(run_name="pipeline") if cfg.mlflow.enabled else nullcontext()
    )

    with run_ctx:
        if cfg.mlflow.enabled:
            mlflow.log_params({
                "input_path": cfg.input_path,
                "validation_codmes": cfg.split.validation_codmes,
                "test_size": cfg.split.test_size,
                "internal_val_size": cfg.split.internal_val_size,
                "cv_folds": cfg.split.cv_folds,
                "optuna_trials": cfg.training.optuna_trials,
                "max_rounds": cfg.training.max_rounds,
                "early_stopping_rounds": cfg.training.early_stopping_rounds,
                "decay_max_pct": cfg.training.decay_max_pct,
                "random_state": cfg.split.random_state,
            })

        # 1. Preprocesamiento
        df_train, df_test, df_val, meta = run_preprocessing(
            data_path=cfg.input_path,
            target_col=cfg.target_col,
            prep_cfg=cfg.preprocessing,
            split_cfg=cfg.split,
        )
        logger.info("Columnas descartadas por NaN: %d", len(meta["dropped"]))
        if cfg.mlflow.enabled:
            mlflow.log_metric("cols_dropped_nan", len(meta["dropped"]))

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
            if cfg.mlflow.enabled:
                mlflow.set_tag("champion", "none")
            return
        champion_name, champion = result
        model = champion["model"]

        if cfg.mlflow.enabled:
            mlflow.set_tag("champion", champion_name)
            mlflow.log_metrics({
                "champion_auc_train": champion["performance"]["auc_train"],
                "champion_auc_test": champion["performance"]["auc_test"],
                "champion_cv_auc": champion["performance"]["cv_auc"],
                "champion_decay_pct": champion["performance"]["decay_percent"],
                "champion_best_iteration": champion["performance"]["best_iteration"],
            })
            mlflow.log_params({
                f"champion_{k}": v for k, v in champion["tuned_params"].items()
            })
            mlflow.sklearn.log_model(model, name="model")

        # 3. Scoring sobre train y validacion temporal
        train_scores = model.predict_proba(train_df.drop(columns=[cfg.target_col]))[:, 1]
        val_scores = model.predict_proba(df_val[features])[:, 1]

        # 4. Monitoreo (incluye recall por decil)
        run_monitoring(
            train_scores,
            val_scores,
            y_val=df_val[cfg.target_col],
            output_dir=cfg.monitoring.output_dir,
            mlflow_active=cfg.mlflow.enabled,
        )

        # 5. Postprocesamiento + replica
        df_post = df_val[cfg.preprocessing.post_cols].copy()
        df_resultado = run_postprocessing(
            val_scores,
            df_post,
            cfg.postprocessing.tlv_output,
        )
        save_replica(
            df_resultado,
            table=cfg.postprocessing.replica.table,
            partition=str(int(cfg.split.validation_codmes)),
            dir_s3=cfg.postprocessing.replica.dir_s3,
            dir_athena=cfg.postprocessing.replica.dir_athena,
            dir_onpremise=cfg.postprocessing.replica.dir_onpremise,
        )

        if cfg.mlflow.enabled:
            mlflow.log_artifact(f"{cfg.monitoring.output_dir}/monitoring.json")
            mlflow.log_artifact(f"{cfg.monitoring.output_dir}/recall_by_decile.csv")
            mlflow.log_artifact(cfg.postprocessing.tlv_output)


if __name__ == "__main__":
    main()
