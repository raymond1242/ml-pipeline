"""
training.py -- Entrenamiento XGBoost + busqueda de hiperparametros con Optuna.
Registra parametros, metricas y modelo en MLflow.
"""

import optuna
import xgboost as xgb
import mlflow, mlflow.xgboost
import pandas as pd
from sklearn.metrics import roc_auc_score

ID_COLS = ["p_codmes", "key_value"]
TARGET_COL = "target"


def _xy(df):
    drop = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    return df.drop(columns=drop), df[TARGET_COL]


def train_and_log(
    train_path, test_path, val_path, n_trials=30, experiment_name="cu_venta_e2e"
):
    """
    Busca hiperparametros con Optuna y registra el mejor modelo en MLflow.
    Returns:
    run_id (str), modelo entrenado con mejores parametros
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train, y_train = _xy(df_train)
    X_test, y_test = _xy(df_test)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = {
        **study.best_params,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test.auc", auc)
        mlflow.xgboost.log_model(model, "model", registered_model_name="cu_venta_xgb")

    return run.info.run_id, model
