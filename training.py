"""
training.py -- Tuning de hiperparametros via Optuna (CV estratificado) +
refit con early stopping. Selecciona el campeon por AUC test con decay
(train-test) bajo umbral; guarda modelo + metadata.

Recibe la configuracion inyectada (no usa globals).
"""

import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import catboost as catb
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import SplitConfig, TrainingConfig

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_col]), df[target_col]


def _align_columns(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindexa test a las columnas de train; rellena faltantes con 0."""
    for c in set(X_train.columns) - set(X_test.columns):
        X_test[c] = 0
    for c in set(X_test.columns) - set(X_train.columns):
        X_train[c] = 0
    return X_train, X_test[X_train.columns]


def _space_xgb(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _space_lgbm(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _space_catb(trial: optuna.Trial) -> dict:
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }


SEARCH_SPACES = {"xgb": _space_xgb, "lgbm": _space_lgbm, "catb": _space_catb}


def _make_model(
    name: str, tuned_params: dict, train_cfg: TrainingConfig, random_state: int
):
    """Construye un modelo con base params + hiperparams del tuning."""
    if name == "xgb":
        return xgb.XGBClassifier(
            n_estimators=train_cfg.max_rounds,
            early_stopping_rounds=train_cfg.early_stopping_rounds,
            eval_metric="auc",
            random_state=random_state,
            **tuned_params,
        )
    if name == "lgbm":
        return lgb.LGBMClassifier(
            n_estimators=train_cfg.max_rounds,
            random_state=random_state,
            verbose=-1,
            **tuned_params,
        )
    if name == "catb":
        return catb.CatBoostClassifier(
            iterations=train_cfg.max_rounds,
            early_stopping_rounds=train_cfg.early_stopping_rounds,
            eval_metric="AUC",
            verbose=0,
            random_state=random_state,
            **tuned_params,
        )
    raise ValueError(f"Modelo desconocido: {name}")


def _fit_with_early_stopping(
    name: str,
    model,
    X_tr,
    y_tr,
    X_es,
    y_es,
    early_stopping_rounds: int,
) -> None:
    """Fit con early stopping; dispatch por libreria (firmas distintas)."""
    if name == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    elif name == "lgbm":
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_es, y_es)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    elif name == "catb":
        model.fit(X_tr, y_tr, eval_set=(X_es, y_es), verbose=False)
    else:
        model.fit(X_tr, y_tr)


def _best_iteration(name: str, model) -> int | None:
    if name == "xgb":
        return getattr(model, "best_iteration", None)
    if name == "lgbm":
        return getattr(model, "best_iteration_", None)
    if name == "catb":
        return model.get_best_iteration()
    return None


def _cv_score(
    name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    train_cfg: TrainingConfig,
    split_cfg: SplitConfig,
) -> float:
    """Mean AUC en K-fold estratificado con early stopping intra-fold."""
    skf = StratifiedKFold(
        n_splits=split_cfg.cv_folds,
        shuffle=True,
        random_state=split_cfg.random_state,
    )
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = _make_model(name, params, train_cfg, split_cfg.random_state)
        _fit_with_early_stopping(
            name,
            model,
            X_tr,
            y_tr,
            X_va,
            y_va,
            train_cfg.early_stopping_rounds,
        )
        aucs.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
    return float(np.mean(aucs))


def _tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_cfg: TrainingConfig,
    split_cfg: SplitConfig,
) -> tuple[dict, float]:
    """Corre Optuna y devuelve (best_params, best_cv_auc)."""
    sampler = optuna.samplers.TPESampler(seed=split_cfg.random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=MedianPruner(),
    )
    study.optimize(
        lambda t: _cv_score(name, SEARCH_SPACES[name](t), X, y, train_cfg, split_cfg),
        n_trials=train_cfg.optuna_trials,
        show_progress_bar=False,
    )
    return study.best_params, study.best_value


def _train_one(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_es: pd.DataFrame,
    y_es: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_cfg: TrainingConfig,
    split_cfg: SplitConfig,
) -> dict:
    """Tuning con Optuna + refit final + metricas."""
    logger.info(
        "Tuning %s con Optuna (%d trials, %d-fold CV)...",
        name,
        train_cfg.optuna_trials,
        split_cfg.cv_folds,
    )
    tune_start = time.time()
    best_params, cv_auc = _tune(name, X_train, y_train, train_cfg, split_cfg)
    tune_elapsed = time.time() - tune_start
    logger.info(
        "  %s tuning: CV AUC=%.4f en %.1fs | params=%s",
        name,
        cv_auc,
        tune_elapsed,
        best_params,
    )

    model = _make_model(name, best_params, train_cfg, split_cfg.random_state)
    fit_start = time.time()
    _fit_with_early_stopping(
        name,
        model,
        X_tr,
        y_tr,
        X_es,
        y_es,
        train_cfg.early_stopping_rounds,
    )
    fit_elapsed = time.time() - fit_start

    auc_train = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    decay_pct = (
        ((auc_train - auc_test) / auc_train) * 100 if auc_train > 0 else float("inf")
    )
    best_iter = _best_iteration(name, model)

    logger.info(
        "Model %s | AUC train=%.4f test=%.4f | decay=%.2f%% | best_iter=%s | fit=%.2fs",
        name,
        auc_train,
        auc_test,
        decay_pct,
        best_iter,
        fit_elapsed,
    )

    return {
        "model": model,
        "performance": {
            "auc_train": auc_train,
            "auc_test": auc_test,
            "cv_auc": cv_auc,
            "decay_percent": decay_pct,
            "training_time_segs": fit_elapsed,
            "tuning_time_segs": tune_elapsed,
            "best_iteration": best_iter,
        },
        "params": model.get_params(),
        "tuned_params": best_params,
    }


def _pick_champion(results: dict, decay_max_pct: float) -> str | None:
    """Mejor AUC test entre los modelos cuyo decay sea aceptable."""
    eligible = {
        n: r
        for n, r in results.items()
        if r["performance"]["decay_percent"] < decay_max_pct
    }
    if not eligible:
        return None
    return max(eligible, key=lambda n: eligible[n]["performance"]["auc_test"])


def _clean_for_json(params: dict) -> dict:
    """Stringifica los valores no serializables a JSON."""
    cleaned = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            cleaned[k] = v
        except (TypeError, OverflowError):
            cleaned[k] = str(v)
    return cleaned


def _library_versions() -> dict:
    return {
        "xgboost": xgb.__version__,
        "lightgbm": lgb.__version__,
        "catboost": catb.__version__,
        "scikit-learn": sklearn.__version__,
        "optuna": optuna.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def save_model(
    model,
    name: str,
    performance: dict,
    params: dict,
    save_dir: Path,
    tuned_params: dict | None = None,
) -> None:
    """Guarda el modelo (joblib) y la metadata (JSON) en save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{name}_model.pkl"
    metadata_path = save_dir / f"{name}_metadata.json"

    joblib.dump(model, model_path)
    logger.info("Modelo guardado en: %s", model_path)

    metadata = {
        "ml_name": name,
        "performance": performance,
        "tuned_hyperparameters": tuned_params or {},
        "all_hyperparameters": _clean_for_json(params),
        "library_versions": _library_versions(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=4))
    logger.info("Metadata guardada en: %s", metadata_path)


def auto_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    train_cfg: TrainingConfig,
    split_cfg: SplitConfig,
):
    """
    Tuning con Optuna (CV) por modelo + refit con early stopping.
    Elige el campeon con mayor AUC test cuyo decay sea menor que decay_max_pct.

    Returns:
        (champion_name, champion_result) o None si ninguno cumple el umbral.
    """
    save_dir = Path(train_cfg.model_save_dir) / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    logger.info("Directorio de modelos: %s", save_dir)

    X_train, y_train = _xy(train_df, target_col)
    X_test, y_test = _xy(test_df, target_col)
    X_train, X_test = _align_columns(X_train, X_test)

    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train,
        y_train,
        test_size=split_cfg.internal_val_size,
        stratify=y_train,
        random_state=split_cfg.random_state,
    )

    results = {
        name: _train_one(
            name,
            X_train,
            y_train,
            X_tr,
            y_tr,
            X_es,
            y_es,
            X_test,
            y_test,
            train_cfg,
            split_cfg,
        )
        for name in SEARCH_SPACES
    }

    champion = _pick_champion(results, train_cfg.decay_max_pct)
    if champion is None:
        logger.warning(
            "No se encontro modelo campeon (decay < %s%%).",
            train_cfg.decay_max_pct,
        )
        return None

    logger.info("Modelo finalista: %s", champion)
    save_model(
        results[champion]["model"],
        champion,
        results[champion]["performance"],
        results[champion]["params"],
        save_dir,
        tuned_params=results[champion]["tuned_params"],
    )
    return champion, results[champion]


if __name__ == "__main__":
    from config import load_config
    from preprocessing import run_preprocessing

    cfg = load_config()
    df_train, df_test, _, _ = run_preprocessing(
        data_path=cfg.input_path,
        target_col=cfg.target_col,
        prep_cfg=cfg.preprocessing,
        split_cfg=cfg.split,
    )
    auto_train(
        train_df=df_train,
        test_df=df_test,
        target_col=cfg.target_col,
        train_cfg=cfg.training,
        split_cfg=cfg.split,
    )
