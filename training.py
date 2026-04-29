"""
training.py -- Entrena XGBoost, LightGBM y CatBoost; selecciona el campeon
por AUC test con decay (train-test) bajo umbral; guarda modelo + metadata.
"""

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import catboost as catb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TARGET_COL = "target"
RANDOM_STATE = 42
DECAY_MAX_PCT = 10.0  # umbral de generalizacion (train vs test)


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def _align_columns(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindexa test a las columnas de train; rellena faltantes con 0."""
    for c in set(X_train.columns) - set(X_test.columns):
        X_test[c] = 0
    for c in set(X_test.columns) - set(X_train.columns):
        X_train[c] = 0
    return X_train, X_test[X_train.columns]


def _build_models() -> dict:
    return {
        "xgb": xgb.XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
        "lgbm": lgb.LGBMClassifier(random_state=RANDOM_STATE),
        "catb": catb.CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    }


def _train_one(name, model, X_train, y_train, X_test, y_test) -> dict:
    """Entrena un modelo y devuelve metricas + el modelo entrenado."""
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    decay_pct = (
        ((auc_train - auc_test) / auc_train) * 100 if auc_train > 0 else float("inf")
    )

    print(f"Model: {name}")
    print(f"  AUC Train: {auc_train:.4f}")
    print(f"  AUC Test:  {auc_test:.4f}")
    print(f"  Decay (%): {decay_pct:.2f}")
    print(f"  Time (s):  {elapsed:.2f}")
    print("-" * 30)

    return {
        "model": model,
        "performance": {
            "auc_train": auc_train,
            "auc_test": auc_test,
            "decay_percent": decay_pct,
            "training_time_segs": elapsed,
        },
        "params": model.get_params(),
    }


def _pick_champion(results: dict, decay_max_pct: float) -> str | None:
    """Mejor AUC test entre los modelos cuyo decay sea aceptable."""
    eligible = {
        n: r for n, r in results.items()
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
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def save_model(
    model, name: str, performance: dict, params: dict, save_dir: Path
) -> None:
    """Guarda el modelo (joblib) y la metadata (JSON) en save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{name}_model.pkl"
    metadata_path = save_dir / f"{name}_metadata.json"

    joblib.dump(model, model_path)
    print(f"Modelo guardado en:   {model_path}")

    metadata = {
        "ml_name": name,
        "performance": performance,
        "hyperparameters": _clean_for_json(params),
        "library_versions": _library_versions(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=4))
    print(f"Metadata guardada en: {metadata_path}")


def auto_train(
    train_path: str,
    test_path: str,
    model_save_dir: str = "models",
    decay_max_pct: float = DECAY_MAX_PCT,
):
    """
    Entrena XGB, LGBM y CatBoost; elige el campeon con mayor AUC test
    cuyo decay (train vs test) sea menor que decay_max_pct.

    Returns:
        (champion_name, champion_result) o None si ninguno cumple el umbral.
    """
    save_dir = Path(model_save_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Directorio de modelos: {save_dir}\n")

    X_train, y_train = _xy(pd.read_csv(train_path))
    X_test, y_test = _xy(pd.read_csv(test_path))
    X_train, X_test = _align_columns(X_train, X_test)

    results = {
        name: _train_one(name, model, X_train, y_train, X_test, y_test)
        for name, model in _build_models().items()
    }

    champion = _pick_champion(results, decay_max_pct)
    if champion is None:
        print(f"\nNo se encontro modelo campeon (decay < {decay_max_pct}%).")
        return None

    print(f"\nModelo finalista: {champion}")
    save_model(
        results[champion]["model"],
        champion,
        results[champion]["performance"],
        results[champion]["params"],
        save_dir,
    )
    return champion, results[champion]


if __name__ == "__main__":
    auto_train(
        train_path="preprocess_data/preprocessed/train_vars_extrac.csv",
        test_path="preprocess_data/preprocessed/test_vars_extrac.csv",
    )
