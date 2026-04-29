"""
monitoring.py -- PSI, AUC y Recall por decil (monitoreo de deriva).

Umbrales PSI:
    < 0.10        -> OK     (sin deriva)
    0.10 - 0.25   -> WARN   (deriva moderada)
    > 0.25        -> ALERT  (deriva severa)
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

PSI_EPS = 1e-6


def psi_flag(psi: float) -> str:
    """Etiqueta de alerta segun el valor de PSI."""
    if psi < 0.10:
        return "OK"
    if psi < 0.25:
        return "WARN"
    return "ALERT"


def compute_psi(train_scores, val_scores, n_bins: int = 10) -> float:
    """PSI sobre deciles definidos con la distribucion de train."""
    edges = np.quantile(train_scores, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    train_pct = np.histogram(train_scores, bins=edges)[0] / len(train_scores)
    val_pct = np.histogram(val_scores, bins=edges)[0] / len(val_scores)

    train_pct = np.where(train_pct == 0, PSI_EPS, train_pct)
    val_pct = np.where(val_pct == 0, PSI_EPS, val_pct)

    return float(np.sum((val_pct - train_pct) * np.log(val_pct / train_pct)))


def compute_recall_by_decile(y_true, scores, n_deciles: int = 10) -> pd.DataFrame:
    """
    Recall acumulado por decil de score (decil 1 = mayor score).

    Returns:
        DataFrame con columnas: decil, recall_acumulado.
    """
    df = pd.DataFrame({"score": scores, "target": y_true})
    df["decil"] = pd.qcut(
        df["score"], q=n_deciles, labels=range(n_deciles, 0, -1)
    )

    total_pos = df["target"].sum()
    if total_pos == 0:
        return pd.DataFrame({"decil": range(1, n_deciles + 1), "recall_acumulado": 0.0})

    pos_by_decile = df.groupby("decil", observed=True)["target"].sum().sort_index()
    recall = pos_by_decile.cumsum() / total_pos

    return pd.DataFrame({
        "decil": recall.index.astype(int),
        "recall_acumulado": recall.values,
    })


def run_monitoring(
    train_scores,
    val_scores,
    y_val,
    output_dir: str = "data/monitoring",
    mlflow_active: bool = False,
) -> dict:
    """
    Calcula PSI sobre deciles de score, AUC y Recall en validacion.
    Guarda recall_by_decile.csv y monitoring.json en output_dir.

    Returns:
        dict con psi_score, psi_flag y model_metrics_val.
    """
    psi = compute_psi(train_scores, val_scores)
    auc = roc_auc_score(y_val, val_scores)
    recall_at_05 = recall_score(y_val, (val_scores >= 0.5).astype(int))
    recall_table = compute_recall_by_decile(y_val, val_scores)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    recall_table.to_csv(out / "recall_by_decile.csv", index=False)

    metrics = {
        "psi_score": psi,
        "psi_flag": psi_flag(psi),
        "model_metrics_val": {"auc": auc, "recall_at_0.5": recall_at_05},
    }
    (out / "monitoring.json").write_text(json.dumps(metrics, indent=2, default=str))

    if mlflow_active:
        mlflow.log_metric("val_psi", psi)
        mlflow.log_metric("val_auc", auc)
        mlflow.log_metric("val_recall_0.5", recall_at_05)

    print(f"PSI:       {psi:.4f} ({psi_flag(psi)})")
    print(f"AUC val:   {auc:.4f}")
    print(f"Recall val (thr=0.5): {recall_at_05:.4f}")
    return metrics
