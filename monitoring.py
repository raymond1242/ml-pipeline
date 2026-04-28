"""
monitoring.py -- PSI, AUC y Recall por decil (monitoreo de deriva).
Umbrales PSI:
< 0.10 -> OK (sin deriva)
0.10 - 0.25 -> WARN (deriva moderada)
> 0.25 -> ALERT (deriva severa)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
import mlflow

def psi_flag(psi: float) -> str:
    """Retorna la etiqueta de alerta segun el valor de PSI."""
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARN"
    return "ALERT"

def run_monitoring(df_train, df_val, val_scores,
    id_cols=None, target_col="target",
    output_dir="data/monitoring", mlflow_active=False):

    """
    Calcula PSI sobre deciles de score, AUC y Recall en validacion.
    Returns:
    dict con psi_score, model_metrics_val
    """
    # ... (implementar calculo PSI y metricas)
    pass

def compute_recall_by_decile(y_true, scores, n_deciles=10):
    """
    Calcula el Recall acumulado por decil de score (decil 1 = mayor score).
    Returns:
    DataFrame con columnas: decil, recall_acumulado
    """
    df = pd.DataFrame({"score": scores, "target": y_true})
    df["decil"] = pd.qcut(df["score"], q=n_deciles,
    labels=range(n_deciles, 0, -1))

    # ... (implementar acumulado)
    return df
