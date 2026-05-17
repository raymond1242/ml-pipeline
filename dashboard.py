"""
dashboard.py -- Dashboard interactivo para visualizar resultados del pipeline.

Lee del output del ultimo run (data/postprocessing/output_tlv.csv,
data/monitoring/*) y del historico de runs registrado en MLflow.

Ejecutar:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

from config import load_config

st.set_page_config(page_title="CU Venta Dashboard", layout="wide")
st.title("Dashboard Pipeline CU Venta")

cfg = load_config()


def _load_tlv() -> pd.DataFrame | None:
    path = Path(cfg.postprocessing.tlv_output)
    return pd.read_csv(path) if path.is_file() else None


def _load_monitoring_json() -> dict | None:
    path = Path(cfg.monitoring.output_dir) / "monitoring.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _load_recall() -> pd.DataFrame | None:
    path = Path(cfg.monitoring.output_dir) / "recall_by_decile.csv"
    return pd.read_csv(path) if path.is_file() else None


def _load_mlflow_runs() -> pd.DataFrame | None:
    if not cfg.mlflow.enabled:
        return None
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    try:
        runs = mlflow.search_runs(experiment_names=[cfg.mlflow.experiment_name])
    except mlflow.exceptions.MlflowException:
        return None
    if runs.empty:
        return None
    return runs.sort_values("start_time")


# ============================== KPIs (último run) ==============================
monitoring = _load_monitoring_json()
runs_df = _load_mlflow_runs()

st.header("Resumen del último run")
if monitoring is None and runs_df is None:
    st.warning(
        "No hay resultados todavía. Ejecuta `python main.py` para generar "
        "el primer run del pipeline."
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
if monitoring is not None:
    col1.metric("PSI", f"{monitoring['psi_score']:.4f}", monitoring["psi_flag"])
    col2.metric("AUC val", f"{monitoring['model_metrics_val']['auc']:.4f}")
    col3.metric("Recall @ 0.5", f"{monitoring['model_metrics_val']['recall_at_0.5']:.4f}")
if runs_df is not None:
    last = runs_df.iloc[-1]
    champion = last.get("tags.champion", "n/a")
    col4.metric("Champion", str(champion))


# =============================== Tabs ===============================
tab_run, tab_history, tab_recall = st.tabs(
    ["Último run", "Histórico (MLflow)", "Recall por decil"]
)

# ---------- TAB 1: Último run ----------
with tab_run:
    df = _load_tlv()
    if df is None:
        st.warning(
            f"No se encontró `{cfg.postprocessing.tlv_output}`. "
            "Ejecuta el pipeline primero."
        )
    else:
        st.subheader("Distribución de grupos de ejecución")
        groups = df["grupo_ejec_tlv"].value_counts().sort_index()
        c1, c2 = st.columns([2, 1])
        c1.bar_chart(groups)
        c2.dataframe(groups.rename("clientes"), use_container_width=True)

        st.subheader("Score promedio y monto promedio por grupo")
        agg = (
            df.groupby("grupo_ejec_tlv", observed=True)
            .agg(
                clientes=("key_value", "count"),
                score_promedio=("prob", "mean"),
                monto_promedio=("monto", "mean"),
                puntuacion_promedio=("puntuacion_tlv", "mean"),
            )
            .round(4)
            .sort_index()
        )
        st.dataframe(agg, use_container_width=True)

        c3, c4 = st.columns(2)
        c3.markdown("**Score promedio (efectividad esperada) por grupo**")
        c3.bar_chart(agg["score_promedio"])
        c4.markdown("**Monto promedio por grupo**")
        c4.bar_chart(agg["monto_promedio"])

        st.subheader("Top-N clientes por puntuación TLV")
        n = st.slider("N", min_value=10, max_value=200, value=20, step=10)
        top = (
            df[["key_value", "prob", "monto", "puntuacion_tlv", "grupo_ejec_tlv"]]
            .sort_values("puntuacion_tlv", ascending=False)
            .head(n)
        )
        st.dataframe(top, use_container_width=True)

# ---------- TAB 2: Histórico MLflow ----------
with tab_history:
    if runs_df is None:
        if not cfg.mlflow.enabled:
            st.info("MLflow está deshabilitado en `config.yaml` (`mlflow.enabled: false`).")
        else:
            st.warning("Aún no hay runs registrados en MLflow.")
    else:
        st.caption(f"{len(runs_df)} run(s) en el experimento `{cfg.mlflow.experiment_name}`.")

        def _col(name):
            return runs_df[name] if name in runs_df.columns else pd.Series([pd.NA] * len(runs_df))

        history = pd.DataFrame({
            "run_id": runs_df["run_id"].str[:8].values,
            "start_time": pd.to_datetime(runs_df["start_time"]).values,
            "validation_codmes": _col("params.validation_codmes").values,
            "champion": _col("tags.champion").values,
            "auc_test": _col("metrics.champion_auc_test").values,
            "auc_val": _col("metrics.val_auc").values,
            "psi": _col("metrics.val_psi").values,
            "decay_pct": _col("metrics.champion_decay_pct").values,
        }).set_index("start_time")

        st.subheader("Evolución de AUC")
        auc_cols = [c for c in ["auc_test", "auc_val"] if history[c].notna().any()]
        if auc_cols:
            st.line_chart(history[auc_cols])
        else:
            st.info("Sin métricas de AUC en los runs disponibles.")

        st.subheader("Evolución de PSI")
        if history["psi"].notna().any():
            st.line_chart(history[["psi"]])
        else:
            st.info("Sin métrica de PSI en los runs disponibles.")

        st.subheader("Tabla de runs")
        st.dataframe(history.reset_index(), use_container_width=True)

# ---------- TAB 3: Recall por decil ----------
with tab_recall:
    recall = _load_recall()
    if recall is None:
        st.warning(
            f"No se encontró `{cfg.monitoring.output_dir}/recall_by_decile.csv`. "
            "Ejecuta el pipeline primero."
        )
    else:
        st.subheader("Recall acumulado por decil")
        st.caption(
            "Decil 1 = scores más altos. Una curva que sube rápido indica "
            "que el modelo concentra positivos en los deciles top (buen lift)."
        )
        st.line_chart(recall.set_index("decil")["recall_acumulado"])
        st.dataframe(recall, use_container_width=True)
