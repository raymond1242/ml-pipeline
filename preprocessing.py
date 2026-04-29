"""
preprocessing.py -- Limpieza, encoding y split del dataset CU Venta.

Lee todos los CSV de train_date/, procesa variables (fillna, casteo,
one-hot encoding) y guarda train/test en preprocess_data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DIR = "train_data"
OUTPUT_DIR = "preprocess_data"
MODEL_NAME = "extrac"
TARGET_COL = "target"
TEST_SIZE = 0.33
RANDOM_STATE = 123
NA_FILL_NUMERIC = -9999999
NA_TOKENS = ["", "null", "None"]
FLOAT_DECIMALS = 4

NUMERIC_COLS = [
    "nro_producto_6m", "prom_uso_tc_rccsf3m", "ctd_sms_received",
    "max_usotcribksf06m", "ctd_camptot06m", "dsv_svppallsf06m",
    "prm_svprmecs06m", "ctd_app_productos_m1", "ctd_campecsm01",
    "lin_tcrrstsf03m", "mnt_ptm", "dif_no_gestionado_4meses",
    "max_campecs06m", "beta_pctusotcr12m", "rat_disefepnm01",
    "flg_saltotppe12m", "prom_sow_lintcribksf3m", "openhtml_1m",
    "nprod_1m", "nro_transfer_6m", "max_usotcrrstsf03m",
    "prm_cnt_fee_amt_u7d", "pas_avg6m_max12m", "beta_saltotppe12m",
    "seg_un", "ant_ultprdallsf", "avg_sald_pas_3m", "pas_1m_avg3m",
    "num_incrsaldispefe06m", "cnl_age_p4m_p12m", "cnl_atm_p4m_p12m",
    "cre_lin_tc_rccibk_m07", "prm_svprmlibdis06m", "ingreso_neto",
    "max_nact_12m", "cre_sldtotfinprm03", "dif_contacto_efectivo_10meses",
    "act_1m_avg3m", "monto_consumos_ecommerce_tc", "ctd_camptotm01",
    "prop_atm_4m", "prom_pct_saldopprcc6m", "apppag_1m",
    "nro_configuracion_6m", "act_avg6m_max12m", "sldvig_tcrsrcf",
    "prom_score_acepta_12meses", "telefonos_6meses", "pas_1m_avg6m",
    "ctd_camptototrcnl06m", "prm_saltotrdpj03m", "bpitrx_1m",
    "prm_lintcribksf03m", "ctd_entrdm01", "avg_openhtml_6m", "tea",
    "pct_usotcrm01", "senthtml_1m",
]

# Categorica -> categorias a conservar; el resto colapsa a "OTRO".
CATEGORICAL_KEEP = {"ent_1erlntcrallsfm01": ["INTERBANK"]}
OTHER_LABEL = "OTRO"

POST_COLS = [
    "partition", "key_value", "codunicocli", "grp_campecs06m",
    "prob_value_contact", "monto",
]


def read_train_data(train_dir: str) -> pd.DataFrame:
    """Concatena todos los CSV en train_dir en un unico DataFrame."""
    files = sorted(Path(train_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def process_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nulls, castea tipos y aplica one-hot a las variables categoricas."""
    df = df.replace(NA_TOKENS, np.nan)

    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(NA_FILL_NUMERIC).astype("float32")
    if "partition" in df.columns:
        df["partition"] = df["partition"].astype("string")

    for col, keep in CATEGORICAL_KEEP.items():
        categories = [*keep, OTHER_LABEL]
        collapsed = df[col].where(df[col].isin(keep), OTHER_LABEL)
        cat_series = collapsed.astype(pd.CategoricalDtype(categories))
        dummies = pd.get_dummies(cat_series, prefix=col)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return _coerce_dtypes(df)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos listos para ML: bool/int -> int; floats -> redondeados."""
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "bool" or pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].astype(int)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(float).round(FLOAT_DECIMALS)
    return df


def _encoded_categorical_cols() -> list[str]:
    return [
        f"{col}_{cat}"
        for col, keep in CATEGORICAL_KEEP.items()
        for cat in [*keep, OTHER_LABEL]
    ]


def split_and_save(df: pd.DataFrame, output_dir: str, model: str = MODEL_NAME) -> None:
    """Train/test split y guardado en features/ (modelo) y business/ (postproc)."""
    feature_cols = NUMERIC_COLS + _encoded_categorical_cols()
    keep_cols = list(dict.fromkeys(feature_cols + POST_COLS))

    x_train, x_test, y_train, y_test = train_test_split(
        df[keep_cols], df[TARGET_COL],
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    features_dir = Path(output_dir) / "features"
    business_dir = Path(output_dir) / "business"
    features_dir.mkdir(parents=True, exist_ok=True)
    business_dir.mkdir(parents=True, exist_ok=True)

    for prefix, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        pd.concat([y, x[feature_cols]], axis=1).to_csv(
            features_dir / f"{prefix}_{model}.csv", index=False,
        )
        x[POST_COLS].to_csv(
            business_dir / f"{prefix}_{model}.csv", index=False,
        )


def run_preprocessing(train_dir: str = TRAIN_DIR, output_dir: str = OUTPUT_DIR) -> None:
    df = read_train_data(train_dir)
    df = process_vars(df)
    split_and_save(df, output_dir)


if __name__ == "__main__":
    run_preprocessing()
