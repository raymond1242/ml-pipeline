"""
postprocessing.py -- Puntuacion TLV y grupos de ejecucion.
Formula:
puntuacion_tlv = prob x prob_value_contact x log(monto + 1) x prob_frescura
"""

import numpy as np
import pandas as pd

DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]


def get_groups(scores, df_post):
    """
    Calcula puntuacion_tlv y asigna grupo_ejec_tlv (1-10).
    Args:
    scores: Array con probabilidades del modelo [0, 1].
    df_post: DataFrame con columnas: grp_campecs06m,

    prob_value_contact, monto.

    Returns:
    df_post con columnas adicionales: prob, prob_frescura,
    puntuacion_tlv, grupo_ejec_tlv.
    """
    df_post["prob"] = scores
    df_post["prob_frescura"] = np.where(
        df_post["grp_campecs06m"] == "G1",
        0.066,
        np.where(
            df_post["grp_campecs06m"] == "G2",
            0.028,
            np.where(
                df_post["grp_campecs06m"] == "G3",
                0.022,
                np.where(df_post["grp_campecs06m"] == "G4", 0.008, 0.004),
            ),
        ),
    )

    df_post["prob_value_contact"] = df_post["prob_value_contact"].fillna(0.000001)
    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"] + 1)
        * df_post["prob_frescura"]
    )
    df_post["grupo_ejec_tlv"] = pd.qcut(
        df_post["puntuacion_tlv"],
        q=DIST_GE,
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )
    return df_post


def run_postprocessing(scores, df_post, output_path=None):
    """Wrapper de get_groups con guardado opcional a CSV."""
    result = get_groups(scores, df_post)
    if output_path:
        result.to_csv(output_path, index=False)
    return result


def save_replica(
    df_post,
    table,
    partition,
    dir_s3="data/replica/s3",
    dir_athena="data/replica/athena",
    dir_onpremise="data/replica/onpremise",
):
    """
    Genera el archivo de replica pipe-delimitado (|) para tres destinos.
    Columnas: codmes | tipdoc | coddoc | puntuacion | modelo |
    fec_replica | grupo_ejec | score | orden |
    variable1 | variable2 | variable3

    """
    # ... (implementar construccion del DataFrame de replica)
    pass
