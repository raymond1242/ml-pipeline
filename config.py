"""
config.py -- Carga y expone la configuracion del pipeline desde config.yaml.

Cada paso del pipeline recibe la sub-config que necesita (inyeccion),
en vez de leer variables globales.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class PreprocessingConfig:
    nan_threshold: float
    na_fill_numeric: int
    na_tokens: list[str]
    float_decimals: int
    post_cols: list[str]


@dataclass(frozen=True)
class SplitConfig:
    validation_codmes: float
    test_size: float
    internal_val_size: float
    cv_folds: int
    random_state: int


@dataclass(frozen=True)
class TrainingConfig:
    decay_max_pct: float
    early_stopping_rounds: int
    max_rounds: int
    optuna_trials: int
    model_save_dir: str


@dataclass(frozen=True)
class MonitoringConfig:
    output_dir: str


@dataclass(frozen=True)
class ReplicaConfig:
    table: str
    dir_s3: str
    dir_athena: str
    dir_onpremise: str


@dataclass(frozen=True)
class PostprocessingConfig:
    tlv_output: str
    replica: ReplicaConfig


@dataclass(frozen=True)
class MLflowConfig:
    enabled: bool
    experiment_name: str
    tracking_uri: str


@dataclass(frozen=True)
class Config:
    input_path: str
    target_col: str
    preprocessing: PreprocessingConfig
    split: SplitConfig
    training: TrainingConfig
    monitoring: MonitoringConfig
    postprocessing: PostprocessingConfig
    mlflow: MLflowConfig


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        input_path=raw["input_path"],
        target_col=raw["target_col"],
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        split=SplitConfig(**raw["split"]),
        training=TrainingConfig(**raw["training"]),
        monitoring=MonitoringConfig(**raw["monitoring"]),
        postprocessing=PostprocessingConfig(
            tlv_output=raw["postprocessing"]["tlv_output"],
            replica=ReplicaConfig(**raw["postprocessing"]["replica"]),
        ),
        mlflow=MLflowConfig(**raw["mlflow"]),
    )
