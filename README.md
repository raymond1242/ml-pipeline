# CU Venta ML Pipeline

End-to-end ML pipeline for ranking customers by Total Lifetime Value (TLV)
in cross-/up-selling campaigns. Trains a binary classifier
(XGBoost / LightGBM / CatBoost) with Optuna hyperparameter tuning,
selects a champion based on test AUC and a generalization-decay constraint,
and produces a pipe-delimited replica file ready for S3, Athena, and
on-premise consumers.

**Repository:** <https://github.com/raymond1242/ml-pipeline>

## Pipeline overview

```
data/raw/Data_CU_venta.csv
    │
    ▼
[1. preprocessing]  ─►  df_train / df_test / df_val (in-memory)
                            │
                            ▼
[2. training]       ─►  models/<timestamp>/  champion model + metadata
                            │
                            ▼
[3. monitoring]     ─►  data/monitoring/    PSI, AUC, recall by decile
                            │
                            ▼
[4. postprocessing] ─►  data/replica/       replica file (S3/Athena/on-prem)
```

All four stages are orchestrated by `main.py`, which loads a single
`config.yaml` and injects the relevant sub-config into each step.
Stage outputs are passed in-memory between steps (no intermediate CSVs).

Every run is also tracked in **MLflow** (params, metrics, the champion
model, and the monitoring/postprocessing artifacts) so you can compare
runs over time and download a model from any past execution.

## Project structure

```
ml-pipeline/
├── main.py              # orchestrator (loads config, wires stages)
├── config.yaml          # all tunable settings (paths, splits, hyperparams)
├── config.py            # YAML loader + frozen dataclasses
├── preprocessing.py     # NaN drop + cast + temporal val split
├── training.py          # Optuna tuning + early stopping + champion pick
├── monitoring.py        # PSI + AUC + recall-by-decile + drift
├── postprocessing.py    # TLV score + execution groups + replica file
├── dashboard.py         # Streamlit dashboard (visualizes pipeline results)
├── requirements.in      # source dependencies
├── requirements.txt     # pinned (generated with pip-compile)
└── data/raw/            # input CSV lives here
```

Directories and files created by the pipeline (auto-created at first
write — you don't need to mkdir them):

```
models/<timestamp>/        # stage 2 output (model + metadata)
data/monitoring/           # stage 3 output (PSI, recall)
data/postprocessing/       # stage 4 intermediate (TLV scores)
data/replica/{s3,athena,onpremise}/   # stage 4 final replica files
mlflow.db                  # MLflow tracking DB (SQLite)
mlruns/                    # MLflow artifacts (model.pkl, JSON, CSVs per run)
catboost_info/             # CatBoost training-time artifact (safe to delete)
```

All of these are in `.gitignore`. After cloning the repo and placing
`Data_CU_venta.csv` in `data/raw/`, you can run `python main.py` directly
— every output directory is materialized on demand.

## Setup

### System requirements

- Python 3.12+ (tested on 3.14)
- On macOS, LightGBM needs OpenMP at runtime:
  ```bash
  brew install libomp
  ```

### Training data

The `data/raw/` folder is **not committed to the repo** (the CSV is too
large). Download `Data_CU_venta.csv` from the class shared drive:

[Google Drive — class dataset](https://drive.google.com/drive/folders/1ViWZBI7Gt5TTSeNTxL5x8EM4ARCi0Ulf?usp=share_link)

Expected layout once downloaded:

```
data/raw/
└── Data_CU_venta.csv
```

The path is configurable in `config.yaml` (`input_path`).

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline (~5–10 min, Optuna tuning dominates):

```bash
python main.py
```

Browse experiments in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Launch the interactive dashboard:

```bash
streamlit run dashboard.py                          # http://localhost:8501
```

Every knob (paths, splits, hyperparams, MLflow toggle) lives in
`config.yaml` — no CLI flags.

## Notes

- `models/<timestamp>/` accumulates one folder per training run; old runs
  are not auto-cleaned.
- CatBoost writes a `catboost_info/` directory to the working directory
  during training. It's a training-time artifact, safe to delete or
  add to `.gitignore`.
- Optuna tuning is the slowest stage (~5–10 min on a laptop CPU for the
  default 30 trials × 3 folds × 3 models). For faster iteration during
  development, drop `training.optuna_trials` in `config.yaml`.
