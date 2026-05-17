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
pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip install -r requirements.txt
```

To refresh pins after editing `requirements.in`:

```bash
pip-compile --upgrade requirements.in -o requirements.txt
```

## Usage

### End-to-end

```bash
python main.py
```

Runs all four stages in order. No CLI flags — everything that's tunable
lives in `config.yaml`. Expect ~5–10 minutes the first time (Optuna tuning
dominates the runtime).

### Per stage

Each module is runnable on its own for iteration; the `__main__` block
loads the config and calls the entry function:

```bash
python preprocessing.py    # exercises the preprocessing path
python training.py         # runs preprocessing + training only
```

### MLflow UI

After at least one run, browse experiments, compare runs, and download
artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://localhost:5000
```

The `--backend-store-uri` flag is required because the pipeline uses a
SQLite backend (set in `config.yaml`); without the flag, `mlflow ui`
defaults to a file-store backend and won't find your runs.

To disable MLflow tracking entirely (faster iteration, no DB writes), set
`mlflow.enabled: false` in `config.yaml`.

### Dashboard (Streamlit)

Interactive dashboard with the pipeline outputs:

```bash
streamlit run dashboard.py
# open http://localhost:8501
```

It pulls data from the latest run (`output_tlv.csv`, `monitoring.json`,
`recall_by_decile.csv`) and from the MLflow history. Sections:

- **KPIs**: PSI, AUC val, recall, champion model.
- **Tab "Último run"**: distribution of execution groups, score and
  average amount per group, top-N customers by TLV score (slider).
- **Tab "Histórico (MLflow)"**: evolution of AUC and PSI across runs,
  full table of runs.
- **Tab "Recall por decil"**: cumulative recall curve.

The dashboard works even before the first pipeline run — it falls back
to an informative message instead of crashing.

## Stages

### 1. Preprocessing — `preprocessing.py`

Drops columns with too many NaNs, imputes numerics with a sentinel,
coerces types, and splits the data into `df_train` / `df_test` /
`df_val`. `df_val` is a true out-of-time hold-out (rows with
`p_codmes == validation_codmes`).

### 2. Training — `training.py`

For each of XGBoost / LightGBM / CatBoost: tunes hyperparameters with
Optuna (stratified CV on `df_train`), then refits with the best params
using early stopping on an internal slice of train. `df_test` is held
out for honest reporting.

The **champion** is the model with the highest `auc_test` whose
`decay_pct = (auc_train − auc_test) / auc_train × 100` stays under
`decay_max_pct` — rejecting high-AUC models that overfit.

Outputs `models/<timestamp>/{name}_model.pkl` + `{name}_metadata.json`.

### 3. Monitoring — `monitoring.py`

Computes PSI (drift), AUC and recall on `df_val`, plus cumulative recall
by decile (lift). Writes `monitoring.json` and `recall_by_decile.csv`.

### 4. Postprocessing — `postprocessing.py`

Combines model probability with business signals:

```
puntuacion_tlv = prob × prob_value × log(monto + 1) × prob_frescura
```

Buckets customers into 10 execution groups (1 = highest priority) and
writes the pipe-delimited replica file to `data/replica/{s3,athena,onpremise}/`.

## Notes

- `models/<timestamp>/` accumulates one folder per training run; old runs
  are not auto-cleaned.
- CatBoost writes a `catboost_info/` directory to the working directory
  during training. It's a training-time artifact, safe to delete or
  add to `.gitignore`.
- Optuna tuning is the slowest stage (~5–10 min on a laptop CPU for the
  default 30 trials × 3 folds × 3 models). For faster iteration during
  development, drop `training.optuna_trials` in `config.yaml`.
