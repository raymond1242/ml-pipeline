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
├── requirements.in      # source dependencies
├── requirements.txt     # pinned (generated with pip-compile)
└── data/raw/            # input CSV lives here
```

Directories created by the pipeline:

```
models/<timestamp>/        # stage 2 output (model + metadata)
data/monitoring/           # stage 3 output (PSI, recall)
data/postprocessing/       # stage 4 intermediate (TLV scores)
data/replica/{s3,athena,onpremise}/   # stage 4 final replica files
```

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
lives in `config.yaml`.

### Per stage

Each module is runnable on its own for iteration; the `__main__` block
loads the config and calls the entry function:

```bash
python preprocessing.py    # exercises the preprocessing path
python training.py         # runs preprocessing + training only
```

## Configuration (`config.yaml`)

The orchestrator passes sub-configs (`PreprocessingConfig`, `SplitConfig`,
`TrainingConfig`, …) into each stage. Nothing in the pipeline reads a
global constant — every knob is here:

| Section                  | Key                       | Default                          | Meaning                                          |
|--------------------------|---------------------------|----------------------------------|--------------------------------------------------|
| (root)                   | `input_path`              | `data/raw/Data_CU_venta.csv`     | Source CSV                                       |
| (root)                   | `target_col`              | `target`                         | Target column name                               |
| `preprocessing`          | `nan_threshold`           | `80`                             | Drop columns with >X% NaN                        |
| `preprocessing`          | `na_fill_numeric`         | `-9999999`                       | Sentinel for missing numeric values              |
| `preprocessing`          | `na_tokens`               | `["", "null", "None"]`           | String tokens treated as NaN                     |
| `preprocessing`          | `float_decimals`          | `4`                              | Float rounding before save                       |
| `preprocessing`          | `post_cols`               | `[p_codmes, key_value, ...]`     | Cols carried for postprocessing (not features)   |
| `split`                  | `validation_codmes`       | `201912.0`                       | Temporal hold-out partition                      |
| `split`                  | `test_size`               | `0.30`                           | Train/test split fraction                        |
| `split`                  | `internal_val_size`       | `0.15`                           | Slice of train used for refit early stopping     |
| `split`                  | `cv_folds`                | `3`                              | K for stratified CV during tuning                |
| `split`                  | `random_state`            | `42`                             | Global seed                                      |
| `training`               | `decay_max_pct`           | `10.0`                           | Max train→test decay for champion eligibility   |
| `training`               | `early_stopping_rounds`   | `50`                             | Rounds w/o improvement before stopping           |
| `training`               | `max_rounds`              | `2000`                           | Upper bound; early stopping decides actual count |
| `training`               | `optuna_trials`           | `30`                             | Trials per model                                 |
| `training`               | `model_save_dir`          | `models`                         | Root for `<timestamp>/` model folders            |
| `monitoring`             | `output_dir`              | `data/monitoring`                | PSI + recall outputs                             |
| `postprocessing`         | `tlv_output`              | `data/postprocessing/output_tlv.csv` | Scored TLV table                             |
| `postprocessing.replica` | `table`                   | `EC_OMNICANAL`                   | Replica file `modelo` value                      |
| `postprocessing.replica` | `dir_{s3,athena,onpremise}` | `data/replica/...`             | Replica destinations                             |

## Stages

### 1. Preprocessing — `preprocessing.py`

Reads `config.input_path` (single CSV), then:

- Drops columns with more than `nan_threshold` percent NaN values.
- Normalizes null tokens (`""`, `"null"`, `"None"` → `NaN`).
- Numeric features: filled with `na_fill_numeric`, cast to `float32`.
- Coerces all columns to ML-ready dtypes (bool → int, floats rounded to
  `float_decimals` decimals).
- **Temporal validation split**: rows with `p_codmes == validation_codmes`
  become `df_val` (true out-of-time hold-out).
- The rest is split 70/30 (configurable via `test_size`) into `df_train`
  and `df_test`.

Returns four objects in-memory: `(df_train, df_test, df_val, metadata)`.
`metadata["dropped"]` lists which columns were removed by the NaN filter.

### 2. Training — `training.py`

For each of XGBoost / LightGBM / CatBoost:

1. **Optuna tuning** (`optuna_trials` trials with `MedianPruner`):
   stratified `cv_folds`-fold CV on `df_train`, optimizing mean AUC.
   Each fold uses early stopping internally.
2. **Refit** with the best hyperparameters on `df_train` minus an
   `internal_val_size` stratified slice, which serves as the
   early-stopping eval set for the final fit. `df_test` is **never**
   used during tuning or refit — it's held out for honest reporting.
3. Metrics computed:
   - `auc_train`, `auc_test`, `cv_auc`
   - `decay_pct = (auc_train − auc_test) / auc_train × 100`
   - `best_iteration`: how many rounds early stopping actually used

The **champion** is the model with the highest `auc_test` **whose
`decay_pct` is under `decay_max_pct`**. This biases selection toward
stable models and rejects high-AUC models that overfit.

Outputs to `models/<YYYY-MM-DD_HH-MM-SS>/`:
- `<name>_model.pkl` — joblib-serialized champion
- `<name>_metadata.json` — performance, tuned hyperparameters, library
  versions, timestamp

### 3. Monitoring — `monitoring.py`

Drift and quality metrics on the temporal hold-out `df_val`:

- **PSI** (Population Stability Index) on score deciles. Bin edges
  defined on train scores, applied to validation:
  - `< 0.10` → OK
  - `0.10 – 0.25` → WARN
  - `> 0.25` → ALERT
- **AUC** on `df_val`.
- **Recall** at threshold 0.5.
- **Cumulative recall by decile** (decile 1 = highest score) — measures lift.

Outputs:
- `data/monitoring/monitoring.json`
- `data/monitoring/recall_by_decile.csv`

### 4. Postprocessing — `postprocessing.py`

Converts model probabilities into a business-prioritized ranking:

```
puntuacion_tlv = prob × prob_value × log(monto + 1) × prob_frescura
```

`prob_frescura` is a recency weight per `grp_campecs06m` group
(G1 = 0.066 → G4 = 0.008, default `0.004`). The TLV score is bucketed
into 10 execution groups using `DIST_GE` thresholds (group 1 = highest
priority).

`save_replica` writes a pipe-delimited file with the schema downstream
systems expect:

```
codmes | tipdoc | coddoc | puntuacion | modelo | fec_replica
       | grupo_ejec | score | orden | variable1 | variable2 | variable3
```

The same file is dropped into three destinations (configurable):

- `data/replica/s3/`
- `data/replica/athena/`
- `data/replica/onpremise/`

## Logging

All stages use Python's `logging` module. `main.py` configures the root
logger at `INFO` with format:

```
YYYY-MM-DD HH:MM:SS [LEVEL] module: message
```

Library modules (`training`, `monitoring`) only attach `getLogger(__name__)`
— they don't configure handlers. Re-route to a file or change verbosity
by editing the `logging.basicConfig` call in `main.py`.

## Notes

- The pipeline now uses a real temporal hold-out (`df_val`) for monitoring
  and postprocessing. `df_test` is only used to report `auc_test` and
  validate the decay gate; it never feeds early stopping or hyperparameter
  selection.
- `models/<timestamp>/` accumulates one folder per training run; old runs
  are not auto-cleaned.
- CatBoost writes a `catboost_info/` directory to the working directory
  during training. It's a training-time artifact, safe to delete or
  add to `.gitignore`.
- Optuna tuning is the slowest stage (~5–10 min on a laptop CPU for the
  default 30 trials × 3 folds × 3 models). For faster iteration during
  development, drop `training.optuna_trials` in `config.yaml`.
