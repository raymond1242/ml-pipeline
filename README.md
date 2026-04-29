# CU Venta ML Pipeline

End-to-end ML pipeline for ranking customers by Total Lifetime Value (TLV)
in cross-/up-selling campaigns. Trains a binary classifier
(XGBoost / LightGBM / CatBoost), selects a champion based on test AUC
and a generalization-decay constraint, and produces a pipe-delimited
replica file ready for S3, Athena, and on-premise consumers.

## Pipeline overview

```
train_data/
    │
    ▼
[1. preprocessing]  ─►  preprocess_data/
                            ├── features/   target + model features
                            └── business/   identifiers + business cols
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

## Project structure

```
ml-pipeline/
├── main.py              # orchestrator
├── preprocessing.py     # data cleanup + train/test split
├── training.py          # 3-model bake-off + champion selection
├── monitoring.py        # PSI + recall-by-decile + drift metrics
├── postprocessing.py    # TLV score + execution groups + replica file
├── requirements.in      # source dependencies
├── requirements.txt     # pinned (generated with pip-compile)
└── train_data/          # input CSVs (e.g. p1_extrac.csv, p2_extrac.csv, ...)
```

The following directories are created by the pipeline:

```
preprocess_data/         # stage 1 output
models/<timestamp>/      # stage 2 output
data/monitoring/         # stage 3 output
data/replica/            # stage 4 output
```

## Setup

### System requirements

- Python 3.12+ (tested on 3.14)
- On macOS, LightGBM needs OpenMP at runtime:
  ```bash
  brew install libomp
  ```

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

Runs all four stages in order. The orchestrator wires file paths between
stages from each module's exported constants, so no CLI flags are needed
for the default flow.

### Per stage

Each module is runnable on its own for iteration:

```bash
python preprocessing.py    # writes preprocess_data/
python training.py         # reads preprocess_data/features/, writes models/<timestamp>/
```

## Stages

### 1. Preprocessing — `preprocessing.py`

Reads every CSV in `train_data/`, concatenates them, then:

- Normalizes null tokens (`""`, `"null"`, `"None"` → `NaN`).
- Numeric columns: filled with sentinel `-9999999`, cast to `float32`.
- Categorical column `ent_1erlntcrallsfm01`: anything outside the keep-list
  collapses to `"OTRO"`, then one-hot encoded.
- Coerces all columns to ML-ready dtypes (bool → int, floats rounded to
  4 decimals) so the saved CSVs feed straight into XGB/LGBM/CatBoost.
- Train/test split (33% test, seed 123).

Outputs:
- `preprocess_data/features/{train,test}_extrac.csv` — target + features
  (training input).
- `preprocess_data/business/{train,test}_extrac.csv` — identifiers + business
  columns (postprocessing input).

### 2. Training — `training.py`

Trains three classifiers with default hyperparameters on the same train set:

- XGBoost
- LightGBM
- CatBoost

For each model: AUC train, AUC test, and **decay**:

```
decay_pct = (auc_train − auc_test) / auc_train × 100
```

The **champion** is the model with the highest test AUC **whose decay is
under 10%**. This biases selection toward stable models and rejects
high-AUC models that overfit.

Outputs to `models/<YYYY-MM-DD_HH-MM-SS>/`:
- `<name>_model.pkl` — joblib-serialized champion
- `<name>_metadata.json` — performance, hyperparameters, library versions,
  timestamp (for reproducibility / audits)

### 3. Monitoring — `monitoring.py`

Drift and quality metrics on the held-out test set:

- **PSI** (Population Stability Index) on score deciles. Bin edges defined
  on train, applied to validation:
  - `< 0.10` → OK
  - `0.10 – 0.25` → WARN
  - `> 0.25` → ALERT
- **AUC** on validation.
- **Recall** at threshold 0.5.
- **Cumulative recall by decile** (decil 1 = highest score) — measures lift.

Outputs:
- `data/monitoring/monitoring.json`
- `data/monitoring/recall_by_decile.csv`

### 4. Postprocessing — `postprocessing.py`

Converts model probabilities into a business-prioritized ranking:

```
puntuacion_tlv = prob × prob_value_contact × log(monto + 1) × prob_frescura
```

`prob_frescura` is a recency weight per `grp_campecs06m` group
(G1 = 0.066 → G4 = 0.008). The TLV puntuation is bucketed into 10
execution groups using `DIST_GE` thresholds (group 1 = highest priority).

`save_replica` writes a pipe-delimited file with the schema downstream
systems expect:

```
codmes | tipdoc | coddoc | puntuacion | modelo | fec_replica
       | grupo_ejec | score | orden | variable1 | variable2 | variable3
```

The same file is dropped into three destinations:

- `data/replica/s3/`
- `data/replica/athena/`
- `data/replica/onpremise/`

## Configuration

Per-module constants worth tuning:

| Module               | Constant           | Default        | Meaning                                       |
|----------------------|--------------------|----------------|-----------------------------------------------|
| `preprocessing.py`   | `TEST_SIZE`        | `0.33`         | Test split fraction                           |
| `preprocessing.py`   | `RANDOM_STATE`     | `123`          | Split seed                                    |
| `preprocessing.py`   | `NA_FILL_NUMERIC`  | `-9999999`     | Sentinel for missing numeric values           |
| `training.py`        | `DECAY_MAX_PCT`    | `10.0`         | Max train→test decay for champion eligibility |
| `training.py`        | `RANDOM_STATE`     | `42`           | Model seed                                    |
| `monitoring.py`      | PSI thresholds     | `0.10 / 0.25`  | OK / WARN / ALERT cutoffs                     |
| `postprocessing.py`  | `DIST_GE`          | 10 percentiles | Execution-group thresholds                    |

## Notes

- The test set currently doubles as the validation set for monitoring and
  postprocessing. For production, add a temporal holdout
  (e.g. on `p_codmes`) and route it through `main.py`.
- `models/<timestamp>/` accumulates one folder per training run; old runs
  are not auto-cleaned.
- CatBoost writes a `catboost_info/` directory to the working directory
  during training. It's a training-time artifact, safe to delete or
  add to `.gitignore`.
