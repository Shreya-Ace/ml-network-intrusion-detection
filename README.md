# Machine Learning-Based Network Intrusion Detection System

Binary classification (**BENIGN** vs **ATTACK**) of network-flow records from the
CIC-IDS2017 dataset, using Logistic Regression, Decision Tree, Random Forest and SVM.

> **Status: Milestone 3 (data preparation).** Parquet loader, data audit, and the
> preparation pipeline (exclusions, global deduplication, sampling, split, feature schema)
> exist. There is no model training, evaluation or dashboard yet. Counts and reports are
> only valid for the files you supply.

## Scope

- **Version 1** - offline, reproducible ML pipeline run from the terminal.
- **Version 2** - Streamlit dashboard that reuses Version 1 modules.
- Out of scope: live packet capture, intrusion *prevention*, operational network connection.
- A predicted ATTACK is a model-generated suspicion, not a confirmed security incident.

## Setup

Requires Python 3.11 or newer (developed against 3.11).

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` makes the `nids` package importable from anywhere in the environment
(the code uses a `src/` layout).

## Verify the setup

```bash
python -c "import nids; print(nids.__version__)"   # expect: 0.1.0
pytest
```

## Dataset location

The CIC-IDS2017 files are **not** included. Place the eight Parquet files in:

```
data/raw/cic-ids2017/
```

Contents of `data/raw/` are ignored by Git.

**Provenance (probable, not verified).** The project owner reported an exact filename match
with Version 3 of the Kaggle dataset `dhoogla/cicids2017`. This could not be independently
confirmed and the local download history is unconfirmed. The files hold 77 numerical feature
columns plus `Label` (no IP, port or timestamp columns). Do not assume the data is identical to
the CSV input used by the training report, or that the report's experiment used this variant.

## Dataset audit (Milestone 2)

```bash
python -m nids.data_loader              # prints the report, saves results/metrics/data_audit.json
python -m nids.data_loader --no-save    # print only
python -m nids.data_loader --data-dir some/other/dir
```

Exit codes: `0` no issues, `1` issues found (unrecognised labels, targets not feasible, ...),
`2` dataset missing, unreadable or schema-incompatible. The audit is read-only and describes
the source data *before* any cleaning. It reports missing/infinite values, duplicates (within
and across files), feature-identical rows with conflicting labels, the label mapping, and
provisional counts of eligible records.

## Data preparation (Milestone 3)

```bash
python -m nids.preprocessing                 # full run, writes artefacts, prints the report
python -m nids.preprocessing --no-write      # run and print only
python -m nids.preprocessing --seed 42 --benign 60000 --attack 40000 --test-size 0.2
```

Agreed design, applied in this order:

1. **Invalid values**: rows with NaN/inf in any feature are removed (counted per file).
2. **Conflicting labels**: a feature vector that occurs with both a benign and an attack label has
   *all* its rows excluded (neither copy is kept). Row-level audit: `data/interim/excluded_conflicts.csv`.
   This can make evaluation look more favourable (it removes ambiguous records; PortScan is
   proportionally most affected) - it does not make a model universally more accurate.
3. **Global deduplication**: key = feature values + canonical attack family (never `source_file`
   or row position). Every source file an observation occurs in is kept in
   `data/interim/provenance_manifest.parquet` (with per-file occurrence counts).
4. **Sampling**: 60,000 benign and 40,000 attack observations, seeded, without replacement, from the
   full eligible pools. No per-file quotas, no per-family quotas (natural attack-family mix).
   If a target is infeasible the run stops and reports actual counts.
5. **Split**: stratified 80:20 on the binary target, same seed.
6. **Features**: constant-feature removal is fitted on the **training split only**
   (`VarianceThreshold`); the retained count is reported, never forced to the report's 70.
   Logistic Regression and SVM additionally get standardisation fitted on training data only.

Artefacts (all git-ignored): `data/interim/{row_index,provenance_manifest}.parquet`,
`data/interim/excluded_conflicts.csv`, `data/processed/{train,test}.parquet`,
`data/processed/{dataset_metadata,feature_schema}.json`, `results/logs/data_preparation.txt`.

Provenance columns (`obs_id`, `source_files`, `attack_family`, ...) are stored beside the features
for analysis but are **never model features**: use `split_features_and_metadata` /
`load_prepared_split`.

Limitations: a random split over data whose attack families are aligned with capture days is not
evidence of generalisation to unseen networks; family-specific metrics are unreliable where the test
set has few records (flagged in the report); the 60/40 class ratio is a design choice, not real traffic.

## Project layout

```
src/nids/       config, data_loader (M2), preprocessing (M3), train, evaluate, predict, utils
tests/          pytest suite (synthetic data only)
data/           raw/ interim/ processed/   (contents not committed)
models/         trained model artifacts     (not committed)
results/        metrics/ figures/ logs/     (not committed)
app.py          Streamlit entry point (Milestone 7)
```

Configuration lives in `src/nids/config.py`. Values taken from the training report
(sample sizes, 70 features) are *targets to verify*, not results.
