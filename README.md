# Machine Learning-Based Network Intrusion Detection System

Binary classification (**BENIGN** vs **ATTACK**) of network-flow records from the
CIC-IDS2017 dataset, using Logistic Regression, Decision Tree, Random Forest and SVM.

> **Status: Milestone 5 (evaluation).** Parquet loader, data audit, preparation pipeline, model
> training and evaluation code exist. There is no prediction CLI or dashboard yet. Counts, metrics
> and reports are only valid for the files you supply.

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

## Model training (Milestone 4)

Trains on the **training split only**; the test split is never loaded and no metrics are computed
(evaluation is Milestone 5).

```bash
python -m nids.train --models logistic_regression decision_tree random_forest   # fast models
python -m nids.train --models svm                                               # separate, slow invocation
python -m nids.train                          # no --models: the three fast models only; never SVM
```

Valid model names: `logistic_regression`, `decision_tree`, `random_forest`, `svm`. Unknown names are
rejected with the valid list. SVM is never trained unless named.

Each model is a complete scikit-learn Pipeline:
`constant-feature selector -> [StandardScaler for logistic_regression and svm] -> classifier`.
It accepts the original input features (in schema order); the selector (fitted on training data only)
may retain fewer. Settings (from the report): Logistic Regression `C=1.0`, L2, `max_iter=1000`,
solver `lbfgs`; Decision Tree gini, `max_depth=20`; Random Forest 100 trees, gini; SVM RBF, `C=1.0`,
`gamma="scale"`, `probability=False`. `random_state=42` everywhere it is accepted - but it has **no
effect** for Logistic Regression with `lbfgs` or for SVC with `probability=False` (recorded in the
metadata). In scikit-learn >= 1.8 `penalty` is deprecated; L2 is expressed as `l1_ratio=0.0`
(numerically identical). With `probability=False` the SVM exposes a decision *score*, not a probability.

Outputs (git-ignored): `models/<name>.joblib` and `models/<name>.json` (hyperparameters, fitted
facts, input/retained feature names, training counts, label mapping, versions, training-file SHA-256,
fit time, warnings, save-and-reload result). A model is written as `.partial`, reloaded and compared
with the in-memory model on a fixed slice of the training data, and only then moved into place; the
JSON (which holds the model file's SHA-256) is moved last, so an interrupted run leaves no completed
artefact. No unit test trains a full-size model.

## Model evaluation (Milestone 5)

```bash
python -m nids.evaluate                       # all four models, one pass over the saved test split
python -m nids.evaluate --models random_forest svm
```

Prediction only: models are loaded after their file hash and hyperparameters are checked against the
metadata, fitting is actively blocked during evaluation, and no setting or threshold is tuned on test
results. ATTACK is the positive class. Reported: accuracy, precision, recall, F1, ROC-AUC,
false-positive rate and confusion counts; ROC-AUC uses `predict_proba[ATTACK]` for Logistic
Regression, Decision Tree and Random Forest and the `decision_function` score for the SVM (a signed
distance, not a probability; `probability=True` is never enabled). Detection recall is also reported
per original attack family, with the training/sample/test record counts and a 95% Wilson interval;
families with fewer than 30 test records are flagged INSUFFICIENT.

Results are compared with the figures claimed in the report (`config.REPORT_CLAIMS`, transcribed from
its Table 5 and Figure 10). The claims only produce difference columns; they never influence a metric.
A model whose evaluation fails is reported as FAILED (never skipped silently) and the exit code is 1.

Outputs (git-ignored): `results/metrics/{evaluation_metrics.json,model_comparison.csv,family_recall.csv,
report_comparison.csv}`, `results/logs/evaluation_report.txt`, and `results/figures/` (metric comparison,
confusion matrix per model plus a combined one, ROC curves, per-family recall, measured-vs-report).

## Interpreting the evaluation results and their limitations

**Where the numbers come from.** Every figure (accuracy, precision, recall, F1, ROC-AUC, confusion
counts, false-positive/negative counts, per-family recall, comparison with the report) is generated by
`python -m nids.evaluate` from the saved models and the saved test split. Nothing is hardcoded in the
code or copied into the documentation: read `results/logs/evaluation_report.txt` and
`results/metrics/` (git-ignored, regenerated on demand). Model metadata files never contain test
metrics.

**How to describe them.** A high score, such as the Random Forest's, is performance on *this*
random-split CIC-IDS2017 experiment. It is not evidence of generalisation to unseen networks or
production traffic.

**Limitations (apply to every reported result)**

- The experiment uses a single public benchmark dataset (a probable, unverified Parquet variant of
  CIC-IDS2017), not real operational traffic.
- The primary evaluation uses one random train/test split.
- Attack families are aligned with capture days/source files (a capture-day/attack-family confound), so
  models may partly learn dataset-specific patterns; a random split cannot detect this.
- Conflicting-label observations (identical features seen with both labels) were excluded from the
  primary experiment. This removes ambiguous, potentially hard records and can make results look more
  favourable; it does not make a model more accurate in general.
- Rare attack families have too few test records for reliable family-level conclusions (flagged
  INSUFFICIENT below 30 test records; some families have 0 or 1).
- The 60/40 benign/attack ratio is a design choice, not real traffic, so precision and false-positive
  counts would differ at realistic attack rates.
- The results do not establish production-network performance.
- Scores: the SVM ROC-AUC uses decision scores (not probabilities); tree-model scores are class
  fractions, not calibrated probabilities.
- The saved SVM's metadata records scikit-learn's FutureWarning that `SVC(probability=...)` is
  deprecated (removal planned for 1.11). It is documented and left as is; the model uses
  `probability=False`.

**Deferred work.** Supplementary robustness experiments (effect of the conflicting-label exclusion,
generalisation across source files or capture days, attack-family performance under more appropriate
test designs) are deferred until after the Version 2 dashboard. Each needs a proposed evaluation design
first, and none may use the current test results to tune models or thresholds.

## Project layout

```
src/nids/       config, data_loader (M2), preprocessing (M3), train (M4), evaluate (M5), predict, utils
tests/          pytest suite (synthetic data only)
data/           raw/ interim/ processed/   (contents not committed)
models/         trained model artifacts     (not committed)
results/        metrics/ figures/ logs/     (not committed)
app.py          Streamlit entry point (Milestone 7)
```

Configuration lives in `src/nids/config.py`. Values taken from the training report
(sample sizes, 70 features) are *targets to verify*, not results.
