# Machine Learning-Based Network Intrusion Detection System

Binary classification (**BENIGN** vs **ATTACK**) of network-flow records from the
CIC-IDS2017 dataset, using Logistic Regression, Decision Tree, Random Forest and SVM.

> **Status: Milestone 2 (dataset loading and audit).** The Parquet loader and a read-only
> data-quality audit exist. There is no preprocessing, sampling, training, evaluation or
> dashboard yet. Results of the audit are only valid for the files you supply.

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

Contents of `data/raw/` are ignored by Git. The provenance of this Parquet variant is
**unknown** (not established); do not assume it matches the data preparation described in the
training report.

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

## Project layout

```
src/nids/       config, data_loader (M2), preprocessing, train, evaluate, predict, utils
tests/          pytest suite (synthetic data only)
data/           raw/ interim/ processed/   (contents not committed)
models/         trained model artifacts     (not committed)
results/        metrics/ figures/ logs/     (not committed)
app.py          Streamlit entry point (Milestone 7)
```

Configuration lives in `src/nids/config.py`. Values taken from the training report
(sample sizes, 70 features) are *targets to verify*, not results.
