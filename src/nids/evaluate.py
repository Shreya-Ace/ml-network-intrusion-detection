"""Model evaluation on the held-out test split (Milestone 5).

Each trained pipeline is loaded, verified against its metadata and used for PREDICTION ONLY:

* nothing is refitted (fitting is actively blocked during evaluation), no setting is changed,
  no threshold is tuned, and the model files are never rewritten;
* class decisions are each model's own ``predict`` output;
* ATTACK (1) is the positive class for every metric;
* ROC-AUC uses ``predict_proba[ATTACK]`` for Logistic Regression, Decision Tree and Random
  Forest, and the ``decision_function`` score for the SVM (a signed distance, NOT a probability;
  ``probability=True`` is never enabled).

Results are compared with the figures claimed in the report (``config.REPORT_CLAIMS``); the
claims are only used to compute differences and never influence any metric.

Run::

    python -m nids.evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC

from nids import config
from nids import data_loader as dl
from nids import preprocessing as pp
from nids import train as tr
from nids.data_loader import DatasetError
from nids.utils import ensure_directories, get_logger

logger = get_logger("nids.evaluate")

METRIC_KEYS = ("accuracy", "precision", "recall", "f1")
SCORE_DESCRIPTIONS = {
    "logistic_regression": "predict_proba[ATTACK] (logistic probability)",
    "decision_tree": "predict_proba[ATTACK] (class fraction in the reached leaf)",
    "random_forest": "predict_proba[ATTACK] (mean of the trees' class fractions)",
    "svm": "decision_function (signed distance to the SVM boundary; NOT a probability)",
}


class EvaluationError(Exception):
    """Evaluation of a model could not be completed."""


class EvaluationDataError(DatasetError):
    """The prepared test data is missing, incompatible or inconsistent."""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """TN/FP/FN/TP with ATTACK (1) as the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise EvaluationError("y_true and y_pred have different shapes.")
    values = set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist())
    if not values <= {0, 1}:
        raise EvaluationError(f"Labels must be 0/1; found {sorted(values)}.")
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def compute_roc_auc(y_true, scores) -> float:
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise EvaluationError("ROC-AUC needs both classes in the test labels.")
    if not np.isfinite(scores).all():
        raise EvaluationError("Scores contain NaN/inf.")
    return float(roc_auc_score(y_true, scores))


def classification_metrics(counts: dict[str, int], roc_auc: float | None = None) -> dict:
    """Metrics from confusion counts. A metric whose denominator is zero is None (never invented)."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "specificity": _ratio(tn, tn + fp),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a proportion (an aid for judging small counts)."""
    if n == 0:
        return None, None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def family_recall_table(
    families,
    y_true,
    y_pred,
    train_counts: dict[str, int] | None = None,
    sample_counts: dict[str, int] | None = None,
    min_test_records: int = config.MIN_FAMILY_TEST_RECORDS,
) -> list[dict]:
    """Detection recall for every attack family, keeping the record counts visible.

    Families with fewer than ``min_test_records`` test records are flagged
    ``sufficient=False``: their recall must not be read as a reliable estimate.
    """
    families, y_true, y_pred = np.asarray(families, dtype=object), np.asarray(y_true), np.asarray(y_pred)
    unknown = sorted(set(families.tolist()) - set(dl.FAMILY_NAMES))
    if unknown:
        raise EvaluationError(f"Unknown attack families in the test data: {unknown}")
    train_counts, sample_counts = train_counts or {}, sample_counts or {}
    rows = []
    for family in dl.FAMILY_NAMES:
        if family == config.BENIGN_FAMILY:
            continue
        mask = families == family
        n = int(mask.sum())
        if n and not (y_true[mask] == 1).all():
            raise EvaluationError(f"Family {family!r} contains rows whose true label is not ATTACK.")
        detected = int((y_pred[mask] == 1).sum())
        low, high = wilson_interval(detected, n)
        rows.append(
            {
                "family": family,
                "test_records": n,
                "detected": detected,
                "missed": n - detected,
                "recall": _ratio(detected, n),
                "recall_ci95_low": low,
                "recall_ci95_high": high,
                "train_records": int(train_counts.get(family, 0)),
                "sample_records": int(sample_counts.get(family, 0)),
                "sufficient": n >= min_test_records,
            }
        )
    return rows


def false_positives_by_source_file(source_files, y_true, y_pred) -> dict[str, dict]:
    """False-positive rate among BENIGN test records, by source file.

    A record belongs to every file it occurs in, so the per-file groups OVERLAP and must not
    be summed. Useful to see whether benign flows from particular capture days are flagged more.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    benign = y_true == 0
    files = pd.Series(np.asarray(source_files, dtype=object)[benign]).str.split("|")
    frame = pd.DataFrame({"file": files.to_numpy(), "fp": (y_pred[benign] == 1)}).explode("file")
    grouped = frame.groupby("file").agg(benign_records=("fp", "size"), false_positives=("fp", "sum"))
    return {
        str(name): {
            "benign_records": int(row["benign_records"]),
            "false_positives": int(row["false_positives"]),
            "false_positive_rate": _ratio(int(row["false_positives"]), int(row["benign_records"])),
        }
        for name, row in grouped.iterrows()
    }


# ---------------------------------------------------------------------------
# Scoring and the no-refit guard
# ---------------------------------------------------------------------------
def model_scores(name: str, pipeline, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Scores for ROC-AUC. SVM: decision_function (never a probability, never probability=True)."""
    classifier = pipeline.named_steps["classifier"]
    if list(classifier.classes_) != [0, 1]:
        raise EvaluationError(f"{name}: expected classes [0, 1], found {list(classifier.classes_)}.")
    if isinstance(classifier, SVC):
        if classifier.get_params().get("probability"):
            raise EvaluationError(f"{name}: SVM was trained with probability=True; its scores are not used as probabilities here.")
        return np.asarray(pipeline.decision_function(X), dtype=float), SCORE_DESCRIPTIONS["svm"]
    return np.asarray(pipeline.predict_proba(X))[:, 1], SCORE_DESCRIPTIONS.get(name, "predict_proba[ATTACK]")


@contextmanager
def _forbid_fitting(pipeline):
    """While active, any attempt to fit the pipeline or one of its steps raises EvaluationError."""

    def refuse(*args, **kwargs):
        raise EvaluationError("Fitting is forbidden during evaluation: evaluation must not change a trained model.")

    targets = [pipeline, *[step for _, step in pipeline.steps]]
    patched = []
    try:
        for obj in targets:
            for method in ("fit", "partial_fit", "fit_transform", "fit_predict"):
                if hasattr(type(obj), method):
                    setattr(obj, method, refuse)
                    patched.append((obj, method))
        yield
    finally:
        for obj, method in patched:
            if method in vars(obj):
                delattr(obj, method)


# ---------------------------------------------------------------------------
# Test data and pre-flight verification
# ---------------------------------------------------------------------------
@dataclass
class EvaluationData:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    original_features: list[str]
    class_counts: dict[str, int]
    test_path: Path
    test_sha256: str
    dataset_meta: dict


def load_test_data(
    paths: pp.ArtifactPaths | None = None, loader: Callable = pp.load_prepared_split
) -> EvaluationData:
    """Load and validate the saved TEST split (the training split is never loaded here)."""
    paths = paths or pp.ArtifactPaths.default()
    X, y, meta = loader("test", paths)  # raises DatasetError with instructions if missing
    for required in (paths.feature_schema, paths.metadata):
        if not Path(required).exists():
            raise EvaluationDataError(f"{required} not found. Run: python -m nids.preprocessing")
    schema = json.loads(Path(paths.feature_schema).read_text(encoding="utf-8"))
    dataset_meta = json.loads(Path(paths.metadata).read_text(encoding="utf-8"))
    if "original_features" not in schema:
        raise EvaluationDataError(f"{paths.feature_schema} lacks 'original_features'; incompatible schema file.")
    try:
        counts = tr.validate_training_inputs(X, y, meta, schema["original_features"], dataset_meta, split="test")
    except tr.TrainingDataError as exc:
        raise EvaluationDataError(str(exc)) from exc
    return EvaluationData(
        X=X, y=y, meta=meta, original_features=list(schema["original_features"]), class_counts=counts,
        test_path=Path(paths.test), test_sha256=tr.file_sha256(paths.test), dataset_meta=dataset_meta,
    )


def _json_roundtrip(value):
    return json.loads(json.dumps(value, default=tr._json_default))


def preflight_model(name: str, models_dir: Path, data: EvaluationData, train_path: Path | None) -> tuple[dict, list[str]]:
    """Verify a model's artefact and metadata before evaluating it. Returns (checks, problems)."""
    problems: list[str] = []
    checks: dict = {}
    state = tr.check_artifacts(models_dir).get(name, {"complete": False, "reason": "unknown model"})
    checks["artifact_complete_and_hash_matches_metadata"] = bool(state["complete"])
    if not state["complete"]:
        return checks, [f"artefact not complete: {state.get('reason')}"]
    metadata = json.loads((Path(models_dir) / f"{name}.json").read_text(encoding="utf-8"))

    def check(label: str, ok: bool, message: str) -> None:
        checks[label] = bool(ok)
        if not ok:
            problems.append(message)

    check("test_set_not_used_in_training", metadata["training"].get("test_set_used") is False,
          "metadata does not record test_set_used=False")
    check("settings_unchanged_since_training",
          _json_roundtrip(metadata["classifier"]["report_settings_intended"]) == _json_roundtrip(config.MODEL_SETTINGS[name]),
          "config.MODEL_SETTINGS differ from the settings the model was trained with")
    check("feature_schema_matches_test_data",
          metadata["features"]["original_input_features"] == data.original_features,
          "the model's input features differ from the test data schema")
    train_rows = data.dataset_meta.get("split", {}).get("train", {}).get("rows")
    check("trained_on_current_split_size", metadata["training"]["records"] == train_rows,
          f"model trained on {metadata['training']['records']} records but the prepared training split has {train_rows}")
    if train_path is not None and Path(train_path).exists():
        check("trained_on_current_training_file", metadata["training"]["train_file_sha256"] == tr.file_sha256(train_path),
              "train.parquet changed since this model was trained")
    return checks, problems


def load_model(name: str, models_dir: Path, loader: Callable = joblib.load):
    """Load a pipeline after verifying its file hash and hyperparameters against the metadata."""
    models_dir = Path(models_dir)
    metadata = json.loads((models_dir / f"{name}.json").read_text(encoding="utf-8"))
    if tr.file_sha256(models_dir / f"{name}.joblib") != metadata["artifact"]["model_sha256"]:
        raise EvaluationError(f"{name}: model file does not match the hash in its metadata.")
    pipeline = loader(models_dir / f"{name}.joblib")
    stored = metadata["classifier"]["hyperparameters"]
    actual = _json_roundtrip(pipeline.named_steps["classifier"].get_params())
    if actual != _json_roundtrip(stored):
        raise EvaluationError(f"{name}: loaded model hyperparameters differ from its metadata.")
    return pipeline, metadata


# ---------------------------------------------------------------------------
# Evaluation of one model / all models
# ---------------------------------------------------------------------------
def evaluate_model(name: str, pipeline, data: EvaluationData) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """Predict on the test split and compute every metric. Returns (result, (fpr, tpr))."""
    if list(getattr(pipeline, "feature_names_in_", [])) != data.original_features:
        raise EvaluationError(f"{name}: the pipeline's input feature names/order differ from the test data.")
    state_before = joblib.hash(pipeline)
    with _forbid_fitting(pipeline):
        y_pred = np.asarray(pipeline.predict(data.X))
        scores, score_description = model_scores(name, pipeline, data.X)
    if joblib.hash(pipeline) != state_before:
        raise EvaluationError(f"{name}: the model's state changed during evaluation.")

    y_true = data.y.to_numpy()
    counts = confusion_counts(y_true, y_pred)
    metrics = classification_metrics(counts, compute_roc_auc(y_true, scores))
    split_meta = data.dataset_meta.get("split", {})
    result = {
        "status": "evaluated",
        "positive_class": config.ATTACK_LABEL,
        "test_records": int(len(y_true)),
        "confusion_matrix": counts,
        "metrics": metrics,
        "roc_auc_score_type": score_description,
        "family_recall": family_recall_table(
            data.meta[config.ATTACK_FAMILY_COLUMN].to_numpy(), y_true, y_pred,
            split_meta.get("train", {}).get("family_counts"), data.dataset_meta.get("sample", {}).get("family_counts"),
        ),
        "false_positives_by_source_file": false_positives_by_source_file(data.meta["source_files"].to_numpy(), y_true, y_pred),
        "model_state_unchanged_by_evaluation": True,
    }
    fpr, tpr, _ = roc_curve(y_true, scores)
    return result, (fpr, tpr)


def implied_confusion_from_report(claimed: dict, attack_n: int, benign_n: int) -> dict[str, int]:
    """Approximate confusion counts implied by the report's (rounded) precision and recall."""
    fn = round((1 - claimed["recall"]) * attack_n)
    tp = attack_n - fn
    fp = round(tp * (1 / claimed["precision"] - 1))
    return {"tn": benign_n - fp, "fp": fp, "fn": fn, "tp": tp}


def compare_with_report(models: dict, claims: dict | None = None) -> dict:
    """Differences between measured results and the report's claims (never alters any metric)."""
    claims = config.REPORT_CLAIMS if claims is None else claims
    attack_n, benign_n = claims["test_attack_records"], claims["test_benign_records"]
    per_model: dict = {}
    for name, claimed in claims["models"].items():
        entry: dict = {
            "report": claimed,
            "report_implied_confusion_approx": implied_confusion_from_report(claimed, attack_n, benign_n),
        }
        result = models.get(name)
        if result and result.get("status") == "evaluated":
            measured = {k: result["metrics"][k] for k in METRIC_KEYS}
            entry["measured"] = measured
            entry["difference_percentage_points"] = {
                k: (None if measured[k] is None else (measured[k] - claimed[k]) * 100) for k in METRIC_KEYS
            }
            counts = result["confusion_matrix"]
            implied = entry["report_implied_confusion_approx"]
            entry["measured_confusion"] = counts
            entry["difference_fp_fn"] = {"fp": counts["fp"] - implied["fp"], "fn": counts["fn"] - implied["fn"]}
        per_model[name] = entry
    evaluated = [n for n in claims["models"] if n in per_model and "measured" in per_model[n]]
    rank = lambda getter: sorted(evaluated, key=lambda n: -getter(n))  # noqa: E731
    ours = rank(lambda n: per_model[n]["measured"]["accuracy"])
    theirs = rank(lambda n: claims["models"][n]["accuracy"])
    comparison = {
        "source": claims["source"],
        "note": claims["note"],
        "per_model": per_model,
        "accuracy_ranking_measured": ours,
        "accuracy_ranking_report": theirs,
        "same_accuracy_ranking": ours == theirs,
    }
    rf = models.get("random_forest")
    if rf and rf.get("status") == "evaluated":
        reported = claims["random_forest_confusion"]
        comparison["random_forest_confusion"] = {
            key: {"measured": rf["confusion_matrix"][key], "report": reported[key], "difference": rf["confusion_matrix"][key] - reported[key]}
            for key in ("tn", "fp", "fn", "tp")
        }
        comparison["random_forest_false_positive_rate"] = {
            "measured": rf["metrics"]["false_positive_rate"],
            "report_stated": claims["random_forest_false_positive_rate_stated"],
        }
    comparison["possible_reasons_for_differences_UNTESTED"] = [
        "Dataset provenance is only a probable, unverified match; the report's input data may differ.",
        "Feature sets differ: the report states 70 final features; this pipeline retains the count recorded in feature_schema.json.",
        "This pipeline excludes conflicting-label records and deduplicates globally across files; the report's handling is not described in that detail.",
        "The report does not state how its 100,000-record sample was drawn, nor seeds other than Random Forest's.",
        "The report's percentages are rounded to two decimals.",
    ]
    return comparison


@dataclass
class EvaluationRun:
    results: dict
    curves: dict = field(default_factory=dict)
    all_ok: bool = False


def run_evaluation(
    names: Sequence[str],
    models_dir: Path = config.MODELS_DIR,
    paths: pp.ArtifactPaths | None = None,
    *,
    data_loader: Callable = pp.load_prepared_split,
    model_loader: Callable = joblib.load,
) -> EvaluationRun:
    """Evaluate each requested model on the same saved test split. A failure is recorded
    against that model (never skipped silently) and the remaining models are still evaluated."""
    paths = paths or pp.ArtifactPaths.default()
    models_dir = Path(models_dir)
    data = load_test_data(paths, data_loader)
    hashes_before = {n: tr.file_sha256(models_dir / f"{n}.joblib") for n in names if (models_dir / f"{n}.joblib").exists()}

    models: dict[str, dict] = {}
    curves: dict[str, tuple] = {}
    preflight: dict[str, dict] = {}
    for name in names:
        checks, problems = preflight_model(name, models_dir, data, paths.train)
        preflight[name] = {"checks": checks, "problems": problems}
        if problems:
            models[name] = {"status": "failed", "error": "pre-flight verification failed: " + "; ".join(problems)}
            logger.error("%s: %s", name, models[name]["error"])
            continue
        try:
            pipeline, metadata = load_model(name, models_dir, model_loader)
            result, curve = evaluate_model(name, pipeline, data)
            installed = sklearn.__version__
            result["trained_with_scikit_learn"] = metadata["software"]["scikit_learn"]
            result["scikit_learn_version_matches_training"] = metadata["software"]["scikit_learn"] == installed
            result["model_file_sha256"] = metadata["artifact"]["model_sha256"]
            models[name], curves[name] = result, curve
            logger.info("%s evaluated.", name)
        except Exception as exc:  # recorded against the model, never hidden
            models[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            logger.error("%s: evaluation failed: %s", name, exc)

    unchanged = {n: tr.file_sha256(models_dir / f"{n}.joblib") == h for n, h in hashes_before.items()}
    for name, same in unchanged.items():
        if name in models and models[name]["status"] == "evaluated":
            models[name]["model_file_unchanged_after_evaluation"] = same
            if not same:
                models[name] = {"status": "failed", "error": "model file changed during evaluation"}

    split_meta = data.dataset_meta.get("split", {})
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": tr._software(),
        "positive_class": config.ATTACK_LABEL,
        "design": "Single evaluation pass; models, settings and thresholds were not changed using test results.",
        "test_set": {
            "path": str(data.test_path),
            "sha256": data.test_sha256,
            "records": int(len(data.y)),
            "class_counts": data.class_counts,
            "family_counts": split_meta.get("test", {}).get("family_counts"),
        },
        "preflight": preflight,
        "models": models,
        "comparison_with_report": compare_with_report(models),
        "limitations": [
            "A random split of data whose attack families are aligned with capture days: strong scores do not show generalisation to unseen networks or periods.",
            "Conflicting-label records were excluded before the split, which removes ambiguous (potentially hard) records and can make results look more favourable.",
            "The 60/40 class ratio is a design choice, not real traffic; precision and false-positive counts would differ at realistic attack rates.",
            f"Families with fewer than {config.MIN_FAMILY_TEST_RECORDS} test records have unreliable recall estimates; they are flagged, not hidden.",
            "ROC-AUC for the SVM uses decision scores, which are not probabilities. Tree-model scores are class fractions, not calibrated probabilities.",
            "Dataset provenance is a probable, unverified match; comparison with the report does not imply the same data.",
        ],
    }
    all_ok = all(models[n]["status"] == "evaluated" for n in names)
    return EvaluationRun(results, curves, all_ok)


# ---------------------------------------------------------------------------
# Outputs: CSV, figures, text report
# ---------------------------------------------------------------------------
def comparison_frame(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results["models"].items():
        if r["status"] != "evaluated":
            rows.append({"model": name, "status": "FAILED", "error": r["error"]})
            continue
        row = {"model": name, "status": "evaluated", **r["metrics"], **r["confusion_matrix"]}
        rows.append(row)
    return pd.DataFrame(rows)


def family_frame(results: dict) -> pd.DataFrame:
    rows = [
        {"model": name, **fam}
        for name, r in results["models"].items() if r["status"] == "evaluated" for fam in r["family_recall"]
    ]
    return pd.DataFrame(rows)


def report_frame(results: dict) -> pd.DataFrame:
    rows = []
    for name, entry in results["comparison_with_report"]["per_model"].items():
        for key in METRIC_KEYS:
            measured = entry.get("measured", {}).get(key)
            rows.append(
                {
                    "model": name, "metric": key, "report": entry["report"][key], "measured": measured,
                    "difference_percentage_points": None if measured is None else (measured - entry["report"][key]) * 100,
                }
            )
    return pd.DataFrame(rows)


def make_figures(run: EvaluationRun, figures_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = Path(figures_dir)
    ensure_directories((figures_dir,))
    evaluated = {n: r for n, r in run.results["models"].items() if r["status"] == "evaluated"}
    written: dict[str, Path] = {}
    if not evaluated:
        return written
    names = list(evaluated)
    pretty = {n: n.replace("_", " ").title() if n != "svm" else "SVM" for n in names}

    # 1. metric comparison + error counts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    x, width = np.arange(len(names)), 0.2
    all_values = [evaluated[n]["metrics"][k] for n in names for k in METRIC_KEYS if evaluated[n]["metrics"][k] is not None]
    for i, key in enumerate(METRIC_KEYS):
        values = [evaluated[n]["metrics"][key] or 0 for n in names]
        bars = ax1.bar(x + (i - 1.5) * width, values, width, label=key.upper() if key == "f1" else key.title())
        for bar, v in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v * 100:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax1.set_xticks(x, [pretty[n] for n in names])
    ax1.set_ylim(max(0.0, min(all_values) - 0.04), 1.0 + 0.02)
    ax1.set_ylabel("score (y-axis truncated)")
    ax1.set_title("Test-set metrics (ATTACK = positive class)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)  # below the bars: never hides data
    for i, (key, label) in enumerate((("fp", "False positives (benign flagged)"), ("fn", "False negatives (attacks missed)"))):
        values = [evaluated[n]["confusion_matrix"][key] for n in names]
        bars = ax2.bar(x + (i - 0.5) * 0.35, values, 0.35, label=label)
        ax2.bar_label(bars, fontsize=8)
    ax2.set_xticks(x, [pretty[n] for n in names])
    ax2.set_title("Error counts on the test set")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)
    fig.tight_layout()
    written["metrics_comparison"] = figures_dir / "metrics_comparison.png"
    fig.savefig(written["metrics_comparison"], dpi=150)
    plt.close(fig)

    # 2. confusion matrices (individual + combined)
    def draw_confusion(ax, counts, title):
        matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        row_totals = matrix.sum(axis=1, keepdims=True)
        labels = np.array([[f"{matrix[i, j]:,}\n({matrix[i, j] / row_totals[i, 0]:.2%} of row)" if row_totals[i, 0] else f"{matrix[i, j]:,}" for j in range(2)] for i in range(2)])
        sns.heatmap(matrix, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred BENIGN", "Pred ATTACK"], yticklabels=["True BENIGN", "True ATTACK"])
        ax.set_title(title)

    for n in names:
        fig, ax = plt.subplots(figsize=(5, 4))
        draw_confusion(ax, evaluated[n]["confusion_matrix"], f"{pretty[n]} (test set)")
        fig.tight_layout()
        written[f"confusion_matrix_{n}"] = figures_dir / f"confusion_matrix_{n}.png"
        fig.savefig(written[f"confusion_matrix_{n}"], dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4))
    for ax, n in zip(np.atleast_1d(axes), names):
        draw_confusion(ax, evaluated[n]["confusion_matrix"], pretty[n])
    fig.tight_layout()
    written["confusion_matrices_all"] = figures_dir / "confusion_matrices_all.png"
    fig.savefig(written["confusion_matrices_all"], dpi=150)
    plt.close(fig)

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n in names:
        if n in run.curves:
            fpr, tpr = run.curves[n]
            ax.plot(fpr, tpr, label=f"{pretty[n]} (AUC {evaluated[n]['metrics']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (recall)")
    ax.set_title("ROC curves on the test set\n(SVM: decision scores, not probabilities)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    written["roc_curves"] = figures_dir / "roc_curves.png"
    fig.savefig(written["roc_curves"], dpi=150)
    plt.close(fig)

    # 4. per-family recall (insufficient families marked)
    family = family_frame(run.results)
    families = [f for f in dl.FAMILY_NAMES if f != config.BENIGN_FAMILY]
    first = family[family["model"] == names[0]].set_index("family")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(families) + 1.5))
    height = 0.8 / len(names)
    for i, n in enumerate(names):
        sub = family[family["model"] == n].set_index("family").reindex(families)
        bars = ax.barh(np.arange(len(families)) + i * height, sub["recall"].fillna(0).to_numpy(), height, label=pretty[n])
        for bar, family_name in zip(bars, families):
            if not first.loc[family_name, "sufficient"]:  # hatched and faded: not a reliable estimate
                bar.set_hatch("//")
                bar.set_alpha(0.45)
    ax.set_yticks(np.arange(len(families)) + height * (len(names) - 1) / 2,
                  [f"{f}  (n={int(first.loc[f, 'test_records'])}{'' if first.loc[f, 'sufficient'] else ', INSUFFICIENT'})" for f in families])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(f"detection recall (families with < {config.MIN_FAMILY_TEST_RECORDS} test records are unreliable)")
    ax.set_title("Recall by original attack family (test set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(names), frameon=False)
    fig.tight_layout()
    written["family_recall"] = figures_dir / "family_recall.png"
    fig.savefig(written["family_recall"], dpi=150)
    plt.close(fig)

    # 5. measured minus report
    rep = report_frame(run.results).dropna(subset=["measured"])
    if not rep.empty:
        pivot = rep.pivot(index="model", columns="metric", values="difference_percentage_points")[list(METRIC_KEYS)]
        limit = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.heatmap(pivot, annot=True, fmt="+.2f", cmap="RdBu", center=0, vmin=-limit, vmax=limit, ax=ax,
                    cbar_kws={"label": "percentage points"})
        ax.set_title("Measured minus report (percentage points)")
        fig.tight_layout()
        written["report_vs_measured"] = figures_dir / "report_vs_measured.png"
        fig.savefig(written["report_vs_measured"], dpi=150)
        plt.close(fig)
    return written


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def format_evaluation_report(results: dict) -> str:
    lines: list[str] = []
    add = lines.append
    bar = "=" * 78

    def section(title: str) -> None:
        add("")
        add(bar)
        add(title)
        add(bar)

    add(bar)
    add("MODEL EVALUATION REPORT (Milestone 5) - held-out test set, ATTACK = positive class")
    add(bar)
    ts = results["test_set"]
    add(f"Test records: {ts['records']:,} (BENIGN {ts['class_counts']['BENIGN']:,}, ATTACK {ts['class_counts']['ATTACK']:,})   SHA-256: {ts['sha256']}")
    add(results["design"])
    failed = [n for n, r in results["models"].items() if r["status"] != "evaluated"]
    if failed:
        add("")
        add("!! EVALUATION FAILED FOR: " + ", ".join(failed))
        for n in failed:
            add(f"   {n}: {results['models'][n]['error']}")

    section("1. METRICS")
    add(f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>9}{'ROC-AUC':>10}{'FPR':>9}")
    for name, r in results["models"].items():
        if r["status"] != "evaluated":
            add(f"{name:<22}FAILED")
            continue
        m = r["metrics"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        add(f"{name:<22}{_pct(m['accuracy']):>10}{_pct(m['precision']):>11}{_pct(m['recall']):>9}{_pct(m['f1']):>9}{auc:>10}{_pct(m['false_positive_rate']):>9}")
    add("ROC-AUC scores: " + "; ".join(f"{n}: {r['roc_auc_score_type']}" for n, r in results["models"].items() if r["status"] == "evaluated"))

    section("2. CONFUSION MATRICES (rows = true class, ATTACK positive)")
    add(f"{'model':<22}{'TN':>8}{'FP':>8}{'FN':>8}{'TP':>8}   benign flagged / attacks missed")
    for name, r in results["models"].items():
        if r["status"] == "evaluated":
            c = r["confusion_matrix"]
            add(f"{name:<22}{c['tn']:>8,}{c['fp']:>8,}{c['fn']:>8,}{c['tp']:>8,}   {c['fp']:,} of {c['tn'] + c['fp']:,} benign / {c['fn']:,} of {c['fn'] + c['tp']:,} attacks")

    section(f"3. DETECTION RECALL BY ATTACK FAMILY (flag: fewer than {config.MIN_FAMILY_TEST_RECORDS} test records)")
    evaluated = [n for n, r in results["models"].items() if r["status"] == "evaluated"]
    if evaluated:
        add(f"{'family':<27}{'test n':>7}{'train n':>9}  " + "  ".join(f"{n[:11]:>11}" for n in evaluated) + "   note")
        by_model = {n: {f['family']: f for f in results['models'][n]['family_recall']} for n in evaluated}
        for family in [f for f in dl.FAMILY_NAMES if f != config.BENIGN_FAMILY]:
            ref = by_model[evaluated[0]][family]
            cells = "  ".join(f"{'n/a' if by_model[n][family]['recall'] is None else format(by_model[n][family]['recall'], '.2%'):>11}" for n in evaluated)
            note = "" if ref["sufficient"] else "INSUFFICIENT - do not rely on this recall"
            add(f"{family:<27}{ref['test_records']:>7}{ref['train_records']:>9}  {cells}   {note}")
        add("Missed attacks by family (false negatives) are in evaluation_metrics.json / family_recall.csv with 95% Wilson intervals.")

    section("4. FALSE-POSITIVE RATE OF BENIGN TEST RECORDS BY SOURCE FILE (groups overlap; do not sum)")
    for name in evaluated:
        add(f"[{name}]")
        for file_name, f in results["models"][name]["false_positives_by_source_file"].items():
            add(f"    {file_name:<45}{f['false_positives']:>6,} of {f['benign_records']:>7,}  ({_pct(f['false_positive_rate'])})")

    cmp_ = results["comparison_with_report"]
    section("5. COMPARISON WITH THE REPORT (claims are never used to compute metrics)")
    add(f"Source: {cmp_['source']}")
    add(f"{'model':<22}{'metric':<11}{'report':>9}{'measured':>10}{'diff (pp)':>11}")
    for name, entry in cmp_["per_model"].items():
        for key in METRIC_KEYS:
            measured = entry.get("measured", {}).get(key)
            diff = entry.get("difference_percentage_points", {}).get(key)
            add(f"{name:<22}{key:<11}{_pct(entry['report'][key]):>9}{_pct(measured):>10}{('n/a' if diff is None else format(diff, '+.2f')):>11}")
    add(f"Accuracy ranking - measured: {cmp_['accuracy_ranking_measured']}")
    add(f"Accuracy ranking - report  : {cmp_['accuracy_ranking_report']}   (same order: {cmp_['same_accuracy_ranking']})")
    if "random_forest_confusion" in cmp_:
        rf = cmp_["random_forest_confusion"]
        add("Random Forest confusion matrix vs report: " + ", ".join(f"{k.upper()} {v['measured']:,} (report {v['report']:,}, diff {v['difference']:+,})" for k, v in rf.items()))
        fpr = cmp_["random_forest_false_positive_rate"]
        add(f"Random Forest false-positive rate: measured {_pct(fpr['measured'])}, report states about {_pct(fpr['report_stated'])}")
    add("Possible reasons for differences (UNTESTED):")
    for reason in cmp_["possible_reasons_for_differences_UNTESTED"]:
        add(f"  - {reason}")

    section("6. INTEGRITY")
    for name, r in results["models"].items():
        if r["status"] == "evaluated":
            add(f"{name:<22}state unchanged by evaluation: {r['model_state_unchanged_by_evaluation']}; model file unchanged: "
                f"{r.get('model_file_unchanged_after_evaluation')}; trained with scikit-learn {r['trained_with_scikit_learn']} "
                f"(matches installed: {r['scikit_learn_version_matches_training']})")

    section("LIMITATIONS")
    for item in results["limitations"]:
        add(f"* {item}")
    if results.get("artifacts"):
        section("ARTEFACTS")
        width = max(len(name) for name in results["artifacts"]) + 2
        for name, path in results["artifacts"].items():
            add(f"{name:<{width}}{path}")
    return "\n".join(lines).encode("ascii", "backslashreplace").decode("ascii")


def write_outputs(run: EvaluationRun, results_dir: Path = config.RESULTS_DIR) -> dict[str, Path]:
    """Write machine-readable results, CSVs, figures and the text report (all git-ignored)."""
    results_dir = Path(results_dir)
    metrics_dir, figures_dir, logs_dir = results_dir / "metrics", results_dir / "figures", results_dir / "logs"
    ensure_directories((metrics_dir, figures_dir, logs_dir))
    written = {
        "evaluation_metrics_json": metrics_dir / "evaluation_metrics.json",
        "model_comparison_csv": metrics_dir / "model_comparison.csv",
        "family_recall_csv": metrics_dir / "family_recall.csv",
        "report_comparison_csv": metrics_dir / "report_comparison.csv",
        "evaluation_report_txt": logs_dir / "evaluation_report.txt",
    }
    comparison_frame(run.results).to_csv(written["model_comparison_csv"], index=False)
    family_frame(run.results).to_csv(written["family_recall_csv"], index=False)
    report_frame(run.results).to_csv(written["report_comparison_csv"], index=False)
    written.update({f"figure_{k}": v for k, v in make_figures(run, figures_dir).items()})
    run.results["artifacts"] = {k: str(v) for k, v in written.items()}
    written["evaluation_metrics_json"].write_text(json.dumps(run.results, indent=2, default=tr._json_default), encoding="utf-8")
    written["evaluation_report_txt"].write_text(format_evaluation_report(run.results) + "\n", encoding="utf-8")
    return written


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        prog="python -m nids.evaluate",
        description="Evaluate the trained models on the saved test split (prediction only; nothing is refitted).",
    )
    parser.add_argument("--models", nargs="+", metavar="MODEL", help=f"default: all of {', '.join(config.MODEL_NAMES)}")
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR, help="metrics/, figures/ and logs/ are created beneath it")
    args = parser.parse_args(argv)

    try:
        names = list(config.MODEL_NAMES) if args.models is None else tr.parse_model_names(args.models)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    paths = pp.ArtifactPaths.under(config.INTERIM_DATA_DIR, args.processed_dir, config.LOGS_DIR)
    try:
        run = run_evaluation(names, args.models_dir, paths)
        written = write_outputs(run, args.results_dir)
    except DatasetError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(format_evaluation_report(run.results))
    return 0 if run.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
