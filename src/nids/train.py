"""Model training and persistence (Milestone 4).

Trains complete scikit-learn Pipelines on the Milestone 3 TRAINING split only::

    original input features (schema order)
      -> training-fitted constant-feature selector
      -> [StandardScaler]                       (Logistic Regression and SVM only)
      -> classifier

Nothing here touches the test split and no performance metrics are computed: evaluation
is Milestone 5. The save-and-reload check compares predictions on a slice of the TRAINING
data and verifies persistence only.

Artefact safety: a model is written to ``<name>.joblib.partial``, reloaded and checked, and
only then moved to its final name; the metadata JSON (the completion marker, which carries
the model file's SHA-256) is moved last. A failed or interrupted run leaves no final file.

Commands::

    python -m nids.train --models logistic_regression decision_tree random_forest
    python -m nids.train --models svm          # separate, expensive invocation
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from nids import config
from nids import preprocessing as pp
from nids.data_loader import DatasetError
from nids.utils import ensure_directories, get_logger

logger = get_logger("nids.train")


class TrainingDataError(DatasetError):
    """The prepared training data is missing, incompatible or inconsistent."""


class TrainingError(Exception):
    """Training did not complete; no final model artefact was written."""


class ModelValidationError(TrainingError):
    """The save-and-reload check failed; no final model artefact was written."""


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def parse_model_names(requested: Sequence[str] | None) -> list[str]:
    """Validate requested model names. With none given, only the three fast models are used."""
    if requested is None:
        return list(config.INITIAL_MODELS)
    if len(requested) == 0:
        raise ValueError(f"No model names given. Valid names: {list(config.MODEL_NAMES)}")
    unknown = [name for name in requested if name not in config.MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}. Valid names: {list(config.MODEL_NAMES)}")
    return list(dict.fromkeys(requested))  # drop duplicates, keep order


def _require_known(name: str) -> None:
    if name not in config.MODEL_NAMES:
        raise ValueError(f"Unknown model name: {name!r}. Valid names: {list(config.MODEL_NAMES)}")


def _logistic_regression_kwargs(settings: dict) -> tuple[dict, str]:
    """Map the report's ``penalty='l2'`` onto the installed scikit-learn's parameters.

    scikit-learn 1.8 deprecated ``penalty`` (removal planned for 1.10) in favour of
    ``l1_ratio`` (0.0 = pure L2). The two are numerically identical; the new form avoids the
    FutureWarning. Returns the kwargs and a note recorded in the model metadata.
    """
    kwargs = dict(settings)
    penalty = kwargs.pop("penalty")
    if penalty != "l2":
        raise ValueError(f"Only L2 regularisation is configured; got {penalty!r}.")
    parameter = inspect.signature(LogisticRegression).parameters.get("penalty")
    if parameter is not None and parameter.default != "deprecated":
        kwargs["penalty"] = "l2"
        return kwargs, "penalty='l2' passed directly (supported without deprecation in this scikit-learn version)"
    kwargs["l1_ratio"] = 0.0
    return kwargs, (
        "report setting penalty='l2' expressed as l1_ratio=0.0 (pure L2) because 'penalty' is "
        "deprecated in this scikit-learn version; the two are numerically identical"
    )


def build_classifier(name: str):
    """The classifier with the report's settings (no tuning, class weights or resampling)."""
    _require_known(name)
    settings = dict(config.MODEL_SETTINGS[name])
    if name == "logistic_regression":
        return LogisticRegression(**_logistic_regression_kwargs(settings)[0])
    if name == "decision_tree":
        return DecisionTreeClassifier(**settings)
    if name == "random_forest":
        return RandomForestClassifier(**settings)
    return SVC(**settings)


def build_pipeline(name: str) -> Pipeline:
    """Constant-feature selector [+ scaler] + classifier, reusing the Milestone 3 preprocessor."""
    _require_known(name)
    preprocessor = pp.preprocessor_for_model(name)
    return Pipeline([*preprocessor.steps, ("classifier", build_classifier(name))])


# What the seed does (per scikit-learn documentation); recorded so the metadata never
# implies a seed influences a component it does not.
SEED_EFFECT = {
    "logistic_regression": (False, "no effect: used only by the 'sag', 'saga' and 'liblinear' solvers; this model uses 'lbfgs'"),
    "decision_tree": (True, "affects the random feature permutation at each split (tie-breaking between equally good splits)"),
    "random_forest": (True, "controls bootstrap sampling of rows and the random feature subset considered at each split"),
    "svm": (False, "no effect: used only to shuffle data for probability estimates, which are disabled (probability=False)"),
}


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class TrainingData:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    original_features: list[str]
    schema_retained_features: list[str]
    class_counts: dict[str, int]
    train_path: Path
    train_sha256: str
    schema_sha256: str


def validate_training_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    original_features: Sequence[str],
    dataset_meta: dict | None = None,
) -> dict[str, int]:
    """Explicit checks on the training data; returns class counts keyed by label name."""
    expected = list(original_features)
    if len(X) == 0:
        raise TrainingDataError("The training split is empty.")
    if list(X.columns) != expected:
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]
        raise TrainingDataError(
            f"Training columns do not match the feature schema (missing {missing}, unexpected {extra}; "
            "names and order must be identical)."
        )
    try:
        values = X.to_numpy(dtype="float64")
    except (ValueError, TypeError) as exc:
        raise TrainingDataError(f"Training features are not all numeric: {exc}") from exc
    if not np.isfinite(values).all():
        raise TrainingDataError(
            f"Training features contain {int((~np.isfinite(values)).sum()):,} NaN/inf value(s); "
            "invalid rows should have been removed by preprocessing."
        )
    if y.isna().any():
        raise TrainingDataError("The target contains missing values.")
    invalid = sorted(set(y.unique()) - {0, 1})
    if invalid:
        raise TrainingDataError(f"Invalid target labels {invalid}; expected only 0 (BENIGN) and 1 (ATTACK).")
    n_benign, n_attack = int((y == 0).sum()), int((y == 1).sum())
    if n_benign == 0 or n_attack == 0:
        raise TrainingDataError(f"Both classes are required; got BENIGN={n_benign}, ATTACK={n_attack}.")
    if len(y) != len(X):
        raise TrainingDataError("Features and target have different lengths.")
    if config.ATTACK_FAMILY_COLUMN in meta.columns:
        consistent = (meta[config.ATTACK_FAMILY_COLUMN].to_numpy() != config.BENIGN_FAMILY).astype(int) == y.to_numpy()
        if not consistent.all():
            raise TrainingDataError(
                f"{int((~consistent).sum()):,} row(s) have a target that disagrees with their attack family."
            )
    if dataset_meta is not None:
        recorded = dataset_meta.get("split", {}).get("train", {})
        if (recorded.get("rows"), recorded.get("benign"), recorded.get("attack")) != (len(X), n_benign, n_attack):
            raise TrainingDataError(
                f"train.parquet ({len(X):,} rows, {n_benign:,} benign, {n_attack:,} attack) does not match "
                f"dataset_metadata.json ({recorded}). Re-run: python -m nids.preprocessing"
            )
    return {config.BENIGN_LABEL: n_benign, config.ATTACK_LABEL: n_attack}


def load_training_data(
    paths: pp.ArtifactPaths | None = None,
    loader: Callable = pp.load_prepared_split,
) -> TrainingData:
    """Load and validate the Milestone 3 TRAINING split. The test split is never loaded."""
    paths = paths or pp.ArtifactPaths.default()
    X, y, meta = loader("train", paths)  # raises DatasetError with instructions if missing
    for required in (paths.feature_schema, paths.metadata):
        if not Path(required).exists():
            raise TrainingDataError(f"{required} not found. Run: python -m nids.preprocessing")
    schema = json.loads(Path(paths.feature_schema).read_text(encoding="utf-8"))
    dataset_meta = json.loads(Path(paths.metadata).read_text(encoding="utf-8"))
    for key in ("original_features", "retained_features"):
        if key not in schema:
            raise TrainingDataError(f"{paths.feature_schema} lacks '{key}'; incompatible schema file.")
    counts = validate_training_inputs(X, y, meta, schema["original_features"], dataset_meta)
    return TrainingData(
        X=X,
        y=y,
        meta=meta,
        original_features=list(schema["original_features"]),
        schema_retained_features=list(schema["retained_features"]),
        class_counts=counts,
        train_path=Path(paths.train),
        train_sha256=file_sha256(paths.train),
        schema_sha256=file_sha256(paths.feature_schema),
    )


# ---------------------------------------------------------------------------
# Fitted-model inspection
# ---------------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def fitted_facts(name: str, classifier) -> dict:
    """Facts read from the FITTED classifier (not the intended configuration)."""
    facts: dict = {
        "classes": [int(c) for c in classifier.classes_],
        "n_features_in": int(classifier.n_features_in_),
    }
    if name == "logistic_regression":
        n_iter = [int(i) for i in np.atleast_1d(classifier.n_iter_)]
        max_iter = classifier.get_params()["max_iter"]
        facts.update(n_iter=n_iter, max_iter=max_iter, reached_max_iter=any(i >= max_iter for i in n_iter))
    elif name == "decision_tree":
        depth = int(classifier.get_depth())
        max_depth = classifier.get_params()["max_depth"]
        facts.update(
            depth=depth,
            n_leaves=int(classifier.get_n_leaves()),
            node_count=int(classifier.tree_.node_count),
            reached_max_depth=bool(max_depth is not None and depth >= max_depth),
        )
    elif name == "random_forest":
        trees = classifier.estimators_
        facts.update(
            n_estimators_fitted=len(trees),
            max_tree_depth=int(max(t.get_depth() for t in trees)),
            mean_nodes_per_tree=float(np.mean([t.tree_.node_count for t in trees])),
        )
    elif name == "svm":
        facts.update(
            fit_status=int(classifier.fit_status_),
            n_support_per_class=[int(n) for n in classifier.n_support_],
            n_support_total=int(np.sum(classifier.n_support_)),
        )
    return facts


def check_fitted_structure(name: str, pipeline: Pipeline, data: TrainingData) -> dict:
    """Verify what the fitted pipeline really contains; raises TrainingError on mismatch."""
    selector = pipeline.named_steps["drop_constant"]
    if list(selector.feature_names_in_) != data.original_features:
        raise TrainingError(f"{name}: the pipeline did not see the original input features in schema order.")
    retained = [str(c) for c in selector.get_feature_names_out()]
    removed = [c for c in data.original_features if c not in set(retained)]
    if retained != data.schema_retained_features:
        raise TrainingError(
            f"{name}: the selector fitted here retains {len(retained)} features, but the Milestone 3 schema "
            f"retains {len(data.schema_retained_features)}. The training data may have changed; re-run preprocessing."
        )
    scaled = "scale" in pipeline.named_steps
    if scaled:
        scaler = pipeline.named_steps["scale"]
        if int(scaler.n_samples_seen_) != len(data.X) or not np.allclose(scaler.mean_, data.X[retained].mean().to_numpy()):
            raise TrainingError(f"{name}: the scaler was not fitted on exactly the training data.")
    classifier = pipeline.named_steps["classifier"]
    if int(classifier.n_features_in_) != len(retained):
        raise TrainingError(f"{name}: classifier expects {classifier.n_features_in_} features but {len(retained)} are produced.")
    if [str(c) for c in getattr(classifier, "feature_names_in_", retained)] != retained:
        raise TrainingError(f"{name}: classifier feature names differ from the selector output.")
    return {
        "original_input_features": list(data.original_features),
        "n_original_input_features": len(data.original_features),
        "retained_features": retained,
        "n_retained_features": len(retained),
        "removed_constant_features": removed,
        "scaling_applied": scaled,
        "n_features_passed_to_classifier": int(classifier.n_features_in_),
        "matches_milestone3_feature_schema": True,
    }


# ---------------------------------------------------------------------------
# Persistence and save-and-reload validation
# ---------------------------------------------------------------------------
def _validation_positions(n_rows: int, size: int) -> np.ndarray:
    """A fixed slice: evenly spaced training rows (deterministic, spread over the data)."""
    return np.unique(np.linspace(0, n_rows - 1, num=min(size, n_rows), dtype=np.int64))


def _score_method(pipeline) -> str | None:
    if hasattr(pipeline, "predict_proba"):
        return "predict_proba"
    if hasattr(pipeline, "decision_function"):
        return "decision_function"
    return None


def validate_roundtrip(
    pipeline: Pipeline,
    model_path: Path,
    data: TrainingData,
    rows: int = 2000,
    loader: Callable = joblib.load,
) -> dict:
    """Reload the saved model and compare it with the in-memory one on TRAINING rows only."""
    loaded = loader(model_path)
    sample = data.X.iloc[_validation_positions(len(data.X), rows)]
    problems = []
    if not np.array_equal(pipeline.predict(sample), loaded.predict(sample)):
        problems.append("class predictions differ")
    method = _score_method(pipeline)
    if method is not None:
        if not hasattr(loaded, method):
            problems.append(f"reloaded model lacks {method}")
        elif not np.array_equal(getattr(pipeline, method)(sample), getattr(loaded, method)(sample)):
            problems.append(f"{method} outputs differ")
    if list(getattr(loaded, "feature_names_in_", [])) != data.original_features:
        problems.append("reloaded model's input feature names/order differ from the schema")
    if problems:
        raise ModelValidationError("Save-and-reload validation FAILED: " + "; ".join(problems))
    return {
        "rows_checked": int(len(sample)),
        "slice_rule": "evenly spaced rows of the training split (no test data used)",
        "predictions_identical": True,
        "scores_compared": method,
        "scores_identical": method is not None,
        "input_feature_names_and_order_identical": True,
    }


@contextmanager
def _heartbeat(label: str, interval: float):
    """Log a progress line every ``interval`` seconds so a long fit is visibly alive."""
    stop = threading.Event()
    started = time.perf_counter()

    def beat() -> None:
        while not stop.wait(interval):
            logger.info("%s is still training... elapsed %.0f s", label, time.perf_counter() - started)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _step_summary(pipeline: Pipeline) -> list[dict]:
    return [
        {"name": step_name, "class": f"{type(step).__module__}.{type(step).__name__}", "params": step.get_params(deep=False)}
        for step_name, step in pipeline.steps
    ]


def _software() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def train_model(
    name: str,
    data: TrainingData,
    models_dir: Path = config.MODELS_DIR,
    *,
    pipeline_factory: Callable[[str], Pipeline] = build_pipeline,
    loader: Callable = joblib.load,
    heartbeat_seconds: float = 60.0,
    validation_rows: int = 2000,
) -> dict:
    """Train one model on the full training split, validate persistence, then publish it.

    Returns the metadata dict. On any failure or interruption no final artefact exists.
    """
    _require_known(name)
    models_dir = Path(models_dir)
    ensure_directories((models_dir,))
    final_model, final_meta = models_dir / f"{name}.joblib", models_dir / f"{name}.json"
    partial_model, partial_meta = (p.with_name(p.name + ".partial") for p in (final_model, final_meta))
    for leftover in (partial_model, partial_meta):
        leftover.unlink(missing_ok=True)  # debris from an earlier interrupted run

    n_records = len(data.X)
    logger.info("Training %s on %s training records (%d input features).", name, f"{n_records:,}", data.X.shape[1])
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # record every warning; none is hidden
            pipeline = pipeline_factory(name)
            started = time.perf_counter()
            try:
                with _heartbeat(name, heartbeat_seconds):
                    pipeline.fit(data.X, data.y)
            except Exception as exc:
                raise TrainingError(
                    f"{name}: training failed ({type(exc).__name__}: {exc}). No model artefact was written."
                ) from exc
            duration = time.perf_counter() - started
        training_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
        for item in training_warnings:
            logger.warning("%s: %s: %s", name, item["category"], item["message"])

        structure = check_fitted_structure(name, pipeline, data)
        classifier = pipeline.named_steps["classifier"]
        facts = fitted_facts(name, classifier)

        joblib.dump(pipeline, partial_model)
        model_sha = file_sha256(partial_model)
        validation = validate_roundtrip(pipeline, partial_model, data, validation_rows, loader)

        seed_affects, seed_note = SEED_EFFECT[name]
        classifier_params = classifier.get_params()
        metadata = {
            "model_name": name,
            "status": "completed_with_warnings" if training_warnings else "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "classifier": {
                "class": f"{type(classifier).__module__}.{type(classifier).__name__}",
                "hyperparameters": classifier_params,
                "report_settings_intended": config.MODEL_SETTINGS[name],
                "regularisation_mapping": (
                    _logistic_regression_kwargs(config.MODEL_SETTINGS[name])[1] if name == "logistic_regression" else None
                ),
                "fitted": facts,
            },
            "pipeline": {
                "steps": _step_summary(pipeline),
                "fitted_on": "training split only (selector and scaler)",
                "accepts": f"the {len(data.original_features)} original input features, in schema order",
            },
            "features": structure,
            "training": {
                "records": n_records,
                "class_counts": data.class_counts,
                "train_file": str(data.train_path),
                "train_file_sha256": data.train_sha256,
                "feature_schema_sha256": data.schema_sha256,
                "test_set_used": False,
            },
            "label_mapping": dict(config.LABEL_TO_INT, positive_class=config.ATTACK_LABEL),
            "seed": {
                "project_seed": config.RANDOM_SEED,
                "classifier_random_state": classifier_params.get("random_state"),
                "classifier_seed_affects_training": seed_affects,
                "note": seed_note,
                "selector_and_scaler": "deterministic; no random component",
            },
            "software": _software(),
            "fit": {"duration_seconds": round(duration, 3)},
            "warnings": training_warnings,
            "validation": {"save_and_reload": validation},
            "artifact": {"model_file": final_model.name, "model_sha256": model_sha, "model_bytes": partial_model.stat().st_size},
        }
        partial_meta.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

        os.replace(partial_model, final_model)
        if file_sha256(final_model) != model_sha:  # pragma: no cover - would indicate disk corruption
            final_model.unlink(missing_ok=True)
            raise ModelValidationError(f"{name}: model file changed while being moved into place.")
        os.replace(partial_meta, final_meta)  # metadata last: it marks the model as complete
        return metadata
    finally:
        for leftover in (partial_model, partial_meta):
            leftover.unlink(missing_ok=True)
        gc.collect()


def check_artifacts(models_dir: Path = config.MODELS_DIR) -> dict[str, dict]:
    """State of each model on disk. A model is complete only if its metadata says so and the
    model file's SHA-256 matches the metadata (detects partial or mismatched artefacts)."""
    models_dir = Path(models_dir)
    state: dict[str, dict] = {}
    for name in config.MODEL_NAMES:
        model_path, meta_path = models_dir / f"{name}.joblib", models_dir / f"{name}.json"
        if not model_path.exists() or not meta_path.exists():
            state[name] = {"complete": False, "reason": "missing model or metadata file"}
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            recorded = metadata["artifact"]["model_sha256"]
            status = metadata["status"]
        except (ValueError, KeyError) as exc:
            state[name] = {"complete": False, "reason": f"unreadable metadata ({exc})"}
            continue
        if file_sha256(model_path) != recorded:
            state[name] = {"complete": False, "reason": "model file does not match the metadata hash"}
        elif not status.startswith("completed"):
            state[name] = {"complete": False, "reason": f"status is {status!r}"}
        else:
            state[name] = {"complete": True, "status": status, "fit_seconds": metadata["fit"]["duration_seconds"]}
    return state


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        prog="python -m nids.train",
        description="Train models on the prepared TRAINING split. No test data is used and no metrics are computed.",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help=f"models to train; valid: {', '.join(config.MODEL_NAMES)}. "
             f"Default: {' '.join(config.INITIAL_MODELS)}. 'svm' is never trained unless named here.",
    )
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DATA_DIR, help="directory with train.parquet etc.")
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR, help="where model artefacts are written")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0, help="progress message interval during fitting")
    args = parser.parse_args(argv)

    try:
        names = parse_model_names(args.models)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    paths = pp.ArtifactPaths.under(config.INTERIM_DATA_DIR, args.processed_dir, config.LOGS_DIR)
    try:
        data = load_training_data(paths)
    except DatasetError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(f"Training data: {len(data.X):,} records "
          f"(BENIGN {data.class_counts[config.BENIGN_LABEL]:,}, ATTACK {data.class_counts[config.ATTACK_LABEL]:,}), "
          f"{len(data.original_features)} input features.")
    print(f"train.parquet SHA-256: {data.train_sha256}")

    outcomes: list[tuple[str, dict | None, str | None]] = []
    try:
        for name in names:
            if name == "svm":
                print("\n" + "=" * 70)
                print(f"SVM (RBF kernel) is about to train on all {len(data.X):,} records.")
                print("This can take a long time and cannot report percentage progress.")
                print(f"A status line is printed every {args.heartbeat_seconds:.0f} s. Do not close this window.")
                print("=" * 70, flush=True)
            try:
                metadata = train_model(name, data, args.models_dir, heartbeat_seconds=args.heartbeat_seconds)
                outcomes.append((name, metadata, None))
                print(f"[OK]   {name}: fitted in {metadata['fit']['duration_seconds']:.1f} s, "
                      f"save-and-reload check passed, status {metadata['status']}", flush=True)
            except TrainingError as exc:
                outcomes.append((name, None, str(exc)))
                print(f"[FAIL] {name}: {exc}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. No partial model artefact was left behind for the interrupted model.", file=sys.stderr)
        return 130

    print("\n" + "=" * 70 + "\nTRAINING SUMMARY (this invocation)\n" + "=" * 70)
    for name, metadata, error in outcomes:
        if metadata is None:
            print(f"{name:<22} FAILED  {error}")
            continue
        print(f"{name:<22} {metadata['status']:<24} fit {metadata['fit']['duration_seconds']:>9.1f} s")
        print(f"    model   : {args.models_dir / (name + '.joblib')}")
        print(f"    metadata: {args.models_dir / (name + '.json')}")
        print(f"    features: {metadata['features']['n_original_input_features']} input -> "
              f"{metadata['features']['n_retained_features']} retained -> "
              f"{metadata['features']['n_features_passed_to_classifier']} to classifier "
              f"(scaling {'on' if metadata['features']['scaling_applied'] else 'off'})")
        for item in metadata["warnings"]:
            print(f"    WARNING {item['category']}: {item['message']}")

    state = check_artifacts(args.models_dir)
    done = [n for n, s in state.items() if s["complete"]]
    pending = {n: s["reason"] for n, s in state.items() if not s["complete"]}
    print(f"\nComplete models on disk: {done or 'none'}")
    if pending:
        print(f"Not yet complete: {pending}")
    else:
        print("All four models are complete (artefact present, hash matches metadata).")
    return 1 if any(metadata is None for _, metadata, _ in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
