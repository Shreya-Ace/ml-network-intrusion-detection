"""Milestone 4 tests. SYNTHETIC data only - no full-size training happens here and nothing
below is a CIC-IDS2017 result. Tests needing PyArrow are skipped when it is not installed.
"""

import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from nids import config
from nids import data_loader as dl
from nids import preprocessing as pp
from nids import train as tr

FEATURES = ["f1", "c1", "f2", "f3", "c2", "f4"]  # c1, c2 are constant by construction
RETAINED = ["f1", "f2", "f3", "f4"]
ALL_MODELS = list(config.MODEL_NAMES)
FORBIDDEN_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "f1_score", "roc_auc", "confusion_matrix", "auc"}


def synthetic_arrays(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"f1": rng.normal(size=n), "c1": 5.0, "f2": rng.normal(size=n), "f3": rng.normal(size=n), "c2": 0.0, "f4": rng.normal(size=n)}
    )[FEATURES]
    y = pd.Series((X["f1"] + 0.5 * X["f2"] > 0).astype("int8"), name="target")
    meta = pd.DataFrame({config.ATTACK_FAMILY_COLUMN: np.where(y == 1, "DoS Hulk", config.BENIGN_FAMILY)})
    return X, y, meta


def make_data(tmp_path, n=300, seed=0):
    X, y, meta = synthetic_arrays(n, seed)
    counts = tr.validate_training_inputs(X, y, meta, FEATURES)
    train_file = tmp_path / "train.parquet"
    train_file.write_bytes(b"synthetic training file")
    return tr.TrainingData(
        X=X, y=y, meta=meta, original_features=list(FEATURES), schema_retained_features=list(RETAINED),
        class_counts=counts, train_path=train_file, train_sha256=tr.file_sha256(train_file), schema_sha256="0" * 64,
    )


def all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from all_keys(item)


# ------------------------------------------------------------------ configuration
def test_parse_model_names_default_never_includes_svm():
    assert tr.parse_model_names(None) == ["logistic_regression", "decision_tree", "random_forest"]
    assert tr.parse_model_names(["svm", "svm", "decision_tree"]) == ["svm", "decision_tree"]


def test_unknown_model_names_are_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="Unknown model name.*Valid names"):
        tr.parse_model_names(["logistic_regression", "xgboost"])
    with pytest.raises(ValueError, match="No model names"):
        tr.parse_model_names([])
    with pytest.raises(ValueError, match="Valid names"):
        tr.build_classifier("bogus")
    with pytest.raises(ValueError, match="Valid names"):
        tr.build_pipeline("bogus")


def test_decision_tree_random_forest_and_svm_use_the_report_settings():
    dt = tr.build_classifier("decision_tree").get_params()
    assert (dt["criterion"], dt["max_depth"], dt["random_state"]) == ("gini", 20, 42)
    rf = tr.build_classifier("random_forest").get_params()
    assert (rf["n_estimators"], rf["criterion"], rf["random_state"]) == (100, "gini", 42)
    svm = tr.build_classifier("svm").get_params()
    assert (svm["kernel"], svm["C"], svm["gamma"], svm["probability"], svm["random_state"]) == ("rbf", 1.0, "scale", False, 42)


def test_logistic_regression_settings_and_l2_regularisation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lr = tr.build_classifier("logistic_regression")
    assert not caught  # no deprecation warning from the parameters we pass
    params = lr.get_params()
    assert (params["C"], params["max_iter"], params["solver"], params["random_state"]) == (1.0, 1000, "lbfgs", 42)
    if params.get("penalty") not in (None, "deprecated"):
        assert params["penalty"] == "l2"
    else:
        assert params["l1_ratio"] == 0.0  # pure L2 in scikit-learn >= 1.8
    assert tr._logistic_regression_kwargs(config.MODEL_SETTINGS["logistic_regression"])[1]  # mapping is documented


def test_no_tuning_class_weights_or_extra_options_are_introduced():
    for name in ALL_MODELS:
        assert tr.build_classifier(name).get_params().get("class_weight") is None
    assert tr.build_classifier("random_forest").get_params()["max_depth"] is None  # report gives none
    assert set(config.MODEL_SETTINGS) == set(config.MODEL_NAMES)


# ------------------------------------------------------------------ pipelines
def test_pipeline_structure_scaling_only_for_lr_and_svm():
    expected = {
        "logistic_regression": ["drop_constant", "scale", "classifier"],
        "svm": ["drop_constant", "scale", "classifier"],
        "decision_tree": ["drop_constant", "classifier"],
        "random_forest": ["drop_constant", "classifier"],
    }
    for name, steps in expected.items():
        pipe = tr.build_pipeline(name)
        assert isinstance(pipe, Pipeline)
        assert [s for s, _ in pipe.steps] == steps


def test_selector_is_fitted_on_training_data_and_pipeline_accepts_original_features(tmp_path):
    data = make_data(tmp_path)
    pipe = tr.build_pipeline("logistic_regression").fit(data.X, data.y)
    selector = pipe.named_steps["drop_constant"]
    assert list(selector.get_feature_names_out()) == RETAINED
    assert list(selector.feature_names_in_) == FEATURES  # accepts the 6 original features
    assert pipe.named_steps["classifier"].n_features_in_ == 4  # but the classifier sees only 4
    shifted = data.X.copy()
    shifted["c1"] = np.arange(len(shifted), dtype=float)  # varies in new data, constant in training
    np.testing.assert_array_equal(pipe.predict(shifted), pipe.predict(data.X))  # ignored: selector fixed at fit time


def test_scaler_is_fitted_on_training_data_only_and_never_refitted(tmp_path):
    data = make_data(tmp_path)
    pipe = tr.build_pipeline("svm").fit(data.X, data.y)
    scaler = pipe.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, data.X[RETAINED].mean().to_numpy())
    before = scaler.mean_.copy()
    pipe.predict(data.X + 100.0)  # very different data
    np.testing.assert_array_equal(scaler.mean_, before)


def test_wrong_column_order_is_rejected_by_the_pipeline(tmp_path):
    data = make_data(tmp_path)
    pipe = tr.build_pipeline("decision_tree").fit(data.X, data.y)
    with pytest.raises(ValueError):
        pipe.predict(data.X[list(reversed(FEATURES))])


# ------------------------------------------------------------------ training data validation
def test_valid_training_inputs_return_label_named_counts():
    X, y, meta = synthetic_arrays()
    counts = tr.validate_training_inputs(X, y, meta, FEATURES)
    assert counts == {"BENIGN": int((y == 0).sum()), "ATTACK": int((y == 1).sum())}


def test_invalid_labels_single_class_and_missing_target_are_rejected():
    X, y, meta = synthetic_arrays()
    with pytest.raises(tr.TrainingDataError, match="Invalid target labels"):
        tr.validate_training_inputs(X, y.where(y.index != 0, 2), meta, FEATURES)
    with pytest.raises(tr.TrainingDataError, match="Both classes"):
        tr.validate_training_inputs(X, y * 0, meta, FEATURES)
    with pytest.raises(tr.TrainingDataError, match="missing values"):
        tr.validate_training_inputs(X, y.astype("float64").where(y.index != 0, np.nan), meta, FEATURES)


def test_target_must_agree_with_attack_family():
    X, y, meta = synthetic_arrays()
    meta.loc[0, config.ATTACK_FAMILY_COLUMN] = "Benign" if y.iloc[0] == 1 else "DoS Hulk"
    with pytest.raises(tr.TrainingDataError, match="disagrees with their attack family"):
        tr.validate_training_inputs(X, y, meta, FEATURES)


def test_incompatible_columns_nan_and_non_numeric_features_are_rejected():
    X, y, meta = synthetic_arrays()
    with pytest.raises(tr.TrainingDataError, match="missing .*'f4'"):
        tr.validate_training_inputs(X.drop(columns=["f4"]), y, meta, FEATURES)
    with pytest.raises(tr.TrainingDataError, match="names and order"):
        tr.validate_training_inputs(X[list(reversed(FEATURES))], y, meta, FEATURES)
    bad = X.copy()
    bad.loc[3, "f1"] = np.nan
    with pytest.raises(tr.TrainingDataError, match="NaN/inf"):
        tr.validate_training_inputs(bad, y, meta, FEATURES)
    text = X.copy().astype({"f1": "object"})
    text.loc[0, "f1"] = "oops"
    with pytest.raises(tr.TrainingDataError, match="numeric"):
        tr.validate_training_inputs(text, y, meta, FEATURES)
    with pytest.raises(tr.TrainingDataError, match="empty"):
        tr.validate_training_inputs(X.iloc[0:0], y.iloc[0:0], meta.iloc[0:0], FEATURES)


def test_training_data_must_match_the_dataset_metadata():
    X, y, meta = synthetic_arrays()
    good = {"split": {"train": {"rows": len(X), "benign": int((y == 0).sum()), "attack": int((y == 1).sum())}}}
    tr.validate_training_inputs(X, y, meta, FEATURES, good)
    stale = {"split": {"train": {"rows": len(X) + 1, "benign": 1, "attack": 1}}}
    with pytest.raises(tr.TrainingDataError, match="does not match dataset_metadata.json"):
        tr.validate_training_inputs(X, y, meta, FEATURES, stale)


def test_file_sha256_known_vector(tmp_path):
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    assert tr.file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def placeholder_paths(tmp_path, schema=None, dataset_meta="match"):
    """Artefact paths with placeholder files (the loader is replaced by a spy in these tests)."""
    X, y, _ = synthetic_arrays()
    paths = pp.ArtifactPaths.under(tmp_path / "i", tmp_path / "p", tmp_path / "l")
    paths.train.parent.mkdir(parents=True)
    paths.train.write_bytes(b"placeholder train file")
    schema = {"original_features": FEATURES, "retained_features": RETAINED} if schema is None else schema
    paths.feature_schema.write_text(json.dumps(schema), encoding="utf-8")
    if dataset_meta == "match":
        dataset_meta = {"split": {"train": {"rows": len(X), "benign": int((y == 0).sum()), "attack": int((y == 1).sum())}}}
    paths.metadata.write_text(json.dumps(dataset_meta), encoding="utf-8")
    return paths


def test_only_the_training_split_is_ever_loaded(tmp_path):
    calls = []

    def spy(name, paths):
        calls.append(name)
        return synthetic_arrays()

    data = tr.load_training_data(placeholder_paths(tmp_path), loader=spy)
    assert calls == ["train"]
    assert data.train_sha256 == hashlib.sha256(b"placeholder train file").hexdigest()  # hash of the actual file contents
    assert data.schema_retained_features == RETAINED and data.original_features == FEATURES


def test_missing_training_data_gives_an_actionable_error(tmp_path):
    empty = pp.ArtifactPaths.under(tmp_path, tmp_path, tmp_path)
    with pytest.raises(dl.DatasetError, match="nids.preprocessing"):
        tr.load_training_data(empty)


def test_incompatible_schema_file_is_rejected(tmp_path):
    paths = placeholder_paths(tmp_path, schema={"original_features": FEATURES})
    with pytest.raises(tr.TrainingDataError, match="retained_features"):
        tr.load_training_data(paths, loader=lambda name, p: synthetic_arrays())
    with pytest.raises(tr.TrainingDataError, match="does not match"):
        tr.load_training_data(placeholder_paths(tmp_path / "x", dataset_meta={"split": {"train": {"rows": 1, "benign": 1, "attack": 0}}}), loader=lambda n, p: synthetic_arrays())


# ------------------------------------------------------------------ training, persistence, metadata
@pytest.mark.parametrize("name", ALL_MODELS)
def test_train_model_publishes_validated_artifacts_and_accurate_metadata(tmp_path, name):
    data = make_data(tmp_path)
    models = tmp_path / "models"
    meta = tr.train_model(name, data, models, heartbeat_seconds=0.01, validation_rows=100)

    assert sorted(p.name for p in models.iterdir()) == [f"{name}.joblib", f"{name}.json"]  # no .partial debris
    stored = json.loads((models / f"{name}.json").read_text(encoding="utf-8"))
    assert stored["model_name"] == name and stored["status"] in ("completed", "completed_with_warnings")

    # metadata describes what was actually fitted
    assert stored["training"]["records"] == len(data.X)
    assert stored["training"]["class_counts"] == {"BENIGN": int((data.y == 0).sum()), "ATTACK": int((data.y == 1).sum())}
    assert stored["training"]["train_file_sha256"] == hashlib.sha256(b"synthetic training file").hexdigest()
    assert stored["training"]["test_set_used"] is False
    assert stored["label_mapping"] == {"BENIGN": 0, "ATTACK": 1, "positive_class": "ATTACK"}
    feats = stored["features"]
    assert feats["original_input_features"] == FEATURES and feats["n_original_input_features"] == 6
    assert feats["retained_features"] == RETAINED and feats["n_retained_features"] == 4
    assert feats["removed_constant_features"] == ["c1", "c2"]
    assert feats["n_features_passed_to_classifier"] == 4
    assert feats["scaling_applied"] is (name in ("logistic_regression", "svm"))
    assert [s["name"] for s in stored["pipeline"]["steps"]][-1] == "classifier"
    assert stored["classifier"]["hyperparameters"]["random_state"] == 42
    assert stored["seed"]["classifier_seed_affects_training"] is (name in ("decision_tree", "random_forest"))
    assert stored["software"]["scikit_learn"] and stored["software"]["python"] and stored["software"]["numpy"] and stored["software"]["pandas"]
    assert stored["fit"]["duration_seconds"] >= 0 and isinstance(stored["warnings"], list)
    assert stored["validation"]["save_and_reload"]["predictions_identical"] is True
    assert stored["validation"]["save_and_reload"]["scores_compared"] == ("decision_function" if name == "svm" else "predict_proba")
    assert not FORBIDDEN_METRIC_KEYS & set(all_keys(stored))  # no predicted metrics anywhere
    assert meta == json.loads(json.dumps(meta, default=tr._json_default))

    # the stored hash is that of the file actually on disk, and the reload really is identical
    assert stored["artifact"]["model_sha256"] == tr.file_sha256(models / f"{name}.joblib")
    reloaded = joblib.load(models / f"{name}.joblib")
    original = tr.build_pipeline(name).fit(data.X, data.y)
    np.testing.assert_array_equal(reloaded.predict(data.X), original.predict(data.X))
    assert tr.check_artifacts(models)[name]["complete"] is True


def test_fitted_facts_describe_the_fitted_models_not_the_intent(tmp_path):
    data = make_data(tmp_path)
    facts = {n: tr.train_model(n, data, tmp_path / "m", validation_rows=50)["classifier"]["fitted"] for n in ALL_MODELS}
    assert facts["logistic_regression"]["reached_max_iter"] is False and facts["logistic_regression"]["n_iter"][0] < 1000
    assert facts["decision_tree"]["depth"] <= 20 and facts["decision_tree"]["reached_max_depth"] is False
    assert facts["random_forest"]["n_estimators_fitted"] == 100
    assert facts["svm"]["fit_status"] == 0 and facts["svm"]["n_support_total"] == sum(facts["svm"]["n_support_per_class"]) > 0
    assert all(f["classes"] == [0, 1] and f["n_features_in"] == 4 for f in facts.values())


def test_convergence_warnings_are_recorded_not_hidden(tmp_path):
    data = make_data(tmp_path)

    def tiny_iteration_budget(name):
        pipe = tr.build_pipeline(name)
        pipe.named_steps["classifier"].set_params(max_iter=1)
        return pipe

    meta = tr.train_model("logistic_regression", data, tmp_path / "m", pipeline_factory=tiny_iteration_budget, validation_rows=50)
    assert meta["status"] == "completed_with_warnings"
    assert any(w["category"] == "ConvergenceWarning" for w in meta["warnings"])
    assert meta["classifier"]["fitted"]["reached_max_iter"] is True


# ------------------------------------------------------------------ failure handling
class Exploding(ClassifierMixin, BaseEstimator):
    def __init__(self, exception=RuntimeError):
        self.exception = exception

    def fit(self, X, y):
        raise self.exception("boom")


def exploding_factory(exception):
    def factory(name):
        return Pipeline([*pp.preprocessor_for_model(name).steps, ("classifier", Exploding(exception))])
    return factory


def test_failed_training_leaves_no_artifacts(tmp_path):
    data = make_data(tmp_path)
    models = tmp_path / "models"
    with pytest.raises(tr.TrainingError, match="training failed.*boom"):
        tr.train_model("decision_tree", data, models, pipeline_factory=exploding_factory(RuntimeError))
    assert list(models.iterdir()) == []


def test_interrupted_training_leaves_no_artifacts(tmp_path):
    data = make_data(tmp_path)
    models = tmp_path / "models"
    with pytest.raises(KeyboardInterrupt):
        tr.train_model("decision_tree", data, models, pipeline_factory=exploding_factory(KeyboardInterrupt))
    assert list(models.iterdir()) == []


def test_failed_save_and_reload_check_leaves_no_artifacts(tmp_path):
    data = make_data(tmp_path)
    models = tmp_path / "models"
    flipped = tr.build_pipeline("decision_tree").fit(data.X, 1 - data.y)  # predicts the opposite classes
    with pytest.raises(tr.ModelValidationError, match="FAILED.*predictions differ"):
        tr.train_model("decision_tree", data, models, loader=lambda path: flipped)
    assert list(models.iterdir()) == []


def test_stale_partial_files_from_an_interrupted_run_are_removed(tmp_path):
    data = make_data(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "decision_tree.joblib.partial").write_bytes(b"half written")
    (models / "decision_tree.json.partial").write_bytes(b"{")
    tr.train_model("decision_tree", data, models, validation_rows=50)
    assert sorted(p.name for p in models.iterdir()) == ["decision_tree.joblib", "decision_tree.json"]


def test_check_artifacts_detects_missing_tampered_and_incomplete_models(tmp_path):
    data = make_data(tmp_path)
    models = tmp_path / "models"
    tr.train_model("decision_tree", data, models, validation_rows=50)
    state = tr.check_artifacts(models)
    assert state["decision_tree"]["complete"] is True
    assert state["svm"] == {"complete": False, "reason": "missing model or metadata file"}
    with open(models / "decision_tree.joblib", "ab") as handle:
        handle.write(b"x")  # corrupt the model file
    assert "hash" in tr.check_artifacts(models)["decision_tree"]["reason"]


# ------------------------------------------------------------------ command line
def test_cli_rejects_unknown_models_and_missing_data(tmp_path):
    assert tr.main(["--models", "bogus"]) == 2
    assert tr.main(["--models", "decision_tree", "--processed-dir", str(tmp_path), "--models-dir", str(tmp_path / "m")]) == 2
    assert not (tmp_path / "m").exists()  # nothing written when the data is missing


def write_prepared_files(directory):
    """A tiny processed dataset in the layout Milestone 3 produces (needs PyArrow)."""
    X, y, _ = synthetic_arrays()
    directory.mkdir(parents=True, exist_ok=True)
    frame = X.copy()
    frame["obs_id"] = [f"{i:032x}-00" for i in range(len(X))]
    frame[config.ATTACK_FAMILY_COLUMN] = np.where(y == 1, "DoS Hulk", "Benign")
    frame[config.ORIGINAL_LABEL_COLUMN] = frame[config.ATTACK_FAMILY_COLUMN]
    frame["target"] = y.to_numpy()
    frame["n_source_files"] = 1
    frame["source_files"] = "a.parquet"
    frame["occurrences"] = 1
    frame.to_parquet(directory / "train.parquet", index=False)
    (directory / "feature_schema.json").write_text(json.dumps({"original_features": FEATURES, "retained_features": RETAINED}), encoding="utf-8")
    meta = {"split": {"train": {"rows": len(X), "benign": int((y == 0).sum()), "attack": int((y == 1).sum())}}}
    (directory / "dataset_metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_cli_default_trains_the_three_fast_models_and_never_svm(tmp_path):
    pytest.importorskip("pyarrow")
    write_prepared_files(tmp_path / "p")
    code = tr.main(["--processed-dir", str(tmp_path / "p"), "--models-dir", str(tmp_path / "m"), "--heartbeat-seconds", "5"])
    assert code == 0
    state = tr.check_artifacts(tmp_path / "m")
    assert [n for n, s in state.items() if s["complete"]] == ["logistic_regression", "decision_tree", "random_forest"]
    assert not (tmp_path / "m" / "svm.joblib").exists()


def test_cli_trains_svm_only_when_requested_and_all_four_can_complete(tmp_path):
    pytest.importorskip("pyarrow")
    write_prepared_files(tmp_path / "p")
    base = ["--processed-dir", str(tmp_path / "p"), "--models-dir", str(tmp_path / "m")]
    assert tr.main(["--models", "logistic_regression", "decision_tree", "random_forest", *base]) == 0
    assert not (tmp_path / "m" / "svm.joblib").exists()
    assert tr.main(["--models", "svm", *base]) == 0
    assert all(s["complete"] for s in tr.check_artifacts(tmp_path / "m").values())


def test_load_training_data_reads_real_parquet_files(tmp_path):
    pytest.importorskip("pyarrow")
    write_prepared_files(tmp_path / "p")
    paths = pp.ArtifactPaths.under(tmp_path, tmp_path / "p", tmp_path)
    data = tr.load_training_data(paths)
    assert list(data.X.columns) == FEATURES and len(data.X) == 300
    assert data.train_sha256 == tr.file_sha256(tmp_path / "p" / "train.parquet")
    assert not set(data.X.columns) & set(data.meta.columns)
