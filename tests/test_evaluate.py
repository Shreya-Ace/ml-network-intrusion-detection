"""Milestone 5 tests. SYNTHETIC data only - nothing here is a CIC-IDS2017 result.

Every expected number is counted by hand or recomputed independently with scikit-learn.
Tests needing PyArrow are skipped when it is not installed.
"""

import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from nids import config
from nids import data_loader as dl
from nids import evaluate as ev
from nids import preprocessing as pp
from nids import train as tr

FEATURES = ["f1", "c1", "f2", "f3", "c2", "f4"]  # c1, c2 constant by construction
RETAINED = ["f1", "f2", "f3", "f4"]
ALL_MODELS = list(config.MODEL_NAMES)


# ------------------------------------------------------------------ a small synthetic world
def _frames(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "c1": 5.0, "f2": rng.normal(size=n), "f3": rng.normal(size=n), "c2": 0.0, "f4": rng.normal(size=n)})[FEATURES]
    y = pd.Series(((X["f1"] + 0.5 * X["f2"] + rng.normal(scale=0.4, size=n)) > 0).astype("int8"), name="target")
    families = np.where(y == 1, rng.choice(["DoS Hulk", "Bot", "DDoS"], size=n, p=[0.6, 0.25, 0.15]), config.BENIGN_FAMILY)
    sources = np.where(y == 1, "c.parquet", rng.choice(["a.parquet", "b.parquet", "a.parquet|b.parquet"], size=n))
    meta = pd.DataFrame({config.ATTACK_FAMILY_COLUMN: families, "source_files": sources})
    return X, y, meta


_TEMPLATE = {}


def _template():
    """Train the four models once on synthetic data; tests copy this directory."""
    if "dir" not in _TEMPLATE:
        root = Path(tempfile.mkdtemp())
        paths = pp.ArtifactPaths.under(root / "interim", root / "processed", root / "logs")
        paths.train.parent.mkdir(parents=True)
        paths.train.write_bytes(b"synthetic train file")
        paths.test.write_bytes(b"synthetic test file")
        Xtr, ytr, mtr = _frames(400, 1)
        Xte, yte, mte = _frames(300, 2)
        paths.feature_schema.write_text(json.dumps({"original_features": FEATURES, "retained_features": RETAINED}), encoding="utf-8")
        fam = lambda m: {k: int(v) for k, v in m[config.ATTACK_FAMILY_COLUMN].value_counts().items()}  # noqa: E731
        paths.metadata.write_text(json.dumps({
            "split": {
                "train": {"rows": len(Xtr), "benign": int((ytr == 0).sum()), "attack": int((ytr == 1).sum()), "family_counts": fam(mtr)},
                "test": {"rows": len(Xte), "benign": int((yte == 0).sum()), "attack": int((yte == 1).sum()), "family_counts": fam(mte)},
            },
            "sample": {"family_counts": {k: fam(mtr).get(k, 0) + fam(mte).get(k, 0) for k in set(fam(mtr)) | set(fam(mte))}},
        }), encoding="utf-8")
        data = tr.TrainingData(Xtr, ytr, mtr[[config.ATTACK_FAMILY_COLUMN]], list(FEATURES), list(RETAINED),
                               tr.validate_training_inputs(Xtr, ytr, mtr, FEATURES), paths.train, tr.file_sha256(paths.train), "0" * 64)
        for name in ALL_MODELS:
            tr.train_model(name, data, root / "models", validation_rows=100)
        _TEMPLATE.update(dir=root, test=(Xte, yte, mte))
    return _TEMPLATE


class World:
    def __init__(self, tmp_path):
        template = _template()
        self.root = Path(tmp_path) / "world"
        shutil.copytree(template["dir"], self.root)
        self.paths = pp.ArtifactPaths.under(self.root / "interim", self.root / "processed", self.root / "logs")
        self.models_dir = self.root / "models"
        self.test = template["test"]
        self.calls = []

    def loader(self, name, paths):
        self.calls.append(name)
        return self.test

    def run(self, names=ALL_MODELS, **kwargs):
        return ev.run_evaluation(names, self.models_dir, self.paths, data_loader=self.loader, **kwargs)

    def data(self):
        return ev.load_test_data(self.paths, self.loader)


# ------------------------------------------------------------------ metrics (hand-counted)
Y_TRUE = [1, 1, 1, 0, 0, 0, 0, 1]
Y_PRED = [1, 0, 1, 0, 1, 0, 0, 1]
SCORES = [0.9, 0.4, 0.8, 0.1, 0.6, 0.2, 0.3, 0.7]


def test_confusion_counts_and_metrics_hand_counted():
    counts = ev.confusion_counts(Y_TRUE, Y_PRED)
    assert counts == {"tp": 3, "fn": 1, "fp": 1, "tn": 3}
    auc = ev.compute_roc_auc(Y_TRUE, SCORES)
    assert auc == pytest.approx(15 / 16)  # 15 of the 16 attack/benign pairs are ranked correctly
    m = ev.classification_metrics(counts, auc)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert (m["precision"], m["recall"], m["f1"]) == (pytest.approx(0.75), pytest.approx(0.75), pytest.approx(0.75))
    assert m["false_positive_rate"] == pytest.approx(0.25) and m["false_negative_rate"] == pytest.approx(0.25)
    assert m["specificity"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(precision_score(Y_TRUE, Y_PRED))
    assert m["f1"] == pytest.approx(f1_score(Y_TRUE, Y_PRED))
    assert auc == pytest.approx(roc_auc_score(Y_TRUE, SCORES))


def test_undefined_metrics_are_none_not_invented():
    counts = ev.confusion_counts([1, 1, 0, 0], [0, 0, 0, 0])  # nothing predicted ATTACK
    m = ev.classification_metrics(counts)
    assert m["precision"] is None and m["f1"] is None and m["recall"] == 0.0 and m["roc_auc"] is None
    assert ev.classification_metrics(ev.confusion_counts([0, 0], [0, 0]))["recall"] is None  # no attacks at all


def test_invalid_labels_and_single_class_auc_are_rejected():
    with pytest.raises(ev.EvaluationError, match="0/1"):
        ev.confusion_counts([0, 2], [0, 1])
    with pytest.raises(ev.EvaluationError, match="same|shape"):
        ev.confusion_counts([0, 1], [0])
    with pytest.raises(ev.EvaluationError, match="both classes"):
        ev.compute_roc_auc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ev.EvaluationError, match="NaN"):
        ev.compute_roc_auc([0, 1], [0.1, float("nan")])


def test_wilson_interval_known_values():
    low, high = ev.wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3) and high == pytest.approx(1.0)
    assert ev.wilson_interval(0, 0) == (None, None)
    low, high = ev.wilson_interval(0, 20)
    assert low == pytest.approx(0.0) and 0.1 < high < 0.2


def test_family_recall_hand_counted_with_insufficient_flags():
    families = ["DoS Hulk"] * 40 + ["Bot"] * 5 + ["Benign"] * 10
    y_true = [1] * 45 + [0] * 10
    y_pred = [1] * 38 + [0] * 2 + [1] * 4 + [0] * 1 + [0] * 10  # Hulk: 38/40, Bot: 4/5
    rows = {r["family"]: r for r in ev.family_recall_table(families, y_true, y_pred, {"DoS Hulk": 200, "Bot": 20}, {"DoS Hulk": 250})}
    assert (rows["DoS Hulk"]["test_records"], rows["DoS Hulk"]["detected"], rows["DoS Hulk"]["missed"]) == (40, 38, 2)
    assert rows["DoS Hulk"]["recall"] == pytest.approx(0.95) and rows["DoS Hulk"]["sufficient"] is True
    assert (rows["DoS Hulk"]["train_records"], rows["DoS Hulk"]["sample_records"]) == (200, 250)
    assert rows["Bot"]["recall"] == pytest.approx(0.8) and rows["Bot"]["sufficient"] is False  # only 5 test records
    assert rows["Heartbleed"]["test_records"] == 0 and rows["Heartbleed"]["recall"] is None and rows["Heartbleed"]["sufficient"] is False
    assert "Benign" not in rows and len(rows) == len(dl.FAMILY_NAMES) - 1
    assert rows["Bot"]["recall_ci95_low"] < 0.8 < rows["Bot"]["recall_ci95_high"]


def test_family_table_rejects_unknown_families_and_inconsistent_truth():
    with pytest.raises(ev.EvaluationError, match="Unknown attack families"):
        ev.family_recall_table(["Mystery"], [1], [1])
    with pytest.raises(ev.EvaluationError, match="not ATTACK"):
        ev.family_recall_table(["Bot"], [0], [1])


def test_false_positives_by_source_file_hand_counted():
    out = ev.false_positives_by_source_file(["a", "b", "a|b", "a", "c"], [0, 0, 0, 0, 1], [1, 0, 1, 0, 1])
    assert out["a"] == {"benign_records": 3, "false_positives": 2, "false_positive_rate": pytest.approx(2 / 3)}
    assert out["b"] == {"benign_records": 2, "false_positives": 1, "false_positive_rate": pytest.approx(0.5)}
    assert "c" not in out  # only attack records there


# ------------------------------------------------------------------ scoring
def test_scores_use_predict_proba_except_the_svm_which_uses_decision_function(tmp_path):
    world = World(tmp_path)
    X = world.test[0]
    for name in ALL_MODELS:
        pipeline, _ = ev.load_model(name, world.models_dir)
        scores, description = ev.model_scores(name, pipeline, X)
        if name == "svm":
            np.testing.assert_array_equal(scores, pipeline.decision_function(X))
            assert "NOT a probability" in description and not hasattr(pipeline, "predict_proba")
        else:
            np.testing.assert_array_equal(scores, pipeline.predict_proba(X)[:, 1])
            assert "predict_proba" in description


def test_svm_trained_with_probability_true_is_refused(tmp_path):
    world = World(tmp_path)
    X, y, _ = _frames(120, 5)
    pipe = Pipeline([*pp.preprocessor_for_model("svm").steps, ("classifier", SVC(probability=True, random_state=0))])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    with pytest.raises(ev.EvaluationError, match="probability=True"):
        ev.model_scores("svm", pipe, X)


# ------------------------------------------------------------------ evaluating the real (synthetic) models
@pytest.mark.parametrize("name", ALL_MODELS)
def test_metrics_match_an_independent_scikit_learn_computation(tmp_path, name):
    world = World(tmp_path)
    data = world.data()
    pipeline, _ = ev.load_model(name, world.models_dir)
    result, curve = ev.evaluate_model(name, pipeline, data)
    y, pred = data.y.to_numpy(), pipeline.predict(data.X)
    scores = pipeline.decision_function(data.X) if name == "svm" else pipeline.predict_proba(data.X)[:, 1]
    m = result["metrics"]
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, scores))
    c = result["confusion_matrix"]
    assert sum(c.values()) == len(y) == result["test_records"]
    assert c["tp"] + c["fn"] == int((y == 1).sum()) and c["tn"] + c["fp"] == int((y == 0).sum())
    families = result["family_recall"]
    assert sum(f["test_records"] for f in families) == int((y == 1).sum())  # sample counts preserved
    assert sum(f["detected"] for f in families) == c["tp"] and sum(f["missed"] for f in families) == c["fn"]
    assert result["positive_class"] == "ATTACK" and result["model_state_unchanged_by_evaluation"] is True
    assert len(curve[0]) == len(curve[1]) >= 2


def test_full_run_evaluates_all_four_models_on_the_same_test_set_and_touches_no_model_file(tmp_path):
    world = World(tmp_path)
    before = {f.name: tr.file_sha256(f) for f in world.models_dir.iterdir()}
    run = world.run()
    assert run.all_ok is True
    assert [n for n, r in run.results["models"].items() if r["status"] == "evaluated"] == ALL_MODELS
    assert world.calls == ["test"]  # the training split is never loaded
    assert {f.name: tr.file_sha256(f) for f in world.models_dir.iterdir()} == before  # models and metadata untouched
    assert all(r["model_file_unchanged_after_evaluation"] for r in run.results["models"].values())
    assert run.results["test_set"]["records"] == 300 and run.results["test_set"]["sha256"] == tr.file_sha256(world.paths.test)
    assert all(not p["problems"] for p in run.results["preflight"].values())


class Refitter(ClassifierMixin, BaseEstimator):
    """A classifier that tries to refit itself while predicting (module level so it can be pickled)."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        self.fit(X, np.zeros(len(X)))
        return np.zeros(len(X), dtype=int)


def test_evaluation_cannot_fit_or_modify_a_model(tmp_path):
    world = World(tmp_path)
    data = world.data()

    X, y, _ = world.test
    pipe = Pipeline([*pp.preprocessor_for_model("decision_tree").steps, ("classifier", Refitter())]).fit(X, y)
    with pytest.raises(ev.EvaluationError, match="Fitting is forbidden"):
        ev.evaluate_model("decision_tree", pipe, data)
    assert not any(k in vars(pipe) or k in vars(pipe.named_steps["classifier"]) for k in ("fit", "partial_fit", "fit_transform"))  # guard removed

    pipeline, _ = ev.load_model("random_forest", world.models_dir)
    state = joblib.hash(pipeline)
    ev.evaluate_model("random_forest", pipeline, data)
    assert joblib.hash(pipeline) == state  # evaluation left the fitted model byte-for-byte equivalent


# ------------------------------------------------------------------ failures are never silent
def test_a_missing_model_is_reported_as_failed_and_the_others_still_run(tmp_path):
    world = World(tmp_path)
    (world.models_dir / "svm.joblib").unlink()
    run = world.run()
    assert run.all_ok is False
    assert run.results["models"]["svm"]["status"] == "failed" and "not complete" in run.results["models"]["svm"]["error"]
    assert all(run.results["models"][n]["status"] == "evaluated" for n in ALL_MODELS if n != "svm")
    report = ev.format_evaluation_report(run.results)
    assert "EVALUATION FAILED FOR: svm" in report and "FAILED" in report


def test_a_tampered_model_file_is_refused(tmp_path):
    world = World(tmp_path)
    with open(world.models_dir / "decision_tree.joblib", "ab") as handle:
        handle.write(b"x")
    run = world.run(["decision_tree"])
    assert run.results["models"]["decision_tree"]["status"] == "failed" and run.all_ok is False


def test_preflight_catches_test_set_use_changed_settings_and_stale_data(tmp_path):
    world = World(tmp_path)
    meta_path = world.models_dir / "logistic_regression.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["training"]["test_set_used"] = True
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    problems = world.run(["logistic_regression"]).results["preflight"]["logistic_regression"]["problems"]
    assert any("test_set_used" in p for p in problems)

    world = World(tmp_path / "second")
    meta_path = world.models_dir / "decision_tree.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["classifier"]["report_settings_intended"]["max_depth"] = 5  # pretend the settings changed after training
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert any("MODEL_SETTINGS" in p for p in world.run(["decision_tree"]).results["preflight"]["decision_tree"]["problems"])

    world = World(tmp_path / "third")
    world.paths.train.write_bytes(b"a different training file")
    assert any("train.parquet changed" in p for p in world.run(["random_forest"]).results["preflight"]["random_forest"]["problems"])


def test_missing_or_incompatible_test_data_is_rejected(tmp_path):
    empty = pp.ArtifactPaths.under(tmp_path, tmp_path, tmp_path)
    with pytest.raises(dl.DatasetError, match="nids.preprocessing"):
        ev.load_test_data(empty)
    world = World(tmp_path / "w")
    X, y, meta = world.test
    with pytest.raises(ev.EvaluationDataError, match="missing .*f4"):
        ev.load_test_data(world.paths, lambda n, p: (X.drop(columns=["f4"]), y, meta))
    bad_y = y.copy()
    bad_y.iloc[0] = 3
    with pytest.raises(ev.EvaluationDataError, match="Invalid target labels"):
        ev.load_test_data(world.paths, lambda n, p: (X, bad_y, meta))
    with pytest.raises(ev.EvaluationDataError, match="test.parquet .* does not match"):
        ev.load_test_data(world.paths, lambda n, p: (X.iloc[:-1], y.iloc[:-1], meta.iloc[:-1]))


# ------------------------------------------------------------------ comparison with the report
def test_report_claims_are_internally_consistent_so_implied_counts_are_trustworthy():
    claims = config.REPORT_CLAIMS
    rf = ev.implied_confusion_from_report(claims["models"]["random_forest"], 8000, 12000)
    assert rf == claims["random_forest_confusion"] == {"tn": 11900, "fp": 100, "fn": 100, "tp": 7900}
    lr = ev.implied_confusion_from_report(claims["models"]["logistic_regression"], 8000, 12000)
    assert (lr["fn"], lr["fp"]) == (380, 420)
    for name, claimed in claims["models"].items():
        c = ev.implied_confusion_from_report(claimed, 8000, 12000)
        assert abs((c["tp"] + c["tn"]) / 20000 - claimed["accuracy"]) < 0.0005  # the report's table agrees with itself


def test_comparison_computes_differences_and_never_changes_the_measured_metrics(tmp_path):
    run = World(tmp_path).run()
    models = run.results["models"]
    measured = {n: dict(models[n]["metrics"]) for n in models}
    altered = json.loads(json.dumps(config.REPORT_CLAIMS))
    for entry in altered["models"].values():
        entry.update(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5)
    default = ev.compare_with_report(models)
    other = ev.compare_with_report(models, altered)
    for name in models:
        assert models[name]["metrics"] == measured[name]  # claims never touch the metrics
        d = default["per_model"][name]
        assert d["difference_percentage_points"]["accuracy"] == pytest.approx((measured[name]["accuracy"] - config.REPORT_CLAIMS["models"][name]["accuracy"]) * 100)
        assert other["per_model"][name]["difference_percentage_points"]["accuracy"] == pytest.approx((measured[name]["accuracy"] - 0.5) * 100)
    assert default["accuracy_ranking_measured"] == sorted(models, key=lambda n: -measured[n]["accuracy"])
    assert set(default["random_forest_confusion"]) == {"tn", "fp", "fn", "tp"}
    assert default["possible_reasons_for_differences_UNTESTED"]


def test_comparison_skips_failed_models_without_inventing_numbers():
    models = {"svm": {"status": "failed", "error": "boom"}}
    entry = ev.compare_with_report(models)["per_model"]["svm"]
    assert "measured" not in entry and entry["report"]["accuracy"] == config.REPORT_CLAIMS["models"]["svm"]["accuracy"]


# ------------------------------------------------------------------ outputs
def test_outputs_are_written_machine_readable_and_ascii_safe(tmp_path):
    world = World(tmp_path)
    run = world.run()
    out = tmp_path / "results"
    written = ev.write_outputs(run, out)
    assert all(Path(p).exists() for p in written.values())
    stored = json.loads(written["evaluation_metrics_json"].read_text(encoding="utf-8"))
    assert set(stored["models"]) == set(ALL_MODELS) and stored["positive_class"] == "ATTACK"
    assert stored["models"]["svm"]["roc_auc_score_type"].startswith("decision_function")
    comparison = pd.read_csv(written["model_comparison_csv"])
    assert list(comparison["model"]) == ALL_MODELS and {"accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "tn", "fp", "fn", "tp"} <= set(comparison.columns)
    assert len(pd.read_csv(written["family_recall_csv"])) == len(ALL_MODELS) * (len(dl.FAMILY_NAMES) - 1)
    assert len(pd.read_csv(written["report_comparison_csv"])) == len(ALL_MODELS) * len(ev.METRIC_KEYS)
    figures = {k: v for k, v in written.items() if k.startswith("figure_")}
    assert {"figure_metrics_comparison", "figure_roc_curves", "figure_family_recall", "figure_confusion_matrices_all", "figure_report_vs_measured"} <= set(figures)
    assert all(f"figure_confusion_matrix_{n}" in figures for n in ALL_MODELS)
    for path in figures.values():
        assert Path(path).read_bytes()[:4] == b"\x89PNG"
    text = written["evaluation_report_txt"].read_text(encoding="utf-8")
    assert text.isascii() and "INSUFFICIENT" in text and "UNTESTED" in text and "NOT a probability" in text


def test_cli_rejects_unknown_models_and_missing_data(tmp_path):
    assert ev.main(["--models", "linearsvc"]) == 2
    assert ev.main(["--processed-dir", str(tmp_path), "--models-dir", str(tmp_path), "--results-dir", str(tmp_path / "r")]) == 2
    assert not (tmp_path / "r").exists()


def test_cli_end_to_end_on_real_parquet_files(tmp_path):
    pytest.importorskip("pyarrow")
    world = World(tmp_path)
    processed = world.root / "processed"
    for split, (X, y, meta) in (("train", _frames(400, 1)), ("test", world.test)):
        frame = X.copy()
        frame["obs_id"] = [f"{i:032x}-00" for i in range(len(X))]
        frame[config.ATTACK_FAMILY_COLUMN] = meta[config.ATTACK_FAMILY_COLUMN].to_numpy()
        frame[config.ORIGINAL_LABEL_COLUMN] = frame[config.ATTACK_FAMILY_COLUMN]
        frame["target"] = y.to_numpy()
        frame["n_source_files"] = 1
        frame["source_files"] = meta["source_files"].to_numpy()
        frame["occurrences"] = 1
        frame.to_parquet(processed / f"{split}.parquet", index=False)
    # the real parquet files differ from the placeholder bytes the models were trained with, so retrain on them
    assert tr.main(["--models", *ALL_MODELS, "--processed-dir", str(processed), "--models-dir", str(world.models_dir)]) == 0
    code = ev.main(["--processed-dir", str(processed), "--models-dir", str(world.models_dir), "--results-dir", str(world.root / "results")])
    assert code == 0
    assert (world.root / "results" / "metrics" / "evaluation_metrics.json").exists()
