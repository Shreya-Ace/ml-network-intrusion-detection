"""Milestone 3 tests. SYNTHETIC data only - nothing here is a CIC-IDS2017 result.

Every expected number below was counted by hand from the scenario definitions.
Tests that need PyArrow are skipped automatically when it is not installed.
"""

import json

import numpy as np
import pandas as pd
import pytest

from nids import config
from nids import data_loader as dl
from nids import preprocessing as pp

FEATURES = ["f1", "f2", "const"]
NAN, INF = float("nan"), float("inf")


def vec(i):
    """A distinct 3-feature vector per i (the last feature is constant by design)."""
    return (float(i), float(i % 5), 3.0)


def make_frame(name, rows):
    """rows: (label, (f1, f2, const)) -> frame in data_loader.load_file's format."""
    frame = pd.DataFrame([v for _, v in rows], columns=FEATURES).astype("float64")
    frame[config.ORIGINAL_LABEL_COLUMN] = pd.Series([lab for lab, _ in rows], dtype="object")
    frame[config.SOURCE_FILE_COLUMN] = name
    return frame


def make_source(frames):
    return pp.FrameSource(names=list(frames), feature_columns=list(FEATURES), loader=lambda n: frames[n].copy())


def scenario_frames():
    """15 rows / 3 files. See the expectations in test_plan_counts_on_scenario."""
    a = [
        ("Benign", vec(1)),
        ("Benign", vec(1)),  # same-label duplicate inside the file
        ("Benign", vec(2)),
        ("Benign", vec(3)),  # conflicts with the attack rows of vec(3) in B and C
        ("Bot", vec(10)),
        ("Benign", (NAN, 0.0, 3.0)),  # invalid: NaN
        ("Benign", (INF, 0.0, 3.0)),  # invalid: inf
    ]
    b = [
        ("Benign", vec(1)),  # same-label duplicate across files
        ("DoS Hulk", vec(3)),  # conflict
        ("DoS Hulk", vec(11)),
        ("DoS GoldenEye", vec(11)),  # identical features, other ATTACK family -> kept
        ("Benign", vec(2)),  # duplicate across files
    ]
    c = [
        ("Bot", vec(10)),  # attack duplicate across files
        ("Benign", vec(4)),
        ("DoS Hulk", vec(3)),  # conflict
    ]
    return {"a.parquet": make_frame("a.parquet", a), "b.parquet": make_frame("b.parquet", b), "c.parquet": make_frame("c.parquet", c)}


def plan_scenario():
    source = make_source(scenario_frames())
    index, file_stats = pp.build_row_index(source)
    return index, file_stats, pp.plan_exclusions_and_dedup(index, source.names), source


# ------------------------------------------------------------------ invalid values
def test_find_invalid_rows_counts_nan_and_inf():
    frame = pd.DataFrame({"x": [1.0, NAN, INF, 4.0], "y": [1.0, 2.0, -INF, NAN]})
    report = pp.find_invalid_rows(frame, ["x", "y"])
    assert report.mask.tolist() == [False, True, True, True]
    assert (report.nan_cells, report.inf_cells) == (2, 2)
    assert (report.rows_with_nan, report.rows_with_inf, report.rows_invalid) == (2, 1, 3)  # row 2 holds two inf cells


def test_find_invalid_rows_requires_the_columns():
    with pytest.raises(dl.SchemaError, match="Missing required"):
        pp.find_invalid_rows(pd.DataFrame({"x": [1.0]}), ["x", "y"])


def test_coerce_features_to_float_reports_unparseable_values():
    frame = pd.DataFrame({"x": ["1", "oops", "3"], "y": [1, 2, 3]})
    out, unparseable = pp.coerce_features_to_float(frame, ["x", "y"])
    assert unparseable == {"x": 1}
    assert out["x"].isna().tolist() == [False, True, False]
    assert all(str(t) == "float64" for t in out.dtypes)
    assert pp.find_invalid_rows(out, ["x", "y"]).rows_invalid == 1  # caught, never imputed silently


def test_align_feature_columns_orders_columns_and_reports_extras():
    frame = pd.DataFrame({"b": [1], " a": [2], "extra": [3]})
    aligned, extras = pp.align_feature_columns(frame, ["a", "b"])
    assert list(aligned.columns) == ["a", "b"]
    assert aligned.iloc[0].tolist() == [2, 1]
    assert extras == ["extra"]


def test_align_feature_columns_missing_column_raises():
    with pytest.raises(dl.SchemaError, match="Missing required feature column"):
        pp.align_feature_columns(pd.DataFrame({"a": [1]}), ["a", "b"])


# ------------------------------------------------------------------ grouping helper
def test_sorted_groups_is_stable_so_first_row_is_earliest():
    keys = [np.array([2, 1, 2, 1, 3], dtype=np.uint64)]
    order, starts, gid = pp._sorted_groups(keys)
    assert starts.size == 3
    assert order[starts].tolist() == [1, 0, 4]  # earliest occurrence of keys 1, 2, 3
    assert gid.tolist() == [0, 0, 1, 1, 2]


# ------------------------------------------------------------------ exclusion / dedup plan
def test_plan_counts_on_scenario():
    index, file_stats, plan, source = plan_scenario()
    s = plan.stats
    assert s["raw_rows"] == 15
    assert s["invalid_values"]["rows_removed"] == 2
    assert [f["rows_invalid"] for f in file_stats] == [2, 0, 0]
    assert (file_stats[0]["rows_with_nan"], file_stats[0]["rows_with_inf"]) == (1, 1)

    c = s["conflicts"]
    assert c["feature_vectors"] == 1
    assert c["rows_excluded"] == 3
    assert (c["benign_rows_excluded"], c["attack_rows_excluded"]) == (1, 2)
    assert c["family_counts_excluded"] == {"Benign": 1, "DoS Hulk": 2}
    assert c["rows_by_source_file"] == {
        "a.parquet": {"benign": 1, "attack": 0},
        "b.parquet": {"benign": 0, "attack": 1},
        "c.parquet": {"benign": 0, "attack": 1},
    }
    assert c["exclusion_rate_by_family"]["DoS Hulk"]["valid_rows"] == 3
    assert c["exclusion_rate_by_family"]["DoS Hulk"]["fraction"] == pytest.approx(2 / 3)
    assert c["exclusion_rate_by_family"]["Benign"]["valid_rows"] == 7
    assert c["attack_family_only_vectors_retained"] == 1  # vec(11): Hulk vs GoldenEye

    d = s["deduplication"]
    assert (d["rows_before"], d["unique_observations"], d["rows_removed"]) == (10, 6, 4)
    assert (d["benign_rows_removed"], d["attack_rows_removed"]) == (3, 1)
    assert d["observations_spanning_multiple_files"] == 3

    e = s["eligible"]
    assert (e["total"], e["benign"], e["attack"]) == (6, 3, 3)
    assert {k: v for k, v in e["by_family"].items() if v} == {"Benign": 3, "Bot": 1, "DoS Hulk": 1, "DoS GoldenEye": 1}
    assert s["accounting_ok"] is True


def test_conflicting_rows_are_excluded_on_both_sides_not_resolved():
    index, _, plan, _ = plan_scenario()
    kept_vectors = set(zip(plan.eligible["h1"], plan.eligible["h2"]))
    conflict_rows = index[plan.status == pp.CONFLICT]
    assert len(conflict_rows) == 3
    assert not (set(zip(conflict_rows["h1"], conflict_rows["h2"])) & kept_vectors)


def test_excluded_conflict_audit_identifies_the_exact_rows():
    _, _, plan, _ = plan_scenario()
    audit = plan.excluded_conflicts
    assert set(zip(audit["source_file"], audit["row_position"])) == {("a.parquet", 3), ("b.parquet", 1), ("c.parquet", 2)}
    assert audit["feature_vector_id"].nunique() == 1
    assert sorted(audit["attack_family"]) == ["Benign", "DoS Hulk", "DoS Hulk"]


def test_representative_row_is_the_earliest_occurrence():
    _, _, plan, _ = plan_scenario()
    el = plan.eligible
    benign_triplicate = el[(~el["is_attack"]) & (el["occurrences"] == 3)]
    assert len(benign_triplicate) == 1
    assert (int(benign_triplicate["rep_file"].iloc[0]), int(benign_triplicate["rep_row"].iloc[0])) == (0, 0)
    bot = el[el["fam"] == dl._FAMILY_IDS["Bot"]]
    assert (int(bot["rep_file"].iloc[0]), int(bot["rep_row"].iloc[0])) == (0, 4)


def test_provenance_manifest_keeps_every_file_membership_and_status():
    _, _, plan, _ = plan_scenario()
    m = plan.manifest
    assert len(m) == 14
    assert m["status"].value_counts().to_dict() == {"retained": 9, "excluded_conflict": 3, "excluded_invalid": 2}
    twice_in_a = m[(m["status"] == "retained") & (m["source_file"] == "a.parquet") & (m["occurrences"] == 2)]
    assert len(twice_in_a) == 1  # benign vec(1) occurs twice in file a ...
    h1, h2 = int(twice_in_a["h1"].iloc[0]), int(twice_in_a["h2"].iloc[0])
    same = m[(m["h1"] == h1) & (m["h2"] == h2) & (m["status"] == "retained")]
    assert dict(zip(same["source_file"].astype(str), same["occurrences"])) == {"a.parquet": 2, "b.parquet": 1}  # ... and once in b


def test_membership_counts_keep_occurrences_unique_and_overlap_apart():
    _, _, plan, source = plan_scenario()
    summary = pp.membership_summary(plan.retained_long, plan.eligible, source.names)
    per_file = summary["all"]["per_file"]
    assert per_file["a.parquet"] == {"occurrences": 4, "unique_observations": 3, "exclusive_to_file": 0, "shared_with_other_files": 3}
    assert per_file["b.parquet"] == {"occurrences": 4, "unique_observations": 4, "exclusive_to_file": 2, "shared_with_other_files": 2}
    assert per_file["c.parquet"] == {"occurrences": 2, "unique_observations": 2, "exclusive_to_file": 1, "shared_with_other_files": 1}
    assert summary["all"]["unique_observations_total"] == 6
    assert summary["all"]["unique_observations_spanning_multiple_files"] == 3
    assert sum(f["unique_observations"] for f in per_file.values()) == 9  # overlaps: NOT 6
    assert sum(f["occurrences"] for f in per_file.values()) == 10  # occurrences DO add up
    assert summary["attack"]["per_file"]["b.parquet"] == {"occurrences": 2, "unique_observations": 2, "exclusive_to_file": 2, "shared_with_other_files": 0}
    assert summary["benign"]["per_file"]["c.parquet"]["exclusive_to_file"] == 1


def test_conflict_inside_one_file_and_three_way_conflict_are_excluded():
    def mk(rows):
        return pd.DataFrame(rows, columns=["file", "row", "h1", "h2", "fam", "valid"]).astype(
            {"file": "int16", "row": "int32", "h1": "uint64", "h2": "uint64", "fam": "int16", "valid": "bool"}
        )

    benign, hulk, bot, ddos = dl._FAMILY_IDS["Benign"], dl._FAMILY_IDS["DoS Hulk"], dl._FAMILY_IDS["Bot"], dl._FAMILY_IDS["DDoS"]
    index = mk([
        (0, 0, 1, 1, benign, True), (0, 1, 1, 1, hulk, True),  # same-file conflict
        (0, 2, 2, 2, benign, True), (1, 0, 2, 2, bot, True), (1, 1, 2, 2, ddos, True),  # three-way conflict
        (1, 2, 3, 3, benign, True),  # clean
    ])
    plan = pp.plan_exclusions_and_dedup(index, ["x", "y"])
    assert plan.stats["conflicts"]["feature_vectors"] == 2
    assert plan.stats["conflicts"]["rows_excluded"] == 5
    assert plan.stats["eligible"]["total"] == 1


def test_plan_rejects_unordered_index_and_unknown_labels():
    base = {"file": [0, 0], "row": [1, 0], "h1": [1, 2], "h2": [1, 2], "fam": [0, 0], "valid": [True, True]}
    with pytest.raises(ValueError, match="ordered"):
        pp.plan_exclusions_and_dedup(pd.DataFrame(base), ["x"])
    base.update(row=[0, 1], fam=[0, -1])
    with pytest.raises(ValueError, match="Unknown labels"):
        pp.plan_exclusions_and_dedup(pd.DataFrame(base), ["x"])


def test_plan_handles_all_rows_invalid():
    frames = {"a.parquet": make_frame("a.parquet", [("Benign", (NAN, 1.0, 3.0)), ("Bot", (INF, 1.0, 3.0))])}
    source = make_source(frames)
    index, _ = pp.build_row_index(source)
    plan = pp.plan_exclusions_and_dedup(index, source.names)
    assert plan.stats["invalid_values"]["rows_removed"] == 2
    assert plan.stats["eligible"]["total"] == 0
    with pytest.raises(pp.InfeasibleSampleError):
        pp.sample_observations(plan.eligible, pp.PreparationParams(n_benign=1, n_attack=1))


def test_unknown_label_stops_indexing_with_the_file_name():
    frames = scenario_frames()
    frames["b.parquet"].loc[0, config.ORIGINAL_LABEL_COLUMN] = "Mystery"
    with pytest.raises(dl.UnknownLabelError, match="b.parquet"):
        pp.build_row_index(make_source(frames))


# ------------------------------------------------------------------ larger synthetic experiment
def big_frames():
    """3 files: 150 unique benign vectors (some in 2 files), attacks aligned with files,
    1 conflicting vector (benign copy of the first DoS Hulk vector) and 1 NaN row."""
    files = {n: [] for n in ("f0.parquet", "f1.parquet", "f2.parquet")}
    names = list(files)
    for i in range(150):
        files[names[i % 3]].append(("Benign", vec(i)))
        if i % 5 == 0:
            files[names[(i + 1) % 3]].append(("Benign", vec(i)))  # cross-file duplicate
    for i in range(40):
        files["f1.parquet"].append(("DoS Hulk", vec(1000 + i)))
    for i in range(25):
        files["f2.parquet"].append(("DDoS", vec(2000 + i)))
    for i in range(8):
        files["f0.parquet"].append(("Bot", vec(3000 + i)))
    for i in range(2):
        files["f1.parquet"].append(("Infiltration", vec(4000 + i)))
    files["f0.parquet"].append(("Benign", vec(1000)))  # conflicts with the first DoS Hulk vector
    files["f2.parquet"].append(("Benign", (NAN, 1.0, 3.0)))
    return {n: make_frame(n, rows) for n, rows in files.items()}


PARAMS = pp.PreparationParams(seed=7, n_benign=30, n_attack=20, test_size=0.2)


def test_prepare_sizes_stratification_and_accounting():
    result = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    meta = result.metadata
    assert meta["sample"]["total"] == 50 and (meta["sample"]["benign"], meta["sample"]["attack"]) == (30, 20)
    assert (len(result.train), len(result.test)) == (40, 10)
    assert (meta["split"]["train"]["benign"], meta["split"]["train"]["attack"]) == (24, 16)
    assert (meta["split"]["test"]["benign"], meta["split"]["test"]["attack"]) == (6, 4)
    assert meta["conflicts"]["feature_vectors"] == 1 and meta["conflicts"]["rows_excluded"] == 2
    assert meta["invalid_values"]["rows_removed"] == 1
    assert meta["eligible"]["attack"] == 39 + 25 + 8 + 2  # the conflicting Hulk vector is gone
    assert meta["eligible"]["benign"] == 150
    assert meta["integrity"]["removal_accounting_ok"] is True
    json.dumps(meta, default=pp._json_default)
    assert pp.format_preparation_report(meta).isascii()


def test_sample_contains_only_valid_unique_non_conflicting_records():
    result = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    both = pd.concat([result.train, result.test])
    assert np.isfinite(both[FEATURES].to_numpy()).all()
    assert both["obs_id"].is_unique
    assert 1000.0 not in set(both["f1"])  # conflicting vector excluded on both sides
    assert not (set(result.train["obs_id"]) & set(result.test["obs_id"]))


def test_metadata_is_preserved_and_never_part_of_the_feature_matrix():
    result = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    x, y, meta = pp.split_features_and_metadata(result.train, FEATURES)
    assert list(x.columns) == FEATURES
    assert not set(x.columns) & set(pp.SAMPLE_METADATA_COLUMNS)
    assert set(y) <= {0, 1} and len(y) == len(x)
    assert {"obs_id", "attack_family", "original_label", "n_source_files", "source_files"} <= set(meta.columns)


def test_file_membership_of_sampled_observations_is_recorded():
    # Request ALL 150 eligible benign vectors so the cross-file duplicate vec(0) is certainly sampled.
    params = pp.PreparationParams(seed=7, n_benign=150, n_attack=20, test_size=0.2)
    result = pp.prepare_from_source(make_source(big_frames()), params)
    both = pd.concat([result.train, result.test], ignore_index=True)
    assert (both["source_files"].str.count(r"\|") + 1 == both["n_source_files"]).all()
    row = both[both["f1"] == 0.0]
    assert len(row) == 1  # vec(0): benign, occurs in f0 and (cross-file duplicate) f1
    assert row["source_files"].iloc[0] == "f0.parquet|f1.parquet"
    assert (row["n_source_files"].iloc[0], row["occurrences"].iloc[0]) == (2, 2)
    single = both[both["f1"] == 1.0]  # vec(1): benign, only in f1
    assert single["source_files"].iloc[0] == "f1.parquet" and single["n_source_files"].iloc[0] == 1
    assert both.loc[both["attack_family"] == "Bot", "source_files"].eq("f0.parquet").all()  # attacks stay file-aligned
    # per-file membership summary for the sample distinguishes occurrences from unique observations
    sample_membership = result.metadata["membership"]["sample"]["all"]
    assert sample_membership["unique_observations_total"] == 170
    assert sample_membership["unique_observations_spanning_multiple_files"] == 30  # i % 5 == 0 -> 30 vectors


def test_feature_column_colliding_with_metadata_name_is_rejected():
    frame = make_frame("a.parquet", [("Benign", vec(1)), ("Bot", vec(2))]).rename(columns={"f1": "target"})
    source = pp.FrameSource(["a.parquet"], ["target", "f2", "const"], lambda n: frame.copy())
    index, _ = pp.build_row_index(source)
    plan = pp.plan_exclusions_and_dedup(index, source.names)
    with pytest.raises(dl.SchemaError, match="collide"):
        pp.materialise_sample(source, plan.eligible, source.names, plan.retained_long)


def test_preparation_is_reproducible_and_seed_dependent():
    first = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    second = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    assert first.train.equals(second.train) and first.test.equals(second.test)
    other = pp.prepare_from_source(make_source(big_frames()), pp.PreparationParams(seed=8, n_benign=30, n_attack=20))
    ids = lambda r: set(pd.concat([r.train, r.test])["obs_id"])  # noqa: E731
    assert ids(first) != ids(other)


def test_sampling_does_not_depend_on_file_order():
    frames = big_frames()
    forward = pp.prepare_from_source(make_source(frames), PARAMS)
    reverse = pp.prepare_from_source(make_source({n: frames[n] for n in reversed(list(frames))}), PARAMS)
    ids = lambda r: sorted(pd.concat([r.train, r.test])["obs_id"])  # noqa: E731
    assert ids(forward) == ids(reverse)


def test_infeasible_request_stops_and_reports_actual_counts():
    with pytest.raises(pp.InfeasibleSampleError, match=r"benign eligible=150.*attack eligible=74"):
        pp.prepare_from_source(make_source(big_frames()), pp.PreparationParams(n_benign=1000, n_attack=20))


def test_test_set_family_counts_are_flagged_when_too_small():
    meta = pp.prepare_from_source(make_source(big_frames()), PARAMS).metadata
    reliability = meta["split"]["family_test_reliability"]
    assert reliability and not any(r["sufficient_for_family_metrics"] for r in reliability.values())  # test set has 10 rows
    assert meta["split"]["test"]["family_counts"].keys() <= set(dl.FAMILY_NAMES)


def test_params_validation():
    with pytest.raises(ValueError):
        pp.PreparationParams(n_benign=0)
    with pytest.raises(ValueError):
        pp.PreparationParams(test_size=1.5)


# ------------------------------------------------------------------ feature selection and scaling
def test_constant_features_are_fitted_on_training_data_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0] * 4, "c": [7.0] * 4})
    test = pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, 5.0], "c": [7.0, 99.0]})  # c varies ONLY in test
    schema = pp.fit_feature_schema(train, test, ["a", "b", "c"])
    assert schema["retained_features"] == ["a"]
    assert schema["removed_constant_features"] == ["b", "c"]  # the test set did not rescue c
    assert schema["n_original"] == 3 and schema["n_retained"] == 1
    assert schema["matches_reported_count"] is False
    assert str(config.REPORTED_FEATURE_COUNT) in schema["discrepancy_note"]


def test_feature_names_and_order_are_consistent_across_train_test_and_scaled_outputs():
    result = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    schema = result.feature_schema
    assert schema["retained_features"] == ["f1", "f2"] and schema["removed_constant_features"] == ["const"]
    x_train, _, _ = pp.split_features_and_metadata(result.train, FEATURES)
    x_test, _, _ = pp.split_features_and_metadata(result.test, FEATURES)
    for model in pp.SCALED_MODELS + pp.UNSCALED_MODELS:
        pipe = pp.preprocessor_for_model(model).fit(x_train)
        assert list(pipe.transform(x_train).columns) == list(pipe.transform(x_test).columns) == schema["retained_features"]


def test_scaler_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(10, 2, 200), "b": rng.normal(-3, 5, 200), "c": 1.0})
    test = pd.DataFrame({"a": rng.normal(50, 2, 50), "b": rng.normal(40, 5, 50), "c": 1.0})
    pipe = pp.preprocessor_for_model("logistic_regression").fit(train)
    scaler = pipe.named_steps["scale"]
    mean_before = scaler.mean_.copy()
    out_train, out_test = pipe.transform(train), pipe.transform(test)
    np.testing.assert_allclose(scaler.mean_, mean_before)  # transform never refits
    np.testing.assert_allclose(scaler.mean_, train[["a", "b"]].mean().to_numpy())
    np.testing.assert_allclose(out_train.mean().to_numpy(), 0.0, atol=1e-9)
    assert out_test["a"].mean() > 10  # test data is shifted, proving it was not used to fit


def test_model_specific_preprocessing_choice():
    assert "scale" in pp.preprocessor_for_model("svm").named_steps
    assert "scale" in pp.preprocessor_for_model("logistic_regression").named_steps
    assert "scale" not in pp.preprocessor_for_model("random_forest").named_steps
    assert "scale" not in pp.preprocessor_for_model("decision_tree").named_steps
    with pytest.raises(ValueError, match="Unknown model"):
        pp.preprocessor_for_model("xgboost")


def test_cli_returns_2_for_missing_data_and_bad_arguments(tmp_path):
    assert pp.main(["--data-dir", str(tmp_path / "missing"), "--no-write"]) == 2
    assert pp.main(["--data-dir", str(tmp_path), "--test-size", "2", "--no-write"]) == 2


# ------------------------------------------------------------------ PyArrow-dependent
def write_parquet_source(directory, frames):
    """Store the in-memory frames as Parquet with mixed integer widths and a messy label name."""
    for position, (name, frame) in enumerate(frames.items()):
        out = frame.drop(columns=[config.SOURCE_FILE_COLUMN])
        out = out.astype({"f2": ["int8", "int16", "int32"][position % 3], "const": "int8"})  # f2/const are always finite
        out = out.rename(columns={config.ORIGINAL_LABEL_COLUMN: " Label" if position == 0 else "Label"})
        out.to_parquet(directory / name)


def test_parquet_pipeline_matches_the_in_memory_pipeline(tmp_path):
    pytest.importorskip("pyarrow")
    write_parquet_source(tmp_path, big_frames())
    from_parquet = pp.prepare_dataset(tmp_path, PARAMS)
    in_memory = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    ids = lambda r: sorted(pd.concat([r.train, r.test])["obs_id"])  # noqa: E731
    assert ids(from_parquet) == ids(in_memory)
    assert from_parquet.metadata["eligible"] == in_memory.metadata["eligible"]
    assert from_parquet.metadata["provenance"]["status"] == "probable_match_unverified"


def test_artifacts_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    result = pp.prepare_from_source(make_source(big_frames()), PARAMS)
    paths = pp.ArtifactPaths.under(tmp_path / "interim", tmp_path / "processed", tmp_path / "logs")
    written = pp.write_artifacts(result, paths)
    assert all(p.exists() for p in written.values())
    x, y, meta = pp.load_prepared_split("train", paths)
    assert list(x.columns) == FEATURES and len(x) == 40 and set(y) <= {0, 1}
    assert not set(x.columns) & set(meta.columns)
    schema = json.loads(paths.feature_schema.read_text(encoding="utf-8"))
    assert schema["retained_features"] == ["f1", "f2"]
    stored = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert stored["artifacts"]["train"]["bytes"] > 0
    assert set(stored["artifacts"]) == set(vars(paths))  # all 8 artefacts are listed ...
    assert stored["artifacts"]["metadata"]["bytes"] is None and stored["artifacts"]["report"]["bytes"] is None
    report_text = paths.report.read_text(encoding="utf-8")  # ... including those written last
    assert all(str(path) in report_text for path in vars(paths).values())
    manifest = pd.read_parquet(paths.manifest)
    assert set(manifest["status"].astype(str)) <= set(pp.STATUS_NAMES)
    assert len(pd.read_csv(paths.excluded_conflicts)) == 2  # the conflicting vector: 1 benign + 1 attack row


def test_load_prepared_split_without_artifacts_gives_a_clear_error(tmp_path):
    paths = pp.ArtifactPaths.under(tmp_path, tmp_path, tmp_path)
    with pytest.raises(dl.DatasetError, match="nids.preprocessing"):
        pp.load_prepared_split("train", paths)
