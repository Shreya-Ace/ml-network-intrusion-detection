"""Milestone 2 tests. SYNTHETIC data only - nothing here is a CIC-IDS2017 result.

Tests that need PyArrow are skipped automatically when it is not installed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids import config
from nids import data_loader as dl

FEATURES = ["f1", "f2", "f3", "const"]
OBSERVED_RAW_LABELS = {
    "Benign": "Benign",
    "Bot": "Bot",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS Hulk": "DoS Hulk",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS slowloris": "DoS slowloris",
    "FTP-Patator": "FTP-Patator",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "SSH-Patator": "SSH-Patator",
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
}
NAN, INF = float("nan"), float("inf")


def make_frame(name, rows):
    """rows: (f1, f2, f3, const, raw_label) tuples -> frame in load_file's format."""
    frame = pd.DataFrame(rows, columns=[*FEATURES, config.ORIGINAL_LABEL_COLUMN])
    frame[FEATURES] = frame[FEATURES].astype("float64")
    frame[config.ORIGINAL_LABEL_COLUMN] = frame[config.ORIGINAL_LABEL_COLUMN].astype("object")
    frame[config.SOURCE_FILE_COLUMN] = name
    return frame


def scenario_frames():
    """13 rows over 3 files with hand-counted duplicates, conflicts and invalid rows."""
    a = [
        (1, 1, 1, 3, "Benign"),
        (1, 1, 1, 3, "Benign"),  # within-file duplicate
        (2, 2, 2, 3, "Benign"),
        (NAN, 0, 0, 3, "Benign"),  # missing value
        (INF, 0, 0, 3, "Benign"),  # infinite value
    ]
    b = [
        (1, 1, 1, 3, "Benign"),  # duplicate of a[0] across files
        (5, 5, 5, 3, "DoS Hulk"),
        (6, 6, 6, 3, "DoS Hulk"),
        (6, 6, 6, 3, "DoS GoldenEye"),  # same features, other attack family
        (2, 2, 2, 3, "DoS Hulk"),  # same features as a[2] but ATTACK vs Benign
    ]
    c = [
        (7, 7, 7, 3, "Bot"),
        (7, 7, 7, 3, "Bot"),  # within-file duplicate
        (5, 5, 5, 3, "DoS Hulk"),  # duplicate of b[1] across files
    ]
    return [("a.parquet", make_frame("a.parquet", a)),
            ("b.parquet", make_frame("b.parquet", b)),
            ("c.parquet", make_frame("c.parquet", c))]


def check_scenario_audit(audit):
    assert audit["totals"]["rows"] == 13
    assert audit["totals"]["files"] == 3
    assert audit["totals"]["rows_with_invalid_values"] == 2

    a, b, c = audit["files"]
    assert a["rows"] == 5
    assert a["missing_values"] == {"cells": 1, "rows_affected": 1, "per_column": {"f1": 1}}
    assert a["infinite_values"] == {"cells": 1, "rows_affected": 1, "per_column": {"f1": 1}}
    assert a["rows_with_invalid_values"] == 2
    assert a["duplicates_within_file"] == {"identical_features": 1, "identical_features_and_family": 1}
    assert b["duplicates_within_file"] == {"identical_features": 1, "identical_features_and_family": 0}
    assert c["duplicates_within_file"] == {"identical_features": 1, "identical_features_and_family": 1}

    dup = audit["duplicates"]
    assert dup["by_identical_features"] == {
        "unique": 7, "duplicate_rows": 6, "within_file_duplicate_rows": 3, "cross_file_only_duplicate_rows": 3}
    assert dup["by_features_and_family"] == {
        "unique": 9, "duplicate_rows": 4, "within_file_duplicate_rows": 2, "cross_file_only_duplicate_rows": 2}
    assert dup["keys_present_in_multiple_files"] == 2
    assert dup["per_file_keys_also_in_other_files"] == {"a.parquet": 1, "b.parquet": 2, "c.parquet": 1}

    conf = audit["conflicts"]
    assert conf["feature_keys_with_conflicting_families"] == 2
    assert conf["rows_involved"] == 4
    assert conf["benign_vs_attack_keys"] == 1
    assert conf["attack_family_only_keys"] == 1
    combos = {tuple(x["families"]) for x in conf["top_family_combinations"]}
    assert combos == {("Benign", "DoS Hulk"), ("DoS GoldenEye", "DoS Hulk")}

    el = audit["eligible_provisional"]
    assert el["valid_rows_before_dedup"] == 11
    assert el["unique_eligible_benign"] == 2
    assert el["unique_eligible_attack"] == 5
    assert el["by_family"]["DoS Hulk"] == 3
    assert el["by_family"]["DoS GoldenEye"] == 1
    assert el["by_family"]["Bot"] == 1
    assert el["by_first_file"] == {
        "a.parquet": {"benign": 2, "attack": 0},
        "b.parquet": {"benign": 0, "attack": 4},
        "c.parquet": {"benign": 0, "attack": 1},
    }
    assert el["benign_target_feasible"] is False and el["attack_target_feasible"] is False
    assert len(audit["issues"]) == 2  # only the two target shortfalls

    assert audit["feature_ranges"]["constant_columns"] == ["const"]
    assert audit["feature_ranges"]["columns_without_finite_values"] == []


# ------------------------------------------------------------------ labels
def test_label_registry_covers_the_15_observed_labels_without_collisions():
    assert dl.FAMILY_NAMES[0] == config.BENIGN_FAMILY
    assert len(dl.FAMILY_NAMES) == 15


@pytest.mark.parametrize("raw,family", list(OBSERVED_RAW_LABELS.items()))
def test_every_observed_raw_label_resolves(raw, family):
    assert dl.resolve_family(raw) == family


def test_label_resolution_ignores_case_and_whitespace():
    assert dl.resolve_family("  benign ") == "Benign"
    assert dl.resolve_family("DOS HULK") == "DoS Hulk"


def test_unknown_and_null_labels_resolve_to_none():
    assert dl.resolve_family("Mystery") is None
    assert dl.resolve_family(None) is None
    assert dl.resolve_family(float("nan")) is None


def test_canonicalise_labels_binary_mapping_and_index():
    raw = pd.Series(["Benign", "DoS Hulk", "Web Attack \ufffd XSS", "benign"], index=[10, 11, 12, 13])
    out = dl.canonicalise_labels(raw)
    assert list(out.index) == [10, 11, 12, 13]
    assert out[config.IS_ATTACK_COLUMN].tolist() == [False, True, True, False]
    assert out[config.ATTACK_FAMILY_COLUMN].tolist() == ["Benign", "DoS Hulk", "Web Attack - XSS", "Benign"]


def test_unknown_label_raises_informative_error():
    with pytest.raises(dl.UnknownLabelError, match="Mystery"):
        dl.canonicalise_labels(pd.Series(["Benign", "Mystery", "Mystery"]))


def test_null_label_raises():
    with pytest.raises(dl.UnknownLabelError, match="null"):
        dl.canonicalise_labels(pd.Series(["Benign", None], dtype="object"))


def test_unknown_label_is_never_treated_as_attack():
    resolution = dl.resolve_labels(pd.Series(["Benign", "Mystery"]))
    assert resolution.family_ids.tolist() == [dl.BENIGN_ID, dl.UNKNOWN_ID]


# ------------------------------------------------------------------ discovery
def test_discover_missing_directory_gives_actionable_error(tmp_path):
    with pytest.raises(dl.DatasetNotFoundError, match="not found"):
        dl.discover_parquet_files(tmp_path / "nope")


def test_discover_empty_directory(tmp_path):
    with pytest.raises(dl.DatasetNotFoundError, match="No .parquet files"):
        dl.discover_parquet_files(tmp_path)


def test_discover_sorted_and_ignores_other_files(tmp_path):
    for name in ["b.parquet", "A.PARQUET", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in dl.discover_parquet_files(tmp_path)] == ["A.PARQUET", "b.parquet"]


def test_cli_returns_2_when_dataset_is_missing(tmp_path):
    assert dl.main(["--data-dir", str(tmp_path / "missing"), "--no-save"]) == 2


# ------------------------------------------------------------------ schema
def info(name, columns, types):
    return dl.FileInfo(name, Path(name), 10, 100, tuple(columns), tuple(types), "test")


GOOD_COLS = ["a", "b", " Label"]
GOOD_TYPES = ["int8", "double", "string"]


def test_normalise_column_names_strips_whitespace():
    assert dl.normalise_column_names([" a", "b "]) == ["a", "b"]


def test_normalise_column_names_rejects_collisions_and_empty():
    with pytest.raises(dl.SchemaError, match="Duplicate"):
        dl.normalise_column_names(["a", " a"])
    with pytest.raises(dl.SchemaError, match="empty"):
        dl.normalise_column_names(["a", "  "])


def test_validate_schemas_accepts_identical_names_and_reports_type_differences():
    report = dl.validate_schemas([
        info("f1", GOOD_COLS, GOOD_TYPES),
        info("f2", ["a", "b", "Label"], ["int32", "double", "dictionary<values=string, indices=int8, ordered=0>"]),
    ])
    assert report.label_column == "Label"
    assert report.feature_columns == ("a", "b")
    assert set(report.type_differences) == {"a", "Label"}


def test_validate_schemas_rejects_missing_extra_and_reordered_columns():
    ref = info("f1", GOOD_COLS, GOOD_TYPES)
    with pytest.raises(dl.SchemaError, match="missing"):
        dl.validate_schemas([ref, info("f2", ["a", "Label"], ["int8", "string"])])
    with pytest.raises(dl.SchemaError, match="unexpected"):
        dl.validate_schemas([ref, info("f2", ["a", "b", "c", "Label"], ["int8", "double", "int8", "string"])])
    with pytest.raises(dl.SchemaError, match="DIFFERENT ORDER"):
        dl.validate_schemas([ref, info("f2", ["b", "a", "Label"], ["double", "int8", "string"])])


def test_validate_schemas_rejects_non_numeric_feature():
    with pytest.raises(dl.SchemaError, match="non-numeric"):
        dl.validate_schemas([info("f1", ["a", "b", "Label"], ["int8", "string", "string"])])


def test_validate_schemas_requires_exactly_one_label_column():
    with pytest.raises(dl.SchemaError, match="exactly one label"):
        dl.validate_schemas([info("f1", ["a", "b"], ["int8", "double"])])
    with pytest.raises(dl.SchemaError, match="exactly one label"):
        dl.validate_schemas([info("f1", ["a", "Label", "label"], ["int8", "string", "string"])])


# ------------------------------------------------------------------ fingerprints
def test_fingerprint_is_deterministic_chunk_independent_and_value_based():
    frame = pd.DataFrame({"x": [1.0, 2.0, 1.0, 3.0], "y": [5.0, 6.0, 5.0, 7.0]})
    h1, h2 = dl.fingerprint_rows(frame, ["x", "y"])
    assert h1[0] == h1[2] and h2[0] == h2[2]
    assert len({int(v) for v in h1}) == 3
    c1, c2 = dl.fingerprint_rows(frame, ["x", "y"], chunk_rows=3)
    assert (h1 == c1).all() and (h2 == c2).all()
    assert (h1 != h2).any()  # two independent hashes


def test_fingerprint_normalises_negative_zero_and_nan_and_depends_on_column_order():
    a = pd.DataFrame({"x": [0.0, np.nan], "y": [1.0, 2.0]})
    b = pd.DataFrame({"x": [-0.0, -np.nan], "y": [1.0, 2.0]})
    assert all((u == v).all() for u, v in zip(dl.fingerprint_rows(a, ["x", "y"]), dl.fingerprint_rows(b, ["x", "y"])))
    swapped = pd.DataFrame({"x": [1.0], "y": [2.0]})
    reordered = pd.DataFrame({"x": [2.0], "y": [1.0]})
    assert dl.fingerprint_rows(swapped, ["x", "y"])[0][0] != dl.fingerprint_rows(reordered, ["x", "y"])[0][0]


def test_fingerprint_has_no_collisions_on_a_dense_grid_of_distinct_rows():
    grid = pd.DataFrame(
        [(i, j, k) for i in range(12) for j in range(12) for k in range(12)], columns=["a", "b", "c"]
    ).astype("float64")
    h1, h2 = dl.fingerprint_rows(grid, ["a", "b", "c"])
    assert len(set(h1.tolist())) == len(grid)
    assert len(set(h2.tolist())) == len(grid)


def test_fingerprint_equal_for_same_values_from_different_integer_widths():
    narrow = pd.DataFrame({"x": np.array([1, 2], dtype="int8")}).astype("float64")
    wide = pd.DataFrame({"x": np.array([1, 2], dtype="int64")}).astype("float64")
    assert (dl.fingerprint_rows(narrow, ["x"])[0] == dl.fingerprint_rows(wide, ["x"])[0]).all()


# ------------------------------------------------------------------ audit
def test_audit_frames_exact_counts_on_scenario():
    check_scenario_audit(dl.audit_frames(scenario_frames(), FEATURES))


def test_audit_is_json_serialisable_and_report_is_ascii_safe():
    frames = scenario_frames()
    frames[1][1].loc[1, config.ORIGINAL_LABEL_COLUMN] = "Web Attack \ufffd XSS"
    audit = dl.audit_frames(frames, FEATURES)
    json.dumps(audit)
    report = dl.format_report(audit)
    assert report.isascii()
    assert "ufffd" in report


def test_unrecognised_and_null_labels_are_reported_not_hidden():
    frames = scenario_frames()
    frames[0][1].loc[0, config.ORIGINAL_LABEL_COLUMN] = "Mystery"
    frames[2][1].loc[0, config.ORIGINAL_LABEL_COLUMN] = None
    audit = dl.audit_frames(frames, FEATURES)
    assert audit["files"][0]["labels"]["unknown"] == {"Mystery": 1}
    assert audit["files"][2]["labels"]["null"] == 1
    assert any("unrecognised" in issue for issue in audit["issues"])
    assert any("null" in issue for issue in audit["issues"])
    assert audit["label_mapping"]["Mystery"]["family"] is None


def test_audit_frames_without_data_raises():
    with pytest.raises(dl.DatasetError):
        dl.audit_frames([], FEATURES)


def test_audit_frames_consumes_frames_lazily():
    seen = []

    def generator():
        for name, frame in scenario_frames():
            seen.append(name)
            yield name, frame

    dl.audit_frames(generator(), FEATURES)
    assert seen == ["a.parquet", "b.parquet", "c.parquet"]


# ------------------------------------------------------------------ Parquet (needs pyarrow)
def write_scenario_parquet(directory):
    """Same scenario, stored with different integer widths and a messy label column name."""
    dtypes = [
        {"f2": "int8", "f3": "int16", "const": "int8"},
        {"f2": "int32", "f3": "int16", "const": "int8"},
        {"f2": "int16", "f3": "int32", "const": "int16"},
    ]
    for (name, frame), types in zip(scenario_frames(), dtypes):
        out = frame.drop(columns=[config.SOURCE_FILE_COLUMN]).astype(types)
        label_name = " Label" if name == "a.parquet" else "Label"
        out = out.rename(columns={config.ORIGINAL_LABEL_COLUMN: label_name})
        if name == "b.parquet":
            out[label_name] = out[label_name].astype("category")  # stored as a dictionary
        out.to_parquet(directory / name)


def test_parquet_roundtrip_load_file(tmp_path):
    pytest.importorskip("pyarrow")
    write_scenario_parquet(tmp_path)
    file_info = dl.read_file_info(tmp_path / "a.parquet")
    assert file_info.num_rows == 5
    frame = dl.load_file(tmp_path / "a.parquet")
    assert list(frame.columns) == [*FEATURES, config.ORIGINAL_LABEL_COLUMN, config.SOURCE_FILE_COLUMN]
    assert all(str(frame[c].dtype) == "float64" for c in FEATURES)
    assert frame[config.SOURCE_FILE_COLUMN].eq("a.parquet").all()
    assert frame[config.ORIGINAL_LABEL_COLUMN].tolist() == ["Benign"] * 5
    assert np.isnan(frame.loc[3, "f1"]) and np.isinf(frame.loc[4, "f1"])


def test_parquet_dictionary_label_is_decoded_to_strings(tmp_path):
    pytest.importorskip("pyarrow")
    write_scenario_parquet(tmp_path)
    frame = dl.load_file(tmp_path / "b.parquet")
    assert set(frame[config.ORIGINAL_LABEL_COLUMN]) == {"Benign", "DoS Hulk", "DoS GoldenEye"}


def test_parquet_expected_columns_mismatch_raises(tmp_path):
    pytest.importorskip("pyarrow")
    write_scenario_parquet(tmp_path)
    with pytest.raises(dl.SchemaError, match="expected schema"):
        dl.load_file(tmp_path / "a.parquet", expected_columns=["f1", "f2", "Label"])


def test_parquet_non_numeric_feature_raises(tmp_path):
    pytest.importorskip("pyarrow")
    pd.DataFrame({"f1": ["x", "y"], "Label": ["Benign", "Bot"]}).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(dl.SchemaError, match="non-numeric"):
        dl.load_file(tmp_path / "bad.parquet")


def test_parquet_unreadable_file_is_reported_with_its_name(tmp_path):
    pytest.importorskip("pyarrow")
    (tmp_path / "broken.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(dl.DatasetError, match="broken.parquet"):
        dl.read_file_info(tmp_path / "broken.parquet")


def test_audit_dataset_end_to_end_matches_in_memory_scenario(tmp_path):
    pytest.importorskip("pyarrow")
    write_scenario_parquet(tmp_path)
    audit = dl.audit_dataset(tmp_path)
    check_scenario_audit(audit)
    assert audit["schema"]["n_feature_columns"] == 4
    assert {"f2", "f3", "const"} <= set(audit["schema"]["type_only_differences"])
    assert "NOT verified" in audit["dataset"]["provenance"]
    json.dumps(audit)
    assert dl.format_report(audit).isascii()


def test_cli_end_to_end_exit_code_when_targets_are_not_met(tmp_path):
    pytest.importorskip("pyarrow")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_scenario_parquet(data_dir)
    # Scenario misses the 60k/40k targets on purpose -> issues -> exit code 1.
    assert dl.main(["--data-dir", str(data_dir), "--no-save"]) == 1
