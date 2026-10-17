"""Dataset discovery, loading, label resolution and data-quality audit.

Scope (Milestone 2): READ the CIC-IDS2017 Parquet files and REPORT on them. Nothing
here cleans, deduplicates, samples or trains - those are Milestone 3+. The audit
describes the *source* data, before any removal.

Run the audit from the project root::

    python -m nids.data_loader

Design notes
------------
* Memory: files are processed one at a time. Per file we keep only compact row
  fingerprints (two independent 64-bit hashes), not the data itself.
* Dtypes: the files store the same columns with different integer widths. Features are
  cast to float64 BEFORE hashing so identical values hash identically across files.
  The cast is "safe": an integer that float64 cannot represent exactly raises an error.
* Duplicate key (agreed design): feature values + canonical attack-family label,
  excluding ``source_file``. Hash-based counting uses two differently seeded 64-bit
  hashes (see fingerprint_rows), so the chance of a false collision is negligible.
* PyArrow is imported lazily so that this module imports (and Version 1 logic that does
  not touch Parquet keeps working) even when it is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from nids import config
from nids.utils import ensure_directories, get_logger

logger = get_logger("nids.data_loader")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class DatasetError(Exception):
    """Base class for dataset problems that must stop the pipeline."""


class DatasetNotFoundError(DatasetError):
    """The dataset directory or its Parquet files are missing."""


class SchemaError(DatasetError):
    """Files are structurally incompatible or contain unexpected columns."""


class UnknownLabelError(DatasetError):
    """A label value is null or not covered by config.LABEL_ALIASES."""


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DatasetError(
            "pyarrow is required to read the Parquet files. Install it with: pip install pyarrow"
        ) from exc
    return pa, pq


# ---------------------------------------------------------------------------
# Label registry
# ---------------------------------------------------------------------------
def _norm_label(text: str) -> str:
    return text.strip().casefold()


def _build_label_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for family, aliases in config.LABEL_ALIASES.items():
        for alias in (family, *aliases):
            key = _norm_label(alias)
            if key in lookup and lookup[key] != family:
                raise ValueError(
                    f"config.LABEL_ALIASES maps {alias!r} to both {lookup[key]!r} and {family!r}"
                )
            lookup[key] = family
    return lookup


_LABEL_LOOKUP: dict[str, str] = _build_label_lookup()
FAMILY_NAMES: tuple[str, ...] = tuple(config.LABEL_ALIASES)
_FAMILY_IDS: dict[str, int] = {name: i for i, name in enumerate(FAMILY_NAMES)}
BENIGN_ID: int = _FAMILY_IDS[config.BENIGN_FAMILY]
UNKNOWN_ID: int = -1


def resolve_family(raw_label: object) -> str | None:
    """Canonical family for a raw label, or None if null/unrecognised."""
    if not isinstance(raw_label, str):
        return None
    return _LABEL_LOOKUP.get(_norm_label(raw_label))


@dataclass(frozen=True)
class LabelResolution:
    family_ids: np.ndarray  # int16 per row; UNKNOWN_ID for null/unrecognised
    raw_counts: dict[str, int]  # raw string -> rows (nulls excluded)
    mapping: dict[str, str | None]  # raw string -> canonical family (None = unknown)
    n_null: int

    @property
    def unknown(self) -> dict[str, int]:
        return {raw: n for raw, n in self.raw_counts.items() if self.mapping[raw] is None}


def resolve_labels(raw: pd.Series) -> LabelResolution:
    """Resolve every raw label WITHOUT raising (used by the audit to report problems)."""
    raw_counts: dict[str, int] = {}
    mapping: dict[str, str | None] = {}
    ids: dict[str, int] = {}
    n_null = 0
    for value, count in raw.value_counts(dropna=False).items():
        if not isinstance(value, str):
            n_null += int(count)
            continue
        family = resolve_family(value)
        raw_counts[value] = int(count)
        mapping[value] = family
        ids[value] = UNKNOWN_ID if family is None else _FAMILY_IDS[family]
    family_ids = raw.map(ids).fillna(UNKNOWN_ID).to_numpy(dtype="int16")
    return LabelResolution(family_ids, raw_counts, mapping, n_null)


def raise_if_unknown(resolution: LabelResolution, context: str = "") -> None:
    """Raise UnknownLabelError if any label is null or unrecognised."""
    if not (resolution.unknown or resolution.n_null):
        return
    lines = [f"  {ascii(v)}: {n:,} row(s)" for v, n in sorted(resolution.unknown.items())]
    if resolution.n_null:
        lines.append(f"  <null>: {resolution.n_null:,} row(s)")
    where = f" in {context}" if context else ""
    raise UnknownLabelError(
        f"Unrecognised or missing label value(s){where}:\n"
        + "\n".join(lines)
        + "\nIf a value is a legitimate spelling of a known family, add it to "
        "config.LABEL_ALIASES explicitly; otherwise fix the data."
    )


def canonicalise_labels(raw: pd.Series) -> pd.DataFrame:
    """Return ``attack_family`` and ``is_attack`` for each row (index preserved).

    Raises UnknownLabelError for null or unrecognised labels: nothing is ever silently
    treated as ATTACK.
    """
    resolution = resolve_labels(raw)
    raise_if_unknown(resolution)
    names = np.array(FAMILY_NAMES, dtype=object)[resolution.family_ids]
    return pd.DataFrame(
        {
            config.ATTACK_FAMILY_COLUMN: pd.Series(names, index=raw.index, dtype="object"),
            config.IS_ATTACK_COLUMN: pd.Series(resolution.family_ids != BENIGN_ID, index=raw.index),
        }
    )


# ---------------------------------------------------------------------------
# Discovery and schema handling
# ---------------------------------------------------------------------------
def discover_parquet_files(data_dir: Path | str = config.CIC_IDS2017_DIR) -> list[Path]:
    """Return the dataset's Parquet files in sorted filename order.

    The sorted order is part of the pipeline's reproducibility: it fixes which file
    "wins" when an identical record appears in several files.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DatasetNotFoundError(
            f"Dataset directory not found: {data_dir.resolve()}\n"
            f"Create it and extract the CIC-IDS2017 Parquet files into it "
            f"(expected: {config.CIC_IDS2017_DIR})."
        )
    files = sorted(
        (p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".parquet"),
        key=lambda p: p.name,
    )
    if not files:
        raise DatasetNotFoundError(f"No .parquet files found in {data_dir.resolve()}.")
    return files


def normalise_column_names(columns: Sequence[str]) -> list[str]:
    """Strip surrounding whitespace; fail loudly on empty or colliding names."""
    cleaned = [str(c).strip() for c in columns]
    if any(c == "" for c in cleaned):
        raise SchemaError("Found an empty column name after normalisation.")
    duplicates = sorted({c for c, n in Counter(cleaned).items() if n > 1})
    if duplicates:
        raise SchemaError(f"Duplicate column names after normalisation: {duplicates}")
    return cleaned


def _is_numeric_type(type_str: str) -> bool:
    return type_str.startswith(("int", "uint", "float", "double", "halffloat"))


def _is_string_type(type_str: str) -> bool:
    return "string" in type_str  # string, large_string, dictionary<values=string,...>


@dataclass(frozen=True)
class FileInfo:
    name: str
    path: Path
    num_rows: int
    size_bytes: int
    columns: tuple[str, ...]  # as stored
    types: tuple[str, ...]  # Arrow type strings, same order
    created_by: str


def read_file_info(path: Path | str) -> FileInfo:
    """Read schema and row count from Parquet metadata (no data is loaded)."""
    _, pq = _require_pyarrow()
    path = Path(path)
    try:
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        return FileInfo(
            name=path.name,
            path=path,
            num_rows=int(parquet_file.metadata.num_rows),
            size_bytes=path.stat().st_size,
            columns=tuple(schema.names),
            types=tuple(str(field.type) for field in schema),
            created_by=str(parquet_file.metadata.created_by),
        )
    except Exception as exc:
        raise DatasetError(
            f"Cannot read Parquet metadata of {path.name}: {type(exc).__name__}: {exc}"
        ) from exc


@dataclass(frozen=True)
class SchemaReport:
    reference_file: str
    columns: tuple[str, ...]  # normalised, in the reference file's order
    label_column: str
    feature_columns: tuple[str, ...]
    type_differences: dict[str, dict[str, str]]  # column -> {file: arrow type}


def validate_schemas(infos: Sequence[FileInfo]) -> SchemaReport:
    """Check that the files can be combined safely.

    Raises SchemaError for structural problems (names, order, missing/extra columns,
    label column, non-numeric features). Differences that are ONLY integer/float widths
    are reported, not errors: load_file casts every feature to float64.
    """
    if not infos:
        raise SchemaError("No files to validate.")
    reference = infos[0]
    try:
        ref_columns = normalise_column_names(reference.columns)
    except SchemaError as exc:
        raise SchemaError(f"{reference.name}: {exc}") from exc
    label_matches = [c for c in ref_columns if c.casefold() == "label"]
    if len(label_matches) != 1:
        raise SchemaError(
            f"{reference.name}: expected exactly one label column, found {label_matches or 'none'}."
        )
    label_column = label_matches[0]

    problems: list[str] = []
    types_by_column: dict[str, dict[str, str]] = {c: {} for c in ref_columns}
    for info in infos:
        try:
            columns = normalise_column_names(info.columns)
        except SchemaError as exc:
            problems.append(f"{info.name}: {exc}")
            continue
        if columns != ref_columns:
            missing = [c for c in ref_columns if c not in columns]
            extra = [c for c in columns if c not in ref_columns]
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if extra:
                detail.append(f"unexpected {extra}")
            if not missing and not extra:
                detail.append("same columns in a DIFFERENT ORDER")
            problems.append(f"{info.name}: columns differ from {reference.name} ({'; '.join(detail)})")
            continue
        for column, type_str in zip(columns, info.types):
            types_by_column[column][info.name] = type_str
            if column == label_column:
                if not _is_string_type(type_str):
                    problems.append(f"{info.name}: label column has non-string type {type_str}")
            elif not _is_numeric_type(type_str):
                problems.append(
                    f"{info.name}: non-numeric feature column {column!r} ({type_str}); "
                    "refusing to drop or coerce it silently"
                )
    if problems:
        raise SchemaError("Incompatible Parquet files:\n  " + "\n  ".join(problems))

    type_differences = {
        column: per_file
        for column, per_file in types_by_column.items()
        if len(set(per_file.values())) > 1
    }
    return SchemaReport(
        reference_file=reference.name,
        columns=tuple(ref_columns),
        label_column=label_column,
        feature_columns=tuple(c for c in ref_columns if c != label_column),
        type_differences=type_differences,
    )


# ---------------------------------------------------------------------------
# Loading one file
# ---------------------------------------------------------------------------
def load_file(path: Path | str, expected_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load one Parquet file as a DataFrame.

    Result: normalised feature columns (all float64, in file order), then the metadata
    columns ``original_label`` (raw string, untouched) and ``source_file``. The metadata
    columns are provenance only and must never be used as model features.
    """
    pa, pq = _require_pyarrow()
    import pyarrow.compute as pc

    path = Path(path)
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise DatasetError(f"Cannot read {path.name}: {type(exc).__name__}: {exc}") from exc

    try:
        names = normalise_column_names(table.column_names)
    except SchemaError as exc:
        raise SchemaError(f"{path.name}: {exc}") from exc
    if expected_columns is not None and names != list(expected_columns):
        raise SchemaError(f"{path.name}: columns differ from the expected schema.")
    label_positions = [i for i, n in enumerate(names) if n.casefold() == "label"]
    if len(label_positions) != 1:
        raise SchemaError(f"{path.name}: expected exactly one label column.")
    label_pos = label_positions[0]

    feature_positions = [i for i in range(len(names)) if i != label_pos]
    feature_names = [names[i] for i in feature_positions]
    bad = []
    for i in feature_positions:
        arrow_type = table.schema.field(i).type
        if not (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)):
            bad.append(names[i])
    if bad:
        raise SchemaError(f"{path.name}: non-numeric feature column(s) {bad}.")

    raw_labels = pc.cast(table.column(label_pos), pa.string()).to_pylist()
    features = table.select(feature_positions).rename_columns(feature_names)
    # Safe cast: raises instead of silently rounding an integer float64 cannot hold.
    features = features.cast(pa.schema([pa.field(n, pa.float64()) for n in feature_names]))
    frame = features.to_pandas()
    frame[config.ORIGINAL_LABEL_COLUMN] = pd.Series(raw_labels, dtype="object")
    frame[config.SOURCE_FILE_COLUMN] = path.name
    return frame


# ---------------------------------------------------------------------------
# Audit: row fingerprints
# ---------------------------------------------------------------------------
_HASH_CHUNK_ROWS = 100_000

# Row fingerprints use a splitmix64 chain over the float64 bit patterns, run twice with
# different seeds. (pandas' own hash_pandas_object ignores its key for numeric data, so
# it cannot provide two independent hashes.) This is a well-mixed, non-cryptographic hash:
# for ~2.3 million rows the chance of a false collision in both 64-bit hashes is negligible.
_SEEDS = (np.uint64(0x243F6A8885A308D3), np.uint64(0x13198A2E03707344))
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_SHIFT_30, _SHIFT_27, _SHIFT_31 = np.uint64(30), np.uint64(27), np.uint64(31)
_NAN_BITS = np.array([np.nan], dtype="float64").view(np.uint64)[0]


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):  # uint64 wrap-around is intended
        x = x + _GOLDEN
        x = (x ^ (x >> _SHIFT_30)) * _MIX_1
        x = (x ^ (x >> _SHIFT_27)) * _MIX_2
        return x ^ (x >> _SHIFT_31)


def fingerprint_rows(
    frame: pd.DataFrame, feature_columns: Sequence[str], chunk_rows: int = _HASH_CHUNK_ROWS
) -> tuple[np.ndarray, np.ndarray]:
    """Two independently seeded 64-bit hashes per row over the feature values only.

    A row's hash depends only on that row's values (and the column order), so it is
    independent of chunk size and row order. -0.0 is normalised to +0.0 and every NaN to
    one canonical NaN, so numerically equal rows always hash equally. Work is done in row
    chunks to bound memory.
    """
    columns = list(feature_columns)
    n_rows = len(frame)
    h1 = np.empty(n_rows, dtype=np.uint64)
    h2 = np.empty(n_rows, dtype=np.uint64)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        state1 = np.full(stop - start, _SEEDS[0], dtype=np.uint64)
        state2 = np.full(stop - start, _SEEDS[1], dtype=np.uint64)
        for column in columns:
            values = frame[column].iloc[start:stop].to_numpy(dtype="float64") + 0.0  # new array; -0.0 -> 0.0
            bits = values.view(np.uint64)
            bits[np.isnan(values)] = _NAN_BITS
            state1 = _splitmix64(state1 ^ bits)
            state2 = _splitmix64(state2 ^ bits)
        h1[start:stop] = state1
        h2[start:stop] = state2
    return h1, h2


@dataclass
class _FileResult:
    name: str
    stats: dict
    h1: np.ndarray
    h2: np.ndarray
    family_id: np.ndarray
    valid: np.ndarray  # True where the row has no NaN/inf in any feature


def _audit_one_frame(
    name: str, frame: pd.DataFrame, feature_columns: list[str], ranges: dict[str, list[float]]
) -> _FileResult:
    n_rows = len(frame)
    any_nan = np.zeros(n_rows, dtype=bool)
    any_inf = np.zeros(n_rows, dtype=bool)
    nan_cells: dict[str, int] = {}
    inf_cells: dict[str, int] = {}
    for column in feature_columns:
        values = frame[column].to_numpy(dtype="float64")
        nan_mask = np.isnan(values)
        inf_mask = np.isinf(values)
        if nan_mask.any():
            nan_cells[column] = int(nan_mask.sum())
            any_nan |= nan_mask
        if inf_mask.any():
            inf_cells[column] = int(inf_mask.sum())
            any_inf |= inf_mask
        finite = values[~(nan_mask | inf_mask)]
        if finite.size:
            low, high, count = ranges[column]
            ranges[column] = [min(low, float(finite.min())), max(high, float(finite.max())), count + int(finite.size)]

    valid = ~(any_nan | any_inf)
    resolution = resolve_labels(frame[config.ORIGINAL_LABEL_COLUMN])
    h1, h2 = fingerprint_rows(frame, feature_columns)
    keys = pd.DataFrame({"h1": h1, "h2": h2, "fam": resolution.family_ids})

    family_counts: Counter = Counter()
    for raw, count in resolution.raw_counts.items():
        family = resolution.mapping[raw]
        if family is not None:
            family_counts[family] += count
    stats = {
        "file": name,
        "rows": n_rows,
        "missing_values": {
            "cells": int(sum(nan_cells.values())),
            "rows_affected": int(any_nan.sum()),
            "per_column": nan_cells,
        },
        "infinite_values": {
            "cells": int(sum(inf_cells.values())),
            "rows_affected": int(any_inf.sum()),
            "per_column": inf_cells,
        },
        "rows_with_invalid_values": int((~valid).sum()),
        "duplicates_within_file": {
            "identical_features": int(keys[["h1", "h2"]].duplicated().sum()),
            "identical_features_and_family": int(keys.duplicated().sum()),
        },
        "labels": {
            "raw_counts": dict(resolution.raw_counts),
            "family_counts": dict(family_counts),
            "unknown": resolution.unknown,
            "null": resolution.n_null,
        },
    }
    return _FileResult(name, stats, h1, h2, resolution.family_ids, valid)


# ---------------------------------------------------------------------------
# Audit: cross-file duplicates, conflicts, eligible counts
# ---------------------------------------------------------------------------
_KEY = ["h1", "h2", "fam"]  # agreed duplicate key: features + canonical family
_FEATURE_KEY = ["h1", "h2"]


def _analyse_duplicates(results: list[_FileResult]) -> tuple[dict, dict, dict]:
    names = [r.name for r in results]
    everything = pd.DataFrame(
        {
            "h1": np.concatenate([r.h1 for r in results]),
            "h2": np.concatenate([r.h2 for r in results]),
            "fam": np.concatenate([r.family_id for r in results]),
            "file": np.concatenate([np.full(len(r.h1), i, dtype=np.int16) for i, r in enumerate(results)]),
            "valid": np.concatenate([r.valid for r in results]),
        }
    )
    total = len(everything)

    def breakdown(key: list[str]) -> dict:
        unique_global = len(everything.drop_duplicates(key))
        unique_within_files = len(everything.drop_duplicates([*key, "file"]))
        return {
            "unique": unique_global,
            "duplicate_rows": total - unique_global,
            "within_file_duplicate_rows": total - unique_within_files,
            "cross_file_only_duplicate_rows": unique_within_files - unique_global,
        }

    per_file_unique = everything.drop_duplicates([*_KEY, "file"])
    in_several_files = per_file_unique.duplicated(_KEY, keep=False)
    shared = per_file_unique[in_several_files]
    shared_by_file = shared.groupby("file").size()
    duplicates = {
        "total_rows": total,
        "hashing": "two differently seeded 64-bit splitmix64 hashes over float64 feature values",
        "by_identical_features": breakdown(_FEATURE_KEY),
        "by_features_and_family": breakdown(_KEY),
        "keys_present_in_multiple_files": int(len(shared.drop_duplicates(_KEY))),
        "per_file_keys_also_in_other_files": {
            names[i]: int(shared_by_file.get(i, 0)) for i in range(len(names))
        },
    }

    # Conflicts: identical features, different canonical family.
    unique_keys = everything.drop_duplicates(_KEY)
    known = unique_keys[unique_keys["fam"] != UNKNOWN_ID]
    conflicted = known[known.duplicated(_FEATURE_KEY, keep=False)]
    combos: Counter = Counter()
    n_keys = benign_vs_attack = 0
    if not conflicted.empty:
        conflicted = conflicted.sort_values([*_FEATURE_KEY, "fam"], kind="mergesort")
        current_key = None
        current: list[int] = []

        def flush() -> None:
            nonlocal n_keys, benign_vs_attack
            if not current:
                return
            n_keys += 1
            combos[tuple(current)] += 1
            if BENIGN_ID in current and any(f != BENIGN_ID for f in current):
                benign_vs_attack += 1

        for a, b, f in zip(
            conflicted["h1"].to_numpy(), conflicted["h2"].to_numpy(), conflicted["fam"].to_numpy()
        ):
            key = (int(a), int(b))
            if key != current_key:
                flush()
                current_key, current = key, []
            current.append(int(f))
        flush()
        rows_involved = len(everything.merge(conflicted[_FEATURE_KEY].drop_duplicates(), on=_FEATURE_KEY))
    else:
        rows_involved = 0
    conflicts = {
        "feature_keys_with_conflicting_families": n_keys,
        "rows_involved": rows_involved,
        "benign_vs_attack_keys": benign_vs_attack,
        "attack_family_only_keys": n_keys - benign_vs_attack,
        "top_family_combinations": [
            {"families": [FAMILY_NAMES[i] for i in combo], "feature_keys": count}
            for combo, count in combos.most_common(20)
        ],
    }

    # Eligible records, PROVISIONALLY assuming rows with NaN/inf are dropped (the report's
    # stated policy; the final policy is fixed in Milestone 3), then globally deduplicated.
    # "first file" = attribution to the first file (sorted order) containing the record.
    eligible_rows = everything[everything["valid"]].drop_duplicates(_KEY)
    fam = eligible_rows["fam"].to_numpy()
    files_idx = eligible_rows["file"].to_numpy()
    is_benign = fam == BENIGN_ID
    is_attack = (fam != BENIGN_ID) & (fam != UNKNOWN_ID)
    eligible = {
        "assumption": "rows with NaN/inf dropped, then global dedup on features+family (provisional)",
        "valid_rows_before_dedup": int(everything["valid"].sum()),
        "unique_eligible_total": int(len(eligible_rows)),
        "unique_eligible_benign": int(is_benign.sum()),
        "unique_eligible_attack": int(is_attack.sum()),
        "unique_eligible_unknown_label": int((fam == UNKNOWN_ID).sum()),
        "by_family": {name: int((fam == i).sum()) for i, name in enumerate(FAMILY_NAMES)},
        "by_first_file": {
            names[i]: {
                "benign": int((is_benign & (files_idx == i)).sum()),
                "attack": int((is_attack & (files_idx == i)).sum()),
            }
            for i in range(len(names))
        },
        "targets": {"benign": config.N_BENIGN_SAMPLES, "attack": config.N_ATTACK_SAMPLES},
        "benign_target_feasible": bool(is_benign.sum() >= config.N_BENIGN_SAMPLES),
        "attack_target_feasible": bool(is_attack.sum() >= config.N_ATTACK_SAMPLES),
    }
    return duplicates, conflicts, eligible


# ---------------------------------------------------------------------------
# Audit: assembly
# ---------------------------------------------------------------------------
def audit_frames(frames: Iterable[tuple[str, pd.DataFrame]], feature_columns: Sequence[str]) -> dict:
    """Audit already-loaded frames (as produced by load_file), one at a time."""
    feature_columns = list(feature_columns)
    ranges: dict[str, list[float]] = {c: [float("inf"), float("-inf"), 0] for c in feature_columns}
    results: list[_FileResult] = []
    for name, frame in frames:
        logger.info("Auditing %s (%s rows)", name, f"{len(frame):,}")
        results.append(_audit_one_frame(name, frame, feature_columns, ranges))
        del frame  # release this file's data before the next one is loaded
    if not results:
        raise DatasetError("No data was supplied to audit.")

    duplicates, conflicts, eligible = _analyse_duplicates(results)
    files = [r.stats for r in results]

    raw_summary: dict[str, dict] = {}
    for stats in files:
        for raw, count in stats["labels"]["raw_counts"].items():
            entry = raw_summary.setdefault(raw, {"rows": 0})
            entry["rows"] += count
    for raw, entry in raw_summary.items():
        family = resolve_family(raw)
        entry["family"] = family
        entry["is_attack"] = None if family is None else family != config.BENIGN_FAMILY

    constants = [c for c, (low, high, count) in ranges.items() if count > 0 and low == high]
    no_finite = [c for c, (_, _, count) in ranges.items() if count == 0]
    unknown_total = sum(sum(s["labels"]["unknown"].values()) for s in files)
    null_total = sum(s["labels"]["null"] for s in files)

    issues: list[str] = []
    if unknown_total:
        issues.append(f"{unknown_total:,} row(s) have unrecognised labels (see label mapping).")
    if null_total:
        issues.append(f"{null_total:,} row(s) have null labels.")
    if not eligible["benign_target_feasible"]:
        issues.append(
            f"Only {eligible['unique_eligible_benign']:,} eligible unique benign rows; "
            f"target is {config.N_BENIGN_SAMPLES:,}."
        )
    if not eligible["attack_target_feasible"]:
        issues.append(
            f"Only {eligible['unique_eligible_attack']:,} eligible unique attack rows; "
            f"target is {config.N_ATTACK_SAMPLES:,}."
        )

    return {
        "totals": {
            "files": len(files),
            "rows": sum(s["rows"] for s in files),
            "feature_columns": len(feature_columns),
            "rows_with_invalid_values": sum(s["rows_with_invalid_values"] for s in files),
        },
        "files": files,
        "label_mapping": raw_summary,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "eligible_provisional": eligible,
        "feature_ranges": {
            "constant_columns": constants,
            "columns_without_finite_values": no_finite,
            "note": "Informational only. Whether to drop any column is decided in Milestone 3.",
        },
        "issues": issues,
    }


def audit_dataset(data_dir: Path | str = config.CIC_IDS2017_DIR) -> dict:
    """Full audit of the Parquet files in ``data_dir`` (read-only)."""
    files = discover_parquet_files(data_dir)
    infos = [read_file_info(p) for p in files]
    schema = validate_schemas(infos)

    def frames() -> Iterable[tuple[str, pd.DataFrame]]:
        for info in infos:
            # No local variable holds the frame, so only the consumer keeps it alive.
            yield info.name, load_file(info.path, expected_columns=schema.columns)

    core = audit_frames(frames(), schema.feature_columns)
    for info, stats in zip(infos, core["files"]):
        if stats["rows"] != info.num_rows:
            raise DatasetError(
                f"{info.name}: loaded {stats['rows']:,} rows but Parquet metadata says {info.num_rows:,}."
            )
    return {
        "dataset": {
            "directory": str(Path(data_dir).resolve()),
            "n_files": len(infos),
            "provenance": config.DATASET_PROVENANCE,
            "files": [
                {
                    "name": i.name,
                    "rows": i.num_rows,
                    "size_mb": round(i.size_bytes / 1e6, 1),
                    "columns": len(i.columns),
                    "created_by": i.created_by,
                }
                for i in infos
            ],
        },
        "schema": {
            "reference_file": schema.reference_file,
            "n_columns": len(schema.columns),
            "label_column": schema.label_column,
            "n_feature_columns": len(schema.feature_columns),
            "feature_columns": list(schema.feature_columns),
            "type_only_differences": schema.type_differences,
        },
        **core,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(audit: dict) -> str:
    """Human-readable report. ASCII-safe (labels are shown with escapes)."""
    lines: list[str] = []
    add = lines.append
    bar = "=" * 78

    def section(title: str) -> None:
        add("")
        add(bar)
        add(title)
        add(bar)

    add(bar)
    add("CIC-IDS2017 DATA-QUALITY AUDIT  (source data, BEFORE any cleaning)")
    add(bar)
    dataset = audit.get("dataset")
    if dataset:
        add(f"Directory  : {dataset['directory']}")
        add(f"Provenance : {dataset['provenance']}")
    totals = audit["totals"]
    add(f"Files: {totals['files']}   Rows: {totals['rows']:,}   Feature columns: {totals['feature_columns']}")

    if dataset:
        section("1. FILES (record counts from Parquet metadata)")
        for f in dataset["files"]:
            add(f"{f['name']:<45} rows={f['rows']:>9,}  cols={f['columns']}  {f['size_mb']:>6.1f} MB")

    schema = audit.get("schema")
    if schema:
        section("2. SCHEMA")
        add(f"Reference file : {schema['reference_file']}")
        add(f"Columns        : {schema['n_columns']} ({schema['n_feature_columns']} features + label '{schema['label_column']}')")
        diffs = schema["type_only_differences"]
        if diffs:
            add(f"{len(diffs)} column(s) are stored with different numeric widths across files "
                "(names/order identical). All features are cast to float64 on load:")
            for column, per_file in list(diffs.items())[:15]:
                add(f"  {column}: {sorted(set(per_file.values()))}")
        else:
            add("Column names, order and types are identical across files.")

    section("3. LABEL MAPPING (raw label -> canonical family -> binary target)")
    for raw, entry in sorted(audit["label_mapping"].items(), key=lambda kv: -kv[1]["rows"]):
        if entry["family"] is None:
            target = "UNRECOGNISED"
        else:
            target = "ATTACK(1)" if entry["is_attack"] else "BENIGN(0)"
        add(f"{ascii(raw):<38} -> {str(entry['family']):<28} {target:<12} {entry['rows']:>10,}")

    section("4. PER-FILE DATA QUALITY")
    add(f"{'file':<45}{'rows':>9}{'NaN rows':>10}{'inf rows':>10}{'invalid':>9}{'dup(f)':>9}{'dup(f+fam)':>11}")
    for s in audit["files"]:
        d = s["duplicates_within_file"]
        add(f"{s['file']:<45}{s['rows']:>9,}{s['missing_values']['rows_affected']:>10,}"
            f"{s['infinite_values']['rows_affected']:>10,}{s['rows_with_invalid_values']:>9,}"
            f"{d['identical_features']:>9,}{d['identical_features_and_family']:>11,}")
    add("(dup(f): identical feature values; dup(f+fam): identical features AND family; "
        "counts are rows beyond the first occurrence, within one file)")
    for kind, key in (("Missing", "missing_values"), ("Infinite", "infinite_values")):
        combined: Counter = Counter()
        for s in audit["files"]:
            combined.update(s[key]["per_column"])
        if combined:
            top = ", ".join(f"{c} ({n:,})" for c, n in combined.most_common(8))
            add(f"{kind} cells by column (top 8): {top}")
        else:
            add(f"{kind} cells: none")
    for s in audit["files"]:
        labels = s["labels"]
        if labels["unknown"] or labels["null"]:
            unknown = {ascii(k): v for k, v in labels["unknown"].items()}
            add(f"LABEL PROBLEM in {s['file']}: unknown={unknown} null={labels['null']}")

    dup = audit["duplicates"]
    section("5. DUPLICATES (all files together, before any removal)")
    for title, key in (("Identical feature values", "by_identical_features"),
                       ("Identical features + family (agreed dedup key)", "by_features_and_family")):
        b = dup[key]
        add(f"{title}:")
        add(f"  unique={b['unique']:,}  duplicate rows={b['duplicate_rows']:,} "
            f"(within a file: {b['within_file_duplicate_rows']:,}; only via another file: "
            f"{b['cross_file_only_duplicate_rows']:,})")
    add(f"Dedup keys present in more than one file: {dup['keys_present_in_multiple_files']:,}")

    conf = audit["conflicts"]
    section("6. CONFLICTS: identical features, different family (reported, never auto-resolved)")
    add(f"Feature vectors with conflicting families: {conf['feature_keys_with_conflicting_families']:,} "
        f"({conf['rows_involved']:,} rows)")
    add(f"  benign vs attack: {conf['benign_vs_attack_keys']:,}    attack family vs attack family: "
        f"{conf['attack_family_only_keys']:,}")
    for combo in conf["top_family_combinations"][:10]:
        add(f"  {combo['feature_keys']:>8,}  {' | '.join(combo['families'])}")

    el = audit["eligible_provisional"]
    section("7. ELIGIBLE RECORDS (PROVISIONAL: NaN/inf rows dropped, then global dedup)")
    add(f"Valid rows before dedup : {el['valid_rows_before_dedup']:,}")
    add(f"Unique eligible benign  : {el['unique_eligible_benign']:,}   (target {el['targets']['benign']:,}) "
        f"-> {'feasible' if el['benign_target_feasible'] else 'NOT FEASIBLE'}")
    add(f"Unique eligible attack  : {el['unique_eligible_attack']:,}   (target {el['targets']['attack']:,}) "
        f"-> {'feasible' if el['attack_target_feasible'] else 'NOT FEASIBLE'}")
    add("Unique eligible records by family:")
    for family, count in sorted(el["by_family"].items(), key=lambda kv: -kv[1]):
        add(f"  {family:<28}{count:>10,}")
    add("Unique eligible records by first file containing them (sorted file order):")
    for file_name, c in el["by_first_file"].items():
        add(f"  {file_name:<45} benign={c['benign']:>9,}  attack={c['attack']:>9,}")

    ranges = audit["feature_ranges"]
    section("8. FEATURE RANGES (informational; no column is removed here)")
    add(f"Constant over all finite values ({len(ranges['constant_columns'])}): {ranges['constant_columns']}")
    add(f"No finite values at all ({len(ranges['columns_without_finite_values'])}): {ranges['columns_without_finite_values']}")

    section("ISSUES")
    if audit["issues"]:
        for issue in audit["issues"]:
            add(f"* {issue}")
    else:
        add("None detected by this audit.")
    add("")
    add("Note: counts are for the source files as supplied. Dataset provenance is unknown, so "
        "do not assume these match the report's data preparation.")
    return "\n".join(lines)


def save_audit(
    audit: dict,
    report: str,
    json_path: Path = config.METRICS_DIR / "data_audit.json",
    text_path: Path = config.LOGS_DIR / "data_audit.txt",
) -> tuple[Path, Path]:
    """Write the machine-readable audit and the text report (both git-ignored)."""
    ensure_directories((json_path.parent, text_path.parent))
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    text_path.write_text(report + "\n", encoding="utf-8")
    return json_path, text_path


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    # Windows consoles often use cp1252; show unencodable characters as escapes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        prog="python -m nids.data_loader",
        description="Read-only data-quality audit of the CIC-IDS2017 Parquet files.",
    )
    parser.add_argument("--data-dir", type=Path, default=config.CIC_IDS2017_DIR,
                        help=f"directory with the .parquet files (default: {config.CIC_IDS2017_DIR})")
    parser.add_argument("--no-save", action="store_true", help="print the report without writing files")
    args = parser.parse_args(argv)

    try:
        audit = audit_dataset(args.data_dir)
    except DatasetError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    report = format_report(audit)
    print(report)
    if not args.no_save:
        json_path, text_path = save_audit(audit, report)
        print(f"\nSaved: {json_path}\nSaved: {text_path}")
    return 1 if audit["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
