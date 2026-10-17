"""Exclusion, global deduplication, sampling and splitting (Milestone 3).

Pipeline (agreed design, in this order)::

    load + validate  ->  row index (compact fingerprints, labels, validity)
      -> drop rows with NaN/inf                       (counted, per file)
      -> exclude conflicting feature vectors          (identical features, >1 target class;
                                                       ALL their rows are excluded)
      -> global deduplication                         (key = features + canonical family;
                                                       provenance of every occurrence kept)
      -> sample 60k benign / 40k attack               (seeded, without replacement)
      -> stratified 80:20 split on the binary target
      -> constant-feature selector fitted on TRAIN only
      -> artefacts + metadata

Memory: only one file is in memory at a time. The global analysis works on compact
arrays (about 32 bytes per raw row), never on the full feature matrix. Only the sampled
rows are re-read from the files.

Provenance: a retained observation can occur in several source files. The compact
``provenance_manifest`` records every (observation, file) pair with its occurrence count.
The ``rep_file``/``rep_row`` of an observation is only the physical row used to read its
(identical) feature values; it has no provenance meaning.

Metadata (provenance, family, ids) is stored next to the features for analysis but is never
part of the model feature matrix: use ``split_features_and_metadata``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nids import config
from nids import data_loader as dl
from nids.data_loader import DatasetError, SchemaError
from nids.utils import ensure_directories, get_logger

logger = get_logger("nids.preprocessing")

# Row status codes in the plan (index-aligned).
RETAINED, CONFLICT, INVALID = 0, 1, 2
STATUS_NAMES = ["retained", "excluded_conflict", "excluded_invalid"]

# Metadata columns stored beside the features. NEVER used as model input.
SAMPLE_METADATA_COLUMNS: tuple[str, ...] = (
    "obs_id",
    config.ATTACK_FAMILY_COLUMN,
    config.ORIGINAL_LABEL_COLUMN,
    "target",
    "n_source_files",
    "source_files",
    "occurrences",
)

SCALED_MODELS = ("logistic_regression", "svm")
UNSCALED_MODELS = ("decision_tree", "random_forest")


class InfeasibleSampleError(DatasetError):
    """The requested sample sizes cannot be met after exclusions and deduplication."""


@dataclass(frozen=True)
class PreparationParams:
    seed: int = config.RANDOM_SEED
    n_benign: int = config.N_BENIGN_SAMPLES
    n_attack: int = config.N_ATTACK_SAMPLES
    test_size: float = config.TEST_SIZE

    def __post_init__(self) -> None:
        if self.n_benign < 1 or self.n_attack < 1:
            raise ValueError("n_benign and n_attack must be at least 1.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be strictly between 0 and 1.")


# ---------------------------------------------------------------------------
# Invalid numerical values (also used by the prediction pipeline later)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InvalidValueReport:
    mask: np.ndarray  # True where the row has NaN or +/-inf in any feature
    nan_cells: int
    inf_cells: int
    rows_with_nan: int
    rows_with_inf: int

    @property
    def rows_invalid(self) -> int:
        return int(self.mask.sum())


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str = "feature") -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required {what} column(s): {missing}")


def find_invalid_rows(frame: pd.DataFrame, columns: Sequence[str]) -> InvalidValueReport:
    """Rows containing NaN or infinite values in any of ``columns`` (no 2-D copy is made)."""
    _require_columns(frame, columns)
    has_nan = np.zeros(len(frame), dtype=bool)
    has_inf = np.zeros(len(frame), dtype=bool)
    nan_cells = inf_cells = 0
    for column in columns:
        values = frame[column].to_numpy(dtype="float64")
        nan, inf = np.isnan(values), np.isinf(values)
        nan_cells += int(nan.sum())
        inf_cells += int(inf.sum())
        has_nan |= nan
        has_inf |= inf
    return InvalidValueReport(has_nan | has_inf, nan_cells, inf_cells, int(has_nan.sum()), int(has_inf.sum()))


def coerce_features_to_float(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert feature columns to float64; unparseable text becomes NaN and is counted.

    Returns the converted copy and ``{column: number of values that could not be parsed}``.
    The NaN values are then caught by ``find_invalid_rows``; nothing is silently imputed.
    """
    _require_columns(frame, columns)
    out = frame.loc[:, list(columns)].copy()
    unparseable: dict[str, int] = {}
    for column in columns:
        original = out[column]
        converted = pd.to_numeric(original, errors="coerce").astype("float64")
        newly_missing = int((converted.isna() & original.notna()).sum())
        if newly_missing:
            unparseable[column] = newly_missing
        out[column] = converted
    return out, unparseable


def align_feature_columns(
    frame: pd.DataFrame, expected_columns: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Return the expected columns, in the expected ORDER, plus any unexpected extras.

    Column names are normalised (whitespace stripped) before matching. Missing required
    columns raise SchemaError; a model must never receive columns in the wrong order.
    """
    names = dl.normalise_column_names(list(frame.columns))
    renamed = frame.copy()
    renamed.columns = names
    missing = [c for c in expected_columns if c not in names]
    if missing:
        raise SchemaError(f"Missing required feature column(s): {missing}")
    extras = [c for c in names if c not in set(expected_columns)]
    return renamed.loc[:, list(expected_columns)].copy(), extras


# ---------------------------------------------------------------------------
# Frame source (real Parquet files, or in-memory frames in tests)
# ---------------------------------------------------------------------------
@dataclass
class FrameSource:
    names: list[str]  # in processing order (sorted file names for real data)
    feature_columns: list[str]
    loader: Callable[[str], pd.DataFrame]  # name -> frame in data_loader.load_file format

    def frames(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for name in self.names:
            yield name, self.loader(name)  # no local reference, so the consumer controls lifetime

    def load(self, name: str) -> pd.DataFrame:
        return self.loader(name)


def parquet_source(data_dir: Path | str = config.CIC_IDS2017_DIR):
    """FrameSource over the Parquet files, plus file infos and the validated schema."""
    files = dl.discover_parquet_files(data_dir)
    infos = [dl.read_file_info(p) for p in files]
    schema = dl.validate_schemas(infos)
    paths = {info.name: info.path for info in infos}
    source = FrameSource(
        names=[info.name for info in infos],
        feature_columns=list(schema.feature_columns),
        loader=lambda name: dl.load_file(paths[name], expected_columns=schema.columns),
    )
    return source, infos, schema


# ---------------------------------------------------------------------------
# Step 1: compact row index
# ---------------------------------------------------------------------------
def build_row_index(source: FrameSource) -> tuple[pd.DataFrame, list[dict]]:
    """One pass over the files: per row keep file, row position, 2 hashes, family id, validity."""
    parts: list[pd.DataFrame] = []
    file_stats: list[dict] = []
    needed = [*source.feature_columns, config.ORIGINAL_LABEL_COLUMN]
    for file_idx, (name, frame) in enumerate(source.frames()):
        _require_columns(frame, needed, what="data")
        resolution = dl.resolve_labels(frame[config.ORIGINAL_LABEL_COLUMN])
        dl.raise_if_unknown(resolution, context=name)
        invalid = find_invalid_rows(frame, source.feature_columns)
        h1, h2 = dl.fingerprint_rows(frame, source.feature_columns)
        n_rows = len(frame)
        logger.info("Indexed %s (%s rows)", name, f"{n_rows:,}")
        parts.append(
            pd.DataFrame(
                {
                    "file": np.full(n_rows, file_idx, dtype=np.int16),
                    "row": np.arange(n_rows, dtype=np.int32),
                    "h1": h1,
                    "h2": h2,
                    "fam": resolution.family_ids.astype(np.int16),
                    "valid": ~invalid.mask,
                }
            )
        )
        file_stats.append(
            {
                "file": name,
                "rows": n_rows,
                "rows_with_nan": invalid.rows_with_nan,
                "rows_with_inf": invalid.rows_with_inf,
                "rows_invalid": invalid.rows_invalid,
                "nan_cells": invalid.nan_cells,
                "inf_cells": invalid.inf_cells,
            }
        )
        del frame  # release this file before the next is loaded
    if not parts:
        raise DatasetError("No data to index.")
    return pd.concat(parts, ignore_index=True), file_stats


# ---------------------------------------------------------------------------
# Step 2: exclusions and global deduplication (pure functions on the index)
# ---------------------------------------------------------------------------
def _sorted_groups(keys: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by ``keys`` (first key is primary) with a STABLE sort and find equal-key groups.

    Returns (order, starts, group_of_sorted): ``order`` sorts the rows, ``starts`` are the
    positions in sorted order where a group begins, ``group_of_sorted[i]`` is the group of
    the i-th sorted row. Ties keep their original order, so the first row of a group is its
    earliest occurrence.
    """
    n = len(keys[0])
    order = np.lexsort(tuple(reversed(keys)))
    boundary = np.zeros(n, dtype=bool)
    if n:
        boundary[0] = True
        for key in keys:
            sorted_key = key[order]
            boundary[1:] |= sorted_key[1:] != sorted_key[:-1]
    return order, np.flatnonzero(boundary), np.cumsum(boundary) - 1


def observation_id(h1: int, h2: int, fam: int) -> str:
    """Stable id of a unique observation: the feature fingerprint plus the family id."""
    return f"{int(h1):016x}{int(h2):016x}-{int(fam):02d}"


def _family_counts(fam: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(fam, dtype=np.int64), minlength=len(dl.FAMILY_NAMES))
    return {name: int(c) for name, c in zip(dl.FAMILY_NAMES, counts)}


@dataclass
class ExclusionPlan:
    status: np.ndarray  # int8 per index row: RETAINED / CONFLICT / INVALID
    eligible: pd.DataFrame  # one row per unique retained observation (canonical order)
    retained_long: pd.DataFrame  # (obs, file, occ): provenance of retained observations
    manifest: pd.DataFrame  # compact provenance manifest incl. excluded observations
    excluded_conflicts: pd.DataFrame  # row-level audit of conflict exclusions
    stats: dict = field(default_factory=dict)


_ELIGIBLE_COLUMNS = ["obs", "h1", "h2", "fam", "is_attack", "occurrences", "rep_file", "rep_row", "n_files"]


def plan_exclusions_and_dedup(index: pd.DataFrame, file_names: Sequence[str]) -> ExclusionPlan:
    """Apply invalid-value removal, conflict exclusion and global deduplication.

    Conflict rule: a feature vector (identical model-relevant values) that occurs with more
    than one binary target class has ALL its rows excluded - neither the benign nor the
    attack copy is kept. Identical features with different *attack families* (both attack)
    are not a binary conflict; they stay as distinct observations (reported separately).
    """
    n_total = len(index)
    file_arr = index["file"].to_numpy()
    row_arr = index["row"].to_numpy()
    fam_all = index["fam"].to_numpy()
    valid = index["valid"].to_numpy()
    if (fam_all == dl.UNKNOWN_ID).any():
        raise ValueError("Unknown labels must be rejected before planning exclusions.")
    key = (file_arr.astype(np.int64) << 32) | row_arr.astype(np.int64)
    if n_total > 1 and not (key[1:] > key[:-1]).all():
        raise ValueError("The row index must be ordered by (file, row).")
    n_files = len(file_names)

    status = np.zeros(n_total, dtype=np.int8)
    status[~valid] = INVALID

    # --- conflicting feature vectors (valid rows only) -------------------------------
    n_conflict_vectors = attack_family_only = 0
    valid_pos = np.flatnonzero(valid)
    if valid_pos.size:
        h1, h2, fam = index["h1"].to_numpy()[valid_pos], index["h2"].to_numpy()[valid_pos], fam_all[valid_pos]
        order, starts, gid = _sorted_groups([h1, h2])
        cls = (fam != dl.BENIGN_ID).astype(np.int8)[order]
        fam_sorted = fam[order]
        cls_min, cls_max = np.minimum.reduceat(cls, starts), np.maximum.reduceat(cls, starts)
        fam_min, fam_max = np.minimum.reduceat(fam_sorted, starts), np.maximum.reduceat(fam_sorted, starts)
        conflict_vector = cls_min != cls_max
        n_conflict_vectors = int(conflict_vector.sum())
        attack_family_only = int(((cls_min == 1) & (cls_max == 1) & (fam_min != fam_max)).sum())
        row_conflict = np.zeros(valid_pos.size, dtype=bool)
        row_conflict[order] = conflict_vector[gid]
        status[valid_pos[row_conflict]] = CONFLICT

    # --- global deduplication of the retained rows ---------------------------------------
    ret_pos = np.flatnonzero(status == RETAINED)
    if ret_pos.size:
        r_h1, r_h2 = index["h1"].to_numpy()[ret_pos], index["h2"].to_numpy()[ret_pos]
        r_fam, r_file, r_row = fam_all[ret_pos], file_arr[ret_pos], row_arr[ret_pos]
        order, starts, gid = _sorted_groups([r_h1, r_h2, r_fam])  # canonical: (h1, h2, fam)
        n_obs = int(starts.size)
        first = order[starts]  # earliest (file, row) occurrence of each observation
        composite = gid.astype(np.int64) * n_files + r_file[order].astype(np.int64)
        unique_pairs, pair_counts = np.unique(composite, return_counts=True)
        obs_of_pair, file_of_pair = unique_pairs // n_files, unique_pairs % n_files
        eligible = pd.DataFrame(
            {
                "obs": np.arange(n_obs, dtype=np.int64),
                "h1": r_h1[first],
                "h2": r_h2[first],
                "fam": r_fam[first].astype(np.int16),
                "is_attack": r_fam[first] != dl.BENIGN_ID,
                "occurrences": np.diff(np.append(starts, order.size)).astype(np.int64),
                "rep_file": r_file[first].astype(np.int16),
                "rep_row": r_row[first].astype(np.int32),
                "n_files": np.bincount(obs_of_pair, minlength=n_obs).astype(np.int16),
            }
        )
        retained_long = pd.DataFrame(
            {"obs": obs_of_pair.astype(np.int64), "file": file_of_pair.astype(np.int16), "occ": pair_counts.astype(np.int64)}
        )
        retained_manifest = pd.DataFrame(
            {
                "h1": eligible["h1"].to_numpy()[obs_of_pair],
                "h2": eligible["h2"].to_numpy()[obs_of_pair],
                "fam": eligible["fam"].to_numpy()[obs_of_pair],
                "file": file_of_pair.astype(np.int16),
                "occurrences": pair_counts.astype(np.int64),
                "status": np.full(unique_pairs.size, RETAINED, dtype=np.int8),
            }
        )
    else:
        n_obs = 0
        eligible = pd.DataFrame({c: pd.Series(dtype="int64") for c in _ELIGIBLE_COLUMNS})
        eligible["is_attack"] = eligible["is_attack"].astype(bool)
        retained_long = pd.DataFrame({"obs": pd.Series(dtype="int64"), "file": pd.Series(dtype="int16"), "occ": pd.Series(dtype="int64")})
        retained_manifest = pd.DataFrame(
            {c: pd.Series(dtype=t) for c, t in [("h1", "uint64"), ("h2", "uint64"), ("fam", "int16"), ("file", "int16"), ("occurrences", "int64"), ("status", "int8")]}
        )

    # --- compact manifest incl. excluded observations -------------------------------------
    manifest_parts = [retained_manifest]
    for code in (CONFLICT, INVALID):
        pos = np.flatnonzero(status == code)
        if pos.size:
            sub = pd.DataFrame(
                {"h1": index["h1"].to_numpy()[pos], "h2": index["h2"].to_numpy()[pos], "fam": fam_all[pos], "file": file_arr[pos]}
            )
            grouped = sub.groupby(["h1", "h2", "fam", "file"], sort=True).size().reset_index(name="occurrences")
            grouped["status"] = np.int8(code)
            manifest_parts.append(grouped)
    raw_manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest = pd.DataFrame(
        {
            "h1": raw_manifest["h1"].to_numpy(),
            "h2": raw_manifest["h2"].to_numpy(),
            "attack_family": pd.Categorical.from_codes(raw_manifest["fam"].to_numpy().astype(np.int64), categories=list(dl.FAMILY_NAMES)),
            "source_file": pd.Categorical.from_codes(raw_manifest["file"].to_numpy().astype(np.int64), categories=list(file_names)),
            "occurrences": raw_manifest["occurrences"].to_numpy().astype(np.int64),
            "status": pd.Categorical.from_codes(raw_manifest["status"].to_numpy().astype(np.int64), categories=STATUS_NAMES),
        }
    )

    # --- row-level audit of the conflict exclusions (transparent and reversible) -------------
    conf_pos = np.flatnonzero(status == CONFLICT)
    conf_fam, conf_file = fam_all[conf_pos], file_arr[conf_pos]
    excluded_conflicts = pd.DataFrame(
        {
            "source_file": np.array(list(file_names), dtype=object)[conf_file] if conf_pos.size else np.array([], dtype=object),
            "row_position": row_arr[conf_pos],
            "feature_vector_id": [f"{int(a):016x}{int(b):016x}" for a, b in zip(index["h1"].to_numpy()[conf_pos], index["h2"].to_numpy()[conf_pos])],
            "attack_family": np.array(dl.FAMILY_NAMES, dtype=object)[conf_fam] if conf_pos.size else np.array([], dtype=object),
            "is_attack": conf_fam != dl.BENIGN_ID,
        }
    )

    # --- statistics -------------------------------------------------------------------------
    valid_family_counts = _family_counts(fam_all[valid])
    conflict_family_counts = _family_counts(conf_fam)
    exclusion_rate = {
        name: {
            "valid_rows": valid_family_counts[name],
            "excluded_rows": count,
            "fraction": (count / valid_family_counts[name]) if valid_family_counts[name] else 0.0,
        }
        for name, count in conflict_family_counts.items()
        if count > 0
    }
    ret_attack_rows = int((fam_all[ret_pos] != dl.BENIGN_ID).sum())
    ret_benign_rows = int(ret_pos.size - ret_attack_rows)
    uniq_attack = int(eligible["is_attack"].sum())
    uniq_benign = n_obs - uniq_attack
    invalid_rows = int((status == INVALID).sum())
    stats = {
        "raw_rows": n_total,
        "invalid_values": {"rows_removed": invalid_rows},
        "conflicts": {
            "feature_vectors": n_conflict_vectors,
            "rows_excluded": int(conf_pos.size),
            "benign_rows_excluded": int((conf_fam == dl.BENIGN_ID).sum()),
            "attack_rows_excluded": int((conf_fam != dl.BENIGN_ID).sum()),
            "family_counts_excluded": {k: v for k, v in conflict_family_counts.items() if v},
            "exclusion_rate_by_family": exclusion_rate,
            "rows_by_source_file": {
                name: {
                    "benign": int(((conf_file == i) & (conf_fam == dl.BENIGN_ID)).sum()),
                    "attack": int(((conf_file == i) & (conf_fam != dl.BENIGN_ID)).sum()),
                }
                for i, name in enumerate(file_names)
            },
            "attack_family_only_vectors_retained": attack_family_only,
        },
        "deduplication": {
            "key": "model feature values + canonical attack family (excludes source_file and row position)",
            "rows_before": int(ret_pos.size),
            "unique_observations": n_obs,
            "rows_removed": int(ret_pos.size) - n_obs,
            "benign_rows_removed": ret_benign_rows - uniq_benign,
            "attack_rows_removed": ret_attack_rows - uniq_attack,
            "observations_spanning_multiple_files": int((eligible["n_files"] > 1).sum()),
        },
        "eligible": {
            "total": n_obs,
            "benign": uniq_benign,
            "attack": uniq_attack,
            "by_family": _family_counts(eligible["fam"].to_numpy()),
        },
    }
    stats["accounting_ok"] = (
        n_total == invalid_rows + int(conf_pos.size) + int(ret_pos.size)
        and int(ret_pos.size) == stats["deduplication"]["rows_removed"] + n_obs
    )
    if not stats["accounting_ok"]:  # pragma: no cover - would indicate a bug
        raise DatasetError("Internal accounting error: removal counts do not add up.")
    return ExclusionPlan(status, eligible, retained_long, manifest, excluded_conflicts, stats)


def membership_summary(
    retained_long: pd.DataFrame,
    eligible: pd.DataFrame,
    file_names: Sequence[str],
    obs_subset: Sequence[int] | None = None,
) -> dict:
    """Per-file counts for retained observations, keeping the three notions apart.

    * ``occurrences``: original rows in the file that belong to these observations.
    * ``unique_observations``: observations with at least one occurrence in the file.
    * ``exclusive_to_file`` / ``shared_with_other_files``: split of the previous number by
      whether the observation occurs in only that file or also elsewhere.

    One observation can belong to several files, so per-file ``unique_observations`` OVERLAP
    and must not be summed. ``n_files`` is the observation's membership in the full
    retained dataset (not restricted to ``obs_subset``).
    """
    long = retained_long
    if obs_subset is not None:
        long = long[long["obs"].isin(np.asarray(obs_subset))]
    obs_idx = long["obs"].to_numpy()
    long = long.assign(
        n_files=eligible["n_files"].to_numpy()[obs_idx],
        is_attack=eligible["is_attack"].to_numpy()[obs_idx],
    )
    out: dict[str, dict] = {}
    for label, selector in (("all", None), ("benign", False), ("attack", True)):
        sub = long if selector is None else long[long["is_attack"] == selector]
        grouped = (
            sub.assign(exclusive=(sub["n_files"] == 1))
            .groupby("file")
            .agg(occurrences=("occ", "sum"), unique_observations=("obs", "size"), exclusive_to_file=("exclusive", "sum"))
        )
        per_file = {}
        for i, name in enumerate(file_names):
            if i in grouped.index:
                row = grouped.loc[i]
                unique, exclusive = int(row["unique_observations"]), int(row["exclusive_to_file"])
                per_file[name] = {
                    "occurrences": int(row["occurrences"]),
                    "unique_observations": unique,
                    "exclusive_to_file": exclusive,
                    "shared_with_other_files": unique - exclusive,
                }
            else:
                per_file[name] = {"occurrences": 0, "unique_observations": 0, "exclusive_to_file": 0, "shared_with_other_files": 0}
        distinct = sub.drop_duplicates("obs")
        out[label] = {
            "per_file": per_file,
            "unique_observations_total": int(len(distinct)),
            "unique_observations_spanning_multiple_files": int((distinct["n_files"] > 1).sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Step 3: sampling and splitting
# ---------------------------------------------------------------------------
def sample_observations(eligible: pd.DataFrame, params: PreparationParams) -> pd.DataFrame:
    """Seeded sample WITHOUT replacement from the eligible unique observations.

    Benign and attack are drawn from their full pools: no per-file quotas and no
    per-family quotas, so the natural attack-family mix is preserved. The pools are in
    canonical (fingerprint) order, so the draw does not depend on file order.
    """
    benign_pool = np.flatnonzero(~eligible["is_attack"].to_numpy())
    attack_pool = np.flatnonzero(eligible["is_attack"].to_numpy())
    if benign_pool.size < params.n_benign or attack_pool.size < params.n_attack:
        raise InfeasibleSampleError(
            "Requested sample sizes are not feasible after exclusions and deduplication: "
            f"benign eligible={benign_pool.size:,} (requested {params.n_benign:,}), "
            f"attack eligible={attack_pool.size:,} (requested {params.n_attack:,}). "
            "Nothing was reduced or fabricated; decide how to proceed."
        )
    benign_rng, attack_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(params.seed).spawn(2))
    picked_benign = benign_pool[benign_rng.choice(benign_pool.size, size=params.n_benign, replace=False)]
    picked_attack = attack_pool[attack_rng.choice(attack_pool.size, size=params.n_attack, replace=False)]
    return eligible.iloc[np.sort(np.concatenate([picked_benign, picked_attack]))].reset_index(drop=True)


def materialise_sample(
    source: FrameSource, selected: pd.DataFrame, file_names: Sequence[str], retained_long: pd.DataFrame
) -> pd.DataFrame:
    """Read the selected observations' feature values and attach metadata columns."""
    features = list(source.feature_columns)
    clash = sorted(set(features) & set(SAMPLE_METADATA_COLUMNS))
    if clash:
        raise SchemaError(f"Feature column name(s) collide with metadata column names: {clash}")

    pieces = []
    for file_idx in np.unique(selected["rep_file"].to_numpy()):
        part = selected[selected["rep_file"] == file_idx]
        name = file_names[int(file_idx)]
        frame = source.load(name)
        rows = frame.iloc[part["rep_row"].to_numpy()]
        h1, h2 = dl.fingerprint_rows(rows, features)
        if not (np.array_equal(h1, part["h1"].to_numpy()) and np.array_equal(h2, part["h2"].to_numpy())):
            raise DatasetError(f"{name}: rows no longer match the index fingerprints; the source files changed.")
        piece = rows[features].reset_index(drop=True)
        piece[config.ORIGINAL_LABEL_COLUMN] = rows[config.ORIGINAL_LABEL_COLUMN].to_numpy()
        piece["obs"] = part["obs"].to_numpy()
        pieces.append(piece)
        del frame, rows
    sample = pd.concat(pieces, ignore_index=True).set_index("obs").loc[selected["obs"].to_numpy()].reset_index()

    sample["obs_id"] = [observation_id(a, b, f) for a, b, f in zip(selected["h1"], selected["h2"], selected["fam"])]
    sample[config.ATTACK_FAMILY_COLUMN] = np.array(dl.FAMILY_NAMES, dtype=object)[selected["fam"].to_numpy()]
    sample["target"] = selected["is_attack"].to_numpy().astype(np.int8)
    sample["n_source_files"] = selected["n_files"].to_numpy().astype(np.int16)
    sample["occurrences"] = selected["occurrences"].to_numpy().astype(np.int64)
    membership = retained_long[retained_long["obs"].isin(selected["obs"].to_numpy())]
    named = pd.Series(np.array(list(file_names), dtype=object)[membership["file"].to_numpy()], index=membership["obs"].to_numpy())
    joined = named.groupby(level=0).agg("|".join)
    sample["source_files"] = joined.reindex(selected["obs"].to_numpy()).to_numpy()
    return sample[["obs", *features, *SAMPLE_METADATA_COLUMNS]]


def split_sample(sample: pd.DataFrame, params: PreparationParams) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test positions (on the binary target), fixed seed."""
    try:
        train_pos, test_pos = train_test_split(
            np.arange(len(sample)),
            test_size=params.test_size,
            stratify=sample["target"].to_numpy(),
            random_state=params.seed,
        )
    except ValueError as exc:
        raise DatasetError(f"Cannot create the stratified split: {exc}") from exc
    return np.sort(train_pos), np.sort(test_pos)


def split_features_and_metadata(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(X, y, metadata). X holds ONLY the feature columns, in schema order."""
    _require_columns(frame, [*feature_columns, "target"], what="data")
    leaked = sorted(set(feature_columns) & set(SAMPLE_METADATA_COLUMNS))
    if leaked:
        raise SchemaError(f"Metadata column(s) present in the feature list: {leaked}")
    meta_columns = [c for c in SAMPLE_METADATA_COLUMNS if c in frame.columns]
    return frame.loc[:, list(feature_columns)], frame["target"].astype("int8"), frame.loc[:, meta_columns]


# ---------------------------------------------------------------------------
# Step 4: model-specific preprocessing, fitted on the training split only
# ---------------------------------------------------------------------------
def build_preprocessor(scale: bool) -> Pipeline:
    """Constant-feature removal, plus standardisation when ``scale`` is True.

    Nothing is fitted here. Fit on the TRAINING features only; the test set must never
    influence which features are removed or the scaling parameters. Output keeps pandas
    column names, so feature names and order stay traceable.
    """
    steps = [("drop_constant", VarianceThreshold(threshold=0.0))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")


def preprocessor_for_model(model_name: str) -> Pipeline:
    """Logistic Regression / SVM get scaling; tree models use unscaled features."""
    if model_name in SCALED_MODELS:
        return build_preprocessor(scale=True)
    if model_name in UNSCALED_MODELS:
        return build_preprocessor(scale=False)
    raise ValueError(f"Unknown model {model_name!r}; expected one of {SCALED_MODELS + UNSCALED_MODELS}.")


def fit_feature_schema(train: pd.DataFrame, test: pd.DataFrame, feature_columns: Sequence[str]) -> dict:
    """Fit the constant-feature selector on TRAIN only and verify name/order consistency."""
    features = list(feature_columns)
    x_train, x_test = train.loc[:, features], test.loc[:, features]
    pipeline = build_preprocessor(scale=False).fit(x_train)
    retained = [str(c) for c in pipeline.named_steps["drop_constant"].get_feature_names_out()]
    removed = [c for c in features if c not in set(retained)]

    # Independent check of the selector against a direct min == max computation on train.
    values = x_train.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise DatasetError("Training features contain NaN/inf; invalid rows should have been removed.")
    constant_in_train = [c for c, lo, hi in zip(features, values.min(axis=0), values.max(axis=0)) if lo == hi]
    if constant_in_train != removed:
        raise DatasetError(f"Selector disagrees with the direct constant check: {removed} vs {constant_in_train}")

    train_out, test_out = pipeline.transform(x_train), pipeline.transform(x_test)
    scaled = build_preprocessor(scale=True).fit(x_train)
    scaled_names = [str(c) for c in scaled.transform(x_test).columns]
    if not (list(train_out.columns) == list(test_out.columns) == retained == scaled_names):
        raise DatasetError("Feature names/order differ between training, test and scaled outputs.")

    reported = config.REPORTED_FEATURE_COUNT
    return {
        "original_features": features,
        "retained_features": retained,
        "removed_constant_features": removed,
        "n_original": len(features),
        "n_retained": len(retained),
        "selector": "VarianceThreshold(threshold=0.0), fitted on the training split only",
        "reported_feature_count": reported,
        "matches_reported_count": len(retained) == reported,
        "discrepancy_note": (
            f"Retained {len(retained)} features after training-fitted constant removal; the report states {reported}."
            if len(retained) != reported
            else f"Retained {len(retained)} features, equal to the report's {reported}."
        ),
        "scaled_models": list(SCALED_MODELS),
        "unscaled_models": list(UNSCALED_MODELS),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class PreparationResult:
    params: PreparationParams
    train: pd.DataFrame
    test: pd.DataFrame
    feature_schema: dict
    metadata: dict
    row_index: pd.DataFrame
    manifest: pd.DataFrame
    excluded_conflicts: pd.DataFrame


def _split_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "benign": int((frame["target"] == 0).sum()),
        "attack": int((frame["target"] == 1).sum()),
        "family_counts": {k: int(v) for k, v in frame[config.ATTACK_FAMILY_COLUMN].value_counts().sort_index().items()},
    }


def _family_dict(names: pd.Series) -> dict[str, int]:
    counts = names.value_counts()
    return {name: int(counts.get(name, 0)) for name in dl.FAMILY_NAMES}


def prepare_from_source(
    source: FrameSource,
    params: PreparationParams = PreparationParams(),
    source_details: list[dict] | None = None,
) -> PreparationResult:
    file_names = list(source.names)
    index, file_stats = build_row_index(source)
    plan = plan_exclusions_and_dedup(index, file_names)
    plan.stats["invalid_values"].update(
        {
            "by_file": file_stats,
            "rows_with_nan": sum(s["rows_with_nan"] for s in file_stats),
            "rows_with_inf": sum(s["rows_with_inf"] for s in file_stats),
        }
    )
    try:
        selected = sample_observations(plan.eligible, params)
    except InfeasibleSampleError as exc:
        raise InfeasibleSampleError(f"{exc}\nPipeline statistics: {json.dumps(plan.stats, default=_json_default)}") from exc

    sample = materialise_sample(source, selected, file_names, plan.retained_long)
    train_pos, test_pos = split_sample(sample, params)
    train, test = sample.iloc[train_pos].reset_index(drop=True), sample.iloc[test_pos].reset_index(drop=True)
    if set(train["obs"]) & set(test["obs"]):  # pragma: no cover - dedup makes this impossible
        raise DatasetError("The same unique observation appears in both train and test.")
    schema = fit_feature_schema(train, test, source.feature_columns)

    membership = {
        "eligible": membership_summary(plan.retained_long, plan.eligible, file_names),
        "sample": membership_summary(plan.retained_long, plan.eligible, file_names, sample["obs"].to_numpy()),
        "train": membership_summary(plan.retained_long, plan.eligible, file_names, train["obs"].to_numpy()),
        "test": membership_summary(plan.retained_long, plan.eligible, file_names, test["obs"].to_numpy()),
    }
    test_family = _family_dict(test[config.ATTACK_FAMILY_COLUMN])
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "parameters": {
            "seed": params.seed,
            "n_benign": params.n_benign,
            "n_attack": params.n_attack,
            "test_size": params.test_size,
            "sampling": "seeded, without replacement, from eligible unique observations; no per-file or per-family quotas",
            "split": "stratified on the binary target, same seed",
            "order_of_operations": "invalid-value removal -> conflict exclusion -> global deduplication -> sampling -> split",
            "hashing": "two differently seeded 64-bit splitmix64 fingerprints over float64 feature values",
        },
        "provenance": dict(config.DATASET_PROVENANCE_DETAIL),
        "source_files": source_details if source_details is not None else [{"name": n} for n in file_names],
        "raw": {"files": len(file_names), "rows": plan.stats["raw_rows"]},
        "invalid_values": plan.stats["invalid_values"],
        "conflicts": plan.stats["conflicts"],
        "deduplication": plan.stats["deduplication"],
        "eligible": plan.stats["eligible"],
        "sample": {
            "total": int(len(sample)),
            "benign": int((sample["target"] == 0).sum()),
            "attack": int((sample["target"] == 1).sum()),
            "family_counts": _family_dict(sample[config.ATTACK_FAMILY_COLUMN]),
        },
        "split": {
            "train": _split_summary(train),
            "test": _split_summary(test),
            "family_test_reliability": {
                name: {
                    "test_records": count,
                    "sufficient_for_family_metrics": count >= config.MIN_FAMILY_TEST_RECORDS,
                }
                for name, count in test_family.items()
                if name != config.BENIGN_FAMILY
            },
            "min_family_test_records": config.MIN_FAMILY_TEST_RECORDS,
        },
        "features": schema,
        "membership": membership,
        "membership_note": (
            "Per-file unique_observations OVERLAP (one observation can occur in several files); "
            "do not sum them to a dataset total."
        ),
        "integrity": {
            "train_test_observations_disjoint": True,
            "metadata_columns_excluded_from_features": sorted(SAMPLE_METADATA_COLUMNS),
            "removal_accounting_ok": plan.stats["accounting_ok"],
        },
        "limitations": [
            "Random train/test split on data whose attack families are aligned with capture days/files: "
            "a strong score does not show generalisation to unseen networks or capture periods.",
            "Excluding conflicting-label records removes ambiguous, potentially hard examples and can make "
            "evaluation look more favourable (PortScan is proportionally most affected); it does not make a "
            "model universally more accurate.",
            "The 60/40 benign/attack ratio is an experimental design choice, not real-world traffic.",
            "Family-specific results are unreliable where the test set holds few records for that family.",
            "Dataset provenance is a probable, unverified match; it is not established as the report's input data.",
        ],
    }
    drop_internal = lambda frame: frame.drop(columns="obs").reset_index(drop=True)  # noqa: E731
    return PreparationResult(params, drop_internal(train), drop_internal(test), schema, metadata, index, plan.manifest, plan.excluded_conflicts)


def prepare_dataset(
    data_dir: Path | str = config.CIC_IDS2017_DIR, params: PreparationParams = PreparationParams()
) -> PreparationResult:
    """Prepare the experiment from the real Parquet files."""
    source, infos, schema = parquet_source(data_dir)
    details = [
        {"name": i.name, "rows": i.num_rows, "size_mb": round(i.size_bytes / 1e6, 1), "columns": len(i.columns)}
        for i in infos
    ]
    result = prepare_from_source(source, params, details)
    result.metadata["source_schema"] = {
        "n_columns": len(schema.columns),
        "n_feature_columns": len(schema.feature_columns),
        "label_column": schema.label_column,
        "numeric_width_differences": {c: sorted(set(t.values())) for c, t in schema.type_differences.items()},
    }
    return result


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactPaths:
    row_index: Path
    manifest: Path
    excluded_conflicts: Path
    train: Path
    test: Path
    metadata: Path
    feature_schema: Path
    report: Path

    @classmethod
    def default(cls) -> "ArtifactPaths":
        return cls.under(config.INTERIM_DATA_DIR, config.PROCESSED_DATA_DIR, config.LOGS_DIR)

    @classmethod
    def under(cls, interim: Path, processed: Path, logs: Path) -> "ArtifactPaths":
        return cls(
            row_index=interim / "row_index.parquet",
            manifest=interim / "provenance_manifest.parquet",
            excluded_conflicts=interim / "excluded_conflicts.csv",
            train=processed / "train.parquet",
            test=processed / "test.parquet",
            metadata=processed / "dataset_metadata.json",
            feature_schema=processed / "feature_schema.json",
            report=logs / "data_preparation.txt",
        )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def write_artifacts(result: PreparationResult, paths: ArtifactPaths | None = None) -> dict[str, Path]:
    """Write all artefacts (git-ignored). Returns name -> path."""
    paths = paths or ArtifactPaths.default()
    ensure_directories(tuple({p.parent for p in vars(paths).values()}))
    result.row_index.to_parquet(paths.row_index, index=False)
    result.manifest.to_parquet(paths.manifest, index=False)
    result.excluded_conflicts.to_csv(paths.excluded_conflicts, index=False, encoding="utf-8")
    result.train.to_parquet(paths.train, index=False)
    result.test.to_parquet(paths.test, index=False)

    paths.feature_schema.write_text(json.dumps(result.feature_schema, indent=2), encoding="utf-8")

    written = {name: path for name, path in vars(paths).items()}
    # Every artefact is listed. Sizes are recorded for files already written; the metadata and
    # report files are written after this listing is built, so their size is not recorded.
    sized = ("row_index", "manifest", "excluded_conflicts", "train", "test", "feature_schema")
    result.metadata["artifacts"] = {
        name: {"path": str(path), "bytes": path.stat().st_size if name in sized else None}
        for name, path in written.items()
    }
    paths.metadata.write_text(json.dumps(result.metadata, indent=2, default=_json_default), encoding="utf-8")
    paths.report.write_text(format_preparation_report(result.metadata) + "\n", encoding="utf-8")
    return written


def load_prepared_split(
    name: str, paths: ArtifactPaths | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load ``train`` or ``test`` as (X, y, metadata). X has the ORIGINAL feature columns in
    schema order (constant-feature removal happens inside the model pipeline)."""
    paths = paths or ArtifactPaths.default()
    if name not in ("train", "test"):
        raise ValueError("name must be 'train' or 'test'")
    path = paths.train if name == "train" else paths.test
    if not path.exists() or not paths.feature_schema.exists():
        raise DatasetError(f"{path} not found. Run: python -m nids.preprocessing")
    schema = json.loads(paths.feature_schema.read_text(encoding="utf-8"))
    return split_features_and_metadata(pd.read_parquet(path), schema["original_features"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_preparation_report(meta: dict) -> str:
    lines: list[str] = []
    add = lines.append
    bar = "=" * 78

    def section(title: str) -> None:
        add("")
        add(bar)
        add(title)
        add(bar)

    def counts_line(label: str, counts: dict[str, int]) -> None:
        add(f"{label}")
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            if n:
                add(f"    {name:<28}{n:>10,}")

    add(bar)
    add("DATA PREPARATION REPORT  (Milestone 3)")
    add(bar)
    p = meta["parameters"]
    add(f"Seed: {p['seed']}   Requested: {p['n_benign']:,} benign / {p['n_attack']:,} attack   Test size: {p['test_size']}")
    add(f"Order: {p['order_of_operations']}")
    add(f"Provenance: {meta['provenance']['status']} - {meta['provenance']['probable_source']}, "
        f"{meta['provenance']['probable_version']}. {meta['provenance']['caution']}")

    section("1. RAW DATA, INVALID VALUES")
    add(f"Files: {meta['raw']['files']}   Raw rows: {meta['raw']['rows']:,}")
    inv = meta["invalid_values"]
    add(f"Rows removed for NaN/inf values: {inv['rows_removed']:,}  (NaN rows {inv.get('rows_with_nan', 0):,}, inf rows {inv.get('rows_with_inf', 0):,})")

    c = meta["conflicts"]
    section("2. CONFLICTING-LABEL EXCLUSIONS (identical features, both classes)")
    add(f"Conflicting feature vectors: {c['feature_vectors']:,}")
    add(f"Rows excluded: {c['rows_excluded']:,}  (benign {c['benign_rows_excluded']:,}, attack {c['attack_rows_excluded']:,})")
    counts_line("Attack-family counts among excluded rows:", {k: v for k, v in c["family_counts_excluded"].items() if k != config.BENIGN_FAMILY})
    add("Share of each family's valid rows excluded:")
    for name, r in sorted(c["exclusion_rate_by_family"].items(), key=lambda kv: -kv[1]["fraction"]):
        add(f"    {name:<28}{r['excluded_rows']:>8,} of {r['valid_rows']:>10,}  ({r['fraction']:.2%})")
    add("Excluded rows by source file (occurrences):")
    for name, r in c["rows_by_source_file"].items():
        add(f"    {name:<45} benign={r['benign']:>6,}  attack={r['attack']:>6,}")
    add(f"Identical features with different ATTACK families (retained as distinct observations): "
        f"{c['attack_family_only_vectors_retained']:,}")
    add("Note: this exclusion can make evaluation look more favourable by removing ambiguous records.")

    d = meta["deduplication"]
    section("3. GLOBAL DEDUPLICATION (key: features + canonical family)")
    add(f"Rows before: {d['rows_before']:,}   Unique observations: {d['unique_observations']:,}   Removed: {d['rows_removed']:,}")
    add(f"Removed by class: benign {d['benign_rows_removed']:,}, attack {d['attack_rows_removed']:,}")
    add(f"Unique observations present in more than one source file: {d['observations_spanning_multiple_files']:,}")

    e = meta["eligible"]
    section("4. ELIGIBLE UNIQUE OBSERVATIONS")
    add(f"Benign: {e['benign']:,}   Attack: {e['attack']:,}   Total: {e['total']:,}")
    counts_line("By family:", e["by_family"])

    s = meta["sample"]
    section("5. SAMPLE")
    add(f"Total {s['total']:,}: benign {s['benign']:,}, attack {s['attack']:,}")
    counts_line("Family counts in the sample:", s["family_counts"])

    sp = meta["split"]
    section("6. TRAIN / TEST SPLIT (stratified on the binary target)")
    for part in ("train", "test"):
        r = sp[part]
        add(f"{part:<6}: {r['rows']:>7,} rows  (benign {r['benign']:,}, attack {r['attack']:,})")
    add(f"{'Family':<30}{'train':>9}{'test':>9}   family-level metrics")
    fam_train, fam_test = sp["train"]["family_counts"], sp["test"]["family_counts"]
    for name in dl.FAMILY_NAMES:
        if name == config.BENIGN_FAMILY:
            continue
        rel = sp["family_test_reliability"][name]
        verdict = "ok" if rel["sufficient_for_family_metrics"] else f"INSUFFICIENT (< {sp['min_family_test_records']} test records)"
        add(f"{name:<30}{fam_train.get(name, 0):>9,}{fam_test.get(name, 0):>9,}   {verdict}")

    f = meta["features"]
    section("7. FEATURES (constant-feature selector fitted on TRAIN only)")
    add(f"Original: {f['n_original']}   Retained: {f['n_retained']}   Removed (constant in train): {len(f['removed_constant_features'])}")
    add(f"Removed: {f['removed_constant_features']}")
    add(f"Report states {f['reported_feature_count']} -> {f['discrepancy_note']}")
    add("Feature names and order verified identical across train, test and scaled pipeline output.")

    section("8. SOURCE-FILE MEMBERSHIP (overlapping counts - do NOT sum unique_observations)")
    for part in ("eligible", "sample"):
        add(f"[{part}]  unique observations: {meta['membership'][part]['all']['unique_observations_total']:,}, "
            f"spanning multiple files: {meta['membership'][part]['all']['unique_observations_spanning_multiple_files']:,}")
        add(f"    {'file':<45}{'occurrences':>12}{'unique':>10}{'exclusive':>11}{'shared':>9}")
        for name, r in meta["membership"][part]["all"]["per_file"].items():
            add(f"    {name:<45}{r['occurrences']:>12,}{r['unique_observations']:>10,}{r['exclusive_to_file']:>11,}{r['shared_with_other_files']:>9,}")

    section("LIMITATIONS")
    for item in meta["limitations"]:
        add(f"* {item}")
    if "artifacts" in meta:
        section("ARTEFACTS")
        for name, a in meta["artifacts"].items():
            size = f"({a['bytes']:,} bytes)" if a.get("bytes") is not None else "(size not recorded)"
            add(f"{name:<20}{a['path']}  {size}")
    return "\n".join(lines).encode("ascii", "backslashreplace").decode("ascii")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        prog="python -m nids.preprocessing",
        description="Prepare the experiment: exclusions, global dedup, sampling, split, feature schema.",
    )
    parser.add_argument("--data-dir", type=Path, default=config.CIC_IDS2017_DIR)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--benign", type=int, default=config.N_BENIGN_SAMPLES, help="benign sample size")
    parser.add_argument("--attack", type=int, default=config.N_ATTACK_SAMPLES, help="attack sample size")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--no-write", action="store_true", help="run and print the report without writing artefacts")
    args = parser.parse_args(argv)
    try:
        params = PreparationParams(args.seed, args.benign, args.attack, args.test_size)
        result = prepare_dataset(args.data_dir, params)
    except (DatasetError, ValueError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    if not args.no_write:
        write_artifacts(result)
    print(format_preparation_report(result.metadata))
    return 0


if __name__ == "__main__":
    sys.exit(main())
