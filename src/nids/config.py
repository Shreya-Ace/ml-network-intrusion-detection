"""Central configuration for the NIDS project.

Conventions
-----------
* All paths are derived from PROJECT_ROOT; no absolute paths anywhere else.
* Parameters that may change between experiments live here, not in code.
* Values under "REPORT CLAIMS" are the experimental setup *described in the
  industrial training report*. They are inputs to be reproduced and verified
  against the real dataset - never results to be hardcoded or forced.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# src/nids/config.py -> parents[0]=nids, [1]=src, [2]=project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
CIC_IDS2017_DIR: Path = RAW_DATA_DIR / "cic-ids2017"  # the 8 Parquet files go here
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
LOGS_DIR: Path = RESULTS_DIR / "logs"

# Directories that must exist for the pipeline to run.
REQUIRED_DIRS: tuple[Path, ...] = (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
)

# --- Target labels ---------------------------------------------------------
# ATTACK is the positive class for every metric.
BENIGN_LABEL: str = "BENIGN"
ATTACK_LABEL: str = "ATTACK"
LABEL_TO_INT: dict[str, int] = {BENIGN_LABEL: 0, ATTACK_LABEL: 1}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}

# --- Dataset columns and provenance ---------------------------------------
# Metadata columns added by the loader. They are NEVER model features.
SOURCE_FILE_COLUMN: str = "source_file"
ORIGINAL_LABEL_COLUMN: str = "original_label"
ATTACK_FAMILY_COLUMN: str = "attack_family"  # canonical family (see LABEL_ALIASES)
IS_ATTACK_COLUMN: str = "is_attack"
METADATA_COLUMNS: tuple[str, ...] = (SOURCE_FILE_COLUMN, ORIGINAL_LABEL_COLUMN)

# The exact origin of this Parquet variant is not established. Do not describe the
# data as an exact reproduction of the report's preparation until it is.
DATASET_PROVENANCE: str = "unknown (download source not established)"

# --- Label registry ----------------------------------------------------------
# canonical family -> raw spellings observed in the Parquet files (verified by the
# inspection run). Matching ignores case and surrounding whitespace. The three web-attack
# raw labels contain U+FFFD (a corrupted dash) in the source data, so they are listed
# explicitly. Any raw label that is not listed here is an ERROR, never silently ATTACK.
BENIGN_FAMILY: str = "Benign"
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "Benign": ("Benign",),
    "Bot": ("Bot",),
    "DDoS": ("DDoS",),
    "DoS GoldenEye": ("DoS GoldenEye",),
    "DoS Hulk": ("DoS Hulk",),
    "DoS Slowhttptest": ("DoS Slowhttptest",),
    "DoS slowloris": ("DoS slowloris",),
    "FTP-Patator": ("FTP-Patator",),
    "Heartbleed": ("Heartbleed",),
    "Infiltration": ("Infiltration",),
    "PortScan": ("PortScan",),
    "SSH-Patator": ("SSH-Patator",),
    "Web Attack - Brute Force": ("Web Attack \ufffd Brute Force",),
    "Web Attack - Sql Injection": ("Web Attack \ufffd Sql Injection",),
    "Web Attack - XSS": ("Web Attack \ufffd XSS",),
}

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED: int = 42

# --- Experiment design (REPORT CLAIMS: to be verified, not assumed) --------
N_BENIGN_SAMPLES: int = 60_000
N_ATTACK_SAMPLES: int = 40_000
TEST_SIZE: float = 0.20  # 80:20 stratified split
REPORTED_FEATURE_COUNT: int = 70  # Verified against real preprocessing in Milestone 3.

# Model hyperparameters are added in Milestone 4, where they are used.
