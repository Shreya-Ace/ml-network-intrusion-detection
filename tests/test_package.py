"""Milestone 1 checks: the package imports and its configuration is sane."""

import importlib

import pytest

import nids
from nids import config, utils


def test_package_imports_and_has_version():
    assert isinstance(nids.__version__, str)
    assert nids.__version__


@pytest.mark.parametrize(
    "module",
    ["config", "utils", "data_loader", "preprocessing", "train", "evaluate", "predict"],
)
def test_submodules_import(module):
    importlib.import_module(f"nids.{module}")


def test_project_root_contains_expected_files():
    assert (config.PROJECT_ROOT / "requirements.txt").is_file()
    assert (config.PROJECT_ROOT / "src" / "nids" / "__init__.py").is_file()


def test_label_mapping_is_binary_with_attack_positive():
    assert config.LABEL_TO_INT == {"BENIGN": 0, "ATTACK": 1}
    assert config.INT_TO_LABEL == {0: "BENIGN", 1: "ATTACK"}


def test_experiment_design_is_internally_consistent():
    total = config.N_BENIGN_SAMPLES + config.N_ATTACK_SAMPLES
    assert total == 100_000
    assert int(total * config.TEST_SIZE) == 20_000


def test_ensure_directories_creates_missing_dirs(tmp_path):
    targets = (tmp_path / "a", tmp_path / "b" / "c")
    utils.ensure_directories(targets)
    assert all(t.is_dir() for t in targets)
    utils.ensure_directories(targets)  # idempotent


def test_get_logger_does_not_duplicate_handlers():
    first = utils.get_logger("nids.test")
    second = utils.get_logger("nids.test")
    assert first is second
    assert len(first.handlers) == 1
