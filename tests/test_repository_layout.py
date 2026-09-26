import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from eeg_pipeline.paths import hbsi_data_root, project_root
from preprocessing.configuration import load_config


def test_root_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("EEG_PROJECT_ROOT", raising=False)
    expected = project_root()
    monkeypatch.chdir(tmp_path)
    assert project_root() == expected


def test_root_override_and_missing_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EEG_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    monkeypatch.setenv("EEG_PROJECT_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        project_root()


def test_legacy_data_lookup(tmp_path):
    subject = tmp_path / "data" / "subject_01"
    subject.mkdir(parents=True)
    (subject / "labels.npy").touch()
    assert hbsi_data_root(tmp_path) == tmp_path / "data"


def test_external_delivery_lookup(tmp_path):
    delivery = tmp_path / "data" / "external" / "member3_output128_causal_full_timeline" / "data"
    subject = delivery / "subject_01"
    subject.mkdir(parents=True)
    (subject / "labels.npy").touch()
    assert hbsi_data_root(tmp_path) == delivery


def test_config_paths_are_relative_to_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("CHBMIT_ROOT", raising=False)
    monkeypatch.delenv("EEG_SIGNAL_ROOT", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("dataset:\n  root: raw\nsignal_root: signals\nserialization:\n  signal_output_dir: signals\n", encoding="utf-8")
    monkeypatch.chdir(project_root())
    resolved = load_config(config)
    assert Path(resolved["dataset"]["root"]) == tmp_path / "raw"
    assert resolved["signal_root"] == resolved["serialization"]["signal_output_dir"]


def test_config_environment_override(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CHBMIT_ROOT", str(tmp_path / "external_raw"))
    monkeypatch.setenv("EEG_SIGNAL_ROOT", str(tmp_path / "external_signals"))
    resolved = load_config(config)
    assert resolved["dataset"]["root"] == str(tmp_path / "external_raw")
    assert resolved["signal_root"] == resolved["serialization"]["signal_output_dir"]


def test_old_uppercase_package_import():
    legacy = importlib.import_module("Preprocessing._utils")
    assert legacy.compute_prediction_target(2) == 1


@pytest.mark.parametrize("module", [
    "preprocessing.stage1_edf_to_signals",
    "preprocessing.stage2_label_refinement",
    "preprocessing.stage3_sequence_build",
    "preprocessing.export_member4",
    "preprocessing.validate",
    "hbsm_loso.hbsi_main_loso",
    "hbsm_loso.hbsi_visualization",
])
def test_module_help_from_other_directory(module, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(project_root() / "src"), "PYTHONUTF8": "1"}
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
