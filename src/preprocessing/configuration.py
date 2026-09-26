"""Relative data paths are resolved against the YAML file, never the shell CWD."""

import os
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")

    def resolve(value):
        value = Path(value).expanduser()
        return str((value if value.is_absolute() else path.parent / value).resolve())

    if os.environ.get("CHBMIT_ROOT"):
        config.setdefault("dataset", {})["root"] = os.environ["CHBMIT_ROOT"]
    if os.environ.get("EEG_SIGNAL_ROOT"):
        config["signal_root"] = os.environ["EEG_SIGNAL_ROOT"]
        config.setdefault("serialization", {})["signal_output_dir"] = os.environ["EEG_SIGNAL_ROOT"]
    if "root" in config.get("dataset", {}):
        config["dataset"]["root"] = resolve(config["dataset"]["root"])
    if "signal_root" in config:
        config["signal_root"] = resolve(config["signal_root"])
    if "signal_output_dir" in config.get("serialization", {}):
        config["serialization"]["signal_output_dir"] = resolve(config["serialization"]["signal_output_dir"])
    return config
