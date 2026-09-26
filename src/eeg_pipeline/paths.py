"""Resolve repository defaults without depending on the shell's working directory."""

import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("EEG_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"EEG_PROJECT_ROOT is not a directory: {root}")
        return root
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(
        "Repository not found. Set EEG_PROJECT_ROOT when using a non-editable installation."
    )


def hbsi_data_root(root: Path | None = None) -> Path:
    root = project_root() if root is None else Path(root)
    candidates = (
        root / "data" / "external" / "member2",
        root / "data",  # Keep existing subject_* deliveries usable without moving arrays.
        root / "data" / "external" / "member3_output128_causal_full_timeline" / "data",
        root / "member3_output128_causal_full_timeline" / "data",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("subject_*/labels.npy")):
            return candidate
    raise FileNotFoundError("No subject_*/labels.npy found. Supply --data-dir explicitly.")
