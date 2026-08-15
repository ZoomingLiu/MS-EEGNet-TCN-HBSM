"""
dataset.py -- PyTorch Datasets for Preprocessing Pipeline.

WindowDatasetV2: returns single windows [22, 1280]
SequenceDatasetV2: returns TCN sequences [24, 22, 1280]

Label interface:
  - label: int64, 5-class (-1=UNKNOWN, 0=INTERICTAL, 1=TRANSITION, 2=PREICTAL, 3=ICTAL)
  - loss_weight: float32 (transition=0.2, others=1.0)
  - supervised_loss_mask: bool (from prediction_loss_mask or fallback)

Both use lazy signal loading via np.load(mmap_mode="r").

DataLoader creators:
  - create_dataloaders()       -- group-holdout split, loads group_holdout_stats.npz
  - create_loso_dataloaders()  -- LOSO per-fold split, loads per-fold loso_stats.npz
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import random


def _get_mask_column(df, primary, fallback):
    """Return the mask column name that exists in df."""
    if primary in df.columns:
        return primary
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Neither {primary} nor {fallback} found in columns: {df.columns.tolist()}")


class WindowDatasetV2(Dataset):
    """
    EEGNet-compatible window dataset.

    Each item returns:
      signal:               [22, 1280] float32
      label:                int64   -- 5-class
      loss_weight:          float32 -- transition=0.2, others=1.0
      supervised_loss_mask: bool    -- False = censored, exclude from loss
      is_valid:             bool    -- overall validity flag
    """

    def __init__(self, metadata_df: pd.DataFrame, split: str = "train",
                 norm_stats: dict = None, interictal_ratio: float = 2.0,
                 seed: int = 42, resample: bool = True,
                 split_col: str = "split"):
        """
        Args:
            split_col: column name for train/val/test assignment.
                       Default "split" (group-holdout).
                       Set to "loso_split" for per-fold LOSO mode.
        """
        self.split = split
        self.norm_mean = norm_stats["mean"] if norm_stats else np.zeros(22, dtype=np.float32)
        self.norm_std = norm_stats["std"] if norm_stats else np.ones(22, dtype=np.float32)
        self.resample = resample and (split == "train")
        self.interictal_ratio = interictal_ratio
        self.seed = seed
        self.rng = random.Random(seed)

        self.df = metadata_df[metadata_df[split_col] == split].copy()

        self.mask_col = _get_mask_column(self.df, "prediction_loss_mask", "supervised_loss_mask")

        if split == "train":
            self.df = self.df[self.df["is_valid"]].copy()
            self.df = self.df[self.df[self.mask_col]].copy()

        self.positive_df = self.df[self.df["state_label"].isin([1, 2])].copy()
        self.interictal_df = self.df[self.df["state_label"] == 0].copy()

        self._build_index()
        self._signal_cache = {}

    def _build_index(self):
        if self.split != "train":
            self.indices = list(range(len(self.df)))
            return

        n_positive = len(self.positive_df)
        n_interictal_target = min(
            int(n_positive * self.interictal_ratio),
            len(self.interictal_df)
        )

        if n_interictal_target > 0 and len(self.interictal_df) > 0:
            interictal_indices = self.rng.sample(
                range(len(self.interictal_df)), n_interictal_target)
        else:
            interictal_indices = list(range(len(self.interictal_df)))

        self.indices = (
            [("positive", i) for i in range(n_positive)] +
            [("interictal", i) for i in interictal_indices]
        )
        self.rng.shuffle(self.indices)

    def resample_interictal(self):
        if not self.resample:
            return
        self.rng = random.Random(self.seed + hash(str(self.epoch)))
        self._build_index()

    @property
    def epoch(self):
        return getattr(self, "_epoch", 0)

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    def _load_signal(self, signal_path: str) -> np.ndarray:
        if signal_path not in self._signal_cache:
            self._signal_cache[signal_path] = np.load(signal_path, mmap_mode="r")
        return self._signal_cache[signal_path]

    def __len__(self):
        return len(self.indices) if self.split == "train" else len(self.df)

    def __getitem__(self, idx):
        if self.split == "train":
            source, local_idx = self.indices[idx]
            if source == "positive":
                row = self.positive_df.iloc[local_idx]
            else:
                row = self.interictal_df.iloc[local_idx]
        else:
            row = self.df.iloc[idx]

        signal = self._load_signal(row["signal_path"])
        x = signal[:, int(row["start_sample"]):int(row["end_sample"])].astype(np.float32)
        x = (x - self.norm_mean.reshape(-1, 1)) / self.norm_std.reshape(-1, 1)
        x = x.astype(np.float32)

        return {
            "signal": torch.from_numpy(x),
            "label": torch.tensor(int(row["state_label"]), dtype=torch.long),
            "loss_weight": torch.tensor(float(row["loss_weight"]), dtype=torch.float32),
            "supervised_loss_mask": torch.tensor(bool(row[self.mask_col]), dtype=torch.bool),
            "is_valid": torch.tensor(bool(row["is_valid"]), dtype=torch.bool),
        }


class SequenceDatasetV2(Dataset):
    """
    TCN-compatible sequence dataset.

    Each item returns:
      signal:               [24, 22, 1280] float32
      labels:               [24] int64    -- 5-class per window
      loss_weights:         [24] float32  -- transition=0.2, others=1.0
      supervised_loss_mask: [24] bool     -- per-window mask
      sequence_target:      int64         -- anchor (last window) 5-class label
      quality_flags:        [24] int64    -- per-window quality flags
    """

    def __init__(self, sequence_df: pd.DataFrame, metadata_df: pd.DataFrame,
                 split: str = "train", norm_stats: dict = None,
                 interictal_ratio: float = 2.0, seed: int = 42,
                 resample: bool = True, split_col: str = "split"):
        """
        Args:
            split_col: column name for train/val/test assignment.
                       Default "split" (group-holdout).
                       Set to "loso_split" for per-fold LOSO mode.
        """
        self.split = split
        self.norm_mean = norm_stats["mean"] if norm_stats else np.zeros(22, dtype=np.float32)
        self.norm_std = norm_stats["std"] if norm_stats else np.ones(22, dtype=np.float32)
        self.resample = resample and (split == "train")
        self.interictal_ratio = interictal_ratio
        self.seed = seed
        self.rng = random.Random(seed)

        self.seq_df = sequence_df[sequence_df[split_col] == split].copy()

        self.window_lookup = {}
        for _, row in metadata_df.iterrows():
            self.window_lookup[row["window_id"]] = {
                "signal_path": row["signal_path"],
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
            }

        if "state_labels" in self.seq_df.columns:
            self.seq_df["_anchor_label"] = self.seq_df["state_labels"].apply(
                lambda arr: int(arr[-1]))
        else:
            raise KeyError("sequence_df missing 'state_labels' column")

        self.positive_seqs = self.seq_df[
            self.seq_df["_anchor_label"].isin([1, 2])
        ].copy()
        self.interictal_seqs = self.seq_df[self.seq_df["_anchor_label"] == 0].copy()

        if "has_boundary_window" in self.seq_df.columns and split == "train":
            self.positive_seqs = self.positive_seqs[~self.positive_seqs["has_boundary_window"]]
            self.interictal_seqs = self.interictal_seqs[~self.interictal_seqs["has_boundary_window"]]

        self._build_index()
        self._signal_cache = {}

    def _build_index(self):
        if self.split != "train":
            self.indices = list(range(len(self.seq_df)))
            return

        n_positive = len(self.positive_seqs)
        n_interictal_target = min(
            int(n_positive * self.interictal_ratio),
            len(self.interictal_seqs)
        )

        if n_interictal_target > 0 and len(self.interictal_seqs) > 0:
            interictal_indices = self.rng.sample(
                range(len(self.interictal_seqs)), n_interictal_target)
        else:
            interictal_indices = list(range(len(self.interictal_seqs)))

        self.indices = (
            [("positive", i) for i in range(n_positive)] +
            [("interictal", i) for i in interictal_indices]
        )
        self.rng.shuffle(self.indices)

    def _load_signal(self, signal_path: str) -> np.ndarray:
        if signal_path not in self._signal_cache:
            self._signal_cache[signal_path] = np.load(signal_path, mmap_mode="r")
        return self._signal_cache[signal_path]

    def __len__(self):
        return len(self.indices) if self.split == "train" else len(self.seq_df)

    def __getitem__(self, idx):
        if self.split == "train":
            source, local_idx = self.indices[idx]
            if source == "positive":
                seq_row = self.positive_seqs.iloc[local_idx]
            else:
                seq_row = self.interictal_seqs.iloc[local_idx]
        else:
            seq_row = self.seq_df.iloc[idx]

        window_ids = seq_row["window_ids"]
        if isinstance(window_ids, str):
            try:
                window_ids = eval(window_ids)
            except Exception:
                window_ids = [window_ids]
        elif isinstance(window_ids, np.ndarray):
            window_ids = window_ids.tolist()
        elif not isinstance(window_ids, list):
            window_ids = list(window_ids)

        state_labels = seq_row["state_labels"]
        loss_weights = seq_row["loss_weights"]
        pred_masks = seq_row["prediction_loss_masks"]
        quality_flags = seq_row["quality_flags"]

        if isinstance(state_labels, np.ndarray):
            state_labels = state_labels.tolist()
        if isinstance(loss_weights, np.ndarray):
            loss_weights = loss_weights.tolist()
        if isinstance(pred_masks, np.ndarray):
            pred_masks = pred_masks.tolist()
        if isinstance(quality_flags, np.ndarray):
            quality_flags = quality_flags.tolist()

        signals = []
        for wid in window_ids:
            w = self.window_lookup[wid]
            signal = self._load_signal(w["signal_path"])
            x = signal[:, w["start_sample"]:w["end_sample"]].astype(np.float32)
            x = (x - self.norm_mean.reshape(-1, 1)) / self.norm_std.reshape(-1, 1)
            x = x.astype(np.float32)
            signals.append(x)

        signal_tensor = torch.from_numpy(np.stack(signals))

        return {
            "signal": signal_tensor,
            "labels": torch.tensor(state_labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "supervised_loss_mask": torch.tensor(pred_masks, dtype=torch.bool),
            "sequence_target": torch.tensor(int(seq_row["_anchor_label"]), dtype=torch.long),
            "quality_flags": torch.tensor(quality_flags, dtype=torch.long),
        }


def load_norm_stats(stats_path: str) -> dict:
    """Load normalization statistics from .npz file."""
    data = np.load(stats_path)
    return {"mean": data["mean"], "std": data["std"]}


def create_dataloaders(data_dir: str, batch_size: int = 32,
                       num_workers: int = 0) -> dict:
    """
    Create train/val/test DataLoaders for GROUP-HOLDOUT split.

    Loads normalization from: normalization/group_holdout_stats.npz
    Uses metadata["split"] column for train/val/test assignment.

    NOTE: For LOSO 23-fold cross-validation, use create_loso_dataloaders()
    instead, which loads per-fold loso_stats.npz (no data leakage).

    Returns:
        {
            "window":   {"train": DataLoader, "validation": DataLoader, "test": DataLoader},
            "sequence": {"train": DataLoader, "validation": DataLoader, "test": DataLoader},
            "norm_stats": dict,
        }
    """
    data_path = Path(data_dir)

    metadata = pd.read_parquet(data_path / "metadata_all.parquet")
    sequences = pd.read_parquet(data_path / "sequence_index.parquet")

    norm_stats = load_norm_stats(
        str(data_path / "normalization" / "group_holdout_stats.npz"))

    sampling_cfg_path = data_path / "sampling_config.yaml"
    if sampling_cfg_path.exists():
        with open(sampling_cfg_path) as f:
            sampling_cfg = yaml.safe_load(f)
        ratio = sampling_cfg.get("interictal_to_preictal_ratio", 2.0)
    else:
        ratio = 2.0

    window_datasets = {}
    sequence_datasets = {}

    for split in ["train", "validation", "test"]:
        resample = (split == "train")
        window_datasets[split] = WindowDatasetV2(
            metadata, split=split, norm_stats=norm_stats,
            interictal_ratio=ratio, resample=resample)
        sequence_datasets[split] = SequenceDatasetV2(
            sequences, metadata, split=split, norm_stats=norm_stats,
            interictal_ratio=ratio, resample=resample)

    dataloaders = {"window": {}, "sequence": {}, "norm_stats": norm_stats}

    for split in ["train", "validation", "test"]:
        shuffle = (split == "train")
        dataloaders["window"][split] = DataLoader(
            window_datasets[split], batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, drop_last=False)
        dataloaders["sequence"][split] = DataLoader(
            sequence_datasets[split], batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, drop_last=False)

    return dataloaders


def create_loso_dataloaders(data_dir: str, fold_id: int,
                            batch_size: int = 32, num_workers: int = 0) -> dict:
    """
    Create per-fold LOSO DataLoaders using per-fold normalization stats.

    Each fold loads its own:
      - folds/loso/fold_XX/normalization/loso_stats.npz
      - folds/loso/fold_XX/sequence_split.parquet

    The per-fold normalization is computed ONLY from train-subject windows,
    preventing data leakage from the held-out test and validation subjects.

    Args:
        data_dir: path to data freeze output (e.g. artifacts/sequence_build/)
        fold_id:  LOSO fold index (0-22 for 23 subjects)
        batch_size: DataLoader batch size
        num_workers: DataLoader worker count

    Returns:
        {
            "window":   {"train": DataLoader, "validation": DataLoader, "test": DataLoader},
            "sequence": {"train": DataLoader, "validation": DataLoader, "test": DataLoader},
            "norm_stats": dict,
            "fold_info": {"fold_id": int, "test_subject": str, "val_subject": str},
        }
    """
    data_path = Path(data_dir)

    # --- Load base data ---
    metadata = pd.read_parquet(data_path / "metadata_all.parquet")
    sequences = pd.read_parquet(data_path / "sequence_index.parquet")

    # --- Load LOSO fold info ---
    loso_folds = pd.read_parquet(data_path / "folds" / "loso_folds.parquet")
    fold_row = loso_folds[loso_folds["fold_id"] == fold_id]
    if len(fold_row) == 0:
        raise ValueError(
            f"Fold {fold_id} not found in loso_folds.parquet. "
            f"Valid range: 0-{len(loso_folds)-1}"
        )
    fold_info = fold_row.iloc[0]
    test_subj = fold_info["test_subject"]
    val_subj = fold_info["val_subject"]
    train_subjs = list(fold_info["train_subjects"])

    # --- Load per-fold normalization (computed ONLY from train subjects) ---
    norm_path = (
        data_path / "folds" / "loso" / f"fold_{fold_id:02d}"
        / "normalization" / "loso_stats.npz"
    )
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Per-fold normalization not found: {norm_path}\n"
            f"Run Stage 3 first to generate LOSO per-fold stats."
        )
    norm_stats = load_norm_stats(str(norm_path))

    # --- Build window-level LOSO splits ---
    metadata_loso = metadata.copy()
    metadata_loso["loso_split"] = "train"
    metadata_loso.loc[metadata_loso["subject_uid"] == test_subj, "loso_split"] = "test"
    metadata_loso.loc[metadata_loso["subject_uid"] == val_subj, "loso_split"] = "validation"

    # --- Build sequence-level LOSO splits ---
    seq_split = pd.read_parquet(
        data_path / "folds" / "loso" / f"fold_{fold_id:02d}" / "sequence_split.parquet"
    )
    seq_split_map = seq_split.set_index("sequence_id")["loso_split"].to_dict()
    sequences_loso = sequences.copy()
    sequences_loso["loso_split"] = sequences_loso["sequence_id"].map(seq_split_map)

    n_missing = sequences_loso["loso_split"].isna().sum()
    if n_missing > 0:
        missing_ids = sequences_loso.loc[
            sequences_loso["loso_split"].isna(), "sequence_id"
        ].unique()[:10]
        raise ValueError(
            f"{n_missing} sequences are missing from the LOSO split manifest "
            f"for fold {fold_id} (test={test_subj}, val={val_subj}).\n"
            f"First missing sequence_id values: {list(missing_ids)}\n"
            f"Expected all sequences to be present in "
            f"folds/loso/fold_{fold_id:02d}/sequence_split.parquet"
        )

    # --- Sampling config ---
    sampling_cfg_path = data_path / "sampling_config.yaml"
    if sampling_cfg_path.exists():
        with open(sampling_cfg_path) as f:
            sampling_cfg = yaml.safe_load(f)
        ratio = sampling_cfg.get("interictal_to_preictal_ratio", 2.0)
    else:
        ratio = 2.0

    # --- Build datasets with split_col="loso_split" ---
    window_datasets = {}
    sequence_datasets = {}

    for split in ["train", "validation", "test"]:
        resample = (split == "train")
        window_datasets[split] = WindowDatasetV2(
            metadata_loso, split=split, norm_stats=norm_stats,
            interictal_ratio=ratio, resample=resample,
            split_col="loso_split",
        )
        sequence_datasets[split] = SequenceDatasetV2(
            sequences_loso, metadata_loso, split=split, norm_stats=norm_stats,
            interictal_ratio=ratio, resample=resample,
            split_col="loso_split",
        )

    # --- Build DataLoaders ---
    dataloaders = {
        "window": {},
        "sequence": {},
        "norm_stats": norm_stats,
        "fold_info": {
            "fold_id": int(fold_id),
            "test_subject": str(test_subj),
            "val_subject": str(val_subj),
            "n_train_subjects": len(train_subjs),
        },
    }

    for split in ["train", "validation", "test"]:
        shuffle = (split == "train")
        dataloaders["window"][split] = DataLoader(
            window_datasets[split], batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, drop_last=False,
        )
        dataloaders["sequence"][split] = DataLoader(
            sequence_datasets[split], batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, drop_last=False,
        )

    return dataloaders
