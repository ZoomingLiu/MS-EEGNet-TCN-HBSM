#!/usr/bin/env python3
"""
stage3_sequence_build.py -- Stage 3: Sequence Index & Cross-Validation Build.

Reuses Stage 2 metadata, labels NPY, candidate pool, quality files, and signal NPY.
Only rebuilds: sequence_index, sampling manifests, LOSO per-fold artifacts,
config, audit report, and SHA256.

Key features:
  1. sequence_target = anchor (last window) prediction_target
  2. sequence_prediction_mask = endpoint prediction_loss_mask AND no boundary AND no ICTAL
  3. sequence_loss_weight = endpoint loss_weight
  4. Per-LOSO-fold normalization + sampling (no group-holdout reuse)
  5. Configurable signal_root

Usage:
  python Preprocessing/stage3_sequence_build.py \
      --config Preprocessing/config.yaml \
      --stage2-dir artifacts/label_refinement \
      --output artifacts/sequence_build/
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from Preprocessing._utils import (
    LABEL_UNKNOWN, LABEL_INTERICTAL, LABEL_TRANSITION, LABEL_PREICTAL, LABEL_ICTAL,
    LABEL_NAMES,
    QUALITY_FLAGS,
)


def sha256_file(filepath: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_sums(output_path: Path) -> list:
    """Compute SHA256 for all files in output directory."""
    results = []
    for fpath in sorted(output_path.rglob("*")):
        if fpath.is_file() and fpath.name != "SHA256SUMS.txt":
            rel = fpath.relative_to(output_path)
            rel_str = str(rel).replace("\\", "/")
            digest = sha256_file(fpath)
            results.append((rel_str, digest))
    return results


# ---------------------------------------------------------------------------
# Phase 1: Load Stage 2 metadata (unchanged)
# ---------------------------------------------------------------------------

def load_stage2_metadata(stage2_dir: Path) -> pd.DataFrame:
    """Load Stage 2 metadata_all.parquet -- source of truth for windows."""
    print("\n[Phase 1] Loading Stage 2 metadata (unchanged)...", flush=True)

    meta_path = stage2_dir / "metadata_all.parquet"
    df = pd.read_parquet(str(meta_path))
    print(f"  Loaded {len(df)} windows from {meta_path}", flush=True)
    print(f"  Columns: {df.columns.tolist()}", flush=True)

    required = [
        "window_id", "case_id", "subject_uid", "record_id",
        "continuous_block_id", "split", "signal_path",
        "start_sample", "end_sample",
        "state_label", "prediction_target", "prediction_loss_mask",
        "state_loss_mask", "is_valid", "boundary_flag",
        "loss_weight", "quality_flags",
        "absolute_start_sec", "absolute_end_sec",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stage 2 metadata missing columns: {missing}")

    print(f"  prediction_target dist: {df['prediction_target'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"  state_label dist: {df['state_label'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"  prediction_loss_mask=True: {df['prediction_loss_mask'].sum()}", flush=True)
    print(f"  boundary_flag=True: {df['boundary_flag'].sum()}", flush=True)

    return df


# ---------------------------------------------------------------------------
# Phase 2: Rebuild sequence_index with ANCHOR-based target
# ---------------------------------------------------------------------------

def rebuild_sequence_index(
    df: pd.DataFrame,
    seq_length: int,
    seq_stride: int,
    step_samp: int,
    min_valid_fraction: float,
) -> pd.DataFrame:
    """
    Rebuild sequence_index with:
    - sequence_target = anchor (LAST window) prediction_target
    - sequence_prediction_mask = endpoint prediction_loss_mask AND no boundary AND no ICTAL
    - sequence_loss_weight = endpoint loss_weight
    """
    print("\n[Phase 2] Rebuilding sequence_index (anchor-based target)...", flush=True)

    sequence_records = []
    seq_global_id = 0

    group_cols = ["case_id", "record_id", "continuous_block_id", "split"]
    n_groups = df.groupby(group_cols).ngroups
    processed_groups = 0

    for (case_id, record_id, block_id, split), group in df.groupby(group_cols):
        group = group.sort_values("start_sample").reset_index(drop=True)
        processed_groups += 1

        for i in range(0, len(group) - seq_length + 1, seq_stride):
            seq_wins = group.iloc[i:i + seq_length]

            # STRICT CHECK 1: start_sample diff must be exactly step_samp
            starts = seq_wins["start_sample"].to_numpy()
            diffs = np.diff(starts)
            if len(diffs) > 0 and not np.all(diffs == step_samp):
                continue

            # STRICT CHECK 2-5: all same subject/record/block/split
            if seq_wins["subject_uid"].nunique() > 1:
                continue
            if seq_wins["record_id"].nunique() > 1:
                continue
            if seq_wins["continuous_block_id"].nunique() > 1:
                continue
            if seq_wins["split"].nunique() > 1:
                continue

            # STRICT CHECK 6: min valid fraction
            valid_frac = seq_wins["is_valid"].mean()
            if valid_frac < min_valid_fraction:
                continue

            # --- Anchor-based target ---
            anchor = seq_wins.iloc[-1]
            anchor_pred_target = int(anchor["prediction_target"])
            anchor_loss_mask = bool(anchor["prediction_loss_mask"])
            anchor_loss_weight = float(anchor["loss_weight"])

            seq_target = anchor_pred_target

            has_boundary = bool(seq_wins["boundary_flag"].any())
            state_labels = seq_wins["state_label"].to_numpy()
            has_ictal = bool((state_labels == LABEL_ICTAL).any())

            seq_pred_mask = anchor_loss_mask and (not has_boundary) and (not has_ictal)
            seq_loss_weight = anchor_loss_weight

            pred_targets = seq_wins["prediction_target"].tolist()
            pred_masks = seq_wins["prediction_loss_mask"].tolist()
            state_masks = seq_wins["state_loss_mask"].tolist()
            loss_weights = seq_wins["loss_weight"].tolist()
            qf_list = seq_wins["quality_flags"].tolist()
            start_samples = seq_wins["start_sample"].tolist()

            seq_id = f"seq_{seq_global_id:08d}"

            sequence_records.append({
                "sequence_id": seq_id,
                "window_ids": list(seq_wins["window_id"]),
                "subject_uid": seq_wins["subject_uid"].iloc[0],
                "case_id": case_id,
                "record_id": record_id,
                "continuous_block_id": int(block_id),
                "split": split,
                "sequence_start_time": float(seq_wins["absolute_start_sec"].iloc[0]),
                "sequence_end_time": float(seq_wins["absolute_end_sec"].iloc[-1]),
                "sequence_target": seq_target,
                "sequence_prediction_mask": bool(seq_pred_mask),
                "sequence_loss_weight": float(seq_loss_weight),
                "has_boundary_window": bool(has_boundary),
                "has_ictal_window": bool(has_ictal),
                "n_valid": int(seq_wins["is_valid"].sum()),
                "n_prediction_masked": int(sum(pred_masks)),
                "n_state_masked": int(sum(state_masks)),
                "start_samples": start_samples,
                "state_labels": list(state_labels.tolist()),
                "prediction_targets": pred_targets,
                "loss_weights": loss_weights,
                "prediction_loss_masks": pred_masks,
                "state_loss_masks": state_masks,
                "quality_flags": qf_list,
            })
            seq_global_id += 1

        if processed_groups % 500 == 0:
            print(f"  Processed {processed_groups}/{n_groups} groups, {seq_global_id} sequences...", flush=True)

    df_seq = pd.DataFrame(sequence_records)
    print(f"  Total sequences: {len(df_seq)}", flush=True)
    print(f"  sequence_target dist: {df_seq['sequence_target'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"  sequence_prediction_mask=True: {df_seq['sequence_prediction_mask'].sum()}", flush=True)
    print(f"  has_boundary_window=True: {df_seq['has_boundary_window'].sum()}", flush=True)
    print(f"  has_ictal_window=True: {df_seq['has_ictal_window'].sum()}", flush=True)

    eligible = df_seq[df_seq["sequence_prediction_mask"] & df_seq["sequence_target"].isin([0, 1])]
    print(f"  Eligible for sampling (mask=True, target in {{0,1}}): {len(eligible)}", flush=True)
    print(f"    target=0 (INTERICTAL): {(eligible['sequence_target']==0).sum()}", flush=True)
    print(f"    target=1 (PREICTAL): {(eligible['sequence_target']==1).sum()}", flush=True)

    return df_seq


# ---------------------------------------------------------------------------
# Phase 3: Rebuild group_holdout sampling manifest
# ---------------------------------------------------------------------------

def rebuild_group_holdout_sampling(
    df_seq: pd.DataFrame,
    ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Rebuild 2:1 sampling manifest from eligible sequences only."""
    print("\n[Phase 3] Rebuilding group_holdout sampling (2:1)...", flush=True)

    eligible = df_seq[
        (df_seq["split"] == "train") &
        df_seq["sequence_prediction_mask"] &
        df_seq["sequence_target"].isin([0, 1])
    ].copy()

    n_inter = (eligible["sequence_target"] == 0).sum()
    n_pre = (eligible["sequence_target"] == 1).sum()
    print(f"  Eligible train sequences: {len(eligible)} (INTER={n_inter}, PRE={n_pre})", flush=True)

    target_inter = int(n_pre * ratio)
    if target_inter > n_inter:
        target_inter = n_inter

    rng = np.random.RandomState(seed)
    inter_indices = eligible[eligible["sequence_target"] == 0].index.to_numpy().copy()
    rng.shuffle(inter_indices)
    sampled_inter = inter_indices[:target_inter]

    pre_indices = eligible[eligible["sequence_target"] == 1].index.to_numpy().copy()

    all_indices = np.concatenate([sampled_inter, pre_indices])
    sampled = df_seq.loc[all_indices].copy()

    print(f"  Sampled: {len(sampled)} (INTER={target_inter}, PRE={len(pre_indices)})", flush=True)
    print(f"  Ratio: {target_inter}:{len(pre_indices)} = {target_inter/max(len(pre_indices),1):.2f}:1", flush=True)

    return sampled


# ---------------------------------------------------------------------------
# Phase 4: Generate LOSO folds with per-fold normalization + sampling
# ---------------------------------------------------------------------------

def generate_loso_folds(
    df: pd.DataFrame,
    df_seq: pd.DataFrame,
    output_path: Path,
    ratio: float,
    seed: int,
    n_channels: int,
):
    """
    For each LOSO fold:
    - test = left-out subject
    - val = next subject in rotation
    - train = all others
    - Per-fold normalization stats (streaming over valid train windows)
    - Per-fold 2:1 sampling manifest
    """
    print("\n[Phase 4] Generating LOSO folds with per-fold normalization + sampling...", flush=True)

    subjects = sorted(df["subject_uid"].unique())
    n_subjects = len(subjects)
    print(f"  Subjects: {n_subjects}", flush=True)

    loso_dir = output_path / "folds" / "loso"
    loso_dir.mkdir(parents=True, exist_ok=True)

    print("  Precomputing per-file stats for normalization...", flush=True)
    valid_windows = df[df["is_valid"]].copy()

    file_stats = {}
    file_to_subject = {}

    for sig_path, group in valid_windows.groupby("signal_path"):
        subj = group["subject_uid"].iloc[0]
        file_to_subject[sig_path] = subj

        try:
            sig = np.load(sig_path, mmap_mode="r")
        except Exception as e:
            print(f"    WARNING: Cannot load {sig_path}: {e}", flush=True)
            continue

        ch_sum = np.zeros(n_channels, dtype=np.float64)
        ch_sq_sum = np.zeros(n_channels, dtype=np.float64)
        total_samples = 0

        starts = group["start_sample"].to_numpy()
        ends = group["end_sample"].to_numpy()

        for start, end in zip(starts, ends):
            s, e = int(start), int(end)
            chunk = sig[:, s:e].astype(np.float64)
            ch_sum += chunk.sum(axis=1)
            ch_sq_sum += (chunk ** 2).sum(axis=1)
            total_samples += chunk.shape[1]

        file_stats[sig_path] = (ch_sum, ch_sq_sum, total_samples, subj)

    print(f"  Precomputed stats for {len(file_stats)} files", flush=True)

    loso_manifest = []

    for fold_idx, test_subj in enumerate(subjects):
        val_subj = subjects[(fold_idx + 1) % n_subjects]
        train_subjects = [s for s in subjects if s != test_subj and s != val_subj]

        test_cases = sorted(df[df["subject_uid"] == test_subj]["case_id"].unique().tolist())
        val_cases = sorted(df[df["subject_uid"] == val_subj]["case_id"].unique().tolist())
        train_cases = sorted(df[df["subject_uid"].isin(train_subjects)]["case_id"].unique().tolist())

        loso_manifest.append({
            "fold_id": fold_idx,
            "test_subject": test_subj,
            "val_subject": val_subj,
            "train_subjects": train_subjects,
            "test_cases": test_cases,
            "val_cases": val_cases,
            "train_cases": train_cases,
        })

        # Per-fold normalization
        fold_sum = np.zeros(n_channels, dtype=np.float64)
        fold_sq_sum = np.zeros(n_channels, dtype=np.float64)
        fold_count = 0

        for sig_path, (ch_sum, ch_sq_sum, count, subj) in file_stats.items():
            if subj in train_subjects:
                fold_sum += ch_sum
                fold_sq_sum += ch_sq_sum
                fold_count += count

        if fold_count > 0:
            fold_mean = fold_sum / fold_count
            fold_var = fold_sq_sum / fold_count - fold_mean ** 2
            fold_var = np.maximum(fold_var, 0)
            fold_std = np.sqrt(fold_var)
        else:
            fold_mean = np.zeros(n_channels, dtype=np.float64)
            fold_std = np.ones(n_channels, dtype=np.float64)

        fold_norm_dir = loso_dir / f"fold_{fold_idx:02d}" / "normalization"
        fold_norm_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(fold_norm_dir / "loso_stats.npz"),
            mean=fold_mean.astype(np.float64),
            std=fold_std.astype(np.float64),
            n_windows=fold_count,
            method="streaming_all_valid_train_loso",
            fold_id=fold_idx,
            test_subject=test_subj,
        )

        # Per-fold sequence split
        fold_seq = df_seq.copy()
        fold_seq["loso_split"] = "train"
        fold_seq.loc[fold_seq["subject_uid"] == test_subj, "loso_split"] = "test"
        fold_seq.loc[fold_seq["subject_uid"] == val_subj, "loso_split"] = "validation"

        fold_seq_split = fold_seq[[
            "sequence_id", "subject_uid", "case_id", "loso_split",
            "sequence_target", "sequence_prediction_mask", "sequence_loss_weight",
        ]].copy()
        fold_seq_split.to_parquet(
            str(loso_dir / f"fold_{fold_idx:02d}" / "sequence_split.parquet"),
            index=False
        )

        # Per-fold 2:1 sampling
        train_eligible = fold_seq[
            (fold_seq["loso_split"] == "train") &
            fold_seq["sequence_prediction_mask"] &
            fold_seq["sequence_target"].isin([0, 1])
        ]

        n_inter_f = (train_eligible["sequence_target"] == 0).sum()
        n_pre_f = (train_eligible["sequence_target"] == 1).sum()
        target_inter_f = int(n_pre_f * ratio)
        if target_inter_f > n_inter_f:
            target_inter_f = n_inter_f

        rng = np.random.RandomState(seed + fold_idx)
        if n_inter_f > 0 and target_inter_f > 0:
            inter_idx = train_eligible[train_eligible["sequence_target"] == 0].index.to_numpy().copy()
            rng.shuffle(inter_idx)
            sampled_inter_f = inter_idx[:target_inter_f]
        else:
            sampled_inter_f = np.array([], dtype=int)

        pre_idx_f = train_eligible[train_eligible["sequence_target"] == 1].index.to_numpy().copy()
        all_idx_f = np.concatenate([sampled_inter_f, pre_idx_f])

        if len(all_idx_f) > 0:
            sampled_f = fold_seq.loc[all_idx_f].copy()
        else:
            sampled_f = pd.DataFrame(columns=fold_seq.columns)

        sampled_f.to_parquet(
            str(loso_dir / f"fold_{fold_idx:02d}" / "train_sampling.parquet"),
            index=False
        )

        if (fold_idx + 1) % 5 == 0 or fold_idx == 0:
            print(f"  Fold {fold_idx:02d}: test={test_subj}, val={val_subj}, "
                  f"train_subjects={len(train_subjects)}, norm_windows={fold_count}, "
                  f"sampled={len(sampled_f)}", flush=True)

    loso_df = pd.DataFrame(loso_manifest)
    loso_df.to_parquet(str(output_path / "folds" / "loso_folds.parquet"), index=False)
    print(f"  LOSO folds: {len(loso_manifest)} folds generated", flush=True)

    return loso_df


# ---------------------------------------------------------------------------
# Phase 5: Copy unchanged artifacts from Stage 2
# ---------------------------------------------------------------------------

def copy_unchanged_artifacts(stage2_dir: Path, output_path: Path):
    """Copy files that must NOT change: metadata, labels, candidate pool, quality."""
    print("\n[Phase 5] Copying unchanged artifacts from Stage 2...", flush=True)

    unchanged_files = [
        "metadata_all.parquet",
        "metadata_all.csv",
        "labels_prediction.npy",
        "labels_state.npy",
        "interictal_candidate_pool.parquet",
        "quality/excluded_records.csv",
        "quality/channel_issues.csv",
        "quality/annotation_issues.csv",
        "normalization/group_holdout_stats.npz",
    ]

    for rel_path in unchanged_files:
        src = stage2_dir / rel_path
        dst = output_path / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            print(f"  Copied: {rel_path}", flush=True)
        else:
            print(f"  WARNING: Source not found: {rel_path}", flush=True)

    src = stage2_dir / "folds" / "group_holdout_split.parquet"
    dst = output_path / "folds" / "group_holdout_split.parquet"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        print(f"  Copied: folds/group_holdout_split.parquet", flush=True)


# ---------------------------------------------------------------------------
# Phase 6: Update config
# ---------------------------------------------------------------------------

def update_config(config: dict, signal_root: str) -> dict:
    """Update config for Stage 3."""
    config["data_version"] = "CHBMIT_DataFreeze_V2.2"
    config["schema_version"] = "2.2"
    config["signal_root"] = signal_root
    if "serialization" not in config:
        config["serialization"] = {}
    config["serialization"]["signal_output_dir"] = signal_root
    return config


# ---------------------------------------------------------------------------
# Phase 7: Generate audit report
# ---------------------------------------------------------------------------

def generate_audit_report(
    df: pd.DataFrame,
    df_seq: pd.DataFrame,
    sampling: pd.DataFrame,
    loso_df: pd.DataFrame,
    config: dict,
    output_path: Path,
):
    """Generate audit report JSON and summary MD."""
    print("\n[Phase 7] Generating audit report...", flush=True)

    qf = df["quality_flags"]
    flag_stats = {}
    for bit_val, name in [
        (1, "nan_inf"), (2, "sample_count_error"), (4, "filter_warmup"),
        (8, "clipping_saturation"), (16, "flat_channel"), (32, "high_amplitude_artifact"),
        (64, "missing_channel"), (128, "time_discontinuity"), (256, "annotation_problem"),
    ]:
        count = int(((qf & bit_val) != 0).sum())
        flag_stats[name] = {"bit": bit_val, "count": count, "pct": round(count / len(df) * 100, 4)}

    clean_count = int((qf == 0).sum())
    flag_stats["clean"] = {"count": clean_count, "pct": round(clean_count / len(df) * 100, 4)}

    seq_stats = {
        "total_sequences": len(df_seq),
        "sequence_target_dist": {str(k): int(v) for k, v in df_seq["sequence_target"].value_counts().sort_index().to_dict().items()},
        "sequence_prediction_mask_true": int(df_seq["sequence_prediction_mask"].sum()),
        "has_boundary_window": int(df_seq["has_boundary_window"].sum()),
        "has_ictal_window": int(df_seq["has_ictal_window"].sum()),
        "eligible_for_sampling": int(df_seq[df_seq["sequence_prediction_mask"] & df_seq["sequence_target"].isin([0, 1])].shape[0]),
    }

    samp_stats = {
        "total_sampled": len(sampling),
        "interictal": int((sampling["sequence_target"] == 0).sum()) if len(sampling) > 0 else 0,
        "preictal": int((sampling["sequence_target"] == 1).sum()) if len(sampling) > 0 else 0,
        "ratio": round(
            (sampling["sequence_target"] == 0).sum() / max((sampling["sequence_target"] == 1).sum(), 1), 2
        ) if len(sampling) > 0 else 0,
    }

    loso_stats = {
        "n_folds": len(loso_df),
        "folds": loso_df[["fold_id", "test_subject", "val_subject"]].to_dict("records"),
    }

    audit = {
        "pipeline": "Stage 3 -- Sequence Build",
        "data_version": "CHBMIT_DataFreeze_V2.2",
        "schema_version": "2.2",
        "generated_at": datetime.now().isoformat(),
        "changes_from_stage2": [
            "sequence_target = anchor (last window) prediction_target (not aggregation)",
            "sequence_prediction_mask = endpoint prediction_loss_mask AND no boundary AND no ICTAL",
            "sequence_loss_weight = endpoint loss_weight",
            "Per-LOSO-fold normalization stats (no group-holdout reuse)",
            "Per-LOSO-fold 2:1 sampling manifests",
            "Configurable signal_root",
        ],
        "unchanged_from_stage2": [
            "metadata_all.parquet/csv",
            "labels_prediction.npy",
            "labels_state.npy",
            "interictal_candidate_pool.parquet",
            "quality/*",
            "group_holdout_split.parquet",
            "signal NPY files",
        ],
        "config": config,
        "window_stats": {
            "total_windows": len(df),
            "quality_flags": flag_stats,
        },
        "sequence_stats": seq_stats,
        "sampling_stats": samp_stats,
        "loso_stats": loso_stats,
    }

    with open(output_path / "audit_report.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)

    lines = [
        "# Stage 3 Audit Summary",
        "",
        f"**Generated:** {audit['generated_at']}",
        f"**Pipeline:** Stage 3 -- Sequence Build",
        "",
        "## Changes from Stage 2",
        "",
    ]
    for c in audit["changes_from_stage2"]:
        lines.append(f"- {c}")

    lines.extend([
        "",
        "## Unchanged from Stage 2",
        "",
    ])
    for c in audit["unchanged_from_stage2"]:
        lines.append(f"- {c}")

    lines.extend([
        "",
        "## Sequence Statistics",
        "",
        f"- Total sequences: {seq_stats['total_sequences']}",
        f"- sequence_target dist: {seq_stats['sequence_target_dist']}",
        f"- sequence_prediction_mask=True: {seq_stats['sequence_prediction_mask_true']}",
        f"- has_boundary_window=True: {seq_stats['has_boundary_window']}",
        f"- has_ictal_window=True: {seq_stats['has_ictal_window']}",
        f"- Eligible for sampling: {seq_stats['eligible_for_sampling']}",
        "",
        "## Sampling Statistics",
        "",
        f"- Total sampled: {samp_stats['total_sampled']}",
        f"- INTERICTAL: {samp_stats['interictal']}",
        f"- PREICTAL: {samp_stats['preictal']}",
        f"- Ratio: {samp_stats['ratio']}:1",
        "",
        "## LOSO",
        "",
        f"- Folds: {loso_stats['n_folds']}",
        f"- Each fold has independent normalization + sampling",
        "",
    ])

    with open(output_path / "audit_summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"  Audit report saved", flush=True)


# ---------------------------------------------------------------------------
# Phase 8: Generate SHA256
# ---------------------------------------------------------------------------

def generate_sha256(output_path: Path):
    """Generate SHA256SUMS.txt."""
    print("\n[Phase 8] Generating SHA256 checksums...", flush=True)

    sums = compute_sha256_sums(output_path)

    with open(output_path / "SHA256SUMS.txt", "w") as f:
        for rel, digest in sums:
            f.write(f"{digest}  {rel}\n")

    print(f"  {len(sums)} files checksummed", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Sequence Build Pipeline")
    parser.add_argument("--config", default="Preprocessing/config.yaml", help="Config YAML path")
    parser.add_argument("--stage2-dir", default="artifacts/label_refinement", help="Stage 2 output directory")
    parser.add_argument("--output", default="artifacts/sequence_build", help="Stage 3 output directory")
    args = parser.parse_args()

    t0 = time.time()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    stage2_dir = Path(args.stage2_dir)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    signal_root = config.get("serialization", {}).get(
        "signal_output_dir",
        config.get("signal_root", "E:/epilepsy_v2_signals")
    )

    print(f"=== Stage 3: Sequence Build ===", flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Stage 2 baseline: {stage2_dir}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Signal root: {signal_root}", flush=True)

    # Phase 1: Load Stage 2 metadata
    df = load_stage2_metadata(stage2_dir)

    # Phase 2: Rebuild sequence_index
    seq_length = config["sequence"]["length"]
    seq_stride = config["sequence"]["stride"]
    step_samp = int(
        config["windowing"]["length_sec"] * config["sampling"]["expected_fs_hz"] *
        (1 - config["windowing"]["overlap"])
    )
    min_valid_fraction = config["sequence"]["min_valid_fraction"]

    df_seq = rebuild_sequence_index(df, seq_length, seq_stride, step_samp, min_valid_fraction)

    # Create output subdirectories
    for subdir in ["folds", "quality", "normalization"]:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    # Save sequence index
    df_seq.to_parquet(str(output_path / "sequence_index.parquet"), index=False)
    print(f"  Saved sequence_index.parquet ({len(df_seq)} sequences)", flush=True)

    train_seqs = df_seq[df_seq["split"] == "train"]
    train_seqs.to_parquet(
        str(output_path / "folds" / "group_holdout_train_all_sequences.parquet"),
        index=False
    )
    print(f"  Saved group_holdout_train_all_sequences.parquet ({len(train_seqs)} sequences)", flush=True)

    # Phase 3: Rebuild group_holdout sampling
    ratio = config["sampling_balance"]["interictal_to_preictal_ratio"]
    seed = config["sampling_balance"].get("seed", 42)
    sampling = rebuild_group_holdout_sampling(df_seq, ratio, seed)
    sampling.to_parquet(
        str(output_path / "folds" / "group_holdout_train_sampling.parquet"),
        index=False
    )
    print(f"  Saved group_holdout_train_sampling.parquet ({len(sampling)} sequences)", flush=True)

    sampling_config = {
        "ratio": ratio,
        "seed": seed,
        "strategy": "balanced_2to1",
        "eligible_criteria": "split=train AND sequence_prediction_mask=True AND sequence_target in {0,1}",
    }
    with open(output_path / "sampling_config.yaml", "w") as f:
        yaml.dump(sampling_config, f, default_flow_style=False)

    # Phase 4: Generate LOSO folds
    n_channels = len(config["channels"]["target_22"])
    loso_df = generate_loso_folds(df, df_seq, output_path, ratio, seed, n_channels)

    # Phase 5: Copy unchanged artifacts
    copy_unchanged_artifacts(stage2_dir, output_path)

    # Phase 6: Update config
    config = update_config(config, signal_root)
    with open(output_path / "config_resolved.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  Saved config_resolved.yaml (data_version={config['data_version']})", flush=True)

    # Phase 7: Generate audit report
    generate_audit_report(df, df_seq, sampling, loso_df, config, output_path)

    # Phase 8: Generate SHA256
    generate_sha256(output_path)

    elapsed = time.time() - t0
    print(f"\n=== Stage 3 Complete ({elapsed:.1f}s) ===", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
