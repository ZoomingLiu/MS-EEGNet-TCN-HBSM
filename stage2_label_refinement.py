#!/usr/bin/env python
"""
stage2_label_refinement.py -- Stage 2: Label & Mask Refinement Pipeline.

Reuses existing NPY signal files from Stage 1. Does NOT re-read or re-filter EDFs.

Features:
  1. boundary_flag threshold = 0.8 (min_supervised_overlap)
  2. Split prediction_loss_mask and state_loss_mask
  3. Rebuild interictal_candidate_pool with correct filters
  4. Fix quality statistics: ((quality_flags & bit) != 0).sum()
  5. Explicit subject mapping (chb01+chb21=subject_01, etc.)
  6. Strict sequence continuity checks
  7. 2:1 sampling from qualified train sequences only
  8. Streaming normalization over all valid train windows
  9. Generate subject-level LOSO folds
 10. Report seizure counts (scanned, retained, lost)

Usage:
  python Preprocessing/stage2_label_refinement.py \
      --config Preprocessing/config.yaml \
      --existing-metadata artifacts/data_freeze_v2/metadata_all.parquet \
      --output artifacts/label_refinement/
"""

import argparse
import hashlib
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Preprocessing._utils import (
    QUALITY_FLAGS,
    LABEL_UNKNOWN, LABEL_INTERICTAL, LABEL_TRANSITION, LABEL_PREICTAL, LABEL_ICTAL,
    LABEL_NAMES,
    MIN_SUPERVISED_OVERLAP,
    parse_time_string,
    parse_summary_enhanced,
    build_patient_timeline,
    get_all_seizures_absolute,
    cluster_seizures,
    build_label_zones_absolute,
    assign_window_label_via_zones,
    compute_prediction_target,
)


# ============================================================================
# STAGE 2 MAIN PIPELINE
# ============================================================================

def run_label_refinement(config: dict, output_dir: str,
                         existing_metadata_path: str = None,
                         max_cases: int = None) -> dict:
    """Run Stage 2 pipeline reusing existing NPY signals from Stage 1."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for subdir in ["quality", "folds", "normalization"]:
        (output_path / subdir).mkdir(exist_ok=True)

    root = Path(config["dataset"]["root"])
    subject_mapping = config["subject_mapping"]
    signal_root = Path(config.get("serialization", {}).get(
        "signal_output_dir", str(output_path / "signals")))

    # Determine case directories
    all_case_dirs = sorted([d for d in root.iterdir()
                            if d.is_dir() and d.name.startswith("chb")])
    if max_cases:
        case_dirs = all_case_dirs[:max_cases]
    else:
        case_dirs = all_case_dirs

    # ---- Phase 1: Load existing Stage 1 metadata ----
    print("\n[Phase 1] Loading existing Stage 1 metadata...", flush=True)

    if existing_metadata_path is None:
        existing_metadata_path = str(Path("artifacts/data_freeze_v2") / "metadata_all.parquet")

    old_df = pd.read_parquet(existing_metadata_path)
    print(f"  Loaded {len(old_df)} windows from {existing_metadata_path}", flush=True)

    if max_cases:
        case_dirs_names = [d.name for d in case_dirs]
        old_df = old_df[old_df["case_id"].isin(case_dirs_names)].copy()
        print(f"  Filtered to {max_cases} cases: {len(old_df)} windows", flush=True)

    # Verify signal files exist
    sample_paths = old_df["signal_path"].unique()
    missing_files = [p for p in sample_paths[:10] if not Path(p).exists()]
    if missing_files:
        print(f"  WARNING: {len(missing_files)} signal files missing (sampled first 10)", flush=True)

    # ---- Phase 2: Re-parse summaries -> timelines -> zones ----
    print("\n[Phase 2] Re-parsing summaries and building label zones...", flush=True)

    all_timelines = {}
    all_zones = {}
    all_clusters = {}
    total_seizures_scanned = 0
    total_seizures_in_excluded = 0
    total_seizures_retained = 0

    existing_records = set()
    for _, row in old_df[["case_id", "record_id"]].drop_duplicates().iterrows():
        existing_records.add((row["case_id"], row["record_id"]))

    for case_dir in case_dirs:
        case_id = case_dir.name
        summary_path = case_dir / f"{case_id}-summary.txt"
        if summary_path.exists():
            parsed = parse_summary_enhanced(str(summary_path))
            timeline = build_patient_timeline(parsed, case_id)
        else:
            timeline = []

        all_timelines[case_id] = timeline
        all_sz = get_all_seizures_absolute(timeline)
        clusters = cluster_seizures(all_sz, config["seizure_cluster"]["gap_threshold_min"])
        all_clusters[case_id] = clusters

        zones = build_label_zones_absolute(
            all_sz, timeline,
            preictal_min=config["preictal"]["duration_min"],
            trans_start_min=config["transition"]["start_min_before_onset"],
            trans_end_min=config["transition"]["end_min_before_onset"],
            interictal_before_min=config["interictal"]["min_minutes_before_next_seizure"],
            interictal_after_min=config["interictal"]["min_minutes_after_prev_seizure_end"],
            postictal_min=config["postictal"]["censoring_min"],
        )
        all_zones[case_id] = zones

        for entry in timeline:
            total_seizures_scanned += len(entry["seizures_abs"])
            if (case_id, entry["record_id"]) not in existing_records:
                total_seizures_in_excluded += len(entry["seizures_abs"])
            else:
                total_seizures_retained += len(entry["seizures_abs"])

    print(f"  Cases scanned: {len(case_dirs)}")
    print(f"  Total seizures scanned: {total_seizures_scanned}")
    print(f"  Seizures in retained records: {total_seizures_retained}")
    print(f"  Seizures lost to exclusion: {total_seizures_in_excluded}")

    # ---- Phase 3: Re-assign window labels with 0.8 threshold ----
    print("\n[Phase 3] Re-assigning window labels (threshold=0.8)...")

    timeline_lookup = {}
    for case_id, timeline in all_timelines.items():
        for entry in timeline:
            timeline_lookup[(case_id, entry["record_id"])] = entry

    new_records = []
    for idx, row in old_df.iterrows():
        case_id = row["case_id"]
        record_id = row["record_id"]

        tl_entry = timeline_lookup.get((case_id, record_id))
        if tl_entry is None:
            w_abs_start = row["absolute_start_sec"]
            w_abs_end = row["absolute_end_sec"]
        else:
            file_abs_start = tl_entry["absolute_start_sec"]
            w_abs_start = file_abs_start + row["window_start_sec"]
            w_abs_end = file_abs_start + row["window_end_sec"]

        zones = all_zones.get(case_id, [])
        label_info = assign_window_label_via_zones(w_abs_start, w_abs_end, zones)

        quality_flags = int(row["quality_flags"])
        is_valid = bool(row["is_valid"])
        quality_score = float(row["quality_score"])

        subject_uid = subject_mapping.get(case_id, case_id)

        splits_cfg = config["splits"]["group_holdout"]
        if case_id in splits_cfg["train_cases"]:
            split = "train"
        elif case_id in splits_cfg["val_cases"]:
            split = "validation"
        elif case_id in splits_cfg["test_cases"]:
            split = "test"
        else:
            split = "train"

        pred_target = compute_prediction_target(label_info["state_label"])

        state_label = label_info["state_label"]
        if state_label == LABEL_TRANSITION:
            lw = config["transition"]["loss_weight"]
        elif state_label == LABEL_PREICTAL:
            lw = config["preictal"]["loss_weight"]
        elif state_label == LABEL_INTERICTAL:
            lw = config["interictal"]["loss_weight"]
        else:
            lw = 0.0

        not_warmup = (quality_flags & QUALITY_FLAGS["filter_warmup"]) == 0
        max_overlap_ok = label_info["maximum_overlap_fraction"] >= MIN_SUPERVISED_OVERLAP

        prediction_loss_mask = (
            is_valid and
            pred_target >= 0 and
            lw > 0 and
            not_warmup and
            max_overlap_ok
        )

        state_loss_mask = (
            is_valid and
            state_label >= 0 and
            not_warmup
        )

        new_records.append({
            "window_id": row["window_id"],
            "case_id": case_id,
            "subject_uid": subject_uid,
            "record_id": record_id,
            "continuous_block_id": int(row["continuous_block_id"]),
            "split": split,
            "signal_path": row["signal_path"],
            "start_sample": int(row["start_sample"]),
            "end_sample": int(row["end_sample"]),
            "window_start_sec": float(row["window_start_sec"]),
            "window_end_sec": float(row["window_end_sec"]),
            "absolute_start_sec": float(w_abs_start),
            "absolute_end_sec": float(w_abs_end),
            "inference_timestamp_sec": float(row["inference_timestamp_sec"]),
            "state_label": state_label,
            "prediction_target": pred_target,
            "loss_weight": float(lw),
            "prediction_loss_mask": prediction_loss_mask,
            "state_loss_mask": state_loss_mask,
            "quality_flags": quality_flags,
            "quality_score": quality_score,
            "is_valid": is_valid,
            "overlap_interictal": float(label_info["overlap_interictal"]),
            "overlap_transition": float(label_info["overlap_transition"]),
            "overlap_preictal": float(label_info["overlap_preictal"]),
            "overlap_ictal": float(label_info["overlap_ictal"]),
            "overlap_unknown": float(label_info["overlap_unknown"]),
            "maximum_overlap_fraction": float(label_info["maximum_overlap_fraction"]),
            "boundary_flag": label_info["boundary_flag"],
            "fs": float(row["fs"]),
            "n_channels": 22,
            "n_samples": int(row["end_sample"]) - int(row["start_sample"]),
        })

    df = pd.DataFrame(new_records)
    print(f"  Total windows: {len(df)}")

    label_dist = df["state_label"].value_counts().sort_index()
    print(f"  Label distribution:")
    for lbl, cnt in label_dist.items():
        print(f"    {LABEL_NAMES.get(lbl, str(lbl))}: {cnt}")

    print(f"  prediction_loss_mask=True: {df['prediction_loss_mask'].sum()}")
    print(f"  state_loss_mask=True: {df['state_loss_mask'].sum()}")
    print(f"  boundary_flag=True (max_overlap<0.8): {df['boundary_flag'].sum()}")

    # ---- Phase 4: Build sequence index with STRICT checks ----
    print("\n[Phase 4] Building sequence index (L=24, strict)...")

    seq_length = config["sequence"]["length"]
    seq_stride = config["sequence"]["stride"]
    step_samp = int(config["windowing"]["length_sec"] * config["sampling"]["expected_fs_hz"] *
                    (1 - config["windowing"]["overlap"]))

    sequence_records = []
    seq_global_id = 0

    for (case_id, record_id, block_id, split), group in df.groupby(
            ["case_id", "record_id", "continuous_block_id", "split"]):
        group = group.sort_values("start_sample").reset_index(drop=True)

        for i in range(0, len(group) - seq_length + 1, seq_stride):
            seq_wins = group.iloc[i:i + seq_length]

            # STRICT CHECK 1: start_sample diff must be exactly step_samp
            diffs = seq_wins["start_sample"].diff().dropna()
            if len(diffs) > 0 and not all(d == step_samp for d in diffs):
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

            # STRICT CHECK 6: no invalid windows in sequence
            valid_frac = seq_wins["is_valid"].mean()
            if valid_frac < config["sequence"]["min_valid_fraction"]:
                continue

            has_boundary = seq_wins["boundary_flag"].any()

            pred_targets = seq_wins["prediction_target"].tolist()
            supervised_targets = [t for t in pred_targets if t >= 0]

            if not supervised_targets:
                seq_target = -1
            elif any(t == 1 for t in supervised_targets):
                seq_target = 1
            else:
                seq_target = 0

            seq_pred_mask = seq_wins["prediction_loss_mask"].tolist()
            seq_state_mask = seq_wins["state_loss_mask"].tolist()

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
                "has_boundary_window": bool(has_boundary),
                "n_valid": int(seq_wins["is_valid"].sum()),
                "n_prediction_masked": int(sum(seq_pred_mask)),
                "n_state_masked": int(sum(seq_state_mask)),
                "start_samples": list(seq_wins["start_sample"].tolist()),
                "state_labels": list(seq_wins["state_label"].tolist()),
                "prediction_targets": pred_targets,
                "loss_weights": list(seq_wins["loss_weight"].tolist()),
                "prediction_loss_masks": seq_pred_mask,
                "state_loss_masks": seq_state_mask,
                "quality_flags": list(seq_wins["quality_flags"].tolist()),
            })
            seq_global_id += 1

    df_seq = pd.DataFrame(sequence_records)
    print(f"  Total sequences: {len(df_seq)}")
    print(f"  Sequence target distribution: {df_seq['sequence_target'].value_counts().to_dict()}")
    print(f"  Sequences with boundary windows: {df_seq['has_boundary_window'].sum()}")

    # ---- Phase 5: Build folds ----
    print("\n[Phase 5] Building folds (group holdout + LOSO)...")

    folds_dir = output_path / "folds"

    split_manifest = df[["window_id", "case_id", "subject_uid", "split"]].drop_duplicates()
    split_manifest.to_parquet(str(folds_dir / "group_holdout_split.parquet"), index=False)

    subjects = sorted(df["subject_uid"].unique())
    loso_folds = []
    for i, test_subj in enumerate(subjects):
        fold = {
            "fold_id": i,
            "test_subject": test_subj,
            "train_subjects": [s for s in subjects if s != test_subj],
            "test_cases": sorted(df[df["subject_uid"] == test_subj]["case_id"].unique().tolist()),
            "train_cases": sorted(df[df["subject_uid"] != test_subj]["case_id"].unique().tolist()),
        }
        loso_folds.append(fold)

    loso_df = pd.DataFrame(loso_folds)
    loso_df.to_parquet(str(folds_dir / "loso_folds.parquet"), index=False)
    print(f"  Group holdout split: {len(split_manifest)} windows")
    print(f"  LOSO folds: {len(loso_folds)} folds ({len(subjects)} subjects)")

    # ---- Phase 6: Normalization (streaming over ALL valid train windows) ----
    print("\n[Phase 6] Fitting normalization (streaming, all train windows)...", flush=True)

    train_valid = df[(df["split"] == "train") & (df["is_valid"])]
    n_train_windows = len(train_valid)
    print(f"  Valid train windows: {n_train_windows}", flush=True)

    channel_sum = np.zeros(22, dtype=np.float64)
    channel_sq_sum = np.zeros(22, dtype=np.float64)
    total_samples = 0
    processed = 0

    for sig_path, group in train_valid.groupby("signal_path"):
        try:
            sig = np.load(sig_path, mmap_mode="r")
        except Exception as e:
            print(f"  WARNING: Cannot load {sig_path}: {e}", flush=True)
            continue

        full_sig = np.asarray(sig, dtype=np.float64)
        n_total_samples = full_sig.shape[1]

        cumsum = np.zeros((22, n_total_samples + 1), dtype=np.float64)
        cumsum[:, 1:] = np.cumsum(full_sig, axis=1)
        cumsum_sq = np.zeros((22, n_total_samples + 1), dtype=np.float64)
        cumsum_sq[:, 1:] = np.cumsum(full_sig ** 2, axis=1)

        starts = group["start_sample"].to_numpy().astype(int)
        ends = group["end_sample"].to_numpy().astype(int)

        window_sums = cumsum[:, ends] - cumsum[:, starts]
        window_sq_sums = cumsum_sq[:, ends] - cumsum_sq[:, starts]

        channel_sum += window_sums.sum(axis=1)
        channel_sq_sum += window_sq_sums.sum(axis=1)
        total_samples += (ends - starts).sum()

        processed += len(group)
        if processed % 100000 == 0:
            print(f"  Processed {processed}/{n_train_windows} windows...", flush=True)

        del full_sig, cumsum, cumsum_sq

    if total_samples > 0:
        global_mean = channel_sum / total_samples
        global_var = channel_sq_sum / total_samples - global_mean ** 2
        global_std = np.sqrt(np.maximum(global_var, 1e-8))
    else:
        global_mean = np.zeros(22, dtype=np.float32)
        global_std = np.ones(22, dtype=np.float32)

    norm_stats = {
        "mean": global_mean.astype(np.float32),
        "std": global_std.astype(np.float32),
    }

    norm_dir = output_path / "normalization"
    np.savez(str(norm_dir / "group_holdout_stats.npz"),
             mean=norm_stats["mean"], std=norm_stats["std"],
             channel_order=config["channels"]["target_22"],
             n_windows=n_train_windows,
             method="streaming_all_valid_train")
    print(f"  Normalization fitted on {n_train_windows} train windows (streaming)")

    # ---- Phase 7: INTERICTAL candidate pool and sampling manifest ----
    print("\n[Phase 7] Building INTERICTAL pool and sampling manifest...")

    interictal_df = df[
        (df["state_label"] == LABEL_INTERICTAL) &
        (df["is_valid"] == True) &
        (df["prediction_loss_mask"] == True)
    ].copy()
    print(f"  INTERICTAL candidates (valid + masked): {len(interictal_df)}")

    train_seq = df_seq[
        (df_seq["split"] == "train") &
        (df_seq["sequence_target"].isin([0, 1])) &
        (~df_seq["has_boundary_window"])
    ].copy()

    train_preictal_seq = train_seq[train_seq["sequence_target"] == 1]
    train_interictal_seq = train_seq[train_seq["sequence_target"] == 0]

    n_preictal = len(train_preictal_seq)
    target_ratio = config["sampling_balance"]["interictal_to_preictal_ratio"]
    n_interictal_target = min(
        int(n_preictal * target_ratio),
        len(train_interictal_seq)
    )

    if n_interictal_target > 0 and len(train_interictal_seq) > 0:
        train_interictal_sampled = train_interictal_seq.sample(
            n=n_interictal_target, random_state=config["seed"])
    else:
        train_interictal_sampled = train_interictal_seq.iloc[:0]

    sampling_manifest = pd.concat([train_preictal_seq, train_interictal_sampled])
    actual_ratio = len(train_interictal_sampled) / max(n_preictal, 1)

    print(f"  Qualified train PREICTAL sequences: {n_preictal}")
    print(f"  Qualified train INTERICTAL sequences: {len(train_interictal_seq)}")
    print(f"  INTERICTAL sampled: {len(train_interictal_sampled)}")
    print(f"  Actual ratio: {actual_ratio:.2f}:1")

    sampling_manifest.to_parquet(
        str(folds_dir / "group_holdout_train_sampling.parquet"), index=False)
    train_seq.to_parquet(
        str(folds_dir / "group_holdout_train_all_sequences.parquet"), index=False)

    # ---- Phase 8: Save all outputs ----
    print("\n[Phase 8] Saving outputs...")

    df.to_csv(str(output_path / "metadata_all.csv"), index=False)
    df.to_parquet(str(output_path / "metadata_all.parquet"), index=False)

    labels_state = df["state_label"].to_numpy(dtype=np.int64)
    labels_pred = df["prediction_target"].to_numpy(dtype=np.int64)
    np.save(str(output_path / "labels_state.npy"), labels_state)
    np.save(str(output_path / "labels_prediction.npy"), labels_pred)

    df_seq.to_parquet(str(output_path / "sequence_index.parquet"), index=False)

    interictal_df.to_parquet(
        str(output_path / "interictal_candidate_pool.parquet"), index=False)

    old_quality_dir = Path("artifacts/data_freeze_v2") / "quality"
    if old_quality_dir.exists():
        for qf in old_quality_dir.glob("*"):
            shutil.copy2(str(qf), str(output_path / "quality" / qf.name))

    sampling_config = {
        "interictal_to_preictal_ratio": target_ratio,
        "actual_ratio": float(actual_ratio),
        "n_preictal_seq": int(n_preictal),
        "n_interictal_sampled": int(len(train_interictal_sampled)),
        "seed": config["seed"],
        "per_epoch_resample": config["sampling_balance"]["per_epoch_resample"],
        "qualified_filter": "sequence_target in {0,1} AND NOT has_boundary_window",
    }
    with open(str(output_path / "sampling_config.yaml"), "w") as f:
        yaml.dump(sampling_config, f, default_flow_style=False)

    with open(str(output_path / "config_resolved.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ---- Phase 9: Audit report ----
    print("\n[Phase 9] Building audit report...")

    quality_flag_dist = {}
    for flag_name, flag_val in QUALITY_FLAGS.items():
        count = int(((df["quality_flags"] & flag_val) != 0).sum())
        quality_flag_dist[flag_name] = count

    split_dist = {}
    for sp in ["train", "validation", "test"]:
        sp_df = df[df["split"] == sp]
        split_dist[sp] = {
            "n_windows": len(sp_df),
            "n_subjects": sp_df["subject_uid"].nunique(),
            "cases": sorted(sp_df["case_id"].unique().tolist()),
            "n_prediction_masked": int(sp_df["prediction_loss_mask"].sum()),
            "n_state_masked": int(sp_df["state_loss_mask"].sum()),
            "n_boundary": int(sp_df["boundary_flag"].sum()),
        }

    audit_report = {
        "pipeline": "Stage 2 -- Label Refinement",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "signal_root": str(signal_root),
        "reuses_stage1_signals": True,
        "stage1_metadata_source": existing_metadata_path,
        "scan_results": {
            "n_cases": len(case_dirs),
            "n_edf_total": len(old_df["record_id"].unique()),
            "n_seizures_scanned": total_seizures_scanned,
            "n_seizures_retained": total_seizures_retained,
            "n_seizures_lost_to_exclusion": total_seizures_in_excluded,
            "n_signal_files": len(old_df["signal_path"].unique()),
        },
        "window_stats": {
            "total_windows": len(df),
            "n_valid": int(df["is_valid"].sum()),
            "n_invalid": int((~df["is_valid"]).sum()),
            "n_prediction_masked": int(df["prediction_loss_mask"].sum()),
            "n_state_masked": int(df["state_loss_mask"].sum()),
            "n_boundary": int(df["boundary_flag"].sum()),
        },
        "label_distribution": {LABEL_NAMES.get(k, str(k)): int(v)
                               for k, v in label_dist.items()},
        "split_distribution": split_dist,
        "sequence_stats": {
            "total_sequences": len(df_seq),
            "by_split": df_seq["split"].value_counts().to_dict() if len(df_seq) > 0 else {},
            "by_target": df_seq["sequence_target"].value_counts().to_dict() if len(df_seq) > 0 else {},
            "n_with_boundary": int(df_seq["has_boundary_window"].sum()) if len(df_seq) > 0 else 0,
        },
        "quality_flag_distribution": quality_flag_dist,
        "quality_counting_method": "((quality_flags & bit) != 0).sum()",
        "sampling": {
            "interictal_pool_size": len(interictal_df),
            "qualified_preictal_seq": n_preictal,
            "qualified_interictal_seq": len(train_interictal_seq),
            "interictal_sampled": len(train_interictal_sampled),
            "actual_ratio": float(actual_ratio),
            "target_ratio": float(target_ratio),
            "qualified_filter": "sequence_target in {0,1} AND NOT has_boundary_window",
        },
        "normalization": {
            "method": "streaming_all_valid_train",
            "n_train_windows": n_train_windows,
            "n_sampled": n_train_windows,
            "mean": norm_stats["mean"].tolist(),
            "std": norm_stats["std"].tolist(),
        },
        "mask_definitions": {
            "prediction_loss_mask": "is_valid AND prediction_target>=0 AND loss_weight>0 AND NOT warmup AND max_overlap>=0.8",
            "state_loss_mask": "is_valid AND state_label>=0 AND NOT warmup",
            "boundary_flag": "maximum_overlap_fraction < 0.8",
        },
        "subject_mapping": subject_mapping,
        "loso_folds": len(loso_folds),
        "clusters_per_patient": {k: len(v) for k, v in all_clusters.items()},
    }

    with open(str(output_path / "audit_report.json"), "w") as f:
        json.dump(audit_report, f, indent=2, default=str)

    # ---- Phase 10: SHA256 checksums ----
    print("\n[Phase 10] Computing SHA256 checksums...")

    sha_lines = []
    for fpath in sorted(output_path.rglob("*")):
        if fpath.is_file() and fpath.name != "SHA256SUMS.txt":
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            rel_path = fpath.relative_to(output_path)
            sha_lines.append(f"{h.hexdigest()}  {rel_path}")

    with open(str(output_path / "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(sha_lines))

    print(f"\n{'='*60}")
    print(f"  Stage 2 Complete!")
    print(f"  Output: {output_dir}")
    print(f"  Windows: {len(df)}")
    print(f"  Sequences: {len(df_seq)}")
    print(f"  LOSO folds: {len(loso_folds)}")
    print(f"  SHA256: {len(sha_lines)} files")
    print(f"{'='*60}")

    return audit_report


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Label Refinement Pipeline")
    parser.add_argument("--config", default="Preprocessing/config.yaml")
    parser.add_argument("--output", default="artifacts/label_refinement")
    parser.add_argument("--existing-metadata", default=None,
                        help="Path to existing Stage 1 metadata_all.parquet")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_label_refinement(config, args.output,
                         existing_metadata_path=args.existing_metadata,
                         max_cases=args.max_cases)


if __name__ == "__main__":
    main()
