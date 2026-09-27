#!/usr/bin/env python
"""
stage1_edf_to_signals.py -- Stage 1: EDF Reading, Causal Filtering, Windowing.

Processes ALL CHB-MIT EDF files:
  1. Scan case directories and parse summary files for seizure annotations
  2. Read EDF signals via MNE, map to 22 common bipolar channels
  3. Apply causal SOS Butterworth bandpass filter (0.5-45 Hz)
  4. Save per-EDF filtered signals as .npy (float32)
  5. Create 5-second sliding windows (50% overlap = 2.5s step)
  6. Assign window labels via label zone overlap
  7. Compute quality flags for each window
  8. Build group-holdout normalization (train-only)
  9. Build initial sequence index and 2:1 sampling manifest

Produces the signal files (reused by Stages 2 & 3) plus window metadata
and initial label assignments.

Usage:
  python -m preprocessing.stage1_edf_to_signals --config configs/preprocessing_legacy.yaml
  python -m preprocessing.stage1_edf_to_signals --config configs/preprocessing_legacy.yaml --max-cases 3
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eeg_pipeline.paths import project_root
from .configuration import load_config

import numpy as np
import pandas as pd
import yaml

from ._utils import (
    QUALITY_FLAGS,
    LABEL_UNKNOWN, LABEL_INTERICTAL, LABEL_TRANSITION,
    LABEL_PREICTAL, LABEL_ICTAL, LABEL_NAMES,
    parse_time_string,
    parse_summary_enhanced,
    parse_seizures_file,
    build_patient_timeline,
    get_all_seizures_absolute,
    cluster_seizures,
    build_label_zones_absolute,
    normalize_channel_name,
    read_edf_with_mne,
    apply_causal_bandpass,
    create_windows_metadata,
    assign_window_label_via_zones,
    compute_prediction_target,
    check_window_quality,
    build_subject_splits,
    fit_normalization,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Main Pipeline
# ======================================================================

def run_pipeline(config: dict, output_dir: str, max_cases: int = None) -> dict:
    """Run the complete Stage 1 pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for subdir in ["signals", "quality", "folds", "normalization"]:
        (output_path / subdir).mkdir(exist_ok=True)

    root = Path(config["dataset"]["root"])
    subject_mapping = config["subject_mapping"]
    target_22 = config["channels"]["target_22"]
    aliases = config["channels"].get("aliases", {})

    # ---- Phase 1: Scan all EDF files ----
    print("\n[Phase 1] Scanning all EDF files...")

    case_dirs = sorted([d for d in root.iterdir()
                        if d.is_dir() and d.name.startswith("chb")])
    if max_cases:
        case_dirs = case_dirs[:max_cases]

    all_edf_paths = []
    all_summaries = {}
    all_timelines = {}
    excluded_records = []
    annotation_issues = []
    channel_issues = []

    total_edf = 0
    total_with_seizures = 0
    total_seizures = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        summary_path = case_dir / f"{case_id}-summary.txt"
        edf_files = sorted(case_dir.glob("*.edf"))
        total_edf += len(edf_files)

        if summary_path.exists():
            parsed = parse_summary_enhanced(str(summary_path))
            all_summaries[case_id] = parsed
            timeline = build_patient_timeline(parsed, case_id)
            all_timelines[case_id] = timeline

            for entry in timeline:
                if entry["has_seizures"]:
                    total_with_seizures += 1
                    total_seizures += len(entry["seizures_abs"])
        else:
            all_summaries[case_id] = {"files": [], "errors": ["No summary file"]}
            timeline = []
            for edf in edf_files:
                timeline.append({
                    "filename": edf.name, "case_id": case_id,
                    "record_id": edf.stem,
                    "absolute_start_sec": 0, "absolute_end_sec": 3600,
                    "duration_sec": 3600, "continuous_block_id": 0,
                    "seizures_abs": [], "has_seizures": False,
                })
            all_timelines[case_id] = timeline

        # Cross-check .seizures files
        for edf in edf_files:
            sz_file = edf.with_suffix(edf.suffix + ".seizures")
            if sz_file.exists():
                sz_from_file = parse_seizures_file(str(sz_file))
                summary_sz = []
                if case_id in all_summaries:
                    for f_entry in all_summaries[case_id].get("files", []):
                        if f_entry["filename"] == edf.name:
                            summary_sz = f_entry.get("seizures", [])
                            break
                if sz_from_file and not summary_sz:
                    annotation_issues.append({
                        "case_id": case_id, "record_id": edf.stem,
                        "issue": "seizures in .seizures file but not in summary",
                        "from_file": sz_from_file,
                    })

        all_edf_paths.extend([(case_id, edf) for edf in edf_files])

    print(f"  Total cases: {len(case_dirs)}")
    print(f"  Total EDF files: {total_edf}")
    print(f"  Files with seizures: {total_with_seizures}")
    print(f"  Total seizures: {total_seizures}")

    # ---- Phase 2: Build seizure clusters per patient ----
    print("\n[Phase 2] Building seizure clusters...")

    all_clusters = {}
    for case_id, timeline in all_timelines.items():
        all_sz = get_all_seizures_absolute(timeline)
        clusters = cluster_seizures(all_sz, config["seizure_cluster"]["gap_threshold_min"])
        all_clusters[case_id] = clusters
        if clusters:
            print(f"  {case_id}: {len(all_sz)} seizures -> {len(clusters)} clusters")

    # ---- Phase 3: Build label zones per patient ----
    print("\n[Phase 3] Building label zones...")

    all_zones = {}
    for case_id, timeline in all_timelines.items():
        all_sz = get_all_seizures_absolute(timeline)
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

    # ---- Phase 4: Process EDF files (read, filter, save signals) ----
    print("\n[Phase 4] Processing EDF files (read, filter, save)...")

    signal_root = Path(config.get("serialization", {}).get(
        "signal_output_dir", str(output_path / "signals")))
    signal_root.mkdir(parents=True, exist_ok=True)
    print(f"  Signal output: {signal_root}")

    split_info = build_subject_splits(subject_mapping, config)

    window_records = []
    signal_files = {}
    window_global_id = 0
    processed_count = 0
    excluded_count = 0

    for case_id, edf_path in all_edf_paths:
        record_id = edf_path.stem
        timeline_entry = None
        for te in all_timelines.get(case_id, []):
            if te["record_id"] == record_id:
                timeline_entry = te
                break

        if timeline_entry is None:
            timeline_entry = {
                "filename": edf_path.name, "case_id": case_id,
                "record_id": record_id,
                "absolute_start_sec": 0, "absolute_end_sec": 3600,
                "duration_sec": 3600, "continuous_block_id": 0,
                "seizures_abs": [], "has_seizures": False,
            }

        # Read EDF with MNE
        signal_22, edf_info = read_edf_with_mne(str(edf_path), target_22, aliases)

        if signal_22 is None:
            excluded_count += 1
            excluded_records.append({
                "case_id": case_id, "record_id": record_id,
                "reason": "; ".join(edf_info.get("errors", ["unknown"])),
                "missing_channels": edf_info.get("missing_channels", []),
            })
            if edf_info.get("missing_channels"):
                channel_issues.append({
                    "case_id": case_id, "record_id": record_id,
                    "missing": edf_info["missing_channels"],
                    "original": edf_info.get("original_channels", []),
                })
            continue

        fs = edf_info["fs"]
        n_samples = edf_info["n_samples"]

        # Check sampling rate
        if abs(fs - config["sampling"]["expected_fs_hz"]) > 1.0:
            excluded_count += 1
            excluded_records.append({
                "case_id": case_id, "record_id": record_id,
                "reason": f"unexpected fs={fs} (expected {config['sampling']['expected_fs_hz']})",
            })
            continue

        # Apply causal filter
        filtered, warmup_n = apply_causal_bandpass(
            signal_22, fs,
            low=config["filtering"]["bandpass_low_hz"],
            high=config["filtering"]["bandpass_high_hz"],
            order=config["filtering"]["order"],
        )

        # Save signal as .npy
        case_signal_dir = signal_root / case_id
        case_signal_dir.mkdir(parents=True, exist_ok=True)
        signal_path = case_signal_dir / f"{record_id}.npy"
        np.save(str(signal_path), filtered.astype(np.float32))
        signal_files[(case_id, record_id)] = str(signal_path)

        processed_count += 1

        # Create windows
        windows_meta = create_windows_metadata(
            fs, n_samples,
            window_sec=config["windowing"]["length_sec"],
            overlap=config["windowing"]["overlap"],
        )

        zones = all_zones.get(case_id, [])
        file_abs_start = timeline_entry["absolute_start_sec"]
        block_id = timeline_entry["continuous_block_id"]

        # Assign labels and quality
        for w in windows_meta:
            w_abs_start = file_abs_start + w["window_start_sec"]
            w_abs_end = file_abs_start + w["window_end_sec"]

            label_info = assign_window_label_via_zones(w_abs_start, w_abs_end, zones)

            sig_slice = filtered[:, w["start_sample"]:w["end_sample"]]
            quality = check_window_quality(
                sig_slice, warmup_n, w["start_sample"], fs,
                thresholds={
                    "flat_diff": config["quality"]["thresholds"]["flat_channel_diff"],
                    "flat_frac": config["quality"]["thresholds"]["flat_channel_fraction"],
                    "sat_zscore": config["quality"]["thresholds"]["saturation_zscore"],
                    "sat_frac": config["quality"]["thresholds"]["saturation_fraction"],
                    "high_amp_zscore": config["quality"]["thresholds"].get("amplitude_outlier_zscore", 10.0),
                    "clipping_uv": config["quality"]["thresholds"]["clipping_uv"],
                }
            )

            supervised = (quality["supervised_loss_mask"] and
                          label_info["state_label"] != LABEL_UNKNOWN and
                          not label_info["boundary_flag"])

            subject_uid = subject_mapping.get(case_id, case_id)
            split = split_info["case_to_split"].get(case_id, "train")

            pred_target = compute_prediction_target(label_info["state_label"])

            if label_info["state_label"] == LABEL_TRANSITION:
                lw = config["transition"]["loss_weight"]
            elif label_info["state_label"] == LABEL_PREICTAL:
                lw = config["preictal"]["loss_weight"]
            elif label_info["state_label"] == LABEL_INTERICTAL:
                lw = config["interictal"]["loss_weight"]
            else:
                lw = 0.0

            window_id = f"{case_id}_{record_id}_{w['start_sample']:08d}"

            window_records.append({
                "window_id": window_id,
                "case_id": case_id,
                "subject_uid": subject_uid,
                "record_id": record_id,
                "continuous_block_id": block_id,
                "split": split,
                "signal_path": str(signal_path),
                "start_sample": w["start_sample"],
                "end_sample": w["end_sample"],
                "window_start_sec": w["window_start_sec"],
                "window_end_sec": w["window_end_sec"],
                "absolute_start_sec": w_abs_start,
                "absolute_end_sec": w_abs_end,
                "inference_timestamp_sec": w["inference_timestamp_sec"],
                "state_label": label_info["state_label"],
                "prediction_target": pred_target,
                "loss_weight": lw,
                "supervised_loss_mask": supervised,
                "quality_flags": quality["quality_flags"],
                "quality_score": quality["quality_score"],
                "is_valid": quality["is_valid"],
                "overlap_interictal": label_info["overlap_interictal"],
                "overlap_transition": label_info["overlap_transition"],
                "overlap_preictal": label_info["overlap_preictal"],
                "overlap_ictal": label_info["overlap_ictal"],
                "overlap_unknown": label_info["overlap_unknown"],
                "maximum_overlap_fraction": label_info["maximum_overlap_fraction"],
                "boundary_flag": label_info["boundary_flag"],
                "fs": fs,
                "n_channels": 22,
                "n_samples": w["end_sample"] - w["start_sample"],
            })
            window_global_id += 1

        if processed_count % 50 == 0:
            print(f"  Processed {processed_count}/{len(all_edf_paths)} files, "
                  f"{len(window_records)} windows, {excluded_count} excluded")

    print(f"\n  Total processed: {processed_count}")
    print(f"  Total excluded: {excluded_count}")
    print(f"  Total windows: {len(window_records)}")

    # ---- Phase 5: Build metadata DataFrame ----
    print("\n[Phase 5] Building metadata...")

    df = pd.DataFrame(window_records)

    label_dist = df["state_label"].value_counts().sort_index()
    print(f"  Label distribution:")
    for lbl, cnt in label_dist.items():
        print(f"    {LABEL_NAMES.get(lbl, str(lbl))}: {cnt}")

    # ---- Phase 6: Build sequences ----
    print("\n[Phase 6] Building TCN sequences (L=24)...")

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

            diffs = seq_wins["start_sample"].diff().dropna()
            if len(diffs) > 0 and not all(d == step_samp for d in diffs):
                continue

            if seq_wins["subject_uid"].nunique() > 1:
                continue

            valid_frac = seq_wins["is_valid"].mean()
            if valid_frac < config["sequence"]["min_valid_fraction"]:
                continue

            seq_id = f"seq_{seq_global_id:08d}"
            state_labels = seq_wins["state_label"].tolist()
            pred_targets = seq_wins["prediction_target"].tolist()
            loss_weights = seq_wins["loss_weight"].tolist()
            supervised_masks = seq_wins["supervised_loss_mask"].tolist()
            quality_flags_list = seq_wins["quality_flags"].tolist()

            has_preictal = any(t == 1 for t in pred_targets if t >= 0)
            seq_target = 1 if has_preictal else (0 if all(t == 0 for t in pred_targets if t >= 0) else -1)

            sequence_records.append({
                "sequence_id": seq_id,
                "window_ids": list(seq_wins["window_id"]),
                "subject_uid": seq_wins["subject_uid"].iloc[0],
                "case_id": case_id,
                "record_id": record_id,
                "continuous_block_id": block_id,
                "split": split,
                "sequence_start_time": seq_wins["absolute_start_sec"].iloc[0],
                "sequence_end_time": seq_wins["absolute_end_sec"].iloc[-1],
                "sequence_target": seq_target,
                "state_labels": str(state_labels),
                "prediction_targets": str(pred_targets),
                "loss_weights": str(loss_weights),
                "supervised_loss_mask": str(supervised_masks),
                "quality_flags": str(quality_flags_list),
                "n_valid": int(seq_wins["is_valid"].sum()),
                "start_samples": str(seq_wins["start_sample"].tolist()),
            })
            seq_global_id += 1

    df_seq = pd.DataFrame(sequence_records)
    print(f"  Total sequences: {len(df_seq)}")

    # ---- Phase 7: Normalization (train-only) ----
    print("\n[Phase 7] Fitting normalization (train-only)...")

    train_df = df[(df["split"] == "train") & (df["is_valid"])]
    if len(train_df) > 0:
        sample_size = min(5000, len(train_df))
        train_sample = train_df.sample(n=sample_size, random_state=42)
        norm_stats = fit_normalization(
            train_sample["signal_path"].tolist(),
            train_sample["start_sample"].tolist(),
            train_sample["end_sample"].tolist(),
        )
    else:
        norm_stats = {"mean": np.zeros(22, dtype=np.float32),
                      "std": np.ones(22, dtype=np.float32)}

    norm_dir = output_path / "normalization"
    np.savez(str(norm_dir / "group_holdout_stats.npz"),
             mean=norm_stats["mean"], std=norm_stats["std"],
             channel_order=target_22)
    print(f"  Normalization fitted on {len(train_df)} train windows")

    # ---- Phase 8: INTERICTAL candidate pool and sampling manifest ----
    print("\n[Phase 8] Building INTERICTAL pool and sampling manifests...")

    interictal_df = df[df["state_label"] == LABEL_INTERICTAL].copy()
    print(f"  INTERICTAL candidates: {len(interictal_df)}")

    train_seq = df_seq[df_seq["split"] == "train"].copy()
    train_preictal_seq = train_seq[train_seq["sequence_target"] == 1]
    train_interictal_seq = train_seq[train_seq["sequence_target"] == 0]

    n_preictal = len(train_preictal_seq)
    n_interictal_target = min(
        int(n_preictal * config["sampling_balance"]["interictal_to_preictal_ratio"]),
        len(train_interictal_seq)
    )

    if n_interictal_target > 0 and len(train_interictal_seq) > 0:
        train_interictal_sampled = train_interictal_seq.sample(
            n=n_interictal_target, random_state=42)
    else:
        train_interictal_sampled = train_interictal_seq

    sampling_manifest = pd.concat([train_preictal_seq, train_interictal_sampled])
    actual_ratio = len(train_interictal_sampled) / max(n_preictal, 1)
    print(f"  PREICTAL sequences: {n_preictal}")
    print(f"  INTERICTAL sampled: {len(train_interictal_sampled)}")
    print(f"  Actual ratio: {actual_ratio:.2f}:1")

    # ---- Phase 9: Save all outputs ----
    print("\n[Phase 9] Saving outputs...")

    df.to_csv(str(output_path / "metadata_all.csv"), index=False)
    df.to_parquet(str(output_path / "metadata_all.parquet"), index=False)

    labels_state = df["state_label"].to_numpy(dtype=np.int64)
    labels_pred = df["prediction_target"].to_numpy(dtype=np.int64)
    np.save(str(output_path / "labels_state.npy"), labels_state)
    np.save(str(output_path / "labels_prediction.npy"), labels_pred)

    df_seq.to_parquet(str(output_path / "sequence_index.parquet"), index=False)
    interictal_df.to_parquet(str(output_path / "interictal_candidate_pool.parquet"), index=False)

    folds_dir = output_path / "folds"
    sampling_manifest.to_parquet(str(folds_dir / "group_holdout_train_sampling.parquet"), index=False)

    split_manifest = df[["window_id", "case_id", "subject_uid", "split"]].drop_duplicates()
    split_manifest.to_parquet(str(folds_dir / "group_holdout_split.parquet"), index=False)

    pd.DataFrame(excluded_records).to_csv(str(output_path / "quality" / "excluded_records.csv"), index=False)
    pd.DataFrame(annotation_issues).to_csv(str(output_path / "quality" / "annotation_issues.csv"), index=False)
    pd.DataFrame(channel_issues).to_csv(str(output_path / "quality" / "channel_issues.csv"), index=False)

    sampling_config = {
        "interictal_to_preictal_ratio": config["sampling_balance"]["interictal_to_preictal_ratio"],
        "actual_ratio": float(actual_ratio),
        "n_preictal_seq": int(n_preictal),
        "n_interictal_sampled": int(len(train_interictal_sampled)),
        "seed": config["sampling_balance"]["seed"],
        "per_epoch_resample": config["sampling_balance"]["per_epoch_resample"],
    }
    with open(str(output_path / "sampling_config.yaml"), "w") as f:
        yaml.dump(sampling_config, f, default_flow_style=False)

    with open(str(output_path / "config_resolved.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ---- Phase 10: Audit report ----
    print("\n[Phase 10] Building audit report...")

    quality_flag_dist = {}
    for flag_name, flag_val in QUALITY_FLAGS.items():
        count = int((df["quality_flags"] & flag_val).sum())
        quality_flag_dist[flag_name] = count

    audit_report = {
        "pipeline": "Stage 1: EDF to Signals",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "scan_results": {
            "n_cases": len(case_dirs),
            "n_edf_total": total_edf,
            "n_edf_with_seizures": total_with_seizures,
            "n_seizures_total": total_seizures,
            "n_edf_processed": processed_count,
            "n_edf_excluded": excluded_count,
        },
        "window_stats": {
            "total_windows": len(df),
            "n_valid": int(df["is_valid"].sum()),
            "n_invalid": int((~df["is_valid"]).sum()),
            "n_supervised": int(df["supervised_loss_mask"].sum()),
        },
        "label_distribution": {LABEL_NAMES.get(k, str(k)): int(v)
                               for k, v in label_dist.items()},
        "split_distribution": df["split"].value_counts().to_dict(),
        "sequence_stats": {
            "total_sequences": len(df_seq),
            "by_split": df_seq["split"].value_counts().to_dict() if len(df_seq) > 0 else {},
            "by_target": df_seq["sequence_target"].value_counts().to_dict() if len(df_seq) > 0 else {},
        },
        "quality_flag_distribution": quality_flag_dist,
        "sampling": {
            "interictal_pool_size": len(interictal_df),
            "preictal_train_seq": n_preictal,
            "interictal_sampled": len(train_interictal_sampled),
            "actual_ratio": float(actual_ratio),
        },
        "normalization": {
            "n_train_windows": len(train_df),
            "n_sampled": sample_size if len(train_df) > 0 else 0,
            "mean": norm_stats["mean"].tolist(),
            "std": norm_stats["std"].tolist(),
        },
        "excluded_records_count": excluded_count,
        "subject_mapping": subject_mapping,
        "split_assignment": split_info["case_to_split"],
        "clusters_per_patient": {k: len(v) for k, v in all_clusters.items()},
    }

    with open(str(output_path / "audit_report.json"), "w") as f:
        json.dump(audit_report, f, indent=2, default=str)

    # Audit summary markdown
    md_lines = [
        "# Stage 1: EDF to Signals — Audit Report",
        f"\n**Generated:** {audit_report['run_at']}",
        f"**Dataset root:** {audit_report['dataset_root']}",
        "",
        "## Scan Results",
        f"- Cases: {audit_report['scan_results']['n_cases']}",
        f"- Total EDF files: {audit_report['scan_results']['n_edf_total']}",
        f"- Files with seizures: {audit_report['scan_results']['n_edf_with_seizures']}",
        f"- Total seizures: {audit_report['scan_results']['n_seizures_total']}",
        f"- Processed: {audit_report['scan_results']['n_edf_processed']}",
        f"- Excluded: {audit_report['scan_results']['n_edf_excluded']}",
        "",
        "## Window Statistics",
        f"- Total windows: {audit_report['window_stats']['total_windows']}",
        f"- Valid: {audit_report['window_stats']['n_valid']}",
        f"- Invalid: {audit_report['window_stats']['n_invalid']}",
        f"- Supervised: {audit_report['window_stats']['n_supervised']}",
        "",
        "## Label Distribution",
    ]
    for lbl, cnt in audit_report["label_distribution"].items():
        md_lines.append(f"- {lbl}: {cnt}")
    md_lines += ["", "## Split Distribution"]
    for sp, cnt in audit_report["split_distribution"].items():
        md_lines.append(f"- {sp}: {cnt}")
    md_lines += ["", "## Quality Flags"]
    for flag_name, cnt in audit_report["quality_flag_distribution"].items():
        md_lines.append(f"- {flag_name} (bit {QUALITY_FLAGS[flag_name]}): {cnt}")
    md_lines += ["", "## Sampling",
                 f"- INTERICTAL pool: {audit_report['sampling']['interictal_pool_size']}",
                 f"- PREICTAL train seq: {audit_report['sampling']['preictal_train_seq']}",
                 f"- INTERICTAL sampled: {audit_report['sampling']['interictal_sampled']}",
                 f"- Actual ratio: {audit_report['sampling']['actual_ratio']:.2f}:1",
                 "", "## Excluded Records"]
    for er in excluded_records[:20]:
        md_lines.append(f"- {er['case_id']}/{er['record_id']}: {er['reason']}")
    if len(excluded_records) > 20:
        md_lines.append(f"- ... and {len(excluded_records) - 20} more")

    with open(str(output_path / "audit_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # ---- Phase 11: SHA256 checksums ----
    print("\n[Phase 11] Computing SHA256 checksums...")

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
    print(f"  Stage 1 Complete!")
    print(f"  Output: {output_dir}")
    print(f"  Windows: {len(df)}")
    print(f"  Sequences: {len(df_seq)}")
    print(f"  Excluded records: {excluded_count}")
    print(f"  SHA256: {len(sha_lines)} files")
    print(f"{'='*60}")

    return audit_report


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: EDF Reading, Filtering, and Windowing")
    parser.add_argument("--config", default=str(project_root() / "configs" / "preprocessing_legacy.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--output", default=str(project_root() / "data" / "interim" / "stage1"),
                        help="Output directory")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)

    run_pipeline(config, args.output, max_cases=args.max_cases)


if __name__ == "__main__":
    main()
