#!/usr/bin/env python3
"""
validate.py -- Pipeline Validation Suite.

Validates all three stages of preprocessing:
  1. Metadata unchanged from baseline
  2. sequence_target == anchor (last window) prediction_target
  3. sequence_prediction_mask consistency
  4. No ICTAL in sampled sequences
  5. Sequence continuity (step=640, no cross)
  6. 2:1 sampling ratio
  7. LOSO per-fold artifacts
  8. Config fixes (data_version, warmup, signal_root)
  9. Quality flag bit counting
 10. SHA256 checksums

Usage:
  python Preprocessing/validate.py --data-dir artifacts/sequence_build --baseline-dir artifacts/label_refinement
"""

import sys
import json
import hashlib
from pathlib import Path

from eeg_pipeline.paths import project_root
from .configuration import load_config

import numpy as np
import pandas as pd
import yaml


LABEL_ICTAL = 3
LABEL_UNKNOWN = -1
EXPECTED_STEP = 640


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return passed


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Preprocessing Pipeline Output")
    parser.add_argument("--data-dir", default=str(project_root() / "data" / "processed" / "legacy_v2_2"), help="Final data directory")
    parser.add_argument("--baseline-dir", default=str(project_root() / "data" / "interim" / "stage2"), help="Stage 2 baseline directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    baseline_dir = Path(args.baseline_dir)

    print(f"=== Preprocessing Pipeline Validation ===")
    print(f"Data dir: {data_dir}")
    print(f"Baseline dir: {baseline_dir}")
    print()

    results = []

    # ---- Load data ----
    print("[Loading data...]")
    df = pd.read_parquet(str(data_dir / "metadata_all.parquet"))
    df_seq = pd.read_parquet(str(data_dir / "sequence_index.parquet"))
    sampling = pd.read_parquet(str(data_dir / "folds" / "group_holdout_train_sampling.parquet"))

    with open(data_dir / "config_resolved.yaml") as f:
        config = yaml.safe_load(f)

    print(f"  Windows: {len(df)}")
    print(f"  Sequences: {len(df_seq)}")
    print(f"  Sampled: {len(sampling)}")
    print()

    # ---- Check 1: metadata unchanged from baseline ----
    print("[Check 1: Metadata unchanged from baseline]")
    meta_sha = sha256_file(data_dir / "metadata_all.parquet")
    baseline_sha = sha256_file(baseline_dir / "metadata_all.parquet")
    results.append(check(
        "1a. metadata_all.parquet SHA256 matches baseline",
        meta_sha == baseline_sha,
        f"Data: {meta_sha[:16]}... Baseline: {baseline_sha[:16]}..."
    ))

    for fname in ["labels_prediction.npy", "labels_state.npy"]:
        data_sh = sha256_file(data_dir / fname)
        base_sh = sha256_file(baseline_dir / fname)
        results.append(check(
            f"1b. {fname} SHA256 matches baseline",
            data_sh == base_sh,
            f"Data: {data_sh[:16]}... Baseline: {base_sh[:16]}..."
        ))

    pool_sh = sha256_file(data_dir / "interictal_candidate_pool.parquet")
    pool_base = sha256_file(baseline_dir / "interictal_candidate_pool.parquet")
    results.append(check(
        "1c. interictal_candidate_pool.parquet SHA256 matches baseline",
        pool_sh == pool_base,
        f"Data: {pool_sh[:16]}... Baseline: {pool_base[:16]}..."
    ))

    # ---- Check 2: sequence_target == anchor window prediction_target ----
    print("\n[Check 2: sequence_target == anchor window prediction_target]")
    sample_idx = np.random.RandomState(42).choice(len(df_seq), min(1000, len(df_seq)), replace=False)
    mismatches = 0
    for idx in sample_idx:
        seq = df_seq.iloc[idx]
        pred_targets = seq["prediction_targets"]
        if isinstance(pred_targets, np.ndarray):
            anchor_target = pred_targets[-1]
        else:
            anchor_target = pred_targets[-1]
        if seq["sequence_target"] != anchor_target:
            mismatches += 1

    results.append(check(
        "2. sequence_target == anchor (last window) prediction_target",
        mismatches == 0,
        f"Checked {len(sample_idx)} sequences, {mismatches} mismatches"
    ))

    # ---- Check 3: sequence_prediction_mask consistency ----
    print("\n[Check 3: sequence_prediction_mask consistency]")
    mask_mismatches = 0
    for idx in sample_idx:
        seq = df_seq.iloc[idx]
        masks = seq["prediction_loss_masks"]
        if isinstance(masks, np.ndarray):
            endpoint_mask = bool(masks[-1])
        else:
            endpoint_mask = bool(masks[-1])
        expected_mask = endpoint_mask and (not seq["has_boundary_window"]) and (not seq["has_ictal_window"])
        if seq["sequence_prediction_mask"] != expected_mask:
            mask_mismatches += 1

    results.append(check(
        "3. sequence_prediction_mask = endpoint_mask AND NOT boundary AND NOT ictal",
        mask_mismatches == 0,
        f"Checked {len(sample_idx)} sequences, {mask_mismatches} mismatches"
    ))

    # ---- Check 4: No ICTAL in sampled sequences ----
    print("\n[Check 4: No ICTAL in sampled sequences]")
    if len(sampling) > 0:
        n_ictal = 0
        n_ictal_endpoint = 0
        n_unknown_endpoint = 0
        for _, seq in sampling.iterrows():
            state_labels = seq["state_labels"]
            if isinstance(state_labels, np.ndarray):
                state_labels = state_labels.tolist()
            if LABEL_ICTAL in state_labels:
                n_ictal += 1
            if state_labels[-1] == LABEL_ICTAL:
                n_ictal_endpoint += 1
            if state_labels[-1] == LABEL_UNKNOWN:
                n_unknown_endpoint += 1

        results.append(check(
            "4a. 0 sampled sequences contain ICTAL",
            n_ictal == 0,
            f"{n_ictal} sequences with ICTAL"
        ))
        results.append(check(
            "4b. 0 sampled sequences end with ICTAL",
            n_ictal_endpoint == 0,
            f"{n_ictal_endpoint} sequences with ICTAL endpoint"
        ))
        results.append(check(
            "4c. 0 sampled sequences end with UNKNOWN",
            n_unknown_endpoint == 0,
            f"{n_unknown_endpoint} sequences with UNKNOWN endpoint"
        ))

        all_masked = sampling["sequence_prediction_mask"].all()
        results.append(check(
            "4d. All sampled have sequence_prediction_mask=True",
            all_masked,
            f"{sampling['sequence_prediction_mask'].sum()}/{len(sampling)}"
        ))

        targets_valid = sampling["sequence_target"].isin([0, 1]).all()
        results.append(check(
            "4e. All sampled have target in {0,1}",
            targets_valid,
            f"Targets: {sampling['sequence_target'].value_counts().to_dict()}"
        ))
    else:
        results.append(check("4. Sampling manifest not empty", False, "Empty sampling"))

    # ---- Check 5: Sequence continuity ----
    print("\n[Check 5: Sequence continuity]")
    step_errors = 0
    for idx in sample_idx:
        seq = df_seq.iloc[idx]
        starts = seq["start_samples"]
        if isinstance(starts, np.ndarray):
            diffs = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
        else:
            diffs = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
        if not all(d == EXPECTED_STEP for d in diffs):
            step_errors += 1

    results.append(check(
        "5a. All sequences have step=640",
        step_errors == 0,
        f"Checked {len(sample_idx)} sequences, {step_errors} with wrong step"
    ))

    n_dup_seq = df_seq["sequence_id"].duplicated().sum()
    results.append(check(
        "5b. sequence_id unique",
        n_dup_seq == 0,
        f"{n_dup_seq} duplicates"
    ))

    n_dup_win = df["window_id"].duplicated().sum()
    results.append(check(
        "5c. window_id unique",
        n_dup_win == 0,
        f"{n_dup_win} duplicates"
    ))

    # ---- Check 6: 2:1 sampling ratio ----
    print("\n[Check 6: 2:1 sampling ratio]")
    if len(sampling) > 0:
        n_inter = (sampling["sequence_target"] == 0).sum()
        n_pre = (sampling["sequence_target"] == 1).sum()
        ratio = n_inter / max(n_pre, 1)
        results.append(check(
            "6. INTERICTAL:PREICTAL ratio = 2.0",
            abs(ratio - 2.0) < 0.01,
            f"{n_inter}:{n_pre} = {ratio:.2f}:1"
        ))

    # ---- Check 7: LOSO per-fold artifacts ----
    print("\n[Check 7: LOSO per-fold artifacts]")
    loso_folds = pd.read_parquet(str(data_dir / "folds" / "loso_folds.parquet"))
    n_folds = len(loso_folds)
    results.append(check(
        "7a. LOSO folds exist",
        n_folds > 0,
        f"{n_folds} folds"
    ))

    loso_dir = data_dir / "folds" / "loso"
    all_norm_exist = True
    all_sampling_exist = True
    all_split_exist = True
    norm_files = 0
    sampling_files = 0

    for i in range(n_folds):
        fold_dir = loso_dir / f"fold_{i:02d}"
        norm_path = fold_dir / "normalization" / "loso_stats.npz"
        samp_path = fold_dir / "train_sampling.parquet"
        split_path = fold_dir / "sequence_split.parquet"

        if not norm_path.exists():
            all_norm_exist = False
        else:
            norm_files += 1

        if not samp_path.exists():
            all_sampling_exist = False
        else:
            sampling_files += 1

        if not split_path.exists():
            all_split_exist = False

    results.append(check(
        "7b. All LOSO folds have normalization stats",
        all_norm_exist,
        f"{norm_files}/{n_folds} folds"
    ))
    results.append(check(
        "7c. All LOSO folds have sampling manifests",
        all_sampling_exist,
        f"{sampling_files}/{n_folds} folds"
    ))
    results.append(check(
        "7d. All LOSO folds have sequence splits",
        all_split_exist,
        ""
    ))

    if all_norm_exist and (data_dir / "normalization" / "group_holdout_stats.npz").exists():
        gh_norm = np.load(str(data_dir / "normalization" / "group_holdout_stats.npz"), allow_pickle=True)
        fold0_norm = np.load(str(loso_dir / "fold_00" / "normalization" / "loso_stats.npz"), allow_pickle=True)
        gh_mean = gh_norm["mean"]
        f0_mean = fold0_norm["mean"]
        means_different = not np.allclose(gh_mean, f0_mean)
        results.append(check(
            "7e. LOSO fold-0 normalization differs from group_holdout",
            means_different,
            f"Max diff: {np.max(np.abs(gh_mean - f0_mean)):.6f}"
        ))

    # ---- Check 8: Config fixes ----
    print("\n[Check 8: Config and doc fixes]")
    results.append(check(
        "8a. data_version contains V2.2",
        "2.2" in str(config.get("data_version", "")),
        f"data_version={config.get('data_version')}"
    ))

    warmup_sec = config.get("filtering", {}).get("warmup_sec", 0)
    results.append(check(
        "8b. warmup_sec = 2.0",
        warmup_sec == 2.0,
        f"warmup_sec={warmup_sec}"
    ))

    signal_root = config.get("signal_root") or config.get("serialization", {}).get("signal_output_dir", "")
    results.append(check(
        "8c. signal_root configurable in config",
        len(signal_root) > 0,
        f"signal_root={signal_root}"
    ))

    # ---- Check 9: Quality flag bit counting ----
    print("\n[Check 9: Quality flag bit counting]")
    with open(data_dir / "audit_report.json") as f:
        audit = json.load(f)

    qf = df["quality_flags"]
    audit_qf = audit.get("window_stats", {}).get("quality_flags", {})
    audit_correct = True
    for name, info in audit_qf.items():
        if name == "clean":
            continue
        bit = info["bit"]
        expected = int(((qf & bit) != 0).sum())
        if info["count"] != expected:
            audit_correct = False

    results.append(check(
        "9. Audit report quality stats use correct bit counting",
        audit_correct,
        ""
    ))

    # ---- Check 10: SHA256SUMS exists ----
    print("\n[Check 10: SHA256SUMS]")
    sha_path = data_dir / "SHA256SUMS.txt"
    results.append(check(
        "10. SHA256SUMS.txt exists",
        sha_path.exists(),
        ""
    ))

    # ---- Summary ----
    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} PASS")
    print("=" * 60)

    if n_pass < n_total:
        print("\nFAILED checks:")
        for i, r in enumerate(results):
            if not r:
                print(f"  Check {i+1}")
        sys.exit(1)
    else:
        print("\nAll checks PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
