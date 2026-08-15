"""
export_member4.py -- Export windows / seizures / monitoring duration
for downstream evaluation (member 4), in LOSO 23-fold format.

Outputs (to --out-dir, default artifacts/export_member4/):
  1. windows.pkl        -- pd.DataFrame pickle (one row per window, 1,370,088 rows)
                           * signal stored as POINTER (signal_path + start/end_sample)
                             -> load with np.load(mmap_mode="r"), see README_export.md
                             (embedding 22x1280 float32 per window would be ~154 GB)
                           * loso_split_codes: np.int8[23] per row, code 0=TRAIN 1=VAL 2=TEST
                             for folds 0..22 (derived from subject membership, identical
                             rule to folds/loso/fold_XX/sequence_split.parquet)
  2. seizures.pkl       -- list[dict]: seizure_id, patient_id, record_id, onset/offset
                           (record-relative AND absolute), cluster_id, subject_uid
  3. monitoring_duration.json -- {"fold_00": {"train": s, "val": s, "test": s}, ...}
                           pure monitoring seconds = recorded time - seizure intervals
                           - post-ictal censoring (config postictal.censoring_min)
  4. README_export.md   -- format doc + mmap loading snippet

Usage:
  python export_member4.py --config config.yaml \
      --data-dir ../artifacts/data_freeze_v2_2 --out-dir ../artifacts/export_member4
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from _utils import (
        parse_summary_enhanced,
        build_patient_timeline,
        get_all_seizures_absolute,
        cluster_seizures,
        LABEL_NAMES,
    )
except ImportError:  # run from project root
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _utils import (
        parse_summary_enhanced,
        build_patient_timeline,
        get_all_seizures_absolute,
        cluster_seizures,
        LABEL_NAMES,
    )

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}  # TRAIN/VAL/TEST per spec


# ----------------------------------------------------------------------
# Seizures: re-parse CHB-MIT summaries -> seizures.pkl (+ cluster ids)
# ----------------------------------------------------------------------

def build_seizure_table(config):
    root = Path(config["dataset"]["root"])
    subject_mapping = config["subject_mapping"]
    gap_min = config["seizure_cluster"]["gap_threshold_min"]

    seizure_rows = []
    # per-case structures for window association & monitoring duration
    case_info = {}

    for case in sorted(subject_mapping.keys()):
        summary_path = root / case / f"{case}-summary.txt"
        if not summary_path.exists():
            print(f"  [WARN] missing summary: {summary_path}")
            continue

        parsed = parse_summary_enhanced(str(summary_path))
        timeline = build_patient_timeline(parsed, case)
        all_sz = get_all_seizures_absolute(timeline)
        clusters = cluster_seizures(all_sz, gap_threshold_min=gap_min)

        record_start = {e["record_id"]: e["absolute_start_sec"] for e in timeline}

        # cluster id per seizure (by position in case-sorted order)
        seizure_cluster = []
        for cl in clusters:
            for sz in cl["seizures"]:
                seizure_cluster.append((sz, cl["cluster_id"]))

        case_seizures = []  # enriched, sorted by onset
        for idx, (sz, cluster_id) in enumerate(seizure_cluster):
            rec_start = record_start.get(sz["record_id"], 0.0)
            row = {
                "seizure_id": f"{case}_sz{idx:02d}",
                "patient_id": case,
                "subject_uid": subject_mapping[case],
                "record_id": sz["record_id"],
                "seizure_id_int": idx,
                "onset_sec": sz["onset_abs_sec"] - rec_start,   # record-relative
                "offset_sec": sz["offset_abs_sec"] - rec_start,
                "onset_abs_sec": sz["onset_abs_sec"],           # patient-timeline
                "offset_abs_sec": sz["offset_abs_sec"],
                "duration_sec": sz["offset_abs_sec"] - sz["onset_abs_sec"],
                "cluster_id": f"{case}_{cluster_id}",
            }
            seizure_rows.append(row)
            case_seizures.append(row)

        case_info[case] = {
            "timeline": timeline,
            "seizures": case_seizures,
            "subject_uid": subject_mapping[case],
        }
        print(f"  {case}: {len(case_seizures)} seizures, "
              f"{len(clusters)} clusters")

    return seizure_rows, case_info


# ----------------------------------------------------------------------
# Windows: metadata_all + seizure association + LOSO split codes
# ----------------------------------------------------------------------

def associate_seizures(df, case_info, config):
    """Assign seizure_id / cluster_id to windows overlapping the
    pre-seizure association zone [onset - trans_start, offset] of a seizure."""
    trans_start_sec = config["transition"]["start_min_before_onset"] * 60.0

    seizure_id = np.full(len(df), -1, dtype=np.int32)
    cluster_ids = np.full(len(df), "", dtype=object)
    sz_ids = np.full(len(df), "", dtype=object)

    case_index = {c: g.index for c, g in df.groupby("case_id", sort=False)}
    for case, info in case_info.items():
        if case not in case_index:
            continue
        idx = case_index[case].to_numpy()
        w_start = df.loc[idx, "absolute_start_sec"].to_numpy()
        w_end = df.loc[idx, "absolute_end_sec"].to_numpy()
        best_overlap = np.zeros(len(idx))

        for k, sz in enumerate(info["seizures"]):
            zone_start = sz["onset_abs_sec"] - trans_start_sec
            zone_end = sz["offset_abs_sec"]
            ov = np.minimum(w_end, zone_end) - np.maximum(w_start, zone_start)
            better = ov > best_overlap
            if better.any():
                best_overlap[better] = ov[better]
                seizure_id[idx[better]] = k
                sz_ids[idx[better]] = sz["seizure_id"]
                cluster_ids[idx[better]] = sz["cluster_id"]

    df = df.copy()
    df["seizure_id"] = sz_ids
    df["seizure_id_int"] = seizure_id
    df["cluster_id"] = cluster_ids
    return df


def add_loso_split_codes(df, folds_df):
    """loso_split_codes: int8[23] per window. Split depends only on
    subject membership, so window-level == sequence-level rule."""
    codes = {"train": 0, "validation": 1, "test": 2}
    subjects = sorted(df["subject_uid"].unique())
    sub_pos = {s: i for i, s in enumerate(subjects)}

    # matrix (n_subjects x n_folds) of split codes
    n_folds = len(folds_df)
    mat = np.zeros((len(subjects), n_folds), dtype=np.int8)
    for _, fold in folds_df.iterrows():
        f = int(fold["fold_id"])
        mat[sub_pos[fold["val_subject"]], f] = codes["validation"]
        mat[sub_pos[fold["test_subject"]], f] = codes["test"]

    df = df.copy()
    df["loso_split_codes"] = list(mat[[sub_pos[s] for s in df["subject_uid"]]])
    return df


# ----------------------------------------------------------------------
# Monitoring duration (pure monitoring seconds, per LOSO fold)
# ----------------------------------------------------------------------

def monitoring_duration_subject(case_info, config, censor_min=None):
    """Pure monitoring seconds per subject:
    recorded timeline duration - merged(seizure intervals extended by
    post-ictal censoring)."""
    if censor_min is None:
        censor_min = config["postictal"]["censoring_min"]
    censor_sec = censor_min * 60.0

    durations = {}
    for case, info in case_info.items():
        subj = info["subject_uid"]
        total = sum(e["duration_sec"] for e in info["timeline"])

        # merged censored intervals, clipped into [span_start, span_end]
        entries = sorted(info["timeline"], key=lambda e: e["absolute_start_sec"])
        span_start = entries[0]["absolute_start_sec"]
        span_end = max(e["absolute_end_sec"] for e in entries)
        intervals = sorted(
            (max(sz["onset_abs_sec"], span_start),
             min(sz["offset_abs_sec"] + censor_sec, span_end))
            for sz in info["seizures"]
        )
        excluded = 0.0
        cur_s, cur_e = None, None
        for s, e in intervals:
            if e <= s:
                continue
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                excluded += cur_e - cur_s
                cur_s, cur_e = s, e
        if cur_s is not None:
            excluded += cur_e - cur_s

        durations[subj] = durations.get(subj, 0.0) + max(total - excluded, 0.0)
    return durations


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--data-dir", default="../artifacts/data_freeze_v2_2")
    ap.add_argument("--out-dir", default="../artifacts/export_member4")
    ap.add_argument("--postictal-censor-min", type=float, default=None,
                    help="override post-ictal censoring window (default: config)")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Parsing CHB-MIT summaries -> seizure table")
    seizure_rows, case_info = build_seizure_table(config)

    print(f"[2/5] Writing seizures.pkl ({len(seizure_rows)} seizures)")
    with open(out_dir / "seizures.pkl", "wb") as f:
        pickle.dump(seizure_rows, f, protocol=4)

    print("[3/5] Loading windows metadata")
    df = pd.read_parquet(data_dir / "metadata_all.parquet")

    print("[4/5] Associating seizures + LOSO split codes to windows")
    df = associate_seizures(df, case_info, config)
    folds_df = pd.read_parquet(data_dir / "folds" / "loso_folds.parquet")
    df = add_loso_split_codes(df, folds_df)

    export_cols = [
        "window_id", "patient_id_str" if False else "case_id", "subject_uid",
        "record_id", "continuous_block_id",
        "signal_path", "start_sample", "end_sample",
        "window_start_sec", "window_end_sec",
        "absolute_start_sec", "absolute_end_sec",
        "inference_timestamp_sec",
        "label", "state_label", "prediction_target", "loss_weight",
        "prediction_loss_mask", "state_loss_mask",
        "is_valid", "quality_flags", "boundary_flag",
        "seizure_id", "seizure_id_int", "cluster_id",
        "split", "loso_split_codes",
    ]
    out = df.copy()
    out["label"] = out["state_label"].map(LABEL_NAMES)
    out = out[export_cols].rename(columns={"case_id": "patient_id",
                                           "split": "split_group_holdout"})
    out.to_pickle(out_dir / "windows.pkl")
    print(f"    windows.pkl: {len(out)} rows")

    print("[5/5] Monitoring duration per LOSO fold")
    censor = args.postictal_censor_min
    subj_dur = monitoring_duration_subject(case_info, config, censor_min=censor)

    folds_out = {}
    for _, fold in folds_df.sort_values("fold_id").iterrows():
        fid = f"fold_{int(fold['fold_id']):02d}"
        folds_out[fid] = {
            "train": float(sum(subj_dur[s] for s in fold["train_subjects"])),
            "val": float(subj_dur[fold["val_subject"]]),
            "test": float(subj_dur[fold["test_subject"]]),
            "test_subject": fold["test_subject"],
            "val_subject": fold["val_subject"],
        }
    meta = {
        "unit": "seconds",
        "definition": ("recorded timeline duration minus seizure intervals "
                       "extended by post-ictal censoring"),
        "postictal_censoring_min": (censor if censor is not None
                                    else config["postictal"]["censoring_min"]),
        "n_folds": len(folds_df),
    }
    with open(out_dir / "monitoring_duration.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "folds": folds_out}, f, indent=2)

    # quick sanity summary
    n_assoc = (out["seizure_id_int"] >= 0).sum()
    print(f"\nDone. Outputs in {out_dir.resolve()}")
    print(f"  windows associated with a seizure: {n_assoc:,} / {len(out):,}")
    for fid in ["fold_00", "fold_22"]:
        d = folds_out[fid]
        print(f"  {fid}: train={d['train']/3600:.1f}h val={d['val']/3600:.1f}h "
              f"test={d['test']/3600:.1f}h")


if __name__ == "__main__":
    main()
