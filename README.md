# CHB-MIT Seizure Prediction — Preprocessing Pipeline

A three-stage preprocessing pipeline for EEG-based epileptic seizure prediction
using the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/).

## Overview

This pipeline transforms raw EDF recordings into training-ready PyTorch datasets with
windowed signals, multi-class labels, quality flags, and leave-one-subject-out (LOSO)
cross-validation splits.

**Key design choices:**
- **Causal** 0.5–45 Hz bandpass filtering (SOS Butterworth, no future leakage)
- 5-class labeling: UNKNOWN / INTERICTAL / TRANSITION / PREICTAL / ICTAL
- 5-second windows with 50% overlap → 1280 samples at 256 Hz
- TCN sequence length L=24 (62.5 s context), anchor-based target
- 22 common bipolar EEG channels across all subjects
- 23 subject-level LOSO folds with per-fold independent normalization
- 2:1 INTERICTAL:PREICTAL balanced sampling

## Prerequisites

| Requirement          | Details                                 |
|----------------------|-----------------------------------------|
| Python               | ≥ 3.10                                  |
| CHB-MIT Dataset      | Download from [Physionet](https://physionet.org/content/chbmit/1.0.0/) |
| Storage (signals)    | ~72 GB for 655 processed .npy files     |
| Storage (metadata)   | ~500 MB for all pipeline outputs        |

**Python dependencies:**
```
numpy, pandas, scipy, pyarrow, PyYAML, torch
```

Optional: `mne` (Stage 1 EDF reading), `pytest` (tests).

## Quick Start

```bash
# 1. Download CHB-MIT dataset and set E:/chbmit_dataset as root

# 2. Edit Preprocessing/config.yaml
#    - Set dataset.root to your CHB-MIT directory
#    - Set signal_root to desired NPY output directory

# 3. Run the pipeline (three stages, sequential)
python Preprocessing/stage1_edf_to_signals.py --config Preprocessing/config.yaml
python Preprocessing/stage2_label_refinement.py --config Preprocessing/config.yaml \
    --existing-metadata artifacts/data_freeze_v2/metadata_all.parquet
python Preprocessing/stage3_sequence_build.py --config Preprocessing/config.yaml \
    --stage2-dir artifacts/label_refinement

# 4. Validate the output
python Preprocessing/validate.py --data-dir artifacts/sequence_build

# 5. Run tests
pytest Preprocessing/test_utils.py -v
```

## Pipeline Stages

### Stage 1: EDF → Bandpass-Filtered Signals

**Script:** `stage1_edf_to_signals.py`

- Reads all EDF recordings (655 files from PhysioNet, 686 total)
- Extracts 22 common bipolar channels (FP1-F7 through FT10-T8)
- Applies **causal** SOS Butterworth 0.5–45 Hz bandpass filter
- Creates 5-second sliding windows with 50% overlap (step=640 samples)
- Saves per-recording .npy signal files (float32, shape (22, N_samples))
- Produces initial window metadata with labels, quality flags, and split assignments

**Label construction:**
Every window is assigned a label based on its temporal relationship to annotated
seizure onsets/offsets:

| Label       | Time relative to seizure onset          | Loss weight |
|-------------|-----------------------------------------|-------------|
| INTERICTAL  | > 90 min before next, > 60 min after prev | 1.0       |
| PREICTAL    | 0–30 min before onset                   | 1.0         |
| TRANSITION  | 30–40 min before onset                  | 0.2         |
| ICTAL       | onset → offset                          | 0.0         |
| UNKNOWN     | censored regions (postictal, near-seizure) | 0.0      |

**Output (artifacts/data_freeze_v2/):**
```
metadata_all.parquet     window-level metadata
labels_state.npy         [N_windows] int64 state labels
labels_prediction.npy    [N_windows] int64 prediction targets
sequence_index.parquet   L=24 sequence index
quality/                 per-recording quality reports
folds/                   cross-validation splits
normalization/           group-holdout normalization stats
```

### Stage 2: Label Refinement & LOSO Folds

**Script:** `stage2_label_refinement.py`

Refines Stage 1 outputs — reuses existing .npy signals without re-reading EDFs:

- Rebuilds label zones from CHB-MIT summary files
- Fixes boundary detection threshold: 0.5 → 0.8 (min_supervised_overlap)
- Splits `prediction_loss_mask` and `state_loss_mask`
- Corrects quality flag bit counting: `((flags & bit) != 0).sum()`
- Generates 23 LOSO (Leave-One-Subject-Out) fold definitions
- Fits streaming normalization over all valid train windows
- Produces 2:1 INTERICTAL:PREICTAL sampling manifests

**Output (artifacts/label_refinement/):**
```
metadata_all.parquet         fixed metadata
labels_state.npy             fixed state labels
labels_prediction.npy        fixed prediction targets
interictal_candidate_pool.parquet  filtered interictal windows
sequence_index.parquet       strict-continuity sequence index
folds/                       LOSO folds + sampling manifests
normalization/               streaming normalization stats
```

### Stage 3: Sequence Index Rebuild & Per-Fold CV

**Script:** `stage3_sequence_build.py`

Rebuilds only the sequence index and cross-validation artifacts — metadata,
labels, and candidate pools are copied unchanged from Stage 2:

- **Anchor-based target:** `sequence_target = last window's prediction_target`
- **Refined mask:** `sequence_prediction_mask = endpoint mask AND no boundary AND no ICTAL`
- **Per-fold normalization** for each of 23 LOSO folds (independent stats)
- **Per-fold 2:1 sampling** manifests (no group-holdout reuse)

**Output (artifacts/sequence_build/):**
```
sequence_index.parquet       anchor-based sequence index
folds/loso/                  per-fold normalization + sampling + splits
config_resolved.yaml         resolved configuration
audit_report.json            full audit report
SHA256SUMS.txt               file integrity checksums
```

## Configuration

All parameters are defined in `Preprocessing/config.yaml`:

```yaml
dataset:
  root: "E:/chbmit_dataset"          # CHB-MIT root directory

sampling:
  expected_fs_hz: 256                 # Target sampling rate (Hz)

windowing:
  length_sec: 5.0                    # Window duration
  overlap: 0.5                       # 50% overlap

sequence:
  length: 24                          # TCN sequence length
  stride: 1                           # Sliding stride
  min_valid_fraction: 1.0            # All windows must be valid

filtering:
  lowcut_hz: 0.5
  highcut_hz: 45.0
  order: 4                            # Butterworth order
  warmup_sec: 2.0                     # Warmup duration (seconds)
  causal: true                        # Forward-only filtering

preictal:
  duration_min: 30                    # 30 min before onset
  loss_weight: 1.0

transition:
  start_min_before_onset: 40
  end_min_before_onset: 30
  loss_weight: 0.2                    # Weak supervision

interictal:
  min_minutes_before_next_seizure: 90
  min_minutes_after_prev_seizure_end: 60
  loss_weight: 1.0

sampling_balance:
  interictal_to_preictal_ratio: 2.0   # 2:1 class balance
  per_epoch_resample: true
  seed: 42
```

## Output Structure

```
artifacts/
├── data_freeze_v2/           # ← Stage 1 output
├── label_refinement/         # ← Stage 2 output
└── sequence_build/           # ← Stage 3 output (final)
    ├── metadata_all.parquet       # Window-level metadata
    ├── metadata_all.csv           # (same, CSV format)
    ├── labels_state.npy           # [N_windows] 5-class labels
    ├── labels_prediction.npy      # [N_windows] binary targets
    ├── sequence_index.parquet     # [N_seq] sequence definitions
    ├── interictal_candidate_pool.parquet  # Filtered interictal windows
    ├── config_resolved.yaml       # Resolved pipeline config
    ├── sampling_config.yaml       # Sampling parameters
    ├── audit_report.json          # Full audit report
    ├── audit_summary.md           # Human-readable summary
    ├── SHA256SUMS.txt             # File integrity checksums
    ├── quality/                   # Per-recording quality reports
    ├── normalization/             # Group-holdout norm stats
    │   └── group_holdout_stats.npz
    └── folds/
        ├── group_holdout_split.parquet
        ├── group_holdout_train_sampling.parquet
        ├── group_holdout_train_all_sequences.parquet
        ├── loso_folds.parquet     # 23-fold LOSO definitions
        └── loso/
            ├── fold_00/
            │   ├── normalization/loso_stats.npz
            │   ├── sequence_split.parquet
            │   └── train_sampling.parquet
            ├── fold_01/
            ...
            └── fold_22/
```

### Key DataFrames

**metadata_all.parquet** (one row per 5 s window):

| Column                   | Type    | Description                          |
|--------------------------|---------|--------------------------------------|
| window_id                | str     | Unique window identifier             |
| case_id                  | str     | CHB-MIT case (e.g., chb01)           |
| subject_uid              | str     | Subject ID (chb01+chb21=subject_01)  |
| record_id                | str     | EDF filename stem                    |
| signal_path              | str     | Path to .npy signal file             |
| start_sample / end_sample | int    | Sample indices into signal           |
| state_label              | int     | 5-class label (-1, 0, 1, 2, 3)      |
| prediction_target        | int     | Binary target (-1, 0, 1)             |
| prediction_loss_mask     | bool    | Include in prediction loss           |
| state_loss_mask          | bool    | Include in state model loss          |
| boundary_flag            | bool    | max_overlap < 0.8                    |
| quality_flags            | int     | Bit-field quality indicators         |
| is_valid                 | bool    | Overall validity                     |
| split                    | str     | train / validation / test            |

**sequence_index.parquet** (one row per 24-window sequence):

| Column                   | Type    | Description                          |
|--------------------------|---------|--------------------------------------|
| sequence_id              | str     | Unique sequence identifier           |
| window_ids               | list    | 24 window IDs in order               |
| sequence_target          | int     | Anchor (last window) target          |
| sequence_prediction_mask | bool    | Training eligibility                 |
| has_boundary_window      | bool    | Any window has low overlap           |
| has_ictal_window         | bool    | Any window is ICTAL                  |
| state_labels             | list    | 24 state labels                      |
| prediction_loss_masks    | list    | 24 prediction masks                  |

## Training Usage

The dataset module provides two DataLoader creators for different split strategies.

### Group-Holdout (quick prototyping)

```python
from Preprocessing.dataset import create_dataloaders

loaders = create_dataloaders("artifacts/sequence_build", batch_size=32)

# Uses: normalization/group_holdout_stats.npz
#       metadata["split"] column (group-holdout assignment)

train_seq_loader = loaders["sequence"]["train"]
val_seq_loader   = loaders["sequence"]["validation"]
test_seq_loader  = loaders["sequence"]["test"]

for batch in train_seq_loader:
    signals = batch["signal"]        # [B, 24, 22, 1280]
    targets = batch["sequence_target"]  # [B] anchor label
```

### LOSO 23-Fold Training (paper results)

Each fold uses its own per-fold normalization computed **only from train subjects**,
preventing data leakage into the held-out test/validation subjects.

```python
from Preprocessing.dataset import create_loso_dataloaders

for fold_id in range(23):
    loaders = create_loso_dataloaders(
        "artifacts/sequence_build", fold_id=fold_id, batch_size=32
    )

    # Uses: folds/loso/fold_XX/normalization/loso_stats.npz
    #       folds/loso/fold_XX/sequence_split.parquet (loso_split column)

    train_loader = loaders["sequence"]["train"]
    val_loader   = loaders["sequence"]["validation"]
    test_loader  = loaders["sequence"]["test"]

    print(f"Fold {fold_id}: test={loaders['fold_info']['test_subject']}, "
          f"val={loaders['fold_info']['val_subject']}")

    # Train model on this fold...
    # model.fit(train_loader, val_loader)
    # metrics = model.evaluate(test_loader)
```

**Normalization paths (summary):**

| Mode           | Norm stats path                                   | Split column    |
|----------------|---------------------------------------------------|-----------------|
| Group-Holdout  | `normalization/group_holdout_stats.npz`            | `split`         |
| LOSO fold XX   | `folds/loso/fold_XX/normalization/loso_stats.npz`  | `loso_split`    |

The per-fold `loso_stats.npz` is generated by Stage 3 and verified by Check 7e
in `validate.py` (ensures fold-0 stats differ from group_holdout).

## FAQ

**Q: Why causal filtering?**
A: Standard `filtfilt` applies forward+backward filtering, leaking future information into
past samples. Causal (forward-only) filtering ensures no look-ahead bias in seizure
prediction.

**Q: What does the TRANSITION label mean?**
A: TRANSITION (30–40 min before seizure onset) is a weak-supervision buffer between
INTERICTAL and PREICTAL. It is NOT a final prediction target — it is refined by the
HSMM state inference model downstream.

**Q: Why 2:1 INTERICTAL:PREICTAL sampling?**
A: INTERICTAL windows vastly outnumber PREICTAL windows (>100:1). 2:1 downsampling
prevents the model from learning a "always predict safe" trivial solution while
preserving enough interictal context.

**Q: How are subject splits handled?**
A: chb01 and chb21 are the same patient → merged as subject_01. chb24 is subject_24.
All data splits operate at the subject level to prevent within-subject leakage.

**Q: What is the anchor-based target?**
A: Stage 3 uses the last window in each sequence as the "anchor" — the sequence target
equals the anchor's prediction_target. This gives the model a clear temporal
reference point rather than aggregating all 24 windows.

**Q: Can I run on fewer cases for testing?**
A: Yes — add `--max-cases 3` to any stage script to process only the first 3 cases.

## Citation

If you use this preprocessing pipeline in your research, please cite:

```bibtex
@misc{chbmit-preprocessing,
  title = {CHB-MIT Seizure Prediction Preprocessing Pipeline},
  year = {2026},
  note = {Causal bandpass filtering, 5-class labeling, LOSO cross-validation for
          EEG-based epileptic seizure prediction with the CHB-MIT Scalp EEG Database},
}

@article{shoeb2010application,
  title = {Application of Machine Learning to Epileptic Seizure Detection},
  author = {Shoeb, Ali Hossam},
  year = {2010},
  school = {Massachusetts Institute of Technology},
}

@article{goldberger2000physiobank,
  title = {PhysioBank, PhysioToolkit, and PhysioNet},
  author = {Goldberger, Ary L and Amaral, Luis AN and Glass, Leon and Hausdorff, Jeffrey M and Ivanov, Plamen Ch and Mark, Roger G and Mietus, Joseph E and Moody, George B and Peng, Chung-Kang and Stanley, H Eugene},
  journal = {Circulation},
  volume = {101},
  number = {23},
  pages = {e215--e220},
  year = {2000},
}
```
