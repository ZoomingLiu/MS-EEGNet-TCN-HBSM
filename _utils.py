#!/usr/bin/env python
"""
_utils.py -- Shared constants and utility functions for the CHB-MIT
preprocessing pipeline (3-stage).

This module is imported by:
  - stage1_edf_to_signals.py   (EDF reading, filtering, windowing)
  - stage2_label_refinement.py (label re-assignment, mask fixes)
  - stage3_sequence_build.py   (sequence index, LOSO folds)
  - validate.py                (data validation checks)
  - dataset.py                 (PyTorch datasets)
  - test_utils.py              (unit tests)
"""

import hashlib
import numpy as np
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.signal import butter, sosfilt

# ======================================================================
# Constants
# ======================================================================

QUALITY_FLAGS = {
    "nan_inf": 1,
    "sample_count_error": 2,
    "filter_warmup": 4,
    "clipping_saturation": 8,
    "flat_channel": 16,
    "high_amplitude_artifact": 32,
    "missing_channel": 64,
    "time_discontinuity": 128,
    "annotation_problem": 256,
}

LABEL_UNKNOWN = -1
LABEL_INTERICTAL = 0
LABEL_TRANSITION = 1
LABEL_PREICTAL = 2
LABEL_ICTAL = 3

LABEL_NAMES = {
    -1: "UNKNOWN", 0: "INTERICTAL", 1: "TRANSITION",
    2: "PREICTAL", 3: "ICTAL"
}

# V2.1 corrected boundary threshold (was 0.5 in V2)
MIN_SUPERVISED_OVERLAP = 0.8


# ======================================================================
# 1. Summary Parser (time parsing, seizure annotations)
# ======================================================================

def parse_time_string(time_str: str) -> tuple:
    """Parse HH:MM:SS, handling 24+, 25+, 26+ hours and cross-midnight."""
    time_str = time_str.strip()
    parts = time_str.split(":")
    if len(parts) != 3:
        return None, None
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None, None
    day_offset = 0
    if h >= 24:
        day_offset = h // 24
        h = h % 24
    total_sec = h * 3600 + m * 60 + s
    return total_sec, day_offset


def parse_summary_enhanced(summary_path: str) -> dict:
    """
    Parse CHB-MIT summary.txt with full time information.

    Returns:
      {
        "sampling_rate": float,
        "channels": list[str],
        "files": [
          {
            "filename": str,
            "start_time_str": str,
            "end_time_str": str,
            "start_sec_of_day": int,
            "end_sec_of_day": int,
            "start_day_offset": int,
            "end_day_offset": int,
            "n_seizures": int,
            "seizures": [{"onset_sec": float, "offset_sec": float}],
          }
        ]
      }
    """
    result = {"sampling_rate": None, "channels": [], "files": [], "errors": []}

    try:
        with open(summary_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        result["errors"].append(f"Cannot read: {e}")
        return result

    # Sampling rate
    m = re.search(r"sampling\s*rate[:\s]*(\d+)", content, re.IGNORECASE)
    if m:
        result["sampling_rate"] = float(m.group(1))

    # Channels
    in_ch = False
    for line in content.split("\n"):
        if re.search(r"Channels?\s+in\s+EDF", line, re.IGNORECASE):
            in_ch = True
            continue
        if in_ch:
            cm = re.match(r"Channel\s+\d+:\s*(.+)", line, re.IGNORECASE)
            if cm:
                result["channels"].append(cm.group(1).strip())
            elif line.strip() and not line.startswith("*"):
                in_ch = False

    # Parse file entries with times
    current_file = None
    onset_stack = []

    for line in content.split("\n"):
        fm = re.match(r"File\s*Name[:\s]+(\S+\.edf)", line, re.IGNORECASE)
        if fm:
            if current_file is not None:
                result["files"].append(current_file)
            current_file = {
                "filename": fm.group(1),
                "start_time_str": None,
                "end_time_str": None,
                "start_sec_of_day": None,
                "end_sec_of_day": None,
                "start_day_offset": 0,
                "end_day_offset": 0,
                "n_seizures": 0,
                "seizures": [],
            }
            onset_stack = []
            continue

        if current_file is None:
            continue

        # File start/end time
        sm = re.search(r"File\s+Start\s+Time[:\s]+(\d{1,2}:\d{2}:\d{2})", line, re.IGNORECASE)
        if sm:
            current_file["start_time_str"] = sm.group(1)
            sec, day_off = parse_time_string(sm.group(1))
            if sec is not None:
                current_file["start_sec_of_day"] = sec
                current_file["start_day_offset"] = day_off
            continue

        em = re.search(r"File\s+End\s+Time[:\s]+(\d{1,2}:\d{2}:\d{2})", line, re.IGNORECASE)
        if em:
            current_file["end_time_str"] = em.group(1)
            sec, day_off = parse_time_string(em.group(1))
            if sec is not None:
                current_file["end_sec_of_day"] = sec
                current_file["end_day_offset"] = day_off
            continue

        # Seizure count
        sc = re.match(r"Number\s+of\s+Seizures\s+in\s+File[:\s]+(\d+)", line, re.IGNORECASE)
        if sc:
            current_file["n_seizures"] = int(sc.group(1))
            continue

        # Seizure start
        start_m = re.search(
            r"Seizure\s+(?:\d+\s+)?Start\s*Time[:\s]*(\d+)\s*seconds?",
            line, re.IGNORECASE)
        if start_m:
            onset_stack.append(float(start_m.group(1)))
            continue

        # Seizure end
        end_m = re.search(
            r"Seizure\s+(?:\d+\s+)?End\s*Time[:\s]*(\d+)\s*seconds?",
            line, re.IGNORECASE)
        if end_m and onset_stack:
            onset = onset_stack.pop(0)
            offset = float(end_m.group(1))
            if onset < offset:
                current_file["seizures"].append({
                    "onset_sec": onset,
                    "offset_sec": offset,
                })

    if current_file is not None:
        result["files"].append(current_file)

    return result


def parse_seizures_file(path: str) -> list:
    """Parse .seizures file as fallback for seizure annotations."""
    seizures = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        onset = float(parts[0])
                        offset = float(parts[1])
                        if onset < offset:
                            seizures.append({"onset_sec": onset, "offset_sec": offset})
                    except ValueError:
                        pass
    except Exception:
        pass
    return seizures


# ======================================================================
# 2. Patient-Level Timeline Builder
# ======================================================================

def build_patient_timeline(parsed_summary: dict, case_id: str) -> list:
    """
    Build patient-level timeline from parsed summary files.
    Connects files by time, identifies continuous blocks.

    Returns list of file entries with:
      - absolute_start_sec: seconds from first file start
      - absolute_end_sec
      - continuous_block_id
      - seizures with absolute times
    """
    files = parsed_summary.get("files", [])
    if not files:
        return []

    def sort_key(f):
        return (f.get("start_day_offset", 0), f.get("start_sec_of_day", 0) or 0)

    files_sorted = sorted(files, key=sort_key)

    # Build absolute timeline starting from first file
    if files_sorted[0].get("start_sec_of_day") is None:
        # No time info, use sequential ordering
        timeline = []
        abs_offset = 0.0
        for i, f in enumerate(files_sorted):
            duration = 3600.0  # assume 1 hour if unknown
            seizures_abs = []
            for sz in f.get("seizures", []):
                seizures_abs.append({
                    "onset_abs_sec": abs_offset + sz["onset_sec"],
                    "offset_abs_sec": abs_offset + sz["offset_sec"],
                    "onset_in_file_sec": sz["onset_sec"],
                    "offset_in_file_sec": sz["offset_sec"],
                })
            timeline.append({
                "filename": f["filename"],
                "case_id": case_id,
                "record_id": Path(f["filename"]).stem,
                "absolute_start_sec": abs_offset,
                "absolute_end_sec": abs_offset + duration,
                "duration_sec": duration,
                "continuous_block_id": 0,
                "seizures_abs": seizures_abs,
                "has_seizures": len(seizures_abs) > 0,
            })
            abs_offset += duration
        return timeline

    # With time info: compute absolute offsets
    first_sec = files_sorted[0]["start_sec_of_day"]
    first_day = files_sorted[0].get("start_day_offset", 0)
    base_sec = first_sec + first_day * 86400

    timeline = []
    prev_end_abs = None
    block_id = 0
    gap_threshold = 30  # seconds

    for f in files_sorted:
        start_sec = f.get("start_sec_of_day", 0)
        day_off = f.get("start_day_offset", 0)
        abs_start = start_sec + day_off * 86400 - base_sec

        end_sec = f.get("end_sec_of_day", 0)
        end_day = f.get("end_day_offset", 0)

        # Handle cross-midnight
        if end_sec <= start_sec and end_day == day_off:
            end_day = day_off + 1

        abs_end = end_sec + end_day * 86400 - base_sec
        duration = abs_end - abs_start

        # Check gap from previous file
        if prev_end_abs is not None:
            gap = abs_start - prev_end_abs
            if gap > gap_threshold:
                block_id += 1

        # Convert seizures to absolute time
        seizures_abs = []
        for sz in f.get("seizures", []):
            onset_abs = abs_start + sz["onset_sec"]
            offset_abs = abs_start + sz["offset_sec"]
            seizures_abs.append({
                "onset_abs_sec": onset_abs,
                "offset_abs_sec": offset_abs,
                "onset_in_file_sec": sz["onset_sec"],
                "offset_in_file_sec": sz["offset_sec"],
            })

        timeline.append({
            "filename": f["filename"],
            "case_id": case_id,
            "record_id": Path(f["filename"]).stem,
            "absolute_start_sec": abs_start,
            "absolute_end_sec": abs_end,
            "duration_sec": duration,
            "continuous_block_id": block_id,
            "seizures_abs": seizures_abs,
            "has_seizures": len(seizures_abs) > 0,
        })
        prev_end_abs = abs_end

    return timeline


def get_all_seizures_absolute(timeline: list) -> list:
    """Get all seizures across files in absolute time."""
    all_sz = []
    for entry in timeline:
        for sz in entry["seizures_abs"]:
            all_sz.append({
                "onset_abs_sec": sz["onset_abs_sec"],
                "offset_abs_sec": sz["offset_abs_sec"],
                "case_id": entry["case_id"],
                "record_id": entry["record_id"],
                "continuous_block_id": entry["continuous_block_id"],
            })
    return sorted(all_sz, key=lambda s: s["onset_abs_sec"])


def cluster_seizures(all_seizures: list, gap_threshold_min: float = 120) -> list:
    """
    Cluster seizures across files. Seizures within gap_threshold_min
    of each other belong to the same cluster.
    """
    if not all_seizures:
        return []

    sorted_sz = sorted(all_seizures, key=lambda s: s["onset_abs_sec"])
    clusters = []
    current = {
        "cluster_id": "cl_000",
        "seizures": [sorted_sz[0]],
        "cluster_start_sec": sorted_sz[0]["onset_abs_sec"],
        "cluster_end_sec": sorted_sz[0]["offset_abs_sec"],
        "continuous_block_id": sorted_sz[0]["continuous_block_id"],
    }

    for i in range(1, len(sorted_sz)):
        sz = sorted_sz[i]
        prev = current["seizures"][-1]
        gap = sz["onset_abs_sec"] - prev["offset_abs_sec"]

        same_block = sz["continuous_block_id"] == current["continuous_block_id"]
        if same_block and gap < gap_threshold_min * 60:
            current["seizures"].append(sz)
            current["cluster_end_sec"] = max(current["cluster_end_sec"], sz["offset_abs_sec"])
        else:
            current["cluster_id"] = f"cl_{len(clusters):03d}"
            clusters.append(current)
            current = {
                "cluster_id": f"cl_{len(clusters):03d}",
                "seizures": [sz],
                "cluster_start_sec": sz["onset_abs_sec"],
                "cluster_end_sec": sz["offset_abs_sec"],
                "continuous_block_id": sz["continuous_block_id"],
            }

    current["cluster_id"] = f"cl_{len(clusters):03d}"
    clusters.append(current)
    return clusters


# ======================================================================
# 3. Label Zone Builder (time-slice based, non-overlapping)
# ======================================================================

def build_label_zones_absolute(
    all_seizures_abs: list,
    timeline: list,
    preictal_min: float = 30,
    trans_start_min: float = 40,
    trans_end_min: float = 30,
    interictal_before_min: float = 90,
    interictal_after_min: float = 60,
    postictal_min: float = 60,
) -> list:
    """
    Build non-overlapping label time slices for the entire patient timeline.

    Uses absolute time within continuous blocks. Does NOT cross block boundaries.

    Returns list of {start_sec, end_sec, label, zone_name, loss_weight}
    sorted by start_sec, non-overlapping, covering the entire block.
    """
    zones = []

    # Group seizures and timeline by continuous block
    blocks = defaultdict(list)
    for entry in timeline:
        blocks[entry["continuous_block_id"]].append(entry)

    block_seizures = defaultdict(list)
    for sz in all_seizures_abs:
        block_seizures[sz["continuous_block_id"]].append(sz)

    for block_id, block_files in blocks.items():
        block_start = min(e["absolute_start_sec"] for e in block_files)
        block_end = max(e["absolute_end_sec"] for e in block_files)
        block_sz = sorted(block_seizures.get(block_id, []),
                          key=lambda s: s["onset_abs_sec"])

        if not block_sz:
            # No seizures in this block: all INTERICTAL
            zones.append({
                "start_sec": block_start,
                "end_sec": block_end,
                "label": LABEL_INTERICTAL,
                "zone_name": "trusted_interictal",
                "loss_weight": 1.0,
            })
            continue

        # For each seizure, define time zones:
        #   ictal: [onset, offset)
        #   preictal: [onset - 30min, onset)
        #   transition: [onset - 40min, onset - 30min)
        #   pre_seizure_unknown: [onset - 90min, onset - 40min)
        #   postictal: [offset, offset + 60min)
        # Priority: ICTAL > PREICTAL > TRANSITION > UNKNOWN > INTERICTAL

        events = []  # (time, type, label, zone_name, loss_weight)

        for sz in block_sz:
            onset = sz["onset_abs_sec"]
            offset = sz["offset_abs_sec"]

            events.append((onset, offset, LABEL_ICTAL, "ictal", 0.0))
            events.append((onset - preictal_min * 60, onset, LABEL_PREICTAL, "preictal", 1.0))
            events.append((onset - trans_start_min * 60, onset - trans_end_min * 60,
                          LABEL_TRANSITION, "transition", 0.2))
            events.append((onset - interictal_before_min * 60, onset - trans_start_min * 60,
                          LABEL_UNKNOWN, "pre_seizure_unknown", 0.0))
            events.append((offset, offset + postictal_min * 60,
                          LABEL_UNKNOWN, "postictal", 0.0))

        # Build non-overlapping slices with priority
        priority = {LABEL_ICTAL: 5, LABEL_PREICTAL: 4, LABEL_TRANSITION: 3,
                    LABEL_UNKNOWN: 2, LABEL_INTERICTAL: 1}

        time_points = set([block_start, block_end])
        for (s, e, _, _, _) in events:
            time_points.add(max(block_start, s))
            time_points.add(min(block_end, e))
        time_points = sorted(time_points)

        for i in range(len(time_points) - 1):
            slice_start = time_points[i]
            slice_end = time_points[i + 1]
            if slice_start >= slice_end:
                continue

            mid = (slice_start + slice_end) / 2
            best_label = LABEL_INTERICTAL
            best_zone = "trusted_interictal"
            best_weight = 1.0
            best_pri = priority[LABEL_INTERICTAL]

            for (s, e, lbl, zone, w) in events:
                if s <= mid < e:
                    pri = priority.get(lbl, 0)
                    if pri > best_pri:
                        best_pri = pri
                        best_label = lbl
                        best_zone = zone
                        best_weight = w

            # Verify trusted interictal zone
            if best_label == LABEL_INTERICTAL:
                next_sz = None
                prev_sz_end = None
                for sz in block_sz:
                    if sz["onset_abs_sec"] > mid:
                        if next_sz is None or sz["onset_abs_sec"] < next_sz:
                            next_sz = sz["onset_abs_sec"]
                    if sz["offset_abs_sec"] <= mid:
                        if prev_sz_end is None or sz["offset_abs_sec"] > prev_sz_end:
                            prev_sz_end = sz["offset_abs_sec"]

                dist_to_next = (next_sz - mid) if next_sz else float('inf')
                dist_from_prev = (mid - prev_sz_end) if prev_sz_end else float('inf')

                if dist_to_next < interictal_before_min * 60 or dist_from_prev < interictal_after_min * 60:
                    best_label = LABEL_UNKNOWN
                    best_zone = "near_seizure_unknown"
                    best_weight = 0.0

            zones.append({
                "start_sec": slice_start,
                "end_sec": slice_end,
                "label": best_label,
                "zone_name": best_zone,
                "loss_weight": best_weight,
            })

    zones.sort(key=lambda z: z["start_sec"])
    return zones


# ======================================================================
# 4. Window Label Assignment (V2.1 corrected: threshold = 0.8)
# ======================================================================

def assign_window_label_via_zones(window_start_sec: float, window_end_sec: float,
                                  zones: list) -> dict:
    """
    Assign label to window based on overlap with non-overlapping zones.

    Uses MIN_SUPERVISED_OVERLAP = 0.8 for boundary_flag (V2.1 corrected from 0.5).
    Returns label, overlaps, boundary_flag.
    """
    w_dur = window_end_sec - window_start_sec
    overlaps = {LABEL_ICTAL: 0.0, LABEL_PREICTAL: 0.0, LABEL_TRANSITION: 0.0,
                LABEL_INTERICTAL: 0.0, LABEL_UNKNOWN: 0.0}
    loss_weight = 0.0

    for z in zones:
        z_start, z_end = z["start_sec"], z["end_sec"]
        if window_end_sec <= z_start or window_start_sec >= z_end:
            continue
        ol = min(window_end_sec, z_end) - max(window_start_sec, z_start)
        if ol <= 0:
            continue
        frac = ol / w_dur
        overlaps[z["label"]] += frac
        if frac > 0:
            loss_weight = max(loss_weight, z.get("loss_weight", 0.0))

    # ICTAL override: any overlap with ictal -> ICTAL
    if overlaps[LABEL_ICTAL] > 0:
        best_label = LABEL_ICTAL
    else:
        # Max overlap, tie: PREICTAL > TRANSITION > UNKNOWN > INTERICTAL
        priority_order = [LABEL_PREICTAL, LABEL_TRANSITION, LABEL_UNKNOWN, LABEL_INTERICTAL]
        best_label = LABEL_UNKNOWN
        best_overlap = 0.0
        for lbl in priority_order:
            if overlaps[lbl] > best_overlap:
                best_overlap = overlaps[lbl]
                best_label = lbl

    # Normalize overlaps to sum=1.0 (handle cross-block edge cases)
    total = sum(overlaps.values())
    if total > 0:
        for k in overlaps:
            overlaps[k] = overlaps[k] / total
    else:
        overlaps[LABEL_UNKNOWN] = 1.0
        best_label = LABEL_UNKNOWN

    max_overlap = max(overlaps.values())

    # V2.1 corrected: boundary_flag uses 0.8 threshold (not 0.5)
    boundary_flag = max_overlap < MIN_SUPERVISED_OVERLAP

    return {
        "state_label": best_label,
        "overlap_interictal": overlaps[LABEL_INTERICTAL],
        "overlap_transition": overlaps[LABEL_TRANSITION],
        "overlap_preictal": overlaps[LABEL_PREICTAL],
        "overlap_ictal": overlaps[LABEL_ICTAL],
        "overlap_unknown": overlaps[LABEL_UNKNOWN],
        "maximum_overlap_fraction": max_overlap,
        "boundary_flag": boundary_flag,
        "loss_weight": loss_weight,
    }


def compute_prediction_target(state_label: int) -> int:
    """
    Map 5-class state label to binary prediction target.

    INTERICTAL -> 0, PREICTAL/TRANSITION -> 1, ICTAL/UNKNOWN -> -1 (censored).
    """
    if state_label == LABEL_INTERICTAL:
        return 0
    elif state_label == LABEL_PREICTAL:
        return 1
    elif state_label == LABEL_TRANSITION:
        return 1  # weak positive for TCN head
    else:
        return -1  # ICTAL, UNKNOWN


# ======================================================================
# 5. Windowing
# ======================================================================

def create_windows_metadata(fs: float, n_samples: int,
                            window_sec: float = 5.0,
                            overlap: float = 0.5) -> list:
    """Generate window metadata (without signal data)."""
    win_samp = int(window_sec * fs)
    step_samp = int(win_samp * (1 - overlap))
    if win_samp <= 0 or step_samp <= 0:
        return []

    windows = []
    for start in range(0, n_samples - win_samp + 1, step_samp):
        end = start + win_samp
        windows.append({
            "start_sample": start,
            "end_sample": end,
            "window_start_sec": start / fs,
            "window_end_sec": end / fs,
            "inference_timestamp_sec": end / fs,
        })
    return windows


# ======================================================================
# 6. EDF Reading (MNE)
# ======================================================================

def normalize_channel_name(name: str) -> str:
    """Normalize channel name: strip, uppercase, remove extra spaces."""
    name = name.strip()
    name = re.sub(r'\s*-\s*', '-', name)
    return name.upper()


def read_edf_with_mne(edf_path: str, target_channels: list,
                      aliases: dict = None) -> tuple:
    """
    Read EDF using MNE. Returns (signal_22ch, info_dict).

    signal shape: (22, n_samples) float32
    info: {fs, n_samples, duration_sec, channels, signal_unit, errors, original_channels}
    """
    info = {
        "fs": 0.0, "n_samples": 0, "duration_sec": 0.0,
        "channels": [], "signal_unit": "unknown",
        "errors": [], "original_channels": [],
        "channel_mapping": {},
    }

    try:
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    except Exception as e:
        info["errors"].append(f"MNE read error: {e}")
        return None, info

    fs = raw.info["sfreq"]
    n_samp = raw.n_times
    duration = n_samp / fs if fs > 0 else 0

    info["fs"] = float(fs)
    info["n_samples"] = int(n_samp)
    info["duration_sec"] = float(duration)
    info["original_channels"] = list(raw.ch_names)

    # Get signal data
    data = raw.get_data()  # (n_channels, n_samples)

    # Check physical unit and convert to microvolts if needed
    if data.size > 0:
        max_val = np.max(np.abs(data))
        if max_val < 0.1:  # Likely in Volts
            data = data * 1e6  # Convert V -> uV
            info["signal_unit"] = "uV"
        else:
            info["signal_unit"] = "uV"  # Already in uV range

    # Build channel mapping (handles MNE duplicate renaming: T8-P8-0, T8-P8-1)
    orig_names_normalized = {}
    for i, ch_name in enumerate(raw.ch_names):
        norm = normalize_channel_name(ch_name)
        base_norm = re.sub(r'-\d+$', '', norm)
        if base_norm not in orig_names_normalized:
            orig_names_normalized[base_norm] = i

    # Build alias lookup
    alias_lookup = {}
    if aliases:
        for target, alts in aliases.items():
            alias_lookup[normalize_channel_name(target)] = normalize_channel_name(target)
            for alt in alts:
                alias_lookup[normalize_channel_name(alt)] = normalize_channel_name(target)

    # Map target channels to original indices
    signal_22 = np.zeros((22, n_samp), dtype=np.float32)
    missing = []
    found = {}

    for i, target in enumerate(target_channels):
        target_norm = normalize_channel_name(target)
        if target_norm in orig_names_normalized:
            idx = orig_names_normalized[target_norm]
            signal_22[i] = data[idx]
            found[target] = idx
            info["channel_mapping"][target] = raw.ch_names[idx]
        elif target_norm in alias_lookup:
            alias_target = alias_lookup[target_norm]
            if alias_target in orig_names_normalized:
                idx = orig_names_normalized[alias_target]
                signal_22[i] = data[idx]
                found[target] = idx
                info["channel_mapping"][target] = raw.ch_names[idx]
            else:
                missing.append(target)
        else:
            missing.append(target)

    info["channels"] = target_channels
    info["missing_channels"] = missing
    info["found_channels"] = found

    if missing:
        info["errors"].append(f"Missing channels: {missing}")
        return None, info

    return signal_22, info


# ======================================================================
# 7. Causal SOS Filter
# ======================================================================

def apply_causal_bandpass(signal: np.ndarray, fs: float,
                          low: float = 0.5, high: float = 45.0,
                          order: int = 4) -> tuple:
    """
    Apply causal SOS Butterworth bandpass filter.
    Returns (filtered_signal, warmup_n_samples).
    """
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")

    n_ch, n_samp = signal.shape
    filtered = np.zeros_like(signal, dtype=np.float32)

    for ch in range(n_ch):
        filtered[ch] = sosfilt(sos, signal[ch].astype(np.float64)).astype(np.float32)

    warmup_samples = int(2.0 * fs)  # 2 second warmup
    return filtered, warmup_samples


# ======================================================================
# 8. Quality Checks (bit flags)
# ======================================================================

def check_window_quality(signal_slice: np.ndarray, warmup_n: int,
                         window_start_sample: int, fs: float,
                         thresholds: dict = None) -> dict:
    """
    Real quality checks with bit flags.
    signal_slice: (22, 1280)

    Returns {quality_flags, quality_score, is_valid, supervised_loss_mask}.
    """
    if thresholds is None:
        thresholds = {
            "flat_diff": 1e-6,
            "flat_frac": 0.5,
            "sat_zscore": 10.0,
            "sat_frac": 0.5,
            "high_amp_zscore": 10.0,
            "clipping_uv": 500.0,
        }

    flags = 0
    n_ch, n_samp = signal_slice.shape

    # 1. NaN/Inf
    if np.any(np.isnan(signal_slice)) or np.any(np.isinf(signal_slice)):
        flags |= QUALITY_FLAGS["nan_inf"]
        return {"quality_flags": flags, "quality_score": 0.0, "is_valid": False,
                "supervised_loss_mask": False}

    # 2. Sample count error
    expected = int(5.0 * fs)
    if n_samp != expected:
        flags |= QUALITY_FLAGS["sample_count_error"]
        return {"quality_flags": flags, "quality_score": 0.0, "is_valid": False,
                "supervised_loss_mask": False}

    # 4. Filter warmup
    if window_start_sample < warmup_n:
        flags |= QUALITY_FLAGS["filter_warmup"]

    # 16. Flat channel (more than flat_frac channels have near-zero signal change)
    flat_count = 0
    for ch in range(n_ch):
        if np.all(np.abs(np.diff(signal_slice[ch])) < thresholds["flat_diff"]):
            flat_count += 1
    if flat_count / n_ch > thresholds["flat_frac"]:
        flags |= QUALITY_FLAGS["flat_channel"]
        return {"quality_flags": flags, "quality_score": 0.0, "is_valid": False,
                "supervised_loss_mask": False}

    # 8. Clipping/saturation
    ch_max = np.max(np.abs(signal_slice), axis=1)
    ch_std = np.std(signal_slice, axis=1)
    robust_std = np.median(ch_std) * 1.4826
    if robust_std > 0:
        sat_count = np.sum(ch_max > thresholds["sat_zscore"] * robust_std)
        if sat_count / n_ch > thresholds["sat_frac"]:
            flags |= QUALITY_FLAGS["clipping_saturation"]

    # 32. High amplitude artifact
    if robust_std > 0 and np.max(ch_max) > thresholds["high_amp_zscore"] * robust_std:
        flags |= QUALITY_FLAGS["high_amplitude_artifact"]

    # Compute quality score
    severe_flags = (QUALITY_FLAGS["nan_inf"] | QUALITY_FLAGS["sample_count_error"] |
                    QUALITY_FLAGS["flat_channel"] | QUALITY_FLAGS["missing_channel"])
    is_valid = (flags & severe_flags) == 0

    # Quality score decreases with flags
    score = 1.0
    if flags & QUALITY_FLAGS["filter_warmup"]:
        score = min(score, 0.7)
    if flags & QUALITY_FLAGS["clipping_saturation"]:
        score = min(score, 0.5)
    if flags & QUALITY_FLAGS["high_amplitude_artifact"]:
        score = min(score, 0.6)

    supervised = is_valid and not (flags & QUALITY_FLAGS["filter_warmup"])

    return {
        "quality_flags": flags,
        "quality_score": score,
        "is_valid": is_valid,
        "supervised_loss_mask": supervised,
    }


# ======================================================================
# 9. Subject-Level Splits
# ======================================================================

def build_subject_splits(subject_mapping: dict, config: dict) -> dict:
    """Build group_holdout split assignments by subject_uid."""
    splits_cfg = config["splits"]["group_holdout"]

    case_to_split = {}
    for case in splits_cfg["train_cases"]:
        case_to_split[case] = "train"
    for case in splits_cfg["val_cases"]:
        case_to_split[case] = "validation"
    for case in splits_cfg["test_cases"]:
        case_to_split[case] = "test"

    # Build subject -> split mapping
    subject_to_split = {}
    for case, subj in subject_mapping.items():
        if case in case_to_split:
            split = case_to_split[case]
            if subj in subject_to_split and subject_to_split[subj] != split:
                raise ValueError(
                    f"Subject {subj} has cases in different splits: "
                    f"{subject_to_split[subj]} vs {split} (case={case})")
            subject_to_split[subj] = split

    return {"case_to_split": case_to_split, "subject_to_split": subject_to_split}


# ======================================================================
# 10. Normalization (two variants)
# ======================================================================

def fit_normalization(signal_paths: list, start_samples: list,
                      end_samples: list, max_windows: int = 5000) -> dict:
    """
    Fit per-channel z-score from a sample of train windows.
    Used by Stage 1 (EDF-to-signals) for initial group-holdout normalization.
    """
    means = []
    for sp, ss, es in zip(signal_paths, start_samples, end_samples):
        try:
            sig = np.load(sp, mmap_mode="r")
            chunk = sig[:, ss:es].astype(np.float32)
            means.append(chunk.mean(axis=1, keepdims=True))
        except Exception:
            continue
        if len(means) >= max_windows:
            break

    if not means:
        return {"mean": np.zeros(22, dtype=np.float32),
                "std": np.ones(22, dtype=np.float32)}

    # Compute global mean
    all_means = np.concatenate(means, axis=1)
    global_mean = all_means.mean(axis=1)

    # Compute std
    stds = []
    for sp, ss, es in zip(signal_paths, start_samples, end_samples):
        try:
            sig = np.load(sp, mmap_mode="r")
            chunk = sig[:, ss:es].astype(np.float32)
            stds.append(((chunk - global_mean.reshape(-1, 1)) ** 2).mean(axis=1, keepdims=True))
        except Exception:
            continue
        if len(stds) >= max_windows:
            break

    if stds:
        all_vars = np.concatenate(stds, axis=1).mean(axis=1)
        global_std = np.sqrt(all_vars + 1e-8)
    else:
        global_std = np.ones(22, dtype=np.float32)

    return {"mean": global_mean.astype(np.float32),
            "std": global_std.astype(np.float32)}


def fit_normalization_streaming(train_valid_df: "pd.DataFrame",
                                n_channels: int = 22) -> dict:
    """
    Streaming normalization over ALL valid train windows.
    Uses cumulative sum per file for O(n_samples + n_windows) complexity.
    Used by Stage 2 (label refinement) and Stage 3 (per-LOSO folds).
    """
    channel_sum = np.zeros(n_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(n_channels, dtype=np.float64)
    total_samples = 0

    for sig_path, group in train_valid_df.groupby("signal_path"):
        try:
            sig = np.load(sig_path, mmap_mode="r")
        except Exception:
            continue

        full_sig = np.asarray(sig, dtype=np.float64)
        n_total_samples = full_sig.shape[1]

        # Cumulative sum and cumulative sum of squares (with zero prefix)
        cumsum = np.zeros((n_channels, n_total_samples + 1), dtype=np.float64)
        cumsum[:, 1:] = np.cumsum(full_sig, axis=1)
        cumsum_sq = np.zeros((n_channels, n_total_samples + 1), dtype=np.float64)
        cumsum_sq[:, 1:] = np.cumsum(full_sig ** 2, axis=1)

        starts = group["start_sample"].to_numpy().astype(int)
        ends = group["end_sample"].to_numpy().astype(int)

        window_sums = cumsum[:, ends] - cumsum[:, starts]
        window_sq_sums = cumsum_sq[:, ends] - cumsum_sq[:, starts]

        channel_sum += window_sums.sum(axis=1)
        channel_sq_sum += window_sq_sums.sum(axis=1)
        total_samples += (ends - starts).sum()

        del full_sig, cumsum, cumsum_sq

    if total_samples > 0:
        global_mean = channel_sum / total_samples
        global_var = channel_sq_sum / total_samples - global_mean ** 2
        global_std = np.sqrt(np.maximum(global_var, 1e-8))
    else:
        global_mean = np.zeros(n_channels, dtype=np.float32)
        global_std = np.ones(n_channels, dtype=np.float32)

    return {
        "mean": global_mean.astype(np.float32),
        "std": global_std.astype(np.float32),
        "n_windows": total_samples,
    }


def compute_per_file_stats(valid_windows: "pd.DataFrame",
                           n_channels: int = 22) -> tuple:
    """
    Precompute per-file (sum, sum_sq, count, subject_uid) for efficient
    per-LOSO-fold normalization.
    Returns (file_stats_dict, file_to_subject_dict).
    Used by Stage 3 (sequence build) for LOSO normalization.
    """
    file_stats = {}
    file_to_subject = {}

    for sig_path, group in valid_windows.groupby("signal_path"):
        subj = group["subject_uid"].iloc[0]
        file_to_subject[sig_path] = subj

        try:
            sig = np.load(sig_path, mmap_mode="r")
        except Exception:
            continue

        ch_sum = np.zeros(n_channels, dtype=np.float64)
        ch_sq_sum = np.zeros(n_channels, dtype=np.float64)
        total_samples = 0

        for _, row in group.iterrows():
            s, e = int(row["start_sample"]), int(row["end_sample"])
            chunk = sig[:, s:e].astype(np.float64)
            ch_sum += chunk.sum(axis=1)
            ch_sq_sum += (chunk ** 2).sum(axis=1)
            total_samples += chunk.shape[1]

        file_stats[sig_path] = (ch_sum, ch_sq_sum, total_samples, subj)

    return file_stats, file_to_subject


# ======================================================================
# 11. SHA256 Utilities
# ======================================================================

def sha256_file(filepath: str) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_sums(output_dir: str) -> list:
    """Compute SHA256 for all files in output directory, return (rel_path, digest) pairs."""
    from pathlib import Path
    output_path = Path(output_dir)
    results = []
    for fpath in sorted(output_path.rglob("*")):
        if fpath.is_file() and fpath.name != "SHA256SUMS.txt":
            rel = fpath.relative_to(output_path)
            rel_str = str(rel).replace("\\", "/")
            digest = sha256_file(str(fpath))
            results.append((rel_str, digest))
    return results
