"""
test_utils.py -- Tests for Preprocessing Pipeline shared utilities.

Tests core functions: summary parsing, timeline building, label zones,
window labeling, quality checks, subject splits.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Preprocessing._utils import (
    parse_time_string,
    parse_summary_enhanced,
    build_patient_timeline,
    get_all_seizures_absolute,
    cluster_seizures,
    build_label_zones_absolute,
    assign_window_label_via_zones,
    check_window_quality,
    compute_prediction_target,
    build_subject_splits,
    apply_causal_bandpass,
    QUALITY_FLAGS,
    LABEL_UNKNOWN, LABEL_INTERICTAL, LABEL_TRANSITION,
    LABEL_PREICTAL, LABEL_ICTAL,
)


class TestTimeParsing:
    def test_normal_time(self):
        sec, day = parse_time_string("11:42:54")
        assert sec == 11 * 3600 + 42 * 60 + 54
        assert day == 0

    def test_midnight(self):
        sec, day = parse_time_string("00:00:00")
        assert sec == 0
        assert day == 0

    def test_24_hours(self):
        sec, day = parse_time_string("24:00:00")
        assert sec == 0
        assert day == 1

    def test_25_hours(self):
        sec, day = parse_time_string("25:30:00")
        assert sec == 1 * 3600 + 30 * 60
        assert day == 1

    def test_invalid(self):
        sec, day = parse_time_string("invalid")
        assert sec is None
        assert day is None


class TestLabelZones:
    def test_single_seizure_zones(self):
        seizures = [{
            "onset_abs_sec": 7200,
            "offset_abs_sec": 7260,
            "case_id": "chb01",
            "record_id": "chb01_03",
            "continuous_block_id": 0,
        }]
        timeline = [{
            "filename": "chb01_03.edf",
            "case_id": "chb01",
            "record_id": "chb01_03",
            "absolute_start_sec": 0,
            "absolute_end_sec": 21600,
            "duration_sec": 21600,
            "continuous_block_id": 0,
        }]
        zones = build_label_zones_absolute(seizures, timeline)

        ictal_zones = [z for z in zones if z["label"] == LABEL_ICTAL]
        assert len(ictal_zones) >= 1
        assert ictal_zones[0]["start_sec"] == 7200
        assert ictal_zones[0]["end_sec"] == 7260

        preictal_zones = [z for z in zones if z["label"] == LABEL_PREICTAL]
        assert len(preictal_zones) >= 1
        assert preictal_zones[0]["end_sec"] == 7200
        assert preictal_zones[0]["start_sec"] == 7200 - 30 * 60

        trans_zones = [z for z in zones if z["label"] == LABEL_TRANSITION]
        assert len(trans_zones) >= 1
        assert trans_zones[0]["loss_weight"] == 0.2

        interictal_zones = [z for z in zones if z["label"] == LABEL_INTERICTAL]
        assert len(interictal_zones) >= 1
        assert interictal_zones[0]["end_sec"] <= 7200 - 90 * 60

    def test_no_seizure_all_interictal(self):
        timeline = [{
            "filename": "chb01_01.edf",
            "case_id": "chb01",
            "record_id": "chb01_01",
            "absolute_start_sec": 0,
            "absolute_end_sec": 3600,
            "duration_sec": 3600,
            "continuous_block_id": 0,
        }]
        zones = build_label_zones_absolute([], timeline)
        assert len(zones) == 1
        assert zones[0]["label"] == LABEL_INTERICTAL

    def test_zones_non_overlapping(self):
        seizures = [{
            "onset_abs_sec": 3600,
            "offset_abs_sec": 3660,
            "case_id": "chb01",
            "record_id": "chb01_03",
            "continuous_block_id": 0,
        }]
        timeline = [{
            "filename": "chb01_03.edf",
            "case_id": "chb01",
            "record_id": "chb01_03",
            "absolute_start_sec": 0,
            "absolute_end_sec": 7200,
            "duration_sec": 3600,
            "continuous_block_id": 0,
        }]
        zones = build_label_zones_absolute(seizures, timeline)
        for i in range(len(zones) - 1):
            assert zones[i]["end_sec"] <= zones[i + 1]["start_sec"]


class TestWindowLabeling:
    def test_window_fully_in_ictal(self):
        zones = [
            {"start_sec": 0, "end_sec": 100, "label": LABEL_INTERICTAL,
             "zone_name": "interictal", "loss_weight": 1.0},
            {"start_sec": 100, "end_sec": 200, "label": LABEL_ICTAL,
             "zone_name": "ictal", "loss_weight": 0.0},
        ]
        result = assign_window_label_via_zones(110, 120, zones)
        assert result["state_label"] == LABEL_ICTAL
        assert result["overlap_ictal"] == 1.0
        assert abs(result["maximum_overlap_fraction"] - 1.0) < 0.01

    def test_window_boundary(self):
        zones = [
            {"start_sec": 0, "end_sec": 10, "label": LABEL_INTERICTAL,
             "zone_name": "interictal", "loss_weight": 1.0},
            {"start_sec": 10, "end_sec": 20, "label": LABEL_PREICTAL,
             "zone_name": "preictal", "loss_weight": 1.0},
        ]
        result = assign_window_label_via_zones(5, 15, zones)
        assert result["state_label"] == LABEL_PREICTAL
        assert result["boundary_flag"] == False

    def test_overlap_sum(self):
        zones = [
            {"start_sec": 0, "end_sec": 5, "label": LABEL_INTERICTAL,
             "zone_name": "i", "loss_weight": 1.0},
            {"start_sec": 5, "end_sec": 10, "label": LABEL_PREICTAL,
             "zone_name": "p", "loss_weight": 1.0},
        ]
        result = assign_window_label_via_zones(0, 10, zones)
        total = (result["overlap_interictal"] + result["overlap_transition"] +
                 result["overlap_preictal"] + result["overlap_ictal"] +
                 result["overlap_unknown"])
        assert abs(total - 1.0) < 0.01


class TestQualityChecks:
    def test_nan_detection(self):
        sig = np.random.randn(22, 1280).astype(np.float32)
        sig[0, 0] = np.nan
        result = check_window_quality(sig, 512, 0, 256)
        assert result["quality_flags"] & QUALITY_FLAGS["nan_inf"]
        assert not result["is_valid"]

    def test_flat_channel(self):
        sig = np.random.randn(22, 1280).astype(np.float32)
        for ch in range(12):
            sig[ch, :] = 1.0
        result = check_window_quality(sig, 512, 0, 256)
        assert result["quality_flags"] & QUALITY_FLAGS["flat_channel"]
        assert not result["is_valid"]

    def test_clean_signal(self):
        sig = np.random.randn(22, 1280).astype(np.float32) * 50
        result = check_window_quality(sig, 512, 600, 256)
        assert result["is_valid"]
        assert result["quality_flags"] == 0

    def test_warmup_flag(self):
        sig = np.random.randn(22, 1280).astype(np.float32) * 50
        result = check_window_quality(sig, 512, 100, 256)
        assert result["quality_flags"] & QUALITY_FLAGS["filter_warmup"]


class TestPredictionTarget:
    def test_interictal_to_0(self):
        assert compute_prediction_target(LABEL_INTERICTAL) == 0

    def test_preictal_to_1(self):
        assert compute_prediction_target(LABEL_PREICTAL) == 1

    def test_transition_to_1(self):
        assert compute_prediction_target(LABEL_TRANSITION) == 1

    def test_ictal_to_neg1(self):
        assert compute_prediction_target(LABEL_ICTAL) == -1

    def test_unknown_to_neg1(self):
        assert compute_prediction_target(LABEL_UNKNOWN) == -1


class TestSubjectSplits:
    def test_chb01_chb21_same_split(self):
        config = {
            "splits": {
                "group_holdout": {
                    "train_cases": ["chb01", "chb21", "chb02"],
                    "val_cases": ["chb17"],
                    "test_cases": ["chb22"],
                }
            }
        }
        subject_mapping = {
            "chb01": "subject_01", "chb21": "subject_01",
            "chb02": "subject_02", "chb17": "subject_17",
            "chb22": "subject_22",
        }
        splits = build_subject_splits(subject_mapping, config)
        assert splits["case_to_split"]["chb01"] == "train"
        assert splits["case_to_split"]["chb21"] == "train"
        assert splits["subject_to_split"]["subject_01"] == "train"

    def test_cross_split_detection(self):
        config = {
            "splits": {
                "group_holdout": {
                    "train_cases": ["chb01"],
                    "val_cases": ["chb21"],
                    "test_cases": [],
                }
            }
        }
        subject_mapping = {"chb01": "subject_01", "chb21": "subject_01"}
        with pytest.raises(ValueError):
            build_subject_splits(subject_mapping, config)


class TestSeizureClustering:
    def test_close_seizures_clustered(self):
        seizures = [
            {"onset_abs_sec": 3600, "offset_abs_sec": 3660,
             "case_id": "chb01", "record_id": "r1", "continuous_block_id": 0},
            {"onset_abs_sec": 3720, "offset_abs_sec": 3780,
             "case_id": "chb01", "record_id": "r1", "continuous_block_id": 0},
        ]
        clusters = cluster_seizures(seizures, gap_threshold_min=120)
        assert len(clusters) == 1
        assert len(clusters[0]["seizures"]) == 2

    def test_far_seizures_separate(self):
        seizures = [
            {"onset_abs_sec": 3600, "offset_abs_sec": 3660,
             "case_id": "chb01", "record_id": "r1", "continuous_block_id": 0},
            {"onset_abs_sec": 3600 + 3 * 3600, "offset_abs_sec": 3600 + 3 * 3600 + 60,
             "case_id": "chb01", "record_id": "r2", "continuous_block_id": 0},
        ]
        clusters = cluster_seizures(seizures, gap_threshold_min=120)
        assert len(clusters) == 2

    def test_different_blocks_separate(self):
        seizures = [
            {"onset_abs_sec": 3600, "offset_abs_sec": 3660,
             "case_id": "chb01", "record_id": "r1", "continuous_block_id": 0},
            {"onset_abs_sec": 3700, "offset_abs_sec": 3760,
             "case_id": "chb01", "record_id": "r2", "continuous_block_id": 1},
        ]
        clusters = cluster_seizures(seizures, gap_threshold_min=120)
        assert len(clusters) == 2


class TestCausalFilter:
    def test_filter_preserves_shape(self):
        sig = np.random.randn(22, 1280).astype(np.float32)
        filtered, warmup = apply_causal_bandpass(sig, 256.0)
        assert filtered.shape == sig.shape
        assert warmup == 512

    def test_filter_removes_high_freq(self):
        t = np.linspace(0, 5, 1280)
        sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
        sig = np.tile(sig, (22, 1)).astype(np.float32)
        filtered, _ = apply_causal_bandpass(sig, 256.0)
        assert np.std(filtered) < np.std(sig)
