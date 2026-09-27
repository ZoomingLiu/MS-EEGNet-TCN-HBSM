"""Validated, serializable output contract for LOSO HBSI inference."""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class HBSIOutput:
    sequence_id: str
    patient_id: str
    record_id: str
    split: str
    timestamps_sec: np.ndarray
    hsmm_state_probabilities: np.ndarray
    hsmm_state_path: np.ndarray
    bocpd_change_probability: np.ndarray
    valid_mask: np.ndarray


def _validated_output(metadata: Mapping, hsmm_probabilities, bocpd_probabilities, length: int, index: int) -> HBSIOutput:
    timestamps = np.asarray(metadata["timestamps_sec"], dtype=np.float64).reshape(-1)
    if timestamps.size != length:
        raise ValueError(f"sequence {index}: timestamps length {timestamps.size} does not match {length}.")
    hsmm = np.asarray(hsmm_probabilities, dtype=np.float64)
    bocpd = np.asarray(bocpd_probabilities, dtype=np.float64).reshape(-1)
    if hsmm.shape != (length, 3):
        raise ValueError(f"sequence {index}: HSMM probabilities must have shape ({length}, 3), got {hsmm.shape}.")
    if bocpd.size != length:
        raise ValueError(f"sequence {index}: BOCPD probabilities length {bocpd.size} does not match {length}.")
    if not np.all(np.isfinite(hsmm)) or not np.all(np.isfinite(bocpd)):
        raise ValueError(f"sequence {index}: probabilities must be finite.")
    hsmm = np.clip(hsmm, 0.0, None)
    row_sums = hsmm.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError(f"sequence {index}: every HSMM row must have positive mass.")
    hsmm /= row_sums
    valid_mask = np.asarray(metadata.get("valid_mask", np.ones(length, dtype=bool)), dtype=bool).reshape(-1)
    if valid_mask.size != length:
        raise ValueError(f"sequence {index}: valid_mask length does not match {length}.")
    return HBSIOutput(
        sequence_id=str(metadata.get("sequence_id", f"seq_{index}")),
        patient_id=str(metadata.get("patient_id", "unknown")),
        record_id=str(metadata.get("record_id", "unknown")),
        split=str(metadata.get("split", "UNKNOWN")),
        timestamps_sec=timestamps,
        hsmm_state_probabilities=hsmm.astype(np.float32),
        hsmm_state_path=np.argmax(hsmm, axis=1).astype(np.int64),
        bocpd_change_probability=np.clip(bocpd, 0.0, 1.0).astype(np.float32),
        valid_mask=valid_mask,
    )


def package_hbsi_results(metadata_list: Sequence[Mapping], all_hsmm_probs: Sequence, all_bocpd_probs: Sequence,
                         original_L_list: Sequence[int] | None = None) -> list[HBSIOutput]:
    """Validate and package sequence results; optionally pad truncated inputs."""
    if not (len(metadata_list) == len(all_hsmm_probs) == len(all_bocpd_probs)):
        raise ValueError("metadata, HSMM, and BOCPD collections must have equal lengths.")
    if original_L_list is not None and len(original_L_list) != len(metadata_list):
        raise ValueError("original_L_list must have the same length as metadata_list.")
    outputs = []
    for index, (metadata, hsmm, bocpd) in enumerate(zip(metadata_list, all_hsmm_probs, all_bocpd_probs)):
        target_length = len(metadata["timestamps_sec"]) if original_L_list is None else int(original_L_list[index])
        hsmm, bocpd = np.asarray(hsmm), np.asarray(bocpd).reshape(-1)
        if hsmm.shape[0] > target_length or bocpd.size > target_length:
            raise ValueError(f"sequence {index}: result is longer than its original sequence.")
        if hsmm.shape[0] != bocpd.size:
            raise ValueError(f"sequence {index}: HSMM and BOCPD result lengths differ.")
        if hsmm.shape[0] < target_length:
            padded_hsmm = np.zeros((target_length, 3), dtype=float)
            padded_hsmm[:, 0] = 1.0
            padded_hsmm[:hsmm.shape[0]] = hsmm
            padded_bocpd = np.zeros(target_length, dtype=float)
            padded_bocpd[:bocpd.size] = bocpd
            hsmm, bocpd = padded_hsmm, padded_bocpd
        outputs.append(_validated_output(metadata, hsmm, bocpd, target_length, index))
    return outputs


def save_to_pickle(hbsi_outputs: Sequence[HBSIOutput], filepath: str | Path) -> None:
    path = Path(filepath)
    with path.open("wb") as file_handle:
        pickle.dump(list(hbsi_outputs), file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Exported %d HBSI outputs to %s", len(hbsi_outputs), path)
