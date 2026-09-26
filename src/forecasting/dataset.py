"""Lazy long-history EEG loading; no concatenation across missing observations."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd  # Same numerical runtime initialization as the preprocessing stack.
import torch
from torch.utils.data import Dataset

from .features import minute_power_features


def load_index(path, config):
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or index.get("protocol") != config.protocol():
        raise ValueError("Index/config protocol mismatch; rebuild the index for this history/horizon")
    for row in index["rows"]:
        if row["split"] not in {"train", "validation", "test"} or (row["target"] not in (0, 1) if row["target_observed"] else row["target"] != -1):
            raise ValueError("Invalid split or censored target encoding")
        if abs(row["endpoint_sec"] - row["history_start_sec"] - config.history_minutes * 60) > 1e-6 or abs(row["forecast_end_sec"] - row["endpoint_sec"] - config.horizon_minutes * 60) > 1e-6:
            raise ValueError("Index time geometry differs from protocol")
        cursor = row["history_start_sec"]
        for record_id, start, end in row["segments"]:
            record = index["records"][record_id]
            if not 0 <= start < end <= record["n_samples"] or (record["subject_id"], record["session_id"], record["split"]) != (row["subject_id"], row["session_id"], row["split"]):
                raise ValueError("History segment crosses identity/split or sample bounds")
            if abs(record["start_sec"] + start / config.sample_rate - cursor) > 1e-6:
                raise ValueError("Gap or overlap in history segments")
            cursor = record["start_sec"] + end / config.sample_rate
        if abs(cursor - row["endpoint_sec"]) > 1e-6:
            raise ValueError("History does not stop exactly at the inference endpoint")
    return index


def split_signature(index, split="train"):
    rows = [r for r in index["rows"] if r["split"] == split and r["target_observed"]]
    used = sorted({segment[0] for row in rows for segment in row["segments"]})
    content = {"protocol": index["protocol"], "rows": rows, "records": [index["records"][i] for i in used]}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class ForecastDataset(Dataset):
    def __init__(self, index, config, split, scaler=None, observed_only=True):
        if index.get("protocol") != config.protocol():
            raise ValueError("Index and model protocols differ")
        self.index, self.config, self.scaler = index, config, scaler
        self.row_ids = [i for i, row in enumerate(index["rows"]) if row["split"] == split and (row["target_observed"] or not observed_only)]
        self.signal_cache, self.feature_cache = OrderedDict(), OrderedDict()

    def __len__(self):
        return len(self.row_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["signal_cache"], state["feature_cache"] = OrderedDict(), OrderedDict()
        return state

    def _signal(self, record_index):
        if record_index not in self.signal_cache:
            record = self.index["records"][record_index]
            path = Path(record["signal_path"])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["file_size"], record["mtime_ns"]):
                raise ValueError(f"Signal changed since index preparation: {path}")
            self.signal_cache[record_index] = np.load(path, mmap_mode="r", allow_pickle=False)
            if len(self.signal_cache) > 4:
                self.signal_cache.popitem(last=False)
        self.signal_cache.move_to_end(record_index)
        return self.signal_cache[record_index]

    def windows(self, i):
        row = self.index["rows"][self.row_ids[i]]
        parts = [self._signal(r)[:, start:end] for r, start, end in row["segments"]]
        raw = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        c = self.config
        expected = (len(c.channel_names), c.history_minutes * 60 * c.sample_rate)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError("Incomplete or nonfinite EEG input; fix/flag data, never silently fill gaps")
        return raw.reshape(len(c.channel_names), c.history_minutes, 60 // c.window_seconds, c.window_seconds * c.sample_rate).transpose(1, 2, 0, 3).copy()

    def power_features(self, i, windows=None):
        row = self.index["rows"][self.row_ids[i]]
        result = []
        for minute in range(self.config.history_minutes):
            key = (row["subject_id"], row["session_id"], row["history_start_sec"] + minute * 60)
            if key not in self.feature_cache:
                if windows is None:
                    windows = self.windows(i)
                self.feature_cache[key] = minute_power_features(windows[minute], self.config.sample_rate)
                if len(self.feature_cache) > 2048:
                    self.feature_cache.popitem(last=False)
            self.feature_cache.move_to_end(key)
            result.append(self.feature_cache[key])
        return np.stack(result)

    def __getitem__(self, i):
        row_id = self.row_ids[i]
        row = self.index["rows"][row_id]
        windows = self.windows(i)
        power = self.power_features(i, windows)
        if self.scaler is not None:
            power = self.scaler.transform(power)
        return {"windows": torch.from_numpy(windows), "power": torch.from_numpy(power),
                "target": torch.tensor(float(row["target"])), "observed": torch.tensor(row["target_observed"]), "row_id": row_id}
