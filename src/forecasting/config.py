"""Validated, serializable settings for the long-context experiment."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class ForecastConfig:
    sample_rate: int = 256
    channel_names: tuple = ("FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                            "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8")
    history_minutes: int = 15
    horizon_minutes: int = 60
    endpoint_stride_seconds: int = 60
    window_seconds: int = 5
    postictal_seconds: int = 3600
    filter_warmup_seconds: float = 2.0
    hidden_dim: int = 64
    dilations: tuple = (1, 2, 4, 8)
    dropout: float = 0.1
    encoder_dropout: float = 0.25
    encoder_chunk_size: int = 24
    checkpoint_encoder: bool = True
    branches: str = "both"
    freeze_encoder: bool = False
    encoder_checkpoint: str | None = None
    batch_size: int = 2
    epochs: int = 40
    patience: int = 6
    lr_encoder: float = 0.0001
    lr_head: float = 0.0003
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self):
        for name in ("freeze_encoder", "checkpoint_encoder"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("postictal_seconds", "num_workers", "seed"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("filter_warmup_seconds", "lr_encoder", "lr_head", "weight_decay", "gradient_clip"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("sample_rate", "history_minutes", "horizon_minutes", "endpoint_stride_seconds", "window_seconds",
                     "hidden_dim", "encoder_chunk_size", "batch_size", "epochs", "patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_rate < 100 or self.window_seconds > 30 or 60 % self.window_seconds or self.sample_rate * self.window_seconds < 128:
            raise ValueError("Require fs >= 100 Hz, at least 128 samples/window, and a window length dividing 60 seconds")
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be a nonempty, ordered list of unique names")
        if not self.dilations or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if self.history_minutes > self.receptive_field:
            raise ValueError("TCN receptive field is shorter than the requested history")
        if not 0 <= self.dropout < 1 or not 0 <= self.encoder_dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.branches not in {"both", "morphology", "power"}:
            raise ValueError("branches must be both, morphology, or power")
        if self.postictal_seconds < 0 or not 0 <= self.filter_warmup_seconds < 60 or self.num_workers < 0:
            raise ValueError("Invalid postictal, warmup, or worker setting")
        if min(self.lr_encoder, self.lr_head, self.gradient_clip) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.freeze_encoder and not self.encoder_checkpoint and self.branches != "power":
            raise ValueError("Refusing to freeze a random encoder: supply an encoder checkpoint from this training split")

    @property
    def receptive_field(self):
        return 1 + 4 * sum(self.dilations)  # two k=3 convolutions per residual block

    @property
    def power_dim(self):
        return len(self.channel_names) * 7 * 3

    def to_dict(self):
        return asdict(self)

    def protocol(self):
        names = ("sample_rate", "channel_names", "history_minutes", "horizon_minutes", "endpoint_stride_seconds",
                 "window_seconds", "postictal_seconds", "filter_warmup_seconds")
        return {name: list(getattr(self, name)) if isinstance(getattr(self, name), tuple) else getattr(self, name) for name in names}

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        for key in ("channel_names", "dilations"):
            if key in values:
                if not isinstance(values[key], (list, tuple)):
                    raise ValueError(f"{key} must be a sequence, not a string")
                values[key] = tuple(values[key])
        return cls(**values)


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if values.get("encoder_checkpoint"):
        checkpoint = Path(values["encoder_checkpoint"])
        values["encoder_checkpoint"] = str((path.parent / checkpoint).resolve())
    return ForecastConfig.from_dict(values)
