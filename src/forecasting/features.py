"""Absolute-scale features from already filtered, unstandardized EEG in microvolts."""

import numpy as np


BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def minute_power_features(windows, sample_rate):
    """[K,C,S] -> [C*7*3], ordered as mean, std, slope of logged window features.

    Uses an explicit one-sided Hann periodogram, not a learned representation.
    Slopes are measured per minute; no future samples or record-wide normalization.
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two finite [K,C,S] windows per minute")
    n = x.shape[-1]
    rms = np.sqrt(np.mean(x * x, axis=-1))
    line_length = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
    hann = np.hanning(n)
    transformed = np.fft.rfft((x - x.mean(axis=-1, keepdims=True)) * hann, axis=-1)
    psd = np.abs(transformed) ** 2 / (sample_rate * np.sum(hann ** 2))
    psd[..., 1:-1 if n % 2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    powers = []
    for low, high in BANDS:
        select = (frequencies >= low) & (frequencies < high)
        if not np.any(select):
            raise ValueError(f"No FFT bins in band {(low, high)}")
        powers.append(psd[..., select].sum(axis=-1) * sample_rate / n)
    values = np.log(np.maximum(np.stack([rms, line_length, *powers], axis=-1), 1e-12))
    time = (np.arange(x.shape[0]) + 0.5) / x.shape[0]
    time -= time.mean()
    slope = np.sum(values * time[:, None, None], axis=0) / np.sum(time ** 2)
    return np.concatenate([values.mean(axis=0).ravel(), values.std(axis=0).ravel(), slope.ravel()]).astype(np.float32)


class FeatureScaler:
    """Streaming mean/std; the training runner only fits this on observed training endpoints."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.ndim != 1 or self.mean.shape != self.std.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid scaler")

    @classmethod
    def fit(cls, features):
        count, total, squares = 0, None, None
        for array in features:
            array = np.asarray(array, dtype=np.float64)
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError("Scaler expects finite [minutes,features] arrays")
            if total is None:
                total = np.zeros(array.shape[1])
                squares = np.zeros_like(total)
            total += array.sum(axis=0)
            squares += (array ** 2).sum(axis=0)
            count += len(array)
        if not count:
            raise ValueError("No training features to fit scaler")
        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-8))
        return cls(mean, std)

    def transform(self, features):
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
