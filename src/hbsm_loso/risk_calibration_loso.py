"""Label-free, strictly causal normalization of fold-dependent TCN risk."""

import numpy as np


class CausalRiskCalibrator:
    """Convert raw risk into deviation-from-baseline evidence.

    The baseline is updated from past observations only and is frozen during
    positive excursions. This removes fold-specific probability offsets without
    inspecting held-out labels or future samples.
    """

    def __init__(self, step_seconds=2.5, baseline_half_life_seconds=300.0,
                 alert_z=3.0, temperature=0.5, initial_variance=1.0,
                 freeze_z=2.0):
        if step_seconds <= 0 or baseline_half_life_seconds <= 0:
            raise ValueError("time parameters must be positive.")
        if temperature <= 0 or initial_variance <= 0:
            raise ValueError("temperature and initial_variance must be positive.")
        self.alpha = 1.0 - np.exp(
            -np.log(2.0) * step_seconds / baseline_half_life_seconds
        )
        self.alert_z = float(alert_z)
        self.temperature = float(temperature)
        self.initial_variance = float(initial_variance)
        self.freeze_z = float(freeze_z)

    @staticmethod
    def _logit(risk):
        clipped = np.clip(risk, 1e-6, 1.0 - 1e-6)
        return np.log(clipped) - np.log1p(-clipped)

    def transform(self, risk_scores):
        risk = np.asarray(risk_scores, dtype=float).reshape(-1)
        if not np.all(np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
            raise ValueError("risk_scores must be finite and lie in [0, 1].")
        if risk.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        logits = self._logit(risk)
        evidence = np.zeros(risk.size, dtype=float)
        mean = float(logits[0])
        variance = self.initial_variance
        for index, value in enumerate(logits[1:], start=1):
            scale = max(np.sqrt(variance), 0.25)
            z_score = (float(value) - mean) / scale
            evidence[index] = 1.0 / (
                1.0 + np.exp(-np.clip((z_score - self.alert_z) / self.temperature, -60, 60))
            )
            if z_score < self.freeze_z:
                delta = float(value) - mean
                mean += self.alpha * delta
                variance = (1.0 - self.alpha) * (
                    variance + self.alpha * delta * delta
                )
        return evidence, logits
