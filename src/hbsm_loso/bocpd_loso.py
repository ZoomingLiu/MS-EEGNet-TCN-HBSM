"""Bayesian online change-point detection for scalar risk scores."""

import numpy as np


def _gammaln(values):
    """Vectorized Lanczos approximation for positive values."""
    coefficients = (
        676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7,
    )
    z = np.asarray(values, dtype=float) - 1.0
    series = np.full_like(z, 0.99999999999980993)
    for index, coefficient in enumerate(coefficients, start=1):
        series += coefficient / (z + index)
    shifted = z + 7.5
    return 0.9189385332046727 + (z + 0.5) * np.log(shifted) - shifted + np.log(series)


class BOCPD:
    """Student-t BOCPD with a Normal-Inverse-Gamma predictive model."""

    def __init__(self, hazard_rate=0.01, mean0=0.5, var0=0.05, alpha0=2.0,
                 beta0=0.05, max_run_length=256, change_window=3):
        if not 0.0 < hazard_rate < 1.0:
            raise ValueError("hazard_rate must lie strictly between 0 and 1.")
        if var0 <= 0 or alpha0 <= 0 or beta0 <= 0:
            raise ValueError("var0, alpha0 and beta0 must be positive.")
        if max_run_length < 1 or change_window < 1:
            raise ValueError("max_run_length and change_window must be positive.")
        self.hazard_rate = float(hazard_rate)
        self.mean0, self.kappa0 = float(mean0), float(var0)
        self.alpha0, self.beta0 = float(alpha0), float(beta0)
        self.max_run_length = int(max_run_length)
        self.change_window = int(change_window)
        self.reset_state()

    def reset_state(self):
        self.observation_count = 0
        self.run_length_posterior = np.array([1.0])
        self.means = np.array([self.mean0])
        self.kappas = np.array([self.kappa0])
        self.alphas = np.array([self.alpha0])
        self.betas = np.array([self.beta0])

    @staticmethod
    def _student_t_pdf(x, mean, kappa, alpha, beta):
        scale_squared = beta * (kappa + 1.0) / (alpha * kappa)
        degrees = 2.0 * alpha
        scale_squared = np.maximum(scale_squared, 1e-12)
        standardized_squared = (x - mean) ** 2 / scale_squared
        log_normalizer = (
            _gammaln(0.5 * (degrees + 1.0))
            - _gammaln(0.5 * degrees)
            - 0.5 * np.log(degrees * np.pi * scale_squared)
        )
        return np.exp(log_normalizer - 0.5 * (degrees + 1.0) * np.log1p(standardized_squared / degrees))

    def update(self, x):
        x = float(x)
        if not np.isfinite(x):
            raise ValueError("BOCPD observations must be finite.")
        predictive = np.maximum(self._student_t_pdf(x, self.means, self.kappas, self.alphas, self.betas), 1e-300)
        message = self.run_length_posterior * predictive
        growth = message * (1.0 - self.hazard_rate)
        change = np.sum(message * self.hazard_rate)
        posterior = np.concatenate(([change], growth))
        posterior /= posterior.sum()

        old_means, old_kappas = self.means, self.kappas
        updated_means = (old_kappas * old_means + x) / (old_kappas + 1.0)
        updated_betas = self.betas + old_kappas * (x - old_means) ** 2 / (2.0 * (old_kappas + 1.0))
        self.means = np.concatenate(([self.mean0], updated_means))
        self.kappas = np.concatenate(([self.kappa0], old_kappas + 1.0))
        self.alphas = np.concatenate(([self.alpha0], self.alphas + 0.5))
        self.betas = np.concatenate(([self.beta0], updated_betas))
        self.run_length_posterior = posterior
        if posterior.size > self.max_run_length + 1:
            keep = self.max_run_length + 1
            self.run_length_posterior = posterior[:keep]
            self.run_length_posterior[-1] += posterior[keep:].sum()
            self.run_length_posterior /= self.run_length_posterior.sum()
            self.means = self.means[:keep]
            self.kappas = self.kappas[:keep]
            self.alphas = self.alphas[:keep]
            self.betas = self.betas[:keep]
        self.observation_count += 1
        if self.observation_count <= self.change_window:
            return 0.0
        return float(self.run_length_posterior[:self.change_window].sum())
