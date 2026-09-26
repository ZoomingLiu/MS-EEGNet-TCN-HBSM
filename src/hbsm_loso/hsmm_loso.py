"""Causal explicit-duration state filter for LOSO warning evidence."""

import numpy as np


def _logsumexp(values, axis=None, keepdims=False):
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide="ignore"):
        result = finite_maximum + np.log(
            np.sum(np.exp(values - finite_maximum), axis=axis, keepdims=True)
        )
    result = np.where(np.isfinite(maximum), result, -np.inf)
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


class HSMM:
    """Three-state filter with explicit attack and release durations.

    ``transition_steps`` consecutive high-evidence observations are required
    to enter preictal warning.  ``release_steps`` consecutive low-evidence
    observations are required to leave it. Expanded microstates implement the
    explicit durations while the public posterior remains three-dimensional.
    """

    def __init__(self, n_states=3, d_max=None, transition_steps=8,
                 release_steps=6, branch_probability=0.5,
                 emission_strength=15.0):
        if n_states != 3:
            raise ValueError("HSMM exposes exactly three states.")
        if d_max is not None:
            transition_steps = int(d_max)
        if transition_steps < 1 or release_steps < 1:
            raise ValueError("transition_steps and release_steps must be positive.")
        if not 0.0 < branch_probability < 1.0:
            raise ValueError("branch_probability must lie strictly between 0 and 1.")
        if emission_strength <= 0:
            raise ValueError("emission_strength must be positive.")

        self.n_states = n_states
        self.transition_steps = int(transition_steps)
        self.release_steps = int(release_steps)
        self.emission_strength = float(emission_strength)
        self.preictal_index = self.transition_steps + 1
        self.release_count = self.release_steps - 1
        self.n_internal = self.preictal_index + 1 + self.release_count
        self.release_slice = slice(self.preictal_index + 1, self.n_internal)

        self.A = np.zeros((self.n_internal, self.n_internal), dtype=float)
        stay = 1.0 - branch_probability
        advance = branch_probability
        self.A[0, 0], self.A[0, 1] = stay, advance
        for state in range(1, self.preictal_index):
            self.A[state, 0], self.A[state, state + 1] = stay, advance

        if self.release_count == 0:
            self.A[self.preictal_index, self.preictal_index] = stay
            self.A[self.preictal_index, 0] = advance
        else:
            first_release = self.preictal_index + 1
            self.A[self.preictal_index, self.preictal_index] = stay
            self.A[self.preictal_index, first_release] = advance
            for state in range(first_release, self.n_internal - 1):
                self.A[state, self.preictal_index] = stay
                self.A[state, state + 1] = advance
            self.A[-1, self.preictal_index], self.A[-1, 0] = stay, advance

        self._log_transition = np.full_like(self.A, -np.inf)
        np.log(self.A, out=self._log_transition, where=self.A > 0)

    def fit_emissions(self, X=None, y=None):
        """Fixed evidence emissions require no held-out-label fitting."""
        return self

    def _compute_emission_log_probs(self, evidence_scores):
        evidence = np.asarray(evidence_scores, dtype=float).reshape(-1)
        if not np.all(np.isfinite(evidence)):
            raise ValueError("evidence_scores must contain only finite values.")
        evidence = np.clip(evidence, 0.0, 1.0)
        safe = -self.emission_strength * evidence**2
        active = -self.emission_strength * (evidence - 1.0) ** 2
        emissions = np.empty((evidence.size, self.n_internal), dtype=float)
        emissions[:, 0] = safe
        emissions[:, 1:self.preictal_index + 1] = active[:, None]
        if self.release_count:
            emissions[:, self.release_slice] = safe[:, None]
        return emissions

    def forward_inference(self, X=None, risk_scores=None):
        """Return strictly causal probabilities for interictal/transition/preictal."""
        if risk_scores is None:
            raise ValueError("risk_scores is required.")
        emissions = self._compute_emission_log_probs(risk_scores)
        if emissions.shape[0] == 0:
            return np.empty((0, self.n_states), dtype=float)
        alpha = np.full_like(emissions, -np.inf)
        alpha[0, 0] = emissions[0, 0]
        for time_index in range(1, emissions.shape[0]):
            alpha[time_index] = _logsumexp(
                alpha[time_index - 1, :, None] + self._log_transition, axis=0
            ) + emissions[time_index]
        internal = np.exp(alpha - _logsumexp(alpha, axis=1, keepdims=True))
        result = np.empty((emissions.shape[0], self.n_states), dtype=float)
        result[:, 0] = internal[:, 0]
        result[:, 1] = internal[:, 1:self.preictal_index].sum(axis=1)
        result[:, 2] = internal[:, self.preictal_index:].sum(axis=1)
        return result
