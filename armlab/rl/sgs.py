"""Success Guided Sampling: A Balanced Data Diet, arXiv:2610.12465v1, §3.2 and B.2."""

from numbers import Integral, Real

import numpy as np


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class SGSSampler:
    """Sample fixed integer configuration IDs in [0, n_configs), with replacement.

    Defaults are the paper's manipulation settings. Empty histories use
    unseen_success_rate (a local choice, not specified by the paper); once an
    episode is recorded, rates use only real outcomes, without pseudo-counts.
    Record once per completed episode, in completion order. Not thread-safe.
    """

    def __init__(
        self,
        n_configs: int,
        *,
        seed: int,
        history_size: int = 100,
        target: float = 0.5,
        concentration: float = 1.0,
        epsilon: float = 1e-4,
        temperature: float = 2.0,
        unseen_success_rate: float = 0.5,
    ):
        n_configs = _integer("n_configs", n_configs, 1)
        history_size = _integer("history_size", history_size, 1)
        seed = _integer("seed", seed, 0)
        settings = (target, concentration, epsilon, temperature, unseen_success_rate)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in settings):
            raise ValueError("sampler hyperparameters must be finite real numbers")
        try:
            settings = tuple(map(float, settings))
        except (OverflowError, ValueError) as exc:
            raise ValueError("sampler hyperparameters must be representable as float64") from exc
        if not all(np.isfinite(v) for v in settings):
            raise ValueError("sampler hyperparameters must be finite real numbers")
        target, concentration, epsilon, temperature, unseen_success_rate = settings
        if not 0 <= target <= 1 or concentration < 0 or not 0 < epsilon <= 1 or temperature <= 0:
            raise ValueError(
                "require target in [0, 1], concentration >= 0, epsilon in (0, 1], temperature > 0"
            )
        if not 0 <= unseen_success_rate <= 1:
            raise ValueError("unseen_success_rate must be in [0, 1]")
        self._target, self._concentration, self._epsilon = target, concentration, epsilon
        self._temperature = max(temperature, 1.0)  # Appendix B.2
        self._unseen = unseen_success_rate
        self._history = np.zeros((n_configs, history_size), dtype=np.bool_)
        self._counts = np.zeros(n_configs, dtype=np.int64)
        self._successes = np.zeros(n_configs, dtype=np.int64)
        self._next = np.zeros(n_configs, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

    def record(self, config_id: int, success: bool | int) -> None:
        """Append a Boolean or integer 0/1 outcome; evict the oldest after H episodes."""
        config_id = _integer("config_id", config_id, 0)
        if config_id >= len(self._counts):
            raise ValueError("unknown config_id")
        if not isinstance(success, (Integral, bool, np.bool_)) or success not in (0, 1):
            raise ValueError("success must be a Boolean or integer 0/1")
        slot = self._next[config_id]
        self._successes[config_id] += int(success) - int(self._history[config_id, slot])
        self._history[config_id, slot] = success
        self._counts[config_id] = min(self._counts[config_id] + 1, self._history.shape[1])
        self._next[config_id] = (slot + 1) % self._history.shape[1]

    def success_rates(self) -> np.ndarray:
        """Return rates in configuration-ID order; empty histories use the explicit fallback."""
        rates = np.full(len(self._counts), self._unseen, dtype=np.float64)
        np.divide(self._successes, self._counts, out=rates, where=self._counts > 0)
        return rates

    def probabilities(self) -> np.ndarray:
        """Equation (1), B.2's floor/clamp, then equation (2), evaluated in log space."""
        p, eps = self.success_rates(), self._epsilon
        # Beta(1 + k*t, 1 + k*(1-t)) kernel; no Beta PDF normalization constant.
        with np.errstate(over="ignore"):  # Very large k may give -inf, safely floored below.
            log_weight = self._concentration * (
                self._target * np.log(p + eps) + (1 - self._target) * np.log(1 - p + eps)
            )
        logits = np.logaddexp(np.maximum(log_weight, np.log(eps)), np.log(eps))
        weights = np.exp((logits - logits.max()) / self._temperature)
        probabilities = weights / weights.sum()
        if not np.all(np.isfinite(probabilities) & (probabilities > 0)):
            raise ValueError("hyperparameters exceed float64 probability range")
        return probabilities

    def sample(self, size: int | None = None) -> int | np.ndarray:
        """Draw one ID or an array of independent IDs; drawing does not record outcomes."""
        if size is not None:
            size = _integer("size", size, 0)
        result = self._rng.choice(len(self._counts), size=size, p=self.probabilities())
        return int(result) if size is None else result
