"""Success Guided Sampling in JAX, for the jitted teacher training loop.

Same math as policy/rl/sgs.py, the NumPy reference this is tested against (arXiv:2610.12465v1
§3.2 Eqs. 1-2, App. B.2): per-configuration circular histories of the last H outcomes,
Beta-shaped weights in log space, the epsilon floor, then a softmax at temperature max(T, 1).
Empty histories use `unseen` (a local choice; the paper does not specify it).

All functions are pure and jit-friendly. Many envs finish episodes in the same step, so record()
takes a batch with a validity mask. Repeated IDs in one batch apply in array order, exactly as
sequential SGSSampler.record calls would, without a per-episode loop.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SGSParams(NamedTuple):
    """Paper manipulation defaults (§3.2)."""

    target: float = 0.5
    concentration: float = 1.0
    epsilon: float = 1e-4
    temperature: float = 2.0
    unseen: float = 0.5


DEFAULTS = SGSParams()


class SGSState(NamedTuple):
    history: jax.Array  # (N, H) bool, circular
    counts: jax.Array  # (N,) outcomes held, <= H
    successes: jax.Array  # (N,)
    next: jax.Array  # (N,) slot the next outcome overwrites


def init(n_configs: int, history_size: int = 100) -> SGSState:
    zeros = jnp.zeros(n_configs, jnp.int32)
    return SGSState(jnp.zeros((n_configs, history_size), bool), zeros, zeros, zeros)


def record(state: SGSState, ids: jax.Array, success: jax.Array, valid: jax.Array | None = None) -> SGSState:
    """Append one outcome per valid entry; entries with valid=False are ignored."""
    n, h = state.history.shape
    ids = jnp.asarray(ids, jnp.int32)
    valid = jnp.ones(ids.shape, bool) if valid is None else jnp.asarray(valid, bool)
    ids = jnp.where(valid, ids, n)  # id n is out of range: dropped below
    # rank of each entry among earlier entries with the same id
    order = jnp.argsort(ids, stable=True)
    sorted_ids = ids[order]
    first = jnp.searchsorted(sorted_ids, sorted_ids, side="left")
    rank = jnp.zeros_like(ids).at[order].set(jnp.arange(ids.size, dtype=jnp.int32) - first)
    added = jnp.bincount(ids, length=n + 1)[:n]
    # only each id's last H outcomes in this batch survive, and their slots are distinct
    keep = rank >= jnp.append(added, 0)[ids] - h
    rows = jnp.where(keep, ids, n)
    cols = (jnp.append(state.next, 0)[ids] + rank) % h
    history = state.history.at[rows, cols].set(jnp.asarray(success, bool), mode="drop")
    return SGSState(
        history,
        jnp.minimum(state.counts + added, h),
        history.sum(axis=1, dtype=jnp.int32),
        (state.next + added) % h,
    )


def success_rates(state: SGSState, params: SGSParams = DEFAULTS) -> jax.Array:
    seen = state.counts > 0
    return jnp.where(seen, state.successes / jnp.maximum(state.counts, 1), params.unseen)


def logits(state: SGSState, params: SGSParams = DEFAULTS) -> jax.Array:
    """Eq. 1 with the B.2 floor, divided by the effective temperature (Eq. 2's softmax input)."""
    p, eps, t = success_rates(state, params), params.epsilon, params.target
    log_w = params.concentration * (t * jnp.log(p + eps) + (1 - t) * jnp.log(1 - p + eps))
    log_eps = jnp.log(eps)
    return jnp.logaddexp(jnp.maximum(log_w, log_eps), log_eps) / max(params.temperature, 1.0)


def probabilities(state: SGSState, params: SGSParams = DEFAULTS) -> jax.Array:
    return jax.nn.softmax(logits(state, params))


def sample(state: SGSState, key: jax.Array, shape=(), params: SGSParams = DEFAULTS) -> jax.Array:
    """IID configuration IDs; drawing does not record outcomes. Inverse-CDF sampling: categorical()
    would draw one Gumbel per (sample, configuration), 537M numbers for 16K draws over 32K IDs."""
    cdf = jnp.cumsum(probabilities(state, params))
    u = jax.random.uniform(key, shape, cdf.dtype) * cdf[-1]
    return jnp.minimum(jnp.searchsorted(cdf, u, side="right"), cdf.size - 1)


def sample_block(
    state: SGSState, key: jax.Array, start: int, size: int, shape=(), params: SGSParams = DEFAULTS
):
    """IID IDs from rows [start, start + size) only, with SGS weights renormalized inside the block."""
    cdf = jnp.cumsum(jax.nn.softmax(jax.lax.dynamic_slice(logits(state, params), (start,), (size,))))
    u = jax.random.uniform(key, shape, cdf.dtype) * cdf[-1]
    return start + jnp.minimum(jnp.searchsorted(cdf, u, side="right"), size - 1)
