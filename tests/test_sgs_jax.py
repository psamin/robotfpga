import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from armlab.rl import sgs_jax  # noqa: E402
from armlab.rl.sgs import SGSSampler  # noqa: E402

jax.config.update("jax_enable_x64", True)  # compare with the float64 NumPy reference


def feed(batches, n, h, params=sgs_jax.DEFAULTS):
    """The same outcomes through the JAX batch recorder and sequential NumPy record calls."""
    ref = SGSSampler(
        n,
        seed=0,
        history_size=h,
        target=params.target,
        concentration=params.concentration,
        epsilon=params.epsilon,
        temperature=params.temperature,
        unseen_success_rate=params.unseen,
    )
    state, rec = sgs_jax.init(n, h), jax.jit(sgs_jax.record)
    for ids, success, valid in batches:
        state = rec(state, jnp.array(ids), jnp.array(success), jnp.array(valid))
        for i, s, v in zip(ids, success, valid, strict=True):
            if v:
                ref.record(int(i), bool(s))
    return state, ref


def random_batches(rng, n, count, size):
    return [
        (rng.integers(n, size=size), rng.random(size) < rng.random(), rng.random(size) < 0.8)
        for _ in range(count)
    ]


@pytest.mark.parametrize("h", [1, 3, 100])
def test_batched_record_matches_sequential_reference(h):
    state, ref = feed(random_batches(np.random.default_rng(h), n=7, count=30, size=16), n=7, h=h)
    assert np.array_equal(state.counts, ref._counts)
    assert np.array_equal(state.successes, ref._successes)
    assert np.array_equal(state.next, ref._next)
    assert np.array_equal(state.history, ref._history)


def test_more_repeats_than_history_in_one_batch_keeps_the_last_h():
    ids = [2] * 7 + [0]
    success = [True, True, True, True, False, True, False, True]
    state, ref = feed([(ids, success, [True] * 8)], n=3, h=3)
    assert np.array_equal(state.history, ref._history)
    assert state.successes[2] == 1 and state.counts[2] == 3


@pytest.mark.parametrize("target,concentration,temperature", [(0.5, 1, 2), (0.66, 5, 2), (0.3, 10, 0.5)])
def test_probabilities_match_reference(target, concentration, temperature):
    params = sgs_jax.SGSParams(target, concentration, 1e-4, temperature, 0.5)
    state, ref = feed(
        random_batches(np.random.default_rng(1), n=9, count=10, size=8), n=9, h=5, params=params
    )
    assert np.allclose(sgs_jax.probabilities(state, params), ref.probabilities(), rtol=1e-12)
    assert np.allclose(sgs_jax.success_rates(state, params), ref.success_rates())


def test_sampling_frequencies_follow_probabilities():
    state, _ = feed([([0] * 4 + [1] * 4, [1, 1, 1, 1, 1, 0, 1, 0], [True] * 8)], n=3, h=4)
    probs = np.asarray(sgs_jax.probabilities(state))
    draws = np.asarray(sgs_jax.sample(state, jax.random.key(0), (200_000,)))
    assert np.allclose(np.bincount(draws, minlength=3) / draws.size, probs, atol=0.005)


def test_invalid_entries_are_ignored():
    state = sgs_jax.record(
        sgs_jax.init(2, 4), jnp.array([0, 1]), jnp.array([True, True]), jnp.array([False, False])
    )
    assert int(state.counts.sum()) == 0 and not bool(state.history.any())
