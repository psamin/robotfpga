import math
from fractions import Fraction

import numpy as np
import pytest

from armlab.rl.sgs import SGSSampler


def populated(successes, **kwargs):
    sampler = SGSSampler(len(successes), seed=7, **kwargs)
    for config_id, count in enumerate(successes):
        for episode in range(100):
            sampler.record(config_id, episode < count)
    return sampler


@pytest.mark.parametrize("target,k,eps,temp", [(0.5, 1, 1e-4, 2), (0.66, 5, 1e-8, 2), (0.5, 10, 1e-4, 0.2)])
def test_paper_equations_against_scalar_reference(target, k, eps, temp):
    sampler = populated([0, 10, 50, 90, 100], target=target, concentration=k, epsilon=eps, temperature=temp)
    # Independent direct-power calculation of §3.2 equations (1)-(2) plus B.2.
    weights = []
    for p in [0, 0.1, 0.5, 0.9, 1]:
        w = (p + eps) ** (k * target) * (1 - p + eps) ** (k * (1 - target))
        weights.append(math.exp(math.log(max(w, eps) + eps) / max(temp, 1)))
    np.testing.assert_allclose(sampler.probabilities(), np.array(weights) / sum(weights), rtol=1e-13)


def test_manipulation_priority_and_distribution():
    sampler = populated([10, 50, 90])
    p = sampler.probabilities()
    assert np.all(np.isfinite(p)) and np.all(p >= 0) and p.sum() == pytest.approx(1)
    assert p[1] > p[0] == pytest.approx(p[2])
    expected = [math.sqrt(math.sqrt((r + 1e-4) * (1 - r + 1e-4)) + 1e-4) for r in [0.1, 0.5, 0.9]]
    np.testing.assert_allclose(p, np.array(expected) / sum(expected), rtol=1e-13)


def test_partial_history_and_repeated_ring_wraps():
    sampler = SGSSampler(2, seed=1, history_size=3)
    outcomes = [True, False, True, True, False, False, False, True, True, True]
    for i, outcome in enumerate(outcomes):
        sampler.record(0, outcome)
        assert sampler.success_rates()[0] == pytest.approx(np.mean(outcomes[max(0, i - 2):i + 1]))
        assert sampler.success_rates()[1] == 0.5


def test_unseen_fallback_and_no_pseudo_counts():
    sampler = SGSSampler(3, seed=1, unseen_success_rate=0.3)
    np.testing.assert_allclose(sampler.success_rates(), 0.3)
    np.testing.assert_allclose(sampler.probabilities(), 1 / 3)
    sampler.record(1, np.bool_(True))
    np.testing.assert_allclose(sampler.success_rates(), [0.3, 1, 0.3])
    sampler.success_rates()[:] = 0  # Returned arrays cannot change sampler state.
    assert sampler.success_rates()[1] == 1


def test_seeded_sampling_and_replacement():
    a, b = populated([10, 50, 90]), populated([10, 50, 90])
    assert a.sample() == b.sample()
    np.testing.assert_array_equal(a.sample(1000), b.sample(1000))
    assert set(a.sample(1000)) == {0, 1, 2}
    np.testing.assert_allclose(a.success_rates(), [0.1, 0.5, 0.9])
    assert SGSSampler(1, seed=0).sample() == 0
    assert a.sample(0).shape == (0,)


@pytest.mark.parametrize("counts,k,target", [([0, 0], 1, 0.5), ([100, 100], 1, 0.5),
                                         ([0, 50, 100], 1000, 0.5), ([0, 50, 100], 0, 0.5),
                                         ([0, 50, 100], 1, 0), ([0, 50, 100], 1, 1)])
def test_extremes_retain_exploration(counts, k, target):
    p = populated(counts, concentration=k, target=target).probabilities()
    assert np.all(np.isfinite(p)) and np.all(p > 0) and p.sum() == pytest.approx(1)
    if len(set(counts)) == 1 or k == 0:
        np.testing.assert_allclose(p, 1 / len(counts))


@pytest.mark.parametrize("kwargs", [{"n_configs": 0}, {"n_configs": True}, {"n_configs": 1.5},
                                    {"history_size": 0}, {"history_size": 2.5}, {"target": -0.1},
                                    {"target": 1.1}, {"concentration": -1}, {"epsilon": 0},
                                    {"epsilon": 2}, {"temperature": 0}, {"temperature": -1},
                                    {"target": np.nan}, {"concentration": np.inf}, {"epsilon": "bad"},
                                    {"temperature": True}, {"seed": -1}, {"seed": 1.5},
                                    {"unseen_success_rate": 1.1}])
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        SGSSampler(**({"n_configs": 2, "seed": 0} | kwargs))


@pytest.mark.parametrize("config_id,outcome", [(-1, True), (2, True), (0.5, True), (True, True),
                                            ("x", True), (0, 2), (0, 0.5), (0, "yes"), (0, np.nan)])
def test_invalid_record_does_not_change_history(config_id, outcome):
    sampler = SGSSampler(2, seed=0)
    with pytest.raises(ValueError):
        sampler.record(config_id, outcome)
    np.testing.assert_allclose(sampler.success_rates(), 0.5)


@pytest.mark.parametrize("size", [-1, 1.5, True, "2"])
def test_invalid_sample_size(size):
    with pytest.raises(ValueError):
        SGSSampler(2, seed=0).sample(size)


def test_real_hyperparameters_are_converted_before_validation():
    np.testing.assert_allclose(SGSSampler(2, seed=0, concentration=10**20).probabilities(), 0.5)
    np.testing.assert_allclose(
        SGSSampler(2, seed=0, target=Fraction(1, 2)).probabilities(), 0.5
    )
    for kwargs in [{"concentration": 10**400}, {"epsilon": 10**20}, {"target": Fraction(3, 2)}]:
        with pytest.raises(ValueError):
            SGSSampler(2, seed=0, **kwargs)


def test_draw_frequencies_match_independent_manipulation_equation():
    sampler = populated([0, 50, 100])
    weights = [math.sqrt(math.sqrt((r + 1e-4) * (1 - r + 1e-4)) + 1e-4) for r in [0, 0.5, 1]]
    expected = np.array(weights) / sum(weights)
    observed = np.bincount(sampler.sample(20000), minlength=3) / 20000
    np.testing.assert_allclose(observed, expected, atol=0.01, rtol=0)


@pytest.mark.parametrize("history_size", [1, 3, 100])
def test_interleaved_histories_match_independent_windows(history_size):
    kwargs = {} if history_size == 100 else {"history_size": history_size}
    sampler = SGSSampler(3, seed=0, **kwargs)  # Also checks the default H=100.
    histories = [[], [], []]
    for episode in range(363):
        cid, outcome = episode % 3, int(episode % 7 < 2)
        sampler.record(np.int64(cid), np.int64(outcome))
        histories[cid].append(outcome)
        expected = [sum(h[-history_size:]) / len(h[-history_size:]) if h else 0.5 for h in histories]
        np.testing.assert_allclose(sampler.success_rates(), expected)


def test_record_refreshes_probabilities_without_consuming_randomness():
    a, b = SGSSampler(2, seed=1), SGSSampler(2, seed=1)
    np.testing.assert_allclose(a.probabilities(), 0.5)
    for sampler in (a, b):
        sampler.record(0, False)
    assert a.probabilities()[0] < a.probabilities()[1]
    np.testing.assert_array_equal(a.sample(100), b.sample(100))
