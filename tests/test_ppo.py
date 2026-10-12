import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from armlab.rl import ppo  # noqa: E402


class Toy:
    """One-step episodes: observe a target in [-0.5, 0.5]^2, reward -|action - target|^2."""

    def init(self, key, batch):
        key, k = jax.random.split(key)
        target = jax.random.uniform(k, (batch, 2), minval=-0.5, maxval=0.5)
        return (key, target), target

    def step(self, carry, action):
        key, target = carry
        reward = -((jnp.clip(action, -1, 1) - target) ** 2).sum(-1)
        (key, new), obs = self.init(key, target.shape[0])
        return (key, new), obs, reward, jnp.ones(target.shape[0], bool), {}


def test_gae_matches_loop_reference():
    rng = np.random.default_rng(0)
    t, b, gamma, lam = 6, 3, 0.9, 0.8
    r, v, last = rng.normal(size=(t, b)), rng.normal(size=(t, b)), rng.normal(size=b)
    d = (rng.random((t, b)) < 0.3).astype(float)
    adv, ret = ppo.gae(jnp.array(r), jnp.array(v), jnp.array(d), jnp.array(last), gamma, lam)
    ref, a, v_next = np.zeros((t, b)), np.zeros(b), last
    for i in reversed(range(t)):
        delta = r[i] + gamma * v_next * (1 - d[i]) - v[i]
        a = delta + gamma * lam * (1 - d[i]) * a
        ref[i], v_next = a, v[i]
    assert np.allclose(adv, ref, atol=1e-5) and np.allclose(ret, ref + v, atol=1e-5)


def test_normalizer_matches_numpy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(2, 3, (50, 4)), rng.normal(-1, 0.5, (30, 4))
    n = ppo.norm_update(ppo.norm_update(ppo.norm_init(4), jnp.array(a)), jnp.array(b))
    both = np.concatenate([a, b])
    assert np.allclose(n.mean, both.mean(0), atol=1e-3) and np.allclose(n.var, both.var(0), rtol=1e-3)


def test_adam_first_step_moves_by_lr_and_clips():
    params = {"w": jnp.array([1.0, -2.0])}
    new, opt = ppo.adam_step(params, {"w": jnp.array([10.0, -0.5])}, ppo.adam_init(params), 0.1, 1.0)
    assert np.allclose(new["w"], [0.9, -1.9], atol=1e-4) and int(opt.t) == 1  # sign(g) * lr


def test_ppo_learns_toy_task():
    cfg = ppo.PPOConfig(envs=256, steps=4, epochs=4, minibatches=4, lr=3e-3, hidden=(64, 64))
    init, iteration = ppo.make_train(Toy(), 2, 2, cfg)
    learner, step = init(jax.random.key(0)), jax.jit(iteration)
    rewards = []
    for _ in range(60):
        learner, m = step(learner)
        rewards.append(float(m["reward"]))
    assert np.mean(rewards[-5:]) > 0.25 * np.mean(rewards[:3])  # error at least 4x smaller
    assert float(m["episodes"]) == 256 * 4
