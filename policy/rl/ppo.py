"""PPO (Schulman et al. 2017) in pure JAX for the SGS teacher (the paper trains with PPO, §3.3).

Generic over a batched env with init(key, batch) -> (carry, obs) and
step(carry, action) -> (carry, obs, reward, done, info). One jitted iteration does the rollout,
GAE, and epochs of minibatch updates. Kept small: Gaussian MLP actor with a state-independent log
std, MLP critic, running observation normalization, Adam with global-norm gradient clipping.
Episodes that end at the step limit are treated as terminal (no bootstrap), a simplification.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

LOG_2PI = jnp.log(2 * jnp.pi)


@dataclass(frozen=True)
class PPOConfig:
    envs: int = 4096
    steps: int = 32  # rollout length per env per iteration
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    reward_scale: float = 0.01  # Eq. 5's success term is 100; keeps value targets near 1
    hidden: tuple = (256, 256, 128)
    init_log_std: float = -0.5


# ---- network ----
def mlp_init(key, sizes, last_scale=1.0):
    keys, ortho = jax.random.split(key, len(sizes) - 1), jax.nn.initializers.orthogonal
    layers = []
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:], strict=True)):
        scale = last_scale if i == len(sizes) - 2 else jnp.sqrt(2.0)
        layers.append((ortho(scale)(k, (n_in, n_out)), jnp.zeros(n_out)))
    return layers


def mlp(layers, x):
    for w, b in layers[:-1]:
        x = jax.nn.elu(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


def init_params(key, obs_dim, act_dim, cfg: PPOConfig):
    ka, kc = jax.random.split(key)
    return {
        "actor": mlp_init(ka, (obs_dim, *cfg.hidden, act_dim), last_scale=0.01),
        "log_std": jnp.full(act_dim, cfg.init_log_std),
        "critic": mlp_init(kc, (obs_dim, *cfg.hidden, 1)),
    }


def log_prob(params, obs, action):
    mean, log_std = mlp(params["actor"], obs), params["log_std"]
    z = (action - mean) / jnp.exp(log_std)
    return -0.5 * (z**2 + LOG_2PI).sum(-1) - log_std.sum()


def value(params, obs):
    return mlp(params["critic"], obs)[..., 0]


# ---- observation normalization (parallel Welford) ----
class Normalizer(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def norm_init(dim):
    return Normalizer(jnp.zeros(dim), jnp.ones(dim), jnp.array(1e-4))


def norm_update(n: Normalizer, x):
    x = x.reshape(-1, x.shape[-1])
    m, v, c = x.mean(0), x.var(0), x.shape[0]
    total = n.count + c
    delta = m - n.mean
    mean = n.mean + delta * c / total
    var = (n.var * n.count + v * c + delta**2 * n.count * c / total) / total
    return Normalizer(mean, var, total)


def normalize(n: Normalizer, x):
    return jnp.clip((x - n.mean) / jnp.sqrt(n.var + 1e-8), -5, 5)


# ---- Adam with global-norm clipping ----
class Adam(NamedTuple):
    m: dict
    v: dict
    t: jax.Array


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return Adam(zeros, zeros, jnp.array(0))


def adam_step(params, grads, opt: Adam, lr, max_norm, b1=0.9, b2=0.999, eps=1e-8):
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, max_norm / (norm + 1e-6)), grads)
    t = opt.t + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt.m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g**2, opt.v, grads)
    step = lambda p, m, v: p - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)  # noqa: E731
    return jax.tree.map(step, params, m, v), Adam(m, v, t)


# ---- GAE ----
def gae(rewards, values, dones, last_value, gamma, lam):
    """rewards, values, dones: (T, B). A done step does not bootstrap from the next value."""

    def back(carry, x):
        adv_next, v_next = carry
        r, v, d = x
        nonterminal = 1.0 - d
        delta = r + gamma * v_next * nonterminal - v
        adv = delta + gamma * lam * nonterminal * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(
        back, (jnp.zeros_like(last_value), last_value), (rewards, values, dones), reverse=True
    )
    return adv, adv + values


# ---- training ----
class Learner(NamedTuple):
    params: dict
    opt: Adam
    norm: Normalizer
    env: object
    obs: jax.Array
    key: jax.Array


def make_train(env, obs_dim: int, act_dim: int, cfg: PPOConfig, metrics_fn: Callable | None = None):
    """Returns (init(key) -> Learner, iteration(Learner) -> (Learner, metrics)); jit the latter."""

    def init(key):
        kp, ke, key = jax.random.split(key, 3)
        params = init_params(kp, obs_dim, act_dim, cfg)
        carry, obs = env.init(ke, cfg.envs)
        return Learner(params, adam_init(params), norm_init(obs_dim), carry, obs, key)

    def loss_fn(params, batch):
        obs, act, logp_old, adv, ret = batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = jnp.exp(log_prob(params, obs, act) - logp_old)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vf = 0.5 * ((value(params, obs) - ret) ** 2).mean()
        entropy = (params["log_std"] + 0.5 * (1 + LOG_2PI)).sum()
        kl = ((ratio - 1) - jnp.log(ratio)).mean()
        return pg + cfg.vf_coef * vf - cfg.ent_coef * entropy, {"vf_loss": vf, "entropy": entropy, "kl": kl}

    def iteration(learner: Learner):
        params, norm = learner.params, learner.norm

        def act(carry, _):
            env_carry, obs, key = carry
            key, k = jax.random.split(key)
            nobs = normalize(norm, obs)
            mean = mlp(params["actor"], nobs)
            action = mean + jnp.exp(params["log_std"]) * jax.random.normal(k, mean.shape)
            env_carry, next_obs, reward, done, info = env.step(env_carry, action)
            out = (obs, nobs, action, log_prob(params, nobs, action), value(params, nobs), reward, done, info)
            return (env_carry, next_obs, key), out

        (env_carry, obs, key), traj = jax.lax.scan(
            act, (learner.env, learner.obs, learner.key), None, length=cfg.steps
        )
        raw, nobs, actions, logps, values, rewards, dones, info = traj
        dones = dones.astype(jnp.float32)
        last = value(params, normalize(norm, obs))
        adv, ret = gae(rewards * cfg.reward_scale, values, dones, last, cfg.gamma, cfg.lam)

        flat = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (nobs, actions, logps, adv, ret))
        n, mb = cfg.steps * cfg.envs, cfg.steps * cfg.envs // cfg.minibatches

        def epoch(carry, k):
            params, opt = carry
            perm = jax.random.permutation(k, n)
            shuffled = jax.tree.map(lambda x: x[perm].reshape(cfg.minibatches, mb, *x.shape[1:]), flat)

            def minibatch(carry, batch):
                params, opt = carry
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
                params, opt = adam_step(params, grads, opt, cfg.lr, cfg.max_grad_norm)
                return (params, opt), aux

            return jax.lax.scan(minibatch, (params, opt), shuffled)

        key, k = jax.random.split(key)
        (params, opt), aux = jax.lax.scan(epoch, (params, learner.opt), jax.random.split(k, cfg.epochs))
        episodes = dones.sum()
        metrics = {
            "reward": rewards.mean(),
            "episodes": episodes,
            **{k: v.mean() for k, v in aux.items()},
        }
        if "success" in info:
            metrics["success"] = (info["success"] * dones).sum() / jnp.maximum(episodes, 1)
        if metrics_fn is not None:
            metrics.update(metrics_fn(dones, info, env_carry))
        new = Learner(params, opt, norm_update(norm, raw), env_carry, obs, key)
        return new, metrics

    return init, iteration
