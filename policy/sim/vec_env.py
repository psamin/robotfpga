"""Batched pick-and-place teacher env with SGS-driven resets over size variants (arXiv:2610.12465v1 §3.2).

Warp cannot batch differently sized models, so there is one TeacherMJX sub-batch per size variant,
all stepped inside the same jitted function. Every env starts each episode from one reset-buffer row
(an SGS configuration). When an episode ends, its terminal success is recorded for that row in one
SGS state shared by all rows, and the env restarts from a row of its own variant, drawn by SGS
(weights renormalized within the variant's block of rows) or uniformly for the baseline.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from policy.rl import sgs_jax
from policy.sim.scene import SceneParams
from policy.sim.teacher_mjx import TeacherMJX

STRATEGIES = ("sgs", "uniform")


class Carry(NamedTuple):
    envs: tuple  # one batched TeacherMJX state per variant
    config: tuple  # per variant: (n,) buffer row each env's episode started from
    sgs: sgs_jax.SGSState
    key: jax.Array


class VariantVecEnv:
    def __init__(
        self,
        buffer: dict,
        variants: list[SceneParams],
        envs_per_variant: int,
        strategy: str = "sgs",
        params: sgs_jax.SGSParams = sgs_jax.DEFAULTS,
        history: int = 100,
        impl: str = "warp",
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        v = np.asarray(buffer["variant"])
        if not (np.diff(v) >= 0).all() or set(v) != set(range(len(variants))):
            raise ValueError("buffer rows must be grouped by variant 0..K-1")
        self.blocks = [(int(np.argmax(v == k)), int((v == k).sum())) for k in range(len(variants))]
        self.rows = {f: jnp.asarray(buffer[f]) for f in ("qpos", "qvel", "ctrl", "plate_xy")}
        self.n, self.per, self.strategy = len(v), envs_per_variant, strategy
        self.params, self.history = params, history
        self.sims = [TeacherMJX(p, batch=envs_per_variant, impl=impl) for p in variants]
        self._reset = [jax.vmap(s.reset) for s in self.sims]
        self._step = [jax.vmap(s.step) for s in self.sims]
        row0 = [self.rows[f][0] for f in ("qpos", "qvel", "ctrl", "plate_xy")]
        self._world = [jax.eval_shape(s.reset, *row0)[0] for s in self.sims]

    @property
    def batch(self) -> int:
        return self.per * len(self.sims)

    def _draw(self, sgs, key, k):
        start, size = self.blocks[k]
        if self.strategy == "uniform":
            return start + jax.random.randint(key, (self.per,), 0, size)
        return sgs_jax.sample_block(sgs, key, start, size, (self.per,), self.params)

    def _start(self, k, ids):
        r = self.rows
        return self._reset[k](r["qpos"][ids], r["qvel"][ids], r["ctrl"][ids], r["plate_xy"][ids])

    def init(self, key: jax.Array, batch: int):
        if batch != self.batch:
            raise ValueError(f"batch must be {self.batch} ({len(self.sims)} variants x {self.per})")
        sgs, envs, configs, obs = sgs_jax.init(self.n, self.history), [], [], []
        for k in range(len(self.sims)):
            key, sub = jax.random.split(key)
            ids = self._draw(sgs, sub, k)
            e, o = self._start(k, ids)
            envs.append(e)
            configs.append(ids)
            obs.append(o)
        return Carry(tuple(envs), tuple(configs), sgs, key), jnp.concatenate(obs)

    def step(self, c: Carry, actions):
        outs = [
            self._step[k](c.envs[k], actions[k * self.per : (k + 1) * self.per])
            for k in range(len(self.sims))
        ]
        done = jnp.concatenate([o[3] for o in outs])
        success = jnp.concatenate([o[4]["success"] for o in outs])
        sgs = sgs_jax.record(c.sgs, jnp.concatenate(c.config), success, done)
        key, envs, configs, obs = c.key, [], [], []
        for k, (env, o, _, d, _) in enumerate(outs):
            key, sub = jax.random.split(key)
            ids = self._draw(sgs, sub, k)
            fresh, fresh_obs = self._start(k, ids)

            def pick(new, old, d=d):
                return jnp.where(d.reshape(d.shape + (1,) * (old.ndim - 1)), new, old)

            def pick_world(new, old, world, d=d):
                return pick(new, old) if old.shape == d.shape + world.shape else old

            envs.append(jax.tree.map(pick_world, fresh, env, self._world[k]))
            configs.append(pick(ids, c.config[k]))
            obs.append(pick(fresh_obs, o))
        info = {
            "success": success,
            "failure": jnp.concatenate([o[4]["failure"] for o in outs]),
            "config": jnp.concatenate(c.config),
        }
        reward = jnp.concatenate([o[2] for o in outs])
        return Carry(tuple(envs), tuple(configs), sgs, key), jnp.concatenate(obs), reward, done, info
