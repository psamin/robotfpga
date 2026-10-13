"""Batched teacher env with SGS-driven resets (arXiv:2610.12465v1 §3.2).

Every env starts each episode from one reset-buffer row (an SGS configuration). When an episode
ends, its terminal success is recorded for that row and the env restarts from a row drawn by the
sampler: SGS weights, or uniform for the baseline. Outcomes are recorded under both strategies, so
success rates per configuration are comparable. step() is jit-friendly; the whole batch shares one
SGS state, which is why this is not a per-env wrapper.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from armlab.rl import sgs_jax
from armlab.rl.teacher_mjx import State, TeacherMJX

STRATEGIES = ("sgs", "uniform")


class Carry(NamedTuple):
    env: State  # batched TeacherMJX state
    config: jax.Array  # (B,) buffer row each env's episode started from
    sgs: sgs_jax.SGSState
    key: jax.Array


class SGSVecEnv:
    def __init__(
        self,
        sim: TeacherMJX,
        buffer: dict,
        strategy: str = "sgs",
        params: sgs_jax.SGSParams = sgs_jax.DEFAULTS,
        history: int = 100,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.rows = {k: jnp.asarray(buffer[k]) for k in ("qpos", "qvel", "ctrl", "instr")}
        self.n = len(self.rows["instr"])
        self.strategy, self.params, self.history = strategy, params, history
        self._reset, self._step = jax.vmap(sim.reset), jax.vmap(sim.step)
        # Per-world leaf shapes. Warp's batched Data also holds leaves shared by the whole batch
        # (the contact pool, rebuilt by collision detection every step, and counters); those are
        # scratch, so finished envs reset only the per-world leaves.
        self._world = jax.eval_shape(sim.reset, *(v[0] for v in self.rows.values()))[0]

    def _draw(self, sgs, key, batch):
        if self.strategy == "uniform":
            return jax.random.randint(key, (batch,), 0, self.n)
        return sgs_jax.sample(sgs, key, (batch,), self.params)

    def _start(self, ids):
        r = self.rows
        return self._reset(r["qpos"][ids], r["qvel"][ids], r["ctrl"][ids], r["instr"][ids])

    def init(self, key: jax.Array, batch: int) -> tuple[Carry, jax.Array]:
        key, k = jax.random.split(key)
        sgs = sgs_jax.init(self.n, self.history)
        ids = self._draw(sgs, k, batch)
        env, obs = self._start(ids)
        return Carry(env, ids, sgs, key), obs

    def step(self, c: Carry, actions) -> tuple[Carry, jax.Array, jax.Array, jax.Array, dict]:
        """Finished envs return their terminal reward/done/info and the next episode's first obs."""
        env, obs, reward, done, info = self._step(c.env, actions)
        sgs = sgs_jax.record(c.sgs, c.config, info["success"], done)
        key, k = jax.random.split(c.key)
        ids = self._draw(sgs, k, done.shape[0])
        fresh, fresh_obs = self._start(ids)

        def pick(new, old):
            return jnp.where(done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old)

        def pick_world(new, old, world):
            return pick(new, old) if old.shape == done.shape + world.shape else old

        env = jax.tree.map(pick_world, fresh, env, self._world)
        info = {**info, "config": c.config}  # the row the finished episode started from
        return Carry(env, pick(ids, c.config), sgs, key), pick(fresh_obs, obs), reward, done, info
