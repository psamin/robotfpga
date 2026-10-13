"""SGS teacher env on MJX (MuJoCo Warp backend): the GPU twin of armlab/rl/teacher_env.py.

Same observation, action, reward and termination as the CPU reference; tests replay identical
actions through both. reset() and step() act on one world and are pure, so callers jit(vmap(.))
over a batch. "Cube touching the robot" uses a contact sensor (cube geom vs. the robot's body
subtree) instead of Warp's private per-batch contact pool; it agrees with
BlockSortEnv.touching_robot on CPU.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from armlab.rl.resets import TABLE_X, TABLE_Y
from armlab.rl.teacher_env import (
    ACTION_SCALE,
    JOINT_SPEED,
    MAX_STEPS,
    PENALTY_CLAMP,
    W_ACTION,
    W_FAILURE,
    W_JOINT_VEL,
    W_RATE,
    W_SUCCESS,
)
from armlab.sim.env import BIN_HALF, INSTRUCTIONS, SCENE, SUBSTEPS

COLORS, SIDES = ("red", "blue"), ("left", "right")
TARGET = jnp.array([COLORS.index(c) for c, _ in INSTRUCTIONS])  # instruction -> cube index
BIN = jnp.array([SIDES.index(s) for _, s in INSTRUCTIONS])  # instruction -> bin index
# Warp preallocates contacts (total over the batch) and constraint rows (per world). Expert rollouts
# peaked at 32 contacts and 190 rows per world; overflow drops contacts with only a printed warning.
CONTACTS_PER_WORLD, NJMAX = 48, 256
# The scene caps the solver at 10 iterations and line search at 20; CPU MuJoCo hits those caps
# silently, Warp prints a line each time. Silence those two, keep every buffer-overflow warning.
QUIET = 512 | 1024  # mujoco_warp.OverflowType.ITERATIONS | LS_ITERATIONS


def build_model() -> mujoco.MjModel:
    """The block-sort scene plus one contact sensor per cube (sensordata[i] > 0: touching robot)."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for c in COLORS:
        s = spec.add_sensor(
            name=f"touch_{c}",
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=f"cube_{c}",
            reftype=mujoco.mjtObj.mjOBJ_XBODY,  # the robot's whole subtree
            refname="base",
        )
        s.intprm[:3] = [1, 1, 1]  # data "found", no reduction ordering, one slot
    return spec.compile()


class State(NamedTuple):
    data: mjx.Data
    target: jax.Array  # joint targets (rad)
    prev: jax.Array  # previous action
    t: jax.Array
    instr: jax.Array


class TeacherMJX:
    def __init__(self, batch: int, impl: str = "warp", max_steps: int = MAX_STEPS):
        m = build_model()
        mx = mjx.put_model(m, impl=impl)
        if impl == "warp":
            warp_opt = mx.opt._impl.replace(warn_overflow=mx.opt._impl.warn_overflow & ~QUIET)
            mx = mx.replace(opt=mx.opt.replace(_impl=warp_opt))
        self.mx = mx
        sizes = {"naconmax": CONTACTS_PER_WORLD * batch, "njmax": NJMAX} if impl == "warp" else {}
        self.d0 = mjx.make_data(m, impl=impl, **sizes)
        self.max_steps = max_steps
        self.lo, self.hi = jnp.array(m.actuator_ctrlrange[:, 0]), jnp.array(m.actuator_ctrlrange[:, 1])
        self.cubes = jnp.array([m.body(f"cube_{c}").id for c in COLORS])
        self.bins = jnp.array([m.body(f"bin_{s}").id for s in SIDES])
        self.site = m.site("gripperframe").id

    def reset(self, qpos, qvel, ctrl, instr) -> tuple[State, jax.Array]:
        d = mjx.forward(self.mx, self.d0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
        s = State(d, jnp.asarray(ctrl), jnp.zeros(6, d.qpos.dtype), jnp.int32(0), jnp.int32(instr))
        return s, self.obs(s)

    def step(self, s: State, action) -> tuple[State, jax.Array, jax.Array, jax.Array, dict]:
        a = jnp.clip(jnp.asarray(action, s.target.dtype), -1, 1)
        target = jnp.clip(s.target + jnp.asarray(ACTION_SCALE, a.dtype) * a, self.lo, self.hi)
        d = s.data.replace(ctrl=target)
        d = jax.lax.fori_loop(0, SUBSTEPS, lambda _, d: mjx.step(self.mx, d), d)
        success, failure = self._success(d, s.instr), self._failed(d)
        clamp = lambda x: jnp.minimum(x, PENALTY_CLAMP)  # noqa: E731
        terms = {
            "success": W_SUCCESS * success,
            "failure": -W_FAILURE * failure,
            "action": -W_ACTION * clamp(a @ a),
            "action_rate": -W_RATE * clamp((a - s.prev) @ (a - s.prev)),
            "joint_vel": -W_JOINT_VEL * clamp(d.qvel[:5] @ d.qvel[:5]),
        }
        s = State(d, target, a, s.t + 1, s.instr)
        done = success | failure | (s.t >= self.max_steps)
        info = {"success": success, "failure": failure, "terms": terms}
        return s, self.obs(s), sum(terms.values()), done, info

    def obs(self, s: State) -> jax.Array:
        d, cube = s.data, self.cubes[TARGET[s.instr]]
        return jnp.concatenate(
            [
                2 * (d.qpos[:6] - self.lo) / (self.hi - self.lo) - 1,
                s.prev,
                d.site_xpos[self.site],
                d.xpos[cube],
                d.xquat[cube],
                d.xpos[self.cubes[1 - TARGET[s.instr]]],
                d.xpos[self.bins[BIN[s.instr]]],
                jax.nn.one_hot(s.instr, len(INSTRUCTIONS)),
            ]
        ).astype(jnp.float32)

    def _success(self, d: mjx.Data, instr) -> jax.Array:
        cube, bin_ = d.xpos[self.cubes[TARGET[instr]]], d.xpos[self.bins[BIN[instr]]]
        in_bin = jnp.all(jnp.abs(cube[:2] - bin_[:2]) < BIN_HALF) & (cube[2] < 0.04)
        return in_bin & (d.sensordata[TARGET[instr]] == 0)

    def _failed(self, d: mjx.Data) -> jax.Array:
        x, y, z = d.xpos[self.cubes].T
        off = (x <= TABLE_X[0]) | (x >= TABLE_X[1]) | (y <= TABLE_Y[0]) | (y >= TABLE_Y[1]) | (z <= -0.01)
        return (jnp.abs(d.qvel[:6]).max() > 2 * JOINT_SPEED) | off.any()
