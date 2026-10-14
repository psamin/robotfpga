"""Pick-and-place teacher env on MJX (MuJoCo Warp): the GPU twin of policy/sim/teacher_env.py.

One instance per size variant: Warp cannot stack differently sized models into one batch, so the
training env runs one compiled sub-batch per variant. reset() and step() act on one world and are
pure, so callers jit(vmap(.)) over the variant's sub-batch. The plate is a mocap body placed per
world. "Bottle touching the robot" comes from a contact sensor (scene.build) instead of Warp's
private contact pool, which is shared across the batch.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from policy.sim.resets import TABLE
from policy.sim.scene import SUBSTEPS, SceneParams, build
from policy.sim.teacher_env import (
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

CONTACTS_PER_WORLD, NJMAX = 64, 320  # Warp buffers: contacts (total = per world x batch), rows per world
QUIET = 512 | 1024  # silence Warp's solver/line-search iteration notices; overflow warnings stay on


class State(NamedTuple):
    data: mjx.Data
    target: jax.Array
    prev: jax.Array
    t: jax.Array


class TeacherMJX:
    def __init__(self, p: SceneParams, batch: int, impl: str = "warp", max_steps: int = MAX_STEPS):
        m = build(p)
        mx = mjx.put_model(m, impl=impl)
        if impl == "warp":
            opt = mx.opt._impl.replace(warn_overflow=mx.opt._impl.warn_overflow & ~QUIET)
            mx = mx.replace(opt=mx.opt.replace(_impl=opt))
        self.mx, self.max_steps = mx, max_steps
        sizes = {"naconmax": CONTACTS_PER_WORLD * batch, "njmax": NJMAX} if impl == "warp" else {}
        self.d0 = mjx.make_data(m, impl=impl, **sizes)
        self.lo, self.hi = jnp.array(m.actuator_ctrlrange[:, 0]), jnp.array(m.actuator_ctrlrange[:, 1])
        self.site, self.bottle, self.plate = (
            m.site("gripperframe").id,
            m.body("bottle").id,
            m.body("plate").id,
        )
        self.dims = jnp.array([p.plate_inner_radius, p.bottle_height, p.body_radius])

    def reset(self, qpos, qvel, ctrl, plate_xy) -> tuple[State, jax.Array]:
        mocap = jnp.concatenate([jnp.asarray(plate_xy, self.d0.mocap_pos.dtype), jnp.zeros(1)])[None]
        d = mjx.forward(self.mx, self.d0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl, mocap_pos=mocap))
        target = jnp.clip(jnp.asarray(ctrl)[:6], self.lo, self.hi)
        s = State(d, target, jnp.zeros(6, target.dtype), jnp.int32(0))
        return s, self.obs(s)

    def step(self, s: State, action):
        a = jnp.clip(jnp.asarray(action, s.target.dtype), -1, 1)
        target = jnp.clip(s.target + jnp.asarray(ACTION_SCALE, a.dtype) * a, self.lo, self.hi)
        d = jax.lax.fori_loop(0, SUBSTEPS, lambda _, d: mjx.step(self.mx, d), s.data.replace(ctrl=target))
        ok, failed = self._success(d), self._failed(d)
        clamp = lambda x: jnp.minimum(x, PENALTY_CLAMP)  # noqa: E731
        terms = {
            "success": W_SUCCESS * ok,
            "failure": -W_FAILURE * failed,
            "action": -W_ACTION * clamp(a @ a),
            "action_rate": -W_RATE * clamp((a - s.prev) @ (a - s.prev)),
            "joint_vel": -W_JOINT_VEL * clamp(d.qvel[:5] @ d.qvel[:5]),
        }
        s = State(d, target, a, s.t + 1)
        done = ok | failed | (s.t >= self.max_steps)
        return s, self.obs(s), sum(terms.values()), done, {"success": ok, "failure": failed, "terms": terms}

    def obs(self, s: State) -> jax.Array:
        d = s.data
        return jnp.concatenate(
            [
                2 * (d.qpos[:6] - self.lo) / (self.hi - self.lo) - 1,
                s.prev,
                d.site_xpos[self.site],
                d.xpos[self.bottle],
                d.xquat[self.bottle],
                d.xpos[self.plate][:2],
                self.dims,
            ]
        ).astype(jnp.float32)

    def _success(self, d: mjx.Data) -> jax.Array:
        pos, q = d.xpos[self.bottle], d.xquat[self.bottle]
        upright = 1 - 2 * (q[1] ** 2 + q[2] ** 2) > 0.95
        inside = jnp.linalg.norm(pos[:2] - d.xpos[self.plate][:2]) < self.dims[0]
        return upright & inside & (pos[2] < 0.012) & (d.sensordata[0] == 0)

    def _failed(self, d: mjx.Data) -> jax.Array:
        x, y, z = d.xpos[self.bottle]
        off = (x <= TABLE[0][0]) | (x >= TABLE[0][1]) | (y <= TABLE[1][0]) | (y >= TABLE[1][1]) | (z <= -0.01)
        return (jnp.abs(d.qvel[:6]).max() > 2 * JOINT_SPEED) | off
