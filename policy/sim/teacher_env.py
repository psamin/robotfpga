"""SGS teacher env on CPU MuJoCo for pick and place: the reference the GPU env must match.

Episodes start from a reset-buffer state. The teacher sees privileged state, moves joint targets by
bounded deltas each 30 Hz tick, and gets the shared manipulation reward of arXiv:2610.12465v1 Eq. 5
with the App. A.3.2 weights. Success is scene.success (bottle upright on the plate, released);
failure is the bottle leaving the table or a joint moving faster than twice the servo's speed.

Observation (OBS_DIM floats): normalized joints (6), previous action (6), fingertip site (3), bottle
position (3) and quaternion (4), plate x, y and flat radius (3), bottle height and body radius (2).
"""

from dataclasses import replace

import mujoco
import numpy as np

from policy.sim.resets import TABLE, ResetState, Sampler, apply
from policy.sim.scene import SUBSTEPS, success

OBS_DIM = 27
ACTION_SCALE = np.array([0.05] * 5 + [0.1])  # rad per 30 Hz tick at |a| = 1
W_SUCCESS, W_FAILURE, W_ACTION, W_RATE, W_JOINT_VEL = 100.0, 1.0, 1e-5, 1e-4, 1e-4
PENALTY_CLAMP = 1e4  # C(x) = min(x, 1e4)
JOINT_SPEED = 4.7  # rad/s, STS3215 no-load speed at 7.4 V; the MJCF has no velocity limits
MAX_STEPS = 360  # 12 s at 30 Hz; the scripted expert needs about 200 steps from rest


class TeacherEnv:
    def __init__(self, sampler: Sampler, max_steps: int = MAX_STEPS):
        self.sampler, self.max_steps = sampler, max_steps
        self.target, self.prev = np.zeros(6), np.zeros(6)

    def reset(self, s: ResetState) -> np.ndarray:
        self.m = self.sampler.model(s.variant)[0]
        self.d = mujoco.MjData(self.m)
        self.p = replace(self.sampler.variants[s.variant], bottle_xy=s.bottle_xy, plate_xy=s.plate_xy)
        apply(self.m, self.d, s)
        self.lo, self.hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        self.target, self.prev, self.t = np.clip(self.d.ctrl[:6], self.lo, self.hi), np.zeros(6), 0
        return self.obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        m, d = self.m, self.d
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.target = np.clip(self.target + ACTION_SCALE * a, self.lo, self.hi)
        d.ctrl[:6] = self.target
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        self.t += 1
        ok, failed = success(m, d, self.p), self._failed()
        clamp = lambda x: min(float(x), PENALTY_CLAMP)  # noqa: E731
        terms = {
            "success": W_SUCCESS * ok,
            "failure": -W_FAILURE * failed,
            "action": -W_ACTION * clamp(a @ a),
            "action_rate": -W_RATE * clamp((a - self.prev) @ (a - self.prev)),
            "joint_vel": -W_JOINT_VEL * clamp(d.qvel[:5] @ d.qvel[:5]),
        }
        self.prev = a
        done = ok or failed or self.t >= self.max_steps
        info = {"success": ok, "failure": failed, "t": self.t, "terms": terms}
        return self.obs(), float(sum(terms.values())), done, info

    def obs(self) -> np.ndarray:
        m, d, p = self.m, self.d, self.p
        bottle = d.body("bottle")
        return np.concatenate(
            [
                2 * (d.qpos[:6] - self.lo) / (self.hi - self.lo) - 1,
                self.prev,
                d.site_xpos[m.site("gripperframe").id],
                bottle.xpos,
                bottle.xquat,
                d.body("plate").xpos[:2],
                [p.plate_inner_radius, p.bottle_height, p.body_radius],
            ]
        ).astype(np.float32)

    def _failed(self) -> bool:
        if np.abs(self.d.qvel[:6]).max() > 2 * JOINT_SPEED:
            return True
        x, y, z = self.d.body("bottle").xpos
        return not (TABLE[0][0] < x < TABLE[0][1] and TABLE[1][0] < y < TABLE[1][1] and z > -0.01)
