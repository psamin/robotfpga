"""SGS teacher environment on CPU MuJoCo: the reference that the MJX env must match.

Episodes start from a reset-buffer state. The teacher sees privileged state, moves joint targets
by bounded deltas, and gets the shared manipulation reward of arXiv:2610.12465v1 Eq. 5 with the
App. A.3.2 weights: a large success term plus small action, action-rate and joint-speed penalties.

Observation (OBS_DIM floats): normalized joints (6), previous action (6), gripper site (3),
target cube position (3) and quaternion (4), other cube position (3), target bin position (3),
instruction one-hot (4).
"""

import mujoco
import numpy as np

from armlab.rl.resets import TABLE_X, TABLE_Y, ResetState, apply
from armlab.sim.env import INSTRUCTIONS, SUBSTEPS, BlockSortEnv

OBS_DIM = 32
ACTION_SCALE = np.array([0.05] * 5 + [0.1])  # rad per 30 Hz tick at |a| = 1
W_SUCCESS, W_FAILURE, W_ACTION, W_RATE, W_JOINT_VEL = 100.0, 1.0, 1e-5, 1e-4, 1e-4
PENALTY_CLAMP = 1e4  # C(x) = min(x, 1e4)
# The MJCF has no velocity limits; the paper fails at 2x the configured limit. Local choice:
# STS3215 no-load speed at 7.4 V (0.222 s / 60 deg).
JOINT_SPEED = 4.7  # rad/s
MAX_STEPS = 300  # 10 s at 30 Hz; the expert needs ~70 steps from rest


class TeacherEnv:
    def __init__(self, max_steps: int = MAX_STEPS):
        self.env = BlockSortEnv(jitter=None)
        self.max_steps = max_steps
        self.target = np.zeros(6)
        self.prev = np.zeros(6)

    def reset(self, state: ResetState) -> np.ndarray:
        apply(self.env, state)
        self.target = self.env.d.ctrl.copy()
        self.prev = np.zeros(6)
        return self.obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        env, d = self.env, self.env.d
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.target = np.clip(self.target + ACTION_SCALE * a, env.lo, env.hi)
        d.ctrl[:] = self.target
        for _ in range(SUBSTEPS):
            mujoco.mj_step(env.m, d)
        env.t += 1
        success, failure = env.success(), self._failed()
        terms = {
            "success": W_SUCCESS * success,
            "failure": -W_FAILURE * failure,
            "action": -W_ACTION * min(a @ a, PENALTY_CLAMP),
            "action_rate": -W_RATE * min((a - self.prev) @ (a - self.prev), PENALTY_CLAMP),
            "joint_vel": -W_JOINT_VEL * min(d.qvel[:5] @ d.qvel[:5], PENALTY_CLAMP),
        }
        self.prev = a
        done = success or failure or env.t >= self.max_steps
        info = {"success": success, "failure": failure, "t": env.t, "terms": terms}
        return self.obs(), float(sum(terms.values())), done, info

    def obs(self) -> np.ndarray:
        env, d = self.env, self.env.d
        color, side = INSTRUCTIONS[env.instr]
        other = "blue" if color == "red" else "red"
        cube = d.body(f"cube_{color}")
        return np.concatenate(
            [
                env.normalize(d.qpos[:6]),
                self.prev,
                d.site("gripperframe").xpos,
                cube.xpos,
                cube.xquat,
                env.cube_pos(other),
                env.bin_pos(side),
                np.eye(len(INSTRUCTIONS))[env.instr],
            ]
        ).astype(np.float32)

    def _failed(self) -> bool:
        if np.abs(self.env.d.qvel[:6]).max() > 2 * JOINT_SPEED:
            return True
        for c in ("red", "blue"):
            x, y, z = self.env.cube_pos(c)
            if not (TABLE_X[0] < x < TABLE_X[1] and TABLE_Y[0] < y < TABLE_Y[1] and z > -0.01):
                return True
        return False
