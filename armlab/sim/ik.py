"""Damped least-squares IK for the SO-101 gripper site.

The arm has 5 joints before the gripper. A top-down grasp fixes position (3) and the approach
axis pointing down (2, but only 1 is independent in the arm's vertical plane), leaving wrist
roll free to set the gripper yaw. Solved with damped least squares on MuJoCo site Jacobians.
"""

import mujoco
import numpy as np

ARM = 5  # shoulder_pan .. wrist_roll
DOWN = np.array([0.0, 0.0, -1.0])
SEEDS = (
    np.array([0.0, 0.0, 0.5, 1.2, 0.0]),
    np.array([0.0, 0.6, -0.6, 1.5, 0.0]),
    np.array([0.0, -0.5, 0.8, 1.0, 0.0]),
)


class TopDownIK:
    def __init__(self, model: mujoco.MjModel, site: str = "gripperframe"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        self.lo = model.jnt_range[:ARM, 0]
        self.hi = model.jnt_range[:ARM, 1]
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:ARM] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def _solve_from(self, q0, pos, yaw, iters, damping, rot_w):
        q = np.array(q0[:ARM], dtype=float)
        for _ in range(iters):
            p, r = self._fk(q)
            e_rot = np.cross(r[:, 0], DOWN)  # approach axis (site x) toward -z
            if yaw is not None:  # site y axis along (cos yaw, sin yaw): rotate about z only
                e_rot[2] += np.cross(r[:, 1], [np.cos(yaw), np.sin(yaw), 0.0])[2]
            e = np.concatenate([pos - p, rot_w * e_rot])
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp[:, :ARM], rot_w * self._jr[:, :ARM]])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e), self.lo, self.hi)
        p, r = self._fk(q)
        return q, float(np.linalg.norm(pos - p)), float(np.linalg.norm(np.cross(r[:, 0], DOWN)))

    def solve(
        self,
        pos,
        yaw: float | None = None,
        q0=None,
        iters: int = 150,
        damping: float = 1e-4,
        rot_w: float = 0.3,
    ) -> tuple[np.ndarray, float]:
        """Returns (5 arm joint angles, position error in m). Tries q0 first, then fixed seeds."""
        pos = np.asarray(pos, dtype=float)
        best = None
        for seed in ([np.asarray(q0)[:ARM]] if q0 is not None else []) + list(SEEDS):
            q, ep, er = self._solve_from(seed, pos, yaw, iters, damping, rot_w)
            if best is None or ep + 0.02 * er < best[1] + 0.02 * best[2]:
                best = (q, ep, er)
            if ep < 1e-3 and er < 1e-2:
                break
        return best[0], best[1]

    def solve_symmetric(self, pos, yaw: float, period: float = np.pi / 2, **kw) -> tuple[np.ndarray, float]:
        """Like solve, for a grasp that repeats every `period` radians of yaw (a cube: pi/2).
        Tries each equivalent yaw and keeps the most accurate, then the smallest wrist roll."""
        best = None
        for k in range(-4, 5):
            y = yaw + k * period
            if abs(y) > np.pi:
                continue
            q, err = self.solve(pos, yaw=y, **kw)
            key = (round(err, 3), abs(q[4]))
            if best is None or key < best[0]:
                best = (key, q, err)
        return best[1], best[2]

    def site_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._fk(np.asarray(q)[:ARM])
