"""Scripted pick-and-place expert for BlockSortEnv.

Reactive: every 30 Hz tick it reads the current (privileged) cube and bin positions and the
current instruction, picks a phase, and servos a commanded gripper pose toward that phase's goal
at a capped Cartesian speed, solved to joint targets with top-down IK. Because it re-plans from
the current state, it recovers from nudges and dropped cubes and follows an instruction swapped
mid-episode. That is what noise-injected (DART-style) data collection and Stage C need.
"""

from enum import Enum

import numpy as np

from armlab.sim.env import INSTRUCTIONS, BlockSortEnv
from armlab.sim.ik import TopDownIK

HOVER_Z = 0.065  # carry / approach height of the gripper site (m)
GRASP_Z = 0.012  # site height when grasping (cube center is 0.0125)
OPEN, CLOSED = 0.6, -0.17  # gripper joint (rad): ~60 mm gap / squeeze
SPEED = 0.22  # m/s cap on the commanded site motion
DT = 1 / 30


class Phase(Enum):
    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    CARRY = 4
    RELEASE = 5
    RETREAT = 6
    DONE = 7


class Expert:
    def __init__(self, env: BlockSortEnv, speed: float = SPEED):
        self.env = env
        self.ik = TopDownIK(env.m)
        self.speed = speed
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.APPROACH
        self.timer = 0
        self.cmd = None  # commanded site position
        self.q = None  # last arm joint solution (warm start)
        self.grip = OPEN
        self.yaw = 0.0

    # ---- helpers ----
    def _targets(self):
        color, side = INSTRUCTIONS[self.env.instr]
        return self.env.cube_pos(color), self.env.bin_pos(side), color

    def _cube_yaw(self, color: str) -> float:
        r = self.env.d.body(f"cube_{color}").xmat.reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def _site(self) -> np.ndarray:
        return self.env.d.site("gripperframe").xpos.copy()

    def _held(self, color: str) -> bool:
        cube = self.env.cube_pos(color)
        return (
            self.env.touching_robot(color) and cube[2] > 0.02 and np.linalg.norm(cube - self._site()) < 0.04
        )

    def _servo(self, goal: np.ndarray, speed: float | None = None) -> bool:
        step = (speed or self.speed) * DT
        delta = goal - self.cmd
        dist = np.linalg.norm(delta)
        self.cmd = goal.copy() if dist <= step else self.cmd + delta * (step / dist)
        return dist <= step

    # ---- main ----
    def act(self) -> np.ndarray:
        """Returns raw joint targets (radians, 6)."""
        env = self.env
        if self.cmd is None:
            self.cmd = self._site()
            self.q = env.d.qpos[:5].copy()
        cube, bin_, color = self._targets()
        p = self.phase

        # Reactive checks: a lost cube sends us back to approach; a finished cube ends the episode.
        if p in (Phase.LIFT, Phase.CARRY) and not self._held(color) and self.timer > 6:
            p = Phase.APPROACH
        if p in (Phase.APPROACH, Phase.DESCEND) and env.in_bin(color, INSTRUCTIONS[env.instr][1]):
            p = Phase.RETREAT

        if p == Phase.APPROACH:
            self.grip = OPEN
            self.yaw = self._cube_yaw(color)
            above = np.array([cube[0], cube[1], HOVER_Z])
            if self.cmd[2] < HOVER_Z - 0.01 and np.linalg.norm(self.cmd[:2] - cube[:2]) > 0.01:
                arrived = False  # rise first so we don't sweep through the cubes
                self._servo(np.array([*self.cmd[:2], HOVER_Z]))
            else:
                arrived = self._servo(above)
            if arrived and np.linalg.norm(self._site()[:2] - cube[:2]) < 0.006:
                p = Phase.DESCEND
        elif p == Phase.DESCEND:
            self.yaw = self._cube_yaw(color)
            if self._servo(np.array([cube[0], cube[1], GRASP_Z]), speed=0.6 * self.speed):
                if abs(self._site()[2] - GRASP_Z) < 0.004:
                    p = Phase.GRASP
        elif p == Phase.GRASP:
            self.grip = CLOSED
            if self.timer >= 8:
                p = Phase.LIFT
        elif p == Phase.LIFT:
            if self._servo(np.array([*self.cmd[:2], HOVER_Z])):
                p = Phase.CARRY
        elif p == Phase.CARRY:
            if (
                self._servo(np.array([bin_[0], bin_[1], HOVER_Z]))
                and np.linalg.norm(self._site()[:2] - bin_[:2]) < 0.01
            ):
                p = Phase.RELEASE
        elif p == Phase.RELEASE:
            self.grip = OPEN
            if self.timer >= 8:
                p = Phase.RETREAT
        elif p == Phase.RETREAT:
            self.grip = OPEN
            if self._servo(np.array([*self.cmd[:2], HOVER_Z + 0.01])):
                p = Phase.DONE

        if p != self.phase:
            self.phase, self.timer = p, 0
        else:
            self.timer += 1

        self.q, _ = self.ik.solve_symmetric(self.cmd, self.yaw, q0=self.q, iters=30)
        return np.concatenate([self.q, [self.grip]])


def run_episode(env: BlockSortEnv, seed: int, instr: int, max_steps: int | None = None, render: bool = False):
    """Runs the expert. Returns (success, steps, failure_reason, frames or None)."""
    env.reset(seed=seed, instr=instr)
    expert = Expert(env)
    frames = [] if render else None
    for t in range(max_steps or env.max_steps):
        q = expert.act()
        env.d.ctrl[:] = q
        _, done, info = env.step(env.normalize(q)) if render else _step_norender(env, q)
        if render:
            frames.append(env.render())
        if info["success"] and expert.phase == Phase.DONE:
            return True, t + 1, "success", frames
        if done and not info["success"]:
            break
    return env.success(), env.t, env.failure_reason(), frames


def _step_norender(env: BlockSortEnv, q: np.ndarray):
    import mujoco

    from armlab.sim.env import SUBSTEPS

    env.d.ctrl[:] = q
    for _ in range(SUBSTEPS):
        mujoco.mj_step(env.m, env.d)
    env.t += 1
    ok = env.success()
    return None, ok or env.t >= env.max_steps, {"success": ok}
