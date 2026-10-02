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


HELD_GRIP = 0.2  # gripper angle below this with the cube in contact = holding it (0.09 on a cube)
EMPTY_GRIP = 0.0  # below this with no contact = closed on nothing (-0.17 fully closed)


class Expert:
    """Stateless: the phase is derived from the world state every tick, and the commanded gripper
    pose steps from the *measured* pose toward the phase goal. So it can label any state, including
    ones a learner drove into (DAgger [Ross2011-DAgger]), not only states it produced itself."""

    def __init__(self, env: BlockSortEnv, speed: float = SPEED):
        self.env = env
        self.ik = TopDownIK(env.m)
        self.speed = speed
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.APPROACH
        self.q = None  # warm start for IK
        self.cmd = None  # commanded site position; integrates through servo tracking lag
        self.grip = OPEN

    # ---- helpers ----
    def _targets(self):
        color, side = INSTRUCTIONS[self.env.instr]
        return self.env.cube_pos(color), self.env.bin_pos(side), color

    def _cube_yaw(self, color: str) -> float:
        r = self.env.d.body(f"cube_{color}").xmat.reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def _site(self) -> np.ndarray:
        return self.env.d.site("gripperframe").xpos.copy()

    def _phase(self, site, cube, bin_, color) -> Phase:
        env = self.env
        g = float(env.d.qpos[5])
        touching = env.touching_robot(color)
        if np.all(np.abs(cube[:2] - bin_[:2]) < 0.045) and not (touching and g < HELD_GRIP):
            return Phase.DONE if site[2] > HOVER_Z - 0.005 and g > OPEN - 0.1 else Phase.RETREAT
        if touching and g < HELD_GRIP:
            if np.linalg.norm(site[:2] - bin_[:2]) < 0.01 and site[2] > HOVER_Z - 0.015:
                return Phase.RELEASE
            return Phase.LIFT if site[2] < HOVER_Z - 0.01 else Phase.CARRY
        # Hysteresis from the world state alone: start descending when aligned within 6 mm, but
        # once below hover height keep descending (and correcting) while within 20 mm, so tracking
        # noise doesn't bounce the arm back up. A gripper closed on nothing reopens via APPROACH.
        dxy = np.linalg.norm(site[:2] - cube[:2])
        missed = g < EMPTY_GRIP and not touching
        if dxy < 0.012 and site[2] < GRASP_Z + 0.006 and not missed:
            return Phase.GRASP
        if not missed and g > HELD_GRIP and (dxy < 0.012 or (dxy < 0.02 and site[2] < HOVER_Z - 0.003)):
            return Phase.DESCEND
        return Phase.APPROACH

    # ---- main ----
    def act(self) -> np.ndarray:
        """Returns raw joint targets (radians, 6) for the current world state."""
        env = self.env
        site = self._site()
        cube, bin_, color = self._targets()
        if self.q is None:
            self.q = env.d.qpos[:5].copy()
        p = self._phase(site, cube, bin_, color)
        yaw, speed = self._cube_yaw(color), self.speed
        if p == Phase.APPROACH:
            self.grip = OPEN
            far = np.linalg.norm(site[:2] - cube[:2]) > 0.02
            goal = (
                np.array([*site[:2], HOVER_Z]) if site[2] < HOVER_Z - 0.01 and far else [*cube[:2], HOVER_Z]
            )
        elif p == Phase.DESCEND:
            self.grip, goal, speed = OPEN, [cube[0], cube[1], GRASP_Z], 0.6 * self.speed
        elif p == Phase.GRASP:
            self.grip, goal = CLOSED, [cube[0], cube[1], GRASP_Z]
        elif p == Phase.LIFT:
            self.grip, goal = CLOSED, [*site[:2], HOVER_Z]
        elif p == Phase.CARRY:
            self.grip, goal = CLOSED, [bin_[0], bin_[1], HOVER_Z]
        elif p == Phase.RELEASE:
            self.grip, goal = OPEN, site
        else:  # RETREAT / DONE
            self.grip, goal = OPEN, [*site[:2], HOVER_Z + 0.005]
        if p in (Phase.LIFT, Phase.CARRY, Phase.RELEASE, Phase.RETREAT, Phase.DONE):
            yaw = None  # keep the current wrist roll while carrying
        self.phase = p
        # Step from the last command (it absorbs servo lag) unless the arm is far from it, which
        # means something else moved it (a learner, a perturbation): then re-anchor on the arm.
        base = self.cmd if self.cmd is not None and np.linalg.norm(self.cmd - site) < 0.02 else site
        delta = np.asarray(goal, float) - base
        dist = np.linalg.norm(delta)
        step = speed * DT
        self.cmd = base + (delta if dist <= step else delta * (step / dist))
        if yaw is None:
            self.q, _ = self.ik.solve(self.cmd, yaw=None, q0=self.q, iters=30)
            self.q[4] = env.d.qpos[4]
        else:
            # a cube grasp repeats every 90 deg: use the copy closest to the gripper's current yaw
            r = env.d.site_xmat[env.m.site("gripperframe").id].reshape(3, 3)
            now = np.arctan2(r[1, 1], r[0, 1])
            yaw = now + (yaw - now + np.pi / 4) % (np.pi / 2) - np.pi / 4
            self.q, err = self.ik.solve(self.cmd, yaw=yaw, q0=self.q, iters=30)
            if err > 0.003:
                self.q, _ = self.ik.solve_symmetric(self.cmd, yaw, q0=self.q, iters=30)
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


def expert_chunk(env: BlockSortEnv, expert: Expert, n: int = 8) -> np.ndarray:
    """The expert's next n normalized actions from the current state: simulate the expert ahead
    on a full copy of MjData (mj_copyData keeps derived quantities and solver warm-start exactly
    as they are), so the real episode is untouched. Used as the action-chunk label."""
    import copy

    import mujoco

    from armlab.sim.env import SUBSTEPS

    real = env.d
    scratch = getattr(env, "_scratch", None)
    if scratch is None:
        scratch = env._scratch = mujoco.MjData(env.m)
    mujoco.mj_copyData(scratch, env.m, real)
    saved_instr, saved_t = env.instr, env.t
    look = copy.copy(expert)
    out = np.empty((n, 6), np.float32)
    env.d = scratch
    try:
        for i in range(n):
            q = look.act()
            out[i] = env.normalize(q)
            scratch.ctrl[:] = q
            for _ in range(SUBSTEPS):
                mujoco.mj_step(env.m, scratch)
    finally:
        env.d, env.instr, env.t = real, saved_instr, saved_t
    return out
