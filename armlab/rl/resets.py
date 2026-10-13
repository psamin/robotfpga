"""Reset families for SGS teacher training: arXiv:2610.12465v1 App. A.2, adapted to cubes and bins.

rest          BlockSortEnv.reset: arm at REST_Q, both cubes on the table.
reaching      Gripper aligned to the target cube anywhere in the spawn region, on the table or in
              the air, with a random aperture: the cube may be held or may drop.
stable_grasp  Target cube held at carry height somewhere on the route from the table to its bin.
near_goal     Target cube held on the drop path into its bin: path fraction 0 rests on the bin
              floor, 1 just clears the walls, up to 1.1 slightly above.

Top-down IK cannot reach much above 6 cm over the bins, so "higher than Near-Goal" (the paper's
Stable Grasp) is the 5-6 cm carry band. A sample is a plain state that apply() restores exactly.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from armlab.sim.env import CUBE_HALF, CUBE_X, CUBE_Y, INSTRUCTIONS, SUBSTEPS, BlockSortEnv
from armlab.sim.expert import CLOSED, HELD_GRIP, OPEN
from armlab.sim.ik import TopDownIK

FAMILIES = ("rest", "reaching", "stable_grasp", "near_goal")
FLOOR_Z = 0.004 + CUBE_HALF  # cube center resting on a bin floor
CARRY_Z = (0.05, 0.06)  # cube center: bottom clears the 3.5 cm walls; IK limit above the bins
REACH_Z = (CUBE_HALF, 0.06)
NEAR_GOAL_FRAC = (0.0, 1.1)
BIN_XY = 0.015  # cube center offset from the bin center (inner half-width is 4.5 cm)
MAX_IK_ERR = 0.003  # m
MAX_PEN = 0.001  # m of penetration allowed when the state is placed
SETTLED_QVEL = 0.1  # max |qvel| after settling
TABLE_X, TABLE_Y = (-0.2, 0.5), (-0.45, 0.45)


@dataclass(frozen=True)
class ResetState:
    family: str
    instr: int
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray


def apply(env: BlockSortEnv, state: ResetState) -> None:
    """Restore a sampled state; the next env.step starts the episode."""
    mujoco.mj_resetData(env.m, env.d)
    env.d.qpos[:], env.d.qvel[:], env.d.ctrl[:] = state.qpos, state.qvel, state.ctrl
    env.instr, env.t = state.instr, 0
    mujoco.mj_forward(env.m, env.d)


def sample(env: BlockSortEnv, ik: TopDownIK, family: str, rng: np.random.Generator) -> ResetState | None:
    """One candidate state, or None if it failed IK, collision or settling checks."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    instr = int(rng.integers(len(INSTRUCTIONS)))
    env.reset(int(rng.integers(2**31)), instr, render=False)
    color, side = INSTRUCTIONS[instr]
    held = family in ("stable_grasp", "near_goal")
    if family == "reaching":
        pos = [*rng.uniform([CUBE_X[0], CUBE_Y[0]], [CUBE_X[1], CUBE_Y[1]]), rng.uniform(*REACH_Z)]
        grip = rng.uniform(CLOSED, OPEN)
    elif family == "stable_grasp":
        start = rng.uniform([CUBE_X[0], CUBE_Y[0]], [CUBE_X[1], CUBE_Y[1]])
        xy = start + rng.uniform() * (env.bin_pos(side)[:2] - start)
        pos, grip = [*xy, rng.uniform(*CARRY_Z)], rng.uniform(CLOSED, 0.0)
    elif family == "near_goal":
        xy = env.bin_pos(side)[:2] + rng.uniform(-BIN_XY, BIN_XY, 2)
        pos = [*xy, FLOOR_Z + rng.uniform(*NEAR_GOAL_FRAC) * (CARRY_Z[0] - FLOOR_Z)]
        grip = rng.uniform(CLOSED, 0.0)
    if family != "rest":
        yaw = rng.uniform(-np.pi, np.pi)
        if not _place_in_gripper(env, ik, color, np.array(pos), yaw, grip):
            return None
    if not _valid(env) or (held and not (env.touching_robot(color) and env.d.qpos[5] < HELD_GRIP)):
        return None
    d = env.d
    return ResetState(family, instr, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())


def _place_in_gripper(env, ik, color, pos, yaw, grip) -> bool:
    q, err = ik.solve_symmetric(pos, yaw)
    if err > MAX_IK_ERR:
        return False
    m, d = env.m, env.d
    adr = m.jnt_qposadr[m.joint(f"cube_{color}").id]
    d.qpos[:5], d.qpos[5] = q, OPEN
    d.qpos[adr : adr + 7] = [*pos, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[:] = 0
    d.ctrl[:5], d.ctrl[5] = q, grip
    mujoco.mj_forward(m, d)
    if any(d.contact[i].dist < -MAX_PEN for i in range(d.ncon)):
        return False
    gravity = m.opt.gravity.copy()
    m.opt.gravity[:] = 0  # close the gripper before the cube can fall
    try:
        _ticks(env, 20)
    finally:
        m.opt.gravity[:] = gravity
    _ticks(env, 30)
    return True


def _valid(env) -> bool:
    for c in ("red", "blue"):
        x, y, z = env.cube_pos(c)
        if not (TABLE_X[0] < x < TABLE_X[1] and TABLE_Y[0] < y < TABLE_Y[1] and z > 0):
            return False
    return np.abs(env.d.qvel).max() < SETTLED_QVEL and not env.success()


def _ticks(env, n: int) -> None:
    for _ in range(n * SUBSTEPS):
        mujoco.mj_step(env.m, env.d)
