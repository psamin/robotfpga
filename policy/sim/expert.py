"""Scripted pick-and-place expert: a sanity check that the randomized scene is solvable in physics.

It grasps the way the demos do, at an angle rather than top-down (docs/smolvla-post-training.md):
unfold from the demos' start pose in joint space, approach the bottle along a tilted direction in the
vertical plane through the base, close the gripper until it stops, lift, carry, lower onto the plate,
release and retreat. The commanded fingertip point moves at a capped speed and is solved to joint
targets every 30 Hz tick by IK with the jaws closing horizontally.
"""

import mujoco
import numpy as np

from policy.sim.ik import TopDownIK
from policy.sim.scene import DEMO_START, SUBSTEPS, SceneParams, build, success
from policy.sim.so101 import gripper_to_sim, lerobot_to_sim

TILTS = (50, 30, 70, 15)  # deg from straight down, tried in order; the demos grasp at 33 to 77 deg
BACK_OFFS = (0.07, 0.05, 0.035)  # m, pre-grasp distance back along the approach
PAD = 0.012  # m: the jaw pads sit this far behind the fingertip site along the approach
# Only one jaw moves. The fixed jaw's inner face sits 2.0 cm from the site along its z axis (the jaw
# axis), toward -z, so the bottle's axis goes FIXED_FACE + radius + JAW_MARGIN along +z from the site.
FIXED_FACE, JAW_MARGIN = -0.020, 0.002
CARRY_Z, PLACE_CLEARANCE = 0.12, 0.012
SPEED, DT = 0.15, 1 / 30
UNFOLD_TICKS, PHASE_LIMIT = 45, 150
OPEN, CLOSED = float(gripper_to_sim(70.0)), float(gripper_to_sim(0.0))
SEEDS = ((-0.5, 0.5, 1.0, 0.0), (0.0, 0.0, 1.5, 0.0), (0.3, -0.7, 1.0, 0.0), (-1.0, 1.2, 0.8, 0.0))


def approach_vec(xy, tilt_deg: float) -> np.ndarray:
    """Unit approach direction tilted tilt_deg from straight down, in the vertical plane through the base."""
    t, radial = np.radians(tilt_deg), np.array([*xy, 0.0]) / np.linalg.norm(xy)
    return np.cos(t) * np.array([0, 0, -1.0]) + np.sin(t) * radial


def solve_best(ik: TopDownIK, target, approach) -> tuple[np.ndarray, float]:
    pan, best = np.arctan2(target[1], target[0]), None
    for s in SEEDS:
        q, err = ik.solve_oriented(target, approach, [pan, *s], iters=300)
        if best is None or err < best[1]:
            best = (q, err)
    return best


def choose_approach(ik: TopDownIK, target) -> tuple[np.ndarray, float]:
    """First (approach, back-off) whose grasp point and back-off point are both reachable."""
    for tilt in TILTS:
        a = approach_vec(target[:2], tilt)
        if solve_best(ik, target + PAD * a, a)[1] < 0.002:
            for back in BACK_OFFS:
                if solve_best(ik, target + (PAD - back) * a, a)[1] < 0.002:
                    return a, back
    return approach_vec(target[:2], 30), BACK_OFFS[-1]


def grasp_site(ik: TopDownIK, point, approach, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Fingertip-site target, joints and IK error that put the bottle's axis through `point`, between
    the jaw pads and just off the fixed jaw."""
    target = np.asarray(point, float) + PAD * approach
    q, _ = solve_best(ik, target, approach)
    jaw = ik.site_pose(q)[1][:, 2]
    target = target - (FIXED_FACE + radius + JAW_MARGIN) * jaw
    q, err = solve_best(ik, target, approach)
    return target, q, err


def grasp_height(p: SceneParams) -> float:
    """Height of the grasp point on the bottle: the upper body, like the demos' grasps."""
    return max(p.bottle_height - p.cap_height - 0.012, 0.02)


class Expert:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, p: SceneParams):
        self.d, self.ik, self.site = data, TopDownIK(model), model.site("gripperframe").id
        z = grasp_height(p)
        self.grasp = np.array([*p.bottle_xy, z])
        self.place = np.array([*p.plate_xy, 0.003 + z + PLACE_CLEARANCE])
        self.app_g, self.back_g = choose_approach(self.ik, self.grasp)
        self.app_p, _ = choose_approach(self.ik, self.place)
        self.g_site = grasp_site(self.ik, self.grasp, self.app_g, p.body_radius)[0]
        self.p_site = grasp_site(self.ik, self.place, self.app_p, p.body_radius)[0]
        self.q_start = data.qpos[:5].copy()
        ready = self.g_site - self.back_g * self.app_g + [0, 0, 0.06]
        self.q_ready = solve_best(self.ik, ready, self.app_g)[0]
        self.q, self.cmd = self.q_start.copy(), None
        self.phase, self.ticks = "unfold", 0

    def _next(self, phase):
        self.phase, self.ticks = phase, 0

    def act(self) -> np.ndarray:
        """Joint targets (radians, 6) for the current state."""
        self.ticks += 1
        if self.phase == "unfold":
            a = min(1.0, self.ticks / UNFOLD_TICKS)
            self.q = self.q_start + a * a * (3 - 2 * a) * (self.q_ready - self.q_start)
            if self.ticks >= UNFOLD_TICKS + 10:
                self._next("approach")
            return np.r_[self.q, OPEN]
        if self.ticks > PHASE_LIMIT:
            self._next("stuck")
        site, grip_vel = self.d.site_xpos[self.site].copy(), abs(self.d.qvel[5])
        g, pl = self.g_site, self.p_site
        settled = self.ticks > 45 or (self.ticks > 10 and grip_vel < 0.05)
        near = lambda goal, tol: np.linalg.norm(site - goal) < tol  # noqa: E731
        plan = {
            "approach": (g - self.back_g * self.app_g, self.app_g, OPEN),
            "advance": (g, self.app_g, OPEN),
            "close": (g, self.app_g, CLOSED),
            "lift": (np.r_[g[:2], CARRY_Z], self.app_g, CLOSED),
            "carry": (np.r_[pl[:2], CARRY_Z], self.app_p, CLOSED),
            "lower": (pl, self.app_p, CLOSED),
            "release": (pl, self.app_p, OPEN),
        }
        goal, approach, grip = plan.get(self.phase, (pl - 0.08 * self.app_p + [0, 0, 0.05], self.app_p, OPEN))
        done = {
            "approach": near(goal, 0.006),
            "advance": near(goal, 0.004),
            "close": settled,
            "lift": near(goal, 0.008),
            "carry": near(goal, 0.008),
            "lower": near(goal, 0.005),
            "release": settled,
        }
        order = ["approach", "advance", "close", "lift", "carry", "lower", "release", "retreat"]
        if done.get(self.phase):
            self._next(order[order.index(self.phase) + 1])
        base = site if self.cmd is None else self.cmd
        delta = goal - base
        self.cmd = base + delta * min(1.0, SPEED * DT / max(np.linalg.norm(delta), 1e-9))
        self.q, _ = self.ik.solve_oriented(self.cmd, approach, self.q)
        return np.r_[self.q, grip]


def run_episode(p: SceneParams, max_steps: int = 450) -> dict:
    """Build the scene, start from the demos' pose, run the expert. Returns success, steps and phase."""
    model = build(p)
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[:5, 0], model.jnt_range[:5, 1]
    data.qpos[:5] = np.clip(lerobot_to_sim(DEMO_START), lo + np.radians(2.0), hi - np.radians(2.0))
    data.qpos[5] = gripper_to_sim(DEMO_START[5])
    data.ctrl[:6] = data.qpos[:6]
    mujoco.mj_forward(model, data)
    for _ in range(30):
        mujoco.mj_step(model, data)
    expert = Expert(model, data, p)
    for t in range(1, max_steps + 1):
        data.ctrl[:6] = expert.act()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        if expert.phase == "retreat" and success(model, data, p):
            return {"success": True, "steps": t, "phase": expert.phase}
        if expert.phase == "stuck":
            break
    return {"success": success(model, data, p), "steps": t, "phase": expert.phase}
