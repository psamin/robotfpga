"""Reset families for SGS teacher training on pick and place (arXiv:2610.12465v1 App. A.2).

rest          the demos' start pose (+-3 deg per joint), bottle and plate placed.
reaching      gripper at an angled grasp pose around the bottle, pulled back up to the back-off
              distance, random aperture: the bottle may end up held or not.
stable_grasp  bottle held and lifted (bottom 6 to 12 cm up) somewhere on the route to the plate.
near_goal     held bottle above the plate: path fraction 0 touches the plate floor, 1 is 4 cm up, to 1.1.

Held states place the bottle between the jaw pads and close the gripper with gravity off before
settling. A state is rejected on penetration when placed, a dropped bottle, or an unsettled scene.
Each state stores its size variant and positions, so apply() rebuilds it exactly.
"""

from collections import Counter
from dataclasses import dataclass

import mujoco
import numpy as np

from policy.sim.expert import choose_approach, grasp_height, grasp_site
from policy.sim.families import FAMILIES
from policy.sim.ik import TopDownIK
from policy.sim.scene import DEMO_START, SUBSTEPS, SceneParams, build, place, sample_positions, success
from policy.sim.so101 import gripper_to_sim, lerobot_to_sim

REST_JITTER_DEG = 3.0
LIFT = (0.06, 0.12)  # m, bottle bottom while carried
NEAR_GOAL_TOP, NEAR_GOAL_FRAC = 0.04, (0.0, 1.1)
OPEN, CLOSED = float(gripper_to_sim(70.0)), float(gripper_to_sim(0.0))
MAX_PEN, SETTLED_QVEL = 0.001, 0.1
LIMIT_MARGIN = np.radians(2.0)  # the demos' start lift (-103 deg) is past the sim's -100 deg limit
TABLE = ((0.0, 0.6), (-0.4, 0.4))


@dataclass(frozen=True)
class ResetState:
    family: str
    variant: int
    bottle_xy: tuple[float, float]
    plate_xy: tuple[float, float]
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray


def apply(model: mujoco.MjModel, data: mujoco.MjData, s: ResetState) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:], data.ctrl[:] = s.qpos, s.qvel, s.ctrl
    data.mocap_pos[model.body_mocapid[model.body("plate").id]] = [*s.plate_xy, 0.0]
    mujoco.mj_forward(model, data)


def _ticks(model, data, n):
    for _ in range(n * SUBSTEPS):
        mujoco.mj_step(model, data)


def held(model, data) -> bool:
    jaws = {model.body("gripper").id, model.body("moving_jaw_so101_v1").id}
    b = model.body("bottle").id
    contact = any(
        {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} & jaws
        and b in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
        for c in data.contact[: data.ncon]
    )
    return bool(contact and data.qpos[5] > CLOSED + 0.05)


class Sampler:
    """Builds each size variant once and samples reset states for it."""

    def __init__(self, variants: list[SceneParams], reachable: np.ndarray):
        self.variants, self.reachable, self._built = variants, reachable, {}
        self.rejected = Counter()  # reason -> count, for diagnostics

    def _reject(self, reason: str) -> None:
        self.rejected[reason] += 1

    def model(self, v: int) -> tuple[mujoco.MjModel, TopDownIK]:
        if v not in self._built:
            m = build(self.variants[v])
            self._built[v] = (m, TopDownIK(m))
        return self._built[v]

    def sample(self, v: int, family: str, rng: np.random.Generator) -> ResetState | None:
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        (m, ik), p = self.model(v), self.variants[v]
        d = mujoco.MjData(m)
        b, pl = sample_positions(rng, self.reachable, p)
        place(m, d, b, pl)
        q = lerobot_to_sim(DEMO_START) + np.radians(rng.uniform(-REST_JITTER_DEG, REST_JITTER_DEG, 5))
        d.qpos[:5] = np.clip(q, m.jnt_range[:5, 0] + LIMIT_MARGIN, m.jnt_range[:5, 1] - LIMIT_MARGIN)
        d.qpos[5] = gripper_to_sim(DEMO_START[5])
        d.ctrl[:6] = d.qpos[:6]
        gz = grasp_height(p)
        if family == "reaching":
            g = np.array([*b, gz])
            a, back = choose_approach(ik, g)
            site, _, _ = grasp_site(ik, g, a, p.body_radius)
            q, err = ik.solve_oriented(
                site - rng.uniform(0, back) * a, a, ik.solve_position(site)[0], iters=300
            )
            if err > 0.002:
                return self._reject("ik")
            d.qpos[:5], d.qpos[5] = q, OPEN
            d.ctrl[:5], d.ctrl[5] = q, gripper_to_sim(rng.uniform(0, 70))
        elif family in ("stable_grasp", "near_goal"):
            if family == "stable_grasp":
                xy = np.add(b, rng.uniform() * np.subtract(pl, b))
                bottom = rng.uniform(*LIFT)
            else:
                xy = np.add(pl, rng.uniform(-0.01, 0.01, 2))
                bottom = 0.003 + rng.uniform(*NEAR_GOAL_FRAC) * NEAR_GOAL_TOP
            g = np.array([*xy, bottom + gz])
            a, _ = choose_approach(ik, g)
            _, q, err = grasp_site(ik, g, a, p.body_radius)
            if err > 0.002:
                return self._reject("ik")
            adr = m.jnt_qposadr[m.joint("bottle").id]
            d.qpos[adr : adr + 3] = [*xy, bottom]
            d.qpos[:5], d.qpos[5] = q, OPEN
            d.ctrl[:5], d.ctrl[5] = q, CLOSED
        mujoco.mj_forward(m, d)
        if any(c.dist < -MAX_PEN for c in d.contact[: d.ncon]):
            return self._reject("penetration")
        if family in ("stable_grasp", "near_goal"):
            gravity = m.opt.gravity.copy()
            m.opt.gravity[:] = 0
            try:
                _ticks(m, d, 20)
            finally:
                m.opt.gravity[:] = gravity
        _ticks(m, d, 30)
        x, y, z = d.body("bottle").xpos
        if not (TABLE[0][0] < x < TABLE[0][1] and TABLE[1][0] < y < TABLE[1][1] and z > -0.01):
            return self._reject("off table")
        if np.abs(d.qvel).max() > SETTLED_QVEL:
            return self._reject("unsettled")
        if success(m, d, p):
            return self._reject("already solved")
        if family in ("stable_grasp", "near_goal") and not held(m, d):
            return self._reject("not held")
        return ResetState(family, v, tuple(b), tuple(pl), d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())
