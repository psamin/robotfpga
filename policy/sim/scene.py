"""Randomized pick-and-place scene: SO-101, a medicine bottle and a plate (docs/smolvla-post-training.md).

Every episode draws its sizes, mass, friction and positions from RANGES, which are centred on the
estimates from the demos and the front video (#172, #174, #176) and wide enough to contain the real
values. Positions cover the arm's reachable desk area (reachable_grid), with a share of episodes at
the demo layout. World frame: robot base at the origin, table top at z = 0, +x forward, +y left.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from policy.sim.ik import TopDownIK
from policy.sim.so101 import CACHE, model_dir

SUBSTEPS = 10
TIMESTEP = 1 / 300  # 10 physics steps per 30 Hz control tick, the demos' and SmolVLA's rate
TABLE_X, TABLE_Y = (-0.10, 0.60), (-0.40, 0.40)
RIM_SEGMENTS = 16
SEPARATION = 0.03  # m between the bottle and the plate rim at the start
MIN_X = 0.10  # objects start in front of the base: the front video shows the table starting at the base

RANGES = {  # metres, kilograms; see the plan's table for the basis of each range
    "bottle_height": (0.07, 0.15),
    "body_radius": (0.015, 0.025),
    "cap_radius": (0.010, 0.016),
    "cap_height": (0.015, 0.025),
    "bottle_mass": (0.02, 0.20),
    "plate_outer_radius": (0.08, 0.11),
    "plate_inner_radius": (0.05, 0.07),
    "rim_height": (0.01, 0.03),
    "friction_scale": (0.7, 1.3),
}
DEMO_BOTTLE, DEMO_PLATE = (0.25, 0.09), (0.33, -0.19)  # estimated from the demos (#172, #174)
DEMO_JITTER = 0.02


@dataclass(frozen=True)
class SceneParams:
    bottle_height: float
    body_radius: float
    cap_radius: float
    cap_height: float
    bottle_mass: float
    plate_outer_radius: float
    plate_inner_radius: float
    rim_height: float
    friction_scale: float
    bottle_xy: tuple[float, float]
    plate_xy: tuple[float, float]


def demo_params() -> SceneParams:
    """Range midpoints at the demo layout."""
    mid = {k: (lo + hi) / 2 for k, (lo, hi) in RANGES.items()}
    return SceneParams(**mid, bottle_xy=DEMO_BOTTLE, plate_xy=DEMO_PLATE)


def sample_params(rng: np.random.Generator, reachable: np.ndarray, demo_share: float = 0.25) -> SceneParams:
    """reachable: (N, 2) desk points the arm can grasp at, from reachable_grid."""
    p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in RANGES.items()}
    p["cap_radius"] = min(p["cap_radius"], p["body_radius"])
    p["plate_inner_radius"] = min(p["plate_inner_radius"], p["plate_outer_radius"] - 0.015)
    if rng.uniform() < demo_share:
        b = np.add(DEMO_BOTTLE, rng.uniform(-DEMO_JITTER, DEMO_JITTER, 2))
        pl = np.add(DEMO_PLATE, rng.uniform(-DEMO_JITTER, DEMO_JITTER, 2))
    else:
        gap = p["plate_outer_radius"] + p["body_radius"] + SEPARATION
        while True:
            b, pl = reachable[rng.integers(len(reachable), size=2)]
            if np.linalg.norm(b - pl) > gap:
                break
    return SceneParams(**p, bottle_xy=tuple(map(float, b)), plate_xy=tuple(map(float, pl)))


def _quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
    return q


def build(p: SceneParams) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(model_dir() / "so101.xml"))
    spec.option.timestep = TIMESTEP
    w = spec.worldbody
    w.add_light(
        name="key",
        pos=[0.2, 0.0, 1.2],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.6, 0.6, 0.6],
    )
    half = [(TABLE_X[1] - TABLE_X[0]) / 2, (TABLE_Y[1] - TABLE_Y[0]) / 2, 0.02]
    w.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=[sum(TABLE_X) / 2, sum(TABLE_Y) / 2, -0.02],
        rgba=[0.92, 0.92, 0.9, 1],
    )
    fr = [1.0 * p.friction_scale, 0.005, 0.0001]

    body_h = p.bottle_height - p.cap_height
    volume = np.pi * (p.body_radius**2 * body_h + p.cap_radius**2 * p.cap_height)
    bottle = w.add_body(name="bottle", pos=[*p.bottle_xy, 0.0])
    bottle.add_freejoint(name="bottle")
    bottle.add_geom(
        name="bottle_body",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[p.body_radius, body_h / 2, 0],
        pos=[0, 0, body_h / 2],
        density=p.bottle_mass / volume,
        friction=fr,
        condim=4,
        rgba=[0.95, 0.85, 0.6, 0.6],
    )
    bottle.add_geom(
        name="bottle_cap",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[p.cap_radius, p.cap_height / 2, 0],
        pos=[0, 0, body_h + p.cap_height / 2],
        density=p.bottle_mass / volume,
        friction=fr,
        condim=4,
        rgba=[1.0, 0.5, 0.1, 1],
    )

    plate = w.add_body(name="plate", pos=[*p.plate_xy, 0.0])
    plate.add_geom(
        name="plate_floor",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[p.plate_inner_radius, 0.0015, 0],
        pos=[0, 0, 0.0015],
        friction=fr,
        rgba=[0.97, 0.97, 0.97, 1],
    )
    width = p.plate_outer_radius - p.plate_inner_radius
    tilt = np.arctan2(p.rim_height, width)
    r_mid = (p.plate_inner_radius + p.plate_outer_radius) / 2
    for k in range(RIM_SEGMENTS):
        a = 2 * np.pi * k / RIM_SEGMENTS
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, _quat([0, 0, 1], a), _quat([0, 1, 0], -tilt))
        plate.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[np.hypot(width, p.rim_height) / 2, np.pi * r_mid / RIM_SEGMENTS * 1.05, 0.0015],
            pos=[r_mid * np.cos(a), r_mid * np.sin(a), p.rim_height / 2],
            quat=q,
            friction=fr,
            rgba=[0.7, 0.75, 0.9, 1],
        )
    return spec.compile()


def reachable_grid(step: float = 0.02, heights=(0.03, 0.12), tol: float = 0.003, max_tilt_deg: float = 85.0):
    """Desk points (x, y) the gripper reaches at every height within tol, tilted at most max_tilt_deg from
    straight down (the demos' most tilted release is 85 deg). Cached per argument set."""
    path = CACHE / f"reachable_{step}_{'_'.join(map(str, heights))}_{tol}_{max_tilt_deg}_{MIN_X}.npy"
    if path.exists():
        return np.load(path)
    ik = TopDownIK(mujoco.MjModel.from_xml_path(str(model_dir() / "so101.xml")))

    def reaches(p, z):
        q, err = ik.solve_position([*p, z])
        return err < tol and np.degrees(ik.tilt(q)) <= max_tilt_deg

    xs = np.arange(TABLE_X[0] + step, TABLE_X[1], step)
    ys = np.arange(TABLE_Y[0] + step, TABLE_Y[1], step)
    pts = [(x, y) for x in xs for y in ys if x >= MIN_X]
    ok = [p for p in pts if all(reaches(p, z) for z in heights)]
    grid = np.array(ok)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid)
    return grid
