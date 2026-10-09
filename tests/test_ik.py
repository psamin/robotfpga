import mujoco
import numpy as np

from armlab.sim.env import SCENE
from armlab.sim.ik import DOWN, TopDownIK


def test_reaches_workspace_top_down():
    ik = TopDownIK(mujoco.MjModel.from_xml_path(str(SCENE)))
    rng = np.random.default_rng(0)
    errs, tilts = [], []
    for _ in range(100):
        pos = rng.uniform([0.15, -0.11, 0.0125], [0.27, 0.11, 0.075])
        q, err = ik.solve_symmetric(pos, yaw=rng.uniform(-0.8, 0.8))
        _, r = ik.site_pose(q)
        errs.append(err)
        tilts.append(np.linalg.norm(np.cross(r[:, 0], DOWN)))
    assert max(errs) < 0.005, max(errs)
    assert max(tilts) < 0.05, max(tilts)


def test_reaches_above_bins():
    ik = TopDownIK(mujoco.MjModel.from_xml_path(str(SCENE)))
    for y in (0.20, -0.20):
        _, err = ik.solve([0.14, y, 0.07])
        assert err < 0.005
