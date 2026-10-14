import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from policy.sim import so101  # noqa: E402
from policy.sim.cameras import Renderer  # noqa: E402
from policy.sim.expert import Expert  # noqa: E402
from policy.sim.scene import build, demo_params  # noqa: E402
from policy.sim.smolvla_sim import TASK, observation, run_episode, to_ctrl  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")
CAMS = {
    "front": {"pos": [0.3, 0.0, 0.7], "quat": [0.7071068, 0, 0, 0.7071068], "fovy": 50.0},
    "wrist": {"body": "gripper", "pos": [0.0, 0.05, 0.0], "quat": [1, 0, 0, 0], "fovy": 60.0},
}


def test_conversions_round_trip():
    q = np.array([0.3, -1.2, 1.0, 0.5, -0.4, 0.2])
    assert np.allclose(to_ctrl(so101.sim_to_lerobot(q)), q)


def test_observation_matches_the_training_layout():
    m = build(demo_params(), CAMS)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    obs = observation(d, Renderer(m))
    assert obs["observation.images.camera1"].shape == (3, 480, 640)
    assert obs["observation.images.camera2"].shape == (3, 640, 480)
    assert 0.0 <= obs["observation.images.camera1"].min() and obs["observation.images.camera1"].max() <= 1.0
    assert obs["observation.state"].shape == (6,) and obs["task"] == TASK


def test_stand_in_policy_through_the_adapter_succeeds():
    """The scripted expert, re-expressed in LeRobot units one action per call, through the full adapter."""
    p = demo_params()
    m = build(p, CAMS)
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[:5, 0] + np.radians(2), m.jnt_range[:5, 1] - np.radians(2)
    d.qpos[:5] = np.clip(so101.lerobot_to_sim([-97, -103, 96, 54, 0]), lo, hi)
    d.qpos[5] = so101.gripper_to_sim(2.0)
    d.ctrl[:6] = d.qpos[:6]
    mujoco.mj_forward(m, d)
    expert = Expert(m, d, p)
    r = run_episode(lambda obs: so101.sim_to_lerobot(expert.act())[None], m, d, p, Renderer(m), max_steps=300)
    assert r["success"], r
