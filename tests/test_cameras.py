import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("cv2")

from policy.calib.camera import mujoco_camera, project_mujoco, wrist_relative  # noqa: E402
from policy.sim import so101  # noqa: E402
from policy.sim.cameras import SIZES, Renderer  # noqa: E402
from policy.sim.scene import build, demo_params  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def front_camera():
    """Straight down from 0.7 m above (0.3, 0), image top toward the robot, like the real front camera."""
    rot_cv = np.column_stack([[0, -1.0, 0], [1.0, 0, 0], [0, 0, -1.0]])  # OpenCV camera axes in world columns
    return mujoco_camera(rot_cv, np.array([0.3, 0.0, 0.7]), K, SIZES["front"], "front")


def test_front_camera_sees_the_cap_where_the_calibration_projects_it():
    p = demo_params()
    cam = front_camera()
    m = build(p, {"front": cam})
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    img = Renderer(m, ("front",)).render(d, "front")
    assert img.shape == (480, 640, 3)
    top = p.bottle_height
    u, v = project_mujoco([*p.bottle_xy, top], cam["pos"], cam["quat"], cam["fovy"], SIZES["front"])
    r, g, b = img[int(round(v)), int(round(u))].astype(int)
    assert r - b > 80, (r, g, b)  # orange cap; the grey table is about (235, 235, 230)


def test_wrist_camera_rides_on_the_gripper():
    rel = {**wrist_relative([0.2, 0.05, 0.15], [1, 0, 0, 0], [0, -30, 40, 50, 0]), "fovy": 60.0}
    m = build(demo_params(), {"wrist": rel})
    d = mujoco.MjData(m)
    for joints in ([0, -30, 40, 50, 0], [20, -10, 30, 60, 10]):
        d.qpos[:5] = so101.lerobot_to_sim(joints)
        mujoco.mj_forward(m, d)
        body = d.body("gripper")
        expect = body.xpos + body.xmat.reshape(3, 3) @ rel["pos"]
        assert np.allclose(d.cam("wrist").xpos, expect, atol=1e-5)
    d.qpos[:5] = so101.lerobot_to_sim([0, -30, 40, 50, 0])
    mujoco.mj_forward(m, d)
    assert np.allclose(d.cam("wrist").xpos, [0.2, 0.05, 0.15], atol=1e-4)
    assert Renderer(m, ("wrist",)).render(d, "wrist").shape == (640, 480, 3)
