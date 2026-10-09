from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "armlab" / "sim" / "assets" / "block_sort.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def test_scene_contents():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    assert m.nu == 6
    assert [m.actuator(i).name for i in range(6)] == JOINTS
    for body in ("bin_left", "bin_right", "cube_red", "cube_blue"):
        assert m.body(body).id >= 0
    assert m.camera("overhead").id >= 0
    assert m.site("gripperframe").id >= 0
    # 10 physics steps per 30 Hz control tick, exactly
    assert abs(10 * m.opt.timestep - 1 / 30) < 1e-12


def test_cubes_rest_on_table():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p0 = d.body("cube_red").xpos.copy()
    for _ in range(300):
        mujoco.mj_step(m, d)
    np.testing.assert_allclose(d.body("cube_red").xpos, p0, atol=1e-3)
