import subprocess

import numpy as np
import pytest

pytest.importorskip("mujoco")
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from policy.sim import so101  # noqa: E402
from policy.sim.demo_layout import estimate, grip_events, gripper_positions  # noqa: E402

HAS_TAG = subprocess.run(["git", "rev-parse", "-q", "--verify", so101.TAG], cwd=so101.REPO).returncode == 0
needs_tag = pytest.mark.skipif(not HAS_TAG, reason=f"git tag {so101.TAG} not fetched")


def trace(held=19.0):
    """Closed, open to 70, close on the bottle at `held`, carry, open to 60, close."""
    parts = [
        np.full(20, 2.0),
        np.linspace(2, 70, 15),
        np.full(20, 70.0),
        np.linspace(70, held, 10),
        np.full(60, held),
        np.linspace(held, 60, 10),
        np.full(20, 60.0),
    ]
    return np.concatenate(parts)


def test_grip_events_find_grasp_hold_and_release():
    g = trace()
    grasp, release, held = grip_events(g)
    assert held == pytest.approx(19.0) and 64 <= grasp <= 66 and 125 <= release <= 128


def test_grip_events_none_when_gripper_never_opens():
    assert grip_events(np.full(100, 2.0)) is None


def test_lerobot_degrees_to_sim_radians_drops_gripper():
    assert np.allclose(so101.lerobot_to_sim([90, -45, 0, 180, 30, 55]), np.radians([90, -45, 0, 180, 30]))


@needs_tag
def test_gripper_positions_match_mujoco_forward_kinematics():
    import mujoco

    m = so101.load_model()
    states = np.array([[0, 0, 0, 0, 0, 0], [-97, -103, 96, 54, 0, 2]], float)
    d = mujoco.MjData(m)
    for q, p in zip(states, gripper_positions(m, states), strict=True):
        d.qpos[:5] = np.radians(q[:5])
        mujoco.mj_kinematics(m, d)
        assert np.allclose(p, d.site("gripperframe").xpos)


@needs_tag
def test_estimate_on_a_synthetic_dataset(tmp_path):
    g = trace()
    body = np.tile([-97.0, -103.0, 96.0, 54.0, 0.0], (len(g), 1))
    rows = {"episode_index": [], "observation.state": []}
    for e in (0, 1, 2):
        rows["episode_index"] += [e] * len(g)
        rows["observation.state"] += np.column_stack([body, g]).astype(np.float32).tolist()
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(rows), tmp_path / "data" / "chunk-000" / "file-000.parquet")
    res = estimate(tmp_path, exclude=(2,))
    assert res["used"] == 2 and res["excluded"] == [2] and set(res["episodes"]) == {0, 1, 2}
    assert res["summary"]["held_grip"]["mean"] == pytest.approx(19.0)
    assert res["summary"]["grasp_xyz"]["std"] == [0.0, 0.0, 0.0]
