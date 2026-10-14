import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from policy.sim import so101  # noqa: E402
from policy.sim.expert import run_episode  # noqa: E402
from policy.sim.scene import build, demo_params, reachable_grid, sample_params, success  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


def placed(p, xy, tilt_deg=0.0, z=0.003):
    m = build(p)
    d = mujoco.MjData(m)
    adr = m.jnt_qposadr[m.joint("bottle").id]
    half = np.radians(tilt_deg) / 2
    d.qpos[adr : adr + 7] = [*xy, z, np.cos(half), np.sin(half), 0, 0]
    d.qpos[:5] = so101.lerobot_to_sim([0, -100, 95, 50, 0])
    mujoco.mj_forward(m, d)
    return m, d


def test_success_needs_upright_inside_and_released():
    p = demo_params()
    assert success(*placed(p, p.plate_xy), p)
    assert not success(*placed(p, np.add(p.plate_xy, [p.plate_inner_radius + 0.01, 0])), p)
    assert not success(*placed(p, p.plate_xy, tilt_deg=40), p)
    assert not success(*placed(p, p.bottle_xy, z=0.0), p)


def test_expert_solves_the_demo_scene():
    r = run_episode(demo_params())
    assert r["success"] and r["steps"] < 300


def test_expert_solves_most_random_scenes():
    rng, grid = np.random.default_rng(0), reachable_grid()
    results = [run_episode(sample_params(rng, grid))["success"] for _ in range(10)]
    assert sum(results) >= 6, results  # 7 of 10 on macOS arm64; physics can differ slightly elsewhere
