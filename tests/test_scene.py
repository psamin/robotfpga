import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from policy.sim import so101  # noqa: E402
from policy.sim.scene import (  # noqa: E402
    DEMO_BOTTLE,
    DEMO_PLATE,
    MIN_X,
    RANGES,
    build,
    demo_params,
    reachable_grid,
    sample_params,
)

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


@pytest.fixture(scope="module")
def grid():
    return reachable_grid()


def settle(model, seconds=1.0):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


def upright(data):
    q = data.body("bottle").xquat
    return 1 - 2 * (q[1] ** 2 + q[2] ** 2)


def test_demo_scene_settles_with_the_bottle_upright_on_the_table():
    p = demo_params()
    d = settle(build(p))
    assert upright(d) > 0.999 and abs(d.body("bottle").xpos[2]) < 0.002
    assert np.allclose(d.body("bottle").xpos[:2], p.bottle_xy, atol=0.002)
    assert np.allclose(d.body("plate").xpos[:2], p.plate_xy)


def test_bottle_height_and_mass_follow_the_params():
    p = demo_params()
    m = build(p)
    assert m.body("bottle").mass[0] == pytest.approx(p.bottle_mass, rel=1e-6)
    top = m.geom("bottle_cap").pos[2] + m.geom("bottle_cap").size[1]
    assert top == pytest.approx(p.bottle_height)


def test_demo_spots_are_reachable(grid):
    for spot in (DEMO_BOTTLE, DEMO_PLATE):
        assert np.min(np.linalg.norm(grid - spot, axis=1)) < 0.02
    assert (grid[:, 0] >= MIN_X).all()


def test_samples_stay_in_range_reachable_and_apart(grid):
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = sample_params(rng, grid, demo_share=0.0)
        for k, (lo, hi) in RANGES.items():
            assert lo <= getattr(p, k) <= hi
        assert np.min(np.linalg.norm(grid - p.bottle_xy, axis=1)) < 1e-9
        assert np.linalg.norm(np.subtract(p.bottle_xy, p.plate_xy)) > p.plate_outer_radius + p.body_radius
        assert p.cap_radius <= p.body_radius and p.plate_inner_radius < p.plate_outer_radius


def test_random_scenes_settle_upright():
    rng, grid = np.random.default_rng(1), reachable_grid()
    for _ in range(10):
        assert upright(settle(build(sample_params(rng, grid)), 0.5)) > 0.99
