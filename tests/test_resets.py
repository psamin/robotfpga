import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from policy.sim import so101  # noqa: E402
from policy.sim.buffer import build, load, save, state  # noqa: E402
from policy.sim.resets import FAMILIES, Sampler, apply, held  # noqa: E402
from policy.sim.scene import reachable_grid, variants  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


@pytest.fixture(scope="module")
def sampler():
    return Sampler(variants(2), reachable_grid())


def draw(sampler, family, n=3, seed=0):
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        s = sampler.sample(int(rng.integers(2)), family, rng)
        if s is not None:
            out.append(s)
    return out


@pytest.mark.parametrize("family", ["stable_grasp", "near_goal"])
def test_held_families_keep_the_bottle_after_apply(sampler, family):
    for s in draw(sampler, family):
        m, _ = sampler.model(s.variant)
        d = mujoco.MjData(m)
        apply(m, d, s)
        for _ in range(150):
            mujoco.mj_step(m, d)
        assert held(m, d) and d.body("bottle").xpos[2] > 0.0
        if family == "near_goal":
            assert np.linalg.norm(d.body("bottle").xpos[:2] - s.plate_xy) < 0.03


def test_rest_starts_at_the_demo_pose_and_apply_is_exact(sampler):
    for s in draw(sampler, "rest"):
        m, _ = sampler.model(s.variant)
        d = mujoco.MjData(m)
        apply(m, d, s)
        assert np.array_equal(d.qpos, s.qpos) and not held(m, d)
        assert np.allclose(d.body("plate").xpos[:2], s.plate_xy)


def test_buffer_grouped_by_variant_then_family_and_worker_independent(tmp_path):
    a, acc = build(2, 4, seed=1)
    b, _ = build(2, 4, seed=1, workers=2)
    assert a["variant"].tolist() == [0] * 4 + [1] * 4 and a["family"].tolist() == [0, 1, 2, 3] * 2
    assert all(np.array_equal(a[k], b[k]) for k in a) and set(acc) == set(FAMILIES)
    save(tmp_path / "r.npz", a)
    c = load(tmp_path / "r.npz")
    assert all(np.array_equal(a[k], c[k]) for k in a)
    assert state(c, 5).family == "reaching" and state(c, 5).variant == 1
