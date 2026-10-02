import numpy as np
import pytest

from armlab.sim.env import INSTRUCTIONS, BlockSortEnv


@pytest.fixture(scope="module")
def env():
    return BlockSortEnv()


def test_obs_shapes(env):
    o = env.reset(seed=0, instr=1)
    assert o["image"].shape == (96, 96, 3) and o["image"].dtype == np.uint8
    assert o["image_full"].shape == (288, 288, 3)
    assert o["state"].shape == (6,) and o["state"].dtype == np.float32
    assert np.all(np.abs(o["state"]) <= 1) and o["instr"] == 1


def test_deterministic_under_seed(env):
    def rollout():
        o = env.reset(seed=123)
        rng = np.random.default_rng(5)
        frames = [o["image"]]
        for _ in range(10):
            o, _, _ = env.step(rng.uniform(-1, 1, 6))
            frames.append(o["image"])
        return np.stack(frames), o["state"], env.instr

    a, b = rollout(), rollout()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_normalize_roundtrip(env):
    q = np.array([0.3, -0.5, 1.0, 0.2, -1.0, 0.8])
    np.testing.assert_allclose(env.denormalize(env.normalize(q)), q, atol=1e-6)


def test_success_check(env):
    env.reset(seed=1, instr=2)  # blue -> left
    color, side = INSTRUCTIONS[2]
    assert not env.success()
    # teleport the blue cube into the left bin and let it settle
    adr = env.m.jnt_qposadr[env.m.joint("cube_blue").id]
    bx, by, _ = env.bin_pos("left")
    env.d.qpos[adr : adr + 7] = [bx, by, 0.03, 1, 0, 0, 0]
    env.d.qvel[:] = 0
    for _ in range(5):
        env.step(env.normalize(env.d.qpos[:6]))
    assert env.in_bin(color, side) and env.success()
    assert env.failure_reason() == "success"
    env.instr = 3  # blue -> right: same state is the wrong bin
    assert env.failure_reason() == "wrong_bin"
