import numpy as np
import pytest

from armlab.rl.buffer import build, load, save, state
from armlab.rl.resets import FAMILIES, apply
from armlab.sim.env import BlockSortEnv


@pytest.fixture(scope="module")
def small():
    return build(8, seed=3)


def test_balanced_and_grouped_by_family(small):
    buf, acceptance = small
    assert buf["family"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert {len(v) for v in buf.values()} == {8}
    assert set(acceptance) == set(FAMILIES) and all(0 < a <= 1 for a in acceptance.values())


def test_same_seed_same_buffer_for_any_worker_count(small):
    a, _ = small
    b, _ = build(8, seed=3, workers=2)
    c, _ = build(8, seed=3, chunk=1)  # different chunking is a different (but valid) buffer
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["qpos"], c["qpos"])


def test_save_load_round_trip_restores_states(small, tmp_path):
    buf, _ = small
    save(tmp_path / "r.npz", buf)
    loaded = load(tmp_path / "r.npz")
    assert all(np.array_equal(buf[k], loaded[k]) for k in buf)
    env = BlockSortEnv(jitter=None)
    for i in range(8):
        s = state(loaded, i)
        assert s.family == FAMILIES[i // 2]
        apply(env, s)
        assert np.array_equal(env.d.qpos, buf["qpos"][i]) and env.instr == buf["instr"][i]


@pytest.mark.parametrize("size", [0, 6])
def test_size_must_split_evenly(size):
    with pytest.raises(ValueError):
        build(size, seed=0)
