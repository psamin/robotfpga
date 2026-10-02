import numpy as np
import pytest

from armlab.data.collect import collect_episode
from armlab.data.replay import replay
from armlab.sim.env import BlockSortEnv


@pytest.mark.parametrize("kind", ["none", "cube_move", "instr_swap"])
def test_replay_reproduces_episode(kind):
    env = BlockSortEnv()
    plan = {"seed": 42, "instr": 1, "sigma": 0.02, "kind": kind}
    ep = collect_episode(env, plan)
    assert ep["meta"]["success"]
    ok, frames = replay(env, ep["meta"], ep["executed"], keep_frames=True)
    assert ok
    np.testing.assert_array_equal(np.stack([f["image"] for f in frames]), ep["image"])
    np.testing.assert_array_equal(np.array([f["instr"] for f in frames]), ep["instr"])
