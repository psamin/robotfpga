import pytest

from armlab.sim.env import BlockSortEnv
from armlab.sim.expert import run_episode


@pytest.mark.parametrize("instr", range(4))
def test_expert_solves_each_instruction(instr):
    ok, steps, why, _ = run_episode(BlockSortEnv(), seed=100 + instr, instr=instr)
    assert ok, why
    assert steps < 300
