import numpy as np
import pytest

from armlab.rl.resets import CARRY_Z, FAMILIES, apply, sample
from armlab.sim.env import BIN_HALF, INSTRUCTIONS, REST_Q, BlockSortEnv
from armlab.sim.expert import HELD_GRIP, Expert, _step_norender
from armlab.sim.ik import TopDownIK


@pytest.fixture(scope="module")
def env_ik():
    env = BlockSortEnv(jitter=None)
    return env, TopDownIK(env.m)


def draw(env_ik, family, n=8, seed=0):
    rng, out = np.random.default_rng(seed), []
    for _ in range(10 * n):
        s = sample(*env_ik, family, rng)
        if s is not None:
            out.append(s)
        if len(out) == n:
            return out
    pytest.fail(f"{family}: acceptance below 10%")


def held(env, instr):
    return env.touching_robot(INSTRUCTIONS[instr][0]) and env.d.qpos[5] < HELD_GRIP


def test_rest_is_the_standard_start(env_ik):
    env, _ = env_ik
    for s in draw(env_ik, "rest"):
        apply(env, s)
        assert np.allclose(env.d.qpos[:6], REST_Q, atol=0.05)
        assert not held(env, s.instr)


@pytest.mark.parametrize("family", ["stable_grasp", "near_goal"])
def test_grasp_families_hold_the_target_cube(env_ik, family):
    env, _ = env_ik
    for s in draw(env_ik, family):
        apply(env, s)
        color, side = INSTRUCTIONS[s.instr]
        assert held(env, s.instr)
        cube = env.cube_pos(color)
        if family == "stable_grasp":
            assert CARRY_Z[0] - 0.01 < cube[2] < CARRY_Z[1] + 0.01
        else:
            assert np.all(np.abs(cube[:2] - env.bin_pos(side)[:2]) < BIN_HALF)
            assert cube[2] < CARRY_Z[0] + 0.01
            assert not env.success()


def test_reaching_both_holds_and_drops(env_ik):
    env, _ = env_ik
    outcomes = set()
    for s in draw(env_ik, "reaching", n=20):
        apply(env, s)
        outcomes.add(held(env, s.instr))
    assert outcomes == {True, False}


def test_apply_restores_state_exactly(env_ik):
    env, _ = env_ik
    s = draw(env_ik, "near_goal", n=1)[0]
    draw(env_ik, "rest", n=1)  # move the env elsewhere
    apply(env, s)
    assert env.instr == s.instr and env.t == 0
    assert np.array_equal(env.d.qpos, s.qpos) and np.array_equal(env.d.ctrl, s.ctrl)


def test_sampling_is_reproducible(env_ik):
    a, b = draw(env_ik, "stable_grasp", n=3, seed=5), draw(env_ik, "stable_grasp", n=3, seed=5)
    assert all(np.array_equal(x.qpos, y.qpos) and x.instr == y.instr for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("family", FAMILIES)
def test_expert_can_finish_from_every_family(env_ik, family):
    env, _ = env_ik
    for s in draw(env_ik, family, n=4, seed=1):
        apply(env, s)
        expert = Expert(env)
        assert any(_step_norender(env, expert.act())[2]["success"] for _ in range(400)), family


def test_unknown_family_is_rejected(env_ik):
    with pytest.raises(ValueError):
        sample(*env_ik, "teleport", np.random.default_rng(0))
