import numpy as np
import pytest

from armlab.rl.resets import sample
from armlab.rl.teacher_env import ACTION_SCALE, OBS_DIM, W_ACTION, W_RATE, TeacherEnv
from armlab.sim.expert import Expert
from armlab.sim.ik import TopDownIK


@pytest.fixture(scope="module")
def tenv():
    return TeacherEnv()


def draw(tenv, family, seed=0):
    ik, rng = TopDownIK(tenv.env.m), np.random.default_rng(seed)
    while (s := sample(tenv.env, ik, family, rng)) is None:
        pass
    return s


def expert_delta(tenv, expert):
    """The scripted expert's absolute targets, as the teacher's bounded delta action."""
    return np.clip((expert.act() - tenv.target) / ACTION_SCALE, -1, 1)


def test_observation_layout(tenv):
    obs = tenv.reset(draw(tenv, "rest"))
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    assert obs[28:].sum() == 1 and obs[28 + tenv.env.instr] == 1
    assert np.allclose(obs[6:12], 0)  # no previous action yet


@pytest.mark.parametrize("family", ["rest", "near_goal"])
def test_expert_through_delta_actions_succeeds(tenv, family):
    tenv.reset(draw(tenv, family, seed=2))
    expert, total, done = Expert(tenv.env), 0.0, False
    while not done:
        _, r, done, info = tenv.step(expert_delta(tenv, expert))
        total += r
    assert info["success"] and not info["failure"]
    assert 99 < total <= 100


def test_zero_action_holds_the_target(tenv):
    tenv.reset(draw(tenv, "stable_grasp"))
    start = tenv.target.copy()
    for _ in range(10):
        _, r, done, info = tenv.step(np.zeros(6))
    assert np.array_equal(tenv.target, start) and not done
    assert info["terms"]["action"] == info["terms"]["action_rate"] == 0


def test_penalty_terms_follow_eq5(tenv):
    tenv.reset(draw(tenv, "rest"))
    a = np.array([0.5, -0.5, 0, 0, 0, 1.0])
    tenv.step(a)
    _, _, _, info = tenv.step(-a)
    assert np.isclose(info["terms"]["action"], -W_ACTION * (a @ a))
    assert np.isclose(info["terms"]["action_rate"], -W_RATE * (4 * a @ a))


def test_cube_off_the_table_fails(tenv):
    tenv.reset(draw(tenv, "rest"))
    adr = tenv.env.m.jnt_qposadr[tenv.env.m.joint("cube_blue").id]
    tenv.env.d.qpos[adr : adr + 3] = [0.15, 0.6, 0.0125]
    _, r, done, info = tenv.step(np.zeros(6))
    assert info["failure"] and done and r < -0.99


def test_step_limit_truncates():
    tenv = TeacherEnv(max_steps=3)
    tenv.reset(draw(tenv, "rest"))
    dones = [tenv.step(np.zeros(6))[2] for _ in range(3)]
    assert dones == [False, False, True]
