import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp")
jnp = jax.numpy

from armlab.rl import sgs_jax  # noqa: E402
from armlab.rl.buffer import build  # noqa: E402
from armlab.rl.resets import FAMILIES  # noqa: E402
from armlab.rl.teacher_mjx import TeacherMJX  # noqa: E402
from armlab.rl.vec_env import Carry, SGSVecEnv  # noqa: E402


@pytest.fixture(scope="module")
def buf():
    return build(8, seed=0)[0]  # two rows per family, in FAMILIES order


def test_truncated_episodes_restart_and_are_recorded(buf):
    env = SGSVecEnv(TeacherMJX(batch=4, max_steps=3), buf)
    c, _ = env.init(jax.random.key(0), 4)
    step = jax.jit(env.step)
    for _ in range(3):
        before = c.config
        c, obs, _, done, info = step(c, jnp.zeros((4, 6)))
    assert bool(done.all()) and np.array_equal(info["config"], before)
    assert np.array_equal(c.env.t, np.zeros(4)) and int(c.sgs.counts.sum()) == 4
    _, first = env._start(c.config)  # returned obs is the next episode's first obs
    assert np.allclose(obs, first, atol=1e-5)


def test_only_finished_envs_reset(buf):
    """A near-goal env that opens its gripper finishes; a rest env doing nothing keeps running."""
    env = SGSVecEnv(TeacherMJX(batch=2), buf)
    rest, near = FAMILIES.index("rest") * 2, FAMILIES.index("near_goal") * 2
    ids = jnp.array([rest, near])
    state, _ = env._start(ids)
    c = Carry(state, ids, sgs_jax.init(env.n), jax.random.key(1))
    open_gripper = jnp.array([[0.0] * 6, [0.0] * 5 + [1.0]])
    step, steps = jax.jit(env.step), 0
    while steps < 60:
        c, _, _, done, info = step(c, open_gripper)
        steps += 1
        if bool(done[1]):
            break
    assert bool(done[1]) and bool(info["success"][1]), "near-goal release should succeed"
    assert not bool(done[0]) and int(c.env.t[0]) == steps and int(c.config[0]) == rest
    assert int(c.sgs.counts[near]) == 1 and int(c.sgs.successes[near]) == 1
    assert int(c.sgs.counts.sum()) == 1


@pytest.mark.parametrize("strategy", ["sgs", "uniform"])
def test_both_strategies_record_outcomes(buf, strategy):
    env = SGSVecEnv(TeacherMJX(batch=3, max_steps=1), buf, strategy=strategy)
    c, _ = env.init(jax.random.key(2), 3)
    c, *_ = jax.jit(env.step)(c, jnp.zeros((3, 6)))
    assert int(c.sgs.counts.sum()) == 3


def test_unknown_strategy_is_rejected(buf):
    with pytest.raises(ValueError):
        SGSVecEnv(TeacherMJX(batch=1), buf, strategy="greedy")
