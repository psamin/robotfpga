import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from policy.sim import so101  # noqa: E402
from policy.sim.expert import Expert  # noqa: E402
from policy.sim.resets import Sampler  # noqa: E402
from policy.sim.scene import reachable_grid, variants  # noqa: E402
from policy.sim.teacher_env import ACTION_SCALE, OBS_DIM, W_ACTION, W_RATE, TeacherEnv  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


@pytest.fixture(scope="module")
def env():
    return TeacherEnv(Sampler(variants(2), reachable_grid()))


def draw(env, family, seed=0, variant=0):
    rng = np.random.default_rng(seed)
    while (s := env.sampler.sample(variant, family, rng)) is None:
        pass
    return s


def test_observation_layout(env):
    s = draw(env, "rest")
    obs = env.reset(s)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    assert np.allclose(obs[6:12], 0) and np.allclose(obs[22:24], s.plate_xy)


def test_scripted_expert_through_delta_actions_succeeds_from_rest(env):
    env.reset(draw(env, "rest", seed=3))
    expert, total, done = Expert(env.m, env.d, env.p), 0.0, False
    while not done:
        _, r, done, info = env.step(np.clip((expert.act() - env.target) / ACTION_SCALE, -1, 1))
        total += r
    assert info["success"] and not info["failure"] and 99 < total <= 100


def test_zero_action_holds_target_and_penalties_follow_eq5(env):
    env.reset(draw(env, "stable_grasp"))
    start = env.target.copy()
    for _ in range(5):
        _, _, done, info = env.step(np.zeros(6))
    assert np.array_equal(env.target, start) and not done
    a = np.array([0.5, -0.5, 0, 0, 0, 1.0])
    env.step(a)
    _, _, _, info = env.step(-a)
    assert np.isclose(info["terms"]["action"], -W_ACTION * (a @ a))
    assert np.isclose(info["terms"]["action_rate"], -W_RATE * 4 * (a @ a))


def test_bottle_off_the_table_fails(env):
    env.reset(draw(env, "rest"))
    adr = env.m.jnt_qposadr[env.m.joint("bottle").id]
    env.d.qpos[adr : adr + 3] = [0.2, 0.6, 0.0]
    _, r, done, info = env.step(np.zeros(6))
    assert info["failure"] and done and r < -0.99


def test_step_limit_truncates():
    env = TeacherEnv(Sampler(variants(2), reachable_grid()), max_steps=3)
    env.reset(draw(env, "rest"))
    assert [env.step(np.zeros(6))[2] for _ in range(3)] == [False, False, True]
