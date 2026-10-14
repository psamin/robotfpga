import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp")
jnp = jax.numpy

from policy.sim import so101  # noqa: E402
from policy.sim.expert import Expert  # noqa: E402
from policy.sim.resets import Sampler  # noqa: E402
from policy.sim.scene import reachable_grid, variants  # noqa: E402
from policy.sim.teacher_env import ACTION_SCALE, OBS_DIM, W_SUCCESS, TeacherEnv  # noqa: E402
from policy.sim.teacher_mjx import TeacherMJX  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")
STEPS = 240
OPEN_GRIPPER = np.array([0, 0, 0, 0, 0, 1.0])


@pytest.fixture(scope="module")
def rollouts():
    """Variant 0: the expert from two rest starts, and opening the gripper from two near-goal starts."""
    sampler = Sampler(variants(2), reachable_grid())
    cpu, rng, starts = TeacherEnv(sampler), np.random.default_rng(0), []
    for family, n in (("rest", 2), ("near_goal", 2)):
        while sum(s.family == family for s in starts) < n:
            s = sampler.sample(0, family, rng)
            if s is not None:
                starts.append(s)
    b = len(starts)
    acts, obs, rew, succ = (
        np.zeros((STEPS, b, 6)),
        np.zeros((STEPS, b, OBS_DIM)),
        np.zeros((STEPS, b)),
        np.zeros((STEPS, b), bool),
    )
    for j, s in enumerate(starts):
        cpu.reset(s)
        expert = Expert(cpu.m, cpu.d, cpu.p) if s.family == "rest" else None
        for i in range(STEPS):
            a = np.clip((expert.act() - cpu.target) / ACTION_SCALE, -1, 1) if expert else OPEN_GRIPPER
            acts[i, j] = a
            obs[i, j], rew[i, j], _, info = cpu.step(a)
            succ[i, j] = info["success"]
    gpu = TeacherMJX(sampler.variants[0], batch=b)
    stack = lambda f: jnp.array(np.stack([getattr(s, f) for s in starts]))  # noqa: E731
    state, _ = jax.jit(jax.vmap(gpu.reset))(stack("qpos"), stack("qvel"), stack("ctrl"), stack("plate_xy"))
    step, out = jax.jit(jax.vmap(gpu.step)), []
    for i in range(STEPS):
        state, o, r, _, info = step(state, jnp.array(acts[i], jnp.float32))
        out.append((np.asarray(o), np.asarray(r), np.asarray(info["success"])))
    g = [np.array(x) for x in zip(*out, strict=True)]
    return {"cpu": (obs, rew, succ), "gpu": g, "families": [s.family for s in starts]}


def first(succ):
    return np.where(succ.any(0), succ.argmax(0), -1)


def test_success_timing_matches_the_cpu_env(rollouts):
    cpu, gpu = first(rollouts["cpu"][2]), first(rollouts["gpu"][2])
    assert (cpu >= 0).sum() >= 2, (rollouts["families"], cpu)
    assert np.array_equal(cpu >= 0, gpu >= 0) and np.abs(cpu - gpu)[cpu >= 0].max() <= 3, (cpu, gpu)


def test_observations_and_shaping_rewards_track_the_cpu_env(rollouts):
    (obs, rew, succ), (gobs, grew, gsucc) = rollouts["cpu"], rollouts["gpu"]
    assert np.abs(gobs[:30, :, :6] - obs[:30, :, :6]).max() < 0.01
    assert np.abs(gobs[:30, :, 15:18] - obs[:30, :, 15:18]).max() < 0.01
    assert np.allclose(gobs[0, :, 22:], obs[0, :, 22:], atol=1e-5)
    shaping, gshaping = rew - W_SUCCESS * succ, grew - W_SUCCESS * gsucc
    assert np.abs(gshaping[:30] - shaping[:30]).max() < 1e-3
