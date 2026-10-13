import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp")
jnp = jax.numpy

import mujoco  # noqa: E402

from armlab.rl.resets import FAMILIES, apply, sample  # noqa: E402
from armlab.rl.teacher_env import ACTION_SCALE, OBS_DIM, W_SUCCESS, TeacherEnv  # noqa: E402
from armlab.rl.teacher_mjx import TeacherMJX, build_model  # noqa: E402
from armlab.sim.expert import Expert  # noqa: E402
from armlab.sim.ik import TopDownIK  # noqa: E402

STEPS = 120


@pytest.fixture(scope="module")
def states():
    cpu, rng, out = TeacherEnv(), np.random.default_rng(0), []
    ik = TopDownIK(cpu.env.m)
    for family in FAMILIES:
        while (s := sample(cpu.env, ik, family, rng)) is None:
            pass
        out.append(s)
    return out


@pytest.fixture(scope="module")
def rollouts(states):
    """The CPU expert's delta actions, replayed open-loop through the MJX env."""
    cpu, k, b = TeacherEnv(), STEPS, len(states)
    acts, obs, rew, succ = (
        np.zeros((k, b, 6)),
        np.zeros((k, b, OBS_DIM)),
        np.zeros((k, b)),
        np.zeros((k, b), bool),
    )
    for j, s in enumerate(states):
        cpu.reset(s)
        expert = Expert(cpu.env)
        for i in range(k):
            acts[i, j] = np.clip((expert.act() - cpu.target) / ACTION_SCALE, -1, 1)
            obs[i, j], rew[i, j], _, info = cpu.step(acts[i, j])
            succ[i, j] = info["success"]
    gpu = TeacherMJX(batch=b)
    state, first_obs = jax.jit(jax.vmap(gpu.reset))(
        *(jnp.array(np.stack([getattr(s, f) for s in states])) for f in ("qpos", "qvel", "ctrl")),
        jnp.array([s.instr for s in states]),
    )
    step, out = jax.jit(jax.vmap(gpu.step)), []
    for i in range(k):
        state, o, r, _, info = step(state, jnp.array(acts[i], jnp.float32))
        out.append((np.asarray(o), np.asarray(r), np.asarray(info["success"])))
    gobs, grew, gsucc = (np.array(x) for x in zip(*out, strict=True))
    return {"cpu": (obs, rew, succ), "gpu": (gobs, grew, gsucc), "first_obs": np.asarray(first_obs)}


def first_success(succ):
    return np.where(succ.any(0), succ.argmax(0), -1)


def test_every_family_succeeds_on_the_same_step(rollouts):
    cpu, gpu = first_success(rollouts["cpu"][2]), first_success(rollouts["gpu"][2])
    assert (cpu >= 0).all(), "expert should finish within the test horizon"
    # float32 Warp vs float64 CPU: a released cube can land a tick apart
    assert np.abs(cpu - gpu).max() <= 2, (cpu, gpu)


def test_observations_and_rewards_track_the_cpu_env(rollouts):
    (obs, rew, succ), (gobs, grew, gsucc) = rollouts["cpu"], rollouts["gpu"]
    assert rollouts["first_obs"].shape == (len(FAMILIES), OBS_DIM)
    assert np.abs(gobs[:30, :, :6] - obs[:30, :, :6]).max() < 0.01  # normalized joints
    assert np.abs(gobs[:30, :, 15:18] - obs[:30, :, 15:18]).max() < 0.01  # target cube (m)
    # success timing is checked above; compare the shaping penalties step by step
    shaping, gshaping = rew - W_SUCCESS * succ, grew - W_SUCCESS * gsucc
    assert np.abs(gshaping[:30] - shaping[:30]).max() < 1e-3


def test_contact_sensor_matches_cpu_touching_check(states):
    from armlab.sim.env import BlockSortEnv

    env = BlockSortEnv(jitter=None)
    env.m = build_model()
    env.d = mujoco.MjData(env.m)
    for s in states:
        apply(env, s)
        for i, c in enumerate(("red", "blue")):
            assert (env.d.sensordata[i] > 0) == env.touching_robot(c)


def test_cube_off_the_table_fails(states):
    gpu = TeacherMJX(batch=1)
    qpos = np.array(states[0].qpos)
    adr = build_model().jnt_qposadr[build_model().joint("cube_blue").id]
    qpos[adr : adr + 3] = [0.15, 0.6, 0.0125]
    s, _ = jax.jit(gpu.reset)(qpos, states[0].qvel, states[0].ctrl, states[0].instr)
    _, _, r, done, info = jax.jit(gpu.step)(s, jnp.zeros(6))
    assert bool(info["failure"]) and bool(done) and float(r) < -0.99
