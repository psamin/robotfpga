import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp")
jnp = jax.numpy

from policy.rl import sgs_jax  # noqa: E402
from policy.sim import so101  # noqa: E402
from policy.sim.buffer import build  # noqa: E402
from policy.sim.scene import variants  # noqa: E402
from policy.sim.teacher_env import OBS_DIM  # noqa: E402
from policy.sim.vec_env import VariantVecEnv  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


@pytest.fixture(scope="module")
def buf():
    return build(2, 8, seed=0)[0]  # 2 variants x 8 rows (2 per family), grouped by variant


def test_sample_block_stays_in_its_block():
    s = sgs_jax.record(sgs_jax.init(10), jnp.array([3, 4, 7]), jnp.array([True, False, True]))
    ids = np.asarray(sgs_jax.sample_block(s, jax.random.key(0), 4, 3, (500,)))
    assert ids.min() >= 4 and ids.max() <= 6 and len(set(ids.tolist())) == 3


@pytest.mark.parametrize("strategy", ["sgs", "uniform"])
def test_episodes_restart_within_their_variant_and_are_recorded(buf, strategy):
    env = VariantVecEnv(buf, variants(2), envs_per_variant=2, strategy=strategy)
    for sim in env.sims:
        sim.max_steps = 2
    c, obs = env.init(jax.random.key(0), env.batch)
    assert obs.shape == (4, OBS_DIM)
    step = jax.jit(env.step)
    for _ in range(2):
        c, obs, reward, done, info = step(c, jnp.zeros((4, 6)))
    assert bool(done.all()) and int(c.sgs.counts.sum()) == 4
    for k, (start, size) in enumerate(env.blocks):
        assert ((np.asarray(c.config[k]) >= start) & (np.asarray(c.config[k]) < start + size)).all()
    assert np.array_equal(np.asarray(c.envs[0].t), [0, 0])


def test_rejects_ungrouped_buffers(buf):
    bad = dict(buf, variant=buf["variant"][::-1])
    with pytest.raises(ValueError):
        VariantVecEnv(bad, variants(2), envs_per_variant=1)
