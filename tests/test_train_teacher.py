import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp")

from policy.rl import ppo  # noqa: E402
from policy.rl.train_teacher import load_checkpoint, main, save_checkpoint  # noqa: E402
from policy.sim import so101  # noqa: E402
from policy.sim.buffer import build, save  # noqa: E402
from policy.sim.families import FAMILIES  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


def test_cli_smoke_writes_metrics_checkpoint_and_meta(tmp_path):
    save(tmp_path / "buf.npz", build(2, 8, seed=0)[0])
    out = tmp_path / "run"
    args = "--envs-per-variant 2 --steps 4 --iters 2 --max-steps 3 --seed 0".split()
    main(["--buffer", str(tmp_path / "buf.npz"), "--out", str(out), *args])
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in rows] == [1, 2] and rows[-1]["episodes"] > 0
    assert all(f"success_{f}" in rows[-1] and f"episodes_{f}" in rows[-1] for f in FAMILIES)
    meta = json.loads((out / "runmeta.json").read_text())
    assert meta["config"]["variants"] == 2 and (out / "checkpoints" / "iter_2.npz").exists()


def test_checkpoint_round_trip(tmp_path):
    class Dummy:
        def init(self, key, batch):
            return None, jax.numpy.zeros((batch, 3))

    cfg = ppo.PPOConfig(envs=2, hidden=(8,))
    learner = ppo.make_train(Dummy(), 3, 2, cfg)[0](jax.random.key(0))
    learner = learner._replace(norm=ppo.norm_update(learner.norm, jax.numpy.ones((5, 3))))
    save_checkpoint(tmp_path / "c.npz", learner)
    params, norm = load_checkpoint(
        tmp_path / "c.npz", ppo.make_train(Dummy(), 3, 2, cfg)[0](jax.random.key(1))
    )
    assert all(jax.tree.leaves(jax.tree.map(np.array_equal, params, learner.params)))
    assert np.allclose(norm.mean, learner.norm.mean)
