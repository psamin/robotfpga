import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_warp")

from armlab.rl import ppo  # noqa: E402
from armlab.rl.buffer import build, save  # noqa: E402
from armlab.rl.resets import FAMILIES  # noqa: E402
from armlab.rl.train_teacher import load_checkpoint, main, save_checkpoint  # noqa: E402


def test_cli_smoke_writes_metrics_checkpoint_and_meta(tmp_path):
    save(tmp_path / "buf.npz", build(8, seed=0)[0])
    out = tmp_path / "run"
    args = "--envs 4 --steps 4 --iters 2 --max-steps 3 --seed 0".split()
    main(["--buffer", str(tmp_path / "buf.npz"), "--out", str(out), *args])
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in rows] == [1, 2]
    assert rows[-1]["episodes"] > 0 and 0 < rows[-1]["sgs_coverage"] <= 1
    assert all(f"success_{f}" in rows[-1] for f in FAMILIES)
    assert (out / "checkpoints" / "iter_2.npz").exists() and (out / "runmeta.json").exists()


def test_checkpoint_round_trip(tmp_path):
    class Dummy:
        def init(self, key, batch):
            return None, jax.numpy.zeros((batch, 3))

    cfg = ppo.PPOConfig(envs=2, hidden=(8,))
    learner = ppo.make_train(Dummy(), 3, 2, cfg)[0](jax.random.key(0))
    learner = learner._replace(norm=ppo.norm_update(learner.norm, jax.numpy.ones((5, 3))))
    save_checkpoint(tmp_path / "c.npz", learner)
    fresh = ppo.make_train(Dummy(), 3, 2, cfg)[0](jax.random.key(1))
    params, norm = load_checkpoint(tmp_path / "c.npz", fresh)
    same = jax.tree.map(np.array_equal, params, learner.params)
    assert all(jax.tree.leaves(same)) and np.allclose(norm.mean, learner.norm.mean)
