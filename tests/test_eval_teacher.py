import json
from dataclasses import asdict

import pytest

jax = pytest.importorskip("jax")

from armlab.rl import ppo  # noqa: E402
from armlab.rl.eval_teacher import main  # noqa: E402
from armlab.rl.resets import FAMILIES  # noqa: E402
from armlab.rl.teacher_env import OBS_DIM  # noqa: E402
from armlab.rl.train_teacher import save_checkpoint  # noqa: E402


def test_expert_reference_succeeds_on_held_out_resets(tmp_path):
    main(["--expert", "--episodes-per-family", "2", "--out", str(tmp_path / "e.json")])
    res = json.loads((tmp_path / "e.json").read_text())["results"]
    assert set(FAMILIES) <= set(res) and all(res[f]["episodes"] == 2 for f in FAMILIES)
    assert res["overall_success"] >= 0.75


def test_untrained_checkpoint_evaluates(tmp_path):
    cfg = ppo.PPOConfig(hidden=(16, 16))
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "runmeta.json").write_text(json.dumps({"config": {"ppo": asdict(cfg)}}))
    params = ppo.init_params(jax.random.key(0), OBS_DIM, 6, cfg)
    save_checkpoint(
        run / "checkpoints" / "iter_1.npz", ppo.Learner(params, None, ppo.norm_init(OBS_DIM), *[None] * 3)
    )
    out = tmp_path / "eval.json"
    ckpt = str(run / "checkpoints" / "iter_1.npz")
    main(["--checkpoint", ckpt, "--out", str(out), *"--episodes-per-family 1 --max-steps 20".split()])
    res = json.loads(out.read_text())["results"]
    assert all(res[f]["episodes"] == 1 and res[f]["mean_steps"] <= 20 for f in FAMILIES)
