import json
from dataclasses import asdict

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")  # eval_teacher -> train_teacher -> teacher_mjx needs the gpu group

from policy.rl import ppo  # noqa: E402
from policy.rl.eval_teacher import main  # noqa: E402
from policy.rl.train_teacher import save_checkpoint  # noqa: E402
from policy.sim import so101  # noqa: E402
from policy.sim.families import FAMILIES  # noqa: E402
from policy.sim.teacher_env import OBS_DIM  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


def test_expert_reference_on_held_out_rest_states(tmp_path):
    main(["--expert", "--variants", "2", "--per-variant", "8", "--out", str(tmp_path / "e.json")])
    res = json.loads((tmp_path / "e.json").read_text())["results"]
    assert res["rest"]["episodes"] == 4 and res["overall_success"] >= 0.5


def test_untrained_checkpoint_evaluates_every_family(tmp_path):
    cfg = ppo.PPOConfig(hidden=(16, 16))
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    meta = {"config": {"ppo": asdict(cfg), "variants": 2, "variant_seed": 0}}
    (run / "runmeta.json").write_text(json.dumps(meta))
    learner = ppo.Learner(
        ppo.init_params(jax.random.key(0), OBS_DIM, 6, cfg), None, ppo.norm_init(OBS_DIM), *[None] * 3
    )
    save_checkpoint(run / "checkpoints" / "iter_1.npz", learner)
    out = tmp_path / "eval.json"
    main(
        [
            "--checkpoint",
            str(run / "checkpoints" / "iter_1.npz"),
            "--out",
            str(out),
            *"--per-variant 4 --max-steps 10".split(),
        ]
    )
    res = json.loads(out.read_text())["results"]
    assert all(res[f]["episodes"] == 2 and res[f]["mean_steps"] <= 10 for f in FAMILIES)
