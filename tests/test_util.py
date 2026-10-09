import json

import numpy as np
import torch

from armlab.util.runmeta import write_runmeta
from armlab.util.seed import seed_everything


def test_seed_everything_is_repeatable():
    a = seed_everything(3).integers(0, 1000, 5), torch.rand(3)
    b = seed_everything(3).integers(0, 1000, 5), torch.rand(3)
    np.testing.assert_array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_runmeta_roundtrip(tmp_path):
    p = tmp_path / "run" / "meta.json"
    write_runmeta(p, {"lr": 1e-3}, seed=7, results={"success": 0.9})
    m = json.loads(p.read_text())
    assert m["seed"] == 7 and m["config"]["lr"] == 1e-3 and m["results"]["success"] == 0.9
    assert "mujoco" in m["versions"] and m["git"]
