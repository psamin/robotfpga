import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "native_camera_shapes", Path(__file__).resolve().parents[1] / "robot/toolkit/native_camera_shapes.py"
)
ncs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncs)


def base_config():
    """input_features as in the -9604 config.json (inherited from lerobot/smolvla_base)."""
    cam = {"type": "VISUAL", "shape": [3, 256, 256]}
    return {
        "type": "smolvla",
        "resize_imgs_with_padding": [512, 512],
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.camera1": dict(cam),
            "observation.images.camera2": dict(cam),
            "observation.images.camera3": dict(cam),
        },
    }


def test_recorded_shapes():
    cfg = base_config()
    out = ncs.native(cfg)
    feats = out["input_features"]
    assert feats["observation.images.camera1"]["shape"] == [3, 480, 640]  # front, landscape
    assert feats["observation.images.camera2"]["shape"] == [3, 640, 480]  # wrist, portrait
    assert feats["observation.images.camera3"] == cfg["input_features"]["observation.images.camera3"]
    assert out["resize_imgs_with_padding"] == [512, 512]
    assert cfg["input_features"]["observation.images.camera1"]["shape"] == [3, 256, 256]  # input not modified


def test_missing_camera():
    cfg = base_config()
    del cfg["input_features"]["observation.images.camera2"]
    with pytest.raises(KeyError):
        ncs.native(cfg)
