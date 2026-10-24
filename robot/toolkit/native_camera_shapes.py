"""Give a SmolVLA checkpoint config the recorded camera shapes, so live frames are not squashed.

LeRobot's async inference resizes every live frame to the policy's `input_features` shape
(`async_inference/helpers.py`, `resize_robot_observation_image`) before SmolVLA letterboxes it to
`resize_imgs_with_padding`. Checkpoints fine-tuned from `lerobot/smolvla_base` keep its (3, 256, 256),
so the 640x480 front and 480x640 wrist frames are squashed to squares, while in training the recorded
frames were letterboxed with their aspect ratio kept. With the recorded shapes the client's resize
changes nothing and the live input matches training. Only config.json changes; weights are untouched.

Copy the checkpoint first; never edit -9604 itself:
    cp -r ~/models/<ckpt> ~/models/<ckpt>-native
    python robot/toolkit/native_camera_shapes.py ~/models/<ckpt>-native/config.json
"""

import json
import sys

# (C, H, W) of the recorded frames (dataset meta/info.json: front 480x640, wrist 640x480), under the
# names the policy uses (rename_map front -> camera1, wrist -> camera2; POLICY_CAMERAS in policy.sh).
SHAPES = {
    "observation.images.camera1": [3, 480, 640],
    "observation.images.camera2": [3, 640, 480],
}


def native(config: dict) -> dict:
    """Return the config with the recorded camera shapes; every other field unchanged."""
    out = json.loads(json.dumps(config))
    for key, shape in SHAPES.items():
        if key not in out["input_features"]:
            raise KeyError(f"{key} not in input_features")
        out["input_features"][key]["shape"] = shape
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        cfg = native(json.load(f))
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(path, {k: cfg["input_features"][k]["shape"] for k in SHAPES})
