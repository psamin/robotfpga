"""Calibrated front and wrist cameras for the scene, and rendering at the real resolutions.

Camera file (JSON): {"front": {"pos", "quat", "fovy"}, "wrist": {"body", "pos", "quat", "fovy"}}.
front is in the world frame, from `python -m policy.calib.camera pose` on a front photo of the tag.
wrist is relative to the gripper body, from the `wrist` command (pose of a wrist photo of the tag
plus the arm's joint readings at capture). quat is w, x, y, z in MuJoCo's camera convention (looks
along -z, +y up); fovy is vertical, in degrees, for the image size below.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

SIZES = {"front": (640, 480), "wrist": (480, 640)}  # (width, height), SETUP.md §2; the wrist image is rotated


def load(path) -> dict:
    cams = json.loads(Path(path).read_text())
    for name, c in cams.items():
        if name not in SIZES or not {"pos", "quat", "fovy"} <= set(c):
            raise ValueError(f"camera {name}: expected one of {tuple(SIZES)} with pos, quat and fovy")
    return cams


def add_to_spec(spec: mujoco.MjSpec, cams: dict) -> None:
    spec.visual.global_.offwidth = max(w for w, _ in SIZES.values())
    spec.visual.global_.offheight = max(h for _, h in SIZES.values())
    for name, c in cams.items():
        parent = spec.worldbody if name == "front" else spec.body(c.get("body", "gripper"))
        parent.add_camera(name=name, pos=c["pos"], quat=c["quat"], fovy=c["fovy"])


class Renderer:
    """One offscreen renderer per camera at its real image size."""

    def __init__(self, model: mujoco.MjModel, names=("front", "wrist")):
        self._r = {n: mujoco.Renderer(model, SIZES[n][1], SIZES[n][0]) for n in names}

    def render(self, data: mujoco.MjData, name: str) -> np.ndarray:
        r = self._r[name]
        r.update_scene(data, camera=name)
        return r.render()

    def close(self) -> None:
        for r in self._r.values():
            r.close()
