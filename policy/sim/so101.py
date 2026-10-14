"""SO-101 arm model (MuJoCo Menagerie copy) and the LeRobot-to-sim joint conversion.

The MJCF and its 18 MB of meshes live at the git tag `archive/cube-sorting`. They are extracted on
first use into a local cache instead of being committed again (CONTRIBUTING.md: no large binaries).

Joint conversion (Q16, not verified on the arm): LeRobot reports the five body joints in degrees,
0 at the middle of each joint's calibrated travel (SETUP.md §2). The sim joints are assumed to read
the same angle in radians with the same sign. The demo data supports this for shoulder_lift,
elbow_flex and wrist_flex (every grasp and release lands at or above table height), and the front
video supports the sign of shoulder_pan (policy/sim/demo_video.py). The gripper's 0-100 scale is not
converted.
"""

import os
import subprocess
import tarfile
from pathlib import Path

import mujoco
import numpy as np

TAG = "archive/cube-sorting"
SRC = "armlab/sim/assets/so101"
REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("ROBOTFPGA_CACHE", Path.home() / ".cache" / "robotfpga"))
BODY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def model_dir() -> Path:
    """Directory holding so101.xml and assets/, extracted from TAG on first use."""
    out = CACHE / "so101"
    if (out / "so101.xml").exists():
        return out
    tar = subprocess.run(["git", "archive", TAG, SRC], cwd=REPO, capture_output=True, check=True).stdout
    tmp = CACHE / "so101.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tar_path = tmp / "so101.tar"
    tar_path.write_bytes(tar)
    with tarfile.open(tar_path) as t:
        t.extractall(tmp, filter="data")
    (tmp / SRC).rename(out)
    tar_path.unlink()
    return out


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(model_dir() / "so101.xml"))


def lerobot_to_sim(state) -> np.ndarray:
    """LeRobot joint state (degrees for the 5 body joints, gripper ignored) -> sim radians (5)."""
    return np.radians(np.asarray(state, dtype=np.float64)[..., :5])
