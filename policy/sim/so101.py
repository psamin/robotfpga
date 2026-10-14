"""SO-101 arm model (MuJoCo Menagerie copy) and the LeRobot-to-sim joint conversion.

The MJCF and its 18 MB of meshes live at the git tag `archive/cube-sorting`. They are extracted on
first use into a local cache instead of being committed again (CONTRIBUTING.md: no large binaries).

Joint conversion (Q16, not verified on the arm): LeRobot reports the five body joints in degrees,
0 at the middle of each joint's calibrated travel (SETUP.md §2). The sim joints are assumed to read
the same angle in radians with the same sign. The demo data supports this for shoulder_lift,
elbow_flex and wrist_flex (every grasp and release lands at or above table height), and the front
video supports the sign of shoulder_pan (policy/sim/demo_video.py).

Gripper (Q16, derived, not measured on the arm): the 0-100 reading is linear in raw counts over the
calibrated range (robot/calibration/), 137.8 deg for this arm. Reading 0 is the calibrated fully
closed position, taken as the sim angle where the jaw pads meet. The sim joint allows 110 deg, so
readings above about 81 clip. Checks: the demos' rest reading (2.0) gives a 3.6 mm gap; the reading
while holding the cap (11.5) gives 21 mm, against 27 mm estimated from the front video.
"""

import json
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
CALIBRATION = REPO / "robot" / "calibration" / "so_follower_my_follower.json"
GRIPPER_RANGE = (-0.174533, 1.7453292)  # sim joint limits (rad), from so101.xml
PADS_TOUCH = -0.205  # sim gripper angle (rad) where the jaw pads meet; tests/test_so101.py recomputes it
FIXED_PADS = tuple(f"fixed_jaw_box{i}" for i in (4, 5, 6, 7))
MOVING_PADS = ("moving_jaw_box2", "moving_jaw_box3")


def tag_available() -> bool:
    return (
        subprocess.run(["git", "rev-parse", "-q", "--verify", TAG], cwd=REPO, capture_output=True).returncode
        == 0
    )


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


def gripper_rad_per_unit() -> float:
    cal = json.loads(CALIBRATION.read_text())["gripper"]
    return (cal["range_max"] - cal["range_min"]) * 2 * np.pi / 4096 / 100


def gripper_to_sim(reading):
    """LeRobot gripper reading (0 closed .. 100 open) -> sim gripper angle (rad), clipped to the joint."""
    return np.clip(PADS_TOUCH + np.asarray(reading, float) * gripper_rad_per_unit(), *GRIPPER_RANGE)


def sim_to_gripper(angle):
    """Sim gripper angle (rad) -> LeRobot gripper reading; inverse of gripper_to_sim inside the clip."""
    return (np.asarray(angle, float) - PADS_TOUCH) / gripper_rad_per_unit()


def jaw_gap(model: mujoco.MjModel, angle: float) -> float:
    """Distance (m) between the fixed and moving jaw pads at a sim gripper angle; negative if they overlap."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("gripper").id]] = angle
    mujoco.mj_forward(model, data)
    pads = [(model.geom(a).id, model.geom(b).id) for a in FIXED_PADS for b in MOVING_PADS]
    return min(mujoco.mj_geomDistance(model, data, a, b, 0.5, np.zeros(6)) for a, b in pads)
