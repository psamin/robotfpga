"""Save the follower's pose before a policy run, and glide back to it afterwards.

  home.py save    read the current joint positions (motors untouched) into presets/home.json
  home.py go      glide from wherever the arm is to presets/home.json over GLIDE_S seconds,
                  then turn the motors off. Expects the motors to still be on from the policy
                  run (robot_client started with --robot.disable_torque_on_disconnect=false),
                  so the arm never drops mid-air.

Talks to the motor bus directly instead of SOFollower.connect(), which turns torque off while it
reconfigures the motors and would let the arm sag. Start policy runs with the arm at rest: that
is where it returns, and where the motors are switched off.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

PORT = "/dev/tty.usbmodem5B790811541"
CALIBRATION = HF_LEROBOT_CALIBRATION / "robots/so_follower/my_follower.json"
HOME = Path(__file__).parent / "presets/home.json"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GLIDE_S = 3.0
HZ = 30


def bus():
    cal = json.loads(CALIBRATION.read_text())
    b = FeetechMotorsBus(PORT, {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper"
                                         else MotorNormMode.DEGREES) for i, n in enumerate(NAMES)},
                         calibration={n: MotorCalibration(**cal[n]) for n in NAMES})
    b.connect()
    return b


def save():
    b = bus()
    pos = b.sync_read("Present_Position")
    b.disconnect(disable_torque=False)
    HOME.parent.mkdir(exist_ok=True)
    HOME.write_text(json.dumps({n: round(float(pos[n]), 2) for n in NAMES}))
    print(f"start pose saved: {json.loads(HOME.read_text())}")


def go():
    target = json.loads(HOME.read_text())
    b = bus()
    try:
        start = b.sync_read("Present_Position")
        b.sync_write("Goal_Position", start)  # hold where it is before enabling torque
        b.enable_torque()
        a = np.array([start[n] for n in NAMES]); z = np.array([target[n] for n in NAMES])
        print(f"returning home over {GLIDE_S:.0f} s (largest move {np.abs(z - a).max():.0f} deg)")
        steps = int(GLIDE_S * HZ)
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)  # ease in and out
            b.sync_write("Goal_Position", {n: float(v) for n, v in zip(NAMES, a + s * (z - a))})
            time.sleep(1 / HZ)
        time.sleep(0.8)  # let it settle on the target
        end = b.sync_read("Present_Position")
        print(f"home: max error {max(abs(end[n] - target[n]) for n in NAMES):.1f} deg; motors off")
    finally:
        b.disconnect(disable_torque=True)


if __name__ == "__main__":
    {"save": save, "go": go}[sys.argv[1] if len(sys.argv) > 1 else "save"]()
