"""LeRobot's async robot_client with command damping: a first-order low-pass filter on every
joint target before it reaches the motors.

    target_t = target_{t-1} + SMOOTH_ALPHA * (policy_action_t - target_{t-1})

SMOOTH_ALPHA=1 is no filtering; smaller is smoother and laggier. At 30 Hz the default 0.5 is a
time constant of about 50 ms (0.7: about 28 ms). 0.5 plus softer servo gains left the arm stuck. Servo-side PID
(P / I / D) is set with LeRobot's own --robot.position_{p,i,d}_coefficient flags.
Takes exactly the same arguments as `python -m lerobot.async_inference.robot_client`.
"""

import os

import lerobot.async_inference.robot_client as rc
from lerobot.robots.so_follower.so_follower import SOFollower

ALPHA = float(os.environ.get("SMOOTH_ALPHA", 0.7))
_raw_send_action = SOFollower.send_action


def _damped_send_action(self, action):
    prev = getattr(self, "_damped_target", None)
    if prev is None:
        smoothed = dict(action)
    else:
        smoothed = {k: (prev[k] + ALPHA * (v - prev[k]) if k in prev and k.endswith(".pos") else v)
                    for k, v in action.items()}
    self._damped_target = smoothed
    return _raw_send_action(self, smoothed)


SOFollower.send_action = _damped_send_action

if __name__ == "__main__":
    print(f"[smooth_client] command damping alpha={ALPHA}", flush=True)
    rc.register_third_party_plugins()
    rc.async_client()
