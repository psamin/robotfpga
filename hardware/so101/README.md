# SO-101 hardware files

Copies of the real-arm calibration and setup files, so the setup can be rebuilt from this repo alone.
Every file is an unmodified copy; the source is listed. See [SETUP.md](../../SETUP.md) §2 for how
they are used.

| File | What | Copied from (psamin's Mac, 2026-10-09) |
|---|---|---|
| `calibration/so_follower_my_follower.json` | LeRobot calibration of the follower arm (`--robot.id=my_follower`): per joint motor id, drive mode, homing offset, range min/max in raw counts (0–4095) | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json` |
| `calibration/so_leader_my_leader.json` | Same for the leader arm (`--teleop.id=my_leader`) | `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json` |
| `roboticsexp/env.sh` | Serial ports, camera config (OpenCV index, size, fps, rotation), dataset name, task, video encoder | [psamin/roboticsexp](https://github.com/psamin/roboticsexp/tree/4d8baf3eba82a4cd5ff067e4a09f47e95f7c05e7/so101) `so101/env.sh` @ `4d8baf3` |
| `roboticsexp/presets/home.json` | Home pose in LeRobot units (degrees; gripper 0–100) | same repo, `so101/presets/home.json` |
| `roboticsexp/presets/latest.json` | Locked camera settings: C920 focus 40, white balance front 3066 K / wrist 4000 K | same repo, `so101/presets/latest.json` |
| `roboticsexp/presets/rollout_settings.json` | Servo P/I/D 16/0/32, target low-pass 0.7, max 10° per step | same repo, `so101/presets/rollout_settings.json` |
| `wrist_camera_mount_IMX307_38mm.stl` | 3D-printable wrist camera mount for the SO-101 (IMX307 camera module, 38 mm board) | `~/Downloads/SO101_wrist_IMX307_38mm.stl` |

## Using the calibration on another computer

LeRobot looks for calibration by robot id under `~/.cache/huggingface/lerobot/calibration/`. To
reuse these files instead of recalibrating:

```sh
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
cp calibration/so_follower_my_follower.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json
cp calibration/so_leader_my_leader.json ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json
```

Then pass `--robot.id=my_follower` / `--teleop.id=my_leader`. The values are only valid for these
two physical arms; the serial port names in `env.sh` are specific to psamin's Mac and differ on any
other computer (find them with `lerobot-find-port`).
