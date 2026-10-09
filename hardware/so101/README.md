# SO-101 hardware files

Copies of the real-arm calibration and setup files, so the setup can be rebuilt from this repo alone.
Every file is an unmodified copy; the source is listed. See [SETUP.md](../../SETUP.md) §2 for how
they are used.

| File | What | Copied from (psamin's Mac, 2026-10-09) |
|---|---|---|
| `calibration/so_follower_my_follower.json` | LeRobot calibration of the follower arm (`--robot.id=my_follower`): per joint motor id, drive mode, homing offset, range min/max in raw counts (0–4095) | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json` |
| `calibration/so_leader_my_leader.json` | Same for the leader arm (`--teleop.id=my_leader`) | `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json` |
| `roboticsexp/` | The complete real-arm toolkit: teleop, recording, replay, rollouts, policy serving, SmolVLA training jobs, camera lock, home pose, voice/arrow-key controls. Every git-tracked file plus all presets | [psamin/roboticsexp](https://github.com/psamin/roboticsexp/tree/4d8baf3eba82a4cd5ff067e4a09f47e95f7c05e7/so101) `so101/` @ `4d8baf3` (session logs excluded) |
| `wrist_camera_mount_IMX307_38mm.stl` | 3D-printable wrist camera mount for the SO-101 (IMX307 camera module, 38 mm board) | `~/Downloads/SO101_wrist_IMX307_38mm.stl` |

## The real-arm toolkit (`roboticsexp/`)

Its own [README](roboticsexp/README.md) is the full guide; [VOICE.md](roboticsexp/VOICE.md) lists
the voice and arrow-key controls. Summary:

| Task | Command (run from the toolkit folder) |
|---|---|
| Lock camera focus + white balance (once per session, nothing else running) | `$PY camera_presets.py lock` |
| Teleoperate (leader → follower) | `./teleop.sh` (`VIEW=true` for the live viewer) |
| Record episodes | `./collect.sh live N` or `./collect.sh replay N` |
| Train SmolVLA on the cluster | `DATASET=<hf id> STEPS=10000 sbatch cluster/train_smolvla.sbatch` |
| Serve a policy on the cluster | `sbatch cluster/serve_policy.sbatch` |
| Run a policy on the arm | `./policy.sh start` / `./policy.sh stop` |
| Rollout session with success logging | `./rollout.sh` (`begin` / `rest` / `success` / `fail` / `quit`) |
| Check a policy server without moving the arm | `$PY test_policy_server.py` |
| Save the pose and glide back to it | `home.py` |

**Paths the scripts assume:**
- `env.sh` expects the toolkit at `~/so101`. On psamin's Mac that's a symlink to roboticsexp's
  `so101/`. On another computer, either symlink `~/so101` to this folder or edit `SO101=` in `env.sh`.
- It also expects LeRobot at `~/Documents/GitHub/lerobot` with its `.venv`. Use
  [psamin/lerobot](https://github.com/psamin/lerobot) at `e624f3f7`, or edit `LEROBOT=`.
- Serial ports and OpenCV camera indexes are specific to psamin's Mac. Find the right ones with
  `lerobot-find-port` and `lerobot-find-cameras opencv`.
- `camera_presets.py` needs `uvc-util` at `~/.local/bin/uvc-util` (macOS).

The toolkit runs SmolVLA today; it is not yet wired to the cube-sorting TinyPolicy or the FPGA
(see [SETUP.md](../../SETUP.md) §9).

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
