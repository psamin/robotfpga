# SO-101 hardware files

Copies of the real-arm calibration and setup files, so the setup can be rebuilt from this repo alone.
Every file is a copy of the source listed, except the toolkit changes under "Changed since the copy".
See [SETUP.md](../SETUP.md) §2 for how they are used.

| File | What | Copied from (psamin's Mac, 2026-10-09) |
|---|---|---|
| `calibration/so_follower_my_follower.json` | LeRobot calibration of the follower arm (`--robot.id=my_follower`): per joint motor id, drive mode, homing offset, range min/max in raw counts (0–4095) | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json` |
| `calibration/so_leader_my_leader.json` | Same for the leader arm (`--teleop.id=my_leader`) | `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json` |
| `toolkit/` | The complete real-arm toolkit: teleop, recording, replay, rollouts, policy serving, SmolVLA training jobs, camera lock, home pose, voice/arrow-key controls. Every git-tracked file plus all presets | [psamin/roboticsexp](https://github.com/psamin/roboticsexp/tree/4d8baf3eba82a4cd5ff067e4a09f47e95f7c05e7/so101) `so101/` @ `4d8baf3` (session logs excluded) |
| `calibration/reference_frames/front.jpg`, `wrist.jpg` | First frame of training episode 0 at the policy's resolution (front 640x480; wrist 480x640 after the 90° rotation): what the cameras must look like | the dataset, episode 0 |
| `wrist_camera_mount_IMX307_38mm.stl` | 3D-printable wrist camera mount for the SO-101 (IMX307 camera module, 38 mm board) | `~/Downloads/SO101_wrist_IMX307_38mm.stl` |

## The real-arm toolkit (`toolkit/`)

Its own [README](toolkit/README.md) is the full guide; [VOICE.md](toolkit/VOICE.md) lists
the voice and arrow-key controls. Summary:

| Task | Command (run from the toolkit folder) |
|---|---|
| Lock camera focus + white balance (once per session, nothing else running) | `$PY camera_presets.py lock` |
| Teleoperate (leader → follower) | `./teleop.sh` (`VIEW=true` for the live viewer) |
| Record episodes | `./collect.sh live N` or `./collect.sh replay N` |
| Train SmolVLA on the cluster | `DATASET=<hf id> STEPS=10000 sbatch toolkit/cluster/train_smolvla.sbatch` |
| Serve a policy on the cluster | `sbatch toolkit/cluster/serve_policy.sbatch` |
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
(see [SETUP.md](../SETUP.md) §9).

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

## Running the best policy

The best policy for the pick-and-place task is the GitHub release
[policy-smolvla-9604](https://github.com/psamin/robotfpga/releases/tag/policy-smolvla-9604) (checkpoint,
replay batches, checksums). On the cluster it is `~/models/smolvla-so101_medicine_bottle_pickplace-9604`,
which `policy.sh` uses by default.

1. **Cameras.** OpenCV index 0 must be the C920 (front, top-down) and 1 the wrist camera. An iPhone
   (Continuity Camera) or another webcam can take an index: turn it off, then check with
   `lerobot-find-cameras opencv`.
2. **Framing.** `VIEW=true ./teleop.sh` opens LeRobot's rerun viewer. Match
   `calibration/reference_frames/front.jpg`: the arm base fully visible at top center, the far table edge
   about 1/5 down the frame. On 2026-10-15 the front camera was tilted about 100 px (20% of the frame)
   away from the arm compared with training.
3. **Lighting.** Run `$PY camera_presets.py lock` again whenever the lighting changes; it lets focus and
   white balance settle, then freezes them. The demos were recorded at front focus 40 / white balance
   3066 K, wrist 4000 K (`presets/latest.json`). The 2026-10-15 locks, lamps on and lamps off, are kept as
   `presets/20261015-213706.json` and `presets/20261015-215648.json` (front 2000 K both times).
4. **Run.** `./policy.sh start` (arm at rest), `./policy.sh stop`. `KEEP_SERVER=1` keeps the server for
   the next try.

## Changed since the copy

| File | Change |
|---|---|
| `toolkit/env.sh` | Puts the LeRobot venv on `PATH`, so `VIEW=true ./teleop.sh` finds the rerun viewer ("Failed to find Rerun Viewer executable in PATH" otherwise) |
| `toolkit/policy.sh` | Defaults to the best checkpoint, not the newest folder in `~/models`; starts the server on the L4 node (`SERVE_ARGS`), whose SSH login the tunnel needs |
| `toolkit/cluster/serve_policy.sbatch` | Comment: which GPU nodes accept SSH |
| `toolkit/presets/20261015-*.json` | Camera locks from 2026-10-15 |
