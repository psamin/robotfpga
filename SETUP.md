# Setup

Everything needed to rebuild the physical setup and run the software. Values come from the files in
this repo, the SmolVLA checkpoint config, and measurements reported in issues. Anything not measured
is marked **UNKNOWN**; don't fill it in by guessing.

## 1. Task

An SO-101 arm picks up a medicine bottle and places it on a plate. The task text the policy is given
is `"Pick up the medicine bottle and place it on the plate"`.

| Item | Value |
|---|---|
| Bottle, plate: size, position, table layout | **Not measured.** Positions estimated from the demo joint data (bottle about 25 cm out, 9 cm left of the base; plate about 32 cm out, 21 cm right): [docs/smolvla-post-training.md](docs/smolvla-post-training.md). Plate about 19 cm across (12 cm inside the patterned rim), cap about 2.7 cm, from the front video. Bottle height, body diameter and mass unknown |
| Episode length | up to 20 s (`EPISODE_S` in `robot/toolkit/env.sh`) |

## 2. Hardware

| Item | What | Notes |
|---|---|---|
| Follower arm | SO-101, Feetech STS3215 servos, LeRobot id `my_follower` | calibrated: `robot/calibration/so_follower_my_follower.json` |
| Leader arm | SO-101 leader, LeRobot id `my_leader` | calibrated: `robot/calibration/so_leader_my_leader.json` |
| Front camera | Logitech C920 looking down at the table from above, 640×480 @ 30 fps | OpenCV index 0 on psamin's Mac |
| Wrist camera | IMX307 module in a 3D-printed mount (`robot/wrist_camera_mount_IMX307_38mm.stl`), 480×640 @ 30 fps, rotated 90° | OpenCV index 1 |
| FPGA board | AMD Kria KR260 (K26 SOM, part `xck26-sfvc784-2LV-c`, 4 GB DDR4) | §6 |
| Controller computers | (1) psamin's MacBook runs the robot toolkit (USB serial). (2) The board owner's Windows 11 laptop: follower COM3, leader COM5, both cameras, Ethernet to the KR260 | the calibration in `robot/calibration/` is installed on (2) and matches both arms' servo EEPROM |
| Training compute | GT Futurama cluster (Slurm) | §5 |

**Robot units (LeRobot).** Joints `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper`, in that order everywhere. Body joints are in degrees, `(raw − mid) · 360 / 4095`, with `mid`
the centre of the calibrated range, so 0° is mid-travel. The gripper is 0–100 across its calibrated
range.

**Camera settings.** `robot/toolkit/camera_presets.py lock` freezes the C920's focus (40) and both cameras'
white balance (front 3066 K, wrist 4000 K). Exposure stays automatic, because manual exposure doesn't
hold on macOS.

**Motion settings** (`robot/toolkit/presets/rollout_settings.json`): servo P 16, I 0, D 32; target
low-pass 0.7; at most 10° per step; servos run under a 50% torque limit. On stop, the arm glides back
to its start pose, then motors turn off.

## 3. Data and policy

| Item | Value |
|---|---|
| Dataset | [flyingturtleboop/so101_medicine_bottle_pickplace](https://huggingface.co/datasets/flyingturtleboop/so101_medicine_bottle_pickplace): LeRobot v3.0, 15 episodes, 7,824 frames, 30 fps, cameras `front` + `wrist`. Episodes 11 and 14 are excluded from training (bad starts) |
| Policy | SmolVLA fine-tuned from `lerobot/smolvla_base`: 10k steps, batch 64, H200, final loss 0.024, action expert only (vision encoder frozen) |
| Checkpoint | `~/models/smolvla-so101_medicine_bottle_pickplace-9604/` on the cluster (866 MB, not in git). 450,046,176 parameters, stored BF16 + FP32 |
| Inputs | `observation.images.camera1` = front, `camera2` = wrist (the rename used in training); images padded and resized to 512×512; 6 joint states, mean/std normalized; the task text |
| Output | 50-step action chunk of 6 joint targets, 10 flow-matching steps per chunk |
| LeRobot | fork [psamin/lerobot](https://github.com/psamin/lerobot) at `e624f3f7` (0.6.2) |

The fine-tuned checkpoint is the project's only policy. Any approved post-training writes a new
checkpoint and leaves this one untouched.

## 4. Real-arm toolkit (`robot/toolkit/`)

A copy of `so101/` from [psamin/roboticsexp](https://github.com/psamin/roboticsexp) at `4d8baf3`.
[robot/README.md](robot/README.md) lists the commands; [robot/toolkit/README.md](robot/toolkit/README.md)
explains each script and the problems already solved.

- `env.sh` expects the toolkit at `~/so101` and LeRobot at `~/Documents/GitHub/lerobot`. Edit `SO101=`
  and `LEROBOT=` on another computer.
- Find this machine's serial ports with `lerobot-find-port`, and camera indexes with
  `lerobot-find-cameras opencv`.
- To reuse the calibration, copy `robot/calibration/*.json` into
  `~/.cache/huggingface/lerobot/calibration/` (see robot/README.md).

## 5. Cluster (GT Futurama)

- Connect GT VPN (GlobalProtect), then SSH to the login node with your GT account. Login nodes only
  submit jobs; all compute goes through `sbatch`.
- Every job sources `cluster/env.sh`. It remaps the home path (`/nethome` is a broken symlink on compute
  nodes), puts caches and venvs in `/dev/shm/robotfpga-<job>`, and syncs the uv environment.
- Use GPU nodes even for CPU-only work: CPU-only nodes failed "user env retrieval". `general` has
  H100, H200 ×4, L40S ×4 and L4 ×4 (1-day limit); `techfee` has a 3-day limit.

| Job | Does |
|---|---|
| `cluster/smolvla_fpga.sbatch` | the engine C model, then the fine-tuned SmolVLA with every linear layer computed by the engine code, compared with PyTorch FP32 on real frames, then the dumped real layers. `sbatch --export=ALL,MODEL=$HOME/models/smolvla-so101_medicine_bottle_pickplace-9604 cluster/smolvla_fpga.sbatch` |
| `cluster/test.sbatch` | lint + engine C model |
| `robot/toolkit/cluster/train_smolvla.sbatch`, `serve_policy.sbatch` | fine-tune SmolVLA (needs owner approval) / serve it for `policy.sh` |

## 6. KR260

To run the policy on the board: [docs/run-smolvla-on-kr260.md](docs/run-smolvla-on-kr260.md).

Reported by the board owner, 2026-10-08/09:

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.4 for Kria, kernel `5.15.0-1027-xilinx-zynqmp`, aarch64 |
| Python / PYNQ | Python 3.10.12, Kria-PYNQ (`sudo bash install.sh -b KR260`), `pynq 3.0.1`, JupyterLab on port 9090 |
| Network | KR260 Ethernet (J10D) → controller laptop; the laptop shares its Wi-Fi (Windows ICS). SSH user `ubuntu` |
| Overlay | the full-precision engine is **not built yet** (Q12) |

**Building the engine** (Vivado / Vitis HLS **2022.2**, Linux or Windows):

1. `make -C fpga/hls test`: C model and test cases.
2. `cd fpga/hls && vitis_hls -f run_float_hls.tcl`: C sim, synthesis at 300 MHz, IP export.
3. Block design and bitstream: `vivado -mode batch -source build_bd.tcl` (in `fpga/hls/`). It applies the
   KR260 preset, wires `hp0..hp3` to S_AXI_HP0..HP3_FPD and `io` to S_AXI_HPC0_FPD (all 128-bit), sets the
   PL clock to 300 MHz, and writes `build/float.bit` + `build/float.hwh` plus timing and utilization
   reports. The script is untested (no Vivado was available when it was written).
4. On the board, from the repo root (copy `float.bit` + `float.hwh` next to each other):
   - `python3 fpga/board/check_engine.py --bit float.bit build/stream/float`: every engine test case vs the
     float64 reference, plus the achieved weight bandwidth (target 16.2 GB/s).
   - `python3 fpga/board/run_smolvla.py --bit float.bit --model <checkpoint dir> --batches <frames dir>`:
     the fine-tuned SmolVLA with every linear layer on the FPGA, replaying real frames that
     `cluster/smolvla_fpga.sbatch` dumps to `runs/smolvla-fpga/batches/`. It prints the difference from the
     PyTorch FP32 reference and the time per 50-action chunk. The arm is not touched.
   - The board needs PyTorch and the LeRobot fork for SmolVLA (aarch64). Weights are packed into DDR via
     PYNQ's CMA; check `grep Cma /proc/meminfo` and raise the `cma=` boot argument if allocation fails
     (needed size UNKNOWN until measured; SmolVLA's linear weights are 0.71 GB as BF16).

Design, verification and the latency model: [docs/smolvla-fpga.md](docs/smolvla-fpga.md).
