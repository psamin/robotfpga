# Setup: the complete physical and software setup

This is the single reference for how everything is set up: the physical scene, the robot interface,
the policy, the training pipeline, the FPGA build and the board. Every number comes from a file on
`main` (the source is named next to it). Anything not built or not measured is marked **NOT BUILT**
or **UNKNOWN**; do not fill those in by guessing. Open questions live in [QUESTIONS.md](QUESTIONS.md).

Last verified on 2026-10-09 against:
- `main`;
- the local LeRobot calibration files;
- psamin/lerobot `e624f3f7`;
- psamin/roboticsexp `4d8baf3`.

---

## 1. Status at a glance

| Part | Status | Evidence |
|---|---|---|
| Simulated scene, expert, data, training, QAT, eval | **BUILT, run** | `armlab/`, [plans/tiny-policy-log.md](plans/tiny-policy-log.md) |
| Trained int8 policy (v3r2) | **BUILT** — sim Stage B 91%, Stage C 90% | [handoff/v3r2/README.md](handoff/v3r2/README.md) |
| Bit-exact int8 reference | **BUILT** | `ref/intref.py` |
| HLS kernel C model vs golden vectors | **PASS 100/100** | `handoff/v3r2/results/hls_csim.txt` |
| HLS IP synthesized for the KR260 | **UNKNOWN** (not reported) | QUESTIONS.md Q3 |
| Board overlay `policy.bit` + `policy.hwh` | **NOT BUILT** | [board/demo/README.md](board/demo/README.md) §3 |
| KR260 booted with Ubuntu + PYNQ | **UNKNOWN** | QUESTIONS.md Q1, Q2 |
| Laptop ↔ board network bridge | **NOT BUILT** (contract only) | §7 |
| Real SO-101 follower + leader arms | **BUILT: connected and calibrated** with LeRobot (`my_follower`, `my_leader`) | §2.1 |
| Real cameras | **BUILT:** front Logitech C920 (top view) + wrist camera. **Not** matched to the sim camera | §2.2, §3.4 |
| Real-arm controller | **BUILT, separate repo:** teleop, recording and rollout in [psamin/roboticsexp `so101/`](https://github.com/psamin/roboticsexp/tree/4d8baf3eba82a4cd5ff067e4a09f47e95f7c05e7/so101). Drives SmolVLA, **not** TinyPolicy or the FPGA | §2.3 |
| TinyPolicy on the real arm (`robot/so101.py` here) | **NOT BUILT** | §9 |
| Sim radians ↔ LeRobot degrees mapping | **UNVERIFIED** (formula known, signs/offsets not measured) | §4.1, QUESTIONS.md Q5 |
| SmolVLA | **BUILT on a different task** (real medicine-bottle pick-and-place, job 9604); not trained on the cube-sorting task | §2.3 |

---

## 2. Hardware list

| Item | What exactly | Status |
|---|---|---|
| Robot arm | SO-101 (TheRobotStudio / LeRobot), 5 arm joints + 1 gripper joint, Feetech STS3215 servos | Owned by psamin; **not connected** |
| Leader (teleop) arm | SO-101 leader, LeRobot id `my_leader` | **Connected and calibrated** |
| FPGA board | AMD Kria **KR260** Robotics Starter Kit (K26 SOM, part `xck26-sfvc784-2LV-c`) | **Boot status UNKNOWN** (Q1) |
| Cameras | `front`: Logitech C920 looking down from above the table, 640×480 @ 30 fps; `wrist`: on the gripper, 480×640 @ 30 fps (rotated 90°) | **Mounted and used for recording**; front pose not measured |
| FPGA workstation | Linux or Windows PC with AMD Vivado/Vitis HLS **2022.2** (does not run on macOS) | A teammate's PC has the tools (reported by psamin) |
| Controller computer | psamin's MacBook (macOS, Apple M3 Pro) runs the real-arm loop over USB serial today | Used by `roboticsexp`; the Ethernet link to the board is not built |
| Training compute | GT Futurama cluster (Slurm, `general` partition, L4/L40 nodes); RunPod H100 available | Used for all training so far |

### 2.1 Arm calibration (LeRobot)

- **Files (local, not in git):**
  - follower: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json`
  - leader: `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json`
- **Format:** one entry per joint (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`, `gripper`), each with:
  - the Feetech motor `id`, numbered 1–6 in that order;
  - `drive_mode` (0 on every joint);
  - `homing_offset`;
  - `range_min` / `range_max` in raw encoder counts, 0–4095 per turn.
- **Ranges:** `wrist_roll` is calibrated over the full 0–4095. The other joints have limited ranges
  set by hand during `lerobot-calibrate`.
- **Software:** LeRobot fork [psamin/lerobot](https://github.com/psamin/lerobot) at `e624f3f7`
  (version 0.6.2), in `~/Documents/GitHub/lerobot/.venv`. Robot types `so101_follower` and
  `so101_leader`; the follower runs with `use_degrees=True`.
- **To recalibrate:** `lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=my_follower`,
  and the same with `--teleop.type=so101_leader --teleop.id=my_leader` for the leader.
  This overwrites the files above.

### 2.2 Ports and cameras (from `so101/env.sh` in roboticsexp)

| Setting | Value |
|---|---|
| Follower serial port | `/dev/tty.usbmodem5B790811541` |
| Leader serial port | `/dev/tty.usbmodem5B901030671` |
| `front` camera | OpenCV index 0, 640×480, 30 fps, warmup 3 s |
| `wrist` camera | OpenCV index 1, 480×640, 30 fps, rotation 90°, warmup 3 s |

OpenCV indexes can change after replugging; check with `lerobot-find-cameras opencv`.

Camera settings: `camera_presets.py lock` freezes the C920's focus and both cameras' white balance.
Exposure stays automatic, because manual exposure doesn't hold on macOS. Latest preset: focus 40,
front white balance 3066 K, wrist 4000 K.

### 2.3 Real-arm controller and data (psamin/roboticsexp `so101/`)

Source: [https://github.com/psamin/roboticsexp/tree/4d8baf3eba82a4cd5ff067e4a09f47e95f7c05e7/so101](https://github.com/psamin/roboticsexp/tree/4d8baf3eba82a4cd5ff067e4a09f47e95f7c05e7/so101), commit `4d8baf3`. It's the working real-arm stack, run from the Mac.

| Piece | What |
|---|---|
| `teleop.sh` | leader → follower teleoperation with locked cameras |
| `collect.sh live N` / `replay N` | record episodes (replay mode: teleop a demo, then the follower replays it while the cameras record) |
| `policy.sh start` / `stop` | run a policy served on a cluster GPU through an SSH tunnel; `stop` glides back to the start pose, then motors off |
| `rollout.sh` | rollout sessions with success/fail logging to `logs/rollouts-*.csv` |
| `home.py` | save the pose before a run and glide back to it afterwards |
| Motion settings | `presets/rollout_settings.json`: P 16, I 0, D 32, `SMOOTH_ALPHA` 0.7 (low-pass on targets), `MAX_STEP` 10°/step; servos run under a 50% torque limit |
| Home pose | `presets/home.json` (LeRobot degrees; gripper in 0–100) |
| Video encoder | `h264_videotoolbox` (the default `libsvtav1` dropped the loop to 4–10 Hz) |

**Real dataset:**
- [flyingturtleboop/so101_medicine_bottle_pickplace](https://huggingface.co/datasets/flyingturtleboop/so101_medicine_bottle_pickplace):
  LeRobot format v3.0, 15 episodes, 7,824 frames, 30 fps, robot type `so_follower`.
- Features: `observation.state` and `action` (6 joint positions), plus `observation.images.front`
  and `observation.images.wrist` videos.
- Task: "Pick up the medicine bottle and place it on the plate". Episodes 11 and 14 are excluded
  from training.
- A local copy is in `~/.cache/huggingface/lerobot/flyingturtleboop/`.

**This is a different task from the simulated cube sorting.** The cube-sorting TinyPolicy has never
run on this stack.

---

## 3. Physical scene

The simulation in [armlab/sim/assets/block_sort.xml](armlab/sim/assets/block_sort.xml) is the
specification for the physical scene. The trained policy has only seen this scene, so the real
one must match it as closely as possible.

### 3.1 Coordinate frame

- **Origin:** the SO-101 base frame from the MuJoCo Menagerie model (`body name="base"` at `0 0 0`
  in `so101.xml`). Where exactly this point sits on the physical base plate is **UNKNOWN (not
  measured)**. Measure it once with the arm in a known pose before laying out the scene.
- **Axes:** right-handed, metres.
  - **+x** = straight ahead of the arm (the direction the arm points when `shoulder_pan` = 0).
  - **+y** = the arm's **left** (standing behind the arm, looking along +x).
  - **+z** = up.
- **Table top is z = 0**, the same plane the arm base stands on.

### 3.2 Table

| Property | Value | Source |
|---|---|---|
| Top surface | z = 0 | `geom name="table"` |
| Extent | x from −0.20 to +0.50, y from −0.45 to +0.45 (0.70 m × 0.90 m) | box centre (0.15, 0, −0.02), half-sizes 0.35 × 0.45 × 0.02 |
| Colour (sim) | RGB (0.62, 0.55, 0.45): light tan/wood | `material name="table"` |

### 3.3 Objects

| Object | Size | Position | Colour (sim RGB) | Mass / friction (sim) |
|---|---|---|---|---|
| Red cube | 25 mm cube | spawned at random, see below | (0.85, 0.12, 0.10) | 20 g, friction 1.0 |
| Blue cube | 25 mm cube | spawned at random, see below | (0.10, 0.25, 0.85) | 20 g, friction 1.0 |
| Left bin | outer 100 × 100 mm, inner 90 × 90 mm, wall height 35 mm, wall thickness 5 mm, floor 4 mm | centre at (x 0.14, y **+0.20**) | (0.35, 0.37, 0.40): grey | static |
| Right bin | same as left | centre at (x 0.14, y **−0.20**) | grey | static |

**Cube placement** (each episode, [armlab/sim/env.py](armlab/sim/env.py)):

- Cube centres are uniform in x ∈ [0.16, 0.26], y ∈ [−0.09, 0.09].
- The two cubes are at least 0.06 m apart.
- Each cube is rotated about z by a uniform angle in [−45°, +45°].
- Cubes rest flat on the table (centre at z = 0.0125).

This region was chosen because a top-down grasp is reachable there (§4.3).

"Left" and "right" are always from the **arm's** point of view (+y = left). In the camera image
they appear swapped (§3.4).

### 3.4 Camera

| Property | Value |
|---|---|
| Position | (0.50, 0.00, 0.55) m: in front of the arm, centred, 0.55 m above the table |
| Aim point | (0.15, 0.00, 0.00) on the table |
| Pitch | about 57.5° below horizontal (atan(0.55 / 0.35)) |
| Roll | image horizontal axis parallel to the table; image **right = +y = the arm's left bin** |
| Field of view | 46° vertical; the image is square, so 46° horizontal too |
| Output to the policy | 96 × 96 RGB, uint8, HWC |
| How 96 × 96 is made | render 288 × 288, then average each 3 × 3 block (`BlockSortEnv.obs`) |

Source: `camera name="overhead"`, quaternion (0.67528548, 0.20973678, 0.20973678, 0.67528548).

**Real camera: NOT BUILT.**
- Model, lens and mounting are **UNKNOWN**.
- To match the sim, it must reproduce the pose above, then centre-crop to a square with a 46° field
  of view, resize to 288 × 288, and 3 × 3 area-average to 96 × 96.
- Its intrinsic calibration (focal length, distortion) has to be measured to compute that crop.

**Why this view:** a straight-down camera hid the cube under the gripper exactly when alignment
mattered. Success was 13% with it, against 31% with this view, both before DAgger
([plans/tiny-policy-log.md](plans/tiny-policy-log.md) §6).

### 3.5 Lighting

The sim uses one overhead directional light (diffuse 0.6) plus a headlight. During training these
were randomized (§5.2). Real lighting has **not been specified or measured**. Aim for even,
diffuse lighting without hard shadows over the bins.

---

## 4. Robot interface (what the policy reads and writes)

### 4.1 Joints

Joint order is fixed everywhere: the state vector, the action vector and the 6 joint bytes of the
FPGA input.

| Index | Joint | Range (rad), sim | Notes |
|---|---|---|---|
| 0 | `shoulder_pan` | −1.91986 … +1.91986 | positive pan swings the arm toward −y |
| 1 | `shoulder_lift` | −1.74533 … +1.74533 | |
| 2 | `elbow_flex` | −1.69 … +1.69 | |
| 3 | `wrist_flex` | −1.65806 … +1.65806 | |
| 4 | `wrist_roll` | −2.74385 … +2.84121 | |
| 5 | `gripper` | −0.17453 … +1.74533 | **larger = more open** |

Source: actuator `ctrlrange` in [armlab/sim/assets/so101/so101.xml](armlab/sim/assets/so101/so101.xml).

**Gripper values used:**
- Open: 0.6 rad, about a 60 mm jaw gap.
- Closed (command): −0.17 rad.
- Holding a 25 mm cube: it settles near 0.09 rad.

**Rest pose:** (0, −1.74, 1.69, 0.6, 0, 0.5) rad (`REST_Q`).

**Real-arm units (LeRobot, verified in `lerobot/motors/motors_bus.py`):**
- Body joints: `degrees = (raw − mid) · 360 / 4095`, where `mid = (range_min + range_max) / 2`
  from the calibration file. So **0° is the middle of each joint's calibrated travel**.
- Gripper: `0–100`, linear from `range_min` (0) to `range_max` (100).

Observed in the real dataset: body joints span about −108° to +97°, and the gripper 0–92.

**Mapping to sim radians: UNVERIFIED (QUESTIONS.md Q5).**
- The calibrated spans are close to the sim ranges: pan ≈ 233° vs 220°, lift ≈ 211° vs 200°,
  elbow ≈ 196° vs 194°, wrist_flex ≈ 207° vs 190°. That suggests
  `q_sim ≈ s · degrees · π/180` with a per-joint sign `s = ±1`.
- Each sign and any zero offset must be checked by putting the real arm and the sim in the same
  known pose (for example, all joints at 0° and a photo compared with the sim render).
- The gripper mapping `0–100 → −0.1745…1.745 rad` is likewise unmeasured.

Until this is measured, the sim-trained policy cannot drive the real arm.

### 4.2 Normalization

- **State and actions** are normalized per joint to [−1, 1] using the ranges in §4.1:
  `norm = 2·(q − lo)/(hi − lo) − 1`; inverse `q = lo + (norm + 1)/2 · (hi − lo)` (clipped to [−1, 1]).
- **FPGA input packing** (`ref/intref.py` `pack_obs`, 27,658 bytes):
  - The 96 × 96 × 3 image as uint8, row-major HWC: 27,648 bytes.
  - Then 10 int8 aux bytes:
    - 6 joint values: `floor(norm · 127 + 0.5)` clamped to [−127, 127].
    - 4 instruction one-hot bytes: 127 for the active instruction, 0 for the others.
- **FPGA output:** 48 int8 bytes. Each byte divided by 127 is a normalized joint target. Reshape to
  [8, 6]: row = time step (0 = next), column = joint (order in §4.1).

### 4.3 Instructions

| id | Instruction | Text form (for SmolVLA) |
|---|---|---|
| 0 | red → left bin | "put the red cube in the left bin" |
| 1 | red → right bin | "put the red cube in the right bin" |
| 2 | blue → left bin | "put the blue cube in the left bin" |
| 3 | blue → right bin | "put the blue cube in the right bin" |

Workspace limit, measured in sim: the gripper can point straight down only up to about 8 cm above
the table, between 0.13 and 0.28 m from the base. That's why cubes stay in x ∈ [0.16, 0.26] and
the gripper carries cubes at 6.5 cm.

### 4.4 Control loop

| Property | Value | Source |
|---|---|---|
| Control rate | 30 Hz (33.3 ms per tick) | `CONTROL_DT_MS` |
| Physics (sim) | 1/300 s steps, 10 per tick | `SUBSTEPS`, XML `timestep` |
| Policy output | 8 future joint targets (one chunk) | §4.2 |
| Replanning | a new inference every 4 ticks (7.5 inferences/s); the newest chunk replaces the old | `ChunkExecutor(replan=4)` |
| With latency of d ticks | the old chunk keeps playing; the new one starts at index d | [armlab/control/chunking.py](armlab/control/chunking.py) |
| Actuation (sim) | joint position control (Menagerie STS3215 gains: kp 998.22, kv 2.731, force ±2.94) | `so101.xml` |
| Episode limit | 600 ticks (20 s) | `max_steps` |
| Success | named cube's centre inside the named bin's 90 mm inner square, below 4 cm, not touching the robot | `BlockSortEnv.success` |

**Safety on the real arm:**
- **Exists in roboticsexp:** a 10°/step target cap, low-pass damping, a 50% servo torque limit, and
  glide-home plus torque-off on stop.
- **Not built in this repo** (build-spec §7): joint-limit margins, the 250 ms watchdog, the
  workspace box and the torque cut-off key.
- The TinyPolicy controller must have the cap and the glide-home behavior at minimum.

---

## 5. Policy and training

### 5.1 Network (the FPGA contract)

| Layer | Op | In → Out |
|---|---|---|
| conv1–5 | 3×3, stride 2, pad 1, ReLU | 96×96×3 → 48×48×16 → 24×24×32 → 12×12×64 → 6×6×96 → 3×3×128 |
| concat | flatten HWC (1,152) + 10 aux bytes | 1,162 |
| fc6, fc7 | fully connected + ReLU | 1,162 → 256 → 256 |
| fc8 | fully connected | 256 → 48 |

- 565,552 parameters; 9,665,024 multiply-adds per inference.
- Int8 weights total 568,240 bytes (`weights.bin`, including the int32 biases).

### 5.2 How v3r2 was trained (all on the GT cluster)

| Stage | What | Key settings |
|---|---|---|
| Expert | Scripted stateless pick-and-place with top-down IK on the `gripperframe` site | speed 0.22 m/s, hover 6.5 cm, grasp height 1.2 cm, align 12 mm; 399/400 in sim |
| Expert data (v3) | 2,000 episodes, 1,894 kept | DART noise σ ∈ {0, 0.005, 0.01, 0.02}, AR(1) ρ = 0.8; 20% cube moved, 20% instruction swapped |
| Labels | the expert's own next-8-step plan from every state | simulated ahead on a copy of the physics state |
| Domain randomization | every episode | camera ±1 cm, FOV ±1.5°, light ±0.15, colours ±0.08 per channel (cube hue preserved) |
| Behaviour cloning | `armlab.policy.train` | L1 loss on the 8-step chunk; AdamW lr 3e-4, wd 1e-4; batch 256; warmup 1k + cosine; EMA 0.999; DrQ random shift ±4 px + brightness/contrast |
| DAgger | `armlab.policy.dagger`, 1,000 learner episodes per round | β = 0.5, 0.3, …; round 0: 40k steps, later rounds: 30k |
| Selected float model | DAgger round 2: **100/100** Stage B | `~/robotfpga-work/runs/dagger-v3/round2/ckpt_030000.pt` on the cluster (**not in git**) |
| QAT | `armlab.policy.qat_train` | 10k steps, lr 1e-4, distillation weight 0.5, 4,096 calibration frames; power-of-two scales; bit-exact with `ref/intref.py` |
| Export | `handoff/v3r2/` | `weights.bin` SHA-256 `c87abfae0cbedf771a9ef927f39302a43cdcbb954f7090aca4baa6bef350a732` |

DAgger rounds 3–4 on v3 were running when results were last read. Their outcome is **UNKNOWN**
until someone reads `~/robotfpga-work/runs/dagger-v3/history.json`.

### 5.3 Sim results for v3r2 (Wilson 95% CI)

| Model | Stage | Injected latency | Success |
|---|---|---|---|
| float | B | 0 ms | 199/200 [0.97, 1.00] |
| float | C | 0 ms | 196/200 [0.95, 0.99] |
| int8 | B | 0 ms | 182/200 [0.86, 0.94] |
| int8 | C | 0 / 33 / 67 / 100 ms | 90% / 86% / 83% / 84% |
| int8 | C | 133 / 200 / 300 / 500 ms | 78% / 64% / 66% / 51% |

Known gap: int8 is 9 points below float on Stage B, more than the spec's 5-point target. The planned
fix (longer QAT) is **not done**.

**No real-world success numbers exist.** All numbers above are in simulation.

---

## 6. Software setup

### 6.1 Repository layout (`main`)

| Path | What |
|---|---|
| `armlab/sim/` | MuJoCo scene, `BlockSortEnv`, IK, expert, Stage C perturbations |
| `armlab/data/` | data collection (expert / DAgger) and exact replay |
| `armlab/policy/` | TinyPolicy, dataset loader, training, DAgger driver, QAT and export |
| `armlab/control/` | action-chunk executor with latency model |
| `armlab/backends/` | `torch` (float) and `intref` (int8 reference) backends |
| `armlab/eval/` | closed-loop evaluation, Wilson intervals |
| `ref/intref.py` | bit-exact int8 reference (shared contract with the FPGA side) |
| `handoff/v3r2/` | trained int8 weights, manifest, 100 golden vectors, eval results |
| `board/demo/` | FPGA handoff preparation and preflight check |
| `cluster/` | Slurm job scripts |
| `scripts/` | expert eval, rollout video, frame packing for vectors |
| `plans/` | build spec, plans, engineering log, verified references |

### 6.2 Python environment (sim and training)

- **Python 3.12 exactly** (`requires-python >=3.12,<3.13`), managed with `uv`.
- **Pinned:** `mujoco==3.14.0`, `torch>=2.7,<2.12`, `numpy>=2.0`, `imageio[ffmpeg]`. Optional
  `lerobot[dataset]==0.6.1` (the `data` extra). Dev: `pytest`, `ruff`. Exact versions are in `uv.lock`.
- **Install:** `uv sync --python 3.12`.
- **Rendering:** EGL on Linux (set automatically in `armlab/sim/__init__.py`); macOS uses its native
  offscreen renderer.
- **Rule: no training, rollouts or large evaluations on laptops.** They run on the cluster.

### 6.3 Cluster (GT Futurama)

- GT VPN (GlobalProtect) must be connected; then `ssh pi34@planetexpress-login.cc.gatech.edu`.
- Login nodes are for submitting jobs only. Every job sources `cluster/env.sh`, which:
  - remaps `/nethome/<user>` to `/nethome-instruction/<user>` (the home symlink is broken on
    compute nodes);
  - puts the uv cache, venv, Python and torch caches in `/dev/shm/armlab-<jobid>`, deleted on exit;
  - sets `MUJOCO_GL=egl`.
- Work tree on the cluster: `~/robotfpga-work/`. Checkouts: `robotfpga`, `robotfpga-v3`,
  `robotfpga-final`. Data: `datasets/v1`, `v2`, `v3`. Runs: `runs/dagger-v2`, `dagger-v3`,
  `final-v2r2`, `final-v3r2`.
- Jobs (all `-p general`, one GPU):

| Script | Does | Example |
|---|---|---|
| `test.sbatch` | ruff + pytest | `sbatch --gres=gpu:l40:1 cluster/test.sbatch` |
| `collect.sbatch` | expert data | `sbatch --export=ALL,OUT=$HOME/robotfpga-work/datasets/v4,EPI=500 cluster/collect.sbatch` |
| `dagger.sbatch` | train → eval → collect, N rounds | `sbatch --export=ALL,NAME=dagger-v4,ARGS="--expert-data <dir> --rounds 4" cluster/dagger.sbatch` |
| `final.sbatch` | QAT → int8 eval → vectors → HLS C model | `sbatch --export=ALL,NAME=final-x,FLOAT=<ckpt>,DATA="<dirs>",EP=50 cluster/final.sbatch` |
| `eval.sbatch`, `train.sbatch`, `qat.sbatch`, `script.sbatch` | single steps | see the header of each file |

---

## 7. Laptop ↔ board bridge: NOT BUILT

The only thing defined is the contract in [plans/build-spec.md](plans/build-spec.md) §5.6:

- **Laptop → board:** the 27,658-byte input packet (§4.2).
- **Board → laptop:** 48 int8 action bytes + one float32 latency.
- **Transport:** TCP over Ethernet.

**Not specified:** framing, port, byte order of the float32, error codes, timeouts and reconnection.
Proposed ownership (QUESTIONS.md Q4, **not agreed**):
- Software builds `remote_backend.py` (laptop) and the SO-101 controller.
- FPGA builds the board inference service (wrapping `PolicyAccel` in `board/run_policy.py`) and the
  overlay.

Latency budget: under 33 ms end to end (one control tick), target under 5 ms for inference.

---

## 8. FPGA setup

### 8.1 Prepare the files (any computer with Git + Python 3.8+ + NumPy)

```sh
git clone --branch board/81-fpga-handoff https://github.com/psamin/robotfpga.git robotfpga-handoff
cd robotfpga-handoff
python3 board/demo/prepare_handoff.py          # expect: Prepared 262 files
cd build/fpga-handoff
python3 -m pip install numpy
python3 board/demo/handoff_check.py artifacts/standin --reference ref/intref.py --verify-reference
                                                # expect: "preflight": "PASS", 100 calls, 40 layer dumps
```

`artifacts/standin/` holds the **trained v3r2** weights despite the name.

`prepare_handoff.py` takes the files from pinned commits, without merging any branch:

| Content | Commit |
|---|---|
| `hls/` | `3ec8fd3` |
| `handoff/v3r2/` | `5ee0e03` |
| `ref/intref.py` | `2155ad6` |
| `handoff_check.py` | `2b713c6` |
| `run_policy.py` | `7cb91b0` |
| `loopback.py` | `4155f11` |
| `board/m0/` | `efd3f9c` |

Checked on 2026-10-09 from a fresh clone: preparation, the preflight check and `make -C hls csim`
(100/100) all pass.

### 8.2 Build the HLS IP (FPGA workstation)

- Tools: **Vivado / Vitis HLS 2022.2**, target part `xck26-sfvc784-2LV-c`, 100 MHz.
- In `hls/`, run `vitis_hls -f run_hls.tcl`. On Windows:
  `C:/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat -f run_hls.tcl`.
- Required results:
  - the 100-vector PASS message;
  - synthesis succeeds;
  - `hls/build/policy_hls/sol1/impl/export.zip` and `impl/ip/component.xml` exist.
- The kernel computes 8 output channels in parallel; one simulated inference is 13.59 ms. That is
  an RTL simulation figure, not a board measurement.

### 8.3 Build the overlay: NOT BUILT

Vivado block design for the KR260 ([board/demo/README.md](board/demo/README.md) §3):

1. MPSoC with the KR260 board preset and a 100 MHz PL clock; DDR access through `S_AXI_HP0_FPD`.
2. AXI DMA in simple mode with 8-bit streams and a **26-bit buffer-length register**. The default
   14-bit register cannot carry the 27,658-byte input or the 568,240-byte weight load.
3. Instantiate `policy_top_0` from the exported IP and `axi_dma_0`. Wire DMA MM2S → policy input
   and policy output → DMA S2MM. AXI-Lite control goes through connection automation.
4. Generate the bitstream. Deliver `policy.bit` + `policy.hwh` with the same base name.

Register map (generated by HLS):

| Register | Offset |
|---|---|
| control | `0x00` |
| status / ap_return | `0x10` |
| mode (0 = load weights, 1 = infer) | `0x18` |
| shifts_lo | `0x20` |
| shifts_hi | `0x28` |
| version | `0x30` |

### 8.4 Board (KR260)

- **Expected software:** Ubuntu 22.04 for Kria and Kria-PYNQ (`sudo bash install.sh -b KR260`).
  Actual versions on the board are **UNKNOWN** (Q1).
- **Bring-up order:**
  - M0: boot plus version inventory (`board/m0/`).
  - M1: DMA loopback with `board/loopback.py`. Record its p50 time as the latency floor.
  - M4: `timeout 300s python3 -u board/run_policy.py policy.bit artifacts/standin`. Acceptance
    requires **100 vectors, 0 failures**. The script can exit 0 after mismatches, so read its output.
- M0, M1 and M4 results: **none reported** (Q2).
- The board's IP address and SSH user are **UNKNOWN**. Never put passwords or keys in the repo.

---

## 9. What has to happen, in order, to run the real arm

1. **Board:** M0 → M1 → overlay (§8.3) → M4: 100/100 golden vectors on the KR260. *(FPGA side)*
2. **Joint mapping:** the arm is already calibrated (§2.1). Still to do: verify the
   LeRobot-degrees → sim-radians signs and offsets (§4.1, Q5). *(software)*
3. **Scene:**
   - Build the table layout of §3.2–3.3: 25 mm red and blue cubes, two 100 mm bins.
   - Add a third camera matching §3.4, or move the front C920 to that pose. The C920 is currently
     looking down from above, which is the view the sim showed works worse.
   *(software)*
4. **Controller + safety:** add a TinyPolicy backend to the roboticsexp-style loop (or write
   `robot/so101.py` here), reusing its damping, step cap and glide-home (§4.4). *(software)*
5. **Bridge:** agree the protocol details in §7, then build `remote_backend.py` and the board
   service. *(both, Q4)*
6. **First real run:** run the controller against the board with the arm powered off, logging actions,
   before any powered run.
7. Expect a gap between sim and real. The plan (build-spec Phase 9) is about 50 teleoperated episodes
   on the real cube task, recorded with the existing leader arm and `collect.sh`, and then a
   fine-tune.
