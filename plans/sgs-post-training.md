# SGS post-training for SmolVLA

Owner: m-alan08 (handed over by psamin). Issue #116.

Goal: replicate the post-training pipeline of *A Balanced Data Diet* (Zhang et al., arXiv:2610.12465v1)
on our block-sorting task, with SmolVLA as the deployed policy. TinyPolicy and the FPGA path are
unchanged; the same teacher can later distill into TinyPolicy too.

## What the paper does, and how we map it

The paper never runs RL on a vision model. It trains a privileged-state **teacher** with PPO, then
distills it into an **RGB student** with DAgger and deploys the student zero-shot (§4.5, App. C.3).

| Paper | Ours |
|---|---|
| Teacher: PPO from scratch on joint state + point cloud, thousands to 1M GPU envs | Teacher MLP on privileged state, PPO, GPU MuJoCo (MJX) on Futurama |
| SGS over a fixed buffer of N=32,768 reset configurations (§3.2, App. B.2) | Same sampler (PR #112), ported to JAX, over a pre-generated reset buffer |
| Reset families: Reaching, Stable Grasp, Near-Goal (App. A.2) | Same three, adapted to cubes and bins, plus our standard Rest start |
| Shared sparse reward, Eq. 5 (App. A.3.2) | Same terms and weights |
| RGB student: side + wrist camera, ResNet-18s, online DAgger | **SmolVLA** (`lerobot/smolvla_base`), front C920 + wrist IMX307, online DAgger |
| Visual randomization, zero-shot to hardware, no real data | Same; real SO-101 through the roboticsexp stack |

Direct PPO on SmolVLA is out of scope: it is not the paper's method, the flow-matching action expert
has no cheap action likelihood, and a ~450M-parameter model with per-step rendering limits us to tens
of environments.

## Task

Unchanged: four instructions (red/blue cube → left/right bin), `armlab/sim/assets/block_sort.xml`,
30 Hz control, joint-position targets. A harder task variant may come later; SGS showed little gain
over uniform on the paper's simpler task (rod-in-hole, Fig. 5c), so expect a small gap here.

## Phase A: teacher (PPO + SGS)

**Reset buffer.** Generated offline with CPU MuJoCo (runs on a laptop), saved as arrays of
`qpos`, gripper command and instruction. Collision-checked. Balanced across families:

| Family | Start state |
|---|---|
| Rest | Current `BlockSortEnv.reset`: arm at `REST_Q`, cubes in the spawn region. This is how real episodes start |
| Reaching | Target cube anywhere in the workspace; gripper aligned to its grasp frame by IK, random aperture (cube may drop) |
| Stable Grasp | Target cube held at carry height (cube center 5-6 cm), anywhere on the route from the table to the target bin |
| Near-Goal | Held cube on the drop path into the target bin: fraction 0 rests on the bin floor, 1 just clears the 3.5 cm walls, up to 1.1 |

Top-down IK cannot reach much above 6 cm over the bins, so the paper's "Stable Grasp higher than
Near-Goal" becomes the 5-6 cm carry band. Each buffer entry is one SGS configuration. Start with N=32,768 (paper's manipulation size). Cubes
are 4-fold symmetric in yaw and the success predicate ignores orientation, so the paper's
symmetry-aware resets (App. C.3) come for free.

**Observation (teacher).** Joint positions, gripper pose, previous action, both cube poses, both
bin positions, instruction one-hot. The critic may also see joint velocities. The paper uses point
clouds to narrow the teacher/student gap; cube poses should suffice for cubes, revisit if
distillation stalls.

**Action.** Normalized joint-position deltas at 30 Hz, converted to absolute targets for the env
(and for the student labels).

**Reward (Eq. 5).** `100·1[success] − 1·1[failure] − 1e-5·C(‖a‖²) − 1e-4·C(‖a − a_prev‖²) − 1e-4·C(‖q̇_arm‖²)`,
with `C(x) = min(x, 1e4)`. Success: `BlockSortEnv.success()`. Failure (terminates): a cube leaves the
workspace bounds, or any joint exceeds 2× its velocity limit. Episode ends on success, failure or
the step limit.

**Sampler.** Paper manipulation defaults: H=100, t=0.5, κ=1, ε=1e-4, T=2. One outcome is recorded per
finished episode. The NumPy `SGSSampler` stays as the reference; a JAX port runs inside the jitted
training loop and is tested against it.

**Simulator.** MJX for PPO, with the same MJCF as the CPU env so CPU and GPU share one physics
model. A parity test steps both from identical states. Backend: MuJoCo Warp (`impl="warp"`) is the
front-runner. In a CPU smoke run (#125) it matched CPU MuJoCo within 0.01 mm, while the JAX backend
launched a cube in one of 8 states. GPU throughput is still unmeasured.

**Experiment.** SGS vs uniform sampling on the same buffer, three training seeds each, at a few
environment counts (e.g. 1K, 4K, 16K on one GPU). Evaluate on freshly sampled resets, never on
buffer entries used in training. Report success per reset family.

## Phase B: cameras

Matched sim cameras are required for zero-shot transfer. SETUP.md lists both real cameras as not
matched to sim, and the front pose as unmeasured.

1. Intrinsics for both cameras: checkerboard + OpenCV calibration.
2. Front C920 pose: AprilTag at a known table position.
3. Wrist IMX307 pose: from `hardware/so101/wrist_camera_mount_IMX307_38mm.stl`, attached to the
   gripper body; checked by overlaying a real frame on a render.
4. Resolutions: front 640×480, wrist 480×640 (rotated 90°), both 30 fps.
5. Add `front` and `wrist` cameras to the MJCF. Keep `overhead` untouched (TinyPolicy uses it).

## Phase C: SmolVLA distillation (online DAgger)

- Student: SmolVLA from `lerobot/smolvla_base`, inputs `front` and `wrist` images, 6 joint states in
  LeRobot degrees, task text from `TASKS`. Pin the LeRobot commit roboticsexp uses (`e624f3f7`).
- Envs: CPU MuJoCo with EGL rendering, 64–256 envs across worker processes, SmolVLA on one GPU.
- Each iteration: the student acts, each visited state is labeled by the teacher, the student takes
  a gradient step, data is discarded (App. C.3). 6.67% of envs are held out for evaluation only.
- Chunk labels: SmolVLA predicts action chunks. Label a chunk by rolling the teacher forward from a
  copy of the student's sim state for the chunk length.
- Visual randomization: lighting, table and cube colors, camera pose and FOV jitter, strong brightness
  jitter on the wrist camera (its exposure cannot be locked).

## Phase D: real arm

Zero-shot through `hardware/so101/roboticsexp` (`policy.sh`). Prerequisites: Q5 (degrees ↔ radians
per joint) answered, and the existing "targets exceed the 10°/step cap" issue understood. Metrics as
in App. C.5: success rate, first-try success, throughput, per instruction.

## Compute

Development and unit tests on a laptop (M5, 16 GB): CPU MuJoCo, JAX CPU at small batch. All training
runs as Slurm jobs on Futurama, submitted by psamin.

## Order of work (one issue and PR each, ~200 lines)

1. This plan.
2. Reset families (#119), then the buffer generator (#122), with tests.
3. Teacher reward, failure and observation on the CPU env, as the reference for MJX (#123).
4. GPU simulator benchmark, MJX JAX vs Warp (#125, needs one GPU job), then the MJX/Warp env and
   its CPU parity test (#129).
5. JAX SGS port, tested against `armlab/rl/sgs.py` (#127).
6. Batched env with SGS resets (#131), PPO (#133), training CLI + Slurm job (#135); first teacher
   run; held-out evaluation; SGS vs uniform.
7. Camera calibration and `front` / `wrist` cameras in the MJCF.
8. SmolVLA sim adapter (images, state units, task text).
9. Online DAgger distillation with held-out evaluation.
10. Real-arm evaluation.

PR #112 (sampler) is reused as is. PR #114 (fixed-bank expert runner) is superseded by step 6.

## Open questions

- Q5 ownership: needed before step 8 is meaningful on hardware.
- Futurama GPU types, CUDA version and job time limits available to us: unknown.
- Warp throughput on Futurama GPUs with the SO-101 mesh contacts: unknown until #125's job runs.
