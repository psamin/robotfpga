# SGS post-training of the fine-tuned SmolVLA (pick and place)

Owner: m-alan08. Approved by psamin in #156. Plan issue: #158.

**Goal.** Improve the demo-fine-tuned SmolVLA on the real task, "Pick up the medicine bottle and place
it on the plate", with the method of *A Balanced Data Diet* (Zhang et al., arXiv:2610.12465v1):
a state-based teacher trained with PPO and Success Guided Sampling (SGS) in simulation, then
distilled into SmolVLA with online DAgger. Start from `smolvla-so101_medicine_bottle_pickplace-9604`
and write a **new** checkpoint; the original is never modified.

## What carries over from the cube-sorting work

All of it is at the tag `archive/cube-sorting` (`armlab/`, `tests/`) and was tested there
(154 tests). It is task-independent except where noted.

| Piece | Archive path | Reuse |
|---|---|---|
| SO-101 MJCF (Menagerie) and top-down IK | `armlab/sim/assets/so101/`, `armlab/sim/ik.py` | as is |
| SGS sampler, NumPy and JAX | `armlab/rl/sgs.py`, `sgs_jax.py` | as is |
| PPO, batched env with SGS resets | `armlab/rl/ppo.py`, `vec_env.py` | as is |
| Reset buffer generator | `armlab/rl/buffer.py` | as is; families change |
| Teacher env, CPU and MJX/Warp | `armlab/rl/teacher_env.py`, `teacher_mjx.py` | observation, success and failure change |
| Training, held-out eval, SGS vs uniform comparison, Slurm jobs | `armlab/rl/train_teacher.py`, `eval_teacher.py`, `compare_runs.py` | small changes |
| GPU simulator benchmark | `scripts/bench_sim.py` | rerun on the new scene |
| Camera calibration (checkerboard, AprilTag pose to a MuJoCo camera) | `armlab/calib/camera.py` | as is |

Findings that still apply: MuJoCo Warp matched CPU MuJoCo within 0.01 mm while MJX's JAX backend
launched a cube in one of 8 states; a top-down grasp could not be held much above 6 cm at the old
bins (x = 0.14 m, y = ±0.20 m from the base), which limits carry heights; AprilTag corner refinement
is needed for millimetre camera poses.

New code goes under `policy/sim/` and `policy/rl/` (software side), with `mujoco`, `jax`,
`mujoco-warp` and `opencv` in an optional dependency group so the FPGA side's setup is unchanged.

## What is new for pick and place

**Scene.** One medicine bottle and one plate on the table, the SO-101 at the origin, `front` and
`wrist` cameras placed from calibration. Sizes, mass and layout are **unknown** (SETUP.md §1); the
scene is not built until they are measured (Q15).

**Success.** The bottle rests on the plate (inside the plate's rim, upright within a tolerance) and
the gripper no longer touches it. Exact thresholds come from the measured plate and bottle.

**Reset families** (paper App. A.2, adapted):

| Family | Start state |
|---|---|
| Rest | Arm at the recorded start pose, bottle and plate placed as in the demos |
| Reaching | Gripper aligned to the bottle's grasp frame by IK, random aperture; the bottle may tip or drop |
| Stable Grasp | Bottle held and lifted on the route to the plate |
| Near-Goal | Held bottle on the way down onto the plate, from touching the plate to just above it |

A cylinder is symmetric about its axis, so grasps and success ignore the bottle's yaw (paper App. C.3).

**Teacher.** Privileged state (joints, gripper pose, bottle pose, plate pose, previous action),
PPO + SGS on the reset buffer, the paper's Eq. 5 reward (success 100, failure 1, small action,
action-rate and joint-speed penalties). Acts in joint-position targets at 30 Hz so its actions map
directly to SmolVLA's action space.

**Student: SmolVLA, online DAgger.**
- Inputs as in training: `camera1` = front, `camera2` = wrist, padded and resized to 512×512; the
  6 joint states in LeRobot units (degrees, gripper 0 to 100); the task text.
- Outputs: 50-step chunks of 6 joint targets. Labels come from rolling the teacher forward from a
  copy of the student's sim state for the chunk length.
- Train the action expert only, vision encoder frozen, as in the original fine-tune.
- Mix DAgger batches with the real demo episodes so the real-image skill is not forgotten.
- 6.67% of sim envs held out for evaluation (paper App. C.3).

**First experiment before any training:** run the unchanged `-9604` checkpoint in the sim with the
calibrated cameras. Its sim success is the baseline, and a very low number measures the sim-to-real
gap that visual randomization has to close.

## Layout estimated from the demos (#171)

`policy/sim/demo_layout.py` runs the dataset's joint states through the SO-101 model and finds where
the gripper closes (approximately the bottle) and opens (approximately the plate). 13 training
episodes (11 and 14 excluded, SETUP.md §3). Robot base frame: origin at the base on the table, +x
forward, +y to the robot's left, z up. **Estimates, not measurements**, under the joint conversion
assumed in `policy/sim/so101.py`.

| Quantity | Mean | Std | Range |
|---|---|---|---|
| Gripper closes (bottle) x, y | 0.249, +0.092 m | 0.027, 0.021 m | x 0.189 to 0.300, y +0.059 to +0.125 |
| Gripper closes, height above table | 0.031 m | 0.018 m | 0.014 to 0.063 |
| Gripper opens (plate) x, y | 0.320, -0.209 m | 0.014, 0.014 m | x 0.297 to 0.349, y -0.230 to -0.180 |
| Gripper opens, height | 0.034 m | 0.020 m | 0.000 to 0.064 |
| Highest point while carrying | 0.152 m | 0.020 m | 0.119 to 0.179 |
| Gripper reading while holding (0 to 100) | 17.1 | 3.7 | two groups: about 11.5 when grasped at 4 to 6 cm, about 19.5 at 1 to 3 cm |

Read as: the bottle starts about 25 cm out and 9 cm to the left; the plate sits about 32 cm out and
21 cm to the right, and moves less than the bottle. The two grip groups suggest a narrower part
higher up (likely the cap) and a wider body. Checks behind the assumption: no grasp or release lands
below the table.

**Checked against the front video (#173).** The front camera looks straight down. `policy/sim/demo_video.py`
finds the bottle's orange cap at the start and after the release in each training demo (25 detections)
and fits the joint-data positions to those pixels in a top view:

| Fit | Mean / max mismatch | Robot base lands at |
|---|---|---|
| shoulder_pan as assumed | 15.7 / 44.6 px (about 1.9 / 5.3 cm) | pixel (323, 45), on the base in the image |
| shoulder_pan sign flipped | 28.3 / 80.9 px | pixel (325, 482), off the table: rejected |

So the assumed pan sign holds. With the fitted scale (842 px/m, at cap height, so table-level sizes
read slightly small): plate center (0.334, -0.189) m, matching the release estimate within 2.5 cm;
plate outer diameter about 0.19 m (a 7-inch paper plate is 0.178 m) with the patterned rim's inner
edge at about 0.12 m; cap diameter about 0.027 m. Still not settled: bottle height, body diameter and
mass, the plate's rim height, and the gripper reading to opening width mapping.

## Unknowns (blocking)

| Unknown | Needed for | Question |
|---|---|---|
| Bottle height, body diameter and mass; plate rim height (plate and cap diameters are estimated above) | scene, success | Q15 |
| Table layout | Rest family, randomization ranges | Estimated from the demos and checked against the front video (above) |
| Sim radians to LeRobot units per joint | every SmolVLA input and label | Derived and checked against the demos (`policy/sim/so101.py`, #175); Q16 asks for an on-arm check |
| Front and wrist camera poses and intrinsics | rendering SmolVLA's inputs | photo session with `calib/camera.py` |

## Order of work (one issue and PR each, about 200 lines)

1. This plan.
2. Restore the task-independent code (SGS NumPy + JAX, PPO, calibration tools) into `policy/rl/`
   and `policy/calib/` with its tests and the optional dependency group.
3. Pick-and-place scene from the Q15 measurements; scripted expert as a sanity check.
4. Reset families, buffer, CPU teacher env with the pick-and-place success and reward.
5. MJX/Warp teacher env with parity tests; batched SGS env; training, eval, comparison jobs.
6. GPU benchmark on the new scene, then the first teacher runs (SGS vs uniform). Each run needs
   psamin's go-ahead on #156.
7. Camera calibration photo session; `front` and `wrist` cameras in the scene.
8. SmolVLA in sim (needs Q16): baseline of `-9604` in sim.
9. Online DAgger into a new checkpoint; held-out sim evaluation.
10. Post results on #156, then real-arm trials only after psamin agrees.
