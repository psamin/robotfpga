# Tiny policy: engineering log

A running record of what we built, what we measured, and why we chose each technique, written
so it can become a blog post. Citation keys like [Zhao2023-ACT] point to
[references.md](references.md), where each paper was checked against arXiv, Crossref or the PDF.
Numbers here are measured unless marked as estimates. Dates are 2026.

## The question

How much task success do we keep, and how much latency and power do we save, when a small
vision policy runs as int8 on an AMD Kria KR260 FPGA instead of as float on a CPU or GPU?
The network is fixed by the hardware contract ([build-spec.md](build-spec.md) §5.3):
5 stride-2 3×3 convolutions (16→128 channels) on a 96×96 image, then three fully connected layers
that output an 8-step × 6-joint action chunk. 565,552 parameters, 9.67 M multiply-adds.

## 1. Simulation (Oct 1–2)

- **Arm model:** MuJoCo Menagerie `robotstudio_so101` [Zakka2022-Menagerie], Apache-2.0, vendored.
  Its position actuators are calibrated to the real STS3215 servos (kp 998, ±2.94 N·m).
- **Timing:** Menagerie's 5 ms step and the spec's 2 ms step both fail to divide the 33.3 ms
  control tick. We use 1/300 s × 10 substeps = exactly 30 Hz.
- **Workspace, measured:** we swept IK over radius and height. A strict top-down grasp can only
  hover up to about 8 cm above the table, at 0.13–0.28 m from the base. We therefore kept cubes
  at x ∈ [0.16, 0.26] m and carry them at 6.5 cm, which clears the 3.5 cm bin walls.
- **Domain randomization** [Tobin2017-DR]: camera pose ±1 cm, field of view ±1.5°, light
  ±0.15, table/bin/cube color ±0.08 per channel. Cube hue is never randomized, since red vs blue
  is the instruction.
- **Rendering:** 288², then a 3×3 area average down to the 96² policy input. The 288² frames
  are kept for SmolVLA, which is designed for much larger images.
  On an M3 Pro a control step with render takes 15.8 ms; on a cluster L4 (EGL), 3.1 ms.

## 2. Scripted expert

- Damped least-squares IK on the gripper site, with the approach axis held vertical. A cube
  grasp looks the same every 90°, so the IK tries each equivalent yaw and keeps the most
  accurate. That fixed every far-edge IK failure.
- **v1 (stateful phase machine):** 400/400 over 100 held-out seeds per instruction
  (Wilson 95% CI [0.963, 1.000] each [Wilson1927]).
- **v2 (stateless):** the phase is recomputed from the world state every tick, so the expert can
  label any state, including states a learner drove into. That is the requirement for DAgger
  [Ross2011-DAgger]. Three bugs found on the way, each measured:
  1. Servoing from the *measured* gripper pose stalled 4% of episodes. The IK flipped between
     equivalent cube yaws tick to tick and the arm dithered. Fix: use the copy of the yaw
     nearest the gripper's current yaw, and keep a command integrator that re-anchors only when
     the arm is >2 cm from it. Result: 400/400 again, and recovery from 80/80 random arm states.
  2. With DART noise the stateless expert bounced between approach and descend at the 6 mm
     alignment threshold. Fix: hysteresis read from the world state (keep descending while
     below hover height and within 15 mm).
  3. The action-chunk label must be the expert's *own* 8-step plan from each state. We compute
     it by simulating the expert ahead on a full copy of MjData (`mj_copyData`). Restoring state
     with `mj_setState` + `mj_forward` instead changed the trajectory, because `mj_forward`
     recomputes derived quantities and the solver warm start.

## 3. Data

- **DART** [Laskey2017-DART]: AR(1)-correlated noise on executed joint targets, with a
  per-episode σ ∈ {0, 0.01, 0.02, 0.03} (normalized units). Labels stay clean. σ = 0.05 broke
  grasps (4/19 succeeded) and was dropped.
- **Stage C disturbances in training data:** 20% of episodes move the target cube mid-reach
  and 20% swap the instruction.
- **v1 dataset:** 2,000 episodes, 1,811 kept, 214,061 frames, collected in 151 s on 48 workers.
- Every episode replays byte-identically from (seed, executed actions, perturbation event).
  This is used to regenerate full-resolution frames for the LeRobot/SmolVLA export.

## 4. Float behavior cloning

Recipe (all cited in references.md): L1 loss on the 8-step chunk [Zhao2023-ACT]; random-shift
augmentation (replicate-pad 4 px, crop) [Kostrikov2021-DrQ, Yarats2022-DrQv2] plus brightness and
contrast, kept on the integer pixel grid so QAT sees real inputs; AdamW [Loshchilov2019-AdamW],
warmup + cosine schedule, EMA weights. Checkpoints are chosen by closed-loop success, not
validation loss [Mandlekar2021-robomimic].

**Run bc-v1 (v1 data, 40k steps, one L4, 7 min):** validation L1 fell to 0.0334, but closed-loop
Stage B success was only **1% / 6% / 7%** at 10k / 20k / 30k steps (25 episodes per instruction
each). Rollout videos showed the policy acting out the expert's motion sequence, for example
closing the gripper on a timing cue before it was over the cube, and never recovering once off
the expert's path. This is the compounding-error problem DAgger addresses [Ross2011-DAgger], and it
is a concrete example of robomimic's warning that validation loss does not predict rollout
success.

## 5. Int8 quantization (QAT), bit-exact with the FPGA reference

- Fake-quant forward pass that computes exactly the integers `ref/intref.py` computes: int8
  weights, int32 biases, one round-half-up shift per layer, ReLU clamp [0, 127], last layer
  [−127, 127], action = int8/127 [Jacob2018-IntOnly]. Gradients pass straight through the
  rounding [Bengio2013-STE].
- Per-tensor power-of-two exponents are chosen by quantization MSE, layer by layer, on a
  calibration batch [Nagel2021-WP]. AMD's own Vitis AI quantizer uses the same symmetric
  power-of-two scheme [VitisAI-pof2s].
- Two parts of the contract aren't powers of two, and we handle both exactly without changing
  the contract:
  - FC6 reads conv5's output and the 10 aux bytes as one vector, so the aux columns are
    rescaled by 2^(a5−7) at conversion.
  - Actions are int8/127, so FC8 is pre-scaled by 127/128.
- **Proof:** QAT (float64) equals `ref/intref.py` byte for byte on 1,000 inputs (`tests/test_qat.py`).
- Smoke test right after calibration: int8 validation L1 0.2218 vs float 0.2231, so
  calibration alone cost nothing measurable on that model.

## 6. DAgger, and a camera the policy can actually use

**Exact chunk labels.** For DAgger the expert must say what to do in states the learner reached.
The right label for an action chunk is the expert's own next-8-step plan from that state, not
the actions at the learner's next states. We simulate the expert 8 ticks ahead on a copy of
the physics state for every frame (about 20 ms per frame on an M3 Pro). Datasets from v2 on use
these labels everywhere, including the expert-driven episodes.

**Collector modes:** expert + DART noise, or learner-driven through the same chunk executor used
at evaluation, with the expert's action substituted at each tick with probability β (DAgger's
mixture policy [Ross2011-DAgger]). DAgger episodes are kept whether they succeed or not; the
failures are where the learning signal is.

**Round 0 on the top-down camera (dataset v2: 1,883 episodes, 199,223 frames):**
13/100 Stage B, CI [0.078, 0.210]. Looking at 12 rollouts:
8 never descended, 3 descended near the cube but missed, 1 succeeded. The policy hovered near the
cube but never met the expert's 6 mm "aligned, now descend" trigger. Two causes:
1. At 96×96 over ~0.6 m, one pixel is ~6 mm, so the trigger asked for 1-pixel precision.
2. A straight-down camera puts the gripper directly between the camera and the cube at the one
   moment alignment matters, and it gives no cue for height.

**Fixes (dataset v3):** the camera moved in front of the arm, 0.55 m up, looking back at ~57°.
The cube stays visible beside and under the jaws, and height shows. The expert's alignment
trigger was relaxed to 12 mm, since the jaws open to ~60 mm around a 25 mm cube; the expert
still succeeds on 399/400. The top-down DAgger run was left going as an ablation.

## 7. Results so far (Oct 2)

Stage B closed-loop success, float TinyPolicy, 25 held-out episodes per instruction (n = 100),
β schedule 0.5 / 0.3 / 0.1 / 0, each round fine-tuned 30k steps on all data so far:

| DAgger round | Top-down camera (v2) | Oblique camera (v3) |
|---|---|---|
| 0 (expert data only) | 13 / 100 | 31 / 100 |
| 1 | 26 / 100 | 48 / 100 |
| 2 | **66 / 100** | **100 / 100** (Wilson 95% CI [0.963, 1.000]) |
| 3 | 60 / 100 | running |
| 4 | 49 / 100 | running |

The camera viewpoint was worth more than every training trick combined: same network, same
data recipe, round 0 more than doubled and round 2 went from 66% to 100%. DAgger did the rest
[Ross2011-DAgger]: going from expert-only data to two learner-driven rounds took the oblique-camera
policy from 31% to 100%.

**Open question: why later rounds got worse on the top-down camera.** Training loss rose every
round as DAgger data was added (0.021 → 0.033) while validation loss on expert episodes stayed
flat. So the new labels were harder to fit, not the model under-trained. Leading hypothesis: the
expert's command integrator is hidden state the policy cannot observe, which makes labels in
learner-driven states inconsistent. A "Markov" expert (every step starts from the measured pose)
is being compared on the cluster (jobs 8984/8985: clean, noisy and recovery success).

**Int8 (QAT) on a real policy (top-down round 2):** calibration alone raised validation L1 from
0.0213 (float) to 0.0362; 10k QAT steps with distillation brought int8 to 0.0251. All layer
shifts land in 5–8 (the hardware accepts 1–30). Exports load in `ref/intref.py` and pass the FPGA
team's HLS C model (100/100 golden vectors, see §5).

## 8. Still running (results land in `~/robotfpga-work/runs/` on the cluster)

| Job | What | Output |
|---|---|---|
| 8986 `final-v3r2` | QAT of the 100% policy → int8 export → float vs int8 Stage B and C (50 per instruction), int8 latency sweep 33–500 ms, golden vectors, HLS C model | `runs/final-v3r2/` (`eval_*.json`, `qat/export/`, `handoff/vectors/`, `hls_csim.txt`) |
| 8983 `final-v2r2` | Same pipeline on the top-down 66% policy (pipeline shakedown) | `runs/final-v2r2/` |
| 8977 `dagger-v3` | Oblique-camera DAgger rounds 3–4 | `runs/dagger-v3/history.json` |
| 8984 / 8985 | Markov vs integrating expert | `robotfpga-final/logs/armlab-script-898{4,5}.out` |

Acceptance still to confirm from job 8986: int8 within 5 points of float (spec Phase 5), Stage C
success, and the latency curve, which is the headline chart.
