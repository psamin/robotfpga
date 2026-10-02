# Software kickoff plan

How the software side (sim, data, training, quantization, eval, board bridge) gets started from
[build-spec.md](build-spec.md). Status as of 2026-10-01.

## Decisions

| Question | Answer |
|---|---|
| FPGA board | **KR260** (K26 SOM), Kria-PYNQ |
| Training GPU | **RunPod H100**. The tiny policy can also train on a laptop. |
| Arm | SO-101 assumed (MuJoCo Menagerie `robotstudio_so101`). **Not confirmed yet.** |
| Python | **3.12** with uv. LeRobot 0.6.1 requires ≥ 3.12; the spec's "3.10+" is too low. |

## Spec gaps to settle with the FPGA side

The arithmetic in spec §5.3 checks out (565,552 params, 9,665,024 MACs, 27,658-byte input). These
items are what the spec leaves undefined. The FPGA branch `hls-v0.1` independently raised #1, #4 and the manifest format.

| # | Issue | Proposal |
|---|---|---|
| 1 | Shift `s = 0` makes `1 << (s-1)` undefined | Require `1 ≤ s ≤ 30` (matches `hls-v0.1`) |
| 2 | FC6 concatenates conv5 output (power-of-two scale) with aux (1/127 scale) | QAT's forward pass is the integer path itself, so FC6 learns weights for one int8 vector. Document it so nobody "fixes" it in hardware. |
| 3 | `round(state * 127)` rounding mode unspecified (NumPy rounds half to even) | `floor(x * 127 + 0.5)` everywhere |
| 4 | Order of bias and requantize; conv1 inputs reach -128 | `acc = Σ w·x + bias`, then round-shift, then clamp. Conv1 inputs span [-128, 127]. |
| 5 | 2 ms physics does not divide 33.3 ms control (16.67 substeps) | Timestep 1/600 s × 20 substeps = exactly 30 Hz |
| 6 | `MUJOCO_GL=egl` is Linux-only | EGL on Linux; native offscreen on macOS (verify in Phase 1) |
| 7 | 96×96 is far below SmolVLA's input size | Record at 256×256, downsample to 96×96 for the tiny policy |
| 8 | 8-step chunks with replan every 4 tolerate ~133 ms before running dry | Latency sweep goes past 133 ms so Stage C shows both regimes |

## Shared contract files

`hls-v0.1` already contains a bit-exact `ref/intref.py`, a manifest format, stand-in weights and 100
golden vectors. To avoid duplicate work:

- The software side **adopts that `intref.py` and manifest format** rather than writing a second one.
  `policy/export.py` writes the same keys.
- Changes to `ref/intref.py`, the manifest format or the vector format need an issue and sign-off
  from both sides (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
- QAT (Phase 5) must match that `intref.py` bit for bit on 1,000 inputs.

## Where things run

| Work | Machine |
|---|---|
| Sim, expert, data collection (P1–3) | Laptop |
| Tiny policy BC and QAT (P4–5) | Laptop; H100 if epochs are slow |
| Eval harness (P6) | Laptop, CPU |
| SmolVLA and ACT fine-tune (P7) | RunPod H100. Stop the pod when the run ends. |
| Board in the loop (P8) | KR260 plus laptop over TCP |

## Order of work

Each phase stops at its acceptance check.

| Phase | Build | Accept |
|---|---|---|
| 0 Bootstrap | uv project (Python 3.12), ruff, pytest, run-metadata helper (config, git hash, seed → JSON) | `pytest` and `ruff check` clean |
| 1 Sim | `scene.xml` (table, 2 cubes, 2 bins, overhead camera at 256²), `BlockSortEnv`, success check | Tests pass, random rollout MP4, same seed gives identical observations |
| 2 Expert | Position IK with top-down wrist constraint (SO-101 has 5 arm joints), damped least squares or `mink` | ≥ 95% over 100 seeds per instruction |
| 3 Data | 300 episodes per instruction plus noise-injected recovery episodes, LeRobot format, replay script | Dataset loads; replay reproduces success |
| 4 Float policy | Exact §5.3 architecture (test asserts 565,552 params), L1 chunk loss, augmentation, closed-loop eval during training | ≥ 80% Stage B, 50 held-out seeds per instruction |
| 5 QAT | Fake-quant that reproduces `intref` exactly; shift choice from activation stats; export | Bit-exact on 1,000 inputs; within 5 points of float |
| 6 Eval | `PolicyBackend`, torch/intref/mock-fpga backends, latency-delayed chunk queue, Stage C perturbations, Wilson CIs, `report.py` | One command makes the 3-row table and success-vs-latency chart |
| 7–10 | SmolVLA + ACT baselines, board in the loop, real arm, dashboard | Planned in detail after Phase 6 |

## Libraries

Depend on these (pinned), do not fork: LeRobot (dataset format, SmolVLA, ACT, SO-101 driver),
MuJoCo Menagerie (arm model, copied with its license), mink (IK, optional), PYNQ.
Write ourselves, because they are what the project measures: TinyPolicy, QAT, export, eval harness,
chunking and latency injection, board bridge.

Baselines: SmolVLA (spec) and ACT (one extra training run in LeRobot, gives the latency chart a
middle point). pi0/pi0.5 are possible on the H100 but deferred until SmolVLA results are in.

## Risks

| Risk | Early signal | Mitigation |
|---|---|---|
| Grasps slip in sim | Expert fails at lift | Tune friction, `condim`, gripper force before blaming IK |
| Power-of-two scales cost accuracy | QAT gap > 5 points | Per-layer shift search, longer QAT, per-channel shifts if the FPGA side agrees |
| Red/blue confusion at 96×96 | Stage B errors on the wrong cube | Check color jitter range; report before changing architecture |
| Idle GPU spend | Pod left running | Stop the pod as the last step of every run |
