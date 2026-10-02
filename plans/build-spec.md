# Build spec: tiny robot policy on an FPGA (software side)

Paste this whole file into Claude Code as the first message in an empty repo.

## 0. What we are building

A robot arm sorts colored blocks into bins. A small neural network (the "tiny policy") looks at a camera image and decides how to move the arm. That network runs on an FPGA board built by two teammates. A much bigger model (SmolVLA) runs on a laptop or GPU as the smart baseline.

The question the project answers: **how much task success do we keep, and how much latency and power do we save, when the policy runs on a small FPGA instead of a GPU?**

You (Claude Code) are building everything except the FPGA hardware itself: simulator, data, training, quantization, the bit-exact reference model, the evaluation harness, and the bridge to the board.

### Assumptions to confirm before coding

Ask me about these three before you start. Do not guess.

1. **Board.** Assumed AMD Kria KV260 or KR260 (both use the K26 module: 4-core ARM Cortex-A53, 4 GB DDR4, 256K logic cells, 1,248 DSP slices). Not confirmed yet.
2. **Arm.** Assumed SO-101 (6 joints counting the gripper, LeRobot ecosystem). Not confirmed yet. If it is a different arm, only `sim/assets/` and `robot/` change.
3. **Training machine.** Tell me what GPU is available (local, Colab, or cloud).

### Rules for you

- Build one phase at a time. Stop at the end of each phase, show me the acceptance check passing, and wait for me.
- Library APIs change. Before using MuJoCo, LeRobot or PYNQ, check the installed version's actual API. Do not trust this file over the installed package.
- Everything must run headless (`MUJOCO_GL=egl`) and on a laptop with no FPGA attached.
- Seeds everywhere. Every eval run writes a JSON with the config, git hash and seed.
- Python 3.10+, `uv` for environments, `pytest` for tests, `ruff` for lint.

## 1. System overview

```
           LAPTOP / GPU                               KRIA BOARD
 ┌──────────────────────────────┐        ┌───────────────────────────────────┐
 │ MuJoCo sim  or  real arm     │        │  ARM CPU (Linux + Python)         │
 │   camera 96x96 + joints      │        │   control loop, camera, safety    │
 │            │                 │        │            │  AXI DMA             │
 │            ▼                 │  same  │            ▼                      │
 │  PolicyBackend.infer(obs) ───┼─ API ──┼──►  FPGA fabric: tiny policy      │
 │   • torch (float)            │        │      int8 conv + FC layers        │
 │   • intref (numpy, int8)     │        │            │                      │
 │   • fpga (PYNQ)              │        │            ▼                      │
 │   • smolvla (baseline)       │        │   48 int8 action values           │
 │            │                 │        └───────────────────────────────────┘
 │            ▼                 │
 │  action chunk → controller → joints
 └──────────────────────────────┘
```

The key design idea: **every policy sits behind one interface**, so the same eval script scores the float model, the integer reference, the FPGA and SmolVLA. Swapping backends is one flag.

## 2. The task

Scene: table, SO-101 arm, one red cube, one blue cube, a left bin and a right bin, one fixed overhead camera.

| Stage | Task | Passes when |
|---|---|---|
| A | Pick the red cube, drop it in the left bin | cube center inside bin, cube released, within 20 s |
| B | Follow one of 4 instructions (see below) | the named cube ends in the named bin |
| C | Stage B, plus a disturbance mid-episode | still succeeds after the cube is moved or the instruction is swapped while the arm is reaching |

Instructions are a fixed set, encoded as a one-hot vector of length 4:

| id | meaning |
|---|---|
| 0 | red → left bin |
| 1 | red → right bin |
| 2 | blue → left bin |
| 3 | blue → right bin |

The tiny policy does not understand free text. SmolVLA gets the same instructions as real sentences.

## 3. Timing

| Thing | Value |
|---|---|
| Physics step | 2 ms |
| Control rate | 30 Hz (one joint target every 33.3 ms) |
| Action chunk | 8 future steps per inference |
| Replan | every 4 steps (about 7.5 inferences per second) |
| Latency budget | inference under 33 ms end to end, so a fresh chunk lands within one control tick |

Model the latency in sim: the eval harness takes `--latency-ms` and delays each chunk by that much before it is applied. This is how we show that a slow policy fails Stage C and a fast one does not.

## 4. Repo layout

```
armlab/
  pyproject.toml
  README.md
  sim/
    assets/            # arm MJCF + scene.xml (bins, cubes, camera)
    env.py             # BlockSortEnv: reset, step, render, success check
    expert.py          # scripted pick-and-place using IK
    perturb.py         # Stage C disturbances
  data/
    collect.py         # run expert, save episodes
    format.py          # write LeRobot-format dataset
  policy/
    tiny.py            # TinyPolicy (float, PyTorch)
    train.py           # behavior cloning
    qat.py             # quantization-aware training
    export.py          # writes weights.bin + manifest.json
  ref/
    intref.py          # numpy int8 forward pass, bit-exact spec
    vectors.py         # generates golden test vectors for the FPGA team
  backends/
    base.py            # PolicyBackend protocol
    torch_backend.py
    intref_backend.py
    fpga_backend.py    # PYNQ, runs on the board
    remote_backend.py  # laptop → board over TCP
    smolvla_backend.py
  control/
    chunking.py        # action chunk queue + replan
    safety.py          # limits, rate clamp, watchdog
  eval/
    run.py             # success rate, latency p50/p95, JSON report
    report.py          # comparison table across backends
  robot/
    so101.py           # real arm adapter
  dash/
    app.py             # live demo dashboard
  tests/
```

## 5. Interfaces

### 5.1 Observation and action

```python
obs = {
  "image": np.uint8  [96, 96, 3],   # overhead camera, RGB, HWC
  "state": np.float32 [6],          # joint positions, normalized to [-1, 1]
  "instr": int                      # 0..3
}
action_chunk = np.float32 [8, 6]    # 8 future joint targets, normalized to [-1, 1]
```

Normalization uses each joint's limits from the MJCF. Gripper is joint 6.

### 5.2 PolicyBackend

```python
class PolicyBackend(Protocol):
    name: str
    def reset(self) -> None: ...
    def infer(self, obs: dict) -> np.ndarray: ...   # returns [8, 6] float32
    def stats(self) -> dict: ...                    # last latency_ms, optional power_w
```

### 5.3 TinyPolicy architecture

This is the contract with the FPGA team. Do not change it without telling them.

| # | Layer | In | Out | Weights | Multiply-adds |
|---|---|---|---|---|---|
| 1 | conv 3x3, stride 2, pad 1, ReLU | 96x96x3 | 48x48x16 | 432 | 995,328 |
| 2 | conv 3x3, stride 2, pad 1, ReLU | 48x48x16 | 24x24x32 | 4,608 | 2,654,208 |
| 3 | conv 3x3, stride 2, pad 1, ReLU | 24x24x32 | 12x12x64 | 18,432 | 2,654,208 |
| 4 | conv 3x3, stride 2, pad 1, ReLU | 12x12x64 | 6x6x96 | 55,296 | 1,990,656 |
| 5 | conv 3x3, stride 2, pad 1, ReLU | 6x6x96 | 3x3x128 | 110,592 | 995,328 |
| – | flatten (HWC order) + concat aux | 1,152 + 10 | 1,162 | – | – |
| 6 | FC, ReLU | 1,162 | 256 | 297,472 | 297,472 |
| 7 | FC, ReLU | 256 | 256 | 65,536 | 65,536 |
| 8 | FC, no activation | 256 | 48 | 12,288 | 12,288 |

Totals: **565,552 parameters** (with biases), about **9.67 million multiply-adds** per inference. At int8 that is about 0.57 MB of weights.

`aux` is 10 values: 6 joint states followed by the 4-wide instruction one-hot.
The 48 outputs reshape to `[8, 6]`, row = time step.

### 5.4 Integer math spec (must be bit-exact)

`ref/intref.py` is the single source of truth. The FPGA must match it bit for bit.

- Image: `x = uint8_pixel - 128`, stored as int8.
- Aux: state is `round(state * 127)` clamped to [-127, 127]. One-hot uses 127 for the hot entry and 0 otherwise.
- Weights: int8 in [-127, 127]. Biases: int32, already in the accumulator's scale.
- Accumulator: int32. Worst case here is 1,162 x 127 x 127 = 18.7 million, far inside int32.
- Requantize per layer with one shift `s` (power-of-two scales only, no multipliers):
  `y = (acc + (1 << (s - 1))) >> s` with arithmetic shift, then clamp.
- Clamp: ReLU layers clamp to [0, 127]. The last layer clamps to [-127, 127].
- Output: `action = int8_out / 127.0`.
- Tensor order everywhere: HWC, row-major. Conv weights are `[out_ch][ky][kx][in_ch]`. FC weights are `[out][in]`.

### 5.5 Files handed to the FPGA team

`policy/export.py` writes:

- `weights.bin`: layers in order, each layer's weights (int8) then biases (int32 little-endian).
- `manifest.json`: per layer type, shapes, shift `s`, byte offsets into `weights.bin`, plus a SHA-256 of the file.
- `vectors/NNN_in.bin` (27,658 bytes: 27,648 image bytes then 10 aux bytes) and `vectors/NNN_out.bin` (48 bytes). Generate 100, including all-zero, all-max and random inputs along with real frames.
- `vectors/NNN_layerK.bin`: the output of every layer for the first 5 vectors, so the hardware team can debug one layer at a time.

### 5.6 Board bridge

- `fpga_backend.py` runs on the board. It loads the overlay with PYNQ, copies the 27,658-byte input into a contiguous buffer, starts the DMA transfer, waits, and reads 48 bytes back. It times the full call.
- `remote_backend.py` runs on the laptop. Tiny TCP protocol: send 27,658 bytes, receive 48 bytes plus a float32 latency. This lets the laptop-side sim drive the board in the loop.
- Until hardware exists, `fpga_backend` falls back to `intref` behind a `--mock` flag so nothing blocks.

## 6. Phases

**Phase 1. Sim.** Get the SO-101 MJCF (MuJoCo Menagerie `robotstudio_so101`, or TheRobotStudio's `SO-ARM100` repo). Build `scene.xml` with bins, cubes and the camera. Implement `BlockSortEnv` and the success check.
Accept: `pytest` passes, a random-action rollout renders to an MP4, the env is deterministic under a fixed seed.

**Phase 2. Scripted expert.** IK-based pick and place for all 4 instructions, with randomized cube positions and small camera, lighting and color jitter.
Accept: at least 95% success over 100 seeded episodes per instruction.

**Phase 3. Data.** Collect 300 episodes per instruction at 30 Hz. Save in LeRobot dataset format so SmolVLA can train on the same data. Add noise-injected episodes where the expert recovers from a nudged state (this is what makes Stage C possible).
Accept: dataset loads, a replay script reproduces an episode in sim.

**Phase 4. Float tiny policy.** Behavior cloning, L1 loss on the 8-step chunk. Image augmentation (shift, brightness).
Accept: at least 80% success on Stage B over 50 held-out seeds per instruction. If it misses, tell me what you would change before changing the architecture.

**Phase 5. Quantize.** QAT with power-of-two scales and int8 weights and activations. Write `intref.py` and prove the PyTorch QAT model and the numpy reference agree exactly on 1,000 inputs.
Accept: intref success within 5 points of float. Export files and vectors generated.

**Phase 6. Eval harness.** `eval/run.py --backend X --stage A|B|C --episodes N --latency-ms L`. Report success rate with a 95% confidence interval, latency p50 and p95, and failure reasons. `report.py` builds the comparison table.
Accept: one command produces the table for torch, intref and mock-fpga.

**Phase 7. SmolVLA baseline.** Fine-tune `lerobot/smolvla_base` on the same dataset with `lerobot-train`, wrap it as a backend, run the same eval. Measure its real latency on my machine.
Accept: it appears as a row in the table.

**Phase 8. Board in the loop.** When the bitstream exists: `fpga_backend` on the board, `remote_backend` on the laptop, sim driven by the FPGA. Also time the same int8 network on the board's ARM CPU alone (ONNX Runtime or a plain C loop), because that is the fair comparison for the fabric.
Accept: FPGA outputs match `intref` on all 100 vectors, and the eval table has FPGA and ARM-CPU rows.

**Phase 9. Real arm.** `robot/so101.py` adapter, camera calibration to match the sim view, collect about 50 teleop episodes, fine-tune, evaluate 20 real trials per instruction.

**Phase 10. Demo dashboard.** Live camera view, current instruction, a button to swap it mid-episode, and a side-by-side readout of latency and watts for FPGA vs laptop.

## 7. Safety layer (real arm, always on)

Lives in `control/safety.py`, outside the learned policy:

- Clamp every target to joint limits minus a margin.
- Limit the change per control step (max joint speed).
- Watchdog: if no fresh chunk arrives within 250 ms, hold position.
- Workspace box: reject targets whose fingertip leaves it.
- A keyboard key that cuts torque.

## 8. What to measure

| Metric | How |
|---|---|
| Success rate | 50+ episodes per instruction, held-out seeds, with confidence interval |
| Latency | observation ready → action ready, p50 and p95, including data transfer |
| Model size | parameters and bytes |
| Power | board power while running (hardware team supplies the method) |
| Robustness | Stage C success as a function of injected latency |

The headline chart: success rate against latency, one point per backend.

## 9. Out of scope for now

Running SmolVLA on the FPGA. Free-form language on the tiny policy. Multiple cameras. ROS.
