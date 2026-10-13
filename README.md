# robotfpga

A 566K-parameter vision policy that drives an SO-101 robot arm, quantized to int8 and run on an
AMD Kria KR260 FPGA. The policy sees a 96×96 camera image, the joint angles and one of 4
instructions ("red/blue cube → left/right bin"), and outputs the next 8 joint targets at 30 Hz.

## Results

| | Value |
|---|---|
| Sim success, float (Stage B / Stage C, n = 200) | 100% / 98% |
| Sim success, int8 (Stage B / Stage C, n = 200) | 91% / 90% |
| FPGA inference, KR260 at 100 MHz | 8.03 ms (vs 70.8 ms on the board's ARM CPU) |
| FPGA vs bit-exact reference | 100/100 golden vectors byte-for-byte |
| Sim closed loop with the FPGA computing every action | 92.5% (37/40) |
| FPGA resources | 1.7% DSP, 31% UltraRAM, 24% BRAM |

How it was built, with citations: [plans/tiny-policy-log.md](plans/tiny-policy-log.md).

## Repository

| Path | What |
|---|---|
| `armlab/` | Python package: MuJoCo sim, scripted expert, data collection, training (BC + DAgger), int8 QAT, eval, RL teacher (`rl/`), camera calibration (`calib/`) |
| `ref/` | Bit-exact int8 reference (`intref.py`) and golden-vector generator: the contract the FPGA must match |
| `handoff/v3r2/` | Trained int8 weights, manifest, 100 golden vectors, eval results, checkpoints |
| `hls/`, `board/` | FPGA: HLS kernel, RTL blocks, KR260 board scripts and runbook |
| `hardware/so101/` | Real-arm calibration, teleop / recording / rollout toolkit, wrist camera mount |
| `cluster/` | Slurm jobs for collection, training, DAgger, QAT and eval |
| `plans/` | Spec, plans, engineering log, references |
| [SETUP.md](SETUP.md) | Full reference: scene geometry, joint table, packet format, timing, every setup step |

## Quick start

**1. Sim and training** (Python 3.12, [uv](https://docs.astral.sh/uv/)):

```sh
uv sync
uv run pytest                                    # tests
uv run python scripts/rollout_video.py           # render a sim rollout
uv run python -m armlab.eval.run --backend intref --ckpt handoff/v3r2 --stage B --episodes 10 --out eval.json
```

Training is GPU and CPU heavy; run it with the Slurm jobs in `cluster/` (see `cluster/final.sbatch`).

**2. Check the FPGA handoff** (any machine with Git, Python 3.8+ and NumPy):

```sh
python3 board/demo/prepare_handoff.py            # gathers the handoff into build/fpga-handoff
cd build/fpga-handoff
python3 board/demo/handoff_check.py artifacts/standin --reference ref/intref.py --verify-reference
```

Expect `PASS` with 100 reference calls and 40 layer dumps. `artifacts/standin/` holds the trained v3r2 policy.

**3. Build and run on the KR260** (Vivado / Vitis HLS 2022.2, part `xck26-sfvc784-2LV-c`):
follow [board/demo/README.md](board/demo/README.md).

**4. Real arm**: calibration and the teleop / recording toolkit are in
[hardware/so101/](hardware/so101/README.md).

## Interfaces

- **Input packet** (27,658 bytes): 96×96×3 RGB uint8 image, then 6 joint bytes
  `floor(norm·127 + 0.5)` and a 4-byte instruction one-hot (127 / 0).
- **Output** (48 bytes): int8 ÷ 127 → [8 steps × 6 joints] normalized targets.
- **Network**: 5 conv (3×3, stride 2, 16→128 channels) + 3 FC (256, 256, 48); one power-of-two
  shift per layer, round-half-up, int32 accumulators. Defined in `ref/intref.py`.
- **Board service**: TCP, 27,658 bytes in → 48 bytes + float32 latency (little-endian) out.

## Status

Done: sim, training, int8 export, FPGA inference on the board, FPGA in the sim loop, real camera
→ FPGA with the arm's torque off.

Open:
- Commit the layer-engine source that runs on the board (#98).
- Laptop client `remote_backend.py`.
- Verify each joint's sign and offset between the real arm and the sim.
- A safety layer, then powered real-arm runs.

Open questions between the teams are tracked in [QUESTIONS.md](QUESTIONS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Coding agents also follow [AGENTS.md](AGENTS.md).

MIT license.
