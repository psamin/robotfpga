# robotfpga

An SO-101 robot arm picks up a medicine bottle and places it on a plate, driven by a **SmolVLA**
policy fine-tuned on our own demonstrations. The goal is to run that policy on an AMD Kria
**KR260** FPGA in full precision: its ~450M parameters stay in the board's DDR as BF16 and stream
into an FPGA matrix engine that computes in FP32, while the ARM cores run everything nonlinear.

## Status

| Part | State |
|---|---|
| Fine-tuned SmolVLA | Trained on 15 real demos ([dataset](https://huggingface.co/datasets/flyingturtleboop/so101_medicine_bottle_pickplace)); checkpoint on the GT cluster |
| Real-arm stack | Teleop, recording, rollouts, calibration: [robot/](robot/README.md) |
| FPGA engine | [fpga/hls/](fpga/hls/): BF16 weights from 4 DDR ports, FP32 math; passes C simulation on real SmolVLA layers |
| SmolVLA with every linear layer in the engine code | Matches PyTorch FP32 to ~0.01° mean, 0.125° max on real frames |
| Estimated FPGA time | ~3–4 s per 50-action chunk (compute-bound; see the plan) |
| Synthesis, bitstream, board run | Next ([QUESTIONS.md](QUESTIONS.md) Q12) |

Plan, design rules and results: [docs/smolvla-fpga.md](docs/smolvla-fpga.md).

## Layout

| Path | What |
|---|---|
| `policy/` | SmolVLA checks: the whole model run through the FPGA engine code |
| `fpga/hls/` | Full-precision engine (HLS C++), C model, packing, latency model, Vitis build script |
| `fpga/board/` | KR260 runtime: swaps SmolVLA's `nn.Linear` layers for FPGA calls |
| `robot/` | SO-101 calibration, real-arm toolkit, wrist camera mount |
| `cluster/` | Slurm jobs (GT Futurama) |
| `docs/` | Plan and verified references |
| [SETUP.md](SETUP.md) | Physical and software setup, step by step |

## Quick start

```sh
make -C fpga/hls test          # engine C model vs float64 reference (needs g++ and numpy)
```

Running SmolVLA through the engine code needs the checkpoint and a cluster node:
`sbatch --export=ALL,MODEL=<checkpoint dir> cluster/smolvla_fpga.sbatch` (see [SETUP.md](SETUP.md)).
Building the bitstream needs Vivado / Vitis HLS 2022.2: `cd fpga/hls && vitis_hls -f run_float_hls.tcl`.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md); coding agents also follow [AGENTS.md](AGENTS.md).
Earlier cube-sorting work (TinyPolicy, int8) is preserved at the git tag `archive/cube-sorting`.

MIT license.
