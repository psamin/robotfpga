# FPGA plan

How the FPGA side gets the tiny policy running in the KR260 fabric, from
[build-spec.md](build-spec.md) and the FPGA brief. Status as of 2026-10-01.

## Decisions

| Question | Answer |
|---|---|
| Board | **KR260** (K26 SOM), Ubuntu 22.04 + Kria-PYNQ |
| Vivado / Vitis version | Picked when the PC is set up; one version for the whole project |
| First pass | **Vitis HLS**, plain C++ so the same code is the C model and the ARM CPU baseline. RTL conv engine later. |
| Weights | Loaded at runtime over DMA, so retraining does not need a new bitstream |
| Shifts | AXI-Lite registers, set from `manifest.json` |
| Clock | 100 MHz (default `pl_clk0`) until M4 passes, then raised in M5 |

## Answers to the software kickoff's spec gaps

| # | Answer from the FPGA side |
|---|---|
| 1 | Agree, `1 ≤ s ≤ 30`. The kernel returns status 1 for anything else and still sends 48 bytes. |
| 2 | Agree. Hardware treats the FC6 input as one int8 vector of 1,162 and has no special case for aux. |
| 3 | No hardware impact: aux arrives already as int8. |
| 4 | Agree. `acc = Σ w·x + bias`, round-shift, clamp. The kernel's multipliers take the full int8 range for conv1's -128. |

## Interface the accelerator exposes

For `armlab`'s `fpga_backend.py`. The `PolicyAccel` class in `board/run_policy.py` already does this, and the backend can wrap it.

| Thing | Value |
|---|---|
| Data path | AXI DMA, 8-bit AXI-Stream, TLAST on the last byte |
| Mode 0 | load `weights.bin`, 568,240 bytes in, nothing out |
| Mode 1 | one inference, 27,658 bytes in, 48 bytes out |
| Registers | `CTRL` (HLS start/done/idle), `mode`, `shifts_lo` = s1..s4 one byte each, `shifts_hi` = s5..s8, `version` |
| Status (`ap_return`) | 0 ok, 1 bad shift, 2 TLAST in the wrong place, 3 bad mode |

## What exists, and how it lands on `main`

Everything below is on the `hls-v0.1` branch and checked on a laptop with no Xilinx tools:
the C model matches all 100 stand-in golden vectors and every per-layer dump, including under
ASan/UBSan, and the tests fail if the rounding or the padding is broken.

That branch is about 1,350 lines, so it lands as a chain of small PRs, each with its own issue.
`hls-v0.1` gets deleted once the chain is merged.

| PR | Files | Lines | Needs |
|---|---|---|---|
| 1 | this plan | ~110 | |
| 2 | `ref/intref.py`, `ref/test_intref.py`, `.gitignore`, `requirements.txt` | ~290 | **sign-off from software** (shared contract) |
| 3 | `ref/make_standin.py` (stand-in weights + vectors in the handoff format) | ~170 | **sign-off from software** (vector format) |
| 4 | `hls/policy_kernel.h`, `hls/tb_kernel.cpp`, `hls/Makefile` | ~430 | over the size target: the kernel can't be tested without its testbench. Split into layer functions + kernel if reviewers prefer. |
| 5 | `hls/policy_top.cpp`, `hls/tb_top.cpp`, `hls/run_hls.tcl` | ~165 | lands after Vitis csim and synthesis pass on the PC |
| 6 | `board/loopback.py` | ~50 | lands after M1 passes on the board |
| 7 | `board/run_policy.py` | ~100 | lands after M4 passes on the board |

## Milestones

Do them in order. Each has a pass condition.

| | Milestone | Pass | Status |
|---|---|---|---|
| C | C model of the whole network | all 100 vectors match | **done** on `hls-v0.1` |
| M0 | Board bring-up | an overlay loads and runs from Python | next |
| M1 | DMA loopback | 27,658 bytes come back identical; round-trip time written down | |
| M2/M3 | Single layers on the board | match `layer8`, then `layer1` vectors | folded into M4 because the C model already runs every layer |
| M4 | Full network | all 100 vectors match on the board; cycles and LUT/FF/DSP/BRAM/URAM reported | |
| M5 | Speed | under 5 ms end to end from Python | |
| M6 | Power | W idle and running, inferences/s, mJ per inference (`xmutil platformstats`) | |

### Setup notes

- Vivado runs only on x86-64 Linux or Windows. Budget 100+ GB disk and 16 GB+ RAM. Install Vitis (includes Vivado and
  Vitis HLS) with only Zynq UltraScale+ devices, then add the KR260 board files from the Vivado Store.
- The KR260 may need a boot firmware update before Ubuntu 22.04 boots. Follow AMD's KR260 "Booting your Starter Kit" guide,
  then `sudo bash install.sh -b KR260` from Kria-PYNQ.
- **AXI DMA: set "Width of Buffer Length Register" to 26.** The default 14 bits caps a transfer at 16,383 bytes, smaller
  than one input packet.
- Copy the `.bit` and the `.hwh` (same base name) to the board. PYNQ needs both.

## M5: from ~100 ms to under 5 ms

v0.1 does one multiply-add per clock: about 10 million cycles, ~100 ms at 100 MHz. That is expected; M4 is about correctness.

| Step | Multiply-adds per clock | Est. time |
|---|---|---|
| v0.1 at 100 MHz | 1 | ~100 ms |
| clock to 200 MHz | 1 | ~50 ms |
| unroll the reduction 8 wide, reshape weights and activations to match | 8 | ~6 ms |
| also unroll 8 output channels | 64 | ~1 ms |
| dataflow between layers with line buffers, or the RTL engine | more | < 1 ms |

The kernel is already shaped for this: weights are `[out][reduction]` arrays and activations are HWC, channels innermost.
FC6 and conv5 weights sit in UltraRAM packed 8 per word; unpacked, FC6 alone would need 73 URAMs (there are 64).

## Open questions

1. Who on the FPGA side owns compute (HLS/RTL) and who owns DMA, block design and board integration?
2. Bitstream rebuild limits: decide once we see build times on the PC.
3. The ARM CPU baseline row: compile `hls/policy_kernel.h` with `-O3` on the board, or use ONNX Runtime? The C kernel
   needs no extra work, since it is the same code.
