# FPGA plan: KR260 tiny policy accelerator

Status as of 2026-10-01: no hardware or Xilinx tools yet. Everything under "Done" was
built and checked on a laptop.

## Done (no hardware needed)

| Piece | Where | Checked how |
|---|---|---|
| Bit-exact int8 reference (stand-in until Praneeth's lands) | `ref/intref.py` | 18 pytest checks, including a slow loop-by-loop conv |
| Stand-in random weights + 100 golden vectors in the handoff format | `ref/make_standin.py` → `artifacts/standin/` | round-trips through `weights.bin` + `manifest.json` |
| HLS C++ kernel, whole network | `hls/policy_kernel.h` | matches all 100 vectors and every layer dump, also under ASan/UBSan |
| Vitis top level (AXI-Stream + AXI-Lite) and its testbench | `hls/policy_top.cpp`, `hls/tb_top.cpp` | compiled and passed against stub headers; real check is Vitis csim |
| Vitis HLS build script | `hls/run_hls.tcl` | not run yet |
| Board scripts for M1 and M4 | `board/loopback.py`, `board/run_policy.py` | not run yet |

The testbench was checked to fail when the rounding term is removed or the padding is off by one.

## Tomorrow, in order

### 1. PC: start the Vivado download first (it is the long pole)

- Vivado does not run on macOS. It needs x86-64 Linux (Ubuntu 22.04 is the easy one) or Windows 10/11.
- Budget 100+ GB free disk and 16 GB RAM (32 GB is more comfortable).
- From amd.com (free account), get the **AMD Unified Installer for FPGAs & Adaptive SoCs**, web installer.
  Choose **Vitis** (it includes Vivado and Vitis HLS). In device selection, keep only
  **Zynq UltraScale+ MPSoC** to cut the download size. The free edition covers the K26.
- Pick one version and stay on it for the whole project. Write it down; the FPGA brief asks for it.
- After install: Vivado → Tools → Vivado Store → Boards → install **Kria KR260**.

### 2. Board: M0 bring-up (while Vivado downloads)

1. Flash the **Ubuntu 22.04 for AMD Kria** image to a microSD card (32 GB+) with Balena Etcher or Raspberry Pi Imager.
2. Follow AMD's "Getting Started with KR260", section "Booting your Starter Kit". The Kria-PYNQ README
   warns that Ubuntu 22.04 may need a **boot firmware update** on the board; that guide covers it.
3. Connect Ethernet (same network as your laptop) and the USB cable for the serial console (115200 baud).
   First login is `ubuntu` / `ubuntu` and forces a password change.
4. Install PYNQ (about 25 minutes):
   ```
   git clone https://github.com/Xilinx/Kria-PYNQ.git
   cd Kria-PYNQ/
   sudo bash install.sh -b KR260
   ```
5. Open Jupyter from the laptop at `http://<board-ip>:9090/lab` and run one of the bundled example notebooks.

**M0 passes** when an overlay loads and runs from Python.

### 3. M1: DMA loopback (first Vivado design)

Block design in Vivado, project targeting the KR260 board:

1. Add **Zynq UltraScale+ MPSoC**, run Block Automation (applies the board preset).
2. In the MPSoC settings, enable a slave port for the DMA to reach DDR: **S_AXI_HP0_FPD**.
3. Add **AXI Direct Memory Access**:
   - uncheck Scatter Gather
   - **Width of Buffer Length Register: 26.** The default (14 bits) caps a transfer at 16,383 bytes,
     which is smaller than our 27,658-byte packet and the 568,240-byte weight load.
   - stream data width 8 bits (matches the HLS kernel for now)
4. Add **AXI4-Stream Data FIFO**. Wire DMA `M_AXIS_MM2S` → FIFO → DMA `S_AXIS_S2MM`.
5. Run Connection Automation, validate, create HDL wrapper, generate bitstream.
6. Copy two files to the board with the same base name: `loopback.bit` and `loopback.hwh`
   (the `.hwh` is under `<project>.gen/sources_1/bd/<design>/hw_handoff/`).
7. On the board: `python3 board/loopback.py loopback.bit`.

**M1 passes** when all bytes match. Write down the p50 round-trip time: that is our latency floor.

### 4. HLS: C simulation and synthesis on the PC

```
python3 ref/make_standin.py --out artifacts/standin    # needs numpy
cd hls
vitis_hls -f run_hls.tcl                     # 2023.2 and older
vitis-run --mode hls --tcl run_hls.tcl       # 2024.1 and newer
```

Check, in `hls/build/policy_hls/sol1/syn/report/`:
- csim prints `PASS: 100 vectors through policy_top`
- timing met at 100 MHz
- BRAM, URAM, DSP, LUT use. Expect FC6 and conv5 weights in URAM (they are packed 8 per word on purpose).
- latency: about **10 million cycles ≈ 100 ms at 100 MHz**. That is expected for v0.1, which does one multiply-add
  per clock. Correctness first, speed in M5.

### 5. M4: full network on the board

Same block design as M1 with the FIFO replaced by the exported `policy_top` IP (add the HLS export folder as an
IP repository). Connect its `s_axi_control` to the MPSoC's `M_AXI_HPM0_FPD` through Connection Automation.
Then `python3 board/run_policy.py policy.bit artifacts/standin`.

## M5: getting from ~100 ms to under 5 ms

| Step | Multiply-adds per clock | Est. time |
|---|---|---|
| v0.1 as written, 100 MHz | 1 | ~100 ms |
| raise the clock to 200 MHz | 1 | ~50 ms |
| unroll the channel (reduction) loop 8 wide, reshape weights and activations to match | 8 | ~6 ms |
| also unroll 8 output channels | 64 | ~1 ms |
| dataflow between layers with line buffers, or the hand-written RTL engine | more | below 1 ms |

The code is already shaped for this: weights are `[out][reduction]` 2-D arrays and activations are HWC with channels innermost.

## Spec notes to send Praneeth

None of these are contradictions between the two docs. They are things the spec leaves implicit.

1. **The shift must be at least 1.** `1 << (s - 1)` is undefined at `s = 0`. The hardware rejects 0 with status 1.
2. **Conv1 activations can be -128** (pixel 0 minus 128), even though weights stop at -127. Matters for DSP packing.
3. **Manifest format.** `ref/intref.py` defines one (`layers[].shift`, `weight_offset`, `bias_offset`, `sha256`...).
   `policy/export.py` should write the same keys, or we agree on theirs.
4. With stride 2, padding 1 and even sizes, the bottom and right padding are never read. Same as PyTorch `padding=1`.

## Open questions from the FPGA brief

1. Board: **KR260**. Vivado/Vitis version: decide tomorrow.
2. HLS first (this repo), RTL conv engine later as the "own GPU" track.
3. Who owns compute vs DMA/board integration: TBD.
4. Bitstream rebuild limits: TBD once we see build times on the PC.
