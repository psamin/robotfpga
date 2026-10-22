# SmolVLA on the KR260 FPGA, full precision

Decided 2026-10-13 by the repo owner. Task: the SO-101 picks up a medicine bottle and places it on a plate.

| Decision | Choice |
|---|---|
| Model | The fine-tuned SmolVLA (trained on our SO-101 demos, `smolvla-so101_medicine_bottle_pickplace-9604`). No new policy is trained without the owner's approval. |
| Precision | Full precision: weights stored **BF16** (as in the checkpoint), activations and all arithmetic **FP32**. No quantization. |
| Split | **Linear layers on the FPGA** (almost all of the compute); softmax, norms, activations, RoPE and the flow-matching update on the ARM cores. |
| Int8 | Retired. The earlier cube-sorting TinyPolicy and int8 engines are preserved at the git tag `archive/cube-sorting`. |

Citation keys point to [fpga-ddr-streaming-references.md](fpga-ddr-streaming-references.md).

## Engine: `fpga/hls/float_linear.h`

`Y[n][m] = Σ_k X[n][k] · W[m][k]`: W in DDR as BF16, X and Y FP32.

| Design rule | Value | Why / source |
|---|---|---|
| Weights stay in DDR and stream in; never all on chip | 0.9 GB of BF16 weights vs ~2.9 MB of on-chip RAM | SmolVLA is ~450 M parameters |
| Four weight ports in parallel | `m_axi` bundles `hp0..hp3` → S_AXI_HP0–3_FPD, 128-bit each | measured 16–18 GB/s on the K26 this way [Li2025-KV260LLM], [Li2025-Hummingbird], [AMD-PG201] |
| Clock | 300 MHz target | 4 × 16 B × 300 MHz = 19.2 GB/s, the DDR peak [AMD-DS925] |
| Lanes | 32 output channels per cycle (8 BF16 per port word) | lanes divide the bus word [Ma2020-PerfModel] |
| Weight layout | pre-tiled in consumption order: port, 32-channel group, k ascending; one flat burst per load | [AMD-UG1399-burst] |
| Bursts | `max_read_burst_length = 256` (4 KB), `num_read_outstanding = 16` | [AMD-UG1399-burst], [AXI4-4KB] |
| Overlap | ping-pong tiles: next group loads while the current one computes | [Zhang2015-Roofline] |
| Weight reuse | 32 tokens kept on chip per pass, so each weight is read once per 32 tokens | roofline: tokens per weight byte decide memory vs compute bound [Williams2009-Roofline], [Zhou2026-VLAXPU] |
| FP32 accumulation at II = 1 | 8 interleaved partial sums per (token, lane), fixed summation order | FP32 adder latency |
| Long reductions | K > 4096 split by the ARM into chunks that it sums | SmolVLM connector has K = 12,288 |
| Hybrid boundary | one FPGA call per linear layer; nonlinear ops on the A53 | nonlinear share was ~2% of a token on a KV260 LLM [Xiang2025-QwenKV260]; keep PS↔PL handoffs few [Zeng2024-FlightLLM] |
| VLM reuse | VLM prefix runs once per observation; the 10 flow-matching steps reuse its K/V | [Shukor2025-SmolVLA] |

Rules 15–18 and 22 of the reference table are int8-specific and don't apply at full precision.

## Files

| File | What |
|---|---|
| `fpga/hls/float_linear.h` | the engine core (HLS-synthesizable C++) |
| `fpga/hls/float_top.cpp`, `run_float_hls.tcl` | Vitis HLS top level and build script (300 MHz, xck26) |
| `fpga/hls/tb_float.cpp`, `Makefile` | C simulation against a float64 reference |
| `fpga/hls/float_shim.cpp` | the same core as a shared library, used to run the whole model in PyTorch |
| `fpga/hls/pack_float.py` | BF16 conversion, port packing, test cases, ctypes loader |
| `fpga/hls/latency.py` | latency model per action chunk |
| `policy/smolvla_fpga.py`, `cluster/smolvla_fpga.sbatch` | the fine-tuned SmolVLA with every linear layer computed by the engine code, vs plain PyTorch FP32 |
| `fpga/hls/build_bd.tcl` | Vivado block design + bitstream: 4 × 128-bit HP ports + HPC0, 300 MHz (untested) |
| `fpga/board/runtime.py` | KR260 runtime: `fpga_linears(policy, Engine("float.bit"))` swaps every `nn.Linear`, frees the PyTorch copy of each weight; `Engine(mock="layout")` runs the board's exact per-chunk DDR layout on the host |
| `fpga/board/check_engine.py` | every engine test case through the runtime; reports bandwidth on the board |
| `fpga/board/run_smolvla.py` | the fine-tuned SmolVLA on the board, replaying real frames dumped on the cluster |

## Verification

All runs on the GT cluster, 2026-10-13/14, fine-tuned checkpoint `smolvla-so101_medicine_bottle_pickplace-9604`
(450,046,176 parameters, stored BF16 + FP32; 302 executed linear layers).

**1. Engine C simulation** (`make -C fpga/hls test`, `tb_float` vs float64 reference)

| Case | Max error / Σ\|x·w\| | |
|---|---|---|
| Random 2560×960, 720 (padded 736)×960, 960×12288 (3 K-chunks), 960×960 (1 token), 32×7 | ≤ 1.2e-7 | PASS |
| Real SmolVLA layers with real activations: vision q_proj 768×768, vision MLP 3072×768 and 768×3072, connector 960×12288 | ≤ 2.9e-7 | PASS |

**2. Whole model, every linear layer computed by the engine code** (`policy/smolvla_fpga.py`, 5 real frames
from the training dataset, same noise; reference = PyTorch FP32)

| Frame | Max \|engine − PyTorch\| (50 steps × 6 joints) | Mean | First action vs recorded: PyTorch / engine |
|---|---|---|---|
| 0 | 0.125° | 0.014° | 0.561° / 0.564° |
| 1781 | 0.049° | 0.010° | 0.549° / 0.554° |
| 3563 | 0.051° | 0.009° | 0.261° / 0.268° |
| 5345 | 0.069° | 0.010° | 0.842° / 0.847° |
| 7489 | 0.042° | 0.009° | 0.789° / 0.791° |

The engine reproduces the model to about 0.01° on average; the remaining difference is FP32 summation
order and the few FP32-stored weights rounded to BF16. Accuracy against the demonstrations is unchanged.

**3. Latency model** (`fpga/hls/latency.py`, 300 MHz, 16.2 GB/s, per 50-action chunk)

| Where the work is | BF16 weights | Calls per chunk | Tokens per call | GMAC |
|---|---|---|---|---|
| Vision encoder (SigLIP, 2 cameras) | 170 MB | 144 | 1,024 | 173.9 |
| Connector | 24 MB | 2 | 64 | 1.5 |
| VLM text layers (prefix, once) | 315 MB | 112 | 177 | 27.8 |
| Action expert (10 flow steps) | 196 MB | 1,120 | 50 / 177 | 51.2 |
| Projections | 3 MB | 41 | 1–50 | 0.8 |
| **Total** | **0.71 GB** | **1,419** | | **255.3** |

| FP32 MACs per clock (TP) | Weight streaming | Compute | Time per chunk |
|---|---|---|---|
| 64 (TP = 2, as committed) | 1.18 s | 13.3 s | 13.3 s |
| 128 (TP = 4) | 1.18 s | 6.7 s | 6.7 s |
| 256 (TP = 8) | 1.18 s | 3.3 s | **3.3 s** |

**At full precision SmolVLA is compute-bound, not bandwidth-bound**: two thirds of the work is the vision
encoder's 1,024 tokens per image, and each FP32 multiply-add costs ~3–5 DSPs (1,248 on the chip), so
~256 MACs/clock (TP = 8) is the practical ceiling. Expect **~3–4 s per 50-action chunk** (estimate until
synthesis). A chunk covers 1.67 s at 30 Hz, so the arm runs the chunk slower than 30 Hz or waits
between chunks. Options, each the owner's call: BF16 multiplies with FP32 accumulation (several times
more MACs per DSP), running the vision encoder at a lower rate, or fewer camera tokens (a model change).

## FPGA side (2026-10-16, branch `fpga/203-float-engine`, #203)

Synthesis on the FPGA laptop (Vitis HLS 2022.2, Windows) of the engine as committed found four things the C
simulation could not show, all fixed on the branch with the host cases and the whole model unchanged
(0.026° max on the base model, 0.0043 normalized on the fine-tuned checkpoint through the board's DDR layout):

| Found in synthesis | Fix |
|---|---|
| `red` loop II = 7 and 515 FP32 adders at TP = 2: the runtime `k % PS` index into fully partitioned registers became muxes HLS could not pipeline | MAC loop over (k, token block) with the block innermost at II = 1; partial sums in LUT RAM indexed by block and `k mod PS`, revisited after NB·PS ≥ 16 cycles (`DEPENDENCE` distance); one multiplier and one adder per lane |
| Weight tile stores II = 4 (8 stores per 128-bit word) and no bursts on the struct-typed `m_axi` pointers (II = 2 per word) | DDR words are `ap_uint<128>` under Vitis (plain structs on the host); the four ports load in one loop (4 bursts at once) |
| Load/compute overlap not real (sequential calls, #167) | the group loop is a `DATAFLOW` region: `tile` is double-buffered by HLS |
| X/Y over a 32-bit io port cost 25–35 % at TP = 4 | 128-bit io words (K must be a multiple of 4; SmolVLA's are) |

Results (TP = 4): `mac`, `load`, `load_x`, `store_row` all II = 1; bursts on hp0–hp3 and io; 644 DSP, 83,431 LUT,
113,161 FF, 48 URAM, 0 BRAM; estimated clock 3.602 ns against the 3.333 ns target (PYNQ can lower `fclk0` at
runtime, `Engine(clock_mhz=...)`). **TP = 8 does not fit**: an FP32 multiply-add costs ~3 DSP plus a fabric adder,
so 256 lanes exceed the 1,248 DSPs; TP = 4 (128 MACs/clock, ~6.7 s per chunk by the latency model) is the FP32
ceiling on this chip. Windows notes: `open_project` must not contain `/`, csim predefines `NT`, and a Windows
update mid-session broke every Vivado implementation until a reboot (`fpga/hls/resume_impl.tcl` resumes one).

Board chain (`fpga/board/deploy_smolvla.py`): `--setup` installs Python 3.12 + the LeRobot fork offline next to
PYNQ's 3.10; the default run pushes code, bitstream, checkpoint and batches, runs `check_engine.py`, starts
`engine_server.py` as root and `serve_policy_fpga.py --selftest`, and with `--serve` leaves the policy server up.
Post-route (Vivado 2022.2, `build_bd.tcl` + `resume_impl.tcl`): WNS −0.026 ns at 300 MHz with hold met, i.e. the
design closes at ~298 MHz; it runs at 250 MHz on the board (`--clock-mhz 250`, 16 GB/s on the weight ports). 52,662
LUT (45 %), 92,506 FF (39 %), 74.5 BRAM tiles (52 %), 48 URAM (75 %). On the board (2026-10-22): `check_engine.py` all 5 cases PASS; `serve_policy_fpga.py --selftest` PASS, max
0.0043 vs PyTorch FP32 on the release frames, **52–56 s per 50-action chunk** (~32 s in the 1,419 engine calls
including socket copies, ~22 s on the ARM). Board fixes this needed: weights in 32 MB CMA slabs (`runtime.Engine`),
and `load_vlm_weights=false` in the board's config copy (avoids loading the 2 GB SmolVLM2 backbone before the
checkpoint overwrites it; 3.19 → 2.35 GB peak RSS, identical actions).

## Next steps (FPGA side)

1. `make -C fpga/hls test`, then `vitis_hls -f run_float_hls.tcl` in `fpga/hls/`: C sim, synthesis at
   300 MHz, IP export. Report timing, II of the `red` loop, DSP/BRAM/URAM use.
2. Block design: Zynq MPSoC with the KR260 preset; `float_top` bundles `hp0..hp3` → S_AXI_HP0..HP3_FPD,
   `io` → S_AXI_HPC0_FPD; PL clock 300 MHz; AXI-Lite on M_AXI_HPM0_FPD. Bitstream → `float.bit` +
   `float.hwh`.
3. On the board: measure the GEMM bandwidth with `fpga/board/runtime.py` (compare with 16.2 GB/s),
   then run the fine-tuned SmolVLA with `fpga_linears` and compare actions with the PyTorch FP32 model.
4. The design is compute-bound (Verification §3): raise `TP` in `float_linear.h` toward 8 until DSPs or timing
   run out, and report the achieved FP32 MACs per clock.
