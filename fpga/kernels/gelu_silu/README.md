# gelu_silu: element-wise activation + fused per-token int16 quantisation

One unit for the three element-wise activations of SmolVLA (checkpoint -12392). It reads the int16 engine's FP32
output and writes int16 rows plus one FP32 scale per row, ready for the next int16 linear:

| mode | y | where (per 50-action chunk) | sequencer op |
|---|---|---|---|
| 0 | tanh-GELU(x) (`gelu_pytorch_tanh`) | SigLIP fc1 -> fc2: 12 layers x 512 rows x 3072 | `OP_GELU` |
| 1 | silu(gate) * up | LM 16 x 81 x 2560, expert 10 steps x 16 x 50 x 2048 -> down_proj | `OP_SILU_MUL` |
| 2 | silu(x) | action-time MLP (`F.silu`, modeling_smolvla.py:673) 10 x 50 x 720 | `OP_SILU_MUL` + `F_NOMUL` |

Per row: `q[k]` int16 and `s` FP32 with `q * s ~= y`, `s ~= max_k |y_k| / 32767`, `|q| <= 32767`. That is the
engine's per-token activation format (`int16_linear.h`, `pack_float.quantize_int16_tokens`).

## Files

| file | what |
|---|---|
| `gelu_silu.h` | header-only HLS core (`gs::gelu_silu_core`), the arithmetic spec in its top comment |
| `gelu_silu_tables.h`, `gs_tables.py` | the ROM (generated: `python gs_tables.py`) |
| `gelu_silu_top.cpp`, `run_hls.tcl` | standalone top with float_top's bus shape, csynth script |
| `tb_gelu_silu.cpp`, `make_vectors.py`, `run_tests.sh` | host testbench (links the HLS source), vector generator, test runner |
| `gs_emu.py` | bit-exact torch emulation of the kernel (the reference the testbench and the policy check use) |
| `cpp_mirror.py` | line-by-line Python transliteration of the C++ (explicit ap_uint widths) vs `gs_emu` |
| `policy_check.py` | full-policy effect on -12392's 3 replay frames (monkeypatches the activations) |
| `time_model.py` | time per chunk from the csynth cycle counts |

## Arithmetic

Both activations are `x * h(x)` with a sigmoid-like gate (`h = Phi_tanh` for GELU, the logistic for SiLU) and
`h(-t) = 1 - h(t)`. The kernel tabulates `g(t) = 1 - h(t)` for `t = |x|` on `[0, 8)` (GELU) / `[0, 16)` (SiLU), with
`g = 0` beyond:

1. `t` to fixed point, `TB = NSB + 12` bits: segment index `NSB` bits, fraction `f` 12 bits (one 24-bit shift).
2. ROM word `V << 15 | D` (21 + 15 bits): `g21 = V - ((D * f + 2048) >> 12)` in Q0.21. `V` is the minimax offset
   over the 4096 sub-positions, so truncating `t` is unbiased (`gs_tables.py`). NSB = 9: 512 segments per mode, GELU
   step 1/64, SiLU 1/32; both modes share one 1024 x 36 ROM = 1 BRAM36, dual-ported for 2 lanes.
3. `x >= 0`: `Y = (mx << 18) - mx * g18` (g in Q0.18) = `|x|(1 - g)`. `x < 0`: `Y = mx * gn`, where `gn` is `g21`
   normalised to a 17-bit significand, so small `g` (the tails) keep their relative precision. One DSP (24 x 17)
   either way. The leading one of `Y` is at bit 41, 40 or 39, so normalising it is a 3-way select.
4. Modes 0 and 2: `y` = FP32 with a 24-bit significand, round half up (flush to zero below the normal range).
   Mode 1: `x h(x)` rounded to 17 bits times the 24-bit significand of `up` (one DSP, 24 x 17), rounded to 24 bits,
   saturated at FP32 max.
5. Per row: `amax` = max `|y|` (FP32 bit compare). `rq = floor(32767 * 2^25 / ma)` (17 bits) is a 17-step divider,
   `r = rq * 2^(125 - ea) <= 32767 / amax`. `s = 1 / r` is correctly rounded (42-step divider), so `q * s` matches the
   `r` actually used. Rows with amax < 2^-111 give q = 0, s = 0.
6. `q = round_half_up(|y| * r)` with the sign: one DSP (24 x 17) + a 17-bit shift.

All integer: `gs_emu.py` is bit-exact, and `cpp_mirror.py` / `tb_gelu_silu.cpp` check that the C++ matches it bit
for bit. NaN / Inf inputs are not handled specially (garbage in, garbage out).

### Two passes over a buffered row

The per-token absmax needs the whole row before the first q, so each row goes through an on-chip row buffer (the
`ys` FIFO). Pass 1 computes y and absmax and pushes y. A sequential divider pair turns absmax into the scale (69
cycles per row, overlapped). Pass 2 pops the row and quantises it. DATAFLOW overlaps pass 1 of row n+1 with pass 2
of row n, so each element costs one DDR read of x (and up) and one 16-bit write, with no second DDR pass. Cost: one
row + 128 entries of slack, KMAX = 3072. 8 lanes: 512 x 256 bit = 15 BRAM18, or 4 URAM with `GS_YS_URAM`. 4 lanes:
896 x 128 bit = 8 BRAM18. The buffer must exceed one row or the pipeline deadlocks, so `K > KMAX` is rejected.

## Interface

Standalone top `gelu_silu_top` (bundles exactly as float_top):

| port | bundle | use |
|---|---|---|
| `w0` .. `w3` | m_axi `hp0` .. `hp3`, 128 bit, read | 8 lanes: x first half (hp0), x second half (hp1), up first half (hp2), up second half (hp3). 4 lanes: x on hp0, up on hp1 |
| `ymem` | m_axi `io`, 128 bit, write | q rows and the scale vector |
| `mode N K a pa b pb c pc f` | s_axi_control (32-bit each) | below; offsets and pitches in 16-byte words relative to the pointers |
| return | s_axi_control | 0, or 1 for a bad shape (nothing read or written) |

| register | meaning | sequencer slot |
|---|---|---|
| `mode` | 0 GELU, 1 SiLU(x) * up, 2 SiLU(x) | opcode, `F_NOMUL` |
| `N` | rows >= 1 | f4 N |
| `K` | elements per row, `K % 8 == 0`, `8 <= K <= 3072` | f3 K |
| `a`, `pa` | x (mode 1: gate) row n at word `a + n * pa`, K FP32 contiguous | A.addr >> 4, A.pitch >> 4 |
| `b`, `pb` | mode 1 up row n at word `b + n * pb` | B |
| `c`, `pc` | q row n at word `c + n * pc`, K int16 = K/8 words, element order below | C |
| `f` | scales: N FP32, 4 per word (lane j of word i = row 4i + j), ceil(N/4) words; lanes past N written 0 | F (16-byte aligned) |

Word formats: FP32 / int16 lane j of a 128-bit word sits at bits `[32j+31:32j]` / `[16j+15:16j]` (little-endian, as
`float_linear.h`). All addresses and pitches must be 16-byte aligned. Writing the last scale word can touch up to 3
floats past the vector; the sequencer's `xs` role is padded to 16 bytes.

### Element order (8 lanes)

The 8-lane build reads each row as two contiguous halves, so it keeps one burst per port per row. Output word w of a
row is `[q[4w..4w+3], q[K/2+4w..K/2+4w+3]]`: position p holds element `gs_emu.out_perm(K)[p]`. The next linear's
sums are exact integers, so either of these gives bit-identical results with no reorder hardware (checked
exactly in this directory):

- A: build the next linear's weights (fc2, down_proj, action_time_mlp_out) as `W[:, gs_emu.next_weight_cols(K)]`.
- B: build the previous linear's rows (fc1, gate_proj and up_proj each, action_time_mlp_in) and bias as
  `W[gs_emu.prev_weight_rows(K), :]`. The q rows then come out in natural order, and only the FP32 intermediate is
  permuted.

The 4-lane build (`GS_LANES=4`) writes natural order: no permutation, about half the area, twice the time.
**fpga/kernels/sequencer's contract says "int16 row-major", so with 8 lanes its weight-blob builder has to apply
A or B.** It owns build_smolvla.py, so that is an open item for that side.

## Resources and time (Vitis HLS 2022.2 csynth, xck26-sfvc784-2LV-c, NSB 9)

The kernel itself is the DATAFLOW region. The standalone top adds five m_axi adapters (2,318 LUT each) and
s_axi_control (1,384 LUT), which float_top already has.

| build | clock | slack | LUT | FF | DSP | BRAM36 | URAM | cycles per element |
|---|---|---|---|---|---|---|---|---|
| 8 lanes | 4.0 ns | 0.00 | 23,607 | 20,907 | 24 | 11.5 (tables 4 + row FIFO 7.5) | 0 | 1/8 (II = 1, 8 per clock) |
| 8 lanes, `GS_YS_URAM` | 4.0 ns | 0.00 | 23,607 | 20,907 | 24 | 4 | 4 | 1/8 |
| 8 lanes | 3.333 ns | -0.00 (marginal) | 23,351 | 22,389 | 24 | 11.5 | 0 | 1/8 |
| 4 lanes | 4.0 ns | 0.00 | 13,983 | 11,721 | 16 | 6 (tables 2 + row FIFO 4) | 0 | 1/4 |

- These are HLS estimates. build/brief measured post-route LUT at 0.68-0.74x HLS for the engine, so roughly 16.5k
  (8 lanes) / 9.8k (4 lanes) post-route. Not measured here: Vivado was not run.
- 8 lanes, by process: pass 1 18,951 LUT / 16 DSP / 4 BRAM36 (II 1, depth 19), scale 659 LUT, pass 2 3,252 LUT /
  8 DSP (II 1, depth 7). DSPs: 3 per lane, for `mx * G`, `m17 * mu` and `my * rq`. The 15 x 12 interpolation
  multiply is fabric (246 LUT/lane); `GS_DF_DSP=1` moves it to 8 DSPs. That build was not synthesised.
- 4 lanes: pass 2 keeps 8 DSPs although its loop is II = 2; the ALLOCATION pragma had no effect.
- NSB 10 (better tables, below) doubles the ROMs: +4 BRAM36 (8 lanes). Not synthesised.
- m_axi bursts were inferred on hp0..hp3 (8 lanes; hp0/hp1 for 4 lanes) and on io. Per-row overhead of pass 1 is
  31 cycles (row loop 415 cycles for 384 words).

Time per chunk at 250 MHz (`python time_model.py`). csynth models the DDR read latency as 0; the "+DDR" column
assumes 40 cycles of read restart per row and 12 GB/s. Both are assumptions, not measurements:

| | 8 lanes | 8 lanes +DDR | 4 lanes | 4 lanes +DDR |
|---|---|---|---|---|
| SigLIP GELU (18.9 M el.) | 10.2 ms | 11.2 ms | 19.7 ms | 20.7 ms |
| LM SiLU * up (3.3 M) | 1.9 ms | 2.8 ms | 3.5 ms | 3.7 ms |
| expert SiLU * up (16.4 M) | 9.4 ms | 13.7 ms | 17.8 ms | 19.0 ms |
| action-time SiLU (0.36 M) | 0.25 ms | 0.33 ms | 0.43 ms | 0.51 ms |
| **total (38.9 M el., 312 MB DDR)** | **21.7 ms** | **28.0 ms** | **41.4 ms** | **44.0 ms** |

At 8 lanes, mode 1 asks for 4 read words + 1 write word per clock (20 GB/s at 250 MHz), above what the DDR gives
(~14 GB/s per int16_linear.h). So the SiLU * up calls are DDR-bound there, and 4 lanes lose less than 2x on them.

## Accuracy

Kernel error against float64 torch, in units of the token's quantisation step s, over all activations of the 3
-12392 forwards (max / max-rms over the frames, policy_check.py F1 runs). "y" is the kernel's FP32 y before int16
rounding; "q" is the int16 output (ideal rounding alone: max 0.5, rms 0.289):

| use | elements per frame | y err max / rms (NSB 9) | q err max / rms (NSB 9) | y err max / rms (NSB 10) |
|---|---|---|---|---|
| GELU, K 3072 | 18.9 M | 0.33 / 0.034 | 0.78 / 0.263 | 0.14 / 0.012 |
| LM SiLU * up, K 2560 | 3.3 M | 1.39 / 0.033 | 1.52 / 0.291 | 0.87 / 0.015 |
| expert SiLU * up, K 2048 | 16.4 M | 1.34 / 0.029 | 1.58 / 0.290 | 0.50 / 0.015 |
| action-time SiLU, K 720 | 0.36 M | 0.26 / 0.040 | 0.73 / 0.292 | 0.13 / 0.015 |

The mode-1 maxima come from the tails: x << 0 times a large up in rows with a small absmax (LM rows go down to
amax 0.148).

Full policy, `predict_action_chunk(batch, noise)` vs `ref_chunk_raw`, max |error| in normalized action units (gate
0.05), frames 0 / 3563 / 7489:

| variant | frame 0 | frame 3563 | frame 7489 |
|---|---|---|---|
| F0: FP32 linears, torch activations (laptop vs reference) | 5.1e-6 | 4.6e-7 | 4.8e-7 |
| F2: FP32 linears, ideal int16 per-token quantiser after the activations | 4.7e-4 | 5.0e-5 | 5.2e-5 |
| F1: FP32 linears, this kernel (NSB 9) | 1.8e-4 | 6.3e-5 | 8.5e-5 |
| I0: int16 engine emulation, ideal quantiser | 2.8e-4 | 1.0e-4 | 1.2e-4 |
| **I1: int16 engine emulation + this kernel (NSB 9)** | **4.8e-4** | **1.4e-4** | **1.1e-4** |
| I1 with NSB 10 | 8.6e-4 | 1.1e-4 | 1.2e-4 |

The kernel's own effect (I1 - I0, max over the chunk) is 7.6e-4 / 6.5e-5 / 1.4e-4, within the 1e-4..1e-3 class.
Frame 0 amplifies any perturbation: the ideal quantiser alone moves it 4.7e-4, and the more accurate NSB 10 moves it
more. **Recommendation: NSB 9** (half the ROM). The per-variant JSON is `build/gelu_silu/policy_check.json`.

## Verification state

- `gs_emu` against float64 torch: the tables above.
- `cpp_mirror.py` (Python transliteration of every C++ statement with explicit widths) is bit-exact to `gs_emu` on
  20 cases x 2 lane counts. Those are synthetic random / edge rows (zeros, denormals, segment boundaries, +-R, 1e4,
  3e38, up = 1e30 saturation, all-zero rows) and real -12392 activations of all four uses, NSB 9 and 10.
- `tb_gelu_silu.cpp` links the HLS source itself. It checks bit equality with `gs_emu`, pitched sequencer-style
  layouts with guard words, the zero padding of the scale word, and bad-shape rejection. **It compiles (both lane
  counts) but was NOT run on this laptop: Windows Smart App Control (enforced) blocks every newly built .exe.** Run
  `sh run_tests.sh` where executables are allowed.
- No C/RTL co-simulation, no Vivado, no board run.

## Integration into float_top

1. Include `gelu_silu.h` and call `gs::gelu_silu_core(w0, w1, w2, w3, ymem, mode, N, K, a, pa, b, pb, c, pc, f)`
   from one site of the sequencer's dispatch (OP_GELU -> 0, OP_SILU_MUL -> 1, or 2 with F_NOMUL). Wrap it
   `INLINE off` so the hardware exists once. Sequencer mode programs the pointers to 0, so the offsets are byte
   address >> 4.
2. Batch items (`B > 1`) are a loop around the call, or rows folded into N when the item stride equals N * pitch.
   build_smolvla uses B = 1 for these ops.
3. The core's word offsets are `ap_uint<36>` (40-bit byte addresses). It reads only hp0..hp3 and writes only
   io, the bundles float_top already has; no register is added beyond what the sequencer command carries.
4. Element order: 8 lanes need option A or B above in the weight-blob builder; 4 lanes need nothing.
5. If float_top's `WPort` / `IoWord` are not `ap_uint<128>` in a host build, add `to_word` / `from_word` overloads
   (gelu_silu.h, "port word access").
