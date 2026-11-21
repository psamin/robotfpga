# RoPE kernel (fpga/kernels/rope)

Rotary position embedding for SmolVLA checkpoint -12392 on the KR260, built to become a mode of
`float_top`. It applies lerobot's `apply_rope` (`smolvlm_with_expert.py:41-64`) to Q and K after the
q/k projections, and its INT16 mode writes the int16 + per-(token, head) scale operand of the attention
engine call in the same pass.

Status: the C model is bit-exact against the torch emulation, the full-policy check passes, and csynth
meets 250 MHz. There is no co-simulation, Vivado implementation, board run or float_top integration yet
(see "Not verified").

## What it computes (lerobot semantics)

- `x` is `[B, L, H, 64]`. `x1 = x[..., :32]`, `x2 = x[..., 32:]`.
  `y1 = x1 cos - x2 sin`, `y2 = x2 cos + x1 sin`.
  `cos/sin(pos / 10000^(2i/64))`, i = 0..31, are computed in FP32 (torch.float32 `radians`, `cos`, `sin`).
- Prefix (prefill, 16 layers): Q (15 heads) and K (5 KV heads) of the 81 prefix tokens, at
  `positions = cumsum(pad_mask) - 1`. Padding tokens repeat the previous position.
- Expert self-attention layers (even, 8 per step): Q (15) and K (5) of the 50 action tokens, at
  `prefix_valid_len + 0..49`. Positions continue after the prefix.
- Expert cross-attention layers (odd, 8 per step): Q only, at positions re-based to `0..49`. The keys
  are `k_proj(cached prefix K)`, which is not rotated again.
- GQA is 15 query heads over 5 KV heads, head_dim 64. Checked against the hooked policy: 272 calls,
  165,920 (token, head) rows per chunk.

## Design decisions

- **cos/sin from DDR, not on chip.** The ARM writes one table row per token: 64 int32 slots,
  `NC[0..31] | NS[0..31]`, where `NC = round(-cos * 2^17)` and `NS = round(-sin * 2^17)`, 18-bit,
  clamped to [-2^17, 2^17-1]. Negating lets 1.0 become -2^17 exactly. Reasons:
  - The table uses the port hp0, which the engine leaves idle in this mode.
  - It costs 0 BRAM. It is read once per token and reused for its H heads, so it is 1/H of the x traffic.
  - The ARM computes it with exactly torch's FP32 cos/sin (`rope_emu.table_rows`).
  - Positions are known at chunk start: per chunk there are 3 tables (prefix 81, expert self 50,
    cross 50 rows = 46 KB). They are constant while the prompt length is constant.
  - Alternatives: an on-chip position-indexed table (256 positions x 64 x 18 bit) is 8 BRAM36, and an
    on-chip sin/cos generator costs more LUT and DSP and no longer matches torch.
- **Natural head_dim order in DDR.** No weight permutation. A row is held in a ping-pong LUT-RAM so that
  the rotation partners `x1[i]` and `x2[i]` (8 words apart) meet. The output is written back in natural
  order. A previous version required permuting the q_proj/k_proj rows and the expert k_proj columns and
  rows offline; that coupling is gone.
- **Block-float integer rotation.** A group is 4 values (two rotation pairs), aligned to their max
  exponent: 27-bit signed x 18-bit table gives 8 DSP48E2 per cycle, and the result is an exact 45-bit
  integer. Each group is then normalized once with a 4-way shift (proved range, below) to 26 bits.
  MODE_F32 and MODE_INT16 both work from that.
- **Power-of-two scales** (`s = 2^e`, no mantissa). This needs no multiplier and no reciprocal. Its
  policy-level cost is in the same 1e-5..1e-4 class as the absmax/32767 format (table below). k-bit
  mantissa scales (qm = 1..6) exist in the emulation only. They did not improve the policy error in a
  measurable way, so they are not in the hardware.

## Arithmetic (bit-exact model: `rope_emu.py`)

For group u of a row, the values are `(x1[2u], x1[2u+1], x2[2u], x2[2u+1])`, lanes 0..3:

```
e_j = FP32 biased exponent (e = 0: zero / denormal -> 0), E = max_j e_j
A_j = sign_j * (((1 << 23 | frac_j) << 2) >> (E - e_j))           27-bit signed (0 if shift > 26)
O0 = A2 NS0 - A0 NC0   O1 = A3 NS1 - A1 NC1                          (y1[2u], y1[2u+1])
O2 = -(A2 NC0 + A0 NS0)   O3 = -(A3 NC1 + A1 NS1)                    (y2[2u], y2[2u+1])
value = O * 2^(E - 169),  |O| < 2^44
Lg = bitlen(max_j |O_j|) in [42, 45] if E > 0 (proof: O0^2 + O2^2 = (A0^2 + A2^2)(NC^2 + NS^2) >= 2^50 * 2^34 (1 - 2^-16)),
     group all zero if E = 0             (rope_emu asserts both on every modelled call)
N_j = |O_j| >> (Lg - 26)  (26 bits),  Mg = E + Lg  (0 if E = 0)
MODE_F32:   y = sign * N * 2^(Mg - 195), RNE to FP32, |y| < 2^-126 -> +0 (overflow -> inf)
MODE_INT16: Mtop = max over the row's 16 groups of Mg
            q = sign * min(32767, rnd(|O| >> (Mtop - E - 15)))  (half away from zero)
              = sign * min(32767, ((N >> a) + 1) >> 1),  a = clamp(Mtop - Mg + 10, 10, 26)   (exact)
            s = 2^(Mtop - 184)   (FP32 field Mtop - 57); Mtop <= 57 (all-zero / tiny row): q = 0, s = 0
```

The engine computes `q · s`. Denormal inputs are read as 0 (abs error <= 2^-126).

## Interface

### Standalone top `rope_top` (csynth)

Ports are as float_top has them: `x`, `y`, `s` on m_axi `io`, `cs` on m_axi `hp0` (read only), and
scalars on `s_axi_control`. Pointers are byte addresses (64-bit, 16-byte aligned). Pitches are in
128-bit words.

| offset | register | meaning |
|---|---|---|
| 0x00 | CTRL | ap_start / done / idle / ready |
| 0x10/0x14 | x | input rows (FP32) |
| 0x1c/0x20 | cs | table rows (hp0) |
| 0x28/0x2c | y | output rows |
| 0x34/0x38 | s | MODE_INT16 scales |
| 0x40 | T | tokens, 1..65535 |
| 0x48 | H | heads, 1..63 (16 H <= 1008) |
| 0x50 | x_pitch | words between tokens of x (== 16 H: one flat burst; else one burst per token) |
| 0x58 | y_pitch | MODE_F32: words between tokens of y (y = x, y_pitch = x_pitch: in place) |
| 0x60 | mode | 0 = MODE_INT16, 1 = MODE_F32 |

MODE_INT16 limit: `ceil(T/4) * H <= 512`. Prefix Q is 21 x 15 = 315 and expert Q is 13 x 15 = 195.

### DDR layouts (word = 16 bytes, lane j = bits [32j+31:32j], little endian)

- **x**: row (t, h) is the 16 words `x[t*x_pitch + 16h .. +15]`, FP32, natural order (words 0..7 =
  x1[0..31], 8..15 = x2[0..31]). This is exactly the q_proj output `[T][H*64]` (x_pitch = 16 H) or the
  K part of the fused k|v output `[T][640]` (x_pitch = 160, H = 5).
- **cs**: T rows of 16 words. Row t is token t's table row, int32 slots: words 0..7 = NC[0..31],
  words 8..15 = NS[0..31]. Only bits [17:0] of a slot are used.
- **y, MODE_F32**: same layout as x with y_pitch (in place allowed).
- **y, MODE_INT16**: int16 row (t, h) is the 8 words `y[8(tH + h) .. +7]`, value k in 16-bit lane
  k % 8 of word k / 8, natural order. The rows are contiguous, token-major: the same memory as an int16
  `[T][H*64]`.
- **s, MODE_INT16**: head-major. The scale of row (t, h) is the FP32 in lane t % 4 of word
  `s[h*ceil(T/4) + t/4]`. Each head's run is 16-byte padded with zero lanes, so it reads as
  `scales[h][a16(T)]`, the sequencer's `Slot(qs, 0, a16(T*4))`.

### Header-only core (`rope_core.h`, namespace `rope`)

- `rot_stage(xs, cs, rs, rows, H)`: x words in, table words in (only on the first head of a token),
  rotated groups out.
- `out_stage(rs, qs, ss, rows, H, tq, nsw, mode)`: FP32 words, or int16 words followed by `nsw` scale
  words.
- Both are II = 1 and keep no state between calls. float_top supplies the movers: its `xmem` (io read)
  for x, `w0` (hp0) for cs, and `ymem` (io write) for y and s. `rope_top.cpp`'s `read_rows`,
  `read_tab` and `write_out` are the reference movers.

## Resources (Vitis HLS 2022.2 csynth estimates, xck26-sfvc784-2LV-c)

At 4.0 ns (250 MHz) timing is met, with an estimated Fmax of 342 MHz:

| block | LUT | FF | DSP | BRAM18 |
|---|---|---|---|---|
| rot_stage (gather + align + 8 DSP rotation, table/x ping-pong LUT-RAM) | 2,100 | 2,570 | 8 | 0 |
| out_stage (normalize, row max, FP32 / int16 + scales, reorder LUT-RAM, scale RAM) | 4,831 | 2,784 | 0 | 1 |
| **core total** (what float_top gains) | **6,931** | **5,354** | **8** | **1** |
| stream FIFOs (xs, csr, qs 16 x 128 SRL; rs, ss; scalars) | 1,337 | 1,980 | 0 | 0 |
| reference movers (read_rows 873, read_tab 389, write_out 1,613) | 2,875 | 2,754 | 6 | 0 |
| rope_df (dataflow region) | 11,280 | 10,094 | 14 | 1 |
| rope_top incl. m_axi adapters + s_axilite (float_top has its own) | 18,363 | 12,833 | 14 | 1 |

At 3.333 ns (300 MHz) timing is not met: the dataflow-level slack is -0.07 ns (estimated 2.49 ns
against a 2.43 ns budget). Resources at 300 MHz are within 1% of the 250 MHz numbers.

Earlier version on disk before this rework: PI-permuted DDR order and 45-bit stored magnitudes. It
estimated 23,418 LUT and 9 DSP, with timing failing at 146 MHz.

## Throughput and work per chunk

- One input word (4 elements) per cycle in every mode, so **0.25 cycles per element**. Writes are
  1 word/cycle (MODE_F32) or 0.5 word/cycle plus `ceil(T/4) * H` scale words after the rows
  (MODE_INT16).
- Per call: about `16 * T * H + ~80` cycles of pipeline, plus `ceil(T/4) * H` (MODE_INT16), plus the
  DDR read latency. When `x_pitch != 16 H`, add the DDR latency again per token, because there is one
  burst per token. That latency is unknown.

| call (per chunk) | count | T x H | mode | rows |
|---|---|---|---|---|
| prefix Q | 16 | 81 x 15 | INT16 | 19,440 |
| prefix K (in place in k\|v, pitch 160) | 16 | 81 x 5 | F32 | 6,480 |
| expert self Q | 80 | 50 x 15 | INT16 | 60,000 |
| expert self K (in place, pitch 160) | 80 | 50 x 5 | F32 | 20,000 |
| expert cross Q | 80 | 50 x 15 | INT16 | 60,000 |
| **total** | **272** | | | **165,920** |

Totals per chunk: 2,654,720 input words, 1,115,520 int16 words, 423,680 FP32 output words and 36,240
scale words, which is about 2.71 M cycles. **At 250 MHz that is about 10.9 ms per 50-action chunk**,
plus 272 call latencies and per-token latencies on the 5,296 pitched K tokens. Those DDR latencies are
unknown. At 100 cycles each they would add 0.1 ms and 2.1 ms. DDR traffic is about 6.6 GB/s (io read
3.9 GB/s, io write 2.3 GB/s, hp0 0.34 GB/s). Whether the io HP port sustains it is unknown, because it
has not been measured on the board.

## Accuracy

Kernel against torch FP32 `apply_rope`, over every RoPE call of frame 0 (272 calls, 165,920 rows,
`kernel_error.py`). Errors are relative to each row's max |result|:

| output | max | rms |
|---|---|---|
| MODE_F32 | 1.05e-5 | 1.18e-6 |
| MODE_INT16 (q · 2^e) | 3.57e-5 | 1.30e-5 |
| reference: torch FP32 -> absmax int16 (s = amax/32767) | 1.53e-5 | 8.74e-6 |

MODE_F32 is not bit-equal to torch: 1.7% of values are equal. The error comes from the Q1.17 table
(2^-18 per cos/sin). MODE_INT16 loses at most 1 bit to the power-of-two scale.

Full policy: each variant is emulated bit-exactly inside `predict_action_chunk(batch, noise=noise)` and
compared with `ref_chunk_raw` (`check_policy.py`, `check_policy_results.jsonl`). The table shows max
|err| in normalized action units; the gate is 0.05.

| variant | frame 0 | frame 3563 | frame 7489 |
|---|---|---|---|
| torch (laptop, unpatched) | 4.71e-6 | 3.58e-7 | 2.98e-7 |
| hook_exact (patch harness check) | 4.71e-6 | 3.58e-7 | 2.98e-7 |
| fmt_absmax (exact RoPE + int16 absmax Q, K: the format alone) | 2.65e-5 | 5.25e-6 | 1.03e-5 |
| kernel_f32 (every RoPE = MODE_F32) | 5.04e-6 | 1.07e-6 | 5.96e-7 |
| kernel_f32_absmax (MODE_F32 then absmax int16 Q, K) | 1.10e-4 | 1.41e-5 | 1.81e-5 |
| **seq_plan** (Q = MODE_INT16; K = MODE_F32 in place, then absmax per key; cross k_proj reads MODE_F32 K) | **7.30e-5** | **1.68e-5** | **1.90e-5** |
| kernel_qm0 (Q and K = MODE_INT16) | 1.29e-4 | 1.60e-5 | 1.67e-5 |
| kernel_qm4 (emulation only: 4-bit mantissa scales) | 1.11e-4 | 5.80e-6 | 1.19e-5 |

All variants are in the 1e-5..1e-4 class, below the int16 engine's own 3.4e-4 on -9604. The int16
linears, the attention products and the V / cross-K quantization are FP32 torch here; their kernels
report those errors.

## Integration notes for float_top / the sequencer

- **Mode dispatch.** float_top's bus interfaces do not change. A RoPE mode is one more DATAFLOW branch
  of `rot_stage` + `out_stage` between float_top's own movers. In sequencer mode (all pointer registers
  0, absolute addresses) the fields come from the `OP_ROPE` record (`fpga/kernels/sequencer/seq_format.h`):

  | kernel | record field |
  |---|---|
  | T | N |
  | H | B |
  | x, x_pitch | A.addr, A.pitch / 16 (A.stride must be 256 B: a token's heads contiguous) |
  | y, y_pitch | C.addr, C.pitch / 16 |
  | mode | F_QOUT ? INT16 : F32 |
  | s | F.addr (F.stride must be a16(4T)) |
  | cs | E.addr |

- **Change needed in `build_smolvla.py`.** `E` must point at a per-call table gathered by token (the
  three tables above, written once per chunk). It cannot be the position-indexed FP32 `rope_table` plus
  `D` positions. `D` is unused.
- **Q int16 layout.** This kernel writes Q token-major (`[T][H][64]` int16). The sequencer currently
  plans head-major `[h][n][64]`. For the Q K^T GEMM, the X operand of head h is
  `Slot(qh + 128 h, pitch = 128 H, stride = 128)`, a view of the same memory. Whether the engine's X
  reader handles a 1,920-byte row pitch efficiently is the engine owner's question. The scales are
  already head-major.
- **Scale format.** `exec_ref.op_rope` with F_QOUT emulates `s = amax/32767`. To make the sequencer's
  reference bit-exact, it should call `rope_emu.quantize(..., 0)` (power-of-two scale) and
  `rope_emu.to_f32` for MODE_F32.
- **K stays FP32 in place** (MODE_F32). The K blob quantizer (OP_QUANT_W) and the expert cross-attention
  k_proj read it as before.

## Files and commands

| file | purpose |
|---|---|
| `rope_core.h` | header-only HLS core (rotation, converters, `rot_stage`, `out_stage`) |
| `rope_top.cpp` | standalone csynth top + reference DDR movers |
| `tb_rope.cpp` | host testbench: bit compare against the emulation, F32 contiguous / F32 in place pitched (gap words untouched) / INT16 pitched |
| `rope_emu.py` | bit-exact torch emulation (`run_rows`, `to_f32`, `quantize`, `kernel_from_natural`, `table_rows`) |
| `gen_vectors.py` | test vectors: 4 real calls of frame 0 + 7 synthetic edge cases (zeros, -0, denormals, exponent spread, all-zero / tiny rows, T = 1 / 65 / 130 / 136, H = 1..15) |
| `check_policy.py` | full-policy check, results in `check_policy_results.jsonl` / `check_policy.log` |
| `kernel_error.py` | element error against torch (`kernel_error.log`) |
| `run_rope_hls.tcl` | csynth (`ROPE_PERIOD`, default 4.0) into `build/rope_p<period>/` |

```sh
export HF_HUB_OFFLINE=1
PY=/c/kp/lerobot-venv/Scripts/python.exe
$PY fpga/kernels/rope/gen_vectors.py                 # -> fpga/kernels/rope/build/vectors
$PY fpga/kernels/rope/check_policy.py                # 8 variants x 3 frames
$PY fpga/kernels/rope/kernel_error.py 0
# testbench (Vitis MinGW, clean PATH, compile and run)
export PATH="/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/mingw64/bin:/c/Xilinx/Vitis_HLS/2022.2/tps/win64/msys64/usr/bin:/c/Windows/System32:/c/Windows"
cd fpga/kernels/rope && g++ -std=c++14 -O2 -I/c/Xilinx/Vitis_HLS/2022.2/include tb_rope.cpp rope_top.cpp -o build/tb_rope.exe && ./build/tb_rope.exe build/vectors
# csynth (PATH without Git dirs)
ROPE_PERIOD=4.0 /c/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls.bat -f fpga/kernels/rope/run_rope_hls.tcl
```

Windows Smart App Control sometimes blocks a freshly linked testbench ("Permission denied" /
"Application Control policy"). Relinking with other flags (`-O1` / `-O2`) gives a binary that runs.

## Not verified

- No C/RTL co-simulation (csim is broken here), Vivado synthesis or implementation, or board run.
  Resources and timing are HLS estimates only.
- DDR latency and bandwidth of the io / hp0 ports on the KR260 are unknown. The per-token burst
  behaviour of the pitched (in-place K) path is from csynth's burst report only, not measured.
- float_top integration and the sequencer-side changes listed above are not done. They are owned by
  other agents.
