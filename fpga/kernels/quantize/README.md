# quantize: per-token int16 activation quantizer (reference arithmetic + QUANT_X mode)

FP32 rows -> absmax -> `s = fl32(max(absmax, 1e-30f) / 32767)` -> `q = clamp(rne(fl32(x / s)), ±32767)` int16,
plus the FP32 `s` per row. This is `build/brief/precision_emulation.py` `quant_sym(x, 32767, dim=1)` bit for bit,
with one convention taken from `fpga/hls/pack_float.py` (the int16 engine's host packer): **an all-zero row
(+0 / -0) gets `s = 1.0f`** (quant_sym would store 1e-30/32767; q = 0 either way, so the engine output is 0
either way). Rows with 0 < absmax < 1e-30 use the clamp exactly like quant_sym.

`quant_int16.h` is the reference implementation the engine's emit path and the other kernels must agree with:
header only, no state, plain C++ on `ap_int.h` (builds in Vitis HLS and in g++).

## Files

| file | what |
|---|---|
| `quant_int16.h` | the arithmetic: `abs_bits`/`amax2`/`word_amax` (absmax), `scale_bits` (s), `recip_seed`/`recip_load`/`recip_step`/`recip_result`/`recip_sig` (R), `row_scale_from_s`/`row_scale` (row parameters), `quant_elem` (one value), `quant8` (one output word), `dequant`, word lane helpers |
| `quant_mode.h` | the QUANT_X mode core `qz::quant_rows(rd0, rd1, io, QCmd)`: header only, so float_top can call it on its own bundles |
| `quant_top.cpp` | standalone Vitis HLS top (`quant_top`) of the same core, for csynth |
| `run_hls.tcl` | csynth script (env `QZ_PERIOD`, `QZ_IMPL=fdiv`, `QZ_FIFO=bram|uram`, `QZ_DEPTH`, `QZ_RBITS`, `QZ_TAG`) |
| `tb_quant.cpp`, `run_tests.sh` | host testbench (g++), see Tests |
| `qz_emul.py` | torch model of the kernel's integer datapath (`quant_kernel`), the reference (`quant_ref`, `quant_ref_engine`), numpy DDR pack/unpack + dequant helpers for the runtime; `--selftest` |
| `check_policy.py` | full-policy check on -12392's 3 replay frames (monkeypatches every nn.Linear) + `--dump` of real activations for the tb |
| `workload.py` | QUANT_X work per chunk from the sequencer's command list + this kernel's cycle model |
| `results/` | `check_policy.log/.json`, `tb_full.log`, `workload.txt`, `captured.qz` (git-ignored: 31 MB) |

## Arithmetic (quant_int16.h)

Per row: absmax is an integer max over the 31 magnitude bits (exact, no FP compare). `s` is computed exactly
from absmax with a 41-bit by 32767 division made of shifts and adds (32767 = 2^15 - 1) and one round-to-nearest-even.
`R = min(floor(2^49 / S), 2^26 - 1)` (S = 24-bit significand of s) by restoring division, one quotient bit a clock.

Per element (no FP divider): `Qc = (X R) >> 24` (X = 24-bit significand of x) is floor(X 2^25 / S) or one less;
the remainder `X 2^25 - Qc S` lies in [0, 2S) so only `(Qc S) mod 2^25` is needed (a 25x17 DSP product plus an 8x7
fabric product); one compare fixes Qc and gives the sticky bit. That is fl32(X / S) exactly; it is rounded to 24 bits
(RNE), scaled by 2^(E - Es) and rounded half-to-even to an integer by a sticky barrel shifter (5 stages, 17 bits).
Rounding fl32(x / s) and not the exact quotient matters: the tb meets ~14 M double-rounding cases in 435 M values.
3 DSP48E2 per lane (X R = 24x26 is two DSPs). `QZ_IMPL_FDIV` swaps in the FP32 divider core per lane (bit-equal,
see Resources for why it is not the default).

## Interface

### Standalone top registers (s_axi_control, from csynth.rpt)

| offset | register | meaning (all addresses / pitches / strides in 128-bit words = bytes / 16) |
|---|---|---|
| 0x00 | CTRL | ap_ctrl_hs (AP_START, AP_DONE, AP_IDLE, AP_READY, AUTO_RESTART) |
| 0x04 / 0x08 / 0x0c | GIER / IP_IER / IP_ISR | interrupt |
| 0x10 | ap_return | 0 = done, 1 = command rejected (nothing read or written) |
| 0x18 / 0x1c | hp0 base (lo / hi) | input buffer base (bundle hp0) |
| 0x24 / 0x28 | hp1 base (lo / hi) | the **same** input base (bundle hp1) |
| 0x30 / 0x34 | io base (lo / hi) | output base (bundle io): int16 rows and scale vectors |
| 0x3c | B | items (1..65535) |
| 0x44 | N | rows per item (1..2^24-1) |
| 0x4c | K | values per row: a multiple of 16, 16..262144 |
| 0x54 / 0x5c / 0x64 | in_off / in_pitch / in_bstride | input row (b, n) at word `in_off + b in_bstride + n in_pitch` (from hp0/hp1 base) |
| 0x6c / 0x74 / 0x7c | out_off / out_pitch / out_bstride | int16 row (b, n) at word `out_off + b out_bstride + n out_pitch` (from io base) |
| 0x84 / 0x8c | scl_off / scl_bstride | scale vector of item b at word `scl_off + b scl_bstride` (from io base) |

Rejected (`ap_return = 1`): B or N = 0, K not a multiple of 16 or out of range, `in_pitch < K/4` or `out_pitch < K/8`
with N > 1, `scl_bstride < ceil(N/4)` with B > 1. Bundles: m_axi hp0, hp1 (read, 128-bit, bursts <= 256, 16
outstanding), m_axi io (write, 128-bit, bursts <= 256), all with 64-bit addresses: the same names and widths as
float_top.

### DDR layouts and word formats (all buffers 16-byte aligned)

- input: per row K FP32, 4 a word, value j of a word in bits [32j+31:32j] (float_linear.h IoWord lane order). Pitch /
  item stride arbitrary (>= the row); the gaps are never read.
- output: per row K int16 (two's complement), 8 a word, value j in bits [16j+15:16j]. Only the row's K/8 words are
  written; pitch gaps are untouched.
- scales: per item a vector of N FP32, 4 a word, written as ceil(N/4) whole words; **the pad lanes of an item's last
  word are written as 1.0f**, so a scale vector must own ceil(N/4) words (the sequencer's `a16` scale slots do).
- one row = one burst of K/4 words per read port; output rows of an item that are contiguous (`out_pitch == K/8`, every
  QUANT_X call in the sequencer's list) are written in bursts of up to 64 rows, else one burst per row; each 64-row
  segment's scale words follow it as one burst.

### Using it from float_top (integration notes)

- Mode call: `qz::quant_rows(w0, w1, ymem, cmd)` with `QCmd{B, N, K, in, out, scl, in_pitch, in_bstride, out_pitch,
  out_bstride, scl_bstride}` (word units; call only after `qz::quant_cmd_ok(cmd)`, an invalid command can deadlock
  the dataflow region). It reads only through `w0`/`w1` (hp0/hp1 read channels, already read-only in float_top) and
  writes only through `ymem` (io write channel): no bundle, port or register change. If the engine's `WPort` and
  `IoWord` stay `ap_uint<128>` (today) the pointers pass as they are.
- Sequencer OP_QUANT_X (seq_format.h): slot A -> in (addr/16, pitch/16, stride/16), slot C -> out, slot F -> scl
  (addr/16, stride/16), B (0 = 1), N, K. Pointers are 0 in sequencer mode, so word index = byte address / 16. Byte
  addresses, pitches and strides must be multiples of 16, K a multiple of 16 (720 / 736 / 960 / 768 / 64 / 320 / 32 /
  12288 all are).
- Other kernels' emit paths (norm, softmax, GELU/SiLU, rope, attention): per row keep `amax2(abs_bits(x))` while the
  FP32 values are produced, then `row_scale(a)` (or `row_scale_from_s(scale_bits(a))` + the recip_* steps when the
  25-clock divider must be interleaved), then `quant_elem` / `quant8` per value. Cost 3 DSP + ~0.6-0.9k LUT per lane
  (below); one divider (~0.15k LUT) per kernel, not per lane.
- Conventions that differ elsewhere in the repo (to settle at integration, not changed here):
  `fpga/kernels/sequencer/exec_ref.py quant_rows` uses `s = absmax / 32767` without the 1e-30 clamp and stores
  `s = 0` for all-zero rows; this kernel follows precision_emulation (clamp) and pack_float (1.0). q is identical
  except for rows with absmax < 1e-30 (none in the policy's activations). The int16 engine's xmem image
  (int16_linear.h DRIVER CONTRACT) wants Xq tiled per 32-token block and s_tok inside xmem, while the sequencer
  contract (build_smolvla.py) has row-major int16 + a separate scale vector; this kernel writes the sequencer's
  row-major form. A tiled-output mode is not implemented.

## Resources (Vitis HLS 2022.2 csynth estimates, xck26-sfvc784-2LV-c; post-route not run: Vivado off-limits)

Default build (`QZ_IMPL` recip, LUTRAM row FIFOs 128 x 256 bit, 1-bit reciprocal x 2 engines), 4.0 ns:
all loops II = 1, estimated Fmax 342 MHz, no timing warning.

| block | LUT | FF | DSP | BRAM36 | what |
|---|---|---|---|---|---|
| qt (8 lanes) | 7,117 | 5,964 | 24 | 0 | 890 LUT + 746 FF + 3 DSP a lane |
| rd x 2 | 2,482 | 2,309 | 0 | 0 | absmax + pairing, one per read port |
| sc | 1,297 | 1,124 | 0 | 0 | scale_bits, scale-word packing |
| rc | 1,272 | 404 | 0 | 0 | two 1-bit restoring dividers |
| wr | 2,026 | 1,452 | 1 | 0 | segments, bursts, address arithmetic |
| FIFOs (f0/f1 LUTRAM, small SRL) | ~2,000 | | | 0 | |
| **quant_rows (the mode)** | **16,282** | **13,734** | **25** | **0** | what float_top adds (its m_axi adapters are shared) |
| quant_top (+ 3 m_axi + s_axilite) | 24,891 | 17,478 | 25 | 0 | standalone IP |

`build/brief` measured Vivado post-route at 0.68-0.74x the HLS LUT estimate on this part, so the mode is ~11-12k
LUT post-route (an estimate, not run). Variants run:

| variant | result |
|---|---|
| 3.333 ns (300 MHz) | estimated Fmax 354 MHz but HLS flags rc -0.40 ns, sc -0.24 ns against its 2.43 ns budget (uncertainty 0.9 ns): marginal, needs a Vivado run |
| `QZ_IMPL=fdiv` 4.0 ns | qt 3,413 LUT / 6,000 FF / 0 DSP, II = 1, depth 35, **plus 8 `fdiv` cores that HLS reports as 0** (their real cost is unknown here; build/brief also notes HLS shows fdiv as zero). Bit-equal in the tb; not the default because its area cannot be verified without Vivado |
| earlier version (before this rework, `build/old_v1`) | 20.4k LUT, 27 DSP, 15 BRAM36 (BRAM FIFOs), per-row loops |

Options: `QZ_FIFO_DEPTH=512 QZ_FIFO=bram` keeps rows up to 4096 values in one DDR pass (2 x 4 BRAM36 instead of
LUTRAM; not synthesised in this version); `QZ_RECIP_ENGINES` (default 2) trades ~0.4k LUT per engine for rows per clock.

## Speed

- qt: 8 values a clock (II = 1, one output word a clock) = **0.125 clocks per element** when not starved.
- rows: rd reads each row as one burst per port, rows alternate between hp0 and hp1, so a row costs about
  `(K/4 + 8 + L) / 2` clocks where L is the DDR read latency of a burst (unknown on this board; not measured);
  rc delivers one R every 13.5 clocks; rows longer than 1024 values take two read passes (absmax, then data).
- One SmolVLA chunk (the sequencer's list, `workload.py`): 230 QUANT_X calls, 19.9 M elements; modelled
  **12.1 / 19.1 / 26.4 ms per chunk at 250 MHz for L = 0 / 40 / 80 clocks** (0.152 / 0.240 / 0.333 clocks per element;
  the no-overhead floor is 9.9 ms). The 12 x 512 x 64 vision Q-head calls (rows of 64 at a 9 KB pitch) are the
  latency-bound part (4.0 -> 15.3 ms); everything else runs near 8 values a clock. Per call: ~48 clocks start/drain.
- Time per 50-action chunk at 250 MHz: 12-26 ms of the ~1.1 s budget (not measured on hardware).

## Accuracy

- vs quant_sym (torch) bit for bit: 319,051,488 real activation values (every nn.Linear input of the 3 replay frames
  of -12392, row lengths 32..12288): **0 q and 0 s mismatches** for the kernel's arithmetic (qz_emul.quant_kernel),
  and 0 mismatches against `pack_float.quantize_int16_tokens`. The C++ header and quant_top on 302 captured tensors
  (5.3 M values, 16 rows of every linear's first input): 0 mismatches vs torch.
- Full policy, max |chunk - ref_chunk_raw| in normalized action units (gate 0.05) / degrees:

| frame | fp32 (sanity) | actq: kernel-quantized activations, FP32 weights | int16: kernel x int16 weights (V4 scheme) | int16 with quant_sym |
|---|---|---|---|---|
| 0 | 5e-6 | 1.086e-3 (0.032 deg) | 4.61e-4 (0.0135 deg) | 4.61e-4, bitwise equal |
| 3563 | 0 | 1.74e-4 (0.0054 deg) | 1.33e-4 (0.0044 deg) | 1.33e-4, bitwise equal |
| 7489 | 0 | 1.55e-4 (0.0079 deg) | 1.27e-4 (0.0047 deg) | 1.27e-4, bitwise equal |

The kernel adds no error of its own beyond the quantization the precision study chose (V4_int16): its chunks are
bitwise equal to quant_sym's on all frames.

## Tests (commands)

```
sh run_tests.sh --quick                               # 21.8 M elements + 18 top shapes, ~5 s
sh run_tests.sh --exhaustive 8 results/captured.qz    # full: see results/tb_full.log
C:\kp\lerobot-venv\Scripts\python.exe qz_emul.py --selftest
HF_HUB_OFFLINE=1 C:\kp\lerobot-venv\Scripts\python.exe check_policy.py --dump results/captured.qz --json results/check_policy.json
C:\kp\lerobot-venv\Scripts\python.exe workload.py
vitis_hls -f run_hls.tcl        (PATH without Git dirs; QZ_PERIOD=3.333, QZ_IMPL=fdiv, ...)
```

tb_quant: scale_bits on 8 exponents x all 2^23 significands + 2 M small / zero; R for all 2^23 S; 435 M elements
(random, uniform, random bits, near ties, near 0.5 / 1.5, denormals, both zeros) for the integer and the FP32-divider
lanes; exhaustive slices (8 scales x 20 exponents x all 2^23 significands, 1.34 G elements); quant_top on 18 shapes
(items, pitches, item gaps, offsets, two-pass rows, zero / denormal rows, 3-segment items, magic-filled guards and
gaps compared word for word) + 7 rejected commands; the captured activations. C simulation of the dataflow uses
unbounded streams, so FIFO-depth deadlocks are argued in quant_mode.h, not simulated (csim / cosim unavailable).

## Not verified

Post-route area and timing (Vivado not run); the FP32 divider core's real cost; DDR read / write-response latency
and bursts on the board (the per-row cost depends on it); RTL co-simulation (no csim/cosim here); the
`QZ_FIFO=bram/uram`, `QZ_DEPTH=512` and `QZ_RBITS>1` builds in this version.
