# float_top integration contract (version 1.0)

Status: 2026-11-20, branch `fpga/int16-engine`. Every kernel, the int16 engine and the runtime follow this
contract for the continuous-motion build of SmolVLA -12392 (int16 arithmetic, decision A-10 in QUESTIONS.md).

- **Single source of truth.** `contract.h` holds every constant, the record, the validation rules and the
  reference arithmetic. `contract.py` parses `contract.h` and mirrors it.
- **Checks.** `tb_contract.cpp` checks contract.h against contract.py and against the HLS quantizer.
- **Command list.** `contract_plan.py` builds the -12392 chunk as real records and times it.
- **Supersedes** the drafts in `fpga/kernels/sequencer/seq_format.h` (128-byte record), the int16 engine's per-call
  xmem image (`fpga/hls/int16_linear.h` DRIVER CONTRACT), `attention/attn_pack` descriptors, and the per-kernel
  register maps. Section 9 maps every review defect to the clause that resolves it.

How sources are labelled:

- **measured**: a run made today, with the command in section 7.
- **HLS**: a Vitis HLS 2022.2 csynth estimate for xck26-sfvc784-2LV-c.
- **est.**: an estimate.
- Nothing here ran on the board.

## 0. Decisions

| # | clause | decision | main number behind it |
|---|---|---|---|
| C1 | activation layout | **Row-major everywhere.** FP32 rows `[token][channel]` and int16 rows `[token][k]` with a byte pitch, plus FP32 scale vectors. The engine changes; the kernels do not. | Row-major: 1 transposer in the engine, est. 3.5k LUT. Engine layout: a transposer or 2 reads in each of 7 consumers. The softmax F_ENGINE mode alone cost +4.5k HLS LUT, ran 42.3-47.9 ms instead of 35.3 ms, and deadlocks for B >= 2 |
| C2 | quantizer | **`quantize/quant_int16.h`, one rule for every producer.** `s = fl32(max(amax, 1e-30)/32767)`, an all-zero row stores 1.0, `q = clamp(rne(fl32(x/s)), ±32767)`. Epilogue: `y = ((fl(acc)·s_row)·s_tok) + bias + D`. | measured: contract.h ref == contract.py == quant_int16.h == pack_float on 40 edge rows (7,301 values) |
| C3 | record | **256 bytes, 64 fields.** 8 slots, each with a 64-bit byte address, a u32 pitch and a u32 item stride. 16 parameters, a version word and an XOR check. All index arithmetic is 64-bit. The hardware runs `validate_hw`; the host runs `validate` on every list. | HLS: decode + validate_hw = 7.6k LUT + 7 DSP beyond the adapters, est. 3.02 ns, 137 clocks per command |
| C4 | modes | **3 units in one float_top**: the engine (GEMM), one shared 4-lane row unit (QUANT_X/W/WT, NORM, GELU, SILU_MUL, ROPE, COPY) and a softmax unit. The m_axi adapters, the quantizer and the divider are shared. GATHER and AXPY are retired. | budget: about 98.6k post-route LUT (84 %) and 1,154 DSP (92 %), only after engine change E-1 |
| C5 | attention | **Q·K^T and P·V run through GEMM**, with one item per KV head and 3 query heads stacked as rows. K and V^T are runtime weight blobs. Masks are int32 rank vectors. P is int16 with a 1/sum scale. | 466 GEMMs, 71 ms per chunk at 250 MHz (est.) |
| C6 | chunk list | **2,701 records per chunk** (1,276 GEMM). The host starts the IP once per chunk. | est. F = 0.724 s, T = 0.784 s at 250 MHz |
| C7 | tests | One test per clause, listed in section 7. | baseline today: every kernel tb passes except tb_seq, which has 1 known bad expectation |

**Budget change against the design study (0.753 s).** At 250 MHz the contract's list gives:

- F = 0.724 s (design 0.693) and T = 0.784 s (design 0.753);
- margin with one inference in flight: 1.633 − 2T = +0.065 s (design +0.13);
- margin with two in flight: +0.125 s (design +0.19).

| mode group | design A250_one | this contract | delta | why |
|---|---|---|---|---|
| linear GEMMs | 487.3 ms | 488.4 ms | +1.1 | same engine model; + residual D read (hidden at K >= 256) |
| attention GEMMs | 53.7 | 71.2 | **+17.5** | the engine's own per-call cost (400 + 450 clocks) paid per item, 2,726 items. With engine change E-7 (items inside one DATAFLOW run, 200 clocks each): 65.4 ms |
| SILU_MUL / GELU / NORM | 36.9 / 18.9 / 23.7 | 37.1 / 19.0 / 24.4 | +1.0 | |
| SOFTMAX | 31.3 | 34.2 | +2.9 | divider floor of 13 clocks per row at L = 64; 150 clocks per item |
| QUANT_X | 13.8 | 20.0 | +6.2 | 236 commands × (400 + 80); the connector's 12,288-value rows are read twice |
| RoPE + K/V packing | 26.6 | 29.4 | +2.8 | QUANT_WT needs two passes (column absmax, then quantize) |
| other | 0.4 | 0.4 | 0 | COPY (pixel shuffle, language rows) |
| **F** | **0.693 s** | **0.724 s** | +0.031 | |

The clock decides whether the contract's list is continuous. The engine as it stands (HLS v6, NBX = 2) estimates
4.50 ns, i.e. 222 MHz.

| clock | F | T | continuous with 1 in flight (2T <= 1.633 s)? |
|---|---|---|---|
| 300 MHz | 0.604 s | 0.664 s | yes, +0.31 |
| **250 MHz** | **0.724** | **0.784** | **yes, +0.065** |
| 222 MHz (engine v6 HLS estimate) | 0.815 | 0.875 | no, −0.12 |
| 200 MHz | 0.905 | 0.965 | no, −0.30 |
| 281 MHz with NBX = 1 (meets 3.56 ns in HLS) | 0.797 | 0.857 | no, −0.08: weights DDR-bound |

**Engine change E-0 (timing to <= 3.6 ns at NBX = 2) is therefore a hard requirement.** Without it the phase-B
fusions of the design study are needed.

## 1. C1: data layouts in DDR (between commands)

Everything below is little-endian. An io word is 128 bits. FP32 lane j of a word is bits [32j+31 : 32j]; int16 lane j
is bits [16j+15 : 16j]. These are the lane orders of `fpga/hls/float_linear.h`.

### 1.1 Formats

| format | element (item b, row n, column k) at byte | constraints | producers | consumers |
|---|---|---|---|---|
| **A: FP32 rows** | `addr + b·stride + n·pitch + 4k` | addr, pitch and stride % 16 == 0; pitch >= 4·cols. GEMM output rows hold all Mp columns, so pitch >= 4·Mp | GEMM (C), NORM / GELU / SILU / ROPE / COPY with FP32 output | row unit (A, B), softmax (A), GEMM addend (D) |
| **X: int16 rows** | `addr + b·stride + n·pitch + 2k` | k < K, K % 8 == 0 (8 per word); values in [−32767, 32767] (−32768 is never written) | row unit q-out, softmax P, the ARM (patches, state) | GEMM (A) |
| **S: FP32 scale vector** | `F + b·stride + j·segf + 4n` | one per (row, segment). The vector holds pad4(rows) floats, pad lanes 1.0f. A scale is a positive normal float or 1.0 | every q-out producer | GEMM s_tok (F) |
| **W: weight blob** (the engine's port tiling, kept) | port p's array at `addr + p·pitch + b·stride`. Word `(g, k)` at `+ (g·K + k)·16`. Lane l = `Wq[32g + 8p + l][k]` | Mp % 32 == 0; pitch >= (Mp/32)·K·16; pad rows 0 | offline packer (`pack_float.pack` bytes), QUANT_W, QUANT_WT, ROPE with F_KW | GEMM (B) over hp0..hp3 |
| **s_row / bias** | FP32 [Mp] at E / G (+ b·stride) | pad rows: s_row 1.0, bias 0.0 | offline packer, QUANT_W/WT (s_row) | GEMM |
| **rank vectors** | int32 [rows] at D, int32 [keys] at E, 4 per word | key k visible to row n iff `0 <= E[k] <= D[n]`; a negative rank means pad or never visible | host, once at load | SOFTMAX with F_MASK |
| **RoPE table** | row t at `E + t·pitch` (pitch >= 256 B): words 0-7 = NC[0..31] int32 = round(−cos·2^17); words 8-15 = NS[0..31] = round(−sin·2^17), clamped to [−2^17, 2^17−1] | built from torch's FP32 cos/sin (`rope/rope_emu.table_rows`), one table per site: prefix 45 rows, expert self 50, cross 50 | host, once | ROPE |

**Segments.** QUANT_X, QUANT_W, QUANT_WT and ROPE work per segment:

- p1 = SEG is 0 (the whole row) or 64 (one head). Nothing else is allowed, so the hardware needs no divider.
- Segment j of row n goes to `C + n·C.pitch + j·p2`. A head-major layout uses p2 = one head block; an in-row layout
  uses p2 = SEG × element bytes.
- Its scale goes to `F + j·p3 + 4n`.
- p5 = NP rows are written. Rows [N, NP) are zero with scale 1.0, so stacked heads and padded blob rows never hold
  junk.

**Weight tiling written at run time** (written once per segment):

- **KW (keys, by QUANT_W or ROPE with F_KW).** Blob row n = key n, reduction index = d < 64.
  - The word for (n, d) is at `C + j·p2 + ((n%32)/8)·C.pitch + ((n/32)·64 + d)·16`, in lane n % 8.
  - NP % 32 == 0. C may point at group g0 = r0/32 of an existing blob, which is how the expert writes its suffix
    keys after the prefix keys.
- **VW (V^T, by QUANT_WT).** Blob row c = channel c < 64, reduction index = key n < NP, with NP % 8 == 0.
  - The word for (c, n) is at `C + j·p2 + ((c%32)/8)·C.pitch + ((c/32)·NP + n)·16`, in lane c % 8.
  - Keys [N, NP) are zero.
  - The scale is per channel, over the item's N keys.

### 1.2 Why row-major (and not the engine's current layout)

The engine's current contract (`int16_linear.h`) has three layouts:

- X interleaved per 32-token block inside one xmem image;
- s_row and s_tok in that image;
- Y group-major `Y[g][Np][32]`.

The kernels and the sequencer use rows. Row-major costs less overall:

| cost | row-major (chosen) | engine layout everywhere |
|---|---|---|
| engine | X staging transposer, est. 3.5k LUT, 0 BRAM; Y store buffers 4 groups for 512-B row bursts: 16 BRAM36 (E-2, E-3) | none |
| softmax | rows layout as built: 35.3 ms, 17.1k HLS LUT | F_ENGINE: +4.5k HLS LUT, 42.3 ms (47.9 with the stalls the review found), and it **deadlocks** for every SmolVLA shape with B >= 2 (review, threaded bounded-FIFO model) |
| norm, gelu/silu, quantize, rope | as built (row-major in and out) | each needs a 32-token transposer for X output, and 32 rows × M FP32 on chip (393 KB at M = 3,072) or a second read of Y |
| attention | P rows and O rows are natural; K and V^T are written in the weight tiling either way | same |
| burst size | X reads: 256 B per token (4 KB when the pitch equals the row); Y writes: 512 B per token row | Y: 4 KB per (group, block); row consumers read 128 B per (token, group) |

The 7 row consumers each need a whole token row: a per-token absmax, norm statistics, softmax rows, RoPE pairs. In
the group-major layout that row is spread over M/32 bursts of 128 B. So the layout cost lands once, in the engine,
instead of 7 times.

### 1.3 What lives where

| region | written by | when | content (-12392, `contract_plan.py`, measured sizes) |
|---|---|---|---|
| weights | host | load | 660.9 MB: int16 blobs + s_row + bias, with folded scales (section 2.4). The 94.6 MB BF16 token-embedding table is not needed: the language rows are constants |
| consts | host | load | 1.3 MB: vision position embedding, 12 language rows (embedded × sqrt(960)), 10 time vectors (W_t·t_emb(t_s) + b), norm gamma/beta, 3 RoPE tables, 6 rank vectors |
| list | host | load | 675 KB per arena (2,701 × 256 B) |
| arena (one per inference in flight) | host + IP | per chunk | 38.8 MB (no role reuse yet): inputs (patches int16 + scales, state int16, x_t = noise), residual streams, KV blobs, scratch |

Every address in a record is absolute. Buffers therefore need not share one contiguous CMA block. The runtime may
allocate each blob and arena as its own buffer object and write its physical address into the records. Sizes:

- one inference in flight: 0.70 GB;
- two in flight: 0.74 GB.

**Unknown, ask on the board.** Where the KR260's CMA sits (low DDR 0x0000_0000-0x7FFF_FFFF or high DDR
0x8_0000_0000-0x8_7FFF_FFFF) and how large it is. The contract does not depend on either, because every address
path is 64-bit (C3). The runtime must read each buffer's physical address from XRT (BO `paddr`) and never assume
low memory. It must also check that the buffers fit the board's CMA.

**Alignment.**

- Every used address, pitch and stride is a multiple of 16 B (ME_ALIGN).
- Recommended: buffers the host touches on 64 B (the A53 cache line), blobs and lists on 4 KB.

**Cache.**

- The host flushes weights, consts and lists once.
- Per chunk it flushes the inputs and invalidates x_t (`xrtmem` range sync).
- The IP never relies on coherency.

## 2. C2: arithmetic

### 2.1 The int16 quantizer, one rule for every producer

Reference: `contract.h` `ctr::ref::quant_row`, `contract.py` `quant_rows`, and the HLS datapath
`quantize/quant_int16.h` (`row_scale` + `quant_elem`, 1 recip bit per clock: `QZ_RECIP_BITS = 1`, everywhere).

- `a = max |x_j|` over the row or segment, as an exact integer max over the magnitude bits (denormals count; ±0 is
  0).
- `s = fl32(max(a, 1e-30f) / 32767)`. **An all-zero row stores s = 1.0**, as do pad rows and the pad lanes of a
  scale vector.
- `q_j = clamp(rne(fl32(x_j / s)), −32767, 32767)`, rounding half to even.
- The stored scale is `s · 2^p4` (p4 in [−8, 8], an exact exponent shift). Q is stored with p4 = −3: the 1/8
  attention scale. A zero row stores 1.0 whatever p4 is.
- Bit-identical to `pack_float._quantize_int16` (the engine's packer) and to `precision_emulation.quant_sym` except
  for all-zero rows (measured, section 7).

Producers that change to this rule (their own rules are retired):

| producer | its rule today | measured difference | change |
|---|---|---|---|
| norm_residual | s = recip(trunc17(recip(a)·32767)); q = rne(z·inv); a < 2^−100 → s = 0 | q ±1 on 1.4-2.9 % | its FP32 z feeds the row unit's shared quantizer |
| gelu_silu | s = 1/r, r = floor(32767·2^25/a); round half up; a < 2^−111 → s = 0 | q ±1 on 1.0-2.35 % | its FP32 y feeds the shared quantizer |
| rope | power-of-two s = 2^(Mtop−184); round half away from zero; zero row s = 0 | 0 % of q or s equal | MODE_F32 rotation (rope_emu.to_f32 bits) feeds the shared quantizer. Measured as `kernel_f32_absmax`: 1.10e-4 / 1.41e-5 / 1.81e-5 |
| sequencer exec_ref | s = amax/32767, no clamp, zero row s = 0 | only rows below 1e-30 | uses `contract.quant_rows` |
| softmax | its own P rule (below) | equal to the quant rule applied to its P on 874 real rows; zero rows 0.0 | **zero rows store 1.0** |

**Softmax output (its own exact rule, part of the contract).**

- `e_k` = 2^(−(m − x_k)·c) in Q0.23 (`ehi`): scores turned into fixed point against the row max m, then a
  1,024-entry table with linear interpolation, c = fl32(p0 · log2 e).
- `q_k = (e_k + 128) >> 8`, so the row maximum gives about 32767.
- `scale = rn(1/Σq)`, correctly rounded.
- Masked keys and pad lanes give q = 0.
- A fully masked row gives q = 0 with **scale 1.0**. The kernel's zero mode writes 0.0 today; F_EMPTY_UNIFORM is
  retired.
- The bits are those of `softmax/smx_emu.kernel(..., empty="zero")`, with the zero-row scale 1.0. Applying the C2 rule
  to its own FP32 P gives back the same q and scale (review: 0 differences on 874 real rows).

### 2.2 Order of the per-element FP32 steps in the row unit

The output is the C2 quantizer applied to the front end's FP32 values. Each front end's FP32 output is bit-exact
with its emulator:

| front end | emulator |
|---|---|
| NORM | `nr_emul` z: exact integer statistics, rsqrt by seed + 2 Newton steps |
| GELU / SILU | `gs_emu` y: NSB 9, 4 lanes, natural column order |
| ROPE | `rope_emu.to_f32` |
| COPY / QUANT | identity |

### 2.3 GEMM epilogue (engine)

`acc` is the exact integer sum (|acc| < 2^42). In this order, each step one FP32 operation of the Xilinx cores (RNE,
denormal inputs and results flushed to zero with the sign kept):

```
y = fmul(fmul(i43_to_f32(acc), s_row[m]), s_tok[n])     // as today
y = fadd(y, bias[m])                                     // F_BIAS (slot G)
y = fadd(y, D[n][m])                                     // F_ADD_D (slot D; pitch 0 = one row broadcast)
  | fadd(y, C[n][m])                                     // F_ACC (D := C, C's addressing; never with F_ADD_D)
```

- Reference: `ctr::ref::gemm_out`, `contract.gemm_out`. Checked equal on 128 values, including a denormal product.
- **There is no output multiplier**: p0 must be 0 (ME_PARAM). The packer folds every constant (2.4).
- The residual add of every block is the epilogue's: `D = C = R`, in place.
- The norm kernel's F_RESID / F_SUM / records layout are retired. They cost an extra 8 B of reads and 4 B of writes
  per element in NORM: about 48 ms instead of 24 (est.).

### 2.4 Constants folded by the packer (the engine has no p0)

| constant | where it goes | exact? |
|---|---|---|
| attention 1/8 | Q's stored scale, via the row unit's p4 = −3 | yes, a power of two (s >= 3.05e-35, so 2^−3 s is still normal) |
| connector × sqrt(960) | the connector blob's s_row: `s_row' = fl32(s_row · 30.98387)` | one rounding moves (est. <= 1 ulp per output) |
| language × sqrt(960) | the constant language rows (computed in FP32 at load) | as torch |
| Euler x += −0.1 v | action_out_proj: W' = −Wq (exact), s_row' = fl32(0.1·s_row), b' = fl32(−0.1·b), D = C = x_t (F_ACC) | one rounding moves; no AXPY mode |
| time MLP, t part | action_time_mlp_in = W_a·a_emb + (W_t·t_emb + b): the t part is a constant FP32 row per step, used as D (pitch 0) | FP32 (finer than int16) |

These change rounding at the 1-ulp level and **have not been run through the policy** (gate T-P1, section 7).

`pack_float.py` is a shared-contract file (CONTRIBUTING.md § Ownership). The folding and the retirement of the xmem
image in the deployed path need sign-off from both sides.

## 3. C3: command record, registers, protocol

### 3.1 Registers (the shipped s_axi_control map, unchanged; 7-bit address)

| offset | name | list mode |
|---|---|---|
| 0x00 | AP_CTRL | bit 0 start, bit 1 done (poll it; the interrupt pin's wiring in the BD is unknown) |
| 0x10 | AP_RETURN | `(next index << 8) \| status` |
| 0x18 / 0x24 / 0x30 / 0x3C / 0x48 / 0x54 | w0..w3, xmem, ymem (64-bit) | **all 0**: the AXI address = the byte address in the record |
| 0x60 | M | `0xA5 << 24 \| step << 23 \| first index` (index <= 0x7FFFFF) |
| 0x68 | K | list base >> 4 (u32: the list sits below 64 GB) |
| 0x70 | N | status block >> 4 (0 = none) |

- **No register is added.** If one is ever needed, only 0x78 keeps the address width at 7.
- Bus interfaces stay exactly the shipped IP's:
  - m_axi hp0..hp3, READ_ONLY, 128-bit, burst 256, 16 outstanding;
  - m_axi io: xmem read, ymem write;
  - one s_axi_control;
  - an int return (ap_ctrl_hs).
- No mode adds a pointer argument. Every mode uses w0..w3, xmem and ymem with absolute addresses.

**There is no per-call protocol.** M without 0xA5 returns `ST_NO_LEGACY` (0x03) and touches nothing.

- A single call is a 2-record list, (GEMM, END), or the same list in step mode.
- An old per-call bitstream given a list start rejects it: M is negative, so the range check returns 1 without
  touching memory (verified by the sequencer review).
- A 128-byte draft list is BAD_MAGIC (0xC5 ≠ 0xC6).

### 3.2 Record: 256 bytes = 16 io words, command i at list + 256·i; field j = lane j%4 of word j/4

| fields | content |
|---|---|
| f0 | [7:0] opcode, [15:8] REC_MAGIC 0xC6, [31:16] flags |
| f1 | tag (the builder uses the record index) |
| f2, f3, f4 | M (GEMM: Mp, a multiple of 32), K (input columns / reduction / keys), N (rows per item) |
| f5 | B items (0 means 1) |
| f6 | CONTRACT_VERSION = 0x00010000 |
| f7 | 0 |
| f8..f39 | slots A..H, 4 fields each: addr[31:0], addr[63:32], pitch (bytes, u32), item stride (bytes, u32) |
| f40..f55 | p0..p15 |
| f56..f62 | 0 |
| f63 | 0x5EC0C6ED ^ f0 ^ ... ^ f62 |

**Addresses are 64-bit everywhere.**

- Hardware and C model compute `addr + b·stride + n·pitch (+ j·seg)` in uint64. This fixes the int overflow above
  2 GB physical that the review found in `int16_linear.h`'s `int` offsets.
- validate_hw rejects any slot address >= 2^40.
- The host's validate also rejects any extent ending past 2^40.
- After that check, HLS may narrow the datapath to ap_uint<40>.
- All 2,701 records were validated at bases 0x4000_0000 and 0x8_0000_0000 (Python and C++) and 0xFF_C000_0000
  (Python); a base of 0xFF_F000_0000 is rejected (measured).

### 3.3 Opcodes, slots, parameters

| op | code | unit | A | B | C | D | E | F | G | params | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NOP / END | 0x00 / 0x01 | sequencer | | | | | | | | | YIELD |
| GEMM | 0x10 | engine | X int16 | W blob (pitch = port pitch) | Y FP32 rows | addend rows | s_row | s_tok | bias | p2 = R rows per C row group (0 = off), p3 = P1 group pitch: row t at `C + b·stride + (t/R)·P1 + (t%R)·pitch` | BIAS, ADD_D, ACC, YIELD |
| QUANT_X | 0x20 | row | FP32 rows | | int16 rows | | | scales | | p1 SEG, p2 SEGC, p3 SEGF, p4 SHIFT, p5 NP | QOUT (required) |
| QUANT_W | 0x21 | row | FP32 rows (keys) | | KW tiling | | | s_row | | as QUANT_X; NP % 32 == 0 | |
| QUANT_WT | 0x22 | row | FP32 rows (keys) | | VW tiling | | | s_row per channel | | p1 = 64, p5 = NP = blob K | |
| NORM | 0x30 | row | FP32 rows | | int16 + F, or FP32 | | gamma [K] (beta at +4K) | scales | | p0 = eps (FP32 bits, in (0, 1)) | QOUT, LN |
| SOFTMAX | 0x40 | softmax | scores FP32 | | P int16 | rank_q int32 | rank_k int32 | 1/sum | | p0 = score multiplier (1.0), p5 NP | QOUT (required), MASK |
| GELU / SILU_MUL | 0x50 / 0x51 | row | x (gate) | up (unless NOMUL) | int16 + F, or FP32 | | | scales | | p4, p5 | QOUT, NOMUL (SILU only) |
| ROPE | 0x60 | row | FP32 rows (q or k part) | | int16 (QOUT), KW (KW flag) or FP32 rows (in place: C = A) | | table rows (pitch >= 256) | scales | | p1 = 64, p2, p3, p4, p5 | QOUT, KW |
| COPY | 0x70 | row | FP32 rows | | FP32 rows | | | | | none | |

Constraints of each opcode:

- **Limits** (`contract.h`):
  - N, B: 1..65535.
  - GEMM: Mp 32..8192, multiple of 32; K 8..4096, K % 8 == 0.
  - Row ops: K % 8 == 0. QUANT_X and COPY: K <= 16,384 (rows above 3,072 are read twice). GELU and SILU:
    K <= 3,072. NORM: K <= 1,024. Segments <= 64.
  - SOFTMAX: L 8..256, L % 8 == 0.
- **Unused slots, parameters and reserved fields must be 0** (ME_SLOT / ME_PARAM). Retired flags must be 0
  (F_RETIRED = RESID, SUM, BF16, ENGINE, EMPTY_UNIFORM, 0x4000, 0x8000).
- **Ordering (R-ORD).** Commands run strictly in list order. A mode returns only after it has received every
  write response (BRESP) of its writes; the next command starts after that. Aliasing within one command is allowed
  only for:
  - GEMM: C == D exactly (in-place residual), and F_ACC;
  - ROPE: FP32 C == A (in place).

  Any other overlap is a builder bug.
- **Ports (phase A).**
  - Activations are read **only on io** (xmem).
  - hp0..hp3 read only:
    - the offline weight blobs;
    - the host-written constants (gamma/beta, RoPE tables);
    - the runtime K / V^T blobs.
  - The runtime blobs are the only cross-master read-after-write: written on io (HPC), read on hp by a later GEMM.
    They rely on R-ORD. Board test BT-1 proves it.
  - This retires the hp0/hp1 activation reads of quantize, norm_residual, gelu_silu and softmax.

### 3.4 Validation

| check | where | what |
|---|---|---|
| `decode` | hardware and host | magic, check word, version, known opcode → ST_BAD_MAGIC / BAD_CHECK / BAD_VERSION / BAD_OPCODE |
| `validate_hw` | hardware, before every dispatch; on failure: ST_MODE_ERROR, word 1 = ME code, nothing touched | flags per opcode; dims and limits above; parameter ranges; unused slots, parameters and reserved fields zero; 16-B alignment; minimum pitches; slot addresses < 2^40. No extent products |
| `validate` | host, on every list it builds (`contract.validate_list`) | validate_hw plus every extent end <= 2^40 |

A list that passed validate never stops the IP with a mode error. The hardware check still catches every
truncation fault the reviews found (rope 9-bit scale index, norm N >= 65,536 and rec 512, H = 0, K > limits).
Modes must size their counters from these limits.

### 3.5 Status block (3 io words = 48 B at N << 4)

| word | lane 0 | lane 1 | lane 2 | lane 3 |
|---|---|---|---|---|
| 0 | status code | next index | tag of the last command (or of the bad record) | commands done in this start |
| 1 | mode error ME_* | 0x53455153 | starts since configuration | last opcode |
| 2 | CONTRACT_VERSION | index of the last finished command (progress) | cycles of this start [31:0] | [63:32] |

Status codes:

| code | name |
|---|---|
| 0x00 | END |
| 0x01 | YIELD (F_YIELD; resume at index + 1) |
| 0x02 | STEP |
| 0x03 | NO_LEGACY |
| 0x10 | BAD_MAGIC |
| 0x11 | BAD_CHECK |
| 0x12 | BAD_OPCODE |
| 0x13 | MODE_ERROR (next = the failing command) |
| 0x14 | LIMIT |
| 0x15 | BAD_VERSION |

The cycle counter, with YIELD at the phase boundaries, is how the board measures F per phase (BT-3).

## 4. C4: modes, units and the area budget

### 4.1 Units

| unit | opcodes | kernel sources it is built from (as front ends / references) |
|---|---|---|
| engine | GEMM | `fpga/hls/int16_linear.h` plus section 8 |
| row unit (one DATAFLOW function, 4 lanes; reader → front end → row buffer + per-segment absmax → shared quantizer → writer) | QUANT_X, QUANT_W, QUANT_WT, NORM, GELU, SILU_MUL, ROPE, COPY | `quantize` (reader, quantizer, writer), `norm_residual` (statistics + affine only), `gelu_silu` (table unit only, 4 lanes, natural order), `rope` (`rot_stage` only), `attention/attn_pack` (KW/VW writer logic only) |
| softmax unit (4 lanes, rows layout only, one io read through a BRAM row buffer) | SOFTMAX | `softmax` (softmax_core minus F_ENGINE, F_EMPTY_UNIFORM, FP32 output, own decode, and the second hp read) |
| sequencer | NOP, END, dispatch | `sequencer/seq_core.h` re-targeted to `contract.h` |

**Sharing.**

- **m_axi adapters**: one per bundle at the top (5 + control).
- **Quantizer**: one per unit that writes int16:
  - the row unit's (4 lanes × 3 DSP + one 1-bit-per-clock divider);
  - the softmax computes q directly and has its own correctly rounded 1/sum divider.
- **rsqrt** (NORM) and the GELU/SiLU ROM sit in the row unit only.
- **exp tables** in the softmax only. There is nothing to share between the exp tables and the SiLU sigmoid: the
  algorithms differ.

**Mode functions.**

- Every mode function is `INLINE off` and has exactly one call site in the dispatch switch (sequencer review: one
  hardware instance).
- No ALLOCATION pragma: it broke timing, at 5.66 ns.

### 4.2 Budget (engine first)

Factors from HLS to post-route LUT:

- FP32 logic: 0.70 (measured on the shipped engine);
- integer and fabric logic: 1.0;
- kernels: 0.85 (est. mix).

The limits: post-route LUT <= 99.5k (85 %), DSP <= 1,185 (95 %).

| block | HLS LUT | DSP | BRAM36 | URAM | post-route LUT est. | basis |
|---|---|---|---|---|---|---|
| engine as of v6 (HLS, NBX = 2) | 81,051 | 1,137 | 8 | 48 | 70.3k | measured: compute16 38.5k (×1.0), dequant16 21.1k (×0.7), loads/stores 5.4k, adapters 14.3k (×0.7) |
| **E-1**: dequantizer 16 → 4 lanes | −16k | −72 | 0 | 0 | −11.3k | est. from the v6 dq pipeline: 18.1k LUT / 96 DSP |
| E-2..E-6: row-major X stage, row-major Y store with a 4-group buffer, D prefetch, epilogue (2 fadd × 4 lanes, full DSP), bias on chip, 64-bit address generators, items | +9k | +16 | +40 | 0 | +9.0k | est. |
| sequencer: 256-B fetch, decode, validate_hw, walk, status, cycle counter | +8.5k | +7 | 0 | 0 | +8.5k | measured probe: 10,449 LUT incl. io 2,318 + control 488, i.e. 7.6k LUT; + 1k walk/status est. |
| **row unit** | **<= 15k** | **<= 50** | <= 10 | 0 | 12.8k | target (the standalone parts sum to more than 40k) |
| **softmax unit** | **<= 8k** | **<= 16** | <= 4 | 0 | 6.8k | target (rows-only build 17.1k) |
| port muxing of 3 units on 5 adapters | +2.5k | 0 | 0 | 0 | 2.5k | est. |
| **total** | **about 108k** | **1,154 (92.5 %)** | **62 (43 %)** | **48 (75 %)** | **98.6k (84.2 %)** | 0.9k LUT under the line |

FF: about 150k (64 %), est.

**Reading.**

- It fits only with E-1 and with the kernel targets.
- Without E-1 it is about 110k LUT (94 %): it does not fit.
- The two kernel targets are hard caps, 53-63 % below the kernels' standalone builds. How they get there:
  1. one reader, row buffer, quantizer and writer for all 8 row ops, where today each kernel has its own;
  2. 4 lanes (io carries 4 FP32 per clock; 8 lanes buy nothing in phase A);
  3. no own decode, no own m_axi adapters, no engine layout, no residual path, no records layout;
  4. FP32 cores mapped to DSP (there are 31 DSP to spare, LUT is the tight resource);
  5. row FIFOs in BRAM, not LUTRAM.

**Fallbacks if the merged csynth overruns, in order:**

1. Fold the softmax into the row unit (shared reader, writer, row buffer): est. −2.5k.
2. GELU in its sigmoid form `x·σ(1.5958x + 0.0713x³)`, sharing the SiLU ROM: est. −1k. This needs a policy check.
3. Drop validate_hw down to decode plus dims, flags and alignment: −2k. The host validation stays.

**Gate.** One csynth of the merged float_top must show:

- LUT <= 108k HLS, with the row unit and the softmax inside their caps (the factors above then give <= 99.5k);
- DSP <= 1,185;
- every loop II = 1;
- estimated path <= 3.6 ns at a 4.0 ns target.

That run decides before Vivado.

## 5. C5: attention through the GEMM

Per attention site, per KV head j (one GEMM item; G = 3 query heads per KV head; head dim 64):

1. **Q.** The row unit (ROPE with QOUT, or QUANT_X for vision) writes Q head-major: `Q[h][TqP][64]` int16,
   TqP = pad4(Tq), with scales `[h][TqP]`, p4 = −3. Rows [Tq, TqP) are zero with scale 1.0. The G heads of KV head
   j are then contiguous: X = Q + 3j·TqP·128, N = 3·TqP, pitch 128. Their scales are contiguous too:
   F + 3j·TqP·4.
2. **K.** ROPE with F_KW (self-attention) or QUANT_W (vision; LM odd layers; cross K') writes the KW blob of item j:
   - rows = keys, Mk = pad32(keys), per-key scale;
   - the expert's self-attention blob holds the prefix keys in rows [0, 64), written by the LM, and the suffix keys
     in [64, 128), written every step.
3. **Q·K^T.** GEMM with W = KW blob of item j, M = Mk, K = 64, N = 3·TqP. Out: FP32 scores `[j][3·TqP][Mk]`,
   already × 1/8.
4. **Softmax.** SOFTMAX over L = Mk, with rank masks:

   | site | rank of query row t = g·TqP + i | rank of key k |
   |---|---|---|
   | vision | none (no mask) | none |
   | LM prefix | image/lang 0, state 1, pad row −1 | same; pad keys −1 |
   | expert self | 2 + i (i < 50), pad −1 | prefix 0 / state 1 / pad −1; suffix key j: 2 + j; pad −1 |
   | expert cross | 0, pad −1 | valid prefix 0, pad −1 |

   This reproduces `make_att_2d_masks` (the attention kernel checked 6,336 masks with 0 failures). Out: P int16 rows
   + 1/sum scales.
5. **V.** QUANT_WT writes the V^T blob of item j: 64 channels × NP keys, with a per-(head, channel) scale over the
   keys. The expert's self-attention keeps two V^T blobs, each with its own channel scales: the prefix one (written
   once per chunk by the LM) and the suffix one (written every step).
6. **P·V.** GEMM with X = P (column offset for the suffix part), W = V^T blob, K = NP. Out: O FP32 with the C row
   groups R = TqP, P1 = 256 B, so row t = g·TqP + i lands at `O + i·pitch + (3j + g)·256`. That is O in
   `[Tq][H·64]` order, directly. The expert's suffix part runs a second GEMM with F_ACC.
7. **O.** QUANT_X over [Tq][960] (one scale per token over all heads) feeds o_proj.

| site | items per command | Q·K^T M × K × N | P·V M × K × N | measured precision (attention kernel, -12392 frames 0 / 3563 / 7489) |
|---|---|---|---|---|
| vision (2 images) | 24 (Q·K^T, softmax), 12 + 12 (P·V per image) | 256 × 64 × 256 | 64 × 256 × 256 | whole policy, int16 attention at all sites with the softmax kernel's P: 5.9e-5 / 2.7e-5 / 4.6e-5 |
| LM prefix (45 tokens) | 5 | 64 × 64 × 144 | 64 × 64 × 144 | whole policy, all sites, prefix compacted to 45: 1.7e-4 / 3.2e-5 / 2.9e-5 |
| expert self | 5 | 128 × 64 × 156 | 64 × 64 × 156, twice (prefix, suffix + ACC) | |
| expert cross | 5 | 64 × 64 × 156 | 64 × 64 × 156 | |

**K/V producers per layer.**

- LM even layer i: writes the prefix rows of expert layer i's self-attention K and V^T blobs. The LM's own
  attention reads them too, as M = 64 (rows 0..63).
- LM odd layer i: keeps its FP32 K (rotated in place) and V for the cross hoist. It packs temporary blobs for its own
  attention.
- Cross hoist (once per chunk): for each odd i, QUANT_X K and V, then k_proj and v_proj, then QUANT_W and QUANT_WT.

LM layer 15's q, attention, o and MLP and the LM final norm are skipped (dead work, exact). Its K and V stay.

## 6. C6: the -12392 chunk as one list

### 6.1 Shape (`contract_plan.py`, measured: 2,701 records, all valid)

| phase | records | per unit |
|---|---|---|
| vision | 182 | patch GEMM (2 items, + bias, D = position embedding with stride 0); 12 × 15; post-LN (FP32 out) |
| connector + prefix | 10 | 4 COPY (pixel shuffle, 8 items × 4 rows × 3,072 values each), QUANT_X (32 × 12,288), 3 GEMM K-chunks (ACC), COPY of the 12 language rows, GEMM state_proj |
| LM prefix | 220 | even layers 14 commands, odd 15 (extra QUANT_W), layer 15: 3 (NORM, GEMM k\|v, ROPE k FP32) |
| cross K'/V' | 48 | 6 per odd layer |
| 10 Euler steps | 2,240 | 6 (time MLP: QUANT_X x_t, a_in, QUANT_X, t_in + D, SiLU, t_out) + 8 × 15 (self) + 8 × 12 (cross) + final NORM + action_out with Euler (ACC) = 224 per step |
| END | 1 | |

By opcode:

| opcode | records |
|---|---|
| GEMM | 1,276 |
| NORM | 386 |
| ROPE | 271 |
| QUANT_X | 236 |
| SOFTMAX | 187 |
| SILU_MUL | 185 |
| QUANT_WT | 115 |
| QUANT_W | 27 |
| GELU | 12 |
| COPY | 5 |
| END | 1 |

The list is 675 KB, static, and built once per arena. Per chunk the host:

1. writes the patches (int16 + scales, by the contract quantizer), the state (int16) and the noise (x_t);
2. flushes them;
3. writes 4 registers (`contract.start_regs`) and starts the IP;
4. polls done;
5. invalidates and reads x_t (50 × 32 FP32).

### 6.2 Time per mode at 250 MHz (est., `contract_plan.py --mhz 250`)

Rules:

- **GEMM**: the engine's own cycle model (`model_int16.call_clocks`: per tile `max(G·nbt·max(K, 256), G·K, DDR,
  next X)`, plus 400 + first tile + 450 per item), with the D read added to the io read and DDR terms.
- **Row unit**: 480 + per item 64 + per row `max(read words × passes, write words, K/4) + 4` clocks.
- **Softmax**: 380 + per item 150 + per row `max(L/4, 13) + 2`.
- **Sequencer**: 80 clocks per record.
- DDR 14 GB/s (unmeasured).

| mode | records | ms | GB |
|---|---|---|---|
| GEMM (linear) | 810 | 488.4 | 3.13 |
| GEMM (attention) | 466 | 71.2 (65.4 with E-7) | 0.34 |
| SILU_MUL | 185 | 37.1 | 0.18 |
| SOFTMAX | 187 | 34.2 | 0.19 |
| NORM | 386 | 24.4 | 0.14 |
| QUANT_X | 236 | 20.0 | 0.11 |
| GELU | 12 | 19.0 | 0.11 |
| QUANT_WT | 115 | 13.3 | 0.06 |
| ROPE | 271 | 11.0 | 0.06 |
| QUANT_W | 27 | 5.1 | 0.03 |
| COPY | 5 | 0.4 | 0.00 |
| **F** | **2,701** | **724** | **4.37 (6.0 GB/s average)** |

By phase:

| phase | ms |
|---|---|
| vision | 278.2 |
| connector | 3.2 |
| LM | 50.4 |
| cross K/V | 1.4 |
| steps | 391.2 (39.1 per step) |

T = F + ARM 30 ms + link 30 ms = 0.784 s. Both of those are estimates, never timed.

Sensitivity:

| change | F |
|---|---|
| DDR 10 GB/s | 0.728 s |
| E-7 | 0.718 s |
| 222 MHz | 0.815 s |
| 200 MHz | 0.905 s |

## 7. C7: test plan (which test proves which clause)

Two scripts, both run today (Vitis MinGW g++ on a clean PATH):

- `sh fpga/kernels/run_contract_tests.sh`: the contract's own checks (the first three rows below).
- `sh fpga/kernels/run_kernel_baseline.sh`: builds every existing kernel and engine testbench and runs it on the
  vectors already in the repo.

| test | clause | status today (measured) | must become |
|---|---|---|---|
| `tb_contract` (`contract.py --vectors` + g++ -ffp-contract=off) | C2, C3 | **ALL PASS**: 45 records with every ME_* and every BAD_*, equal in C++ and Python, for validate and validate_hw; quantizer 40 rows / 7,301 values (ref == contract.py == quant_int16.h); epilogue 128 values | stays green on every contract.h edit |
| `contract.py --selftest` | C1, C2, C3 | **PASS**: round trip; quantizer == `pack_float.quantize_int16_tokens`; `pack_w` == `pack_float.pack`; `w_word_addr` | same |
| `contract_plan.py` + `tb_contract <list>` | C3, C6 | **PASS**: 2,701 records valid in Python and C++ at 0x4000_0000 and 0x8_0000_0000, in Python at 0xFF_C000_0000; a base of 0xFF_F000_0000 is rejected (ME_ADDR) | + golden-executor run (T-P1) |
| HLS probe `hls_probe/run_contract_hls.tcl` | C3 hardware cost | **csynth**: 10,449 LUT / 5,395 FF / 7 DSP, est. 3.02 ns (331 MHz; −0.10 ns against the 2.92 ns budget), 137 clocks per command | inside the merged-top gate |
| engine `tb_float` int16 (16 cases), `--units`, `tb_int16_top` (13 cases), all against the per-call layout | C2 epilogue part | **pass** (16 / units / 13) | new cases in the contract layouts: row-major X/Y with pitch, N % 32 ≠ 0, K 8..4096, items, C row groups, bias / D / ACC, both against `contract.gemm_out`; chaos and NBX = 1 builds |
| `quantize/tb_quant --quick` | C2 | **TB PASSED** (21.8M elements, 18 shapes, 7 rejects) | row-unit tb with every row op against contract.quant_rows of its front end's FP32 output |
| `norm_residual/tb_nr` (19 cases) | C2 front end | **ALL PASS** (bit-exact vs nr_emul) | + the all-equal LayerNorm row case (K = 768/960) |
| `gelu_silu/tb_gelu_silu`, lanes 4 and 8 | C2 front end | **PASS** (vec9 + real9: 0 q, 0 s mismatches) | 4 lanes only, y then the contract quantizer |
| `rope/tb_rope` | C2 front end | **PASS** (0 of 1,084,244) | + the 2^−126 boundary cases (12 known mismatches) fixed |
| `softmax/tb_softmax` (default 46/46, rows 19/19) | C2 softmax rule | **pass** | rows only; zero rows 1.0; **threaded bounded-FIFO dataflow model** (deadlock) |
| `attention/tb_attn`, model and real engine core | C5 | **ALL PASS** (51 checks) | re-targeted to contract layouts, with the softmax's P (smx_emu) between the two GEMMs (never run so far) |
| `sequencer/tb_seq` | C3 walk | **FAILED (1)**: tb_seq.cpp:248 expects BAD_CHECK where F_YIELD stops first (0x301); a test bug (review) | re-targeted to 256-B records; walk paths END/YIELD/STEP/BAD_*/BAD_VERSION/MODE_ERROR/NO_LEGACY; status words 0-2 |
| **T-P1 golden executor** (not built) | C1-C6 numerics | **not run** | a torch executor of contract records with every mode bit-exact (smx_emu, gs_emu, nr_emul, rope_emu + contract.quant_rows, contract.gemm_out), full policy on the 3 -12392 frames: <= 0.006 on frame 0 and <= 2e-4 on the benign frames (the I7e class), and FP32-mode wiring check <= 1e-5 |
| **T-S1 shim** (not built) | C6 | — | `float_shim` running contract.h + the HLS mode code on the same list == the golden executor, bit for bit |
| deadlock model, every DATAFLOW mode | C4 | — | threads + FIFOs bounded at the pragma depths (the softmax review's dfsim), on the policy's shapes |
| merged-top csynth gate | C4 | — | section 4.2 gate |
| BT-1 cross-master RAW | C3 R-ORD | board | QUANT_W/WT via io, then GEMM via hp, stressed in a loop; compare to the shim |
| BT-2 burst efficiency | C1 | board | 256/512-B row bursts against 4 KB (a K = 64 GEMM with large M as the probe) |
| BT-3 per-phase time | C6 | board | YIELD at phase boundaries; status word 2 cycles against 6.2 |
| BT-4 DDR, BT-5 clock | C6 | board | `set_pl_clock(250)`; WNS >= 0 in the Vivado report |

## 8. Engine changes (for the integrator, who owns fpga/hls)

The interfaces stay as listed (E-11). Each item is precise; costs are estimates.

| id | change | why | cost |
|---|---|---|---|
| **E-0** | Close timing: `compute16` main estimates 4.50 ns at NBX = 2 (v6: −1.58 ns against 2.92). It must be <= 3.6 ns at NBX = 2. NBX = 1 (3.56 ns) is not a way out: +100 ms per chunk, weights DDR-bound | at 222 MHz T = 0.875 s: not continuous | engine owner's work, in progress |
| **E-1** | `dequant16` 16 → 4 lanes: 4 × (i43_to_f32, 2 fmul), fed by a 16 → 4 width step on `es`. The store takes 4 FP32 per clock, so the shadow already drains at the store's pace (model: per block max(K, 256)). Time is unchanged | area: −72 DSP, about −16k HLS LUT. Without it the design does not fit (section 4.2) | est. |
| **E-2** | X from slot A as row-major int16 rows (pitch, item stride), s_tok from slot F, s_row from E, bias from G. No xmem image. A 32-token × 128-k LUTRAM staging (ping-pong) turns rows into the xbuf's 4-token words at >= 1 io word per clock, with 256-B bursts per token (4 KB when the pitch is 2K = 128). Rows n >= N of the last block are zero and not read | C1 | est. 3.5k LUT |
| **E-3** | Y to slot C as row-major FP32 with pitch, optional row groups (p2 = R, p3 = P1), exactly N rows × Mp columns. Buffer 4 groups (128 columns) of the tile in BRAM (64 tokens × 128 × 4 B × 2 = 16 BRAM36) so a token row segment leaves as one 512-B burst (M = 64: 256 B) | C1, C5 | 1.5k LUT + 16 BRAM36 |
| **E-4** | Epilogue after the 4-lane dequantizer: + bias[m] (on chip, 8 KB × 4), then + D[n][m] or, with F_ACC, + C[n][m] (C's addressing). FP32 adds (the Xilinx core, full DSP); order and FTZ as in 2.3. D is prefetched over the io read channel into a 16-BRAM36 buffer of the tile's 4-group block. C may alias D exactly | C2; the residual add leaves the norm (−23 ms against F_RESID) | 2 fadd × 4 lanes = 16 DSP, about 2k LUT, 24 BRAM36 |
| **E-5** | Items: B per command. Item b uses the slots' `addr + b·stride`. The weight blob's port p array is at `B.addr + p·B.pitch + b·B.stride` | C5 | 0.5k LUT |
| **E-6** | 64-bit addresses: every index, offset and pointer sum in uint64 (or ap_uint<40> after validate_hw). Replace the `int off_stok, off_x, sbase` of `load_io16` and the `int` counters of `compute16` and `store16` where they form addresses. The pointer registers are 0, so the AXI address = the byte address | the review's int overflow above 2 GB physical | 1k LUT |
| **E-7** | (performance, optional) Run a command's items inside one DATAFLOW run: the next item's X tile and weights stream while the previous one drains. Per-item fixed cost 850 → about 200 clocks | −5.8 ms per chunk | est. |
| **E-8** | No output multiplier (p0 = 0); constants are folded by the packer (2.4) | 4 fmul fewer | 0 (avoids 12 DSP) |
| **E-9** | The integrated float_top body is the sequencer. Remove the per-call xmem protocol from that build (ST_NO_LEGACY). Keep the per-call variant only as a separate bring-up build | one X loader and one Y store, not two | 0 (avoids about 4k LUT) |
| **E-10** | Limits: Mp <= 8192, K <= 4096 with K % 8 == 0, N <= 65535, B <= 65535. Size the block, tile and row counters from these | C3 | 0 |
| **E-11** | Interfaces frozen: m_axi hp0..hp3 READ_ONLY (no write through w0..w3), io = xmem read + ymem write, s_axi_control 7-bit and unchanged, `int` return. Verify the `component.xml` diff = the IP version only | the block design cannot be regenerated here | 0 |

## 9. Defects from the reviews and their resolution

| # | source | defect | resolution (clause) | owner |
|---|---|---|---|---|
| 1 | sequencer | tb_seq.cpp:248 expects 0x311 where YIELD stops first (0x301); confirmed FAILED today | re-target the tb to 256-B records with the correct expectation (C7) | sequencer |
| 2 | sequencer | exec_ref quantizer s = amax/32767, zero row 0 | `contract.quant_rows` (C2) | sequencer |
| 3 | sequencer | GEMM contract needs bias/D/C/p0 the engine lacks | E-4, E-8 (C2.3) | engine |
| 4 | sequencer | the legacy path cannot share the dispatch | no per-call protocol (E-9, 3.1) | engine, sequencer |
| 5 | sequencer | int index overflow above 2 GB; a CMA plan above low DDR | 64-bit everywhere, 40-bit check, per-buffer physical addresses (3.2, 1.3, E-6) | all |
| 6 | sequencer | "v0 3.239 ns (meets)" overclaim; II = 2 binary stream stubs | row unit: one io read stream with on-chip second operands (SILU reads gate\|up as one row); stubs retired | sequencer, row unit |
| 7 | sequencer | weak stub checks; K > 1024 in the stub; the exec_ref twin diverges | validate_hw is the one rule set; the golden executor implements MODE_ERROR/STEP/LIMIT (C3, T-P1) | sequencer |
| 8 | sequencer | QUANT_W has no port pitch | C.pitch = the port pitch (1.1 KW) | row unit |
| 9 | sequencer | the 2.5 ms per-call host time is unmeasured | not used: one start per chunk; the board measures (BT-3) | runtime |
| 10 | attention | QZ_RECIP_BITS clash between attn_pack and quant_mode in one TU | QZ_RECIP_BITS = 1 everywhere; attn_pack is not integrated (4.1) | row unit |
| 11 | attention | 28k LUT; ALLOCATION sharing impossible (the functions are INLINE) | one shared quantizer instance inside the row unit (4.1) | row unit |
| 12 | attention | no data-path bursts | writer rule: KW/VW written in runs of >= 16 words; readers burst per row; per command DDR latency paid once (1.1, 4.2) | row unit |
| 13 | attention | N[23:16] head encoding conflicts with the sequencer; version skew | items B in the record (E-5); BAD_VERSION, NO_LEGACY (3.1) | engine |
| 14 | attention | P·V with K = 81/131 is not % 8; int16 krank vs int32 ranks | K = pad of the key count (NP), zero columns; int32 ranks only (1.1, 5) | builder |
| 15 | attention | ftz flushes to +0 in attn_int16.py; the engine keeps the sign | emulators use `contract.ftz` (sign kept) | attention |
| 16 | attention | scale_mul wrap; rows_per_seg truncation | p4 limited to ±8 (exact for every scale: the smallest, 3.05e-35 · 2^−8 = 1.2e-37, is still normal); limits checked by validate_hw | row unit |
| 17 | attention | headline row used torch P | precision quotes the int16_smx row (5.9e-5 / 2.7e-5 / 4.6e-5) (5) | — |
| 18 | rope | power-of-two scale, round half away, zero row 0 | contract quantizer on the FP32 rotation (C2) | row unit |
| 19 | rope | table / record mismatch (FP32 table + positions vs a per-token Q1.17 table) | per-row Q1.17 table at E, no positions slot (1.1) | builder, row unit |
| 20 | rope | 9-bit sx alias; H = 0/64 deadlock | counters sized from the limits; validate_hw (3.4) | row unit |
| 21 | rope | 2^−126 flush boundary; NaN as finite | fix to_f32 at the boundary; non-finite inputs are undefined but must not hang or write outside the declared extents (4.1) | row unit |
| 22 | gelu_silu | own quantizer; 8 lanes with permuted order; 23.6k LUT | 4 lanes, natural order, contract quantizer (C2, 4.2) | row unit |
| 23 | gelu_silu | tail resolution 2^−21 (SiLU for x < −15 is 0) | documented limit; no policy impact (−12392: at most 0.037 step); a per-segment exponent only if T-P1 shows an effect | row unit |
| 24 | gelu_silu | rows loop not flattened (latency per row) | reader rule: one burst stream per command (4.2) | row unit |
| 25 | quantize | hp0/hp1 activation reads (cross-master) | activations on io only (3.3) | row unit |
| 26 | quantize | 16.3k LUT for 8 lanes; overlap unchecked | 4 lanes, one reader; validate_hw + builder overlap rule (R-ORD) | row unit |
| 27 | quantize | NaN/Inf give finite garbage | documented undefined; the host checks x_t for NaN | runtime |
| 28 | norm | 28.5k LUT, 43 BRAM18; the records layout is not built by the list | statistics + affine only, residual in the GEMM (E-4), BRAM row buffer shared (4.2) | row unit |
| 29 | norm | rec = 512 truncation; N >= 65536 truncation | records layout retired; N <= 65535 by validate_hw | row unit |
| 30 | norm | LayerNorm mean not correctly rounded (constant rows give ±32767 noise) | fix: exact integer correction of mean·K against S in the statistics (T: all-equal rows at K = 768/960) | row unit |
| 31 | norm | hp activation reads | io only (3.3) | row unit |
| 32 | softmax | **F_ENGINE deadlock for B >= 2** | engine layout retired (C1); rows layout with a BRAM row buffer; the bounded-FIFO model in the tb (C7) | softmax |
| 33 | softmax | stale 342 MHz claim: the current source misses by 0.29 ns in the pass-1 max | register the running-max compare (section 4.2 gate) | softmax |
| 34 | softmax | own decode (1.2k LUT, 10 DSP); flags outside seq_format.h | parameters from `ctr::Cmd`; flags only from contract.h | softmax |
| 35 | softmax | zero rows scale 0.0 | 1.0 (C2) | softmax |
| 36 | softmax | int overflow on a malformed N/B | validate_hw limits | softmax |

## 10. Runtime and builder obligations

- **List builder** (replaces `sequencer/build_smolvla.py`):
  - built on `contract.py`, structured as `contract_plan.py`;
  - real buffer addresses from XRT;
  - `validate_list` must be empty before upload;
  - one list per arena.
- **Constants at load:**
  - the language rows: tokenizer padding "longest" (12 tokens for this task) × sqrt(960), in FP32;
  - the 10 time vectors;
  - the RoPE tables (torch's FP32 cos/sin → Q1.17 rows);
  - the rank vectors;
  - the folded weights (2.4).
- **ARM per chunk:**
  - resize, pad and normalize the images;
  - im2col of the patches, then int16 by `contract.quant_rows` (bit-exact with the hardware rule);
  - normalize the state, then int16;
  - draw the noise;
  - flush, start (`start_regs`), poll AP_CTRL bit 1, invalidate x_t, check `parse_return` and the status block
    (magic, version, starts).
- **One engine thread** owns the IP and runs (arena, list) jobs in FIFO order. Two arenas for two inferences in
  flight (design study section 6).
- `SmolVLAPolicy.predict_action_chunk` is not re-entrant (it shares `_queues`): preprocess under a lock.
- The task string is fixed, so the prefix length is 45 for this list. A new task string means a new list.

## 11. Unknowns (not guessed)

| unknown | why it matters | how to resolve |
|---|---|---|
| KR260 CMA location and size | where the buffers land (the contract supports any 40-bit address) | read `/proc/iomem` and the XRT BO paddr on the board |
| whether the BD wires the IP interrupt | completion by interrupt or by poll | Vivado BD, or poll |
| DDR latency, sustained bandwidth, 256/512-B burst efficiency | every io-bound mode; Y and X bursts | BT-2, BT-4 |
| the post-route LUT ratio of the int16 engine | the whole area budget (0.9k LUT margin) | one Vivado run of the merged top |
| whether E-0 reaches <= 3.6 ns | continuous or not (T 0.784 against 0.875 s) | engine csynth |
| the end-to-end precision of this exact recipe (folded scales, the contract quantizer in every producer, the smx P inside P·V) | the 0.05 gate (expected in the 1e-4 class on the benign frames) | T-P1 |
| the ARM preprocessing time and the link time (30 + 30 ms assumed) | T | board timing |
| whether cross-master read-after-write is safe after BRESP on HPC → HP | correctness of the runtime K/V blobs | BT-1; if it fails, the io port's AxCACHE (IP parameter C_M_AXI_IO_CACHE_VALUE). Whether changing that re-generates the SmartConnect is unknown |
