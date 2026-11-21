# float_top command sequencer

The ARM writes one command list per arena into CMA once. Each chunk it writes only the per-inference
inputs and starts `float_top` once. The IP walks the list through its `io` port and dispatches every
command to one mode function (GEMM, norm, softmax, ...). This replaces today's ~1,419 engine calls per
chunk at ~2.5 ms each, plus all the ARM glue between them.

The design adds no register and no port. It uses `float_top`'s existing argument list, bundles and
`s_axi_control` map. The modes themselves are other kernels' work. In this directory they are stubs: a
small but real int16 GEMM, plus placeholder streamers.

| file | what |
|---|---|
| `seq_format.h` | the record format, opcodes, flags and status codes. This is the single source of truth: `seq_format.py` parses it |
| `seq_core.h` | header-only sequencer: `seq::run<Modes>()` (fetch, check, decode, the single dispatch site, END / YIELD / STEP / error, status block). Used unchanged from `float_top` |
| `seq_stubs.h` | stub modes for the standalone synthesis: `gemm_stub` (32 lanes x 2 tokens int16 MAC, the engine's weight layout over hp0..3) and `stream_stub<KIND>` |
| `seq_top.cpp` | standalone HLS top with the same signature and pragmas as `fpga/hls/float_top.cpp`. `SEQ_VARIANT` 0..5 selects the dispatch experiments |
| `run_seq_hls.tcl`, `run_hls_all.sh` | csynth (xck26-sfvc784-2LV-c, 4.0 / 3.333 ns). The shell script runs the variants one after another |
| `tb_seq.cpp` | host testbench: GEMM stub bit-exact against a scalar reference, every walk and error path, the legacy protocol, and the whole -12392 list decoded by the C++ core against the Python builder's trace |
| `seq_format.py` | Python twin of the format: `Cmd`, `Slot`, `encode_list` / `decode_list`, `start_regs`, `parse_return` |
| `arena.py` | CMA plan: a shared region (weights, consts, lists) plus two per-inference arenas, and the bump allocator with reusable "role" scratch buffers |
| `build_smolvla.py` | builder: one -12392 chunk (vision, connector, LM prefix, cross-KV hoist, 10 Euler steps x 16 expert layers) as a command list with every buffer address |
| `exec_ref.py` | PyTorch executor of a list: the same walk as `seq_core.h` over byte-accurate CMA, each mode's contract, and the exact int16 GEMM arithmetic |
| `check_policy.py` | full-policy check: `predict_action_chunk(d["batch"], noise=d["noise"])` with `sample_actions` replaced by host inputs, then the list walk, then x_t. Compared with `d["ref_chunk_raw"]` on -12392's 3 frames, alternating arenas |
| `count_cmds.py` | commands per chunk / layer / step, a lint of the list against the mode contracts, and the host-time estimate |
| `hls_summary.py` | csynth totals, instance tree, and how many GEMM-engine / MAC-pipeline instances each variant has |

## Register protocol (unchanged `s_axi_control` map)

Offsets: `AP_CTRL` 0x00, `GIE` 0x04, `IER` 0x08, `ISR` 0x0c, `AP_RETURN` 0x10, `w0..w3` 0x18 / 0x24 /
0x30 / 0x3c, `xmem` 0x48, `ymem` 0x54, `M` 0x60, `K` 0x68, `N` 0x70. These are the same as `float_top`.
They were checked in the generated `xseq_top_hw.h`.

| register | sequencer mode |
|---|---|
| `M` | `[31:24]` = `MODE_MAGIC` 0xA5 selects the sequencer. `[23]` = single step. `[22:0]` = index of the first command |
| `K` | list base physical address >> 4 (16-byte aligned) |
| `N` | status block physical address >> 4 (0 = no status block) |
| `w0..w3`, `xmem`, `ymem` | **0**. Records carry absolute physical byte addresses, so AXI address = 16 x array index |
| `AP_RETURN` | `(next index << 8) \| status code` |

When `M[31:24] != 0xA5`, `float_top` runs one legacy call exactly as today: `run()` executes the
caller-built legacy command through the same dispatch site. Completion is `ap_done`. The IP has its
`interrupt` pin (GIE + IER bit 0, ISR toggle-on-write). Whether the board design wires it to the PS is
**unknown**: there is no BD script in `fpga/`. Until then, poll `AP_CTRL` bit 1.

Status codes: `END` 0x00, `YIELD` 0x01 (a command had `F_YIELD`; resume at next), `STEP` 0x02, `LEGACY`
0x03, `BAD_MAGIC` 0x10, `BAD_CHECK` 0x11, `BAD_OPCODE` 0x12, `MODE_ERROR` 0x13 (next = the failing command),
`LIMIT` 0x14.

Status block (2 words at `N << 4`): word 0 = `code, next, tag, done`. Word 1 = `mode error, 0x53455153 'SEQS',
starts, last opcode`. `tag` is the tag of the last command run, or of the bad record. `done` counts the
commands dispatched in this start. `starts` counts sequencer starts since configuration, so a stale
block is visible. Mode errors: `ME_DIMS` 1, `ME_ALIGN` 2.

## Command record (128 bytes = 8 io words, 128-bit aligned; command i at list + 128 i)

32 little-endian uint32 fields. Field j is lane `j % 4` of word `j / 4`, the io lane order of `float_linear.h`.

| field | content |
|---|---|
| f0 | `[7:0]` opcode, `[15:8]` 0xC5 magic, `[31:16]` flags |
| f1 | tag. The builder uses `stage << 24 \| layer << 16 \| step << 8 \| counter` |
| f2, f3, f4 | M (output columns), K (input columns / reduction), N (rows per batch item) |
| f5 | `[15:0]` B batch items (0 = 1), `[31:16]` G: operand B advances every G items (GQA; 0 = 1) |
| f6..f23 | six operand slots A..F, three fields each: `addr[31:0]`, `addr[39:32] \| pitch[23:0] << 8` (row pitch in bytes; 0 = broadcast where allowed), batch stride in bytes. Item b uses `addr + b * stride` (B and E: `(b / G) * stride`) |
| f24..f30 | p0..p6 opcode parameters (FP32 bit patterns or integers) |
| f31 | check = 0x5EC0C5ED ^ f0 ^ ... ^ f30 |

Why 128 bytes: six fully strided operands with 40-bit addresses take 18 fields. 64 bytes would force
implicit layouts, and the fetch is one 8-beat burst either way. The list is position-independent:
operand addresses are absolute and the list base is a register. It does not depend on the inputs, so it
is built once per arena (per language-token count if `compact`) and stays in CMA.

### Opcodes and mode contracts (what `exec_ref.py` implements and the mode kernels must match)

All FP32 tensors are row-major with a pitch. "q-out" (`F_QOUT`) means: write int16 rows to C and one FP32
scale per row to F, with `s = amax|row| / 32767` and `q = round_half_even(x / s)`, the engine's
`quantize_int16_tokens` rule.

| op | code | slots / params |
|---|---|---|
| NOP / END | 0x00 / 0x01 | END stops with `ST_END` (not counted in `done`) |
| GEMM | 0x10 | A = Xq int16 [N][K], F = s_tok FP32 [N]. B = weight blob data (pitch = per-port array bytes, the engine's port tiling), E = s_row FP32 [M] (bias FP32 [M] at E + 4M if `F_BIAS`). C = Y FP32. D = FP32 rows added if `F_ADD_D` (pitch 0 = one row broadcast). `F_ACC`: C += result. p0 = FP32 output multiplier, p1 = true (unpadded) M. `y = fp32(int48 acc) * s_row * s_tok * p0 (+ bias) (+ D) (+ C)`, in that order. M % 32 = 0, K % 8 = 0, K <= 4096 |
| QUANT_X | 0x20 | A FP32 [N][K] to C int16 + F scales (q-out always) |
| QUANT_W | 0x21 | A FP32 [N][K] to a weight blob at C (rows padded to 32, the engine's layout) + F row scales. Used for K blobs |
| QUANT_WT | 0x22 | A FP32 [N][K] transposed to a blob of K rows x p0 (>= N, zero-padded) reduction columns at C + F scales. Used for V^T blobs |
| NORM | 0x30 | h = A (+ B if `F_RESID`; B pitch/stride 0 = broadcast). `F_SUM`: D = h (may alias B). `F_LN`: LayerNorm with gamma at E and beta at E + 4K; else RMSNorm with gamma at E. eps = p0. Output C, FP32 or q-out |
| SOFTMAX | 0x40 | rows of A (K keys) x p0. `F_MASK`: key k is visible to row n iff `0 <= E[k] <= D[n]` (int32 vectors, per-item stride). A fully masked row outputs zeros. Output C, q-out in the list |
| GELU | 0x50 | tanh-GELU of A to C (q-out in the list) |
| SILU_MUL | 0x51 | SiLU(A) * B (`F_NOMUL`: SiLU(A)) to C (q-out in the list) |
| ROPE | 0x60 | rows of A (K = 64 per head item). Positions int32 at D (per row). Table at E: row = position, `[cos(32) \| sin(32)]` FP32, pitch E.pitch. y1 = x1 cos - x2 sin, y2 = x2 cos + x1 sin. In place allowed (C = A) |
| COPY | 0x70 | K bytes per row, N rows, B items, A to C (pixel shuffle = 4 commands) |
| GATHER | 0x71 | table at A (BF16 if `F_BF16`, pitch A.pitch), ids int32 at D, N rows of K to C FP32, x p0 |
| AXPY | 0x80 | C = A + p0 * B (Euler step; in place) |

Commands run strictly in order. A mode returns only after its last write. In-place / aliasing uses
are: ROPE C = A, NORM D = B, AXPY C = A, GEMM `F_ACC`.

## DDR layout (`arena.py`, measured sizes for -12392 int16)

One physically contiguous CMA block: `[weights | consts | lists] [arena 0] [arena 1]`. The FPGA runs
arena A while the ARM fills arena B's inputs and reads arena A's actions, so two inferences are in
flight. Buffers are 64-byte aligned (one A53 cache line). Weight blobs and role buffers are 4 KB
aligned.

| region | reserved | used (int16) | content |
|---|---|---|---|
| weights | 1024 MB | 768.7 MB (806,053,120 B) | int16 blobs `[Mp][K]` in port tiling + FP32 row scales (+ bias); BF16 token-embedding table |
| consts | 8 MB | 1.19 MB | vision pos-embedding + patch bias, RoPE table (256 x 64), per-step time bias (10 x 736), all norm gammas/betas |
| lists | 4 MB | 0.69 MB | one list per arena: 2,824 records = 361,472 B each |
| arena 0 / 1 | 96 MB each | 40.4 MB (42,326,016 B) each | status, inputs (im2col patches 1.5 MB, ids, state, x_t = noise, positions, mask block ids), persistent KV caches (16 layers x 131 rows), cross-attention K/V blobs, h_vis / h_prefix / h_suffix, and 13 role scratch buffers shared by all stages (xq 3.1 MB, sc 6.3 MB, t 6.3 MB, ...) |

The used total is ~0.85 GB. The reservations can shrink to ~0.95 GB total. The board's CMA size is
**unknown** (not checked here).

Per chunk the host writes about 1.6 MB of inputs (`Program.host_inputs`), flushes them, starts the IP
once, and reads 6.4 KB of x_t.

## Commands per chunk (-12392, `count_cmds.py`)

**2,824 records** (2,823 dispatched + END). 1,283 of them are GEMM commands, covering 6,731 GEMM items
and 110.1 GMAC. Attention is batched over heads (B = 12 / 15, G = 3 for GQA). q|k|v, k|v and gate|up are
fused.

| stage | commands | per unit |
|---|---|---|
| vision (2 images batched, 512 tokens) | 207 | 17 per layer (+3 for the embedding) |
| connector + prefix | 11 | pixel shuffle 4 COPY, 3 K-split GEMMs (`F_ACC`), gather, state |
| LM prefix | 243 | 16 per layer. The last layer only fills its KV cache (3) |
| cross-attention K/V hoist (once per chunk) | 32 | 4 per odd expert layer |
| expert (10 steps x 16 layers) | 2,300 | 16 per self-attention layer, 12 per cross-attention layer, plus 6 per step for the suffix embedding (tagged layer 0) |
| action head + Euler | 31 | |
| **per Euler step** | 233 | |

By opcode: GEMM 1283, NORM 386, ROPE 271, QUANT_X 230, SOFTMAX 187, SILU_MUL 185, QUANT_W 127,
QUANT_WT 127, GELU 12, AXPY 10, COPY 4, GATHER 1, END 1. The lint is clean: GEMM dims, 16-byte alignment
of every streamed slot, B % G.

**Host time.** Today: 1,419 calls x 2.5 ms = **3.55 s per chunk**. With the sequencer: 1 start x ~0.1 ms +
~3.1 ms of input copy/flush = **~3.2 ms** (11 starts if `F_YIELD` after every Euler step: ~4.2 ms; one
start per layer, 190 starts: ~22 ms). So **~3.54 s of host time is saved per chunk**. The 0.1 ms per start
and 3.1 ms for inputs are assumptions: the hardware_options "tightened driver" figure and xrtmem's
measured 3 MB copy. Neither was measured for this flow. The ARM nonlinear glue also disappears, but that
saving belongs to the mode kernels.

## Synthesis (Vitis HLS 2022.2 csynth, xck26-sfvc784-2LV-c)

| variant | clock | est. | LUT | FF | DSP | BRAM18 | GEMM engine instances | MAC pipeline instances |
|---|---|---|---|---|---|---|---|---|
| v0: one call site per mode in a switch (the design) | 4.0 | 3.239 | 54,915 | 53,399 | 141 | 16 | **1** | **1** (64 DSP) |
| v0 | 3.333 | 3.239 | 54,902 | 59,374 | 141 | 16 | 1 | 1 |
| v1: two exclusive call sites, same args | 4.0 | 3.239 | 54,944 | 53,403 | 141 | 16 | 1 | 1 |
| v2: v1 + `ALLOCATION instances limit=1` | 4.0 | **5.661** | 53,213 | 44,966 | 116 | 16 | 1 | 1 |
| v4: two sequential call sites (items split in half) | 4.0 | 3.239 | 55,300 | 53,636 | 123 | 16 | **1** | **1** |
| v5: v4 + `ALLOCATION` | 4.0 | **5.661** | 53,569 | 45,199 | 98 | 16 | 1 | 1 |
| v3: no modes (null consumer of every field) | 4.0 | 2.920 | 8,413 | 6,003 | 5 | 0 | 0 | 0 |
| v3 | 3.333 | 2.433 | 8,411 | 6,006 | 5 | 0 | 0 | 0 |

Totals include the AXI adapters (control 1,064 LUT, hp0..3 2,318 each, io 3,155). `float_top` already
has those. The 5 DSP at the top are the stub's legacy-layout multiplies.

**Cost of the sequencer itself**, beyond what `float_top` already has:
- v3: `run` 4,020 LUT / 3,792 FF minus the null consumer 1,212 / 141 gives about **2.8k LUT, 3.65k FF, 0 DSP, 0 BRAM**. That includes the fetch (341 LUT) and the status writer (166).
- v0, with 8 mode instances multiplexed onto the ports: `run`'s own logic is 41,178 - 37,360 = **3.8k LUT** and 47,688 - 44,650 = **3.0k FF**, again 0 DSP / 0 BRAM. That is about 10 % of the 35-45k LUT kernel budget.

The stubs are not the modes' costs: GEMM stub 23k LUT / 94 DSP, streamers 1.7-2.4k LUT each.

**Dispatch without duplicating the engine.**
- One non-inlined mode function called from a `switch` inside the command loop is one hardware instance. The csynth instance tree shows a single `gemm_stub` and a single `Pipeline_mac` (64 DSP).
- Two call sites do not duplicate it either. This was tested both for mutually exclusive branches (v1) and for two sequential calls in one path (v4): Vitis 2022.2 time-shares a non-inlined function across sequential call sites. v4 only adds the address multipliers of the split.
- `ALLOCATION function instances limit=1` is unnecessary and harmful here. It re-flattens the GEMM emit loop, and the estimate drops to 5.66 ns, which fails 250 MHz. Do not use it.
- Requirement: every mode function must be `#pragma HLS INLINE off`. An inlined mode is duplicated per call site, and HLS auto-inlines small functions.

**Cycles per command** (HLS schedule):
- fetch 19 cycles (8-beat burst plus the AXI model), decode/check about 3, null mode 7. The `cmd_loop` iteration is 3 ~ 32 cycles.
- So about **25 cycles = 0.1 us per command at 250 MHz, or 0.28 ms per chunk** for 2,823 commands. The real DDR read latency adds to the fetch, maybe 25-50 cycles more (**unknown**, not measured on the board). That gives ~0.5-0.9 ms per chunk, under 0.1 % of the 1.1 s budget, so prefetching the next record is not needed.
- The modes' own start/drain is theirs to report.

## Accuracy (full policy, -12392, 3 replay frames, vs `ref_chunk_raw`)

The sequencer adds no arithmetic. What the check measures is that the list is correct (addresses,
pitches, aliasing, order) and the int16 arithmetic of all GEMM operands. All other modes are kept FP32;
their error is their owners' to report.

| mode | frame 0 | frame 3563 | frame 7489 |
|---|---|---|---|
| laptop torch FP32 policy (no list) | 5.1e-6 | 4.6e-7 | 4.8e-7 |
| `exact`: list walked in FP32 (checks the list) | 1.27e-5 | 3.0e-7 | 3.0e-7 |
| `int16`: int16 GEMMs incl. Q K^T and P V, exact sums | **1.96e-4** (mean 4.6e-5) | **1.20e-4** (3.5e-5) | **1.34e-4** (4.4e-5) |
| `int16c`: + prefix compacted to its 45 valid rows | 2.45e-4 | 1.69e-4 | 1.39e-4 |

Values are max |chunk - ref|, normalized actions. The gate is 0.05. Every frame ran from one start,
and the arenas alternated 0 / 1 / 0.

## Integration into `float_top`

1. Body: decode the legacy registers into a `seq::Cmd`, then `return seq::run<Modes>(w0, w1, w2, w3, xmem, ymem, M, K, N, legacy)`.
   `Modes::dispatch` is a switch with exactly one call per mode function, each `INLINE off`. The engine's DATAFLOW region becomes the GEMM mode function, which restarts for every GEMM command.
2. Modes take `const seq::Cmd&` and the six port pointers. They address `port[(addr >> 4) + i]` with running pointers (adds, no runtime multiplies; see the stubs). They return `ME_*`.
3. Layouts to reconcile with `int16_linear.h`'s current driver contract:
   - X: the engine reads its own xmem image (s_row, s_tok, then X interleaved per 32-token block). The list gives row-major int16 rows with a pitch, s_tok in slot F, and s_row in slot E inside the weight blob.
   - Y: the engine writes group-major `Y[g][Np][32]`. The list expects row-major FP32 with a pitch. Row-major Y means 8-beat (128 B) bursts per (token, group) unless the store buffers several groups. This needs the engine owner's decision.
   - Alternative: keep the native layouts and have QUANT_X and the Y consumers use them. Only `build_smolvla.py`'s slots and `exec_ref`'s views would change.
4. Weights read through hp0..3. Blobs that QUANT_W/QUANT_WT write through io (K, V^T) are read back through hp0..3 by the next GEMM. **Write-to-read ordering across two AXI masters** relies on each mode having all its write responses before it returns. Verify in cosim or on the board; this is not verified here.
5. Host: build `Shared` + `Program` per arena once and write the lists and consts once. Flush the lists once. Per chunk: write the inputs, flush, start (`start_regs`), wait, invalidate x_t, and check `parse_return` (and the status block's `starts` / `magic`).

## Commands run

```
# full-policy check (torch venv, HF_HUB_OFFLINE=1); writes build/py/int16.json and the dump for tb_seq
C:\kp\lerobot-venv\Scripts\python.exe check_policy.py <ckpt>/smolvla-so101_medicine_bottle_pickplace-12392-native <ckpt>/batches-12392 --modes int16,int16c --dump build/py --out build/py/int16.json
#   (exact mode: build/py/exact.json, from the run before the crash)
python count_cmds.py --out build/py/counts.json
sh run_hls_all.sh "0:4.0 1:4.0 2:4.0 3:4.0 0:3.333"; sh run_hls_all.sh "4:4.0 5:4.0 3:3.333"
python hls_summary.py --out build/hls_summary.json
# host testbench (Vitis MinGW, clean PATH)
g++ -std=c++14 -O2 -Wall -Wno-unknown-pragmas -Wno-unused-label -o build/tb_seq.exe tb_seq.cpp && build/tb_seq.exe build/py
```

`tb_seq.cpp` compiles cleanly for variants 0, 3 and 4. **It has not been run.** Since 19:34 on
2026-11-20, Windows Smart App Control blocks every newly built unsigned executable on this laptop
(CodeIntegrity events 3077/3118), including the other kernels' testbenches. The C++ core's decode, walk
and error paths, and the GEMM stub's bit-exactness, are therefore verified only by synthesis and by the
Python twin (`exec_ref.py`). Run `build/tb_seq.exe build/py` once executables are allowed again.
