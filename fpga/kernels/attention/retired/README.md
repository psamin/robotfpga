# Retired attention hardware (pre-contract and the first conform attempt)

Kept so the measurements below can be reproduced. None of this is part of the contract build
(`fpga/kernels/CONTRACT.md` 4.1, 5): attention runs on the GEMM, the softmax unit and the row unit.

| file | what it was | why retired |
|---|---|---|
| `attn_pack.h`, `attn_pack_top.cpp` | descriptor-driven packer (QX / KW / VW / OX), own 384-byte descriptor, own register map | superseded by the contract record (CONTRACT.md 3); QZ_RECIP_BITS = 2 clashed with the row unit's 1 bit in one TU (review); 27.5-28.3k HLS LUT, no data-path bursts |
| `attn_engine_model.h` | C model of the per-call engine with the proposed `N[23:16]` head loop | head batching is now B items in the record (E-5); the tb uses the contract GEMM model or the engine core |
| `attn_kv.h`, `attn_kv_top.cpp`, `run_attn_hls.tcl` | QUANT_W / QUANT_WT as a separate contract mode function `attn::quant_kv(cmd, xmem, ymem)` (the first conform attempt, WIP 52f4962) | a second reader / scaler / quantizer / writer next to the row unit; the row unit (`quantize/quant_mode.h`, `qz::row_mode`) already runs QUANT_W / QUANT_WT and matched the contract reference on every attention command (`../results/tb_attn.log`) |
| `count_cost.py`, `smx_bridge.py` | pre-contract cost model, torch bridge to the softmax emulation | replaced by `../attn_cost.py` (contract_plan's model) and `attn_int16` calling smx_emu directly |
| `README_pre_contract.md` | the pre-contract README (descriptors, masks, accuracy, cost) | |

`attn::quant_kv` csynth, Vitis HLS 2022.2, xck26-sfvc784-2LV-c, 4.0 ns target (run 2026-11-21 on the current
file, before it was moved here; log `../results/hls_kv_final_p40.log`):

| | LUT | FF | DSP | BRAM18 | estimated | loops |
|---|---|---|---|---|---|---|
| quant_kv (mode function) | 26,241 | 19,455 | 12 | 16 | 2.920 ns (342 MHz) | every pipelined loop II = 1; bursts inferred on the row reads, the absmax scan and the write runs |
| of which writer (8 x 8 x 32 LUTRAM tile + emit mux) | 9,379 | 5,480 | 0 | 0 | | |
| of which scaler (25-stage reciprocal) + quant (4 lanes) | 8,549 | 6,663 | 12 | 0 | | |
| top incl. io adapter and control | 32,934 | 23,331 | 12 | 16 | | |

The contract's cap for the whole row unit (8 opcodes) is 15k HLS LUT, so a separate QUANT_W / QUANT_WT unit
of 26k cannot be added. The earlier WIP build (2026-11-20 22:50) was 32,618 LUT / 21 DSP at 4.062 ns.

To rebuild the retired csynth: `cd retired && vitis_hls -f run_attn_hls.tcl` (PATH without Git dirs).
