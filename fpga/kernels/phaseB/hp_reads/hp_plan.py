"""Per-chunk time of the -12392 list (fpga/kernels/contract_plan.py) with and without the hp_reads fusion.

The fusion (INTEGRATION.md): a row-unit / softmax command whose record sets FLD_HPR (field 56) reads the rows of
its slot A over two ports at once, rows 2i over io (xmem) and rows 2i+1 over one HP port, and processes the pair
in lockstep on 8 lanes (4 per row). Writes stay on io (one 128-bit word a clock). Per pair of rows:

    contract (4 lanes):  2 x (max(rd * passes, wr, K/4) + ROW_GAP)
    hp_reads (8 lanes):  max(rd * passes, 2 wr, K/4) + ROW_GAP          (the two reads run in parallel)
    softmax 4 lanes:     2 x (max(L/4, 13) + 2)
    softmax hp_reads:    max(L/4, 2 L/8, 13) + 2                         (two 1/sum dividers, one per row)

Command, item and zero-row terms are unchanged. Which opcodes take the fusion is a --ops choice; FP32-output
row ops (COPY, NORM without QOUT, ROPE FP32) gain nothing (write-bound at 4 FP32 a clock) and are left alone by
the builder rule `hpr_eligible`.

    python hp_plan.py [--mhz 214.2857 250 200] [--ops qz,softmax,norm,rope] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1]
sys.path.insert(0, str(KERNELS))
import contract_plan as CP  # noqa: E402
from contract import C  # noqa: E402

QZ_OPS = {C.OP_QUANT_X, C.OP_QUANT_W, C.OP_QUANT_WT, C.OP_COPY}
GROUPS = {
    "qz": QZ_OPS,
    "norm": {C.OP_NORM},
    "rope": {C.OP_ROPE},
    "act": {C.OP_GELU, C.OP_SILU_MUL},
    "softmax": {C.OP_SOFTMAX},
}


def q_out(c) -> bool:
    return bool(c.flags & C.F_QOUT or c.op in (C.OP_QUANT_W, C.OP_QUANT_WT) or c.flags & C.F_KW)


def hpr_eligible(c) -> bool:
    """The builder rule: only commands whose output is int16 (write words per row < read words per row) and that
    have at least 2 rows in the paired sequence gain from the second read port."""
    if c.op == C.OP_SOFTMAX:
        return c.B * c.N >= 2
    if c.op not in QZ_OPS | {C.OP_NORM, C.OP_ROPE, C.OP_GELU, C.OP_SILU_MUL}:
        return False
    return q_out(c) and c.B * c.N >= 2


def hp_clocks(c, ddr_bpc: float) -> tuple[float, int]:
    """hp_reads model of one command (same terms as contract_plan.cmd_clocks)."""
    if c.op == C.OP_SOFTMAX:
        L = c.K
        np_ = c.par(C.P_NP) or c.N
        per_pair = max(L / 4, 2 * L / 8, CP.SMX_DIV) + 2
        pairs = math.ceil(c.B * c.N / 2)
        zero = c.B * (np_ - c.N) * L / 8
        return CP.SEQ_FETCH + CP.SMX_C0 + c.B * CP.SMX_ITEM + pairs * per_pair + zero, c.B * c.N * L * 6
    K, N, seg = c.K, c.N, c.par(C.P_SEG) or c.K
    np_ = c.par(C.P_NP) or N
    q = q_out(c)
    rd = K / 4 * (2 if c.op == C.OP_SILU_MUL and not c.flags & C.F_NOMUL else 1)
    passes = 2 if c.op == C.OP_QUANT_WT or K > C.ROW_BUF else 1
    wr = K / 8 + (K // seg) / 4 if q else K / 4
    if c.op == C.OP_QUANT_WT:
        wr = K / 8
    per_pair = max(rd * passes, 2 * wr, K / CP.LANES) + CP.ROW_GAP
    pairs = math.ceil(c.B * N / 2)
    zero = c.B * (np_ - N) * wr
    byts = int(c.B * (N * rd * 16 * passes + np_ * wr * 16))
    return CP.SEQ_FETCH + CP.ROW_C0 + c.B * CP.ROW_ITEM + pairs * per_pair + zero, byts


def run(P, mhz: float, ddr_gbs: float, ops: set[int]) -> dict:
    ddr_bpc = ddr_gbs * 1e9 / (mhz * 1e6)
    ms = 1e3 / (mhz * 1e6)
    base = defaultdict(float)
    fused = defaultdict(float)
    ncmd = defaultdict(int)
    nfused = defaultdict(int)
    tot_b = tot_f = 0.0
    for c in P.cmds:
        clk, _ = CP.cmd_clocks(c, ddr_bpc)
        name = C.opnames[c.op]
        f = clk
        if c.op in ops and hpr_eligible(c):
            f, _ = hp_clocks(c, ddr_bpc)
            nfused[name] += 1
        if c.op != C.OP_GEMM:
            base[name] += clk
            fused[name] += f
            ncmd[name] += 1
        tot_b += clk
        tot_f += f
    Fb, Ff = tot_b * ms / 1e3, tot_f * ms / 1e3
    arm_link = 0.060
    return {
        "mhz": mhz,
        "F_contract_s": Fb,
        "F_hp_reads_s": Ff,
        "T_contract_s": Fb + arm_link,
        "T_hp_reads_s": Ff + arm_link,
        "saved_ms": (tot_b - tot_f) * ms,
        "saved_clocks": tot_b - tot_f,
        "by_op": {
            k: {
                "cmds": ncmd[k],
                "fused_cmds": nfused[k],
                "contract_ms": base[k] * ms,
                "hp_reads_ms": fused[k] * ms,
                "saved_ms": (base[k] - fused[k]) * ms,
            }
            for k in sorted(base, key=lambda k: -base[k])
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, nargs="*", default=[300 * 5 / 7, 250.0, 200.0])
    ap.add_argument("--ddr", type=float, default=14.0)
    ap.add_argument("--ops", default="qz,softmax,norm,rope")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    ops = set()
    for g in a.ops.split(","):
        if g:
            ops |= GROUPS[g]
    P = CP.build(0x8_0000_0000)
    assert not CP.validate_list(P.cmds)
    out = []
    for mhz in a.mhz:
        r = run(P, mhz, a.ddr, ops)
        r["ops"] = a.ops
        out.append(r)
        print(
            f"@ {mhz:.2f} MHz  ops={a.ops}: F {r['F_contract_s']:.4f} -> {r['F_hp_reads_s']:.4f} s "
            f"(saved {r['saved_ms']:.2f} ms = {r['saved_clocks']:.0f} clocks); T {r['T_contract_s']:.3f} -> "
            f"{r['T_hp_reads_s']:.3f} s; 2T {2 * r['T_hp_reads_s']:.3f} (bound 1.633)"
        )
        print("| opcode | cmds | fused | contract ms | hp_reads ms | saved ms |")
        print("|---|---|---|---|---|---|")
        for k, v in r["by_op"].items():
            print(
                f"| {k} | {v['cmds']} | {v['fused_cmds']} | {v['contract_ms']:.2f} | {v['hp_reads_ms']:.2f} | "
                f"{v['saved_ms']:.2f} |"
            )
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
