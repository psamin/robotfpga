"""Time per chunk of the QUANT_W / QUANT_WT mode (attn_kv.h) on the contract's -12392 command list.

Takes every QUANT_W and QUANT_WT record of contract_plan.build() (the 2,701-record chunk, CONTRACT.md 6) and times it
with a clock model of attn_kv.h's loops, against contract_plan's own row-unit rule for the same records.

Model (clocks at the kernel clock; estimates, nothing here ran on the board or in co-simulation):
  - every burst costs its beats + its latency (L for a read burst until the first beat, Lw for a write burst until
    its write response: HLS waits for BRESP at the end of each burst loop) + LOOP clocks of loop entry and exit;
    L and Lw on the KR260 are unknown (contract BT-2), so they are swept
  - a DATAFLOW pass costs PASS0 to start and drain; per (block, chunk) the slowest of reader / quant / writer counts
    (the FIFOs between them hold more than one chunk, tb_attn's bounded-FIFO model checks they never deadlock)
  - the scaler: one scale per 13 clocks (2 dividers of 26 clocks); QUANT_WT waits for its table (13 cw clocks)
  - SEQ: the sequencer's 80 clocks per record (contract_plan's figure), included on both sides

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\attention\\attn_cost.py [--mhz 250]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract_plan as CP  # noqa: E402
from contract import C  # noqa: E402

PASS0 = 60  # DATAFLOW start + pipeline fill / drain of 4 processes (quant_elem is about 12 stages)
MODE0 = 20  # mode entry and pass setup
LOOP = 3  # entry / exit of one burst loop
DIV = 13  # clocks per scale (2 staggered dividers of 26 clocks)


def chunks(k: int, cw: int) -> list[int]:
    n = -(-k // cw)
    return [cw] * (n - 1) + [k - (n - 1) * cw]


def kw_clocks(c, L: float, Lw: float) -> tuple[float, int, int]:
    """(clocks, read bursts, write bursts) of one QUANT_W record."""
    K, N, NP = c.K, c.N, c.par(C.P_NP) or c.N
    seg64 = c.par(C.P_SEG) != 0
    W = 64 if seg64 else K
    long = W > 256
    cws = chunks(K, 256 if (seg64 or long) else K)
    nblk = NP // 8
    t, rb, wb = 0.0, 0, 0
    for _ in range(c.B):
        if not long:
            t += PASS0
            for _blk in range(nblk):
                for cw in cws:
                    segs = cw // 64 if seg64 else 1
                    rd = 8 * (cw / 4 + L + LOOP)
                    qt = 2 * cw
                    wr = cw + 8 * segs + segs * ((W + Lw + LOOP) + (2 + Lw + LOOP))
                    t += max(rd, qt, wr)
                    rb += 8
                    wb += 2 * segs
        else:
            for _blk in range(nblk):
                t += sum(8 * (cw / 4 + L + LOOP) for cw in cws) + 8 * DIV + 26 + PASS0 + 8 + Lw + LOOP
                t += sum(max(8 * (cw / 4 + L + LOOP), 2 * cw, 2 * cw + Lw + LOOP) for cw in cws)
                rb += 16 * len(cws)
                wb += len(cws) + 1
    return CP.SEQ_FETCH + MODE0 + t, rb, wb


def vw_clocks(c, L: float, Lw: float) -> tuple[float, int, int]:
    """(clocks, read bursts, write bursts) of one QUANT_WT record."""
    K, N, NP = c.K, c.N, c.par(C.P_NP) or c.N
    seg64 = c.par(C.P_SEG) != 0
    t, rb, wb = 0.0, 0, 0
    for _ in range(c.B):
        for cw in chunks(K, 128):
            t += 32 + N * (cw / 4 + L + LOOP)  # scan: clear, then the column absmax
            two = seg64 and cw > 64
            runs, runw = (2, 16) if two else (1, cw // 4)
            t += PASS0 + DIV * cw + 26 + runs * (4 * runw + Lw + LOOP)  # table, then its scale words (II = 4)
            rb += N
            wb += runs
            n0 = 0
            while n0 < NP:
                r = min(16, NP - n0)
                rd = r * (cw / 4 + L + LOOP)
                qt = r * cw / 4
                wr = r * cw / 8 + (cw / 8) * (r + Lw + LOOP)
                t += max(rd, qt, wr)
                rb += r
                wb += cw // 8
                n0 += r
    return CP.SEQ_FETCH + MODE0 + t, rb, wb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, default=250.0)
    ap.add_argument("--ddr", type=float, default=14.0, help="GB/s for contract_plan's own rule")
    ap.add_argument("--out", default=str(HERE / "results" / "cost.json"))
    args = ap.parse_args()
    plan = CP.build(0x8_0000_0000)
    recs = [c for c in plan.cmds if c.op in (C.OP_QUANT_W, C.OP_QUANT_WT)]
    ddr_bpc = args.ddr * 1e9 / (args.mhz * 1e6)
    ms = 1e3 / (args.mhz * 1e6)
    res = dict(mhz=args.mhz, records={}, plan_rule_ms={}, mode_ms={}, bursts={})
    counts = defaultdict(int)
    for c in recs:
        counts[C.opnames[c.op]] += 1
    res["records"] = dict(counts)
    for op in (C.OP_QUANT_W, C.OP_QUANT_WT):
        res["plan_rule_ms"][C.opnames[op]] = (
            sum(CP.cmd_clocks(c, ddr_bpc)[0] for c in recs if c.op == op) * ms
        )
    for L in (0, 32, 64, 100):
        row, br = {}, {}
        for op, fn in ((C.OP_QUANT_W, kw_clocks), (C.OP_QUANT_WT, vw_clocks)):
            tot = [fn(c, L, L) for c in recs if c.op == op]
            row[C.opnames[op]] = sum(x[0] for x in tot) * ms
            br[C.opnames[op]] = dict(read=sum(x[1] for x in tot), write=sum(x[2] for x in tot))
        row["total"] = sum(row.values())
        res["mode_ms"][f"L={L}"] = row
        res["bursts"] = br
    res["plan_rule_ms"]["total"] = sum(res["plan_rule_ms"].values())
    print(json.dumps(res, indent=1))
    Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
