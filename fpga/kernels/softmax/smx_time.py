"""Cycle model of the SOFTMAX unit (softmax_core.h) on the SOFTMAX records of the contract's -12392 chunk list
(fpga/kernels/contract_plan.py), against the contract's own budget rule (CONTRACT.md 6.2).

The model follows the unit's structure; every term is an estimate (no cosim, no board):

  reader   per item: [rank_k burst: W words] + per block of <= 256 rows [rank_q burst: rows / 4 words] +
           the scores (one burst per block when the A pitch is 4 L, else one per row). Each burst costs its
           words + RD_LAT (DDR read latency, unmeasured: --rd-lat) + RD_FILL (the words_r pipeline, 15 stages,
           csynth)
  compute  NP rows x (W + 1) clocks (1 setup clock per row), W = L / 4; + 3 pad scale lanes
  divider  NP4 rows x 14 clocks
  writer   per block of <= 256 rows: the P rows as one burst when the C pitch is 2 L (else one burst per row),
           then the block's scale words as one burst; each burst costs its words + WR_BURST (csynth: words_w
           pipeline 3 + loop 2)
  an item takes max(compute, divider, writer) + the reader's exposed burst latencies; a command adds
  CMD_FIXED (prologue, DATAFLOW start / drain) + RD_LAT + WR_LAT (the last write response).

    python fpga/kernels/softmax/smx_time.py [--mhz 250] [--rd-lat 64] [--wr-lat 64] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract_plan as CP  # noqa: E402
from contract import C  # noqa: E402

RD_FILL, WR_BURST, CMD_FIXED, DIV_ROW, BLK = 15, 5, 40, 14, 256


def cmd_clocks(c, rd_lat: int, wr_lat: int) -> dict:
    L, N, B = c.K, c.N, c.B or 1
    NP = c.par(C.P_NP) or N
    NP4 = CP.pad4(NP)
    W, W2 = L // 4, L // 8
    mask = bool(c.flags & C.F_MASK)
    a, cs = c.slot("A"), c.slot("C")
    contig, c_contig = a.pitch == 4 * L, cs.pitch == 2 * L
    e_every_item = mask and c.slot("E").stride != 0
    blocks = math.ceil(N / BLK)
    total = CMD_FIXED + rd_lat + wr_lat
    rd_exposed = 0
    for b in range(B):
        bursts = (
            (blocks if contig else N)
            + (blocks if mask else 0)
            + (1 if mask and (b == 0 or e_every_item) else 0)
        )
        exposed = bursts * (rd_lat + RD_FILL)
        compute = NP * (W + 1) + 3
        divider = NP4 * DIV_ROW
        writer = 0
        for r0 in range(0, NP, BLK):
            rows = min(BLK, NP - r0)
            writer += (
                (rows * W2 + WR_BURST if c_contig else rows * (W2 + WR_BURST)) + (rows + 3) // 4 + WR_BURST
            )
        total += max(compute, divider, writer) + exposed
        rd_exposed += exposed
    return dict(clocks=total, exposed=rd_exposed, rows=B * N, elems=B * N * L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, default=250.0)
    ap.add_argument("--rd-lat", type=int, default=64, help="DDR read latency in clocks (unmeasured)")
    ap.add_argument("--wr-lat", type=int, default=64, help="write response latency in clocks (unmeasured)")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x8_0000_0000)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    P = CP.build(a.base)
    ms = 1e3 / (a.mhz * 1e6)
    ddr_bpc = 14.0e9 / (a.mhz * 1e6)
    sites = defaultdict(lambda: dict(cmds=0, items=0, rows=0, elems=0, clocks=0.0, contract=0.0, exposed=0.0))
    shapes = defaultdict(int)
    for c in P.cmds:
        if c.op != C.OP_SOFTMAX:
            continue
        r = cmd_clocks(c, a.rd_lat, a.wr_lat)
        ph = c.note.split(":")[0]
        s = sites[ph]
        s["cmds"] += 1
        s["items"] += c.B or 1
        s["rows"] += r["rows"]
        s["elems"] += r["elems"]
        s["clocks"] += r["clocks"]
        s["exposed"] += r["exposed"]
        s["contract"] += CP.cmd_clocks(c, ddr_bpc)[0] - CP.SEQ_FETCH
        shapes[(ph, c.B, c.N, c.par(C.P_NP) or c.N, c.K, bool(c.flags & C.F_MASK))] += 1
    tot = {
        k: sum(v[k] for v in sites.values())
        for k in ("cmds", "items", "rows", "elems", "clocks", "contract", "exposed")
    }
    print(
        f"SOFTMAX records of the contract list at {a.mhz:.0f} MHz, "
        f"read latency {a.rd_lat}, write {a.wr_lat} clocks"
    )
    print(
        "| phase | records | items | rows | unit model ms | of which exposed read latency ms |"
        " contract rule ms |"
    )
    print("|---|---|---|---|---|---|---|")
    for k, v in list(sites.items()) + [("total", tot)]:
        print(
            f"| {k} | {v['cmds']} | {v['items']} | {v['rows']} | {v['clocks'] * ms:.2f} |"
            f" {v['exposed'] * ms:.2f} |"
            f" {v['contract'] * ms:.2f} |"
        )
    print("shapes (phase, B, N, NP, L, mask): records")
    for k, v in sorted(shapes.items()):
        print(f"  {k}: {v}")
    if a.json:
        out = dict(
            mhz=a.mhz,
            rd_lat=a.rd_lat,
            wr_lat=a.wr_lat,
            sites={
                k: {kk: (vv * ms if kk in ("clocks", "contract", "exposed") else vv) for kk, vv in v.items()}
                for k, v in sites.items()
            },
            total_ms=tot["clocks"] * ms,
            contract_ms=tot["contract"] * ms,
        )
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
