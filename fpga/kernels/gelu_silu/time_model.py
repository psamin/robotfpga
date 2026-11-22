"""Time per 50-action chunk of the GELU / SILU_MUL mode on the -12392 list (contract_plan.py), contract 1.0.

Every GELU / SILU_MUL record of the list goes through the same path rule as gs_plan (gelu_silu.h): dense
read (one burst stream per item, pitch gaps <= GAPW words dropped) and dense write (one burst per chunk of
WCH rows, then the chunk's scale words). The cycle model of one command, from the csynth schedule (every loop
II = 1):

    clocks = FILL + B * (LAT_RD + N * per_row) + LAT_WR
    per_row = max(words read per row (rper), front-end iterations per row (rowits), quantizer words (K / 4),
                  scale-unit clocks per row (SC_ROW))

FILL is the pipeline fill of the chain rd -> fe -> am -> sc -> qt -> wr (csynth depths 4 + 21 + 4 + SC_ROW +
13 + 4) plus the mode's entry. The writer's per-chunk write-response waits are hidden: it drains at 1 word a
clock while the int16 rows arrive at 1 word per 2 clocks (half a word per input word), and the row buffer
(896 words) absorbs the stall. LAT_RD / LAT_WR are the DDR read / write round trips: csynth models 0; --lat
sets them (an assumption, not measured on the board). The contract's own estimate for the same records
(contract_plan.cmd_clocks: 480 + 64 per item + (max(read, write, K/4) + 4) per row + 80 for the record fetch)
is printed alongside.

    python time_model.py [--mhz 250] [--lat 100] [--ddr 14]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract_plan as CP  # noqa: E402
from contract import C  # noqa: E402

GAPW, WCH = 64, 64  # gelu_silu.h
GAP = 32  # SILU_MUL idle iterations for K / 4 < GAP
SC_ROW = 34  # gs_sc 'rows' iteration: scale_bits + 25-step recip + stored scale (csynth)
DEPTHS = 4 + 21 + 4 + 13 + 4  # rd, fe, am, qt, wr pipeline depths (csynth, gelu_silu v5)
ENTRY = 12  # act_mode: gs_plan + dataflow start (csynth)


def act_clocks(c, lat: int) -> tuple[float, bool, bool]:
    mul = c.op == C.OP_SILU_MUL and not c.flags & C.F_NOMUL
    k4 = c.K // 4
    rw = 2 * k4 if mul else k4
    a, b, out = c.slot("A"), c.slot("B"), c.slot("C")
    pa = a.pitch // 16
    one = c.N == 1
    fused = not mul or (
        b.addr == a.addr + 4 * c.K and (one or b.pitch == a.pitch) and (c.B <= 1 or b.stride == a.stride)
    )
    rdense = (one or (pa >= rw and pa - rw <= GAPW)) and fused
    q = bool(c.flags & C.F_QOUT)
    orw = k4 // 2 if q else k4
    np_ = c.par(C.P_NP) or c.N
    wdense = np_ == 1 or out.pitch // 16 == orw
    rper = (rw if one else pa) if rdense else rw
    rowits = (k4 + max(GAP - k4, 0) + k4) if mul else k4
    per_row = max(rper, rowits, k4, SC_ROW)
    if not rdense:  # one burst per row (A, then B): the read latency on every row
        per_row += lat * (2 if mul else 1)
    clocks = DEPTHS + SC_ROW + ENTRY + max(c.B, 1) * (lat + c.N * per_row) + lat
    return clocks, rdense, wdense


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, default=250.0)
    ap.add_argument("--lat", type=int, default=100, help="DDR read / write round trip in clocks (assumed)")
    ap.add_argument("--ddr", type=float, default=14.0, help="GB/s, for the contract's model (unmeasured)")
    a = ap.parse_args()
    plan = CP.build(0x8_0000_0000)
    ddr_bpc = a.ddr * 1e9 / (a.mhz * 1e6)
    groups: dict = defaultdict(lambda: [0, 0.0, 0.0, 0, 0, 0])
    for c in plan.cmds:
        if c.op not in (C.OP_GELU, C.OP_SILU_MUL):
            continue
        clk, rd, wd = act_clocks(c, a.lat)
        cclk, _ = CP.cmd_clocks(c, ddr_bpc)
        name = "GELU" if c.op == C.OP_GELU else ("SILU" if c.flags & C.F_NOMUL else "SILU_MUL")
        g = groups[(name, c.K, c.N)]
        g[0] += 1
        g[1] += clk
        g[2] += cclk
        g[3] += c.N * c.K * max(c.B, 1)
        g[4] += rd
        g[5] += wd
    print(f"{a.mhz:.0f} MHz, DDR round trip {a.lat} clocks (assumed); contract model at {a.ddr:.0f} GB/s")
    print("| op | K | N | commands | dense read / write | elements | this model ms | contract model ms |")
    print("|---|---|---|---|---|---|---|---|")
    tot = ctot = 0.0
    for (name, K, N), g in sorted(groups.items()):
        ms, cms = g[1] / (a.mhz * 1e3), g[2] / (a.mhz * 1e3)
        tot += ms
        ctot += cms
        print(
            f"| {name} | {K} | {N} | {g[0]} | {g[4]} / {g[5]} | {g[3] / 1e6:.2f} M | {ms:.2f} | {cms:.2f} |"
        )
    print(f"| **total** | | | {sum(g[0] for g in groups.values())} | | | **{tot:.2f}** | **{ctot:.2f}** |")


if __name__ == "__main__":
    main()
