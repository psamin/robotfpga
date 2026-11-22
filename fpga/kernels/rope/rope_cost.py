"""ROPE work and time per -12392 chunk: the contract list (contract_plan.build) with this mode's pipeline.

For every ROPE record of the list: contract.validate must pass, and the shape is classified (Q int16
head-major, KW blob, FP32 in place). Time per record, in clocks (rope_mode.h, csynth of build/rope_p40):
  seq   80                      the sequencer's record fetch + decode (contract_plan SEQ_FETCH)
  fill  FILL (+ 64 for F_KW)    pipeline depths from csynth: rd 25, rot 12 + 16 (one segment of ping-pong),
                                emit 10 + 16, sc 6 + rc 27, qt 12, kw 5 (KW: + 64, one block), pk 2 (+ 16 for
                                the last scale unit), wr 15: about 250
  ddr   2 * LAT                 the first read burst and the last write response (LAT unknown, --lat)
  body  16 * B * H * NP         rd's II = 16 read unit per (row, head) segment is the bottleneck: rot / emit /
                                qt take 16 clocks a segment, pk writes <= 16 words a segment (Q 8 + 1/4 scale
                                word per row, KW 8.25, F32 16)

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\rope\\rope_cost.py [--mhz 250] [--lat 100]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract  # noqa: E402
import contract_plan as CP  # noqa: E402

C = contract.C

SEQ, FILL, KW_FILL = 80, 250, 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, default=250.0)
    ap.add_argument("--lat", type=int, default=100, help="DDR latency in clocks (unknown; an assumption)")
    a = ap.parse_args()
    P = CP.build(0x8_0000_0000)
    by = defaultdict(lambda: [0, 0, 0, 0.0])  # records, segments, rows, clocks
    bad = 0
    for c in P.cmds:
        if c.op != C.OP_ROPE:
            continue
        if contract.validate(c) != C.ME_OK:
            bad += 1
        H = c.K // C.ROPE_SEG
        np_ = c.par(C.P_NP) or c.N
        kw = bool(c.flags & C.F_KW)
        kind = "KW" if kw else "Q int16" if c.flags & C.F_QOUT else "FP32"
        key = f"{kind:7s} {c.note.split(':')[0]:6s} N={c.N:3d} NP={np_:3d} H={H:2d} B={c.B}"
        segs = c.B * H * np_
        clk = SEQ + FILL + (KW_FILL if kw else 0) + 2 * a.lat + 16 * segs
        r = by[key]
        r[0] += 1
        r[1] += segs
        r[2] += c.B * c.N * H
        r[3] += clk
    tot = [sum(v[i] for v in by.values()) for i in range(4)]
    ms = 1e3 / (a.mhz * 1e6)
    print(f"{'shape':44s} {'records':>7s} {'segments':>9s} {'ms':>7s}")
    for k, v in sorted(by.items()):
        print(f"{k:44s} {v[0]:7d} {v[1]:9d} {v[3] * ms:7.2f}")
    print(f"{'total':44s} {tot[0]:7d} {tot[1]:9d} {tot[3] * ms:7.2f}")
    fixed = tot[0] * (SEQ + FILL + 2 * a.lat) * ms
    print(
        f"records invalid: {bad}; (row, head) segments read: {tot[2]} live + {tot[1] - tot[2]} zero; "
        f"per-record fixed cost {fixed:.2f} ms of it (DDR latency {a.lat} clocks assumed, unmeasured)"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
