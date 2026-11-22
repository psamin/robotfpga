"""Time of this unit's commands (QUANT_X / QUANT_W / QUANT_WT / COPY) in one -12392 chunk, from the contract's
command list (fpga/kernels/contract_plan.py --out list.bin) and the cycle model of quant_mode.h:

  rd   reads 1 io word (4 FP32) a clock in 16-word bursts (rd16 II = 16), across rows and items: per row
       ceil(W / 16) units (QUANT_X rows above ROW_BUF values twice), per 64-value segment 1 unit, QUANT_WT
       2 units per (key, 64-column chunk) (column absmax pass, data pass).
       Rows of < 16 words: rd1, 1 word a clock.
  sr   2 dividers: >= 13.5 clocks per row / segment / column scale
  qt   1 input word a clock; wb / wr: 1 output word a clock (int16 output is half the input words)
  per command: CMD clocks (dataflow start + pipeline depths from csynth: rd16 90, wr8 78, the rest ~45) + L,
  the DDR read latency of the first burst (unknown on the board; reported for L = 0 / 64 / 128 clocks).

    python workload.py [--list build/chunk_hi.bin] [--mhz 250]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract as C  # noqa: E402

CMD = (
    215  # clocks per command besides the data (HLS: row_df min latency 175 for a 1-row call, + 40 sequencer)
)
ROW_BUF_W = C.C.ROW_BUF // 4
OPS = {C.C.OP_QUANT_X: "QUANT_X", C.C.OP_QUANT_W: "QUANT_W", C.C.OP_QUANT_WT: "QUANT_WT", C.C.OP_COPY: "COPY"}


def clocks(c) -> tuple[float, int]:
    """(data clocks, elements) of one command."""
    W, rows = c.K // 4, max(c.B, 1) * c.N
    seg = c.p.get(C.C.P_SEG, 0)
    if c.op == C.C.OP_QUANT_WT:
        chunks = -(-c.K // 64)
        return max(c.B, 1) * chunks * max(2 * c.N * 16, 64 * 13.5), rows * c.K  # P1 / P2 interleaved
    if c.op == C.C.OP_QUANT_W or (c.op == C.C.OP_QUANT_X and seg):
        nseg = c.K // 64
        return rows * nseg * max(16, 13.5), rows * c.K
    if W < 16:
        return rows * max(W, 13.5), rows * c.K
    units = -(-W // 16) * (2 if (c.op == C.C.OP_QUANT_X and W > ROW_BUF_W) else 1)
    return rows * max(units * 16, 13.5), rows * c.K  # a shifted-back last unit is a full 16-word unit too


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, default=HERE / "build" / "chunk_hi.bin")
    ap.add_argument("--mhz", type=float, default=250.0)
    args = ap.parse_args()
    cmds = [c for st, c in C.decode_list(args.list.read_bytes()) if c is not None and c.op in OPS]
    agg = defaultdict(lambda: [0, 0.0, 0])
    for c in cmds:
        d, el = clocks(c)
        a = agg[OPS[c.op]]
        a[0] += 1
        a[1] += d
        a[2] += el
    hz = args.mhz * 1e6
    print(f"{args.list}: {len(cmds)} commands of this unit, {args.mhz:.0f} MHz")
    print("| op | commands | M values | ms at L = 0 / 64 / 128 |")
    print("|---|---|---|---|")
    tot = [0.0, 0.0, 0.0]
    for name in ("QUANT_X", "QUANT_W", "QUANT_WT", "COPY"):
        n, d, el = agg[name]
        ms = [(d + n * (CMD + L)) / hz * 1e3 for L in (0, 64, 128)]
        tot = [t + m for t, m in zip(tot, ms, strict=True)]
        print(f"| {name} | {n} | {el / 1e6:.2f} | {ms[0]:.2f} / {ms[1]:.2f} / {ms[2]:.2f} |")
    print(f"| total | {len(cmds)} | | {tot[0]:.2f} / {tot[1]:.2f} / {tot[2]:.2f} |")


if __name__ == "__main__":
    main()
