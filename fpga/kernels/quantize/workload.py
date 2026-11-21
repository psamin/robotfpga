"""QUANT_X workload of one SmolVLA chunk and this kernel's time for it (cycle model of quant_mode.h).

Reads the sequencer's command list (fpga/kernels/sequencer/build/py/chunk_list.bin, written by its
check_policy.py --dump; read only) and, per QUANT_X command (B items x N rows x K), models

  rd   per port, its rows one after another: K/4 + RD_OVH + L clocks a row (L = DDR read latency of one
       burst, unknown on this board: reported for L = 0, 40, 80); two ports, alternate rows
  rc   (2 + 25) / 2 = 13.5 clocks a row
  qt   K / 8 clocks a row (8 values a clock)
  wr   K / 8 clocks a row + (WR_OVH + Lw) per 64-row segment (Lw: write response latency, taken = L)
  a row costs the max of the four, plus the per-call start / drain (CALL_OVH clocks)

    python workload.py [--list path] [--mhz 250]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ = HERE.parent / "sequencer"
RD_OVH = 8      # clocks a row besides the burst in rd (HLS schedule: scan loop depth 4 + outer loop)
WR_OVH = 20     # clocks a segment besides the data in wr (two loop entries + the scale burst)
CALL_OVH = 60   # dataflow start, pipeline fill / drain per call (HLS latency of a 1-row call ~48)
RC = 13.5


def cycles(B: int, N: int, K: int, L: int) -> float:
    rows = B * N
    rd = (K / 4 + RD_OVH + L) / 2
    qt = K / 8
    wr = K / 8 + (WR_OVH + L) / 64
    two_pass = K > 1024
    if two_pass:
        rd *= 2  # absmax pass + data pass
    return rows * max(rd, RC, qt, wr) + CALL_OVH + L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, default=SEQ / "build/py/chunk_list.bin")
    ap.add_argument("--mhz", type=float, default=250.0)
    args = ap.parse_args()
    sys.path.insert(0, str(SEQ))
    from seq_format import C, decode_list  # noqa: E402

    raw = args.list.read_bytes()
    cmds = decode_list(raw)
    q = [c for c in cmds if c.op == C.OP_QUANT_X]
    groups = defaultdict(lambda: [0, 0])  # (B, N, K) -> [calls, elements]
    for c in q:
        B = c.B or 1
        groups[(B, c.N, c.K)][0] += 1
        groups[(B, c.N, c.K)][1] += B * c.N * c.K
    tot_el = sum(v[1] for v in groups.values())
    print(f"{len(cmds)} commands, {len(q)} QUANT_X, {tot_el / 1e6:.2f} M elements per chunk")
    print(f"{'B':>3} {'N':>5} {'K':>6} {'calls':>6} {'Melem':>7} | ms at L=0 / 40 / 80 clocks ({args.mhz:.0f} MHz)")
    totals = {0: 0.0, 40: 0.0, 80: 0.0}
    for (B, N, K), (n, el) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        ms = {L: n * cycles(B, N, K, L) / (args.mhz * 1e3) for L in totals}
        for L in totals:
            totals[L] += ms[L]
        print(f"{B:>3} {N:>5} {K:>6} {n:>6} {el / 1e6:>7.3f} | {ms[0]:.3f} / {ms[40]:.3f} / {ms[80]:.3f}")
    ideal = tot_el / 8 / (args.mhz * 1e3)
    print(f"total: {totals[0]:.2f} / {totals[40]:.2f} / {totals[80]:.2f} ms per chunk (8 values a clock with no "
          f"overhead: {ideal:.2f} ms); clocks per element {totals[0] * args.mhz * 1e3 / tot_el:.3f} / "
          f"{totals[40] * args.mhz * 1e3 / tot_el:.3f} / {totals[80] * args.mhz * 1e3 / tot_el:.3f}")


if __name__ == "__main__":
    main()
