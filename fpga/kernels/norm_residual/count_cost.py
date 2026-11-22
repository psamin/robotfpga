"""NORM time per -12392 chunk under contract 1.0: the 386 NORM records of fpga/kernels/contract_plan.py, timed
with this kernel's cycle model (norm_residual.h) and with the contract's row-unit model
(contract_plan.cmd_clocks), side by side.

Kernel model, per command (one DATAFLOW region; every flat loop one 128-bit word a clock):
  per row   max(A words read   [flat read: the row pitch in words, gap words read and dropped; per-row bursts:
                                K/4 + ddr_lat, the DDR latency is not overlapped across rows],
                C words written [K/8 int16 or K/4 FP32; + wresp per row when C rows are not contiguous],
                K/4             [stats, affine, absmax, quantize],
                l_scal          [p_scal: mean / var dividers, csynth row latency, LN or RMS],
                l_rsqrt         [p_rsqrt: eps add + rsqrt, csynth row latency])
  per chunk of <= 256 rows   a scale burst (rows / 4 words) + 2 wresp (the data burst's and the scale burst's)
  per command   gamma (| beta) load (K/4 or K/2 words + ddr_lat) + ddr_lat + region fill (one row + l_scal +
                l_rsqrt + l_pipe) + call_ovh
l_scal / l_rsqrt / l_pipe come from csynth (nr_top_csynth.rpt); ddr_lat, wresp and call_ovh are NOT measured
on the board: assumptions, printed as such.

  C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\norm_residual\\count_cost.py [--mhz 250]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract_plan as CP  # noqa: E402
from contract import C  # noqa: E402


def kernel_clocks(c, a) -> float:
    K, N, B = c.K, c.N, c.B
    ln, q = bool(c.flags & C.F_LN), bool(c.flags & C.F_QOUT)
    NP = c.par(C.P_NP) or N
    kw, cw = K / 4, (K / 8 if q else K / 4)
    A, Cs = c.slot("A"), c.slot("C")
    apw = A.pitch / 16
    flat = N == 1 or (kw <= apw <= 1.5 * kw)
    rd = apw if flat else kw + a.ddr_lat
    cflat = NP == 1 or Cs.pitch / 16 == cw
    wr = cw if cflat else cw + a.wresp
    l_scal = a.l_scal_ln if ln else a.l_scal_rms
    row = max(rd, wr, kw, l_scal, a.l_rsqrt)
    chunks = -(-NP // 256)
    per_chunk = (NP / chunks / 4 if q else 0) + 2 * a.wresp
    fill = (2 * kw if ln else kw) + 2 * a.ddr_lat + kw + l_scal + a.l_rsqrt + a.l_pipe + a.call_ovh
    return B * (N * row + (NP - N) * cw + chunks * per_chunk) + fill


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, default=250.0)
    ap.add_argument("--l_scal_ln", type=float, default=167, help="csynth: p_scal clocks per LayerNorm row")
    ap.add_argument("--l_scal_rms", type=float, default=127, help="csynth: p_scal clocks per RMSNorm row")
    ap.add_argument("--l_rsqrt", type=float, default=127, help="csynth: p_rsqrt clocks per row")
    ap.add_argument("--l_pipe", type=float, default=80, help="csynth: summed flat-loop pipeline depths")
    ap.add_argument("--ddr_lat", type=float, default=100, help="ASSUMED: DDR read latency, clocks")
    ap.add_argument("--wresp", type=float, default=60, help="ASSUMED: write-response wait per burst, clocks")
    ap.add_argument("--call_ovh", type=float, default=100, help="ASSUMED: dispatch + region start, clocks")
    a = ap.parse_args()
    P = CP.build(0x8_0000_0000)
    ddr_bpc = 14e9 / (a.mhz * 1e6)
    groups = defaultdict(lambda: [0, 0, 0.0, 0.0])  # (kind, K, out) -> calls, rows, kernel clk, contract clk
    for c in P.cmds:
        if c.op != C.OP_NORM:
            continue
        key = ("LN" if c.flags & C.F_LN else "RMS", c.K, "q16" if c.flags & C.F_QOUT else "f32")
        g = groups[key]
        g[0] += 1
        g[1] += c.B * c.N
        g[2] += kernel_clocks(c, a)
        g[3] += CP.cmd_clocks(c, ddr_bpc)[0]
    tk = tc = 0.0
    print(
        f"NORM records of contract_plan.py at {a.mhz:.0f} MHz (kernel model vs the contract's row-unit model)"
    )
    print(
        f"{'kind':4s} {'K':>5s} {'out':>4s} {'calls':>6s} {'rows':>7s} "
        f"{'kernel ms':>10s} {'contract ms':>12s}"
    )
    for k, (n, rows, ck, cc) in sorted(groups.items()):
        print(
            f"{k[0]:4s} {k[1]:5d} {k[2]:>4s} {n:6d} {rows:7d} "
            f"{ck / a.mhz / 1e3:10.2f} {cc / a.mhz / 1e3:12.2f}"
        )
        tk += ck
        tc += cc
    n = sum(g[0] for g in groups.values())
    print(f"total {n} records: kernel {tk / a.mhz / 1e3:.2f} ms, contract model {tc / a.mhz / 1e3:.2f} ms")
    print(
        f"assumed (not measured): ddr_lat {a.ddr_lat:.0f}, wresp {a.wresp:.0f}, "
        f"call_ovh {a.call_ovh:.0f} clocks"
    )


if __name__ == "__main__":
    main()
