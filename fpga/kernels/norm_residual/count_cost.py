"""Time per -12392 chunk of the norm_residual kernel at 250 MHz, from the per-chunk call counts that
policy_check.py measured (policy_check.json) and the kernel's cycle model (norm_residual.h):

  per row     max(K/4                      [add, stats, norm, quant: one word a clock, flat loops],
                  A words read             [K/4 contiguous, 3K/8 when A is a record-layout residual stream],
                  words written            [WM_REC: K/4 + K/8; WM_C: K/8 (int16) or K/4 (FP32 z);
                                            WM_ROW: K/4 + K/8 + 2 (seg_ovh + wresp) per row])
  per chunk   (<= 256 rows) 2 x (seg_ovh + wresp) + rows / 4   [flat burst end, scale burst]
  per call    fill = ddr_lat + 3 K/4 + l_scal + l_qscal + l_pipe + call_ovh

seg_ovh (clocks a burst helper costs besides its words), l_scal / l_qscal (row scalar latencies) and l_pipe (sum of
the flat loops' pipeline depths) are csynth numbers. wresp (the write-response wait at the end of every burst:
HLS schedules writeresp at each burst loop's exit), ddr_lat (first read beat) and call_ovh are NOT measured on
the board: assumed, and printed as such.

Call structure per chunk (flags as the sequencer issues them, fpga/kernels/sequencer/build_smolvla.py):
  vision, 2 images in one call (B = 1, N = 512 rows): ln1[0] (no add), 23 x add + sum + norm, post (add, FP32 z)
  LM prefix: input_layernorm[0] (no add), 31 x add + sum, final norm (add; output unused: skippable)
  expert, per step x 10: input_layernorm[0] (no add), 31 x add + sum, final norm (add, int16 to action_out_proj)
Layouts: "records" = residual stream written as records [y FP32 row | q int16 row] (WM_REC) and read back flat
by the next call's A; "separate" = D and C in separate regions (WM_ROW: two bursts per row).
"""
from __future__ import annotations

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def call_clocks(K, rows, resid, sumo, qout, a_rec, layout, a):
    KW, CW = K / 4, (K / 8 if qout else K / 4)
    seg = a.seg_ovh + a.wresp
    rd = 1.5 * KW if a_rec else KW
    if sumo and layout == "records":
        wr = KW + CW
    elif sumo:
        wr = KW + CW + 2 * seg
    else:
        wr = CW
    row = max(KW, rd, wr)
    chunks = -(-rows // 256)
    per_chunk = 2 * seg + (rows / chunks / 4 if qout else 0)  # flat burst end + scale burst (its words)
    fill = a.ddr_lat + 3 * KW + a.l_scal + a.l_qscal + a.l_pipe + a.call_ovh
    return rows * row + chunks * per_chunk + fill, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "policy_check.json"))
    ap.add_argument("--layout", default="records", choices=["records", "separate"])
    ap.add_argument("--seg_ovh", type=float, default=6, help="csynth: clocks per burst helper besides its words")
    ap.add_argument("--l_scal", type=float, default=126, help="csynth: row scalar latency (stats -> mean, rstd)")
    ap.add_argument("--l_qscal", type=float, default=103, help="csynth: row quant scalar latency")
    ap.add_argument("--l_pipe", type=float, default=90, help="csynth: sum of the flat loops' pipeline depths")
    ap.add_argument("--wresp", type=float, default=60, help="ASSUMED: write-response wait per burst (clocks)")
    ap.add_argument("--ddr_lat", type=float, default=150, help="ASSUMED: clocks to the first read beat")
    ap.add_argument("--call_ovh", type=float, default=100, help="ASSUMED: clocks of start / done per call")
    ap.add_argument("--mhz", type=float, default=250.0)
    a = ap.parse_args()
    js = json.load(open(a.json))
    counts = {(c["group"], c["D"]): c for c in js["counts"]}
    nv = counts[("vision", 768)]["rows"] // counts[("vision", 768)]["calls"] * 2  # both images in one call
    n_lm = counts[("lm", 960)]["rows"] // counts[("lm", 960)]["calls"]
    n_ex = counts[("expert", 720)]["rows"] // counts[("expert", 720)]["calls"]
    steps = counts[("expert_final", 720)]["calls"]
    rec = a.layout == "records"
    # (name, K, rows, resid, sum, qout, A is a record stream, calls)
    calls = [
        ("vision ln1[0]", 768, nv, False, False, True, False, 1),
        ("vision ln2[0] (A = embed)", 768, nv, True, True, True, False, 1),
        ("vision add+norm", 768, nv, True, True, True, rec, 22),
        ("vision post (FP32 z)", 768, nv, True, False, False, rec, 1),
        ("lm in[0]", 960, n_lm, False, False, True, False, 1),
        ("lm post[0] (A = embed)", 960, n_lm, True, True, True, False, 1),
        ("lm add+norm", 960, n_lm, True, True, True, rec, 30),
        ("lm final (unused)", 960, n_lm, True, False, True, rec, 1),
        ("expert in[0]", 720, n_ex, False, False, True, False, steps),
        ("expert post[0] (A = embed)", 720, n_ex, True, True, True, False, steps),
        ("expert add+norm", 720, n_ex, True, True, True, rec, 30 * steps),
        ("expert final", 720, n_ex, True, False, True, rec, steps),
    ]
    tot_clk = tot_el = tot_calls = 0
    print(f"layout: {a.layout}")
    print(f"{'call':28s} {'K':>4s} {'rows':>5s} {'calls':>5s} {'clk/row':>8s} {'Mclk':>7s} {'el/clk':>6s}")
    for name, K, rows, resid, sumo, qout, a_rec, n in calls:
        clk1, row = call_clocks(K, rows, resid, sumo, qout, a_rec, a.layout, a)
        clk = n * clk1
        el = n * rows * K
        tot_clk += clk
        tot_el += el
        tot_calls += n
        print(f"{name:28s} {K:4d} {rows:5d} {n:5d} {row:8.1f} {clk / 1e6:7.3f} {el / clk:6.2f}")
    ms = tot_clk / a.mhz / 1e3
    print(f"total: {tot_calls} calls, {tot_el / 1e6:.2f} M elements, {tot_clk / 1e6:.2f} M clocks = {ms:.1f} ms at "
          f"{a.mhz:.0f} MHz ({tot_el / tot_clk:.2f} elements / clock, {tot_clk / tot_el:.3f} clocks / element)")
    print(f"assumed (not measured on the board): wresp = {a.wresp:.0f}, ddr_lat = {a.ddr_lat:.0f}, "
          f"call_ovh = {a.call_ovh:.0f} clocks")
    elem_check = sum(c["elements"] for c in js["counts"])
    print(f"policy_check counted {elem_check / 1e6:.2f} M norm elements per chunk (this model {tot_el / 1e6:.2f} M)")


if __name__ == "__main__":
    main()
