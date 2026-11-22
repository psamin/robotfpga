"""Cycle and DDR savings of the qk_softmax fusion per -12392 chunk, with the contract's plan and time model
(contract_plan.py as patched in integration/: build(), cmd_clocks and the QK_SOFTMAX rule qks_cmd_clocks).

For each clock and engine option it builds the contract 1.0 list and the fused list (fuse_qk_softmax), times both
with the same model, and reports F, T = F + ARM 30 ms + link 30 ms (both estimates), the continuity margins
(1 in flight: 2 x 0.8167 - 2 T; 2 in flight: 1.6333 - (F + T)), the attention part per site, and the DDR bytes.

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\phaseB\\qk_softmax\\qks_time.py [--json results/time.json]

Labels: every number is an estimate of the contract's model (no csim cycle count, no board).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "integration" / "kernels"))
import contract_plan as CP  # noqa: E402  (the integration copy: fuse_qk_softmax, qks_cmd_clocks)
from contract import C  # noqa: E402

assert hasattr(C, "OP_QK_SOFTMAX") and CP.__file__.startswith(str(HERE)), CP.__file__


def attn_rows(P, mhz, ddr):
    """Per site (phase), the clocks of the records the fusion touches (QK GEMM + SOFTMAX, or QK_SOFTMAX)."""
    bpc = ddr * 1e9 / (mhz * 1e6)
    out = defaultdict(lambda: [0, 0.0, 0])
    for c in P.cmds:
        if (c.op == C.OP_GEMM and c.note.endswith(":qk")) or c.op in (C.OP_SOFTMAX, C.OP_QK_SOFTMAX):
            clk, byts = CP.cmd_clocks(c, bpc)
            site = c.note.split(":")[0]
            if site == "steps":
                site = "steps (self + cross)"
            out[site][0] += 1
            out[site][1] += clk
            out[site][2] += byts
    return out


def run(mhz, ddr, item_fixed, nbx=2):
    CP.OPT["nbx"], CP.OPT["item_fixed"] = nbx, item_fixed
    res = {}
    for fuse in (False, True):
        CP.OPT["fuse_qk"] = fuse
        P = CP.build(0x8_0000_0000)
        bad = CP.validate_list(P.cmds)
        assert not bad, bad[:3]
        r = CP.report(P, mhz, ddr, 30.0, 30.0)
        r["attn"] = {k: {"records": v[0], "ms": v[1] / (mhz * 1e3), "MB": v[2] / 1e6} for k, v in attn_rows(P, mhz, ddr).items()}
        res["fused" if fuse else "contract"] = r
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "results" / "time.json"))
    ap.add_argument("--ddr", type=float, default=14.0)
    a = ap.parse_args()
    out = {}
    print("| clock | engine items | records | F contract | F fused | saved | T fused | margin 1 in flight | "
          "margin 2 in flight | DDR GB saved |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mhz in (300.0, 250.0, 300.0 * 5 / 7, 200.0, 187.5):
        for e7 in (None, 200):
            r = run(mhz, a.ddr, e7)
            cu, fu = r["contract"], r["fused"]
            key = f"{mhz:.1f}MHz_{'E7' if e7 else 'noE7'}"
            out[key] = r
            print(
                f"| {mhz:.1f} MHz | {'E-7 (200 clk/item)' if e7 else 'per-item runs'} | {cu['commands']} -> "
                f"{fu['commands']} | {cu['F_s']:.4f} s | {fu['F_s']:.4f} s | {1e3 * (cu['F_s'] - fu['F_s']):.1f} ms | "
                f"{fu['T_s']:.4f} s | {fu['margin_1_in_flight_s']:+.4f} s (was {cu['margin_1_in_flight_s']:+.4f}) | "
                f"{fu['margin_2_in_flight_s']:+.4f} s (was {cu['margin_2_in_flight_s']:+.4f}) | "
                f"{cu['ddr_GB'] - fu['ddr_GB']:.3f} |"
            )
    for key in ("214.3MHz_noE7", "214.3MHz_E7", "250.0MHz_noE7"):
        r = out[key]
        print(f"\n{key}: attention records the fusion touches")
        print("| site | contract records | contract ms | contract MB | fused records | fused ms | fused MB |")
        print("|---|---|---|---|---|---|---|")
        for site in r["contract"]["attn"]:
            cu, fu = r["contract"]["attn"][site], r["fused"]["attn"][site]
            print(f"| {site} | {cu['records']} | {cu['ms']:.2f} | {cu['MB']:.1f} | {fu['records']} | {fu['ms']:.2f} | "
                  f"{fu['MB']:.1f} |")
        tot_c = sum(v["ms"] for v in r["contract"]["attn"].values())
        tot_f = sum(v["ms"] for v in r["fused"]["attn"].values())
        mb_c = sum(v["MB"] for v in r["contract"]["attn"].values())
        mb_f = sum(v["MB"] for v in r["fused"]["attn"].values())
        print(f"| total | | {tot_c:.2f} | {mb_c:.1f} | | {tot_f:.2f} | {mb_f:.1f} |")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
