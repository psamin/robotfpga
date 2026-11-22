"""Time per chunk of the attention path under the contract: its records in contract_plan's -12392 list.

Attention is no unit of its own (CONTRACT.md 5): it is GEMMs (Q.K^T, P.V), SOFTMAX and row-unit records
(the Q producer with p4 = -3, the K blob by QUANT_W or ROPE F_KW, the V^T blob by QUANT_WT, the o_proj
input QUANT_X). This picks those records out of contract_plan.build() (the 2,701-record chunk) and sums
contract_plan's own clock model (cmd_clocks: the engine's cycle model for GEMM, the row-unit and softmax
rules, 80 clocks of sequencer fetch per record), per role, at several clocks, with and without engine change
E-7 (items inside one DATAFLOW run). Estimates: nothing here ran in co-simulation or on the board, and the
DDR rate is the contract's unmeasured 14 GB/s.

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\attention\\attn_cost.py [--ddr 14]
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

# the pre-contract design (attn_pack era, results/cost.json before this conform), prefix compacted to 45,
# ms at 250 MHz: engine 57.2 (61.8 with the engine's own call model, review), packer 72.4 (single-beat
# bursts: est. 180-290), softmax 27.8; and 948 kernel calls per chunk (19-95 ms of host time)
PRE_CONTRACT_MS = {"engine": 57.2, "engine_review": 61.8, "packer": 72.4, "softmax": 27.8}


def role(c) -> str | None:
    n = c.note or ""
    if c.op == C.OP_GEMM:
        if n.endswith(":qk"):
            return "GEMM Q.K^T"
        if ":pv" in n:
            return "GEMM P.V"
        return None
    if c.op == C.OP_SOFTMAX:
        return "SOFTMAX"
    if c.op == C.OP_QUANT_W or (c.op == C.OP_ROPE and c.flags & C.F_KW):
        return "K blob (QUANT_W / ROPE F_KW)"
    if c.op == C.OP_QUANT_WT:
        return "V^T blob (QUANT_WT)"
    if c.op in (C.OP_QUANT_X, C.OP_ROPE) and c.par(C.P_SHIFT) == (-3 & 0xFFFFFFFF):
        return "Q (QUANT_X / ROPE, p4 = -3)"
    if c.op == C.OP_QUANT_X and n.endswith("o.in"):
        return "o_proj input (QUANT_X)"
    return None


def tally(plan, mhz: float, ddr: float) -> dict:
    bpc = ddr * 1e9 / (mhz * 1e6)
    ms = 1e3 / (mhz * 1e6)
    out = defaultdict(lambda: {"records": 0, "items": 0, "ms": 0.0})
    total = 0.0
    for c in plan.cmds:
        clk = CP.cmd_clocks(c, bpc)[0]
        total += clk
        r = role(c)
        if r is None:
            continue
        out[r]["records"] += 1
        out[r]["items"] += c.B
        out[r]["ms"] += clk * ms
    res = {k: {**v, "ms": round(v["ms"], 2)} for k, v in sorted(out.items(), key=lambda x: -x[1]["ms"])}
    att = sum(v["ms"] for v in out.values())
    return {
        "mhz": mhz,
        "roles": res,
        "attention_ms": round(att, 1),
        "chunk_F_ms": round(total * ms, 1),
        "attention_share": round(att / (total * ms), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddr", type=float, default=14.0, help="GB/s (contract figure, unmeasured)")
    ap.add_argument("--out", default=str(HERE / "results" / "cost.json"))
    a = ap.parse_args()
    plan = CP.build(0x8_0000_0000)
    runs = {}
    for mhz in (250.0, 222.0, 200.0):
        CP.OPT["item_fixed"] = None
        runs[f"{mhz:.0f}MHz"] = tally(plan, mhz, a.ddr)
        CP.OPT["item_fixed"] = 200  # E-7
        runs[f"{mhz:.0f}MHz_E7"] = tally(plan, mhz, a.ddr)
    CP.OPT["item_fixed"] = None
    res = {"ddr_GBps": a.ddr, "pre_contract_ms_250": PRE_CONTRACT_MS, "runs": runs}
    for k, v in runs.items():
        share = v["attention_share"]
        print(f"{k:10s} attention {v['attention_ms']:6.1f} ms of F {v['chunk_F_ms']:6.1f} ms ({share:.1%})")
        if k == "250MHz":
            for r, x in v["roles"].items():
                print(f"    {r:30s} {x['records']:4d} records {x['items']:5d} items {x['ms']:6.1f} ms")
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
