"""Cycle savings of emit_act per -12392 chunk, on the contract's own list and time model (contract_plan.py).

The unfused list is contract_plan.build() as it is (2,701 records). The fused list replaces every
(GEMM -> GELU | SILU_MUL | SILU_MUL+NOMUL) pair whose act record reads the GEMM's output by:

  standalone  GEMM with p6 = P_ACT writing the activated FP32 rows (EPI_SILU_MUL: M / 2 columns), then the
              QUANT_X that turns them into the int16 X + s_tok the next GEMM reads (same C / F slots as the act
              record had). This is what the emit_act patch alone needs.
  +emit_amax  GEMM with p6 = P_ACT only: the QUANT_X goes too, because the next GEMM quantizes FP32 X on load
              (phase B emit_amax, P_AMX_SG / P_XQ). Whatever P_AMX_SG / P_XQ cost or save on the GEMMs themselves
              is emit_amax's to count; here the two GEMMs are timed as plain GEMMs.

Every fused record is checked with the patched contract.validate (patched/kernels/contract.py). The GEMM time rule
is contract_plan.gemm_item_clocks with the two changes of the patch:
  - the dequantizer takes 2 values a clock with P_ACT: the per-(group, block) term is max(K, 512, EMIT) instead of
    max(K, 256, EMIT), and the drain after the last block 256 clocks longer;
  - EPI_SILU_MUL writes M / 2 FP32 columns: half the Y bytes in the DDR term and in the byte count.

    C:/kp/lerobot-venv/Scripts/python.exe fpga/kernels/phaseB/emit_act/savings.py [--mhz 250 214.29 200 187.5]
        [--json out.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patched" / "kernels"))  # the patched contract (P_ACT, EPI_*) first
sys.path.insert(1, str(HERE.parents[1]))  # fpga/kernels: contract_plan
import contract  # noqa: E402  (patched/kernels/contract.py, reads patched/kernels/contract.h)
import contract_plan as cp  # noqa: E402

C, Cmd, Slot = contract.C, contract.Cmd, contract.Slot
assert Path(contract.__file__).resolve().parent == HERE / "patched" / "kernels", contract.__file__
DQ_ACT = 2 * cp.YW_BLK  # dequantizer clocks per (group, block) with P_ACT
TAIL_ACT = cp.YW_BLK  # extra drain of the last block


def gemm_item_clocks_act(mp, k, n, ddr_bpc, d_read, first, act):
    """cp.gemm_item_clocks with the emit_act changes (act = EPI_*; 0 = the unchanged rule)."""
    if not act:
        return cp.gemm_item_clocks(mp, k, n, ddr_bpc, d_read, first)
    nbx = cp.OPT["nbx"]
    g, nblk = mp // 32, math.ceil(n / 32)
    tiles = [min(nbx, nblk - t * nbx) for t in range(math.ceil(nblk / nbx))]
    fixed = cp.L0 + cp.TAIL if first or cp.OPT["item_fixed"] is None else cp.OPT["item_fixed"]
    total = fixed + TAIL_ACT + 4 * tiles[0] * k
    yb = 4096 if act != C.EPI_SILU_MUL else 2048  # Y bytes per (group, block)
    for t, nbt in enumerate(tiles):
        nxt = tiles[t + 1] if t + 1 < len(tiles) else 0
        dw = cp.YW_BLK * g * nbt if d_read else 0
        total += max(
            g * nbt * max(k, DQ_ACT, cp.EMIT),
            g * k,
            (64 * g * k + yb * g * nbt + 16 * dw + 64 * nxt * k) / ddr_bpc,
            4 * nxt * k + dw,
        )
    return total


def cmd_clocks(c: Cmd, ddr_bpc: float):
    act = c.par(C.P_ACT) if c.op == C.OP_GEMM else 0
    if not act:
        return cp.cmd_clocks(c, ddr_bpc)
    d = bool(c.flags & (C.F_ADD_D | C.F_ACC)) and c.slot("D").pitch != 0 or bool(c.flags & C.F_ACC)
    clk = sum(gemm_item_clocks_act(c.M, c.K, c.N, ddr_bpc, d, b == 0, act) for b in range(min(c.B, 2)))
    clk += max(c.B - 2, 0) * gemm_item_clocks_act(c.M, c.K, c.N, ddr_bpc, d, False, act)
    mo = c.M // 2 if act == C.EPI_SILU_MUL else c.M
    byts = c.B * (2 * c.M * c.K + 2 * c.N * c.K + 4 * c.N * mo * (2 if d else 1))
    return cp.SEQ_FETCH + clk, byts


def fuse(cmds: list[Cmd], amax: bool) -> tuple[list[Cmd], list[dict]]:
    out, pairs, i = [], [], 0
    while i < len(cmds):
        c = cmds[i]
        nx = cmds[i + 1] if i + 1 < len(cmds) else None
        if (
            c.op == C.OP_GEMM
            and nx is not None
            and nx.op in (C.OP_GELU, C.OP_SILU_MUL)
            and nx.slot("A").addr == c.slot("C").addr
        ):
            act = (
                C.EPI_GELU
                if nx.op == C.OP_GELU
                else (C.EPI_SILU if nx.flags & C.F_NOMUL else C.EPI_SILU_MUL)
            )
            g = copy.deepcopy(c)
            g.p = dict(g.p)
            g.p[C.P_ACT] = act
            F = c.M // 2 if act == C.EPI_SILU_MUL else c.M
            if act == C.EPI_SILU_MUL:
                assert nx.K == F and nx.slot("B").addr == c.slot("C").addr + 4 * F, nx.note
            ys = c.slot("C")
            g.s = dict(g.s)
            g.s["C"] = Slot(ys.addr, 4 * F if act == C.EPI_SILU_MUL else ys.pitch, ys.stride)
            g.note = c.note + "+act"
            out.append(g)
            pairs.append(dict(gemm=c.note, act=nx.note, epi=act, M=c.M, K=c.K, N=c.N, F=F))
            if not amax:
                q = Cmd(
                    C.OP_QUANT_X,
                    C.F_QOUT,
                    0,
                    0,
                    nx.K,
                    nx.N,
                    nx.B,
                    {"A": Slot(ys.addr, g.s["C"].pitch, ys.stride), "C": nx.slot("C"), "F": nx.slot("F")},
                    {},
                )
                q.note = nx.note.split(":")[0] + ":act.quant_x"
                out.append(q)
            i += 2
            continue
        out.append(c)
        i += 1
    for j, c in enumerate(out):
        c.tag = j
    return out, pairs


def report(cmds: list[Cmd], mhz: float, ddr_gbs: float) -> dict:
    ddr_bpc = ddr_gbs * 1e9 / (mhz * 1e6)
    by = defaultdict(lambda: [0, 0.0, 0])
    tot = 0.0
    for c in cmds:
        clk, byts = cmd_clocks(c, ddr_bpc)
        name = C.opnames[c.op]
        if c.op == C.OP_GEMM and c.par(C.P_ACT):
            name = "GEMM+ACT"
        elif c.note.endswith("act.quant_x"):
            name = "QUANT_X(act)"
        elif c.op == C.OP_GEMM and (c.K == cp.HD or "pv" in c.note):
            name = "GEMM.attn"
        by[name][0] += 1
        by[name][1] += clk
        by[name][2] += byts
        tot += clk
    ms = 1e3 / (mhz * 1e6)
    return {
        "records": len(cmds),
        "F_s": tot * ms / 1e3,
        "clocks": tot,
        "ddr_GB": sum(v[2] for v in by.values()) / 1e9,
        "by_op": {k: {"n": v[0], "ms": v[1] * ms} for k, v in sorted(by.items(), key=lambda x: -x[1][1])},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, nargs="*", default=[250.0, 214.29, 200.0, 187.5])
    ap.add_argument("--ddr", type=float, default=14.0)
    ap.add_argument("--arm-ms", type=float, default=30.0)
    ap.add_argument("--link-ms", type=float, default=30.0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    P = cp.build(0x8_0000_0000)
    base = P.cmds
    assert not contract.validate_list(base), "the unfused list must stay valid under the patched contract"
    fs, pairs = fuse(base, amax=False)
    fa, _ = fuse(base, amax=True)
    for nm, L in (("standalone", fs), ("+emit_amax", fa)):
        bad = contract.validate_list(L)
        assert not bad, (nm, [(i, L[i].note, C.menames[r]) for i, r in bad[:5]])
    kinds = defaultdict(int)
    for p in pairs:
        kinds[(p["epi"], p["M"], p["K"], p["N"])] += 1
    print(f"unfused list {len(base)} records; fused pairs {len(pairs)}; standalone {len(fs)} records, "
          f"+emit_amax {len(fa)} records; all valid (patched contract.validate)")
    epn = {C.EPI_GELU: "GELU", C.EPI_SILU_MUL: "SILU_MUL", C.EPI_SILU: "SILU"}
    for (e, M, K, N), n in sorted(kinds.items()):
        print(f"  {n:4d} x {epn[e]:8s} GEMM M={M} K={K} N={N}")
    res = []
    for mhz in a.mhz:
        r0, r1, r2 = (report(L, mhz, a.ddr) for L in (base, fs, fa))
        extra = (a.arm_ms + a.link_ms) / 1e3
        row = {
            "mhz": mhz,
            "unfused": r0,
            "standalone": r1,
            "with_emit_amax": r2,
            "saved_ms_standalone": (r0["F_s"] - r1["F_s"]) * 1e3,
            "saved_ms_with_emit_amax": (r0["F_s"] - r2["F_s"]) * 1e3,
            "saved_clocks_standalone": r0["clocks"] - r1["clocks"],
            "saved_clocks_with_emit_amax": r0["clocks"] - r2["clocks"],
            "T_unfused_s": r0["F_s"] + extra,
            "T_standalone_s": r1["F_s"] + extra,
            "T_with_emit_amax_s": r2["F_s"] + extra,
        }
        res.append(row)
        print(f"\n@ {mhz:.2f} MHz, DDR {a.ddr:.0f} GB/s (T = F + {a.arm_ms + a.link_ms:.0f} ms ARM + link, est.)")
        print(f"  F unfused {r0['F_s']:.4f} s | standalone {r1['F_s']:.4f} s (-{row['saved_ms_standalone']:.2f} ms, "
              f"-{row['saved_clocks_standalone'] / 1e6:.3f} M clocks) | +emit_amax {r2['F_s']:.4f} s "
              f"(-{row['saved_ms_with_emit_amax']:.2f} ms, -{row['saved_clocks_with_emit_amax'] / 1e6:.3f} M clocks)")
        print(f"  DDR GB/chunk {r0['ddr_GB']:.3f} | {r1['ddr_GB']:.3f} | {r2['ddr_GB']:.3f}")
        names = sorted(set(r0["by_op"]) | set(r1["by_op"]) | set(r2["by_op"]), key=lambda k: -r0["by_op"].get(k, {"ms": 0})["ms"])
        print("  | mode | unfused n / ms | standalone n / ms | +emit_amax n / ms |")
        for k in names:
            cells = [f"{r['by_op'][k]['n']} / {r['by_op'][k]['ms']:.2f}" if k in r["by_op"] else "-" for r in (r0, r1, r2)]
            print(f"  | {k} | " + " | ".join(cells) + " |")
    if a.json:
        a.json.write_text(json.dumps({"pairs": pairs, "results": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
