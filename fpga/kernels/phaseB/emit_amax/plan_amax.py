"""emit_amax on the -12392 chunk: the contract's command list (contract_plan.py) with every QUANT_X whose input is a
GEMM output fused away, validated, and timed with the contract's own model (+ the fusion's terms).

Fused (producer GEMM gets P_AMX_SG + slot H; consumer GEMM gets P_XQ + F = the absmax parts, A = the FP32 rows):
  vision q    qkv (36 head segments)            -> Q.K^T, split into one command per image (12 items), p4 = -3
  o.in        P.V (vision 2 x 12 items; LM / expert 5 items x 3 row groups; expert self: pv1 with F_ACC)
              -> o_proj over 12 / 15 parts
  x_t         action_out + Euler (F_ACC) of the step before (step 0: the host writes the noise's absmax) -> a_in
  a_emb       a_in -> t_in (Mp 736, K 720: the pad columns are exact zeros)
  v.in        LM odd layer qkv (5 segments of 320) -> the cross v_proj over part 4
Not fused: conn.in (COPY producer), cross k.in (ROPE in-place producer).

    python plan_amax.py [--mhz 214.2857 250 187.5 200] [--ql 4] [--json out.json] [--out fused_list.bin]
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
sys.path.insert(0, str(HERE.parents[1]))  # fpga/kernels
import contract_plan as CP  # noqa: E402
from contract import C, Cmd, Slot, encode_list, pad4, validate  # noqa: E402

P_SHIFT, P_SEGC = C.P_SHIFT, C.P_SEGC
P_AMX_SG, P_XQ = 7, 8  # patched/kernels/contract.h (INTEGRATION.md)
AMX_NS_MAX, AMX_PARTS_MAX = 64, 64
QL = 4
XQ_SCALE_LAT = 40  # clocks: divider pipeline + loop entry, per tile (est. from the csynth schedule depth)


# ---------------------------------------------------------------- validate (patched contract.h, mirrored)
def validate_amax(c: Cmd, full: bool = True) -> int:
    if c.op != C.OP_GEMM:
        return validate(c, full)
    p = {i: c.par(i) for i in range(C.NPARAMS)}
    sg, xq, sh = p[P_AMX_SG], p[P_XQ], p[P_SHIFT]
    sh = sh - (1 << 32) if sh & 0x80000000 else sh
    c0 = copy.copy(c)
    c0.p = {i: v for i, v in c.p.items() if i not in (P_SHIFT, P_AMX_SG, P_XQ)}
    c0.s = {k: v for k, v in c.s.items() if k != "H"}
    r = validate(c0, full)
    if r:
        return r
    M, N, K, B = c.M, c.N, c.K, c.B
    G, R = M // 32, p[P_SEGC]
    if sh < -C.SHIFT_MAX or sh > C.SHIFT_MAX or (xq == 0 and sh != 0):
        return C.ME_PARAM
    if xq > AMX_PARTS_MAX or sg > G or (sg and R and (sg != G or R % 4)):
        return C.ME_PARAM
    if full and sg and (G % sg or G // sg > AMX_NS_MAX):
        return C.ME_PARAM
    A = c.slot("A")
    if xq:
        if A.addr % 16 or A.pitch % 16 or A.stride % 16 or (N > 1 and A.pitch < 4 * K):
            return C.ME_ALIGN
        if full and A.addr + (B - 1) * A.stride + (N - 1) * A.pitch + 4 * K > C.ADDR_LIMIT:
            return C.ME_ADDR
        F = c.slot("F")
        if F.addr % 16 or F.pitch % 16 or F.stride % 16 or (xq > 1 and F.pitch < 4 * pad4(N)):
            return C.ME_ALIGN
        if full and F.addr + (B - 1) * F.stride + (xq - 1) * F.pitch + 4 * pad4(N) > C.ADDR_LIMIT:
            return C.ME_ADDR
    H = c.slot("H")
    if sg:
        rows = min(N, R) if R else N
        multi = N > R if R else G > sg
        if H.addr % 16 or H.pitch % 16 or H.stride % 16 or (multi and H.pitch < 4 * pad4(rows)):
            return C.ME_ALIGN
        parts = (N - 1) // R + 1 if R else G // sg
        if full and H.addr + (B - 1) * H.stride + (parts - 1) * H.pitch + 4 * pad4(rows) > C.ADDR_LIMIT:
            return C.ME_ADDR
    elif not H.zero():
        return C.ME_SLOT
    return C.ME_OK


# ---------------------------------------------------------------- the fusion of the plan
def fuse(P: CP.Plan) -> tuple[list[Cmd], list[dict]]:
    cmds = P.cmds
    mem = P.mem
    out: list[Cmd] = []
    log: list[dict] = []
    drop: set[int] = set()
    repl: dict[int, list[Cmd]] = {}  # index -> commands replacing it
    mod: dict[int, Cmd] = {}

    def get(i: int) -> Cmd:
        if i not in mod:
            mod[i] = copy.deepcopy(cmds[i])
        return mod[i]

    def prev_gemm(i: int, pred) -> int:
        for j in range(i - 1, -1, -1):
            if cmds[j].op == C.OP_GEMM and pred(cmds[j]):
                return j
        return -1

    def next_gemm(i: int, xaddr: int) -> int:
        for j in range(i + 1, len(cmds)):
            if cmds[j].op == C.OP_GEMM and cmds[j].slot("A").addr == xaddr:
                return j
        raise RuntimeError(f"no consumer for {cmds[i].note}")

    def set_amax(j: int, sg: int, H: Slot) -> None:
        c = get(j)
        assert c.par(P_AMX_SG) in (0, sg), c.note
        c.p[P_AMX_SG] = sg
        c.s["H"] = H

    for i, q in enumerate(cmds):
        if q.op != C.OP_QUANT_X:
            continue
        ph, nm = q.note.split(":", 1)
        A, Cq, Fq = q.slot("A"), q.slot("C"), q.slot("F")
        sh = q.par(P_SHIFT)
        sh = sh - (1 << 32) if sh & 0x80000000 else sh
        rec = dict(index=i, note=q.note, K=q.K, N=q.N)
        if nm == "q" and ph == "vision":  # per head, p4 = -3
            jp = prev_gemm(i, lambda c: c.note.endswith(":qkv"))
            pc = cmds[jp]
            G, n = pc.M // 32, pc.N
            Hp = 4 * pad4(n)
            Ha = mem.alloc((G // 2) * Hp, 64)
            set_amax(jp, 2, Slot(Ha, Hp, 0))
            jc = next_gemm(i, Cq.addr)
            cc = cmds[jc]
            ws, ss, ys = cc.slot("B").stride, cc.slot("E").stride, cc.slot("C").stride
            new = []
            for img in range(CP.IMGS):  # item b = 2 head + img -> per image: items = heads
                c2 = copy.deepcopy(cc)
                c2.B = CP.V_H
                c2.s["A"] = Slot(A.addr + img * CP.V_T * A.pitch, A.pitch, 4 * CP.HD)
                c2.s["B"] = Slot(cc.slot("B").addr + img * ws, cc.slot("B").pitch, 2 * ws)
                c2.s["E"] = Slot(cc.slot("E").addr + img * ss, 0, 2 * ss)
                c2.s["C"] = Slot(cc.slot("C").addr + img * ys, cc.slot("C").pitch, 2 * ys)
                c2.s["F"] = Slot(Ha + 4 * img * CP.V_T, 0, Hp)
                c2.p[P_XQ] = 1
                c2.p[P_SHIFT] = sh & 0xFFFFFFFF
                c2.note = cc.note + f".img{img}"
                new.append(c2)
            repl[jc] = new
            drop.add(i)
            rec.update(producers=[cmds[jp].note], consumer=cc.note, parts=1, split=2, sg=2)
        elif nm == "o.in":
            # the P.V commands since the softmax
            js = []
            for j in range(i - 1, -1, -1):
                if cmds[j].op == C.OP_SOFTMAX:
                    break
                if cmds[j].op == C.OP_GEMM and cmds[j].note.split(":")[1].startswith("pv"):
                    js.append(j)
            js.sort()
            jc = next_gemm(i, Cq.addr)
            cons = get(jc)
            if ph == "vision":  # 2 commands (images), 12 items (heads) each, R = 0
                n_all = q.N
                Hp = 4 * pad4(n_all)
                Ha = mem.alloc(CP.V_H * Hp, 64)
                for img, j in enumerate(js):
                    set_amax(j, 2, Slot(Ha + 4 * img * CP.V_T, 0, Hp))
                parts, F = CP.V_H, Slot(Ha, Hp, 0)
                carriers = js
            else:  # 5 items x 3 row groups of R rows; the self-attention's pv1 (ACC) carries it
                last = js[-1]
                pc = cmds[last]
                R = pc.par(P_SEGC)
                Hp = 4 * pad4(R)
                ng = (pc.N - 1) // R + 1
                Ha = mem.alloc(pc.B * ng * Hp, 64)
                set_amax(last, pc.M // 32, Slot(Ha, Hp, ng * Hp))
                parts, F = pc.B * ng, Slot(Ha, Hp, 0)
                carriers = [last]
            cons.s["A"] = A
            cons.s["F"] = F
            cons.p[P_XQ] = parts
            drop.add(i)
            rec.update(producers=[cmds[j].note for j in carriers], consumer=cons.note, parts=parts)
        elif nm == "x_t":
            jp = prev_gemm(i, lambda c: c.note.endswith("a_out+euler"))
            Hp = 4 * pad4(q.N)
            Ha = mem.alloc(Hp, 64)
            if jp >= 0:
                set_amax(jp, 1, Slot(Ha, 0, 0))
            jc = next_gemm(i, Cq.addr)
            cons = get(jc)
            cons.s["A"], cons.s["F"] = A, Slot(Ha, 0, 0)
            cons.p[P_XQ] = 1
            drop.add(i)
            rec.update(producers=[cmds[jp].note] if jp >= 0 else ["host (noise absmax)"], consumer=cons.note, parts=1)
        elif nm == "a_emb":
            jp = prev_gemm(i, lambda c: c.note.endswith(":a_in"))
            pc = cmds[jp]
            Hp = 4 * pad4(q.N)
            Ha = mem.alloc(Hp, 64)
            set_amax(jp, pc.M // 32, Slot(Ha, 0, 0))
            jc = next_gemm(i, Cq.addr)
            cons = get(jc)
            cons.s["A"], cons.s["F"] = A, Slot(Ha, 0, 0)
            cons.p[P_XQ] = 1
            drop.add(i)
            rec.update(producers=[pc.note], consumer=cons.note, parts=1)
        elif nm == "v.in":
            col = 4 * 1280
            jp = prev_gemm(i, lambda c, a=A.addr - col: c.slot("C").addr == a)
            pc = cmds[jp]
            G = pc.M // 32
            sg = 10
            Hp = 4 * pad4(pc.N)
            Ha = mem.alloc((G // sg) * Hp, 64)
            set_amax(jp, sg, Slot(Ha, Hp, 0))
            jc = next_gemm(i, Cq.addr)
            cons = get(jc)
            cons.s["A"], cons.s["F"] = A, Slot(Ha + 4 * Hp, 0, 0)
            cons.p[P_XQ] = 1
            drop.add(i)
            rec.update(producers=[pc.note], consumer=cons.note, parts=1, sg=sg)
        else:
            rec.update(fused=False, why="producer is not a GEMM (COPY / ROPE in place)")
        log.append(rec)
    for i, c in enumerate(cmds):
        if i in drop:
            continue
        if i in repl:
            out.extend(repl[i])
        else:
            out.append(mod.get(i, c))
    return out, log


# ---------------------------------------------------------------- timing (contract_plan.py's model + the fusion)
def gemm_item_clocks(c: Cmd, ddr_bpc: float, d_read: bool, first: bool) -> float:
    mp, k, n = c.M, c.K, c.N
    nbx = CP.OPT["nbx"]
    g, nblk = mp // 32, math.ceil(n / 32)
    tiles = [min(nbx, nblk - t * nbx) for t in range(math.ceil(nblk / nbx))]
    xq, sg = c.par(P_XQ), c.par(P_AMX_SG)
    R = c.par(P_SEGC)
    ns = (g // sg) if sg else 0
    nch = math.ceil(k / 64)

    def xload(t: int) -> float:  # load_io clocks for X of tile t (+ its scales)
        nbt = tiles[t]
        if not xq:
            return 4 * nbt * k
        tok0, ntok = t * nbx * 32, nbt * 32
        live = min(max(n - tok0, 0), ntok)
        return (live * 16 * (4 // QL) + (ntok - live) * 8) * nch + xq * 16 + 64 + XQ_SCALE_LAT

    fixed = CP.L0 + CP.TAIL if first or CP.OPT["item_fixed"] is None else CP.OPT["item_fixed"]
    total = fixed + xload(0)
    for t, nbt in enumerate(tiles):
        nxt = tiles[t + 1] if t + 1 < len(tiles) else 0
        nv = min(n - t * nbx * 32, nbt * 32)
        dw = CP.YW_BLK * g * nbt if d_read else 0
        aw = ns * math.ceil(nv / 4)  # absmax words written by the store (one clock each)
        xb = (128 if xq else 64) * nxt * k
        total += max(
            g * nbt * max(k, CP.EMIT),
            g * nbt * CP.YW_BLK + aw,
            g * k,
            (64 * g * k + 4096 * g * nbt + 16 * dw + xb + 16 * aw) / ddr_bpc,
            (xload(t + 1) if nxt else 0) + dw,
        )
    return total


def cmd_clocks(c: Cmd, ddr_bpc: float) -> tuple[float, int]:
    if c.op != C.OP_GEMM:
        return CP.cmd_clocks(c, ddr_bpc)
    d = bool(c.flags & (C.F_ADD_D | C.F_ACC)) and c.slot("D").pitch != 0 or bool(c.flags & C.F_ACC)
    clk = sum(gemm_item_clocks(c, ddr_bpc, d, b == 0) for b in range(min(c.B, 2)))
    clk += max(c.B - 2, 0) * gemm_item_clocks(c, ddr_bpc, d, False)
    xq, sg = c.par(P_XQ), c.par(P_AMX_SG)
    xbytes = 4 * c.N * c.K if xq else 2 * c.N * c.K
    abytes = 4 * xq * pad4(c.N)  # absmax parts read
    if sg:
        R = c.par(P_SEGC)
        parts = (c.N - 1) // R + 1 if R else (c.M // 32) // sg
        rows = min(c.N, R) if R else c.N
        abytes += 4 * parts * pad4(rows)  # absmax parts written
    byts = c.B * (2 * c.M * c.K + xbytes + abytes + 4 * c.N * c.M * (2 if d else 1))
    return CP.SEQ_FETCH + clk, byts


def report(cmds: list[Cmd], mhz: float, ddr_gbs: float, clocks) -> dict:
    ddr_bpc = ddr_gbs * 1e9 / (mhz * 1e6)
    by_op = defaultdict(lambda: [0, 0.0, 0])
    tot, byts = 0.0, 0
    for c in cmds:
        clk, b = clocks(c, ddr_bpc)
        name = C.opnames[c.op] + (".attn" if c.op == C.OP_GEMM and (c.K == CP.HD or "pv" in c.note) else "")
        by_op[name][0] += 1
        by_op[name][1] += clk
        by_op[name][2] += b
        tot += clk
        byts += b
    ms = 1e3 / (mhz * 1e6)
    F = tot * ms / 1e3
    return dict(
        mhz=mhz,
        records=len(cmds),
        clocks=tot,
        F_s=F,
        ddr_GB=byts / 1e9,
        by_op={k: dict(cmds=v[0], ms=v[1] * ms, GB=v[2] / 1e9) for k, v in by_op.items()},
    )


def site(note: str) -> str:
    ph, nm = note.split(":", 1)
    return {"q": "vision Q", "o.in": f"{ph} o.in", "x_t": "x_t", "a_emb": "a_emb", "v.in": "cross v.in"}.get(nm, nm)


def main() -> int:
    global QL
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, nargs="*", default=[300 * 5 / 7, 250.0, 187.5, 200.0])
    ap.add_argument("--ddr", type=float, default=14.0)
    ap.add_argument("--ql", type=int, default=4, help="quantizer lanes of the X loader (AMX_QL)")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x8_0000_0000)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--out", type=Path, help="write the fused list (for tb_amax --list)")
    ap.add_argument("--arm-ms", type=float, default=30.0)
    ap.add_argument("--link-ms", type=float, default=30.0)
    a = ap.parse_args()
    QL = a.ql
    P = CP.build(a.base)
    base_cmds = P.cmds
    fused, log = fuse(P)
    bad = [(i, validate_amax(c)) for i, c in enumerate(fused) if validate_amax(c)]
    bad_hw = [(i, validate_amax(c, False)) for i, c in enumerate(fused) if validate_amax(c, False)]
    for i, r in bad[:10]:
        print(f"INVALID {i} {fused[i].note}: ME_{C.menames[r]}")
    nf = sum(1 for r in log if r.get("fused", True))
    print(
        f"-12392 list: {len(base_cmds)} records -> {len(fused)} ({nf} QUANT_X fused away of "
        f"{sum(1 for c in base_cmds if c.op == C.OP_QUANT_X)}; vision Q.K^T split per image: "
        f"+{sum(r.get('split', 1) - 1 for r in log)}); {len(bad)} invalid (validate), {len(bad_hw)} (validate_hw)"
    )
    res = dict(ql=QL, ddr_GBs=a.ddr, fused_sites=log, runs=[])
    for mhz in a.mhz:
        r0 = report(base_cmds, mhz, a.ddr, CP.cmd_clocks)
        r0m = report(base_cmds, mhz, a.ddr, cmd_clocks)  # this script's model on the unfused list (must match)
        r1 = report(fused, mhz, a.ddr, cmd_clocks)
        assert abs(r0["clocks"] - r0m["clocks"]) < 1e-6 * r0["clocks"], (r0["clocks"], r0m["clocks"])
        T0 = r0["F_s"] + (a.arm_ms + a.link_ms) / 1e3
        T1 = r1["F_s"] + (a.arm_ms + a.link_ms) / 1e3
        run = dict(
            mhz=mhz,
            before=r0,
            after=r1,
            saved_clocks=r0["clocks"] - r1["clocks"],
            saved_ms=(r0["F_s"] - r1["F_s"]) * 1e3,
            saved_GB=r0["ddr_GB"] - r1["ddr_GB"],
            T_before=T0,
            T_after=T1,
            margin1_before=2 * 0.8167 - 2 * T0,
            margin1_after=2 * 0.8167 - 2 * T1,
            margin2_before=1.6333 - (r0["F_s"] + T0),
            margin2_after=1.6333 - (r1["F_s"] + T1),
        )
        res["runs"].append(run)
        print(
            f"\n@ {mhz:.1f} MHz, DDR {a.ddr:.0f} GB/s, AMX_QL {QL}: F {r0['F_s']:.4f} -> {r1['F_s']:.4f} s "
            f"(-{run['saved_ms']:.2f} ms, -{run['saved_clocks'] / 1e6:.3f} M clocks); DDR {r0['ddr_GB']:.3f} -> "
            f"{r1['ddr_GB']:.3f} GB (-{run['saved_GB'] * 1e3:.1f} MB); T {T0:.3f} -> {T1:.3f} s; margin 1 in flight "
            f"{run['margin1_before']:+.3f} -> {run['margin1_after']:+.3f} s, 2 in flight {run['margin2_before']:+.3f} "
            f"-> {run['margin2_after']:+.3f} s"
        )
        print("| mode | records before | ms before | records after | ms after | GB before | GB after |")
        print("|---|---|---|---|---|---|---|")
        for k in sorted(set(r0["by_op"]) | set(r1["by_op"]), key=lambda k: -r0["by_op"].get(k, {"ms": 0})["ms"]):
            b0 = r0["by_op"].get(k, dict(cmds=0, ms=0, GB=0))
            b1 = r1["by_op"].get(k, dict(cmds=0, ms=0, GB=0))
            print(f"| {k} | {b0['cmds']} | {b0['ms']:.2f} | {b1['cmds']} | {b1['ms']:.2f} | {b0['GB']:.3f} | {b1['GB']:.3f} |")
    # per site at the first clock: the QUANT_X time removed, the consumers' and producers' extra time
    mhz = a.mhz[0]
    bpc = a.ddr * 1e9 / (mhz * 1e6)
    ms = 1e3 / (mhz * 1e6)
    per = defaultdict(lambda: dict(n=0, quant_ms=0.0, quant_MB=0.0))
    for r in log:
        if not r.get("fused", True):
            continue
        s = site(r["note"])
        clk, b = CP.cmd_clocks(base_cmds[r["index"]], bpc)
        per[s]["n"] += 1
        per[s]["quant_ms"] += clk * ms
        per[s]["quant_MB"] += b / 1e6
    # consumer / producer deltas by site, from the two lists (matched by note)
    def by_note(cmds, clocks):
        d = defaultdict(lambda: [0.0, 0])
        for c in cmds:
            clk, b = clocks(c, bpc)
            d[c.note.split(".img")[0]][0] += clk
            d[c.note.split(".img")[0]][1] += b
        return d

    d0, d1 = by_note(base_cmds, CP.cmd_clocks), by_note(fused, cmd_clocks)
    gemm_delta_ms = sum(d1[k][0] - d0[k][0] for k in d0 if k in d1) * ms
    # the GEMMs each site changes (by note: every layer's instance of it), counted once
    seen = set()
    for r in log:
        if not r.get("fused", True):
            continue
        s = site(r["note"])
        per[s].setdefault("gemm_ms", 0.0)
        per[s].setdefault("gemm_MB", 0.0)
        for nt in [r["consumer"]] + r["producers"]:
            if nt in seen or nt not in d0:
                continue
            seen.add(nt)
            per[s]["gemm_ms"] += (d1[nt][0] - d0[nt][0]) * ms
            per[s]["gemm_MB"] += (d1[nt][1] - d0[nt][1]) / 1e6
    print(f"\nper site @ {mhz:.1f} MHz, AMX_QL {QL} (all GEMMs together: {gemm_delta_ms:+.2f} ms)")
    print("| site | QUANT_X removed | their ms | their DDR MB | producer + consumer GEMMs: ms | MB | net ms saved | net MB saved |")
    print("|---|---|---|---|---|---|---|---|")
    for s, v in per.items():
        print(
            f"| {s} | {v['n']} | {v['quant_ms']:.2f} | {v['quant_MB']:.1f} | {v['gemm_ms']:+.2f} | {v['gemm_MB']:+.1f} | "
            f"{v['quant_ms'] - v['gemm_ms']:.2f} | {v['quant_MB'] - v['gemm_MB']:.1f} |"
        )
    res["per_site"] = per
    res["gemm_delta_ms"] = gemm_delta_ms
    if a.out:
        a.out.write_bytes(encode_list(fused))
        print(f"wrote {a.out} ({len(fused)} records)")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=float))
    return 1 if bad or bad_hw else 0


if __name__ == "__main__":
    raise SystemExit(main())
