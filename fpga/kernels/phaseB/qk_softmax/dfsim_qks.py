"""Clock-level token model of the fused item's DATAFLOW region (gemm_qks.h + qks_tail.h) with every FIFO bounded at
its pragma depth: a deadlock check of the whole process network and a cycle count per item, to check the closed
form qks_item_clocks (integration contract_plan.py) that the per-chunk savings use. Estimates: the processes'
pipeline depths, the m_axi latency and the weight stream are not modelled (weights are assumed to be ready; for
K = 64 the engine takes 64 weight words per group and tile, the HP ports give one per clock).

Per clock each process makes at most the moves its HLS loop makes in one II = 1 iteration; a FIFO write is visible
the next clock and a write needs room at the start of the clock (conservative).

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\phaseB\\qk_softmax\\dfsim_qks.py [--json results/dfsim.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
TB, NBX, NTT, EROWS, DQS, SETTLE, EDONE, XSETTLE, SB_SETTLE = 32, 2, 64, 64, 4, 8, 6, 4, 8
DSTEPS = 13


class Fifo:
    def __init__(self, name, cap):
        self.name, self.cap, self.n, self.pw, self.pr, self.moves, self.peak = name, cap, 0, 0, 0, 0, 0

    def can_w(self):
        return self.n + self.pw < self.cap

    def w(self):
        assert self.can_w()
        self.pw += 1

    def can_r(self):
        return self.n - self.pr > 0

    def r(self):
        assert self.can_r()
        self.pr += 1

    def commit(self):
        self.moves += self.pw + self.pr
        self.n += self.pw - self.pr
        self.peak = max(self.peak, self.n)
        self.pw = self.pr = 0


def tiles_of(n):
    nblk = math.ceil(n / TB)
    return [min(NBX, nblk - t * NBX) for t in range(math.ceil(nblk / NBX))]


def item(L, K, N, mask, depths, NP=None, fused=True):
    """One QK_SOFTMAX item (fused=False: the GEMM item of the unfused pair, Y to the store): returns (clocks, peaks)
    or raises on deadlock."""
    NP = NP or N
    G, W, W2 = L // 32, L // 4, L // 8
    tl = tiles_of(N)
    nv = [min(NTT, N - NTT * t) for t in range(len(tl))]
    nch = math.ceil(K / 64)
    F = {k: Fifo(k, v) for k, v in depths.items()}
    xr, xs, stok, srow, rs, es, ss = (F[k] for k in ("xr", "xs", "stok", "srow", "rs", "es", "ss"))
    sxs, smx, eq, sq, rq = (F[k] for k in ("sxs", "smx", "eq", "sq", "rq"))
    ys = Fifo("ys", 32)
    F["ys"] = ys
    out_f = ss if fused else ys
    if not fused:
        mask = False

    # ---- load_io: its program as (fifo, words) segments, one word per clock
    def units(t):
        return tl[t] * nch * TB if t < len(tl) else 0

    prog = [(stok, 16), (xr, units(0) * 8), (srow, G * 8)]
    if mask:
        prog += [(rs, G * 8), (rs, 16)]
    for t in range(len(tl)):
        if t > 0:
            prog.append((stok, 16))
            if mask:
                prog.append((rs, 16))
        un = units(t + 1)
        for g in range(G):  # the credit walk: units u with u * G < (g + 1) * un
            a, b = math.ceil(g * un / G), math.ceil((g + 1) * un / G)
            prog.append((xr, (b - a) * 8))
    prog = [p for p in prog if p[1]]

    def load_io():
        for f, n in prog:
            for _ in range(n):
                while not f.can_w():
                    yield
                f.w()
                yield

    # ---- xpose: halves of 32 tokens x 8 words in, kc x 4 words out (kc = 64 or the last chunk's K)
    def xpose():
        halves = len([0 for t in range(len(tl)) for _ in range(tl[t] * nch)])
        kcs = [(K - 64 * (nch - 1)) if (h % nch) == nch - 1 else 64 for h in range(halves)]
        wdone = rdone = 0
        wcnt = rcnt = 0
        wseen = deque([False] * XSETTLE)
        ravail = 0
        wbusy = 0
        rseen = deque([False] * XSETTLE)
        while rdone < halves:
            wfin = rfin = False
            if ravail and xs.can_w():
                xs.w()
                rcnt += 1
                if rcnt == kcs[rdone] * 4:
                    rcnt, rfin = 0, True
            if wdone < halves and wbusy < 2 and xr.can_r():
                xr.r()
                wcnt += 1
                if wcnt == 256:
                    wcnt, wfin = 0, True
            if wfin:
                wdone += 1
            if rfin:
                rdone += 1
            we, re_ = wseen.popleft(), rseen.popleft()
            wseen.append(wfin)
            rseen.append(rfin)
            wbusy += (1 if wfin else 0) - (1 if re_ else 0)
            ravail += (1 if we else 0) - (1 if rfin else 0)
            yield

    # ---- compute16: prefill tile 0, then MACs (K clocks per block, >= 8), capture, 64-word emit; fill tile t+1
    def compute16():
        words = [tl[t] * K * 4 for t in range(len(tl))]
        for _ in range(words[0]):
            while not xs.can_r():
                yield
            xs.r()
            yield
        blocks = [(t, g, b) for t in range(len(tl)) for g in range(G) for b in range(tl[t])]
        klen = max(K, 8)
        filled_at = {0: 0}
        clk = 0
        bi, k = 0, 0
        ft, fcnt = 1, 0
        sbusy, cap_at, emit_left, sfree_at = False, None, 0, None
        while bi < len(blocks) or sbusy:
            clk += 1
            # fill side: tile ft while tile ft - 1 computes
            cur_t = blocks[bi][0] if bi < len(blocks) else len(tl)
            if ft < len(tl) and ft <= cur_t + 1 and xs.can_r():
                xs.r()
                fcnt += 1
                if fcnt == words[ft]:
                    filled_at[ft] = clk
                    ft, fcnt = ft + 1, 0
            # emit side
            if cap_at is not None and clk >= cap_at and emit_left and es.can_w():
                es.w()
                emit_left -= 1
                if emit_left == 0:
                    sfree_at = clk + EDONE
            if sbusy and emit_left == 0 and sfree_at is not None and clk >= sfree_at:
                sbusy, cap_at, sfree_at = False, None, None
            # MAC side
            if bi < len(blocks):
                t = blocks[bi][0]
                ready = t in filled_at and clk >= filled_at[t] + SETTLE
                last = k == klen - 1
                if ready and not (last and sbusy):
                    if last:
                        sbusy, cap_at, emit_left = True, clk + 2, EROWS
                        bi, k = bi + 1, 0
                    else:
                        k += 1
            yield

    # ---- dequant_ep: s_row once, per tile s_tok, then 4 clocks per emit word, valid tokens write ss
    def dequant():
        for _ in range(G * 8):
            while not srow.can_r():
                yield
            srow.r()
            yield
        for t in range(len(tl)):
            for _ in range(16):
                while not stok.can_r():
                    yield
                stok.r()
                yield
            for g in range(G):
                for b in range(tl[t]):
                    for r in range(EROWS):
                        tok = b * TB + (r >> 1)
                        valid = t * NTT + tok < N
                        for q in range(DQS):
                            while (q == 0 and not es.can_r()) or (valid and not out_f.can_w()):
                                yield
                            if q == 0:
                                es.r()
                            if valid:
                                out_f.w()
                            yield

    # ---- sbuf
    def sbuf():
        if mask:
            for _ in range(W):
                while not rs.can_r():
                    yield
                rs.r()
                yield
        fill = [(t, g, row) for t in range(len(tl)) for g in range(G) for row in range(nv[t])]
        fi, fw, frq = 0, 0, 16 if mask else 0
        ffull = davail = 0
        dt, drow, dw = 0, 0, 0
        fl, dl = deque([False] * SB_SETTLE), deque([False] * SB_SETTLE)
        ftile = 0
        while dt < len(tl):
            ffin = dfin = False
            stall = False
            # drain (blocking writes stall the whole loop)
            if davail:
                f = smx if dw == 0 else sxs
                if not f.can_w():
                    stall = True
            if not stall:
                if ftile < len(tl) and ffull < 2:
                    if frq:
                        if rs.can_r():
                            rs.r()
                            frq -= 1
                    elif ss.can_r():
                        ss.r()
                        fw += 1
                        if fw == 8:
                            fw = 0
                            fi += 1
                            if fi == len(fill) or fill[fi][0] != ftile:
                                ffin = True
                                ftile += 1
                                frq = 16 if mask else 0
                if davail:
                    (smx if dw == 0 else sxs).w()
                    if dw == W:
                        dw = 0
                        drow += 1
                        if drow == nv[dt]:
                            drow, dfin = 0, True
                            dt += 1
                    else:
                        dw += 1
                fe, de = fl.popleft(), dl.popleft()
                fl.append(ffin)
                dl.append(dfin)
                ffull += (1 if ffin else 0) - (1 if de else 0)
                davail += (1 if fe else 0) - (1 if dfin else 0)
            yield

    # ---- smx::compute: per real row 1 setup clock (smx) + W words (sxs); eq every 2 words; sq at the row end
    def compute():
        for n in range(NP):
            real = n < N
            for w in range(W + 1):
                while True:
                    need_r = real and (smx if w == 0 else sxs)
                    ok = (not need_r or need_r.can_r()) and (w == 0 or w % 2 or eq.can_w()) and (w < W or sq.can_w())
                    if ok:
                        break
                    yield
                if need_r:
                    need_r.r()
                if w and w % 2 == 0:
                    eq.w()
                if w == W:
                    sq.w()
                yield
        for _ in range(-NP % 4):
            while not sq.can_w():
                yield
            sq.w()
            yield

    # ---- divider: 14 clocks per row, a scale word every 4 rows
    def divider():
        rows = NP + (-NP % 4)
        for i in range(rows):
            while not sq.can_r():
                yield
            sq.r()
            yield
            for s in range(DSTEPS):
                if s == DSTEPS - 1 and i % 4 == 3:
                    while not rq.can_w():
                        yield
                    rq.w()
                yield

    # ---- store: per block of 256 rows the P words then the block's scale words
    def store():
        for r0 in range(0, NP, 256):
            rows = min(256, NP - r0)
            for f, n in ((eq, rows * W2), (rq, (rows + 3) // 4)):
                for _ in range(n):
                    while not f.can_r():
                        yield
                    f.r()
                    yield

    def store_y():  # the GEMM's row-major Y: 8 words per (token, group), one per clock
        for _ in range(N * G * 8):
            while not ys.can_r():
                yield
            ys.r()
            yield

    procs = {"load_io": load_io(), "xpose": xpose(), "compute16": compute16(), "dequant": dequant()}
    if fused:
        procs |= {"sbuf": sbuf(), "compute": compute(), "divider": divider(), "store": store()}
    else:
        procs["store"] = store_y()
    live = dict(procs)
    clk, idle = 0, 0
    while live:
        clk += 1
        for name in list(live):
            try:
                next(live[name])
            except StopIteration:
                del live[name]
        before = sum(f.moves for f in F.values())
        for f in F.values():
            f.commit()
        idle = 0 if sum(f.moves for f in F.values()) != before else idle + 1
        if idle > 20000:
            occ = {f.name: f.n for f in F.values()}
            raise RuntimeError(f"DEADLOCK at clock {clk}: blocked {sorted(live)}; occupancy {occ}")
    left = {f.name: f.n for f in F.values() if f.n}
    assert not left, f"FIFOs left non-empty: {left}"
    return clk, {f.name: f.peak for f in F.values()}


DEPTHS = {"xr": 64, "xs": 32, "stok": 32, "srow": 64, "rs": 32, "es": 4, "ss": 32,
          "sxs": 64, "smx": 4, "eq": 16, "sq": 4, "rq": 64}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "results" / "dfsim.json"))
    a = ap.parse_args()
    import sys

    sys.path.insert(0, str(HERE / "integration" / "kernels"))
    import contract_plan as CP

    CP.OPT["item_fixed"] = None
    shapes = {"vision": (256, 64, 256, False), "lm": (64, 64, 144, True), "expert_self": (128, 64, 156, True),
              "expert_cross": (64, 64, 156, True), "edge_n1": (32, 64, 1, True), "edge_n65_L256": (256, 64, 65, True),
              "edge_n300_L32": (32, 64, 300, False), "edge_k8": (64, 8, 100, True), "edge_k136": (96, 136, 130, True)}
    out = {}
    print("| shape | L x K x N | QK_SOFTMAX model | closed form | ratio | its GEMM item alone: model | closed form | "
          "ratio | peak FIFO words (fused) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (L, K, N, mask) in shapes.items():
        clk, peaks = item(L, K, N, mask, DEPTHS)
        tot, last = CP.qks_item_clocks(L, K, N, 14e9 / 250e6, True)
        closed = tot + last - (CP.L0 + CP.TAIL)  # the model's pipeline part (no per-item constants)
        gclk, _ = item(L, K, N, mask, DEPTHS, fused=False)
        gclosed = CP.gemm_item_clocks(L, K, N, 14e9 / 250e6, False, True) - (CP.L0 + CP.TAIL)
        out[name] = {"L": L, "K": K, "N": N, "mask": mask, "dfsim_clocks": clk, "closed_form_clocks": closed,
                     "peaks": peaks, "gemm_dfsim_clocks": gclk, "gemm_closed_form_clocks": gclosed}
        pk = ", ".join(f"{k} {v}" for k, v in peaks.items() if k in ("rs", "ss", "sxs", "eq", "rq", "es"))
        print(f"| {name} | {L} x {K} x {N} | {clk} | {closed:.0f} | {clk / closed:.3f} | {gclk} | {gclosed:.0f} | "
              f"{gclk / gclosed:.3f} | {pk} |")
    # adversarial depths of the fusion's own FIFOs (the engine's keep their pragma depths: its s_tok FIFO must hold
    # one tile's 16 words, since load_io sends s_tok(0) before the s_row that dequant_ep reads first)
    small = DEPTHS | {"rs": 1, "ss": 2, "sxs": 2, "smx": 2, "eq": 2, "sq": 2, "rq": 64}
    try:
        item(256, 64, 256, False, DEPTHS | {"stok": 8})
        out["engine_stok8"] = "no deadlock"
    except RuntimeError:
        out["engine_stok8"] = "deadlock: the engine's own s_tok FIFO needs >= 16 words (its pragma depth is 32)"
    print(f"\n(engine s_tok FIFO at 8 words: {out['engine_stok8']})")
    print("the fusion's FIFOs at depth 1-2 (rs 1; ss, sxs, smx, eq, sq 2; rq 64 = one block of scales):")
    for name in ("vision", "lm", "expert_self", "edge_n300_L32"):
        L, K, N, mask = shapes[name]
        clk, _ = item(L, K, N, mask, small)
        out[name]["small_fifo_clocks"] = clk
        print(f"  {name}: no deadlock, {clk} clocks")
    # negative control: the divider's scale FIFO below one block of scales (as the SOFTMAX unit's own control)
    try:
        item(256, 64, 256, False, DEPTHS | {"rq": 56})
        print("negative control (rq 56 < 64 words, 256-row block): NOT DETECTED")
        out["negative_control_rq56"] = "not detected"
    except RuntimeError as e:
        print(f"negative control (rq 56 < 64 words, 256-row block): deadlock detected as required: {str(e)[:120]}")
        out["negative_control_rq56"] = "deadlock detected"
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
