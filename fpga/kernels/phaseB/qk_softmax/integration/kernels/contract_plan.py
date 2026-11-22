"""The -12392 per-chunk command list under the contract (CONTRACT.md 6) and its time per mode.

Builds every record of one chunk with real addresses (a bump allocator from --base, 64-bit), checks each one
with contract.validate (the C++ rules: tb_contract.cpp re-checks the written list), and times it with the
engine's cycle model (fpga/hls/model_int16.py rules, plus the D read of the epilogue) and the io-bound
row-unit / softmax rules of CONTRACT.md 6.2. All times are estimates: nothing here ran on the board.

    .venv/Scripts/python.exe fpga/kernels/contract_plan.py [--base 0x800000000] [--mhz 250] [--out list.bin]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from contract import C, Cmd, Slot, encode_list, f32bits, pad4, pad32, validate_list

# ---------------------------------------------------------------- -12392 shapes (config.json, cd_inventory)
V_D, V_L, V_H, V_FF, V_T, IMGS = 768, 12, 12, 3072, 256, 2
L_D, L_L, L_HQ, L_HKV, L_FF = 960, 16, 15, 5, 2560
E_D, E_DP, E_FF = 720, 736, 2048
HD, G = 64, 3  # head dim, query heads per KV head
TQ_P = 12 + 2 * 16 + 1  # prefix: 2 x 16 image tokens + 12 language tokens ("longest") + state = 45
TQ_E, A_D, STEPS = 50, 32, 10
EPS_LN, EPS_RMS = 1e-6, 1e-5
CONN_K, CONN_N = 16 * V_D, 2 * 16  # pixel shuffle x4: 12288 values, 16 rows per image


def p4(n: int) -> int:
    return pad4(n)


# ---------------------------------------------------------------- allocator
class Mem:
    def __init__(self, base: int):
        self.top, self.regions, self.cur = base, {}, None

    def region(self, name: str) -> None:
        self.top = (self.top + 0xFFFFF) & ~0xFFFFF  # regions on 1 MB
        self.regions[name] = [self.top, self.top]
        self.cur = name

    def alloc(self, nbytes: int, align: int = 64) -> int:
        a = (self.top + align - 1) // align * align
        self.top = a + nbytes
        self.regions[self.cur][1] = self.top
        return a


@dataclass
class Blob:  # one weight blob: B items, each Mp rows x K, port tiling + s_row (+ bias)
    w: int
    port: int  # bytes between port arrays
    item: int  # bytes between items inside a port array
    srow: int
    bias: int
    mp: int
    k: int


def blob(mem: Mem, mp: int, k: int, items: int = 1, bias: bool = False) -> Blob:
    item = (mp // 32) * k * 16
    port = items * item
    w = mem.alloc(4 * port, 4096)
    srow = mem.alloc(4 * mp * items, 64)
    b = mem.alloc(4 * mp * items, 64) if bias else 0
    return Blob(w, port, item, srow, b, mp, k)


# ---------------------------------------------------------------- command builders
class Plan:
    def __init__(self, base: int):
        self.mem = Mem(base)
        self.cmds: list[Cmd] = []
        self.phase = ""

    def add(self, c: Cmd, note: str) -> None:
        c.note = f"{self.phase}:{note}"
        c.tag = len(self.cmds)
        self.cmds.append(c)

    def gemm(
        self,
        note,
        w: Blob,
        x: int,
        xp: int,
        n: int,
        y: int,
        yp: int,
        stok: int,
        *,
        k=None,
        b=1,
        xs=0,
        ws=0,
        ss=0,
        fs=0,
        ys=0,
        bias=False,
        d=None,
        acc=False,
        R=0,
        P1=0,
        woff=0,
    ):
        k = k or w.k
        s = {
            "A": Slot(x, xp, xs),
            "B": Slot(w.w + woff, w.port, ws),
            "C": Slot(y, yp, ys),
            "E": Slot(w.srow, 0, ss),
            "F": Slot(stok, 0, fs),
        }
        fl = 0
        if bias:
            fl |= C.F_BIAS
            s["G"] = Slot(w.bias)
        if d is not None:
            fl |= C.F_ADD_D
            s["D"] = d
        if acc:
            fl |= C.F_ACC
        p = {C.P_SEGC: R, C.P_SEGF: P1} if R else {}
        self.add(Cmd(C.OP_GEMM, fl, 0, w.mp, k, n, b, s, p), note)

    def row(
        self,
        op,
        note,
        k,
        n,
        a: Slot,
        c: Slot,
        f: Slot | None = None,
        *,
        flags=C.F_QOUT,
        b=1,
        seg=0,
        segc=0,
        segf=0,
        shift=0,
        np_=0,
        e: Slot | None = None,
        bsl: Slot | None = None,
        eps=0.0,
    ):
        s = {"A": a, "C": c}
        if f is not None:
            s["F"] = f
        if e is not None:
            s["E"] = e
        if bsl is not None:
            s["B"] = bsl
        p = {C.P_SEG: seg, C.P_SEGC: segc, C.P_SEGF: segf, C.P_SHIFT: shift & 0xFFFFFFFF, C.P_NP: np_}
        if eps:
            p[C.P_EPS] = f32bits(eps)
        self.add(Cmd(op, flags, 0, 0, k, n, b, s, {i: v for i, v in p.items() if v}), note)

    def softmax(self, note, L, n, b, a: Slot, c: Slot, f: Slot, d: Slot | None, e: Slot | None, np_=0):
        s = {"A": a, "C": c, "F": f}
        fl = C.F_QOUT
        if d is not None:
            fl |= C.F_MASK
            s["D"], s["E"] = d, e
        p = {C.P_EPS: f32bits(1.0)} | ({C.P_NP: np_} if np_ else {})
        self.add(Cmd(C.OP_SOFTMAX, fl, 0, 0, L, n, b, s, p), note)


def kv_blobs(mem: Mem, keys_p: int, keys_v: int, heads: int) -> tuple[Blob, Blob]:
    """K blob (rows = keys_p padded keys, K = 64) and V^T blob (64 channel rows, K = keys_v), per KV head."""
    return blob(mem, keys_p, HD, heads), blob(mem, HD, keys_v, heads)


def build(base: int) -> Plan:  # noqa: C901 (the whole chunk, in order)
    P = Plan(base)
    m = P.mem
    # ---------------- shared region: weights + constants (written once at load)
    m.region("weights")
    W = {"patch": blob(m, V_D, 768, bias=True)}
    for i in range(V_L):
        W[f"v{i}.qkv"] = blob(m, 3 * V_D, V_D, bias=True)
        W[f"v{i}.o"] = blob(m, V_D, V_D, bias=True)
        W[f"v{i}.fc1"] = blob(m, V_FF, V_D, bias=True)
        W[f"v{i}.fc2"] = blob(m, V_D, V_FF, bias=True)
    W["conn"] = [blob(m, L_D, 4096) for _ in range(3)]  # sqrt(960) folded into s_row
    W["state"] = blob(m, L_D, A_D, bias=True)
    for i in range(L_L):
        W[f"l{i}.qkv"] = blob(m, L_D + 2 * 320, L_D) if i < L_L - 1 else blob(m, 640, L_D)  # layer 15: k|v
        if i < L_L - 1:
            W[f"l{i}.o"] = blob(m, L_D, L_D)
            W[f"l{i}.gu"] = blob(m, 2 * L_FF, L_D)
            W[f"l{i}.down"] = blob(m, L_D, L_FF)
    W["a_in"] = blob(m, E_DP, A_D, bias=True)
    W["t_in"] = blob(m, E_DP, E_D)  # the a_emb half of action_time_mlp_in; t half + bias = the D vectors
    W["t_out"] = blob(m, E_DP, E_D, bias=True)
    W["a_out"] = blob(m, A_D, E_D, bias=True)  # Euler folded: -W, s_row * 0.1, bias * -0.1
    for i in range(L_L):
        if i % 2 == 0:
            W[f"e{i}.qkv"] = blob(m, L_D + 2 * 320, E_D)
        else:
            W[f"e{i}.q"] = blob(m, L_D, E_D)
            W[f"e{i}.ck"] = blob(m, 320, 320)
            W[f"e{i}.cv"] = blob(m, 320, 320)
        W[f"e{i}.o"] = blob(m, E_DP, L_D)
        W[f"e{i}.gu"] = blob(m, 2 * E_FF, E_D)
        W[f"e{i}.down"] = blob(m, E_DP, E_FF)
    m.region("consts")
    pos = m.alloc(V_T * V_D * 4)
    lang = m.alloc(12 * L_D * 4)
    tvec = m.alloc(STEPS * E_DP * 4)
    gam = {k: m.alloc(8 * 1024) for k in [f"v{i}.ln{j}" for i in range(V_L) for j in (1, 2)] + ["v.post"]}
    gam |= {f"l{i}.n{j}": m.alloc(4 * 1024) for i in range(L_L) for j in (1, 2)}
    gam |= {f"e{i}.n{j}": m.alloc(4 * 1024) for i in range(L_L) for j in (1, 2)} | {"e.final": m.alloc(4096)}
    rope = {k: m.alloc(n * C.ROPE_TAB_BYTES) for k, n in (("pre", TQ_P), ("self", TQ_E), ("cross", TQ_E))}
    NQP, NQE = G * p4(TQ_P), G * p4(TQ_E)  # stacked query rows per KV head: 144, 156
    LP, LS = pad32(TQ_P), pad32(TQ_P) + pad32(TQ_E)  # keys: prefix 64, prefix + suffix 128
    rq = {"pre": m.alloc(4 * NQP), "self": m.alloc(4 * NQE), "cross": m.alloc(4 * NQE)}
    rk = {"pre": m.alloc(4 * LP), "self": m.alloc(4 * LS), "cross": m.alloc(4 * LP)}
    # ---------------- arena (one inference)
    m.region("arena")
    status = m.alloc(C.STATUS_BYTES)
    patches, s_patch = m.alloc(IMGS * V_T * 768 * 2), m.alloc(4 * IMGS * V_T)
    xstate, s_state = m.alloc(64), m.alloc(16)
    x_t = m.alloc(TQ_E * A_D * 4)
    R_vis, Z = m.alloc(IMGS * V_T * V_D * 4), m.alloc(IMGS * V_T * V_D * 4)
    X, sX = m.alloc(IMGS * V_T * V_FF * 2), m.alloc(4 * IMGS * V_T)  # int16 rows scratch (largest: fc1 out)
    QKV = m.alloc(IMGS * V_T * 3 * V_D * 4)
    H1 = m.alloc(IMGS * V_T * V_FF * 4)
    Qh, sQ = m.alloc(V_H * IMGS * V_T * HD * 2), m.alloc(4 * V_H * IMGS * V_T)
    KWv, VWv = kv_blobs(m, V_T, V_T, V_H * IMGS)
    S = m.alloc(V_H * IMGS * V_T * V_T * 4)
    Pm, sP = m.alloc(V_H * IMGS * V_T * V_T * 2), m.alloc(4 * V_H * IMGS * V_T)
    Ob = m.alloc(IMGS * V_T * V_D * 4)
    SH, XC, sXC = m.alloc(CONN_N * CONN_K * 4), m.alloc(CONN_N * CONN_K * 2), m.alloc(4 * CONN_N)
    R_pre = m.alloc(TQ_P * L_D * 4)
    QKV_l = {i: m.alloc(TQ_P * 1600 * 4) for i in range(1, L_L, 2)}  # odd layers: kept for the cross K/V
    QKV_s = m.alloc(TQ_P * 1600 * 4)
    KB = {i: kv_blobs(m, LS, pad32(TQ_P), L_HKV)[0] for i in range(0, L_L, 2)}  # self K: prefix + suffix rows
    VBp = {i: blob(m, HD, pad32(TQ_P), L_HKV) for i in range(0, L_L, 2)}  # self V^T, prefix keys
    VBs = {i: blob(m, HD, pad32(TQ_E), L_HKV) for i in range(0, L_L, 2)}  # self V^T, suffix keys
    KT, VT = kv_blobs(m, LP, LP, L_HKV)  # odd LM layers' own attention (temporary)
    KC = {i: kv_blobs(m, LP, LP, L_HKV) for i in range(1, L_L, 2)}  # cross K', V'^T (whole chunk)
    GU = m.alloc(TQ_P * 2 * L_FF * 4)
    Kp, Vp = m.alloc(TQ_P * 320 * 4), m.alloc(TQ_P * 320 * 4)
    R_suf, A_emb, Hm = (m.alloc(TQ_E * E_DP * 4) for _ in range(3))
    arena_end = m.top

    def sl(a, pitch=0, stride=0):
        return Slot(a, pitch, stride)

    # ================= vision: both images, 512 tokens
    P.phase = "vision"
    nv = IMGS * V_T
    P.gemm(
        "patch",
        W["patch"],
        patches,
        1536,
        V_T,
        R_vis,
        4 * V_D,
        s_patch,
        b=IMGS,
        xs=V_T * 1536,
        fs=4 * V_T,
        ys=V_T * 4 * V_D,
        bias=True,
        d=sl(pos, 4 * V_D, 0),
    )
    for i in range(V_L):
        P.row(
            C.OP_NORM,
            "ln1",
            V_D,
            nv,
            sl(R_vis, 4 * V_D),
            sl(X, 2 * V_D),
            sl(sX),
            flags=C.F_QOUT | C.F_LN,
            e=sl(gam[f"v{i}.ln1"]),
            eps=EPS_LN,
        )
        P.gemm("qkv", W[f"v{i}.qkv"], X, 2 * V_D, nv, QKV, 12 * V_D, sX, bias=True)
        P.row(
            C.OP_QUANT_X,
            "q",
            V_D,
            nv,
            sl(QKV, 12 * V_D),
            sl(Qh, 2 * HD),
            sl(sQ),
            seg=HD,
            segc=nv * 2 * HD,
            segf=4 * nv,
            shift=-3,
        )
        P.row(
            C.OP_QUANT_W,
            "k",
            V_D,
            nv,
            sl(QKV + 4 * V_D, 12 * V_D),
            sl(KWv.w, KWv.port),
            sl(KWv.srow),
            flags=0,
            seg=HD,
            segc=IMGS * KWv.item,
            segf=4 * nv,
            np_=nv,
        )
        P.row(
            C.OP_QUANT_WT,
            "v",
            V_D,
            V_T,
            sl(QKV + 8 * V_D, 12 * V_D, V_T * 12 * V_D),
            sl(VWv.w, VWv.port, VWv.item),
            sl(VWv.srow, 0, 4 * HD),
            flags=0,
            b=IMGS,
            seg=HD,
            segc=IMGS * VWv.item,
            segf=IMGS * 4 * HD,
            np_=V_T,
        )
        P.gemm(
            "qk",
            KWv,
            Qh,
            2 * HD,
            V_T,
            S,
            4 * V_T,
            sQ,
            k=HD,
            b=V_H * IMGS,
            xs=V_T * 2 * HD,
            ws=KWv.item,
            ss=4 * V_T,
            fs=4 * V_T,
            ys=V_T * 4 * V_T,
        )
        P.softmax(
            "softmax",
            V_T,
            V_T,
            V_H * IMGS,
            sl(S, 4 * V_T, V_T * 4 * V_T),
            sl(Pm, 2 * V_T, V_T * 2 * V_T),
            sl(sP, 0, 4 * V_T),
            None,
            None,
        )
        for img in range(IMGS):
            w = Blob(VWv.w + img * VWv.item, VWv.port, VWv.item, VWv.srow + img * 4 * HD, 0, HD, V_T)
            P.gemm(
                f"pv{img}",
                w,
                Pm + img * V_T * 2 * V_T,
                2 * V_T,
                V_T,
                Ob + img * V_T * 4 * V_D,
                4 * V_D,
                sP + img * 4 * V_T,
                b=V_H,
                xs=IMGS * V_T * 2 * V_T,
                ws=IMGS * VWv.item,
                ss=IMGS * 4 * HD,
                fs=IMGS * 4 * V_T,
                ys=4 * HD,
            )
        P.row(C.OP_QUANT_X, "o.in", V_D, nv, sl(Ob, 4 * V_D), sl(X, 2 * V_D), sl(sX))
        P.gemm("o", W[f"v{i}.o"], X, 2 * V_D, nv, R_vis, 4 * V_D, sX, bias=True, d=sl(R_vis, 4 * V_D))
        P.row(
            C.OP_NORM,
            "ln2",
            V_D,
            nv,
            sl(R_vis, 4 * V_D),
            sl(X, 2 * V_D),
            sl(sX),
            flags=C.F_QOUT | C.F_LN,
            e=sl(gam[f"v{i}.ln2"]),
            eps=EPS_LN,
        )
        P.gemm("fc1", W[f"v{i}.fc1"], X, 2 * V_D, nv, H1, 4 * V_FF, sX, bias=True)
        P.row(C.OP_GELU, "gelu", V_FF, nv, sl(H1, 4 * V_FF), sl(X, 2 * V_FF), sl(sX))
        P.gemm("fc2", W[f"v{i}.fc2"], X, 2 * V_FF, nv, R_vis, 4 * V_D, sX, bias=True, d=sl(R_vis, 4 * V_D))
    P.row(
        C.OP_NORM,
        "post_ln",
        V_D,
        nv,
        sl(R_vis, 4 * V_D),
        sl(Z, 4 * V_D),
        flags=C.F_LN,
        e=sl(gam["v.post"]),
        eps=EPS_LN,
    )
    # ================= connector + prefix assembly
    P.phase = "connector"
    tok = 4 * V_D  # one source token row, bytes
    for hi in range(4):  # out row r = (h4, w4), segment hi*4 + wi <- token (4 h4 + hi) * 16 + 4 w4 + wi
        P.row(
            C.OP_COPY,
            f"shuffle{hi}",
            4 * V_D,
            4,
            sl(Z + hi * 16 * tok, 4 * tok, 64 * tok),
            sl(SH + hi * 4 * tok, 4 * CONN_K, 4 * 4 * CONN_K),
            flags=0,
            b=IMGS * 4,
        )
    P.row(C.OP_QUANT_X, "conn.in", CONN_K, CONN_N, sl(SH, 4 * CONN_K), sl(XC, 2 * CONN_K), sl(sXC))
    for j in range(3):
        P.gemm(f"conn.k{j}", W["conn"][j], XC + j * 8192, 2 * CONN_K, CONN_N, R_pre, 4 * L_D, sXC, acc=j > 0)
    P.row(C.OP_COPY, "lang", L_D, 12, sl(lang, 4 * L_D), sl(R_pre + CONN_N * 4 * L_D, 4 * L_D), flags=0)
    P.gemm("state", W["state"], xstate, 64, 1, R_pre + (TQ_P - 1) * 4 * L_D, 4 * L_D, s_state, bias=True)
    # ================= LM prefix
    P.phase = "lm"
    qh_b, sq_b = p4(TQ_P) * 2 * HD, 4 * p4(TQ_P)  # one q head of Qh / sQ
    for i in range(L_L):
        qkv = QKV_l.get(i, QKV_s)
        P.row(
            C.OP_NORM,
            "n1",
            L_D,
            TQ_P,
            sl(R_pre, 4 * L_D),
            sl(X, 2 * L_D),
            sl(sX),
            e=sl(gam[f"l{i}.n1"]),
            eps=EPS_RMS,
        )
        if i == L_L - 1:  # dead work skipped: only the K/V cache of the last layer (cross layer 15 reads it)
            P.gemm("kv", W[f"l{i}.qkv"], X, 2 * L_D, TQ_P, qkv + 4 * L_D, 4 * 1600, sX)
            P.row(
                C.OP_ROPE,
                "k.f32",
                320,
                TQ_P,
                sl(qkv + 4 * L_D, 4 * 1600),
                sl(qkv + 4 * L_D, 4 * 1600),
                flags=0,
                seg=HD,
                segc=4 * HD,
                e=sl(rope["pre"], C.ROPE_TAB_BYTES),
            )
            break
        P.gemm("qkv", W[f"l{i}.qkv"], X, 2 * L_D, TQ_P, qkv, 4 * 1600, sX)
        P.row(
            C.OP_ROPE,
            "q",
            L_D,
            TQ_P,
            sl(qkv, 4 * 1600),
            sl(Qh, 2 * HD),
            sl(sQ),
            seg=HD,
            segc=qh_b,
            segf=sq_b,
            shift=-3,
            np_=p4(TQ_P),
            e=sl(rope["pre"], C.ROPE_TAB_BYTES),
        )
        if i % 2 == 0:
            kb, vb = KB[i], VBp[i]
            P.row(
                C.OP_ROPE,
                "k.kw",
                320,
                TQ_P,
                sl(qkv + 4 * L_D, 4 * 1600),
                sl(kb.w, kb.port),
                sl(kb.srow),
                flags=C.F_KW,
                seg=HD,
                segc=kb.item,
                segf=4 * LS,
                np_=LP,
                e=sl(rope["pre"], C.ROPE_TAB_BYTES),
            )
        else:
            kb, vb = KT, VT
            P.row(
                C.OP_ROPE,
                "k.f32",
                320,
                TQ_P,
                sl(qkv + 4 * L_D, 4 * 1600),
                sl(qkv + 4 * L_D, 4 * 1600),
                flags=0,
                seg=HD,
                segc=4 * HD,
                e=sl(rope["pre"], C.ROPE_TAB_BYTES),
            )
            P.row(
                C.OP_QUANT_W,
                "k.kw",
                320,
                TQ_P,
                sl(qkv + 4 * L_D, 4 * 1600),
                sl(kb.w, kb.port),
                sl(kb.srow),
                flags=0,
                seg=HD,
                segc=kb.item,
                segf=4 * LP,
                np_=LP,
            )
        P.row(
            C.OP_QUANT_WT,
            "v.vw",
            320,
            TQ_P,
            sl(qkv + 4 * 1280, 4 * 1600),
            sl(vb.w, vb.port),
            sl(vb.srow),
            flags=0,
            seg=HD,
            segc=vb.item,
            segf=4 * HD,
            np_=LP,
        )
        P.gemm(
            "qk",
            kb,
            Qh,
            2 * HD,
            NQP,
            S,
            4 * LP,
            sQ,
            k=HD,
            b=L_HKV,
            xs=G * qh_b,
            ws=kb.item,
            ss=4 * kb.mp,
            fs=G * sq_b,
            ys=NQP * 4 * LP,
        )
        kb_m = Blob(kb.w, kb.port, kb.item, kb.srow, 0, LP, HD)  # the LM reads the prefix rows only
        P.cmds[-1].M = kb_m.mp
        P.softmax(
            "softmax",
            LP,
            NQP,
            L_HKV,
            sl(S, 4 * LP, NQP * 4 * LP),
            sl(Pm, 2 * LP, NQP * 2 * LP),
            sl(sP, 0, 4 * NQP),
            sl(rq["pre"]),
            sl(rk["pre"]),
        )
        P.gemm(
            "pv",
            vb,
            Pm,
            2 * LP,
            NQP,
            Ob,
            4 * L_D,
            sP,
            k=LP,
            b=L_HKV,
            xs=NQP * 2 * LP,
            ws=vb.item,
            ss=4 * HD,
            fs=4 * NQP,
            ys=G * 4 * HD,
            R=p4(TQ_P),
            P1=4 * HD,
        )
        P.row(C.OP_QUANT_X, "o.in", L_D, TQ_P, sl(Ob, 4 * L_D), sl(X, 2 * L_D), sl(sX))
        P.gemm("o", W[f"l{i}.o"], X, 2 * L_D, TQ_P, R_pre, 4 * L_D, sX, d=sl(R_pre, 4 * L_D))
        P.row(
            C.OP_NORM,
            "n2",
            L_D,
            TQ_P,
            sl(R_pre, 4 * L_D),
            sl(X, 2 * L_D),
            sl(sX),
            e=sl(gam[f"l{i}.n2"]),
            eps=EPS_RMS,
        )
        P.gemm("gu", W[f"l{i}.gu"], X, 2 * L_D, TQ_P, GU, 8 * L_FF, sX)
        P.row(
            C.OP_SILU_MUL,
            "silu",
            L_FF,
            TQ_P,
            sl(GU, 8 * L_FF),
            sl(X, 2 * L_FF),
            sl(sX),
            bsl=sl(GU + 4 * L_FF, 8 * L_FF),
        )
        P.gemm("down", W[f"l{i}.down"], X, 2 * L_FF, TQ_P, R_pre, 4 * L_D, sX, d=sl(R_pre, 4 * L_D))
    # ================= cross K', V' (once per chunk)
    P.phase = "cross_kv"
    for i in range(1, L_L, 2):
        qkv, (kc, vc) = QKV_l[i], KC[i]
        P.row(C.OP_QUANT_X, "k.in", 320, TQ_P, sl(qkv + 4 * L_D, 4 * 1600), sl(X, 640), sl(sX))
        P.gemm("kproj", W[f"e{i}.ck"], X, 640, TQ_P, Kp, 1280, sX)
        P.row(
            C.OP_QUANT_X, "v.in", 320, TQ_P, sl(qkv + 4 * 1280, 4 * 1600), sl(X + 65536, 640), sl(sX + 4096)
        )
        P.gemm("vproj", W[f"e{i}.cv"], X + 65536, 640, TQ_P, Vp, 1280, sX + 4096)
        P.row(
            C.OP_QUANT_W,
            "k.kw",
            320,
            TQ_P,
            sl(Kp, 1280),
            sl(kc.w, kc.port),
            sl(kc.srow),
            flags=0,
            seg=HD,
            segc=kc.item,
            segf=4 * LP,
            np_=LP,
        )
        P.row(
            C.OP_QUANT_WT,
            "v.vw",
            320,
            TQ_P,
            sl(Vp, 1280),
            sl(vc.w, vc.port),
            sl(vc.srow),
            flags=0,
            seg=HD,
            segc=vc.item,
            segf=4 * HD,
            np_=LP,
        )
    # ================= 10 Euler steps of the expert
    qe_b, se_b = p4(TQ_E) * 2 * HD, 4 * p4(TQ_E)
    for s in range(STEPS):
        P.phase = "steps"
        P.row(C.OP_QUANT_X, "x_t", A_D, TQ_E, sl(x_t, 4 * A_D), sl(X, 2 * A_D), sl(sX))
        P.gemm("a_in", W["a_in"], X, 2 * A_D, TQ_E, A_emb, 4 * E_DP, sX, bias=True)
        P.row(C.OP_QUANT_X, "a_emb", E_D, TQ_E, sl(A_emb, 4 * E_DP), sl(X, 2 * E_D), sl(sX))
        P.gemm("t_in", W["t_in"], X, 2 * E_D, TQ_E, Hm, 4 * E_DP, sX, d=sl(tvec + s * 4 * E_DP, 0))
        P.row(
            C.OP_SILU_MUL,
            "t_silu",
            E_D,
            TQ_E,
            sl(Hm, 4 * E_DP),
            sl(X, 2 * E_D),
            sl(sX),
            flags=C.F_QOUT | C.F_NOMUL,
        )
        P.gemm("t_out", W["t_out"], X, 2 * E_D, TQ_E, R_suf, 4 * E_DP, sX, bias=True)
        for i in range(L_L):
            P.row(
                C.OP_NORM,
                "n1",
                E_D,
                TQ_E,
                sl(R_suf, 4 * E_DP),
                sl(X, 2 * E_D),
                sl(sX),
                e=sl(gam[f"e{i}.n1"]),
                eps=EPS_RMS,
            )
            if i % 2 == 0:
                kb, vbp, vbs = KB[i], VBp[i], VBs[i]
                P.gemm("qkv", W[f"e{i}.qkv"], X, 2 * E_D, TQ_E, QKV_s, 4 * 1600, sX)
                P.row(
                    C.OP_ROPE,
                    "q",
                    L_D,
                    TQ_E,
                    sl(QKV_s, 4 * 1600),
                    sl(Qh, 2 * HD),
                    sl(sQ),
                    seg=HD,
                    segc=qe_b,
                    segf=se_b,
                    shift=-3,
                    np_=p4(TQ_E),
                    e=sl(rope["self"], C.ROPE_TAB_BYTES),
                )
                off = (LP // 32) * HD * 16  # suffix keys start at blob row LP
                P.row(
                    C.OP_ROPE,
                    "k.kw",
                    320,
                    TQ_E,
                    sl(QKV_s + 4 * L_D, 4 * 1600),
                    sl(kb.w + off, kb.port),
                    sl(kb.srow + 4 * LP),
                    flags=C.F_KW,
                    seg=HD,
                    segc=kb.item,
                    segf=4 * LS,
                    np_=pad32(TQ_E),
                    e=sl(rope["self"], C.ROPE_TAB_BYTES),
                )
                P.row(
                    C.OP_QUANT_WT,
                    "v.vw",
                    320,
                    TQ_E,
                    sl(QKV_s + 4 * 1280, 4 * 1600),
                    sl(vbs.w, vbs.port),
                    sl(vbs.srow),
                    flags=0,
                    seg=HD,
                    segc=vbs.item,
                    segf=4 * HD,
                    np_=pad32(TQ_E),
                )
                P.gemm(
                    "qk",
                    kb,
                    Qh,
                    2 * HD,
                    NQE,
                    S,
                    4 * LS,
                    sQ,
                    k=HD,
                    b=L_HKV,
                    xs=G * qe_b,
                    ws=kb.item,
                    ss=4 * LS,
                    fs=G * se_b,
                    ys=NQE * 4 * LS,
                )
                P.softmax(
                    "softmax",
                    LS,
                    NQE,
                    L_HKV,
                    sl(S, 4 * LS, NQE * 4 * LS),
                    sl(Pm, 2 * LS, NQE * 2 * LS),
                    sl(sP, 0, 4 * NQE),
                    sl(rq["self"]),
                    sl(rk["self"]),
                )
                for part, (vb, col) in enumerate(((vbp, 0), (vbs, 2 * LP))):
                    P.gemm(
                        f"pv{part}",
                        vb,
                        Pm + col,
                        2 * LS,
                        NQE,
                        Ob,
                        4 * L_D,
                        sP,
                        k=vb.k,
                        b=L_HKV,
                        xs=NQE * 2 * LS,
                        ws=vb.item,
                        ss=4 * HD,
                        fs=4 * NQE,
                        ys=G * 4 * HD,
                        R=p4(TQ_E),
                        P1=4 * HD,
                        acc=part == 1,
                    )
            else:
                kc, vc = KC[i]
                P.gemm("q", W[f"e{i}.q"], X, 2 * E_D, TQ_E, QKV_s, 4 * 1600, sX)
                P.row(
                    C.OP_ROPE,
                    "q",
                    L_D,
                    TQ_E,
                    sl(QKV_s, 4 * 1600),
                    sl(Qh, 2 * HD),
                    sl(sQ),
                    seg=HD,
                    segc=qe_b,
                    segf=se_b,
                    shift=-3,
                    np_=p4(TQ_E),
                    e=sl(rope["cross"], C.ROPE_TAB_BYTES),
                )
                P.gemm(
                    "qk",
                    kc,
                    Qh,
                    2 * HD,
                    NQE,
                    S,
                    4 * LP,
                    sQ,
                    k=HD,
                    b=L_HKV,
                    xs=G * qe_b,
                    ws=kc.item,
                    ss=4 * LP,
                    fs=G * se_b,
                    ys=NQE * 4 * LP,
                )
                P.softmax(
                    "softmax",
                    LP,
                    NQE,
                    L_HKV,
                    sl(S, 4 * LP, NQE * 4 * LP),
                    sl(Pm, 2 * LP, NQE * 2 * LP),
                    sl(sP, 0, 4 * NQE),
                    sl(rq["cross"]),
                    sl(rk["cross"]),
                )
                P.gemm(
                    "pv",
                    vc,
                    Pm,
                    2 * LP,
                    NQE,
                    Ob,
                    4 * L_D,
                    sP,
                    k=LP,
                    b=L_HKV,
                    xs=NQE * 2 * LP,
                    ws=vc.item,
                    ss=4 * HD,
                    fs=4 * NQE,
                    ys=G * 4 * HD,
                    R=p4(TQ_E),
                    P1=4 * HD,
                )
            P.row(C.OP_QUANT_X, "o.in", L_D, TQ_E, sl(Ob, 4 * L_D), sl(X, 2 * L_D), sl(sX))
            P.gemm("o", W[f"e{i}.o"], X, 2 * L_D, TQ_E, R_suf, 4 * E_DP, sX, d=sl(R_suf, 4 * E_DP))
            P.row(
                C.OP_NORM,
                "n2",
                E_D,
                TQ_E,
                sl(R_suf, 4 * E_DP),
                sl(X, 2 * E_D),
                sl(sX),
                e=sl(gam[f"e{i}.n2"]),
                eps=EPS_RMS,
            )
            P.gemm("gu", W[f"e{i}.gu"], X, 2 * E_D, TQ_E, GU, 8 * E_FF, sX)
            P.row(
                C.OP_SILU_MUL,
                "silu",
                E_FF,
                TQ_E,
                sl(GU, 8 * E_FF),
                sl(X, 2 * E_FF),
                sl(sX),
                bsl=sl(GU + 4 * E_FF, 8 * E_FF),
            )
            P.gemm("down", W[f"e{i}.down"], X, 2 * E_FF, TQ_E, R_suf, 4 * E_DP, sX, d=sl(R_suf, 4 * E_DP))
        P.row(
            C.OP_NORM,
            "final",
            E_D,
            TQ_E,
            sl(R_suf, 4 * E_DP),
            sl(X, 2 * E_D),
            sl(sX),
            e=sl(gam["e.final"]),
            eps=EPS_RMS,
        )
        P.gemm("a_out+euler", W["a_out"], X, 2 * E_D, TQ_E, x_t, 4 * A_D, sX, bias=True, acc=True)
    P.phase = "end"
    P.add(Cmd(C.OP_END), "end")
    if OPT.get("fuse_qk"):
        P.cmds = fuse_qk_softmax(P.cmds)
    P.layout = {"status": status, "x_t": x_t, "arena_end": arena_end, "regions": m.regions}  # type: ignore[attr-defined]
    return P


# ---------------------------------------------------------------- timing (CONTRACT.md 6.2), clocks
L0, TAIL, YW_BLK, EMIT = 400, 450, 256, 70  # the engine model's per-call terms (fpga/hls/model_int16.py)
SEQ_FETCH = 80  # 16-beat record fetch + DDR latency + decode (est.)
ROW_C0, ROW_ITEM, ROW_GAP, LANES = 400, 64, 4, 4  # row unit: per command, per item, per row (est.)
SMX_C0, SMX_ITEM, SMX_DIV = 300, 150, 13  # softmax: per command, per item, divider clocks per row


OPT = {
    "nbx": 2,
    "item_fixed": None,
    "fuse_qk": False,
}  # --nbx, --item-overhead (engine change E-7: items inside one DATAFLOW run)


def gemm_item_clocks(mp: int, k: int, n: int, ddr_bpc: float, d_read: bool, first: bool = True) -> float:
    nbx = OPT["nbx"]
    g, nblk = mp // 32, math.ceil(n / 32)
    tiles = [min(nbx, nblk - t * nbx) for t in range(math.ceil(nblk / nbx))]
    fixed = L0 + TAIL if first or OPT["item_fixed"] is None else OPT["item_fixed"]
    total = fixed + 4 * tiles[0] * k
    for t, nbt in enumerate(tiles):
        nxt = tiles[t + 1] if t + 1 < len(tiles) else 0
        dw = YW_BLK * g * nbt if d_read else 0  # epilogue addend rows over the io read channel
        total += max(
            g * nbt * max(k, YW_BLK, EMIT),
            g * k,
            (64 * g * k + 4096 * g * nbt + 16 * dw + 64 * nxt * k) / ddr_bpc,
            4 * nxt * k + dw,
        )
    return total


def cmd_clocks(c: Cmd, ddr_bpc: float) -> tuple[float, int]:
    """(clocks, DDR bytes) of one command, sequencer fetch included."""
    if c.op == C.OP_END:
        return SEQ_FETCH, 0
    if c.op == C.OP_GEMM:
        d = bool(c.flags & (C.F_ADD_D | C.F_ACC)) and c.slot("D").pitch != 0 or bool(c.flags & C.F_ACC)
        clk = sum(gemm_item_clocks(c.M, c.K, c.N, ddr_bpc, d, b == 0) for b in range(min(c.B, 2)))
        clk += max(c.B - 2, 0) * gemm_item_clocks(c.M, c.K, c.N, ddr_bpc, d, False)
        byts = c.B * (2 * c.M * c.K + 2 * c.N * c.K + 4 * c.N * c.M * (2 if d else 1))
        return SEQ_FETCH + clk, byts
    if c.op == C.OP_QK_SOFTMAX:
        return qks_cmd_clocks(c, ddr_bpc)
    if c.op == C.OP_SOFTMAX:
        L = c.K
        per = max(L / 4, SMX_DIV, L / 8) + 2
        np_ = c.par(C.P_NP) or c.N
        return SEQ_FETCH + SMX_C0 + c.B * (SMX_ITEM + c.N * per + (np_ - c.N) * L / 8), c.B * c.N * L * 6
    K, N, seg = c.K, c.N, c.par(C.P_SEG) or c.K
    np_ = c.par(C.P_NP) or N
    q = c.flags & C.F_QOUT or c.op in (C.OP_QUANT_W, C.OP_QUANT_WT) or c.flags & C.F_KW
    rd = K / 4 * (2 if c.op == C.OP_SILU_MUL and not c.flags & C.F_NOMUL else 1)
    passes = 2 if c.op == C.OP_QUANT_WT or K > C.ROW_BUF else 1
    wr = K / 8 + (K // seg) / 4 if q else K / 4
    if c.op == C.OP_QUANT_WT:  # channels x keys written once per item after the column-amax pass
        wr = K / 8
    per = max(rd * passes, wr, K / LANES) + ROW_GAP
    rows = N * per + (np_ - N) * wr
    byts = int(c.B * (N * rd * 16 * passes + np_ * wr * 16))
    return SEQ_FETCH + ROW_C0 + c.B * (ROW_ITEM + rows), byts


# ---------------------------------------------------------------- phase B: QK_SOFTMAX (fpga/kernels/phaseB/qk_softmax)
def fuse_qk_softmax(cmds: list[Cmd]) -> list[Cmd]:
    """Every (GEMM "qk", SOFTMAX) pair -> one QK_SOFTMAX record: A, B of the GEMM (Q rows, KW blob); C, D, E, F of
    the SOFTMAX (P rows, rank_q, rank_k, P scales); G = the GEMM's s_tok (F), H = its s_row (E); the SOFTMAX's
    flags and p0 / p5. The score buffer of the pair is no longer written or read. Tags are renumbered."""
    out: list[Cmd] = []
    i = 0
    while i < len(cmds):
        g = cmds[i]
        if g.op == C.OP_GEMM and g.note.endswith(":qk"):
            s = cmds[i + 1]
            assert s.op == C.OP_SOFTMAX and s.note.endswith(":softmax"), (i, g.note, s.note)
            a, y = s.slot("A"), g.slot("C")
            assert (a.addr, a.pitch, a.stride) == (y.addr, y.pitch, y.stride), (i, a, y)
            assert (s.K, s.N, s.B) == (g.M, g.N, g.B) and not g.flags and not g.p, (i, g, s)
            sl = {"A": g.slot("A"), "B": g.slot("B"), "C": s.slot("C"), "F": s.slot("F"), "G": g.slot("F"),
                  "H": g.slot("E")}
            if s.flags & C.F_MASK:
                sl["D"], sl["E"] = s.slot("D"), s.slot("E")
            f = Cmd(C.OP_QK_SOFTMAX, s.flags, 0, g.M, g.K, g.N, g.B, sl, dict(s.p))
            f.note = g.note[: -len("qk")] + "qk_softmax"
            out.append(f)
            i += 2
        else:
            out.append(cmds[i])
            i += 1
    for j, c in enumerate(out):
        c.tag = j
    return out


QKS_TAIL = 128  # QK_SOFTMAX: after the last row, the divider (14) and the last scale burst + write response (est.)


def qks_item_clocks(L: int, k: int, n: int, ddr_bpc: float, first: bool = True, prev_drain: float = 0.0):
    """(clocks, drain of the last tile) of one QK_SOFTMAX item. Fill of tile t = the GEMM tile with the 4-lane
    dequantizer as the pace (the store's Y words now go to the score buffer: no Y in the DDR term); drain of tile t
    = its rows at smx::compute's pace (W + 1 clocks per row, W = L / 4; at least the divider's 14). Fill of tile
    t + 1 overlaps drain of tile t (2 halves). With E-7 (item_fixed set) the next item's tile 0 also overlaps this
    item's last drain (prev_drain)."""
    nbx = OPT["nbx"]
    g, nblk = L // 32, math.ceil(n / 32)
    tiles = [min(nbx, nblk - t * nbx) for t in range(math.ceil(nblk / nbx))]
    fixed = L0 + TAIL if first or OPT["item_fixed"] is None else OPT["item_fixed"]
    lanes = OPT.get("qks_lanes", 4)  # dequantizer and softmax lanes (4 as built: no DSP added)
    row = max(L // lanes + 1, SMX_DIV + 1)
    fill, drain = [], []
    for t, nbt in enumerate(tiles):
        nxt = tiles[t + 1] if t + 1 < len(tiles) else 0
        blk = YW_BLK * 4 // lanes  # clocks per 32 x 32 block at the dequantizer
        fill.append(max(g * nbt * max(k, blk, EMIT), g * k, (64 * g * k + 64 * nxt * k) / ddr_bpc, 4 * nxt * k))
        drain.append(min(32 * nbt, n - 32 * nbx * t) * row)
    total = fixed + 4 * tiles[0] * k + max(fill[0], prev_drain)
    for t in range(1, len(tiles)):
        total += max(fill[t], drain[t - 1])
    return total, drain[-1]


def qks_cmd_clocks(c: Cmd, ddr_bpc: float) -> tuple[float, int]:
    L, n = c.M, c.N
    e7 = OPT["item_fixed"] is not None
    clk, prev = 0.0, 0.0
    for b in range(c.B):
        t, d = qks_item_clocks(L, c.K, n, ddr_bpc, b == 0, prev if e7 else 0.0)
        clk += t if e7 else t + d
        prev = d
    if e7:
        clk += prev  # the last item's drain
    clk += QKS_TAIL * (1 if e7 else c.B)
    np_ = c.par(C.P_NP) or n
    byts = c.B * (2 * L * c.K + 2 * n * c.K + 2 * np_ * L + 4 * pad4(np_) + 4 * pad4(n) + 4 * L
                  + (4 * pad4(n) + 4 * L if c.flags & C.F_MASK else 0))
    return SEQ_FETCH + clk, byts


def report(P: Plan, mhz: float, ddr_gbs: float, arm_ms: float, link_ms: float) -> dict:
    ddr_bpc = ddr_gbs * 1e9 / (mhz * 1e6)
    by_op, by_phase = defaultdict(lambda: [0, 0.0, 0]), defaultdict(lambda: [0, 0.0])
    total = 0.0
    for c in P.cmds:
        clk, byts = cmd_clocks(c, ddr_bpc)
        name = C.opnames[c.op] + (".attn" if c.op == C.OP_GEMM and c.K == HD or "pv" in c.note else "")
        by_op[name][0] += 1
        by_op[name][1] += clk
        by_op[name][2] += byts
        ph = c.note.split(":")[0]
        by_phase[ph][0] += 1
        by_phase[ph][1] += clk
        total += clk
    ms = 1e3 / (mhz * 1e6)
    F = total * ms / 1e3
    T = F + (arm_ms + link_ms) / 1e3
    return {
        "mhz": mhz,
        "commands": len(P.cmds),
        "F_s": F,
        "T_s": T,
        "margin_1_in_flight_s": 2 * 0.8167 - 2 * T,
        "margin_2_in_flight_s": 1.6333 - (F + T),
        "by_op": {
            k: {"cmds": v[0], "ms": v[1] * ms, "GB": v[2] / 1e9}
            for k, v in sorted(by_op.items(), key=lambda x: -x[1][1])
        },
        "by_phase": {k: {"cmds": v[0], "ms": v[1] * ms} for k, v in by_phase.items()},
        "ddr_GB": sum(v[2] for v in by_op.values()) / 1e9,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x8_0000_0000, help="CMA base (physical)")
    ap.add_argument("--mhz", type=float, nargs="*", default=[250.0, 222.0, 200.0, 300.0])
    ap.add_argument("--ddr", type=float, default=14.0, help="DDR GB/s (unmeasured literature figure)")
    ap.add_argument("--arm-ms", type=float, default=30.0)
    ap.add_argument("--link-ms", type=float, default=30.0)
    ap.add_argument("--out", type=Path, help="write the encoded list here (for tb_contract)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--nbx", type=int, default=2, help="engine blocks per tile (1: no weight re-use)")
    ap.add_argument("--item-overhead", type=int, help="clocks per extra GEMM item (default: a full call)")
    ap.add_argument("--fuse-qk", action="store_true", help="phase B: (GEMM qk, SOFTMAX) -> QK_SOFTMAX")
    a = ap.parse_args()
    OPT["nbx"], OPT["item_fixed"], OPT["fuse_qk"] = a.nbx, a.item_overhead, a.fuse_qk
    P = build(a.base)
    bad = validate_list(P.cmds)
    for i, r in bad[:20]:
        print(f"INVALID {i} {P.cmds[i].note} {C.opnames[P.cmds[i].op]}: ME_{C.menames[r]}")
    lay = P.layout  # type: ignore[attr-defined]
    regs = {k: (v[1] - v[0]) / 2**20 for k, v in lay["regions"].items()}
    print(
        f"{len(P.cmds)} records ({len(P.cmds) * C.REC_BYTES / 1024:.0f} KB), {len(bad)} invalid; "
        f"regions MB: " + ", ".join(f"{k} {v:.1f}" for k, v in regs.items()) + f"; top 0x{lay['arena_end']:x}"
    )
    ops = defaultdict(int)
    for c in P.cmds:
        ops[C.opnames[c.op]] += 1
    print("by opcode: " + ", ".join(f"{k} {v}" for k, v in sorted(ops.items(), key=lambda x: -x[1])))
    res = []
    for mhz in a.mhz:
        r = report(P, mhz, a.ddr, a.arm_ms, a.link_ms)
        res.append(r)
        print(
            f"\n@ {mhz:.0f} MHz, DDR {a.ddr:.0f} GB/s: F {r['F_s']:.3f} s  T {r['T_s']:.3f} s  "
            f"margin 1 in flight {r['margin_1_in_flight_s']:+.3f} s, "
            f"2 in flight {r['margin_2_in_flight_s']:+.3f} s"
            f"  DDR {r['ddr_GB']:.2f} GB/chunk"
        )
        print("| mode | commands | ms | GB |")
        print("|---|---|---|---|")
        for k, v in r["by_op"].items():
            print(f"| {k} | {v['cmds']} | {v['ms']:.1f} | {v['GB']:.2f} |")
        print(
            "phases: " + ", ".join(f"{k} {v['cmds']} cmds {v['ms']:.1f} ms" for k, v in r["by_phase"].items())
        )
    if a.out:
        a.out.write_bytes(encode_list(P.cmds))
        print(f"\nwrote {a.out}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
