"""Build integration/ = the snapshot (snap/, SHA256SUMS) with the qk_softmax fusion applied, and the unified diffs
integration/patches/*.diff (repo paths, `patch -p1` / `git apply` from the repo root).

Every edit is an exact anchored replacement that must match once, so a changed base fails loudly instead of
patching the wrong place. The C++ fork used by this directory's own builds (gemm_qks.h + contract_qks.h) and the
integration copy (gemm_contract.h with the opcode in contract.h) differ only in where the opcode's constants and
validate branch live; this script derives the latter from the former.

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\phaseB\\qk_softmax\\make_integration.py
"""

from __future__ import annotations

import difflib
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE / "snap"
OUT = HERE / "integration"
REPO = "fpga"  # snap/hls -> fpga/hls, snap/kernels -> fpga/kernels


def sub1(txt: str, old: str, new: str, what: str) -> str:
    n = txt.count(old)
    if n != 1:
        raise SystemExit(f"{what}: anchor found {n} times:\n{old}")
    return txt.replace(old, new)


# ---------------------------------------------------------------- contract.h
def contract_h(t: str) -> str:
    t = sub1(
        t,
        "constexpr uint32_t SLOT_G = 6;  // GEMM bias\nconstexpr uint32_t SLOT_H = 7;  // unused in 1.0 (must be 0)\n",
        "constexpr uint32_t SLOT_G = 6;  // GEMM bias / QK_SOFTMAX s_tok of Q\n"
        "constexpr uint32_t SLOT_H = 7;  // QK_SOFTMAX s_row of K (unused by every other opcode: must be 0)\n",
        "contract.h slots",
    )
    t = sub1(
        t,
        "constexpr uint32_t OP_SOFTMAX = 0x40;   // softmax unit: masked rows -> int16 P + 1/sum scale\n",
        "constexpr uint32_t OP_SOFTMAX = 0x40;   // softmax unit: masked rows -> int16 P + 1/sum scale\n"
        "constexpr uint32_t OP_QK_SOFTMAX = 0x41;  // engine (GEMM unit): Q.K^T + softmax in one command -> int16 P +\n"
        "                                        // 1/sum scale; the scores stay on chip (phase B, qk_softmax)\n",
        "contract.h opcode",
    )
    t = sub1(
        t,
        "constexpr uint32_t FLAGS_SOFTMAX = F_YIELD | F_QOUT | F_MASK;\n",
        "constexpr uint32_t FLAGS_SOFTMAX = F_YIELD | F_QOUT | F_MASK;\nconstexpr uint32_t FLAGS_QKSMX = F_YIELD | F_QOUT | F_MASK;\n",
        "contract.h flags",
    )
    t = sub1(
        t,
        "constexpr uint32_t ME_SLOT = 6;   // an unused slot or reserved field not zero\n",
        "constexpr uint32_t ME_SLOT = 6;   // an unused slot or reserved field not zero\n"
        "constexpr uint32_t ME_UNIT = 7;   // the opcode's unit is not in this bitstream (a fused build without the\n"
        "                                  // SOFTMAX unit returns it for OP_SOFTMAX)\n",
        "contract.h ME_UNIT",
    )
    t = sub1(
        t,
        "constexpr uint32_t SMX_L_MAX = 256;          // SOFTMAX keys per row\n",
        "constexpr uint32_t SMX_L_MAX = 256;          // SOFTMAX keys per row\n"
        "constexpr uint32_t QKS_L_MAX = 256;          // QK_SOFTMAX keys per row (M): the score buffer holds 2 x 64 rows\n",
        "contract.h limits",
    )
    t = sub1(
        t,
        "         op == OP_ROPE || op == OP_COPY;\n",
        "         op == OP_ROPE || op == OP_COPY || op == OP_QK_SOFTMAX;\n",
        "contract.h known_op",
    )
    # the validate branch: contract_qks.h's validate_qks_t, placed before validate_t, and dispatched from it
    q = (HERE / "contract_qks.h").read_text()
    body = q[q.index("// validate / validate_hw of OP_QK_SOFTMAX") : q.index("static inline uint32_t validate_qks(")]
    body = body.replace("QKS_L_MAX", "QKS_L_MAX")
    t = sub1(t, "template <bool FULL>\nstatic inline uint32_t validate_t(const Cmd& c) {\n",
             "// OP_QK_SOFTMAX (phase B, fpga/kernels/phaseB/qk_softmax): M = L keys, K = head dim; slots A Q int16 rows, B KW\n"
             "// blob, C P int16 rows, D rank_q, E rank_k (F_MASK), F P scales, G s_tok of Q, H s_row of K; p0 score\n"
             "// multiplier, p5 NP. The same rules as the GEMM and SOFTMAX branches with these slot roles.\n"
             + body
             + "template <bool FULL>\nstatic inline uint32_t validate_t(const Cmd& c) {\n", "contract.h validate_qks_t")
    t = sub1(
        t,
        "  if (c.flags & F_RETIRED) return ME_FLAGS;\n  if (c.op == OP_NOP || c.op == OP_END) {\n    allowed = F_YIELD;",
        "  if (c.flags & F_RETIRED) return ME_FLAGS;\n  if (c.op == OP_QK_SOFTMAX) return validate_qks_t<FULL>(c);\n"
        "  if (c.op == OP_NOP || c.op == OP_END) {\n    allowed = F_YIELD;",
        "contract.h validate dispatch",
    )
    return t


# ---------------------------------------------------------------- contract.py
CPY_BRANCH = '''    elif c.op == C.OP_QK_SOFTMAX:  # = ctr::validate_qks_t (phase B)
        if not c.flags & C.F_QOUT:
            return C.ME_FLAGS
        if M < C.ENGINE_LANES or M > C.QKS_L_MAX or M % C.ENGINE_LANES or K < 8 or K > C.GEMM_K_MAX or K % 8:
            return C.ME_DIMS
        NP = p[C.P_NP] or N
        if NP < N or NP > C.N_MAX:
            return C.ME_PARAM
        mul = bits_f32(p[C.P_EPS])
        if not (0.0 < mul < 1e30):
            return C.ME_PARAM
        if any(v for i, v in enumerate(p) if i not in (C.P_EPS, C.P_NP)):
            return C.ME_PARAM
        used = {"A", "B", "C", "F", "G", "H"}
        if r := _check_slot(S["A"], B, N, 2 * K, 2 * K, False, full):
            return r
        port = (M // C.ENGINE_LANES) * K * 16
        if not _aligned(S["B"]) or S["B"].pitch < port:
            return C.ME_ALIGN
        if S["B"].addr >= C.ADDR_LIMIT or (full and _end(S["B"], B, C.ENGINE_PORTS, port) > C.ADDR_LIMIT):
            return C.ME_ADDR
        for r in (
            _check_slot(S["C"], B, NP, 2 * M, 2 * M, False, full),
            _check_slot(S["F"], B, 1, 4 * pad4(NP), 0, True, full),
            _check_slot(S["G"], B, 1, 4 * pad4(N), 0, True, full),
            _check_slot(S["H"], B, 1, 4 * M, 0, True, full),
        ):
            if r:
                return r
        if c.flags & C.F_MASK:
            used |= {"D", "E"}
            if r := _check_slot(S["D"], B, 1, 4 * pad4(N), 0, True, full):
                return r
            if r := _check_slot(S["E"], B, 1, 4 * M, 0, True, full):
                return r
'''


CPY_CASES = '''    qks = Cmd(  # QK_SOFTMAX (phase B): the expert self-attention shape
        C.OP_QK_SOFTMAX,
        C.F_QOUT | C.F_MASK,
        11,
        128,
        64,
        156,
        5,
        {
            "A": Slot(0x7B00_0000, 128, 156 * 128),
            "B": Slot(0x7C00_0000, 4 * 64 * 16, 4 * 4 * 64 * 16),
            "C": Slot(0x7D00_0000, 256, 156 * 256),
            "D": Slot(0x7E00_0000),
            "E": Slot(0x7E00_1000),
            "F": Slot(0x7E00_2000, 0, 624),
            "G": Slot(0x7E00_4000, 0, 624),
            "H": Slot(0x7E00_6000, 0, 512),
        },
        {C.P_EPS: f32bits(1.0)},
    )

    def qv(name, **kw):
        s = dict(qks.s) | kw.pop("s", {})
        for k in kw.pop("drop", ()):
            s.pop(k)
        out.append((name, Cmd(qks.op, kw.pop("flags", qks.flags), 11, kw.pop("M", 128), kw.pop("K", 64),
                              kw.pop("N", 156), 5, s, kw.pop("p", dict(qks.p)))))

    qv("qks_ok")
    qv("qks_vision_ok", flags=C.F_QOUT, drop=("D", "E"), M=256, N=256,
       s={"A": Slot(0x7B00_0000, 128, 256 * 128), "B": Slot(0x7C00_0000, 8 * 64 * 16, 4 * 8 * 64 * 16),
          "C": Slot(0x7D00_0000, 512, 256 * 512), "F": Slot(0x7E00_2000, 0, 1024), "G": Slot(0x7E00_4000, 0, 1024),
          "H": Slot(0x7E00_6000, 0, 1024)})
    qv("qks_np_ok", p={C.P_EPS: f32bits(1.0), C.P_NP: 160})
    qv("qks_m288", M=288)
    qv("qks_m48", M=48)
    qv("qks_k81", K=81)
    qv("qks_noqout", flags=C.F_MASK)
    qv("qks_bias_flag", flags=C.F_QOUT | C.F_MASK | C.F_BIAS)
    qv("qks_retired_flag", flags=C.F_QOUT | C.F_MASK | 0x1000)
    qv("qks_np_lt_n", p={C.P_EPS: f32bits(1.0), C.P_NP: 100})
    qv("qks_p0_zero", p={})
    qv("qks_p2_set", p={C.P_EPS: f32bits(1.0), C.P_SEGC: 4})
    qv("qks_short_c_pitch", s={"C": Slot(0x7D00_0000, 128, 156 * 256)})
    qv("qks_short_port", s={"B": Slot(0x7C00_0000, 2048, 4 * 4 * 64 * 16)})
    qv("qks_unaligned_h", s={"H": Slot(0x7E00_6008, 0, 512)})
    qv("qks_h_past_40bit", s={"H": Slot((1 << 40) - 2048, 0, 512)})
    qv("qks_h_bit40", s={"H": Slot(1 << 40, 0, 0)})
    qv("qks_stray_d", flags=C.F_QOUT)
    qv("qks_unaligned_e", s={"E": Slot(0x7E00_1004)})
'''
CPY_EXPECT = (
    '"qks_ok": "OK", "qks_vision_ok": "OK", "qks_np_ok": "OK", "qks_m288": "DIMS", "qks_m48": "DIMS",\n'
    '    "qks_k81": "DIMS", "qks_noqout": "FLAGS", "qks_bias_flag": "FLAGS", "qks_retired_flag": "FLAGS",\n'
    '    "qks_np_lt_n": "PARAM", "qks_p0_zero": "PARAM", "qks_p2_set": "PARAM", "qks_short_c_pitch": "ALIGN",\n'
    '    "qks_short_port": "ALIGN", "qks_unaligned_h": "ALIGN", "qks_h_past_40bit": "ADDR", "qks_h_bit40": "ADDR",\n'
    '    "qks_stray_d": "SLOT", "qks_unaligned_e": "ALIGN",\n'
)


def contract_py(t: str) -> str:
    t = sub1(t, "    return out\n\n\nEXPECT = {", CPY_CASES + "    return out\n\n\nEXPECT = {", "contract.py cases")
    t = sub1(t, '"gelu_n0": "DIMS", "gelu_n65536": "DIMS",\n',
             '"gelu_n0": "DIMS", "gelu_n65536": "DIMS",\n    ' + CPY_EXPECT, "contract.py expect")
    t = sub1(t, "        C.OP_SOFTMAX: C.FLAGS_SOFTMAX,\n",
             "        C.OP_SOFTMAX: C.FLAGS_SOFTMAX,\n        C.OP_QK_SOFTMAX: C.FLAGS_QKSMX,\n", "contract.py allowed")
    t = sub1(t, "    elif c.op == C.OP_SOFTMAX:\n        if K < 8 or K > C.SMX_L_MAX or K % 8:\n",
             CPY_BRANCH + "    elif c.op == C.OP_SOFTMAX:\n        if K < 8 or K > C.SMX_L_MAX or K % 8:\n",
             "contract.py validate branch")
    return t


# ---------------------------------------------------------------- seq_core.h
def seq_core(t: str) -> str:
    return sub1(
        t,
        "  if (op == ctr::OP_GEMM) return U_GEMM;\n",
        "  if (op == ctr::OP_GEMM) return U_GEMM;\n"
        "  if (op == ctr::OP_QK_SOFTMAX) return U_GEMM;  // phase B: the GEMM unit runs Q.K^T + softmax\n",
        "seq_core.h unit_of",
    )


# ---------------------------------------------------------------- float_seq.h
def float_seq(t: str) -> str:
    t = sub1(
        t,
        "//   GEMM                          fgemm::gemm_unit (gemm_contract.h: the int16 engine behind the contract record)\n",
        "//   GEMM, QK_SOFTMAX              fgemm::gemm_unit (gemm_contract.h: the int16 engine behind the contract record;\n"
        "//                                 QK_SOFTMAX adds the score buffer and the softmax arithmetic, qks_tail.h)\n",
        "float_seq.h header",
    )
    t = sub1(
        t,
        "//   SOFTMAX                       smx::softmax_mode                (fpga/kernels/softmax/softmax_core.h)\n",
        "//   SOFTMAX                       smx::softmax_mode                (fpga/kernels/softmax/softmax_core.h), only with\n"
        "//                                 -DFSEQ_SOFTMAX_UNIT=1 (bring-up builds); else ME_UNIT: the list builder\n"
        "//                                 emits QK_SOFTMAX, whose softmax processes live in the GEMM unit\n",
        "float_seq.h softmax line",
    )
    t = sub1(
        t,
        "namespace fseq {\n",
        "#ifndef FSEQ_SOFTMAX_UNIT\n#define FSEQ_SOFTMAX_UNIT 0  // phase B qk_softmax: the standalone SOFTMAX unit is not built\n#endif\n\nnamespace fseq {\n",
        "float_seq.h option",
    )
    t = sub1(
        t,
        "#pragma HLS INLINE\n    return smx::softmax_mode(c, kw(io_rd), kw(io_wr));\n  }\n",
        "#pragma HLS INLINE\n#if FSEQ_SOFTMAX_UNIT\n    return smx::softmax_mode(c, kw(io_rd), kw(io_wr));\n#else\n"
        "    (void)c;\n    (void)io_rd;\n    (void)io_wr;\n    return ctr::ME_UNIT;  // no SOFTMAX unit in this build (nothing is touched)\n#endif\n  }\n",
        "float_seq.h Units::softmax",
    )
    return t


# ---------------------------------------------------------------- contract_plan.py (list builder + time model)
PLAN_FUSE = '''

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
'''


def contract_plan(t: str) -> str:
    t = sub1(t, 'OPT = {\n    "nbx": 2,\n    "item_fixed": None,\n}',
             'OPT = {\n    "nbx": 2,\n    "item_fixed": None,\n    "fuse_qk": False,\n}', "plan OPT")
    t = sub1(t, '    P.phase = "end"\n    P.add(Cmd(C.OP_END), "end")\n',
             '    P.phase = "end"\n    P.add(Cmd(C.OP_END), "end")\n    if OPT.get("fuse_qk"):\n'
             '        P.cmds = fuse_qk_softmax(P.cmds)\n', "plan build fuse")
    t = sub1(t, '    if c.op == C.OP_SOFTMAX:\n        L = c.K\n',
             '    if c.op == C.OP_QK_SOFTMAX:\n        return qks_cmd_clocks(c, ddr_bpc)\n'
             '    if c.op == C.OP_SOFTMAX:\n        L = c.K\n', "plan cmd_clocks")
    t = sub1(t, "\n\ndef report(P: Plan,", PLAN_FUSE + "\n\ndef report(P: Plan,", "plan fuse functions")
    t = sub1(t, '    ap.add_argument("--item-overhead", type=int, help="clocks per extra GEMM item (default: a full call)")\n',
             '    ap.add_argument("--item-overhead", type=int, help="clocks per extra GEMM item (default: a full call)")\n'
             '    ap.add_argument("--fuse-qk", action="store_true", help="phase B: (GEMM qk, SOFTMAX) -> QK_SOFTMAX")\n',
             "plan argparse")
    t = sub1(t, '    OPT["nbx"], OPT["item_fixed"] = a.nbx, a.item_overhead\n',
             '    OPT["nbx"], OPT["item_fixed"], OPT["fuse_qk"] = a.nbx, a.item_overhead, a.fuse_qk\n', "plan OPT set")
    return t


# ---------------------------------------------------------------- exec_ref.py (golden executor)
EXEC_OP = '''
    # ------------------------------------------------------------ QK_SOFTMAX (phase B: GEMM unit)
    def op_qk_softmax(self, c: K.Cmd) -> None:
        """Q.K^T with the GEMM epilogue (no bias / D), then the SOFTMAX rule on the FP32 scores of each row; the
        scores never reach memory. = (GEMM, SOFTMAX) of contract 1.0 minus the score buffer."""
        assert not self.exact, "QK_SOFTMAX: exact mode not supported"
        L, Kc, N = c.M, c.K, c.N
        NP = c.par(C.P_NP) or N
        mul = K.bits_f32(c.par(C.P_EPS))
        A, Bw, cs, D, E, fs, G, H = (c.slot(n) for n in "ABCDEFGH")
        for b in range(c.B):
            x = self.rows_i16(A, b, N, Kc)
            w = self.weight(Bw, b, L, Kc)
            acc = torch.matmul(x, w.T).numpy()
            y = K.gemm_out(np.rint(acc).astype(np.int64), self.vec_f32(H, b, L).numpy(), self.vec_f32(G, b, N).numpy())
            xs = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            if c.flags & C.F_MASK:
                rq = self.vec_i32(D, b, N).long()
                rk = self.vec_i32(E, b, L).long()
                valid = (rk[None, :] >= 0) & (rk[None, :] <= rq[:, None])
            else:
                valid = torch.ones(N, L, dtype=torch.bool)
            if "smx" in self.kops:
                q, s = self.smx.kernel(xs, self.smx.c_from_mult(mul), valid, empty="zero")
                s = torch.where(q.sum(-1, keepdim=True) == 0, torch.ones_like(s), s)
                q, s = q.numpy().astype(np.int16), s.reshape(-1).numpy().astype(np.float32)
            else:
                p = torch.softmax(torch.where(valid, xs * mul, torch.tensor(float("-inf"))), -1)
                q, s, _ = self.quant(torch.nan_to_num(p, nan=0.0), 0)
            qp = np.zeros((NP, L), np.int16)
            qp[:N] = q
            self.mem.view(cs.addr + b * cs.stride, (NP, L), (cs.pitch, 2), torch.int16).copy_(torch.from_numpy(qp))
            self.put_scales(fs, b, 0, 0, s, NP)
'''


def exec_ref(t: str) -> str:
    t = sub1(t, "def unit_of(op: int) -> int:\n    if op == C.OP_GEMM:\n",
             "def unit_of(op: int) -> int:\n    if op == C.OP_GEMM or op == getattr(C, \"OP_QK_SOFTMAX\", -1):\n",
             "exec_ref unit_of")
    t = sub1(t, "    # ------------------------------------------------------------ row unit\n    def op_quant_x(",
             EXEC_OP.lstrip("\n") + "\n    # ------------------------------------------------------------ row unit\n    def op_quant_x(",
             "exec_ref op")
    return t


# ---------------------------------------------------------------- build_smolvla.py (the -12392 data)
def build_smolvla(t: str) -> str:
    t = sub1(t, "            if c.op != C.OP_SOFTMAX or not c.flags & C.F_MASK:\n                continue\n            nq, L = c.N, c.K\n",
             "            if c.op not in (C.OP_SOFTMAX, getattr(C, \"OP_QK_SOFTMAX\", -1)) or not c.flags & C.F_MASK:\n"
             "                continue\n            nq, L = c.N, (c.K if c.op == C.OP_SOFTMAX else c.M)\n",
             "build_smolvla ranks")
    return t


# ---------------------------------------------------------------- gemm_contract.h = gemm_qks.h with the opcode in contract.h
def gemm_contract() -> str:
    t = (HERE / "gemm_qks.h").read_text()
    t = sub1(t, '#include "contract_qks.h"\n', "", "gemm_qks.h contract_qks include")
    return t


FILES = {  # integration path -> (snapshot path, transform)
    "hls/gemm_contract.h": ("hls/gemm_contract.h", None),
    "hls/qks_tail.h": (None, None),
    "hls/float_seq.h": ("hls/float_seq.h", float_seq),
    "kernels/contract.h": ("kernels/contract.h", contract_h),
    "kernels/contract.py": ("kernels/contract.py", contract_py),
    "kernels/contract_plan.py": ("kernels/contract_plan.py", contract_plan),
    "kernels/sequencer/seq_core.h": ("kernels/sequencer/seq_core.h", seq_core),
    "kernels/sequencer/exec_ref.py": ("kernels/sequencer/exec_ref.py", exec_ref),
    "kernels/sequencer/build_smolvla.py": ("kernels/sequencer/build_smolvla.py", build_smolvla),
}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "patches").mkdir(parents=True)
    allp = []
    for dst, (src, fn) in FILES.items():
        old = (SNAP / src).read_text() if src else ""
        if dst == "hls/gemm_contract.h":
            new = gemm_contract()
        elif dst == "hls/qks_tail.h":
            new = (HERE / "qks_tail.h").read_text()
        else:
            new = fn(old)
        p = OUT / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new, newline="\n")
        a = f"a/{REPO}/{src}" if src else "/dev/null"
        d = difflib.unified_diff(old.splitlines(True), new.splitlines(True), a, f"b/{REPO}/{dst}", n=3)
        txt = "".join(d)
        name = dst.replace("/", "__") + ".diff"
        (OUT / "patches" / name).write_text(txt, newline="\n")
        allp.append(txt)
        nadd = sum(1 for ln in txt.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        ndel = sum(1 for ln in txt.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        print(f"{dst:40s} +{nadd} -{ndel}")
    (OUT / "patches" / "all.diff").write_text("".join(allp), newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
