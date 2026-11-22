"""Python side of the float_top integration contract (fpga/kernels/CONTRACT.md).

Every constant comes from contract.h (parsed), so C++ and Python cannot drift. This module has the record
encoder/decoder, the validation rules (the same as ctr::validate), the start registers and status parsing,
the layout address helpers, the weight-tiling packer, and the reference arithmetic (int16 quantizer, GEMM
epilogue) that every producer, the golden list executor and the ARM-side input quantization must match.

    from contract import C, Cmd, Slot, encode_list, validate
    cmd = Cmd(C.OP_GEMM, M=768, K=768, N=512, s={"A": Slot(x, 1536), "B": Slot(w, port), ...})

    .venv/Scripts/python.exe fpga/kernels/contract.py --selftest [--vectors DIR]   # checks + C++ test vectors
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_H = Path(__file__).with_name("contract.h")
_OPS = {ast.BitOr: operator.or_, ast.LShift: operator.lshift, ast.Add: operator.add, ast.Sub: operator.sub}


def _eval(expr: str, env: dict[str, int]) -> int:
    expr = re.sub(r"(?<=\d)u?ll\b|(?<=[0-9A-Fa-f])u\b", "", expr.strip())

    def ev(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"contract.h: unsupported constant expression {expr!r}")

    return ev(ast.parse(expr, mode="eval"))


class _Consts:
    def __init__(self, path: Path):
        txt = path.read_text()
        self.values: dict[str, int] = {}
        for m in re.finditer(r"constexpr\s+uint(?:32|64)_t\s+(\w+)\s*=\s*([^;]+);", txt):
            self.values[m.group(1)] = _eval(m.group(2), self.values)
        for k, v in self.values.items():
            setattr(self, k, v)
        self.opnames = {v: k[3:] for k, v in self.values.items() if k.startswith("OP_")}
        self.stnames = {v: k[3:] for k, v in self.values.items() if k.startswith("ST_")}
        self.menames = {v: k[3:] for k, v in self.values.items() if k.startswith("ME_")}
        self.flagnames = {v: k[2:] for k, v in self.values.items() if k.startswith("F_") and k != "F_RETIRED"}


C = _Consts(_H)
SLOTS = "ABCDEFGH"
U32 = 0xFFFFFFFF


def f32bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & U32))[0]


def pad32(n: int) -> int:
    return (n + 31) // 32 * 32


def pad8(n: int) -> int:
    return (n + 7) // 8 * 8


def pad4(n: int) -> int:
    return (n + 3) // 4 * 4


# ---------------------------------------------------------------- record
@dataclass
class Slot:
    addr: int = 0  # physical byte address, 64-bit field (validate: < ADDR_LIMIT)
    pitch: int = 0  # bytes between rows (u32)
    stride: int = 0  # bytes between items (u32)

    def zero(self) -> bool:
        return self.addr == 0 and self.pitch == 0 and self.stride == 0

    def __add__(self, off: int) -> Slot:  # same pitch / stride, address moved by off bytes
        return Slot(self.addr + off, self.pitch, self.stride)


@dataclass
class Cmd:
    op: int
    flags: int = 0
    tag: int = 0
    M: int = 0
    K: int = 0
    N: int = 0
    B: int = 1
    s: dict[str, Slot] = field(default_factory=dict)  # slot letter -> Slot; missing = zero
    p: dict[int, int] = field(default_factory=dict)  # parameter index -> u32 value; missing = 0
    note: str = ""  # builder annotation (not encoded)

    def slot(self, name: str) -> Slot:
        return self.s.get(name, Slot())

    def par(self, i: int) -> int:
        return self.p.get(i, 0) & U32


def encode(c: Cmd) -> bytes:
    f = [0] * C.REC_FIELDS
    f[C.FLD_OP] = (c.op & 0xFF) | (C.REC_MAGIC << 8) | ((c.flags & 0xFFFF) << 16)
    f[C.FLD_TAG] = c.tag & U32
    f[C.FLD_M], f[C.FLD_K], f[C.FLD_N], f[C.FLD_B] = c.M, c.K, c.N, c.B
    f[C.FLD_VERSION] = C.CONTRACT_VERSION
    for i, name in enumerate(SLOTS):
        sl = c.slot(name)
        base = C.SLOT_BASE + C.SLOT_FIELDS * i
        f[base : base + 4] = [sl.addr & U32, (sl.addr >> 32) & U32, sl.pitch & U32, sl.stride & U32]
    for i, v in c.p.items():
        f[C.PARAM_BASE + i] = v & U32
    x = C.CHECK_SEED
    for v in f[: C.CHECK_FIELD]:
        x ^= v
    f[C.CHECK_FIELD] = x
    return struct.pack("<64I", *f)


def decode(rec: bytes) -> tuple[int, Cmd | None]:
    """(ST_END, Cmd) or (ST_BAD_MAGIC | ST_BAD_CHECK | ST_BAD_VERSION | ST_BAD_OPCODE, Cmd | None)."""
    f = struct.unpack("<64I", rec)
    if (f[C.FLD_OP] >> 8) & 0xFF != C.REC_MAGIC:
        return C.ST_BAD_MAGIC, None
    x = C.CHECK_SEED
    for v in f[: C.CHECK_FIELD]:
        x ^= v
    if x != f[C.CHECK_FIELD]:
        return C.ST_BAD_CHECK, None
    if f[C.FLD_VERSION] != C.CONTRACT_VERSION:
        return C.ST_BAD_VERSION, None
    s = {}
    for i, name in enumerate(SLOTS):
        b = C.SLOT_BASE + C.SLOT_FIELDS * i
        s[name] = Slot(f[b] | (f[b + 1] << 32), f[b + 2], f[b + 3])
    c = Cmd(
        f[C.FLD_OP] & 0xFF,
        f[C.FLD_OP] >> 16,
        f[C.FLD_TAG],
        f[C.FLD_M],
        f[C.FLD_K],
        f[C.FLD_N],
        f[C.FLD_B] or 1,
        s,
        {i: f[C.PARAM_BASE + i] for i in range(C.NPARAMS) if f[C.PARAM_BASE + i]},
    )
    if c.op not in C.opnames:
        return C.ST_BAD_OPCODE, c
    return C.ST_END, c


def encode_list(cmds: list[Cmd]) -> bytes:
    return b"".join(encode(c) for c in cmds)


def decode_list(blob: bytes) -> list[tuple[int, Cmd | None]]:
    return [decode(blob[i : i + C.REC_BYTES]) for i in range(0, len(blob), C.REC_BYTES)]


# ---------------------------------------------------------------- validation (= ctr::validate in contract.h)
def _aligned(s: Slot) -> bool:
    return s.addr % C.ALIGN == 0 and s.pitch % C.ALIGN == 0 and s.stride % C.ALIGN == 0


def _end(s: Slot, b: int, rows: int, row_bytes: int) -> int:
    return s.addr + (b - 1) * s.stride + (rows - 1) * s.pitch + row_bytes


def _check_slot(
    s: Slot, b: int, rows: int, row_bytes: int, min_pitch: int, pitch0_ok: bool, full: bool
) -> int:
    if not _aligned(s):
        return C.ME_ALIGN
    if rows > 1 and not (pitch0_ok and s.pitch == 0) and s.pitch < min_pitch:
        return C.ME_ALIGN
    if s.addr >= C.ADDR_LIMIT or (full and _end(s, b, rows, row_bytes) > C.ADDR_LIMIT):
        return C.ME_ADDR
    return C.ME_OK


def _i32(v: int) -> int:
    v &= U32
    return v - (1 << 32) if v >> 31 else v


def validate(c: Cmd, full: bool = True) -> int:  # noqa: C901 (one rule per line, mirrors contract.h)
    """ctr::validate (full=True) or ctr::validate_hw (full=False: no extent checks)."""
    B, N, K, M = c.B, c.N, c.K, c.M
    p = [c.par(i) for i in range(C.NPARAMS)]
    S = {n: c.slot(n) for n in SLOTS}
    if c.flags & C.F_RETIRED:
        return C.ME_FLAGS
    allowed = {
        C.OP_NOP: C.F_YIELD,
        C.OP_END: C.F_YIELD,
        C.OP_GEMM: C.FLAGS_GEMM,
        C.OP_QUANT_X: C.FLAGS_QUANT,
        C.OP_QUANT_W: C.FLAGS_QUANT,
        C.OP_QUANT_WT: C.FLAGS_QUANT,
        C.OP_NORM: C.FLAGS_NORM,
        C.OP_GELU: C.FLAGS_ACT,
        C.OP_SILU_MUL: C.FLAGS_ACT,
        C.OP_ROPE: C.FLAGS_ROPE,
        C.OP_SOFTMAX: C.FLAGS_SOFTMAX,
        C.OP_COPY: C.FLAGS_COPY,
    }.get(c.op, 0)
    if c.flags & ~allowed:
        return C.ME_FLAGS
    if c.op in (C.OP_NOP, C.OP_END):
        if any(not s.zero() for s in S.values()):
            return C.ME_SLOT
        return C.ME_PARAM if any(p) else C.ME_OK
    if N < 1 or N > C.N_MAX or B < 1 or B > C.B_MAX:
        return C.ME_DIMS
    used = set()
    if c.op == C.OP_GEMM:
        acc, addd, bias = c.flags & C.F_ACC, c.flags & C.F_ADD_D, c.flags & C.F_BIAS
        if acc and addd:
            return C.ME_FLAGS
        if M < C.ENGINE_LANES or M > C.GEMM_M_MAX or M % C.ENGINE_LANES or K < 8 or K > C.GEMM_K_MAX or K % 8:
            return C.ME_DIMS
        R, P1 = p[C.P_SEGC], p[C.P_SEGF]
        if R > N or (R == 0 and P1 != 0) or P1 % C.ALIGN:
            return C.ME_PARAM
        act = p[C.P_ACT]
        if act > C.EPI_MAX or (act and R):  # no C row groups with an activation
            return C.ME_PARAM
        if act and (acc or (act == C.EPI_SILU_MUL and addd)):
            return C.ME_FLAGS
        MO = M // 2 if act == C.EPI_SILU_MUL else M  # output columns
        if any(v for i, v in enumerate(p) if i not in (C.P_SEGC, C.P_SEGF, C.P_ACT)):
            return C.ME_PARAM
        used = {"A", "B", "C", "E", "F"}
        if r := _check_slot(S["A"], B, N, 2 * K, 2 * K, False, full):
            return r
        port = (M // C.ENGINE_LANES) * K * 16
        if not _aligned(S["B"]) or S["B"].pitch < port:
            return C.ME_ALIGN
        if S["B"].addr >= C.ADDR_LIMIT or (full and _end(S["B"], B, C.ENGINE_PORTS, port) > C.ADDR_LIMIT):
            return C.ME_ADDR
        y = S["C"]
        if not _aligned(y) or (N > 1 and R != 1 and y.pitch < 4 * MO):
            return C.ME_ALIGN
        in_grp, ngrp = (min(N, R), (N - 1) // R) if R else (N, 0)
        yend = y.addr + (B - 1) * y.stride + ngrp * P1 + (in_grp - 1) * y.pitch + 4 * MO
        if y.addr >= C.ADDR_LIMIT or (full and yend > C.ADDR_LIMIT):
            return C.ME_ADDR
        if addd:
            used.add("D")
            if r := _check_slot(S["D"], B, N, 4 * M, 4 * M, True, full):
                return r
        if r := _check_slot(S["E"], B, 1, 4 * M, 0, True, full):
            return r
        if r := _check_slot(S["F"], B, 1, 4 * pad4(N), 0, True, full):
            return r
        if bias:
            used.add("G")
            if r := _check_slot(S["G"], B, 1, 4 * M, 0, True, full):
                return r
    elif c.op == C.OP_SOFTMAX:
        if K < 8 or K > C.SMX_L_MAX or K % 8:
            return C.ME_DIMS
        NP = p[C.P_NP] or N
        if NP < N or NP > C.N_MAX:
            return C.ME_PARAM
        mul = bits_f32(p[C.P_EPS])
        if not (0.0 < mul < 1e30):
            return C.ME_PARAM
        if any(v for i, v in enumerate(p) if i not in (C.P_EPS, C.P_NP)):
            return C.ME_PARAM
        if not c.flags & C.F_QOUT:
            return C.ME_FLAGS
        used = {"A", "C", "F"}
        for r in (
            _check_slot(S["A"], B, N, 4 * K, 4 * K, False, full),
            _check_slot(S["C"], B, NP, 2 * K, 2 * K, False, full),
            _check_slot(S["F"], B, 1, 4 * pad4(NP), 0, True, full),
        ):
            if r:
                return r
        if c.flags & C.F_MASK:
            used |= {"D", "E"}
            if r := _check_slot(S["D"], B, 1, 4 * pad4(N), 0, True, full):
                return r
            if r := _check_slot(S["E"], B, 1, 4 * K, 0, True, full):
                return r
    else:  # row unit
        if p[C.P_SEG] not in (0, C.SEG_HEAD):
            return C.ME_DIMS
        SEG = C.SEG_HEAD if p[C.P_SEG] else K
        if K < 8 or K % 8 or (p[C.P_SEG] and (K % C.SEG_HEAD or K // C.SEG_HEAD > C.SEG_MAX)):
            return C.ME_DIMS
        nseg, NP, segc, segf, sh = K // SEG, p[C.P_NP] or N, p[C.P_SEGC], p[C.P_SEGF], _i32(p[C.P_SHIFT])
        if not -C.SHIFT_MAX <= sh <= C.SHIFT_MAX:
            return C.ME_PARAM
        if segc % C.ALIGN or segf % C.ALIGN or (nseg == 1 and (segc or segf)):
            return C.ME_PARAM
        if any(p[C.P_NP + 1 :]):
            return C.ME_PARAM
        q = bool(c.flags & C.F_QOUT)
        used = {"A", "C"}
        if c.op == C.OP_COPY:
            if K % 4 or K > C.ROW_K_MAX or nseg != 1 or NP != N or sh or p[C.P_EPS]:
                return C.ME_DIMS
            for r in (
                _check_slot(S["A"], B, N, 4 * K, 4 * K, False, full),
                _check_slot(S["C"], B, N, 4 * K, 4 * K, False, full),
            ):
                if r:
                    return r
        else:
            if p[C.P_EPS] and c.op != C.OP_NORM:
                return C.ME_PARAM
            if NP < N or NP > C.N_MAX:
                return C.ME_PARAM
            if r := _check_slot(S["A"], B, N, 4 * K, 4 * K, False, full):
                return r
            kw = c.op == C.OP_QUANT_W or (c.op == C.OP_ROPE and c.flags & C.F_KW)
            if c.op in (C.OP_QUANT_W, C.OP_QUANT_WT) and q:
                return C.ME_FLAGS
            if c.op == C.OP_ROPE and q and c.flags & C.F_KW:
                return C.ME_FLAGS
            scales = q or kw or c.op == C.OP_QUANT_WT
            if c.op == C.OP_QUANT_X and not q:
                return C.ME_FLAGS
            if c.op == C.OP_NORM:
                if K > C.NORM_K_MAX or nseg != 1:
                    return C.ME_DIMS
                if not (0.0 < bits_f32(p[C.P_EPS]) < 1.0):
                    return C.ME_PARAM
                used.add("E")
                if r := _check_slot(S["E"], 1, 1, (8 if c.flags & C.F_LN else 4) * K, 0, True, full):
                    return r
            elif c.op in (C.OP_GELU, C.OP_SILU_MUL):
                if K > C.ACT_K_MAX or nseg != 1:
                    return C.ME_DIMS
                if c.op == C.OP_GELU and c.flags & C.F_NOMUL:
                    return C.ME_FLAGS
                if c.op == C.OP_SILU_MUL and not c.flags & C.F_NOMUL:
                    used.add("B")
                    if r := _check_slot(S["B"], B, N, 4 * K, 4 * K, False, full):
                        return r
            elif c.op == C.OP_ROPE:
                if SEG != C.ROPE_SEG:
                    return C.ME_DIMS
                used.add("E")
                if r := _check_slot(S["E"], B, N, C.ROPE_TAB_BYTES, C.ROPE_TAB_BYTES, False, full):
                    return r
            elif K > C.ROW_K_MAX:
                return C.ME_DIMS
            if not scales and (sh or NP != N):
                return C.ME_PARAM
            o = S["C"]
            if not _aligned(o):
                return C.ME_ALIGN
            if kw:
                if NP % C.ENGINE_LANES:
                    return C.ME_DIMS
                port = (NP // C.ENGINE_LANES) * SEG * 16
                if o.pitch < port or (nseg > 1 and segc < port):
                    return C.ME_ALIGN
                end = o.addr + (B - 1) * o.stride + (nseg - 1) * segc + 3 * o.pitch + port
            elif c.op == C.OP_QUANT_WT:
                if SEG % C.ENGINE_LANES or NP % 8:
                    return C.ME_DIMS
                port = (SEG // C.ENGINE_LANES) * NP * 16
                if o.pitch < port or (nseg > 1 and segc < port):
                    return C.ME_ALIGN
                end = o.addr + (B - 1) * o.stride + (nseg - 1) * segc + 3 * o.pitch + port
            else:
                eb = 2 if q else 4
                seg_row = SEG * eb if nseg > 1 else K * eb
                if NP > 1 and o.pitch < seg_row:
                    return C.ME_ALIGN
                if nseg > 1 and segc < seg_row:
                    return C.ME_ALIGN
                end = o.addr + (B - 1) * o.stride + (nseg - 1) * segc + (NP - 1) * o.pitch + seg_row
            if o.addr >= C.ADDR_LIMIT or (full and end > C.ADDR_LIMIT):
                return C.ME_ADDR
            if scales:
                used.add("F")
                nsc = SEG if c.op == C.OP_QUANT_WT else NP
                f = S["F"]
                if not _aligned(f):
                    return C.ME_ALIGN
                if nseg > 1 and segf < 4 * pad4(nsc):
                    return C.ME_ALIGN
                fend = f.addr + (B - 1) * f.stride + (nseg - 1) * segf + 4 * pad4(nsc)
                if f.addr >= C.ADDR_LIMIT or (full and fend > C.ADDR_LIMIT):
                    return C.ME_ADDR
            elif segf:
                return C.ME_PARAM
    for n in SLOTS:
        if n not in used and not S[n].zero():
            return C.ME_SLOT
    return C.ME_OK


def validate_list(cmds: list[Cmd]) -> list[tuple[int, int]]:
    """(index, ME code) of every command that would stop the IP with ST_MODE_ERROR; [] = clean."""
    return [(i, r) for i, c in enumerate(cmds) if (r := validate(c))]


# ---------------------------------------------------------------- registers and status
def start_regs(list_base: int, status_addr: int, first: int = 0, step: bool = False) -> dict[int, int]:
    """Register writes for one list run (offset -> u32), ap_start last. Pointer registers are 0."""
    assert list_base % 16 == 0 and status_addr % 16 == 0 and 0 <= first <= C.MAX_INDEX
    assert list_base >> 4 <= U32 and status_addr >> 4 <= U32
    regs = {o: 0 for o in (C.REG_W0, C.REG_W1, C.REG_W2, C.REG_W3, C.REG_XMEM, C.REG_YMEM)}
    regs |= {o + 4: 0 for o in list(regs)}
    regs[C.REG_M] = (C.MODE_MAGIC << 24) | (int(step) << C.MODE_STEP_BIT) | first
    regs[C.REG_K] = list_base >> 4
    regs[C.REG_N] = status_addr >> 4
    return regs


def parse_return(v: int) -> tuple[int, str]:
    """ap_return -> (next index, status name)."""
    return v >> 8, C.stnames.get(v & 0xFF, f"0x{v & 0xFF:02x}")


def parse_status(blob: bytes) -> dict:
    w = struct.unpack("<12I", blob[: C.STATUS_BYTES])
    return {
        "code": C.stnames.get(w[0], hex(w[0])),
        "next": w[1],
        "tag": w[2],
        "done": w[3],
        "mode_error": C.menames.get(w[4], hex(w[4])),
        "magic_ok": w[5] == C.STATUS_MAGIC,
        "starts": w[6],
        "last_op": C.opnames.get(w[7], hex(w[7])),
        "version": w[8],
        "last_finished": w[9],
        "cycles": w[10] | (w[11] << 32),
    }


# ---------------------------------------------------------------- layouts
def row_addr(s: Slot, b: int, n: int) -> int:
    return s.addr + b * s.stride + n * s.pitch


def row_addr2(s: Slot, b: int, t: int, R: int, P1: int) -> int:
    return row_addr(s, b, t) if R == 0 else s.addr + b * s.stride + (t // R) * P1 + (t % R) * s.pitch


def seg_addr(s: Slot, b: int, n: int, j: int, seg_bytes: int) -> int:
    return s.addr + b * s.stride + j * seg_bytes + n * s.pitch


def scale_addr(f: Slot, b: int, j: int, segf: int, n: int) -> int:
    return f.addr + b * f.stride + j * segf + 4 * n


def w_word_addr(w: Slot, b: int, m: int, k: int, kblob: int) -> int:
    lanes = C.ENGINE_LANES
    return w.addr + b * w.stride + ((m % lanes) // 8) * w.pitch + ((m // lanes) * kblob + k) * 16


def w_lane(m: int) -> int:
    return m % 8


def pack_w(wq: np.ndarray) -> np.ndarray:
    """int16 [Mp][K] -> the 4 port arrays [4][Mp/32][K][8] (uint16), port p = rows 8p..8p+7 of each group.
    The same bytes as fpga/hls/pack_float.pack of the int16 bit patterns (runtime.chunk_layout)."""
    mp, k = wq.shape
    assert mp % C.ENGINE_LANES == 0
    v = wq.astype(np.int16).view(np.uint16).reshape(mp // 32, 4, 8, k)  # [g][p][lane][k]
    return np.ascontiguousarray(v.transpose(1, 0, 3, 2))  # [p][g][k][lane]


# ---------------------------------------------------------------- reference arithmetic
TINY = np.float32(bits_f32(C.TINY_BITS))


def quant_rows(x: np.ndarray, shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Contract quantizer over the last axis: s = fl32(max(amax, 1e-30) / 32767) (1.0 for an all-zero row),
    q = clamp(rne(fl32(x / s)), +-32767); the stored scale is s * 2^shift (zero rows keep 1.0)."""
    x = np.ascontiguousarray(x, np.float32)
    a = np.abs(x).max(axis=-1, keepdims=True)
    s = np.maximum(a, TINY) / np.float32(C.QMAX)
    zero = a == 0
    s = np.where(zero, np.float32(1.0), s).astype(np.float32)
    q = np.clip(np.rint(x / s), -C.QMAX, C.QMAX).astype(np.int16)
    stored = np.where(zero, np.float32(1.0), np.ldexp(s, shift).astype(np.float32)).astype(np.float32)
    return q, stored[..., 0]


def ftz(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32)
    return np.where((u >> 23) & 0xFF == 0, u & np.uint32(0x80000000), u).astype(np.uint32).view(np.float32)


def gemm_out(acc: np.ndarray, s_row: np.ndarray, s_tok: np.ndarray, bias=None, addend=None) -> np.ndarray:
    """Epilogue: y = ((fl(acc) * s_row[m]) * s_tok[n]) [+ bias[m]] [+ addend[n][m]], one FP32 op per step,
    denormals flushed (sign kept). acc: int64 [N][M] exact sums."""
    with np.errstate(over="ignore"):
        y = ftz(ftz(acc.astype(np.float32)) * ftz(s_row[None, :]))
        y = ftz(y * ftz(s_tok[:, None]))
        if bias is not None:
            y = ftz(y + ftz(bias[None, :]))
        if addend is not None:
            y = ftz(y + ftz(addend))
    return y


# ---------------------------------------------------------------- self test + vectors for tb_contract.cpp
def _edge_rows(rng: np.random.Generator) -> list[np.ndarray]:
    rows = [
        rng.standard_normal(64, dtype=np.float32),
        (rng.standard_normal(768) * 50).astype(np.float32),
        np.zeros(16, np.float32),
        -np.zeros(16, np.float32),
        np.full(24, 3.7, np.float32),
        np.full(24, -1e-31, np.float32),  # below the 1e-30 floor
        np.array([1e-38, -2e-39, 5e-45, 0.0] * 4, np.float32),  # denormals
        np.array([3.4e38, -1e37, 1.0, 0.5] * 4, np.float32),
        (np.arange(-40, 40, dtype=np.float32) + 0.5) * np.float32(1 / 32767),  # ties around q = k + 0.5
        np.concatenate([np.ones(63, np.float32), np.float32([65534.0])]),  # one outlier
    ]
    for _ in range(30):
        n = int(rng.integers(8, 400))
        rows.append((rng.standard_t(3, n) * 10.0 ** rng.integers(-20, 20)).astype(np.float32))
    return rows


def _gemm_cmd(**kw) -> Cmd:
    M, K, N = kw.get("M", 64), kw.get("K", 64), kw.get("N", 50)
    port = (M // 32) * K * 16
    s = {
        "A": Slot(0x8_0000_0000, 2 * K),
        "B": Slot(0x8_1000_0000, port),
        "C": Slot(0x8_2000_0000, 4 * M),
        "E": Slot(0x8_3000_0000),
        "F": Slot(0x8_3000_1000),
    }
    s.update(kw.get("s", {}))
    return Cmd(C.OP_GEMM, kw.get("flags", 0), 7, M, K, N, kw.get("B", 1), s, kw.get("p", {}))


def _cases() -> list[tuple[str, Cmd]]:
    """Commands with a known validation result (each mutation breaks one rule)."""
    out = [("end", Cmd(C.OP_END)), ("nop_yield", Cmd(C.OP_NOP, C.F_YIELD))]
    out.append(("gemm_ok", _gemm_cmd()))
    out.append(
        (
            "gemm_bias_d",
            _gemm_cmd(flags=C.F_BIAS | C.F_ADD_D, s={"D": Slot(0x8_4000_0000, 0), "G": Slot(0x8_5000_0000)}),
        )
    )
    out.append(("gemm_acc_and_d", _gemm_cmd(flags=C.F_ACC | C.F_ADD_D, s={"D": Slot(0x8_4000_0000, 256)})))
    out.append(("gemm_m_not32", _gemm_cmd(M=48)))
    out.append(("gemm_k_not8", _gemm_cmd(K=81)))
    out.append(("gemm_k_4104", _gemm_cmd(K=4104)))
    out.append(("gemm_unaligned_a", _gemm_cmd(s={"A": Slot(0x8_0000_0008, 128)})))
    out.append(("gemm_short_pitch", _gemm_cmd(s={"C": Slot(0x8_2000_0000, 128)})))
    out.append(("gemm_high_addr_ok", _gemm_cmd(s={"C": Slot(0xFF_F000_0000, 256)})))
    out.append(("gemm_past_40bit", _gemm_cmd(s={"C": Slot(0xFF_FFFF_F000, 256)})))
    out.append(("gemm_addr_bit40", _gemm_cmd(s={"A": Slot(1 << 40, 128)})))
    out.append(
        (
            "gemm_rowgroups",
            _gemm_cmd(N=156, p={C.P_SEGC: 52, C.P_SEGF: 256}, s={"C": Slot(0x8_2000_0000, 3840, 768)}, B=5),
        )
    )
    out.append(("gemm_bad_R", _gemm_cmd(N=10, p={C.P_SEGC: 11})))
    out.append(("gemm_stray_slot", _gemm_cmd(s={"H": Slot(16)})))
    out.append(("gemm_retired_flag", _gemm_cmd(flags=0x0004)))
    out.append(("gemm_p0_set", _gemm_cmd(p={0: f32bits(0.125)})))
    q = Cmd(
        C.OP_QUANT_X,
        C.F_QOUT,
        1,
        0,
        768,
        512,
        1,
        {"A": Slot(0x6000_0000, 9216), "C": Slot(0x6100_0000, 128), "F": Slot(0x6200_0000)},
        {C.P_SEG: 64, C.P_SEGC: 512 * 128, C.P_SEGF: 2048, C.P_SHIFT: (-3) & U32},
    )
    out.append(("quant_x_seg", q))
    out.append(("quant_x_shift9", Cmd(q.op, q.flags, 1, 0, 768, 512, 1, dict(q.s), q.p | {C.P_SHIFT: 9})))
    out.append(("quant_x_noqout", Cmd(q.op, 0, 1, 0, 768, 512, 1, dict(q.s), dict(q.p))))
    out.append(
        ("quant_x_seg_overlap_f", Cmd(q.op, q.flags, 1, 0, 768, 512, 1, dict(q.s), q.p | {C.P_SEGF: 1024}))
    )
    out.append(
        (
            "quant_x_k12288",
            Cmd(
                C.OP_QUANT_X,
                C.F_QOUT,
                0,
                0,
                12288,
                32,
                1,
                {"A": Slot(0x7000_0000, 49152), "C": Slot(0x7100_0000, 24576), "F": Slot(0x7200_0000)},
            ),
        )
    )
    kw = Cmd(
        C.OP_QUANT_W,
        0,
        2,
        0,
        768,
        512,
        1,
        {"A": Slot(0x6000_0000 + 3072, 9216), "C": Slot(0x6300_0000, 16 * 1024 * 12), "F": Slot(0x6400_0000)},
        {C.P_SEG: 64, C.P_SEGC: 16 * 1024, C.P_SEGF: 2048, C.P_NP: 512},
    )
    out.append(("quant_w_kw", kw))
    out.append(("quant_w_np_not32", Cmd(kw.op, 0, 2, 0, 768, 500, 1, dict(kw.s), kw.p | {C.P_NP: 500})))
    vw = Cmd(
        C.OP_QUANT_WT,
        0,
        3,
        0,
        768,
        256,
        2,
        {
            "A": Slot(0x6000_0000 + 6144, 9216, 256 * 9216),
            "C": Slot(0x6500_0000, 16384 * 12, 8192),
            "F": Slot(0x6600_0000, 0, 256),
        },
        {C.P_SEG: 64, C.P_SEGC: 16384, C.P_SEGF: 512, C.P_NP: 256},
    )
    out.append(("quant_wt_vw", vw))
    norm = Cmd(
        C.OP_NORM,
        C.F_QOUT | C.F_LN,
        4,
        0,
        768,
        512,
        1,
        {
            "A": Slot(0x6700_0000, 3072),
            "C": Slot(0x6800_0000, 1536),
            "E": Slot(0x6900_0000),
            "F": Slot(0x6A00_0000),
        },
        {C.P_EPS: f32bits(1e-6)},
    )
    out.append(("norm_ln", norm))
    out.append(("norm_eps0", Cmd(norm.op, norm.flags, 4, 0, 768, 512, 1, dict(norm.s), {})))
    out.append(
        (
            "norm_k1032",
            Cmd(
                norm.op,
                norm.flags,
                4,
                0,
                1032,
                512,
                1,
                dict(norm.s) | {"A": Slot(0x6700_0000, 4128), "C": Slot(0x6800_0000, 2064)},
                dict(norm.p),
            ),
        )
    )
    smx = Cmd(
        C.OP_SOFTMAX,
        C.F_QOUT | C.F_MASK,
        5,
        0,
        128,
        156,
        5,
        {
            "A": Slot(0x6B00_0000, 512, 156 * 512),
            "C": Slot(0x6C00_0000, 256, 156 * 256),
            "F": Slot(0x6D00_0000, 0, 624),
            "D": Slot(0x6E00_0000),
            "E": Slot(0x6E00_1000),
        },
        {C.P_EPS: f32bits(1.0)},
    )
    out.append(("softmax_mask", smx))
    out.append(("softmax_l264", Cmd(smx.op, smx.flags, 5, 0, 264, 156, 5, dict(smx.s), dict(smx.p))))
    out.append(("softmax_fp32out", Cmd(smx.op, C.F_MASK, 5, 0, 128, 156, 5, dict(smx.s), dict(smx.p))))
    rope = Cmd(
        C.OP_ROPE,
        C.F_QOUT,
        6,
        0,
        960,
        50,
        1,
        {
            "A": Slot(0x6F00_0000, 6400),
            "C": Slot(0x7300_0000, 128),
            "E": Slot(0x7400_0000, 256),
            "F": Slot(0x7500_0000),
        },
        {C.P_SEG: 64, C.P_SEGC: 52 * 128, C.P_SEGF: 208, C.P_SHIFT: (-3) & U32, C.P_NP: 52},
    )
    out.append(("rope_q", rope))
    out.append(
        ("rope_seg32", Cmd(rope.op, rope.flags, 6, 0, 960, 50, 1, dict(rope.s), rope.p | {C.P_SEG: 32}))
    )
    out.append(
        ("rope_kw_and_q", Cmd(rope.op, C.F_QOUT | C.F_KW, 6, 0, 960, 50, 1, dict(rope.s), dict(rope.p)))
    )
    act = Cmd(
        C.OP_SILU_MUL,
        C.F_QOUT,
        8,
        0,
        2048,
        50,
        1,
        {
            "A": Slot(0x7600_0000, 16384),
            "B": Slot(0x7600_2000, 16384),
            "C": Slot(0x7700_0000, 4096),
            "F": Slot(0x7800_0000),
        },
    )
    out.append(("silu_mul", act))
    out.append(("silu_nomul_stray_b", Cmd(act.op, C.F_QOUT | C.F_NOMUL, 8, 0, 2048, 50, 1, dict(act.s), {})))
    cp = Cmd(
        C.OP_COPY,
        0,
        9,
        0,
        3072,
        4,
        8,
        {"A": Slot(0x7900_0000, 12288, 196608), "C": Slot(0x7A00_0000, 49152, 196608)},
    )
    out.append(("copy_shuffle", cp))
    out.append(("copy_flag_q", Cmd(cp.op, C.F_QOUT, 9, 0, 3072, 4, 8, dict(cp.s), {})))
    out.append(
        (
            "gelu_n0",
            Cmd(
                C.OP_GELU,
                C.F_QOUT,
                10,
                0,
                3072,
                0,
                1,
                {"A": Slot(0x100, 12288), "C": Slot(0x10000, 6144), "F": Slot(0x20000)},
            ),
        )
    )
    out.append(
        (
            "gelu_n65536",
            Cmd(
                C.OP_GELU,
                C.F_QOUT,
                10,
                0,
                3072,
                65536,
                1,
                {"A": Slot(0x100, 12288), "C": Slot(0x10000, 6144), "F": Slot(0x20000)},
            ),
        )
    )
    return out


EXPECT = {  # case -> expected ME code name
    "end": "OK", "nop_yield": "OK", "gemm_ok": "OK", "gemm_bias_d": "OK", "gemm_acc_and_d": "FLAGS",
    "gemm_m_not32": "DIMS", "gemm_k_not8": "DIMS", "gemm_k_4104": "DIMS", "gemm_unaligned_a": "ALIGN",
    "gemm_short_pitch": "ALIGN", "gemm_high_addr_ok": "OK", "gemm_past_40bit": "ADDR",
    "gemm_addr_bit40": "ADDR",
    "gemm_rowgroups": "OK", "gemm_bad_R": "PARAM", "gemm_stray_slot": "SLOT", "gemm_retired_flag": "FLAGS",
    "gemm_p0_set": "PARAM", "quant_x_seg": "OK", "quant_x_shift9": "PARAM", "quant_x_noqout": "FLAGS",
    "quant_x_seg_overlap_f": "ALIGN", "quant_x_k12288": "OK", "quant_w_kw": "OK", "quant_w_np_not32": "DIMS",
    "quant_wt_vw": "OK", "norm_ln": "OK", "norm_eps0": "PARAM", "norm_k1032": "DIMS", "softmax_mask": "OK",
    "softmax_l264": "DIMS", "softmax_fp32out": "FLAGS", "rope_q": "OK", "rope_seg32": "DIMS",
    "rope_kw_and_q": "FLAGS", "silu_mul": "OK", "silu_nomul_stray_b": "SLOT", "copy_shuffle": "OK",
    "copy_flag_q": "FLAGS", "gelu_n0": "DIMS", "gelu_n65536": "DIMS",
}  # fmt: skip


def selftest(vec_dir: Path | None) -> int:
    rng = np.random.default_rng(12392)
    fails = 0
    # 1. record round trip and record-level faults
    cases = _cases()
    for name, c in cases:
        st, d = decode(encode(c))
        if st != C.ST_END or encode(d) != encode(c):
            print(f"FAIL roundtrip {name}: {C.stnames.get(st)}")
            fails += 1
        got, hw = C.menames[validate(c)], C.menames[validate(c, False)]
        if got != EXPECT[name] or (hw != got and not (got == "ADDR" and hw == "OK")):
            print(f"FAIL validate {name}: {got} (hw {hw}), expected {EXPECT[name]}")
            fails += 1
    good = bytearray(encode(cases[2][1]))
    bad = {
        "BAD_MAGIC": bytes(good[:1] + b"\xc5" + good[2:]),
        "BAD_CHECK": bytes(good[:20] + bytes([good[20] ^ 1]) + good[21:]),
    }
    f = list(struct.unpack("<64I", bytes(good)))
    f[C.FLD_VERSION] ^= 1
    f[C.CHECK_FIELD] ^= 1
    bad["BAD_VERSION"] = struct.pack("<64I", *f)
    f = list(struct.unpack("<64I", bytes(good)))
    f[0] = (f[0] & ~0xFF) | 0x71
    f[C.CHECK_FIELD] ^= 0x10 ^ 0x71
    bad["BAD_OPCODE"] = struct.pack("<64I", *f)
    for want, rec in bad.items():
        st, _ = decode(rec)
        if C.stnames[st] != want:
            print(f"FAIL decode {want}: got {C.stnames[st]}")
            fails += 1
    # 2. quantizer: against fpga/hls/pack_float (the engine's host packer) where importable
    rows = _edge_rows(rng)
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hls"))
        import pack_float as pf

        for r in rows:
            q, s = quant_rows(r[None, :])
            q2, s2 = pf.quantize_int16_tokens(r[None, :])
            if not (
                np.array_equal(q, np.asarray(q2).reshape(q.shape))
                and np.array_equal(
                    s.view(np.uint32), np.asarray(s2, np.float32).reshape(s.shape).view(np.uint32)
                )
            ):
                print(f"FAIL quant vs pack_float on a row of {r.size} (amax {np.abs(r).max():.3g})")
                fails += 1
        print(f"quantizer == pack_float.quantize_int16_tokens on {len(rows)} rows")
    except ImportError as e:
        print(f"pack_float not importable ({e}); quantizer cross-check skipped")
    # 3. weight tiling helper == pack_w
    wq = rng.integers(-32767, 32768, (64, 24)).astype(np.int16)
    pw = pack_w(wq).reshape(-1)
    wsl = Slot(0, (64 // 32) * 24 * 16)
    for m, k in [(0, 0), (9, 3), (31, 23), (32, 0), (63, 17)]:
        word = w_word_addr(wsl, 0, m, k, 24) // 2  # in uint16 units
        if pw[word + w_lane(m)] != np.uint16(wq[m, k].view(np.uint16)):
            print(f"FAIL w_word_addr m={m} k={k}")
            fails += 1
    if "pf" in locals() and not np.array_equal(pw, pf.pack(wq.view(np.uint16))[0]):
        print("FAIL pack_w != pack_float.pack")
        fails += 1
    # 4. vectors for the C++ testbench
    if vec_dir:
        vec_dir.mkdir(parents=True, exist_ok=True)
        (vec_dir / "cases.bin").write_bytes(encode_list([c for _, c in cases]) + b"".join(bad.values()))
        meta = [
            {"name": n, "status": "END", "me": EXPECT[n], "hw": C.menames[validate(c, False)]}
            for n, c in cases
        ]
        meta += [{"name": k.lower(), "status": k, "me": "", "hw": ""} for k in bad]
        (vec_dir / "cases.json").write_text(json.dumps(meta, indent=0))
        with open(vec_dir / "quant.bin", "wb") as fh:
            for i, r in enumerate(rows):
                sh = int(i % 3) - 1
                q, s = quant_rows(r[None, :], sh)
                fh.write(struct.pack("<Ii", r.size, sh) + r.tobytes() + q.tobytes() + s.tobytes())
        acc = rng.integers(-(2**41), 2**41, (8, 16)).astype(np.int64)
        acc[0, :4] = [0, 1, -1, 2**24 + 1]
        sr = (rng.random(16) * 1e-3).astype(np.float32)
        sr[1] = 1e-38  # denormal product path
        st = (rng.random(8) * 1e-2).astype(np.float32)
        bias = rng.standard_normal(16).astype(np.float32)
        add = rng.standard_normal((8, 16)).astype(np.float32)
        y = gemm_out(acc, sr, st, bias, add)
        (vec_dir / "epilogue.bin").write_bytes(
            acc.tobytes() + sr.tobytes() + st.tobytes() + bias.tobytes() + add.tobytes() + y.tobytes()
        )
        print(f"vectors: {len(meta)} records, {len(rows)} quant rows, 8x16 epilogue -> {vec_dir}")
    print("contract.py selftest:", "PASS" if fails == 0 else f"FAILED ({fails})")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--vectors", type=Path, help="write the C++ testbench vectors here")
    a = ap.parse_args()
    if a.selftest or a.vectors:
        return selftest(a.vectors)
    print(
        f"contract {C.CONTRACT_VERSION >> 16}.{C.CONTRACT_VERSION & 0xFFFF}: "
        f"{len(C.values)} constants from {_H.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
