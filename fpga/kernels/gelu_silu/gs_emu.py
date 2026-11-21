"""Bit-exact torch emulation of the gelu_silu kernel (fpga/kernels/gelu_silu/gelu_silu.h), int64 tensor arithmetic.

    y = act(x)            mode 0: tanh-GELU(x);  mode 1: SiLU(gate) * up;  mode 2: SiLU(x)
    (q, s) = quant(y)     per token (row): s = FP32 scale, q int16 with q * s ~= y, |q| <= 32767

Same integer steps as the HLS core (see gelu_silu.h for the commented spec); test_vectors() writes binary inputs and
the expected outputs for the C++ testbench, which must match bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import gs_tables

FB, VB, DB = gs_tables.FB, gs_tables.VB, gs_tables.DB
QMAX = 32767

_TAB: dict[int, torch.Tensor] = {}


def tab(nsb: int) -> torch.Tensor:
    if nsb not in _TAB:
        _TAB[nsb] = torch.from_numpy(gs_tables.cached_table(nsb).astype(np.int64))
    return _TAB[nsb]


def f2b(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> its bits as non-negative int64."""
    return x.detach().to(torch.float32).contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def b2f(b: torch.Tensor) -> torch.Tensor:
    b = b.to(torch.int64)
    b = torch.where(b >= (1 << 31), b - (1 << 32), b)
    return b.to(torch.int32).view(torch.float32)


def msb(v: torch.Tensor) -> torch.Tensor:
    """Index of the leading one of positive int64 values < 2^53 (exact through float64 frexp); -1 for 0."""
    _, e = torch.frexp(v.to(torch.float64))
    return torch.where(v > 0, e.to(torch.int64) - 1, torch.full_like(v, -1))


def pass1(xb: torch.Tensor, ub: torch.Tensor | None, mode: int, nsb: int = 9) -> torch.Tensor:
    """Per element: FP32 bits of y, as the hardware rounds them.
    mode 0: y = x * Phi_tanh(x) (tanh-GELU); mode 1: y = x * sigmoid(x) * u; mode 2: y = x * sigmoid(x)."""
    assert mode in (0, 1, 2)
    nseg = 1 << nsb
    TB = nsb + FB
    tsel = 0 if mode == 0 else 1  # table: GELU or SiLU
    rb = 3 if mode == 0 else 4
    sx = (xb >> 31) & 1
    ex = (xb >> 23) & 0xFF
    xz = ex == 0
    mx = torch.where(xz, torch.zeros_like(xb), (xb & 0x7FFFFF) | (1 << 23))
    sh = 150 + rb - TB - ex
    sat = (~xz) & (sh < 24 - TB)
    inr = (~xz) & (sh >= 24 - TB) & (sh < 24)
    T = torch.where(inr, mx >> sh.clamp(0, 63), torch.zeros_like(mx))
    idx = T >> FB
    fr = T & ((1 << FB) - 1)
    w = tab(nsb)[tsel * nseg + idx]
    V, D = w >> DB, w & ((1 << DB) - 1)
    g21 = (V - ((D * fr + (1 << (FB - 1))) >> FB)).clamp_min(0)
    g21 = torch.where(sat, torch.zeros_like(g21), g21)
    # x >= 0: |x| (1 - g) with g in Q0.18 (17 bits: one DSP, 24 x 17); 1 - g >= 1/2, so this is relative 2^-18
    g18 = ((g21 + 4) >> 3).clamp_max((1 << 17) - 1)
    # x < 0: |x| g with g21 normalised to a 17-bit significand (leading one at bit 16): relative precision of
    # g21 kept for small g (the SiLU / GELU tails), same DSP
    pz = 20 - msb(g21)  # leading zeros of the 21-bit g21 (21 for 0)
    gn = ((g21 << pz.clamp(0, 21)) + 8) >> 4  # [2^16, 2^17], 0 for g21 == 0
    cn = gn >= (1 << 17)
    gn = torch.where(cn, torch.full_like(gn, 1 << 16), gn)
    pz = torch.where(cn, pz - 1, pz)
    G = torch.where(sx == 1, gn, g18)
    P = mx * G  # 41 bits
    Y = torch.where(sx == 1, P, (mx << 18) - P)  # x >= 0: [2^40, 2^42); x < 0: [2^39, 2^41) or 0
    lz = 41 - msb(Y)  # 0, 1, 2 (42 when Y == 0)
    Yn = Y << lz.clamp(0, 63)  # bit 41 set (0 when Y == 0)
    e0 = ex - lz + torch.where(sx == 1, 1 - pz, torch.zeros_like(pz))
    if mode != 1:
        m = (Yn + (1 << 17)) >> 18  # 24 bits + carry, round half up
        carry = m >= (1 << 24)
        m = torch.where(carry, m >> 1, m)
        ey = torch.where(carry, e0 + 1, e0)
        zero = (Y == 0) | (ey <= 0)
        return torch.where(zero, torch.zeros_like(xb), (sx << 31) | (ey << 23) | (m & 0x7FFFFF))
    # SiLU-gated: x sigmoid(x) rounded to a 17-bit significand, times the 24-bit significand of up: one DSP
    m17 = (Yn + (1 << 24)) >> 25  # [2^16, 2^17]
    c17 = m17 >= (1 << 17)
    m17 = torch.where(c17, torch.full_like(m17, 1 << 16), m17)
    e17 = torch.where(c17, e0 + 1, e0)
    z1 = (Y == 0) | (e17 <= 0)
    su = (ub >> 31) & 1
    eu = (ub >> 23) & 0xFF
    mu = (ub & 0x7FFFFF) | (1 << 23)
    P3 = mu * m17  # [2^39, 2^41)
    top = P3 >= (1 << 40)
    m3 = torch.where(top, (P3 + (1 << 16)) >> 17, (P3 + (1 << 15)) >> 16)
    e3 = e17 + eu - 127 + top.to(torch.int64)
    c3 = m3 >= (1 << 24)
    m3 = torch.where(c3, m3 >> 1, m3)
    e3 = torch.where(c3, e3 + 1, e3)
    big = e3 >= 255
    m3 = torch.where(big, torch.full_like(m3, (1 << 24) - 1), m3)
    e3 = torch.where(big, torch.full_like(e3, 254), e3)
    zero = z1 | (eu == 0) | (e3 <= 0)
    return torch.where(zero, torch.zeros_like(xb), ((sx ^ su) << 31) | (e3 << 23) | (m3 & 0x7FFFFF))


def row_scale(amax_b: torch.Tensor):
    """Per row from the bits of max |y|: (rq, ea, s_bits, zero_row). r = rq * 2^(125 - ea) <= 32767 / amax, s = 1 / r."""
    ea = amax_b >> 23
    ma = (amax_b & 0x7FFFFF) | (1 << 23)
    zero = ea < 16  # amax == 0 (or < 2^-111): q = 0, s = 0
    rq = (QMAX << 25) // ma  # 17 bits, [65534, 131068]
    q25 = (1 << 41) // rq  # [2^24, 2^25 + 1024]
    big = q25 >= (1 << 25)
    sm = torch.where(big, (q25 + 2) >> 2, (q25 + 1) >> 1)
    es = ea - 16 + torch.where(big, 2, 1)
    c = sm >= (1 << 24)
    sm = torch.where(c, sm >> 1, sm)
    es = torch.where(c, es + 1, es)
    s_bits = torch.where(zero, torch.zeros_like(ea), (es << 23) | (sm & 0x7FFFFF))
    return rq, ea, s_bits, zero


def pass2(yb: torch.Tensor, rq: torch.Tensor, ea: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """q (int64 in [-32767, 32767]) from y bits (..., K) and the row's (rq, ea, zero) of shape (..., 1)."""
    sy = (yb >> 31) & 1
    ey = (yb >> 23) & 0xFF
    my = (yb & 0x7FFFFF) | (1 << 23)
    P2 = my * rq  # 41 bits
    sh2 = ea - ey + 25  # >= 25
    t = P2 >> (sh2 - 1).clamp(0, 63)
    qm = ((t + 1) >> 1).clamp_max(QMAX)
    qm = torch.where((ey == 0) | (sh2 > 41) | zero, torch.zeros_like(qm), qm)
    return torch.where(sy == 1, -qm, qm)


def kernel(x: torch.Tensor, u: torch.Tensor | None, mode: int, nsb: int = 9):
    """x (and u): (..., K) float tensors. Returns q (int16, natural element order), s (FP32, (..., 1)), y bits."""
    xb = f2b(x)
    ub = f2b(u) if u is not None else None
    yb = pass1(xb, ub, mode, nsb)
    amax = (yb & 0x7FFFFFFF).amax(-1, keepdim=True)
    rq, ea, s_bits, zero = row_scale(amax)
    q = pass2(yb, rq, ea, zero)
    return q.to(torch.int16), b2f(s_bits), yb


def dequant(x: torch.Tensor, u: torch.Tensor | None, mode: int, nsb: int = 9) -> torch.Tensor:
    """q * s in FP32: what the next int16 linear effectively sees."""
    q, s, _ = kernel(x, u, mode, nsb)
    return q.to(torch.float32) * s


def out_perm(K: int, lanes: int = 8) -> np.ndarray:
    """Output word order of the kernel: position p of a row holds element out_perm(K)[p].
    8 lanes: word c = [elements 4c..4c+3 (first half of the row), K/2 + 4c .. K/2 + 4c + 3 (second half)].
    4 lanes: natural order."""
    p = np.arange(K)
    if lanes == 4:
        return p
    c, i = p // 8, p % 8
    return np.where(i < 4, 4 * c + i, K // 2 + 4 * c + (i - 4))


def next_weight_cols(K: int, lanes: int = 8) -> np.ndarray:
    """Option A (README "Element order"): the next linear (fc2 / down_proj / action_time_mlp_out) uses
    W[:, next_weight_cols(K)] with the kernel's q rows as they are in memory."""
    return out_perm(K, lanes)


def prev_weight_rows(K: int, lanes: int = 8) -> np.ndarray:
    """Option B: the previous linear (fc1 / gate_proj and up_proj separately / action_time_mlp_in) uses
    W[prev_weight_rows(K), :] (and bias[prev_weight_rows(K)]); then the kernel's q rows come out in natural order."""
    return np.argsort(out_perm(K, lanes))


def act_exact(x: torch.Tensor, u: torch.Tensor | None, mode: int) -> torch.Tensor:
    """float64 reference of the activation."""
    x = x.double()
    if mode == 0:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    if mode == 2:
        return x * torch.sigmoid(x)
    return x * torch.sigmoid(x) * u.double()
