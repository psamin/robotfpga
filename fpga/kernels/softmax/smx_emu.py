"""Bit-exact torch emulation of the FPGA row-softmax kernel (fpga/kernels/softmax/softmax_core.h).

The kernel's arithmetic, step by step (one row of L FP32 scores x_j, temperature folded into c):

  c        = temperature * log2(e)  (FP32 register)  ->  Mc = round(mant(c) * 2^16) in [2^16, 2^17], ec = exp(c)
  valid(j) = j < L and j < n_valid and (bitmap bit j, if bitmap mode) and (j <= row_in_head + causal_off, if causal)
  m        = max over valid j of x_j, compared as FP32 total-order integer keys
  fix(x)   = x * 2^ec on a fixed Q.21 grid in a 48-bit signed integer: M * 2^(e - 129 + ec), truncated toward
             zero (M = 24-bit significand, e = biased exponent; e == 0 -> 0; overflow saturates to +-(2^47-1))
  u_j      = min(fix(m) - fix(x_j), 2^26 - 1)                (0 for every j < L when no element of the row is valid)
  t_j      = u_j * Mc                                         (Q.37: log2 units, -t_j = log2 of the probability)
  I = t >> 37, idx = (t >> 27) & 1023, flo = (t >> 14) & 8191
  e23      = T23[idx] - ((S13[idx] * flo) >> 13)              T23[i] = round(2^23 * 2^(-i/1024) * 32767/32768)
                                                              S13[i] = round(T23[i] * ln2 / 1024)
  ehi      = round(e23 / 2^I) = ((2 e23 >> I) + 1) >> 1       (0 when I >= 24)  ~ 32767/32768 * 2^23 * 2^(-t)
  uint16 mode: q_j = (ehi + 128) >> 8 in [0, 32767]; scale = RN(1 / sum_j q_j)  ->  P_j = q_j * scale
  FP32  mode: p_j = (float)ehi * RN(1 / sum_j ehi)            (one IEEE FP32 multiply)
  RN(1 / S)  = the correctly rounded (nearest even) FP32 reciprocal of the exact integer row sum S, computed by
             the kernel's divider as q = floor(2^(25+p) / S), p = msb(S), sticky = remainder != 0 (rn_recip()).
  invalid j (masked, or the padding of the last 128-bit word) -> 0.

  c          = fl32(p0 * fl32(log2 e)) from the record's p0 (the score multiplier, e.g. 1/sqrt(64)): c_from_mult()
  masks      : valid(n, k) = 0 <= E[k] <= D[n] on int32 rank vectors, applied after the 16-bit clamps of the kernel's
             tables (E < 0 or E >= 0x7FFF -> never visible, D -> [-1, 0x7FFE]): rank_valid()

A fully masked row is zero (q = 0, scale = 0: the sequencer contract) or, with empty="uniform" (F_EMPTY_UNIFORM),
uniform over its L keys like torch's softmax(where(mask, x * temperature, finfo.min)) (u = 0 for all j < L).
"""

from __future__ import annotations

import math

import torch

LOG2E = 1.4426950408889634
QMAX = 32767


def tables():
    """T23 (23-bit) and S13 (13-bit) as Python ints, the exact contents of softmax_tables.h."""
    t23, s13 = [], []
    for i in range(1024):
        t = round((2.0**23) * (2.0 ** (-i / 1024.0)) * (32767.0 / 32768.0))
        s = round(t * math.log(2.0) / 1024.0)
        assert 0 < t < (1 << 23) and 0 < s < (1 << 13)
        t23.append(t)
        s13.append(s)
    return t23, s13


_T23, _S13 = tables()
T23 = torch.tensor(_T23, dtype=torch.int64)
S13 = torch.tensor(_S13, dtype=torch.int64)


def f32_bits(x: float) -> int:
    return int(torch.tensor([x], dtype=torch.float32).view(torch.int32)[0]) & 0xFFFFFFFF


def c_params(c: float):
    """(Mc, ec) from the FP32 register c > 0 (normal): Mc = round(significand * 2^16), ec = unbiased exponent."""
    b = f32_bits(c)
    e = (b >> 23) & 0xFF
    assert 0 < e < 255 and not (b >> 31), "c must be a positive normal FP32"
    m24 = (1 << 23) | (b & 0x7FFFFF)
    mc = (m24 + 64) >> 7
    ec = e - 127
    if mc >> 17:  # rounded up to 2.0
        mc >>= 1
        ec += 1
    return mc, ec


def fix(bits: torch.Tensor, ec: int) -> torch.Tensor:
    """FP32 bit patterns (int64 tensor of the uint32 values) -> int64 Q.21 of x * 2^ec (48-bit signed, saturating)."""
    e = (bits >> 23) & 0xFF
    m = (bits & 0x7FFFFF) | (1 << 23)
    k = 24 - (e - 129 + ec)  # right shift of M << 24
    kc = k.clamp(0, 63)
    v = (m << 24) >> kc
    v = torch.where(k >= 48, torch.zeros_like(v), v)
    v = torch.where((k <= 0) | (e == 255), torch.full_like(v, (1 << 47) - 1), v)
    v = torch.where(e == 0, torch.zeros_like(v), v)
    return torch.where((bits >> 31) == 1, -v, v)


def keys(bits: torch.Tensor) -> torch.Tensor:
    """FP32 total-order integer keys (signed): bits ^ 0x7FFFFFFF for negatives."""
    s = bits >> 31
    k = torch.where(s == 1, bits ^ 0x7FFFFFFF, bits)
    return torch.where(k >= (1 << 31), k - (1 << 32), k)


def rn_recip(s: torch.Tensor) -> torch.Tensor:
    """RN(1 / s) as FP32 for integer s >= 1 (int64, < 2^35), exactly as recip_unit in softmax_core.h computes it."""
    s = s.to(torch.int64)
    assert bool((s >= 0).all()) and bool((s < (1 << 35)).all())
    zero = s == 0
    s = torch.where(zero, torch.ones_like(s), s)
    p = (torch.frexp(s.double())[1] - 1).to(torch.int64)  # msb position (exact: s < 2^53)
    num = torch.ones_like(s) << (25 + p)
    q = num // s  # in (2^24, 2^25]
    rem = num - q * s
    up = ((q & 1) == 1) & ((rem != 0) | (((q >> 1) & 1) == 1))
    sig = (q >> 1) + up.to(torch.int64)
    ex = 126 - p + (sig >> 24)
    bits = torch.where(q == (1 << 25), (127 - p) << 23, (ex << 23) | (sig & 0x7FFFFF))
    bits = torch.where(zero, torch.zeros_like(bits), bits)  # S = 0 (a zero row): scale 0
    return bits.to(torch.int32).view(torch.float32)


LOG2E_F32 = float(torch.tensor(LOG2E, dtype=torch.float32))


def c_from_mult(p0: float) -> float:
    """The kernel's c from the record's p0: one IEEE FP32 multiply, fl32(p0) * fl32(log2 e)."""
    return float(torch.tensor(p0, dtype=torch.float32) * torch.tensor(LOG2E_F32, dtype=torch.float32))


def rank_valid(E: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """E (..., L) key ranks, D (..., N) row ranks (int) -> valid (..., N, L): 0 <= E[k] <= D[n] after the kernel's
    16-bit clamps (E < 0 or E >= 0x7FFF never visible; D clamped to [-1, 0x7FFE])."""
    E = E.to(torch.int64)
    D = D.to(torch.int64)
    Ec = torch.where((E < 0) | (E >= 0x7FFF), torch.full_like(E, 0x7FFF), E)
    Dc = D.clamp(-1, 0x7FFE)
    return Ec[..., None, :] <= Dc[..., :, None]


def ranks_from_mask(valid: torch.Tensor):
    """valid (..., N, L) bool -> (E (..., L), D (..., N)) int32 with rank_valid(E, D) == valid, or None when the
    rows' visible key sets are not nested (never the case for SmolVLA's masks). D[n] = number of visible keys - 1
    (-1: fully masked row), E[k] = the smallest D of a row that sees k (0x7FFF: no row)."""
    v = valid.bool()
    D = v.sum(-1) - 1
    cand = torch.where(v, D[..., :, None].expand(v.shape), torch.full(v.shape, 0x7FFF, dtype=D.dtype))
    E = cand.amin(-2)
    E = torch.where(E < 0, torch.full_like(E, 0x7FFF), E)  # only fully masked rows "see" it: nobody
    if not torch.equal(rank_valid(E, D), v):
        return None
    return E.to(torch.int32), D.to(torch.int32)


def kernel(x: torch.Tensor, c: float, valid: torch.Tensor | None = None, fp32_out: bool = False,
           empty: str = "uniform"):
    """Emulate the kernel on rows x (..., L) FP32 with optional bool mask valid (..., L) (True = may attend);
    empty: "uniform" (F_EMPTY_UNIFORM) or "zero" for fully masked rows.

    Returns (q int64, scale float32 (..., 1)) in uint16 mode, or p float32 in FP32 mode."""
    x = x.to(torch.float32).contiguous()
    shp = x.shape
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    if valid is None:
        valid = torch.ones(shp, dtype=torch.bool)
    valid = valid.expand(shp)
    mc, ec = c_params(c)
    k = keys(bits)
    kmin = -(1 << 31)
    km = torch.where(valid, k, torch.full_like(k, kmin)).amax(-1, keepdim=True)
    anyv = valid.any(-1, keepdim=True)
    # back to FP32 bits of the max (the kernel carries the bits of the winning element)
    idx = torch.where(valid, k, torch.full_like(k, kmin)).argmax(-1, keepdim=True)
    mbits = torch.gather(bits, -1, idx)
    xm = fix(mbits, ec)
    xx = fix(bits, ec)
    u = (xm - xx).clamp(0, (1 << 26) - 1)
    u = torch.where(anyv, u, torch.zeros_like(u))
    t = u * mc
    I = t >> 37
    ix = (t >> 27) & 1023
    flo = (t >> 14) & 8191
    e23 = T23[ix] - ((S13[ix] * flo) >> 13)
    ehi = torch.where(I >= 24, torch.zeros_like(e23), (((e23 << 1) >> I.clamp(max=24)) + 1) >> 1)
    assert empty in ("uniform", "zero")
    use = (valid | ~anyv) if empty == "uniform" else valid  # fully masked rows: uniform over all L keys, or zero
    ehi = torch.where(use, ehi, torch.zeros_like(ehi))
    if fp32_out:
        r = rn_recip(ehi.sum(-1, keepdim=True))
        return ehi.to(torch.float32) * r  # IEEE FP32 multiply (ehi < 2^23 is exact in FP32)
    q = (ehi + 128) >> 8
    scale = rn_recip(q.sum(-1, keepdim=True))
    return q, scale


def softmax_u16(x, c, valid=None, empty="uniform"):
    """Probabilities as the P.V engine sees them: q * scale (FP32)."""
    q, s = kernel(x, c, valid, empty=empty)
    return q.to(torch.float32) * s


def softmax_f32(x, c, valid=None, empty="uniform"):
    return kernel(x, c, valid, fp32_out=True, empty=empty)
