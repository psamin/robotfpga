"""Bit-exact torch emulation of the SOFTMAX unit (fpga/kernels/softmax/softmax_core.h, integration contract
1.0).

One row of L FP32 scores x_j (the score multiplier p0 folded into c):

  c        = fl32(max(p0, 2^-126) * fl32(log2 e))        (c_from_mult; p0 > 0 by validate)
  Mc, ec   = c's significand rounded half up to 17 bits, c's unbiased exponent (+1 if that rounding carries)
  valid(j) = 0 <= E[j] <= D[n] (int32 rank vectors, F_MASK) or all keys
  m        = max over valid j of x_j as FP32 total-order integer keys (KMIN if none)
  fix(x)   = x * 2^(ec + 21) truncated toward zero on 47 bits + sign (e == 0 -> 0; overflow, inf, NaN
  saturate)
  u_j      = min(fix(m) - fix(x_j), 2^26 - 1)
  t_j      = u_j * Mc (Q.37); I = t >> 37, idx = (t >> 27) & 1023, flo = (t >> 14) & 8191
  e23      = T23[idx] - ((S13[idx] * flo) >> 13)          T23[i] = round(2^23 * 2^(-i/1024) * 32767/32768)
                                                           S13[i] = round(T23[i] * ln2 / 1024)
  q_j      = ((round(e23 / 2^I) + 128) >> 8) = ((2 e23 >> I) + 257) >> 9 for I < 17, else 0;  0 if not valid
  scale    = RN(1 / sum_j q_j), the correctly rounded FP32 reciprocal (exact divider); 1.0 when the sum is 0
  P_j      = q_j * scale; q in [0, 32767] and the row max gives 32767.

Rows [N, NP) of a command are zero with scale 1.0, and so are the pad lanes of the scale vector (contract C2).
The arithmetic of q and scale is the pre-contract kernel's (`empty="zero"`) except the zero-row scale (0.0
then).

run_cmd(cmd, mem, base) executes one contract SOFTMAX record on a byte image: the golden executor of this
mode.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract as CT  # noqa: E402

C = CT.C
LOG2E = 1.4426950408889634
QMAX = 32767
ONE_BITS = 0x3F800000
P0_MIN = 2.0**-126


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
LOG2E_F32 = float(np.float32(LOG2E))


def f32_bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def c_from_mult(p0: float) -> float:
    """The unit's c from the record's p0 (> 0): fl32(max(p0, 2^-126) * fl32(log2 e)), IEEE RNE."""
    p = np.float32(p0)
    assert p > 0, "p0 must be positive (validate)"
    p = max(p, np.float32(P0_MIN))
    return float(np.float32(p) * np.float32(LOG2E_F32))


def c_params(c: float):
    """(Mc, ec) from the FP32 c > 0 (normal): Mc = significand rounded half up to 17 bits, ec = unbiased
    exponent."""
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
    """FP32 bit patterns (int64 tensor of the uint32 values) -> int64 Q.21 of x * 2^ec (48-bit signed,
    saturating)."""
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


def key_bits(k: torch.Tensor) -> torch.Tensor:
    """Inverse of keys(): the FP32 bit pattern (int64 of the uint32) of a signed key."""
    u = k & 0xFFFFFFFF
    return torch.where((u >> 31) == 1, u ^ 0x7FFFFFFF, u)


def rn_recip(s: torch.Tensor) -> torch.Tensor:
    """RN(1 / s) as FP32 for integer s >= 0 (int64, < 2^35), exactly as the unit's divider computes it; s = 0
    -> 1.0."""
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
    bits = torch.where(zero, torch.full_like(bits, ONE_BITS), bits)  # a zero row: scale 1.0 (contract)
    return bits.to(torch.int32).view(torch.float32)


def rank_valid(E: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """E (..., L) key ranks, D (..., N) row ranks (int32 values) -> valid (..., N, L): 0 <= E[k] <= D[n]."""
    E = E.to(torch.int64)
    D = D.to(torch.int64)
    return (E[..., None, :] >= 0) & (E[..., None, :] <= D[..., :, None])


def ranks_from_mask(valid: torch.Tensor):
    """valid (..., N, L) bool -> (E (..., L), D (..., N)) int32 with rank_valid(E, D) == valid, or None when
    the
    rows' visible key sets are not nested (never the case for SmolVLA's masks). D[n] = number of visible keys
    - 1
    (-1: fully masked row), E[k] = the smallest D of a row that sees k (-1: no row)."""
    v = valid.bool()
    D = v.sum(-1) - 1
    big = 1 << 40
    cand = torch.where(v, D[..., :, None].expand(v.shape), torch.full(v.shape, big, dtype=D.dtype))
    E = cand.amin(-2)
    E = torch.where(E >= big, torch.full_like(E, -1), E)
    if not torch.equal(rank_valid(E, D), v):
        return None
    return E.to(torch.int32), D.to(torch.int32)


def kernel(x: torch.Tensor, c: float, valid: torch.Tensor | None = None, empty: str = "zero"):
    """The unit on rows x (..., L) FP32 with an optional bool mask valid (..., L) (True = may attend).
    Returns (q int64 (..., L), scale float32 (..., 1)). `empty` is kept for old callers; only "zero"
    exists."""
    assert empty == "zero", "F_EMPTY_UNIFORM is retired (contract 1.0)"
    x = x.to(torch.float32).contiguous()
    shp = x.shape
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    if valid is None:
        valid = torch.ones(shp, dtype=torch.bool)
    valid = valid.expand(shp)
    mc, ec = c_params(c)
    kmin = -(1 << 31)
    km = torch.where(valid, keys(bits), torch.full_like(bits, kmin)).amax(-1, keepdim=True)
    mbits = key_bits(km)  # the hardware carries the key and turns it back into bits
    u = (fix(mbits, ec) - fix(bits, ec)).clamp(0, (1 << 26) - 1)
    t = u * mc
    ip = t >> 37
    ix = (t >> 27) & 1023
    flo = (t >> 14) & 8191
    e23 = T23[ix] - ((S13[ix] * flo) >> 13)
    q = torch.where(ip >= 17, torch.zeros_like(e23), (((e23 << 1) >> ip.clamp(max=17)) + 257) >> 9)
    q = torch.where(valid, q, torch.zeros_like(q))
    return q, rn_recip(q.sum(-1, keepdim=True))


def softmax_u16(x, c, valid=None):
    """Probabilities as the P.V GEMM sees them: q * scale (FP32)."""
    q, s = kernel(x, c, valid)
    return q.to(torch.float32) * s


# ---------------------------------------------------------------- golden executor of one SOFTMAX record
def _rows(mem: np.ndarray, base: int, addr: int, pitch: int, n: int, nbytes: int) -> np.ndarray:
    off = addr - base + pitch * np.arange(n)[:, None] + np.arange(nbytes)[None, :]
    return mem[off]


def run_cmd(c: CT.Cmd, mem: np.ndarray, base: int = 0) -> None:
    """Execute the SOFTMAX record c (contract.Cmd, already valid) on the byte image mem (np.uint8) whose byte
    0
    is at physical address `base`. Writes P rows and the scale vectors in place, exactly as the unit does."""
    assert c.op == C.OP_SOFTMAX and c.flags & C.F_QOUT
    L, N, B = c.K, c.N, c.B or 1
    NP = c.par(C.P_NP) or N
    NP4 = CT.pad4(NP)
    A, Cs, F, D, E = (c.slot(s) for s in "ACFDE")
    cv = c_from_mult(CT.bits_f32(c.par(C.P_EPS)))
    mask = bool(c.flags & C.F_MASK)
    for b in range(B):
        x = _rows(mem, base, A.addr + b * A.stride, A.pitch, N, 4 * L).copy().view(np.float32).reshape(N, L)
        v = None
        if mask:
            Ek = mem[E.addr + b * E.stride - base + np.arange(4 * L)].copy().view(np.int32)
            Dn = mem[D.addr + b * D.stride - base + np.arange(4 * N)].copy().view(np.int32)
            v = rank_valid(torch.from_numpy(Ek), torch.from_numpy(Dn))
        q, s = kernel(torch.from_numpy(x), cv, v)
        P = np.zeros((NP, L), np.int16)
        P[:N] = q.numpy().astype(np.int16)
        off = Cs.addr + b * Cs.stride - base + Cs.pitch * np.arange(NP)[:, None] + np.arange(2 * L)[None, :]
        mem[off] = P.view(np.uint8).reshape(NP, 2 * L)
        sc = np.full(NP4, np.float32(1.0), np.float32)
        sc[:N] = s[:, 0].numpy()
        o = F.addr + b * F.stride - base
        mem[o : o + 4 * NP4] = sc.view(np.uint8)
