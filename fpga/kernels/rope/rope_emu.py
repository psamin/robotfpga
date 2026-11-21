"""Bit-exact torch model of the RoPE front end (rope_core.h) and of the ROPE mode outputs under the
integration contract (fpga/kernels/CONTRACT.md 2.1, 2.2; rope_mode.h).

The kernel works on one (token, head) segment of 64 values at a time, natural lerobot order (x1 = x[0:32],
x2 = x[32:64]). Internally a segment is 16 "groups" of 4 values, group u = (x1[2u], x1[2u+1], x2[2u],
x2[2u+1]) (PI below: the two rotation pairs that share one block exponent). The hardware gathers a group
from x1 word u/2 and x2 word 8 + u/2 and writes the results back in natural order.

Arithmetic, per group u (all integer, exact in int64 here):
  e_j      = biased FP32 exponent of the 4 inputs; E_u = max_j e_j (denormals/zero: e = 0 -> value 0)
  A_j      = sign * (((1 << 23 | frac) << 2) >> (E_u - e_j))      26-bit magnitude, 27-bit signed
  NC, NS   = round(-cos * 2^17), round(-sin * 2^17) clamped to [-2^17, 2^17 - 1]   (18-bit, the table)
  O0 = A2*NS0 - A0*NC0      (x1 cos - x2 sin, lane 0 = y1[2u])
  O1 = A3*NS1 - A1*NC1      (lane 1 = y1[2u+1])
  O2 = -(A2*NC0 + A0*NS0)   (x2 cos + x1 sin, lane 2 = y2[2u])
  O3 = -(A3*NC1 + A1*NS1)   (lane 3 = y2[2u+1])                     |O| < 2^44, value = O * 2^(E_u - 169)
FP32 rotation (to_f32; CONTRACT.md 2.2): per group L = bitlen(max |O|), N = |O| >> (L - 26) (26 bits); each
  sign * N * 2^(E_u + L - 195) is rounded to 24 significant bits (RNE, unbounded exponent), then a result
  below 2^-126 becomes +0 and one >= 2^128 becomes inf (sign kept). This is the hardware's order (round,
  then flush). Before the contract the emulator rounded to FP32 with subnormals first, which differed from
  the hardware for results that round up to 2^-126 (review: 12 mismatches on crafted 2^-126 inputs).
Outputs (rope_mode.h):
  F32        the rotation itself
  QOUT / KW  the contract quantizer (contract.quant_rows) over each (token, head) segment of the rotation:
             s = fl32(max(amax, 1e-30) / 32767), q = clamp(rne(fl32(x / s)), +-32767), an all-zero
             segment stores s = 1.0; the stored scale is s * 2^p4. The prototype's power-of-two scale
             (MODE_INT16) is retired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract  # noqa: E402  (fpga/kernels/contract.py: the contract quantizer)

D = 64  # head_dim
DH = 32  # rotation pairs per head
WORDS = 16  # io words (4 x FP32) per (token, head)
CS_FRAC = 17  # table: Q1.17 in 18-bit signed
CS_MAX = (1 << CS_FRAC) - 1
CS_MIN = -(1 << CS_FRAC)
GUARD = 2  # extra alignment bits: 24-bit significand << 2 -> 26-bit magnitude
EXP_OFF = 169  # value = O * 2^(E - 169) = 150 + GUARD + CS_FRAC

# PI[n] = natural head_dim index carried at kernel position n = 4u + j
PI = torch.tensor([[2 * u, 2 * u + 1, DH + 2 * u, DH + 2 * u + 1][j] for u in range(WORDS) for j in range(4)])
PI_INV = torch.argsort(PI)


def cs_table_float(positions: torch.Tensor, max_wavelength: float = 10_000.0):
    """cos/sin exactly as lerobot apply_rope computes them (FP32). positions: [T] ints -> [T, 32] each."""
    freq_exponents = (2.0 / D) * torch.arange(DH, dtype=torch.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[:, None].to(torch.float32) / timescale[None, :].to(torch.float32)
    return torch.cos(radians), torch.sin(radians)


def cs_table_int(positions: torch.Tensor):
    """The DDR table the ARM writes: NC = -cos, NS = -sin in Q1.17 (int64 here). [T, 32] each."""
    cos, sin = cs_table_float(positions)
    nc = torch.round(-cos.double() * (1 << CS_FRAC)).clamp(CS_MIN, CS_MAX).long()
    ns = torch.round(-sin.double() * (1 << CS_FRAC)).clamp(CS_MIN, CS_MAX).long()
    return nc, ns


def cs_words(nc: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
    """Table rows as the kernel reads them: [T, 16, 4] int32 slots, word u = (NC[2u], NC[2u+1], NS[2u],
    NS[2u+1])."""
    t = nc.shape[0]
    w = torch.empty(t, WORDS, 4, dtype=torch.int64)
    w[:, :, 0] = nc[:, 0::2]
    w[:, :, 1] = nc[:, 1::2]
    w[:, :, 2] = ns[:, 0::2]
    w[:, :, 3] = ns[:, 1::2]
    return w.to(torch.int32)


def table_rows(positions: torch.Tensor) -> torch.Tensor:
    """The DDR table rows the ARM writes, one per token: [T, 64] int32 = NC[0..31] | NS[0..31] (Q1.17)."""
    nc, ns = cs_table_int(positions)
    return torch.cat([nc, ns], -1).to(torch.int32)


def rows_from_table(tab: torch.Tensor) -> torch.Tensor:
    """[T, 64] table rows -> [T, 16, 4] group words (NC[2u], NC[2u+1], NS[2u], NS[2u+1])."""
    tab = tab.long()
    return cs_words(tab[:, :DH], tab[:, DH:])


def run_rows(x_nat: torch.Tensor, tab: torch.Tensor):
    """The kernel on x_nat [T, H, 64] FP32 (natural order) with table rows [T, 64] -> (o, E) per group,
    [T, H, 16, 4] / [T, H, 16]."""
    T, H, _ = x_nat.shape
    xw = x_nat.float()[..., PI].reshape(T, H, WORDS, 4).contiguous()
    return rotate(xw, rows_from_table(tab)[:, None])


def to_natural(v: torch.Tensor) -> torch.Tensor:
    """[..., 16, 4] group order -> [..., 64] natural order."""
    return v.reshape(*v.shape[:-2], D)[..., PI_INV]


def bitlen(v: torch.Tensor) -> torch.Tensor:
    """Bit length of non-negative int64 values (< 2^53), exact."""
    _, e = torch.frexp(v.double())
    return torch.where(v > 0, e.long(), torch.zeros_like(v))


def rotate(xw: torch.Tensor, csw: torch.Tensor):
    """xw: [..., 16, 4] FP32 in kernel (PI) order; csw: [..., 16, 4] int table words broadcastable.
    Returns O [..., 16, 4] int64 and E [..., 16] int64."""
    bits = xw.contiguous().view(torch.int32).long()
    e = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    neg = bits < 0  # sign bit (int32 view)
    E = e.amax(dim=-1)
    mag = torch.bitwise_left_shift(frac | (1 << 23), GUARD)
    sh = (E[..., None] - e).clamp(0, 63)
    a = torch.bitwise_right_shift(mag, sh)
    a = torch.where((e == 0) | (sh > 26), torch.zeros_like(a), a)
    a = torch.where(neg, -a, a)
    c = csw.long()
    a0, a1, a2, a3 = a.unbind(-1)
    nc0, nc1, ns0, ns1 = c.unbind(-1)
    o = torch.stack(
        [a2 * ns0 - a0 * nc0, a3 * ns1 - a1 * nc1, -(a2 * nc0 + a0 * ns0), -(a3 * nc1 + a1 * ns1)], -1
    )
    # the ranges rope_core.h relies on (proof there): E > 0 <=> non-zero group, and then L in [42, 45]
    gmax = o.abs().amax(dim=-1)
    L = bitlen(gmax)
    assert bool(((E > 0) == (gmax > 0)).all()), "a group with E > 0 rotated to zero"
    assert bool(((E == 0) | ((L >= 42) & (L <= 45))).all()), "bitlen(max |O|) outside [42, 45]"
    return o, E


def to_f32(o: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """FP32 rotation: per group Lg = bitlen(max |O|), N = |O| >> (Lg - 26) (26 bits); each
    sign * N * 2^(E + Lg - 195) is rounded to 24 significant bits (RNE, unbounded exponent), then
    |y| < 2^-126 -> +0 and |y| >= 2^128 -> inf (rope_core.h to_f32: round first, then flush)."""
    mag = o.abs()
    L = bitlen(mag.amax(dim=-1))
    sh = (L - 26).clamp_min(0)[..., None]
    n = torch.bitwise_right_shift(mag, sh)  # < 2^26
    drop = (bitlen(n) - 24).clamp_min(0)  # 0..2 bits below the 24-bit significand
    m = torch.bitwise_right_shift(n, drop)
    rem = n - torch.bitwise_left_shift(m, drop)
    half = torch.bitwise_right_shift(
        torch.bitwise_left_shift(torch.ones_like(n), drop), 1
    )  # 0 when drop == 0
    up = (drop > 0) & ((rem > half) | ((rem == half) & (m % 2 == 1)))
    m = m + up.long()  # may reach 2^24: still exact below
    y = torch.sign(o).double() * m.double() * torch.pow(2.0, (E[..., None] + sh + drop - EXP_OFF).double())
    y = torch.where(y.abs() < 2.0**-126, torch.zeros_like(y), y)
    return y.to(torch.float32)  # exact for every normal result, inf at >= 2^128


def rope_rows(x_nat: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    """The FP32 rotation of x_nat [T, H, 64] (natural order) with table rows [T, 64] -> [T, H, 64] FP32."""
    o, E = run_rows(x_nat, tab)
    return to_natural(to_f32(o, E))


def quant_segments(y: torch.Tensor, shift: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract quantizer (contract.quant_rows) over the last axis (one head segment): q int16
    [..., 64] and the stored scale [...] (s * 2^shift, 1.0 for an all-zero segment)."""
    q, s = contract.quant_rows(y.detach().cpu().float().numpy(), shift)
    return torch.from_numpy(q.astype(np.int16)), torch.from_numpy(np.ascontiguousarray(s, np.float32))


# ------------------------------------------------------------------------------------------ policy-level ops
def kernel_from_natural(x: torch.Tensor, positions: torch.Tensor, mode: str) -> torch.Tensor:
    """x: [B, L, H, 64] FP32 in natural head_dim order (as torch has it); positions [B, L].
    Runs the kernel and returns its result as FP32 in natural order: mode 'f32' -> the FP32 rotation,
    'q' -> q * s of the contract quantizer per (token, head) (what the int16 attention GEMM multiplies:
    exact int products times the two FP32 scales)."""
    B, Lt, H, _ = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    for b in range(B):
        y = rope_rows(x[b].float(), table_rows(positions[b]))
        if mode == "q":
            q, s = quant_segments(y)
            y = (q.double() * s.double()[..., None]).to(torch.float32)
        out[b] = y
    return out


def fmt_contract(y: torch.Tensor) -> torch.Tensor:
    """The format reference: FP32 values [..., 64] through the contract quantizer per segment, as q * s."""
    q, s = quant_segments(y.reshape(-1, D))
    return (q.double() * s.double()[..., None]).to(torch.float32).reshape(y.shape)


def selftest():
    torch.manual_seed(0)
    T, H = 81, 5
    pos = torch.arange(T)
    x = torch.randn(1, T, H, D) * 3
    d_half = D // 2
    cos, sin = cs_table_float(pos)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :d_half], x[..., d_half:]
    ref = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    f = kernel_from_natural(x, pos[None], "f32")
    rel = ((f - ref).abs().amax(-1) / ref.abs().amax(-1)).max()
    print(f"f32: max err / head amax = {rel:.3e}")
    y = kernel_from_natural(x, pos[None], "q")
    rel = ((y - ref).abs().amax(-1) / ref.abs().amax(-1)).max()
    rms = ((y - ref).pow(2).mean(-1).sqrt() / ref.abs().amax(-1)).mean()
    print(f"q (contract quantizer): max err / head amax = {rel:.3e}, mean rms / amax = {rms:.3e}")
    y = fmt_contract(ref)
    rel = ((y - ref).abs().amax(-1) / ref.abs().amax(-1)).max()
    print(f"contract quantizer of the exact rope: max err / head amax = {rel:.3e}")
    # Round, then flush at 2^-126 (the contract boundary fix). A group with bitlen(max |O|) = 42 and
    # E = 1 has the value n 2^-152: n = 2^26 - 3 is below 2^-126 after rounding to 24 bits (-> +0; the
    # old double -> float conversion rounded it up to 2^-126); n = 2^26 - 1 rounds up to 2^-126.
    o = torch.tensor(
        [[[((1 << 26) - 3) << 16, 0, 0, 0], [((1 << 26) - 1) << 16, 0, 0, 0]]], dtype=torch.int64
    )
    v = to_f32(o, torch.ones(1, 2, dtype=torch.int64)).view(torch.int32)
    got = [int(v[0, 0, 0]), int(v[0, 1, 0])]
    print("2^-126 boundary:", [f"{u:#010x}" for u in got], "PASS" if got == [0, 0x00800000] else "FAIL")
    assert got == [0, 0x00800000]


if __name__ == "__main__":
    selftest()
