"""Bit-exact torch model of the RoPE kernel (fpga/kernels/rope/rope_core.h, README.md).

The kernel works on one (token, head) row of 64 values at a time, natural lerobot order (x1 = x[0:32],
x2 = x[32:64]); internally a row is 16 "groups" of 4 values, group u = (x1[2u], x1[2u+1], x2[2u], x2[2u+1])
(PI below: the two rotation pairs that share one block exponent). The hardware gathers a group from x1
word u/2 and x2 word 8 + u/2 and writes the results back in natural order; nothing is permuted in DDR.

Arithmetic, per group u (all integer, exact in int64 here):
  e_j      = biased FP32 exponent of the 4 inputs; E_u = max_j e_j (denormals/zero: e = 0 -> value 0)
  A_j      = sign * (((1 << 23 | frac) << 2) >> (E_u - e_j))      26-bit magnitude, 27-bit signed
  NC, NS   = round(-cos * 2^17), round(-sin * 2^17) clamped to [-2^17, 2^17 - 1]   (18-bit, the table)
  O0 = A2*NS0 - A0*NC0      (x1 cos - x2 sin, lane 0 = y1[2u])
  O1 = A3*NS1 - A1*NC1      (lane 1 = y1[2u+1])
  O2 = -(A2*NC0 + A0*NS0)   (x2 cos + x1 sin, lane 2 = y2[2u])
  O3 = -(A3*NC1 + A1*NS1)   (lane 3 = y2[2u+1])                     |O| < 2^44, value = O * 2^(E_u - 169)
Output, MODE_F32: per group L = bitlen(max |O|), N = |O| >> (L - 26) (26 bits); each sign * N * 2^(E_u + L - 195)
  rounded to FP32 (RNE), results below 2^-126 -> 0 (the hardware normalizes each group once, 4-way).
Output, MODE_INT16 (per-(token, head) scale over the 64 values; the hardware implements qm = 0 only):
  L_u = bitlen(max_j |O_j|), M_u = E_u + L_u (magnitude exponent), Mtop = max_u M_u
  qm = 0 (power-of-two scale): q = rnd(O >> (Mtop - E_u - 15)), s = 2^(Mtop - 184)
  qm = k (k-bit mantissa scale, emulation only): t = top k+1 bits of the largest |O| among the groups with
      M_u = Mtop, N = sign(O) (|O| >> (Mtop - E_u - 26)), q = rnd(N * R[t] >> (25 + F)),
      s = SM[t] * 2^(Mtop - 195), R[t] = round(32767 * 2^(k + F) / (t + 1)), SM[t] = fl32(2^(25 + F) / R[t])
  rnd(P >> n) = (P + 2^(n-1) - (P < 0)) >> n (round half away from zero), n <= 0: exact left shift;
  q clamped to +-32767; all-zero row or s below 2^-126: q = 0, s = 0.
"""

from __future__ import annotations

import math

import torch

D = 64  # head_dim
DH = 32  # rotation pairs per head
WORDS = 16  # io words (4 x FP32) per (token, head)
CS_FRAC = 17  # table: Q1.17 in 18-bit signed
CS_MAX = (1 << CS_FRAC) - 1
CS_MIN = -(1 << CS_FRAC)
GUARD = 2  # extra alignment bits: 24-bit significand << 2 -> 26-bit magnitude
FQ = 2  # R table fraction bits beyond k (R < 2^17)
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
    """Table rows as the kernel reads them: [T, 16, 4] int32 slots, word u = (NC[2u], NC[2u+1], NS[2u], NS[2u+1])."""
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


def rshift_round(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """rnd(p >> n): round half away from zero for n > 0, exact left shift for n <= 0."""
    n = n.expand_as(p)
    pos = n > 0
    ns = n.clamp(1, 62)
    half = torch.bitwise_left_shift(torch.ones_like(p), ns - 1)
    r = torch.bitwise_right_shift(p + half - (p < 0).long(), ns)
    r = torch.where(n > 62, torch.zeros_like(p), r)
    left = torch.bitwise_left_shift(p, (-n).clamp(0, 62))
    return torch.where(pos, r, left)


def rshift_floor(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    n = n.expand_as(p)
    r = torch.bitwise_right_shift(p, n.clamp(0, 63))
    left = torch.bitwise_left_shift(p, (-n).clamp(0, 62))
    return torch.where(n >= 0, r, left)


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
    o = torch.stack([a2 * ns0 - a0 * nc0, a3 * ns1 - a1 * nc1, -(a2 * nc0 + a0 * ns0), -(a3 * nc1 + a1 * ns1)], -1)
    return o, E


def to_f32(o: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """FP32 output mode: per group Lg = bitlen(max |O|), N = |O| >> (Lg - 26) (26 bits), value
    sign * N * 2^(E + Lg - 195), RNE to FP32, flush below 2^-126 (rope_core.h to_f32 does the same)."""
    mag = o.abs()
    L = bitlen(mag.amax(dim=-1))
    sh = (L - 26).clamp_min(0)[..., None]
    n = torch.bitwise_right_shift(mag, sh)
    y = (torch.sign(o).double() * n.double() * torch.pow(2.0, (E[..., None] + sh - EXP_OFF).double())).to(torch.float32)
    return torch.where(y.abs() < 2.0**-126, torch.zeros_like(y), y)


def r_tables(k: int):
    n = 1 << k
    t = torch.arange(n, 2 * n, dtype=torch.float64)
    R = torch.round(32767.0 * 2.0 ** (k + FQ) / (t + 1)).long()
    SM = (2.0 ** (25 + FQ) / R.double()).to(torch.float32)
    assert int(R.max()) < (1 << 17)
    return R, SM


def quantize(o: torch.Tensor, E: torch.Tensor, qm: int = 0):
    """INT16 mode over each pair (dims [..., 16, 4] -> one scale per leading index).
    Returns q [..., 16, 4] int16 values (int64 tensor) and s [...] FP32."""
    absmax = o.abs().amax(dim=-1)  # [..., 16]
    L = bitlen(absmax)
    # the ranges rope_core.h relies on (proof there): E > 0 <=> non-zero group, and then L in [42, 45]
    assert bool(((E > 0) == (absmax > 0)).all()), "a group with E > 0 rotated to zero"
    assert bool(((E == 0) | ((L >= 42) & (L <= 45))).all()), "bitlen(max |O|) outside [42, 45]"
    M = torch.where(absmax == 0, torch.zeros_like(E), E + L)  # all-zero word: M = 0 (Mtop = 0 <=> all zero)
    Mtop = M.amax(dim=-1)  # [...]
    allzero = (absmax == 0).all(dim=-1)
    if qm == 0:
        sh = (Mtop[..., None] - E - 15)[..., None]
        q = rshift_round(o, sh)
        field = Mtop - 57  # biased FP32 exponent of s = 2^(Mtop - 184)
        s = torch.pow(2.0, (Mtop - 184).double()).to(torch.float32)
    else:
        k = qm
        R, SM = r_tables(k)
        tsh = (L - 1 - k)
        tw = rshift_floor(absmax, tsh).clamp_max((2 << k) - 1)  # top k+1 bits of each word's max (garbage where absmax == 0)
        tw = torch.where(M == Mtop[..., None], tw, torch.zeros_like(tw))
        t = tw.amax(dim=-1).clamp(1 << k, (2 << k) - 1)
        idx = t - (1 << k)
        nsh = (Mtop[..., None] - E - 26)[..., None]
        N = torch.sign(o) * rshift_floor(o.abs(), nsh)  # sign-magnitude truncation
        q = rshift_round(N * R[idx][..., None, None], torch.tensor(25 + FQ))
        sm_field = (SM.view(torch.int32).long() >> 23) & 0xFF
        field = sm_field[idx] + Mtop - 195  # biased FP32 exponent of s = SM * 2^(Mtop - 195)
        s = (SM[idx].double() * torch.pow(2.0, (Mtop - 195).double())).to(torch.float32)
    q = q.clamp(-32767, 32767)
    dead = allzero | (field <= 0)
    s = torch.where(dead, torch.zeros_like(s), s)
    q = torch.where(dead[..., None, None], torch.zeros_like(q), q)
    return q, s


# ------------------------------------------------------------------------------------------ policy-level ops
def kernel_from_natural(x: torch.Tensor, positions: torch.Tensor, mode: str, qm: int = 0) -> torch.Tensor:
    """x: [B, L, H, 64] FP32 in natural head_dim order (as torch has it); positions [B, L].
    Applies the PI permutation the offline weight reordering produces, runs the kernel, and returns the
    result back in natural order, as FP32: mode 'f32' -> the FP32-mode output, 'int16' -> q * s
    (what the int16 attention engine multiplies: exact int products times the two FP32 scales)."""
    B, Lt, H, _ = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    for b in range(B):
        nc, ns = cs_table_int(positions[b])
        csw = cs_words(nc, ns)[:, None, :, :]  # [L, 1, 16, 4]
        xw = x[b].float()[..., PI].reshape(Lt, H, WORDS, 4)
        o, E = rotate(xw, csw)
        if mode == "f32":
            y = to_f32(o, E)
        else:
            q, s = quantize(o, E, qm)
            y = (q.double() * s.double()[..., None, None]).to(torch.float32)
        out[b] = y.reshape(Lt, H, D)[..., PI_INV]
    return out


def absmax_int16(x: torch.Tensor) -> torch.Tensor:
    """The format reference: FP32 values quantized per (token, head) with s = amax/32767, q = round(x/s)."""
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 32767
    return torch.round(x / s).clamp(-32767, 32767) * s


def selftest():
    torch.manual_seed(0)
    T, H = 81, 5
    pos = torch.arange(T)
    x = torch.randn(1, T, H, D) * 3
    from_torch = None
    # reference rope (lerobot)
    d_half = D // 2
    cos, sin = cs_table_float(pos)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :d_half], x[..., d_half:]
    ref = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    f = kernel_from_natural(x, pos[None], "f32")
    rel = ((f - ref).abs().amax(-1) / ref.abs().amax(-1)).max()
    print(f"f32 mode: max err / head amax = {rel:.3e}")
    for qm in (0, 3, 4):
        y = kernel_from_natural(x, pos[None], "int16", qm)
        rel = ((y - ref).abs().amax(-1) / ref.abs().amax(-1)).max()
        rms = ((y - ref).pow(2).mean(-1).sqrt() / ref.abs().amax(-1)).mean()
        print(f"int16 qm={qm}: max err / head amax = {rel:.3e}, mean rms / amax = {rms:.3e}")
    y = absmax_int16(ref)
    rel = ((y - ref).abs().amax(-1) / ref.abs().amax(-1)).max()
    rms = ((y - ref).pow(2).mean(-1).sqrt() / ref.abs().amax(-1)).mean()
    print(f"absmax int16 of exact rope: max err / head amax = {rel:.3e}, mean rms / amax = {rms:.3e}")
    del from_torch


if __name__ == "__main__":
    selftest()

