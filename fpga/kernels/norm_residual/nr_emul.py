"""Bit-exact torch model of the norm_residual kernel (fpga/kernels/norm_residual/norm_residual.h; tb_nr.cpp checks it bit for bit).

The kernel takes a row y = x + residual (the FP32 add itself is IEEE, identical to torch's, so the model
starts from y), computes the row statistics exactly in integers, normalizes in FP32 and quantizes to int16
with a per-row scale. Every step below is the kernel's own arithmetic, in the kernel's order:

  stats (block floating point, exact integers):
    e_i, m_i   exponent field and 24-bit significand of y_i (e_i = 0 -> treated as 0)
    Emax       max e_i over the row
    Y_i        = round_half_away(m_i * 2^(1 + e_i - Emax)), signed, |Y_i| <= 2^25 (y_i = Y_i * 2^(Emax-151))
    S = sum Y_i, Q = sum Y_i^2 (exact), num = D*Q - S^2 (LayerNorm) or D*Q (RMSNorm), exact (<= 72 bits)
  row scalars (FP32, IEEE round-to-nearest-even, one op at a time):
    invD = recip_nr(D), invD2 = recip_nr(D*D)                        (once per call)
    var  = ldexp(i2f(num) * invD2, 2*(Emax-151))
    mean = ldexp(i2f(S) * invD, Emax-151)       (LayerNorm; 0 for RMSNorm)
    rstd = rsqrt_nr(var + eps)
  normalize (FP32): z = ((y - mean) * rstd) * g + b      (RMSNorm: b = 0, mean = 0)
  quantize: A = max|z|; inv = trunc17(32767 * recip_nr(A)); s = recip_nr(inv)
            q = clamp(rne(z * inv), -32767, 32767)  (z*inv exact: 24 x 17 significant bits)
            A < 2^-100 (e.g. an all-zero RMSNorm row): inv = 0, s = 0, q = 0
  the next linear sees q * s.

recip_nr / rsqrt_nr: 128-entry seed table on the top fraction bits + two Newton-Raphson steps in FP32.
i2f: exact integer -> FP32 round-to-nearest-even. ldexp: exact power-of-two scaling, flush to zero below
the normal range (the Xilinx FP cores flush denormals; the model does the same where it matters).
"""
from __future__ import annotations

import math

import numpy as np
import torch

QMAX = 32767
INV_MASK = 0xFFFFFF80  # keep sign, exponent and the top 16 fraction bits: a 17-bit significand
TINY_A = 2.0**-100     # rows whose |z| max is below this quantize to zero (recip seed range guard)

# ---------------------------------------------------------------------------- seed tables (as in the C++)
RECIP_SEED = np.array([1.0 / (1.0 + (i + 0.5) / 128.0) for i in range(128)], dtype=np.float32)
RSQRT_SEED = np.array(
    [1.0 / math.sqrt((1.0 + (i + 0.5) / 64.0) * (2.0 if p else 1.0)) for p in range(2) for i in range(64)],
    dtype=np.float32,
)
_RECIP_SEED_BITS = torch.from_numpy(RECIP_SEED.view(np.int32).astype(np.int64))
_RSQRT_SEED_BITS = torch.from_numpy(RSQRT_SEED.view(np.int32).astype(np.int64))


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32).to(torch.int64)


def _from_bits(b: torch.Tensor) -> torch.Tensor:
    return b.to(torch.int32).view(torch.float32)


def recip_nr(a: torch.Tensor) -> torch.Tensor:
    """1/a for positive normal FP32 a in [2^-125, 2^125]; a == 0 -> 0."""
    a = a.to(torch.float32)
    b = _bits(a)
    E = ((b >> 23) & 0xFF) - 127
    idx = (b >> 16) & 0x7F
    y = _from_bits(_RECIP_SEED_BITS[idx] - (E << 23))
    two = torch.tensor(2.0, dtype=torch.float32)
    for _ in range(2):
        t = a * y
        t = two - t
        y = y * t
    return torch.where(a == 0, torch.zeros_like(a), y)


def rsqrt_nr(v: torch.Tensor) -> torch.Tensor:
    """1/sqrt(v) for positive normal FP32 v in [2^-124, 2^124]."""
    v = v.to(torch.float32)
    b = _bits(v)
    E = ((b >> 23) & 0xFF) - 127
    p = E & 1
    Eh = (E - p) >> 1  # floor(E / 2)
    idx = (p << 6) | ((b >> 17) & 0x3F)
    y = _from_bits(_RSQRT_SEED_BITS[idx] - (Eh << 23))
    h = v * torch.tensor(0.5, dtype=torch.float32)
    c15 = torch.tensor(1.5, dtype=torch.float32)
    for _ in range(2):
        t = y * y
        t = h * t
        t = c15 - t
        y = y * t
    return y


def i2f_rne(n: int) -> np.float32:
    """Exact Python int -> FP32 with round-to-nearest-even (no double rounding)."""
    if n == 0:
        return np.float32(0.0)
    s = -1 if n < 0 else 1
    a = -n if n < 0 else n
    bl = a.bit_length()
    if bl <= 24:
        return np.float32(s * a)
    sh = bl - 24
    m = a >> sh
    rem = a & ((1 << sh) - 1)
    half = 1 << (sh - 1)
    if rem > half or (rem == half and (m & 1)):
        m += 1
    return np.float32(s * math.ldexp(m, sh))  # m <= 2^24: exact in FP32 (range permitting)


def ldexp_ftz(f: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """f * 2^k exactly, flushing results below 2^-126 to zero (as the kernel's exponent adder)."""
    # in double: f * 2^k is exact there for any k the kernel produces (|k| <= 302), so the only rounding
    # is the final cast, which cannot round a result that is normal in FP32 (flushed below, never overflows)
    r = (f.to(torch.float64) * torch.pow(torch.tensor(2.0, dtype=torch.float64), k.to(torch.float64)))
    r = r.to(torch.float32)
    return torch.where(r.abs() < 2.0**-126, torch.zeros_like(r), r)


# ---------------------------------------------------------------------------- the kernel
def block_fixed(y: torch.Tensor):
    """y: (R, D) FP32 -> Y (R, D) int64, Emax (R,) int64."""
    b = _bits(y)
    e = (b >> 23) & 0xFF
    m = torch.where(e > 0, (b & 0x7FFFFF) | 0x800000, torch.zeros_like(b))
    neg = (b >> 31) & 1
    Emax = e.amax(dim=1)
    sh = Emax[:, None] - e - 1
    shc = sh.clamp(1, 62)
    rnd = (m + (torch.ones_like(m) << (shc - 1))) >> shc
    mag = torch.where(sh == -1, m << 1, torch.where(sh == 0, m, torch.where(sh <= 24, rnd, torch.zeros_like(m))))
    Y = torch.where(neg == 1, -mag, mag)
    return Y, Emax


def row_stats(y: torch.Tensor, layernorm: bool, eps: float):
    """Kernel's mean, rstd per row (FP32 tensors of shape (R,))."""
    R, D = y.shape
    Y, Emax = block_fixed(y)
    S = Y.sum(dim=1)
    Q = (Y * Y).sum(dim=1)  # <= 1024 * 2^50 < 2^63
    numf, Sf = [], []
    for s_, q_ in zip(S.tolist(), Q.tolist()):
        num = D * q_ - (s_ * s_ if layernorm else 0)
        assert num >= 0
        numf.append(i2f_rne(num))
        Sf.append(i2f_rne(s_))
    numf = torch.tensor(np.array(numf, np.float32))
    Sf = torch.tensor(np.array(Sf, np.float32))
    invD = recip_nr(torch.tensor([float(D)], dtype=torch.float32))
    invD2 = recip_nr(torch.tensor([float(D * D)], dtype=torch.float32))
    var = ldexp_ftz(numf * invD2, 2 * (Emax - 151))
    if layernorm:
        mean = ldexp_ftz(Sf * invD, Emax - 151)
    else:
        mean = torch.zeros(R, dtype=torch.float32)
    ve = var + torch.tensor(eps, dtype=torch.float32)
    rstd = rsqrt_nr(ve)
    return mean, rstd


def normalize(y, mean, rstd, g, b):
    d = y - mean[:, None]
    t = d * rstd[:, None]
    t = t * g[None, :]
    if b is None:
        b = torch.zeros_like(g)
    return t + b[None, :]


def quantize(z: torch.Tensor):
    """z (R, D) FP32 -> q (R, D) int32 in [-32767, 32767], s (R,) FP32, inv (R,) FP32 (17-bit significand)."""
    A = z.abs().amax(dim=1)
    r = recip_nr(A)
    inv = r * torch.tensor(float(QMAX), dtype=torch.float32)
    inv = _from_bits(_bits(inv) & INV_MASK)
    tiny = A < TINY_A  # absmax below 2^-100 (exponent field < 27): the row quantizes to q = 0, s = 0
    inv = torch.where(tiny, torch.zeros_like(inv), inv)
    s = recip_nr(inv)
    p = z.double() * inv.double()[:, None]  # exact: 24 x 17 significant bits
    q = torch.round(p).clamp_(-QMAX, QMAX).to(torch.int32)  # torch.round: half to even
    return q, s, inv


def kernel(y: torch.Tensor, g: torch.Tensor, b: torch.Tensor | None, eps: float, layernorm: bool):
    """Full kernel on rows y (R, D) FP32. Returns z (FP32), q (int32), s (FP32), mean, rstd."""
    y = y.to(torch.float32).contiguous()
    mean, rstd = row_stats(y, layernorm, eps)
    z = normalize(y, mean, rstd, g.to(torch.float32), None if b is None else b.to(torch.float32))
    q, s, _ = quantize(z)
    return z, q, s, mean, rstd


def torch_quant_absmax(z: torch.Tensor):
    """Reference int16 quantization of the engine emulation (precision_emulation.quant_sym, per token)."""
    s = z.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / QMAX
    q = torch.round(z / s).clamp_(-QMAX, QMAX)
    return q, s.squeeze(1)


if __name__ == "__main__":  # self-checks of the scalar units
    torch.manual_seed(0)
    a = torch.exp(torch.empty(200000).uniform_(-60, 60)).to(torch.float32)
    r = recip_nr(a)
    ex = (1.0 / a.double())
    print("recip_nr max rel err", float(((r.double() - ex) / ex).abs().max()), "ulp-ish 2^-24 =", 2.0**-24)
    rs = rsqrt_nr(a)
    ex = 1.0 / a.double().sqrt()
    print("rsqrt_nr max rel err", float(((rs.double() - ex) / ex).abs().max()))
    for n in (0, 1, 2**24 + 1, 2**24 + 3, 2**72 - 1, -(2**40 + 2**16), 3 * 2**60 + 2**36):
        assert float(i2f_rne(n)) == float(np.float32(float(n))) or n.bit_length() > 53, n
    print("i2f_rne ok")
    y = torch.randn(64, 768) * 3 + 0.5
    y[:, 7] *= 40  # an outlier channel
    g = torch.rand(768) + 0.5
    b = torch.randn(768) * 0.1
    z, q, s, mean, rstd = kernel(y, g, b, 1e-6, True)
    zt = torch.nn.functional.layer_norm(y, (768,), g, b, 1e-6)
    print("LN z vs torch max abs", float((z - zt).abs().max()), "max |z|", float(zt.abs().max()))
    deq = q.float() * s[:, None]
    print("LN dequant vs torch max abs / row absmax", float(((deq - zt).abs().amax(1) / zt.abs().amax(1)).max()))
