"""Bit-exact model of the NORM mode (contract 1.0, fpga/kernels/CONTRACT.md) as built in norm_residual.h.

tb_nr.cpp checks the C++ against it word for word on whole DDR images (gen_vectors.py writes them).

NORM reads FP32 rows A (K values, K % 8 == 0, 8 <= K <= 1024), normalizes each row (LayerNorm with F_LN, else
RMSNorm) and writes either int16 rows + per-row scales (F_QOUT, through the contract quantizer) or FP32 rows.
The residual add is no longer here: it is the GEMM epilogue (contract 2.3). Per row, in the hardware's order:

  stats (block floating point, exact integers):
    e_i, m_i   exponent field and 24-bit significand of y_i (e_i = 0, a zero or denormal: m_i = 0)
    Emax       max e_i over the row
    Y_i        sign * round_half_away(m_i * 2^(1 + e_i - Emax)), |Y_i| < 2^25 (y_i ~ Y_i 2^(Emax-151))
    S = sum Y_i, Q = sum Y_i^2 (exact), num = K Q - S^2 (LayerNorm) or K Q (RMSNorm), exact (< 2^70)
  row scalars, correctly rounded (one integer divider, 1 quotient bit a clock):
    mean = rn(S / K) * 2^(Emax-151)      (LayerNorm; 0 for RMSNorm)
    var  = rn(num / K^2) * 2^(2 (Emax-151))
    results below 2^-126 flush to +0, above the FP32 range give +inf
  rstd (FP32 cores, denormals flushed to zero with the sign kept, contract 2.3):
    ve = max(fl(var + eps), 2^-126); rstd = rsqrt_nr(ve) (128-entry seed + 2 Newton steps), 0 if ve is inf
  normalize, per element (FP32 cores, flushed): z = ((y - mean) * rstd) * g + b   (RMSNorm: b = 0)
  output: F_QOUT: q, s = contract.quant_rows(z, p4) (the shared quantizer, quantize/quant_int16.h);
          else FP32 z.

Changes against the pre-contract kernel (the norm_residual review, build/brief/integration_inputs.json):
  - mean and var are correctly rounded (exact integer division) instead of fl(fl(S) * recip_nr(K)): an
    all-equal LayerNorm row now gives y - mean = 0 exactly, so z = beta (torch's answer), not +-32767 noise;
  - the quantizer is the contract's (s = fl32(max(amax, 1e-30) / 32767), zero rows 1.0), not 1 / trunc17;
  - every FP32 operation flushes denormals like the Xilinx cores, so the model is the hardware, not the C++
    host float arithmetic.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract  # noqa: E402  (fpga/kernels/contract.py: the shared quantizer and ftz)

QMAX = 32767
FLT_MIN_BITS = 0x00800000

# ---------------------------------------------------------------------------- rsqrt seed (as in the C++)
RSQRT_SEED = np.array(
    [1.0 / math.sqrt((1.0 + (i + 0.5) / 64.0) * (2.0 if p else 1.0)) for p in range(2) for i in range(64)],
    dtype=np.float32,
)
_RSQRT_SEED_BITS = RSQRT_SEED.view(np.uint32).astype(np.int64)

ftz = contract.ftz


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def fmul(a, b) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(_f32(a)) * ftz(_f32(b)))


def fadd(a, b) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(_f32(a)) + ftz(_f32(b)))


def fsub(a, b) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(_f32(a)) - ftz(_f32(b)))


def rsqrt_nr(v: np.ndarray) -> np.ndarray:
    """1/sqrt(v), positive normal FP32 v: seed on (exponent parity, top 6 fraction bits) + 2 Newton steps."""
    v = _f32(v)
    b = v.view(np.uint32).astype(np.int64)
    E = ((b >> 23) & 0xFF) - 127
    p = E & 1
    Eh = (E - p) >> 1  # floor(E / 2), exact: E - p is even
    idx = (p << 6) | ((b >> 17) & 0x3F)
    y = ((_RSQRT_SEED_BITS[idx] - (Eh << 23)) & 0xFFFFFFFF).astype(np.uint32).view(np.float32)
    h = fmul(v, np.float32(0.5))
    for _ in range(2):
        t = fmul(y, y)
        t = fmul(h, t)
        t = fsub(np.float32(1.5), t)
        y = fmul(y, t)
    return y


# ---------------------------------------------------------------------------- exact integer scalars
def div_rn_bits(n: int, d: int, k: int, neg: bool = False) -> int:
    """FP32 bits of rn(n / d) * 2^k (n >= 0, d > 0 integers): correctly rounded to 24 bits (half to even),
    then scaled by 2^k exactly; a biased exponent <= 0 gives +0, >= 255 gives +inf (sign applied to inf)."""
    if n == 0:
        return 0
    # q = floor(n 2^s / d) with 2^24 <= q < 2^25, sticky = remainder != 0
    s = 24 - (n.bit_length() - d.bit_length())
    while True:
        num, den = (n << s, d) if s >= 0 else (n, d << -s)
        q, r = divmod(num, den)
        if q >= 1 << 25:
            s -= 1
        elif q < 1 << 24:
            s += 1
        else:
            break
    m, rb, st = q >> 1, q & 1, r != 0
    if rb and (st or (m & 1)):
        m += 1
    ex = 1 - s  # n / d ~ m 2^(1 - s) = (m / 2^23) 2^(24 - s)
    if m == 1 << 24:
        m >>= 1
        ex += 1
    field = ex + 23 + 127 + k  # value = (m / 2^23) * 2^(ex + 23 + k)
    sign = 0x80000000 if neg else 0
    if field <= 0:
        return 0
    if field >= 255:
        return sign | 0x7F800000
    return sign | (field << 23) | (m & 0x7FFFFF)


def block_fixed(y: np.ndarray):
    """y: (R, K) FP32 -> Y (R, K) int64, Emax (R,) int64 (the hardware's per-element shift and rounding)."""
    b = _f32(y).view(np.uint32).astype(np.int64)
    e = (b >> 23) & 0xFF
    m = np.where(e > 0, (b & 0x7FFFFF) | 0x800000, 0)
    neg = (b >> 31) & 1
    Emax = e.max(axis=1)
    sh = Emax[:, None] - e - 1
    shc = np.clip(sh, 1, 62)
    rnd = (m + (np.int64(1) << (shc - 1))) >> shc
    mag = np.where(sh == -1, m << 1, np.where(sh == 0, m, np.where(sh <= 24, rnd, 0)))
    return np.where(neg == 1, -mag, mag), Emax


def row_scalars(y: np.ndarray, layernorm: bool):
    """mean, var (FP32, shape (R,)) exactly as the hardware's stats + divider."""
    R, K = y.shape
    Y, Emax = block_fixed(y)
    S = Y.sum(axis=1)  # |S| < 2^35
    Q = (Y * Y).sum(axis=1)  # < 2^60
    mean = np.zeros(R, np.uint32)
    var = np.zeros(R, np.uint32)
    for r in range(R):
        s_, q_, k1 = int(S[r]), int(Q[r]), int(Emax[r]) - 151
        num = K * q_ - (s_ * s_ if layernorm else 0)
        assert num >= 0
        var[r] = div_rn_bits(num, K * K, 2 * k1)
        if layernorm:
            mean[r] = div_rn_bits(abs(s_), K, k1, s_ < 0)
    return mean.view(np.float32), var.view(np.float32)


def rstd_of(var: np.ndarray, eps: float) -> np.ndarray:
    ve = fadd(var, np.float32(eps))
    vb = ve.view(np.uint32)
    vb = np.where(vb < FLT_MIN_BITS, np.uint32(FLT_MIN_BITS), vb).astype(np.uint32)
    inf = vb >= 0x7F800000
    r = rsqrt_nr(vb.view(np.float32))
    return np.where(inf, np.float32(0.0), r).astype(np.float32)


def normalize(y, mean, rstd, g, b):
    d = fsub(y, mean[:, None])
    t = fmul(d, rstd[:, None])
    t = fmul(t, _f32(g)[None, :])
    bb = np.zeros_like(_f32(g)) if b is None else _f32(b)
    return fadd(t, bb[None, :])


def kernel(y, g, b, eps: float, layernorm: bool, shift: int = 0):
    """Rows y (R, K) FP32 -> z FP32 (R, K), q int16 (R, K), s FP32 (R,) (s = the stored scale, * 2^shift)."""
    y = np.ascontiguousarray(_f32(y))
    mean, var = row_scalars(y, layernorm)
    rstd = rstd_of(var, eps)
    z = normalize(y, mean, rstd, g, b)
    q, s = contract.quant_rows(z, shift)
    return z, q, s, mean, rstd


def torch_quant(z: np.ndarray):
    """The contract quantizer on any FP32 rows (what the int16 path does to torch's own norm output)."""
    return contract.quant_rows(_f32(z))


if __name__ == "__main__":  # self-checks of the scalar units
    from fractions import Fraction

    rng = np.random.default_rng(0)
    bad = 0
    for it in range(20000):
        n = int(rng.integers(0, 1 << 62)) << int(rng.integers(0, 9)) | int(rng.integers(0, 256))
        if it % 5 == 0:
            n = int(rng.integers(1, 1 << 40))
        d = int(rng.integers(1, 1 << 20)) if it % 2 else int(rng.integers(8, 1025)) ** 2
        k = int(rng.integers(-200, 60))
        got = div_rn_bits(n, d, k)
        want = np.float32(0.0)
        if n:
            fr = Fraction(n, d) * Fraction(2) ** k
            # correctly rounded to 24 bits with an unbounded exponent, then flushed / overflowed
            e = math.floor(math.log2(fr))
            while Fraction(2) ** e > fr:
                e -= 1
            while Fraction(2) ** (e + 1) <= fr:
                e += 1
            sc = fr / Fraction(2) ** (e - 23)
            m = math.floor(sc)
            rem = sc - m
            if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m % 2):
                m += 1
            if m == 1 << 24:
                m >>= 1
                e += 1
            f = e + 127
            want = 0 if f <= 0 else (0x7F800000 if f >= 255 else (f << 23) | (m & 0x7FFFFF))
        if int(want) != got:
            bad += 1
    print("div_rn_bits vs Fraction:", "exact" if not bad else f"{bad} mismatches")
    a = np.exp(rng.uniform(-80, 80, 200000)).astype(np.float32)
    a = a[(a > 1.2e-38) & np.isfinite(a)]
    r = rsqrt_nr(a).astype(np.float64)
    ex = 1.0 / np.sqrt(a.astype(np.float64))
    print("rsqrt_nr max rel err", float(np.abs((r - ex) / ex).max()), "(2^-24 =", 2.0**-24, ")")
    y = (rng.standard_normal((64, 768)) * 3 + 0.5).astype(np.float32)
    y[:, 7] *= 40
    g = (rng.random(768) + 0.5).astype(np.float32)
    b = (rng.standard_normal(768) * 0.1).astype(np.float32)
    z, q, s, mean, rstd = kernel(y, g, b, 1e-6, True)
    yd = y.astype(np.float64)
    zt = (yd - yd.mean(1, keepdims=True)) / np.sqrt(yd.var(1, keepdims=True) + 1e-6) * g + b
    print("LN z vs float64 max abs / row max", float((np.abs(z - zt).max(1) / np.abs(zt).max(1)).max()))
    c = np.full((3, 960), 2.0, np.float32)
    c[1] = -1e4
    c[2] = 3.7
    zc, qc, sc_, _, _ = kernel(c, np.ones(960, np.float32), np.zeros(960, np.float32), 1e-6, True)
    print("all-equal LN rows: max |z|", float(np.abs(zc).max()), "q nonzero", int((qc != 0).sum()))
