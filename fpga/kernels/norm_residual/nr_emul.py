"""Bit-exact model of the NORM mode (contract 1.0, fpga/kernels/CONTRACT.md) as built in norm_residual.h.

tb_nr.cpp checks the C++ against it byte for byte on whole DDR images (gen_vectors.py writes them).

NORM reads FP32 rows A (K values, K % 8 == 0, 8 <= K <= 1024), normalizes each row (LayerNorm with F_LN, else
RMSNorm) and writes either int16 rows + per-row scales (F_QOUT, through the contract quantizer) or FP32 rows.
The residual add is not here: it is the GEMM epilogue (contract 2.3). Per row, in the hardware's order:

  stats (block floating point, one pass with a running exponent, 4 values = one io word at a time):
    e_i, m_i   exponent field and 24-bit significand of y_i (e_i = 0, a zero or denormal: m_i = 0)
    E          running max of e_i, updated once per word (before the word's values are added)
    when E grows by d: S := floor((S + 2^(d-1)) / 2^d), Q := floor((Q + 2^(2d-1)) / 2^(2d)) (round half up)
    Y_i        sign * round_half_away(m_i * 2^(1 + e_i - E)), |Y_i| < 2^25 (y_i ~ Y_i 2^(E-151))
    S += Y_i, Q += Y_i^2 (exact integers)
  row scalars, correctly rounded (bit-serial integer division):
    num  = max(K Q - S^2, 0) (LayerNorm) or K Q (RMSNorm)
    mean = rn(S / K) * 2^(E-151)        (LayerNorm; 0 for RMSNorm)
    var  = rn(num / K^2) * 2^(2 (E-151))
    results below 2^-126 flush to +0, above the FP32 range give +inf
  rstd (integer arithmetic, no FP32 core):
    ve = max(fl(var + eps), 2^-126)  (IEEE RNE add, denormal inputs flushed); rstd = rn(1 / sqrt(ve)),
    correctly rounded; 0 if ve is inf
  normalize, per element (FP32 cores, denormals flushed with the sign kept, contract 2.3):
    z = ((y - mean) * rstd) * g + b   (RMSNorm: b = 0)
  output: F_QOUT: q, s = contract.quant_rows(z, p4) (the shared quantizer, quantize/quant_int16.h);
          else FP32 z.

Against the pre-contract kernel (review defects, build/brief/integration_inputs.json): mean / var
correctly rounded (a constant LayerNorm row gives z = beta exactly, not +-32767 noise); the contract
quantizer; every FP32 core operation flushes denormals like the Xilinx cores. Against the paused conform
draft: the
statistics take a running exponent (no exponent pass, one row FIFO fewer) and rstd is correctly rounded in
integers (no FP32 core, no seed ROM).
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
INF_BITS = 0x7F800000

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
        return sign | INF_BITS
    return sign | (field << 23) | (m & 0x7FFFFF)


def fadd_pos_bits(a: int, b: int) -> int:
    """fl(a + b) of two non-negative FP32 bit patterns, RNE, denormal inputs flushed (as norm_residual.h)."""
    va = np.uint32(a if (a >> 23) & 0xFF else 0).view(np.float32)
    vb = np.uint32(b if (b >> 23) & 0xFF else 0).view(np.float32)
    with np.errstate(over="ignore"):
        return int(np.float32(va + vb).view(np.uint32))


def rsqrt_cr_bits(vb: int) -> int:
    """FP32 bits of rn(1 / sqrt(v)) for a positive normal FP32 v, correctly rounded (the hardware's
    integer algorithm: D = floor(2^76 / M'), R = isqrt(D), a sticky from both remainders)."""
    f = (vb >> 23) & 0xFF
    M = (vb & 0x7FFFFF) | 0x800000
    e = f - 150
    M2, e2 = (2 * M, e - 1) if e & 1 else (M, e)
    D, r1 = divmod(1 << 76, M2)
    R = math.isqrt(D)
    st = r1 != 0 or D != R * R
    if R >= 1 << 26:
        top, m, rb, st = 26, R >> 3, (R >> 2) & 1, st or (R & 3) != 0
    else:
        top, m, rb, st = 25, R >> 2, (R >> 1) & 1, st or (R & 1) != 0
    field = top - 38 - e2 // 2 + 127
    if rb and (st or (m & 1)):
        m += 1
    if m == 1 << 24:
        m >>= 1
        field += 1
    return (field << 23) | (m & 0x7FFFFF)


def _rshr(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """floor((x + 2^(s-1)) / 2^s) for s >= 1, x for s = 0 (int64 arrays; |x| < 2^62)."""
    s1 = np.clip(s - 1, 0, 63)
    t = x >> s1
    return np.where(s == 0, x, (t + 1) >> 1)


def stats(y: np.ndarray):
    """y: (R, K) FP32 -> S (R,) int64, Q (R,) int64, E (R,) int64 exactly as p_stats (running exponent)."""
    b = _f32(y).view(np.uint32).astype(np.int64)
    R, K = b.shape
    e = (b >> 23) & 0xFF
    m = np.where(e > 0, (b & 0x7FFFFF) | 0x800000, 0)
    neg = (b >> 31) & 1
    S = np.zeros(R, np.int64)
    Q = np.zeros(R, np.int64)
    E = np.zeros(R, np.int64)
    for j in range(0, K, 4):
        ew, mw, nw = e[:, j : j + 4], m[:, j : j + 4], neg[:, j : j + 4]
        wm = ew.max(axis=1)
        E1 = wm if j == 0 else np.maximum(E, wm)
        d = E1 - E if j else np.zeros(R, np.int64)
        S = _rshr(S, d)
        Q = _rshr(Q, 2 * d)
        sh = E1[:, None] - ew - 1  # Y = m 2^-sh, rounded half away
        shc = np.clip(sh, 1, 62)
        rnd = (mw + (np.int64(1) << (shc - 1))) >> shc
        mag = np.where(sh == -1, mw << 1, np.where(sh == 0, mw, np.where(sh <= 25, rnd, 0)))
        S = S + np.where(nw == 1, -mag, mag).sum(axis=1)
        Q = Q + (mag * mag).sum(axis=1)
        E = E1
    return S, Q, E


def row_scalars(y: np.ndarray, layernorm: bool):
    """mean, var (FP32, shape (R,)) exactly as the hardware's stats + dividers."""
    R, K = y.shape
    S, Q, E = stats(y)
    mean = np.zeros(R, np.uint32)
    var = np.zeros(R, np.uint32)
    for r in range(R):
        s_, q_, k1 = int(S[r]), int(Q[r]), int(E[r]) - 151
        num = max(K * q_ - (s_ * s_ if layernorm else 0), 0)
        var[r] = div_rn_bits(num, K * K, 2 * k1)
        if layernorm:
            mean[r] = div_rn_bits(abs(s_), K, k1, s_ < 0)
    return mean.view(np.float32), var.view(np.float32)


def rstd_of(var: np.ndarray, eps: float) -> np.ndarray:
    eb = int(np.float32(eps).view(np.uint32))
    out = np.zeros(var.shape, np.uint32)
    for i, vb in enumerate(_f32(var).view(np.uint32).tolist()):
        v = fadd_pos_bits(vb, eb)
        v = max(v, FLT_MIN_BITS)
        out[i] = 0 if v >= INF_BITS else rsqrt_cr_bits(v)
    return out.view(np.float32)


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


def _check_rsqrt(rng, n: int) -> int:
    """rsqrt_cr_bits against the exact midpoint test: (m - 1/2)^2 u^2 v <= 1 <= (m + 1/2)^2 u^2 v."""
    from fractions import Fraction

    bad = 0
    fields = rng.integers(1, 255, n)
    fracs = rng.integers(0, 1 << 23, n)
    fracs[:64] = 0
    fracs[64:128] = (1 << 23) - 1
    for f, fr in zip(fields.tolist(), fracs.tolist(), strict=True):
        vb = (f << 23) | fr
        v = Fraction((fr | 0x800000), 1 << 23) * Fraction(2) ** (f - 127)
        rb = rsqrt_cr_bits(vb)
        rf, rm = (rb >> 23) & 0xFF, (rb & 0x7FFFFF) | 0x800000
        u = Fraction(2) ** (rf - 127 - 23)  # one ulp of the result
        lo, hi = (Fraction(2 * rm - 1, 2) * u) ** 2 * v, (Fraction(2 * rm + 1, 2) * u) ** 2 * v
        if rm == 0x800000:  # at a power of two the ulp below is half as large
            lo = (Fraction(4 * rm - 1, 4) * u) ** 2 * v
        if not (lo <= 1 <= hi):
            bad += 1
    return bad


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
    print("div_rn_bits vs Fraction (20000):", "exact" if not bad else f"{bad} mismatches")
    nb = _check_rsqrt(rng, 30000)
    print("rsqrt_cr_bits correctly rounded (30000 exact midpoint checks):", "yes" if not nb else f"{nb} FAIL")
    a = np.exp(rng.uniform(-80, 80, 20000)).astype(np.float32)
    a = a[(a > 1.2e-38) & np.isfinite(a)]
    r = np.array([rsqrt_cr_bits(int(x)) for x in a.view(np.uint32)], np.uint32).view(np.float32)
    ex = 1.0 / np.sqrt(a.astype(np.float64))
    print("rsqrt_cr max rel err", float(np.abs((r - ex) / ex).max()), "(2^-24 =", 2.0**-24, ")")
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
