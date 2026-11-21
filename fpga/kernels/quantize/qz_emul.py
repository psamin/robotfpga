"""Torch / numpy model of the int16 per-token quantizer (quant_int16.h), bit for bit.

quant_ref(x)     the definition: build/brief/precision_emulation.py quant_sym(x, 32767, 1)
quant_ref_engine(x)  the same with fpga/hls/pack_float.py's convention for all-zero rows (scale 1.0
                 instead of 1e-30 / 32767; q = 0 either way), which the kernel follows
quant_kernel(x)  the kernel's integer datapath (QZ_IMPL_RECIP) re-done in torch int64: absmax on the
                 bit patterns, s from a by the shift-add division by 32767, R = floor(2^49 / S), then
                 per element Qc = X R >> 24, the 25-bit remainder fix, the FP32 rounding of X / S and the
                 half-even integer rounding with one left shift. Returns (q int16, s float32).
Runtime helpers (numpy): pack_rows / unpack_q / unpack_scales for the DDR word layouts, dequant.

    python qz_emul.py --selftest      # kernel == reference on random and adversarial rows
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

QMAX = 32767
TINY = 1e-30  # precision_emulation.py TINY
TINY_BITS = 0x0DA24260  # bits of float32(1e-30)
ONE_BITS = 0x3F800000  # scale of an all-zero row (pack_float.py _quantize_int16)
M25 = (1 << 25) - 1


# ------------------------------------------------------------------------------- reference
def quant_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """precision_emulation.py quant_sym(x, 32767, dim=1) on an FP32 (rows, cols) tensor."""
    s = x.abs().amax(1, keepdim=True).clamp_min(TINY) / QMAX
    q = torch.round(x / s).clamp_(-QMAX, QMAX)
    return q.to(torch.int16), s.squeeze(1)


def quant_ref_engine(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """quant_ref with pack_float's scale 1.0 for all-zero rows (the kernel's and the int16 engine's convention)."""
    q, s = quant_ref(x)
    return q, torch.where(x.abs().amax(1) == 0, torch.ones_like(s), s)


# ------------------------------------------------------------------------------- bit helpers
def f32_bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def bits_f32(b: torch.Tensor) -> torch.Tensor:
    b = b.to(torch.int64) & 0xFFFFFFFF
    return torch.where(b >= 2**31, b - 2**32, b).to(torch.int32).view(torch.float32)


# ------------------------------------------------------------------------------- the kernel
def scale_bits(a: torch.Tensor) -> torch.Tensor:
    """s = fl32(max(a, 1e-30f) / 32767) from the absmax bit pattern a (int64), integers only."""
    a = torch.clamp_min(a, TINY_BITS)
    ea = a >> 23
    n = ((a & 0x7FFFFF) | (1 << 23)) << 17
    q = (n + (n >> 15) + (n >> 30)) >> 15  # exact quotient or one less
    rem = n - q * QMAX
    fix = (rem >= QMAX).to(torch.int64)
    q = q + fix
    rem = rem - fix * QMAX
    st0 = rem != 0
    big = ((q >> 26) & 1).bool()
    m = torch.where(big, q >> 3, q >> 2)
    rb = torch.where(big, (q >> 2) & 1, (q >> 1) & 1).bool()
    st = torch.where(big, (q & 3) != 0, (q & 1) != 0) | st0
    es = ea - 15 + big.to(torch.int64)
    m = m + (rb & (st | (m & 1).bool())).to(torch.int64)
    ov = (m >> 24).bool()
    m = torch.where(ov, torch.full_like(m, 1 << 23), m)
    es = es + ov.to(torch.int64)
    return (es << 23) | (m & 0x7FFFFF)


def rne_sat(m: torch.Tensor, sh: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """quant_int16.h rne_sat: round m * 2^-sh half to even, saturate at 32767, sign."""
    k = (sh - 9).clamp(0, 16)
    y = (m >> 8) >> k  # floor(m / 2^(sh - 1))
    rb = (y & 1).bool()
    ip = y >> 1
    below = (torch.ones_like(sh) << (sh - 1).clamp(0, 24)) - 1
    st = (m & below) != 0
    v = ip + (rb & (st | (ip & 1).bool())).to(torch.int64)
    v = v.clamp_max(QMAX)
    v = torch.where(sh < 9, torch.full_like(v, QMAX), v)
    v = torch.where(sh > 25, torch.zeros_like(v), v)
    return torch.where(neg, -v, v)


def row_scale(a: torch.Tensor):
    """quant_int16.h row_scale on absmax bit patterns (rows,): s bits, S, R, T, B (each (rows, 1))."""
    sb = scale_bits(a)
    es = (sb >> 23)[:, None]
    big_s = ((sb & 0x7FFFFF) | (1 << 23))[:, None]
    r = torch.clamp_max((1 << 49) // big_s, (1 << 26) - 1)  # recip_sig (proved equal exhaustively in tb_quant)
    return sb, big_s, r, (1 << 25) - big_s, es + 23


def quant_elems(xb: torch.Tensor, big_s, r, t, b) -> torch.Tensor:
    """quant_int16.h quant_elem_recip on FP32 bit patterns (int64), row parameters broadcast."""
    e = (xb >> 23) & 0xFF
    neg = (xb >> 31).bool()
    xm = (xb & 0x7FFFFF) | (1 << 23)
    qc = (xm * r) >> 24
    lo = ((qc * (big_s & 0x1FFFF)) + ((((qc & 0xFF) * (big_s >> 17)) & 0xFF) << 17)) & M25
    lz = lo == 0
    ge = ~lz & (lo <= t)
    st0 = torch.where(ge, lo != t, ~lz)
    qq = qc + ge.to(torch.int64)
    big = ((qq >> 25) & 1).bool()
    m = torch.where(big, qq >> 2, qq >> 1)
    rb = torch.where(big, (qq >> 1) & 1, qq & 1).bool()
    st = torch.where(big, (qq & 1).bool() | st0, st0)
    sh = torch.where(big, b - e, b + 1 - e)
    m = m + (rb & (st | (m & 1).bool())).to(torch.int64)
    q = rne_sat(m, sh, neg)
    return torch.where(e == 0, torch.zeros_like(q), q)


def quant_kernel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel on an FP32 (rows, cols) tensor: (q int16 (rows, cols), s float32 (rows,))."""
    assert x.dtype == torch.float32 and x.dim() == 2
    xb = f32_bits(x)
    a = (xb & 0x7FFFFFFF).amax(1)  # absmax on the magnitude bits
    sb, big_s, r, t, b = row_scale(a)
    q = quant_elems(xb, big_s, r, t, b).to(torch.int16)
    sb = torch.where(a == 0, torch.full_like(sb, ONE_BITS), sb)  # all-zero row: scale 1.0 (q is 0 anyway)
    return q, bits_f32(sb)


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fl32(q * s) per row, what quant_int16.h dequant gives."""
    return q.float() * s.float()[:, None]


# ------------------------------------------------------------------------------- DDR layouts (numpy)
def pack_rows(x: np.ndarray, in_stride_words: int | None = None) -> np.ndarray:
    """FP32 (rows, cols) -> the kernel's input buffer: row r at word r * in_stride (4 FP32 a word).
    cols must be a multiple of 16; pad shorter rows with zeros (zeros change neither absmax nor q)."""
    rows, cols = x.shape
    assert cols % 16 == 0, "pad cols to a multiple of 16 with zeros"
    st = in_stride_words or cols // 4
    buf = np.zeros((rows, st * 4), np.float32)
    buf[:, :cols] = x
    return buf


def unpack_q(buf: np.ndarray, rows: int, cols: int, out_stride_words: int | None = None) -> np.ndarray:
    """int16 output buffer (bytes or int16) -> (rows, cols) int16; row r at word r * out_stride (8 a word)."""
    st = out_stride_words or cols // 8
    return np.frombuffer(np.ascontiguousarray(buf).tobytes(), np.int16)[: rows * st * 8].reshape(rows, st * 8)[:, :cols]


def unpack_scales(buf: np.ndarray, rows: int) -> np.ndarray:
    """Scale buffer (4 FP32 a word, the last word zero-padded) -> (rows,) float32."""
    return np.frombuffer(np.ascontiguousarray(buf).tobytes(), np.float32)[:rows]


def dequant_np(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * s.astype(np.float32)[:, None]


# ------------------------------------------------------------------------------- self test
def _adversarial(rows: int, cols: int, g: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, cols, generator=g) * torch.exp2(torch.randint(-30, 30, (rows, 1), generator=g).float())
    x[::7] = 0.0
    x[1::7] = torch.randn(len(x[1::7]), cols, generator=g) * 1e-38  # denormal rows (below the clamp)
    # values near half-integers of x / s
    a = x.abs().amax(1, keepdim=True).clamp_min(TINY)
    s = a / QMAX
    k = torch.randint(0, QMAX, (rows, cols), generator=g).double() + 0.5
    v = (k * s.double()).float()
    vb = f32_bits(v) + torch.randint(-2, 3, (rows, cols), generator=g)
    near = bits_f32(vb)
    near = torch.where(near.abs() <= a, near, v)
    sel = torch.rand(rows, cols, generator=g) < 0.5
    sel[::7] = False
    sel[1::7] = False
    return torch.where(sel, near * torch.where(torch.rand(rows, cols, generator=g) < 0.5, -1.0, 1.0), x)


def selftest(n_rows: int = 20000, cols: int = 512) -> None:
    g = torch.Generator().manual_seed(0)
    bad = tot = 0
    for _ in range(n_rows // 2000):
        x = _adversarial(2000, cols, g)
        qk, sk = quant_kernel(x)
        qr, sr = quant_ref_engine(x)
        bad += int((qk != qr).sum()) + int((f32_bits(sk) != f32_bits(sr)).sum())
        nz = x.abs().amax(1) != 0  # the rows where quant_sym's own scale must match too
        bad += int((f32_bits(sk[nz]) != f32_bits(quant_ref(x)[1][nz])).sum())
        tot += x.numel()
    # every significand of the absmax at a few exponents
    m = torch.arange(1 << 23, dtype=torch.int64)
    sbad = 0
    for e in (27, 127, 254):
        a = (e << 23) | m
        want = f32_bits((bits_f32(a).clamp_min(TINY) / QMAX))
        sbad += int((scale_bits(a) != want).sum())
    print(f"qz_emul selftest: {tot} elements, {bad} mismatches kernel vs quant_sym; scale over 3 x 2^23 absmax: {sbad} mismatches")
    assert bad == 0 and sbad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
