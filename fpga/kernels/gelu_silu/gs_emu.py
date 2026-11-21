"""Bit-exact torch emulation of the gelu_silu front end (fpga/kernels/gelu_silu/gelu_silu.h), contract 1.0.

    y = act(x)            GELU:      tanh-GELU(x)            = x * h(x), h = Phi_tanh
                          SILU_MUL:  silu(gate) * up         = (gate * h(gate)) * up, h = sigmoid
                          + F_NOMUL: silu(x)                 = x * h(x)
    (q, s) = contract quantizer of y (fpga/kernels/contract.py quant_rows = quantize/quant_int16.h)

h(x) comes from a table (gs_tables.py) as an exact FP32 value of 21 fraction bits; the products are FP32
multiplies with the Xilinx cores' rule (round to nearest even, denormal inputs and results flushed to zero, sign
kept: contract.ftz). The C++ testbench checks the HLS code against this file bit for bit (y, q, s).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import gs_tables  # noqa: E402

FB, VB, DB = gs_tables.FB, gs_tables.VB, gs_tables.DB
MODE_GELU, MODE_SILU_MUL, MODE_SILU = 0, 1, 2  # OP_GELU, OP_SILU_MUL, OP_SILU_MUL + F_NOMUL

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
    """Index of the leading one of non-negative int64 values < 2^53 (exact through float64 frexp); -1 for 0."""
    _, e = torch.frexp(v.to(torch.float64))
    return torch.where(v > 0, e.to(torch.int64) - 1, torch.full_like(v, -1))


def ftz_bits(b: torch.Tensor) -> torch.Tensor:
    """contract.ftz on bit patterns: a zero exponent field flushes to a zero with the sign kept."""
    return torch.where(((b >> 23) & 0xFF) == 0, b & 0x80000000, b)


def fmul_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One Xilinx FP32 multiply: ftz(ftz(a) * ftz(b)), round to nearest even (torch's CPU float32 multiply)."""
    p = b2f(ftz_bits(a)) * b2f(ftz_bits(b))
    return ftz_bits(f2b(p))


def h_bits(xb: torch.Tensor, silu: bool, nsb: int = 9) -> torch.Tensor:
    """FP32 bits of the gate h(x) in [0, 1]: Phi_tanh (GELU) or sigmoid (SiLU).

    t = |x| in fixed point (TB = NSB + 12 bits: segment index NSB bits, fraction f 12 bits);
    g21 = max(0, (V << 12) + 2048 - D f) >> 12   = g(t) = 1 - h(t) in Q0.21 (0 beyond the range R);
    H = g21 (x < 0) or 2^21 - g21 (x >= 0), h = H 2^-21 exactly (H <= 2^21 has at most 22 bits)."""
    nseg = 1 << nsb
    TB = nsb + FB
    rb = 4 if silu else 3  # R = 16 (SiLU) or 8 (GELU)
    sx = (xb >> 31) & 1
    ex = (xb >> 23) & 0xFF
    xz = ex == 0
    mx = (xb & 0x7FFFFF) | (1 << 23)
    sh = 150 + rb - TB - ex  # T = floor(|x| 2^(TB - rb)) = mx >> sh
    sat = (~xz) & (sh < 24 - TB)  # |x| >= R
    inr = (~xz) & (sh >= 24 - TB) & (sh < 24)
    T = torch.where(inr, mx >> sh.clamp(0, 63), torch.zeros_like(mx))
    idx = T >> FB
    fr = T & ((1 << FB) - 1)
    w = tab(nsb)[(nseg if silu else 0) + idx]
    V, D = w >> DB, w & ((1 << DB) - 1)
    t = (V << FB) + (1 << (FB - 1)) - D * fr
    g21 = torch.where(sat | (t < 0), torch.zeros_like(t), t >> FB)
    H = torch.where(sx == 1, g21, (1 << VB) - g21)
    p = msb(H)
    m = (H << (23 - p).clamp(0, 63)) & 0x7FFFFF
    return torch.where(H == 0, torch.zeros_like(H), ((106 + p) << 23) | m)


def act_bits(xb: torch.Tensor, ub: torch.Tensor | None, mode: int, nsb: int = 9) -> torch.Tensor:
    """FP32 bits of y, as the front end computes them (inputs: FP32 bits as int64)."""
    assert mode in (MODE_GELU, MODE_SILU_MUL, MODE_SILU)
    y = fmul_bits(xb, h_bits(xb, mode != MODE_GELU, nsb))
    if mode == MODE_SILU_MUL:
        y = fmul_bits(y, ub)
    return y


def act(x: torch.Tensor, u: torch.Tensor | None, mode: int, nsb: int = 9) -> torch.Tensor:
    """The front end's FP32 y for float tensors x (and u)."""
    return b2f(act_bits(f2b(x), f2b(u) if u is not None else None, mode, nsb))


def quant_rows(y: torch.Tensor, shift: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract quantizer over the last axis (contract.quant_rows): q int16, stored scale FP32 (..., 1)."""
    import contract

    q, s = contract.quant_rows(y.detach().to(torch.float32).cpu().numpy(), shift)
    return torch.from_numpy(q), torch.from_numpy(s)[..., None]


def kernel(x: torch.Tensor, u: torch.Tensor | None, mode: int, nsb: int = 9):
    """x (and u): (..., K) float tensors. Returns q (int16), s (FP32, (..., 1)) and the y bits (int64)."""
    yb = act_bits(f2b(x), f2b(u) if u is not None else None, mode, nsb)
    q, s = quant_rows(b2f(yb))
    return q, s, yb


def dequant(x: torch.Tensor, u: torch.Tensor | None, mode: int, nsb: int = 9) -> torch.Tensor:
    """q * s in FP32: what the next int16 linear effectively sees."""
    q, s, _ = kernel(x, u, mode, nsb)
    return q.to(torch.float32) * s


def act_exact(x: torch.Tensor, u: torch.Tensor | None, mode: int) -> torch.Tensor:
    """float64 reference of the activation."""
    x = x.double()
    if mode == MODE_GELU:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    if mode == MODE_SILU:
        return x * torch.sigmoid(x)
    return x * torch.sigmoid(x) * u.double()


def _selftest() -> None:
    g = torch.Generator().manual_seed(7)
    for mode in (MODE_GELU, MODE_SILU_MUL, MODE_SILU):
        x = torch.randn(64, 512, generator=g) * 3
        u = torch.randn(64, 512, generator=g) if mode == MODE_SILU_MUL else None
        y = act(x, u, mode).double()
        ref = act_exact(x, u, mode)
        s = ref.abs().amax(-1, keepdim=True) / 32767
        print(f"mode {mode}: max |y - ref| / s = {float(((y - ref).abs() / s).max()):.4f} steps")
    # exact cases: h(0) = 1/2 path, saturation (h = 1 for x >= R, 0 for x <= -R), zeros and denormals
    xs = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 8.0, 100.0, -8.0, -100.0, 3e38, -3e38])
    print("gelu edge y:", act(xs, None, MODE_GELU).tolist())


if __name__ == "__main__":
    _selftest()
