"""Test vectors for tb_gelu_silu.cpp: inputs, gs_emu's bit-exact y, and the contract quantizer's q and s of y.

    python make_vectors.py OUT_DIR [--nsb 9]       # synthetic cases (random, edge, small shapes, adversarial)
    policy_check.py --dump OUT_DIR                   # real activations of checkpoint -12392 (same format)

Case directory: meta.txt "mode N K nsb check" (mode 0 GELU, 1 SiLU(x) * u, 2 SiLU(x); check 0 = undefined inputs
(NaN / Inf): the testbench only checks that the run ends and stays inside its extents), x.bin (N x K FP32),
u.bin (mode 1), y.bin (N x K uint32: FP32 bits of y), q.bin (N x K int16), s.bin (N uint32: FP32 scale bits).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import gs_emu


def write_case(d: Path, mode: int, x: torch.Tensor, u: torch.Tensor | None, nsb: int, check: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    x = x.to(torch.float32).contiguous()
    N, K = x.shape
    q, s, yb = gs_emu.kernel(x, u, mode, nsb)
    (d / "meta.txt").write_text(f"{mode} {N} {K} {nsb} {int(check)}\n")
    x.numpy().tofile(d / "x.bin")
    if u is not None:
        u.to(torch.float32).contiguous().numpy().tofile(d / "u.bin")
    yb.numpy().astype(np.uint32).tofile(d / "y.bin")
    q.numpy().astype(np.int16).tofile(d / "q.bin")
    gs_emu.f2b(s.reshape(-1)).numpy().astype(np.uint32).tofile(d / "s.bin")


def edge_rows(K: int, R: float, g: torch.Generator) -> torch.Tensor:
    """Rows of special values: zeros, denormals, segment boundaries, around +-R, huge, tiny, one-hot rows."""
    rows = [torch.zeros(K)]  # all-zero row: q = 0, s = 1.0
    v = torch.zeros(K)
    v[17 % K] = 3.0
    rows.append(v)  # one non-zero element
    rows.append(torch.arange(K, dtype=torch.float32) * (2 * R / K) - R)  # sweep [-R, R)
    nb = torch.tensor([R, -R, R * (1 - 2**-24), -R * (1 - 2**-24), R * (1 + 2**-23), 1e4, -1e4, 3e38, -3e38])
    v = torch.randn(K, generator=g)
    v[: min(K, nb.numel())] = nb[: min(K, nb.numel())]
    rows.append(v)
    rows.append(torch.tensor([1e-40, -1e-40, 1.2e-38, -1.2e-38, 1e-30, -1e-30, 0.0, -0.0]).repeat(K // 8))
    seg = torch.arange(K, dtype=torch.float32) / 64.0  # exact segment boundaries (GELU NSB 9 step 1/64)
    rows.append(seg * torch.where(torch.arange(K) % 2 == 0, 1.0, -1.0))
    rows.append(torch.randn(K, generator=g) * 1e-3)
    rows.append(torch.randn(K, generator=g) * 300.0)
    rows.append(-torch.zeros(K))  # -0 row
    for c in (1.0, -1.0, 7.99, -7.99, 15.99, -15.99, 1e-20, 3e38, -3e38, 2.0**-100):  # all-equal rows
        rows.append(torch.full((K,), c))
    return torch.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--nsb", type=int, default=9)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(1234)
    o = a.out
    # GELU, K = 3072 (SigLIP fc1)
    x = torch.randn(24, 3072, generator=g) * 1.5
    x[3, 100] = 40.0  # outlier
    write_case(o / "gelu_rand", 0, x, None, a.nsb)
    write_case(o / "gelu_edge", 0, edge_rows(3072, 8.0, g), None, a.nsb)
    # SiLU * up, K = 2560 (LM) and 2048 (expert)
    write_case(o / "silu_rand_2560", 1, torch.randn(20, 2560, generator=g) * 2.0, torch.randn(20, 2560, generator=g), a.nsb)
    e = edge_rows(2048, 16.0, g)
    e = torch.where(e.abs() > 1e4, torch.sign(e) * 1e4, e)  # keep |x * up| finite (y = Inf is undefined)
    ue = torch.randn(e.shape, generator=g)
    ue[2, :8] = torch.tensor([0.0, -0.0, 1e-40, 3e38, -3e38, 1e-30, 1.0, -1.0])
    ue[3, :] = 1e30  # large y: |y| up to 1e34
    ue[6, :] = 1e-38  # denormal-range products flush to 0
    write_case(o / "silu_edge_2048", 1, e, ue, a.nsb)
    # gate in the far tails times a large up (review: the 2^-21 tail resolution; documented limit)
    xt = torch.tensor([[-15.99, -16.5, -17.0, -12.0, -8.0, -1.3, 0.5, 3.0] * 8])
    ut = torch.tensor([[1e6, 1e6, 3.0, 7.5, 1e3, 7.5, 1.0, -2.0] * 8])
    write_case(o / "silu_tail_64", 1, xt, ut, a.nsb)
    # plain SiLU (mode 2), K = 720 (action_time_mlp: 50 x 720 per Euler step)
    xs = torch.randn(50, 720, generator=g) * 2.0
    xs[7, 5] = -30.0
    xs[9, :] = 0.0
    write_case(o / "silu1_720", 2, xs, None, a.nsb)
    write_case(o / "silu1_edge_2048", 2, edge_rows(2048, 16.0, g), None, a.nsb)
    # small shapes: K = 8 / 16 / 24 / 120 (K / 4 < GAP: the SILU_MUL idle iterations), N = 1 / 2 / 3 / 5 / 7
    for mode in (0, 1, 2):
        for K in (8, 16, 24, 120):
            for N in (1, 2, 3, 5, 7):
                xk = torch.randn(N, K, generator=g) * 4.0
                uk = torch.randn(N, K, generator=g) if mode == 1 else None
                write_case(o / f"small_m{mode}_k{K}_n{N}", mode, xk, uk, a.nsb)
    # amax at the bottom of the range (2^-109 .. 2^-125) and tiny gates
    xa = torch.stack([torch.full((64,), 2.0**-e) * torch.sign(torch.randn(64, generator=g)) for e in range(109, 126)])
    write_case(o / "tiny_m0", 0, xa, None, a.nsb)
    write_case(o / "tiny_m1", 1, xa, torch.full_like(xa, 3.0), a.nsb)
    # undefined inputs: NaN / Inf rows (only the run's termination and extents are checked)
    xn = torch.randn(6, 256, generator=g)
    xn[1, 5] = float("nan")
    xn[3, 7] = float("inf")
    xn[4, 9] = float("-inf")
    for mode in (0, 1, 2):
        write_case(o / f"nan_m{mode}", mode, xn, torch.randn(6, 256, generator=g) if mode == 1 else None, a.nsb, check=False)
    print("wrote", o)


if __name__ == "__main__":
    main()
