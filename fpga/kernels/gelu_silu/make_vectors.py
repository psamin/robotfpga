"""Test vectors for tb_gelu_silu.cpp: inputs and the bit-exact expected outputs of gs_emu.

    python make_vectors.py OUT_DIR [--nsb 9]       # synthetic cases (random + edge values)
    policy_check.py --dump OUT_DIR                   # real activations of checkpoint -12392 (same format)

Case directory: meta.txt "mode N K nsb" (mode 0 GELU, 1 SiLU(x) * u, 2 SiLU(x)), x.bin (N x K FP32), u.bin (mode 1), q.bin (N x K int16, natural element
order), s.bin (N uint32: FP32 scale bits).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import gs_emu


def write_case(d: Path, mode: int, x: torch.Tensor, u: torch.Tensor | None, nsb: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    x = x.to(torch.float32).contiguous()
    N, K = x.shape
    q, s, _ = gs_emu.kernel(x, u, mode, nsb)
    (d / "meta.txt").write_text(f"{mode} {N} {K} {nsb}\n")
    x.numpy().tofile(d / "x.bin")
    if u is not None:
        u.to(torch.float32).contiguous().numpy().tofile(d / "u.bin")
    q.numpy().astype(np.int16).tofile(d / "q.bin")
    gs_emu.f2b(s.reshape(-1)).numpy().astype(np.uint32).tofile(d / "s.bin")


def edge_rows(K: int, R: float, g: torch.Generator) -> torch.Tensor:
    """Rows of special values: zeros, denormals, segment boundaries, around +-R, huge, tiny, one-hot rows."""
    rows = []
    rows.append(torch.zeros(K))  # all-zero row: q = 0, s = 0
    v = torch.zeros(K)
    v[17] = 3.0
    rows.append(v)  # one non-zero element
    b = torch.arange(K, dtype=torch.float32) * (2 * R / K) - R  # sweep [-R, R)
    rows.append(b)
    nb = torch.tensor([R, -R, R * (1 - 2**-24), -R * (1 - 2**-24), R * (1 + 2**-23), 1e4, -1e4, 3e38, -3e38])
    v = torch.randn(K, generator=g)
    v[: nb.numel()] = nb
    rows.append(v)
    den = torch.tensor([1e-40, -1e-40, 1.2e-38, -1.2e-38, 1e-30, -1e-30, 0.0, -0.0]).repeat(K // 8)
    rows.append(den)  # tiny / denormal row (absmax 1.2e-38 -> zero row by the ea < 16 rule? no: ea = 1)
    seg = torch.arange(K, dtype=torch.float32) / 64.0  # exact segment boundaries (GELU NSB 9 step 1/64)
    rows.append(seg * torch.where(torch.arange(K) % 2 == 0, 1.0, -1.0))
    rows.append(torch.randn(K, generator=g) * 1e-3)  # small row
    rows.append(torch.randn(K, generator=g) * 300.0)  # large row
    return torch.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--nsb", type=int, default=9)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(1234)
    # GELU, K = 3072 (SigLIP fc1)
    x = torch.randn(24, 3072, generator=g) * 1.5
    x[3, 100] = 40.0  # outlier
    write_case(a.out / "gelu_rand", 0, x, None, a.nsb)
    write_case(a.out / "gelu_edge", 0, edge_rows(3072, 8.0, g), None, a.nsb)
    # SiLU * up, K = 2560 (LM) and 2048 (expert)
    gt = torch.randn(20, 2560, generator=g) * 2.0
    up = torch.randn(20, 2560, generator=g) * 1.0
    write_case(a.out / "silu_rand_2560", 1, gt, up, a.nsb)
    e = edge_rows(2048, 16.0, g)
    ue = torch.randn(e.shape, generator=g)
    ue[2, :8] = torch.tensor([0.0, -0.0, 1e-40, 3e38, -3e38, 1e-30, 1.0, -1.0])
    ue[3, :] = 1e30  # overflow of y: saturates at the largest finite FP32
    write_case(a.out / "silu_edge_2048", 1, e, ue, a.nsb)
    # plain SiLU (mode 2), K = 720 (action_time_mlp: 50 x 720 per Euler step)
    xs = torch.randn(50, 720, generator=g) * 2.0
    xs[7, 5] = -30.0
    xs[9, :] = 0.0
    write_case(a.out / "silu1_720", 2, xs, None, a.nsb)
    write_case(a.out / "silu1_edge_2048", 2, edge_rows(2048, 16.0, g), None, a.nsb)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
