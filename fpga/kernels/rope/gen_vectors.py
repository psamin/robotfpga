"""Test vectors for tb_rope.cpp from rope_emu.py (the bit-exact model) and contract.py (the quantizer).

Real cases are captured from -12392's frame 0 (apply_rope inputs: prefix layer 0 Q, prefix layer 1 K,
expert step 0 layer 0 self-attention Q and K at positions 45..94, expert step 0 layer 1 cross-attention Q
at 0..49). Synthetic cases stress the edges: wide exponent spread inside a group, zeros, -0, denormals,
all-zero and tiny segments, large positions, T not a multiple of 4 or 8, H = 64, all-equal rows, results
at the 2^-126 flush boundary (random and crafted), FP32 overflow.
Per case directory (natural order, one row = H heads of 64 values):
  meta.txt  "T H F32ONLY"     F32ONLY = 1: infinite results, so only the FP32 output is defined
  x.bin     T*H*64 float32 [t][h*64]      tab.bin  T*64 int32 (table rows NC | NS, rope_emu.table_rows)
  f32.bin   T*H*64 float32  the FP32 rotation (rope_emu.rope_rows)
  q.bin     T*H*64 int16    contract.quant_rows of each (t, h) segment of f32 (shift 0)
  s.bin     T*H float32     its stored scales [t][h]
The testbench places these in every layout of the contract (FP32 rows, int16 segments + scales, KW
tiling) and derives the expected bytes with contract.h's own helpers and ctr::ref::quant_row (it re-checks
q.bin / s.bin against that).

    C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\rope\gen_vectors.py [out_dir]
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rope_emu as R  # noqa: E402


def write_case(d: Path, x_nat: torch.Tensor, pos: torch.Tensor) -> None:
    """x_nat [T, H, 64] FP32 natural order, pos [T] ints."""
    d.mkdir(parents=True, exist_ok=True)
    T, H, _ = x_nat.shape
    tab = R.table_rows(pos)
    y = R.rope_rows(x_nat, tab)
    f32only = int(not bool(torch.isfinite(y).all()))
    (d / "meta.txt").write_text(f"{T} {H} {f32only}\n")
    x_nat.float().contiguous().numpy().astype(np.float32).tofile(d / "x.bin")
    tab.numpy().astype(np.int32).tofile(d / "tab.bin")
    y.numpy().astype(np.float32).tofile(d / "f32.bin")
    with np.errstate(all="ignore"):
        q, s = R.quant_segments(y.reshape(T * H, 64))
    q.numpy().astype(np.int16).tofile(d / "q.bin")
    s.numpy().astype(np.float32).tofile(d / "s.bin")


def capture():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import lerobot.policies.smolvla.smolvlm_with_expert as swe
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    root = HERE.parents[2] / "build/ckpt/smolvla-12392"
    policy = (
        SmolVLAPolicy.from_pretrained(str(root / "smolvla-so101_medicine_bottle_pickplace-12392-native"))
        .eval()
        .float()
    )
    fr = torch.load(sorted((root / "batches-12392").glob("frame_*.pt"))[0], weights_only=False)
    got, orig = [], swe.apply_rope

    def hook(x, positions, max_wavelength=10_000):
        got.append((x[0].detach().clone(), positions[0].detach().clone()))
        return orig(x, positions, max_wavelength)

    swe.apply_rope = hook
    try:
        with torch.no_grad():
            policy.predict_action_chunk(copy.deepcopy(fr["batch"]), noise=fr["noise"].clone())
    finally:
        swe.apply_rope = orig
    # call order: prefill 16 x (Q, K); then per step 16 layers: even (Q, K), odd (Q)
    return {
        "prefix_l0_q": got[0],
        "prefix_l1_k": got[3],
        "expert_s0_l0_q": got[32],
        "expert_s0_l0_k": got[33],
        "expert_s0_l1_q": got[34],
    }


def synthetic(seed: int, T: int, H: int):
    g = torch.Generator().manual_seed(seed)
    mag = torch.exp(torch.randn(T, H, 64, generator=g) * 6.0)  # e^(+-18): 2^-26 .. 2^26 within a head
    sign = torch.where(torch.rand(T, H, 64, generator=g) < 0.5, -1.0, 1.0)
    x = (mag * sign).float()
    flat = x.view(-1)
    idx = torch.randperm(flat.numel(), generator=g)
    flat[idx[:20]] = 0.0
    flat[idx[20:30]] = -0.0
    flat[idx[30:40]] = torch.tensor(
        [1e-40, -3e-39, 1e-45, 5e-41, -1e-44, 2e-39, 1.1e-38, -1.17e-38, 7e-39, 1e-42]
    )
    x[0, 0] = 0.0  # all-zero segment: scale 1.0
    x[1 % T, 0] = torch.randn(64, generator=g) * 1e-36  # tiny segment: s = 1e-30 / 32767, q = 0
    x[2 % T, 1 % H] = torch.randn(64, generator=g) * 3e4  # large
    x[3 % T, 0, :32] = 0.0  # one half zero
    x[4 % T, H - 1] = -0.0  # all -0: scale 1.0
    pos = torch.randint(0, 1000, (T,), generator=g)
    pos[0] = 0
    return x, pos


def all_equal(T: int, H: int):
    x = torch.empty(T, H, 64)
    for t in range(T):
        for h in range(H):
            x[t, h] = [1.0, -3.0, 0.5, 1e-3][(t + h) % 4]
    return x, torch.arange(T) * 7


def boundary(seed: int, T: int, H: int):
    """Inputs of exponent 1..3 at small positions: many rotated values straddle 2^-126 (round, then flush)."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randint(1, 4, (T, H, 64), generator=g)
    m = torch.randint(0, 1 << 23, (T, H, 64), generator=g)
    s = torch.randint(0, 2, (T, H, 64), generator=g)
    x = ((s << 31) | (e << 23) | m).to(torch.int32).view(torch.float32)
    pos = torch.randint(1, 4, (T,), generator=g)
    return x.clone(), pos


def crafted_boundary():
    """40 tokens, 2 heads. Head 0 of token t holds one value x1[i] of exponent field 1 (all others 0) at
    a position whose table entry |NC_i| < 2^17 makes the rotated lane |O| have bitlen 42 with
    N = |O| >> 16 in 2^26 - {5, 4, 3, 2, 1}: the value N 2^-152 sits just below 2^-126. Rounded to 24 bits
    first (the hardware and the contract), 2^26 - 4 and - 3 flush to +0 and - 2, - 1 round up to 2^-126;
    the old emulator (FP32 rounding with subnormals, then flush) gave 2^-126 for - 4 and - 3. Head 1:
    random inputs of exponent 1..3."""
    pos_all = torch.arange(0, 64)
    nc = R.table_rows(pos_all)[:, :32].long()
    T, H = 40, 2
    x = torch.zeros(T, H, 64, dtype=torch.int32)
    pos = torch.zeros(T, dtype=torch.int64)
    t = 0
    for target in ((1 << 26) - 5, (1 << 26) - 4, (1 << 26) - 3, (1 << 26) - 2, (1 << 26) - 1):
        found = 0
        for p in range(1, 64):
            for i in range(0, 32, 2):
                c = abs(int(nc[p, i]))
                if found == 8 or c == 0 or c >= (1 << 17):
                    continue
                lo, hi = target << 16, (target + 1) << 16
                a = -(-lo // c)
                a += (-a) % 4
                if a * c < hi and (1 << 25) <= a < (1 << 26) and (a * c).bit_length() == 42:
                    v = ((t % 2) << 31) | (1 << 23) | ((a - (1 << 25)) >> 2)
                    x[t, 0, i] = v - (1 << 32) if v >= 1 << 31 else v  # the int32 view of the FP32 bits
                    pos[t] = p
                    t += 1
                    found += 1
        assert found == 8, hex(target)
    g = torch.Generator().manual_seed(9)
    e = torch.randint(1, 4, (T, 64), generator=g)
    m = torch.randint(0, 1 << 23, (T, 64), generator=g)
    sg = torch.randint(0, 2, (T, 64), generator=g)
    x[:, 1] = ((sg << 31) | (e << 23) | m).to(torch.int32)
    return x.view(torch.float32).clone(), pos


def huge(seed: int, T: int, H: int):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(T, H, 64, generator=g) * 0.4 + 0.8) * 3.0e38
    x = torch.where(torch.rand(T, H, 64, generator=g) < 0.5, -x, x).float()
    return x, torch.randint(0, 1000, (T,), generator=g)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "build/vectors"
    names = []
    for name, (x, pos) in capture().items():
        write_case(out / name, x, pos)
        names.append(name)
    for i, (T, H) in enumerate([(7, 3), (13, 5), (1, 1), (65, 1), (3, 15), (130, 2), (40, 64)]):
        x, pos = synthetic(100 + i, T, H)
        name = f"synth{i}_{T}x{H}"
        write_case(out / name, x, pos)
        names.append(name)
    for name, (x, pos) in {
        "equal_9x3": all_equal(9, 3),
        "boundary_64x2": boundary(7, 64, 2),
        "boundary_crafted_40x2": crafted_boundary(),
        "huge_5x2": huge(8, 5, 2),
    }.items():
        write_case(out / name, x, pos)
        names.append(name)
    (out / "cases.txt").write_text("\n".join(names) + "\n")
    print("wrote", out, names)


if __name__ == "__main__":
    main()
