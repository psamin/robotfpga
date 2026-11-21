"""Test vectors for tb_rope.cpp from rope_emu.py (the bit-exact model).

Real cases are captured from -12392's frame 0 (apply_rope inputs: prefix layer 0 Q, prefix layer 1 K,
expert step 0 layer 0 self-attention Q at positions 45..94, expert step 0 layer 1 cross-attention Q at
0..49); synthetic cases stress the edges (wide exponent spread inside a group, zeros, -0, denormals,
all-zero and tiny rows, large positions, T not a multiple of 4). Per case directory (natural order):
  meta.txt  "T H"
  x.bin     T*H*64 float32 [t][h][64]                  tab.bin  T*64 int32 (table rows: NC | NS)
  f32.bin   T*H*64 float32 (MODE_F32 output)
  q0.bin    T*H*64 int16 (MODE_INT16 output)           s0.bin   H*ceil(T/4)*4 float32, head-major, zero padded

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\rope\\gen_vectors.py [out_dir]
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

def write_case(d: Path, x_nat: torch.Tensor, pos: torch.Tensor):
    """x_nat [T, H, 64] FP32 natural order, pos [T] ints."""
    d.mkdir(parents=True, exist_ok=True)
    T, H, _ = x_nat.shape
    tab = R.table_rows(pos)
    o, E = R.run_rows(x_nat, tab)
    (d / "meta.txt").write_text(f"{T} {H}\n")
    x_nat.float().contiguous().numpy().astype(np.float32).tofile(d / "x.bin")
    tab.numpy().astype(np.int32).tofile(d / "tab.bin")
    R.to_natural(R.to_f32(o, E)).numpy().astype(np.float32).tofile(d / "f32.bin")
    q, sc = R.quantize(o, E, 0)
    R.to_natural(q).numpy().astype(np.int16).tofile(d / "q0.bin")
    tq = (T + 3) // 4
    sh = torch.zeros(H, tq * 4, dtype=torch.float32)
    sh[:, :T] = sc.t()
    sh.numpy().astype(np.float32).tofile(d / "s0.bin")


def capture():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import lerobot.policies.smolvla.smolvlm_with_expert as swe
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    root = HERE.parents[2] / "build/ckpt/smolvla-12392"
    policy = SmolVLAPolicy.from_pretrained(str(root / "smolvla-so101_medicine_bottle_pickplace-12392-native")).eval().float()
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
    return {"prefix_l0_q": got[0], "prefix_l1_k": got[3], "expert_s0_l0_q": got[32], "expert_s0_l1_q": got[34]}


def synthetic(seed: int, T: int, H: int):
    g = torch.Generator().manual_seed(seed)
    mag = torch.exp(torch.randn(T, H, 64, generator=g) * 6.0)  # e^(+-18): 2^-26 .. 2^26 within a head
    sign = torch.where(torch.rand(T, H, 64, generator=g) < 0.5, -1.0, 1.0)
    x = (mag * sign).float()
    flat = x.view(-1)
    idx = torch.randperm(flat.numel(), generator=g)
    flat[idx[:20]] = 0.0
    flat[idx[20:30]] = -0.0
    flat[idx[30:40]] = torch.tensor([1e-40, -3e-39, 1e-45, 5e-41, -1e-44, 2e-39, 1.1e-38, -1.17e-38, 7e-39, 1e-42])
    x[0, 0] = 0.0  # all-zero pair
    x[1 % T, 0] = torch.randn(64, generator=g) * 1e-36  # tiny pair: scale flushes to 0
    x[2 % T, 1 % H] = torch.randn(64, generator=g) * 3e4  # large
    x[3 % T, 0, :32] = 0.0  # one half zero
    pos = torch.randint(0, 1000, (T,), generator=g)
    pos[0] = 0
    return x, pos


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "build/vectors"
    names = []
    for name, (x, pos) in capture().items():
        write_case(out / name, x, pos)
        names.append(name)
    for i, (T, H) in enumerate([(7, 3), (13, 5), (1, 1), (65, 1), (3, 15), (130, 2), (136, 15)]):
        x, pos = synthetic(100 + i, T, H)
        name = f"synth{i}_{T}x{H}"
        write_case(out / name, x, pos)
        names.append(name)
    (out / "cases.txt").write_text("\n".join(names) + "\n")
    print("wrote", out, names)


if __name__ == "__main__":
    main()
