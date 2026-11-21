"""Test vectors for tb_nr.cpp: whole DDR images before / after one kernel call, expected from nr_emul.py.

Each case file (little-endian):
  u32 magic 0x3143524E ('NRC1'), u32 NCFG = 20, NCFG x u32 config (nr_top's scalar arguments, in order),
  u32 W (image size in 128-bit words), W x 16 bytes input image, W x 16 bytes expected image.
Every region (A, B, E, C, D, F) sits in the image with sentinel words around it and in its pitch / stride gaps;
the expected image keeps the sentinels, so any write outside the kernel's outputs is a mismatch.
The script also prints the emulation's error against torch (F.layer_norm / the Llama RMSNorm) per case.

  C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\norm_residual\\gen_vectors.py OUTDIR [--real]
--real adds cases built from real -12392 norm inputs (frame 0), captured with hooks (loads the policy).
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nr_emul  # noqa: E402

F_QOUT, F_RESID, F_SUM, F_LN = 0x2, 0x4, 0x8, 0x10
SENT = 0xA5A5A5A5
MAGIC = 0x3143524E


def f32bits(x: float) -> int:
    return int(np.array([x], np.float32).view(np.uint32)[0])


class Image:
    def __init__(self):
        self.words = []  # list of np.uint32 arrays of 4

    def alloc(self, nwords: int) -> int:
        self.pad(3)
        off = len(self.words)
        self.words.extend([np.full(4, SENT, np.uint32) for _ in range(nwords)])
        self.pad(5)
        return off

    def pad(self, n):
        self.words.extend([np.full(4, SENT, np.uint32) for _ in range(n)])

    def arr(self):
        return np.stack(self.words).copy()


def put_rows(img: np.ndarray, off, pitch, stride, rows: np.ndarray, lanes_per_word: int):
    """rows (B, N, K) of uint32 (FP32 bits) or uint16 (int16 bits)."""
    B, N, K = rows.shape
    flat = img.reshape(-1).view(np.uint8)
    for b in range(B):
        for n in range(N):
            base = (off + b * stride + n * pitch) * 16
            data = rows[b, n].tobytes()
            flat[base:base + len(data)] = np.frombuffer(data, np.uint8)


def torch_norm(y, g, b, eps, ln):
    if ln:
        return F.layer_norm(y.double(), (y.shape[-1],), g.double(), b.double(), eps).float()
    h = y.double()
    return (g.double() * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))).float()


def make_case(path, name, A, Bm, g, beta, eps, ln, resid, sumo, qout, pad=(0, 0, 0, 0), rec_out=False,
              a_gap=None):
    """A, Bm: (B, N, K) FP32 tensors. pad: extra words per row pitch for A, B, C, D.
    rec_out: D and C as records (C row r right after D row r, both with pitch K/4 + C row words): WM_REC.
    a_gap: A rows with this many gap words after each row (K/8: A is a record-layout residual stream)."""
    Bn, N, K = A.shape
    KW = K // 4
    CW = K // 8 if qout else KW
    ap, bp, cp, dp = KW + pad[0], KW + pad[1], CW + pad[2], KW + pad[3]
    if a_gap is not None:
        ap = KW + a_gap
    if rec_out:
        assert sumo
        cp = dp = KW + CW
    img = Image()
    a_stride = N * ap + (2 if pad[0] else 0)
    b_stride = N * bp + (1 if pad[1] else 0)
    c_stride = N * cp + (3 if pad[2] else 0)
    d_stride = N * dp + (1 if pad[3] else 0)
    FW = (N + 3) // 4
    f_stride = FW + (2 if any(pad) else 0)
    a_off = img.alloc(Bn * a_stride)
    b_off = img.alloc(Bn * b_stride) if resid else 0
    e_off = img.alloc(2 * KW if ln else KW)
    if rec_out:
        d_stride = c_stride = N * dp + (4 if any(pad) else 0)
        d_off = img.alloc(Bn * d_stride)
        c_off = d_off + KW
    else:
        c_off = img.alloc(Bn * c_stride)
        d_off = img.alloc(Bn * d_stride) if sumo else 0
    f_off = img.alloc(Bn * f_stride) if qout else 0
    im = img.arr()
    put_rows(im, a_off, ap, a_stride, A.numpy().view(np.uint32), 4)
    if resid:
        put_rows(im, b_off, bp, b_stride, Bm.numpy().view(np.uint32), 4)
    e = g.numpy().view(np.uint32)
    if ln:
        e = np.concatenate([e, beta.numpy().view(np.uint32)])
    put_rows(im, e_off, 0, 0, e.reshape(1, 1, -1), 4)
    # expected
    y = (A + Bm) if resid else A.clone()
    y2 = y.reshape(-1, K)
    z, q, s, mean, rstd = nr_emul.kernel(y2, g, beta if ln else None, eps, ln)
    ex = im.copy()
    if sumo:
        put_rows(ex, d_off, dp, d_stride, y.numpy().view(np.uint32), 4)
    if qout:
        put_rows(ex, c_off, cp, c_stride, q.to(torch.int16).numpy().view(np.uint16).reshape(Bn, N, K), 8)
        sb = s.numpy().view(np.uint32).reshape(Bn, N)
        for b in range(Bn):
            sw = np.zeros(FW * 4, np.uint32)
            sw[:N] = sb[b]
            put_rows(ex, f_off + b * f_stride, 1, 0, sw.reshape(1, FW, 4), 4)
    else:
        put_rows(ex, c_off, cp, c_stride, z.numpy().view(np.uint32).reshape(Bn, N, K), 4)
    flags = (F_LN if ln else 0) | (F_RESID if resid else 0) | (F_SUM if sumo else 0) | (F_QOUT if qout else 0)
    cfg = [flags, K, N, Bn, f32bits(eps), a_off, ap, a_stride, b_off, bp, b_stride, e_off, c_off, cp, c_stride,
           d_off, dp, d_stride, f_off, f_stride]
    with open(path, "wb") as fh:
        fh.write(struct.pack("<II", MAGIC, len(cfg)))
        fh.write(struct.pack(f"<{len(cfg)}I", *cfg))
        fh.write(struct.pack("<I", im.shape[0]))
        fh.write(im.tobytes())
        fh.write(ex.tobytes())
    # error of the kernel (= emulation) against torch in double
    zt = torch_norm(y2, g, beta, eps, ln)
    am = zt.abs().amax(1).clamp_min(1e-30)
    zrel = float(((z - zt).abs().amax(1) / am).max())
    deq = q.float() * s[:, None]
    drel = float(((deq - zt).abs().amax(1) / am).max())
    if ln:  # torch's own FP32 LayerNorm against the same double reference (cancellation class)
        z32 = F.layer_norm(y2, (K,), g, beta, eps)
    else:
        z32 = g * (y2 * torch.rsqrt(y2.pow(2).mean(-1, keepdim=True) + eps))
    trel = float(((z32 - zt).abs().amax(1) / am).max())
    print(f"{name:22s} K={K:4d} N={N:3d} B={Bn} flags=0x{flags:02x} words={im.shape[0]:7d}  "
          f"z vs torch(f64) max {zrel:.2e} of row max|z| (torch f32 {trel:.2e}), int16 dequant {drel:.2e}")


def synth_cases(out: pathlib.Path):
    torch.manual_seed(1)
    cases = []

    def rnd(*shape, scale=1.0, off=0.0):
        return (torch.randn(*shape) * scale + off).float()

    # vision LN, residual stream with an outlier channel and a mean offset
    K = 768
    A = rnd(1, 64, K, scale=3.0, off=0.7)
    A[..., 11] *= 30
    Bm = rnd(1, 64, K, scale=0.5)
    g, beta = (torch.rand(K) + 0.3).float(), rnd(K, scale=0.2)
    make_case(out / "ln768_resid.bin", "ln768_resid", A, Bm, g, beta, 1e-6, True, True, True, True)
    # LM RMSNorm
    K = 960
    A, Bm = rnd(1, 81, K, scale=2.0), rnd(1, 81, K, scale=0.3)
    A[..., 5] += 80
    g = (torch.rand(K) * 2).float()
    make_case(out / "rms960_resid.bin", "rms960_resid", A, Bm, g, None, 1e-5, False, True, True, True)
    # expert RMSNorm, two batch items, padded pitches and strides (per-row bursts)
    K = 720
    A, Bm = rnd(2, 50, K, scale=1.5), rnd(2, 50, K, scale=0.4)
    g = (torch.rand(K) + 0.5).float()
    make_case(out / "rms720_batch_pad.bin", "rms720_batch_pad", A, Bm, g, None, 1e-5, False, True, True, True,
              pad=(2, 1, 3, 1))
    # post_layernorm: FP32 z out, no sum out
    K = 768
    A, Bm = rnd(1, 32, K, scale=4.0, off=-0.3), rnd(1, 32, K)
    g, beta = (torch.rand(K) + 0.3).float(), rnd(K, scale=0.2)
    make_case(out / "ln768_post_f32.bin", "ln768_post_f32", A, Bm, g, beta, 1e-6, True, True, False, False)
    # first layer: no residual, no sum
    K = 720
    A = rnd(1, 50, K, scale=0.8)
    g = (torch.rand(K) + 0.5).float()
    make_case(out / "rms720_first.bin", "rms720_first", A, None, g, None, 1e-5, False, False, False, True)
    # edge rows (N = 13: the last scale word is partial)
    K = 64
    rows = []
    rows.append(torch.zeros(K))                                    # zero row
    rows.append(torch.full((K,), 3.25))                            # constant row (var 0)
    rows.append(1e4 + torch.randn(K) * 1e-2)                       # huge mean, small spread
    rows.append(torch.randn(K) * 1e-20)                            # tiny values
    rows.append(torch.randn(K) * 1e15)                             # huge values
    r = torch.randn(K)
    r[3] = 1e9
    rows.append(r)                                                 # exponents spanning > 2^25
    oh = torch.zeros(K)
    oh[7] = -2.5
    rows.append(oh)                                                # one-hot
    rows.append(torch.tensor([(-1) ** i * 2.0 ** (i % 30 - 15) for i in range(K)]))  # powers of two
    rows.append(torch.randn(K) * 1e-3 + 5e-4)                      # small, offset
    rows.append(torch.full((K,), -7.0) + (torch.arange(K) % 2) * 2 ** -20)  # tiny spread at -7
    for _ in range(3):
        rows.append(torch.randn(K) * torch.rand(1) * 10)
    A = torch.stack(rows).float().unsqueeze(0)
    Bz = torch.zeros_like(A)
    Bz[0, 2] = torch.randn(K) * 1e-3
    g, beta = (torch.rand(K) + 0.3).float(), rnd(K, scale=0.2)
    make_case(out / "edge_ln64.bin", "edge_ln64", A, Bz, g, beta, 1e-6, True, True, True, True)
    make_case(out / "edge_rms64.bin", "edge_rms64", A, Bz, g, None, 1e-5, False, True, True, True)
    make_case(out / "edge_ln64_f32.bin", "edge_ln64_f32", A, Bz, g, beta, 1e-6, True, True, True, False)
    # record layouts (the fast path the sequencer should use): output records, record-layout A input,
    # N > 256 (two write chunks, two scale bursts), and a batch of records with item strides
    K = 720
    A, Bm = rnd(1, 300, K, scale=1.5), rnd(1, 300, K, scale=0.4)
    g = (torch.rand(K) + 0.5).float()
    make_case(out / "rec720_n300.bin", "rec720_n300", A, Bm, g, None, 1e-5, False, True, True, True,
              rec_out=True, a_gap=K // 8)
    K = 768
    A, Bm = rnd(2, 40, K, scale=2.0, off=0.3), rnd(2, 40, K, scale=0.5)
    g, beta = (torch.rand(K) + 0.3).float(), rnd(K, scale=0.2)
    make_case(out / "rec768_batch.bin", "rec768_batch", A, Bm, g, beta, 1e-6, True, True, True, True,
              pad=(0, 0, 0, 0), rec_out=True, a_gap=K // 8)
    make_case(out / "rec768_f32.bin", "rec768_f32", A, Bm, g, beta, 1e-6, True, True, True, False, rec_out=True)
    # A with a gap too large for the flat read (per-row bursts), contiguous C (WM_C), N = 257
    K = 960
    A = rnd(1, 257, K, scale=1.0)
    g = (torch.rand(K) * 2).float()
    make_case(out / "gap960_n257.bin", "gap960_n257", A, None, g, None, 1e-5, False, False, False, True,
              a_gap=K // 4)
    # the longest row
    K = 1024
    A, Bm = rnd(1, 9, K, scale=2.0, off=1.0), rnd(1, 9, K)
    g, beta = (torch.rand(K) + 0.3).float(), rnd(K, scale=0.2)
    make_case(out / "ln1024.bin", "ln1024", A, Bm, g, beta, 1e-6, True, True, True, True)


def real_cases(out: pathlib.Path):
    """Real norm inputs of -12392 frame 0: one call per group, y = the module input (A), no residual operand."""
    import policy_check as pc
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    p = SmolVLAPolicy.from_pretrained(str(pc.CKPT)).to("cpu").eval().float()
    vwe = p.model.vlm_with_expert
    vm = vwe.get_vlm_model()
    want = {
        "real_vision_l5_ln2": (vm.vision_model.encoder.layers[5].layer_norm2, True),
        "real_vision_post": (vm.vision_model.post_layernorm, True),
        "real_lm_l7_in": (vm.text_model.layers[7].input_layernorm, False),
        "real_lm_l15_post": (vm.text_model.layers[15].post_attention_layernorm, False),
        "real_expert_l3_post": (vwe.lm_expert.layers[3].post_attention_layernorm, False),
        "real_expert_final": (vwe.lm_expert.norm, False),
    }
    got = {}
    hooks = []
    for name, (mod, ln) in want.items():
        def hook(m, inp, outp, name=name):
            if name not in got:
                got[name] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().clone()
        hooks.append(mod.register_forward_hook(hook))
    frames = sorted(pc.BATCHES.glob("frame_*.pt"))
    d = torch.load(frames[0], weights_only=False)
    p.reset()
    with torch.no_grad():
        p.predict_action_chunk(d["batch"], noise=d["noise"])
    for h in hooks:
        h.remove()
    for name, (mod, ln) in want.items():
        y = got[name]
        A = y.unsqueeze(0)
        g = mod.weight.detach().float()
        beta = mod.bias.detach().float() if ln else None
        eps = mod.eps if ln else mod.variance_epsilon
        qout = name != "real_vision_post"
        make_case(out / f"{name}.bin", name, A, None, g, beta, eps, ln, False, False, qout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    synth_cases(out)
    if a.real:
        real_cases(out)


if __name__ == "__main__":
    main()
