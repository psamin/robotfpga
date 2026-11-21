"""Test vectors for tb_attn.cpp from real SmolVLA -12392 attention calls (check_policy.py --dump).

For every site (vision layer 0 image 0, LM prefix layer 0, expert self-attention and cross-attention of step 0)
this writes one DDR image (vectors/<site>/mem.bin) holding FP32 Q, K, V as the projections (or the KV cache)
leave them, the key-rank vector of the mask, the attn_pack descriptors and the int16 P of the softmax (torch FP32
softmax + the shared quantizer here: the softmax kernel is not part of this test), plus the expected bits of every
operand and output from the torch emulation (attn_int16.py), and script.txt:
  pack Q / K / V  ->  engine Q K^T (all KV heads, one call)  ->  [P placed]  ->  engine P V  ->  pack o_proj input
with a bit-exact check after each step. The chain is also asserted equal to attn_int16.attention (the function the
full-policy check runs), so the layouts carry exactly the arithmetic whose policy error check_policy.py measures.

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\attention\\make_vectors.py [vectors/calls.pt]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attn_int16 as A  # noqa: E402

SUFFIX = 50  # action tokens: expert self-attention keys = prefix KV cache + these
O_PROJ_M = {"vision": 768, "prefix": 960, "expert_self": 720, "expert_cross": 720}  # o_proj outputs (s_row)


def u32(a) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(a).tobytes(), np.uint32)


class Mem:
    """Bump allocator of 128-bit words; regions start on 16-word boundaries with 16 words of slack."""

    def __init__(self):
        self.top = 16
        self.regions = []

    def alloc(self, nwords: int, data=None) -> int:
        off = self.top
        self.top += A.pad_to(nwords, 16) + 16
        if data is not None:
            d = u32(data)
            assert d.size <= 4 * nwords, (d.size, nwords)
            self.regions.append((off, d))
        return off

    def image(self) -> np.ndarray:
        m = np.zeros(self.top * 4, np.uint32)
        for off, d in self.regions:
            m[4 * off: 4 * off + d.size] = d
        return m


def ytiled_input(mem: Mem, x2d: torch.Tensor, nseg: int):
    """x2d (T, nseg * 64) FP32 as an engine call writes it (Y[2 nseg][Np][32]) -> region."""
    t = x2d.shape[0]
    npad = A.pad_to(t, A.TB)
    full = torch.zeros(npad, x2d.shape[1])
    full[:t] = x2d
    off = mem.alloc(npad * x2d.shape[1] // 4, A.y_tiled(full.numpy()))
    return A.region_ytiled(off, npad, nseg)


def rowmajor_input(mem: Mem, x2d: torch.Tensor, nseg: int):
    off = mem.alloc(x2d.numel() // 4, x2d.contiguous().numpy().astype(np.float32))
    return A.region_rowmajor(off, nseg)


def build(site: str, rec: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    q, k, v = rec["q"][0].float(), rec["k"][0].float(), rec["v"][0].float()  # (H, Tq, D), (Hkv, Tk, D)
    mask = None if rec["mask"] is None else rec["mask"][0].bool()
    scale = float(rec["scale"])
    H, Tq, D = q.shape
    Hkv, Tk = k.shape[0], k.shape[1]
    G = H // Hkv
    assert D == 64 and scale == 0.125
    N = G * Tq  # stacked rows per KV head
    Np = A.pad_to(N, A.TB)
    Mk = A.pad_to(Tk, A.TB)
    XS_qk, WS_qk = A.xmem_words(Mk, N, 64), Mk // 32 * 64
    XS_pv, WS_pv = A.xmem_words(64, N, Tk), 2 * Tk
    YS_qk, YS_pv = Mk * Np // 4, 64 * Np // 4
    mem = Mem()
    script, info = [], dict(site=site, H=H, Hkv=Hkv, G=G, Tq=Tq, Tk=Tk, N=N, Np=Np, Mk=Mk)

    def expect(name, off, arr, f32=0):
        d = u32(arr)
        assert d.size % 4 == 0
        fn = f"exp_{name}.bin"
        d.tofile(out / fn)
        script.append(f"check {name} {off} {d.size // 4} {fn} {f32}")

    # ---- inputs as the previous calls leave them
    reg_q = ytiled_input(mem, q.permute(1, 0, 2).reshape(Tq, H * D), H)
    k2d, v2d = k.permute(1, 0, 2).reshape(Tk, Hkv * D), v.permute(1, 0, 2).reshape(Tk, Hkv * D)
    if site == "expert_self":  # keys = the prefix KV cache (row-major host buffer) + this step's suffix (Y tiles)
        nsplit = Tk - SUFFIX
        reg_k = [rowmajor_input(mem, k2d[:nsplit], Hkv), ytiled_input(mem, k2d[nsplit:], Hkv)]
        reg_v = [rowmajor_input(mem, v2d[:nsplit], Hkv), ytiled_input(mem, v2d[nsplit:], Hkv)]
    else:
        nsplit = Tk
        reg_k, reg_v = [ytiled_input(mem, k2d, Hkv)], [ytiled_input(mem, v2d, Hkv)]
    if mask is not None:
        krank, qrank, ok = A.rank_descriptor(mask[None])
        assert ok
        kr = np.full(A.pad_to(Tk, 8), A.KRANK_NEVER, np.int16)
        kr[:Tk] = krank[0].numpy().astype(np.int16)
        krank_off = mem.alloc(kr.size // 8, kr)
        kval = mask.any(0).float()
    else:
        krank_off = A.NONE
        kval = torch.ones(Tk)

    # ---- the engine buffers: xmem images (one per KV head), port arrays [p][head][g][k], outputs
    x_qk = mem.alloc(Hkv * XS_qk)
    w_qk = mem.alloc(4 * Hkv * WS_qk)
    y_qk = mem.alloc(Hkv * YS_qk)
    x_pv = mem.alloc(Hkv * XS_pv)
    w_pv = mem.alloc(4 * Hkv * WS_pv)
    y_pv = mem.alloc(Hkv * YS_pv)
    stok_qk = x_qk + Mk // 4
    d_qx = A.desc("QX", Hkv, N, Np, Tq, G, 1, scale, A.NONE, Tq, [reg_q], stok_qk + Np // 4, XS_qk, 1, stok_qk, XS_qk)
    d_kw = A.desc("KW", Hkv, Tk, Mk, Tk, 1, 1, 1.0, krank_off, nsplit, reg_k, w_qk, WS_qk, Hkv * WS_qk, x_qk, XS_qk)
    d_vw = A.desc("VW", Hkv, Tk, Tk, Tk, 1, 1, 1.0, krank_off, nsplit, reg_v, w_pv, WS_pv, Hkv * WS_pv, x_pv, XS_pv)
    desc_off = mem.alloc(3 * 24, np.concatenate([d_qx, d_kw, d_vw]))
    script.append(f"pack {desc_off} 3")

    # ---- expected operands (attn_int16.quant: the function the policy check uses), padded rows are zeros
    k0, v0 = k * kval[None, :, None], v * kval[None, :, None]
    qs = torch.zeros(Hkv, Np, D)
    qs[:, :N] = q.reshape(Hkv, N, D)  # row t = g * Tq + i of KV head j
    qq, sq = A.quant(qs, -1, scale)
    kp = torch.zeros(Hkv, Mk, D)
    kp[:, :Tk] = k0
    kq, sk = A.quant(kp, -1)
    vq, sv = A.quant(v0, -2)  # (Hkv, Tk, D), (Hkv, 1, D)
    img_qk = [A.xmem_image(sk[j, :, 0].numpy(), sq[j, :, 0].numpy(), qq[j].numpy()) for j in range(Hkv)]
    expect("x_qk", x_qk, np.concatenate(img_qk))
    wk = [A.w_ports(kq[j].numpy()).reshape(4, -1) for j in range(Hkv)]
    expect("w_qk", w_qk, np.concatenate([wk[j][p] for p in range(4) for j in range(Hkv)]))
    wv = [A.w_ports(vq[j].T.numpy()).reshape(4, -1) for j in range(Hkv)]
    expect("w_pv", w_pv, np.concatenate([wv[j][p] for p in range(4) for j in range(Hkv)]))

    # ---- engine Q K^T: all KV heads in one call
    script.append(f"engine {Mk} 64 {N | (Hkv - 1) << 16} " + " ".join(str(w_qk + p * Hkv * WS_qk) for p in range(4))
                  + f" {x_qk} {y_qk}")
    scores = A.engine(qq, sq, kq, sk.transpose(-1, -2))  # (Hkv, Np, Mk)
    expect("scores", y_qk, np.concatenate([A.y_tiled(scores[j].numpy()) for j in range(Hkv)]), 1)

    # ---- softmax (torch FP32 here) -> int16 P per row, written as the softmax kernel would: the X / s_tok sections
    s4 = scores[:, :N, :Tk].reshape(H, Tq, Tk)
    if mask is not None:
        s4 = torch.where(mask[None], s4, torch.finfo(torch.float32).min)
    p = torch.softmax(s4, -1)
    pp = torch.zeros(Hkv, Np, Tk)
    pp[:, :N] = p.reshape(Hkv, N, Tk)
    pq, sp = A.quant(pp, -1)
    for j in range(Hkv):
        img = A.xmem_image(np.zeros(64, np.float32), sp[j, :, 0].numpy(), pq[j].numpy())
        mem.regions.append((x_pv + j * XS_pv + 16, img[64:]))  # behind the 16 s_row words attn_pack writes
    img_pv_srow = [sv[j, 0].numpy().astype(np.float32) for j in range(Hkv)]
    for j in range(Hkv):
        script.append(f"check srow_pv{j} {x_pv + j * XS_pv} 16 exp_srow_pv{j}.bin 0")
        u32(img_pv_srow[j]).tofile(out / f"exp_srow_pv{j}.bin")

    # ---- engine P V
    script.append(f"engine 64 {Tk} {N | (Hkv - 1) << 16} " + " ".join(str(w_pv + p * Hkv * WS_pv) for p in range(4))
                  + f" {x_pv} {y_pv}")
    o = A.engine(pq, sp, vq.transpose(-1, -2), sv)  # (Hkv, Np, 64)
    expect("out", y_pv, np.concatenate([A.y_tiled(o[j].numpy()) for j in range(Hkv)]), 1)
    o_heads = o[:, :N].reshape(H, Tq, D)
    ref = A.attention(q[None], k[None], v[None], scale, None if mask is None else mask[None])[0]
    same = bool(torch.equal(o_heads, ref))
    print(f"{site}: layout chain == attn_int16.attention (the policy emulation) bit for bit: {same}")
    assert same

    # ---- pack the o_proj input: X [pad32(Tq)/32][H*64][32] + s_tok, one scale per query over all heads
    Mo = A.pad_to(O_PROJ_M[site], 32)
    XS_o = A.xmem_words(Mo, Tq, H * D)
    x_o = mem.alloc(XS_o)
    npo = A.pad_to(Tq, A.TB)
    reg_o = A.region(y_pv, 8, Np * 8, [(h // G) * YS_pv + (h % G) * Tq * 8 for h in range(H)])
    d_ox = A.desc("OX", 1, Tq, npo, Tq, 0, H, 1.0, A.NONE, Tq, [reg_o], x_o + Mo // 4 + npo // 4, 0, 1,
                  x_o + Mo // 4, 0)
    dox_off = mem.alloc(24, d_ox)
    script.append(f"pack {dox_off} 1")
    op = torch.zeros(npo, H * D)
    op[:Tq] = o_heads.permute(1, 0, 2).reshape(Tq, H * D)
    oq, so = A.quant(op, -1)
    img_o = A.xmem_image(np.zeros(Mo, np.float32), so[:, 0].numpy(), oq.numpy())
    expect("x_o", x_o + Mo // 4, img_o[Mo:])  # s_tok + X (the s_row section is the o_proj weights', host-written)

    mem.image().tofile(out / "mem.bin")
    (out / "script.txt").write_text("\n".join(script) + "\n")
    info.update(words=mem.top, checks=sum(1 for s in script if s.startswith("check")))
    print(f"  {out}: {json.dumps(info)}")
    return info


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "vectors" / "calls.pt"
    calls = torch.load(src, weights_only=False)
    infos = [build(site, calls[site], HERE / "vectors" / site) for site in ("vision", "prefix", "expert_self",
                                                                            "expert_cross")]
    (HERE / "vectors" / "cases.json").write_text(json.dumps(infos, indent=1))


if __name__ == "__main__":
    main()
