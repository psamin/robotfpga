"""Test cases for tb_softmax.cpp: whole DDR images (one SOFTMAX record + scores + rank tables + output regions full
of 0xA5 canaries) and the image the kernel must leave behind, computed with the bit-exact torch emulation
(smx_emu.py). Also prints each case's error against FP64 softmax.

    python make_vectors.py [--out DIR] [--real scores.pt]

Each case directory holds params.txt ("rec_word total_words status_fp32build status_u16build"), mem.bin and
expect.bin (little-endian 128-bit words). Layouts and the record: README.md.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch

import smx_emu as E

HERE = Path(__file__).resolve().parent
F_QOUT, F_MASK, F_ENGINE, F_EMPTY_UNIFORM = 0x0002, 0x0100, 0x1000, 0x2000
ST_OK, ST_DIMS, ST_ALIGN, ST_OPCODE = 0, 1, 2, 3
L_MAX, R_MAX = 256, 256
CANARY = 0xA5A5A5A5


def f32u(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def pad(n, m):
    return (n + m - 1) // m * m


class Image:
    """Word-addressed DDR image; word 0..7 hold the record."""

    def __init__(self):
        self.n = 8
        self.puts = []

    def alloc(self, words, align=4):
        off = pad(self.n, align)
        self.n = off + max(int(words), 0)
        return off

    def put(self, off, u32):
        self.puts.append((off, np.asarray(u32, dtype=np.uint32).reshape(-1)))

    def build(self):
        mem = np.full(pad(self.n, 4) * 4 + 64, CANARY, dtype=np.uint32)
        for off, a in self.puts:
            mem[4 * off : 4 * off + a.size] = a
        return mem


def record(flags, L, N, B, p0, slots, opcode=0x40):
    """32 uint32 fields; slots: dict name -> (byte addr, pitch bytes, stride bytes)."""
    f = [0] * 32
    f[0] = (opcode & 0xFF) | (0xC5 << 8) | (flags << 16)
    f[1] = 0x5F7A0001
    f[3], f[4], f[5] = L, N, B
    for s, name in enumerate("ABCDEF"):
        a, p, st = slots.get(name, (0, 0, 0))
        f[6 + 3 * s] = a & 0xFFFFFFFF
        f[7 + 3 * s] = ((a >> 32) & 0xFF) | (p << 8)
        f[8 + 3 * s] = st
    f[24] = f32u(p0)
    x = 0x5EC0C5ED
    for v in f[:31]:
        x ^= v
    f[31] = x
    return np.array(f, dtype=np.uint32)


def words_of_rows(a: np.ndarray, lanes: int) -> np.ndarray:
    """(R, n) values of 32 / lanes bits -> uint32 array of whole 128-bit words per row (row-major)."""
    return np.ascontiguousarray(a).view(np.uint32).reshape(-1)


def write_case(out: Path, name, x, p0, *, engine=False, qout=True, ranks=None, empty="zero", a_pitch=None,
               c_pitch=None, item_gap=0, misalign=False, opcode=0x40, L_rec=None, N_rec=None, seed=0,
               emulate=True):
    """x (B, N, L) float32 scores; ranks None or (E (Be, L), D (Bd, N)) int with Be, Bd in {1, B}."""
    g = torch.Generator().manual_seed(seed)
    B, N, L = x.shape
    L4 = (L + 3) // 4
    flags = (F_QOUT if qout else 0) | (F_MASK if ranks is not None else 0) | (F_ENGINE if engine else 0) | (
        F_EMPTY_UNIFORM if empty == "uniform" else 0)
    img = Image()
    c = E.c_from_mult(p0)
    # ---- scores
    if engine:
        Np, Mk = pad(N, 32), pad(L, 32)
        G = Mk // 32
        aw = G * Np * 8
        a_stride = aw + item_gap
        a_off = img.alloc(B * a_stride)
        for b in range(B):
            y = torch.randn(Np, Mk, generator=g) * 50.0  # padding keys / rows: finite garbage
            y[:N, :L] = x[b]
            yt = y.numpy().astype(np.float32).reshape(Np, G, 32).transpose(1, 0, 2)  # [g][n][32]
            img.put(a_off + b * a_stride, np.ascontiguousarray(yt).view(np.uint32))
        a_pitch_b = 0
    else:
        a_pitch = a_pitch or L4
        a_stride = N * a_pitch + item_gap
        a_off = img.alloc(B * a_stride)
        for b in range(B):
            y = torch.randn(N, a_pitch * 4, generator=g) * 1e3
            y[:, :L] = x[b]
            img.put(a_off + b * a_stride, y.numpy().astype(np.float32).view(np.uint32))
        a_pitch_b = a_pitch * 16
    # ---- rank tables
    slots = {"A": (a_off * 16 + (8 if misalign else 0), a_pitch_b, a_stride * 16)}
    if ranks is not None:
        Ek, Dn = ranks
        Ek, Dn = torch.as_tensor(Ek).to(torch.int64), torch.as_tensor(Dn).to(torch.int64)
        e_st = pad(L, 4) // 4 if Ek.shape[0] > 1 else 0
        d_st = pad(N, 4) // 4 if Dn.shape[0] > 1 else 0
        e_off = img.alloc(max(Ek.shape[0], 1) * (pad(L, 4) // 4))
        d_off = img.alloc(max(Dn.shape[0], 1) * (pad(N, 4) // 4))
        for i in range(Ek.shape[0]):
            v = np.zeros(pad(L, 4), np.int64)
            v[:L] = Ek[i].numpy()
            img.put(e_off + i * e_st, (v & 0xFFFFFFFF).astype(np.uint32))
        for i in range(Dn.shape[0]):
            v = np.zeros(pad(N, 4), np.int64)
            v[:N] = Dn[i].numpy()
            img.put(d_off + i * d_st, (v & 0xFFFFFFFF).astype(np.uint32))
        slots["D"] = (d_off * 16, 0, d_st * 16)
        slots["E"] = (e_off * 16, 0, e_st * 16)
    # ---- outputs
    if engine:
        Np = pad(N, 32)
        cw = Np * L // 8  # X: [Np/32][L][4 words]
        c_stride = cw + item_gap
        c_off = img.alloc(B * c_stride)
        f_stride = Np // 4
        f_off = img.alloc(B * f_stride)
        slots["C"] = (c_off * 16, 0, c_stride * 16)
        slots["F"] = (f_off * 16, 0, f_stride * 16)
    else:
        c_pitch = c_pitch or ((L + 7) // 8 if qout else L4)
        c_stride = N * c_pitch + item_gap
        c_off = img.alloc(B * c_stride)
        f_stride = (N + 3) // 4
        f_off = img.alloc(B * f_stride) if qout else 0
        slots["C"] = (c_off * 16, c_pitch * 16, c_stride * 16)
        if qout:
            slots["F"] = (f_off * 16, 0, f_stride * 16)
    rec = record(flags, L if L_rec is None else L_rec, N if N_rec is None else N_rec, B, p0, slots, opcode)
    img.put(0, rec)
    mem = img.build()
    exp = mem.copy()
    # ---- expected results (emulation) and errors vs FP64
    errs = [torch.zeros(1)]
    for b in range(B if emulate else 0):
        if ranks is not None:
            Ek, Dn = ranks
            valid = E.rank_valid(torch.as_tensor(Ek[b if Ek.shape[0] > 1 else 0]),
                                 torch.as_tensor(Dn[b if Dn.shape[0] > 1 else 0]))
        else:
            valid = torch.ones(N, L, dtype=torch.bool)
        xb = x[b].float()
        if qout:
            q, s = E.kernel(xb, c, valid, empty=empty)
            p_hat = q.double() * s.double()
        else:
            p = E.kernel(xb, c, valid, fp32_out=True, empty=empty)
            p_hat = p.double()
        if engine:
            Np = pad(N, 32)
            xi = np.zeros((Np, L), np.uint16)
            xi[:N] = q.numpy().astype(np.uint16)
            xi = np.ascontiguousarray(xi.reshape(Np // 32, 32, L).transpose(0, 2, 1))  # [b][k][32 rows]
            o = c_off + b * c_stride
            exp[4 * o : 4 * o + xi.size // 2] = xi.reshape(-1).view(np.uint32)
            st = np.zeros(Np, np.float32)
            st[:N] = s[:, 0].numpy()
            o = f_off + b * f_stride
            exp[4 * o : 4 * o + Np] = st.view(np.uint32)
        else:
            o = c_off + b * c_stride
            if qout:
                rows = np.zeros((N, c_pitch * 8), np.uint16)
                rows[:, :L] = q.numpy().astype(np.uint16)
            else:
                rows = np.zeros((N, c_pitch * 4), np.float32)
                rows[:, :L] = p.numpy()
            exp[4 * o : 4 * o + N * c_pitch * 4] = rows.view(np.uint32).reshape(-1)
            if qout:
                st = np.zeros(f_stride * 4, np.float32)
                st[:N] = s[:, 0].numpy()
                o = f_off + b * f_stride
                exp[4 * o : 4 * o + f_stride * 4] = st.view(np.uint32)
        # FP64 torch softmax with the same mask and empty-row rule
        temp = c / E.LOG2E
        xs = torch.where(valid, xb.double() * temp, torch.full_like(xb.double(), -1e300))
        ref = torch.softmax(xs, -1)
        if empty == "zero":
            ref = torch.where(valid.any(-1, keepdim=True), ref, torch.zeros_like(ref))
        errs.append((p_hat - ref).abs())
    err = torch.cat([e.reshape(-1) for e in errs])
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "mem.bin").write_bytes(mem.tobytes())
    (d / "expect.bin").write_bytes(exp.tobytes())
    return d, mem, err


def finish(d: Path, mem, st_fp1, st_fp0):
    (d / "params.txt").write_text(f"0 {mem.size // 4} {st_fp1} {st_fp0}\n")


def case(out, res, name, x, p0, st_fp1=ST_OK, st_fp0=None, **kw):
    d, mem, err = write_case(out, name, x, p0, emulate=st_fp1 == ST_OK, **kw)
    if st_fp0 is None:
        st_fp0 = st_fp1 if kw.get("qout", True) else ST_DIMS  # FP32 output needs SMX_FP32_OUT=1
    finish(d, mem, st_fp1, st_fp0)
    if st_fp1 == ST_OK:
        ref_max = 1.0
        print(f"{name:26s} B {x.shape[0]:2d} N {x.shape[1]:4d} L {x.shape[2]:4d}: max|p-p64| {float(err.max()):.3e}"
              f"  mean {float(err.mean()):.2e}")
        res.append(dict(case=name, max_abs=float(err.max()), mean_abs=float(err.mean())))
    else:
        print(f"{name:26s} expect status {st_fp1} / {st_fp0}")


def prefix_ranks(n_img=32, n_lang=48, n_real=17, n_state=1, pad_keys=0):
    """SmolVLA prefix: images + language (padded) bidirectional (block 0), state in block 1; pad tokens never
    visible and their rows fully masked. Keys padded with pad_keys never-visible keys."""
    L = n_img + n_lang + n_state
    blk = torch.zeros(L, dtype=torch.long)
    blk[n_img + n_lang :] = 1
    padm = torch.ones(L, dtype=torch.bool)
    padm[n_img + n_real : n_img + n_lang] = False
    Ek = torch.where(padm, blk, torch.full_like(blk, -1))
    Dn = torch.where(padm, blk, torch.full_like(blk, -1))
    if pad_keys:
        Ek = torch.cat([Ek, torch.full((pad_keys,), -1, dtype=torch.long)])
    return Ek[None], Dn[None]


def expert_self_ranks(n_pre=81, n_real=17, n_act=50):
    """Suffix queries over prefix + suffix keys: prefix (pads never) fully visible, actions causal."""
    Ekp, _ = prefix_ranks(n_real=n_real)
    Ek = torch.cat([torch.where(Ekp[0] < 0, -1, 0), torch.arange(1, n_act + 1)])
    Dn = torch.arange(1, n_act + 1)
    return Ek[None], Dn[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "build/cases"))
    ap.add_argument("--real", default=str(HERE / "build/real_scores.pt"))
    a = ap.parse_args()
    out = Path(a.out)
    g = torch.Generator().manual_seed(1)
    res = []
    T8 = 0.125
    # ---------------- rows layout (sequencer contract)
    x = torch.randn(2, 64, 256, generator=g) * 12.0
    case(out, res, "rows_vision_u16", x, T8)
    case(out, res, "rows_vision_f32", x, T8, qout=False)
    case(out, res, "rows_vision_pitch_u16", x, T8, a_pitch=72, c_pitch=40, item_gap=12)
    Ek, Dn = prefix_ranks(pad_keys=15)  # 81 rows, 96 keys (15 padding keys never visible)
    x = torch.randn(3, 81, 96, generator=g) * 20.0
    case(out, res, "rows_prefix_mask_u16", x, T8, ranks=(Ek, Dn))
    case(out, res, "rows_prefix_uniform_u16", x, T8, ranks=(Ek, Dn), empty="uniform")
    case(out, res, "rows_prefix_mask_f32", x, T8, ranks=(Ek, Dn), qout=False)
    Ek, Dn = expert_self_ranks()
    x = torch.randn(2, 50, 131, generator=g) * 15.0
    case(out, res, "rows_expert_self_u16", x, T8, ranks=(Ek, Dn))
    # per-item masks (non-zero E / D strides): valid length per row + causal, different per item
    B, N, L = 3, 40, 64
    Ek = torch.arange(L)[None].repeat(B, 1)
    Dn = torch.stack([torch.arange(N) + off for off in (0, -5, 30)])
    Dn[1, :3] = -1
    x = torch.randn(B, N, L, generator=g) * 4.0
    case(out, res, "rows_causal_items_u16", x, 1.0, ranks=(Ek, Dn))
    case(out, res, "rows_causal_items_unif", x, 1.0, ranks=(Ek, Dn), empty="uniform")
    # clamps: E >= 0x7FFF never visible, D > 0x7FFE clamped
    Ek = torch.tensor([[0, 5, 40000, 0x7FFE, -7, 3, 2, 1]])
    Dn = torch.tensor([[40000, 5, 4, 0, -1, 0x7FFF, 2, 3, 1 << 30]])
    case(out, res, "rows_rank_clamp_u16", torch.randn(1, 9, 8, generator=g), 1.0, ranks=(Ek, Dn))
    # edge values
    x = torch.randn(1, 64, 40, generator=g)
    x[0, 0] = 1e4
    x[0, 1, :] = torch.tensor([3.0e4, -3.0e4] * 20)
    x[0, 2, :] = 1e-30
    x[0, 3, :20] = 1e-42
    x[0, 4, :] = 0.0
    x[0, 4, ::2] = -0.0
    x[0, 5, :] = torch.linspace(-1e3, 1e3, 40)
    x[0, 6, :] = -1e30
    x[0, 7, :] = torch.tensor([5.0, 5.0, -5.0, 5.0] * 10)
    case(out, res, "rows_edge_u16", x, 1.0 / E.LOG2E)
    case(out, res, "rows_edge_f32", x, 1.0 / E.LOG2E, qout=False)
    case(out, res, "rows_edge_bigc_u16", x[:, :, :37], 26.0)
    case(out, res, "rows_edge_smallc_u16", x[:, :, :37] * 1e3, 2.0e-4)
    case(out, res, "rows_one_u16", torch.randn(1, 9, 1, generator=g), 1.0)
    case(out, res, "rows_five_f32", torch.randn(2, 9, 5, generator=g), 1.0, qout=False)
    x = torch.randn(1, 300, 256, generator=g) * 30  # two scale blocks of 256 rows
    case(out, res, "rows_lmax_300_u16", x, T8)
    case(out, res, "rows_lmax_f32", x[:, :6] * 0.01, T8, qout=False)  # FP32 sums > 2^24
    case(out, res, "rows_short_u16", torch.randn(2, 90, 9, generator=g), 1.0)  # divider-paced rows
    # ---------------- engine layout (int16_linear.h Y in, X + s_tok out)
    x = torch.randn(2, 256, 256, generator=g) * 12.0
    case(out, res, "eng_vision_u16", x, T8, engine=True)
    Ek, Dn = prefix_ranks()
    Dn3 = Dn.repeat(1, 3)  # 3 query heads stacked per KV head: row t = h * 81 + i
    x = torch.randn(2, 243, 81, generator=g) * 20.0
    case(out, res, "eng_prefix_mask_u16", x, T8, engine=True, ranks=(Ek, Dn3), item_gap=20)
    case(out, res, "eng_prefix_uniform_u16", x, T8, engine=True, ranks=(Ek, Dn3), empty="uniform")
    Ek, Dn = expert_self_ranks()
    x = torch.randn(2, 150, 131, generator=g) * 15.0
    case(out, res, "eng_expert_self_u16", x, T8, engine=True, ranks=(Ek, Dn.repeat(1, 3)))
    case(out, res, "eng_small_u16", torch.randn(3, 5, 7, generator=g) * 3, 1.0, engine=True)
    case(out, res, "eng_lmax_u16", torch.randn(1, 256, 256, generator=g) * 30, T8, engine=True)
    # ---------------- errors (nothing written)
    x = torch.randn(1, 8, 16, generator=g)
    case(out, res, "err_opcode", x, 1.0, opcode=0x41, st_fp1=ST_OPCODE)
    case(out, res, "err_align", x, 1.0, misalign=True, st_fp1=ST_ALIGN)
    case(out, res, "err_L0", x, 1.0, L_rec=0, st_fp1=ST_DIMS)
    case(out, res, "err_Lbig", x, 1.0, L_rec=L_MAX + 1, st_fp1=ST_DIMS)
    case(out, res, "err_pitch", x, 1.0, L_rec=24, st_fp1=ST_DIMS)  # a_pitch 4 words < ceil(24 / 4)
    case(out, res, "err_cneg", x, -1.0, st_fp1=ST_DIMS)
    case(out, res, "err_eng_rows", torch.randn(1, 257, 8, generator=g), 1.0, engine=True, st_fp1=ST_DIMS)
    case(out, res, "err_eng_f32", x, 1.0, engine=True, qout=False, st_fp1=ST_DIMS, st_fp0=ST_DIMS)
    Ek, Dn = torch.zeros(1, 8, dtype=torch.long), torch.zeros(1, 300, dtype=torch.long)
    case(out, res, "err_mask_rows", torch.randn(1, 300, 8, generator=g), 1.0, ranks=(Ek, Dn), st_fp1=ST_DIMS)
    # ---------------- real score rows of -12392 (policy_check.py --dump-scores), both layouts
    rp = Path(a.real)
    if rp.exists():
        real = torch.load(rp, weights_only=False)
        for k, v in real.items():
            xs, msk, rph = v["scores"], v.get("valid"), v["rph"]
            n, L = xs.shape
            x = xs.reshape(n // rph, rph, L)  # heads as items
            rk = None
            if msk is not None:
                r = E.ranks_from_mask(msk[:rph])
                assert r is not None, f"{k}: mask not rank-representable"
                rk = (r[0][None], r[1][None])
            case(out, res, f"real_{k}_u16", x, T8, ranks=rk)
            case(out, res, f"real_{k}_f32", x, T8, ranks=rk, qout=False)
            xe = x.reshape(1, n, L) if n <= R_MAX else x  # stacked heads as one engine item when they fit
            rke = None if rk is None else (rk[0], rk[1].repeat(1, xe.shape[1] // rph))
            case(out, res, f"real_{k}_eng", xe, T8, engine=True, ranks=rke)
    else:
        print(f"(no real scores at {rp}; run policy_check.py --dump-scores first for real-score cases)")
    worst = max(res, key=lambda r: r["max_abs"])
    print(f"worst max|p-p64| {worst['max_abs']:.3e} ({worst['case']})")


if __name__ == "__main__":
    main()
