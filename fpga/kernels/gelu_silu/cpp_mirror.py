"""Line-by-line Python transliteration of gelu_silu.h (act1, row_scale, quant1, the pass-1 / pass-2 word mapping),
every ap_uint<W> / ap_int<W> assignment written as an explicit wrap to W bits, checked bit for bit against gs_emu.

This is a review aid for when the C++ testbench cannot be executed (on this laptop Smart App Control blocks newly
built .exe files): it catches width / sign / shift mistakes in reading the C++, not compiler behaviour. The real
check stays tb_gelu_silu.cpp (run_tests.sh), which links the HLS source itself.

    python cpp_mirror.py [CASE_DIR ...]       # default: random + edge cases built here; CASE_DIR as make_vectors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_emu  # noqa: E402
import gs_tables  # noqa: E402

FB, VB, DB = 12, 21, 15
QMAX = 32767


def u(v, w):  # ap_uint<w>(v)
    return v & ((1 << w) - 1)


def s(v, w):  # ap_int<w>(v)
    v = v & ((1 << w) - 1)
    return np.where(v >= (1 << (w - 1)), v - (1 << w), v)


def clz(v, w):  # ap_uint<w>::countLeadingZeros
    out = np.full(v.shape, w, dtype=np.int64)
    for b in range(w - 1, -1, -1):
        hit = ((v >> b) & 1).astype(bool) & (out == w)
        out = np.where(hit, w - 1 - b, out)
    return out


def rsh5(v, sh, w):
    r = u(v, w)
    for i in range(5):
        r = np.where((sh >> i) & 1 == 1, u(r >> (1 << i), w), r)
    return r


def norm21(v):
    c = np.zeros_like(v)
    for k, top in ((16, 5), (8, 13), (4, 17), (2, 19), (1, 20)):
        z = (v >> top) == 0  # bits 20 .. top all zero
        v = np.where(z, u(v << k, 21), v)
        c = np.where(z, c | k, c)
    return v, c


def act1(xb, ub, tsilu: bool, gmul: bool, nsb: int):
    TB = nsb + FB
    T_rom = gs_tables.cached_table(nsb).astype(np.int64)
    sx = (xb >> 31) & 1
    ex = u(xb >> 23, 8)
    xz = ex == 0
    mx = np.where(xz, 0, u((1 << 23) | u(xb, 23), 24))
    sh = s((150 + (4 if tsilu else 3) - TB) - ex, 10)
    sat = (~xz) & (sh < 24 - TB)
    inr = (~xz) & (sh >= 24 - TB) & (sh < 24)
    shu = u(sh, 5)
    T = np.where(inr, u(rsh5(mx, shu, 24), TB), 0)
    idx = u(T >> FB, nsb)
    f = u(T, FB)
    addr = u(((1 if tsilu else 0) << nsb) | idx, nsb + 1)
    w = T_rom[addr]
    V = u(w >> DB, VB)
    D = u(w, DB)
    df = u(D * f, DB + FB)
    corr = u((u(df >> (FB - 1), DB + 2) + 1) >> 1, DB)
    gd = s(V - corr, VB + 1)
    g21 = np.where(sat | (gd < 0), 0, u(gd, VB))
    g18r = u((u(g21 >> 2, VB) + 1) >> 1, 18)
    g18 = np.where((g18r >> 17) & 1 == 1, 0x1FFFF, u(g18r, 17))
    g21n, pzc = norm21(g21)
    gnr = u((u(g21n >> 3, 19) + 1) >> 1, 18)
    gn = u(gnr, 17)
    pz = s(pzc, 7)
    c = ((gnr >> 17) & 1) == 1
    gn = np.where(c, 1 << 16, gn)
    pz = np.where(c, s(pz - 1, 7), pz)
    G = np.where(sx == 1, gn, g18)
    P = u(mx * G, 41)
    Y = np.where(sx == 1, u(P, 42), u(u(mx << 18, 42) - P, 42))
    b41, b40, b39 = (Y >> 41) & 1, (Y >> 40) & 1, (Y >> 39) & 1
    zY = (b41 | b40 | b39) == 0
    lz = np.where(b41 == 1, 0, np.where(b40 == 1, 1, 2))
    Yt = np.where(b41 == 1, u(Y >> 17, 25), np.where(b40 == 1, u(Y >> 16, 25), u(Y >> 15, 25)))
    e0 = s(s(ex - lz, 11) + np.where(sx == 1, s(1 - pz, 11), 0), 11)
    if not gmul:
        m25 = u((u(Yt, 26) + 1) >> 1, 25)
        ey = e0.copy()
        m = u(m25, 24)
        cy = ((m25 >> 24) & 1) == 1
        m = np.where(cy, 1 << 23, m)
        ey = np.where(cy, s(ey + 1, 11), ey)
        z = zY | (ey <= 0)
        r = (sx << 31) | (u(ey, 8) << 23) | u(m, 23)
        return np.where(z, 0, r)
    m18 = u((u(Yt >> 7, 19) + 1) >> 1, 18)
    m17 = u(m18, 17)
    e17 = e0.copy()
    c = ((m18 >> 17) & 1) == 1
    m17 = np.where(c, 1 << 16, m17)
    e17 = np.where(c, s(e17 + 1, 11), e17)
    z1 = zY | (e17 <= 0)
    su = (ub >> 31) & 1
    eu = u(ub >> 23, 8)
    mu = u((1 << 23) | u(ub, 23), 24)
    P3 = u(mu * m17, 41)
    top = ((P3 >> 40) & 1) == 1
    Ps = np.where(top, u(P3 >> 16, 25), u(P3 >> 15, 25))
    m3r = u((u(Ps, 26) + 1) >> 1, 25)
    e3 = s(e17 + eu - 127 + top.astype(np.int64), 11)
    m3 = u(m3r, 24)
    c = ((m3r >> 24) & 1) == 1
    m3 = np.where(c, 1 << 23, m3)
    e3 = np.where(c, s(e3 + 1, 11), e3)
    big = e3 >= 255
    e3 = np.where(big, 254, e3)
    m3 = np.where(big, (1 << 24) - 1, m3)
    z3 = z1 | (eu == 0) | (e3 <= 0)
    r = ((sx ^ su) << 31) | (u(e3, 8) << 23) | u(m3, 23)
    return np.where(z3, 0, r)


def row_scale(amax: int):
    ea = u(amax >> 23, 8)
    ma = u((1 << 23) | u(amax, 23), 24)
    zero = ea < 16
    num = u(QMAX << 25, 40)
    rem = u(num >> 17, 25)
    rq = 0
    for i in range(16, -1, -1):
        rem = u((rem << 1) | ((num >> i) & 1), 25)
        ge = rem >= ma
        if ge:
            rem = u(rem - ma, 25)
        rq |= int(ge) << i
    rem2, q25 = 0, 0
    for i in range(41, -1, -1):
        rem2 = u((rem2 << 1) | (1 if i == 41 else 0), 18)
        ge = rem2 >= rq
        if ge:
            rem2 = u(rem2 - rq, 18)
        if i < 26:
            q25 |= int(ge) << i
    big = (q25 >> 25) & 1
    sm = u((q25 + 2) >> 2, 25) if big else u((q25 + 1) >> 1, 25)
    es = u(ea - 16 + (2 if big else 1), 9)
    if (sm >> 24) & 1:
        sm = 1 << 23
        es = u(es + 1, 9)
    s_bits = 0 if zero else ((u(es, 8) << 23) | u(sm, 23))
    return rq, ea, zero, s_bits


def quant1(yb, rq, ea, zero):
    sy = (yb >> 31) & 1
    ey = u(yb >> 23, 8)
    my = u((1 << 23) | u(yb, 23), 24)
    P2 = u(my * rq, 41)
    d = u(ea - ey, 9)
    t = u(P2 >> 24, 17)
    ts = np.where(d > 16, 0, u(t >> np.minimum(d, 31), 17))
    qm = u((ts + 1) >> 1, 16)
    qm = np.where(qm > QMAX, QMAX, qm)
    qm = np.where((ey == 0) | zero, 0, qm)
    return np.where(sy == 1, -qm, qm)


def kernel_words(x: np.ndarray, uu: np.ndarray | None, mode: int, nsb: int, lanes: int = 8):
    """Mirror of gs_pass1 / gs_scale / gs_pass2 on (N, K) FP32: returns q in output-word order (N, K) and s bits."""
    N, K = x.shape
    tsilu, gmul = mode != 0, mode == 1
    xb = x.view(np.uint32).astype(np.int64)
    ubits = uu.view(np.uint32).astype(np.int64) if uu is not None else np.zeros_like(xb)
    wh = K // 8
    qout = np.zeros((N, K), dtype=np.int64)
    sb = np.zeros(N, dtype=np.int64)
    for n in range(N):
        # word w: lanes 0-3 = first-half word w, lanes 4-7 = second-half word w
        if lanes == 8:
            lanes_x = np.concatenate([xb[n, : K // 2].reshape(wh, 4), xb[n, K // 2 :].reshape(wh, 4)], axis=1)
            lanes_u = np.concatenate([ubits[n, : K // 2].reshape(wh, 4), ubits[n, K // 2 :].reshape(wh, 4)], axis=1)
        else:  # one 4-lane entry per input word; pass 2 packs entries 2w, 2w + 1 into output word w
            lanes_x, lanes_u = xb[n].reshape(K // 4, 4), ubits[n].reshape(K // 4, 4)
        y = act1(lanes_x, lanes_u, tsilu, gmul, nsb)
        amax = int(u(y, 31).max())
        rq, ea, zero, s_bits = row_scale(amax)
        q = quant1(y, rq, ea, zero)
        qout[n] = q.reshape(-1)
        sb[n] = s_bits
    # gs_pass2's scale words: s0..s2 registers, word written at j == 3 or the last row
    words = []
    s0 = s1 = s2 = 0
    for n in range(N):
        j, b = n & 3, int(sb[n])
        if j == 3 or n == N - 1:
            words.append([b if j == 0 else s0, b if j == 1 else (s1 if j > 1 else 0),
                          b if j == 2 else (s2 if j > 2 else 0), b if j == 3 else 0])
        s0, s1, s2 = (b if j == 0 else s0), (b if j == 1 else s1), (b if j == 2 else s2)
    flat = np.array(words, dtype=np.int64).reshape(-1)
    assert (flat[:N] == sb).all() and (flat[N:] == 0).all(), "scale word packing"
    return qout, sb


def check(x: torch.Tensor, uu: torch.Tensor | None, mode: int, nsb: int, name: str, lanes: int = 8) -> bool:
    q_ref, s_ref, _ = gs_emu.kernel(x, uu, mode, nsb)
    perm = gs_emu.out_perm(x.shape[1], lanes)
    qm, sm = kernel_words(x.numpy().astype(np.float32), None if uu is None else uu.numpy().astype(np.float32), mode,
                          nsb, lanes)
    qn = np.empty_like(qm)
    qn[:, perm] = qm  # undo the output permutation: position p holds element perm[p]
    bq = int((qn != q_ref.numpy().astype(np.int64)).sum())
    bs = int((sm != gs_emu.f2b(s_ref.reshape(-1)).numpy()).sum())
    print(f"{name:24s} lanes {lanes} mode {mode} nsb {nsb} N {x.shape[0]:4d} K {x.shape[1]:4d}: q mismatches {bq}, s mismatches {bs}")
    return bq == 0 and bs == 0


def main() -> None:
    ok = True
    if len(sys.argv) > 1:
        for d in map(Path, sys.argv[1:]):
            mode, N, K, nsb = map(int, (d / "meta.txt").read_text().split())
            x = torch.from_numpy(np.fromfile(d / "x.bin", dtype=np.float32).reshape(N, K))
            uu = torch.from_numpy(np.fromfile(d / "u.bin", dtype=np.float32).reshape(N, K)) if mode == 1 else None
            for lanes in (8, 4):
                ok &= check(x, uu, mode, nsb, d.name, lanes)
    else:
        sys.path.insert(0, str(HERE))
        from make_vectors import edge_rows

        g = torch.Generator().manual_seed(7)
        for nsb in (9, 10):
            for mode, K, R in ((0, 3072, 8.0), (1, 2560, 16.0), (2, 720, 16.0)):
                x = torch.randn(6, K, generator=g) * 3.0
                x[0, :64] = -torch.linspace(0.0, 20.0, 64)
                uu = torch.randn(6, K, generator=g) * 4.0 if mode == 1 else None
                for lanes in (8, 4):
                    ok &= check(x, uu, mode, nsb, f"rand_{mode}_{K}", lanes)
                e = edge_rows(K, R, g)
                ue = torch.randn(e.shape, generator=g) if mode == 1 else None
                if ue is not None:
                    ue[3, :] = 1e30
                for lanes in (8, 4):
                    ok &= check(e, ue, mode, nsb, f"edge_{mode}_{K}", lanes)
    print("MIRROR PASS" if ok else "MIRROR FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
