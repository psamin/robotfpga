"""Test cases for tb_softmax.cpp (integration contract 1.0): whole DDR images holding one 256-byte SOFTMAX
record
(contract.encode), the scores, the rank vectors and output regions full of 0xA5 canaries, and the image the
unit
must leave behind, from the golden executor smx_emu.run_cmd. Also prints each case's error against FP64
softmax,
and checks the emulation's closed forms against the pre-contract formulas.

    python make_vectors.py [--out DIR] [--real scores.pt] [--quick]

Each case directory holds params.txt ("<base hex> <record byte offset> <bytes> <expected return hex>
<direct>"),
mem.bin and expect.bin. The expected return of softmax_top: 0 when the record runs; (ME << 16) | 0x13 when
validate_hw (contract.validate(full=False)) rejects it; an ST_BAD_* code when it does not decode.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract as CT  # noqa: E402
import smx_emu as E  # noqa: E402

C = CT.C
CANARY = 0xA5
LOG2E = E.LOG2E


class Image:
    """Byte image at physical address `base`; the record sits at offset 0."""

    def __init__(self, base: int = 0):
        self.base = base
        self.n = C.REC_BYTES
        self.puts: list[tuple[int, bytes]] = []

    def alloc(self, nbytes: int, align: int = 64) -> int:
        off = -(-self.n // align) * align
        self.n = off + max(int(nbytes), 0)
        return self.base + off

    def put(self, addr: int, data) -> None:
        self.puts.append((addr - self.base, bytes(np.ascontiguousarray(data).view(np.uint8).reshape(-1))))

    def build(self) -> np.ndarray:
        mem = np.full(-(-self.n // 64) * 64 + 64, CANARY, dtype=np.uint8)
        for off, b in self.puts:
            mem[off : off + len(b)] = np.frombuffer(b, np.uint8)
        return mem


def expect_code(c: CT.Cmd) -> int:
    me = CT.validate(c, full=False)
    return (me << 16) | C.ST_MODE_ERROR if me else C.ST_END


def write_case(
    out: Path,
    name: str,
    img: Image,
    c: CT.Cmd,
    *,
    direct=False,
    raw_rec: bytes | None = None,
    want: int | None = None,
):
    rec = raw_rec if raw_rec is not None else CT.encode(c)
    img.put(img.base, np.frombuffer(rec, np.uint8))
    mem = img.build()
    exp = mem.copy()
    if want is None:
        want = expect_code(c)
    if want == C.ST_END:
        assert CT.validate(c, full=True) == C.ME_OK, name
        E.run_cmd(c, exp, img.base)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "mem.bin").write_bytes(mem.tobytes())
    (d / "expect.bin").write_bytes(exp.tobytes())
    (d / "params.txt").write_text(f"{img.base:x} 0 {mem.size} {want:x} {int(direct)}\n")
    return mem, exp


def smx_case(
    out,
    res,
    name,
    x,
    *,
    p0=1.0,
    ranks=None,
    np_=0,
    a_pad=0,
    c_pad=0,
    gap=0,
    base=0,
    direct=False,
    share_ranks=True,
    mutate=None,
    raw=None,
    want=None,
    flags=None,
    f_pad=0,
):
    """x (B, N, L) float32 scores. ranks None or (E (Be, L), D (Bd, N)) int with Be, Bd in {1, B}
    (1 = one vector shared by every item: stride 0). a_pad / c_pad: extra bytes of row pitch; gap: bytes
    between items. mutate(cmd) edits the record before encoding (error cases)."""
    B, N, L = x.shape
    NP = np_ or N
    img = Image(base)
    g = torch.Generator().manual_seed(len(name))
    a_pitch = 4 * L + a_pad
    a_stride = N * a_pitch + gap
    A = img.alloc(B * a_stride)
    for b in range(B):
        y = (
            torch.randn(N, a_pitch // 4, generator=g) * 1e3
        )  # pitch padding: finite garbage, never read as data
        y[:, :L] = x[b]
        img.put(A + b * a_stride, y.numpy().astype(np.float32))
    c_pitch = 2 * L + c_pad
    c_stride = NP * c_pitch + gap
    Cb = img.alloc(B * c_stride)
    f_stride = 4 * CT.pad4(NP) + f_pad
    Fb = img.alloc(B * f_stride)
    s = {
        "A": CT.Slot(A, a_pitch, a_stride),
        "C": CT.Slot(Cb, c_pitch, c_stride),
        "F": CT.Slot(Fb, 0, f_stride),
    }
    fl = C.F_QOUT if flags is None else flags
    if ranks is not None:
        fl |= C.F_MASK
        Ek, Dn = (torch.as_tensor(r).to(torch.int64) for r in ranks)
        e_st = 0 if Ek.shape[0] == 1 else 4 * L + 16
        d_st = 0 if Dn.shape[0] == 1 else 4 * CT.pad4(N) + 32
        Ea = img.alloc(max(1, Ek.shape[0]) * (e_st or 4 * L))
        Da = img.alloc(max(1, Dn.shape[0]) * (d_st or 4 * CT.pad4(N)))
        for i in range(Ek.shape[0]):
            img.put(Ea + i * e_st, Ek[i].numpy().astype(np.int64).astype(np.int32))
        for i in range(Dn.shape[0]):
            v = np.full(CT.pad4(N), -1, np.int32)  # pad lanes: never read
            v[:N] = Dn[i].numpy().astype(np.int64).astype(np.int32)
            img.put(Da + i * d_st, v)
        s["D"], s["E"] = CT.Slot(Da, 0, d_st), CT.Slot(Ea, 0, e_st)
    p = {C.P_EPS: CT.f32bits(p0)}
    if np_:
        p[C.P_NP] = np_
    c = CT.Cmd(C.OP_SOFTMAX, fl, 0x5F7A0001, 0, L, N, B, s, p)
    if mutate:
        mutate(c)
    mem, exp = write_case(out, name, img, c, direct=direct, raw_rec=raw, want=want)
    code = int((out / name / "params.txt").read_text().split()[3], 16)
    if code == C.ST_END:
        err = fp64_error(c, exp, img.base, x, ranks)
        res.append(dict(case=name, max_abs=err))
        print(f"{name:26s} B {B:2d} N {N:4d} NP {NP:4d} L {L:3d}: max|P-P64| {err:.3e}")
    else:
        print(f"{name:26s} expect return 0x{code:05x}")


def fp64_error(c, exp, base, x, ranks) -> float:
    """max |q * scale - softmax64| over the items (fully masked rows: zero on both sides)."""
    B, N, L = x.shape
    Cs, F = c.slot("C"), c.slot("F")
    cv = E.c_from_mult(CT.bits_f32(c.par(C.P_EPS)))
    worst = 0.0
    for b in range(B):
        q = (
            E._rows(exp, base, Cs.addr + b * Cs.stride, Cs.pitch, N, 2 * L)
            .copy()
            .view(np.int16)
            .reshape(N, L)
        )
        o = F.addr + b * F.stride - base
        sc = exp[o : o + 4 * N].copy().view(np.float32)
        p = q.astype(np.float64) * sc[:, None].astype(np.float64)
        if ranks is not None:
            Ek, Dn = ranks
            v = E.rank_valid(
                torch.as_tensor(Ek[b if Ek.shape[0] > 1 else 0]),
                torch.as_tensor(Dn[b if Dn.shape[0] > 1 else 0]),
            ).numpy()
        else:
            v = np.ones((N, L), bool)
        xs = np.where(v, x[b].double().numpy() * (cv / LOG2E), -np.inf)
        with np.errstate(invalid="ignore", over="ignore"):
            mx = xs.max(-1, keepdims=True)
            ex = np.where(v, np.exp(xs - np.where(np.isfinite(mx), mx, 0)), 0.0)
            ref = ex / np.maximum(ex.sum(-1, keepdims=True), 1e-300)
        # rows with inf / NaN have no FP64 reference; rows with |x c| >= 2^25 are in fix()'s documented
        # saturation range (|x 2^ec| >= 2^26 compares as equal): both are reported apart, not in this number
        xa = np.abs(x[b].numpy().astype(np.float64))
        ok = np.isfinite(xa).all(-1) & (xa.max(-1, initial=0.0) * cv < 2.0**25)
        worst = max(worst, float(np.abs(p - ref)[ok].max(initial=0.0)))
    return worst


# ---------------------------------------------------------------- SmolVLA rank vectors (CONTRACT.md 5)
PRE, TQP_LM, TQ_E, TQP_E, LP, LS = 45, 48, 50, 52, 64, 128


def prefix_ranks():
    """LM prefix: 45 tokens (2 x 16 image, 12 language: rank 0; state: rank 1), rows 3 heads x 48 (pad rows
    -1),
    keys 64 (pad keys -1)."""
    tok = torch.full((TQP_LM,), -1)
    tok[: PRE - 1] = 0
    tok[PRE - 1] = 1
    Ek = torch.full((LP,), -1)
    Ek[:PRE] = tok[:PRE]
    return Ek[None], tok.repeat(3)[None]


def expert_self_ranks():
    """Expert self-attention: rows 3 heads x 52 (i < 50: rank 2 + i); keys: prefix [0, 64) (0 / state 1 / pad
    -1),
    suffix [64, 128): key j < 50 rank 2 + j, pad -1."""
    Ek = torch.full((LS,), -1)
    Ek[: PRE - 1] = 0
    Ek[PRE - 1] = 1
    Ek[LP : LP + TQ_E] = 2 + torch.arange(TQ_E)
    row = torch.full((TQP_E,), -1)
    row[:TQ_E] = 2 + torch.arange(TQ_E)
    return Ek[None], row.repeat(3)[None]


def expert_cross_ranks():
    Ek = torch.full((LP,), -1)
    Ek[:PRE] = 0
    row = torch.full((TQP_E,), -1)
    row[:TQ_E] = 0
    return Ek[None], row.repeat(3)[None]


def self_checks() -> None:
    """The emulation's closed forms against the pre-contract kernel's formulas and the IEEE reference."""
    g = torch.Generator().manual_seed(7)
    e23 = torch.randint(0, 1 << 23, (1 << 20,), generator=g)
    ip = torch.randint(0, 32, (1 << 20,), generator=g)
    old = torch.where(ip >= 24, torch.zeros_like(e23), (((e23 << 1) >> ip.clamp(max=24)) + 1) >> 1)
    old = (old + 128) >> 8
    new = torch.where(ip >= 17, torch.zeros_like(e23), (((e23 << 1) >> ip.clamp(max=17)) + 257) >> 9)
    assert torch.equal(old, new), "q closed form"
    from fractions import Fraction

    s = torch.cat([torch.randint(1, 1 << 23, (4000,), generator=g), torch.tensor([1, 2, 3, 32767, 8388352])])
    r = E.rn_recip(s)
    for sv, rv in zip(
        s.tolist(), r.tolist(), strict=True
    ):  # exactly the nearest FP32 (ties to even) of 1 / s
        f = np.float32(rv)
        lo, hi = np.nextafter(f, np.float32(0)), np.nextafter(f, np.float32(1))
        ex = Fraction(1, sv)
        dist = abs(Fraction(float(f)) - ex)
        assert dist <= abs(Fraction(float(lo)) - ex) and dist <= abs(Fraction(float(hi)) - ex), (
            "rn_recip",
            sv,
        )
    assert float(E.rn_recip(torch.tensor([0]))[0]) == 1.0, "zero row scale"
    print(f"self checks: q closed form on {e23.numel()} values, rn_recip exact on {s.numel()} sums: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "build/cases"))
    ap.add_argument("--real", default=str(HERE / "build/real_scores.pt"))
    a = ap.parse_args()
    out = Path(a.out)
    self_checks()
    g = torch.Generator().manual_seed(1)
    res = []
    rn = lambda *s, k=1.0: torch.randn(*s, generator=g) * k  # noqa: E731
    # ---------------- the four SOFTMAX shapes of the -12392 chunk (contract_plan.py), p0 = 1 (Q carries 1/8)
    smx_case(out, res, "plan_vision", rn(24, 256, 256, k=1.5))
    smx_case(out, res, "plan_lm_prefix", rn(5, 144, 64, k=2.5), ranks=prefix_ranks())
    smx_case(out, res, "plan_expert_self", rn(5, 156, 128, k=2.0), ranks=expert_self_ranks())
    smx_case(out, res, "plan_expert_cross", rn(5, 156, 64, k=2.0), ranks=expert_cross_ranks())
    # the same scores with p0 = 1/8 on 8x larger scores: the same bits (1/8 is exact)
    x = rn(2, 64, 256, k=12.0)
    smx_case(out, res, "p0_eighth", x, p0=0.125)
    smx_case(out, res, "p0_one_prescaled", x * 0.125)
    # ---------------- addressing: pitches, gaps, per-item ranks, blocks, NP, high addresses
    Ek = torch.arange(64)[None].repeat(3, 1)
    Dn = torch.stack([torch.arange(40) + off for off in (0, -5, 30)])
    Dn[1, :3] = -1
    xi = rn(3, 40, 64, k=4.0)
    smx_case(out, res, "items_ranks_strided", xi, ranks=(Ek, Dn), a_pad=48, c_pad=32, gap=80, f_pad=16)
    smx_case(out, res, "pitch_a_only", rn(2, 30, 96, k=3.0), a_pad=16)
    smx_case(out, res, "pitch_c_only", rn(2, 30, 96, k=3.0), c_pad=16)
    smx_case(out, res, "blocks_300", rn(2, 300, 64, k=5.0))
    Ek = torch.arange(32)[None]
    Dn = (torch.arange(300) % 40 - 4)[None]
    smx_case(out, res, "blocks_300_mask_np", rn(2, 300, 32, k=5.0), ranks=(Ek, Dn), np_=307)
    smx_case(out, res, "blocks_513_pitch", rn(1, 513, 16, k=5.0), a_pad=32, c_pad=16, np_=515)
    smx_case(out, res, "np_pad", rn(3, 5, 8, k=2.0), np_=11)
    smx_case(out, res, "n1", rn(1, 1, 8))
    smx_case(out, res, "short_rows_l8", rn(2, 90, 8, k=3.0))  # divider-paced (2 words per row)
    smx_case(out, res, "short_rows_l24", rn(2, 70, 24, k=3.0))
    smx_case(out, res, "high_32g", rn(5, 144, 64, k=2.5), ranks=prefix_ranks(), base=0x8_0000_0000)
    smx_case(
        out,
        res,
        "high_1t",
        rn(5, 156, 64, k=2.0),
        ranks=expert_cross_ranks(),
        base=0xFF_0000_0000,
        direct=True,
    )
    # ---------------- ranks: int32 semantics (values beyond 16 bits, negatives, fully masked rows)
    Ek = torch.tensor(
        [[0, 5, 40000, 0x7FFE, -7, 3, 2, 1, 0x7FFFFFFF, 70000, -(2**31), 65536, 65535, 9, 1 << 20, 0]]
    )
    Dn = torch.tensor(
        [[40000, 5, 4, 0, -1, 0x7FFF, 2, 3, 1 << 30, 0x7FFFFFFF, -(2**31), 65535, 69999, 70000]]
    )
    smx_case(out, res, "ranks_int32", rn(1, 14, 16), ranks=(Ek, Dn))
    Dn = torch.full((1, 9), -1)
    smx_case(out, res, "all_rows_masked", rn(2, 9, 16), ranks=(torch.zeros(1, 16), Dn), np_=10)
    # ---------------- edge values and multipliers
    x = rn(1, 64, 40)
    x[0, 0] = 1e4
    x[0, 1, :] = torch.tensor([3.0e4, -3.0e4] * 20)
    x[0, 2, :] = 1e-30
    x[0, 3, :20] = 1e-42
    x[0, 4, :] = 0.0
    x[0, 4, ::2] = -0.0
    x[0, 5, :] = torch.linspace(-1e3, 1e3, 40)
    x[0, 6, :] = -1e30
    x[0, 7, :] = torch.tensor([5.0, 5.0, -5.0, 5.0] * 10)
    x[0, 8, :] = 3.4e38
    x[0, 9, :] = -3.4e38
    x[0, 10, 3] = float("inf")
    x[0, 11, 5] = float("-inf")
    x[0, 12, 7] = float("nan")
    x[0, 13, :] = float("-inf")
    x[0, 14, :] = torch.tensor([6e8, 5.9e8] * 20)
    x[0, 15, 0] = -float("nan")
    smx_case(out, res, "edge_c1", x, p0=1.0 / LOG2E)
    smx_case(out, res, "edge_c26", x[:, :, :32], p0=26.0)
    smx_case(out, res, "edge_c2e-4", x[:, :, :32] * 1e3, p0=2.0e-4)
    smx_case(out, res, "edge_p0_denormal", x[:, :, :32], p0=1e-39)
    smx_case(out, res, "edge_p0_tiny_normal", x[:, :, :32] * 1e30, p0=1.2e-38)
    smx_case(out, res, "edge_p0_1e29", rn(1, 16, 32) * 1e-30, p0=9.9e29)
    smx_case(out, res, "lmax_x30", rn(1, 64, 256, k=30.0), p0=0.125)
    # ---------------- records validate_hw rejects (nothing may be written)
    xe = rn(1, 8, 16)

    def mut(**kw):
        def f(c):
            for k, v in kw.items():
                setattr(c, k, v)

        return f

    def mut_p(i, v):
        def f(c):
            c.p[i] = v

        return f

    def mut_slot(n, s):
        def f(c):
            c.s[n] = s

        return f

    smx_case(out, res, "err_k0", xe, mutate=mut(K=0))
    smx_case(out, res, "err_k264", xe, mutate=mut(K=264))
    smx_case(out, res, "err_k12", xe, mutate=mut(K=12))
    smx_case(out, res, "err_n0", xe, mutate=mut(N=0))
    smx_case(out, res, "err_no_qout", xe, flags=0)
    smx_case(out, res, "err_f_engine", xe, flags=C.F_QOUT | 0x1000)
    smx_case(out, res, "err_f_uniform", xe, flags=C.F_QOUT | 0x2000)
    smx_case(out, res, "err_p0_zero", xe, mutate=mut_p(C.P_EPS, 0))
    smx_case(out, res, "err_p0_neg", xe, mutate=mut_p(C.P_EPS, CT.f32bits(-1.0)))
    smx_case(out, res, "err_p0_nan", xe, mutate=mut_p(C.P_EPS, 0x7FC00000))
    smx_case(out, res, "err_np_lt_n", xe, mutate=mut_p(C.P_NP, 4))
    smx_case(out, res, "err_param", xe, mutate=mut_p(3, 1))
    smx_case(out, res, "err_slot_b", xe, mutate=mut_slot("B", CT.Slot(0x1000, 0, 0)))
    smx_case(out, res, "err_misaligned", xe, mutate=lambda c: c.s.__setitem__("A", c.s["A"] + 8))
    smx_case(out, res, "err_pitch", xe, mutate=lambda c: c.s.__setitem__("C", CT.Slot(c.s["C"].addr, 16, 0)))
    smx_case(out, res, "err_addr", xe, mutate=lambda c: c.s.__setitem__("A", CT.Slot(1 << 40, 64, 0)))
    good = CT.encode(CT.Cmd(C.OP_SOFTMAX, C.F_QOUT, 0, 0, 16, 8, 1, {}, {C.P_EPS: CT.f32bits(1.0)}))
    bad = bytearray(good)
    bad[1] ^= 0xFF  # REC_MAGIC, f0[15:8]
    smx_case(out, res, "err_bad_magic", xe, raw=bytes(bad), want=C.ST_BAD_MAGIC)
    bad = bytearray(good)
    bad[100] ^= 1
    smx_case(out, res, "err_bad_check", xe, raw=bytes(bad), want=C.ST_BAD_CHECK)
    f = list(struct.unpack("<64I", good))
    f[C.FLD_VERSION] = 0x0000FFFF
    x_ = C.CHECK_SEED
    for v in f[:63]:
        x_ ^= v
    f[63] = x_
    smx_case(out, res, "err_bad_version", xe, raw=struct.pack("<64I", *f), want=C.ST_BAD_VERSION)
    norm = CT.encode(CT.Cmd(C.OP_NORM, 0, 0, 0, 16, 8, 1, {}, {}))
    smx_case(out, res, "err_not_softmax", xe, raw=norm, want=C.ST_BAD_OPCODE)
    # ---------------- real -12392 score rows (policy_check.py --dump-scores): 2 heads of every site, frame 0.
    # The contract feeds scores already x 1/8 (Q's stored scale) with p0 = 1; keys padded to a multiple of 8
    # with never-visible keys (rank -1).
    rp = Path(a.real)
    if rp.exists():
        real = torch.load(rp, weights_only=False)
        for k, v in real.items():
            xs, msk, rph = v["scores"].float() * 0.125, v.get("valid"), v["rph"]
            n, L = xs.shape
            L8 = CT.pad8(L)
            xp = torch.cat([xs, torch.randn(n, L8 - L, generator=g) * 50], 1)
            if msk is not None or L8 != L:
                vv = torch.ones(n, L, dtype=torch.bool) if msk is None else msk
                vv = torch.cat([vv, torch.zeros(n, L8 - L, dtype=torch.bool)], 1)
                r = E.ranks_from_mask(vv.reshape(n // rph, rph, L8))
                assert r is not None, f"{k}: mask not rank-representable"
                rk = r
            else:
                rk = None
            smx_case(out, res, f"real_{k}", xp.reshape(n // rph, rph, L8), ranks=rk)
    else:
        print(f"(no real scores at {rp}; run policy_check.py --dump-scores first for real-score cases)")
    worst = max(res, key=lambda r: r["max_abs"])
    print(f"worst max|P-P64| {worst['max_abs']:.3e} ({worst['case']})")


if __name__ == "__main__":
    main()
