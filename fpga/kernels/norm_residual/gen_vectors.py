"""Test vectors for tb_nr.cpp: whole DDR images before / after one NORM command (contract 1.0 record), the
expected image computed by nr_emul.py, plus records that validate_hw must reject with the image untouched.

Case file (little-endian): u32 magic 0x3243524E ('NRC2'), u64 base (DDR byte address of image byte 0),
u64 list (byte address of the record), i32 want (nr_top's expected return: 0, or (status << 8) | ME code),
u32 W (image size in 128-bit words), W x 16 bytes input image, W x 16 bytes expected image.
Every region sits between sentinel bytes and leaves its pitch / stride gaps as sentinels, so any write outside
the command's outputs (C rows incl. the zero rows [N, NP), F scale vectors incl. their pad lanes) is a
mismatch.
The script also prints, per case, the kernel's error against a float64 torch norm (and torch's own FP32 one).

  C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\norm_residual\\gen_vectors.py OUTDIR [--real]
--real adds cases from real -12392 norm inputs (frame 0), captured with hooks (loads the policy).
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import contract  # noqa: E402
import nr_emul  # noqa: E402
from contract import C, Cmd, Slot  # noqa: E402

SENT = 0xA5
MAGIC = 0x3243524E


class Image:
    """A DDR window at byte address `base`: regions get 16-byte aligned addresses with sentinels around."""

    def __init__(self, base: int):
        self.base = base
        self.buf = bytearray()

    def alloc(self, nbytes: int, align: int = 64) -> int:
        self.buf += bytes([SENT]) * 48
        while (self.base + len(self.buf)) % align:
            self.buf.append(SENT)
        addr = self.base + len(self.buf)
        self.buf += bytes([SENT]) * nbytes
        return addr

    def finish(self) -> np.ndarray:
        self.buf += bytes([SENT]) * 80
        while len(self.buf) % 16:
            self.buf.append(SENT)
        return np.frombuffer(bytes(self.buf), np.uint8).copy()


def put(img: np.ndarray, base: int, addr: int, data: bytes) -> None:
    o = addr - base
    img[o : o + len(data)] = np.frombuffer(data, np.uint8)


def put_rows(img, base, s: Slot, rows: np.ndarray, nrows: int | None = None) -> None:
    """rows (B, N, X) of any dtype: row n of item b at s.addr + b s.stride + n s.pitch."""
    for b in range(rows.shape[0]):
        for n in range(rows.shape[1] if nrows is None else nrows):
            put(img, base, s.addr + b * s.stride + n * s.pitch, rows[b, n].tobytes())


def torch_norm64(y, g, b, eps, ln):
    yd = y.astype(np.float64)
    if ln:
        z = (yd - yd.mean(1, keepdims=True)) / np.sqrt(yd.var(1, keepdims=True) + eps) * g + b
    else:
        z = g * (yd / np.sqrt((yd * yd).mean(1, keepdims=True) + eps))
    return z


def torch_norm32(y, g, b, eps, ln):
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(y)
    if ln:
        return F.layer_norm(t, (y.shape[1],), torch.from_numpy(g), torch.from_numpy(b), eps).numpy()
    h = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
    return (torch.from_numpy(g) * h).numpy()


def write_case(path, base, list_addr, want, im, ex):
    assert im.size % 16 == 0
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IQQiI", MAGIC, base, list_addr, want, im.size // 16))
        fh.write(im.tobytes())
        fh.write(ex.tobytes())


def make_case(
    out,
    name,
    A,
    g,
    beta,
    eps,
    ln,
    qout,
    *,
    a_pad=0,
    a_gap_item=0,
    c_pad=0,
    c_gap_item=0,
    f_gap_item=0,
    np_extra=0,
    shift=0,
    base=0x1000_0000,
):
    """A: (B, N, K) FP32. *_pad: extra bytes per row pitch; *_gap_item: extra bytes per item stride."""
    A = np.ascontiguousarray(A, np.float32)
    Bn, N, K = A.shape
    NP = N + np_extra
    img = Image(base)
    a_pitch = 4 * K + a_pad
    a_stride = N * a_pitch + a_gap_item
    a_addr = img.alloc(Bn * a_stride)
    e_addr = img.alloc((8 if ln else 4) * K)
    eb = 2 if qout else 4
    c_pitch = eb * K + c_pad
    c_stride = NP * c_pitch + c_gap_item
    c_addr = img.alloc(Bn * c_stride)
    f_stride = 4 * contract.pad4(NP) + f_gap_item if qout else 0
    f_addr = img.alloc(Bn * max(f_stride, 16)) if qout else 0
    list_addr = img.alloc(C.REC_BYTES, 256)
    s = {
        "A": Slot(a_addr, a_pitch, a_stride if Bn > 1 else 0),
        "E": Slot(e_addr),
        "C": Slot(c_addr, c_pitch, c_stride if Bn > 1 else 0),
    }
    if qout:
        s["F"] = Slot(f_addr, 0, f_stride if Bn > 1 else 0)
    p = {C.P_EPS: contract.f32bits(eps)}
    if np_extra:
        p[C.P_NP] = NP
    if shift:
        p[C.P_SHIFT] = shift & 0xFFFFFFFF
    flags = (C.F_QOUT if qout else 0) | (C.F_LN if ln else 0)
    cmd = Cmd(C.OP_NORM, flags, 7, 0, K, N, Bn, s, p)
    me = contract.validate(cmd)
    assert me == C.ME_OK, (name, me)
    im = img.finish()
    base = img.base
    put_rows(im, base, s["A"], A)
    e = np.asarray(g, np.float32).tobytes() + (np.asarray(beta, np.float32).tobytes() if ln else b"")
    put(im, base, e_addr, e)
    put(im, base, list_addr, contract.encode(cmd))
    y2 = A.reshape(-1, K)
    z, q, sc, _, _ = nr_emul.kernel(y2, g, beta if ln else None, eps, ln, shift)
    ex = im.copy()
    if qout:
        qq = np.zeros((Bn, NP, K), np.int16)
        qq[:, :N] = q.reshape(Bn, N, K)
        put_rows(ex, base, s["C"], qq)
        sv = np.ones((Bn, contract.pad4(NP)), np.float32)
        sv[:, :N] = sc.reshape(Bn, N)
        for b in range(Bn):
            put(ex, base, f_addr + b * f_stride, sv[b].tobytes())
    else:
        put_rows(ex, base, s["C"], z.reshape(Bn, N, K))
    write_case(out / f"{name}.bin", base, list_addr, 0, im, ex)
    # accuracy against float64 (and torch's FP32 norm against the same reference)
    with np.errstate(all="ignore"):
        zt = torch_norm64(y2, g.astype(np.float64), beta if ln else 0.0, eps, ln)
        am = np.maximum(np.abs(zt).max(1), 1e-30)
        zrel = float((np.abs(z - zt).max(1) / am).max())
        deq = q.astype(np.float32) * (sc / np.float32(2.0**shift))[:, None]
        drel = float((np.abs(deq - zt).max(1) / am).max())
        z32 = torch_norm32(
            y2, np.asarray(g, np.float32), np.asarray(beta, np.float32) if ln else None, eps, ln
        )
        trel = float((np.abs(z32 - zt).max(1) / am).max())
    print(
        f"{name:22s} K={K:4d} N={N:3d} NP={NP:3d} B={Bn} flags=0x{flags:02x} words={im.size // 16:7d}  "
        f"z vs f64 {zrel:.2e} of row max (torch f32 {trel:.2e}), int16 dequant {drel:.2e}"
    )
    return z


def make_reject(out, name, cmd: Cmd, *, corrupt=None, base=0x2000_0000):
    """A record validate_hw (or decode) rejects: the expected image is the input image."""
    img = Image(base)
    list_addr = img.alloc(C.REC_BYTES, 256)
    im = img.finish()
    rec = bytearray(contract.encode(cmd))
    if corrupt:
        corrupt(rec)
    st, c = contract.decode(bytes(rec))
    if st != C.ST_END:
        want = st << 8
    elif c.op != C.OP_NORM:
        want = C.ST_BAD_OPCODE << 8
    else:
        me = contract.validate(c, full=False)
        assert me != C.ME_OK, name
        want = (C.ST_MODE_ERROR << 8) | me
    put(im, base, list_addr, bytes(rec))
    write_case(out / f"{name}.bin", base, list_addr, want, im, im.copy())
    print(f"{name:22s} rejected: want 0x{want:04x}")


def reject_cases(out):
    good = dict(
        op=C.OP_NORM,
        flags=C.F_QOUT,
        K=720,
        N=50,
        B=1,
        s={"A": Slot(0x1000, 2944), "E": Slot(0x40000), "C": Slot(0x80000, 1440), "F": Slot(0xC0000)},
        p={C.P_EPS: contract.f32bits(1e-5)},
    )
    assert contract.validate(Cmd(**good)) == C.ME_OK

    def var(**kw):
        d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in good.items()}
        for k, v in kw.items():
            if k in ("s", "p"):
                d[k].update(v)
            else:
                d[k] = v
        return Cmd(**d)

    make_reject(out, "rej_n0", var(N=0))
    make_reject(out, "rej_n65536", var(N=65536))
    make_reject(out, "rej_k1032", var(K=1032, s={"A": Slot(0x1000, 4128), "C": Slot(0x80000, 2064)}))
    make_reject(out, "rej_k12", var(K=12))
    make_reject(out, "rej_b65536", var(B=65536))
    make_reject(out, "rej_flag_resid", var(flags=C.F_QOUT | 0x4))
    make_reject(out, "rej_flag_sum", var(flags=C.F_QOUT | 0x8))
    make_reject(out, "rej_flag_bias", var(flags=C.F_QOUT | C.F_BIAS))
    make_reject(out, "rej_eps0", var(p={C.P_EPS: 0}))
    make_reject(out, "rej_eps1", var(p={C.P_EPS: contract.f32bits(1.0)}))
    make_reject(out, "rej_shift9", var(p={C.P_SHIFT: 9}))
    make_reject(out, "rej_seg64", var(p={C.P_SEG: 64}, K=768))
    make_reject(out, "rej_a_misaligned", var(s={"A": Slot(0x1008, 2944)}))
    make_reject(out, "rej_c_pitch", var(s={"C": Slot(0x80000, 1424)}))
    make_reject(out, "rej_slot_b", var(s={"B": Slot(0x9000)}))
    make_reject(out, "rej_f32_np", var(flags=0, s={"C": Slot(0x80000, 2880), "F": Slot()}, p={C.P_NP: 52}))
    make_reject(out, "rej_f32_fslot", var(flags=0, s={"C": Slot(0x80000, 2880)}))
    make_reject(out, "rej_addr40", var(s={"A": Slot(1 << 40, 2944)}))
    make_reject(out, "rej_magic", var(), corrupt=lambda r: r.__setitem__(1, 0xC5))
    make_reject(out, "rej_check", var(), corrupt=lambda r: r.__setitem__(20, r[20] ^ 1))

    def bad_version(r):
        f = list(struct.unpack("<64I", bytes(r)))
        f[C.FLD_VERSION] ^= 1
        f[C.CHECK_FIELD] ^= 1
        r[:] = struct.pack("<64I", *f)

    make_reject(out, "rej_version", var(), corrupt=bad_version)


def synth_cases(out):
    rng = np.random.default_rng(1)

    def rnd(*shape, scale=1.0, off=0.0):
        return (rng.standard_normal(shape) * scale + off).astype(np.float32)

    def gam(K, lo=0.3):
        return (rng.random(K) + lo).astype(np.float32)

    K = 768  # vision LN, residual stream with an outlier channel and a mean offset
    A = rnd(1, 64, K, scale=3.0, off=0.7)
    A[..., 11] *= 30
    make_case(out, "ln768_q", A, gam(K), rnd(K, scale=0.2), 1e-6, True, True)
    K = 960  # LM RMSNorm, 45 prefix rows
    A = rnd(1, 45, K, scale=2.0)
    A[..., 5] += 80
    make_case(out, "rms960_q", A, (rng.random(K) * 2).astype(np.float32), None, 1e-5, False, True)
    K = 720  # expert RMSNorm: rows of the 736-wide residual stream (flat read, 4 gap words a row), 2 items
    make_case(
        out,
        "rms720_b2_gap",
        rnd(2, 50, K, scale=1.5),
        gam(K, 0.5),
        None,
        1e-5,
        False,
        True,
        a_pad=64,
        a_gap_item=48,
        c_gap_item=32,
        f_gap_item=16,
    )
    K = 768  # post_layernorm: FP32 z out
    make_case(
        out, "ln768_f32", rnd(1, 32, K, scale=4.0, off=-0.3), gam(K), rnd(K, scale=0.2), 1e-6, True, False
    )
    K = 720  # A gap over half a row (one burst a row) and C rows with a gap (one write burst a row), 2 items
    make_case(
        out,
        "rms720_rowio",
        rnd(2, 20, K),
        gam(K, 0.5),
        None,
        1e-5,
        False,
        True,
        a_pad=4 * K // 2 + 16,
        c_pad=32,
        c_gap_item=64,
    )
    make_case(
        out,
        "ln720_rowio_f32",
        rnd(2, 9, K),
        gam(K),
        rnd(K, scale=0.1),
        1e-6,
        True,
        False,
        a_pad=2048,
        c_pad=16,
    )
    K = 720  # two writer chunks (N > 256), zero rows [N, NP) with scale 1.0, a scale shift (the Q rule's -3)
    make_case(
        out, "rms720_n300_np304_sh", rnd(1, 300, K), gam(K), None, 1e-5, False, True, np_extra=4, shift=-3
    )
    make_case(
        out,
        "ln768_n257_np260_sh5",
        rnd(1, 257, 768, off=2),
        gam(768),
        rnd(768),
        1e-6,
        True,
        True,
        np_extra=3,
        shift=5,
    )
    K = 1024  # the longest row
    make_case(out, "ln1024", rnd(1, 9, K, scale=2.0, off=1.0), gam(K), rnd(K, scale=0.2), 1e-6, True, True)
    make_case(out, "rms1024_f32", rnd(1, 7, K), gam(K), None, 1e-5, False, False)
    K = 8  # the shortest rows (scalar-bound: the per-row processes, not io, set the pace)
    make_case(out, "rms8_q", rnd(1, 37, K), gam(K), None, 1e-5, False, True, np_extra=3)
    make_case(out, "ln8_f32", rnd(3, 5, K, off=1.0), gam(K), rnd(K), 1e-6, True, False)
    # all-equal LayerNorm rows (review: the double-rounded mean gave +-32767 noise; torch gives beta)
    for K in (768, 960, 720):
        vals = [-1e4, 2.0, 3.7, 0.1, -7.25, 1e-3, 123456.79, -0.3333333, 0.0]
        A = np.stack([np.full(K, v, np.float32) for v in vals])[None]
        z = make_case(out, f"const_ln{K}", A, gam(K), rnd(K, scale=0.2), 1e-6, True, True)
        bz = make_case(out, f"const_ln{K}_f32", A, gam(K), np.zeros(K, np.float32), 1e-6, True, False)
        assert np.abs(bz).max() == 0.0, "constant LN rows must normalize to exactly 0 (z = beta = 0)"
        del z
    # edge rows (K = 64, N = 16: the last scale word is partial with NP = 18)
    K = 64
    rows = [
        np.zeros(K),  # zero row
        np.full(K, 3.25),  # constant row (var 0)
        1e4 + rng.standard_normal(K) * 1e-2,  # huge mean, small spread
        rng.standard_normal(K) * 1e-20,  # tiny values (var flushes to 0)
        rng.standard_normal(K) * 1e15,  # huge values
        np.r_[rng.standard_normal(K - 1), 1e9],  # exponents spanning more than 2^25
        np.eye(K)[7] * -2.5,  # one-hot
        np.array([(-1) ** i * 2.0 ** (i % 30 - 15) for i in range(K)]),  # powers of two
        rng.standard_normal(K) * 1e-3 + 5e-4,  # small, offset
        np.full(K, -7.0) + (np.arange(K) % 2) * 2**-20,  # tiny spread at -7
        rng.standard_normal(K) * 1e-39,  # denormals only (the cores flush them: a zero row)
        np.r_[rng.standard_normal(K // 2) * 1e-40, rng.standard_normal(K // 2)],  # half denormal
        np.r_[rng.standard_normal(K - 1), 2e20],  # LN var overflows FP32: rstd = 0, z = beta
        rng.standard_normal(K) * 3.0,
        rng.standard_normal(K) * 0.7 - 0.2,
        np.full(K, -0.0),  # negative zeros
    ]
    A = np.stack(rows).astype(np.float32)[None]
    g, beta = gam(K), rnd(K, scale=0.2)
    g[3] = 1e-39  # a denormal gamma lane (flushed)
    make_case(out, "edge_ln64", A, g, beta, 1e-6, True, True, np_extra=2)
    make_case(out, "edge_rms64", A, g, None, 1e-5, False, True, np_extra=2, shift=-8)
    make_case(out, "edge_ln64_f32", A, g, beta, 1e-6, True, False)
    make_case(out, "eps_denormal", A[:, :4], g, beta, 1e-45, True, True)  # eps flushes to 0: the 2^-126 clamp
    # 64-bit addressing: images that straddle 4 GB and sit in high DDR (0x8_0000_0000, the KR260's upper bank)
    make_case(
        out, "rms720_at_4g", rnd(2, 50, 720), gam(720), None, 1e-5, False, True, a_pad=64, base=0xFFFF_0000
    )
    make_case(out, "ln768_hi_ddr", rnd(1, 40, 768), gam(768), rnd(768), 1e-6, True, True, base=0x8_7000_0000)
    make_case(
        out, "rms960_64g", rnd(1, 12, 960), gam(960), None, 1e-5, False, True, base=(1 << 36) - (1 << 19)
    )  # word indices past 2^32


def real_cases(out):
    """Real norm inputs of -12392 frame 0: one command per group, A = the module input, contract layouts."""
    import policy_check as pc
    import torch
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
    got, hooks = {}, []
    for name, (mod, _) in want.items():

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
        y = got[name].numpy()
        g = mod.weight.detach().float().numpy()
        beta = mod.bias.detach().float().numpy() if ln else None
        eps = mod.eps if ln else mod.variance_epsilon
        qout = name != "real_vision_post"
        pad = 64 if y.shape[1] == 720 else 0  # the expert's residual rows hold 736 FP32 (contract_plan.py)
        make_case(out, name, y[None], g, beta, eps, ln, qout, a_pad=pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.bin"):
        f.unlink()
    reject_cases(out)
    synth_cases(out)
    if a.real:
        real_cases(out)


if __name__ == "__main__":
    main()
