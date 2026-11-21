"""fpga/board/runtime.py without a board, in both arithmetics: the board code path (Engine(mock="ip"): slabs,
pointer registers, the xmem image, tiled result buffer, un-tile with the bias fused in) against the layout
mock bit for bit, the int16 quantizers and operands against the precision study's quant_sym, the int16 result
against the DRIVER CONTRACT formula (fpga/hls/int16_linear.h) bit for bit, fused layers against separate
ones, the input memos, the register cache, the arithmetic probe, the FP32-stored layer list and the
structural group detection.
Needs numpy, torch and the host cores fpga/hls/build/libfloat_linear.so and libfloat_linear_int16.so
(make -C fpga/hls lib lib-int16).

    <venv>/python -m pytest fpga/board/test_runtime.py -q
    <venv>/python fpga/board/test_runtime.py            # without pytest
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import (  # noqa: E402
    QMAX16,
    TINY,
    Engine,
    FpgaLinear,
    chunk_layout,
    fp32_stored_linears,
    fpga_linears,
    fusion_groups,
    memo_hits,
    memo_modules,
    pack_weights,
    quantize_int16_into,
    token_scales_int16,
    xmem_bytes,
)

SHAPES = [  # (M, K, N): padded M, partial last group, 3 K chunks, N not a multiple of 32, tiny
    (64, 16, 3),
    (720, 960, 50),
    (320, 320, 177),
    (960, 12288, 64),
    (32, 720, 50),
    (2304, 768, 40),
]
SHAPES16 = SHAPES + [(96, 4100, 33), (736, 720, 81), (32, 8, 1)]  # a K = 4 chunk; the expert's 720 (pad)


def _layer(rng, m, k, bias=True):
    pf = Engine(mock="layout").pf
    w = rng.normal(0, 0.05, (m, k)).astype(np.float32)
    w *= np.exp(rng.normal(0, 0.5, (m, 1))).astype(np.float32)  # rows of different magnitude
    w = pf.bf16_bits_to_f32(pf.to_bf16_bits(w))  # BF16-exact, as the checkpoint's weights
    b = rng.normal(0, 0.5, m).astype(np.float32) if bias else None
    return w, b


def _x(rng, n, k):
    x = rng.normal(0, 1, (n, k)).astype(np.float32)
    x[:, rng.integers(0, k, max(1, k // 200))] *= 20  # outlier channels (SmolVLA crest factors 10-16)
    x *= np.exp(rng.normal(0, 1.0, (n, 1))).astype(np.float32)
    if n >= 3:
        x[1] = 0  # an all-zero token
    return x


def _check_ref(y, x, w, b, tol=1e-5):
    ref = x.astype(np.float64) @ w.astype(np.float64).T + (0 if b is None else b)
    scale = np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64)).T + 1e-30
    assert float((np.abs(y - ref) / scale).max()) <= tol


def quant_sym(x: torch.Tensor):
    """build/brief/precision_emulation.quant_sym(x, 32767, 1), verbatim (the precision study's V4)."""
    s = x.abs().amax(1, keepdim=True).clamp_min(TINY) / QMAX16
    q = torch.round(x / s).clamp_(-QMAX16, QMAX16)
    return q, s[:, 0]


def contract16(w: np.ndarray, b, x: np.ndarray) -> np.ndarray:
    """The DRIVER CONTRACT's result in numpy FP32: per K chunk y_c = (f32(sum Xq Wq) * s_row) * s_tok (each
    product rounded to FP32), chunks added in chunk order, then the bias. Inputs FP32."""
    pf = Engine(mock="layout").pf
    wq, sw = pf.quantize_int16_rows(w)
    xq, sx = pf.quantize_int16_tokens(x)
    y = np.zeros((x.shape[0], w.shape[0]), np.float32)
    for k0 in range(0, x.shape[1], 4096):
        acc = xq[:, k0 : k0 + 4096].astype(np.int64) @ wq[:, k0 : k0 + 4096].astype(np.int64).T
        y += (acc.astype(np.float64).astype(np.float32) * sw[None, :]) * sx[:, None]
    return y if b is None else y + b


def test_ip_matches_layout_bitwise():
    rng = np.random.default_rng(0)
    for arith in ("fp32", "int16"):
        e_l, e_i = Engine(mock="layout", arith=arith), Engine(mock="ip", arith=arith)
        for m, k, n in SHAPES16 if arith == "int16" else SHAPES:
            w, b = _layer(rng, m, k)
            x = _x(rng, n, k)
            fl, fi = FpgaLinear(w, b, e_l), FpgaLinear(w, b, e_i)
            xt = torch.from_numpy(x)[None]  # (1, n, k): the model's shape
            yl, yi = fl(xt), fi(xt)
            assert yl.shape == yi.shape == (1, n, m)
            assert torch.equal(yl, yi), (arith, m, k, n)
            _check_ref(yi[0].numpy(), x, w, b, 1e-5 if arith == "fp32" else 1e-3)
            # strided input (every other column of a wider tensor): copied/quantized straight into the buffer
            wide = torch.from_numpy(rng.normal(0, 1, (1, n, 2 * k)).astype(np.float32))
            assert not wide[..., ::2].is_contiguous()
            assert torch.equal(fi(wide[..., ::2]), fl(wide[..., ::2].contiguous()))
            # numpy path (check_engine.py, engine_server.py): gemm() returns [N][Mp]
            yn_l = e_l.gemm(fl.wbuf, fl.layout, fl.mp, x, srow=fl.srow)
            yn_i = e_i.gemm(fi.wbuf, fi.layout, fi.mp, x, srow=fi.srow)
            assert isinstance(yn_i, np.ndarray) and yn_i.shape == (n, fl.mp)
            assert np.array_equal(yn_l, yn_i)
            assert np.array_equal(yn_i[:, :m] + b, yi[0].numpy())


def test_quantizer_bits():
    """token_scales_int16 + quantize_int16_into == pack_float.quantize_int16_tokens == quant_sym, bit for
    bit, on outliers, zero rows, -0.0, tiny / denormal / huge rows, near-half quotients and strided views."""
    pf = Engine(mock="layout").pf
    rng = np.random.default_rng(5)
    x = _x(rng, 70, 1000)
    x[2] = 0
    x[3] = -0.0
    x[4] = rng.normal(0, 1e-36, 1000)  # below the 1e-30 floor
    x[5] = rng.normal(0, 1e-41, 1000)  # denormals
    x[6] = rng.normal(0, 1e37, 1000)
    x[7, :] = 0
    x[7, 17] = -3.5  # one non-zero value
    s7 = np.float32(1.0) / np.float32(QMAX16)
    x[8] = (np.arange(1000, dtype=np.float32) - 500 + 0.5) * s7  # near-.5 quotients
    x[8, 0] = 1.0
    xt = torch.from_numpy(x)
    for src in (xt, torch.from_numpy(np.ascontiguousarray(x.T)).T):  # contiguous and transposed (strided)
        s = token_scales_int16(src)
        q = torch.empty(src.shape, dtype=torch.int16)
        quantize_int16_into(src, s, torch.empty(src.shape), q)
        q_pf, s_pf = pf.quantize_int16_tokens(x)
        assert np.array_equal(q.numpy(), q_pf) and np.array_equal(
            s.numpy().view(np.uint32), s_pf.view(np.uint32)
        )
        q_e, s_e = quant_sym(xt)
        assert torch.equal(q.to(torch.float32), q_e)
        nz = (xt != 0).any(1)
        assert torch.equal(s[nz], s_e[nz]) and bool((s[~nz] == 1.0).all())
        assert int(q.min()) >= -QMAX16 and int(q.max()) <= QMAX16


def _capture_xmem(eng):
    """Record the xmem image and M/K/N of every int16 run of Engine(mock="ip")."""
    seen, start = [], eng.fake._start16

    def hook():
        mp, k, n = (eng.fake.regs[r] for r in (0x60, 0x68, 0x70))
        seen.append((mp, k, n, eng.fake._at(eng.fake._r64(0x48), xmem_bytes(mp, n, k)).copy()))
        start()

    eng.fake._start16 = hook
    return seen


def test_int16_operands_and_contract():
    """int16 random layers: weight and activation operands bit-identical to the precision study's quantizers
    (activations read back from the xmem image the board path wrote), y bit-identical to the contract
    formula, within FP32 rounding of the float64 dequantized product, and 1e-3 sum|x w| of the exact one."""
    pf = Engine(mock="layout").pf
    rng = np.random.default_rng(16)
    eng = Engine(mock="ip", arith="int16")
    seen = _capture_xmem(eng)
    for m, k, n in SHAPES16:
        w, b = _layer(rng, m, k)
        x = _x(rng, n, k)
        if m >= 64:
            w[m - 1] = 0  # an all-zero weight row
        # weights: pack_weights == pack(quant_sym rows)
        packed, srow, mp = pack_weights(eng, [w])
        wq_e, sw_e = quant_sym(torch.from_numpy(w))
        ref_packed, _ = pf.pack(wq_e.numpy().astype(np.int16).view(np.uint16))
        assert np.array_equal(packed, ref_packed)
        nzr = np.abs(w).max(1) > 0
        assert np.array_equal(srow[:m][nzr], sw_e.numpy()[nzr]) and np.all(srow[:m][~nzr] == 1.0)
        assert np.all(srow[m:] == 1.0)
        fl = FpgaLinear(w, b, eng)
        del seen[:]
        y = fl(torch.from_numpy(x))
        # activations: the xmem image of each K chunk, decoded
        assert [(s[0], s[1], s[2]) for s in seen] == [(mp, min(4096, k - k0), n) for k0, _, _ in fl.layout]
        xq_e, sx_e = quant_sym(torch.from_numpy(x))
        npad = -(-n // 32) * 32
        for (k0, _, _), (_, kc, _, img) in zip(fl.layout, seen, strict=True):
            f = img[: 4 * (mp + npad)].view(np.float32)
            assert np.array_equal(f[:mp].view(np.uint32), srow.view(np.uint32))
            st = f[mp : mp + npad]
            nzt = np.abs(x).max(1) > 0
            assert np.array_equal(st[:n][nzt], sx_e.numpy()[nzt]) and np.all(st[:n][~nzt] == 1.0)
            assert np.all(st[n:] == 1.0)
            xi = img[4 * (mp + npad) :].view(np.int16).reshape(npad // 32, kc, 32).transpose(0, 2, 1)
            xi = xi.reshape(npad, kc)
            assert np.array_equal(xi[:n], xq_e.numpy()[:, k0 : k0 + kc].astype(np.int16)), (m, k, n, k0)
            assert not xi[n:].any()
        # result: the contract formula bit for bit
        yc = contract16(w, b, x)
        assert np.array_equal(y.numpy().view(np.uint32), yc.view(np.uint32)), (m, k, n)
        # within FP32 rounding of the float64 dequantized product (one chunk: three FP32 roundings, float of
        # the sum, x s_row, x s_tok: |err| <= 3 * 2^-24 |y|; measured up to 2.007 ulp next to a binade) ...
        acc = xq_e.double() @ wq_e.double().T
        y64 = (acc * sx_e.double()[:, None] * sw_e.double()[None, :]).numpy()
        yg = eng.gemm(fl.wbuf, fl.layout, fl.mp, x, srow=fl.srow)  # no bias
        # the pre-quantized numpy path (check_engine.py, the probe) writes the same xmem image
        xq_pf, sx_pf = pf.quantize_int16_tokens(x)
        assert np.array_equal(eng.gemm_q16(fl.wbuf, fl.layout, fl.mp, xq_pf, sx_pf, fl.srow), yg)
        yn = yg[:, :m].astype(np.float64)
        if k <= 4096:
            assert np.all(np.abs(yn - y64) <= 3 * 2.0**-24 * np.abs(y64)), (m, k, n)
        else:
            sa = (xq_e.abs().double() * sx_e.double()[:, None]) @ (
                wq_e.abs().double() * sw_e.double()[:, None]
            ).T
            assert float((np.abs(yn - y64) / (sa.numpy() + 1e-30)).max()) <= 1e-6
        # ... and of the unquantized product
        _check_ref(y.numpy(), x, w, b, 1e-3)


def test_fused_matches_separate():
    rng = np.random.default_rng(1)
    for arith in ("fp32", "int16"):
        for mock in ("layout", "ip"):
            eng = Engine(mock=mock, arith=arith)
            for ms, k, n in [
                ((960, 320, 320), 720, 50),
                ((720, 48, 2048), 64, 37),
                ((768, 768, 768), 768, 70),
            ]:
                layers = [_layer(rng, m, k, bias=i != 1) for i, m in enumerate(ms)]
                sep = [FpgaLinear(w, b, eng) for w, b in layers]
                fused = FpgaLinear([w for w, _ in layers], [b for _, b in layers], eng)
                assert fused.mp == sum(-(-m // 32) * 32 for m in ms)
                x = torch.from_numpy(_x(rng, n, k)[None])
                ys = [s(x) for s in sep]
                calls = eng.calls
                order = [2, 0, 1]
                yf = {i: fused(x, i) for i in order}
                assert eng.calls == calls + 1, "one engine call serves all members"
                for i in range(len(ms)):
                    assert yf[i].shape == (1, n, ms[i])
                    assert torch.equal(yf[i], ys[i]), (arith, mock, ms, i)
                    assert np.array_equal(yf[i].numpy().view(np.uint32), ys[i].numpy().view(np.uint32))
                assert not fused._stash, "all members collected: nothing left behind"
                # a member called with another input runs its own fused call
                x2 = x + 1
                assert torch.equal(fused(x2, 1), sep[1](x2))
                assert eng.calls == calls + 3
                # an in-place change of the input between members invalidates the stash
                fused(x, 0)
                x.add_(0.0)  # bumps the version counter
                calls = eng.calls
                assert torch.equal(fused(x, 1), ys[1]) and eng.calls == calls + 1


def test_memo():
    rng = np.random.default_rng(2)
    for arith in ("fp32", "int16"):
        eng = Engine(mock="ip", arith=arith)
        w, b = _layer(rng, 320, 320)
        fl = FpgaLinear(w, b, eng, memo=True)
        x = torch.from_numpy(rng.normal(0, 1, (1, 177, 320)).astype(np.float32))
        calls = eng.calls
        y1 = fl(x)
        assert eng.calls == calls + 1
        y2 = fl(x.clone())  # same bits, another tensor
        assert eng.calls == calls + 1 and fl.memo_hits == 1 and torch.equal(y1, y2)
        y2.add_(1.0)  # the caller may modify what it got
        fl(x + 1e-3)
        assert eng.calls == calls + 2 and torch.equal(fl(x), y1)


def test_register_cache():
    rng = np.random.default_rng(3)
    for arith in ("fp32", "int16"):
        for cache, second in ((True, 1), (False, 16)):
            eng = Engine(mock="ip", cache_regs=cache, arith=arith, probe=False)
            writes = []
            wr = eng._wr
            eng._wr = lambda off, val, _wr=wr, _l=writes: (_l.append(off), _wr(off, val))
            w, b = _layer(rng, 320, 720)
            fl = FpgaLinear(w, b, eng)
            x = torch.from_numpy(rng.normal(0, 1, (1, 50, 720)).astype(np.float32))
            fl(x)
            assert len(writes) == 16, "first call: 15 words + ap_start"
            del writes[:]
            y = fl(x)
            assert len(writes) == second, (cache, writes)  # same layer and shape: only ap_start
            _check_ref(y[0].numpy(), x[0].numpy(), w, b, 1e-5 if arith == "fp32" else 1e-3)
            w2, b2 = _layer(rng, 320, 720)
            fl2 = FpgaLinear(w2, b2, eng)
            del writes[:]
            fl2(x)
            if cache:
                assert (
                    0x00 in writes
                    and 0x60 not in writes
                    and all(r in writes for r in (0x18, 0x24, 0x30, 0x3C))
                )


def test_probe_and_reject():
    """The probe catches a bitstream of the other arithmetic (same registers, garbage results); the int16
    IP's range check (M > 8192) surfaces as an error."""
    for arith, other in (("int16", "fp32"), ("fp32", "int16")):
        eng = Engine(mock="ip", arith=arith, probe=False)
        assert eng.probe() < (1e-5 if arith == "fp32" else 1e-3)
        eng.fake.arith, eng.fake.lib = other, Engine(mock="layout", arith=other).lib
        try:
            eng.probe()
        except RuntimeError as e:
            assert "other arithmetic" in str(e)
        else:
            raise AssertionError(f"probe passed an {other} IP for arith={arith}")
    eng = Engine(mock="ip", arith="int16")
    fl = FpgaLinear(np.ones((8224, 8), np.float32), None, eng)
    try:
        fl(torch.ones(1, 8))
    except RuntimeError as e:
        assert "status 1" in str(e)
    else:
        raise AssertionError("M = 8224 was accepted")


class _Attn(torch.nn.Module):
    def __init__(self, din, dq, dkv):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (
            torch.nn.Linear(din, dq),
            torch.nn.Linear(din, dkv),
            torch.nn.Linear(din, dkv),
        )
        self.o_proj = torch.nn.Linear(dq, din)

    def forward(self, h):
        return self.o_proj(self.q_proj(h) + torch.cat([self.k_proj(h), self.v_proj(h)], -1))


class _Mlp(torch.nn.Module):
    def __init__(self, d, di):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = (
            torch.nn.Linear(d, di, bias=False),
            torch.nn.Linear(d, di, bias=False),
            torch.nn.Linear(di, d, bias=False),
        )

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _Cross(torch.nn.Module):  # the expert's cross-attention: k/v project a 320-wide cache
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (
            torch.nn.Linear(96, 64),
            torch.nn.Linear(32, 64),
            torch.nn.Linear(32, 64),
        )

    def forward(self, h, kv):
        return self.q_proj(h) + self.k_proj(kv) + self.v_proj(kv)


def _toy():
    torch.manual_seed(0)
    model = torch.nn.ModuleDict({"attn": _Attn(96, 128, 64), "mlp": _Mlp(96, 256), "cross": _Cross()})
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.data = mod.weight.data.to(torch.bfloat16).float()
    return model


def test_fusion_groups_and_model_forward():
    model = _toy()
    assert fusion_groups(model) == [
        ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
    ]
    h = torch.randn(1, 40, 96)
    kv = torch.randn(1, 40, 32)
    with torch.no_grad():
        ref = (model["attn"](h), model["mlp"](h), model["cross"](h, kv))
        for arith in ("fp32", "int16"):
            m2 = _toy()
            eng = Engine(mock="ip", arith=arith)
            n = fpga_linears(m2, eng, keep_fp32=False)
            assert n == 10
            eng.calls = 0
            out = (m2["attn"](h), m2["mlp"](h), m2["cross"](h, kv))
            assert eng.calls == 2 + 2 + 3, "q/k/v and gate/up fused; the cross-attention stays 3 calls"
            for r, o in zip(ref, out, strict=True):
                assert float((r - o).abs().max()) < 3e-3 * float(r.abs().max()), (
                    arith,
                    float((r - o).abs().max()),
                )
            assert all(mod.weight.numel() == 0 for mod in m2.modules() if isinstance(mod, torch.nn.Linear))
            # unfused: same bits, one call per Linear
            m3 = _toy()
            eng3 = Engine(mock="ip", arith=arith)
            fpga_linears(m3, eng3, keep_fp32=False, fuse=False)
            eng3.calls = 0
            out3 = (m3["attn"](h), m3["mlp"](h), m3["cross"](h, kv))
            assert eng3.calls == 10
            assert all(torch.equal(a, c) for a, c in zip(out, out3, strict=True))


def _write_safetensors(path: Path, dtypes: dict) -> None:
    """A header-only .safetensors (zero-length tensors): enough for fp32_stored_linears."""
    header = {k: {"dtype": v, "shape": [0], "data_offsets": [0, 0]} for k, v in dtypes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_keep_fp32_and_kept_memo():
    """The checkpoint's F32 linear weights stay nn.Linear in torch (untouched, exact), the rest go to the
    engine; a kept cross-attention k/v module gets the input memo (same bits as recomputing)."""
    model = _toy()
    del model["cross"]
    vwe = torch.nn.Module()  # memo_modules' view of SmolVLA: the cross-attention layer's k/v
    vwe.lm_expert = torch.nn.Module()
    vwe.lm_expert.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    cross = vwe.lm_expert.layers[1].self_attn = _Cross()
    vwe.attention_mode, vwe.self_attn_every_n_layers = "cross_attn", 2
    vwe.num_vlm_layers = vwe.num_expert_layers = 2
    model["vwe"] = vwe
    kv_k, kv_v = "vwe.lm_expert.layers.1.self_attn.k_proj", "vwe.lm_expert.layers.1.self_attn.v_proj"
    assert memo_modules(model) == [kv_k, kv_v]
    with tempfile.TemporaryDirectory() as td:
        st = Path(td) / "model.safetensors"
        _write_safetensors(
            st,
            {
                "attn.q_proj.weight": "BF16",
                "attn.q_proj.bias": "F32",
                "attn.o_proj.weight": "F32",
                kv_k + ".weight": "F32",
                kv_v + ".weight": "F32",
                "mlp.gate_proj.weight": "BF16",
                "vwe.norm.weight": "F32",  # F32 but not a Linear
            },
        )
        assert fp32_stored_linears(model, td) == sorted(["attn.o_proj", kv_k, kv_v])
        z, kv = torch.randn(1, 40, 128), torch.randn(1, 40, 32)
        with torch.no_grad():
            ref_o = model["attn"].o_proj(z)
            ref_kv = (cross.k_proj(kv), cross.v_proj(kv))
            eng = Engine(mock="layout", arith="int16")
            info = {}
            fpga_linears(model, eng, checkpoint=st, info=info)
            assert info["kept_fp32"] == sorted(["attn.o_proj", kv_k, kv_v])
            assert model["attn"].o_proj.weight.numel() > 0  # kept: weights still in torch
            assert model["attn"].q_proj.weight.numel() == 0  # on the engine
            assert torch.equal(model["attn"].o_proj(z), ref_o)
            eng.calls = 0
            for _ in range(3):  # three denoising steps over the same prefix cache
                out = (cross.k_proj(kv.clone()), cross.v_proj(kv.clone()))
                assert all(torch.equal(a, r) for a, r in zip(out, ref_kv, strict=True))
            assert eng.calls == 0 and memo_hits(model) == 4
            out = cross.k_proj(kv + 1)
            assert torch.equal(
                out, torch.nn.functional.linear(kv + 1, cross.k_proj.weight, cross.k_proj.bias)
            )
    try:
        fpga_linears(_toy(), Engine(mock="layout"), keep_fp32=["nope"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown keep_fp32 name accepted")


def test_memo_modules_names():
    class _Vwe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_expert = torch.nn.Module()
            self.lm_expert.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
            self.attention_mode = "cross_attn"
            self.self_attn_every_n_layers = 2
            self.num_vlm_layers = self.num_expert_layers = 4

    root = torch.nn.Module()
    root.vwe = _Vwe()
    assert memo_modules(root) == [
        "vwe.lm_expert.layers.1.self_attn.k_proj",
        "vwe.lm_expert.layers.1.self_attn.v_proj",
        "vwe.lm_expert.layers.3.self_attn.k_proj",
        "vwe.lm_expert.layers.3.self_attn.v_proj",
    ]
    root.vwe.attention_mode = "self_attn"
    assert memo_modules(root) == []


def test_chunk_layout():
    assert chunk_layout(960, 12288) == [
        (0, 0, 30 * 4096),
        (4096, 4 * 30 * 4096, 30 * 4096),
        (8192, 8 * 30 * 4096, 30 * 4096),
    ]
    assert xmem_bytes(960, 50, 720) == 4 * 960 + 4 * 64 + 2 * 64 * 720


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}", flush=True)
    print("ALL PASS")
