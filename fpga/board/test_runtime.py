"""fpga/board/runtime.py without a board: the board code path (Engine(mock="ip"): slabs, pointer registers,
tiled result buffer, un-tile with the bias fused in) against the layout mock bit for bit, fused layers
against separate ones, the input memo, the register cache and the structural group detection.
Needs numpy, torch and fpga/hls/build/libfloat_linear.so (fpga/hls/float_shim.cpp).

    <venv>/python -m pytest fpga/board/test_runtime.py -q
    <venv>/python fpga/board/test_runtime.py            # without pytest
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import (  # noqa: E402
    Engine,
    FpgaLinear,
    chunk_layout,
    fpga_linears,
    fusion_groups,
    memo_modules,
)

SHAPES = [  # (M, K, N): padded M, partial last group, 3 K chunks, N not a multiple of 32, tiny
    (64, 16, 3),
    (720, 960, 50),
    (320, 320, 177),
    (960, 12288, 64),
    (32, 720, 50),
    (2304, 768, 40),
]


def _layer(rng, m, k, bias=True):
    pf = Engine(mock="layout").pf
    w = pf.bf16_bits_to_f32(pf.to_bf16_bits(rng.normal(0, 0.05, (m, k)).astype(np.float32)))
    b = rng.normal(0, 0.5, m).astype(np.float32) if bias else None
    return w, b


def _check_ref(y, x, w, b):
    ref = x.astype(np.float64) @ w.astype(np.float64).T + (0 if b is None else b)
    scale = np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64)).T + 1e-30
    assert float((np.abs(y - ref) / scale).max()) <= 1e-5


def test_ip_matches_layout_bitwise():
    rng = np.random.default_rng(0)
    e_l, e_i = Engine(mock="layout"), Engine(mock="ip")
    for m, k, n in SHAPES:
        w, b = _layer(rng, m, k)
        x = rng.normal(0, 1, (n, k)).astype(np.float32)
        fl, fi = FpgaLinear(w, b, e_l), FpgaLinear(w, b, e_i)
        xt = torch.from_numpy(x)[None]  # (1, n, k): the model's shape
        yl, yi = fl(xt), fi(xt)
        assert yl.shape == yi.shape == (1, n, m)
        assert torch.equal(yl, yi), (m, k, n)
        _check_ref(yi[0].numpy(), x, w, b)
        # strided input (every other column of a wider tensor): copied straight into the buffer
        wide = torch.from_numpy(rng.normal(0, 1, (1, n, 2 * k)).astype(np.float32))
        assert not wide[..., ::2].is_contiguous()
        assert torch.equal(fi(wide[..., ::2]), fl(wide[..., ::2].contiguous()))
        # numpy path (check_engine.py, engine_server.py): gemm() returns [N][Mp]
        yn_l = e_l.gemm(fl.wbuf, fl.layout, fl.mp, x)
        yn_i = e_i.gemm(fi.wbuf, fi.layout, fi.mp, x)
        assert isinstance(yn_i, np.ndarray) and yn_i.shape == (n, fl.mp)
        assert np.array_equal(yn_l, yn_i)
        assert np.array_equal(yn_i[:, :m] + b, yi[0].numpy())


def test_fused_matches_separate():
    rng = np.random.default_rng(1)
    for mock in ("layout", "ip"):
        eng = Engine(mock=mock)
        for ms, k, n in [((960, 320, 320), 720, 50), ((720, 48, 2048), 64, 37), ((768, 768, 768), 768, 70)]:
            layers = [_layer(rng, m, k, bias=i != 1) for i, m in enumerate(ms)]
            sep = [FpgaLinear(w, b, eng) for w, b in layers]
            fused = FpgaLinear([w for w, _ in layers], [b for _, b in layers], eng)
            assert fused.mp == sum(-(-m // 32) * 32 for m in ms)
            x = torch.from_numpy(rng.normal(0, 1, (1, n, k)).astype(np.float32))
            ys = [s(x) for s in sep]
            calls = eng.calls
            order = [2, 0, 1]
            yf = {i: fused(x, i) for i in order}
            assert eng.calls == calls + 1, "one engine call serves all members"
            for i in range(len(ms)):
                assert yf[i].shape == (1, n, ms[i])
                assert torch.equal(yf[i], ys[i]), (mock, ms, i)
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
    eng = Engine(mock="ip")
    w, b = _layer(rng, 320, 320)
    fl = FpgaLinear(w, b, eng, memo=True)
    x = torch.from_numpy(rng.normal(0, 1, (1, 177, 320)).astype(np.float32))
    y1 = fl(x)
    assert eng.calls == 1
    y2 = fl(x.clone())  # same bits, another tensor
    assert eng.calls == 1 and fl.memo_hits == 1 and torch.equal(y1, y2)
    y2.add_(1.0)  # the caller may modify what it got
    fl(x + 1e-3)
    assert eng.calls == 2 and torch.equal(fl(x), y1)


def test_register_cache():
    rng = np.random.default_rng(3)
    for cache, second in ((True, 1), (False, 16)):
        eng = Engine(mock="ip", cache_regs=cache)
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
        _check_ref(y[0].numpy(), x[0].numpy(), w, b)
        w2, b2 = _layer(rng, 320, 720)
        fl2 = FpgaLinear(w2, b2, eng)
        del writes[:]
        fl2(x)
        if cache:
            assert (
                0x00 in writes and 0x60 not in writes and all(r in writes for r in (0x18, 0x24, 0x30, 0x3C))
            )


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


def test_fusion_groups_and_model_forward():
    torch.manual_seed(0)
    model = torch.nn.ModuleDict({"attn": _Attn(96, 128, 64), "mlp": _Mlp(96, 256), "cross": _Cross()})
    assert fusion_groups(model) == [
        ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
    ]
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.data = mod.weight.data.to(torch.bfloat16).float()
    h = torch.randn(1, 40, 96)
    kv = torch.randn(1, 40, 32)
    with torch.no_grad():
        ref = (model["attn"](h), model["mlp"](h), model["cross"](h, kv))
        eng = Engine(mock="ip")
        n = fpga_linears(model, eng, fuse=True)
        assert n == 10
        eng.calls = 0
        out = (model["attn"](h), model["mlp"](h), model["cross"](h, kv))
    assert eng.calls == 2 + 2 + 3, "q/k/v and gate/up fused; the cross-attention stays 3 calls"
    for r, o in zip(ref, out, strict=True):
        assert float((r - o).abs().max()) < 1e-3 * float(r.abs().max()), float((r - o).abs().max())
    assert all(mod.weight.numel() == 0 for mod in model.modules() if isinstance(mod, torch.nn.Linear))


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}", flush=True)
    print("ALL PASS")
