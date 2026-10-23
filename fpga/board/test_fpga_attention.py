"""Tests for fpga/board/fpga_attention.py on a host, the HLS C core as the engine (Engine(mock="layout")).

    python fpga/board/test_fpga_attention.py identity    # the 3-part BF16 split is exact, bit for bit
    python fpga/board/test_fpga_attention.py unit        # random Q,K,V (1,12,1024,64) vs torch sdpa, float64
    python fpga/board/test_fpga_attention.py model --model <checkpoint dir> --batches <dir of frame_*.pt>
    python fpga/board/test_fpga_attention.py all --model ... --batches ...

Needs torch (LeRobot's venv for `model`) and fpga/hls/build/libfloat_linear.so (`make -C fpga/hls lib`; on
this Windows laptop the MinGW g++ bundled with Vitis, see fpga/hls/float_shim.cpp). Prints JSON lines of
numbers; exits 1 on a failed check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.fpga_attention import (  # noqa: E402
    FpgaAttention,
    attention_on_engine,
    parts_sum,
    split_bf16_parts,
    split_bf16_parts_torch,
)
from fpga.board.runtime import Engine, _pack_float, fast_tanh_gelu, fpga_linears  # noqa: E402

MINGW_BIN = r"C:\Xilinx\Vitis_HLS\2022.2\tps\win64\msys64\mingw64\bin"  # libgomp for a dynamic .so build
FAILED = []


def out(**kw) -> None:
    print(json.dumps(kw), flush=True)


def check(name: str, ok: bool, **kw) -> None:
    out(check=name, ok=bool(ok), **kw)
    if not ok:
        FAILED.append(name)


def engine(mock: str = "layout") -> Engine:
    if os.name == "nt" and os.path.isdir(MINGW_BIN):
        os.add_dll_directory(MINGW_BIN)
    return Engine(mock=mock)


# ---- 1. the split identity -----------------------------------------------------------------------
def torch_bf16_bits(x: np.ndarray) -> np.ndarray:
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    )


BF16_OVERFLOW = 0x7F7F8000  # |x| at or above this FP32 bit pattern (3.396e38) rounds to a BF16 infinity


def identity_stats(x: np.ndarray, pf) -> dict:
    """How many values the 3-part split (and the pack_float rounding against torch's) gets exactly."""
    x = np.ascontiguousarray(x, np.float32).reshape(-1)
    with np.errstate(invalid="ignore"):
        parts = split_bf16_parts(x, pf)
        s = parts_sum(parts, pf)
    overflow = (x.view(np.uint32) & 0x7FFFFFFF) >= BF16_OVERFLOW
    value_fail = int(np.count_nonzero((s != x) & ~overflow))
    bit_fail = int(np.count_nonzero((s.view(np.uint32) != x.view(np.uint32)) & ~overflow))
    r = (x - s)[~overflow]  # what a 4th part would have to carry
    hi_torch = int(np.count_nonzero(parts[0] != torch_bf16_bits(x)))
    import torch

    tparts = split_bf16_parts_torch(torch.from_numpy(x))  # the split attend() uses: same bits as numpy's
    torch_split = int(sum(np.count_nonzero((a != b) & ~overflow) for a, b in zip(parts, tparts, strict=True)))
    return {
        "torch_split_mismatch": torch_split,
        "n": int(x.size),
        "above_bf16_max": int(np.count_nonzero(overflow)),  # hi = inf: never exact, excluded below
        "value_mismatch": value_fail,
        "bit_mismatch": bit_fail,  # value_mismatch plus signed zeros (-0.0 splits to +0.0 parts)
        "max_abs_residual": float(np.abs(r).max()) if r.size else 0.0,
        "hi_vs_torch_bf16_mismatch": hi_torch,
    }


def random_fp32(rng, n: int, emin: int, emax: int) -> np.ndarray:
    """Random sign and mantissa bits, exponent field uniform in [emin, emax] (0 = denormal, 254 = top)."""
    e = rng.integers(emin, emax + 1, n, dtype=np.uint32)
    m = rng.integers(0, 1 << 23, n, dtype=np.uint32)
    sgn = rng.integers(0, 2, n, dtype=np.uint32) << 31
    return (sgn | (e << 23) | m).view(np.float32)


def test_identity(args) -> None:
    pf = _pack_float()
    rng = np.random.default_rng(0)
    n = 1 << 20
    sets = {
        "normal_0_1": rng.normal(0, 1, n).astype(np.float32),
        "normal_0_30": rng.normal(0, 30, n).astype(np.float32),
        "uniform_pm1e-3": rng.uniform(-1e-3, 1e-3, n).astype(np.float32),
        "bits_exp_24_254": random_fp32(rng, n, 24, 254),  # |x| >= 2^-103: the exact range
        "bits_exp_1_23": random_fp32(rng, n, 1, 23),  # normal but a part may be denormal
        "bits_exp_0": random_fp32(rng, n, 0, 0),  # FP32 denormals
        "bf16_exact": pf.bf16_bits_to_f32(pf.to_bf16_bits(rng.normal(0, 1, n).astype(np.float32))),
        "binade_edges": np.concatenate(
            [
                np.nextafter(np.float32(2.0) ** np.arange(-100, 100, dtype=np.float32), np.float32(0)),
                np.nextafter(np.float32(2.0) ** np.arange(-100, 100, dtype=np.float32), np.float32(np.inf)),
                np.float32(2.0) ** np.arange(-100, 100, dtype=np.float32),
                np.array([0.0, -0.0, 1.0, -1.0, 3.4e38, -3.4e38, 1.17549435e-38], np.float32),
            ]
        ),
    }
    for name, x in sets.items():
        st = identity_stats(x, pf)
        exact_expected = name not in ("bits_exp_1_23", "bits_exp_0")
        out(identity=name, **st)
        check(f"identity_torch_split_{name}", st["torch_split_mismatch"] == 0)
        if exact_expected:
            check(f"identity_{name}", st["value_mismatch"] == 0 and st["hi_vs_torch_bf16_mismatch"] == 0)
    # the all-exponents sweep: every finite value with exponent field >= 24 (|x| >= 2^-103) below the
    # BF16 overflow point must be exact; below exponent 24 a part can be an FP32 denormal
    x = random_fp32(rng, n, 1, 254)
    bits = x.view(np.uint32)
    e = (bits >> 23) & 0xFF
    with np.errstate(invalid="ignore"):
        s = parts_sum(split_bf16_parts(x, pf), pf)
    bad = (s != x) & ((bits & 0x7FFFFFFF) < BF16_OVERFLOW)
    out(
        identity="sweep_exp_1_254",
        mismatches_below_overflow=int(bad.sum()),
        max_exponent_field_of_mismatch=int(e[bad].max()) if bad.any() else None,
        above_bf16_max=int(((bits & 0x7FFFFFFF) >= BF16_OVERFLOW).sum()),
    )
    check("identity_sweep_exponent_ge_24_exact", not bad.any() or int(e[bad].max()) < 24)


# ---- 2. attention on random Q, K, V ------------------------------------------------------------------
def errors(y, ref64) -> dict:
    y64 = y.astype(np.float64)
    err = np.abs(y64 - ref64)
    scale = np.abs(ref64).max()
    return {
        "max_abs": float(err.max()),
        "rms": float(np.sqrt((err**2).mean())),
        "max_abs_over_max_ref": float(err.max() / scale),
        "max_rel_floor1e-2": float((err / np.maximum(np.abs(ref64), 1e-2)).max()),
    }


def test_unit(args) -> None:
    import torch
    import torch.nn.functional as F

    eng = engine()
    torch.manual_seed(0)
    b, h, t, d = 1, 12, 1024, 64
    scale = d**-0.5
    for std in (1.0, 4.0):  # std 4: sharp softmax rows, as the real tower's later layers
        q = torch.randn(b, h, t, d) * std
        k = torch.randn(b, h, t, d) * std
        v = torch.randn(b, h, t, d)
        q64, k64, v64 = (x.double().numpy()[0] for x in (q, k, v))
        s64 = np.einsum("htd,hkd->htk", q64, k64) * scale
        p64 = np.exp(s64 - s64.max(-1, keepdims=True))
        p64 /= p64.sum(-1, keepdims=True)
        o64 = np.einsum("htk,hkd->htd", p64, v64)
        # torch's own FP32 paths against float64
        t0 = time.perf_counter()
        o_sdpa = F.scaled_dot_product_attention(q, k, v, scale=scale)
        t_sdpa = time.perf_counter() - t0
        s32 = torch.matmul(q, k.transpose(-1, -2)) * scale
        o_eager = torch.matmul(torch.softmax(s32, dim=-1), v)
        out(unit="torch_sdpa_fp32", std=std, seconds=round(t_sdpa, 3), **errors(o_sdpa.numpy()[0], o64))
        out(unit="torch_eager_fp32", std=std, **errors(o_eager.numpy()[0], o64))
        out(unit="torch_scores_fp32", std=std, **errors(s32.numpy()[0], s64))
        # the engine path: (B, T, H, D) in and out
        qt, kt, vt = (x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
        results = {}
        for mode, qk, pv in (
            ("qk_pv", "k", "m"),  # default
            ("qk_pv", "m", "m"),
            ("qk_pv", "separate", "separate"),
            ("qk_pv", "k", "separate"),
            ("qk_pv", "m", "k"),
            ("qk_pv", "k", "k"),
            ("qk", "k", "m"),
            ("pv", "k", "m"),
            ("torch", "k", "m"),
        ):
            captured = {}

            def softmax_capture(s, _c=captured):
                _c["scores"] = s.clone()
                p = torch.softmax(s, dim=-1)
                _c["probs"] = p
                return p

            fa = FpgaAttention(eng, mode=mode, softmax=softmax_capture, qk_layout=qk, pv_layout=pv)
            t0 = time.perf_counter()
            o = fa.attend(qt, kt, vt, None, scale)
            dt = time.perf_counter() - t0
            o = o.permute(0, 2, 1, 3).contiguous().numpy()[0]
            tag = f"{mode}:{qk}/{pv}"
            results[tag] = o
            row = dict(
                unit="engine_attention",
                std=std,
                mode=mode,
                qk_layout=qk,
                pv_layout=pv,
                engine_calls=fa.calls,
                macs=fa.macs,
                seconds=round(dt, 2),
                **errors(o, o64),
            )
            if mode in ("qk_pv", "qk"):  # the scores of the last head, against float64
                row["scores_last_head"] = errors(captured["scores"].numpy(), s64[h - 1])
            if mode in ("qk_pv", "pv"):  # P V of the last head with the probabilities this path used
                pv64 = captured["probs"].double().numpy() @ v64[h - 1]
                row["pv_last_head"] = errors(o[h - 1], pv64)
            out(**row)
            e_sdpa = errors(o_sdpa.numpy()[0], o64)["max_abs"]
            e_mine = errors(o, o64)["max_abs"]
            check(
                f"unit_error_class_{tag}_std{std}",
                e_mine <= 4 * max(e_sdpa, 1e-7),
                max_abs_mine=e_mine,
                max_abs_torch_sdpa=e_sdpa,
            )

        # parts stacked along M and three separate calls compute the same lanes and add them in the same
        # order on the ARM: bit for bit the same (Q K^T with the P V layout fixed, then P V with Q K^T fixed)
        def same(a, b, _r=results):
            return bool(np.array_equal(_r[a].view(np.uint32), _r[b].view(np.uint32)))

        check(f"unit_qk_m_vs_separate_bitwise_std{std}", same("qk_pv:m/m", "qk_pv:separate/separate"))
        check(f"unit_pv_m_vs_separate_bitwise_std{std}", same("qk_pv:k/m", "qk_pv:k/separate"))
    # a bool mask and a batch of 2 on a small case (the code path, not the model's)
    q = torch.randn(2, 3, 96, 32)
    k = torch.randn(2, 3, 80, 32)
    v = torch.randn(2, 3, 80, 32)
    mask = torch.rand(2, 1, 96, 80) > 0.2
    mask[..., 0] = True
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=32**-0.5)
    fa = FpgaAttention(eng)
    o = fa.attend(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        mask,
        32**-0.5,
    )
    diff = float((o.permute(0, 2, 1, 3) - ref).abs().max())
    check("unit_bool_mask_batch2", diff < 1e-5, max_abs_vs_torch=diff, engine_calls=fa.calls)
    # a scale that is not a power of two goes onto the scores
    ref = F.scaled_dot_product_attention(q, k, v, scale=0.3)
    o = fa.attend(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        None,
        0.3,
    )
    diff = float((o.permute(0, 2, 1, 3) - ref).abs().max())
    check("unit_scale_not_pow2", diff < 1e-5, max_abs_vs_torch=diff)
    # the board-side plumbing over the host library: two threads at once (overlap_image_embedding), each
    # with its own scratch weight buffer, every output bit for bit the mock engine's
    fake = _FakeBoard(eng)
    fa_hw = FpgaAttention(fake)
    q, k, v = (torch.randn(1, 256, 4, 64) for _ in range(3))
    ref = FpgaAttention(eng).attend(q, k, v, None, 0.125)
    outs = [None, None]

    def work(i):
        outs[i] = fa_hw.attend(q, k, v, None, 0.125)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    same = all(o is not None and torch.equal(o, ref) for o in outs)
    check(
        "unit_fake_board_scratch_two_threads", same and fake.allocs == 2, allocs=fake.allocs, calls=fake.calls
    )
    # the same through runtime's register-level emulation of float_top, when this runtime has it: the
    # scratch weights go through the real board code (_gemm_hw, pointers in registers, tiled result)
    try:
        ip = Engine(mock="ip")
    except (ValueError, TypeError):
        out(unit="engine_mock_ip", available=False)
    else:
        fa_ip = FpgaAttention(ip)
        o_ip = fa_ip.attend(q, k, v, None, 0.125)
        check("unit_mock_ip_bitwise_vs_layout", torch.equal(o_ip, ref), engine_calls=fa_ip.calls)


class _FakeBoard:
    """Engine's board-side interface over the host library: mock = None, scratch buffers from _alloc(),
    gemm() with torch tensors in and out, so FpgaAttention's per-thread scratch-weight path runs here."""

    class Buf:
        def __init__(self, n):
            self.array = np.zeros(n, np.uint8)
            self.physical_address = 0
            self.flushed = 0

        def flush(self, off, n):
            self.flushed += n

        def free(self):
            self.array = np.zeros(0, np.uint8)

    def __init__(self, host: Engine):
        self.pf, self.lib, self.mock = host.pf, host.lib, None
        self.calls, self.fpga_s, self.allocs = 0, 0.0, 0
        self.lock = threading.Lock()

    def _alloc(self, n):
        self.allocs += 1
        return _FakeBoard.Buf(n)

    def gemm(self, wbuf, layout, mp, x):
        import torch

        words = wbuf.slab.array[wbuf.offset : wbuf.offset + wbuf.nbytes].view(np.uint16)
        n, k = x.shape
        y = np.zeros((n, mp), np.float32)
        for k0, word_off, pw in layout:
            kc = min(4096, k - k0)
            y += self.lib(words[8 * word_off : 8 * (word_off + 4 * pw)], mp, x[:, k0 : k0 + kc].numpy())
        with self.lock:
            self.calls += 1
        return torch.from_numpy(y)


# ---- 3. the policy on a recorded frame -------------------------------------------------------------
def test_model(args) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(args.threads)
    eng = engine(args.engine)
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.model).to("cpu").eval()
    n_lin = fpga_linears(policy, eng)
    policy.float()
    n_gelu = fast_tanh_gelu(policy)
    vwe = policy.model.vlm_with_expert
    images = []  # the preprocessed camera images, for the tower-only comparison below
    orig_embed_image = vwe.embed_image

    def embed_image_recording(img):
        images.append(img.detach().clone())
        return orig_embed_image(img)

    vwe.embed_image = embed_image_recording  # before overlap_image_embedding, which captures embed_image
    if args.overlap:
        from fpga.board.runtime import overlap_image_embedding

        overlap_image_embedding(policy, eng)
    out(
        model="loaded",
        linears_on_engine=n_lin,
        tanh_gelu_patched=n_gelu,
        seconds=round(time.perf_counter() - t0, 1),
    )
    frames = sorted(Path(args.batches).glob("frame_*.pt"))[: args.frames]
    if not frames:
        sys.exit(f"no frame_*.pt in {args.batches}")

    def run(tag, fa=None):
        d = torch.load(frames[0], weights_only=False)
        eng.calls, eng.fpga_s = 0, 0.0
        policy.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(d["batch"], noise=d["noise"])
        wall = time.perf_counter() - t0
        diff = float((chunk.float() - d["ref_chunk_raw"].float()).abs().max())
        row = dict(
            model=tag,
            frame=d["frame"],
            max_abs_diff_normalized=diff,
            seconds=round(wall, 1),
            engine_calls=eng.calls,
            engine_seconds=round(eng.fpga_s, 1),
        )
        if fa is not None:
            row.update(
                attention_calls=fa.attn_calls,
                attention_engine_calls=fa.calls,
                attention_macs=fa.macs,
                attention_seconds={k: round(v, 2) for k, v in fa.t.items()},
                kv_values_checked=fa.exact_total,
                kv_split_not_exact=fa.exact_fail,
                tower_attention_before=fa.attn_implementation,
            )
        out(**row)
        return chunk, diff

    chunk0, diff0 = run("baseline_linears_only")
    fa = attention_on_engine(policy, eng, mode=args.mode, qk_layout=args.qk_layout, pv_layout=args.pv_layout)
    fa.check_exact = True
    chunk1, diff1 = run(f"attention_{args.mode}_{args.qk_layout}_{args.pv_layout}", fa)
    out(model="chunk_vs_chunk", max_abs_diff_between_paths=float((chunk1 - chunk0).abs().max()))
    check("model_within_threshold", diff1 < 0.05, max_abs_diff_normalized=diff1, baseline=diff0)
    check(
        "model_same_error_class", diff1 <= 3 * max(diff0, 1e-3), max_abs_diff_normalized=diff1, baseline=diff0
    )
    check("model_kv_split_exact", fa.exact_fail == 0, checked=fa.exact_total, not_exact=fa.exact_fail)
    # the vision tower alone on the first recorded image: engine attention vs torch's, same linears
    vm = vwe.get_vlm_model().vision_model
    img = images[0].to(next(vm.parameters()).dtype)
    with torch.no_grad():
        h1 = vm(pixel_values=img).last_hidden_state
        fa.restore()
        h0 = vm(pixel_values=img).last_hidden_state
    out(
        model="vision_tower_one_image",
        tokens=list(h0.shape),
        max_abs_diff=float((h1 - h0).abs().max()),
        max_abs_ref=float(h0.abs().max()),
        rms_diff=float(((h1 - h0) ** 2).mean().sqrt()),
    )
    check("vision_tower_close", float((h1 - h0).abs().max()) < 1e-2 * float(h0.abs().max()))
    if len(frames) > 1:
        fa = attention_on_engine(
            policy, eng, mode=args.mode, qk_layout=args.qk_layout, pv_layout=args.pv_layout
        )
        for f in frames[1:]:
            d = torch.load(f, weights_only=False)
            policy.reset()
            with torch.no_grad():
                chunk = policy.predict_action_chunk(d["batch"], noise=d["noise"])
            diff = float((chunk.float() - d["ref_chunk_raw"].float()).abs().max())
            out(model="attention_more_frames", frame=d["frame"], max_abs_diff_normalized=diff)
            check(f"model_frame_{d['frame']}", diff < 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["identity", "unit", "model", "all"])
    ap.add_argument("--model", help="SmolVLA checkpoint dir")
    ap.add_argument("--batches", help="dir of frame_*.pt")
    ap.add_argument("--frames", type=int, default=1, help="how many recorded frames to run")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--engine", default="layout", help="Engine mock for `model`: layout, or ip (board code path)"
    )
    ap.add_argument("--mode", default="qk_pv")
    ap.add_argument("--qk-layout", default="k")
    ap.add_argument("--pv-layout", default="m")
    ap.add_argument(
        "--overlap", action="store_true", help="also install overlap_image_embedding (two threads)"
    )
    args = ap.parse_args()
    if args.part in ("identity", "all"):
        test_identity(args)
    if args.part in ("unit", "all"):
        test_unit(args)
    if args.part in ("model", "all"):
        if not (args.model and args.batches):
            sys.exit("model needs --model and --batches")
        test_model(args)
    out(result="FAIL" if FAILED else "PASS", failed=FAILED)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
