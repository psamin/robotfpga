"""Full-policy effect of the NORM mode (contract 1.0) on checkpoint -12392 (3 replay frames).

Every LayerNorm (vision) and RMSNorm (LM prefix, action expert) module instance gets its forward replaced by
the mode's exact arithmetic (nr_emul.py: integer statistics, correctly rounded mean / var, FP32 cores with
denormals flushed, the contract quantizer). The residual adds stay torch's: under the contract they are the
GEMM epilogue's FP32 add (contract 2.3), the same IEEE operation. Variants, all with FP32 linears so that only
the norms change:

  torch      no patch (sanity: the laptop against the cluster reference)
  zf32       the mode's statistics + normalize, FP32 z out everywhere (the normalize arithmetic alone)
  kernel     the mode as deployed: int16 q * s (contract quantizer, quantize/quant_int16.h) into every linear
             that follows a norm, FP32 z for the vision post_layernorm (no F_QOUT: the pixel shuffle follows)
  torchq16   torch's norm + the same contract quantizer at the same points: what the int16 path does to these
             activations anyway, with an exact norm in front
  kfold      (a proposal, not the contract) gamma / beta folded into the next linear: the mode quantizes
             ((y - mean) * rstd) and the linear sees (q * s) * g + b (FP32 linears: only the activation-side
             effect; the re-quantization of W * diag(g) is not modelled). Saves 8 of the 16 FP32 cores
For each frame: max |chunk - ref_chunk_raw| (normalized action units; gate 0.05), and kernel vs torchq16.
Also: norm calls / rows / elements per chunk, and row statistics (|mean|/std, max|y|/std).

  set HF_HUB_OFFLINE=1
  C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\norm_residual\policy_check.py [--frames 0,1,2]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import nr_emul  # noqa: E402

CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"

MODE = {"v": "torch"}
COUNT = collections.Counter()  # (group, kind, D, out) -> calls ; rows
ROWS = collections.Counter()
STATS = collections.defaultdict(lambda: dict(n=0, mean_over_std=0.0, max_over_std=0.0, minstd=1e30))
KERR = collections.defaultdict(
    lambda: dict(n=0, z_rel=0.0, deq_rel=0.0, q_mismatch=0, q_total=0, qdiff_max=0)
)


def _np(t):
    return None if t is None else t.detach().to(torch.float32).cpu().numpy()


def kernel_norm(y, weight, bias, eps, layernorm, out):
    """y (..., D). out = 'q16' (dequantized int16) or 'zf32'."""
    shp = y.shape
    y2 = _np(y.reshape(-1, shp[-1]))
    z, q, s, _, _ = nr_emul.kernel(y2, _np(weight), _np(bias), eps, layernorm)
    r = z if out == "zf32" else q.astype(np.float32) * s[:, None]
    return torch.from_numpy(np.ascontiguousarray(r)).reshape(shp).to(y.dtype)


def kernel_norm_fold(y, weight, bias, eps, layernorm):
    """gamma / beta folded downstream: quantize the normalized row, then apply g, b in FP32."""
    shp = y.shape
    g = _np(weight)
    one = np.ones_like(g)
    zero = np.zeros_like(g) if layernorm else None
    _, q, s, _, _ = nr_emul.kernel(_np(y.reshape(-1, shp[-1])), one, zero, eps, layernorm)
    r = (q.astype(np.float32) * s[:, None]) * g[None, :]
    if bias is not None:
        r = r + _np(bias)[None, :]
    return torch.from_numpy(np.ascontiguousarray(r, np.float32)).reshape(shp).to(y.dtype)


def torch_norm(y, weight, bias, eps, layernorm):
    if layernorm:
        return F.layer_norm(y, (y.shape[-1],), weight, bias, eps)
    h = y.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return weight * h.to(y.dtype)


def make_forward(mod, group, layernorm, out_kind):
    """out_kind: 'q16' (int16 to a linear) or 'zf32' (FP32 out: post_layernorm) or 'unused'."""
    eps = mod.eps if layernorm else mod.variance_epsilon
    bias = mod.bias if layernorm else None

    def fwd(y):
        D = y.shape[-1]
        rows = y.numel() // D
        COUNT[(group, "LN" if layernorm else "RMS", D, out_kind)] += 1
        ROWS[(group, "LN" if layernorm else "RMS", D, out_kind)] += rows
        v = MODE["v"]
        if v == "stats":
            y2 = y.reshape(-1, D).double()
            m = y2.mean(1)
            sd = y2.std(1, unbiased=False)
            st = STATS[group]
            st["n"] += rows
            st["mean_over_std"] = max(st["mean_over_std"], float((m.abs() / sd).max()))
            st["max_over_std"] = max(st["max_over_std"], float((y2.abs().amax(1) / sd).max()))
            st["minstd"] = min(st["minstd"], float(sd.min()))
            # kernel vs torch on this very call
            y2f = y.reshape(-1, D).to(torch.float32)
            zt = _np(
                torch_norm(y2f, mod.weight.detach(), None if bias is None else bias.detach(), eps, layernorm)
            )
            z, q, s, _, _ = nr_emul.kernel(_np(y2f), _np(mod.weight), _np(bias), eps, layernorm)
            amax = np.maximum(np.abs(zt).max(1), 1e-30)
            ke = KERR[(group, "LN" if layernorm else "RMS")]
            ke["n"] += rows
            ke["z_rel"] = max(ke["z_rel"], float((np.abs(z - zt).max(1) / amax).max()))
            deq = q.astype(np.float32) * s[:, None]
            ke["deq_rel"] = max(ke["deq_rel"], float((np.abs(deq - zt).max(1) / amax).max()))
            qt, _ = nr_emul.torch_quant(zt)
            dq = qt.astype(np.int32) - q.astype(np.int32)
            ke["q_mismatch"] += int((dq != 0).sum())
            ke["q_total"] += q.size
            ke["qdiff_max"] = max(ke["qdiff_max"], int(np.abs(dq).max()))
            return torch_norm(y, mod.weight, bias, eps, layernorm)
        if v in ("torch", "count"):
            return torch_norm(y, mod.weight, bias, eps, layernorm)
        if v == "zf32" or (v == "kernel" and out_kind != "q16"):
            return kernel_norm(y, mod.weight, bias, eps, layernorm, "zf32")
        if v == "kernel":
            return kernel_norm(y, mod.weight, bias, eps, layernorm, "q16")
        if v == "kfold":
            if out_kind != "q16":
                return kernel_norm(y, mod.weight, bias, eps, layernorm, "zf32")
            return kernel_norm_fold(y, mod.weight, bias, eps, layernorm)
        if v == "torchq16":
            z = torch_norm(y, mod.weight, bias, eps, layernorm)
            if out_kind != "q16":
                return z
            q, s = nr_emul.torch_quant(_np(z.reshape(-1, D)))
            r = q.astype(np.float32) * s[:, None]
            return torch.from_numpy(r).reshape(y.shape).to(y.dtype)
        raise ValueError(v)

    return fwd


def install(p):
    vwe = p.model.vlm_with_expert
    vm = vwe.get_vlm_model()
    vis = vm.vision_model
    n = 0
    for L in vis.encoder.layers:
        L.layer_norm1.forward = make_forward(L.layer_norm1, "vision", True, "q16")
        L.layer_norm2.forward = make_forward(L.layer_norm2, "vision", True, "q16")
        n += 2
    vis.post_layernorm.forward = make_forward(vis.post_layernorm, "vision_post", True, "zf32")
    n += 1
    tm = vm.text_model
    for L in tm.layers:
        L.input_layernorm.forward = make_forward(L.input_layernorm, "lm", False, "q16")
        L.post_attention_layernorm.forward = make_forward(L.post_attention_layernorm, "lm", False, "q16")
        n += 2
    tm.norm.forward = make_forward(tm.norm, "lm_final", False, "unused")
    n += 1
    ex = vwe.lm_expert
    for L in ex.layers:
        L.input_layernorm.forward = make_forward(L.input_layernorm, "expert", False, "q16")
        L.post_attention_layernorm.forward = make_forward(L.post_attention_layernorm, "expert", False, "q16")
        n += 2
    ex.norm.forward = make_forward(ex.norm, "expert_final", False, "q16")
    n += 1
    # every norm module in the policy must be one of ours
    total = sum(1 for m in p.modules() if isinstance(m, nn.LayerNorm) or type(m).__name__.endswith("RMSNorm"))
    return n, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--variants", default="torch,zf32,kernel,torchq16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "policy_check.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    p = SmolVLAPolicy.from_pretrained(str(CKPT)).to("cpu").eval().float()
    print(f"loaded in {time.perf_counter() - t0:.1f} s")
    vwe = p.model.vlm_with_expert
    vc, tc, ec = vwe.config.vision_config, vwe.config.text_config, vwe.lm_expert.config
    cfg = dict(
        vision=dict(
            layers=vc.num_hidden_layers,
            hidden=vc.hidden_size,
            ln_eps=vc.layer_norm_eps,
            image_size=vc.image_size,
            patch=vc.patch_size,
        ),
        text=dict(
            layers=vwe.num_vlm_layers,
            hidden=tc.hidden_size,
            heads=tc.num_attention_heads,
            kv_heads=tc.num_key_value_heads,
            rms_eps=tc.rms_norm_eps,
        ),
        expert=dict(
            layers=vwe.num_expert_layers,
            hidden=ec.hidden_size,
            heads=ec.num_attention_heads,
            kv_heads=ec.num_key_value_heads,
            rms_eps=ec.rms_norm_eps,
            intermediate=ec.intermediate_size,
        ),
        num_steps=p.config.num_steps,
        chunk=p.config.chunk_size,
    )
    print("config:", json.dumps(cfg))
    n, total = install(p)
    print(f"patched {n} norm modules of {total} in the policy")
    frames = sorted(BATCHES.glob("frame_*.pt"))
    fsel = [int(f) for f in a.frames.split(",")]
    res = dict(config=cfg, frames={}, counts=None)
    for fi in fsel:
        d = torch.load(frames[fi], weights_only=False)
        ref = d["ref_chunk_raw"].float()
        out = {}
        chunks = {}
        for v in ["stats"] + a.variants.split(","):
            MODE["v"] = v
            COUNT.clear()
            ROWS.clear()
            p.reset()
            t1 = time.perf_counter()
            with torch.no_grad():
                ch = p.predict_action_chunk(d["batch"], noise=d["noise"]).float()
            wall = time.perf_counter() - t1
            chunks[v] = ch
            err = float((ch - ref).abs().max())
            mean_err = float((ch - ref).abs().mean())
            out[v] = dict(max_err=err, mean_err=mean_err, wall=wall)
            print(
                f"frame {d['frame']} {v:9s} max|chunk-ref| {err:.3e} mean {mean_err:.3e} ({wall:.1f} s)",
                flush=True,
            )
            if v == "stats" and res["counts"] is None:
                res["counts"] = [
                    dict(
                        group=k[0],
                        kind=k[1],
                        D=k[2],
                        out=k[3],
                        calls=c,
                        rows=ROWS[k],
                        elements=ROWS[k] * k[2],
                    )
                    for k, c in sorted(COUNT.items())
                ]
        if "kernel" in chunks and "torchq16" in chunks:
            out["kernel_vs_torchq16_max"] = float((chunks["kernel"] - chunks["torchq16"]).abs().max())
        if "zf32" in chunks and "torch" in chunks:
            out["zf32_vs_torch_max"] = float((chunks["zf32"] - chunks["torch"]).abs().max())
        print(
            f"frame {d['frame']}: kernel vs torchq16 {out.get('kernel_vs_torchq16_max')}, "
            f"zf32 vs torch {out.get('zf32_vs_torch_max')}"
        )
        res["frames"][str(d["frame"])] = out
    res["row_stats"] = dict(STATS)
    res["kernel_vs_torch_per_call"] = {f"{k[0]}/{k[1]}": v for k, v in KERR.items()}
    print("counts per chunk:")
    for c in res["counts"]:
        print("  ", c)
    print("row stats (frames run):", json.dumps(res["row_stats"], indent=1))
    print(
        "kernel vs torch per norm call (all calls of the frames run):",
        json.dumps(res["kernel_vs_torch_per_call"], indent=1),
    )
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
