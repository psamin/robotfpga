"""Full-policy check of the RoPE kernel arithmetic on checkpoint -12392's 3 replay frames.

Monkeypatches lerobot's apply_rope (module global, so every call site picks it up) with the bit-exact
kernel model of rope_emu.py and runs predict_action_chunk(batch, noise=noise); compares with the
cluster's FP32 reference chunk (ref_chunk_raw) and with this laptop's unpatched torch chunk. One JSON
line per (variant, frame).

Variants
  torch         no patch (laptop FP32 torch vs the cluster reference)
  hook_exact    apply_rope replaced by a call to the original (checks the hooking)
  fmt_contract  exact FP32 RoPE, then Q and K through the contract quantizer per (token, head)
                (contract.quant_rows: s = fl32(max(amax, 1e-30) / 32767), q = rne(fl32(x / s)), zero
                segment s = 1.0): the cost of the int16 attention format itself, not of the kernel; the
                cross k_proj reads the FP32 K
  kernel_f32    every RoPE output = the kernel's FP32 rotation (rope_emu.to_f32: Q1.17 table, block-float
                alignment, round to 24 bits, then flush below 2^-126)
  contract      the deployment of CONTRACT.md (the -12392 list of contract_plan.py), the one this kernel
                targets: Q = ROPE with F_QOUT (the FP32 rotation through the contract quantizer per
                (token, head); p4 = -3 folds the exact 1/8, which torch applies itself here); K = ROPE with
                F_KW (even layers) or ROPE FP32 in place + QUANT_W (odd layers), both the contract
                quantizer per (key, head) of the FP32 rotation; the expert cross-attention k_proj reads the
                odd layers' FP32 rotated K
Not modelled here (other kernels' business): the int16 linears, the int16 attention products, V and
the cross-attention keys (expert k_proj outputs) quantization; they are FP32 torch.

    set HF_HUB_OFFLINE=1
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\rope\\check_policy.py
        [--variants a,b] [--frames 0,1,2]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import rope_emu  # noqa: E402

CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"
VARIANTS = ["torch", "hook_exact", "fmt_contract", "kernel_f32", "contract"]


class Patch:
    """Installs the hooks for one variant; restore() puts lerobot back."""

    def __init__(self, swe, variant: str):
        self.swe, self.variant = swe, variant
        self.orig_rope = swe.apply_rope
        cls = swe.SmolVLMWithExpertModel
        self.cls = cls
        self.orig_attn = cls.forward_attn_layer
        self.orig_cross = cls.forward_cross_attn_layer
        self.state = dict(layer=None, prefix=False, n=0)
        self.f32_k = {}  # layer -> FP32-mode prefix K [B, L, Hkv, 64]
        self.stats = dict(calls=0, pairs=0)
        if variant == "torch":
            return
        swe.apply_rope = self.rope
        me = self

        def attn(model, model_layers, inputs_embeds, layer_idx, *a, **k):
            me.state.update(layer=layer_idx, prefix=inputs_embeds[0] is not None, n=0)
            return me.orig_attn(model, model_layers, inputs_embeds, layer_idx, *a, **k)

        def cross(
            model,
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            use_cache=True,
            past_key_values=None,
        ):
            me.state.update(layer=layer_idx, prefix=False, n=0)
            swap = past_key_values is not None and layer_idx in me.f32_k
            if swap:
                lay = past_key_values.layers[layer_idx]
                keep = lay.keys
                lay.keys = me.f32_k[layer_idx].transpose(1, 2)
            try:
                return me.orig_cross(
                    model,
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
            finally:
                if swap:
                    lay.keys = keep

        cls.forward_attn_layer = attn
        cls.forward_cross_attn_layer = cross

    def restore(self):
        self.swe.apply_rope = self.orig_rope
        self.cls.forward_attn_layer = self.orig_attn
        self.cls.forward_cross_attn_layer = self.orig_cross

    def rope(self, x, positions, max_wavelength=10_000):
        assert max_wavelength == 10_000 and x.shape[-1] == rope_emu.D
        st = self.state
        is_k = st["n"] == 1  # forward_attn_layer: query first, then key
        st["n"] += 1
        self.stats["calls"] += 1
        self.stats["pairs"] += x.shape[1] * x.shape[2]
        v = self.variant
        if v == "hook_exact":
            return self.orig_rope(x, positions, max_wavelength)
        if v == "fmt_contract":
            r = self.orig_rope(x, positions, max_wavelength)
            if is_k and st["prefix"]:
                self.f32_k[st["layer"]] = r  # the cross-attention k_proj reads the FP32 K
            return rope_emu.fmt_contract(r).to(x.dtype)
        f = rope_emu.kernel_from_natural(x, positions, "f32")
        if v == "kernel_f32":
            return f
        assert v == "contract", v
        if is_k and st["prefix"]:
            self.f32_k[st["layer"]] = f
        return rope_emu.fmt_contract(f).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--out", default=str(HERE / "check_policy_results.jsonl"))
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import lerobot.policies.smolvla.smolvlm_with_expert as swe
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to("cpu").eval().float()
    vwe = policy.model.vlm_with_expert
    l0 = vwe.vlm.model.text_model.layers[0].self_attn if hasattr(vwe.vlm.model, "text_model") else None
    e0 = vwe.lm_expert.layers[0].self_attn
    print(
        "q_proj bias (vlm, expert):",
        None if l0 is None else l0.q_proj.bias is not None,
        e0.q_proj.bias is not None,
        flush=True,
    )
    files = sorted(BATCHES.glob("frame_*.pt"))
    frames = [torch.load(files[int(i)], weights_only=False) for i in args.frames.split(",")]
    base = {}
    out = open(args.out, "a")
    for v in args.variants.split(","):
        for fr in frames:
            p = Patch(swe, v)
            t0 = time.perf_counter()
            try:
                policy.reset()
                with torch.no_grad():
                    chunk = policy.predict_action_chunk(
                        copy.deepcopy(fr["batch"]), noise=fr["noise"].clone()
                    ).float()
            finally:
                p.restore()
            dt = time.perf_counter() - t0
            fid = int(fr["frame"])
            if v == "torch":
                base[fid] = chunk
            ref = fr["ref_chunk_raw"].float()
            err = (chunk - ref).abs()
            rec = dict(
                variant=v,
                frame=fid,
                max_abs_err=float(err.max()),
                mean_abs_err=float(err.mean()),
                max_vs_laptop_torch=None if fid not in base else float((chunk - base[fid]).abs().max()),
                rope_calls=p.stats["calls"],
                rope_pairs=p.stats["pairs"],
                seconds=round(dt, 1),
            )
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
