"""Full-policy check of the int16 attention data path on checkpoint -12392's 3 replay frames.

Every attention of SmolVLA (the SigLIP vision tower, the LM prefix, the action expert's self- and
cross-attention) is re-routed through attn_int16.attention, which reproduces the engine's arithmetic
exactly (int16 Q, K, V, P with FP32 per-row scales, exact integer sums, FP32 dequantisation, FP32 softmax);
everything else (the linears, norms, GELU, ...) stays torch FP32, so the error is the attention path's
own. Per frame: predict_action_chunk(d["batch"], noise=d["noise"]) against d["ref_chunk_raw"] (the
cluster's FP32 chunk) and against this laptop's unpatched run. The policy is loaded once; patches go on
the instances (nothing in site-packages is modified).

    set HF_HUB_OFFLINE=1
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\attention\\check_policy.py            # all variants
    ... --variants int16,bf16 --frames 0          # a subset
    ... --dump vectors/calls.pt                    # also save the first call of every site (make_vectors.py)

Variants: orig (LeRobot as loaded: sdpa vision, eager LM), fp32 (explicit FP32 Q K^T / softmax / P V through
the same stacked-GQA code path, padding keys zeroed), int16 (the design), int16_qk / int16_pv (one product
int16, the other FP32), int16_vision / int16_prefix / int16_expert (int16 at one site only), int16_pow2
(power-of-two scales for Q, K, V, P), int16_ropepow2 (power-of-two scales for Q and K only: the RoPE kernel's
ROPE_QM = 0 output), int16_smx (P from the softmax kernel's own arithmetic, smx_emu.py: q in [0, 32767] with
scale 1 / sum q), int16_compact (int16 on the valid keys / queries only: the prefix padding dropped),
bf16 (Q, K, V, P rounded to BF16, FP32 sums in torch's order).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import attn_int16 as A  # noqa: E402

CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"
GATE = 0.05

SITES = ("vision", "prefix", "expert_self", "expert_cross")
VARIANTS = {
    "orig": None,
    "fp32": dict(qk="fp32", pv="fp32"),
    "int16": dict(qk="int16", pv="int16"),
    "int16_qk": dict(qk="int16", pv="fp32"),
    "int16_pv": dict(qk="fp32", pv="int16"),
    "int16_vision": dict(qk="int16", pv="int16", sites=("vision",)),
    "int16_prefix": dict(qk="int16", pv="int16", sites=("prefix",)),
    "int16_expert": dict(qk="int16", pv="int16", sites=("expert_self", "expert_cross")),
    "int16_pow2": dict(qk="int16", pv="int16", pow2=True),
    "int16_ropepow2": dict(qk="int16", pv="int16", qk_pow2=True),
    "int16_smx": dict(qk="int16", pv="int16", p_mode="smx"),
    "int16_compact": dict(qk="int16", pv="int16", compact=True),
    "bf16": dict(qk="bf16", pv="bf16"),
}

STATE = dict(variant=None, prefix_len=None, dump=None, dumped=set(), record=False)
CALLS: list[dict] = []  # per attention call: site, shapes, kernel error
MASKS = dict(checked=0, failed=0)
CMP = dict(calls=0, bit_identical=0, max_abs_diff=0.0)  # int16_compact vs int16 on the valid rows


def site_cfg(site: str):
    v = STATE["variant"]
    if v is None:
        return None
    sites = v.get("sites")
    if sites is not None and site not in sites:
        return dict(qk="fp32", pv="fp32")
    return v


def run_attention(site, q, k, v, scale, mask):
    """q (B,H,Tq,D), k/v (B,Hkv,Tk,D), mask (B,Tq,Tk) bool or None -> (B,H,Tq,D)."""
    cfg = site_cfg(site)
    if mask is not None:
        _, _, ok = A.rank_descriptor(mask)
        MASKS["checked"] += 1
        MASKS["failed"] += 0 if ok else 1
    out = A.attention(q, k, v, scale, mask, cfg["qk"], cfg["pv"], cfg.get("pow2", False),
                      cfg.get("qk_pow2", False) and site != "vision", cfg.get("p_mode", "quant"))
    if cfg.get("compact") and mask is not None:  # the same on the valid keys / queries only
        assert q.shape[0] == 1
        kv, qv = mask[0].any(0), mask[0].any(1)
        oc = A.attention(q[:, :, qv], k[:, :, kv], v[:, :, kv], scale, mask[:, qv][:, :, kv], cfg["qk"], cfg["pv"])
        CMP["calls"] += 1
        CMP["bit_identical"] += int(torch.equal(oc, out[:, :, qv]))
        CMP["max_abs_diff"] = max(CMP["max_abs_diff"], float((oc - out[:, :, qv]).abs().max()))
        out = out.clone()
        out[:, :, qv] = oc
    rec = dict(site=site, B=q.shape[0], H=q.shape[1], Hkv=k.shape[1], Tq=q.shape[2], Tk=k.shape[2], D=q.shape[3])
    if mask is not None:
        rec["valid_keys"] = int(mask.any(1).sum(-1).max())
        rec["valid_queries"] = int(mask.any(2).sum(-1).max())
    if STATE["record"] and (cfg["qk"] != "fp32" or cfg["pv"] != "fp32"):
        ref = A.attention(q, k, v, scale, mask, "fp32", "fp32")
        if mask is not None:  # rows no key is allowed to: not used by the model
            rv = mask.any(-1)[:, None, :, None]
            d, r = (out - ref) * rv, ref * rv
        else:
            d, r = out - ref, ref
        rec["max_abs_err"] = float(d.abs().max())
        rec["max_abs_ref"] = float(r.abs().max())
        rec["rms_err"] = float(d.pow(2).mean().sqrt())
        rec["rms_ref"] = float(r.pow(2).mean().sqrt())
    CALLS.append(rec)
    dump = STATE["dump"]
    if dump is not None and site not in STATE["dumped"]:
        STATE["dumped"].add(site)
        dump[site] = dict(q=q.clone(), k=k.clone(), v=v.clone(), scale=scale,
                          mask=None if mask is None else mask.clone())
    return out


def make_vision_forward(orig):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if STATE["variant"] is None:
            return orig(hidden_states, attention_mask, **kwargs)
        assert attention_mask is None, "SmolVLA runs SigLIP without a patch mask"
        shp = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shp).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shp).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shp).transpose(1, 2)
        out = run_attention("vision", q, k, v, self.scale, None)
        out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.out_proj(out), None

    return forward


def make_lm_attention(orig):
    def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        if STATE["variant"] is None:
            return orig(attention_mask, batch_size, head_dim, query_states, key_states, value_states)
        Tq, Tk = query_states.shape[1], key_states.shape[1]
        if Tq == Tk:
            site = "prefix"
            STATE["prefix_len"] = Tq
        elif STATE["prefix_len"] is not None and Tk == STATE["prefix_len"] + Tq:
            site = "expert_self"
        else:
            site = "expert_cross"
        q = query_states.float().transpose(1, 2)  # (B, H, Tq, D)
        k = key_states.float().transpose(1, 2)  # (B, Hkv, Tk, D): not repeated
        v = value_states.float().transpose(1, 2)
        out = run_attention(site, q, k, v, head_dim**-0.5, attention_mask)
        return out.permute(0, 2, 1, 3).reshape(batch_size, -1, q.shape[1] * head_dim)

    return eager_attention_forward


def install(policy) -> int:
    n = 0
    for _, mod in policy.named_modules():
        if type(mod).__name__ == "SmolVLMVisionAttention":
            mod.forward = types.MethodType(make_vision_forward(mod.forward), mod)
            n += 1
    vwe = policy.model.vlm_with_expert
    vwe.eager_attention_forward = types.MethodType(make_lm_attention(vwe.eager_attention_forward), vwe)
    return n


def run_frame(policy, d):
    batch = copy.deepcopy(d["batch"])
    policy.reset()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch, noise=d["noise"].clone())
    return chunk.float()


def summarize_calls(calls):
    out = {}
    for s in SITES:
        cs = [c for c in calls if c["site"] == s]
        if not cs:
            continue
        e = dict(calls=len(cs), shape={k: cs[0][k] for k in ("B", "H", "Hkv", "Tq", "Tk", "D")})
        if "valid_keys" in cs[0]:
            e["valid_keys"] = cs[0]["valid_keys"]
            e["valid_queries"] = cs[0]["valid_queries"]
        errs = [c for c in cs if "max_abs_err" in c]
        if errs:
            e["max_abs_err"] = max(c["max_abs_err"] for c in errs)
            e["max_rel_err"] = max(c["max_abs_err"] / max(c["max_abs_ref"], 1e-30) for c in errs)
            e["rms_rel_err"] = max(c["rms_err"] / max(c["rms_ref"], 1e-30) for c in errs)
        out[s] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dump", default=None, help="save the first call of every site (int16 variant, frame 0)")
    ap.add_argument("--out", default=str(HERE / "results" / "policy_check.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to("cpu").eval().float()
    n_vis = install(policy)
    print(f"loaded {CKPT.name} in {time.perf_counter() - t0:.1f} s; patched {n_vis} vision attention modules",
          flush=True)
    files = sorted(BATCHES.glob("frame_*.pt"))
    frames = [torch.load(files[int(i)], weights_only=False) for i in args.frames.split(",")]
    results = dict(ckpt=CKPT.name, frames=[int(d["frame"]) for d in frames], variants={})
    laptop_ref = {}
    for name in args.variants.split(","):
        STATE["variant"] = VARIANTS[name]
        rows = []
        for d in frames:
            CALLS.clear()
            STATE["record"] = name.startswith("int16") or name == "bf16"
            if args.dump and name == "int16" and d is frames[0]:
                STATE["dump"] = {}
            t1 = time.perf_counter()
            chunk = run_frame(policy, d)
            dt = time.perf_counter() - t1
            if STATE["dump"] is not None:
                Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
                torch.save(STATE["dump"], args.dump)
                print(f"dumped sites {sorted(STATE['dump'])} to {args.dump}", flush=True)
                STATE["dump"] = None
            ref = d["ref_chunk_raw"].float()
            err = (chunk - ref).abs()
            row = dict(frame=int(d["frame"]), max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                       seconds=round(dt, 1), finite=bool(torch.isfinite(chunk).all()))
            if name == "orig":
                laptop_ref[int(d["frame"])] = chunk
            if int(d["frame"]) in laptop_ref:
                row["max_abs_err_vs_laptop_orig"] = float((chunk - laptop_ref[int(d["frame"])]).abs().max())
            if CALLS:
                row["sites"] = summarize_calls(CALLS)
            rows.append(row)
            print(json.dumps(dict(variant=name, **{k: v for k, v in row.items() if k != "sites"})), flush=True)
            if "sites" in row and name in ("int16", "bf16", "int16_pow2", "int16_smx"):
                print("   ", json.dumps(row["sites"]), flush=True)
        results["variants"][name] = dict(cfg=VARIANTS[name], frames=rows,
                                         worst_max_abs_err=max(r["max_abs_err"] for r in rows))
    results["mask_rank_descriptor"] = dict(MASKS)
    results["compact_vs_full"] = dict(CMP)
    print("compacted vs full attention (valid query rows):", CMP, flush=True)
    print("mask rank descriptor checks:", MASKS, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
