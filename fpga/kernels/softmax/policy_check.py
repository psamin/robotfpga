"""Full-policy effect of the FPGA softmax kernel on SmolVLA -12392 (3 replay frames), plus the softmax inventory.

Every attention softmax of the policy (SigLIP vision tower, VLM prefix, action expert self- and cross-attention)
is replaced by the kernel's bit-exact torch emulation (smx_emu.py); everything else stays FP32 as loaded. Modes:
  orig          the policy as loaded (sdpa vision attention, LeRobot eager attention)          control
  explicit      explicit Q K^T, torch.softmax, P V in FP32 (the code path of the kernel modes)  control
  smx_u16       kernel, int16 output: P = q * scale (q in [0, 32767], scale = RN(1 / sum q)), P V in FP32;
                fully masked rows zero (the default, the sequencer contract)
  smx_u16_unif  the same with F_EMPTY_UNIFORM (fully masked rows uniform, as torch)
  smx_f32       kernel, FP32 output (SMX_FP32_OUT=1 build), fully masked rows zero
Each attention's raw scores Q K^T (FP32) go into the kernel with p0 = 1/sqrt(64) (c = fl32(p0 * log2 e)), masks as
the kernel's rank vectors (E per key, D per row; every mask is checked to be rank-representable). Errors:
max |chunk - ref_chunk_raw| (normalized action units, gate 0.05) per frame, and max |chunk - chunk_explicit|
(the kernel's own effect).

    set HF_HUB_OFFLINE=1
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\softmax\\policy_check.py [--frames 0,1,2] [--modes ...]
    ... --dump-scores     also writes fpga/kernels/softmax/build/real_scores.pt (real score rows for make_vectors.py)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import smx_emu as E  # noqa: E402

SRC_CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
NOVLM_CKPT = ROOT / "build/brief/int16_12392/ckpt_novlm"  # same files, load_vlm_weights=false (deploy_smolvla.py)
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"
OUT = HERE / "build"
BIG_NEG = torch.finfo(torch.float32).min

STATE = dict(mode="orig", phase="vision", log=None, dump=None)
CLK = 250e6
BR = 256


def smx_softmax(scores: torch.Tensor, temp: float, mask, where: str) -> torch.Tensor:
    """scores (B,H,Tq,Tk) raw FP32 Q K^T; mask None or bool broadcastable to it (True = may attend)."""
    B, H, Tq, Tk = scores.shape
    log = STATE["log"]
    if log is not None:
        log.append(dict(site=where, phase=STATE["phase"], B=B, H=H, Hkv=STATE.get("hkv", H), Tq=Tq, Tk=Tk,
                        masked=mask is not None))
    valid = None if mask is None else mask.expand(B, H, Tq, Tk)
    dump = STATE["dump"]
    if dump is not None and where not in dump:
        n = min(H, 2) * Tq
        dump[where] = dict(scores=scores[0, :2].reshape(-1, Tk).clone(), rph=Tq, c=temp * E.LOG2E,
                           valid=None if valid is None else valid[0, :2].reshape(-1, Tk).clone())
    mode = STATE["mode"]
    if mode == "explicit":
        s = scores * temp
        if mask is not None:
            s = torch.where(mask, s, BIG_NEG)
        return torch.softmax(s, dim=-1)
    c = E.c_from_mult(temp)
    x = scores.reshape(-1, Tk)
    v = None
    if mask is not None:  # the kernel sees rank vectors: E per key, D per row (one pair per batch item)
        m = mask.expand(B, 1, Tq, Tk)[:, 0]
        r = E.ranks_from_mask(m)
        assert r is not None, f"{where}: mask not rank-representable"
        v = E.rank_valid(r[0], r[1])  # (B, Tq, Tk), == m
        STATE["ranked"] = STATE.get("ranked", 0) + 1
        v = v[:, None].expand(B, H, Tq, Tk).reshape(-1, Tk)
    if mode == "smx_u16":
        p = E.softmax_u16(x, c, v, empty="zero")
    elif mode == "smx_u16_unif":
        p = E.softmax_u16(x, c, v, empty="uniform")
    elif mode == "smx_f32":
        p = E.softmax_f32(x, c, v, empty="zero")
    else:
        raise ValueError(mode)
    return p.reshape(B, H, Tq, Tk)


def make_vision_forward(orig):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if STATE["mode"] == "orig":
            if STATE["log"] is not None:
                B, T, _ = hidden_states.shape
                STATE["log"].append(dict(site="vision", phase="vision", B=B, H=self.num_heads, Tq=T, Tk=T, masked=False))
            return orig(hidden_states, attention_mask, **kwargs)
        assert attention_mask is None, "SmolVLA runs SigLIP without a patch mask"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        p = smx_softmax(q @ k.transpose(-1, -2), self.scale, None, "vision")
        out = (p @ v).transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
        return self.out_proj(out), None

    return forward


def make_lm_attention(orig):
    def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        if STATE["mode"] == "orig":
            if STATE["log"] is not None:
                STATE["log"].append(dict(site=lm_site(query_states, key_states), phase=STATE["phase"], B=batch_size,
                                         H=self.num_attention_heads, Hkv=self.num_key_value_heads,
                                         Tq=query_states.shape[1],
                                         Tk=key_states.shape[1], masked=True))
            return orig(attention_mask, batch_size, head_dim, query_states, key_states, value_states)
        H, Hkv = self.num_attention_heads, self.num_key_value_heads
        STATE["hkv"] = Hkv
        g = H // Hkv
        Tk = key_states.shape[1]
        k = key_states[:, :, :, None, :].expand(batch_size, Tk, Hkv, g, head_dim).reshape(batch_size, Tk, H, head_dim)
        v = value_states[:, :, :, None, :].expand(batch_size, Tk, Hkv, g, head_dim).reshape(batch_size, Tk, H, head_dim)
        q = query_states.to(torch.float32).transpose(1, 2)
        k = k.to(torch.float32).transpose(1, 2)
        v = v.transpose(1, 2)
        p = smx_softmax(q @ k.transpose(-1, -2), head_dim**-0.5, attention_mask[:, None, :, :],
                        lm_site(query_states, key_states))
        out = torch.matmul(p.to(v.dtype), v)
        return out.permute(0, 2, 1, 3).reshape(batch_size, -1, H * head_dim)

    return eager_attention_forward


def lm_site(qs, ks):
    Tq, Tk = qs.shape[1], ks.shape[1]
    if STATE["phase"] == "prefill":
        return "lm_prefix"
    return "expert_self" if Tk > Tq and Tk != PREFIX_LEN[0] else "expert_cross"


PREFIX_LEN = [None]


def install(policy):
    n = 0
    for _, mod in policy.named_modules():
        if type(mod).__name__ == "SmolVLMVisionAttention":
            mod.forward = types.MethodType(make_vision_forward(mod.forward), mod)
            n += 1
    vwe = policy.model.vlm_with_expert
    vwe.eager_attention_forward = types.MethodType(make_lm_attention(vwe.eager_attention_forward), vwe)
    # phase tracking: vision tower, prefix prefill, denoising steps
    model = policy.model
    orig_embed_image = vwe.embed_image

    def embed_image(*a, **k):
        prev = STATE["phase"]
        STATE["phase"] = "vision"
        try:
            return orig_embed_image(*a, **k)
        finally:
            STATE["phase"] = prev

    vwe.embed_image = embed_image
    orig_forward = vwe.forward

    def forward(*a, **k):
        embeds = k.get("inputs_embeds")
        if embeds is not None and embeds[0] is not None and (len(embeds) < 2 or embeds[1] is None):
            STATE["phase"] = "prefill"
            PREFIX_LEN[0] = embeds[0].shape[1]
        else:
            STATE["phase"] = "step"
        return orig_forward(*a, **k)

    vwe.forward = forward
    return n


def run(policy, frame, mode):
    STATE["mode"] = mode
    policy.reset()
    batch = copy.deepcopy(frame["batch"])
    t0 = time.perf_counter()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch, noise=frame["noise"].clone())
    return chunk.float(), time.perf_counter() - t0


def inventory(log):
    """Rows / elements per site and the kernel's cycle estimates, both layouts (README.md "Time"):
    rows   one call per attention invocation: B = H items of Tq rows (vision: one item of H * Tq rows, as the
           sequencer batches it), a_pitch = ceil(Tk / 4) words; both passes stream one word per cycle in parallel,
           pass 2 one row behind pass 1; rows shorter than 13 words are paced by the divider.
    engine items = KV heads, N = (H / Hkv) * Tq stacked rows, Np = pad32(N), Mk = pad32(Tk): G = Mk / 32 groups,
           Np * Mk / 4 words per item; pass 2 starts when pass 1 reaches the item's last group, so the call takes
           (items + (G - 1) / G) item times.
    Both: ~150 cycles per item (loop restart + DDR latency) and ~300 per call."""
    sites = defaultdict(lambda: dict(calls=0, rows=0, elems=0, words_in=0, cyc_rows=0, cyc_engine=0,
                                     words_engine=0, shapes=set()))
    for e in log:
        s = sites[e["site"]]
        H, Hkv, Tq, Tk = e["H"], e.get("Hkv", e["H"]), e["Tq"], e["Tk"]
        rows = e["B"] * H * Tq
        L4 = (Tk + 3) // 4
        s["calls"] += 1
        s["rows"] += rows
        s["elems"] += rows * Tk
        s["words_in"] += rows * L4
        items = e["B"] * (1 if e["site"] == "vision" else H)
        s["cyc_rows"] += rows * max(L4, 13) + L4 + items * 150 + 300
        Np, G = -(-(H // Hkv) * Tq // 32) * 32, -(-Tk // 32)
        iw = Np * G * 8
        nit = e["B"] * Hkv
        s["words_engine"] += nit * iw
        s["cyc_engine"] += int((nit + (G - 1) / G) * iw) + nit * 150 + 300
        s["shapes"].add((e["B"], H, Hkv, Tq, Tk))
    keys = ("calls", "rows", "elems", "words_in", "words_engine", "cyc_rows", "cyc_engine")
    tot = {k: 0 for k in keys}
    out = {}
    for k, s in sites.items():
        out[k] = {kk: s[kk] for kk in keys}
        out[k]["ms_rows"] = round(s["cyc_rows"] / CLK * 1e3, 3)
        out[k]["ms_engine"] = round(s["cyc_engine"] / CLK * 1e3, 3)
        out[k]["shapes(B,H,Hkv,Tq,Tk)"] = sorted(s["shapes"])
        for t in keys:
            tot[t] += s[t]
    tot["ms_rows"] = round(tot["cyc_rows"] / CLK * 1e3, 3)
    tot["ms_engine"] = round(tot["cyc_engine"] / CLK * 1e3, 3)
    tot["ms_ideal_stream_rows"] = round(tot["words_in"] / CLK * 1e3, 3)
    out["total"] = tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--modes", default="orig,explicit,smx_u16,smx_u16_unif,smx_f32")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dump-scores", action="store_true")
    ap.add_argument("--out", default=str(OUT / "policy_check.json"))
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if a.threads:
        torch.set_num_threads(a.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ckpt = NOVLM_CKPT if NOVLM_CKPT.exists() else SRC_CKPT
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to("cpu").eval().float()
    print(f"loaded {ckpt} in {time.perf_counter() - t0:.1f} s; {install(policy)} vision attention modules patched",
          flush=True)
    files = sorted(BATCHES.glob("frame_*.pt"))
    frames = [torch.load(files[int(i)], weights_only=False) for i in a.frames.split(",")]
    modes = a.modes.split(",")
    res = dict(ckpt=str(ckpt), frames=[int(f["frame"]) for f in frames], modes={}, torch=torch.__version__)
    OUT.mkdir(parents=True, exist_ok=True)
    for fi, fr in enumerate(frames):
        ref = fr["ref_chunk_raw"].float()
        chunks = {}
        for m in modes:
            STATE["log"] = [] if fi == 0 and m == modes[0] else None
            STATE["dump"] = {} if (a.dump_scores and fi == 0 and m == "explicit") else None
            ch, dt = run(policy, fr, m)
            chunks[m] = ch
            if STATE["log"] is not None:
                res["inventory"] = inventory(STATE["log"])
                res["inventory_source_mode"] = m
            if STATE["dump"] is not None:
                torch.save(STATE["dump"], OUT / "real_scores.pt")
                print(f"wrote {OUT / 'real_scores.pt'} ({list(STATE['dump'])})", flush=True)
            err = (ch - ref).abs()
            r = dict(max_abs_err=float(err.max()), mean_abs_err=float(err.mean()), seconds=round(dt, 1))
            if "explicit" in chunks and m != "explicit":
                r["max_abs_vs_explicit"] = float((ch - chunks["explicit"]).abs().max())
            if "orig" in chunks and m != "orig":
                r["max_abs_vs_orig"] = float((ch - chunks["orig"]).abs().max())
            res["modes"].setdefault(m, {})[int(fr["frame"])] = r
            print(f"frame {int(fr['frame']):5d} {m:9s}: max|chunk-ref| {r['max_abs_err']:.3e} mean {r['mean_abs_err']:.3e}"
                  + (f"  vs explicit {r['max_abs_vs_explicit']:.3e}" if "max_abs_vs_explicit" in r else "")
                  + f"  ({dt:.1f} s)", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1, default=list), encoding="utf-8")
    inv = res.get("inventory")
    if inv:
        for k, v in inv.items():
            print(k, v)
    res["masked_calls_rank_checked"] = STATE.get("ranked", 0)
    Path(a.out).write_text(json.dumps(res, indent=1, default=list), encoding="utf-8")
    print(f"masked softmax calls converted to rank vectors and checked: {STATE.get('ranked', 0)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
