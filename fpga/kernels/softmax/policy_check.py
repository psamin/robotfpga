r"""Full-policy effect of the SOFTMAX unit (integration contract 1.0) on SmolVLA -12392 (3 replay frames), plus
the softmax inventory and the unit's time model.

Every attention softmax of the policy (SigLIP vision tower, VLM prefix, action expert self- and
cross-attention)
is replaced by the unit's bit-exact torch emulation (smx_emu.py); everything else stays FP32 as loaded. Modes:
  orig      the policy as loaded (sdpa vision attention, LeRobot eager attention)                 control
  explicit  explicit Q K^T, torch.softmax, P V in FP32 (the code path of the unit modes)          control
  smx       the contract's arithmetic: the scores enter already x 1/8 (the 1/8 rides in Q's stored scale,
            p4 = -3, CONTRACT.md 2.4), p0 = 1.0 so c = fl32(log2 e); masks as int32 rank vectors
            (0 <= E[k] <= D[n]); P = q * scale with q in [0, 32767], scale = RN(1 / sum q), fully masked rows
            q = 0 with scale 1.0; P V in FP32
  smx_p0    the same unit on the raw scores with p0 = 1/8 (the pre-contract way): must give the same chunk
            bit for bit (1/8 is exact, and fix() only moves the exponent); checked, not assumed
  smx_pv16  smx, and P V as the contract's GEMM computes it from the unit's output (CONTRACT.md 2.3, 5):
            X = q (int16 P rows), s_tok = the unit's 1/sum scale, W = V^T quantized per (head, channel) over
            the keys by contract.quant_rows, acc exact, y = ftz(ftz(fl32(acc) * s_row) * s_tok); the expert
            self-attention as two GEMMs (prefix keys, then suffix keys with F_ACC: + C, one FP32 add), each
            with its own V^T blob and channel scales. Q K^T stays FP32 (the attention kernel's part)
Errors: max |chunk - ref_chunk_raw| (normalized action units, gate 0.05) per frame, and max |chunk -
chunk_explicit|
(the unit's own effect).

    set HF_HUB_OFFLINE=1
    C:\kp\lerobot-venv\Scripts\python.exe fpga\kernels\softmax\policy_check.py [--frames 0,1,2] [--modes ...]
    ... --dump-scores     also writes fpga/kernels/softmax/build/real_scores.pt (real score rows for
    make_vectors.py)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import smx_emu as E  # noqa: E402

SRC_CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
NOVLM_CKPT = (
    ROOT / "build/brief/int16_12392/ckpt_novlm"
)  # same files, load_vlm_weights=false (deploy_smolvla.py)
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"
OUT = HERE / "build"
BIG_NEG = torch.finfo(torch.float32).min

STATE = dict(mode="orig", phase="vision", log=None, dump=None)


def smx_softmax(scores: torch.Tensor, temp: float, mask, where: str) -> torch.Tensor:
    """scores (B,H,Tq,Tk) raw FP32 Q K^T; mask None or bool broadcastable to it (True = may attend)."""
    B, H, Tq, Tk = scores.shape
    log = STATE["log"]
    if log is not None:
        log.append(
            dict(
                site=where,
                phase=STATE["phase"],
                B=B,
                H=H,
                Hkv=STATE.get("hkv", H),
                Tq=Tq,
                Tk=Tk,
                masked=mask is not None,
            )
        )
    valid = None if mask is None else mask.expand(B, H, Tq, Tk)
    dump = STATE["dump"]
    if dump is not None and where not in dump:
        dump[where] = dict(
            scores=scores[0, :2].reshape(-1, Tk).clone(),
            rph=Tq,
            c=temp * E.LOG2E,
            valid=None if valid is None else valid[0, :2].reshape(-1, Tk).clone(),
        )
    mode = STATE["mode"]
    if mode == "explicit":
        s = scores * temp
        if mask is not None:
            s = torch.where(mask, s, BIG_NEG)
        return torch.softmax(s, dim=-1)
    if mode in ("smx", "smx_pv16"):  # contract: x = scores * 1/8 exactly in FP32 (Q's stored scale), p0 = 1
        x, p0 = (scores * torch.tensor(temp, dtype=torch.float32)).reshape(-1, Tk), 1.0
    elif mode == "smx_p0":
        x, p0 = scores.reshape(-1, Tk), temp
    else:
        raise ValueError(mode)
    c = E.c_from_mult(p0)
    v = None
    if mask is not None:  # the unit sees int32 rank vectors: E per key, D per row (one pair per batch item)
        m = mask.expand(B, 1, Tq, Tk)[:, 0]
        r = E.ranks_from_mask(m)
        assert r is not None, f"{where}: mask not rank-representable"
        v = E.rank_valid(r[0], r[1])  # (B, Tq, Tk), == m
        assert torch.equal(v, m.bool()), where
        STATE["ranked"] = STATE.get("ranked", 0) + 1
        v = v[:, None].expand(B, H, Tq, Tk).reshape(-1, Tk)
    if mode == "smx_pv16":
        q, sc = E.kernel(x, c, v)
        return q.reshape(B, H, Tq, Tk), sc.reshape(B, H, Tq)
    p = E.softmax_u16(x, c, v)
    return p.reshape(B, H, Tq, Tk)


def _ftz(x: torch.Tensor) -> torch.Tensor:
    """contract.ftz in torch: FP32 denormals to zero, sign kept."""
    u = x.contiguous().view(torch.int32)
    den = (u & 0x7F800000) == 0
    return torch.where(den, u & torch.tensor(-0x80000000, dtype=torch.int32), u).view(torch.float32)


def _pv_gemm(q: torch.Tensor, s_tok: torch.Tensor, v: torch.Tensor, acc_c: torch.Tensor | None = None):
    """One contract GEMM of P V: q (..., Tq, Tk) int, s_tok (..., Tq) FP32, v (..., Tk, 64) FP32. W = V^T with
    one scale per (head, channel) over these Tk keys (contract quantizer); y = ftz(ftz(fl32(acc) s_row) s_tok)
    [+ C, F_ACC]."""
    vt = v.to(torch.float32).transpose(-1, -2).contiguous()  # (..., 64, Tk): blob rows = channels
    a = vt.abs().amax(-1, keepdim=True)
    sv = torch.where(a == 0, torch.ones_like(a), torch.clamp(a, min=1e-30) / torch.tensor(32767.0))
    qv = torch.clamp(torch.round(vt / sv), -32767, 32767)  # torch.round: half to even, as np.rint
    acc = q.to(torch.float64) @ qv.to(torch.float64).transpose(-1, -2)  # exact: |acc| < 2^38
    y = _ftz(_ftz(acc.to(torch.float32)) * _ftz(sv[..., 0][..., None, :]))
    y = _ftz(y * _ftz(s_tok[..., :, None]))
    if acc_c is not None:
        y = _ftz(y + _ftz(acc_c))
    return y


def attn_out(p, v: torch.Tensor, where: str) -> torch.Tensor:
    """P V for every mode: FP32 matmul, or (smx_pv16) the contract's GEMMs on the unit's (q, scale)."""
    if STATE["mode"] != "smx_pv16":
        return torch.matmul(p.to(v.dtype), v)
    q, sc = p
    if where == "expert_self":  # prefix keys, then the suffix keys accumulated into the same rows (F_ACC)
        n0 = PREFIX_LEN[0]
        y = _pv_gemm(q[..., :n0], sc, v[..., :n0, :])
        return _pv_gemm(q[..., n0:], sc, v[..., n0:, :], acc_c=y).to(v.dtype)
    return _pv_gemm(q, sc, v).to(v.dtype)


def make_vision_forward(orig):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if STATE["mode"] == "orig":
            if STATE["log"] is not None:
                B, T, _ = hidden_states.shape
                STATE["log"].append(
                    dict(site="vision", phase="vision", B=B, H=self.num_heads, Tq=T, Tk=T, masked=False)
                )
            return orig(hidden_states, attention_mask, **kwargs)
        assert attention_mask is None, "SmolVLA runs SigLIP without a patch mask"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        p = smx_softmax(q @ k.transpose(-1, -2), self.scale, None, "vision")
        out = attn_out(p, v, "vision").transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
        return self.out_proj(out), None

    return forward


def make_lm_attention(orig):
    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        if STATE["mode"] == "orig":
            if STATE["log"] is not None:
                STATE["log"].append(
                    dict(
                        site=lm_site(query_states, key_states),
                        phase=STATE["phase"],
                        B=batch_size,
                        H=self.num_attention_heads,
                        Hkv=self.num_key_value_heads,
                        Tq=query_states.shape[1],
                        Tk=key_states.shape[1],
                        masked=True,
                    )
                )
            return orig(attention_mask, batch_size, head_dim, query_states, key_states, value_states)
        H, Hkv = self.num_attention_heads, self.num_key_value_heads
        STATE["hkv"] = Hkv
        g = H // Hkv
        Tk = key_states.shape[1]
        k = (
            key_states[:, :, :, None, :]
            .expand(batch_size, Tk, Hkv, g, head_dim)
            .reshape(batch_size, Tk, H, head_dim)
        )
        v = (
            value_states[:, :, :, None, :]
            .expand(batch_size, Tk, Hkv, g, head_dim)
            .reshape(batch_size, Tk, H, head_dim)
        )
        q = query_states.to(torch.float32).transpose(1, 2)
        k = k.to(torch.float32).transpose(1, 2)
        v = v.transpose(1, 2)
        site = lm_site(query_states, key_states)
        p = smx_softmax(q @ k.transpose(-1, -2), head_dim**-0.5, attention_mask[:, None, :, :], site)
        out = attn_out(p, v, site)
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
    """Softmax calls, rows and elements per site of the LeRobot forward (uncompacted: the prefix is 81 tokens
    here, 45 in the contract's list). The unit's time is smx_time.py's, on the contract list itself."""
    sites = defaultdict(lambda: dict(calls=0, rows=0, elems=0, shapes=set()))
    for e in log:
        s = sites[e["site"]]
        H, Hkv, Tq, Tk = e["H"], e.get("Hkv", e["H"]), e["Tq"], e["Tk"]
        rows = e["B"] * H * Tq
        s["calls"] += 1
        s["rows"] += rows
        s["elems"] += rows * Tk
        s["shapes"].add((e["B"], H, Hkv, Tq, Tk))
    out = {
        k: dict(calls=v["calls"], rows=v["rows"], elems=v["elems"], shapes=sorted(v["shapes"]))
        for k, v in sites.items()
    }
    out["total"] = {k: sum(v[k] for v in sites.values()) for k in ("calls", "rows", "elems")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--modes", default="orig,explicit,smx,smx_p0,smx_pv16")
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
    print(
        f"loaded {ckpt} in {time.perf_counter() - t0:.1f} s;"
        f" {install(policy)} vision attention modules patched",
        flush=True,
    )
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
            if m == "smx_pv16" and "smx" in chunks:
                r["max_abs_vs_smx"] = float((ch - chunks["smx"]).abs().max())
            if m == "smx_p0" and "smx" in chunks:
                r["bit_equal_to_smx"] = bool(torch.equal(ch, chunks["smx"]))
            res["modes"].setdefault(m, {})[int(fr["frame"])] = r
            print(
                f"frame {int(fr['frame']):5d} {m:9s}: max|chunk-ref| {r['max_abs_err']:.3e}"
                + f" mean {r['mean_abs_err']:.3e}"
                + (f"  vs explicit {r['max_abs_vs_explicit']:.3e}" if "max_abs_vs_explicit" in r else "")
                + (f"  bit-equal to smx: {r['bit_equal_to_smx']}" if "bit_equal_to_smx" in r else "")
                + (f"  vs smx {r['max_abs_vs_smx']:.3e}" if "max_abs_vs_smx" in r else "")
                + f"  ({dt:.1f} s)",
                flush=True,
            )
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
