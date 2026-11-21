"""Full-policy effect of the gelu_silu kernel on checkpoint -12392 and its 3 replay frames.

The kernel's exact arithmetic (gs_emu, bit-identical to the HLS C model) replaces SigLIP's tanh-GELU and the LM /
expert silu(gate) * up, and its int16 output + per-token scale feeds fc2 / down_proj. Variants (per frame, max and
mean |chunk - ref_chunk_raw| in normalized action units; gate 0.05):

  F0        FP32 linears, torch activations                                (control: laptop vs cluster reference)
  F2        FP32 linears, torch activations, fc2/down inputs int16 per-token absmax (ideal quantiser, torch order)
  F1_nsbN   FP32 linears, kernel activations: q * s into fc2/down          (F1 - F2 = the kernel's own arithmetic)
            (all three uses: SigLIP GELU mode 0, LM/expert silu(gate) * up mode 1, action-time MLP silu mode 2)
  I0        int16 engine emulation (int16 W per row x int16 X per token, exact sums; the 21 FP32-stored layers FP32),
            torch activations                                              (the engine alone)
  I1_nsbN   I0 with the kernel: its q and s go straight into fc2/down      (the proposed arithmetic)

    set HF_HUB_OFFLINE=1
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\gelu_silu\\policy_check.py [--variants F0,I1_nsb9] [--frames 0]
    ... --dump build/gelu_silu/real9 --nsb 9      # also write real-activation test vectors for tb_gelu_silu
Writes one JSON line per (variant, frame) to stdout and the summary to build/gelu_silu/policy_check.json.
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
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_emu  # noqa: E402

ROOT = HERE.parents[2]
CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"
OUT = ROOT / "build/gelu_silu"
TINY = 1e-30

CFG = dict(linear="fp32", act="torch", nsb=9)
STATS: dict = {}
DUMP: dict = {}


def quant_sym(x: torch.Tensor, qmax: int = 32767):
    s = x.abs().amax(-1, keepdim=True).clamp_min(TINY) / qmax
    return torch.round(x / s).clamp_(-qmax, qmax), s


class Lin:
    """Replacement forward of one nn.Linear: FP32 as loaded, or the int16 engine (exact integer sums)."""

    def __init__(self, name: str, mod: nn.Linear, keep_fp32: bool):
        self.name, self.mod, self.keep = name, mod, keep_fp32
        self.w = mod.weight.detach()
        self.b = None if mod.bias is None else mod.bias.detach()
        self.wq = None

    def prep(self):
        if self.wq is None:
            q, s = quant_sym(self.w)
            self.wq, self.sw = q.double(), s.squeeze(1).double()
        return self.wq, self.sw

    def int16_mode(self) -> bool:
        return CFG["linear"] == "int16" and not self.keep

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.int16_mode():
            return F.linear(x, self.w, self.b)
        xq, sx = quant_sym(x)
        return self.preq(xq, sx)

    def preq(self, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Already-quantised input (q integer-valued, s per token): the engine's acc * s_row * s_tok."""
        shape = q.shape
        if not self.int16_mode():  # FP32 linear fed the dequantised kernel output
            return F.linear((q.to(torch.float32) * s), self.w, self.b)
        wq, sw = self.prep()
        acc = q.reshape(-1, shape[-1]).double() @ wq.t()
        y = (acc * s.reshape(-1, 1).double() * sw[None, :]).float()
        if self.b is not None:
            y = y + self.b
        return y.reshape(*shape[:-1], wq.shape[0])


def note(kind: str, x: torch.Tensor, u: torch.Tensor | None, q: torch.Tensor, s: torch.Tensor, yb: torch.Tensor):
    st = STATS.setdefault(kind, dict(calls=0, rows=[], K=int(x.shape[-1]), elems=0, lsb_max=0.0, lsb_sq=0.0,
                                     kern_lsb_max=0.0, kern_lsb_sq=0.0, n=0, amax_min=1e30, amax_max=0.0))
    st["calls"] += 1
    rows = x.numel() // x.shape[-1]
    st["rows"].append(rows)
    st["elems"] += x.numel()
    ref = gs_emu.act_exact(x, u, {"gelu": 0, "silu1": 2}.get(kind, 1))
    sd = s.double()
    ok = sd > 0
    e = ((q.double() * sd - ref).abs() / sd.clamp_min(1e-300))[ok.expand_as(ref)]
    ek = ((gs_emu.b2f(yb).double() - ref).abs() / sd.clamp_min(1e-300))[ok.expand_as(ref)]
    st["lsb_max"] = max(st["lsb_max"], float(e.max()))
    st["lsb_sq"] += float((e**2).sum())
    st["kern_lsb_max"] = max(st["kern_lsb_max"], float(ek.max()))
    st["kern_lsb_sq"] += float((ek**2).sum())
    st["n"] += int(e.numel())
    am = sd * 32767
    st["amax_min"] = min(st["amax_min"], float(am.min()))
    st["amax_max"] = max(st["amax_max"], float(am.max()))


def make_vision_mlp(mod, lin1: Lin, lin2: Lin, idx: int):
    def forward(self, h):
        a = lin1(h)
        if CFG["act"] == "torch":
            return lin2(self.activation_fn(a))
        if CFG["act"] == "torch_q":
            q, s = quant_sym(self.activation_fn(a))
            return lin2.preq(q, s)
        q, s, yb = gs_emu.kernel(a, None, 0, CFG["nsb"])
        note("gelu", a, None, q, s, yb)
        if CFG.get("dump") and idx == 6 and "gelu" not in DUMP:
            DUMP["gelu"] = (a.reshape(-1, a.shape[-1]).clone(), None)
        return lin2.preq(q, s)

    return forward


def make_llama_mlp(mod, lg: Lin, lu: Lin, ld: Lin, kind: str, idx: int):
    def forward(self, x):
        g, u = lg(x), lu(x)
        if CFG["act"] == "torch":
            return ld(self.act_fn(g) * u)
        if CFG["act"] == "torch_q":
            q, s = quant_sym(self.act_fn(g) * u)
            return ld.preq(q, s)
        q, s, yb = gs_emu.kernel(g, u, 1, CFG["nsb"])
        note(kind, g, u, q, s, yb)
        if CFG.get("dump") and idx == 3 and kind not in DUMP:
            DUMP[kind] = (g.reshape(-1, g.shape[-1]).clone(), u.reshape(-1, u.shape[-1]).clone())
        return ld.preq(q, s)

    return forward


def make_silu_alone():
    """F.silu of modeling_smolvla (action_time_mlp: silu(action_time_mlp_in(.)) -> action_time_mlp_out), mode 2.
    The kernel's q * s goes to the next Lin, which (int16 mode) re-quantizes it: the same q and s up to one ulp of
    s (q * s / s' rounds back to q since |q| < 2^15)."""
    silu = F.silu

    def f(x, inplace=False):
        if CFG["act"] == "torch":
            return silu(x)
        if CFG["act"] == "torch_q":
            q, s = quant_sym(silu(x))
            return q * s
        q, s, yb = gs_emu.kernel(x, None, 2, CFG["nsb"])
        note("silu1", x, None, q, s, yb)
        if CFG.get("dump") and "silu1" not in DUMP:
            DUMP["silu1"] = (x.reshape(-1, x.shape[-1]).clone(), None)
        return q.to(torch.float32) * s

    return f


def fp32_stored(ckpt: Path) -> set[str]:
    from safetensors import safe_open

    names = set()
    with safe_open(str(ckpt / "model.safetensors"), "pt") as sf:
        for k in sf.keys():
            if k.endswith(".weight") and sf.get_slice(k).get_dtype() == "F32":
                names.add(k[: -len(".weight")])
    return names


def install(policy, ckpt: Path) -> None:
    keep = fp32_stored(ckpt)
    lins: dict[str, Lin] = {}
    n_keep = 0
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            k = any(name.endswith(kn) or kn.endswith(name) for kn in keep)
            n_keep += k
            lp = Lin(name, mod, k)
            lins[name] = lp
            mod.forward = types.MethodType(lambda self, x, _lp=lp: _lp(x), mod)
    nv = nl = ne = 0
    for name, mod in policy.named_modules():
        tn = type(mod).__name__
        if tn == "SmolVLMVisionMLP":
            mod.forward = types.MethodType(make_vision_mlp(mod, lins[name + ".fc1"], lins[name + ".fc2"], nv), mod)
            nv += 1
        elif tn == "LlamaMLP":
            assert type(mod.act_fn).__name__ == "SiLUActivation" or "silu" in str(mod.act_fn).lower(), mod.act_fn
            kind = "expert" if "lm_expert" in name else "lm"
            idx = ne if kind == "expert" else nl
            mod.forward = types.MethodType(
                make_llama_mlp(mod, lins[name + ".gate_proj"], lins[name + ".up_proj"], lins[name + ".down_proj"], kind, idx),
                mod,
            )
            if kind == "expert":
                ne += 1
            else:
                nl += 1
    import lerobot.policies.smolvla.modeling_smolvla as ms

    fp = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    fp.silu = make_silu_alone()
    ms.F = fp  # only modeling_smolvla's own F.silu (the action-time MLP) is redirected
    print(f"patched {len(lins)} linears ({n_keep} kept FP32 as stored), {nv} vision MLPs, {nl} LM MLPs, {ne} expert MLPs",
          file=sys.stderr, flush=True)


VARIANTS = {
    "F0": dict(linear="fp32", act="torch"),
    "F2": dict(linear="fp32", act="torch_q"),
    "F1_nsb9": dict(linear="fp32", act="kernel", nsb=9),
    "F1_nsb10": dict(linear="fp32", act="kernel", nsb=10),
    "I0": dict(linear="int16", act="torch_q"),
    "I1_nsb9": dict(linear="int16", act="kernel", nsb=9),
    "I1_nsb10": dict(linear="int16", act="kernel", nsb=10),
}


def run(policy, frame: dict) -> torch.Tensor:
    policy.reset()
    with torch.no_grad():
        return policy.predict_action_chunk(copy.deepcopy(frame["batch"]), noise=frame["noise"].clone()).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--nsb", type=int, default=9, help="table size for --dump")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to("cpu").eval().float()
    install(policy, CKPT)
    print(f"loaded in {time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    files = sorted(BATCHES.glob("frame_*.pt"))
    frames = [torch.load(files[int(i)], weights_only=False) for i in a.frames.split(",")]

    if a.dump:
        sys.path.insert(0, str(HERE))
        from make_vectors import write_case

        CFG.update(linear="fp32", act="kernel", nsb=a.nsb, dump=True)
        run(policy, frames[0])
        for kind, (x, u) in DUMP.items():
            write_case(a.dump / f"real_{kind}", {"gelu": 0, "silu1": 2}.get(kind, 1), x, u, a.nsb)
            print(f"dumped real_{kind}: {tuple(x.shape)}", file=sys.stderr)
        CFG["dump"] = False
        return

    results = []
    chunks: dict = {}
    for vn in a.variants.split(","):
        v = VARIANTS[vn]
        CFG.update(linear=v["linear"], act=v["act"], nsb=v.get("nsb", 9))
        for fr in frames:
            STATS.clear()
            t1 = time.perf_counter()
            ch = run(policy, fr)
            dt = time.perf_counter() - t1
            ref = fr["ref_chunk_raw"].float()
            err = (ch - ref).abs()
            chunks[(vn, int(fr["frame"]))] = ch
            r = dict(variant=vn, frame=int(fr["frame"]), max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                     seconds=round(dt, 1))
            for base in ("F2", "I0"):
                if vn[0] == base[0] and (base, r["frame"]) in chunks and vn != base:
                    r[f"max_vs_{base}"] = float((ch - chunks[(base, r["frame"])]).abs().max())
            if STATS:
                r["kernel"] = {
                    k: dict(calls=s["calls"], K=s["K"], elems=s["elems"], rows_per_call=sorted(set(s["rows"])),
                            q_err_lsb_max=round(s["lsb_max"], 4), q_err_lsb_rms=round((s["lsb_sq"] / s["n"]) ** 0.5, 4),
                            y_err_lsb_max=round(s["kern_lsb_max"], 4),
                            y_err_lsb_rms=round((s["kern_lsb_sq"] / s["n"]) ** 0.5, 5),
                            amax_min=s["amax_min"], amax_max=s["amax_max"])
                    for k, s in STATS.items()
                }
            print(json.dumps(r), flush=True)
            results.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "policy_check.json"
    old = []
    if out.exists():
        old = [r for r in json.loads(out.read_text()) if (r["variant"], r["frame"]) not in {(x["variant"], x["frame"]) for x in results}]
    out.write_text(json.dumps(old + results, indent=1))
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
