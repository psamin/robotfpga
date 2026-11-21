"""Full-policy effect of the gelu_silu front end on checkpoint -12392 and its 3 replay frames, contract 1.0 arithmetic.

The front end's exact arithmetic (gs_emu, bit-identical to the HLS C model) replaces SigLIP's tanh-GELU, the LM /
expert silu(gate) * up and the action-time F.silu; its y goes through the contract quantizer (contract.quant_rows)
into fc2 / down_proj / action_time_mlp_out. Variants (per frame, max and mean |chunk - ref_chunk_raw| in normalized
action units; gate 0.05):

  F0   FP32 linears, torch activations                                     (control: laptop vs cluster reference)
  F1   FP32 linears, the kernel's q * s into fc2 / down / time_out           (the kernel alone, on FP32 linears)
  C0   contract engine emulation, torch activations quantized by the contract quantizer      (the engine alone)
  C1   C0 with the kernel: its q and s go straight into fc2 / down / time_out  (the proposed arithmetic)

Contract engine emulation (CONTRACT.md 2): every linear on int16, including the 21 FP32-stored layers (design study
(b)); weights and activations per row / per token by the contract quantizer (s = fl32(max(a, 1e-30) / 32767), all-zero
rows 1.0, q = clamp(rne(fl32(x / s)))); exact integer sums; epilogue y = ((fl(acc) * s_row) * s_tok) + bias, one FP32
operation per step with denormals flushed (contract.gemm_out); action_time_mlp_in's time half is the constant FP32 row
W_t t_emb + b added as D (2.4). Not emulated here (the contract owner's golden executor T-P1): int16 attention and
the softmax kernel, the folded sqrt(960) / Euler constants, norm / rope kernels, the trimmed language tokens.

    set HF_HUB_OFFLINE=1
    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\gelu_silu\\policy_check.py [--variants F0,C1] [--frames 0]
    ... --dump build/gelu_silu/c1/real       # write real-activation test vectors (frame 0, C1) for tb_gelu_silu
Writes one JSON line per (variant, frame) to stdout and the summary to build/gelu_silu/c1/policy_check.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import contract  # noqa: E402
import gs_emu  # noqa: E402

ROOT = HERE.parents[2]
CKPT = ROOT / "build/ckpt/smolvla-12392/smolvla-so101_medicine_bottle_pickplace-12392-native"
BATCHES = ROOT / "build/ckpt/smolvla-12392/batches-12392"
OUT = ROOT / "build/gelu_silu/c1"
TINY = torch.tensor(1e-30, dtype=torch.float32)
QMAX = 32767

CFG = dict(linear="fp32", act="torch", nsb=9)
STATS: dict = {}
DUMP: dict = {}


def ftz(t: torch.Tensor) -> torch.Tensor:
    """contract.ftz: FP32 values with a zero exponent field become a zero of the same sign."""
    b = t.contiguous().view(torch.int32)
    return torch.where((b & 0x7F800000) == 0, b & torch.tensor(-(1 << 31), dtype=torch.int32), b).view(torch.float32)


def cquant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract quantizer over the last axis (= contract.quant_rows, checked in main): q (float, integer-valued),
    s (..., 1) FP32."""
    x = x.to(torch.float32)
    a = x.abs().amax(-1, keepdim=True)
    s = torch.where(a == 0, torch.ones_like(a), torch.maximum(a, TINY) / QMAX)
    return torch.round(x / s).clamp_(-QMAX, QMAX), s


def gemm_out(acc: torch.Tensor, s_row: torch.Tensor, s_tok: torch.Tensor, bias, addend=None) -> torch.Tensor:
    """contract.gemm_out in torch: acc (rows, M) float64 exact sums; s_row (M,), s_tok (rows, 1) FP32."""
    y = ftz(ftz(acc.to(torch.float32)) * ftz(s_row)[None, :])
    y = ftz(y * ftz(s_tok))
    if bias is not None:
        y = ftz(y + ftz(bias.to(torch.float32))[None, :])
    if addend is not None:
        y = ftz(y + ftz(addend))
    return y


class Lin:
    """Replacement forward of one nn.Linear: FP32 as loaded, or the contract int16 engine."""

    def __init__(self, name: str, mod: nn.Linear):
        self.name, self.mod = name, mod
        self.w = mod.weight.detach().float()
        self.b = None if mod.bias is None else mod.bias.detach().float()
        self.time_in = name.endswith("action_time_mlp_in")  # [a_emb | t_emb]: the t half is an FP32 D row
        self.wq = None

    def prep(self):
        if self.wq is None:
            w = self.w[:, : self.w.shape[1] // 2] if self.time_in else self.w
            q, s = cquant(w)
            self.wq, self.sw = q.double(), s.squeeze(1)
        return self.wq, self.sw

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if CFG["linear"] != "int16":
            return F.linear(x, self.w, self.b)
        if self.time_in:
            h = x.shape[-1] // 2
            d = F.linear(x[..., h:], self.w[:, h:], self.b)  # W_t t_emb + b: constant FP32 rows (host, at load)
            q, s = cquant(x[..., :h])
            return self.preq(q, s, addend=d, bias=False)
        q, s = cquant(x)
        return self.preq(q, s)

    def preq(self, q: torch.Tensor, s: torch.Tensor, addend=None, bias=True) -> torch.Tensor:
        """Already-quantised input (q integer-valued, s per token)."""
        shape = q.shape
        if CFG["linear"] != "int16":  # FP32 linear fed the dequantised kernel output
            return F.linear(q.to(torch.float32) * s, self.w, self.b)
        wq, sw = self.prep()
        acc = q.reshape(-1, shape[-1]).double() @ wq.t()
        ad = None if addend is None else addend.reshape(-1, wq.shape[0])
        y = gemm_out(acc, sw, s.reshape(-1, 1).to(torch.float32), self.b if bias else None, ad)
        return y.reshape(*shape[:-1], wq.shape[0])


def note(kind: str, x: torch.Tensor, u: torch.Tensor | None, q: torch.Tensor, s: torch.Tensor, yb: torch.Tensor):
    st = STATS.setdefault(kind, dict(calls=0, rows=[], K=int(x.shape[-1]), elems=0, q_max=0.0, q_sq=0.0, y_max=0.0,
                                     y_sq=0.0, n=0))
    st["calls"] += 1
    st["rows"].append(x.numel() // x.shape[-1])
    st["elems"] += x.numel()
    ref = gs_emu.act_exact(x, u, {"gelu": 0, "silu1": 2}.get(kind, 1))
    sd = s.double()
    e = (q.double() * sd - ref).abs() / sd
    ey = (gs_emu.b2f(yb).double() - ref).abs() / sd
    st["q_max"] = max(st["q_max"], float(e.max()))
    st["q_sq"] += float((e**2).sum())
    st["y_max"] = max(st["y_max"], float(ey.max()))
    st["y_sq"] += float((ey**2).sum())
    st["n"] += int(e.numel())


def run_kernel(kind: str, x: torch.Tensor, u: torch.Tensor | None):
    mode = {"gelu": 0, "silu1": 2}.get(kind, 1)
    q, s, yb = gs_emu.kernel(x, u, mode, CFG["nsb"])
    note(kind, x, u, q, s, yb)
    if CFG.get("dump") and kind not in DUMP and (kind != "gelu" or CFG.get("vis_idx") == 6):
        DUMP[kind] = (x.reshape(-1, x.shape[-1]).clone(), None if u is None else u.reshape(-1, u.shape[-1]).clone())
    return q.to(torch.float32), s


def make_vision_mlp(mod, lin1: Lin, lin2: Lin, idx: int):
    def forward(self, h):
        a = lin1(h)
        if CFG["act"] == "torch":
            y = self.activation_fn(a)
            return lin2(y) if CFG["linear"] != "int16" else lin2.preq(*cquant(y))
        CFG["vis_idx"] = idx
        return lin2.preq(*run_kernel("gelu", a, None))

    return forward


def make_llama_mlp(mod, lg: Lin, lu: Lin, ld: Lin, kind: str):
    def forward(self, x):
        g, u = lg(x), lu(x)
        if CFG["act"] == "torch":
            y = self.act_fn(g) * u
            return ld(y) if CFG["linear"] != "int16" else ld.preq(*cquant(y))
        return ld.preq(*run_kernel(kind, g, u))

    return forward


def make_silu_alone():
    """F.silu of modeling_smolvla (action_time_mlp: silu(action_time_mlp_in(.)) -> action_time_mlp_out). The kernel's
    q * s goes to action_time_mlp_out, which (int16) re-quantizes it: q * s re-quantizes to the same q and s up to one
    ulp of s (|q| < 2^15), so q * s is passed as the activation."""
    silu = F.silu

    def f(x, inplace=False):
        if CFG["act"] == "torch":
            return silu(x)
        q, s = run_kernel("silu1", x, None)
        return q * s

    return f


def install(policy) -> None:
    lins: dict[str, Lin] = {}
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            lp = Lin(name, mod)
            lins[name] = lp
            mod.forward = types.MethodType(lambda self, x, _lp=lp: _lp(x), mod)
    nv = nl = ne = 0
    for name, mod in policy.named_modules():
        tn = type(mod).__name__
        if tn == "SmolVLMVisionMLP":
            mod.forward = types.MethodType(make_vision_mlp(mod, lins[name + ".fc1"], lins[name + ".fc2"], nv), mod)
            nv += 1
        elif tn == "LlamaMLP":
            kind = "expert" if "lm_expert" in name else "lm"
            mod.forward = types.MethodType(
                make_llama_mlp(mod, lins[name + ".gate_proj"], lins[name + ".up_proj"], lins[name + ".down_proj"], kind),
                mod,
            )
            ne += kind == "expert"
            nl += kind == "lm"
    import lerobot.policies.smolvla.modeling_smolvla as ms

    fp = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    fp.silu = make_silu_alone()
    ms.F = fp  # only modeling_smolvla's own F.silu (the action-time MLP) is redirected
    nt = sum(lp.time_in for lp in lins.values())
    print(f"patched {len(lins)} linears ({nt} time_in), {nv} vision MLPs, {nl} LM MLPs, {ne} expert MLPs",
          file=sys.stderr, flush=True)


VARIANTS = {
    "F0": dict(linear="fp32", act="torch"),
    "F1": dict(linear="fp32", act="kernel"),
    "C0": dict(linear="int16", act="torch"),
    "C1": dict(linear="int16", act="kernel"),
    "C1n10": dict(linear="int16", act="kernel", nsb=10),  # sensitivity: the 1024-segment table
}


def run(policy, frame: dict) -> torch.Tensor:
    policy.reset()
    with torch.no_grad():
        return policy.predict_action_chunk(copy.deepcopy(frame["batch"]), noise=frame["noise"].clone()).float()


def check_cquant() -> None:
    """cquant (torch) == contract.quant_rows (numpy) on edge rows: the rule this file applies everywhere."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, 960, generator=g) * torch.logspace(-35, 35, 64)[:, None]
    x[0] = 0.0
    x[1] = -0.0
    x[2, :5] = torch.tensor([1e-40, -1e-38, 1e-31, 0.0, 2e-31])
    x[2, 5:] = 0.0
    q, s = cquant(x)
    qc, sc = contract.quant_rows(x.numpy())
    assert np.array_equal(q.numpy().astype(np.int16), qc) and np.array_equal(s[:, 0].numpy(), sc), "cquant != contract"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="F0,F1,C0,C1")
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--nsb", type=int, default=9)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    check_cquant()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to("cpu").eval().float()
    install(policy)
    print(f"loaded in {time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    files = sorted(BATCHES.glob("frame_*.pt"))
    frames = [torch.load(files[int(i)], weights_only=False) for i in a.frames.split(",")]

    if a.dump:
        from make_vectors import write_case

        CFG.update(linear="int16", act="kernel", nsb=a.nsb, dump=True)
        run(policy, frames[0])
        for kind, (x, u) in DUMP.items():
            write_case(a.dump / f"real_{kind}", {"gelu": 0, "silu1": 2}.get(kind, 1), x, u, a.nsb)
            print(f"dumped real_{kind}: {tuple(x.shape)}", file=sys.stderr)
        return

    results = []
    chunks: dict = {}
    for vn in a.variants.split(","):
        v = VARIANTS[vn]
        CFG.update(linear=v["linear"], act=v["act"], nsb=v.get("nsb", a.nsb), dump=False)
        for fr in frames:
            STATS.clear()
            t1 = time.perf_counter()
            ch = run(policy, fr)
            dt = time.perf_counter() - t1
            ref = fr["ref_chunk_raw"].float()
            err = (ch - ref).abs()
            fi = int(fr["frame"])
            chunks[(vn, fi)] = ch
            r = dict(variant=vn, frame=fi, max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                     seconds=round(dt, 1))
            base = {"F1": "F0", "C1": "C0", "C1n10": "C0"}.get(vn)
            if base and (base, fi) in chunks:
                r[f"max_vs_{base}"] = float((ch - chunks[(base, fi)]).abs().max())
            if STATS:
                r["kernel"] = {
                    k: dict(calls=s["calls"], K=s["K"], elems=s["elems"], rows_per_call=sorted(set(s["rows"])),
                            q_err_steps_max=round(s["q_max"], 4), q_err_steps_rms=round((s["q_sq"] / s["n"]) ** 0.5, 4),
                            y_err_steps_max=round(s["y_max"], 4), y_err_steps_rms=round((s["y_sq"] / s["n"]) ** 0.5, 5))
                    for k, s in STATS.items()
                }
            print(json.dumps(r), flush=True)
            results.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "policy_check.json"
    old = []
    if out.exists():
        done = {(x["variant"], x["frame"]) for x in results}
        old = [r for r in json.loads(out.read_text()) if (r["variant"], r["frame"]) not in done]
    out.write_text(json.dumps(old + results, indent=1))
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
