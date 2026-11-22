"""Full-policy check of emit_act on checkpoint -12392 and its 3 replay frames (frames 0 / 3563 / 7489).

Reuses fpga/kernels/gelu_silu/policy_check.py (the contract int16 engine emulation: contract quantizer, exact sums,
epilogue with FP32 cores and flushes, action_time_mlp_in's t half as the D row) and gs_emu (the bit-exact model of
the gelu_silu front end), and adds the fused path:

  E10  emit_act, NSB 10. Vision MLP: fc1's GEMM output a -> y = act(a) in FP32 (the fused GEMM's C) -> QUANT_X
       (contract quantizer) -> fc2. LM / expert MLP: ONE GEMM over the interleaved gate|up blob (blob row 32 i + c =
       gate row 16 i + c, 32 i + 16 + c = up row 16 i + c), its FP32 output de-interleaved by the epilogue's pairing,
       y = silu(gate) * up -> QUANT_X -> down_proj. Action-time MLP: y = silu(action_time_mlp_in(.)) in FP32 ->
       action_time_mlp_out quantizes it (QUANT_X).
  U10  the unfused contract path with the same kernel (policy_check variant C1n10): GEMM -> GELU / SILU_MUL with
       F_QOUT -> the next GEMM. (Its action-time SiLU passes q * s, which the next linear re-quantizes: up to one ulp
       of s away from the exact contract; E10 is the exact contract there.)

On every call E10 also computes the unfused kernel's (q, s) (gs_emu.kernel) and the separate gate / up GEMMs, and
asserts bit equality: de-interleaved fused GEMM == [gate GEMM, up GEMM]; QUANT_X(fused y) == the kernel's q and s.

    set HF_HUB_OFFLINE=1
    C:/kp/lerobot-venv/Scripts/python.exe fpga/kernels/phaseB/emit_act/check_policy.py [--variants U10,E10]
        [--frames 0,1,2] [--threads 8]
Writes JSON lines to stdout and build/policy_check.json next to this file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
GS = HERE.parents[1] / "gelu_silu"
sys.path.insert(0, str(GS))
import policy_check as pc  # noqa: E402  (fpga/kernels/gelu_silu/policy_check.py)

gs_emu = pc.gs_emu
NSB = 10
MODE = {"v": "U10"}
EQ = {"calls": 0, "values": 0, "gu_calls": 0}


def interleave_idx(F: int) -> torch.Tensor:
    """natural row of blob row r: [gate F rows; up F rows] -> per 32 rows 16 gate then 16 up"""
    r = torch.arange(2 * F)
    i, c = r // 32, r % 32
    return torch.where(c < 16, 16 * i + c, F + 16 * i + (c - 16))


def fused_act(kind: str, x: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
    """y of the fused GEMM (FP32) and the per-call equality check against the unfused kernel's (q, s)."""
    mode = {"gelu": 0, "silu1": 2}.get(kind, 1)
    yb = gs_emu.act_bits(gs_emu.f2b(x), gs_emu.f2b(u) if u is not None else None, mode, NSB)
    y = gs_emu.b2f(yb)
    q, s = pc.cquant(y)  # QUANT_X of the fused output
    qk, sk, _ = gs_emu.kernel(x, u, mode, NSB)  # the unfused act command with F_QOUT
    assert torch.equal(q.to(torch.int64), qk.to(torch.int64)), f"{kind}: q differs"
    assert torch.equal(s.contiguous().view(torch.int32), sk.to(torch.float32).contiguous().view(torch.int32)), (
        f"{kind}: s differs"
    )
    EQ["calls"] += 1
    EQ["values"] += int(q.numel())
    return y


def make_vision_mlp(mod, lin1, lin2, idx):
    orig = ORIG["vision"](mod, lin1, lin2, idx)

    def forward(self, h):
        if MODE["v"] != "E10":
            return orig(self, h)
        a = lin1(h)  # the GEMM's epilogue output (fc1 + bias), as the fused GEMM computes it before the act
        return lin2.preq(*pc.cquant(fused_act("gelu", a, None)))

    return forward


def make_llama_mlp(mod, lg, lu, ld, kind):
    orig = ORIG["llama"](mod, lg, lu, ld, kind)

    def forward(self, x):
        if MODE["v"] != "E10":
            return orig(self, x)
        q, s = pc.cquant(x)  # the NORM's int16 output feeding gate|up
        wg, sg = lg.prep()
        wu, su = lu.prep()
        F = wg.shape[0]
        nat = interleave_idx(F)
        wi = torch.cat([wg, wu], 0)[nat]
        si = torch.cat([sg, su], 0)[nat]
        shape = q.shape
        acc = q.reshape(-1, shape[-1]).double() @ wi.t()
        yi = pc.gemm_out(acc, si, s.reshape(-1, 1).to(torch.float32), None)  # the fused GEMM's epilogue, interleaved
        r = torch.arange(F)
        gpos = 32 * (r // 16) + r % 16  # blob row of gate row r / of up row r
        g, u = yi[:, gpos], yi[:, gpos + 16]
        g0, u0 = lg.preq(q, s).reshape(-1, F), lu.preq(q, s).reshape(-1, F)  # the unfused gate and up GEMMs
        assert torch.equal(g.contiguous().view(torch.int32), g0.contiguous().view(torch.int32)), "gate differs"
        assert torch.equal(u.contiguous().view(torch.int32), u0.contiguous().view(torch.int32)), "up differs"
        EQ["gu_calls"] += 1
        y = fused_act(kind, g, u).reshape(*shape[:-1], F)
        return ld.preq(*pc.cquant(y))

    return forward


def make_silu_alone():
    orig = ORIG["silu"]()

    def f(x, inplace=False):
        if MODE["v"] != "E10":
            return orig(x, inplace)
        return fused_act("silu1", x, None)  # FP32 y; action_time_mlp_out quantizes it (QUANT_X)

    return f


ORIG = {"vision": pc.make_vision_mlp, "llama": pc.make_llama_mlp, "silu": pc.make_silu_alone}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="U10,E10")
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pc.check_cquant()
    pc.make_vision_mlp, pc.make_llama_mlp, pc.make_silu_alone = make_vision_mlp, make_llama_mlp, make_silu_alone
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(str(pc.CKPT))
    vlm = pc.local_vlm(cfg.vlm_model_name)
    if vlm:
        cfg.vlm_model_name = vlm
    policy = SmolVLAPolicy.from_pretrained(str(pc.CKPT), config=cfg).to("cpu").eval().float()
    pc.install(policy)
    print(f"loaded in {time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    files = sorted(pc.BATCHES.glob("frame_*.pt"))
    frames = [torch.load(files[int(i)], weights_only=False) for i in a.frames.split(",")]
    pc.CFG.update(linear="int16", act="kernel", nsb=NSB, dump=False)
    results, chunks = [], {}
    for vn in a.variants.split(","):
        MODE["v"] = vn
        for fr in frames:
            for k in EQ:
                EQ[k] = 0
            pc.STATS.clear()
            t1 = time.perf_counter()
            ch = pc.run(policy, fr)
            ref = fr["ref_chunk_raw"].float()
            err = (ch - ref).abs()
            fi = int(fr["frame"])
            chunks[(vn, fi)] = ch
            r = dict(variant=vn, frame=fi, max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                     seconds=round(time.perf_counter() - t1, 1))
            if vn == "E10":
                r["fused_eq_unfused"] = dict(EQ)
                if ("U10", fi) in chunks:
                    r["max_vs_U10"] = float((ch - chunks[("U10", fi)]).abs().max())
            print(json.dumps(r), flush=True)
            results.append(r)
    out = HERE / "build" / "policy_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
