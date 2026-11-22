"""Full-policy check of SmolVLA -12392 through the contract's chunk list (contract_plan.py), against
ref_chunk_raw.

predict_action_chunk(d["batch"], noise=d["noise"]) runs as usual (image prep, state pad, unnormalize), with
VLAFlowMatching.sample_actions replaced by build_smolvla.Program.run: host inputs -> memory -> exec_ref walks
the list from index 0 -> x_t. One Program (one arena) serves all frames in order, so nothing may leak from one
chunk to the next.

Modes (exec_ref.py):
  exact    no quantization, FP32 torch front ends: checks the list (addresses, tilings, order,
           aliasing, folds)
  int16    contract quantizer + epilogue everywhere, FP32 torch front ends (torch softmax P through the
           quantizer)
  int16k   contract quantizer + epilogue, the kernels' bit-exact emulators (nr_emul, gs_emu, smx_emu,
           rope_emu): the golden executor of CONTRACT.md 7 (T-P1); int16k:norm+act (any of norm, act,
           rope, smx) uses the emulators of those front ends only

    HF_HUB_OFFLINE=1 python check_policy.py <ckpt dir> <batches dir> [--modes exact,int16,int16k]
        [--frames 0,3563] [--out x.json] [--no-plan-fix] [--nsb 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_smolvla import Program  # noqa: E402

KDIR = pathlib.Path(__file__).resolve().parents[1]
SOURCES = [
    "contract.h",
    "contract.py",
    "contract_plan.py",
    "sequencer/exec_ref.py",
    "sequencer/build_smolvla.py",
    "norm_residual/nr_emul.py",
    "gelu_silu/gs_emu.py",
    "softmax/smx_emu.py",
    "rope/rope_emu.py",
]


def sources() -> dict[str, str]:
    """sha256[:12] and mtime of every file the executor's arithmetic comes from (the kernels' emulators change
    while their owners work on them: the result belongs to these versions)."""
    out = {}
    for s in SOURCES:
        f = KDIR / s
        mtime = time.strftime("%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))
        out[s] = f"{hashlib.sha256(f.read_bytes()).hexdigest()[:12]} {mtime}"
    return out


def sampler(prog):
    """VLAFlowMatching.sample_actions replaced by one list start of prog (build_smolvla.Program.run)."""

    def sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
        return prog.run(images, img_masks, lang_tokens, lang_masks, state, noise)

    return sample_actions


MODES = {"exact": (True, "torch"), "int16": (False, "torch"), "int16k": (False, "kernels")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("bdir")
    ap.add_argument("--modes", default="exact,int16,int16k")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x8_0000_0000)
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--no-plan-fix", action="store_true", help="contract_plan.build as is (build_smolvla.chunk_plan)"
    )
    ap.add_argument("--frames", default="", help="comma-separated frame numbers (default: all)")
    ap.add_argument("--nsb", type=int, default=10, help="gelu_silu table size for fe=kernels (9 or 10)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    p = SmolVLAPolicy.from_pretrained(a.ckpt).to("cpu").eval().float()
    frames = [torch.load(f, weights_only=False) for f in sorted(pathlib.Path(a.bdir).glob("frame_*.pt"))]
    if a.frames:
        keep = {int(x) for x in a.frames.split(",")}
        frames = [d for d in frames if int(d["frame"]) in keep]
    res = {
        "frames": [int(d["frame"]) for d in frames],
        "base": hex(a.base),
        "plan_fix": not a.no_plan_fix,
        "nsb": a.nsb,
        "sources": sources(),
        "modes": {},
    }
    orig = p.model.sample_actions
    errs = []
    for d in frames:
        p.reset()
        with torch.no_grad():
            ch = p.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
        errs.append(float((ch.float() - d["ref_chunk_raw"].float()).abs().max()))
    res["torch_fp32_vs_ref"] = errs
    print("torch fp32 policy (no list) vs ref_chunk_raw:", [f"{e:.3e}" for e in errs], flush=True)
    for mode in a.modes.split(","):
        exact, fe = MODES[mode.split(":")[0]]
        if ":" in mode:  # int16k:norm+act = the kernels' emulators for these front ends only (attribution)
            fe = "kernels:" + mode.split(":", 1)[1]
        t0 = time.perf_counter()
        prog = Program(p, base=a.base, exact=exact, fe=fe, plan_fix=not a.no_plan_fix, nsb=a.nsb)
        tb = time.perf_counter() - t0
        print(
            f"[{mode}] {len(prog.cmds)} records, {prog.n_blobs} weight blobs written ({tb:.1f} s)", flush=True
        )

        p.model.sample_actions = sampler(prog)
        per = []
        try:
            for d in frames:
                p.reset()
                t1 = time.perf_counter()
                with torch.no_grad():
                    ch = p.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
                dt = time.perf_counter() - t1
                e = (ch.float() - d["ref_chunk_raw"].float()).abs()
                st = prog.last_status
                per.append(
                    {
                        "frame": int(d["frame"]),
                        "max_abs": float(e.max()),
                        "mean_abs": float(e.mean()),
                        "list_starts": st["list_starts"],
                        "status": {
                            k: st[k] for k in ("code", "next", "done", "starts", "version", "last_finished")
                        },
                        "exec_s": round(dt, 1),
                    }
                )
                print(
                    f"[{mode}] frame {d['frame']}: max |chunk - ref| = {float(e.max()):.3e}"
                    f"  mean {float(e.mean()):.3e}"
                    f"  (status {st['code']} next {st['next']} done {st['done']}, {dt:.1f} s)",
                    flush=True,
                )
        finally:
            p.model.sample_actions = orig
        res["modes"][mode] = {"exact": exact, "fe": fe, "frames": per, "op_calls": dict(prog.ex.counts)}
        del prog
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
