"""Run SmolVLA -12392 through the command list on the laptop and compare with the reference chunks.

predict_action_chunk(d["batch"], noise=d["noise"]) runs as usual (image prep, state pad, unpad), with
VLAFlowMatching.sample_actions replaced by: host_inputs -> arena bytes -> exec_ref walks the list from
index 0 (restarting after every YIELD, as the ARM would) -> read x_t from the arena. Frames alternate
between the two arenas (arena i % 2), as two inferences in flight would.

    python check_policy.py <ckpt dir> <batches dir> [--modes exact,int16,int16c] [--dump <dir>]
Modes: exact = qbytes 4 (FP32 everywhere: checks the list), int16 = int16 GEMM operands, int16c = int16
with the prefix compacted to its valid tokens. --dump writes the int16 list of arena 0 and its expected
decode trace for tb_seq.cpp.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from arena import CmaPlan  # noqa: E402
from build_smolvla import Program, Shared  # noqa: E402
from exec_ref import Executor, Memory  # noqa: E402
from seq_format import C, encode_list  # noqa: E402

CMA_BASE = 0x4000_0000  # placeholder physical base (the board's CMA address is only known at run time)


def setup(policy, qbytes, compact, n_lang_valid):
    plan = CmaPlan(base=CMA_BASE, weights_size=(2 << 30) if qbytes == 4 else (1 << 30))
    sh = Shared(policy, plan, qbytes=qbytes)
    nl = n_lang_valid if compact else 48
    progs = [Program(sh, plan.arena(i), n_prefix_lang=nl, compact=compact) for i in range(2)]
    mem = Memory()
    for r in (plan.consts, plan.lists, *plan.arenas):
        mem.add_region(r.base, r.size)
    for addr, raw in sh.const_bytes:
        mem.write(addr, raw)
    mem.blobs, mem.tables = sh.blobs, sh.tables
    for pr in progs:
        lb = pr.place_list(plan.lists)
        mem.write(lb.addr, encode_list(pr.cmds))
    return plan, sh, progs, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("bdir")
    ap.add_argument("--modes", default="exact,int16,int16c")
    ap.add_argument("--dump", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    p = SmolVLAPolicy.from_pretrained(args.ckpt).to("cpu").eval().float()
    frames = [torch.load(f, weights_only=False) for f in sorted(pathlib.Path(args.bdir).glob("frame_*.pt"))]
    n_valid = {int(d["batch"]["observation.language.attention_mask"].sum()) for d in frames}
    assert len(n_valid) == 1, n_valid
    n_valid = n_valid.pop()
    results = dict(frames=[int(d["frame"]) for d in frames], modes={})

    # the unmodified policy on this laptop, for reference
    orig = p.model.sample_actions
    errs = []
    for d in frames:
        p.reset()
        with torch.no_grad():
            ch = p.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
        errs.append(float((ch.float() - d["ref_chunk_raw"].float()).abs().max()))
    results["torch_fp32_vs_ref"] = errs
    print("torch fp32 policy vs ref_chunk_raw:", errs, flush=True)

    for mode in args.modes.split(","):
        qbytes = 4 if mode == "exact" else 2
        compact = mode.endswith("c")
        t0 = time.perf_counter()
        plan, sh, progs, mem = setup(p, qbytes, compact, n_valid)
        tb = time.perf_counter() - t0
        ex = Executor(mem, qbytes, gelu=sh.gelu)
        state = {}

        def sample_actions(images, img_masks, lang_tokens, lang_masks, state_t, noise=None, **kw):
            pr = progs[state["arena"]]
            for addr, raw in pr.host_inputs(images, img_masks, lang_tokens, lang_masks, state_t, noise).items():
                mem.write(addr, raw)
            idx, starts = 0, 0
            while True:
                code, idx, done = ex.run(pr.list_buf.addr, idx, pr.status.addr)
                starts += 1
                if code != C.ST_YIELD:
                    break
            assert code == C.ST_END, C.stnames.get(code)
            state["starts"] = starts
            x = mem.view(pr.x_t.addr, (1, 50, 32), (50 * 128, 128, 4), torch.float32).clone()
            return x

        p.model.sample_actions = sample_actions
        per = []
        for i, d in enumerate(frames):
            state["arena"] = i % 2
            p.reset()
            t1 = time.perf_counter()
            with torch.no_grad():
                ch = p.predict_action_chunk(dict(d["batch"]), noise=d["noise"])
            dt = time.perf_counter() - t1
            e = (ch.float() - d["ref_chunk_raw"].float()).abs()
            per.append(dict(frame=int(d["frame"]), arena=i % 2, max_abs=float(e.max()), mean_abs=float(e.mean()),
                            starts=state["starts"], exec_s=round(dt, 1)))
            print(f"[{mode}] frame {d['frame']} arena {i % 2}: max |chunk - ref| = {float(e.max()):.3e} "
                  f"mean {float(e.mean()):.3e}  ({dt:.1f} s)", flush=True)
        p.model.sample_actions = orig
        cnt = progs[0].counts()
        results["modes"][mode] = dict(qbytes=qbytes, compact=compact, NP=progs[0].NP, frames=per, counts=cnt,
                                      build_s=round(tb, 1), cma=plan.summary(),
                                      arena_bytes=plan.arena(0).used, weights_bytes=plan.weights.used,
                                      consts_bytes=plan.consts.used, lists_bytes=plan.lists.used,
                                      roles={k: v for k, v in plan.arena(0).roles.items()})
        print(f"[{mode}] commands per chunk: {cnt}", flush=True)
        print(plan.summary(), flush=True)
        if args.dump and mode == "int16":
            out = pathlib.Path(args.dump)
            out.mkdir(parents=True, exist_ok=True)
            pr = progs[0]
            (out / "chunk_list.bin").write_bytes(encode_list(pr.cmds))
            (out / "chunk_trace.txt").write_text(
                "".join(c.trace_line(i) + "\n" for i, c in enumerate(pr.cmds) if c.op != C.OP_END))
            (out / "chunk_list.txt").write_text("".join(
                f"{i:5d} {C.opnames[c.op]:9s} tag {c.tag:08x} M {c.M} K {c.K} N {c.N} B {c.B} G {c.G}  {c.note}\n"
                for i, c in enumerate(pr.cmds)))
            # one SigLIP layer and one expert self-attention step-layer, for reading
            print("dumped", out, flush=True)
        del mem, sh, progs
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
