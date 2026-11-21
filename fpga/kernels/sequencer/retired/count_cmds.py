"""Commands per chunk, a lint of the list against the mode contracts, and the host-time estimate.

Reads the int16 list that check_policy.py --dump wrote (build/py/chunk_list.bin + chunk_list.txt for the
notes); needs no torch and no checkpoint.

    python count_cmds.py [build/py] [--out build/py/counts.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from seq_format import C, decode_list  # noqa: E402

STAGES = {1: "vision", 2: "connector+prefix", 3: "lm", 4: "cross_kv_hoist", 5: "expert", 6: "head+euler"}

# Host-side cost model (inputs; the per-call figure is the task's, the others are labelled assumptions)
CALLS_TODAY = 1419          # engine calls per chunk today (build/brief/count_calls.json)
MS_PER_CALL_TODAY = 2.5     # host ms per call today (task statement)
MS_PER_START = 0.1          # one sequencer start: ~6 register writes (2.3 us each, measured reg read) + the
                            # ap_done / interrupt wait (assumption: the hardware_options.md "tightened driver" figure)
MS_INPUTS = 3.1             # host inputs per chunk: 1.6 MB patches + ids/state/noise/positions copy + flush
                            # (xrtmem.py measured 3.1 ms for a 3 MB cacheable copy_in; the inputs are ~1.6 MB)


def lint(cmds):
    """Constraints the modes are specified with (README "Mode contracts"). Returns a list of problems."""
    bad = []
    for i, c in enumerate(cmds):
        a16 = lambda v: v % 16 == 0
        if c.op == C.OP_GEMM:
            if c.M % 32 or c.K % 8 or not 1 <= c.K <= 4096 or c.N < 1:
                bad.append((i, "gemm dims", c.M, c.K, c.N))
            for nm in "abcef":
                s = getattr(c, nm)
                if not a16(s.addr):
                    bad.append((i, f"gemm slot {nm} addr", hex(s.addr)))
            for nm in "ac":
                if not a16(getattr(c, nm).pitch):
                    bad.append((i, f"gemm slot {nm} pitch", getattr(c, nm).pitch))
            if c.B % c.G:
                bad.append((i, "gemm B % G", c.B, c.G))
        elif c.op in (C.OP_NORM, C.OP_SOFTMAX, C.OP_GELU, C.OP_SILU_MUL, C.OP_ROPE, C.OP_QUANT_X, C.OP_AXPY):
            for nm in "ac":
                s = getattr(c, nm)
                if not a16(s.addr) or (c.N > 1 and not a16(s.pitch)):
                    bad.append((i, f"{C.opnames[c.op]} slot {nm} not 16-byte aligned", hex(s.addr), s.pitch))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(pathlib.Path(__file__).resolve().parent / "build/py"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    blob = (d / "chunk_list.bin").read_bytes()
    cmds = decode_list(blob)
    notes = [(re.search(r" G \d+  ?(.*)$", ln) or re.search("()$", ln)).group(1)
             for ln in (d / "chunk_list.txt").read_text().splitlines()]
    assert len(notes) == len(cmds)

    by_op = Counter(C.opnames[c.op] for c in cmds)
    by_stage = Counter(STAGES.get(c.tag >> 24, "?") for c in cmds)
    # per unit of work
    vis_layer = Counter((c.tag >> 16) & 0xFF for c in cmds if c.tag >> 24 == 1)
    lm_layer = Counter((c.tag >> 16) & 0xFF for c in cmds if c.tag >> 24 == 3)
    ex = defaultdict(int)
    for c in cmds:
        if c.tag >> 24 == 5:
            ex[((c.tag >> 8) & 0xFF, (c.tag >> 16) & 0xFF)] += 1
    step_cmds = Counter(((c.tag >> 8) & 0xFF) for c in cmds if c.tag >> 24 in (5, 6) and c.op != C.OP_END)
    gemm_by_note = Counter(n for c, n in zip(cmds, notes) if c.op == C.OP_GEMM)
    gemm_items = sum(c.B for c in cmds if c.op == C.OP_GEMM)
    macs = sum(c.B * c.N * c.M * c.K for c in cmds if c.op == C.OP_GEMM)

    n = len(cmds)
    starts = {"per chunk (no YIELD)": 1, "per Euler step (F_YIELD on the 10 AXPYs)": 11,
              "per layer (F_YIELD after every layer: 12 + 16 + 160 + 2)": 190}
    today_s = CALLS_TODAY * MS_PER_CALL_TODAY / 1e3
    host = {k: round((v * MS_PER_START + MS_INPUTS) / 1e3, 4) for k, v in starts.items()}

    res = dict(
        records=n, list_bytes=len(blob), dispatched=n - 1, by_op=dict(by_op), by_stage=dict(by_stage),
        vision_per_layer=sorted(set(vis_layer.values())), lm_per_layer=dict(sorted(lm_layer.items())),
        expert_per_step_layer={"self-attention layers (even)": sorted({v for (s, l), v in ex.items() if l % 2 == 0}),
                               "cross-attention layers (odd)": sorted({v for (s, l), v in ex.items() if l % 2 == 1})},
        per_euler_step=sorted(set(step_cmds.values())),
        gemm_commands=by_op["GEMM"], gemm_items=gemm_items, gemm_macs=macs, gemm_by_note=dict(gemm_by_note),
        lint=lint(cmds),
        host_today_s=today_s, host_with_sequencer_s=host,
        host_saved_s={k: round(today_s - v, 3) for k, v in host.items()},
        assumptions=dict(ms_per_call_today=MS_PER_CALL_TODAY, ms_per_start=MS_PER_START, ms_inputs=MS_INPUTS),
    )
    for k, v in res.items():
        if k != "gemm_by_note":
            print(f"{k}: {v}")
    print("gemm_by_note:")
    for k, v in sorted(gemm_by_note.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
