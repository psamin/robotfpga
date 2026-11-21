"""The contract's -12392 chunk list (fpga/kernels/contract_plan.py build()) at a given CMA base, for tb_seq.cpp:
<out>.bin (the records) and <out>.trace.txt (exec_ref.trace_line of every record a start dispatches, from
contract.decode in Python). Also checks contract.validate on every record and prints the list's shape.

    python make_chunk.py --base 0x800000000 --out build/chunk/chunk_hi
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract as K  # noqa: E402
import contract_plan  # noqa: E402
from contract import C  # noqa: E402
from exec_ref import U_NONE, trace_line, unit_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x8_0000_0000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    P = contract_plan.build(a.base)
    bad = K.validate_list(P.cmds)
    blob = K.encode_list(P.cmds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".bin").write_bytes(blob)
    lines = []
    for i in range(len(P.cmds)):
        st, c = K.decode(blob[i * C.REC_BYTES : (i + 1) * C.REC_BYTES])
        assert st == C.ST_END, (i, st)
        if c.op != C.OP_END and unit_of(c.op) != U_NONE:
            lines.append(trace_line(c, unit_of(c.op)))
    a.out.with_suffix(".trace.txt").write_text("\n".join(lines) + "\n")
    ops = collections.Counter(C.opnames[c.op] for c in P.cmds)
    hi = sum(1 for c in P.cmds for s in c.s.values() if s.addr >> 32)
    print(
        f"{len(P.cmds)} records at base 0x{a.base:x}, {len(bad)} invalid, {len(lines)} unit calls, "
        f"{hi} slot addresses >= 4 GB; " + ", ".join(f"{k} {v}" for k, v in ops.most_common())
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
