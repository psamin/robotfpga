"""Replays every start tb_seq.cpp logged (--twin DIR) through exec_ref.py and compares, start by start: the return
value, status words 0-9 (all but the cycle count), the dispatch trace (every unit call with every decoded field)
and the output bytes the testbench checked. This is the C++ / Python twin check of the walk (review defect 11).

    python check_twin.py build/twin/twin_lo.txt build/twin/twin_hi.txt
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from exec_ref import Executor, Memory

WINDOW = 512 << 20  # tb_seq.cpp's memory window


def parse(path: Path):
    starts, cur = [], None
    for ln in path.read_text().splitlines():
        tag, _, rest = ln.partition(" ")
        if tag == "START":
            kv = dict(x.split("=") for x in rest.split()[1:])
            cur = {k: int(v, 16) if k in "MKN" else int(v) for k, v in kv.items()}
            cur |= {"writes": [], "trace": [], "outs": [], "status": None}
        elif tag == "W":
            a, n, h = rest.split()
            cur["writes"].append((int(a, 16), bytes.fromhex(h)))
        elif tag == "RET":
            cur["ret"] = int(rest, 16)
        elif tag == "STATUS":
            cur["status"] = [int(v, 16) for v in rest.split()]
        elif tag == "T":
            cur["trace"].append(rest)
        elif tag == "O":
            a, n, h = rest.split()
            cur["outs"].append((int(a, 16), bytes.fromhex(h)))
        elif tag == "END":
            starts.append(cur)
    return starts


def replay(path: Path) -> int:
    starts = parse(path)
    base = min(a for s in starts for a, _ in s["writes"]) & ~(WINDOW - 1) if starts else 0
    mem = Memory()
    mem.add_region(base, WINDOW)
    ex = Executor(mem, fe="torch")
    ex.trace = []
    fails, offset = 0, None
    for i, s in enumerate(starts):
        for a, raw in s["writes"]:
            mem.write(a, raw)
        ex.trace.clear()
        ex.dry, ex.fault = bool(s["dry"]), s["fault"]
        ret = ex.start(s["M"], s["K"], s["N"])
        bad = []
        if ret != s["ret"]:
            bad.append(f"return 0x{ret:x} != 0x{s['ret']:x}")
        if s["status"] is not None:
            got = list(struct.unpack("<10I", mem.read(s["N"] << 4, 40)))
            want = list(s["status"])
            if got[5] == want[5] == 0x53455153:  # written by this start (the magic word)
                if offset is None:
                    offset = want[6] - got[6]  # starts since configuration: the C++ process ran other starts first
                got[6] += offset
            if got != want:
                bad.append(f"status {[hex(v) for v in got]} != {[hex(v) for v in want]}")
        if ex.trace != s["trace"]:
            bad.append(f"trace {len(ex.trace)} vs {len(s['trace'])} lines")
            for a, b in zip(ex.trace, s["trace"]):
                if a != b:
                    bad.append(f"  py  {a}\n  c++ {b}")
                    break
        for a, raw in s["outs"]:
            got = mem.read(a, len(raw))
            if got != raw:
                nd = sum(x != y for x, y in zip(got, raw))
                bad.append(f"output 0x{a:x}+{len(raw)}: {nd} bytes differ")
        if bad:
            fails += 1
            print(f"FAIL {path.name} start {i} (M=0x{s['M']:x}):\n  " + "\n  ".join(bad))
    n_out = sum(len(s["outs"]) for s in starts)
    print(f"{path.name}: {len(starts)} starts, {sum(len(s['trace']) for s in starts)} unit calls, {n_out} output "
          f"ranges: {'identical' if not fails else f'{fails} starts differ'}")
    return fails


def main() -> int:
    fails = sum(replay(Path(p)) for p in sys.argv[1:])
    print("twin:", "PASS" if fails == 0 else f"FAILED ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
