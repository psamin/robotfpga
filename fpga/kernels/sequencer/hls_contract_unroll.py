"""Writes a copy of ../contract.h with `#pragma HLS UNROLL` in the loops of decode() and validate_t() and,
with
--inline-decode, `#pragma HLS INLINE` in decode() (a synthesis experiment for a change proposed to the
contract owner: the sequencer itself decodes with seq::decode_hw, so only validate_t's loops matter to it;
README "Synthesis"). Prints the diff.

    python hls_contract_unroll.py build/contract_unrolled/contract.h [--inline-decode]
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "contract.h"
INLINE_DECODE = "--inline-decode" in sys.argv


def patch(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    fn = ""
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"static inline uint32_t (decode|validate_t)\(", ln)
        if m:
            fn = m.group(1)
            if fn == "decode" and INLINE_DECODE:
                out += [ln, "#pragma HLS INLINE"]
                i += 1
                continue
        elif re.match(r"^(static|template|}\s*$)", ln) and not ln.startswith(
            "static inline uint32_t validate_t"
        ):
            if ln.startswith("static") or ln.startswith("template"):
                fn = ""
        f = re.match(r"^(\s*)for \(uint32_t \w+ = [^;]+; \w+ < \w+; \w+\+\+\)(.*)$", ln)
        if fn and f:
            ind, rest = f.group(1), f.group(2).strip()
            head = ln[: len(ln) - len(f.group(2))]
            if rest == "{" and False:
                pass
            if rest == "{":
                out += [ln, "#pragma HLS UNROLL"]
            elif rest:
                out += [head + " {", "#pragma HLS UNROLL", ind + "  " + rest, ind + "}"]
            else:
                out += [head + " {", "#pragma HLS UNROLL", lines[i + 1], ind + "}"]
                i += 1
            i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def main() -> int:
    dst = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
    src = SRC.read_text()
    new = patch(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new)
    d = list(
        difflib.unified_diff(
            src.split("\n"), new.split("\n"), "contract.h", "contract_unrolled.h", lineterm=""
        )
    )
    print("\n".join(d))
    print(f"{sum(1 for x in d if x.startswith('+#pragma'))} loops unrolled -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
