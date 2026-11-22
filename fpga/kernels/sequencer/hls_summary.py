"""Summary of the sequencer's csynth runs (build/hls_<tag>/sol1/syn/report): estimated clock, the totals of
the
top, the m_axi / control adapters, the sequencer region (walk + cycle counter), the null units, validate_hw,
and the cycles per command (cmd_loop iteration latency).

    python hls_summary.py [build/hls_p40 build/hls_p40_u0 ...] [--json out.json]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(rpt: str) -> dict[str, list[str]]:
    """Module rows of csynth.rpt's 'Performance & Resource Estimates' table, by name."""
    sec = rpt.split("Performance & Resource Estimates", 1)[1].split("+ HW Interfaces", 1)[0]
    out = {}
    for ln in sec.splitlines():
        p = [x.strip() for x in ln.split("|")]
        if len(p) > 14 and p[1][:1] in "+o":
            out.setdefault(p[1].lstrip("+o ").strip(), p)
    return out


def num(s: str) -> int:
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else 0


def summary(d: Path) -> dict:
    rep = d / "sol1/syn/report"
    r = rows((rep / "csynth.rpt").read_text())
    top = (rep / "seq_top_csynth.rpt").read_text()
    est = re.search(r"\|ap_clk\s*\|\s*([\d.]+) ns\|\s*([\d.]+) ns\|", top)
    inst = {}
    for ln in top.split("* Instance:", 1)[1].split("* DSP:", 1)[0].splitlines():
        p = [x.strip() for x in ln.split("|")]
        if len(p) > 7 and p[1] and not p[1].startswith(("Instance", "Total")):
            inst[p[2]] = {"dsp": num(p[4]), "ff": num(p[5]), "lut": num(p[6])}

    def res(name: str) -> dict:
        for k, p in r.items():
            if k.startswith(name):
                return {
                    "lut": num(p[13]),
                    "ff": num(p[12]),
                    "dsp": num(p[11]),
                    "lat": num(p[4]),
                    "iter": num(p[6]),
                }
        return {}

    nulls = [res(k) for k in r if k.startswith("null_unit")]
    return {
        "target_ns": float(est.group(1)) if est else None,
        "estimated_ns": float(est.group(2)) if est else None,
        "top": res("seq_top"),
        "adapters_lut": sum(v["lut"] for k, v in inst.items() if k.endswith(("_m_axi", "_s_axi"))),
        "region": res("region_"),
        "null_units_lut": sum(n["lut"] for n in nulls),
        "validate_hw": res("validate_t"),
        "cycles_per_cmd": res("cmd_loop").get("iter"),
    }


def main() -> int:
    argv = sys.argv[1:]
    out = argv.pop(argv.index("--json") + 1) if "--json" in argv else None
    args = [a for a in argv if not a.startswith("--")]
    dirs = [Path(a) for a in args] or sorted((HERE / "build").glob("hls_p*"))
    res = {}
    for d in dirs:
        if (d / "sol1/syn/report/csynth.rpt").exists():
            s = summary(d)
            res[d.name] = s
            seq = s["region"]["lut"] - s["null_units_lut"]
            print(
                f"{d.name}: est {s['estimated_ns']} ns (target {s['target_ns']}); top {s['top']['lut']} LUT "
                f"{s['top']['ff']} FF {s['top']['dsp']} DSP; adapters {s['adapters_lut']} LUT; region "
                f"{s['region']['lut']} LUT {s['region']['ff']} FF; null units {s['null_units_lut']} LUT; "
                f"sequencer {seq} LUT; validate_hw {s['validate_hw'].get('lut')} LUT / "
                f"{s['validate_hw'].get('lat')} "
                f"clocks; {s['cycles_per_cmd']} clocks per command"
            )
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
