"""Summarize a Vitis HLS csynth report of quant_top: per-process resources, loop II, estimated clock.

    python hls_summary.py build/qzm_recip_p40[/sol1/syn/report]

Prints the row_mode / row_df totals (what float_top adds), each DATAFLOW process, every loop with its II and
pipeline depth, the estimated clock, and the largest expressions of each process (to find area hogs).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def report_dir(arg: str) -> Path:
    p = Path(arg)
    if (p / "csynth.rpt").exists():
        return p
    return p / "sol1" / "syn" / "report"


def totals(rpt: Path) -> dict:
    """The Total row of a module report's Utilization Estimates."""
    txt = rpt.read_text(errors="replace")
    m = re.search(r"\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", txt)
    if not m:
        return {}
    b, d, f, lut, u = (int(x) for x in m.groups())
    return dict(BRAM18=b, DSP=d, FF=f, LUT=lut, URAM=u)


def expressions(rpt: Path, top: int = 4) -> list[tuple[str, str, int]]:
    txt = rpt.read_text(errors="replace")
    sec = txt.split("* Expression:", 1)
    if len(sec) < 2:
        return []
    body = sec[1].split("* Multiplexer", 1)[0]
    rows = []
    for line in body.splitlines():
        c = [x.strip() for x in line.split("|")]
        if len(c) > 6 and c[5].isdigit() and c[1] != "Total":
            rows.append((c[1], c[2], int(c[5])))
    rows.sort(key=lambda r: -r[2])
    return rows[:top]


def main() -> None:
    d = report_dir(sys.argv[1])
    main_rpt = (d / "csynth.rpt").read_text(errors="replace")
    top = (d / "quant_top_csynth.rpt").read_text(errors="replace")
    est = re.search(r"\|ap_clk\s*\|\s*([\d.]+) ns\|\s*([\d.]+) ns\|\s*([\d.]+) ns\|", top)
    print(f"report {d}")
    for name in ("row_mode", "row_df"):
        f = d / f"{name}_csynth.rpt"
        if f.exists():
            print(f"{name:12s} {totals(f)}")
    df = (d / "row_df_csynth.rpt").read_text(errors="replace") if (d / "row_df_csynth.rpt").exists() else ""
    util = df.split("== Utilization Estimates", 1)[-1]
    m = re.search(r"\* Instance:(.*?)\* DSP", util, re.S)
    if m:
        for line in m.group(1).splitlines():
            c = [x.strip() for x in line.split("|")]
            if len(c) > 6 and c[1].endswith("_U0"):
                print(f"  {c[2]:12s} BRAM18 {c[3]:>3s} DSP {c[4]:>3s} FF {c[5]:>6s} LUT {c[6]:>6s}")
    fifo = re.search(r"\|FIFO\s*\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|\s*(\S+)\|", df)
    if fifo:
        g = fifo.groups()
        print(f"  FIFOs        BRAM18 {g[0]} FF {g[2]} LUT {g[3]} URAM {g[4]}")
    print("loops (name, II, iteration latency):")
    for line in main_rpt.splitlines():
        if re.match(r"\s*\|\s+o ", line):
            c = [x.strip() for x in line.split("|")]
            print(f"  {c[1]:28s} II {c[7]:>3s}  depth {c[6]:>5s}  issue {c[2]}")
    sl = re.search(r"\|\+ quant_top\s*\|\s*(\S+)\|\s*(\S+)\|", main_rpt)
    if sl:
        print(f"top issue {sl.group(1)}, slack {sl.group(2)} ns")
    if est:
        t, e, u = (float(x) for x in est.groups())
        print(
            f"clock target {t} ns, estimated {e} ns (budget {t - u:.2f} ns after {u} ns uncertainty), "
            f"Fmax {1000 / e:.0f} MHz"
        )
    print("largest expressions per process:")
    for f in sorted(d.glob("qz_*_csynth.rpt")):
        ex = expressions(f)
        if ex and ex[0][2] >= 60:
            print(f"  {f.stem.replace('_csynth', ''):28s} " + ", ".join(f"{n}({o}) {v}" for n, o, v in ex))


if __name__ == "__main__":
    main()
