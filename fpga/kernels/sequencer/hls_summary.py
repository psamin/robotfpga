"""Summarise the sequencer csynth projects: totals, the per-instance breakdown, and how many hardware
instances of the GEMM engine (and of its MAC pipeline) each variant has.

    python hls_summary.py [build] [--out build/hls_summary.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import xml.etree.ElementTree as ET


def res(node):
    r = node.find("AreaEstimates/Resources")
    return {k: int(r.findtext(k, "0")) for k in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")}


def walk_instances(rpt_dir: pathlib.Path, module: str, path=(), out=None, depth=0):
    """Expand the instance tree from the per-module .rpt files ("Instance" tables)."""
    out = [] if out is None else out
    f = rpt_dir / f"{module}_csynth.rpt"
    if not f.exists() or depth > 12:
        return out
    txt = f.read_text(errors="replace")
    m = re.search(r"\* Instance:\s*\n(.*?)\n\s*\n", txt[txt.find("== Utilization Estimates"):], re.S)
    if not m:
        return out
    for ln in m.group(1).splitlines():
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cols) < 7 or cols[0] in ("Instance", "Total") or cols[0].startswith("+"):
            continue
        inst, mod = cols[0], cols[1]
        try:
            bram, dsp, ff, lut = (int(cols[i]) for i in range(2, 6))
        except ValueError:
            continue
        out.append(dict(path="/".join(path + (inst,)), module=mod, BRAM_18K=bram, DSP=dsp, FF=ff, LUT=lut))
        walk_instances(rpt_dir, mod, path + (inst,), out, depth + 1)
    return out


def summarize(proj: pathlib.Path):
    sol = proj / "sol1/syn/report"
    xml = sol / "csynth.xml"
    if not xml.exists():
        return None
    root = ET.parse(xml).getroot()
    tot = res(root)
    timing = root.find("PerformanceEstimates/SummaryOfTimingAnalysis")
    est = float(timing.findtext("EstimatedClockPeriod"))
    target = float(root.findtext("UserAssignments/TargetClockPeriod"))
    inst = walk_instances(sol, "seq_top")
    # the engine proper (gemm_mode is a thin wrapper around it; its Pipeline_* children are its loops)
    gemm = [i for i in inst if i["module"].startswith("gemm_stub_") and "Pipeline" not in i["module"]]
    mac = [i for i in inst if i["module"].endswith("_Pipeline_mac")]
    return dict(project=proj.name, target_ns=target, estimated_ns=est, total=tot,
                gemm_instances=[(i["path"], i["module"], i["LUT"], i["DSP"]) for i in gemm],
                mac_pipeline_instances=[(i["path"], i["LUT"], i["DSP"]) for i in mac],
                instances=[i for i in inst if i["path"].count("/") <= 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build", nargs="?", default=str(pathlib.Path(__file__).resolve().parent / "build"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    allres = []
    for proj in sorted(pathlib.Path(args.build).glob("seq_v*_p*")):
        s = summarize(proj)
        if not s:
            print(proj.name, ": no csynth report")
            continue
        allres.append(s)
        t = s["total"]
        print(f"== {s['project']}: target {s['target_ns']} ns, estimated {s['estimated_ns']} ns; "
              f"LUT {t['LUT']} FF {t['FF']} DSP {t['DSP']} BRAM18 {t['BRAM_18K']} URAM {t['URAM']}")
        print(f"   GEMM engine instances: {len(s['gemm_instances'])} {s['gemm_instances']}")
        print(f"   MAC pipeline instances: {len(s['mac_pipeline_instances'])} {s['mac_pipeline_instances']}")
        for i in s["instances"]:
            print(f"     {i['path']:60s} {i['module']:45s} LUT {i['LUT']:6d} FF {i['FF']:6d} DSP {i['DSP']:4d} "
                  f"BRAM18 {i['BRAM_18K']}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
