"""Per-mode clocks of one SmolVLA -12392 chunk on the integrated float_top, from the contract's list
(fpga/kernels/contract_plan.py) with its time model and the per-record terms set from this build:

  sequencer  --seq-clk clocks per record (csynth of the sequencer: fetch, decode_hw, validate_hw, walk) plus
             the DDR read latency of the record fetch (unmeasured: --ddr-lat)
  GEMM       the engine's per-tile model (contract_plan.gemm_item_clocks = model_int16.call_clocks plus the
             addend read); the unit's first-tile X and its store both move 1 io word per clock (csynth: II = 8
             per 8-word unit); --l0 / --tail per item (estimates)
  row ops / softmax  contract_plan's io-bound rules (1 io word per clock, the kernels' csynth IIs) with their
             per-command and per-item constants

    python fpga/hls/model_seq.py [--mhz 250 300 214.3] [--seq-clk 72] [--ddr-lat 100] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernels"))
import contract_plan as CP  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, nargs="+", default=[250.0])
    ap.add_argument("--ddr", type=float, default=14.0, help="DDR GB/s (unmeasured)")
    ap.add_argument("--seq-clk", type=int, default=72, help="sequencer clocks per record (csynth)")
    ap.add_argument(
        "--ddr-lat", type=int, default=100, help="DDR read latency per record fetch, clocks (unmeasured)"
    )
    ap.add_argument(
        "--l0", type=int, default=400, help="GEMM per-item start clocks (DATAFLOW start + DDR latency)"
    )
    ap.add_argument(
        "--tail", type=int, default=450, help="GEMM per-item drain clocks (last block: emit + store)"
    )
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    CP.SEQ_FETCH = a.seq_clk + a.ddr_lat
    CP.L0, CP.TAIL = a.l0, a.tail
    P = CP.build(0x8_0000_0000)
    out = {}
    for mhz in a.mhz:
        r = CP.report(P, mhz, a.ddr, 30.0, 30.0)
        out[str(mhz)] = r
        print(
            f"{mhz:.0f} MHz: F = {r['F_s']:.3f} s, T = {r['T_s']:.3f} s (ARM 30 + link 30 ms assumed), "
            f"2T margin vs 1.633 s: {r['margin_1_in_flight_s']:+.3f} s, {r['commands']} records"
        )
        for k, v in r["by_op"].items():
            print(f"   {k:14s} {v['cmds']:5d} records  {v['ms']:7.1f} ms  {v['GB']:.2f} GB")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
