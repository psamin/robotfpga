"""Load the engine bitstream once with PYNQ (root, its Python 3.10) and exit; it stays programmed.

    sudo /usr/local/share/pynq-venv/bin/python3 fpga/board/load_overlay.py --bit float_tp8.bit --clock-mhz 250

Prints the IP base address the in-process engine (runtime.Engine(backend="xrt")) uses.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fpga.board.runtime import IP_BASE, set_pl_clock  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit", required=True)
    ap.add_argument("--clock-mhz", type=float, default=None)
    args = ap.parse_args()
    from pynq import Overlay

    ol = Overlay(args.bit)
    mhz = set_pl_clock(args.clock_mhz) if args.clock_mhz else None
    base = ol.ip_dict["float_top_0"]["phys_addr"]
    if base != IP_BASE:
        print(f"WARNING: float_top_0 is at {base:#x}, runtime.IP_BASE says {IP_BASE:#x}", flush=True)
    print(
        f"overlay loaded: {args.bit}, float_top_0 at {base:#x}, PL clock {mhz or 'unchanged'} MHz", flush=True
    )
    print("overlay ready", flush=True)


if __name__ == "__main__":
    main()
