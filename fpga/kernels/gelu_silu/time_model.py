"""Time per 50-action chunk of the gelu_silu kernel (checkpoint -12392), from the csynth cycle counts.

Per call: rows * (ceil(K / LANES) + O_ROW) + tail, where O_ROW is the csynth 'rows' iteration latency minus the
'words' trip count of pass 1 (the slower process; pass 2 overlaps it), and tail = pass 2 of the last row + the scale
latency. csynth models the m_axi read latency as 0, so --ddr-lat adds a per-row restart (an assumption, not
measured), and --ddr-gbs bounds each call by its DDR bytes (also an assumption: the engine notes ~14 GB/s).

    python time_model.py [--lanes 8] [--orow 31] [--mhz 250] [--ddr-lat 40] [--ddr-gbs 12]
"""

from __future__ import annotations

import argparse

# (name, mode, calls per chunk, rows per call, K): shapes measured in policy_check.py (kernel stats) and
# fpga/kernels/sequencer/build_smolvla.py (vision batches both images: 512 rows per call)
CALLS = [
    ("SigLIP fc1 -> GELU", 0, 12, 512, 3072),
    ("LM gate|up -> SiLU*up", 1, 16, 81, 2560),
    ("expert gate|up -> SiLU*up", 1, 160, 50, 2048),
    ("action-time MLP -> SiLU", 2, 10, 50, 720),
]
SCALE_LAT = 69  # gs_scale 'rows' iteration latency (csynth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--orow", type=int, default=31, help="per-row overhead of pass 1 (csynth: 415 - 384 = 31)")
    ap.add_argument("--mhz", type=float, default=250.0)
    ap.add_argument("--ddr-lat", type=int, default=0, help="extra cycles per row for the DDR read restart (assumed)")
    ap.add_argument("--ddr-gbs", type=float, default=0.0, help="DDR bandwidth bound in GB/s (0: none; assumed)")
    a = ap.parse_args()
    tot_c = tot_b = 0.0
    tot_t = 0.0
    print(f"lanes {a.lanes}, {a.mhz:.0f} MHz, O_ROW {a.orow} + DDR restart {a.ddr_lat} cycles/row, "
          f"DDR bound {a.ddr_gbs or 'none'} GB/s")
    print(f"{'call':28s} {'calls':>5s} {'rows':>4s} {'K':>5s} {'elements':>11s} {'Mcycles':>8s} {'MB':>7s} {'ms':>7s}")
    for name, mode, calls, rows, K in CALLS:
        per_row = -(-K // a.lanes) + a.orow + a.ddr_lat
        tail = -(-K // a.lanes) + 23 + SCALE_LAT
        cyc = calls * (rows * per_row + tail)
        el = calls * rows * K
        byts = el * ((8 if mode == 1 else 4) + 2) + calls * rows * 4
        t = cyc / (a.mhz * 1e6)
        if a.ddr_gbs:
            t = max(t, byts / (a.ddr_gbs * 1e9))
        tot_c += cyc
        tot_b += byts
        tot_t += t
        print(f"{name:28s} {calls:5d} {rows:4d} {K:5d} {el:11,d} {cyc / 1e6:8.3f} {byts / 1e6:7.1f} {t * 1e3:7.2f}")
    print(f"{'total':28s} {'':5s} {'':4s} {'':5s} {'':11s} {tot_c / 1e6:8.3f} {tot_b / 1e6:7.1f} {tot_t * 1e3:7.2f}")


if __name__ == "__main__":
    main()
