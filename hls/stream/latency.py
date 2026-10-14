"""Latency model for SmolVLA on the full-precision engine (issue #148).

Reads the executed linear layers from scripts/smolvla_fpga.py's summary.json and estimates the FPGA
time per action chunk. Per call, the engine streams every weight once per NT-token pass (BF16) and
does TP * 32 FP32 multiply-adds per clock; with ping-pong tiles, load and compute overlap, so each
call takes max(weight bytes / bandwidth, MACs / MAC rate). Activations in and out are added.

Defaults are the design targets in plans/smolvla-fpga.md: 300 MHz, 16.2 GB/s over HP0-HP3
([Li2025-KV260LLM]), TP = 2 (64 FP32 MACs/clock), NT = 32.

    python hls/stream/latency.py runs/smolvla-fpga/summary.json [--tp 4 --mhz 300 --gbps 16.2]
"""

import argparse
import json
import math


def model(layers, mhz, gbps, tp, nt=32, lanes=32):
    rate_mac = tp * lanes * mhz * 1e6
    bw = gbps * 1e9
    total = mem_s = comp_s = 0.0
    weight_bytes_streamed = 0
    for L in layers:
        mp = math.ceil(L["M"] / lanes) * lanes
        k = L["K"]
        tok_per_call = L["tokens"] / L["calls"]
        passes = math.ceil(tok_per_call / nt)
        w_bytes = passes * mp * k * 2
        act_bytes = tok_per_call * (k + mp) * 4
        t_mem = (w_bytes + act_bytes) / bw
        t_comp = tok_per_call * mp * k / rate_mac
        total += L["calls"] * max(t_mem, t_comp)
        mem_s += L["calls"] * t_mem
        comp_s += L["calls"] * t_comp
        weight_bytes_streamed += L["calls"] * w_bytes
    return {
        "seconds_per_chunk": total,
        "memory_bound_seconds": mem_s,
        "compute_bound_seconds": comp_s,
        "weight_GB_streamed_per_chunk": weight_bytes_streamed / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--mhz", type=float, default=300)
    ap.add_argument("--gbps", type=float, default=16.2)
    ap.add_argument("--tp", type=int, default=2)
    args = ap.parse_args()
    s = json.load(open(args.summary))
    layers = s["layers"]
    out = {
        "linear_layers": len(layers),
        "linear_calls_per_chunk": sum(L["calls"] for L in layers),
        "bf16_weight_GB": s["linear_weight_bytes_bf16"] / 1e9,
        "macs_per_chunk_G": s["linear_macs_per_chunk"] / 1e9,
    }
    for tp in sorted({1, 2, 4, 8, args.tp}):
        r = model(layers, args.mhz, args.gbps, tp)
        out[f"tp{tp}"] = {k: round(v, 4) for k, v in r.items()} | {"fp32_macs_per_clock": tp * 32}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
