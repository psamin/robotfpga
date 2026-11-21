"""Cycle model of the int16 engine (int16_linear.h) over SmolVLA's engine calls, per action chunk.

    .venv/Scripts/python.exe fpga/hls/model_int16.py                  # tables for -12392 and -9604
    .venv/Scripts/python.exe fpga/hls/model_int16.py --mhz 300 --ddr 10 --json out.json

One engine call = one K chunk (<= 4096) of one linear: M outputs (padded to 32: G groups), N tokens
(ceil(N / 32) blocks of 32), tiles of NBX blocks. compute16 runs (tile, group, block, k) with k innermost, one
k of 32 tokens x 32 lanes per clock. Per tile t (nbt blocks) the model takes the slowest of

    compute   G * nbt * max(K, 256)   a block is K clocks; its 4 KB of Y leave over the 16-byte io write
                                      port in 256 clocks (the 16-wide emit/dequantizer needs 70)
    ports     G * K                   every port delivers one 128-bit weight word per clock at most
    DDR       bytes / (GB/s / f)      weights 64 G K + Y 4096 G nbt + next tile's X 64 nbt' K, shared DDR
    X next    4 * nbt' * K            the next tile's X over the io read port (double buffer), 1 word/clock

plus, per call, L0 (start, first DDR latencies, pipeline fill; assumed 400 clocks), the first tile's X before
the first MAC (4 * nbt0 * K clocks: not overlapped) and the tail (last block's emit, dequantization and Y
store, ~450 clocks). The weight FIFOs hold a whole group (NBX >= 2), so a group's weights stream in while the
previous group is re-used: the per-tile max is the model, not a per-group max. The compute term is the C
model's clock counter for compute-bound calls (tb_float int16 mode prints compute16's clocks: G * nblk * K +
72 with NBX = 2); the DDR figure (14 GB/s) is the unmeasured literature value: DDR-bound rows scale with it.

Host time per call (the ARM's quantize/copy/start/wait, or a sequencer) is NOT in the engine time; the
report adds it separately for 0.1 ms (tightened driver) and 0.02 ms (command sequencer) per call.
"""

import argparse
import json
import math

LANES, TB, K_MAX, KMIN, EROWS, EDELAY, YW_BLK = 32, 32, 4096, 8, 64, 6, 256
L0, TAIL = 400, 450  # clocks per call: start + first reads + pipeline fill; last emit + dequant + store

# Engine calls per chunk as (part, M outputs, K inputs, N tokens, calls). The 21 FP32-stored layers
# (state_proj, action_in/out_proj, action_time_mlp_in/out, the expert's cross-attention k/v 320 x 320) stay
# on the ARM in FP32.
LAYERS_12392 = [  # 256-px checkpoint (build/brief/cd_inventory.json, frame 0)
    ("vision", 768, 768, 256, 96),  # q/k/v/out, 12 layers x 2 images
    ("vision", 3072, 768, 256, 24),  # fc1
    ("vision", 768, 3072, 256, 24),  # fc2
    ("connector", 960, 12288, 16, 2),  # 3 K chunks each
    ("lm", 960, 960, 81, 32),  # q, o
    ("lm", 320, 960, 81, 32),  # k, v
    ("lm", 2560, 960, 81, 32),  # gate, up
    ("lm", 960, 2560, 81, 16),  # down
    ("expert", 960, 720, 50, 160),  # q
    ("expert", 320, 720, 50, 160),  # self-attention k, v (8 layers x 10 steps each)
    ("expert", 720, 960, 50, 160),  # o
    ("expert", 2048, 720, 50, 320),  # gate, up
    ("expert", 720, 2048, 50, 160),  # down
]
FP32_12392 = [
    ("state_proj", 960, 32, 1, 1),
    ("action_in_proj", 720, 32, 50, 10),
    ("action_time_mlp_in", 720, 1440, 50, 10),
    ("action_time_mlp_out", 720, 720, 50, 10),
    ("action_out_proj", 32, 720, 50, 10),
    ("cross k/v", 320, 320, 81, 160),
]
LAYERS_9604 = (
    [  # 512-px checkpoint (build/brief/count_calls.json)
        ("vision", 768, 768, 1024, 96),
        ("vision", 3072, 768, 1024, 24),
        ("vision", 768, 3072, 1024, 24),
        ("connector", 960, 12288, 64, 2),
        ("lm", 960, 960, 177, 32),
        ("lm", 320, 960, 177, 32),
        ("lm", 2560, 960, 177, 32),
        ("lm", 960, 2560, 177, 16),
    ]
    + LAYERS_12392[8:]
)
FP32_9604 = FP32_12392[:5] + [("cross k/v", 320, 320, 177, 160)]


def call_clocks(m: int, k: int, n: int, nbx: int, ddr_b_per_clk: float) -> dict:
    """Clocks of one engine call (k <= 4096) and what bounds its tiles."""
    g = math.ceil(m / LANES)
    nblk = math.ceil(n / TB)
    tiles = [min(nbx, nblk - t * nbx) for t in range(math.ceil(nblk / nbx))]
    kp = max(k, KMIN)
    x0 = 4 * tiles[0] * k
    total, bound = L0 + x0 + TAIL, {}
    for t, nbt in enumerate(tiles):
        nxt = tiles[t + 1] if t + 1 < len(tiles) else 0
        terms = {
            "compute": g * nbt * max(kp, YW_BLK, EROWS + EDELAY),
            "ports": g * k,
            "ddr": (64 * g * k + 4096 * g * nbt + 64 * nxt * k) / ddr_b_per_clk,
            "xnext": 4 * nxt * k,
        }
        why = max(terms, key=terms.get)
        bound[why] = bound.get(why, 0) + terms[why]
        total += terms[why]
    return {
        "clocks": total,
        "x0": x0,
        "fixed": L0 + TAIL,
        "bound": bound,
        "macs_padded": g * 32 * nblk * 32 * k,
    }


def chunk_time(layers, nbx: int, mhz: float, ddr_gbps: float) -> dict:
    ddr = ddr_gbps * 1e9 / (mhz * 1e6)  # bytes per clock
    parts, calls, macs = {}, 0, 0
    for part, m, k, n, cnt in layers:
        nch = math.ceil(k / K_MAX)
        for c in range(nch):
            kc = min(K_MAX, k - c * K_MAX)
            r = call_clocks(m, kc, n, nbx, ddr)
            p = parts.setdefault(
                part, {"calls": 0, "macs": 0, "clocks": 0.0, "x0": 0, "fixed": 0, "bound": {}}
            )
            p["calls"] += cnt
            p["macs"] += cnt * m * kc * n
            p["clocks"] += cnt * r["clocks"]
            p["x0"] += cnt * r["x0"]
            p["fixed"] += cnt * r["fixed"]
            for b, v in r["bound"].items():
                p["bound"][b] = p["bound"].get(b, 0) + cnt * v
            calls += cnt
            macs += cnt * m * kc * n
    total = sum(p["clocks"] for p in parts.values())
    for p in parts.values():
        p["ms"] = p["clocks"] / (mhz * 1e3)
    return {
        "nbx": nbx,
        "mhz": mhz,
        "ddr_gbps": ddr_gbps,
        "calls": calls,
        "gmac": macs / 1e9,
        "engine_ms": total / (mhz * 1e3),
        "ideal_ms": macs / 1024 / (mhz * 1e3),
        "parts": parts,
    }


def detail(layers, nbx: int, mhz: float, ddr_gbps: float) -> None:
    """Clocks per (group, first tile) for each call shape: compute (blocks x max(K, 256)), weight load (K
    words per port), its DDR time (weights + Y of the tile-group), the X of the next tile spread over the
    groups, the emit (70 per block, overlapped), and per call the first tile's X and the whole call."""
    ddr = ddr_gbps * 1e9 / (mhz * 1e6)
    print(f"\n### NBX={nbx}: clocks per (group, first tile) at {mhz:.0f} MHz, DDR {ddr_gbps:.0f} GB/s")
    print(
        "| part | M x K x N | groups | tiles | compute | weight words | weight DDR | next X / group | emit "
        "| first X / call | call clocks | call us |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for part, m, k, n, _ in layers:
        kc = min(k, K_MAX)
        g, nblk = math.ceil(m / LANES), math.ceil(n / TB)
        tiles = [min(nbx, nblk - t * nbx) for t in range(math.ceil(nblk / nbx))]
        nbt, nxt = tiles[0], (tiles[1] if len(tiles) > 1 else 0)
        r = call_clocks(m, kc, n, nbx, ddr)
        print(
            f"| {part} | {m}x{k}x{n} | {g} | {len(tiles)} | {nbt * max(kc, YW_BLK)} | {kc} | "
            f"{(64 * kc + 4096 * nbt) / ddr:.0f} | {4 * nxt * kc / g:.0f} | {nbt * (EROWS + EDELAY)} | "
            f"{r['x0']} | {r['clocks']:.0f} | {r['clocks'] / mhz:.1f} |"
        )


def fp32_gmac(layers) -> float:
    return sum(m * k * n * cnt for _, m, k, n, cnt in layers) / 1e9


def report(name: str, layers, fp32, args) -> list:
    out = []
    print(
        f"\n## {name}: {fp32_gmac(layers):.2f} GMAC on the engine, {fp32_gmac(fp32):.2f} GMAC FP32 on the ARM"
    )
    for nbx in (1, 2):
        r = chunk_time(layers, nbx, args.mhz, args.ddr)
        out.append(r)
        host, seq = r["engine_ms"] + 0.1 * r["calls"], r["engine_ms"] + 0.02 * r["calls"]
        print(
            f"NBX={nbx} @ {args.mhz:.0f} MHz, DDR {args.ddr:.0f} GB/s: {r['calls']} engine calls, "
            f"{r['gmac']:.2f} GMAC, engine {r['engine_ms']:.1f} ms "
            f"(1,024 MAC/clock ideal {r['ideal_ms']:.1f} ms); "
            f"+ host 0.1 ms/call {host:.1f} ms, + sequencer 0.02 ms/call {seq:.1f} ms"
        )
        print("| part | calls | GMAC | engine ms | first-tile X ms | fixed ms | tile bounds (ms) |")
        print("|---|---|---|---|---|---|---|")
        for part, p in r["parts"].items():
            f = args.mhz * 1e3
            bounds = ", ".join(f"{b} {v / f:.1f}" for b, v in sorted(p["bound"].items(), key=lambda x: -x[1]))
            print(
                f"| {part} | {p['calls']} | {p['macs'] / 1e9:.2f} | {p['ms']:.1f} | {p['x0'] / f:.1f} | "
                f"{p['fixed'] / f:.1f} | {bounds} |"
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mhz", type=float, default=250.0)
    ap.add_argument("--ddr", type=float, default=14.0, help="DDR GB/s shared by all ports (unmeasured)")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    res = {
        "12392": report("-12392 (256 px)", LAYERS_12392, FP32_12392, args),
        "9604": report("-9604 (512 px)", LAYERS_9604, FP32_9604, args),
    }
    for nbx in (2, 1):
        detail(LAYERS_12392, nbx, args.mhz, args.ddr)
    print("\n## sensitivity (engine ms per chunk, NBX=1 / NBX=2)")
    print(
        "| ckpt | 250 MHz 10 GB/s | 250 MHz 14 GB/s | 250 MHz 16 GB/s | 300 MHz 14 GB/s | 300 MHz 16 GB/s |"
    )
    print("|---|---|---|---|---|---|")
    for name, layers in (("-12392", LAYERS_12392), ("-9604", LAYERS_9604)):
        cells = []
        for mhz, ddr in ((250, 10), (250, 14), (250, 16), (300, 14), (300, 16)):
            a, b = (chunk_time(layers, nbx, mhz, ddr)["engine_ms"] for nbx in (1, 2))
            cells.append(f"{a:.0f} / {b:.0f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
