"""Engine calls, MACs, DDR words and time per 50-action chunk of SmolVLA -12392's attention on the int16 engine.

Shapes: check_policy.py's run (results/policy_check.json, int16 variant, frame 0; the same on every frame of this
checkpoint: 45 of the 81 prefix tokens are valid). Engine timing from the structure of fpga/hls/int16_linear.h
(32 tokens x 32 lanes = 1,024 MACs per clock, 32-token blocks, 64-token tiles, weights re-used over the 2 blocks of
a tile, emit/dequant 16 values per clock, Y stored at one 128-bit word = 4 FP32 per clock on io, X read on io),
per head iteration:
    mac   = groups * blocks * max(K, 8)                      the MAC loop alone
    model = max(groups * blocks * max(K, 64), Y words, X words, W words per port) + X tile-0 load + HEAD_OVH
            (a block shorter than 64 clocks waits for the emit; Y = M * Np / 4 words is the usual bound: every
            attention call has K <= 256)
HEAD_OVH (dataflow restart + first DDR latency per head iteration) and the per-call host cost are assumptions,
not measurements (README.md "Cost").  attn_pack cycles from attn_pack.h's loops (one input word per clock; csynth
II = 1), DDR single-beat efficiency not modelled.

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga\\kernels\\attention\\count_cost.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = 250e6
TB, LANES, MACS = 32, 32, 1024
HEAD_OVH = 200  # clocks per head iteration (assumption)
DESC_OVH = 24 + 256 + 64  # attn_pack per descriptor: read, valid-row load (NMAX), dataflow fill (estimate)
CALL_S = {"driver_0.1ms": 1e-4, "sequencer_0.02ms": 2e-5}  # host cost per kernel call (brief's assumptions)
STEPS, LAYERS_V, LAYERS_LM, IMAGES = 10, 12, 16, 2


def ceil(a, b):
    return -(-a // b)


def pad(a, b):
    return ceil(a, b) * b


def engine_head(M, K, N):
    """One head iteration of W M x K (M padded to 32) against X N x K."""
    Mp, Np = pad(M, LANES), pad(N, TB)
    g, nb = Mp // LANES, Np // TB
    nt = ceil(nb, 2)
    mac = g * nb * max(K, 8)
    y = Mp * Np // 4
    x = Np * K // 8 + Np // 4 + Mp // 4
    w = nt * g * K
    x0 = min(nb, 2) * K * 4 + 16
    model = max(g * nb * max(K, 64), y, x, w) + x0 + HEAD_OVH
    return dict(
        mac_cycles=mac, model_cycles=model, y_words=y, x_words=x, w_words=4 * w, padded_macs=g * nb * K * MACS
    )


def attention_instance(H, Hkv, Tq, Tk, images=1):
    """Q K^T + P V of one attention (all heads; `images` batched as extra head iterations)."""
    G = H // Hkv
    iters = Hkv * images
    qk = engine_head(Tk, 64, G * Tq)
    pv = engine_head(64, Tk, G * Tq)
    r = {k: iters * (qk[k] + pv[k]) for k in qk}
    r["useful_macs"] = 2 * images * H * Tq * Tk * 64
    r["head_iters"] = 2 * iters
    r["softmax_elems"] = images * H * Tq * Tk
    r["shapes"] = dict(
        qk=dict(M=pad(Tk, 32), K=64, N=G * Tq, H=iters), pv=dict(M=64, K=Tk, N=G * Tq, H=iters)
    )
    return r


def pack_cycles(H, Hkv, Tq, Tk, images=1, kv=True, q_and_o=True):
    """attn_pack clocks: QX / KW one read (16 clocks per 64-value row, writer 520 per 512), VW / OX two reads."""
    G = H // Hkv
    c = 0
    if q_and_o:
        c += images * (Hkv * (pad(G * Tq, 32) * 16 * 520 // 512 + 64) + DESC_OVH)  # QX
        c += images * (2 * pad(Tq, 32) * H * 16 + pad(Tq, 32) + 64 + DESC_OVH)  # OX
    if kv:
        c += images * (Hkv * (pad(Tk, 32) * 16 * 520 // 512 + 64) + DESC_OVH)  # KW
        c += images * (
            Hkv * (2 * Tk * 16 + 64 * 14 + 64) + DESC_OVH
        )  # VW: the 64 channel scales first, 14 clk each
    return c


def sites(shapes, compact: bool):
    v, p, es, ec = (shapes[k] for k in ("vision", "prefix", "expert_self", "expert_cross"))
    pre = p["valid_keys"] if compact else p["Tk"]
    suf = es["Tq"]
    return {
        "vision": dict(n=LAYERS_V, H=v["H"], Hkv=v["Hkv"], Tq=v["Tq"], Tk=v["Tk"], images=IMAGES),
        "prefix": dict(n=LAYERS_LM, H=p["H"], Hkv=p["Hkv"], Tq=pre, Tk=pre, images=1),
        "expert_self": dict(
            n=LAYERS_LM // 2 * STEPS, H=es["H"], Hkv=es["Hkv"], Tq=suf, Tk=pre + suf, images=1
        ),
        "expert_cross": dict(
            n=LAYERS_LM // 2 * STEPS, H=ec["H"], Hkv=ec["Hkv"], Tq=ec["Tq"], Tk=pre, images=1
        ),
    }


def evaluate(st):
    rows, tot = {}, {}
    for name, s in st.items():
        a = attention_instance(s["H"], s["Hkv"], s["Tq"], s["Tk"], s["images"])
        n = s["n"]
        r = {k: n * a[k] for k in a if k != "shapes"}
        r["engine_calls"] = 2 * n
        r["softmax_calls"] = n
        if name == "expert_cross":  # K / V depend on the prefix only: packed once per chunk per layer
            r["pack_cycles"] = n * pack_cycles(s["H"], s["Hkv"], s["Tq"], s["Tk"], kv=False) + (
                LAYERS_LM // 2
            ) * pack_cycles(s["H"], s["Hkv"], s["Tq"], s["Tk"], q_and_o=False)
            r["pack_calls"] = 2 * n + LAYERS_LM // 2
        else:
            r["pack_cycles"] = n * pack_cycles(s["H"], s["Hkv"], s["Tq"], s["Tk"], s["images"])
            r["pack_calls"] = 2 * n
        r["repeats"] = n
        r["shape"] = dict(
            H=s["H"],
            Hkv=s["Hkv"],
            G=s["H"] // s["Hkv"],
            Tq=s["Tq"],
            Tk=s["Tk"],
            images=s["images"],
            **a["shapes"],
        )
        r["ms_mac"] = r["mac_cycles"] / F * 1e3
        r["ms_model"] = r["model_cycles"] / F * 1e3
        r["ms_pack"] = r["pack_cycles"] / F * 1e3
        r["ms_ideal"] = r["useful_macs"] / MACS / F * 1e3
        rows[name] = r
        for k, val in r.items():
            if isinstance(val, (int, float)):
                tot[k] = tot.get(k, 0) + val
    tot["calls"] = tot["engine_calls"] + tot["softmax_calls"] + tot["pack_calls"]
    for k, c in CALL_S.items():
        tot[f"ms_calls_{k}"] = tot["calls"] * c * 1e3
    tot["ms_softmax_1word_per_clk"] = tot["softmax_elems"] / 4 / F * 1e3  # the softmax kernel's own rate
    tot.pop("repeats", None)
    return rows, tot


def main():
    res = json.loads((HERE / "results" / "policy_check.json").read_text())
    s0 = res["variants"]["int16"]["frames"][0]["sites"]
    shapes = {
        k: dict(**v["shape"], **{kk: v[kk] for kk in ("valid_keys", "valid_queries") if kk in v})
        for k, v in s0.items()
    }
    out = {}
    for compact in (False, True):
        rows, tot = evaluate(sites(shapes, compact))
        key = "compacted_prefix" if compact else "as_run"
        out[key] = dict(sites=rows, total=tot)
        print(f"\n== {key} (prefix {'45 valid tokens' if compact else 'all 81 tokens incl. padding'})")
        print(
            f"{'site':13s} {'eng':>4s} {'iters':>6s} {'GMAC':>6s} {'ideal':>6s} {'mac ms':>7s} {'model':>7s} "
            f"{'Y Mw':>6s} {'pack ms':>7s}"
        )
        for n, r in list(rows.items()) + [("total", tot)]:
            print(
                f"{n:13s} {r['engine_calls']:4d} {r['head_iters']:6d} {r['useful_macs'] / 1e9:6.3f} "
                f"{r['ms_ideal']:6.2f} {r['ms_mac']:7.2f} {r['ms_model']:7.2f} {r['y_words'] / 1e6:6.2f} "
                f"{r['ms_pack']:7.2f}"
            )
        print(
            f"calls per chunk: {tot['engine_calls']} engine + {tot['softmax_calls']} softmax + {tot['pack_calls']} "
            f"pack = {tot['calls']}: {tot['ms_calls_driver_0.1ms']:.1f} ms at 0.1 ms/call, "
            f"{tot['ms_calls_sequencer_0.02ms']:.1f} ms at 0.02 ms/call; softmax elements "
            f"{tot['softmax_elems'] / 1e6:.2f} M ({tot['ms_softmax_1word_per_clk']:.1f} ms at 4 per clock)"
        )
    (HERE / "results" / "cost.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {HERE / 'results' / 'cost.json'}")


if __name__ == "__main__":
    main()
