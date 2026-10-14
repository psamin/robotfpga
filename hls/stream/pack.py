"""Pack an int8 policy for the DDR weight-streaming engine (issue #148), plus a generic reference.

Layout written to OUT/:
  table.bin   n_layers x 64-byte LayerDesc (16 little-endian int32, see stream_engine.h)
  words.bin   weight words, 16 bytes each: word[g][k][lane] = W[16*g + lane][k]
              (k = (ky*3 + kx)*Cin + ci for a conv, the input index for an FC)
  bias.bin    int32 little-endian, layer after layer
  vectors/    NNN_in.bin (27,658 bytes) and NNN_out.bin (48 bytes), optional

Usage:
  python hls/stream/pack.py --model handoff/v3r2 --out build/stream/v3r2 --vectors handoff/v3r2/vectors
  python hls/stream/pack.py --random tinypolicy-l --out build/stream/tpl --n-vectors 20
  python hls/stream/pack.py --linear-test --out build/stream/linear   # random GEMM case
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

LANES = 16
REF = Path(__file__).resolve().parents[2] / "ref" / "intref.py"
SHAPES = {  # conv widths (5) and FC widths (fc6, fc7); fc8 is always 48
    "v3r2": ([16, 32, 64, 96, 128], [256, 256]),
    "tinypolicy-l": ([32, 64, 128, 192, 256], [384, 384]),
}


def load_intref():
    spec = importlib.util.spec_from_file_location("intref", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def layer_list(convs, fcs):
    """Layer dicts in the intref style for arbitrary widths."""
    layers, h, c = [], 96, 3
    for i, co in enumerate(convs):
        layers.append(
            {
                "name": f"conv{i + 1}",
                "type": "conv",
                "in_shape": [h, h, c],
                "out_shape": [h // 2, h // 2, co],
                "relu": True,
            }
        )
        h, c = h // 2, co
    n = h * h * c + 10
    for i, f in enumerate(fcs + [48]):
        layers.append(
            {"name": f"fc{6 + i}", "type": "fc", "in_shape": [n], "out_shape": [f], "relu": i < len(fcs)}
        )
        n = f
    return layers


def weight_shape(layer):
    if layer["type"] == "conv":
        return (layer["out_shape"][2], 3, 3, layer["in_shape"][2])
    return (layer["out_shape"][0], layer["in_shape"][0])


# ---- reference: same integer math as ref/intref.py, any widths ----
def requant(acc, s, relu):
    y = (acc.astype(np.int64) + (1 << (s - 1))) >> s
    return np.clip(y, 0 if relu else -127, 127).astype(np.int8)


def forward(packet: bytes, layers, params):
    p = np.frombuffer(packet, np.uint8)
    x = (p[:27648].astype(np.int16) - 128).astype(np.int8).reshape(96, 96, 3)
    aux = p[27648:].view(np.int8)
    outs = []
    for layer, (w, b) in zip(layers, params, strict=True):
        if layer["type"] == "conv":
            h, wd, _ = x.shape
            xp = np.zeros((h + 2, wd + 2, x.shape[2]), np.int64)
            xp[1:-1, 1:-1] = x
            acc = np.broadcast_to(b.astype(np.int64), (h // 2, wd // 2, w.shape[0])).copy()
            for ky in range(3):
                for kx in range(3):
                    acc += xp[ky : ky + h : 2, kx : kx + wd : 2] @ w[:, ky, kx, :].astype(np.int64).T
            x = requant(acc, layer["shift"], True)
        else:
            if layer["name"] == "fc6":
                x = np.concatenate([x.reshape(-1), aux])
            x = requant(w.astype(np.int64) @ x.astype(np.int64) + b, layer["shift"], layer["relu"])
        outs.append(x)
    return x.reshape(-1), outs


# ---- packing ----
def pack(layers, params, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    table, words, biases, w_off, b_off = [], [], [], 0, 0
    for layer, (w, b) in zip(layers, params, strict=True):
        co = w.shape[0]
        if co % LANES:
            raise ValueError(f"{layer['name']}: {co} output channels is not a multiple of {LANES}")
        mat = w.reshape(co, -1)  # [out][k], k order = (ky, kx, ci) for conv, matching intref
        k_len = mat.shape[1]
        grouped = mat.reshape(co // LANES, LANES, k_len).transpose(0, 2, 1)  # [g][k][lane]
        words.append(np.ascontiguousarray(grouped).astype(np.int8).reshape(-1))
        biases.append(b.astype("<i4"))
        conv = layer["type"] == "conv"
        h = layer["in_shape"][0] if conv else 1
        cin = layer["in_shape"][2] if conv else layer["in_shape"][0]
        desc = [
            0 if conv else 1,
            h,
            h,
            cin,
            co,
            layer["shift"],
            int(layer["relu"]),
            w_off,
            b_off,
            int(layer["name"] == "fc6"),
        ] + [0] * 6
        table.append(np.array(desc, "<i4"))
        w_off += (co // LANES) * k_len
        b_off += co
    np.concatenate(table).tofile(out / "table.bin")
    np.concatenate(words).tofile(out / "words.bin")
    np.concatenate(biases).tofile(out / "bias.bin")
    meta = {
        "n_layers": len(layers),
        "weight_words": w_off,
        "weight_bytes": 16 * w_off,
        "biases": b_off,
        "layers": [{k: v for k, v in L.items()} for L in layers],
    }
    (out / "stream.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def random_model(name, rng):
    convs, fcs = SHAPES[name]
    layers, params = layer_list(convs, fcs), []
    for layer in layers:
        w = rng.integers(-127, 128, weight_shape(layer)).astype(np.int8)
        params.append((w, rng.integers(-2000, 2000, w.shape[0]).astype(np.int32)))
    # per layer, the smallest shift that saturates under 2% of outputs on one random packet
    pkt = rng.integers(0, 256, 27658, dtype=np.uint8).tobytes()
    for i, layer in enumerate(layers):
        for s in range(1, 31):
            layer["shift"] = s
            y, _ = forward(pkt, layers[: i + 1], params[: i + 1])
            if np.mean(np.abs(y.astype(int)) == 127) < 0.02:
                break
    return layers, params


def write_vectors(layers, params, packets, vdir: Path):
    vdir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(packets):
        y, _ = forward(p, layers, params)
        (vdir / f"{i:03d}_in.bin").write_bytes(p)
        (vdir / f"{i:03d}_out.bin").write_bytes(y.astype(np.int8).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="intref model dir (manifest.json + weights.bin)")
    ap.add_argument("--vectors", help="existing vectors dir to copy and check against")
    ap.add_argument("--random", choices=sorted(SHAPES), help="random weights with these widths")
    ap.add_argument("--n-vectors", type=int, default=20)
    ap.add_argument("--linear-test", action="store_true", help="random int8 GEMM for linear mode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, rng = Path(args.out), np.random.default_rng(args.seed)

    if args.linear_test:
        out.mkdir(parents=True, exist_ok=True)
        for m, k, n in [(64, 960, 7), (2560, 960, 70), (960, 2560, 3)]:
            x = rng.integers(-127, 128, (n, k)).astype(np.int8)
            w = rng.integers(-127, 128, (m, k)).astype(np.int8)
            write_linear_case(out / f"case_{m}x{k}x{n}", w, x)
        print(f"wrote linear test cases to {out}")
        return

    if args.model:
        ref = load_intref()
        model = ref.Model.load(args.model)
        layers, params = model.layers, model.params
    else:
        layers, params = random_model(args.random, rng)
    meta = pack(layers, params, out)
    if args.vectors:  # check the generic reference against the committed golden vectors, then copy them
        vin = sorted(Path(args.vectors).glob("*_in.bin"))
        for f in vin:
            y, _ = forward(f.read_bytes(), layers, params)
            want = (f.parent / f.name.replace("_in", "_out")).read_bytes()
            assert y.astype(np.int8).tobytes() == want, f"reference mismatch on {f.name}"
        shutil.copytree(args.vectors, out / "vectors", dirs_exist_ok=True)
        print(f"reference matches all {len(vin)} golden vectors")
    else:
        pkts = [rng.integers(0, 256, 27658, dtype=np.uint8).tobytes() for _ in range(args.n_vectors)]
        write_vectors(layers, params, pkts, out / "vectors")
    print(f"packed {meta['n_layers']} layers, {meta['weight_bytes']:,} weight bytes -> {out}")


def write_linear_case(d: Path, w, x):
    """W [M][K] and X [N][K] int8 -> words.bin (packed W), x.bin, y_ref.bin (int32 [N][M]), dims.txt."""
    d.mkdir(parents=True, exist_ok=True)
    m, k = w.shape
    grouped = w.reshape(m // LANES, LANES, k).transpose(0, 2, 1)
    np.ascontiguousarray(grouped).astype(np.int8).tofile(d / "words.bin")
    x.astype(np.int8).tofile(d / "x.bin")
    (x.astype(np.int64) @ w.astype(np.int64).T).astype("<i4").tofile(d / "y_ref.bin")
    (d / "dims.txt").write_text(f"{m} {k} {x.shape[0]}\n")


if __name__ == "__main__":
    main()
