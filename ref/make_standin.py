"""Generate stand-in weights and golden vectors in the exact handoff format.

Until trained weights exist, the hardware is verified against random weights
with the real shapes. Shifts are calibrated per layer so activations land in a
useful range (mostly unsaturated, some zeros from ReLU, a little clamping).

    python ref/make_standin.py --out artifacts/standin --seed 0

Writes weights.bin, manifest.json and vectors/NNN_in.bin, NNN_out.bin,
NNN_layerK.bin (K = 1..8, first 5 vectors only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from intref import (
    AUX_LEN,
    IMG_C,
    IMG_H,
    IMG_W,
    LAYERS,
    Model,
    PACKET_BYTES,
    conv_acc,
    fc_acc,
    forward,
    requant,
    unpack_input,
    weight_shape,
)

N_VECTORS = 100
N_LAYER_DUMPS = 5


def make_aux(rng: np.random.Generator, instr: int | None = None) -> np.ndarray:
    joints = rng.integers(-127, 128, 6)
    onehot = np.zeros(4, dtype=np.int64)
    onehot[rng.integers(0, 4) if instr is None else instr] = 127
    return np.concatenate([joints, onehot]).astype(np.int8)


def make_scene(rng: np.random.Generator) -> np.ndarray:
    """A crude overhead view: table, two bins, a red and a blue cube, an arm blob."""
    img = np.empty((IMG_H, IMG_W, IMG_C), dtype=np.float64)
    img[:] = rng.uniform(140, 190) + rng.normal(0, 4, 3)  # table, slight tint
    for x0 in (4, 64):  # left and right bins
        img[60:92, x0 : x0 + 28] = rng.uniform(50, 80)
        img[63:89, x0 + 3 : x0 + 25] = rng.uniform(90, 110)
    for color in ([200, 30, 30], [30, 40, 200]):  # red cube, blue cube
        y, x = rng.integers(8, 52), rng.integers(8, 84)
        img[y : y + 6, x : x + 6] = np.array(color) + rng.normal(0, 10, 3)
    y, x = rng.integers(0, 40), rng.integers(20, 70)  # arm / gripper blob
    img[y : y + 18, x : x + 8] = rng.uniform(200, 240)
    img += rng.normal(0, 3, img.shape) * rng.uniform(0.5, 2)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def packet(img: np.ndarray, aux: np.ndarray) -> bytes:
    assert img.shape == (IMG_H, IMG_W, IMG_C) and img.dtype == np.uint8
    assert aux.shape == (AUX_LEN,) and aux.dtype == np.int8
    return img.tobytes() + aux.tobytes()


def make_inputs(rng: np.random.Generator) -> list[bytes]:
    z = np.zeros(AUX_LEN, np.int8)
    full = lambda v: np.full((IMG_H, IMG_W, IMG_C), v, np.uint8)  # noqa: E731
    noise = lambda: rng.integers(0, 256, (IMG_H, IMG_W, IMG_C), dtype=np.uint8)  # noqa: E731

    vecs = [
        bytes(PACKET_BYTES),  # 000 all zero bytes (image becomes -128 everywhere)
        packet(full(255), np.full(AUX_LEN, 127, np.int8)),  # 001 all max
        packet(full(128), z),  # 002 image is exactly 0 after the offset: biases only
        packet(make_scene(rng), make_aux(rng)),  # 003 scene
        packet(noise(), make_aux(rng)),  # 004 noise
    ]
    vecs += [packet(make_scene(rng), make_aux(rng, i % 4)) for i in range(55)]  # 005..059
    vecs += [packet(noise(), make_aux(rng)) for _ in range(30)]  # 060..089

    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    checker = (((yy + xx) % 2) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    grad = np.stack([yy * 255 // 95, xx * 255 // 95, (yy + xx) * 255 // 190], -1).astype(np.uint8)
    corners = full(128)
    corners[0, 0], corners[0, -1], corners[-1, 0], corners[-1, -1] = 255, 0, 255, 0
    edges = [
        packet(checker, z),  # 090
        packet(255 - checker, z),  # 091
        packet(grad, z),  # 092
        packet(grad[::-1, ::-1].copy(), z),  # 093
        packet(corners, z),  # 094 border/padding handling
        packet(full(128), np.full(AUX_LEN, -127, np.int8)),  # 095 aux min
        packet(full(128), np.full(AUX_LEN, 127, np.int8)),  # 096 aux max
        packet(full(0), np.full(AUX_LEN, 127, np.int8)),  # 097
        packet(full(255), np.full(AUX_LEN, -127, np.int8)),  # 098
        packet(make_scene(rng), np.array([0] * 6 + [0, 0, 0, 127], np.int8)),  # 099
    ]
    vecs += edges
    assert len(vecs) == N_VECTORS and all(len(v) == PACKET_BYTES for v in vecs)
    return vecs


def make_model(rng: np.random.Generator, calib: list[bytes]) -> Model:
    """Random int8 weights, then per layer pick bias and shift from calibration activations."""
    xs = [unpack_input(p) for p in calib]
    acts = [img for img, _ in xs]
    auxs = [aux for _, aux in xs]
    layers, params = [], []
    for spec in LAYERS:
        w = np.clip(np.round(rng.normal(0, 40, weight_shape(spec))), -127, 127).astype(np.int8)
        w.flat[0], w.flat[1] = 127, -127  # make sure the extremes are exercised
        if spec["name"] == "fc6":
            acts = [np.concatenate([a.reshape(-1), aux]) for a, aux in zip(acts, auxs)]

        zero_b = np.zeros(w.shape[0], np.int32)
        acc_fn = conv_acc if spec["type"] == "conv" else fc_acc

        def raw_acc(x: np.ndarray) -> np.ndarray:
            return acc_fn(x, w, zero_b)

        accs = np.concatenate([raw_acc(a).reshape(-1, w.shape[0]) for a in acts])
        b = np.round(rng.normal(0, 0.3 * accs.std() + 1, w.shape[0])).astype(np.int32)
        p = np.percentile(np.abs(accs + b), 99.5)
        s = int(max(1, np.ceil(np.log2(max(p, 1) / 127))))
        lo = 0 if spec["relu"] else -127
        acts = [requant(raw_acc(a) + b, s, lo, 127) for a in acts]
        layers.append(dict(spec, shift=s))
        params.append((w, b))
    return Model(layers, params)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/standin")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    vecs = make_inputs(rng)
    model = make_model(rng, vecs[3:60])  # calibrate on scenes and noise, not the edge cases
    out = Path(args.out)
    manifest = model.save(out, note=f"STAND-IN random weights, seed {args.seed}. Not a trained model.")

    model = Model.load(out)  # round-trip through the files the hardware will see
    vdir = out / "vectors"
    vdir.mkdir(exist_ok=True)
    stats = np.zeros((len(LAYERS), 3))  # fraction zero, fraction saturated, mean
    for i, p in enumerate(vecs):
        y, layer_outs = forward(p, model)
        (vdir / f"{i:03d}_in.bin").write_bytes(p)
        (vdir / f"{i:03d}_out.bin").write_bytes(y.astype(np.int8).tobytes())
        for k, a in enumerate(layer_outs):
            if i < N_LAYER_DUMPS:
                (vdir / f"{i:03d}_layer{k + 1}.bin").write_bytes(a.astype(np.int8).tobytes())
            if 3 <= i < 60:
                stats[k] += [(a == 0).mean(), (np.abs(a) == 127).mean(), a.astype(float).mean()]

    print(f"wrote {out}/weights.bin ({manifest['total_bytes']:,} bytes), manifest.json, {N_VECTORS} vectors")
    print("layer   shift  zero%  sat%   mean   (over scene/noise vectors)")
    for layer, (z, s, m) in zip(manifest["layers"], stats / 57):
        print(f"{layer['name']:<7} {layer['shift']:>5}  {100 * z:5.1f}  {100 * s:4.1f}  {m:6.1f}")


if __name__ == "__main__":
    main()
