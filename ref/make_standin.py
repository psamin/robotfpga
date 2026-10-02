"""Stand-in weights and golden vectors, until trained weights exist.

    python ref/make_standin.py --out artifacts/standin --seed 0

Random int8 weights with the real shapes. Biases and shifts are calibrated per
layer so activations land in a useful range: mostly unsaturated, some zeros from
ReLU, a little clamping. "Frames" are synthetic overhead scenes. Output is the
exact handoff format (ref/README.md), so the hardware does not care that the
weights mean nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from intref import (
    IMG_C,
    IMG_H,
    IMG_W,
    LAYERS,
    Model,
    conv_acc,
    fc_acc,
    forward,
    requant,
    unpack_input,
    weight_shape,
)
from vectors import N_VECTORS, make_vector_set, packet, write_vectors

N_FRAMES = 58


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


def make_frame(rng: np.random.Generator, instr: int) -> bytes:
    onehot = np.zeros(4, np.int64)
    onehot[instr] = 127
    aux = np.concatenate([rng.integers(-127, 128, 6), onehot]).astype(np.int8)
    return packet(make_scene(rng), aux)


def make_model(rng: np.random.Generator, calib: list[bytes]) -> Model:
    """Random int8 weights, then per layer pick bias and shift from calibration activations."""
    xs = [unpack_input(p) for p in calib]
    acts = [img for img, _ in xs]
    layers, params = [], []
    for spec in LAYERS:
        w = np.clip(np.round(rng.normal(0, 40, weight_shape(spec))), -127, 127).astype(np.int8)
        w.flat[0], w.flat[1] = 127, -127  # make sure the extremes are exercised
        if spec["name"] == "fc6":
            acts = [np.concatenate([a.reshape(-1), aux]) for a, (_, aux) in zip(acts, xs)]
        acc_fn = conv_acc if spec["type"] == "conv" else fc_acc
        zero_b = np.zeros(w.shape[0], np.int32)
        accs = [acc_fn(a, w, zero_b) for a in acts]
        flat = np.concatenate([a.reshape(-1, w.shape[0]) for a in accs])
        b = np.round(rng.normal(0, 0.3 * flat.std() + 1, w.shape[0])).astype(np.int32)
        s = int(max(1, np.ceil(np.log2(max(np.percentile(np.abs(flat + b), 99.5), 1) / 127))))
        acts = [requant(a + b, s, 0 if spec["relu"] else -127, 127) for a in accs]
        layers.append(dict(spec, shift=s))
        params.append((w, b))
    return Model(layers, params)


def build_standin(out: str | Path, seed: int = 0) -> tuple[Model, list[bytes]]:
    rng = np.random.default_rng(seed)
    frames = [make_frame(rng, i % 4) for i in range(N_FRAMES)]
    vecs = make_vector_set(frames, rng)
    model = make_model(rng, frames)
    model.save(out, note=f"STAND-IN random weights, seed {seed}. Not a trained model.")
    model = Model.load(out)  # round-trip through the files the hardware will see
    write_vectors(model, vecs, out)
    return model, frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/standin")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model, frames = build_standin(args.out, args.seed)

    stats = np.zeros((len(LAYERS), 3))  # fraction zero, fraction saturated, mean
    for p in frames:
        for k, a in enumerate(forward(p, model)[1]):
            stats[k] += [(a == 0).mean(), (np.abs(a) == 127).mean(), a.astype(float).mean()]
    print(f"wrote {args.out}: weights.bin, manifest.json, {N_VECTORS} vectors")
    print("layer   shift  zero%  sat%   mean   (over the frames)")
    for layer, (z, s, m) in zip(model.layers, stats / len(frames)):
        print(f"{layer['name']:<7} {layer['shift']:>5}  {100 * z:5.1f}  {100 * s:4.1f}  {m:6.1f}")


if __name__ == "__main__":
    main()
