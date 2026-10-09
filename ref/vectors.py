"""Golden test vectors for the FPGA team (format: ref/README.md).

    python ref/vectors.py --model DIR --packets packets.npy [--seed 0]

DIR holds manifest.json + weights.bin; vectors go to DIR/vectors/. packets.npy is
uint8 [N, 27658]: real frames packed with intref.pack_obs, N >= 58. From Python,
call make_vector_set() and write_vectors() directly, as make_standin.py does.

Layout of the 100 vectors:
  000 all-zero bytes, 001 all-max, 002 image 128 (zero after the offset: biases only),
  003 first frame, 004 noise                 <- these 5 also get every layer's output
  005-013 more edge cases (checkerboards, gradients, corner pixels, aux extremes)
  014-   frames and uniform-noise packets, interleaved
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from intref import AUX_LEN, IMG_C, IMG_H, IMG_W, PACKET_BYTES, Model, forward

N_VECTORS = 100
N_NOISE = 30
N_LAYER_DUMPS = 5


def packet(img: np.ndarray, aux: np.ndarray) -> bytes:
    assert img.shape == (IMG_H, IMG_W, IMG_C) and img.dtype == np.uint8
    assert aux.shape == (AUX_LEN,) and aux.dtype == np.int8
    return img.tobytes() + aux.tobytes()


def _full(v: int) -> np.ndarray:
    return np.full((IMG_H, IMG_W, IMG_C), v, np.uint8)


def _aux(v: int) -> np.ndarray:
    return np.full(AUX_LEN, v, np.int8)


def edge_case_packets() -> list[bytes]:
    """12 fixed inputs: extremes, biases only, padding and border patterns, aux extremes."""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    checker = (((yy + xx) % 2) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    grad = np.stack([yy * 255 // 95, xx * 255 // 95, (yy + xx) * 255 // 190], -1).astype(np.uint8)
    corners = _full(128)
    corners[0, 0], corners[0, -1], corners[-1, 0], corners[-1, -1] = 255, 0, 255, 0
    return [
        bytes(PACKET_BYTES),  # all zero bytes: image is -128 everywhere
        packet(_full(255), _aux(127)),  # all max
        packet(_full(128), _aux(0)),  # image exactly 0 after the offset: biases only
        packet(checker, _aux(0)),
        packet(255 - checker, _aux(0)),
        packet(grad, _aux(0)),
        packet(grad[::-1, ::-1].copy(), _aux(0)),
        packet(corners, _aux(0)),  # border and padding handling
        packet(_full(128), _aux(-127)),
        packet(_full(128), _aux(127)),
        packet(_full(0), _aux(127)),
        packet(_full(255), _aux(-127)),
    ]


def noise_packet(rng: np.random.Generator) -> bytes:
    img = rng.integers(0, 256, (IMG_H, IMG_W, IMG_C), dtype=np.uint8)
    onehot = np.zeros(4, np.int64)
    onehot[rng.integers(0, 4)] = 127
    return packet(img, np.concatenate([rng.integers(-127, 128, 6), onehot]).astype(np.int8))


def make_vector_set(frames: list[bytes], rng: np.random.Generator) -> list[bytes]:
    """The 100 inputs: edge cases, N_NOISE noise packets, and frames for the rest."""
    edges = edge_case_packets()
    noise = [noise_packet(rng) for _ in range(N_NOISE)]
    n_frames = N_VECTORS - len(edges) - N_NOISE
    if len(frames) < n_frames:
        raise ValueError(f"need at least {n_frames} frames, got {len(frames)}")
    frames, rest = list(frames[:n_frames]), []
    for i in range(max(len(frames) - 1, len(noise) - 1)):
        rest += frames[1 + i : 2 + i] + noise[1 + i : 2 + i]
    vecs = edges[:3] + [frames[0], noise[0]] + edges[3:] + rest
    assert len(vecs) == N_VECTORS and all(len(v) == PACKET_BYTES for v in vecs)
    return vecs


def write_vectors(model: Model, packets: list[bytes], out_dir: str | Path) -> Path:
    """Writes NNN_in.bin, NNN_out.bin, and NNN_layerK.bin (K = 1..8) for the first 5."""
    vdir = Path(out_dir) / "vectors"
    vdir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(packets):
        y, layer_outs = forward(p, model)
        (vdir / f"{i:03d}_in.bin").write_bytes(p)
        (vdir / f"{i:03d}_out.bin").write_bytes(y.astype(np.int8).tobytes())
        for k, a in enumerate(layer_outs if i < N_LAYER_DUMPS else []):
            (vdir / f"{i:03d}_layer{k + 1}.bin").write_bytes(a.astype(np.int8).tobytes())
    return vdir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="directory with manifest.json and weights.bin")
    ap.add_argument("--packets", required=True, help="uint8 [N, 27658] .npy of packed real frames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    frames = np.load(args.packets)
    if frames.dtype != np.uint8 or frames.ndim != 2 or frames.shape[1] != PACKET_BYTES:
        raise SystemExit(f"--packets must be uint8 [N, {PACKET_BYTES}], got {frames.dtype} {frames.shape}")
    vecs = make_vector_set([f.tobytes() for f in frames], np.random.default_rng(args.seed))
    print(f"wrote {len(vecs)} vectors to {write_vectors(Model.load(args.model), vecs, args.model)}")


if __name__ == "__main__":
    main()
