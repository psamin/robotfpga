"""Packing, test cases and a ctypes loader for the full-precision linear engine (issue #148).

Layout (matches float_shim.cpp and float_linear.h): output channels padded to a multiple of 32;
for each 4096-wide K chunk, for each of the 4 ports p, for each 32-channel group g, for each k:
one 128-bit word of 8 BF16 weights W[32 g + 8 p + l][k], l = 0..7.

    python fpga/hls/pack_float.py --tests build/stream/float   # random cases for tb_float
"""

import argparse
import ctypes
from pathlib import Path

import numpy as np

PORTS, FL, LANES, K_MAX = 4, 8, 32, 4096
HERE = Path(__file__).resolve().parent


def to_bf16_bits(w: np.ndarray) -> np.ndarray:
    """FP32 -> BF16 bit patterns, round to nearest even (exact for weights already in BF16)."""
    u = np.ascontiguousarray(w, np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def pack(w_bits: np.ndarray) -> tuple[np.ndarray, int]:
    """[M][K] BF16 bits -> (flat uint16 words, padded M)."""
    m, k = w_bits.shape
    mp = -(-m // LANES) * LANES
    w = np.zeros((mp, k), np.uint16)
    w[:m] = w_bits
    parts = []
    for k0 in range(0, k, K_MAX):
        wc = w[:, k0 : k0 + K_MAX]  # [Mp][kc]
        g = wc.reshape(mp // LANES, PORTS, FL, wc.shape[1])  # [g][p][l][k]
        parts.append(np.ascontiguousarray(g.transpose(1, 0, 3, 2)).reshape(-1))  # [p][g][k][l]
    return np.concatenate(parts), mp


def write_case(d: Path, w_bits: np.ndarray, x: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    m, k = w_bits.shape
    packed, mp = pack(w_bits)
    wf = bf16_bits_to_f32(w_bits).astype(np.float64)
    x64 = x.astype(np.float32).astype(np.float64)
    y = np.zeros((x.shape[0], mp))
    y[:, :m] = x64 @ wf.T
    s = np.zeros((x.shape[0], mp))
    s[:, :m] = np.abs(x64) @ np.abs(wf).T
    packed.tofile(d / "packed.bin")
    x.astype(np.float32).tofile(d / "x.bin")
    y.tofile(d / "y_ref.bin")
    s.tofile(d / "sumabs.bin")
    (d / "dims.txt").write_text(f"{mp} {k} {x.shape[0]} {m}\n")


class Lib:
    """The HLS core as a shared library: y = x @ W^T with BF16 W and FP32 everything else."""

    def __init__(self, path: Path = HERE / "build" / "libfloat_linear.so"):
        self.lib = ctypes.CDLL(str(path))
        self.lib.linear_bf16.restype = ctypes.c_int

    def __call__(self, packed: np.ndarray, mp: int, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, np.float32)
        n, k = x.shape
        y = np.empty((n, mp), np.float32)
        st = self.lib.linear_bf16(
            packed.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p),
            mp,
            k,
            n,
        )
        if st:
            raise RuntimeError(f"linear_bf16 status {st}")
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="write random test cases here")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.tests)
    # SmolVLA-like shapes: VLM 960/2560, expert 720 (padded to 736), connector K = 12288 (3 chunks).
    # K must be a multiple of 4 (128-bit activation words on the io port).
    for m, k, n in [(960, 960, 1), (2560, 960, 50), (720, 960, 137), (960, 12288, 64), (32, 8, 3)]:
        w = to_bf16_bits(rng.normal(0, 0.05, (m, k)).astype(np.float32))
        x = rng.normal(0, 1, (n, k)).astype(np.float32)
        write_case(out / f"case_{m}x{k}x{n}", w, x)
    print(f"wrote float test cases to {out}")


if __name__ == "__main__":
    main()
