"""Packing, test cases and a ctypes loader for the full-precision linear engine (issue #148).

Layout (matches float_shim.cpp and float_linear.h): output channels padded to a multiple of 32;
for each 4096-wide K chunk, for each of the 4 ports p, for each 32-channel group g, for each k:
one 128-bit word of 8 BF16 weights W[32 g + 8 p + l][k], l = 0..7.

    python fpga/hls/pack_float.py --tests build/stream/float   # random cases for tb_float
    python fpga/hls/pack_float.py --tests-int16 build/stream/int16   # int16 engine cases (FSTREAM_ARITH=16)

int16 engine (int16_linear.h, DRIVER CONTRACT there): quantize_int16_rows / quantize_int16_tokens are
build/brief/precision_emulation.py's quant_sym(., 32767, 1) bit for bit (fpga/hls/test_quant_int16.py), except
that an all-zero row gets scale 1.0 instead of 1e-30 / 32767 (its values are 0 either way); the weights go
through pack() as their uint16 bit patterns, pack_xmem_int16 builds the X buffer, unpack_y_int16 un-tiles Y.
"""

import argparse
import ctypes
from pathlib import Path

import numpy as np

PORTS, FL, LANES, K_MAX = 4, 8, 32, 4096
QMAX16 = 32767  # symmetric int16: -32767..32767 (|w x| < 2^30, so a 4096-long block sum stays below 2^42)
TINY = 1e-30  # precision_emulation.TINY: the scale floor of a non-zero row
TB = 32  # tokens per engine block (int16_linear.h TB): X and Y are padded to multiples of 32 tokens
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


def _quantize_int16(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric absmax int16 per row of a 2-D array: precision_emulation.quant_sym(a, 32767, 1) in FP32
    (scale = max(rowmax|a|, 1e-30) / 32767, q = round-half-even(a / scale) clamped to +-32767), int16 values
    and FP32 scales; all-zero rows get scale 1.0."""
    a = np.ascontiguousarray(a, np.float32)
    am = np.abs(a).max(axis=1)
    s = np.maximum(am, np.float32(TINY)) / np.float32(QMAX16)
    q = np.clip(np.round(a / s[:, None]), -QMAX16, QMAX16).astype(np.int16)
    s[am == 0] = np.float32(1.0)
    return q, s.astype(np.float32)


def quantize_int16_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[M][K] weights -> (int16 [M][K], s_row FP32 [M]): w ~= q * s_row[:, None]."""
    return _quantize_int16(w)


def quantize_int16_tokens(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N][K] activations -> (int16 [N][K], s_tok FP32 [N]), one scale per token over the whole row (for K
    longer than 4096 the host passes the same s_tok with every K chunk)."""
    return _quantize_int16(x)


def pack_xmem_int16(s_row: np.ndarray, s_tok: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """The int16 engine's xmem buffer for one call (K <= 4096), as bytes: s_row (FP32, Mp = len(s_row), a
    multiple of 32), s_tok (FP32, padded to Np = 32 ceil(N / 32) with 1.0), then Xq as
    [Np / 32 blocks][K][32 tokens] int16 (padding tokens zero). A 16-byte multiple; place it 16-byte aligned
    (page-aligned is best)."""
    s_row = np.ascontiguousarray(s_row, np.float32)
    n, k = xq.shape
    if len(s_row) % LANES or not 1 <= k <= K_MAX:
        raise ValueError(
            f"pack_xmem_int16: M {len(s_row)} must be a multiple of {LANES}, K {k} in 1..{K_MAX}"
        )
    npad = -(-n // TB) * TB
    st = np.ones(npad, np.float32)
    st[:n] = s_tok
    xi = np.zeros((npad, k), np.int16)
    xi[:n] = xq
    xi = np.ascontiguousarray(xi.reshape(npad // TB, TB, k).transpose(0, 2, 1))  # [b][k][a]
    return np.concatenate([s_row.view(np.uint8), st.view(np.uint8), xi.reshape(-1).view(np.uint8)])


def unpack_y_int16(y_tiled: np.ndarray, n: int, mp: int) -> np.ndarray:
    """Y[Mp/32][Np][32] FP32 as the engine writes it -> y[N][Mp]."""
    npad = -(-n // TB) * TB
    y = np.asarray(y_tiled, np.float32).reshape(mp // LANES, npad, LANES)
    return np.ascontiguousarray(y.transpose(1, 0, 2).reshape(npad, mp)[:n])


def write_case_int16(d: Path, w: np.ndarray, x: np.ndarray) -> None:
    """One tb_float int16 case (see tb_float.cpp): quantized operands, scales, the exact dequantized integer
    product (float64), the unquantized float64 product and, for K <= 4096, the xmem image of the call."""
    d.mkdir(parents=True, exist_ok=True)
    m, k = w.shape
    n = x.shape[0]
    wq, sw = quantize_int16_rows(w)
    xq, sx = quantize_int16_tokens(x)
    packed, mp = pack(wq.view(np.uint16))
    srow = np.ones(mp, np.float32)
    srow[:m] = sw
    acc = xq.astype(np.float64) @ wq.astype(np.float64).T  # exact: integer partial sums < 2^44 < 2^53
    y = np.zeros((n, mp))
    y[:, :m] = acc * sw.astype(np.float64)[None, :] * sx.astype(np.float64)[:, None]
    yf = np.zeros((n, mp))
    yf[:, :m] = x.astype(np.float32).astype(np.float64) @ w.astype(np.float32).astype(np.float64).T
    sa = np.zeros((n, mp))
    sa[:, :m] = np.abs(xq * sx.astype(np.float64)[:, None]) @ np.abs(wq * sw.astype(np.float64)[:, None]).T
    packed.tofile(d / "packed.bin")
    srow.tofile(d / "srow.bin")
    xq.tofile(d / "xq.bin")
    sx.tofile(d / "stok.bin")
    y.tofile(d / "y_ref.bin")
    yf.tofile(d / "y_float.bin")
    sa.tofile(d / "sumabs.bin")
    if k <= K_MAX:
        pack_xmem_int16(srow, sx, xq).tofile(d / "xmem.bin")
    elif (d / "xmem.bin").exists():
        (d / "xmem.bin").unlink()
    (d / "dims.txt").write_text(f"{mp} {k} {n} {m} int16\n")


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


class Lib16:
    """The int16 core as a shared library (build/libfloat_linear_int16.so, FSTREAM_ARITH=16)."""

    def __init__(self, path: Path = HERE / "build" / "libfloat_linear_int16.so"):
        self.lib = ctypes.CDLL(str(path))
        self.lib.linear_int16.restype = ctypes.c_int
        self.lib.linear_int16_raw.restype = ctypes.c_int

    def __call__(
        self, packed: np.ndarray, s_row: np.ndarray, xq: np.ndarray, s_tok: np.ndarray, mp: int
    ) -> np.ndarray:
        """y[N][Mp] = float(xq @ Wq^T) * s_row * s_tok, K chunks of 4096 added in FP32."""
        xq = np.ascontiguousarray(xq, np.int16)
        s_row = np.ascontiguousarray(s_row, np.float32)
        s_tok = np.ascontiguousarray(s_tok, np.float32)
        n, k = xq.shape
        y = np.empty((n, mp), np.float32)
        ptr = ctypes.c_void_p
        st = self.lib.linear_int16(
            packed.ctypes.data_as(ptr),
            s_row.ctypes.data_as(ptr),
            xq.ctypes.data_as(ptr),
            s_tok.ctypes.data_as(ptr),
            y.ctypes.data_as(ptr),
            mp,
            k,
            n,
        )
        if st:
            raise RuntimeError(f"linear_int16 status {st}")
        return y

    def raw(self, packed: np.ndarray, xmem: np.ndarray, mp: int, k: int, n: int) -> np.ndarray:
        """One engine call on DDR images (packed W of one K chunk, pack_xmem_int16): Y[Mp/32][Np][32] flat."""
        npad = -(-n // TB) * TB
        yt = np.empty(mp * npad, np.float32)
        ptr = ctypes.c_void_p
        st = self.lib.linear_int16_raw(
            packed.ctypes.data_as(ptr), xmem.ctypes.data_as(ptr), yt.ctypes.data_as(ptr), mp, k, n
        )
        if st:
            raise RuntimeError(f"linear_int16_raw status {st}")
        return yt


# int16 cases: the FP32 shapes, SmolVLA-shaped calls (vision 768/3072 on 256 tokens, connector K = 12288 on 16
# tokens, LM 2560x960 on the 81-token prefix, expert 736/720 on 50 tokens) and edges of the engine's control:
# K < 8 (zero-padded blocks), K = 64 (emit-paced blocks), K = 4096 and 4097 (a K = 1 chunk), N = 33 / 130
# (a last tile of one block, three tiles), odd K.
INT16_SHAPES = [
    (960, 960, 1),
    (2560, 960, 50),
    (720, 960, 137),
    (960, 12288, 64),
    (32, 8, 3),
    (768, 768, 256),
    (3072, 768, 256),
    (960, 12288, 16),
    (2560, 960, 81),
    (736, 720, 50),
    (720, 960, 50),
    (64, 5, 33),
    (32, 37, 1),
    (96, 64, 130),
    (32, 4096, 70),
    (64, 4097, 3),
]


def int16_operands(rng: np.random.Generator, m: int, k: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Random FP32 operands with uneven scales: rows of different magnitude, a few outlier input channels
    (SmolVLA's activations have crest factors of 10-16), an all-zero weight row and an all-zero token."""
    w = rng.normal(0, 0.05, (m, k)).astype(np.float32)
    w *= np.exp(rng.normal(0, 0.7, (m, 1))).astype(np.float32)
    x = rng.normal(0, 1, (n, k)).astype(np.float32)
    x[:, rng.integers(0, k, max(1, k // 200))] *= 20
    x *= np.exp(rng.normal(0, 1.0, (n, 1))).astype(np.float32)
    if m >= 2:
        w[m - 1] = 0
    if n >= 2:
        x[1] = 0
    return w, x


def check_lib_int16(lib: Lib16, rng: np.random.Generator) -> int:
    """The Python side of the driver contract against the C model through ctypes: quantize_int16_* -> pack /
    pack_xmem_int16 -> one raw engine call per K chunk -> unpack_y_int16, summed in chunk order, must equal
    Lib16 (the shim's own host layout) bit for bit and the exact dequantized product within 1e-6 sum|xw|."""
    fails = 0
    for m, k, n in [(736, 720, 50), (320, 960, 81), (96, 4100, 33), (32, 8, 1)]:
        w, x = int16_operands(rng, m, k, n)
        wq, sw = quantize_int16_rows(w)
        xq, sx = quantize_int16_tokens(x)
        mp = -(-m // LANES) * LANES
        srow = np.ones(mp, np.float32)
        srow[:m] = sw
        y = np.zeros((n, mp), np.float32)
        for k0 in range(0, k, K_MAX):
            kc = min(K_MAX, k - k0)
            packed, _ = pack(wq[:, k0 : k0 + kc].view(np.uint16))  # one chunk = the 4 port arrays
            xmem = pack_xmem_int16(srow, sx, np.ascontiguousarray(xq[:, k0 : k0 + kc]))
            y += unpack_y_int16(lib.raw(packed, xmem, mp, kc, n), n, mp)
        full, _ = pack(wq.view(np.uint16))
        y_lib = lib(full, srow, xq, sx, mp)
        acc = xq.astype(np.float64) @ wq.astype(np.float64).T
        ref = acc * sw.astype(np.float64)[None, :] * sx.astype(np.float64)[:, None]
        sa = np.abs(xq * sx.astype(np.float64)[:, None]) @ np.abs(wq * sw.astype(np.float64)[:, None]).T
        rel = float((np.abs(y[:, :m] - ref) / (sa + 1e-30)).max())
        same = np.array_equal(y.view(np.uint32), y_lib.view(np.uint32))
        ok = same and rel <= 1e-6 and not np.any(y[:, m:])
        fails += not ok
        print(
            f"{'PASS' if ok else 'FAIL'}: contract path {m}x{k}x{n}: raw calls == linear_int16 bits {same}, "
            f"max err/sum|xw| {rel:.3g}, padded columns zero {not np.any(y[:, m:])}"
        )
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", help="write random FP32-engine test cases here")
    ap.add_argument("--tests-int16", help="write random int16-engine test cases here")
    ap.add_argument(
        "--check-int16-lib",
        action="store_true",
        help="driver-contract helpers vs build/libfloat_linear_int16.so",
    )
    args = ap.parse_args()
    if args.check_int16_lib:
        raise SystemExit(1 if check_lib_int16(Lib16(), np.random.default_rng(1616)) else 0)
    if not args.tests and not args.tests_int16:
        ap.error("give --tests, --tests-int16 or --check-int16-lib")
    if args.tests:
        rng = np.random.default_rng(0)
        out = Path(args.tests)
        # SmolVLA-like shapes: VLM 960/2560, expert 720 (padded to 736), connector K = 12288 (3 chunks).
        # K must be a multiple of 4 (128-bit activation words on the io port).
        for m, k, n in [(960, 960, 1), (2560, 960, 50), (720, 960, 137), (960, 12288, 64), (32, 8, 3)]:
            w = to_bf16_bits(rng.normal(0, 0.05, (m, k)).astype(np.float32))
            x = rng.normal(0, 1, (n, k)).astype(np.float32)
            write_case(out / f"case_{m}x{k}x{n}", w, x)
        print(f"wrote float test cases to {out}")
    if args.tests_int16:
        rng = np.random.default_rng(16)
        out = Path(args.tests_int16)
        for m, k, n in INT16_SHAPES:
            w, x = int16_operands(rng, m, k, n)
            write_case_int16(out / f"case_{m}x{k}x{n}", w, x)
        print(f"wrote {len(INT16_SHAPES)} int16 test cases to {out}")


if __name__ == "__main__":
    main()
