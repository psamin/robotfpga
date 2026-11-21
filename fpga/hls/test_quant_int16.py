"""pack_float's int16 quantizers against build/brief/precision_emulation.py's quant_sym, bit for bit.

The int16 engine's operands must be exactly what the precision study measured (variant V4_int16: max chunk
error 3.4e-4): q = round-half-even(a / s).clamp(+-32767), s = max(rowmax|a|, 1e-30) / 32767, in FP32. Needs
torch (the lerobot venv):

    C:\\kp\\lerobot-venv\\Scripts\\python.exe fpga/hls/test_quant_int16.py

Random data: Gaussian, heavy-tailed, outlier channels, exact ties (values at k + 0.5 quantization steps),
values at the clamp, denormals, all-zero rows and rows below the 1e-30 floor, and bf16-rounded weights as the
checkpoint stores them. Every q must be equal; every scale must be equal except all-zero rows, where
pack_float uses 1.0 (precision_emulation: 1e-30 / 32767; q = 0 in both, so outputs are 0 either way).
Also checks the dequantized integer product on random data against the emulation's int16 linear (FP64).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pack_float as pf  # noqa: E402


def load_emulation():
    path = ROOT / "build/brief/precision_emulation.py"
    spec = importlib.util.spec_from_file_location("precision_emulation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cases(rng: np.random.Generator):
    yield "gauss", rng.normal(0, 1, (300, 777)).astype(np.float32)
    yield (
        "gauss_small_rows",
        (rng.normal(0, 1, (200, 960)) * np.exp(rng.normal(0, 3, (200, 1)))).astype(np.float32),
    )
    yield "heavy_tail", rng.standard_t(2, (256, 1024)).astype(np.float32)
    a = rng.normal(0, 1, (128, 3072)).astype(np.float32)
    a[:, rng.integers(0, 3072, 20)] *= 50
    yield "outlier_channels", a
    # exact ties: rows whose max is 32767 * 2^-10, so s = 2^-10 exactly and (k + 0.5) * s are ties
    t = (rng.integers(-32766, 32766, (64, 512)) + 0.5).astype(np.float64) * 2.0**-10
    t[:, 0] = 32767 * 2.0**-10
    yield "ties", t.astype(np.float32)
    c = rng.normal(0, 1, (64, 256)).astype(np.float32)
    c[:, 7] = np.abs(c).max(1) * np.where(rng.random(64) < 0.5, -1, 1)  # the max at both signs
    yield "clamp_edges", c
    d = (rng.normal(0, 1, (32, 128)) * 1e-39).astype(np.float32)  # denormal rows: below the 1e-30 floor
    yield "denormal_rows", d
    e = (rng.normal(0, 1, (32, 128)) * 1e-31).astype(np.float32)  # normal but below the floor
    yield "below_floor", e
    z = rng.normal(0, 1, (40, 300)).astype(np.float32)
    z[::3] = 0
    z[1, :] = -0.0
    yield "zero_rows", z
    w = (
        torch.from_numpy(rng.normal(0, 0.05, (512, 960)).astype(np.float32))
        .to(torch.bfloat16)
        .float()
        .numpy()
    )
    yield "bf16_weights", w
    yield "one_column", rng.normal(0, 1, (50, 1)).astype(np.float32)
    yield "one_row", rng.normal(0, 1, (1, 12288)).astype(np.float32)


def main() -> int:
    em = load_emulation()
    rng = np.random.default_rng(2026)
    fails = 0
    total = 0
    for name, a in cases(rng):
        qt, st = em.quant_sym(torch.from_numpy(a), 32767, 1)
        qt = qt.numpy()
        st = st.squeeze(1).numpy()
        for which, fn in (("rows", pf.quantize_int16_rows), ("tokens", pf.quantize_int16_tokens)):
            q, s = fn(a)
            zero = np.abs(a).max(1) == 0
            q_ok = q.dtype == np.int16 and np.array_equal(q.astype(np.float32), qt)
            s_ok = np.array_equal(s[~zero].view(np.uint32), st[~zero].view(np.uint32))
            z_ok = bool(np.all(s[zero] == 1.0) and np.all(q[zero] == 0))
            ok = q_ok and s_ok and z_ok
            fails += not ok
            total += a.size
            print(
                f"{'PASS' if ok else 'FAIL'}: {name:18s} {which:6s} {a.shape[0]}x{a.shape[1]}: "
                f"q equal {q_ok}, scales equal {s_ok} ({int((~zero).sum())} rows), "
                f"zero rows {int(zero.sum())} -> scale 1 / q 0 {z_ok}"
            )
    # the dequantized integer product: the emulation's V4 int16 linear (FP64 acc, double dequant, then FP32)
    # against pack_float's operands with an exact float64 product; equal up to the final FP32 rounding
    w = rng.normal(0, 0.05, (736, 720)).astype(np.float32)
    x = (rng.normal(0, 1, (50, 720)) * np.exp(rng.normal(0, 1, (50, 1)))).astype(np.float32)
    xq_t, sx_t = em.quant_sym(torch.from_numpy(x), 32767, 1)
    wq_t, sw_t = em.quant_sym(torch.from_numpy(w), 32767, 1)
    y_em = (
        ((xq_t.double() @ wq_t.double().t()) * sx_t.double() * sw_t.squeeze(1).double()[None, :])
        .float()
        .numpy()
    )
    wq, sw = pf.quantize_int16_rows(w)
    xq, sx = pf.quantize_int16_tokens(x)
    y_pf = (xq.astype(np.float64) @ wq.astype(np.float64).T) * sx.astype(np.float64)[:, None] * sw[None, :]
    same = np.array_equal(y_pf.astype(np.float32), y_em)
    fails += not same
    print(
        f"{'PASS' if same else 'FAIL'}: dequantized int16 product (736x720 on 50 tokens) "
        "equal to the emulation's V4"
    )
    print(f"{'PASS' if not fails else 'FAIL'}: {total} values quantized, {fails} failing checks")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
