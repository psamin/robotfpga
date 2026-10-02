"""Checks the reference against a slow, literal reading of the spec."""

from pathlib import Path

import numpy as np
import pytest

from intref import LAYERS, Model, conv3x3s2, fc, forward, requant, weight_shape

STANDIN = Path(__file__).resolve().parent.parent / "artifacts" / "standin"


def naive_conv(x, w, b, s, relu=True):
    h, wd, ci = x.shape
    co = w.shape[0]
    y = np.zeros((h // 2, wd // 2, co), np.int8)
    for oy in range(h // 2):
        for ox in range(wd // 2):
            for o in range(co):
                acc = int(b[o])
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = 2 * oy + ky - 1, 2 * ox + kx - 1
                        if 0 <= iy < h and 0 <= ix < wd:
                            for c in range(ci):
                                acc += int(x[iy, ix, c]) * int(w[o, ky, kx, c])
                v = (acc + (1 << (s - 1))) >> s  # Python >> is arithmetic
                y[oy, ox, o] = min(max(v, 0 if relu else -127), 127)
    return y


def test_architecture_totals():
    params = sum(np.prod(weight_shape(l)) + l["out_shape"][-1] for l in LAYERS)
    macs = sum(
        np.prod(l["out_shape"]) * np.prod(weight_shape(l)[1:]) if l["type"] == "conv" else np.prod(weight_shape(l))
        for l in LAYERS
    )
    assert params == 565_552
    assert macs == 9_665_024


@pytest.mark.parametrize(
    "acc,s,lo,expect",
    [(3, 1, -127, 2), (2, 1, -127, 1), (-1, 1, -127, 0), (-2, 1, -127, -1), (-3, 1, -127, -1),
     (-5, 2, -127, -1), (-6, 2, -127, -1), (-7, 2, -127, -2), (10**6, 4, 0, 127), (-(10**6), 4, -127, -127),
     (-(10**6), 4, 0, 0)],
)
def test_requant_rounding(acc, s, lo, expect):
    assert requant(np.array([acc]), s, lo, 127)[0] == expect


def test_requant_rejects_zero_shift():
    with pytest.raises(ValueError):
        requant(np.array([1]), 0, 0, 127)


@pytest.mark.parametrize("h,ci,co", [(8, 3, 4), (6, 5, 3), (4, 16, 8)])
def test_conv_matches_naive(h, ci, co):
    rng = np.random.default_rng(h * 100 + ci)
    x = rng.integers(-128, 128, (h, h, ci)).astype(np.int8)
    w = rng.integers(-127, 128, (co, 3, 3, ci)).astype(np.int8)
    b = rng.integers(-5000, 5000, co).astype(np.int32)
    for s in (1, 5, 9):
        np.testing.assert_array_equal(conv3x3s2(x, w, b, s), naive_conv(x, w, b, s))


def test_fc_matches_naive():
    rng = np.random.default_rng(1)
    x = rng.integers(-127, 128, 37).astype(np.int8)
    w = rng.integers(-127, 128, (11, 37)).astype(np.int8)
    b = rng.integers(-9000, 9000, 11).astype(np.int32)
    for relu in (True, False):
        y = fc(x, w, b, 7, relu)
        for o in range(11):
            acc = int(b[o]) + sum(int(x[i]) * int(w[o, i]) for i in range(37))
            assert y[o] == min(max((acc + 64) >> 7, 0 if relu else -127), 127)


@pytest.mark.skipif(not STANDIN.exists(), reason="run ref/make_standin.py first")
def test_standin_vectors_and_conv1_naive():
    model = Model.load(STANDIN)
    m = model.layers
    assert [l["name"] for l in m] == [l["name"] for l in LAYERS]
    offset = 0
    for l in m:  # layout: contiguous, weights then biases, in order
        assert l["weight_offset"] == offset and l["bias_offset"] == offset + l["weight_bytes"]
        offset = l["bias_offset"] + l["bias_bytes"]
    assert offset == (STANDIN / "weights.bin").stat().st_size == 568_240

    vdir = STANDIN / "vectors"
    for i in range(100):
        p = (vdir / f"{i:03d}_in.bin").read_bytes()
        y, outs = forward(p, model)
        assert y.tobytes() == (vdir / f"{i:03d}_out.bin").read_bytes()
        if i < 5:
            for k, a in enumerate(outs):
                assert a.tobytes() == (vdir / f"{i:03d}_layer{k + 1}.bin").read_bytes()

    # conv1 on a real-ish frame, the slow way, top-left corner (exercises padding)
    p = np.frombuffer((vdir / "003_in.bin").read_bytes(), np.uint8)
    img = (p[:27648].astype(np.int16) - 128).astype(np.int8).reshape(96, 96, 3)[:16, :16]
    w, b = model.params[0]
    ref = np.frombuffer((vdir / "003_layer1.bin").read_bytes(), np.int8).reshape(48, 48, 16)
    np.testing.assert_array_equal(naive_conv(img, w, b, m[0]["shift"])[:7, :7], ref[:7, :7])
