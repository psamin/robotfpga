"""Checks the reference against a slow, literal reading of the spec."""

import numpy as np
import pytest
from intref import (
    LAYERS,
    PACKET_BYTES,
    Model,
    conv3x3s2,
    fc,
    forward,
    pack_obs,
    requant,
    unpack_actions,
    unpack_input,
    weight_shape,
)


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


def random_model(seed=0):
    rng = np.random.default_rng(seed)
    layers = [dict(layer, shift=12) for layer in LAYERS]
    params = [
        (
            rng.integers(-127, 128, weight_shape(layer)).astype(np.int8),
            rng.integers(-10_000, 10_000, layer["out_shape"][-1]).astype(np.int32),
        )
        for layer in LAYERS
    ]
    return Model(layers, params)


def test_architecture_totals():
    params = sum(np.prod(weight_shape(layer)) + layer["out_shape"][-1] for layer in LAYERS)
    macs = sum(
        np.prod(layer["out_shape"]) * np.prod(weight_shape(layer)[1:])
        if layer["type"] == "conv"
        else np.prod(weight_shape(layer))
        for layer in LAYERS
    )
    assert params == 565_552
    assert macs == 9_665_024


@pytest.mark.parametrize(
    "acc,s,lo,expect",
    [
        (3, 1, -127, 2),
        (2, 1, -127, 1),
        (-1, 1, -127, 0),
        (-2, 1, -127, -1),
        (-3, 1, -127, -1),
        (-5, 2, -127, -1),
        (-6, 2, -127, -1),
        (-7, 2, -127, -2),
        (10**6, 4, 0, 127),
        (-(10**6), 4, -127, -127),
        (-(10**6), 4, 0, 0),
    ],
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


def test_pack_obs_and_unpack_actions():
    image = np.arange(96 * 96 * 3, dtype=np.int64).reshape(96, 96, 3).astype(np.uint8)
    state = np.array([0.5, -0.5, 1.0, -1.0, 2.0, 0.3], np.float32)
    packet = pack_obs({"image": image, "state": state, "instr": 2})
    assert len(packet) == PACKET_BYTES
    img, aux = unpack_input(packet)
    np.testing.assert_array_equal(img, (image.astype(np.int16) - 128).astype(np.int8))
    # 63.5 -> 64 and -63.5 -> -63 (half up, not numpy's half-to-even), +-1 -> +-127, 2 clamps, 38.1 -> 38
    assert aux.tolist() == [64, -63, 127, -127, 127, 38, 0, 0, 127, 0]

    y = np.array([127, -127, 0] + [1] * 45, np.int8)
    a = unpack_actions(y)
    assert a.shape == (8, 6) and a.dtype == np.float32
    assert a[0, 0] == 1.0 and a[0, 1] == -1.0 and a[0, 2] == 0.0


def test_save_load_roundtrip(tmp_path):
    model = random_model()
    manifest = model.save(tmp_path)
    offset = 0
    for layer in manifest["layers"]:  # layout: contiguous, weights then biases, in order
        assert layer["weight_offset"] == offset
        assert layer["bias_offset"] == offset + layer["weight_bytes"]
        offset = layer["bias_offset"] + layer["bias_bytes"]
    assert offset == manifest["total_bytes"] == (tmp_path / "weights.bin").stat().st_size == 568_240

    loaded = Model.load(tmp_path)
    packet = bytes(np.random.default_rng(3).integers(0, 256, PACKET_BYTES, dtype=np.uint8))
    y1, outs1 = forward(packet, model)
    y2, _ = forward(packet, loaded)
    assert y1.shape == (48,) and y1.tobytes() == y2.tobytes()
    assert [o.shape for o in outs1] == [
        (48, 48, 16),
        (24, 24, 32),
        (12, 12, 64),
        (6, 6, 96),
        (3, 3, 128),
        (256,),
        (256,),
        (48,),
    ]

    blob = bytearray((tmp_path / "weights.bin").read_bytes())
    blob[0] ^= 1
    (tmp_path / "weights.bin").write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="sha256"):
        Model.load(tmp_path)
