"""Checks the golden-vector files against the reference and against the handoff format."""

import numpy as np
import pytest
from intref import PACKET_BYTES, Model, forward, unpack_input
from make_standin import build_standin
from test_intref import naive_conv
from vectors import N_LAYER_DUMPS, N_VECTORS, edge_case_packets, make_vector_set

LAYER_BYTES = [48 * 48 * 16, 24 * 24 * 32, 12 * 12 * 64, 6 * 6 * 96, 3 * 3 * 128, 256, 256, 48]


@pytest.fixture(scope="module")
def standin(tmp_path_factory):
    out = tmp_path_factory.mktemp("standin")
    build_standin(out, seed=0)
    return out


def test_files_and_sizes(standin):
    vdir = standin / "vectors"
    assert (standin / "weights.bin").stat().st_size == 568_240
    for i in range(N_VECTORS):
        assert (vdir / f"{i:03d}_in.bin").stat().st_size == PACKET_BYTES
        assert (vdir / f"{i:03d}_out.bin").stat().st_size == 48
    for i in range(N_LAYER_DUMPS):
        for k, n in enumerate(LAYER_BYTES):
            assert (vdir / f"{i:03d}_layer{k + 1}.bin").stat().st_size == n
    assert not (vdir / f"{N_LAYER_DUMPS:03d}_layer1.bin").exists()
    assert len({(vdir / f"{i:03d}_in.bin").read_bytes() for i in range(N_VECTORS)}) == N_VECTORS


def test_outputs_match_reference(standin):
    model = Model.load(standin)
    vdir = standin / "vectors"
    for i in range(N_VECTORS):
        y, outs = forward((vdir / f"{i:03d}_in.bin").read_bytes(), model)
        assert y.tobytes() == (vdir / f"{i:03d}_out.bin").read_bytes()
        for k, a in enumerate(outs if i < N_LAYER_DUMPS else []):
            assert a.tobytes() == (vdir / f"{i:03d}_layer{k + 1}.bin").read_bytes()


def test_activations_are_not_degenerate(standin):
    model = Model.load(standin)
    _, outs = forward((standin / "vectors" / "003_in.bin").read_bytes(), model)
    for a in outs[:-1]:  # ReLU layers: some zeros, mostly not saturated
        assert 0.05 < (a == 0).mean() < 0.95 and (a == 127).mean() < 0.05


def test_conv1_on_a_frame_the_slow_way(standin):
    model = Model.load(standin)
    img, _ = unpack_input((standin / "vectors" / "003_in.bin").read_bytes())
    w, b = model.params[0]
    ref = np.frombuffer((standin / "vectors" / "003_layer1.bin").read_bytes(), np.int8).reshape(48, 48, 16)
    # top-left corner exercises the zero padding; a 16x16 crop is enough for 8x8 outputs
    np.testing.assert_array_equal(naive_conv(img[:16, :16], w, b, model.layers[0]["shift"]), ref[:8, :8])


def test_vector_set_layout():
    rng = np.random.default_rng(0)
    frames = [bytes([i]) * PACKET_BYTES for i in range(60)]
    vecs = make_vector_set(frames, rng)
    edges = edge_case_packets()
    assert vecs[:3] == edges[:3] and vecs[3] == frames[0] and vecs[5:14] == edges[3:]
    assert vecs[14] == frames[1]
    with pytest.raises(ValueError, match="frames"):
        make_vector_set(frames[:10], rng)
