"""QAT forward == ref/intref.py, byte for byte (build spec Phase 5)."""

import numpy as np
import pytest
import torch

from armlab.policy.qat import convert, load_intref
from armlab.policy.tiny import TinyPolicy, encode_aux

intref = load_intref()


def random_inputs(rng, n):
    img = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    # smooth "scenes" too: low-frequency images exercise less saturated activations
    smooth = np.repeat(np.repeat(rng.integers(0, 256, (n, 12, 12, 3), dtype=np.uint8), 8, 1), 8, 2)
    img[n // 2 :] = smooth[n // 2 :]
    aux = encode_aux(rng.uniform(-1.1, 1.1, (n, 6)), rng.integers(0, 4, n))
    return img, aux


@pytest.mark.parametrize("seed", [0, 1])
def test_qat_matches_intref_bit_exact(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = TinyPolicy()
    img, aux = random_inputs(rng, 64)
    q = convert(f, torch.from_numpy(img), torch.from_numpy(aux))
    q = q.double()  # exact integer arithmetic for the proof
    model = q.to_intref(intref)

    img, aux = random_inputs(rng, 500)
    with torch.no_grad():
        y_qat = q(torch.from_numpy(img), torch.from_numpy(aux)).numpy()
    y_int = np.round(y_qat * 127).astype(np.int64)
    assert np.allclose(y_qat * 127, y_int, atol=1e-9)  # QAT outputs lie on the int8 grid
    for i in range(len(img)):
        y_ref, _ = intref.forward(img[i].tobytes() + aux[i].tobytes(), model)
        np.testing.assert_array_equal(y_int[i].reshape(-1), y_ref.astype(np.int64))
