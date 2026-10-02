import numpy as np
import torch

from armlab.policy.tiny import TinyPolicy, encode_aux


def test_param_count_matches_contract():
    assert sum(p.numel() for p in TinyPolicy().parameters()) == 565_552


def test_forward_shape_and_backward():
    m = TinyPolicy()
    img = torch.randint(0, 256, (4, 96, 96, 3), dtype=torch.uint8)
    aux = torch.from_numpy(encode_aux(np.zeros((4, 6)), np.array([0, 1, 2, 3])))
    y = m(*m.prep(img, aux))
    assert y.shape == (4, 8, 6)
    y.abs().mean().backward()
    assert m.convs[0].weight.grad is not None


def test_encode_aux_rounds_half_up():
    a = encode_aux(np.array([0.5 / 127, -0.5 / 127, 1.2, -1.2, 0, 1]), np.array(2))
    assert a.tolist() == [1, 0, 127, -127, 0, 127, 0, 0, 127, 0]
