import torch

from armlab.policy.train import augment, chunk_l1


def test_augment_keeps_shape_dtype_and_grid():
    img = torch.randint(0, 256, (5, 96, 96, 3), dtype=torch.uint8)
    out = augment(img)
    assert out.shape == img.shape and out.dtype == torch.uint8


def test_augment_is_a_shift_without_photometric_jitter():
    img = torch.arange(96 * 96 * 3, dtype=torch.int64).remainder(251).to(torch.uint8).view(1, 96, 96, 3)
    out = augment(img, pad=4, bright=0.0, contrast=0.0)
    # the 88x88 interior of any shift by <= 4 px is a window of the original
    inner = out[0, 4:92, 4:92]
    found = any(
        torch.equal(inner, img[0, 4 + dy : 92 + dy, 4 + dx : 92 + dx])
        for dy in range(-4, 5)
        for dx in range(-4, 5)
    )
    assert found


def test_chunk_l1_ignores_padding():
    pred = torch.zeros(1, 8, 6)
    target = torch.zeros(1, 8, 6)
    target[0, 5:] = 10.0
    pad = torch.zeros(1, 8, dtype=torch.bool)
    pad[0, 5:] = True
    assert chunk_l1(pred, target, pad).item() == 0.0
