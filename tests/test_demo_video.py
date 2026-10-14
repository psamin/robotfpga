import numpy as np
import pytest

pytest.importorskip("pyarrow")

from policy.sim.demo_video import cap_pixel, fit_similarity, plate_circle, to_world  # noqa: E402


def hsv_image(h=480, w=640):
    img = np.zeros((h, w, 3))
    img[..., 2] = 230  # bright, unsaturated table
    return img


def project(world, scale=840.0, angle=0.2, base=(320.0, 45.0)):
    """Top view with +x pointing roughly down the image, as on the real front camera."""
    c, s = np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)
    x, y = np.asarray(world).T
    u = base[0] + scale * (c * x - s * y)
    v = -(-base[1] + scale * (s * x + c * y))
    return np.column_stack([u, v])


def test_fit_recovers_scale_and_base_and_rejects_the_flipped_pan():
    world = np.random.default_rng(0).uniform([0.15, -0.25], [0.35, 0.15], (20, 2))
    px = project(world)
    params, res = fit_similarity(world, px)
    _, res_flip = fit_similarity(world, px, flip_pan=True)
    assert res.max() < 1e-6 and res_flip.mean() > 20
    assert np.hypot(*params[:2]) == pytest.approx(840.0)
    assert np.allclose(to_world(params, *px[3]), world[3])


def test_plate_circle_finds_ring():
    img = hsv_image()
    yy, xx = np.mgrid[:480, :640]
    r = np.hypot(xx - 100, yy - 290)
    ring = (r > 50) & (r < 80)
    img[ring, 0], img[ring, 1] = 160, 200  # saturated blue pattern
    cx, cy, r_out, r_in = plate_circle(img)
    assert abs(cx - 100) < 1 and abs(cy - 290) < 1 and abs(r_out - 80) < 2 and abs(r_in - 50) < 2


def test_cap_pixel_finds_orange_disk():
    img = hsv_image()
    yy, xx = np.mgrid[:480, :640]
    disk = np.hypot(xx - 340, yy - 275) < 12
    img[disk] = [25 * 255 / 360, 230, 230]
    u, v = cap_pixel(img)
    assert abs(u - 340) < 0.5 and abs(v - 275) < 0.5
    assert cap_pixel(hsv_image()) is None
