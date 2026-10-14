import numpy as np
import pytest

pytest.importorskip("mujoco")

from policy.sim import so101  # noqa: E402

needs_tag = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


def test_gripper_scale_comes_from_calibration():
    assert np.degrees(so101.gripper_rad_per_unit() * 100) == pytest.approx(137.8, abs=0.1)


def test_gripper_mapping_is_monotonic_clipped_and_invertible():
    r = np.linspace(0, 100, 101)
    q = so101.gripper_to_sim(r)
    assert (np.diff(q) >= 0).all() and q[0] == so101.GRIPPER_RANGE[0] and q[-1] == so101.GRIPPER_RANGE[1]
    inside = np.linspace(5, 80, 16)
    assert np.allclose(so101.sim_to_gripper(so101.gripper_to_sim(inside)), inside)


@needs_tag
def test_pads_touch_angle_matches_the_model_geometry():
    m = so101.load_model()
    qs = np.linspace(-0.25, -0.15, 101)
    gaps = np.array([so101.jaw_gap(m, q) for q in qs])
    assert qs[np.argmin(np.abs(gaps))] == pytest.approx(so101.PADS_TOUCH, abs=0.005)


@needs_tag
def test_demo_readings_give_plausible_gaps():
    m = so101.load_model()
    assert so101.jaw_gap(m, so101.gripper_to_sim(2.0)) < 0.005  # rest pose: closed
    cap = so101.jaw_gap(m, so101.gripper_to_sim(11.5))
    assert abs(cap - 0.027) < 0.008  # cap estimated at 27 mm from the front video (#174)
    assert so101.jaw_gap(m, so101.gripper_to_sim(19.5)) > cap  # the body is wider than the cap
