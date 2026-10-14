import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from policy.sim import so101  # noqa: E402
from policy.sim.ik import TopDownIK  # noqa: E402

pytestmark = pytest.mark.skipif(not so101.tag_available(), reason=f"git tag {so101.TAG} not fetched")


@pytest.fixture(scope="module")
def ik():
    return TopDownIK(so101.load_model())


@pytest.mark.parametrize("xy", [(0.25, 0.09), (0.33, -0.19)])
@pytest.mark.parametrize("z", [0.03, 0.12])
def test_position_solve_reaches_the_demo_spots_at_grasp_and_carry_height(ik, xy, z):
    q, err = ik.solve_position([*xy, z])
    assert err < 0.003 and np.degrees(ik.tilt(q)) <= 85
    assert np.allclose(ik.site_pose(q)[0], [*xy, z], atol=0.003)


def test_top_down_solve_has_no_tilt(ik):
    q, err = ik.solve([0.2, 0.0, 0.03])
    assert err < 0.002 and np.degrees(ik.tilt(q)) < 3
