import numpy as np
from hothouse import sun_calc


def test_rotate_u_x():
    r"""Test rotation equation around x."""
    u = np.array([1.0, 0.0, 0.0])
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([1.0, 0.0, 0.0]),
                          np.pi/2.0, u),
        np.array([1.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([0.0, 1.0, 0.0]),
                          np.pi/2.0, u),
        np.array([0.0, 0.0, 1.0]))
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([0.0, 0.0, 1.0]),
                          np.pi/2.0, u),
        np.array([0.0, -1.0, 0.0]))
    
def test_rotate_u_y():
    r"""Test rotation equation around y."""
    u = np.array([0.0, 1.0, 0.0])
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([1.0, 0.0, 0.0]),
                          np.pi/2.0, u),
        np.array([0.0, 0.0, -1.0]))
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([0.0, 1.0, 0.0]),
                          np.pi/2.0, u),
        np.array([0.0, 1.0, 0.0]))
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([0.0, 0.0, 1.0]),
                          np.pi/2.0, u),
        np.array([1.0, 0.0, 0.0]))

def test_rotate_u_z():
    r"""Test rotation equation around z."""
    u = np.array([0.0, 0.0, 1.0])
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([1.0, 0.0, 0.0]),
                          np.pi/2.0, u),
        np.array([0.0, 1.0, 0.0]))
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([0.0, 1.0, 0.0]),
                          np.pi/2.0, u),
        np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(
        sun_calc.rotate_u(np.array([0.0, 0.0, 1.0]),
                          np.pi/2.0, u),
        np.array([0.0, 0.0, 1.0]))

