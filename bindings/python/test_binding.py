import math

import numpy as np

import vsr_runtime as vsr


def test_signature():
    e3 = vsr.Signature.e3()
    assert e3.n == 3
    assert e3.size == 8
    assert "e1" in e3.blade_name(1)


def test_multivector_gp():
    sig = vsr.Signature.e3()
    e1 = vsr.Multivector.blade(sig, 1, 1.0)
    e2 = vsr.Multivector.blade(sig, 2, 1.0)
    assert abs(e1.gp(e1).scalar_part() - 1.0) < 1e-12
    assert abs(e1.gp(e2).scalar_part()) < 1e-12
    assert e1.gp(e2).coefficients().shape == (8,)


def test_e3_rotor():
    rotor = vsr.e3_rotor(0, 0, 1, math.pi / 2)
    image = vsr.e3_rotate(vsr.e3_vec(1, 0, 0), rotor)
    np.testing.assert_allclose(vsr.e3_coords(image), [0, 1, 0], atol=1e-12)


def test_e3_rotate_many():
    rotor = vsr.e3_rotor(0, 0, 1, math.pi / 2)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out = vsr.e3_rotate_many(vectors, rotor)
    np.testing.assert_allclose(out, [[0, 1, 0], [-1, 0, 0]], atol=1e-12)


def test_p3_point_and_motor():
    p = vsr.p3_point(1, 2, 3)
    np.testing.assert_allclose(vsr.p3_point_coords(p), [1, 2, 3], atol=1e-12)

    motor = vsr.p3_motor(
        vsr.p3_rotor(0, 0, 1, math.pi / 2),
        vsr.p3_translator(1, 2, 3),
    )
    moved = vsr.p3_apply(motor, vsr.p3_point(1, 0, 0))
    np.testing.assert_allclose(vsr.p3_point_coords(moved), [1, 3, 3], atol=1e-12)


def test_p3_plane_meet():
    p0 = vsr.p3_point(0, 1, 2)
    p1 = vsr.p3_point(0, -1, 2)
    line = vsr.p3_line(p0, p1)
    hit = vsr.p3_meet_plane_line(vsr.p3_plane(0, 1, 0, 0), line)
    np.testing.assert_allclose(vsr.p3_point_coords(hit), [0, 0, 2], atol=1e-12)


if __name__ == "__main__":
    test_signature()
    test_multivector_gp()
    test_e3_rotor()
    test_e3_rotate_many()
    test_p3_point_and_motor()
    test_p3_plane_meet()
    print("python binding tests passed")
