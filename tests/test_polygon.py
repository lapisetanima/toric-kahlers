from fractions import Fraction

import pytest

from extremal import Polygon, blowup_cp2, hirzebruch, product, projective_plane, random_delzant, x, y


def test_orientation_and_convexity():
    P = Polygon([(0, 0), (0, 1), (1, 0)])  # clockwise input
    assert P.area == Fraction(1, 2)
    assert P.vertices[1] == (1, 0)
    with pytest.raises(ValueError):
        Polygon([(0, 0), (1, 0), (2, 0), (0, 1)])  # collinear vertex
    with pytest.raises(TypeError):
        Polygon([(0, 0), (1.5, 0), (0, 1)])  # floats are rejected


def test_edges_and_delzant():
    P = projective_plane(1)
    assert [e.normal for e in P.edges] == [(0, 1), (-1, -1), (1, 0)]
    assert [e.constant for e in P.edges] == [0, -1, 0]
    assert all(e.lattice_length == 1 for e in P.edges)
    assert P.is_delzant
    W = Polygon([(0, 0), (2, 0), (0, 1)])  # weighted projective plane P(1,1,2)
    assert not W.is_delzant
    assert W.singular_vertices == (2,)  # the vertex (0, 1)


def test_cp2_invariants():
    P = projective_plane(1)
    assert P.boundary_measure == 3
    assert P.S_hat == 6
    assert P.futaki == (0, 0)
    assert P.A_ext.is_constant and P.A_ext.c0 == 6


def test_product_invariants():
    P = product(2, 3)
    assert P.S_hat == Fraction(2 * 5, 6)
    assert P.futaki_vanishes
    assert P.A_ext.c0 == P.S_hat and P.A_ext.is_constant


def test_blowup_has_nonconstant_A_ext():
    P = blowup_cp2(3, 1)
    assert P.is_delzant
    assert not P.futaki_vanishes
    assert not P.A_ext.is_constant
    # defining property of A_ext
    for f in (1, x, y):
        assert P.L(f) == 0
    # Futaki = sigma(∂P) (boundary barycentre - area barycentre)
    bb, ab = P.boundary_barycenter, P.area_barycenter
    assert P.futaki == (P.boundary_measure * (bb[0] - ab[0]), P.boundary_measure * (bb[1] - ab[1]))


def test_exact_moments():
    P = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    assert P.integrate(x**2 * y) == Fraction(8, 3) * Fraction(1, 2)
    assert P.boundary_integrate(1) == 6


def test_crease_functional():
    P = blowup_cp2(3, 1)
    for a, b, c in [(1, 0, -1), (1, 1, -2), (-1, 2, 1)]:
        # max(0, g) - max(0, -g) = g is affine, so L_{A_ext} of it vanishes
        assert P.L_crease(a, b, c) - P.L_crease(-a, -b, -c) == P.L(a * x + b * y + c) == 0
    assert P.L_crease(0, 0, -1) == 0  # f ≡ 0


def test_invariance_under_gl2z():
    P = blowup_cp2(3, 1)
    Q = P.transform(((1, 1), (0, 1)), shift=(5, -2))
    assert Q.S_hat == P.S_hat
    assert Q.is_equivalent(P)
    assert Q.L_crease(1, 0, -4) == P.L_crease(1, 1, 1)  # same crease pulled back


def test_normal_forms_identify_blowup_of_cp2_with_F1():
    assert blowup_cp2(3, 1).is_equivalent(hirzebruch(1, 1, 2))
    assert not hirzebruch(0, 2, 2).is_equivalent(hirzebruch(2, 2, 2))


def test_random_delzant():
    for seed in range(20):
        P, history = random_delzant(n_blowups=3, seed=seed)
        assert P.is_delzant
        assert len(history) >= 1
        assert all(c.denominator == 1 for v in P.vertices for c in v)
