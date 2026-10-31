import random
from fractions import Fraction

import pytest
import sympy as sp

from extremal import Polygon, projective_plane
from extremal.exact import X, Y
from extremal.parametric import ParametricPolygon

a, b, e = sp.symbols("a b e", positive=True)


def pentagon():
    """CP1 x CP1 = [0, a] x [0, b] blown up at the origin by e."""
    P = Polygon([(1, 0), (3, 0), (3, 2), (0, 2), (0, 1)])
    # edges: 0 bottom (y >= 0), 1 right (x <= a), 2 top (y <= b), 3 left (x >= 0), 4 cut (x + y >= e)
    return ParametricPolygon.from_polygon(P, {1: -a, 2: -b, 4: e}, params=(a, b, e), name="CP1xCP1 # 1")


def q(v):
    return sp.Rational(v.numerator, v.denominator)


def test_simplex():
    t = sp.Symbol("t", positive=True)
    S = ParametricPolygon.from_polygon(projective_plane(1), {1: -t}, params=(t,))
    assert S.is_delzant
    assert sp.simplify(S.area - t**2 / 2) == 0
    assert sp.simplify(S.S_hat - 6 / t) == 0
    assert sp.simplify(S.A_ext - 6 / t) == 0
    assert S.futaki == (0, 0)


def test_agrees_with_polygon_at_rational_points():
    Pp = pentagon()
    assert Pp.is_delzant
    rng = random.Random(1)
    for _ in range(4):
        av, bv = Fraction(rng.randint(3, 9), rng.randint(1, 3)), Fraction(rng.randint(3, 9), rng.randint(1, 3))
        ev = min(av, bv) * Fraction(rng.randint(1, 9), 10)
        vals = {"a": av, "b": bv, "e": ev}
        assert Pp.is_valid_at(vals)
        P = Pp.at(vals)
        sub = {a: q(av), b: q(bv), e: q(ev)}
        assert P.is_delzant
        assert Pp.area.subs(sub) == q(P.area)
        assert Pp.boundary_measure.subs(sub) == q(P.boundary_measure)
        assert Pp.S_hat.subs(sub) == q(P.S_hat)
        assert tuple(c.subs(sub) for c in Pp.A_ext_coefficients) == tuple(q(c) for c in (P.A_ext.c0, P.A_ext.c1, P.A_ext.c2))
        assert tuple(f.subs(sub) for f in Pp.futaki) == tuple(q(f) for f in P.futaki)
        assert Pp.L(X**2 + X * Y).subs(sub) == q(P.L(X**2 + X * Y))
        assert Pp.calabi_energy.subs(sub) == q(P.integrate(P.A_ext.sympy() ** 2))


def test_A_ext_kills_affine_functions_identically():
    Pp = pentagon()
    for f in (1, X, Y):
        assert sp.simplify(Pp.L(f)) == 0


def test_validity_constraints():
    Pp = pentagon()
    assert Pp.is_valid_at(a=3, b=2, e=1)
    assert not Pp.is_valid_at(a=3, b=2, e=2)  # cut as large as the top-left edge: the left edge disappears
    with pytest.raises(ValueError):
        ParametricPolygon([(1, 0), (0, 1), (-1, 0), (0, -1)][::-1], [0, 0, -1, -1])
