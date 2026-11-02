"""No floating-point numbers may leak into exact results.

A sympy expression built from plain Python ints can silently pick up a float (2/12 in Python is 0.1666...);
that once cost the threshold λ* all but 17 of its digits. These tests walk the exact outputs and fail on
any sympy Float or Python float.
"""

from fractions import Fraction

import sympy as sp

from extremal import Polygon, ToricMetric, blowup_cp2, projective_plane
from extremal.ambitoric import conic
from extremal.parametric import ParametricPolygon

Q12 = Polygon([(0, 0), (5, 0), (4, 1), (0, 2)], labels=(1, 12, 12, 12))


def assert_exact(obj, where=""):
    if isinstance(obj, float):
        raise AssertionError(f"Python float in {where}: {obj!r}")
    if isinstance(obj, (tuple, list)):
        for i, o in enumerate(obj):
            assert_exact(o, f"{where}[{i}]")
        return
    if isinstance(obj, sp.MatrixBase):
        for i, o in enumerate(obj):
            assert_exact(o, f"{where}[{i}]")
        return
    if isinstance(obj, sp.Basic):
        floats = obj.atoms(sp.Float)
        assert not floats, f"sympy Float in {where}: {floats}"
        return
    assert isinstance(obj, (int, Fraction, sp.Rational)), f"unexpected type in {where}: {type(obj)}"


def test_polygon_invariants_are_exact():
    for P in (projective_plane(1), blowup_cp2(3, 1), Q12):
        A = P.A_ext
        assert_exact([P.area, P.boundary_measure, P.S_hat, A.c0, A.c1, A.c2, list(P.futaki)], P.name or "P")
        assert_exact([P.L_crease(1, 2, Fraction(-3, 2))], "L_crease")
        assert_exact([e.ell_sympy() for e in P.edges], "ell_sympy")


def test_parametric_results_are_exact():
    a, b, e = sp.symbols("a b e", positive=True)
    P = Polygon([(1, 0), (3, 0), (3, 2), (0, 2), (0, 1)])
    Pp = ParametricPolygon.from_polygon(P, {1: -a, 2: -b, 4: e}, params=(a, b, e))
    assert_exact([Pp.area, Pp.boundary_measure, Pp.S_hat, list(Pp.futaki), Pp.A_ext, Pp.calabi_energy], "parametric")
    assert_exact([list(v) for v in Pp.vertices] + list(Pp.lattice_lengths), "parametric vertices")


def test_symbolic_metric_is_exact():
    g = ToricMetric(projective_plane(1))
    assert_exact(g.potential(), "potential")
    assert_exact(sp.simplify(g.scalar_curvature_symbolic()), "scalar curvature")


def test_ambitoric_conic_is_exact():
    assert_exact(conic(Q12), "conic")
