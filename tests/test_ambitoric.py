from fractions import Fraction

import pytest
import sympy as sp

from extremal import Polygon, blowup_cp2
from extremal.ambitoric import acg_margin, conic, edge_functions
from extremal.crease_exact import crease_minimum


def P(lam, k=12):
    lam = Fraction(lam)
    yv = (8 - lam) / 3
    return Polygon([(0, 0), (lam, 0), (lam - yv, yv), (0, 2)], labels=(1, k, k, k))


def test_conic_passes_through_the_normals_and_is_orthogonal_to_zeta():
    Q = conic(P(5))
    for f in edge_functions(P(5)):
        v = sp.Matrix([sp.Rational(c.numerator, c.denominator) for c in f])
        assert (v.T * Q * v)[0] == 0
    A = P(5).A_ext
    z = sp.Matrix([sp.Rational(c.numerator, c.denominator) for c in (A.c0, A.c1, A.c2)])
    assert (sp.Matrix([1, 0, 0]).T * Q * z)[0] == 0
    assert Q.det() != 0  # Veronese case


@pytest.mark.parametrize("lam", ["43/10", "433/100", "5", "7", "79/10", "795/100"])
def test_acg_criterion_agrees_with_the_exact_crease_minimum(lam):
    """ACG's coordinate lines + diagonals detect instability exactly when some simple crease does."""
    a = acg_margin(P(lam))["value"]
    c = float(crease_minimum(P(lam)).value)
    assert (a < 0) == (c < 0)


def test_stable_examples():
    assert acg_margin(blowup_cp2(3, 1))["value"] >= 0
    assert acg_margin(P(5, k=4))["value"] >= 0
