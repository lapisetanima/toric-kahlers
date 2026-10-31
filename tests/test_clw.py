"""Independent check of the Calabi-functional computation in Chen–LeBrun–Weber, §3.

X. Chen, C. LeBrun, B. Weber, "On conformally Kähler, Einstein manifolds", JAMS 21 (2008),
arXiv:0705.0710. They use the LeBrun–Simanca formulas for the Futaki invariant; here everything
is recomputed from the polygon with Donaldson's toric formulas.

Dictionary. CLW's class (β+ε)(F1+F2) − εE on CP2 # 2 CP2-bar is the square of lattice side β+ε
with a corner cut of lattice length ε (areas are 2π times lattice lengths). With s = 2S and
dμ_g = (2π)² dμ, their 𝒜 = (1/32π²) ∫ s² dμ_g is ½ ∫ A_ext² dμ.
"""

import pytest
import sympy as sp

from extremal import ParametricPolygon, Polygon
from extremal.exact import X, Y

b, d, x = sp.symbols("beta epsilon x", positive=True)

CLW_F = 3 * (32 + 176 * x + 318 * x**2 + 280 * x**3 + 132 * x**4 + 32 * x**5 + 3 * x**6) / (
    12 + 72 * x + 138 * x**2 + 120 * x**3 + 54 * x**4 + 12 * x**5 + x**6
)
CLW_VARIANCE = 12 * b**6 + 72 * b**5 * d + 138 * b**4 * d**2 + 120 * b**3 * d**3 + 54 * b**2 * d**4 + 12 * b * d**5 + d**6
CLW_FUTAKI = 4 * b * d * (d**2 / 3 + b * d + b**2)


@pytest.fixture(scope="module")
def clw():
    square = Polygon([(1, 0), (2, 0), (2, 2), (0, 2), (0, 1)])  # edges: bottom, right, top, left, cut
    return ParametricPolygon.from_polygon(square, {1: -(b + d), 2: -(b + d), 4: d}, params=(b, d))


def test_class_dictionary(clw):
    # (−1)-curves F_i − E have area β, E has area ε, generic fibres β + ε; [ω]² = 2(β+ε)² − ε²
    assert clw.lattice_lengths == (b, b + d, b + d, b, d)
    assert sp.expand(2 * clw.area - (2 * (b + d) ** 2 - d**2)) == 0


def test_calabi_functional_matches_clw(clw):
    assert sp.simplify(clw.calabi_energy / 2 - CLW_F.subs(x, d / b)) == 0


def test_lebrun_simanca_inputs(clw):
    """Both inputs agree up to constants (normalizations of the Hamiltonian t and of 2π)."""
    w2 = 2 * clw.area  # [ω]² in lattice units
    t = X + Y  # Hamiltonian of the diagonal circle action Ξ
    t0 = clw.integrate(t) / clw.area
    variance = clw.integrate(sp.expand((t - t0) ** 2))
    futaki = clw.futaki[0] + clw.futaki[1]  # ∫ (S − Ŝ) t dμ
    assert sp.simplify(w2 * variance / CLW_VARIANCE) == sp.Rational(1, 36)
    assert sp.simplify(w2 * futaki / CLW_FUTAKI) == sp.Rational(1, 2)


def test_critical_class():
    df = sp.diff(CLW_F, x)
    assert (CLW_F.subs(x, 0), df.subs(x, 0), sp.limit(CLW_F, x, sp.oo)) == (8, -4, 9)
    assert df.subs(x, 1) == sp.Rational(3 * 1152, 409**2)
    num = sp.Poly(sp.numer(sp.together(df)), x)
    assert num.is_irreducible and num.degree() == 10
    roots = [r for r in num.real_roots() if r > 0]
    assert len(roots) == 1
    x0 = roots[0]
    assert abs(sp.N(x0, 20) - sp.Float("0.95771280518776186069", 20)) < 1e-18
    assert CLW_F.subs(x, x0).evalf() < 8
