import numpy as np
import sympy as sp

from extremal import Polygon, ToricMetric, projective_plane
from extremal.exact import X, Y
from extremal.quadrature import polygon_quadrature
from extremal.solver import ExtremalSolver, PolynomialPotential

F1 = Polygon([(1, 0), (3, 0), (0, 3), (0, 1)])


def calabi_F1_hessian(x, y):
    """Calabi's extremal metric on F1 (this polygon): u = ½(x log x + y log y − w log w) + F(w), w = x + y,
    F'' = 1/Θ with Θ = (3 − w)(w − 1)(w² + 7w + 3)/(11 w). Then h = u − u_G depends on w only, and
    h_xx = h_xy = h_yy = −(w² + 5w + 3) / (2 w (w² + 7w + 3))."""
    w = x + y
    return -(w**2 + 5 * w + 3) / (2 * w * (w**2 + 7 * w + 3))


def test_basis_is_orthonormal_and_spans_affine_first():
    s = ExtremalSolver(F1, degree=10)
    Q = s.basis.derivatives(s.x, s.y, 0)[(0, 0)]
    assert np.abs(Q.T @ (s.w[:, None] * Q) - np.eye(Q.shape[1])).max() < 1e-12
    assert list(s.basis.total_degree[:3]) == [0, 1, 1]


def test_fubini_study_is_a_fixed_point():
    s = ExtremalSolver(projective_plane(1), degree=6)
    grad, hess = s.gradient_and_hessian(np.zeros(s.free.size))
    assert np.abs(grad).max() < 1e-12
    assert np.linalg.eigvalsh(hess).min() > 0
    r = s.solve()
    assert r.converged and np.abs(r.potential.coeffs).max() < 1e-12


def test_numeric_potential_matches_symbolic_metric():
    """ToricMetric with a numeric potential agrees with the sympy route for the same polynomial h."""
    s = ExtremalSolver(F1, degree=4)
    h_sym = sp.Rational(1, 30) * X**3 + sp.Rational(1, 10) * Y**2 - sp.Rational(1, 50) * X**2 * Y**2
    f = sp.lambdify((X, Y), h_sym)
    Q = s.basis.derivatives(s.x, s.y, 0)[(0, 0)]
    pot = PolynomialPotential(s.basis, Q.T @ (s.w * f(s.x, s.y)))
    pts = (np.array([1.0, 0.7, 2.0]), np.array([1.0, 0.8, 0.4]))
    S_num = ToricMetric(F1, h=pot).scalar_curvature(*pts)
    S_sym = ToricMetric(F1, h=h_sym).scalar_curvature(*pts)
    assert np.allclose(S_num, S_sym, atol=1e-10)


def test_recovers_calabi_metric_on_F1():
    x, y, _ = polygon_quadrature(F1, 30)
    exact = calabi_F1_hessian(x, y)
    errors = []
    for d in (8, 14):
        r = ExtremalSolver(F1, degree=d).solve()
        assert r.converged
        errors.append(max(np.abs(r.potential.derivative(a, b, x, y) - exact).max() for a, b in ((2, 0), (1, 1), (0, 2))))
    assert errors[1] < 1e-6 and errors[1] < errors[0] / 100
    assert r.defect()["sup"] < 1e-5
