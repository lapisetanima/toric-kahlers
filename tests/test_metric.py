import numpy as np
import pytest
import sympy as sp

from extremal import ToricMetric, blowup_cp2, integrate_numeric, product, projective_plane, x, y
from extremal.quadrature import polygon_quadrature


def interior_points(P, n=40, seed=0):
    xs, ys, _ = polygon_quadrature(P, order=8)
    idx = np.random.default_rng(seed).choice(xs.size, n, replace=False)
    return xs[idx], ys[idx]


def test_fubini_study():
    g = ToricMetric(projective_plane(1))
    px, py = interior_points(g.polygon)
    assert np.allclose(g.scalar_curvature(px, py), 6)
    assert np.allclose(g.riemannian_scalar_curvature(px, py), 12)
    assert np.allclose(g.ricci_norm_sq(px, py), 36)  # Ric = 3 g in real dimension 4
    assert np.allclose(g.traceless_ricci_norm_sq(px, py), 0, atol=1e-8)
    assert np.allclose(g.futaki_kernel(px, py), 0, atol=1e-9)


def test_product_of_spheres_is_csc():
    g = ToricMetric(product(2, 3))
    px, py = interior_points(g.polygon)
    assert np.allclose(g.scalar_curvature(px, py), float(g.polygon.S_hat))


def test_numeric_matches_symbolic_curvature():
    g = ToricMetric(blowup_cp2(3, 1), h=x**3 / 30 + y**2 / 10)
    S_sym = sp.lambdify((x, y), g.scalar_curvature_symbolic(), "numpy")
    px, py = interior_points(g.polygon, n=10)
    assert np.allclose(g.scalar_curvature(px, py), S_sym(px, py), rtol=1e-8)


@pytest.mark.parametrize("h", [0, x**3 / 30 + y**2 / 10])
def test_futaki_kernel_moments_equal_futaki_invariant(h):
    g = ToricMetric(blowup_cp2(3, 1), h=h)
    P = g.polygon
    assert np.min(g.min_hessian_eigenvalue(*polygon_quadrature(P, 12)[:2])) > 0
    Fx = integrate_numeric(P, lambda a, b: g.futaki_kernel(a, b) * a, order=40)
    Fy = integrate_numeric(P, lambda a, b: g.futaki_kernel(a, b) * b, order=40)
    assert Fx == pytest.approx(float(P.futaki[0]), abs=1e-7)
    assert Fy == pytest.approx(float(P.futaki[1]), abs=1e-7)


def test_donaldson_integration_by_parts():
    """1/2 ∫ K^{ij} f_ij = L_S(f) = ∫_∂P f dsigma - ∫ S f, for f = x^2."""
    g = ToricMetric(blowup_cp2(3, 1), h=x**3 / 30)
    P = g.polygon
    qx, qy, w = polygon_quadrature(P, order=40)
    H, _, _ = g._hessian_derivatives(qx, qy)
    K = np.linalg.inv(H)
    lhs = 0.5 * np.sum(w * 2 * K[:, 0, 0])
    rhs = float(P.boundary_integrate(x**2)) - np.sum(w * g.scalar_curvature(qx, qy) * qx**2)
    assert lhs == pytest.approx(rhs, abs=1e-7)
