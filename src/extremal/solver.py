"""Numerical extremal metrics: minimize Donaldson's functional by Newton's method.

We look for u = u_G + h with h smooth on the closed polygon and S(u) = A_ext. This is the
Euler–Lagrange equation of

    F(u) = -∫_P log det Hess u dμ + 2 L_A(u),

since dF(f) = -∫ u^{ij} f_ij dμ + 2 L_A(f) = 2 ∫ (S(u) - A) f dμ by Donaldson's integration by
parts (½ ∫ u^{ij} f_ij = L_{S(u)}(f)). F is convex, and strictly convex modulo affine functions,
which it does not see. Writing Hess u = H_G (I + K_G Hess h) with K_G = H_G^{-1} and dropping the
h-independent term -∫ log det H_G leaves

    F(h) = -∫_P log det(I + K_G Hess h) dμ + 2 L_A(h),

whose integrand is smooth up to ∂P. With K = (Hess u)^{-1}:

    gradient  g_k  = -∫ tr(K Hess φ_k) dμ + 2 L_A(φ_k)
    Hessian   H_kl =  ∫ tr(K Hess φ_k K Hess φ_l) dμ.

Writing K = L Lᵀ and M_k = Lᵀ Hess φ_k L, H_kl = ∫ <M_k, M_l>, assembled as one matrix product.

The basis. h ranges over polynomials of total degree <= `degree`, modulo affine functions.
Monomials, and even Legendre polynomials on the bounding box, are exponentially ill-conditioned
on a polygon that does not fill its box. So the basis is orthonormalized in L²(P) by
"Vandermonde with Arnoldi" (Brubeck–Nakatsukasa–Trefethen, SIAM Review 2021): each new
polynomial is ξ or η times an earlier one, orthogonalized against all earlier ones on the
quadrature nodes. The recurrence is stored, so the basis and its derivatives can be evaluated
anywhere by replaying it.

The solver only uses the second-order weak form; the result is checked independently through
`ToricMetric`, which computes S from fourth derivatives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import legendre

from .metric import ToricMetric
from .polygon import Affine, Polygon
from .quadrature import polygon_quadrature


def _multi_indices(max_order: int) -> list[tuple[int, int]]:
    """(a, b) = (number of x-derivatives, number of y-derivatives), |a| + |b| <= max_order, by order."""
    return [(a, s - a) for s in range(max_order + 1) for a in range(s, -1, -1)]


class PolygonBasis:
    """Polynomials of total degree <= `degree`, orthonormal in the discrete L²(P) inner product.

    Coordinates are first mapped affinely so the bounding box becomes [-1, 1]². Basis function k
    is q_k = (v_k q_{p_k} - Σ_{i<k} r_ik q_i) / r_kk with v_k ∈ {ξ, η}; functions 0, 1, 2 span the
    affine functions.
    """

    def __init__(self, polygon: Polygon, degree: int, x, y, w):
        V = np.array([[float(c) for c in v] for v in polygon.vertices])
        x0, x1, y0, y1 = V[:, 0].min(), V[:, 0].max(), V[:, 1].min(), V[:, 1].max()
        self.box = (x0, x1, y0, y1)
        self.scale = (2 / (x1 - x0), 2 / (y1 - y0))
        self.degree = degree
        xi, eta = self._maps(x, y)
        sw = np.sqrt(w)
        # recipe: (parent, variable) with variable 0 = ξ, 1 = η, ordered by total degree
        recipe = [(-1, -1)]
        blocks = [[0]]
        for s in range(1, degree + 1):
            prev = blocks[-1]
            new = []
            for p in prev:
                recipe.append((p, 0))
                new.append(len(recipe) - 1)
            recipe.append((prev[-1], 1))
            new.append(len(recipe) - 1)
            blocks.append(new)
        n = len(recipe)
        W = np.empty((x.size, n))  # sqrt(w) * q_k at the nodes
        R = np.zeros((n, n))
        R[0, 0] = np.linalg.norm(sw)
        W[:, 0] = sw / R[0, 0]
        for k in range(1, n):
            p, var = recipe[k]
            v = (xi if var == 0 else eta) * W[:, p]
            r = W[:, :k].T @ v
            v = v - W[:, :k] @ r
            r2 = W[:, :k].T @ v  # re-orthogonalize once ("twice is enough")
            v = v - W[:, :k] @ r2
            R[:k, k] = r + r2
            R[k, k] = np.linalg.norm(v)
            W[:, k] = v / R[k, k]
        self.recipe, self.R, self.n = recipe, R, n
        self.total_degree = np.array([s for s, b in enumerate(blocks) for _ in b])

    def _maps(self, x, y):
        x0, x1, y0, y1 = self.box
        return (2 * np.asarray(x, float) - (x0 + x1)) / (x1 - x0), (2 * np.asarray(y, float) - (y0 + y1)) / (y1 - y0)

    def derivatives(self, x, y, max_order: int, chunk: int = 4096) -> dict[tuple[int, int], np.ndarray]:
        """{(a, b): array (npts, n)} with ∂_x^a ∂_y^b q_k at flattened points, for a + b <= max_order."""
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        if x.size > chunk:
            parts = [self.derivatives(x[i : i + chunk], y[i : i + chunk], max_order) for i in range(0, x.size, chunk)]
            return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        xi, eta = self._maps(x, y)
        alphas = _multi_indices(max_order)
        D = {al: np.zeros((x.size, self.n)) for al in alphas}
        D[(0, 0)][:, 0] = 1 / self.R[0, 0]
        for k in range(1, self.n):
            p, var = self.recipe[k]
            mult = xi if var == 0 else eta
            r, rkk = self.R[:k, k], self.R[k, k]
            for al in alphas:
                col = mult * D[al][:, p]
                if al[var] > 0:  # product rule: ∂^α (v q) = v ∂^α q + α_v ∂^(α - e_v) q
                    lower = (al[0] - 1, al[1]) if var == 0 else (al[0], al[1] - 1)
                    col = col + al[var] * D[lower][:, p]
                D[al][:, k] = (col - D[al][:, :k] @ r) / rkk
        sx, sy = self.scale
        return {al: (sx ** al[0] * sy ** al[1]) * D[al] for al in alphas}


class PolynomialPotential:
    """h = Σ c_k q_k for a `PolygonBasis` (the affine functions q_0, q_1, q_2 have coefficient 0)."""

    def __init__(self, basis: PolygonBasis, coeffs: np.ndarray):
        self.basis = basis
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.degree = basis.degree

    def derivative(self, a: int, b: int, x, y) -> np.ndarray:
        """∂_x^a ∂_y^b h at points x, y (any matching shapes)."""
        shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
        xs, ys = (np.broadcast_to(np.asarray(v, float), shape).ravel() for v in (x, y))
        return (self.basis.derivatives(xs, ys, a + b)[(a, b)] @ self.coeffs).reshape(shape)

    def __call__(self, x, y):
        return self.derivative(0, 0, x, y)

    def _derivatives(self, x, y, order: int):
        """Basis derivatives up to order 4, cached for the last point set (ToricMetric asks for orders 2, 3, 4)."""
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        c = getattr(self, "_cache", None)
        if c is None or c[0].shape != x.shape or not (np.array_equal(c[0], x) and np.array_equal(c[1], y)):
            self._cache = c = (x.copy(), y.copy(), self.basis.derivatives(x, y, max(order, 4)))
        return c[2]

    def tensor(self, order: int, x, y) -> np.ndarray:
        """All order-`order` partials at flattened points, shape (N, 2, ..., 2) (the `ToricMetric` interface)."""
        D = self._derivatives(x, y, order)
        vals = {al: D[al] @ self.coeffs for al in D if sum(al) == order}
        out = np.empty((D[(0, 0)].shape[0],) + (2,) * order)
        for multi in np.ndindex(*(2,) * order):
            b = sum(multi)
            out[(slice(None),) + multi] = vals[(order - b, b)]
        return out

    def __str__(self) -> str:
        return f"polynomial of degree {self.degree}"


@dataclass
class SolveResult:
    polygon: Polygon
    potential: PolynomialPotential
    converged: bool
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def metric(self) -> ToricMetric:
        return ToricMetric(self.polygon, h=self.potential, name=f"extremal (degree {self.potential.degree})")

    def defect(self, order: int = 40, margin: float = 1e-2) -> dict:
        """sup and L² (area-normalized) of S − A_ext at quadrature nodes at lattice distance >= margin from ∂P.

        Very close to ∂P, evaluating S through fourth derivatives loses digits to cancellation
        (even for the exact Fubini–Study metric), so those points are left out of the sup.
        """
        x, y, w = polygon_quadrature(self.polygon, order)
        U = np.array([e.normal for e in self.polygon.edges], float)
        lam = np.array([float(e.constant) for e in self.polygon.edges])
        keep = (x[:, None] * U[:, 0] + y[:, None] * U[:, 1] - lam).min(axis=1) >= margin
        d = self.metric.extremal_defect(x[keep], y[keep])
        return {"sup": float(np.max(np.abs(d))), "L2": float(np.sqrt(np.sum(w[keep] * d**2) / np.sum(w[keep])))}

    def __str__(self) -> str:
        lines = [f"{'iter':>4} {'F':>22} {'decrement²':>12} {'step':>6}"]
        lines += [f"{r['iter']:>4} {r['F']:>22.15f} {r['decrement2']:>12.3e} {r['step']:>6.3f}" for r in self.history]
        status = "converged" if self.converged else "NOT converged"
        lines.append(f"{status} in {len(self.history)} Newton steps, {self.seconds:.2f} s")
        return "\n".join(lines)


class ExtremalSolver:
    """Newton's method for Donaldson's functional on polynomial potentials of a given degree.

    `degree` is the maximal total degree of h; there are (d+1)(d+2)/2 - 3 unknowns.
    `quad_order` is the number of Gauss points per direction on each fan triangle
    (default degree + 12). `A` defaults to A_ext; any affine A (or a constant) is allowed.
    """

    def __init__(self, polygon: Polygon, degree: int = 12, A=None, quad_order: int | None = None):
        if not polygon.is_delzant:
            raise ValueError("the solver needs a Delzant polygon")
        if degree < 2:
            raise ValueError("degree must be at least 2")
        self.polygon = polygon
        self.degree = degree
        self.A = polygon.A_ext if A is None else A
        self.quad_order = quad_order or degree + 12

        x, y, w = polygon_quadrature(polygon, self.quad_order)
        self.x, self.y, self.w = x, y, w
        self.basis = PolygonBasis(polygon, degree, x, y, w)
        self.free = np.arange(3, self.basis.n)  # drop the affine functions

        U = np.array([e.normal for e in polygon.edges], float)
        lam = np.array([float(e.constant) for e in polygon.edges])
        ell = x[:, None] * U[:, 0] + y[:, None] * U[:, 1] - lam
        self.HG = 0.5 * np.einsum("qk,ka,kb->qab", 1 / ell, U, U)
        self.KG = np.linalg.inv(self.HG)
        D = self.basis.derivatives(x, y, 2)
        self.Bxx, self.Bxy, self.Byy = (D[al][:, self.free] for al in ((2, 0), (1, 1), (0, 2)))
        self.Lvec = self._L_of_basis(D[(0, 0)][:, self.free])

    def _L_of_basis(self, interior_values) -> np.ndarray:
        """L_A(φ_k) = ∫_∂P φ_k dσ − ∫_P A φ_k dμ for every basis function."""
        g, gw = legendre.leggauss(self.quad_order)
        t, tw = (g + 1) / 2, gw / 2
        boundary = np.zeros(self.free.size)
        for e in self.polygon.edges:
            p, q = np.array(e.start, float), np.array(e.end, float)
            pts = p + t[:, None] * (q - p)
            vals = self.basis.derivatives(pts[:, 0], pts[:, 1], 0)[(0, 0)][:, self.free]
            boundary += float(e.lattice_length) * tw @ vals
        if isinstance(self.A, Affine):
            a_vals = self.A(self.x, self.y)
        else:
            a_vals = float(self.A) * np.ones_like(self.x)
        return boundary - (self.w * a_vals) @ interior_values

    # ------------------------------------------------------------------ the functional

    def hessian_of_u(self, c: np.ndarray) -> np.ndarray:
        H = self.HG.copy()
        hxy = self.Bxy @ c
        H[:, 0, 0] += self.Bxx @ c
        H[:, 0, 1] += hxy
        H[:, 1, 0] += hxy
        H[:, 1, 1] += self.Byy @ c
        return H

    def objective(self, c: np.ndarray) -> float:
        """F(h) up to an h-independent constant; +inf where Hess u is not positive definite."""
        hxx, hxy, hyy = self.Bxx @ c, self.Bxy @ c, self.Byy @ c
        K = self.KG
        m00 = K[:, 0, 0] * hxx + K[:, 0, 1] * hxy
        m01 = K[:, 0, 0] * hxy + K[:, 0, 1] * hyy
        m10 = K[:, 1, 0] * hxx + K[:, 1, 1] * hxy
        m11 = K[:, 1, 0] * hxy + K[:, 1, 1] * hyy
        det = (1 + m00) * (1 + m11) - m01 * m10
        if np.any(det <= 0) or np.any(self.HG[:, 0, 0] + hxx <= 0):
            return np.inf
        return float(-self.w @ np.log(det) + 2 * self.Lvec @ c)

    def gradient_and_hessian(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = np.linalg.inv(self.hessian_of_u(c))
        w = self.w
        grad = -(w * K[:, 0, 0]) @ self.Bxx - 2 * (w * K[:, 0, 1]) @ self.Bxy - (w * K[:, 1, 1]) @ self.Byy
        grad += 2 * self.Lvec
        # K = L Lᵀ (2x2 Cholesky); M_k = Lᵀ Hess φ_k L; tr(K Φ_k K Φ_l) = <M_k, M_l>_Frobenius
        l00 = np.sqrt(K[:, 0, 0])
        l10 = K[:, 1, 0] / l00
        l11 = np.sqrt(K[:, 1, 1] - l10**2)
        sw = np.sqrt(w)[:, None]
        m00 = sw * ((l00**2)[:, None] * self.Bxx + (2 * l00 * l10)[:, None] * self.Bxy + (l10**2)[:, None] * self.Byy)
        m01 = sw * (l11[:, None] * (l00[:, None] * self.Bxy + l10[:, None] * self.Byy))
        m11 = sw * ((l11**2)[:, None] * self.Byy)
        hess = m00.T @ m00 + 2 * (m01.T @ m01) + m11.T @ m11
        return grad, hess

    # ------------------------------------------------------------------ Newton

    def solve(self, c0=None, tol: float = 1e-28, max_iter: int = 60, verbose: bool = False) -> SolveResult:
        """Damped Newton with backtracking (Armijo).

        Stops when the Newton decrement² < tol, or when it stops shrinking once it is below 1e-18
        (round-off). The tolerance must be tiny: near the solution the decrement² is about
        ‖S − A‖² divided by the (large) Hessian of the high modes, so a loose tolerance stops
        while S − A is still visibly nonzero. `c0` may be a previous `SolveResult`
        (e.g. at lower degree) to start from.
        """
        start = time.perf_counter()
        c = np.zeros(self.free.size)
        if isinstance(c0, SolveResult):
            c = self._project(c0.potential)
        elif c0 is not None:
            c = np.asarray(c0, float).copy()
        F = self.objective(c)
        if not np.isfinite(F):
            c, F = np.zeros(self.free.size), self.objective(np.zeros(self.free.size))
        history, converged = [], False
        for it in range(max_iter):
            grad, hess = self.gradient_and_hessian(c)
            d = 1 / np.sqrt(np.maximum(np.diag(hess), 1e-300))  # symmetric diagonal scaling
            try:
                Lc = np.linalg.cholesky(d[:, None] * hess * d[None, :])
                z = np.linalg.solve(Lc.T, np.linalg.solve(Lc, -d * grad))
            except np.linalg.LinAlgError:
                z = np.linalg.lstsq(d[:, None] * hess * d[None, :], -d * grad, rcond=None)[0]
            step = d * z
            decrement2 = float(-grad @ step)
            t = 1.0
            F_new = self.objective(c + step)
            while F_new > F - 1e-4 * t * decrement2 and t > 1e-10:
                t /= 2
                F_new = self.objective(c + t * step)
            history.append({"iter": it, "F": F, "decrement2": decrement2, "step": t})
            if verbose:
                print(f"{it:>3}  F = {F:.15f}   decrement² = {decrement2:.3e}   step = {t:.3g}")
            if t > 1e-10:
                c, F = c + t * step, F_new
            stalled = it > 0 and decrement2 < 1e-18 and decrement2 > 0.25 * history[-2]["decrement2"]
            if decrement2 < tol or stalled:
                converged = True
                break
            if t <= 1e-10:
                break
        pot = PolynomialPotential(self.basis, np.concatenate([np.zeros(3), c]))
        return SolveResult(self.polygon, pot, converged, history, time.perf_counter() - start)

    def _project(self, potential: PolynomialPotential) -> np.ndarray:
        """Coefficients in this basis of another polynomial potential (L²(P) projection at the nodes)."""
        vals = potential(self.x, self.y)
        Q = self.basis.derivatives(self.x, self.y, 0)[(0, 0)]
        coeffs = Q.T @ (self.w * vals)
        return coeffs[self.free]
