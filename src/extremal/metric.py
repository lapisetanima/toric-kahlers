"""Toric Kähler metrics on a Delzant polygon, given by symplectic potentials u = u_G + h.

Curvature is evaluated numerically (vectorized numpy) from closed-form derivatives of
H = Hess(u): the Guillemin part is differentiated by hand, and `h` either symbolically with
sympy or, for a numeric potential (e.g. from the extremal solver), through its own `tensor` method.
`scalar_curvature_symbolic` gives an independent sympy route, used in the tests.
See CONVENTIONS.md for the formulas.
"""

from __future__ import annotations

import itertools

import numpy as np
import sympy as sp

from .exact import X, Y
from .polygon import Polygon


def _tensor_function(expr: sp.Expr, order: int):
    """Return f(x, y) -> array (N, 2, ..., 2) with all order-`order` partials of expr."""
    idx = list(itertools.product(range(2), repeat=order))
    parts = []
    for multi in idx:
        d = expr
        for k in multi:
            d = sp.diff(d, (X, Y)[k])
        parts.append(d)
    if all(p == 0 for p in parts):
        return None
    fns = [sp.lambdify((X, Y), p, "numpy") for p in parts]

    def f(x, y):
        out = np.empty((x.size,) + (2,) * order)
        for multi, fn in zip(idx, fns):
            out[(slice(None),) + multi] = np.broadcast_to(fn(x, y), x.shape)
        return out

    return f


class ToricMetric:
    """The toric Kähler metric with symplectic potential u = 1/2 Σ l_i log l_i + h on `polygon`.

    `h` is a sympy expression in `extremal.x, extremal.y` (default 0: Guillemin's metric), or a
    numeric potential: any object with a method `tensor(order, x, y)` returning the array
    (N, 2, ..., 2) of all order-`order` partial derivatives at flattened points.
    All field methods take numpy arrays x, y of the same shape and return arrays of that shape.
    """

    def __init__(self, polygon: Polygon, h=0, name: str | None = None):
        # Non-Delzant or labelled polygons give orbifold metrics; the formulas are the same.
        self.polygon = polygon
        self._U = np.array([[float(c) for c in e.labelled_normal] for e in polygon.edges])
        self._lam = np.array([float(e.labelled_constant) for e in polygon.edges])
        if hasattr(h, "tensor"):
            self.h = h
            self.name = name or f"Guillemin + {h}"
            self._h_derivs = {k: (lambda x, y, k=k: h.tensor(k, x, y)) for k in (2, 3, 4)}
        else:
            self.h = sp.sympify(h)
            self.name = name or ("Guillemin" if self.h == 0 else f"Guillemin + ({self.h})")
            self._h_derivs = {k: _tensor_function(self.h, k) for k in (2, 3, 4)}

    # ------------------------------------------------------------------ core numerics

    def _hessian_derivatives(self, x: np.ndarray, y: np.ndarray):
        """H, ∂H, ∂∂H at points, shapes (N,2,2), (N,2,2,2), (N,2,2,2,2); last indices are derivatives."""
        U = self._U
        ell = x[:, None] * U[:, 0] + y[:, None] * U[:, 1] - self._lam  # (N, m)
        H = 0.5 * np.einsum("nk,ka,kb->nab", 1 / ell, U, U)
        dH = -0.5 * np.einsum("nk,ka,kb,kj->nabj", 1 / ell**2, U, U, U)
        d2H = np.einsum("nk,ka,kb,kj,ke->nabje", 1 / ell**3, U, U, U, U)
        for order, target in ((2, H), (3, dH), (4, d2H)):
            fn = self._h_derivs[order]
            if fn is not None:
                target += fn(x, y)
        return H, dH, d2H

    def _core(self, x, y):
        """Return (H, M) at flattened points, with M_eb = ∂_e V^b, V^b = -Σ_j ∂_j K^{bj}."""
        H, dH, d2H = self._hessian_derivatives(x, y)
        K = np.linalg.inv(H)
        dK = -np.einsum("nab,nbcj,ncd->nadj", K, dH, K)  # ∂_j K
        d2K = -(
            np.einsum("nabe,nbcj,ncd->nadje", dK, dH, K)
            + np.einsum("nab,nbcje,ncd->nadje", K, d2H, K)
            + np.einsum("nab,nbcj,ncde->nadje", K, dH, dK)
        )  # ∂_e ∂_j K
        M = -np.einsum("nbjje->neb", d2K)
        return H, M

    def _apply(self, x, y, fn):
        """Evaluate fn(H, M) at interior points; points on or outside ∂P give NaN."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shape = np.broadcast(x, y).shape
        xf, yf = (np.broadcast_to(a, shape).ravel() for a in (x, y))
        ell = xf[:, None] * self._U[:, 0] + yf[:, None] * self._U[:, 1] - self._lam
        inside = np.all(ell > 0, axis=1)
        out = np.full(xf.shape, np.nan)
        if inside.any():
            out[inside] = fn(*self._core(xf[inside], yf[inside]))
        return out.reshape(shape)

    def affine_gradient_and_laplacian(self, x, y, c1: float, c2: float):
        """|∇f|² and Δf for f = c0 + c1 x + c2 y (torus-invariant), Δ = div grad (analysts' sign).

        In momentum coordinates g = H dx² + H⁻¹ dθ² has Riemannian volume dx dθ, so for
        f = f(x): |∇f|² = K^{ij} f_i f_j and Δf = ∂_i (K^{ij} f_j), with K = H⁻¹.
        """
        c = np.array([c1, c2], float)

        def fn(xf, yf):
            H, dH, _ = self._hessian_derivatives(xf, yf)
            K = np.linalg.inv(H)
            dK = -np.einsum("nab,nbcj,ncd->nadj", K, dH, K)
            div = np.einsum("njbj->nb", dK)  # Σ_j ∂_j K^{jb}
            return np.einsum("nab,a,b->n", K, c, c), div @ c

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shape = np.broadcast(x, y).shape
        xf, yf = (np.broadcast_to(a, shape).ravel() for a in (x, y))
        ell = xf[:, None] * self._U[:, 0] + yf[:, None] * self._U[:, 1] - self._lam
        inside = np.all(ell > 0, axis=1)
        grad2, lap = np.full(xf.shape, np.nan), np.full(xf.shape, np.nan)
        if inside.any():
            grad2[inside], lap[inside] = fn(xf[inside], yf[inside])
        return grad2.reshape(shape), lap.reshape(shape)

    # ------------------------------------------------------------------ fields

    def scalar_curvature(self, x, y):
        """S = -1/2 Σ ∂_i ∂_j K^{ij} (Abreu normalization; half the Riemannian scalar curvature)."""
        return self._apply(x, y, lambda H, M: 0.5 * np.trace(M, axis1=1, axis2=2))

    def riemannian_scalar_curvature(self, x, y):
        return self._apply(x, y, lambda H, M: np.trace(M, axis1=1, axis2=2))

    def ricci_norm_sq(self, x, y):
        """|Ric|^2 with respect to g, over all four real dimensions."""
        return self._apply(x, y, lambda H, M: 0.5 * np.einsum("nab,nba->n", M, M))

    def traceless_ricci_norm_sq(self, x, y):
        """|Ric - (s/4) g|^2 = |Ric|^2 - s^2/4; zero exactly when the metric is Kähler–Einstein."""
        return self._apply(
            x, y, lambda H, M: 0.5 * np.einsum("nab,nba->n", M, M) - np.trace(M, axis1=1, axis2=2) ** 2 / 4
        )

    def futaki_kernel(self, x, y):
        """S - S_hat: its moments against x and y are the Futaki invariant."""
        return self.scalar_curvature(x, y) - float(self.polygon.S_hat)

    def extremal_defect(self, x, y):
        """S - A_ext: identically zero exactly for the extremal metric."""
        return self.scalar_curvature(x, y) - self.polygon.A_ext(np.asarray(x, float), np.asarray(y, float))

    def det_hessian(self, x, y):
        return self._apply(x, y, lambda H, M: np.linalg.det(H))

    def min_hessian_eigenvalue(self, x, y):
        """Smallest eigenvalue of Hess u; must be > 0 everywhere for u to define a metric."""
        return self._apply(x, y, lambda H, M: np.linalg.eigvalsh(H)[:, 0])

    FIELDS = {
        "S": ("scalar_curvature", "scalar curvature S"),
        "S - S_hat": ("futaki_kernel", "Futaki kernel S − Ŝ"),
        "S - A_ext": ("extremal_defect", "extremal defect S − A_ext"),
        "|Ric|^2": ("ricci_norm_sq", "|Ric|²"),
        "|Ric_0|^2": ("traceless_ricci_norm_sq", "|Ric − (s/4)g|²"),
        "det Hess u": ("det_hessian", "det Hess u"),
    }

    def field(self, name: str):
        """Look up a field by its short name (see FIELDS); returns (callable, label)."""
        method, label = self.FIELDS[name]
        return getattr(self, method), label

    # ------------------------------------------------------------------ symbolic route

    def potential(self) -> sp.Expr:
        if not isinstance(self.h, sp.Expr):
            raise TypeError("the symbolic route needs h to be a sympy expression")
        return sp.Rational(1, 2) * sum(e.ell_sympy() * sp.log(e.ell_sympy()) for e in self.polygon.edges) + self.h

    def scalar_curvature_symbolic(self) -> sp.Expr:
        """S = -1/2 Σ ∂_i ∂_j K^{ij} computed fully in sympy (slow; for checking)."""
        H = sp.hessian(self.potential(), (X, Y))
        K = H.inv()
        v = (X, Y)
        return -sp.Rational(1, 2) * sum(sp.diff(K[i, j], v[i], v[j]) for i in range(2) for j in range(2))

    def __repr__(self) -> str:
        return f"ToricMetric({self.name} on {self.polygon!r})"
