"""Numerical integration of smooth functions over a convex polygon (Gauss–Legendre on a fan triangulation)."""

from __future__ import annotations

import numpy as np

from .polygon import Polygon


def polygon_quadrature(
    polygon: Polygon, order: int = 24, scheme: str = "vertex", grading: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes (x, y) and weights for ∫_P f dmu. All nodes lie strictly inside P.

    scheme="vertex" (default): fan triangles (v0, vi, vi+1), each the image of the unit square
    under the Duffy map p = v0 + s (vi - v0) + s t (vi+1 - vi), with Jacobian s * |det|.

    scheme="edges": fan triangles (c, vi, vi+1) from the area barycenter c, one per edge, so that
    s -> 1 approaches every edge in the same way. With `grading` p > 1 the radial variable is
    s = 1 - (1 - σ)^p with σ Gauss–Legendre, which packs nodes towards the edges; use it to resolve
    boundary layers (nodes at distance ~ 1/order^(2p) from ∂P instead of ~ 1/order²).
    """
    g, w = np.polynomial.legendre.leggauss(order)
    g, w = (g + 1) / 2, w / 2
    V = np.array([[float(c) for c in v] for v in polygon.vertices])
    if scheme == "vertex":
        s, t = np.meshgrid(g, g, indexing="ij")
        ws = np.outer(w, w) * s
        triangles = [(V[0], V[i], V[i + 1]) for i in range(1, len(V) - 1)]
    elif scheme == "edges":
        sg = 1 - (1 - g) ** grading
        dsg = grading * (1 - g) ** (grading - 1)
        s, t = np.meshgrid(sg, g, indexing="ij")
        ws = np.outer(w * dsg, w) * s
        c = np.array([float(v) for v in polygon.area_barycenter])
        triangles = [(c, V[i], V[(i + 1) % len(V)]) for i in range(len(V))]
    else:
        raise ValueError("scheme must be 'vertex' or 'edges'")
    xs, ys, wts = [], [], []
    for a, b, c in triangles:
        jac = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        pts = a + s[..., None] * (b - a) + (s * t)[..., None] * (c - b)
        xs.append(pts[..., 0].ravel())
        ys.append(pts[..., 1].ravel())
        wts.append((ws * jac).ravel())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(wts)


def integrate_numeric(polygon: Polygon, fn, order: int = 24, **kwargs) -> float:
    """∫_P fn(x, y) dmu for a vectorized function fn (keyword arguments go to `polygon_quadrature`)."""
    x, y, w = polygon_quadrature(polygon, order, **kwargs)
    return float(np.sum(w * fn(x, y)))
