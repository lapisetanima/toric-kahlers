"""Numerical integration of smooth functions over a convex polygon (Gauss–Legendre on a fan triangulation)."""

from __future__ import annotations

import numpy as np

from .polygon import Polygon


def polygon_quadrature(polygon: Polygon, order: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes (x, y) and weights for ∫_P f dmu. All nodes lie strictly inside P.

    Each fan triangle (v0, vi, vi+1) is the image of the unit square under the Duffy map
    p = v0 + s (vi - v0) + s t (vi+1 - vi), with Jacobian s * |det|.
    """
    g, w = np.polynomial.legendre.leggauss(order)
    g, w = (g + 1) / 2, w / 2
    s, t = np.meshgrid(g, g, indexing="ij")
    ws = np.outer(w, w) * s
    V = np.array([[float(c) for c in v] for v in polygon.vertices])
    xs, ys, wts = [], [], []
    for i in range(1, len(V) - 1):
        a, b, c = V[0], V[i], V[i + 1]
        jac = abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        pts = a + s[..., None] * (b - a) + (s * t)[..., None] * (c - b)
        xs.append(pts[..., 0].ravel())
        ys.append(pts[..., 1].ravel())
        wts.append((ws * jac).ravel())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(wts)


def integrate_numeric(polygon: Polygon, fn, order: int = 24) -> float:
    """∫_P fn(x, y) dmu for a vectorized function fn."""
    x, y, w = polygon_quadrature(polygon, order)
    return float(np.sum(w * fn(x, y)))
