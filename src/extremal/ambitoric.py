"""Stability of labelled quadrilaterals by Apostolov–Calderbank–Gauduchon's criterion.

ACG, *Ambitoric geometry II: extremal toric surfaces and Einstein 4-orbifolds* (Ann. Sci. ENS 48, 2015;
arXiv:1302.6979), Thm 1 and Remark 6: a labelled quadrilateral admits an extremal metric iff it is
relatively K-polystable, and for that it suffices to test the simple creases along the two diagonals and
along the *coordinate lines* of an ambitoric factorization.

In affine-invariant terms: write affine functions f = c0 + c1 x + c2 y as vectors in h = R³. The conic
C(Δ, L) is the symmetric form Q on h, unique up to scale, with Q(ℓ_j, ℓ_j) = 0 for the four edge
functions and Q(1, ζ) = 0, where ζ ∝ A_ext is the extremal affine function (ACG §4.1, Lemma 4). The
coordinate lines are the zero sets {f = 0} of the null vectors Q(f, f) = 0: a one-parameter family,
parametrized here by an angle (lines in P(h) through the null vector ℓ_0).

This gives an independent check of `crease_exact`, which searches all simple creases: for P(λ) the
two agree on the threshold λ* to 40+ digits (notebook 04).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import sympy as sp

from .polygon import Polygon
from .stability import CreaseProfile

_IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def edge_functions(P: Polygon) -> list[tuple[Fraction, Fraction, Fraction]]:
    """The edge functions ℓ_j = <u_j, x> − λ_j as (c0, c1, c2) (labels only rescale them)."""
    return [(-e.constant, Fraction(e.normal[0]), Fraction(e.normal[1])) for e in P.edges]


def conic(P: Polygon) -> sp.Matrix:
    """The symmetric 3x3 matrix Q of the conic C(Δ, L), exact, normalized so its largest entry is ±1."""
    if P.n != 4:
        raise ValueError("ACG's criterion is for quadrilaterals")
    A = P.A_ext
    zeta = (A.c0, A.c1, A.c2)

    def null_row(v):
        return [v[i] * v[j] * (1 if i == j else 2) for i, j in _IDX]

    def orth_row(u, v):
        return [u[i] * v[j] + (u[j] * v[i] if i != j else 0) for i, j in _IDX]

    rows = [null_row(v) for v in edge_functions(P)] + [orth_row((1, 0, 0), zeta)]
    M = sp.Matrix([[sp.Rational(c.numerator, c.denominator) for c in map(Fraction, r)] for r in rows])
    ns = M.nullspace()
    if len(ns) != 1:
        raise ValueError(f"conic not unique (nullspace of dimension {len(ns)})")
    q = ns[0]
    q = q / max(q, key=abs)
    Q = sp.zeros(3, 3)
    for (i, j), c in zip(_IDX, q):
        Q[i, j] = Q[j, i] = c
    return Q


def coordinate_lines(P: Polygon, n: int = 720) -> np.ndarray:
    """n coordinate lines f = (c0, c1, c2), Q(f, f) = 0, sampled uniformly in the angle parametrization."""
    Q = np.array(conic(P), dtype=float)
    v0 = np.array([float(c) for c in edge_functions(P)[0]])
    # orthonormal complement of v0 (Euclidean), then f(t) = Q(w,w) v0 − 2 Q(v0,w) w with w = cos t a + sin t b
    _, _, Vt = np.linalg.svd(v0[None, :])
    a, b = Vt[1], Vt[2]
    t = np.pi * np.arange(n) / n
    w = np.cos(t)[:, None] * a + np.sin(t)[:, None] * b
    Qww = np.einsum("ni,ij,nj->n", w, Q, w)
    Qvw = np.einsum("i,ij,nj->n", v0, Q, w)
    return Qww[:, None] * v0[None, :] - 2 * Qvw[:, None] * w


def _crease_L(P: Polygon, f) -> float:
    """L_{A_ext}(max(0, f)) / |∇f| for f = (c0, c1, c2) (0 if the line misses the interior)."""
    c0, c1, c2 = (float(v) for v in f)
    g = math.hypot(c1, c2)
    if g == 0:
        return 0.0
    prof = CreaseProfile(P, (c1 / g, c2 / g))
    value = float(prof.L(-c0 / g))
    return 0.0 if abs(value) < 1e-12 else value  # lines (nearly) missing the interior give L = 0


def acg_margin(P: Polygon, n: int = 720, refine: int = 60) -> dict:
    """min of L(max(0, f))/|∇f| over the two diagonals and the coordinate lines (sampled, then refined).

    Negative means unstable, by a crease ACG's theorem singles out; by Thm 1 a polygon with a positive
    value on all of them (and nonzero values off trivial lines) is K-polystable. Returns the value, the
    line f, and which family it came from.
    """
    V = [(float(x), float(y)) for x, y in P.vertices]
    cands = []
    for i in (0, 1):  # diagonals v_i v_{i+2}
        (x0, y0), (x1, y1) = V[i], V[i + 2]
        f = (x0 * y1 - x1 * y0, y0 - y1, x1 - x0)  # vanishes at both vertices
        cands.append((_crease_L(P, f), f, f"diagonal {i}"))
    lines = coordinate_lines(P, n)
    vals = np.array([_crease_L(P, f) for f in lines])
    k = int(np.argmin(vals))
    best_t, best_v = math.pi * k / n, vals[k]
    if best_v < 0:  # golden-section refinement on the angle
        Q = np.array(conic(P), dtype=float)
        v0 = np.array([float(c) for c in edge_functions(P)[0]])
        _, _, Vt = np.linalg.svd(v0[None, :])
        a, b = Vt[1], Vt[2]

        def line(t):
            w = math.cos(t) * a + math.sin(t) * b
            return (w @ Q @ w) * v0 - 2 * (v0 @ Q @ w) * w

        lo, hi = best_t - math.pi / n, best_t + math.pi / n
        r = (math.sqrt(5) - 1) / 2
        for _ in range(refine):
            m1, m2 = hi - r * (hi - lo), lo + r * (hi - lo)
            if _crease_L(P, line(m1)) < _crease_L(P, line(m2)):
                hi = m2
            else:
                lo = m1
        best_t = (lo + hi) / 2
        best_v = _crease_L(P, line(best_t))
        lines_best = line(best_t)
    else:
        lines_best = lines[k]
    cands.append((best_v, tuple(lines_best), "coordinate line"))
    v, f, kind = min(cands, key=lambda c: c[0])
    return {"value": v, "line": f, "kind": kind}
