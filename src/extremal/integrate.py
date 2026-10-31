"""Exact integration of polynomials over convex polygons given by CCW vertex lists.

These functions do not assume the Delzant condition, so they also work on pieces of a
polygon (for example the part cut off by a crease).
"""

from __future__ import annotations

from fractions import Fraction

from .exact import Point, Poly, segment_integral


def area(verts: list[Point]) -> Fraction:
    n = len(verts)
    return sum(
        (verts[i][0] * verts[(i + 1) % n][1] - verts[(i + 1) % n][0] * verts[i][1] for i in range(n)),
        Fraction(0),
    ) / 2


def area_integral(verts: list[Point], poly: Poly) -> Fraction:
    """∫_P poly dmu for a CCW polygon, via Green's theorem: ∫ x^a y^b = ∮ x^(a+1) y^b / (a+1) dy."""
    if len(verts) < 3:
        return Fraction(0)
    n = len(verts)
    total = Fraction(0)
    for i in range(n):
        p, q = verts[i], verts[(i + 1) % n]
        dy = q[1] - p[1]
        if dy == 0:
            continue
        primitive_in_x = {(a + 1, b): c / (a + 1) for (a, b), c in poly.items()}
        total += dy * segment_integral(p, q, primitive_in_x)
    return total


def clip_halfplane(verts: list[Point], a: Fraction, b: Fraction, c: Fraction) -> list[Point]:
    """Intersect a CCW convex polygon with {a x + b y + c >= 0} (Sutherland–Hodgman, exact)."""
    out: list[Point] = []
    n = len(verts)
    g = [a * v[0] + b * v[1] + c for v in verts]
    for i in range(n):
        p, q = verts[i], verts[(i + 1) % n]
        gp, gq = g[i], g[(i + 1) % n]
        if gp >= 0:
            out.append(p)
        if (gp > 0 and gq < 0) or (gp < 0 and gq > 0):
            t = gp / (gp - gq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    # drop consecutive duplicates (vertices lying exactly on the line)
    dedup: list[Point] = []
    for v in out:
        if not dedup or dedup[-1] != v:
            dedup.append(v)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup
