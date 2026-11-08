"""Standard Delzant polygons and a random generator.

Every Delzant polygon is, up to GL(2, Z) and translation, obtained from CP^2 or a
Hirzebruch surface F_n by a sequence of corner cuts (toric blow-ups). The generator
follows that recipe, so each random polygon comes with its construction history.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from .exact import as_fraction
from .polygon import Polygon


def projective_plane(a=1) -> Polygon:
    """CP^2: the simplex with vertices (0,0), (a,0), (0,a)."""
    a = as_fraction(a)
    return Polygon(((0, 0), (a, 0), (0, a)), name="CP2")


def hirzebruch(n: int, a=1, b=1) -> Polygon:
    """F_n: vertices (0,0), (a + n b, 0), (a, b), (0, b). n = 0 is CP1 x CP1, n = 1 is CP^2 blown up once."""
    if n < 0:
        raise ValueError("n must be >= 0")
    a, b = as_fraction(a), as_fraction(b)
    return Polygon(((0, 0), (a + n * b, 0), (a, b), (0, b)), name=f"F{n}")


def product(a=1, b=1) -> Polygon:
    """CP1 x CP1 with lattice side lengths a, b."""
    P = hirzebruch(0, a, b)
    return Polygon(P.vertices, name="CP1xCP1")


def blowup_cp2(a=3, size=1) -> Polygon:
    """CP^2 (simplex of size a) blown up at one fixed point, cut size `size`."""
    P = projective_plane(a).blow_up(0, size)
    return Polygon(P.vertices, name="CP2#1")


def cp2_two_blowups(s, t, k=1) -> Polygon:
    """CP^2 blown up at two fixed points: the unit simplex with the corners (1,0), (0,1) cut, sizes s, t.

    Edges x ≥ 0, y ≥ 0, x ≤ 1 − s, x + y ≤ 1, y ≤ 1 − t; inward normals (1,0), (0,1), (−1,0), (−1,−1),
    (0,−1). A pentagon for s, t > 0, s + t < 1. Label k on the edge x + y = 1, labels 1 elsewhere.
    s = 0 or t = 0 drops that cut: the quadrilateral (or, for both, the triangle) on the boundary of the family.
    """
    s, t, k = as_fraction(s), as_fraction(t), as_fraction(k)
    if not (s >= 0 and t >= 0 and s + t < 1):
        raise ValueError("need s, t >= 0 and s + t < 1")
    verts, labels = [(0, 0)], [1]  # edge from each vertex to the next
    verts += [(1 - s, 0), (1 - s, s)] if s else [(1, 0)]
    labels += [1, k] if s else [k]
    verts += [(t, 1 - t), (0, 1 - t)] if t else [(0, 1)]
    labels += [1, 1] if t else [1]
    return Polygon(tuple(verts), labels=tuple(labels), name=f"CP2#2 (s={s}, t={t}, k={k})")


def wang_zhou_nonagon(alpha=1000) -> Polygon:
    """The 9-vertex Delzant polygon P* of Wang–Zhou (B. Zhou's thesis, §3.2.2).

    Vertices (0,α*), (1,α*), (2,α*−1), (3,α*−3), (4,α*−6), (5,α*−10), (7,α), (7,−α), (0,−α),
    with α* = α + 20. It is shown there to be relatively K-unstable for α large, via horizontal
    creases max(0, y − α + k) with α ≫ k ≫ 1 (numerically, L → 72.476… − k/7 as α → ∞; the constant matches 1522/21 to 7 digits).

    That argument is asymptotic. Computed here: horizontal creases first destabilize for α between
    about 51571 and 51659, while tilted creases (normal at about 20°) already do for α between
    about 6854 and 6882. At α = 10000 no horizontal crease destabilizes, but a tilted one does.
    """
    a = as_fraction(alpha)
    s = a + 20
    verts = ((0, s), (1, s), (2, s - 1), (3, s - 3), (4, s - 6), (5, s - 10), (7, a), (7, -a), (0, -a))
    return Polygon(verts, name=f"Wang–Zhou nonagon (α={alpha})")


def random_delzant(
    n_blowups: int = 2,
    seed: int | None = None,
    integral: bool = True,
    max_n: int = 3,
    scale: int = 6,
) -> tuple[Polygon, list[str]]:
    """A random Delzant polygon built from CP^2 or F_n by `n_blowups` corner cuts.

    integral=True keeps all vertices in Z^2 (cut sizes are integers), which gives
    cleaner numbers; otherwise cut sizes are random rationals.
    Returns the polygon and a list describing how it was built.
    """
    rng = random.Random(seed)
    if rng.random() < 0.25:
        P = projective_plane(scale)
        history = [f"CP2(a={scale})"]
    else:
        n = rng.randint(0, max_n)
        a, b = rng.randint(2, scale), rng.randint(2, scale)
        P = hirzebruch(n, a, b)
        history = [f"F{n}(a={a}, b={b})"]

    for _ in range(n_blowups):
        candidates = []
        for i in range(P.n):
            room = min(P.edges[i - 1].lattice_length, P.edges[i].lattice_length)
            if (integral and room >= 2) or (not integral and room > 0):
                candidates.append((i, room))
        if not candidates:
            break
        i, room = rng.choice(candidates)
        if integral:
            size = Fraction(rng.randint(1, math.ceil(room) - 1))
        else:
            size = room * Fraction(rng.randint(1, 9), 10)
        P = P.blow_up(i, size)
        history.append(f"blow up vertex {i} by {size}")
    return Polygon(P.vertices, name="random"), history
