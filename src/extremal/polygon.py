"""The `Polygon` data class: vertices go in, exact toric invariants come out on demand.

Every derived quantity is a `cached_property`, so it is computed the first time it is
asked for and stored on the instance. `Polygon.computed()` lists what has been filled in.
See CONVENTIONS.md for the definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

import numpy as np
import sympy as sp

from . import integrate
from .exact import (
    X,
    Y,
    Point,
    Poly,
    as_fraction,
    as_point,
    det2,
    poly_from,
    poly_mul,
    primitive,
    segment_integral,
    solve_exact,
)


@dataclass(frozen=True)
class Edge:
    """Edge `index` runs from `start` to `end` (CCW). It lies on {<normal, x> = constant}.

    `label` m > 0 (default 1) makes the polygon labelled (Lerman–Tolman): the defining function
    is l = m (<normal, x> - constant), which enters Guillemin's potential ½ Σ l log l, and the
    boundary measure on the edge, normalized by dσ ∧ dl = dx, has total mass lattice_length / m.
    An integer m > 1 gives an orbifold, with cone angle 2π/m along the corresponding divisor.
    """

    index: int
    start: Point
    end: Point
    direction: tuple[int, int]  # primitive, start -> end
    lattice_length: Fraction
    normal: tuple[int, int]  # primitive, inward
    constant: Fraction  # lambda
    label: Fraction = Fraction(1)

    @property
    def labelled_normal(self) -> tuple[Fraction, Fraction]:
        return (self.label * self.normal[0], self.label * self.normal[1])

    @property
    def labelled_constant(self) -> Fraction:
        return self.label * self.constant

    @property
    def sigma_length(self) -> Fraction:
        """σ(edge) = lattice_length / label."""
        return self.lattice_length / self.label

    def ell(self, x, y):
        """Affine function l = m (<u, x> - lambda); works on Fractions, floats or numpy arrays."""
        base = self.normal[0] * x + self.normal[1] * y - self.constant
        if self.label == 1:
            return base
        return (float(self.label) if hasattr(base, "dtype") else self.label) * base

    def ell_sympy(self) -> sp.Expr:
        q = lambda c: sp.Rational(c.numerator, c.denominator)  # noqa: E731
        return q(self.label) * (self.normal[0] * X + self.normal[1] * Y - q(self.constant))


@dataclass(frozen=True)
class Affine:
    """A(x, y) = c0 + c1 x + c2 y with exact coefficients."""

    c0: Fraction
    c1: Fraction
    c2: Fraction

    def __call__(self, x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return float(self.c0) + float(self.c1) * np.asarray(x) + float(self.c2) * np.asarray(y)
        return self.c0 + self.c1 * x + self.c2 * y

    @property
    def poly(self) -> Poly:
        return {(0, 0): self.c0, (1, 0): self.c1, (0, 1): self.c2}

    @property
    def is_constant(self) -> bool:
        return self.c1 == 0 and self.c2 == 0

    def sympy(self) -> sp.Expr:
        q = lambda c: sp.Rational(c.numerator, c.denominator)  # noqa: E731
        return q(self.c0) + q(self.c1) * X + q(self.c2) * Y

    def __str__(self) -> str:
        return str(self.sympy())


@dataclass(frozen=True, eq=False)
class Polygon:
    """A convex lattice-normal polygon. Pass vertices in either orientation; they are stored CCW.

    Coordinates must be exact: ints, Fractions, or strings like "1/2".
    `labels` (optional) gives a positive label for each edge, edge i running from input vertex i
    to input vertex i+1; see `Edge`. Unlabelled means all labels are 1.
    """

    vertices: tuple[Point, ...]
    name: str = ""
    labels: tuple | None = None

    def __post_init__(self):
        verts = [as_point(v) for v in self.vertices]
        if len(verts) < 3:
            raise ValueError("a polygon needs at least 3 vertices")
        labels = [Fraction(1)] * len(verts) if self.labels is None else [as_fraction(m) for m in self.labels]
        if len(labels) != len(verts) or any(m <= 0 for m in labels):
            raise ValueError("labels must be positive, one per edge")
        if integrate.area(verts) < 0:
            verts = [verts[0]] + verts[:0:-1]  # reverse, keeping the first vertex first
            labels = labels[::-1]  # edge j of the reversed list is edge n-1-j of the input
        n = len(verts)
        for i in range(n):
            a, b, c = verts[i - 1], verts[i], verts[(i + 1) % n]
            turn = det2((b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1]))
            if turn <= 0:
                raise ValueError(f"vertices are not strictly convex at vertex {i} = {b}")
        object.__setattr__(self, "vertices", tuple(verts))
        object.__setattr__(self, "labels", None if all(m == 1 for m in labels) else tuple(labels))

    # ------------------------------------------------------------------ combinatorics

    @property
    def n(self) -> int:
        return len(self.vertices)

    @cached_property
    def edges(self) -> tuple[Edge, ...]:
        out = []
        for i in range(self.n):
            p, q = self.vertices[i], self.vertices[(i + 1) % self.n]
            e, length = primitive((q[0] - p[0], q[1] - p[1]))
            u = (-e[1], e[0])  # rotate +90°: inward for CCW
            lam = u[0] * p[0] + u[1] * p[1]
            label = self.labels[i] if self.labels else Fraction(1)
            out.append(Edge(i, p, q, e, length, u, lam, label))
        return tuple(out)

    @cached_property
    def vertex_determinants(self) -> tuple[int, ...]:
        """det(e_{i-1}, e_i) at each vertex i; all equal to 1 iff the polygon is Delzant."""
        return tuple(int(det2(self.edges[i - 1].direction, self.edges[i].direction)) for i in range(self.n))

    @property
    def is_labelled(self) -> bool:
        return self.labels is not None

    @cached_property
    def is_delzant(self) -> bool:
        return all(d == 1 for d in self.vertex_determinants)

    @cached_property
    def singular_vertices(self) -> tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.vertex_determinants) if d != 1)

    # ------------------------------------------------------------------ measures & moments

    @cached_property
    def area(self) -> Fraction:
        return integrate.area(list(self.vertices))

    @cached_property
    def boundary_measure(self) -> Fraction:
        """sigma(∂P): the total lattice length of the boundary (each edge divided by its label)."""
        return sum((e.sigma_length for e in self.edges), Fraction(0))

    def integrate(self, f) -> Fraction:
        """∫_P f dmu for a polynomial f (sympy expression in x, y, number, or Poly dict)."""
        return integrate.area_integral(list(self.vertices), poly_from(f))

    def boundary_integrate(self, f) -> Fraction:
        """∫_{∂P} f dsigma for a polynomial f."""
        poly = poly_from(f)
        return sum((e.sigma_length * segment_integral(e.start, e.end, poly) for e in self.edges), Fraction(0))

    @cached_property
    def area_barycenter(self) -> Point:
        return (self.integrate(X) / self.area, self.integrate(Y) / self.area)

    @cached_property
    def boundary_barycenter(self) -> Point:
        s = self.boundary_measure
        return (self.boundary_integrate(X) / s, self.boundary_integrate(Y) / s)

    # ------------------------------------------------------------------ invariants

    @cached_property
    def S_hat(self) -> Fraction:
        """Average scalar curvature sigma(∂P) / mu(P) (Abreu normalization)."""
        return self.boundary_measure / self.area

    @cached_property
    def futaki(self) -> tuple[Fraction, Fraction]:
        """(F(x), F(y)) with F(f) = L_{S_hat}(f)."""
        return (self.L(X, A=self.S_hat), self.L(Y, A=self.S_hat))

    @cached_property
    def futaki_vanishes(self) -> bool:
        return self.futaki == (0, 0)

    @cached_property
    def A_ext(self) -> Affine:
        """The extremal affine function: L_A(f) = 0 for all affine f."""
        basis = [{(0, 0): Fraction(1)}, {(1, 0): Fraction(1)}, {(0, 1): Fraction(1)}]
        gram = [[self.integrate(poly_mul(p, q)) for p in basis] for q in basis]
        rhs = [self.boundary_integrate(q) for q in basis]
        return Affine(*solve_exact(gram, rhs))

    def _weight(self, A) -> Poly:
        if A is None:
            return self.A_ext.poly
        if isinstance(A, Affine):
            return A.poly
        return poly_from(A)

    def L(self, f, A=None) -> Fraction:
        """Donaldson's functional L_A(f) = ∫_∂P f dsigma - ∫_P A f dmu, for polynomial f.

        `A` defaults to `A_ext`; it may be a number, an `Affine`, or a polynomial.
        """
        poly = poly_from(f)
        return self.boundary_integrate(poly) - self.integrate(poly_mul(self._weight(A), poly))

    def L_crease(self, a, b, c, A=None) -> Fraction:
        """L_A(f) for the simple crease function f = max(0, a x + b y + c), which is convex.

        A negative value with A = A_ext certifies relative K-instability.
        """
        a, b, c = as_fraction(a), as_fraction(b), as_fraction(c)
        g = {(0, 0): c, (1, 0): a, (0, 1): b}
        piece = integrate.clip_halfplane(list(self.vertices), a, b, c)
        interior = integrate.area_integral(piece, poly_mul(self._weight(A), g))
        boundary = Fraction(0)
        for e in self.edges:
            g0 = a * e.start[0] + b * e.start[1] + c
            g1 = a * e.end[0] + b * e.end[1] + c
            if g0 <= 0 and g1 <= 0:
                continue
            if g0 >= 0 and g1 >= 0:
                t0, t1 = Fraction(0), Fraction(1)
            else:
                t_star = g0 / (g0 - g1)
                t0, t1 = (Fraction(0), t_star) if g0 > 0 else (t_star, Fraction(1))
            gt0, gt1 = g0 + t0 * (g1 - g0), g0 + t1 * (g1 - g0)
            boundary += e.sigma_length * (t1 - t0) * (gt0 + gt1) / 2
        return boundary - interior

    # ------------------------------------------------------------------ constructions

    def blow_up(self, vertex: int, size) -> Polygon:
        """Toric blow-up: cut the corner at `vertex` at lattice distance `size` along both edges."""
        size = as_fraction(size)
        if self.vertex_determinants[vertex] != 1:
            raise ValueError(f"vertex {vertex} is not smooth; blow-up is only defined at Delzant vertices")
        e_in, e_out = self.edges[vertex - 1], self.edges[vertex]
        if not (0 < size < e_in.lattice_length and size < e_out.lattice_length):
            raise ValueError(
                f"size must be in (0, min({e_in.lattice_length}, {e_out.lattice_length})) for vertex {vertex}"
            )
        v = self.vertices[vertex]
        before = (v[0] - size * e_in.direction[0], v[1] - size * e_in.direction[1])
        after = (v[0] + size * e_out.direction[0], v[1] + size * e_out.direction[1])
        verts = list(self.vertices)
        verts[vertex : vertex + 1] = [before, after]
        name = f"{self.name}+blowup" if self.name else ""
        labels = None
        if self.labels:  # the new edge (the exceptional divisor) gets label 1
            labels = list(self.labels)
            labels[vertex:vertex] = [Fraction(1)]
        return Polygon(tuple(verts), name=name, labels=labels)

    def transform(self, matrix, shift=(0, 0)) -> Polygon:
        """Image under x -> matrix @ x + shift, with matrix in GL(2, Z)."""
        (a, b), (c, d) = matrix
        if abs(a * d - b * c) != 1 or not all(isinstance(m, int) for m in (a, b, c, d)):
            raise ValueError("matrix must be an integer matrix with determinant ±1")
        s = as_point(shift)
        verts = tuple((a * p[0] + b * p[1] + s[0], c * p[0] + d * p[1] + s[1]) for p in self.vertices)
        return Polygon(verts, name=self.name, labels=self.labels)

    def normal_form(self) -> tuple[Point, ...]:
        """Canonical vertex tuple up to GL(2, Z) and translation (requires Delzant).

        At each vertex, map the two primitive edge directions to (1, 0) and (0, 1) and put
        the vertex at the origin; return the lexicographically smallest resulting vertex list.
        """
        if not self.is_delzant:
            raise ValueError("normal form is only defined for Delzant polygons")
        if self.is_labelled:
            raise NotImplementedError("normal form ignores labels; not defined for labelled polygons")
        best = None
        for i in range(self.n):
            e_next = self.edges[i].direction
            e_prev = tuple(-k for k in self.edges[i - 1].direction)
            for ea, eb in ((e_next, e_prev), (e_prev, e_next)):
                det = ea[0] * eb[1] - eb[0] * ea[1]  # ±1
                inv = ((eb[1] * det, -eb[0] * det), (-ea[1] * det, ea[0] * det))
                v0 = self.vertices[i]
                img = [
                    (
                        inv[0][0] * (p[0] - v0[0]) + inv[0][1] * (p[1] - v0[1]),
                        inv[1][0] * (p[0] - v0[0]) + inv[1][1] * (p[1] - v0[1]),
                    )
                    for p in self.vertices
                ]
                verts = Polygon(tuple(img)).vertices
                k = verts.index((0, 0))
                candidate = verts[k:] + verts[:k]
                if best is None or candidate < best:
                    best = candidate
        return best

    def is_equivalent(self, other: Polygon) -> bool:
        return self.normal_form() == other.normal_form()

    # ------------------------------------------------------------------ reporting

    def computed(self) -> list[str]:
        """Names of the cached quantities that have been computed so far."""
        return sorted(k for k in self.__dict__ if k not in ("vertices", "name", "labels"))

    def report(self) -> str:
        """A Markdown summary of the polygon and its invariants (computes them if needed)."""
        fmt = lambda p: f"({p[0]}, {p[1]})"  # noqa: E731
        lines = [f"### {self.name or 'Polygon'}", ""]
        lines.append(f"- vertices: {', '.join(fmt(v) for v in self.vertices)}")
        status = "yes" if self.is_delzant else f"no (singular vertices {list(self.singular_vertices)})"
        lines.append(f"- Delzant: {status}")
        if self.is_labelled:
            lines.append(f"- labels: {', '.join(str(e.label) for e in self.edges)} (an orbifold if integers > 1)")
        lines.append(f"- area μ(P) = {self.area},  boundary σ(∂P) = {self.boundary_measure}")
        lines.append(f"- average scalar curvature Ŝ = {self.S_hat}")
        lines.append(f"- A_ext = {self.A_ext}")
        F = self.futaki
        lines.append(f"- Futaki (F(x), F(y)) = ({F[0]}, {F[1]})" + ("  — vanishes" if self.futaki_vanishes else ""))
        lines += ["", "| edge | normal u | λ | lattice length | label | σ(edge) |", "|---|---|---|---|---|---|"]
        for e in self.edges:
            lines.append(f"| {e.index} | {e.normal} | {e.constant} | {e.lattice_length} | {e.label} | {e.sigma_length} |")
        return "\n".join(lines)

    def __repr__(self) -> str:
        verts = ", ".join(f"({p[0]}, {p[1]})" for p in self.vertices)
        lab = f", labels={tuple(str(m) for m in self.labels)}" if self.labels else ""
        return f"Polygon({self.name + ': ' if self.name else ''}{verts}{lab})"
