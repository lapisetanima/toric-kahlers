"""Polygons whose Kähler class depends on symbolic parameters.

A `ParametricPolygon` is P = {<u_i, x> >= lambda_i} with fixed primitive inward normals u_i
(listed CCW) and support constants lambda_i given as sympy expressions in some parameters.
Fixing the normals fixes the toric surface; the lambda_i are its Kähler class. The Delzant
condition only involves the normals, so it holds for every parameter value or for none.

Invariants come out as exact rational functions of the parameters. This is sympy, so it is
much slower than `Polygon`; use `at(...)` to get an ordinary `Polygon` at a parameter value.
"""

from __future__ import annotations

from functools import cached_property

import sympy as sp

from .exact import X, Y
from .polygon import Polygon

T = sp.Symbol("t")


def _segment_integral(p, q, expr) -> sp.Expr:
    """∫_0^1 expr(p + t (q - p)) dt for a polynomial expr in X, Y."""
    along = sp.expand(expr.subs({X: p[0] + T * (q[0] - p[0]), Y: p[1] + T * (q[1] - p[1])}, simultaneous=True))
    return sp.Poly(along, T).integrate().eval(1)


def _affine_coefficients(A) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    A = sp.expand(sp.sympify(A))
    c1, c2 = A.diff(X), A.diff(Y)
    if (c1.free_symbols | c2.free_symbols) & {X, Y}:
        raise ValueError(f"weight {A} is not affine in x, y")
    return A.subs({X: 0, Y: 0}), c1, c2


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


class ParametricPolygon:
    """P = {<u_i, x> >= lambda_i}, normals fixed, constants symbolic.

    Edge i lies on {<u_i, x> = lambda_i} and runs from vertex i to vertex i+1, where vertex i
    is the intersection of edges i-1 and i (the same indexing as `Polygon`).
    """

    def __init__(self, normals, constants, params=None, name: str = ""):
        if len(normals) != len(constants) or len(normals) < 3:
            raise ValueError("need at least 3 normals, and one constant per normal")
        self.normals = tuple((int(u[0]), int(u[1])) for u in normals)
        self.constants = tuple(sp.sympify(c) for c in constants)
        free = set().union(*(c.free_symbols for c in self.constants))
        self.params = tuple(params) if params is not None else tuple(sorted(free, key=str))
        if free - set(self.params):
            raise ValueError(f"constants use symbols not in params: {free - set(self.params)}")
        self.name = name
        n = len(self.normals)
        for i in range(n):
            if _det(self.normals[i - 1], self.normals[i]) <= 0:
                raise ValueError(f"normals {i - 1} and {i} are not in CCW order")

    @classmethod
    def from_polygon(cls, P: Polygon, constants: dict, params=None, name: str = "") -> ParametricPolygon:
        """Start from a `Polygon` and replace some support constants: {edge index: expression}."""
        lams = [sp.Rational(e.constant.numerator, e.constant.denominator) for e in P.edges]
        for i, c in constants.items():
            lams[i] = sp.sympify(c)
        return cls([e.normal for e in P.edges], lams, params=params, name=name or P.name)

    @property
    def n(self) -> int:
        return len(self.normals)

    # ------------------------------------------------------------------ combinatorics

    @cached_property
    def directions(self) -> tuple[tuple[int, int], ...]:
        """Primitive edge directions (the inverse of `Polygon`'s rule u = (-e1, e0))."""
        return tuple((u[1], -u[0]) for u in self.normals)

    @cached_property
    def vertex_determinants(self) -> tuple[int, ...]:
        d = self.directions
        return tuple(_det(d[i - 1], d[i]) for i in range(self.n))

    @cached_property
    def is_delzant(self) -> bool:
        return all(v == 1 for v in self.vertex_determinants)

    @cached_property
    def vertices(self) -> tuple[tuple[sp.Expr, sp.Expr], ...]:
        out = []
        for i in range(self.n):
            (a1, b1), (a2, b2) = self.normals[i - 1], self.normals[i]
            l1, l2 = self.constants[i - 1], self.constants[i]
            det = a1 * b2 - a2 * b1
            out.append((sp.expand((l1 * b2 - l2 * b1) / det), sp.expand((a1 * l2 - a2 * l1) / det)))
        return tuple(out)

    @cached_property
    def lattice_lengths(self) -> tuple[sp.Expr, ...]:
        out = []
        for i, e in enumerate(self.directions):
            p, q = self.vertices[i], self.vertices[(i + 1) % self.n]
            out.append(sp.expand(((q[0] - p[0]) * e[0] + (q[1] - p[1]) * e[1]) / (e[0] ** 2 + e[1] ** 2)))
        return tuple(out)

    def constraints(self) -> tuple[sp.Expr, ...]:
        """Expressions that must all be > 0 for the combinatorial type to be the given one."""
        return self.lattice_lengths

    def is_valid_at(self, values: dict | None = None, **kw) -> bool:
        values = self._values(values, kw)
        return all(c.subs(values) > 0 for c in self.constraints())

    def _values(self, values: dict | None, kw: dict) -> dict:
        by_name = {p.name: p for p in self.params}
        merged = {**(values or {}), **kw}
        return {by_name[k] if isinstance(k, str) else k: sp.nsimplify(v) for k, v in merged.items()}

    def at(self, values: dict | None = None, **kw) -> Polygon:
        """The ordinary (exact, fast) `Polygon` at a rational parameter value: P.at(t=Fraction(1, 3))."""
        values = self._values(values, kw)
        verts = []
        for vx, vy in self.vertices:
            px, py = vx.subs(values), vy.subs(values)
            if not (px.is_Rational and py.is_Rational):
                raise ValueError(f"vertex ({px}, {py}) is not rational; give rational values for all parameters")
            verts.append((px, py))
        return Polygon(tuple(verts), name=self.name)

    # ------------------------------------------------------------------ measures & moments
    # Moments are kept as expanded polynomials in the parameters; rational functions are only
    # formed (and factored) once, in the invariants below. Factoring every moment is what is slow.

    def integrate(self, f) -> sp.Expr:
        """∫_P f dmu for a polynomial f in x, y, via Green's theorem: ∫ x^a y^b = ∮ x^(a+1) y^b / (a+1) dy."""
        primitive_in_x = sp.integrate(sp.expand(sp.sympify(f)), X)
        total = sp.Integer(0)
        for i in range(self.n):
            p, q = self.vertices[i], self.vertices[(i + 1) % self.n]
            total += (q[1] - p[1]) * _segment_integral(p, q, primitive_in_x)
        return sp.expand(total)

    def boundary_integrate(self, f) -> sp.Expr:
        """∫_{∂P} f dsigma for a polynomial f."""
        f = sp.sympify(f)
        total = sp.Integer(0)
        for i in range(self.n):
            p, q = self.vertices[i], self.vertices[(i + 1) % self.n]
            total += self.lattice_lengths[i] * _segment_integral(p, q, f)
        return sp.expand(total)

    @cached_property
    def area(self) -> sp.Expr:
        return sp.factor(self.integrate(1))

    @cached_property
    def boundary_measure(self) -> sp.Expr:
        return sp.factor(sum(self.lattice_lengths))

    # ------------------------------------------------------------------ invariants

    @cached_property
    def S_hat(self) -> sp.Expr:
        return sp.factor(self.boundary_measure / self.area)

    @cached_property
    def futaki(self) -> tuple[sp.Expr, sp.Expr]:
        """(F(x), F(y)) with F(f) = L_{S_hat}(f)."""
        return (self.L(X, A=self.S_hat), self.L(Y, A=self.S_hat))

    @cached_property
    def _A_ext_cramer(self) -> tuple[tuple[sp.Expr, sp.Expr, sp.Expr], sp.Expr, tuple[sp.Expr, sp.Expr, sp.Expr]]:
        """Numerators N_k, denominator D (polynomials) with A_ext coefficients N_k / D; also the rhs moments."""
        basis = [sp.Integer(1), X, Y]
        gram = sp.Matrix(3, 3, lambda i, j: self.integrate(basis[i] * basis[j]))
        rhs = sp.Matrix([self.boundary_integrate(p) for p in basis])
        D = sp.expand(gram.det(method="berkowitz"))
        nums = []
        for k in range(3):
            M = gram.copy()
            M[:, k] = rhs
            nums.append(sp.expand(M.det(method="berkowitz")))
        return tuple(nums), D, tuple(rhs)

    @cached_property
    def A_ext_coefficients(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        """(c0, c1, c2) with A_ext = c0 + c1 x + c2 y, as rational functions of the parameters."""
        nums, D, _ = self._A_ext_cramer
        return tuple(sp.factor(sp.cancel(N / D)) for N in nums)

    @cached_property
    def A_ext(self) -> sp.Expr:
        """The extremal affine function, as an expression in x, y and the parameters."""
        c0, c1, c2 = self.A_ext_coefficients
        return c0 + c1 * X + c2 * Y

    def L(self, f, A=None) -> sp.Expr:
        """Donaldson's functional L_A(f) = ∫_∂P f dsigma - ∫_P A f dmu, for polynomial f.

        `A` must be affine in x, y (coefficients may depend on the parameters); it defaults to A_ext.
        """
        f = sp.sympify(f)
        moments = [self.integrate(f), self.integrate(X * f), self.integrate(Y * f)]
        if A is None:
            nums, D, _ = self._A_ext_cramer
            total = (D * self.boundary_integrate(f) - sum(N * m for N, m in zip(nums, moments))) / D
        else:
            c = _affine_coefficients(A)
            total = self.boundary_integrate(f) - sum(ck * m for ck, m in zip(c, moments))
        return sp.factor(sp.cancel(sp.together(total)))

    @cached_property
    def calabi_energy(self) -> sp.Expr:
        """∫_P A_ext² dmu: the Calabi energy ∫ S² dmu of an extremal metric in this class, if one exists.

        If G c = r is the system defining A_ext, then ∫ A_ext² = cᵀ G c = cᵀ r.
        """
        nums, D, rhs = self._A_ext_cramer
        return sp.factor(sp.cancel(sp.expand(sum(N * r for N, r in zip(nums, rhs))) / D))

    def __repr__(self) -> str:
        return f"ParametricPolygon({self.name + ': ' if self.name else ''}normals={self.normals}, λ={self.constants})"
