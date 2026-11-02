"""Exact critical simple creases: the optimal destabilizer and the boundary-normalized minimum.

A simple crease is f = max(0, ℓ), ℓ = n·x − s. Write the line as x + m y = s (chart 0, n = (1, m))
or m x + y = s (chart 1, n = (m, 1)). Fix the pair of edges (e_i, e_j) the line crosses: on that
cell of the line arrangement the crossing points p, q are rational in (m, s), and so is every
moment of the piece X = [p, v_{i+1}, ..., v_j, q] and of its boundary arc ∂P ∩ X. Every quantity
below is therefore a rational function of (m, s) with rational coefficients, and its critical
points are the common zeros of two polynomials: eliminate s by a resultant, isolate the real
roots of the resulting integer polynomial in m exactly (`AlgebraicReal`), and recover s the same way.

Two objectives (both scale invariant, so the chart does not matter):

* Optimal destabilizer (Donaldson 2005, Székelyhidi 2008). Minimize W(f) = L_A(f) + ½‖f‖² over
  convex f, A = A_ext. For f = c φ + affine with φ = max(0, ℓ), the minimum over c and the affine
  part is f = c φ⊥ (φ⊥ = φ minus its L²(μ)-projection to affine functions), c = −L_A(φ)/‖φ⊥‖²,
  and ‖f‖ = −L_A(φ)/‖φ⊥‖. So the optimal simple crease maximizes −L_A(φ)/‖φ⊥‖. Its critical
  points are the zeros of

      L^X_A(g) ‖φ⊥‖² − L_A(φ) ∫_X φ⊥ g dμ,   g = 1, x,    L^X_A(g) = ∫_{∂P∩X} g dσ − ∫_X A g dμ,

  equivalently: A − f is, on each piece, the extremal affine function of that piece with the
  measure σ on ∂P ∩ X and zero measure on the crease (which then forces A_1 = A_2 on the crease).

* Boundary-normalized minimum (the uniform-stability margin of `stability.crease_search`):
  minimize L_A(f)/∫_∂P f dσ over one-sided creases f that vanish on the side containing the
  area barycenter c. Candidates: critical points on each side, and the lines through c, where
  the normalization switches sides and the ratio jumps.

Both ratios are C¹ across the walls of the arrangement (lines through a vertex): the first
derivatives of L_A(φ), ‖φ⊥‖² and ∫_∂P φ dσ only involve masses of the two sides. So a critical
point on a wall is a solution of either adjacent cell's equations, extended to the closed cell.
Near the boundary of the space of creases (tiny corner cuts, thin slabs along an edge) the
optimal ratio tends to −∞ and the boundary-normalized ratio to 1, so neither extremum escapes.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction

import mpmath
import sympy as sp

from .polygon import Affine, Polygon

m_, s_ = sp.symbols("m s")
DPS = 50


def _q(c) -> sp.Rational:
    c = Fraction(c)
    return sp.Rational(c.numerator, c.denominator)


def _moments(V) -> list:
    """∫ over the CCW polygon V of 1, x, y, x², xy, y² (Lebesgue measure)."""
    out = [0] * 6
    for a in range(len(V)):
        (x0, y0), (x1, y1) = V[a], V[(a + 1) % len(V)]
        c = x0 * y1 - x1 * y0
        out[0] += c / 2
        out[1] += (x0 + x1) * c / 6
        out[2] += (y0 + y1) * c / 6
        out[3] += (x0 * x0 + x0 * x1 + x1 * x1) * c / 12
        out[4] += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * c / 24
        out[5] += (y0 * y0 + y0 * y1 + y1 * y1) * c / 12
    return out


def _gram(M) -> list:
    return [[M[0], M[1], M[2]], [M[1], M[3], M[4]], [M[2], M[4], M[5]]]


K_, mK, sK = sp.field("m,s", sp.QQ)  # the rational function field Q(m, s): fast exact arithmetic


def _k(c):
    c = Fraction(c)
    return K_(sp.QQ(c.numerator, c.denominator))


@dataclass
class _Cell:
    """Rational functions of (m, s) attached to one cell (edges i, j crossed; chart 0 or 1).

    Values in `ex` are elements of Q(m, s) (sympy FracElement)."""

    polygon: Polygon
    A: Affine
    i: int
    j: int
    chart: int
    ex: dict = field(init=False)
    _num: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        P, i, j = self.polygon, self.i, self.j
        V = [(_k(v[0]), _k(v[1])) for v in P.vertices]
        k = len(V)
        n = (K_(1), mK) if self.chart == 0 else (mK, K_(1))
        sig = [_k(e.sigma_length) for e in P.edges]
        Ac = [_k(self.A.c0), _k(self.A.c1), _k(self.A.c2)]

        def cut(e):
            P0, P1 = V[e], V[(e + 1) % k]
            d = (P1[0] - P0[0], P1[1] - P0[1])
            t = (sK - n[0] * P0[0] - n[1] * P0[1]) / (n[0] * d[0] + n[1] * d[1])
            return t, (P0[0] + t * d[0], P0[1] + t * d[1])

        ti, p = cut(i)
        tj, q = cut(j)
        inner_idx = [(i + 1 + r) % k for r in range((j - i) % k)]
        inner = [V[v] for v in inner_idx]
        X = [p] + inner + [q]
        segs = [(p, V[(i + 1) % k], sig[i] * (1 - ti))]
        segs += [(V[(i + r) % k], V[(i + r + 1) % k], sig[(i + r) % k]) for r in range(1, (j - i) % k)]
        segs.append((V[j], q, sig[j] * tj))
        bX = [sum((w for _, _, w in segs), K_(0)), sum((w * (a[0] + b[0]) / 2 for a, b, w in segs), K_(0)),
              sum((w * (a[1] + b[1]) / 2 for a, b, w in segs), K_(0))]
        GX = _gram(_moments(X))
        GPl = _gram(_moments(V))
        GPinv = sp.Matrix(_gram(_moments([(_q(v[0]), _q(v[1])) for v in P.vertices]))).inv()
        GPi = [[_k(Fraction(int(sp.numer(GPinv[r, c])), int(sp.denom(GPinv[r, c])))) for c in range(3)] for r in range(3)]
        bP = [_k(P.boundary_measure), _k(P.boundary_barycenter[0] * P.boundary_measure),
              _k(P.boundary_barycenter[1] * P.boundary_measure)]
        lc = [-sK, n[0], n[1]]  # ℓ = lc · (1, x, y)
        mphi = [sum((lc[r] * GX[r][c] for r in range(3)), K_(0)) for c in range(3)]  # ∫_X ℓ g
        proj = [sum((GPi[r][c] * mphi[c] for c in range(3)), K_(0)) for r in range(3)]  # Π(ℓ 1_X) = proj · (1, x, y)
        norm2 = sum((lc[r] * mphi[r] for r in range(3)), K_(0)) - sum((mphi[r] * proj[r] for r in range(3)), K_(0))
        L_phi = sum((lc[r] * bX[r] - Ac[r] * mphi[r] for r in range(3)), K_(0))
        LX = [bX[c] - sum((Ac[r] * GX[r][c] for r in range(3)), K_(0)) for c in range(3)]
        perp = [mphi[c] - sum((proj[r] * GX[r][c] for r in range(3)), K_(0)) for c in range(3)]
        L_full = sum((lc[r] * bP[r] for r in range(3)), K_(0)) - sum(
            (Ac[r] * lc[t] * GPl[t][r] for r in range(3) for t in range(3)), K_(0))
        B_X = sum((lc[r] * bX[r] for r in range(3)), K_(0))  # ∫_{∂P∩X} ℓ dσ
        B_full = sum((lc[r] * bP[r] for r in range(3)), K_(0))
        self.ex = dict(area_X=GX[0][0], ti=ti, tj=tj, L_phi=L_phi, norm2=norm2, B_X=B_X, L_full=L_full, B_full=B_full,
                       opt=[LX[c] * norm2 - L_phi * perp[c] for c in (0, 1)],
                       piece=dict(GX=GX, bX=bX, proj=proj),
                       inner=[tuple(_q(c) for c in P.vertices[v]) for v in inner_idx], n=(1, m_) if self.chart == 0 else (m_, 1))

    def num(self, name: str, *args):
        """Cached mpmath-callable for the expression `name` (a key of `ex`, or a key with an index)."""
        key = (name,) + args
        if key not in self._num:
            e = self.ex[name]
            for a in args:
                e = e[a]
            self._num[key] = _lambdify(e)
        return self._num[key]


def _lambdify(e):
    if hasattr(e, "numer"):  # FracElement: evaluate numerator and denominator separately
        N = sp.lambdify((m_, s_), e.numer.as_expr(), "mpmath")
        D = sp.lambdify((m_, s_), e.denom.as_expr(), "mpmath")
        return lambda a, b: N(a, b) / D(a, b)
    return sp.lambdify((m_, s_), e, "mpmath")


def _numerator(expr) -> sp.Poly | None:
    """Numerator of an element of Q(m, s), without its factors free of s (squarefree)."""
    num = expr.numer
    if not num:
        return None
    _, factors = num.factor_list()
    keep = [f for f, _ in factors if f.degree(sK.numer.ring.gens[1]) > 0]
    out = sp.Integer(1)
    for f in keep:
        out *= f.as_expr()
    return sp.Poly(out, s_, m_)


@dataclass(frozen=True)
class AlgebraicReal:
    """A real algebraic number, exactly: its minimal polynomial (irreducible over Q, primitive in Z[t])
    and a rational isolating interval [lo, hi] on which that polynomial changes sign, plus a
    high-precision value."""

    poly: sp.Poly
    lo: Fraction
    hi: Fraction
    value: mpmath.mpf

    def __float__(self) -> float:
        return float(self.value)


def _real_roots(poly: sp.Poly) -> list:
    """All real roots of a univariate integer polynomial, as `AlgebraicReal`s.

    Each irreducible factor is solved numerically (mpmath.polyroots, at high precision); the
    real roots are then certified: the number found must equal the Sturm count of real roots,
    and each gets a rational interval with a sign change of the factor. On any failure the
    factor falls back to sympy's exact isolation.
    """
    t = poly.gens[0]
    out = []
    for fac, _ in poly.factor_list()[1]:
        fac = sp.Poly(fac, t)
        if fac.degree() <= 0:
            continue
        if fac.degree() == 1:
            a, b = fac.all_coeffs()
            r = Fraction(-int(b), int(a))
            out.append(AlgebraicReal(fac, r, r, mpmath.mpf(r.numerator) / r.denominator))
            continue
        out.extend(_certified_roots(fac))
    return out


def _sign_at(poly: sp.Poly, x: Fraction) -> int:
    v = sum(Fraction(int(c)) * x ** k for (k,), c in poly.terms())
    return (v > 0) - (v < 0)


def _flint_roots(fac: sp.Poly) -> list | None:
    """Real roots via python-flint's rigorous complex root isolation (Arb), if flint is installed.

    For an irreducible (so squarefree) real polynomial, each isolating box meeting the real axis
    holds a real root (a non-real root would bring its conjugate); each is also checked by an
    exact sign change on a rational interval. Returns None if flint is missing or a check fails.
    """
    try:
        import flint
    except ImportError:
        return None
    F = flint.fmpz_poly([int(c) for c in reversed(fac.all_coeffs())])
    old = flint.ctx.dps
    flint.ctx.dps = DPS + 20
    try:
        boxes = [r for r, _ in F.complex_roots()]
    finally:
        flint.ctx.dps = old
    out = []
    for r in boxes:
        if not r.imag.contains(0):
            continue
        m, e = r.real.mid().man_exp()
        rm, re_ = r.real.rad().man_exp()
        mid = Fraction(int(m)) * Fraction(2) ** int(e)
        rad = Fraction(int(rm)) * Fraction(2) ** int(re_) * 2 + Fraction(1, 10 ** (DPS + 15))
        lo, hi = mid - rad, mid + rad
        if _sign_at(fac, lo) * _sign_at(fac, hi) >= 0:
            return None
        out.append(AlgebraicReal(fac, lo, hi, mpmath.mpf(mid.numerator) / mid.denominator))
    return out


def _certified_roots(fac: sp.Poly) -> list:
    fast = _flint_roots(fac)
    if fast is not None:
        return fast
    count = fac.count_roots()
    if count == 0:
        return []
    with mpmath.workdps(DPS + 20):
        try:
            zs = mpmath.polyroots([mpmath.mpf(int(c)) for c in fac.all_coeffs()], maxsteps=300, extraprec=4 * DPS)
        except mpmath.libmp.libhyper.NoConvergence:
            zs = None
        if zs is not None:
            real = sorted(mpmath.re(z) for z in zs if abs(mpmath.im(z)) < mpmath.mpf(10) ** (-DPS // 2) * max(1, abs(z)))
            if len(real) == count:
                out = []
                for k, r in enumerate(real):
                    gap = min([abs(r - z) for z in zs if abs(r - z) > 0] or [mpmath.mpf(1)])
                    delta = min(gap / 4, mpmath.mpf(10) ** (-DPS // 2) * max(1, abs(r)))
                    lo = Fraction(mpmath.nstr(r - delta, DPS + 10, min_fixed=-mpmath.inf, max_fixed=mpmath.inf))
                    hi = Fraction(mpmath.nstr(r + delta, DPS + 10, min_fixed=-mpmath.inf, max_fixed=mpmath.inf))
                    if _sign_at(fac, lo) * _sign_at(fac, hi) >= 0:
                        break
                    out.append(AlgebraicReal(fac, lo, hi, r))
                else:
                    return out
    out = []  # fallback: sympy's exact isolation
    for root in fac.real_roots():
        lo, hi = root._get_interval().a, root._get_interval().b
        out.append(AlgebraicReal(fac, Fraction(int(sp.numer(lo)), int(sp.denom(lo))),
                                 Fraction(int(sp.numer(hi)), int(sp.denom(hi))), mpmath.mpf(sp.N(root, DPS + 10))))
    return out


def _common_zeros(f1: sp.Poly, f2: sp.Poly) -> list:
    """Real common zeros (m, s) of two polynomials in (s, m): [(m exact, m, s)].

    Both coordinates are exact algebraic numbers: m is a real root of Res_s(f1, f2), s of
    Res_m(f1, f2), each isolated exactly by sympy; the pairs are matched by evaluating f1, f2
    at high precision (relative to the size of their terms).
    """
    if f1.degree(s_) <= 0 or f2.degree(s_) <= 0:
        return []
    g = sp.gcd(f1, f2)
    if g.degree(s_) > 0:  # a curve of critical points; drop it (not isolated) and keep the rest
        f1, f2 = sp.Poly(sp.quo(f1, g), s_, m_), sp.Poly(sp.quo(f2, g), s_, m_)
        if f1.degree(s_) <= 0 or f2.degree(s_) <= 0:
            return []
    Rm = sp.Poly(sp.resultant(f1.as_expr(), f2.as_expr(), s_), m_)
    Rs = sp.Poly(sp.resultant(f1.as_expr(), f2.as_expr(), m_), s_)
    if Rm.is_zero or Rs.is_zero:
        return []
    ms = [(r, r.value) for r in _real_roots(Rm)]
    ss = _real_roots(Rs)
    Fs = [sp.lambdify((m_, s_), f.as_expr(), "mpmath") for f in (f1, f2)]
    Fabs = [sp.lambdify((m_, s_), sum(abs(c) * s_ ** a * m_ ** b for (a, b), c in f.terms()), "mpmath") for f in (f1, f2)]
    tol = mpmath.mpf(10) ** (-DPS // 2)
    out = []
    for root, m0 in ms:
        for s_alg in ss:
            s0 = s_alg.value
            if all(abs(F(m0, s0)) <= tol * Fa(abs(m0), abs(s0)) for F, Fa in zip(Fs, Fabs)):
                out.append((root, m0, s0, s_alg))
    return out


def _univariate_zeros(f: sp.Expr) -> list:
    """Exact real roots of a univariate rational function's numerator in m."""
    num = sp.numer(sp.together(f))
    if num == 0:
        return []
    poly = sp.Poly(sp.expand(num), m_)
    if poly.degree() <= 0:
        return []
    return [(r, r.value) for r in _real_roots(poly)]


@dataclass
class CriticalCrease:
    """A critical simple crease max(0, n·x − s) found exactly. n = (1, m) (chart 0) or (m, 1) (chart 1)."""

    cell: tuple[int, int]
    chart: int
    m_exact: AlgebraicReal
    m: mpmath.mpf
    s: mpmath.mpf
    L: mpmath.mpf  # L_A(max(0, n·x − s))
    value: mpmath.mpf  # the objective at this crease
    kind: str = "critical"  # 'critical', or 'barycentric' for the jump lines (boundary-normalized)
    s_exact: AlgebraicReal | None = None

    @property
    def normal(self) -> tuple:
        return (mpmath.mpf(1), self.m) if self.chart == 0 else (self.m, mpmath.mpf(1))

    @property
    def theta(self) -> float:
        """Angle of the unit normal in [0, π), radians."""
        a, b = self.normal
        return float(mpmath.atan2(b, a)) % math.pi

    @property
    def unit_offset(self) -> float:
        """s for the unit normal at angle θ (the convention of stability.py)."""
        a, b = self.normal
        sign = 1 if float(mpmath.atan2(b, a)) % (2 * math.pi) < math.pi else -1
        return float(sign * self.s / mpmath.sqrt(a * a + b * b))

    def minimal_polynomial_m(self) -> sp.Poly:
        return self.m_exact.poly

    def __str__(self) -> str:
        a, b = self.normal
        line = f"x + {mpmath.nstr(b, 15)}·y = {mpmath.nstr(self.s, 15)}" if self.chart == 0 else \
            f"{mpmath.nstr(a, 15)}·x + y = {mpmath.nstr(self.s, 15)}"
        return f"{line}  (θ = {math.degrees(self.theta):.6f}°, edges {self.cell}, {self.kind}): value {mpmath.nstr(self.value, 12)}"


def _cells(P: Polygon, A: Affine):
    k = P.n
    for i, j in itertools.combinations(range(k), 2):
        for chart in (0, 1):
            yield _Cell(P, A, i, j, chart)


def _in_cell(cell: _Cell, m0, s0, tol=mpmath.mpf(10) ** -30) -> bool:
    """The line crosses the closed edges i, j and cuts P into two pieces of positive area."""
    try:
        ti, tj = cell.num("ti")(m0, s0), cell.num("tj")(m0, s0)
        aX = cell.num("area_X")(m0, s0)
    except ZeroDivisionError:
        return False
    area = mpmath.mpf(sp.N(_q(cell.polygon.area), DPS))
    eps = mpmath.mpf(10) ** -20 * area
    return -tol <= ti <= 1 + tol and -tol <= tj <= 1 + tol and eps < abs(aX) < area - eps


def _side_sign(cell: _Cell, m0, s0) -> int:
    """Sign of ℓ on the piece X (evaluated at its vertex farthest from the line)."""
    n = cell.ex["n"]
    vals = []
    for v in cell.ex["inner"]:
        a = mpmath.mpf(1) if n[0] == 1 else m0
        b = m0 if n[0] == 1 else mpmath.mpf(1)
        vals.append(a * mpmath.mpf(sp.N(v[0], DPS)) + b * mpmath.mpf(sp.N(v[1], DPS)) - s0)
    best = max(vals, key=abs)
    return 1 if best > 0 else -1


def _dedupe(found: list) -> list:
    out = []
    for c in found:
        a, b = c.normal
        r = mpmath.sqrt(a * a + b * b)
        key = (a / r, b / r, c.s / r)
        if not any(max(abs(key[0] - k[0]), abs(key[1] - k[1]), abs(key[2] - k[2])) < 1e-25
                   or max(abs(key[0] + k[0]), abs(key[1] + k[1]), abs(key[2] + k[2])) < 1e-25 for k, _ in out):
            out.append((key, c))
    return [c for _, c in out]


# ---------------------------------------------------------------- optimal destabilizer


@dataclass
class OptimalCrease:
    """The optimal simple destabilizer f* = c (φ − Πφ), φ = max(0, n·x − s), of `optimal_crease`.

    `norm` = ‖f*‖_{L²(μ)} = sup over simple creases of −L_A(f)/‖f‖, Donaldson's lower bound
    for ‖S − A_ext‖_{L²(μ)} over toric metrics. `pieces` holds the extremal affine functions
    (c0, c1, c2) of the two pieces, with zero measure on the crease; theory predicts
    A − f* = A_i on piece i, so `continuity_residual` (the jump of A_1 − A_2 across the crease,
    at its two ends) and `piece_residual` (max coefficient error of A − f* versus A_i) vanish.
    """

    polygon: Polygon
    crease: CriticalCrease | None
    candidates: list
    norm: mpmath.mpf = mpmath.mpf(0)
    c: mpmath.mpf = mpmath.mpf(0)
    affine: tuple = (0, 0, 0)  # f* = c·max(0, ℓ) + affine
    pieces: tuple = ()
    continuity_residual: mpmath.mpf = mpmath.mpf(0)
    piece_residual: mpmath.mpf = mpmath.mpf(0)
    identity_residual: mpmath.mpf = mpmath.mpf(0)  # L_A(f*) + ‖f*‖²

    @property
    def stable(self) -> bool:
        return self.crease is None

    def __call__(self, x, y):
        """f*(x, y) in floats (numpy arrays allowed)."""
        import numpy as np
        if self.crease is None:
            return 0 * np.asarray(x, float)
        a, b = (float(v) for v in self.crease.normal)
        ell = a * np.asarray(x, float) + b * np.asarray(y, float) - float(self.crease.s)
        a0, a1, a2 = (float(v) for v in self.affine)
        return float(self.c) * np.maximum(0, ell) + a0 + a1 * np.asarray(x, float) + a2 * np.asarray(y, float)

    def __str__(self) -> str:
        if self.crease is None:
            return "no destabilizing simple crease: f* = 0 (semistable)"
        cr = self.crease
        a, b = cr.normal
        return "\n".join([
            f"optimal simple crease: {cr}",
            f"  ‖f*‖ = {mpmath.nstr(self.norm, 20)}   (lower bound for ‖S − A_ext‖_L²(μ))",
            f"  f* = {mpmath.nstr(self.c, 12)}·max(0, {mpmath.nstr(a, 12)}x + {mpmath.nstr(b, 12)}y − {mpmath.nstr(cr.s, 12)})"
            f" + ({mpmath.nstr(self.affine[0], 8)}) + ({mpmath.nstr(self.affine[1], 8)})x + ({mpmath.nstr(self.affine[2], 8)})y",
            f"  minimal polynomial of the slope m: degree {cr.minimal_polynomial_m().degree()}",
            f"  checks: L(f*) + ‖f*‖² = {mpmath.nstr(self.identity_residual, 3)},  "
            f"A − f* vs piece extremal functions: {mpmath.nstr(self.piece_residual, 3)},  "
            f"A_1 − A_2 on the crease: {mpmath.nstr(self.continuity_residual, 3)}",
        ])


def optimal_crease(P: Polygon) -> OptimalCrease:
    """The simple crease maximizing −L_{A_ext}(φ)/‖φ⊥‖, found exactly (see the module docstring)."""
    A = P.A_ext
    found = []
    with mpmath.workdps(DPS):
        for cell in _cells(P, A):
            if cell.chart == 1:  # chart 1 only adds the direction m = 0, i.e. n = (0, 1)
                sols = _restricted_zeros(cell.ex["opt"])
            else:
                eqs = [_numerator(e) for e in cell.ex["opt"]]
                if any(e is None for e in eqs):
                    continue
                sols = _common_zeros(*eqs)
            for root, m0, s0, s_alg in sols:
                if not _in_cell(cell, m0, s0):
                    continue
                norm2 = cell.num("norm2")(m0, s0)
                if norm2 <= mpmath.mpf(10) ** -40:
                    continue
                L = _side_sign(cell, m0, s0) * cell.num("L_phi")(m0, s0)
                found.append(CriticalCrease((cell.i, cell.j), cell.chart, root, m0, s0, L, -L / mpmath.sqrt(norm2),
                                            s_exact=s_alg))
        found = sorted(_dedupe(found), key=lambda c: -c.value)
        best = found[0] if found and found[0].L < 0 else None
        if best is None:
            return OptimalCrease(P, None, found)
        return _finish_optimal(P, A, best, found)


def _restricted_zeros(fracs) -> list:
    """Common zeros with m = 0 (chart 1: the single direction n = (0, 1)) of two elements of Q(m, s)."""
    polys = []
    for e in fracs:
        f = sp.Poly(e.numer.as_expr(), m_, s_)
        if f.is_zero:
            return []
        k = min(mon[0] for mon in f.monoms())  # strip the power of m (a factor free of s)
        polys.append(sp.Poly(sum(c * s_ ** b for (a, b), c in f.terms() if a == k), s_))
    f1, f2 = polys
    g = sp.gcd(f1, f2)
    if g.degree() <= 0:
        return []
    zero = AlgebraicReal(sp.Poly(m_, m_), Fraction(0), Fraction(0), mpmath.mpf(0))
    return [(zero, mpmath.mpf(0), r.value, r) for r in _real_roots(g)]


def _finish_optimal(P: Polygon, A: Affine, best: CriticalCrease, found: list) -> OptimalCrease:
    cell = _Cell(P, A, *best.cell, best.chart)
    m0, s0 = best.m, best.s
    sgn = _side_sign(cell, m0, s0)
    norm2 = cell.num("norm2")(m0, s0)
    L = best.L
    c = -L / norm2
    proj = [cell.num("piece", "proj", r)(m0, s0) for r in range(3)]
    # φ = max(0, ℓ): equals sgn·ℓ·1_X on X if sgn = +1; otherwise φ = ℓ − ℓ 1_X.
    a, b = best.normal
    if sgn > 0:
        aff = [-c * proj[r] for r in range(3)]
    else:  # φ = ℓ + (−ℓ1_X)... Π(φ) = ℓ − Π(ℓ1_X)
        lc = [-s0, a, b]
        aff = [-c * (lc[r] - proj[r]) for r in range(3)]
    # piece extremal functions (zero measure on the crease)
    GX = [[cell.num("piece", "GX", r, t)(m0, s0) for t in range(3)] for r in range(3)]
    bX = [cell.num("piece", "bX", r)(m0, s0) for r in range(3)]
    Vf = [(mpmath.mpf(sp.N(_q(v[0]), DPS)), mpmath.mpf(sp.N(_q(v[1]), DPS))) for v in P.vertices]
    GP = [[mpmath.mpf(sp.N(v, DPS)) for v in row] for row in _gram(_moments([(_q(x), _q(y)) for x, y in P.vertices]))]
    bP = [mpmath.mpf(sp.N(_q(v), DPS)) for v in (P.boundary_measure, P.boundary_barycenter[0] * P.boundary_measure,
                                                   P.boundary_barycenter[1] * P.boundary_measure)]
    AX = mpmath.lu_solve(mpmath.matrix(GX), mpmath.matrix(bX))
    GY = mpmath.matrix([[GP[r][t] - GX[r][t] for t in range(3)] for r in range(3)])
    AY = mpmath.lu_solve(GY, mpmath.matrix([bP[r] - bX[r] for r in range(3)]))
    Ac = [mpmath.mpf(sp.N(_q(v), DPS)) for v in (A.c0, A.c1, A.c2)]
    lc = [-s0, a, b]
    # f* on X: c·max(0,ℓ)|_X + aff; max(0, ℓ)|_X = ℓ if sgn > 0 else 0
    fX = [aff[r] + (c * lc[r] if sgn > 0 else 0) for r in range(3)]
    fY = [aff[r] + (0 if sgn > 0 else c * lc[r]) for r in range(3)]
    piece_res = max(max(abs(Ac[r] - fX[r] - AX[r]) for r in range(3)), max(abs(Ac[r] - fY[r] - AY[r]) for r in range(3)))
    p_q = []
    for t_name, e in (("ti", best.cell[0]), ("tj", best.cell[1])):
        t = cell.num(t_name)(m0, s0)
        P0, P1 = Vf[e], Vf[(e + 1) % P.n]
        p_q.append((P0[0] + t * (P1[0] - P0[0]), P0[1] + t * (P1[1] - P0[1])))
    cont = max(abs((AX[0] - AY[0]) + (AX[1] - AY[1]) * x + (AX[2] - AY[2]) * y) for x, y in p_q)
    norm = -L / mpmath.sqrt(norm2)
    ident = c * L + c * c * norm2  # L(f*) + ‖f*‖² with L(f*) = c L(φ), ‖f*‖² = c² ‖φ⊥‖²
    return OptimalCrease(P, best, found, norm=norm, c=c, affine=tuple(aff),
                         pieces=(tuple(AX), tuple(AY)), continuity_residual=cont, piece_residual=piece_res,
                         identity_residual=ident)


# ---------------------------------------------------------------- boundary-normalized minimum


@dataclass
class CreaseMinimum:
    """Exact minimum of L_A(f)/∫_∂P f dσ over simple creases (f vanishing at the area barycenter).

    `value` is the minimum, attained at `crease` (L there is L_A(max(0, n·x − s))); if no critical
    candidate beats the limit 1 at degenerate creases, `crease` is None and `value` = 1.
    """

    value: mpmath.mpf
    crease: CriticalCrease | None
    candidates: list


def crease_minimum(P: Polygon, A: Affine | None = None) -> CreaseMinimum:
    A = A or P.A_ext
    c = P.area_barycenter
    cx, cy = _q(c[0]), _q(c[1])
    found = []
    with mpmath.workdps(DPS):
        for cell in _cells(P, A):
            ex = cell.ex
            n = ex["n"]
            ratios = {+1: ex["L_phi"] / ex["B_X"], -1: (ex["L_full"] - ex["L_phi"]) / (ex["B_full"] - ex["B_X"])}
            num_ratio = {k: _lambdify(v) for k, v in ratios.items()}
            ell_c = n[0] * cx + n[1] * cy - s_

            def admit(m0, s0, side, kind, root, s_alg=None):
                """side = +1: f supported on X; valid if the barycenter is not inside X."""
                if not _in_cell(cell, m0, s0):
                    return
                try:
                    val = num_ratio[side](m0, s0)
                except ZeroDivisionError:
                    return
                sx = _side_sign(cell, m0, s0)
                lc = sp.lambdify((m_, s_), ell_c, "mpmath")(m0, s0)
                on_X = lc * sx > mpmath.mpf(10) ** -30  # barycenter strictly inside X
                in_Y = lc * sx < -mpmath.mpf(10) ** -30
                if kind == "critical" and ((side == 1 and on_X) or (side == -1 and in_Y)):
                    return
                L = sx * cell.num("L_phi")(m0, s0)
                found.append(CriticalCrease((cell.i, cell.j), cell.chart, root, m0, s0, L, val, kind, s_alg))

            for side, R in ratios.items():
                derivs = [R.diff(v) for v in (sK, mK)]
                if cell.chart == 1:
                    sols = _restricted_zeros(derivs)
                else:
                    eqs = [_numerator(e) for e in derivs]
                    sols = [] if any(e is None for e in eqs) else _common_zeros(*eqs)
                for root, m0, s0, s_alg in sols:
                    admit(m0, s0, side, "critical", root, s_alg)
                # the lines through the barycenter, s = n·c: both one-sided creases are normalized there
                if cell.chart == 0:
                    Rm = (R.numer.as_expr() / R.denom.as_expr()).subs(s_, n[0] * cx + n[1] * cy)
                    for root, m0 in _univariate_zeros(sp.diff(Rm, m_)):
                        admit(m0, m0 * mpmath.mpf(sp.N(cy, DPS)) + mpmath.mpf(sp.N(cx, DPS)), side, "barycentric", root)
                else:
                    admit(mpmath.mpf(0), mpmath.mpf(sp.N(cy, DPS)), side, "barycentric",
                          AlgebraicReal(sp.Poly(m_, m_), Fraction(0), Fraction(0), mpmath.mpf(0)))
        found = sorted(_dedupe(found), key=lambda c: c.value)
        if found and found[0].value < 1:
            return CreaseMinimum(found[0].value, found[0], found)
        return CreaseMinimum(mpmath.mpf(1), None, found)


# ---------------------------------------------------------------- is the simple crease optimal among all convex f?


def _clip(V: list, a: float, b: float, c: float) -> list:
    """{a x + b y + c >= 0} ∩ V for a CCW convex polygon V of float points (Sutherland–Hodgman)."""
    out = []
    for k in range(len(V)):
        P0, P1 = V[k], V[(k + 1) % len(V)]
        g0, g1 = a * P0[0] + b * P0[1] + c, a * P1[0] + b * P1[1] + c
        if g0 >= 0:
            out.append(P0)
        if (g0 >= 0) != (g1 >= 0):
            t = g0 / (g0 - g1)
            out.append((P0[0] + t * (P1[0] - P0[0]), P0[1] + t * (P1[1] - P0[1])))
    return out


def _pair_integral(V: list, u, v) -> float:
    """∫_V (u·(1, x, y)) (v·(1, x, y)) dμ over a float polygon."""
    if len(V) < 3:
        return 0.0
    G = _gram(_moments(V))
    return sum(u[r] * G[r][t] * v[t] for r in range(3) for t in range(3))


def optimality_scan(opt: OptimalCrease, n_angles: int = 180, samples: int = 120) -> dict:
    """Scan simple creases h for the first-order optimality condition of f* among all convex f.

    W is convex, so f* minimizes W over all convex functions iff L_{A−f*}(h) = L_A(h) + ∫ f* h dμ
    ≥ 0 for every convex h. This samples simple creases h, normalized like `crease_search`
    (one-sided, vanishing at the barycenter, divided by ∫_∂P h dσ): `margin[i]` is the smallest
    value in direction `theta[i]`. At f*'s own crease the value is 0.

    Caveat: Wang–Zhou's reduction to simple creases is proved for affine A; A − f* is only
    piecewise affine, so a nonnegative scan is a necessary condition, not a proof.
    """
    import numpy as np

    from .stability import CreaseProfile

    P = opt.polygon
    A = P.A_ext
    V = [(float(x), float(y)) for x, y in P.vertices]
    if opt.crease is None:
        raise ValueError("f* = 0: nothing to check (use crease_search for the stability margin)")
    a, b = (float(v) for v in opt.crease.normal)
    s0 = float(opt.crease.s)
    c = float(opt.c)
    aff = [float(v) for v in opt.affine]
    lc = [-s0, a, b]
    upper = _clip(V, a, b, -s0)  # {ℓ ≥ 0}, where max(0, ℓ) = ℓ
    theta = np.linspace(0, np.pi, n_angles, endpoint=False)
    margin = np.empty(n_angles)
    for k, th in enumerate(theta):
        n = (math.cos(th), math.sin(th))
        prof = CreaseProfile(P, n, A)
        lo, hi = (float(v) for v in prof.t_range)
        q = (1 - np.cos(np.linspace(0, math.pi, samples + 2)[1:-1])) / 2
        ts = lo + (hi - lo) * q
        L = prof.stability_ratio(ts)[0]
        B = prof.boundary_normalizer(ts)
        vals = []
        for t, Lt, Bt in zip(ts, L, B):
            h = [-t, n[0], n[1]]
            fh = c * _pair_integral(_clip(upper, n[0], n[1], -t), lc, h) + _pair_integral(_clip(V, n[0], n[1], -t), aff, h)
            vals.append((Lt + fh) / Bt)
        margin[k] = min(vals)
    i = int(np.argmin(margin))
    return dict(theta=theta, margin=margin, min=float(margin[i]), at=float(theta[i]))
