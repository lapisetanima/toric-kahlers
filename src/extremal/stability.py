"""Destabilizing simple crease functions, found by an exact one-variable reduction.

Fix a direction n and write the crease function as f_s = max(0, n·x − s), for s in
[t_min, t_max] = [min_P n·x, max_P n·x]. Since d²f_s/ds² = δ(n·x − s),

    L_{A}(f_s) = ∫_s^{t_max} (t − s) dρ(t),

where ρ is the measure on [t_min, t_max] with density

    ρ(t) = Σ_{edges crossed by n·x = t} L_i / |Δt_i|   −   w(t) · (A(p(t)) + A(q(t))) / 2,

plus a point mass L_i at t_max for an edge parallel to the crease. Here L_i is the lattice
length of the edge, Δt_i the change of n·x along it, p(t), q(t) the ends of the chord
{n·x = t} ∩ P, and w(t) = |q − p| / |n|. Between consecutive vertex values ρ is a quadratic
polynomial in t, so L is piecewise quartic in s and can be minimized exactly piece by piece.

The function is homogeneous: scaling (n, s) by λ > 0 scales L by λ. And n, −n give the same
values (the two creases differ by an affine function). So the whole family of simple creases
is the half-circle of unit directions θ ∈ [0, π) times one offset s.

Theory (dimension 2): by Wang–Zhou (Corollary 3.5 / Theorem 3.4 of B. Zhou's thesis; Wang–Zhou,
Adv. Math. 2011), if L_A(u) < 0 for some convex u, then L_A(f) < 0 for some simple crease f;
no sign condition on A is needed. Donaldson (2002, Prop. 6.1) shows the crease can then be
taken rational. So checking simple rational creases decides whether L_{A_ext} ≥ 0 on all
convex functions.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .exact import as_fraction
from .polygon import Affine, Polygon


def _poly_eval(coeffs, t):
    out = 0 * t
    for c in reversed(coeffs):
        out = out * t + c
    return out


@dataclass
class CreaseProfile:
    """The exact one-variable function s -> L_A(max(0, n·x − s)) for a fixed direction n.

    Built from Fractions when n is rational (then `L` is exact for rational s), or from
    floats when n is a float vector. `A` defaults to the polygon's A_ext.
    """

    polygon: Polygon
    n: tuple
    A: Affine | None = None
    breaks: list = field(init=False)  # t_min = breaks[0] < ... < breaks[-1] = t_max
    rho: list = field(init=False)  # rho[k] = [r0, r1, r2]: ρ = r0 + r1 u + r2 u², u = t − breaks[k]
    top_mass: object = field(init=False)
    boundary_density: list = field(init=False)
    _m0: list = field(init=False)  # ∫ ρ over piece k
    _m1: list = field(init=False)  # ∫ u ρ over piece k

    def __post_init__(self):
        P = self.polygon
        exact = all(isinstance(c, (int, Fraction)) for c in self.n)
        num = (lambda v: as_fraction(v)) if exact else float
        n = tuple(num(c) for c in self.n)
        if n == (0, 0):
            raise ValueError("direction must be nonzero")
        self.n = n
        A = self.A or P.A_ext
        c0, c1, c2 = num(A.c0), num(A.c1), num(A.c2)
        V = [(num(v[0]), num(v[1])) for v in P.vertices]
        lengths = [num(e.sigma_length) for e in P.edges]  # σ-mass of each edge
        tv = [n[0] * v[0] + n[1] * v[1] for v in V]
        if not exact:  # snap values equal up to round-off, so no spurious sliver intervals appear
            tol = 1e-12 * max(1.0, max(abs(t) for t in tv))
            ordered = sorted(tv)
            for i in range(1, len(ordered)):
                if ordered[i] - ordered[i - 1] < tol:
                    ordered[i] = ordered[i - 1]
            snap = dict(zip(sorted(tv), ordered))
            tv = [snap[t] for t in tv]
        self.breaks = sorted(set(tv))
        nn = n[0] * n[0] + n[1] * n[1]
        m = len(V)

        # point mass from an edge parallel to the crease at the top level t_max
        t_max = self.breaks[-1]
        self.top_mass = sum((lengths[i] for i in range(m) if tv[i] == t_max and tv[(i + 1) % m] == t_max), 0 * nn)

        self.rho = []
        self.boundary_density = []  # the ∂P part of ρ, constant on each piece
        for k in range(len(self.breaks) - 1):
            lo, hi = self.breaks[k], self.breaks[k + 1]
            mid = (lo + hi) / 2
            crossings = []  # (point(t) = a + b t as vectors, boundary density)
            for i in range(m):
                t0, t1 = tv[i], tv[(i + 1) % m]
                if min(t0, t1) < mid < max(t0, t1):
                    P0, P1 = V[i], V[(i + 1) % m]
                    b = ((P1[0] - P0[0]) / (t1 - t0), (P1[1] - P0[1]) / (t1 - t0))
                    a = (P0[0] - b[0] * t0, P0[1] - b[1] * t0)
                    crossings.append((a, b, lengths[i] / abs(t1 - t0)))
            if len(crossings) != 2:
                raise RuntimeError("a line should cross a convex polygon's boundary exactly twice")
            # chord ends at u = t − lo: p = ap + bp u, q = aq + bq u
            (ap, bp, dp), (aq, bq, dq) = [((a[0] + b[0] * lo, a[1] + b[1] * lo), b, d) for a, b, d in crossings]
            D0 = (aq[0] - ap[0], aq[1] - ap[1])
            D1 = (bq[0] - bp[0], bq[1] - bp[1])
            w0 = n[0] * D0[1] - n[1] * D0[0]
            w1 = n[0] * D1[1] - n[1] * D1[0]
            sign = 1 if w0 + w1 * (mid - lo) > 0 else -1
            w0, w1 = sign * w0 / nn, sign * w1 / nn  # w = |q − p| / |n| = w0 + w1 u
            # A(p) + A(q) = s0 + s1 u
            s0 = 2 * c0 + c1 * (ap[0] + aq[0]) + c2 * (ap[1] + aq[1])
            s1 = c1 * (bp[0] + bq[0]) + c2 * (bp[1] + bq[1])
            self.rho.append([dp + dq - w0 * s0 / 2, -(w0 * s1 + w1 * s0) / 2, -w1 * s1 / 2])
            self.boundary_density.append(dp + dq)

        self._m0, self._m1 = [], []
        for k, r in enumerate(self.rho):
            h = self.breaks[k + 1] - self.breaks[k]
            self._m0.append(sum(r[j] * h ** (j + 1) / (j + 1) for j in range(3)))
            self._m1.append(sum(r[j] * h ** (j + 2) / (j + 2) for j in range(3)))

    @property
    def t_range(self):
        return self.breaks[0], self.breaks[-1]

    def _piece(self, s) -> int:
        k = bisect.bisect_right(self.breaks, s) - 1
        return min(max(k, 0), len(self.rho) - 1)

    def L(self, s):
        """L_A(max(0, n·x − s)). Exact if n and s are rational; s outside [t_min, t_max] gives 0."""
        lo, hi = self.t_range
        if s <= lo or s >= hi:
            return 0 * s
        k = self._piece(s)
        h = self.breaks[k + 1] - self.breaks[k]
        sig = s - self.breaks[k]
        r = self.rho[k]
        # ∫_σ^h (u − σ) ρ(u) du on the piece containing s
        out = sum(
            r[j] * ((h ** (j + 2) - sig ** (j + 2)) / (j + 2) - sig * (h ** (j + 1) - sig ** (j + 1)) / (j + 1))
            for j in range(3)
        )
        for j in range(k + 1, len(self.rho)):  # ∫ (t − s) ρ over the later pieces
            out += self._m1[j] + (self.breaks[j] - s) * self._m0[j]
        return out + self.top_mass * (hi - s)

    def L_values(self, s) -> np.ndarray:
        """Float values of L at an array of offsets s."""
        exact = isinstance(self.breaks[0], Fraction)
        s = np.asarray(s, float)
        return np.array([float(self.L(Fraction(v) if exact else v)) for v in s.ravel()]).reshape(s.shape)

    def _upper_integrals(self, s: np.ndarray, which: str) -> np.ndarray:
        """Vectorized ∫_{t>s} (t − s) dν for ν = ρ ('L') or its boundary part ('B'), in floats."""
        s = np.asarray(s, float)
        out = float(self.top_mass) * np.clip(float(self.breaks[-1]) - s, 0, None)
        for k in range(len(self.rho)):
            lo = float(self.breaks[k])
            h = float(self.breaks[k + 1]) - lo
            r = [float(c) for c in self.rho[k]] if which == "L" else [float(self.boundary_density[k]), 0.0, 0.0]
            d = s - lo
            sig = np.clip(d, 0, h)
            for j in range(3):
                if r[j]:
                    out = out + r[j] * ((h ** (j + 2) - sig ** (j + 2)) / (j + 2) - d * (h ** (j + 1) - sig ** (j + 1)) / (j + 1))
        return out

    @property
    def barycenter_level(self) -> float:
        """n·c for the area barycenter c: where the normalized crease switches sides."""
        c = self.polygon.area_barycenter
        return float(self.n[0]) * float(c[0]) + float(self.n[1]) * float(c[1])

    def stability_ratio(self, s, side: str = "auto") -> tuple[np.ndarray, np.ndarray]:
        """(L(f), L(f) / ∫_∂P f dσ) at offsets s, with f the crease normalized to vanish at the barycenter.

        f is max(0, n·x − s) ('upper') if the barycenter c has n·c <= s, else max(0, s − n·x)
        ('lower'); the two differ by an affine function, so L(f) is the same. At s = n·c both are
        normalized and the ratio jumps. The ratio is scale-invariant; Donaldson's uniform stability
        asks for it to be bounded below by a positive constant.
        """
        s = np.asarray(s, float)
        L = self._upper_integrals(s, "L")
        with np.errstate(divide="ignore", invalid="ignore"):
            return L, L / self.boundary_normalizer(s, side)

    def boundary_normalizer(self, s, side: str = "auto") -> np.ndarray:
        """∫_∂P f dσ for the crease f of `stability_ratio` (normalized to vanish at the barycenter)."""
        P = self.polygon
        n = [float(c) for c in self.n]
        s = np.asarray(s, float)
        B_up = self._upper_integrals(s, "B")
        bx, by = (float(P.boundary_barycenter[i] * P.boundary_measure) for i in range(2))
        first_moment = n[0] * bx + n[1] * by - s * float(P.boundary_measure)  # ∫_∂P (n·x − s) dσ
        upper = {"auto": s >= self.barycenter_level, "upper": True, "lower": False}[side]
        return np.where(upper, B_up, B_up - first_moment)

    def density(self, t: np.ndarray) -> np.ndarray:
        """ρ(t) (float), without the point mass at t_max."""
        t = np.asarray(t, float)
        out = np.full(t.shape, np.nan)
        for k, r in enumerate(self.rho):
            lo, hi = float(self.breaks[k]), float(self.breaks[k + 1])
            sel = (t >= lo) & (t <= hi)
            out[sel] = _poly_eval([float(c) for c in r], t[sel] - lo)
        return out

    def minimize(self) -> tuple[float, float]:
        """(s*, L(s*)) minimizing L over [t_min, t_max], located in floats from the exact pieces.

        On piece k, with σ = s − breaks[k] and R(σ) = ∫_0^σ ρ, L'(s) = R(σ) − R(h) − (mass of ρ
        beyond the piece), a cubic in σ; its real roots and the breakpoints are the candidates.
        """
        exact = isinstance(self.breaks[0], Fraction)
        tail = float(self.top_mass)
        tails = [0.0] * len(self.rho)
        for k in range(len(self.rho) - 1, -1, -1):
            tails[k] = tail
            tail += float(self._m0[k])
        best_s, best_L = float(self.breaks[0]), 0.0
        for k, r in enumerate(self.rho):
            lo = self.breaks[k]
            h = float(self.breaks[k + 1] - lo)
            rf = [float(c) for c in r]
            R = [0.0, rf[0], rf[1] / 2, rf[2] / 3]  # coefficients of σ^0..σ^3
            deriv = [R[0] - _poly_eval(R, h) - tails[k], R[1], R[2], R[3]]
            coeffs = list(reversed(deriv))
            while len(coeffs) > 1 and coeffs[0] == 0:
                coeffs.pop(0)
            roots = np.roots(coeffs) if len(coeffs) > 1 else []
            sigmas = [h] + [z.real for z in roots if abs(z.imag) < 1e-9 * max(1.0, h) and 0 < z.real < h]
            for sig in sigmas:
                s = lo + (Fraction(sig) if exact else sig)
                value = float(self.L(s))
                if value < best_L:
                    best_s, best_L = float(s), value
        return best_s, best_L


@dataclass
class CreaseSearchResult:
    """Outcome of `crease_search`.

    `margin[i]` is the smallest normalized value L(f) / ∫_∂P f dσ over creases with unit normal at
    angle `theta[i]` (f normalized to vanish at the barycenter). It is scale-invariant: positive
    means every crease in that direction is stabilizing, negative means some crease destabilizes.
    """

    theta: np.ndarray
    margin: np.ndarray
    best_theta: float
    best_s: float
    best_margin: float
    certificate: tuple[int, int, Fraction] | None  # (a, b, c): f = max(0, a x + b y + c)
    certificate_value: Fraction | None  # exact L_{A}(f) < 0

    @property
    def unstable(self) -> bool:
        return self.certificate_value is not None and self.certificate_value < 0

    def __str__(self) -> str:
        lines = [f"crease stability margin min L(f)/∫_∂P f dσ = {self.best_margin:.6g} "
                 f"at θ = {math.degrees(self.best_theta):.3f}°, s = {self.best_s:.6g}"]
        if self.unstable:
            a, b, c = self.certificate
            lines.append(f"UNSTABLE: exact certificate f = max(0, {a}x + {b}y + {c}), "
                         f"L = {float(self.certificate_value):.6g} (exact rational: {self.certificate_value})")
        elif self.best_margin < 0:
            lines.append("negative margin found but no exact rational certificate yet: try certify() with more tries")
        else:
            lines.append("no destabilizing crease found (sampled search, not a proof)")
        return "\n".join(lines)


def _direction_margin(P: Polygon, A: Affine, theta: float, samples: int) -> tuple[float, float]:
    """(min over s of the normalized ratio, minimizing s) for the unit direction at angle theta."""
    prof = CreaseProfile(P, (math.cos(theta), math.sin(theta)), A)
    lo, hi = (float(v) for v in prof.t_range)
    q = (1 - np.cos(np.linspace(0, math.pi, samples + 2)[1:-1])) / 2  # denser near the ends
    s = lo + (hi - lo) * q
    _, ratio = prof.stability_ratio(s)
    i = int(np.nanargmin(ratio))
    best = (float(ratio[i]), float(s[i]))
    # the ratio jumps at s = n·c, where both one-sided creases are normalized: evaluate both exactly
    sc = np.array([prof.barycenter_level])
    for side in ("upper", "lower"):
        value = float(prof.stability_ratio(sc, side=side)[1][0])
        if value < best[0]:
            best = (value, float(sc[0]))
    return best


def _integer_directions_near(theta: float, max_coeff: int):
    """Primitive integer vectors (a, b), |a|, |b| <= max_coeff, sorted by angular distance to theta."""
    target = theta % math.pi
    cands = []
    for a in range(-max_coeff, max_coeff + 1):
        for b in range(0, max_coeff + 1):
            if (a, b) == (0, 0) or math.gcd(a, b) != 1 or (b == 0 and a < 0):
                continue
            ang = math.atan2(b, a) % math.pi
            cands.append((min(abs(ang - target), math.pi - abs(ang - target)), a, b))
    cands.sort()
    return [(a, b) for _, a, b in cands]


def certify(P: Polygon, theta: float, A: Affine | None = None, max_coeff: int = 60, tries: int = 40,
            max_den: int = 10_000) -> tuple[tuple[int, int, Fraction], Fraction] | None:
    """Look for a rational crease near angle θ whose L_A is exactly negative.

    For each nearby integer direction, the exact profile locates the best offset; that offset is
    rounded to a rational and the value is recomputed independently with Polygon.L_crease.
    """
    A = A or P.A_ext
    for a, b in _integer_directions_near(theta, max_coeff)[:tries]:
        s, value = CreaseProfile(P, (a, b), A).minimize()
        if value >= 0:
            continue
        s_rat = Fraction(s).limit_denominator(max_den)
        exact = P.L_crease(a, b, -s_rat, A=A)
        if exact < 0:
            return (a, b, -s_rat), exact
    return None


def crease_search(P: Polygon, n_angles: int = 360, samples: int = 200, refine: int = 3,
                  A: Affine | None = None, tol: float = 1e-9, exact: bool = True) -> CreaseSearchResult:
    """Minimize L(f)/∫_∂P f dσ over all simple creases, and certify instability exactly.

    With `exact=True` (default) the minimum is located exactly by `crease_exact.crease_minimum`
    (critical points of the ratio on each cell of the line arrangement, as algebraic numbers);
    with `exact=False` by a sampled scan over directions θ ∈ [0, π) and offsets s with local
    refinement. Either way `margin` holds the sampled per-direction minima (for plots), and if
    the minimum is below −tol an exact rational certificate is searched for nearby: instability
    is reported only after a rational crease has been checked in exact arithmetic.
    """
    A = A or P.A_ext
    theta = np.linspace(0, math.pi, n_angles, endpoint=False)
    margin = np.empty(n_angles)
    for i, t in enumerate(theta):
        margin[i], _ = _direction_margin(P, A, t, samples)
    found = None
    if exact:
        from .crease_exact import crease_minimum

        found = crease_minimum(P, A).crease
    if found is not None:
        best_t, best_s, best_margin = found.theta, found.unit_offset, float(found.value)
        if margin.min() < best_margin - 1e-9 * max(1.0, abs(best_margin)):
            import warnings
            warnings.warn(f"sampled margin {margin.min():.12g} is below the exact minimum {best_margin:.12g}")
    else:
        best_t = float(theta[int(np.argmin(margin))])
        width = math.pi / n_angles
        for _ in range(refine):
            local = np.linspace(best_t - width, best_t + width, 21)
            vals = [_direction_margin(P, A, t, samples)[0] for t in local]
            best_t, width = float(local[int(np.argmin(vals))]), width / 10
        best_margin, best_s = _direction_margin(P, A, best_t, samples)
    cert = certify(P, best_t, A) if best_margin < -tol else None
    return CreaseSearchResult(
        theta=theta, margin=margin, best_theta=best_t, best_s=best_s, best_margin=best_margin,
        certificate=cert[0] if cert else None, certificate_value=cert[1] if cert else None,
    )
