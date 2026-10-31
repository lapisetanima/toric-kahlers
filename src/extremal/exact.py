"""Exact rational arithmetic helpers: conversion, primitive vectors, polynomials, linear solves."""

from __future__ import annotations

import numbers
from fractions import Fraction
from math import gcd, lcm

import sympy as sp

Point = tuple[Fraction, Fraction]
# A bivariate polynomial: {(a, b): coeff} meaning sum coeff * x**a * y**b.
Poly = dict[tuple[int, int], Fraction]

X, Y = sp.symbols("x y", real=True)


def as_fraction(value) -> Fraction:
    """Convert an exact number (int, Fraction, 'p/q' string, sympy Rational) to a Fraction.

    Floats are rejected on purpose: everything in the exact core must stay in Q.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("booleans are not numbers here")
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, sp.Rational):
        return Fraction(int(value.p), int(value.q))
    if isinstance(value, float):
        raise TypeError(f"float {value!r} not allowed; use an int, Fraction, or a string like '1/3'")
    raise TypeError(f"cannot convert {value!r} to an exact rational")


def as_point(p) -> Point:
    if len(p) != 2:
        raise ValueError(f"expected a 2D point, got {p!r}")
    return (as_fraction(p[0]), as_fraction(p[1]))


def primitive(v: Point) -> tuple[tuple[int, int], Fraction]:
    """Write a nonzero rational vector as v = length * e with e integral and primitive."""
    den = lcm(v[0].denominator, v[1].denominator)
    a, b = int(v[0] * den), int(v[1] * den)
    g = gcd(a, b)
    if g == 0:
        raise ValueError("zero vector has no primitive direction")
    return (a // g, b // g), Fraction(g, den)


def det2(a, b) -> Fraction:
    return a[0] * b[1] - a[1] * b[0]


# --- univariate polynomials in t, as coefficient lists -------------------------------------


def _tpoly_mul(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return out


def _tpoly_pow(p: list[Fraction], n: int) -> list[Fraction]:
    out = [Fraction(1)]
    for _ in range(n):
        out = _tpoly_mul(out, p)
    return out


def segment_integral(p: Point, q: Point, poly: Poly) -> Fraction:
    """∫_0^1 poly(p + t (q - p)) dt, exactly."""
    xt = [p[0], q[0] - p[0]]
    yt = [p[1], q[1] - p[1]]
    total = Fraction(0)
    for (a, b), c in poly.items():
        if not c:
            continue
        coeffs = _tpoly_mul(_tpoly_pow(xt, a), _tpoly_pow(yt, b))
        total += c * sum(ck / (k + 1) for k, ck in enumerate(coeffs))
    return total


# --- bivariate polynomials -----------------------------------------------------------------


def poly_from(f) -> Poly:
    """Accept a Poly dict, a number, or a sympy polynomial expression in X, Y with rational coefficients."""
    if isinstance(f, dict):
        return {k: as_fraction(v) for k, v in f.items()}
    if isinstance(f, (numbers.Rational, str)) and not isinstance(f, bool):
        return {(0, 0): as_fraction(f)}
    expr = sp.sympify(f)
    extra = expr.free_symbols - {X, Y}
    if extra:
        raise ValueError(f"polynomial may only use x, y; found {extra}")
    poly = sp.Poly(expr, X, Y)
    out: Poly = {}
    for (a, b), c in poly.terms():
        if not c.is_rational:
            raise ValueError(f"coefficient {c} is not rational")
        out[(a, b)] = as_fraction(c)
    return out


def poly_mul(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for (a1, b1), c1 in p.items():
        for (a2, b2), c2 in q.items():
            k = (a1 + a2, b1 + b2)
            out[k] = out.get(k, Fraction(0)) + c1 * c2
    return out


def poly_to_sympy(p: Poly) -> sp.Expr:
    return sp.Add(*[sp.Rational(c.numerator, c.denominator) * X**a * Y**b for (a, b), c in p.items()])


# --- exact linear algebra --------------------------------------------------------------------


def solve_exact(A: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    """Solve A z = b over Q by Gaussian elimination. Raises if A is singular."""
    n = len(A)
    M = [list(map(Fraction, row)) + [Fraction(b[i])] for i, row in enumerate(A)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if M[r][col] != 0), None)
        if pivot is None:
            raise ZeroDivisionError("singular system")
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0:
                factor = M[r][col] / M[col][col]
                M[r] = [a - factor * c for a, c in zip(M[r], M[col])]
    return [M[i][n] / M[i][i] for i in range(n)]
