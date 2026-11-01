"""Labelled polygons (orbifolds): u_G = ½ Σ l log l with l = m (<u, x> - λ), and σ(edge) = lattice length / m."""

from fractions import Fraction as Fr

import numpy as np

from extremal import Polygon
from extremal.quadrature import polygon_quadrature
from extremal.solver import ExtremalSolver
from extremal.stability import crease_search

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]  # edges: 0 bottom, 1 right, 2 top, 3 left


def football_S(m, n, t):
    """Extremal scalar curvature on [0, 1] with labels m at 0 and n at 1: Θ = t(1-t)(2/m + (2/n - 2/m) t), S = -½Θ''."""
    return Fr(4, m) - Fr(2, n) + 6 * (Fr(1, n) - Fr(1, m)) * t


def test_labels_follow_edges_when_orientation_is_reversed():
    P = Polygon(SQUARE, labels=(2, 3, 5, 7))
    Q = Polygon(SQUARE[::-1], labels=(5, 3, 2, 7))  # CW input: edge i of the input runs from vertex i to i+1
    by_edge = lambda R: {(e.start, e.end) if e.start < e.end else (e.end, e.start): e.label for e in R.edges}  # noqa: E731
    assert by_edge(P) == by_edge(Q)
    assert Polygon(SQUARE, labels=(1, 1, 1, 1)).labels is None


def test_product_of_footballs():
    m, n, p, q = 3, 1, 2, 5  # left, right, bottom, top
    P = Polygon(SQUARE, labels=(p, n, q, m))
    assert P.boundary_measure == Fr(1, m) + Fr(1, n) + Fr(1, p) + Fr(1, q)
    A = P.A_ext
    assert A.c0 == football_S(m, n, 0) + football_S(p, q, 0)
    assert A.c1 == football_S(m, n, 1) - football_S(m, n, 0)
    assert A.c2 == football_S(p, q, 1) - football_S(p, q, 0)
    r = ExtremalSolver(P, degree=12).solve()
    x, y, _ = polygon_quadrature(P, 25)
    theta = x * (1 - x) * (2 / m + (2 / n - 2 / m) * x)
    exact_hxx = 1 / theta - (m / (2 * x) + n / (2 * (1 - x)))
    assert np.abs(r.potential.derivative(2, 0, x, y) - exact_hxx).max() < 1e-3
    assert np.abs(r.potential.derivative(1, 1, x, y)).max() < 1e-8


def test_labelled_quadrilateral_turns_unstable():
    V = [(0, 0), (5, 0), (4, 1), (0, 2)]
    assert not crease_search(Polygon(V, labels=(1, 4, 4, 4)), n_angles=90).unstable
    r = crease_search(Polygon(V, labels=(1, 12, 12, 12)), n_angles=90)
    assert r.unstable and r.certificate_value < 0
