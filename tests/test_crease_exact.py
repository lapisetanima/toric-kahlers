import math

import numpy as np
import pytest

from extremal import Polygon, blowup_cp2, projective_plane
from extremal.crease_exact import crease_minimum, optimal_crease, optimality_scan
from extremal.quadrature import polygon_quadrature
from extremal.stability import CreaseProfile, crease_search

Q12 = Polygon([(0, 0), (5, 0), (4, 1), (0, 2)], labels=(1, 12, 12, 12))


@pytest.fixture(scope="module")
def q12_optimal():
    return optimal_crease(Q12)


def test_optimal_crease_satisfies_the_piece_decomposition(q12_optimal):
    o = q12_optimal
    assert not o.stable
    assert float(o.norm) == pytest.approx(0.16417411137442506876, abs=1e-15)
    # A − f* is the extremal function of each piece (zero measure on the crease), so A_1 = A_2 on it
    assert o.piece_residual < 1e-30 and o.continuity_residual < 1e-30 and abs(o.identity_residual) < 1e-30
    assert o.crease.minimal_polynomial_m().degree() == 18


def test_optimal_crease_norm_by_quadrature(q12_optimal):
    """‖f*‖² = −L_A(f*), and f* ⊥ affine functions, checked independently with floats."""
    o = q12_optimal
    a, b = (float(v) for v in o.crease.normal)
    L_phi = CreaseProfile(Q12, (a, b)).L(float(o.crease.s))
    assert float(o.c) * L_phi == pytest.approx(-float(o.norm) ** 2, rel=1e-9)
    x, y, w = polygon_quadrature(Q12, 400)
    f = o(x, y)
    assert np.sum(w * f * f) == pytest.approx(float(o.norm) ** 2, rel=1e-4)
    for g in (1, x, y):
        assert abs(np.sum(w * f * g)) < 1e-6


def test_optimal_crease_passes_the_first_order_scan(q12_optimal):
    scan = optimality_scan(q12_optimal, n_angles=90, samples=80)
    assert scan["min"] > -1e-6
    assert math.degrees(scan["at"]) == pytest.approx(math.degrees(q12_optimal.crease.theta), abs=2.5)


@pytest.mark.parametrize("P", [projective_plane(1), blowup_cp2(3, 1)], ids=["CP2", "F1"])
def test_stable_polygons_have_no_destabilizer(P):
    assert optimal_crease(P).stable


@pytest.mark.parametrize(
    "P, value", [(projective_plane(1), 1 / 3), (blowup_cp2(3, 1), 0.299378880093045), (Q12, -0.361405161298762)],
    ids=["CP2", "F1", "Q12"],
)
def test_exact_boundary_minimum_matches_and_beats_the_scan(P, value):
    exact = crease_minimum(P)
    assert float(exact.value) == pytest.approx(value, abs=1e-12)
    scan = crease_search(P, n_angles=360, exact=False)
    assert float(exact.value) <= scan.best_margin + 1e-12
    assert scan.best_margin == pytest.approx(float(exact.value), abs=1e-3)
