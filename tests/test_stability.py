import random
from fractions import Fraction

import pytest

from extremal import blowup_cp2, projective_plane, random_delzant
from extremal.models import wang_zhou_nonagon
from extremal.stability import CreaseProfile, crease_search


@pytest.mark.parametrize(
    "P", [blowup_cp2(3, 1), wang_zhou_nonagon(100), random_delzant(4, seed=3)[0]], ids=["F1", "WZ100", "random"]
)
def test_exact_profile_matches_direct_crease_integral(P):
    rng = random.Random(0)
    for a, b in [(1, 0), (0, 1), (1, 1), (2, -1), (-1, 3), (27, 10)]:
        exact = CreaseProfile(P, (a, b))
        floaty = CreaseProfile(P, (float(a), float(b)))
        lo, hi = exact.t_range
        for _ in range(5):
            s = lo + (hi - lo) * Fraction(rng.randint(1, 99), 100)
            value = exact.L(s)
            assert value == P.L_crease(a, b, -s)
            assert float(floaty.L(float(s))) == pytest.approx(float(value), rel=1e-7, abs=1e-7)


def test_profile_vanishes_at_both_ends_and_is_symmetric_in_n():
    P = blowup_cp2(3, 1)
    prof = CreaseProfile(P, (2, 1))
    lo, hi = prof.t_range
    assert prof.L(lo) == 0 and prof.L(hi) == 0
    # n and −n give the same values: max(0, g) and max(0, −g) differ by the affine g
    neg = CreaseProfile(P, (-2, -1))
    s = lo + (hi - lo) / 3
    assert prof.L(s) == neg.L(-s)


def test_stable_examples_have_positive_margin():
    for P in (projective_plane(1), blowup_cp2(3, 1)):
        result = crease_search(P, n_angles=180)
        assert result.best_margin > 0.1
        assert not result.unstable
    assert crease_search(projective_plane(1), n_angles=180).best_margin == pytest.approx(1 / 3, abs=1e-3)


def test_wang_zhou_nonagon_is_unstable_for_large_alpha():
    P = wang_zhou_nonagon(10_000)
    result = crease_search(P, n_angles=180)
    assert result.unstable
    a, b, c = result.certificate
    assert P.L_crease(a, b, c) == result.certificate_value < 0
