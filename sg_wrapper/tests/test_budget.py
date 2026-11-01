"""CPU-only checks for the total-budget allocator (seconds, no repo deps).

Run from the repo root:

    python -m pytest sg_wrapper/tests/test_budget.py -q
    # or without pytest:  python sg_wrapper/tests/test_budget.py
"""
import torch

from sg_wrapper.budget import (
    BASE_FLOATS,
    GRID,
    LOBE_FLOATS,
    allocate,
    allocate_two_stage,
    dense_floats,
    ratio_to_kappa,
    snap_budget,
)


def _toy(P, L=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(P, generator=g)
    d = torch.rand(P, L, generator=g) * torch.linspace(4.0, 1.0, L)
    return o, d


def _capacity(P, L=3):
    return P * (BASE_FLOATS + LOBE_FLOATS * L)


def test_snap_and_ratio():
    assert snap_budget(100) % GRID == 0
    assert snap_budget(1) == BASE_FLOATS                 # never below one point
    assert dense_floats(24, 256, 256) == 35 * 24 * 256 * 256
    k16 = ratio_to_kappa(16, 24, 256, 256)
    assert abs(k16 - dense_floats(24, 256, 256) / 16) <= 1


def test_exact_hit_within_capacity():
    """Inside capacity the payload must land on F (unused <= 13 floats)."""
    P, L = 4096, 3
    o, d = _toy(P, L)
    cap = _capacity(P, L)
    for frac in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        a = allocate(int(cap * frac), o, d)
        assert a.achieved <= a.target
        assert a.target - a.achieved < BASE_FLOATS
        assert (a.target - a.achieved) % GRID == 0
        assert a.achieved == BASE_FLOATS * a.n_points + LOBE_FLOATS * a.n_lobes
        assert not (a.keep_lobes[~a.keep_points]).any()
        assert int(a.keep_lobes.sum()) == a.n_lobes
        assert int(a.keep_points.sum()) == a.n_points
        assert int(a.keep_lobes.sum(dim=1).max()) <= L


def test_over_capacity_is_graceful():
    """A budget above capacity saturates at full payload instead of raising."""
    P, L = 512, 3
    o, d = _toy(P, L)
    a = allocate(_capacity(P, L) * 10, o, d)
    assert a.achieved <= a.target
    assert a.n_points == P
    assert a.n_lobes <= P * L


def test_monotone_in_budget():
    P, L = 2048, 3
    o, d = _toy(P, L)
    cap = _capacity(P, L)
    prev = None
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        a = allocate(int(cap * frac), o, d)
        if prev is not None:
            assert a.n_points >= prev.n_points
            assert a.n_lobes >= prev.n_lobes
            assert a.achieved >= prev.achieved
        prev = a


def test_flat_dynamic_range_buys_points():
    """Zero-valued lobes are never bought: the budget goes to points only."""
    P, L = 4096, 3
    o, _ = _toy(P, L)
    d0 = torch.zeros(P, L)
    a = allocate(_capacity(P, L) // 4, o, d0)
    assert a.n_lobes == 0
    assert a.n_points == a.target // BASE_FLOATS


def test_all_value_in_lobes():
    """With flat opacity and rich lobes the margin must buy lobes."""
    P, L = 4096, 3
    o = torch.full((P,), 0.01)
    d = torch.full((P, L), 1.0)
    budget = P * BASE_FLOATS // 8
    a = allocate(budget, o, d)
    assert a.n_lobes > 0
    assert a.target - a.achieved < BASE_FLOATS


def test_two_stage_matches_budget_contract():
    P, L = 2048, 3
    o, d = _toy(P, L)
    for split in (0.3, 0.6, 0.9):
        a = allocate_two_stage(_capacity(P, L) // 3, o, d, split=split)
        assert a.achieved <= a.target
        assert a.achieved == BASE_FLOATS * a.n_points + LOBE_FLOATS * a.n_lobes


def test_no_lobes_available():
    P = 4096
    o, _ = _toy(P, 3)
    a = allocate(1_500, o, torch.zeros(P, 0))
    assert a.n_lobes == 0 and a.keep_lobes.shape[1] == 0
    assert a.target - a.achieved < BASE_FLOATS


def test_deterministic():
    P, L = 1024, 3
    o, d = _toy(P, L, seed=7)
    a1 = allocate(_capacity(P, L) // 2, o, d)
    a2 = allocate(_capacity(P, L) // 2, o, d)
    assert torch.equal(a1.keep_points, a2.keep_points)
    assert torch.equal(a1.keep_lobes, a2.keep_lobes)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all checks passed")
