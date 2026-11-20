"""Equivalence checks: batched price rule vs the pre-vectorization loops.

The reference implementations below are the old per-view code kept verbatim
(Python loops, ``.item()`` syncs).  On tie-free random inputs the batched
core in ``sg_wrapper.budget`` must reproduce them bit for bit; degenerate
inputs are checked against the contract (grid, no overspend) instead, because
exact ties make the prefix cut ambiguous.

Run from the repo root:

    python -m pytest sg_wrapper/tests/test_budget_vectorized.py -q
"""
import torch

from sg_wrapper.budget import (
    BASE_FLOATS,
    GRID,
    LOBE_FLOATS,
    Allocation,
    allocate_by_price_batched,
    snap_budget,
    _normalise,
    _spend_leftover,
)
from sg_wrapper.budget_select import select_by_budget


# ---------------------------------------------------------------- reference
# Pre-vectorization allocate_by_price (per-view loop + per-step .item()).

def _reference_allocate_by_price(kappa, opacity, D, lobe_max=None,
                                 iters=48, repair=True):
    P = int(opacity.shape[0])
    L = int(D.shape[1]) if D.dim() == 2 else 0
    Lmax = L if lobe_max is None else max(0, min(int(lobe_max), L))
    F = snap_budget(kappa)
    if P == 0:
        return Allocation(opacity.new_zeros(0, dtype=torch.bool),
                          opacity.new_zeros(0, 0, dtype=torch.bool),
                          0, 0, 0, F, 0.0)

    dev = opacity.device
    o = _normalise(opacity.reshape(P).float())
    if Lmax > 0:
        d = _normalise(D[:, :Lmax].float())
        d, _ = d.sort(dim=1, descending=True)
    else:
        d = o.new_zeros((P, 0))

    full = BASE_FLOATS * P + LOBE_FLOATS * P * Lmax
    if full <= F:
        return Allocation(torch.ones(P, dtype=torch.bool, device=dev),
                          torch.ones((P, Lmax), dtype=torch.bool, device=dev),
                          P, P * Lmax, full, F, 0.0)

    lv = torch.arange(Lmax + 1, device=dev)
    costs = (BASE_FLOATS + LOBE_FLOATS * lv).to(o.dtype)
    padded = torch.cat([o.new_zeros((P, 1)), d.cumsum(dim=1)], 1)

    def _choice(beta):
        m = (d >= beta * LOBE_FLOATS).sum(dim=1)
        value = o + padded.gather(1, m[:, None]).squeeze(1)
        keep = value >= beta * costs[m]
        return keep, m, int((costs[m] * keep).sum().item())

    hi = float(max(o.max().item(), d.max().item() if Lmax > 0 else 0.0)) + 1e-6
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _choice(mid)[2] <= F:
            hi = mid
        else:
            lo = mid
    beta = hi
    m = (d >= beta * LOBE_FLOATS).sum(dim=1)
    menu_cost = costs[m]
    value = o + padded.gather(1, m[:, None]).squeeze(1)
    order = (value / menu_cost).argsort(descending=True)
    cum = menu_cost[order].cumsum(0)
    n_pts = int((cum <= F).sum().item())
    n_pts = min(n_pts, int((value[order] > 0).sum().item()))
    sel = order[:n_pts]
    m = m[sel].clone()
    leftover = F - int(menu_cost[sel].sum().item())

    if repair:
        sel, m, leftover = _spend_leftover(sel, m, leftover, o, d, Lmax, dev)

    n_pts = int(sel.numel())
    n_lobes = int(m.sum().item())
    achieved = BASE_FLOATS * n_pts + LOBE_FLOATS * n_lobes

    keep_points = torch.zeros(P, dtype=torch.bool, device=dev)
    keep_points[sel] = True
    keep_lobes = torch.zeros((P, Lmax), dtype=torch.bool, device=dev)
    if Lmax > 0:
        keep_lobes[sel] = (torch.arange(Lmax, device=dev)[None, :] < m[:, None])
    return Allocation(keep_points, keep_lobes, n_pts, n_lobes, achieved, F, beta)


def _random_case(R, P, L, seed):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(R, P, generator=g)
    d = torch.rand(R, P, L, generator=g) * torch.linspace(4.0, 1.0, L)
    return o, d


# ------------------------------------------------------------------ checks

def test_batched_matches_reference():
    for seed in range(4):
        R, P, L = 3, 512, 3
        o, d = _random_case(R, P, L, seed)
        cap = P * (BASE_FLOATS + LOBE_FLOATS * L)
        for frac in (0.02, 0.1, 0.3, 0.5, 0.7, 0.95):
            kappa = int(cap * frac)
            a = allocate_by_price_batched(kappa, o, d)
            for ri in range(R):
                ref = _reference_allocate_by_price(kappa, o[ri], d[ri])
                assert torch.equal(a.keep_points[ri], ref.keep_points), (seed, frac, ri)
                assert torch.equal(a.keep_lobes[ri], ref.keep_lobes), (seed, frac, ri)
                assert int(a.n_points[ri]) == ref.n_points
                assert int(a.n_lobes[ri]) == ref.n_lobes
                assert int(a.achieved[ri]) == ref.achieved
                assert float(a.beta[ri]) == ref.beta


def test_batched_matches_reference_no_repair():
    for seed in (11, 12):
        R, P, L = 2, 256, 3
        o, d = _random_case(R, P, L, seed)
        cap = P * (BASE_FLOATS + LOBE_FLOATS * L)
        for frac in (0.15, 0.55):
            kappa = int(cap * frac)
            a = allocate_by_price_batched(kappa, o, d, repair=False)
            for ri in range(R):
                ref = _reference_allocate_by_price(kappa, o[ri], d[ri], repair=False)
                assert torch.equal(a.keep_points[ri], ref.keep_points)
                assert torch.equal(a.keep_lobes[ri], ref.keep_lobes)
                assert int(a.achieved[ri]) == ref.achieved


def test_batched_matches_reference_lobe_max():
    R, P, L = 2, 256, 3
    o, d = _random_case(R, P, L, 21)
    cap = P * (BASE_FLOATS + LOBE_FLOATS * L)
    for lmax in (0, 1, 2):
        kappa = int(cap * 0.4)
        a = allocate_by_price_batched(kappa, o, d, lobe_max=lmax)
        for ri in range(R):
            ref = _reference_allocate_by_price(kappa, o[ri], d[ri], lobe_max=lmax)
            assert torch.equal(a.keep_points[ri], ref.keep_points), lmax
            assert torch.equal(a.keep_lobes[ri], ref.keep_lobes), lmax
            assert int(a.achieved[ri]) == ref.achieved, lmax


def test_degenerate_contract_only():
    """Tie-heavy inputs: check the contract, not the ambiguous tie order."""
    R, P, L = 4, 128, 3
    cases = [
        (torch.zeros(R, P), torch.zeros(R, P, L)),          # nothing has value
        (torch.ones(R, P), torch.ones(R, P, L)),            # everything ties
        (torch.rand(R, P), torch.zeros(R, P, L)),           # lobes all dead
    ]
    cap = P * (BASE_FLOATS + LOBE_FLOATS * L)
    for o, d in cases:
        for frac in (0.1, 0.5):
            a = allocate_by_price_batched(int(cap * frac), o, d)
            assert not (a.keep_lobes & ~a.keep_points[..., None]).any()
            assert torch.all(a.achieved <= a.target)
            assert torch.all((a.target - a.achieved) % GRID == 0)
            assert torch.all(a.achieved == BASE_FLOATS * a.n_points
                             + LOBE_FLOATS * a.n_lobes)


def test_batched_over_capacity_and_empty():
    R, P, L = 2, 64, 3
    o, d = _random_case(R, P, L, 31)
    cap = P * (BASE_FLOATS + LOBE_FLOATS * L)
    a = allocate_by_price_batched(cap * 10, o, d)
    assert torch.all(a.n_points == P) and torch.all(a.n_lobes == P * L)
    assert torch.all(a.beta == 0.0)
    e = allocate_by_price_batched(1000, o[:, :0], d[:, :0])
    assert e.keep_points.shape == (R, 0) and int(e.n_points.sum()) == 0


def test_select_by_budget_matches_reference():
    """The padded selection must equal the old per-view loop reconstruction."""
    b, v, p, L = 1, 3, 64, 3
    o, d = _random_case(b * v, p, L, 41)
    o = o.reshape(b, v, p)
    d = d.reshape(b, v, p, L)
    kappa = p * (BASE_FLOATS + LOBE_FLOATS * L) // 3
    sel = select_by_budget(o, d, kappa, iters=32)

    for bi in range(b):
        for vi in range(v):
            dv = d[bi, vi].float()
            order = dv.argsort(dim=1, descending=True)
            ref = _reference_allocate_by_price(
                kappa, o[bi, vi], dv.gather(1, order), iters=32)
            # rebuild the sorted-column mask in original order
            keep = torch.zeros((p, L), dtype=torch.bool)
            keep.scatter_(1, order[:, :L], ref.keep_lobes)
            idx = ref.keep_points.nonzero(as_tuple=False).squeeze(1)
            n = int(idx.numel())
            assert int(sel.valid[bi, vi].sum()) == n
            assert torch.equal(sel.index[bi, vi, :n].sort().values, idx)
            assert torch.equal(sel.keep_lobes[bi, vi, :n], keep[idx])
            assert not torch.any(sel.keep_lobes[bi, vi, n:])


def test_select_stats_and_shapes():
    b, v, p, L = 2, 4, 32, 3
    o = torch.rand(b, v, p)
    d = torch.rand(b, v, p, L)
    sel = select_by_budget(o, d, 14 * p // 2)
    assert sel.index.shape[:2] == (b, v)
    assert sel.keep_lobes.shape[-1] == L
    assert sel.n_points >= 0 and sel.beta >= 0.0
    assert sel.target == float(snap_budget(14 * p // 2))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all checks passed")
