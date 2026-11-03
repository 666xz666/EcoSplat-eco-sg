"""Total-budget allocation: one compression-ratio knob, one ranking, exact hit.

Budget model (float32 payload per scene)

    kappa = 14 * N + 7 * L

``N`` is the number of kept Gaussians, ``L`` the number of kept spherical
lobes.  14 = position 3 + opacity 1 + scale 3 + rotation 4 + c0 3;
7 = lobe axis 3 + sharpness 1 + amplitude 3.  Every cost is a multiple of 7,
so a requested ``kappa`` is snapped to ``F = round(kappa / 7) * 7`` (residual
<= 14 bytes) and every reachable payload is also a multiple of 7.

Allocation strategy
-------------------
Each point is a multiple-choice bundle: taking it at lobe level ``m`` costs
``14 + 7m`` and yields the value

    v_i(m) = opacity_hat_i + sum_{l<=m} D_hat_il

with ``opacity_hat`` and ``D_hat`` normalised by their global maxima so that
the two axes are comparable - the normalisation is what lets one price
transfer between scenes.  Bundles are ranked by value density ``v / cost`` and
taken greedily, which respects both constraints for free: one level per point
(multiple choice) and lobes only on points that were taken (precedence).  The
remaining budget then buys next lobes of already-taken points at their
marginal density, and finally extra points at level 0; the payload therefore
lands on ``F`` or ``F - 7`` floats and the exact count is asserted at runtime.

Compared with ``allocate_two_stage`` this removes the external split ratio: the
trade-off between points and lobes is resolved by the shared ranking, and the
density at the cut is reported as ``beta`` (value per float, i.e. the shadow
price of the constraint).

The bundle ranking maximises the average density over the level, so it can
prefer a level-0 point over a level-3 point whose lobes are worth more per
float.  :func:`allocate_by_price` splits the two decisions: a price ``beta``
fixes the level of every point (lobes above the price, plus the point when its
value covers its own cost), and the budget then rations the headcount by value
density at that level.  ``beta`` is the shadow price of a float, so the level
is where the byte is actually spent.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .sg_prune import BASE_FLOATS, LOBE_FLOATS

GRID = LOBE_FLOATS  # both costs are multiples of 7 (base 14 = 2 * 7)


@dataclass
class Allocation:
    """Result of :func:`allocate` (all counts in float32 parameters)."""

    keep_points: Tensor      # (P,) bool
    keep_lobes: Tensor       # (P, Lmax) bool
    n_points: int
    n_lobes: int
    achieved: int            # 14 * n_points + 7 * n_lobes
    target: int              # snapped budget F
    beta: float              # value per float at the cut


def dense_floats(num_views: int, height: int, width: int,
                 lobes_per_point: int = 3) -> int:
    """Full-payload budget of the dense model (the compression-ratio baseline).

    ``lobes_per_point`` is the level the model predicts at full budget; our SG
    head uses 3, i.e. 14 + 7 * 3 = 35 float32 per point.
    """
    return (BASE_FLOATS + LOBE_FLOATS * int(lobes_per_point)) *         int(num_views) * int(height) * int(width)


def ratio_to_kappa(ratio: float, num_views: int, height: int, width: int,
                   lobes_per_point: int = 3) -> int:
    """Turn a user-facing compression ratio (e.g. 16 -> 1/16 payload) into kappa."""
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return int(round(dense_floats(num_views, height, width, lobes_per_point) / float(ratio)))


def snap_budget(kappa: int) -> int:
    """Round a byte budget down/up to the reachable 7-float grid (min one point)."""
    return max(BASE_FLOATS, int(round(int(kappa) / GRID)) * GRID)


def _normalise(x: Tensor) -> Tensor:
    x = x.clamp_min(0.0)
    return x / x.max().clamp_min(1e-12)


def _spend_leftover(sel: Tensor, m: Tensor, leftover: int, o: Tensor,
                    d: Tensor, Lmax: int, dev) -> tuple[Tensor, Tensor, int]:
    """Spend the sub-point remainder: next lobes first, then extra level-0 points.

    Both loops stop at zero-value items, so a budget larger than the available
    value returns floats unspent instead of buying dead lobes or empty points;
    the caller then only has to check that the payload is still on the 7-float
    grid.

    Args:
        sel: (n,) indices of the points the ranking already took.
        m: (n,) lobe level per taken point; mutated in place.
        leftover: floats still available (a multiple of ``GRID``).
        o: (P,) normalised opacity score.
        d: (P, Lmax) normalised lobe values, sorted descending per point.
        Lmax: lobe cap per point.
        dev: device of the tensors.

    Returns:
        ``(sel, m, leftover)`` with the extra level-0 points appended.
    """
    # Step 1: grow already-taken points by their next lobe (cost 7).
    while leftover >= LOBE_FLOATS:
        room = (m < Lmax).nonzero(as_tuple=False).squeeze(1)
        if room.numel() == 0:
            break
        value = d[sel[room], m[room]]
        room = room[value > 0]                     # never spend on a zero-value lobe
        if room.numel() == 0:
            break
        marginal = d[sel[room], m[room]] / LOBE_FLOATS
        take = room[marginal.argsort(descending=True)[: int(leftover // LOBE_FLOATS)]]
        m[take] += 1
        leftover -= LOBE_FLOATS * int(take.numel())

    # Step 2: add new points at level 0 (cost 14).
    while leftover >= BASE_FLOATS:
        taken = torch.zeros(o.shape[0], dtype=torch.bool, device=dev)
        taken[sel] = True
        cand = (~taken).nonzero(as_tuple=False).squeeze(1)
        cand = cand[o[cand] > 0]                   # never spend on a zero-value point
        if cand.numel() == 0:
            break
        room = int(leftover // BASE_FLOATS)
        pick = cand[(o[cand]).argsort(descending=True)[:room]]
        sel = torch.cat([sel, pick])
        m = torch.cat([m, torch.zeros_like(pick)])
        leftover -= BASE_FLOATS * int(pick.numel())

    return sel, m, leftover


def allocate(kappa: int, opacity: Tensor, D: Tensor,
             lobe_max: int | None = None) -> Allocation:
    """Allocate a float budget between Gaussians and their lobes.

    Args:
        kappa: target payload in float32 parameters (per scene).
        opacity: (P,) per-point opacity score; higher keeps the point.
        D: (P, L) per-lobe dynamic range ``|a| (1 - exp(-2s))``.
        lobe_max: optional cap on lobes per point (defaults to ``D.shape[1]``).

    Returns:
        :class:`Allocation` with boolean masks, counts, achieved payload and the
        marginal density at the cut.
    """
    P = int(opacity.shape[0])
    if P == 0:
        return Allocation(opacity.new_zeros(0, dtype=torch.bool),
                          opacity.new_zeros(0, 0, dtype=torch.bool),
                          0, 0, 0, snap_budget(kappa), 0.0)

    dev = opacity.device
    L = int(D.shape[1]) if D.dim() == 2 else 0
    Lmax = L if lobe_max is None else max(0, min(int(lobe_max), L))
    F = snap_budget(kappa)

    o = _normalise(opacity.reshape(P).float())
    if Lmax > 0:
        d = D[:, :Lmax].float()
        d = _normalise(d)
        d, _ = d.sort(dim=1, descending=True)      # marginal lobes at the front
        cum = d.cumsum(dim=1)                      # (P, Lmax)
    else:
        d = o.new_zeros((P, 0))
        cum = o.new_zeros((P, 0))

    # Bundle table: level m costs 14 + 7m, value = opacity_hat + sum of top-m D.
    lv = torch.arange(Lmax + 1, device=dev)
    costs = (BASE_FLOATS + LOBE_FLOATS * lv).to(o.dtype)          # (M,)
    val = o[:, None].expand(P, Lmax + 1)
    if Lmax > 0:
        val = torch.cat([o[:, None], o[:, None] + cum], dim=1)    # (P, M)
    dens = val / costs                                            # (P, M)

    best_dens, best_m = dens.max(dim=1)
    best_cost = costs[best_m]

    # Greedy prefix over points ordered by their best bundle density.
    order = best_dens.argsort(descending=True)
    cum_cost = best_cost[order].cumsum(0)
    n_pts = int((cum_cost <= F).sum().item())
    n_pts = max(n_pts, 1)                       # F >= 14 guarantees room for one point
    sel = order[:n_pts]
    m = best_m[sel].clone()
    leftover = F - int(costs[m].sum().item())
    beta = float(best_dens[sel[-1]].item())

    sel, m, leftover = _spend_leftover(sel, m, leftover, o, d, Lmax, dev)

    n_pts = int(sel.numel())
    n_lobes = int(m.sum().item())
    achieved = BASE_FLOATS * n_pts + LOBE_FLOATS * n_lobes
    if achieved > F or (F - achieved) % GRID != 0:
        raise AssertionError(
            f"budget mismatch: achieved={achieved} target={F} "
            f"(N={n_pts}, L={n_lobes})")

    keep_points = torch.zeros(P, dtype=torch.bool, device=dev)
    keep_points[sel] = True
    keep_lobes = torch.zeros((P, Lmax), dtype=torch.bool, device=dev)
    if Lmax > 0:
        keep_lobes[sel] = (torch.arange(Lmax, device=dev)[None, :] < m[:, None])

    return Allocation(keep_points, keep_lobes, n_pts, n_lobes, achieved, F, beta)


def allocate_by_price(kappa: int, opacity: Tensor, D: Tensor,
                      lobe_max: int | None = None, iters: int = 48,
                      repair: bool = True) -> Allocation:
    """Allocate with one marginal price instead of a per-point bundle ranking.

    Point and lobe are priced on their own: a point costs 14 floats and is worth
    ``opacity_hat``, lobe ``l`` costs 7 and is worth ``D_hat_il``.  At a price
    ``beta`` (value per float) each point solves its own multiple-choice
    problem, so the choice separates into a threshold per item:

    $$ m_i(beta) = number of lobes l with D_hat_il / 7 >= beta $$

    $$ keep_i(beta) = [ opacity_hat_i + sum_{l<=m_i} D_hat_il >= beta * (14 + 7 m_i) ] $$

    The cost of that choice is non-increasing in ``beta`` (a higher price can
    only shrink the level), so bisection returns the largest price whose cost
    fits ``F``.  That price then fixes the level of *every* point and the budget
    only decides how many points get in: the kept set is the prefix of the
    value density ``(opacity_hat + sum D_hat) / (14 + 7 m_i)`` evaluated at the
    price-fixed level.  Separating the two decisions is what stops the level
    from collapsing to 0 at tight budgets - the average density of a bundle no
    longer decides whether its lobes are bought, the marginal price does.  The
    sub-point remainder is spent by :func:`_spend_leftover`.

    Args:
        kappa: target payload in float32 parameters (per scene).
        opacity: (P,) per-point opacity score; higher keeps the point.
        D: (P, L) per-lobe dynamic range ``|a| (1 - exp(-2s))``.
        lobe_max: optional cap on lobes per point (defaults to ``D.shape[1]``).
        iters: bisection steps on ``beta`` (48 resolves ~1e-14 of the range).
        repair: spend the remainder; ``False`` returns the pure threshold
            solution, which is what the unit checks inspect.

    Returns:
        :class:`Allocation` with the masks, counts, achieved payload and the
        price at the cut.
    """
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

    # Step 1: a budget above the dense payload leaves nothing to choose.
    full = BASE_FLOATS * P + LOBE_FLOATS * P * Lmax
    if full <= F:
        keep_points = torch.ones(P, dtype=torch.bool, device=dev)
        keep_lobes = torch.ones((P, Lmax), dtype=torch.bool, device=dev)
        return Allocation(keep_points, keep_lobes, P, P * Lmax, full, F, 0.0)

    lv = torch.arange(Lmax + 1, device=dev)
    costs = (BASE_FLOATS + LOBE_FLOATS * lv).to(o.dtype)       # (M,) = 14 + 7m
    padded = torch.cat([o.new_zeros((P, 1)), d.cumsum(dim=1)], 1)  # (P, M)

    def _choice(beta: float) -> tuple[Tensor, Tensor, int]:
        """Best level per point at price ``beta``; returns (keep, m, cost)."""
        m = (d >= beta * LOBE_FLOATS).sum(dim=1)               # (P,)
        value = o + padded.gather(1, m[:, None]).squeeze(1)
        keep = value >= beta * costs[m]
        return keep, m, int((costs[m] * keep).sum().item())

    # Step 2: bisect for the largest feasible price; lo is infeasible (full
    # payload), hi is feasible (nothing bought), so the invariant holds.
    hi = float(max(o.max().item(),
                   d.max().item() if Lmax > 0 else 0.0)) + 1e-6
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _choice(mid)[2] <= F:
            hi = mid
        else:
            lo = mid
    # Step 3: the price fixes the level menu, the budget fixes the headcount.
    beta = hi
    m = (d >= beta * LOBE_FLOATS).sum(dim=1)                   # (P,)
    menu_cost = costs[m]                                       # (P,)
    value = o + padded.gather(1, m[:, None]).squeeze(1)
    order = (value / menu_cost).argsort(descending=True)
    cum = menu_cost[order].cumsum(0)
    n_pts = int((cum <= F).sum().item())
    sel = order[:n_pts]
    m = m[sel].clone()
    leftover = F - int(menu_cost[sel].sum().item())

    # Step 3: mop up the sub-threshold remainder on the grid.
    if repair:
        sel, m, leftover = _spend_leftover(sel, m, leftover, o, d, Lmax, dev)

    n_pts = int(sel.numel())
    n_lobes = int(m.sum().item())
    achieved = BASE_FLOATS * n_pts + LOBE_FLOATS * n_lobes
    if achieved > F or (F - achieved) % GRID != 0:
        raise AssertionError(
            f"budget mismatch: achieved={achieved} target={F} "
            f"(N={n_pts}, L={n_lobes})")

    keep_points = torch.zeros(P, dtype=torch.bool, device=dev)
    keep_points[sel] = True
    keep_lobes = torch.zeros((P, Lmax), dtype=torch.bool, device=dev)
    if Lmax > 0:
        keep_lobes[sel] = (torch.arange(Lmax, device=dev)[None, :] < m[:, None])

    return Allocation(keep_points, keep_lobes, n_pts, n_lobes, achieved, F,
                      beta)


def allocate_two_stage(kappa: int, opacity: Tensor, D: Tensor,
                       split: float = 0.6) -> Allocation:
    """Two-stage baseline: split the budget, top-k points, then top-L lobes.

    ``split`` is the share of ``F`` handed to the geometry axis; the point
    budget is ``N = split * F / 14`` clamped to the feasible window, and every
    remaining float goes to lobes taken by ``D`` over the survivors.  Kept for
    the A/B comparison against :func:`allocate`.
    """
    P = int(opacity.shape[0])
    L = int(D.shape[1]) if D.dim() == 2 else 0
    F = snap_budget(kappa)
    if P == 0:
        return Allocation(opacity.new_zeros(0, dtype=torch.bool),
                          opacity.new_zeros(0, 0, dtype=torch.bool),
                          0, 0, 0, F, 0.0)

    dev = opacity.device
    lo = -(-F // (BASE_FLOATS + LOBE_FLOATS * max(L, 0))) if L > 0 else -(-F // BASE_FLOATS)
    lo = max(1, lo)
    hi = max(1, F // BASE_FLOATS)
    n_pts = int(round(split * F / BASE_FLOATS))
    n_pts = max(lo, min(hi, n_pts))
    n_lobes = (F - BASE_FLOATS * n_pts) // LOBE_FLOATS
    n_lobes = max(0, min(n_lobes, L * n_pts))

    o = _normalise(opacity.reshape(P).float())
    sel = o.argsort(descending=True)[:n_pts]

    keep_lobes = torch.zeros((P, L), dtype=torch.bool, device=dev)
    if n_lobes > 0 and L > 0:
        d = D[sel].float()
        flat = d.reshape(-1)
        take = flat.argsort(descending=True)[:n_lobes]
        rows = torch.div(take, L, rounding_mode="floor")
        cols = take % L
        keep_lobes[sel[rows], cols] = True
        n_lobes = int(keep_lobes.sum().item())

    keep_points = torch.zeros(P, dtype=torch.bool, device=dev)
    keep_points[sel] = True
    achieved = BASE_FLOATS * n_pts + LOBE_FLOATS * n_lobes
    beta = float(o[sel[-1]].item()) if n_pts else 0.0
    return Allocation(keep_points, keep_lobes, n_pts, n_lobes, achieved, F, beta)
