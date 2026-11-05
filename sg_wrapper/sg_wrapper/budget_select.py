"""Budget-mode selection: one payload cap decides points and lobes together.

The PLGC path keeps a fixed *fraction of points* and every kept point carries
all ``num_lobes`` lobes, so the colour axis is frozen and the payload moves
only with the point count.  This module turns the sampled rate into a *total
payload* instead: ``kappa = 35 * H * W * rate`` floats per view, split by
:func:`sg_wrapper.budget.allocate_by_price` between points (14 floats each)
and lobes (7 each).  The step is non-differentiable -- like the top-k it
replaces -- so gradients still flow through the parameters of the selected
set and through ``pred_opacity``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .budget import allocate_by_price


@dataclass
class BudgetSelection:
    """Per-view selection under one total payload cap."""

    index: Tensor       # (B, V, N) long, padded with 0
    valid: Tensor       # (B, V, N) bool, False for padding
    keep_lobes: Tensor  # (B, V, N, L) bool, in the original lobe order
    n_points: float     # mean over views
    n_lobes: float      # mean over views
    achieved: float     # mean payload per view
    target: float       # snapped payload per view
    beta: float         # mean shadow price at the cut


@torch.no_grad()
def select_by_budget(opacity: Tensor, lobe_value: Tensor, kappa_per_view: int,
                     lobes_per_point: Optional[int] = None,
                     iters: int = 32) -> BudgetSelection:
    """Select points and lobes per view under one float budget.

    Args:
        opacity: (B, V, P) per-point opacity score; higher keeps the point.
        lobe_value: (B, V, P, L) per-lobe value ``|a| (1 - exp(-2 s))``.
        kappa_per_view: float budget of a single view.
        lobes_per_point: cap on lobes per kept point; defaults to ``L``.
        iters: bisection steps of :func:`allocate_by_price`.

    Returns:
        :class:`BudgetSelection` with padded gather indices, a padding mask,
        and the lobe mask in the **original** lobe order (not the D-sorted one
        the allocator works in).
    """
    b, v, p = opacity.shape
    L = int(lobe_value.shape[-1])
    lmax = L if lobes_per_point is None else max(0, min(int(lobes_per_point), L))
    device = opacity.device

    rows: list[tuple[Tensor, Tensor, int, int, int, int, float]] = []
    for bi in range(b):
        for vi in range(v):
            # 步骤 1：先按 D 降序排一次——分配器的档位计数默认「前排的 lobe
            # 更值钱」，排序只服务这一条，最后会把 mask 散射回原始顺序。
            d = lobe_value[bi, vi].float()
            order = d.argsort(dim=1, descending=True)
            d_sorted = d.gather(1, order)
            alloc = allocate_by_price(
                int(kappa_per_view), opacity[bi, vi].float(), d_sorted,
                lobe_max=lmax, iters=iters,
            )
            # 步骤 2：散射回原始列序，调用方才能直接乘到颜色通道上；少这一步
            # 会按排序后的列去剪 lobe，颜色就错位了。
            keep = torch.zeros((p, L), dtype=torch.bool, device=device)
            if lmax > 0:
                keep.scatter_(1, order[:, :lmax], alloc.keep_lobes)
            idx = alloc.keep_points.nonzero(as_tuple=False).squeeze(1)
            rows.append((idx, keep[idx], alloc.n_points, alloc.n_lobes,
                         alloc.achieved, alloc.target, alloc.beta))

    # 步骤 3：各视角保留数不同，gather 需要矩形索引，因此补到最大值；补进来
    # 的槽位在 valid 里是 False，调用方把密度置零即可（不参与渲染）。
    nmax = max(1, max(int(r[0].numel()) for r in rows))
    index = torch.zeros((b, v, nmax), dtype=torch.long, device=device)
    valid = torch.zeros((b, v, nmax), dtype=torch.bool, device=device)
    keep_lobes = torch.zeros((b, v, nmax, L), dtype=torch.bool, device=device)
    for k, (idx, keep, *_rest) in enumerate(rows):
        n = int(idx.numel())
        if n == 0:
            continue
        bi, vi = divmod(k, v)
        index[bi, vi, :n] = idx
        valid[bi, vi, :n] = True
        keep_lobes[bi, vi, :n] = keep

    n_views = len(rows)
    return BudgetSelection(
        index=index,
        valid=valid,
        keep_lobes=keep_lobes,
        n_points=sum(r[2] for r in rows) / n_views,
        n_lobes=sum(r[3] for r in rows) / n_views,
        achieved=sum(r[4] for r in rows) / n_views,
        target=sum(r[5] for r in rows) / n_views,
        beta=sum(r[6] for r in rows) / n_views,
    )
