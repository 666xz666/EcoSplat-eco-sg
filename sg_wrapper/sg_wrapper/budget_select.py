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

from .budget import allocate_by_price_batched


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

    实现约定：所有视角一次进 ``allocate_by_price_batched``（R = B*V），整帧只
    同步两次（``nmax`` 与统计量）；旧实现是逐视角循环 + 每步二分一次
    ``.item()``，约 ``iters * B * V`` 次同步，实测 ~95 ms/帧。
    """
    b, v, p = opacity.shape
    L = int(lobe_value.shape[-1])
    lmax = L if lobes_per_point is None else max(0, min(int(lobes_per_point), L))
    device = opacity.device
    r = b * v

    # 步骤 1：整批按 D 降序排一次——分配器的档位计数默认「前排的 lobe
    # 更值钱」，排序只服务这一条，最后会把 mask 散射回原始顺序。
    d = lobe_value.reshape(r, p, L).float()
    order = d.argsort(dim=2, descending=True)
    d_sorted = d.gather(2, order)
    alloc = allocate_by_price_batched(
        int(kappa_per_view), opacity.reshape(r, p).float(), d_sorted,
        lobe_max=lmax, iters=iters,
    )

    # 步骤 2：散射回原始列序，调用方才能直接乘到颜色通道上；少这一步
    # 会按排序后的列去剪 lobe，颜色就错位了。
    keep = torch.zeros((r, p, L), dtype=torch.bool, device=device)
    if lmax > 0:
        keep.scatter_(2, order[:, :, :lmax], alloc.keep_lobes)

    # 步骤 3：各视角保留数不同，gather 需要矩形索引，因此补到最大值。
    # stable 降序 argsort 把 True 排到前面且保持索引升序，与旧实现的
    # nonzero() 逐位一致；补进来的槽位在 valid 里是 False、不携带任何
    # lobe，调用方把密度置零即可（不参与渲染）。
    n_pts = alloc.n_points
    nmax = max(1, int(n_pts.max()))
    idx_all = alloc.keep_points.argsort(dim=1, descending=True, stable=True)
    index = idx_all[:, :nmax]
    valid = torch.arange(nmax, device=device)[None, :] < n_pts[:, None]
    index = torch.where(valid, index, torch.zeros_like(index))
    keep_lobes = keep.gather(1, index.unsqueeze(-1).expand(r, nmax, L)) \
        & valid[..., None]

    # 步骤 4：统计量一次同步取回。
    stats = torch.stack([
        n_pts.to(torch.float64).mean(),
        alloc.n_lobes.to(torch.float64).mean(),
        alloc.achieved.to(torch.float64).mean(),
        alloc.beta.mean(),
    ]).tolist()
    return BudgetSelection(
        index=index.reshape(b, v, nmax),
        valid=valid.reshape(b, v, nmax),
        keep_lobes=keep_lobes.reshape(b, v, nmax, L),
        n_points=stats[0],
        n_lobes=stats[1],
        achieved=stats[2],
        target=float(alloc.target),
        beta=stats[3],
    )
