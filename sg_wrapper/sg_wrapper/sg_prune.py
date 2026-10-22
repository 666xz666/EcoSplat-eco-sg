"""Post-hoc lobe budgeting: the inference half of the second control axis.

MEGS² prunes lobes by driving sharpness to zero during optimization and then
removing the flattened lobes. A feed-forward model cannot run that iteration,
so the budget is applied after the fact:

    D_l = mean_c |a_lc| * (1 - exp(-2 s_l))        # MEGS² range metric
    keep the ``round(L_bar * G)`` lobes with the largest D
    fold each removed lobe's sphere average into c0:
        dc0 = a_l * (1 - exp(-2 s_l)) / (2 s_l)    # exact (MEGS² Eq. 12)

Removed lobes are zeroed in amplitude instead of being physically dropped.
That keeps every tensor rectangular and batch-friendly, and is numerically
identical to removal because a zero-amplitude lobe contributes nothing. The
memory claim is accounted from the per-point kept count (``14 + 7L`` floats),
which is what a grouped export would actually write.

Everything here is pure tensor math: no host imports, no CUDA.
"""

from __future__ import annotations

import copy
import os

import torch
from torch import Tensor

# Stored floats per Gaussian (position 3 + opacity 1 + scale 3 + rotation 4
# + c0 3) and per lobe (direction 3 + sharpness 1 + rgb 3).
BASE_FLOATS = 14
LOBE_FLOATS = 7

# Below this sharpness the exact compensation ratio loses precision; the
# series 1 - s + (2/3)s² agrees with it to O(s³).
_SMALL_S = 1e-3


def lobe_importance(sg_rgb: Tensor, sg_sharpness: Tensor) -> Tensor:
    """MEGS²'s range metric, per lobe.

    Args:
        sg_rgb: (..., L, 3) amplitudes in SH space.
        sg_sharpness: (..., L, 1) activated sharpness (>= 0).

    Returns:
        (..., L) importance; higher means the lobe carries more directional
        energy and should be kept first.
    """
    s = sg_sharpness.squeeze(-1)
    if s.shape != sg_rgb.shape[:-1]:
        raise ValueError(
            f"sharpness {tuple(sg_sharpness.shape)} does not match rgb "
            f"{tuple(sg_rgb.shape)}"
        )
    return sg_rgb.abs().mean(dim=-1) * (1.0 - torch.exp(-2.0 * s))


def lobes_to_keep(n_points: int, avg_lobes: float, num_lobes: int) -> int:
    """Round ``avg_lobes * n_points`` to an integer count and clamp."""
    if not 0.0 <= avg_lobes <= float(num_lobes):
        raise ValueError(
            f"avg_lobes must be in [0, {num_lobes}], got {avg_lobes}"
        )
    return int(max(0, min(num_lobes * n_points, avg_lobes * n_points + 0.5)))


def lobe_budget_mask(D: Tensor, avg_lobes: float) -> Tensor:
    """Bool mask keeping the top ``avg_lobes * G`` lobes per scene.

    Args:
        D: (G, L) for a single scene or (B, G, L) for a batch. The budget is
            applied independently per leading batch element.
        avg_lobes: target average lobes per Gaussian, in [0, L].

    Returns:
        Bool mask with the same shape as ``D``; exactly
        ``round(avg_lobes * G)`` entries are True per scene.
    """
    if D.dim() not in (2, 3):
        raise ValueError(f"expected (G, L) or (B, G, L), got {tuple(D.shape)}")
    L = D.shape[-1]
    if D.dim() == 2:
        flat = D.reshape(1, -1)
        single = True
    else:
        flat = D.reshape(D.shape[0], -1)
        single = False
    n_points = flat.shape[1] // L
    k = lobes_to_keep(n_points, avg_lobes, L)
    if k == 0:
        mask = torch.zeros_like(flat, dtype=torch.bool)
    elif k == flat.shape[1]:
        mask = torch.ones_like(flat, dtype=torch.bool)
    else:
        keep_idx = torch.topk(flat, k, dim=1).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(1, keep_idx, True)
    if single:
        return mask[0].reshape(D.shape)
    return mask.reshape_as(D)


def compensation_factor(sg_sharpness: Tensor) -> Tensor:
    """Sphere average of ``exp(s(cosθ - 1))``: (1 - e^{-2s}) / (2s).

    Uses the series expansion near s = 0 so zero-sharpness lobes (which are
    direction-constant) compensate exactly instead of producing 0/0.
    """
    s = sg_sharpness.clamp_min(0.0)
    exact = (1.0 - torch.exp(-2.0 * s)) / (2.0 * s).clamp_min(1e-12)
    series = 1.0 - s + (2.0 / 3.0) * s * s
    return torch.where(s < _SMALL_S, series, exact)


def compensate_pruned_lobes(
    rgb_base: Tensor,
    sg_rgb: Tensor,
    sg_sharpness: Tensor,
    keep_mask: Tensor,
) -> Tensor:
    """Add the removed lobes' sphere-average color back into c0.

    Args:
        rgb_base: (..., 3) diffuse term in SH space.
        sg_rgb: (..., L, 3) amplitudes in SH space.
        sg_sharpness: (..., L, 1) activated sharpness.
        keep_mask: (..., L) bool; True keeps a lobe.

    Returns:
        (..., 3) compensated c0. The per-view color average over the sphere
        is preserved exactly.
    """
    removed = (~keep_mask).to(sg_rgb.dtype).unsqueeze(-1)
    energy = sg_rgb * removed * compensation_factor(sg_sharpness)
    return rgb_base + energy.sum(dim=-2)


def lobe_stats(keep_mask: Tensor) -> dict:
    """Counts and distribution of kept lobes."""
    L = keep_mask.shape[-1]
    counts = keep_mask.sum(dim=-1)
    n_points = counts.numel()
    kept = int(counts.sum().item())
    hist = torch.bincount(counts.reshape(-1), minlength=L + 1).tolist()
    return {
        "n_points": n_points,
        "kept_lobes": kept,
        "total_lobes": n_points * L,
        "avg_lobes": kept / n_points if n_points else 0.0,
        "histogram": hist,  # index d = number of points keeping d lobes
    }


def storage_floats(n_points: int, kept_lobes: int) -> int:
    """Stored floats for a grouped export: base per point plus kept lobes."""
    return BASE_FLOATS * n_points + LOBE_FLOATS * kept_lobes


def budget_floats(n_points: int, avg_lobes: float) -> int:
    """Planned float budget for ``avg_lobes``; the κ axis in the 2D plane."""
    return BASE_FLOATS * n_points + LOBE_FLOATS * int(
        avg_lobes * n_points + 0.5
    )


def storage_report(g, keep_mask: Tensor | None = None) -> dict:
    """Storage accounting for an SG Gaussian container or a mask.

    ``g`` needs ``sg_rgb`` shaped (..., L, 3). When ``keep_mask`` is omitted
    every lobe is assumed kept (the unpruned model).
    """
    n_points = g.sg_rgb.shape[:-2].numel()
    L = g.sg_rgb.shape[-2]
    if keep_mask is None:
        kept = n_points * L
    else:
        kept = int(keep_mask.sum().item())
    floats = storage_floats(n_points, kept)
    return {
        "n_points": n_points,
        "kept_lobes": kept,
        "avg_lobes": kept / n_points if n_points else 0.0,
        "floats": floats,
        "bytes": floats * 4,
    }


_RUNTIME_AVG: float | None = None


def set_lobe_budget(avg_lobes: float | None) -> None:
    """Set (or clear with ``None``) the process-wide lobe budget.

    The decoder hook calls :func:`apply_runtime_lobe_budget` on every SG
    forward; leaving this unset keeps evaluation bit-identical to the
    unpruned model.
    """
    global _RUNTIME_AVG
    if avg_lobes is not None and avg_lobes < 0:
        raise ValueError(f"avg_lobes must be >= 0, got {avg_lobes}")
    _RUNTIME_AVG = None if avg_lobes is None else float(avg_lobes)


def get_lobe_budget() -> float | None:
    """Runtime budget from :func:`set_lobe_budget`, else ``SG_LOBE_AVG``."""
    if _RUNTIME_AVG is not None:
        return _RUNTIME_AVG
    env = os.environ.get("SG_LOBE_AVG")
    if env is None or env == "":
        return None
    return float(env)


def apply_runtime_lobe_budget(g):
    """Prune to the configured budget, or return ``g`` unchanged.

    Called from the decoder's SG branch. The pruning is a deep copy, so the
    caller's tensors (and any autograd graph) are untouched; at evaluation
    time that is the whole point, at training time the budget stays unset.
    """
    avg = get_lobe_budget()
    if avg is None:
        return g
    out, _ = prune_lobes(g, avg, inplace=False)
    return out


def prune_lobes(g, avg_lobes: float, inplace: bool = False):
    """Apply a global lobe budget to an SG Gaussian container.

    For each scene, ranks all ``G x L`` lobes by :func:`lobe_importance` and
    keeps the top ``round(avg_lobes * G)``. Removed lobes get zero amplitude
    and their sphere-average color is added to ``rgb_base``.

    Args:
        g: object with ``rgb_base`` (..., 3), ``sg_rgb`` (..., L, 3) and
            ``sg_sharpness`` (..., L, 1). ``SGGaussians`` satisfies this.
        avg_lobes: target average lobes per Gaussian, in [0, L].
        inplace: mutate ``g`` instead of returning a deep copy.

    Returns:
        ``(g_out, stats)``; stats comes from :func:`lobe_stats`.
    """
    if not all(hasattr(g, a) for a in ("rgb_base", "sg_rgb", "sg_sharpness")):
        raise TypeError("prune_lobes expects an SG Gaussian container")
    out = g if inplace else copy.deepcopy(g)
    with torch.no_grad():
        D = lobe_importance(out.sg_rgb, out.sg_sharpness)
        mask = lobe_budget_mask(D, avg_lobes)
        # Compensate before zeroing amplitudes: the removed energy is read
        # from the original lobes.
        out.rgb_base = compensate_pruned_lobes(
            out.rgb_base, out.sg_rgb, out.sg_sharpness, mask
        )
        out.sg_rgb = out.sg_rgb * mask.unsqueeze(-1).to(out.sg_rgb.dtype)
    return out, lobe_stats(mask)
