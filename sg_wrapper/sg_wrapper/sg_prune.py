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
from dataclasses import dataclass

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


@dataclass
class PackedLobes:
    """Rectangular-free lobe storage for one scene.

    ``directions/sharpness/rgb`` hold only the kept lobes, flattened to
    ``[K, ...]``; ``point_index[k]`` says which Gaussian owns lobe ``k``.
    ``counts[g]`` is the per-point kept count (0..L). Compared with the padded
    ``[G, L, 7]`` layout this uses ``K`` rows instead of ``G*L``; at L̄=1 it is
    3x smaller.
    """

    rgb_base: Tensor  # [G, 3], compensated diffuse term
    directions: Tensor  # [K, 3]
    sharpness: Tensor  # [K, 1]
    rgb: Tensor  # [K, 3]
    point_index: Tensor  # [K] long
    counts: Tensor  # [G] long
    n_points: int
    n_lobes: int
    num_lobes: int


def pack_lobes(g, avg_lobes: float | None = None, total_lobes: int | None = None):
    """Rank, compensate and pack one scene's lobes.

    Exactly the same selection as :func:`prune_lobes`, but the result is a
    :class:`PackedLobes` whose lobe tensors have one row per kept lobe instead
    of ``G x L`` rows. Geometry is not stored here (it stays on the Gaussian
    container); only ``c0`` and the lobe payload are needed for color.

    Args:
        g: object with ``rgb_base`` [G,3], ``sg_directions`` [G,L,3],
            ``sg_sharpness`` [G,L,1] and ``sg_rgb`` [G,L,3]. Batched
            containers are not supported (evaluation uses one scene).
        avg_lobes: target average lobes per Gaussian; ``total_lobes`` may be
            given instead for an exact byte budget.
        total_lobes: exact number of lobes to keep (0..G*L).
    """
    if not all(
        hasattr(g, a)
        for a in ("rgb_base", "sg_directions", "sg_sharpness", "sg_rgb")
    ):
        raise TypeError("pack_lobes expects an SG Gaussian container")
    sg_rgb, sg_dirs, sg_sharp, rgb_base_in = (
        g.sg_rgb,
        g.sg_directions,
        g.sg_sharpness,
        g.rgb_base,
    )
    if sg_rgb.dim() == 4:
        if sg_rgb.shape[0] != 1:
            raise ValueError(
                "pack_lobes supports a leading batch of 1; got "
                f"{tuple(sg_rgb.shape)}"
            )
        sg_rgb, sg_dirs, sg_sharp, rgb_base_in = (
            sg_rgb[0],
            sg_dirs[0],
            sg_sharp[0],
            rgb_base_in[0],
        )
    elif sg_rgb.dim() != 3:
        raise ValueError(
            "pack_lobes expects [G,L,3] or [1,G,L,3]; got "
            f"{tuple(sg_rgb.shape)}"
        )
    n_points, num_lobes, _ = sg_rgb.shape
    if total_lobes is None:
        if avg_lobes is None:
            avg_lobes = float(num_lobes)
        total_lobes = lobes_to_keep(n_points, avg_lobes, num_lobes)
    total_lobes = int(max(0, min(n_points * num_lobes, total_lobes)))
    avg = total_lobes / n_points if n_points else 0.0

    with torch.no_grad():
        D = lobe_importance(sg_rgb, sg_sharp)
        mask = lobe_budget_mask(D, avg)
        rgb_base = compensate_pruned_lobes(
            rgb_base_in, sg_rgb, sg_sharp, mask
        )
        idx = mask.nonzero(as_tuple=False)  # [K, 2] (point, lobe)
        point_index = idx[:, 0].to(torch.long)
        lobe_index = idx[:, 1].to(torch.long)
        directions = sg_dirs[point_index, lobe_index]
        sharpness = sg_sharp[point_index, lobe_index]
        rgb = sg_rgb[point_index, lobe_index]
        counts = mask.sum(dim=-1).to(torch.long)
    return PackedLobes(
        rgb_base=rgb_base,
        directions=directions,
        sharpness=sharpness,
        rgb=rgb,
        point_index=point_index,
        counts=counts,
        n_points=n_points,
        n_lobes=int(point_index.numel()),
        num_lobes=num_lobes,
    )


def compute_packed_sg_colors(
    means: Tensor,
    rgb_base: Tensor,
    directions: Tensor,
    sharpness: Tensor,
    rgb: Tensor,
    point_index: Tensor,
    extrinsics: Tensor,
    clamp_colors: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """Evaluate colors from a packed lobe set.

    Same math as :func:`compute_sg_colors`, but the lobes are a flat list with
    ``point_index`` mapping each kept lobe to its Gaussian; the per-lobe
    contribution is accumulated with ``index_add_``. Only the kept lobes are
    stored, which is where the VRAM reduction comes from.

    Args:
        means: [B, G, 3]
        rgb_base: [B, G, 3] in SH space (already compensated)
        directions: [K, 3]
        sharpness: [K, 1]
        rgb: [K, 3]
        point_index: [K] long, values in [0, G)
        extrinsics: [B, 4, 4]
        clamp_colors: clamp to [0, 1] like MEGS²

    Returns:
        [B, G, 3] RGB colors ready for gsplat's colors_precomp.
    """
    if means.dim() != 3:
        raise ValueError(f"means must be [B,G,3], got {tuple(means.shape)}")
    b, g, _ = means.shape
    k = directions.shape[0]
    if point_index.numel() != k or rgb.shape[0] != k:
        raise ValueError("packed lobe tensors disagree on K")
    dirs = torch.nn.functional.normalize(directions, dim=-1, eps=eps)
    out = []
    cam = extrinsics[:, :3, 3]
    for bi in range(b):
        view = cam[bi][None, :] - means[bi]  # [G, 3]
        view = view / (view.norm(dim=-1, keepdim=True) + eps)
        colors = rgb_base[bi].clone()
        if k:
            view_l = view[point_index]  # [K, 3]
            cos = (dirs * view_l).sum(dim=-1, keepdim=True)  # [K, 1]
            weighted = rgb * torch.exp(sharpness * (cos - 1.0))  # [K, 3]
            colors.index_add_(0, point_index, weighted)
        out.append(colors)
    colors = torch.stack(out, dim=0)
    colors = colors * 0.28209479177387814 + 0.5  # SH2RGB
    if clamp_colors:
        colors = colors.clamp(0.0, 1.0)
    return colors


def packed_floats(packed: PackedLobes) -> int:
    """Stored floats for a packed scene: base per point + kept lobes."""
    return storage_floats(packed.n_points, packed.n_lobes)


def packed_bytes(packed: PackedLobes) -> int:
    return packed_floats(packed) * 4


def lobes_for_bytes(n_points: int, target_bytes: int, num_lobes: int) -> int:
    """Exact lobe count that fits ``target_bytes`` (clamped to [0, G*L])."""
    max_lobes = n_points * num_lobes
    lobes = int((target_bytes / 4 - BASE_FLOATS * n_points) / LOBE_FLOATS + 0.5)
    return max(0, min(max_lobes, lobes))


def avg_lobes_for_bytes(n_points: int, target_bytes: int, num_lobes: int) -> float:
    """Average lobes per Gaussian for a byte budget."""
    return lobes_for_bytes(n_points, target_bytes, num_lobes) / n_points


def pack_lobes_to_bytes(g, target_bytes: int):
    """Pack a scene at an exact byte budget (base floats + kept lobes)."""
    shape = g.sg_rgb.shape
    n_points, num_lobes = shape[-3], shape[-2]
    total = lobes_for_bytes(n_points, target_bytes, num_lobes)
    return pack_lobes(g, total_lobes=total)


def pack_into(g, avg_lobes: float):
    """Return a copy of ``g`` whose color payload is packed, geometry intact.

    The padded ``sg_directions/sg_sharpness/sg_rgb`` tensors are dropped (set
    to ``None``) so the old rectangular payload can be freed; the decoder reads
    the ``sg_packed_*`` fields instead. This is what makes the budget affect
    actual VRAM rather than only accounting.
    """
    squeeze = g.sg_rgb.dim() == 4
    packed = pack_lobes(g, avg_lobes=avg_lobes)
    out = copy.deepcopy(g)
    out.rgb_base = packed.rgb_base.unsqueeze(0) if squeeze else packed.rgb_base
    out.sg_directions = None
    out.sg_sharpness = None
    out.sg_rgb = None
    out.sg_packed_directions = packed.directions
    out.sg_packed_sharpness = packed.sharpness
    out.sg_packed_rgb = packed.rgb
    out.sg_packed_index = packed.point_index
    out.sg_lobe_counts = packed.counts
    out.sg_num_lobes = packed.num_lobes
    return out, packed


_RUNTIME_PACKED = False


def set_packed(enabled: bool | None = None) -> None:
    """Enable/disable packed color storage for the runtime decoder hook."""
    global _RUNTIME_PACKED
    if enabled is None:
        _RUNTIME_PACKED = os.environ.get("SG_PACKED", "0").lower() in (
            "1",
            "true",
            "yes",
        )
    else:
        _RUNTIME_PACKED = bool(enabled)


def get_packed() -> bool:
    return _RUNTIME_PACKED or os.environ.get("SG_PACKED", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def apply_runtime_lobe_budget(g):
    """Prune (and optionally pack) to the configured budget, or return ``g``.

    Unset budget -> no-op. ``SG_LOBE_AVG`` sets an average-lobe budget;
    ``SG_BUDGET_MB`` sets a byte budget for the color payload
    (``14*N + 7*L_total`` floats) and is converted per scene. ``SG_PACKED=1``
    additionally drops the padded lobe tensors in favor of a packed layout,
    which is what actually reduces VRAM.
    """
    if getattr(g, "sg_packed_directions", None) is not None:
        return g
    avg = get_lobe_budget()
    if avg is None:
        budget_mb = os.environ.get("SG_BUDGET_MB", "")
        if budget_mb:
            if g.sg_rgb.dim() != 3:
                raise ValueError(
                    "SG_BUDGET_MB supports one scene per forward; got "
                    f"{tuple(g.sg_rgb.shape)}"
                )
            n_points, num_lobes, _ = g.sg_rgb.shape
            avg = avg_lobes_for_bytes(
                n_points, int(float(budget_mb) * 1024 * 1024), num_lobes
            )
    if avg is None:
        return g
    if get_packed():
        out, _ = pack_into(g, avg)
        return out
    out, _ = prune_lobes(g, avg, inplace=False)
    return out
