"""Post-hoc SH band truncation: the SH arm of the colour-budget plane.

The SG arm (``sg_prune.py``) drops lobes and folds their sphere average back
into c0. The SH arm drops whole frequency bands instead: keeping bands 0..d
stores ``3*(d+1)^2`` coefficients per point rather than the full set. No
compensation term is needed, and that asymmetry is what the comparison is
meant to expose: for every l >= 1 the SH basis integrates to zero over the
sphere, so truncation already preserves each point's mean colour -- only the
angular detail is lost. SG lobe removal has no such property, which is why it
needs the closed-form c0 correction.

Coefficients are sliced rather than zeroed, so dropped bands actually stop
occupying memory: the per-point float count becomes ``11 + 3*(d+1)^2`` (see
:func:`sh_storage_floats`). That is the number to quote when comparing
against the SG arm's ``14 + 7L``.

Runtime hook: ``SH_KEEP_DEGREE=<d>`` truncates at inference. Unset means a
no-op, so training and the unpruned evaluation are untouched.

Everything here is pure tensor math: no host imports, no CUDA.
"""

from __future__ import annotations

import dataclasses
import os
from math import isqrt

import torch
from torch import Tensor

# Stored floats per Gaussian outside the colour payload (position 3 +
# opacity 1 + scale 3 + rotation 4). Named apart from sg_prune.BASE_FLOATS,
# which is 14 because it already includes the SG arm's c0.
SH_BASE_FLOATS = 11


def sh_num_coeffs(degree: int) -> int:
    """Coefficients per colour channel for this degree: ``(degree + 1) ** 2``.

    Args:
        degree: SH degree, 0..4 in practice.

    Returns:
        Number of coefficients along the last axis (1, 4, 9, 16, 25, ...).
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    return (degree + 1) ** 2


def sh_storage_floats(degree: int) -> int:
    """Per-point float count at this degree.

    Args:
        degree: SH degree.

    Returns:
        ``11 + 3 * (degree + 1) ** 2``: 14 for degree 0, 23 / 38 / 59 / 86 for
        degrees 1..4. Compare with the SG arm's ``14 + 7L`` (14 / 21 / 28 / 35).
    """
    return SH_BASE_FLOATS + 3 * sh_num_coeffs(degree)


def lobe_equivalent(degree: int) -> float:
    """SG lobe budget carrying the same colour floats: ``3(d+1)^2 / 7``.

    Args:
        degree: SH degree.

    Returns:
        L such that ``7L == 3*(d+1)**2``; degree 1 maps to 1.71 lobes and
        degree 2 to 3.86, i.e. SH degree 2 costs more than SG's L=3.
    """
    return 3 * sh_num_coeffs(degree) / 7


def degree_of(harmonics: Tensor) -> int:
    """Infer the SH degree from the coefficient axis width.

    Args:
        harmonics: ``(..., 3, d_sh)`` tensor.

    Returns:
        ``d`` with ``d_sh == (d + 1) ** 2``.

    Raises:
        ValueError: if ``d_sh`` is not a perfect square of that form.
    """
    d_sh = harmonics.shape[-1]
    degree = isqrt(d_sh) - 1
    if d_sh < 1 or sh_num_coeffs(degree) != d_sh:
        raise ValueError(
            f"harmonic width {d_sh} is not (degree+1)^2; expected 1/4/9/16/25"
        )
    return degree


def truncate_sh(g, keep_degree: int):
    """Return a copy of ``g`` with every band above ``keep_degree`` dropped.

    The kept coefficients are a prefix slice of the original tensor: SH bands
    are stored in ascending order, so ``harmonics[..., :(keep+1)**2]`` is
    exactly bands 0..keep. Nothing is rescaled (see the module docstring).

    Args:
        g: Gaussians (or any dataclass holding ``harmonics``) with
            ``harmonics`` of shape ``(..., 3, d_sh)``.
        keep_degree: highest band to keep, ``0 <= keep_degree <= degree_of(g)``.

    Returns:
        ``g`` itself when nothing is dropped, otherwise a
        ``dataclasses.replace`` copy whose ``harmonics`` is narrower.
    """
    if not dataclasses.is_dataclass(g):
        raise TypeError(
            f"truncate_sh expects a Gaussians dataclass, got {type(g).__name__}"
        )
    degree = degree_of(g.harmonics)
    if not 0 <= keep_degree <= degree:
        raise ValueError(
            f"keep_degree must be in [0, {degree}], got {keep_degree}"
        )
    if keep_degree == degree:
        return g
    n = sh_num_coeffs(keep_degree)
    # contiguous() materialises the slice: without it the narrow tensor
    # would keep the full-width storage alive, contradicting the float count.
    return dataclasses.replace(g, harmonics=g.harmonics[..., :n].contiguous())


# ---------------------------------------------------------------------------
# Per-point degree allocation (the strongest SH baseline)
#
# Uniform truncation (:func:`truncate_sh`) is the weakest way to spend an SH
# budget: every point gets the same degree. The strongest post-hoc SH
# opponent gives each point its own degree, storing a coefficient prefix
# (bands are ordered, so degree k means bands 0..k). The allocation below
# searches two-level splits — n points stay at the full degree, the rest drop
# to a lower one — and keeps the split with the largest preserved band
# energy, which is the optimal allocation within that family and mirrors how
# mixed-degree storage is actually laid out. Coefficients above a point's
# degree are zeroed rather than sliced away (the tensor stays rectangular);
# since bands l >= 1 integrate to zero over the sphere, zeroing is
# numerically identical to removal.
# ---------------------------------------------------------------------------


def band_energy(harmonics: Tensor) -> Tensor:
    """Per-point L2 energy of each non-DC band.

    Args:
        harmonics: (..., 3, d_sh) coefficients.

    Returns:
        (..., degree) tensor; entry k is the energy of band k+1, i.e. the
        band a point only stores when its degree is >= k+1.
    """
    degree = degree_of(harmonics)
    return torch.stack(
        [
            harmonics[..., sh_num_coeffs(b - 1) : sh_num_coeffs(b)].norm(dim=(-1, -2))
            for b in range(1, degree + 1)
        ],
        dim=-1,
    )


def two_level_split(n_points: int, hi_c: int, lo_c: int, color_floats: float) -> int:
    """Points kept at the high level when the rest drop to the low one.

    Args:
        n_points: total number of points.
        hi_c: coefficients per channel at the high level.
        lo_c: coefficients per channel at the low level.
        color_floats: average colour floats per point (budget / n_points).

    Returns:
        Floor of the solution of ``n_hi * hi_c + (n - n_hi) * lo_c ==
        budget / 3``; any fractional remainder stays unspent (at scene scale
        the slack is < one upgrade and shrinking with G).
    """
    if hi_c == lo_c:
        return n_points
    total_coeffs = color_floats / 3.0 * n_points
    n_hi = (total_coeffs - lo_c * n_points) / (hi_c - lo_c)
    return int(min(max(n_hi, 0.0), float(n_points)))


def allocate_degrees(harmonics: Tensor, color_floats: float) -> Tensor:
    """Per-point SH degrees under an average colour-float budget.

    Args:
        harmonics: (..., 3, d_sh) coefficients.
        color_floats: target average floats per point spent on colour
            coefficients (3 for degree 0, up to 3 * (degree + 1) ** 2).

    Returns:
        (...) long tensor of per-point degrees.

    Raises:
        ValueError: if the budget is below the degree-0 floor (3 floats).
    """
    if color_floats < 3:
        raise ValueError(f"color_floats must be >= 3 (degree-0 floor), got {color_floats}")
    degree = degree_of(harmonics)
    shape = harmonics.shape[:-2]
    if color_floats >= 3 * sh_num_coeffs(degree):
        return torch.full(shape, degree, dtype=torch.long, device=harmonics.device)

    flat = harmonics.reshape(-1, *harmonics.shape[-2:])  # (G, 3, d_sh)
    G = flat.shape[0]
    energies = band_energy(flat)  # (G, degree)

    best_degrees: Tensor | None = None
    best_value = -1.0
    for low in range(0, degree):
        n_hi = two_level_split(G, sh_num_coeffs(degree), sh_num_coeffs(low), color_floats)
        # Marginal value of high vs low for each point: bands low+1..degree.
        marginal = energies[:, low:].sum(dim=-1)
        order = torch.argsort(marginal, descending=True)
        degrees = torch.full((G,), low, dtype=torch.long, device=flat.device)
        if n_hi > 0:
            degrees[order[:n_hi]] = degree
        # Preserved energy: everyone keeps bands 1..low, the chosen few also low+1..degree.
        value = energies[:, :low].sum().item() + marginal[order[:n_hi]].sum().item()
        if value > best_value:
            best_value, best_degrees = value, degrees
    assert best_degrees is not None
    return best_degrees.reshape(shape)


def allocation_floats(degrees: Tensor) -> float:
    """Average colour floats per point implied by a degree assignment.

    Args:
        degrees: (...) long tensor from :func:`allocate_degrees`.

    Returns:
        ``mean(3 * (degrees + 1) ** 2)`` — the number to quote next to the
        budget on the same x-axis as the SG arm's ``14 + 7L``.
    """
    return float((3 * (degrees + 1) ** 2).sum() / degrees.numel())


def apply_sh_allocation(g, color_floats: float):
    """Return ``(g, degrees)`` with coefficients above each point degree zeroed.

    The harmonics tensor keeps its width (rectangular, batch-friendly); the
    accounting in :func:`allocation_floats` counts only the stored prefix.

    Args:
        g: Gaussians dataclass with ``harmonics`` (..., 3, d_sh).
        color_floats: average colour-float budget per point.

    Returns:
        Tuple of the budgeted Gaussians copy and the (...) degree tensor.
    """
    degrees = allocate_degrees(g.harmonics, color_floats)
    d_sh = g.harmonics.shape[-1]
    keep = torch.arange(d_sh, device=g.harmonics.device) < (degrees + 1) ** 2
    h = g.harmonics * keep.unsqueeze(-2)
    return dataclasses.replace(g, harmonics=h.contiguous()), degrees


_RUNTIME_KEEP: int | None = None


def set_sh_keep_degree(degree: int | None) -> None:
    """Set (or clear with ``None``) the process-wide SH degree budget.

    The decoder hook calls :func:`apply_runtime_sh_budget` on every SH
    forward; leaving this unset keeps evaluation bit-identical to the
    unpruned model.
    """
    global _RUNTIME_KEEP
    if degree is not None and degree < 0:
        raise ValueError(f"keep_degree must be >= 0, got {degree}")
    _RUNTIME_KEEP = None if degree is None else int(degree)


def get_sh_keep_degree() -> int | None:
    """Runtime budget from :func:`set_sh_keep_degree`, else ``SH_KEEP_DEGREE``."""
    if _RUNTIME_KEEP is not None:
        return _RUNTIME_KEEP
    env = os.environ.get("SH_KEEP_DEGREE", "")
    if env == "":
        return None
    return int(env)


_RUNTIME_ALLOC: float | None = None


def set_sh_alloc(color_floats: float | None) -> None:
    """Set (or clear with ``None``) the per-point colour-float budget."""
    global _RUNTIME_ALLOC
    if color_floats is not None and color_floats < 3:
        raise ValueError(f"color_floats must be >= 3, got {color_floats}")
    _RUNTIME_ALLOC = None if color_floats is None else float(color_floats)


def get_sh_alloc() -> float | None:
    """Runtime allocation budget, else ``SH_ALLOC_COLOR_FLOATS``.

    Takes precedence over ``SH_KEEP_DEGREE`` when both are set: per-point
    allocation is the stronger axis.
    """
    if _RUNTIME_ALLOC is not None:
        return _RUNTIME_ALLOC
    env = os.environ.get("SH_ALLOC_COLOR_FLOATS", "")
    if env == "":
        return None
    return float(env)


def apply_runtime_sh_budget(g):
    """Truncate to the configured degree, or return ``g`` unchanged.

    Called from the decoder's SH branch. The result is a new dataclass holding
    a narrower, contiguous harmonics tensor, so the caller's tensors (and any
    autograd graph) are untouched.
    """
    alloc = get_sh_alloc()
    if alloc is not None:
        out, _ = apply_sh_allocation(g, alloc)
        return out
    keep = get_sh_keep_degree()
    if keep is None:
        return g
    return truncate_sh(g, keep)
