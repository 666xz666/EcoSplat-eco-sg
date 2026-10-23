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


def apply_runtime_sh_budget(g):
    """Truncate to the configured degree, or return ``g`` unchanged.

    Called from the decoder's SH branch. The result is a new dataclass holding
    a narrower, contiguous harmonics tensor, so the caller's tensors (and any
    autograd graph) are untouched.
    """
    keep = get_sh_keep_degree()
    if keep is None:
        return g
    return truncate_sh(g, keep)
