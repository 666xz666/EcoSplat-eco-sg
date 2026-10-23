from .sg_config import SGAdapterCfg
from .sg_adapter import (
    UnifiedSGGaussianAdapter,
    flatten_sg_gaussians,
    _get_sg_gaussians_class,
)
from .sg_colors import C0, compute_sg_colors
from .sg_prune import (
    BASE_FLOATS,
    LOBE_FLOATS,
    PackedLobes,
    apply_runtime_lobe_budget,
    avg_lobes_for_bytes,
    budget_floats,
    compensate_pruned_lobes,
    compensation_factor,
    compute_packed_sg_colors,
    get_lobe_budget,
    get_packed,
    lobe_budget_mask,
    lobe_importance,
    lobe_stats,
    lobes_for_bytes,
    lobes_to_keep,
    pack_into,
    pack_lobes,
    pack_lobes_to_bytes,
    packed_bytes,
    packed_floats,
    prune_lobes,
    set_lobe_budget,
    set_packed,
    storage_floats,
    storage_report,
)
from .stage1_loader import init_encoder_from_stage1

__all__ = [
    "SGAdapterCfg",
    "SGGaussians",
    "UnifiedSGGaussianAdapter",
    "flatten_sg_gaussians",
    "compute_sg_colors",
    "init_encoder_from_stage1",
    "C0",
    "BASE_FLOATS",
    "LOBE_FLOATS",
    "PackedLobes",
    "apply_runtime_lobe_budget",
    "avg_lobes_for_bytes",
    "budget_floats",
    "compensate_pruned_lobes",
    "compensation_factor",
    "compute_packed_sg_colors",
    "get_lobe_budget",
    "get_packed",
    "lobe_budget_mask",
    "lobe_importance",
    "lobe_stats",
    "lobes_for_bytes",
    "lobes_to_keep",
    "pack_into",
    "pack_lobes",
    "pack_lobes_to_bytes",
    "packed_bytes",
    "packed_floats",
    "prune_lobes",
    "set_lobe_budget",
    "set_packed",
    "storage_floats",
    "storage_report",
]


def __getattr__(name):
    # SGGaussians subclasses the host's src.model.types.Gaussians, so
    # creating it imports the host repo; resolve it lazily to keep importing
    # this package torch-only (the color-math unit test runs without src).
    if name == "SGGaussians":
        return _get_sg_gaussians_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
