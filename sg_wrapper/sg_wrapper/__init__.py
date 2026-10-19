from .sg_config import SGAdapterCfg
from .sg_adapter import (
    UnifiedSGGaussianAdapter,
    flatten_sg_gaussians,
    _get_sg_gaussians_class,
)
from .sg_colors import C0, compute_sg_colors
from .stage1_loader import init_encoder_from_stage1

__all__ = [
    "SGAdapterCfg",
    "SGGaussians",
    "UnifiedSGGaussianAdapter",
    "flatten_sg_gaussians",
    "compute_sg_colors",
    "init_encoder_from_stage1",
    "C0",
]


def __getattr__(name):
    # SGGaussians subclasses the host's src.model.types.Gaussians, so
    # creating it imports the host repo; resolve it lazily to keep importing
    # this package torch-only (the color-math unit test runs without src).
    if name == "SGGaussians":
        return _get_sg_gaussians_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
