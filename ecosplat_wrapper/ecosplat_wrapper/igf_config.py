from dataclasses import dataclass


@dataclass
class IGFConfig:
    loss_weight: float = 0.1
    io_weight: float = 0.1
    acc_weight: float = 1.0
    plgc_decay_steps: int = 1000
    plgc_decay: float = 0.05
    plgc_floor: float = 0.05
    plgc_max: float = 0.95
    plgc_min0: float = 0.85
    zero_init_rate_embed: bool = True
    inference_rho: float = 0.4  # fixed protect_rate used at eval/inference (paper κ_i preset)
    # Budget mode: read protect_rate as the fraction of the FULL float payload
    # (35 floats per point at 3 lobes) instead of the fraction of points, and
    # let the fused allocator split that payload between points and lobes.
    # Off = stock PLGC (top-k points, every point keeps all lobes).
    budget_mode: bool = False
