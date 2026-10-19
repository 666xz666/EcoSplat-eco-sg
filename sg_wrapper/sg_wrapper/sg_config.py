"""Config for the spherical-Gaussian color parametrization."""

from dataclasses import dataclass


@dataclass
class SGAdapterCfg:
    """Parameters of the SG color head.

    num_lobes: lobes per Gaussian (MEGS² upper bound is 3).
    rgb_scale: fixed multiplier on predicted lobe amplitudes. The feed-forward
        analog of MEGS²'s `rgb * 0.1` initialization: lobes start small and
        grow if the loss asks for view dependence.
    sharpness_scale: fixed multiplier on |predicted sharpness| before the
        exponential. 1.0 keeps MEGS²'s scale.
    clamp_colors: clamp evaluated colors to [0, 1] (MEGS² does; the SH path
        does not — keep False for a fair comparison).
    """

    num_lobes: int = 3
    rgb_scale: float = 0.1
    sharpness_scale: float = 1.0
    clamp_colors: bool = False
