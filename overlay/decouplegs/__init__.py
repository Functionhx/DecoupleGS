"""Open reimplementation of DecoupleGS on top of HUGSIM.

The package is deliberately usable without the HUGSIM CUDA dependencies.  Core
geometry, compression, registration, relighting, and traffic behavior can be
tested on CPU; :mod:`decouplegs.hugsim` provides the optional renderer bridge.
"""

from .transforms import RealSHRotator, transform_gaussians
from .types import GaussianSet

__all__ = [
    "GaussianSet",
    "RealSHRotator",
    "transform_gaussians",
]

__version__ = "0.1.0"
