"""
Mono-speed halo distribution.

This module provides a numerical approximation to a mono-energetic
dark-matter population.
The distribution is represented as a three-node tabulated speed
distribution centered at a prescribed speed.

The speed probability density is constructed from the nodes

    [vc - w/2, vc, vc + w/2]

with tabulated values

    [0, 1, 0].

After normalization by TabulatedSpeedDistribution, this produces
a compact triangular approximation to a Dirac delta function in
speed space.

The parameter ``full_width`` controls the numerical width of the
approximation and does not represent a physical velocity
dispersion.
"""

import numpy as np

from src.halo.tabulated_halo import TabulatedSpeedDistribution


class MonoSpeedDistribution(TabulatedSpeedDistribution):
    """
    Numerical approximation to a mono-speed population.

    A three-node tabulated distribution is constructed around the
    central speed ``vc`` using [vc - w/2, vc, vc + w/2]
    and tabulated values [0, 1, 0].

    After normalization, the resulting distribution is a compact
    triangular approximation to a speed-space Dirac delta function.

    Parameters
    ----------
    vc : float
        Central speed in km/s.

    full_width : float
        Numerical width of the delta approximation in km/s.
        Smaller values produce a narrower approximation but may
        require finer integration grids.
    """
    def __init__(self,
                 vc: float,
                 full_width: float):

        if not np.isfinite(vc) or vc <= 0:
            raise ValueError("Input vc must be finite and positive")

        if not np.isfinite(full_width) or full_width <= 0:
            raise ValueError("Input full_width must be finite and positive")

        half_width = full_width / 2

        if (vc < half_width):
            raise ValueError("Distribution width extends below zero")

        v = np.array([vc - half_width, vc, vc + half_width])
        f = np.array([0.0, 1.0, 0.0])

        super().__init__(v=v, f=f, normalize=True)
