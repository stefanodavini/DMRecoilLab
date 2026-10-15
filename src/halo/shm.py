from dataclasses import dataclass
from src.halo.base_halo_distribution import IsotropicVelocityDistribution
from src.utils.constants import SHM_V0_kmps, SHM_VESC_kmps

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import maxwell


@dataclass
class StandardHaloModel(IsotropicVelocityDistribution):
    v0: float = SHM_V0_kmps
    vesc: float = SHM_VESC_kmps
    """
    Standard Halo Model (SHM).

    Truncated Maxwell-Boltzmann speed distribution
    commonly used in WIMP direct-detection studies.

    The Galactic-frame velocity PDF is
    f(v) ∝ exp(-(v/v0)^2)
    for v < vesc
    and vanishes above the escape speed.

    The normalization is chosen so that
    ∫ 4π v² f(v) dv = 1.

    Parameters
    ----------
    v0 :
        Maxwellian speed parameter (km/s).
    v_esc :
        Galactic escape speed (km/s).
    """
    def __post_init__(self):
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")

        if self.vesc <= 0:
            raise ValueError("vesc must be positive")

        if self.vesc <= self.v0:
            raise ValueError("vesc <= v0 is unusual for a Standard Halo Model")

        self._norm = (np.pi**1.5 * self.v0**3 
                      * maxwell.cdf(self.vesc, scale=self.v0/np.sqrt(2)))

    def velocity_pdf(self, v: ArrayLike) -> NDArray[np.float64]:
        """
        Velocity distribution f(v) for the Standard Halo Model.

        f(v) = N exp(-(v/v0)^2) for v < vesc
             = 0                for v >= vesc

        normalized such that
        ∫ 4π v² f(v) dv = 1
        """
        v = np.asarray(v)
        return np.exp(- (v / self.v0)**2) * (v < self.vesc) / self._norm

    def eta(self, vmin: ArrayLike) -> NDArray[np.float64]:
        """
        Halo integral for the Standard Halo Model.

        eta(vmin) = ∫_{v>vmin} g(v)/v dv

        The analytical computation is:

        η(vmin) = (2πv0²/N) [ exp(-(vmin/v0)²) - exp(-(vesc/v0)²)]
        for vmin < vesc

        Parameters
        ----------
        vmin : array_like
            Minimum speed.

        Returns
        -------
        ndarray
        """

        x = np.asarray(vmin) / self.v0
        eesc = np.exp(-(self.vesc/self.v0)**2)

        return np.where(vmin < self.vesc,
            (2* np.pi * self.v0**2 / self._norm) * (np.exp(-x**2) - eesc),
            0.0)

    @property
    def maximum_speed(self) -> float:
        return self.vesc