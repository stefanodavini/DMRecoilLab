"""
Abstract base classes and common utilities for isotropic dark-matter
halo velocity distributions.

This module defines the IsotropicSpeedDistribution interface,
together with generic functionality shared by all halo models,
including:
- velocity modulus distribution f(v)
- speed distributions g(v) = 4π v² f(v),
- halo integrals η(vmin),
- isotropic boosts,
- tabulated export,
- normalization

The normalization is such that
∫ 4π v² f(v) dv = 1,
equivalent to
∫ g(v) dv = 1.

Concrete halo models should derive from
IsotropicSpeedyDistribution and implement velocity_pdf()
and maximum_speed.
"""

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

from src.utils.constants import VEARTH_kmps
from src.utils.tabulated_data import TabulatedData

class IsotropicSpeedDistribution(ABC):
    """
    Isotropic speed distribution.

    Implementations must provide velocity_pdf(v),
    normalized such that
    ∫ 4π v² f(v) dv = 1.

    The speed distribution is derived as
    g(v) = 4π v² f(v).

    The halo integral is defined as
    eta(vmin) = ∫_{v>vmin} g(v)/v dv.
    """

    @abstractmethod
    def velocity_pdf(self, v: ArrayLike) -> NDArray[np.float64]:
        """
        Isotropic velocity modulus distribution f(v).

        For an isotropic halo:
        ∫ 4π v² f(v) dv = 1
        where v is the velocity modulus.

        Returns the normalized f(v).
        """

    def speed_pdf(self, v: ArrayLike) -> NDArray[np.float64]:
        """
        Speed distribution g(v).

        Returns g(v) defined as
        g(v) = 4 pi v^2 f(v)

        The normalization is such that
        integral from 0 to infinity g(v) dv = 1.
        """
        return 4 * np.pi * np.asarray(v)**2 * self.velocity_pdf(v)

    def eta(self, vmin: ArrayLike) -> NDArray[np.float64]:
        """
        Halo integral

        eta(vmin) = ∫_{v>vmin} g(v)/v dv.

        Parameters
        ----------
        vmin : ArrayLike
            Minimum speed.

        Returns
        -------
        ndarray
        """
        v, eta = self._eta_lookup_table
        return np.interp(np.asarray(vmin), v, eta, left=eta[0], right=0.0)

    def boost(self, vboost: float = VEARTH_kmps):
        """
        Return the isotropic (angular-averaged) distribution
        seen by an observer moving at speed vboost.

        The resulting distribution corresponds to the
        angular average of

        f(|v + vboost|) = f(sqrt(v²+vb²+2vvb cosθ))

        over cos(theta).
        """
        if vboost == 0:
            return self
        return _BoostedDistribution(self, vboost)

    def to_file(self,
                filename: str,
                n_points: int = 1000):
        """
        Save the speed distribution to a two-column text file.

        The first column contains the speed v and the second column
        contains the corresponding values of f(v).

        Parameters
        ----------
        filename: str
            Output filename.

        n_points: int
            Number of sample points used to tabulate f(v).
        """
        v = np.linspace(0.0, self.maximum_speed, n_points)
        data = TabulatedData(x=v,
                             y=self.velocity_pdf(v),
                             x_name="v", y_name="f(v)")

        data.to_file(filename=filename)

    def normalization(self, v: ArrayLike) -> float:
        """
        Compute the normalization of the speed distribution.

        Parameters
        ----------
        v : ArrayLike
            Integration grid.

        Returns
        -------
        float
            Value of ∫ g(v) dv
            where g(v) = 4π v² f(v).

        A properly normalized distribution returns unity.
        """
        v = np.asarray(v)
        return float(np.trapezoid(self.speed_pdf(v), v))

    @property
    def maximum_speed(self) -> float:
        """
        Maximum speed supported by the distribution.

        Returns
        -------
        float
            Largest speed for which the distribution can be nonzero.
        """
        raise NotImplementedError

    @cached_property
    def _eta_lookup_table(self, grid_size: int =10000):
        """
        Compute η(vmin) = ∫_vmin^∞ g(v)/v dv
        by integrating from vmax down to 0 on a reversed grid.
        """

        v = np.linspace(0.0, self.maximum_speed, grid_size)
        mask = v > 0.0

        integrand = np.zeros_like(v)
        integrand[mask] = self.speed_pdf(v[mask]) / v[mask]

        v_rev = v[::-1]
        f_rev = integrand[::-1]

        eta_rev = cumulative_trapezoid(f_rev, -v_rev, initial=0.0)
        eta = eta_rev[::-1]

        return v, eta


class _BoostedDistribution(IsotropicSpeedDistribution):
    """
    An isotropic speed distribution obtained by angular averaging
    of a boosted parent distribution.
    
    The averaged velocity PDF is precomputed on a grid during construction 
    and subsequently evaluated via interpolation
    """

    def __init__(
        self,
        parent: IsotropicSpeedDistribution,
        vboost: float,
        ngrid: int = 10000,
    ):
        self._parent = parent
        self._vboost = float(vboost)

        self._vgrid = np.linspace(0.0, self.maximum_speed, ngrid)

        self._pdf_grid = self._compute_velocity_pdf()

    def _compute_velocity_pdf(self) -> NDArray:
        """
        Angular average of

        f_lab(v) = (1/2) ∫ dcosθ f_parent(sqrt(v²+vb²+2vvb cosθ))

        Using the change of variable w = v²+vb²+2vvb cosθ,
        the integral can be rewritten analytically
        in terms of the primitive

        F(x)=∫ w f(w) dw

        giving

        [F(v+vb)-F(|v-vb|)]/(2 v vb)
        """
        w = np.linspace(0.0, self.maximum_speed, len(self._vgrid))

        fw = self._parent.velocity_pdf(w)

        # Primitive:
        # F(x) = /int_0^x w f(w) dw
        primitive = cumulative_trapezoid(w * fw, w, initial=0.0)

        def F(x):
            return np.interp(x, w,
            primitive, left=0.0,
            right=primitive[-1])

        vb = self._vboost
        v = self._vgrid

        lower = np.abs(v - vb)
        upper = v + vb

        numerator = F(upper) - F(lower)

        mask = v > 0
        pdf = np.zeros_like(v)
        pdf[mask] = numerator[mask] / (2.0 * v[mask] * vb)

        # v->0 limit of the boost formula
        # lim_{v→0} f_lab(v) = f_parent(vboost)
        pdf[0] = self._parent.velocity_pdf(vb)

        return pdf

    def velocity_pdf(self, v: ArrayLike) -> NDArray:
        """
        Interpolated velocity PDF of the boosted distribution.

        Parameters
        ----------
        v : ArrayLike
            Speed at which the distribution is evaluated.

        Returns
        -------
        ndarray
            Boosted velocity distribution evaluated at v.
        """
        return np.interp(
            np.asarray(v),
            self._vgrid,
            self._pdf_grid,
            left=self._pdf_grid[0],
            right=0.0)

    @property
    def maximum_speed(self) -> float:
        """
        Maximum speed of the boosted distribution.
        """
        return self._parent.maximum_speed + self._vboost
