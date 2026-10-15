from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

from src.utils.constants import VEARTH_kmps

class IsotropicVelocityDistribution(ABC):
    """
    Isotropic velocity distribution.

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
        Isotropic three-dimensional velocity distribution f(v).

        For an isotropic halo:
        ∫ 4π v² f(v) dv = 1
        where v is the velocity modulus.

        Returns the normalized f(v).
        """

    def speed_pdf(self, v: ArrayLike) -> NDArray[np.float64]:
        """
        Speed distribution in the galactic frame.

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
        vmin : array_like
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
        Save the velocity distribution to a tabulated file.

        Parameters
        ----------
        filename: str
            Output filename.

        n_points: int
            Number of sample points used to tabulate f(v).
        """
        if not filename:
            raise ValueError("Invalid filename")

        v = np.linspace(0.0, self.maximum_speed, n_points)

        data = np.column_stack([v, self.velocity_pdf(v)])

        np.savetxt(filename, data, header="v f(v)")

    @property
    def maximum_speed(self) -> float:
        """
        Internal helper returning the maximum
        physically allowed speed supported by
        the distribution.
        """
        raise NotImplementedError

    _eta_grid_size = 10000

    @cached_property
    def _eta_lookup_table(self):
        """
        Compute η(vmin) = ∫_vmin^∞ g(v)/v dv
        by integrating from vmax down to 0 on a reversed grid.
        """

        v = np.linspace(0.0, self.maximum_speed, self._eta_grid_size)
        mask = v > 0.0

        integrand = np.zeros_like(v)
        integrand[mask] = self.speed_pdf(v[mask]) / v[mask]

        v_rev = v[::-1]
        f_rev = integrand[::-1]

        eta_rev = cumulative_trapezoid(f_rev, -v_rev, initial=0.0)
        eta = eta_rev[::-1]

        return v, eta


class _BoostedDistribution(IsotropicVelocityDistribution):
    """
    An isotropic distribution obtained by angular averaging
    of a boosted parent distribution.
    
    The averaged velocity PDF is precomputed on a grid during construction 
    and subsequently evaluated via interpolation
    """

    def __init__(
        self,
        parent: IsotropicVelocityDistribution,
        vboost: float,
        ngrid: int = 10000,
    ):
        self._parent = parent
        self._vboost = float(vboost)

        self._vgrid = np.linspace(0.0, self.maximum_speed, ngrid)

        self._pdf_grid = self._compute_velocity_pdf()


    def _compute_velocity_pdf(self):
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

    def velocity_pdf(self, v):
        return np.interp(
            np.asarray(v),
            self._vgrid,
            self._pdf_grid,
            left=self._pdf_grid[0],
            right=0.0)

    @property
    def maximum_speed(self) -> float:
        return self._parent.maximum_speed + self._vboost