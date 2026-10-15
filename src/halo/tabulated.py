import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Self

from src.halo.base_halo_distribution import IsotropicVelocityDistribution

class TabulatedVelocityDistribution(IsotropicVelocityDistribution):
    """
    Isotropic speed distribution defined by tabulated values.

    The distribution is specified by arrays of speeds and corresponding
    values of f(v). Values between tabulated points are obtained by
    linear interpolation. Values outside the tabulated range are zero.

    Parameters
    ----------
    v : ArrayLike
        Monotonically increasing speed grid.
    f : ArrayLike
        Tabulated values of the speed distribution.
    """

    def __init__(self,
                 v: ArrayLike,
                 f: ArrayLike):
        
        self._v = np.asarray(v, dtype=float)
        self._f = np.asarray(f, dtype=float)

        if self._v.ndim != 1:
            raise ValueError("v must be a one-dimensional array.")

        if self._f.ndim != 1:
            raise ValueError("f must be a one-dimensional array.")

        if len(self._v) != len(self._f):
            raise ValueError("v and f must have the same length.")

        if len(self._v) < 2:
            raise ValueError("At least two tabulation points are required.")

        if np.any(~np.isfinite(self._v)):
            raise ValueError("v must contain only finite values.")

        if np.any(~np.isfinite(self._f)):
            raise ValueError("f must contain only finite values.")

        if np.any(np.diff(self._v) <= 0):
            raise ValueError("v must be strictly increasing.")

        if np.any(self._f < 0):
            raise ValueError("f must be non-negative.")

    def velocity_pdf(self, v: ArrayLike) -> NDArray:
        """
        Evaluate the tabulated velocity distribution.

        Parameters
        ----------
        v: ArrayLike
            Speed(s) at which the distribution is evaluated.

        Returns
        -------
        ndarray
            Interpolated values of f(v).
        """
        v = np.asarray(v, dtype=float)

        return np.interp(v, self._v, self._f,
                         left=0.0, right=0.0)

    @property
    def maximum_speed(self) -> float:
        """
        Largest speed for which the tabulated distribution is nonzero.
        """
        mask = self._f > 0.0

        if not np.any(mask):
            return 0.0

        return float(self._v[mask][-1])

    @property
    def normalization(self) -> float:
        """
        Numerical normalization of the tabulated distribution.
        """
        return float(np.trapezoid(self._f, self._v))

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Construct a velocity distribution from a file produced by
        ``to_file``.

        Parameters
        ----------
        filename: str
            Input filename.

        Returns
        -------
        TabulatedVelocityDistribution
        """
        if not filename:
            raise ValueError("Invalid filename")

        data = np.loadtxt(filename)

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Input file must contain two columns: v and f(v).")

        return cls(v=data[:, 0], f=data[:, 1])