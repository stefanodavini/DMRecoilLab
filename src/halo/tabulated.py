"""
Tabulated halo speed distributions.

This module provides TabulatedSpeedDistribution, allowing
isotropic halo distributions to be constructed from tabulated
values supplied either directly as arrays or loaded from
external text files.

The tabulated quantity is the velocity-modulus distribution
f(v), normalized according to

    ∫ 4π v² f(v) dv = 1.

Input distributions may optionally be renormalized during
construction.

Values between tabulated points are obtained by linear
interpolation, while values outside the tabulated range
are set to zero.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Self

from src.halo.base_halo_distribution import IsotropicSpeedDistribution

class TabulatedSpeedDistribution(IsotropicSpeedDistribution):
    """
    Isotropic speed distribution defined by tabulated values.

    The distribution is specified by arrays of speeds and corresponding
    values of the velocity-modulus distribution f(v).

    The physical normalization convention is
    ∫ 4π v² f(v) dv = 1,
    equivalently
    ∫ g(v) dv = 1,
    where g(v) = 4π v² f(v) is the speed distribution.

    Values between tabulated points are obtained by linear interpolation.
    Values outside the tabulated range are taken to be zero.

    Parameters
    ----------
    v : ArrayLike
        Monotonically increasing speed grid.
    f : ArrayLike
        Tabulated values of the speed distribution.
    """

    def __init__(self,
                 v: ArrayLike,
                 f: ArrayLike,
                 normalize: bool =True):
        
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

        self._validate_and_normalize(normalize=normalize)

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

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Construct a tabulated distribution from a text file produced
        by to_file().

        The file must contain exactly two columns corresponding to
        speed and velocity-distribution values.

        Parameters
        ----------
        filename: str
            Input filename.

        Returns
        -------
        TabulatedSpeedDistribution
        """
        if not filename:
            raise ValueError("Invalid filename")

        data = np.loadtxt(filename)

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Input file must contain two columns: v and f(v).")

        return cls(v=data[:, 0], f=data[:, 1])

    def _validate_and_normalize(self, normalize: bool = True) -> None:
        """
        Validate and optionally normalize the speed distribution.

        Parameters
        ----------
        normalize : bool, optional
            If True, renormalize the distribution so that
            ∫ g(v) dv = 1,
            where g(v) = 4 pi v^2 f(v).
            If False, require the input distribution to already be
            normalized within numerical precision.

        Raises
        ------
        ValueError
        If the normalization is non-finite or non-positive, or if
        ``normalize=False`` and the distribution is not normalized.
        """
        norm = self.normalization(self._v)

        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid normalization of 4 pi v^2 f(v).")

        if not normalize:
            if not np.isclose(norm, 1.0, rtol=1e-3):
                raise ValueError(f"Integral of g(v) is {norm:.6g}, expected 1.")
  
        else:
            self._f /= norm
