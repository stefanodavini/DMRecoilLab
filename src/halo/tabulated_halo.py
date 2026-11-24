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
from src.utils.tabulated_data import TabulatedData

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
    normalize : bool, optional
        If True, renormalize the distribution so that
        ∫ g(v) dv = 1,
        where g(v) = 4 pi v^2 f(v).
        If False, require the input distribution to already be
        normalized within numerical precision.
    trim_trailing_zeros: bool, optional
        If true, extra trailing zeros in f(v) are not stored.

    Notes
    -----
    If trimming is enabled, retain exactly one zero node
    beyond the last non-zero value.
    """

    def __init__(self,
                 v: ArrayLike,
                 f: ArrayLike,
                 normalize: bool = True,
                 trim_trailing_zeros: bool = False):

        self._data = TabulatedData(x=v, y=f,
                                  x_name="v",
                                  y_name="f(v)")

        self._data.check_x_non_negative()
        self._data.check_y_non_negative()

        self._v = np.asarray(self._data.x)
        self._f = np.asarray(self._data.y)

        self._validate_and_normalize(normalize=normalize)

        if trim_trailing_zeros:
            self._trim_trailing_zeros()

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
        Largest speed in the tabulated distribution.
        """
        return float(self._v[-1])

    @classmethod
    def from_file(cls,
                  filename: str,
                  normalize : bool = True,
                  trim_trailing_zeros: bool = False) -> Self:
        """
        Construct a tabulated distribution from a text file produced
        by to_file().

        The file must contain exactly two columns corresponding to
        speed and velocity-distribution values.

        Parameters
        ----------
        filename: str
            Input filename.

        normalize : bool, optional
            If True, renormalize the distribution so that
            ∫ g(v) dv = 1,
            where g(v) = 4 pi v^2 f(v).
            If False, require the input distribution to already be
            normalized within numerical precision.

        trim_trailing_zeros: bool, optional
            If true, extra trailing zeros in f(v) are not stored.

        Notes
        -----
        If trimming is enabled, retain exactly one zero node
        beyond the last non-zero value.

        Returns
        -------
        TabulatedSpeedDistribution
        """
        data = TabulatedData.from_file(filename)
        return cls(v=data.x, f=data.y,
                   normalize=normalize,
                   trim_trailing_zeros=trim_trailing_zeros)

    def integration_grid(self,
                         vmin: float | None = None,
                         num: int | None = None,
                         ) -> NDArray:
        """
        Return native speed nodes without resampling.

        Parameters
        ----------
        vmin: float, optional
            Minimum speed (in km/s) in the integration grid.
            If not provided, return the complete native grid.
            Otherwise, retain nodes at or above vmin and the
            immediately preceding node, if present,
            to preserve the boundary interval.

        Notes
        -----
        Return an empty array if vmin exceeds the final native node.

        Input num is accepted for interface compatibility and ignored.
        The returned grid is a copy.
        """
        if vmin is None:
            return self._v.copy()

        vmin = float(vmin)
        if not np.isfinite(vmin) or vmin < 0:
            raise ValueError("vmin must be finite and non-negative")

        if vmin > self._v[-1]:
            return np.empty(0, dtype=float)

        start = max(0,
                    int(np.searchsorted(self._v, vmin, side="left")) - 1)

        return self._v[start:].copy()

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

    def _trim_trailing_zeros(self):
        """
        Trim trailing zeros.

        Retain exactly one zero node
        beyond the last non-zero value.
        """
        mask = self._f > 0.0

        if np.any(mask):
            last_nonzero = np.nonzero(mask)[0][-1]

            # retain one trailing node if available
            stop = min(last_nonzero + 2, len(self._v))

            self._v = self._v[0:stop]
            self._f = self._f[0:stop]

