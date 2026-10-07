"""
Nuclear form factors for dark-matter direct detection.

This module provides implementations of scalar nuclear
form factors, which account for the finite spatial extent
of the nucleus and suppress coherent scattering at large
momentum transfer.

The form factor F(q) modifies the scattering amplitude,
while the corresponding nuclear response is

    |F(q)|².

Implemented models
------------------
PointLikeFormFactor
    Idealized point-like nucleus with

        F(q) = 1.

HelmFormFactor
    Standard Helm parameterization based on a
    unifrom shperical nuclear density profile
    convolved with a Gaussian surface thickness.

Notes
-----
Momentum transfers are expressed in eV.
Natural units (c = ħ = 1) are used throughout
the module.

References
----------
Lewin, J. D. and Smith, P. F.
'Review of mathematics, numerical factors, and
corrections for dark matter experiments',
Astroparticle Physics 6 (1996) 87-112.
"""

from src.nuclei.base_nuclear_response import NuclearResponse
from src.nuclei.nucleus import Nucleus
from src.utils.conversions import convert_qr_to_dimensionless

from scipy.special import spherical_jn

from abc import abstractmethod
from numpy.typing import NDArray
import numpy as np

class FormFactor(NuclearResponse):
    """
    Scalar nuclear form factor.

    The form factor modifies the coherent
    scattering amplitude to account for the
    finite spatial extent of the nucleus.

    The corresponding nuclear response
    is |F(q)|^2.
    """

    @abstractmethod
    def F(self, q: NDArray) -> NDArray:
        pass

    def response(self, q: NDArray) -> NDArray:
        """
        Response for a scalar nuclear form factor.

        The response is evaluated as |F(q)|^2
        """
        return self.F(q)**2

class PointLikeFormFactor(FormFactor):
    """
    Point like form factor.

    It can represent one of the physical limits:
    - zero momentum transfer
    - point-like nucleus

    The form factor is given by
    F(q) = 1
    """
    def __init__(self):
        pass

    def F(self, q: NDArray) -> NDArray:
        """
        Point Like form factor

        Parameters
        ----------
        q : ndarray
            Momentum transfer in eV.

        Returns
        -------
        ndarray of 1.
        """
        q = np.asarray(q)

        return np.ones_like(q, dtype=float)

class HelmFormFactor(FormFactor):
    """
    Helm nuclear form factor.

    The form factor is given by

    F(q) = 3 j1(qR1)/(qR1) exp(-(qs)^2/2)

    using the Lewin-Smith parameterization.

    Parameters
    ----------
    nucleus
        Target nucleus.

    References
    ----------
    Lewin and Smith (1996),
    Astroparticle Physics 6, 87.
    """
    def __init__(self, nucleus: Nucleus):
        self.A = nucleus.A
        self.c_fm = 1.23 * self.A**(1/3) - 0.6
        self.s_fm = 0.9
        self.R1_fm = self.R1

    def F(self, q: NDArray) -> NDArray:
        """
        Helm Form Factor

        Parameters
        ----------
        q : ndarray
            Momentum transfer in eV.

        Returns
        -------
        ndarray
            Helm form factor F(q).

        Notes
        -----
        The normalization satisfies
        F(0) = 1.

        The implementation uses the
        Lewin-Smith parameterization.
        """
        q = np.asarray(q)

        qR1 = convert_qr_to_dimensionless(q * self.R1_fm)
        qs2 = convert_qr_to_dimensionless(q * self.s_fm) ** 2

        F = np.ones_like(qR1)
        mask = qR1 != 0

        F[mask] = 3 * spherical_jn(1, qR1[mask]) / qR1[mask] * np.exp(-qs2[mask] / 2)

        return F

    @property
    def R1(self):
        """
        Effective radius (fm) in Lewin-Smith prescription

        Notes
        -----
        The effective radius is
        R1² = c² + 7π²a²/3 - 5s².
        Default parameters follow Lewin & Smith (1996).
        """
        a_fm = 0.52
        return np.sqrt(
            self.c_fm**2 + 7 / 3 * np.pi**2 * a_fm**2  - 5 * self.s_fm**2)

