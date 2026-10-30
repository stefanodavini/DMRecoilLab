import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.kinematics.scattering_kinematics import ScatteringKinematics


class AbsorptionKinematics(ScatteringKinematics):
    """
    Leading-order monochromatic fermionic DM absorption
    chi + N -> nu + N
    for  mX << mT.

    Parameters
    ----------
    mX : float
        Dark matter mass in eV.
    mT : float
        Target nucleus mass in eV.

    Notes
    -----
    Assumes a massless neutrino and neglects velocity broadening.
    The recoil energy is ER = mX**2 / (2*mT).

    vmin_from_ER returns zero only as the interface placeholder requested
    for this first implementation; it does not determine recoil support.

    Reference: Dror, Elor and McGehee, arXiv:1908.10861, Eq. (3.4).
    """

    def __init__(self, mX: float, mT: float):
        super().__init__(mX=mX, mT=mT)

    @property
    def ER(self) -> float:
        """
        Return the monochromatic nuclear recoil energy in eV.
        """
        return self.mX**2 / (2.0 * self.mT)

    def vmin_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Return the zero-speed placeholder, preserving the input shape.
        """
        ER = np.asarray(ER)
        if ER.ndim == 0:
            return 0.0
        return np.zeros_like(ER, dtype=float)

    def ER_min(self, v: ArrayLike) -> NDArray:
        """
        Return the monochromatic recoil in eV with the shape of v.
        """
        v = np.asarray(v)
        if v.ndim == 0:
            return self.ER
        return np.full_like(v, self.ER, dtype=float)

    def ER_max(self, v: ArrayLike) -> NDArray:
        """
        Return the monochromatic recoil in eV with the shape of v.
        """
        v = np.asarray(v)
        if v.ndim == 0:
            return self.ER
        return np.full_like(v, self.ER, dtype=float)

    def q_from_v_theta(self, v: ArrayLike,
                       theta: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Angular kinematics - Not implemented.
        """
        raise NotImplementedError(
            "Angular kinematics are not defined for absorption.")

