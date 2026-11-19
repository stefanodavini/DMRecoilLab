import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.kinematics.scattering_kinematics import NuclearRecoilKinematics


class AbsorptionKinematics(NuclearRecoilKinematics):
    """
    Leading-order monochromatic chi + N -> nu + N,
    for mX << mT, neglecting incident velocity broadening.

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

    def q_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Absorbed DM momentum scale inferred from the monochromatic recoil.

        Nonrelativistic target-recoil approximation.
        """
        ER = np.asarray(ER, dtype=float)
        with np.errstate(invalid="ignore"):
            return np.sqrt(2 * self.mT * ER)

    def ER_min(self, v: ArrayLike) -> NDArray:
        """
        Return the monochromatic recoil in eV with the shape of v.
        """
        return np.full_like(np.asarray(v, dtype=float), self.ER)

    def ER_max(self, v: ArrayLike) -> NDArray:
        """
        Return the monochromatic recoil in eV with the shape of v.
        """
        return np.full_like(np.asarray(v, dtype=float), self.ER)

    @property
    def threshold_speed(self) -> float:
        """
        Minimum physically allowed incoming WIMP speed.
        """
        return 0.0

    @property
    def supports_scattering_interactions(self) -> bool:
        """
        The class instance can not be used
        in scattering interaction models
        such as SI/SD
        """
        return False

