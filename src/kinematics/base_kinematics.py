"""
Abstract base classes for nuclear recoil kinematics.

Units
-----
All quantities are expressed in natural units:

    c = 1

with:

    mass      : eV
    energy    : eV
    momentum  : eV
    velocity  : dimensionless

The module defines NuclearRecoilKinematics,
the top-level abstraction for recoil-producing
dark-matter interactions.

Concrete implementations may describe
scattering, absorption, or other processes
leading to nuclear recoils.
"""

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray
import numpy as np

class NuclearRecoilKinematics(ABC):
    """
    Abstract base class for nuclear recoil kinematics.

    Defines the minimal interface required by any
    process producing nuclear recoils, regardless of
    the underlying interaction mechanism.

    Implementations must provide:

    - recoil-energy bounds
    - momentum transfer from recoil energy
    - threshold speed

    Examples include elastic scattering,
    inelastic scattering, and dark-matter absorption.
    """
    def __init__(self, mX: float, mT: float):
        self.mX = mX
        self.mT = mT
        self._check_masses()

    @abstractmethod
    def q_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Return the momentum scale associated with a
        recoil energy ER.

        The precise interpretation of the returned
        quantity depends on the physical process.

        For scattering implementations this is
        typically the momentum transfer.

        Parameters
        ----------
        ER : ArrayLike
            Recoil energy in eV.

        Returns
        -------
        ndarray
            Momentum scale in eV.
        """
        pass

    @abstractmethod
    def ER_min(self, v: ArrayLike) -> NDArray:
        """
        Minumum recoil energy allowed for a
        dark-matter speed v.
        
        Parameters
        ----------
        v : ArrayLike
            Incoming dark-matter speed (c=1).
        
        Returns
        -------
        ndarray
            Lower kinematic recoil-energy bound
            in eV.
        
        Notes
        -----
        For elastic scattering, the mimimum recoil energy is zero.
        
        For endothermic scattering, speeds below
        the threshold speed are kinematically forbidden.
                
        Endothermic kinematics implementations 
        must return NaN for recoil energy below the theshold speed.
        """
        pass

    @abstractmethod
    def ER_max(self, v: ArrayLike) -> NDArray:
        """
        Maximum recoil energy allowed for
        a dark-matter speed v.

        Parameters
        ----------
        v : ArrayLike
            Incoming dark-matter speed (c=1).

        Returns
        -------
        ndarray
            Upper kinematic recoil-energy bound
            in eV.
        """
        pass

    def ER_bounds(self,
                  v: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Allowed recoil-energy interval at speed v.

        Parameters
        ----------
        v : ArrayLike
            Incoming dark-matter speed.

        Returns
        -------
        tuple
            (ER_min, ER_max)

        Notes
        -----
        Elastic scattering yields ER_min = 0,
        while inelastic scattering generally
        produces a non-zero speed-dependent lower bound.
        """
        return (self.ER_min(v), self.ER_max(v))

    @property
    @abstractmethod
    def threshold_speed(self) -> float:
        """
        Minimum physically allowed incoming dark-matter speed.
        """
        pass

    @property
    def mass_target(self) -> float:
        """
        Target mass
        """
        return self.mT

    @property
    def mass_wimp(self) -> float:
        """
        Dark matter mass
        """
        return self.mX

    def _check_masses(self):
        """
        Raise error in case of unphysical masses.
        """
        if not np.isfinite(self.mX) or self.mX <= 0:
            raise ValueError("Unphysical dark-matter mass")
        
        if not np.isfinite(self.mT) or self.mT <=0:
            raise ValueError("Unphysical target mass")
