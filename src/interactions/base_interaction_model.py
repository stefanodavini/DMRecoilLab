"""
Base classes for WIMP-nucleus interaction models.

Interaction models compute differential
and integrated scattering cross sections
for a specified WIMP, target nucleus,
and scattering kinematics.

The base class provides common machinery
for
- velocity/recoil-energy broadcasting
- kinematic masking
- recoil-energy integration
- reduced-mass calculations

while derived classes implement the
interaction-specific differential
cross section.

Units
-----
mass      -> eV
energy    -> eV
momentum  -> eV
velocity  -> c = 1
cross section -> cm²

Notes
-----
Unlike several higher-level modules
(e.g. halo models or kinematics),
interaction models operate directly
on broadcasted NumPy arrays.
Public methods therefore generally
expect NDArray inputs rather than
generic ArrayLike objects.
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

from src.interactions.particles import WIMP
from src.nuclei.nucleus import Nucleus
from src.kinematics import ScatteringKinematics
from src.utils.constants import PROTON_MASS_MeV

class InteractionModel(ABC):
    """
    Abstract base class for WIMP-nucleus
    interaction models.

    An interaction model is responsible for
    evaluating the differential scattering
    cross section

    dσ/dER (v)

    for specified incoming WIMP speeds and
    recoil energies.

    The class provides common functionality:
    - recoil-energy masking
    - kinematic threshold masking
    - velocity/recoil-energy grid broadcasting
    - differential cross section integration

    Derived classes only need to implement
    the interaction-specific cross section.
    """
    def __init__(
        self,
        wimp: WIMP,
        nucleus: Nucleus,
        kinematics: ScatteringKinematics,
    ):
        self.wimp = wimp
        self.nucleus = nucleus
        self.kin = kinematics

        if kinematics.mass_wimp != wimp.mass:
            raise ValueError("Masses in WIMP and ScatteringKinematics are not the same")
        if kinematics.mass_target != nucleus.mass:
            raise ValueError("Masses Nucleus and ScatteringKinematics are not the same")

    def kinematic_mask(self, v: NDArray, ER: NDArray) -> NDArray:
        """
        Return a boolean mask selecting
        physically allowed kinematics region.

        The final mask combines:
        - recoil-energy bounds
        - kinematic velocity constraints
        such as the minimum speed required for
        endothermic inelastic scattering.

        Parameters
        ----------
        v : ndarray, shape (Nv,)
            WIMP-speed grid.
            Units c=1.

        ER : ndarray, shape (NER,)
            Recoil-energy grid, in eV.

        Returns
        -------
        ndarray, shape (Nv, NER)
            Boolean mask identifying
            physically allowed kinematic regions.
        """
        ER_grid, v_grid = self._broadcast_grids(v=v, ER=ER)

        ER_min, ER_max = self.kin.ER_bounds(v_grid)
        ER_mask = ((ER_grid >= ER_min) & (ER_grid <= ER_max))

        speed_mask = self.kin.speed_mask(v_grid)

        return ER_mask & speed_mask

    @abstractmethod
    def dsigma_dER(self, v: NDArray, ER: NDArray) -> NDArray:
        """
        Differential cross section.

        Parameters
        ----------
        v
            Incoming WIMP-speed grid.
            Units c=1.
        ER
            Recoil-energy grid, in eV.

        Returns
        -------
        ndarray
            Shape: (Nv, NER)
        """
        pass

    def _broadcast_grids(self,
                         v: NDArray, 
                         ER: NDArray) -> tuple[NDArray, NDArray]:
        """
        Convert one-dimensional velocity and
        recoil-energy grids into broadcastable
        two-dimensional arrays.

        Both returned arrays are views and do not
        copy the underlying data.

        Parameters
        ----------
        v : ndarray, shape (Nv,)
        ER : ndarray, shape (NER,)

        Returns
        -------
        tuple
        ER_grid : shape (1, NER)
        v_grid : shape (Nv, 1)

        Notes
        -----
        All interaction models use the
        convention:

        axis 0 -> velocity
        axis 1 -> recoil energy
        """
        ER = np.asarray(ER)
        v = np.asarray(v)

        return (ER[None, :], v[:, None])

    def _apply_mask_to_dsigmadER(self,
                                 dsigmadER: NDArray,
                                 v: NDArray,
                                 ER: NDArray) -> NDArray:
        """
        Apply the kinematic mask to a
        differential cross-section array.

        Kinematically forbidden regions
        are set to zero.
        """
        mask = self.kinematic_mask(v=v, ER=ER)
        return np.where(mask, dsigmadER, 0.0)

    def integrate_sigma(self, v: NDArray, ER_min=0.,
                        ER_max=None, num=10000) -> NDArray:
        """
        Integrate the differential cross section
        over recoil energy.

        Parameters
        ----------
        v : ndarray, shape (Nv)
            Incoming WIMP-speed grid (units c=1).

        ER_min: float, optional
            Minimum recoil energy in eV.
            Default is zero.

        ER_max: float, optional
            Maximum recoil energy in eV.

            Default is the maximum energy
            allowed by the kinematics.

        num: int, optional
            Number of points
            for the energy grid.

        Returns
        -------
        ndarray, shape (Nv,)
            Total cross section evaluated for
            each incoming WIMP speed.

        Notes
        -----
        The integration is performed over the
        recoil-energy axis.

        For elastic scattering with a
        PointLikeFormFactor, the result should
        reproduce the analytic WIMP-nucleus
        cross section independently of the incoming speed.

        Kinematically forbidden recoil energies
        are automatically excluded through the
        interaction masking machinery.
        """
        if ER_max is None:
            ER_max = np.nanmax(self.kin.ER_max(v=v))

        ERrange = np.linspace(ER_min, ER_max, num=num)
        dsigma = self.dsigma_dER(v=v, ER=ERrange)
        return np.trapezoid(dsigma, ERrange, axis=1)

    @property
    def _reduced_mass_nucleus(self) -> float:
        """
        WIMP-nucleus reduced mass (eV)
        """
        return self.kin.mu

    @property
    def _reduced_mass_proton(self) -> float:
        """
        WIMP-proton reduced mass (eV)
        """
        mp = 1e6 * PROTON_MASS_MeV
        return (mp * self.mass_wimp ) / (mp + self.mass_wimp)

    @property 
    def mass_wimp(self) -> float:
        """
        WIMP mass (eV)
        """
        return self.kin.mass_wimp

    @property
    def mass_nucleus(self) -> float:
        """
        Nucleus mass (eV)
        """
        return self.kin.mass_target

    @property
    def scattering_kinematics(self) -> ScatteringKinematics:
        """
        Returns the ScatteringKinematics instance
        """
        return self.kin

    @property
    @abstractmethod
    def sigma_proton(self) -> float:
        """
        WIMP-proton cross-section parameter used to
        normalize the specific interaction model.

        Returns
        -------
        float
            Reference WIMP-proton cross section in cm^2.

        Notes
        -----
        Sensitivity calculations operate on this
        quantity and therefore require interaction
        models to implement this method to
        expose a meaningful proton-level
        cross-section normalization.
        """
        pass
