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
- nuclear structure momentum dependence
- propagator momentum dependence

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
from src.interactions.propagators import Propagator, PointLikePropagator
from src.nuclei.nucleus import Nucleus
from src.nuclei.base_nuclear_response import NuclearResponse, PointLikeNuclearResponse
from src.kinematics.base_kinematics import NuclearRecoilKinematics
from src.utils.constants import PROTON_MASS_MeV, NEUTRON_MASS_MeV

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
    - momentum dependence due to finite size

    Derived classes only need to implement
    the interaction-specific cross section.
    """
    def __init__(
        self,
        wimp: WIMP,
        nucleus: Nucleus,
        kinematics: NuclearRecoilKinematics,
        nuclear_response: NuclearResponse | None = None,
        propagator: Propagator | None = None,
    ):
        """
        Parameters
        ----------
        wimp : WIMP
            Incident dark matter particle.
        nucleus : Nucleus
            Target nucleus.
        kinematics : NuclearRecoilKinematics
            Kinematics model.
        nuclear_response : NuclearResponse, optional
            Momentum-transfer dependent nuclear
            response function.

            If omitted, a PointLikeNuclearResponse
            is used, corresponding to
            response(q) = 1
            for all momentum transfers.
        propagator : Propagator, optional
            Momentum-transfer dependent propagator
            response function.

            If omitted, a PointLikePropagator
            is used, corresponding to
            response(q) = 1
            for all momentum transfers.
        """
        self.wimp = wimp
        self.nucleus = nucleus
        self.kin = kinematics

        if kinematics.mass_wimp != wimp.mass:
            raise ValueError("Masses in WIMP and Kinematics are not the same")
        if kinematics.mass_target != nucleus.mass:
            raise ValueError("Masses in Nucleus and Kinematics are not the same")

        if nuclear_response is None:
            nuclear_response = PointLikeNuclearResponse()

        self.nuclear_response = nuclear_response

        if propagator is None:
            propagator = PointLikePropagator()

        self.propagator = propagator

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

        masked = np.where(mask, dsigmadER, 0.0)

        return np.nan_to_num(masked, nan=0.0,
                             posinf=0.0,neginf=0.0)

    def integrate_sigma(self, v: NDArray,
                        ER_min: float = 0.,
                        ER_max: float | None = None,
                        num: int = 10000) -> NDArray:
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

        For elastic scattering with point-like
        nuclear and propagator responses, the
        result should reproduce the analytic
        WIMP-nucleus cross section independently
        of the incoming WIMP speed.

        Kinematically forbidden recoil energies
        are automatically excluded through the
        interaction masking machinery.
        """
        if ER_max is None:
            ER_max = np.nanmax(self.kin.ER_max(v=v))

        if ER_min == ER_max:
            return np.zeros_like(v, dtype=float)

        ERrange = np.linspace(ER_min, ER_max, num=num)
        dsigma = self.dsigma_dER(v=v, ER=ERrange)
        return np.trapezoid(dsigma, ERrange, axis=1)

    def momentum_transfer_response(self, ER: NDArray) -> NDArray:
        """
        Evaluate the momentum-transfer response
        associated with the interaction.

        Parameters
        ----------
        ER : ndarray
            Recoil energy transfer in eV.

        Returns
        -------
        ndarray
            Dimensionless suppression or enhancement
            factor evaluated at the recoil energy.

        Notes
        -----
        Conversion from recoil energy to momentum-transfer
        happens internally.

        The effects include:
        - the nuclear response function
        - the propagator-dependent terms.
        """
        q = self.kin.q_from_ER(ER=ER)

        return (
            self.nuclear_response.response(q)
            * self.propagator.response(q)
        )

    def _convert_validate_v_ER(self,
                               v: NDArray,
                               ER: NDArray,
                               relativistic=False,
                               ) -> tuple[NDArray, NDArray]:
        """
        Convert inputs to float NumPy arrays and validate
        common interaction-model grids.

        Parameters
        ----------
        v : ndarray, shape (Nv,)
            Non-empty, finite, non-negative speed grid in units beta=v/c.
            For grids containing more than one point,
            values must be strictly increasing.
        
        ER : ndarray, shape (NER,)
            Non-empty, finite, non-negative, energy recoil grid in eV.
            For grids containing more than one point,
            values must be strictly increasing.
        
        relativistic: bool, optional
            Optionally check that v < 1.
            Default false.

        Returns
        -------
        tuple of ndarray
            Validated floating-point arrays (v, ER).

        Raises
        ------
        ValueError
            If either grid is empty, not one-dimensional or invalid.
        """
        v = self.kinematics._validate_domain_speed(
            v=v,
            require_grid=True)
        ER = self.kinematics._validate_domain_ER(
            ER=ER,
            require_grid=True)

        if (len(v) > 1) and np.any(np.diff(v) <= 0.):
            raise ValueError("v must be strictly increasing.")

        if (len(ER) > 1) and np.any(np.diff(ER) <= 0.):
            raise ValueError("ER must be strictly increasing.")

        if relativistic and np.any(v > 1.):
            raise ValueError("v (beta=v/c) must be <= 1.")
        
        return v, ER

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
    def _reduced_mass_neutron(self) -> float:
        """
        WIMP-neutron reduced mass (eV)
        """
        mn = 1e6 * NEUTRON_MASS_MeV
        return (mn * self.mass_wimp ) / (mn + self.mass_wimp)

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
    def scattering_kinematics(self) -> NuclearRecoilKinematics:
        """
        Returns the kinematics instance
        """
        # name is outdated, kept for legacy
        # it should be removed in a refactoring
        return self.kin

    @property
    def kinematics(self) -> NuclearRecoilKinematics:
        """
        Returns the kinematics instance
        """
        return self.kin

    @property
    @abstractmethod
    def sigma_reference(self) -> float:
        """
        WIMP-nucleon reference cross section used to
        normalize the specific interaction model.

        Returns
        -------
        float
            Reference cross section in cm^2.

        Notes
        -----
        Sensitivity calculations operate on this
        quantity and therefore require interaction
        models to implement this method to
        expose a meaningful WIMP-nucleon reference
        cross section normalization.
        """
        pass

    def _validate_sigma_reference(self) -> None:
        """
        Validates the reference cross section.
        """
        sigma = float(self.sigma_reference)

        if not np.isfinite(sigma):
            raise ValueError("Reference cross section must be finite.")

        if sigma < 0:
            raise ValueError("Reference cross section must be non-negative.")
