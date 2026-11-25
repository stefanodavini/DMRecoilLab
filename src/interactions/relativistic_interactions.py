"""
Scalar-nucleon interactions with relativistic scattering kinematics.

Implements the ss and ps prescriptions of arXiv:2609.14799, Eqs. (4)-(7).
The first current in each name is the DM current;
the second is the nucleon current.
These equations retain the paper's matrix-element
approximations, rather than a fully relativistic matrix-element calculation.

Units:
    mass      -> eV
    energy    -> eV
    momentum  -> eV
    Lambda    -> eV
    velocity  -> c = 1
    cross section -> cm^2

Nuclear and propagator responses follow InteractionModel's convention:
they are multiplicative cross-section responses evaluated at spatial |q|.
The explicit ps momentum factor instead uses Q2 = -t = 2*m_A*ER,
matching the paper's q^2 prescription.

Internally the interaction model remains normalized through
the target-independent proton reference cross section sigma_p.

In EFT, Lagrangian coupling strengths are parameterized by
the quantity Lambda.

O_ss = 1 / Lambda^2 (bar chi1 chi2) (bar N N)
O_ps = 1 / Lambda^2 (bar chi1  i gamma_5 chi2) (bar N N)

Internally, Lambda is defined through the
nucleus reference normalization

    sigma_A
    =
    2 mu_A^2 (hbar c)^2
    / (pi Lambda^4).

The relativistic differential cross section
is obtained by multiplying this normalization
by the appropriate kinematic, response and
operator-dependent factors.
"""

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.interactions.base_interaction_model import InteractionModel
from src.interactions.particles import WIMP
from src.interactions.propagators import Propagator
from src.kinematics.relativistic_kinematics import RelativisticKinematics
from src.nuclei.form_factors import FormFactor, PointLikeFormFactor
from src.nuclei.nucleus import Nucleus
from src.utils.constants import HBAR_C_MeV_fm


_HBARC_EV_CM = HBAR_C_MeV_fm * 1e-7 # 1.973269804e-5


class RelativisticDMScalarInteraction(InteractionModel):
    """Abstract base for DM currents coupled to a scalar nucleon current.

    Parameters
    ----------
    wimp : WIMP 
        Incoming WIMP.
    nucleus : Nucleus
        Target nucleus.
    kinematics : RelativisticKinematics
        Relativistic elastic or inelastic scattering kinematics.
    sigma_p : float, optional
        Required target-independent proton reference cross section, cm^2.
        sigma_p = 2*mu_p^2 / (pi*Lambda^4) * (hbar*c)^2 .
        Default: 1e-40 cm^2.
        Defined independently of momentum for both ss and ps.
        For ps it is a coupling normalization, not a proton total cross
        section; all momentum suppression remains explicit.
    form_factor : FormFactor, optional
        Nuclear cross-section response F^2. Defaults to point-like.
    propagator : Propagator, optional
        Additional cross-section response. Defaults to point-like.
    fn_over_fp : float, optional
        Neutron/proton scalar coupling ratio; defaults to 1.

    Notes
    -----
    d sigma / d ER = m_A * sigma_p/(2*mu_p^2)
                    * (1-v^2)/v^2 * [Z+N*fn_over_fp]^2 * response
                    * DM momentum factor.
    Grid validation and kinematic masking follow InteractionModel.
    At v=0 the differential cross section is assigned zero, as in SI.
    """

    def __init__(
        self,
        *,
        wimp: WIMP,
        nucleus: Nucleus,
        kinematics: RelativisticKinematics,
        sigma_p: float = 1e-40,
        form_factor: FormFactor | None = None,
        propagator: Propagator | None = None,
        fn_over_fp: float = 1.0,
    ):
        if not isinstance(kinematics, RelativisticKinematics):
            raise TypeError("Requires RelativisticKinematics.")

        if form_factor is None:
            form_factor = PointLikeFormFactor()
        if not isinstance(form_factor, FormFactor):
            raise TypeError("Requires a FormFactor nuclear response.")

        super().__init__(
            wimp=wimp,
            nucleus=nucleus,
            kinematics=kinematics,
            nuclear_response=form_factor,
            propagator=propagator,
        )
        self.sigma_p = float(sigma_p)
        self._validate_sigma_reference()

        self.fn_over_fp = float(fn_over_fp)
        if not np.isfinite(self.fn_over_fp):
            raise ValueError("fn_over_fp must be finite.")

    def dsigma_dER(self, v: NDArray, ER: NDArray) -> NDArray:
        """
        Differential cross section, masked.

        Evaluates

        dsigma/dER = mA sigma_p /(2 μp²)
        * coherent_coupling²
        * (1-v²)/v²
        * response(q)
        * DM_momentum_factor

        with response(q) including the nuclear
        form factor and propagator response.

        Kinematically forbidden points are
        returned as zero.

        Parameters
        ----------
        v : ndarray, shape (Nv,)
            Incoming WIMP-speed grid
            in unit c=1.
        ER : ndarray, shape (NER,)
            Recoil-energy grid in eV.

        Returns
        -------
        ndarray, shape (Nv, NER)

            Differential cross section evaluated
            on the speed-recoil-energy mesh,
            in cm^2/eV.

            Axis convention:
            axis 0 -> speed
            axis 1 -> recoil energy

        Both inputs must be non-empty, finite, strictly increasing 1D
        grids (singletons are allowed), with 0 <= v < 1 and ER >= 0.
        Closed channels and recoils outside ER_bounds return zero.
        """
        v, ER = self._convert_validate_v_ER(v=v, ER=ER)
        ER_grid, v_grid = self._broadcast_grids(v=v, ER=ER)

        velocity_factor = np.divide(
            (1 - v_grid) * (1 + v_grid),
            v_grid**2,
            out=np.zeros_like(v_grid),
            where=v_grid > 0,
        )
        prefactor = self.mass_nucleus / (2 * self._reduced_mass_proton**2)
        response = self.momentum_transfer_response(ER=ER_grid)
        dsigma = (prefactor * self.sigma_reference * self.coherent_coupling**2
                  * velocity_factor * response * self._dm_momentum_factor(ER_grid))

        return self._apply_mask_to_dsigmadER(
            dsigmadER=dsigma, v=v, ER=ER,
        )

    @abstractmethod
    def _dm_momentum_factor(self, ER: NDArray) -> NDArray:
        """Dimensionless factor from the DM current, at recoil energy ER."""
        pass

    @property
    def sigma_reference(self) -> float:
        """Target-independent proton reference cross section, cm^2."""
        return self.sigma_proton

    @property
    def sigma_proton(self) -> float:
        """Proton reference normalization, cm^2."""
        return self.sigma_p

    @property
    def coherent_coupling(self) -> float:
        """Z + N*fn_over_fp; equals A for equal nucleon couplings."""
        return self.nucleus.Z + self.nucleus.N * self.fn_over_fp

    @property
    def sigma_nucleus(self) -> float:
        """
        Reference nucleus cross section normalization, in cm^2.

        Defined as
        
        sigma_A = sigma_p * (mu_A/mu_p)^2 * coherent_coupling^2.

        This quantity is useful when comparing to
        nucleus-normalized conventions appearing in
        the relativistic-scattering literature.

        It is generally a normalization parameter,
        not an integrated physical cross section.
        """
        return (self.sigma_reference
                * (self._reduced_mass_nucleus / self._reduced_mass_proton)**2
                * self.coherent_coupling**2)

    
    @property
    def Lambda (self) -> float:
        r"""
        Effective interaction scale appearing
        in the relativistic DM-nucleon operator.

        Defined through the nuclear reference cross section
        sigma_A = 2 mu_A^2 (hbar c)^2 / (pi Lambda_A^4),

        where
        sigma_A = sigma_p * (mu_A/mu_p)^2 * coherent_coupling^2.

        Returns
        -------
        float
            Interaction scale (in eV)
            appearing in the effective Lagrangian.

        Notes
        -----
        This quantity corresponds to the convention often
        adopted in relativistic scattering papers where

        d sigma / d ER = m_A / (pi Lambda_A^4) * K(v) * R(q).
        """
        return self._Lambda_from_sigma(mu=self._reduced_mass_nucleus,
                                       sigma=self.sigma_nucleus)

    def set_Lambda(self, Lambda):
        """
        Sets the internal proton reference cross section
        such that the equivalent nucleus-level coupling
        scale equals the supplied Lambda.

        Parameters
        ----------
        Lambda : float
            Nucleus interaction scale in eV.

        Notes
        -----
        Internally computes the nucleus reference 
        normalization

        sigma_A = 2 mu_A^2 (hbar c)^2 / (pi Lambda^4)

        and updates the stored proton reference
        cross section sigma_p accordingly.
        """
        self.sigma_p = (self._sigma_from_Lambda(mu=self._reduced_mass_proton,
                                                Lambda=Lambda)
                        / self.coherent_coupling**2)

    @staticmethod
    def _sigma_from_Lambda(mu: float, Lambda: float) -> float:
        """
        Convert an effective contact scale into
        a reference cross section normalization.
        """
        if not np.isfinite(Lambda) or Lambda <= 0:
            raise ValueError("Lambda must be finite and positive")
        return (2 * mu**2 * _HBARC_EV_CM**2 / (np.pi * Lambda**4))

    @staticmethod
    def _Lambda_from_sigma(mu: float, sigma: float) -> float:
        """
        Inverse of _sigma_from_Lambda.
        Returns infinity when sigma is zero.
        """
        if sigma == 0:
            return float("inf")

        return float((2 * mu**2 * _HBARC_EV_CM**2 / (np.pi * sigma))**0.25)

class RelativisticScalarScalarInteraction(RelativisticDMScalarInteraction):
    """
    Scalar DM current coupled to a scalar nucleon current.

    Uses a unit DM momentum factor.
    The reference sigma_p equals the elastic NR
    zero-speed-limit proton total cross section.
    With inelastic kinematics it remains a reference normalization.
    Constructor parameters are documented in RelativisticDMScalarInteraction.
    """

    def _dm_momentum_factor(self, ER: NDArray) -> NDArray:
        return np.ones_like(ER)


class RelativisticPseudoscalarScalarInteraction(RelativisticDMScalarInteraction):
    """
    Pseudoscalar DM current coupled to a scalar nucleon current.

    Uses the explicit momentum factor Q2/(4*m_chi^2), where
    Q2 = -t = 2*m_A*ER, matching the paper's q^2 prescription.

    sigma_p = 2*mu_p^2*(hbar*c)^2/(pi*Lambda^4), just as for ss.
    It normalizes the coupling before momentum suppression;
    it is not the zero-momentum or integrated proton cross section.
    No reference momentum is required.
    Constructor parameters are documented in RelativisticDMScalarInteraction.
    """

    def _dm_momentum_factor(self, ER: NDArray) -> NDArray:
        return self.kin.Q2_from_ER(ER=ER) / (4 * self.mass_wimp**2)
