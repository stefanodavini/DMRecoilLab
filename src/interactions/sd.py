"""
Spin-dependent WIMP-nucleus interactions.

This module implements the standard
spin-dependent contact interaction
commonly used in direct-detection analyses.

The implementation supports:
- elastic and inelastic kinematics
- arbitrary proton/neutron coupling ratios
- point-like nuclei
- form factor equals 1

The normalization follows the
WIMP-proton cross section sigma_p.

Units:
    mass      -> eV
    energy    -> eV
    momentum  -> eV
    velocity  -> c = 1
    cross section -> cm^2

Examples
--------
>>> from src.nuclei.isotopes import Xe131
>>> from src.interactions.particles import WIMP

>>> wimp = WIMP(mass=50e9, spin=0)

>>> kin = ElasticKinematics(mX=wimp.mass, mT=Xe131.mass)

>>> interaction = SDInteraction(
...     wimp=wimp,
...     nucleus=Xe131,
...     kinematics=kin,
...     sigma_p=1e-45,,
... )

>>> v = np.linspace(0, 2.5e-3, 100)
>>> ER = np.logspace(3, 6)

>>> dsigma = interaction.dsigma_dER(v=v, ER=ER)

>>> sigma_over = interaction.integrate_dsigma(v=v, ERmin=1e4)
"""

from numpy.typing import NDArray
import numpy as np

from src.kinematics import ScatteringKinematics
from src.interactions.base_interaction_model import InteractionModel
from src.interactions.particles import WIMP
from src.nuclei.nucleus import Nucleus

class SDInteraction(InteractionModel):
    """
    Spin-dependent (SD) WIMP-nucleus interaction.

    The interaction is normalized to the WIMP-proton
    cross section sigma_p and supports generic
    proton and neutron couplings through the ratio

        an_over_ap = an / ap.

    The nuclear cross section is

        sigma_A 
        = sigma_p *
        4/3 * (mu_A / mu_p)^2
        * (J+1)/J 
        * [<Sp> + <Sn> an_over_ap]^2

    where:
        sigma_p
            WIMP-proton cross section.
        mu_p
            WIMP-proton reduced mass.
        mu_A
            WIMP-nucleus reduced mass.
        Z
            Proton number.
        A
            Mass number.

    The differential cross section is

        dσ/dER (v) = m_A sigma_A / (2 mu_A^2 v^2) F_SD^2(q)

    where:
        ER
            Nuclear recoil energy.
        v
            Incoming WIMP speed.
        F_SD(q)
            Spin dependent nuclear form factor.

    Parameters
    ----------
    wimp : WIMP
        WIMP properties.

    nucleus : Nucleus
        Target nucleus.

    kinematics : ScatteringKinematics
        Elastic or inelastic scattering
        kinematics.

    sigma_p : float
        WIMP-proton cross section (in cm^2).

    an_over_ap : float, optional
        Ratio between neutron and proton
        couplings.
        Default is 1 (isospin-conserving).
    """

    def __init__(
        self,
        *,
        wimp: WIMP,
        nucleus: Nucleus,
        kinematics: ScatteringKinematics,
        form_factor= None,
        sigma_p: float = 1e-45,
        an_over_ap: float = 1.,
        ):

        super().__init__(
            wimp=wimp,
            nucleus=nucleus,
            kinematics=kinematics)

        if form_factor is not None:
            raise NotImplementedError("SD form factors not implemented yet")

        if sigma_p < 0:
            raise ValueError("sigma_p must be non-negative")

        self.sigma_p = sigma_p
        self.an_over_ap = an_over_ap

    @property
    def spin_coupling(self) -> float:

        if np.isnan(self.an_over_ap):
            return self.nucleus.Sn

        return (
            self.nucleus.Sp
            + self.an_over_ap * self.nucleus.Sn
        )

    @property
    def sigma_nucleus(self) -> float:
        J = self.nucleus.spin

        if J == 0.:
            return 0.

        mu_ratio2 = (self._reduced_mass_nucleus / self._reduced_mass_proton) ** 2
        # in the neutron only, scale by reduced mass ratio squared?

        spin_factor = 4 * (J + 1) / (3 * J)
    
        return (self.sigma_p * mu_ratio2 * spin_factor * self.spin_coupling**2)

    def dsigma_dER(self,
                   v: NDArray,
                   ER: NDArray) -> NDArray:

        ER_grid, v_grid = self._broadcast_grids(ER=ER, v=v)

        # check shape here, must be coherent with return type
        if self.nucleus.spin == 0:
            return np.zeros((len(v), len(ER)), dtype=float)

        kinematic_prefactor = np.where(
            v_grid > 0,
            self.nucleus.mass / (2* self._reduced_mass_nucleus**2 * v_grid**2),
            0.0)

        F2 = 1 # modify here to include form factor implementations

        dsigma_unmasked = (kinematic_prefactor * self.sigma_nucleus * F2)

        return self._apply_mask_to_dsigmadER(
            dsigmadER=dsigma_unmasked, v=v, ER=ER)

    @property
    def sigma_proton(self) -> float:
        if np.isnan(self.an_over_ap):
            return 0.

        return self.sigma_p


