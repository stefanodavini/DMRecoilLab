"""
Spin-dependent WIMP-nucleus interactions.

This module implements the standard
spin-dependent contact interaction
commonly used in direct-detection analyses.

The implementation supports:
- elastic and inelastic kinematics
- arbitrary proton/neutron coupling ratios
- point-like SD response
- F_SD^2(q) = 1
- propagator momentum dependence

The normalization is expressed in terms
of a reference nucleon cross section.
For proton-normalized interactions the
usual sigma_proton convention is recovered.
For neutron-only interactions the
reference cross section is sigma_neutron.

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
...     sigma_reference=1e-45,
...     an_over_ap=1,
... )

>>> v = np.linspace(0, 2.5e-3, 100)
>>> ER = np.logspace(3, 6)

>>> dsigma = interaction.dsigma_dER(v=v, ER=ER)

>>> sigma_over = interaction.integrate_sigma(v=v, ER_min=1e4)
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

    The normalization is expressed in terms
    of a reference nucleon cross section.

    For proton-normalized interactions the
    usual sigma_p convention is recovered.

    For neutron-only interactions the
    reference cross section is sigma_n.

    The interaction supports generic
    proton and neutron couplings through the ratio

        an_over_ap = an / ap.

    The nuclear cross section for finite an/ap is

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

    form_factor : optional
        Reserved for future SD form-factor
        implementations.

        Currently only None is supported.
        Supplying any value raises
        NotImplementedError.

    sigma_reference : float
        Reference WIMP-nucleon cross section (in cm^2).

        For finite an/ap this corresponds to the
        WIMP-proton cross section.

        For neutron-only interactions
        (an_over_ap = NaN), this corresponds to
        the WIMP-neutron cross section.

    an_over_ap : float, optional
        Ratio between neutron and proton
        couplings.
        Default is 1 (isospin-conserving).

        If NaN, a neutron-only interaction
        is assumed and the cross section is
        normalized according to the internal
        neutron-only convention.
    """

    def __init__(
        self,
        *,
        wimp: WIMP,
        nucleus: Nucleus,
        kinematics: ScatteringKinematics,
        form_factor= None,
        sigma_reference: float = 1e-45,
        an_over_ap: float = 1.,
        ):

        if form_factor is not None:
            raise NotImplementedError("SD form factors not implemented yet")

        super().__init__(
            wimp=wimp,
            nucleus=nucleus,
            kinematics=kinematics,
            nuclear_response=form_factor)

        if sigma_reference < 0:
            raise ValueError("sigma_reference must be non-negative")

        self.sigma_ref = sigma_reference
        self.an_over_ap = an_over_ap

    @property
    def spin_coupling(self) -> float:
        """
        Effective spin coupling entering the SD cross section.

        Returns
        -------
        float
            Effective nuclear spin coupling

            <Sp> + (an/ap) <Sn>

        for finite an/ap.

        If an_over_ap is NaN, a neutron-only
        convention is assumed and the returned
        coupling is simply <Sn>.
        """

        if self.is_neutron_only:
            return self.nucleus.Sn

        return self.nucleus.Sp + self.an_over_ap * self.nucleus.Sn

    @property
    def sigma_nucleus(self) -> float:
        """
        Effective SD WIMP-nucleus cross section.

        Returns
        -------
        float
            Zero-momentum-transfer SD cross section
            in cm^2.

        Notes
        -----
        The normalization follows a reference
        WIMP-nucleon cross section.

        For finite an/ap this corresponds to
        sigma_proton.

        For neutron-only interactions
        (an_over_ap = NaN) it corresponds to
        sigma_neutron.
        """
        J = self.nucleus.spin

        if J == 0.:
            return 0.

        if not self.is_neutron_only:
            mu_ratio2 = (self._reduced_mass_nucleus / self._reduced_mass_proton) ** 2
        else:
            mu_ratio2 = (self._reduced_mass_nucleus / self._reduced_mass_neutron) ** 2

        spin_factor = 4 * (J + 1) / (3 * J)
    
        return (self.sigma_ref * mu_ratio2 * spin_factor * self.spin_coupling**2)

    def dsigma_dER(self,
                   v: NDArray,
                   ER: NDArray) -> NDArray:
        """
        Differential spin-dependent cross section.

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
            on the speed-recoil-energy mesh.

            Axis convention:
            axis 0 -> speed
            axis 1 -> recoil energy

        Notes
        -----
        The returned values already include:
        - spin coupling enhancement
        - reduced-mass scaling
        - kinematic masking

        Kinematically forbidden regions are
        automatically set to zero.

        For a point-like nucleus and elastic
        scattering, integrating the differential
        cross section over the physically allowed
        recoil-energy range reproduces the total
        WIMP-nucleus total cross section sigma_A.
        """

        ER_grid, v_grid = self._broadcast_grids(ER=ER, v=v)

        if self.nucleus.spin == 0.:
            return np.zeros_like(v_grid * ER_grid, dtype=float)

        kinematic_prefactor = np.where(
            v_grid > 0,
            self.nucleus.mass / (2* self._reduced_mass_nucleus**2 * v_grid**2),
            0.0)

        q_response = self.momentum_transfer_response(ER=ER_grid)

        dsigma_unmasked = (kinematic_prefactor * self.sigma_nucleus * q_response)

        return self._apply_mask_to_dsigmadER(
            dsigmadER=dsigma_unmasked, v=v, ER=ER)

    @property
    def is_neutron_only(self) -> bool:
        """
        Whether the interaction uses the
        neutron-only normalization convention.

        Returns
        -------
        bool
            True when an_over_ap is NaN.
        """
        return np.isnan(self.an_over_ap)

    @property
    def sigma_reference(self) -> float:
        """
        Reference cross-section parameter used to
        normalize the SD interaction model.

        The reference cross section is:
        - the WIMP-neutron cross section in case
            of neutron-only coupling
        - the WIMP-proton cross section otherwise

        Returns
        -------
        float
            Reference cross section in cm^2.

        Notes
        -----
        Sensitivity calculations operate on this
        quantity.
        """
        return self.sigma_ref

    @property
    def sigma_proton(self) -> float:
        """
        WIMP-proton cross section in cm^2.

        Returns
        -------
        float
            The proton cross-section in cm^2.

        For neutron-only interactions
        (an_over_ap = NaN), zero is returned.
        """
        if self.is_neutron_only:
            return 0.

        return self.sigma_ref


