"""
Spin-independent WIMP-nucleus interactions.

This module implements the standard
spin-independent contact interaction
commonly used in direct-detection analyses.

The implementation supports:
- elastic and inelastic kinematics
- arbitrary proton/neutron coupling ratios
- point-like and finite-size nuclei
- arbitrary nuclear form factors

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
>>> from src.nuclei.form_factors import HelmFormFactor
>>> from src.interactions.particles import WIMP

>>> wimp = WIMP(mass=50e9, spin=0)

>>> kin = ElasticKinematics(mX=wimp.mass, mT=Xe131.mass)

>>> interaction = SIInteraction(
...     wimp=wimp,
...     nucleus=Xe131,
...     kinematics=kin,
...     sigma_p=1e-45,
...     form_factor=HelmFormFactor(Xe131),
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
from src.nuclei.form_factors import FormFactor, PointLikeFormFactor


class SIInteraction(InteractionModel):
    """
    Spin-independent (SI) WIMP-nucleus interaction.

    The interaction is normalized to the WIMP-proton
    cross section sigma_p and supports generic
    proton and neutron couplings through the ratio

        fn_over_fp = fn / fp.

    The nuclear cross section is

        sigma_A 
        =
        sigma_p * (mu_A / mu_p)^2
        * [Z + (A - Z) fn_over_fp]^2

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

        dσ/dER (v) = m_A sigma_A / (2 mu_A^2 v^2) F^2(q)

    where:
        ER
            Nuclear recoil energy.
        v
            Incoming WIMP speed.
        F(q)
            Nuclear form factor.

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

    fn_over_fp : float, optional
        Ratio between neutron and proton
        couplings.

        Default is 1 (isospin-conserving).

    form_factor : FormFactor, optional
        Nuclear form factor.

        If omitted, a PointLikeFormFactor
        is used (F(q) = 1).
    """

    def __init__(
        self,
        *,
        wimp: WIMP,
        nucleus: Nucleus,
        kinematics: ScatteringKinematics,
        form_factor: FormFactor = None,
        sigma_p: float = 1e-45,
        fn_over_fp: float = 1.):

        super().__init__(
            wimp=wimp,
            nucleus=nucleus,
            kinematics=kinematics)

        if form_factor is None:
            form_factor = PointLikeFormFactor()

        self.form_factor = form_factor
        self.sigma_p = sigma_p
        self.fn_over_fp = fn_over_fp

    def dsigma_dER(self, v: NDArray, ER: NDArray) -> NDArray:
        """
        Differential spin-independent cross section.

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
        - coherent nuclear enhancement
        - reduced-mass scaling
        - nuclear form factor suppression
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

        kinematic_prefactor = np.where(
            v_grid > 0,
            self.nucleus.mass / (2 * self._reduced_mass_nucleus**2 * v_grid**2),
            0.0)

        q = self.kin.q_from_ER(ER=ER_grid)
        F2 = self.form_factor.response(q=q)

        dsigma_unmasked = (kinematic_prefactor * self.sigma_nucleus * F2)

        return self._apply_mask_to_dsigmadER(
            dsigmadER=dsigma_unmasked, v=v, ER=ER)

    @property
    def coherent_coupling(self) -> float:
        """
        Effective coherent nuclear coupling.

        Notes
        -----
        The coherent enhancement factor is
        Z + (A - Z) fn/fp

        and reduces to A
        for isospin-conserving interactions
        (fn/fp = 1).
        """
        return self.nucleus.Z + self.nucleus.N * self.fn_over_fp

    @property
    def sigma_nucleus(self) -> float:
        """
        Effective WIMP-nucleus total cross section.

        Returns
        -------
        float
            Effective WIMP-nucleus cross section in cm².

        Notes
        -----
        The nuclear cross section is

        sigma_A
        =
        sigma_p *(mu_A / mu_p)^2
        * [Z + (A - Z) fn/fp]^2.

        For isospin-conserving interactions,

        sigma_A = sigma_p * (mu_A / mu_p)^2 *A^2.
        """
        mu_ratio2 = (self._reduced_mass_nucleus /self._reduced_mass_proton)**2
        return self.sigma_p *  mu_ratio2 * self.coherent_coupling**2
