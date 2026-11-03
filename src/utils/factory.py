"""
Factory functions.

This module provides convenience helpers that select the
appropriate class instances based on
high-level physical parameters.
"""

from src.interactions.particles import WIMP
from src.interactions.propagators import Propagator
from src.nuclei.nucleus import Nucleus

from src.kinematics.scattering_kinematics import ScatteringKinematics, ElasticKinematics, InelasticKinematics
from src.interactions.si import SIInteraction
from src.nuclei.form_factors import FormFactor, HelmFormFactor

def create_kinematics(
    wimp: WIMP,
    nucleus: Nucleus,
    delta: float = 0.0) -> ScatteringKinematics:
    """
    Create a kinematics model for WIMP-nucleus scattering.

    Parameters
    ----------
    wimp: WIMP
        Incident WIMP particle.
    nucleus : Nucleus
        Target nucleus.
    delta : float
        Mass splitting in eV.

        A value of zero returns an ElasticKinematics instance.
        Non-zero values return an InelasticKinematics instance.

    Returns
    -------
    ScatteringKinematics
        Kinematics model configured for the requested scattering
        scenario.
    """
    if delta == 0:
        return ElasticKinematics(
            mX=wimp.mass,
            mT=nucleus.mass)

    return InelasticKinematics(
        mX=wimp.mass,
        mT=nucleus.mass,
        delta=delta)


def create_interaction_si(
    wimp: WIMP,
    nucleus: Nucleus,
    delta: float = 0.0,
    form_factor: FormFactor | None = None,
    propagator: Propagator | None = None,
    sigma_p: float = 1e-45,
    fn_over_fp: float = 1,
) -> SIInteraction:
    """
    Create a spin-independent WIMP-nucleus interaction model.

    Parameters
    ----------
    wimp : WIMP
        Incident WIMP particle.

    nucleus : Nucleus
        Target nucleus.

    delta : float, optional
        Mass splitting in eV.
        A value of zero produces an elastic interaction.
        Non-zero values produce an inelastic interaction.

    form_factor : FormFactor, optional
        Nuclear form factor model.
        If None, a HelmFormFactor instance is created
        automatically for the specified nucleus.

    propagator : Propagator, optional
        Mediator propagator model.
        If None, the default propagator behavior of
        SIInteraction is used.

    sigma_p : float, optional
        Reference WIMP-proton cross section in cm^2.

    fn_over_fp : float, optional
        Ratio of neutron and proton couplings.

    Returns
    -------
    SIInteraction
        Spin-independent interaction model configured
        with kinematics appropriate for the specified
        WIMP, target nucleus, and mass splitting.

    Notes
    -----
    This is a convenience factory that automatically

    - constructs the appropriate scattering kinematics
      (elastic or inelastic),
    - creates a Helm form factor when none is supplied,
    - returns a fully configured SIInteraction instance.
    """

    if form_factor is None:
        form_factor = HelmFormFactor(nucleus=nucleus)

    kin = create_kinematics(wimp=wimp, nucleus=nucleus, delta=delta)

    return SIInteraction(wimp=wimp,
                         nucleus=nucleus,
                         kinematics=kin,
                         form_factor=form_factor,
                         propagator=propagator,
                         sigma_p=sigma_p,
                         fn_over_fp=fn_over_fp)
