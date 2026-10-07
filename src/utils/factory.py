"""
Factory functions.

This module provides convenience helpers that select the
appropriate class instances based on
high-level physical parameters.
"""

from src.interactions.particles import WIMP
from src.nuclei.nucleus import Nucleus

from src.kinematics import ScatteringKinematics, ElasticKinematics, InelasticKinematics

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