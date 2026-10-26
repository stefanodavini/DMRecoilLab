"""
High-level sensitivity calculations for
spin-independent WIMP scattering.

This module provides convenience functions that
combine the core DMRecoilLab APIs
- kinematics,
- interaction models,
- rate calculations,
- statistical sensitivity estimators,
to compute exclusion sensitivities as a function
of WIMP mass.

The functions in this module are intended for
physics analyses and notebooks rather than as
low-level building blocks.
"""

from src.halo.base_halo_distribution import IsotropicSpeedDistribution
from src.halo.shm import StandardHaloModel
from src.detectors.detectors import IdealDetector
from src.sensitivity.poisson import PoissonUpperLimit
from src.sensitivity.converter import sigma_upper_limit
from src.interactions.particles import WIMP
from src.interactions.si import SIInteraction
from src.nuclei.form_factors import FormFactor, HelmFormFactor
from src.utils.factory import create_kinematics
from src.rates import RateCalculator

from numpy.typing import NDArray, ArrayLike
import numpy as np

def compute_mass_sigma_limit_si(
    masses: ArrayLike,
    detector: IdealDetector,
    delta: float = 0.0,
    halo: IsotropicSpeedDistribution | None = None,
    form_factor: FormFactor | None = None,
    cl: float = 0.9,
    sigma_reference: float = 1e-45,
    ) -> NDArray:
    """
    Compute a spin-independent WIMP-proton sensitivity curve.

    Parameters
    ----------
    masses: NDarray
        WIMP-mass grid in eV.
    detector: IdealDetector
        Detector configuration.
    delta: float
        Inelastic mass splitting in eV.
        If zero, uses elasic kinematics.
    halo: IsotropicSpeedDistribution
        Halo speed distribution.
        If None, uses SHM in Earth's frame.
    form_factor: FormFactor
        Nucleus form factor.
        If None, uses the Helm form factor.
    cl: float
        Confidence level.
    sigma_reference
        Reference cross section used
        to compute expected counts.

    Returns
    -------
    NDarray
        sigma_limits
        Upper limit on sigma_reference
        over the input WIMP-mass grid.

    Notes
    -----
    The sensitivity is computed assuming a counting
    experiment with the statistical prescription
    implemented by PoissonUpperLimit.

    For each WIMP mass, the function constructs
    - a WIMP instance,
    - the appropriate scattering kinematics,
    - a spin-independent interaction model,
    - a RateCalculator,

    and converts the resulting signal-count limit
    into an upper limit on the reference cross section.

    For SI interactions the reference
    cross section coincides with sigma_p.
    """
    masses = np.asarray(masses)

    if halo is None:
        halo = StandardHaloModel().boost()

    if form_factor is None:
        form_factor = HelmFormFactor(nucleus=detector.nucleus)
    
    statistic = PoissonUpperLimit(cl=cl)
    sigma_limits = []
    
    for mass in masses:
    
        wimp = WIMP(mass=mass, spin=0)
    
        kin = create_kinematics(wimp=wimp, nucleus=detector.nucleus,
                                delta=delta)
    
        interaction = SIInteraction(wimp=wimp,nucleus=detector.nucleus,
                                    kinematics=kin,
                                    form_factor=form_factor,
                                    sigma_p=sigma_reference)
    
        rate = RateCalculator(halo=halo,
                              interaction=interaction,
                              detector=detector)
    
        try:
            point = sigma_upper_limit(
                    statistic=statistic,
                    rate_calculator=rate)
    
            sigma_limits.append(point.sigma_reference_upper)
    
        except ValueError:
            sigma_limits.append(np.nan)
    
    return np.asarray(sigma_limits)
