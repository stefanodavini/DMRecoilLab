from src.sensitivity.converter import sigma_upper_limit
from src.sensitivity.poisson import PoissonUpperLimit
from src.sensitivity.types import CountingExperiment

from src.halo import StandardHaloModel
from src.kinematics import ElasticKinematics
from src.interactions.si import SIInteraction
from src.interactions.particles import WIMP
from src.nuclei.nucleus import Nucleus
from src.nuclei.isotopes import Ar40, Xe131
from src.detectors.detectors import IdealDetector
from src.rates import RateCalculator

import numpy as np
import pytest


def _initialize_default_rate_calculator(target: Nucleus,
                                        sigma_p=1e-45,
                                        ) -> RateCalculator:
    shm = StandardHaloModel().boost()
    wimp = WIMP(mass=1e11, spin=0)
    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    si = SIInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       sigma_p=sigma_p)
    
    detector = IdealDetector(nucleus=target, mass_kg=1e3,
                             exposure_days=365,
                             threshold_energy=10e3)
    
    return RateCalculator(wimp=wimp, halo=shm, interaction=si, detector=detector)

def test_sigma_upper_limit_returns_correct_mass():
    limit = PoissonUpperLimit(cl=0.9)
    rc = _initialize_default_rate_calculator(target=Ar40)
    result = sigma_upper_limit(statistic=limit, rate_calculator=rc)

    assert result.mass_wimp == rc.mass_wimp

def test_sigma_upper_limit_is_positive():
    limit = PoissonUpperLimit(cl=0.95)
    rc = _initialize_default_rate_calculator(target=Xe131)
    result = sigma_upper_limit(statistic=limit, rate_calculator=rc)

    assert result.sigma_p_upper > 0

def test_sigma_upper_limit_reference_sigma_independence():
    """
    Chech the upper-limit result to be independent 
    of the arbitrary WIMP-proton cross section reference choice.
    """
    limit = PoissonUpperLimit(cl=0.90)
    rc1 = _initialize_default_rate_calculator(target=Xe131, sigma_p=1e-45)
    rc2 = _initialize_default_rate_calculator(target=Xe131, sigma_p=2e-47)

    result1 = sigma_upper_limit(statistic=limit, rate_calculator=rc1)
    result2 = sigma_upper_limit(statistic=limit, rate_calculator=rc2)

    assert np.isclose(result1.sigma_p_upper, result2.sigma_p_upper, rtol=1e-12)

def test_sigma_upper_limit_default_experiment_matches_null_experiment():
    limit = PoissonUpperLimit(cl=0.95)
    rc = _initialize_default_rate_calculator(target=Ar40)
    experiment=CountingExperiment(counts=0, background=0)
    result_default = sigma_upper_limit(statistic=limit, rate_calculator=rc)
    result_null = sigma_upper_limit(statistic=limit, rate_calculator=rc,
                                    experiment=experiment)

    assert np.isclose(result_default.sigma_p_upper,
                      result_null.sigma_p_upper)

def test_sigma_upper_limit_raises_zero_expected_count():
    shm = StandardHaloModel().boost()
    wimp = WIMP(mass=1e11, spin=0)
    kin = ElasticKinematics(mX=wimp.mass, mT=Xe131.mass)
    si = SIInteraction(wimp=wimp, nucleus=Xe131,
                       kinematics=kin)

    high_threshold = 1e6 #eV
    detector = IdealDetector(nucleus=Xe131, mass_kg=1e3,
                             exposure_days=365,
                             threshold_energy=high_threshold)
    
    rc0 = RateCalculator(wimp=wimp, halo=shm,
                        interaction=si, detector=detector)

    limit = PoissonUpperLimit(cl=0.95)

    with pytest.raises(ValueError):
        sigma_upper_limit(statistic=limit, rate_calculator=rc0)

def test_sigma_upper_limit_nobkg_linearity_exposure():
    """
    Sensitivity should improve inversely with exposure.
    """
    shm = StandardHaloModel().boost()
    wimp = WIMP(mass=1e11, spin=0)
    kin = ElasticKinematics(mX=wimp.mass, mT=Xe131.mass)
    si = SIInteraction(wimp=wimp, nucleus=Xe131,
                       kinematics=kin,
                       sigma_p=1)

    limit = PoissonUpperLimit(cl=0.95)

    years = [1, 2]
    sigma_lims =[]
    for y in years:
        days = 365 * y

        detector = IdealDetector(nucleus=Xe131, mass_kg=1e3,
                                 exposure_days=days,
                                 threshold_energy=3e3)
        rc0 = RateCalculator(wimp=wimp, halo=shm,
                             interaction=si, detector=detector)

        result =  sigma_upper_limit(statistic=limit, rate_calculator=rc0)
        sigma_lims.append(result.sigma_p_upper)

    ratio = sigma_lims[0] / sigma_lims[1]
    assert np.isclose(2, ratio, rtol=1e-6)





