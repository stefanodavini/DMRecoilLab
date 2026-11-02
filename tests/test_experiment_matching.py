from src.halo.shm import StandardHaloModel
from src.kinematics.scattering_kinematics import ElasticKinematics
from src.interactions.si import SIInteraction
from src.interactions.particles import WIMP
from src.nuclei.nucleus import Nucleus
from src.nuclei.isotopes import Ar40, Xe131
from src.detectors.detectors import BaseDetector, IdealDetector
from src.detectors.reference_detectors import DS50_2018, XENON100_2012
from src.rates import RateCalculator

from src.analysis.experiment_matching import predict_counts_from_counts_match
from src.analysis.experiment_matching import predict_counts_from_counts_match_same_target
from src.analysis.experiment_matching import predict_counts_from_counts_match_different_target

import pytest
import numpy as np

def _initialize_si_elastic(wimp: WIMP,
                           target: Nucleus,
                           sigma_p=1e-45) -> SIInteraction:

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    si = SIInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       sigma_p=sigma_p)
    return si

def _initialize_halo_shm_boost():
    shm = StandardHaloModel()
    return shm.boost()

def _initialize_default_si_elastic_rate(detector: BaseDetector,
                                        sigma_p=1e-45,
                                        ) -> RateCalculator:
    target = detector.nucleus
    wimp = WIMP(mass=1e11, spin=0)
    si = _initialize_si_elastic(wimp=wimp, target=target, sigma_p=sigma_p)
    halo = _initialize_halo_shm_boost()
    return RateCalculator(halo=halo, interaction=si, detector=detector)

def test_experiment_matching_counts_missing_interaction_different_target():
    rcA = _initialize_default_si_elastic_rate(detector=XENON100_2012)
    with pytest.raises(ValueError, match="required"):
        predict_counts_from_counts_match(rate_calculator_a=rcA,
                                         counts_a=10,
                                         detector_b=DS50_2018)

def test_experiment_matching_counts_wimp_mass_mismatch():
    rcA = _initialize_default_si_elastic_rate(detector=XENON100_2012)

    wimp_b = WIMP(mass=10 * rcA.wimp.mass + 1,
                  spin=0)

    kin_b = ElasticKinematics(mX=wimp_b.mass, mT=Ar40.mass)
    si_b = SIInteraction(wimp=wimp_b, nucleus=Ar40,
                         kinematics=kin_b)

    with pytest.raises(ValueError, match="same WIMP mass"):
        predict_counts_from_counts_match(rate_calculator_a=rcA,
                                         counts_a=10,
                                         detector_b=DS50_2018,
                                         interaction_b=si_b)

def test_experiment_matching_counts_same_target_mismatch():
    rcA = _initialize_default_si_elastic_rate(detector=XENON100_2012)
    with pytest.raises(ValueError, match="nuclei"):
        predict_counts_from_counts_match_same_target(
            rate_calculator_a=rcA,
            counts_a=10,
            detector_b=DS50_2018)

def test_experiment_matching_counts_detector_interaction_mismatch():
    rcA = _initialize_default_si_elastic_rate(detector=XENON100_2012)
    with pytest.raises(ValueError, match="identical"):
        predict_counts_from_counts_match_different_target(
            rate_calculator_a=rcA,
            counts_a=10,
            interaction_b=rcA.interaction,
            detector_b=DS50_2018)

def test_experiment_matching_counts_same_detector_identity():
        rcA = _initialize_default_si_elastic_rate(sigma_p=1,
                                                  detector=XENON100_2012)

        countsA = rcA.expected_counts()
        countsB = predict_counts_from_counts_match(
            rate_calculator_a=rcA,
            counts_a=countsA,
            detector_b=XENON100_2012)

        ratio = countsB / countsA
        assert np.isclose(ratio, 1, rtol=1e-6)

def test_experiment_matching_counts_same_detector_exposure_linearity():
        detectorA = IdealDetector(nucleus=Xe131,
                                  mass_kg=50,
                                  exposure_days=100)
        detectorB = IdealDetector(nucleus=Xe131,
                                  mass_kg=50,
                                  exposure_days=200)
        
        rcA = _initialize_default_si_elastic_rate(sigma_p=1,
                                                  detector=detectorA)

        countsA = rcA.expected_counts()
        countsB = predict_counts_from_counts_match(
            rate_calculator_a=rcA,
            counts_a=countsA,
            detector_b=detectorB)
        
        ratio = countsB / countsA
        assert np.isclose(ratio, 2, rtol=1e-6)

def test_experiment_matching_counts_nan_propagation():
    detectorA = IdealDetector(nucleus=Xe131, mass_kg=0)
    rcA = _initialize_default_si_elastic_rate(detector=detectorA)

    with pytest.warns(RuntimeWarning, match="zero"):
        countsB = predict_counts_from_counts_match(
                rate_calculator_a=rcA,
                counts_a=10,
                detector_b=detectorA)

    assert np.isnan(countsB)
