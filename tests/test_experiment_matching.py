from src.halo.shm import StandardHaloModel
from src.interactions.particles import WIMP
from src.nuclei.isotopes import Ar40, Xe131
from src.detectors.detectors import BaseDetector, IdealDetector
from src.detectors.reference_detectors import DS50_2018, XENON100_2012
from src.utils.factory import create_interaction_si
from src.rates import RateCalculator

from src.analysis.experiment_matching import predict_counts_from_counts_match
from src.analysis.experiment_matching import predict_counts_from_counts_match_same_target
from src.analysis.experiment_matching import predict_counts_from_counts_match_different_target
from src.analysis.experiment_matching import matched_counts_grid_si

import pytest
import numpy as np


def _initialize_halo_shm_boost():
    return StandardHaloModel().boost()

def _initialize_default_si_elastic_rate(detector: BaseDetector,
                                        sigma_p=1e-45,
                                        ) -> RateCalculator:
    target = detector.nucleus
    wimp = WIMP(mass=1e11, spin=0)
    si = create_interaction_si(wimp=wimp, nucleus=target, sigma_p=sigma_p)
    halo = _initialize_halo_shm_boost()
    return RateCalculator(halo=halo, interaction=si, detector=detector)

def test_experiment_matching_invalid_counts_raise():
    rcA = _initialize_default_si_elastic_rate(detector=XENON100_2012)
    counts = [np.inf, -np.inf, np.nan, 0, -1.]
    for c in counts:
        with pytest.raises(ValueError, match="finite and positive"):
                predict_counts_from_counts_match(
                     rate_calculator_a=rcA,
                     counts_a=c,
                     detector_b=XENON100_2012)

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

    si_b = create_interaction_si(wimp=wimp_b, nucleus=Ar40)

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


def test_experiment_matched_counts_grid_shape():
    masses = [1e10, 1e11]
    deltas = [0, 1e5, 2e5]

    grid = matched_counts_grid_si(wimp_masses=masses,
                                  deltas=deltas,
                                  counts_a=1,
                                  detector_a=XENON100_2012,
                                  detector_b=DS50_2018)

    assert grid.shape == (2, 3)
    assert np.all(grid.masses == masses)
    assert np.all(grid.deltas == deltas)

def test_experiment_matched_counts_identity_detector():
    masses = [1e11, 1e12]
    deltas = [0, 1e5]
    count = 7

    grid = matched_counts_grid_si(wimp_masses=masses,
                                  deltas=deltas,
                                  counts_a=count,
                                  detector_a=XENON100_2012,
                                  detector_b=XENON100_2012)

    assert np.allclose(grid.counts_b, count)

def test_experiment_matched_counts_grid_sigma_forwarding():
    """
    Verifies that the matching procedure is independent
    of the arbitrary reference cross section normalization.
    """
    masses = [1e11]
    deltas = [0]

    grid1 = matched_counts_grid_si(
        wimp_masses=masses,
        deltas=deltas,
        counts_a=1,
        detector_a=XENON100_2012,
        detector_b=DS50_2018,
        interaction_a_kwargs={"sigma_p": 1e-45},
        interaction_b_kwargs={"sigma_p": 1e-45},
    )

    grid2 = matched_counts_grid_si(
        wimp_masses=masses,
        deltas=deltas,
        counts_a=1,
        detector_a=XENON100_2012,
        detector_b=DS50_2018,
        interaction_a_kwargs={"sigma_p": 1.},
        interaction_b_kwargs={"sigma_p": 1.},
    )

    assert np.allclose(grid1.counts_b, grid2.counts_b)
