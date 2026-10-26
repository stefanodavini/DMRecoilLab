import numpy as np
import pytest

from src.detectors.detectors import IdealDetector
from src.nuclei.isotopes import Xe131
from src.utils.constants import DAY_TO_SECONDS
from src.utils.conversions import convert_from_kg_to_eV


def make_detector(**kwargs):
    return IdealDetector(nucleus=Xe131, mass_kg=10.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [

        {"mass_kg": -1},
        {"mass_kg": np.nan},
        {"mass_kg": np.inf},
        {"target_mass_fraction": -1},
        {"target_mass_fraction": 1.1},
        {"target_mass_fraction": np.nan},
        {"exposure_days": -1},
        {"exposure_days": np.inf},
        {"threshold_energy": -1},
        {"threshold_energy": np.nan},
        {"threshold_energy": np.inf},
    ],
)
def test_detector_invalid_parameters_raise(kwargs):
    params = {"mass_kg": 10.0, **kwargs}

    with pytest.raises(ValueError):
        IdealDetector(nucleus=Xe131, **params)


def test_detector_default_values():
    detector = make_detector()

    assert detector.exposure_days == 365
    assert detector.threshold_energy == 0.0
    assert detector.target_mass_fraction == 1.0


def test_detector_number_of_targets_accounts_for_target_mass_fraction():
    detector = make_detector(target_mass_fraction=0.25)

    expected = convert_from_kg_to_eV(10.0 * 0.25) / Xe131.mass
    assert detector.number_of_targets == pytest.approx(expected)


def test_exposure_seconds():
    detector = make_detector(exposure_days=2)

    assert detector.exposure_seconds == pytest.approx(2 * DAY_TO_SECONDS)