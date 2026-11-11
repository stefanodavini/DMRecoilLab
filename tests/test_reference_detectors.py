import numpy as np
import pytest

from src.detectors.detectors import IdealDetector, ResponseDetector
from src.detectors.reference_detectors import (
    DEAP_2026_RESPONSE,
    LZ_2024,
    LZ_2026_RESPONSE,
)


def test_lz_2024_ideal_reference_is_preserved():
    assert isinstance(LZ_2024, IdealDetector)
    assert LZ_2024.mass_kg == 5800
    assert LZ_2024.exposure_days == 280


def test_lz_2026_response_configuration_and_acceptance():
    assert isinstance(LZ_2026_RESPONSE, ResponseDetector)
    assert LZ_2026_RESPONSE.mass_kg == 4710
    assert LZ_2026_RESPONSE.exposure_days == 220
    #assert LZ_2026_RESPONSE.target_mass_fraction == pytest.approx(0.21)
    assert LZ_2026_RESPONSE.has_acceptance
    assert LZ_2026_RESPONSE.has_resolution
    assert LZ_2026_RESPONSE.resolution_sigma == 1300

    energy = np.array([14e3, 100e3, 269.9e3])
    acceptance = LZ_2026_RESPONSE._apply_acceptance(
        energy, np.ones_like(energy)
    )
    np.testing.assert_allclose(
        acceptance, [0.944187, 0.955805, 0.498860], rtol=1e-4
    )


def test_deap_2026_response_configuration_and_acceptance():
    assert isinstance(DEAP_2026_RESPONSE, ResponseDetector)
    assert DEAP_2026_RESPONSE.mass_kg == 1266
    assert DEAP_2026_RESPONSE.exposure_days == 790.8
    assert DEAP_2026_RESPONSE.has_acceptance
    assert DEAP_2026_RESPONSE.has_resolution
    assert DEAP_2026_RESPONSE.resolution_sigma == 3e3

    energy = np.array([90e3 / 6.1, 150e3 / 6.1, 200e3 / 6.1])
    acceptance = DEAP_2026_RESPONSE._apply_acceptance(
        energy, np.ones_like(energy)
    )
    np.testing.assert_allclose(acceptance, [0.45, 0.61, 0.64])
