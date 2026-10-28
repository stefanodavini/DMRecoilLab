import numpy as np
import pytest

from src.nuclei.isotopes import Ar40
from src.detectors.detectors import ResponseDetector
from src.utils.tabulated_data import TabulatedData


@pytest.mark.parametrize(
    "sigma",
    [np.nan, np.inf, -np.inf]
)
def test_detectors_resolution_sigma_must_be_finite(sigma):
    with pytest.raises(ValueError, match="finite"):
        ResponseDetector(nucleus=Ar40, mass_kg=1.0, resolution_sigma=sigma)

def test_detectors_resolution_sigma_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        ResponseDetector(nucleus=Ar40, mass_kg=1.0, resolution_sigma=-1.0)

def test_detectors_acceptance_energy_must_be_non_negative():
    curve = TabulatedData(x=[-1.0, 10.0], y=[0.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        ResponseDetector(nucleus=Ar40, mass_kg=1.0, acceptance_curve=curve)

def test_detectors_acceptance_must_be_non_negative():
    curve = TabulatedData(x=[1.0, 10.0], y=[0.0, -1.0])
    with pytest.raises(ValueError, match=">= 0"):
        ResponseDetector(nucleus=Ar40, mass_kg=1.0, acceptance_curve=curve)

def test_detectors_acceptance_must_be_less_than_one():
    curve = TabulatedData(x=[1.0, 10.0], y=[0.0, 1.2])
    with pytest.raises(ValueError, match="<= 1"):
        ResponseDetector(nucleus=Ar40, mass_kg=1.0, acceptance_curve=curve)

def test_detector_acceptance_curve_default_identity():
    """
    Check that the default value of acceptance_curve
    leaves the spectrum unchanged
    """
    detector = ResponseDetector(nucleus=Ar40,mass_kg=1.0)

    energy = np.array([1.0, 2.0, 3.0])
    spectrum = np.array([10.0, 20.0, 30.0])

    result = detector._apply_acceptance(energy=energy,
                                        spectrum=spectrum)

    assert detector.has_acceptance == 0
    np.testing.assert_allclose(result, spectrum, atol=1e-12)

def test_detectors_response_apply_acceptance():
    curve = TabulatedData(x=[0.0, 100.0],
                          y=[0.5, 1.0])

    detector = ResponseDetector(nucleus=Ar40, mass_kg=1.0,
                                acceptance_curve=curve)
    
    assert detector.has_acceptance == 1

    energy = np.array([0.0, 50.0, 100.0])
    spectrum = np.array([10.0, 10.0, 10.0])
    result = detector._apply_acceptance(energy=energy,
                                        spectrum=spectrum)
    
    expected = np.array([5.0, 7.5, 10.0])
    np.testing.assert_allclose(result, expected, atol=1e-6)

def test_detectors_acceptance_below_range_is_zero():
    curve = TabulatedData(x=[10.0, 20.0],
                          y=[0.5, 1.0])

    detector = ResponseDetector(nucleus=Ar40,
                                mass_kg=1.0,
                                acceptance_curve=curve)

    energy = np.array([5.0])
    spectrum = np.array([10.0])

    result = detector._apply_acceptance(energy=energy,
                                        spectrum=spectrum)

    assert result[0] == 0.

def test_detectors_acceptance_above_range_is_last_value():
    last_value = 0.99
    curve = TabulatedData(x=[10.0, 20.0],
                          y=[0.5, last_value])

    detector = ResponseDetector(nucleus=Ar40,
                                mass_kg=1.0,
                                acceptance_curve=curve)

    energy = np.array([25.0])
    spectrum = np.array([3e-3])

    result = detector._apply_acceptance(energy=energy,
                                        spectrum=spectrum)

    assert np.isclose(result[0], last_value * spectrum[0], rtol=1e-9)

def test_detector_apply_acceptance_does_not_modify_input():
    """
    Check that _apply_acceptance leaves input spectrum unmodified,
    as we explicitly adopted that design contract.
    """
    curve = TabulatedData(x=[0.0, 100.0],
                          y=[0.5, 1.0])

    detector = ResponseDetector(nucleus=Ar40, mass_kg=1.0,
                                acceptance_curve=curve)

    energy = np.array([0.0, 50.0, 100.0])
    spectrum = np.array([10.0, 20.0, 30.0,])
    original = spectrum.copy()

    detector._apply_acceptance(energy, spectrum)
    assert np.all(spectrum == original)

