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
    with pytest.raises(ValueError, match="non-negative"):
        ResponseDetector(nucleus=Ar40, mass_kg=1.0, acceptance_curve=curve)

def test_detectors_acceptance_must_be_less_than_one():
    curve = TabulatedData(x=[1.0, 10.0], y=[0.0, 1.2])
    with pytest.raises(ValueError, match="<= 1"):
        ResponseDetector(nucleus=Ar40, mass_kg=1.0, acceptance_curve=curve)

def test_detector_response_default_identity():
    """
    Check that the default value of ResponseDetector
    leaves the spectrum unchanged
    """
    detector = ResponseDetector(nucleus=Ar40,mass_kg=1.0)

    energy = np.array([1.0, 2.0, 3.0])
    spectrum = np.array([10.0, 20.0, 30.0])

    result = detector.apply_detector_response(energy=energy,
                                              spectrum=spectrum)

    assert not detector.has_resolution
    assert not detector.has_acceptance
    assert np.all(result == spectrum)

def test_detector_resolution_sigma_default_identity():
    """
    Check that the default value of resolution_sigma
    leaves the spectrum unchanged
    """
    detector = ResponseDetector(nucleus=Ar40,mass_kg=1.0)

    energy = np.array([1.0, 2.0, 3.0])
    spectrum = np.array([10.0, 20.0, 30.0])

    result = detector._apply_resolution(energy=energy,
                                        spectrum=spectrum)

    assert not detector.has_resolution
    assert np.all(result == spectrum)

def test_detector_resolution_conserves_counts():
    """
    Check that the counts are conserved
    when applying the integration kernel.

    Notes
    -----
    The relative tolerance of the check can improve
    by increasing the spacing of the energy grid.
    We keep only 1% tolerance to avoid the test to last long.

    Manual checks have verified the tolerance improves
    when the energy grid is finer.
    """
    detector = ResponseDetector(nucleus=Ar40, mass_kg=1.0,
                                resolution_sigma=5.0)

    energy = np.linspace(0, 100, 1000)

    spectrum_true = np.exp(- energy / 20)
    cnt_before = np.trapezoid(spectrum_true, energy)

    spectrum_obs = detector._apply_resolution(energy=energy,
                                              spectrum=spectrum_true)
    cnt_after = np.trapezoid(spectrum_obs, energy)

    assert np.isclose(cnt_before, cnt_after, rtol=1e-2)

def test_detector_resolution_to_delta_peak():
    """
    Check the effect on a delta-like peak:
    broadening and symmetry.
    """
    detector = ResponseDetector(nucleus=Ar40, mass_kg=1.0,
                                resolution_sigma=2.0)

    energy = np.linspace(0, 100, 101)

    spectrum = np.zeros_like(energy)
    spectrum[50] = 1.0

    smeared = detector._apply_resolution(energy=energy,
                                         spectrum=spectrum)

    assert detector.has_resolution == 1
    assert smeared[50] < spectrum[50]
    assert smeared[51] > 0.
    assert smeared[51] < smeared[50]
    assert smeared[49] > 0.
    assert smeared[49] < smeared[50]
    assert smeared[52] < smeared[51]
    assert smeared[48] < smeared[49]

    # check symmetry
    np.testing.assert_allclose(smeared[30:50], smeared[51:71][::-1], rtol=1e-3)

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

    assert not detector.has_acceptance
    assert np.all(result == spectrum)

def test_detector_apply_resolution_does_not_modify_input():
    """
    Check that _apply_resolution leaves input spectrum unmodified,
    as we explicitly adopted that design contract.
    """

    detector = ResponseDetector(nucleus=Ar40, mass_kg=1.0,
                                resolution_sigma=10.)

    energy = np.array([0.0, 50.0, 100.0])
    spectrum = np.array([30.0, 20.0, 10.0,])
    original = spectrum.copy()

    detector._apply_resolution(energy, spectrum)
    assert np.all(spectrum == original)

def test_detectors_response_apply_acceptance():
    curve = TabulatedData(x=[0.0, 100.0],
                          y=[0.5, 1.0])

    detector = ResponseDetector(nucleus=Ar40, mass_kg=1.0,
                                acceptance_curve=curve)
    
    assert detector.has_acceptance

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


def test_detector_response_ordering():
    """
    Check the order of apply_detector_response is:
    - apply_resolution
    - apply_acceptance.
    """
    acceptance = TabulatedData(x=[20.0, 40.0],
                               y=[0.0, 1.0])
    detector = ResponseDetector(nucleus=Ar40, mass_kg=1.0,
                                resolution_sigma=2.0,
                                acceptance_curve=acceptance)

    assert detector.has_resolution
    assert detector.has_acceptance

    energy = np.linspace(0, 100, 100)
    spectrum_true = np.exp(- energy / 20)

    spectrum_resp = detector.apply_detector_response(energy=energy,
                                                     spectrum=spectrum_true)

    spectrum_wres = detector._apply_resolution(energy=energy,
                                               spectrum=spectrum_true)

    spectrum_wrac = detector._apply_acceptance(energy=energy,
                                               spectrum=spectrum_wres)

    assert np.array_equiv(spectrum_resp, spectrum_wrac)
    assert np.all(spectrum_resp == spectrum_wrac)


    
