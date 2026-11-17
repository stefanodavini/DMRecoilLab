import numpy as np
import pytest

from src.detectors.detectors import IdealDetector, ResponseDetector
from src.detectors.reference_detectors import (
    DEAP3600_2026_RESPONSE,
    DS20K_10YEAR_RESPONSE,
    DS50_2018_RESPONSE,
    LZ_2024,
    LZ_2026_RESPONSE,
    XENONnT_2024_RESPONSE,
    _lz_acceptance,
)


def test_lz_2024_ideal_reference_is_preserved():
    assert isinstance(LZ_2024, IdealDetector)
    assert LZ_2024.mass_kg == 5800
    assert LZ_2024.exposure_days == 280


def test_lz_2026_response_configuration_and_acceptance():
    assert isinstance(LZ_2026_RESPONSE, ResponseDetector)
    assert LZ_2026_RESPONSE.mass_kg == 4710
    assert LZ_2026_RESPONSE.exposure_days == 220
    assert LZ_2026_RESPONSE.has_acceptance
    assert LZ_2026_RESPONSE.has_resolution
    assert LZ_2026_RESPONSE.resolution_sigma == 1300

    # Include 250 keV near the signal-hint energy to guard the high-energy response.
    energy = np.array([0.0, 14e3, 100e3, 250e3, 269.9e3, 1e9])
    acceptance = LZ_2026_RESPONSE._apply_acceptance(
        energy, np.ones_like(energy)
    )
    np.testing.assert_allclose(
        acceptance, [0.0, 0.944187, 0.955805, 0.937306, 0.498860, 0.0],
        rtol=1e-4, atol=0.0,
    )

    he_sb_energy = np.array([302e3, 700e3, 701e3])
    he_sb_acceptance = LZ_2026_RESPONSE._apply_acceptance(
        he_sb_energy, np.ones_like(he_sb_energy)
    )
    np.testing.assert_allclose(he_sb_acceptance, [1.0, 1.0, 0.0])


def test_lz_acceptance_optional_he_sideband():
    with_he_sb = _lz_acceptance(include_he_sb=True)
    without_he_sb = _lz_acceptance(include_he_sb=False)
    default = _lz_acceptance()

    np.testing.assert_array_equal(default.x, with_he_sb.x)
    np.testing.assert_array_equal(default.y, with_he_sb.y)
    # Both choices retain the complete WS curve through 300 keVnr.
    ws_points = with_he_sb.x <= 300e3
    np.testing.assert_array_equal(without_he_sb.x[:-1], with_he_sb.x[ws_points])
    np.testing.assert_array_equal(without_he_sb.y[:-1], with_he_sb.y[ws_points])


@pytest.mark.parametrize(
    ("include_he_sb", "expected"),
    [
        (True, [0.937306, 0.001609, 0.0008045 + 0.5, 1.0, 1.0, 0.5, 0.0, 0.0]),
        (False, [0.937306, 0.0008045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    ids=["with-he-sideband", "without-he-sideband"],
)
def test_lz_acceptance_sideband_interpolation_and_cutoff(include_he_sb, expected):
    detector = ResponseDetector(
        nucleus=LZ_2026_RESPONSE.nucleus,
        mass_kg=LZ_2026_RESPONSE.mass_kg,
        acceptance_curve=_lz_acceptance(include_he_sb=include_he_sb),
    )
    energy = np.array([
        250e3, 300.5e3, 301.5e3, 302e3, 500e3, 700.5e3, 701e3, 1e9,
    ])
    np.testing.assert_allclose(
        detector._apply_acceptance(energy, np.ones_like(energy)),
        expected,
        atol=0.0,
    )


def test_deap3600_2026_response_configuration_and_acceptance():
    assert isinstance(DEAP3600_2026_RESPONSE, ResponseDetector)
    assert DEAP3600_2026_RESPONSE.mass_kg == 1266
    assert DEAP3600_2026_RESPONSE.exposure_days == 790.8
    assert DEAP3600_2026_RESPONSE.has_acceptance
    assert DEAP3600_2026_RESPONSE.has_resolution
    assert DEAP3600_2026_RESPONSE.resolution_sigma == 10e3

    # ER light yield 6.1 PE/keVee and argon Q = E_ee/E_nr = 0.25
    # give a mean detected NR light yield of 1.525 PE/keVnr.
    pe_values = np.array([0.0, 90.0, 150.0, 200.0])
    energy = np.concatenate((
        pe_values / 1.525 * 1e3, [1000e3, 1000.5e3, 1001e3, 1e9],
    ))
    acceptance = DEAP3600_2026_RESPONSE._apply_acceptance(
        energy, np.ones_like(energy)
    )
    np.testing.assert_allclose(
        acceptance, [0.0, 0.18, 0.27, 0.30, 0.30, 0.15, 0.0, 0.0], atol=0.0
    )


def test_ds50_2018_response_configuration_and_acceptance():
    assert isinstance(DS50_2018_RESPONSE, ResponseDetector)
    assert DS50_2018_RESPONSE.mass_kg == 31.3
    assert DS50_2018_RESPONSE.exposure_days == 532.4
    assert DS50_2018_RESPONSE.has_acceptance
    assert DS50_2018_RESPONSE.has_resolution
    assert DS50_2018_RESPONSE.resolution_sigma == 6e3

    energy = np.array([0.0, 40e3, 60e3, 80e3, 1e9])
    acceptance = DS50_2018_RESPONSE._apply_acceptance(
        energy, np.ones_like(energy)
    )
    np.testing.assert_allclose(acceptance, [0.0, 0.15, 0.62, 0.725, 0.0], atol=0.0)


def test_ds20k_10year_response_configuration_and_acceptance():
    assert isinstance(DS20K_10YEAR_RESPONSE, ResponseDetector)
    assert DS20K_10YEAR_RESPONSE.mass_kg == 20000
    assert DS20K_10YEAR_RESPONSE.exposure_days == 3650
    assert DS20K_10YEAR_RESPONSE.has_acceptance
    assert DS20K_10YEAR_RESPONSE.has_resolution
    assert DS20K_10YEAR_RESPONSE.resolution_sigma == 3e3

    energy = np.array([0.0, 30e3, 40e3, 50e3, 1e9])
    acceptance = DS20K_10YEAR_RESPONSE._apply_acceptance(
        energy, np.ones_like(energy)
    )
    np.testing.assert_allclose(acceptance, [0.0, 0.30, 0.72, 0.90, 0.0], atol=0.0)


def test_xenont_2024_response_configuration_and_acceptance():
    assert isinstance(XENONnT_2024_RESPONSE, ResponseDetector)
    assert XENONnT_2024_RESPONSE.mass_kg == 5900
    assert XENONnT_2024_RESPONSE.exposure_days == 278
    assert XENONnT_2024_RESPONSE.has_acceptance
    assert XENONnT_2024_RESPONSE.has_resolution
    assert XENONnT_2024_RESPONSE.resolution_sigma == 3e3

    # With 6 photons/keVnr and g1 = 0.151 PE/photon, 1 keVnr
    # produces a mean S1 of 0.906 PE.
    s1_pe = np.array([0.0, 3.0, 5.0, 10.0])
    energy = np.concatenate((
        s1_pe / (0.151 * 6.0) * 1e3, [1000e3, 1000.5e3, 1001e3, 1e9],
    ))
    acceptance = XENONnT_2024_RESPONSE._apply_acceptance(
        energy, np.ones_like(energy)
    )
    np.testing.assert_allclose(
        acceptance, [0.0, 0.55, 0.80, 0.92, 0.92, 0.46, 0.0, 0.0], atol=0.0
    )
