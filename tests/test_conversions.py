from src.utils.conversions import convert_from_kmps_to_beta, convert_from_beta_to_kmps
from src.utils.conversions import convert_from_kg_to_eV
from src.utils.conversions import convert_from_momentum_eV_to_fm, convert_from_fm_to_momentum_eV
from src.utils.conversions import convert_qr_to_dimensionless

from src.utils.constants import PROTON_MASS_MeV, HBAR_C_MeV_fm

import numpy as np

def test_consistency_kms_beta():
    beta = np.linspace(0., 1., 1000)
    result = convert_from_kmps_to_beta(
        convert_from_beta_to_kmps(beta))

    assert np.allclose(result, beta, atol=1e-9)

def test_lightspeed_beta_one():
    c = 299792.458
    assert np.isclose(convert_from_kmps_to_beta(c),
                      1., rtol=1e-9)

def test_beta_one_lightspeed():
    assert np.isclose(convert_from_beta_to_kmps(1),
                      299792458e-3, rtol=1e-9)

def test_lightspeed_zero():
    assert convert_from_kmps_to_beta(0.) == 0
    assert convert_from_beta_to_kmps(0.) == 0

def test_kg_to_eV_proton():
    mp_kg = 1.67262192595e-27
    mp_eV = PROTON_MASS_MeV * 1e6
    assert np.isclose(convert_from_kg_to_eV(m=mp_kg),
                      mp_eV, rtol=1e-9)

def test_consistency_ev_fm():
    x = np.logspace(-15, -9)
    result = convert_from_momentum_eV_to_fm(
        convert_from_fm_to_momentum_eV(x))

    assert np.allclose(result, x, rtol=1e-9)

def test_consistency_dimensionless_qr():
    q = 1e8  # eV
    r = convert_from_momentum_eV_to_fm(q)

    assert np.isclose(
        convert_qr_to_dimensionless(q * r),
        1.0)

def test_hbar_c_dimensionless_qr():
    hbarc = 1e6 * HBAR_C_MeV_fm
    assert np.isclose(
        convert_qr_to_dimensionless(hbarc),
        1.0)