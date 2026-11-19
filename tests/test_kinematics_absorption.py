import numpy as np
import pytest

from src.kinematics.scattering_kinematics import NuclearRecoilKinematics
from src.kinematics.absorption_kinematics import AbsorptionKinematics

def test_absorption_kinematics_not_supported_by_scattering_interactions():
    kin = AbsorptionKinematics(mX=40e6, mT=40e9)
    assert not kin.supports_scattering_interactions

@pytest.mark.parametrize("mX, mT", [
    (0.0, 40e9),
    (-1e6, 40e9),
    (40e6, 0.0),
    (40e6, -40e9),
])
def test_absorption_kinematics_mass_validation(mX, mT):
    with pytest.raises(ValueError):
        AbsorptionKinematics(mX=mX, mT=mT)

def test_absorption_kinematics_inheritance():
    kin = AbsorptionKinematics(mX=1e6, mT=1e12)
    assert isinstance(kin, NuclearRecoilKinematics)

@pytest.mark.parametrize("mX, mT, expected_eV", [
    (10e6, 40e9, 1250.0),
    (40e6, 40e9, 20000.0),
    (40e6, 125e9, 6400.0),
])
def test_absorption_kinematics_recoil_energy(mX, mT, expected_eV):
    kin = AbsorptionKinematics(mX=mX, mT=mT)
    assert kin.ER == pytest.approx(expected_eV)

@pytest.mark.parametrize("mX, mT", [(10e6, 40e9),
                                    (40e6, 40e9),
                                    (40e6, 125e9)])
def test_absorption_kinematics_q_from_ER(mX, mT):
    kin = AbsorptionKinematics(mX=mX, mT=mT)
    assert kin.q_from_ER(kin.ER) == pytest.approx(mX)


@pytest.mark.parametrize("v", [
    0.0,
    np.array([0.0, 1e-4, 1e-3]),
])
def test_absorption_kinematics_ER_bounds_are_identical(v):
    kin = AbsorptionKinematics(mX=40e6, mT=40e9)
    ER_min, ER_max = kin.ER_bounds(v)
    np.testing.assert_allclose(ER_min, ER_max)
    np.testing.assert_allclose(ER_min, 20000.0)


@pytest.mark.parametrize("values", [
    0.0,
    np.array(0.0),
    [0.0, 1e-4, 1e-3],
    np.zeros(3, dtype=int),
])
@pytest.mark.parametrize("method", ["ER_min", "ER_max"])
def test_absorption_kinematics_array_inputs(values, method):
    kin = AbsorptionKinematics(mX=7e6, mT=122e9)

    result = getattr(kin, method)(values)
    expected = kin.ER

    assert np.shape(result) == np.shape(values)
    assert np.asarray(result).dtype.kind == "f"
    np.testing.assert_allclose(result, expected)

def test_absorption_recoil_independent_of_speed():
    kin = AbsorptionKinematics(40e6, 40e9)

    np.testing.assert_allclose(
        kin.ER_min(1e-5),
        kin.ER_min(1e-3),
    )