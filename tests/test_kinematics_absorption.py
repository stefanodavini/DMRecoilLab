import numpy as np
import pytest

from src.kinematics.scattering_kinematics import ScatteringKinematics
from src.kinematics.absorption_kinematics import AbsorptionKinematics


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
    assert isinstance(kin, ScatteringKinematics)

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
@pytest.mark.parametrize("method", ["ER_min", "ER_max", "vmin_from_ER"])
def test_absorption_kinematics_array_inputs(values, method):
    kin = AbsorptionKinematics(mX=7e6, mT=122e9)

    result = getattr(kin, method)(values)
    expected = 0.0 if method == "vmin_from_ER" else kin.ER

    assert np.shape(result) == np.shape(values)
    assert np.asarray(result).dtype.kind == "f"
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("v", [
    0.0,
    np.array(0.0),
    [0.0, 1e-4, 1e-3],
    np.zeros(3, dtype=int),
])
def test_absorption_kinematics_speed_mask_all_true(v):
    kin = AbsorptionKinematics(mX=40e6, mT=40e9)
    mask = kin.speed_mask(v)

    assert np.shape(mask) == np.shape(v)
    assert np.asarray(mask).dtype.kind == "b"
    assert np.all(mask)
    assert kin.threshold_speed == 0.0

def test_absorption_kinematics_angular_not_implemented():
    kin = AbsorptionKinematics(mX=40e6, mT=40e9)
    with pytest.raises(NotImplementedError):
        kin.q_from_v_theta(1e-3, 0.0)

def test_absorption_recoil_independent_of_speed():
    kin = AbsorptionKinematics(40e6, 40e9)

    np.testing.assert_allclose(
        kin.ER_min(1e-5),
        kin.ER_min(1e-3),
    )