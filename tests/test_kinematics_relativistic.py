from src.kinematics.base_kinematics import NuclearRecoilKinematics
from src.kinematics.scattering_kinematics import (
    ScatteringKinematics, NonRelativisticKinematics
    )
from src.kinematics.relativistic_kinematics import (
    RelativisticKinematics,
    RelativisticElasticKinematics,
    RelativisticInelasticKinematics
    )

import numpy as np
import pytest

def test_kinematics_relativistic_inheritance():
    kin = RelativisticKinematics(mX=1e11, mT=122e9, m2=1e11)

    assert isinstance(kin, NuclearRecoilKinematics)
    assert isinstance(kin, ScatteringKinematics)
    assert not isinstance(kin, NonRelativisticKinematics)

def test_kinematics_relativistic_elastic_inheritance():
    kin = RelativisticElasticKinematics(mX=1e11, mT=122e9)

    assert isinstance(kin, NuclearRecoilKinematics)
    assert isinstance(kin, ScatteringKinematics)
    assert isinstance(kin, RelativisticKinematics)
    assert not isinstance(kin, NonRelativisticKinematics)

def test_kinematics_relativistic_inelastic_inheritance():
    kin = RelativisticInelasticKinematics(mX=1e11, mT=122e9, m2=1.1e11)

    assert isinstance(kin, NuclearRecoilKinematics)
    assert isinstance(kin, ScatteringKinematics)
    assert isinstance(kin, RelativisticKinematics)
    assert not isinstance(kin, NonRelativisticKinematics)

def test_kinematics_relativistic_unphysical_mass_raises():
    m_unphysical = [-1, 0., np.inf, np.nan]
    m_good = 1e11

    for m in m_unphysical:
        with pytest.raises(ValueError,
                           match="Unphysical dark-matter mass"):
            RelativisticElasticKinematics(mX=m, mT=m_good)

    for m in m_unphysical:
        with pytest.raises(ValueError,
                           match="Unphysical target mass"):
            RelativisticElasticKinematics(mX=m_good, mT=m)

    for m in m_unphysical:
        with pytest.raises(ValueError,
                           match="Unphysical final-state WIMP mass"):
            RelativisticKinematics(mX=m_good, mT=m_good, m2=m)


def test_kinematics_relativistic_nonscalar_m2_raises():
    m_bad = [[], [1e11], [1e11, 2e11]]
    for m in m_bad:
        with pytest.raises(TypeError, match="scalar"):
            RelativisticInelasticKinematics(mX=10, mT=30, m2=m)

@pytest.fixture
def kin():
    return RelativisticInelasticKinematics(mX=10., mT=30., m2=12.)


SPEED_APIS = [
    'E1_from_v', 'speed_mask', 'ER_bounds', 'ER_min', 'ER_max',
    'q_from_v_theta', 'speed_theta_mask',
]


def call_speed_api(kin, name, v):
    return getattr(kin, name)(v, 0.) if 'theta' in name else getattr(kin, name)(v)


@pytest.mark.parametrize('name', SPEED_APIS)
@pytest.mark.parametrize('bad', [-1., np.nan, np.inf, 1., 1.1, [.2, -1.]])
def test_speed_domain_errors(kin, name, bad):
    with pytest.raises(ValueError):
        call_speed_api(kin, name, bad)

@pytest.mark.parametrize('name', ['q_from_ER', 'Q2_from_ER', 'vmin_from_ER'])
@pytest.mark.parametrize('bad', [-1., np.nan, np.inf, [0., -1.], [[1.], [-np.inf]]])
def test_recoil_domain_errors(kin, name, bad):
    with pytest.raises(ValueError):
        getattr(kin, name)(bad)

@pytest.mark.parametrize('name', ['s_from_E1', 'energy_mask', 'ER_bounds_from_E1'])
@pytest.mark.parametrize('bad', [9., -1., np.nan, np.inf, [10., 9.]])
def test_total_energy_domain_errors(kin, name, bad):
    with pytest.raises(ValueError):
        getattr(kin, name)(bad)
