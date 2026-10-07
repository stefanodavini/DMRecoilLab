from src.kinematics import ElasticKinematics, InelasticKinematics
import numpy as np
import pytest

def test_mX_zero():
    with pytest.raises(ValueError):
        ElasticKinematics(mX=0, mT=40e9)

def test_mX_negative():
    with pytest.raises(ValueError):
        ElasticKinematics(mX=-1e9, mT=40e9)

def test_mT_zero():
    with pytest.raises(ValueError):
        ElasticKinematics(mX=-1e12, mT=0)

def test_mT_negative():
    with pytest.raises(ValueError):
        ElasticKinematics(mX=1e12, mT=-122e9)

def test_mu_limit():
    m_low = 1e3
    m_hig = 1e9
    kin = ElasticKinematics(mX=m_low, mT=m_hig)
    assert np.isclose(kin.mu, m_low, rtol=1e-6)

def test_mu_equal():
    m = 40e9
    kin = ElasticKinematics(mX=m, mT=m)
    assert np.isclose(kin.mu, m/2, rtol=1e-9)

def test_elastic_q_from_ER():
    mT = 40e9
    kin = ElasticKinematics(mX=1e10, mT=mT)
    ER = np.linspace(1e3, 1e5, 100)
    expected = np.sqrt(2 * mT * ER)
    assert np.array_equiv(kin.q_from_ER(ER), expected)

def test_elastic_vmin_from_ER_zero():
    kin = ElasticKinematics(mX=1e12, mT=40e9)
    assert kin.vmin_from_ER(0) == 0.

def test_elastic_ER_max_zero():
    kin = ElasticKinematics(mX=1e12, mT=40e9)
    assert kin.ER_max(0) == 0.

def test_elastic_vmin_of_ERmax():
    # test that vmin and ERmax are inverse functions
    kin = ElasticKinematics(mX=1e12, mT=122e9)
    v = np.linspace(1e-6, 1e-3, 100)
    retval = kin.vmin_from_ER(kin.ER_max(v))
    assert np.allclose(retval, v, rtol=1e-6)

def test_elastic_ER_bounds():
    kin = ElasticKinematics(mX=1e12, mT=122e9)
    v = 1e-3
    ERmax = kin.ER_max(v)
    assert kin.ER_bounds(v) == (0., ERmax)

def test_elastic_ER_bound_sorting():
    kin = ElasticKinematics(mX=1e12, mT=40e9)
    v = np.linspace(1e-6, 1e-3, 1000)
    bounds = kin.ER_bounds(v)
    bound_min = bounds[0]
    bound_max = bounds[1]
    assert np.all(bound_min == 0)
    assert np.all(bound_max > 0)

def test_endhothermic_threshold_speed():
    delta = 1e5
    kin = InelasticKinematics(mX=1e10, mT=40e9, delta=delta)

    assert kin.is_endothermic is True
    expected = np.sqrt(2 * delta / kin.mu)
    assert kin.threshold_speed == expected

def test_exothermic_threshold_speed():
    delta = -1e5
    kin = InelasticKinematics(mX=1e10, mT=40e9, delta=delta)

    assert kin.is_endothermic is False
    assert kin.threshold_speed == 0.

def test_exothermic_speed_mask():
    delta = -2e5
    kin = InelasticKinematics(mX=1e11, mT=40e9, delta=delta)
    v = np.logspace(-9, -3)
    assert np.all(kin.speed_mask(v) == 1.)

def test_inelastic_q_from_ER():
    mT = 40e9
    kin = InelasticKinematics(mX=1e10, mT=mT, delta=1e5)
    ER = np.linspace(1e3, 1e5, 100)
    expected = np.sqrt(2 * mT * ER)
    assert np.array_equiv(kin.q_from_ER(ER), expected)

def test_inelastic_vmin_delta_zero():
    mX = 1e12
    mT = 40e9
    kin_e = ElasticKinematics(mX=mX, mT=mT)
    kin_i = InelasticKinematics(mX=mX, mT=mT, delta=0)
    ER = np.linspace(1e3, 1e5, 100)
    assert np.array_equiv(kin_i.vmin_from_ER(ER), kin_e.vmin_from_ER(ER))

def test_inelastic_vmin_at_ERthr():
    delta = 2e5
    mT = 40e9
    kin = InelasticKinematics(mX=1e12, mT=mT, delta=delta)
    mu = kin.mu
    ERthr = mu * delta / mT
    expected = np.sqrt(2 * delta / mu)
    assert np.isclose(kin.vmin_from_ER(ERthr), expected, rtol=1e-6)

def test_inelastic_ER_bound_sorting():
    kin = InelasticKinematics(mX=1e11, mT=40e9, delta=1e5)
    vthr = kin.threshold_speed
    v = np.linspace(vthr + 1e-5, vthr + 1e-3, 100)
    bounds = kin.ER_bounds(v)
    bound_min = bounds[0]
    bound_max = bounds[1]
    assert np.all(bound_min <= bound_max)

def test_inelastic_masking_v():
    delta = 2e5
    kin = InelasticKinematics(mX=1e11, mT=40e9, delta=delta)
    vthr = np.sqrt(2 * delta / kin.mu)
    v = np.array([0.9 * vthr, 1.1 * vthr])
    expected = np.array([False,  True])
    assert np.array_equiv(kin.speed_mask(v), expected)

def test_inelastic_ER_bounds_nan_below_threshold():
    kin = InelasticKinematics(mX=1e11, mT=40e9, delta=2e5)
    v = 0.9 * kin.threshold_speed
    assert np.isnan(kin.ER_min(v))
    assert np.isnan(kin.ER_max(v))

def test_inelastic_ER_bounds_nan_mask():
    kin = InelasticKinematics(mX=1e11, mT=40e9, delta=2e5)
    vthr = kin.threshold_speed
    v = np.array([0.9 * vthr, 1.1 * vthr])

    ERmin = kin.ER_min(v)
    ERmax = kin.ER_max(v)

    assert np.isnan(ERmin[0])
    assert np.isnan(ERmax[0])
    assert np.isfinite(ERmin[1])
    assert np.isfinite(ERmax[1])

