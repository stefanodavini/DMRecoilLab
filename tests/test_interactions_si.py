import numpy as np
from src.kinematics import ElasticKinematics, InelasticKinematics
from src.interactions.particles import WIMP
from src.interactions.si import SIInteraction
from src.nuclei.form_factors import PointLikeFormFactor, HelmFormFactor
from src.nuclei.isotopes import Ar40, Xe131

def _initialize_si_elastic() -> SIInteraction:
    wimp = WIMP(mass=1e11, spin=0)
    target = Ar40

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    si = SIInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin, form_factor=PointLikeFormFactor())
    return si

def _initialize_si_inelastic() -> SIInteraction:
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = InelasticKinematics(mX=wimp.mass, mT=target.mass, delta=300e3)
    si = SIInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin, form_factor=PointLikeFormFactor())
    return si

def test_si_sigma_proton():
    sigma_p = 1e-48
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    si = SIInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       sigma_p=sigma_p)

    assert si.sigma_proton == sigma_p

def test_si_elastic_shape():
    v = np.linspace(1e-5, 1e-3, 100)
    ER = np.linspace(1e3, 1e5, 50)

    si = _initialize_si_elastic()
    dsigma = si.dsigma_dER(v=v, ER=ER)

    assert dsigma.shape == (100, 50)

def test_si_inelastic_shape():
    v = np.linspace(1e-5, 1e-3, 100)
    ER = np.linspace(1e3, 1e5, 50)

    si = _initialize_si_inelastic()
    dsigma = si.dsigma_dER(v=v, ER=ER)

    assert dsigma.shape == (100, 50)

def test_si_elastic_non_negative():
    ER = np.linspace(1e3, 1e5, 50)
    v = np.linspace(1e-5, 1e-3, 100)

    si = _initialize_si_elastic()
    dsigma = si.dsigma_dER(v=v, ER=ER)

    assert np.all(dsigma >= 0)

def test_si_inelastic_non_negative():
    ER = np.linspace(1e3, 1e5, 50)
    v = np.linspace(1e-5, 1e-3, 100)

    si = _initialize_si_inelastic()
    dsigma = si.dsigma_dER(v=v, ER=ER)

    assert np.all(dsigma >= 0)

def test_si_elastic_masking():
    v = np.array([1e-4])

    si = _initialize_si_elastic()
    kin = si.scattering_kinematics
    Emax =  kin.ER_max(v=v)

    ER = np.array([2 * Emax[0] + 1e3])

    dsigma = si.dsigma_dER(v=v, ER=ER)
    assert dsigma[0, 0] == 0

def test_si_endothermic_masking_below_threshold():
    si = _initialize_si_inelastic()
    kin = si.scattering_kinematics

    v = np.array([0.9 * kin.threshold_speed])
    ER = np.linspace(1e4, 1e5, 10)

    dsigma = si.dsigma_dER(ER=ER, v=v)
    assert np.all(dsigma == 0)

def test_si_endothermic_positive_above_threshold():
    si = _initialize_si_inelastic()
    kin = si.scattering_kinematics

    v = np.array([1.1 * kin.threshold_speed + 1e-3])
    ERmin = kin.ER_min(v)[0]
    ERmax = kin.ER_max(v)[0]
    ER = np.linspace(ERmin, ERmax, 100)

    dsigma = si.dsigma_dER(ER=ER, v=v)
    assert np.all(dsigma > 0)

def test_si_elastic_speed_scaling():
    # test the v^-2 scaling
    v1 = 1e-3
    v2 = 2e-3
    v3 = 3e-3
    v = np.array([v1, v2, v3])
    si = _initialize_si_elastic()
    kin = si.scattering_kinematics

    ER_allowed = kin.ER_max(v1) / 2
    ER = np.array([ER_allowed])

    dsigma = si.dsigma_dER(v=v, ER=ER)
    ds1, ds2, ds3 = dsigma[0, 0], dsigma[1, 0], dsigma[2, 0]
    ratio_12 = ds1 / ds2
    ratio_13 = ds1 / ds3
    ratio_23 = ds2 / ds3
    assert np.isclose(ratio_12, 4., rtol=1e-4)
    assert np.isclose(ratio_13, 9., rtol=1e-4)
    assert np.isclose(ratio_23, 9./4., rtol=1e-4)

def test_si_helm_suppression():

    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)

    si_point = SIInteraction(
        wimp=wimp, nucleus=target, kinematics=kin,
        form_factor=PointLikeFormFactor())

    si_helm = SIInteraction(
        wimp=wimp, nucleus=target, kinematics=kin,
        form_factor=HelmFormFactor(target))

    v = np.linspace(1e-4, 1e-3, 50)
    ER = np.logspace(3, 6, 100)
    ER[0] = 0.

    ds_point = si_point.dsigma_dER(v=v, ER=ER)
    ds_helm = si_helm.dsigma_dER(v=v, ER=ER)

    assert np.all(ds_helm <= ds_point)
    assert np.all(ds_point[:, 0] == ds_helm[:, 0])

def test_si_elastic_total_cross_section():
    """
    Check that that integral of dsigma/dER
    int_0^ERmax dsigma_dER dER 
    is the nucleus cross section
    """
    si = _initialize_si_elastic()
    v = np.array([1e-3])

    sigma_num = si.integrate_sigma(v=v)[0]
    sigma_nucleus = si.sigma_nucleus

    assert np.isclose(sigma_num, sigma_nucleus, rtol=1e-6)

def test_si_sigma_nucleus_A4():
    """
    Check the approximate scaling of the nucleus cross section
    as A^4 when the reduced mass is approximately the target's
    """
    # Heavy WIMP, such that mu is approx. mT
    wimp = WIMP(mass=1e12, spin=0)
    target = Ar40

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)

    sigma_p = 1.
    si = SIInteraction(wimp=wimp, nucleus=target, 
                       kinematics=kin,
                       sigma_p=sigma_p)

    sigma_expected = sigma_p * (target.A)**4
    assert np.isclose(sigma_expected, si.sigma_nucleus, rtol=1e-1)

def test_si_sigma_nucleus():
    """
    Check the cross secion normalization
    as parametrized by the appropriate reduced masses
    """
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)

    sigma_p = 1e-10
    si = SIInteraction(wimp=wimp, nucleus=target, 
                       kinematics=kin,
                       sigma_p=sigma_p)

    mu_ratio2 = (si._reduced_mass_nucleus / si._reduced_mass_proton)**2

    sigma_expected = sigma_p * mu_ratio2 * (target.A)**2
    assert np.isclose(sigma_expected, si.sigma_nucleus, rtol=1e-6)

def test_si_isospin_cancellation():
    """
    Check that fn/fp = -Z/N
    cancels the cross sections
    """
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131
    
    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    
    fn_over_fp = - target.Z / target.N

    si = SIInteraction(wimp=wimp, nucleus=target, 
                       kinematics=kin, sigma_p=1,
                       fn_over_fp=fn_over_fp)
    
    assert np.isclose(0.0, si.sigma_nucleus, atol=1e-16)

def test_si_isospin_proton_only():
    """
    Check that if fn/fp = 0
    the coherent coupling is Z^2
    """
    wimp = WIMP(mass=2e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)

    sigma_p = 1
    si = SIInteraction(wimp=wimp, nucleus=target, 
                       kinematics=kin,
                       sigma_p=sigma_p,
                       fn_over_fp=0)

    mu_ratio2 = (si._reduced_mass_nucleus / si._reduced_mass_proton)**2

    sigma_expected = sigma_p * mu_ratio2 * (target.Z)**2
    assert np.isclose(sigma_expected, si.sigma_nucleus, rtol=1e-6)
