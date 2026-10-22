import numpy as np
from src.kinematics import ElasticKinematics, InelasticKinematics
from src.interactions.particles import WIMP
from src.interactions.sd import SDInteraction
from src.nuclei.isotopes import Ar40, Xe131

def _initialize_sd_elastic() -> SDInteraction:
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    sd = SDInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin, form_factor=None)
    return sd

def _initialize_sd_inelastic() -> SDInteraction:
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = InelasticKinematics(mX=wimp.mass, mT=target.mass, delta=300e3)
    sd = SDInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin, form_factor=None)
    return sd

def test_sd_sigma_proton():
    sigma_p = 1e-48
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    sd = SDInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       sigma_p=sigma_p)

    assert sd.sigma_proton == sigma_p

def test_sd_proton_only_coupling():
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    sd = SDInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       an_over_ap=0)

    assert sd.spin_coupling == Xe131.Sp

def test_sd_neutron_only_coupling():
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    sd = SDInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       an_over_ap=np.nan)

    assert sd.sigma_proton == 0.
    assert sd.spin_coupling == Xe131.Sn


def test_sd_elastic_shape():
    v = np.linspace(1e-5, 1e-3, 100)
    ER = np.linspace(1e3, 1e5, 50)

    sd = _initialize_sd_elastic()
    dsigma = sd.dsigma_dER(v=v, ER=ER)

    assert dsigma.shape == (100, 50)

def test_sd_inelastic_shape():
    v = np.linspace(1e-5, 1e-3, 100)
    ER = np.linspace(1e3, 1e5, 50)

    sd = _initialize_sd_inelastic()
    dsigma = sd.dsigma_dER(v=v, ER=ER)

    assert dsigma.shape == (100, 50)

def test_sd_elastic_non_negative():
    ER = np.linspace(1e3, 1e5, 50)
    v = np.linspace(1e-5, 1e-3, 100)

    sd = _initialize_sd_elastic()
    dsigma = sd.dsigma_dER(v=v, ER=ER)

    assert np.all(dsigma >= 0)

def test_sd_inelastic_non_negative():
    ER = np.linspace(1e3, 1e5, 50)
    v = np.linspace(1e-5, 1e-3, 100)

    sd = _initialize_sd_inelastic()
    dsigma = sd.dsigma_dER(v=v, ER=ER)

    assert np.all(dsigma >= 0)

def test_sd_elastic_masking():
    v = np.array([1e-4])

    sd = _initialize_sd_elastic()
    kin = sd.scattering_kinematics
    Emax =  kin.ER_max(v=v)

    ER = np.array([2 * Emax[0] + 1e3])

    dsigma = sd.dsigma_dER(v=v, ER=ER)
    assert dsigma[0, 0] == 0

def test_sd_endothermic_masking_below_threshold():
    sd = _initialize_sd_inelastic()
    kin = sd.scattering_kinematics

    v = np.array([0.9 * kin.threshold_speed])
    ER = np.linspace(1e4, 1e5, 10)

    dsigma = sd.dsigma_dER(ER=ER, v=v)
    assert np.all(dsigma == 0)

def test_si_endothermic_positive_above_threshold():
    sd = _initialize_sd_inelastic()
    kin = sd.scattering_kinematics

    v = np.array([1.1 * kin.threshold_speed + 1e-3])
    ERmin = kin.ER_min(v)[0]
    ERmax = kin.ER_max(v)[0]
    ER = np.linspace(ERmin, ERmax, 100)

    dsigma = sd.dsigma_dER(ER=ER, v=v)
    assert np.all(dsigma > 0)

def test_sd_spinless_nucleus():
    wimp = WIMP(mass=1e11, spin=0)
    target = Ar40

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    sd = SDInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin)

    v = np.linspace(1e-5, 1e-3, 100)
    ER = np.linspace(0, 1e5, 50)
    dsigma = sd.dsigma_dER(v, ER)

    assert dsigma.shape == (100, 50)
    assert np.all(dsigma == 0)

def test_sd_zero_sigma_proton():
    sigma_p = 0.
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    sd = SDInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       sigma_p=sigma_p)

    v = np.linspace(1e-5, 1e-3, 100)
    ER = np.linspace(0, 1e5, 50)
    dsigma = sd.dsigma_dER(v, ER)

    assert sd.sigma_proton == 0.
    assert np.all(dsigma == 0)

def test_sd_elastic_speed_scaling():
    """
    Check the speed scaling as v^-2,
    away from masking boundaries
    """
    v1 = 1e-3
    v2 = 2e-3
    v3 = 3e-3
    v = np.array([v1, v2, v3])
    sd = _initialize_sd_elastic()
    kin = sd.scattering_kinematics

    ER_allowed = kin.ER_max(v1) / 2
    ER = np.array([ER_allowed])

    dsigma = sd.dsigma_dER(v=v, ER=ER)
    ds1, ds2, ds3 = dsigma[0, 0], dsigma[1, 0], dsigma[2, 0]
    ratio_12 = ds1 / ds2
    ratio_13 = ds1 / ds3
    ratio_23 = ds2 / ds3
    assert np.isclose(ratio_12, 4., rtol=1e-4)
    assert np.isclose(ratio_13, 9., rtol=1e-4)
    assert np.isclose(ratio_23, 9./4., rtol=1e-4)