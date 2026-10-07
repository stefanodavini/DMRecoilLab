from src.halo import StandardHaloModel
from src.kinematics import ElasticKinematics
from src.interactions.si import SIInteraction
from src.interactions.particles import WIMP
from src.nuclei.nucleus import Nucleus
from src.nuclei.isotopes import Ar40, Xe131
from src.detectors.detectors import IdealDetector
from src.rates import RateCalculator

import numpy as np

def _initialize_si_elastic(wimp: WIMP,
                           target: Nucleus,
                           sigma_p=1e-45) -> SIInteraction:

    kin = ElasticKinematics(mX=wimp.mass, mT=target.mass)
    si = SIInteraction(wimp=wimp, nucleus=target,
                       kinematics=kin,
                       sigma_p=sigma_p)
    return si

def _initialize_halo_shm_boost():
    shm = StandardHaloModel()
    return shm.boost()

def _initialize_default_si_elastic_rate(target: Nucleus,
                                        sigma_p=1e-45) -> RateCalculator:
    wimp = WIMP(mass=1e11, spin=0)
    si = _initialize_si_elastic(wimp=wimp, target=target, sigma_p=sigma_p)
    halo = _initialize_halo_shm_boost()
    return RateCalculator(wimp=wimp, halo=halo, interaction=si)

def test_rate_ERmax_is_scalar():
    rate = _initialize_default_si_elastic_rate(Ar40)
    assert np.isscalar(rate.ER_max)

def test_rate_dRdE():
    ratecalc = _initialize_default_si_elastic_rate(Ar40)
    ER_max = ratecalc.ER_max
    ER = np.linspace(0, ER_max, 100)
    dRdE = ratecalc.dRdER(ER=ER) 

    assert dRdE.shape == ER.shape
    assert np.all(dRdE >= 0)

def test_rate_dRdE_above_ERmax():
    ratecalc = _initialize_default_si_elastic_rate(Xe131)
    ERmax = ratecalc.ER_max
    ER = np.array([1.2 * ERmax, 2.0 * ERmax])

    assert np.all(ratecalc.dRdER(ER=ER) == 0)

def test_rate_dRdE_Emax_suppression():
    ratecalc = _initialize_default_si_elastic_rate(Ar40)
    ERmax = ratecalc.ER_max
    ER = np.array([0.9 * ERmax, 1.1 * ERmax])
    dRdER = ratecalc.dRdER(ER=ER)

    assert dRdER[0] > 0
    assert dRdER[1] == 0

def test_rate_Emax_physically_valid():
    """
    Check that ER max is in the correct
    physics range (50 keV - 500 keV)
    """
    ratecalc = _initialize_default_si_elastic_rate(Xe131)

    assert 5e4 < ratecalc.ER_max < 5e5

def test_rate_density_zero():
    wimp = WIMP(mass=1e11, spin=0)
    target = Xe131
    si = _initialize_si_elastic(wimp=wimp, target=target)
    halo = _initialize_halo_shm_boost()

    ratecalc = RateCalculator(wimp=wimp, halo=halo, interaction=si,
                              rho_dm=0.)
    
    ER = np.linspace(0, 1e5, 100)
    dRdE = ratecalc.dRdER(ER)

    assert np.all(dRdE == 0)

def test_rate_sigma_linearity():
    target = Xe131
    r2 = _initialize_default_si_elastic_rate(target, sigma_p=2)
    r3 = _initialize_default_si_elastic_rate(target, sigma_p=3)

    ER = np.linspace(1, 1e5, 100)
    ratio = r2.dRdER(ER) / r3.dRdER(ER) 

    assert np.allclose(ratio , 2./3, rtol=1e-6)

def test_rate_integrated_shape():
    ratecalc = _initialize_default_si_elastic_rate(Xe131)

    ER = np.linspace(1, 1e5, 100)
    intrate = ratecalc.integrated_rate_above_threshold(ER)

    assert intrate.shape == ER.shape
    assert len(intrate) == len(ER)
    assert np.all(intrate >= 0)
    assert np.all(np.diff(intrate) <= 0)

def test_rate_detector_mass_linearity():
    wimp=WIMP(mass=100e9, spin=0)
    nucleus = Xe131
    det_10 = IdealDetector(nucleus=nucleus, mass_kg=10)
    det_500 = IdealDetector(nucleus=nucleus, mass_kg=500)

    shm = _initialize_halo_shm_boost()
    si = _initialize_si_elastic(wimp=wimp, target=nucleus)

    rate_10 = RateCalculator(wimp=wimp, halo=shm,
                            interaction=si,
                            detector=det_10)
    rate_500 = RateCalculator(wimp=wimp, halo=shm,
                              interaction=si,
                              detector=det_500)

    ER = np.linspace(1, 1e5, 10)
    ratio =  rate_500.dRdER(ER) / rate_10.dRdER(ER)

    assert np.allclose(ratio, 50)

def test_rate_detector_exposure_linearity():
    wimp=WIMP(mass=100e9, spin=0)
    nucleus = Xe131
    det_1day = IdealDetector(nucleus=nucleus, mass_kg=100,
                             exposure_days=1)
    det_1year = IdealDetector(nucleus=nucleus, mass_kg=100,
                              exposure_days=365)

    shm = _initialize_halo_shm_boost()
    si = _initialize_si_elastic(wimp=wimp, target=nucleus)

    rate_1day = RateCalculator(wimp=wimp, halo=shm,
                            interaction=si,
                            detector=det_1day)
    rate_1year = RateCalculator(wimp=wimp, halo=shm,
                              interaction=si,
                              detector=det_1year)

    ER = np.linspace(1, 1e5, 10)
    ratio =  rate_1year.dRdER(ER) / rate_1day.dRdER(ER)

    assert np.allclose(ratio, 365)



