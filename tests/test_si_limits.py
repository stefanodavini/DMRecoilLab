from src.analysis.si_limits import compute_mass_sigma_limit_si
from src.detectors.detectors import IdealDetector
from src.detectors.reference_detectors import LZ_1000DAYS
from src.nuclei.isotopes import Ar40
from src.nuclei.form_factors import PointLikeFormFactor
import numpy as np

def test_compute_mass_sigma_limit_si_shape():
    masses = np.geomspace(1e10, 1e12, 6)
    
    sigma = compute_mass_sigma_limit_si(
        masses=masses,
        detector=LZ_1000DAYS
        )

    assert sigma.ndim == 1
    assert sigma.shape == masses.shape

def test_compute_mass_sigma_limit_si_minimum():
    """
    Check that the sigma more constrained
    happens for WIMP mass around target mass
    """
    nucleus = Ar40
    mT = nucleus.mass

    detector = IdealDetector(nucleus=nucleus,
                             mass_kg=100,
                             exposure_days=365,
                             threshold_energy=0)

    masses = [0.75 * mT, mT, 1.25 * mT]

    sigma = compute_mass_sigma_limit_si(masses=masses,
                                        detector=detector,
                                        form_factor=PointLikeFormFactor())
    imin = np.nanargmin(sigma)
    assert imin == 1

def test_compute_mass_sigma_limit_si_independent_sigma_ref():
    """
    Check that sigma limit is independent
    wrt the reference cross section
    """
    mass = [1e11]

    sigma_1 = compute_mass_sigma_limit_si(masses=mass,
                                          detector=LZ_1000DAYS,
                                          sigma_reference=1)
    
    sigma_2 = compute_mass_sigma_limit_si(masses=mass,
                                          detector=LZ_1000DAYS,
                                          sigma_reference=2)

    ratio = sigma_2[0] / sigma_1[0]

    assert np.isclose(ratio, 1, atol=1e-12, rtol=1e-12)
