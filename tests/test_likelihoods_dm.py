import numpy as np
import pytest

from src.analysis.likelihoods import DMRecoilExtendedLikelihood
from src.analysis.likelihoods import LikelihoodGridVSMassDelta
from src.analysis.likelihoods import likelihood_grid_si_mass_delta


from src.rates import RateCalculator
from src.halo.shm import StandardHaloModel
from src.utils.factory import create_interaction_si
from src.detectors.reference_detectors import XENON100_2012
from src.interactions.particles import WIMP

def _make_reference_rc():
    return RateCalculator(
        halo=StandardHaloModel().boost(),
        interaction=create_interaction_si(wimp=WIMP(mass=1e11, spin=0),
                                          nucleus = XENON100_2012.nucleus,
                                          delta=0.),
        detector=XENON100_2012)

def test_likelihood_dm_no_ratecalculator():

    with pytest.raises(TypeError, match="RateCalculator"):
        DMRecoilExtendedLikelihood(rate_calculator=XENON100_2012)

def test_likelihood_dm_with_ratecalculator():

    rc = _make_reference_rc()
    llh = DMRecoilExtendedLikelihood(rate_calculator=rc)

    assert np.isfinite(llh.mu)
    assert llh.mu > 0
    assert llh.f is not None
    assert callable(llh.f)

def test_likelihood_dm_one_event_poisson_max():

    rc = _make_reference_rc()
    llh = DMRecoilExtendedLikelihood(rate_calculator=rc)

    logL = llh.logL_poisson_max(N=1)
    assert np.isclose(logL, -1)

def test_likelihood_dm_one_event_with_pdf():

    rc = _make_reference_rc()
    llh = DMRecoilExtendedLikelihood(rate_calculator=rc)
    Ei = [50e3]

    logL_pdf = llh.logL_pdf(x=Ei)
    assert np.isfinite(logL_pdf)
    assert logL_pdf < 0.

    logL_tot = llh.logL_muhat(x=Ei)
    assert np.isfinite(logL_tot)
    assert logL_tot < logL_pdf
    assert np.isclose(logL_tot, logL_pdf - 1.0)

def test_likelihood_grid_si_mass_delta_zero_grid():

    masses = []
    deltas = []

    with pytest.raises(ValueError, match="size"):
        likelihood_grid_si_mass_delta(
            wimp_masses=masses,
            deltas=deltas,
            detector=XENON100_2012,
            energies=[250e3],
    )

def test_likelihood_grid_si_mass_delta():

    masses = [5e11, 1e12]
    deltas = [-1e5, 0., 1e5]

    grid = likelihood_grid_si_mass_delta(
        wimp_masses=masses,
        deltas=deltas,
        detector=XENON100_2012,
        energies=[250e3],
    )

    assert isinstance(grid, LikelihoodGridVSMassDelta)
    assert grid.shape == (2, 3)

    assert len(grid.masses) == 2
    assert len(grid.deltas) == 3

    assert grid.logL.shape == (2, 3)
    assert grid.q.shape == (2, 3)

    assert np.max(grid.logL) == 0.
    assert np.min(grid.q) == 0.
