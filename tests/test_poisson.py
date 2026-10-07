import numpy as np
import pytest

from src.sensitivity.poisson import PoissonUpperLimit
from src.sensitivity.types import CountingExperiment


def test_poisson_upper_cl_negative():
    with pytest.raises(ValueError):
        PoissonUpperLimit(cl=-0.1)
    with pytest.raises(ValueError):
        PoissonUpperLimit(cl=0.)

def test_poisson_upper_cl_greater_than_one():
    with pytest.raises(ValueError):
        PoissonUpperLimit(cl=1)
    with pytest.raises(ValueError):
        PoissonUpperLimit(cl=1.1)

def test_poisson_upper_limit_90_percent():
    experiment = CountingExperiment(counts=0, background=0)
    limit = PoissonUpperLimit(cl=0.9)
    assert np.isclose(limit.signal_upper_bound(experiment),
                      2.302585092994046)

def test_poisson_upper_limit_95_percent():
    experiment = CountingExperiment(counts=0)
    limit = PoissonUpperLimit(cl=0.95)
    assert np.isclose(limit.signal_upper_bound(experiment),
                      2.995732273553991)

def test_poisson_upper_nonzero_counts_not_implemented():
    experiment = CountingExperiment(counts=1)
    limit = PoissonUpperLimit()
    with pytest.raises(NotImplementedError):
        limit.signal_upper_bound(experiment)

def test_poisson_upper_nonzero_background_not_implemented():
    experiment = CountingExperiment(counts=0, background=0.1)
    limit = PoissonUpperLimit()
    with pytest.raises(NotImplementedError):
        limit.signal_upper_bound(experiment)
