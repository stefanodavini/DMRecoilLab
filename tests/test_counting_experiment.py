import pytest

from src.sensitivity.types import CountingExperiment


def test_negative_counts():
    with pytest.raises(ValueError):
        CountingExperiment(counts=-1)

def test_negative_background():
    with pytest.raises(ValueError):
        CountingExperiment(counts=0, background=-0.1)
