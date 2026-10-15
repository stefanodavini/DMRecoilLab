import numpy as np
import pytest

from src.halo import StandardHaloModel, TabulatedVelocityDistribution

def test_tabulated_velocity_distribution_round_trip():
    output_file = "tests/output/shm_roundtrip.txt"
    shm = StandardHaloModel()
    shm.to_file(output_file)
    halo = TabulatedVelocityDistribution.from_file(output_file)
    v = np.linspace(0, shm.maximum_speed, 1000)

    assert np.allclose(halo.velocity_pdf(v), shm.velocity_pdf(v))

def test_tabulated_from_file_invalud_filename():
    with pytest.raises(ValueError, match="filename"):
        TabulatedVelocityDistribution.from_file(None)
    with pytest.raises(ValueError, match="filename"):
        TabulatedVelocityDistribution.from_file(filename="")

def test_to_file_invalid_filename():
    halo = StandardHaloModel()
    with pytest.raises(ValueError, match="filename"):
        halo.to_file(None)
    with pytest.raises(ValueError, match="filename"):
        halo.to_file(filename="")