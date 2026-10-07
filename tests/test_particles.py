from src.interactions.particles import WIMP
import numpy as np

def test_wimp_spin():
    # check numerical equivalence of spin 0.5 and 1/2
    chi = WIMP(mass=1e9, spin=0.5)
    assert np.equal(chi.spin, 1/2)