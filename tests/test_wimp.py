from src.interactions.particles import WIMP

import pytest
from dataclasses import FrozenInstanceError

def test_wimp_is_immutable():
    wimp = WIMP(mass=1e12, spin=0)
    with pytest.raises(FrozenInstanceError):
        wimp.mass = 1e11

def test_wimp_default_spin_zero():
    wimp = WIMP(mass=1e11)
    assert wimp.spin == 0
