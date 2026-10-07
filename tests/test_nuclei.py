from src.nuclei.nucleus import Nucleus
from src.nuclei.isotopes import Ar40, XENON, Xe129, Xe131

import pytest
import numpy as np
from dataclasses import FrozenInstanceError

def test_neutron_number():
    xe = Nucleus(
        symbol="Xe129",
        A=129,
        Z=54,
        mass=1.2e11,
        spin=0.5,
    )

    assert xe.N == 75

def test_ar40_properties():
    ar = Ar40
    assert ar.symbol == "Ar40"
    assert ar.A == 40
    assert ar.Z == 18
    assert ar.N == 22
    assert ar.spin == 0
    assert np.isclose(ar.mass, 37.5e9, rtol=1e-2)


def test_xe131_properties():
    xe = Xe131
    assert xe.symbol == "Xe131"
    assert xe.A == 131
    assert xe.Z == 54
    assert xe.N == 77
    assert xe.spin == 1.5
    assert np.isclose(xe.mass, 122.7e9, rtol=1e-2)

def test_nucleus_is_immutable():
    ar = Ar40
    with pytest.raises(FrozenInstanceError):
        ar.A = 39

def test_xenon_dictionary():
    assert XENON[129] is Xe129
    assert XENON[131] is Xe131