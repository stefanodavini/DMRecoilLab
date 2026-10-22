from src.nuclei.nucleus import Nucleus
from src.nuclei.isotopes import Ar40, XENON, Xe129, Xe131, Ge73, Si29

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
    assert np.isclose(xe.Sp, -0.009, rtol=1e-2)
    assert np.isclose(xe.Sn, -0.227, rtol=1e-2)
    assert np.isclose(xe.mass, 122.7e9, rtol=1e-2)

def test_ge73_properties():
    ge = Ge73
    assert ge.symbol == "Ge73"
    assert ge.A == 73
    assert ge.Z == 32
    assert ge.N == 41
    assert ge.spin == 4.5
    assert np.isclose(ge.Sp, 0.03, rtol=1e-2)
    assert np.isclose(ge.Sn, 0.378, rtol=1e-2)
    assert np.isclose(ge.mass, 68.0e9, rtol=1e-2)

def test_si29_properties():
    si = Si29
    assert si.symbol == "Si29"
    assert si.A == 29
    assert si.Z == 14
    assert si.N == 15
    assert si.spin == 0.5
    assert np.isclose(si.Sp, -0.002, rtol=1e-2)
    assert np.isclose(si.Sn, 0.130, rtol=1e-2)

def test_default_spin_expectation_values():
    ar = Ar40

    assert ar.Sp == 0.0
    assert ar.Sn == 0.0

def test_odd_isotopes_have_nonzero_spin():
    assert Xe129.spin > 0
    assert Xe131.spin > 0
    assert Ge73.spin > 0
    assert Si29.spin > 0

def test_nucleus_is_immutable():
    ar = Ar40
    with pytest.raises(FrozenInstanceError):
        ar.A = 39

def test_xenon_dictionary():
    assert XENON[129] is Xe129
    assert XENON[131] is Xe131