import numpy as np
import pytest

from src.halo import StandardHaloModel


def test_shm_input_v0():
    with pytest.raises(ValueError):
        StandardHaloModel(v0=0)

def test_shm_input_vesc():
    with pytest.raises(ValueError):
        StandardHaloModel(vesc=0)

def test_shm_input_vesc_v0():
    with pytest.raises(ValueError):
        StandardHaloModel(v0=500, vesc=200)

def test_shm_maximum_speed():
    vesc = 500
    shm = StandardHaloModel(v0=200, vesc=vesc)
    assert shm.maximum_speed == vesc

def test_shm_velocity_pdf():
    v = np.linspace(0, 600, 10000)

    shm = StandardHaloModel(v0=200, vesc=500)
    fv = shm.velocity_pdf(v=v)
    assert np.all(fv >= 0)

    norm = np.trapezoid(4* np.pi * v**2*fv, v)
    assert np.isclose(norm, 1.0, rtol=1e-5)

def test_shm_speed_pdf():
    vgrid = np.linspace(0, 600, 10000)

    shm = StandardHaloModel()
    gv = shm.speed_pdf(v=vgrid)
    assert np.all(gv >= 0)

    norm = np.trapezoid(gv, vgrid)
    assert np.isclose(norm, 1.0, rtol=1e-5)

def test_shm_eta_zero():
     # η(0) should equal ∫ g(v)/v dv
    vgrid = np.linspace(0, 800, 1000)
    shm = StandardHaloModel()
    gv = shm.speed_pdf(v=vgrid)

    eta_0 = shm.eta(vmin=vgrid)[0]
    expected = np.trapezoid(gv[1:] / vgrid[1:], vgrid[1:])
    assert np.isclose(eta_0, expected, rtol=1e-3)

def test_shm_eta_decreasing():
    vgrid = np.linspace(0, 600, 10000)
    shm = StandardHaloModel()
    eta = shm.eta(vmin=vgrid)
    assert np.all(np.diff(eta) <= 0)

def test_shm_eta_at_escape_speed():
    vesc = 600
    shm = StandardHaloModel(vesc=vesc)
    assert np.isclose(shm.eta(vmin=vesc), 0.)

def test_shm_eta_huge_vesc():
    v0 = 200
    vesc = 10 * v0
    shm = StandardHaloModel(v0=v0, vesc=vesc)
    expected = 2 / (np.sqrt(np.pi) * v0)
    assert np.isclose(shm.eta(0), expected, rtol=1e-6)

def test_shm_beyond_vesc():
    vesc = 600
    shm = StandardHaloModel(vesc=600)
    vbeyond = np.linspace(vesc, vesc + 300, 10)

    assert np.all(shm.velocity_pdf(vbeyond) == 0.)
    assert np.all(shm.speed_pdf(vbeyond) == 0.)
    assert np.all(shm.eta(vbeyond) == 0.)

def test_boost_zero():
    shm = StandardHaloModel()
    boosted = shm.boost(vboost=0.0)

    v = np.linspace(0, 800, 1000)

    assert np.array_equiv(
        boosted.velocity_pdf(v=v),
        shm.velocity_pdf(v=v))

def test_shm_boost_maximum_speed():
    shm = StandardHaloModel(vesc=500.)
    boosted = shm.boost(vboost=200.)

    assert np.isclose(boosted.maximum_speed, 700., rtol=1e-6)

def test_shm_boost_implementation():
    vgrid = np.linspace(0, 800, 10000)
    shm = StandardHaloModel()
    shm_lab = shm.boost()

    boost_fv = shm_lab.velocity_pdf(v=vgrid)
    assert np.all(boost_fv >= 0)
    norm = np.trapezoid(4* np.pi * vgrid**2 * boost_fv, vgrid)
    assert np.isclose(norm, 1.0, rtol=1e-4)

    boost_gv = shm_lab.speed_pdf(v=vgrid)
    assert np.all(boost_gv >= 0)
    norm = np.trapezoid(boost_gv, vgrid)
    assert np.isclose(norm, 1.0, rtol=1e-4)

    boost_eta = shm_lab.eta(vmin=vgrid)
    assert np.all(boost_eta >= 0)
    assert np.all(np.diff(boost_eta) <= 0)
    assert np.isclose(boost_eta[-1], 0.0, atol=1e-5)

def test_shm_boost_shift_peak():
    vgrid = np.linspace(0, 800, 1600)
    shm = StandardHaloModel()
    shm_earth = shm.boost(vboost=232)

    boost_gv = shm_earth.speed_pdf(v=vgrid)

    assert np.argmax(boost_gv) > np.argmax(shm.speed_pdf(v=vgrid))

def test_shm_boost_eta_zero():
    vgrid = np.linspace(0, 800, 1000)
    shm = StandardHaloModel()
    shm_earth = shm.boost()

    boost_gv = shm_earth.speed_pdf(v=vgrid)

    # η(0) should equal ∫ g(v)/v dv
    eta_0 = shm_earth.eta(vmin=vgrid)[0]
    expected = np.trapezoid(boost_gv[1:] / vgrid[1:], vgrid[1:])

    assert np.isclose(eta_0, expected, rtol=1e-3)

def test_shm_boost_eta_tails():
    # Check a boost populates the high-velocity tails of eta
    
    shm = StandardHaloModel(v0=220, vesc=500)
    shm_lab = shm.boost(vboost=240)

    vtail = np.linspace(200, 800, 600)
    gal_eta_tail = shm.eta(vmin=vtail)
    lab_eta_tail = shm_lab.eta(vmin=vtail)

    assert np.all(lab_eta_tail - gal_eta_tail >= 0)
