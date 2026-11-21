import pytest
import numpy as np

from src.halo.mono_speed_distribution import MonoSpeedDistribution


@pytest.mark.parametrize(
    "vc",
    [-1.0, 0.0, np.inf, -np.inf, np.nan],
)
def test_halo_mono_speed_invalid_vc(vc):
    with pytest.raises(ValueError, match="vc"):
        MonoSpeedDistribution(vc=vc, full_width=1.0)

@pytest.mark.parametrize(
    "width",
    [-1.0, 0.0, np.inf, -np.inf, np.nan],
)
def test_halo_mono_speed_invalid_width(width):
    with pytest.raises(ValueError, match="full_width"):
        MonoSpeedDistribution(vc=100.0, full_width=width)

def test_halo_mono_speed_width_cannot_extend_below_zero():
    with pytest.raises(ValueError, match="extend"):
        MonoSpeedDistribution(vc=1.0, full_width=4.0)

def test_halo_mono_speed_integration_nodes():
    halo = MonoSpeedDistribution(
        vc=200.0,full_width=20.0)

    np.testing.assert_allclose(
        halo.integration_grid(),
        [190.0, 200.0, 210.0],
    )

def test_halo_mono_speed_maximum_speed():
    halo = MonoSpeedDistribution(
        vc=200.0, full_width=20.0)

    assert np.isclose(halo.maximum_speed, 210.0, rtol=1e-4)

def test_halo_mono_speed_normalization():
    halo = MonoSpeedDistribution(
        vc=200.0, full_width=20.0)

    vgrid = halo.integration_grid()
    norm = halo.normalization(vgrid)

    assert np.isclose(norm, 1.0, rtol=1e-6, atol=1e-6)

def test_halo_mono_speed_pdf_vanishes_outside_support():
    halo = MonoSpeedDistribution(
        vc=200.0, full_width=20.0)

    assert halo.speed_pdf(189.0) == 0.0
    assert halo.speed_pdf(211.0) == 0.0

def test_halo_mono_speed_peak_at_central_velocity():
    halo = MonoSpeedDistribution(
        vc=200.0, full_width=20.0)

    assert (halo.speed_pdf(200.) > halo.speed_pdf(195.))
    assert (halo.speed_pdf(200.) > halo.speed_pdf(205.))

def test_halo_mono_speed_delta_property():
    """
    Check that approximately
    integral h(v) delta(v - vc) = h(vc) 
    """
    vc = 200.0
    halo = MonoSpeedDistribution(vc=vc, full_width=1.0)
    v = halo.integration_grid()

    def h(x):
        return (
            3.0 - 0.2 * x
            + 1.5e-2 * x**2 - 2.0e-3 * x**3
            )

    integrand = h(v) * halo.speed_pdf(v)

    result = np.trapezoid(integrand, v)

    assert np.isclose(result, h(vc),
                      rtol=1e-4, atol=1e-8)
