from src.nuclei.isotopes import Xe131, Na23
from src.nuclei.form_factors import HelmFormFactor, PointLikeFormFactor
from src.utils.conversions import convert_from_fm_to_momentum_eV

import numpy as np

def _initialize_helm_xenon():
    return HelmFormFactor(nucleus=Xe131)

def test_form_factor_helm_R1():
    """
    Check that R1 is close to the
    approximated value 1.14 A^(1/3) fm
    """
    nucleus = Xe131
    helm = HelmFormFactor(nucleus=nucleus)
        
    r_approx = 1.14 * (nucleus.A)**(1/3)
    assert np.isclose(r_approx, helm.R1, rtol=0.05)

def test_form_factor_helm_q_zero():
    """
    Check behavior at zero momentum transfer
    """
    helm = _initialize_helm_xenon()
    q = np.array([0.], dtype=float)

    F = helm.F(q)
    assert F[0] == 1

    F2 = helm.response(q)
    assert F2[0] == 1

def test_form_factor_helm_finite():
    helm = _initialize_helm_xenon()
    q = np.logspace(3, 8)
    q[0] = 0.
    F = helm.F(q)

    assert np.all(np.isfinite(F))

def test_response_helm_non_negative():
    helm = _initialize_helm_xenon()
    q = np.logspace(3, 8)
    F2 = helm.response(q)

    assert np.all(np.isfinite(F2))
    assert np.all(F2 >= 0)

def test_response_helm_bounded():
    helm = _initialize_helm_xenon()
    q = np.logspace(1, 8)
    absF = abs(helm.F(q))
    assert np.all(absF <= 1)

def test_helm_large_q_suppression():
    helm = _initialize_helm_xenon()
    q = np.array([1e3, 1e8])
    F2 = helm.response(q)

    assert F2[1] < F2[0]

def test_helm_first_dip_position():
    """
    Check that the first Helm dip
    is at the first zero of j1
    """
    helm = HelmFormFactor(Xe131)
    # q grid with first minumum only
    q = np.linspace(0, 250e6, 10000)  # eV

    F = helm.F(q)
    q_dip = q[np.argmin(np.abs(F))]

    # expected first Helm zero
    x_zero = 4.493 
    q_expected = x_zero * convert_from_fm_to_momentum_eV(helm.R1_fm) 
    assert np.isclose(q_dip, q_expected, rtol=0.1)

def test_helm_xe131_dip_literature_value():
    """
    Check that the first Helm dip in Xe131
    is at about q = 151 MeV 
    """
    helm = HelmFormFactor(Xe131)
    # q grid with first minumum only
    q = np.linspace(0, 250e6, 1000)  # eV

    F = helm.F(q)
    q_dip = q[np.argmin(np.abs(F))]
    assert np.isclose(q_dip, 1.5e8, rtol=0.1)

def test_helm_na23_ER1MeV_literature_value():
    """
    Check that the value of F^2
    at ER = 1 MeV is close to the one
    in Fig 8 of Lewin and Smith
    """
    helm = HelmFormFactor(Na23)
    ER =  1e6 # ev
    q = np.sqrt(2 * Na23.mass * ER)

    F2 = helm.response(q)
    expected = 2e-2

    assert np.isclose(F2, expected, rtol=0.2)

def test_helm_na23_2ndmax_literature_value():
    """
    Check that the 2nd maximum for Na23
    is close to its position
    in Fig 6 of Lewin and Smith
    """
    helm = HelmFormFactor(Na23)
    # q grid for second oscillation
    q = np.linspace(2.7e8, 4.5e8, 100) # ev

    F2 = helm.response(q)
    F2max = max(F2)
    expected = 6e-4

    assert np.isclose(F2max, expected, rtol=0.15)


def test_point_like_form_factor():
    ptff = PointLikeFormFactor()
    q = np.logspace(3, 8)

    assert np.array_equiv(ptff.F(q),
                          np.ones_like(q))

    assert np.array_equiv(ptff.response(q),
                          np.ones_like(q))