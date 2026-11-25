import numpy as np
import pytest

from src.interactions.particles import WIMP
from src.nuclei.nucleus import Nucleus
from src.nuclei.isotopes import Xe131, Ar40
from src.nuclei.form_factors import PointLikeFormFactor
from src.kinematics.scattering_kinematics import ElasticKinematics
from src.kinematics.relativistic_kinematics import (
    RelativisticKinematics,
    RelativisticElasticKinematics)
from src.interactions.relativistic_interactions import (
    RelativisticScalarScalarInteraction,
    RelativisticPseudoscalarScalarInteraction)
from src.utils.factory import create_interaction_si

def _make_sigma_ss(wimp: WIMP,
                   nucleus: Nucleus,
                   sigma_p: float = 1e-40,
                   fn_over_fp: float = 1.
                   ) -> RelativisticScalarScalarInteraction:
    mX = wimp.mass
    mT = nucleus.mass

    kin = RelativisticKinematics(mX=mX, mT=mT, m2=mX)
    return RelativisticScalarScalarInteraction(
        wimp=wimp,
        nucleus=nucleus,
        kinematics=kin,
        sigma_p=sigma_p,
        fn_over_fp=fn_over_fp,
    )

def _make_sigma_ps(wimp: WIMP,
                   nucleus: Nucleus,
                   sigma_p: float = 1e-40
                   ) -> RelativisticScalarScalarInteraction:
    mX = wimp.mass
    mT = nucleus.mass

    kin = RelativisticElasticKinematics(mX=mX, mT=mT)
    return RelativisticPseudoscalarScalarInteraction(
        wimp=wimp,
        nucleus=nucleus,
        kinematics=kin,
        sigma_p=sigma_p
    )

def test_inteaction_relativistic_reject_non_relativistic_kinematics():
    wimp = WIMP(mass=1e6)
    target = Xe131
    with pytest.raises(TypeError, match="RelativisticKinematics"):
        RelativisticScalarScalarInteraction(
            wimp=wimp,
            nucleus=target,
            kinematics=ElasticKinematics(mX=wimp.mass,
                                         mT=target.mass),
            sigma_p=1e-40
        )

@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_reject_invalid_fn_over_fp(value):
    with pytest.raises(ValueError, match="fn_over_fp"):
        _make_sigma_ss(wimp=WIMP(mass=1e9),
                       nucleus=Xe131,
                       fn_over_fp=value)

def test_interaction_relativistic_dm_scalar_lambda_round_trip_from_sigma():
    sigma0 = 1e-40

    model = _make_sigma_ss(wimp=WIMP(mass=1e9), nucleus=Ar40,
                           sigma_p=sigma0)

    LambdaA = model.Lambda
    model.set_Lambda(Lambda=LambdaA)

    ratio = model.sigma_proton / sigma0

    assert np.isclose(ratio, 1., rtol=1e-14, atol=1e-14)

def test_interaction_relativistic_dm_scalar_lambda_round_trip_from_lambda():
    
    model = _make_sigma_ss(wimp=WIMP(mass=1e9), nucleus=Xe131)

    LambdaA0 = 1e12
    model.set_Lambda(Lambda=LambdaA0)

    ratio = model.Lambda / LambdaA0

    assert np.isclose(ratio, 1., rtol=1e-14, atol=1e-14)

def test_interaction_relativistic_dm_scalar_invalid_lambda_raises():
    model = _make_sigma_ss(wimp=WIMP(mass=1e9), nucleus=Xe131)

    Lambda = [np.inf, -np.inf, np.nan, 0, -1]
    for lam in Lambda:
        with pytest.raises(ValueError,
                           match="Lambda must be finite and positive"):
            model.set_Lambda(Lambda=lam)

def test_interaction_relativistic_dm_scalar_zero_sigma_gives_infinite_lambda():
    model = _make_sigma_ss(wimp=WIMP(mass=1e9),nucleus=Xe131,
                           sigma_p=0.0)

    assert np.isinf(model.Lambda)

def test_interaction_relativistic_dm_scalar_sigma_nucleus_definition():
    sigma_p = 1e-42

    model = _make_sigma_ss(wimp=WIMP(mass=5e9), nucleus=Xe131,
                           sigma_p=sigma_p)

    expected = ( sigma_p *
                (model._reduced_mass_nucleus / model._reduced_mass_proton)**2
                * model.coherent_coupling**2)

    ratio = model.sigma_nucleus / expected

    assert np.isclose(ratio, 1. ,rtol=1e-14, atol=1e-14)

def test_relativistic_interactions_lambda_matches_sigma_nucleus_definition():
    model = _make_sigma_ss(wimp=WIMP(mass=5e9), nucleus=Xe131)

    hbarc_ev_cm = 1.973269804e-5

    expected = (2 * model._reduced_mass_nucleus**2
                * hbarc_ev_cm**2
                / (np.pi * model.sigma_nucleus))**0.25

    ratio = model.Lambda / expected
    assert np.isclose(ratio, 1, rtol=1e-14, atol=1e-14)

def test_interaction_relativistic_ss_coherent_coupling_is_A():
    model = _make_sigma_ss(wimp=WIMP(mass=1e9), nucleus=Xe131)

    assert model.coherent_coupling == Xe131.A

def test_interaction_relativistic_ps_momentum_factor_zero_recoil():
    model = _make_sigma_ps(wimp=WIMP(mass=1e9), nucleus=Xe131)

    factor = model._dm_momentum_factor(np.array([0.0]))
    assert factor[0] == 0.0

def test_interaction_relativistic_ps_momentum_factor_linear_in_ER():

    model = _make_sigma_ps(wimp=WIMP(mass=1e9), nucleus=Xe131)

    f1 = model._dm_momentum_factor(np.array([1e3]))[0]
    f2 = model._dm_momentum_factor(np.array([2e3]))[0]

    assert np.isclose(f2 / f1, 2.0, rtol=1e-5)

def test_interaction_relativistic_ps_momentum_factor_matches_definition():

    model = _make_sigma_ps(wimp=WIMP(mass=1e9), nucleus=Xe131)

    ER = np.array([1e4])
    expected = (model.kinematics.Q2_from_ER(ER) / (4 * model.mass_wimp**2))

    actual = model._dm_momentum_factor(ER=ER)
    ratio = actual / expected

    assert np.allclose(ratio, 1, rtol=1e-5)

def test_relativistic_interactions_ss_and_ps_momentum_factors_differ_at_zero_recoil():
    wimp = WIMP(mass=1e9)
    nucleus = Xe131

    ss = _make_sigma_ss(wimp=wimp, nucleus=nucleus)
    ps = _make_sigma_ps(wimp=wimp, nucleus=nucleus)

    ER = np.array([0.0])

    assert ss._dm_momentum_factor(ER=ER)[0] == 1.0
    assert ps._dm_momentum_factor(ER=ER)[0] == 0.0

def test_relativistic_ss_dsigma_zero_above_recoil_endpoint():
    model = _make_sigma_ss(wimp=WIMP(mass=1e9), nucleus=Xe131)

    v = np.array([0.2])
    ER_max = model.kinematics.ER_max(v=v)[0]

    dsigma = model.dsigma_dER(v=v, ER=np.array([1.1 * ER_max]))
    assert dsigma[0, 0] == 0.0


def test_relativistic_ss_dsigma_shape():
    model = _make_sigma_ss(wimp=WIMP(mass=1e9),nucleus=Xe131)

    v = np.array([0.1, 0.2, 0.3])
    ER = np.array([1e3, 2e3])

    dsigma = model.dsigma_dER(v=v, ER=ER)
    assert dsigma.shape == (3, 2)

def test_relativistic_ss_dsigma_scales_with_sigma_p():
    model1 = _make_sigma_ss(wimp=WIMP(mass=1e9), nucleus=Xe131,
                            sigma_p=1e-40)

    model2 = _make_sigma_ss(wimp=WIMP(mass=1e9), nucleus=Xe131,
                            sigma_p=3e-40)

    v = np.array([0.2])
    ER = np.array([1e5])

    ds1 = model1.dsigma_dER(v=v, ER=ER)[0, 0]
    ds2 = model2.dsigma_dER(v=v, ER=ER)[0, 0]

    assert np.isclose(ds2 / ds1, 3.0)

def test_relativistic_ss_dsigma_scales_with_coherent_coupling():
    wimp = WIMP(mass=1e9)

    model1 = _make_sigma_ss(wimp=wimp, nucleus=Xe131,
                            fn_over_fp=1.0)

    model2 = _make_sigma_ss(wimp=wimp, nucleus=Xe131,
                            fn_over_fp=0.5)

    v = np.array([0.2])
    ER = np.array([1e5])

    ds1 = model1.dsigma_dER(v=v, ER=ER)[0, 0]
    ds2 = model2.dsigma_dER(v=v, ER=ER)[0, 0]

    expected = (model2.coherent_coupling / model1.coherent_coupling)**2

    assert np.isclose(ds2 / ds1, expected, rtol=1e-4)

def test_relativistic_ss_matches_nr_si_at_low_speed():
    wimp = WIMP(mass=50e9)
    nucleus = Xe131
    sigma_p = 1e-40

    int_rel = _make_sigma_ss(wimp=wimp, nucleus=nucleus,
                             sigma_p=sigma_p)

    int_nr = create_interaction_si(wimp=wimp,
                                   nucleus=nucleus,
                                   sigma_p=sigma_p,
                                   form_factor=PointLikeFormFactor())

    v = np.array([1e-3])
    ER = np.array([1e4])

    ds_rel = int_rel.dsigma_dER(v=v, ER=ER)[0, 0]
    ds_nr = int_nr.dsigma_dER(v=v, ER=ER)[0, 0]
    ratio = ds_rel / ds_nr

    assert np.isclose(ratio, 1, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("v0", [1e-3, 1e-1, 0.5, 0.9, 0.99])
def test_relativistic_ss_reproduces_exact_low_velocity_factor(v0):
    wimp = WIMP(mass=50e9)
    nucleus = Xe131
    sigma_p = 1e-40

    int_rel = _make_sigma_ss(wimp=wimp, nucleus=nucleus,
                             sigma_p=sigma_p)

    int_nr = create_interaction_si(wimp=wimp,
                                   nucleus=nucleus,
                                   sigma_p=sigma_p,
                                   form_factor=PointLikeFormFactor())

    ER = np.array([1e4])
    ds_rel = int_rel.dsigma_dER(v=np.array([v0]), ER=ER)[0, 0]
    ds_nr = int_nr.dsigma_dER(v=np.array([v0]), ER=ER)[0, 0]

    expected = 1.0 - v0**2

    assert np.isclose(ds_rel / ds_nr, expected, rtol=1e-8)

def test_relativistic_ps_to_ss_ratio_matches_q2_factor():

    wimp = WIMP(mass=50e9)
    nucleus = Xe131
    sigma_p = 1e-40

    ss = _make_sigma_ss(wimp=wimp, nucleus=nucleus,
                        sigma_p=sigma_p)
    
    ps = _make_sigma_ps(wimp=wimp, nucleus=nucleus,
                        sigma_p=sigma_p)

    v = np.array([0.2])
    ER = np.array([1e5])

    ds_ss = ss.dsigma_dER(v=v, ER=ER)[0, 0]
    ds_ps = ps.dsigma_dER(v=v, ER=ER)[0, 0]

    ratio = ds_ps / ds_ss,

    expected = (ss.kinematics.Q2_from_ER(ER)[0]
                / (4 * wimp.mass**2))

    assert np.isclose(ratio, expected,
                      rtol=1e-12, atol=1e-12)
