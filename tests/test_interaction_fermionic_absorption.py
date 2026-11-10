"""Unit and analytic-rate tests for monochromatic NC absorption.

The Dror convention is sigma_abs = beta * sigma_A(beta, q=0).
Consequently beta * integral(d_sigma/d_ER) recovers sigma_abs for
point-like responses. The cross section itself scales as 1/beta.
"""

import numpy as np
import pytest

from src.detectors.detectors import IdealDetector
from src.halo.base_halo_distribution import IsotropicSpeedDistribution
from src.rates import RateCalculator
from src.utils.constants import C_kmps, KM_TO_M, CM_TO_M
from src.interactions.fermionic_absorption import FermionicAbsorptionInteraction
from src.interactions.particles import WIMP
from src.interactions.propagators import MassiveMediatorPropagator, PointLikePropagator
from src.kinematics.absorption_kinematics import AbsorptionKinematics
from src.kinematics.scattering_kinematics import ElasticKinematics, InelasticKinematics
from src.nuclei.base_nuclear_response import NuclearResponse, PointLikeNuclearResponse
from src.nuclei.form_factors import TabulatedFormFactor
from src.nuclei.isotopes import Ar40, Xe131


def make_interaction(nucleus=Ar40, sigma_abs=1e-40, **kwargs):
    wimp = WIMP(mass=40e6, spin=0.5)
    kin = AbsorptionKinematics(mX=wimp.mass, mT=nucleus.mass)
    return FermionicAbsorptionInteraction(
        wimp=wimp, nucleus=nucleus, kinematics=kin,
        sigma_abs=sigma_abs, **kwargs,
    )


@pytest.mark.parametrize("inelastic", [False, True])
def test_fermionic_absorption_scattering_kinematics_raises(inelastic):
    wimp = WIMP(mass=40e6, spin=0.5)
    if inelastic:
        kin = InelasticKinematics(mX=wimp.mass, mT=Ar40.mass, delta=1e3)
    else:
        kin = ElasticKinematics(mX=wimp.mass, mT=Ar40.mass)
    with pytest.raises(ValueError, match="AbsorptionKinematics"):
        FermionicAbsorptionInteraction(wimp=wimp, nucleus=Ar40, kinematics=kin)


@pytest.mark.parametrize("sigma", [-1e-40, np.nan, np.inf, -np.inf])
def test_fermionic_absorption_invalid_sigma_abs_raises(sigma):
    with pytest.raises(ValueError, match="sigma_abs"):
        make_interaction(sigma_abs=sigma)


def test_fermionic_absorption_sigma_reference_equals_sigma_abs():
    interaction = make_interaction(sigma_abs=3e-40)
    assert interaction.sigma_reference == interaction.sigma_abs == 3e-40
    assert isinstance(interaction.nuclear_response, PointLikeNuclearResponse)
    assert isinstance(interaction.propagator, PointLikePropagator)


def test_fermionic_absorption_monochromatic_dsigma_shape_and_speeds():
    interaction = make_interaction()
    v = np.array([1e-4, 1e-3, 2e-3])
    ER = np.linspace(0.0, 2.0 * interaction.kin.ER, 1001)
    dsigma = interaction.dsigma_dER(v=v, ER=ER)
    assert dsigma.shape == (len(v), len(ER))
    assert np.count_nonzero(dsigma[0]) == 1
    assert np.argmax(dsigma[0]) == np.argmin(np.abs(ER - interaction.kin.ER))
    weighted = v[:, None] * dsigma
    np.testing.assert_allclose(
        weighted, np.broadcast_to(weighted[0], weighted.shape), rtol=1e-13, atol=0.0,
    )


@pytest.mark.parametrize("nucleus", [Ar40, Xe131])
@pytest.mark.parametrize("num", [2, 3, 17, 10000])
def test_fermionic_absorption_integrated_sigma_times_beta_recovers_sigma_abs(nucleus, num):
    interaction = make_interaction(nucleus=nucleus)
    v = np.array([1e-4, 1e-3, 2e-3])
    # ER_max defaults to E_abs: this explicitly tests the upper endpoint.
    np.testing.assert_allclose(
        v * interaction.integrate_sigma(v=v, num=num),
        interaction.sigma_abs, rtol=1e-13, atol=0.0,
    )


@pytest.mark.parametrize("relative_grid", [
    [0.0, 0.2, 0.9, 1.4, 2.0],  # Non-uniform; line between grid points.
    [1.0, 1.2, 1.5, 2.0],       # Line at the lower endpoint.
    [0.0, 0.1, 0.7, 1.0],       # Line at the upper endpoint.
])
def test_fermionic_absorption_delta_normalization_on_arbitrary_grids(relative_grid):
    interaction = make_interaction()
    ER = interaction.kin.ER * np.array(relative_grid)
    dsigma = interaction.dsigma_dER(v=np.array([1e-3]), ER=ER)
    np.testing.assert_allclose(
        1e-3 * np.trapezoid(dsigma, ER, axis=1), interaction.sigma_abs,
        rtol=1e-13, atol=0.0,
    )


@pytest.mark.parametrize("relative_grid", [[0.1, 0.5, 0.9], [1.1, 1.5, 2.0]])
def test_fermionic_absorption_line_outside_grid_returns_zero(relative_grid):
    interaction = make_interaction()
    ER = interaction.kin.ER * np.array(relative_grid)
    assert np.all(interaction.dsigma_dER(v=np.array([1e-3]), ER=ER) == 0.0)


def test_fermionic_absorption_nuclear_response_applied():
    # TabulatedFormFactor.response returns F^2; no extra squaring is needed.
    ff = TabulatedFormFactor(
        q=[0.0, 80e6], F=[0.5, 0.5], check_normalization=False,
    )
    interaction = make_interaction(nuclear_response=ff)
    np.testing.assert_allclose(
        1e-3 * interaction.integrate_sigma(v=np.array([1e-3])),
        interaction.sigma_abs / 4.0, rtol=1e-13, atol=0.0,
    )


def test_fermionic_absorption_arbitrary_nuclear_response_supported():
    class ConstantResponse(NuclearResponse):
        def response(self, q):
            return np.full_like(np.asarray(q), 0.3, dtype=float)

    interaction = make_interaction(nuclear_response=ConstantResponse())
    np.testing.assert_allclose(
        1e-3 * interaction.integrate_sigma(v=np.array([1e-3])),
        0.3 * interaction.sigma_abs, rtol=1e-13, atol=0.0,
    )


def test_fermionic_absorption_propagator_applied_at_true_line():
    interaction = make_interaction(
        propagator=MassiveMediatorPropagator(mediator_mass=40e6),
    )
    # q_abs = m_chi = mediator_mass: propagator response = 1/4.
    ER = interaction.kin.ER * np.array([0.0, 0.8, 1.4, 2.0])
    dsigma = interaction.dsigma_dER(v=np.array([1e-3]), ER=ER)
    np.testing.assert_allclose(
        1e-3 * np.trapezoid(dsigma, ER, axis=1), interaction.sigma_abs / 4.0,
        rtol=1e-13, atol=0.0,
    )


def test_fermionic_absorption_scattering_mask_is_not_applied(monkeypatch):
    interaction = make_interaction()
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Scattering masking must not be called.")
    monkeypatch.setattr(interaction, "_apply_mask_to_dsigmadER", fail_if_called)
    ER = interaction.kin.ER * np.array([0.0, 0.8, 1.4, 2.0])
    dsigma = interaction.dsigma_dER(v=np.array([1e-3]), ER=ER)
    assert np.count_nonzero(dsigma) == 1


@pytest.mark.parametrize("v, ER", [
    ([], [0.0, 1.0]), ([1e-3], []), ([], []),
])
def test_fermionic_absorption_empty_inputs_raise(v, ER):
    with pytest.raises(ValueError, match="non-empty"):
        make_interaction().dsigma_dER(v=np.array(v), ER=np.array(ER))


@pytest.mark.parametrize("ER", [
    [0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0, 1.0],
    [0.0, np.nan], [0.0, np.inf], [[0.0, 1.0], [2.0, 3.0]],
])
def test_fermionic_absorption_invalid_energy_grid_raises(ER):
    with pytest.raises(ValueError):
        make_interaction().dsigma_dER(v=np.array([1e-3]), ER=np.array(ER))


@pytest.mark.parametrize("v", [[0.0], [-1e-3], [np.nan], [np.inf], 1e-3, [[1e-3]]])
def test_fermionic_absorption_invalid_speed_grid_raises(v):
    with pytest.raises(ValueError):
        make_interaction().dsigma_dER(v=np.array(v), ER=np.array([0.0, 1.0]))


def test_fermionic_absorption_zero_sigma_abs():
    interaction = make_interaction(sigma_abs=0.0)
    assert np.all(interaction.integrate_sigma(v=np.array([1e-4, 1e-3])) == 0.0)




class TriangularSpeedHalo(IsotropicSpeedDistribution):
    """Normalized toy halo: g(v)=2*v/V^2 on (0,V], <v>=2*V/3."""
    def __init__(self, maximum_speed):
        self._maximum_speed = maximum_speed

    @property
    def maximum_speed(self):
        return self._maximum_speed

    def velocity_pdf(self, v):
        v = np.asarray(v, dtype=float)
        result = np.zeros_like(v)
        mask = (v > 0) & (v <= self.maximum_speed)
        result[mask] = 1.0 / (2.0 * np.pi * self.maximum_speed**2 * v[mask])
        return result


@pytest.mark.parametrize("nucleus", [Ar40, Xe131])
@pytest.mark.parametrize("maximum_speed", [300.0, 600.0])
def test_fermionic_absorption_counts_recover_dror_rate(nucleus, maximum_speed):
    sigma_NC = 2e-48  # Per-nucleon coefficient of Dror Eq. (3.7), in cm^2.
    interaction = make_interaction(nucleus=nucleus, sigma_abs=nucleus.A**2 * sigma_NC)
    detector = IdealDetector(nucleus=nucleus, mass_kg=2.0, exposure_days=3.0)
    rate = RateCalculator(
        halo=TriangularSpeedHalo(maximum_speed), interaction=interaction,
        detector=detector, rho_dm=0.4e9,
    )
    c_cm_s = C_kmps * KM_TO_M / CM_TO_M
    # Dror Eq. (3.7), point-like, per nucleus, with physical units restored.
    expected = (
        rate.wimp_number_density * sigma_NC * nucleus.A**2 * c_cm_s
        * detector.number_of_targets * detector.exposure_seconds
    )
    np.testing.assert_allclose(rate.expected_counts(num=501), expected,
                               rtol=1e-11, atol=0.0)


def test_fermionic_absorption_coefficient_recovers_same_rate_for_different_halos():
    interaction = make_interaction()
    detector = IdealDetector(nucleus=interaction.nucleus, mass_kg=1.0, exposure_days=1.0)
    counts = [RateCalculator(
        halo=TriangularSpeedHalo(vmax), interaction=interaction,
        detector=detector,
    ).expected_counts(num=501) for vmax in [200.0, 800.0]]
    np.testing.assert_allclose(counts[0], counts[1], rtol=1e-11, atol=0.0)

