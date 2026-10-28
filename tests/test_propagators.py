"""
Tests for propagator models.
"""

import numpy as np
import pytest

from src.interactions.propagators import (
    MassiveMediatorPropagator,
    Propagator,
    PointLikePropagator,
)
from src.interactions.particles import WIMP
from src.interactions.si import SIInteraction
from src.interactions.sd import SDInteraction
from src.kinematics.scattering_kinematics import ElasticKinematics
from src.nuclei.isotopes import Xe131

def test_propagator_is_abstract():
    """
    Propagator cannot be instantiated
    because response() is abstract.
    """
    with pytest.raises(TypeError):
        Propagator()

def test_pointlike_propagator_scalar():
    """
    PointLikePropagator returns unity
    for scalar momentum transfer.
    """
    propagator = PointLikePropagator()
    q = np.asarray(1e6)
    response = propagator.response(q)

    assert response == 1.0

def test_pointlike_propagator_vector():
    """
    PointLikePropagator returns unity
    for all momentum transfers.
    """
    propagator = PointLikePropagator()
    q = np.array([0.0, 1e6, 1e8])
    response = propagator.response(q)

    np.testing.assert_allclose(response, np.ones_like(q, dtype=float))

def test_pointlike_propagator_preserves_shape():
    """
    Output shape matches input shape.
    """
    propagator = PointLikePropagator()
    q = np.linspace(0, 1e8, 50)
    response = propagator.response(q)

    assert response.shape == q.shape
    assert response.ndim == q.ndim
    assert len(q) == len(response)


@pytest.mark.parametrize("mediator_mass", [0.0, -1.0, np.nan, np.inf])
def test_massive_mediator_rejects_invalid_mass(mediator_mass):
    """
    MassiveMediatorPropagator rejects non-positive
    and non-finite mediator masses.
    """
    with pytest.raises(ValueError, match="mediator_mass"):
        MassiveMediatorPropagator(mediator_mass=mediator_mass)


def test_massive_mediator_response():
    """
    MassiveMediatorPropagator returns the squared
    response at zero, characteristic, and large q.
    """
    mediator_mass = 100e6
    propagator = MassiveMediatorPropagator(mediator_mass=mediator_mass)
    q = np.array([0.0, mediator_mass, 10 * mediator_mass])

    response = propagator.response(q)

    np.testing.assert_allclose(response, [1.0, 0.25, 1 / 10201])


def test_massive_mediator_response_scalar():
    """
    MassiveMediatorPropagator accepts scalar momentum
    transfer and returns a scalar-shaped result.
    """
    mediator_mass = 100e6
    propagator = MassiveMediatorPropagator(mediator_mass=mediator_mass)

    response = propagator.response(np.asarray(mediator_mass))

    assert response.shape == ()


def test_massive_mediator_response_preserves_shape():
    """
    MassiveMediatorPropagator preserves multidimensional
    input shape.
    """
    propagator = MassiveMediatorPropagator(mediator_mass=100e6)
    q = np.linspace(0.0, 1e8, 50).reshape(5, 10)

    response = propagator.response(q)

    assert response.shape == q.shape


def test_massive_mediator_is_applied_by_si_interaction():
    """
    SIInteraction applies the massive-mediator response
    in its momentum-transfer response.
    """
    wimp = WIMP(mass=1e11, spin=0)
    kinematics = ElasticKinematics(mX=wimp.mass, mT=Xe131.mass)
    propagator = MassiveMediatorPropagator(mediator_mass=100e6)
    interaction = SIInteraction(
        wimp=wimp,
        nucleus=Xe131,
        kinematics=kinematics,
        propagator=propagator,
    )
    recoil_energy = np.array([1e3, 1e4, 1e5])
    q = kinematics.q_from_ER(ER=recoil_energy)

    np.testing.assert_allclose(
        interaction.momentum_transfer_response(ER=recoil_energy),
        propagator.response(q),
    )


def test_massive_mediator_modifies_sd_differential_cross_section():
    """
    SDInteraction scales its differential cross section
    by the massive-mediator response.
    """
    wimp = WIMP(mass=1e11, spin=0)
    kinematics = ElasticKinematics(mX=wimp.mass, mT=Xe131.mass)
    pointlike_interaction = SDInteraction(
        wimp=wimp,
        nucleus=Xe131,
        kinematics=kinematics,
        propagator=PointLikePropagator(),
    )
    massive_interaction = SDInteraction(
        wimp=wimp,
        nucleus=Xe131,
        kinematics=kinematics,
        propagator=MassiveMediatorPropagator(mediator_mass=100e6),
    )
    velocity = np.array([1e-3, 2e-3])
    recoil_energy = np.array([1e3, 1e4, 3e4])
    q = kinematics.q_from_ER(ER=recoil_energy)

    pointlike_dsigma = pointlike_interaction.dsigma_dER(
        v=velocity,
        ER=recoil_energy,
    )
    massive_dsigma = massive_interaction.dsigma_dER(
        v=velocity,
        ER=recoil_energy,
    )

    np.testing.assert_allclose(
        massive_dsigma,
        pointlike_dsigma * massive_interaction.propagator.response(q)[None, :],
    )
