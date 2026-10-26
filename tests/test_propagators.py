"""
Tests for propagator models.
"""

import numpy as np
import pytest

from src.interactions.propagators import Propagator, PointLikePropagator

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
