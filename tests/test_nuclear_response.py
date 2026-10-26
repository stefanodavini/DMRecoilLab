"""
Tests for generic nuclear response models.
"""

import numpy as np
import pytest

from src.nuclei.base_nuclear_response import NuclearResponse, PointLikeNuclearResponse

def test_nuclear_response_is_abstract():
    """
    NuclearResponse cannot be instantiated
    because response() is abstract.
    """
    with pytest.raises(TypeError):
        NuclearResponse()

def test_pointlike_nuclear_response_scalar():
    """
    PointLikeNuclearResponse returns unity
    for scalar momentum transfer.
    """
    nr = PointLikeNuclearResponse()
    q = np.asarray(1e6)
    response = nr.response(q)

    assert response == 1.0

def test_pointlike_nuclear_response_vector():
    """
    PointLikeNuclearResponse returns unity
    for all momentum transfers.
    """
    nr = PointLikeNuclearResponse()
    q = np.array([0.0, 1e6, 1e8])
    response = nr.response(q)

    np.testing.assert_allclose(response, np.ones_like(q, dtype=float))

def test_pointlike_nuclear_response_preserves_shape():
    """
    Output shape matches input shape.
    """
    nr = PointLikeNuclearResponse()
    q = np.linspace(0, 1e8, 50)
    response = nr.response(q)

    assert response.shape == q.shape
    assert response.ndim == q.ndim
    assert len(q) == len(response)
