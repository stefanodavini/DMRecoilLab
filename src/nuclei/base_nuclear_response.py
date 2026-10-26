"""
Base classes for nuclear response functions.

A nuclear response is a dimensionless
function of momentum transfer q that
encodes the finite size and internal
structure of the target nucleus.

The base class defined in this module provides a
common interface for response functions depending
on momentum transfer q.

Current implementations include:
- PointLikeNuclearResponse

Derived modules may implement specific
response families such as:
- spin-independent form factors
- spin-dependent form factors
- EFT nuclear responses
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

class NuclearResponse(ABC):
    """
    Base class for nuclear responses.

    A nuclear response is a function
    of momentum transfer q.
    """

    @abstractmethod
    def response(self, q: NDArray) -> NDArray:
        """
        Evaluate the nuclear response.

        Parameters
        ----------
        q : ndarray
            Momentum transfer in eV.

        Returns
        -------
        ndarray
            Dimensionless response evaluated at
            the supplied momentum transfer.

        Notes
        -----
        The returned quantity is typically
        normalized so that
        response(0) = 1
        although specific response models may
        adopt different conventions.
        """
        pass


class PointLikeNuclearResponse(NuclearResponse):
    """
    Point like nuclear response.

    It can represent one of the physical limits:
    - zero momentum transfer
    - point-like nucleus

    The response equals 1
    for all momentum transfer
    """
    def __init__(self):
        pass

    def response(self, q: NDArray) -> NDArray:
        """
        Point Like nuclear response

        Parameters
        ----------
        q : ndarray
            Momentum transfer in eV.

        Returns
        -------
        ndarray of 1.
        """
        q = np.asarray(q)

        return np.ones_like(q, dtype=float)