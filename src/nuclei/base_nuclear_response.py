"""
Base classes for nuclear response functions.

A nuclear response is a dimensionless
function of momentum transfer q that
encodes the finite size and internal
structure of the target nucleus.

The base class defined in this module provides a
common interface for response functions depending
on momentum transfer q.
 
Current implementations include scalar nuclear
form factors, while future extensions may include
spin-dependent and effective-field-theory nuclear
responses.
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray

class NuclearResponse(ABC):
    """
    Base class for nuclear responses.

    A nuclear response is a function
    of momentum transfer q.
    """

    @abstractmethod
    def response(self, q: NDArray) -> NDArray:
        """
        Evaluate the response function.
        """
        pass
