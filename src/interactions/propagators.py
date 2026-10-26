"""
Classes for propagator responses.

A propagator response is a dimensionless
function of momentum transfer q that
encodes mediator-dependent modifications
to the scattering amplitude.

Propagators are independent of the target
nuclear structure and therefore complement,
rather than replace, nuclear response
models.

Current implementations include:

- point-like (contact) propagators

Future extensions may include:

- massive mediator propagators
- light mediator propagators
- dark-photon propagators
- generic momentum-dependent interactions
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

class Propagator(ABC):
    """
    Abstract base class for propagator
    responses.

    A propagator response is a dimensionless
    function of momentum transfer q that
    modifies the scattering rate through
    mediator-dependent effects.

    Derived classes implement a specific
    momentum-transfer dependence while
    preserving a common interface used by
    interaction models.

    Notes
    -----
    The propagator response is typically
    normalized to unity in the contact
    interaction limit.

    Examples
    --------
    Contact interaction:

        response(q) = 1

    Massive mediator:

                    M²
        response = -------
                  M² + q²

    or its squared form, depending on the
    convention adopted by the interaction
    model.
    """

    @abstractmethod
    def response(self, q: NDArray) -> NDArray:
        """
        Evaluate the propagator response.

        Parameters
        ----------
        q : ndarray
            Momentum transfer in eV.

        Returns
        -------
        ndarray
            Dimensionless propagator response
            evaluated at the supplied momentum
            transfer.

        Notes
        -----
        The returned quantity represents the
        propagator-dependent modification of
        the scattering interaction.

        The exact normalization convention is
        implementation dependent but should be
        documented by derived classes.
        """

class PointLikePropagator(Propagator):
    """
    Point-like propagator response.

    This propagator corresponds to the
    contact-interaction limit, where the
    mediator mass is much larger than the
    momentum transfer involved in the
    scattering process.

    The response is independent of
    momentum transfer and equals unity
    for all q:
    
        response(q) = 1

    This class is intended as the default
    propagator for standard SI and SD
    contact interactions.
    """

    def response(self, q: NDArray) -> NDArray:
        """
        Evaluate the point-like propagator
        response.

        Parameters
        ----------
        q : ndarray
            Momentum transfer in eV.

        Returns
        -------
        ndarray
            Array of ones with the same shape
            as q.

        Notes
        -----
        The point-like propagator introduces
        no momentum-transfer dependence and
        therefore leaves the interaction
        unchanged.
        """
        q = np.asarray(q)

        return np.ones_like(q, dtype=float)
