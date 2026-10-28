"""
Classes for propagator responses.

A propagator response is a dimensionless
function of momentum transfer q that
encodes mediator-dependent modifications
to the differential cross section.

Propagators are independent of the target
nuclear structure and therefore complement,
rather than replace, nuclear response
models.

Current implementations include:

- point-like (contact) propagators
- massive mediator propagators

Future extensions may include:

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
    modifies the differential cross section
    through mediator-dependent effects.

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

    Massive mediator rate response:

                         M²
        response = (-------------)²
                    M² + q²

    Where M is the mediator mass and q is the momentum transfer, both in eV.
    The response is applied directly to the
    differential cross section. Implementations
    should therefore include any matrix-element
    squaring required by their convention.
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
        The returned quantity is applied directly
        to the differential cross section. It
        represents the propagator-dependent
        modification, including any matrix-element
        squaring required by the implementation.

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
    pass

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


class MassiveMediatorPropagator(Propagator):
    """
    Massive mediator response normalized to unity at q = 0.

    The response is applied directly to the differential
    cross section and is defined as

                         M²
        response(q) = (-------------)²,
                        M² + q²

    where M is the mediator mass and q is the momentum
    transfer, both in eV.
    """

    def __init__(self, mediator_mass: float):
        if not np.isfinite(mediator_mass) or mediator_mass <= 0:
            raise ValueError("mediator_mass must be a finite positive number.")

        self.mediator_mass = float(mediator_mass)

    def response(self, q: NDArray) -> NDArray:
        """
        Evaluate the massive-mediator cross-section response.

        Parameters
        ----------
        q : ndarray
            Momentum transfer in eV.

        Returns
        -------
        ndarray
            Dimensionless squared propagator response with
            the same shape as q.
        """
        q = np.asarray(q, dtype=float)
        amplitude_ratio = np.square(self.mediator_mass / np.hypot(self.mediator_mass, q))
        return np.asarray(np.square(amplitude_ratio))
