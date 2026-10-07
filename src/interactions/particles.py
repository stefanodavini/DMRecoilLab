"""
Particle definitions used by the Dark Matter Project.

This module contains lightweight immutable
data containers describing the particle
properties required by scattering and
rate calculations.

Currently implemented
---------------------
WIMP
    Weakly Interacting Massive Particle
    characterized by its mass and spin.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class WIMP:
    """
    WIMP particle properties.

    Parameters
    ----------
    mass : float
        WIMP mass in eV.

    spin : float
        WIMP spin J.
    """

    mass: float
    spin: float