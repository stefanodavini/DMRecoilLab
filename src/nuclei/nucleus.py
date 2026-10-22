"""
Nuclear data structures used throughout the project.

This module defines the immutable `Nucleus`
dataclass, which stores the basic nuclear
properties required by scattering kinematics,
form factors, cross sections, and event-rate
calculations.

Units
-----
Masses are expressed as energies (eV, c=1).

Notes
-----
Only a minimal set of nuclear properties is
currently included:

- mass number A
- atomic number Z
- nuclear mass
- nuclear spin
- expectation value of proton spin
- expectation value of neutron spin

Additional quantities relevant for future
interaction models (e.g. spin-dependent
couplings) may be added in later versions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Nucleus:
    """
    Nuclear target properties.

    Parameters
    ----------
    symbol
        Human-readable isotope label.
    A : int
        Mass number.
    Z : int
        Atomic number.
    mass : float
        Nuclear mass in eV (c=1).
    spin : float
        Nuclear spin J.
    Sp : float
        Expectation value of proton spin.
    Sn : float
        Expectation value of neutron spin.
    """
    symbol: str
    A: int
    Z: int

    mass: float
    spin: float

    Sp: float = 0.
    Sn: float = 0.

    @property
    def N(self) -> int:
        """
        Number of neutrons

        Returns
        -------
        int
            N = A - Z.
        """
        return self.A - self.Z

