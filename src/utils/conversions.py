"""
Unit conversion utilities used throughout the project.

These functions provide conversions between SI quantities and the
natural units:
- velocity (km/s <-> c=1)
- mass (kg -> eV)
- lenght/momentum (fm <-> eV)
- action (eV * fm -> natural units)
"""

from src.utils.constants import C_kmps, HBAR_C_MeV_fm, ELECTRON_CHARGE_coulomb

from numpy.typing import NDArray

def convert_from_kmps_to_beta(v: float | NDArray) -> float | NDArray:
    """
    Convert velocity from km/s to dimensionless beta=v/c.

    Parameters
    ----------
    v : float or ndarray
        Velocity in km/s.

    Returns
    -------
    float or ndarray
        Velocity expressed as beta=v/c.
    """
    return v / C_kmps

def convert_from_beta_to_kmps(beta: float | NDArray) -> float | NDArray:
    """
    Convert velocity from dimensionless beta=v/c to km/s.

    Parameters
    ----------
    beta : float or ndarray
        Velocity expressed as beta=v/c.

    Returns
    -------
    float or ndarray
        Velocity expressed in km/s.
    """
    return beta * C_kmps

def convert_from_kg_to_eV(m: float) -> float:
    """
    Convert mass from kg to electronvolt.

    Parameters
    ----------
    m : float
        Mass in kg.

    Returns
    -------
    float
        Mass in electronvolt.
    """
    c_mps = C_kmps * 1e3
    eV = abs(ELECTRON_CHARGE_coulomb)
    return m * c_mps ** 2 / eV

def convert_from_fm_to_momentum_eV(x: float) -> float:
    """
    Convert lenght in fm to
    momentum in eV.

    The implemented relation is q = hbar * c / x
    """
    return 1e6 * HBAR_C_MeV_fm / x

def convert_from_momentum_eV_to_fm(q: float) -> float:
    """
    Convert momentum in eV to
    lenght in fm.

    The implemented relation is x = hbar * c / q
    """
    return 1e6 * HBAR_C_MeV_fm / q

def convert_qr_to_dimensionless(qr: float | NDArray) -> float | NDArray: 
    """
    Convert an input with size of
    momentum (eV) * length (fm)
    in a dimensionless unit
    with hbar = 1, c = 1 
    """
    return 1e-6 * qr / HBAR_C_MeV_fm

