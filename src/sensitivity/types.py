"""
Data structures used by statistical sensitivity calculations.

This module defines lightweight containers describing counting
experiments and statistical results. These objects are intended
to be independent of any specific dark-matter interaction model
and may be reused by different statistical approaches.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CountingExperiment:
    """
    Description of a counting experiment.

    Parameters
    ----------
    counts : int
        Number of observed events.

    background : float
        Expected number of background events.
    """
    counts: int
    background: float = 0.0

    def __post_init__(self):
        if self.counts < 0:
            raise ValueError("counts must be non-negative.")

        if self.background < 0:
            raise ValueError("background must be non-negative.")

@dataclass(frozen=True)
class MassSigmaSensitivityPoint:
    """
    Sensitivity result at a single WIMP mass.

    Parameters
    ----------
    mass: float
        WIMP mass in eV.
    sigma_p_upper : float
        Upper limit on the WIMP-proton cross section.
    cl : float
        Confidence Level
    """
    mass_wimp: float
    sigma_p_upper: float
    cl: float
