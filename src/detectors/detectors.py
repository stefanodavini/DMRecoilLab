from numpy import isfinite

from src.nuclei.nucleus import Nucleus
from src.utils.conversions import convert_from_kg_to_eV
from src.utils.constants import DAY_TO_SECONDS

from dataclasses import dataclass

@dataclass(frozen=True)
class IdealDetector:
    """
    Ideal detector description.

    Parameters
    ----------
    nucleus
        Target isotope.

    mass_kg
        Active fiducial mass in kg.

    exposure_days
        Exposure time in days.

    threshold_energy
        Analysis threshold in eV.
    
    target_mass_fraction
        Mass fraction of the target isotope in the detector. Default is 1.0 (isotopically pure).

    Notes
    -----
    This class assumes:
    - 100% detection efficiency
    - perfect energy resolution
    - isotopically pure target unless specified

    It is intended for theoretical
    rate calculations and detector
    comparisons.
    """
    nucleus: Nucleus
    mass_kg: float
    exposure_days: float = 365
    threshold_energy: float = 0.0
    target_mass_fraction: float = 1.0

    def __post_init__(self):
        if not isfinite(self.mass_kg) or self.mass_kg < 0:
            raise ValueError("mass_kg must be non negative.")
        if not isfinite(self.target_mass_fraction) or not 0 <= self.target_mass_fraction <= 1:
            raise ValueError("target_mass_fraction must be finite and in the interval [0, 1].")
        if not isfinite(self.exposure_days) or self.exposure_days < 0:
            raise ValueError("exposure_days must be a finite non-negative number.")
        if not isfinite(self.threshold_energy) or self.threshold_energy < 0:
            raise ValueError("threshold_energy must be a finite non-negative number.")

    @property
    def number_of_targets(self) -> float:
        """
        Number of nuclei contained in the detector.

        Returns
        -------
        float

        Notes
        -----
        The result is computed as
        N = Mdet / MT
        where
        - Mdet is detector mass (converted to eV)
        - MT is target-nucleus mass (eV)
        """
        return convert_from_kg_to_eV(self.mass_kg * self.target_mass_fraction) / self.nucleus.mass

    @property
    def exposure_seconds(self) -> float:
        return DAY_TO_SECONDS * self.exposure_days